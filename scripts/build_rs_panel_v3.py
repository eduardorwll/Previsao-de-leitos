"""Extrator BigQuery v3 (RS) e construtor canônico de `rs_painel_semanal_regioes_v3.parquet`.

Executa de forma segura e cacheada:
1. Extrai (se não estiver em cache em `data/raw/`) as 3 tabelas SIH v3:
   - `sih_censo_ativo_semanal` (sem double-count `+1`, folga `±1` ano)
   - `sih_w_ref_externa_treino` (`origem != destino`, pré-LGPD, `<= 2022`)
   - `sih_delay_triangle` (triângulo de atraso `2020-2026` para `ChainLadderNowcaster`)
2. Reconstrói `data/processed/rs_painel_semanal_regioes_v3.parquet` com:
   - Censo ativo semanal SIH-SUS
   - Capacidade mensal CNES causal (`join_asof backward`)
   - Exposição MapBiomas ponderada por população (`mun_weights`, anti-MAUP)
   - Propagação multi-salto (`multi_hop=True`) e `W_ref` externa causal

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import date
from pathlib import Path

import polars as pl

from previsao_leitos.domain import CodigoUF
from previsao_leitos.geo import (
    NasaPowerHydroAdapter,
    RegionCentroid,
    aggregate_daily_to_iso_weekly,
    build_complete_daily_grid,
    build_rs_v3_panel_from_frames,
    compute_calendar_hydro_features,
)
from previsao_leitos.ingestion import (
    BigQueryAdapter,
    IngestionConfig,
    extract_rs_v3_sih_pack,
)

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")


def _load_dotenv_if_present(env_path: Path = Path(".env")) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


def _latest(pattern: str) -> Path:
    matches = sorted(DATA_RAW.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em {DATA_RAW} para {pattern!r}")
    return matches[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extrai SIH v3 e constrói painel semanal RS v3.")
    parser.add_argument(
        "--max-bq-mb",
        type=int,
        default=18000,
        help="Teto por consulta BigQuery em MB (padrão 18000 MB para censo ativo RS 2009-2026).",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Usa snapshots v3 já presentes em data/raw/ sem consultar BigQuery.",
    )
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    _load_dotenv_if_present()

    if not args.skip_extract:
        project_id = os.environ.get("GCP_PROJECT_ID")
        if not project_id:
            raise RuntimeError("GCP_PROJECT_ID não definido no ambiente ou em .env.")
        adapter = BigQueryAdapter(project_id=project_id)
        cols = adapter.inspect_columns("basedosdados.br_ms_sih.aihs_reduzidas")
        print(f"[OK] inspect_columns(aihs_reduzidas): {len(cols)} colunas verificadas (0 bytes)")

        cfg = IngestionConfig(
            uf=CodigoUF("43"),
            ano_inicio=2010,
            ano_fim=2025,
            data_dir=Path("data"),
            max_mb_scanned=args.max_bq_mb,
            min_cell_count=5,
            suppress_demand_microstrata=False,
        )
        results = extract_rs_v3_sih_pack(
            source=adapter,
            config=cfg,
            train_end_year=2022,
            delay_start_year=2020,
            delay_end_year=2026,
            max_delay_weeks=12,
        )
        for name, res in results.items():
            print(
                f"[OK] {name}: {res.dataframe.height:,} linhas | "
                f"cache_hit={res.manifest.cache_hit} | arquivo={res.manifest.parquet_path.name}"
            )

    centroids = pl.read_parquet(DATA_RAW / "rs_regioes_saude_centroides_v1.parquet")
    centroid_objs = tuple(
        RegionCentroid(
            entity_id=str(r["id_regiao_saude"]),
            latitude=float(r["latitude"]),
            longitude=float(r["longitude"]),
        )
        for r in centroids.iter_rows(named=True)
    )
    hydro_adapter = NasaPowerHydroAdapter(cache_dir=DATA_RAW, max_workers=4)
    daily_hydro = hydro_adapter.fetch_daily_hydro(
        centroids=centroid_objs,
        start_date=date(2010, 1, 1),
        end_date=date(2025, 12, 31),
    )
    daily_grid = build_complete_daily_grid(
        daily_hydro,
        entity_col="entity_id",
        date_col="data",
        value_cols=("precip_dia_mm", "temp_media_c", "umidade_solo_m3m3", "vento_sul_ms"),
    )
    daily_feats = compute_calendar_hydro_features(
        daily_grid,
        entity_col="entity_id",
        date_col="data",
        precip_col="precip_dia_mm",
        k_decay=0.88,
    )
    weekly_hydro = (
        aggregate_daily_to_iso_weekly(
            daily_feats,
            entity_col="entity_id",
            date_col="data",
            precip_col="precip_dia_mm",
        )
        .rename({"entity_id": "id_regiao_saude"})
        .join(
            centroids.select(["id_regiao_saude", "nome_regiao_saude"]),
            on="id_regiao_saude",
            how="left",
        )
    )

    sih_census = pl.read_parquet(_latest("sih_censo_ativo_semanal_v2_*.parquet"))
    w_ref_ext = pl.read_parquet(_latest("sih_w_ref_externa_treino_v2_*.parquet"))
    cnes_hosp = pl.read_parquet(_latest("cnes_hospitais_escola_v2_*.parquet"))
    cnes_beds = pl.read_parquet(_latest("cnes_leitos_especialidade_v2_*.parquet"))
    ibge = pl.read_parquet(_latest("ibge_municipios_rs_v1_*.parquet"))
    mapbiomas = pl.read_parquet(_latest("mapbiomas_cobertura_rs_1985_2024_v1.parquet"))

    panel_v3 = build_rs_v3_panel_from_frames(
        weekly_hydro=weekly_hydro,
        sih_census=sih_census,
        cnes_hospitals=cnes_hosp,
        cnes_beds=cnes_beds,
        ibge=ibge,
        mapbiomas=mapbiomas,
        w_ref_flows=w_ref_ext,
        train_end_year=2022,
        min_year=2010,
        max_year=2025,
    )

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "rs_painel_semanal_regioes_v3.parquet"
    tmp_path = out_path.with_suffix(".parquet.tmp")
    panel_v3.write_parquet(tmp_path)
    tmp_path.replace(out_path)
    dt = time.perf_counter() - t0
    print(
        f"[OK] Painel semanal RS v3 construído em {dt:.2f}s: "
        f"{panel_v3.height:,} linhas ({panel_v3['id_regiao_saude'].n_unique()} regiões, "
        f"{len(panel_v3.columns)} colunas) -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
