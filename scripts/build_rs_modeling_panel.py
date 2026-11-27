"""Monta o painel de modelagem do RS (v3 + OOF F03.1 + ENSO + capacidade CNES).

Grava `data/processed/rs_modeling_panel_v3.parquet` com **apenas anos <= 2024**,
deixando o holdout (`2025`) fisicamente fora do arquivo. O benchmark, a CLI e os
relatórios consomem esse artefato, evitando rede e re-treino de hidrologia.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

from previsao_leitos.domain import DataContractError, SourceUnavailableError
from previsao_leitos.geo import (
    build_causal_bom_rnino_weekly,
    build_causal_enso_weekly_features,
    build_causal_roni_weekly_features,
    build_causal_soi_weekly_features,
)
from previsao_leitos.ingestion import (
    fetch_bom_rnino_weekly,
    fetch_noaa_enso_monthly,
    fetch_noaa_roni_monthly,
    fetch_noaa_soi_monthly,
)

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
_HOLDOUT_YEAR = 2025

_REQUIRED_PANEL_COLS = (
    "nome_regiao_saude",
    "bacia_id",
    "ano_epi",
    "semana_epi",
    "cnes_uti_sus_leitos",
    "cnes_total_sus_leitos",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_modeling_panel(
    *,
    panel_path: Path,
    oof_path: Path,
    enso_cache_dir: Path = DATA_RAW / "wave2a",
    max_year: int = 2024,
) -> pl.DataFrame:
    """Combina painel v3, OOF hidrológico e ENSO sem jamais carregar o holdout."""
    if max_year >= _HOLDOUT_YEAR:
        raise ValueError(
            f"max_year={max_year} atinge o holdout {_HOLDOUT_YEAR}; use <= 2024."
        )
    if not panel_path.exists():
        raise FileNotFoundError(f"Painel v3 ausente: {panel_path}")
    if not oof_path.exists():
        raise FileNotFoundError(
            f"OOF hidrológico ausente: {oof_path}. Rode a Task 1.2.2 antes do benchmark."
        )

    panel = pl.read_parquet(panel_path).filter(pl.col("ano_epi") <= max_year)
    missing = [col for col in _REQUIRED_PANEL_COLS if col not in panel.columns]
    if missing:
        raise ValueError(f"Painel v3 sem colunas obrigatórias: {sorted(missing)}")

    oof = pl.read_parquet(oof_path).filter(pl.col("ano_epi") <= max_year)
    panel = panel.join(
        oof.select(
            [
                "nome_regiao_saude",
                "ano_epi",
                "semana_epi",
                "prob_enchente_f03_oof",
                "severidade_hidro_f03_oof",
            ]
        ),
        on=["nome_regiao_saude", "ano_epi", "semana_epi"],
        how="left",
    ).with_columns(
        pl.col("prob_enchente_f03_oof").fill_null(0.05),
        pl.col("severidade_hidro_f03_oof").fill_null(0.0),
    )

    monthly, _ = fetch_noaa_enso_monthly(
        cache_dir=enso_cache_dir, include_mei_v2=True, max_year=max_year
    )
    enso_weekly = build_causal_enso_weekly_features(monthly, lags=(4, 12)).select(
        ["ano_epi", "semana_epi", "el_nino_intensity", "enso_regime", "oni_c_lag4"]
    )
    panel = panel.join(enso_weekly, on=["ano_epi", "semana_epi"], how="left").with_columns(
        pl.col("el_nino_intensity").fill_null(0.0),
        pl.col("oni_c_lag4").fill_null(0.0),
        pl.col("enso_regime").fill_null("neutro"),
    )
    # RONI (Task 1.3.10 / E1): opcional — se o CPC estiver inacessível, o painel
    # segue sem `roni_c`/`roni_regime` e `--roni` falhará com DataContractError.
    try:
        roni_monthly, _ = fetch_noaa_roni_monthly(cache_dir=enso_cache_dir, max_year=max_year)
        roni_weekly = build_causal_roni_weekly_features(roni_monthly, lags=(4, 12)).select(
            ["ano_epi", "semana_epi", "roni_c", "roni_regime"]
        )
        panel = panel.join(roni_weekly, on=["ano_epi", "semana_epi"], how="left").with_columns(
            pl.col("roni_c").fill_null(0.0),
            pl.col("roni_regime").fill_null("neutro"),
        )
    except (SourceUnavailableError, DataContractError):
        pass
    # SOI (Task 1.3.10 / E3): opcional, como o RONI — se inacessível, `--soi` falha na guarda.
    try:
        soi_monthly, _ = fetch_noaa_soi_monthly(cache_dir=enso_cache_dir, max_year=max_year)
        soi_weekly = build_causal_soi_weekly_features(soi_monthly, lags=(4, 12)).select(
            ["ano_epi", "semana_epi", "soi_std"]
        )
        panel = panel.join(soi_weekly, on=["ano_epi", "semana_epi"], how="left").with_columns(
            pl.col("soi_std").fill_null(0.0),
        )
    except (SourceUnavailableError, DataContractError):
        pass
    # BOM Niño 3.4 relativo semanal (Task 1.3.10 / E2): opcional, como RONI/SOI.
    try:
        bom_weekly, _ = fetch_bom_rnino_weekly(cache_dir=enso_cache_dir, max_year=max_year)
        bom_wk = build_causal_bom_rnino_weekly(bom_weekly).select(
            ["ano_epi", "semana_epi", "bom_rnino"]
        )
        panel = panel.join(bom_wk, on=["ano_epi", "semana_epi"], how="left").with_columns(
            pl.col("bom_rnino").fill_null(0.0),
        )
    except (SourceUnavailableError, DataContractError):
        pass
    dengue_path = DATA_PROCESSED / "rs_dengue_regiao_semanal.parquet"
    if dengue_path.exists():
        dengue = pl.read_parquet(dengue_path).select(
            ["nome_regiao_saude", "ano_epi", "semana_epi", "dengue_casos"]
        )
        panel = panel.join(
            dengue, on=["nome_regiao_saude", "ano_epi", "semana_epi"], how="left"
        ).with_columns(pl.col("dengue_casos").fill_null(0.0))

    panel = panel.sort(["nome_regiao_saude", "ano_epi", "semana_epi"])
    # Memória causal do choque (Task 1.3.5): a resposta de leitos chega semanas depois
    # da enchente; estas colunas dão ao modelo a exposição acumulada, não só a de t.
    panel = panel.with_columns(
        pl.col("prob_enchente_f03_oof")
        .rolling_sum(window_size=4, min_samples=1)
        .over("nome_regiao_saude")
        .alias("shock_cum_4w"),
        pl.col("prob_enchente_f03_oof")
        .rolling_sum(window_size=8, min_samples=1)
        .over("nome_regiao_saude")
        .alias("shock_cum_8w"),
        pl.col("prob_enchente_f03_oof")
        .rolling_max(window_size=8, min_samples=1)
        .over("nome_regiao_saude")
        .alias("shock_max_8w"),
        pl.col("prob_enchente_f03_oof")
        .ewm_mean(alpha=0.16, min_samples=1)
        .over("nome_regiao_saude")
        .alias("shock_ewm_4w"),
    ).with_columns(
        pl.col("shock_cum_4w").fill_null(0.0),
        pl.col("shock_cum_8w").fill_null(0.0),
        pl.col("shock_max_8w").fill_null(0.0),
        pl.col("shock_ewm_4w").fill_null(0.0),
    )
    return panel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Monta o painel de modelagem do RS (<= 2024).")
    parser.add_argument(
        "--panel-path", default=str(DATA_PROCESSED / "rs_painel_semanal_regioes_v3.parquet")
    )
    parser.add_argument(
        "--oof-path", default=str(DATA_PROCESSED / "rs_f03_hydro_oof_v3.parquet")
    )
    parser.add_argument(
        "--out", default=str(DATA_PROCESSED / "rs_modeling_panel_v3.parquet")
    )
    parser.add_argument("--max-year", type=int, default=2024)
    args = parser.parse_args(argv)

    panel_path = Path(args.panel_path)
    panel = build_modeling_panel(
        panel_path=panel_path, oof_path=Path(args.oof_path), max_year=args.max_year
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(out_path)

    manifest = {
        "artifact": out_path.name,
        "sha256": _sha256(out_path),
        "rows": panel.height,
        "regions": panel["nome_regiao_saude"].n_unique(),
        "years": [int(v) for v in sorted(panel["ano_epi"].unique().to_list())],
        "inputs": {
            "panel_v3_sha256": _sha256(panel_path),
            "oof_sha256": _sha256(Path(args.oof_path)),
        },
        "holdout_policy": f"max_year={args.max_year} < {_HOLDOUT_YEAR}",
        "disclaimer": "Not for patient care or live clinical use.",
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Painel de modelagem: {out_path} ({panel.height:,} linhas, "
        f"{manifest['regions']} regiões, anos {manifest['years']})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
