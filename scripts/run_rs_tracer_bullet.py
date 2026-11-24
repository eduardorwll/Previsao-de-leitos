"""Executa o Tracer Bullet semanal 100% RS (30 Regiões de Saúde, 2010-2024).

Aviso: exclusivamente para pesquisa e avaliação metodológica
(Not for patient care or live clinical use).
O ano 2025 (holdout intocado) NUNCA é lido para avaliação neste script.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import polars as pl

from previsao_leitos.geo import (
    RS_BASIN_EDGES,
    BasinTopology,
    NasaPowerHydroAdapter,
    RegionCentroid,
    SplitPolicy,
    aggregate_daily_to_iso_weekly,
    build_complete_daily_grid,
    compute_calendar_hydro_features,
    evaluate_distributed_lag_tracer_bullet,
    map_rs_health_region_to_basin,
)

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")


def _latest(pattern: str) -> Path:
    matches = sorted(DATA_RAW.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em {DATA_RAW} para {pattern!r}")
    return matches[-1]


def build_rs_regional_weekly_panel() -> pl.DataFrame:
    """Constrói painel semanal 100% RS (30 Regiões com Hidrologia montante + SIH + CNES)."""
    t0 = time.perf_counter()
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
    ibge = pl.read_parquet(_latest("ibge_municipios_rs_v1_*.parquet"))
    cnes = pl.read_parquet(_latest("cnes_hospitais_escola_v2_*.parquet"))
    sih = pl.read_parquet(_latest("sih_demanda_semanal_agg_v3_*.parquet"))

    # 1. Grade diária completa + API causal + agregação ISO semanal (30/30 regiões)
    daily_grid = build_complete_daily_grid(
        daily_hydro,
        entity_col="entity_id",
        date_col="data",
        value_cols=("precip_dia_mm", "temp_media_c", "umidade_solo_m3m3"),
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

    # 2. Propagação estritamente montante -> jusante (DAG oficial de bacias RS)
    basin_map = {
        str(r["id_regiao_saude"]): map_rs_health_region_to_basin(str(r["nome_regiao_saude"]))
        for r in centroids.iter_rows(named=True)
    }
    weekly_hydro = weekly_hydro.with_columns(
        pl.col("id_regiao_saude").replace_strict(basin_map).alias("bacia_id")
    )
    topo = BasinTopology(edges=RS_BASIN_EDGES)
    weekly_hydro = topo.propagate_upstream(
        weekly_hydro,
        basin_col="bacia_id",
        value_col="chuva_sem_mm",
        output_col="chuva_montante_mm",
    )
    weekly_hydro = topo.propagate_upstream(
        weekly_hydro,
        basin_col="bacia_id",
        value_col="api_max_sem",
        output_col="api_montante_max",
    )

    # 3. Mapeamento CNES -> município -> Região de Saúde + indicador de hospital-escola
    mun_to_reg = ibge.select(["id_municipio", "id_regiao_saude"]).unique(subset=["id_municipio"])
    cnes_reg = (
        cnes.sort(["ano", "mes"])
        .group_by("id_estabelecimento_cnes")
        .agg(
            pl.col("id_municipio").last().alias("id_municipio"),
            pl.col("is_hospital_ensino").any().alias("is_hospital_ensino"),
        )
        .join(mun_to_reg, on="id_municipio", how="inner")
    )

    sih_reg = sih.join(cnes_reg, on="id_estabelecimento_cnes", how="inner")

    # 4. Agrega alvos semanais por Região de Saúde (todos hospitais + hospitais-escola)
    demand_agg = sih_reg.group_by(["id_regiao_saude", "ano_epi", "semana_epi"]).agg(
        pl.col("internacoes").sum().cast(pl.Float64).alias("internacoes_total"),
        pl.col("internacoes")
        .filter(pl.col("especialidade") == "uti_adulto")
        .sum()
        .cast(pl.Float64)
        .alias("uti_adulto_internacoes"),
        pl.col("diarias_uti_total")
        .filter(pl.col("especialidade") == "uti_adulto")
        .sum()
        .cast(pl.Float64)
        .alias("uti_adulto_diarias"),
        pl.col("diarias_uti_total")
        .filter((pl.col("especialidade") == "uti_adulto") & pl.col("is_hospital_ensino"))
        .sum()
        .cast(pl.Float64)
        .alias("hu_uti_adulto_diarias"),
        pl.col("internacoes")
        .filter(pl.col("especialidade") == "clinico")
        .sum()
        .cast(pl.Float64)
        .alias("clinico_internacoes"),
        pl.col("internacoes")
        .filter(pl.col("grupo_cid") == "respiratorio")
        .sum()
        .cast(pl.Float64)
        .alias("respiratorio_internacoes"),
        pl.col("internacoes")
        .filter(pl.col("grupo_cid") == "leptospirose")
        .sum()
        .cast(pl.Float64)
        .alias("leptospirose_internacoes"),
        pl.col("internacoes")
        .filter(pl.col("grupo_cid") == "trauma")
        .sum()
        .cast(pl.Float64)
        .alias("trauma_internacoes"),
    )

    target_cols = [
        "internacoes_total",
        "uti_adulto_internacoes",
        "uti_adulto_diarias",
        "hu_uti_adulto_diarias",
        "clinico_internacoes",
        "respiratorio_internacoes",
        "leptospirose_internacoes",
        "trauma_internacoes",
    ]
    panel = (
        weekly_hydro.filter((pl.col("ano_epi") >= 2010) & (pl.col("ano_epi") <= 2025))
        .join(demand_agg, on=["id_regiao_saude", "ano_epi", "semana_epi"], how="left")
        .with_columns([pl.col(c).fill_null(0.0) for c in target_cols])
        .sort(["id_regiao_saude", "ano_epi", "semana_epi"])
    )

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "rs_painel_semanal_regioes_v1.parquet"
    panel.write_parquet(out_path)
    dt = time.perf_counter() - t0
    print(
        f"[OK] Painel semanal RS construído em {dt:.2f}s: "
        f"{panel.height:,} linhas ({panel['id_regiao_saude'].n_unique()} regiões) -> {out_path}"
    )
    return panel


def main() -> None:
    panel = build_rs_regional_weekly_panel()
    policy = SplitPolicy(train_end_year=2022, cal_year=2023, dev_audit_year=2024, holdout_year=2025)
    hydro_features = (
        "chuva_sem_mm",
        "chuva_montante_mm",
        "api_max_sem",
        "api_montante_max",
        "umidade_solo_sem",
        "temp_media_sem_c",
    )

    targets = [
        ("uti_adulto_diarias", "Diárias UTI Adulto (30 Regiões RS)"),
        ("uti_adulto_internacoes", "Internações UTI Adulto (30 Regiões RS)"),
        ("respiratorio_internacoes", "Internações Respiratório / SRAG (30 Regiões RS)"),
        ("leptospirose_internacoes", "Internações Leptospirose — Onda 2 (30 Regiões RS)"),
        ("clinico_internacoes", "Internações Clínicas (30 Regiões RS)"),
    ]

    for eval_year in (2023, 2024):
        print(f"\n=== TRACER BULLET RS (Treino <= 2022 | Avaliação {eval_year}) ===")
        print(
            "| Alvo | h (sem.) | MAE Modelo | MAE Persist. | MAE Sazonal "
            "| MASE vs Pers. | MASE vs Saz. | R²_delta |"
        )
        print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for col, label in targets:
            for h in (1, 2, 3, 4):
                rep = evaluate_distributed_lag_tracer_bullet(
                    panel,
                    entity_col="nome_regiao_saude",
                    target_col=col,
                    feature_cols=hydro_features,
                    horizon_weeks=h,
                    lag_weeks=(0, 1, 2, 3, 4),
                    eval_year=eval_year,
                    policy=policy,
                    allow_holdout=False,
                    ridge_alpha=50.0,
                )
                print(
                    f"| {label} | h={h} | {rep.model_mae:.2f} | {rep.persistence_mae:.2f} "
                    f"| {rep.seasonal_naive_mae:.2f} | {rep.mase_vs_persistence:.3f} "
                    f"| {rep.mase_vs_seasonal:.3f} | {rep.r2_delta:+.3f} |"
                )

    # Detalhamento regional em h=2 semanas para 2024 (UTI Adulto Diárias e Respiratório)
    for col, title in (
        ("uti_adulto_diarias", "Diárias UTI Adulto (h=2 sem., 2024)"),
        ("respiratorio_internacoes", "Internações Respiratório (h=2 sem., 2024)"),
    ):
        rep = evaluate_distributed_lag_tracer_bullet(
            panel,
            entity_col="nome_regiao_saude",
            target_col=col,
            feature_cols=hydro_features,
            horizon_weeks=2,
            lag_weeks=(0, 1, 2, 3, 4),
            eval_year=2024,
            policy=policy,
            allow_holdout=False,
            ridge_alpha=50.0,
        )
        print(f"\n--- Recorte Regional Destaque: {title} ---")
        focus = rep.by_entity.filter(
            pl.col("nome_regiao_saude").str.contains("10 -|08 -|29 -|21 -|17 -|23 -")
        ).sort("nome_regiao_saude")
        print(focus)


if __name__ == "__main__":
    main()
