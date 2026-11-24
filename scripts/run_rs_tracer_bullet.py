"""Executa o Tracer Bullet semanal 100% RS (F02 vs. F02.1 Intermunicipal/Aterro/Remanso).

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
    RS_BASIN_WEIGHTED_EDGES,
    BasinTopology,
    NasaPowerHydroAdapter,
    RegionCentroid,
    SplitPolicy,
    aggregate_daily_to_iso_weekly,
    build_causal_landfill_exposure,
    build_causal_referral_matrix,
    build_complete_daily_grid,
    compute_calendar_hydro_features,
    compute_catchment_flood_pressure,
    compute_estuarine_backwater_index,
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


def build_rs_regional_weekly_panel(*, train_end_year: int = 2022) -> pl.DataFrame:
    """Constrói painel semanal 100% RS (30 Regiões: Hidrologia + Aterro + Remanso + W_ref)."""
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
    flows = pl.read_parquet(_latest("sih_fluxos_referencia_semanal_v2_*.parquet"))
    mapbiomas = pl.read_parquet(_latest("mapbiomas_cobertura_rs_1985_2024_v1.parquet"))

    # 1. Grade diária completa + API causal + agregação ISO semanal (30/30 regiões)
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

    # 2. Exposição causal em áreas aterradas/banhados/pôlderes (MapBiomas <= train_end_year)
    mun_to_reg = ibge.select(["id_municipio", "id_regiao_saude"]).unique(subset=["id_municipio"])
    landfill = build_causal_landfill_exposure(
        mapbiomas, mun_to_reg=mun_to_reg, ref_year=train_end_year
    )
    weekly_hydro = weekly_hydro.join(landfill, on="id_regiao_saude", how="left")

    # 3. Roteamento hidrológico montante -> jusante (F02 simples + F02.1 ponderado/trânsito)
    basin_map = {
        str(r["id_regiao_saude"]): map_rs_health_region_to_basin(str(r["nome_regiao_saude"]))
        for r in centroids.iter_rows(named=True)
    }
    weekly_hydro = weekly_hydro.with_columns(
        pl.col("id_regiao_saude").replace_strict(basin_map).alias("bacia_id")
    )
    topo_simple = BasinTopology(edges=RS_BASIN_EDGES)
    weekly_hydro = topo_simple.propagate_upstream(
        weekly_hydro,
        basin_col="bacia_id",
        value_col="chuva_sem_mm",
        output_col="chuva_montante_mm",
    )
    weekly_hydro = topo_simple.propagate_upstream(
        weekly_hydro,
        basin_col="bacia_id",
        value_col="api_max_sem",
        output_col="api_montante_max",
    )

    topo_weighted = BasinTopology.from_weighted_edges(RS_BASIN_WEIGHTED_EDGES)
    weekly_hydro = topo_weighted.propagate_weighted_upstream(
        weekly_hydro,
        basin_col="bacia_id",
        value_col="chuva_sem_mm",
        output_col="chuva_montante_pond",
    )
    weekly_hydro = topo_weighted.propagate_weighted_upstream(
        weekly_hydro,
        basin_col="bacia_id",
        value_col="api_max_sem",
        output_col="api_montante_pond",
    )
    weekly_hydro = compute_estuarine_backwater_index(
        weekly_hydro,
        basin_col="bacia_id",
        south_wind_col="vento_sul_max_sem",
        storage_col="api_montante_pond",
        output_col="remanso_lagunar_idx",
    ).with_columns(
        (
            pl.col("indice_aterro_varzea") * (pl.col("api_max_sem") + pl.col("api_montante_pond"))
        ).alias("exposicao_aterro_ativa")
    )

    # 4. Grafo assistencial intermunicipal causal W_ref (MUNIC_RES -> CNES, <= train_end_year)
    cnes_reg = (
        cnes.sort(["ano", "mes"])
        .group_by("id_estabelecimento_cnes")
        .agg(
            pl.col("id_municipio").last().alias("id_municipio"),
            pl.col("is_hospital_ensino").any().alias("is_hospital_ensino"),
        )
        .join(mun_to_reg, on="id_municipio", how="inner")
    )
    w_ref = build_causal_referral_matrix(
        flows, cnes_to_reg=cnes_reg, mun_to_reg=mun_to_reg, max_year=train_end_year
    )
    weekly_hydro = compute_catchment_flood_pressure(
        weekly_hydro,
        w_ref,
        entity_col="id_regiao_saude",
        value_col="chuva_sem_mm",
        output_col="chuva_catchment_mm",
    )
    weekly_hydro = compute_catchment_flood_pressure(
        weekly_hydro,
        w_ref,
        entity_col="id_regiao_saude",
        value_col="api_max_sem",
        output_col="api_catchment_max",
    )

    # 5. Agrega alvos semanais SIH-SUS por Região de Saúde
    sih_reg = sih.join(cnes_reg, on="id_estabelecimento_cnes", how="inner")
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
    out_path = DATA_PROCESSED / "rs_painel_semanal_regioes_v2.parquet"
    panel.write_parquet(out_path)
    dt = time.perf_counter() - t0
    print(
        f"[OK] Painel semanal RS v2 construído em {dt:.2f}s: "
        f"{panel.height:,} linhas ({panel['id_regiao_saude'].n_unique()} regiões) -> {out_path}"
    )
    return panel


def main() -> None:
    panel = build_rs_regional_weekly_panel(train_end_year=2022)
    policy = SplitPolicy(train_end_year=2022, cal_year=2023, dev_audit_year=2024, holdout_year=2025)

    feats_f02 = (
        "chuva_sem_mm",
        "chuva_montante_mm",
        "api_max_sem",
        "api_montante_max",
        "umidade_solo_sem",
        "temp_media_sem_c",
    )
    feats_f02_1 = (
        "chuva_sem_mm",
        "chuva_montante_pond",
        "api_max_sem",
        "api_montante_pond",
        "remanso_lagunar_idx",
        "exposicao_aterro_ativa",
        "chuva_catchment_mm",
        "api_catchment_max",
        "umidade_solo_sem",
        "temp_media_sem_c",
    )

    # Ranking físico MapBiomas + Pôlder nas regiões-chave
    print("\n=== TOP 8 REGIÕES RS POR ÍNDICE DE ATERRO/VÁRZEA (MapBiomas <= 2022) ===")
    top_landfill = (
        panel.select(
            [
                "nome_regiao_saude",
                "bacia_id",
                "frac_urbano",
                "frac_banhado_agua",
                "frac_area_polder_regiao",
                "indice_aterro_varzea",
            ]
        )
        .unique()
        .sort("indice_aterro_varzea", descending=True)
        .head(8)
    )
    print(top_landfill)

    targets = [
        ("uti_adulto_diarias", "Diárias UTI Adulto"),
        ("respiratorio_internacoes", "Respiratório / SRAG"),
        ("leptospirose_internacoes", "Leptospirose (Onda 2)"),
        ("clinico_internacoes", "Clínico"),
    ]

    print("\n=== ABLAÇÃO F02 vs. F02.1 (Treino <= 2022 | Avaliação 2024 | h=2 sem.) ===")
    print(
        "| Alvo (30 Regiões RS) | MAE F02 | MAE F02.1 "
        "| MASE F02 | MASE F02.1 | R²_d F02 | R²_d F02.1 |"
    )
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for col, label in targets:
        r0 = evaluate_distributed_lag_tracer_bullet(
            panel,
            entity_col="nome_regiao_saude",
            target_col=col,
            feature_cols=feats_f02,
            horizon_weeks=2,
            eval_year=2024,
            policy=policy,
            ridge_alpha=50.0,
        )
        r1 = evaluate_distributed_lag_tracer_bullet(
            panel,
            entity_col="nome_regiao_saude",
            target_col=col,
            feature_cols=feats_f02_1,
            horizon_weeks=2,
            eval_year=2024,
            policy=policy,
            ridge_alpha=80.0,
        )
        print(
            f"| {label} | {r0.model_mae:.2f} | {r1.model_mae:.2f} "
            f"| {r0.mase_vs_persistence:.3f} | {r1.mase_vs_persistence:.3f} "
            f"| {r0.r2_delta:+.3f} | {r1.r2_delta:+.3f} |"
        )

    # Destaque POA (Região 10) e Pelotas/Sul (Região 21) em 2024 h=2 sem.
    for col, title in (
        ("uti_adulto_diarias", "Diárias UTI Adulto (h=2 sem., 2024)"),
        ("leptospirose_internacoes", "Leptospirose — Onda 2 (h=2 sem., 2024)"),
    ):
        r0 = evaluate_distributed_lag_tracer_bullet(
            panel,
            entity_col="nome_regiao_saude",
            target_col=col,
            feature_cols=feats_f02,
            horizon_weeks=2,
            eval_year=2024,
            policy=policy,
            ridge_alpha=50.0,
        )
        r1 = evaluate_distributed_lag_tracer_bullet(
            panel,
            entity_col="nome_regiao_saude",
            target_col=col,
            feature_cols=feats_f02_1,
            horizon_weeks=2,
            eval_year=2024,
            policy=policy,
            ridge_alpha=80.0,
        )
        comp = (
            r0.by_entity.select(
                ["nome_regiao_saude", pl.col("mase_vs_persistence").alias("mase_f02")]
            )
            .join(
                r1.by_entity.select(
                    ["nome_regiao_saude", pl.col("mase_vs_persistence").alias("mase_f02_1")]
                ),
                on="nome_regiao_saude",
            )
            .filter(pl.col("nome_regiao_saude").str.contains("10 -|21 -|08 -|29 -|23 -"))
            .sort("nome_regiao_saude")
        )
        print(f"\n--- Comparativo Regional ({title}) ---")
        print(comp)


if __name__ == "__main__":
    main()
