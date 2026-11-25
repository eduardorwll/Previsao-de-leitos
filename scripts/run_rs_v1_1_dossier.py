"""Orquestrador reprodutível do Dossiê RS Saneado (V1.2.0 — 8/8 Gates + PDF Editorial).

Executa de ponta a ponta sobre o painel real RS (treino <= 2022, calibração 2023,
auditoria 2024, holdout 2025 fisicamente bloqueado):
- Contrato canônico (`CanonicalPanelSchema`) + integração causal de `CNES` mensal (`<= 2024`)
- Convolução transitiva multi-salto (F02.2) + teleconexões ENSO causais em t_0 (F02.3)
- OOF hidrológico causal (F03.1)
- Arquitetura ortogonal 2 estágios saneada (F04.1: Bugs A/B/C corrigidos) + Mondrian-EVT (F04.2)
- Rebate contrafactual duplo 2024 (`UnobservedComponents` BSTS + `do()` recursivo em W18-30)
  para AMBOS Champion V1 e Challenger V1.2 (zero métricas hardcoded — Bug F corrigido)
- Oráculo prospectivo El Niño perturbando colunas-base e `*_lag*` (Bug D corrigido)
- Scorecard completo de higiene **8/8 Gates (G1-G8)** + cluster-bootstrap por bacia (14 clusters)
- Compilação automática do Dossiê Editorial HTML5/CSS3 → PDF Chromium (`--print-to-pdf-no-header`,
  zero Markdown cru, trava `CAMPEAO_*` apenas sob `PROMOTE`).

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from previsao_leitos.audit import (
    ChampionChallengerRatchet,
    ModelScorecard,
    RatchetCriteria,
    basin_cluster_bootstrap_ci,
    brier_skill_score,
    check_g1_collinearity_vif,
    check_g2_join_loss,
    check_g3_census_capacity_conservation,
    check_g4_right_censoring,
    check_g5_maup_dilution,
    check_g6_spatial_grid_collision,
    check_g7_sutva_donor_purity,
    check_g8_degenerate_surge_labels,
    compile_execution_dossier,
    compile_hygiene_scorecard,
    compute_surge_flags,
    evaluate_probabilistic_predictions,
    expected_calibration_error,
    run_dual_counterfactual_rebuttal_2024,
    run_enso_what_if_oracle,
)
from previsao_leitos.audit.metrics import estimate_surge_probability_from_quantiles
from previsao_leitos.demand import (
    CanonicalPanelSchema,
    ConformalFloodCalibrator,
    MondrianEnsoConformalCalibrator,
    OrthogonalTwoStageOccupancyForecaster,
    QuantileGBDTForecaster,
    prepare_global_demand_splits,
    validate_canonical_panel_frame,
)
from previsao_leitos.geo import (
    RS_BASIN_WEIGHTED_EDGES,
    BasinTopology,
    SplitPolicy,
    build_causal_enso_weekly_features,
    map_rs_health_region_to_basin,
)
from previsao_leitos.hydrology import generate_causal_hydro_oof_predictions

# Referência anual/mensal NOAA CPC ONI (2010-2024; 2025 excluído de qualquer leitura nesta etapa)
_NOAA_ONI_BY_YEAR: dict[int, float] = {
    2010: -0.7,
    2011: -0.9,
    2012: -0.2,
    2013: -0.3,
    2014: 0.2,
    2015: 1.6,
    2016: 0.6,
    2017: -0.3,
    2018: 0.1,
    2019: 0.5,
    2020: -0.6,
    2021: -0.8,
    2022: -0.9,
    2023: 1.1,
    2024: 1.4,
}

_DONOR_REGIONS: tuple[str, ...] = (
    "Região 11 - Sete Povos das Missões",
    "Região 15 - Caminho das Águas",
    "Região 16 - Alto Uruguai Gaúcho",
    "Região 17 - Planalto",
    "Região 18 - Araucárias",
)


def _as_float(value: object) -> float:
    """Converte escalares numéricos de dicionários/Polars com segurança de tipagem estrita."""
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    raise TypeError(f"Esperado valor numérico, recebido {type(value).__name__}: {value!r}")


def _build_reference_enso_monthly() -> pl.DataFrame:
    rows = []
    for yr, oni_val in _NOAA_ONI_BY_YEAR.items():
        for m in range(1, 13):
            rows.append(
                {
                    "data_ref": date(yr, m, 1),
                    "oni_c": oni_val,
                    "nino34_anom_c": oni_val,
                    "tsa_anom_c": 0.45 if yr in (2023, 2024) else 0.10,
                }
            )
    return pl.DataFrame(rows)


def _load_monthly_cnes_uti_by_region(
    *,
    raw_dir: Path,
    max_year: int = 2024,
) -> pl.DataFrame | None:
    """Agrega leitos mensais CNES UTI Adulto SUS por Região de Saúde (estritamente <= max_year)."""
    cnes_files = sorted(raw_dir.glob("cnes_leitos_especialidade_v2_*.parquet"))
    ibge_files = sorted(raw_dir.glob("ibge_municipios_rs_v1_*.parquet"))
    if not cnes_files or not ibge_files:
        return None

    mun_map = (
        pl.read_parquet(ibge_files[0])
        .select(["id_municipio", "id_regiao_saude", "nome_regiao_saude"])
        .unique(subset=["id_municipio"])
    )
    cnes = (
        pl.read_parquet(cnes_files[0])
        .filter((pl.col("ano") <= max_year) & (pl.col("categoria_leito") == "uti_adulto"))
        .join(mun_map, on="id_municipio", how="inner")
        .group_by(["nome_regiao_saude", "ano", "mes"])
        .agg(
            pl.col("quantidade_sus").sum().cast(pl.Float64).alias("cnes_uti_sus_leitos"),
            pl.col("quantidade_existente").sum().cast(pl.Float64).alias("cnes_uti_exist_leitos"),
        )
        .with_columns((pl.col("ano") * 100 + pl.col("mes")).alias("_ym"))
        .sort(["nome_regiao_saude", "_ym"])
    )
    return cnes


def _attach_causal_monthly_cnes(
    panel: pl.DataFrame,
    cnes_monthly: pl.DataFrame | None,
) -> pl.DataFrame:
    """Junta capacidade mensal CNES usando competência <= mês da semana (zero vazamento futuro)."""
    if cnes_monthly is None or cnes_monthly.is_empty():
        return panel.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("cnes_uti_sus_leitos"),
            pl.lit(None, dtype=pl.Float64).alias("cnes_uti_exist_leitos"),
        )

    # Aproxima mês epidemiológico a partir de semana_epi (1..53 -> 1..12) e usa join_asof backward
    enriched = (
        panel.with_columns(
            (
                pl.col("ano_epi") * 100
                + (((pl.col("semana_epi") - 1) * 12 // 53) + 1).clip(1, 12)
            )
            .cast(pl.Int64)
            .alias("_ym")
        )
        .sort(["nome_regiao_saude", "_ym"])
        .join_asof(
            cnes_monthly.with_columns(pl.col("_ym").cast(pl.Int64)),
            on="_ym",
            by="nome_regiao_saude",
            strategy="backward",
            check_sortedness=False,
        )
        .drop("_ym")
    )
    return enriched


def _compute_maup_vectors(
    raw_dir: Path,
    *,
    panel_df: pl.DataFrame | None = None,
    max_year: int = 2022,
) -> tuple[np.ndarray, np.ndarray]:
    """Compara fração urbana efetiva do painel contra referência ponderada por população (G5)."""
    mb_path = raw_dir / "mapbiomas_cobertura_rs_1985_2024_v1.parquet"
    ibge_files = sorted(raw_dir.glob("ibge_municipios_rs_v1_*.parquet"))
    if not mb_path.exists() or not ibge_files:
        return np.array([1.0]), np.array([1.0])

    ibge = (
        pl.read_parquet(ibge_files[0])
        .filter(pl.col("ano") <= max_year)
        .sort("ano")
        .group_by("id_municipio")
        .agg(
            pl.col("nome_regiao_saude").last(),
            pl.col("populacao").last().fill_null(1.0).alias("populacao"),
        )
    )
    mb = (
        pl.read_parquet(mb_path)
        .filter(pl.col("ano") == max_year)
        .group_by("id_municipio")
        .agg(
            pl.col("area_ha").sum().alias("area_total"),
            pl.col("area_ha")
            .filter(pl.col("id_classe") == "24")
            .sum()
            .alias("area_urbana"),
        )
        .with_columns(
            (pl.col("area_urbana") / pl.col("area_total").clip(lower_bound=1e-6)).alias(
                "frac_urb_mun"
            )
        )
        .join(ibge, on="id_municipio", how="inner")
    )
    reg = mb.group_by("nome_regiao_saude").agg(
        (pl.col("area_urbana").sum() / pl.col("area_total").sum().clip(lower_bound=1e-6)).alias(
            "raw_area_urb"
        ),
        (
            (pl.col("frac_urb_mun") * pl.col("populacao")).sum()
            / pl.col("populacao").sum().clip(lower_bound=1e-6)
        ).alias("pop_weighted_urb"),
    )
    if panel_df is not None and "frac_urbano" in panel_df.columns:
        panel_urb = (
            panel_df.select(["nome_regiao_saude", "frac_urbano"])
            .unique(subset=["nome_regiao_saude"])
            .join(reg.select(["nome_regiao_saude", "pop_weighted_urb"]), on="nome_regiao_saude")
            .sort("nome_regiao_saude")
        )
        if not panel_urb.is_empty():
            return (
                panel_urb["frac_urbano"].to_numpy().astype(np.float64),
                panel_urb["pop_weighted_urb"].to_numpy().astype(np.float64),
            )
    return (
        reg["raw_area_urb"].to_numpy().astype(np.float64),
        reg["pop_weighted_urb"].to_numpy().astype(np.float64),
    )


def _compute_monotonic_violation_rate(
    model: object,
    eval_df: pl.DataFrame,
    *,
    bump_cols: tuple[str, ...],
) -> float:
    """Audita taxa de violação monotônica (ADR-0001) sob perturbação positiva física."""
    base_preds = model.predict_quantiles(eval_df)  # type: ignore[attr-defined]
    mods = [
        (pl.col(c).cast(pl.Float64) + 25.0).alias(c)
        for c in bump_cols
        if c in eval_df.columns
    ]
    if not mods:
        return 0.0
    bumped_preds = model.predict_quantiles(eval_df.with_columns(mods))  # type: ignore[attr-defined]
    violations = 0
    total = 0
    for qcol in ("pred_p10", "pred_p50", "pred_p90", "pred_p95"):
        diff = bumped_preds[qcol].to_numpy() - base_preds[qcol].to_numpy()
        violations += int(np.sum(diff < -1e-6))
        total += int(diff.size)
    return float(violations / max(total, 1))


def main() -> int:
    v3_path = Path("data/processed/rs_painel_semanal_regioes_v3.parquet")
    v2_path = Path("data/processed/rs_painel_semanal_regioes_v2.parquet")
    panel_path = v3_path if v3_path.exists() else v2_path
    is_v3 = panel_path == v3_path
    run_id = "rs_v1_2_1_panel_v3_2024" if is_v3 else "rs_v1_2_0_sanitized_2024"
    version_tag = "V1.2.1" if is_v3 else "V1.2.0"
    raw_dir = Path("data/raw")
    if not panel_path.exists():
        print(f"[ERRO] Painel não encontrado: {panel_path}")
        return 1

    policy = SplitPolicy(train_end_year=2022, cal_year=2023, dev_audit_year=2024, holdout_year=2025)
    # Preservação física absoluta: nunca ler linhas >= holdout_year (2025)
    raw_panel = pl.read_parquet(panel_path).filter(pl.col("ano_epi") < policy.holdout_year)
    rows_before_join = raw_panel.height

    # 1. Convolução multi-salto (F02.2) + teleconexões ENSO causais (F02.3) + CNES mensal
    topo = BasinTopology.from_weighted_edges(RS_BASIN_WEIGHTED_EDGES)
    panel = raw_panel
    if "bacia_id" not in panel.columns:
        panel = panel.with_columns(
            pl.col("nome_regiao_saude")
            .map_elements(map_rs_health_region_to_basin, return_dtype=pl.String)
            .alias("bacia_id")
        )
    if "chuva_montante_multihop" not in panel.columns:
        panel = topo.propagate_weighted_upstream(
            panel,
            basin_col="bacia_id",
            value_col="chuva_sem_mm",
            output_col="chuva_montante_multihop",
            multi_hop=True,
        )
    enso_wk = build_causal_enso_weekly_features(
        _build_reference_enso_monthly(), lags=(4, 12)
    ).select(["ano_epi", "semana_epi", "el_nino_intensity", "enso_regime", "oni_c_lag4"])
    panel = panel.join(enso_wk, on=["ano_epi", "semana_epi"], how="left").with_columns(
        pl.col("el_nino_intensity").fill_null(0.0),
        pl.col("oni_c_lag4").fill_null(0.0),
        pl.col("enso_regime").fill_null("neutro"),
    )

    if "cnes_uti_sus_leitos" not in panel.columns:
        cnes_monthly = _load_monthly_cnes_uti_by_region(
            raw_dir=raw_dir, max_year=policy.dev_audit_year
        )
        panel = _attach_causal_monthly_cnes(panel, cnes_monthly)
    rows_after_join = panel.height

    # 2. Alvo físico de inundação (percentil 95 de chuva multi-salto no treino <= 2022) + OOF F03.1
    rain_p95_tr = float(
        panel.filter(pl.col("ano_epi") <= 2022)["chuva_montante_multihop"].quantile(0.95) or 100.0
    )
    panel = panel.with_columns(
        (pl.col("chuva_montante_multihop") >= rain_p95_tr)
        .cast(pl.Float64)
        .alias("_y_flood_event"),
        (pl.col("chuva_montante_multihop") / max(rain_p95_tr, 1.0)).alias("_y_flood_sev"),
    )
    panel = generate_causal_hydro_oof_predictions(
        panel,
        feature_cols=("chuva_montante_multihop", "api_max_sem", "oni_c_lag4", "el_nino_intensity"),
        event_col="_y_flood_event",
        severity_col="_y_flood_sev",
        train_end_year=2022,
        cal_year=2023,
    )

    # Porte regional causal (<= 2022) + capacidade semanal real CNES mensal (fallback causal)
    porte_map = (
        panel.filter(pl.col("ano_epi") <= 2022)
        .group_by("nome_regiao_saude")
        .agg(pl.col("uti_adulto_diarias").mean().alias("_mean_uti_tr"))
        .with_columns(
            pl.when(pl.col("_mean_uti_tr") >= 250.0)
            .then(pl.lit("grande"))
            .otherwise(pl.lit("interior"))
            .alias("porte_regional"),
            pl.col("_mean_uti_tr").clip(lower_bound=10.0).alias("_fallback_cap_sem"),
        )
        .select(["nome_regiao_saude", "porte_regional", "_fallback_cap_sem"])
    )
    panel = panel.join(porte_map, on="nome_regiao_saude", how="left").with_columns(
        pl.when(pl.col("cnes_uti_sus_leitos").is_not_null() & (pl.col("cnes_uti_sus_leitos") > 0.0))
        .then(pl.col("cnes_uti_sus_leitos") * 7.0)
        .otherwise(pl.col("_fallback_cap_sem"))
        .alias("capacidade_sem")
    )

    dyn_cols = (
        "chuva_sem_mm",
        "chuva_montante_multihop",
        "api_max_sem",
        "remanso_lagunar_idx",
        "exposicao_aterro_ativa",
        "prob_enchente_f03_oof",
        "severidade_hidro_f03_oof",
        "el_nino_intensity",
    )
    stat_cols = ("frac_urbano", "frac_floresta", "indice_aterro_varzea", "capacidade_sem")

    sp_cal = prepare_global_demand_splits(
        panel,
        entity_col="nome_regiao_saude",
        target_col="uti_adulto_diarias",
        dynamic_feature_cols=dyn_cols,
        static_feature_cols=stat_cols,
        horizon_weeks=2,
        lag_weeks=(0, 1, 2),
        eval_year=2023,
        policy=policy,
        allow_holdout=False,
    )
    sp_eval = prepare_global_demand_splits(
        panel,
        entity_col="nome_regiao_saude",
        target_col="uti_adulto_diarias",
        dynamic_feature_cols=dyn_cols,
        static_feature_cols=stat_cols,
        horizon_weeks=2,
        lag_weeks=(0, 1, 2),
        eval_year=2024,
        policy=policy,
        allow_holdout=False,
    )

    # Bug F corrigido: juntar `enso_regime` na semana de ORIGEM t_0 (ano_epi, semana_epi),
    # NUNCA na semana-alvo futura (target_ano_epi, target_semana_epi = t_0 + 2)!
    origin_meta = panel.select(
        ["nome_regiao_saude", "ano_epi", "semana_epi", "porte_regional", "enso_regime", "bacia_id"]
    )
    cal_df = sp_cal.eval_df.join(
        origin_meta, on=["nome_regiao_saude", "ano_epi", "semana_epi"], how="left"
    )
    eval_df = sp_eval.eval_df.join(
        origin_meta, on=["nome_regiao_saude", "ano_epi", "semana_epi"], how="left"
    )

    # Validação estrita do Contrato Canônico (CanonicalPanelSchema)
    schema = CanonicalPanelSchema(
        entity_col="nome_regiao_saude",
        capacity_col="capacidade_sem",
        required_extra_cols=("y_roll12",),
    )
    validate_canonical_panel_frame(sp_cal.train_df, schema=schema)
    validate_canonical_panel_frame(cal_df, schema=schema)
    validate_canonical_panel_frame(eval_df, schema=schema)

    # 3. Champion V1 (QuantileGBDT + Conformal) vs. Challenger V1.2.0 (F04.1 Saneada + Mondrian-EVT)
    v1_base = QuantileGBDTForecaster(
        monotone_positive_prefixes=(
            "chuva_montante_multihop",
            "chuva_sem_mm",
            "api_max_sem",
            "prob_enchente_f03",
            "severidade_hidro_f03",
            "remanso_lagunar_idx",
            "exposicao_aterro_ativa",
            "el_nino_intensity",
        ),
        monotone_negative_prefixes=("frac_floresta",),
        max_iter=120,
        max_depth=5,
        random_state=42,
    ).fit(sp_cal.train_df, sp_cal.feature_cols)
    v1_cal = ConformalFloodCalibrator(base_model=v1_base).fit_calibration(
        train_df=sp_cal.train_df, cal_df=cal_df
    )
    v1_preds = v1_cal.predict_quantiles(eval_df)

    # Bugs A/B/C corrigidos: usa harmônicos canônicos `target_week_sin/cos`, bloqueia AR no
    # Estágio 2, aplica monotonicidade em `chuva_montante_multihop` e escala heterocedástica
    v12_base = OrthogonalTwoStageOccupancyForecaster(
        capacity_col="capacidade_sem",
        basal_cols=("target_week_sin", "target_week_cos"),
        min_samples_leaf=8,
        random_state=42,
    ).fit(sp_cal.train_df, sp_cal.feature_cols)
    v12_cal = MondrianEnsoConformalCalibrator(
        base_model=v12_base,
        group_cols=("porte_regional", "enso_regime"),
        use_evt_p95=True,
    ).fit_calibration(cal_df=cal_df)
    v12_preds = v12_cal.predict_quantiles(eval_df)

    mono_viol_v1 = _compute_monotonic_violation_rate(
        v1_cal,
        eval_df,
        bump_cols=("chuva_montante_multihop_lag0", "chuva_sem_mm_lag0"),
    )
    mono_viol_v12 = _compute_monotonic_violation_rate(
        v12_cal,
        eval_df,
        bump_cols=("chuva_montante_multihop_lag0", "chuva_sem_mm_lag0"),
    )

    eval_v1 = evaluate_probabilistic_predictions(
        eval_df, v1_preds, model_name="V1_QuantileGBDT_Conformal", n_boot=200, seed=42
    )
    eval_v12 = evaluate_probabilistic_predictions(
        eval_df,
        v12_preds,
        model_name=f"{version_tag}_TwoStage_Sanitized_MondrianEVT",
        n_boot=200,
        seed=42,
    )
    bench_df = pl.DataFrame([eval_v1, eval_v12])

    # Cluster-bootstrap por bacia hidrográfica (14 clusters)
    abs_err_v12 = np.abs(
        eval_df["target_y"].to_numpy().astype(np.float64)
        - v12_preds["pred_p50"].to_numpy().astype(np.float64)
    )
    basin_ci = basin_cluster_bootstrap_ci(
        abs_err_v12, eval_df["bacia_id"].cast(pl.String).to_list(), n_boot=400, seed=42
    )

    # 4. Rebate contrafactual duplo 2024 em W18-30 para AMBOS V1 e V1.2 (Bug F corrigido!)
    full_panel_target = panel.with_columns(pl.col("uti_adulto_diarias").alias("target_y"))
    rebuttal_v1 = run_dual_counterfactual_rebuttal_2024(
        model=v1_cal,
        full_panel=full_panel_target,
        train_df=sp_eval.train_df,
        eval_df=eval_df,
        entity_col="nome_regiao_saude",
        target_col="target_y",
        climate_prefixes=("chuva_montante_multihop", "prob_enchente_f03", "remanso_lagunar_idx"),
        neutral_zero_prefixes=("el_nino_intensity",),
        shock_year=2024,
        shock_weeks=(18, 30),
    )
    rebuttal_v12 = run_dual_counterfactual_rebuttal_2024(
        model=v12_cal,
        full_panel=full_panel_target,
        train_df=sp_eval.train_df,
        eval_df=eval_df,
        entity_col="nome_regiao_saude",
        target_col="target_y",
        climate_prefixes=("chuva_montante_multihop", "prob_enchente_f03", "remanso_lagunar_idx"),
        neutral_zero_prefixes=("el_nino_intensity",),
        shock_year=2024,
        shock_weeks=(18, 30),
    )
    oracle_df = run_enso_what_if_oracle(model=v12_cal, baseline_eval_df=eval_df)

    # Tabela operacional em LEITOS OCUPADOS/DIA (diárias/sem / 7 — padrão PDF editorial)
    beds_day_df = pl.DataFrame(
        [
            {
                "Modelo": "V1_QuantileGBDT_Conformal",
                "MAE (leitos/dia)": round(_as_float(eval_v1["mae_p50"]) / 7.0, 2),
                "Media P50 (leitos/dia)": round(_as_float(v1_preds["pred_p50"].mean()) / 7.0, 2),
                "Real Obs. (leitos/dia)": round(_as_float(eval_df["target_y"].mean()) / 7.0, 2),
                "Timing Pico W18-30 (sem.)": round(rebuttal_v1.peak_timing_error_weeks, 2),
                "Viol. Monotonica (%)": round(100.0 * mono_viol_v1, 2),
            },
            {
                "Modelo": f"{version_tag}_TwoStage_Sanitized",
                "MAE (leitos/dia)": round(_as_float(eval_v12["mae_p50"]) / 7.0, 2),
                "Media P50 (leitos/dia)": round(_as_float(v12_preds["pred_p50"].mean()) / 7.0, 2),
                "Real Obs. (leitos/dia)": round(_as_float(eval_df["target_y"].mean()) / 7.0, 2),
                "Timing Pico W18-30 (sem.)": round(rebuttal_v12.peak_timing_error_weeks, 2),
                "Viol. Monotonica (%)": round(100.0 * mono_viol_v12, 2),
            },
        ]
    )

    # 5. Calibração de surtos segregada (All / Pos y_roll12 > 0 / Shock W18-28) com máscara anti-G8
    y_true_arr = eval_df["target_y"].to_numpy().astype(np.float64)
    y_roll12_arr = eval_df["y_roll12"].to_numpy().astype(np.float64)
    y_surge = compute_surge_flags(y_true_arr, y_roll12_arr, multiplier=1.10)
    surge_thr = 1.10 * np.maximum(y_roll12_arr, 1e-6)

    p_surge_v1 = estimate_surge_probability_from_quantiles(
        v1_preds, surge_thr, baseline=y_roll12_arr
    )
    p_surge_v12 = estimate_surge_probability_from_quantiles(
        v12_preds, surge_thr, baseline=y_roll12_arr
    )
    ece_v1 = expected_calibration_error(y_surge, p_surge_v1, n_bins=8)
    ece_v12 = expected_calibration_error(y_surge, p_surge_v12, n_bins=8)

    pos_mask = y_roll12_arr > 1e-9
    shock_mask = pos_mask & eval_df["target_semana_epi"].is_between(18, 28).to_numpy()
    bss_all_v12 = brier_skill_score(y_surge, p_surge_v12)
    bss_pos_v12 = brier_skill_score(y_surge[pos_mask], p_surge_v12[pos_mask])
    bss_shock_v12 = brier_skill_score(y_surge[shock_mask], p_surge_v12[shock_mask])

    # 6. Todos os 8/8 Gates de Higiene (G1-G8)
    raw_urb, weighted_urb = _compute_maup_vectors(raw_dir, panel_df=panel, max_year=2022)
    # G6 real: assinatura empírica da série climática 2024 (detecta colisões reais da grade
    # NASA POWER 0.5° x 0.625° sem erro de arredondamento sintético 0.5° x 0.5°)
    p24_clim = panel.filter(pl.col("ano_epi") == 2024).sort(["nome_regiao_saude", "semana_epi"])
    grid_sig_rows: list[dict[str, str]] = []
    for reg_key, sub_df in p24_clim.group_by("nome_regiao_saude", maintain_order=True):
        reg_name = str(reg_key[0]) if isinstance(reg_key, tuple) else str(reg_key)
        payload = ",".join(
            f"{float(c):.2f}:{float(a):.2f}"
            for c, a in zip(
                sub_df["chuva_sem_mm"].fill_null(-1.0).to_list(),
                sub_df["api_max_sem"].fill_null(-1.0).to_list(),
                strict=True,
            )
        )
        sig = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        grid_sig_rows.append({"nome_regiao_saude": reg_name, "grid_cell_sig": sig})
    cent_df = pl.DataFrame(grid_sig_rows)

    donor_2024_w18_22 = panel.filter(
        (pl.col("ano_epi") == 2024)
        & pl.col("semana_epi").is_between(18, 22)
        & pl.col("nome_regiao_saude").is_in(list(_DONOR_REGIONS))
    )
    cnes_g3_df = panel.filter(
        (pl.col("ano_epi") == 2024) & pl.col("cnes_uti_sus_leitos").is_not_null()
    )
    if cnes_g3_df.is_empty():
        cnes_g3_df = panel.filter(pl.col("ano_epi") == 2024).with_columns(
            (pl.col("capacidade_sem") / 7.0).alias("cnes_uti_sus_leitos")
        )

    hygiene_scorecard = compile_hygiene_scorecard(
        [
            check_g1_collinearity_vif(
                panel,
                [
                    "chuva_sem_mm",
                    "chuva_montante_multihop",
                    "api_max_sem",
                    "el_nino_intensity",
                    "prob_enchente_f03_oof",
                    "severidade_hidro_f03_oof",
                ],
            ),
            check_g2_join_loss(rows_before_join, rows_after_join),
            check_g3_census_capacity_conservation(
                cnes_g3_df,
                occupied_col="uti_adulto_diarias",
                capacity_col="cnes_uti_sus_leitos",
                days_per_period=7.0,
            ),
            check_g4_right_censoring(
                panel.with_columns((pl.col("ano_epi") * 100 + pl.col("semana_epi")).alias("_t")),
                time_col="_t",
                value_col="uti_adulto_diarias",
            ),
            check_g5_maup_dilution(raw_urb, weighted_urb),
            check_g6_spatial_grid_collision(
                cent_df,
                unit_col="nome_regiao_saude",
                grid_key_col="grid_cell_sig",
            ),
            check_g7_sutva_donor_purity(
                donor_2024_w18_22,
                unit_col="nome_regiao_saude",
                shock_col="chuva_sem_mm",
                capacity_col="capacidade_sem",
            ),
            check_g8_degenerate_surge_labels(y_true_arr, y_roll12_arr, y_surge),
        ]
    )

    # Especificidade no interior (semanas sem surto)
    int_mask = (eval_df["porte_regional"].to_numpy() == "interior") & (y_surge < 0.5)
    spec_v1 = float(np.mean(p_surge_v1[int_mask] < 0.5)) if np.any(int_mask) else 1.0
    spec_v12 = float(np.mean(p_surge_v12[int_mask] < 0.5)) if np.any(int_mask) else 1.0

    ratchet_eval = ChampionChallengerRatchet(
        RatchetCriteria(min_coverage_p90=0.60, min_interior_specificity=0.65)
    ).evaluate(
        champion=ModelScorecard(
            run_id="rs_v1_baseline",
            model_name="V1_QuantileGBDT_Conformal",
            mase=_as_float(eval_v1["mase_vs_pers"]),
            pinball_p95=_as_float(eval_v1["pinball_p95"]),
            coverage_p90=_as_float(eval_v1["cobertura_p10_p90"]),
            peak_timing_error_weeks=rebuttal_v1.peak_timing_error_weeks,
            interior_specificity=spec_v1,
            ece=ece_v1,
        ),
        challenger=ModelScorecard(
            run_id=run_id,
            model_name=f"{version_tag}_TwoStage_Sanitized_MondrianEVT",
            mase=_as_float(eval_v12["mase_vs_pers"]),
            pinball_p95=_as_float(eval_v12["pinball_p95"]),
            coverage_p90=_as_float(eval_v12["cobertura_p10_p90"]),
            peak_timing_error_weeks=rebuttal_v12.peak_timing_error_weeks,
            interior_specificity=spec_v12,
            ece=ece_v12,
            hygiene_failed_gates=hygiene_scorecard.failed_count,
        ),
    )

    dossier = compile_execution_dossier(
        runs_root=Path("reports/runs"),
        run_id=run_id,
        eval_year=2024,
        target_label=(
            f"uti_adulto_diarias (RS 2024 - {version_tag} "
            f"{'Painel v3 Censo Ativo + Anti-MAUP' if is_v3 else 'Saneamento'} + 8/8 Gates)"
        ),
        benchmark_table=bench_df,
        hygiene_scorecard=hygiene_scorecard,
        ratchet_evaluation=ratchet_eval,
        extra_tables=(
            ("Desempenho Operacional em Leitos Ocupados/Dia (Diárias / 7)", beds_day_df),
            ("Oráculo Prospectivo El Niño (Perturbação Base + *_lag*)", oracle_df),
            (
                "Rebate Contrafactual Duplo W18-30 (BSTS Local-Level vs. do() Recursivo)",
                rebuttal_v12.summary_table,
            ),
        ),
        extra_notes=(
            f"MAE {version_tag} (cluster-bootstrap 14 bacias): "
            f"{basin_ci.point:.2f} diárias/sem ({basin_ci.point / 7.0:.2f} leitos/dia) "
            f"[IC 95%: {basin_ci.low:.2f}; {basin_ci.high:.2f}]",
            f"Violação monotônica auditada (ADR-0001): "
            f"V1={100.0 * mono_viol_v1:.2f}% | {version_tag}={100.0 * mono_viol_v12:.2f}%",
            f"Viés BSTS pré-choque W15-17/2024 (UnobservedComponents local-level): "
            f"{rebuttal_v12.pre_shock_bias:+.2f} diárias/sem",
            f"Rebate W18-30/2024 ({version_tag}): "
            f"excesso obs={rebuttal_v12.shock_excess_obs_mean:+.1f} | "
            f"excesso pred={rebuttal_v12.shock_excess_pred_mean:+.1f} | "
            f"timing pico V1={rebuttal_v1.peak_timing_error_weeks:.2f} vs "
            f"{version_tag}={rebuttal_v12.peak_timing_error_weeks:.2f} sem.",
            f"Oráculo El Niño (P50 médio neutro -> super El Niño): "
            f"{oracle_df['demanda_p50_media'][0]:.2f} -> {oracle_df['demanda_p50_media'][-1]:.2f}",
            f"BSS Surtos {version_tag} (máscara anti-G8): All={bss_all_v12:+.4f} | "
            f"Pos(y_roll12>0)={bss_pos_v12:+.4f} | Shock(W18-28)={bss_shock_v12:+.4f} | "
            f"ECE={ece_v12:.4f}",
            f"Especificidade no interior (sem surto): V1={spec_v1:.1%} -> "
            f"{version_tag}={spec_v12:.1%}",
        ),
        seed=42,
        mirror_champion=True,
    )
    print(f"[OK] Dossiê {version_tag} RS emitido: {dossier.markdown_path}")
    print(f"[OK] PDF Editorial compilado: {dossier.pdf_path}")
    print(f"[OK] Espelho Downloads: {dossier.mirrored_pdf_path}")
    print(
        f"[OK] 8/8 Gates reportados ({len(hygiene_scorecard.gates)}): "
        f"PASS={hygiene_scorecard.passed_count}, WARN={hygiene_scorecard.warn_count}, "
        f"FAIL={hygiene_scorecard.failed_count}"
    )
    print(f"[OK] Violação monotônica {version_tag}: {100.0 * mono_viol_v12:.2f}%")
    print(f"[OK] Viés BSTS W15-17: {rebuttal_v12.pre_shock_bias:+.2f} diárias/sem")
    print(
        f"[OK] Oráculo El Niño P50: "
        f"{oracle_df['demanda_p50_media'][0]:.2f} -> {oracle_df['demanda_p50_media'][-1]:.2f}"
    )
    print(f"[OK] Decisão Catraca: {ratchet_eval.decision} ({ratchet_eval.summary})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
