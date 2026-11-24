"""Orquestrador reprodutível do Dossiê Final V1.1 RS (Rebate 2024 + El Niño).

Executa de ponta a ponta sobre o painel real RS (treino <= 2022, calibração 2023,
auditoria 2024, holdout 2025 bloqueado):
- Convolução transitiva multi-salto (F02.2) + teleconexões ENSO (F02.3)
- OOF hidrológico causal (F03.1)
- Arquitetura ortogonal 2 estágios em taxa relativa (F04.1) + Mondrian-EVT (F04.2)
- Rebate contrafactual duplo 2024 (BSTS Pré-Choque + do() recursivo) + Oráculo El Niño
- Cluster-bootstrap por bacia hidrográfica (14 clusters)
- Compilação automática do Dossiê Final (.md + .pdf + Fig. C1-C7 + JSONs) em
  `reports/runs/rs_v1_1_final_2024/` e espelhamento em `~/Downloads/`.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from previsao_leitos.audit import (
    ChampionChallengerRatchet,
    ModelScorecard,
    RatchetCriteria,
    basin_cluster_bootstrap_ci,
    check_g1_collinearity_vif,
    check_g4_right_censoring,
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
    ConformalFloodCalibrator,
    MondrianEnsoConformalCalibrator,
    OrthogonalTwoStageOccupancyForecaster,
    QuantileGBDTForecaster,
    prepare_global_demand_splits,
)
from previsao_leitos.geo import (
    RS_BASIN_WEIGHTED_EDGES,
    BasinTopology,
    SplitPolicy,
    build_causal_enso_weekly_features,
    map_rs_health_region_to_basin,
)
from previsao_leitos.hydrology import generate_causal_hydro_oof_predictions

# Referência anual/mensal simplificada NOAA CPC ONI (2010-2025) para enriquecimento offline RS
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
    2025: -0.2,
}


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


def main() -> int:
    panel_path = Path("data/processed/rs_painel_semanal_regioes_v2.parquet")
    if not panel_path.exists():
        print(f"[ERRO] Painel não encontrado: {panel_path}")
        return 1

    policy = SplitPolicy(train_end_year=2022, cal_year=2023, dev_audit_year=2024, holdout_year=2025)
    raw_panel = pl.read_parquet(panel_path)

    # 1. Convolução multi-salto (F02.2) + teleconexões ENSO causais (F02.3)
    topo = BasinTopology.from_weighted_edges(RS_BASIN_WEIGHTED_EDGES)
    panel = raw_panel.with_columns(
        pl.col("nome_regiao_saude")
        .map_elements(map_rs_health_region_to_basin, return_dtype=pl.String)
        .alias("bacia_id")
    )
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

    # Porte regional causal (<= 2022) para Mondrian Conformal
    porte_map = (
        panel.filter(pl.col("ano_epi") <= 2022)
        .group_by("nome_regiao_saude")
        .agg(pl.col("uti_adulto_diarias").mean().alias("_mean_uti"))
        .with_columns(
            pl.when(pl.col("_mean_uti") >= 250.0)
            .then(pl.lit("grande"))
            .otherwise(pl.lit("interior"))
            .alias("porte_regional"),
            pl.col("_mean_uti").clip(lower_bound=10.0).alias("capacidade_sem"),
        )
        .select(["nome_regiao_saude", "porte_regional", "capacidade_sem"])
    )
    panel = panel.join(porte_map, on="nome_regiao_saude", how="left")

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

    # Anexa porte_regional e enso_regime para Mondrian Conformal
    meta_cols = panel.select(
        ["nome_regiao_saude", "ano_epi", "semana_epi", "porte_regional", "enso_regime", "bacia_id"]
    ).rename({"ano_epi": "target_ano_epi", "semana_epi": "target_semana_epi"})
    cal_df = sp_cal.eval_df.join(
        meta_cols, on=["nome_regiao_saude", "target_ano_epi", "target_semana_epi"], how="left"
    )
    eval_df = sp_eval.eval_df.join(
        meta_cols, on=["nome_regiao_saude", "target_ano_epi", "target_semana_epi"], how="left"
    )

    # 3. Champion V1 (QuantileGBDT + Conformal 2023) vs. Challenger V1.1 (F04.1 + Mondrian-EVT)
    v1_base = QuantileGBDTForecaster(max_iter=120, max_depth=5, random_state=42).fit(
        sp_cal.train_df, sp_cal.feature_cols
    )
    v1_cal = ConformalFloodCalibrator(base_model=v1_base).fit_calibration(
        train_df=sp_cal.train_df, cal_df=cal_df
    )
    v1_preds = v1_cal.predict_quantiles(eval_df)

    v11_base = OrthogonalTwoStageOccupancyForecaster(
        capacity_col="capacidade_sem",
        basal_cols=("sin_target_wk", "cos_target_wk"),
        min_samples_leaf=8,
        random_state=42,
    ).fit(sp_cal.train_df, sp_cal.feature_cols)
    v11_cal = MondrianEnsoConformalCalibrator(
        base_model=v11_base,
        group_cols=("porte_regional", "enso_regime"),
        use_evt_p95=True,
    ).fit_calibration(cal_df=cal_df)
    v11_preds = v11_cal.predict_quantiles(eval_df)

    eval_v1 = evaluate_probabilistic_predictions(
        eval_df, v1_preds, model_name="V1_QuantileGBDT_Conformal", n_boot=200, seed=42
    )
    eval_v11 = evaluate_probabilistic_predictions(
        eval_df, v11_preds, model_name="V1.1_TwoStage_MondrianEVT", n_boot=200, seed=42
    )
    bench_df = pl.DataFrame([eval_v1, eval_v11])

    # Cluster-bootstrap por bacia hidrográfica (14 clusters)
    abs_err_v11 = np.abs(
        eval_df["target_y"].to_numpy().astype(np.float64)
        - v11_preds["pred_p50"].to_numpy().astype(np.float64)
    )
    basin_ci = basin_cluster_bootstrap_ci(
        abs_err_v11, eval_df["bacia_id"].cast(pl.String).to_list(), n_boot=400, seed=42
    )

    # 4. Rebate contrafactual duplo 2024 + Oráculo prospectivo El Niño
    rebuttal = run_dual_counterfactual_rebuttal_2024(
        model=v11_cal,
        full_panel=panel.with_columns(pl.col("uti_adulto_diarias").alias("target_y")),
        train_df=sp_eval.train_df,
        eval_df=eval_df,
        entity_col="nome_regiao_saude",
        target_col="target_y",
        climate_prefixes=("chuva_montante_multihop", "prob_enchente_f03", "remanso_lagunar_idx"),
        neutral_zero_prefixes=("el_nino_intensity",),
        shock_year=2024,
        shock_weeks=(18, 22),
    )
    oracle_df = run_enso_what_if_oracle(model=v11_cal, baseline_eval_df=eval_df)

    # 5. Gates de Higiene + Catraca Champion-Challenger + Dossiê Final (.md + .pdf + ~/Downloads)
    y_true_arr = eval_df["target_y"].to_numpy().astype(np.float64)
    y_roll12_arr = eval_df["y_roll12"].to_numpy().astype(np.float64)
    y_surge = compute_surge_flags(y_true_arr, y_roll12_arr, multiplier=1.10)
    surge_thr = 1.10 * np.maximum(y_roll12_arr, 1e-6)
    ece_v1 = expected_calibration_error(
        y_surge, estimate_surge_probability_from_quantiles(v1_preds, surge_thr), n_bins=8
    )
    ece_v11 = expected_calibration_error(
        y_surge, estimate_surge_probability_from_quantiles(v11_preds, surge_thr), n_bins=8
    )

    hygiene_scorecard = compile_hygiene_scorecard(
        [
            check_g1_collinearity_vif(
                panel.filter(pl.col("ano_epi") <= 2024),
                ["chuva_sem_mm", "chuva_montante_multihop", "el_nino_intensity"],
            ),
            check_g4_right_censoring(
                panel.filter(pl.col("ano_epi") <= 2024).with_columns(
                    (pl.col("ano_epi") * 100 + pl.col("semana_epi")).alias("_t")
                ),
                time_col="_t",
                value_col="uti_adulto_diarias",
            ),
            check_g8_degenerate_surge_labels(y_true_arr, y_roll12_arr, y_surge),
        ]
    )

    # Especificidade no interior (semanas sem surto)
    int_mask = (eval_df["porte_regional"].to_numpy() == "interior") & (y_surge < 0.5)
    p_int_v1 = estimate_surge_probability_from_quantiles(v1_preds, surge_thr)[int_mask]
    p_int_v11 = estimate_surge_probability_from_quantiles(v11_preds, surge_thr)[int_mask]
    spec_v1 = float(np.mean(p_int_v1 < 0.5)) if np.any(int_mask) else 1.0
    spec_v11 = float(np.mean(p_int_v11 < 0.5)) if np.any(int_mask) else 1.0

    ratchet_eval = ChampionChallengerRatchet(
        RatchetCriteria(min_coverage_p90=0.60, min_interior_specificity=0.65)
    ).evaluate(
        champion=ModelScorecard(
            run_id="rs_v1_baseline",
            model_name="V1_QuantileGBDT_Conformal",
            mase=float(eval_v1["mase_vs_pers"]),
            pinball_p95=float(eval_v1["pinball_p95"]),
            coverage_p90=float(eval_v1["cobertura_p10_p90"]),
            peak_timing_error_weeks=2.0,
            interior_specificity=spec_v1,
            ece=ece_v1,
        ),
        challenger=ModelScorecard(
            run_id="rs_v1_1_final_2024",
            model_name="V1.1_TwoStage_MondrianEVT",
            mase=float(eval_v11["mase_vs_pers"]),
            pinball_p95=float(eval_v11["pinball_p95"]),
            coverage_p90=float(eval_v11["cobertura_p10_p90"]),
            peak_timing_error_weeks=rebuttal.peak_timing_error_weeks,
            interior_specificity=spec_v11,
            ece=ece_v11,
            hygiene_failed_gates=hygiene_scorecard.failed_count,
        ),
    )

    dossier = compile_execution_dossier(
        runs_root=Path("reports/runs"),
        run_id="rs_v1_1_final_2024",
        eval_year=2024,
        target_label="uti_adulto_diarias (RS 2024 — V1.1 Rebate Duplo + El Niño)",
        benchmark_table=bench_df,
        hygiene_scorecard=hygiene_scorecard,
        ratchet_evaluation=ratchet_eval,
        extra_notes=(
            f"MAE V1.1 (cluster-bootstrap 14 bacias): "
            f"{basin_ci.point:.2f} [IC 95%: {basin_ci.low:.2f}; {basin_ci.high:.2f}]",
            f"Especificidade no interior: V1={spec_v1:.1%} -> V1.1={spec_v11:.1%}",
            f"Rebate 2024 (BSTS vs do() recursivo): "
            f"obs={rebuttal.shock_excess_obs_mean:+.1f} | "
            f"pred={rebuttal.shock_excess_pred_mean:+.1f} | "
            f"erro timing pico={rebuttal.peak_timing_error_weeks:.2f} sem.",
            f"Oráculo El Niño (P50 médio neutro -> super El Niño): "
            f"{oracle_df['demanda_p50_media'][0]:.2f} -> {oracle_df['demanda_p50_media'][-1]:.2f}",
        ),
        seed=42,
        mirror_champion=True,
    )
    print(f"[OK] Dossiê Final V1.1 RS emitido: {dossier.markdown_path}")
    print(f"[OK] PDF compilado: {dossier.pdf_path}")
    print(f"[OK] Espelho Downloads: {dossier.mirrored_pdf_path}")
    print(f"[OK] Decisão Catraca: {ratchet_eval.decision} ({ratchet_eval.summary})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
