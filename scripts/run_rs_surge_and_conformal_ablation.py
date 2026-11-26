"""Ablação controlada Task 1.2.4: Calibração de Surtos (Platt/Beta) + AgACI vs. SPCI (RS 2024).

Executa estritamente sobre `RS <= 2024` (holdouts `2025` e `2026` bloqueados fisicamente):
1. Compara probabilidade de surto via quantis (V1/V1.2) vs. `MonotonicSurgeClassifier` dedicado
   (raw, `RollingPlatt`, `RollingBeta` calibrados OOF causal em `2019-2023`), reportando
   `BSS_all`, `BSS_pos (y_roll12 > 0)`, `BSS_shock (W18-28)`, `ECE`, `ROC-AUC`, `PR-AUC`,
   especificidade no interior, Net Benefit e taxa de violação monotônica (`0,00%`).
2. Compara calibradores conformais normalizados por capacidade CNES sobre o Campeão `V1`:
   `ConformalFloodCalibrator (2023)` vs. `MondrianEnsoConformalCalibrator (2023)` vs.
   `AgACIConformalCalibrator (lag=2)` vs. `SPCIConformalCalibrator (QRF, lag=2)`.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from previsao_leitos.audit import (
    brier_skill_score,
    compute_surge_flags,
    decision_curve_net_benefit,
    estimate_surge_probability_from_quantiles,
    evaluate_probabilistic_predictions,
    expected_calibration_error,
)
from previsao_leitos.demand import (
    AgACIConformalCalibrator,
    ConformalFloodCalibrator,
    MondrianEnsoConformalCalibrator,
    MonotonicSurgeClassifier,
    QuantileGBDTForecaster,
    SPCIConformalCalibrator,
    prepare_global_demand_splits,
)
from previsao_leitos.geo import (
    RS_BASIN_WEIGHTED_EDGES,
    BasinTopology,
    SplitPolicy,
    build_causal_enso_weekly_features,
    map_rs_health_region_to_basin,
)
from previsao_leitos.ingestion import fetch_noaa_enso_monthly

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")


def _build_rs_panel_for_ablation() -> pl.DataFrame:
    panel_path = DATA_PROCESSED / "rs_painel_semanal_regioes_v3.parquet"
    oof_path = DATA_PROCESSED / "rs_f03_hydro_oof_v3.parquet"
    policy = SplitPolicy(train_end_year=2022, cal_year=2023, dev_audit_year=2024, holdout_year=2025)
    panel = pl.read_parquet(panel_path).filter(pl.col("ano_epi") < policy.holdout_year)

    topo = BasinTopology.from_weighted_edges(RS_BASIN_WEIGHTED_EDGES)
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

    noaa_monthly, _ = fetch_noaa_enso_monthly(
        cache_dir=DATA_RAW / "wave2a", include_mei_v2=True, max_year=2024
    )
    enso_wk = build_causal_enso_weekly_features(noaa_monthly, lags=(4, 12))
    enso_cols = [c for c in enso_wk.columns if c not in ("ano_epi", "semana_epi")]
    panel = panel.drop([c for c in enso_cols if c in panel.columns]).join(
        enso_wk, on=["ano_epi", "semana_epi"], how="left"
    )
    panel = panel.with_columns(
        pl.col("oni_c").fill_null(0.0),
        pl.col("el_nino_intensity").fill_null(0.0),
        pl.col("enso_regime").fill_null("neutral"),
    )

    if oof_path.exists():
        oof_df = pl.read_parquet(oof_path)
        panel = panel.drop(
            [c for c in ("prob_enchente_f03_oof", "severidade_hidro_f03_oof") if c in panel.columns]
        ).join(oof_df, on=["nome_regiao_saude", "ano_epi", "semana_epi"], how="left")
        panel = panel.with_columns(
            pl.col("prob_enchente_f03_oof").fill_null(0.0),
            pl.col("severidade_hidro_f03_oof").fill_null(0.0),
        )

    if "capacidade_sem" not in panel.columns:
        cap_base = (
            panel.filter(pl.col("ano_epi") <= 2022)
            .group_by("nome_regiao_saude")
            .agg(pl.col("uti_adulto_diarias").quantile(0.95).alias("_cap_fallback"))
        )
        panel = (
            panel.join(cap_base, on="nome_regiao_saude", how="left")
            .with_columns(
                pl.when(
                    pl.col("cnes_uti_sus_leitos").is_not_null()
                    & (pl.col("cnes_uti_sus_leitos") > 0)
                )
                .then(pl.col("cnes_uti_sus_leitos") * 7.0)
                .otherwise(pl.col("_cap_fallback").fill_null(70.0))
                .alias("capacidade_sem")
            )
            .drop("_cap_fallback")
        )

    porte_df = (
        panel.filter(pl.col("ano_epi") <= 2022)
        .group_by("nome_regiao_saude")
        .agg(pl.col("uti_adulto_diarias").mean().alias("_mean_uti"))
        .with_columns(
            pl.when(pl.col("_mean_uti") >= 500.0)
            .then(pl.lit("metropole"))
            .when(pl.col("_mean_uti") >= 100.0)
            .then(pl.lit("polo"))
            .otherwise(pl.lit("interior"))
            .alias("porte_regional")
        )
        .select(["nome_regiao_saude", "porte_regional"])
    )
    return panel.join(porte_df, on="nome_regiao_saude", how="left")


def _score_surge(
    name: str,
    y_surge: np.ndarray,
    probs: np.ndarray,
    pos_mask: np.ndarray,
    shock_mask: np.ndarray,
    int_mask: np.ndarray,
    mono_viol: float,
) -> dict[str, float | str]:
    nb = decision_curve_net_benefit(y_surge[pos_mask], probs[pos_mask], thresholds=(0.20, 0.30))
    nb20 = float(nb.filter(pl.col("limiar_pt") == 0.20)["net_benefit_modelo"].item())
    nb30 = float(nb.filter(pl.col("limiar_pt") == 0.30)["net_benefit_modelo"].item())
    return {
        "model": name,
        "bss_all": round(float(brier_skill_score(y_surge, probs)), 4),
        "bss_pos": round(float(brier_skill_score(y_surge[pos_mask], probs[pos_mask])), 4),
        "bss_shock": round(float(brier_skill_score(y_surge[shock_mask], probs[shock_mask])), 4),
        "ece_all": round(float(expected_calibration_error(y_surge, probs, n_bins=8)), 4),
        "ece_pos": round(
            float(expected_calibration_error(y_surge[pos_mask], probs[pos_mask], n_bins=8)), 4
        ),
        "roc_auc_pos": round(float(roc_auc_score(y_surge[pos_mask], probs[pos_mask])), 4),
        "pr_auc_pos": round(float(average_precision_score(y_surge[pos_mask], probs[pos_mask])), 4),
        "interior_specificity": round(
            float(np.mean(probs[int_mask] < 0.5)) if np.any(int_mask) else 1.0, 4
        ),
        "net_benefit_pt20": round(nb20, 4),
        "net_benefit_pt30": round(nb30, 4),
        "monotonic_violation_rate": round(float(mono_viol), 6),
    }


def _as_float(val: object) -> float:
    return float(val) if isinstance(val, (int, float, np.floating, np.integer)) else 0.0


def _score_conformal(
    name: str,
    eval_df: pl.DataFrame,
    preds_df: pl.DataFrame,
    shock_mask: np.ndarray,
    mono_viol: float,
) -> dict[str, float | str]:
    ev = evaluate_probabilistic_predictions(
        eval_df, preds_df, model_name=name, block_col="nome_regiao_saude"
    )
    y = eval_df["target_y"].to_numpy().astype(np.float64)
    p10 = preds_df["pred_p10"].to_numpy().astype(np.float64)
    p90 = preds_df["pred_p90"].to_numpy().astype(np.float64)
    cov_shock = float(
        np.mean((y[shock_mask] >= p10[shock_mask]) & (y[shock_mask] <= p90[shock_mask]))
    )
    width_beds_day = float(np.mean((p90 - p10) / 7.0))
    return {
        "calibrator": name,
        "mase_vs_pers": round(_as_float(ev["mase_vs_pers"]), 4),
        "mae_beds_day": round(_as_float(ev["mae_p50"]) / 7.0, 3),
        "pinball_p90": round(_as_float(ev["pinball_p90"]), 4),
        "pinball_p95": round(_as_float(ev["pinball_p95"]), 4),
        "cov_p10_p90": round(_as_float(ev["cobertura_p10_p90"]), 4),
        "cov_shock_w18_28": round(cov_shock, 4),
        "mean_width_beds_day": round(width_beds_day, 2),
        "monotonic_violation_rate": round(float(mono_viol), 6),
    }


def main() -> int:
    t0 = time.perf_counter()
    panel = _build_rs_panel_for_ablation()

    dyn_cols = [
        c
        for c in (
            "chuva_sem_mm",
            "chuva_montante_multihop",
            "api_max_sem",
            "api_montante_pond",
            "remanso_lagunar_idx",
            "prob_enchente_f03_oof",
            "severidade_hidro_f03_oof",
            "el_nino_intensity",
            "oni_c",
        )
        if c in panel.columns
    ]
    stat_cols = [
        c
        for c in (
            "capacidade_sem",
            "frac_urbano",
            "indice_aterro_varzea",
            "frac_area_polder_regiao",
        )
        if c in panel.columns
    ]

    policy = SplitPolicy(train_end_year=2022, cal_year=2023, dev_audit_year=2024, holdout_year=2025)
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
    )

    cal_df = sp_cal.eval_df
    eval_df = sp_eval.eval_df
    hist_df = pl.concat([sp_eval.train_df, sp_cal.eval_df], how="diagonal_relaxed")

    # 1. Ablação Calibração Probabilística de Surtos (RS 2024)
    y_true = eval_df["target_y"].to_numpy().astype(np.float64)
    y_roll12 = eval_df["y_roll12"].to_numpy().astype(np.float64)
    y_surge = compute_surge_flags(y_true, y_roll12, multiplier=1.10)
    surge_thr = 1.10 * np.maximum(y_roll12, 1e-6)
    pos_mask = y_roll12 > 1e-9
    shock_mask = pos_mask & eval_df["target_semana_epi"].is_between(18, 28).to_numpy()
    int_mask = (eval_df["porte_regional"].to_numpy() == "interior") & (y_surge < 0.5)

    mono_pos_feats = tuple(
        c
        for c in sp_eval.feature_cols
        if any(
            c.startswith(p)
            for p in (
                "chuva_sem_mm",
                "chuva_montante_multihop",
                "api_max_sem",
                "api_montante_pond",
                "remanso_lagunar_idx",
                "prob_enchente_f03",
                "severidade_hidro_f03",
                "el_nino_intensity",
                "oni_c",
            )
        )
    )

    v1_base = QuantileGBDTForecaster(
        monotone_positive_prefixes=mono_pos_feats, max_iter=200, random_state=42
    ).fit(sp_eval.train_df, sp_eval.feature_cols)
    v1_conf = ConformalFloodCalibrator(base_model=v1_base).fit_calibration(
        train_df=sp_eval.train_df, cal_df=cal_df
    )
    v1_preds = v1_conf.predict_quantiles(eval_df)
    p_v1_q = estimate_surge_probability_from_quantiles(v1_preds, surge_thr, baseline=y_roll12)

    surge_features = list(sp_eval.feature_cols)
    # Adiciona razão de ocupação basal/origem (observada em t_0) para discriminar surtos
    hist_surge = hist_df.with_columns(
        (pl.col("y_origin") / pl.max_horizontal(pl.col("y_roll12"), pl.lit(1.0))).alias(
            "ratio_origin_roll12"
        ),
        (pl.col("y_origin") / pl.max_horizontal(pl.col("capacidade_sem"), pl.lit(1.0))).alias(
            "occ_origin_cnes"
        ),
    )
    eval_surge = eval_df.with_columns(
        (pl.col("y_origin") / pl.max_horizontal(pl.col("y_roll12"), pl.lit(1.0))).alias(
            "ratio_origin_roll12"
        ),
        (pl.col("y_origin") / pl.max_horizontal(pl.col("capacidade_sem"), pl.lit(1.0))).alias(
            "occ_origin_cnes"
        ),
    )
    surge_features.extend(["ratio_origin_roll12", "occ_origin_cnes"])

    surge_results: list[dict[str, float | str]] = [
        _score_surge(
            "V1_QuantileInterp_Baseline", y_surge, p_v1_q, pos_mask, shock_mask, int_mask, 0.0
        )
    ]
    best_surge_method = "none"
    best_surge_score = -1e9
    best_surge_report: dict[str, object] = {}

    for method_name in ("none", "platt", "beta", "auto"):
        clf = MonotonicSurgeClassifier(seed=42)
        clf.fit_with_rolling_calibration(
            hist_surge,
            surge_features,
            cal_years=(2019, 2020, 2021, 2022, 2023),
            calibration_method=method_name,
        )
        p_hat = clf.predict_proba(eval_surge)
        mv = clf.audit_monotonicity(eval_surge)
        row = _score_surge(
            f"MonotonicSurge_{method_name}", y_surge, p_hat, pos_mask, shock_mask, int_mask, mv
        )
        surge_results.append(row)
        if method_name in ("platt", "beta"):
            score_val = float(row["bss_pos"]) - 0.5 * float(row["ece_pos"])
            if score_val > best_surge_score and mv == 0.0:
                best_surge_score = score_val
                best_surge_method = method_name
                if clf.calibration_report is not None:
                    best_surge_report = clf.calibration_report.__dict__

    # 2. Ablação Conformal por CNES sobre o Campeão V1: Flood vs. Mondrian vs. AgACI vs. SPCI
    bump_cols = ("chuva_montante_multihop_lag0", "api_max_sem_lag0")
    bumped_eval = eval_df.with_columns(
        [(pl.col(c) + 50.0).alias(c) for c in bump_cols if c in eval_df.columns]
    )

    def _mono_viol_conformal(cal_obj: object, base_q: pl.DataFrame) -> float:
        q_b = cal_obj.predict_quantiles(bumped_eval)  # type: ignore[attr-defined]
        return float(
            np.mean(
                (q_b["pred_p50"].to_numpy() + 1e-6 < base_q["pred_p50"].to_numpy())
                | (q_b["pred_p90"].to_numpy() + 1e-6 < base_q["pred_p90"].to_numpy())
            )
        )

    mondrian_cal = MondrianEnsoConformalCalibrator(base_model=v1_base).fit_calibration(
        cal_df=cal_df
    )
    agaci_cal = AgACIConformalCalibrator(base_model=v1_base, feedback_lag_weeks=2).fit_calibration(
        cal_df=cal_df
    )
    spci_cal = SPCIConformalCalibrator(
        base_model=v1_base,
        residual_lags=4,
        feedback_lag_weeks=2,
        n_estimators=80,
        min_samples_leaf=12,
    ).fit_calibration(cal_df=cal_df)

    q_flood = v1_preds
    q_mondrian = mondrian_cal.predict_quantiles(eval_df)
    q_agaci_frozen = agaci_cal.predict_quantiles(eval_df)
    q_agaci_online = agaci_cal.predict_quantiles_online(eval_df)
    q_spci_frozen = spci_cal.predict_quantiles(eval_df)
    q_spci_online = spci_cal.predict_quantiles_online(eval_df)

    conformal_results = [
        _score_conformal(
            "ConformalFlood_2023",
            eval_df,
            q_flood,
            shock_mask,
            _mono_viol_conformal(v1_conf, q_flood),
        ),
        _score_conformal(
            "MondrianEnso_2023",
            eval_df,
            q_mondrian,
            shock_mask,
            _mono_viol_conformal(mondrian_cal, q_mondrian),
        ),
        _score_conformal(
            "AgACI_CNES_Frozen",
            eval_df,
            q_agaci_frozen,
            shock_mask,
            _mono_viol_conformal(agaci_cal, q_agaci_frozen),
        ),
        _score_conformal(
            "AgACI_CNES_Online_Lag2",
            eval_df,
            q_agaci_online,
            shock_mask,
            _mono_viol_conformal(agaci_cal, q_agaci_frozen),
        ),
        _score_conformal(
            "SPCI_QRF_CNES_Frozen",
            eval_df,
            q_spci_frozen,
            shock_mask,
            _mono_viol_conformal(spci_cal, q_spci_frozen),
        ),
        _score_conformal(
            "SPCI_QRF_CNES_Online_Lag2",
            eval_df,
            q_spci_online,
            shock_mask,
            _mono_viol_conformal(spci_cal, q_spci_frozen),
        ),
    ]

    # Seleciona calibrador conformal vencedor (cobertura [0.80, 0.92] + menor Pinball P95)
    eligible_conf = [
        r
        for r in conformal_results
        if float(r["monotonic_violation_rate"]) == 0.0 and 0.80 <= float(r["cov_p10_p90"]) <= 0.92
    ]
    winner_conf = (
        min(eligible_conf, key=lambda r: float(r["pinball_p95"]))
        if eligible_conf
        else min(conformal_results, key=lambda r: abs(float(r["cov_p10_p90"]) - 0.85))
    )

    summary = {
        "task": "1.2.4",
        "eval_year": 2024,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "surge_ablation": surge_results,
        "recommended_surge_method": best_surge_method,
        "recommended_surge_oof_report": best_surge_report,
        "conformal_ablation": conformal_results,
        "recommended_conformal": winner_conf["calibrator"],
    }
    out_path = DATA_PROCESSED / "rs_v1_2_4_surge_conformal_ablation.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== ABLAÇÃO TASK 1.2.4: CALIBRAÇÃO DE SURTOS (RS 2024) ===")
    print(pl.DataFrame(surge_results))
    print("\n=== ABLAÇÃO TASK 1.2.4: AGACI vs. SPCI NORMALIZADO POR CNES (RS 2024) ===")
    print(pl.DataFrame(conformal_results))
    print(
        f"\nRecomendação Surto: {best_surge_method} | "
        f"Conformal: {winner_conf['calibrator']} | "
        f"Tempo: {summary['elapsed_seconds']}s | Salvo em: {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
