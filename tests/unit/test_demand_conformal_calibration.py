"""Testes Prove-It da Calibração Conformal Heterocedástica (cal_year=2023, zero vazamento)."""

from __future__ import annotations

import numpy as np
import polars as pl

from previsao_leitos.demand import (
    ConformalFloodCalibrator,
    QuantileGBDTForecaster,
    prepare_global_demand_splits,
)
from previsao_leitos.geo import SplitPolicy


def _heteroscedastic_flood_panel() -> pl.DataFrame:
    rng = np.random.default_rng(99)
    rows: list[dict[str, object]] = []
    weeks = [(y, w) for y in range(2019, 2026) for w in range(1, 53)]
    for reg, scale in [("Regiao 10 - Capital", 2.0), ("Regiao 21 - Sul", 1.0)]:
        rain = rng.gamma(2.0, 12.0, size=len(weeks))
        upstream = rain * 1.2
        remanso = upstream * 1.5
        # Choques com variância maior em treino (2020), calibração (2023) e auditoria (2024)
        for s_idx in (70, 225, 278):
            rain[s_idx] = 260.0
            upstream[s_idx] = 350.0
            remanso[s_idx] = 900.0

        y = np.zeros(len(weeks), dtype=np.float64)
        for i, (_, w) in enumerate(weeks):
            up2 = upstream[i - 2] if i >= 2 else 0.0
            noise_sd = (3.0 + 0.08 * up2) * scale
            y[i] = max(
                0.0,
                100.0 * scale
                + 10.0 * np.sin(2.0 * np.pi * w / 52.0)
                + 0.20 * up2
                + rng.normal(0.0, noise_sd),
            )
        for i, (yr, wk) in enumerate(weeks):
            rows.append(
                {
                    "nome_regiao_saude": reg,
                    "ano_epi": yr,
                    "semana_epi": wk,
                    "chuva_sem_mm": float(rain[i]),
                    "chuva_montante_pond": float(upstream[i]),
                    "api_max_sem": float(rain[i] * 0.8),
                    "api_montante_pond": float(upstream[i] * 0.8),
                    "remanso_lagunar_idx": float(remanso[i]),
                    "uti_adulto_diarias": float(round(y[i], 2)),
                }
            )
    return pl.DataFrame(rows)


def test_conformal_calibrator_is_leak_free_monotonic_and_expands_under_flood_stress() -> None:
    df = _heteroscedastic_flood_panel()
    policy = SplitPolicy(train_end_year=2022, cal_year=2023, dev_audit_year=2024, holdout_year=2025)
    dyn = (
        "chuva_sem_mm",
        "chuva_montante_pond",
        "api_max_sem",
        "api_montante_pond",
        "remanso_lagunar_idx",
    )

    sp_cal = prepare_global_demand_splits(
        df,
        entity_col="nome_regiao_saude",
        target_col="uti_adulto_diarias",
        dynamic_feature_cols=dyn,
        horizon_weeks=2,
        lag_weeks=(0, 1, 2),
        eval_year=2023,
        policy=policy,
    )
    sp_eval = prepare_global_demand_splits(
        df,
        entity_col="nome_regiao_saude",
        target_col="uti_adulto_diarias",
        dynamic_feature_cols=dyn,
        horizon_weeks=2,
        lag_weeks=(0, 1, 2),
        eval_year=2024,
        policy=policy,
    )

    base = QuantileGBDTForecaster(
        monotone_positive_prefixes=(
            "chuva_montante_pond",
            "api_montante_pond",
            "remanso_lagunar_idx",
        ),
        max_iter=60,
        random_state=42,
    ).fit(sp_cal.train_df, sp_cal.feature_cols)

    cal = ConformalFloodCalibrator(
        base_model=base,
        stress_cols=("api_max_sem_lag0", "api_montante_pond_lag0", "remanso_lagunar_idx_lag0"),
        stress_gamma=0.5,
    )
    cal.fit_calibration(train_df=sp_cal.train_df, cal_df=sp_cal.eval_df)

    raw_preds = base.predict_quantiles(sp_eval.eval_df)
    cal_preds = cal.predict_quantiles(sp_eval.eval_df)

    # 1. Não-negatividade e não-cruzamento estrito
    assert (cal_preds["pred_p10"] >= 0.0).all()
    assert (cal_preds["pred_p10"] <= cal_preds["pred_p50"] + 1e-9).all()
    assert (cal_preds["pred_p50"] <= cal_preds["pred_p90"] + 1e-9).all()
    assert (cal_preds["pred_p90"] <= cal_preds["pred_p95"] + 1e-9).all()

    # 2. Monotonicidade preservada em P50/P90/P95 sob aumento de estresse hidrológico
    bumped = sp_eval.eval_df.with_columns(
        (pl.col("remanso_lagunar_idx_lag0") + 600.0).alias("remanso_lagunar_idx_lag0"),
        (pl.col("api_montante_pond_lag0") + 150.0).alias("api_montante_pond_lag0"),
    )
    cal_bumped = cal.predict_quantiles(bumped)
    assert (cal_bumped["pred_p50"] >= cal_preds["pred_p50"] - 1e-6).all()
    assert (cal_bumped["pred_p90"] >= cal_preds["pred_p90"] - 1e-6).all()
    assert (cal_bumped["pred_p95"] >= cal_preds["pred_p95"] - 1e-6).all()

    # 3. Banda calibrada em 2023 melhora ou iguala cobertura empírica no ano 2024
    y_true = sp_eval.eval_df["target_y"].to_numpy()
    cov_raw = float(
        np.mean(
            (y_true >= raw_preds["pred_p10"].to_numpy())
            & (y_true <= raw_preds["pred_p90"].to_numpy())
        )
    )
    cov_cal = float(
        np.mean(
            (y_true >= cal_preds["pred_p10"].to_numpy())
            & (y_true <= cal_preds["pred_p90"].to_numpy())
        )
    )
    assert cov_cal >= cov_raw
    assert cov_cal >= 0.72
