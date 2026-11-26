"""Testes unitários e Prove-It da Task 1.2.4: MonotonicSurgeClassifier, Platt/Beta, AgACI e SPCI."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from previsao_leitos.demand import (
    AgACIConformalCalibrator,
    MonotonicSurgeClassifier,
    QuantileGBDTForecaster,
    RollingBetaCalibrator,
    RollingPlattCalibrator,
    SPCIConformalCalibrator,
    build_surge_target,
    compute_cnes_capacity_scale,
)
from previsao_leitos.domain import DataContractError


def _synthetic_surge_panel(n_years: int = 8) -> pl.DataFrame:
    rng = np.random.default_rng(123)
    rows: list[dict[str, object]] = []
    years = list(range(2017, 2017 + n_years))  # 2017..2024
    regions = [("Reg_POA", 1500.0), ("Reg_Sul", 400.0), ("Reg_Zero", 0.0)]
    for reg, cap in regions:
        for yr in years:
            for wk in range(1, 31):
                rain = float(rng.gamma(shape=1.5, scale=22.0))
                if yr == 2024 and 18 <= wk <= 22:
                    rain += 120.0
                api = 0.6 * rain + float(rng.normal(0.0, 5.0))
                el_nino = 1.5 if yr in (2023, 2024) else 0.0
                base = 0.72 * cap
                if cap <= 0.0:
                    y_val = 0.0
                else:
                    surge_signal = 0.0025 * rain + 0.04 * el_nino + float(rng.normal(0.0, 0.08))
                    y_val = max(0.0, base * (1.0 + surge_signal))
                rows.append(
                    {
                        "nome_regiao_saude": reg,
                        "target_ano_epi": yr,
                        "target_semana_epi": wk,
                        "capacidade_sem": cap,
                        "y_roll12": base,
                        "y_origin": base,
                        "chuva_sem_mm_lag0": rain,
                        "api_max_sem_lag0": max(0.0, api),
                        "el_nino_intensity_lag0": el_nino,
                        "target_y": y_val,
                        "target_delta": y_val - base,
                    }
                )
    return pl.DataFrame(rows)


def test_platt_and_beta_calibrators_are_strictly_non_decreasing() -> None:
    rng = np.random.default_rng(7)
    raw = rng.uniform(0.05, 0.95, size=200)
    true_p = 1.0 / (1.0 + np.exp(-(3.2 * np.log(raw / (1.0 - raw)) - 1.1)))
    y = (rng.uniform(0.0, 1.0, size=200) < true_p).astype(np.float64)

    platt = RollingPlattCalibrator().fit(raw, y)
    beta = RollingBetaCalibrator().fit(raw, y)
    assert platt.params[0] >= 0.0
    assert beta.params[0] >= 0.0 and beta.params[1] >= 0.0

    grid = np.linspace(0.01, 0.99, 99)
    p_platt = platt.transform(grid)
    p_beta = beta.transform(grid)
    assert np.all(np.diff(p_platt) >= -1e-12)
    assert np.all(np.diff(p_beta) >= -1e-12)


def test_monotonic_surge_classifier_zero_violations_anti_g8_and_causal_prove_it() -> None:
    df = _synthetic_surge_panel()
    feats = ["chuva_sem_mm_lag0", "api_max_sem_lag0", "el_nino_intensity_lag0", "y_roll12"]

    clf_a = MonotonicSurgeClassifier(max_iter=60, min_samples_leaf=10, seed=42)
    clf_a.fit_with_rolling_calibration(
        df, feats, cal_years=(2019, 2020, 2021, 2022, 2023), calibration_method="auto"
    )
    rep = clf_a.calibration_report
    assert rep is not None
    assert rep.selected_method in ("platt", "beta")
    assert rep.n_oof_samples > 0

    eval_24 = df.filter(pl.col("target_ano_epi") == 2024)
    p_a = clf_a.predict_proba(eval_24)
    assert clf_a.audit_monotonicity(eval_24) == 0.0

    # Anti-G8: região com y_roll12 == 0 deve sempre receber probabilidade 0.0
    zero_mask = eval_24["y_roll12"].to_numpy() <= 1e-9
    assert np.any(zero_mask)
    assert np.all(p_a[zero_mask] == 0.0)

    # Prove-It anti-vazamento: corromper target_y de 2024 NÃO altera nada ajustado em <= 2023
    df_corrupted = df.with_columns(
        pl.when(pl.col("target_ano_epi") == 2024)
        .then(pl.lit(99999.0))
        .otherwise(pl.col("target_y"))
        .alias("target_y")
    )
    clf_b = MonotonicSurgeClassifier(max_iter=60, min_samples_leaf=10, seed=42)
    clf_b.fit_with_rolling_calibration(
        df_corrupted, feats, cal_years=(2019, 2020, 2021, 2022, 2023), calibration_method="auto"
    )
    p_b = clf_b.predict_proba(eval_24)
    np.testing.assert_allclose(p_a, p_b, atol=1e-12)

    # build_surge_target suporta modo capacity / either sem violar anti-G8
    y_either, valid_either = build_surge_target(eval_24, surge_definition="either")
    assert np.all(y_either[~valid_either] == 0.0)


def test_agaci_and_spci_cnes_normalized_causal_lag_and_monotonicity() -> None:
    df = _synthetic_surge_panel().filter(pl.col("capacidade_sem") > 0.0)
    feats = [
        "chuva_sem_mm_lag0",
        "api_max_sem_lag0",
        "el_nino_intensity_lag0",
        "y_roll12",
        "y_origin",
    ]
    train_df = df.filter(pl.col("target_ano_epi") <= 2022)
    cal_df = df.filter(pl.col("target_ano_epi") == 2023)
    eval_df = df.filter(pl.col("target_ano_epi") == 2024)

    base = QuantileGBDTForecaster(
        monotone_positive_prefixes=(
            "chuva_sem_mm_lag0",
            "api_max_sem_lag0",
            "el_nino_intensity_lag0",
        ),
        max_iter=60,
        random_state=42,
    ).fit(train_df, feats)

    agaci = AgACIConformalCalibrator(base_model=base, feedback_lag_weeks=2).fit_calibration(
        cal_df=cal_df
    )
    spci = SPCIConformalCalibrator(
        base_model=base, residual_lags=3, feedback_lag_weeks=2, n_estimators=25, min_samples_leaf=5
    ).fit_calibration(cal_df=cal_df)

    scale = compute_cnes_capacity_scale(eval_df)
    assert np.all(scale >= 1.0)

    for cal in (agaci, spci):
        frozen_q = cal.predict_quantiles(eval_df)
        online_q = cal.predict_quantiles_online(eval_df)
        for q_df in (frozen_q, online_q):
            p10 = q_df["pred_p10"].to_numpy()
            p50 = q_df["pred_p50"].to_numpy()
            p90 = q_df["pred_p90"].to_numpy()
            p95 = q_df["pred_p95"].to_numpy()
            assert np.all(p10 >= 0.0)
            assert np.all((p10 <= p50 + 1e-9) & (p50 <= p90 + 1e-9) & (p90 <= p95 + 1e-9))

        # Monotonicidade hidrológica preservada (0.00% violação em P50/P90/P95)
        bumped = eval_df.with_columns(
            (pl.col("chuva_sem_mm_lag0") + 80.0).alias("chuva_sem_mm_lag0"),
            (pl.col("api_max_sem_lag0") + 40.0).alias("api_max_sem_lag0"),
        )
        q_bump = cal.predict_quantiles(bumped)
        assert (
            float(np.mean(q_bump["pred_p90"].to_numpy() + 1e-9 < frozen_q["pred_p90"].to_numpy()))
            == 0.0
        )

        # Prove-It causalidade de atraso h=2: alterar target_y na semana >= 10 NÃO muda semana <= 11
        eval_mut = eval_df.with_columns(
            pl.when(pl.col("target_semana_epi") >= 10)
            .then(pl.lit(50000.0))
            .otherwise(pl.col("target_y"))
            .alias("target_y")
        )
        online_mut = cal.predict_quantiles_online(eval_mut)
        early_mask = eval_df["target_semana_epi"].to_numpy() <= 11
        np.testing.assert_allclose(
            online_q["pred_p90"].to_numpy()[early_mask],
            online_mut["pred_p90"].to_numpy()[early_mask],
            atol=1e-12,
        )


def test_calibrators_raise_data_contract_error_when_unfitted() -> None:
    with pytest.raises(DataContractError):
        RollingPlattCalibrator().transform(np.array([0.5]))
    with pytest.raises(DataContractError):
        RollingBetaCalibrator().transform(np.array([0.5]))
