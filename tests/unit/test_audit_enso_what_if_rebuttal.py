"""Testes unitários da Task 4.2: oráculo what-if El Niño e rebate contrafactual duplo 2024."""

from __future__ import annotations

import numpy as np
import polars as pl

from previsao_leitos.audit.enso_what_if import (
    DEFAULT_ENSO_SCENARIOS,
    run_dual_counterfactual_rebuttal_2024,
    run_enso_what_if_oracle,
)
from previsao_leitos.demand.two_stage import OrthogonalTwoStageOccupancyForecaster


def _make_rebuttal_dataset() -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(2024)
    rows = []
    for yr in (2020, 2021, 2022, 2023, 2024):
        for wk in range(1, 31):
            is_shock = (
                (yr == 2020 and wk in (8, 9))
                or (yr == 2021 and wk in (14, 15))
                or (yr == 2022 and wk in (22, 23))
                or (yr == 2024 and wk in (19, 20))
            )
            oni = 2.0 if yr in (2023, 2024) else 0.0
            prob_f03 = 0.85 if is_shock else float(rng.uniform(0.05, 0.20))
            rain = 280.0 if is_shock else float(rng.uniform(15.0, 50.0))
            cap = 1000.0
            base_rate = 0.70 + 0.04 * np.sin(2.0 * np.pi * wk / 52.0)
            shock_rate = 0.25 * prob_f03 + 0.03 * max(0.0, oni)
            y = cap * (base_rate + shock_rate + float(rng.normal(0.0, 0.01)))
            rows.append(
                {
                    "nome_regiao_saude": "Região 10 - Capital",
                    "ano_epi": yr,
                    "semana_epi": wk,
                    "target_ano_epi": yr,
                    "target_semana_epi": wk,
                    "capacidade_sem": cap,
                    "y_origin": cap * base_rate,
                    "sin_target_wk": float(np.sin(2.0 * np.pi * wk / 52.0)),
                    "cos_target_wk": float(np.cos(2.0 * np.pi * wk / 52.0)),
                    "prob_enchente_f03_oof": prob_f03,
                    "chuva_montante_pond_lag0": rain,
                    "el_nino_intensity": max(0.0, oni),
                    "srag_viral_confirmado": 10.0,
                    "target_y": max(0.0, y),
                }
            )
    full = pl.DataFrame(rows)
    return full.filter(pl.col("ano_epi") <= 2022), full


def test_enso_what_if_oracle_is_monotone_across_el_nino_regimes() -> None:
    train_df, full_df = _make_rebuttal_dataset()
    eval_2024 = full_df.filter(pl.col("ano_epi") == 2024)
    feature_cols = (
        "sin_target_wk",
        "cos_target_wk",
        "prob_enchente_f03_oof",
        "chuva_montante_pond_lag0",
        "el_nino_intensity",
    )
    model = OrthogonalTwoStageOccupancyForecaster(
        capacity_col="capacidade_sem",
        basal_cols=("sin_target_wk", "cos_target_wk"),
        min_samples_leaf=5,
        random_state=42,
    ).fit(train_df, feature_cols)

    oracle_df = run_enso_what_if_oracle(
        model=model,
        baseline_eval_df=eval_2024,
        scenarios=DEFAULT_ENSO_SCENARIOS,
    )
    assert oracle_df.height == len(DEFAULT_ENSO_SCENARIOS)
    p50_vals = oracle_df["demanda_p50_media"].to_list()
    # Monotonicidade entre cenários: neutro <= moderado <= forte <= super El Niño
    for i in range(len(p50_vals) - 1):
        assert p50_vals[i] <= p50_vals[i + 1] + 1e-6


def test_dual_counterfactual_rebuttal_2024_integrates_bsts_and_recursive_do() -> None:
    train_df, full_df = _make_rebuttal_dataset()
    eval_2024 = full_df.filter(pl.col("ano_epi") == 2024)
    feature_cols = (
        "sin_target_wk",
        "cos_target_wk",
        "prob_enchente_f03_oof",
        "chuva_montante_pond_lag0",
        "el_nino_intensity",
    )
    model = OrthogonalTwoStageOccupancyForecaster(
        capacity_col="capacidade_sem",
        basal_cols=("sin_target_wk", "cos_target_wk"),
        min_samples_leaf=5,
        random_state=42,
    ).fit(train_df, feature_cols)

    rebuttal = run_dual_counterfactual_rebuttal_2024(
        model=model,
        full_panel=full_df,
        train_df=train_df,
        eval_df=eval_2024,
        entity_col="nome_regiao_saude",
        target_col="target_y",
        climate_prefixes=("prob_enchente_f03", "chuva_montante_pond"),
        neutral_zero_prefixes=("el_nino_intensity",),
        shock_weeks=(18, 22),
    )
    assert rebuttal.shock_excess_obs_mean > 50.0
    assert rebuttal.shock_excess_pred_mean > 50.0
    assert rebuttal.peak_timing_error_weeks <= 1.0
    assert not rebuttal.summary_table.is_empty()
