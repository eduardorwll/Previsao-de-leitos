"""Testes unitários da Task 3.2: arquitetura ortogonal 2 estágios e roteador esparso."""

from __future__ import annotations

import numpy as np
import polars as pl

from previsao_leitos.demand.models import NegativeBinomialHurdleForecaster
from previsao_leitos.demand.two_stage import (
    OrthogonalTwoStageOccupancyForecaster,
    select_demand_forecaster,
)


def _make_synthetic_panel() -> pl.DataFrame:
    rng = np.random.default_rng(123)
    rows = []
    # Duas regiões de portes muito distintos: Capital (cap=3000) e Interior (cap=100)
    for ent, cap in (("Capital", 3000.0), ("Interior", 100.0)):
        for yr in (2020, 2021, 2022, 2024):
            for wk in range(1, 31):
                prob_f03 = (
                    0.85 if (yr == 2024 and wk in (19, 20)) else float(rng.uniform(0.02, 0.25))
                )
                if (yr == 2020 and wk in (8, 9)) or (yr == 2021 and wk in (15, 16)) or (
                    yr == 2022 and wk in (23, 24)
                ):
                    prob_f03 = 0.80
                sev_f03 = prob_f03 * 2.0
                season = 0.72 + 0.05 * np.sin(2.0 * np.pi * wk / 52.0)
                shock_rate = 0.28 * prob_f03 + 0.04 * sev_f03
                y_val = cap * (season + shock_rate + float(rng.normal(0.0, 0.015)))
                rows.append(
                    {
                        "nome_regiao_saude": ent,
                        "target_ano_epi": yr,
                        "target_semana_epi": wk,
                        "capacidade_sem": cap,
                        "y_origin": cap * season,
                        "y_roll12": cap * 0.72,
                        "sin_target_wk": float(np.sin(2.0 * np.pi * wk / 52.0)),
                        "cos_target_wk": float(np.cos(2.0 * np.pi * wk / 52.0)),
                        "prob_enchente_f03_oof": prob_f03,
                        "severidade_hidro_f03_oof": sev_f03,
                        "chuva_montante_pond_lag0": 20.0 + 300.0 * prob_f03,
                        "target_y": max(0.0, y_val),
                    }
                )
    return pl.DataFrame(rows)


def test_two_stage_orthogonal_tracks_peak_without_phase_lag_and_is_monotonic() -> None:
    df = _make_synthetic_panel()
    train_df = df.filter(pl.col("target_ano_epi") <= 2022)
    eval_df = df.filter(pl.col("target_ano_epi") == 2024)

    feature_cols = (
        "sin_target_wk",
        "cos_target_wk",
        "prob_enchente_f03_oof",
        "severidade_hidro_f03_oof",
        "chuva_montante_pond_lag0",
    )
    model = OrthogonalTwoStageOccupancyForecaster(
        capacity_col="capacidade_sem",
        basal_cols=("sin_target_wk", "cos_target_wk"),
        monotone_positive_prefixes=(
            "prob_enchente_f03",
            "severidade_hidro_f03",
            "chuva_montante_pond",
        ),
        random_state=42,
    ).fit(train_df, feature_cols)

    preds = model.predict_quantiles(eval_df)
    # Não-cruzamento de quantis
    assert np.all(preds["pred_p10"].to_numpy() <= preds["pred_p50"].to_numpy() + 1e-9)
    assert np.all(preds["pred_p50"].to_numpy() <= preds["pred_p90"].to_numpy() + 1e-9)
    assert np.all(preds["pred_p90"].to_numpy() <= preds["pred_p95"].to_numpy() + 1e-9)

    # Monotonicidade exata: aumentar prob_enchente_f03_oof nunca reduz pred_p50
    bumped = eval_df.with_columns(
        (pl.col("prob_enchente_f03_oof") + 0.15)
        .clip(upper_bound=1.0)
        .alias("prob_enchente_f03_oof")
    )
    preds_bumped = model.predict_quantiles(bumped)
    assert np.all(preds_bumped["pred_p50"].to_numpy() >= preds["pred_p50"].to_numpy() - 1e-6)

    # Timing de pico sem atraso de fase nas duas escalas (Capital e Interior)
    eval_with_pred = eval_df.with_columns(preds["pred_p50"])
    for ent in ("Capital", "Interior"):
        sub = eval_with_pred.filter(pl.col("nome_regiao_saude") == ent)
        wk_obs = int(sub.sort("target_y", descending=True)["target_semana_epi"][0])
        wk_pred = int(sub.sort("pred_p50", descending=True)["target_semana_epi"][0])
        assert abs(wk_pred - wk_obs) <= 1


def test_select_demand_forecaster_routes_sparse_counts_to_negbin_hurdle() -> None:
    sparse_df = pl.DataFrame({"target_y": [0.0] * 80 + [1.0, 2.0, 5.0] * 6})
    dense_df = pl.DataFrame({"target_y": [50.0 + i for i in range(100)]})

    m_sparse = select_demand_forecaster(sparse_df)
    m_dense = select_demand_forecaster(dense_df)

    assert isinstance(m_sparse, NegativeBinomialHurdleForecaster)
    assert isinstance(m_dense, OrthogonalTwoStageOccupancyForecaster)
