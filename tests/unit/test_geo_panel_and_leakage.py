"""Testes Prove-It de vazamento zero, bloqueio do holdout 2025 e baseline Distributed Lag."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from previsao_leitos.domain import LeakageViolationError
from previsao_leitos.geo.panel_builder import (
    SplitPolicy,
    evaluate_distributed_lag_tracer_bullet,
    prepare_causal_weekly_panel,
)


def _synthetic_rs_panel() -> pl.DataFrame:
    """Gera 5 anos (2020..2024) semanais onde demanda em t+2 depende da chuva em t."""
    rng = np.random.default_rng(42)
    rows: list[dict[str, object]] = []
    for regiao, scale in [("Regiao 10 - Capital", 1.5), ("Regiao 17 - Planalto", 0.4)]:
        weeks = [(y, w) for y in range(2020, 2026) for w in range(1, 53)]
        rain = rng.gamma(shape=2.0, scale=15.0, size=len(weeks))
        # Injeta 3 eventos extremos de enchente (1 em treino, 1 em cal, 1 em 2024)
        rain[60] = 280.0
        rain[170] = 260.0
        rain[225] = 350.0
        upstream = rain * (1.2 if "Capital" in regiao else 0.1)

        base = 40.0 * scale
        demand = np.zeros(len(weeks), dtype=np.float64)
        for i in range(len(weeks)):
            lag2_rain = rain[i - 2] if i >= 2 else 0.0
            lag2_up = upstream[i - 2] if i >= 2 else 0.0
            season = 8.0 * np.sin(2.0 * np.pi * (weeks[i][1] / 52.0))
            demand[i] = max(
                0.0,
                base
                + season
                + 0.18 * scale * lag2_rain
                + 0.12 * scale * lag2_up
                + rng.normal(0.0, 1.5),
            )

        for i, (y, w) in enumerate(weeks):
            rows.append(
                {
                    "nome_regiao_saude": regiao,
                    "ano_epi": y,
                    "semana_epi": w,
                    "chuva_sem_mm": float(rain[i]),
                    "chuva_montante_mm": float(upstream[i]),
                    "internacoes": float(round(demand[i], 2)),
                }
            )
    return pl.DataFrame(rows)


def test_prove_it_perturbing_2024_2025_leaves_2010_2022_features_bit_identical() -> None:
    base_df = _synthetic_rs_panel()
    perturbed_df = base_df.with_columns(
        pl.when(pl.col("ano_epi") >= 2024)
        .then(pl.col("chuva_sem_mm") * 999.0 + 12345.0)
        .otherwise(pl.col("chuva_sem_mm"))
        .alias("chuva_sem_mm"),
        pl.when(pl.col("ano_epi") >= 2024)
        .then(pl.col("internacoes") * 50.0)
        .otherwise(pl.col("internacoes"))
        .alias("internacoes"),
    )

    p_base = prepare_causal_weekly_panel(
        base_df,
        entity_col="nome_regiao_saude",
        target_col="internacoes",
        feature_cols=("chuva_sem_mm", "chuva_montante_mm"),
        horizon_weeks=2,
        lag_weeks=(0, 1, 2, 3, 4),
    )
    p_pert = prepare_causal_weekly_panel(
        perturbed_df,
        entity_col="nome_regiao_saude",
        target_col="internacoes",
        feature_cols=("chuva_sem_mm", "chuva_montante_mm"),
        horizon_weeks=2,
        lag_weeks=(0, 1, 2, 3, 4),
    )

    # Todas as linhas cuja DATA-ALVO (target_ano_epi) <= 2023 devem ser bit-idênticas!
    tr_base = p_base.filter(pl.col("target_ano_epi") <= 2023)
    tr_pert = p_pert.filter(pl.col("target_ano_epi") <= 2023)
    assert tr_base.equals(tr_pert)


def test_split_policy_blocks_touching_2025_holdout_during_dev_evaluation() -> None:
    df = _synthetic_rs_panel()
    policy = SplitPolicy(train_end_year=2022, cal_year=2023, dev_audit_year=2024, holdout_year=2025)

    with pytest.raises(LeakageViolationError, match="2025"):
        evaluate_distributed_lag_tracer_bullet(
            df,
            entity_col="nome_regiao_saude",
            target_col="internacoes",
            feature_cols=("chuva_sem_mm", "chuva_montante_mm"),
            horizon_weeks=2,
            eval_year=2025,
            policy=policy,
            allow_holdout=False,
        )


def test_distributed_lag_tracer_bullet_beats_persistence_on_2week_flood_lag() -> None:
    df = _synthetic_rs_panel()
    report = evaluate_distributed_lag_tracer_bullet(
        df,
        entity_col="nome_regiao_saude",
        target_col="internacoes",
        feature_cols=("chuva_sem_mm", "chuva_montante_mm"),
        horizon_weeks=2,
        eval_year=2024,
    )

    assert report.horizon_weeks == 2
    assert report.n_eval_rows > 80
    # Quando a demanda em t+2 responde à chuva em t, Distributed Lag deve bater Persistência
    assert report.model_mae < report.persistence_mae
    assert report.mase_vs_persistence < 1.0
    assert report.r2_delta > 0.2
