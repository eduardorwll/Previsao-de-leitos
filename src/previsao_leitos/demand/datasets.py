"""Construção de splits causais globais multi-região para modelagem de demanda (F04)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from previsao_leitos.domain import DataContractError, LeakageViolationError
from previsao_leitos.geo import SplitPolicy, prepare_causal_weekly_panel


@dataclass(frozen=True)
class GlobalDemandSplits:
    """Partições temporais causais multi-região prontas para treino e avaliação."""

    train_df: pl.DataFrame
    eval_df: pl.DataFrame
    feature_cols: tuple[str, ...]
    entity_col: str
    target_col: str
    horizon_weeks: int
    eval_year: int


def prepare_global_demand_splits(
    weekly_df: pl.DataFrame,
    *,
    entity_col: str,
    target_col: str,
    dynamic_feature_cols: Sequence[str],
    static_feature_cols: Sequence[str] = (),
    horizon_weeks: int = 2,
    lag_weeks: Sequence[int] = (0, 1, 2, 3, 4),
    eval_year: int = 2024,
    policy: SplitPolicy | None = None,
    allow_holdout: bool = False,
) -> GlobalDemandSplits:
    """Constrói painel causal global (todas as regiões) bloqueando o holdout `2025`.

    Adiciona estatísticas autorregressivas estritamente causais (`y_roll4`, `y_roll12`)
    calculadas apenas sobre `y <= t_0` antes de qualquer divisão temporal.
    """
    active_policy = policy or SplitPolicy()
    if eval_year >= active_policy.holdout_year and not allow_holdout:
        raise LeakageViolationError(
            f"Acesso bloqueado ao holdout intocado ({active_policy.holdout_year}): "
            f"eval_year={eval_year} exige `allow_holdout=True` apenas na auditoria final."
        )

    panel = prepare_causal_weekly_panel(
        weekly_df,
        entity_col=entity_col,
        target_col=target_col,
        feature_cols=dynamic_feature_cols,
        horizon_weeks=horizon_weeks,
        lag_weeks=lag_weeks,
    )

    # Estatísticas autorregressivas estritamente causais (<= t_0) para capturar porte regional
    enriched = panel.with_columns(
        pl.col("y_origin")
        .rolling_mean(window_size=4, min_samples=1)
        .over(entity_col)
        .alias("y_roll4"),
        pl.col("y_origin")
        .rolling_mean(window_size=12, min_samples=1)
        .over(entity_col)
        .alias("y_roll12"),
        (
            pl.col("y_origin")
            - pl.col("y_origin").rolling_mean(window_size=12, min_samples=1).over(entity_col)
        ).alias("y_dev_roll12"),
    )

    lag_cols = [f"{col}_lag{k}" for col in dynamic_feature_cols for k in lag_weeks]
    valid_static = [c for c in static_feature_cols if c in enriched.columns]
    feature_cols = tuple(
        [
            *lag_cols,
            *valid_static,
            "y_origin",
            "y_roll4",
            "y_roll12",
            "y_dev_roll12",
            "target_week_sin",
            "target_week_cos",
        ]
    )

    required = [*feature_cols, "target_y", "target_delta", "target_ano_epi", "target_semana_epi"]
    clean = enriched.drop_nulls(subset=required)

    train_df = clean.filter(pl.col("target_ano_epi") <= active_policy.train_end_year)
    eval_df = clean.filter(pl.col("target_ano_epi") == eval_year)

    if train_df.height < 20 or eval_df.height == 0:
        raise DataContractError(
            f"Amostras insuficientes no painel global "
            f"(train={train_df.height}, eval={eval_df.height})."
        )

    return GlobalDemandSplits(
        train_df=train_df,
        eval_df=eval_df,
        feature_cols=feature_cols,
        entity_col=entity_col,
        target_col=target_col,
        horizon_weeks=horizon_weeks,
        eval_year=eval_year,
    )
