"""Painel semanal causal e primeiro baseline Distributed Lag (Tracer Bullet h=1..4 sem.)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from previsao_leitos.domain import DataContractError, LeakageViolationError
from previsao_leitos.geo.calendar_grid import build_ex_ante_distributed_lags


@dataclass(frozen=True)
class SplitPolicy:
    """Política congelada de partições temporais (ADR-0003)."""

    train_end_year: int = 2022
    cal_year: int = 2023
    dev_audit_year: int = 2024
    holdout_year: int = 2025


@dataclass(frozen=True)
class TracerBulletReport:
    """Relatório objetivo comparando Distributed Lag contra Persistência e Sazonal-Ingênuo."""

    horizon_weeks: int
    eval_year: int
    n_train_rows: int
    n_eval_rows: int
    model_mae: float
    persistence_mae: float
    seasonal_naive_mae: float
    mase_vs_persistence: float
    mase_vs_seasonal: float
    r2_delta: float
    by_entity: pl.DataFrame


def prepare_causal_weekly_panel(
    weekly_df: pl.DataFrame,
    *,
    entity_col: str,
    target_col: str,
    feature_cols: Sequence[str],
    horizon_weeks: int = 2,
    lag_weeks: Sequence[int] = (0, 1, 2, 3, 4),
) -> pl.DataFrame:
    """Constrói painel semanal estritamente causal para prever `target_col` em `t + horizon_weeks`.

    Todas as features em uma linha pertencem a instantes <= t_0 (origem da previsão).
    O alvo (`target_y`, `target_delta`) e o carimbo (`target_ano_epi`, `target_semana_epi`)
    correspondem exatamente a `t_0 + horizon_weeks` sobre uma grade semanal ISO sem lacunas.
    """
    if horizon_weeks < 1:
        raise DataContractError(f"horizon_weeks deve ser >= 1 (recebido {horizon_weeks}).")

    lagged = build_ex_ante_distributed_lags(
        weekly_df,
        entity_col=entity_col,
        feature_cols=feature_cols,
        lag_weeks=lag_weeks,
    )

    # Sazonalidade determinística conhecida para a semana-alvo e referência sazonal (52 sem. antes)
    seasonal_shift = 52 - horizon_weeks
    panel = lagged.with_columns(
        pl.col(target_col).alias("y_origin"),
        pl.col(target_col).shift(-horizon_weeks).over(entity_col).alias("target_y"),
        pl.col("ano_epi").shift(-horizon_weeks).over(entity_col).alias("target_ano_epi"),
        pl.col("semana_epi").shift(-horizon_weeks).over(entity_col).alias("target_semana_epi"),
        pl.col(target_col).shift(seasonal_shift).over(entity_col).alias("y_seasonal_naive"),
    ).with_columns(
        (pl.col("target_y") - pl.col("y_origin")).alias("target_delta"),
        (2.0 * np.pi * pl.col("target_semana_epi").cast(pl.Float64) / 52.0)
        .sin()
        .alias("target_week_sin"),
        (2.0 * np.pi * pl.col("target_semana_epi").cast(pl.Float64) / 52.0)
        .cos()
        .alias("target_week_cos"),
    )
    return panel


def _fit_predict_ridge_delta(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    alpha: float = 10.0,
) -> np.ndarray:
    """Ajusta Ridge Regression com padronização estritamente sobre o conjunto de treino."""
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    z_train = (x_train - mean) / std
    z_eval = (x_eval - mean) / std

    ones_tr = np.ones((z_train.shape[0], 1), dtype=np.float64)
    ones_ev = np.ones((z_eval.shape[0], 1), dtype=np.float64)
    a_tr = np.hstack([ones_tr, z_train])
    a_ev = np.hstack([ones_ev, z_eval])

    reg = np.eye(a_tr.shape[1], dtype=np.float64) * alpha
    reg[0, 0] = 0.0  # não penalizar intercepto
    coef = np.linalg.solve(a_tr.T @ a_tr + reg, a_tr.T @ y_train)
    return np.asarray(a_ev @ coef, dtype=np.float64)


def evaluate_distributed_lag_tracer_bullet(
    weekly_df: pl.DataFrame,
    *,
    entity_col: str,
    target_col: str,
    feature_cols: Sequence[str],
    horizon_weeks: int = 2,
    lag_weeks: Sequence[int] = (0, 1, 2, 3, 4),
    eval_year: int = 2024,
    policy: SplitPolicy | None = None,
    allow_holdout: bool = False,
    ridge_alpha: float = 10.0,
) -> TracerBulletReport:
    """Treina Distributed Lag causal em `<= train_end_year` e avalia em `eval_year`."""
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
        feature_cols=feature_cols,
        horizon_weeks=horizon_weeks,
        lag_weeks=lag_weeks,
    )

    model_feature_cols = [
        *(f"{col}_lag{k}" for col in feature_cols for k in lag_weeks),
        "y_origin",
        "target_week_sin",
        "target_week_cos",
    ]
    required = [*model_feature_cols, "target_y", "target_delta", "target_ano_epi"]
    clean = panel.drop_nulls(subset=required)

    # Split rigoroso: treino termina no máximo quando target_ano_epi <= train_end_year
    train_df = clean.filter(pl.col("target_ano_epi") <= active_policy.train_end_year)
    eval_df = clean.filter(pl.col("target_ano_epi") == eval_year)

    if train_df.height < 10 or eval_df.height == 0:
        raise DataContractError(
            f"Amostras insuficientes para avaliação "
            f"(train={train_df.height}, eval={eval_df.height})."
        )

    entities = sorted(train_df[entity_col].unique().to_list())
    entity_preds: list[pl.DataFrame] = []
    for ent in entities:
        tr_e = train_df.filter(pl.col(entity_col) == ent)
        ev_e = eval_df.filter(pl.col(entity_col) == ent)
        if tr_e.height < 10 or ev_e.height == 0:
            continue

        x_tr = tr_e.select(model_feature_cols).to_numpy().astype(np.float64)
        y_tr = tr_e["target_delta"].to_numpy().astype(np.float64)
        x_ev = ev_e.select(model_feature_cols).to_numpy().astype(np.float64)

        pred_delta = _fit_predict_ridge_delta(x_tr, y_tr, x_ev, alpha=ridge_alpha)
        y_orig = ev_e["y_origin"].to_numpy().astype(np.float64)
        pred_y = np.maximum(0.0, y_orig + pred_delta)

        entity_preds.append(
            ev_e.select(
                [
                    entity_col,
                    "target_ano_epi",
                    "target_semana_epi",
                    "y_origin",
                    "y_seasonal_naive",
                    "target_y",
                    "target_delta",
                ]
            ).with_columns(
                pl.Series("pred_delta", pred_delta, dtype=pl.Float64),
                pl.Series("pred_y", pred_y, dtype=pl.Float64),
            )
        )

    if not entity_preds:
        raise DataContractError(
            "Nenhuma entidade com histórico suficiente para treino e avaliação."
        )

    scored = pl.concat(entity_preds)
    y_true = scored["target_y"].to_numpy()
    y_pred = scored["pred_y"].to_numpy()
    y_pers = scored["y_origin"].to_numpy()
    d_true = scored["target_delta"].to_numpy()
    d_pred = y_pred - y_pers

    model_mae = float(np.mean(np.abs(y_true - y_pred)))
    persistence_mae = float(np.mean(np.abs(y_true - y_pers)))
    mase_pers = model_mae / persistence_mae if persistence_mae > 1e-12 else float("inf")

    seasonal_valid = scored.drop_nulls(subset=["y_seasonal_naive"])
    if seasonal_valid.height > 0:
        s_true = seasonal_valid["target_y"].to_numpy()
        s_pred = seasonal_valid["y_seasonal_naive"].to_numpy()
        s_model_pred = seasonal_valid["pred_y"].to_numpy()
        seasonal_mae = float(np.mean(np.abs(s_true - s_pred)))
        s_model_mae = float(np.mean(np.abs(s_true - s_model_pred)))
        mase_seas = s_model_mae / seasonal_mae if seasonal_mae > 1e-12 else float("inf")
    else:
        seasonal_mae = float("nan")
        mase_seas = float("nan")

    ss_res = float(np.sum((d_true - d_pred) ** 2))
    ss_tot = float(np.sum((d_true - np.mean(d_true)) ** 2))
    r2_delta = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    by_entity = (
        scored.group_by(entity_col)
        .agg(
            pl.len().alias("semanas"),
            (pl.col("target_y") - pl.col("pred_y")).abs().mean().alias("model_mae"),
            (pl.col("target_y") - pl.col("y_origin")).abs().mean().alias("persistence_mae"),
            pl.col("target_y").mean().alias("media_real"),
        )
        .with_columns(
            (pl.col("model_mae") / pl.col("persistence_mae")).alias("mase_vs_persistence")
        )
        .sort("mase_vs_persistence")
    )

    return TracerBulletReport(
        horizon_weeks=horizon_weeks,
        eval_year=eval_year,
        n_train_rows=train_df.height,
        n_eval_rows=scored.height,
        model_mae=model_mae,
        persistence_mae=persistence_mae,
        seasonal_naive_mae=seasonal_mae,
        mase_vs_persistence=mase_pers,
        mase_vs_seasonal=mase_seas,
        r2_delta=r2_delta,
        by_entity=by_entity,
    )
