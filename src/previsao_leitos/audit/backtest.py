"""Backtest de origem móvel: régua de avaliação entre folds.

Substitui o julgamento por um único ano por uma sequência de folds causais
(`train <= y-2`, calibração `y-1`, avaliação `y`), com métricas robustas a
heterogeneidade de porte e a eventos raros de enchente.

Decisões de desenho:
- ranking por **MASE mediano entre folds** (não pela média, que é sensível a um fold);
- métricas *pooled* e *macro* reportadas em paralelo (30 regiões não têm o mesmo peso);
- métrica **evento-condicional** (semanas com choque alto), porque enchente é rara;
- erro de **fase** (pico com sinal) separado de erro de **nível**.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from previsao_leitos.audit.metrics import pinball_loss
from previsao_leitos.audit.model_benchmark import run_benchmark
from previsao_leitos.demand.modeling import ModelingSplits, make_modeling_splits
from previsao_leitos.domain import DataContractError
from previsao_leitos.geo import SplitPolicy

_SHOCK_COL = "prob_enchente_f03_oof"
_EVENT_THRESHOLD = 0.50
_MIN_REGION_ROWS = 5


@dataclass(frozen=True)
class FoldMetrics:
    """Métricas de um modelo em um fold."""

    eval_year: int
    candidate: str
    status: str
    mase_pooled: float
    mase_macro: float
    mae_pooled: float
    mae_macro: float
    coverage_p10_p90: float
    pinball_mean: float
    pinball_p95: float
    peak_signed_weeks: float
    peak_abs_weeks: float
    shape_corr: float
    bias_relative: float
    event_mae: float
    event_share: float
    regions_used: int
    monotonic_violations: int
    fit_seconds: float


@dataclass(frozen=True)
class BacktestResult:
    """Resultado consolidado de todos os folds."""

    target_key: str
    eval_years: tuple[int, ...]
    fold_metrics: pl.DataFrame
    summary: pl.DataFrame
    predictions: pl.DataFrame
    candidates: tuple[str, ...]


def _region_stats(
    eval_df: pl.DataFrame, preds: pl.DataFrame
) -> tuple[float, float, int]:
    """MAE macro, MASE macro e nº de regiões usadas.

    Regiões com persistência quase perfeita (MAE < 1 leito-dia/semana) tornam o
    MASE macro instável — uma série quase constante gera razão explosiva. Elas são
    excluídas do *macro* e contabilizadas em `regions_used`; o MASE *pooled*
    continua usando todas as linhas.
    """
    frame = eval_df.select(
        [
            pl.col("nome_regiao_saude").alias("entity"),
            pl.col("target_y").alias("observed"),
            pl.col("y_origin").alias("persistence"),
        ]
    ).with_columns(pl.Series("pred_p50", preds["pred_p50"].to_numpy()))
    by_region = frame.group_by("entity").agg(
        pl.len().alias("n"),
        pl.col("observed").mean().alias("obs_mean"),
        (pl.col("pred_p50") - pl.col("observed")).abs().mean().alias("mae"),
        (pl.col("persistence") - pl.col("observed")).abs().mean().alias("mae_pers"),
    )
    valid = by_region.filter(
        (pl.col("n") >= _MIN_REGION_ROWS) & (pl.col("mae_pers") >= 1.0)
    )
    if valid.height == 0:
        return float("nan"), float("nan"), 0
    ratios = valid["mae"].to_numpy() / valid["mae_pers"].to_numpy()
    return (
        float(np.mean(valid["mae"].to_numpy())),
        float(np.mean(ratios)),
        int(valid.height),
    )


def _peak_errors(
    eval_df: pl.DataFrame, preds: pl.DataFrame, *, entity_col: str
) -> tuple[float, float]:
    """Erro de pico com sinal (predito - observado) e em módulo, em semanas."""
    frame = eval_df.select(
        [
            pl.col(entity_col),
            pl.col("target_semana_epi"),
            pl.col("target_y"),
        ]
    ).with_columns(pl.Series("pred_p50", preds["pred_p50"].to_numpy()))
    signed: list[float] = []
    for _, group in frame.group_by(entity_col):
        if group.height < 2:
            continue
        obs_idx = int(np.asarray(group["target_y"].arg_max()).item())
        pred_idx = int(np.asarray(group["pred_p50"].arg_max()).item())
        obs_week = float(np.asarray(group["target_semana_epi"][obs_idx]).item())
        pred_week = float(np.asarray(group["target_semana_epi"][pred_idx]).item())
        signed.append(pred_week - obs_week)
    if not signed:
        return float("nan"), float("nan")
    arr = np.asarray(signed, dtype=np.float64)
    return float(np.mean(arr)), float(np.mean(np.abs(arr)))


def _shape_correlation(eval_df: pl.DataFrame, preds: pl.DataFrame) -> float:
    """Correlação entre as séries semanais agregadas (forma da curva)."""
    frame = eval_df.select(
        [pl.col("target_semana_epi").alias("week"), pl.col("target_y").alias("observed")]
    ).with_columns(pl.Series("pred_p50", preds["pred_p50"].to_numpy()))
    agg = frame.group_by("week").agg(
        pl.col("observed").sum(),
        pl.col("pred_p50").sum(),
    ).sort("week")
    if agg.height < 3:
        return float("nan")
    obs = agg["observed"].to_numpy()
    pred = agg["pred_p50"].to_numpy()
    if np.std(obs) < 1e-9 or np.std(pred) < 1e-9:
        return float("nan")
    return float(np.corrcoef(obs, pred)[0, 1])


def _event_mae(
    eval_df: pl.DataFrame, preds: pl.DataFrame, *, threshold: float
) -> tuple[float, float]:
    """MAE nas semanas de choque alto e fração de linhas que são evento."""
    if _SHOCK_COL not in eval_df.columns:
        return float("nan"), 0.0
    shock = eval_df[_SHOCK_COL].fill_null(0.0).to_numpy()
    mask = shock >= threshold
    share = float(np.mean(mask)) if mask.size else 0.0
    if mask.sum() == 0:
        return float("nan"), share
    observed = eval_df["target_y"].to_numpy()
    predicted = preds["pred_p50"].to_numpy()
    return float(np.mean(np.abs(predicted[mask] - observed[mask]))), share


def _pinball_mean(eval_df: pl.DataFrame, preds: pl.DataFrame) -> float:
    y = eval_df["target_y"].to_numpy()
    levels = {"pred_p10": 0.10, "pred_p50": 0.50, "pred_p90": 0.90, "pred_p95": 0.95}
    return float(
        np.mean(
            [pinball_loss(y, preds[col].to_numpy(), alpha=alpha) for col, alpha in levels.items()]
        )
    )


def evaluate_fold(
    splits: ModelingSplits,
    result: Any,
    *,
    run_id: str,
) -> pl.DataFrame:
    """Converte a saída do benchmark de um fold em métricas da régua."""
    rows: list[dict[str, object]] = []
    for row in result.metrics.to_dicts():
        candidate = str(row["candidate"])
        if row.get("status") != "ok":
            rows.append(
                {
                    "eval_year": splits.eval_year,
                    "candidate": candidate,
                    "status": str(row.get("status")),
                }
            )
            continue
        preds = result.predictions.filter(pl.col("model") == candidate).sort(
            ["entity", "target_week"]
        )
        ranked = splits.eval_df.sort(["nome_regiao_saude", "target_semana_epi"])
        if preds.height != ranked.height:
            continue
        mae_macro, mase_macro, regions_used = _region_stats(ranked, preds)
        peak_signed, peak_abs = _peak_errors(ranked, preds, entity_col="nome_regiao_saude")
        event_mae, event_share = _event_mae(ranked, preds, threshold=_EVENT_THRESHOLD)
        observed_mean = float(np.mean(ranked["target_y"].to_numpy()))
        bias = float(np.mean(preds["pred_p50"].to_numpy() - ranked["target_y"].to_numpy()))
        rows.append(
            {
                "eval_year": splits.eval_year,
                "candidate": candidate,
                "status": "ok",
                "mase_pooled": float(row["mase_vs_pers"]),
                "mase_macro": mase_macro,
                "mae_pooled": float(row["mae_p50"]),
                "mae_macro": mae_macro,
                "coverage_p10_p90": float(row["cobertura_p10_p90"]),
                "pinball_mean": _pinball_mean(ranked, preds),
                "pinball_p95": float(row["pinball_p95"]),
                "peak_signed_weeks": peak_signed,
                "peak_abs_weeks": peak_abs,
                "shape_corr": _shape_correlation(ranked, preds),
                "bias_relative": bias / max(observed_mean, 1e-9),
                "event_mae": event_mae,
                "event_share": event_share,
                "regions_used": regions_used,
                "monotonic_violations": int(row.get("monotonic_violations") or 0),
                "wmape": float(row.get("wmape") or float("nan")),
                "erro_pct_mediana": float(row.get("erro_pct_mediana") or float("nan")),
                "erro_pct_p90": float(row.get("erro_pct_p90") or float("nan")),
                "subestimacao_pct": float(row.get("subestimacao_pct") or float("nan")),
                "fit_seconds": float(row.get("fit_seconds") or 0.0),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def summarize_backtest(fold_metrics: pl.DataFrame) -> pl.DataFrame:
    """Ranking por MASE mediano, com dispersão entre folds e nº de vitórias."""
    ok = fold_metrics.filter(pl.col("status") == "ok")
    if ok.height == 0:
        raise DataContractError("Backtest sem folds válidos.")
    per_fold_rank = ok.with_columns(
        pl.col("mase_macro").rank("ordinal").over("eval_year").alias("rank_fold")
    )
    summary = per_fold_rank.group_by("candidate").agg(
        pl.col("mase_macro").median().alias("mase_median"),
        pl.col("mase_macro").mean().alias("mase_mean"),
        pl.col("mase_macro").max().alias("mase_worst"),
        pl.col("mase_macro").min().alias("mase_best"),
        pl.col("rank_fold").median().alias("rank_median"),
        (pl.col("rank_fold") == 1).sum().alias("fold_wins"),
        pl.len().alias("n_folds"),
        pl.col("coverage_p10_p90").median().alias("coverage_median"),
        pl.col("peak_abs_weeks").median().alias("peak_abs_median"),
        pl.col("monotonic_violations").sum().alias("monotonic_violations"),
        pl.col("fit_seconds").median().alias("fit_seconds_median"),
        pl.col("wmape").median().alias("wmape_median"),
        pl.col("erro_pct_mediana").median().alias("erro_pct_mediana"),
        pl.col("erro_pct_p90").median().alias("erro_pct_p90"),
        pl.col("subestimacao_pct").median().alias("subestimacao_pct"),
        (pl.col("mase_macro").std() / pl.col("mase_macro").mean())
        .fill_null(0.0)
        .alias("mase_cv"),
    ).sort("mase_median")
    return summary


def run_rolling_backtest(
    panel: pl.DataFrame,
    *,
    target: str,
    eval_years: Sequence[int] = (2019, 2020, 2021, 2022, 2023, 2024),
    candidates: str | Sequence[str] = "all",
    n_boot: int = 50,
    seed: int = 42,
    allow_holdout: bool = False,
    shock_memory: bool = False,
    admissions_momentum: bool = False,
    dengue_features: bool = False,
    roni_features: bool = False,
    soi_features: bool = False,
    bom_features: bool = False,
    temperature_features: bool = False,
    run_id: str = "backtest",
) -> BacktestResult:
    """Roda o benchmark em cada fold e consolida a régua."""
    if any(year >= 2025 for year in eval_years) and not allow_holdout:
        raise DataContractError("Backtest não pode incluir anos >= 2025 (holdout do M2).")

    fold_frames: list[pl.DataFrame] = []
    prediction_frames: list[pl.DataFrame] = []
    for year in eval_years:
        policy = SplitPolicy(
            train_end_year=year - 2,
            cal_year=year - 1,
            dev_audit_year=year,
            holdout_year=2025,
        )
        splits = make_modeling_splits(
            panel,
            target=target,
            eval_year=year,
            policy=policy,
            allow_holdout=allow_holdout,
            shock_memory=shock_memory,
            admissions_momentum=admissions_momentum,
            dengue_features=dengue_features,
            roni_features=roni_features,
            soi_features=soi_features,
            bom_features=bom_features,
            temperature_features=temperature_features,
        )
        result = run_benchmark(
            splits,
            run_id=f"{run_id}:{target}:{year}",
            candidates=candidates,
            n_boot=n_boot,
            seed=seed,
            include_ratchet=False,
        )
        fold_frames.append(evaluate_fold(splits, result, run_id=run_id))
        prediction_frames.append(
            result.predictions.with_columns(pl.lit(year).alias("eval_year"))
        )

    fold_metrics = pl.concat(fold_frames, how="diagonal_relaxed")
    summary = summarize_backtest(fold_metrics)
    predictions = (
        pl.concat(prediction_frames, how="diagonal_relaxed")
        if prediction_frames
        else pl.DataFrame()
    )
    resolved = tuple(sorted(fold_metrics["candidate"].unique().to_list()))
    return BacktestResult(
        target_key=target,
        eval_years=tuple(eval_years),
        fold_metrics=fold_metrics,
        summary=summary,
        predictions=predictions,
        candidates=resolved,
    )
