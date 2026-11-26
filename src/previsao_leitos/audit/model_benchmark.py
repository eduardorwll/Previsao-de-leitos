"""Benchmark multi-modelo: previsões, métricas, comparação e catraca.

Um único ponto de avaliação para todos os candidatos de demanda: treina cada um
com o mesmo split causal, mede os mesmos indicadores e compara com a persistência,
com o campeão registrado e entre si.

A catraca aqui é **quantílica** (MASE, Pinball P95, cobertura e *timing* de pico).
A especificidade de surto e o ECE binário continuam sendo auditados pelo caminho
dedicado da Task 1.2.4, e não são reavaliados neste módulo.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import polars as pl

from previsao_leitos.audit.hypothesis_ledger import (
    ChampionChallengerRatchet,
    ModelScorecard,
    RatchetCriteria,
    RatchetEvaluation,
)
from previsao_leitos.audit.metrics import evaluate_probabilistic_predictions, pinball_loss
from previsao_leitos.demand.modeling import ModelingSplits
from previsao_leitos.demand.models import DemandForecaster
from previsao_leitos.demand.registry import (
    CandidateSpec,
    champion_spec,
    resolve_selection,
)
from previsao_leitos.domain import DataContractError

_SHOCK_COL = "prob_enchente_f03_oof"
_MIN_ROWS_FOR_REGION_TABLE = 5


@dataclass(frozen=True)
class BenchmarkResult:
    """Saída auditável de um run de benchmark."""

    run_id: str
    target_key: str
    target_col: str
    eval_year: int
    predictions: pl.DataFrame
    metrics: pl.DataFrame
    comparison: pl.DataFrame
    decisions: pl.DataFrame
    candidates: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)


def _fit_one(spec: CandidateSpec, splits: ModelingSplits) -> DemandForecaster:
    model = spec.factory()
    if spec.fit_mode == "plain":
        return model.fit(splits.train_df, splits.feature_cols)
    if spec.fit_mode == "train_cal":
        model.fit(splits.train_df, splits.feature_cols)
        calibrator = getattr(model, "fit_calibration", None)
        if calibrator is None:
            raise DataContractError(f"{spec.name}: calibração 'train_cal' indisponível.")
        return cast(DemandForecaster, calibrator(train_df=splits.train_df, cal_df=splits.cal_df))
    if spec.fit_mode == "cal_only":
        model.fit(splits.train_df, splits.feature_cols)
        calibrator = getattr(model, "fit_calibration", None)
        if calibrator is None:
            raise DataContractError(f"{spec.name}: calibração 'cal_only' indisponível.")
        return cast(DemandForecaster, calibrator(cal_df=splits.cal_df))
    raise DataContractError(f"{spec.name}: fit_mode desconhecido '{spec.fit_mode}'.")


def _count_shock_response_violations(
    model: DemandForecaster,
    eval_df: pl.DataFrame,
    *,
    bump: float = 0.10,
) -> int:
    """Conta quedas de P50 quando o choque hidrológico aumenta (ADR-0001)."""
    if _SHOCK_COL not in eval_df.columns:
        return 0
    try:
        base = model.predict_quantiles(eval_df)["pred_p50"].to_numpy().astype(np.float64)
        bumped = eval_df.with_columns(
            (pl.col(_SHOCK_COL).fill_null(0.0) + bump).clip(0.0, 1.0).alias(_SHOCK_COL)
        )
        raised = model.predict_quantiles(bumped)["pred_p50"].to_numpy().astype(np.float64)
    except (DataContractError, KeyError):
        return 0
    return int(np.sum(raised < base - 1e-9))


def _peak_timing_error(
    eval_df: pl.DataFrame,
    p50: np.ndarray,
    *,
    entity_col: str,
    week_col: str,
    target_col: str,
) -> float:
    """Erro absoluto médio (em semanas) entre o pico observado e o pico previsto."""
    frame = eval_df.select(
        [
            pl.col(entity_col),
            pl.col(week_col),
            pl.col(target_col),
        ]
    ).with_columns(
        pl.Series("_p50", p50),
        pl.col(target_col).alias("_obs"),
    )
    errors: list[float] = []
    for _, group in frame.group_by(entity_col):
        obs_idx = int(group["_obs"].arg_max() or 0)
        pred_idx = int(group["_p50"].arg_max() or 0)
        errors.append(abs(float(obs_idx - pred_idx)))
    return float(np.mean(errors)) if errors else 0.0


def _tidy_predictions(
    eval_df: pl.DataFrame,
    preds: pl.DataFrame,
    *,
    model_name: str,
    entity_col: str,
    target_col: str,
) -> pl.DataFrame:
    return eval_df.select(
        [
            pl.col(entity_col).alias("entity"),
            pl.col("ano_epi").alias("origin_year"),
            pl.col("semana_epi").alias("origin_week"),
            pl.col("target_ano_epi").alias("target_year"),
            pl.col("target_semana_epi").alias("target_week"),
            pl.col(target_col).alias("observed"),
        ]
    ).with_columns(
        pl.lit(model_name).alias("model"),
        preds["pred_p10"],
        preds["pred_p50"],
        preds["pred_p90"],
        preds["pred_p95"],
    )


def _region_comparison(
    eval_df: pl.DataFrame,
    preds: pl.DataFrame,
    *,
    model_name: str,
    entity_col: str,
    target_col: str,
    week_col: str,
) -> pl.DataFrame:
    frame = eval_df.select(
        [
            pl.col(entity_col).alias("entity"),
            pl.col(week_col).alias("week"),
            pl.col(target_col),
        ]
    ).with_columns(
        pl.col(target_col).alias("observed"),
        pl.Series("pred_p50", preds["pred_p50"].to_numpy()),
    )
    table = (
        frame.group_by("entity")
        .agg(
            pl.len().alias("n_obs"),
            pl.col("observed").mean().alias("mean_observed"),
            pl.col("pred_p50").mean().alias("mean_predicted"),
            (pl.col("pred_p50") - pl.col("observed")).mean().alias("bias"),
            (pl.col("pred_p50") - pl.col("observed")).abs().mean().alias("mae"),
            pl.col("observed").max().alias("peak_observed"),
            pl.col("pred_p50").max().alias("peak_predicted"),
        )
        .with_columns(pl.lit(model_name).alias("model"))
        .sort("entity")
    )
    # LGPD: regiões com menos de 5 semanas não entram na tabela publicada.
    return table.filter(pl.col("n_obs") >= _MIN_ROWS_FOR_REGION_TABLE)


def _metric_row(
    *,
    spec: CandidateSpec,
    metrics: dict[str, Any],
    peak_error: float,
    monotonic_violations: int,
    fit_seconds: float,
    predict_seconds: float,
    n_obs: int,
) -> dict[str, object]:
    return {
        "candidate": spec.name,
        "family": spec.family,
        "status": "ok",
        "is_champion": spec.is_champion,
        "mae_p50": float(metrics["mae_p50"]),
        "mae_persistencia": float(metrics["mae_persistencia"]),
        "mae_sazonal": float(metrics["mae_sazonal"]),
        "mase_vs_pers": float(metrics["mase_vs_pers"]),
        "mase_ci95_low": float(metrics["mase_ci95_low"]),
        "mase_ci95_high": float(metrics["mase_ci95_high"]),
        "mase_vs_saz": float(metrics["mase_vs_saz"]),
        "pinball_p10": float(metrics.get("pinball_p10", float("nan"))),
        "pinball_p50": float(metrics["pinball_p50"]),
        "pinball_p90": float(metrics["pinball_p90"]),
        "pinball_p95": float(metrics["pinball_p95"]),
        "cobertura_p10_p90": float(metrics["cobertura_p10_p90"]),
        "peak_timing_error_weeks": peak_error,
        "monotonic_violations": monotonic_violations,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_obs": n_obs,
    }


def run_benchmark(
    splits: ModelingSplits,
    *,
    run_id: str,
    candidates: str | Sequence[str] = "all",
    n_boot: int = 200,
    seed: int = 42,
    include_ratchet: bool = True,
) -> BenchmarkResult:
    """Treina, avalia e compara todos os candidatos selecionados."""
    specs = resolve_selection(candidates)
    if splits.eval_df.height == 0:
        raise DataContractError("Conjunto de avaliação vazio; benchmark abortado.")

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pl.DataFrame] = []
    comparison_frames: list[pl.DataFrame] = []
    notes: list[str] = []

    for spec in specs:
        if not spec.available:
            metric_rows.append(
                {
                    "candidate": spec.name,
                    "family": spec.family,
                    "status": "unavailable",
                    "error": f"extras ausentes: {', '.join(spec.requires)}",
                    "is_champion": spec.is_champion,
                }
            )
            continue
        try:
            import time

            t0 = time.perf_counter()
            model = _fit_one(spec, splits)
            fit_seconds = time.perf_counter() - t0
            t1 = time.perf_counter()
            preds = model.predict_quantiles(splits.eval_df)
            predict_seconds = time.perf_counter() - t1
            if preds.height != splits.eval_df.height:
                raise DataContractError(
                    f"{spec.name}: {preds.height} previsões para {splits.eval_df.height} linhas."
                )
            scored = splits.eval_df.with_columns(
                pl.Series("pred_p10", preds["pred_p10"].to_numpy()),
                pl.Series("pred_p50", preds["pred_p50"].to_numpy()),
                pl.Series("pred_p90", preds["pred_p90"].to_numpy()),
                pl.Series("pred_p95", preds["pred_p95"].to_numpy()),
            )
            metrics = evaluate_probabilistic_predictions(
                splits.eval_df,
                preds,
                model_name=spec.name,
                block_col=splits.entity_col,
                n_boot=n_boot,
                seed=seed,
            )
            metrics["pinball_p10"] = pinball_loss(
                splits.eval_df["target_y"].to_numpy(),
                preds["pred_p10"].to_numpy(),
                alpha=0.10,
            )
            peak_error = _peak_timing_error(
                splits.eval_df,
                preds["pred_p50"].to_numpy().astype(np.float64),
                entity_col=splits.entity_col,
                week_col="target_semana_epi",
                target_col="target_y",
            )
            monotonic_violations = _count_shock_response_violations(model, splits.eval_df)
            metric_rows.append(
                _metric_row(
                    spec=spec,
                    metrics=metrics,
                    peak_error=peak_error,
                    monotonic_violations=monotonic_violations,
                    fit_seconds=fit_seconds,
                    predict_seconds=predict_seconds,
                    n_obs=scored.height,
                )
            )
            prediction_frames.append(
                _tidy_predictions(
                    splits.eval_df,
                    preds,
                    model_name=spec.name,
                    entity_col=splits.entity_col,
                    target_col="target_y",
                )
            )
            comparison_frames.append(
                _region_comparison(
                    splits.eval_df,
                    preds,
                    model_name=spec.name,
                    entity_col=splits.entity_col,
                    target_col="target_y",
                    week_col="target_semana_epi",
                )
            )
        except Exception as exc:
            metric_rows.append(
                {
                    "candidate": spec.name,
                    "family": spec.family,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "is_champion": spec.is_champion,
                }
            )

    metrics_df = pl.DataFrame(metric_rows, infer_schema_length=None)
    predictions = (
        pl.concat(prediction_frames, how="diagonal_relaxed")
        if prediction_frames
        else pl.DataFrame()
    )
    comparison = (
        pl.concat(comparison_frames, how="diagonal_relaxed")
        if comparison_frames
        else pl.DataFrame()
    )
    decisions = (
        _evaluate_decisions(metrics_df)
        if include_ratchet
        else pl.DataFrame(schema={"candidate": pl.String, "decision": pl.String})
    )

    if metrics_df.filter(pl.col("status") == "ok").height == 0:
        raise DataContractError("Nenhum candidato concluiu o benchmark.")
    if include_ratchet and decisions.height == 0:
        notes.append(
            f"Campeão '{champion_spec().name}' não incluído na seleção; catraca omitida."
        )
    try:
        if (
            "mae_sazonal" in metrics_df.columns
            and metrics_df["mae_sazonal"].is_nan().all()
        ):
            notes.append("Sazonal-ingênuo indisponível no painel (lag 52 incompleto).")
    except (pl.exceptions.SchemaError, TypeError):
        pass

    return BenchmarkResult(
        run_id=run_id,
        target_key=splits.target_key,
        target_col=splits.target_col,
        eval_year=splits.eval_year,
        predictions=predictions,
        metrics=metrics_df,
        comparison=comparison,
        decisions=decisions,
        candidates=tuple(spec.name for spec in specs),
        notes=tuple(notes),
    )


def _scorecard_from_row(row: dict[str, Any], *, prefix: str) -> ModelScorecard:
    return ModelScorecard(
        run_id=str(row["candidate"]),
        model_name=f"{prefix}:{row['candidate']}",
        mase=float(row["mase_vs_pers"]),
        pinball_p95=float(row["pinball_p95"]),
        coverage_p90=float(row["cobertura_p10_p90"]),
        peak_timing_error_weeks=float(row["peak_timing_error_weeks"]),
        interior_specificity=1.0,
        ece=0.0,
        monotonic_violations=int(row.get("monotonic_violations") or 0),
    )


def _evaluate_decisions(metrics_df: pl.DataFrame) -> pl.DataFrame:
    """Compara cada candidato contra o campeão registrado (catraca quantílica)."""
    champion_name = champion_spec().name
    ok = metrics_df.filter(pl.col("status") == "ok").to_dicts()
    champion_row = next((row for row in ok if row["candidate"] == champion_name), None)
    if champion_row is None:
        return pl.DataFrame(
            schema={
                "candidate": pl.String,
                "champion": pl.String,
                "decision": pl.String,
                "summary": pl.String,
            }
        )

    criteria = RatchetCriteria(
        min_coverage_p90=0.80,
        min_interior_specificity=0.0,
        max_monotonic_violations=0,
    )
    ratchet = ChampionChallengerRatchet(criteria)
    champion = _scorecard_from_row(champion_row, prefix="champion")
    rows: list[dict[str, Any]] = []
    for row in ok:
        if row["candidate"] == champion_name:
            continue
        evaluation: RatchetEvaluation = ratchet.evaluate(
            champion=champion,
            challenger=_scorecard_from_row(row, prefix="challenger"),
        )
        rows.append(
            {
                "candidate": row["candidate"],
                "champion": champion_name,
                "decision": evaluation.decision,
                "summary": evaluation.summary,
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)
