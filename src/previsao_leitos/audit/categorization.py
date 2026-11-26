"""Categorização de modelos por quesito, com rubrica congelada (`v1`).

Cada modelo recebe uma letra por critério (acurácia, fase, calibração, robustez,
consistência e custo) e um selo geral. O selo é o **pior** entre acurácia, fase e
calibração: erro de fase não se compensa com acurácia.

Mudar limiares exige nova versão da rubrica (`v2`) e reexecução completa — mesmo
princípio dos pré-registros.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

RUBRIC_VERSION = "v1"
GRADE_ORDER = ("A", "B", "C", "D")
NOT_APPLICABLE = "N/A"
REPROVED = "Reprovado"

# Limiares (<= valor ⇒ nota); o último degrau é "D" acima do limite de "C".
_ACCURACY_WMAPE = (8.0, 12.0, 20.0)
_PHASE_WEEKS = (2.0, 4.0, 6.0)
_CALIBRATION_GAP = (0.03, 0.07, 0.12)
_ROBUSTNESS_MASE_WORST = (1.00, 1.10, 1.25)
_CONSISTENCY_CV = (0.05, 0.10, 0.20)
_COST_SECONDS = (1.0, 10.0, 60.0)


def _grade(value: float, thresholds: tuple[float, float, float]) -> str:
    if value is None or not np.isfinite(value):
        return NOT_APPLICABLE
    for grade, limit in zip(GRADE_ORDER, thresholds, strict=False):
        if value <= limit:
            return grade
    return "D"


def _worst(grades: list[str]) -> str:
    usable = [g for g in grades if g in GRADE_ORDER]
    if not usable:
        return NOT_APPLICABLE
    return max(usable, key=GRADE_ORDER.index)


def _fmt(value: float, suffix: str, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}{suffix}"


def categorize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Aplica a rubrica a uma linha normalizada.

    Colunas esperadas: `candidate`, `status`, `wmape`, `coverage`, `peak_weeks`,
    `mase_worst`, `mase_cv`, `fit_seconds`, `monotonic_violations`.
    """
    status = str(row.get("status") or "ok")
    base = {
        "candidate": row.get("candidate"),
        "status": status,
        "rubric": RUBRIC_VERSION,
    }
    if status != "ok":
        base.update(
            {
                "acuracia": NOT_APPLICABLE,
                "fase": NOT_APPLICABLE,
                "calibracao": NOT_APPLICABLE,
                "robustez": NOT_APPLICABLE,
                "consistencia": NOT_APPLICABLE,
                "custo": NOT_APPLICABLE,
                "selo": NOT_APPLICABLE,
                "justificativa": f"candidato {status}",
            }
        )
        return base

    wmape = float(row.get("wmape", np.nan))
    coverage = float(row.get("coverage", np.nan))
    peak = float(row.get("peak_weeks", np.nan))
    worst_mase = float(row.get("mase_worst", np.nan))
    cv = float(row.get("mase_cv", np.nan))
    cost = float(row.get("fit_seconds", np.nan))
    violations = int(row.get("monotonic_violations") or 0)

    gap = abs(coverage - 0.80) if np.isfinite(coverage) else np.nan
    grades = {
        "acuracia": _grade(wmape, _ACCURACY_WMAPE),
        "fase": _grade(peak, _PHASE_WEEKS),
        "calibracao": _grade(gap, _CALIBRATION_GAP),
        "robustez": _grade(worst_mase, _ROBUSTNESS_MASE_WORST),
        "consistencia": _grade(cv, _CONSISTENCY_CV),
        "custo": _grade(cost, _COST_SECONDS),
    }
    if violations > 0:
        seal = REPROVED
    else:
        seal = _worst([grades["acuracia"], grades["fase"], grades["calibracao"]])

    weak: list[str] = []
    if grades["acuracia"] in ("C", "D"):
        weak.append(f"acurácia {_fmt(wmape, '%')}")
    if grades["fase"] in ("C", "D"):
        weak.append(f"fase {_fmt(peak, ' sem')}")
    if grades["calibracao"] in ("C", "D"):
        weak.append(f"cobertura {_fmt(coverage * 100, '%')}")
    if violations > 0:
        weak.append(f"{violations} violações monotônicas")
    justification = "; ".join(weak) if weak else "nenhum critério crítico"

    base.update(grades)
    base["selo"] = seal
    base["justificativa"] = justification
    return base


def categorize(df: pl.DataFrame) -> pl.DataFrame:
    """Categoriza um DataFrame já normalizado (uma linha por candidato)."""
    if df.height == 0:
        return pl.DataFrame(
            schema={
                "candidate": pl.String,
                "selo": pl.String,
                "justificativa": pl.String,
            }
        )
    rows = [categorize_row(row) for row in df.to_dicts()]
    return pl.DataFrame(rows, infer_schema_length=None)


def normalize_benchmark(metrics: pl.DataFrame) -> pl.DataFrame:
    """Adapta a tabela de benchmark (um ano) para o formato canônico."""
    return metrics.select(
        pl.col("candidate"),
        pl.col("status"),
        pl.col("wmape"),
        pl.col("cobertura_p10_p90").alias("coverage"),
        pl.col("peak_timing_error_weeks").alias("peak_weeks"),
        pl.col("mase_vs_pers").alias("mase_worst"),
        pl.lit(float("nan")).alias("mase_cv"),
        pl.col("fit_seconds"),
        pl.col("monotonic_violations"),
    )


def normalize_backtest(summary: pl.DataFrame) -> pl.DataFrame:
    """Adapta o sumário do backtest (múltiplos folds) para o formato canônico."""
    return summary.select(
        pl.col("candidate"),
        pl.lit("ok").alias("status"),
        pl.col("wmape_median").alias("wmape"),
        pl.col("coverage_median").alias("coverage"),
        pl.col("peak_abs_median").alias("peak_weeks"),
        pl.col("mase_worst"),
        pl.col("mase_cv"),
        pl.col("fit_seconds_median").alias("fit_seconds"),
        pl.col("monotonic_violations"),
    )
