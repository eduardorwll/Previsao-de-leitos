"""Testes da rubrica de categorização (v1)."""

from __future__ import annotations

import polars as pl

from previsao_leitos.audit.categorization import (
    NOT_APPLICABLE,
    REPROVED,
    RUBRIC_VERSION,
    categorize,
    categorize_row,
    normalize_backtest,
)


def _row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "candidate": "modelo",
        "status": "ok",
        "wmape": 6.0,
        "coverage": 0.80,
        "peak_weeks": 1.5,
        "mase_worst": 0.95,
        "mase_cv": 0.03,
        "fit_seconds": 0.5,
        "monotonic_violations": 0,
    }
    base.update(overrides)
    return base


def test_all_criteria_get_top_grade_when_perfect() -> None:
    graded = categorize_row(_row())
    assert graded["rubric"] == RUBRIC_VERSION
    for criterion in ("acuracia", "fase", "calibracao", "robustez", "consistencia", "custo"):
        assert graded[criterion] == "A", criterion
    assert graded["selo"] == "A"


def test_grade_boundaries_are_inclusive() -> None:
    assert categorize_row(_row(wmape=8.0))["acuracia"] == "A"
    assert categorize_row(_row(wmape=8.01))["acuracia"] == "B"
    assert categorize_row(_row(wmape=12.0))["acuracia"] == "B"
    assert categorize_row(_row(wmape=12.01))["acuracia"] == "C"
    assert categorize_row(_row(wmape=20.01))["acuracia"] == "D"
    assert categorize_row(_row(peak_weeks=2.0))["fase"] == "A"
    assert categorize_row(_row(peak_weeks=2.01))["fase"] == "B"


def test_overall_seal_is_the_worst_of_accuracy_phase_and_calibration() -> None:
    graded = categorize_row(_row(peak_weeks=9.0))
    assert graded["fase"] == "D"
    assert graded["selo"] == "D"
    # robustez/consistência não pioram o selo, apenas informam.
    graded = categorize_row(_row(mase_worst=1.4, mase_cv=0.3))
    assert graded["robustez"] == "D"
    assert graded["consistencia"] == "D"
    assert graded["selo"] == "A"


def test_monotonic_violation_reproves_regardless_of_other_grades() -> None:
    graded = categorize_row(_row(monotonic_violations=3))
    assert graded["selo"] == REPROVED
    assert "violações" in graded["justificativa"]


def test_unavailable_candidate_is_not_applicable_and_never_omitted() -> None:
    graded = categorize_row(_row(status="unavailable"))
    assert graded["selo"] == NOT_APPLICABLE
    assert graded["acuracia"] == NOT_APPLICABLE


def test_missing_metric_degrades_gracefully() -> None:
    graded = categorize_row(_row(mase_cv=float("nan")))
    assert graded["consistencia"] == NOT_APPLICABLE


def test_backtest_summary_adapter_and_dataframe_output() -> None:
    summary = pl.DataFrame(
        {
            "candidate": ["a", "b"],
            "wmape_median": [6.0, 15.0],
            "coverage_median": [0.80, 0.70],
            "peak_abs_median": [1.5, 5.0],
            "mase_worst": [0.95, 1.2],
            "mase_cv": [0.03, 0.15],
            "fit_seconds_median": [0.5, 2.0],
            "monotonic_violations": [0, 0],
        }
    )
    normalized = normalize_backtest(summary)
    graded = categorize(normalized)
    assert graded.height == 2
    assert set(graded.columns) >= {"candidate", "selo", "justificativa"}
    assert graded.filter(pl.col("candidate") == "a")["selo"][0] == "A"
    # wmape 15 (C), pico 5 sem (C) e cobertura 70% (gap 10 p.p. = C) => selo C.
    assert graded.filter(pl.col("candidate") == "b")["selo"][0] == "C"


def test_empty_dataframe_returns_empty_schema() -> None:
    graded = categorize(pl.DataFrame())
    assert graded.height == 0
    assert "selo" in graded.columns
