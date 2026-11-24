"""Testes Prove-It da Feature F07 (CLI + logging JSON) e Net Benefit / ECE / BSS / TRIPOD+AI."""

from __future__ import annotations

import json

import numpy as np
import pytest

from previsao_leitos.audit import (
    brier_skill_score,
    decision_curve_net_benefit,
    expected_calibration_error,
)
from previsao_leitos.cli import cli_main, format_json_log_record


def test_decision_curve_net_benefit_ece_and_bss_match_exact_formulas() -> None:
    y_event = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float64)
    good_prob = np.array([0.05, 0.10, 0.12, 0.15, 0.85, 0.88, 0.90, 0.95], dtype=np.float64)

    ece = expected_calibration_error(y_event, good_prob, n_bins=4)
    bss = brier_skill_score(y_event, good_prob)
    assert 0.0 <= ece < 0.15
    assert bss > 0.85

    dca = decision_curve_net_benefit(y_event, good_prob, thresholds=(0.20, 0.30, 0.50))
    assert dca.height == 3
    # Em p_t = 0.50, modelo perfeito acerta todos os 4 eventos (TP=4, FP=0, N=8 -> NB = 0.50)
    row_50 = dca.filter(dca["limiar_pt"] == 0.50)
    assert row_50["net_benefit_modelo"][0] == pytest.approx(0.50)
    assert row_50["net_benefit_modelo"][0] > row_50["net_benefit_tratar_todos"][0]
    assert row_50["net_benefit_modelo"][0] > row_50["net_benefit_tratar_nenhum"][0]


def test_structured_json_log_contains_required_observability_fields() -> None:
    raw = format_json_log_record(
        run_id="rs_check_01",
        entry_point="cli:doctor",
        event="health_check",
        payload={"status": "ok"},
    )
    parsed = json.loads(raw)
    assert parsed["run_id"] == "rs_check_01"
    assert parsed["entry_point"] == "cli:doctor"
    assert parsed["event"] == "health_check"
    assert "git_sha" in parsed
    assert "timestamp_utc" in parsed
    assert parsed["payload"]["status"] == "ok"


def test_cli_doctor_succeeds_and_verify_rs_blocks_2025_holdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = cli_main(["doctor"])
    assert code == 0
    captured = capsys.readouterr()
    record = json.loads(captured.out.strip().splitlines()[-1])
    assert record["event"] == "doctor_complete"

    # Tentar rodar verify-rs em 2025 sem --allow-holdout deve retornar erro (código != 0)
    blocked_code = cli_main(["verify-rs", "--eval-year", "2025"])
    assert blocked_code == 2
    err_out = capsys.readouterr().err
    assert "2025" in err_out
