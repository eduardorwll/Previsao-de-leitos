"""Testes unitários do registro de hipóteses e catraca Champion-Challenger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from previsao_leitos.audit.hypothesis_ledger import (
    ChampionChallengerRatchet,
    HypothesisEntry,
    HypothesisLedger,
    ModelScorecard,
    RatchetCriteria,
)
from previsao_leitos.domain.exceptions import DataContractError


def _make_scorecard(
    run_id: str,
    *,
    mase: float = 0.82,
    pinball_p95: float = 14.0,
    coverage_p90: float = 0.88,
    peak_timing_error_weeks: float = 1.5,
    interior_specificity: float = 0.72,
    ece: float = 0.05,
    monotonic_violations: int = 0,
    hygiene_failed_gates: int = 0,
) -> ModelScorecard:
    return ModelScorecard(
        run_id=run_id,
        model_name="QuantileGBDT",
        mase=mase,
        pinball_p95=pinball_p95,
        coverage_p90=coverage_p90,
        peak_timing_error_weeks=peak_timing_error_weeks,
        interior_specificity=interior_specificity,
        ece=ece,
        monotonic_violations=monotonic_violations,
        hygiene_failed_gates=hygiene_failed_gates,
    )


def test_ratchet_promotes_valid_challenger_and_computes_ablation_delta() -> None:
    champion = _make_scorecard("run_v1", mase=0.82, pinball_p95=14.0, peak_timing_error_weeks=2.0)
    challenger = _make_scorecard(
        "run_v1_1", mase=0.74, pinball_p95=12.1, peak_timing_error_weeks=0.5
    )

    ratchet = ChampionChallengerRatchet(RatchetCriteria(min_primary_improvement=0.01))
    result = ratchet.evaluate(champion=champion, challenger=challenger)

    assert result.decision == "PROMOTE"
    assert result.winner_run_id == "run_v1_1"
    assert result.ablation_delta["mase"]["delta"] == pytest.approx(-0.08)
    assert result.ablation_delta["peak_timing_error_weeks"]["delta"] == pytest.approx(-1.5)
    assert result.blocking_reasons == ()


def test_ratchet_rejects_monotonic_violation_or_hygiene_failure() -> None:
    champion = _make_scorecard("run_v1", mase=0.82)
    challenger_bad_mono = _make_scorecard("run_bad_mono", mase=0.65, monotonic_violations=3)
    challenger_bad_hyg = _make_scorecard("run_bad_hyg", mase=0.65, hygiene_failed_gates=2)

    ratchet = ChampionChallengerRatchet()
    res_mono = ratchet.evaluate(champion=champion, challenger=challenger_bad_mono)
    assert res_mono.decision == "REJECT"
    assert res_mono.winner_run_id == "run_v1"
    assert any("monotôn" in reason.lower() for reason in res_mono.blocking_reasons)

    res_hyg = ratchet.evaluate(champion=champion, challenger=challenger_bad_hyg)
    assert res_hyg.decision == "REJECT"
    assert any("higiene" in reason.lower() for reason in res_hyg.blocking_reasons)


def test_ratchet_rejects_interior_specificity_collapse_or_coverage_drop() -> None:
    champion = _make_scorecard("run_v1", mase=0.82, interior_specificity=0.75, coverage_p90=0.88)
    challenger_collapse = _make_scorecard(
        "run_collapse", mase=0.70, interior_specificity=0.091, coverage_p90=0.60
    )

    ratchet = ChampionChallengerRatchet(
        RatchetCriteria(min_coverage_p90=0.80, min_interior_specificity=0.65)
    )
    res = ratchet.evaluate(champion=champion, challenger=challenger_collapse)
    assert res.decision == "REJECT"
    assert len(res.blocking_reasons) == 2


def test_ratchet_keeps_champion_on_marginal_tie() -> None:
    champion = _make_scorecard("run_v1", mase=0.820)
    challenger = _make_scorecard("run_tie", mase=0.818)

    ratchet = ChampionChallengerRatchet(RatchetCriteria(min_primary_improvement=0.01))
    res = ratchet.evaluate(champion=champion, challenger=challenger)
    assert res.decision == "KEEP_CHAMPION"
    assert res.winner_run_id == "run_v1"


def test_hypothesis_ledger_register_evaluate_persist_and_reload(tmp_path: Path) -> None:
    champion = _make_scorecard("run_v1", mase=0.82, pinball_p95=14.0)
    ledger = HypothesisLedger(champion=champion)

    entry = HypothesisEntry(
        hypothesis_id="H-C2",
        issue_code="C2",
        title="Agregar W_ref antes da supressão LGPD elimina colinearidade VIF=12.742",
        expected_effect="VIF < 5 e redução de MASE em Diárias UTI",
        discriminant_test="Comparar VIF (G1) e MASE UTI 2024 com W_ref externa vs. V1",
    )
    ledger.register_hypothesis(entry)

    challenger = _make_scorecard("run_v1_c2", mase=0.76, pinball_p95=12.8)
    evaluation = ledger.record_experiment(
        hypothesis_id="H-C2",
        challenger=challenger,
        ratchet=ChampionChallengerRatchet(),
    )
    assert evaluation.decision == "PROMOTE"
    assert ledger.champion.run_id == "run_v1_c2"
    assert ledger.get_hypothesis("H-C2").status == "CONFIRMED"

    ledger_path = ledger.save_json(tmp_path / "hypothesis_ledger.json")
    delta_path = evaluation.save_ablation_delta(tmp_path / "ablation_delta.json")
    assert delta_path.exists()
    delta_payload = json.loads(delta_path.read_text(encoding="utf-8"))
    assert delta_payload["decision"] == "PROMOTE"

    reloaded = HypothesisLedger.load_json(ledger_path)
    assert reloaded.champion.run_id == "run_v1_c2"
    assert len(reloaded.entries) == 1
    assert reloaded.get_hypothesis("H-C2").status == "CONFIRMED"

    with pytest.raises(DataContractError, match="duplicada"):
        reloaded.register_hypothesis(entry)
