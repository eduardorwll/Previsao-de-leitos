"""Registro versionado de hipóteses e catraca científica Champion-Challenger.

Permite formalizar hipóteses sobre falhas de dados, covariáveis físicas ou
mudanças arquiteturais, comparando objetivamente um Challenger contra o
Champion vigente sob travas não negociáveis de física, higiene e calibração.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from previsao_leitos.audit.data_hygiene import CLINICAL_DISCLAIMER
from previsao_leitos.domain.exceptions import DataContractError

HypothesisStatus = Literal["PROPOSED", "CONFIRMED", "REFUTED", "INCONCLUSIVE"]
PromotionDecision = Literal["PROMOTE", "REJECT", "KEEP_CHAMPION"]


@dataclass(frozen=True)
class ModelScorecard:
    """Métricas-chave padronizadas de uma execução auditada de modelo."""

    run_id: str
    model_name: str
    mase: float
    pinball_p95: float
    coverage_p90: float
    peak_timing_error_weeks: float
    interior_specificity: float
    ece: float
    monotonic_violations: int = 0
    hygiene_failed_gates: int = 0
    extra_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializa o scorecard do modelo para dicionário JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelScorecard:
        """Reconstrói um ModelScorecard a partir de dicionário JSON."""
        return cls(
            run_id=str(payload["run_id"]),
            model_name=str(payload["model_name"]),
            mase=float(payload["mase"]),
            pinball_p95=float(payload["pinball_p95"]),
            coverage_p90=float(payload["coverage_p90"]),
            peak_timing_error_weeks=float(payload["peak_timing_error_weeks"]),
            interior_specificity=float(payload["interior_specificity"]),
            ece=float(payload["ece"]),
            monotonic_violations=int(payload.get("monotonic_violations", 0)),
            hygiene_failed_gates=int(payload.get("hygiene_failed_gates", 0)),
            extra_metrics={
                str(k): float(v) for k, v in dict(payload.get("extra_metrics", {})).items()
            },
        )


@dataclass(frozen=True)
class RatchetCriteria:
    """Critérios não negociáveis da catraca Champion-Challenger."""

    min_primary_improvement: float = 0.005
    max_secondary_regression_pct: float = 5.0
    min_coverage_p90: float = 0.80
    min_interior_specificity: float = 0.65
    max_monotonic_violations: int = 0
    max_hygiene_failed_gates: int = 0


@dataclass(frozen=True)
class RatchetEvaluation:
    """Resultado auditável de comparação Champion vs. Challenger."""

    champion_run_id: str
    challenger_run_id: str
    decision: PromotionDecision
    winner_run_id: str
    ablation_delta: dict[str, dict[str, float]]
    blocking_reasons: tuple[str, ...]
    summary: str
    evaluated_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    disclaimer: str = CLINICAL_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        """Serializa o resultado da avaliação da catraca para JSON."""
        return {
            "champion_run_id": self.champion_run_id,
            "challenger_run_id": self.challenger_run_id,
            "decision": self.decision,
            "winner_run_id": self.winner_run_id,
            "ablation_delta": self.ablation_delta,
            "blocking_reasons": list(self.blocking_reasons),
            "summary": self.summary,
            "evaluated_at_utc": self.evaluated_at_utc,
            "disclaimer": self.disclaimer,
        }

    def save_ablation_delta(self, path: Path) -> Path:
        """Persiste o resultado da comparação em ablation_delta.json de forma atômica."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        tmp_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        tmp_path.replace(path)
        return path


def _metric_delta(champion_val: float, challenger_val: float) -> dict[str, float]:
    delta = challenger_val - champion_val
    rel_pct = 0.0 if abs(champion_val) <= 1e-12 else 100.0 * delta / abs(champion_val)
    return {
        "champion": round(champion_val, 6),
        "challenger": round(challenger_val, 6),
        "delta": round(delta, 6),
        "delta_pct": round(rel_pct, 4),
    }


class ChampionChallengerRatchet:
    """Catraca científica determinística que promove ou rejeita um Challenger."""

    def __init__(self, criteria: RatchetCriteria | None = None) -> None:
        self.criteria = criteria or RatchetCriteria()

    def evaluate(
        self,
        *,
        champion: ModelScorecard,
        challenger: ModelScorecard,
    ) -> RatchetEvaluation:
        """Compara Challenger contra Champion sob travas físicas, de higiene e calibração."""
        c = self.criteria
        ablation_delta: dict[str, dict[str, float]] = {
            "mase": _metric_delta(champion.mase, challenger.mase),
            "pinball_p95": _metric_delta(champion.pinball_p95, challenger.pinball_p95),
            "coverage_p90": _metric_delta(champion.coverage_p90, challenger.coverage_p90),
            "peak_timing_error_weeks": _metric_delta(
                champion.peak_timing_error_weeks,
                challenger.peak_timing_error_weeks,
            ),
            "interior_specificity": _metric_delta(
                champion.interior_specificity,
                challenger.interior_specificity,
            ),
            "ece": _metric_delta(champion.ece, challenger.ece),
        }

        blocking: list[str] = []
        if challenger.monotonic_violations > c.max_monotonic_violations:
            blocking.append(
                f"Violação monotônica detectada ({challenger.monotonic_violations} > "
                f"{c.max_monotonic_violations})."
            )
        if challenger.hygiene_failed_gates > c.max_hygiene_failed_gates:
            blocking.append(
                f"Falha em gates de higiene de dados ({challenger.hygiene_failed_gates} > "
                f"{c.max_hygiene_failed_gates})."
            )
        if challenger.coverage_p90 < c.min_coverage_p90:
            blocking.append(
                f"Cobertura P90 abaixo do piso ({challenger.coverage_p90:.3f} < "
                f"{c.min_coverage_p90:.3f})."
            )
        if challenger.interior_specificity < c.min_interior_specificity:
            blocking.append(
                f"Especificidade no interior abaixo do piso "
                f"({challenger.interior_specificity:.3f} < {c.min_interior_specificity:.3f})."
            )

        if blocking:
            return RatchetEvaluation(
                champion_run_id=champion.run_id,
                challenger_run_id=challenger.run_id,
                decision="REJECT",
                winner_run_id=champion.run_id,
                ablation_delta=ablation_delta,
                blocking_reasons=tuple(blocking),
                summary="Challenger rejeitado por violar travas não negociáveis.",
            )

        secondary_regressions: list[str] = []
        if ablation_delta["pinball_p95"]["delta_pct"] > c.max_secondary_regression_pct:
            secondary_regressions.append(
                f"Regressão em Pinball P95 (+{ablation_delta['pinball_p95']['delta_pct']:.2f}%)."
            )
        if ablation_delta["ece"]["delta_pct"] > c.max_secondary_regression_pct:
            secondary_regressions.append(
                f"Regressão em ECE (+{ablation_delta['ece']['delta_pct']:.2f}%)."
            )

        if secondary_regressions:
            return RatchetEvaluation(
                champion_run_id=champion.run_id,
                challenger_run_id=challenger.run_id,
                decision="REJECT",
                winner_run_id=champion.run_id,
                ablation_delta=ablation_delta,
                blocking_reasons=tuple(secondary_regressions),
                summary="Challenger rejeitado por regressão em métrica secundária.",
            )

        mase_improvement = champion.mase - challenger.mase
        timing_improvement = champion.peak_timing_error_weeks - challenger.peak_timing_error_weeks
        if mase_improvement >= c.min_primary_improvement or (
            mase_improvement >= 0.0 and timing_improvement >= 0.5
        ):
            return RatchetEvaluation(
                champion_run_id=champion.run_id,
                challenger_run_id=challenger.run_id,
                decision="PROMOTE",
                winner_run_id=challenger.run_id,
                ablation_delta=ablation_delta,
                blocking_reasons=(),
                summary=(
                    f"Challenger '{challenger.run_id}' promovido a Campeão "
                    f"(delta MASE = {-mase_improvement:+.4f}, "
                    f"delta timing = {-timing_improvement:+.2f} sem)."
                ),
            )

        if mase_improvement < -c.min_primary_improvement:
            return RatchetEvaluation(
                champion_run_id=champion.run_id,
                challenger_run_id=challenger.run_id,
                decision="REJECT",
                winner_run_id=champion.run_id,
                ablation_delta=ablation_delta,
                blocking_reasons=(
                    f"Regressão em MASE ({champion.mase:.4f} -> {challenger.mase:.4f}).",
                ),
                summary="Challenger rejeitado por piorar o erro primário (MASE).",
            )

        return RatchetEvaluation(
            champion_run_id=champion.run_id,
            challenger_run_id=challenger.run_id,
            decision="KEEP_CHAMPION",
            winner_run_id=champion.run_id,
            ablation_delta=ablation_delta,
            blocking_reasons=(),
            summary="Empate técnico dentro da margem de tolerância; Campeão mantido.",
        )


@dataclass
class HypothesisEntry:
    """Registro estruturado de uma hipótese científica de saneamento ou modelagem."""

    hypothesis_id: str
    issue_code: str
    title: str
    expected_effect: str
    discriminant_test: str
    status: HypothesisStatus = "PROPOSED"
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    decision: PromotionDecision | None = None
    challenger_run_id: str | None = None
    ablation_delta: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serializa a entrada de hipótese para JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HypothesisEntry:
        """Reconstrói HypothesisEntry a partir de dicionário JSON."""
        return cls(
            hypothesis_id=str(payload["hypothesis_id"]),
            issue_code=str(payload["issue_code"]),
            title=str(payload["title"]),
            expected_effect=str(payload["expected_effect"]),
            discriminant_test=str(payload["discriminant_test"]),
            status=payload.get("status", "PROPOSED"),
            created_at_utc=str(payload.get("created_at_utc", datetime.now(UTC).isoformat())),
            decision=payload.get("decision"),
            challenger_run_id=payload.get("challenger_run_id"),
            ablation_delta=dict(payload.get("ablation_delta", {})),
            notes=str(payload.get("notes", "")),
        )


class HypothesisLedger:
    """Livro-razão versionado de hipóteses e histórico de promoções do Campeão."""

    def __init__(
        self,
        *,
        champion: ModelScorecard,
        entries: list[HypothesisEntry] | None = None,
    ) -> None:
        self.champion = champion
        self.entries: list[HypothesisEntry] = list(entries or [])

    def register_hypothesis(self, entry: HypothesisEntry) -> None:
        """Registra uma nova hipótese garantindo unicidade do identificador."""
        if any(e.hypothesis_id == entry.hypothesis_id for e in self.entries):
            raise DataContractError(
                f"Hipótese duplicada no livro-razão: '{entry.hypothesis_id}'."
            )
        self.entries.append(entry)

    def get_hypothesis(self, hypothesis_id: str) -> HypothesisEntry:
        """Recupera uma hipótese pelo identificador."""
        for entry in self.entries:
            if entry.hypothesis_id == hypothesis_id:
                return entry
        raise DataContractError(f"Hipótese não encontrada: '{hypothesis_id}'.")

    def record_experiment(
        self,
        *,
        hypothesis_id: str,
        challenger: ModelScorecard,
        ratchet: ChampionChallengerRatchet | None = None,
    ) -> RatchetEvaluation:
        """Avalia o Challenger via catraca, atualiza a hipótese e promove o Campeão se vencer."""
        entry = self.get_hypothesis(hypothesis_id)
        evaluator = ratchet or ChampionChallengerRatchet()
        evaluation = evaluator.evaluate(champion=self.champion, challenger=challenger)

        entry.decision = evaluation.decision
        entry.challenger_run_id = challenger.run_id
        entry.ablation_delta = evaluation.ablation_delta
        entry.notes = evaluation.summary

        if evaluation.decision == "PROMOTE":
            entry.status = "CONFIRMED"
            self.champion = challenger
        elif evaluation.decision == "REJECT":
            entry.status = "REFUTED"
        else:
            entry.status = "INCONCLUSIVE"

        return evaluation

    def to_dict(self) -> dict[str, Any]:
        """Serializa o livro-razão completo para JSON."""
        return {
            "disclaimer": CLINICAL_DISCLAIMER,
            "champion": self.champion.to_dict(),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def save_json(self, path: Path) -> Path:
        """Persiste o livro-razão em hypothesis_ledger.json de forma atômica."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        tmp_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        tmp_path.replace(path)
        return path

    @classmethod
    def load_json(cls, path: Path) -> HypothesisLedger:
        """Carrega um HypothesisLedger a partir de um arquivo JSON."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        champion = ModelScorecard.from_dict(payload["champion"])
        entries = [HypothesisEntry.from_dict(item) for item in payload.get("entries", [])]
        return cls(champion=champion, entries=entries)
