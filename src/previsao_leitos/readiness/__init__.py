"""Módulo de prontidão hospitalar, simulação SimPy e gatilhos observáveis (Feature F05)."""

from previsao_leitos.readiness.simulator import (
    CounterfactualReadinessComparison,
    HospitalReadinessConfig,
    PolicyMode,
    PolicySimulationResult,
    simulate_counterfactual_readiness,
)
from previsao_leitos.readiness.triggers import (
    ObservableTriggerAssessment,
    evaluate_observable_triggers,
)

__all__ = [
    "CounterfactualReadinessComparison",
    "HospitalReadinessConfig",
    "ObservableTriggerAssessment",
    "PolicyMode",
    "PolicySimulationResult",
    "evaluate_observable_triggers",
    "simulate_counterfactual_readiness",
]
