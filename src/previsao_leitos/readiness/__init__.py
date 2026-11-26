"""Módulo de prontidão hospitalar, simulação SimPy e gatilhos observáveis (Feature F05)."""

from previsao_leitos.readiness.pareto import (
    ContingencyPolicyDecision,
    ParetoOptimizationResult,
    ParetoPolicyEvaluation,
    optimize_contingency_pareto_front,
)
from previsao_leitos.readiness.simulator import (
    CounterfactualReadinessComparison,
    CrnScenarioStream,
    HospitalReadinessConfig,
    PolicyMode,
    PolicySimulationResult,
    evaluate_policy_on_crn,
    generate_crn_scenarios,
    simulate_counterfactual_readiness,
)
from previsao_leitos.readiness.triggers import (
    ObservableTriggerAssessment,
    evaluate_observable_triggers,
)

__all__ = [
    "ContingencyPolicyDecision",
    "CounterfactualReadinessComparison",
    "CrnScenarioStream",
    "HospitalReadinessConfig",
    "ObservableTriggerAssessment",
    "ParetoOptimizationResult",
    "ParetoPolicyEvaluation",
    "PolicyMode",
    "PolicySimulationResult",
    "evaluate_observable_triggers",
    "evaluate_policy_on_crn",
    "generate_crn_scenarios",
    "optimize_contingency_pareto_front",
    "simulate_counterfactual_readiness",
]
