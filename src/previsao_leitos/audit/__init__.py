"""Módulo de auditoria estatística, análise contrafactual e relatórios (Feature F06)."""

from previsao_leitos.audit.counterfactual import (
    CounterfactualAccuracyReport,
    NoFloodSyntheticControl,
    decompose_intermunicipal_and_landfill_effect,
    evaluate_counterfactual_accuracy,
)
from previsao_leitos.audit.metrics import (
    BootstrapCI,
    block_bootstrap_ci,
    empirical_coverage,
    evaluate_probabilistic_predictions,
    pinball_loss,
)
from previsao_leitos.audit.plots import (
    plot_c1_counterfactual_trajectory,
    plot_c2_shock_excess_accuracy,
    plot_c3_placebo_validation,
    plot_c4_intermunicipal_landfill_waterfall,
    plot_c5_simpy_contingency,
)
from previsao_leitos.audit.report_builder import (
    DatasetAuditEntry,
    VerificationReportArtifacts,
    build_verification_report,
)

__all__ = [
    "BootstrapCI",
    "CounterfactualAccuracyReport",
    "DatasetAuditEntry",
    "NoFloodSyntheticControl",
    "VerificationReportArtifacts",
    "block_bootstrap_ci",
    "build_verification_report",
    "decompose_intermunicipal_and_landfill_effect",
    "empirical_coverage",
    "evaluate_counterfactual_accuracy",
    "evaluate_probabilistic_predictions",
    "pinball_loss",
    "plot_c1_counterfactual_trajectory",
    "plot_c2_shock_excess_accuracy",
    "plot_c3_placebo_validation",
    "plot_c4_intermunicipal_landfill_waterfall",
    "plot_c5_simpy_contingency",
]
