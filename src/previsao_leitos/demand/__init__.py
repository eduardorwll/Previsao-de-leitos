"""Módulo de modelagem probabilística de demanda, calibração conformal e LOS (Feature F04)."""

from previsao_leitos.demand.adaptive_conformal import (
    AgACIConformalCalibrator,
    SPCIConformalCalibrator,
    compute_cnes_capacity_scale,
)
from previsao_leitos.demand.calibration import ConformalFloodCalibrator
from previsao_leitos.demand.datasets import (
    CanonicalPanelSchema,
    GlobalDemandSplits,
    prepare_global_demand_splits,
    validate_canonical_panel_frame,
)
from previsao_leitos.demand.extreme_uncertainty import (
    ExtremeValueGpdTail,
    HierarchicalBasinBayesShrinkage,
    HierarchicalPosteriorSummary,
    MondrianEnsoConformalCalibrator,
)
from previsao_leitos.demand.los import (
    AggregatedCompetingRisksLosEstimator,
    AggregatedCompetingRisksSummary,
    SpecialtyLosEstimator,
)
from previsao_leitos.demand.models import (
    QUANTILE_COL_NAMES,
    QUANTILE_LEVELS,
    DemandForecaster,
    NegativeBinomialHurdleForecaster,
    PooledRidgeDeltaForecaster,
    QuantileGBDTForecaster,
)
from previsao_leitos.demand.neural_quantile import NeuralQuantileForecaster
from previsao_leitos.demand.nowcasting import (
    ChainLadderNowcaster,
    CompetingRisksLosEstimator,
    CompetingRisksSummary,
)
from previsao_leitos.demand.surge_calibration import (
    MonotonicSurgeClassifier,
    RollingBetaCalibrator,
    RollingPlattCalibrator,
    SurgeCalibrationReport,
    build_surge_target,
)
from previsao_leitos.demand.two_stage import (
    OrthogonalTwoStageOccupancyForecaster,
    select_demand_forecaster,
)

__all__ = [
    "QUANTILE_COL_NAMES",
    "QUANTILE_LEVELS",
    "AgACIConformalCalibrator",
    "AggregatedCompetingRisksLosEstimator",
    "AggregatedCompetingRisksSummary",
    "CanonicalPanelSchema",
    "ChainLadderNowcaster",
    "CompetingRisksLosEstimator",
    "CompetingRisksSummary",
    "ConformalFloodCalibrator",
    "DemandForecaster",
    "ExtremeValueGpdTail",
    "GlobalDemandSplits",
    "HierarchicalBasinBayesShrinkage",
    "HierarchicalPosteriorSummary",
    "MondrianEnsoConformalCalibrator",
    "MonotonicSurgeClassifier",
    "NegativeBinomialHurdleForecaster",
    "NeuralQuantileForecaster",
    "OrthogonalTwoStageOccupancyForecaster",
    "PooledRidgeDeltaForecaster",
    "QuantileGBDTForecaster",
    "RollingBetaCalibrator",
    "RollingPlattCalibrator",
    "SPCIConformalCalibrator",
    "SpecialtyLosEstimator",
    "SurgeCalibrationReport",
    "build_surge_target",
    "compute_cnes_capacity_scale",
    "prepare_global_demand_splits",
    "select_demand_forecaster",
    "validate_canonical_panel_frame",
]
