"""Módulo de modelagem probabilística de demanda, calibração conformal e LOS (Feature F04)."""

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
from previsao_leitos.demand.los import SpecialtyLosEstimator
from previsao_leitos.demand.models import (
    QUANTILE_COL_NAMES,
    QUANTILE_LEVELS,
    DemandForecaster,
    NegativeBinomialHurdleForecaster,
    PooledRidgeDeltaForecaster,
    QuantileGBDTForecaster,
)
from previsao_leitos.demand.nowcasting import (
    ChainLadderNowcaster,
    CompetingRisksLosEstimator,
    CompetingRisksSummary,
)
from previsao_leitos.demand.two_stage import (
    OrthogonalTwoStageOccupancyForecaster,
    select_demand_forecaster,
)

__all__ = [
    "QUANTILE_COL_NAMES",
    "QUANTILE_LEVELS",
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
    "NegativeBinomialHurdleForecaster",
    "OrthogonalTwoStageOccupancyForecaster",
    "PooledRidgeDeltaForecaster",
    "QuantileGBDTForecaster",
    "SpecialtyLosEstimator",
    "prepare_global_demand_splits",
    "select_demand_forecaster",
    "validate_canonical_panel_frame",
]
