"""Módulo de hidrologia calibrada (Feature F03: HydroFiLMMonotonicNet + TemperatureScaler)."""

from previsao_leitos.hydrology.calibration import TemperatureScaler
from previsao_leitos.hydrology.dataset import IndexedHydroSequenceDataset
from previsao_leitos.hydrology.model import HydroFiLMMonotonicNet, HydroLightningModule
from previsao_leitos.hydrology.oof import (
    audit_autograd_monotonicity,
    generate_causal_hydro_oof_predictions,
)

__all__ = [
    "HydroFiLMMonotonicNet",
    "HydroLightningModule",
    "IndexedHydroSequenceDataset",
    "TemperatureScaler",
    "audit_autograd_monotonicity",
    "generate_causal_hydro_oof_predictions",
]
