"""Módulo de hidrologia calibrada (Feature F03/F03.1: HydroFiLM-ENSO + Accelerator + OOF)."""

from previsao_leitos.hydrology.calibration import TemperatureScaler
from previsao_leitos.hydrology.dataset import IndexedHydroSequenceDataset
from previsao_leitos.hydrology.device import (
    AcceleratorSpec,
    release_gpu_memory,
    resolve_accelerator,
)
from previsao_leitos.hydrology.model import HydroFiLMMonotonicNet, HydroLightningModule
from previsao_leitos.hydrology.oof import (
    NeuralHydroOofReport,
    audit_autograd_monotonicity,
    audit_dataset_autograd_monotonicity,
    generate_causal_hydro_oof_predictions,
    generate_neural_hydro_oof_predictions,
)

__all__ = [
    "AcceleratorSpec",
    "HydroFiLMMonotonicNet",
    "HydroLightningModule",
    "IndexedHydroSequenceDataset",
    "NeuralHydroOofReport",
    "TemperatureScaler",
    "audit_autograd_monotonicity",
    "audit_dataset_autograd_monotonicity",
    "generate_causal_hydro_oof_predictions",
    "generate_neural_hydro_oof_predictions",
    "release_gpu_memory",
    "resolve_accelerator",
]
