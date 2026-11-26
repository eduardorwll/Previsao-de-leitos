"""Ensemble de modelos de demanda com pesos calibrados fora do ano de avaliação.

Os pesos vêm do **ano de calibração** (`y-1`), nunca do ano avaliado: assim o
ensemble não é ajustado ao resultado que está sendo medido. Sem calibração
disponível, os membros entram com peso igual.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import polars as pl

from previsao_leitos.demand.models import (
    DemandForecaster,
    _enforce_non_crossing_non_negative,
)
from previsao_leitos.domain import DataContractError

_QUANTILE_COLS = ("pred_p10", "pred_p50", "pred_p90", "pred_p95")


class EnsembleForecaster:
    """Média ponderada de previsões probabilísticas, com pesos por MAE de calibração."""

    def __init__(
        self,
        *,
        members: Sequence[tuple[str, Callable[[], DemandForecaster]]],
        min_weight: float = 0.05,
        weight_power: float = 1.0,
    ) -> None:
        if len(members) < 2:
            raise DataContractError("Ensemble exige ao menos dois membros.")
        if min_weight <= 0.0 or min_weight * len(members) >= 1.0:
            raise DataContractError("min_weight inválido para o número de membros.")
        self._member_specs = tuple(members)
        self._min_weight = float(min_weight)
        self._weight_power = float(weight_power)
        self._models: dict[str, DemandForecaster] = {}
        self._weights: dict[str, float] = {}
        self._fitted = False

    @property
    def model_name(self) -> str:
        return "Ensemble(" + "+".join(name for name, _ in self._member_specs) + ")"

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    def fit(
        self, train_df: pl.DataFrame, feature_cols: Sequence[str]
    ) -> EnsembleForecaster:
        self._models = {
            name: factory().fit(train_df, feature_cols)
            for name, factory in self._member_specs
        }
        equal = 1.0 / len(self._models)
        self._weights = {name: equal for name in self._models}
        self._fitted = True
        return self

    def fit_calibration(
        self, *, train_df: pl.DataFrame, cal_df: pl.DataFrame
    ) -> EnsembleForecaster:
        """Recalibra os pesos no ano de calibração (fora do ano avaliado)."""
        if not self._fitted:
            raise DataContractError("Ensemble precisa de `fit` antes de `fit_calibration`.")
        if "target_y" not in cal_df.columns:
            raise DataContractError("`cal_df` sem `target_y` para calibrar o ensemble.")
        observed = cal_df["target_y"].to_numpy()
        errors: dict[str, float] = {}
        for name, model in self._models.items():
            predicted = model.predict_quantiles(cal_df)["pred_p50"].to_numpy()
            errors[name] = float(np.mean(np.abs(predicted - observed)))
        # Pesos ∝ 1/MAE^power, com piso para não zerar um membro.
        raw = {
            name: (1.0 / max(value, 1e-9)) ** self._weight_power
            for name, value in errors.items()
        }
        total = sum(raw.values())
        weights = {name: value / total for name, value in raw.items()}
        # Projeção com piso: quem está abaixo do mínimo sobe para o mínimo e o
        # excedente é redistribuído proporcionalmente entre os demais.
        for _ in range(len(weights) + 1):
            below = [name for name, value in weights.items() if value < self._min_weight]
            if not below:
                break
            deficit = sum(self._min_weight - weights[name] for name in below)
            for name in below:
                weights[name] = self._min_weight
            above = [name for name in weights if name not in below]
            share = sum(weights[name] for name in above)
            if share <= 1e-12:
                break
            for name in above:
                weights[name] -= deficit * weights[name] / share
        self._weights = weights
        return self

    def predict_quantiles(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self._fitted:
            raise DataContractError("EnsembleForecaster não foi ajustado.")
        stacked: dict[str, Any] = {col: np.zeros(df.height) for col in _QUANTILE_COLS}
        for name, model in self._models.items():
            weight = self._weights.get(name, 0.0)
            if weight <= 0.0:
                continue
            predictions = model.predict_quantiles(df)
            for col in _QUANTILE_COLS:
                stacked[col] += weight * predictions[col].to_numpy().astype(np.float64)
        return _enforce_non_crossing_non_negative(
            stacked["pred_p10"],
            stacked["pred_p50"],
            stacked["pred_p90"],
            stacked["pred_p95"],
        )
