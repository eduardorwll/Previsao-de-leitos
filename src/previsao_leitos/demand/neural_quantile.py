"""Modelo neural quantílico de demanda (spike V2; análogo de demanda ao `HydroTFTNet`).

Um MLP com cabeça quantílica (P10/P50/P90/P95) treinado com *pinball loss*. O objetivo é
testar, na mesma régua, se uma classe **neural (não-árvore)** supera o GBDT monotônico em
`h = 2` — e, se sim, justificar o investimento no TFT completo (`V2`, `ADR-0004`).

`torch` é importado de forma **lazy** dentro de `fit`/`predict_quantiles` (regra `C-07`:
`import previsao_leitos` sem carregar GPU/torch). Determinístico (`seed`, CPU).

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from previsao_leitos.demand.models import (
    QUANTILE_LEVELS,
    _enforce_non_crossing_non_negative,
)
from previsao_leitos.domain import DataContractError


def _pinball_loss(pred: Any, target: Any, quantiles: Any) -> Any:
    """Pinball (quantile) loss média sobre as `quantiles` colunas previstas."""
    err = target.unsqueeze(1) - pred
    loss = (quantiles * err).clamp_min(0.0) + ((quantiles - 1.0) * err).clamp_min(0.0)
    return loss.mean()


class NeuralQuantileForecaster:
    """MLP quantílico determinístico (CPU) para demanda de leitos — spike V2."""

    def __init__(
        self,
        *,
        hidden: int = 64,
        epochs: int = 400,
        learning_rate: float = 1e-2,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        self._hidden = hidden
        self._epochs = epochs
        self._lr = learning_rate
        self._seed = seed
        self._device = device
        self._feature_cols: tuple[str, ...] = ()
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._y_mean = 0.0
        self._y_std = 1.0
        self._state: dict[str, Any] | None = None

    @property
    def model_name(self) -> str:
        return "NeuralQuantile (MLP)"

    def fit(
        self, train_df: pl.DataFrame, feature_cols: Sequence[str]
    ) -> NeuralQuantileForecaster:
        import torch
        from torch import nn

        self._feature_cols = tuple(feature_cols)
        if not self._feature_cols:
            raise DataContractError("NeuralQuantileForecaster: feature_cols vazio.")
        if "target_y" not in train_df.columns:
            raise DataContractError("NeuralQuantileForecaster: falta 'target_y'.")

        x = train_df.select(list(self._feature_cols)).to_numpy().astype(np.float64)
        y = train_df["target_y"].to_numpy().astype(np.float64)

        self._x_mean = np.nanmean(x, axis=0)
        x_std = np.nanstd(x, axis=0)
        self._x_std = np.where(x_std < 1e-8, 1.0, x_std)
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) or 1.0

        x_norm = np.nan_to_num((x - self._x_mean) / self._x_std)
        y_norm = (y - self._y_mean) / self._y_std

        torch.manual_seed(self._seed)
        device = torch.device(self._device)
        n_quantiles = len(QUANTILE_LEVELS)

        model = nn.Sequential(
            nn.Linear(len(self._feature_cols), self._hidden),
            nn.ReLU(),
            nn.Linear(self._hidden, self._hidden),
            nn.ReLU(),
            nn.Linear(self._hidden, n_quantiles),
        ).to(device)

        x_t = torch.tensor(x_norm, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_norm, dtype=torch.float32, device=device)
        q_t = torch.tensor(QUANTILE_LEVELS, dtype=torch.float32, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)

        model.train()
        for _ in range(self._epochs):
            optimizer.zero_grad()
            pred = model(x_t)
            loss = _pinball_loss(pred, y_t, q_t)
            loss.backward()
            optimizer.step()

        self._state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return self

    def predict_quantiles(self, df: pl.DataFrame) -> pl.DataFrame:
        import torch
        from torch import nn

        if self._state is None or self._x_mean is None or self._x_std is None:
            raise DataContractError("NeuralQuantileForecaster não foi ajustado.")
        if not self._feature_cols:
            raise DataContractError("NeuralQuantileForecaster sem feature_cols.")

        device = torch.device(self._device)
        model = nn.Sequential(
            nn.Linear(len(self._feature_cols), self._hidden),
            nn.ReLU(),
            nn.Linear(self._hidden, self._hidden),
            nn.ReLU(),
            nn.Linear(self._hidden, len(QUANTILE_LEVELS)),
        ).to(device)
        model.load_state_dict({k: v.to(device) for k, v in self._state.items()})
        model.eval()

        x = df.select(list(self._feature_cols)).to_numpy().astype(np.float64)
        x_norm = np.nan_to_num((x - self._x_mean) / self._x_std)
        with torch.no_grad():
            pred = model(torch.tensor(x_norm, dtype=torch.float32, device=device)).cpu().numpy()
        pred = pred * self._y_std + self._y_mean
        return _enforce_non_crossing_non_negative(
            pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        )
