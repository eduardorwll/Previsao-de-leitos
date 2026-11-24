"""Calibração probabilística pós-hoc via Temperature + Bias (Platt) Scaling (calibrada em 2023)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from previsao_leitos.domain import DataContractError


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -40.0, 40.0)
    return np.asarray(1.0 / (1.0 + np.exp(-clipped)), dtype=np.float64)


def _binary_nll(logits: np.ndarray, y: np.ndarray, temperature: float, bias: float) -> float:
    probs = np.clip(_sigmoid(logits / temperature + bias), 1e-7, 1.0 - 1e-7)
    return float(-np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))


class TemperatureScaler:
    """Calibra probabilidades pós-hoc `sigmoid(logits / T + b)` minimizando NLL em `cal_year=2023`.

    Restringe `T > 0` (mantendo estrita monotonicidade em `logits`) e corrige o viés de
    prior induzido por `pos_weight` na perda BCE.
    """

    def __init__(self, *, fit_bias: bool = False) -> None:
        self._fit_bias = fit_bias
        self._temperature: float = 1.0
        self._bias: float = 0.0
        self._nll_before: float = 0.0
        self._nll_after: float = 0.0
        self._fitted: bool = False

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def bias(self) -> float:
        return self._bias

    @property
    def cal_nll_before(self) -> float:
        return self._nll_before

    @property
    def cal_nll_after(self) -> float:
        return self._nll_after

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> TemperatureScaler:
        z = np.asarray(logits, dtype=np.float64)
        y = np.asarray(y_true, dtype=np.float64)
        if len(z) < 5:
            raise DataContractError(f"Amostras insuficientes para calibração (n={len(z)}).")

        self._nll_before = _binary_nll(z, y, 1.0, 0.0)

        if self._fit_bias:
            def _obj_affine(params: np.ndarray) -> float:
                return _binary_nll(z, y, float(np.exp(float(params[0]))), float(params[1]))

            res = minimize(_obj_affine, x0=np.array([0.0, 0.0], dtype=np.float64), method="BFGS")
            self._temperature = float(np.clip(np.exp(float(res.x[0])), 0.1, 20.0))
            self._bias = float(res.x[1])
        else:
            def _obj_temp(params: np.ndarray) -> float:
                return _binary_nll(z, y, float(np.exp(float(params[0]))), 0.0)

            res = minimize(_obj_temp, x0=np.array([0.0], dtype=np.float64), method="BFGS")
            self._temperature = float(np.clip(np.exp(float(res.x[0])), 0.1, 20.0))
            self._bias = 0.0

        self._nll_after = _binary_nll(z, y, self._temperature, self._bias)
        self._fitted = True
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise DataContractError("TemperatureScaler precisa ser ajustado antes de prever.")
        return _sigmoid(np.asarray(logits, dtype=np.float64) / self._temperature + self._bias)
