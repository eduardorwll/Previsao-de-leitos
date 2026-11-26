"""Testes do ensemble com pesos calibrados."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from previsao_leitos.demand.ensemble import EnsembleForecaster
from previsao_leitos.demand.models import _enforce_non_crossing_non_negative
from previsao_leitos.domain import DataContractError


class _ConstantForecaster:
    def __init__(self, value: float) -> None:
        self._value = value

    @property
    def model_name(self) -> str:
        return f"const-{self._value}"

    def fit(self, train_df: pl.DataFrame, feature_cols: object) -> _ConstantForecaster:
        del train_df, feature_cols
        return self

    def predict_quantiles(self, df: pl.DataFrame) -> pl.DataFrame:
        base = np.full(df.height, self._value)
        return _enforce_non_crossing_non_negative(base - 1, base, base + 1, base + 2)


def _frame(n: int = 6, target: float = 10.0) -> pl.DataFrame:
    return pl.DataFrame(
        {"target_y": [target] * n, "x": list(range(n))}
    )


def test_ensemble_requires_two_members() -> None:
    with pytest.raises(DataContractError):
        EnsembleForecaster(members=(("a", lambda: _ConstantForecaster(1.0)),))


def test_weights_prefer_the_better_member_on_calibration() -> None:
    model = EnsembleForecaster(
        members=(
            ("bom", lambda: _ConstantForecaster(10.0)),
            ("ruim", lambda: _ConstantForecaster(30.0)),
        ),
        min_weight=0.01,
    )
    train = _frame(target=10.0)
    model.fit(train, ("x",))
    model.fit_calibration(train_df=train, cal_df=_frame(target=10.0))
    assert model.weights["bom"] > model.weights["ruim"]
    assert sum(model.weights.values()) == pytest.approx(1.0)


def test_weight_floor_prevents_zeroing_a_member() -> None:
    model = EnsembleForecaster(
        members=(
            ("bom", lambda: _ConstantForecaster(10.0)),
            ("pessimo", lambda: _ConstantForecaster(10_000.0)),
        ),
        min_weight=0.10,
    )
    train = _frame(target=10.0)
    model.fit(train, ("x",))
    model.fit_calibration(train_df=train, cal_df=_frame(target=10.0))
    assert model.weights["pessimo"] >= 0.10 - 1e-9


def test_prediction_is_bounded_by_members_and_ordered() -> None:
    model = EnsembleForecaster(
        members=(
            ("a", lambda: _ConstantForecaster(10.0)),
            ("b", lambda: _ConstantForecaster(20.0)),
        ),
        min_weight=0.10,
    )
    train = _frame(target=10.0)
    model.fit(train, ("x",))
    model.fit_calibration(train_df=train, cal_df=_frame(target=15.0))
    preds = model.predict_quantiles(_frame(target=15.0))
    p50 = preds["pred_p50"].to_numpy()
    assert np.all(p50 >= 9.0 - 1e-9) and np.all(p50 <= 21.0 + 1e-9)
    assert np.all(preds["pred_p10"].to_numpy() <= p50)
    assert np.all(p50 <= preds["pred_p95"].to_numpy())


def test_equal_weights_before_calibration() -> None:
    model = EnsembleForecaster(
        members=(
            ("a", lambda: _ConstantForecaster(1.0)),
            ("b", lambda: _ConstantForecaster(2.0)),
        )
    )
    model.fit(_frame(), ("x",))
    assert model.weights["a"] == pytest.approx(0.5)
    assert model.weights["b"] == pytest.approx(0.5)


def test_predict_before_fit_raises() -> None:
    model = EnsembleForecaster(
        members=(
            ("a", lambda: _ConstantForecaster(1.0)),
            ("b", lambda: _ConstantForecaster(2.0)),
        )
    )
    with pytest.raises(DataContractError):
        model.predict_quantiles(_frame())
