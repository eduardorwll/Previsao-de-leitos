"""Teste do modelo neural quantílico (spike V2)."""

from __future__ import annotations

import numpy as np
import polars as pl

from previsao_leitos.demand.neural_quantile import NeuralQuantileForecaster


def _frame(n: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 100.0 + 10.0 * x1 - 5.0 * x2 + rng.normal(scale=5.0, size=n)
    return pl.DataFrame({"x1": x1, "x2": x2, "target_y": y})


def test_neural_quantile_fit_predict_ordered_and_nonnegative() -> None:
    train = _frame(500)
    model = NeuralQuantileForecaster(hidden=16, epochs=100, seed=0)
    model.fit(train, ["x1", "x2"])
    pred = model.predict_quantiles(train)

    assert pred.columns == ["pred_p10", "pred_p50", "pred_p90", "pred_p95"]
    assert (pred["pred_p10"] <= pred["pred_p50"]).all()
    assert (pred["pred_p50"] <= pred["pred_p90"]).all()
    assert (pred["pred_p90"] <= pred["pred_p95"]).all()
    assert (pred["pred_p10"] >= 0.0).all()
    # O P50 médio deve ficar próximo da média do alvo (aprendeu o nível).
    assert abs(float(pred["pred_p50"].mean()) - float(train["target_y"].mean())) < 50.0
