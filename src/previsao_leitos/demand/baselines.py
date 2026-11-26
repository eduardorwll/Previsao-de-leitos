"""Baselines probabilísticas obrigatórias: persistência e sazonal-ingênuo.

Ambas produzem os mesmos quantis `P10/P50/P90/P95` dos modelos de demanda, com
dispersão estimada **apenas no treino** (`<= 2022`), para que qualquer comparação
modelo-a-modelo tenha piso honesto e causalmente limpo.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from previsao_leitos.demand.models import _enforce_non_crossing_non_negative
from previsao_leitos.domain import DataContractError

_QUANTILE_LEVELS: tuple[float, ...] = (0.10, 0.50, 0.90, 0.95)


class _AnchorResidualBaseline:
    """Base comum: ponto ancorado em uma coluna causal + quantis dos resíduos de treino."""

    _ANCHOR_COL: str = ""
    _NAME: str = ""

    def __init__(self) -> None:
        self._residual_quantiles: dict[float, float] = {}
        self._fitted: bool = False

    @property
    def model_name(self) -> str:
        return self._NAME

    def _anchor(self, df: pl.DataFrame) -> np.ndarray:
        if self._ANCHOR_COL not in df.columns:
            raise DataContractError(
                f"{self._NAME}: coluna de âncora '{self._ANCHOR_COL}' ausente no painel."
            )
        values = df[self._ANCHOR_COL].cast(pl.Float64, strict=False).fill_null(0.0).to_numpy()
        return np.asarray(np.maximum(values, 0.0), dtype=np.float64)

    def fit(self, train_df: pl.DataFrame, feature_cols: Sequence[str]) -> _AnchorResidualBaseline:
        del feature_cols
        if "target_y" not in train_df.columns:
            raise DataContractError(f"{self._NAME}: 'target_y' ausente no treino.")
        anchor = self._anchor(train_df)
        y_true = train_df["target_y"].cast(pl.Float64, strict=False).to_numpy()
        residuals = np.asarray(y_true, dtype=np.float64) - anchor
        self._residual_quantiles = {
            q: float(np.quantile(residuals, q)) for q in _QUANTILE_LEVELS
        }
        self._fitted = True
        return self

    def predict_quantiles(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self._fitted:
            raise DataContractError(f"{self._NAME} não foi ajustado.")
        anchor = self._anchor(df)
        return _enforce_non_crossing_non_negative(
            *(anchor + self._residual_quantiles[q] for q in _QUANTILE_LEVELS)
        )


class PersistenceForecaster(_AnchorResidualBaseline):
    """Repete a observação da semana de origem (`y_origin`, valor em `t_0`)."""

    _ANCHOR_COL = "y_origin"
    _NAME = "Baseline-Persistencia"


class SeasonalNaiveForecaster(_AnchorResidualBaseline):
    """Repete a mesma semana do ano anterior (`y_seasonal_naive`, defasagem de 52 semanas)."""

    _ANCHOR_COL = "y_seasonal_naive"
    _NAME = "Baseline-Sazonal-Ingenuo"

    def _anchor(self, df: pl.DataFrame) -> np.ndarray:
        if self._ANCHOR_COL in df.columns:
            seasonal = df[self._ANCHOR_COL].cast(pl.Float64, strict=False)
            if "y_origin" in df.columns:
                seasonal = seasonal.fill_null(df["y_origin"].cast(pl.Float64, strict=False))
            values = seasonal.fill_null(0.0).to_numpy()
            return np.asarray(np.maximum(values, 0.0), dtype=np.float64)
        return super()._anchor(df)
