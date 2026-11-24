"""Arquitetura ortogonal em 2 estágios em taxa de ocupação relativa (F04.1).

Elimina o atraso de fase causado por `y_origin` curto e a dominância de regiões
de grande porte (POA) no MSE bruto:
- Estágio 1 (Basal Lento): sazonalidade + regime viral/SRAG sem `y_origin` curto;
- Estágio 2 (Choque Ex-Ante): HistGradientBoosting monotônico sobre o resíduo em
  taxa de ocupação relativa (`target_y / capacidade`) acoplado às saídas OOF da F03.1.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from previsao_leitos.demand.models import (
    DemandForecaster,
    NegativeBinomialHurdleForecaster,
    _enforce_non_crossing_non_negative,
)
from previsao_leitos.domain import DataContractError

_DEFAULT_BASAL_COLS: tuple[str, ...] = (
    "sin_target_wk",
    "cos_target_wk",
    "srag_viral_confirmado",
    "frac_srag_viral",
)

_DEFAULT_MONO_POS: tuple[str, ...] = (
    "prob_enchente_f03",
    "severidade_hidro_f03",
    "chuva_montante_pond",
    "api_montante_pond",
    "remanso_lagunar_idx",
    "exposicao_aterro_ativa",
    "el_nino_intensity",
)

_DEFAULT_MONO_NEG: tuple[str, ...] = ("frac_floresta",)


class OrthogonalTwoStageOccupancyForecaster:
    """Modelo ortogonal em 2 estágios em taxa relativa (ocupação / capacidade CNES)."""

    def __init__(
        self,
        *,
        capacity_col: str | None = "capacidade_sem",
        basal_cols: Sequence[str] = _DEFAULT_BASAL_COLS,
        monotone_positive_prefixes: Sequence[str] = _DEFAULT_MONO_POS,
        monotone_negative_prefixes: Sequence[str] = _DEFAULT_MONO_NEG,
        basal_alpha: float = 10.0,
        max_iter: int = 120,
        max_depth: int = 4,
        min_samples_leaf: int = 8,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self._capacity_col = capacity_col
        self._basal_cols_cfg = tuple(basal_cols)
        self._mono_pos = tuple(monotone_positive_prefixes)
        self._mono_neg = tuple(monotone_negative_prefixes)
        self._basal_alpha = basal_alpha
        self._max_iter = max_iter
        self._max_depth = max_depth
        self._min_samples_leaf = min_samples_leaf
        self._lr = learning_rate
        self._seed = random_state

        self._basal_cols: tuple[str, ...] = ()
        self._shock_cols: tuple[str, ...] = ()
        self._basal_model: Ridge | None = None
        self._shock_model: HistGradientBoostingRegressor | None = None
        self._resid_sd: float = 0.05
        self._fitted: bool = False

    @property
    def model_name(self) -> str:
        return "OrthogonalTwoStageOccupancy (F04.1)"

    def _extract_capacity(self, df: pl.DataFrame) -> np.ndarray:
        if self._capacity_col is not None and self._capacity_col in df.columns:
            cap = df[self._capacity_col].cast(pl.Float64).fill_null(1.0).to_numpy()
            return np.asarray(np.maximum(cap, 1.0), dtype=np.float64)
        if "y_roll12" in df.columns:
            roll = df["y_roll12"].cast(pl.Float64).fill_null(1.0).to_numpy()
            return np.asarray(np.maximum(roll, 1.0), dtype=np.float64)
        return np.ones(df.height, dtype=np.float64)

    def _monotone_cst(self, cols: Sequence[str]) -> list[int]:
        cst: list[int] = []
        for col in cols:
            if any(col.startswith(p) for p in self._mono_pos):
                cst.append(1)
            elif any(col.startswith(p) for p in self._mono_neg):
                cst.append(-1)
            else:
                cst.append(0)
        return cst

    def fit(
        self,
        train_df: pl.DataFrame,
        feature_cols: Sequence[str],
    ) -> OrthogonalTwoStageOccupancyForecaster:
        """Ajusta Estágio 1 (basal lento) e Estágio 2 (choque ex-ante monotônico)."""
        if train_df.height < 12:
            raise DataContractError("OrthogonalTwoStage requer ao menos 12 linhas de treino.")

        cap = self._extract_capacity(train_df)
        y_rate = train_df["target_y"].cast(pl.Float64).to_numpy() / cap

        # Estágio 1: apenas covariáveis basais lentas (proíbe explicitamente y_origin curto)
        self._basal_cols = tuple(
            c for c in self._basal_cols_cfg if c in train_df.columns and c != "y_origin"
        )
        if self._basal_cols:
            x_basal = (
                train_df.select(list(self._basal_cols))
                .fill_null(0.0)
                .to_numpy()
                .astype(np.float64)
            )
            self._basal_model = Ridge(alpha=self._basal_alpha).fit(x_basal, y_rate)
            basal_hat = np.asarray(self._basal_model.predict(x_basal), dtype=np.float64)
        else:
            self._basal_model = None
            basal_hat = np.full(train_df.height, float(np.mean(y_rate)), dtype=np.float64)

        shock_resid = y_rate - basal_hat

        # Estágio 2: covariáveis físicas/ENSO/F03.1 ortogonais ao basal (sem y_origin)
        shock_candidates = [
            c
            for c in feature_cols
            if c in train_df.columns and c not in self._basal_cols and c != "y_origin"
        ]
        if not shock_candidates:
            raise DataContractError(
                "Nenhuma covariável de choque disponível para o Estágio 2 da F04.1."
            )
        self._shock_cols = tuple(shock_candidates)
        x_shock = (
            train_df.select(list(self._shock_cols))
            .fill_null(0.0)
            .to_numpy()
            .astype(np.float64)
        )
        self._shock_model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=self._max_iter,
            max_depth=self._max_depth,
            min_samples_leaf=self._min_samples_leaf,
            learning_rate=self._lr,
            monotonic_cst=self._monotone_cst(self._shock_cols),
            random_state=self._seed,
        ).fit(x_shock, shock_resid)

        shock_hat = np.asarray(self._shock_model.predict(x_shock), dtype=np.float64)
        final_resid = y_rate - (basal_hat + shock_hat)
        self._resid_sd = (
            max(float(np.std(final_resid, ddof=1)), 1e-3) if len(final_resid) > 1 else 0.05
        )
        self._fitted = True
        return self

    def predict_quantiles(self, df: pl.DataFrame) -> pl.DataFrame:
        """Prevê quantis P10/P50/P90/P95 re-escalados para a unidade original."""
        if not self._fitted or self._shock_model is None:
            raise DataContractError("OrthogonalTwoStageOccupancyForecaster não foi ajustado.")

        cap = self._extract_capacity(df)
        if self._basal_model is not None and self._basal_cols:
            x_basal = (
                df.select(list(self._basal_cols)).fill_null(0.0).to_numpy().astype(np.float64)
            )
            basal_hat = np.asarray(self._basal_model.predict(x_basal), dtype=np.float64)
        else:
            basal_hat = np.zeros(df.height, dtype=np.float64)

        x_shock = (
            df.select(list(self._shock_cols)).fill_null(0.0).to_numpy().astype(np.float64)
        )
        shock_hat = np.asarray(self._shock_model.predict(x_shock), dtype=np.float64)

        rate_p50 = basal_hat + shock_hat
        rate_p10 = rate_p50 - 1.2816 * self._resid_sd
        rate_p90 = rate_p50 + 1.2816 * self._resid_sd
        rate_p95 = rate_p50 + 1.6449 * self._resid_sd

        return _enforce_non_crossing_non_negative(
            rate_p10 * cap,
            rate_p50 * cap,
            rate_p90 * cap,
            rate_p95 * cap,
        )


def select_demand_forecaster(
    train_df: pl.DataFrame,
    *,
    target_col: str = "target_y",
    sparse_zero_threshold: float = 0.35,
    capacity_col: str | None = "capacidade_sem",
    random_state: int = 42,
) -> DemandForecaster:
    """Roteia agravos esparsos (ex.: leptospirose) para NegBinHurdle e densos para F04.1."""
    if target_col not in train_df.columns or train_df.is_empty():
        raise DataContractError(f"Coluna-alvo '{target_col}' ausente ou vazia no roteador.")
    y = train_df[target_col].cast(pl.Float64).to_numpy()
    zero_frac = float(np.mean(y <= 0.0))
    median_val = float(np.median(y))
    if zero_frac >= sparse_zero_threshold or median_val < 2.0:
        return NegativeBinomialHurdleForecaster()
    return OrthogonalTwoStageOccupancyForecaster(
        capacity_col=capacity_col,
        random_state=random_state,
    )
