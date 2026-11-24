"""Modelos globais ex-ante de demanda: QuantileGBDT monotônico, NegBin Hurdle e PooledRidge."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

from previsao_leitos.domain import DataContractError

QUANTILE_LEVELS: tuple[float, ...] = (0.10, 0.50, 0.90, 0.95)
QUANTILE_COL_NAMES: tuple[str, ...] = ("pred_p10", "pred_p50", "pred_p90", "pred_p95")


class DemandForecaster(Protocol):
    """Interface unificada de modelos probabilísticos ex-ante (quantis P10/P50/P90/P95)."""

    @property
    def model_name(self) -> str: ...

    def fit(self, train_df: pl.DataFrame, feature_cols: Sequence[str]) -> DemandForecaster: ...

    def predict_quantiles(self, df: pl.DataFrame) -> pl.DataFrame: ...


def _enforce_non_crossing_non_negative(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    q95: np.ndarray,
) -> pl.DataFrame:
    """Garante 0 <= P10 <= P50 <= P90 <= P95 por ordenação monotônica linha a linha."""
    stacked = np.vstack([q10, q50, q90, q95]).T
    stacked = np.maximum(0.0, stacked)
    stacked.sort(axis=1)
    return pl.DataFrame(
        {
            "pred_p10": stacked[:, 0],
            "pred_p50": stacked[:, 1],
            "pred_p90": stacked[:, 2],
            "pred_p95": stacked[:, 3],
        }
    )


class QuantileGBDTForecaster:
    """GBDT Location-Scale com monotonicidade exata (ADR-0001) e quantis P10/P50/P90/P95.

    Usa `HistGradientBoostingRegressor(loss='squared_error', monotonic_cst=...)` (que preserva
    monotonicidade exata em 100% das folhas, evitando a quebra de monotonicidade do
    `update_leaves_values` em `loss='quantile'` do scikit-learn) acoplado a quantis
    residuais heterocedásticos escalados pelo porte regional (`y_roll12`).
    """

    def __init__(
        self,
        *,
        monotone_positive_prefixes: Sequence[str] = (),
        monotone_negative_prefixes: Sequence[str] = (),
        max_iter: int = 160,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        min_samples_leaf: int = 25,
        l2_regularization: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self._pos_prefixes = tuple(monotone_positive_prefixes)
        self._neg_prefixes = tuple(monotone_negative_prefixes)
        self._max_iter = max_iter
        self._max_depth = max_depth
        self._lr = learning_rate
        self._min_leaf = min_samples_leaf
        self._l2 = l2_regularization
        self._seed = random_state
        self._feature_cols: tuple[str, ...] = ()
        self._loc_model: Any = None
        self._scale_quantiles: dict[float, float] = {}

    @property
    def model_name(self) -> str:
        return "QuantileGBDT (Monotonic)"

    def _build_monotonic_cst(self, feature_cols: Sequence[str]) -> list[int]:
        cst: list[int] = []
        for col in feature_cols:
            if any(col.startswith(p) for p in self._pos_prefixes):
                cst.append(1)
            elif any(col.startswith(p) for p in self._neg_prefixes):
                cst.append(-1)
            else:
                cst.append(0)
        return cst

    @staticmethod
    def _regional_scale(df: pl.DataFrame) -> np.ndarray:
        if "y_roll12" in df.columns:
            base = df["y_roll12"].to_numpy().astype(np.float64)
        else:
            base = df["y_origin"].to_numpy().astype(np.float64)
        return np.asarray(np.maximum(1.0, np.sqrt(np.maximum(base, 1.0))), dtype=np.float64)

    def fit(self, train_df: pl.DataFrame, feature_cols: Sequence[str]) -> QuantileGBDTForecaster:
        self._feature_cols = tuple(feature_cols)
        x_tr = train_df.select(list(self._feature_cols)).to_numpy().astype(np.float64)
        y_delta = train_df["target_delta"].to_numpy().astype(np.float64)
        mono_cst = self._build_monotonic_cst(self._feature_cols)

        est = HistGradientBoostingRegressor(
            loss="squared_error",
            monotonic_cst=mono_cst,
            max_iter=self._max_iter,
            max_depth=self._max_depth,
            learning_rate=self._lr,
            min_samples_leaf=self._min_leaf,
            l2_regularization=self._l2,
            random_state=self._seed,
        )
        est.fit(x_tr, y_delta)
        self._loc_model = est

        pred_tr = np.asarray(est.predict(x_tr), dtype=np.float64)
        scale_tr = self._regional_scale(train_df)
        std_res = (y_delta - pred_tr) / scale_tr
        self._scale_quantiles = {q: float(np.quantile(std_res, q)) for q in QUANTILE_LEVELS}
        return self

    def predict_quantiles(self, df: pl.DataFrame) -> pl.DataFrame:
        if self._loc_model is None:
            raise DataContractError("QuantileGBDTForecaster precisa ser ajustado antes de prever.")
        x_ev = df.select(list(self._feature_cols)).to_numpy().astype(np.float64)
        y_orig = df["y_origin"].to_numpy().astype(np.float64)
        loc = y_orig + np.asarray(self._loc_model.predict(x_ev), dtype=np.float64)
        scale = self._regional_scale(df)
        return _enforce_non_crossing_non_negative(
            loc + scale * self._scale_quantiles[0.10],
            loc + scale * self._scale_quantiles[0.50],
            loc + scale * self._scale_quantiles[0.90],
            loc + scale * self._scale_quantiles[0.95],
        )


class NegativeBinomialHurdleForecaster:
    """Modelo de Contagem Hurdle / Binomial Negativa (statsmodels) para agravos esparsos (Onda 2).

    Combina GLM Binomial (probabilidade de contagem positiva) e GLM Binomial Negativa
    (link logarítmico) para produzir quantis preditivos analíticos (P10/P50/P90/P95).
    """

    def __init__(self, *, alpha_dispersion: float = 0.75, ridge_l2: float = 1e-2) -> None:
        self._alpha_disp = alpha_dispersion
        self._ridge_l2 = ridge_l2
        self._feature_cols: tuple[str, ...] = ()
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._zero_res: Any = None
        self._count_res: Any = None
        self._fallback_rate: float = 0.0

    @property
    def model_name(self) -> str:
        return "NegBinHurdle (GLM)"

    def _scale(self, x: np.ndarray) -> np.ndarray:
        if self._mean is None or self._std is None:
            raise DataContractError("Escalador não inicializado em NegativeBinomialHurdle.")
        z = (x - self._mean) / self._std
        z = np.clip(z, -6.0, 6.0)
        return np.asarray(sm.add_constant(z, has_constant="add"), dtype=np.float64)

    def fit(
        self, train_df: pl.DataFrame, feature_cols: Sequence[str]
    ) -> NegativeBinomialHurdleForecaster:
        self._feature_cols = tuple(feature_cols)
        x_tr = train_df.select(list(self._feature_cols)).to_numpy().astype(np.float64)
        y_tr = np.maximum(0.0, train_df["target_y"].to_numpy().astype(np.float64))

        self._mean = np.mean(x_tr, axis=0)
        std = np.std(x_tr, axis=0)
        self._std = np.where(std < 1e-8, 1.0, std)
        self._fallback_rate = float(np.mean(y_tr))

        x_mat = self._scale(x_tr)
        y_pos = (y_tr > 0.0).astype(np.float64)

        if 0.0 < float(np.mean(y_pos)) < 1.0:
            binom_glm = sm.GLM(y_pos, x_mat, family=sm.families.Binomial())
            self._zero_res = binom_glm.fit_regularized(alpha=self._ridge_l2, L1_wt=0.0)
        else:
            self._zero_res = None

        nb_glm = sm.GLM(
            y_tr,
            x_mat,
            family=sm.families.NegativeBinomial(alpha=self._alpha_disp),
        )
        self._count_res = nb_glm.fit_regularized(alpha=self._ridge_l2, L1_wt=0.0)
        return self

    def predict_quantiles(self, df: pl.DataFrame) -> pl.DataFrame:
        if self._count_res is None:
            raise DataContractError("NegativeBinomialHurdleForecaster não foi ajustado.")
        x_ev = df.select(list(self._feature_cols)).to_numpy().astype(np.float64)
        x_mat = self._scale(x_ev)

        count_params = np.asarray(self._count_res.params, dtype=np.float64)
        eta_count = np.clip(x_mat @ count_params, -10.0, 9.0)
        mu_raw = np.exp(eta_count)
        if self._zero_res is not None:
            zero_params = np.asarray(self._zero_res.params, dtype=np.float64)
            eta_z = np.clip(x_mat @ zero_params, -10.0, 10.0)
            p_pos = 1.0 / (1.0 + np.exp(-eta_z))
            mu = np.maximum(1e-6, p_pos * mu_raw)
        else:
            mu = np.maximum(1e-6, mu_raw)

        # Parametrização scipy.stats.nbinom(n=1/alpha, p=n/(n+mu))
        n_param = 1.0 / max(self._alpha_disp, 1e-4)
        p_param = n_param / (n_param + mu)
        q10 = np.asarray(stats.nbinom.ppf(0.10, n_param, p_param), dtype=np.float64)
        q50 = np.asarray(stats.nbinom.ppf(0.50, n_param, p_param), dtype=np.float64)
        q90 = np.asarray(stats.nbinom.ppf(0.90, n_param, p_param), dtype=np.float64)
        q95 = np.asarray(stats.nbinom.ppf(0.95, n_param, p_param), dtype=np.float64)
        return _enforce_non_crossing_non_negative(q10, q50, q90, q95)


class PooledRidgeDeltaForecaster:
    """Baseline linear global (Ridge sobre delta) com quantis empíricos dos resíduos de treino."""

    def __init__(self, *, alpha: float = 100.0) -> None:
        self._alpha = alpha
        self._feature_cols: tuple[str, ...] = ()
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._coef: np.ndarray | None = None
        self._res_quantiles: dict[float, float] = {}

    @property
    def model_name(self) -> str:
        return "PooledRidge (Linear)"

    def fit(
        self, train_df: pl.DataFrame, feature_cols: Sequence[str]
    ) -> PooledRidgeDeltaForecaster:
        self._feature_cols = tuple(feature_cols)
        x_tr = train_df.select(list(self._feature_cols)).to_numpy().astype(np.float64)
        y_delta = train_df["target_delta"].to_numpy().astype(np.float64)

        self._mean = np.mean(x_tr, axis=0)
        std = np.std(x_tr, axis=0)
        self._std = np.where(std < 1e-8, 1.0, std)
        z_tr = (x_tr - self._mean) / self._std
        a_tr = np.hstack([np.ones((z_tr.shape[0], 1), dtype=np.float64), z_tr])

        reg = np.eye(a_tr.shape[1], dtype=np.float64) * self._alpha
        reg[0, 0] = 0.0
        self._coef = np.linalg.solve(a_tr.T @ a_tr + reg, a_tr.T @ y_delta)
        residuals = y_delta - (a_tr @ self._coef)
        self._res_quantiles = {q: float(np.quantile(residuals, q)) for q in QUANTILE_LEVELS}
        return self

    def predict_quantiles(self, df: pl.DataFrame) -> pl.DataFrame:
        if self._coef is None or self._mean is None or self._std is None:
            raise DataContractError("PooledRidgeDeltaForecaster não foi ajustado.")
        x_ev = df.select(list(self._feature_cols)).to_numpy().astype(np.float64)
        y_orig = df["y_origin"].to_numpy().astype(np.float64)
        z_ev = (x_ev - self._mean) / self._std
        a_ev = np.hstack([np.ones((z_ev.shape[0], 1), dtype=np.float64), z_ev])
        center = y_orig + (a_ev @ self._coef)
        return _enforce_non_crossing_non_negative(
            center + self._res_quantiles[0.10],
            center + self._res_quantiles[0.50],
            center + self._res_quantiles[0.90],
            center + self._res_quantiles[0.95],
        )
