"""Calibração conformal adaptativa normalizada por capacidade CNES: AgACI vs. SPCI (Task 1.2.4).

Implementa dois calibradores conformais modernos sobre qualquer `DemandForecaster`:
1. `AgACIConformalCalibrator` (Aggregated Adaptive Conformal Inference - Zaffran et al., ICML 2022):
   combina múltiplos especialistas ACI (`gammas`) via agregação exponencial de perda pinball
   com atraso causal exato de `feedback_lag_weeks = h` semanas (nunca espiando `y_{t+h}` em `t`).
2. `SPCIConformalCalibrator` (Sequential Predictive Conformal Inference - Xu & Xie, ICML 2023):
   modela os quantis condicionais dos resíduos normalizados por capacidade CNES via
   Quantile Random Forest (pesos empíricos de folhas sobre lags de resíduos estritamente observados
   até a semana de origem `t`), preservando `0,00%` de violações monotônicas hidrológicas.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor

from previsao_leitos.demand.models import DemandForecaster, _enforce_non_crossing_non_negative
from previsao_leitos.domain import DataContractError


def compute_cnes_capacity_scale(
    df: pl.DataFrame,
    *,
    capacity_col: str = "capacidade_sem",
    fallback_col: str = "y_roll12",
    min_scale: float = 1.0,
) -> np.ndarray:
    """Calcula escala de normalização CNES por linha (homogeneíza POA vs. interior).

    Usa `max(capacidade_sem, fallback, min_scale)` garantindo finitude e positividade estrita.
    """
    n = df.height
    cap = (
        df[capacity_col].to_numpy().astype(np.float64)
        if capacity_col in df.columns
        else np.zeros(n, dtype=np.float64)
    )
    cap = np.where(np.isfinite(cap) & (cap > 0.0), cap, 0.0)
    fb = (
        df[fallback_col].to_numpy().astype(np.float64)
        if fallback_col in df.columns
        else np.ones(n, dtype=np.float64)
    )
    fb = np.where(np.isfinite(fb) & (fb > 0.0), fb, min_scale)
    scale = np.where(cap > min_scale, cap, np.maximum(fb, min_scale))
    return np.asarray(np.maximum(scale, min_scale), dtype=np.float64)


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    arr = np.asarray(scores, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    a = float(np.clip(alpha, 1e-4, 1.0 - 1e-4))
    q_idx = min(1.0, max(0.0, ((1.0 - a) * (arr.size + 1.0)) / arr.size))
    return float(np.quantile(arr, q_idx, method="higher"))


def _pinball_loss_scalar(y_val: float, q_hat: float, level: float) -> float:
    err = y_val - q_hat
    return float(level * err) if err >= 0.0 else float((level - 1.0) * err)


def _time_order_keys(
    df: pl.DataFrame,
    *,
    year_col: str = "target_ano_epi",
    week_col: str = "target_semana_epi",
) -> np.ndarray:
    if year_col in df.columns and week_col in df.columns:
        yrs = df[year_col].to_numpy().astype(np.int64)
        wks = df[week_col].to_numpy().astype(np.int64)
        return np.asarray(yrs * 100 + wks, dtype=np.int64)
    return np.arange(df.height, dtype=np.int64)


class AgACIConformalCalibrator:
    """Aggregated Adaptive Conformal Inference (Zaffran et al. 2022) normalizado por CNES.

    - Em `predict_quantiles(df)` (modo congelado/stateless): aplica os quantis calibrados em
      `cal_df` normalizados pela capacidade CNES (seguro para oráculos `do()`/El Niño).
    - Em `predict_quantiles_online(eval_df)`: executa adaptação sequencial causal por semana
      respeitando rigorosamente `feedback_lag_weeks` (ex.: `h = 2`), agregando especialistas
      `gammas` por pesos exponenciais da perda pinball.
    """

    def __init__(
        self,
        *,
        base_model: DemandForecaster,
        gammas: Sequence[float] = (0.005, 0.01, 0.025, 0.05, 0.10),
        feedback_lag_weeks: int = 2,
        window_size: int = 360,
        capacity_col: str = "capacidade_sem",
        fallback_col: str = "y_roll12",
        eta: float = 1.5,
    ) -> None:
        if not gammas or any(g <= 0.0 for g in gammas):
            raise DataContractError("gammas em AgACIConformalCalibrator devem ser > 0.")
        if feedback_lag_weeks < 1:
            raise DataContractError("feedback_lag_weeks deve ser >= 1 para preservar causalidade.")
        self._base = base_model
        self._gammas = tuple(float(g) for g in gammas)
        self._lag_weeks = int(feedback_lag_weeks)
        self._window = int(window_size)
        self._cap_col = capacity_col
        self._fb_col = fallback_col
        self._eta = float(eta)

        self._hist_low_10: list[float] = []
        self._hist_high_90: list[float] = []
        self._hist_high_95: list[float] = []
        self._q_low_10: float = 0.0
        self._q_high_90: float = 0.0
        self._q_high_95: float = 0.0
        self._fitted: bool = False

    @property
    def model_name(self) -> str:
        return f"{self._base.model_name} + AgACI(CNES)"

    def fit(self, train_df: pl.DataFrame, feature_cols: Sequence[str]) -> AgACIConformalCalibrator:
        self._base.fit(train_df, feature_cols)
        return self

    def fit_calibration(
        self,
        *,
        cal_df: pl.DataFrame,
        target_col: str = "target_y",
    ) -> AgACIConformalCalibrator:
        """Ajusta janela inicial de escores normalizados por CNES sobre `cal_df` (<= 2023)."""
        if cal_df.height < 10:
            raise DataContractError(f"cal_df insuficiente para AgACI (n={cal_df.height}).")
        raw = self._base.predict_quantiles(cal_df)
        y = cal_df[target_col].to_numpy().astype(np.float64)
        scale = compute_cnes_capacity_scale(
            cal_df, capacity_col=self._cap_col, fallback_col=self._fb_col
        )
        p10 = raw["pred_p10"].to_numpy().astype(np.float64)
        p90 = raw["pred_p90"].to_numpy().astype(np.float64)
        p95 = raw["pred_p95"].to_numpy().astype(np.float64)

        s_low_10 = (p10 - y) / scale
        s_high_90 = (y - p90) / scale
        s_high_95 = (y - p95) / scale

        self._hist_low_10 = list(s_low_10[-self._window :])
        self._hist_high_90 = list(s_high_90[-self._window :])
        self._hist_high_95 = list(s_high_95[-self._window :])

        self._q_low_10 = max(0.0, _finite_sample_quantile(s_low_10, alpha=0.10))
        self._q_high_90 = max(0.0, _finite_sample_quantile(s_high_90, alpha=0.10))
        self._q_high_95 = max(0.0, _finite_sample_quantile(s_high_95, alpha=0.05))
        self._fitted = True
        return self

    def predict_quantiles(self, df: pl.DataFrame) -> pl.DataFrame:
        """Emite quantis calibrados (estado congelado, seguro para cenários `do()`)."""
        if not self._fitted:
            raise DataContractError(
                "AgACIConformalCalibrator precisa de fit_calibration antes de prever."
            )
        raw = self._base.predict_quantiles(df)
        scale = compute_cnes_capacity_scale(
            df, capacity_col=self._cap_col, fallback_col=self._fb_col
        )
        p10 = raw["pred_p10"].to_numpy().astype(np.float64) - scale * self._q_low_10
        p50 = raw["pred_p50"].to_numpy().astype(np.float64)
        p90 = raw["pred_p90"].to_numpy().astype(np.float64) + scale * self._q_high_90
        p95 = raw["pred_p95"].to_numpy().astype(np.float64) + scale * self._q_high_95
        return _enforce_non_crossing_non_negative(p10, p50, p90, p95)

    def predict_quantiles_online(
        self,
        eval_df: pl.DataFrame,
        *,
        target_col: str = "target_y",
        year_col: str = "target_ano_epi",
        week_col: str = "target_semana_epi",
    ) -> pl.DataFrame:
        """Adaptação sequencial causal semana a semana respeitando atraso `feedback_lag_weeks`.

        Na semana `t_idx`, somente observações reais com índice `<= t_idx - feedback_lag_weeks`
        são reveladas para atualizar os especialistas ACI e seus pesos BOA.
        """
        if not self._fitted:
            raise DataContractError(
                "AgACIConformalCalibrator precisa de fit_calibration antes de prever."
            )
        if target_col not in eval_df.columns:
            raise DataContractError(
                f"Coluna '{target_col}' obrigatória em predict_quantiles_online."
            )

        raw = self._base.predict_quantiles(eval_df)
        scale = compute_cnes_capacity_scale(
            eval_df, capacity_col=self._cap_col, fallback_col=self._fb_col
        )
        p10_raw = raw["pred_p10"].to_numpy().astype(np.float64)
        p50_raw = raw["pred_p50"].to_numpy().astype(np.float64)
        p90_raw = raw["pred_p90"].to_numpy().astype(np.float64)
        p95_raw = raw["pred_p95"].to_numpy().astype(np.float64)
        y_true = eval_df[target_col].to_numpy().astype(np.float64)

        t_keys = _time_order_keys(eval_df, year_col=year_col, week_col=week_col)
        unique_times = np.unique(t_keys)
        k_exp = len(self._gammas)

        # Estado isolado por execução (não muta estado congelado de fit_calibration)
        buf_low = list(self._hist_low_10)
        buf_hi90 = list(self._hist_high_90)
        buf_hi95 = list(self._hist_high_95)

        alpha_low = np.full(k_exp, 0.10, dtype=np.float64)
        alpha_hi90 = np.full(k_exp, 0.10, dtype=np.float64)
        alpha_hi95 = np.full(k_exp, 0.05, dtype=np.float64)

        cum_loss_low = np.zeros(k_exp, dtype=np.float64)
        cum_loss_hi90 = np.zeros(k_exp, dtype=np.float64)
        cum_loss_hi95 = np.zeros(k_exp, dtype=np.float64)

        out_p10 = np.empty_like(p10_raw)
        out_p50 = p50_raw.copy()
        out_p90 = np.empty_like(p90_raw)
        out_p95 = np.empty_like(p95_raw)

        # Armazena previsões dos especialistas para atualização causal atrasada em h semanas
        pending: list[
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []

        for step_idx, tk in enumerate(unique_times):
            # 1. Revela estritamente observações já maturadas (atraso >= feedback_lag_weeks)
            if step_idx >= self._lag_weeks:
                (
                    obs_low,
                    obs_hi90,
                    obs_hi95,
                    q_exp_low,
                    q_exp_hi90,
                    q_exp_hi95,
                ) = pending[step_idx - self._lag_weeks]

                for k, gamma in enumerate(self._gammas):
                    err_low = float(np.mean(obs_low > q_exp_low[k]))
                    err_hi90 = float(np.mean(obs_hi90 > q_exp_hi90[k]))
                    err_hi95 = float(np.mean(obs_hi95 > q_exp_hi95[k]))

                    alpha_low[k] = float(
                        np.clip(alpha_low[k] + gamma * (0.10 - err_low), 0.005, 0.45)
                    )
                    alpha_hi90[k] = float(
                        np.clip(alpha_hi90[k] + gamma * (0.10 - err_hi90), 0.005, 0.45)
                    )
                    alpha_hi95[k] = float(
                        np.clip(alpha_hi95[k] + gamma * (0.05 - err_hi95), 0.002, 0.30)
                    )

                    cum_loss_low[k] += float(
                        np.mean(
                            [
                                _pinball_loss_scalar(float(s), float(q_exp_low[k]), 0.90)
                                for s in obs_low
                            ]
                        )
                    )
                    cum_loss_hi90[k] += float(
                        np.mean(
                            [
                                _pinball_loss_scalar(float(s), float(q_exp_hi90[k]), 0.90)
                                for s in obs_hi90
                            ]
                        )
                    )
                    cum_loss_hi95[k] += float(
                        np.mean(
                            [
                                _pinball_loss_scalar(float(s), float(q_exp_hi95[k]), 0.95)
                                for s in obs_hi95
                            ]
                        )
                    )

                buf_low.extend(obs_low.tolist())
                buf_hi90.extend(obs_hi90.tolist())
                buf_hi95.extend(obs_hi95.tolist())
                if len(buf_low) > self._window:
                    buf_low = buf_low[-self._window :]
                    buf_hi90 = buf_hi90[-self._window :]
                    buf_hi95 = buf_hi95[-self._window :]

            # 2. Calcula quantis de cada especialista e agrega via pesos exponenciais BOA
            arr_low = np.asarray(buf_low, dtype=np.float64)
            arr_hi90 = np.asarray(buf_hi90, dtype=np.float64)
            arr_hi95 = np.asarray(buf_hi95, dtype=np.float64)

            q_exp_low_now = np.array(
                [max(0.0, _finite_sample_quantile(arr_low, float(a))) for a in alpha_low],
                dtype=np.float64,
            )
            q_exp_hi90_now = np.array(
                [max(0.0, _finite_sample_quantile(arr_hi90, float(a))) for a in alpha_hi90],
                dtype=np.float64,
            )
            q_exp_hi95_now = np.array(
                [max(0.0, _finite_sample_quantile(arr_hi95, float(a))) for a in alpha_hi95],
                dtype=np.float64,
            )

            def _exp_weights(losses: np.ndarray) -> np.ndarray:
                shifted = -self._eta * (losses - float(np.min(losses)))
                w = np.exp(np.clip(shifted, -50.0, 0.0))
                return np.asarray(w / float(np.sum(w)), dtype=np.float64)

            q_low_agg = float(np.dot(_exp_weights(cum_loss_low), q_exp_low_now))
            q_hi90_agg = float(np.dot(_exp_weights(cum_loss_hi90), q_exp_hi90_now))
            q_hi95_agg = float(np.dot(_exp_weights(cum_loss_hi95), q_exp_hi95_now))

            mask = t_keys == tk
            sc_m = scale[mask]
            out_p10[mask] = p10_raw[mask] - sc_m * q_low_agg
            out_p90[mask] = p90_raw[mask] + sc_m * q_hi90_agg
            out_p95[mask] = p95_raw[mask] + sc_m * q_hi95_agg

            # 3. Enfileira escores reais desta semana para revelação futura após `lag_weeks`
            pending.append(
                (
                    (p10_raw[mask] - y_true[mask]) / sc_m,
                    (y_true[mask] - p90_raw[mask]) / sc_m,
                    (y_true[mask] - p95_raw[mask]) / sc_m,
                    q_exp_low_now,
                    q_exp_hi90_now,
                    q_exp_hi95_now,
                )
            )

        return _enforce_non_crossing_non_negative(out_p10, out_p50, out_p90, out_p95)


class SPCIConformalCalibrator:
    """Sequential Predictive Conformal Inference (Xu & Xie 2023) via QRF em resíduos CNES.

    Estima os quantis condicionais `(0.10, 0.90, 0.95)` do resíduo assinado normalizado por
    capacidade CNES `r_t = (y_t - pred_p50_t) / cap_t` a partir da janela causal de resíduos
    passados `(r_{t-h}, r_{t-h-1}, ..., r_{t-h-W+1})` via Quantile Random Forest (pesos de folhas).
    Como os preditores do QRF dependem apenas de resíduos históricos observados (e não das
    covariáveis hidrológicas de `t_0`), a monotonicidade hidrológica do modelo base é preservada
    em `100%` (`0,00%` violações).
    """

    def __init__(
        self,
        *,
        base_model: DemandForecaster,
        residual_lags: int = 4,
        feedback_lag_weeks: int = 2,
        n_estimators: int = 60,
        min_samples_leaf: int = 12,
        capacity_col: str = "capacidade_sem",
        fallback_col: str = "y_roll12",
        entity_col: str = "nome_regiao_saude",
        seed: int = 42,
    ) -> None:
        if residual_lags < 1 or feedback_lag_weeks < 1:
            raise DataContractError("residual_lags e feedback_lag_weeks devem ser >= 1.")
        self._base = base_model
        self._n_lags = int(residual_lags)
        self._lag_weeks = int(feedback_lag_weeks)
        self._n_estimators = int(n_estimators)
        self._min_leaf = int(min_samples_leaf)
        self._cap_col = capacity_col
        self._fb_col = fallback_col
        self._entity_col = entity_col
        self._seed = int(seed)

        self._rf: RandomForestRegressor | None = None
        self._train_leaves: np.ndarray | None = None
        self._train_res: np.ndarray | None = None
        self._entity_tail_res: dict[str, list[float]] = {}
        self._global_q: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._fitted: bool = False

    @property
    def model_name(self) -> str:
        return f"{self._base.model_name} + SPCI-QRF(CNES)"

    def fit(self, train_df: pl.DataFrame, feature_cols: Sequence[str]) -> SPCIConformalCalibrator:
        self._base.fit(train_df, feature_cols)
        return self

    def _build_lag_matrix_from_residuals(
        self,
        entities: list[str],
        t_keys: np.ndarray,
        residuals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
        """Constrói matriz causal `(X_lag, y_res)` com atraso mínimo `feedback_lag_weeks`."""
        order = np.argsort(t_keys, kind="stable")
        by_entity: dict[str, list[float]] = {}
        x_rows: list[list[float]] = []
        y_rows: list[float] = []

        for idx in order:
            ent = entities[int(idx)]
            hist = by_entity.setdefault(ent, [])
            if len(hist) >= self._lag_weeks + self._n_lags:
                # Lags estritamente observados <= t - feedback_lag_weeks
                end_pos = len(hist) - self._lag_weeks
                lag_vec = [hist[end_pos - 1 - j] for j in range(self._n_lags)]
                x_rows.append(lag_vec)
                y_rows.append(float(residuals[int(idx)]))
            hist.append(float(residuals[int(idx)]))

        if not x_rows:
            return (
                np.empty((0, self._n_lags), dtype=np.float64),
                np.empty(0, dtype=np.float64),
                by_entity,
            )
        return np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.float64), by_entity

    def _predict_qrf_quantiles(
        self, x_query: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calcula quantis empíricos ponderados pelas folhas do Random Forest (Meinshausen 2006)."""
        n_q = x_query.shape[0]
        if self._rf is None or self._train_leaves is None or self._train_res is None:
            q10, q90, q95 = self._global_q
            return (
                np.full(n_q, q10, dtype=np.float64),
                np.full(n_q, q90, dtype=np.float64),
                np.full(n_q, q95, dtype=np.float64),
            )
        q_leaves = self._rf.apply(x_query)
        tr_leaves = self._train_leaves
        tr_res = self._train_res
        n_trees = tr_leaves.shape[1]

        out_10 = np.empty(n_q, dtype=np.float64)
        out_90 = np.empty(n_q, dtype=np.float64)
        out_95 = np.empty(n_q, dtype=np.float64)

        for i in range(n_q):
            # Vetor de pesos por observação de treino proporcional à co-ocorrência nas folhas
            matches = tr_leaves == q_leaves[i : i + 1, :]
            leaf_counts = np.maximum(1.0, np.sum(matches, axis=0, dtype=np.float64))
            weights = np.sum(matches / leaf_counts[None, :], axis=1) / float(n_trees)
            w_sum = float(np.sum(weights))
            if w_sum <= 1e-12:
                out_10[i], out_90[i], out_95[i] = self._global_q
                continue
            weights /= w_sum
            ord_idx = np.argsort(tr_res)
            sorted_r = tr_res[ord_idx]
            cdf = np.cumsum(weights[ord_idx])
            out_10[i] = float(sorted_r[min(len(sorted_r) - 1, int(np.searchsorted(cdf, 0.10)))])
            out_90[i] = float(sorted_r[min(len(sorted_r) - 1, int(np.searchsorted(cdf, 0.90)))])
            out_95[i] = float(sorted_r[min(len(sorted_r) - 1, int(np.searchsorted(cdf, 0.95)))])
        return out_10, out_90, out_95

    def fit_calibration(
        self,
        *,
        cal_df: pl.DataFrame,
        target_col: str = "target_y",
        year_col: str = "target_ano_epi",
        week_col: str = "target_semana_epi",
    ) -> SPCIConformalCalibrator:
        """Ajusta QRF sobre resíduos normalizados por CNES em `cal_df` (<= 2023)."""
        if cal_df.height < 10:
            raise DataContractError(f"cal_df insuficiente para SPCI (n={cal_df.height}).")
        raw = self._base.predict_quantiles(cal_df)
        y = cal_df[target_col].to_numpy().astype(np.float64)
        p50 = raw["pred_p50"].to_numpy().astype(np.float64)
        scale = compute_cnes_capacity_scale(
            cal_df, capacity_col=self._cap_col, fallback_col=self._fb_col
        )
        res = (y - p50) / scale

        q10 = float(np.quantile(res, 0.10))
        q90 = float(np.quantile(res, 0.90))
        q95 = float(np.quantile(res, 0.95))
        self._global_q = (q10, q90, q95)

        entities = (
            cal_df[self._entity_col].cast(pl.String).to_list()
            if self._entity_col in cal_df.columns
            else ["_global_"] * cal_df.height
        )
        t_keys = _time_order_keys(cal_df, year_col=year_col, week_col=week_col)
        x_lag, y_res, hist_map = self._build_lag_matrix_from_residuals(entities, t_keys, res)
        self._entity_tail_res = {
            k: v[-(self._n_lags + self._lag_weeks + 4) :] for k, v in hist_map.items()
        }

        if x_lag.shape[0] >= max(20, 2 * self._min_leaf):
            rf = RandomForestRegressor(
                n_estimators=self._n_estimators,
                min_samples_leaf=self._min_leaf,
                random_state=self._seed,
                n_jobs=1,
            )
            rf.fit(x_lag, y_res)
            self._rf = rf
            self._train_leaves = rf.apply(x_lag)
            self._train_res = y_res
        else:
            self._rf = None
            self._train_leaves = None
            self._train_res = None

        self._fitted = True
        return self

    def _query_vector_for_entity(self, hist: list[float]) -> list[float]:
        if len(hist) >= self._n_lags:
            return [hist[-1 - j] for j in range(self._n_lags)]
        pad = [0.0] * (self._n_lags - len(hist))
        return list(reversed(hist)) + pad

    def predict_quantiles(self, df: pl.DataFrame) -> pl.DataFrame:
        """Emite quantis SPCI usando cauda congelada de `cal_df` (seguro para `do()`/oráculos)."""
        if not self._fitted:
            raise DataContractError(
                "SPCIConformalCalibrator precisa de fit_calibration antes de prever."
            )
        raw = self._base.predict_quantiles(df)
        p50 = raw["pred_p50"].to_numpy().astype(np.float64)
        scale = compute_cnes_capacity_scale(
            df, capacity_col=self._cap_col, fallback_col=self._fb_col
        )
        entities = (
            df[self._entity_col].cast(pl.String).to_list()
            if self._entity_col in df.columns
            else ["_global_"] * df.height
        )
        x_q = np.asarray(
            [self._query_vector_for_entity(self._entity_tail_res.get(e, [])) for e in entities],
            dtype=np.float64,
        )
        q10, q90, q95 = self._predict_qrf_quantiles(x_q)
        return _enforce_non_crossing_non_negative(
            p50 + scale * q10,
            p50,
            p50 + scale * q90,
            p50 + scale * q95,
        )

    def predict_quantiles_online(
        self,
        eval_df: pl.DataFrame,
        *,
        target_col: str = "target_y",
        year_col: str = "target_ano_epi",
        week_col: str = "target_semana_epi",
    ) -> pl.DataFrame:
        """Predição sequencial causal SPCI revelando resíduos apenas após `feedback_lag_weeks`."""
        if not self._fitted:
            raise DataContractError(
                "SPCIConformalCalibrator precisa de fit_calibration antes de prever."
            )
        if target_col not in eval_df.columns:
            raise DataContractError(
                f"Coluna '{target_col}' obrigatória em predict_quantiles_online."
            )

        raw = self._base.predict_quantiles(eval_df)
        p50 = raw["pred_p50"].to_numpy().astype(np.float64)
        y_true = eval_df[target_col].to_numpy().astype(np.float64)
        scale = compute_cnes_capacity_scale(
            eval_df, capacity_col=self._cap_col, fallback_col=self._fb_col
        )
        entities = (
            eval_df[self._entity_col].cast(pl.String).to_list()
            if self._entity_col in eval_df.columns
            else ["_global_"] * eval_df.height
        )
        t_keys = _time_order_keys(eval_df, year_col=year_col, week_col=week_col)
        unique_times = np.unique(t_keys)

        hist_map: dict[str, list[float]] = {k: list(v) for k, v in self._entity_tail_res.items()}
        out_10 = np.empty_like(p50)
        out_90 = np.empty_like(p50)
        out_95 = np.empty_like(p50)
        pending: list[list[tuple[str, float]]] = []

        for step_idx, tk in enumerate(unique_times):
            # Revela estritamente resíduos observados com atraso >= feedback_lag_weeks
            if step_idx >= self._lag_weeks:
                for ent, res_val in pending[step_idx - self._lag_weeks]:
                    buf = hist_map.setdefault(ent, [])
                    buf.append(res_val)
                    if len(buf) > self._n_lags + 16:
                        del buf[: -self._n_lags - 16]

            mask = t_keys == tk
            idxs = np.flatnonzero(mask)
            x_q = np.asarray(
                [self._query_vector_for_entity(hist_map.get(entities[int(i)], [])) for i in idxs],
                dtype=np.float64,
            )
            q10, q90, q95 = self._predict_qrf_quantiles(x_q)
            out_10[idxs] = p50[idxs] + scale[idxs] * q10
            out_90[idxs] = p50[idxs] + scale[idxs] * q90
            out_95[idxs] = p50[idxs] + scale[idxs] * q95

            step_res = [
                (entities[int(i)], float((y_true[int(i)] - p50[int(i)]) / scale[int(i)]))
                for i in idxs
            ]
            pending.append(step_res)

        return _enforce_non_crossing_non_negative(out_10, p50, out_90, out_95)
