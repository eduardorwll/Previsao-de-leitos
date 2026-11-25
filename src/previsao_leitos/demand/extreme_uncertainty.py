"""Incerteza de cauda extrema sob El Niño: EVT-GPD, Mondrian Conformal e Pooling Hierárquico.

Resolve o colapso de especificidade nas regiões do interior (estratificando a
calibração conformal por porte regional x regime ENSO), modela caudas pesadas
via Teoria de Valores Extremos (Generalized Pareto — POT) e aplica pooling
parcial hierárquico (Hospital/Região ⊂ Bacia).

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import genpareto

from previsao_leitos.demand.models import DemandForecaster, _enforce_non_crossing_non_negative
from previsao_leitos.domain import DataContractError


class ExtremeValueGpdTail:
    """Ajuste Peaks-Over-Threshold (POT) com Distribuição de Pareto Generalizada (EVT-GPD)."""

    def __init__(self, *, threshold_quantile: float = 0.85, max_shape: float = 0.5) -> None:
        if not 0.5 <= threshold_quantile < 0.99:
            raise DataContractError("threshold_quantile deve estar em [0.5, 0.99).")
        self._q_u = threshold_quantile
        self._max_shape = max_shape
        self._u: float = 0.0
        self._shape: float = 0.0
        self._scale: float = 1.0
        self._exceed_prob: float = 1.0 - threshold_quantile
        self._fitted: bool = False

    def fit(self, scores: np.ndarray) -> ExtremeValueGpdTail:
        """Ajusta GPD sobre os excessos acima do limiar `u`."""
        arr = np.asarray(scores, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size < 15:
            raise DataContractError("EVT-GPD requer ao menos 15 observações finitas.")

        self._u = float(np.quantile(arr, self._q_u))
        exceedances = arr[arr > self._u] - self._u
        self._exceed_prob = max(float(exceedances.size) / float(arr.size), 1e-4)

        if exceedances.size < 5 or float(np.std(exceedances, ddof=1)) <= 1e-12:
            self._shape = 0.0
            self._scale = max(float(np.mean(exceedances)) if exceedances.size else 1e-3, 1e-3)
        else:
            c_hat, _, scale_hat = genpareto.fit(exceedances, floc=0.0)
            self._shape = float(np.clip(c_hat, -0.25, self._max_shape))
            self._scale = max(float(scale_hat), 1e-6)

        self._fitted = True
        return self

    def return_level(self, prob: float) -> float:
        """Retorna o nível de retorno para probabilidade acumulada `prob` (ex.: 0.95, 0.99)."""
        if not self._fitted:
            raise DataContractError("ExtremeValueGpdTail precisa ser ajustado antes de prever.")
        if prob <= 1.0 - self._exceed_prob:
            return self._u
        tail_ratio = self._exceed_prob / max(1.0 - prob, 1e-6)
        if abs(self._shape) < 1e-6:
            excess = self._scale * float(np.log(max(tail_ratio, 1.0)))
        else:
            excess = (self._scale / self._shape) * (float(tail_ratio**self._shape) - 1.0)
        return self._u + max(0.0, excess)


class MondrianEnsoConformalCalibrator:
    """Calibração Conformal Mondrian estratificada por porte regional e regime ENSO."""

    def __init__(
        self,
        *,
        base_model: DemandForecaster,
        group_cols: Sequence[str] = ("porte_regional", "enso_regime"),
        min_group_size: int = 12,
        use_evt_p95: bool = True,
    ) -> None:
        self._base = base_model
        self._group_cols = tuple(group_cols)
        self._min_group_size = min_group_size
        self._use_evt_p95 = use_evt_p95
        self._global_q: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._group_q: dict[tuple[str, ...], tuple[float, float, float]] = {}
        self._fitted: bool = False

    @property
    def model_name(self) -> str:
        return f"{self._base.model_name} + MondrianConformal-EVT"

    def fit(
        self,
        train_df: pl.DataFrame,
        feature_cols: Sequence[str],
    ) -> MondrianEnsoConformalCalibrator:
        """Ajusta o modelo base subjacente (compatibilidade com o protocolo `DemandForecaster`)."""
        self._base.fit(train_df, feature_cols)
        return self

    @staticmethod
    def _finite_q(scores: np.ndarray, alpha: float) -> float:
        n = len(scores)
        q_idx = min(1.0, ((1.0 - alpha) * (n + 1.0)) / max(n, 1))
        return max(0.0, float(np.quantile(scores, q_idx, method="higher")))

    def _compute_group_quantiles(
        self, s_low_10: np.ndarray, s_high_90: np.ndarray, s_high_95: np.ndarray
    ) -> tuple[float, float, float]:
        q10 = self._finite_q(s_low_10, alpha=0.10)
        q90 = self._finite_q(s_high_90, alpha=0.10)
        q95 = self._finite_q(s_high_95, alpha=0.05)
        if self._use_evt_p95 and s_high_95.size >= 20:
            evt = ExtremeValueGpdTail(threshold_quantile=0.80).fit(s_high_95)
            q95 = max(q95, evt.return_level(0.95))
        return (q10, q90, q95)

    def _group_keys(self, df: pl.DataFrame) -> list[tuple[str, ...]]:
        present = [c for c in self._group_cols if c in df.columns]
        if not present:
            return [("__global__",) for _ in range(df.height)]
        cols = [df[c].cast(pl.String).fill_null("NA").to_list() for c in present]
        return [tuple(vals) for vals in zip(*cols, strict=True)]

    def fit_calibration(self, *, cal_df: pl.DataFrame) -> MondrianEnsoConformalCalibrator:
        """Ajusta quantis conformais por estrato (porte x ENSO) sobre `cal_df` (2023)."""
        if cal_df.height < 10:
            raise DataContractError("cal_df insuficiente para MondrianConformal.")

        raw = self._base.predict_quantiles(cal_df)
        y_cal = cal_df["target_y"].cast(pl.Float64).to_numpy()
        p10 = raw["pred_p10"].to_numpy().astype(np.float64)
        p90 = raw["pred_p90"].to_numpy().astype(np.float64)
        p95 = raw["pred_p95"].to_numpy().astype(np.float64)

        s_low_10 = p10 - y_cal
        s_high_90 = y_cal - p90
        s_high_95 = y_cal - p95

        self._global_q = self._compute_group_quantiles(s_low_10, s_high_90, s_high_95)
        self._group_q = {}

        keys = self._group_keys(cal_df)
        unique_keys = sorted(set(keys))
        for key in unique_keys:
            mask = np.array([k == key for k in keys], dtype=bool)
            n_g = int(np.sum(mask))
            if n_g >= self._min_group_size:
                self._group_q[key] = self._compute_group_quantiles(
                    s_low_10[mask], s_high_90[mask], s_high_95[mask]
                )
            elif n_g >= 3:
                w = float(n_g) / float(self._min_group_size)
                gq = self._compute_group_quantiles(
                    s_low_10[mask], s_high_90[mask], s_high_95[mask]
                )
                self._group_q[key] = (
                    w * gq[0] + (1.0 - w) * self._global_q[0],
                    w * gq[1] + (1.0 - w) * self._global_q[1],
                    w * gq[2] + (1.0 - w) * self._global_q[2],
                )
            else:
                self._group_q[key] = self._global_q

        self._fitted = True
        return self

    def predict_quantiles(self, df: pl.DataFrame) -> pl.DataFrame:
        """Emite quantis calibrados por estrato com garantia de não-cruzamento."""
        if not self._fitted:
            raise DataContractError("MondrianEnsoConformalCalibrator não foi calibrado.")
        raw = self._base.predict_quantiles(df)
        keys = self._group_keys(df)
        q10_vec = np.array([self._group_q.get(k, self._global_q)[0] for k in keys], dtype=float)
        q90_vec = np.array([self._group_q.get(k, self._global_q)[1] for k in keys], dtype=float)
        q95_vec = np.array([self._group_q.get(k, self._global_q)[2] for k in keys], dtype=float)

        p10 = raw["pred_p10"].to_numpy().astype(np.float64) - q10_vec
        p50 = raw["pred_p50"].to_numpy().astype(np.float64)
        p90 = raw["pred_p90"].to_numpy().astype(np.float64) + q90_vec
        p95 = raw["pred_p95"].to_numpy().astype(np.float64) + q95_vec
        return _enforce_non_crossing_non_negative(p10, p50, p90, p95)


@dataclass(frozen=True)
class HierarchicalPosteriorSummary:
    """Resumo do ajuste hierárquico Empirical Bayes (Global -> Bacia -> Entidade)."""

    global_mean: float
    basin_means: dict[str, float]
    entity_means: dict[tuple[str, str], float]
    within_variance: float
    between_entity_variance: float


class HierarchicalBasinBayesShrinkage:
    """Modelo hierárquico Bayesiano de pooling parcial (Hospital/Região ⊂ Bacia)."""

    def __init__(self, *, prior_strength: float = 3.0) -> None:
        if prior_strength <= 0.0:
            raise DataContractError("prior_strength deve ser > 0.")
        self._prior_strength = prior_strength
        self._summary: HierarchicalPosteriorSummary | None = None

    def fit(
        self,
        df: pl.DataFrame,
        *,
        basin_col: str = "bacia_id",
        entity_col: str = "entidade_id",
        value_col: str = "efeito_choque",
    ) -> HierarchicalBasinBayesShrinkage:
        """Estima médias posteriores com encolhimento hierárquico (Empirical Bayes)."""
        for col in (basin_col, entity_col, value_col):
            if col not in df.columns:
                raise DataContractError(f"Coluna '{col}' ausente em HierarchicalBasinBayes.")
        if df.is_empty():
            raise DataContractError("DataFrame vazio em HierarchicalBasinBayes.")

        vals = df[value_col].cast(pl.Float64).to_numpy()
        global_mean = float(np.mean(vals))
        total_var = float(np.var(vals, ddof=1)) if vals.size > 1 else 1e-2

        basin_means: dict[str, float] = {}
        basin_groups = df.group_by(basin_col).agg(
            pl.col(value_col).cast(pl.Float64).mean().alias("mu"),
            pl.len().alias("n"),
        )
        for row in basin_groups.iter_rows(named=True):
            b = str(row[basin_col])
            n_b = float(row["n"])
            mu_b = float(row["mu"])
            basin_means[b] = (n_b * mu_b + self._prior_strength * global_mean) / (
                n_b + self._prior_strength
            )

        entity_means: dict[tuple[str, str], float] = {}
        ent_groups = df.group_by([basin_col, entity_col]).agg(
            pl.col(value_col).cast(pl.Float64).mean().alias("mu"),
            pl.len().alias("n"),
        )
        for row in ent_groups.iter_rows(named=True):
            b = str(row[basin_col])
            e = str(row[entity_col])
            n_e = float(row["n"])
            mu_e = float(row["mu"])
            mu_prior = basin_means.get(b, global_mean)
            entity_means[(b, e)] = (n_e * mu_e + self._prior_strength * mu_prior) / (
                n_e + self._prior_strength
            )

        between_var = (
            float(np.var(list(entity_means.values()), ddof=1))
            if len(entity_means) > 1
            else total_var * 0.5
        )
        self._summary = HierarchicalPosteriorSummary(
            global_mean=global_mean,
            basin_means=basin_means,
            entity_means=entity_means,
            within_variance=max(total_var - between_var, 1e-6),
            between_entity_variance=max(between_var, 1e-6),
        )
        return self

    def predict_entity(self, basin_id: str, entity_id: str) -> float:
        """Retorna a estimativa posterior hierárquica (com fallback para bacia -> global)."""
        if self._summary is None:
            raise DataContractError("HierarchicalBasinBayesShrinkage não foi ajustado.")
        if (basin_id, entity_id) in self._summary.entity_means:
            return self._summary.entity_means[(basin_id, entity_id)]
        return self._summary.basin_means.get(basin_id, self._summary.global_mean)
