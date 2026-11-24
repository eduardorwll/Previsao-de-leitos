"""Calibração conformal heterocedástica sob estresse hidrológico (calibrada em 2023)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from previsao_leitos.demand.models import DemandForecaster, _enforce_non_crossing_non_negative
from previsao_leitos.domain import DataContractError


class ConformalFloodCalibrator:
    """Wrapper conformal (CQR heterocedástico) sobre qualquer `DemandForecaster`.

    Calibrado estritamente em `cal_df` (ano 2023), usando estatísticas de padronização
    de estresse hidrológico ajustadas apenas em `train_df` (<= 2022). Garante cobertura
    finita em eventos extremos e preserva monotonicidade em P50/P90/P95.
    """

    def __init__(
        self,
        *,
        base_model: DemandForecaster,
        stress_cols: Sequence[str] = (
            "api_max_sem_lag0",
            "api_montante_pond_lag0",
            "remanso_lagunar_idx_lag0",
        ),
        stress_gamma: float = 0.65,
    ) -> None:
        if stress_gamma < 0.0:
            raise DataContractError(f"stress_gamma deve ser >= 0 (recebido {stress_gamma}).")
        self._base = base_model
        self._stress_cols = tuple(stress_cols)
        self._gamma = stress_gamma
        self._stress_stats: dict[str, tuple[float, float]] = {}
        self._q_low_10: float = 0.0
        self._q_high_90: float = 0.0
        self._q_high_95: float = 0.0
        self._fitted: bool = False

    @property
    def model_name(self) -> str:
        return f"{self._base.model_name} + Conformal(2023)"

    def fit(self, train_df: pl.DataFrame, feature_cols: Sequence[str]) -> ConformalFloodCalibrator:
        self._base.fit(train_df, feature_cols)
        return self

    def _compute_scale(self, df: pl.DataFrame) -> np.ndarray:
        if "y_roll12" in df.columns:
            base = df["y_roll12"].to_numpy().astype(np.float64)
        else:
            base = df["y_origin"].to_numpy().astype(np.float64)
        size_scale = np.maximum(1.0, np.sqrt(np.maximum(base, 1.0)))

        if not self._stress_stats:
            return np.asarray(size_scale, dtype=np.float64)

        z_acc = np.zeros(df.height, dtype=np.float64)
        for col, (mu, sd) in self._stress_stats.items():
            if col in df.columns:
                vals = df[col].to_numpy().astype(np.float64)
                z_pos = np.maximum(0.0, (vals - mu) / sd)
                z_acc += z_pos
        z_mean = z_acc / max(1, len(self._stress_stats))
        return np.asarray(size_scale * (1.0 + self._gamma * z_mean), dtype=np.float64)

    @staticmethod
    def _finite_sample_quantile(scores: np.ndarray, target_level: float) -> float:
        n = len(scores)
        q_idx = min(1.0, ((1.0 - target_level) * (n + 1.0)) / n)
        return float(np.quantile(scores, q_idx, method="higher"))

    def fit_calibration(
        self,
        *,
        train_df: pl.DataFrame,
        cal_df: pl.DataFrame,
    ) -> ConformalFloodCalibrator:
        """Ajusta escalas de estresse em `train_df` (<=2022) e quantis conformais em `cal_df`."""
        if cal_df.height < 10:
            raise DataContractError(
                f"Conjunto de calibração insuficiente para conformal (n={cal_df.height})."
            )

        self._stress_stats = {}
        for col in self._stress_cols:
            if col in train_df.columns:
                arr = train_df[col].to_numpy().astype(np.float64)
                mu = float(np.mean(arr))
                sd = float(np.std(arr))
                self._stress_stats[col] = (mu, sd if sd > 1e-8 else 1.0)

        raw = self._base.predict_quantiles(cal_df)
        y_cal = cal_df["target_y"].to_numpy().astype(np.float64)
        p10 = raw["pred_p10"].to_numpy().astype(np.float64)
        p90 = raw["pred_p90"].to_numpy().astype(np.float64)
        p95 = raw["pred_p95"].to_numpy().astype(np.float64)
        scale = self._compute_scale(cal_df)

        # Scores CQR normalizados pelo fator de estresse (sempre >= 0 para expansão segura)
        s_low_10 = (p10 - y_cal) / scale
        s_high_90 = (y_cal - p90) / scale
        s_high_95 = (y_cal - p95) / scale

        # Quantil simétrico 90% por cauda (alpha=0.10 em cada cauda -> cobertura nominal >= 80%)
        self._q_low_10 = max(0.0, self._finite_sample_quantile(s_low_10, target_level=0.10))
        self._q_high_90 = max(0.0, self._finite_sample_quantile(s_high_90, target_level=0.10))
        self._q_high_95 = max(0.0, self._finite_sample_quantile(s_high_95, target_level=0.05))
        self._fitted = True
        return self

    def predict_quantiles(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self._fitted:
            raise DataContractError(
                "ConformalFloodCalibrator precisa executar `fit_calibration` antes de prever."
            )
        raw = self._base.predict_quantiles(df)
        scale = self._compute_scale(df)
        p10 = raw["pred_p10"].to_numpy().astype(np.float64) - scale * self._q_low_10
        p50 = raw["pred_p50"].to_numpy().astype(np.float64)
        p90 = raw["pred_p90"].to_numpy().astype(np.float64) + scale * self._q_high_90
        p95 = raw["pred_p95"].to_numpy().astype(np.float64) + scale * self._q_high_95
        return _enforce_non_crossing_non_negative(p10, p50, p90, p95)
