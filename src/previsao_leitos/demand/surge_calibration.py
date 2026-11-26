"""Cabeça classificadora monotônica de surtos e calibração Rolling Platt/Beta (Task 1.2.4).

Implementa:
1. `RollingPlattCalibrator`: Platt scaling estritamente não-decrescente (`a >= 0`) com pesos
   de recência temporal para janelas deslizantes/expansivas (ex.: `2019-2023`).
2. `RollingBetaCalibrator`: calibração Beta (Kull et al., AISTATS 2017) estritamente
   monotônica (`a >= 0, b >= 0`), capturando assimetria de cauda em surtos hospitalares raros.
3. `MonotonicSurgeClassifier`: classificador probabilístico dedicado (`HistGradientBoosting`
   com `monotonic_cst` exato conforme ADR-0001) acoplado à calibração *out-of-fold* estritamente
   causal e máscara estrutural anti-G8 (`y_roll12 <= 1e-9 -> p = 0.0`).

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier

from previsao_leitos.audit.metrics import brier_skill_score, expected_calibration_error
from previsao_leitos.domain import DataContractError

CalibrationMethod = Literal["auto", "platt", "beta", "none"]
SurgeDefinition = Literal["relative", "capacity", "either"]

_HYDRO_POSITIVE_PREFIXES: tuple[str, ...] = (
    "chuva_sem_mm",
    "chuva_montante_multihop",
    "api_max_sem",
    "api_montante_pond",
    "remanso_lagunar_idx",
    "prob_enchente_f03",
    "severidade_hidro_f03",
    "el_nino_intensity",
    "oni_c",
    "pressao_hidro_captacao",
)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z_clip = np.clip(np.asarray(z, dtype=np.float64), -35.0, 35.0)
    return np.asarray(1.0 / (1.0 + np.exp(-z_clip)), dtype=np.float64)


def _safe_clip_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.asarray(np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps), dtype=np.float64)


def _weighted_log_loss(y: np.ndarray, p: np.ndarray, weights: np.ndarray) -> float:
    p_c = _safe_clip_prob(p)
    w_sum = float(np.sum(weights))
    if w_sum <= 1e-12:
        return 0.0
    loss = -(y * np.log(p_c) + (1.0 - y) * np.log(1.0 - p_c))
    return float(np.sum(weights * loss) / w_sum)


def _normalize_weights(n: int, sample_weight: np.ndarray | None) -> np.ndarray:
    if sample_weight is None:
        return np.ones(n, dtype=np.float64) / max(1, n)
    w = np.asarray(sample_weight, dtype=np.float64)
    if w.shape != (n,) or np.any(w < 0.0) or float(np.sum(w)) <= 1e-12:
        raise DataContractError("sample_weight inválido na calibração probabilística.")
    return np.asarray(w / float(np.sum(w)), dtype=np.float64)


def build_surge_target(
    df: pl.DataFrame,
    *,
    target_col: str = "target_y",
    baseline_col: str = "y_roll12",
    capacity_col: str = "capacidade_sem",
    surge_multiplier: float = 1.10,
    capacity_occupancy_threshold: float = 0.90,
    surge_definition: SurgeDefinition = "relative",
    min_positive_value: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Constrói vetor binário de surto e máscara estrutural anti-G8 (`valid_mask`).

    Quando `baseline <= min_positive_value`, o rótulo é forçado a `0.0` (anti-G8: impede `0 >= 0`).
    """
    if target_col not in df.columns or baseline_col not in df.columns:
        raise DataContractError(
            f"Colunas '{target_col}' e '{baseline_col}' são obrigatórias em build_surge_target."
        )
    y = df[target_col].to_numpy().astype(np.float64)
    base = df[baseline_col].to_numpy().astype(np.float64)
    valid = np.isfinite(y) & np.isfinite(base) & (base > min_positive_value)

    rel_surge = valid & (y >= surge_multiplier * base)
    if surge_definition == "relative":
        flag = rel_surge
    else:
        if capacity_col not in df.columns:
            raise DataContractError(
                f"Coluna '{capacity_col}' obrigatória para surge_definition='{surge_definition}'."
            )
        cap = df[capacity_col].to_numpy().astype(np.float64)
        cap_valid = valid & np.isfinite(cap) & (cap > min_positive_value)
        cap_surge = cap_valid & (y >= capacity_occupancy_threshold * cap)
        if surge_definition == "capacity":
            flag = cap_surge
            valid = cap_valid
        else:
            flag = rel_surge | cap_surge
    return flag.astype(np.float64), valid


class RollingPlattCalibrator:
    """Calibrador Platt monotonicamente não-decrescente (`a >= 0`) com pesos temporais."""

    def __init__(self, *, l2_reg: float = 1e-3) -> None:
        if l2_reg < 0.0:
            raise DataContractError("l2_reg deve ser >= 0.")
        self._l2 = float(l2_reg)
        self._a: float = 1.0
        self._b: float = 0.0
        self._fitted: bool = False

    @property
    def params(self) -> tuple[float, float]:
        return (self._a, self._b)

    def fit(
        self,
        raw_probs: np.ndarray,
        y_true: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> RollingPlattCalibrator:
        p = _safe_clip_prob(raw_probs)
        y = np.asarray(y_true, dtype=np.float64)
        if p.shape != y.shape or p.size < 4:
            raise DataContractError(
                "Dados insuficientes ou incompatíveis em RollingPlattCalibrator."
            )
        w = _normalize_weights(p.size, sample_weight)
        logits = np.log(p / (1.0 - p))

        def _obj(theta: np.ndarray) -> float:
            a, b = float(theta[0]), float(theta[1])
            pred = _sigmoid(a * logits + b)
            reg = self._l2 * ((a - 1.0) ** 2 + b**2)
            return _weighted_log_loss(y, pred, w) + reg

        res = minimize(
            _obj,
            x0=np.array([1.0, 0.0], dtype=np.float64),
            method="L-BFGS-B",
            bounds=[(0.0, 20.0), (-10.0, 10.0)],
        )
        self._a = float(max(0.0, res.x[0]))
        self._b = float(res.x[1])
        self._fitted = True
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise DataContractError(
                "RollingPlattCalibrator precisa executar `fit` antes de `transform`."
            )
        p = _safe_clip_prob(raw_probs)
        logits = np.log(p / (1.0 - p))
        return _sigmoid(self._a * logits + self._b)


class RollingBetaCalibrator:
    """Calibrador Beta (Kull et al. 2017) estritamente monotônico (`a >= 0, b >= 0`)."""

    def __init__(self, *, l2_reg: float = 1e-3) -> None:
        if l2_reg < 0.0:
            raise DataContractError("l2_reg deve ser >= 0.")
        self._l2 = float(l2_reg)
        self._a: float = 1.0
        self._b: float = 1.0
        self._c: float = 0.0
        self._fitted: bool = False

    @property
    def params(self) -> tuple[float, float, float]:
        return (self._a, self._b, self._c)

    def fit(
        self,
        raw_probs: np.ndarray,
        y_true: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> RollingBetaCalibrator:
        p = _safe_clip_prob(raw_probs)
        y = np.asarray(y_true, dtype=np.float64)
        if p.shape != y.shape or p.size < 4:
            raise DataContractError(
                "Dados insuficientes ou incompatíveis em RollingBetaCalibrator."
            )
        w = _normalize_weights(p.size, sample_weight)
        log_p = np.log(p)
        neg_log_1mp = -np.log(1.0 - p)

        def _obj(theta: np.ndarray) -> float:
            a, b, c = float(theta[0]), float(theta[1]), float(theta[2])
            pred = _sigmoid(a * log_p + b * neg_log_1mp + c)
            reg = self._l2 * ((a - 1.0) ** 2 + (b - 1.0) ** 2 + c**2)
            return _weighted_log_loss(y, pred, w) + reg

        res = minimize(
            _obj,
            x0=np.array([1.0, 1.0, 0.0], dtype=np.float64),
            method="L-BFGS-B",
            bounds=[(0.0, 20.0), (0.0, 20.0), (-10.0, 10.0)],
        )
        self._a = float(max(0.0, res.x[0]))
        self._b = float(max(0.0, res.x[1]))
        self._c = float(res.x[2])
        self._fitted = True
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise DataContractError(
                "RollingBetaCalibrator precisa executar `fit` antes de `transform`."
            )
        p = _safe_clip_prob(raw_probs)
        return _sigmoid(self._a * np.log(p) - self._b * np.log(1.0 - p) + self._c)


@dataclass(frozen=True)
class SurgeCalibrationReport:
    """Relatório auditável da calibração OOF causal (`2019-2023`) do classificador de surtos."""

    selected_method: str
    cal_years: tuple[int, ...]
    n_oof_samples: int
    oof_prevalence: float
    oof_bss_raw: float
    oof_bss_cal: float
    oof_ece_raw: float
    oof_ece_cal: float
    calibrator_params: tuple[float, ...]


class MonotonicSurgeClassifier:
    """Classificador probabilístico monotônico dedicado a surtos hospitalares (ADR-0001).

    Combina:
    - `HistGradientBoostingClassifier(loss="log_loss", monotonic_cst=...)` garantindo que maior
      risco hidrológico/ENSO jamais reduza a probabilidade prevista de surto (`0,00%` viol.);
    - Calibração pós-hoc Rolling Platt ou Beta estritamente não-decrescente ajustada sobre
      previsões *out-of-fold* expansivas nos anos de calibração (`<= 2023`, zero vazamento);
    - Máscara estrutural anti-G8 (`y_roll12 <= 1e-9 -> p = 0.0`).
    """

    def __init__(
        self,
        *,
        positive_monotonic_prefixes: Sequence[str] = _HYDRO_POSITIVE_PREFIXES,
        surge_multiplier: float = 1.10,
        capacity_occupancy_threshold: float = 0.90,
        surge_definition: SurgeDefinition = "relative",
        recency_decay: float = 0.80,
        max_iter: int = 180,
        max_depth: int = 4,
        min_samples_leaf: int = 20,
        learning_rate: float = 0.04,
        l2_regularization: float = 2.0,
        seed: int = 42,
    ) -> None:
        if not (0.0 < recency_decay <= 1.0):
            raise DataContractError(
                f"recency_decay deve estar em (0, 1]: recebido {recency_decay}."
            )
        self._pos_prefixes = tuple(positive_monotonic_prefixes)
        self._surge_mult = float(surge_multiplier)
        self._cap_thr = float(capacity_occupancy_threshold)
        self._surge_def: SurgeDefinition = surge_definition
        self._decay = float(recency_decay)
        self._max_iter = max_iter
        self._max_depth = max_depth
        self._min_samples_leaf = min_samples_leaf
        self._lr = learning_rate
        self._l2 = l2_regularization
        self._seed = seed

        self._feature_cols: tuple[str, ...] = ()
        self._mono_cst: list[int] = []
        self._clf: HistGradientBoostingClassifier | None = None
        self._calibrator: RollingPlattCalibrator | RollingBetaCalibrator | None = None
        self._report: SurgeCalibrationReport | None = None
        self._const_prob: float | None = None

    @property
    def calibration_report(self) -> SurgeCalibrationReport | None:
        return self._report

    def _build_monotonic_cst(self, feature_cols: Sequence[str]) -> list[int]:
        cst: list[int] = []
        for col in feature_cols:
            if any(col == pref or col.startswith(f"{pref}_lag") for pref in self._pos_prefixes):
                cst.append(1)
            else:
                cst.append(0)
        return cst

    def _extract_features(self, df: pl.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise DataContractError(f"Features ausentes em MonotonicSurgeClassifier: {missing}")
        x = df.select(list(feature_cols)).to_numpy().astype(np.float64)
        return np.asarray(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)

    def _fit_base_clf(
        self, x: np.ndarray, y: np.ndarray, mono_cst: list[int]
    ) -> tuple[HistGradientBoostingClassifier | None, float | None]:
        uniq = np.unique(y)
        if uniq.size < 2:
            return None, float(np.clip(np.mean(y), 1e-4, 1.0 - 1e-4))
        clf = HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=self._max_iter,
            max_depth=self._max_depth,
            min_samples_leaf=self._min_samples_leaf,
            learning_rate=self._lr,
            l2_regularization=self._l2,
            monotonic_cst=mono_cst,
            random_state=self._seed,
        )
        clf.fit(x, y.astype(np.int64))
        return clf, None

    @staticmethod
    def _predict_base(
        clf: HistGradientBoostingClassifier | None,
        const_prob: float | None,
        x: np.ndarray,
    ) -> np.ndarray:
        if clf is None:
            p = 0.0 if const_prob is None else const_prob
            return np.full(x.shape[0], p, dtype=np.float64)
        proba = np.asarray(clf.predict_proba(x)[:, 1], dtype=np.float64)
        return np.asarray(np.clip(proba, 0.0, 1.0), dtype=np.float64)

    def fit_with_rolling_calibration(
        self,
        df: pl.DataFrame,
        feature_cols: Sequence[str],
        *,
        year_col: str = "target_ano_epi",
        target_col: str = "target_y",
        baseline_col: str = "y_roll12",
        capacity_col: str = "capacidade_sem",
        cal_years: Sequence[int] = (2019, 2020, 2021, 2022, 2023),
        calibration_method: CalibrationMethod = "auto",
        min_train_rows: int = 30,
    ) -> MonotonicSurgeClassifier:
        """Treina OOF expansivo em `cal_years` para calibração Platt/Beta e refit final causal.

        Garantia anti-vazamento: cada ano `yr` em `cal_years` é pontuado por um modelo ajustado
        estritamente em `year_col < yr`, e o modelo final é ajustado apenas em
        `year_col <= max(cal_years)`.
        """
        if year_col not in df.columns:
            raise DataContractError(
                f"Coluna '{year_col}' obrigatória em fit_with_rolling_calibration."
            )
        cal_years_sorted = tuple(sorted(set(int(y) for y in cal_years)))
        if not cal_years_sorted:
            raise DataContractError("cal_years não pode ser vazio.")
        max_cal_year = cal_years_sorted[-1]

        hist_df = df.filter(pl.col(year_col) <= max_cal_year)
        if hist_df.height < min_train_rows:
            raise DataContractError(
                f"Histórico insuficiente (<= {max_cal_year}): {hist_df.height} < {min_train_rows}."
            )

        self._feature_cols = tuple(feature_cols)
        self._mono_cst = self._build_monotonic_cst(self._feature_cols)

        oof_probs_list: list[np.ndarray] = []
        oof_y_list: list[np.ndarray] = []
        oof_w_list: list[np.ndarray] = []
        used_years: list[int] = []

        for yr in cal_years_sorted:
            tr_df = hist_df.filter(pl.col(year_col) < yr)
            va_df = hist_df.filter(pl.col(year_col) == yr)
            if tr_df.height < min_train_rows or va_df.height == 0:
                continue
            y_tr, m_tr = build_surge_target(
                tr_df,
                target_col=target_col,
                baseline_col=baseline_col,
                capacity_col=capacity_col,
                surge_multiplier=self._surge_mult,
                capacity_occupancy_threshold=self._cap_thr,
                surge_definition=self._surge_def,
            )
            y_va, m_va = build_surge_target(
                va_df,
                target_col=target_col,
                baseline_col=baseline_col,
                capacity_col=capacity_col,
                surge_multiplier=self._surge_mult,
                capacity_occupancy_threshold=self._cap_thr,
                surge_definition=self._surge_def,
            )
            if int(np.sum(m_tr)) < min_train_rows or int(np.sum(m_va)) == 0:
                continue
            x_tr = self._extract_features(tr_df, self._feature_cols)[m_tr]
            x_va = self._extract_features(va_df, self._feature_cols)[m_va]
            fold_clf, fold_const = self._fit_base_clf(x_tr, y_tr[m_tr], self._mono_cst)
            p_va = self._predict_base(fold_clf, fold_const, x_va)

            age = float(max_cal_year - yr)
            yr_weight = self._decay**age
            oof_probs_list.append(p_va)
            oof_y_list.append(y_va[m_va])
            oof_w_list.append(np.full(p_va.size, yr_weight, dtype=np.float64))
            used_years.append(yr)

        if not oof_probs_list:
            raise DataContractError(
                f"Nenhum fold OOF válido gerado em cal_years={cal_years_sorted}."
            )

        oof_p = np.concatenate(oof_probs_list)
        oof_y = np.concatenate(oof_y_list)
        oof_w = np.concatenate(oof_w_list)
        oof_w_norm = _normalize_weights(oof_p.size, oof_w)

        raw_bss = brier_skill_score(oof_y, oof_p)
        raw_ece = expected_calibration_error(oof_y, oof_p, n_bins=8)

        selected_name = "none"
        selected_cal: RollingPlattCalibrator | RollingBetaCalibrator | None = None
        cal_p = oof_p
        cal_params: tuple[float, ...] = ()

        if calibration_method != "none" and np.unique(oof_y).size >= 2:
            candidates: list[tuple[str, RollingPlattCalibrator | RollingBetaCalibrator]] = []
            if calibration_method in ("auto", "platt"):
                candidates.append(
                    ("platt", RollingPlattCalibrator().fit(oof_p, oof_y, sample_weight=oof_w))
                )
            if calibration_method in ("auto", "beta"):
                candidates.append(
                    ("beta", RollingBetaCalibrator().fit(oof_p, oof_y, sample_weight=oof_w))
                )

            best_score = float("inf")
            for name, cal_obj in candidates:
                pred_c = cal_obj.transform(oof_p)
                # Critério combinado ponderado: Brier + 0.15 * ECE (favorece calibração limpa)
                w_brier = float(np.sum(oof_w_norm * (pred_c - oof_y) ** 2))
                ece_c = expected_calibration_error(oof_y, pred_c, n_bins=8)
                score = w_brier + 0.15 * ece_c
                if score < best_score:
                    best_score = score
                    selected_name = name
                    selected_cal = cal_obj
                    cal_p = pred_c
                    cal_params = cal_obj.params

        self._calibrator = selected_cal
        self._report = SurgeCalibrationReport(
            selected_method=selected_name,
            cal_years=tuple(used_years),
            n_oof_samples=int(oof_p.size),
            oof_prevalence=float(np.mean(oof_y)),
            oof_bss_raw=float(raw_bss),
            oof_bss_cal=float(brier_skill_score(oof_y, cal_p)),
            oof_ece_raw=float(raw_ece),
            oof_ece_cal=float(expected_calibration_error(oof_y, cal_p, n_bins=8)),
            calibrator_params=cal_params,
        )

        # Refit final estritamente causal sobre todos os dados válidos <= max_cal_year
        y_all, m_all = build_surge_target(
            hist_df,
            target_col=target_col,
            baseline_col=baseline_col,
            capacity_col=capacity_col,
            surge_multiplier=self._surge_mult,
            capacity_occupancy_threshold=self._cap_thr,
            surge_definition=self._surge_def,
        )
        x_all = self._extract_features(hist_df, self._feature_cols)[m_all]
        self._clf, self._const_prob = self._fit_base_clf(x_all, y_all[m_all], self._mono_cst)
        return self

    def predict_proba_raw(
        self,
        df: pl.DataFrame,
        *,
        baseline_col: str = "y_roll12",
        min_positive_value: float = 1e-9,
    ) -> np.ndarray:
        """Retorna probabilidade bruta (pré-Platt/Beta) com máscara estrutural anti-G8."""
        if not self._feature_cols or (self._clf is None and self._const_prob is None):
            raise DataContractError(
                "MonotonicSurgeClassifier precisa executar fit antes de prever."
            )
        x = self._extract_features(df, self._feature_cols)
        probs = self._predict_base(self._clf, self._const_prob, x)
        if baseline_col in df.columns:
            base = df[baseline_col].to_numpy().astype(np.float64)
            probs = np.where(np.isfinite(base) & (base > min_positive_value), probs, 0.0)
        return np.asarray(np.clip(probs, 0.0, 1.0), dtype=np.float64)

    def predict_proba(
        self,
        df: pl.DataFrame,
        *,
        baseline_col: str = "y_roll12",
        min_positive_value: float = 1e-9,
    ) -> np.ndarray:
        """Retorna probabilidade calibrada de surto com máscara anti-G8 e monotonicidade exata."""
        raw = self.predict_proba_raw(
            df, baseline_col=baseline_col, min_positive_value=min_positive_value
        )
        if self._calibrator is None:
            return raw
        cal = self._calibrator.transform(raw)
        if baseline_col in df.columns:
            base = df[baseline_col].to_numpy().astype(np.float64)
            cal = np.where(np.isfinite(base) & (base > min_positive_value), cal, 0.0)
        return np.asarray(np.clip(cal, 0.0, 1.0), dtype=np.float64)

    def audit_monotonicity(
        self,
        df: pl.DataFrame,
        *,
        bump_cols: Sequence[str] | None = None,
        bump_delta: float = 50.0,
        tol: float = 1e-9,
    ) -> float:
        """Audita taxa de violação monotônica (ADR-0001) sob aumento hidrológico positivo."""
        cols_to_bump = (
            [c for c in bump_cols if c in df.columns]
            if bump_cols is not None
            else [
                c
                for c, sign in zip(self._feature_cols, self._mono_cst, strict=True)
                if sign == 1 and c in df.columns
            ]
        )
        if not cols_to_bump or df.is_empty():
            return 0.0
        base_p = self.predict_proba(df)
        bumped_df = df.with_columns(
            [(pl.col(c).cast(pl.Float64) + float(bump_delta)).alias(c) for c in cols_to_bump]
        )
        bumped_p = self.predict_proba(bumped_df)
        return float(np.mean(bumped_p + tol < base_p))
