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
from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from previsao_leitos.demand.models import (
    QUANTILE_LEVELS,
    DemandForecaster,
    NegativeBinomialHurdleForecaster,
    _enforce_non_crossing_non_negative,
)
from previsao_leitos.domain import DataContractError

# Colunas basais canônicas (`datasets.py` gera `target_week_sin`/`target_week_cos`);
# aliases legados (`sin_target_wk`/`cos_target_wk`) são aceitos apenas como fallback.
_DEFAULT_BASAL_COLS: tuple[str, ...] = (
    "target_week_sin",
    "target_week_cos",
    "srag_viral_confirmado",
    "frac_srag_viral",
)

_LEGACY_HARMONIC_ALIASES: tuple[tuple[str, str], ...] = (
    ("target_week_sin", "sin_target_wk"),
    ("target_week_cos", "cos_target_wk"),
)

# Colunas autorregressivas curtas e colunas-alvo proibidas em ambos os estágios (Bug B)
_FORBIDDEN_AR_COLS: frozenset[str] = frozenset(
    {
        "y_origin",
        "y_roll4",
        "y_roll12",
        "y_dev_roll12",
        "y_seasonal_naive",
        "target_y",
        "target_delta",
    }
)

_DEFAULT_MONO_POS: tuple[str, ...] = (
    "prob_enchente_f03",
    "severidade_hidro_f03",
    "chuva_montante_pond",
    "chuva_montante_multihop",
    "chuva_montante",
    "chuva_sem_mm",
    "api_montante_pond",
    "api_max_sem",
    "remanso_lagunar_idx",
    "exposicao_aterro_ativa",
    "indice_aterro_varzea",
    "el_nino_intensity",
    "oni_c",
)

_DEFAULT_MONO_NEG: tuple[str, ...] = ("frac_floresta",)


@dataclass(frozen=True)
class TwoStageComponents:
    """Decomposição algébrica auditável do ponto F04.1.

    `point_beds = (basal_rate + shock_rate) * capacity` reproduz exatamente a predição
    pontual (P50) do modelo; a separação permite auditar SHAP por estágio sem recomputar
    a álgebra fora do modelo.
    """

    capacity: np.ndarray
    basal_rate: np.ndarray
    shock_rate: np.ndarray
    shock_matrix: np.ndarray
    shock_cols: tuple[str, ...]

    @property
    def point_rate(self) -> np.ndarray:
        """Taxa de ocupação relativa prevista (basal + choque), antes do reescalonamento."""
        return np.asarray(self.basal_rate + self.shock_rate, dtype=np.float64)

    @property
    def point_beds(self) -> np.ndarray:
        """Ponto em leitos ocupados/dia (unidade original da F04.1)."""
        return np.asarray(self.point_rate * self.capacity, dtype=np.float64)


class OrthogonalTwoStageOccupancyForecaster:
    """Modelo ortogonal em 2 estágios em taxa relativa (ocupação / capacidade CNES).

    - Estágio 1 (Basal Lento): Ridge sobre harmônicos sazonais canônicos (`target_week_sin/cos`)
      e covariáveis virais exógenas, excluindo todas as colunas autorregressivas (`y_origin`,
      `y_roll4`, `y_roll12`, `y_dev_roll12`).
    - Estágio 2 (Choque Ex-Ante): `HistGradientBoostingRegressor(loss="squared_error")` com
      restrição monotônica exata (`ADR-0001`) sobre o resíduo em taxa relativa.
    - Incerteza Heterocedástica: quantis empíricos dos resíduos padronizados por `sqrt(cap)`,
      preservando 0,00% de violações monotônicas em P10/P50/P90/P95 sem inflar grandes polos.
    """

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
        use_local_level: bool = False,
        level_source: str = "y_roll12",
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
        self._use_local_level = use_local_level
        self._level_source = level_source

        self._basal_cols: tuple[str, ...] = ()
        self._shock_cols: tuple[str, ...] = ()
        self._monotone_map: dict[str, int] = {}
        self._basal_model: Ridge | None = None
        self._basal_mean: float = 0.0
        self._shock_model: HistGradientBoostingRegressor | None = None
        self._scale_quantiles: dict[float, float] = {
            0.10: -1.2816,
            0.50: 0.0,
            0.90: 1.2816,
            0.95: 1.6449,
        }
        self._fitted: bool = False

    @property
    def model_name(self) -> str:
        return "OrthogonalTwoStageOccupancy (F04.1)"

    @property
    def basal_cols(self) -> tuple[str, ...]:
        """Colunas efetivamente selecionadas para o Estágio 1 (basal lento)."""
        return self._basal_cols

    @property
    def shock_cols(self) -> tuple[str, ...]:
        """Colunas efetivamente selecionadas para o Estágio 2 (choque ex-ante)."""
        return self._shock_cols

    @property
    def shock_estimator(self) -> HistGradientBoostingRegressor:
        """Estimador monotônico do Estágio 2 já ajustado (somente leitura para auditoria)."""
        if self._shock_model is None:
            raise DataContractError("OrthogonalTwoStageOccupancyForecaster não foi ajustado.")
        return self._shock_model

    @property
    def monotone_constraints(self) -> dict[str, int]:
        """Mapa auditável `{coluna: restricao_monotonica}` aplicado no Estágio 2."""
        return dict(self._monotone_map)

    @staticmethod
    def _is_forbidden_ar(col: str) -> bool:
        return col in _FORBIDDEN_AR_COLS or col.startswith(
            ("y_origin", "y_roll", "y_dev_roll", "y_seasonal", "target_y", "target_delta")
        )

    def _resolve_basal_cols(self, df: pl.DataFrame) -> tuple[str, ...]:
        resolved: list[str] = []
        seen: set[str] = set()
        alias_map = dict(_LEGACY_HARMONIC_ALIASES)
        reverse_alias_map = {legacy: canon for canon, legacy in _LEGACY_HARMONIC_ALIASES}

        for col in self._basal_cols_cfg:
            if self._is_forbidden_ar(col):
                continue
            chosen: str | None = None
            if col in df.columns:
                chosen = col
            elif col in alias_map and alias_map[col] in df.columns:
                chosen = alias_map[col]
            elif col in reverse_alias_map and reverse_alias_map[col] in df.columns:
                chosen = reverse_alias_map[col]
            if chosen is not None and chosen not in seen:
                resolved.append(chosen)
                seen.add(chosen)
        return tuple(resolved)

    def _extract_basal_matrix(self, df: pl.DataFrame) -> np.ndarray:
        alias_map = dict(_LEGACY_HARMONIC_ALIASES)
        reverse_alias_map = {legacy: canon for canon, legacy in _LEGACY_HARMONIC_ALIASES}
        selected: list[pl.Expr] = []
        for col in self._basal_cols:
            if col in df.columns:
                selected.append(pl.col(col).cast(pl.Float64).fill_null(0.0).alias(col))
            elif col in alias_map and alias_map[col] in df.columns:
                selected.append(pl.col(alias_map[col]).cast(pl.Float64).fill_null(0.0).alias(col))
            elif col in reverse_alias_map and reverse_alias_map[col] in df.columns:
                selected.append(
                    pl.col(reverse_alias_map[col]).cast(pl.Float64).fill_null(0.0).alias(col)
                )
            else:
                raise DataContractError(
                    f"Coluna basal '{col}' ausente no DataFrame de inferência da F04.1."
                )
        if self._use_local_level:
            # Nível local suavizado (média móvel de 12 semanas) em taxa relativa: fornece
            # âncora de patamar ao Estágio 1 sem copiar a última observação (que causaria
            # atraso de fase). Não é AR de curto prazo: usa média, não `y_origin`.
            if self._level_source not in df.columns:
                raise DataContractError(
                    f"Nível local exige a coluna '{self._level_source}' na inferência da F04.1."
                )
            if self._capacity_col is not None and self._capacity_col in df.columns:
                cap_expr = (
                    pl.col(self._capacity_col).cast(pl.Float64).fill_null(1.0).clip(lower_bound=1.0)
                )
            else:
                cap_expr = pl.lit(1.0, dtype=pl.Float64)
            selected.append(
                (
                    pl.col(self._level_source).cast(pl.Float64).fill_null(0.0) / cap_expr
                ).alias("_local_level_rate")
            )
        mat = df.select(selected).to_numpy().astype(np.float64)
        return np.asarray(np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)

    def _extract_capacity(self, df: pl.DataFrame) -> np.ndarray:
        if self._capacity_col is not None and self._capacity_col in df.columns:
            cap = df[self._capacity_col].cast(pl.Float64).fill_null(1.0).to_numpy()
            cap = np.nan_to_num(cap, nan=1.0, posinf=1.0, neginf=1.0)
            return np.asarray(np.maximum(cap, 1.0), dtype=np.float64)
        if "y_roll12" in df.columns:
            roll = df["y_roll12"].cast(pl.Float64).fill_null(1.0).to_numpy()
            roll = np.nan_to_num(roll, nan=1.0, posinf=1.0, neginf=1.0)
            return np.asarray(np.maximum(roll, 1.0), dtype=np.float64)
        return np.ones(df.height, dtype=np.float64)

    @staticmethod
    def _heteroscedastic_scale(cap: np.ndarray) -> np.ndarray:
        """Escala heterocedástica `sqrt(cap)` consistente com variância hospitalar por porte."""
        return np.asarray(np.sqrt(np.maximum(cap, 1.0)), dtype=np.float64)

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
        y_series = train_df["target_y"].cast(pl.Float64, strict=False)
        if y_series.null_count() > 0:
            raise DataContractError(
                "OrthogonalTwoStage: 'target_y' contém valores nulos/inválidos."
            )
        y_true = y_series.to_numpy().astype(np.float64)
        if not np.all(np.isfinite(y_true)):
            raise DataContractError("OrthogonalTwoStage: 'target_y' contém NaN ou infinito.")
        y_rate = y_true / cap

        # Estágio 1: apenas covariáveis basais lentas (sem qualquer AR curto)
        self._basal_cols = self._resolve_basal_cols(train_df)
        self._basal_mean = float(np.mean(y_rate))
        if self._basal_cols:
            x_basal = self._extract_basal_matrix(train_df)
            self._basal_model = Ridge(alpha=self._basal_alpha).fit(x_basal, y_rate)
            basal_hat = np.asarray(self._basal_model.predict(x_basal), dtype=np.float64)
        else:
            self._basal_model = None
            basal_hat = np.full(train_df.height, self._basal_mean, dtype=np.float64)

        shock_resid = y_rate - basal_hat

        # Estágio 2: covariáveis físicas/ENSO/F03.1 estritamente ortogonais (sem AR nem basais)
        excluded = set(self._basal_cols)
        shock_candidates = [
            c
            for c in feature_cols
            if c in train_df.columns and c not in excluded and not self._is_forbidden_ar(c)
        ]
        if not shock_candidates:
            raise DataContractError(
                "Nenhuma covariável de choque disponível para o Estágio 2 da F04.1."
            )
        self._shock_cols = tuple(shock_candidates)
        mono_cst = self._monotone_cst(self._shock_cols)
        self._monotone_map = dict(zip(self._shock_cols, mono_cst, strict=True))

        x_shock = np.nan_to_num(
            train_df.select(list(self._shock_cols))
            .fill_null(0.0)
            .to_numpy()
            .astype(np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self._shock_model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=self._max_iter,
            max_depth=self._max_depth,
            min_samples_leaf=self._min_samples_leaf,
            learning_rate=self._lr,
            monotonic_cst=mono_cst,
            random_state=self._seed,
        ).fit(x_shock, shock_resid)

        shock_hat = np.asarray(self._shock_model.predict(x_shock), dtype=np.float64)
        loc_count = (basal_hat + shock_hat) * cap
        scale = self._heteroscedastic_scale(cap)
        std_resid = (y_true - loc_count) / scale
        self._scale_quantiles = {q: float(np.quantile(std_resid, q)) for q in QUANTILE_LEVELS}
        self._fitted = True
        return self

    def predict_stage_components(self, df: pl.DataFrame) -> TwoStageComponents:
        """Decompõe a predição pontual em Estágio 1 (basal) e Estágio 2 (choque).

        Os componentes retornados são exatamente os usados por `predict_quantiles`:
        `point_beds` reproduz o P50 sem qualquer reajuste.
        """
        if not self._fitted or self._shock_model is None:
            raise DataContractError("OrthogonalTwoStageOccupancyForecaster não foi ajustado.")

        cap = self._extract_capacity(df)
        if self._basal_model is not None and self._basal_cols:
            x_basal = self._extract_basal_matrix(df)
            basal_hat = np.asarray(self._basal_model.predict(x_basal), dtype=np.float64)
        else:
            basal_hat = np.full(df.height, self._basal_mean, dtype=np.float64)

        missing_shock = [c for c in self._shock_cols if c not in df.columns]
        if missing_shock:
            raise DataContractError(
                f"Colunas de choque ausentes na inferência da F04.1: {missing_shock}"
            )
        x_shock = np.nan_to_num(
            df.select(list(self._shock_cols)).fill_null(0.0).to_numpy().astype(np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        shock_hat = np.asarray(self._shock_model.predict(x_shock), dtype=np.float64)
        return TwoStageComponents(
            capacity=cap,
            basal_rate=basal_hat,
            shock_rate=shock_hat,
            shock_matrix=x_shock,
            shock_cols=self._shock_cols,
        )

    def predict_quantiles(self, df: pl.DataFrame) -> pl.DataFrame:
        """Prevê quantis P10/P50/P90/P95 heterocedásticos re-escalados para unidade original."""
        comp = self.predict_stage_components(df)
        loc_count = comp.point_beds
        scale = self._heteroscedastic_scale(comp.capacity)

        return _enforce_non_crossing_non_negative(
            loc_count + scale * self._scale_quantiles[0.10],
            loc_count + scale * self._scale_quantiles[0.50],
            loc_count + scale * self._scale_quantiles[0.90],
            loc_count + scale * self._scale_quantiles[0.95],
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
