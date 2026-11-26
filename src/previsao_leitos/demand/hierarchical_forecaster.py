"""Adaptador do pooling hierárquico bayesiano para a interface `DemandForecaster`.

Permite que `demand/hierarchical_bayes.py` entre na catraca Champion-Challenger
com o mesmo contrato de quantis dos demais modelos, sem duplicar a amostragem.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from previsao_leitos.demand.hierarchical_bayes import (
    HierarchicalBayesConfig,
    fit_hierarchical_bayes,
    validate_hierarchical_panel,
)
from previsao_leitos.demand.models import _enforce_non_crossing_non_negative
from previsao_leitos.domain import DataContractError

_QUANTILE_LEVELS: tuple[float, ...] = (0.10, 0.50, 0.90, 0.95)


class HierarchicalBayesForecaster:
    """Pooling parcial `Região ⊂ Bacia` com quantis preditivos via posterior de `mu`."""

    def __init__(self, *, config: HierarchicalBayesConfig | None = None) -> None:
        # Modo benchmark: mesmas prioris, cadeias mais curtas (o gate de convergência
        # roda no teste @slow com o default rigoroso do módulo). O backend é lido do
        # ambiente para sobreviver ao executor paralelo por alvo.
        self._config = config or HierarchicalBayesConfig(
            n_chains=2,
            n_draws=250,
            n_tune=250,
            nuts_sampler=os.environ.get("PREVISAO_BAYES_SAMPLER", "auto"),
            device=os.environ.get("PREVISAO_BAYES_DEVICE", "auto"),
        )
        self._model: Any = None
        self._idata: Any = None
        self._entities: tuple[str, ...] = ()
        self._basins: tuple[str, ...] = ()
        self._entity_basin: dict[str, str] = {}
        self._shock_mean: float = 0.0
        self._fitted: bool = False

    @property
    def model_name(self) -> str:
        return "HierarchicalBayes (Regiao<Bacia)"

    def fit(
        self, train_df: pl.DataFrame, feature_cols: Sequence[str]
    ) -> HierarchicalBayesForecaster:
        del feature_cols
        validated = validate_hierarchical_panel(train_df, config=self._config)
        model, idata = fit_hierarchical_bayes(validated, config=self._config)
        self._model = model
        self._idata = idata
        self._entities = tuple(str(v) for v in idata.posterior["entity"].to_numpy())
        self._basins = tuple(str(v) for v in idata.posterior["basin"].to_numpy())
        self._entity_basin = {
            str(entity): str(basin)
            for entity, basin in zip(
                validated["nome_regiao_saude"].to_list(),
                validated["bacia_id"].to_list(),
                strict=True,
            )
        }
        shock = validated["prob_enchente_f03_oof"].fill_null(0.0).to_numpy()
        self._shock_mean = float(np.mean(shock))
        self._fitted = True
        return self

    def _draws(self, name: str) -> np.ndarray:
        assert self._idata is not None
        values = self._idata.posterior[name].to_numpy()
        return np.asarray(values).reshape(-1, *np.asarray(values).shape[2:])

    def predict_quantiles(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self._fitted or self._idata is None:
            raise DataContractError("HierarchicalBayesForecaster não foi ajustado.")
        from scipy import stats

        if "capacidade_sem" not in df.columns:
            raise DataContractError("HierarchicalBayesForecaster exige 'capacidade_sem'.")
        if "prob_enchente_f03_oof" not in df.columns:
            raise DataContractError("HierarchicalBayesForecaster exige 'prob_enchente_f03_oof'.")

        n_rows = df.height
        cap = np.maximum(df["capacidade_sem"].to_numpy().astype(np.float64), 1.0)
        shock = df["prob_enchente_f03_oof"].fill_null(0.0).to_numpy().astype(np.float64)
        shock = shock - self._shock_mean

        beta0 = self._draws("beta0")
        u_entity = self._draws("u_entity")
        u_basin = self._draws("u_basin")
        betas = self._draws("beta_fourier")
        alpha = self._draws("alpha")

        entity_names = [str(v) for v in df["nome_regiao_saude"].to_list()]
        e_index = np.array(
            [self._entities.index(e) if e in self._entities else -1 for e in entity_names]
        )
        entity_effect = np.zeros((beta0.size, n_rows), dtype=np.float64)
        known = e_index >= 0
        if np.any(known):
            entity_effect[:, known] = u_entity[:, e_index[known]]

        basin_names = [str(v) for v in df["bacia_id"].to_list()] if "bacia_id" in df.columns else []
        b_index = np.array(
            [self._basins.index(b) if b in self._basins else -1 for b in basin_names]
        )
        basin_effect = np.zeros((beta0.size, n_rows), dtype=np.float64)
        if b_index.size == n_rows:
            known_b = b_index >= 0
            if np.any(known_b):
                basin_effect[:, known_b] = u_basin[:, b_index[known_b]]

        weeks = df["target_semana_epi"].to_numpy().astype(np.float64)
        n_harmonics = self._config.n_harmonics
        harmonics = [
            fn(2.0 * np.pi * k * weeks / 52.0)
            for k in range(1, n_harmonics + 1)
            for fn in (np.sin, np.cos)
        ]
        fourier = np.column_stack(harmonics) if harmonics else np.zeros((n_rows, 0))
        fourier_effect = fourier @ betas.T  # (n_rows, draws)
        fourier_effect = fourier_effect.T  # (draws, n_rows)

        if "delta_shock" in self._idata.posterior:
            delta = self._draws("delta_shock")
            shock_effect = delta[:, None] * shock[None, :]
        else:
            delta_basin = self._draws("delta_basin")
            slope = np.array(
                [
                    delta_basin[:, b] if b >= 0 else np.zeros(beta0.size)
                    for b in b_index
                ]
            ).T  # (draws, n_rows)
            shock_effect = slope * shock[None, :]

        log_mu = (
            np.log(cap)[None, :]
            + beta0[:, None]
            + entity_effect
            + basin_effect
            + fourier_effect
            + shock_effect
        )
        mu = np.exp(np.clip(log_mu, -20.0, 20.0))
        n_disp = np.clip(alpha, 1e-3, None)[:, None]
        prob = n_disp / (n_disp + mu)

        quantiles = [
            np.mean(stats.nbinom.ppf(q, n_disp, prob), axis=0) for q in _QUANTILE_LEVELS
        ]
        return _enforce_non_crossing_non_negative(*quantiles)
