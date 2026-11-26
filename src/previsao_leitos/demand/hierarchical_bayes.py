"""Pooling hierárquico bayesiano de demanda de leitos (Hospital/Região ⊂ Bacia).

Modelo generativo em log-taxa para contagem de leitos ocupados/dia, com efeitos
aleatórios não-centrados de entidade e bacia, sazonalidade de Fourier, tendência
suave e inclinação de choque hidrológico ex-ante por bacia.

Amostragem em CPU via `nutpie` (NUTS compilado), diagnósticos com `ArviZ`.
Nenhuma informação do holdout (`ano_epi >= 2025`) é lida: a validação lança
`LeakageViolationError` antes de qualquer estatística de agrupamento.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from previsao_leitos.domain import DataContractError, LeakageViolationError

_ENTITY_COL = "nome_regiao_saude"
_BASIN_COL = "bacia_id"
_CAPACITY_COL = "capacidade_sem"
_TARGET_COL = "target_y"
_SHOCK_COL = "prob_enchente_f03_oof"
_YEAR_COL = "target_ano_epi"
_WEEK_COL = "target_semana_epi"

_REQUIRED_COLUMNS: tuple[str, ...] = (
    _ENTITY_COL,
    _BASIN_COL,
    _CAPACITY_COL,
    _TARGET_COL,
    _SHOCK_COL,
    _YEAR_COL,
    _WEEK_COL,
)


@dataclass(frozen=True)
class HierarchicalBayesConfig:
    """Configuração auditável da amostragem (CPU) e dos gates de vazamento."""

    n_chains: int = 4
    n_draws: int = 1000
    n_tune: int = 1000
    target_accept: float = 0.95
    random_seed: int = 42
    n_harmonics: int = 2
    nuts_sampler: str = "nutpie"
    mp_ctx: str = "spawn"
    train_end_year: int = 2022
    holdout_year: int = 2025
    hierarchical_shock_slope: bool = False
    include_trend: bool = False


@dataclass(frozen=True)
class HierarchicalBayesDiagnostics:
    """Gates mecânicos de convergência e calibração preditiva."""

    r_hat_max: float
    ess_bulk_min: float
    divergences: int
    loo_elpd: float
    pareto_k_max: float
    n_obs: int

    @property
    def passes(self) -> bool:
        """`True` apenas se todos os gates da Task 1.2.6 forem satisfeitos."""
        return (
            self.r_hat_max < 1.01
            and self.ess_bulk_min >= 400
            and self.divergences == 0
            and np.isfinite(self.loo_elpd)
            and self.pareto_k_max < 0.7
        )


def validate_hierarchical_panel(
    panel: pl.DataFrame,
    *,
    config: HierarchicalBayesConfig | None = None,
) -> pl.DataFrame:
    """Aplica os contratos de dados e vazamento antes de qualquer amostragem.

    Retorna o painel filtrado em `ano_epi <= train_end_year` e ordenado de forma
    determinística por `(entidade, ano, semana)`.
    """
    cfg = config or HierarchicalBayesConfig()
    missing = [col for col in _REQUIRED_COLUMNS if col not in panel.columns]
    if missing:
        raise DataContractError(f"Painel hierárquico sem colunas obrigatórias: {missing}")
    if panel.height == 0:
        raise DataContractError("Painel hierárquico vazio.")

    max_year = panel[_YEAR_COL].max()
    if max_year is not None and int(np.asarray(max_year).item()) >= cfg.holdout_year:
        raise LeakageViolationError(
            f"Painel hierárquico contém ano_epi={int(np.asarray(max_year).item())} "
            f">= {cfg.holdout_year}; "
            "o holdout permanece lacrado até a Task 1.3.0."
        )

    if bool((panel[_CAPACITY_COL] <= 0).any()):
        raise DataContractError("capacidade_sem deve ser estritamente positiva.")
    if bool((panel[_TARGET_COL] < 0).any()):
        raise DataContractError("target_y (leitos ocupados/dia) não pode ser negativo.")

    filtered = panel.filter(pl.col(_YEAR_COL) <= cfg.train_end_year)
    if filtered.height == 0:
        raise DataContractError(
            f"Painel hierárquico sem linhas até train_end_year={cfg.train_end_year}."
        )
    return filtered.sort([_ENTITY_COL, _YEAR_COL, _WEEK_COL])


@dataclass(frozen=True)
class _HierarchicalDesign:
    """Matrizes alinhadas ao painel validado (sem qualquer estatística do holdout)."""

    y: np.ndarray
    log_capacity: np.ndarray
    entity_index: np.ndarray
    basin_index: np.ndarray
    entity_basin_index: np.ndarray
    shock: np.ndarray
    fourier: np.ndarray
    trend: np.ndarray
    entities: tuple[str, ...]
    basins: tuple[str, ...]


def _build_design(
    panel: pl.DataFrame, *, config: HierarchicalBayesConfig
) -> _HierarchicalDesign:
    entities = tuple(sorted(panel[_ENTITY_COL].unique().to_list()))
    basins = tuple(sorted(panel[_BASIN_COL].unique().to_list()))
    entity_index = np.array(
        [entities.index(e) for e in panel[_ENTITY_COL].to_list()], dtype=np.int64
    )
    basin_index = np.array(
        [basins.index(b) for b in panel[_BASIN_COL].to_list()], dtype=np.int64
    )

    # Mapa entidade -> bacia (cada entidade pertence a exatamente uma bacia).
    entity_basin_index = np.zeros(len(entities), dtype=np.int64)
    for entity, basin in zip(
        panel[_ENTITY_COL].to_list(), panel[_BASIN_COL].to_list(), strict=True
    ):
        entity_basin_index[entities.index(entity)] = basins.index(basin)

    weeks = panel[_WEEK_COL].to_numpy().astype(np.float64)
    years = panel[_YEAR_COL].to_numpy().astype(np.float64)
    harmonics = [
        fn(2.0 * np.pi * k * weeks / 52.0)
        for k in range(1, config.n_harmonics + 1)
        for fn in (np.sin, np.cos)
    ]
    fourier = np.column_stack(harmonics) if harmonics else np.empty((panel.height, 0))

    week_global = (years - float(years.min())) * 53.0 + weeks
    trend = (week_global - float(week_global.mean())) / max(1.0, float(week_global.std(ddof=1)))

    # Centrar o choque desacopla `beta0` da inclinação e melhora a geometria do NUTS.
    shock = panel[_SHOCK_COL].fill_null(0.0).to_numpy().astype(np.float64)
    shock = shock - float(shock.mean())

    capacity = panel[_CAPACITY_COL].to_numpy().astype(np.float64)
    return _HierarchicalDesign(
        y=panel[_TARGET_COL].to_numpy().astype(np.float64),
        log_capacity=np.log(np.maximum(capacity, 1.0)),
        entity_index=entity_index,
        basin_index=basin_index,
        entity_basin_index=entity_basin_index,
        shock=shock,
        fourier=np.asarray(fourier, dtype=np.float64),
        trend=np.asarray(trend, dtype=np.float64),
        entities=entities,
        basins=basins,
    )


def build_hierarchical_model(
    panel: pl.DataFrame,
    *,
    config: HierarchicalBayesConfig | None = None,
) -> Any:
    """Constrói (sem amostrar) o modelo PyMC de pooling parcial."""
    import pymc as pm  # lazy: preserva C-07 e o CI em `--extra cpu`

    cfg = config or HierarchicalBayesConfig()
    validated = validate_hierarchical_panel(panel, config=cfg)
    design = _build_design(validated, config=cfg)

    coords = {
        "entity": list(design.entities),
        "basin": list(design.basins),
        "obs": np.arange(validated.height),
    }

    with pm.Model(coords=coords) as model:
        beta0 = pm.Normal("beta0", mu=-0.3, sigma=1.0)
        sigma_entity = pm.HalfNormal("sigma_entity", sigma=0.5)
        sigma_basin = pm.HalfNormal("sigma_basin", sigma=0.5)

        raw_entity = pm.Normal("raw_entity", mu=0.0, sigma=1.0, dims="entity")
        raw_basin = pm.Normal("raw_basin", mu=0.0, sigma=1.0, dims="basin")
        # Interceptos aninhados: a entidade é desvio em torno da sua bacia (evita a
        # não-identificabilidade de soma de dois interceptos aleatórios independentes).
        u_basin = pm.Deterministic(
            "u_basin", (raw_basin - raw_basin.mean()) * sigma_basin, dims="basin"
        )
        u_entity = pm.Deterministic(
            "u_entity",
            u_basin[design.entity_basin_index] + raw_entity * sigma_entity,
            dims="entity",
        )

        n_harmonics = cfg.n_harmonics
        beta_fourier = pm.Normal("beta_fourier", mu=0.0, sigma=0.5, shape=2 * n_harmonics)

        log_rate = (
            beta0
            + u_entity[design.entity_index]
            + pm.math.dot(design.fourier, beta_fourier)
        )

        if cfg.include_trend:
            # Tendência só é identificável com mais de dois ciclos anuais; ligada por
            # configuração explícita para não confundir com sazonalidade.
            eta_trend = pm.Normal("eta_trend", mu=0.0, sigma=0.1)
            log_rate = log_rate + eta_trend * design.trend

        # ADR-0001: enchente nunca reduz demanda. A inclinação do choque é estritamente
        # positiva por construção; um `Normal` permitia slope negativo e violava o gate.
        if cfg.hierarchical_shock_slope:
            sigma_delta = pm.HalfNormal("sigma_delta", sigma=0.3)
            delta_basin = pm.HalfNormal("delta_basin", sigma=sigma_delta, dims="basin")
            log_rate = log_rate + delta_basin[design.basin_index] * design.shock
        else:
            delta_shock = pm.HalfNormal("delta_shock", sigma=0.3)
            log_rate = log_rate + delta_shock * design.shock

        alpha = pm.LogNormal("alpha", mu=4.0, sigma=1.5)

        mu = pm.Deterministic("mu", pm.math.exp(design.log_capacity + log_rate), dims="obs")
        pm.NegativeBinomial("y", mu=mu, alpha=alpha, observed=design.y, dims="obs")

    return model


def _loo_elpd(loo_result: Any) -> float:
    """Extrai o ELPD do PSIS-LOO (ArviZ 1.x usa `.elpd`; versões antigas `.elpd_loo`)."""
    for attr in ("elpd", "elpd_loo"):
        value = getattr(loo_result, attr, None)
        if value is not None:
            return float(np.ravel(np.asarray(value))[0])
    raise DataContractError("Resultado de PSIS-LOO sem atributo de ELPD reconhecido.")


def fit_hierarchical_bayes(
    panel: pl.DataFrame,
    *,
    config: HierarchicalBayesConfig | None = None,
) -> tuple[Any, Any]:
    """Ajusta o modelo hierárquico e retorna `(modelo, InferenceData)`."""
    import pymc as pm  # lazy

    cfg = config or HierarchicalBayesConfig()
    model = build_hierarchical_model(panel, config=cfg)
    with model:
        idata = pm.sample(
            draws=cfg.n_draws,
            tune=cfg.n_tune,
            chains=cfg.n_chains,
            cores=cfg.n_chains,
            target_accept=cfg.target_accept,
            nuts_sampler=cfg.nuts_sampler,
            random_seed=cfg.random_seed,
            mp_ctx=cfg.mp_ctx,
            progressbar=False,
        )
    # `nutpie` não persiste o grupo de log-verossimilhança; PSIS-LOO exige calculá-lo.
    idata = pm.compute_log_likelihood(
        idata, model=model, extend_inferencedata=True, progressbar=False
    )
    return model, idata


def summarize_diagnostics(idata: Any) -> HierarchicalBayesDiagnostics:
    """Consolida R-hat, ESS, divergências e PSIS-LOO em estrutura auditável."""
    import arviz as az  # lazy

    groups = set(idata.groups)
    if not ({"log_likelihood", "/log_likelihood"} & groups):
        raise DataContractError(
            "InferenceData sem grupo 'log_likelihood'; rode `pm.compute_log_likelihood` antes."
        )

    # ArviZ 1.x expõe `idata.posterior` como DataTree; `.dataset` devolve o Dataset.
    posterior = idata.posterior.dataset
    r_hat = az.rhat(posterior).to_array().to_numpy()
    ess = az.ess(posterior, method="bulk").to_array().to_numpy()
    divergences = int(np.asarray(idata.sample_stats["diverging"].to_numpy()).sum())

    loo_result = az.loo(idata)
    pareto_k = np.asarray(loo_result.pareto_k.to_numpy(), dtype=np.float64)

    observed_group = getattr(idata.observed_data, "dataset", idata.observed_data)
    return HierarchicalBayesDiagnostics(
        r_hat_max=float(np.nanmax(r_hat)),
        ess_bulk_min=float(np.nanmin(ess)),
        divergences=divergences,
        loo_elpd=_loo_elpd(loo_result),
        pareto_k_max=float(np.nanmax(pareto_k)),
        n_obs=int(np.asarray(observed_group["y"]).size),
    )


def ppc_tail_probability(model: Any, idata: Any, observed: np.ndarray) -> float:
    """P-valor preditivo posterior por estatística de discrepância (0,5 = ajuste central).

    Usa `T(y) = mean((y - média_preditiva)²)` — testa simultaneamente vício e dispersão.
    Valores próximos de 0 ou 1 indicam que as réplicas são sistematicamente mais
    concentradas/dispersas que os dados observados.
    """
    import pymc as pm  # lazy

    observed_arr = np.asarray(observed, dtype=np.float64)
    posterior_predictive = pm.sample_posterior_predictive(
        idata, model=model, progressbar=False, extend_inferencedata=True
    )
    sims = posterior_predictive.posterior_predictive["y"].to_numpy()
    flat = sims.reshape(-1, sims.shape[-1])
    if flat.shape[-1] != observed_arr.size:
        raise DataContractError(
            f"PPC com {flat.shape[-1]} pontos simulados para {observed_arr.size} observados."
        )
    predictive_mean = flat.mean(axis=0)
    t_obs = float(np.mean((observed_arr - predictive_mean) ** 2))
    t_rep = np.mean((flat - predictive_mean) ** 2, axis=1)
    return float(np.mean(t_rep >= t_obs))
