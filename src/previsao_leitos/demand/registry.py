"""Registro único de candidatos de demanda (baselines, campeões e challengers).

Evita que cada script instancie modelos com hiperparâmetros divergentes. Cada
candidato declara família, modo de calibração (`plain`, `train_cal`, `cal_only`)
e dependências opcionais; o ambiente decide se ele está disponível.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from previsao_leitos.demand.baselines import (
    PersistenceForecaster,
    SeasonalNaiveForecaster,
)
from previsao_leitos.demand.calibration import ConformalFloodCalibrator
from previsao_leitos.demand.models import (
    DemandForecaster,
    NegativeBinomialHurdleForecaster,
    PooledRidgeDeltaForecaster,
    QuantileGBDTForecaster,
)
from previsao_leitos.demand.two_stage import OrthogonalTwoStageOccupancyForecaster
from previsao_leitos.domain import DataContractError

#: Prefixos monotônicos do GBDT campeão `V1` (ADR-0001: enchente não reduz demanda).
_GBDT_MONO_POS: tuple[str, ...] = (
    "chuva_montante_multihop",
    "chuva_sem_mm",
    "api_max_sem",
    "prob_enchente_f03",
    "severidade_hidro_f03",
    "remanso_lagunar_idx",
    "exposicao_aterro_ativa",
    "el_nino_intensity",
)


@dataclass(frozen=True)
class CandidateSpec:
    """Metadados auditáveis de um candidato do benchmark."""

    name: str
    family: str
    fit_mode: str
    factory: Callable[[], DemandForecaster]
    requires: tuple[str, ...] = ()
    is_champion: bool = False
    online: bool = False
    description: str = ""

    @property
    def available(self) -> bool:
        return all(importlib.util.find_spec(mod) is not None for mod in self.requires)


def _two_stage() -> OrthogonalTwoStageOccupancyForecaster:
    return OrthogonalTwoStageOccupancyForecaster(
        capacity_col="capacidade_sem",
        basal_cols=("target_week_sin", "target_week_cos"),
        min_samples_leaf=8,
        random_state=42,
    )


def _two_stage_local_level() -> OrthogonalTwoStageOccupancyForecaster:
    return OrthogonalTwoStageOccupancyForecaster(
        capacity_col="capacidade_sem",
        basal_cols=("target_week_sin", "target_week_cos"),
        min_samples_leaf=8,
        random_state=42,
        use_local_level=True,
    )


def _ridge_pooled() -> PooledRidgeDeltaForecaster:
    return PooledRidgeDeltaForecaster()


def _ensemble_top3() -> DemandForecaster:
    from previsao_leitos.demand.ensemble import EnsembleForecaster

    return EnsembleForecaster(
        members=(
            ("gbdt_v1", _gbdt_v1),
            ("ridge_pooled", _ridge_pooled),
            ("f041_local_level", _two_stage_local_level),
        )
    )


def _gbdt_v1() -> QuantileGBDTForecaster:
    return QuantileGBDTForecaster(
        monotone_positive_prefixes=_GBDT_MONO_POS,
        monotone_negative_prefixes=("frac_floresta",),
        max_iter=120,
        max_depth=5,
        random_state=42,
    )


def _mondrian() -> DemandForecaster:
    from previsao_leitos.demand.extreme_uncertainty import MondrianEnsoConformalCalibrator

    return MondrianEnsoConformalCalibrator(
        base_model=_two_stage_local_level(),
        group_cols=("porte_regional", "enso_regime"),
        use_evt_p95=True,
    )


def _agaci() -> DemandForecaster:
    from previsao_leitos.demand.adaptive_conformal import AgACIConformalCalibrator

    return AgACIConformalCalibrator(base_model=_two_stage_local_level())


def _gbdt_agaci() -> DemandForecaster:
    from previsao_leitos.demand.adaptive_conformal import AgACIConformalCalibrator

    return AgACIConformalCalibrator(base_model=_gbdt_v1())


def _gbdt_spci() -> DemandForecaster:
    from previsao_leitos.demand.adaptive_conformal import SPCIConformalCalibrator

    return SPCIConformalCalibrator(base_model=_gbdt_v1())


def _spci() -> DemandForecaster:
    from previsao_leitos.demand.adaptive_conformal import SPCIConformalCalibrator

    return SPCIConformalCalibrator(base_model=_two_stage_local_level())


def _hierarchical() -> DemandForecaster:
    from previsao_leitos.demand.hierarchical_forecaster import HierarchicalBayesForecaster

    return HierarchicalBayesForecaster()


_CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        name="persistencia",
        family="baseline",
        fit_mode="plain",
        factory=PersistenceForecaster,
        description="Repete a ocupação da semana de origem (t_0).",
    ),
    CandidateSpec(
        name="sazonal_ingenuo",
        family="baseline",
        fit_mode="plain",
        factory=SeasonalNaiveForecaster,
        description="Repete a mesma semana do ano anterior (lag 52).",
    ),
    CandidateSpec(
        name="ridge_pooled",
        family="linear",
        fit_mode="plain",
        factory=PooledRidgeDeltaForecaster,
        description="Ridge global sobre delta com quantis empíricos.",
    ),
    CandidateSpec(
        name="negbin_hurdle",
        family="contagem",
        fit_mode="plain",
        factory=NegativeBinomialHurdleForecaster,
        description="Hurdle + Binomial Negativa para agravos esparsos.",
    ),
    CandidateSpec(
        name="gbdt_v1",
        family="gbdt",
        fit_mode="plain",
        factory=_gbdt_v1,
        description="Campeão V1: GBDT monotônico location-scale.",
    ),
    CandidateSpec(
        name="gbdt_conformal",
        family="gbdt+conformal",
        fit_mode="train_cal",
        factory=lambda: ConformalFloodCalibrator(base_model=_gbdt_v1()),
        is_champion=True,
        description="Campeão V1 auditado: GBDT monotônico + conformal (2023).",
    ),
    CandidateSpec(
        name="gbdt_agaci",
        family="gbdt+conformal-online",
        fit_mode="cal_only",
        factory=_gbdt_agaci,
        online=True,
        description="Campeão V1 + AgACI adaptativo normalizado por CNES (recalibra em 2024).",
    ),
    CandidateSpec(
        name="gbdt_spci",
        family="gbdt+conformal-online",
        fit_mode="cal_only",
        factory=_gbdt_spci,
        online=True,
        description="Campeão V1 + SPCI (QRF em resíduos, recalibra em 2024 com lag 2).",
    ),
    CandidateSpec(
        name="f041_core",
        family="2-estagios",
        fit_mode="plain",
        factory=_two_stage,
        description="Núcleo ortogonal F04.1 (basal + choque monotônico).",
    ),
    CandidateSpec(
        name="ensemble_top3",
        family="ensemble",
        fit_mode="train_cal",
        factory=_ensemble_top3,
        description="Média ponderada (pesos do ano de calibração) de GBDT, Ridge e F04.1.",
    ),
    CandidateSpec(
        name="f041_local_level",
        family="2-estagios+nivel-local",
        fit_mode="plain",
        factory=_two_stage_local_level,
        description="F04.1 com nível local suavizado (y_roll12/capacidade) no Estágio 1.",
    ),
    CandidateSpec(
        name="f041_mondrian",
        family="2-estagios+conformal",
        fit_mode="cal_only",
        factory=_mondrian,
        description="Challenger V1.2: F04.1 + Mondrian ENSO/porte com cauda EVT.",
    ),
    CandidateSpec(
        name="f041_agaci",
        family="2-estagios+conformal",
        fit_mode="cal_only",
        factory=_agaci,
        description="Challenger 1.2.4: F04.1 + AgACI adaptativo normalizado por CNES.",
    ),
    CandidateSpec(
        name="f041_spci",
        family="2-estagios+conformal",
        fit_mode="cal_only",
        factory=_spci,
        description="Challenger 1.2.4: F04.1 + SPCI (QRF em resíduos defasados).",
    ),
    CandidateSpec(
        name="hierarchical_bayes",
        family="bayes",
        fit_mode="plain",
        factory=_hierarchical,
        requires=("pymc", "arviz"),
        description="Challenger 1.2.6: pooling parcial Região ⊂ Bacia (Negative Binomial).",
    ),
)

_BY_NAME: dict[str, CandidateSpec] = {spec.name: spec for spec in _CANDIDATES}

_CHAMPION = next(spec for spec in _CANDIDATES if spec.is_champion)


def list_candidates() -> tuple[CandidateSpec, ...]:
    """Todos os candidatos declarados, na ordem canônica do relatório."""
    return _CANDIDATES


def get_candidate(name: str) -> CandidateSpec:
    if name not in _BY_NAME:
        raise DataContractError(
            f"Candidato desconhecido: '{name}'. Disponíveis: {sorted(_BY_NAME)}."
        )
    return _BY_NAME[name]


def resolve_selection(
    selection: str | Sequence[str] = "all",
) -> tuple[CandidateSpec, ...]:
    """Resolve `all`, `champion`, `baselines` ou uma lista explícita de nomes."""
    if isinstance(selection, str):
        if selection == "all":
            return _CANDIDATES
        if selection == "champion":
            return (_CHAMPION,)
        if selection == "baselines":
            return tuple(spec for spec in _CANDIDATES if spec.family == "baseline")
        names = tuple(part.strip() for part in selection.split(",") if part.strip())
    else:
        names = tuple(selection)
    if not names:
        raise DataContractError("Seleção de candidatos vazia.")
    return tuple(get_candidate(name) for name in names)


def champion_spec() -> CandidateSpec:
    """Campeão registrado (usado pela catraca Champion-Challenger)."""
    return _CHAMPION
