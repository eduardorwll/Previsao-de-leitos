"""Montagem do painel de modelagem e partições causais por alvo de leitos.

Centraliza o que antes vivia apenas em scripts: unidade de cada alvo, capacidade
semanal correspondente, porte regional causal e as três partições
(`train <= 2022`, `cal = eval_year - 1`, `eval = eval_year`).

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from previsao_leitos.demand.datasets import prepare_global_demand_splits
from previsao_leitos.domain import DataContractError
from previsao_leitos.geo import SplitPolicy

#: Colunas dinâmicas ex-ante usadas por todos os modelos de demanda.
DYNAMIC_FEATURE_COLS: tuple[str, ...] = (
    "chuva_sem_mm",
    "chuva_montante_multihop",
    "api_max_sem",
    "remanso_lagunar_idx",
    "exposicao_aterro_ativa",
    "prob_enchente_f03_oof",
    "severidade_hidro_f03_oof",
    "el_nino_intensity",
)

#: Colunas estáticas (atributos estruturais da região).
STATIC_FEATURE_COLS: tuple[str, ...] = (
    "frac_urbano",
    "frac_floresta",
    "indice_aterro_varzea",
    "capacidade_sem",
)

#: Dengue (SINAN agregado por região/semana), opcional via `dengue_features=True`.
DENGUE_FEATURE_COLS: tuple[str, ...] = ("dengue_casos",)

#: Internações por causa (momentum assistencial), opcional via `admissions_momentum=True`.
ADMISSIONS_MOMENTUM_COLS: tuple[str, ...] = (
    "internacoes_total",
    "respiratorio_internacoes",
    "leptospirose_internacoes",
    "trauma_internacoes",
)

#: Memória causal do choque (Task 1.3.5), opcional via `shock_memory=True`.
SHOCK_MEMORY_COLS: tuple[str, ...] = (
    "shock_cum_4w",
    "shock_cum_8w",
    "shock_max_8w",
    "shock_ewm_4w",
)

#: RONI (índice relativo do CPC, Task 1.3.10 / E1), opcional via `roni_features=True`.
RONI_FEATURE_COLS: tuple[str, ...] = ("roni_c",)

#: Colunas que precisam existir no painel montado antes do split.
REQUIRED_MODELING_COLS: tuple[str, ...] = (
    "nome_regiao_saude",
    "bacia_id",
    "ano_epi",
    "semana_epi",
    *DYNAMIC_FEATURE_COLS,
    "cnes_uti_sus_leitos",
    "cnes_total_sus_leitos",
    "enso_regime",
)


@dataclass(frozen=True)
class TargetSpec:
    """Metadados de um alvo de leitos: coluna, unidade e capacidade correspondente."""

    key: str
    column: str
    label: str
    unit: str
    capacity_column: str
    capacity_scale: float
    large_region_threshold: float


TARGET_SPECS: dict[str, TargetSpec] = {
    "uti_adulto": TargetSpec(
        key="uti_adulto",
        column="uti_adulto_diarias",
        label="UTI adulto",
        unit="leitos-dia/semana",
        capacity_column="cnes_uti_sus_leitos",
        capacity_scale=7.0,
        large_region_threshold=250.0,
    ),
    "uti_total": TargetSpec(
        key="uti_total",
        column="diarias_uti_total_ativas",
        label="UTI total",
        unit="leitos-dia/semana",
        capacity_column="cnes_uti_sus_leitos",
        capacity_scale=7.0,
        large_region_threshold=250.0,
    ),
    "clinico": TargetSpec(
        key="clinico",
        column="clinico_pacientes_dia",
        label="Clínico",
        unit="pacientes-dia/dia",
        capacity_column="cnes_total_sus_leitos",
        capacity_scale=1.0,
        large_region_threshold=800.0,
    ),
    "respiratorio": TargetSpec(
        key="respiratorio",
        column="respiratorio_pacientes_dia",
        label="Respiratório",
        unit="pacientes-dia/dia",
        capacity_column="cnes_total_sus_leitos",
        capacity_scale=1.0,
        large_region_threshold=800.0,
    ),
}


#: Alvos de **necessidade** (pré-registro v3), derivados dos alvos de ocupação.
NEED_TARGET_SPECS: dict[str, TargetSpec] = {
    f"need_{key}": TargetSpec(
        key=f"need_{key}",
        column=f"need_{key}",
        label=f"Necessidade — {spec.label}",
        unit=spec.unit,
        capacity_column=spec.capacity_column,
        capacity_scale=spec.capacity_scale,
        large_region_threshold=spec.large_region_threshold,
    )
    for key, spec in TARGET_SPECS.items()
}

#: Catálogo completo (ocupação + necessidade).
ALL_TARGET_SPECS: dict[str, TargetSpec] = {**TARGET_SPECS, **NEED_TARGET_SPECS}


@dataclass(frozen=True)
class ModelingSplits:
    """Partições causais prontas para treino, calibração conformal e avaliação."""

    train_df: pl.DataFrame
    cal_df: pl.DataFrame
    eval_df: pl.DataFrame
    feature_cols: tuple[str, ...]
    entity_col: str
    target_col: str
    target_key: str
    eval_year: int
    horizon_weeks: int


def resolve_targets(selection: str | Sequence[str]) -> tuple[TargetSpec, ...]:
    """Resolve `all`, uma chave ou uma lista de chaves de alvo."""
    if isinstance(selection, str):
        keys = tuple(TARGET_SPECS) if selection == "all" else (selection,)
    else:
        keys = tuple(selection)
    unknown = [k for k in keys if k not in ALL_TARGET_SPECS]
    if unknown:
        raise DataContractError(
            f"Alvos desconhecidos: {unknown}. Disponíveis: {sorted(ALL_TARGET_SPECS)}."
        )
    return tuple(ALL_TARGET_SPECS[k] for k in keys)


def attach_capacity(panel: pl.DataFrame, spec: TargetSpec) -> pl.DataFrame:
    """Deriva `capacidade_sem` na unidade do alvo, com fallback causal (`<= 2022`)."""
    if spec.capacity_column not in panel.columns:
        raise DataContractError(
            f"Coluna de capacidade '{spec.capacity_column}' ausente para o alvo '{spec.key}'."
        )
    fallback = (
        panel.filter(pl.col("ano_epi") <= 2022)
        .group_by("nome_regiao_saude")
        .agg(pl.col(spec.column).mean().alias("_mean_tr"))
        .with_columns(
            pl.when(pl.col("_mean_tr") >= spec.large_region_threshold)
            .then(pl.lit("grande"))
            .otherwise(pl.lit("interior"))
            .alias("porte_regional"),
            pl.col("_mean_tr").clip(lower_bound=1.0).alias("_fallback_cap"),
        )
        .select(["nome_regiao_saude", "porte_regional", "_fallback_cap"])
    )
    return panel.join(fallback, on="nome_regiao_saude", how="left").with_columns(
        pl.when(
            pl.col(spec.capacity_column).is_not_null() & (pl.col(spec.capacity_column) > 0.0)
        )
        .then(pl.col(spec.capacity_column) * spec.capacity_scale)
        .otherwise(pl.col("_fallback_cap"))
        .alias("capacidade_sem")
    )


def validate_modeling_panel(panel: pl.DataFrame, spec: TargetSpec) -> None:
    """Falha cedo e com mensagem acionável se o painel montado estiver incompleto."""
    missing = [c for c in REQUIRED_MODELING_COLS if c not in panel.columns]
    if spec.column not in panel.columns:
        missing.append(spec.column)
    if missing:
        raise DataContractError(
            f"Painel de modelagem incompleto para '{spec.key}': faltam {sorted(set(missing))}. "
            "Gere-o com `scripts/build_rs_modeling_panel.py`."
        )


def make_modeling_splits(
    panel: pl.DataFrame,
    *,
    target: str | TargetSpec = "uti_adulto",
    eval_year: int = 2024,
    horizon_weeks: int = 2,
    policy: SplitPolicy | None = None,
    allow_holdout: bool = False,
    shock_memory: bool = False,
    admissions_momentum: bool = False,
    dengue_features: bool = False,
    roni_features: bool = False,
) -> ModelingSplits:
    """Constrói `train`/`cal`/`eval` causais para um alvo, sem tocar no holdout."""
    spec = ALL_TARGET_SPECS[target] if isinstance(target, str) else target
    active_policy = policy or SplitPolicy()
    if eval_year >= active_policy.holdout_year and not allow_holdout:
        raise DataContractError(
            f"eval_year={eval_year} atinge o holdout ({active_policy.holdout_year}); "
            "reservado ao Marco M2 com `--allow-holdout`."
        )
    validate_modeling_panel(panel, spec)

    enriched = attach_capacity(panel, spec)
    dynamic_cols: tuple[str, ...] = DYNAMIC_FEATURE_COLS
    if admissions_momentum:
        missing_adm = [c for c in ADMISSIONS_MOMENTUM_COLS if c not in enriched.columns]
        if missing_adm:
            raise DataContractError(
                f"Momentum de internações pedido, mas faltam {missing_adm} no painel."
            )
        dynamic_cols = (*DYNAMIC_FEATURE_COLS, *ADMISSIONS_MOMENTUM_COLS)
    if dengue_features:
        missing_dengue = [c for c in DENGUE_FEATURE_COLS if c not in enriched.columns]
        if missing_dengue:
            raise DataContractError(
                f"Dengue pedida, mas faltam {missing_dengue}. "
                "Rode o runbook SINAN e regere o painel."
            )
        dynamic_cols = (*dynamic_cols, *DENGUE_FEATURE_COLS)
    if roni_features:
        missing_roni = [c for c in RONI_FEATURE_COLS if c not in enriched.columns]
        if missing_roni:
            raise DataContractError(
                f"RONI pedido, mas faltam {missing_roni}. "
                "Regere o painel com `scripts/build_rs_modeling_panel.py`."
            )
        dynamic_cols = (*dynamic_cols, *RONI_FEATURE_COLS)
    extra_cols: tuple[str, ...] = ()
    if shock_memory:
        missing = [c for c in SHOCK_MEMORY_COLS if c not in enriched.columns]
        if missing:
            raise DataContractError(
                f"Memória do choque pedida, mas faltam {missing}. "
                "Regere o painel com `scripts/build_rs_modeling_panel.py`."
            )
        extra_cols = SHOCK_MEMORY_COLS
    origin_meta = enriched.select(
        ["nome_regiao_saude", "ano_epi", "semana_epi", "porte_regional", "enso_regime", "bacia_id"]
    )

    cal_year = eval_year - 1
    splits_cal = prepare_global_demand_splits(
        enriched,
        entity_col="nome_regiao_saude",
        target_col=spec.column,
        dynamic_feature_cols=dynamic_cols,
        static_feature_cols=STATIC_FEATURE_COLS,
        extra_feature_cols=extra_cols,
        horizon_weeks=horizon_weeks,
        lag_weeks=(0, 1, 2),
        eval_year=cal_year,
        policy=active_policy,
        allow_holdout=allow_holdout,
    )
    splits_eval = prepare_global_demand_splits(
        enriched,
        entity_col="nome_regiao_saude",
        target_col=spec.column,
        dynamic_feature_cols=dynamic_cols,
        static_feature_cols=STATIC_FEATURE_COLS,
        extra_feature_cols=extra_cols,
        horizon_weeks=horizon_weeks,
        lag_weeks=(0, 1, 2),
        eval_year=eval_year,
        policy=active_policy,
        allow_holdout=allow_holdout,
    )

    join_keys = ["nome_regiao_saude", "ano_epi", "semana_epi"]
    cal_df = splits_cal.eval_df.join(origin_meta, on=join_keys, how="left")
    eval_df = splits_eval.eval_df.join(origin_meta, on=join_keys, how="left")
    return ModelingSplits(
        train_df=splits_eval.train_df,
        cal_df=cal_df,
        eval_df=eval_df,
        feature_cols=splits_eval.feature_cols,
        entity_col="nome_regiao_saude",
        target_col=spec.column,
        target_key=spec.key,
        eval_year=eval_year,
        horizon_weeks=horizon_weeks,
    )


def summarise_unavailable(reason: str) -> str:
    """Mensagem padronizada para candidatos que não podem rodar no ambiente."""
    return f"indisponível: {reason}"


def is_finite(value: float) -> bool:
    return isinstance(value, float) and math.isfinite(value)
