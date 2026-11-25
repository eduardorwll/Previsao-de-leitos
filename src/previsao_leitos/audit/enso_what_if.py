"""Oráculo prospectivo de cenários El Niño (what-if) e Rebate Contrafactual Duplo 2024.

Combina o motor BSTS Pré-Choque (com controle SIVEP-Gripe), a intervenção
dinâmica recursiva `do(Clima=Climatologia, ENSO=Neutro)` e cenários ordenados
de intensidade de El Niño para auditar o choque de 2024 e projetar regimes futuros.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from previsao_leitos.audit.bsts_counterfactual import (
    PreShockBstsCounterfactual,
    recursive_dynamic_do_intervention,
)
from previsao_leitos.demand.models import DemandForecaster
from previsao_leitos.domain import DataContractError


@dataclass(frozen=True)
class EnsoScenarioSpec:
    """Especificação de um cenário prospectivo de regime El Niño / chuva extrema."""

    name: str
    el_nino_intensity: float
    rain_multiplier: float
    flood_prob_floor: float


DEFAULT_ENSO_SCENARIOS: tuple[EnsoScenarioSpec, ...] = (
    EnsoScenarioSpec(
        name="neutro",
        el_nino_intensity=0.0,
        rain_multiplier=1.0,
        flood_prob_floor=0.05,
    ),
    EnsoScenarioSpec(
        name="el_nino_moderado",
        el_nino_intensity=1.2,
        rain_multiplier=1.35,
        flood_prob_floor=0.35,
    ),
    EnsoScenarioSpec(
        name="el_nino_forte",
        el_nino_intensity=2.0,
        rain_multiplier=1.80,
        flood_prob_floor=0.70,
    ),
    EnsoScenarioSpec(
        name="super_el_nino",
        el_nino_intensity=2.5,
        rain_multiplier=2.20,
        flood_prob_floor=0.90,
    ),
)


@dataclass(frozen=True)
class DualRebuttalSummary:
    """Resultado consolidado do rebate contrafactual duplo (BSTS + do() recursivo)."""

    shock_excess_obs_mean: float
    shock_excess_pred_mean: float
    shock_excess_mae: float
    peak_timing_error_weeks: float
    summary_table: pl.DataFrame
    pre_shock_bias: float = 0.0


def _matches_prefix(col: str, prefixes: tuple[str, ...]) -> bool:
    return any(col == p or col.startswith(f"{p}_lag") for p in prefixes)


def run_enso_what_if_oracle(
    *,
    model: DemandForecaster,
    baseline_eval_df: pl.DataFrame,
    scenarios: Sequence[EnsoScenarioSpec] = DEFAULT_ENSO_SCENARIOS,
) -> pl.DataFrame:
    """Simula cenários prospectivos de El Niño perturbando colunas-base e defasadas (`*_lag*`)."""
    if baseline_eval_df.is_empty():
        raise DataContractError("baseline_eval_df vazio em run_enso_what_if_oracle.")
    if not scenarios:
        raise DataContractError("scenarios vazio em run_enso_what_if_oracle.")

    enso_cols = [
        c
        for c in baseline_eval_df.columns
        if _matches_prefix(c, ("el_nino_intensity", "oni_c"))
    ]
    rain_cols = [
        c
        for c in baseline_eval_df.columns
        if _matches_prefix(
            c,
            ("chuva_montante_pond", "chuva_montante_multihop", "chuva_sem_mm"),
        )
    ]
    prob_cols = [
        c
        for c in baseline_eval_df.columns
        if _matches_prefix(c, ("prob_enchente_f03_oof", "prob_enchente_f03"))
    ]
    sev_cols = [
        c
        for c in baseline_eval_df.columns
        if _matches_prefix(c, ("severidade_hidro_f03_oof", "severidade_hidro_f03"))
    ]

    rows: list[dict[str, object]] = []
    neutral_p50: float | None = None
    neutral_p90: float | None = None

    for scen in scenarios:
        mods: list[pl.Expr] = []
        for col in enso_cols:
            mods.append(pl.lit(scen.el_nino_intensity, dtype=pl.Float64).alias(col))
        for col in rain_cols:
            mods.append(
                (pl.col(col).cast(pl.Float64) * scen.rain_multiplier).alias(col)
            )
        for col in prob_cols:
            mods.append(
                pl.col(col)
                .cast(pl.Float64)
                .clip(lower_bound=scen.flood_prob_floor, upper_bound=1.0)
                .alias(col)
            )
        for col in sev_cols:
            mods.append(
                pl.max_horizontal(
                    pl.col(col).cast(pl.Float64) * scen.rain_multiplier,
                    pl.lit(scen.flood_prob_floor, dtype=pl.Float64),
                ).alias(col)
            )

        scen_df = baseline_eval_df.with_columns(mods) if mods else baseline_eval_df.clone()
        preds = model.predict_quantiles(scen_df)
        p50_mean = float(preds["pred_p50"].to_numpy().mean())
        p90_peak = float(preds["pred_p90"].to_numpy().max())
        p95_peak = float(preds["pred_p95"].to_numpy().max())

        if neutral_p50 is None:
            neutral_p50 = p50_mean
            neutral_p90 = p90_peak

        rows.append(
            {
                "cenario": scen.name,
                "el_nino_intensity": scen.el_nino_intensity,
                "rain_multiplier": scen.rain_multiplier,
                "demanda_p50_media": round(p50_mean, 4),
                "demanda_p90_pico": round(p90_peak, 4),
                "demanda_p95_pico": round(p95_peak, 4),
                "excesso_sobre_neutro_p50": round(p50_mean - neutral_p50, 4),
                "excesso_sobre_neutro_p90": round(
                    p90_peak - (neutral_p90 if neutral_p90 is not None else p90_peak), 4
                ),
            }
        )

    return pl.DataFrame(rows)


def run_dual_counterfactual_rebuttal_2024(
    *,
    model: DemandForecaster,
    full_panel: pl.DataFrame,
    train_df: pl.DataFrame,
    eval_df: pl.DataFrame,
    entity_col: str = "nome_regiao_saude",
    target_col: str = "target_y",
    climate_prefixes: Sequence[str] = (),
    neutral_zero_prefixes: Sequence[str] = (),
    shock_year: int = 2024,
    shock_weeks: tuple[int, int] = (18, 30),
    horizon_weeks: int = 2,
) -> DualRebuttalSummary:
    """Executa rebate contrafactual duplo: BSTS Pré-Choque vs. do() dinâmico recursivo."""
    bsts = PreShockBstsCounterfactual(
        shock_year=shock_year,
        shock_start_week=shock_weeks[0],
        control_cols=("srag_viral_confirmado", "frac_srag_viral"),
    ).fit(full_panel, entity_col=entity_col, target_col=target_col)

    bsts_cf = bsts.predict_counterfactual(
        eval_df,
        entity_col=entity_col,
        target_col=target_col,
    )

    real_preds = model.predict_quantiles(eval_df)
    do_cf = recursive_dynamic_do_intervention(
        model=model,
        train_df=train_df,
        eval_df=eval_df,
        entity_col=entity_col,
        horizon_weeks=horizon_weeks,
        intervention_start_week=shock_weeks[0],
        intervention_start_year=shock_year,
        climate_prefixes=climate_prefixes,
        neutral_zero_prefixes=neutral_zero_prefixes,
    )

    merged = (
        eval_df.select([entity_col, "target_ano_epi", "target_semana_epi", target_col])
        .with_columns(
            real_preds["pred_p50"],
            real_preds["pred_p90"],
            do_cf["pred_p50_cf"],
            (real_preds["pred_p50"] - do_cf["pred_p50_cf"]).alias("excess_pred_do"),
        )
        .join(
            bsts_cf.select(
                [entity_col, "target_ano_epi", "target_semana_epi", "y_cf_no_flood", "excess_obs"]
            ),
            on=[entity_col, "target_ano_epi", "target_semana_epi"],
            how="inner",
        )
    )

    pre_win = merged.filter(
        pl.col("target_semana_epi").is_between(max(1, shock_weeks[0] - 3), shock_weeks[0] - 1)
    )
    pre_bias = (
        float(
            np.mean(
                pre_win["y_cf_no_flood"].to_numpy().astype(np.float64)
                - pre_win[target_col].to_numpy().astype(np.float64)
            )
        )
        if not pre_win.is_empty()
        else 0.0
    )

    shock = merged.filter(pl.col("target_semana_epi").is_between(shock_weeks[0], shock_weeks[1]))
    if shock.is_empty():
        raise DataContractError("Janela de choque vazia no rebate contrafactual duplo.")

    obs_ex = shock["excess_obs"].to_numpy().astype(np.float64)
    pred_ex = shock["excess_pred_do"].to_numpy().astype(np.float64)
    obs_mean = float(np.mean(obs_ex))
    pred_mean = float(np.mean(pred_ex))
    mae_ex = float(np.mean(np.abs(obs_ex - pred_ex)))

    timing_errs: list[float] = []
    for ent in sorted(shock[entity_col].unique().to_list()):
        sub = shock.filter(pl.col(entity_col) == ent)
        if sub.height >= 2:
            wk_obs = int(
                sub.sort(["excess_obs", "target_semana_epi"], descending=[True, False])[
                    "target_semana_epi"
                ][0]
            )
            wk_pred = int(
                sub.sort(["excess_pred_do", "target_semana_epi"], descending=[True, False])[
                    "target_semana_epi"
                ][0]
            )
            timing_errs.append(abs(float(wk_pred - wk_obs)))
    timing_err = float(np.mean(timing_errs)) if timing_errs else 0.0

    summary_table = pl.DataFrame(
        [
            {
                "janela": f"Choque Sem. {shock_weeks[0]}-{shock_weeks[1]} ({shock_year})",
                "vies_pre_choque_bsts": round(pre_bias, 4),
                "excesso_obs_bsts_medio": round(obs_mean, 4),
                "excesso_pred_do_medio": round(pred_mean, 4),
                "mae_excesso": round(mae_ex, 4),
                "erro_timing_pico_sem": round(timing_err, 4),
            }
        ]
    )

    return DualRebuttalSummary(
        shock_excess_obs_mean=obs_mean,
        shock_excess_pred_mean=pred_mean,
        shock_excess_mae=mae_ex,
        peak_timing_error_weeks=timing_err,
        summary_table=summary_table,
        pre_shock_bias=pre_bias,
    )
