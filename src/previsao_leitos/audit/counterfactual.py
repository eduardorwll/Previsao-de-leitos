"""Análise contrafactual estruturada (Controle Sintético, Placebo e Intervenções do())."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from previsao_leitos.demand.models import DemandForecaster
from previsao_leitos.domain import DataContractError


@dataclass(frozen=True)
class CounterfactualAccuracyReport:
    """Métricas de acurácia contrafactual em janela de choque + validação placebo."""

    shock_excess_mae: float
    shock_excess_bias: float
    peak_timing_error_weeks: float
    interval_coverage_90: float
    placebo_bias: float
    placebo_mae: float
    monotonic_violation_rate: float
    series_df: pl.DataFrame
    placebo_df: pl.DataFrame


class NoFloodSyntheticControl:
    """Estima trajetória contrafactual 'sem enchente' via controle sintético causal (<= 2022).

    Usa exclusivamente séries de regiões-doadoras não inundadas + harmônicos sazonais
    ajustados estritamente em `ano_epi <= train_end_year` (zero vazamento de 2024/2025).
    """

    def __init__(
        self,
        *,
        donor_entities: Sequence[str],
        train_end_year: int = 2022,
        ridge_alpha: float = 25.0,
    ) -> None:
        if not donor_entities:
            raise DataContractError(
                "Forneça ao menos uma região doadora para o controle sintético."
            )
        self._donors = tuple(donor_entities)
        self._train_end_year = train_end_year
        self._alpha = ridge_alpha
        self._weights: dict[str, np.ndarray] = {}
        self._res_sd: dict[str, float] = {}
        self._donor_cols: tuple[str, ...] = ()

    def _pivot_donors(
        self, full_panel: pl.DataFrame, *, entity_col: str, target_col: str
    ) -> pl.DataFrame:
        donor_sub = full_panel.filter(pl.col(entity_col).is_in(list(self._donors)))
        if donor_sub.is_empty():
            raise DataContractError("Nenhuma região doadora encontrada no painel fornecido.")
        pivoted = donor_sub.pivot(
            on=entity_col,
            index=["ano_epi", "semana_epi"],
            values=target_col,
            aggregate_function="mean",
        ).sort(["ano_epi", "semana_epi"])
        return pivoted

    def fit(
        self,
        full_panel: pl.DataFrame,
        *,
        entity_col: str,
        target_col: str,
    ) -> NoFloodSyntheticControl:
        pivoted = self._pivot_donors(full_panel, entity_col=entity_col, target_col=target_col)
        self._donor_cols = tuple(c for c in pivoted.columns if c not in ("ano_epi", "semana_epi"))

        train_donors = pivoted.filter(pl.col("ano_epi") <= self._train_end_year).drop_nulls()
        entities = sorted(full_panel[entity_col].unique().to_list())

        self._weights = {}
        self._res_sd = {}
        for ent in entities:
            ent_tr = (
                full_panel.filter(
                    (pl.col(entity_col) == ent) & (pl.col("ano_epi") <= self._train_end_year)
                )
                .select(["ano_epi", "semana_epi", target_col])
                .join(train_donors, on=["ano_epi", "semana_epi"], how="inner")
                .drop_nulls()
            )
            if ent_tr.height < 20:
                continue

            x_don = self._design_matrix(ent_tr)
            y_ent = ent_tr[target_col].to_numpy().astype(np.float64)
            reg = np.eye(x_don.shape[1], dtype=np.float64) * self._alpha
            reg[0, 0] = 0.0
            coef = np.linalg.solve(x_don.T @ x_don + reg, x_don.T @ y_ent)
            res = y_ent - (x_don @ coef)
            self._weights[ent] = coef
            self._res_sd[ent] = float(np.std(res, ddof=1)) if len(res) > 1 else 0.0
        return self

    def _design_matrix(self, df: pl.DataFrame) -> np.ndarray:
        week = df["semana_epi"].to_numpy().astype(np.float64)
        sin_w = np.sin(2.0 * np.pi * week / 52.0)[:, None]
        cos_w = np.cos(2.0 * np.pi * week / 52.0)[:, None]
        donors_mat = df.select(list(self._donor_cols)).to_numpy().astype(np.float64)
        ones = np.ones((df.height, 1), dtype=np.float64)
        return np.hstack([ones, donors_mat, sin_w, cos_w])

    def predict_no_flood(
        self,
        eval_df: pl.DataFrame,
        *,
        full_panel: pl.DataFrame,
        entity_col: str,
        target_col: str,
    ) -> pl.DataFrame:
        if not self._weights:
            raise DataContractError("NoFloodSyntheticControl precisa ser ajustado antes de prever.")
        pivoted = self._pivot_donors(full_panel, entity_col=entity_col, target_col=target_col)
        joined = eval_df.select(
            [entity_col, "target_ano_epi", "target_semana_epi", "target_y"]
        ).join(
            pivoted.rename({"ano_epi": "target_ano_epi", "semana_epi": "target_semana_epi"}),
            on=["target_ano_epi", "target_semana_epi"],
            how="left",
        )

        out_frames: list[pl.DataFrame] = []
        for ent, coef in self._weights.items():
            sub = joined.filter(pl.col(entity_col) == ent).drop_nulls(subset=list(self._donor_cols))
            if sub.is_empty():
                continue
            mat_df = sub.rename({"target_semana_epi": "semana_epi"})
            x_ev = self._design_matrix(mat_df)
            y_cf = np.maximum(0.0, x_ev @ coef)
            sd = self._res_sd[ent]
            out_frames.append(
                sub.select([entity_col, "target_ano_epi", "target_semana_epi", "target_y"])
                .with_columns(
                    pl.Series("y_cf_no_flood", y_cf, dtype=pl.Float64),
                    pl.Series("y_cf_p05", np.maximum(0.0, y_cf - 1.96 * sd), dtype=pl.Float64),
                    pl.Series("y_cf_p95", np.maximum(0.0, y_cf + 1.96 * sd), dtype=pl.Float64),
                )
                .with_columns((pl.col("target_y") - pl.col("y_cf_no_flood")).alias("excess_obs"))
            )
        if not out_frames:
            raise DataContractError("Nenhuma linha avaliável no controle sintético contrafactual.")
        return pl.concat(out_frames).sort([entity_col, "target_ano_epi", "target_semana_epi"])


def _climatology_feature_frame(
    train_df: pl.DataFrame,
    eval_df: pl.DataFrame,
    *,
    feature_prefixes: Sequence[str],
) -> pl.DataFrame:
    """Substitui covariáveis climáticas/hidrológicas pela mediana climatológica de treino (P50)."""
    target_cols = [
        c for c in eval_df.columns if any(c.startswith(prefix) for prefix in feature_prefixes)
    ]
    if not target_cols:
        return eval_df.clone()
    medians = {c: float(train_df.select(pl.col(c).median()).item()) for c in target_cols}
    return eval_df.with_columns([pl.lit(val).alias(c) for c, val in medians.items()])


def evaluate_counterfactual_accuracy(
    *,
    model: DemandForecaster,
    train_df: pl.DataFrame,
    eval_df: pl.DataFrame,
    no_flood_df: pl.DataFrame,
    entity_col: str,
    focus_entities: Sequence[str],
    shock_weeks: tuple[int, int] = (18, 28),
    placebo_weeks: tuple[int, int] = (4, 15),
    hydro_feature_prefixes: Sequence[str] = (),
    monotone_check_cols: Sequence[str] = (),
) -> CounterfactualAccuracyReport:
    """Audita acurácia do modelo na janela de choque + testes placebo + violação monotônica."""
    preds_real = model.predict_quantiles(eval_df)
    clim_df = _climatology_feature_frame(
        train_df, eval_df, feature_prefixes=hydro_feature_prefixes
    )
    preds_clim = model.predict_quantiles(clim_df)

    merged = (
        eval_df.select([entity_col, "target_ano_epi", "target_semana_epi", "target_y", "y_origin"])
        .with_columns(
            preds_real["pred_p10"],
            preds_real["pred_p50"],
            preds_real["pred_p90"],
            preds_real["pred_p95"],
            preds_clim["pred_p50"].alias("pred_p50_clim"),
        )
        .with_columns((pl.col("pred_p50") - pl.col("pred_p50_clim")).alias("excess_pred"))
        .join(
            no_flood_df.select(
                [
                    entity_col,
                    "target_ano_epi",
                    "target_semana_epi",
                    "y_cf_no_flood",
                    "y_cf_p05",
                    "y_cf_p95",
                    "excess_obs",
                ]
            ),
            on=[entity_col, "target_ano_epi", "target_semana_epi"],
            how="inner",
        )
    )

    focus = merged.filter(pl.col(entity_col).is_in(list(focus_entities)))
    shock = focus.filter(pl.col("target_semana_epi").is_between(shock_weeks[0], shock_weeks[1]))
    placebo = focus.filter(
        pl.col("target_semana_epi").is_between(placebo_weeks[0], placebo_weeks[1])
    )

    if shock.is_empty():
        raise DataContractError("Janela de choque vazia na auditoria contrafactual.")

    s_obs = shock["excess_obs"].to_numpy().astype(np.float64)
    s_pred = shock["excess_pred"].to_numpy().astype(np.float64)
    shock_mae = float(np.mean(np.abs(s_obs - s_pred)))
    shock_bias = float(np.mean(s_pred - s_obs))
    cov90 = float(
        np.mean(
            (shock["target_y"].to_numpy() >= shock["pred_p10"].to_numpy())
            & (shock["target_y"].to_numpy() <= shock["pred_p90"].to_numpy())
        )
    )

    # Erro de timing de pico (semanas) por região atingida
    timing_errors: list[float] = []
    for ent in focus_entities:
        sub = shock.filter(pl.col(entity_col) == ent)
        if sub.height >= 2:
            wk_obs = int(sub.sort("excess_obs", descending=True)["target_semana_epi"][0])
            wk_pred = int(sub.sort("excess_pred", descending=True)["target_semana_epi"][0])
            timing_errors.append(abs(float(wk_pred - wk_obs)))
    peak_timing_err = float(np.mean(timing_errors)) if timing_errors else 0.0

    if not placebo.is_empty():
        p_obs = placebo["excess_obs"].to_numpy().astype(np.float64)
        placebo_bias = float(np.mean(p_obs))
        placebo_mae = float(np.mean(np.abs(p_obs)))
    else:
        placebo_bias = 0.0
        placebo_mae = 0.0

    # Auditoria de violação monotônica sob perturbação positiva de risco físico
    valid_mono = [c for c in monotone_check_cols if c in eval_df.columns]
    if valid_mono:
        bumped = eval_df.with_columns(
            [
                (
                    pl.col(c) + max(float(train_df.select(pl.col(c).std()).item() or 1.0), 1.0)
                ).alias(c)
                for c in valid_mono
            ]
        )
        preds_bumped = model.predict_quantiles(bumped)
        viol = preds_bumped["pred_p50"].to_numpy() < (preds_real["pred_p50"].to_numpy() - 1e-6)
        mono_viol_rate = float(np.mean(viol))
    else:
        mono_viol_rate = 0.0

    return CounterfactualAccuracyReport(
        shock_excess_mae=shock_mae,
        shock_excess_bias=shock_bias,
        peak_timing_error_weeks=peak_timing_err,
        interval_coverage_90=cov90,
        placebo_bias=placebo_bias,
        placebo_mae=placebo_mae,
        monotonic_violation_rate=mono_viol_rate,
        series_df=focus.sort([entity_col, "target_ano_epi", "target_semana_epi"]),
        placebo_df=placebo.sort([entity_col, "target_ano_epi", "target_semana_epi"]),
    )


def decompose_intermunicipal_and_landfill_effect(
    *,
    model: DemandForecaster,
    train_df: pl.DataFrame,
    eval_df: pl.DataFrame,
    feature_groups: Mapping[str, Sequence[str]],
) -> pl.DataFrame:
    """Decompõe sequencialmente o excesso previsto via intervenções do() por bloco físico."""
    if eval_df.is_empty():
        raise DataContractError("eval_df vazio na decomposição contrafactual.")

    all_prefixes = [p for group in feature_groups.values() for p in group]
    base_df = _climatology_feature_frame(train_df, eval_df, feature_prefixes=all_prefixes)
    prev_mean = float(
        model.predict_quantiles(base_df).select(pl.col("pred_p50").mean()).item()
    )

    rows: list[dict[str, object]] = [
        {"etapa": "climatologia_p50", "nivel_medio": prev_mean, "contribuicao_delta": 0.0}
    ]
    current_df = base_df.clone()
    for group_name, prefixes in feature_groups.items():
        cols_to_activate = [
            c for c in eval_df.columns if any(c.startswith(prefix) for prefix in prefixes)
        ]
        if cols_to_activate:
            current_df = current_df.with_columns([eval_df[c] for c in cols_to_activate])
        cur_mean = float(
            model.predict_quantiles(current_df).select(pl.col("pred_p50").mean()).item()
        )
        rows.append(
            {
                "etapa": group_name,
                "nivel_medio": cur_mean,
                "contribuicao_delta": cur_mean - prev_mean,
            }
        )
        prev_mean = cur_mean

    return pl.DataFrame(rows)
