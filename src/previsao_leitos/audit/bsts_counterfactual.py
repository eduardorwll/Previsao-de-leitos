"""Motor Contrafactual BSTS Pré-Choque (com SIVEP-Gripe) e do() Dinâmico Recursivo (F06.2).

Substitui o controle sintético intra-RS contaminado por um modelo estrutural
Bayesiano ajustado estritamente antes do choque (com controle explícito de onda
viral SIVEP-Gripe) e executa intervenções dinâmicas recursivas `do(Clima=Climatologia)`
e `do(ENSO=Neutro)` propagando `y_origin_cf` sem contaminação pós-tratamento.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl
from statsmodels.tsa.statespace.structural import UnobservedComponents

from previsao_leitos.demand.models import DemandForecaster
from previsao_leitos.domain import DataContractError

# Quantil exato Normal padrão para intervalo preditivo bilateral de 90% (P05 / P95)
_Z_90: float = 1.6448536269514722
_EPOCH_MONDAY: date = date(1970, 1, 5)


def _iso_week_ordinal(year: int, week: int) -> int:
    """Converte `(ano_epi, semana_epi)` em índice inteiro semanal contínuo sem colisões anuais."""
    monday = date.fromisocalendar(int(year), 1, 1) + timedelta(weeks=int(week) - 1)
    return (monday - _EPOCH_MONDAY).days // 7


@dataclass
class _EntityStateSpaceModel:
    """Estado ajustado por entidade (UnobservedComponents local-level ou fallback Ridge)."""

    mode: str  # "uc" | "constant" | "ridge"
    pre_index: dict[tuple[int, int], int]
    pre_ordinals: tuple[int, ...] = ()
    last_pre_ord: int = 0
    uc_result: Any = None
    const_mean: float = 0.0
    beta_hat: np.ndarray | None = None
    cov_post: np.ndarray | None = None
    sigma2: float = 1e-6


class PreShockBstsCounterfactual:
    """Controle Estrutural Bayesiano Pré-Choque (nível local estocástico + harmônicos + SIVEP).

    Ajusta por entidade um modelo estrutural de espaço de estados (`UnobservedComponents`
    com nível local estocástico + 2 pares harmônicos sazonais + covariáveis epidemiológicas
    exógenas como SRAG viral) estritamente antes do choque `(ano < shock_year)` ou
    `(ano == shock_year & semana < shock_start_week)`, eliminando o viés da tendência
    linear global sobre séries com expansão recente de leitos (Bug G).
    """

    def __init__(
        self,
        *,
        shock_year: int = 2024,
        shock_start_week: int = 18,
        control_cols: Sequence[str] = ("srag_viral_confirmado",),
        prior_precision: float = 1.0,
        max_iter: int = 100,
    ) -> None:
        self._shock_year = shock_year
        self._shock_start_week = shock_start_week
        self._control_cols_cfg = tuple(control_cols)
        self._prior_precision = prior_precision
        self._max_iter = max_iter
        self._active_controls: tuple[str, ...] = ()
        self._models: dict[str, _EntityStateSpaceModel] = {}
        self._t_origin: float = 2020.0

    def _exog_matrix(self, df: pl.DataFrame, *, week_col: str) -> np.ndarray:
        wk = df[week_col].cast(pl.Float64).to_numpy()
        cols = [
            np.sin(2.0 * np.pi * wk / 52.0),
            np.cos(2.0 * np.pi * wk / 52.0),
            np.sin(4.0 * np.pi * wk / 52.0),
            np.cos(4.0 * np.pi * wk / 52.0),
        ]
        for ctrl in self._active_controls:
            if ctrl in df.columns:
                arr = df[ctrl].cast(pl.Float64, strict=False).fill_null(0.0).to_numpy()
                cols.append(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0))
            else:
                cols.append(np.zeros(df.height, dtype=np.float64))
        return np.asarray(
            np.nan_to_num(np.column_stack(cols), nan=0.0, posinf=0.0, neginf=0.0),
            dtype=np.float64,
        )

    def _ridge_design_matrix(
        self,
        df: pl.DataFrame,
        *,
        year_col: str,
        week_col: str,
    ) -> np.ndarray:
        yr = df[year_col].cast(pl.Float64).to_numpy()
        wk = df[week_col].cast(pl.Float64).to_numpy()
        t_scaled = (yr - self._t_origin) + (wk - 1.0) / 52.0
        exog = self._exog_matrix(df, week_col=week_col)
        return np.column_stack([np.ones(df.height, dtype=np.float64), t_scaled, exog])

    def _fit_ridge_fallback(
        self,
        sub: pl.DataFrame,
        *,
        year_col: str,
        week_col: str,
        target_col: str,
        pre_index: dict[tuple[int, int], int],
    ) -> _EntityStateSpaceModel:
        x_mat = self._ridge_design_matrix(sub, year_col=year_col, week_col=week_col)
        y_vec = sub[target_col].cast(pl.Float64).to_numpy()
        p = x_mat.shape[1]
        prior_prec = np.eye(p, dtype=np.float64) * self._prior_precision
        prior_prec[0, 0] = 1e-4

        gram = x_mat.T @ x_mat + prior_prec
        beta_hat = np.linalg.solve(gram, x_mat.T @ y_vec)
        resid = y_vec - x_mat @ beta_hat
        dof = max(1, sub.height - p)
        sigma2 = max(float(np.sum(resid**2) / dof), 1e-6)
        cov_post = sigma2 * np.linalg.inv(gram)
        return _EntityStateSpaceModel(
            mode="ridge",
            pre_index=pre_index,
            beta_hat=beta_hat,
            cov_post=cov_post,
            sigma2=sigma2,
        )

    def fit(
        self,
        panel_df: pl.DataFrame,
        *,
        entity_col: str = "nome_regiao_saude",
        target_col: str = "target_y",
        year_col: str = "ano_epi",
        week_col: str = "semana_epi",
    ) -> PreShockBstsCounterfactual:
        """Ajusta o modelo de espaço de estados pré-choque para cada entidade."""
        pre_mask = (pl.col(year_col) < self._shock_year) | (
            (pl.col(year_col) == self._shock_year) & (pl.col(week_col) < self._shock_start_week)
        )
        pre_df = panel_df.filter(pre_mask)
        if pre_df.is_empty():
            raise DataContractError("Sem observações pré-choque para ajustar BSTS.")

        self._t_origin = float(pre_df[year_col].cast(pl.Float64).to_numpy().min())
        self._active_controls = tuple(
            c for c in self._control_cols_cfg if c in pre_df.columns
        )
        self._models = {}

        for ent in sorted(pre_df[entity_col].unique().to_list()):
            sub = (
                pre_df.filter(pl.col(entity_col) == ent)
                .drop_nulls(subset=[target_col])
                .sort([year_col, week_col])
            )
            if sub.height < 12:
                continue

            yrs = sub[year_col].to_list()
            wks = sub[week_col].to_list()
            pre_index = {
                (int(yr), int(wk)): idx
                for idx, (yr, wk) in enumerate(zip(yrs, wks, strict=True))
            }
            pre_ordinals = tuple(
                _iso_week_ordinal(int(yr), int(wk))
                for yr, wk in zip(yrs, wks, strict=True)
            )
            last_pre_ord = pre_ordinals[-1]
            y_vec = sub[target_col].cast(pl.Float64).to_numpy()

            if float(np.std(y_vec)) < 1e-6:
                self._models[str(ent)] = _EntityStateSpaceModel(
                    mode="constant",
                    pre_index=pre_index,
                    pre_ordinals=pre_ordinals,
                    last_pre_ord=last_pre_ord,
                    const_mean=max(0.0, float(np.mean(y_vec))),
                )
                continue

            if sub.height >= 24:
                exog_pre = self._exog_matrix(sub, week_col=week_col)
                try:
                    uc_mod = UnobservedComponents(
                        endog=y_vec,
                        exog=exog_pre,
                        level="local level",
                    )
                    uc_res = uc_mod.fit(disp=False, maxiter=self._max_iter)
                    if np.all(np.isfinite(np.asarray(uc_res.params, dtype=np.float64))):
                        self._models[str(ent)] = _EntityStateSpaceModel(
                            mode="uc",
                            pre_index=pre_index,
                            pre_ordinals=pre_ordinals,
                            last_pre_ord=last_pre_ord,
                            uc_result=uc_res,
                        )
                        continue
                except (ValueError, np.linalg.LinAlgError, RuntimeError):
                    self._models[str(ent)] = self._fit_ridge_fallback(
                        sub,
                        year_col=year_col,
                        week_col=week_col,
                        target_col=target_col,
                        pre_index=pre_index,
                    )
                    continue

            self._models[str(ent)] = self._fit_ridge_fallback(
                sub,
                year_col=year_col,
                week_col=week_col,
                target_col=target_col,
                pre_index=pre_index,
            )

        if not self._models:
            raise DataContractError("Nenhuma entidade com histórico suficiente para BSTS.")
        return self

    def predict_counterfactual(
        self,
        eval_df: pl.DataFrame,
        *,
        entity_col: str = "nome_regiao_saude",
        year_col: str = "target_ano_epi",
        week_col: str = "target_semana_epi",
        target_col: str = "target_y",
    ) -> pl.DataFrame:
        """Prevê trajetória contrafactual BSTS (sem enchente) com IC 90% (z = 1.6449)."""
        if not self._models:
            raise DataContractError("PreShockBstsCounterfactual precisa ser ajustado antes.")

        frames: list[pl.DataFrame] = []
        for ent, model_state in self._models.items():
            sub = eval_df.filter(pl.col(entity_col) == ent).sort([year_col, week_col])
            if sub.is_empty():
                continue

            n_rows = sub.height
            if model_state.mode == "constant":
                y_cf = np.full(n_rows, model_state.const_mean, dtype=np.float64)
                pred_sd = np.zeros(n_rows, dtype=np.float64)
            elif model_state.mode == "uc" and model_state.uc_result is not None:
                y_cf = np.empty(n_rows, dtype=np.float64)
                pred_sd = np.empty(n_rows, dtype=np.float64)
                exog_ev = self._exog_matrix(sub, week_col=week_col)
                yrs = sub[year_col].to_list()
                wks = sub[week_col].to_list()

                in_sample_pos: list[tuple[int, int]] = []
                missing_pre_pos: list[tuple[int, int]] = []
                future_steps: list[tuple[int, int]] = []
                for row_i, (yr, wk) in enumerate(zip(yrs, wks, strict=True)):
                    key = (int(yr), int(wk))
                    if key in model_state.pre_index:
                        in_sample_pos.append((row_i, model_state.pre_index[key]))
                    else:
                        ord_k = _iso_week_ordinal(int(yr), int(wk))
                        if ord_k > model_state.last_pre_ord:
                            future_steps.append((row_i, ord_k - model_state.last_pre_ord))
                        else:
                            missing_pre_pos.append((row_i, ord_k))

                if in_sample_pos:
                    min_idx = min(p for _, p in in_sample_pos)
                    max_idx = max(p for _, p in in_sample_pos)
                    pred_in = model_state.uc_result.get_prediction(start=min_idx, end=max_idx)
                    in_mean = np.asarray(pred_in.predicted_mean, dtype=np.float64)
                    in_var = np.asarray(pred_in.var_pred_mean, dtype=np.float64)
                    for row_i, orig_idx in in_sample_pos:
                        rel = orig_idx - min_idx
                        y_cf[row_i] = max(0.0, float(in_mean[rel]))
                        pred_sd[row_i] = float(np.sqrt(max(float(in_var[rel]), 1e-9)))

                if missing_pre_pos:
                    params = np.asarray(model_state.uc_result.params, dtype=np.float64)
                    sigma2_irr = max(float(params[0]), 0.0)
                    sigma2_lvl = max(float(params[1]), 0.0)
                    beta_exog = params[2:]
                    filt_state = np.asarray(
                        model_state.uc_result.filtered_state, dtype=np.float64
                    )
                    filt_cov = np.asarray(
                        model_state.uc_result.filtered_state_cov, dtype=np.float64
                    )
                    for row_i, ord_k in missing_pre_pos:
                        ins = bisect_right(model_state.pre_ordinals, ord_k)
                        prev_idx = max(0, ins - 1)
                        gap = max(1, abs(ord_k - model_state.pre_ordinals[prev_idx]))
                        mu_hat = float(filt_state[0, prev_idx]) + float(
                            exog_ev[row_i] @ beta_exog
                        )
                        var_hat = (
                            float(filt_cov[0, 0, prev_idx]) + gap * sigma2_lvl + sigma2_irr
                        )
                        y_cf[row_i] = max(0.0, mu_hat)
                        pred_sd[row_i] = float(np.sqrt(max(var_hat, 1e-9)))

                if future_steps:
                    max_step = max(step for _, step in future_steps)
                    exog_grid = np.zeros((max_step, exog_ev.shape[1]), dtype=np.float64)
                    for row_i, step in future_steps:
                        exog_grid[step - 1] = exog_ev[row_i]
                    fc = model_state.uc_result.get_forecast(
                        steps=max_step,
                        exog=exog_grid,
                    )
                    fc_mean = np.asarray(fc.predicted_mean, dtype=np.float64)
                    fc_var = np.asarray(fc.var_pred_mean, dtype=np.float64)
                    for row_i, step in future_steps:
                        y_cf[row_i] = max(0.0, float(fc_mean[step - 1]))
                        pred_sd[row_i] = float(np.sqrt(max(float(fc_var[step - 1]), 1e-9)))
            else:
                x_ev = self._ridge_design_matrix(sub, year_col=year_col, week_col=week_col)
                beta = model_state.beta_hat
                cov = model_state.cov_post
                if beta is None or cov is None:
                    raise DataContractError(f"Estado Ridge inválido para entidade {ent}.")
                y_cf = np.maximum(0.0, x_ev @ beta)
                pred_var = model_state.sigma2 + np.einsum("ij,jk,ik->i", x_ev, cov, x_ev)
                pred_sd = np.sqrt(np.maximum(pred_var, 1e-9))

            base_cols = [entity_col, year_col, week_col]
            if target_col in sub.columns:
                base_cols.append(target_col)
            out = sub.select(base_cols).with_columns(
                pl.Series("y_cf_no_flood", y_cf, dtype=pl.Float64),
                pl.Series("y_cf_p05", np.maximum(0.0, y_cf - _Z_90 * pred_sd), dtype=pl.Float64),
                pl.Series("y_cf_p95", np.maximum(0.0, y_cf + _Z_90 * pred_sd), dtype=pl.Float64),
            )
            if target_col in sub.columns:
                out = out.with_columns(
                    (pl.col(target_col).cast(pl.Float64) - pl.col("y_cf_no_flood")).alias(
                        "excess_obs"
                    )
                )
            frames.append(out)

        if not frames:
            raise DataContractError("Nenhuma entidade avaliável em PreShockBstsCounterfactual.")
        return pl.concat(frames).sort([entity_col, year_col, week_col])


def recursive_dynamic_do_intervention(
    *,
    model: DemandForecaster,
    train_df: pl.DataFrame,
    eval_df: pl.DataFrame,
    entity_col: str = "nome_regiao_saude",
    horizon_weeks: int = 2,
    intervention_start_week: int = 18,
    intervention_start_year: int | None = None,
    climate_prefixes: Sequence[str] = (),
    neutral_zero_prefixes: Sequence[str] = (),
) -> pl.DataFrame:
    """Executa intervenção dinâmica recursiva `do(Clima=Climatologia, ENSO=Neutro)`.

    Substitui covariáveis climáticas pela mediana de treino, zera índices de
    teleconexão (`neutral_zero_prefixes`, ex.: `oni_c`, `el_nino_intensity`) e
    propaga recursivamente `y_origin`, `y_roll4`, `y_roll12` e `y_dev_roll12`
    a partir das previsões contrafactuais anteriores (`t - horizon_weeks`),
    impedindo que observações inundadas contaminem o contrafactual pós-choque.
    """
    if eval_df.is_empty():
        raise DataContractError("eval_df vazio em recursive_dynamic_do_intervention.")

    clim_cols = [
        c for c in eval_df.columns if any(c.startswith(p) for p in climate_prefixes)
    ]
    zero_cols = [
        c for c in eval_df.columns if any(c.startswith(p) for p in neutral_zero_prefixes)
    ]
    clim_medians = {
        c: float(train_df.select(pl.col(c).cast(pl.Float64).median()).item() or 0.0)
        for c in clim_cols
        if c in train_df.columns
    }

    has_y_origin = "y_origin" in eval_df.columns
    has_y_roll4 = "y_roll4" in eval_df.columns
    has_y_roll12 = "y_roll12" in eval_df.columns
    has_y_dev_roll12 = "y_dev_roll12" in eval_df.columns

    start_ord = (
        _iso_week_ordinal(intervention_start_year, intervention_start_week)
        if intervention_start_year is not None
        else None
    )

    out_frames: list[pl.DataFrame] = []
    for ent in sorted(eval_df[entity_col].unique().to_list()):
        sub = eval_df.filter(pl.col(entity_col) == ent).sort(
            ["target_ano_epi", "target_semana_epi"]
        )
        cf_history: dict[int, float] = {}
        intervened_ords: set[int] = set()
        origin_window: list[float] = []
        p50_cf_list: list[float] = []
        p90_cf_list: list[float] = []

        for idx in range(sub.height):
            row_df = sub.slice(idx, 1)
            yr = int(row_df["target_ano_epi"][0])
            wk = int(row_df["target_semana_epi"][0])
            wk_ord = _iso_week_ordinal(yr, wk)
            mods: list[pl.Expr] = []

            y_orig_eff: float | None = (
                float(row_df["y_origin"][0])
                if has_y_origin and row_df["y_origin"][0] is not None
                else None
            )

            in_intervention = (
                wk_ord >= start_ord if start_ord is not None else wk >= intervention_start_week
            )
            replaced_origin = False
            if in_intervention:
                intervened_ords.add(wk_ord)
                for col, med in clim_medians.items():
                    mods.append(pl.lit(med, dtype=pl.Float64).alias(col))
                for col in zero_cols:
                    mods.append(pl.lit(0.0, dtype=pl.Float64).alias(col))
                origin_ord = wk_ord - horizon_weeks
                # Só substitui y_origin quando a semana de origem já foi afetada pela intervenção
                if origin_ord in intervened_ords and origin_ord in cf_history and has_y_origin:
                    y_orig_eff = cf_history[origin_ord]
                    mods.append(pl.lit(y_orig_eff, dtype=pl.Float64).alias("y_origin"))
                    replaced_origin = True

            if y_orig_eff is not None:
                origin_window.append(y_orig_eff)
                if replaced_origin:
                    roll4_val = float(np.mean(origin_window[-4:]))
                    roll12_val = float(np.mean(origin_window[-12:]))
                    if has_y_roll4:
                        mods.append(pl.lit(roll4_val, dtype=pl.Float64).alias("y_roll4"))
                    if has_y_roll12:
                        mods.append(pl.lit(roll12_val, dtype=pl.Float64).alias("y_roll12"))
                    if has_y_dev_roll12:
                        mods.append(
                            pl.lit(y_orig_eff - roll12_val, dtype=pl.Float64).alias("y_dev_roll12")
                        )

            if mods:
                row_df = row_df.with_columns(mods)

            pred = model.predict_quantiles(row_df)
            p50_val = float(pred["pred_p50"][0])
            p90_val = float(pred["pred_p90"][0])
            cf_history[wk_ord] = p50_val
            p50_cf_list.append(p50_val)
            p90_cf_list.append(p90_val)

        out_frames.append(
            sub.with_columns(
                pl.Series("pred_p50_cf", p50_cf_list, dtype=pl.Float64),
                pl.Series("pred_p90_cf", p90_cf_list, dtype=pl.Float64),
            )
        )

    return pl.concat(out_frames).sort([entity_col, "target_ano_epi", "target_semana_epi"])
