"""Motor Contrafactual BSTS Pré-Choque (com SIVEP-Gripe) e do() Dinâmico Recursivo (F06.2).

Substitui o controle sintético intra-RS contaminado por um modelo estrutural
Bayesiano ajustado estritamente antes do choque (com controle explícito de onda
viral SIVEP-Gripe) e executa intervenções dinâmicas recursivas `do(Clima=Climatologia)`
e `do(ENSO=Neutro)` propagando `y_origin_cf` sem contaminação pós-tratamento.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from previsao_leitos.demand.models import DemandForecaster
from previsao_leitos.domain import DataContractError


class PreShockBstsCounterfactual:
    """Controle Estrutural Bayesiano Pré-Choque (tendência + harmônicos + SIVEP-Gripe).

    Não depende de regiões doadoras intra-RS contaminadas nem de outras UFs na V1.1:
    cada entidade é ajustada estritamente em `(ano < shock_year)` ou
    `(ano == shock_year & semana < shock_start_week)` usando regressão Bayesiana
    conjugada Normal-Normal (tendência temporal + 2 pares harmônicos sazonais +
    covariáveis epidemiológicas exógenas como SRAG viral).
    """

    def __init__(
        self,
        *,
        shock_year: int = 2024,
        shock_start_week: int = 18,
        control_cols: Sequence[str] = ("srag_viral_confirmado",),
        prior_precision: float = 1.0,
    ) -> None:
        self._shock_year = shock_year
        self._shock_start_week = shock_start_week
        self._control_cols_cfg = tuple(control_cols)
        self._prior_precision = prior_precision
        self._active_controls: tuple[str, ...] = ()
        self._posterior_mean: dict[str, np.ndarray] = {}
        self._posterior_cov: dict[str, np.ndarray] = {}
        self._sigma2: dict[str, float] = {}
        self._t_origin: float = 2020.0

    def _design_matrix(
        self,
        df: pl.DataFrame,
        *,
        year_col: str,
        week_col: str,
    ) -> np.ndarray:
        yr = df[year_col].cast(pl.Float64).to_numpy()
        wk = df[week_col].cast(pl.Float64).to_numpy()
        t_scaled = (yr - self._t_origin) + (wk - 1.0) / 52.0
        cols = [
            np.ones(df.height, dtype=np.float64),
            t_scaled,
            np.sin(2.0 * np.pi * wk / 52.0),
            np.cos(2.0 * np.pi * wk / 52.0),
            np.sin(4.0 * np.pi * wk / 52.0),
            np.cos(4.0 * np.pi * wk / 52.0),
        ]
        for ctrl in self._active_controls:
            if ctrl in df.columns:
                cols.append(df[ctrl].cast(pl.Float64).fill_null(0.0).to_numpy())
            else:
                cols.append(np.zeros(df.height, dtype=np.float64))
        return np.column_stack(cols)

    def fit(
        self,
        panel_df: pl.DataFrame,
        *,
        entity_col: str = "nome_regiao_saude",
        target_col: str = "target_y",
        year_col: str = "ano_epi",
        week_col: str = "semana_epi",
    ) -> PreShockBstsCounterfactual:
        """Ajusta a distribuição posterior pré-choque para cada entidade."""
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
        self._posterior_mean = {}
        self._posterior_cov = {}
        self._sigma2 = {}

        for ent in sorted(pre_df[entity_col].unique().to_list()):
            sub = pre_df.filter(pl.col(entity_col) == ent).drop_nulls(subset=[target_col])
            if sub.height < 12:
                continue
            x_mat = self._design_matrix(sub, year_col=year_col, week_col=week_col)
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

            self._posterior_mean[str(ent)] = beta_hat
            self._posterior_cov[str(ent)] = cov_post
            self._sigma2[str(ent)] = sigma2

        if not self._posterior_mean:
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
        """Prevê trajetória contrafactual BSTS (sem enchente) com IC 90% preditivo posterior."""
        if not self._posterior_mean:
            raise DataContractError("PreShockBstsCounterfactual precisa ser ajustado antes.")

        frames: list[pl.DataFrame] = []
        for ent, beta in self._posterior_mean.items():
            sub = eval_df.filter(pl.col(entity_col) == ent)
            if sub.is_empty():
                continue
            x_ev = self._design_matrix(sub, year_col=year_col, week_col=week_col)
            y_cf = np.maximum(0.0, x_ev @ beta)
            cov = self._posterior_cov[ent]
            sig2 = self._sigma2[ent]
            pred_var = sig2 + np.einsum("ij,jk,ik->i", x_ev, cov, x_ev)
            pred_sd = np.sqrt(np.maximum(pred_var, 1e-9))

            base_cols = [entity_col, year_col, week_col]
            if target_col in sub.columns:
                base_cols.append(target_col)
            out = sub.select(base_cols).with_columns(
                pl.Series("y_cf_no_flood", y_cf, dtype=pl.Float64),
                pl.Series("y_cf_p05", np.maximum(0.0, y_cf - 1.96 * pred_sd), dtype=pl.Float64),
                pl.Series("y_cf_p95", np.maximum(0.0, y_cf + 1.96 * pred_sd), dtype=pl.Float64),
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
    climate_prefixes: Sequence[str] = (),
    neutral_zero_prefixes: Sequence[str] = (),
) -> pl.DataFrame:
    """Executa intervenção dinâmica recursiva `do(Clima=Climatologia, ENSO=Neutro)`.

    Substitui covariáveis climáticas pela mediana de treino, zera índices de
    teleconexão (`neutral_zero_prefixes`, ex.: `oni_c`, `el_nino_intensity`) e
    propaga recursivamente `y_origin` a partir das previsões contrafactuais
    anteriores (`t - horizon_weeks`), impedindo que `y_origin` inundado contamine
    o contrafactual nas semanas subsequentes ao choque.
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

    out_frames: list[pl.DataFrame] = []
    for ent in sorted(eval_df[entity_col].unique().to_list()):
        sub = eval_df.filter(pl.col(entity_col) == ent).sort(
            ["target_ano_epi", "target_semana_epi"]
        )
        cf_history: dict[int, float] = {}
        p50_cf_list: list[float] = []
        p90_cf_list: list[float] = []

        for idx in range(sub.height):
            row_df = sub.slice(idx, 1)
            wk = int(row_df["target_semana_epi"][0])
            mods: list[pl.Expr] = []

            if wk >= intervention_start_week:
                for col, med in clim_medians.items():
                    mods.append(pl.lit(med, dtype=pl.Float64).alias(col))
                for col in zero_cols:
                    mods.append(pl.lit(0.0, dtype=pl.Float64).alias(col))
                origin_wk = wk - horizon_weeks
                if origin_wk in cf_history and "y_origin" in row_df.columns:
                    mods.append(pl.lit(cf_history[origin_wk], dtype=pl.Float64).alias("y_origin"))

            if mods:
                row_df = row_df.with_columns(mods)

            pred = model.predict_quantiles(row_df)
            p50_val = float(pred["pred_p50"][0])
            p90_val = float(pred["pred_p90"][0])
            cf_history[wk] = p50_val
            p50_cf_list.append(p50_val)
            p90_cf_list.append(p90_val)

        out_frames.append(
            sub.with_columns(
                pl.Series("pred_p50_cf", p50_cf_list, dtype=pl.Float64),
                pl.Series("pred_p90_cf", p90_cf_list, dtype=pl.Float64),
            )
        )

    return pl.concat(out_frames).sort([entity_col, "target_ano_epi", "target_semana_epi"])
