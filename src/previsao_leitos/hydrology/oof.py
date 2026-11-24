"""Auditoria autograd de monotonicidade e gerador causal out-of-fold (F03.1)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from previsao_leitos.domain import DataContractError
from previsao_leitos.hydrology.calibration import TemperatureScaler
from previsao_leitos.hydrology.model import HydroFiLMMonotonicNet


def audit_autograd_monotonicity(
    net: HydroFiLMMonotonicNet,
    *,
    x_dyn: torch.Tensor,
    x_ctx: torch.Tensor,
    x_mono_pos: torch.Tensor,
    x_mono_neg: torch.Tensor,
    x_enso: torch.Tensor | None = None,
) -> float:
    """Audita via torch.autograd.grad a taxa de violação monotônica em ambas as cabeças."""
    pos_in = x_mono_pos.detach().clone().requires_grad_(True)
    neg_in = x_mono_neg.detach().clone().requires_grad_(True)

    logits, sev = net(x_dyn, x_ctx, pos_in, neg_in, x_enso=x_enso)
    g_pos_l, g_neg_l = torch.autograd.grad(
        logits.sum(), (pos_in, neg_in), retain_graph=True
    )
    g_pos_s, g_neg_s = torch.autograd.grad(sev.sum(), (pos_in, neg_in))

    violations = (
        int((g_pos_l < -1e-7).sum().item())
        + int((g_pos_s < -1e-7).sum().item())
        + int((g_neg_l > 1e-7).sum().item())
        + int((g_neg_s > 1e-7).sum().item())
    )
    total = int(g_pos_l.numel() + g_pos_s.numel() + g_neg_l.numel() + g_neg_s.numel())
    return float(violations / max(total, 1))


def generate_causal_hydro_oof_predictions(
    panel_df: pl.DataFrame,
    *,
    feature_cols: Sequence[str],
    event_col: str,
    severity_col: str,
    train_end_year: int = 2022,
    cal_year: int = 2023,
) -> pl.DataFrame:
    """Gera previsões hidrológicas out-of-fold estritamente causais calibradas para F04.1.

    - Para cada ano `y <= train_end_year` (a partir do 2o ano disponível), treina apenas em
      `< y` e prevê out-of-fold em `y` (evitando vazamento e overfitting no acoplamento F03->F04).
    - Para `y > train_end_year`, usa modelo congelado em `<= train_end_year`, com temperatura
      calibrada exclusivamente em `cal_year`.
    """
    required = ["ano_epi", event_col, severity_col, *feature_cols]
    missing = [c for c in required if c not in panel_df.columns]
    if missing:
        raise DataContractError(f"Colunas ausentes para OOF hidrológico F03.1: {missing}")

    years = sorted(panel_df["ano_epi"].unique().to_list())
    train_years = [int(y) for y in years if int(y) <= train_end_year]
    if len(train_years) < 2:
        raise DataContractError("F03.1 OOF requer ao menos 2 anos em <= train_end_year.")

    x_all = (
        panel_df.select(list(feature_cols))
        .fill_null(0.0)
        .to_numpy()
        .astype(np.float64, copy=False)
    )
    y_ev_all = panel_df[event_col].fill_null(0.0).to_numpy().astype(np.float64, copy=False)
    y_sev_all = panel_df[severity_col].fill_null(0.0).to_numpy().astype(np.float64, copy=False)
    yr_all = panel_df["ano_epi"].to_numpy()

    oof_logits = np.zeros(panel_df.height, dtype=np.float64)
    oof_sev = np.zeros(panel_df.height, dtype=np.float64)

    def _fit_and_score(
        fit_mask: np.ndarray, pred_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        x_tr = x_all[fit_mask]
        y_ev_tr = (y_ev_all[fit_mask] > 0.5).astype(int)
        y_sev_tr = y_sev_all[fit_mask]
        x_te = x_all[pred_mask]

        if len(np.unique(y_ev_tr)) < 2:
            p_const = float(np.clip(np.mean(y_ev_tr), 1e-4, 1.0 - 1e-4))
            logits_te = np.full(x_te.shape[0], np.log(p_const / (1.0 - p_const)), dtype=np.float64)
        else:
            clf = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(C=1.0, max_iter=500)),
                ]
            )
            clf.fit(x_tr, y_ev_tr)
            logits_te = np.asarray(clf.decision_function(x_te), dtype=np.float64)

        reg = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=5.0))])
        reg.fit(x_tr, y_sev_tr)
        sev_te = np.maximum(0.0, np.asarray(reg.predict(x_te), dtype=np.float64))
        return logits_te, sev_te

    # Primeiro ano de treino recebe prioridade empírica causal simples (sem usar futuro)
    first_mask = yr_all == train_years[0]
    oof_logits[first_mask] = -2.0
    oof_sev[first_mask] = 0.0

    for yr in train_years[1:]:
        fit_m = yr_all < yr
        pred_m = yr_all == yr
        oof_logits[pred_m], oof_sev[pred_m] = _fit_and_score(fit_m, pred_m)

    # Anos pós-treino (> train_end_year): modelo congelado em <= train_end_year
    post_mask = yr_all > train_end_year
    if np.any(post_mask):
        fit_m = yr_all <= train_end_year
        oof_logits[post_mask], oof_sev[post_mask] = _fit_and_score(fit_m, post_mask)

    # Calibração de temperatura estritamente em cal_year (2023)
    cal_mask = yr_all == cal_year
    if np.sum(cal_mask) >= 10 and len(np.unique((y_ev_all[cal_mask] > 0.5).astype(int))) >= 2:
        scaler = TemperatureScaler().fit(oof_logits[cal_mask], y_ev_all[cal_mask])
        probs = scaler.predict_proba(oof_logits)
    else:
        probs = 1.0 / (1.0 + np.exp(-np.clip(oof_logits, -30.0, 30.0)))

    return panel_df.with_columns(
        pl.Series("prob_enchente_f03_oof", probs, dtype=pl.Float64),
        pl.Series("severidade_hidro_f03_oof", oof_sev, dtype=pl.Float64),
    )
