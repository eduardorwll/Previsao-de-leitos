"""Testes unitários da Task 3.1: HydroFiLM-ENSO e previsões out-of-fold causais."""

from __future__ import annotations

import numpy as np
import polars as pl
import torch

from previsao_leitos.hydrology.model import HydroFiLMMonotonicNet
from previsao_leitos.hydrology.oof import (
    audit_autograd_monotonicity,
    generate_causal_hydro_oof_predictions,
)


def test_hydro_film_enso_preserves_exact_autograd_monotonicity() -> None:
    torch.manual_seed(42)
    net = HydroFiLMMonotonicNet(
        dyn_dim=4,
        ctx_dim=3,
        enso_dim=3,
        mono_pos_dim=3,
        mono_neg_dim=1,
        hidden_dim=16,
        num_heads=2,
        dropout=0.0,
    )
    net.eval()

    x_dyn = torch.randn(8, 6, 4)
    x_ctx = torch.randn(8, 3)
    x_enso = torch.randn(8, 3)
    x_mono_pos = torch.randn(8, 3, requires_grad=True)
    x_mono_neg = torch.randn(8, 1, requires_grad=True)

    viol_rate = audit_autograd_monotonicity(
        net,
        x_dyn=x_dyn,
        x_ctx=x_ctx,
        x_mono_pos=x_mono_pos,
        x_mono_neg=x_mono_neg,
        x_enso=x_enso,
    )
    assert viol_rate == 0.0

    # Alterar o vetor ENSO deve modular a saída via FiLM
    logits_a, sev_a = net(x_dyn, x_ctx, x_mono_pos.detach(), x_mono_neg.detach(), x_enso=x_enso)
    logits_b, sev_b = net(
        x_dyn, x_ctx, x_mono_pos.detach(), x_mono_neg.detach(), x_enso=x_enso + 2.5
    )
    assert not torch.allclose(logits_a, logits_b)
    assert not torch.allclose(sev_a, sev_b)


def test_generate_causal_hydro_oof_predictions_zero_future_leakage() -> None:
    rng = np.random.default_rng(7)
    rows = []
    for yr in (2020, 2021, 2022, 2023, 2024):
        for wk in range(1, 21):
            chuva = float(rng.gamma(2.0, 25.0))
            oni = 1.8 if yr in (2023, 2024) else -0.5
            evento = 1.0 if (chuva + 20.0 * max(0.0, oni)) > 95.0 else 0.0
            sev = max(0.0, (chuva + 20.0 * max(0.0, oni) - 60.0) / 50.0)
            rows.append(
                {
                    "nome_regiao_saude": "R10",
                    "ano_epi": yr,
                    "semana_epi": wk,
                    "chuva_montante_pond": chuva,
                    "api_max_sem": chuva * 0.6,
                    "oni_c_lag4": oni,
                    "tsa_anom_c_lag4": 0.4 if yr >= 2023 else 0.0,
                    "y_flood_event": evento,
                    "y_flood_severity": sev,
                }
            )
    df_a = pl.DataFrame(rows)
    # Perturba 2024 fortemente -> OOF em <= 2023 deve ser bit-idêntico
    df_b = df_a.with_columns(
        pl.when(pl.col("ano_epi") == 2024)
        .then(pl.lit(1.0))
        .otherwise(pl.col("y_flood_event"))
        .alias("y_flood_event")
    )

    oof_a = generate_causal_hydro_oof_predictions(
        df_a,
        feature_cols=("chuva_montante_pond", "api_max_sem", "oni_c_lag4", "tsa_anom_c_lag4"),
        event_col="y_flood_event",
        severity_col="y_flood_severity",
        train_end_year=2022,
        cal_year=2023,
    )
    oof_b = generate_causal_hydro_oof_predictions(
        df_b,
        feature_cols=("chuva_montante_pond", "api_max_sem", "oni_c_lag4", "tsa_anom_c_lag4"),
        event_col="y_flood_event",
        severity_col="y_flood_severity",
        train_end_year=2022,
        cal_year=2023,
    )

    assert "prob_enchente_f03_oof" in oof_a.columns
    assert "severidade_hidro_f03_oof" in oof_a.columns
    probs = oof_a["prob_enchente_f03_oof"].to_numpy()
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    pre_a = oof_a.filter(pl.col("ano_epi") <= 2023)
    pre_b = oof_b.filter(pl.col("ano_epi") <= 2023)
    assert pre_a.equals(pre_b)
