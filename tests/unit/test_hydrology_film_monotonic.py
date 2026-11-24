"""Testes Prove-It da Feature F03: monotonicidade exata via autograd, lacunas e calibração."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch

from previsao_leitos.hydrology import (
    HydroFiLMMonotonicNet,
    HydroLightningModule,
    IndexedHydroSequenceDataset,
    TemperatureScaler,
)


def test_hydro_film_monotonic_net_exact_autograd_derivatives_on_both_heads() -> None:
    torch.manual_seed(42)
    net = HydroFiLMMonotonicNet(
        dyn_dim=6,
        ctx_dim=2,
        mono_pos_dim=4,
        mono_neg_dim=1,
        hidden_dim=16,
        num_heads=2,
    )
    net.eval()

    x_dyn = torch.randn(12, 8, 6)
    x_ctx = torch.randn(12, 2)
    x_pos = torch.rand(12, 4, requires_grad=True)
    x_neg = torch.rand(12, 1, requires_grad=True)

    logits, severity = net(x_dyn, x_ctx, x_pos, x_neg)

    # Derivadas parciais exatas para event_logits
    g_pos_l, g_neg_l = torch.autograd.grad(logits.sum(), (x_pos, x_neg), retain_graph=True)
    assert bool((g_pos_l > 0.0).all().item())
    assert bool((g_neg_l < 0.0).all().item())

    # Derivadas parciais exatas para severity_head (corrige vazamento do legado ADR-0001!)
    g_pos_s, g_neg_s = torch.autograd.grad(severity.sum(), (x_pos, x_neg))
    assert bool((g_pos_s > 0.0).all().item())
    assert bool((g_neg_s < 0.0).all().item())

    # Verificação por diferenças finitas: aumentar aterro/impermeabilização eleva risco
    x_pos_up = x_pos.detach() + 0.5
    x_neg_up = x_neg.detach() + 0.5
    logits_up_pos, sev_up_pos = net(x_dyn, x_ctx, x_pos_up, x_neg.detach())
    logits_up_neg, sev_up_neg = net(x_dyn, x_ctx, x_pos.detach(), x_neg_up)
    assert bool((logits_up_pos > logits.detach()).all().item())
    assert bool((sev_up_pos > severity.detach()).all().item())
    assert bool((logits_up_neg < logits.detach()).all().item())
    assert bool((sev_up_neg < severity.detach()).all().item())


def test_indexed_hydro_sequence_dataset_never_bridges_entity_or_calendar_gaps() -> None:
    # Entidade A: semanas 1, 2, 3, 4 (contíguas) + salto -> semana 8, 9 (lacuna!)
    # Entidade B: semanas 1, 2, 3, 4 (contíguas)
    df = pl.DataFrame(
        {
            "entity": ["A"] * 6 + ["B"] * 4,
            "ano_epi": [2022] * 10,
            "semana_epi": [1, 2, 3, 4, 8, 9, 1, 2, 3, 4],
            "chuva": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 5.0, 6.0, 7.0, 8.0],
            "ctx": [1.0] * 10,
            "aterro": [0.2] * 6 + [0.01] * 4,
            "floresta": [0.1] * 6 + [0.5] * 4,
            "y_event": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "y_sev": [0.1, 0.2, 0.8, 0.9, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1],
        }
    )

    ds = IndexedHydroSequenceDataset(
        df,
        entity_col="entity",
        dyn_cols=("chuva",),
        ctx_cols=("ctx",),
        mono_pos_cols=("aterro",),
        mono_neg_cols=("floresta",),
        event_col="y_event",
        severity_col="y_sev",
        seq_len=3,
    )
    # Apenas janelas contíguas de tamanho 3: A(1-3), A(2-4), B(1-3), B(2-4) -> exatamente 4!
    # As semanas 8-9 de A NÃO podem cruzar o salto da semana 4!
    assert len(ds) == 4
    sample = ds[0]
    assert sample["x_dyn"].shape == (3, 1)


def test_lightning_module_step_and_temperature_scaler_calibration() -> None:
    module = HydroLightningModule(
        dyn_dim=4,
        ctx_dim=2,
        mono_pos_dim=3,
        mono_neg_dim=1,
        hidden_dim=16,
        num_heads=2,
        pos_weight=3.0,
    )
    batch = {
        "x_dyn": torch.randn(8, 4, 4),
        "x_ctx": torch.randn(8, 2),
        "x_mono_pos": torch.rand(8, 3),
        "x_mono_neg": torch.rand(8, 1),
        "y_event": torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]),
        "y_sev": torch.rand(8),
    }
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0

    # Temperature scaling reduz NLL sobre logits superconfiantes
    rng = np.random.default_rng(42)
    probs_true = rng.uniform(0.05, 0.95, size=300)
    y_cal = (rng.uniform(0.0, 1.0, size=300) < probs_true).astype(np.float64)
    # Logits propositalmente superconfiantes (multiplicados por 4.0)
    overconfident_logits = 4.0 * np.log(probs_true / (1.0 - probs_true))

    scaler = TemperatureScaler().fit(overconfident_logits, y_cal)
    assert scaler.temperature > 1.5
    cal_probs = scaler.predict_proba(overconfident_logits)
    assert float(np.min(cal_probs)) >= 0.0
    assert float(np.max(cal_probs)) <= 1.0
    assert scaler.cal_nll_after < scaler.cal_nll_before
    assert pytest.approx(1.0 / scaler.temperature, abs=0.25) == 0.25
