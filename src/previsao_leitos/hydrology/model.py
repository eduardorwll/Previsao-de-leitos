"""HydroFiLMMonotonicNet e HydroLightningModule com monotonicidade estrutural real (ADR-0001)."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class HydroFiLMMonotonicNet(nn.Module):
    """Rede hidrológica Bi-GRU + Multi-Head Attention + FiLM + Ramo Monotônico `F.softplus`.

    Garante matematicamente (verificado por `torch.autograd.grad`):
    - `d(event_logits)/d(x_mono_pos) > 0` e `d(severity)/d(x_mono_pos) > 0`
    - `d(event_logits)/d(x_mono_neg) < 0` e `d(severity)/d(x_mono_neg) < 0`
    Sem `mun_embedding` e sem impermeabilização/aterro no ramo dinâmico ou FiLM.
    """

    def __init__(
        self,
        *,
        dyn_dim: int,
        ctx_dim: int,
        mono_pos_dim: int,
        mono_neg_dim: int,
        enso_dim: int = 0,
        hidden_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self._enso_dim = enso_dim
        self._gru = nn.GRU(
            input_size=dyn_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        attn_dim = hidden_dim * 2
        self._attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self._norm = nn.LayerNorm(attn_dim)
        # FiLM: modula representações temporais pelo regime climático da bacia + ENSO/Atlântico
        film_in_dim = ctx_dim + enso_dim
        self._film_gamma = nn.Linear(film_in_dim, attn_dim)
        self._film_beta = nn.Linear(film_in_dim, attn_dim)

        self._base_logit_head = nn.Sequential(
            nn.Linear(attn_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._base_sev_head = nn.Sequential(
            nn.Linear(attn_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Ramo monotônico via F.softplus (estritamente positivo -> aplicado a AMBAS cabeças)
        self._raw_w_pos_logit = nn.Parameter(torch.zeros(mono_pos_dim))
        self._raw_w_neg_logit = nn.Parameter(torch.zeros(mono_neg_dim))
        self._raw_w_pos_sev = nn.Parameter(torch.zeros(mono_pos_dim))
        self._raw_w_neg_sev = nn.Parameter(torch.zeros(mono_neg_dim))

    def forward(
        self,
        x_dyn: torch.Tensor,
        x_ctx: torch.Tensor,
        x_mono_pos: torch.Tensor,
        x_mono_neg: torch.Tensor,
        x_enso: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_out, _ = self._gru(x_dyn)
        attn_out, _ = self._attn(seq_out, seq_out, seq_out, need_weights=False)
        h = self._norm(seq_out + attn_out)[:, -1, :]

        if self._enso_dim > 0:
            enso_vec = (
                x_enso
                if x_enso is not None
                else torch.zeros(
                    (x_ctx.shape[0], self._enso_dim),
                    dtype=x_ctx.dtype,
                    device=x_ctx.device,
                )
            )
            film_in = torch.cat([x_ctx, enso_vec], dim=-1)
        else:
            film_in = x_ctx

        gamma = 1.0 + 0.25 * torch.tanh(self._film_gamma(film_in))
        beta = self._film_beta(film_in)
        h_mod = gamma * h + beta

        base_logit = self._base_logit_head(h_mod).squeeze(-1)
        base_sev = self._base_sev_head(h_mod).squeeze(-1)

        w_pos_l = F.softplus(self._raw_w_pos_logit)
        w_neg_l = F.softplus(self._raw_w_neg_logit)
        w_pos_s = F.softplus(self._raw_w_pos_sev)
        w_neg_s = F.softplus(self._raw_w_neg_sev)

        mono_logit = (x_mono_pos @ w_pos_l) - (x_mono_neg @ w_neg_l)
        mono_sev = (x_mono_pos @ w_pos_s) - (x_mono_neg @ w_neg_s)

        return base_logit + mono_logit, base_sev + mono_sev


class HydroLightningModule(L.LightningModule):
    """Módulo PyTorch Lightning limpo (sem transferências redundantes nem duplo sigmoid)."""

    def __init__(
        self,
        *,
        dyn_dim: int,
        ctx_dim: int,
        mono_pos_dim: int,
        mono_neg_dim: int,
        enso_dim: int = 0,
        hidden_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.15,
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-4,
        pos_weight: float = 4.0,
        severity_loss_weight: float = 0.35,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = HydroFiLMMonotonicNet(
            dyn_dim=dyn_dim,
            ctx_dim=ctx_dim,
            mono_pos_dim=mono_pos_dim,
            mono_neg_dim=mono_neg_dim,
            enso_dim=enso_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self._lr = learning_rate
        self._wd = weight_decay
        self._sev_weight = severity_loss_weight
        self.register_buffer("_pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(
        self,
        x_dyn: torch.Tensor,
        x_ctx: torch.Tensor,
        x_mono_pos: torch.Tensor,
        x_mono_neg: torch.Tensor,
        x_enso: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out: tuple[torch.Tensor, torch.Tensor] = self.net(
            x_dyn, x_ctx, x_mono_pos, x_mono_neg, x_enso=x_enso
        )
        return out

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        logits, sev = self(
            batch["x_dyn"],
            batch["x_ctx"],
            batch["x_mono_pos"],
            batch["x_mono_neg"],
            x_enso=batch.get("x_enso"),
        )
        pos_w = self.get_buffer("_pos_weight")
        bce = F.binary_cross_entropy_with_logits(logits, batch["y_event"], pos_weight=pos_w)
        huber = F.huber_loss(sev, batch["y_sev"])
        loss = bce + self._sev_weight * huber
        if self._trainer is not None:
            self.log(f"{stage}_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, "val")

    def configure_optimizers(self) -> Any:
        return torch.optim.AdamW(self.parameters(), lr=self._lr, weight_decay=self._wd)
