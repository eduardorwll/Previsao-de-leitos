"""Treina e calibra HydroFiLMMonotonicNet (PyTorch Lightning CPU) no RS (2010-2024).

Aviso: exclusivamente para pesquisa e avaliação metodológica
(Not for patient care or live clinical use).
Splits congelados: Treino <= 2022 | Calibração = 2023 | Auditoria Dev = 2024 | 2025 intocado.
"""

from __future__ import annotations

import time
from pathlib import Path

import lightning as L
import numpy as np
import polars as pl
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from previsao_leitos.audit import brier_skill_score, expected_calibration_error
from previsao_leitos.hydrology import (
    HydroLightningModule,
    IndexedHydroSequenceDataset,
    TemperatureScaler,
)

PANEL_PATH = Path("data/processed/rs_painel_semanal_regioes_v2.parquet")

DYN_COLS = (
    "chuva_sem_mm",
    "chuva_montante_pond",
    "api_max_sem",
    "api_montante_pond",
    "remanso_lagunar_idx",
    "umidade_solo_sem",
    "temp_media_sem_c",
    "week_sin",
    "week_cos",
)
CTX_COLS = ("ctx_rain_mean", "ctx_rain_std")
MONO_POS_COLS = (
    "frac_urbano",
    "frac_banhado_agua",
    "frac_area_polder_regiao",
    "indice_aterro_varzea",
)
MONO_NEG_COLS = ("frac_floresta",)


def _predict_logits(
    module: HydroLightningModule, ds: IndexedHydroSequenceDataset
) -> tuple[np.ndarray, np.ndarray]:
    module.eval()
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    logits_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            lg, _ = module(
                batch["x_dyn"], batch["x_ctx"], batch["x_mono_pos"], batch["x_mono_neg"]
            )
            logits_all.append(lg.cpu().numpy())
            y_all.append(batch["y_event"].cpu().numpy())
    return np.concatenate(logits_all), np.concatenate(y_all)


def main() -> None:
    t0 = time.perf_counter()
    L.seed_everything(42, workers=True)
    panel = pl.read_parquet(PANEL_PATH).filter(pl.col("ano_epi") <= 2024)

    # Sazonalidade cíclica + contexto físico climatológico de bacia ajustado APENAS em <= 2022
    panel = panel.with_columns(
        (2.0 * np.pi * pl.col("semana_epi").cast(pl.Float64) / 52.0).sin().alias("week_sin"),
        (2.0 * np.pi * pl.col("semana_epi").cast(pl.Float64) / 52.0).cos().alias("week_cos"),
    )
    train_raw = panel.filter(pl.col("ano_epi") <= 2022)
    ctx_stats = train_raw.group_by("nome_regiao_saude").agg(
        pl.col("chuva_sem_mm").mean().alias("ctx_rain_mean"),
        pl.col("chuva_sem_mm").std().fill_null(1.0).alias("ctx_rain_std"),
    )
    panel = panel.join(ctx_stats, on="nome_regiao_saude", how="left")

    # Alvo físico ex-ante (t + 2 sem.): excesso severo de saturação/enchente (>= P90 <= 2022)
    def _z_stats(col: str) -> tuple[float, float]:
        mu = float(train_raw.select(pl.col(col).mean()).item())
        sd = float(train_raw.select(pl.col(col).std()).item() or 1.0)
        return mu, sd if sd > 1e-8 else 1.0

    m_api, s_api = _z_stats("api_max_sem")
    m_up, s_up = _z_stats("api_montante_pond")
    m_rem, s_rem = _z_stats("remanso_lagunar_idx")

    panel = panel.sort(["nome_regiao_saude", "ano_epi", "semana_epi"]).with_columns(
        (
            (pl.col("api_max_sem") - m_api) / s_api
            + (pl.col("api_montante_pond") - m_up) / s_up
            + 0.5 * (pl.col("remanso_lagunar_idx") - m_rem) / s_rem
        ).alias("_hydro_sev_t")
    )
    sev_p90_train = float(
        panel.filter(pl.col("ano_epi") <= 2022).select(pl.col("_hydro_sev_t").quantile(0.90)).item()
    )
    panel = panel.with_columns(
        pl.col("_hydro_sev_t").shift(-2).over("nome_regiao_saude").alias("y_sev_t2"),
        pl.col("ano_epi").shift(-2).over("nome_regiao_saude").alias("target_ano_epi"),
    ).with_columns(
        (pl.col("y_sev_t2") >= sev_p90_train).cast(pl.Float32).alias("y_event_t2")
    )

    # Padroniza x_dyn e x_ctx estritamente sobre <= 2022 (x_mono preservado positivo [0, 1])
    scale_cols = (*DYN_COLS, *CTX_COLS)
    train_for_scale = panel.filter(pl.col("target_ano_epi") <= 2022)
    scale_exprs: list[pl.Expr] = []
    for col in scale_cols:
        mu = float(train_for_scale.select(pl.col(col).mean()).item())
        sd = float(train_for_scale.select(pl.col(col).std()).item() or 1.0)
        scale_exprs.append(((pl.col(col) - mu) / (sd if sd > 1e-8 else 1.0)).alias(col))
    scaled = panel.with_columns(scale_exprs)

    def _make_ds(sub_df: pl.DataFrame) -> IndexedHydroSequenceDataset:
        return IndexedHydroSequenceDataset(
            sub_df,
            entity_col="nome_regiao_saude",
            dyn_cols=DYN_COLS,
            ctx_cols=CTX_COLS,
            mono_pos_cols=MONO_POS_COLS,
            mono_neg_cols=MONO_NEG_COLS,
            event_col="y_event_t2",
            severity_col="y_sev_t2",
            seq_len=8,
        )

    ds_train = _make_ds(scaled.filter(pl.col("target_ano_epi") <= 2022))
    ds_cal = _make_ds(
        scaled.filter((pl.col("ano_epi") >= 2022) & (pl.col("target_ano_epi") == 2023))
    )
    ds_2024 = _make_ds(
        scaled.filter((pl.col("ano_epi") >= 2023) & (pl.col("target_ano_epi") == 2024))
    )

    module = HydroLightningModule(
        dyn_dim=len(DYN_COLS),
        ctx_dim=len(CTX_COLS),
        mono_pos_dim=len(MONO_POS_COLS),
        mono_neg_dim=len(MONO_NEG_COLS),
        hidden_dim=32,
        num_heads=4,
        learning_rate=1.5e-3,
        pos_weight=1.0,
    )
    trainer = L.Trainer(
        max_epochs=8,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
    )
    trainer.fit(
        module,
        train_dataloaders=DataLoader(ds_train, batch_size=128, shuffle=True),
        val_dataloaders=DataLoader(ds_cal, batch_size=256, shuffle=False),
    )

    # Calibração pós-hoc estritamente sobre 2023 -> avaliação honesta em 2024
    logits_cal, y_cal = _predict_logits(module, ds_cal)
    logits_2024, y_2024 = _predict_logits(module, ds_2024)

    scaler = TemperatureScaler().fit(logits_cal, y_cal)
    p_raw = 1.0 / (1.0 + np.exp(-np.clip(logits_2024, -40.0, 40.0)))
    p_cal = scaler.predict_proba(logits_2024)

    # Auditoria autograd em 100% do lote de 2024
    sample_loader = DataLoader(ds_2024, batch_size=256, shuffle=False)
    batch0 = next(iter(sample_loader))
    xp = batch0["x_mono_pos"].clone().detach().requires_grad_(True)
    xn = batch0["x_mono_neg"].clone().detach().requires_grad_(True)
    lg_chk, sv_chk = module(batch0["x_dyn"], batch0["x_ctx"], xp, xn)
    gp_l, gn_l = torch.autograd.grad(lg_chk.sum(), (xp, xn), retain_graph=True)
    gp_s, gn_s = torch.autograd.grad(sv_chk.sum(), (xp, xn))
    mono_ok = bool(
        (gp_l > 0).all().item()
        and (gn_l < 0).all().item()
        and (gp_s > 0).all().item()
        and (gn_s < 0).all().item()
    )

    mono_status = "APROVADA (0.00% viol.)" if mono_ok else "FALHA"
    print(
        f"=== HydroFiLMMonotonicNet RS (Treino <= 2022 | Cal 2023 | Audit 2024 | h=2 sem.) ===\n"
        f"Amostras: train={len(ds_train):,} | cal2023={len(ds_cal):,} "
        f"| audit2024={len(ds_2024):,}\n"
        f"TemperatureScaler (2023): T={scaler.temperature:.3f} "
        f"(NLL {scaler.cal_nll_before:.4f} -> {scaler.cal_nll_after:.4f})\n"
        f"Audit 2024 | ROC-AUC={roc_auc_score(y_2024, p_cal):.4f} "
        f"| PR-AUC={average_precision_score(y_2024, p_cal):.4f}\n"
        f"Calibração 2024 | ECE raw={expected_calibration_error(y_2024, p_raw):.4f} -> "
        f"cal={expected_calibration_error(y_2024, p_cal):.4f} "
        f"| BSS raw={brier_skill_score(y_2024, p_raw):+.4f} -> "
        f"cal={brier_skill_score(y_2024, p_cal):+.4f}\n"
        f"Auditoria Autograd Monotonicidade (ADR-0001): {mono_status}\n"
        f"Tempo total CPU: {time.perf_counter() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
