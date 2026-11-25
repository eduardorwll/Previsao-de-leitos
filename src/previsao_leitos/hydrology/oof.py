"""Auditoria autograd fp32 de monotonicidade e gerador causal out-of-fold (F03.1)."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

import lightning as L
import numpy as np
import polars as pl
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from previsao_leitos.domain import DataContractError
from previsao_leitos.hydrology.calibration import TemperatureScaler
from previsao_leitos.hydrology.dataset import IndexedHydroSequenceDataset
from previsao_leitos.hydrology.device import (
    DeviceRequest,
    release_gpu_memory,
    resolve_accelerator,
)
from previsao_leitos.hydrology.model import HydroFiLMMonotonicNet, HydroLightningModule


@dataclass(frozen=True)
class NeuralHydroOofReport:
    """Metadados auditáveis da execução OOF causal real de HydroFiLM-ENSO."""

    backend: str
    device_name: str | None
    n_folds: int
    seeds: tuple[int, ...]
    autograd_violation_rate: float
    elapsed_seconds: float
    peak_vram_mb: float
    corr_rain_severity: float
    temperature: float


def audit_autograd_monotonicity(
    net: HydroFiLMMonotonicNet,
    *,
    x_dyn: torch.Tensor,
    x_ctx: torch.Tensor,
    x_mono_pos: torch.Tensor,
    x_mono_neg: torch.Tensor,
    x_enso: torch.Tensor | None = None,
) -> float:
    """Audita em fp32 via torch.autograd.grad a taxa de violação monotônica em ambas as cabeças.

    Move automaticamente todos os tensores para o dispositivo do modelo (`next(net.parameters())`)
    e desativa `autocast` para evitar underflow numérico em mixed precision.
    """
    was_training = net.training
    net.eval()
    param = next(net.parameters(), None)
    dev = param.device if param is not None else x_dyn.device

    dyn_in = x_dyn.detach().to(device=dev, dtype=torch.float32)
    ctx_in = x_ctx.detach().to(device=dev, dtype=torch.float32)
    pos_in = x_mono_pos.detach().to(device=dev, dtype=torch.float32).clone().requires_grad_(True)
    neg_in = x_mono_neg.detach().to(device=dev, dtype=torch.float32).clone().requires_grad_(True)
    enso_in = (
        x_enso.detach().to(device=dev, dtype=torch.float32) if x_enso is not None else None
    )

    device_type = "cuda" if dev.type == "cuda" else "cpu"
    with torch.amp.autocast(device_type=device_type, enabled=False):
        logits, sev = net(dyn_in, ctx_in, pos_in, neg_in, x_enso=enso_in)
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
    if was_training:
        net.train()
    return float(violations / max(total, 1))


def audit_dataset_autograd_monotonicity(
    net: HydroFiLMMonotonicNet,
    dataset: IndexedHydroSequenceDataset,
    *,
    batch_size: int = 256,
) -> float:
    """Audita via autograd fp32 sobre 100% do dataset (nunca apenas batch0)."""
    if len(dataset) == 0:
        raise DataContractError("audit_dataset_autograd_monotonicity: dataset vazio.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_viol = 0.0
    total_samples = 0
    for batch in loader:
        n_b = int(batch["x_dyn"].shape[0])
        rate_b = audit_autograd_monotonicity(
            net,
            x_dyn=batch["x_dyn"],
            x_ctx=batch["x_ctx"],
            x_mono_pos=batch["x_mono_pos"],
            x_mono_neg=batch["x_mono_neg"],
            x_enso=batch.get("x_enso"),
        )
        total_viol += rate_b * n_b
        total_samples += n_b
    return float(total_viol / max(total_samples, 1))


def _predict_module_on_dataset(
    module: HydroLightningModule,
    dataset: IndexedHydroSequenceDataset,
    *,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Inferência determinística movendo tensores para `module.device`."""
    module.eval()
    dev = module.device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logits_all: list[np.ndarray] = []
    sev_all: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x_dyn = batch["x_dyn"].to(dev)
            x_ctx = batch["x_ctx"].to(dev)
            x_pos = batch["x_mono_pos"].to(dev)
            x_neg = batch["x_mono_neg"].to(dev)
            x_enso = batch["x_enso"].to(dev) if "x_enso" in batch else None
            lg, sv = module(x_dyn, x_ctx, x_pos, x_neg, x_enso=x_enso)
            logits_all.append(lg.detach().cpu().numpy().astype(np.float64))
            sev_all.append(sv.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(logits_all), np.concatenate(sev_all)


def generate_neural_hydro_oof_predictions(
    panel_df: pl.DataFrame,
    *,
    entity_col: str,
    dyn_cols: Sequence[str],
    ctx_cols: Sequence[str],
    mono_pos_cols: Sequence[str],
    mono_neg_cols: Sequence[str],
    event_col: str,
    severity_col: str,
    enso_cols: Sequence[str] = (),
    train_end_year: int = 2022,
    cal_year: int = 2023,
    seq_len: int = 8,
    max_epochs: int = 8,
    batch_size: int = 128,
    hidden_dim: int = 32,
    num_heads: int = 4,
    learning_rate: float = 1.5e-3,
    seeds: Sequence[int] = (42, 137, 2024),
    requested_device: DeviceRequest = "auto",
    preload_vram: bool = False,
    rain_col_for_corr: str = "chuva_montante_multihop",
    log_progress: bool = False,
) -> tuple[pl.DataFrame, NeuralHydroOofReport]:
    """Treina `HydroFiLMMonotonicNet` em folds anuais expansivos causais (multi-seed).

    - Elimina o Bug E (OOF linear circular) treinando a rede neural monotônica real
      sobre janelas sequenciais causais `<= y - 1` para prever `y` (`<= train_end_year`),
      e modelo congelado em `<= train_end_year` para anos `> train_end_year`.
    - Executa auditoria `autograd` em `fp32` sobre 100% do split de cada fold.
    - Calibra temperatura (`TemperatureScaler`) estritamente sobre `cal_year`.
    """
    t0 = time.perf_counter()
    required = [
        entity_col,
        "ano_epi",
        "semana_epi",
        event_col,
        severity_col,
        *dyn_cols,
        *ctx_cols,
        *enso_cols,
        *mono_pos_cols,
        *mono_neg_cols,
    ]
    missing = [c for c in required if c not in panel_df.columns]
    if missing:
        raise DataContractError(f"Colunas ausentes para OOF neural F03.1: {missing}")

    years = sorted(int(y) for y in panel_df["ano_epi"].unique().to_list())
    train_years = [y for y in years if y <= train_end_year]
    if len(train_years) < 2:
        raise DataContractError("F03.1 OOF neural requer ao menos 2 anos em <= train_end_year.")

    spec = resolve_accelerator(prefer_gpu=True, requested_device=requested_device)
    if spec.accelerator == "gpu" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Reduz banners repetitivos por fold sem suprimir warnings Python
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
    logging.getLogger("lightning.fabric").setLevel(logging.WARNING)
    logging.getLogger("lightning_fabric").setLevel(logging.WARNING)

    scale_cols = (*dyn_cols, *ctx_cols, *enso_cols)
    key_cols = [entity_col, "ano_epi", "semana_epi"]
    ordered_base = panel_df.sort(key_cols)

    # Acumuladores por chave (entity, ano_epi, semana_epi)
    pred_records: list[pl.DataFrame] = []
    max_autograd_viol = 0.0
    n_folds_run = 0
    post_years_all = {y for y in years if y > train_end_year}
    total_folds = len(train_years[1:]) + (1 if post_years_all else 0)

    def _standardize_pair(
        fit_df: pl.DataFrame, score_df: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        exprs: list[pl.Expr] = []
        for col in scale_cols:
            arr = fit_df[col].fill_null(0.0).to_numpy().astype(np.float64)
            finite = arr[np.isfinite(arr)]
            mu = float(np.mean(finite)) if finite.size > 0 else 0.0
            sd = float(np.std(finite)) if finite.size > 1 else 1.0
            sd = sd if sd > 1e-8 else 1.0
            exprs.append(((pl.col(col).fill_null(mu).cast(pl.Float64) - mu) / sd).alias(col))
        return fit_df.with_columns(exprs), score_df.with_columns(exprs)

    def _fit_and_score_fold(
        fit_raw: pl.DataFrame,
        score_raw: pl.DataFrame,
        target_years: set[int],
    ) -> pl.DataFrame | None:
        nonlocal max_autograd_viol, n_folds_run
        fit_sc, score_sc = _standardize_pair(fit_raw, score_raw)
        dev_preload = spec.device_str if (preload_vram and spec.accelerator == "gpu") else None

        ds_tr = IndexedHydroSequenceDataset(
            fit_sc,
            entity_col=entity_col,
            dyn_cols=dyn_cols,
            ctx_cols=ctx_cols,
            enso_cols=enso_cols,
            mono_pos_cols=mono_pos_cols,
            mono_neg_cols=mono_neg_cols,
            event_col=event_col,
            severity_col=severity_col,
            seq_len=seq_len,
            preload_device=dev_preload,
        )
        ds_sc = IndexedHydroSequenceDataset(
            score_sc,
            entity_col=entity_col,
            dyn_cols=dyn_cols,
            ctx_cols=ctx_cols,
            enso_cols=enso_cols,
            mono_pos_cols=mono_pos_cols,
            mono_neg_cols=mono_neg_cols,
            event_col=event_col,
            severity_col=severity_col,
            seq_len=seq_len,
            preload_device=None,
        )
        if len(ds_tr) < 16 or len(ds_sc) == 0:
            return None

        logits_seeds: list[np.ndarray] = []
        sev_seeds: list[np.ndarray] = []

        fold_t0 = time.perf_counter()
        for seed in seeds:
            L.seed_everything(int(seed), workers=True, verbose=False)
            module = HydroLightningModule(
                dyn_dim=len(dyn_cols),
                ctx_dim=len(ctx_cols),
                enso_dim=len(enso_cols),
                mono_pos_dim=len(mono_pos_cols),
                mono_neg_dim=len(mono_neg_cols),
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                learning_rate=learning_rate,
                pos_weight=1.0,
            )
            trainer = L.Trainer(
                max_epochs=max_epochs,
                accelerator=spec.accelerator,
                devices=1,
                precision=spec.precision,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                deterministic="warn" if spec.deterministic_mode == "warn" else True,
            )
            trainer.fit(
                module,
                train_dataloaders=DataLoader(ds_tr, batch_size=batch_size, shuffle=True),
            )

            module.to(torch.device(spec.device_str))
            lg_s, sv_s = _predict_module_on_dataset(module, ds_sc, batch_size=256)
            viol = audit_dataset_autograd_monotonicity(module.net, ds_sc, batch_size=256)
            max_autograd_viol = max(max_autograd_viol, viol)

            logits_seeds.append(lg_s)
            sev_seeds.append(np.maximum(0.0, sv_s))

            del trainer, module
            release_gpu_memory()

        n_folds_run += 1
        if log_progress:
            fold_dt = time.perf_counter() - fold_t0
            print(
                f"[F03.1 OOF] Fold {n_folds_run}/{total_folds} "
                f"(alvo={sorted(target_years)}, tr={len(ds_tr):,}, sc={len(ds_sc):,}) "
                f"ok em {fold_dt:.1f}s (viol={max_autograd_viol:.4f})",
                flush=True,
            )
        meta = ds_sc.sample_metadata.with_columns(
            pl.Series("_oof_logit", np.mean(logits_seeds, axis=0), dtype=pl.Float64),
            pl.Series("_oof_sev", np.mean(sev_seeds, axis=0), dtype=pl.Float64),
        ).filter(pl.col("ano_epi").is_in(sorted(target_years)))
        return meta

    # Folds anuais expansivos dentro do treino (<= train_end_year)
    for yr in train_years[1:]:
        fit_df = ordered_base.filter(pl.col("ano_epi") < yr)
        # Inclui contexto do ano anterior apenas para formar janelas de tamanho seq_len no início
        score_df = ordered_base.filter((pl.col("ano_epi") >= yr - 1) & (pl.col("ano_epi") <= yr))
        fold_out = _fit_and_score_fold(fit_df, score_df, {yr})
        if fold_out is not None and not fold_out.is_empty():
            pred_records.append(fold_out)

    # Fold final congelado em <= train_end_year para anos pós-treino (cal_year + eval_year)
    post_years = {y for y in years if y > train_end_year}
    if post_years:
        fit_df = ordered_base.filter(pl.col("ano_epi") <= train_end_year)
        score_df = ordered_base.filter(pl.col("ano_epi") >= train_end_year)
        fold_out = _fit_and_score_fold(fit_df, score_df, post_years)
        if fold_out is not None and not fold_out.is_empty():
            pred_records.append(fold_out)

    if pred_records:
        all_preds = pl.concat(pred_records).unique(subset=key_cols, keep="last")
        merged = ordered_base.join(all_preds, on=key_cols, how="left")
    else:
        merged = ordered_base.with_columns(
            pl.lit(-2.0, dtype=pl.Float64).alias("_oof_logit"),
            pl.lit(0.0, dtype=pl.Float64).alias("_oof_sev"),
        )

    logits_arr = merged["_oof_logit"].fill_null(-2.0).to_numpy().astype(np.float64)
    sev_arr = np.maximum(0.0, merged["_oof_sev"].fill_null(0.0).to_numpy().astype(np.float64))
    y_ev_all = merged[event_col].fill_null(0.0).to_numpy().astype(np.float64)
    yr_all = merged["ano_epi"].to_numpy()

    cal_mask = (yr_all == cal_year) & merged["_oof_logit"].is_not_null().to_numpy()
    temp_val = 1.0
    if int(np.sum(cal_mask)) >= 10 and len(np.unique((y_ev_all[cal_mask] > 0.5).astype(int))) >= 2:
        scaler = TemperatureScaler().fit(logits_arr[cal_mask], y_ev_all[cal_mask])
        probs = scaler.predict_proba(logits_arr)
        temp_val = float(scaler.temperature)
    else:
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits_arr, -30.0, 30.0)))

    corr_val = 0.0
    if rain_col_for_corr in merged.columns:
        rain_arr = merged[rain_col_for_corr].fill_null(0.0).to_numpy().astype(np.float64)
        valid_c = merged["_oof_sev"].is_not_null().to_numpy() & np.isfinite(rain_arr)
        if (
            int(np.sum(valid_c)) > 5
            and np.std(rain_arr[valid_c]) > 1e-8
            and np.std(sev_arr[valid_c]) > 1e-8
        ):
            corr_val = float(np.corrcoef(rain_arr[valid_c], sev_arr[valid_c])[0, 1])

    peak_vram_mb = 0.0
    if spec.accelerator == "gpu" and torch.cuda.is_available():
        peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

    enriched = merged.drop(["_oof_logit", "_oof_sev"]).with_columns(
        pl.Series("prob_enchente_f03_oof", probs, dtype=pl.Float64),
        pl.Series("severidade_hidro_f03_oof", sev_arr, dtype=pl.Float64),
    )
    report = NeuralHydroOofReport(
        backend=spec.backend,
        device_name=spec.device_name,
        n_folds=n_folds_run,
        seeds=tuple(int(s) for s in seeds),
        autograd_violation_rate=round(max_autograd_viol, 8),
        elapsed_seconds=round(time.perf_counter() - t0, 3),
        peak_vram_mb=round(peak_vram_mb, 2),
        corr_rain_severity=round(corr_val, 6),
        temperature=round(temp_val, 4),
    )
    return enriched, report


def generate_causal_hydro_oof_predictions(
    panel_df: pl.DataFrame,
    *,
    feature_cols: Sequence[str],
    event_col: str,
    severity_col: str,
    train_end_year: int = 2022,
    cal_year: int = 2023,
) -> pl.DataFrame:
    """Baseline rápido linear out-of-fold (compatibilidade de testes unitários leves)."""
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
            logits_te = np.full(
                x_te.shape[0], np.log(p_const / (1.0 - p_const)), dtype=np.float64
            )
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

    first_mask = yr_all == train_years[0]
    oof_logits[first_mask] = -2.0
    oof_sev[first_mask] = 0.0

    for yr in train_years[1:]:
        fit_m = yr_all < yr
        pred_m = yr_all == yr
        oof_logits[pred_m], oof_sev[pred_m] = _fit_and_score(fit_m, pred_m)

    post_mask = yr_all > train_end_year
    if np.any(post_mask):
        fit_m = yr_all <= train_end_year
        oof_logits[post_mask], oof_sev[post_mask] = _fit_and_score(fit_m, post_mask)

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
