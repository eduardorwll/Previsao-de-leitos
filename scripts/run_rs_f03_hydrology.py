"""Treina e calibra HydroFiLM-ENSO (PyTorch Lightning GPU ROCm/CUDA ou CPU) no RS (2010-2024).

Aviso: exclusivamente para pesquisa e avaliação metodológica
(Not for patient care or live clinical use).
Splits congelados: Treino <= 2022 | Calibração = 2023 | Auditoria Dev = 2024 | 2025 intocado.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightning as L
import numpy as np
import polars as pl
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from previsao_leitos.audit import brier_skill_score, expected_calibration_error
from previsao_leitos.geo import build_causal_enso_weekly_features
from previsao_leitos.hydrology import (
    HydroLightningModule,
    IndexedHydroSequenceDataset,
    TemperatureScaler,
    audit_dataset_autograd_monotonicity,
    generate_neural_hydro_oof_predictions,
    release_gpu_memory,
    resolve_accelerator,
)
from previsao_leitos.hydrology.device import DeviceRequest
from previsao_leitos.ingestion import (
    fetch_noaa_enso_monthly,
    fetch_zenodo_iph_rs_metadata,
)

V3_PANEL_PATH = Path("data/processed/rs_painel_semanal_regioes_v3.parquet")
V2_PANEL_PATH = Path("data/processed/rs_painel_semanal_regioes_v2.parquet")

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
ENSO_COLS = ("el_nino_intensity", "oni_c_lag4")
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
    dev = module.device
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    logits_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x_enso = batch["x_enso"].to(dev) if "x_enso" in batch else None
            lg, _ = module(
                batch["x_dyn"].to(dev),
                batch["x_ctx"].to(dev),
                batch["x_mono_pos"].to(dev),
                batch["x_mono_neg"].to(dev),
                x_enso=x_enso,
            )
            logits_all.append(lg.detach().cpu().numpy())
            y_all.append(batch["y_event"].detach().cpu().numpy())
    return np.concatenate(logits_all), np.concatenate(y_all)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Treina F03.1 HydroFiLM-ENSO (GPU/CPU) no RS.")
    parser.add_argument("--device", choices=("auto", "cpu", "rocm", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--preload-vram", action="store_true", help="Pré-carrega dataset em VRAM.")
    parser.add_argument("--run-oof", action="store_true", help="Gera e salva OOF 13x3 em Parquet.")
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,137,2024",
        help="Seeds separadas por vírgula para OOF multi-seed.",
    )
    parser.add_argument(
        "--output-oof",
        type=Path,
        default=Path("data/processed/rs_f03_hydro_oof_v3.parquet"),
    )
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    req_dev: DeviceRequest = args.device
    spec = resolve_accelerator(prefer_gpu=True, requested_device=req_dev)
    print(
        f"[F03.1] Acelerador: {spec.accelerator} ({spec.backend}) | "
        f"GPU={spec.device_name} | runtime={spec.runtime_version} | det={spec.deterministic_mode}"
    )

    panel_path = V3_PANEL_PATH if V3_PANEL_PATH.exists() else V2_PANEL_PATH
    panel = pl.read_parquet(panel_path).filter(pl.col("ano_epi") <= 2024)

    # Ingestão ao vivo Onda 2A (NOAA ONI/MEI + metadados Zenodo IPH-UFRGS com cache SHA-256)
    wave2a_dir = Path("data/raw/wave2a")
    noaa_monthly, noaa_manifests = fetch_noaa_enso_monthly(
        cache_dir=wave2a_dir, include_mei_v2=True, max_year=2024
    )
    print(
        f"[Onda 2A] NOAA ENSO carregado: {noaa_monthly.height} meses "
        f"(cache_hit={[m.cache_hit for m in noaa_manifests]})"
    )
    try:
        zen = fetch_zenodo_iph_rs_metadata(cache_dir=wave2a_dir)
        print(f"[Onda 2A] Zenodo {zen.doi}: {len(zen.files)} arquivo(s) catalogados")
    except Exception as exc:
        print(f"[Onda 2A] Zenodo offline/indisponível ({exc}); prosseguindo com alvo físico RS.")

    enso_wk = build_causal_enso_weekly_features(noaa_monthly, lags=(4, 12)).select(
        ["ano_epi", "semana_epi", "el_nino_intensity", "oni_c_lag4"]
    )
    panel = (
        panel.drop([c for c in ("el_nino_intensity", "oni_c_lag4") if c in panel.columns])
        .join(enso_wk, on=["ano_epi", "semana_epi"], how="left")
        .with_columns(
            pl.col("el_nino_intensity").fill_null(0.0),
            pl.col("oni_c_lag4").fill_null(0.0),
            (2.0 * np.pi * pl.col("semana_epi").cast(pl.Float64) / 52.0).sin().alias("week_sin"),
            (2.0 * np.pi * pl.col("semana_epi").cast(pl.Float64) / 52.0).cos().alias("week_cos"),
        )
    )

    train_raw = panel.filter(pl.col("ano_epi") <= 2022)
    ctx_stats = train_raw.group_by("nome_regiao_saude").agg(
        pl.col("chuva_sem_mm").mean().alias("ctx_rain_mean"),
        pl.col("chuva_sem_mm").std().fill_null(1.0).alias("ctx_rain_std"),
    )
    panel = panel.join(ctx_stats, on="nome_regiao_saude", how="left")

    # Alvo físico ex-ante (t + 2 sem.): excesso de saturação/enchente (>= P90 <= 2022)
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

    if args.run_oof:
        seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
        oof_panel, oof_rep = generate_neural_hydro_oof_predictions(
            panel.drop_nulls(subset=["y_event_t2", "y_sev_t2"]),
            entity_col="nome_regiao_saude",
            dyn_cols=DYN_COLS,
            ctx_cols=CTX_COLS,
            enso_cols=ENSO_COLS,
            mono_pos_cols=MONO_POS_COLS,
            mono_neg_cols=MONO_NEG_COLS,
            event_col="y_event_t2",
            severity_col="y_sev_t2",
            train_end_year=2022,
            cal_year=2023,
            seq_len=8,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            seeds=seeds,
            requested_device=req_dev,
            preload_vram=args.preload_vram,
            rain_col_for_corr="chuva_montante_multihop"
            if "chuva_montante_multihop" in panel.columns
            else "chuva_montante_pond",
            log_progress=True,
        )
        oof_out = oof_panel.select(
            [
                "nome_regiao_saude",
                "ano_epi",
                "semana_epi",
                "prob_enchente_f03_oof",
                "severidade_hidro_f03_oof",
            ]
        )
        args.output_oof.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = args.output_oof.with_suffix(".parquet.tmp")
        oof_out.write_parquet(tmp_p)
        tmp_p.replace(args.output_oof)
        meta_path = args.output_oof.with_suffix(".report.json")
        meta_path.write_text(json.dumps(oof_rep.__dict__, indent=2), encoding="utf-8")

        ev24 = oof_panel.filter(pl.col("ano_epi") == 2024)
        y24 = ev24["y_event_t2"].to_numpy()
        p24 = ev24["prob_enchente_f03_oof"].to_numpy()
        print(
            f"=== F03.1 HydroFiLM-ENSO OOF Real "
            f"({oof_rep.n_folds} folds x {len(seeds)} seeds) ===\n"
            f"Backend: {oof_rep.backend} ({oof_rep.device_name}) | "
            f"Tempo: {oof_rep.elapsed_seconds:.1f}s | Pico VRAM: {oof_rep.peak_vram_mb:.1f} MB\n"
            f"Autograd fp32 (100% splits): {oof_rep.autograd_violation_rate * 100:.4f}% viol. | "
            f"corr(chuva, sev_oof): {oof_rep.corr_rain_severity:.4f}\n"
            f"Audit 2024 | ROC-AUC={roc_auc_score(y24, p24):.4f} | "
            f"ECE={expected_calibration_error(y24, p24):.4f} | "
            f"BSS={brier_skill_score(y24, p24):+.4f}\n"
            f"Salvo em: {args.output_oof}"
        )
        return 0

    # Execução single-fit de verificação rápida (Treino <= 2022 | Cal 2023 | Audit 2024)
    L.seed_everything(42, workers=True)
    scale_cols = (*DYN_COLS, *CTX_COLS, *ENSO_COLS)
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
            enso_cols=ENSO_COLS,
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
        enso_dim=len(ENSO_COLS),
        mono_pos_dim=len(MONO_POS_COLS),
        mono_neg_dim=len(MONO_NEG_COLS),
        hidden_dim=32,
        num_heads=4,
        learning_rate=1.5e-3,
        pos_weight=1.0,
    )
    trainer = L.Trainer(
        max_epochs=args.epochs,
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
        train_dataloaders=DataLoader(ds_train, batch_size=args.batch_size, shuffle=True),
        val_dataloaders=DataLoader(ds_cal, batch_size=256, shuffle=False),
    )

    module.to(torch.device(spec.device_str))
    logits_cal, y_cal = _predict_logits(module, ds_cal)
    logits_2024, y_2024 = _predict_logits(module, ds_2024)

    scaler = TemperatureScaler().fit(logits_cal, y_cal)
    p_raw = 1.0 / (1.0 + np.exp(-np.clip(logits_2024, -40.0, 40.0)))
    p_cal = scaler.predict_proba(logits_2024)

    # Auditoria autograd fp32 sobre 100% do split 2024 (nunca apenas batch0)
    viol_rate = audit_dataset_autograd_monotonicity(module.net, ds_2024, batch_size=256)
    mono_status = f"APROVADA ({viol_rate * 100:.4f}% viol.)" if viol_rate == 0.0 else "FALHA"
    print(
        f"=== HydroFiLM-ENSO RS (Treino <= 2022 | Cal 2023 | Audit 2024 | h=2 sem.) ===\n"
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
        f"Auditoria Autograd Monotonicidade (100% split fp32): {mono_status}\n"
        f"Tempo total ({spec.backend}): {time.perf_counter() - t0:.1f}s"
    )
    del trainer, module
    release_gpu_memory()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
