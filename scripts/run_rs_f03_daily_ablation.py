"""Ablação controlada Onda 2B: F03.1 HydroFiLM-ENSO Diário (60d) vs. Semanal (8 sem.) na GPU.

Executa de ponta a ponta sobre o RS (<= 2024; 2025 e 2026 bloqueados):
1. Valida clientes Onda 2B ao vivo (`MergeCptecClient` amostra 2024-05-01 GRIB2 +
   `AnaHidrowebClient` com fallback automático Zenodo `10.5281/zenodo.16929471`).
2. Constrói série diária 0.1° ERA5-Land + V10M (`~164k` linhas diárias, 30 regiões, 0 colisões)
   projetando features lentas (`MapBiomas <= 2022`, `NOAA ENSO` causal) sem vazamento futuro.
3. Treina `HydroFiLM-ENSO` diário (`seq_len=60` dias, `13` folds x `3` seeds, encerrado no sábado
   ISO) na GPU ROCm/CUDA e compara objetivamente contra o OOF semanal (`1.2.2`) na catraca.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from previsao_leitos.audit import brier_skill_score, expected_calibration_error
from previsao_leitos.geo import (
    NasaPowerHydroAdapter,
    OpenMeteoEra5Adapter,
    RegionCentroid,
    build_causal_enso_weekly_features,
    build_complete_daily_grid,
    compute_calendar_hydro_features,
)
from previsao_leitos.hydrology import (
    generate_neural_hydro_oof_predictions,
    resolve_accelerator,
)
from previsao_leitos.hydrology.device import DeviceRequest
from previsao_leitos.ingestion import (
    AnaHidrowebClient,
    MergeCptecClient,
    fetch_noaa_enso_monthly,
)

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")

DAILY_DYN_COLS = (
    "precip_dia_mm",
    "api_dia",
    "temp_media_c",
    "umidade_solo_m3m3",
    "vento_sul_ms",
    "day_sin",
    "day_cos",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Roda validação Onda 2B e ablação diária (60d) vs. semanal F03.1 em GPU."
    )
    parser.add_argument("--device", choices=("auto", "cpu", "rocm", "cuda"), default="auto")
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", type=str, default="42,137,2024")
    parser.add_argument("--preload-vram", action="store_true")
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    req_dev: DeviceRequest = args.device
    spec = resolve_accelerator(prefer_gpu=True, requested_device=req_dev)
    print(
        f"[Onda 2B / F03.1 Diário] Acelerador: {spec.accelerator} ({spec.backend}) | "
        f"GPU={spec.device_name} | seq_len={args.seq_len} dias"
    )

    # 1. Validação ao vivo Onda 2B: MERGE-CPTEC amostra 2024-05-01 + ANA API v2 (fallback Zenodo)
    wave2b_dir = DATA_RAW / "wave2b"
    try:
        merge_client = MergeCptecClient(cache_dir=wave2b_dir)
        m_sample = merge_client.fetch_daily_grib2(date(2024, 5, 1))
        print(
            f"[Onda 2B] MERGE-CPTEC 0.1° validado: {m_sample.cache_path.name} "
            f"({m_sample.size_bytes:,} B, cache_hit={m_sample.cache_hit}, "
            f"sha256={m_sample.sha256[:12]})"
        )
    except Exception as exc:
        print(f"[Onda 2B] MERGE-CPTEC offline/indisponível ({exc}); seguindo com ERA5-Land 0.1°.")

    ana_client = AnaHidrowebClient(cache_dir=DATA_RAW / "wave2a")
    ana_res = ana_client.fetch_or_fallback(
        station_codes=("87450004", "86510000"), bacia_id="guaiba", allow_zenodo_fallback=True
    )
    print(
        f"[Onda 2B] ANA HidroWeb API v2 status: source={ana_res.source} "
        f"(credenciais={ana_client.has_credentials})"
    )

    # 2. Carrega grade diária 0.1° ERA5-Land + V10M NASA (<= 2024) + alvos semanais v3
    panel_v3 = pl.read_parquet(DATA_PROCESSED / "rs_painel_semanal_regioes_v3.parquet").filter(
        pl.col("ano_epi") <= 2024
    )
    centroids = pl.read_parquet(DATA_RAW / "rs_regioes_saude_centroides_v1.parquet")
    centroid_objs = tuple(
        RegionCentroid(
            entity_id=str(r["id_regiao_saude"]),
            latitude=float(r["latitude"]),
            longitude=float(r["longitude"]),
        )
        for r in centroids.iter_rows(named=True)
    )
    era5_daily = OpenMeteoEra5Adapter(cache_dir=DATA_RAW, batch_size=10).fetch_daily_hydro(
        centroids=centroid_objs, start_date=date(2010, 1, 1), end_date=date(2025, 12, 31)
    )
    nasa_daily = NasaPowerHydroAdapter(cache_dir=DATA_RAW, max_workers=4).fetch_daily_hydro(
        centroids=centroid_objs, start_date=date(2010, 1, 1), end_date=date(2025, 12, 31)
    )
    daily_raw = (
        era5_daily.join(
            nasa_daily.select(["entity_id", "data", "vento_sul_ms"]),
            on=["entity_id", "data"],
            how="left",
        )
        .with_columns(pl.col("vento_sul_ms").fill_null(0.0))
        .filter(pl.col("data") <= date(2024, 12, 31))
    )
    daily_grid = build_complete_daily_grid(
        daily_raw,
        entity_col="entity_id",
        date_col="data",
        value_cols=("precip_dia_mm", "temp_media_c", "umidade_solo_m3m3", "vento_sul_ms"),
    )
    daily_feats = (
        compute_calendar_hydro_features(
            daily_grid,
            entity_col="entity_id",
            date_col="data",
            precip_col="precip_dia_mm",
            k_decay=0.88,
        )
        .rename({"entity_id": "id_regiao_saude", "api_indice_saturacao": "api_dia"})
        .join(
            centroids.select(["id_regiao_saude", "nome_regiao_saude"]),
            on="id_regiao_saude",
            how="left",
        )
        .with_columns(
            pl.col("data").dt.iso_year().cast(pl.Int32).alias("ano_epi"),
            pl.col("data").dt.week().cast(pl.Int8).alias("semana_epi"),
            (pl.col("data").dt.weekday() == 6).alias("is_week_end"),  # sábado ISO (1=Seg..7=Dom)
            (2.0 * np.pi * pl.col("data").dt.ordinal_day().cast(pl.Float64) / 365.25)
            .sin()
            .alias("day_sin"),
            (2.0 * np.pi * pl.col("data").dt.ordinal_day().cast(pl.Float64) / 365.25)
            .cos()
            .alias("day_cos"),
        )
    )

    # Alvos físicos ex-ante (t_0 + 2 semanas) idênticos ao benchmark semanal para comparação justa
    train_raw = panel_v3.filter(pl.col("ano_epi") <= 2022)
    ctx_stats = train_raw.group_by("nome_regiao_saude").agg(
        pl.col("chuva_sem_mm").mean().alias("ctx_rain_mean"),
        pl.col("chuva_sem_mm").std().fill_null(1.0).alias("ctx_rain_std"),
    )
    def _z_stats(col: str) -> tuple[float, float]:
        mu = float(train_raw.select(pl.col(col).mean()).item())
        sd = float(train_raw.select(pl.col(col).std()).item() or 1.0)
        return mu, sd if sd > 1e-8 else 1.0

    m_api, s_api = _z_stats("api_max_sem")
    m_up, s_up = _z_stats("api_montante_pond")
    m_rem, s_rem = _z_stats("remanso_lagunar_idx")
    wk_targets = (
        panel_v3.sort(["nome_regiao_saude", "ano_epi", "semana_epi"])
        .with_columns(
            (
                (pl.col("api_max_sem") - m_api) / s_api
                + (pl.col("api_montante_pond") - m_up) / s_up
                + 0.5 * (pl.col("remanso_lagunar_idx") - m_rem) / s_rem
            ).alias("_hydro_sev_t")
        )
    )
    sev_p90 = float(
        wk_targets.filter(pl.col("ano_epi") <= 2022)
        .select(pl.col("_hydro_sev_t").quantile(0.90))
        .item()
    )
    wk_targets = (
        wk_targets.with_columns(
            pl.col("_hydro_sev_t").shift(-2).over("nome_regiao_saude").alias("y_sev_t2"),
        )
        .with_columns(
            (pl.col("y_sev_t2") >= sev_p90).cast(pl.Float32).alias("y_event_t2")
        )
        .select(
            [
                "nome_regiao_saude",
                "ano_epi",
                "semana_epi",
                "chuva_montante_multihop",
                "frac_urbano",
                "frac_banhado_agua",
                "frac_area_polder_regiao",
                "indice_aterro_varzea",
                "frac_floresta",
                "y_event_t2",
                "y_sev_t2",
            ]
        )
    )

    noaa_monthly, _ = fetch_noaa_enso_monthly(
        cache_dir=DATA_RAW / "wave2a", include_mei_v2=True, max_year=2024
    )
    enso_wk = build_causal_enso_weekly_features(noaa_monthly, lags=(4, 12)).select(
        ["ano_epi", "semana_epi", "el_nino_intensity", "oni_c_lag4"]
    )

    daily_model_df = (
        daily_feats.join(ctx_stats, on="nome_regiao_saude", how="left")
        .join(enso_wk, on=["ano_epi", "semana_epi"], how="left")
        .join(wk_targets, on=["nome_regiao_saude", "ano_epi", "semana_epi"], how="left")
        .with_columns(
            pl.col("el_nino_intensity").fill_null(0.0),
            pl.col("oni_c_lag4").fill_null(0.0),
        )
        .drop_nulls(subset=["y_event_t2", "y_sev_t2"])
        .filter((pl.col("ano_epi") >= 2010) & (pl.col("ano_epi") <= 2024))
    )
    print(
        f"[F03.1 Diário] Painel diário pronto: {daily_model_df.height:,} dias | "
        f"{daily_model_df.filter(pl.col('is_week_end')).height:,} sábados ISO (t_0)"
    )

    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    oof_daily, rep_daily = generate_neural_hydro_oof_predictions(
        daily_model_df,
        entity_col="nome_regiao_saude",
        dyn_cols=DAILY_DYN_COLS,
        ctx_cols=CTX_COLS,
        enso_cols=ENSO_COLS,
        mono_pos_cols=MONO_POS_COLS,
        mono_neg_cols=MONO_NEG_COLS,
        event_col="y_event_t2",
        severity_col="y_sev_t2",
        train_end_year=2022,
        cal_year=2023,
        seq_len=args.seq_len,
        freq="daily",
        date_col="data",
        sample_mask_col="is_week_end",
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        seeds=seeds,
        requested_device=req_dev,
        preload_vram=args.preload_vram,
        rain_col_for_corr="chuva_montante_multihop",
        log_progress=True,
    )

    daily_out_path = DATA_PROCESSED / f"rs_f03_hydro_oof_daily_{args.seq_len}d_v3.parquet"
    oof_daily_sel = oof_daily.select(
        [
            "nome_regiao_saude",
            "ano_epi",
            "semana_epi",
            "prob_enchente_f03_oof",
            "severidade_hidro_f03_oof",
        ]
    )
    tmp_p = daily_out_path.with_suffix(".parquet.tmp")
    oof_daily_sel.write_parquet(tmp_p)
    tmp_p.replace(daily_out_path)

    ev24_d = oof_daily.filter(pl.col("ano_epi") == 2024)
    y24_d = ev24_d["y_event_t2"].to_numpy()
    p24_d = ev24_d["prob_enchente_f03_oof"].to_numpy()
    roc_d = float(roc_auc_score(y24_d, p24_d))
    pr_d = float(average_precision_score(y24_d, p24_d))
    ece_d = float(expected_calibration_error(y24_d, p24_d))
    bss_d = float(brier_skill_score(y24_d, p24_d))

    # Compara diretamente com OOF semanal (recém-recalculado ou salvo em rs_f03_hydro_oof_v3)
    weekly_path = DATA_PROCESSED / "rs_f03_hydro_oof_v3.parquet"
    weekly_metrics: dict[str, float] = {}
    promote_daily = False
    if weekly_path.exists():
        wk_oof = pl.read_parquet(weekly_path).join(
            wk_targets.select(["nome_regiao_saude", "ano_epi", "semana_epi", "y_event_t2"]),
            on=["nome_regiao_saude", "ano_epi", "semana_epi"],
            how="inner",
        ).filter(pl.col("ano_epi") == 2024)
        y24_w = wk_oof["y_event_t2"].to_numpy()
        p24_w = wk_oof["prob_enchente_f03_oof"].to_numpy()
        roc_w = float(roc_auc_score(y24_w, p24_w))
        ece_w = float(expected_calibration_error(y24_w, p24_w))
        bss_w = float(brier_skill_score(y24_w, p24_w))
        weekly_metrics = {"roc_auc": roc_w, "ece": ece_w, "bss": bss_w}
        # Critério de catraca: autograd == 0.00% e ganho em BSS ou ROC-AUC sem piorar ECE > 0.01
        promote_daily = (
            rep_daily.autograd_violation_rate == 0.0
            and (bss_d > bss_w + 0.005 or roc_d > roc_w + 0.01)
            and ece_d <= ece_w + 0.015
        )
        if promote_daily:
            tmp_w = weekly_path.with_suffix(".parquet.tmp")
            oof_daily_sel.write_parquet(tmp_w)
            tmp_w.replace(weekly_path)

    ablation_summary = {
        "daily_seq_len": args.seq_len,
        "daily_report": rep_daily.__dict__,
        "daily_2024": {"roc_auc": roc_d, "pr_auc": pr_d, "ece": ece_d, "bss": bss_d},
        "weekly_2024": weekly_metrics,
        "promoted_daily_over_weekly": promote_daily,
        "total_elapsed_s": round(time.perf_counter() - t0, 2),
    }
    ablation_json = DATA_PROCESSED / f"rs_f03_ablation_daily_{args.seq_len}d_vs_weekly.json"
    ablation_json.write_text(json.dumps(ablation_summary, indent=2), encoding="utf-8")

    print(
        f"=== ABLAÇÃO F03.1 DIÁRIO ({args.seq_len}d) vs. SEMANAL (RS 2024, RX 7600) ===\n"
        f"Diário ({args.seq_len}d): Tempo={rep_daily.elapsed_seconds:.1f}s | "
        f"Pico VRAM={rep_daily.peak_vram_mb:.1f} MB | "
        f"Autograd fp32={rep_daily.autograd_violation_rate * 100:.4f}% | "
        f"corr={rep_daily.corr_rain_severity:.4f}\n"
        f"Diário 2024  | ROC-AUC={roc_d:.4f} | PR-AUC={pr_d:.4f} | "
        f"ECE={ece_d:.4f} | BSS={bss_d:+.4f}\n"
        f"Semanal 2024 | {weekly_metrics}\n"
        f"Catraca F03.1 Diário -> {'PROMOTE' if promote_daily else 'KEEP_WEEKLY_CHAMPION'}\n"
        f"Salvo em: {daily_out_path} e {ablation_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
