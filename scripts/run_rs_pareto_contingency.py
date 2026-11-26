"""Otimização Pareto NSGA-II (pymoo + SimPy CRN 2 Estágios) para POA e Sul em RS 2024 (Task 1.2.5).

Executa sobre dados reais de `rs_painel_semanal_regioes_v3.parquet` (`RS <= 2024`, `2025/2026`
bloqueados fisicamente) usando a previsão conformal `V1` (`previsao_p90_semanal`) e capacidade
CNES real de UTI Adulto SUS, otimizando simultaneamente:
- Pico de fila de espera (`f1`),
- Eletivas adiadas (`f2`),
- Leitos-dia de surge mobilizados (`f3`),
com Estágio 1 (`R_search = 6` CRN) multi-core e Estágio 2 (`R_audit = 30` CRN independentes).

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import polars as pl

from previsao_leitos.demand import (
    ConformalFloodCalibrator,
    QuantileGBDTForecaster,
    prepare_global_demand_splits,
)
from previsao_leitos.geo import (
    RS_BASIN_WEIGHTED_EDGES,
    BasinTopology,
    SplitPolicy,
    build_causal_enso_weekly_features,
    map_rs_health_region_to_basin,
)
from previsao_leitos.ingestion import fetch_noaa_enso_monthly
from previsao_leitos.readiness import (
    HospitalReadinessConfig,
    optimize_contingency_pareto_front,
)

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")

TARGET_REGIONS: tuple[tuple[str, float, float], ...] = (
    # (nome_regiao_saude, polder_loss_frac, los_days)
    ("Região 10 - Capital e Vale do Gravataí", 0.15, 6.5),
    ("Região 21 - Sul", 0.10, 6.2),
)


def _prepare_v1_eval_2024() -> pl.DataFrame:
    policy = SplitPolicy(train_end_year=2022, cal_year=2023, dev_audit_year=2024, holdout_year=2025)
    panel = pl.read_parquet(DATA_PROCESSED / "rs_painel_semanal_regioes_v3.parquet").filter(
        pl.col("ano_epi") < policy.holdout_year
    )
    topo = BasinTopology.from_weighted_edges(RS_BASIN_WEIGHTED_EDGES)
    if "bacia_id" not in panel.columns:
        panel = panel.with_columns(
            pl.col("nome_regiao_saude")
            .map_elements(map_rs_health_region_to_basin, return_dtype=pl.String)
            .alias("bacia_id")
        )
    if "chuva_montante_multihop" not in panel.columns:
        panel = topo.propagate_weighted_upstream(
            panel,
            basin_col="bacia_id",
            value_col="chuva_sem_mm",
            output_col="chuva_montante_multihop",
            multi_hop=True,
        )

    noaa_monthly, _ = fetch_noaa_enso_monthly(
        cache_dir=DATA_RAW / "wave2a", include_mei_v2=True, max_year=2024
    )
    enso_wk = build_causal_enso_weekly_features(noaa_monthly, lags=(4, 12))
    enso_cols = [c for c in enso_wk.columns if c not in ("ano_epi", "semana_epi")]
    panel = panel.drop([c for c in enso_cols if c in panel.columns]).join(
        enso_wk, on=["ano_epi", "semana_epi"], how="left"
    )
    panel = panel.with_columns(
        pl.col("oni_c").fill_null(0.0),
        pl.col("el_nino_intensity").fill_null(0.0),
    )

    oof_path = DATA_PROCESSED / "rs_f03_hydro_oof_v3.parquet"
    if oof_path.exists():
        oof_df = pl.read_parquet(oof_path)
        panel = panel.drop(
            [c for c in ("prob_enchente_f03_oof", "severidade_hidro_f03_oof") if c in panel.columns]
        ).join(oof_df, on=["nome_regiao_saude", "ano_epi", "semana_epi"], how="left")
        panel = panel.with_columns(
            pl.col("prob_enchente_f03_oof").fill_null(0.0),
            pl.col("severidade_hidro_f03_oof").fill_null(0.0),
        )

    dyn_cols = [
        c
        for c in (
            "chuva_sem_mm",
            "chuva_montante_multihop",
            "api_max_sem",
            "api_montante_pond",
            "remanso_lagunar_idx",
            "prob_enchente_f03_oof",
            "severidade_hidro_f03_oof",
            "el_nino_intensity",
        )
        if c in panel.columns
    ]
    sp_cal = prepare_global_demand_splits(
        panel,
        entity_col="nome_regiao_saude",
        target_col="uti_adulto_diarias",
        dynamic_feature_cols=dyn_cols,
        horizon_weeks=2,
        lag_weeks=(0, 1, 2),
        eval_year=2023,
        policy=policy,
    )
    sp_eval = prepare_global_demand_splits(
        panel,
        entity_col="nome_regiao_saude",
        target_col="uti_adulto_diarias",
        dynamic_feature_cols=dyn_cols,
        horizon_weeks=2,
        lag_weeks=(0, 1, 2),
        eval_year=2024,
        policy=policy,
    )
    mono_pos = (
        "chuva_montante_multihop",
        "chuva_sem_mm",
        "api_max_sem",
        "prob_enchente_f03",
        "severidade_hidro_f03",
        "remanso_lagunar_idx",
        "el_nino_intensity",
    )
    v1 = QuantileGBDTForecaster(
        monotone_positive_prefixes=mono_pos, max_iter=120, max_depth=5, random_state=42
    ).fit(sp_eval.train_df, sp_eval.feature_cols)
    v1_cal = ConformalFloodCalibrator(base_model=v1).fit_calibration(
        train_df=sp_eval.train_df, cal_df=sp_cal.eval_df
    )
    preds = v1_cal.predict_quantiles(sp_eval.eval_df)
    return sp_eval.eval_df.with_columns(
        preds["pred_p50"].alias("pred_p50"),
        preds["pred_p90"].alias("pred_p90"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Roda otimização Pareto NSGA-II no RS 2024.")
    parser.add_argument("--r-search", type=int, default=6)
    parser.add_argument("--r-audit", type=int, default=30)
    parser.add_argument("--pop-size", type=int, default=24)
    parser.add_argument("--n-gen", type=int, default=12)
    parser.add_argument("--workers", type=int, default=min(8, max(1, os.cpu_count() or 1)))
    args = parser.parse_args(argv)

    eval_2024 = _prepare_v1_eval_2024()
    panel_2024 = pl.read_parquet(DATA_PROCESSED / "rs_painel_semanal_regioes_v3.parquet").filter(
        pl.col("ano_epi") == 2024
    )

    out_summary: dict[str, object] = {
        "task": "1.2.5",
        "eval_year": 2024,
        "r_search": args.r_search,
        "r_audit": args.r_audit,
        "pop_size": args.pop_size,
        "n_gen": args.n_gen,
        "workers": args.workers,
        "regions": [],
    }
    region_entries: list[dict[str, object]] = []

    for reg_name, polder_loss, los_days in TARGET_REGIONS:
        sub = eval_2024.filter(
            (pl.col("nome_regiao_saude") == reg_name)
            & pl.col("target_semana_epi").is_between(12, 37)
        ).sort("target_semana_epi")
        cnes_sub = panel_2024.filter(
            (pl.col("nome_regiao_saude") == reg_name) & pl.col("cnes_uti_sus_leitos").is_not_null()
        )
        base_beds = (
            round(float(np.median(cnes_sub["cnes_uti_sus_leitos"].cast(pl.Float64).to_numpy())))
            if not cnes_sub.is_empty()
            else 100
        )
        # Escala diárias semanais -> chegadas semanais via Lei de Little (diarias / los_days)
        arr_weekly = np.maximum(1.0, sub["target_y"].to_numpy().astype(np.float64) / los_days)
        p90_weekly = np.maximum(1.0, sub["pred_p90"].to_numpy().astype(np.float64) / los_days)
        # Indicador hidrológico exógeno (chuva/API acima de p85 na região durante 2024)
        api_vals = sub["api_max_sem_lag0"].to_numpy().astype(np.float64)
        api_thr = float(np.quantile(api_vals, 0.80))
        flood_flag = (api_vals >= api_thr).astype(np.int8)

        sched = pl.DataFrame(
            {
                "chegadas_semanais": arr_weekly,
                "previsao_p90_semanal": p90_weekly,
                "em_cheia_hidro": flood_flag,
            }
        )
        base_cfg = HospitalReadinessConfig(
            base_beds=base_beds,
            surge_beds=max(10, round(0.20 * base_beds)),
            los_days=los_days,
            elective_share=0.25,
            elective_deferral_frac=0.75,
            surge_lead_time_days=2,
            polder_hospital_loss_frac=polder_loss,
            region_label=reg_name,
        )

        pareto_res = optimize_contingency_pareto_front(
            sched,
            base_cfg,
            r_search=args.r_search,
            r_audit=args.r_audit,
            pop_size=args.pop_size,
            n_gen=args.n_gen,
            n_workers=args.workers,
            seed=42,
        )
        print(f"\n=== FRONTEIRA DE PARETO NSGA-II: {reg_name} (base={base_beds} leitos) ===")
        print(
            f"Tempo: {pareto_res.elapsed_seconds}s | Fronteira: {len(pareto_res.pareto_front)} "
            f"soluções | Dominam V1: {pareto_res.dominating_v1_count}"
        )
        print(pareto_res.front_df.head(10))
        rec = pareto_res.recommended_policy
        print(
            f"Recomendada ({rec.label}): trig={rec.decision.trigger_occ_threshold:.3f} "
            f"close={rec.decision.hysteresis_close_threshold:.3f} "
            f"defer={rec.decision.elective_deferral_frac:.3f} "
            f"surge={rec.decision.surge_beds} | fila={rec.mean_peak_queue} "
            f"({rec.queue_reduction_vs_reactive_pct:+.1f}% vs H0) | "
            f"elet_vs_V1={rec.elective_reduction_vs_v1_pct:+.1f}% | "
            f"surge_vs_V1={rec.surge_cost_reduction_vs_v1_pct:+.1f}% | "
            f"domina_V1={rec.dominates_v1}"
        )

        region_entries.append(
            {
                "region": reg_name,
                "base_beds": base_beds,
                "elapsed_seconds": pareto_res.elapsed_seconds,
                "n_pareto_solutions": len(pareto_res.pareto_front),
                "dominating_v1_count": pareto_res.dominating_v1_count,
                "reactive_h0": pareto_res.reactive_baseline.__dict__
                | {"decision": pareto_res.reactive_baseline.decision.__dict__},
                "v1_fixed": pareto_res.v1_fixed_baseline.__dict__
                | {"decision": pareto_res.v1_fixed_baseline.decision.__dict__},
                "recommended": rec.__dict__ | {"decision": rec.decision.__dict__},
                "front_table": pareto_res.front_df.to_dicts(),
            }
        )

    out_summary["regions"] = region_entries
    out_path = DATA_PROCESSED / "rs_v1_2_5_pareto_contingency.json"
    out_path.write_text(json.dumps(out_summary, indent=2), encoding="utf-8")
    print(f"\nSalvo em: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
