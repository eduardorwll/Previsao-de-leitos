"""Ponto de entrada CLI único (`previsao-leitos`) com composição fina e guardrails."""

from __future__ import annotations

import argparse
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from previsao_leitos.audit import (
    DatasetAuditEntry,
    NoFloodSyntheticControl,
    brier_skill_score,
    build_verification_report,
    decision_curve_net_benefit,
    decompose_intermunicipal_and_landfill_effect,
    estimate_surge_probability_from_quantiles,
    evaluate_counterfactual_accuracy,
    evaluate_probabilistic_predictions,
    expected_calibration_error,
)
from previsao_leitos.cli.logging_json import emit_json_log
from previsao_leitos.demand import (
    ConformalFloodCalibrator,
    QuantileGBDTForecaster,
    SpecialtyLosEstimator,
    prepare_global_demand_splits,
)
from previsao_leitos.domain import LeakageViolationError
from previsao_leitos.geo import SplitPolicy
from previsao_leitos.readiness import HospitalReadinessConfig, simulate_counterfactual_readiness

_DONOR_REGIONS = (
    "Região 11 - Sete Povos das Missões",
    "Região 15 - Caminho das Águas",
    "Região 16 - Alto Uruguai Gaúcho",
    "Região 17 - Planalto",
    "Região 18 - Araucárias",
)
_FOCUS_REGIONS = (
    "Região 10 - Capital e Vale do Gravataí",
    "Região 21 - Sul",
    "Região 08 - Vale do Caí e Metropolitana",
    "Região 29 - Vales e Montanhas",
)
_DYNAMIC_FEATURES = (
    "chuva_sem_mm",
    "chuva_montante_pond",
    "api_max_sem",
    "api_montante_pond",
    "remanso_lagunar_idx",
    "exposicao_aterro_ativa",
    "chuva_catchment_mm",
    "api_catchment_max",
    "umidade_solo_sem",
    "temp_media_sem_c",
)
_STATIC_FEATURES = (
    "frac_urbano",
    "frac_banhado_agua",
    "frac_floresta",
    "frac_area_polder_regiao",
    "indice_aterro_varzea",
)
_MONOTONE_POS = (
    "chuva_montante_pond",
    "api_montante_pond",
    "remanso_lagunar_idx",
    "exposicao_aterro_ativa",
    "indice_aterro_varzea",
    "frac_area_polder_regiao",
)


def _cmd_doctor() -> int:
    raw_dir = Path("data/raw")
    proc_panel = Path("data/processed/rs_painel_semanal_regioes_v2.parquet")
    raw_parquets = len(list(raw_dir.glob("*.parquet"))) if raw_dir.exists() else 0
    emit_json_log(
        run_id="doctor",
        entry_point="cli:doctor",
        event="doctor_complete",
        payload={
            "python": platform.python_version(),
            "raw_parquets_count": raw_parquets,
            "rs_weekly_panel_v2_ready": proc_panel.exists(),
            "clinical_boundary": "Not for patient care or live clinical use",
        },
    )
    return 0


def _cmd_verify_rs(args: argparse.Namespace) -> int:
    policy = SplitPolicy(train_end_year=2022, cal_year=2023, dev_audit_year=2024, holdout_year=2025)
    eval_year = int(args.eval_year)
    allow_holdout = bool(args.allow_holdout)
    run_id = str(args.run_id or f"rs_{eval_year}_{args.target}")

    if eval_year >= policy.holdout_year and not allow_holdout:
        print(
            f"[BLOQUEADO] Acesso ao holdout intocado ({policy.holdout_year}) exige "
            f"--allow-holdout explicitamente na auditoria final.",
            file=sys.stderr,
        )
        return 2

    panel_path = Path(args.panel_path)
    if not panel_path.exists():
        print(f"[ERRO] Painel não encontrado: {panel_path}", file=sys.stderr)
        return 1

    panel = pl.read_parquet(panel_path)
    target_col = str(args.target)
    seed = int(args.seed)

    emit_json_log(
        run_id=run_id,
        entry_point="cli:verify-rs",
        event="verification_started",
        payload={"eval_year": eval_year, "target": target_col, "seed": seed},
    )

    try:
        sp_cal = prepare_global_demand_splits(
            panel,
            entity_col="nome_regiao_saude",
            target_col=target_col,
            dynamic_feature_cols=_DYNAMIC_FEATURES,
            static_feature_cols=_STATIC_FEATURES,
            horizon_weeks=2,
            lag_weeks=(0, 1, 2, 3, 4),
            eval_year=policy.cal_year,
            policy=policy,
            allow_holdout=False,
        )
        sp_eval = prepare_global_demand_splits(
            panel,
            entity_col="nome_regiao_saude",
            target_col=target_col,
            dynamic_feature_cols=_DYNAMIC_FEATURES,
            static_feature_cols=_STATIC_FEATURES,
            horizon_weeks=2,
            lag_weeks=(0, 1, 2, 3, 4),
            eval_year=eval_year,
            policy=policy,
            allow_holdout=allow_holdout,
        )
    except LeakageViolationError as exc:
        print(f"[BLOQUEADO] {exc}", file=sys.stderr)
        return 2

    base = QuantileGBDTForecaster(
        monotone_positive_prefixes=_MONOTONE_POS,
        monotone_negative_prefixes=("frac_floresta",),
        max_iter=140,
        max_depth=5,
        learning_rate=0.05,
        random_state=seed,
    ).fit(sp_cal.train_df, sp_cal.feature_cols)

    cal_model = ConformalFloodCalibrator(
        base_model=base,
        stress_cols=("api_max_sem_lag0", "api_montante_pond_lag0", "remanso_lagunar_idx_lag0"),
        stress_gamma=0.65,
    ).fit_calibration(train_df=sp_cal.train_df, cal_df=sp_cal.eval_df)

    raw_preds = base.predict_quantiles(sp_eval.eval_df)
    cal_preds = cal_model.predict_quantiles(sp_eval.eval_df)
    bench_df = pl.DataFrame(
        [
            evaluate_probabilistic_predictions(
                sp_eval.eval_df, raw_preds, model_name=base.model_name, n_boot=200, seed=seed
            ),
            evaluate_probabilistic_predictions(
                sp_eval.eval_df, cal_preds, model_name=cal_model.model_name, n_boot=200, seed=seed
            ),
        ]
    )

    # Net Benefit / ECE / BSS sobre evento de sobrecarga (> 110% da média causal de 12 semanas)
    surge_thr = 1.10 * sp_eval.eval_df["y_roll12"].to_numpy().astype(np.float64)
    y_surge = (sp_eval.eval_df["target_y"].to_numpy().astype(np.float64) >= surge_thr).astype(
        np.float64
    )
    p_surge = estimate_surge_probability_from_quantiles(cal_preds, surge_thr)
    ece_val = expected_calibration_error(y_surge, p_surge, n_bins=8)
    bss_val = brier_skill_score(y_surge, p_surge)
    dca_df = decision_curve_net_benefit(y_surge, p_surge)

    sc = NoFloodSyntheticControl(donor_entities=_DONOR_REGIONS, train_end_year=2022).fit(
        panel, entity_col="nome_regiao_saude", target_col=target_col
    )
    cf_eval = sc.predict_no_flood(
        sp_eval.eval_df, full_panel=panel, entity_col="nome_regiao_saude", target_col=target_col
    )
    cf_rep = evaluate_counterfactual_accuracy(
        model=cal_model,
        train_df=sp_eval.train_df,
        eval_df=sp_eval.eval_df,
        no_flood_df=cf_eval,
        entity_col="nome_regiao_saude",
        focus_entities=_FOCUS_REGIONS,
        shock_weeks=(18, 28),
        placebo_weeks=(4, 15),
        hydro_feature_prefixes=_DYNAMIC_FEATURES,
        monotone_check_cols=("chuva_montante_pond_lag0", "remanso_lagunar_idx_lag0"),
    )
    poa_sul = sp_eval.eval_df.filter(
        pl.col("nome_regiao_saude").is_in(
            ["Região 10 - Capital e Vale do Gravataí", "Região 21 - Sul"]
        )
        & pl.col("target_semana_epi").is_between(19, 24)
    )
    decomp = decompose_intermunicipal_and_landfill_effect(
        model=cal_model,
        train_df=sp_eval.train_df,
        eval_df=poa_sul,
        feature_groups={
            "chuva_local": ("chuva_sem_mm", "api_max_sem", "umidade_solo_sem"),
            "escoamento_montante": ("chuva_montante_pond", "api_montante_pond"),
            "remanso_e_aterro": ("remanso_lagunar_idx", "exposicao_aterro_ativa"),
            "transbordamento_w_ref": ("chuva_catchment_mm", "api_catchment_max"),
        },
    )

    sim_comp = None
    if target_col == "uti_adulto_diarias":
        sih_matches = sorted(Path("data/raw").glob("sih_demanda_semanal_agg_v3_*.parquet"))
        icu_los = 6.46
        if sih_matches:
            los = SpecialtyLosEstimator.fit_causal(
                pl.read_parquet(sih_matches[-1]), train_end_year=2022
            )
            icu_los = los.mean_icu_days("uti_adulto")
        poa_sched = (
            sp_eval.eval_df.with_columns(cal_preds["pred_p90"])
            .filter(
                (pl.col("nome_regiao_saude") == "Região 10 - Capital e Vale do Gravataí")
                & (pl.col("target_semana_epi").is_between(10, 35))
            )
            .sort("target_semana_epi")
            .with_columns(
                (pl.col("target_y") / icu_los).alias("chegadas_semanais"),
                (pl.col("pred_p90") / icu_los).alias("previsao_p90_semanal"),
            )
        )
        if not poa_sched.is_empty():
            sim_comp = simulate_counterfactual_readiness(
                poa_sched,
                HospitalReadinessConfig(
                    region_label="Região 10 - Capital/POA (431 UTI SUS + Pôlder 15%)",
                    base_beds=431,
                    surge_beds=90,
                    los_days=icu_los,
                    elective_share=0.22,
                    surge_lead_time_days=3,
                    polder_hospital_loss_frac=0.15,
                ),
                seed=seed,
            )

    artifacts = build_verification_report(
        output_dir=Path(args.output_dir),
        run_id=run_id,
        eval_year=eval_year,
        target_label=f"{target_col} (RS {eval_year})",
        benchmark_table=bench_df,
        counterfactual_report=cf_rep,
        decomposition_df=decomp,
        datasets=(
            DatasetAuditEntry(
                "Painel Semanal RS v2 (SIH+CNES+NASA POWER+MapBiomas+W_ref)",
                "2010-2025",
                panel.height,
                0,
                True,
                f"ECE_surge={ece_val:.3f} | BSS_surge={bss_val:+.3f}",
            ),
        ),
        policy=policy,
        allow_holdout=allow_holdout,
        seed=seed,
        simpy_comparison=sim_comp,
        net_benefit_df=dca_df,
    )
    emit_json_log(
        run_id=run_id,
        entry_point="cli:verify-rs",
        event="verification_complete",
        payload={
            "report_path": str(artifacts.report_path),
            "n_figures": len(artifacts.figure_paths),
            "ece_surge": round(ece_val, 4),
            "bss_surge": round(bss_val, 4),
            "monotonic_violation_rate": cf_rep.monotonic_violation_rate,
        },
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Constrói o parser da CLI `previsao-leitos`."""
    parser = argparse.ArgumentParser(
        prog="previsao-leitos",
        description="CLI de pesquisa aberta para previsão ex-ante de leitos sob enchentes.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Verifica ambiente e prontidão das bases locais em JSON.")

    p_ver = sub.add_parser(
        "verify-rs",
        help="Executa verificação completa RS (F04+F05+F06) gerando relatório e Fig. C1-C5.",
    )
    p_ver.add_argument("--eval-year", type=int, default=2024)
    p_ver.add_argument("--target", type=str, default="uti_adulto_diarias")
    p_ver.add_argument(
        "--panel-path",
        type=str,
        default="data/processed/rs_painel_semanal_regioes_v2.parquet",
    )
    p_ver.add_argument("--output-dir", type=str, default="reports/cli_rs_verify")
    p_ver.add_argument("--run-id", type=str, default="")
    p_ver.add_argument("--seed", type=int, default=42)
    p_ver.add_argument("--allow-holdout", action="store_true", default=False)
    return parser


def cli_main(argv: Sequence[str] | None = None) -> int:
    """Ponto de entrada principal da CLI."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "doctor":
        return _cmd_doctor()
    if args.command == "verify-rs":
        return _cmd_verify_rs(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(cli_main())
