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
    check_g1_collinearity_vif,
    check_g4_right_censoring,
    check_g8_degenerate_surge_labels,
    compile_execution_dossier,
    compile_hygiene_scorecard,
    compute_surge_flags,
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

    # Net Benefit / ECE / BSS sobre evento de sobrecarga (> 110% da média causal, sem 0.0 >= 0.0)
    y_true_arr = sp_eval.eval_df["target_y"].to_numpy().astype(np.float64)
    y_roll12_arr = sp_eval.eval_df["y_roll12"].to_numpy().astype(np.float64)
    surge_thr = 1.10 * np.maximum(y_roll12_arr, 1e-6)
    y_surge = compute_surge_flags(y_true_arr, y_roll12_arr, multiplier=1.10)
    p_surge = estimate_surge_probability_from_quantiles(cal_preds, surge_thr)
    ece_val = expected_calibration_error(y_surge, p_surge, n_bins=8)
    bss_val = brier_skill_score(y_surge, p_surge)
    dca_df = decision_curve_net_benefit(y_surge, p_surge)

    hygiene_gates = [check_g8_degenerate_surge_labels(y_true_arr, y_roll12_arr, y_surge)]
    if "chuva_sem_mm" in panel.columns and "chuva_catchment_mm" in panel.columns:
        hygiene_gates.append(
            check_g1_collinearity_vif(panel, ["chuva_sem_mm", "chuva_catchment_mm"])
        )
    if "ano_epi" in panel.columns and "semana_epi" in panel.columns:
        panel_with_t = panel.with_columns(
            (pl.col("ano_epi") * 100 + pl.col("semana_epi")).alias("_ano_sem")
        )
        if panel_with_t["_ano_sem"].n_unique() >= 20:
            hygiene_gates.append(
                check_g4_right_censoring(panel_with_t, time_col="_ano_sem", value_col=target_col)
            )
    hygiene_scorecard = compile_hygiene_scorecard(hygiene_gates)

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
    dossier = compile_execution_dossier(
        runs_root=Path(args.runs_root),
        run_id=run_id,
        eval_year=eval_year,
        target_label=f"{target_col} (RS {eval_year})",
        benchmark_table=bench_df,
        hygiene_scorecard=hygiene_scorecard,
        extra_figures=artifacts.figure_paths,
        extra_notes=(
            f"ECE_surge={ece_val:.4f} | BSS_surge={bss_val:+.4f}",
            f"Violação monotônica = {cf_rep.monotonic_violation_rate:.4f}",
        ),
        seed=seed,
        mirror_champion=not bool(args.no_mirror_downloads),
    )
    emit_json_log(
        run_id=run_id,
        entry_point="cli:verify-rs",
        event="verification_complete",
        payload={
            "report_path": str(artifacts.report_path),
            "dossier_md_path": str(dossier.markdown_path),
            "dossier_pdf_path": str(dossier.pdf_path),
            "n_figures": len(dossier.figure_paths),
            "ece_surge": round(ece_val, 4),
            "bss_surge": round(bss_val, 4),
            "monotonic_violation_rate": cf_rep.monotonic_violation_rate,
            "hygiene_overall_status": hygiene_scorecard.overall_status,
        },
    )
    return 0


def _detect_accelerator() -> str:
    """Nome do acelerador disponível (informativo; não altera o cálculo tabular)."""
    try:
        from previsao_leitos.hydrology.device import resolve_accelerator

        spec = resolve_accelerator(prefer_gpu=True, requested_device="auto")
        return f"{spec.backend}:{spec.device_name}"
    except Exception:
        return "indisponível"


def _git_sha() -> str:
    """SHA curto do commit atual, ou `unknown` fora de um repositório Git."""
    import subprocess

    try:
        return subprocess.check_output(
            ["/usr/bin/git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


_PARALLEL_ARTIFACTS: tuple[str, ...] = (
    "benchmark.csv",
    "predictions.parquet",
    "comparison.csv",
    "decisions.csv",
    "eligibility.csv",
)


def _default_jobs(n_targets: int) -> int:
    """Um processo por alvo, sem estourar os núcleos (cada NUTS usa 2 cadeias)."""
    import os

    cpus = os.cpu_count() or 1
    return max(1, min(n_targets, max(1, cpus // 2)))


def _merge_part_runs(parts: list[Path], run_dir: Path) -> None:
    """Consolida os artefatos dos processos por alvo no run final."""
    for name in _PARALLEL_ARTIFACTS:
        frames = []
        for part in parts:
            path = part / name
            if not path.exists():
                continue
            frames.append(pl.read_parquet(path) if name.endswith(".parquet") else pl.read_csv(path))
        if not frames:
            continue
        combined = pl.concat(frames, how="diagonal_relaxed")
        destination = run_dir / name
        if name.endswith(".parquet"):
            combined.write_parquet(destination)
        else:
            combined.write_csv(destination)


def _run_targets_parallel(
    *,
    targets: list[object],
    args: argparse.Namespace,
    run_dir: Path,
    jobs: int,
) -> list[Path] | None:
    """Executa cada alvo em um processo próprio (`python -m previsao_leitos`)."""
    import os
    import subprocess
    import time

    parts_root = run_dir / "_parts"
    parts_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[var] = "1"

    pending = list(targets)
    running: dict[str, tuple[object, subprocess.Popen[str], Path]] = {}
    finished: list[Path] = []
    failures: list[str] = []

    while pending or running:
        while pending and len(running) < jobs:
            spec = pending.pop(0)
            key = str(getattr(spec, "key", spec))
            part_run_id = f"part_{key}"
            command = [
                sys.executable,
                "-m",
                "previsao_leitos",
                "benchmark",
                "--target",
                key,
                "--models",
                args.models,
                "--panel-path",
                args.panel_path,
                "--output-dir",
                str(parts_root),
                "--run-id",
                part_run_id,
                "--eval-year",
                str(args.eval_year),
                "--seed",
                str(args.seed),
                "--n-boot",
                str(args.n_boot),
                "--no-report",
                "--no-mirror-downloads",
            ]
            if args.allow_holdout:
                command.append("--allow-holdout")
            process = subprocess.Popen(  # noqa: S603 - comando montado internamente
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[key] = (spec, process, parts_root / part_run_id)
        for key in list(running):
            _spec, process, part_dir = running[key]
            if process.poll() is None:
                continue
            output = process.stdout.read() if process.stdout else ""
            if process.returncode != 0:
                failures.append(f"{key}: {output.strip()[-400:]}")
            else:
                finished.append(part_dir)
            del running[key]
        if running:
            time.sleep(0.5)

    if failures:
        for failure in failures:
            print(f"[erro] alvo falhou -> {failure}")
        return None
    return finished


def _cmd_benchmark(args: argparse.Namespace) -> int:
    import json
    import os
    import shutil
    import time
    from datetime import UTC, datetime

    from previsao_leitos.audit.benchmark_report import (
        build_figures,
        mirror_to_downloads,
        render_model_cards,
        render_report,
        sha256_of,
    )
    from previsao_leitos.audit.model_benchmark import run_benchmark
    from previsao_leitos.demand.modeling import make_modeling_splits, resolve_targets
    from previsao_leitos.demand.registry import resolve_selection
    from previsao_leitos.domain import DataContractError

    panel_path = Path(args.panel_path)
    if not panel_path.exists():
        print(
            f"[erro] painel de modelagem ausente: {panel_path}. "
            "Rode `python scripts/build_rs_modeling_panel.py`."
        )
        return 2

    os.environ["PREVISAO_BAYES_SAMPLER"] = args.sampler
    os.environ["PREVISAO_BAYES_DEVICE"] = args.device

    request_start = time.perf_counter()
    panel = pl.read_parquet(panel_path)
    run_id = args.run_id or (
        f"rs_benchmark_{args.eval_year}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    specs = resolve_selection(args.models)
    target_selection: str | tuple[str, ...] = args.target
    if isinstance(target_selection, str) and "," in target_selection:
        target_selection = tuple(p.strip() for p in target_selection.split(",") if p.strip())
    targets = resolve_targets(target_selection)
    jobs = args.jobs if args.jobs and args.jobs > 0 else _default_jobs(len(targets))
    jobs = max(1, min(jobs, len(targets)))
    parallel = jobs > 1 and len(targets) > 1

    notes: list[str] = []
    if parallel:
        print(
            f"[benchmark] execução paralela: {len(targets)} alvos em até {jobs} processos "
            "(2 cadeias NUTS por processo)"
        )
        finished = _run_targets_parallel(
            targets=list(targets), args=args, run_dir=run_dir, jobs=jobs
        )
        if finished is None:
            return 4
        _merge_part_runs(finished, run_dir)
        shutil.rmtree(run_dir / "_parts", ignore_errors=True)
        notes.append(f"execução paralela com {jobs} processos (um por alvo)")
    else:
        metrics_frames: list[pl.DataFrame] = []
        prediction_frames: list[pl.DataFrame] = []
        comparison_frames: list[pl.DataFrame] = []
        decision_frames: list[pl.DataFrame] = []
        eligibility_frames: list[pl.DataFrame] = []
        for spec in targets:
            try:
                splits = make_modeling_splits(
                    panel,
                    target=spec.key,
                    eval_year=args.eval_year,
                    allow_holdout=args.allow_holdout,
                )
            except DataContractError as exc:
                print(f"[erro] {spec.key}: {exc}")
                return 3
            result = run_benchmark(
                splits,
                run_id=f"{run_id}:{spec.key}",
                candidates=[s.name for s in specs],
                n_boot=args.n_boot,
                seed=args.seed,
            )
            metrics_frames.append(result.metrics.with_columns(pl.lit(spec.key).alias("target")))
            prediction_frames.append(
                result.predictions.with_columns(pl.lit(spec.key).alias("target"))
            )
            if result.comparison.height:
                comparison_frames.append(
                    result.comparison.with_columns(pl.lit(spec.key).alias("target"))
                )
            if result.decisions.height:
                decision_frames.append(
                    result.decisions.with_columns(pl.lit(spec.key).alias("target"))
                )
            if result.eligibility.height:
                eligibility_frames.append(
                    result.eligibility.with_columns(pl.lit(spec.key).alias("target"))
                )
            notes.extend(f"{spec.key}: {note}" for note in result.notes)

        pl.concat(metrics_frames, how="diagonal_relaxed").write_csv(run_dir / "benchmark.csv")
        pl.concat(prediction_frames, how="diagonal_relaxed").write_parquet(
            run_dir / "predictions.parquet"
        )
        if comparison_frames:
            pl.concat(comparison_frames, how="diagonal_relaxed").write_csv(
                run_dir / "comparison.csv"
            )
        if decision_frames:
            pl.concat(decision_frames, how="diagonal_relaxed").write_csv(
                run_dir / "decisions.csv"
            )
        if eligibility_frames:
            pl.concat(eligibility_frames, how="diagonal_relaxed").write_csv(
                run_dir / "eligibility.csv"
            )

    metrics_all = pl.read_csv(run_dir / "benchmark.csv")
    manifest = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "eval_year": args.eval_year,
        "targets": [spec.key for spec in targets],
        "models": [s.name for s in specs],
        "seed": args.seed,
        "n_boot": args.n_boot,
        "jobs": jobs,
        "parallel": parallel,
        "sampler": args.sampler,
        "device": args.device,
        "accelerator": _detect_accelerator(),
        "panel_path": str(panel_path),
        "panel_sha256": sha256_of(panel_path),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "elapsed_seconds": round(time.perf_counter() - request_start, 2),
        "candidates_ok": int(metrics_all.filter(pl.col("status") == "ok").height),
        "candidates_total": metrics_all.height,
        "notes": notes,
        "holdout_policy": "anos >= 2025 bloqueados; generalização avaliada apenas no M2",
        "disclaimer": "Not for patient care or live clinical use",
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    build_figures(run_dir, target_labels={spec.key: spec.label for spec in targets})
    cards_path = render_model_cards(
        run_dir, target_labels={spec.key: spec.label for spec in targets}
    )
    print(f"[benchmark] cartões: {cards_path}")

    print(
        f"[benchmark] run={run_id} | alvos={len(targets)} | jobs={jobs} | "
        f"candidatos ok={manifest['candidates_ok']}/{manifest['candidates_total']} | "
        f"{manifest['elapsed_seconds']}s | saida={run_dir}"
    )
    best = (
        metrics_all.filter(pl.col("status") == "ok")
        .sort("mase_vs_pers")
        .select(["target", "candidate", "mase_vs_pers", "cobertura_p10_p90"])
        .head(5)
    )
    print(best)

    if not args.no_report:
        _, pdf_path = render_report(
            run_dir,
            target_labels={spec.key: spec.label for spec in targets},
            eval_year=args.eval_year,
            models_requested=[s.name for s in specs],
        )
        print(f"[benchmark] PDF: {pdf_path}")
        if not args.no_mirror_downloads:
            mirrored = mirror_to_downloads(pdf_path)
            print(f"[benchmark] espelho verificado (SHA-256): {mirrored}")
    return 0


def _parse_years(value: str) -> tuple[int, ...]:
    """Aceita `2019-2024` ou `2019,2021,2024`."""
    value = value.strip()
    if "-" in value and "," not in value:
        start, end = value.split("-", 1)
        return tuple(range(int(start), int(end) + 1))
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _cmd_backtest(args: argparse.Namespace) -> int:
    import json
    import time
    from datetime import UTC, datetime

    from previsao_leitos.audit.backtest import run_rolling_backtest
    from previsao_leitos.audit.backtest_report import (
        plot_backtest_ranking,
        plot_error_decomposition,
        render_backtest_report,
        sha256_of,
    )
    from previsao_leitos.audit.benchmark_report import mirror_to_downloads
    from previsao_leitos.demand.modeling import resolve_targets
    from previsao_leitos.demand.registry import resolve_selection

    panel_path = Path(args.panel_path)
    if not panel_path.exists():
        print(f"[erro] painel de modelagem ausente: {panel_path}")
        return 2

    start = time.perf_counter()
    panel = pl.read_parquet(panel_path)
    run_id = args.run_id or (
        f"rs_backtest_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    excluded = {name.strip() for name in args.exclude.split(",") if name.strip()}
    selected = [spec.name for spec in resolve_selection(args.models) if spec.name not in excluded]
    years = _parse_years(args.folds)
    target_selection: str | tuple[str, ...] = args.target
    if isinstance(target_selection, str) and "," in target_selection:
        target_selection = tuple(
            part.strip() for part in target_selection.split(",") if part.strip()
        )
    targets = resolve_targets(target_selection)

    fold_frames: list[pl.DataFrame] = []
    summary_frames: list[pl.DataFrame] = []
    for spec in targets:
        result = run_rolling_backtest(
            panel,
            target=spec.key,
            eval_years=years,
            candidates=selected,
            n_boot=args.n_boot,
            seed=args.seed,
            shock_memory=args.shock_memory,
            admissions_momentum=args.admissions_momentum,
            run_id=run_id,
        )
        fold_frames.append(result.fold_metrics.with_columns(pl.lit(spec.key).alias("target")))
        summary_frames.append(result.summary.with_columns(pl.lit(spec.key).alias("target")))
        target_dir = run_dir / spec.key if len(targets) > 1 else run_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        if target_dir != run_dir:
            result.fold_metrics.write_csv(target_dir / "backtest_folds.csv")
            result.summary.write_csv(target_dir / "backtest_summary.csv")
        plot_backtest_ranking(
            result.fold_metrics,
            result.summary,
            output_path=target_dir / "fig_backtest_mase.png",
            target_label=spec.label,
        )
        plot_error_decomposition(
            result.summary,
            output_path=target_dir / "fig_error_decomposition.png",
            target_label=spec.label,
        )

    folds_all = pl.concat(fold_frames, how="diagonal_relaxed")
    summary_all = pl.concat(summary_frames, how="diagonal_relaxed")
    folds_all.write_csv(run_dir / "backtest_folds.csv")
    summary_all.write_csv(run_dir / "backtest_summary.csv")

    manifest = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "kind": "rolling_origin_backtest",
        "targets": [spec.key for spec in targets],
        "folds": list(years),
        "models": selected,
        "excluded": sorted(excluded),
        "seed": args.seed,
        "n_boot": args.n_boot,
        "panel_path": str(panel_path),
        "panel_sha256": sha256_of(panel_path),
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "holdout_policy": "folds <= 2024; 2025/2026 reservados ao M2",
        "disclaimer": "Not for patient care or live clinical use",
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"[backtest] run={run_id} | alvos={len(targets)} | folds={years[0]}-{years[-1]} | "
        f"modelos={len(selected)} | {manifest['elapsed_seconds']}s | saida={run_dir}"
    )
    best = summary_all.sort("mase_median").head(6)
    print(best)

    if not args.no_report:
        for spec in targets:
            target_dir = run_dir / spec.key if len(targets) > 1 else run_dir
            _, pdf_path = render_backtest_report(
                target_dir,
                target_label=spec.label,
                eval_years=years,
                pdf_name=f"relatorio_backtest_{spec.key}.pdf",
            )
            print(f"[backtest] PDF: {pdf_path}")
            if not args.no_mirror_downloads:
                print(f"[backtest] espelho: {mirror_to_downloads(pdf_path)}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from previsao_leitos.audit.benchmark_report import (
        build_figures,
        mirror_to_downloads,
        render_model_cards,
        render_report,
    )
    from previsao_leitos.demand.modeling import TARGET_SPECS

    run_dir = Path(args.output_dir) / args.run_id
    metrics_path = run_dir / "benchmark.csv"
    if not metrics_path.exists():
        print(f"[erro] benchmark.csv ausente em {run_dir}")
        return 2
    metrics = pl.read_csv(metrics_path)
    available = set(metrics["target"].unique().to_list())
    targets = [key for key in TARGET_SPECS if key in available]
    if not targets:
        print("[erro] nenhum alvo reconhecido em benchmark.csv")
        return 2
    models = sorted({str(m) for m in metrics["candidate"].unique().to_list()})
    build_figures(run_dir, target_labels={t: TARGET_SPECS[t].label for t in targets})
    render_model_cards(run_dir, target_labels={t: TARGET_SPECS[t].label for t in targets})
    _, pdf_path = render_report(
        run_dir,
        target_labels={t: TARGET_SPECS[t].label for t in targets},
        eval_year=args.eval_year,
        models_requested=models,
    )
    print(f"[report] PDF: {pdf_path}")
    if not args.no_mirror_downloads:
        mirrored = mirror_to_downloads(pdf_path)
        print(f"[report] espelho verificado (SHA-256): {mirrored}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Constrói o parser da CLI `previsao-leitos`."""
    parser = argparse.ArgumentParser(
        prog="previsao-leitos",
        description="CLI de pesquisa aberta para previsão ex-ante de leitos sob enchentes.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Verifica ambiente e prontidão das bases locais em JSON.")

    for name, help_text in (
        ("benchmark", "Treina, avalia e compara todos os modelos e gera o relatório PDF."),
        ("run", "Executa benchmark completo e relatório em um único comando."),
    ):
        p_bench = sub.add_parser(name, help=help_text)
        p_bench.add_argument(
            "--target",
            type=str,
            default="all",
            help=(
                "all | uti_adulto | uti_total | clinico | respiratorio "
                "(ou lista separada por vírgula)."
            ),
        )
        p_bench.add_argument("--eval-year", type=int, default=2024)
        p_bench.add_argument(
            "--models",
            type=str,
            default="all",
            help="all | champion | baselines | lista separada por vírgula.",
        )
        p_bench.add_argument(
            "--panel-path",
            type=str,
            default="data/processed/rs_modeling_panel_v3.parquet",
        )
        p_bench.add_argument("--output-dir", type=str, default="reports/runs")
        p_bench.add_argument("--run-id", type=str, default="")
        p_bench.add_argument("--seed", type=int, default=42)
        p_bench.add_argument("--n-boot", type=int, default=200)
        p_bench.add_argument(
            "--jobs",
            type=int,
            default=0,
            help="Processos paralelos, um por alvo (0 = automático pelo número de núcleos).",
        )
        p_bench.add_argument(
            "--sampler",
            choices=("auto", "nutpie", "numpyro"),
            default="auto",
            help="Amostrador NUTS do bayesiano; `numpyro` habilita GPU via JAX.",
        )
        p_bench.add_argument(
            "--device",
            choices=("auto", "cpu", "gpu"),
            default="auto",
            help="Dispositivo do amostrador bayesiano (GPU exige numpyro+JAX).",
        )
        p_bench.add_argument(
            "--admissions-momentum",
            action="store_true",
            default=False,
            help="Inclui internações por causa (com defasagens) como features.",
        )
        p_bench.add_argument(
            "--shock-memory",
            action="store_true",
            default=False,
            help="Inclui memória causal do choque (somas/EWMA) como features.",
        )
        p_bench.add_argument("--no-report", action="store_true", default=False)
        p_bench.add_argument("--no-mirror-downloads", action="store_true", default=False)
        p_bench.add_argument("--allow-holdout", action="store_true", default=False)

    p_bt = sub.add_parser(
        "backtest",
        help="Roda a régua de origem móvel (múltiplos folds) e emite o diagnóstico.",
    )
    p_bt.add_argument("--target", type=str, default="uti_adulto")
    p_bt.add_argument("--folds", type=str, default="2019-2024")
    p_bt.add_argument("--models", type=str, default="all")
    p_bt.add_argument(
        "--exclude",
        type=str,
        default="hierarchical_bayes",
        help="Modelos a excluir (lista separada por vírgula); bayes é caro por fold.",
    )
    p_bt.add_argument(
        "--panel-path",
        type=str,
        default="data/processed/rs_modeling_panel_v3.parquet",
    )
    p_bt.add_argument("--output-dir", type=str, default="reports/runs")
    p_bt.add_argument("--run-id", type=str, default="")
    p_bt.add_argument("--seed", type=int, default=42)
    p_bt.add_argument("--n-boot", type=int, default=50)
    p_bt.add_argument(
        "--admissions-momentum",
        action="store_true",
        default=False,
        help="Inclui internações por causa (com defasagens) como features.",
    )
    p_bt.add_argument(
        "--shock-memory",
        action="store_true",
        default=False,
        help="Inclui memória causal do choque (somas/EWMA) como features.",
    )
    p_bt.add_argument("--no-report", action="store_true", default=False)
    p_bt.add_argument("--no-mirror-downloads", action="store_true", default=False)

    p_report = sub.add_parser("report", help="Regera o PDF editorial a partir de um run existente.")
    p_report.add_argument("--run-id", type=str, required=True)
    p_report.add_argument("--output-dir", type=str, default="reports/runs")
    p_report.add_argument("--eval-year", type=int, default=2024)
    p_report.add_argument("--no-mirror-downloads", action="store_true", default=False)

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
    p_ver.add_argument("--runs-root", type=str, default="reports/runs")
    p_ver.add_argument("--run-id", type=str, default="")
    p_ver.add_argument("--seed", type=int, default=42)
    p_ver.add_argument("--allow-holdout", action="store_true", default=False)
    p_ver.add_argument("--no-mirror-downloads", action="store_true", default=False)
    return parser


def cli_main(argv: Sequence[str] | None = None) -> int:
    """Ponto de entrada principal da CLI."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "doctor":
        return _cmd_doctor()
    if args.command == "verify-rs":
        return _cmd_verify_rs(args)
    if args.command in ("benchmark", "run"):
        return _cmd_benchmark(args)
    if args.command == "backtest":
        return _cmd_backtest(args)
    if args.command == "report":
        return _cmd_report(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(cli_main())
