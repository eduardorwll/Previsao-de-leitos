"""Executa Etapa 3 (Fatia 1): Calibração Conformal (2023) + SimPy + Relatório Completo (Fig. C1-C5).

Aviso: exclusivamente para pesquisa e avaliação metodológica
(Not for patient care or live clinical use).
O ano 2025 (holdout intocado) NUNCA é acessado neste script.
"""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl

from previsao_leitos.audit import (
    DatasetAuditEntry,
    NoFloodSyntheticControl,
    build_verification_report,
    decompose_intermunicipal_and_landfill_effect,
    evaluate_counterfactual_accuracy,
    evaluate_probabilistic_predictions,
)
from previsao_leitos.demand import (
    ConformalFloodCalibrator,
    QuantileGBDTForecaster,
    SpecialtyLosEstimator,
    prepare_global_demand_splits,
)
from previsao_leitos.geo import SplitPolicy
from previsao_leitos.readiness import (
    HospitalReadinessConfig,
    evaluate_observable_triggers,
    simulate_counterfactual_readiness,
)

DATA_RAW = Path("data/raw")
PANEL_PATH = Path("data/processed/rs_painel_semanal_regioes_v2.parquet")
REPORTS_ROOT = Path("reports")

DONOR_REGIONS = (
    "Região 11 - Sete Povos das Missões",
    "Região 15 - Caminho das Águas",
    "Região 16 - Alto Uruguai Gaúcho",
    "Região 17 - Planalto",
    "Região 18 - Araucárias",
)
FOCUS_REGIONS = (
    "Região 10 - Capital e Vale do Gravataí",
    "Região 21 - Sul",
    "Região 08 - Vale do Caí e Metropolitana",
    "Região 29 - Vales e Montanhas",
)
DYNAMIC_FEATURES = (
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
STATIC_FEATURES = (
    "frac_urbano",
    "frac_banhado_agua",
    "frac_floresta",
    "frac_area_polder_regiao",
    "indice_aterro_varzea",
)
MONOTONE_POS = (
    "chuva_montante_pond",
    "api_montante_pond",
    "remanso_lagunar_idx",
    "exposicao_aterro_ativa",
    "indice_aterro_varzea",
    "frac_area_polder_regiao",
)


def _latest(pattern: str) -> Path:
    matches = sorted(DATA_RAW.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Arquivo não encontrado em {DATA_RAW}: {pattern}")
    return matches[-1]


def main() -> None:
    t0 = time.perf_counter()
    panel = pl.read_parquet(PANEL_PATH)
    sih = pl.read_parquet(_latest("sih_demanda_semanal_agg_v3_*.parquet"))
    cnes = pl.read_parquet(_latest("cnes_hospitais_escola_v2_*.parquet"))
    cnes_leitos = pl.read_parquet(_latest("cnes_leitos_especialidade_v2_*.parquet"))
    ibge = pl.read_parquet(_latest("ibge_municipios_rs_v1_*.parquet"))
    flows = pl.read_parquet(_latest("sih_fluxos_referencia_semanal_v2_*.parquet"))
    mapbiomas = pl.read_parquet(_latest("mapbiomas_cobertura_rs_1985_2024_v1.parquet"))
    nasa = pl.read_parquet(_latest("nasa_power_daily_2010_2025_*.parquet"))

    policy = SplitPolicy(train_end_year=2022, cal_year=2023, dev_audit_year=2024, holdout_year=2025)
    los = SpecialtyLosEstimator.fit_causal(sih, train_end_year=policy.train_end_year)
    icu_los = los.mean_icu_days("uti_adulto")

    # 1. Splits estritamente separados: Treino <= 2022 | Calibração = 2023 | Auditoria = 2024
    sp_cal = prepare_global_demand_splits(
        panel,
        entity_col="nome_regiao_saude",
        target_col="uti_adulto_diarias",
        dynamic_feature_cols=DYNAMIC_FEATURES,
        static_feature_cols=STATIC_FEATURES,
        horizon_weeks=2,
        lag_weeks=(0, 1, 2, 3, 4),
        eval_year=2023,
        policy=policy,
    )
    sp_2024 = prepare_global_demand_splits(
        panel,
        entity_col="nome_regiao_saude",
        target_col="uti_adulto_diarias",
        dynamic_feature_cols=DYNAMIC_FEATURES,
        static_feature_cols=STATIC_FEATURES,
        horizon_weeks=2,
        lag_weeks=(0, 1, 2, 3, 4),
        eval_year=2024,
        policy=policy,
    )

    base_gbdt = QuantileGBDTForecaster(
        monotone_positive_prefixes=MONOTONE_POS,
        monotone_negative_prefixes=("frac_floresta",),
        max_iter=140,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
    ).fit(sp_cal.train_df, sp_cal.feature_cols)

    cal_gbdt = ConformalFloodCalibrator(
        base_model=base_gbdt,
        stress_cols=("api_max_sem_lag0", "api_montante_pond_lag0", "remanso_lagunar_idx_lag0"),
        stress_gamma=0.65,
    ).fit_calibration(train_df=sp_cal.train_df, cal_df=sp_cal.eval_df)

    raw_preds = base_gbdt.predict_quantiles(sp_2024.eval_df)
    cal_preds = cal_gbdt.predict_quantiles(sp_2024.eval_df)

    bench_df = pl.DataFrame(
        [
            evaluate_probabilistic_predictions(
                sp_2024.eval_df,
                raw_preds,
                model_name=base_gbdt.model_name,
                n_boot=250,
                seed=42,
            ),
            evaluate_probabilistic_predictions(
                sp_2024.eval_df,
                cal_preds,
                model_name=cal_gbdt.model_name,
                n_boot=250,
                seed=42,
            ),
        ]
    )

    # 2. Auditoria Contrafactual (Sem Enchente + Placebo + Waterfall)
    sc = NoFloodSyntheticControl(donor_entities=DONOR_REGIONS, train_end_year=2022).fit(
        panel, entity_col="nome_regiao_saude", target_col="uti_adulto_diarias"
    )
    cf_eval = sc.predict_no_flood(
        sp_2024.eval_df,
        full_panel=panel,
        entity_col="nome_regiao_saude",
        target_col="uti_adulto_diarias",
    )
    cf_raw = evaluate_counterfactual_accuracy(
        model=base_gbdt,
        train_df=sp_2024.train_df,
        eval_df=sp_2024.eval_df,
        no_flood_df=cf_eval,
        entity_col="nome_regiao_saude",
        focus_entities=FOCUS_REGIONS,
        shock_weeks=(18, 28),
        placebo_weeks=(4, 15),
        hydro_feature_prefixes=DYNAMIC_FEATURES,
        monotone_check_cols=("chuva_montante_pond_lag0", "remanso_lagunar_idx_lag0"),
    )
    cf_cal = evaluate_counterfactual_accuracy(
        model=cal_gbdt,
        train_df=sp_2024.train_df,
        eval_df=sp_2024.eval_df,
        no_flood_df=cf_eval,
        entity_col="nome_regiao_saude",
        focus_entities=FOCUS_REGIONS,
        shock_weeks=(18, 28),
        placebo_weeks=(4, 15),
        hydro_feature_prefixes=DYNAMIC_FEATURES,
        monotone_check_cols=("chuva_montante_pond_lag0", "remanso_lagunar_idx_lag0"),
    )
    print(
        f"Cobertura P10-P90 no Choque (Sem. 18-28): "
        f"Sem calibração={100.0 * cf_raw.interval_coverage_90:.1f}% -> "
        f"Com Conformal(2023)={100.0 * cf_cal.interval_coverage_90:.1f}% "
        f"(mono_viol={100.0 * cf_cal.monotonic_violation_rate:.2f}%)"
    )

    poa_sul_shock = sp_2024.eval_df.filter(
        pl.col("nome_regiao_saude").is_in(
            ["Região 10 - Capital e Vale do Gravataí", "Região 21 - Sul"]
        )
        & pl.col("target_semana_epi").is_between(19, 24)
    )
    decomp = decompose_intermunicipal_and_landfill_effect(
        model=cal_gbdt,
        train_df=sp_2024.train_df,
        eval_df=poa_sul_shock,
        feature_groups={
            "chuva_local": ("chuva_sem_mm", "api_max_sem", "umidade_solo_sem"),
            "escoamento_montante": ("chuva_montante_pond", "api_montante_pond"),
            "remanso_e_aterro": ("remanso_lagunar_idx", "exposicao_aterro_ativa"),
            "transbordamento_w_ref": ("chuva_catchment_mm", "api_catchment_max"),
        },
    )

    # 3. Simulação SimPy sobre oferta real pré-cheia CNES (Abr/2024) em POA e Pelotas/Sul
    mun_reg = ibge.select(["id_municipio", "nome_regiao_saude"]).unique(subset=["id_municipio"])
    sus_beds_by_reg = (
        cnes_leitos.filter(
            (pl.col("ano") == 2024)
            & (pl.col("mes") == 4)
            & (pl.col("categoria_leito") == "uti_adulto")
        )
        .join(mun_reg, on="id_municipio", how="inner")
        .group_by("nome_regiao_saude")
        .agg(pl.col("quantidade_sus").sum().alias("leitos_sus"))
    )
    beds_map = {
        str(r["nome_regiao_saude"]): int(r["leitos_sus"])
        for r in sus_beds_by_reg.iter_rows(named=True)
    }

    scored_poa = (
        sp_2024.eval_df.with_columns(cal_preds["pred_p90"])
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
    poa_beds = beds_map.get("Região 10 - Capital e Vale do Gravataí", 431)
    # Cenário realista RS 2024: oferta SUS 431 + 15% perda temporária em pôlderes/acesso + 90 surge
    sim_cfg_poa = HospitalReadinessConfig(
        region_label="Região 10 - Capital/POA (431 UTI SUS + Estresse Pôlder 15%)",
        base_beds=poa_beds,
        surge_beds=90,
        los_days=icu_los,
        elective_share=0.22,
        elective_deferral_frac=0.75,
        surge_lead_time_days=3,
        polder_hospital_loss_frac=0.15,
    )
    sim_poa = simulate_counterfactual_readiness(scored_poa, sim_cfg_poa, seed=42)

    peak_p90_beds = float(scored_poa.select(pl.col("pred_p90").max()).item()) / 7.0
    trig = evaluate_observable_triggers(
        predicted_p90_beds=peak_p90_beds,
        base_beds=poa_beds,
        upstream_api_mm=260.0,
        remanso_idx=400.0,
    )
    print(
        f"SimPy POA (431 UTI SUS, estresse pôlder 15%) | "
        f"Reativo h=0 pico={sim_poa.reactive_h0.peak_queue_patients} pac. "
        f"({sim_poa.reactive_h0.overload_days_gt95} dias >=95%) -> "
        f"Proativo h=2 pico={sim_poa.proactive_h2.peak_queue_patients} pac. "
        f"({sim_poa.proactive_h2.overload_days_gt95} dias >=95%) | "
        f"Redução pico={sim_poa.queue_peak_reduction_pct:.1f}% | Gatilho={trig.level}"
    )

    datasets_audit = (
        DatasetAuditEntry("SIH-SUS (Demanda Semanal Agg v3)", "2010-2025", sih.height, 0, True),
        DatasetAuditEntry("SIH-SUS (Fluxos W_ref v2)", "2010-2022", flows.height, 0, True),
        DatasetAuditEntry("CNES (Hospitais-Escola v2)", "2010-2025", cnes.height, 0, True),
        DatasetAuditEntry(
            "CNES (Leitos Especialidade v2)", "2010-2025", cnes_leitos.height, 0, True
        ),
        DatasetAuditEntry("NASA POWER (+V10M Vento Sul)", "2010-2025", nasa.height, 0, True),
        DatasetAuditEntry("MapBiomas (Cobertura RS)", "1985-2022", mapbiomas.height, 0, True),
    )

    artifacts = build_verification_report(
        output_dir=REPORTS_ROOT / "rs_2024_etapa3_uti",
        run_id="rs_2024_etapa3_uti",
        eval_year=2024,
        target_label="Diárias de UTI Adulto (Conformal 2023 + SimPy)",
        benchmark_table=bench_df,
        counterfactual_report=cf_cal,
        decomposition_df=decomp,
        datasets=datasets_audit,
        entity_col="nome_regiao_saude",
        policy=policy,
        allow_holdout=False,
        seed=42,
        simpy_comparison=sim_poa,
    )
    print(
        f"[OK] Relatório completo (5 figuras Fig. C1-C5) gerado em "
        f"{time.perf_counter() - t0:.1f}s -> {artifacts.report_path}"
    )


if __name__ == "__main__":
    main()
