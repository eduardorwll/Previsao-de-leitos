"""Executa benchmark global Etapa 2 (F04) e Relatório Contrafactual de Verificação (F06) no RS.

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
    DemandForecaster,
    NegativeBinomialHurdleForecaster,
    PooledRidgeDeltaForecaster,
    QuantileGBDTForecaster,
    SpecialtyLosEstimator,
    prepare_global_demand_splits,
)
from previsao_leitos.geo import SplitPolicy

DATA_RAW = Path("data/raw")
PANEL_PATH = Path("data/processed/rs_painel_semanal_regioes_v2.parquet")
REPORTS_ROOT = Path("reports")

# Regiões-doadoras do Planalto/Norte/Alto Uruguai (menor impacto de cheia lagunar)
DONOR_REGIONS = (
    "Região 11 - Sete Povos das Missões",
    "Região 15 - Caminho das Águas",
    "Região 16 - Alto Uruguai Gaúcho",
    "Região 17 - Planalto",
    "Região 18 - Araucárias",
)

FOCUS_FLOOD_REGIONS = (
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
    flows = pl.read_parquet(_latest("sih_fluxos_referencia_semanal_v2_*.parquet"))
    mapbiomas = pl.read_parquet(_latest("mapbiomas_cobertura_rs_1985_2024_v1.parquet"))
    nasa = pl.read_parquet(_latest("nasa_power_daily_2010_2025_*.parquet"))

    policy = SplitPolicy(train_end_year=2022, cal_year=2023, dev_audit_year=2024, holdout_year=2025)

    # 1. Estimador causal de LOS por especialidade (<= 2022)
    los = SpecialtyLosEstimator.fit_causal(sih, train_end_year=policy.train_end_year)
    print("=== LOS CAUSAL RS POR ESPECIALIDADE (<= 2022) ===")
    print(los.summary_table)

    datasets_audit = (
        DatasetAuditEntry(
            "SIH-SUS (Demanda Semanal Agg v3)",
            "2010-2025 (2025 bloqueado)",
            sih.height,
            0,
            True,
            "Zero PII; agregação no BigQuery + supressão LGPD n<5",
        ),
        DatasetAuditEntry(
            "SIH-SUS (Fluxos Referência W_ref v2)",
            "2010-2022 (congelado)",
            flows.height,
            0,
            True,
            "Matriz intermunicipal MUNIC_RES -> CNES estritamente causal",
        ),
        DatasetAuditEntry(
            "CNES (Hospitais-Escola + Leitos v2)",
            "2010-2025",
            cnes.height,
            0,
            True,
            "44 hospitais-escola RS + estratificação UTI Adulto/Ped/Neo",
        ),
        DatasetAuditEntry(
            "NASA POWER (Hidrologia + V10M Vento Sul)",
            "2010-2025 (30/30 Regiões)",
            nasa.height,
            int(nasa.null_count().sum_horizontal().item()),
            True,
            "PRECTOTCORR, GWETTOP, T2M, V10M; 0 lacunas no RS",
        ),
        DatasetAuditEntry(
            "MapBiomas (Cobertura Municipal RS)",
            "1985-2022 (congelado)",
            mapbiomas.height,
            0,
            True,
            "Classes 24 (Urbano), 11/84 (Banhado/Marisma), 33 (Água)",
        ),
    )

    jobs: list[tuple[str, str, str, bool]] = [
        ("uti_adulto_diarias", "Diárias de UTI Adulto (30 Regiões RS)", "rs_2024_dev_uti", False),
        (
            "leptospirose_internacoes",
            "Internações Leptospirose - Onda 2 (30 Regiões RS)",
            "rs_2024_dev_lepto",
            True,
        ),
        (
            "respiratorio_internacoes",
            "Internações Respiratório / SRAG (30 Regiões RS)",
            "rs_2024_dev_resp",
            False,
        ),
    ]

    for target_col, label, run_id, prefer_hurdle in jobs:
        splits = prepare_global_demand_splits(
            panel,
            entity_col="nome_regiao_saude",
            target_col=target_col,
            dynamic_feature_cols=DYNAMIC_FEATURES,
            static_feature_cols=STATIC_FEATURES,
            horizon_weeks=2,
            lag_weeks=(0, 1, 2, 3, 4),
            eval_year=2024,
            policy=policy,
            allow_holdout=False,
        )

        candidates: list[DemandForecaster] = [
            QuantileGBDTForecaster(
                monotone_positive_prefixes=MONOTONE_POS,
                monotone_negative_prefixes=("frac_floresta",),
                max_iter=140,
                max_depth=5,
                learning_rate=0.05,
                random_state=42,
            ),
            PooledRidgeDeltaForecaster(alpha=150.0),
        ]
        if prefer_hurdle:
            candidates.append(NegativeBinomialHurdleForecaster(alpha_dispersion=0.8, ridge_l2=0.05))

        rows: list[dict[str, object]] = []
        fitted: dict[str, DemandForecaster] = {}
        for m in candidates:
            m.fit(splits.train_df, splits.feature_cols)
            preds = m.predict_quantiles(splits.eval_df)
            metrics_row = evaluate_probabilistic_predictions(
                splits.eval_df,
                preds,
                model_name=m.model_name,
                block_col="nome_regiao_saude",
                n_boot=250,
                seed=42,
            )
            rows.append(metrics_row)
            fitted[m.model_name] = m

        bench_df = pl.DataFrame(rows).sort("mase_vs_pers")
        best_name = str(bench_df["modelo"][0])
        best_model = fitted[best_name]

        # Auditoria Contrafactual (Controle Sintético Independente + Placebo + Waterfall)
        sc = NoFloodSyntheticControl(
            donor_entities=DONOR_REGIONS, train_end_year=policy.train_end_year
        )
        sc.fit(panel, entity_col="nome_regiao_saude", target_col=target_col)
        cf_eval = sc.predict_no_flood(
            splits.eval_df,
            full_panel=panel,
            entity_col="nome_regiao_saude",
            target_col=target_col,
        )
        cf_rep = evaluate_counterfactual_accuracy(
            model=best_model,
            train_df=splits.train_df,
            eval_df=splits.eval_df,
            no_flood_df=cf_eval,
            entity_col="nome_regiao_saude",
            focus_entities=FOCUS_FLOOD_REGIONS,
            shock_weeks=(18, 28),
            placebo_weeks=(4, 15),
            hydro_feature_prefixes=DYNAMIC_FEATURES,
            monotone_check_cols=("chuva_montante_pond_lag0", "remanso_lagunar_idx_lag0"),
        )

        # Decomposição waterfall em Porto Alegre e Pelotas/Sul nas semanas de cheia (19-24)
        poa_sul_shock = splits.eval_df.filter(
            pl.col("nome_regiao_saude").is_in(
                [
                    "Região 10 - Capital e Vale do Gravataí",
                    "Região 21 - Sul",
                ]
            )
            & pl.col("target_semana_epi").is_between(19, 24)
        )
        decomp = decompose_intermunicipal_and_landfill_effect(
            model=best_model,
            train_df=splits.train_df,
            eval_df=poa_sul_shock,
            feature_groups={
                "chuva_local": ("chuva_sem_mm", "api_max_sem", "umidade_solo_sem"),
                "escoamento_montante": ("chuva_montante_pond", "api_montante_pond"),
                "remanso_e_aterro": ("remanso_lagunar_idx", "exposicao_aterro_ativa"),
                "transbordamento_w_ref": ("chuva_catchment_mm", "api_catchment_max"),
            },
        )

        artifacts = build_verification_report(
            output_dir=REPORTS_ROOT / run_id,
            run_id=run_id,
            eval_year=2024,
            target_label=label,
            benchmark_table=bench_df,
            counterfactual_report=cf_rep,
            decomposition_df=decomp,
            datasets=datasets_audit,
            entity_col="nome_regiao_saude",
            policy=policy,
            allow_holdout=False,
            seed=42,
        )

        print(f"\n=== {label} (Melhor: {best_name}) -> {artifacts.report_path} ===")
        print(bench_df)
        print(
            f"Contrafactual | shock_mae={cf_rep.shock_excess_mae:.2f} "
            f"| shock_bias={cf_rep.shock_excess_bias:+.2f} "
            f"| peak_err={cf_rep.peak_timing_error_weeks:.2f} sem "
            f"| cov90={100.0 * cf_rep.interval_coverage_90:.1f}% "
            f"| placebo_bias={cf_rep.placebo_bias:+.2f} "
            f"| mono_viol={100.0 * cf_rep.monotonic_violation_rate:.2f}%"
        )

    elapsed = time.perf_counter() - t0
    print(f"\n[OK] Verificação Etapa 2 + Relatórios Contrafactuais concluída em {elapsed:.1f}s")


if __name__ == "__main__":
    main()
