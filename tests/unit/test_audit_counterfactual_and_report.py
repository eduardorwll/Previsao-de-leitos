"""Testes Prove-It F06: métricas, controle sintético contrafactual, gráficos e relatório."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from previsao_leitos.audit import (
    DatasetAuditEntry,
    NoFloodSyntheticControl,
    block_bootstrap_ci,
    build_verification_report,
    decompose_intermunicipal_and_landfill_effect,
    empirical_coverage,
    evaluate_counterfactual_accuracy,
    evaluate_probabilistic_predictions,
    pinball_loss,
)
from previsao_leitos.demand import QuantileGBDTForecaster, prepare_global_demand_splits
from previsao_leitos.domain import LeakageViolationError
from previsao_leitos.geo import SplitPolicy


def _synthetic_audit_panel() -> pl.DataFrame:
    rng = np.random.default_rng(77)
    rows: list[dict[str, object]] = []
    weeks = [(y, w) for y in range(2019, 2026) for w in range(1, 53)]
    specs = [
        ("Regiao 10 - Capital", "guaiba_capital", 0.020, 1.5, True),
        ("Regiao 21 - Sul", "lagoa_patos_sul", 0.008, 1.0, True),
        ("Regiao 17 - Planalto", "planalto_alto_jacui", 0.0005, 0.8, False),
        ("Regiao 23 - Serra", "serra_taquari_antas", 0.0002, 0.7, False),
    ]
    for reg, basin, aterro, scale, flooded in specs:
        rain = rng.gamma(2.0, 12.0, size=len(weeks))
        upstream = rain * (1.2 if flooded else 0.15)
        remanso = np.where(flooded, upstream * 1.2, 0.0)
        for shock_idx in (70, 278):  # 2020 (treino) e Sem 18 de 2024 (auditoria)
            if flooded:
                rain[shock_idx] = 260.0
                upstream[shock_idx] = 340.0
                remanso[shock_idx] = 850.0

        y = np.zeros(len(weeks), dtype=np.float64)
        for i, (_, w) in enumerate(weeks):
            up2 = upstream[i - 2] if i >= 2 else 0.0
            rem2 = remanso[i - 2] if i >= 2 else 0.0
            season = 10.0 * scale * np.sin(2.0 * np.pi * w / 52.0)
            y[i] = max(0.0, 100.0 * scale + season + 0.25 * up2 + 0.08 * rem2 + rng.normal(0, 2.5))

        for i, (yr, wk) in enumerate(weeks):
            rows.append(
                {
                    "id_regiao_saude": reg[:9],
                    "nome_regiao_saude": reg,
                    "bacia_id": basin,
                    "ano_epi": yr,
                    "semana_epi": wk,
                    "chuva_sem_mm": float(rain[i]),
                    "chuva_montante_pond": float(upstream[i]),
                    "api_max_sem": float(rain[i] * 0.8),
                    "api_montante_pond": float(upstream[i] * 0.8),
                    "remanso_lagunar_idx": float(remanso[i]),
                    "indice_aterro_varzea": float(aterro),
                    "exposicao_aterro_ativa": float(aterro * (rain[i] + upstream[i])),
                    "chuva_catchment_mm": float(0.6 * rain[i] + 0.4 * upstream[i]),
                    "api_catchment_max": float(0.6 * rain[i] + 0.4 * upstream[i]),
                    "umidade_solo_sem": 0.6,
                    "temp_media_sem_c": 18.0,
                    "uti_adulto_diarias": float(round(y[i], 2)),
                }
            )
    return pl.DataFrame(rows)


def test_probabilistic_metrics_and_block_bootstrap_are_deterministic() -> None:
    y = np.array([10.0, 20.0, 30.0, 40.0])
    p50 = np.array([12.0, 18.0, 29.0, 41.0])
    p10 = np.array([8.0, 15.0, 25.0, 35.0])
    p90 = np.array([15.0, 25.0, 35.0, 45.0])

    assert pinball_loss(y, p50, alpha=0.5) == pytest.approx(0.5 * np.mean(np.abs(y - p50)))
    assert empirical_coverage(y, p10, p90) == pytest.approx(1.0)

    df = pl.DataFrame(
        {"bloco": ["A", "A", "B", "B", "C", "C"], "err": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    )
    ci1 = block_bootstrap_ci(
        df, block_col="bloco", metric_fn=lambda d: float(d["err"].mean()), n_boot=200, seed=42
    )
    ci2 = block_bootstrap_ci(
        df, block_col="bloco", metric_fn=lambda d: float(d["err"].mean()), n_boot=200, seed=42
    )
    assert ci1 == ci2
    assert ci1.low <= ci1.point <= ci1.high

    eval_df = pl.DataFrame(
        {
            "nome_regiao_saude": ["R1", "R1", "R2", "R2"],
            "target_y": [10.0, 20.0, 30.0, 40.0],
            "y_origin": [12.0, 15.0, 25.0, 35.0],
            "y_seasonal_naive": [11.0, 18.0, 28.0, 38.0],
        }
    )
    preds_df = pl.DataFrame(
        {
            "pred_p10": [8.0, 15.0, 26.0, 34.0],
            "pred_p50": [10.5, 19.5, 30.5, 39.5],
            "pred_p90": [14.0, 24.0, 35.0, 44.0],
            "pred_p95": [15.0, 25.0, 36.0, 46.0],
        }
    )
    row = evaluate_probabilistic_predictions(eval_df, preds_df, model_name="Test", n_boot=50)
    assert float(str(row["mase_vs_pers"])) < 1.0
    assert float(str(row["cobertura_p10_p90"])) == pytest.approx(1.0)


def test_counterfactual_synthetic_control_placebo_and_decomposition_and_report(
    tmp_path: Path,
) -> None:
    df = _synthetic_audit_panel()
    policy = SplitPolicy(train_end_year=2022, cal_year=2023, dev_audit_year=2024, holdout_year=2025)
    dyn_cols = (
        "chuva_sem_mm",
        "chuva_montante_pond",
        "remanso_lagunar_idx",
        "exposicao_aterro_ativa",
        "chuva_catchment_mm",
    )
    splits = prepare_global_demand_splits(
        df,
        entity_col="nome_regiao_saude",
        target_col="uti_adulto_diarias",
        dynamic_feature_cols=dyn_cols,
        static_feature_cols=("indice_aterro_varzea",),
        horizon_weeks=2,
        lag_weeks=(0, 1, 2),
        eval_year=2024,
        policy=policy,
    )

    model = QuantileGBDTForecaster(
        monotone_positive_prefixes=(
            "chuva_montante_pond",
            "remanso_lagunar_idx",
            "exposicao_aterro_ativa",
            "indice_aterro_varzea",
        ),
        max_iter=60,
        random_state=42,
    )
    model.fit(splits.train_df, splits.feature_cols)

    sc = NoFloodSyntheticControl(
        donor_entities=("Regiao 17 - Planalto", "Regiao 23 - Serra"),
        train_end_year=2022,
    )
    sc.fit(df, entity_col="nome_regiao_saude", target_col="uti_adulto_diarias")
    cf_eval = sc.predict_no_flood(
        splits.eval_df,
        full_panel=df,
        entity_col="nome_regiao_saude",
        target_col="uti_adulto_diarias",
    )

    cf_report = evaluate_counterfactual_accuracy(
        model=model,
        train_df=splits.train_df,
        eval_df=splits.eval_df,
        no_flood_df=cf_eval,
        entity_col="nome_regiao_saude",
        focus_entities=("Regiao 10 - Capital", "Regiao 21 - Sul"),
        shock_weeks=(18, 26),
        placebo_weeks=(4, 14),
        hydro_feature_prefixes=dyn_cols,
        monotone_check_cols=("chuva_montante_pond_lag0", "remanso_lagunar_idx_lag0"),
    )

    # Zero violação monotônica e erro placebo contido
    assert cf_report.monotonic_violation_rate == pytest.approx(0.0)
    assert abs(cf_report.placebo_bias) < 15.0
    assert cf_report.shock_excess_mae >= 0.0

    decomp = decompose_intermunicipal_and_landfill_effect(
        model=model,
        train_df=splits.train_df,
        eval_df=splits.eval_df.filter(
            (pl.col("nome_regiao_saude") == "Regiao 10 - Capital")
            & (pl.col("target_semana_epi").is_between(19, 22))
        ),
        feature_groups={
            "chuva_local": ("chuva_sem_mm",),
            "escoamento_montante": ("chuva_montante_pond",),
            "remanso_e_aterro": ("remanso_lagunar_idx", "exposicao_aterro_ativa"),
            "transbordamento_w_ref": ("chuva_catchment_mm",),
        },
    )
    assert decomp.height == 5  # climatologia + 4 blocos de intervenção

    # Bloqueio do holdout 2025 no gerador de relatório
    with pytest.raises(LeakageViolationError, match="2025"):
        build_verification_report(
            output_dir=tmp_path / "blocked",
            run_id="test_blocked",
            eval_year=2025,
            target_label="Diárias UTI Adulto",
            benchmark_table=pl.DataFrame({"modelo": ["GBDT"], "mae": [10.0]}),
            counterfactual_report=cf_report,
            decomposition_df=decomp,
            datasets=(DatasetAuditEntry("SIH-SUS", "2010-2024", 1000, 0, True),),
            allow_holdout=False,
        )

    out_dir = tmp_path / "rs_2024_check"
    artifacts = build_verification_report(
        output_dir=out_dir,
        run_id="test_2024",
        eval_year=2024,
        target_label="Diárias UTI Adulto",
        benchmark_table=pl.DataFrame({"modelo": ["QuantileGBDT"], "mae": [10.0], "mase": [0.85]}),
        counterfactual_report=cf_report,
        decomposition_df=decomp,
        datasets=(DatasetAuditEntry("SIH-SUS v3", "2010-2024", 1929455, 0, True),),
        allow_holdout=False,
    )
    assert artifacts.report_path.exists()
    assert artifacts.manifest_path.exists()
    assert len(artifacts.figure_paths) == 4
    for fig_p in artifacts.figure_paths:
        assert fig_p.exists() and fig_p.stat().st_size > 1000
    text = artifacts.report_path.read_text(encoding="utf-8")
    assert "Not for patient care or live clinical use" in text
    assert "Metodologia" in text
    assert "Ferramentas" in text
    assert "Datasets" in text
    assert "Análise Contrafactual" in text
