"""Testes de regressão da Task 1.2.0: saneamento dos bugs V1.1, CanonicalPanelSchema e PDF."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from previsao_leitos.audit.bsts_counterfactual import (
    PreShockBstsCounterfactual,
    recursive_dynamic_do_intervention,
)
from previsao_leitos.audit.data_hygiene import (
    check_g2_join_loss,
    check_g8_degenerate_surge_labels,
    compile_hygiene_scorecard,
)
from previsao_leitos.audit.dossier_compiler import (
    _render_html_from_dossier,
    _render_pdf_via_chromium,
    compile_execution_dossier,
)
from previsao_leitos.audit.enso_what_if import (
    DEFAULT_ENSO_SCENARIOS,
    run_enso_what_if_oracle,
)
from previsao_leitos.audit.hypothesis_ledger import (
    ChampionChallengerRatchet,
    ModelScorecard,
)
from previsao_leitos.audit.metrics import estimate_surge_probability_from_quantiles
from previsao_leitos.demand.datasets import (
    CanonicalPanelSchema,
    validate_canonical_panel_frame,
)
from previsao_leitos.demand.extreme_uncertainty import MondrianEnsoConformalCalibrator
from previsao_leitos.demand.models import DemandForecaster
from previsao_leitos.demand.two_stage import OrthogonalTwoStageOccupancyForecaster
from previsao_leitos.domain import DataContractError
from previsao_leitos.readiness.simulator import (
    HospitalReadinessConfig,
    _run_single_simpy_policy,
    simulate_counterfactual_readiness,
)


def _make_lagged_two_stage_df() -> pl.DataFrame:
    rng = np.random.default_rng(120)
    rows = []
    for ent, cap in (("Capital", 2800.0), ("Interior", 80.0)):
        for yr in (2020, 2021, 2022, 2024):
            for wk in range(1, 31):
                is_shock = (yr == 2024 and wk in (19, 20)) or (yr <= 2022 and wk in (10, 11))
                rain = 320.0 if is_shock else float(rng.uniform(10.0, 55.0))
                el_nino = 2.0 if (yr == 2024 or (yr == 2022 and wk > 15)) else 0.0
                prob_f03 = 0.85 if is_shock else float(rng.uniform(0.03, 0.20))
                season = 0.75 + 0.06 * np.sin(2.0 * np.pi * wk / 52.0)
                shock_rate = 0.0007 * rain + 0.04 * el_nino + 0.18 * prob_f03
                y_val = cap * (season + shock_rate) + float(rng.normal(0.0, np.sqrt(cap) * 0.4))
                rows.append(
                    {
                        "nome_regiao_saude": ent,
                        "target_ano_epi": yr,
                        "target_semana_epi": wk,
                        "capacidade_sem": cap,
                        "y_origin": cap * season,
                        "y_roll4": cap * season,
                        "y_roll12": cap * 0.75,
                        "y_dev_roll12": cap * (season - 0.75),
                        "target_week_sin": float(np.sin(2.0 * np.pi * wk / 52.0)),
                        "target_week_cos": float(np.cos(2.0 * np.pi * wk / 52.0)),
                        "chuva_montante_multihop_lag0": rain,
                        "el_nino_intensity_lag0": el_nino,
                        "prob_enchente_f03_oof_lag0": prob_f03,
                        "frac_floresta": 0.20 if ent == "Capital" else 0.55,
                        "target_y": max(0.0, y_val),
                    }
                )
    return pl.DataFrame(rows)


def test_canonical_panel_schema_validates_and_blocks_pii_and_bad_columns() -> None:
    df = _make_lagged_two_stage_df()
    validated = validate_canonical_panel_frame(df)
    assert validated.height == df.height

    with pytest.raises(DataContractError, match="LGPD"):
        validate_canonical_panel_frame(df.with_columns(pl.lit("000").alias("cpf")))

    with pytest.raises(DataContractError, match="colunas obrigatórias ausentes"):
        validate_canonical_panel_frame(df.drop("target_week_sin"))

    with pytest.raises(DataContractError, match="estritamente positivo"):
        validate_canonical_panel_frame(
            df.with_columns(pl.lit(0.0).alias("capacidade_sem")),
            schema=CanonicalPanelSchema(capacity_col="capacidade_sem"),
        )


def test_two_stage_fixes_bugs_a_b_c_and_preserves_zero_monotonic_violations() -> None:
    df = _make_lagged_two_stage_df()
    train_df = df.filter(pl.col("target_ano_epi") <= 2022)
    eval_df = df.filter(pl.col("target_ano_epi") == 2024)

    feature_cols = (
        "target_week_sin",
        "target_week_cos",
        "y_origin",
        "y_roll4",
        "y_roll12",
        "y_dev_roll12",
        "chuva_montante_multihop_lag0",
        "el_nino_intensity_lag0",
        "prob_enchente_f03_oof_lag0",
        "frac_floresta",
    )
    model = OrthogonalTwoStageOccupancyForecaster(
        capacity_col="capacidade_sem",
        random_state=42,
    ).fit(train_df, feature_cols)

    # Bug A: harmônicos canônicos `target_week_sin/cos` reconhecidos no Estágio 1
    assert model.basal_cols == ("target_week_sin", "target_week_cos")

    # Bug B: todas as colunas AR proibidas excluídas do Estágio 2
    for forbidden in ("y_origin", "y_roll4", "y_roll12", "y_dev_roll12"):
        assert forbidden not in model.shock_cols
        assert forbidden not in model.basal_cols

    # Bug C: restrições monotônicas aplicadas corretamente em `chuva_montante_multihop_lag0`
    mono = model.monotone_constraints
    assert mono["chuva_montante_multihop_lag0"] == 1
    assert mono["el_nino_intensity_lag0"] == 1
    assert mono["prob_enchente_f03_oof_lag0"] == 1
    assert mono["frac_floresta"] == -1

    # Bug A fallback: mesmo sem basal_cols, predict_quantiles mantém basal_mean (sem colapso a 0)
    no_basal_model = OrthogonalTwoStageOccupancyForecaster(
        capacity_col="capacidade_sem",
        basal_cols=(),
        random_state=42,
    ).fit(train_df, feature_cols)
    preds_no_basal = no_basal_model.predict_quantiles(eval_df)
    assert float(preds_no_basal["pred_p50"].mean()) > 0.5 * float(eval_df["target_y"].mean())

    # Bug B heterocedasticidade: largura relativa (P90 - P10) / cap é menor na Capital
    preds = model.predict_quantiles(eval_df)
    ev_w = eval_df.with_columns(
        ((preds["pred_p90"] - preds["pred_p10"]) / pl.col("capacidade_sem")).alias("rel_width")
    )
    w_cap = float(ev_w.filter(pl.col("nome_regiao_saude") == "Capital")["rel_width"].mean())
    w_int = float(ev_w.filter(pl.col("nome_regiao_saude") == "Interior")["rel_width"].mean())
    assert w_cap < w_int

    # Monotonicidade exata (0.00% violações em todos os 4 quantis)
    bumped = eval_df.with_columns(
        (pl.col("chuva_montante_multihop_lag0") + 80.0).alias("chuva_montante_multihop_lag0")
    )
    preds_bumped = model.predict_quantiles(bumped)
    for qcol in ("pred_p10", "pred_p50", "pred_p90", "pred_p95"):
        assert np.all(preds_bumped[qcol].to_numpy() >= preds[qcol].to_numpy() - 1e-6)


def test_enso_oracle_perturbs_lagged_columns_and_increases_strictly() -> None:
    df = _make_lagged_two_stage_df()
    train_df = df.filter(pl.col("target_ano_epi") <= 2022)
    eval_df = df.filter(pl.col("target_ano_epi") == 2024)
    feature_cols = (
        "target_week_sin",
        "target_week_cos",
        "chuva_montante_multihop_lag0",
        "el_nino_intensity_lag0",
        "prob_enchente_f03_oof_lag0",
        "frac_floresta",
    )
    model = OrthogonalTwoStageOccupancyForecaster(
        capacity_col="capacidade_sem",
        random_state=42,
    ).fit(train_df, feature_cols)

    oracle_df = run_enso_what_if_oracle(
        model=model,
        baseline_eval_df=eval_df,
        scenarios=DEFAULT_ENSO_SCENARIOS,
    )
    p50_vals = oracle_df["demanda_p50_media"].to_list()
    # Bug D corrigido: não fica plano; super El Niño excede estritamente o cenário neutro
    assert p50_vals[-1] > p50_vals[0]
    for i in range(len(p50_vals) - 1):
        assert p50_vals[i] <= p50_vals[i + 1] + 1e-6


def test_preshock_bsts_adapts_to_level_shift_and_uses_z90_quantile() -> None:
    rng = np.random.default_rng(77)
    rows = []
    # Expansão estrutural de leitos em 2022-2024 (+120 diárias): OLS linear superestimaria,
    # enquanto UnobservedComponents local-level acompanha o patamar pré-choque W15-17!
    for yr in range(2016, 2025):
        level = 200.0 if yr <= 2021 else 320.0
        for wk in range(1, 53):
            season = 12.0 * np.sin(2.0 * np.pi * wk / 52.0)
            shock = 150.0 if (yr == 2024 and 18 <= wk <= 26) else 0.0
            y = level + season + shock + float(rng.normal(0.0, 4.0))
            rows.append(
                {
                    "nome_regiao_saude": "Região Teste",
                    "ano_epi": yr,
                    "semana_epi": wk,
                    "target_y": y,
                }
            )
    panel = pl.DataFrame(rows)
    bsts = PreShockBstsCounterfactual(shock_year=2024, shock_start_week=18, control_cols=()).fit(
        panel
    )
    ev = panel.filter(pl.col("ano_epi") == 2024).rename(
        {"ano_epi": "target_ano_epi", "semana_epi": "target_semana_epi"}
    )
    cf = bsts.predict_counterfactual(ev)

    pre_win = cf.filter(pl.col("target_semana_epi").is_between(15, 17))
    pre_bias = float(np.mean(pre_win["y_cf_no_flood"].to_numpy() - pre_win["target_y"].to_numpy()))
    assert abs(pre_bias) < 10.0

    # Verifica consistência do IC 90% (p05 <= mean <= p95 e largura finita)
    shock_win = cf.filter(pl.col("target_semana_epi").is_between(18, 26))
    assert float(shock_win["excess_obs"].mean()) > 110.0
    assert np.all(cf["y_cf_p05"].to_numpy() <= cf["y_cf_no_flood"].to_numpy() + 1e-9)
    assert np.all(cf["y_cf_no_flood"].to_numpy() <= cf["y_cf_p95"].to_numpy() + 1e-9)


class _RollSensitiveDummyModel(DemandForecaster):
    """Modelo determinístico que depende explicitamente de y_roll4, y_roll12 e y_dev_roll12."""

    @property
    def model_name(self) -> str:
        return "RollSensitiveDummy"

    def fit(self, train_df: pl.DataFrame, feature_cols: object) -> _RollSensitiveDummyModel:
        return self

    def predict_quantiles(self, df: pl.DataFrame) -> pl.DataFrame:
        p50 = (
            0.5 * df["y_roll4"].to_numpy()
            + 0.5 * df["y_roll12"].to_numpy()
            + 0.25 * df["y_dev_roll12"].to_numpy()
        )
        return pl.DataFrame(
            {
                "pred_p10": np.maximum(0.0, p50 - 5.0),
                "pred_p50": np.maximum(0.0, p50),
                "pred_p90": np.maximum(0.0, p50 + 5.0),
                "pred_p95": np.maximum(0.0, p50 + 8.0),
            }
        )


def test_recursive_dynamic_do_updates_y_roll4_y_roll12_and_y_dev_roll12() -> None:
    train_df = pl.DataFrame({"chuva_lag0": [20.0, 20.0], "y_origin": [100.0, 100.0]})
    # Semanas 19 e 20 carregam y_roll4/y_roll12 contaminados pela cheia (300.0)
    eval_df = pl.DataFrame(
        {
            "nome_regiao_saude": ["POA"] * 5,
            "target_ano_epi": [2024] * 5,
            "target_semana_epi": [16, 17, 18, 19, 20],
            "y_origin": [100.0, 100.0, 100.0, 300.0, 300.0],
            "y_roll4": [100.0, 100.0, 100.0, 300.0, 300.0],
            "y_roll12": [100.0, 100.0, 100.0, 300.0, 300.0],
            "y_dev_roll12": [0.0, 0.0, 0.0, 100.0, 100.0],
            "chuva_lag0": [20.0, 20.0, 280.0, 20.0, 20.0],
        }
    )
    cf = recursive_dynamic_do_intervention(
        model=_RollSensitiveDummyModel(),
        train_df=train_df,
        eval_df=eval_df,
        horizon_weeks=1,
        intervention_start_week=18,
        climate_prefixes=("chuva",),
    )
    p50_w19 = float(cf.filter(pl.col("target_semana_epi") == 19)["pred_p50_cf"][0])
    p50_w20 = float(cf.filter(pl.col("target_semana_epi") == 20)["pred_p50_cf"][0])
    assert p50_w19 == pytest.approx(100.0, abs=1e-6)
    assert p50_w20 == pytest.approx(100.0, abs=1e-6)


def test_surge_probability_masks_zero_baseline_rows_to_zero() -> None:
    preds = pl.DataFrame(
        {
            "pred_p10": [0.0, 90.0],
            "pred_p50": [1.0, 120.0],
            "pred_p90": [3.0, 150.0],
            "pred_p95": [4.0, 160.0],
        }
    )
    y_roll12 = np.array([0.0, 100.0])
    thr = 1.10 * np.maximum(y_roll12, 1e-6)
    probs = estimate_surge_probability_from_quantiles(preds, thr, baseline=y_roll12)
    assert probs[0] == 0.0
    assert probs[1] > 0.5


def test_dossier_editorial_html_has_no_raw_markdown_and_reject_protects_champion(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    downloads = tmp_path / "Downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    # Cria um CAMPEAO existente que NÃO pode ser sobrescrito em caso de REJECT
    existing_champ_pdf = downloads / "CAMPEAO_previsao_leitos_relatorio.pdf"
    existing_champ_pdf.write_bytes(b"%PDF-OLD-CHAMPION")

    scorecard = compile_hygiene_scorecard(
        [
            check_g2_join_loss(1000, 1000),
            check_g8_degenerate_surge_labels([0.0, 8.0], [0.0, 5.0], [False, True]),
        ]
    )
    bench_df = pl.DataFrame(
        {
            "modelo": ["V1", "V1.2_worse"],
            "mase_vs_pers": [0.80, 0.95],
            "pinball_p95": [12.0, 18.0],
            "cobertura_p10_p90": [0.82, 0.70],
        }
    )
    champ = ModelScorecard(
        run_id="v1",
        model_name="V1",
        mase=0.80,
        pinball_p95=12.0,
        coverage_p90=0.82,
        peak_timing_error_weeks=1.0,
        interior_specificity=0.80,
        ece=0.04,
    )
    worse_chall = ModelScorecard(
        run_id="v1_2_reject_test",
        model_name="V1.2_worse",
        mase=0.95,
        pinball_p95=18.0,
        coverage_p90=0.70,
        peak_timing_error_weeks=2.5,
        interior_specificity=0.60,
        ece=0.10,
    )
    ratchet_reject = ChampionChallengerRatchet().evaluate(champion=champ, challenger=worse_chall)
    assert ratchet_reject.decision == "REJECT"

    # Verifica HTML5/CSS3 editorial sem <pre> nem Markdown cru
    html_str = _render_html_from_dossier(
        title="Teste Editorial",
        run_id="v1_2_reject_test",
        eval_year=2024,
        target_label="uti_adulto_diarias",
        generated_at="2026-11-25T12:00:00+00:00",
        seed=42,
        git_sha="abc1234",
        hygiene_scorecard=scorecard,
        benchmark_table=bench_df,
        ratchet_evaluation=ratchet_reject,
        extra_notes=("Nota de teste",),
        figure_paths=(),
    )
    assert "<pre>" not in html_str
    assert "| --- |" not in html_str
    assert "<table class='data-table'>" in html_str
    assert "Not for patient care or live clinical use" in html_str
    assert "Capítulo 1" in html_str and "Capítulo 6" in html_str

    artifacts = compile_execution_dossier(
        runs_root=runs_root,
        run_id="v1_2_reject_test",
        eval_year=2024,
        target_label="uti_adulto_diarias",
        benchmark_table=bench_df,
        hygiene_scorecard=scorecard,
        ratchet_evaluation=ratchet_reject,
        mirror_champion=True,
        mirror_dir=downloads,
        prefer_chromium=False,
    )
    # Sob REJECT, o run específico é espelhado, mas CAMPEAO_* permanece INTACTO!
    assert (downloads / "v1_2_reject_test_relatorio_execucao_completo.pdf").exists()
    assert existing_champ_pdf.read_bytes() == b"%PDF-OLD-CHAMPION"

    # Alinhamento de run_manifest.json
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    for key in ("git_sha", "device", "precision", "deterministic_mode", "seed", "tools"):
        assert key in manifest


def test_simpy_uses_exogenous_flood_indicator_and_proactive_lead_time_lookahead() -> None:
    # Demanda constante moderada (~64% ocupação), MAS em_cheia_hidro=1 nas semanas 5..8
    # ativa perda de pôlder de 25% independentemente da demanda (zero circularidade!)
    rows = []
    for wk in range(1, 15):
        rows.append(
            {
                "semana_epi": wk,
                "chegadas_semanais": 75.0,
                "previsao_p90_semanal": 150.0 if 5 <= wk <= 8 else 80.0,
                "em_cheia_hidro": 1 if 5 <= wk <= 8 else 0,
            }
        )
    sched_flood = pl.DataFrame(rows)
    sched_dry = sched_flood.with_columns(pl.lit(0).alias("em_cheia_hidro"))

    cfg = HospitalReadinessConfig(
        base_beds=100,
        surge_beds=25,
        los_days=6.0,
        surge_lead_time_days=2,
        polder_hospital_loss_frac=0.25,
    )
    comp_flood = simulate_counterfactual_readiness(sched_flood, cfg, seed=42)
    comp_dry = simulate_counterfactual_readiness(sched_dry, cfg, seed=42)

    # Mesmo com chegadas idênticas e estáveis, a cheia hidrológica exógena reduz leitos no pôlder
    min_beds_flood = int(comp_flood.reactive_h0.daily_df["leitos_ativos"].min())
    min_beds_dry = int(comp_dry.reactive_h0.daily_df["leitos_ativos"].min())
    assert min_beds_flood == 75
    assert min_beds_dry == 100

    # Modo proativo pré-ativa leitos surge ANTES da semana 5 (lookahead de surge_lead_time_days)
    w5_day1_pro = int(
        comp_flood.proactive_h2.daily_df.filter(pl.col("semana_rel") == 5)["leitos_ativos"][0]
    )
    w5_day1_rea = int(
        comp_flood.reactive_h0.daily_df.filter(pl.col("semana_rel") == 5)["leitos_ativos"][0]
    )
    assert w5_day1_pro > w5_day1_rea
    assert comp_flood.proactive_h2.peak_queue_patients <= comp_flood.reactive_h0.peak_queue_patients


def test_canonical_panel_schema_rejects_nan_inf_null_and_bad_harmonics() -> None:
    df = _make_lagged_two_stage_df().head(10)

    with pytest.raises(DataContractError, match="target_y"):
        validate_canonical_panel_frame(df.with_columns(pl.lit(float("nan")).alias("target_y")))

    with pytest.raises(DataContractError, match="target_y"):
        validate_canonical_panel_frame(
            df.with_columns(pl.lit(None, dtype=pl.Float64).alias("target_y"))
        )

    with pytest.raises(DataContractError, match="capacidade_sem"):
        validate_canonical_panel_frame(
            df.with_columns(pl.lit(float("nan")).alias("capacidade_sem"))
        )

    with pytest.raises(DataContractError, match="capacidade_sem"):
        validate_canonical_panel_frame(
            df.with_columns(pl.lit(float("inf")).alias("capacidade_sem"))
        )

    with pytest.raises(DataContractError, match="harmônico"):
        validate_canonical_panel_frame(df.with_columns(pl.lit(1.5).alias("target_week_sin")))

    with pytest.raises(DataContractError, match="nulos"):
        validate_canonical_panel_frame(
            df.with_columns(pl.lit(None, dtype=pl.Int64).alias("target_semana_epi"))
        )


def test_two_stage_blocks_target_leakage_handles_nan_cap_legacy_aliases_and_high_zeros() -> None:
    rng = np.random.default_rng(99)
    n = 120
    cap = rng.uniform(25.0, 250.0, size=n)
    rain = rng.uniform(0.0, 300.0, size=n)
    y = np.where(rng.random(n) < 0.60, 0.0, cap * (0.45 + 0.0015 * rain))
    df = pl.DataFrame(
        {
            "target_week_sin": np.sin(np.arange(n) * 2.0 * np.pi / 52.0),
            "target_week_cos": np.cos(np.arange(n) * 2.0 * np.pi / 52.0),
            "capacidade_sem": cap,
            "chuva_montante_multihop_lag0": rain,
            "y_seasonal_naive": y,
            "target_y": y,
            "target_delta": np.zeros(n),
        }
    )
    model = OrthogonalTwoStageOccupancyForecaster(random_state=42).fit(
        df,
        (
            "target_week_sin",
            "target_week_cos",
            "chuva_montante_multihop_lag0",
            "y_seasonal_naive",
            "target_y",
            "target_delta",
        ),
    )
    for forbidden in ("y_seasonal_naive", "target_y", "target_delta"):
        assert forbidden not in model.shock_cols

    # Legacy harmonic alias resolution at predict_quantiles time
    df_legacy = df.rename(
        {"target_week_sin": "sin_target_wk", "target_week_cos": "cos_target_wk"}
    )
    preds_canon = model.predict_quantiles(df)
    preds_legacy = model.predict_quantiles(df_legacy)
    assert np.allclose(preds_canon["pred_p50"].to_numpy(), preds_legacy["pred_p50"].to_numpy())

    # NaN capacity sanitized cleanly (no NaN outputs)
    preds_nan = model.predict_quantiles(
        df.with_columns(pl.lit(float("nan")).alias("capacidade_sem"))
    )
    assert int(preds_nan["pred_p50"].is_nan().sum()) == 0
    assert np.all(np.isfinite(preds_nan["pred_p95"].to_numpy()))

    # Monotonicity preserved under 60% zero-inflated targets and dynamic capacity
    preds_bump = model.predict_quantiles(
        df.with_columns(
            (pl.col("chuva_montante_multihop_lag0") + 60.0).alias("chuva_montante_multihop_lag0")
        )
    )
    for qcol in ("pred_p10", "pred_p50", "pred_p90", "pred_p95"):
        assert np.all(preds_bump[qcol].to_numpy() >= preds_canon[qcol].to_numpy() - 1e-6)


def test_preshock_bsts_handles_gapped_future_missing_preshock_short_ridge_and_constant() -> None:
    rng = np.random.default_rng(77)
    rows = []
    for yr in range(2016, 2025):
        level = 200.0 if yr <= 2021 else 320.0
        for wk in range(1, 53):
            y = level + 12.0 * np.sin(2.0 * np.pi * wk / 52.0) + float(rng.normal(0.0, 4.0))
            rows.append(
                {"nome_regiao_saude": "Shift", "ano_epi": yr, "semana_epi": wk, "target_y": y}
            )
    panel = pl.DataFrame(rows)
    bsts = PreShockBstsCounterfactual(
        shock_year=2024, shock_start_week=18, control_cols=()
    ).fit(panel)

    ev_full = panel.filter((pl.col("ano_epi") == 2024) & (pl.col("semana_epi") >= 18)).rename(
        {"ano_epi": "target_ano_epi", "semana_epi": "target_semana_epi"}
    )
    ev_gap = ev_full.filter(pl.col("target_semana_epi").is_in([20, 26, 34]))
    cf_full = bsts.predict_counterfactual(ev_full).filter(
        pl.col("target_semana_epi").is_in([20, 26, 34])
    )
    cf_gap = bsts.predict_counterfactual(ev_gap)

    # Exact match between gapped and contiguous forecasts (no underestimated horizon variance!)
    assert np.allclose(
        cf_full["y_cf_no_flood"].to_numpy(), cf_gap["y_cf_no_flood"].to_numpy(), atol=1e-6
    )
    assert np.allclose(cf_full["y_cf_p95"].to_numpy(), cf_gap["y_cf_p95"].to_numpy(), atol=1e-6)
    assert np.allclose(cf_full["y_cf_p05"].to_numpy(), cf_gap["y_cf_p05"].to_numpy(), atol=1e-6)

    # Missing pre-shock week W10 is not treated as future step 1 and does not shift W18/W19
    panel_no_w10 = panel.filter(~((pl.col("ano_epi") == 2024) & (pl.col("semana_epi") == 10)))
    bsts_gap_pre = PreShockBstsCounterfactual(
        shock_year=2024, shock_start_week=18, control_cols=()
    ).fit(panel_no_w10)
    ev_w10_and_fut = panel.filter(
        (pl.col("ano_epi") == 2024) & (pl.col("semana_epi").is_in([10, 18, 19]))
    ).rename({"ano_epi": "target_ano_epi", "semana_epi": "target_semana_epi"})
    ev_fut_only = ev_w10_and_fut.filter(pl.col("target_semana_epi") >= 18)
    cf_both = bsts_gap_pre.predict_counterfactual(ev_w10_and_fut).filter(
        pl.col("target_semana_epi") >= 18
    )
    cf_only = bsts_gap_pre.predict_counterfactual(ev_fut_only)
    assert np.allclose(cf_both["y_cf_p95"].to_numpy(), cf_only["y_cf_p95"].to_numpy(), atol=1e-6)

    # Short series (16 weeks -> Ridge fallback) & constant series (constant mode)
    short_rows = [
        {
            "nome_regiao_saude": "Short",
            "ano_epi": 2024,
            "semana_epi": wk,
            "target_y": 50.0 + 2.0 * wk,
        }
        for wk in range(1, 22)
    ] + [
        {"nome_regiao_saude": "Const", "ano_epi": 2024, "semana_epi": wk, "target_y": 0.0}
        for wk in range(1, 22)
    ]
    short_panel = pl.DataFrame(short_rows)
    bsts_sc = PreShockBstsCounterfactual(
        shock_year=2024, shock_start_week=18, control_cols=()
    ).fit(short_panel)
    ev_sc = short_panel.rename({"ano_epi": "target_ano_epi", "semana_epi": "target_semana_epi"})
    cf_sc = bsts_sc.predict_counterfactual(ev_sc)
    const_cf = cf_sc.filter(pl.col("nome_regiao_saude") == "Const")
    short_cf = cf_sc.filter(pl.col("nome_regiao_saude") == "Short")
    assert np.all(const_cf["y_cf_no_flood"].to_numpy() == 0.0)
    assert np.all(short_cf["y_cf_p95"].to_numpy() > short_cf["y_cf_no_flood"].to_numpy())


def test_recursive_do_preserves_pre_intervention_origin_and_avoids_multiyear_collision() -> None:
    class _BiasedOriginModel(DemandForecaster):
        @property
        def model_name(self) -> str:
            return "BiasedOrigin"

        def fit(self, train_df: pl.DataFrame, feature_cols: object) -> _BiasedOriginModel:
            return self

        def predict_quantiles(self, df: pl.DataFrame) -> pl.DataFrame:
            p50 = df["y_origin"].to_numpy() + 50.0
            return pl.DataFrame(
                {"pred_p10": p50, "pred_p50": p50, "pred_p90": p50, "pred_p95": p50}
            )

    train_df = pl.DataFrame({"chuva_lag0": [20.0, 20.0], "y_origin": [100.0, 100.0]})
    ev_pre = pl.DataFrame(
        {
            "nome_regiao_saude": ["POA"] * 5,
            "target_ano_epi": [2024] * 5,
            "target_semana_epi": [16, 17, 18, 19, 20],
            "y_origin": [100.0, 100.0, 100.0, 100.0, 300.0],
            "chuva_lag0": [20.0] * 5,
        }
    )
    cf_pre = recursive_dynamic_do_intervention(
        model=_BiasedOriginModel(),
        train_df=train_df,
        eval_df=ev_pre,
        horizon_weeks=2,
        intervention_start_week=18,
        climate_prefixes=("chuva",),
    )
    # W18 (origin=W16 < 18) and W19 (origin=W17 < 18) keep real pre-flood y_origin=100 -> 150!
    w18_val = float(cf_pre.filter(pl.col("target_semana_epi") == 18)["pred_p50_cf"][0])
    w19_val = float(cf_pre.filter(pl.col("target_semana_epi") == 19)["pred_p50_cf"][0])
    w20_val = float(cf_pre.filter(pl.col("target_semana_epi") == 20)["pred_p50_cf"][0])
    assert w18_val == pytest.approx(150.0)
    assert w19_val == pytest.approx(150.0)
    # W20 (origin=W18 >= 18) replaces contaminated y_origin=300 with cf(W18)=150 -> 200!
    assert w20_val == pytest.approx(200.0)

    # Multi-year eval_df: 2023 W18 must NOT contaminate 2024 W19 when 2024 starts at W19!
    ev_2yr = pl.DataFrame(
        {
            "nome_regiao_saude": ["POA"] * 4,
            "target_ano_epi": [2023, 2023, 2024, 2024],
            "target_semana_epi": [18, 19, 18, 19],
            "y_origin": [100.0, 100.0, 500.0, 999.0],
            "chuva_lag0": [20.0] * 4,
        }
    )
    cf_2yr = recursive_dynamic_do_intervention(
        model=_BiasedOriginModel(),
        train_df=train_df,
        eval_df=ev_2yr,
        horizon_weeks=1,
        intervention_start_week=18,
        climate_prefixes=("chuva",),
    )
    p24_w19 = float(
        cf_2yr.filter((pl.col("target_ano_epi") == 2024) & (pl.col("target_semana_epi") == 19))[
            "pred_p50_cf"
        ][0]
    )
    assert p24_w19 == pytest.approx(600.0)  # uses 2024 W18 (550) + 50, NOT 2023 W18 (150)!


def test_simpy_resets_stale_surge_request_after_transient_spike() -> None:
    n_weeks = 7
    n_days = n_weeks * 7
    sched = pl.DataFrame(
        {
            "semana_epi": list(range(1, n_weeks + 1)),
            "chegadas_semanais": [10.0] * n_weeks,
            "previsao_p90_semanal": [10.0] * n_weeks,
        }
    )
    urg = [0] * n_days
    urg[9] = 100   # 1-day blip on day 10 (< lead_time=5 -> must NOT linger!)
    urg[34] = 100  # new crisis on day 35 -> must respect full 5-day lead time!
    los = [[1.5] * max(1, u) for u in urg]
    cfg = HospitalReadinessConfig(
        base_beds=100, surge_beds=30, los_days=1.5, surge_lead_time_days=5
    )
    res = _run_single_simpy_policy(
        weekly_schedule=sched,
        config=cfg,
        policy_mode="reactive_h0",
        daily_arrivals_urgent=urg,
        daily_arrivals_elective=[0] * n_days,
        daily_los_samples=los,
    )
    day36_beds = int(res.daily_df.filter(pl.col("dia") == 36)["leitos_ativos"][0])
    assert day36_beds == 100


def test_chromium_pdf_renderer_passes_no_header_flags_and_wide_table_uses_compact_css(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_cmd: list[str] = []

    class _FakeProc:
        returncode = 0

    def _fake_run(cmd: list[str], **kwargs: object) -> _FakeProc:
        captured_cmd.extend(cmd)
        for arg in cmd:
            if arg.startswith("--print-to-pdf="):
                Path(arg.split("=", 1)[1]).write_bytes(b"%PDF-1.4\n%%EOF\n")
        return _FakeProc()

    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/chromium" if name == "chromium" else None
    )
    monkeypatch.setattr("subprocess.run", _fake_run)

    html_p = tmp_path / "doc.html"
    pdf_p = tmp_path / "doc.pdf"
    html_p.write_text("<html><body>ok</body></html>", encoding="utf-8")
    assert _render_pdf_via_chromium(html_p, pdf_p) is True
    assert "--no-pdf-header-footer" in captured_cmd
    assert "--print-to-pdf-no-header" in captured_cmd

    # Wide 12-column benchmark table uses compact fixed layout without clipping
    wide_bench = pl.DataFrame(
        {
            "modelo": ["V1_QuantileGBDT_Conformal"],
            "mae_p50": [48.4],
            "mae_persistencia": [59.8],
            "mae_sazonal": [65.9],
            "mase_vs_pers": [0.81],
            "mase_ci95_low": [0.76],
            "mase_ci95_high": [0.85],
            "mase_vs_saz": [0.73],
            "pinball_p50": [24.2],
            "pinball_p90": [13.1],
            "pinball_p95": [8.69],
            "cobertura_p10_p90": [0.806],
        }
    )
    scorecard = compile_hygiene_scorecard([check_g2_join_loss(100, 100)])
    html_str = _render_html_from_dossier(
        title="Wide Table Test",
        run_id="wide_test",
        eval_year=2024,
        target_label="uti_adulto_diarias",
        generated_at="2026-11-25T12:00:00+00:00",
        seed=42,
        git_sha="abc1234",
        hygiene_scorecard=scorecard,
        benchmark_table=wide_bench,
        ratchet_evaluation=None,
        extra_notes=(),
        figure_paths=(),
    )
    assert "<table class='data-table compact'>" in html_str
    assert "cobertura p10 p90" in html_str
    assert "_<wbr>" in html_str
    assert "Holdout 2025 preservado" in html_str

    # Holdout eval_year=2025 must NOT claim "Holdout 2025 preservado"
    html_2025 = _render_html_from_dossier(
        title="Holdout Test",
        run_id="holdout_test",
        eval_year=2025,
        target_label="uti_adulto_diarias",
        generated_at="2026-11-25T12:00:00+00:00",
        seed=42,
        git_sha="abc1234",
        hygiene_scorecard=scorecard,
        benchmark_table=wide_bench,
        ratchet_evaluation=None,
        extra_notes=(),
        figure_paths=(),
    )
    assert "Holdout 2025 preservado" not in html_2025
    assert "holdout/prospectiva (2025)" in html_2025


def test_review_regressions_nan_fit_bsts_multiyear_do_schema_and_simulator() -> None:
    # 1. CanonicalPanelSchema raises DataContractError (not raw Polars error) on string numeric col
    bad_str_df = pl.DataFrame(
        {
            "nome_regiao_saude": ["POA"],
            "target_ano_epi": [2024],
            "target_semana_epi": [10],
            "target_y": ["not_a_number"],
            "target_week_sin": [0.5],
            "target_week_cos": [0.5],
            "capacidade_sem": [100.0],
        }
    )
    with pytest.raises(DataContractError, match="target_y"):
        validate_canonical_panel_frame(bad_str_df)

    # 2. OrthogonalTwoStageOccupancyForecaster.fit sanitizes NaN/Inf in basal & shock cols
    n = 30
    nan_fit_df = pl.DataFrame(
        {
            "target_week_sin": [float("nan"), float("inf")] + [0.1] * (n - 2),
            "target_week_cos": [0.9] * n,
            "capacidade_sem": [100.0] * n,
            "chuva_montante_multihop_lag0": [float("inf")] + [15.0] * (n - 1),
            "target_y": [50.0 + float(i % 5) for i in range(n)],
            "porte_regional": ["interior"] * n,
            "enso_regime": ["neutro"] * n,
        }
    )
    f04 = OrthogonalTwoStageOccupancyForecaster(random_state=42).fit(
        nan_fit_df, ("target_week_sin", "target_week_cos", "chuva_montante_multihop_lag0")
    )
    preds = f04.predict_quantiles(nan_fit_df)
    assert np.all(np.isfinite(preds["pred_p50"].to_numpy()))

    # Missing shock col in inference raises DataContractError
    with pytest.raises(DataContractError, match="Colunas de choque ausentes"):
        f04.predict_quantiles(nan_fit_df.drop("chuva_montante_multihop_lag0"))

    # NaN in target_y during fit raises DataContractError
    with pytest.raises(DataContractError, match="target_y"):
        OrthogonalTwoStageOccupancyForecaster(random_state=42).fit(
            nan_fit_df.with_columns(pl.lit(float("nan")).alias("target_y")),
            ("target_week_sin", "target_week_cos", "chuva_montante_multihop_lag0"),
        )

    # 3. MondrianEnsoConformalCalibrator conforms to DemandForecaster.fit()
    cal = MondrianEnsoConformalCalibrator(base_model=f04)
    cal.fit(nan_fit_df, ("target_week_sin", "target_week_cos", "chuva_montante_multihop_lag0"))
    cal.fit_calibration(cal_df=nan_fit_df)
    assert cal.predict_quantiles(nan_fit_df).height == n

    # 4. PreShockBstsCounterfactual sanitizes NaN in exogenous control column (no MissingDataError)
    rng = np.random.default_rng(1)
    bsts_rows = [
        {
            "nome_regiao_saude": "R1",
            "ano_epi": yr,
            "semana_epi": wk,
            "srag_viral_confirmado": float("nan") if (yr == 2023 and wk == 5) else 10.0,
            "target_y": 200.0 + float(rng.normal(0.0, 4.0)),
        }
        for yr in range(2022, 2025)
        for wk in range(1, 53)
    ]
    bsts_panel = pl.DataFrame(bsts_rows)
    bsts = PreShockBstsCounterfactual(
        shock_year=2024, shock_start_week=18, control_cols=("srag_viral_confirmado",)
    ).fit(bsts_panel)
    ev_bsts = bsts_panel.filter(pl.col("ano_epi") == 2024).rename(
        {"ano_epi": "target_ano_epi", "semana_epi": "target_semana_epi"}
    )
    cf_bsts = bsts.predict_counterfactual(ev_bsts)
    assert int(cf_bsts["y_cf_no_flood"].is_nan().sum()) == 0

    # 5. recursive_dynamic_do_intervention with intervention_start_year stays active across year end
    class _RainEchoModel(DemandForecaster):
        @property
        def model_name(self) -> str:
            return "RainEcho"

        def fit(self, train_df: pl.DataFrame, feature_cols: object) -> _RainEchoModel:
            return self

        def predict_quantiles(self, df: pl.DataFrame) -> pl.DataFrame:
            v = df["chuva_lag0"].to_numpy()
            return pl.DataFrame({"pred_p10": v, "pred_p50": v, "pred_p90": v, "pred_p95": v})

    train_do = pl.DataFrame({"chuva_lag0": [10.0, 10.0], "y_origin": [50.0, 50.0]})
    ev_cross_yr = pl.DataFrame(
        {
            "nome_regiao_saude": ["POA"] * 4,
            "target_ano_epi": [2024, 2024, 2025, 2025],
            "target_semana_epi": [51, 52, 1, 2],
            "y_origin": [50.0] * 4,
            "chuva_lag0": [200.0] * 4,
        }
    )
    cf_cross = recursive_dynamic_do_intervention(
        model=_RainEchoModel(),
        train_df=train_do,
        eval_df=ev_cross_yr,
        horizon_weeks=1,
        intervention_start_week=18,
        intervention_start_year=2024,
        climate_prefixes=("chuva",),
    )
    assert np.allclose(cf_cross["pred_p50_cf"].to_numpy(), 10.0)

    # 6. Empty scenarios in run_enso_what_if_oracle raises DataContractError
    with pytest.raises(DataContractError, match="scenarios vazio"):
        run_enso_what_if_oracle(model=_RainEchoModel(), baseline_eval_df=ev_cross_yr, scenarios=())

    # 7. Simulator guards empty weekly_schedule and invalid config bounds
    empty_sched = pl.DataFrame(
        {
            "chegadas_semanais": pl.Series([], dtype=pl.Float64),
            "previsao_p90_semanal": pl.Series([], dtype=pl.Float64),
        }
    )
    with pytest.raises(DataContractError, match="vazio"):
        simulate_counterfactual_readiness(
            empty_sched, HospitalReadinessConfig(base_beds=10, surge_beds=5, los_days=3.0)
        )
    with pytest.raises(DataContractError, match="surge_lead_time_days"):
        HospitalReadinessConfig(base_beds=10, surge_beds=5, los_days=3.0, surge_lead_time_days=-1)
    with pytest.raises(DataContractError, match="elective_deferral_frac"):
        HospitalReadinessConfig(
            base_beds=10, surge_beds=5, los_days=3.0, elective_deferral_frac=1.5
        )
