"""Testes da Task 1.2.6 — explicabilidade SHAP auditada em 2 estágios (F04.1).

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from previsao_leitos.audit.shap_audit import (
    ConformalMargins,
    ShapPointDecomposition,
    additivity_residual,
    decompose_two_stage_point,
)
from previsao_leitos.demand.calibration import ConformalFloodCalibrator
from previsao_leitos.demand.two_stage import OrthogonalTwoStageOccupancyForecaster

FEATURE_COLS: tuple[str, ...] = (
    "sin_target_wk",
    "cos_target_wk",
    "prob_enchente_f03_oof",
    "severidade_hidro_f03_oof",
    "chuva_montante_pond_lag0",
)


def _make_synthetic_panel() -> pl.DataFrame:
    """Painel sintético causal: 2 regiões de portes distintos, 2020-2024."""
    rng = np.random.default_rng(2026)
    rows: list[dict[str, float | int | str]] = []
    for ent, cap in (("Capital", 3000.0), ("Interior", 100.0)):
        for yr in (2020, 2021, 2022, 2023, 2024):
            for wk in range(1, 31):
                prob_f03 = (
                    0.85 if (yr == 2024 and wk in (19, 20)) else float(rng.uniform(0.02, 0.25))
                )
                if (yr == 2020 and wk in (8, 9)) or (yr == 2022 and wk in (23, 24)):
                    prob_f03 = 0.80
                sev_f03 = prob_f03 * 2.0
                season = 0.72 + 0.05 * np.sin(2.0 * np.pi * wk / 52.0)
                shock_rate = 0.28 * prob_f03 + 0.04 * sev_f03
                y_val = cap * (season + shock_rate + float(rng.normal(0.0, 0.015)))
                rows.append(
                    {
                        "nome_regiao_saude": ent,
                        "target_ano_epi": yr,
                        "target_semana_epi": wk,
                        "capacidade_sem": cap,
                        "y_origin": cap * season,
                        "y_roll12": cap * 0.72,
                        "sin_target_wk": float(np.sin(2.0 * np.pi * wk / 52.0)),
                        "cos_target_wk": float(np.cos(2.0 * np.pi * wk / 52.0)),
                        "prob_enchente_f03_oof": prob_f03,
                        "severidade_hidro_f03_oof": sev_f03,
                        "chuva_montante_pond_lag0": 20.0 + 300.0 * prob_f03,
                        "target_y": max(0.0, y_val),
                    }
                )
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def panel_split() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    df = _make_synthetic_panel()
    train_df = df.filter(pl.col("target_ano_epi") <= 2022)
    cal_df = df.filter(pl.col("target_ano_epi") == 2023)
    eval_df = df.filter(pl.col("target_ano_epi") == 2024)
    return df, train_df, cal_df, eval_df


@pytest.fixture(scope="module")
def fitted_model(
    panel_split: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame],
) -> OrthogonalTwoStageOccupancyForecaster:
    _, train_df, _, _ = panel_split
    return OrthogonalTwoStageOccupancyForecaster().fit(train_df, FEATURE_COLS)


def test_additivity_residual_in_rate_space_below_1e6(
    panel_split: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame],
    fitted_model: OrthogonalTwoStageOccupancyForecaster,
) -> None:
    _, train_df, _, eval_df = panel_split
    dec = decompose_two_stage_point(fitted_model, eval_df, background=train_df)

    assert isinstance(dec, ShapPointDecomposition)
    rate_residual, _ = additivity_residual(dec)
    assert rate_residual < 1e-6
    assert dec.conformal_margins is None


def test_composed_point_reconstructs_p50_in_beds(
    panel_split: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame],
    fitted_model: OrthogonalTwoStageOccupancyForecaster,
) -> None:
    _, train_df, _, eval_df = panel_split
    dec = decompose_two_stage_point(fitted_model, eval_df, background=train_df)

    p50 = eval_df_p50(fitted_model, eval_df)
    shap_total = np.sum(np.column_stack(list(dec.shap_values.values())), axis=1)
    reconstructed = (
        dec.basal_contribution + dec.base_value + shap_total + dec.median_shift
    )

    np.testing.assert_allclose(
        reconstructed, p50, rtol=0.0, atol=1e-6 * float(np.max(dec.capacity))
    )
    np.testing.assert_allclose(dec.pred_p50, p50, rtol=0.0, atol=1e-9)
    _, beds_residual = additivity_residual(dec)
    assert beds_residual < 1e-6 * float(np.max(dec.capacity))


def test_conformal_margin_is_reported_separately_from_shap_point(
    panel_split: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame],
    fitted_model: OrthogonalTwoStageOccupancyForecaster,
) -> None:
    _, train_df, cal_df, eval_df = panel_split
    calibrator = ConformalFloodCalibrator(base_model=fitted_model)
    calibrator.fit_calibration(train_df=train_df, cal_df=cal_df)

    base_q = fitted_model.predict_quantiles(eval_df)
    conf_q = calibrator.predict_quantiles(eval_df)
    dec = decompose_two_stage_point(
        fitted_model, eval_df, background=train_df, conformal=calibrator
    )

    margins = dec.conformal_margins
    assert isinstance(margins, ConformalMargins)
    # P50 é ponto: a calibração conformal não desloca o ponto.
    np.testing.assert_allclose(dec.pred_p50, base_q["pred_p50"].to_numpy(), atol=1e-9)
    np.testing.assert_allclose(
        base_q["pred_p50"].to_numpy(), conf_q["pred_p50"].to_numpy(), atol=1e-9
    )
    # A margem conformal é exatamente a meia-largura adicionada a P90, fora do SHAP.
    high_90 = conf_q["pred_p90"].to_numpy() - base_q["pred_p90"].to_numpy()
    np.testing.assert_allclose(margins.high_90, high_90, atol=1e-12)
    assert np.all(margins.high_90 >= 0.0)
    # A soma SHAP continua explicando o ponto (P50), nunca P90.
    shap_total = np.sum(np.column_stack(list(dec.shap_values.values())), axis=1)
    np.testing.assert_allclose(
        dec.base_value + shap_total,
        dec.pred_p50 - dec.basal_contribution - dec.median_shift,
        atol=1e-6,
    )


def test_background_with_audit_year_is_rejected(
    panel_split: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame],
    fitted_model: OrthogonalTwoStageOccupancyForecaster,
) -> None:
    df, _, _, eval_df = panel_split
    with pytest.raises(Exception) as excinfo:
        decompose_two_stage_point(fitted_model, eval_df, background=df)
    assert "background" in str(excinfo.value).lower()


def test_decomposition_excludes_forbidden_ar_features(
    panel_split: tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame],
    fitted_model: OrthogonalTwoStageOccupancyForecaster,
) -> None:
    _, train_df, _, eval_df = panel_split
    dec = decompose_two_stage_point(fitted_model, eval_df, background=train_df)
    forbidden = {"y_origin", "y_roll4", "y_roll12", "y_dev_roll12", "y_seasonal_naive", "target_y"}
    assert set(dec.shap_values) == set(fitted_model.shock_cols)
    assert not (set(dec.shap_values) & forbidden)


def test_shap_is_imported_lazily() -> None:
    """C-07: `import shap_audit` não pode importar `shap` no topo do módulo."""
    src = Path(__file__).resolve().parents[2] / "src/previsao_leitos/audit/shap_audit.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "shap" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "shap"


def eval_df_p50(
    model: OrthogonalTwoStageOccupancyForecaster, df: pl.DataFrame
) -> np.ndarray:
    return model.predict_quantiles(df)["pred_p50"].to_numpy().astype(np.float64)
