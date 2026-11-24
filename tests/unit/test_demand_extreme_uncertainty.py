"""Testes unitários da Task 3.3: EVT-GPD, Mondrian Conformal e pooling hierárquico."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from previsao_leitos.demand.extreme_uncertainty import (
    ExtremeValueGpdTail,
    HierarchicalBasinBayesShrinkage,
    MondrianEnsoConformalCalibrator,
)
from previsao_leitos.demand.two_stage import OrthogonalTwoStageOccupancyForecaster


def test_evt_gpd_tail_fits_heavy_tail_and_is_monotone_in_return_prob() -> None:
    rng = np.random.default_rng(42)
    # Cauda pesada sintética (Pareto + ruído normal)
    samples = np.concatenate([rng.normal(0.0, 1.0, size=250), 2.0 + rng.pareto(2.5, size=60)])
    evt = ExtremeValueGpdTail(threshold_quantile=0.80).fit(samples)

    q90 = evt.return_level(0.90)
    q95 = evt.return_level(0.95)
    q99 = evt.return_level(0.99)
    assert q90 <= q95 <= q99
    assert q99 > float(np.quantile(samples, 0.80))


def test_mondrian_enso_conformal_preserves_interior_specificity_and_non_crossing() -> None:
    rng = np.random.default_rng(99)
    rows = []
    for yr in (2021, 2022, 2023, 2024):
        for ent, porte, scale in (("POA", "grande", 500.0), ("Interior_A", "interior", 20.0)):
            for wk in range(1, 31):
                enso = "el_nino" if yr in (2023, 2024) else "neutro"
                rain = float(rng.gamma(2.0, 20.0))
                noise = float(rng.normal(0.0, 0.08 * scale if enso == "neutro" else 0.14 * scale))
                y = max(0.0, scale + 0.2 * rain + noise)
                rows.append(
                    {
                        "nome_regiao_saude": ent,
                        "porte_regional": porte,
                        "enso_regime": enso,
                        "target_ano_epi": yr,
                        "target_semana_epi": wk,
                        "y_origin": scale,
                        "y_roll12": scale,
                        "capacidade_sem": scale,
                        "chuva_montante_pond_lag0": rain,
                        "target_y": y,
                    }
                )
    df = pl.DataFrame(rows)
    tr = df.filter(pl.col("target_ano_epi") <= 2022)
    cal = df.filter(pl.col("target_ano_epi") == 2023)
    ev = df.filter(pl.col("target_ano_epi") == 2024)

    base = OrthogonalTwoStageOccupancyForecaster(capacity_col="capacidade_sem").fit(
        tr, ["chuva_montante_pond_lag0"]
    )
    mondrian = MondrianEnsoConformalCalibrator(
        base_model=base,
        group_cols=("porte_regional", "enso_regime"),
        use_evt_p95=True,
    ).fit_calibration(cal_df=cal)

    preds = mondrian.predict_quantiles(ev)
    assert np.all(preds["pred_p10"].to_numpy() <= preds["pred_p50"].to_numpy() + 1e-9)
    assert np.all(preds["pred_p50"].to_numpy() <= preds["pred_p90"].to_numpy() + 1e-9)
    assert np.all(preds["pred_p90"].to_numpy() <= preds["pred_p95"].to_numpy() + 1e-9)

    # Largura da banda no interior deve ser muito menor que em POA (sem contaminação de escala)
    ev_pred = ev.with_columns(
        (preds["pred_p90"] - preds["pred_p10"]).alias("band_width")
    )
    w_poa = float(ev_pred.filter(pl.col("porte_regional") == "grande")["band_width"].mean())
    w_int = float(ev_pred.filter(pl.col("porte_regional") == "interior")["band_width"].mean())
    assert w_int < w_poa * 0.25


def test_hierarchical_basin_bayes_shrinkage_pools_small_hospital_toward_basin() -> None:
    df = pl.DataFrame(
        {
            "bacia_id": ["guaiba"] * 12 + ["taquari"] * 12,
            "entidade_id": (
                ["HCPA"] * 10
                + ["HospPequenoGuaiba"] * 2
                + ["HospLajeado"] * 10
                + ["HospPeqTaq"] * 2
            ),
            "efeito_choque": [
                0.30, 0.32, 0.28, 0.31, 0.29, 0.30, 0.33, 0.27, 0.31, 0.29,
                0.80, 0.80,  # Hospital pequeno ruidoso (n=2) puxado para média da bacia (~0.30)
                0.50, 0.52, 0.48, 0.51, 0.49, 0.50, 0.53, 0.47, 0.51, 0.49,
                0.10, 0.10,
            ],
        }
    )
    hier = HierarchicalBasinBayesShrinkage().fit(
        df, basin_col="bacia_id", entity_col="entidade_id", value_col="efeito_choque"
    )
    est_hcpa = hier.predict_entity("guaiba", "HCPA")
    est_small = hier.predict_entity("guaiba", "HospPequenoGuaiba")
    # HCPA (n=10) fica próximo de 0.30; HospPequenoGuaiba (n=2, média bruta 0.80) encolhe para bacia
    assert est_hcpa == pytest.approx(0.30, abs=0.05)
    assert est_small < 0.78
    assert est_small > est_hcpa
