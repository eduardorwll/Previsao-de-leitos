"""Testes unitários da Task 4.1: BSTS Pré-Choque com SIVEP-Gripe e do() recursivo."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from previsao_leitos.audit.bsts_counterfactual import (
    PreShockBstsCounterfactual,
    recursive_dynamic_do_intervention,
)
from previsao_leitos.demand.models import PooledRidgeDeltaForecaster


def test_preshock_bsts_separates_viral_srag_wave_from_flood_excess() -> None:
    rng = np.random.default_rng(7)
    rows = []
    for yr in (2020, 2021, 2022, 2023, 2024):
        for wk in range(1, 36):
            srag_viral = 60.0 if 23 <= wk <= 27 else 10.0
            season = 200.0 + 15.0 * np.sin(2.0 * np.pi * wk / 52.0)
            flood_excess = 120.0 if (yr == 2024 and 18 <= wk <= 22) else 0.0
            y = season + 0.8 * srag_viral + flood_excess + float(rng.normal(0.0, 4.0))
            rows.append(
                {
                    "nome_regiao_saude": "Região 10 - Capital",
                    "ano_epi": yr,
                    "semana_epi": wk,
                    "srag_viral_confirmado": srag_viral,
                    "target_y": y,
                }
            )
    panel = pl.DataFrame(rows)
    bsts = PreShockBstsCounterfactual(
        shock_year=2024,
        shock_start_week=18,
        control_cols=("srag_viral_confirmado",),
    ).fit(panel, entity_col="nome_regiao_saude", target_col="target_y")

    eval_2024 = panel.filter(pl.col("ano_epi") == 2024).rename(
        {"ano_epi": "target_ano_epi", "semana_epi": "target_semana_epi"}
    )
    cf = bsts.predict_counterfactual(eval_2024, entity_col="nome_regiao_saude")

    # Na janela de enchente (Sem. 18-22), excesso observado deve capturar ~+120
    flood_win = cf.filter(pl.col("target_semana_epi").is_between(18, 22))
    assert float(flood_win["excess_obs"].mean()) == pytest.approx(120.0, abs=15.0)

    # Na onda viral SRAG (Sem. 23-27), excesso atribuído à enchente deve ser ~0 (controlado!)
    viral_win = cf.filter(pl.col("target_semana_epi").is_between(23, 27))
    assert abs(float(viral_win["excess_obs"].mean())) < 15.0
    assert np.all(cf["y_cf_p05"].to_numpy() <= cf["y_cf_no_flood"].to_numpy())
    assert np.all(cf["y_cf_no_flood"].to_numpy() <= cf["y_cf_p95"].to_numpy())


def test_recursive_dynamic_do_propagates_counterfactual_y_origin_without_flood_leak() -> None:
    # Treina um modelo simples que depende de y_origin e chuva_montante_pond_lag0
    rows = []
    for yr in (2021, 2022):
        for wk in range(1, 21):
            rain = 150.0 if wk == 10 else 20.0
            y_orig = 100.0
            y = y_orig + 0.5 * (rain - 20.0)
            rows.append(
                {
                    "nome_regiao_saude": "POA",
                    "target_ano_epi": yr,
                    "target_semana_epi": wk,
                    "y_origin": y_orig,
                    "chuva_montante_pond_lag0": rain,
                    "oni_c_lag4": 0.0,
                    "target_y": y,
                    "target_delta": y - y_orig,
                }
            )
    train_df = pl.DataFrame(rows)
    model = PooledRidgeDeltaForecaster().fit(train_df, ["chuva_montante_pond_lag0", "oni_c_lag4"])

    # Em 2024, Sem. 18 tem enchente (chuva=300), Sem. 19 e 20 têm y_origin contaminado (240!)
    eval_df = pl.DataFrame(
        {
            "nome_regiao_saude": ["POA"] * 4,
            "target_ano_epi": [2024] * 4,
            "target_semana_epi": [17, 18, 19, 20],
            "y_origin": [100.0, 100.0, 240.0, 240.0],
            "chuva_montante_pond_lag0": [20.0, 300.0, 20.0, 20.0],
            "oni_c_lag4": [1.8, 1.8, 1.5, 1.2],
            "target_y": [100.0, 240.0, 220.0, 180.0],
        }
    )

    cf_df = recursive_dynamic_do_intervention(
        model=model,
        train_df=train_df,
        eval_df=eval_df,
        entity_col="nome_regiao_saude",
        horizon_weeks=1,
        intervention_start_week=18,
        climate_prefixes=("chuva_montante_pond",),
        neutral_zero_prefixes=("oni_c",),
    )

    # Sob do(Clima=Climatologia, ENSO=0), a Sem. 19 não pode herdar y_origin=240 da enchente!
    p50_cf_w19 = float(cf_df.filter(pl.col("target_semana_epi") == 19)["pred_p50_cf"][0])
    assert p50_cf_w19 < 150.0
