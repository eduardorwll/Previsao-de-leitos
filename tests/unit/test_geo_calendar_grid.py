"""Testes Prove-It de grade calendário completa, lacunas e API causal (zero compressão temporal)."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from previsao_leitos.domain import DataContractError
from previsao_leitos.geo.calendar_grid import (
    aggregate_daily_to_iso_weekly,
    build_complete_daily_grid,
    build_ex_ante_distributed_lags,
    compute_calendar_hydro_features,
)


def test_complete_daily_grid_expands_ten_day_gap_as_null_not_zero() -> None:
    raw = pl.DataFrame(
        {
            "id_municipio": ["4314902", "4314902"],
            "data": [date(2024, 5, 1), date(2024, 5, 12)],
            "precip_dia_mm": [50.0, 80.0],
        }
    )

    grid = build_complete_daily_grid(
        raw, entity_col="id_municipio", date_col="data", value_cols=("precip_dia_mm",)
    )

    assert grid.height == 12
    # Os 10 dias ausentes entre 02/05 e 11/05 devem ser null (jamais 0.0 mm)
    assert grid["precip_dia_mm"].null_count() == 10
    assert grid.filter(pl.col("data") == date(2024, 5, 1))["precip_dia_mm"][0] == 50.0
    assert grid.filter(pl.col("data") == date(2024, 5, 12))["precip_dia_mm"][0] == 80.0


def test_calendar_hydro_features_api_is_strictly_causal_and_respects_gaps() -> None:
    raw = pl.DataFrame(
        {
            "id_municipio": ["4314902", "4314902", "4314902"],
            "data": [date(2024, 5, 1), date(2024, 5, 2), date(2024, 5, 5)],
            "precip_dia_mm": [100.0, 20.0, 10.0],
        }
    )
    grid = build_complete_daily_grid(
        raw, entity_col="id_municipio", date_col="data", value_cols=("precip_dia_mm",)
    )
    feats = compute_calendar_hydro_features(grid, entity_col="id_municipio", k_decay=0.5)

    # Dia 1: API inicial = 0.0 (chuva do próprio dia 1 NÃO entra em api[dia 1])
    # Dia 2: 0.5 * 0 + 100 = 100.0
    # Dia 3 (gap null): decai 0.5 * 100 + 20 = 70.0
    # Dia 4 (gap null): decai 0.5 * 70 + 0 (sem nova leitura) = 35.0
    # Dia 5: decai 0.5 * 35 = 17.5 (se o gap fosse comprimido, teria sido 70.0!)
    api_vals = feats["api_indice_saturacao"].to_list()
    assert api_vals[0] == pytest.approx(0.0)
    assert api_vals[1] == pytest.approx(100.0)
    assert api_vals[2] == pytest.approx(70.0)
    assert api_vals[3] == pytest.approx(35.0)
    assert api_vals[4] == pytest.approx(17.5)

    # Janela de 3 dias no dia 5 (dias 3, 4, 5): apenas dia 5 tem leitura (10.0), não soma dia 1/2!
    assert feats["precip_acum_3d"][4] == pytest.approx(10.0)
    assert feats["dias_obs_15d"][4] == 3


def test_iso_weekly_aggregation_and_distributed_lags_preserve_week_spacing() -> None:
    # 3 semanas: sem 18 (100mm), sem 19 ausente (gap!), sem 20 (20mm)
    daily = pl.DataFrame(
        {
            "id_municipio": ["4314902", "4314902"],
            "data": [date(2024, 5, 1), date(2024, 5, 15)],  # Sem 18 e Sem 20 de 2024
            "precip_dia_mm": [100.0, 20.0],
        }
    )
    grid = build_complete_daily_grid(
        daily, entity_col="id_municipio", date_col="data", value_cols=("precip_dia_mm",)
    )
    feats = compute_calendar_hydro_features(grid, entity_col="id_municipio")
    weekly = aggregate_daily_to_iso_weekly(feats, entity_col="id_municipio")
    lagged = build_ex_ante_distributed_lags(
        weekly,
        entity_col="id_municipio",
        feature_cols=("chuva_sem_mm",),
        lag_weeks=(0, 1, 2),
    )

    sem20 = lagged.filter((pl.col("ano_epi") == 2024) & (pl.col("semana_epi") == 20))
    assert sem20["chuva_sem_mm_lag0"][0] == pytest.approx(20.0)
    # Sem 19 teve 0 dias observados -> chuva_sem_mm é null (não 0.0!)
    assert sem20["chuva_sem_mm_lag1"][0] is None
    # Sem 18 (2 semanas antes) = 100.0 mm
    assert sem20["chuva_sem_mm_lag2"][0] == pytest.approx(100.0)


def test_calendar_grid_rejects_duplicates_invalid_decay_and_negative_lags() -> None:
    dup = pl.DataFrame(
        {
            "id_municipio": ["4314902", "4314902"],
            "data": [date(2024, 5, 1), date(2024, 5, 1)],
            "precip_dia_mm": [10.0, 20.0],
        }
    )
    with pytest.raises(DataContractError, match="Duplicatas"):
        build_complete_daily_grid(dup, entity_col="id_municipio")

    valid = dup.head(1)
    with pytest.raises(DataContractError, match="k_decay"):
        compute_calendar_hydro_features(valid, entity_col="id_municipio", k_decay=1.5)

    weekly = pl.DataFrame(
        {"id_municipio": ["4314902"], "ano_epi": [2024], "semana_epi": [18], "x": [1.0]}
    )
    with pytest.raises(DataContractError, match="Lag negativo"):
        build_ex_ante_distributed_lags(
            weekly, entity_col="id_municipio", feature_cols=("x",), lag_weeks=(-1, 0)
        )
