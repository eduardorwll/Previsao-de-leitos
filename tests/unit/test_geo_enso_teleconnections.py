"""Testes unitários da Task 2.4: conector causal de teleconexões El Niño / ENSO."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from previsao_leitos.geo.enso_teleconnections import (
    build_causal_enso_weekly_features,
    classify_enso_regime,
    parse_noaa_oni_table,
)


def test_classify_enso_regime_boundaries() -> None:
    assert classify_enso_regime(2.0) == "el_nino_forte"
    assert classify_enso_regime(0.8) == "el_nino"
    assert classify_enso_regime(0.1) == "neutro"
    assert classify_enso_regime(-0.9) == "la_nina"


def test_build_causal_enso_weekly_features_has_zero_future_leakage() -> None:
    # Série mensal sintética de 2022-01 a 2024-06
    dates = []
    oni = []
    tsa = []
    sallj = []
    for yr in (2022, 2023, 2024):
        for m in range(1, 13):
            if yr == 2024 and m > 6:
                continue
            dates.append(date(yr, m, 1))
            oni.append(-0.8 if yr == 2022 else (1.6 if yr == 2023 else 1.2))
            tsa.append(0.1 if yr == 2022 else 0.7)
            sallj.append(1.0 if yr == 2022 else 2.5)

    monthly_a = pl.DataFrame(
        {
            "data_ref": dates,
            "oni_c": oni,
            "nino34_anom_c": oni,
            "soi_std": [-v for v in oni],
            "mei_v2": oni,
            "tsa_anom_c": tsa,
            "sallj_proxy": sallj,
        }
    )
    # Perturba 2024-05 e 2024-06 (futuro) -> semanas até 2024-04 devem ficar bit-idênticas
    monthly_b = monthly_a.with_columns(
        pl.when(pl.col("data_ref") >= date(2024, 5, 1))
        .then(pl.lit(9.9))
        .otherwise(pl.col("oni_c"))
        .alias("oni_c")
    )

    wk_a = build_causal_enso_weekly_features(monthly_a, lags=(1, 4, 12, 24))
    wk_b = build_causal_enso_weekly_features(monthly_b, lags=(1, 4, 12, 24))

    assert "oni_c_lag4" in wk_a.columns
    assert "oni_c_lag24" in wk_a.columns
    assert "el_nino_intensity" in wk_a.columns
    assert "enso_regime" in wk_a.columns

    pre_a = wk_a.filter((pl.col("ano_epi") < 2024) | (pl.col("semana_epi") <= 16))
    pre_b = wk_b.filter((pl.col("ano_epi") < 2024) | (pl.col("semana_epi") <= 16))
    assert pre_a.equals(pre_b)

    # Intensidade El Niño é sempre >= 0
    assert float(wk_a["el_nino_intensity"].to_numpy().min()) >= 0.0


def test_parse_noaa_oni_table_reads_official_ascii_format() -> None:
    sample_noaa = (
        "SEAS YR TOTAL ANOM\n"
        "DJF 2023 26.0 -0.7\n"
        "JFM 2023 26.5 -0.4\n"
        "OND 2023 28.6  1.9\n"
        "AMJ 2024 27.8  0.4\n"
    )
    df = parse_noaa_oni_table(sample_noaa)
    assert df.height == 4
    assert df["data_ref"][0] == date(2023, 1, 1)
    assert df["oni_c"][2] == pytest.approx(1.9)
