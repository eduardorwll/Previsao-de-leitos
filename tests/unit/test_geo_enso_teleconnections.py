"""Testes unitários da Task 2.4: conector causal de teleconexões El Niño / ENSO."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from previsao_leitos.geo.enso_teleconnections import (
    build_causal_bom_rnino_weekly,
    build_causal_enso_weekly_features,
    build_causal_roni_weekly_features,
    build_causal_soi_weekly_features,
    classify_enso_regime,
    parse_bom_rnino_table,
    parse_noaa_oni_table,
    parse_noaa_roni_table,
    parse_noaa_soi_table,
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


def test_parse_noaa_roni_table_reads_official_ascii_format() -> None:
    sample_roni = (
        "SEAS   YR  ANOM\n"
        "DJF  2023 -0.86\n"
        "JFM  2023 -0.68\n"
        "OND  2023  1.42\n"
        "AMJ  2024 -0.28\n"
    )
    df = parse_noaa_roni_table(sample_roni)
    assert df.height == 4
    assert df["data_ref"][0] == date(2023, 1, 1)
    assert df["roni_c"][2] == pytest.approx(1.42)


def test_build_causal_roni_weekly_features_regime_lags_and_zero_future_leakage() -> None:
    dates: list[date] = []
    roni: list[float] = []
    for yr in (2022, 2023, 2024):
        for m in range(1, 13):
            if yr == 2024 and m > 6:
                continue
            dates.append(date(yr, m, 1))
            roni.append(-0.9 if yr == 2022 else (1.4 if yr == 2023 else 0.6))

    monthly_a = pl.DataFrame({"data_ref": dates, "roni_c": roni})
    # Perturba 2024-05 e 2024-06 (futuro) -> semanas até 2024-04 devem ficar bit-idênticas
    monthly_b = monthly_a.with_columns(
        pl.when(pl.col("data_ref") >= date(2024, 5, 1))
        .then(pl.lit(9.9))
        .otherwise(pl.col("roni_c"))
        .alias("roni_c")
    )

    wk_a = build_causal_roni_weekly_features(monthly_a, lags=(4, 12))
    wk_b = build_causal_roni_weekly_features(monthly_b, lags=(4, 12))

    assert "roni_c" in wk_a.columns
    assert "roni_regime" in wk_a.columns
    assert "roni_c_lag4" in wk_a.columns
    assert "roni_c_lag12" in wk_a.columns

    pre_a = wk_a.filter((pl.col("ano_epi") < 2024) | (pl.col("semana_epi") <= 16))
    pre_b = wk_b.filter((pl.col("ano_epi") < 2024) | (pl.col("semana_epi") <= 16))
    assert pre_a.equals(pre_b)

    # Classificação de regime reutiliza os limiares ±0,5 / ±1,5 do CPC.
    regimes = wk_a["roni_regime"].unique().to_list()
    assert "la_nina" in regimes  # 2022 (roni ~ -0.9)
    assert "el_nino" in regimes  # 2023 (roni ~ 1.4)


def test_parse_noaa_soi_table_reads_standardized_block_and_skips_missing() -> None:
    sample_soi = (
        "(STAND TAHITI - STAND DARWIN)  SEA LEVEL PRESS\n"
        "                    ANOMALY\n"
        "YEAR   JAN   FEB   MAR   APR   MAY   JUN   JUL   AUG   SEP   OCT   NOV   DEC\n"
        "1951   2.5   1.5  -0.2  -0.5  -1.1   0.3  -1.7  -0.4  -1.8  -1.6  -1.3  -1.2\n"
        "(STAND TAHITI - STAND DARWIN)  SEA LEVEL PRESS\n"
        "                    STANDARDIZED    DATA\n"
        "YEAR   JAN   FEB   MAR   APR   MAY   JUN   JUL   AUG   SEP   OCT   NOV   DEC\n"
        "1951   1.5   0.9  -0.1  -0.3  -0.7   0.2  -1.0  -0.2  -1.1  -1.0  -0.8  -0.7\n"
        "1952  -0.9  -0.6   0.5  -0.2   0.8   0.7   0.5   0.1  -0.2   0.4   0.0  -1.2\n"
        "2024   0.5  -1.4   0.4  -0.2   0.5   0.0  -0.7   0.9  -0.1   0.5   0.5   1.2\n"
        "2026   1.1   1.4   1.2  -0.6  -0.9  -1.4  -2.4  -1.1  -2.0  -1.1-999.9-999.9\n"
    )
    df = parse_noaa_soi_table(sample_soi)
    # 1951 (12) + 1952 (12) + 2024 (12) + 2026 (10 válidos; Nov/Dez = -999.9 ignorados)
    assert df.height == 46
    assert df["data_ref"][0] == date(1951, 1, 1)
    assert df["soi_std"][0] == pytest.approx(1.5)  # bloco STANDARDIZED, não ANOMALY (2.5)
    assert df["data_ref"][-1] == date(2026, 10, 1)
    assert float(df["soi_std"][-1]) == pytest.approx(-1.1)
    assert (df["soi_std"] <= -999.0).sum() == 0


def test_build_causal_soi_weekly_features_lags_and_zero_future_leakage() -> None:
    dates: list[date] = []
    soi: list[float] = []
    for yr in (2022, 2023, 2024):
        for m in range(1, 13):
            if yr == 2024 and m > 6:
                continue
            dates.append(date(yr, m, 1))
            soi.append(0.5 if yr == 2022 else (-1.0 if yr == 2023 else 1.2))

    monthly_a = pl.DataFrame({"data_ref": dates, "soi_std": soi})
    monthly_b = monthly_a.with_columns(
        pl.when(pl.col("data_ref") >= date(2024, 5, 1))
        .then(pl.lit(9.9))
        .otherwise(pl.col("soi_std"))
        .alias("soi_std")
    )

    wk_a = build_causal_soi_weekly_features(monthly_a, lags=(4, 12))
    wk_b = build_causal_soi_weekly_features(monthly_b, lags=(4, 12))

    assert "soi_std" in wk_a.columns
    assert "soi_std_lag4" in wk_a.columns
    assert "soi_std_lag12" in wk_a.columns

    pre_a = wk_a.filter((pl.col("ano_epi") < 2024) | (pl.col("semana_epi") <= 16))
    pre_b = wk_b.filter((pl.col("ano_epi") < 2024) | (pl.col("semana_epi") <= 16))
    assert pre_a.equals(pre_b)


def test_parse_bom_rnino_table_reads_csv_format() -> None:
    sample = (
        "20080728,20080803,0.20\n"
        "20080804,20080810,0.22\n"
        "20080811,20080817,-0.26\n"
    )
    df = parse_bom_rnino_table(sample)
    assert df.height == 3
    assert df["week_start"][0] == date(2008, 7, 28)
    assert df["week_end"][0] == date(2008, 8, 3)
    assert df["bom_rnino"][0] == pytest.approx(0.20)
    assert df["bom_rnino"][2] == pytest.approx(-0.26)


def test_build_causal_bom_rnino_weekly_applies_one_week_lag_and_iso_mapping() -> None:
    df = pl.DataFrame(
        {
            "week_start": [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15)],
            "week_end": [date(2024, 1, 7), date(2024, 1, 14), date(2024, 1, 21)],
            "bom_rnino": [1.0, 2.0, 3.0],
        }
    )
    wk = build_causal_bom_rnino_weekly(df)
    assert wk.columns == ["ano_epi", "semana_epi", "bom_rnino"]
    # 2024-01-01 é segunda-feira -> semana ISO 1 de 2024
    assert wk["ano_epi"][0] == 2024
    assert wk["semana_epi"][0] == 1
    # lag de 1 semana: semana 1 fica nula; semana 2 recebe o valor da semana 1; etc.
    assert wk["bom_rnino"].to_list() == [None, 1.0, 2.0]
