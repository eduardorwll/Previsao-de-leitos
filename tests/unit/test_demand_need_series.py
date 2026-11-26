"""Testes da série de necessidade (pré-registro v1)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from previsao_leitos.demand.need_series import (
    NEED_SPECS,
    acceptance,
    build_need_series,
    check_monotonic_in_shock,
    validate_need_series,
)
from previsao_leitos.domain import DataContractError, LeakageViolationError


def _panel(
    *,
    years: tuple[int, ...] = tuple(range(2015, 2025)),
    shock_2024: float = 0.9,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for region, base in (("A", 100.0), ("B", 40.0)):
        for year in years:
            for week in range(1, 53):
                is_event = year == 2024 and 18 <= week <= 30
                admitted = base / 5.0 + (base / 2.0 if is_event else 0.0)
                occupancy = base + (base if is_event else 0.0)
                shock = shock_2024 if is_event else 0.05
                rows.append(
                    {
                        "nome_regiao_saude": region,
                        "bacia_id": f"Bacia_{region}",
                        "ano_epi": year,
                        "semana_epi": week,
                        "uti_adulto_diarias": occupancy,
                        "uti_adulto_internacoes": admitted,
                        "diarias_uti_total_ativas": occupancy * 1.1,
                        "clinico_pacientes_dia": occupancy * 2.0,
                        "clinico_internacoes": admitted * 2.0,
                        "respiratorio_pacientes_dia": occupancy * 1.5,
                        "respiratorio_internacoes": admitted * 1.5,
                        "prob_enchente_f03_oof": shock,
                    }
                )
    return pl.DataFrame(rows)


def test_need_is_never_below_baseline_and_is_zero_excess_without_shock() -> None:
    panel = _panel()
    quiet = panel.with_columns(pl.lit(0.0).alias("prob_enchente_f03_oof"))
    series = build_need_series(panel=quiet, spec=NEED_SPECS["uti_adulto"])
    assert series.height > 0
    assert series["need"].min() >= 0.0
    # Sem choque, a necessidade é o máximo entre ocupação e baseline (A7).
    expected = np.maximum(
        series["occupancy"].to_numpy(), series["baseline"].to_numpy()
    )
    assert np.allclose(series["need"].to_numpy(), expected, atol=1e-9)
    assert (series["need"] >= series["occupancy"] - 1e-9).all()


def test_shock_raises_need_monotonically() -> None:
    panel = _panel()
    low = build_need_series(
        panel=panel.with_columns(pl.lit(0.2).alias("prob_enchente_f03_oof")),
        spec=NEED_SPECS["uti_adulto"],
    )
    high = build_need_series(
        panel=panel.with_columns(pl.lit(0.8).alias("prob_enchente_f03_oof")),
        spec=NEED_SPECS["uti_adulto"],
    )
    joined = low.select(["nome_regiao_saude", "ano_epi", "semana_epi", "need"]).join(
        high.select(["nome_regiao_saude", "ano_epi", "semana_epi", "need"]),
        on=["nome_regiao_saude", "ano_epi", "semana_epi"],
        suffix="_high",
    )
    assert (joined["need_high"] >= joined["need"] - 1e-9).all()


def test_baseline_uses_only_prior_years() -> None:
    panel = _panel()
    seasonal = build_need_series(
        panel=panel, spec=NEED_SPECS["uti_adulto"], baseline_mode="seasonal"
    )
    # 2015-2016 não têm histórico suficiente (>= 2 anos anteriores) e são descartados.
    assert seasonal["ano_epi"].min() >= 2017
    row_2024 = seasonal.filter(
        (pl.col("ano_epi") == 2024) & (pl.col("semana_epi") == 10)
    ).head(1)
    assert row_2024.height == 1
    assert float(row_2024["baseline"][0]) == pytest.approx(100.0)  # mediana 2021-2023


def test_level_correction_tracks_a_shift() -> None:
    panel = _panel().with_columns(
        pl.when(pl.col("ano_epi") >= 2023)
        .then(pl.col("uti_adulto_diarias") * 1.3)
        .otherwise(pl.col("uti_adulto_diarias"))
        .alias("uti_adulto_diarias")
    )
    corrected = build_need_series(panel=panel, spec=NEED_SPECS["uti_adulto"])
    row = corrected.filter((pl.col("ano_epi") == 2024) & (pl.col("semana_epi") == 10)).head(1)
    # Patamar subiu 30% a partir de 2023 (dentro da banda de nível): fator acompanha.
    assert float(row["level_factor"][0]) == pytest.approx(1.3, rel=0.10)
    assert float(row["baseline"][0]) == pytest.approx(130.0, rel=0.10)


def test_holdout_year_is_rejected() -> None:
    panel = pl.concat(
        [_panel(), _panel().with_columns(pl.lit(2025, dtype=pl.Int64).alias("ano_epi"))]
    )
    with pytest.raises(LeakageViolationError):
        build_need_series(panel=panel, spec=NEED_SPECS["uti_adulto"])


def test_acceptance_reports_criteria() -> None:
    series = build_need_series(panel=_panel(), spec=NEED_SPECS["uti_adulto"])
    monotone = check_monotonic_in_shock(_panel(), spec=NEED_SPECS["uti_adulto"])
    ok, criteria = acceptance(series, monotone_in_shock=monotone)
    assert ok is True
    assert criteria["no_holdout"] is True
    assert criteria["non_negative"] is True
    assert float(criteria["event_excess_2024"]) > 0.0
    assert criteria["monotone_in_shock"] is True
    assert set(validate_need_series(series)) >= {
        "no_holdout",
        "non_negative",
        "above_baseline",
        "event_excess_2024",
        "diagnostic_corr_with_shock",
        "pre_event_level_ratio",
    }


def test_unknown_weighting_is_rejected() -> None:
    with pytest.raises(DataContractError):
        build_need_series(
            panel=_panel(), spec=NEED_SPECS["uti_adulto"], weighting="invalido"
        )
