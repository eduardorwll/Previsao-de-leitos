"""Testes do estimador agregado de LOS com riscos competitivos (Task 2.2)."""

from __future__ import annotations

import polars as pl
import pytest

from previsao_leitos.demand.los import (
    AggregatedCompetingRisksLosEstimator,
    AggregatedCompetingRisksSummary,
)
from previsao_leitos.domain import DataContractError


def _risk_df() -> pl.DataFrame:
    # bucket, em_risco no início do bucket, altas e óbitos no bucket
    return pl.DataFrame(
        {
            "ano_epi": [2020, 2020, 2020, 2024],
            "bucket_dia_internacao": [1, 2, 3, 1],
            "em_risco": [100.0, 55.0, 22.0, 100.0],
            "alta_viva": [40.0, 30.0, 18.0, 99.0],
            "obito": [5.0, 3.0, 1.0, 0.0],
        }
    )


def test_causal_fit_excludes_future_years() -> None:
    est = AggregatedCompetingRisksLosEstimator.fit_causal(
        _risk_df(), train_end_year=2022
    )
    summary = est.summary
    assert isinstance(summary, AggregatedCompetingRisksSummary)
    # O bucket 1 de 2024 (alta=99) NÃO pode entrar: em_risco do bucket 1 = 100 (só 2020)
    assert summary.n_internacoes == 100
    assert len(summary.bucket_dias) == 3


def test_aalen_johansen_cif_and_mean_los() -> None:
    est = AggregatedCompetingRisksLosEstimator.fit_causal(
        _risk_df(), train_end_year=2022
    )
    s = est.summary
    # h_alta: 0.40, 30/55, 18/22 ; h_obito: 0.05, 3/55, 1/22
    assert s.cif_discharge[-1] == pytest.approx(0.88, abs=1e-9)
    assert s.cif_death[-1] == pytest.approx(0.09, abs=1e-9)
    # E[LOS] nao-restrito = sum(bucket * incidencia combinada) = 1.68
    assert est.mean_los_days == pytest.approx(1.68, abs=1e-9)
    # E[LOS | alta] = 1.54 / 0.88
    assert s.mean_los_discharge_days == pytest.approx(1.54 / 0.88, abs=1e-9)
    # E[LOS | obito] = 0.14 / 0.09
    assert s.mean_los_death_days == pytest.approx(0.14 / 0.09, abs=1e-9)


def test_overflow_hazards_are_normalized() -> None:
    # em_risco < alta+obito (dado inconsistente) -> normaliza para somar 1
    df = pl.DataFrame(
        {
            "ano_epi": [2020],
            "bucket_dia_internacao": [1],
            "em_risco": [10.0],
            "alta_viva": [8.0],
            "obito": [4.0],
        }
    )
    est = AggregatedCompetingRisksLosEstimator.fit_causal(df)
    s = est.summary
    assert s.hazard_discharge[0] + s.hazard_death[0] == pytest.approx(1.0)
    assert s.hazard_discharge[0] == pytest.approx(8.0 / 12.0)


def test_missing_columns_raise() -> None:
    with pytest.raises(DataContractError):
        AggregatedCompetingRisksLosEstimator.fit_causal(pl.DataFrame({"ano_epi": [2020]}))
