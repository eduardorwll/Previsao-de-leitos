"""Testes unitários da Task 3.4: Chain-Ladder Nowcasting e LOS de riscos competitivos."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from previsao_leitos.demand.nowcasting import (
    ChainLadderNowcaster,
    CompetingRisksLosEstimator,
)


def test_chain_ladder_nowcaster_corrects_right_censoring_drop() -> None:
    # Triângulo histórico de atraso (lags 0..4 semanas):
    # lag 0 captura ~10%, lag 1 ~45%, lag 2 ~75%, lag 3 ~92%, lag 4 ~100%
    cum_fracs = [0.10, 0.45, 0.75, 0.92, 1.00]
    rows = []
    for wk in range(1, 21):
        true_total = 1000.0 + 20.0 * (wk % 4)
        for d, f in enumerate(cum_fracs):
            rows.append(
                {
                    "event_week": wk,
                    "delay_weeks": d,
                    "cumulative_count": true_total * f,
                }
            )
    triangle_df = pl.DataFrame(rows)
    nowcaster = ChainLadderNowcaster(max_delay_weeks=4).fit(triangle_df)

    assert nowcaster.completeness_fraction(0) == pytest.approx(0.10, abs=0.02)
    assert nowcaster.completeness_fraction(4) == pytest.approx(1.00, abs=0.02)

    # Semana 52 censurada: observou apenas 100 de 1000 (lag=0, queda de -90%)
    recent = pl.DataFrame(
        {
            "event_week": [49, 50, 51, 52],
            "delay_weeks": [3, 2, 1, 0],
            "observed_count": [920.0, 750.0, 450.0, 100.0],
        }
    )
    corrected = nowcaster.nowcast(recent)
    assert corrected["nowcast_p50"][3] == pytest.approx(1000.0, rel=0.05)
    assert np.all(corrected["nowcast_p10"].to_numpy() <= corrected["nowcast_p50"].to_numpy())
    assert np.all(corrected["nowcast_p50"].to_numpy() <= corrected["nowcast_p90"].to_numpy())


def test_competing_risks_los_estimator_separates_discharge_and_death_cifs() -> None:
    df = pl.DataFrame(
        {
            "ano_epi": [2021] * 10 + [2024] * 2,
            "especialidade": ["uti_adulto"] * 12,
            "los_days": [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 40, 50],
            "outcome": [
                "alta_viva",
                "alta_viva",
                "obito",
                "alta_viva",
                "alta_viva",
                "obito",
                "alta_viva",
                "alta_viva",
                "obito",
                "alta_viva",
                "obito",  # 2024: não deve entrar no treino causal <= 2022
                "obito",
            ],
        }
    )
    est = CompetingRisksLosEstimator(train_end_year=2022).fit(df)
    summary = est.summarize("uti_adulto", horizon_days=14)

    # Apenas as 10 internações de 2021 entram (7 altas vivas, 3 óbitos; 2024 ignorado)
    assert summary.cif_discharge + summary.cif_death <= 1.0 + 1e-9
    assert summary.cif_discharge == pytest.approx(0.6, abs=0.15)
    assert summary.cif_death == pytest.approx(0.3, abs=0.10)
    assert summary.mean_los_discharge < 15.0
