"""Testes unitários da Task 2.2: GERINT-POA, SINAN (1os sintomas) e SIVEP-Gripe."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from previsao_leitos.domain import DataContractError
from previsao_leitos.ingestion.epidemiology_sources import (
    aggregate_gerint_poa_weekly,
    aggregate_sinan_leptospirose_weekly,
    aggregate_sivep_gripe_weekly,
)


def test_aggregate_gerint_poa_weekly_computes_unmet_queue_and_blocks_pii() -> None:
    raw = pl.DataFrame(
        {
            "data_solicitacao": [date(2024, 5, 6)] * 8 + [date(2024, 5, 7)] * 4,
            "tipo_leito": ["uti_adulto"] * 12,
            "tempo_espera_horas": [
                4.0, 8.0, 12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 6.0, 10.0, 20.0, 40.0
            ],
            "internado": [
                True, True, True, True, True, True, False, False, True, True, False, False
            ],
        }
    )
    out = aggregate_gerint_poa_weekly(raw, min_cell_count=5)
    assert out.height == 1
    assert out["ano_epi"][0] == 2024
    assert out["semana_epi"][0] == 19
    assert out["solicitacoes_totais"][0] == 12
    assert out["fila_reprimida_nao_internada"][0] == 4
    assert out["razao_demanda_reprimida"][0] == pytest.approx(4.0 / 12.0)
    assert out["tempo_espera_p90_h"][0] >= out["tempo_espera_mediano_h"][0]

    # Bloqueio LGPD de PII
    with_pii = raw.with_columns(pl.lit("12345678901").alias("cpf_paciente"))
    with pytest.raises(DataContractError, match="PII"):
        aggregate_gerint_poa_weekly(with_pii)


def test_aggregate_sinan_leptospirose_uses_symptom_onset_date_and_lgpd() -> None:
    raw = pl.DataFrame(
        {
            "data_primeiros_sintomas": [date(2024, 5, 13)] * 7 + [date(2024, 5, 20)] * 2,
            "nome_regiao_saude": ["Região 10 - Capital e Vale do Gravataí"] * 9,
            "confirmado": [True] * 6 + [False] + [True, True],
            "hospitalizado": [True] * 5 + [False, False, True, False],
        }
    )
    out = aggregate_sinan_leptospirose_weekly(raw, min_cell_count=5)
    assert out.height == 2
    sem20 = out.filter(pl.col("semana_epi") == 20)
    sem21 = out.filter(pl.col("semana_epi") == 21)
    assert sem20["casos_notificados"][0] == 7
    assert sem20["casos_confirmados"][0] == 6
    # Semana 21 tem n=2 (<5): suprimida por LGPD
    assert sem21["casos_notificados"][0] is None


def test_aggregate_sivep_gripe_separates_viral_srag_wave() -> None:
    raw = pl.DataFrame(
        {
            "data_primeiros_sintomas": [date(2024, 6, 17)] * 10,
            "nome_regiao_saude": ["Região 10 - Capital e Vale do Gravataí"] * 10,
            "uti": [True] * 6 + [False] * 4,
            "viral_confirmado": [True] * 8 + [False] * 2,
        }
    )
    out = aggregate_sivep_gripe_weekly(raw, min_cell_count=5)
    assert out.height == 1
    assert out["semana_epi"][0] == 25
    assert out["srag_total"][0] == 10
    assert out["srag_uti"][0] == 6
    assert out["srag_viral_confirmado"][0] == 8
    assert out["frac_srag_viral"][0] == pytest.approx(0.8)
