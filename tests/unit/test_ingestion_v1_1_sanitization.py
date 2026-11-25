"""Testes unitários da Task 2.1: censo diário ativo, W_ref externa e CNES temporal."""

from __future__ import annotations

import polars as pl
import pytest

from previsao_leitos.domain import CodigoUF, DataContractError
from previsao_leitos.ingestion.pipeline import aggregate_regional_weekly_demand
from previsao_leitos.ingestion.queries import (
    build_sih_active_census_query,
    build_sih_external_referral_matrix_query,
)
from previsao_leitos.ingestion.schemas import (
    validate_sih_active_census_frame,
    validate_sih_external_referral_matrix_frame,
)


def test_sih_active_census_query_uses_date_expansion_and_params() -> None:
    spec = build_sih_active_census_query(uf=CodigoUF("43"), ano_inicio=2010, ano_fim=2026)
    assert spec.name == "sih_censo_ativo_semanal"
    assert spec.schema_version == "v2"
    assert "GENERATE_DATE_ARRAY" in spec.sql
    assert "@ano_inicio - 1 AND @ano_fim + 1" in spec.sql
    assert "n_dias_ativos - 1" in spec.sql
    assert "@uf_code" in spec.sql
    assert "@ano_inicio" in spec.sql
    assert "@ano_fim" in spec.sql
    assert "pacientes_dia_ativos" in spec.expected_columns
    assert "diarias_uti_ativas" in spec.expected_columns
    assert "ano_cmpt_max" in spec.expected_columns
    # Zero PII
    for forbidden in ("cpf", "cns", "nome_paciente", "data_nascimento"):
        assert forbidden not in spec.sql.lower()


def test_sih_external_referral_matrix_query_filters_self_loops_before_lgpd() -> None:
    spec = build_sih_external_referral_matrix_query(
        uf=CodigoUF("43"), ano_inicio=2010, train_end_year=2022
    )
    assert spec.name == "sih_w_ref_externa_treino"
    assert (
        "SUBSTR(id_municipio_paciente, 1, 6) != SUBSTR(id_municipio_estabelecimento, 1, 6)"
        in spec.sql
    )
    assert "@train_end_year" in spec.sql
    assert "HAVING COUNT(*) >= @min_cell_count" in spec.sql
    assert "internacoes_urgencia_total" in spec.expected_columns


def test_validate_sih_active_census_and_external_w_ref_frames() -> None:
    census_df = pl.DataFrame(
        {
            "id_estabelecimento_cnes": ["2237253"],
            "ano_epi": [2024],
            "semana_epi": [19],
            "especialidade": ["uti_adulto"],
            "grupo_cid": ["respiratorio"],
            "carater_atendimento": ["urgencia"],
            "internacoes_iniciadas": [12],
            "pacientes_dia_ativos": [84.0],
            "diarias_uti_ativas": [70.0],
            "ano_cmpt_max": [2024],
        }
    )
    validated_census = validate_sih_active_census_frame(census_df)
    assert validated_census["id_estabelecimento_cnes"][0] == "2237253"
    assert validated_census["pacientes_dia_ativos"][0] == pytest.approx(84.0)

    w_ref_df = pl.DataFrame(
        {
            "id_municipio_residencia_6d": ["430460"],
            "id_municipio_estabelecimento_6d": ["431490"],
            "id_estabelecimento_cnes": ["2237253"],
            "internacoes_urgencia_total": [420],
            "diarias_uti_total": [850],
        }
    )
    validated_w = validate_sih_external_referral_matrix_frame(w_ref_df)
    assert validated_w.height == 1

    # Auto-referência (origem == destino) deve ser rejeitada pelo contrato
    bad_w = w_ref_df.with_columns(pl.lit("431490").alias("id_municipio_residencia_6d"))
    with pytest.raises(DataContractError, match="auto-referência"):
        validate_sih_external_referral_matrix_frame(bad_w)


def test_aggregate_regional_weekly_demand_temporal_cnes_no_future_leakage() -> None:
    # Hospital 0000001 NÃO era hospital-escola em 2020, mas virou em 2024.
    # O join temporal não pode vazar is_hospital_ensino=True para 2020 (.any() bug).
    hospitals_df = pl.DataFrame(
        {
            "id_estabelecimento_cnes": ["0000001", "0000001"],
            "id_municipio": ["4314902", "4314902"],
            "ano": [2020, 2024],
            "mes": [6, 6],
            "is_hospital_ensino": [False, True],
        }
    )
    ibge_df = pl.DataFrame(
        {
            "id_municipio": ["4314902"],
            "nome_regiao_saude": ["Região 10 - Capital e Vale do Gravataí"],
        }
    )
    sih_df = pl.DataFrame(
        {
            "id_estabelecimento_cnes": ["0000001", "0000001", "9999999"],
            "ano_epi": [2020, 2024, 2024],
            "semana_epi": [10, 10, 10],
            "grupo_cid": ["respiratorio", "respiratorio", "respiratorio"],
            "internacoes": [20, 30, 15],
            "dias_permanencia_total": [100, 150, 60],
            "diarias_uti_total": [40, 60, 10],
        }
    )

    # Apenas hospitais-escola: 2020 excluído (False em 2020), 2024 mantido (True em 2024)
    he_only = aggregate_regional_weekly_demand(
        sih_df=sih_df,
        hospitals_df=hospitals_df,
        ibge_df=ibge_df,
        only_teaching_hospitals=True,
        min_cell_count=5,
    )
    assert he_only["ano_epi"].to_list() == [2024]
    assert he_only["internacoes"].to_list() == [30]

    # Todos os hospitais: 100% das AIHs preservadas (incluindo CNES 9999999 sem cadastro no recorte)
    all_hosp = aggregate_regional_weekly_demand(
        sih_df=sih_df,
        hospitals_df=hospitals_df,
        ibge_df=ibge_df,
        only_teaching_hospitals=False,
        min_cell_count=5,
    )
    assert int(all_hosp["internacoes"].sum()) == 65
