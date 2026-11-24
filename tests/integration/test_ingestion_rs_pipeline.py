"""Teste de integração offline da Trilha 1A (RS Hospitais-Escola) usando ParquetFixtureAdapter."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from previsao_leitos.domain import parse_codigo_uf
from previsao_leitos.ingestion import (
    IngestionConfig,
    ParquetFixtureAdapter,
    aggregate_regional_weekly_demand,
    extract_rs_core_health_pack,
)


def _write_rs_fixtures(fixtures_dir: Path) -> None:
    pl.DataFrame(
        {
            "id_estabelecimento_cnes": ["2237601", "2244306", "2252694", "2232103"],
            "id_municipio": ["4314902", "4316907", "4314407", "4315602"],
            "sigla_uf": ["RS", "RS", "RS", "RS"],
            "ano": [2024, 2024, 2024, 2024],
            "mes": [5, 5, 5, 5],
            "tipo_unidade": ["05", "05", "05", "05"],
            "is_hospital_ensino": [True, True, True, True],
        }
    ).write_parquet(fixtures_dir / "cnes_hospitais_escola.parquet")

    pl.DataFrame(
        {
            "id_estabelecimento_cnes": ["2237601", "2237601", "2244306"],
            "id_municipio": ["4314902", "4314902", "4316907"],
            "ano": [2024, 2024, 2024],
            "mes": [5, 5, 5],
            "categoria_leito": ["uti_adulto", "clinico", "uti_pediatrica"],
            "quantidade_existente": [120, 480, 22],
            "quantidade_sus": [110, 450, 22],
        }
    ).write_parquet(fixtures_dir / "cnes_leitos_especialidade.parquet")

    pl.DataFrame(
        {
            "id_estabelecimento_cnes": ["2237601", "2237601", "2237601"],
            "ano_epi": [2024, 2024, 2024],
            "semana_epi": [20, 20, 20],
            "especialidade": ["clinico", "clinico", "clinico"],
            "grupo_cid": ["leptospirose", "trauma", "respiratorio"],
            "carater_atendimento": ["urgencia", "urgencia", "urgencia"],
            "internacoes": [3, 8, 22],
            "dias_permanencia_total": [21, 40, 132],
            "diarias_uti_total": [6, 12, 30],
            "obitos": [1, 1, 2],
            "altas_vivas": [2, 7, 20],
        }
    ).write_parquet(fixtures_dir / "sih_demanda_semanal_agg.parquet")

    pl.DataFrame(
        {
            "id_municipio_residencia_6d": ["430460", "431490", "431690"],
            "id_estabelecimento_cnes": ["2237601", "2237601", "2244306"],
            "ano_epi": [2024, 2024, 2024],
            "semana_epi": [20, 20, 20],
            "internacoes": [4, 30, 15],
        }
    ).write_parquet(fixtures_dir / "sih_fluxos_referencia_semanal.parquet")

    pl.DataFrame(
        {
            "id_municipio": ["4314902", "4316907", "4314407", "4315602"],
            "nome": ["Porto Alegre", "Santa Maria", "Pelotas", "Rio Grande"],
            "sigla_uf": ["RS", "RS", "RS", "RS"],
            "ano": [2024, 2024, 2024, 2024],
            "populacao": [1332845.0, 271735.0, 325685.0, 191900.0],
        }
    ).write_parquet(fixtures_dir / "ibge_municipios_rs.parquet")


def test_rs_core_health_pack_runs_end_to_end_offline_with_cache_and_lgpd(tmp_path: Path) -> None:
    fixtures_dir = tmp_path / "fixtures"
    data_dir = tmp_path / "data"
    fixtures_dir.mkdir()
    _write_rs_fixtures(fixtures_dir)

    config = IngestionConfig(
        uf=parse_codigo_uf("43"),
        ano_inicio=2024,
        ano_fim=2024,
        data_dir=data_dir,
        max_mb_scanned=2000,
        min_cell_count=5,
    )
    adapter = ParquetFixtureAdapter(fixtures_dir=fixtures_dir)

    first_run = extract_rs_core_health_pack(source=adapter, config=config)
    second_run = extract_rs_core_health_pack(source=adapter, config=config)

    assert set(first_run.keys()) == {
        "cnes_hospitais_escola",
        "cnes_leitos_especialidade",
        "sih_demanda_semanal_agg",
        "sih_fluxos_referencia_semanal",
        "ibge_municipios_rs",
    }

    # Supressão primária (3 -> None) e complementar (8 -> None) no estrato HCPA semana 20
    sih_df = first_run["sih_demanda_semanal_agg"].dataframe
    assert sih_df["internacoes"].to_list() == [None, None, 22]
    assert first_run["sih_demanda_semanal_agg"].manifest.suppressed_cells >= 2
    assert first_run["sih_demanda_semanal_agg"].manifest.cache_hit is False
    assert second_run["sih_demanda_semanal_agg"].manifest.cache_hit is True


def test_regional_aggregation_preserves_wave2_signal_and_enforces_lgpd() -> None:
    sih_df = pl.DataFrame(
        {
            "id_estabelecimento_cnes": ["2237601", "2237253", "2237601", "2244306"],
            "ano_epi": [2024, 2024, 2024, 2024],
            "semana_epi": [21, 21, 21, 21],
            "especialidade": ["clinico", "clinico", "uti_adulto", "clinico"],
            "grupo_cid": ["leptospirose", "leptospirose", "respiratorio", "leptospirose"],
            "carater_atendimento": ["urgencia", "urgencia", "urgencia", "urgencia"],
            "internacoes": [3, 4, 20, 2],
            "dias_permanencia_total": [21, 28, 160, 14],
            "diarias_uti_total": [6, 8, 90, 0],
            "obitos": [0, 1, 2, 0],
            "altas_vivas": [3, 3, 18, 2],
        }
    )
    hosp_df = pl.DataFrame(
        {
            "id_estabelecimento_cnes": ["2237601", "2237253", "2244306"],
            "id_municipio": ["4314902", "4314902", "4316907"],
            "is_hospital_ensino": [True, True, True],
        }
    )
    ibge_df = pl.DataFrame(
        {
            "id_municipio": ["4314902", "4316907"],
            "nome_regiao_saude": ["Regiao 10 - Capital", "Regiao 01 - Verdes Campos"],
        }
    )

    regional = aggregate_regional_weekly_demand(
        sih_df=sih_df,
        hospitals_df=hosp_df,
        ibge_df=ibge_df,
        group_by_col="grupo_cid",
        only_teaching_hospitals=True,
        min_cell_count=5,
    )

    # Na Capital, 3 + 4 = 7 (>= 5) preserva sinal de leptospirose; em Verdes Campos, 2 < 5 -> None
    cap_lepto = regional.filter(
        (pl.col("nome_regiao_saude") == "Regiao 10 - Capital")
        & (pl.col("grupo_cid") == "leptospirose")
    )["internacoes"].to_list()
    sm_lepto = regional.filter(
        (pl.col("nome_regiao_saude") == "Regiao 01 - Verdes Campos")
        & (pl.col("grupo_cid") == "leptospirose")
    )["internacoes"].to_list()
    assert cap_lepto == [7]
    assert sm_lepto == [None]
