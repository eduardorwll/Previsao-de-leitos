"""Testes TDD e Prove-It anti-vazamento da Task 1.2.1 (BigQuery v3 + Painel RS v3)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from previsao_leitos.demand.nowcasting import ChainLadderNowcaster
from previsao_leitos.domain import CodigoUF, DataContractError
from previsao_leitos.geo.v3_panel import build_rs_v3_panel_from_frames
from previsao_leitos.ingestion import (
    IngestionConfig,
    ParquetFixtureAdapter,
    build_sih_active_census_query,
    build_sih_delay_triangle_query,
    build_sih_external_referral_matrix_query,
    extract_rs_v3_sih_pack,
    validate_sih_delay_triangle_frame,
)


def test_v3_sih_queries_eliminate_double_count_add_border_margin_and_delay_triangle() -> None:
    uf = CodigoUF("43")
    census = build_sih_active_census_query(uf=uf, ano_inicio=2010, ano_fim=2025)
    assert census.schema_version == "v2"
    assert "DATE_DIFF(data_saida, data_internacao, DAY) + 1" not in census.sql
    assert "WHERE ano BETWEEN @ano_inicio - 1 AND @ano_fim + 1" in census.sql
    assert "WHERE ano_epi BETWEEN @ano_inicio AND @ano_fim" in census.sql
    assert "STARTS_WITH(id_municipio_estabelecimento, @uf_code)" in census.sql

    w_ref = build_sih_external_referral_matrix_query(uf=uf, ano_inicio=2010, train_end_year=2022)
    assert w_ref.schema_version == "v2"
    assert "WHERE ano BETWEEN @ano_inicio AND @train_end_year + 1" in w_ref.sql
    assert "BETWEEN @ano_inicio AND @train_end_year" in w_ref.sql

    tri = build_sih_delay_triangle_query(
        uf=uf, ano_inicio=2020, ano_fim=2026, max_delay_weeks=12
    )
    assert tri.name == "sih_delay_triangle"
    assert tri.schema_version == "v1"
    for p in tri.params:
        assert f"@{p.name}" in tri.sql
    assert "cumulative_count" in tri.expected_columns
    assert "SUM(incremental_count) OVER" in tri.sql
    for pii in ("cpf", "cns", "nome_paciente", "data_nascimento"):
        assert pii not in tri.sql.lower()


def test_validate_sih_delay_triangle_and_chain_ladder_integration() -> None:
    df = pl.DataFrame(
        {
            "ano_epi": [2024, 2024, 2024, 2024, 2024, 2024],
            "semana_epi": [10, 10, 10, 11, 11, 11],
            "delay_weeks": [0, 1, 2, 0, 1, 2],
            "incremental_count": [50, 30, 20, 40, 35, 25],
            "cumulative_count": [50, 80, 100, 40, 75, 100],
        }
    )
    validated = validate_sih_delay_triangle_frame(df)
    assert validated.height == 6

    # Compatível diretamente com ChainLadderNowcaster
    tri_for_cl = validated.with_columns(
        (pl.col("ano_epi") * 100 + pl.col("semana_epi")).alias("event_week")
    )
    nc = ChainLadderNowcaster(max_delay_weeks=2).fit(tri_for_cl)
    assert 0.35 <= nc.completeness_fraction(0) <= 0.55
    assert nc.completeness_fraction(2) == pytest.approx(1.0)

    # Contrato rejeita cumulative < incremental e semana fora de 1..53
    bad_cum = df.with_columns(pl.lit(10).alias("cumulative_count"))
    with pytest.raises(DataContractError, match="cumulative_count"):
        validate_sih_delay_triangle_frame(bad_cum)

    bad_wk = df.with_columns(pl.lit(54).alias("semana_epi"))
    with pytest.raises(DataContractError, match="semana_epi"):
        validate_sih_delay_triangle_frame(bad_wk)


def test_extract_rs_v3_sih_pack_offline_with_fixtures(tmp_path: Path) -> None:
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    pl.DataFrame(
        {
            "id_estabelecimento_cnes": ["2237253"],
            "ano_epi": [2024],
            "semana_epi": [19],
            "especialidade": ["uti_adulto"],
            "grupo_cid": ["respiratorio"],
            "carater_atendimento": ["urgencia"],
            "internacoes_iniciadas": [12],
            "pacientes_dia_ativos": [70.0],
            "diarias_uti_ativas": [56.0],
            "ano_cmpt_max": [2024],
        }
    ).write_parquet(fixtures_dir / "sih_censo_ativo_semanal.parquet")
    pl.DataFrame(
        {
            "id_municipio_residencia_6d": ["430460"],
            "id_municipio_estabelecimento_6d": ["431490"],
            "id_estabelecimento_cnes": ["2237253"],
            "internacoes_urgencia_total": [250],
            "diarias_uti_total": [500],
        }
    ).write_parquet(fixtures_dir / "sih_w_ref_externa_treino.parquet")
    pl.DataFrame(
        {
            "ano_epi": [2024, 2024],
            "semana_epi": [19, 19],
            "delay_weeks": [0, 1],
            "incremental_count": [100, 25],
            "cumulative_count": [100, 125],
        }
    ).write_parquet(fixtures_dir / "sih_delay_triangle.parquet")

    cfg = IngestionConfig(
        uf=CodigoUF("43"),
        ano_inicio=2010,
        ano_fim=2025,
        data_dir=tmp_path / "data",
        max_mb_scanned=2000,
    )
    res = extract_rs_v3_sih_pack(source=ParquetFixtureAdapter(fixtures_dir), config=cfg)
    assert set(res.keys()) == {
        "sih_censo_ativo_semanal",
        "sih_w_ref_externa_treino",
        "sih_delay_triangle",
    }
    assert res["sih_censo_ativo_semanal"].dataframe["diarias_uti_ativas"][0] == pytest.approx(56.0)
    assert res["sih_delay_triangle"].dataframe["cumulative_count"][-1] == 125


def _build_synthetic_v3_inputs(*, future_multiplier: float = 1.0) -> dict[str, pl.DataFrame]:
    weekly_rows = []
    for yr in (2021, 2022, 2024):
        for wk in (1, 2, 18, 19):
            mult = future_multiplier if yr >= 2024 else 1.0
            for rid, rname in (
                ("10", "Região 10 - Capital e Vale do Gravataí"),
                ("29", "Região 29 - Vales e Montanhas"),
            ):
                weekly_rows.append(
                    {
                        "id_regiao_saude": rid,
                        "nome_regiao_saude": rname,
                        "ano_epi": yr,
                        "semana_epi": wk,
                        "chuva_sem_mm": (40.0 if wk < 18 else 180.0) * mult,
                        "api_max_sem": (25.0 if wk < 18 else 120.0) * mult,
                        "vento_sul_max_sem": 6.0 * mult,
                    }
                )
    weekly_hydro = pl.DataFrame(weekly_rows)

    sih_rows = []
    for yr in (2021, 2022, 2024):
        mult = future_multiplier if yr >= 2024 else 1.0
        for wk in (1, 2, 18, 19):
            sih_rows.append(
                {
                    "id_estabelecimento_cnes": "2237253",
                    "ano_epi": yr,
                    "semana_epi": wk,
                    "especialidade": "uti_adulto",
                    "grupo_cid": "respiratorio",
                    "carater_atendimento": "urgencia",
                    "internacoes_iniciadas": int(10 * mult),
                    "pacientes_dia_ativos": 70.0 * mult,
                    "diarias_uti_ativas": 50.0 * mult,
                    "ano_cmpt_max": yr,
                }
            )
    sih_census = pl.DataFrame(sih_rows)

    # Hospital 2237253 vira hospital-escola apenas em 2024 (não pode vazar para 2021/2022)
    cnes_hospitals = pl.DataFrame(
        {
            "id_estabelecimento_cnes": ["2237253", "2237253", "2237253"],
            "id_municipio": ["4314902", "4314902", "4314902"],
            "ano": [2021, 2022, 2024],
            "mes": [1, 1, 1],
            "is_hospital_ensino": [False, False, True],
        }
    )
    cnes_beds = pl.DataFrame(
        {
            "id_estabelecimento_cnes": ["2237253", "2237253", "2237253"],
            "id_municipio": ["4314902", "4314902", "4314902"],
            "ano": [2021, 2022, 2024],
            "mes": [1, 1, 1],
            "categoria_leito": ["uti_adulto", "uti_adulto", "uti_adulto"],
            "quantidade_existente": [12, 12, int(20 * future_multiplier)],
            "quantidade_sus": [10, 10, int(18 * future_multiplier)],
        }
    )
    ibge = pl.DataFrame(
        {
            "id_municipio": ["4314902", "4304606", "4311403", "4314902"],
            "ano": [2022, 2022, 2022, 2024],
            "populacao": [1_400_000.0, 350_000.0, 90_000.0, 9_999_999.0 * future_multiplier],
            "id_regiao_saude": ["10", "10", "29", "10"],
            "nome_regiao_saude": [
                "Região 10 - Capital e Vale do Gravataí",
                "Região 10 - Capital e Vale do Gravataí",
                "Região 29 - Vales e Montanhas",
                "Região 10 - Capital e Vale do Gravataí",
            ],
        }
    )
    mapbiomas = pl.DataFrame(
        {
            "ano": [2000, 2022, 2022, 2022, 2024, 2024],
            "id_municipio": ["4314902", "4314902", "4304606", "4311403", "4314902", "4311403"],
            "id_classe": ["24", "24", "11", "3", "24", "24"],
            "area_ha": [10000.0, 15000.0, 5000.0, 20000.0, 999999.0 * future_multiplier, 999999.0],
        }
    )
    w_ref_flows = pl.DataFrame(
        {
            "id_municipio_residencia_6d": ["431140"],
            "id_municipio_estabelecimento_6d": ["431490"],
            "id_estabelecimento_cnes": ["2237253"],
            "internacoes_urgencia_total": [120],
            "diarias_uti_total": [300],
        }
    )
    return {
        "weekly_hydro": weekly_hydro,
        "sih_census": sih_census,
        "cnes_hospitals": cnes_hospitals,
        "cnes_beds": cnes_beds,
        "ibge": ibge,
        "mapbiomas": mapbiomas,
        "w_ref_flows": w_ref_flows,
    }


def test_prove_it_v3_panel_zero_future_leakage_and_anti_maup_and_cnes_monthly() -> None:
    """Prove-It: perturbar 2024+ deixa 100% das linhas <= 2022 bit-a-bit idênticas."""
    inp_a = _build_synthetic_v3_inputs(future_multiplier=1.0)
    inp_b = _build_synthetic_v3_inputs(future_multiplier=7.5)

    p_a = build_rs_v3_panel_from_frames(**inp_a, train_end_year=2022, min_year=2021, max_year=2024)
    p_b = build_rs_v3_panel_from_frames(**inp_b, train_end_year=2022, min_year=2021, max_year=2024)

    tr_a = p_a.filter(pl.col("ano_epi") <= 2022)
    tr_b = p_b.filter(pl.col("ano_epi") <= 2022)
    assert tr_a.equals(tr_b), "Vazamento temporal detectado: perturbação em 2024 alterou <= 2022!"

    # Hospital 2237253 só virou hospital-escola em 2024: hu_uti_adulto_diarias deve ser 0 em <= 2022
    r10_2022 = p_a.filter((pl.col("id_regiao_saude") == "10") & (pl.col("ano_epi") == 2022))
    r10_2024 = p_a.filter((pl.col("id_regiao_saude") == "10") & (pl.col("ano_epi") == 2024))
    assert float(r10_2022["hu_uti_adulto_diarias"].sum()) == pytest.approx(0.0)
    assert float(r10_2024["hu_uti_adulto_diarias"].sum()) > 0.0

    # CNES mensal causal: 10 leitos em 2022 e 18 leitos em 2024
    assert r10_2022["cnes_uti_sus_leitos"][0] == pytest.approx(10.0)
    assert r10_2024["cnes_uti_sus_leitos"][0] == pytest.approx(18.0)

    # Anti-MAUP: frac_urbano da Região 10 ponderado pela população (1.4M urbano vs 350k banhado)
    # deve ser 1.4M / (1.4M + 350k) = 0.80 (vs 0.75 sem peso por área)
    assert r10_2022["frac_urbano"][0] == pytest.approx(0.8, rel=1e-4)

    # W_ref externa inter-regional (exclude_self_region=True) propaga chuva da Região 29 -> 10
    assert float(r10_2022["chuva_catchment_mm"].sum()) > 0.0
    # E ignora auto-referência intra-regional (municípios distintos na mesma Região 10)
    inp_intra = _build_synthetic_v3_inputs(future_multiplier=1.0)
    inp_intra["w_ref_flows"] = pl.DataFrame(
        {
            "id_municipio_residencia_6d": ["430460", "431140"],
            "id_municipio_estabelecimento_6d": ["431490", "431490"],
            "id_estabelecimento_cnes": ["2237253", "2237253"],
            "internacoes_urgencia_total": [900, 100],
            "diarias_uti_total": [1800, 200],
        }
    )
    p_intra = build_rs_v3_panel_from_frames(
        **inp_intra, train_end_year=2022, min_year=2021, max_year=2024
    )
    r10_intra = p_intra.filter((pl.col("id_regiao_saude") == "10") & (pl.col("ano_epi") == 2022))
    assert r10_intra["chuva_catchment_mm"].to_list() == pytest.approx(
        r10_2022["chuva_catchment_mm"].to_list()
    )
