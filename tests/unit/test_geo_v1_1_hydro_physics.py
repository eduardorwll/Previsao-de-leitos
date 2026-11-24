"""Testes unitários da Task 2.3: convolução multi-salto, MapBiomas ponderado e hidro fina."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from previsao_leitos.geo.basin_graph import RS_BASIN_WEIGHTED_EDGES, BasinTopology
from previsao_leitos.geo.fine_hydro_sources import (
    aggregate_ana_hidroweb_weekly,
    aggregate_merge_cptec_weekly,
    aggregate_sar_flood_extent,
)
from previsao_leitos.geo.landfill_exposure import build_causal_landfill_exposure


def test_basin_topology_multi_hop_reaches_poa_and_lagoa_dos_patos() -> None:
    topo = BasinTopology.from_weighted_edges(RS_BASIN_WEIGHTED_EDGES)
    paths = topo.transitive_upstream_paths()

    poa_sources = {p.src for p in paths if p.dst == "guaiba_capital"}
    lagoa_sources = {p.src for p in paths if p.dst == "lagoa_patos_sul"}
    assert poa_sources == topo.upstream_of("guaiba_capital")
    assert lagoa_sources == topo.upstream_of("lagoa_patos_sul")
    assert "serra_taquari_antas" in poa_sources
    assert "serra_taquari_antas" in lagoa_sources
    assert "planalto_alto_jacui" in lagoa_sources

    # Chuva extrema exclusivamente na Serra (cabeceira) na Semana 18 deve chegar a POA (Sem. 18)
    # e à Lagoa dos Patos (Sem. 19 devido ao travel_weeks=1 do Guaíba -> Lagoa dos Patos)
    rows = []
    for bacia in (
        "serra_taquari_antas",
        "taquari_antas",
        "jacui_baixo",
        "guaiba_capital",
        "lagoa_patos_sul",
    ):
        for wk in (18, 19):
            chuva = 400.0 if (bacia == "serra_taquari_antas" and wk == 18) else 0.0
            rows.append(
                {"bacia_id": bacia, "ano_epi": 2024, "semana_epi": wk, "chuva_sem_mm": chuva}
            )
    weekly = pl.DataFrame(rows)
    out = topo.propagate_weighted_upstream(weekly, multi_hop=True)

    poa_w18 = out.filter((pl.col("bacia_id") == "guaiba_capital") & (pl.col("semana_epi") == 18))[
        "chuva_montante_pond"
    ][0]
    lagoa_w18 = out.filter(
        (pl.col("bacia_id") == "lagoa_patos_sul") & (pl.col("semana_epi") == 18)
    )["chuva_montante_pond"][0]
    lagoa_w19 = out.filter(
        (pl.col("bacia_id") == "lagoa_patos_sul") & (pl.col("semana_epi") == 19)
    )["chuva_montante_pond"][0]

    assert poa_w18 > 0.0
    assert lagoa_w18 == pytest.approx(0.0)
    assert lagoa_w19 > 0.0


def test_landfill_exposure_weighted_by_population_eliminates_maup_dilution() -> None:
    # Município A: urbano pequeno em pôlder (1.000 ha, 50% urbano, 30% banhado, 500k hab)
    # Município B: rural gigante (100.000 ha, 1% urbano, 1% banhado, 10k hab)
    mapbiomas = pl.DataFrame(
        {
            "id_municipio": ["4314902", "4314902", "4314902", "4300001", "4300001", "4300001"],
            "ano": [2022] * 6,
            "id_classe": ["24", "11", "3", "24", "11", "3"],
            "area_ha": [500.0, 300.0, 200.0, 1000.0, 1000.0, 98000.0],
        }
    )
    mun_to_reg = pl.DataFrame(
        {"id_municipio": ["4314902", "4300001"], "id_regiao_saude": ["R10", "R10"]}
    )
    mun_weights = pl.DataFrame(
        {"id_municipio": ["4314902", "4300001"], "peso_exposicao": [500_000.0, 10_000.0]}
    )

    unweighted = build_causal_landfill_exposure(mapbiomas, mun_to_reg=mun_to_reg, ref_year=2022)
    weighted = build_causal_landfill_exposure(
        mapbiomas, mun_to_reg=mun_to_reg, ref_year=2022, mun_weights=mun_weights
    )

    # Sem peso, área rural dilui frac_urbano para ~1.5%; com peso populacional, reflete ~49%!
    assert unweighted["frac_urbano"][0] < 0.05
    assert weighted["frac_urbano"][0] > 0.45
    assert weighted["indice_aterro_varzea"][0] > unweighted["indice_aterro_varzea"][0] * 10.0


def test_fine_hydro_sources_ana_merge_and_sar_aggregators() -> None:
    ana_df = pl.DataFrame(
        {
            "codigo_estacao": ["87450004", "87450004"],
            "bacia_id": ["guaiba_capital", "guaiba_capital"],
            "data_medicao": [date(2024, 5, 4), date(2024, 5, 5)],
            "cota_cm": [520.0, 535.0],
            "vazao_m3s": [25000.0, 28000.0],
            "cota_inundacao_cm": [300.0, 300.0],
        }
    )
    ana_wk = aggregate_ana_hidroweb_weekly(ana_df)
    assert ana_wk.height == 1
    assert ana_wk["cota_max_cm"][0] == pytest.approx(535.0)
    assert ana_wk["excedencia_cota_inundacao_cm"][0] == pytest.approx(235.0)

    merge_df = pl.DataFrame(
        {
            "id_municipio": ["4314902", "4314902"],
            "data": [date(2024, 5, 1), date(2024, 5, 2)],
            "chuva_mm": [110.0, 95.0],
        }
    )
    merge_wk = aggregate_merge_cptec_weekly(merge_df)
    assert merge_wk["chuva_merge_sem_mm"][0] == pytest.approx(205.0)
    assert merge_wk["chuva_merge_max_24h_mm"][0] == pytest.approx(110.0)

    sar_df = pl.DataFrame(
        {
            "id_municipio": ["4314902", "4304606"],
            "nome_regiao_saude": ["R10", "R10"],
            "area_urbana_km2": [100.0, 50.0],
            "area_urbana_inundada_km2": [25.0, 20.0],
            "hospitais_atingidos": [4, 2],
        }
    )
    sar_reg = aggregate_sar_flood_extent(sar_df)
    assert sar_reg["frac_area_urbana_inundada_sar"][0] == pytest.approx(45.0 / 150.0)
    assert sar_reg["hospitais_atingidos_sar"][0] == 6
