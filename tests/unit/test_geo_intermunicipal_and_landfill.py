"""Testes Prove-It F02.1: roteamento ponderado com tempo de trânsito, remanso, aterro e W_ref."""

from __future__ import annotations

import polars as pl
import pytest

from previsao_leitos.domain import DataContractError
from previsao_leitos.geo.basin_graph import (
    RS_BASIN_WEIGHTED_EDGES,
    BasinTopology,
    WeightedBasinEdge,
    compute_estuarine_backwater_index,
)
from previsao_leitos.geo.landfill_exposure import (
    RS_POLDER_MUNICIPALITIES,
    build_causal_landfill_exposure,
)
from previsao_leitos.geo.referral_graph import (
    build_causal_referral_matrix,
    compute_catchment_flood_pressure,
)


def test_weighted_basin_routing_applies_weights_and_travel_lag_without_future_leak() -> None:
    topo = BasinTopology.from_weighted_edges(
        (
            WeightedBasinEdge("taquari_antas", "guaiba_capital", weight=0.75, travel_weeks=0),
            WeightedBasinEdge("cai", "guaiba_capital", weight=0.25, travel_weeks=0),
            WeightedBasinEdge("guaiba_capital", "lagoa_patos_sul", weight=1.0, travel_weeks=1),
        )
    )
    weekly = pl.DataFrame(
        {
            "bacia_id": [
                "taquari_antas",
                "cai",
                "guaiba_capital",
                "lagoa_patos_sul",
                "taquari_antas",
                "cai",
                "guaiba_capital",
                "lagoa_patos_sul",
            ],
            "ano_epi": [2024] * 8,
            "semana_epi": [18, 18, 18, 18, 19, 19, 19, 19],
            "chuva_sem_mm": [400.0, 100.0, 200.0, 10.0, 20.0, 20.0, 30.0, 15.0],
        }
    )

    out = topo.propagate_weighted_upstream(
        weekly, basin_col="bacia_id", value_col="chuva_sem_mm", output_col="chuva_montante_pond"
    )
    sem18_guaiba = out.filter(
        (pl.col("bacia_id") == "guaiba_capital") & (pl.col("semana_epi") == 18)
    )["chuva_montante_pond"][0]
    # 0.75 * 400 + 0.25 * 100 = 325.0 (ponderado, não média simples 250.0)
    assert sem18_guaiba == pytest.approx(325.0)

    sem19_pelotas = out.filter(
        (pl.col("bacia_id") == "lagoa_patos_sul") & (pl.col("semana_epi") == 19)
    )["chuva_montante_pond"][0]
    # Onda da semana 18 no Guaíba (200.0) chega em Pelotas/Lagoa dos Patos na semana 19 (lag=1)!
    assert sem19_pelotas == pytest.approx(200.0)

    # Cabeceira nunca recebe chuva de jusante
    sem18_taquari = out.filter(
        (pl.col("bacia_id") == "taquari_antas") & (pl.col("semana_epi") == 18)
    )["chuva_montante_pond"][0]
    assert sem18_taquari == pytest.approx(0.0)


def test_weighted_basin_rejects_negative_travel_lag_and_rs_edges_are_acyclic() -> None:
    with pytest.raises(DataContractError, match="negativo"):
        WeightedBasinEdge("A", "B", weight=1.0, travel_weeks=-1)

    topo = BasinTopology.from_weighted_edges(RS_BASIN_WEIGHTED_EDGES)
    assert "guaiba_capital" in topo.upstream_of("lagoa_patos_sul")
    assert "taquari_antas" in topo.upstream_of("guaiba_capital")


def test_estuarine_backwater_only_activates_under_south_wind_in_estuarine_basins() -> None:
    df = pl.DataFrame(
        {
            "bacia_id": [
                "guaiba_capital",
                "guaiba_capital",
                "lagoa_patos_sul",
                "serra_taquari_antas",
            ],
            "vento_sul_ms": [6.0, -4.0, 5.0, 8.0],
            "api_montante_max": [100.0, 100.0, 80.0, 120.0],
        }
    )
    out = compute_estuarine_backwater_index(df)
    vals = out["remanso_lagunar_idx"].to_list()
    # Guaíba com vento Sul (+6 m/s) -> remanso ativo (6 * 100 = 600)
    assert vals[0] == pytest.approx(600.0)
    # Guaíba com vento Norte (-4 m/s) -> sem remanso (0.0)
    assert vals[1] == pytest.approx(0.0)
    # Lagoa dos Patos (Pelotas/RG) com vento Sul (+5 m/s) -> remanso ativo (400)
    assert vals[2] == pytest.approx(400.0)
    # Serra/cabeceira com vento Sul (+8 m/s) -> sem remanso estuarino (0.0)
    assert vals[3] == pytest.approx(0.0)


def test_causal_referral_graph_ignores_2024_flows_and_propagates_neighbor_flood_pressure() -> None:
    flows_base = pl.DataFrame(
        {
            "id_municipio_residencia_6d": ["430460", "431490", "430460", "430510"],
            "id_estabelecimento_cnes": ["2237601", "2237601", "2237601", "2223538"],
            "ano_epi": [2022, 2022, 2024, 2022],
            "semana_epi": [10, 10, 20, 10],
            "internacoes": [40, 60, 99999, 100],
        }
    )
    # Perturba 2024 drasticamente — matriz congelada em <= 2022 deve ficar bit-idêntica!
    flows_pert = flows_base.with_columns(
        pl.when(pl.col("ano_epi") >= 2024)
        .then(pl.col("internacoes") * 0)
        .otherwise(pl.col("internacoes"))
        .alias("internacoes")
    )
    mun_to_reg = pl.DataFrame(
        {
            "id_municipio": ["4304606", "4314902", "4305108"],
            "id_regiao_saude": ["43008", "43010", "43023"],
        }
    )
    cnes_to_reg = pl.DataFrame(
        {
            "id_estabelecimento_cnes": ["2237601", "2223538"],
            "id_regiao_saude": ["43010", "43023"],
        }
    )

    w_base = build_causal_referral_matrix(
        flows_base, cnes_to_reg=cnes_to_reg, mun_to_reg=mun_to_reg, max_year=2022
    )
    w_pert = build_causal_referral_matrix(
        flows_pert, cnes_to_reg=cnes_to_reg, mun_to_reg=mun_to_reg, max_year=2022
    )
    assert w_base.equals(w_pert)

    # Na Região 10 (POA), 40% dos pacientes vêm da Região 08 (Canoas/Caí) e 60% da própria 10
    weekly = pl.DataFrame(
        {
            "id_regiao_saude": ["43008", "43010", "43023"],
            "ano_epi": [2024, 2024, 2024],
            "semana_epi": [18, 18, 18],
            "chuva_sem_mm": [300.0, 50.0, 10.0],
        }
    )
    enriched = compute_catchment_flood_pressure(
        weekly, w_base, value_col="chuva_sem_mm", output_col="chuva_catchment_mm"
    )
    poa_val = enriched.filter(pl.col("id_regiao_saude") == "43010")["chuva_catchment_mm"][0]
    # 0.4 * 300 + 0.6 * 50 = 150.0 (enxerga a chuva extrema na cidade vizinha que envia pacientes!)
    assert poa_val == pytest.approx(150.0)


def test_causal_landfill_exposure_is_leak_free_and_ranks_floodplains_higher() -> None:
    mapbiomas = pl.DataFrame(
        {
            "ano": [1985, 1985, 1985, 2022, 2022, 2022, 2024, 1985, 1985, 2022, 2022],
            "id_municipio": [
                "4314902",
                "4314902",
                "4314902",
                "4314902",
                "4314902",
                "4314902",
                "4314902",
                "4305108",
                "4305108",
                "4305108",
                "4305108",
            ],
            "id_classe": ["24", "11", "3", "24", "11", "3", "24", "24", "3", "24", "3"],
            "area_ha": [
                13000.0,
                4200.0,
                11000.0,
                19000.0,
                3500.0,
                10000.0,
                999999.0,  # 2024 não pode vazar para ref_year=2022!
                4000.0,
                90000.0,
                9000.0,
                75000.0,
            ],
        }
    )
    mun_to_reg = pl.DataFrame(
        {
            "id_municipio": ["4314902", "4305108"],
            "id_regiao_saude": ["43010", "43023"],
        }
    )

    exp_a = build_causal_landfill_exposure(mapbiomas, mun_to_reg=mun_to_reg, ref_year=2022)
    exp_b = build_causal_landfill_exposure(
        mapbiomas.filter(pl.col("ano") <= 2022), mun_to_reg=mun_to_reg, ref_year=2022
    )
    assert exp_a.equals(exp_b)
    assert "4314902" in RS_POLDER_MUNICIPALITIES
    assert "4314407" in RS_POLDER_MUNICIPALITIES  # Pelotas

    poa = exp_a.filter(pl.col("id_regiao_saude") == "43010")["indice_aterro_varzea"][0]
    caxias = exp_a.filter(pl.col("id_regiao_saude") == "43023")["indice_aterro_varzea"][0]
    assert poa > caxias


def test_referral_and_landfill_reject_empty_inputs_gracefully() -> None:
    empty_flows = pl.DataFrame(
        schema={
            "id_municipio_residencia_6d": pl.String,
            "id_estabelecimento_cnes": pl.String,
            "ano_epi": pl.Int32,
            "semana_epi": pl.Int8,
            "internacoes": pl.Int64,
        }
    )
    empty_map = pl.DataFrame(schema={"id_municipio": pl.String, "id_regiao_saude": pl.String})
    empty_cnes = pl.DataFrame(
        schema={"id_estabelecimento_cnes": pl.String, "id_regiao_saude": pl.String}
    )
    with pytest.raises(DataContractError, match="vazia"):
        build_causal_referral_matrix(
            empty_flows, cnes_to_reg=empty_cnes, mun_to_reg=empty_map, max_year=2022
        )

    out = compute_catchment_flood_pressure(
        pl.DataFrame(
            {"id_regiao_saude": ["43010"], "ano_epi": [2024], "semana_epi": [18], "x": [10.0]}
        ),
        pl.DataFrame(
            schema={
                "id_regiao_origem": pl.String,
                "id_regiao_destino": pl.String,
                "peso_referencia": pl.Float64,
            }
        ),
        value_col="x",
    )
    assert out["chuva_catchment_mm"][0] == pytest.approx(0.0)
