"""Testes unitários do grafo dirigido de bacias (DAG montante -> jusante)."""

from __future__ import annotations

import polars as pl
import pytest

from previsao_leitos.domain import DataContractError
from previsao_leitos.geo.basin_graph import (
    RS_BASIN_EDGES,
    BasinTopology,
    map_rs_health_region_to_basin,
)


def test_basin_topology_rejects_cycle_as_invalid_dag() -> None:
    with pytest.raises(DataContractError, match="ciclo"):
        BasinTopology(edges=(("A", "B"), ("B", "C"), ("C", "A")))


def test_upstream_rain_propagates_downstream_but_downstream_never_alters_upstream() -> None:
    topo = BasinTopology(
        edges=(
            ("taquari_antas", "jacui_baixo"),
            ("jacui_baixo", "guaiba_capital"),
            ("guaiba_capital", "lagoa_patos_sul"),
        )
    )
    assert topo.upstream_of("guaiba_capital") == frozenset({"taquari_antas", "jacui_baixo"})
    assert topo.upstream_of("taquari_antas") == frozenset()

    weekly = pl.DataFrame(
        {
            "bacia_id": [
                "taquari_antas",
                "jacui_baixo",
                "guaiba_capital",
                "lagoa_patos_sul",
            ],
            "ano_epi": [2024, 2024, 2024, 2024],
            "semana_epi": [18, 18, 18, 18],
            "chuva_sem_mm": [300.0, 200.0, 50.0, 10.0],
        }
    )

    enriched = topo.propagate_upstream(
        weekly, basin_col="bacia_id", value_col="chuva_sem_mm", output_col="chuva_montante_mm"
    )
    by_basin = dict(
        zip(
            enriched["bacia_id"].to_list(),
            enriched["chuva_montante_mm"].to_list(),
            strict=True,
        )
    )

    # Serra/Taquari-Antas (cabeceira) não recebe chuva de jusante (0.0)
    assert by_basin["taquari_antas"] == pytest.approx(0.0)
    # Guaíba recebe média/soma ponderada da chuva a montante (Taquari 300 + Jacuí 200 -> média 250)
    assert by_basin["guaiba_capital"] == pytest.approx(250.0)
    # Lagoa dos Patos recebe montante de Taquari + Jacuí + Guaíba -> média (300+200+50)/3
    assert by_basin["lagoa_patos_sul"] == pytest.approx(550.0 / 3.0)


def test_rs_official_basin_edges_form_valid_acyclic_graph() -> None:
    topo = BasinTopology(edges=RS_BASIN_EDGES)
    assert "taquari_antas" in topo.upstream_of("guaiba_capital")
    assert "guaiba_capital" in topo.upstream_of("lagoa_patos_sul")
    assert "guaiba_capital" not in topo.upstream_of("taquari_antas")
    assert (
        map_rs_health_region_to_basin("Região 10 - Capital e Vale do Gravataí") == "guaiba_capital"
    )
    assert map_rs_health_region_to_basin("Região 21 - Sul") == "lagoa_patos_sul"
    assert map_rs_health_region_to_basin("Desconhecida") == "outras_bacias"


def test_propagate_upstream_weights_each_upstream_basin_equally_across_multiple_regions() -> None:
    topo = BasinTopology(edges=(("serra", "guaiba"), ("cai", "guaiba")))
    weekly = pl.DataFrame(
        {
            "regiao": ["R23", "R24", "R25", "R08", "R10"],
            "bacia_id": ["serra", "serra", "serra", "cai", "guaiba"],
            "ano_epi": [2024, 2024, 2024, 2024, 2024],
            "semana_epi": [18, 18, 18, 18, 18],
            "chuva_sem_mm": [300.0, 300.0, 300.0, 100.0, 50.0],
        }
    )
    enriched = topo.propagate_upstream(weekly)
    guaiba_val = enriched.filter(pl.col("bacia_id") == "guaiba")["chuva_montante_mm"][0]
    # Média equitativa entre as 2 bacias a montante: (300 + 100) / 2 = 200 (não (900+100)/4 = 250)
    assert guaiba_val == pytest.approx(200.0)
