"""Construtor canônico causal do Painel Semanal RS v3 (Censo Ativo, CNES Mensal, Anti-MAUP).

Integra de forma estritamente causal (`<= train_end_year` para artefatos estáticos/estruturais
e `join_asof(..., strategy="backward")` para atributos temporais):
- Censo ativo semanal SIH-SUS (`sih_censo_ativo_semanal` sem double-count `+1`);
- Capacidade mensal CNES por especialidade (`cnes_leitos_especialidade`);
- Matriz de referência externa pré-LGPD (`sih_w_ref_externa_treino`);
- Exposição MapBiomas ponderada por população (`mun_weights`, eliminando diluição MAUP);
- Propagação multi-salto nas bacias hidrográficas do RS (`multi_hop=True`).

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import polars as pl

from previsao_leitos.domain import DataContractError
from previsao_leitos.geo.basin_graph import (
    RS_BASIN_WEIGHTED_EDGES,
    BasinTopology,
    compute_estuarine_backwater_index,
    map_rs_health_region_to_basin,
)
from previsao_leitos.geo.landfill_exposure import build_causal_landfill_exposure
from previsao_leitos.geo.referral_graph import (
    build_causal_referral_matrix,
    compute_catchment_flood_pressure,
)


def _compute_causal_cnes_hospital_map(
    *,
    cnes_hospitals: pl.DataFrame,
    mun_to_reg: pl.DataFrame,
    sih_census: pl.DataFrame,
) -> pl.DataFrame:
    """Mapeia cada linha do censo SIH para (id_regiao_saude, is_hospital_ensino) sem vazamento."""
    hosp_yearly = (
        cnes_hospitals.with_columns(pl.col("ano").cast(pl.Int32).alias("ano_cnes"))
        .group_by(["id_estabelecimento_cnes", "ano_cnes"])
        .agg(
            pl.col("id_municipio").last().alias("id_municipio"),
            pl.col("is_hospital_ensino").any().alias("is_hospital_ensino"),
        )
        .sort(["id_estabelecimento_cnes", "ano_cnes"])
    )
    earliest_mun = (
        hosp_yearly.group_by("id_estabelecimento_cnes")
        .agg(pl.col("id_municipio").first().alias("_fallback_mun"))
    )

    sih_sorted = sih_census.with_columns(pl.col("ano_epi").cast(pl.Int32)).sort(
        ["id_estabelecimento_cnes", "ano_epi"]
    )
    joined = (
        sih_sorted.join_asof(
            hosp_yearly,
            left_on="ano_epi",
            right_on="ano_cnes",
            by="id_estabelecimento_cnes",
            strategy="backward",
            check_sortedness=False,
        )
        .drop("ano_cnes")
        .join(earliest_mun, on="id_estabelecimento_cnes", how="left")
        .with_columns(
            pl.coalesce([pl.col("id_municipio"), pl.col("_fallback_mun")]).alias("id_municipio"),
            pl.col("is_hospital_ensino").fill_null(False),
        )
        .drop("_fallback_mun")
        .join(
            mun_to_reg.select(["id_municipio", "id_regiao_saude"]).unique(subset=["id_municipio"]),
            on="id_municipio",
            how="inner",
        )
    )
    return joined


def _build_monthly_cnes_capacity_by_region(
    *,
    cnes_beds: pl.DataFrame,
    mun_to_reg: pl.DataFrame,
) -> pl.DataFrame:
    """Agrega leitos mensais CNES (UTI Adulto SUS/Existente e Total SUS) por Região de Saúde."""
    return (
        cnes_beds.join(
            mun_to_reg.select(["id_municipio", "id_regiao_saude"]).unique(subset=["id_municipio"]),
            on="id_municipio",
            how="inner",
        )
        .group_by(["id_regiao_saude", "ano", "mes"])
        .agg(
            pl.col("quantidade_sus")
            .filter(pl.col("categoria_leito") == "uti_adulto")
            .sum()
            .cast(pl.Float64)
            .alias("cnes_uti_sus_leitos"),
            pl.col("quantidade_existente")
            .filter(pl.col("categoria_leito") == "uti_adulto")
            .sum()
            .cast(pl.Float64)
            .alias("cnes_uti_exist_leitos"),
            pl.col("quantidade_sus").sum().cast(pl.Float64).alias("cnes_total_sus_leitos"),
        )
        .with_columns(
            (pl.col("ano").cast(pl.Int64) * 100 + pl.col("mes").cast(pl.Int64)).alias("_ym")
        )
        .sort(["id_regiao_saude", "_ym"])
        .select(
            [
                "id_regiao_saude",
                "_ym",
                "cnes_uti_sus_leitos",
                "cnes_uti_exist_leitos",
                "cnes_total_sus_leitos",
            ]
        )
    )


def build_rs_v3_panel_from_frames(
    *,
    weekly_hydro: pl.DataFrame,
    sih_census: pl.DataFrame,
    cnes_hospitals: pl.DataFrame,
    cnes_beds: pl.DataFrame,
    ibge: pl.DataFrame,
    mapbiomas: pl.DataFrame,
    w_ref_flows: pl.DataFrame,
    train_end_year: int = 2022,
    min_year: int = 2010,
    max_year: int = 2025,
) -> pl.DataFrame:
    """Constrói o painel semanal canônico RS v3 de forma pura e determinística (sem vazamento).

    Garante invariante Prove-It: perturbações em qualquer tabela para anos `> train_end_year`
    deixam todas as linhas do painel com `ano_epi <= train_end_year` bit-a-bit idênticas.
    """
    if weekly_hydro.is_empty():
        raise DataContractError("weekly_hydro não pode ser vazio.")
    if sih_census.is_empty():
        raise DataContractError("sih_census não pode ser vazio.")

    mun_to_reg = (
        ibge.select(["id_municipio", "id_regiao_saude", "nome_regiao_saude"])
        .unique(subset=["id_municipio"])
    )

    # 1. Pesos municipais estritamente causais (população <= train_end_year) para anti-MAUP (G5)
    ibge_causal = ibge.filter(pl.col("ano") <= train_end_year)
    if ibge_causal.is_empty():
        raise DataContractError(f"Sem dados IBGE disponíveis até train_end_year={train_end_year}.")
    mun_weights = (
        ibge_causal.sort("ano")
        .group_by("id_municipio")
        .agg(pl.col("populacao").last().fill_null(1.0).alias("peso_exposicao"))
    )

    landfill_weighted = build_causal_landfill_exposure(
        mapbiomas,
        mun_to_reg=mun_to_reg,
        ref_year=train_end_year,
        mun_weights=mun_weights,
    )
    panel = weekly_hydro.join(landfill_weighted, on="id_regiao_saude", how="left")

    # 2. Propagação hidrológica multi-salto (F02.2) + remanso estuarino + exposição em aterros
    if "bacia_id" not in panel.columns:
        panel = panel.with_columns(
            pl.col("nome_regiao_saude")
            .map_elements(map_rs_health_region_to_basin, return_dtype=pl.String)
            .alias("bacia_id")
        )
    topo = BasinTopology.from_weighted_edges(RS_BASIN_WEIGHTED_EDGES)
    panel = topo.propagate_weighted_upstream(
        panel,
        basin_col="bacia_id",
        value_col="chuva_sem_mm",
        output_col="chuva_montante_multihop",
        multi_hop=True,
    )
    panel = topo.propagate_weighted_upstream(
        panel,
        basin_col="bacia_id",
        value_col="api_max_sem",
        output_col="api_montante_multihop",
        multi_hop=True,
    )
    panel = panel.with_columns(
        pl.col("chuva_montante_multihop").alias("chuva_montante_pond"),
        pl.col("api_montante_multihop").alias("api_montante_pond"),
    )
    panel = compute_estuarine_backwater_index(
        panel,
        basin_col="bacia_id",
        south_wind_col="vento_sul_max_sem",
        storage_col="api_montante_multihop",
        output_col="remanso_lagunar_idx",
    ).with_columns(
        (
            pl.col("indice_aterro_varzea")
            * (pl.col("api_max_sem") + pl.col("api_montante_multihop"))
        ).alias("exposicao_aterro_ativa")
    )

    # 3. Grafo assistencial causal W_ref (estritamente <= train_end_year, externo pré-LGPD)
    cnes_train_reg = (
        cnes_hospitals.filter(pl.col("ano") <= train_end_year)
        .sort(["ano", "mes"])
        .group_by("id_estabelecimento_cnes")
        .agg(pl.col("id_municipio").last().alias("id_municipio"))
        .join(
            mun_to_reg.select(["id_municipio", "id_regiao_saude"]).unique(subset=["id_municipio"]),
            on="id_municipio",
            how="inner",
        )
    )
    w_ref = build_causal_referral_matrix(
        w_ref_flows,
        cnes_to_reg=cnes_train_reg,
        mun_to_reg=mun_to_reg,
        max_year=train_end_year,
        exclude_self_region=True,
    )
    panel = compute_catchment_flood_pressure(
        panel,
        w_ref,
        entity_col="id_regiao_saude",
        value_col="chuva_sem_mm",
        output_col="chuva_catchment_mm",
    )
    panel = compute_catchment_flood_pressure(
        panel,
        w_ref,
        entity_col="id_regiao_saude",
        value_col="api_max_sem",
        output_col="api_catchment_max",
    )

    # 4. Censo ativo semanal SIH-SUS agregado por Região de Saúde (join CNES causal <= ano_epi)
    sih_reg = _compute_causal_cnes_hospital_map(
        cnes_hospitals=cnes_hospitals,
        mun_to_reg=mun_to_reg,
        sih_census=sih_census,
    )
    demand_agg = sih_reg.group_by(["id_regiao_saude", "ano_epi", "semana_epi"]).agg(
        pl.col("internacoes_iniciadas").sum().cast(pl.Float64).alias("internacoes_total"),
        pl.col("pacientes_dia_ativos").sum().cast(pl.Float64).alias("pacientes_dia_ativos_total"),
        pl.col("internacoes_iniciadas")
        .filter(pl.col("especialidade") == "uti_adulto")
        .sum()
        .cast(pl.Float64)
        .alias("uti_adulto_internacoes"),
        pl.col("diarias_uti_ativas")
        .filter(pl.col("especialidade") == "uti_adulto")
        .sum()
        .cast(pl.Float64)
        .alias("uti_adulto_diarias"),
        pl.col("diarias_uti_ativas").sum().cast(pl.Float64).alias("diarias_uti_total_ativas"),
        pl.col("diarias_uti_ativas")
        .filter((pl.col("especialidade") == "uti_adulto") & pl.col("is_hospital_ensino"))
        .sum()
        .cast(pl.Float64)
        .alias("hu_uti_adulto_diarias"),
        pl.col("internacoes_iniciadas")
        .filter(pl.col("especialidade") == "clinico")
        .sum()
        .cast(pl.Float64)
        .alias("clinico_internacoes"),
        pl.col("pacientes_dia_ativos")
        .filter(pl.col("especialidade") == "clinico")
        .sum()
        .cast(pl.Float64)
        .alias("clinico_pacientes_dia"),
        pl.col("internacoes_iniciadas")
        .filter(pl.col("grupo_cid") == "respiratorio")
        .sum()
        .cast(pl.Float64)
        .alias("respiratorio_internacoes"),
        pl.col("pacientes_dia_ativos")
        .filter(pl.col("grupo_cid") == "respiratorio")
        .sum()
        .cast(pl.Float64)
        .alias("respiratorio_pacientes_dia"),
        pl.col("internacoes_iniciadas")
        .filter(pl.col("grupo_cid") == "leptospirose")
        .sum()
        .cast(pl.Float64)
        .alias("leptospirose_internacoes"),
        pl.col("internacoes_iniciadas")
        .filter(pl.col("grupo_cid") == "trauma")
        .sum()
        .cast(pl.Float64)
        .alias("trauma_internacoes"),
        pl.col("ano_cmpt_max").max().cast(pl.Int32).alias("ano_cmpt_max"),
    )

    target_float_cols = (
        "internacoes_total",
        "pacientes_dia_ativos_total",
        "uti_adulto_internacoes",
        "uti_adulto_diarias",
        "diarias_uti_total_ativas",
        "hu_uti_adulto_diarias",
        "clinico_internacoes",
        "clinico_pacientes_dia",
        "respiratorio_internacoes",
        "respiratorio_pacientes_dia",
        "leptospirose_internacoes",
        "trauma_internacoes",
    )

    panel = (
        panel.filter((pl.col("ano_epi") >= min_year) & (pl.col("ano_epi") <= max_year))
        .join(demand_agg, on=["id_regiao_saude", "ano_epi", "semana_epi"], how="left")
        .with_columns(
            *[pl.col(c).fill_null(0.0) for c in target_float_cols],
            pl.col("ano_cmpt_max").fill_null(pl.col("ano_epi").cast(pl.Int32)),
        )
    )

    # 5. Capacidade mensal CNES acoplada via join_asof causal (ano*100+mes <= ano_epi*100+mes_epi)
    cnes_monthly = _build_monthly_cnes_capacity_by_region(
        cnes_beds=cnes_beds, mun_to_reg=mun_to_reg
    )
    panel = (
        panel.with_columns(
            (
                pl.col("ano_epi").cast(pl.Int64) * 100
                + (((pl.col("semana_epi").cast(pl.Int64) - 1) * 12 // 53) + 1).clip(1, 12)
            ).alias("_ym")
        )
        .sort(["id_regiao_saude", "_ym"])
        .join_asof(
            cnes_monthly,
            on="_ym",
            by="id_regiao_saude",
            strategy="backward",
            check_sortedness=False,
        )
        .drop("_ym")
        .with_columns(
            pl.col("cnes_uti_sus_leitos").fill_null(0.0),
            pl.col("cnes_uti_exist_leitos").fill_null(0.0),
            pl.col("cnes_total_sus_leitos").fill_null(0.0),
        )
        .sort(["id_regiao_saude", "ano_epi", "semana_epi"])
    )
    return panel
