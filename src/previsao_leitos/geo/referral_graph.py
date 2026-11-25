"""Grafo assistencial intermunicipal causal (G_health: matriz empírica MUNIC_RES -> CNES)."""

from __future__ import annotations

import polars as pl

from previsao_leitos.domain import DataContractError


def build_causal_referral_matrix(
    flows_df: pl.DataFrame,
    *,
    cnes_to_reg: pl.DataFrame,
    mun_to_reg: pl.DataFrame,
    max_year: int,
    exclude_self_region: bool = False,
) -> pl.DataFrame:
    """Constrói matriz causal `W_ref(origem -> destino)` estritamente com `ano_epi <= max_year`.

    Retorna colunas `["id_regiao_origem", "id_regiao_destino", "peso_referencia"]` onde
    a soma de `peso_referencia` para cada `id_regiao_destino` é 1.0.
    """
    if flows_df.is_empty():
        raise DataContractError("Tabela de fluxos SIH vazia ao construir matriz de referência.")

    mun_map = (
        mun_to_reg.with_columns(pl.col("id_municipio").str.slice(0, 6).alias("_mun6"))
        .select(["_mun6", pl.col("id_regiao_saude").alias("id_regiao_origem")])
        .unique(subset=["_mun6"])
    )
    cnes_map = cnes_to_reg.select(
        ["id_estabelecimento_cnes", pl.col("id_regiao_saude").alias("id_regiao_destino")]
    ).unique(subset=["id_estabelecimento_cnes"])

    if "ano_epi" in flows_df.columns:
        causal_flows = flows_df.filter(pl.col("ano_epi") <= max_year)
    else:
        # Matriz externa pré-agregada no SQL (sih_w_ref_externa_treino, <= train_end_year)
        causal_flows = flows_df
    if causal_flows.is_empty():
        raise DataContractError(
            f"Nenhum fluxo SIH disponível até max_year={max_year} para matriz causal."
        )

    vol_col = (
        "internacoes"
        if "internacoes" in causal_flows.columns
        else "internacoes_urgencia_total"
    )
    if vol_col not in causal_flows.columns:
        raise DataContractError(
            "flows_df deve conter 'internacoes' ou 'internacoes_urgencia_total'."
        )

    agg = (
        causal_flows.join(
            mun_map, left_on="id_municipio_residencia_6d", right_on="_mun6", how="inner"
        )
        .join(cnes_map, on="id_estabelecimento_cnes", how="inner")
    )
    if exclude_self_region:
        agg = agg.filter(pl.col("id_regiao_origem") != pl.col("id_regiao_destino"))
    agg = agg.group_by(["id_regiao_origem", "id_regiao_destino"]).agg(
        pl.col(vol_col).sum().cast(pl.Float64).alias("_vol")
    )

    totals = agg.group_by("id_regiao_destino").agg(pl.col("_vol").sum().alias("_tot_dst"))
    return (
        agg.join(totals, on="id_regiao_destino", how="inner")
        .filter(pl.col("_tot_dst") > 0.0)
        .with_columns((pl.col("_vol") / pl.col("_tot_dst")).alias("peso_referencia"))
        .select(["id_regiao_origem", "id_regiao_destino", "peso_referencia"])
        .sort(["id_regiao_destino", "id_regiao_origem"])
    )


def compute_catchment_flood_pressure(
    weekly_df: pl.DataFrame,
    referral_matrix: pl.DataFrame,
    *,
    entity_col: str = "id_regiao_saude",
    value_col: str = "chuva_sem_mm",
    output_col: str = "chuva_catchment_mm",
) -> pl.DataFrame:
    """Calcula exposição hidrológica ponderada pela rede histórica de referência assistencial."""
    if weekly_df.is_empty() or referral_matrix.is_empty():
        return weekly_df.with_columns(pl.lit(0.0).alias(output_col))

    src = weekly_df.select([entity_col, "ano_epi", "semana_epi", value_col]).rename(
        {entity_col: "id_regiao_origem", value_col: "_src_val"}
    )
    weighted = (
        referral_matrix.join(src, on="id_regiao_origem", how="inner")
        .filter(pl.col("_src_val").is_not_null())
        .group_by(["id_regiao_destino", "ano_epi", "semana_epi"])
        .agg(
            (
                (pl.col("_src_val") * pl.col("peso_referencia")).sum()
                / pl.col("peso_referencia").sum()
            ).alias(output_col)
        )
        .rename({"id_regiao_destino": entity_col})
    )
    return weekly_df.join(
        weighted, on=[entity_col, "ano_epi", "semana_epi"], how="left"
    ).with_columns(pl.col(output_col).fill_null(0.0))
