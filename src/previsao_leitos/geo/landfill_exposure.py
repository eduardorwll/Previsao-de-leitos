"""Exposição física em áreas aterradas, várzeas/banhados urbanizados e sistemas de pôlderes."""

from __future__ import annotations

import polars as pl

from previsao_leitos.domain import DataContractError

# Municípios do RS com sistemas formais de diques/pôlderes e bombeamento (DMAE, SANEP, Metroplan)
# Uso estritamente combinado com descritores físicos contínuos (ADR-0006 — jamais ID isolado)
RS_POLDER_MUNICIPALITIES: frozenset[str] = frozenset(
    {
        "4314902",  # Porto Alegre (23 EBAPs / Muro da Mauá / Diques Sarandi, Humaitá, Guaíba)
        "4304606",  # Canoas (Diques Mathias Velho, Rio Branco, Niterói)
        "4318705",  # São Leopoldo (Sistema de diques e casas de bombas do Rio dos Sinos)
        "4313409",  # Novo Hamburgo (Dique Santo Afonso)
        "4307708",  # Esteio
        "4320008",  # Sapucaia do Sul
        "4306767",  # Eldorado do Sul (planície Delta do Jacuí)
        "4303103",  # Cachoeirinha
        "4300604",  # Alvorada
        "4314407",  # Pelotas (Diques Laranjal, Navegantes, Canal São Gonçalo / SANEP)
        "4315602",  # Rio Grande (áreas aterradas e margens estuarinas Lagoa dos Patos)
    }
)

_URBAN_CLASSES = ("24",)
_WETLAND_WATER_CLASSES = ("11", "84", "33")
_FOREST_CLASSES = ("3", "49")


def build_causal_landfill_exposure(
    mapbiomas_df: pl.DataFrame,
    *,
    mun_to_reg: pl.DataFrame,
    ref_year: int = 2022,
    polder_municipalities: frozenset[str] = RS_POLDER_MUNICIPALITIES,
    mun_weights: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Calcula exposição causal (`ano <= ref_year`) de áreas urbanas sobre várzeas/banhados.

    Quando `mun_weights` (`id_municipio`, `peso_exposicao` — ex.: população ou leitos CNES)
    é fornecido, agrega as frações municipais por média ponderada para eliminar a diluição
    espacial MAUP causada por municípios rurais extensos na mesma Região de Saúde.
    """
    causal = mapbiomas_df.filter(pl.col("ano") <= ref_year)
    if causal.is_empty():
        raise DataContractError(f"Sem dados MapBiomas disponíveis até ref_year={ref_year}.")

    base_year = int(causal.select(pl.col("ano").min()).item())
    latest_year = int(causal.select(pl.col("ano").max()).item())

    def _summarize_year(year: int, suffix: str) -> pl.DataFrame:
        return (
            causal.filter(pl.col("ano") == year)
            .group_by("id_municipio")
            .agg(
                pl.col("area_ha").sum().alias(f"total_ha_{suffix}"),
                pl.col("area_ha")
                .filter(pl.col("id_classe").is_in(_URBAN_CLASSES))
                .sum()
                .alias(f"urb_ha_{suffix}"),
                pl.col("area_ha")
                .filter(pl.col("id_classe").is_in(_WETLAND_WATER_CLASSES))
                .sum()
                .alias(f"wet_ha_{suffix}"),
                pl.col("area_ha")
                .filter(pl.col("id_classe").is_in(_FOREST_CLASSES))
                .sum()
                .alias(f"flor_ha_{suffix}"),
            )
        )

    base_agg = _summarize_year(base_year, "base")
    ref_agg = _summarize_year(latest_year, "ref")
    polder_list = sorted(polder_municipalities)

    mun_metrics = (
        ref_agg.join(base_agg, on="id_municipio", how="left")
        .with_columns(
            pl.col("urb_ha_base").fill_null(pl.col("urb_ha_ref")),
            pl.col("wet_ha_base").fill_null(pl.col("wet_ha_ref")),
            pl.when(pl.col("id_municipio").is_in(polder_list))
            .then(pl.col("total_ha_ref"))
            .otherwise(0.0)
            .alias("polder_area_ha"),
            (pl.col("wet_ha_base").fill_null(pl.col("wet_ha_ref")) - pl.col("wet_ha_ref"))
            .clip(lower_bound=0.0)
            .alias("perda_banhado_ha"),
            (pl.col("urb_ha_ref") - pl.col("urb_ha_base").fill_null(pl.col("urb_ha_ref")))
            .clip(lower_bound=0.0)
            .alias("expansao_urb_ha"),
        )
        .join(
            mun_to_reg.select(["id_municipio", "id_regiao_saude"]).unique(subset=["id_municipio"]),
            on="id_municipio",
            how="inner",
        )
    )

    if mun_weights is not None:
        w_df = (
            mun_weights.select(["id_municipio", "peso_exposicao"])
            .with_columns(pl.col("peso_exposicao").cast(pl.Float64).clip(lower_bound=0.0))
            .unique(subset=["id_municipio"])
        )
        mun_fracs = (
            mun_metrics.filter(pl.col("total_ha_ref") > 0.0)
            .with_columns(
                (pl.col("urb_ha_ref") / pl.col("total_ha_ref")).alias("_f_urb"),
                (pl.col("wet_ha_ref") / pl.col("total_ha_ref")).alias("_f_wet"),
                (pl.col("flor_ha_ref") / pl.col("total_ha_ref")).alias("_f_flor"),
                (pl.col("perda_banhado_ha") / pl.col("total_ha_ref")).alias("_f_perda"),
                (pl.col("expansao_urb_ha") / pl.col("total_ha_ref")).alias("_f_exp"),
                (pl.col("polder_area_ha") / pl.col("total_ha_ref")).alias("_f_polder"),
            )
            .join(w_df, on="id_municipio", how="left")
            .with_columns(pl.col("peso_exposicao").fill_null(1.0).alias("_w"))
        )
        reg_base = (
            mun_fracs.group_by("id_regiao_saude")
            .agg(
                pl.col("_w").sum().alias("_w_sum"),
                (pl.col("_f_urb") * pl.col("_w")).sum().alias("_w_urb"),
                (pl.col("_f_wet") * pl.col("_w")).sum().alias("_w_wet"),
                (pl.col("_f_flor") * pl.col("_w")).sum().alias("_w_flor"),
                (pl.col("_f_perda") * pl.col("_w")).sum().alias("_w_perda"),
                (pl.col("_f_exp") * pl.col("_w")).sum().alias("_w_exp"),
                (pl.col("_f_polder") * pl.col("_w")).sum().alias("_w_polder"),
            )
            .filter(pl.col("_w_sum") > 0.0)
            .with_columns(
                (pl.col("_w_urb") / pl.col("_w_sum")).alias("frac_urbano"),
                (pl.col("_w_wet") / pl.col("_w_sum")).alias("frac_banhado_agua"),
                (pl.col("_w_flor") / pl.col("_w_sum")).alias("frac_floresta"),
                (pl.col("_w_perda") / pl.col("_w_sum")).alias("perda_banhado_rel"),
                (pl.col("_w_exp") / pl.col("_w_sum")).alias("expansao_urbana_rel"),
                (pl.col("_w_polder") / pl.col("_w_sum")).alias("frac_area_polder_regiao"),
            )
        )
    else:
        reg_base = (
            mun_metrics.group_by("id_regiao_saude")
            .agg(
                pl.col("total_ha_ref").sum().alias("_tot"),
                pl.col("urb_ha_ref").sum().alias("_urb"),
                pl.col("wet_ha_ref").sum().alias("_wet"),
                pl.col("flor_ha_ref").sum().alias("_flor"),
                pl.col("perda_banhado_ha").sum().alias("_perda_wet"),
                pl.col("expansao_urb_ha").sum().alias("_exp_urb"),
                pl.col("polder_area_ha").sum().alias("_polder_ha"),
            )
            .filter(pl.col("_tot") > 0.0)
            .with_columns(
                (pl.col("_urb") / pl.col("_tot")).alias("frac_urbano"),
                (pl.col("_wet") / pl.col("_tot")).alias("frac_banhado_agua"),
                (pl.col("_flor") / pl.col("_tot")).alias("frac_floresta"),
                (pl.col("_perda_wet") / pl.col("_tot")).alias("perda_banhado_rel"),
                (pl.col("_exp_urb") / pl.col("_tot")).alias("expansao_urbana_rel"),
                (pl.col("_polder_ha") / pl.col("_tot")).alias("frac_area_polder_regiao"),
            )
        )

    reg_agg = (
        reg_base.with_columns(
            (
                pl.col("frac_urbano")
                * (pl.col("frac_banhado_agua") + pl.col("perda_banhado_rel"))
                + 0.5 * pl.col("expansao_urbana_rel") * pl.col("frac_banhado_agua")
                + 0.25 * pl.col("frac_area_polder_regiao") * pl.col("frac_urbano")
            ).alias("indice_aterro_varzea")
        )
        .select(
            [
                "id_regiao_saude",
                "frac_urbano",
                "frac_banhado_agua",
                "frac_floresta",
                "perda_banhado_rel",
                "expansao_urbana_rel",
                "frac_area_polder_regiao",
                "indice_aterro_varzea",
            ]
        )
        .sort("id_regiao_saude")
    )
    return reg_agg
