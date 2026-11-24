"""Conectores de física hidrológica fina: ANA HidroWeb, MERGE-CPTEC 0.1° e SAR 2024.

Fornece agregadores determinísticos por semana ISO e bacia/região de saúde para
substituir grades grosseiras (0.5°) e aproximações por área bruta.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import polars as pl

from previsao_leitos.domain import DataContractError


def _require_columns(df: pl.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataContractError(f"{name}: colunas obrigatórias ausentes: {missing}")


def aggregate_ana_hidroweb_weekly(df: pl.DataFrame) -> pl.DataFrame:
    """Agrega cotas (cm) e vazões (m3/s) diárias ANA HidroWeb por bacia e semana ISO."""
    required = ("bacia_id", "data_medicao", "cota_cm", "vazao_m3s", "cota_inundacao_cm")
    _require_columns(df, required, "ana_hidroweb")
    if df.is_empty():
        raise DataContractError("ana_hidroweb: DataFrame vazio.")

    norm = df.with_columns(
        pl.col("data_medicao").cast(pl.Date),
        pl.col("cota_cm").cast(pl.Float64),
        pl.col("vazao_m3s").cast(pl.Float64).clip(lower_bound=0.0),
        pl.col("cota_inundacao_cm").cast(pl.Float64),
    ).with_columns(
        pl.col("data_medicao").dt.iso_year().cast(pl.Int32).alias("ano_epi"),
        pl.col("data_medicao").dt.week().cast(pl.Int8).alias("semana_epi"),
        (pl.col("cota_cm") - pl.col("cota_inundacao_cm"))
        .clip(lower_bound=0.0)
        .alias("_excedencia_cm"),
    )

    return (
        norm.group_by(["bacia_id", "ano_epi", "semana_epi"])
        .agg(
            pl.col("cota_cm").max().alias("cota_max_cm"),
            pl.col("cota_cm").mean().alias("cota_media_cm"),
            pl.col("vazao_m3s").max().alias("vazao_max_m3s"),
            pl.col("_excedencia_cm").max().alias("excedencia_cota_inundacao_cm"),
        )
        .sort(["bacia_id", "ano_epi", "semana_epi"])
    )


def aggregate_merge_cptec_weekly(df: pl.DataFrame) -> pl.DataFrame:
    """Agrega precipitação diária MERGE-CPTEC (0.1°) nos municípios-sede por semana ISO."""
    required = ("id_municipio", "data", "chuva_mm")
    _require_columns(df, required, "merge_cptec")
    if df.is_empty():
        raise DataContractError("merge_cptec: DataFrame vazio.")

    norm = df.with_columns(
        pl.col("id_municipio").cast(pl.String).str.strip_chars().str.zfill(7),
        pl.col("data").cast(pl.Date),
        pl.col("chuva_mm").cast(pl.Float64).clip(lower_bound=0.0),
    ).with_columns(
        pl.col("data").dt.iso_year().cast(pl.Int32).alias("ano_epi"),
        pl.col("data").dt.week().cast(pl.Int8).alias("semana_epi"),
    )

    return (
        norm.group_by(["id_municipio", "ano_epi", "semana_epi"])
        .agg(
            pl.col("chuva_mm").sum().alias("chuva_merge_sem_mm"),
            pl.col("chuva_mm").max().alias("chuva_merge_max_24h_mm"),
        )
        .sort(["id_municipio", "ano_epi", "semana_epi"])
    )


def aggregate_sar_flood_extent(
    df: pl.DataFrame,
    *,
    region_col: str = "nome_regiao_saude",
) -> pl.DataFrame:
    """Agrega mancha de inundação urbana Sentinel-1 SAR / IPH-UFRGS por região de saúde."""
    required = (region_col, "area_urbana_km2", "area_urbana_inundada_km2", "hospitais_atingidos")
    _require_columns(df, required, "sar_flood_extent")
    if df.is_empty():
        raise DataContractError("sar_flood_extent: DataFrame vazio.")

    return (
        df.with_columns(
            pl.col("area_urbana_km2").cast(pl.Float64).clip(lower_bound=0.0),
            pl.col("area_urbana_inundada_km2").cast(pl.Float64).clip(lower_bound=0.0),
            pl.col("hospitais_atingidos").cast(pl.Int64).clip(lower_bound=0),
        )
        .group_by(region_col)
        .agg(
            pl.col("area_urbana_km2").sum().alias("_urb_km2"),
            pl.col("area_urbana_inundada_km2").sum().alias("area_urbana_inundada_km2"),
            pl.col("hospitais_atingidos").sum().alias("hospitais_atingidos_sar"),
        )
        .with_columns(
            (
                pl.col("area_urbana_inundada_km2")
                / pl.col("_urb_km2").clip(lower_bound=1e-6)
            )
            .clip(upper_bound=1.0)
            .alias("frac_area_urbana_inundada_sar")
        )
        .drop("_urb_km2")
        .sort(region_col)
    )
