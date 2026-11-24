"""Conectores e agregadores LGPD para GERINT-POA, SINAN e SIVEP-Gripe (F01.2).

Desacopla demanda solicitada/fila reprimida (GERINT-POA), início real de
sintomas de leptospirose (SINAN DT_SIN_PRI) e onda viral respiratória de
inverno (SIVEP-Gripe SRAG) sob conformidade estrita LGPD (zero PII, n < 5).

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from previsao_leitos.domain import DataContractError
from previsao_leitos.ingestion.lgpd import suppress_small_cells, validate_no_pii_columns


def _load_frame(data: pl.DataFrame | Path) -> pl.DataFrame:
    if isinstance(data, Path):
        if not data.exists():
            raise DataContractError(f"Arquivo epidemiológico não encontrado: {data}")
        if data.suffix.lower() == ".parquet":
            return pl.read_parquet(data)
        return pl.read_csv(data, try_parse_dates=True)
    return data


def _require_cols(df: pl.DataFrame, cols: tuple[str, ...], source_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataContractError(f"{source_name}: colunas obrigatórias ausentes: {missing}")


def aggregate_gerint_poa_weekly(
    data: pl.DataFrame | Path,
    *,
    min_cell_count: int = 5,
) -> pl.DataFrame:
    """Agrega demanda solicitada e fila reprimida semanal do GERINT-POA (zero PII)."""
    df = _load_frame(data)
    required = ("data_solicitacao", "tipo_leito", "tempo_espera_horas", "internado")
    validate_no_pii_columns(df, allowed_columns=required)
    _require_cols(df, required, "gerint_poa")

    normalized = df.with_columns(
        pl.col("data_solicitacao").cast(pl.Date),
        pl.col("tipo_leito").cast(pl.String).str.strip_chars().str.to_lowercase(),
        pl.col("tempo_espera_horas").cast(pl.Float64).clip(lower_bound=0.0),
        pl.col("internado").cast(pl.Boolean),
    ).with_columns(
        pl.col("data_solicitacao").dt.iso_year().cast(pl.Int32).alias("ano_epi"),
        pl.col("data_solicitacao").dt.week().cast(pl.Int8).alias("semana_epi"),
    )

    agg = (
        normalized.group_by(["ano_epi", "semana_epi", "tipo_leito"])
        .agg(
            pl.len().cast(pl.Int64).alias("solicitacoes_totais"),
            (~pl.col("internado")).sum().cast(pl.Int64).alias("fila_reprimida_nao_internada"),
            pl.col("tempo_espera_horas").median().alias("tempo_espera_mediano_h"),
            pl.col("tempo_espera_horas").quantile(0.90).alias("tempo_espera_p90_h"),
        )
        .with_columns(
            (
                pl.col("fila_reprimida_nao_internada").cast(pl.Float64)
                / pl.col("solicitacoes_totais").cast(pl.Float64).clip(lower_bound=1.0)
            ).alias("razao_demanda_reprimida")
        )
        .sort(["ano_epi", "semana_epi", "tipo_leito"])
    )
    return suppress_small_cells(
        agg,
        count_columns=("solicitacoes_totais",),
        stratum_columns=("ano_epi", "semana_epi"),
        min_cell_count=min_cell_count,
    )


def aggregate_sinan_leptospirose_weekly(
    data: pl.DataFrame | Path,
    *,
    region_col: str = "nome_regiao_saude",
    min_cell_count: int = 5,
) -> pl.DataFrame:
    """Agrega notificações SINAN de leptospirose pela data de primeiros sintomas (DT_SIN_PRI)."""
    df = _load_frame(data)
    required = ("data_primeiros_sintomas", region_col, "confirmado", "hospitalizado")
    validate_no_pii_columns(df, allowed_columns=required)
    _require_cols(df, required, "sinan_leptospirose")

    normalized = df.with_columns(
        pl.col("data_primeiros_sintomas").cast(pl.Date),
        pl.col(region_col).cast(pl.String).str.strip_chars(),
        pl.col("confirmado").cast(pl.Boolean),
        pl.col("hospitalizado").cast(pl.Boolean),
    ).with_columns(
        pl.col("data_primeiros_sintomas").dt.iso_year().cast(pl.Int32).alias("ano_epi"),
        pl.col("data_primeiros_sintomas").dt.week().cast(pl.Int8).alias("semana_epi"),
    )

    agg = (
        normalized.group_by([region_col, "ano_epi", "semana_epi"])
        .agg(
            pl.len().cast(pl.Int64).alias("casos_notificados"),
            pl.col("confirmado").sum().cast(pl.Int64).alias("casos_confirmados"),
            pl.col("hospitalizado").sum().cast(pl.Int64).alias("casos_hospitalizados"),
        )
        .sort([region_col, "ano_epi", "semana_epi"])
    )
    return suppress_small_cells(
        agg,
        count_columns=("casos_notificados", "casos_confirmados", "casos_hospitalizados"),
        stratum_columns=(region_col, "ano_epi", "semana_epi"),
        min_cell_count=min_cell_count,
    )


def aggregate_sivep_gripe_weekly(
    data: pl.DataFrame | Path,
    *,
    region_col: str = "nome_regiao_saude",
    min_cell_count: int = 5,
) -> pl.DataFrame:
    """Agrega SIVEP-Gripe (SRAG) por semana de primeiros sintomas isolando onda viral."""
    df = _load_frame(data)
    required = ("data_primeiros_sintomas", region_col, "uti", "viral_confirmado")
    validate_no_pii_columns(df, allowed_columns=required)
    _require_cols(df, required, "sivep_gripe")

    normalized = df.with_columns(
        pl.col("data_primeiros_sintomas").cast(pl.Date),
        pl.col(region_col).cast(pl.String).str.strip_chars(),
        pl.col("uti").cast(pl.Boolean),
        pl.col("viral_confirmado").cast(pl.Boolean),
    ).with_columns(
        pl.col("data_primeiros_sintomas").dt.iso_year().cast(pl.Int32).alias("ano_epi"),
        pl.col("data_primeiros_sintomas").dt.week().cast(pl.Int8).alias("semana_epi"),
    )

    agg = (
        normalized.group_by([region_col, "ano_epi", "semana_epi"])
        .agg(
            pl.len().cast(pl.Int64).alias("srag_total"),
            pl.col("uti").sum().cast(pl.Int64).alias("srag_uti"),
            pl.col("viral_confirmado").sum().cast(pl.Int64).alias("srag_viral_confirmado"),
        )
        .with_columns(
            (
                pl.col("srag_viral_confirmado").cast(pl.Float64)
                / pl.col("srag_total").cast(pl.Float64).clip(lower_bound=1.0)
            ).alias("frac_srag_viral")
        )
        .sort([region_col, "ano_epi", "semana_epi"])
    )
    return suppress_small_cells(
        agg,
        count_columns=("srag_total", "srag_uti", "srag_viral_confirmado"),
        stratum_columns=(region_col, "ano_epi", "semana_epi"),
        min_cell_count=min_cell_count,
    )
