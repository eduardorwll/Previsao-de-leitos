"""Contratos de schema Polars na fronteira de ingestão preservando zeros à esquerda."""

from __future__ import annotations

import polars as pl

from previsao_leitos.domain import DataContractError

VALID_BED_CATEGORIES: frozenset[str] = frozenset(
    {
        "uti_adulto",
        "uti_pediatrica",
        "uti_neonatal",
        "uci",
        "clinico",
        "cirurgico",
        "obstetrico",
        "pediatrico",
        "outros",
    }
)


def _require_columns(df: pl.DataFrame, required: tuple[str, ...], dataset_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DataContractError(f"Colunas obrigatórias ausentes em {dataset_name}: {missing}")


def _assert_fixed_digits(df: pl.DataFrame, col: str, digits: int, dataset_name: str) -> None:
    pattern = rf"^\d{{{digits}}}$"
    invalid = df.filter(pl.col(col).is_null() | ~pl.col(col).str.contains(pattern))
    if invalid.height > 0:
        raise DataContractError(
            f"{dataset_name}.{col} contém {invalid.height} valor(es) fora do padrão "
            f"de {digits} dígitos numéricos."
        )


def validate_cnes_teaching_hospitals_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Valida e normaliza tipos do catálogo de estabelecimentos/hospitais-escola CNES."""
    required = (
        "id_estabelecimento_cnes",
        "id_municipio",
        "sigla_uf",
        "ano",
        "mes",
        "tipo_unidade",
        "is_hospital_ensino",
    )
    _require_columns(df, required, "cnes_hospitais_escola")

    normalized = df.select(list(required)).with_columns(
        pl.col("id_estabelecimento_cnes").cast(pl.String).str.strip_chars().str.zfill(7),
        pl.col("id_municipio").cast(pl.String).str.strip_chars().str.zfill(7),
        pl.col("sigla_uf").cast(pl.String).str.strip_chars().str.to_uppercase(),
        pl.col("ano").cast(pl.Int32),
        pl.col("mes").cast(pl.Int8),
        pl.col("tipo_unidade").cast(pl.String).str.strip_chars(),
        pl.col("is_hospital_ensino").cast(pl.Boolean),
    )

    _assert_fixed_digits(normalized, "id_estabelecimento_cnes", 7, "cnes_hospitais_escola")
    _assert_fixed_digits(normalized, "id_municipio", 7, "cnes_hospitais_escola")

    bad_mes = normalized.filter((pl.col("mes") < 1) | (pl.col("mes") > 12))
    if bad_mes.height > 0:
        raise DataContractError("cnes_hospitais_escola.mes fora do intervalo 1..12.")

    return normalized


def validate_cnes_beds_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Valida estoque mensal CNES por categoria específica de leito (UTI separada de UCI)."""
    required = (
        "id_estabelecimento_cnes",
        "id_municipio",
        "ano",
        "mes",
        "categoria_leito",
        "quantidade_existente",
        "quantidade_sus",
    )
    _require_columns(df, required, "cnes_leitos_especialidade")

    normalized = df.select(list(required)).with_columns(
        pl.col("id_estabelecimento_cnes").cast(pl.String).str.strip_chars().str.zfill(7),
        pl.col("id_municipio").cast(pl.String).str.strip_chars().str.zfill(7),
        pl.col("ano").cast(pl.Int32),
        pl.col("mes").cast(pl.Int8),
        pl.col("categoria_leito").cast(pl.String).str.strip_chars().str.to_lowercase(),
        pl.col("quantidade_existente").cast(pl.Int32),
        pl.col("quantidade_sus").cast(pl.Int32),
    )

    _assert_fixed_digits(normalized, "id_estabelecimento_cnes", 7, "cnes_leitos_especialidade")
    _assert_fixed_digits(normalized, "id_municipio", 7, "cnes_leitos_especialidade")

    invalid_cats = normalized.filter(~pl.col("categoria_leito").is_in(list(VALID_BED_CATEGORIES)))
    if invalid_cats.height > 0:
        bad_vals = invalid_cats["categoria_leito"].unique().to_list()
        raise DataContractError(
            f"categoria_leito inválida em cnes_leitos_especialidade: {bad_vals}"
        )

    neg = normalized.filter(
        (pl.col("quantidade_existente") < 0) | (pl.col("quantidade_sus") < 0)
    )
    if neg.height > 0:
        raise DataContractError("Quantidades negativas encontradas em cnes_leitos_especialidade.")

    sus_gt_total = normalized.filter(pl.col("quantidade_sus") > pl.col("quantidade_existente"))
    if sus_gt_total.height > 0:
        raise DataContractError(
            "Violação de contrato: quantidade_sus excede quantidade_existente em "
            f"{sus_gt_total.height} linha(s)."
        )

    return normalized


def validate_sih_weekly_demand_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Valida tabela semanal agregada de internações e LOS do SIH-SUS (sem PII)."""
    required = (
        "id_estabelecimento_cnes",
        "ano_epi",
        "semana_epi",
        "especialidade",
        "grupo_cid",
        "carater_atendimento",
        "internacoes",
        "dias_permanencia_total",
        "diarias_uti_total",
        "obitos",
        "altas_vivas",
    )
    _require_columns(df, required, "sih_demanda_semanal_agg")

    normalized = df.select(list(required)).with_columns(
        pl.col("id_estabelecimento_cnes").cast(pl.String).str.strip_chars().str.zfill(7),
        pl.col("ano_epi").cast(pl.Int32),
        pl.col("semana_epi").cast(pl.Int8),
        pl.col("especialidade").cast(pl.String).str.strip_chars().str.to_lowercase(),
        pl.col("grupo_cid").cast(pl.String).str.strip_chars().str.to_lowercase(),
        pl.col("carater_atendimento").cast(pl.String).str.strip_chars().str.to_lowercase(),
        pl.col("internacoes").cast(pl.Int64),
        pl.col("dias_permanencia_total").cast(pl.Int64),
        pl.col("diarias_uti_total").cast(pl.Int64),
        pl.col("obitos").cast(pl.Int64),
        pl.col("altas_vivas").cast(pl.Int64),
    )

    _assert_fixed_digits(normalized, "id_estabelecimento_cnes", 7, "sih_demanda_semanal_agg")

    bad_week = normalized.filter((pl.col("semana_epi") < 1) | (pl.col("semana_epi") > 53))
    if bad_week.height > 0:
        raise DataContractError("sih_demanda_semanal_agg.semana_epi fora do intervalo 1..53.")

    for col in (
        "internacoes",
        "dias_permanencia_total",
        "diarias_uti_total",
        "obitos",
        "altas_vivas",
    ):
        neg = normalized.filter(pl.col(col) < 0)
        if neg.height > 0:
            raise DataContractError(f"Valores negativos em sih_demanda_semanal_agg.{col}.")

    return normalized


def validate_sih_referral_flows_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Valida matriz semanal de fluxo de referência MUNIC_RES (6d) -> CNES (7d)."""
    required = (
        "id_municipio_residencia_6d",
        "id_estabelecimento_cnes",
        "ano_epi",
        "semana_epi",
        "internacoes",
    )
    _require_columns(df, required, "sih_fluxos_referencia_semanal")

    normalized = df.select(list(required)).with_columns(
        pl.col("id_municipio_residencia_6d").cast(pl.String).str.strip_chars().str.zfill(6),
        pl.col("id_estabelecimento_cnes").cast(pl.String).str.strip_chars().str.zfill(7),
        pl.col("ano_epi").cast(pl.Int32),
        pl.col("semana_epi").cast(pl.Int8),
        pl.col("internacoes").cast(pl.Int64),
    )

    _assert_fixed_digits(
        normalized, "id_municipio_residencia_6d", 6, "sih_fluxos_referencia_semanal"
    )
    _assert_fixed_digits(normalized, "id_estabelecimento_cnes", 7, "sih_fluxos_referencia_semanal")

    bad_week = normalized.filter((pl.col("semana_epi") < 1) | (pl.col("semana_epi") > 53))
    if bad_week.height > 0:
        raise DataContractError("sih_fluxos_referencia_semanal.semana_epi fora de 1..53.")

    if normalized.filter(pl.col("internacoes") < 0).height > 0:
        raise DataContractError("Valores negativos em sih_fluxos_referencia_semanal.internacoes.")

    return normalized


def validate_ibge_municipios_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Valida diretório municipal + população anual IBGE sem imputação futura."""
    required = ("id_municipio", "nome", "sigla_uf", "ano", "populacao")
    _require_columns(df, required, "ibge_municipios_rs")

    optional_cols = [
        col for col in ("id_regiao_saude", "nome_regiao_saude") if col in df.columns
    ]
    selected = list(required) + optional_cols

    normalized = df.select(selected).with_columns(
        pl.col("id_municipio").cast(pl.String).str.strip_chars().str.zfill(7),
        pl.col("nome").cast(pl.String).str.strip_chars(),
        pl.col("sigla_uf").cast(pl.String).str.strip_chars().str.to_uppercase(),
        pl.col("ano").cast(pl.Int32),
        pl.col("populacao").cast(pl.Float64),
    )

    _assert_fixed_digits(normalized, "id_municipio", 7, "ibge_municipios_rs")

    if normalized.filter(pl.col("populacao") < 0).height > 0:
        raise DataContractError("Valores negativos em ibge_municipios_rs.populacao.")

    return normalized


def validate_sih_active_census_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Valida tabela de censo ativo semanal SIH-SUS expandido por [DT_INTER..DT_SAIDA]."""
    required = (
        "id_estabelecimento_cnes",
        "ano_epi",
        "semana_epi",
        "especialidade",
        "grupo_cid",
        "carater_atendimento",
        "internacoes_iniciadas",
        "pacientes_dia_ativos",
        "diarias_uti_ativas",
        "ano_cmpt_max",
    )
    _require_columns(df, required, "sih_censo_ativo_semanal")

    normalized = df.select(list(required)).with_columns(
        pl.col("id_estabelecimento_cnes").cast(pl.String).str.strip_chars().str.zfill(7),
        pl.col("ano_epi").cast(pl.Int32),
        pl.col("semana_epi").cast(pl.Int8),
        pl.col("especialidade").cast(pl.String).str.strip_chars().str.to_lowercase(),
        pl.col("grupo_cid").cast(pl.String).str.strip_chars().str.to_lowercase(),
        pl.col("carater_atendimento").cast(pl.String).str.strip_chars().str.to_lowercase(),
        pl.col("internacoes_iniciadas").cast(pl.Int64),
        pl.col("pacientes_dia_ativos").cast(pl.Float64),
        pl.col("diarias_uti_ativas").cast(pl.Float64),
        pl.col("ano_cmpt_max").cast(pl.Int32),
    )
    _assert_fixed_digits(normalized, "id_estabelecimento_cnes", 7, "sih_censo_ativo_semanal")

    bad_week = normalized.filter((pl.col("semana_epi") < 1) | (pl.col("semana_epi") > 53))
    if bad_week.height > 0:
        raise DataContractError("sih_censo_ativo_semanal.semana_epi fora do intervalo 1..53.")

    for col in ("internacoes_iniciadas", "pacientes_dia_ativos", "diarias_uti_ativas"):
        if normalized.filter(pl.col(col) < 0).height > 0:
            raise DataContractError(f"Valores negativos em sih_censo_ativo_semanal.{col}.")

    return normalized


def validate_sih_external_referral_matrix_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Valida matriz W_ref externa congelada no treino (origem != destino, sem auto-loop)."""
    required = (
        "id_municipio_residencia_6d",
        "id_municipio_estabelecimento_6d",
        "id_estabelecimento_cnes",
        "internacoes_urgencia_total",
        "diarias_uti_total",
    )
    _require_columns(df, required, "sih_w_ref_externa_treino")

    normalized = df.select(list(required)).with_columns(
        pl.col("id_municipio_residencia_6d").cast(pl.String).str.strip_chars().str.zfill(6),
        pl.col("id_municipio_estabelecimento_6d").cast(pl.String).str.strip_chars().str.zfill(6),
        pl.col("id_estabelecimento_cnes").cast(pl.String).str.strip_chars().str.zfill(7),
        pl.col("internacoes_urgencia_total").cast(pl.Int64),
        pl.col("diarias_uti_total").cast(pl.Int64),
    )
    _assert_fixed_digits(
        normalized, "id_municipio_residencia_6d", 6, "sih_w_ref_externa_treino"
    )
    _assert_fixed_digits(
        normalized, "id_municipio_estabelecimento_6d", 6, "sih_w_ref_externa_treino"
    )
    _assert_fixed_digits(normalized, "id_estabelecimento_cnes", 7, "sih_w_ref_externa_treino")

    self_loops = normalized.filter(
        pl.col("id_municipio_residencia_6d") == pl.col("id_municipio_estabelecimento_6d")
    )
    if self_loops.height > 0:
        raise DataContractError(
            f"sih_w_ref_externa_treino contém {self_loops.height} linha(s) de auto-referência "
            "(origem == destino)."
        )

    for col in ("internacoes_urgencia_total", "diarias_uti_total"):
        if normalized.filter(pl.col(col) < 0).height > 0:
            raise DataContractError(f"Valores negativos em sih_w_ref_externa_treino.{col}.")

    return normalized


def validate_sih_delay_triangle_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Valida triângulo de atraso de processamento SIH-SUS para ChainLadderNowcaster."""
    required = (
        "ano_epi",
        "semana_epi",
        "delay_weeks",
        "incremental_count",
        "cumulative_count",
    )
    _require_columns(df, required, "sih_delay_triangle")

    normalized = df.select(list(required)).with_columns(
        pl.col("ano_epi").cast(pl.Int32),
        pl.col("semana_epi").cast(pl.Int8),
        pl.col("delay_weeks").cast(pl.Int32),
        pl.col("incremental_count").cast(pl.Int64),
        pl.col("cumulative_count").cast(pl.Int64),
    )

    bad_week = normalized.filter((pl.col("semana_epi") < 1) | (pl.col("semana_epi") > 53))
    if bad_week.height > 0:
        raise DataContractError("sih_delay_triangle.semana_epi fora do intervalo 1..53.")

    for col in ("delay_weeks", "incremental_count", "cumulative_count"):
        if normalized.filter(pl.col(col) < 0).height > 0:
            raise DataContractError(f"Valores negativos em sih_delay_triangle.{col}.")

    bad_cum = normalized.filter(pl.col("cumulative_count") < pl.col("incremental_count"))
    if bad_cum.height > 0:
        raise DataContractError(
            "sih_delay_triangle: cumulative_count menor que incremental_count."
        )

    return normalized.sort(["ano_epi", "semana_epi", "delay_weeks"])
