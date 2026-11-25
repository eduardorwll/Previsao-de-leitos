"""Orquestrador profundo de ingestão da Trilha 1A (cache + validação + LGPD + manifesto)."""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

from previsao_leitos.ingestion.cache import AtomicParquetCache
from previsao_leitos.ingestion.lgpd import (
    count_suppressed_cells,
    suppress_small_cells,
    validate_no_pii_columns,
)
from previsao_leitos.ingestion.ports import (
    IngestionConfig,
    IngestionResult,
    QuerySpec,
    TabularSourcePort,
)
from previsao_leitos.ingestion.queries import (
    build_cnes_beds_by_specialty_query,
    build_cnes_teaching_hospitals_query,
    build_ibge_municipios_query,
    build_sih_active_census_query,
    build_sih_delay_triangle_query,
    build_sih_external_referral_matrix_query,
    build_sih_referral_flows_query,
    build_sih_weekly_demand_query,
)
from previsao_leitos.ingestion.schemas import (
    validate_cnes_beds_frame,
    validate_cnes_teaching_hospitals_frame,
    validate_ibge_municipios_frame,
    validate_sih_active_census_frame,
    validate_sih_delay_triangle_frame,
    validate_sih_external_referral_matrix_frame,
    validate_sih_referral_flows_frame,
    validate_sih_weekly_demand_frame,
)

ValidatorFn = Callable[[pl.DataFrame], pl.DataFrame]


def _ingest_one(
    *,
    source: TabularSourcePort,
    cache: AtomicParquetCache,
    spec: QuerySpec,
    validator: ValidatorFn,
    config: IngestionConfig,
    lgpd_count_columns: tuple[str, ...] = (),
    lgpd_stratum_columns: tuple[str, ...] = (),
) -> IngestionResult:
    cached = cache.get_with_manifest(spec)
    if cached is not None:
        return cached

    estimated_bytes = source.estimate_bytes(spec)
    raw_df = source.fetch(spec, max_bytes_billed=config.max_bytes_billed)
    actual_bytes = int(getattr(source, "last_bytes_processed", 0) or estimated_bytes or 0)
    validate_no_pii_columns(raw_df, allowed_columns=spec.expected_columns)
    validated_df = validator(raw_df)

    suppressed_cells = 0
    final_df = validated_df
    if lgpd_count_columns:
        final_df = suppress_small_cells(
            validated_df,
            count_columns=lgpd_count_columns,
            stratum_columns=lgpd_stratum_columns,
            min_cell_count=config.min_cell_count,
        )
        suppressed_cells = count_suppressed_cells(
            validated_df, final_df, count_columns=lgpd_count_columns
        )

    rows_dropped = max(0, raw_df.height - final_df.height)
    manifest = cache.put(
        spec,
        final_df,
        bytes_scanned=actual_bytes,
        rows_dropped=rows_dropped,
        suppressed_cells=suppressed_cells,
        source_name=source.source_name,
    )
    return IngestionResult(dataframe=final_df, manifest=manifest)


def extract_rs_core_health_pack(
    *,
    source: TabularSourcePort,
    config: IngestionConfig,
    include_v3_sih: bool = False,
    train_end_year: int = 2022,
    delay_start_year: int = 2020,
    max_delay_weeks: int = 12,
) -> dict[str, IngestionResult]:
    """Executa a ingestão completa da Trilha 1A (Saúde & Demografia) com cache e LGPD."""
    cache = AtomicParquetCache(root_dir=config.data_dir / "raw")

    tasks: list[tuple[QuerySpec, ValidatorFn, tuple[str, ...], tuple[str, ...]]] = [
        (
            build_cnes_teaching_hospitals_query(
                uf=config.uf, ano_inicio=config.ano_inicio, ano_fim=config.ano_fim
            ),
            validate_cnes_teaching_hospitals_frame,
            (),
            (),
        ),
        (
            build_cnes_beds_by_specialty_query(
                uf=config.uf, ano_inicio=config.ano_inicio, ano_fim=config.ano_fim
            ),
            validate_cnes_beds_frame,
            (),
            (),
        ),
        (
            build_sih_weekly_demand_query(
                uf=config.uf, ano_inicio=config.ano_inicio, ano_fim=config.ano_fim
            ),
            validate_sih_weekly_demand_frame,
            ("internacoes", "obitos", "altas_vivas") if config.suppress_demand_microstrata else (),
            ("id_estabelecimento_cnes", "ano_epi", "semana_epi"),
        ),
        (
            build_sih_referral_flows_query(
                uf=config.uf, ano_inicio=config.ano_inicio, ano_fim=config.ano_fim
            ),
            validate_sih_referral_flows_frame,
            ("internacoes",),
            ("id_estabelecimento_cnes", "ano_epi", "semana_epi"),
        ),
        (
            build_ibge_municipios_query(
                uf=config.uf, ano_inicio=config.ano_inicio, ano_fim=config.ano_fim
            ),
            validate_ibge_municipios_frame,
            (),
            (),
        ),
    ]

    if include_v3_sih:
        tasks.extend(
            [
                (
                    build_sih_active_census_query(
                        uf=config.uf, ano_inicio=config.ano_inicio, ano_fim=config.ano_fim
                    ),
                    validate_sih_active_census_frame,
                    ("internacoes_iniciadas",) if config.suppress_demand_microstrata else (),
                    ("id_estabelecimento_cnes", "ano_epi", "semana_epi"),
                ),
                (
                    build_sih_external_referral_matrix_query(
                        uf=config.uf,
                        ano_inicio=config.ano_inicio,
                        train_end_year=train_end_year,
                        min_cell_count=config.min_cell_count,
                    ),
                    validate_sih_external_referral_matrix_frame,
                    (),
                    (),
                ),
                (
                    build_sih_delay_triangle_query(
                        uf=config.uf,
                        ano_inicio=delay_start_year,
                        ano_fim=max(config.ano_fim, 2026),
                        max_delay_weeks=max_delay_weeks,
                    ),
                    validate_sih_delay_triangle_frame,
                    ("incremental_count",),
                    ("ano_epi", "semana_epi"),
                ),
            ]
        )

    results: dict[str, IngestionResult] = {}
    for spec, validator, lgpd_counts, lgpd_strata in tasks:
        results[spec.name] = _ingest_one(
            source=source,
            cache=cache,
            spec=spec,
            validator=validator,
            config=config,
            lgpd_count_columns=lgpd_counts,
            lgpd_stratum_columns=lgpd_strata,
        )
    return results


def extract_rs_v3_sih_pack(
    *,
    source: TabularSourcePort,
    config: IngestionConfig,
    train_end_year: int = 2022,
    delay_start_year: int = 2020,
    delay_end_year: int = 2026,
    max_delay_weeks: int = 12,
) -> dict[str, IngestionResult]:
    """Extrai exclusivamente as 3 tabelas SIH v3 (censo ativo, W_ref externa, triângulo atraso)."""
    cache = AtomicParquetCache(root_dir=config.data_dir / "raw")
    tasks: tuple[tuple[QuerySpec, ValidatorFn, tuple[str, ...], tuple[str, ...]], ...] = (
        (
            build_sih_active_census_query(
                uf=config.uf, ano_inicio=config.ano_inicio, ano_fim=config.ano_fim
            ),
            validate_sih_active_census_frame,
            ("internacoes_iniciadas",) if config.suppress_demand_microstrata else (),
            ("id_estabelecimento_cnes", "ano_epi", "semana_epi"),
        ),
        (
            build_sih_external_referral_matrix_query(
                uf=config.uf,
                ano_inicio=config.ano_inicio,
                train_end_year=train_end_year,
                min_cell_count=config.min_cell_count,
            ),
            validate_sih_external_referral_matrix_frame,
            (),
            (),
        ),
        (
            build_sih_delay_triangle_query(
                uf=config.uf,
                ano_inicio=delay_start_year,
                ano_fim=delay_end_year,
                max_delay_weeks=max_delay_weeks,
            ),
            validate_sih_delay_triangle_frame,
            ("incremental_count",),
            ("ano_epi", "semana_epi"),
        ),
    )
    results: dict[str, IngestionResult] = {}
    for spec, validator, lgpd_counts, lgpd_strata in tasks:
        results[spec.name] = _ingest_one(
            source=source,
            cache=cache,
            spec=spec,
            validator=validator,
            config=config,
            lgpd_count_columns=lgpd_counts,
            lgpd_stratum_columns=lgpd_strata,
        )
    return results


def aggregate_regional_weekly_demand(
    *,
    sih_df: pl.DataFrame,
    hospitals_df: pl.DataFrame,
    ibge_df: pl.DataFrame,
    group_by_col: str = "grupo_cid",
    only_teaching_hospitals: bool = False,
    min_cell_count: int = 5,
) -> pl.DataFrame:
    """Agrega demanda semanal SIH-SUS por Região de Saúde e aplica supressão LGPD (n < 5).

    Preserva o sinal epidemiológico de agravos raros pós-enchente (como leptospirose) que
    seriam apagados se suprimidos em micro-estratos de hospital isolado, garantindo conformidade
    LGPD na tabela regional de divulgação.
    """
    if "ano" in hospitals_df.columns:
        hosp_yearly = (
            hospitals_df.with_columns(pl.col("ano").cast(pl.Int32).alias("ano_cnes"))
            .group_by(["id_estabelecimento_cnes", "ano_cnes"])
            .agg(
                pl.col("id_municipio").last().alias("id_municipio"),
                pl.col("is_hospital_ensino").any().alias("is_hospital_ensino"),
            )
            .sort(["ano_cnes", "id_estabelecimento_cnes"])
        )
        sih_sorted = sih_df.with_columns(pl.col("ano_epi").cast(pl.Int32)).sort(
            ["ano_epi", "id_estabelecimento_cnes"]
        )
        # Join temporal estritamente causal (ano_cnes <= ano_epi): zero vazamento futuro
        joined = sih_sorted.join_asof(
            hosp_yearly,
            left_on="ano_epi",
            right_on="ano_cnes",
            by="id_estabelecimento_cnes",
            strategy="backward",
            check_sortedness=False,
        ).drop("ano_cnes")
    else:
        hosp_map = hospitals_df.group_by("id_estabelecimento_cnes").agg(
            pl.col("id_municipio").last().alias("id_municipio"),
            pl.col("is_hospital_ensino").any().alias("is_hospital_ensino"),
        )
        joined = sih_df.join(hosp_map, on="id_estabelecimento_cnes", how="left")

    reg_cols = ["id_municipio"]
    if "nome_regiao_saude" in ibge_df.columns:
        reg_cols.append("nome_regiao_saude")
    reg_map = ibge_df.select(reg_cols).unique(subset=["id_municipio"])
    if "nome_regiao_saude" not in reg_map.columns:
        reg_map = reg_map.with_columns(pl.lit("NAO_INFORMADA").alias("nome_regiao_saude"))

    joined = joined.join(reg_map, on="id_municipio", how="left")
    if only_teaching_hospitals:
        joined = joined.filter(pl.col("is_hospital_ensino").fill_null(False))

    aggregated = (
        joined.with_columns(pl.col("nome_regiao_saude").fill_null("NAO_INFORMADA"))
        .group_by(["nome_regiao_saude", "ano_epi", "semana_epi", group_by_col])
        .agg(
            pl.col("internacoes").sum().alias("internacoes"),
            pl.col("dias_permanencia_total").sum().alias("dias_permanencia_total"),
            pl.col("diarias_uti_total").sum().alias("diarias_uti_total"),
        )
        .sort(["nome_regiao_saude", "ano_epi", "semana_epi", group_by_col])
    )
    return suppress_small_cells(
        aggregated,
        count_columns=("internacoes",),
        stratum_columns=("nome_regiao_saude", "ano_epi", "semana_epi"),
        min_cell_count=min_cell_count,
    )
