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
    build_sih_referral_flows_query,
    build_sih_weekly_demand_query,
)
from previsao_leitos.ingestion.schemas import (
    validate_cnes_beds_frame,
    validate_cnes_teaching_hospitals_frame,
    validate_ibge_municipios_frame,
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
) -> dict[str, IngestionResult]:
    """Executa a ingestão completa da Trilha 1A (Saúde & Demografia) com cache e LGPD."""
    cache = AtomicParquetCache(root_dir=config.data_dir / "raw")

    tasks: tuple[tuple[QuerySpec, ValidatorFn, tuple[str, ...], tuple[str, ...]], ...] = (
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
            ("internacoes", "obitos", "altas_vivas"),
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
