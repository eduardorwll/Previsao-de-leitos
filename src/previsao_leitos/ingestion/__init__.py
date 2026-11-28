"""Módulo de ingestão segura, parametrizada e conforme LGPD (Trilha 1A)."""

from previsao_leitos.ingestion.adapters import BigQueryAdapter, ParquetFixtureAdapter
from previsao_leitos.ingestion.cache import AtomicParquetCache
from previsao_leitos.ingestion.data_lake import (
    scan_hive_partitioned,
    sink_hive_partitioned,
    write_hive_partitioned,
)
from previsao_leitos.ingestion.epidemiology_sources import (
    aggregate_gerint_poa_weekly,
    aggregate_sinan_leptospirose_weekly,
    aggregate_sivep_gripe_weekly,
)
from previsao_leitos.ingestion.lgpd import (
    count_suppressed_cells,
    suppress_small_cells,
    validate_no_pii_columns,
)
from previsao_leitos.ingestion.pipeline import (
    aggregate_regional_weekly_demand,
    extract_rs_core_health_pack,
    extract_rs_v3_sih_pack,
)
from previsao_leitos.ingestion.ports import (
    UF_CODE_TO_SIGLA,
    IngestionConfig,
    IngestionManifest,
    IngestionResult,
    QueryParam,
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
    validate_sih_active_census_frame,
    validate_sih_delay_triangle_frame,
    validate_sih_external_referral_matrix_frame,
)
from previsao_leitos.ingestion.wave2a_clients import (
    CachedPayload,
    HttpFetcher,
    UrllibHttpFetcher,
    ZenodoRecordSummary,
    discover_gerint_poa_csv_urls,
    fetch_cached_bytes,
    fetch_noaa_enso_monthly,
    fetch_zenodo_iph_rs_metadata,
    ingest_gerint_poa_csv_lgpd,
    ingest_opendatasus_sinan_csv_lgpd,
    ingest_opendatasus_sivep_csv_lgpd,
    parse_noaa_mei_v2_table,
)
from previsao_leitos.ingestion.wave2b_clients import (
    AnaHidrowebClient,
    AnaHidrowebCredentials,
    AnaStationFetchResult,
    MergeCptecClient,
    build_merge_cptec_daily_url,
)

__all__ = [
    "UF_CODE_TO_SIGLA",
    "AnaHidrowebClient",
    "AnaHidrowebCredentials",
    "AnaStationFetchResult",
    "AtomicParquetCache",
    "BigQueryAdapter",
    "CachedPayload",
    "HttpFetcher",
    "IngestionConfig",
    "IngestionManifest",
    "IngestionResult",
    "MergeCptecClient",
    "ParquetFixtureAdapter",
    "QueryParam",
    "QuerySpec",
    "TabularSourcePort",
    "UrllibHttpFetcher",
    "ZenodoRecordSummary",
    "aggregate_gerint_poa_weekly",
    "aggregate_regional_weekly_demand",
    "aggregate_sinan_leptospirose_weekly",
    "aggregate_sivep_gripe_weekly",
    "build_cnes_beds_by_specialty_query",
    "build_cnes_teaching_hospitals_query",
    "build_ibge_municipios_query",
    "build_merge_cptec_daily_url",
    "build_sih_active_census_query",
    "build_sih_delay_triangle_query",
    "build_sih_external_referral_matrix_query",
    "build_sih_referral_flows_query",
    "build_sih_weekly_demand_query",
    "count_suppressed_cells",
    "discover_gerint_poa_csv_urls",
    "extract_rs_core_health_pack",
    "extract_rs_v3_sih_pack",
    "fetch_cached_bytes",
    "fetch_noaa_enso_monthly",
    "fetch_zenodo_iph_rs_metadata",
    "ingest_gerint_poa_csv_lgpd",
    "ingest_opendatasus_sinan_csv_lgpd",
    "ingest_opendatasus_sivep_csv_lgpd",
    "parse_noaa_mei_v2_table",
    "scan_hive_partitioned",
    "sink_hive_partitioned",
    "suppress_small_cells",
    "validate_no_pii_columns",
    "validate_sih_active_census_frame",
    "validate_sih_delay_triangle_frame",
    "validate_sih_external_referral_matrix_frame",
    "write_hive_partitioned",
]
