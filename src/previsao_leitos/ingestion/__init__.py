"""Módulo de ingestão segura, parametrizada e conforme LGPD (Trilha 1A)."""

from previsao_leitos.ingestion.adapters import BigQueryAdapter, ParquetFixtureAdapter
from previsao_leitos.ingestion.cache import AtomicParquetCache
from previsao_leitos.ingestion.lgpd import (
    count_suppressed_cells,
    suppress_small_cells,
    validate_no_pii_columns,
)
from previsao_leitos.ingestion.pipeline import extract_rs_core_health_pack
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
    build_sih_referral_flows_query,
    build_sih_weekly_demand_query,
)

__all__ = [
    "UF_CODE_TO_SIGLA",
    "AtomicParquetCache",
    "BigQueryAdapter",
    "IngestionConfig",
    "IngestionManifest",
    "IngestionResult",
    "ParquetFixtureAdapter",
    "QueryParam",
    "QuerySpec",
    "TabularSourcePort",
    "build_cnes_beds_by_specialty_query",
    "build_cnes_teaching_hospitals_query",
    "build_ibge_municipios_query",
    "build_sih_referral_flows_query",
    "build_sih_weekly_demand_query",
    "count_suppressed_cells",
    "extract_rs_core_health_pack",
    "suppress_small_cells",
    "validate_no_pii_columns",
]
