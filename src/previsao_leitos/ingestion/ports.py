"""Tipos de contrato, configuração, manifestos e costura TabularSourcePort do módulo ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import polars as pl

from previsao_leitos.domain import CodigoUF

UF_CODE_TO_SIGLA: dict[str, str] = {
    "11": "RO",
    "12": "AC",
    "13": "AM",
    "14": "RR",
    "15": "PA",
    "16": "AP",
    "17": "TO",
    "21": "MA",
    "22": "PI",
    "23": "CE",
    "24": "RN",
    "25": "PB",
    "26": "PE",
    "27": "AL",
    "28": "SE",
    "29": "BA",
    "31": "MG",
    "32": "ES",
    "33": "RJ",
    "35": "SP",
    "41": "PR",
    "42": "SC",
    "43": "RS",
    "50": "MS",
    "51": "MT",
    "52": "GO",
    "53": "DF",
}


@dataclass(frozen=True)
class QueryParam:
    """Parâmetro tipado para consulta SQL parametrizada (BigQuery ScalarQueryParameter)."""

    name: str
    bq_type: str
    value: str | int | float


@dataclass(frozen=True)
class QuerySpec:
    """Especificação imutável de consulta parametrizada e contrato de saída."""

    name: str
    sql: str
    params: tuple[QueryParam, ...]
    schema_version: str
    expected_columns: tuple[str, ...]


@dataclass(frozen=True)
class IngestionConfig:
    """Parâmetros de execução de ingestão, teto de custo e limiar LGPD."""

    uf: CodigoUF
    ano_inicio: int
    ano_fim: int
    data_dir: Path
    max_mb_scanned: int = 2000
    min_cell_count: int = 5
    suppress_demand_microstrata: bool = True

    @property
    def max_bytes_billed(self) -> int:
        """Converte teto configurado de MB para bytes."""
        return self.max_mb_scanned * 1024 * 1024

    @property
    def sigla_uf(self) -> str:
        """Retorna a sigla oficial de duas letras correspondente ao código de UF IBGE."""
        return UF_CODE_TO_SIGLA[str(self.uf)]


@dataclass(frozen=True)
class IngestionManifest:
    """Metadados auditáveis gravados ao lado de cada snapshot Parquet em cache."""

    query_name: str
    cache_key: str
    schema_version: str
    source_name: str
    executed_at_utc: str
    bytes_scanned: int
    rows_returned: int
    rows_dropped: int
    suppressed_cells: int
    cache_hit: bool
    parquet_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class IngestionResult:
    """Retorno validado de uma etapa de ingestão com dados e manifesto."""

    dataframe: pl.DataFrame
    manifest: IngestionManifest


class TabularSourcePort(Protocol):
    """Costura real com dois adaptadores: BigQueryAdapter e ParquetFixtureAdapter."""

    @property
    def source_name(self) -> str: ...

    def inspect_columns(self, table_ref: str) -> tuple[str, ...]: ...

    def estimate_bytes(self, spec: QuerySpec) -> int | None: ...

    def fetch(self, spec: QuerySpec, *, max_bytes_billed: int) -> pl.DataFrame: ...
