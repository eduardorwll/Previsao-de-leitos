"""Adaptadores reais atrás de TabularSourcePort: BigQueryAdapter e ParquetFixtureAdapter."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import polars as pl
from google.cloud import bigquery

from previsao_leitos.domain import (
    ConfigurationError,
    DataContractError,
    SourceUnavailableError,
)
from previsao_leitos.ingestion.ports import QuerySpec

_TABLE_REF_RE = re.compile(r"^[a-zA-Z0-9_-]+\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+$")
_SAFE_FIXTURE_RE = re.compile(r"^[a-z0-9_]+$")


class BigQueryAdapter:
    """Adaptador BigQuery com consultas 100% parametrizadas e guarda obrigatória via dry_run."""

    def __init__(
        self,
        *,
        project_id: str | None = None,
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        elif project_id:
            self._client = bigquery.Client(project=project_id)
        else:
            raise ConfigurationError(
                "Forneça `project_id` ou um `client` BigQuery explicitamente injetado."
            )

    @property
    def source_name(self) -> str:
        return "bigquery"

    def inspect_columns(self, table_ref: str) -> tuple[str, ...]:
        """Consulta metadados gratuitos (0 bytes) validando o identificador da tabela."""
        if not _TABLE_REF_RE.match(table_ref):
            raise DataContractError(f"Identificador de tabela BigQuery inválido: {table_ref!r}")
        table = self._client.get_table(table_ref)
        return tuple(str(field.name) for field in table.schema)

    def _build_query_parameters(self, spec: QuerySpec) -> list[bigquery.ScalarQueryParameter]:
        return [
            bigquery.ScalarQueryParameter(param.name, param.bq_type, param.value)
            for param in spec.params
        ]

    def estimate_bytes(self, spec: QuerySpec) -> int | None:
        """Executa `dry_run=True` (0 bytes cobrados) validando SQL e estimando bytes.

        Em tabelas BigQuery com Row-Level Security (ex.: `basedosdados.br_ms_cnes`),
        o dry_run valida sintaxe/colunas/parâmetros mas retorna `None` para bytes.
        """
        dry_cfg = bigquery.QueryJobConfig(
            query_parameters=self._build_query_parameters(spec),
            dry_run=True,
            use_query_cache=False,
        )
        job = self._client.query(spec.sql, job_config=dry_cfg)
        raw_bytes = job.total_bytes_processed
        return None if raw_bytes is None else int(raw_bytes)

    def fetch(self, spec: QuerySpec, *, max_bytes_billed: int) -> pl.DataFrame:
        """Valida custo via dry_run antes de executar consulta faturável e retorna pl.DataFrame."""
        if max_bytes_billed <= 0:
            raise ConfigurationError("max_bytes_billed deve ser estritamente positivo.")

        estimated_bytes = self.estimate_bytes(spec)
        if estimated_bytes is not None and estimated_bytes > max_bytes_billed:
            est_mb = estimated_bytes / (1024 * 1024)
            limit_mb = max_bytes_billed / (1024 * 1024)
            raise ConfigurationError(
                f"Consulta {spec.name!r} bloqueada no dry_run: estimativa de {est_mb:.1f} MB "
                f"excede o teto permitido de {limit_mb:.1f} MB."
            )

        run_cfg = bigquery.QueryJobConfig(
            query_parameters=self._build_query_parameters(spec),
            maximum_bytes_billed=max_bytes_billed,
            use_query_cache=True,
        )
        job = self._client.query(spec.sql, job_config=run_cfg)
        self.last_bytes_processed: int = int(job.total_bytes_processed or 0)
        arrow_table = job.to_arrow()
        result = pl.from_arrow(arrow_table)
        if not isinstance(result, pl.DataFrame):
            raise DataContractError(f"Retorno inesperado da consulta {spec.name!r}.")
        return result


class ParquetFixtureAdapter:
    """Adaptador offline determinístico para testes unitários, integração e CI sem nuvem."""

    def __init__(self, fixtures_dir: Path) -> None:
        self._fixtures_dir = fixtures_dir.resolve()

    @property
    def source_name(self) -> str:
        return "parquet_fixture"

    def inspect_columns(self, table_ref: str) -> tuple[str, ...]:
        fixture_name = table_ref.split(".")[-1]
        path = self._resolve_path(fixture_name)
        return tuple(pl.scan_parquet(path).collect_schema().names())

    def estimate_bytes(self, spec: QuerySpec) -> int:
        return 0

    def fetch(self, spec: QuerySpec, *, max_bytes_billed: int) -> pl.DataFrame:
        path = self._resolve_path(spec.name)
        return pl.read_parquet(path)

    def _resolve_path(self, name: str) -> Path:
        if not _SAFE_FIXTURE_RE.match(name):
            raise DataContractError(f"Nome de fixture inválido: {name!r}")
        path = (self._fixtures_dir / f"{name}.parquet").resolve()
        if self._fixtures_dir not in path.parents or not path.exists():
            raise SourceUnavailableError(f"Fixture Parquet não encontrada: {path}")
        return path
