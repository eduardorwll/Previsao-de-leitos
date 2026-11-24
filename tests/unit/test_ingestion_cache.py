"""Testes unitários do cache Parquet atômico com chave determinística SHA-256."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from previsao_leitos.domain import ConfigurationError
from previsao_leitos.ingestion.cache import AtomicParquetCache
from previsao_leitos.ingestion.ports import QueryParam, QuerySpec


def _sample_spec(ano_fim: int = 2024, schema_version: str = "v1") -> QuerySpec:
    return QuerySpec(
        name="cnes_leitos_rs",
        sql="SELECT id_estabelecimento_cnes FROM tbl WHERE ano BETWEEN @ano_inicio AND @ano_fim",
        params=(
            QueryParam("ano_inicio", "INT64", 2010),
            QueryParam("ano_fim", "INT64", ano_fim),
        ),
        schema_version=schema_version,
        expected_columns=("id_estabelecimento_cnes",),
    )


def test_cache_key_changes_when_params_or_schema_version_change(tmp_path: Path) -> None:
    cache = AtomicParquetCache(root_dir=tmp_path)
    k_base = cache.compute_key(_sample_spec(ano_fim=2024, schema_version="v1"))
    k_same = cache.compute_key(_sample_spec(ano_fim=2024, schema_version="v1"))
    k_year = cache.compute_key(_sample_spec(ano_fim=2025, schema_version="v1"))
    k_schema = cache.compute_key(_sample_spec(ano_fim=2024, schema_version="v2"))

    assert k_base == k_same
    assert k_base != k_year
    assert k_base != k_schema


def test_atomic_write_persists_parquet_and_manifest_without_leftover_tmp(tmp_path: Path) -> None:
    cache = AtomicParquetCache(root_dir=tmp_path)
    spec = _sample_spec()
    df = pl.DataFrame({"id_estabelecimento_cnes": ["0012345", "2237601"]})

    manifest = cache.put(
        spec,
        df,
        bytes_scanned=1024,
        rows_dropped=0,
        suppressed_cells=0,
        source_name="fixture",
    )

    loaded = cache.get(spec)
    assert loaded is not None
    assert loaded["id_estabelecimento_cnes"].to_list() == ["0012345", "2237601"]
    assert manifest.rows_returned == 2
    assert manifest.bytes_scanned == 1024
    assert list(tmp_path.rglob("*.tmp")) == []
    assert manifest.manifest_path.exists()


def test_cache_blocks_path_traversal_in_query_name(tmp_path: Path) -> None:
    cache = AtomicParquetCache(root_dir=tmp_path)
    evil_spec = QuerySpec(
        name="../../etc/passwd",
        sql="SELECT 1",
        params=(),
        schema_version="v1",
        expected_columns=("x",),
    )

    with pytest.raises(ConfigurationError):
        cache.paths_for(evil_spec)
