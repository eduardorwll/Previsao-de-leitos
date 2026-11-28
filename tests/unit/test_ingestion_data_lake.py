"""Testes do data lake Hive Parquet em Polars (Task 2.1, ADR-0008)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from previsao_leitos.domain import DataContractError
from previsao_leitos.ingestion import (
    scan_hive_partitioned,
    sink_hive_partitioned,
    write_hive_partitioned,
)


def _df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "uf": ["RS", "RS", "SC"],
            "bacia": ["Guaiba", "Guaiba", "Itajai"],
            "ano_epi": [2024, 2023, 2024],
            "valor": [10.0, 20.0, 30.0],
        }
    )


def test_write_and_scan_round_trip(tmp_path: Path) -> None:
    df = _df()
    written = write_hive_partitioned(df, root=tmp_path)
    assert len(written) == 2  # (RS, Guaiba) + (SC, Itajai)

    got = scan_hive_partitioned(tmp_path, lazy=False).select(df.columns)
    assert set(got.columns) == {"uf", "bacia", "ano_epi", "valor"}
    assert got.sort("valor").equals(df.sort("valor"))


def test_hive_layout_partition_dirs(tmp_path: Path) -> None:
    write_hive_partitioned(_df(), root=tmp_path)
    assert (tmp_path / "uf=RS" / "bacia=Guaiba" / "part-0.parquet").exists()
    assert (tmp_path / "uf=SC" / "bacia=Itajai" / "part-0.parquet").exists()


def test_sink_hive_partitioned_matches_in_memory(tmp_path: Path) -> None:
    lazy = _df().lazy()
    written = sink_hive_partitioned(lazy, root=tmp_path)
    assert len(written) == 2
    got = scan_hive_partitioned(tmp_path, lazy=False).select(_df().columns)
    assert got.sort("valor").equals(_df().sort("valor"))


def test_unsafe_partition_col_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DataContractError):
        write_hive_partitioned(_df(), root=tmp_path, partition_cols=("cpf",))


def test_missing_partition_col_raises(tmp_path: Path) -> None:
    with pytest.raises(DataContractError):
        write_hive_partitioned(_df().drop("uf"), root=tmp_path, partition_cols=("uf",))


def test_scan_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(DataContractError):
        scan_hive_partitioned(tmp_path / "nao_existe")
