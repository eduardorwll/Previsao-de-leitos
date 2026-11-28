"""Data lake nacional em Polars, 2 níveis (`ADR-0008`) — Hive Parquet por UF/bacia.

Estratégia de escala (`Task 2.1`) rejeitando o `dask`:
- **Nível 1 — passe global** (`< 16 GiB`): `scan_parquet(..., hive_partitioning=True)`
  sobre todas as partições, permitindo junções exatas de Ottobacias e `W_ref`
  interestaduais em nó único.
- **Nível 2 — streaming portátil** (`< 6 GiB`): materialização partição a partição
  com `sink_parquet` (nunca carrega o recorte inteiro em memória).

Layout: `root/uf=RS/bacia=Guaiba/part-0.parquet`. Escrita atômica (`.tmp` + `os.replace`).

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from previsao_leitos.domain import DataContractError

_SAFE_PARTITION_RE = ("uf", "bacia", "regiao", "ano_epi")


def _validate_partition_cols(partition_cols: Sequence[str]) -> None:
    if not partition_cols:
        raise DataContractError("partition_cols não pode ser vazio.")
    unsafe = [c for c in partition_cols if c not in _SAFE_PARTITION_RE]
    if unsafe:
        # Restringe a atributos territoriais conhecidos (evita partição por PII/medida).
        raise DataContractError(
            f"Partições não permitidas {unsafe}; use apenas {list(_SAFE_PARTITION_RE)}."
        )


def _partition_dir(root: Path, parts: dict[str, object]) -> Path:
    return root.joinpath(*[f"{key}={value}" for key, value in parts.items()])


def write_hive_partitioned(
    df: pl.DataFrame,
    *,
    root: Path,
    partition_cols: Sequence[str] = ("uf", "bacia"),
    filename: str = "part-0.parquet",
) -> list[Path]:
    """Grava `df` em Hive Parquet atômico, uma pasta por combinação de partição."""
    _validate_partition_cols(partition_cols)
    missing = [c for c in partition_cols if c not in df.columns]
    if missing:
        raise DataContractError(f"Colunas de partição ausentes no DataFrame: {missing}.")

    resolved_root = root.resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    groups = df.partition_by(list(partition_cols), as_dict=True, maintain_order=True)
    for keys, group in groups.items():
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        parts = dict(zip(partition_cols, key_tuple, strict=True))
        part_dir = _partition_dir(resolved_root, parts).resolve()
        if resolved_root not in part_dir.parents:
            raise DataContractError("Caminho de partição escapou do diretório raiz.")
        part_dir.mkdir(parents=True, exist_ok=True)
        out = part_dir / filename
        tmp = out.with_suffix(".parquet.tmp")
        group.drop(list(partition_cols)).write_parquet(tmp)
        os.replace(tmp, out)
        written.append(out)
    return written


def scan_hive_partitioned(root: Path, *, lazy: bool = True) -> pl.LazyFrame | pl.DataFrame:
    """Scan (lazy) de todas as partições Hive, com pushdown de partição/projeção."""
    resolved_root = root.resolve()
    if not resolved_root.exists():
        raise DataContractError(f"Data lake inexistente: {resolved_root}.")
    lazy_frame = pl.scan_parquet(
        str(resolved_root / "**" / "*.parquet"),
        hive_partitioning=True,
    )
    return lazy_frame if lazy else lazy_frame.collect()


def sink_hive_partitioned(
    lazy_frame: pl.LazyFrame,
    *,
    root: Path,
    partition_cols: Sequence[str] = ("uf", "bacia"),
    filename: str = "part-0.parquet",
) -> list[Path]:
    """Nível 2 (streaming): materializa partição a partição com `sink_parquet`.

    Enumera as combinações de partição com um `select(...).unique()` barato e então
    filtra + `sink_parquet` por partição, mantendo o uso de memória baixo.
    """
    _validate_partition_cols(partition_cols)
    resolved_root = root.resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)

    combos = (
        lazy_frame.select(list(partition_cols)).unique(maintain_order=True).collect()
    )
    if combos.is_empty():
        raise DataContractError("Data lake vazio: nenhuma partição para materializar.")

    written: list[Path] = []
    for row in combos.iter_rows(named=True):
        condition = None
        for col in partition_cols:
            expr = pl.col(col) == row[col]
            condition = expr if condition is None else condition & expr
        assert condition is not None
        part_dir = _partition_dir(resolved_root, row).resolve()
        if resolved_root not in part_dir.parents:
            raise DataContractError("Caminho de partição escapou do diretório raiz.")
        part_dir.mkdir(parents=True, exist_ok=True)
        out = part_dir / filename
        tmp = out.with_suffix(".parquet.tmp")
        lazy_frame.filter(condition).drop(list(partition_cols)).sink_parquet(tmp)
        os.replace(tmp, out)
        written.append(out)
    return written
