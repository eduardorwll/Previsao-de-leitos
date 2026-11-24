"""Cache Parquet atômico com chave determinística SHA-256 e confinamento em diretório raiz."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from previsao_leitos.domain import ConfigurationError
from previsao_leitos.ingestion.ports import IngestionManifest, IngestionResult, QuerySpec

_SAFE_QUERY_NAME_RE = re.compile(r"^[a-z0-9_]+$")


class AtomicParquetCache:
    """Armazena snapshots Parquet e manifestos JSON sob `root_dir` via escrita atômica."""

    def __init__(self, root_dir: Path) -> None:
        self._root = root_dir.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root_dir(self) -> Path:
        return self._root

    def compute_key(self, spec: QuerySpec) -> str:
        """Gera hash SHA-256 determinístico de SQL, parâmetros ordenados e versão do schema."""
        payload = {
            "name": spec.name,
            "sql": spec.sql.strip(),
            "params": sorted(
                [
                    {"name": p.name, "bq_type": p.bq_type, "value": p.value}
                    for p in spec.params
                ],
                key=lambda item: str(item["name"]),
            ),
            "schema_version": spec.schema_version,
            "expected_columns": list(spec.expected_columns),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def paths_for(self, spec: QuerySpec) -> tuple[Path, Path]:
        """Resolve caminhos seguros (parquet, manifesto) garantindo confinamento em `root_dir`."""
        if not _SAFE_QUERY_NAME_RE.match(spec.name):
            raise ConfigurationError(
                f"Nome de consulta inválido para cache (permitido [a-z0-9_]): {spec.name!r}"
            )
        key = self.compute_key(spec)
        stem = f"{spec.name}_{spec.schema_version}_{key}"
        parquet_path = (self._root / f"{stem}.parquet").resolve()
        manifest_path = (self._root / f"{stem}.manifest.json").resolve()

        if self._root not in parquet_path.parents or self._root not in manifest_path.parents:
            raise ConfigurationError("Caminho de cache escapou do diretório raiz permitido.")
        return parquet_path, manifest_path

    def get(self, spec: QuerySpec) -> pl.DataFrame | None:
        """Lê snapshot em cache se existir, retornando `None` em cache miss."""
        parquet_path, _ = self.paths_for(spec)
        if not parquet_path.exists():
            return None
        return pl.read_parquet(parquet_path)

    def get_with_manifest(self, spec: QuerySpec) -> IngestionResult | None:
        """Lê snapshot e manifesto em cache marcando `cache_hit=True`."""
        parquet_path, manifest_path = self.paths_for(spec)
        if not parquet_path.exists() or not manifest_path.exists():
            return None

        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = IngestionManifest(
            query_name=str(raw["query_name"]),
            cache_key=str(raw["cache_key"]),
            schema_version=str(raw["schema_version"]),
            source_name=str(raw["source_name"]),
            executed_at_utc=str(raw["executed_at_utc"]),
            bytes_scanned=int(raw["bytes_scanned"]),
            rows_returned=int(raw["rows_returned"]),
            rows_dropped=int(raw["rows_dropped"]),
            suppressed_cells=int(raw["suppressed_cells"]),
            cache_hit=True,
            parquet_path=parquet_path,
            manifest_path=manifest_path,
        )
        return IngestionResult(dataframe=pl.read_parquet(parquet_path), manifest=manifest)

    def put(
        self,
        spec: QuerySpec,
        df: pl.DataFrame,
        *,
        bytes_scanned: int,
        rows_dropped: int,
        suppressed_cells: int,
        source_name: str,
    ) -> IngestionManifest:
        """Grava Parquet e manifesto JSON de forma atômica via `.tmp` + `os.replace`."""
        parquet_path, manifest_path = self.paths_for(spec)
        tmp_parquet = parquet_path.with_suffix(".parquet.tmp")
        tmp_manifest = manifest_path.with_suffix(".json.tmp")

        df.write_parquet(tmp_parquet)
        os.replace(tmp_parquet, parquet_path)

        executed_at_utc = datetime.now(UTC).isoformat()
        cache_key = self.compute_key(spec)
        manifest_payload = {
            "query_name": spec.name,
            "cache_key": cache_key,
            "schema_version": spec.schema_version,
            "source_name": source_name,
            "executed_at_utc": executed_at_utc,
            "bytes_scanned": bytes_scanned,
            "rows_returned": df.height,
            "rows_dropped": rows_dropped,
            "suppressed_cells": suppressed_cells,
            "cache_hit": False,
            "parquet_file": parquet_path.name,
        }
        tmp_manifest.write_text(
            json.dumps(manifest_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp_manifest, manifest_path)

        return IngestionManifest(
            query_name=spec.name,
            cache_key=cache_key,
            schema_version=spec.schema_version,
            source_name=source_name,
            executed_at_utc=executed_at_utc,
            bytes_scanned=bytes_scanned,
            rows_returned=df.height,
            rows_dropped=rows_dropped,
            suppressed_cells=suppressed_cells,
            cache_hit=False,
            parquet_path=parquet_path,
            manifest_path=manifest_path,
        )
