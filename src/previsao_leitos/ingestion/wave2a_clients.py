"""Clientes HTTP ao vivo da Onda 2A com cache atômico SHA-256 e conformidade LGPD.

Fontes suportadas (todas injetáveis via `HttpFetcher` para testes 100% offline no CI):
- NOAA CPC ONI (`oni.ascii.txt`) e NOAA PSL MEI.v2 (`meiv2.data`)
- Zenodo IPH-UFRGS / ANA RS 2024 (`DOI 10.5281/zenodo.16929471`)
- GERINT-POA Dados Abertos (API CKAN `dadosabertos.poa.br`)
- OpenDataSUS (`SIVEP-Gripe` e `SINAN` Leptospirose) com agregação imediata LGPD (n < 5)

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol

import polars as pl

from previsao_leitos.domain import DataContractError, SourceUnavailableError
from previsao_leitos.geo.enso_teleconnections import (
    parse_bom_rnino_table,
    parse_noaa_oni_table,
    parse_noaa_roni_table,
    parse_noaa_soi_table,
)
from previsao_leitos.ingestion.epidemiology_sources import (
    aggregate_gerint_poa_weekly,
    aggregate_sinan_leptospirose_weekly,
    aggregate_sivep_gripe_weekly,
)

NOAA_ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
NOAA_MEI_V2_URL = "https://psl.noaa.gov/enso/mei/data/meiv2.data"
#: RONI oficial do CPC (índice relativo). O JSON do elninolive (`/api/enso/roni`) é
#: apenas cross-check de parsing em desenvolvimento; nunca é dependência de runtime.
NOAA_RONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt"
#: SOI oficial do CPC (componente atmosférico do ENSO, bloco STANDARDIZED).
NOAA_SOI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/soi"
#: BOM Niño 3.4 relativo semanal (limiar El Niño +0,8 °C, diferente do NOAA ±0,5 °C).
BOM_RNINO_URL = "https://www.bom.gov.au/clim_data/IDCK000072/rnino_3.4.txt"
ZENODO_IPH_RS_RECORD_URL = "https://zenodo.org/api/records/16929471"
GERINT_POA_CKAN_SEARCH_URL = "https://dadosabertos.poa.br/api/3/action/package_search?q=gerint"


class HttpFetcher(Protocol):
    """Protocolo mínimo de download HTTP injetável para testes offline determinísticos."""

    def get_bytes(self, url: str, *, timeout: float = 30.0) -> bytes: ...


class UrllibHttpFetcher:
    """Adaptador HTTP padrão baseado em `urllib.request` com validação de esquema HTTPS."""

    def __init__(self, *, user_agent: str = "PrevisaoDeLeitos-Research/1.2.2") -> None:
        self._user_agent = user_agent

    def get_bytes(self, url: str, *, timeout: float = 30.0) -> bytes:
        if not url.startswith("https://"):
            raise DataContractError(f"Apenas URLs HTTPS são permitidas na ingestão: {url!r}")
        req = urllib.request.Request(  # noqa: S310
            url, headers={"User-Agent": self._user_agent}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                data = resp.read()
        except Exception as exc:
            raise SourceUnavailableError(f"Falha ao baixar {url}: {exc}") from exc
        if not isinstance(data, bytes):
            raise SourceUnavailableError(f"Resposta inválida de {url}.")
        return data


@dataclass(frozen=True)
class CachedPayload:
    """Artefato bruto em cache atômico verificado por SHA-256."""

    url: str
    cache_path: Path
    manifest_path: Path
    sha256: str
    size_bytes: int
    cache_hit: bool


def fetch_cached_bytes(
    url: str,
    *,
    cache_dir: Path,
    filename: str,
    fetcher: HttpFetcher | None = None,
    timeout: float = 30.0,
    force_refresh: bool = False,
) -> tuple[bytes, CachedPayload]:
    """Baixa recurso HTTP com cache local atômico e manifesto SHA-256."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / filename
    manifest_path = cache_dir / f"{filename}.manifest.json"

    if target.exists() and not force_refresh:
        raw = target.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        return raw, CachedPayload(
            url=url,
            cache_path=target,
            manifest_path=manifest_path,
            sha256=digest,
            size_bytes=len(raw),
            cache_hit=True,
        )

    active_fetcher = fetcher or UrllibHttpFetcher()
    raw = active_fetcher.get_bytes(url, timeout=timeout)
    digest = hashlib.sha256(raw).hexdigest()

    tmp = target.with_suffix(f"{target.suffix}.tmp")
    tmp.write_bytes(raw)
    tmp.replace(target)

    manifest = {
        "url": url,
        "filename": filename,
        "sha256": digest,
        "size_bytes": len(raw),
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
    }
    tmp_m = manifest_path.with_suffix(".json.tmp")
    tmp_m.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_m.replace(manifest_path)

    return raw, CachedPayload(
        url=url,
        cache_path=target,
        manifest_path=manifest_path,
        sha256=digest,
        size_bytes=len(raw),
        cache_hit=False,
    )


def parse_noaa_mei_v2_table(raw_text: str) -> pl.DataFrame:
    """Converte tabela mensal oficial NOAA PSL MEI.v2 em DataFrame `(data_ref, mei_v2)`."""
    rows: list[dict[str, object]] = []
    for line in raw_text.splitlines():
        parts = line.strip().split()
        if len(parts) != 13 or not parts[0].isdigit():
            continue
        yr = int(parts[0])
        if yr < 1979 or yr > 2100:
            continue
        for month_idx, token in enumerate(parts[1:], start=1):
            val = float(token)
            if val <= -99.0:
                continue
            rows.append({"data_ref": date(yr, month_idx, 1), "mei_v2": val})
    if not rows:
        raise DataContractError("parse_noaa_mei_v2_table: nenhum registro válido encontrado.")
    return pl.DataFrame(rows).sort("data_ref")


def fetch_noaa_enso_monthly(
    *,
    cache_dir: Path,
    fetcher: HttpFetcher | None = None,
    include_mei_v2: bool = True,
    max_year: int | None = None,
) -> tuple[pl.DataFrame, list[CachedPayload]]:
    """Baixa e consolida índices mensais oficiais NOAA CPC ONI (+ MEI.v2 opcional)."""
    manifests: list[CachedPayload] = []
    oni_bytes, oni_meta = fetch_cached_bytes(
        NOAA_ONI_URL,
        cache_dir=cache_dir,
        filename="noaa_cpc_oni.ascii.txt",
        fetcher=fetcher,
    )
    manifests.append(oni_meta)
    df = parse_noaa_oni_table(oni_bytes.decode("utf-8", errors="replace")).with_columns(
        pl.col("oni_c").alias("nino34_anom_c")
    )

    if include_mei_v2:
        try:
            mei_bytes, mei_meta = fetch_cached_bytes(
                NOAA_MEI_V2_URL,
                cache_dir=cache_dir,
                filename="noaa_psl_meiv2.data",
                fetcher=fetcher,
            )
            manifests.append(mei_meta)
            mei_df = parse_noaa_mei_v2_table(mei_bytes.decode("utf-8", errors="replace"))
            df = df.join(mei_df, on="data_ref", how="left")
        except (SourceUnavailableError, DataContractError):
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("mei_v2"))

    if max_year is not None:
        df = df.filter(pl.col("data_ref").dt.year() <= max_year)
    return df.sort("data_ref"), manifests


def fetch_noaa_roni_monthly(
    *,
    cache_dir: Path,
    fetcher: HttpFetcher | None = None,
    max_year: int | None = None,
) -> tuple[pl.DataFrame, CachedPayload]:
    """Baixa e parseia a série mensal oficial NOAA CPC RONI (1950-presente)."""
    roni_bytes, roni_meta = fetch_cached_bytes(
        NOAA_RONI_URL,
        cache_dir=cache_dir,
        filename="noaa_cpc_roni.ascii.txt",
        fetcher=fetcher,
    )
    df = parse_noaa_roni_table(roni_bytes.decode("utf-8", errors="replace"))
    if max_year is not None:
        df = df.filter(pl.col("data_ref").dt.year() <= max_year)
    return df.sort("data_ref"), roni_meta


def fetch_noaa_soi_monthly(
    *,
    cache_dir: Path,
    fetcher: HttpFetcher | None = None,
    max_year: int | None = None,
) -> tuple[pl.DataFrame, CachedPayload]:
    """Baixa e parseia a série mensal oficial NOAA CPC SOI (STANDARDIZED, 1951-presente)."""
    soi_bytes, soi_meta = fetch_cached_bytes(
        NOAA_SOI_URL,
        cache_dir=cache_dir,
        filename="noaa_cpc_soi.txt",
        fetcher=fetcher,
    )
    df = parse_noaa_soi_table(soi_bytes.decode("utf-8", errors="replace"))
    if max_year is not None:
        df = df.filter(pl.col("data_ref").dt.year() <= max_year)
    return df.sort("data_ref"), soi_meta


def fetch_bom_rnino_weekly(
    *,
    cache_dir: Path,
    fetcher: HttpFetcher | None = None,
    max_year: int | None = None,
) -> tuple[pl.DataFrame, CachedPayload]:
    """Baixa e parseia a série semanal oficial BOM Niño 3.4 relativo (2008-presente)."""
    bom_bytes, bom_meta = fetch_cached_bytes(
        BOM_RNINO_URL,
        cache_dir=cache_dir,
        filename="bom_rnino_3_4.txt",
        fetcher=fetcher,
    )
    df = parse_bom_rnino_table(bom_bytes.decode("utf-8", errors="replace"))
    if max_year is not None:
        df = df.filter(pl.col("week_start").dt.year() <= max_year)
    return df.sort("week_start"), bom_meta


@dataclass(frozen=True)
class ZenodoRecordSummary:
    """Resumo de arquivos disponíveis no registro Zenodo IPH-UFRGS/ANA (16929471)."""

    record_id: int
    doi: str
    title: str
    files: tuple[tuple[str, int, str], ...]
    manifest: CachedPayload


def fetch_zenodo_iph_rs_metadata(
    *,
    cache_dir: Path,
    fetcher: HttpFetcher | None = None,
) -> ZenodoRecordSummary:
    """Consulta e cacheia metadados oficiais do registro Zenodo 16929471 (IPH-UFRGS/ANA)."""
    raw, meta = fetch_cached_bytes(
        ZENODO_IPH_RS_RECORD_URL,
        cache_dir=cache_dir,
        filename="zenodo_16929471_record.json",
        fetcher=fetcher,
    )
    payload = json.loads(raw.decode("utf-8"))
    files_list: list[tuple[str, int, str]] = []
    for fobj in payload.get("files", []):
        key = str(fobj.get("key", ""))
        size = int(fobj.get("size", 0))
        dl_url = str(fobj.get("links", {}).get("self", ""))
        if key and dl_url:
            files_list.append((key, size, dl_url))

    return ZenodoRecordSummary(
        record_id=int(payload.get("id", 16929471)),
        doi=str(payload.get("doi", "10.5281/zenodo.16929471")),
        title=str(payload.get("metadata", {}).get("title", "IPH-UFRGS RS Flood Dataset")),
        files=tuple(files_list),
        manifest=meta,
    )


def discover_gerint_poa_csv_urls(
    *,
    cache_dir: Path,
    fetcher: HttpFetcher | None = None,
) -> tuple[list[tuple[str, str]], CachedPayload]:
    """Descobre recursos CSV abertos do GERINT-POA na API CKAN de Porto Alegre."""
    raw, meta = fetch_cached_bytes(
        GERINT_POA_CKAN_SEARCH_URL,
        cache_dir=cache_dir,
        filename="gerint_poa_ckan_search.json",
        fetcher=fetcher,
    )
    payload = json.loads(raw.decode("utf-8"))
    results = payload.get("result", {}).get("results", [])
    resources: list[tuple[str, str]] = []
    for pkg in results:
        for res in pkg.get("resources", []):
            fmt = str(res.get("format", "")).upper()
            url = str(res.get("url", ""))
            name = str(res.get("name", "gerint_resource"))
            if fmt == "CSV" and url.startswith("https://"):
                resources.append((name, url))
    return resources, meta


def ingest_gerint_poa_csv_lgpd(
    url: str,
    *,
    cache_dir: Path,
    filename: str = "gerint_poa_raw.csv",
    fetcher: HttpFetcher | None = None,
    min_cell_count: int = 5,
    separator: str = ";",
) -> tuple[pl.DataFrame, CachedPayload]:
    """Baixa CSV GERINT-POA e devolve exclusivamente agregado semanal LGPD (zero PII)."""
    raw, meta = fetch_cached_bytes(url, cache_dir=cache_dir, filename=filename, fetcher=fetcher)
    df = pl.read_csv(
        io.BytesIO(raw), separator=separator, try_parse_dates=True, ignore_errors=True
    )
    agg = aggregate_gerint_poa_weekly(df, min_cell_count=min_cell_count)
    return agg, meta


def ingest_opendatasus_sivep_csv_lgpd(
    url: str,
    *,
    cache_dir: Path,
    filename: str = "sivep_gripe_raw.csv",
    fetcher: HttpFetcher | None = None,
    min_cell_count: int = 5,
    separator: str = ";",
) -> tuple[pl.DataFrame, CachedPayload]:
    """Baixa CSV SIVEP-Gripe e devolve agregado semanal LGPD (zero PII)."""
    raw, meta = fetch_cached_bytes(url, cache_dir=cache_dir, filename=filename, fetcher=fetcher)
    df = pl.read_csv(
        io.BytesIO(raw), separator=separator, try_parse_dates=True, ignore_errors=True
    )
    agg = aggregate_sivep_gripe_weekly(df, min_cell_count=min_cell_count)
    return agg, meta


def ingest_opendatasus_sinan_csv_lgpd(
    url: str,
    *,
    cache_dir: Path,
    filename: str = "sinan_lepto_raw.csv",
    fetcher: HttpFetcher | None = None,
    min_cell_count: int = 5,
    separator: str = ";",
) -> tuple[pl.DataFrame, CachedPayload]:
    """Baixa CSV SINAN Leptospirose e devolve agregado semanal LGPD (zero PII)."""
    raw, meta = fetch_cached_bytes(url, cache_dir=cache_dir, filename=filename, fetcher=fetcher)
    df = pl.read_csv(
        io.BytesIO(raw), separator=separator, try_parse_dates=True, ignore_errors=True
    )
    agg = aggregate_sinan_leptospirose_weekly(df, min_cell_count=min_cell_count)
    return agg, meta
