"""Adaptadores de dados hidrológicos 100% RS (Open-Meteo ERA5-Land + Fixture offline)."""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import polars as pl

from previsao_leitos.domain import ConfigurationError, SourceUnavailableError


@dataclass(frozen=True)
class RegionCentroid:
    """Centróide geográfico (WGS84) de uma Região de Saúde ou município."""

    entity_id: str
    latitude: float
    longitude: float


class HydroSourcePort(Protocol):
    """Costura para fontes hidrológicas diárias contínuas (ERA5-Land/MERGE vs. Fixture offline)."""

    def fetch_daily_hydro(
        self,
        *,
        centroids: tuple[RegionCentroid, ...],
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame: ...


class HydroFixtureAdapter:
    """Adaptador offline determinístico para testes unitários e de integração."""

    def __init__(self, parquet_path: Path) -> None:
        self._path = parquet_path.resolve()

    def fetch_daily_hydro(
        self,
        *,
        centroids: tuple[RegionCentroid, ...],
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        if not self._path.exists():
            raise SourceUnavailableError(f"Fixture hidrológica não encontrada: {self._path}")
        return pl.read_parquet(self._path).filter(
            (pl.col("data") >= start_date) & (pl.col("data") <= end_date)
        )


class OpenMeteoEra5Adapter:
    """Extrai reanálise ERA5-Land diária (chuva, umidade do solo e temperatura) com cache atômico.

    Cobre 100% do território sem chave de API, resolvendo a lacuna de ~90% das estações INMET.
    """

    _BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(
        self,
        *,
        cache_dir: Path,
        timeout_seconds: float = 45.0,
        pause_seconds: float = 1.0,
        batch_size: int = 10,
        max_retries: int = 3,
        backoff_seconds: float = 5.0,
    ) -> None:
        if batch_size < 1:
            raise ConfigurationError(f"batch_size deve ser >= 1 (recebido {batch_size}).")
        self._cache_dir = cache_dir.resolve()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout_seconds
        self._pause = pause_seconds
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._backoff = backoff_seconds

    def _fetch_json_with_retry(self, url: str, label: str) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                with urlopen(url, timeout=self._timeout) as resp:  # noqa: S310  # HTTPS fixo
                    return json.loads(resp.read().decode("utf-8"))
            except HTTPError as exc:
                last_exc = exc
                if exc.code in (429, 500, 502, 503, 504) and attempt < self._max_retries:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    wait = (
                        float(retry_after) + 2.0
                        if retry_after and retry_after.isdigit()
                        else self._backoff * (2**attempt)
                    )
                    if wait > 0:
                        time.sleep(wait)
                    continue
                raise SourceUnavailableError(
                    f"Falha ao consultar Open-Meteo ERA5-Land ({label}): {exc}"
                ) from exc
            except Exception as exc:
                raise SourceUnavailableError(
                    f"Falha ao consultar Open-Meteo ERA5-Land ({label}): {exc}"
                ) from exc
        raise SourceUnavailableError(
            f"Falha ao consultar Open-Meteo ERA5-Land ({label}): {last_exc}"
        ) from last_exc

    def _cache_path(
        self,
        centroids: tuple[RegionCentroid, ...],
        start_date: date,
        end_date: date,
    ) -> Path:
        payload = {
            "centroids": sorted(
                [(c.entity_id, round(c.latitude, 4), round(c.longitude, 4)) for c in centroids]
            ),
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "vars": ["precipitation_sum", "soil_moisture_0_to_7cm_mean", "temperature_2m_mean"],
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return self._cache_dir / f"era5_land_daily_{start_date.year}_{end_date.year}_{key}.parquet"

    def fetch_daily_hydro(
        self,
        *,
        centroids: tuple[RegionCentroid, ...],
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        if not centroids:
            raise ConfigurationError("Lista de centróides vazia para extração ERA5-Land.")

        cache_path = self._cache_path(centroids, start_date, end_date)
        if cache_path.exists():
            return pl.read_parquet(cache_path)

        entity_cache_dir = self._cache_dir / "era5_entities"
        entity_cache_dir.mkdir(parents=True, exist_ok=True)

        frames: list[pl.DataFrame] = []
        pending: list[RegionCentroid] = []
        for c in centroids:
            ec_path = entity_cache_dir / (
                f"{c.entity_id}_{start_date.isoformat()}_{end_date.isoformat()}.parquet"
            )
            if ec_path.exists():
                frames.append(pl.read_parquet(ec_path))
            else:
                pending.append(c)

        batches = [
            pending[i : i + self._batch_size]
            for i in range(0, len(pending), self._batch_size)
        ]
        for b_idx, batch in enumerate(batches):
            params = urlencode(
                {
                    "latitude": ",".join(f"{c.latitude:.4f}" for c in batch),
                    "longitude": ",".join(f"{c.longitude:.4f}" for c in batch),
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "daily": "precipitation_sum,temperature_2m_mean,soil_moisture_0_to_7cm_mean",
                    "timezone": "America/Sao_Paulo",
                }
            )
            url = f"{self._BASE_URL}?{params}"
            label = f"lote {b_idx + 1}/{len(batches)} ({batch[0].entity_id}..{batch[-1].entity_id})"
            raw_payload = self._fetch_json_with_retry(url, label)
            items: list[dict[str, Any]] = (
                raw_payload if isinstance(raw_payload, list) else [raw_payload]
            )
            if len(items) != len(batch):
                raise SourceUnavailableError(
                    f"Open-Meteo retornou {len(items)} locais para lote de {len(batch)} ({label})."
                )

            for c, item in zip(batch, items, strict=True):
                daily = item.get("daily", {})
                times = daily.get("time", [])
                if not times:
                    raise SourceUnavailableError(
                        f"Resposta vazia do Open-Meteo para {c.entity_id!r}."
                    )
                entity_df = pl.DataFrame(
                    {
                        "entity_id": [c.entity_id] * len(times),
                        "data": [date.fromisoformat(t) for t in times],
                        "precip_dia_mm": daily.get("precipitation_sum", [None] * len(times)),
                        "temp_media_c": daily.get("temperature_2m_mean", [None] * len(times)),
                        "umidade_solo_m3m3": daily.get(
                            "soil_moisture_0_to_7cm_mean", [None] * len(times)
                        ),
                    }
                ).with_columns(
                    pl.col("precip_dia_mm").cast(pl.Float64),
                    pl.col("temp_media_c").cast(pl.Float64),
                    pl.col("umidade_solo_m3m3").cast(pl.Float64),
                )
                ec_path = entity_cache_dir / (
                    f"{c.entity_id}_{start_date.isoformat()}_{end_date.isoformat()}.parquet"
                )
                ec_tmp = ec_path.with_suffix(".parquet.tmp")
                entity_df.write_parquet(ec_tmp)
                os.replace(ec_tmp, ec_path)
                frames.append(entity_df)
            if b_idx + 1 < len(batches) and self._pause > 0:
                time.sleep(self._pause)

        combined = pl.concat(frames).sort(["entity_id", "data"])
        tmp_path = cache_path.with_suffix(".parquet.tmp")
        combined.write_parquet(tmp_path)
        os.replace(tmp_path, cache_path)
        return combined


class NasaPowerHydroAdapter:
    """Adaptador NASA POWER (MERRA-2/IMERG `PRECTOTCORR`, `GWETTOP`, `T2M`) com cache atômico.

    Cobre 100% do território diário (1981-presente) sem chave de API e converte sentinelas
    NASA (`-999.0`) explicitamente para `null` (jamais confundidas com `0.0 mm`).
    """

    _BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
    _MISSING_SENTINEL = -900.0

    def __init__(
        self,
        *,
        cache_dir: Path,
        timeout_seconds: float = 45.0,
        max_workers: int = 4,
    ) -> None:
        if max_workers < 1:
            raise ConfigurationError(f"max_workers deve ser >= 1 (recebido {max_workers}).")
        self._cache_dir = cache_dir.resolve()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout_seconds
        self._max_workers = max_workers

    def _cache_path(
        self,
        centroids: tuple[RegionCentroid, ...],
        start_date: date,
        end_date: date,
    ) -> Path:
        payload = {
            "source": "nasa_power_v1",
            "centroids": sorted(
                [(c.entity_id, round(c.latitude, 4), round(c.longitude, 4)) for c in centroids]
            ),
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "vars": ["PRECTOTCORR", "GWETTOP", "T2M"],
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return self._cache_dir / f"nasa_power_daily_{start_date.year}_{end_date.year}_{key}.parquet"

    @classmethod
    def _clean_value(cls, val: object) -> float | None:
        if val is None:
            return None
        num = float(str(val))
        return None if num <= cls._MISSING_SENTINEL else num

    def _fetch_one(
        self,
        centroid: RegionCentroid,
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        params = urlencode(
            {
                "parameters": "PRECTOTCORR,GWETTOP,T2M",
                "community": "AG",
                "longitude": f"{centroid.longitude:.4f}",
                "latitude": f"{centroid.latitude:.4f}",
                "start": start_date.strftime("%Y%m%d"),
                "end": end_date.strftime("%Y%m%d"),
                "format": "JSON",
            }
        )
        url = f"{self._BASE_URL}?{params}"
        try:
            with urlopen(url, timeout=self._timeout) as resp:  # noqa: S310  # HTTPS fixo NASA
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise SourceUnavailableError(
                f"Falha ao consultar NASA POWER para {centroid.entity_id!r}: {exc}"
            ) from exc

        param_map = payload.get("properties", {}).get("parameter", {})
        precip_map: dict[str, object] = param_map.get("PRECTOTCORR", {})
        soil_map: dict[str, object] = param_map.get("GWETTOP", {})
        temp_map: dict[str, object] = param_map.get("T2M", {})
        if not precip_map:
            raise SourceUnavailableError(
                f"Resposta NASA POWER sem série PRECTOTCORR para {centroid.entity_id!r}."
            )

        keys = sorted(precip_map.keys())
        dates = [date(int(k[:4]), int(k[4:6]), int(k[6:8])) for k in keys]
        return pl.DataFrame(
            {
                "entity_id": [centroid.entity_id] * len(keys),
                "data": dates,
                "precip_dia_mm": [self._clean_value(precip_map.get(k)) for k in keys],
                "umidade_solo_m3m3": [self._clean_value(soil_map.get(k)) for k in keys],
                "temp_media_c": [self._clean_value(temp_map.get(k)) for k in keys],
            },
            schema={
                "entity_id": pl.String,
                "data": pl.Date,
                "precip_dia_mm": pl.Float64,
                "umidade_solo_m3m3": pl.Float64,
                "temp_media_c": pl.Float64,
            },
        )

    def fetch_daily_hydro(
        self,
        *,
        centroids: tuple[RegionCentroid, ...],
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        if not centroids:
            raise ConfigurationError("Lista de centróides vazia para extração NASA POWER.")

        cache_path = self._cache_path(centroids, start_date, end_date)
        if cache_path.exists():
            return pl.read_parquet(cache_path)

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            frames = list(pool.map(lambda c: self._fetch_one(c, start_date, end_date), centroids))

        combined = pl.concat(frames).sort(["entity_id", "data"])
        tmp_path = cache_path.with_suffix(".parquet.tmp")
        combined.write_parquet(tmp_path)
        os.replace(tmp_path, cache_path)
        return combined
