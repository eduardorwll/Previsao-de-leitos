"""Testes unitários dos adaptadores hidrológicos (Fixture offline + Open-Meteo ERA5-Land)."""

from __future__ import annotations

import io
import json
from datetime import date
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import polars as pl
import pytest

from previsao_leitos.domain import ConfigurationError, SourceUnavailableError
from previsao_leitos.geo.hydro_sources import (
    HydroFixtureAdapter,
    NasaPowerHydroAdapter,
    OpenMeteoEra5Adapter,
    RegionCentroid,
)


def test_hydro_fixture_adapter_filters_date_window_and_raises_when_missing(tmp_path: Path) -> None:
    missing_adapter = HydroFixtureAdapter(tmp_path / "nao_existe.parquet")
    with pytest.raises(SourceUnavailableError):
        missing_adapter.fetch_daily_hydro(
            centroids=(RegionCentroid("R10", -30.03, -51.23),),
            start_date=date(2024, 5, 1),
            end_date=date(2024, 5, 3),
        )

    fixture_path = tmp_path / "hydro_fixture.parquet"
    pl.DataFrame(
        {
            "entity_id": ["R10", "R10", "R10"],
            "data": [date(2024, 4, 30), date(2024, 5, 1), date(2024, 5, 5)],
            "precip_dia_mm": [10.0, 120.0, 5.0],
        }
    ).write_parquet(fixture_path)

    adapter = HydroFixtureAdapter(fixture_path)
    out = adapter.fetch_daily_hydro(
        centroids=(RegionCentroid("R10", -30.03, -51.23),),
        start_date=date(2024, 5, 1),
        end_date=date(2024, 5, 3),
    )
    assert out.height == 1
    assert out["precip_dia_mm"][0] == pytest.approx(120.0)


def test_open_meteo_era5_adapter_batches_retries_429_and_caches_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    class _FakeResponse:
        def __init__(self, payload: Any) -> None:
            self._buf = io.BytesIO(json.dumps(payload).encode("utf-8"))

        def read(self) -> bytes:
            return self._buf.read()

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def _fake_urlopen(url: str, timeout: float = 30.0) -> _FakeResponse:
        nonlocal calls
        calls += 1
        assert "archive-api.open-meteo.com" in url
        assert timeout == pytest.approx(15.0)
        if calls == 1:
            raise HTTPError(url, 429, "Too Many Requests", Message(), None)
        return _FakeResponse(
            [
                {
                    "daily": {
                        "time": ["2024-05-01", "2024-05-02"],
                        "precipitation_sum": [110.5, 85.2],
                        "temperature_2m_mean": [18.1, 17.4],
                        "soil_moisture_0_to_7cm_mean": [0.44, 0.48],
                    }
                },
                {
                    "daily": {
                        "time": ["2024-05-01", "2024-05-02"],
                        "precipitation_sum": [12.0, 18.0],
                        "temperature_2m_mean": [16.0, 15.8],
                        "soil_moisture_0_to_7cm_mean": [0.35, 0.37],
                    }
                },
            ]
        )

    monkeypatch.setattr("previsao_leitos.geo.hydro_sources.urlopen", _fake_urlopen)

    adapter = OpenMeteoEra5Adapter(
        cache_dir=tmp_path,
        timeout_seconds=15.0,
        pause_seconds=0.0,
        batch_size=10,
        max_retries=2,
        backoff_seconds=0.0,
    )
    centroids = (
        RegionCentroid("R10", -30.0346, -51.2177),
        RegionCentroid("R21", -31.7654, -52.3376),
    )

    first = adapter.fetch_daily_hydro(
        centroids=centroids, start_date=date(2024, 5, 1), end_date=date(2024, 5, 2)
    )
    second = adapter.fetch_daily_hydro(
        centroids=centroids, start_date=date(2024, 5, 1), end_date=date(2024, 5, 2)
    )

    assert calls == 2  # 1ª tentativa 429 + 1 retry OK em lote; 0 chamadas na 2ª (hit de cache)
    assert first.height == 4
    assert first.equals(second)
    assert first.schema["precip_dia_mm"] == pl.Float64
    assert first.schema["umidade_solo_m3m3"] == pl.Float64


def test_open_meteo_era5_adapter_validates_empty_centroids_and_network_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ConfigurationError):
        OpenMeteoEra5Adapter(cache_dir=tmp_path, batch_size=0)

    adapter = OpenMeteoEra5Adapter(cache_dir=tmp_path, pause_seconds=0.0, backoff_seconds=0.0)
    with pytest.raises(ConfigurationError):
        adapter.fetch_daily_hydro(
            centroids=(), start_date=date(2024, 5, 1), end_date=date(2024, 5, 2)
        )

    def _boom(url: str, timeout: float = 30.0) -> object:
        raise URLError("offline")

    monkeypatch.setattr("previsao_leitos.geo.hydro_sources.urlopen", _boom)
    with pytest.raises(SourceUnavailableError, match="R10"):
        adapter.fetch_daily_hydro(
            centroids=(RegionCentroid("R10", -30.03, -51.21),),
            start_date=date(2024, 5, 1),
            end_date=date(2024, 5, 2),
        )


def test_nasa_power_adapter_converts_missing_sentinel_to_null_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    class _FakeResp:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._buf = io.BytesIO(json.dumps(payload).encode("utf-8"))

        def read(self) -> bytes:
            return self._buf.read()

        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def _fake_urlopen(url: str, timeout: float = 45.0) -> _FakeResp:
        nonlocal calls
        calls += 1
        assert "power.larc.nasa.gov" in url
        return _FakeResp(
            {
                "properties": {
                    "parameter": {
                        "PRECTOTCORR": {"20240501": 74.8, "20240502": -999.0},
                        "GWETTOP": {"20240501": 0.89, "20240502": -999.0},
                        "T2M": {"20240501": 22.1, "20240502": 19.5},
                    }
                }
            }
        )

    monkeypatch.setattr("previsao_leitos.geo.hydro_sources.urlopen", _fake_urlopen)
    adapter = NasaPowerHydroAdapter(cache_dir=tmp_path, max_workers=2)
    centroids = (RegionCentroid("43010", -30.03, -51.21),)

    first = adapter.fetch_daily_hydro(
        centroids=centroids, start_date=date(2024, 5, 1), end_date=date(2024, 5, 2)
    )
    second = adapter.fetch_daily_hydro(
        centroids=centroids, start_date=date(2024, 5, 1), end_date=date(2024, 5, 2)
    )

    assert calls == 1
    assert first.equals(second)
    assert first["precip_dia_mm"][0] == pytest.approx(74.8)
    # Sentinela NASA -999.0 deve virar null, jamais -999 ou 0.0!
    assert first["precip_dia_mm"][1] is None
    assert first["umidade_solo_m3m3"][1] is None
