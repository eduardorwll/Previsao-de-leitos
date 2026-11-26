"""Clientes HTTP ao vivo da Onda 2B: MERGE-CPTEC 0.1° e ANA HidroWeb API v2 (RHN).

- `MergeCptecClient`: baixa e cacheia por SHA-256 arquivos diários GRIB2 `MERGE-CPTEC 0.1°`
  do CPTEC/INPE (`https://ftp.cptec.inpe.br/modelos/tempo/MERGE/GPM/DAILY/YYYY/MM/`).
- `AnaHidrowebClient`: autentica via OAuth/token na API v2 da Rede Hidrometeorológica Nacional
  (credenciais lidas apenas de variáveis de ambiente `ANA_HIDROWEB_USER` / `ANA_HIDROWEB_PASSWORD`),
  com fallback transparente para o catálogo aberto Zenodo IPH-UFRGS (`10.5281/zenodo.16929471`)
  enquanto as credenciais não estiverem configuradas.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from previsao_leitos.domain import DataContractError, SourceUnavailableError
from previsao_leitos.geo.fine_hydro_sources import aggregate_ana_hidroweb_weekly
from previsao_leitos.ingestion.wave2a_clients import (
    CachedPayload,
    HttpFetcher,
    UrllibHttpFetcher,
    ZenodoRecordSummary,
    fetch_cached_bytes,
    fetch_zenodo_iph_rs_metadata,
)

MERGE_CPTEC_DAILY_BASE_URL = "https://ftp.cptec.inpe.br/modelos/tempo/MERGE/GPM/DAILY"
ANA_HIDROWEB_AUTH_URL = (
    "https://www.ana.gov.br/hidrowebservice/EstacoesTelemetricas/OAUth/v1"
)
ANA_HIDROWEB_SERIES_URL = (
    "https://www.ana.gov.br/hidrowebservice/EstacoesTelemetricas/HidroinfoanaSerieTelemetricaAdotada/v1"
)


def build_merge_cptec_daily_url(target_date: date) -> str:
    """Constrói a URL HTTPS canônica do arquivo diário GRIB2 do MERGE-CPTEC 0.1°."""
    return (
        f"{MERGE_CPTEC_DAILY_BASE_URL}/"
        f"{target_date.year:04d}/{target_date.month:02d}/"
        f"MERGE_CPTEC_{target_date.strftime('%Y%m%d')}.grib2"
    )


class MergeCptecClient:
    """Cliente streaming com cache atômico SHA-256 para GRIB2 diários MERGE-CPTEC 0.1°."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        fetcher: HttpFetcher | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._cache_dir = cache_dir
        self._fetcher = fetcher or UrllibHttpFetcher()
        self._timeout = timeout

    def fetch_daily_grib2(self, target_date: date, *, force_refresh: bool = False) -> CachedPayload:
        """Baixa (ou lê do cache SHA-256) o GRIB2 diário MERGE-CPTEC para `target_date`."""
        url = build_merge_cptec_daily_url(target_date)
        filename = f"MERGE_CPTEC_{target_date.strftime('%Y%m%d')}.grib2"
        _, payload = fetch_cached_bytes(
            url,
            cache_dir=self._cache_dir / "merge_cptec",
            filename=filename,
            fetcher=self._fetcher,
            timeout=self._timeout,
            force_refresh=force_refresh,
        )
        return payload

    def fetch_date_range(
        self, start_date: date, end_date: date
    ) -> tuple[CachedPayload, ...]:
        """Baixa sequencialmente um intervalo de datas diárias [start_date, end_date]."""
        if end_date < start_date:
            raise DataContractError("end_date deve ser >= start_date em MergeCptecClient.")
        items: list[CachedPayload] = []
        cur = start_date
        while cur <= end_date:
            items.append(self.fetch_daily_grib2(cur))
            cur += timedelta(days=1)
        return tuple(items)


@dataclass(frozen=True)
class AnaHidrowebCredentials:
    """Credenciais OAuth da ANA HidroWeb API v2 lidas exclusivamente do ambiente."""

    identificador: str
    senha: str

    @classmethod
    def from_env(cls) -> AnaHidrowebCredentials | None:
        """Carrega `ANA_HIDROWEB_USER` e `ANA_HIDROWEB_PASSWORD` do ambiente (ou `None`)."""
        user = os.environ.get("ANA_HIDROWEB_USER", "").strip()
        pwd = os.environ.get("ANA_HIDROWEB_PASSWORD", "").strip()
        if not user or not pwd:
            return None
        return cls(identificador=user, senha=pwd)


@dataclass(frozen=True)
class AnaStationFetchResult:
    """Resultado da consulta ANA API v2 ou fallback aberto Zenodo IPH-UFRGS."""

    source: str
    weekly_df: pl.DataFrame | None
    zenodo_fallback: ZenodoRecordSummary | None
    station_codes: tuple[str, ...]


class AnaHidrowebClient:
    """Cliente OAuth da API v2 ANA HidroWeb (RHN) com fallback automático para Zenodo 16929471."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        credentials: AnaHidrowebCredentials | None = None,
        fetcher: HttpFetcher | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._cache_dir = cache_dir
        self._credentials = (
            credentials if credentials is not None else AnaHidrowebCredentials.from_env()
        )
        self._fetcher = fetcher or UrllibHttpFetcher()
        self._timeout = timeout

    @property
    def has_credentials(self) -> bool:
        """Indica se credenciais OAuth ANA foram configuradas."""
        return self._credentials is not None

    def authenticate(self) -> str:
        """Obtém token Bearer OAuth na ANA HidroWeb API v2."""
        if self._credentials is None:
            raise SourceUnavailableError(
                "Credenciais ANA_HIDROWEB_USER/ANA_HIDROWEB_PASSWORD não configuradas no ambiente."
            )
        req = urllib.request.Request(
            ANA_HIDROWEB_AUTH_URL,
            headers={
                "Identificador": self._credentials.identificador,
                "Senha": self._credentials.senha,
                "Accept": "application/json",
                "User-Agent": "PrevisaoDeLeitos-Research/1.2.3",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise SourceUnavailableError(
                f"Falha na autenticação OAuth ANA HidroWeb: {exc}"
            ) from exc

        token = str(payload.get("items", {}).get("tokenautenticacao", "")).strip()
        if not token:
            raise SourceUnavailableError("Resposta OAuth ANA sem campo tokenautenticacao.")
        return token

    def parse_telemetric_series_json(
        self,
        raw_json: bytes,
        *,
        bacia_id: str,
        cota_inundacao_cm: float,
    ) -> pl.DataFrame:
        """Converte payload JSON da ANA API v2 em série semanal agregada."""
        payload = json.loads(raw_json.decode("utf-8"))
        items = payload.get("items", [])
        if not items:
            raise DataContractError("Payload ANA HidroWeb sem registros em 'items'.")
        rows: list[dict[str, object]] = []
        for item in items:
            raw_dt = str(item.get("Data_Hora_Medicao", "")).strip()
            if len(raw_dt) < 10:
                continue
            cota_val = item.get("Cota_Adotada")
            vazao_val = item.get("Vazao_Adotada")
            if cota_val is None and vazao_val is None:
                continue
            rows.append(
                {
                    "bacia_id": bacia_id,
                    "data_medicao": date.fromisoformat(raw_dt[:10]),
                    "cota_cm": float(cota_val) if cota_val is not None else 0.0,
                    "vazao_m3s": float(vazao_val) if vazao_val is not None else 0.0,
                    "cota_inundacao_cm": float(cota_inundacao_cm),
                }
            )
        if not rows:
            raise DataContractError("Nenhuma medição válida de cota/vazão no payload ANA.")
        return aggregate_ana_hidroweb_weekly(pl.DataFrame(rows))

    def fetch_or_fallback(
        self,
        *,
        station_codes: Sequence[str],
        bacia_id: str = "guaiba",
        cota_inundacao_cm: float = 300.0,
        allow_zenodo_fallback: bool = True,
    ) -> AnaStationFetchResult:
        """Consulta ANA API v2 se autenticado; caso contrário aciona fallback Zenodo IPH-UFRGS."""
        codes_tuple = tuple(str(c) for c in station_codes)
        if not self.has_credentials:
            if not allow_zenodo_fallback:
                raise SourceUnavailableError("ANA API v2 sem credenciais e fallback desativado.")
            zen = fetch_zenodo_iph_rs_metadata(
                cache_dir=self._cache_dir, fetcher=self._fetcher
            )
            return AnaStationFetchResult(
                source="zenodo_fallback",
                weekly_df=None,
                zenodo_fallback=zen,
                station_codes=codes_tuple,
            )

        token = self.authenticate()
        frames: list[pl.DataFrame] = []
        for code in codes_tuple:
            url = (
                f"{ANA_HIDROWEB_SERIES_URL}?"
                f"CodigoDaEstacao={code}&TipoFiltroData=DATA_LEITURA"
                "&RangeIntervaloDeBusca=DIAS_30"
            )
            req = urllib.request.Request(  # noqa: S310
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                    "User-Agent": "PrevisaoDeLeitos-Research/1.2.3",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
                    raw = resp.read()
                frames.append(
                    self.parse_telemetric_series_json(
                        raw, bacia_id=bacia_id, cota_inundacao_cm=cota_inundacao_cm
                    )
                )
            except (SourceUnavailableError, DataContractError, OSError):
                continue

        if frames:
            return AnaStationFetchResult(
                source="ana_hidroweb_v2",
                weekly_df=pl.concat(frames).sort(["bacia_id", "ano_epi", "semana_epi"]),
                zenodo_fallback=None,
                station_codes=codes_tuple,
            )

        if allow_zenodo_fallback:
            zen = fetch_zenodo_iph_rs_metadata(
                cache_dir=self._cache_dir, fetcher=self._fetcher
            )
            return AnaStationFetchResult(
                source="zenodo_fallback",
                weekly_df=None,
                zenodo_fallback=zen,
                station_codes=codes_tuple,
            )
        raise SourceUnavailableError("Nenhuma estação ANA retornou séries válidas.")
