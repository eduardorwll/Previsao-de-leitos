"""Testes TDD offline da Task 1.2.3 (MERGE-CPTEC 0.1°, ANA API v2/Zenodo, Dataset Diário)."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import torch

from previsao_leitos.domain import DataContractError, SourceUnavailableError
from previsao_leitos.hydrology import (
    HydroFiLMMonotonicNet,
    IndexedHydroSequenceDataset,
    audit_dataset_autograd_monotonicity,
    generate_neural_hydro_oof_predictions,
)
from previsao_leitos.ingestion import (
    AnaHidrowebClient,
    MergeCptecClient,
    build_merge_cptec_daily_url,
)


class _FakeHttpFetcher:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self._payloads = payloads
        self.calls: list[str] = []

    def get_bytes(self, url: str, *, timeout: float = 30.0) -> bytes:
        del timeout
        self.calls.append(url)
        if url not in self._payloads:
            raise KeyError(f"URL inesperada no mock: {url}")
        return self._payloads[url]


def test_merge_cptec_client_url_and_sha256_cache_offline(tmp_path: Path) -> None:
    d1 = date(2024, 5, 1)
    d2 = date(2024, 5, 2)
    u1 = build_merge_cptec_daily_url(d1)
    u2 = build_merge_cptec_daily_url(d2)
    assert u1.endswith("/2024/05/MERGE_CPTEC_20240501.grib2")
    assert u2.endswith("/2024/05/MERGE_CPTEC_20240502.grib2")

    fetcher = _FakeHttpFetcher({u1: b"GRIB2-FAKE-DAY1", u2: b"GRIB2-FAKE-DAY2"})
    client = MergeCptecClient(cache_dir=tmp_path, fetcher=fetcher)
    res1 = client.fetch_date_range(d1, d2)
    assert len(res1) == 2
    assert all(not r.cache_hit for r in res1)
    assert res1[0].size_bytes == len(b"GRIB2-FAKE-DAY1")

    # 2ª leitura deve acertar cache SHA-256 sem nova requisição HTTP
    res2 = client.fetch_date_range(d1, d2)
    assert all(r.cache_hit for r in res2)
    assert len(fetcher.calls) == 2

    with pytest.raises(DataContractError, match="end_date"):
        client.fetch_date_range(d2, d1)


def test_ana_hidroweb_client_zenodo_fallback_and_json_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANA_HIDROWEB_USER", raising=False)
    monkeypatch.delenv("ANA_HIDROWEB_PASSWORD", raising=False)

    zen_url = "https://zenodo.org/api/records/16929471"
    zen_payload = json.dumps(
        {
            "id": 16929471,
            "doi": "10.5281/zenodo.16929471",
            "metadata": {"title": "RS 2024 IPH-UFRGS"},
            "files": [
                {
                    "key": "cotas_guaiba.csv",
                    "size": 512,
                    "links": {"self": "https://zenodo.org/api/records/16929471/files/cotas.csv"},
                }
            ],
        }
    ).encode()
    client = AnaHidrowebClient(cache_dir=tmp_path, fetcher=_FakeHttpFetcher({zen_url: zen_payload}))
    assert not client.has_credentials

    res = client.fetch_or_fallback(station_codes=("87450004",), allow_zenodo_fallback=True)
    assert res.source == "zenodo_fallback"
    assert res.zenodo_fallback is not None
    assert res.zenodo_fallback.record_id == 16929471

    with pytest.raises(SourceUnavailableError):
        client.fetch_or_fallback(station_codes=("87450004",), allow_zenodo_fallback=False)

    # Testa parser oficial de série telemétrica JSON da ANA API v2
    ana_json = json.dumps(
        {
            "items": [
                {
                    "Data_Hora_Medicao": "2024-05-03 12:00:00",
                    "Cota_Adotada": 460.0,
                    "Vazao_Adotada": 28000.0,
                },
                {
                    "Data_Hora_Medicao": "2024-05-04 12:00:00",
                    "Cota_Adotada": 530.0,
                    "Vazao_Adotada": 31000.0,
                },
            ]
        }
    ).encode()
    wk_df = client.parse_telemetric_series_json(
        ana_json, bacia_id="guaiba", cota_inundacao_cm=300.0
    )
    assert wk_df.height == 1
    assert float(wk_df["cota_max_cm"][0]) == pytest.approx(530.0)
    assert float(wk_df["excedencia_cota_inundacao_cm"][0]) == pytest.approx(230.0)


def test_indexed_hydro_dataset_daily_mode_gap_guard_and_week_end_mask() -> None:
    # 12 dias contíguos + salto de 2 dias + 5 dias contíguos (não pode cruzar a lacuna!)
    d0 = date(2024, 1, 1)
    dates = [d0 + timedelta(days=i) for i in range(12)] + [
        d0 + timedelta(days=14 + i) for i in range(5)
    ]
    n = len(dates)
    df = pl.DataFrame(
        {
            "entity": ["R10"] * n,
            "data": dates,
            "ano_epi": [d.isocalendar().year for d in dates],
            "semana_epi": [d.isocalendar().week for d in dates],
            "is_week_end": [d.weekday() == 5 for d in dates],  # sábado ISO
            "chuva": [10.0 + i for i in range(n)],
            "ctx": [1.0] * n,
            "oni": [1.2] * n,
            "aterro": [0.3] * n,
            "floresta": [0.2] * n,
            "y_event": [0.0] * n,
            "y_sev": [0.1] * n,
        }
    )

    ds_all = IndexedHydroSequenceDataset(
        df,
        entity_col="entity",
        dyn_cols=("chuva",),
        ctx_cols=("ctx",),
        enso_cols=("oni",),
        mono_pos_cols=("aterro",),
        mono_neg_cols=("floresta",),
        event_col="y_event",
        severity_col="y_sev",
        seq_len=6,
        freq="daily",
    )
    # Primeiros 12 dias contíguos geram 12 - 6 + 1 = 7 janelas; 2o bloco tem apenas 5 dias (< 6)
    assert ds_all.freq == "daily"
    assert len(ds_all) == 7
    assert ds_all[0]["x_dyn"].shape == (6, 1)

    # Com sample_mask_col="is_week_end", seleciona apenas sábados com >= 6 dias contíguos atrás
    ds_sat = IndexedHydroSequenceDataset(
        df,
        entity_col="entity",
        dyn_cols=("chuva",),
        ctx_cols=("ctx",),
        enso_cols=("oni",),
        mono_pos_cols=("aterro",),
        mono_neg_cols=("floresta",),
        event_col="y_event",
        severity_col="y_sev",
        seq_len=6,
        freq="daily",
        sample_mask_col="is_week_end",
    )
    # 2024-01-06 (dia 6, índice 5 -> run_len=6) é o único sábado no bloco de 12 dias
    assert len(ds_sat) == 1

    torch.manual_seed(42)
    net = HydroFiLMMonotonicNet(
        dyn_dim=1, ctx_dim=1, enso_dim=1, mono_pos_dim=1, mono_neg_dim=1, hidden_dim=8, num_heads=2
    )
    assert audit_dataset_autograd_monotonicity(net, ds_all, batch_size=4) == 0.0


def test_generate_neural_hydro_oof_daily_mode_aligns_to_weekly_keys() -> None:
    rng = np.random.default_rng(11)
    rows: list[dict[str, object]] = []
    start = date(2021, 1, 1)
    # 4 anos (~1461 dias) para 2 entidades -> folds diários com janelas L=14 dias
    for ent in ("R10", "R21"):
        for offset in range(1460):
            cur = start + timedelta(days=offset)
            iso = cur.isocalendar()
            rain = float(rng.gamma(1.5, 6.0))
            oni = 1.4 if iso.year >= 2023 else -0.4
            ev = 1.0 if rain > 22.0 else 0.0
            sev = max(0.0, (rain - 10.0) / 15.0)
            rows.append(
                {
                    "nome_regiao_saude": ent,
                    "data": cur,
                    "ano_epi": int(iso.year),
                    "semana_epi": int(iso.week),
                    "is_week_end": cur.weekday() == 5,
                    "precip_dia_mm": rain,
                    "chuva_montante_multihop": rain,
                    "api_dia": rain * 0.8,
                    "ctx_rain_mean": 40.0 if ent == "R10" else 30.0,
                    "oni_c_lag4": oni,
                    "frac_urbano": 0.6 if ent == "R10" else 0.2,
                    "frac_floresta": 0.1 if ent == "R10" else 0.4,
                    "y_ev": ev,
                    "y_sev": sev,
                }
            )
    daily_df = pl.DataFrame(rows).filter((pl.col("ano_epi") >= 2021) & (pl.col("ano_epi") <= 2024))
    oof_wk, rep = generate_neural_hydro_oof_predictions(
        daily_df,
        entity_col="nome_regiao_saude",
        dyn_cols=("precip_dia_mm", "api_dia"),
        ctx_cols=("ctx_rain_mean",),
        enso_cols=("oni_c_lag4",),
        mono_pos_cols=("frac_urbano",),
        mono_neg_cols=("frac_floresta",),
        event_col="y_ev",
        severity_col="y_sev",
        train_end_year=2022,
        cal_year=2023,
        seq_len=14,
        freq="daily",
        date_col="data",
        sample_mask_col="is_week_end",
        max_epochs=1,
        batch_size=64,
        hidden_dim=8,
        num_heads=2,
        seeds=(42,),
        requested_device="cpu",
    )
    assert rep.autograd_violation_rate == 0.0
    assert "prob_enchente_f03_oof" in oof_wk.columns
    # Saída alinhada à granularidade semanal (~2 entidades x ~208 semanas)
    assert oof_wk.height == daily_df.filter(pl.col("is_week_end")).height


class _FakeOAuthResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _FakeOAuthResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def test_ana_hidroweb_uses_documented_routes_and_query_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trava o contrato da API v2: rota `OAUth/v1`, headers e query do manual oficial."""
    monkeypatch.setenv("ANA_HIDROWEB_USER", "identificador-teste")
    monkeypatch.setenv("ANA_HIDROWEB_PASSWORD", "senha-teste")

    captured: list[tuple[str, dict[str, str]]] = []

    def _fake_urlopen(req: Any, timeout: float = 30.0) -> _FakeOAuthResponse:
        del timeout
        url = str(req.full_url)
        headers = {str(k).lower(): str(v) for k, v in req.header_items()}
        captured.append((url, headers))
        if "OAUth" in url:
            return _FakeOAuthResponse(
                json.dumps({"items": {"tokenautenticacao": "token-abc"}}).encode()
            )
        return _FakeOAuthResponse(
            json.dumps(
                {
                    "items": [
                        {
                            "Data_Hora_Medicao": "2024-05-03 12:00:00.0",
                            "Cota_Adotada": "460.00",
                            "Vazao_Adotada": "28000.00",
                        }
                    ]
                }
            ).encode()
        )

    monkeypatch.setattr(
        "previsao_leitos.ingestion.wave2b_clients.urllib.request.urlopen", _fake_urlopen
    )

    client = AnaHidrowebClient(cache_dir=tmp_path)
    res = client.fetch_or_fallback(station_codes=("87450004",), allow_zenodo_fallback=False)
    assert res.source == "ana_hidroweb_v2"

    auth_url, auth_headers = captured[0]
    assert auth_url == (
        "https://www.ana.gov.br/hidrowebservice/EstacoesTelemetricas/OAUth/v1"
    )
    assert auth_headers["identificador"] == "identificador-teste"
    assert auth_headers["senha"] == "senha-teste"

    series_url, series_headers = captured[1]
    assert "HidroinfoanaSerieTelemetricaAdotada/v1" in series_url
    assert "CodigoDaEstacao=87450004" in series_url
    assert "TipoFiltroData=DATA_LEITURA" in series_url
    assert "RangeIntervaloDeBusca=DIAS_30" in series_url
    assert series_headers["authorization"] == "Bearer token-abc"
