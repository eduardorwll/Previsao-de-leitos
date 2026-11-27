"""Testes TDD offline da Task 1.2.2 (Onda 2A HTTP mocks, GPU/Dataset fixes, OOF neural)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from previsao_leitos.domain import DataContractError
from previsao_leitos.hydrology import (
    HydroFiLMMonotonicNet,
    IndexedHydroSequenceDataset,
    audit_dataset_autograd_monotonicity,
    generate_neural_hydro_oof_predictions,
    resolve_accelerator,
)
from previsao_leitos.ingestion import (
    discover_gerint_poa_csv_urls,
    fetch_noaa_enso_monthly,
    fetch_noaa_roni_monthly,
    fetch_zenodo_iph_rs_metadata,
    ingest_gerint_poa_csv_lgpd,
    ingest_opendatasus_sinan_csv_lgpd,
    ingest_opendatasus_sivep_csv_lgpd,
    parse_noaa_mei_v2_table,
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


def test_wave2a_noaa_zenodo_gerint_and_opendatasus_clients_offline_with_sha256_cache(
    tmp_path: Path,
) -> None:
    oni_txt = (
        b"SEAS YR TOTAL ANOM\n"
        b"DJF 2023 26.0 -0.7\n"
        b"JFM 2023 26.5 -0.4\n"
        b"AMJ 2024 27.5 0.8\n"
        b"JJA 2025 27.0 0.1\n"
    )
    mei_txt = (
        b"1979 2025\n"
        b"2023 -0.8 -0.6 -0.3 0.1 0.4 0.7 0.9 1.0 1.1 1.2 1.1 1.0\n"
        b"2024  0.9  0.8  0.7 0.5 0.3 0.1 -0.1 -0.2 -0.3 -999.0 -999.0 -999.0\n"
    )
    zenodo_json = json.dumps(
        {
            "id": 16929471,
            "doi": "10.5281/zenodo.16929471",
            "metadata": {"title": "RS 2024 Flood IPH-UFRGS"},
            "files": [
                {
                    "key": "cotas_rs_2024.csv",
                    "size": 12345,
                    "links": {"self": "https://zenodo.org/api/records/16929471/files/cotas.csv"},
                }
            ],
        }
    ).encode()
    ckan_json = json.dumps(
        {
            "success": True,
            "result": {
                "results": [
                    {
                        "resources": [
                            {
                                "name": "GERINT 2024",
                                "format": "CSV",
                                "url": "https://dadosabertos.poa.br/gerint_2024.csv",
                            }
                        ]
                    }
                ]
            },
        }
    ).encode()
    gerint_csv = (
        "data_solicitacao;tipo_leito;tempo_espera_horas;internado\n"
        + "\n".join("2024-05-06;uti_adulto;12.0;false" for _ in range(6))
    ).encode()
    sivep_csv = (
        "data_primeiros_sintomas;nome_regiao_saude;uti;viral_confirmado\n"
        + "\n".join("2024-05-06;Região 10;true;true" for _ in range(6))
    ).encode()
    sinan_csv = (
        "data_primeiros_sintomas;nome_regiao_saude;confirmado;hospitalizado\n"
        + "\n".join("2024-05-15;Região 10;true;true" for _ in range(6))
    ).encode()

    fetcher = _FakeHttpFetcher(
        {
            "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt": oni_txt,
            "https://psl.noaa.gov/enso/mei/data/meiv2.data": mei_txt,
            "https://zenodo.org/api/records/16929471": zenodo_json,
            "https://dadosabertos.poa.br/api/3/action/package_search?q=gerint": ckan_json,
            "https://dadosabertos.poa.br/gerint_2024.csv": gerint_csv,
            "https://opendatasus.saude.gov.br/sivep.csv": sivep_csv,
            "https://opendatasus.saude.gov.br/sinan.csv": sinan_csv,
        }
    )

    # 1. NOAA ONI + MEI.v2 com corte max_year=2024 (nunca lê 2025) e cache hit na 2ª chamada
    mei_parsed = parse_noaa_mei_v2_table(mei_txt.decode())
    assert mei_parsed.height == 21
    noaa_df, m1 = fetch_noaa_enso_monthly(cache_dir=tmp_path, fetcher=fetcher, max_year=2024)
    assert int(noaa_df["data_ref"].dt.year().max()) == 2024
    assert all(not m.cache_hit for m in m1)
    _, m2 = fetch_noaa_enso_monthly(cache_dir=tmp_path, fetcher=fetcher, max_year=2024)
    assert all(m.cache_hit for m in m2)

    # 2. Zenodo IPH-UFRGS
    zen = fetch_zenodo_iph_rs_metadata(cache_dir=tmp_path, fetcher=fetcher)
    assert zen.record_id == 16929471
    assert len(zen.files) == 1

    # 3. GERINT-POA + SIVEP + SINAN agregados com LGPD (n >= 5 preservado)
    urls, _ = discover_gerint_poa_csv_urls(cache_dir=tmp_path, fetcher=fetcher)
    assert urls == [("GERINT 2024", "https://dadosabertos.poa.br/gerint_2024.csv")]
    g_agg, _ = ingest_gerint_poa_csv_lgpd(urls[0][1], cache_dir=tmp_path, fetcher=fetcher)
    assert int(g_agg["solicitacoes_totais"].sum()) == 6

    sv_agg, _ = ingest_opendatasus_sivep_csv_lgpd(
        "https://opendatasus.saude.gov.br/sivep.csv", cache_dir=tmp_path, fetcher=fetcher
    )
    assert int(sv_agg["srag_total"].sum()) == 6

    sn_agg, _ = ingest_opendatasus_sinan_csv_lgpd(
        "https://opendatasus.saude.gov.br/sinan.csv", cache_dir=tmp_path, fetcher=fetcher
    )
    assert int(sn_agg["casos_notificados"].sum()) == 6


def test_fetch_noaa_roni_monthly_parses_primary_cpc_source(tmp_path: Path) -> None:
    roni_txt = (
        b"SEAS   YR  ANOM\n"
        b"DJF  2023 -0.86\n"
        b"JFM  2023 -0.68\n"
        b"OND  2023  1.42\n"
        b"AMJ  2024 -0.28\n"
        b"JJA  2025 -0.41\n"
    )
    fetcher = _FakeHttpFetcher(
        {"https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt": roni_txt}
    )

    roni_df, meta = fetch_noaa_roni_monthly(cache_dir=tmp_path, fetcher=fetcher, max_year=2024)
    assert int(roni_df["data_ref"].dt.year().max()) == 2024
    assert roni_df.height == 4
    assert set(roni_df.columns) == {"data_ref", "roni_c"}
    assert not meta.cache_hit
    # 2ª chamada usa o cache atômico SHA-256.
    _, meta2 = fetch_noaa_roni_monthly(cache_dir=tmp_path, fetcher=fetcher, max_year=2024)
    assert meta2.cache_hit


def test_resolve_accelerator_cpu_and_dataset_enso_cols_and_full_split_autograd() -> None:
    spec_cpu = resolve_accelerator(prefer_gpu=False)
    assert spec_cpu.accelerator == "cpu"
    assert spec_cpu.backend == "cpu"
    assert spec_cpu.deterministic_mode == "strict"

    # Dataset com enso_cols + preload CPU + trava ADR-0001 contra sobreposição
    df = pl.DataFrame(
        {
            "entity": ["A"] * 10,
            "ano_epi": [2022] * 10,
            "semana_epi": list(range(1, 11)),
            "chuva": [10.0 + i for i in range(10)],
            "ctx": [1.0] * 10,
            "oni": [0.5] * 10,
            "aterro": [0.2] * 10,
            "floresta": [0.3] * 10,
            "y_event": [0.0, 0.0, 1.0, 1.0, 0.0] * 2,
            "y_sev": [0.1, 0.2, 0.8, 0.9, 0.2] * 2,
        }
    )
    with pytest.raises(DataContractError, match="ADR-0001"):
        IndexedHydroSequenceDataset(
            df,
            entity_col="entity",
            dyn_cols=("chuva",),
            ctx_cols=("ctx",),
            enso_cols=("aterro",),
            mono_pos_cols=("aterro",),
            mono_neg_cols=("floresta",),
            event_col="y_event",
            severity_col="y_sev",
            seq_len=4,
        )

    ds = IndexedHydroSequenceDataset(
        df,
        entity_col="entity",
        dyn_cols=("chuva",),
        ctx_cols=("ctx",),
        enso_cols=("oni",),
        mono_pos_cols=("aterro",),
        mono_neg_cols=("floresta",),
        event_col="y_event",
        severity_col="y_sev",
        seq_len=4,
        preload_device="cpu",
    )
    assert len(ds) == 7
    assert "x_enso" in ds[0]
    assert ds[0]["x_enso"].shape == (1,)

    torch.manual_seed(7)
    net = HydroFiLMMonotonicNet(
        dyn_dim=1, ctx_dim=1, enso_dim=1, mono_pos_dim=1, mono_neg_dim=1, hidden_dim=8, num_heads=2
    )
    assert audit_dataset_autograd_monotonicity(net, ds, batch_size=3) == 0.0


def test_generate_neural_hydro_oof_predictions_causal_and_zero_autograd_violations() -> None:
    rng = np.random.default_rng(42)
    rows = []
    for ent in ("R10", "R21"):
        for yr in (2020, 2021, 2022, 2023, 2024):
            for wk in range(1, 15):
                rain = float(rng.gamma(2.0, 25.0))
                oni = 1.4 if yr >= 2023 else -0.4
                ev = 1.0 if rain + 15.0 * max(oni, 0.0) > 85.0 else 0.0
                sev = max(0.0, (rain - 30.0) / 40.0 + 0.15 * float(rng.normal()))
                rows.append(
                    {
                        "nome_regiao_saude": ent,
                        "ano_epi": yr,
                        "semana_epi": wk,
                        "chuva_sem_mm": rain,
                        "chuva_montante_multihop": rain * 1.1,
                        "api_max_sem": rain * 0.6,
                        "ctx_rain_mean": 45.0 if ent == "R10" else 35.0,
                        "oni_c_lag4": oni,
                        "frac_urbano": 0.6 if ent == "R10" else 0.25,
                        "frac_floresta": 0.1 if ent == "R10" else 0.4,
                        "y_ev": ev,
                        "y_sev": sev,
                    }
                )
    df = pl.DataFrame(rows)
    out, rep = generate_neural_hydro_oof_predictions(
        df,
        entity_col="nome_regiao_saude",
        dyn_cols=("chuva_sem_mm", "api_max_sem"),
        ctx_cols=("ctx_rain_mean",),
        enso_cols=("oni_c_lag4",),
        mono_pos_cols=("frac_urbano",),
        mono_neg_cols=("frac_floresta",),
        event_col="y_ev",
        severity_col="y_sev",
        train_end_year=2022,
        cal_year=2023,
        seq_len=4,
        max_epochs=1,
        batch_size=32,
        hidden_dim=8,
        num_heads=2,
        seeds=(42,),
        requested_device="cpu",
    )
    assert rep.backend == "cpu"
    assert rep.autograd_violation_rate == 0.0
    assert rep.n_folds >= 2
    assert abs(rep.corr_rain_severity) < 0.99
    assert "prob_enchente_f03_oof" in out.columns
    assert "severidade_hidro_f03_oof" in out.columns
    probs = out["prob_enchente_f03_oof"].to_numpy()
    assert np.all((probs >= 0.0) & (probs <= 1.0))
