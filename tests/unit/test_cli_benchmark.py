"""Testes da CLI de benchmark (`previsao-leitos run|benchmark|report`)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from previsao_leitos.cli.main import cli_main

_CHROMIUM = shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which(
    "google-chrome"
)


def _write_panel(path: Path, *, years: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024)) -> Path:
    rng = np.random.default_rng(3)
    rows: list[dict[str, object]] = []
    for i in range(4):
        entity = f"Regiao_{i}"
        cap = 250.0 if i % 2 == 0 else 90.0
        for year in years:
            for week in range(1, 53):
                rain = float(rng.uniform(0.0, 100.0))
                shock = float(min(1.0, rain / 180.0))
                rate = 0.70 + 0.05 * np.sin(2.0 * np.pi * week / 52.0) + 0.08 * shock
                rows.append(
                    {
                        "nome_regiao_saude": entity,
                        "bacia_id": f"Bacia_{i % 2}",
                        "ano_epi": year,
                        "semana_epi": week,
                        "uti_adulto_diarias": cap * 7.0 * rate,
                        "diarias_uti_total_ativas": cap * 7.0 * rate,
                        "clinico_pacientes_dia": cap * 3.0 * rate,
                        "respiratorio_pacientes_dia": cap * 2.0 * rate,
                        "chuva_sem_mm": rain,
                        "chuva_montante_multihop": rain * 1.1,
                        "api_max_sem": rain * 0.7,
                        "remanso_lagunar_idx": shock * 0.4,
                        "exposicao_aterro_ativa": float(i % 2),
                        "prob_enchente_f03_oof": shock,
                        "severidade_hidro_f03_oof": shock * 1.5,
                        "el_nino_intensity": float(rng.normal(0.0, 0.4)),
                        "enso_regime": "neutro" if week % 4 else "el_nino",
                        "frac_urbano": 0.5,
                        "frac_floresta": 0.2,
                        "indice_aterro_varzea": 0.05 * i,
                        "cnes_uti_sus_leitos": cap,
                        "cnes_total_sus_leitos": cap * 3.0,
                    }
                )
    frame = pl.DataFrame(rows)
    frame.write_parquet(path)
    return path


def test_benchmark_command_writes_artifacts(tmp_path: Path) -> None:
    panel = _write_panel(tmp_path / "panel.parquet")
    out = tmp_path / "runs"
    code = cli_main(
        [
            "benchmark",
            "--panel-path",
            str(panel),
            "--output-dir",
            str(out),
            "--run-id",
            "unit",
            "--target",
            "uti_adulto",
            "--models",
            "persistencia,f041_core",
            "--n-boot",
            "10",
            "--no-report",
        ]
    )
    assert code == 0
    run_dir = out / "unit"
    for name in (
        "benchmark.csv",
        "predictions.parquet",
        "comparison.csv",
        "run_manifest.json",
    ):
        assert (run_dir / name).exists(), name
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["holdout_policy"].startswith("anos >= 2025")
    assert manifest["candidates_ok"] >= 1
    assert "panel_sha256" in manifest

    metrics = pl.read_csv(run_dir / "benchmark.csv")
    assert set(metrics["candidate"].to_list()) == {"persistencia", "f041_core"}
    assert (run_dir / "fig_observed_vs_predicted_uti_adulto.png").exists()


def test_benchmark_blocks_holdout_year(tmp_path: Path) -> None:
    panel = _write_panel(tmp_path / "panel.parquet")
    code = cli_main(
        [
            "benchmark",
            "--panel-path",
            str(panel),
            "--output-dir",
            str(tmp_path / "runs"),
            "--target",
            "uti_adulto",
            "--models",
            "persistencia",
            "--eval-year",
            "2025",
            "--no-report",
        ]
    )
    assert code == 3


def test_benchmark_reports_missing_panel(tmp_path: Path) -> None:
    code = cli_main(
        [
            "benchmark",
            "--panel-path",
            str(tmp_path / "ausente.parquet"),
            "--output-dir",
            str(tmp_path / "runs"),
            "--no-report",
        ]
    )
    assert code == 2


@pytest.mark.skipif(_CHROMIUM is None, reason="Chromium ausente para o PDF editorial")
def test_report_command_renders_pdf(tmp_path: Path) -> None:
    panel = _write_panel(tmp_path / "panel.parquet")
    out = tmp_path / "runs"
    assert (
        cli_main(
            [
                "benchmark",
                "--panel-path",
                str(panel),
                "--output-dir",
                str(out),
                "--run-id",
                "withreport",
                "--target",
                "uti_adulto",
                "--models",
                "persistencia,f041_core",
                "--n-boot",
                "10",
                "--no-report",
            ]
        )
        == 0
    )
    code = cli_main(
        [
            "report",
            "--run-id",
            "withreport",
            "--output-dir",
            str(out),
            "--no-mirror-downloads",
        ]
    )
    assert code == 0
    pdf = out / "withreport" / "relatorio_benchmark_leitos.pdf"
    assert pdf.exists() and pdf.stat().st_size > 5000
    html_text = (out / "withreport" / "relatorio_benchmark_leitos.html").read_text(
        encoding="utf-8"
    )
    assert "Not for patient care or live clinical use" in html_text
    assert "```" not in html_text


def test_parallel_jobs_merge_targets(tmp_path: Path) -> None:
    panel = _write_panel(tmp_path / "panel.parquet")
    out = tmp_path / "runs"
    code = cli_main(
        [
            "benchmark",
            "--panel-path",
            str(panel),
            "--output-dir",
            str(out),
            "--run-id",
            "par",
            "--target",
            "uti_adulto,clinico",
            "--models",
            "persistencia",
            "--jobs",
            "2",
            "--n-boot",
            "10",
            "--no-report",
        ]
    )
    assert code == 0
    run_dir = out / "par"
    metrics = pl.read_csv(run_dir / "benchmark.csv")
    assert set(metrics["target"].to_list()) == {"uti_adulto", "clinico"}
    assert (run_dir / "predictions.parquet").exists()
    assert not (run_dir / "_parts").exists()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["parallel"] is True
    assert manifest["jobs"] == 2
