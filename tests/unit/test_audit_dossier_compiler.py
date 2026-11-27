"""Testes unitários da Task 1.3: correções (0>=0, ddof=1, eval_year) e dossier_compiler."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from previsao_leitos.audit.data_hygiene import (
    check_g2_join_loss,
    check_g8_degenerate_surge_labels,
    compile_hygiene_scorecard,
)
from previsao_leitos.audit.dossier_compiler import (
    DossierArtifacts,
    compile_execution_dossier,
    compute_surge_flags,
)
from previsao_leitos.audit.hypothesis_ledger import (
    ChampionChallengerRatchet,
    ModelScorecard,
)
from previsao_leitos.audit.plots import (
    plot_c1_counterfactual_trajectory,
    plot_c6_hygiene_scorecard,
    plot_c7_ablation_delta,
)


def test_compute_surge_flags_blocks_zero_ge_zero_and_passes_g8() -> None:
    y_true = np.array([0.0, 0.0, 1.0, 10.0, 12.0])
    y_roll12 = np.array([0.0, 0.0, 2.0, 10.0, 10.0])

    flags = compute_surge_flags(y_true, y_roll12, multiplier=1.10)
    assert flags.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]

    g8 = check_g8_degenerate_surge_labels(y_true, y_roll12, flags)
    assert g8.status == "PASS"
    assert g8.observed_value == 0.0


def test_plot_c1_accepts_dynamic_eval_year(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "nome_regiao_saude": ["Região 10"] * 4,
            "target_semana_epi": [18, 19, 20, 21],
            "target_y": [100.0, 140.0, 160.0, 130.0],
            "y_cf_no_flood": [95.0, 96.0, 97.0, 98.0],
            "y_cf_p05": [85.0, 86.0, 87.0, 88.0],
            "y_cf_p95": [105.0, 106.0, 107.0, 108.0],
            "pred_p10": [90.0, 110.0, 120.0, 105.0],
            "pred_p50": [102.0, 135.0, 155.0, 125.0],
            "pred_p90": [115.0, 160.0, 185.0, 145.0],
        }
    )
    out = plot_c1_counterfactual_trajectory(
        df,
        entity_col="nome_regiao_saude",
        target_label="UTI Adulto",
        output_path=tmp_path / "fig_c1.png",
        eval_year=2025,
    )
    assert out.exists()
    assert out.stat().st_size > 1_000


def test_plot_c6_and_c7_generate_valid_pngs(tmp_path: Path) -> None:
    scorecard = compile_hygiene_scorecard(
        [
            check_g2_join_loss(1000, 1000),
            check_g8_degenerate_surge_labels([0.0, 5.0], [0.0, 3.0], [False, True]),
        ]
    )
    c6 = plot_c6_hygiene_scorecard(scorecard, output_path=tmp_path / "fig_c6.png")
    assert c6.exists()
    assert c6.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    champ = ModelScorecard(
        run_id="v1",
        model_name="GBDT",
        mase=0.82,
        pinball_p95=14.0,
        coverage_p90=0.85,
        peak_timing_error_weeks=2.0,
        interior_specificity=0.70,
        ece=0.06,
    )
    chall = ModelScorecard(
        run_id="v1_1",
        model_name="GBDT_v1_1",
        mase=0.72,
        pinball_p95=11.5,
        coverage_p90=0.90,
        peak_timing_error_weeks=0.5,
        interior_specificity=0.78,
        ece=0.04,
    )
    ev = ChampionChallengerRatchet().evaluate(champion=champ, challenger=chall)
    c7 = plot_c7_ablation_delta(ev, output_path=tmp_path / "fig_c7.png")
    assert c7.exists()
    assert c7.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_compile_execution_dossier_creates_md_pdf_and_mirrors_champion(tmp_path: Path) -> None:
    runs_root = tmp_path / "reports" / "runs"
    downloads_mirror = tmp_path / "Downloads"

    scorecard = compile_hygiene_scorecard(
        [
            check_g2_join_loss(1000, 1000),
            check_g8_degenerate_surge_labels([0.0, 8.0], [0.0, 5.0], [False, True]),
        ]
    )
    champ = ModelScorecard(
        run_id="v1_base",
        model_name="QuantileGBDT",
        mase=0.82,
        pinball_p95=14.0,
        coverage_p90=0.85,
        peak_timing_error_weeks=2.0,
        interior_specificity=0.70,
        ece=0.06,
    )
    chall = ModelScorecard(
        run_id="rs_2024_smoke",
        model_name="QuantileGBDT + Conformal",
        mase=0.74,
        pinball_p95=12.0,
        coverage_p90=0.90,
        peak_timing_error_weeks=0.5,
        interior_specificity=0.78,
        ece=0.04,
    )
    ratchet_eval = ChampionChallengerRatchet().evaluate(champion=champ, challenger=chall)
    bench_df = pl.DataFrame(
        {
            "modelo": ["QuantileGBDT", "QuantileGBDT + Conformal"],
            "mase": [0.82, 0.74],
            "pinball_p95": [14.0, 12.0],
        }
    )

    artifacts = compile_execution_dossier(
        runs_root=runs_root,
        run_id="rs_2024_smoke",
        eval_year=2024,
        target_label="uti_adulto_diarias (RS 2024)",
        benchmark_table=bench_df,
        hygiene_scorecard=scorecard,
        ratchet_evaluation=ratchet_eval,
        extra_figures=(),
        mirror_champion=True,
        mirror_dir=downloads_mirror,
        prefer_chromium=False,
    )

    assert isinstance(artifacts, DossierArtifacts)
    assert artifacts.markdown_path.exists()
    assert artifacts.pdf_path.exists()
    assert artifacts.pdf_path.read_bytes().startswith(b"%PDF-")
    assert artifacts.hygiene_json_path.exists()
    assert artifacts.ablation_json_path is not None and artifacts.ablation_json_path.exists()
    assert artifacts.manifest_path.exists()
    assert len(artifacts.figure_paths) >= 2

    md_text = artifacts.markdown_path.read_text(encoding="utf-8")
    assert "Not for patient care or live clinical use" in md_text
    assert "G2" in md_text and "G8" in md_text
    assert "PROMOTE" in md_text

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "rs_2024_smoke"
    assert manifest["eval_year"] == 2024

    # Espelhamento do Campeão no diretório mirror_dir (simula ~/Downloads)
    assert artifacts.mirrored_markdown_path is not None
    assert artifacts.mirrored_pdf_path is not None
    assert artifacts.mirrored_markdown_path.exists()
    assert artifacts.mirrored_pdf_path.exists()
    assert artifacts.mirrored_pdf_path.read_bytes().startswith(b"%PDF-")
    assert (downloads_mirror / "CAMPEAO_previsao_leitos_relatorio.pdf").exists()
    assert (downloads_mirror / "CAMPEAO_previsao_leitos_relatorio.md").exists()


def test_pending_items_render_html_lists_data_access_and_open_analysis() -> None:
    from previsao_leitos.audit.pending_items import (
        PENDING_ITEMS,
        render_pending_items_html,
    )

    assert len(PENDING_ITEMS) >= 5
    html_out = render_pending_items_html()
    assert html_out.startswith("<ul")
    assert html_out.endswith("</ul>")
    # Cobre os acessos pendentes e os pontos analíticos em aberto
    for keyword in ("SIVEP-Gripe", "SINAN", "S2iD", "ANA HidroWeb", "fase", "ENSO", "catraca"):
        assert keyword in html_out
