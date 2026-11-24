"""Compilador automático de Dossiê Científico Pós-Treino (.md + .pdf + espelho).

Gera automaticamente ao final de toda execução de treinamento:
- `reports/runs/<run_id>/relatorio_execucao_completo.md`
- `reports/runs/<run_id>/relatorio_execucao_completo.pdf` (Chromium headless ou fallback PdfPages)
- `reports/runs/<run_id>/data_hygiene_scorecard.json`
- `reports/runs/<run_id>/ablation_delta.json` (quando houver avaliação Champion-Challenger)
- `reports/runs/<run_id>/run_manifest.json`
- Figuras `Fig. C6` e `Fig. C7` (além de referenciar `Fig. C1-C5` quando fornecidas)
- Espelhamento atômico do Campeão em `~/Downloads/` (ou `mirror_dir` configurável).

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import base64
import html
import json
import platform
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import polars as pl

from previsao_leitos.audit.data_hygiene import CLINICAL_DISCLAIMER, DataHygieneScorecard
from previsao_leitos.audit.hypothesis_ledger import RatchetEvaluation
from previsao_leitos.audit.plots import plot_c6_hygiene_scorecard, plot_c7_ablation_delta

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


@dataclass(frozen=True)
class DossierArtifacts:
    """Artefatos emitidos pelo compilador de dossiê pós-treino."""

    run_dir: Path
    markdown_path: Path
    pdf_path: Path
    hygiene_json_path: Path
    ablation_json_path: Path | None
    manifest_path: Path
    figure_paths: tuple[Path, ...]
    mirrored_markdown_path: Path | None = None
    mirrored_pdf_path: Path | None = None


def compute_surge_flags(
    y_true: Sequence[float] | np.ndarray[Any, Any],
    baseline: Sequence[float] | np.ndarray[Any, Any],
    *,
    multiplier: float = 1.10,
    min_positive_value: float = 1e-9,
) -> np.ndarray[Any, Any]:
    """Calcula indicador binário de surto sem rótulo degenerado (0.0 >= 0.0 bloqueado).

    Uma observação só é marcada como surto (`1.0`) quando `y_true > min_positive_value`
    e `y_true >= multiplier * baseline` com `baseline > min_positive_value` (ou, se o
    baseline for nulo, apenas quando `y_true >= 1.0`).
    """
    y_arr = np.asarray(y_true, dtype=np.float64)
    base_arr = np.asarray(baseline, dtype=np.float64)
    valid_pos = y_arr > min_positive_value
    above_base = np.where(
        base_arr > min_positive_value,
        y_arr >= multiplier * base_arr,
        y_arr >= 1.0,
    )
    return (valid_pos & above_base).astype(np.float64)


def _df_to_md(df: pl.DataFrame) -> str:
    cols = df.columns
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in df.iter_rows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_html_from_dossier(
    *,
    title: str,
    markdown_text: str,
    figure_paths: Sequence[Path],
) -> str:
    escaped_md = html.escape(markdown_text)
    img_blocks: list[str] = []
    for fig in figure_paths:
        if fig.exists():
            b64 = base64.b64encode(fig.read_bytes()).decode("ascii")
            img_blocks.append(
                f'<div class="figure"><h4>{html.escape(fig.name)}</h4>'
                f'<img src="data:image/png;base64,{b64}" /></div>'
            )
    images_html = "\n".join(img_blocks)
    return (
        "<!DOCTYPE html><html lang='pt-BR'><head><meta charset='utf-8'/>"
        f"<title>{html.escape(title)}</title>"
        "<style>"
        "body{font-family:DejaVu Sans,Arial,sans-serif;margin:28px;color:#111;line-height:1.45;}"
        "h1{color:#0b3954;border-bottom:2px solid #0b3954;padding-bottom:6px;}"
        ".banner{background:#fff3cd;border-left:4px solid #d55e00;padding:10px;margin:12px 0;}"
        "pre{white-space:pre-wrap;font-family:DejaVu Sans Mono,monospace;font-size:10.5px;"
        "background:#f7f7f7;padding:12px;border-radius:4px;}"
        ".figure{page-break-inside:avoid;margin:16px 0;text-align:center;}"
        ".figure img{max-width:96%;height:auto;border:1px solid #ddd;}"
        "</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f"<div class='banner'><strong>Fronteira Clínica:</strong> "
        f"{html.escape(CLINICAL_DISCLAIMER)}</div>"
        f"<pre>{escaped_md}</pre>"
        f"{images_html}"
        "</body></html>"
    )


def _render_pdf_via_chromium(html_path: Path, pdf_path: Path) -> bool:
    chromium_bin = (
        shutil.which("chromium")
        or shutil.which("chromium-browser")
        or shutil.which("google-chrome")
        or shutil.which("google-chrome-stable")
    )
    if not chromium_bin:
        return False
    cmd = [
        chromium_bin,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        f"--print-to-pdf={pdf_path.resolve()}",
        html_path.resolve().as_uri(),
    ]
    try:
        proc = subprocess.run(  # noqa: S603
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=45,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and pdf_path.exists() and pdf_path.stat().st_size > 0


def _render_pdf_fallback_matplotlib(
    *,
    title: str,
    markdown_text: str,
    figure_paths: Sequence[Path],
    pdf_path: Path,
) -> Path:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    lines = markdown_text.splitlines()
    lines_per_page = 52

    with PdfPages(pdf_path) as pdf:
        for start in range(0, max(1, len(lines)), lines_per_page):
            chunk = "\n".join(lines[start : start + lines_per_page])
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.06, 0.96, title, fontsize=11, fontweight="bold", va="top")
            fig.text(0.06, 0.92, chunk, fontsize=7.5, family="monospace", va="top")
            pdf.savefig(fig)
            plt.close(fig)

        for fig_path in figure_paths:
            if fig_path.exists():
                img = plt.imread(fig_path)
                fig, ax = plt.subplots(figsize=(8.27, 6.2))
                ax.imshow(img)
                ax.set_title(fig_path.name, fontsize=10, fontweight="bold")
                ax.axis("off")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    return pdf_path


def _atomic_copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(f"{dst.suffix}.tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)
    return dst


def compile_execution_dossier(
    *,
    runs_root: Path,
    run_id: str,
    eval_year: int,
    target_label: str,
    benchmark_table: pl.DataFrame,
    hygiene_scorecard: DataHygieneScorecard,
    ratchet_evaluation: RatchetEvaluation | None = None,
    extra_figures: Sequence[Path] = (),
    extra_notes: Sequence[str] = (),
    seed: int = 42,
    mirror_champion: bool = False,
    mirror_dir: Path | None = None,
    prefer_chromium: bool = True,
) -> DossierArtifacts:
    """Compila o dossiê completo (.md + .pdf + JSONs + Fig. C1-C7) e espelha o Campeão."""
    run_dir = (runs_root / run_id).resolve()
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig_c6 = plot_c6_hygiene_scorecard(
        hygiene_scorecard,
        output_path=fig_dir / "fig_c6_higiene_dados.png",
    )
    generated_figs: list[Path] = [*[Path(p) for p in extra_figures if Path(p).exists()], fig_c6]

    ablation_json_path: Path | None = None
    if ratchet_evaluation is not None:
        ablation_json_path = ratchet_evaluation.save_ablation_delta(
            run_dir / "ablation_delta.json"
        )
        fig_c7 = plot_c7_ablation_delta(
            ratchet_evaluation,
            output_path=fig_dir / "fig_c7_ablacao_champion_challenger.png",
        )
        generated_figs.append(fig_c7)

    hygiene_json_path = hygiene_scorecard.save_json(run_dir / "data_hygiene_scorecard.json")

    gate_rows = [
        {
            "gate": g.gate_id,
            "indicador": g.name,
            "status": g.status,
            "observado": f"{g.observed_value:.4g} {g.unit}",
            "limite": f"<= {g.threshold:.4g} {g.unit}",
            "resumo": g.summary,
        }
        for g in hygiene_scorecard.gates
    ]
    gates_md = _df_to_md(pl.DataFrame(gate_rows))
    bench_md = _df_to_md(benchmark_table)

    ratchet_section = "_Sem comparação Champion-Challenger registrada nesta execução._"
    if ratchet_evaluation is not None:
        delta_rows = [
            {
                "metrica": metric,
                "champion": vals["champion"],
                "challenger": vals["challenger"],
                "delta": vals["delta"],
                "delta_pct": vals["delta_pct"],
            }
            for metric, vals in ratchet_evaluation.ablation_delta.items()
        ]
        ratchet_section = (
            f"- **Decisão da Catraca:** `{ratchet_evaluation.decision}` "
            f"(Vencedor: `{ratchet_evaluation.winner_run_id}`)\n"
            f"- **Resumo:** {ratchet_evaluation.summary}\n\n"
            + _df_to_md(pl.DataFrame(delta_rows))
        )

    notes_md = (
        "\n".join(f"- {note}" for note in extra_notes)
        if extra_notes
        else "- Execução concluída sob políticas causais congeladas."
    )
    fig_list_md = "\n".join(f"- `{p.name}`" for p in generated_figs)

    generated_at = datetime.now(UTC).isoformat()
    md_content = "\n".join(
        [
            f"# Relatório Completo de Execução Científica — {run_id}",
            "",
            f"> **Aviso Obrigatório:** {CLINICAL_DISCLAIMER}",
            f"> **Alvo:** `{target_label}` | **Ano Avaliado:** `{eval_year}` | "
            f"**Gerado em (UTC):** `{generated_at}` | **Seed:** `{seed}`",
            "",
            "---",
            "",
            "## 1. Scorecard de Higiene de Dados Cruzados (Gates G1-G8)",
            f"- **Status Global:** `{hygiene_scorecard.overall_status}` "
            f"(PASS={hygiene_scorecard.passed_count}, WARN={hygiene_scorecard.warn_count}, "
            f"FAIL={hygiene_scorecard.failed_count})",
            "",
            gates_md,
            "",
            "## 2. Desempenho Preditivo e Calibração (Benchmark)",
            "",
            bench_md,
            "",
            "## 3. Catraca Científica Champion-Challenger",
            "",
            ratchet_section,
            "",
            "## 4. Notas Metodológicas e Governança (TRIPOD+AI / RECORD)",
            "",
            notes_md,
            "",
            "## 5. Figuras Geradas (Fig. C1-C7)",
            "",
            fig_list_md,
            "",
        ]
    )

    markdown_path = run_dir / "relatorio_execucao_completo.md"
    markdown_path.write_text(md_content, encoding="utf-8")

    pdf_path = run_dir / "relatorio_execucao_completo.pdf"
    html_path = run_dir / "_relatorio_render.html"
    html_text = _render_html_from_dossier(
        title=f"Previsão de Leitos — Dossiê Científico ({run_id})",
        markdown_text=md_content,
        figure_paths=generated_figs,
    )
    html_path.write_text(html_text, encoding="utf-8")

    pdf_ok = False
    if prefer_chromium:
        pdf_ok = _render_pdf_via_chromium(html_path, pdf_path)
    if not pdf_ok:
        _render_pdf_fallback_matplotlib(
            title=f"Previsão de Leitos — Dossiê Científico ({run_id})",
            markdown_text=md_content,
            figure_paths=generated_figs,
            pdf_path=pdf_path,
        )
    html_path.unlink(missing_ok=True)

    manifest_path = run_dir / "run_manifest.json"
    manifest_payload = {
        "run_id": run_id,
        "generated_at_utc": generated_at,
        "eval_year": eval_year,
        "target_label": target_label,
        "seed": seed,
        "python_version": platform.python_version(),
        "hygiene_overall_status": hygiene_scorecard.overall_status,
        "ratchet_decision": ratchet_evaluation.decision if ratchet_evaluation else None,
        "artifacts": {
            "markdown": str(markdown_path),
            "pdf": str(pdf_path),
            "hygiene_scorecard": str(hygiene_json_path),
            "ablation_delta": str(ablation_json_path) if ablation_json_path else None,
            "figures": [str(p) for p in generated_figs],
        },
        "disclaimer": CLINICAL_DISCLAIMER,
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    mirrored_md: Path | None = None
    mirrored_pdf: Path | None = None
    if mirror_champion:
        target_mirror = (mirror_dir or (Path.home() / "Downloads")).resolve()
        mirrored_md = _atomic_copy(
            markdown_path, target_mirror / f"{run_id}_relatorio_execucao_completo.md"
        )
        mirrored_pdf = _atomic_copy(
            pdf_path, target_mirror / f"{run_id}_relatorio_execucao_completo.pdf"
        )
        _atomic_copy(markdown_path, target_mirror / "CAMPEAO_previsao_leitos_relatorio.md")
        _atomic_copy(pdf_path, target_mirror / "CAMPEAO_previsao_leitos_relatorio.pdf")

    return DossierArtifacts(
        run_dir=run_dir,
        markdown_path=markdown_path,
        pdf_path=pdf_path,
        hygiene_json_path=hygiene_json_path,
        ablation_json_path=ablation_json_path,
        manifest_path=manifest_path,
        figure_paths=tuple(generated_figs),
        mirrored_markdown_path=mirrored_md,
        mirrored_pdf_path=mirrored_pdf,
    )
