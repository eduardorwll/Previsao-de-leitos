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
from previsao_leitos.audit.pending_items import render_pending_items_html
from previsao_leitos.audit.plots import plot_c6_hygiene_scorecard, plot_c7_ablation_delta
from previsao_leitos.audit.report_builder import collect_runtime_manifest_metadata

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


def _status_badge_html(val: str) -> str:
    upper = val.strip().upper()
    if upper in {"PASS", "PROMOTE", "OK"}:
        cls = "badge-pass"
    elif upper in {"WARN"}:
        cls = "badge-warn"
    elif upper in {"FAIL", "REJECT"}:
        cls = "badge-fail"
    else:
        return html.escape(val)
    return f'<span class="badge {cls}">{html.escape(upper)}</span>'


def _escape_with_wbr(text: str) -> str:
    """Escapa HTML inserindo quebra suave após '_' para evitar cortes no meio de palavras."""
    return html.escape(text).replace("_", "_<wbr>")


def _df_to_html_table(df: pl.DataFrame) -> str:
    cols = df.columns
    is_wide = len(cols) >= 8
    headers = "".join(
        f"<th>{html.escape(str(c).replace('_', ' '))}</th>"
        for c in cols
    )
    body_rows: list[str] = []
    for row in df.iter_rows():
        tds: list[str] = []
        for col, v in zip(cols, row, strict=True):
            if isinstance(v, float):
                if not np.isfinite(v):
                    cell = "—"
                elif is_wide:
                    cell = f"{v:.3f}"
                else:
                    cell = f"{v:.4f}"
                tds.append(f'<td class="num">{cell}</td>')
            elif isinstance(v, int) and not isinstance(v, bool):
                tds.append(f'<td class="num">{v:,}</td>')
            else:
                sval = str(v)
                if col.lower() in {"status", "decisao", "decision"}:
                    tds.append(f"<td>{_status_badge_html(sval)}</td>")
                elif is_wide:
                    tds.append(f"<td>{_escape_with_wbr(sval)}</td>")
                else:
                    tds.append(f"<td>{html.escape(sval)}</td>")
        body_rows.append("<tr>" + "".join(tds) + "</tr>")
    cls = "data-table compact" if is_wide else "data-table"
    return (
        f"<table class='{cls}'>"
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def _extract_kpi_cards(
    *,
    eval_year: int,
    hygiene_scorecard: DataHygieneScorecard,
    benchmark_table: pl.DataFrame,
    ratchet_evaluation: RatchetEvaluation | None,
    seed: int,
) -> list[tuple[str, str, str]]:
    total_gates = len(hygiene_scorecard.gates)
    gates_val = f"{hygiene_scorecard.passed_count}/{total_gates} PASS"
    gates_sub = (
        f"WARN={hygiene_scorecard.warn_count} | FAIL={hygiene_scorecard.failed_count}"
    )

    mase_val = "—"
    mase_sub = "Sem coluna MASE"
    for mcol in ("mase_vs_pers", "mase"):
        if mcol in benchmark_table.columns and benchmark_table.height > 0:
            vals = [
                float(x)
                for x in benchmark_table[mcol].to_list()
                if x is not None and np.isfinite(float(x))
            ]
            if vals:
                best_mase = min(vals)
                mase_val = f"{best_mase:.3f}"
                mase_sub = "Menor MASE no benchmark (< 1,00 vence persistência)"
                break

    cov_val = "—"
    cov_sub = "Alvo nominal P10-P90: 80%"
    if "cobertura_p10_p90" in benchmark_table.columns and benchmark_table.height > 0:
        vals = [
            float(x)
            for x in benchmark_table["cobertura_p10_p90"].to_list()
            if x is not None and np.isfinite(float(x))
        ]
        if vals:
            cov_val = f"{100.0 * vals[-1]:.1f}%"
            cov_sub = "Cobertura empírica da banda P10-P90"

    p95_val = "—"
    p95_sub = "Perda quantílica de cauda (P95)"
    if "pinball_p95" in benchmark_table.columns and benchmark_table.height > 0:
        vals = [
            float(x)
            for x in benchmark_table["pinball_p95"].to_list()
            if x is not None and np.isfinite(float(x))
        ]
        if vals:
            p95_val = f"{min(vals):.2f}"
            p95_sub = "Menor Pinball Loss P95 auditada"

    ratchet_val = ratchet_evaluation.decision if ratchet_evaluation else "N/A"
    ratchet_sub = (
        f"Vencedor: {ratchet_evaluation.winner_run_id}"
        if ratchet_evaluation
        else "Sem comparação Champion-Challenger"
    )

    holdout_note = (
        "Holdout 2025 preservado"
        if eval_year < 2025
        else f"Auditoria holdout/prospectiva ({eval_year})"
    )
    return [
        ("Higiene de Dados (G1-G8)", gates_val, gates_sub),
        ("Melhor MASE (vs. Persistência)", mase_val, mase_sub),
        ("Pinball Loss P95", p95_val, p95_sub),
        ("Cobertura Empírica P10-P90", cov_val, cov_sub),
        ("Catraca Champion-Challenger", ratchet_val, ratchet_sub),
        ("Ano Auditado / Seed", f"{eval_year} (seed {seed})", holdout_note),
    ]


def _render_html_from_dossier(
    *,
    title: str,
    run_id: str,
    eval_year: int,
    target_label: str,
    generated_at: str,
    seed: int,
    git_sha: str,
    hygiene_scorecard: DataHygieneScorecard,
    benchmark_table: pl.DataFrame,
    ratchet_evaluation: RatchetEvaluation | None,
    extra_notes: Sequence[str],
    figure_paths: Sequence[Path],
    extra_tables: Sequence[tuple[str, pl.DataFrame]] = (),
) -> str:
    """Renderiza dossiê em HTML5/CSS3 editorial (padrao-relatorio-tecnico-pdf.md, zero Markdown)."""
    gate_df = pl.DataFrame(
        [
            {
                "Gate": g.gate_id,
                "Indicador": g.name,
                "Status": g.status,
                "Observado": f"{g.observed_value:.4g} {g.unit}",
                "Limite": f"<= {g.threshold:.4g} {g.unit}",
                "Resumo": g.summary,
            }
            for g in hygiene_scorecard.gates
        ]
    )

    kpis = _extract_kpi_cards(
        eval_year=eval_year,
        hygiene_scorecard=hygiene_scorecard,
        benchmark_table=benchmark_table,
        ratchet_evaluation=ratchet_evaluation,
        seed=seed,
    )
    kpi_html = "".join(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>{html.escape(lbl)}</div>"
        f"<div class='kpi-value'>{html.escape(val)}</div>"
        f"<div class='kpi-sub'>{html.escape(sub)}</div>"
        f"</div>"
        for lbl, val, sub in kpis
    )

    ratchet_html = (
        "<div class='callout info'>Sem comparação Champion-Challenger registrada nesta execução."
        "</div>"
    )
    if ratchet_evaluation is not None:
        delta_df = pl.DataFrame(
            [
                {
                    "Métrica": metric,
                    "Champion": vals["champion"],
                    "Challenger": vals["challenger"],
                    "Delta": vals["delta"],
                    "Delta (%)": vals["delta_pct"],
                }
                for metric, vals in ratchet_evaluation.ablation_delta.items()
            ]
        )
        callout_cls = "success" if ratchet_evaluation.decision == "PROMOTE" else "warn"
        ratchet_html = (
            f"<div class='callout {callout_cls}'>"
            f"<strong>Decisão da Catraca:</strong> "
            f"{_status_badge_html(ratchet_evaluation.decision)} "
            f"| <strong>Vencedor:</strong> {html.escape(ratchet_evaluation.winner_run_id)}<br/>"
            f"{html.escape(ratchet_evaluation.summary)}"
            f"</div>"
            + _df_to_html_table(delta_df)
        )

    extra_tables_html = ""
    for tbl_title, tbl_df in extra_tables:
        if not tbl_df.is_empty():
            extra_tables_html += (
                f"<h3>{html.escape(tbl_title)}</h3>{_df_to_html_table(tbl_df)}"
            )

    notes_items = "".join(
        f"<li>{html.escape(note)}</li>"
        for note in (
            extra_notes
            if extra_notes
            else ("Execução concluída sob políticas causais congeladas (ADR-0003).",)
        )
    )

    img_blocks: list[str] = []
    for idx, fig in enumerate(figure_paths, start=1):
        if fig.exists():
            b64 = base64.b64encode(fig.read_bytes()).decode("ascii")
            caption = fig.stem.replace("_", " ").upper()
            img_blocks.append(
                f"<figure class='figure'>"
                f"<img src='data:image/png;base64,{b64}' alt='{html.escape(fig.name)}' />"
                f"<figcaption><strong>Figura {idx}:</strong> {html.escape(caption)} "
                f"({html.escape(fig.name)} — paleta Okabe-Ito 300 DPI)</figcaption>"
                f"</figure>"
            )
    images_html = "\n".join(img_blocks)

    split_policy_note = (
        f"treino &le; 2022, calibração 2023, auditoria {eval_year}, "
        "holdout 2025 intocado — ADR-0003"
        if eval_year < 2025
        else f"treino &le; 2022, calibração 2023, avaliação holdout/prospectiva "
        f"{eval_year} — ADR-0003"
    )

    return (
        "<!DOCTYPE html><html lang='pt-BR'><head><meta charset='utf-8'/>"
        f"<title>{html.escape(title)}</title>"
        "<style>"
        "@page{size:A4;margin:15mm 14mm 18mm 14mm;"
        "@bottom-left{content:'Pesquisa aberta - Not for patient care or live clinical use "
        "| LGPD n < 5';"
        "font-family:DejaVu Sans,Arial,sans-serif;font-size:8px;color:#475569;}"
        "@bottom-right{content:'Página ' counter(page) ' de ' counter(pages);"
        "font-family:DejaVu Sans,Arial,sans-serif;font-size:8px;color:#475569;}}"
        "body{font-family:DejaVu Sans,Arial,sans-serif;margin:0;color:#0f172a;"
        "font-size:10.5px;line-height:1.48;}"
        ".cover{background:linear-gradient(135deg,#072a40 0%,#0b4a6f 55%,#115e59 100%);"
        "color:#fff;padding:22px 24px;border-radius:8px;margin-bottom:14px;}"
        ".cover h1{margin:0 0 6px 0;font-size:19px;letter-spacing:-0.2px;}"
        ".cover .meta{font-size:9.5px;opacity:0.92;}"
        ".banner-clinical{background:#fff7ed;border-left:5px solid #d55e00;color:#7c2d12;"
        "padding:10px 14px;border-radius:4px;margin:10px 0 14px 0;font-size:10px;}"
        ".kpi-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:12px 0 18px 0;"
        "page-break-inside:avoid;}"
        ".kpi-card{background:#f8fafc;border:1px solid #cbd5e1;border-top:3px solid #0b4a6f;"
        "border-radius:6px;padding:10px 12px;}"
        ".kpi-label{font-size:8.5px;text-transform:uppercase;color:#475569;font-weight:700;}"
        ".kpi-value{font-size:15px;font-weight:700;color:#072a40;margin:4px 0 2px 0;"
        "font-variant-numeric:tabular-nums;}"
        ".kpi-sub{font-size:8.5px;color:#64748b;}"
        "h2.chapter{background:#0b3954;color:#fff;padding:7px 12px;border-radius:4px;"
        "font-size:12.5px;margin:18px 0 10px 0;page-break-after:avoid;}"
        "h3{color:#0b4a6f;font-size:11px;margin:12px 0 6px 0;page-break-after:avoid;}"
        ".data-table{width:100%;border-collapse:collapse;margin:8px 0 14px 0;font-size:9px;"
        "page-break-inside:avoid;}"
        ".data-table thead{display:table-header-group;}"
        ".data-table tr{page-break-inside:avoid;}"
        ".data-table.compact{table-layout:fixed;font-size:6.8px;}"
        ".data-table thead th{background:#0b3954;color:#fff;padding:6px 7px;text-align:left;"
        "font-weight:700;}"
        ".data-table.compact thead th{padding:4px 2.5px;line-height:1.2;"
        "word-break:normal;overflow-wrap:break-word;}"
        ".data-table.compact thead th:first-child,.data-table.compact tbody td:first-child{"
        "width:22%;}"
        ".data-table tbody td{padding:5px 7px;border-bottom:1px solid #e2e8f0;vertical-align:top;}"
        ".data-table.compact tbody td{padding:4px 2.5px;"
        "word-break:normal;overflow-wrap:break-word;}"
        ".data-table tbody tr:nth-child(even){background:#f8fafc;}"
        ".data-table td.num{text-align:right;font-variant-numeric:tabular-nums;}"
        ".badge{display:inline-block;padding:2px 6px;border-radius:10px;font-size:8px;"
        "font-weight:700;letter-spacing:0.3px;white-space:nowrap;}"
        ".chapter-block{page-break-inside:avoid;}"
        ".badge-pass{background:#dcfce7;color:#166534;}"
        ".badge-warn{background:#fef3c7;color:#92400e;}"
        ".badge-fail{background:#fee2e2;color:#991b1b;}"
        ".callout{padding:10px 12px;border-radius:5px;margin:10px 0;font-size:9.8px;"
        "page-break-inside:avoid;}"
        ".callout.info{background:#eff6ff;border-left:4px solid #0072b2;color:#1e3a8a;}"
        ".callout.warn{background:#fff7ed;border-left:4px solid #d55e00;color:#7c2d12;}"
        ".callout.success{background:#ecfdf5;border-left:4px solid #009e73;color:#065f46;}"
        "ul.notes{margin:6px 0 12px 18px;padding:0;}"
        "ul.notes li{margin:4px 0;}"
        ".figure{page-break-inside:avoid;margin:14px 0;text-align:center;"
        "border:1px solid #e2e8f0;border-radius:6px;padding:8px;background:#fff;}"
        ".figure img{max-width:96%;height:auto;}"
        ".figure figcaption{font-size:8.8px;color:#334155;margin-top:6px;}"
        ".page-break{page-break-before:always;}"
        "</style></head><body>"
        "<header class='cover'>"
        f"<h1>{html.escape(title)}</h1>"
        f"<div class='meta'><strong>Execução:</strong> {html.escape(run_id)} | "
        f"<strong>Alvo:</strong> {html.escape(target_label)} | "
        f"<strong>Ano Auditado:</strong> {eval_year} | "
        f"<strong>Commit:</strong> {html.escape(git_sha)} | "
        f"<strong>Gerado (UTC):</strong> {html.escape(generated_at)}</div>"
        "</header>"
        "<div class='banner-clinical'><strong>Fronteira Clínica e LGPD:</strong> "
        f"{html.escape(CLINICAL_DISCLAIMER)} Zero PII de pacientes; agregação na fronteira "
        "BigQuery/OpenDataSUS com supressão de pequenas células (n &lt; 5 — ADR-0005).</div>"
        f"<section class='kpi-grid'>{kpi_html}</section>"
        "<h2 class='chapter'>Capítulo 1 — Escopo Metodológico, Modelos e Contratos Causais</h2>"
        "<div class='callout info'>Previsão estritamente <em>ex-ante</em> (h = 2 semanas / D+14) "
        "em séries semanais ISO por Região de Saúde, com restrição monotônica estrutural "
        "(ADR-0001), acoplamento hidrológico-assistencial (ADR-0006), splits temporais congelados "
        f"({split_policy_note}) "
        "e avaliação contrafactual (ADR-0007).</div>"
        "<h2 class='chapter'>Capítulo 2 — Significado Estatístico e Benchmark Preditivo</h2>"
        f"{_df_to_html_table(benchmark_table)}"
        "<section class='chapter-block'>"
        "<h2 class='chapter'>Capítulo 3 — Rastreabilidade e Scorecard de Higiene "
        "(Gates G1-G8)</h2>"
        f"<div class='callout info'><strong>Status Global dos Gates:</strong> "
        f"{_status_badge_html(hygiene_scorecard.overall_status)} "
        f"(PASS={hygiene_scorecard.passed_count}, WARN={hygiene_scorecard.warn_count}, "
        f"FAIL={hygiene_scorecard.failed_count})</div>"
        f"{_df_to_html_table(gate_df)}"
        "</section>"
        "<h2 class='chapter'>Capítulo 4 — Simulações Contrafactuais e Catraca "
        "Champion-Challenger</h2>"
        f"{ratchet_html}"
        f"{extra_tables_html}"
        f"{images_html}"
        "<section class='chapter-block'>"
        "<h2 class='chapter'>Capítulo 5 — Arquitetura, Reprodutibilidade e Governança</h2>"
        f"<ul class='notes'>{notes_items}</ul>"
        "</section>"
        "<section class='chapter-block'>"
        "<h2 class='chapter'>Capítulo 6 — Limitações Auditadas e Próximas Etapas</h2>"
        "<div class='callout info'>Todos os resultados, falhas de higiene e eventuais rejeições "
        "da catraca Champion-Challenger são reportados integralmente sem omissão. O artefato "
        "CAMPEAO é atualizado exclusivamente mediante decisão PROMOTE.</div>"
        "<h3>Pontos pendentes para estudo posterior (acessos a dados e análises em aberto)</h3>"
        f"{render_pending_items_html()}"
        "</section>"
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
        "--no-pdf-header-footer",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={pdf_path.resolve()}",
        html_path.resolve().as_uri(),
    ]
    try:
        proc = subprocess.run(  # noqa: S603
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
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
    extra_tables: Sequence[tuple[str, pl.DataFrame]] = (),
    seed: int = 42,
    mirror_champion: bool = False,
    mirror_dir: Path | None = None,
    prefer_chromium: bool = True,
    device: str = "cpu",
    gpu_name: str | None = None,
    cuda_version: str | None = None,
    precision: str = "fp32",
    deterministic_mode: str = "strict",
) -> DossierArtifacts:
    """Compila o dossiê completo (.md + .pdf editorial + JSONs + Fig. C1-C7) e espelha Campeão."""
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

    extra_tables_md = ""
    for tbl_title, tbl_df in extra_tables:
        if not tbl_df.is_empty():
            extra_tables_md += f"\n### {tbl_title}\n\n{_df_to_md(tbl_df)}\n"

    notes_md = (
        "\n".join(f"- {note}" for note in extra_notes)
        if extra_notes
        else "- Execução concluída sob políticas causais congeladas."
    )
    fig_list_md = "\n".join(f"- `{p.name}`" for p in generated_figs)

    runtime_meta = collect_runtime_manifest_metadata(
        seed=seed,
        device=device,
        gpu_name=gpu_name,
        cuda_version=cuda_version,
        precision=precision,
        deterministic_mode=deterministic_mode,
    )
    git_sha = str(runtime_meta["git_sha"])
    generated_at = datetime.now(UTC).isoformat()
    md_content = "\n".join(
        [
            f"# Relatório Completo de Execução Científica — {run_id}",
            "",
            f"> **Aviso Obrigatório:** {CLINICAL_DISCLAIMER}",
            f"> **Alvo:** `{target_label}` | **Ano Avaliado:** `{eval_year}` | "
            f"**Commit:** `{git_sha}` | **Gerado em (UTC):** `{generated_at}` | **Seed:** `{seed}`",
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
            extra_tables_md,
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
        run_id=run_id,
        eval_year=eval_year,
        target_label=target_label,
        generated_at=generated_at,
        seed=seed,
        git_sha=git_sha,
        hygiene_scorecard=hygiene_scorecard,
        benchmark_table=benchmark_table,
        ratchet_evaluation=ratchet_evaluation,
        extra_notes=extra_notes,
        figure_paths=generated_figs,
        extra_tables=extra_tables,
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
        "git_sha": git_sha,
        "python_version": runtime_meta["python_version"],
        "device": runtime_meta["device"],
        "gpu_name": runtime_meta["gpu_name"],
        "cuda_version": runtime_meta["cuda_version"],
        "precision": runtime_meta["precision"],
        "deterministic_mode": runtime_meta["deterministic_mode"],
        "eval_year": eval_year,
        "target_label": target_label,
        "seed": seed,
        "tools": runtime_meta["tools"],
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
        # Trava de integridade (Bug J): só sobrescreve CAMPEAO_* sob PROMOTE (ou sem catraca)
        can_promote_champion = (
            ratchet_evaluation is None or ratchet_evaluation.decision == "PROMOTE"
        )
        if can_promote_champion:
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
