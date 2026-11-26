"""Relatório comparativo do benchmark: HTML5/CSS3 editorial e PDF via Chromium.

Consome apenas artefatos de um run (`benchmark.csv`, `predictions.parquet`,
`comparison.csv`, `decisions.csv`) e as figuras já geradas. Nenhum Markdown cru
entra no PDF; a fronteira clínica aparece no cabeçalho e no rodapé.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import hashlib
import html
import json
import shutil
from pathlib import Path

import polars as pl

from previsao_leitos.audit.dossier_compiler import _render_pdf_via_chromium

_DISCLAIMER = "Not for patient care or live clinical use"

_CSS = """
:root{--ink:#111827;--muted:#475569;--line:#cbd5e1;--bg:#ffffff;--accent:#0f766e}
*{box-sizing:border-box}
body{font-family:"DejaVu Serif",Georgia,serif;color:var(--ink);margin:0;line-height:1.45}
main{max-width:190mm;margin:0 auto;padding:14mm 10mm 22mm}
h1{font-size:20pt;margin:0 0 4px;letter-spacing:.2px}
h2{font-size:13pt;margin:22px 0 8px;padding-bottom:4px;border-bottom:1.5px solid var(--line)}
h3{font-size:11pt;margin:14px 0 6px}
p,li{font-size:9.6pt}
.meta{color:var(--muted);font-size:8.6pt;margin-bottom:14px}
.badge{display:inline-block;border:1px solid var(--accent);color:var(--accent);
 border-radius:3px;padding:1px 6px;font-size:8pt;margin-right:6px}
table{border-collapse:collapse;width:100%;margin:8px 0 14px;font-size:7.6pt}
th,td{border-bottom:1px solid var(--line);padding:3px 4px;text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{background:#f1f5f9;font-weight:700}
tr.highlight td{background:#ecfdf5;font-weight:700}
figure{margin:12px 0 18px;page-break-inside:avoid}
figure img{width:100%;border:1px solid var(--line)}
figcaption{font-size:8pt;color:var(--muted);margin-top:4px}
.note{background:#f8fafc;border-left:3px solid var(--accent);padding:8px 10px;font-size:8.8pt}
footer{margin-top:26px;border-top:1px solid var(--line);padding-top:6px;
 color:var(--muted);font-size:7.6pt}
@page{size:A4;margin:15mm 12mm 18mm 12mm}
"""


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "sim" if value else "não"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        if value != value:  # NaN
            return "—"
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def _table(df: pl.DataFrame, columns: list[str], *, highlight_column: str | None = None) -> str:
    headers = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    rows: list[str] = []
    for row in df.select(columns).iter_rows(named=True):
        css = ""
        if highlight_column and row.get(highlight_column) is True:
            css = ' class="highlight"'
        cells = "".join(f"<td>{_fmt(row.get(col))}</td>" for col in columns)
        rows.append(f"<tr{css}>{cells}</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _figure(path: Path, caption: str) -> str:
    if not path.exists():
        return ""
    return (
        f'<figure><img src="{path.name}" alt="{html.escape(caption)}"/>'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def _target_sections(run_dir: Path, metrics: pl.DataFrame, target: str, label: str) -> str:
    subset = metrics.filter(pl.col("target") == target)
    if subset.height == 0:
        return ""
    ok = subset.filter(pl.col("status") == "ok")
    columns = [
        "candidate",
        "status",
        "mae_p50",
        "mase_vs_pers",
        "pinball_p95",
        "cobertura_p10_p90",
        "peak_timing_error_weeks",
        "monotonic_violations",
        "fit_seconds",
    ]
    parts = [
        f"<h2>{html.escape(label)}</h2>",
        "<p>Métricas no ano de avaliação, com treino até 2022 e calibração conformal em 2023. "
        "MASE &lt; 1 supera a persistência; cobertura ideal de P10-P90 é 80%.</p>",
        _table(
            subset.sort("mase_vs_pers", nulls_last=True),
            columns,
            highlight_column="is_champion",
        ),
    ]
    if ok.height:
        best = ok.sort("mase_vs_pers").head(1)["candidate"].to_list()[0]
        champion_rows = ok.filter(pl.col("is_champion"))
        champion = champion_rows["candidate"].to_list()[0] if champion_rows.height else best
        chosen = list(dict.fromkeys([champion, best]))[:2]
        parts.append(
            _figure(
                run_dir / f"fig_observed_vs_predicted_{target}.png",
                f"Leitos observados vs. P50 previsto (soma regional por semana) — {label}.",
            )
        )
        parts.append(
            _figure(
                run_dir / f"fig_residuals_{target}.png",
                f"Viés médio por região de saúde ({', '.join(chosen)}).",
            )
        )
        parts.append(
            _figure(
                run_dir / f"fig_calibration_{target}.png",
                f"Cobertura empírica e Pinball P95 por modelo — {label}.",
            )
        )
    return "".join(parts)


def render_report(
    run_dir: Path,
    *,
    target_labels: dict[str, str],
    eval_year: int,
    models_requested: list[str],
    pdf_name: str = "relatorio_benchmark_leitos.pdf",
) -> tuple[Path, Path]:
    """Gera `relatorio_benchmark_leitos.html` e o PDF editorial correspondente."""
    metrics_path = run_dir / "benchmark.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"benchmark.csv ausente em {run_dir}")
    metrics = pl.read_csv(metrics_path)
    if "target" not in metrics.columns:
        raise ValueError("benchmark.csv sem coluna 'target'.")

    manifest_path = run_dir / "run_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    decisions_path = run_dir / "decisions.csv"
    decisions = pl.read_csv(decisions_path) if decisions_path.exists() else pl.DataFrame()

    ok = metrics.filter(pl.col("status") == "ok")
    n_ok = ok.height
    n_total = metrics.height
    sections = "".join(
        _target_sections(run_dir, metrics, target, label)
        for target, label in target_labels.items()
    )

    decisions_block = ""
    if decisions.height:
        decisions_block = (
            "<h2>Catraca Champion-Challenger</h2>"
            "<p>Decisão quantílica contra o campeão registrado "
            "(MASE, Pinball P95, cobertura e timing de pico).</p>"
            + _table(
                decisions,
                ["candidate", "champion", "decision", "summary"],
            )
        )

    manifest_block = ""
    if manifest:
        items = "".join(
            f"<li><strong>{html.escape(str(k))}</strong>: {_fmt(v)}</li>"
            for k, v in manifest.items()
            if not isinstance(v, (dict, list))
        )
        manifest_block = f"<h2>Reprodutibilidade</h2><ul>{items}</ul>"

    body = f"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="utf-8"/>
<title>Relatório de Benchmark — Leitos RS {eval_year}</title>
<style>{_CSS}</style></head>
<body><main>
<h1>Benchmark de Modelos de Demanda de Leitos</h1>
<div class="meta">
<span class="badge">RS {eval_year}</span>
<span class="badge">Pesquisa aberta</span>
<span class="badge">{n_ok}/{n_total} candidatos concluídos</span><br/>
Comparação entre leitos ocupados observados e preditos, com calibração, catraca e
limitações explícitas.
</div>

<h2>1. Escopo e método</h2>
<p>Treino causal até 2022, calibração conformal em 2023 e avaliação em {eval_year}.
O holdout de 2025 e o ano prospectivo de 2026 permanecem lacrados. Nenhuma métrica
deste relatório é afirmação de desempenho <em>out-of-time</em>.</p>
<p>Modelos avaliados nesta execução: {html.escape(", ".join(models_requested))}.</p>

<h2>2. Modelos e famílias</h2>
<p>Baselines obrigatórios (persistência e sazonal-ingênuo) definem o piso; o campeão
registrado aparece destacado nas tabelas. Candidatos indisponíveis por dependência
opcional permanecem listados com status próprio, nunca omitidos.</p>

{sections}

{decisions_block}

<h2>5. Calibração, monotonicidade e riscos</h2>
<div class="note">
A coluna <strong>monotonic_violations</strong> conta quedas de P50 quando o choque
hidrológico ex-ante aumenta (ADR-0001); o gate é zero. Cobertura fora da faixa
75-85% indica necessidade de recalibração antes de qualquer uso operacional.
</div>
<p>Baselines ancorados em observação não usam as covariáveis de choque, portanto
não exercem o teste de monotonicidade — isso é registrado, não escondido.</p>

<h2>6. Limitações e fronteira clínica</h2>
<ul>
<li>{_DISCLAIMER}; uso exclusivo em pesquisa e planejamento agregado.</li>
<li>Resultados referem-se ao ano de desenvolvimento/auditoria ({eval_year}); a
generalização é testada apenas no Marco M2.</li>
<li>Tabelas por região suprimem células com menos de 5 semanas (LGPD).</li>
<li>Custo de amostragem bayesiana é reportado por modelo; candidatos lentos não são
removidos do relatório.</li>
</ul>

{manifest_block}

<footer>
Relatório gerado a partir dos artefatos do run <strong>{html.escape(run_dir.name)}</strong>.
Pesquisa aberta — {_DISCLAIMER} | LGPD n &lt; 5.
</footer>
</main></body></html>"""

    html_path = run_dir / "relatorio_benchmark_leitos.html"
    html_path.write_text(body, encoding="utf-8")
    pdf_path = run_dir / pdf_name
    if not _render_pdf_via_chromium(html_path, pdf_path):
        raise RuntimeError(
            "Chromium indisponível para gerar o PDF editorial. "
            "Instale chromium/google-chrome ou use o HTML gerado."
        )
    return html_path, pdf_path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mirror_to_downloads(pdf_path: Path, *, downloads_dir: Path | None = None) -> Path:
    """Espelha o PDF em `~/Downloads` e confere o SHA-256 do espelho."""
    target_dir = downloads_dir or (Path.home() / "Downloads")
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / pdf_path.name
    shutil.copy2(pdf_path, destination)
    if sha256_of(destination) != sha256_of(pdf_path):
        raise RuntimeError("Espelho em ~/Downloads corrompido (SHA-256 divergente).")
    return destination
