"""Relatório comparativo do benchmark: HTML5/CSS3 editorial e PDF via Chromium.

Consome apenas artefatos de um run (`benchmark.csv`, `predictions.parquet`,
`comparison.csv`, `decisions.csv`) e as figuras já geradas. Nenhum Markdown cru
entra no PDF; a fronteira clínica aparece no cabeçalho e no rodapé.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import contextlib
import hashlib
import html
import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl

from previsao_leitos.audit.categorization import (
    categorize,
    categorize_row,
    normalize_benchmark,
)
from previsao_leitos.audit.comparison_plots import (
    plot_calibration,
    plot_observed_history,
    plot_observed_vs_predicted,
    plot_pct_error,
    plot_region_grid,
    plot_residuals_by_region,
)
from previsao_leitos.audit.dossier_compiler import _render_pdf_via_chromium
from previsao_leitos.audit.pending_items import render_pending_items_html
from previsao_leitos.demand.modeling import TARGET_SPECS

_DISCLAIMER = "Not for patient care or live clinical use"

__all__ = [
    "_CSS",
    "_figure",
    "_render_pdf_via_chromium",
    "_table",
    "build_figures",
    "mirror_to_downloads",
    "render_report",
    "sha256_of",
]

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
table{border-collapse:collapse;width:100%;margin:8px 0 12px;font-size:7.2pt;table-layout:fixed}
th,td{border-bottom:1px solid var(--line);padding:3px 4px;text-align:right;overflow-wrap:anywhere}
th:first-child,td:first-child{text-align:left;width:24%;overflow-wrap:normal}
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


_HEADER_LABELS = {
    "candidate": "Modelo",
    "status": "Status",
    "mae_p50": "MAE P50",
    "mase_vs_pers": "MASE",
    "pinball_p95": "Pinball P95",
    "cobertura_p10_p90": "Cob. P10-90",
    "peak_timing_error_weeks": "Erro pico (sem.)",
    "monotonic_violations": "Viol. mono.",
    "champion": "Campeão",
    "decision": "Decisão",
    "summary": "Resumo",
}


def _table(df: pl.DataFrame, columns: list[str], *, highlight_column: str | None = None) -> str:
    headers = "".join(
        f"<th>{html.escape(_HEADER_LABELS.get(col, col))}</th>" for col in columns
    )
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


def _select_plot_models(metrics: pl.DataFrame, *, max_models: int = 2) -> list[str]:
    """Campeão registrado + melhor MASE (evita poluir os gráficos)."""
    ok = metrics.filter(pl.col("status") == "ok")
    if ok.height == 0:
        return []
    chosen: list[str] = []
    champion = ok.filter(pl.col("is_champion"))
    if champion.height:
        chosen.append(str(champion["candidate"][0]))
    for name in ok.sort("mase_vs_pers")["candidate"].to_list():
        if name not in chosen:
            chosen.append(str(name))
        if len(chosen) >= max_models:
            break
    return chosen


def build_figures(
    run_dir: Path,
    *,
    target_labels: dict[str, str],
) -> dict[str, dict[str, Path]]:
    """Regera todas as figuras a partir dos artefatos do run (sem re-treinar)."""
    predictions_path = run_dir / "predictions.parquet"
    metrics_path = run_dir / "benchmark.csv"
    if not predictions_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(
            f"Artefatos ausentes em {run_dir}: predictions.parquet e benchmark.csv são necessários."
        )
    predictions = pl.read_parquet(predictions_path)
    metrics = pl.read_csv(metrics_path)
    comparison_path = run_dir / "comparison.csv"
    comparison = pl.read_csv(comparison_path) if comparison_path.exists() else pl.DataFrame()

    manifest_path = run_dir / "run_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    panel = None
    panel_path = Path(str(manifest.get("panel_path", "")))
    if panel_path.is_file():
        panel = pl.read_parquet(panel_path)

    figures: dict[str, dict[str, Path]] = {}
    for target, label in target_labels.items():
        target_metrics = metrics.filter(pl.col("target") == target)
        target_predictions = predictions.filter(pl.col("target") == target)
        if target_metrics.height == 0 or target_predictions.height == 0:
            continue
        models = _select_plot_models(target_metrics)
        if not models:
            continue
        paths: dict[str, Path] = {}
        spec = TARGET_SPECS.get(target)
        if panel is not None and spec is not None:
            with contextlib.suppress(KeyError, ValueError):
                paths["history"] = plot_observed_history(
                    panel,
                    target_column=spec.column,
                    entity_col="nome_regiao_saude",
                    target_label=label,
                    eval_year=int(manifest.get("eval_year", 2024)),
                    output_path=run_dir / f"fig_history_{target}.png",
                )
        paths["observed"] = plot_observed_vs_predicted(
            target_predictions,
            models=models,
            output_path=run_dir / f"fig_observed_vs_predicted_{target}.png",
            target_label=label,
            eval_year=int(manifest.get("eval_year", 2024)),
        )
        paths["grid"] = plot_region_grid(
            target_predictions,
            models=models,
            output_path=run_dir / f"fig_regions_{target}.png",
            target_label=label,
            eval_year=int(manifest.get("eval_year", 2024)),
        )
        if comparison.height:
            target_comparison = comparison.filter(pl.col("target") == target)
            if target_comparison.height:
                paths["residuals"] = plot_residuals_by_region(
                    target_comparison,
                    models=models,
                    output_path=run_dir / f"fig_residuals_{target}.png",
                    target_label=label,
                )
        paths["calibration"] = plot_calibration(
            target_metrics,
            output_path=run_dir / f"fig_calibration_{target}.png",
            target_label=label,
        )
        paths["pct_error"] = plot_pct_error(
            target_predictions,
            models=models,
            output_path=run_dir / f"fig_pct_error_{target}.png",
            target_label=label,
        )
        figures[target] = paths
    return figures


def _target_sections(
    run_dir: Path, metrics: pl.DataFrame, target: str, label: str, *, eval_year: int
) -> str:
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
    ]
    parts = [
        f"<h3>{html.escape(label)}</h3>",
        "<p>Métricas no ano de avaliação, com treino até 2022 e calibração conformal em 2023. "
        "MASE &lt; 1 supera a persistência; cobertura ideal de P10-P90 é 80%.</p>",
        _table(
            subset.sort("mase_vs_pers", nulls_last=True),
            columns,
            highlight_column="is_champion",
        ),
    ]
    if ok.height:
        min_coverage = float(np.asarray(ok["cobertura_p10_p90"].min()).item())
        champion_rows = ok.filter(pl.col("is_champion"))
        champion_name = (
            str(champion_rows["candidate"][0]) if champion_rows.height else "—"
        )
        champion_coverage = (
            float(np.asarray(champion_rows["cobertura_p10_p90"][0]).item())
            if champion_rows.height
            else float("nan")
        )
        verdict = (
            "abaixo do piso nominal de 80% — recalibrar antes de uso"
            if min_coverage < 0.80
            else "acima do piso nominal de 80%"
        )
        parts.append(
            f'<p class="note">Campeão <strong>{html.escape(champion_name)}</strong>: cobertura '
            f"P10-P90 de <strong>{champion_coverage:.1%}</strong>. Pior caso do alvo: "
            f"<strong>{min_coverage:.1%}</strong> ({verdict}).</p>"
        )
        best = ok.sort("mase_vs_pers").head(1)["candidate"].to_list()[0]
        champion_rows = ok.filter(pl.col("is_champion"))
        champion = champion_rows["candidate"].to_list()[0] if champion_rows.height else best
        chosen = list(dict.fromkeys([champion, best]))[:2]
        parts.append(
            _figure(
                run_dir / f"fig_history_{target}.png",
                f"Contexto histórico do período: total regional observado por semana — {label}.",
            )
        )
        parts.append(
            _figure(
                run_dir / f"fig_observed_vs_predicted_{target}.png",
                f"Leitos observados vs. P50 previsto (soma regional por semana) — {label}.",
            )
        )
        parts.append(
            _figure(
                run_dir / f"fig_regions_{target}.png",
                f"Leitos observados vs. preditos, região por região — {label} ({eval_year}).",
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
    eligibility_path = run_dir / "eligibility.csv"
    eligibility = (
        pl.read_csv(eligibility_path) if eligibility_path.exists() else pl.DataFrame()
    )

    ok = metrics.filter(pl.col("status") == "ok")
    n_ok = ok.height
    n_total = metrics.height
    sections = "".join(
        _target_sections(run_dir, metrics, target, label, eval_year=eval_year)
        for target, label in target_labels.items()
    )

    decisions_block = ""
    if decisions.height:
        decisions_block = (
            "<h2>6. Catraca Champion-Challenger</h2>"
            "<p>Decisão quantílica contra o campeão registrado "
            "(MASE, Pinball P95, cobertura e timing de pico).</p>"
            + _table(
                decisions,
                ["candidate", "champion", "decision", "summary"],
            )
        )

    pct_block = ""
    categories_block = ""
    if "wmape" in metrics.columns:
        pct_parts: list[str] = [
            "<h2>4. Erro percentual vs. dados reais</h2>",
            "<p>Erro absoluto como fração do valor observado, com piso de 5% da média "
            "para não explodir em regiões pequenas. A barra do gráfico marca o p90 "
            "(pior caso típico).</p>",
        ]
        for target, label in target_labels.items():
            subset = metrics.filter(
                (pl.col("target") == target) & (pl.col("status") == "ok")
            ).sort("wmape")
            if subset.height == 0:
                continue
            pct_parts.append(f"<h3>{html.escape(label)}</h3>")
            pct_parts.append(
                _table(
                    subset,
                    [
                        "candidate",
                        "wmape",
                        "erro_pct_mediana",
                        "erro_pct_p90",
                        "subestimacao_pct",
                        "erro_pct_evento",
                    ],
                )
            )
            pct_parts.append(
                _figure(
                    run_dir / f"fig_pct_error_{target}.png",
                    f"Distribuição do erro percentual por modelo — {label}.",
                )
            )
        pct_block = "".join(pct_parts)

        grade_parts: list[str] = [
            "<h2>5. Categorias por quesito (rubrica v1)</h2>",
            "<p>Letras por critério; o selo é o <strong>pior</strong> entre acurácia, "
            "fase e calibração — erro de fase não se compensa com acurácia. Violação "
            "monotônica reprova automaticamente.</p>",
        ]
        for target, label in target_labels.items():
            subset = metrics.filter(pl.col("target") == target)
            if subset.height == 0:
                continue
            graded = categorize(
                normalize_benchmark(subset).with_columns(pl.lit(target).alias("target"))
            )
            grade_parts.append(f"<h3>{html.escape(label)}</h3>")
            grade_parts.append(
                _table(
                    graded,
                    [
                        "candidate",
                        "acuracia",
                        "fase",
                        "calibracao",
                        "robustez",
                        "consistencia",
                        "custo",
                        "selo",
                        "justificativa",
                    ],
                )
            )
        categories_block = "".join(grade_parts)

    eligibility_block = ""
    if eligibility.height and "eligible" in eligibility.columns:
        blocks: list[str] = [
            "<h2>7. Elegibilidade para uso</h2>",
            "<p>Um modelo é <strong>elegível</strong> quando atinge cobertura P10-P90 "
            "&ge; 80% e zero violações monotônicas. A catraca promove apenas com ganho de "
            "MASE; esta tabela responde outra pergunta: quem é confiável para uso.</p>",
        ]
        for target, label in target_labels.items():
            subset = eligibility.filter(pl.col("target") == target)
            if subset.height == 0:
                continue
            n_eligible = int(subset.filter(pl.col("eligible")).height)
            header = (
                f"<h3>{html.escape(label)}</h3>"
                f"<p>{n_eligible} de {subset.height} modelos elegíveis.</p>"
            )
            blocks.append(
                header
                + _table(
                    subset,
                    [
                        "candidate",
                        "mase_vs_pers",
                        "cobertura_p10_p90",
                        "monotonic_violations",
                        "eligible",
                    ],
                )
            )
    
    eligibility_block = "".join(blocks)

    manifest_block = ""
    if manifest:
        items = "".join(
            f"<li><strong>{html.escape(str(k))}</strong>: {_fmt(v)}</li>"
            for k, v in manifest.items()
            if not isinstance(v, (dict, list))
        )
        manifest_block = f"<h2>Reprodutibilidade</h2><ul>{items}</ul>"

    pending_items_block = render_pending_items_html()

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

<h2>3. Resultados por alvo</h2>
<p>Cada alvo tem sua própria unidade: UTI em leitos-dia por semana e clínico/respiratório
em pacientes-dia. Comparações de MASE são válidas dentro do alvo, nunca entre alvos.</p>

{sections}

{pct_block}

{categories_block}

{decisions_block}

{eligibility_block}

<h2>8. Calibração, monotonicidade e riscos</h2>
<div class="note">
A coluna <strong>monotonic_violations</strong> conta quedas de P50 quando o choque
hidrológico ex-ante aumenta (ADR-0001); o gate é zero. Cobertura fora da faixa
75-85% indica necessidade de recalibração antes de qualquer uso operacional.
</div>
<p>Baselines ancorados em observação não usam as covariáveis de choque, portanto
não exercem o teste de monotonicidade — isso é registrado, não escondido.</p>

<h2>9. Limitações e fronteira clínica</h2>
<ul>
<li>{_DISCLAIMER}; uso exclusivo em pesquisa e planejamento agregado.</li>
<li>Resultados referem-se ao ano de desenvolvimento/auditoria ({eval_year}); a
generalização é testada apenas no Marco M2.</li>
<li>Tabelas por região suprimem células com menos de 5 semanas (LGPD).</li>
<li>Custo de amostragem bayesiana é reportado por modelo; candidatos lentos não são
removidos do relatório.</li>
</ul>

<h2>10. Pontos pendentes para estudo posterior</h2>
<p>Acessos a dados e análises ainda em aberto, registrados para continuidade do projeto:</p>
{pending_items_block}

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


_METRIC_GLOSSARY = """**Como ler estas métricas**

- **wMAPE** — erro percentual ponderado: soma dos erros ÷ soma dos leitos observados.
  Responde *quanto do volume real se erra*; robusto a regiões pequenas; menor é melhor.
- **MASE** — erro do modelo ÷ erro da persistência (repetir a semana de origem).
  `1,00` = empatar com o ingênuo; `0,90` = 10% melhor. Abaixo de `1,00` é o mínimo.
- **Fase** — distância, em semanas, entre o pico previsto e o observado. Mede se o modelo
  avisa *quando* o pico chega, não só o tamanho. Alvo operacional: ≤ 2 semanas.
- **Cobertura P10-P90** — fração de semanas com o valor real dentro do intervalo previsto.
  Nominal 80%: muito acima = intervalo conservador; muito abaixo = confiança excessiva.
- **P90 do erro** — erro percentual superado em apenas 10% das semanas: o *pior caso típico*.
"""

_SEAL_ORDER = ("Reprovado", "D", "C", "B", "A", "N/A")


def _worst_seal(seals: list[str]) -> str:
    usable = [s for s in seals if s in _SEAL_ORDER]
    if not usable:
        return "N/A"
    return min(usable, key=_SEAL_ORDER.index)


def render_model_cards(run_dir: Path, *, target_labels: dict[str, str]) -> Path:
    """Gera um cartão por modelo e o índice consolidado (Markdown)."""
    metrics_path = run_dir / "benchmark.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"benchmark.csv ausente em {run_dir}")
    metrics = pl.read_csv(metrics_path)
    manifest_path = run_dir / "run_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    cards_dir = run_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    overview: list[str] = [
        f"# Cartões de modelo — `{run_dir.name}`",
        "",
        f"> Rubrica `v1` · run `{run_dir.name}` · painel "
        f"`{str(manifest.get('panel_sha256', '—'))[:12]}`",
        "",
        _METRIC_GLOSSARY,
        "",
        "| Modelo | Selo | Alvos | Justificativa |",
        "| :--- | :---: | :--- | :--- |",
    ]
    card_files: list[str] = []

    for candidate in sorted(metrics["candidate"].unique().to_list()):
        rows = metrics.filter(pl.col("candidate") == candidate).sort("target")
        normalized = normalize_benchmark(rows)
        graded_rows = [
            {**categorize_row(row), "target": target}
            for row, target in zip(normalized.to_dicts(), rows["target"].to_list(), strict=False)
        ]
        seals = [str(item["selo"]) for item in graded_rows]
        model_seal = _worst_seal(seals)
        weakest = next(
            (item["justificativa"] for item in graded_rows if item["selo"] == model_seal),
            "—",
        )
        overview.append(
            f"| [{candidate}](cards/{candidate}.md) | **{model_seal}** | "
            f"{', '.join(target_labels.get(t, t) for t in rows['target'].to_list())} | "
            f"{weakest} |"
        )

        lines: list[str] = [
            f"# {candidate}",
            "",
            f"**Selo geral:** `{model_seal}` · **Rubrica:** `v1` · "
            f"**Família:** `{rows['family'][0]}`",
            "",
            _METRIC_GLOSSARY,
            "",
            "## Resultados por alvo",
            "",
            "| Alvo | wMAPE | MASE | Fase (sem.) | Cobertura | P90 do erro | Subestimação |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows.to_dicts():
            label = target_labels.get(str(row["target"]), str(row["target"]))
            if row.get("status") != "ok":
                lines.append(f"| {label} | — | — | — | — | — | — |")
                continue
            lines.append(
                "| {label} | {wmape:.1f}% | {mase:.3f} | {peak:.1f} | {cov:.1%} | "
                "{p90:.1f}% | {under:.1%} |".format(
                    label=label,
                    wmape=float(row["wmape"]),
                    mase=float(row["mase_vs_pers"]),
                    peak=float(row["peak_timing_error_weeks"]),
                    cov=float(row["cobertura_p10_p90"]),
                    p90=float(row["erro_pct_p90"]),
                    under=float(row["subestimacao_pct"]),
                )
            )
        lines.extend(
            [
                "",
                "## Categorias",
                "",
                "| Alvo | Acurácia | Fase | Calibração | Robustez | Consistência | Custo | Selo |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for item in graded_rows:
            label = target_labels.get(str(item["target"]), str(item["target"]))
            lines.append(
                f"| {label} | {item['acuracia']} | {item['fase']} | {item['calibracao']} | "
                f"{item['robustez']} | {item['consistencia']} | {item['custo']} | "
                f"**{item['selo']}** |"
            )
        lines.extend(
            [
                "",
                "## Leitura",
                "",
                f"- Selo geral (pior caso entre alvos): **{model_seal}**.",
                f"- Ponto mais fraco: {weakest}.",
                "- wMAPE e MASE medem tamanho do erro; fase mede o *quando*; cobertura e P90 "
                "medem a confiança.",
                "",
                "## Reprodutibilidade",
                "",
                f"- Run: `{run_dir.name}`",
                f"- Git: `{manifest.get('git_sha', '—')}` · seed `{manifest.get('seed', '—')}`",
                f"- Painel SHA-256: `{manifest.get('panel_sha256', '—')}`",
                "",
                "> Not for patient care or live clinical use. LGPD: agregação e supressão n < 5.",
            ]
        )
        card_path = cards_dir / f"{candidate}.md"
        card_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        card_files.append(str(card_path.name))

    overview.append("")
    overview.append(f"_{len(card_files)} cartões gerados._")
    index_path = run_dir / "benchmark_cards.md"
    index_path.write_text("\n".join(overview) + "\n", encoding="utf-8")
    return index_path
