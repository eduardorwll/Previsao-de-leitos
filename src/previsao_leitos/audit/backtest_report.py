"""Relatório do backtest de origem móvel: figuras, HTML e PDF.

Reutiliza o CSS editorial e o renderizador Chromium do relatório de benchmark,
para manter uma única identidade visual entre os dois artefatos.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import matplotlib
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from previsao_leitos.audit import benchmark_report as _benchmark_report

_COLOR_MAIN = "#D55E00"
_COLOR_A = "#0072B2"
_COLOR_B = "#009E73"
_COLOR_MUTED = "#475569"
_PALETTE = ("#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")


def _atomic_save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{path.suffix}")
    fig.savefig(
        tmp,
        format=path.suffix.lstrip("."),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    tmp.replace(path)
    return path


def plot_backtest_ranking(
    fold_metrics: pl.DataFrame,
    summary: pl.DataFrame,
    *,
    output_path: Path,
    target_label: str,
) -> Path:
    """MASE macro por fold, destacando os modelos do topo do ranking mediano."""
    top = summary.head(5)["candidate"].to_list()
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for index, candidate in enumerate(top):
        rows = fold_metrics.filter(
            (pl.col("candidate") == candidate) & (pl.col("status") == "ok")
        ).sort("eval_year")
        ax.plot(
            rows["eval_year"].to_numpy(),
            rows["mase_macro"].to_numpy(),
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            color=_PALETTE[index % len(_PALETTE)],
            label=candidate,
        )
        median = float(summary.filter(pl.col("candidate") == candidate)["mase_median"][0])
        ax.axhline(median, color=_PALETTE[index % len(_PALETTE)], linewidth=0.7, alpha=0.35)
    ax.axhline(1.0, color=_COLOR_MUTED, linewidth=1.0, linestyle=":")
    ax.set_xlabel("Ano de avaliação (fold)")
    ax.set_ylabel("MASE macro vs. persistência")
    ax.set_title(
        f"{target_label} — MASE macro por fold (linha tracejada = mediana do modelo)",
        fontsize=11,
        loc="left",
    )
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.text(
        0.01,
        0.01,
        "Pesquisa aberta — Not for patient care or live clinical use | LGPD n < 5",
        fontsize=7,
        color=_COLOR_MUTED,
    )
    return _atomic_save(fig, output_path)


def plot_error_decomposition(
    summary: pl.DataFrame,
    *,
    output_path: Path,
    target_label: str,
) -> Path:
    """MASE mediano vs. erro de pico mediano: acurácia de nível não é acurácia de fase."""
    top = summary.sort("mase_median").head(8)
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    x = top["peak_abs_median"].to_numpy()
    y = top["mase_median"].to_numpy()
    ax.scatter(x, y, s=48, color=_COLOR_A, zorder=3)
    for name, xi, yi in zip(top["candidate"].to_list(), x, y, strict=True):
        ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.axhline(1.0, color=_COLOR_MUTED, linestyle=":", linewidth=1.0)
    ax.set_xlabel("Erro absoluto mediano do pico (semanas)")
    ax.set_ylabel("MASE mediano entre folds")
    ax.set_title(
        f"{target_label} — nível vs. fase: onde cada modelo erra",
        fontsize=11,
        loc="left",
    )
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.text(
        0.01,
        0.01,
        "Pesquisa aberta — Not for patient care or live clinical use | LGPD n < 5",
        fontsize=7,
        color=_COLOR_MUTED,
    )
    return _atomic_save(fig, output_path)


def render_backtest_report(
    run_dir: Path,
    *,
    target_label: str,
    eval_years: tuple[int, ...],
    pdf_name: str = "relatorio_backtest.pdf",
) -> tuple[Path, Path]:
    """Gera HTML e PDF de uma página com o diagnóstico do backtest."""
    summary = pl.read_csv(run_dir / "backtest_summary.csv")

    columns = [
        "candidate",
        "mase_median",
        "mase_worst",
        "rank_median",
        "fold_wins",
        "coverage_median",
        "peak_abs_median",
        "monotonic_violations",
    ]
    top = summary.sort("mase_median").head(8)
    best = summary.sort("mase_median").head(1).to_dicts()[0]
    worst_fold = float(best["mase_worst"])
    coverage = float(best["coverage_median"])
    peak = float(best["peak_abs_median"])

    body = f"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="utf-8"/>
<title>Backtest — {target_label}</title><style>{_benchmark_report._CSS}</style></head>
<body><main>
<h1>Régua de Backtest — {target_label}</h1>
<div class="meta">
<span class="badge">RS</span>
<span class="badge">folds {min(eval_years)}-{max(eval_years)}</span>
<span class="badge">ranking por mediana</span><br/>
Origem móvel com treino expansivo até <em>y-2</em>, calibração em <em>y-1</em>
e avaliação em <em>y</em>.
</div>

<h2>1. Por que trocar a régua</h2>
<p>Julgar modelos por um único ano confunde sorte com mérito. Com enchente sendo evento raro,
um fold favorável pode eleger um modelo pior. A régua passa a exigir consistência: mediana do
MASE macro entre folds, pior fold explícito e número de vitórias.</p>

<h2>2. Ranking</h2>
<p>Modelo líder: <strong>{best["candidate"]}</strong> — MASE mediano
<strong>{float(best["mase_median"]):.3f}</strong>,
pior fold <strong>{worst_fold:.3f}</strong>, cobertura mediana <strong>{coverage:.1%}</strong>,
erro de pico mediano <strong>{peak:.1f} semanas</strong>.</p>
{_benchmark_report._table(top, columns)}

<h2>3. Estabilidade entre folds</h2>
{_benchmark_report._figure(
    run_dir / "fig_backtest_mase.png",
    "MASE macro por fold; a linha fina é a mediana de cada modelo.",
)}

<h2>4. Nível vs. fase</h2>
<p>Um modelo pode acertar o nível médio e errar a semana do pico — o que é inútil para planejar
leitos. O quadrante superior direito deste gráfico é o pior cenário para a operação.</p>
{_benchmark_report._figure(
    run_dir / "fig_error_decomposition.png",
    "MASE mediano vs. erro absoluto do pico (semanas).",
)}

<h2>5. Leitura e próximos passos</h2>
<ul>
<li>Cobertura mediana de <strong>{coverage:.1%}</strong> — o piso nominal é 80%.</li>
<li>Erro de pico de <strong>{peak:.1f} semanas</strong> — a meta da fase
seguinte é ≤ 2 semanas.</li>
<li>O ranking desta régua substitui o julgamento por 2024 nas decisões
de promoção.</li>
</ul>

<footer>Pesquisa aberta — Not for patient care or live clinical use | LGPD n &lt; 5.
Gerado a partir de <strong>{run_dir.name}</strong>.</footer>
</main></body></html>"""

    html_path = run_dir / "relatorio_backtest.html"
    html_path.write_text(body, encoding="utf-8")
    pdf_path = run_dir / pdf_name
    if not _benchmark_report._render_pdf_via_chromium(html_path, pdf_path):
        raise RuntimeError("Chromium indisponível para gerar o PDF do backtest.")
    return html_path, pdf_path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "plot_backtest_ranking",
    "plot_error_decomposition",
    "render_backtest_report",
    "sha256_of",
]
