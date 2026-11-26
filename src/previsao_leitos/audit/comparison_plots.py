"""Figuras comparativas: leitos observados vs. preditos e diagnóstico por modelo.

Paleta Okabe-Ito (WCAG 2.2), 300 DPI, sem dependência de seaborn. Cada figura é
escrita de forma atômica para não deixar PNG truncado em caso de falha.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

_COLOR_REAL = "#000000"
_COLOR_CHAMPION = "#D55E00"
_COLOR_A = "#0072B2"
_COLOR_B = "#009E73"
_COLOR_C = "#CC79A7"
_PALETTE = (_COLOR_CHAMPION, _COLOR_A, _COLOR_B, _COLOR_C)


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


def _weekly_totals(predictions: pl.DataFrame, model: str) -> pl.DataFrame:
    return (
        predictions.filter(pl.col("model") == model)
        .group_by(["target_year", "target_week"])
        .agg(
            pl.col("observed").sum().alias("observed"),
            pl.col("pred_p50").sum().alias("pred_p50"),
            pl.col("pred_p10").sum().alias("pred_p10"),
            pl.col("pred_p90").sum().alias("pred_p90"),
        )
        .sort(["target_year", "target_week"])
    )


def plot_observed_vs_predicted(
    predictions: pl.DataFrame,
    *,
    models: Sequence[str],
    output_path: Path,
    target_label: str,
    eval_year: int,
) -> Path:
    """Soma regional por semana: observado vs. P50 dos modelos (com banda P10-P90 do 1º)."""
    if predictions.height == 0 or not models:
        raise ValueError("plot_observed_vs_predicted exige previsões e ao menos um modelo.")
    fig, ax = plt.subplots(figsize=(11, 4.6))
    reference = _weekly_totals(predictions, models[0])
    x = np.arange(reference.height)
    ax.fill_between(
        x,
        reference["pred_p10"].to_numpy(),
        reference["pred_p90"].to_numpy(),
        color=_COLOR_CHAMPION,
        alpha=0.16,
        label=f"P10-P90 ({models[0]})",
    )
    ax.plot(
        x,
        reference["observed"].to_numpy(),
        color=_COLOR_REAL,
        linewidth=2.0,
        label="Observado",
    )
    for index, model in enumerate(models):
        totals = _weekly_totals(predictions, model)
        n = min(totals.height, reference.height)
        ax.plot(
            x[:n],
            totals["pred_p50"].to_numpy()[:n],
            color=_PALETTE[index % len(_PALETTE)],
            linewidth=1.7,
            linestyle="-" if index == 0 else "--",
            label=f"P50 ({model})",
        )
    ax.set_title(
        f"{target_label} — leitos ocupados/dia observados vs. preditos (RS, {eval_year})",
        fontsize=11,
        loc="left",
    )
    ax.set_xlabel("Semanas do ano de avaliação")
    ax.set_ylabel("Total regional (unidade do alvo)")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.text(
        0.01,
        0.01,
        "Pesquisa aberta — Not for patient care or live clinical use | LGPD n < 5",
        fontsize=7,
        color="#475569",
    )
    return _atomic_save(fig, output_path)


def plot_residuals_by_region(
    comparison: pl.DataFrame,
    *,
    models: Sequence[str],
    output_path: Path,
    target_label: str,
) -> Path:
    """Viés médio (predito - observado) por região, para os modelos escolhidos."""
    frame = comparison.filter(pl.col("model").is_in(list(models)))
    if frame.height == 0:
        raise ValueError("Sem linhas de comparação para os modelos pedidos.")
    regions = sorted(frame["entity"].unique().to_list())
    positions = np.arange(len(regions))
    width = 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(regions)), 4.2))
    for index, model in enumerate(models):
        subset = frame.filter(pl.col("model") == model)
        bias_map = dict(zip(subset["entity"].to_list(), subset["bias"].to_list(), strict=False))
        values = [float(bias_map.get(region, 0.0)) for region in regions]
        ax.bar(
            positions + index * width,
            values,
            width=width,
            color=_PALETTE[index % len(_PALETTE)],
            label=model,
        )
    ax.axhline(0.0, color=_COLOR_REAL, linewidth=1.0)
    ax.set_xticks(positions + width * (len(models) - 1) / 2)
    ax.set_xticklabels(regions, rotation=75, ha="right", fontsize=6)
    ax.set_ylabel("Viés médio (predito - observado)")
    ax.set_title(f"{target_label} — viés por região de saúde", fontsize=11, loc="left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.text(
        0.01,
        0.01,
        "Pesquisa aberta — Not for patient care or live clinical use | LGPD n < 5",
        fontsize=7,
        color="#475569",
    )
    return _atomic_save(fig, output_path)


def plot_calibration(
    metrics: pl.DataFrame,
    *,
    output_path: Path,
    target_label: str,
) -> Path:
    """Cobertura empírica P10-P90 e Pinball P95 por candidato."""
    ok = metrics.filter(pl.col("status") == "ok").sort("cobertura_p10_p90")
    if ok.height == 0:
        raise ValueError("Sem candidatos concluídos para a figura de calibração.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 0.45 * ok.height + 2.2), sharey=True)
    names = ok["candidate"].to_list()
    positions = np.arange(len(names))
    axes[0].barh(positions, ok["cobertura_p10_p90"].to_numpy(), color=_COLOR_A)
    axes[0].axvline(0.80, color=_COLOR_CHAMPION, linestyle="--", linewidth=1.2)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_title("Cobertura empírica P10-P90", fontsize=10, loc="left")
    axes[0].grid(axis="x", alpha=0.25, linewidth=0.6)

    axes[1].barh(positions, ok["pinball_p95"].to_numpy(), color=_COLOR_B)
    axes[1].set_title("Pinball P95 (menor é melhor)", fontsize=10, loc="left")
    axes[1].grid(axis="x", alpha=0.25, linewidth=0.6)

    fig.suptitle(f"{target_label} — calibração e cauda por modelo", fontsize=11, x=0.01, ha="left")
    fig.text(
        0.01,
        0.01,
        "Pesquisa aberta — Not for patient care or live clinical use | LGPD n < 5",
        fontsize=7,
        color="#475569",
    )
    return _atomic_save(fig, output_path)


def plot_region_grid(
    predictions: pl.DataFrame,
    *,
    models: Sequence[str],
    output_path: Path,
    target_label: str,
    eval_year: int,
    max_regions: int = 30,
) -> Path:
    """Small multiples: série real vs. P50 dos modelos, uma região por painel."""
    if predictions.height == 0 or not models:
        raise ValueError("plot_region_grid exige previsões e ao menos um modelo.")
    champion = models[0]
    regions = sorted(predictions["entity"].unique().to_list())[:max_regions]
    n_cols = 5
    n_rows = max(1, (len(regions) + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 2.15 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()
    palette = (_COLOR_CHAMPION, _COLOR_A, _COLOR_B)

    for ax, region in zip(axes_flat, regions, strict=False):
        subset = predictions.filter(pl.col("entity") == region)
        reference = subset.filter(pl.col("model") == champion).sort("target_week")
        weeks = reference["target_week"].to_numpy()
        ax.plot(weeks, reference["observed"].to_numpy(), color=_COLOR_REAL, linewidth=1.6)
        ax.fill_between(
            weeks,
            reference["pred_p10"].to_numpy(),
            reference["pred_p90"].to_numpy(),
            color=_COLOR_CHAMPION,
            alpha=0.16,
            linewidth=0,
        )
        for index, model in enumerate(models):
            rows = subset.filter(pl.col("model") == model).sort("target_week")
            ax.plot(
                rows["target_week"].to_numpy(),
                rows["pred_p50"].to_numpy(),
                color=palette[index % len(palette)],
                linewidth=1.2,
                linestyle="-" if index == 0 else "--",
            )
        ax.set_title(region[:24], fontsize=6.2, pad=2)
        ax.tick_params(labelsize=5.2, length=2)
        ax.grid(alpha=0.2, linewidth=0.4)

    for ax in axes_flat[len(regions) :]:
        ax.axis("off")

    handles: list[Any] = [
        Line2D([], [], color=_COLOR_REAL, linewidth=1.6, label="Observado"),
        Line2D(
            [], [], color=palette[0], linewidth=1.2, label=f"P50 {champion}"
        ),
    ]
    if len(models) > 1:
        handles.append(
            Line2D(
                [], [], color=palette[1], linewidth=1.2, linestyle="--", label=f"P50 {models[1]}"
            )
        )
    handles.append(
        Rectangle((0, 0), 1, 1, color=_COLOR_CHAMPION, alpha=0.16, label="P10-P90")
    )
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle(
        f"{target_label} — leitos observados vs. preditos por região de saúde ({eval_year})",
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.text(
        0.01,
        0.005,
        "Pesquisa aberta — Not for patient care or live clinical use | LGPD n < 5",
        fontsize=6.5,
        color="#475569",
    )
    return _atomic_save(fig, output_path)


def plot_observed_history(
    panel: pl.DataFrame,
    *,
    target_column: str,
    entity_col: str,
    target_label: str,
    eval_year: int,
    output_path: Path,
) -> Path:
    """Contexto histórico: total regional observado por semana, com o ano de avaliação sombreado."""
    frame = (
        panel.select(
            [
                pl.col("ano_epi"),
                pl.col("semana_epi"),
                pl.col(target_column).alias("value"),
            ]
        )
        .group_by(["ano_epi", "semana_epi"])
        .agg(pl.col("value").sum())
        .sort(["ano_epi", "semana_epi"])
    )
    if frame.height == 0:
        raise ValueError("plot_observed_history sem dados.")
    x = np.arange(frame.height)
    years = frame["ano_epi"].to_numpy()
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(x, frame["value"].to_numpy(), color=_COLOR_REAL, linewidth=1.1)
    eval_mask = years == eval_year
    if eval_mask.any():
        ax.axvspan(
            float(x[eval_mask][0]),
            float(x[eval_mask][-1]),
            color=_COLOR_CHAMPION,
            alpha=0.14,
            linewidth=0,
            label=f"Ano de avaliação ({eval_year})",
        )
    tick_positions = [int(np.argmax(years == y)) for y in sorted(set(years.tolist()))]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(y) for y in sorted(set(years.tolist()))], fontsize=7, rotation=60)
    ax.set_ylabel("Total regional por semana")
    ax.set_title(
        f"{target_label} — série observada completa (contexto do período avaliado)",
        fontsize=11,
        loc="left",
    )
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.text(
        0.01,
        0.01,
        f"Pesquisa aberta — Not for patient care or live clinical use | LGPD n < 5 · {entity_col}",
        fontsize=7,
        color="#475569",
    )
    return _atomic_save(fig, output_path)


def plot_pct_error(
    predictions: pl.DataFrame,
    *,
    models: Sequence[str],
    output_path: Path,
    target_label: str,
    floor_fraction: float = 0.05,
) -> Path:
    """Distribuição do erro percentual por modelo (caixa = quartis, ponto = p90)."""
    frame = predictions.filter(pl.col("model").is_in(list(models)))
    if frame.height == 0:
        raise ValueError("plot_pct_error sem previsões para os modelos pedidos.")
    fig, ax = plt.subplots(figsize=(8.6, 0.55 * len(models) + 1.8))
    data: list[np.ndarray] = []
    labels: list[str] = []
    for model in models:
        rows = frame.filter(pl.col("model") == model)
        observed = rows["observed"].to_numpy().astype(np.float64)
        predicted = rows["pred_p50"].to_numpy().astype(np.float64)
        floor = max(floor_fraction * float(np.mean(np.abs(observed))), 1e-9)
        err = 100.0 * np.abs(predicted - observed) / np.maximum(np.abs(observed), floor)
        data.append(err)
        labels.append(model)
    box = ax.boxplot(
        data,
        vert=False,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111827", "linewidth": 1.4},
    )
    for patch, colour in zip(box["boxes"], _PALETTE, strict=False):
        patch.set_facecolor(colour)
        patch.set_alpha(0.45)
        patch.set_edgecolor("#475569")
    for index, err in enumerate(data, start=1):
        ax.scatter(
            [float(np.percentile(err, 90))],
            [index],
            marker="|",
            s=160,
            color=_COLOR_CHAMPION,
            zorder=4,
        )
    ax.set_yticks(range(1, len(labels) + 1))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Erro percentual vs. observado (%) — piso de 5% da média")
    ax.set_title(
        f"{target_label} — distribuição do erro percentual (barra = p90)",
        fontsize=11,
        loc="left",
    )
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    fig.text(
        0.01,
        0.01,
        "Pesquisa aberta — Not for patient care or live clinical use | LGPD n < 5",
        fontsize=7,
        color="#475569",
    )
    return _atomic_save(fig, output_path)
