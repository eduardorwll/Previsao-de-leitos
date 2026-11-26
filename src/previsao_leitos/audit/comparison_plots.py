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
