"""Gráficos contrafactuais acessíveis (paleta Okabe-Ito WCAG 2.2, 300 DPI)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import polars as pl

from previsao_leitos.audit.data_hygiene import DataHygieneScorecard
from previsao_leitos.audit.hypothesis_ledger import RatchetEvaluation
from previsao_leitos.readiness.simulator import CounterfactualReadinessComparison

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paleta Okabe-Ito amigável a daltonismo (WCAG 2.2)
_COLOR_REAL = "#000000"
_COLOR_CF = "#0072B2"
_COLOR_MODEL = "#D55E00"
_COLOR_BAND = "#E69F00"
_COLOR_POS = "#CC79A7"
_COLOR_NEG = "#009E73"


def plot_c1_counterfactual_trajectory(
    series_df: pl.DataFrame,
    *,
    entity_col: str,
    target_label: str,
    output_path: Path,
    week_range: tuple[int, int] = (10, 35),
    eval_year: int = 2024,
) -> Path:
    """Fig. C1: Observado vs. Contrafactual Sem Enchente (IC 95%) vs. Previsão Ex-Ante (P10-P90)."""
    sub = series_df.filter(pl.col("target_semana_epi").is_between(week_range[0], week_range[1]))
    entities = sorted(sub[entity_col].unique().to_list())[:4]
    n_panels = max(1, len(entities))

    fig, axes = plt.subplots(n_panels, 1, figsize=(10.5, 3.2 * n_panels), sharex=True)
    axes_list = [axes] if n_panels == 1 else list(axes)

    for ax, ent in zip(axes_list, entities, strict=True):
        edf = sub.filter(pl.col(entity_col) == ent).sort("target_semana_epi")
        wks = edf["target_semana_epi"].to_numpy()
        ax.fill_between(
            wks,
            edf["y_cf_p05"].to_numpy(),
            edf["y_cf_p95"].to_numpy(),
            color=_COLOR_CF,
            alpha=0.14,
            label="Contrafactual Sem Enchente (IC 95%)",
        )
        ax.plot(
            wks,
            edf["y_cf_no_flood"].to_numpy(),
            color=_COLOR_CF,
            linestyle="--",
            linewidth=1.8,
            label="Contrafactual Sem Enchente (Média)",
        )
        ax.fill_between(
            wks,
            edf["pred_p10"].to_numpy(),
            edf["pred_p90"].to_numpy(),
            color=_COLOR_BAND,
            alpha=0.22,
            label="Previsão Ex-Ante (Banda P10-P90)",
        )
        ax.plot(
            wks,
            edf["pred_p50"].to_numpy(),
            color=_COLOR_MODEL,
            linewidth=2.0,
            label="Previsão Ex-Ante (P50)",
        )
        ax.plot(
            wks,
            edf["target_y"].to_numpy(),
            color=_COLOR_REAL,
            linewidth=2.2,
            marker="o",
            markersize=3.2,
            label="Real Observado (SIH-SUS)",
        )
        ax.axvspan(18, 22, color="#999999", alpha=0.12)
        ax.set_title(f"{ent} — {target_label}", fontsize=10.5, fontweight="bold")
        ax.set_ylabel("Volume Semanal")
        ax.grid(alpha=0.25)

    axes_list[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes_list[-1].set_xlabel(f"Semana Epidemiológica ISO ({eval_year})")
    fig.suptitle(
        f"Fig. C1 — Trajetória Observada vs. Contrafactual Sem Enchente ({target_label})",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_c2_shock_excess_accuracy(
    series_df: pl.DataFrame,
    *,
    target_label: str,
    output_path: Path,
    shock_weeks: tuple[int, int] = (14, 32),
) -> Path:
    """Fig. C2: Excesso Real Observado vs. Excesso Previsto Ex-Ante (timing Onda 1 / Onda 2)."""
    agg = (
        series_df.filter(pl.col("target_semana_epi").is_between(shock_weeks[0], shock_weeks[1]))
        .group_by("target_semana_epi")
        .agg(
            pl.col("excess_obs").mean().alias("excesso_obs"),
            pl.col("excess_pred").mean().alias("excesso_pred"),
        )
        .sort("target_semana_epi")
    )
    wks = agg["target_semana_epi"].to_numpy()
    obs = agg["excesso_obs"].to_numpy()
    pred = agg["excesso_pred"].to_numpy()

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.axhline(0.0, color="#555555", linewidth=1.0, linestyle=":")
    ax.bar(
        wks - 0.18,
        obs,
        width=0.35,
        color=_COLOR_REAL,
        alpha=0.80,
        label="Excesso Real Observado (Real - Contrafactual)",
    )
    ax.bar(
        wks + 0.18,
        pred,
        width=0.35,
        color=_COLOR_MODEL,
        alpha=0.85,
        label="Excesso Previsto Ex-Ante (Real - Climatologia P50)",
    )
    ax.axvspan(17.5, 22.5, color="#999999", alpha=0.14, label="Pico Hidro + Onda 2 (Sem. 18-22)")
    ax.set_title(
        f"Fig. C2 — Acurácia de Excesso na Janela de Choque ({target_label})",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.set_xlabel("Semana Epidemiológica ISO")
    ax.set_ylabel("Excesso Semanal Médio nas Regiões Atingidas")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_c3_placebo_validation(
    placebo_df: pl.DataFrame,
    *,
    target_label: str,
    output_path: Path,
) -> Path:
    """Fig. C3: Validação Placebo em semanas sem enchente (auditoria contra falsos alarmes)."""
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    if not placebo_df.is_empty():
        p_obs = placebo_df["excess_obs"].to_numpy()
        p_pred = placebo_df["excess_pred"].to_numpy()
        ax.hist(
            p_obs,
            bins=15,
            alpha=0.55,
            color=_COLOR_CF,
            label="Resíduo Contrafactual Placebo (Real - Sem Enchente)",
        )
        ax.hist(
            p_pred,
            bins=15,
            alpha=0.60,
            color=_COLOR_MODEL,
            label="Excesso Previsto Placebo pelo Modelo",
        )
    ax.axvline(0.0, color=_COLOR_REAL, linestyle="--", linewidth=1.6, label="Zero Esperado")
    ax.set_title(
        f"Fig. C3 — Validação Placebo em Períodos Sem Enchente ({target_label})",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.set_xlabel("Desvio / Excesso Estimado em Semanas Sem Cheia")
    ax.set_ylabel("Frequência (Regiões x Semanas)")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_c4_intermunicipal_landfill_waterfall(
    decomposition_df: pl.DataFrame,
    *,
    target_label: str,
    output_path: Path,
) -> Path:
    """Fig. C4: Decomposição waterfall (local -> montante -> remanso/aterro -> W_ref)."""
    labels = decomposition_df["etapa"].to_list()
    deltas = decomposition_df["contribuicao_delta"].to_list()
    levels = decomposition_df["nivel_medio"].to_list()

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    colors = [
        _COLOR_CF if i == 0 else (_COLOR_POS if d >= 0.0 else _COLOR_NEG)
        for i, d in enumerate(deltas)
    ]
    vals = [levels[0], *deltas[1:]]
    bottoms = [0.0]
    running = levels[0]
    for d in deltas[1:]:
        bottoms.append(running if d >= 0.0 else running + d)
        running += d

    x = list(range(len(labels)))
    ax.bar(x, [abs(v) for v in vals], bottom=bottoms, color=colors, width=0.55, edgecolor="#222")
    for idx, (b, v) in enumerate(zip(bottoms, vals, strict=True)):
        txt = f"{v:.1f}" if idx == 0 else f"{v:+.2f}"
        ax.text(idx, b + abs(v) + 0.02 * max(levels), txt, ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.margins(y=0.14)
    ax.set_ylabel("Previsão Média P50 / Contribuição Incremental")
    ax.set_title(
        f"Fig. C4 — Decomposição Contrafactual Intermunicipal e Áreas Aterradas ({target_label})",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_c5_simpy_contingency(
    comparison: CounterfactualReadinessComparison,
    *,
    target_label: str,
    output_path: Path,
) -> Path:
    """Fig. C5: Contrafactual de filas SimPy — Acionamento Antecipado (h=2) vs. Reativo (h=0)."""
    reac = comparison.reactive_h0.daily_df
    proa = comparison.proactive_h2.daily_df
    days = reac["dia"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.0, 6.4), sharex=True)

    ax1.plot(
        days,
        reac["fila_espera_pacientes"].to_numpy(),
        color=_COLOR_MODEL,
        linewidth=2.2,
        label=f"Reativo h=0 (pico={comparison.reactive_h0.peak_queue_patients} pac.)",
    )
    ax1.plot(
        days,
        proa["fila_espera_pacientes"].to_numpy(),
        color=_COLOR_NEG,
        linewidth=2.2,
        linestyle="--",
        label=f"Proativo h=2 sem. (pico={comparison.proactive_h2.peak_queue_patients} pac.)",
    )
    ax1.fill_between(
        days,
        proa["fila_espera_pacientes"].to_numpy(),
        reac["fila_espera_pacientes"].to_numpy(),
        where=reac["fila_espera_pacientes"].to_numpy() >= proa["fila_espera_pacientes"].to_numpy(),
        color=_COLOR_NEG,
        alpha=0.14,
        label=f"Redução de pico: {comparison.queue_peak_reduction_pct:.1f}%",
    )
    ax1.set_ylabel("Fila de Espera (Pacientes)")
    ax1.set_title(
        f"Fig. C5 — Simulação Contrafactual de Contingência SimPy ({target_label})",
        fontsize=11.5,
        fontweight="bold",
    )
    ax1.legend(fontsize=8.5, loc="upper right")
    ax1.grid(alpha=0.25)

    ax2.plot(
        days,
        reac["ocupacao_pct"].to_numpy(),
        color=_COLOR_MODEL,
        linewidth=1.8,
        label=f"Ocupação Reativo h=0 ({comparison.reactive_h0.overload_days_gt95} dias >=95%)",
    )
    ax2.plot(
        days,
        proa["ocupacao_pct"].to_numpy(),
        color=_COLOR_NEG,
        linewidth=1.8,
        linestyle="--",
        label=f"Ocupação Proativo h=2 ({comparison.proactive_h2.overload_days_gt95} dias >=95%)",
    )
    ax2.axhline(95.0, color=_COLOR_REAL, linestyle=":", linewidth=1.2, label="Limiar 95%")
    ax2.set_ylabel("Ocupação Efetiva (%)")
    ax2.set_xlabel("Dia da Janela Simulada")
    ax2.legend(fontsize=8.5, loc="lower right")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_c6_hygiene_scorecard(
    scorecard: DataHygieneScorecard,
    *,
    output_path: Path,
) -> Path:
    """Fig. C6: Painel visual de auditoria dos gates de higiene de dados (G1-G8)."""
    gates = scorecard.gates
    labels = [f"{g.gate_id}: {g.name}" for g in gates]
    status_score = {"PASS": 1.0, "WARN": 0.5, "FAIL": 0.0}
    status_color = {"PASS": _COLOR_NEG, "WARN": _COLOR_BAND, "FAIL": _COLOR_MODEL}

    y_pos = list(range(len(gates)))
    vals = [status_score[g.status] for g in gates]
    colors = [status_color[g.status] for g in gates]

    fig, ax = plt.subplots(figsize=(10.2, max(3.6, 0.55 * len(gates) + 1.4)))
    ax.barh(y_pos, [1.0] * len(gates), color="#EEEEEE", edgecolor="#CCCCCC", height=0.6)
    ax.barh(y_pos, vals, color=colors, edgecolor="#222222", height=0.6)

    for idx, gate in enumerate(gates):
        txt = (
            f"[{gate.status}] obs={gate.observed_value:.3g} {gate.unit} "
            f"(lim <= {gate.threshold:.3g})"
        )
        ax.text(1.03, idx, txt, va="center", ha="left", fontsize=8.5, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.85)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["FAIL", "WARN", "PASS"])
    ax.set_title(
        f"Fig. C6 — Scorecard de Higiene de Dados Cruzados (Status: {scorecard.overall_status})",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_c7_ablation_delta(
    evaluation: RatchetEvaluation,
    *,
    output_path: Path,
) -> Path:
    """Fig. C7: Delta percentual Champion vs. Challenger na catraca científica."""
    metrics = list(evaluation.ablation_delta.keys())
    deltas_pct = [evaluation.ablation_delta[m]["delta_pct"] for m in metrics]
    higher_is_better = {"coverage_p90", "interior_specificity"}
    colors = [
        _COLOR_NEG
        if ((m in higher_is_better and d >= 0.0) or (m not in higher_is_better and d <= 0.0))
        else _COLOR_MODEL
        for m, d in zip(metrics, deltas_pct, strict=True)
    ]

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    y_pos = list(range(len(metrics)))
    ax.axvline(0.0, color=_COLOR_REAL, linestyle="--", linewidth=1.2)
    ax.barh(y_pos, deltas_pct, color=colors, edgecolor="#222222", height=0.55)
    for idx, val in enumerate(deltas_pct):
        ha = "left" if val >= 0.0 else "right"
        offset = 0.8 if val >= 0.0 else -0.8
        ax.text(val + offset, idx, f"{val:+.2f}%", va="center", ha=ha, fontsize=8.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Variação Percentual Challenger vs. Champion (%)")
    ax.set_title(
        f"Fig. C7 — Ablação Champion-Challenger ({evaluation.decision}: "
        f"{evaluation.challenger_run_id} vs. {evaluation.champion_run_id})",
        fontsize=11.0,
        fontweight="bold",
    )
    ax.margins(x=0.18)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
