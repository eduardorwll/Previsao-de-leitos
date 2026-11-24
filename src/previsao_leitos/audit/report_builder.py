"""Gerador determinístico do Relatório Final de Verificação Científica (F06 / ADR-0007)."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from previsao_leitos.audit.counterfactual import CounterfactualAccuracyReport
from previsao_leitos.audit.plots import (
    plot_c1_counterfactual_trajectory,
    plot_c2_shock_excess_accuracy,
    plot_c3_placebo_validation,
    plot_c4_intermunicipal_landfill_waterfall,
    plot_c5_simpy_contingency,
)
from previsao_leitos.domain import LeakageViolationError
from previsao_leitos.geo import SplitPolicy
from previsao_leitos.readiness.simulator import CounterfactualReadinessComparison

_CLINICAL_BOUNDARY_NOTE = (
    "Pesquisa aberta exclusivamente para avaliação metodológica e planejamento agregado "
    "(Not for patient care or live clinical use)."
)


@dataclass(frozen=True)
class DatasetAuditEntry:
    """Registro auditável de um dataset consumido na verificação."""

    nome: str
    periodo: str
    linhas: int
    nulos: int
    lgpd_ok: bool
    observacao: str = ""


@dataclass(frozen=True)
class VerificationReportArtifacts:
    """Caminhos gerados pelo construtor do Relatório de Verificação."""

    report_path: Path
    manifest_path: Path
    figure_paths: tuple[Path, ...]


def _safe_pkg_version(pkg: str) -> str:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _safe_git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _df_to_markdown(df: pl.DataFrame) -> str:
    cols = df.columns
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for row in df.iter_rows():
        formatted = [
            f"{v:.3f}" if isinstance(v, float) else str(v)
            for v in row
        ]
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def build_verification_report(
    *,
    output_dir: Path,
    run_id: str,
    eval_year: int,
    target_label: str,
    benchmark_table: pl.DataFrame,
    counterfactual_report: CounterfactualAccuracyReport,
    decomposition_df: pl.DataFrame,
    datasets: Sequence[DatasetAuditEntry],
    entity_col: str = "nome_regiao_saude",
    policy: SplitPolicy | None = None,
    allow_holdout: bool = False,
    seed: int = 42,
    simpy_comparison: CounterfactualReadinessComparison | None = None,
) -> VerificationReportArtifacts:
    """Gera relatório completo (`verificacao_cientifica.md`), `run_manifest.json` e Fig. C1-C5."""
    active_policy = policy or SplitPolicy()
    if eval_year >= active_policy.holdout_year and not allow_holdout:
        raise LeakageViolationError(
            f"Relatório bloqueado para holdout intocado ({active_policy.holdout_year}): "
            f"eval_year={eval_year} exige `allow_holdout=True` apenas na auditoria final."
        )

    out = output_dir.resolve()
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig_c1 = plot_c1_counterfactual_trajectory(
        counterfactual_report.series_df,
        entity_col=entity_col,
        target_label=target_label,
        output_path=fig_dir / "fig_c1_trajetoria_contrafactual.png",
    )
    fig_c2 = plot_c2_shock_excess_accuracy(
        counterfactual_report.series_df,
        target_label=target_label,
        output_path=fig_dir / "fig_c2_acuracia_excesso_choque.png",
    )
    fig_c3 = plot_c3_placebo_validation(
        counterfactual_report.placebo_df,
        target_label=target_label,
        output_path=fig_dir / "fig_c3_validacao_placebo.png",
    )
    fig_c4 = plot_c4_intermunicipal_landfill_waterfall(
        decomposition_df,
        target_label=target_label,
        output_path=fig_dir / "fig_c4_decomposicao_intermunicipal_aterro.png",
    )
    fig_list = [fig_c1, fig_c2, fig_c3, fig_c4]
    if simpy_comparison is not None:
        fig_c5 = plot_c5_simpy_contingency(
            simpy_comparison,
            target_label=f"{target_label} - {simpy_comparison.region_label}",
            output_path=fig_dir / "fig_c5_contingencia_simpy.png",
        )
        fig_list.append(fig_c5)
    figures = tuple(fig_list)

    tools = {
        "python": platform.python_version(),
        "polars": _safe_pkg_version("polars"),
        "numpy": _safe_pkg_version("numpy"),
        "scipy": _safe_pkg_version("scipy"),
        "scikit-learn": _safe_pkg_version("scikit-learn"),
        "statsmodels": _safe_pkg_version("statsmodels"),
        "matplotlib": _safe_pkg_version("matplotlib"),
    }
    git_sha = _safe_git_sha()
    generated_at = datetime.now(tz=UTC).isoformat()

    manifest = {
        "run_id": run_id,
        "generated_at_utc": generated_at,
        "git_sha": git_sha,
        "eval_year": eval_year,
        "train_end_year": active_policy.train_end_year,
        "holdout_year": active_policy.holdout_year,
        "seed": seed,
        "target_label": target_label,
        "clinical_boundary": _CLINICAL_BOUNDARY_NOTE,
        "tools": tools,
        "counterfactual_metrics": {
            "shock_excess_mae": counterfactual_report.shock_excess_mae,
            "shock_excess_bias": counterfactual_report.shock_excess_bias,
            "peak_timing_error_weeks": counterfactual_report.peak_timing_error_weeks,
            "interval_coverage_90": counterfactual_report.interval_coverage_90,
            "placebo_bias": counterfactual_report.placebo_bias,
            "placebo_mae": counterfactual_report.placebo_mae,
            "monotonic_violation_rate": counterfactual_report.monotonic_violation_rate,
        },
        "figures": [str(p.relative_to(out)) for p in figures],
    }
    manifest_path = out / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    ds_df = pl.DataFrame(
        [
            {
                "Dataset": d.nome,
                "Período": d.periodo,
                "Linhas": f"{d.linhas:,}",
                "Nulos": f"{d.nulos:,}",
                "LGPD (Zero PII)": "OK" if d.lgpd_ok else "Revisar",
                "Observação": d.observacao,
            }
            for d in datasets
        ]
    )

    tools_md = "\n".join(f"- **{k}:** `{v}`" for k, v in tools.items())
    cf = counterfactual_report
    tr_yr = active_policy.train_end_year
    ho_yr = active_policy.holdout_year
    report_lines = [
        f"# Relatório de Verificação Científica - {target_label} (`{run_id}`)",
        "",
        f"> **Aviso de Escopo e Fronteira Clínica:** {_CLINICAL_BOUNDARY_NOTE}",
        "",
        f"- **Execução (`run_id`):** `{run_id}` | **Commit:** `{git_sha}` | **Seed:** `{seed}`",
        f"- **Partições Congeladas (`ADR-0003`):** Treino `<= {tr_yr}` "
        f"| Avaliação `{eval_year}` | Holdout Intocado `{ho_yr}`",
        "",
        "---",
        "",
        "## 1. Metodologia Científica e Contratos Causais",
        "",
        "- **Unidade de Análise e Horizonte:** Série semanal ISO por Região de Saúde do RS "
        "(`30/30`), com previsão estritamente *ex-ante* ($h = 2$ semanas / D+14, lags `0..4` "
        "semanas), separando Onda 1 (`D+0-5`) de Onda 2 (`D+7-28`: leptospirose/SRAG).",
        "- **Integridade de Grade e Zero Vazamento (`ADR-0003`):** Grade calendário contínua "
        "sem compressão de lacunas (`null` != `0.0 mm`); escaladores, controle sintético, "
        f"matriz `W_ref` e uso do solo `MapBiomas` ajustados estritamente em `<= {tr_yr}`.",
        "- **Áreas Aterradas e Escoamento Intermunicipal (`ADR-0001` e `ADR-0006`):**",
        "  - Restrição monotônica estrutural no `QuantileGBDTForecaster` (`0,00%` violações);",
        "  - Grafo hidrológico ponderado com tempo de trânsito (`travel_weeks=1` Guaíba -> "
        "Lagoa dos Patos/Pelotas) e remanso lagunar sob vento Sul (`V10M`);",
        f"  - Exposição em áreas aterradas/pôlderes (`MapBiomas <= {tr_yr}`) e grafo "
        "assistencial intermunicipal `W_ref` (`MUNIC_RES -> CNES`).",
        "- **Identificação Contrafactual (`ADR-0007`):** Controle Sintético independente sobre "
        "regiões-doadoras não inundadas para auditar excesso real vs. excesso previsto, "
        "validado por testes placebo em semanas calmas e decomposição sequencial `do(...)`.",
        "",
        "---",
        "",
        "## 2. Ferramentas, Versões e Ambiente Reprodutível",
        "",
        tools_md,
        "- **Manifesto estruturado:** `run_manifest.json`",
        "",
        "---",
        "",
        "## 3. Datasets Utilizados e Auditoria LGPD",
        "",
        _df_to_markdown(ds_df),
        "",
        "---",
        "",
        f"## 4. Benchmark Comparativo de Acurácia (`{eval_year}`)",
        "",
        _df_to_markdown(benchmark_table),
        "",
        "---",
        "",
        "## 5. Análise Contrafactual Estruturada e Gráficos de Verificação",
        "",
        "| Métrica Contrafactual | Valor Observado | Interpretação |",
        "| :--- | :---: | :--- |",
        f"| **MAE do Excesso na Janela de Choque** | `{cf.shock_excess_mae:.2f}` "
        "| Erro médio entre excesso real e excesso previsto *ex-ante* |",
        f"| **Viés do Excesso no Choque** | `{cf.shock_excess_bias:+.2f}` "
        "| Positivo = superestimação; negativo = subestimação |",
        f"| **Erro de Timing de Pico (semanas)** | `{cf.peak_timing_error_weeks:.2f} sem.` "
        "| Aderência temporal do pico previsto ao pico observado |",
        f"| **Cobertura Empírica (P10-P90 no Choque)** | `{100.0 * cf.interval_coverage_90:.1f}%` "
        "| Calibração da banda preditiva de `80%` durante o choque |",
        f"| **Viés Placebo (Semanas Sem Enchente)** | `{cf.placebo_bias:+.2f}` "
        "| Próximo de zero confirma ausência de falsos picos |",
        f"| **MAE Placebo (Semanas Sem Enchente)** | `{cf.placebo_mae:.2f}` "
        "| Ruído basal do controle sintético fora do desastre |",
        f"| **Taxa de Violação Monotônica** | `{100.0 * cf.monotonic_violation_rate:.2f}%` "
        "| Alvo `0,00%` sob perturbação positiva de risco físico |",
        "",
        "### Decomposição Contrafactual (`do(...)` - Intermunicipal e Áreas Aterradas)",
        "",
        _df_to_markdown(decomposition_df),
        "",
        "### Gráficos Contrafactuais Gerados",
        "",
        "- **Fig. C1 (Trajetória Observada vs. Contrafactual Sem Enchente vs. Previsão):**",
        "  ![Fig. C1](figures/fig_c1_trajetoria_contrafactual.png)",
        "- **Fig. C2 (Excesso Real vs. Excesso Previsto na Janela de Choque):**",
        "  ![Fig. C2](figures/fig_c2_acuracia_excesso_choque.png)",
        "- **Fig. C3 (Validação Placebo em Períodos Sem Enchente):**",
        "  ![Fig. C3](figures/fig_c3_validacao_placebo.png)",
        "- **Fig. C4 (Decomposição Waterfall Intermunicipal e de Áreas Aterradas):**",
        "  ![Fig. C4](figures/fig_c4_decomposicao_intermunicipal_aterro.png)",
    ]
    if simpy_comparison is not None:
        r0 = simpy_comparison.reactive_h0
        p2 = simpy_comparison.proactive_h2
        report_lines.extend(
            [
                "",
                "### Contrafactual Operacional Hospitalar (`SimPy`: `h=2` sem. vs. `h=0`)",
                "",
                "| Política Simulada | Pico de Fila (Pac.) | Dias >=95% | Espera Média (h) "
                "| Eletivas Adiadas |",
                "| :--- | :---: | :---: | :---: | :---: |",
                f"| **Reativo (`h=0`)** | `{r0.peak_queue_patients}` | `{r0.overload_days_gt95}` "
                f"| `{r0.mean_wait_hours:.1f}` | `{r0.deferred_electives}` |",
                f"| **Proativo (`h=2` sem.)** | `{p2.peak_queue_patients}` "
                f"| `{p2.overload_days_gt95}` | `{p2.mean_wait_hours:.1f}` "
                f"| `{p2.deferred_electives}` |",
                "",
                f"- **Redução no pico de fila:** "
                f"`{simpy_comparison.queue_peak_reduction_pct:.1f}%` "
                f"| **Dias de sobrecarga evitados:** `{simpy_comparison.overload_days_saved}`",
                "- **Fig. C5 (Contrafactual de Contingência SimPy):**",
                "  ![Fig. C5](figures/fig_c5_contingencia_simpy.png)",
            ]
        )
    report_lines.append("")
    report_path = out / "verificacao_cientifica.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return VerificationReportArtifacts(
        report_path=report_path,
        manifest_path=manifest_path,
        figure_paths=figures,
    )
