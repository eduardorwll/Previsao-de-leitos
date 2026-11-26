"""Gera o snapshot versionado do benchmark (`docs/benchmark/RESULTADOS.md`)."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from previsao_leitos.audit.categorization import categorize, normalize_benchmark
from previsao_leitos.demand.modeling import TARGET_SPECS

_LABELS = {key: spec.label for key, spec in TARGET_SPECS.items()}


def _table(df: pl.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(":---" for _ in columns) + " |"
    rows = []
    for row in df.select(columns).to_dicts():
        cells = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, float):
                cells.append(f"{value:.3f}" if col.startswith("mase") else f"{value:.1f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Snapshot do benchmark para docs/.")
    parser.add_argument("--run-dir", default="reports/runs/rs_v1_3_benchmark_2024")
    parser.add_argument("--backtest-occ", default="reports/runs/bt_final_occ")
    parser.add_argument("--backtest-need", default="reports/runs/bt_final_need")
    parser.add_argument("--out", default="docs/benchmark/RESULTADOS.md")
    args = parser.parse_args(argv)

    metrics = pl.read_csv(Path(args.run_dir) / "benchmark.csv")
    lines = [
        "# Resultados do benchmark (snapshot)",
        "",
        f"> Run de referência: `{Path(args.run_dir).name}` · rubrica `v1`.",
        "> Este arquivo é gerado por `scripts/render_benchmark_docs.py`; não editar à mão.",
        "",
        "## 1. Benchmark anual (`--eval-year 2024`)",
        "",
    ]
    for target in metrics["target"].unique().to_list():
        subset = metrics.filter(pl.col("target") == target)
        graded = categorize(normalize_benchmark(subset))
        merged = subset.join(
            graded.select(["candidate", "acuracia", "fase", "calibracao", "selo"]),
            on="candidate",
            how="left",
        ).sort("wmape")
        lines.append(f"### {_LABELS.get(target, target)}")
        lines.append("")
        lines.append(
            _table(
                merged.head(8),
                ["candidate", "wmape", "mase_vs_pers", "peak_timing_error_weeks",
                 "cobertura_p10_p90", "erro_pct_p90", "acuracia", "fase", "selo"],
            )
        )
        lines.append("")

    backtest_lines = ["## 2. Régua de origem móvel (medianas entre folds)", ""]
    for label, path in (("Ocupação", args.backtest_occ), ("Necessidade", args.backtest_need)):
        summary_path = Path(path) / "backtest_summary.csv"
        if not summary_path.exists():
            continue
        summary = pl.read_csv(summary_path)
        backtest_lines.append(f"### {label}")
        backtest_lines.append("")
        backtest_lines.append(
            _table(
                summary.sort("mase_median").head(8),
                ["target", "candidate", "mase_median", "mase_worst", "coverage_median",
                 "peak_abs_median"],
            )
        )
        backtest_lines.append("")
    lines.extend(backtest_lines)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[docs] {out} gerado a partir de {args.run_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
