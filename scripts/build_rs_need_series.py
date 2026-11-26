"""Constrói e valida as séries de necessidade (pré-registro v1) para todos os alvos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from previsao_leitos.demand.need_series import (
    NEED_SPECS,
    acceptance,
    build_need_series,
    check_monotonic_in_shock,
    save_need_series,
)

DATA_PROCESSED = Path("data/processed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Constrói as séries de necessidade (<= 2024).")
    parser.add_argument(
        "--panel-path", default=str(DATA_PROCESSED / "rs_modeling_panel_v3.parquet")
    )
    parser.add_argument("--target", default="all")
    parser.add_argument("--weighting", choices=("shock", "full"), default="shock")
    parser.add_argument("--baseline-stat", choices=("median", "mean"), default="median")
    parser.add_argument(
        "--baseline-mode", choices=("level_corrected", "seasonal"), default="level_corrected"
    )
    parser.add_argument("--version", default="v3")
    args = parser.parse_args(argv)

    panel_path = Path(args.panel_path)
    if not panel_path.exists():
        print(f"[erro] painel ausente: {panel_path}")
        return 2
    panel = pl.read_parquet(panel_path)

    keys = list(NEED_SPECS) if args.target == "all" else [args.target]
    failures = 0
    for key in keys:
        spec = NEED_SPECS[key]
        series = build_need_series(
            panel,
            spec=spec,
            weighting=args.weighting,
            baseline_stat=args.baseline_stat,
            baseline_mode=args.baseline_mode,
        )
        monotone = check_monotonic_in_shock(
            panel, spec=spec, baseline_mode=args.baseline_mode
        )
        ok, criteria = acceptance(series, monotone_in_shock=monotone)
        out_path = DATA_PROCESSED / f"rs_need_{key}_{args.version}.parquet"
        save_need_series(
            series,
            out_path=out_path,
            panel_path=panel_path,
            target_key=key,
            criteria=criteria,
        )
        status = "OK" if ok else "FALHOU"
        print(
            f"[{status}] {key}: {series.height:,} linhas | {out_path.name} | "
            f"excesso_2024_W18-30={float(criteria['event_excess_2024']):,.0f} | "
            f"monotônico={criteria['monotone_in_shock']} | "
            f"nível_pre={float(criteria['pre_event_level_ratio']):.3f}"
        )
        if args.weighting == "shock":
            pass
        if not ok:
            failures += 1
            print(f"  critérios: {criteria}")
    if failures == 0:
        _write_need_panel(panel, keys)
    return 1 if failures else 0


def _write_need_panel(panel: pl.DataFrame, keys: list[str]) -> None:
    """Anexa as colunas `need_<alvo>` ao painel de modelagem."""
    joined = panel
    for key in keys:
        series = pl.read_parquet(DATA_PROCESSED / f"rs_need_{key}_v3.parquet").select(
            ["nome_regiao_saude", "ano_epi", "semana_epi", pl.col("need").alias(f"need_{key}")]
        )
        joined = joined.join(
            series, on=["nome_regiao_saude", "ano_epi", "semana_epi"], how="left"
        )
    out = DATA_PROCESSED / "rs_modeling_panel_need_v3.parquet"
    joined.write_parquet(out)
    print(f"[OK] painel de necessidade: {out} ({joined.height:,} linhas)")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
