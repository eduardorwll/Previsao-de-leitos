"""Gera as figuras comparativas (modelo vs. observado) exibidas no README.

Lê o painel de modelagem local, roda o campeão (`gbdt_conformal`) e a persistência
nos alvos primários (UTI adulto e clínico) e grava `docs/figures/comparativo_<alvo>.png`
em 300 DPI (paleta Okabe-Ito). Requer o painel `rs_modeling_panel_v3.parquet`.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from previsao_leitos.audit.comparison_plots import plot_observed_vs_predicted
from previsao_leitos.audit.model_benchmark import run_benchmark
from previsao_leitos.demand.modeling import make_modeling_splits, resolve_targets

FIGURES_DIR = Path("docs/figures")
DEFAULT_PANEL = Path("data/processed/rs_modeling_panel_v3.parquet")
#: Alvos primários e modelos exibidos no README (campeão + baseline ingênuo).
TARGETS: tuple[str, ...] = ("uti_adulto", "clinico")
MODELS: tuple[str, ...] = ("gbdt_conformal", "persistencia")


def build_readme_figures(
    *,
    panel_path: Path,
    output_dir: Path,
    eval_year: int = 2024,
) -> list[Path]:
    """Roda o benchmark por alvo e escreve as figuras comparativas."""
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Painel ausente: {panel_path}. Rode `scripts/build_rs_modeling_panel.py`."
        )
    panel = pl.read_parquet(panel_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for key in TARGETS:
        spec = resolve_targets(key)[0]
        splits = make_modeling_splits(panel, target=key, eval_year=eval_year)
        result = run_benchmark(
            splits,
            run_id=f"readme:{key}",
            candidates=MODELS,
            n_boot=50,
            seed=42,
            include_ratchet=False,
        )
        out = output_dir / f"comparativo_{key}.png"
        plot_observed_vs_predicted(
            result.predictions,
            models=list(MODELS),
            output_path=out,
            target_label=spec.label,
            eval_year=eval_year,
        )
        written.append(out)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gera as figuras comparativas do README.")
    parser.add_argument("--panel-path", default=str(DEFAULT_PANEL))
    parser.add_argument("--output-dir", default=str(FIGURES_DIR))
    parser.add_argument("--eval-year", type=int, default=2024)
    args = parser.parse_args(argv)

    written = build_readme_figures(
        panel_path=Path(args.panel_path),
        output_dir=Path(args.output_dir),
        eval_year=args.eval_year,
    )
    for path in written:
        print(f"Figura: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
