"""Testes da Task 1.2.6 — pooling hierárquico bayesiano (PyMC + nutpie + ArviZ).

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from previsao_leitos.demand.hierarchical_bayes import (
    HierarchicalBayesConfig,
    HierarchicalBayesDiagnostics,
    build_hierarchical_model,
    fit_hierarchical_bayes,
    ppc_tail_probability,
    summarize_diagnostics,
    validate_hierarchical_panel,
)
from previsao_leitos.domain import DataContractError, LeakageViolationError


def _panel(n_weeks: int = 40, seed: int = 7) -> pl.DataFrame:
    """Painel sintético hierárquico: 12 regiões em 6 bacias, ruído NB (alpha=40).

    O ruído é Negative Binomial (não Normal) para casar com a verossimilhança do
    modelo; caso contrário o parâmetro de dispersão fica no limite e a geometria do
    NUTS degrada por má especificação, não por defeito do modelo.
    """
    rng = np.random.default_rng(seed)
    n_entities, n_basins = 12, 6
    basin_eff = {f"Bacia_{j}": 0.10 * ((-1) ** j) for j in range(n_basins)}
    slope = {f"Bacia_{j}": 0.10 + 0.05 * j for j in range(n_basins)}
    rows: list[dict[str, float | int | str]] = []
    for i in range(n_entities):
        ent = f"Regiao_{i}"
        bacia = f"Bacia_{i % n_basins}"
        for yr in (2021, 2022):
            for wk in range(1, n_weeks + 1):
                shock = float(rng.uniform(0.02, 0.30))
                rate = (
                    0.70 * (1.0 + 0.18 * ((-1) ** i))
                    + 0.04 * np.sin(2.0 * np.pi * wk / 52.0)
                    + slope[bacia] * shock
                    + basin_eff[bacia]
                )
                mu = 400.0 * rate
                y = rng.negative_binomial(40, 40 / (40 + mu))
                rows.append(
                    {
                        "nome_regiao_saude": ent,
                        "bacia_id": bacia,
                        "target_ano_epi": yr,
                        "target_semana_epi": wk,
                        "capacidade_sem": 400.0,
                        "target_week_sin": float(np.sin(2.0 * np.pi * wk / 52.0)),
                        "target_week_cos": float(np.cos(2.0 * np.pi * wk / 52.0)),
                        "prob_enchente_f03_oof": shock,
                        "target_y": float(y),
                    }
                )
    return pl.DataFrame(rows)


def test_validate_filters_to_train_end_year() -> None:
    late = _panel(n_weeks=4).with_columns(
        pl.lit(2024, dtype=pl.Int64).alias("target_ano_epi")
    )
    df = pl.concat([_panel(n_weeks=4), late])
    out = validate_hierarchical_panel(df)
    assert out["target_ano_epi"].max() <= 2022
    assert out.height > 0


def test_validate_rejects_holdout_rows() -> None:
    holdout = _panel(n_weeks=4).with_columns(
        pl.lit(2025, dtype=pl.Int64).alias("target_ano_epi")
    )
    df = pl.concat([_panel(n_weeks=4), holdout])
    with pytest.raises(LeakageViolationError):
        validate_hierarchical_panel(df)


def test_validate_rejects_non_positive_capacity() -> None:
    df = _panel(n_weeks=4).with_columns(pl.lit(0.0).alias("capacidade_sem"))
    with pytest.raises(DataContractError):
        validate_hierarchical_panel(df)


def test_validate_rejects_missing_columns() -> None:
    df = _panel(n_weeks=4).drop("target_y")
    with pytest.raises(DataContractError):
        validate_hierarchical_panel(df)


def test_validate_rejects_negative_target() -> None:
    df = _panel(n_weeks=4).with_columns(pl.lit(-1.0).alias("target_y"))
    with pytest.raises(DataContractError):
        validate_hierarchical_panel(df)


def test_config_defaults_are_auditable() -> None:
    cfg = HierarchicalBayesConfig()
    assert cfg.n_chains >= 2
    assert cfg.n_draws >= 500
    assert cfg.n_tune >= 500
    assert 0.8 <= cfg.target_accept <= 0.99
    assert cfg.random_seed == 42
    assert cfg.train_end_year <= cfg.holdout_year


def test_bayes_is_imported_lazily() -> None:
    """C-07: `import hierarchical_bayes` não pode importar pymc/arviz/nutpie no topo."""
    src = (
        Path(__file__).resolve().parents[2]
        / "src/previsao_leitos/demand/hierarchical_bayes.py"
    )
    tree = ast.parse(src.read_text(encoding="utf-8"))
    banned = {"pymc", "arviz", "nutpie", "pytensor"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] not in banned for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in banned


@pytest.mark.slow
def test_fit_passes_all_diagnostics_and_ppc() -> None:
    pytest.importorskip("pymc")
    pytest.importorskip("arviz")
    pytest.importorskip("nutpie")

    panel = _panel()
    cfg = HierarchicalBayesConfig(
        n_chains=4,
        n_draws=3000,
        n_tune=1500,
        target_accept=0.99,
        random_seed=11,
    )
    model, idata = fit_hierarchical_bayes(panel, config=cfg)
    diag = summarize_diagnostics(idata)

    assert isinstance(diag, HierarchicalBayesDiagnostics)
    assert diag.r_hat_max < 1.01
    assert diag.ess_bulk_min >= 400
    assert diag.divergences == 0
    assert np.isfinite(diag.loo_elpd)
    assert diag.pareto_k_max < 0.7

    observed = panel.sort(["nome_regiao_saude", "target_ano_epi", "target_semana_epi"])[
        "target_y"
    ].to_numpy()
    p_value = ppc_tail_probability(model, idata, observed)
    assert 0.02 < p_value < 0.98


@pytest.mark.slow
def test_partial_pooling_shrinks_small_entity() -> None:
    pytest.importorskip("pymc")
    pytest.importorskip("arviz")
    pytest.importorskip("nutpie")

    rng = np.random.default_rng(3)
    rows: list[dict[str, float | int | str]] = []
    for ent in ("Grande_A", "Grande_B", "Grande_C"):
        for wk in range(1, 81):
            rows.append(
                {
                    "nome_regiao_saude": ent,
                    "bacia_id": "Bacia_0",
                    "target_ano_epi": 2021 + (wk - 1) // 40,
                    "target_semana_epi": (wk - 1) % 40 + 1,
                    "capacidade_sem": 500.0,
                    "target_week_sin": float(np.sin(2 * np.pi * wk / 52)),
                    "target_week_cos": float(np.cos(2 * np.pi * wk / 52)),
                    "prob_enchente_f03_oof": float(rng.uniform(0.02, 0.3)),
                    "target_y": 350.0 + float(rng.normal(0, 8)),
                }
            )
    for wk in range(1, 3):
        rows.append(
            {
                "nome_regiao_saude": "Pequena",
                "bacia_id": "Bacia_0",
                "target_ano_epi": 2022,
                "target_semana_epi": wk,
                "capacidade_sem": 500.0,
                "target_week_sin": float(np.sin(2 * np.pi * wk / 52)),
                "target_week_cos": float(np.cos(2 * np.pi * wk / 52)),
                "prob_enchente_f03_oof": 0.05,
                "target_y": 620.0,
            }
        )
    panel = pl.DataFrame(rows)
    cfg = HierarchicalBayesConfig(n_chains=2, n_draws=800, n_tune=800, random_seed=5)
    _model, idata = fit_hierarchical_bayes(panel, config=cfg)

    posterior = idata.posterior
    entities = list(posterior["entity"].to_numpy())
    idx = entities.index("Pequena")
    basin_offset = float(posterior["u_basin"].isel({"basin": 0}).mean())
    deviation = float(posterior["u_entity"].isel({"entity": idx}).mean()) - basin_offset

    rates = panel.with_columns(
        (pl.col("target_y") / pl.col("capacidade_sem")).alias("_rate")
    )
    naive_global = float(np.log(rates["_rate"].mean()))
    naive_small = float(
        np.log(rates.filter(pl.col("nome_regiao_saude") == "Pequena")["_rate"].mean())
    )
    naive_dev = naive_small - naive_global

    # O desvio da entidade pequena deve ser puxado para a bacia (shrinkage parcial).
    assert np.sign(deviation) == np.sign(naive_dev)
    assert abs(deviation) < 0.9 * abs(naive_dev)


def test_build_model_requires_fitted_columns() -> None:
    pytest.importorskip("pymc")
    panel = _panel().drop("prob_enchente_f03_oof")
    with pytest.raises(DataContractError):
        build_hierarchical_model(panel)


def test_sampling_backend_resolution_is_fail_closed() -> None:
    from previsao_leitos.demand.hierarchical_bayes import (
        HierarchicalBayesConfig,
        resolve_sampling_backend,
    )

    sampler, detail = resolve_sampling_backend(
        HierarchicalBayesConfig(nuts_sampler="nutpie", device="cpu")
    )
    assert sampler == "nutpie"
    assert "CPU" in detail

    with pytest.raises(DataContractError):
        resolve_sampling_backend(
            HierarchicalBayesConfig(nuts_sampler="nutpie", device="gpu")
        )
    with pytest.raises(DataContractError):
        resolve_sampling_backend(
            HierarchicalBayesConfig(nuts_sampler="numpyro", device="gpu")
        )
    with pytest.raises(DataContractError):
        resolve_sampling_backend(HierarchicalBayesConfig(nuts_sampler="invalido"))
