"""Testes da Task 1.3.x — registro de candidatos, splits e benchmark multi-modelo."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from previsao_leitos.audit.model_benchmark import run_benchmark
from previsao_leitos.demand.baselines import PersistenceForecaster
from previsao_leitos.demand.modeling import (
    TARGET_SPECS,
    make_modeling_splits,
    resolve_targets,
)
from previsao_leitos.demand.registry import (
    champion_spec,
    get_candidate,
    list_candidates,
    resolve_selection,
)
from previsao_leitos.domain import DataContractError


def _synthetic_panel(
    n_regions: int = 5,
    years: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024),
) -> pl.DataFrame:
    """Painel sintético com todas as colunas exigidas pelo pipeline de modelagem."""
    rng = np.random.default_rng(11)
    rows: list[dict[str, object]] = []
    for i in range(n_regions):
        entity = f"Regiao_{i}"
        basin = f"Bacia_{i % 2}"
        cap = 300.0 if i % 2 == 0 else 80.0
        for year in years:
            for week in range(1, 53):
                season = 0.70 + 0.05 * np.sin(2.0 * np.pi * week / 52.0)
                rain = float(rng.uniform(0.0, 120.0))
                shock = float(min(1.0, rain / 200.0))
                rate = season + 0.10 * shock
                rows.append(
                    {
                        "nome_regiao_saude": entity,
                        "bacia_id": basin,
                        "ano_epi": year,
                        "semana_epi": week,
                        "uti_adulto_diarias": cap * 7.0 * rate,
                        "diarias_uti_total_ativas": cap * 7.0 * rate * 1.1,
                        "clinico_pacientes_dia": cap * 3.0 * rate,
                        "respiratorio_pacientes_dia": cap * 2.0 * rate,
                        "chuva_sem_mm": rain,
                        "chuva_montante_multihop": rain * 1.2,
                        "api_max_sem": rain * 0.8,
                        "remanso_lagunar_idx": shock * 0.5,
                        "exposicao_aterro_ativa": float(i % 3),
                        "prob_enchente_f03_oof": shock,
                        "severidade_hidro_f03_oof": shock * 2.0,
                        "el_nino_intensity": float(rng.normal(0.0, 0.5)),
                        "enso_regime": "neutro" if week % 3 else "el_nino",
                        "frac_urbano": 0.4 + 0.1 * i,
                        "frac_floresta": 0.2,
                        "indice_aterro_varzea": 0.1 * i,
                        "cnes_uti_sus_leitos": cap,
                        "cnes_total_sus_leitos": cap * 3.0,
                    }
                )
    return pl.DataFrame(rows)


def test_registry_exposes_all_candidates_and_champion() -> None:
    names = [spec.name for spec in list_candidates()]
    assert names[0] == "persistencia"
    assert "f041_mondrian" in names
    assert champion_spec().name == "f041_mondrian"
    assert get_candidate("hierarchical_bayes").requires == ("pymc", "arviz")


def test_resolve_selection_supports_all_champion_baselines_and_list() -> None:
    assert len(resolve_selection("all")) == len(list_candidates())
    assert [s.name for s in resolve_selection("champion")] == ["f041_mondrian"]
    assert all(s.family == "baseline" for s in resolve_selection("baselines"))
    assert [s.name for s in resolve_selection("persistencia,ridge_pooled")] == [
        "persistencia",
        "ridge_pooled",
    ]
    with pytest.raises(DataContractError):
        resolve_selection("modelo_inexistente")


def test_resolve_targets_covers_four_bed_targets() -> None:
    assert set(TARGET_SPECS) == {"uti_adulto", "uti_total", "clinico", "respiratorio"}
    assert len(resolve_targets("all")) == 4
    with pytest.raises(DataContractError):
        resolve_targets("uti_neonatal")


def test_baseline_predicts_ordered_quantiles_anchored_on_origin() -> None:
    panel = _synthetic_panel()
    splits = make_modeling_splits(panel, target="uti_adulto", eval_year=2024)
    model = PersistenceForecaster().fit(splits.train_df, splits.feature_cols)
    preds = model.predict_quantiles(splits.eval_df)

    assert preds.columns == ["pred_p10", "pred_p50", "pred_p90", "pred_p95"]
    p10 = preds["pred_p10"].to_numpy()
    p95 = preds["pred_p95"].to_numpy()
    assert np.all(p10 <= p95)
    assert np.all(p10 >= 0.0)
    origin = splits.eval_df["y_origin"].to_numpy()
    p50 = preds["pred_p50"].to_numpy()
    shift = p50 - origin
    # A persistência é a âncora mais o quantil mediano dos resíduos de treino (constante).
    assert np.allclose(shift, shift[0], atol=1e-9)


def test_splits_block_holdout_year() -> None:
    panel = _synthetic_panel()
    with pytest.raises(DataContractError):
        make_modeling_splits(panel, target="uti_adulto", eval_year=2025)


def test_splits_join_origin_metadata_for_conformal_grouping() -> None:
    panel = _synthetic_panel()
    splits = make_modeling_splits(panel, target="clinico", eval_year=2024)
    for frame in (splits.cal_df, splits.eval_df):
        assert "porte_regional" in frame.columns
        assert "enso_regime" in frame.columns
        assert "bacia_id" in frame.columns
    assert splits.eval_df["target_ano_epi"].max() == 2024
    assert splits.cal_df["target_ano_epi"].max() == 2023
    assert splits.train_df["target_ano_epi"].max() <= 2022


def test_benchmark_produces_metrics_predictions_and_decisions() -> None:
    panel = _synthetic_panel()
    splits = make_modeling_splits(panel, target="uti_adulto", eval_year=2024)
    result = run_benchmark(
        splits,
        run_id="unit-test",
        candidates="persistencia,sazonal_ingenuo,ridge_pooled,f041_core,f041_mondrian",
        n_boot=20,
        seed=7,
    )

    assert set(result.metrics["candidate"].to_list()) == {
        "persistencia",
        "sazonal_ingenuo",
        "ridge_pooled",
        "f041_core",
        "f041_mondrian",
    }
    assert (result.metrics.filter(pl.col("status") == "ok").height) >= 3
    for column in ("mase_vs_pers", "pinball_p95", "cobertura_p10_p90", "monotonic_violations"):
        assert column in result.metrics.columns

    assert result.predictions.height > 0
    assert set(result.predictions.columns) >= {
        "entity",
        "target_week",
        "model",
        "observed",
        "pred_p50",
    }
    assert result.predictions["observed"].null_count() == 0
    assert result.predictions["pred_p50"].null_count() == 0

    # LGPD: nenhuma região publicada com menos de 5 semanas.
    if result.comparison.height:
        assert result.comparison["n_obs"].min() >= 5

    assert result.decisions.height >= 1
    assert set(result.decisions["decision"].to_list()) <= {
        "PROMOTE",
        "REJECT",
        "KEEP_CHAMPION",
    }


def test_benchmark_is_deterministic_for_same_seed() -> None:
    panel = _synthetic_panel()
    splits = make_modeling_splits(panel, target="uti_adulto", eval_year=2024)
    kwargs = {
        "run_id": "det",
        "candidates": "persistencia,f041_core",
        "n_boot": 10,
        "seed": 3,
    }
    first = run_benchmark(splits, **kwargs)
    second = run_benchmark(splits, **kwargs)
    key = ["candidate", "mae_p50", "mase_vs_pers", "pinball_p95"]
    assert first.metrics.select(key).equals(second.metrics.select(key))


def test_benchmark_marks_unavailable_candidate_instead_of_dropping_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    import previsao_leitos.demand.registry as registry

    panel = _synthetic_panel()
    splits = make_modeling_splits(panel, target="uti_adulto", eval_year=2024)
    unavailable = replace(
        get_candidate("hierarchical_bayes"), requires=("modulo_inexistente_xyz",)
    )
    monkeypatch.setitem(registry._BY_NAME, "hierarchical_bayes", unavailable)

    result = run_benchmark(
        splits,
        run_id="unavail",
        candidates=["persistencia", "hierarchical_bayes"],
        n_boot=10,
        seed=5,
    )
    status = result.metrics.filter(pl.col("candidate") == "hierarchical_bayes")["status"]
    assert status.to_list() == ["unavailable"]
