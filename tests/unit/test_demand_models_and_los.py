"""Testes Prove-It da Feature F04: modelos globais, quantis monotônicos, NegBin Hurdle e LOS."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from previsao_leitos.demand import (
    NegativeBinomialHurdleForecaster,
    PooledRidgeDeltaForecaster,
    QuantileGBDTForecaster,
    SpecialtyLosEstimator,
    prepare_global_demand_splits,
)
from previsao_leitos.domain import LeakageViolationError
from previsao_leitos.geo import SplitPolicy


def _synthetic_multi_region_panel() -> pl.DataFrame:
    """Painel sintético multi-região (2018-2025) com sinal de aterro, remanso e lag +2 sem."""
    rng = np.random.default_rng(123)
    regions = [
        ("Regiao 10 - Capital", "guaiba_capital", 0.020, 1.8),
        ("Regiao 21 - Sul", "lagoa_patos_sul", 0.008, 1.0),
        ("Regiao 08 - Vale do Cai", "cai", 0.006, 0.9),
        ("Regiao 17 - Planalto", "planalto_alto_jacui", 0.0005, 0.7),
        ("Regiao 23 - Serra", "serra_taquari_antas", 0.0002, 0.6),
    ]
    rows: list[dict[str, object]] = []
    weeks = [(y, w) for y in range(2018, 2026) for w in range(1, 53)]
    for reg, basin, aterro, scale in regions:
        rain = rng.gamma(shape=2.0, scale=15.0, size=len(weeks))
        upstream = rain * (1.3 if "Capital" in reg or "Sul" in reg else 0.2)
        remanso = np.where(basin in {"guaiba_capital", "lagoa_patos_sul"}, upstream * 1.5, 0.0)
        # Injeta enchentes extremas em treino (2020), cal (2023) e auditoria (2024)
        for idx_shock in (110, 270, 330):
            if "Capital" in reg or "Sul" in reg or "Cai" in reg:
                rain[idx_shock] = 280.0
                upstream[idx_shock] = 360.0
                remanso[idx_shock] = (
                    900.0 if basin in {"guaiba_capital", "lagoa_patos_sul"} else 0.0
                )

        demand = np.zeros(len(weeks), dtype=np.float64)
        lepto = np.zeros(len(weeks), dtype=np.float64)
        for i, (_, w) in enumerate(weeks):
            up_lag2 = upstream[i - 2] if i >= 2 else 0.0
            rem_lag2 = remanso[i - 2] if i >= 2 else 0.0
            season = 12.0 * scale * np.sin(2.0 * np.pi * w / 52.0)
            shock_term = (0.20 * up_lag2 + 0.08 * rem_lag2) * (1.0 + 40.0 * aterro)
            demand[i] = max(0.0, 80.0 * scale + season + shock_term + rng.normal(0.0, 4.0))
            lam = np.exp(-2.0 + 0.007 * up_lag2 + 35.0 * aterro)
            lepto[i] = float(rng.poisson(min(lam, 30.0)))

        for i, (y, w) in enumerate(weeks):
            rows.append(
                {
                    "id_regiao_saude": reg[:9],
                    "nome_regiao_saude": reg,
                    "bacia_id": basin,
                    "ano_epi": y,
                    "semana_epi": w,
                    "chuva_sem_mm": float(rain[i]),
                    "chuva_montante_pond": float(upstream[i]),
                    "api_max_sem": float(rain[i] * 0.8),
                    "api_montante_pond": float(upstream[i] * 0.8),
                    "remanso_lagunar_idx": float(remanso[i]),
                    "indice_aterro_varzea": float(aterro),
                    "exposicao_aterro_ativa": float(aterro * (rain[i] + upstream[i])),
                    "chuva_catchment_mm": float(0.7 * rain[i] + 0.3 * upstream[i]),
                    "api_catchment_max": float(0.7 * rain[i] + 0.3 * upstream[i]),
                    "umidade_solo_sem": 0.65,
                    "temp_media_sem_c": 19.0,
                    "uti_adulto_diarias": float(round(demand[i], 2)),
                    "leptospirose_internacoes": float(lepto[i]),
                }
            )
    return pl.DataFrame(rows)


def test_global_demand_splits_block_2025_holdout_and_are_leak_free() -> None:
    df = _synthetic_multi_region_panel()
    policy = SplitPolicy(train_end_year=2022, cal_year=2023, dev_audit_year=2024, holdout_year=2025)

    with pytest.raises(LeakageViolationError, match="2025"):
        prepare_global_demand_splits(
            df,
            entity_col="nome_regiao_saude",
            target_col="uti_adulto_diarias",
            dynamic_feature_cols=("chuva_sem_mm", "chuva_montante_pond", "remanso_lagunar_idx"),
            static_feature_cols=("indice_aterro_varzea",),
            horizon_weeks=2,
            eval_year=2025,
            policy=policy,
            allow_holdout=False,
        )

    perturbed = df.with_columns(
        pl.when(pl.col("ano_epi") >= 2024)
        .then(pl.col("uti_adulto_diarias") * 100.0 + 5000.0)
        .otherwise(pl.col("uti_adulto_diarias"))
        .alias("uti_adulto_diarias")
    )
    sp_a = prepare_global_demand_splits(
        df,
        entity_col="nome_regiao_saude",
        target_col="uti_adulto_diarias",
        dynamic_feature_cols=("chuva_sem_mm", "chuva_montante_pond", "remanso_lagunar_idx"),
        static_feature_cols=("indice_aterro_varzea",),
        horizon_weeks=2,
        eval_year=2024,
        policy=policy,
    )
    sp_b = prepare_global_demand_splits(
        perturbed,
        entity_col="nome_regiao_saude",
        target_col="uti_adulto_diarias",
        dynamic_feature_cols=("chuva_sem_mm", "chuva_montante_pond", "remanso_lagunar_idx"),
        static_feature_cols=("indice_aterro_varzea",),
        horizon_weeks=2,
        eval_year=2024,
        policy=policy,
    )
    assert sp_a.train_df.equals(sp_b.train_df)


def test_quantile_gbdt_enforces_non_crossing_monotonicity_and_beats_persistence() -> None:
    df = _synthetic_multi_region_panel()
    splits = prepare_global_demand_splits(
        df,
        entity_col="nome_regiao_saude",
        target_col="uti_adulto_diarias",
        dynamic_feature_cols=(
            "chuva_sem_mm",
            "chuva_montante_pond",
            "remanso_lagunar_idx",
            "exposicao_aterro_ativa",
            "chuva_catchment_mm",
        ),
        static_feature_cols=("indice_aterro_varzea",),
        horizon_weeks=2,
        lag_weeks=(0, 1, 2),
        eval_year=2024,
    )

    model = QuantileGBDTForecaster(
        monotone_positive_prefixes=(
            "chuva_montante_pond",
            "remanso_lagunar_idx",
            "exposicao_aterro_ativa",
            "indice_aterro_varzea",
        ),
        max_iter=80,
        random_state=42,
    )
    model.fit(splits.train_df, splits.feature_cols)
    preds = model.predict_quantiles(splits.eval_df)

    # 1. Não-negatividade e não-cruzamento estrito: 0 <= P10 <= P50 <= P90 <= P95
    assert (preds["pred_p10"] >= 0.0).all()
    assert (preds["pred_p10"] <= preds["pred_p50"] + 1e-9).all()
    assert (preds["pred_p50"] <= preds["pred_p90"] + 1e-9).all()
    assert (preds["pred_p90"] <= preds["pred_p95"] + 1e-9).all()

    # 2. Prove-It de monotonicidade física: aumentar remanso + chuva montante NUNCA reduz P50/P90
    bumped_eval = splits.eval_df.with_columns(
        (pl.col("remanso_lagunar_idx_lag0") + 500.0).alias("remanso_lagunar_idx_lag0"),
        (pl.col("chuva_montante_pond_lag0") + 150.0).alias("chuva_montante_pond_lag0"),
        (pl.col("indice_aterro_varzea") + 0.02).alias("indice_aterro_varzea"),
    )
    preds_bumped = model.predict_quantiles(bumped_eval)
    assert (preds_bumped["pred_p50"] >= preds["pred_p50"] - 1e-6).all()
    assert (preds_bumped["pred_p90"] >= preds["pred_p90"] - 1e-6).all()

    # 3. Supera persistência no conjunto de avaliação sintético
    y_true = splits.eval_df["target_y"].to_numpy()
    y_pers = splits.eval_df["y_origin"].to_numpy()
    mae_mod = float(np.mean(np.abs(y_true - preds["pred_p50"].to_numpy())))
    mae_pers = float(np.mean(np.abs(y_true - y_pers)))
    assert mae_mod < mae_pers


def test_negbin_hurdle_and_pooled_ridge_produce_valid_quantiles() -> None:
    df = _synthetic_multi_region_panel()
    splits = prepare_global_demand_splits(
        df,
        entity_col="nome_regiao_saude",
        target_col="leptospirose_internacoes",
        dynamic_feature_cols=("chuva_sem_mm", "chuva_montante_pond", "remanso_lagunar_idx"),
        static_feature_cols=("indice_aterro_varzea",),
        horizon_weeks=2,
        lag_weeks=(0, 1),
        eval_year=2024,
    )

    hurdle = NegativeBinomialHurdleForecaster(alpha_dispersion=0.5)
    hurdle.fit(splits.train_df, splits.feature_cols)
    hp = hurdle.predict_quantiles(splits.eval_df)
    assert (hp["pred_p10"] >= 0.0).all()
    assert (hp["pred_p10"] <= hp["pred_p50"] + 1e-9).all()
    assert (hp["pred_p50"] <= hp["pred_p90"] + 1e-9).all()
    assert (hp["pred_p90"] <= hp["pred_p95"] + 1e-9).all()

    ridge = PooledRidgeDeltaForecaster(alpha=50.0)
    ridge.fit(splits.train_df, splits.feature_cols)
    rp = ridge.predict_quantiles(splits.eval_df)
    assert (rp["pred_p10"] >= 0.0).all()
    assert (rp["pred_p50"] <= rp["pred_p95"] + 1e-9).all()


def test_specialty_los_estimator_is_causal_and_converts_bed_days_via_littles_law() -> None:
    sih = pl.DataFrame(
        {
            "ano_epi": [2021, 2022, 2024],
            "especialidade": ["uti_adulto", "uti_adulto", "uti_adulto"],
            "internacoes": [100, 100, 50],
            "dias_permanencia_total": [900, 1100, 99999],
            "diarias_uti_total": [700, 700, 99999],
        }
    )
    est = SpecialtyLosEstimator.fit_causal(sih, train_end_year=2022)
    # 2024 (99999) não pode vazar! Média causal <= 2022: 10.0 dias perm / 7.0 diárias UTI
    assert est.mean_los_days("uti_adulto") == pytest.approx(10.0)
    assert est.mean_icu_days("uti_adulto") == pytest.approx(7.0)
    # 700 diárias semanais / 7 dias = 100.0 leitos ocupados médios por dia
    assert est.weekly_bed_days_to_occupied_beds(700.0) == pytest.approx(100.0)
