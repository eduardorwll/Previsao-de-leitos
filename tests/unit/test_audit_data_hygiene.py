"""Testes unitários dos 8 sensores de higiene de dados cruzados (Gates G1-G8)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from previsao_leitos.audit.data_hygiene import (
    DataHygieneScorecard,
    check_g1_collinearity_vif,
    check_g2_join_loss,
    check_g3_census_capacity_conservation,
    check_g4_right_censoring,
    check_g5_maup_dilution,
    check_g6_spatial_grid_collision,
    check_g7_sutva_donor_purity,
    check_g8_degenerate_surge_labels,
    compile_hygiene_scorecard,
)
from previsao_leitos.domain.exceptions import DataContractError


def test_g1_collinearity_vif_pass_and_fail_c2() -> None:
    rng = np.random.default_rng(42)
    x1 = rng.normal(0.0, 1.0, size=200)
    x2 = rng.normal(0.0, 1.0, size=200)
    df_clean = pl.DataFrame({"chuva_sem_mm": x1, "pressao_ref_ext": x2})
    res_pass = check_g1_collinearity_vif(df_clean, ["chuva_sem_mm", "pressao_ref_ext"])
    assert res_pass.gate_id == "G1"
    assert res_pass.status == "PASS"
    assert res_pass.observed_value < 2.0

    # Reproduz C2: W_ref colapsada em auto-referência quase colinear a chuva_sem_mm
    x_collinear = x1 + rng.normal(0.0, 0.005, size=200)
    df_bad = pl.DataFrame({"chuva_sem_mm": x1, "pressao_ref": x_collinear})
    res_fail = check_g1_collinearity_vif(df_bad, ["chuva_sem_mm", "pressao_ref"], max_vif=5.0)
    assert res_fail.status == "FAIL"
    assert res_fail.observed_value > 100.0
    assert "vif_by_feature" in res_fail.details


def test_g2_join_loss_pass_fail_and_cartesian_explosion() -> None:
    res_pass = check_g2_join_loss(total_before=1_000_000, total_after=999_950, max_loss_pct=0.1)
    assert res_pass.gate_id == "G2"
    assert res_pass.status == "PASS"
    assert res_pass.observed_value == pytest.approx(0.005)

    # Reproduz M10: 175.542 AIHs perdidas no inner join com cnes_he
    res_fail = check_g2_join_loss(total_before=1_930_000, total_after=1_754_458, max_loss_pct=0.1)
    assert res_fail.status == "FAIL"
    assert res_fail.observed_value > 9.0

    # Explosão cartesiana (total_after > total_before) deve falhar explicitamente
    res_explosion = check_g2_join_loss(total_before=1_000, total_after=1_250)
    assert res_explosion.status == "FAIL"
    assert res_explosion.details["cartesian_explosion"] is True


def test_g3_census_capacity_conservation_detects_lump_at_admission_c4() -> None:
    df_clean = pl.DataFrame(
        {
            "regiao": ["R10", "R21"],
            "diarias_uti_sem": [431 * 7 * 0.88, 120 * 7 * 0.92],
            "leitos_uti_sus": [431.0, 120.0],
        }
    )
    res_pass = check_g3_census_capacity_conservation(
        df_clean,
        occupied_col="diarias_uti_sem",
        capacity_col="leitos_uti_sus",
        days_per_period=7.0,
    )
    assert res_pass.gate_id == "G3"
    assert res_pass.status == "PASS"
    assert res_pass.observed_value < 1.0

    # Reproduz C4: ocupação impossível de 316% por lump-at-admission
    df_bad = pl.DataFrame(
        {
            "regiao": ["R10", "R08"],
            "diarias_uti_sem": [431 * 7 * 0.90, 50 * 7 * 3.16],
            "leitos_uti_sus": [431.0, 50.0],
        }
    )
    res_fail = check_g3_census_capacity_conservation(
        df_bad,
        occupied_col="diarias_uti_sem",
        capacity_col="leitos_uti_sus",
        days_per_period=7.0,
        max_occupancy_ratio=1.30,
    )
    assert res_fail.status == "FAIL"
    assert res_fail.observed_value == pytest.approx(3.16, rel=1e-2)


def test_g4_right_censoring_detects_2025_week52_drop_c1() -> None:
    weeks = list(range(1, 21))
    stable_vals = [1000.0 + (i % 3) * 15.0 for i in weeks]
    df_clean = pl.DataFrame({"semana_idx": weeks, "aih_total": stable_vals})
    res_pass = check_g4_right_censoring(df_clean, time_col="semana_idx", value_col="aih_total")
    assert res_pass.gate_id == "G4"
    assert res_pass.status == "PASS"
    assert res_pass.observed_value < 5.0

    # Reproduz C1: queda de -96% na última semana de 2025 por falta de competências de 2026
    censored_vals = [*stable_vals[:-1], 40.0]
    df_censored = pl.DataFrame({"semana_idx": weeks, "aih_total": censored_vals})
    res_fail = check_g4_right_censoring(df_censored, time_col="semana_idx", value_col="aih_total")
    assert res_fail.status == "FAIL"
    assert res_fail.observed_value > 90.0


def test_g5_maup_dilution_detects_7_9x_landfill_distortion_a8() -> None:
    res_pass = check_g5_maup_dilution(
        raw_area_index=[0.10, 0.25, 0.40],
        weighted_index=[0.12, 0.28, 0.45],
        max_dilution_ratio=2.0,
    )
    assert res_pass.gate_id == "G5"
    assert res_pass.status == "PASS"
    assert res_pass.observed_value < 1.5

    # Reproduz A8: diluição espacial MAUP de 7,9x no MapBiomas por área bruta
    res_fail = check_g5_maup_dilution(
        raw_area_index=[0.05, 0.10],
        weighted_index=[0.395, 0.11],
        max_dilution_ratio=2.0,
    )
    assert res_fail.status == "FAIL"
    assert res_fail.observed_value == pytest.approx(7.9, rel=1e-2)


def test_g6_spatial_grid_collision_detects_vale_cai_vs_caxias_a9() -> None:
    df_clean = pl.DataFrame(
        {
            "municipio": ["Porto Alegre", "Caxias do Sul", "Montenegro"],
            "grid_cell": ["(-30.0,-51.2)", "(-29.2,-51.2)", "(-29.7,-51.5)"],
            "altitude_m": [10.0, 800.0, 20.0],
        }
    )
    res_pass = check_g6_spatial_grid_collision(
        df_clean,
        unit_col="municipio",
        grid_key_col="grid_cell",
        elevation_col="altitude_m",
    )
    assert res_pass.gate_id == "G6"
    assert res_pass.status == "PASS"

    # Reproduz A9: Vale do Caí (20m) e Caxias do Sul (800m) na mesma célula NASA POWER 0,5°
    df_collision = pl.DataFrame(
        {
            "municipio": ["Montenegro", "Caxias do Sul", "Porto Alegre"],
            "grid_cell": ["(-29.5,-51.5)", "(-29.5,-51.5)", "(-30.0,-51.0)"],
            "altitude_m": [20.0, 800.0, 10.0],
        }
    )
    res_fail = check_g6_spatial_grid_collision(
        df_collision,
        unit_col="municipio",
        grid_key_col="grid_cell",
        elevation_col="altitude_m",
        max_elevation_diff_m=250.0,
    )
    assert res_fail.status == "FAIL"
    assert res_fail.details["max_elevation_diff_m"] == pytest.approx(780.0)


def test_g7_sutva_donor_purity_detects_contaminated_and_zero_bed_donors_c5() -> None:
    df_clean = pl.DataFrame(
        {
            "regiao": ["R01", "R02", "R03"],
            "chuva_choque_mm": [32.0, 45.0, 50.0],
            "leitos_uti_sus": [25.0, 40.0, 18.0],
        }
    )
    res_pass = check_g7_sutva_donor_purity(
        df_clean,
        unit_col="regiao",
        shock_col="chuva_choque_mm",
        capacity_col="leitos_uti_sus",
        max_donor_shock=100.0,
    )
    assert res_pass.gate_id == "G7"
    assert res_pass.status == "PASS"

    # Reproduz C5: R17 recebeu 280,4 mm na Sem. 18 e R18 tem 0 leitos UTI SUS
    df_bad = pl.DataFrame(
        {
            "regiao": ["R17", "R18", "R01"],
            "chuva_choque_mm": [280.4, 25.0, 30.0],
            "leitos_uti_sus": [30.0, 0.0, 20.0],
        }
    )
    res_fail = check_g7_sutva_donor_purity(
        df_bad,
        unit_col="regiao",
        shock_col="chuva_choque_mm",
        capacity_col="leitos_uti_sus",
        max_donor_shock=100.0,
    )
    assert res_fail.status == "FAIL"
    assert set(res_fail.details["violating_units"]) == {"R17", "R18"}


def test_g8_degenerate_surge_labels_detects_zero_ge_zero_c6() -> None:
    y = np.array([0.0, 0.0, 5.0, 12.0])
    base = np.array([0.0, 0.0, 4.0, 8.0])
    clean_flags = np.array([False, False, True, True])
    res_pass = check_g8_degenerate_surge_labels(y, base, clean_flags)
    assert res_pass.gate_id == "G8"
    assert res_pass.status == "PASS"
    assert res_pass.observed_value == 0.0

    # Reproduz C6: 0.0 >= 1.10 * 0.0 rotulou zeros como surto
    buggy_flags = y >= 1.10 * base
    res_fail = check_g8_degenerate_surge_labels(y, base, buggy_flags)
    assert res_fail.status == "FAIL"
    assert res_fail.observed_value == 2.0
    assert res_fail.details["degenerate_rate_pct"] == pytest.approx(50.0)


def test_compile_scorecard_json_and_require_all_pass(tmp_path: Path) -> None:
    g2_ok = check_g2_join_loss(1000, 1000)
    g8_fail = check_g8_degenerate_surge_labels([0.0, 1.0], [0.0, 0.5], [True, True])
    scorecard = compile_hygiene_scorecard([g2_ok, g8_fail])

    assert isinstance(scorecard, DataHygieneScorecard)
    assert scorecard.overall_status == "FAIL"
    assert scorecard.passed_count == 1
    assert scorecard.failed_count == 1
    assert "Not for patient care" in scorecard.disclaimer

    out_path = scorecard.save_json(tmp_path / "data_hygiene_scorecard.json")
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["overall_status"] == "FAIL"
    assert len(payload["gates"]) == 2

    with pytest.raises(DataContractError, match="G8"):
        scorecard.require_all_pass()
