"""Testes Prove-It da Feature F05: simulador de filas SimPy, evacuação em pôlder e Fig. C5."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from previsao_leitos.audit import plot_c5_simpy_contingency
from previsao_leitos.domain import DataContractError
from previsao_leitos.readiness import (
    HospitalReadinessConfig,
    evaluate_observable_triggers,
    simulate_counterfactual_readiness,
)


def _synthetic_flood_weekly_schedule() -> pl.DataFrame:
    """20 semanas com onda de cheia intensa nas semanas 8..13 e previsão antecipada P90."""
    rows: list[dict[str, object]] = []
    for wk in range(1, 21):
        in_flood = 8 <= wk <= 13
        arrivals = 180.0 if in_flood else 95.0
        # Previsão ex-ante (h=2 sem.) já sinaliza pico nas semanas 6..11
        pred_p90 = 195.0 if 6 <= wk <= 12 else 110.0
        rows.append(
            {
                "semana_epi": wk,
                "chegadas_semanais": arrivals,
                "previsao_p90_semanal": pred_p90,
            }
        )
    return pl.DataFrame(rows)


def test_readiness_config_validates_physical_bounds() -> None:
    with pytest.raises(DataContractError):
        HospitalReadinessConfig(base_beds=0, surge_beds=20, los_days=6.5)
    with pytest.raises(DataContractError):
        HospitalReadinessConfig(base_beds=100, surge_beds=20, los_days=-1.0)


def test_proactive_h2_reduces_queue_peak_and_polder_evacuation_increases_stress(
    tmp_path: Path,
) -> None:
    schedule = _synthetic_flood_weekly_schedule()
    cfg = HospitalReadinessConfig(
        region_label="Região 10 - Capital (Hospitais-Escola)",
        base_beds=120,
        surge_beds=35,
        los_days=6.5,
        elective_share=0.25,
        surge_lead_time_days=2,
        polder_hospital_loss_frac=0.0,
    )

    comp = simulate_counterfactual_readiness(schedule, cfg, seed=42)
    assert comp.proactive_h2.peak_queue_patients < comp.reactive_h0.peak_queue_patients
    assert comp.proactive_h2.overload_days_gt95 <= comp.reactive_h0.overload_days_gt95
    assert comp.queue_peak_reduction_pct > 0.0

    # Cenário de estresse composto: perda de 20% dos leitos por evacuação em área aterrada/pôlder
    cfg_polder = HospitalReadinessConfig(
        region_label="Região 10 - Capital (Estresse Pôlder)",
        base_beds=120,
        surge_beds=35,
        los_days=6.5,
        elective_share=0.25,
        surge_lead_time_days=2,
        polder_hospital_loss_frac=0.20,
    )
    comp_polder = simulate_counterfactual_readiness(schedule, cfg_polder, seed=42)
    assert comp_polder.reactive_h0.peak_queue_patients > comp.reactive_h0.peak_queue_patients
    assert (
        comp_polder.proactive_h2.peak_queue_patients < comp_polder.reactive_h0.peak_queue_patients
    )

    # Gatilhos observáveis (sempre marcados como pesquisa)
    triggers = evaluate_observable_triggers(
        predicted_p90_beds=135.0,
        base_beds=120,
        upstream_api_mm=240.0,
        remanso_idx=350.0,
    )
    assert triggers.level in {"LARANJA", "VERMELHO"}
    assert "Not for patient care" in triggers.disclaimer

    # Fig. C5 gerada limpa em PNG 300 DPI
    fig_path = plot_c5_simpy_contingency(
        comp,
        target_label="UTI Adulto - Região 10 (Capital/POA)",
        output_path=tmp_path / "fig_c5_contingencia_simpy.png",
    )
    assert fig_path.exists() and fig_path.stat().st_size > 1000
