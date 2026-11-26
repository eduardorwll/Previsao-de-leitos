"""Testes unitários da Task 1.2.5: bloqueador persistente SimPy + Pareto NSGA-II CRN 2 estágios."""

from __future__ import annotations

import polars as pl
import pytest

from previsao_leitos.domain import DataContractError
from previsao_leitos.readiness import (
    HospitalReadinessConfig,
    evaluate_policy_on_crn,
    generate_crn_scenarios,
    optimize_contingency_pareto_front,
)


def _stress_weekly_schedule() -> pl.DataFrame:
    n_wk = 16
    arr = [92.0 + (28.0 if 6 <= i <= 9 else 0.0) for i in range(n_wk)]
    p90 = [a * 1.12 for a in arr]
    flood = [1 if 6 <= i <= 9 else 0 for i in range(n_wk)]
    return pl.DataFrame(
        {
            "chegadas_semanais": arr,
            "previsao_p90_semanal": p90,
            "em_cheia_hidro": flood,
        }
    )


def test_crn_streams_are_deterministic_and_track_surge_bed_days() -> None:
    sched = _stress_weekly_schedule()
    cfg = HospitalReadinessConfig(
        base_beds=100,
        surge_beds=20,
        los_days=6.5,
        polder_hospital_loss_frac=0.15,
        region_label="Regiao_Teste",
    )
    streams_a = generate_crn_scenarios(sched, cfg, seeds=(10, 11))
    streams_b = generate_crn_scenarios(sched, cfg, seeds=(10, 11))
    assert streams_a == streams_b

    res_r = evaluate_policy_on_crn(
        weekly_schedule=sched, config=cfg, policy_mode="reactive_h0", stream=streams_a[0]
    )
    res_p = evaluate_policy_on_crn(
        weekly_schedule=sched, config=cfg, policy_mode="proactive_h2", stream=streams_a[0]
    )
    assert res_p.peak_queue_patients <= res_r.peak_queue_patients
    assert res_p.surge_bed_days_active >= 0
    assert res_p.daily_df.height == sched.height * 7


def test_nsga2_pareto_front_two_stage_crn_is_valid_and_non_dominated() -> None:
    sched = _stress_weekly_schedule()
    cfg = HospitalReadinessConfig(
        base_beds=100,
        surge_beds=20,
        los_days=6.5,
        elective_share=0.25,
        elective_deferral_frac=0.75,
        polder_hospital_loss_frac=0.15,
        region_label="Regiao_Pareto_Teste",
    )
    res = optimize_contingency_pareto_front(
        sched,
        cfg,
        r_search=2,
        r_audit=4,
        pop_size=8,
        n_gen=3,
        n_workers=1,
        seed=42,
    )
    assert res.r_search == 2 and res.r_audit == 4
    assert len(res.pareto_front) >= 1
    assert res.front_df.height == 2 + len(res.pareto_front)

    # Verifica estrita não-dominância entre as soluções retornadas na fronteira auditada
    pts = [
        (e.mean_peak_queue, e.mean_deferred_electives, e.mean_surge_bed_days)
        for e in res.pareto_front
    ]
    for i, a in enumerate(pts):
        for j, b in enumerate(pts):
            if i == j:
                continue
            assert not (
                b[0] <= a[0]
                and b[1] <= a[1]
                and b[2] <= a[2]
                and (b[0] < a[0] or b[1] < a[1] or b[2] < a[2])
            )

    # Política recomendada possui histerese válida (close < trigger)
    rec = res.recommended_policy.decision
    assert rec.hysteresis_close_threshold < rec.trigger_occ_threshold


def test_pareto_and_config_validate_invalid_bounds() -> None:
    sched = _stress_weekly_schedule()
    with pytest.raises(DataContractError):
        HospitalReadinessConfig(
            base_beds=50, surge_beds=10, los_days=5.0, trigger_occ_threshold=0.1
        )
    cfg = HospitalReadinessConfig(base_beds=50, surge_beds=10, los_days=5.0)
    with pytest.raises(DataContractError):
        optimize_contingency_pareto_front(sched, cfg, r_search=0, r_audit=2)
