"""Otimização multiobjetivo NSGA-II (pymoo) de contingência hospitalar em 2 estágios CRN.

Avalia políticas proativas `h = 2` semanas sobre o simulador SimPy (`readiness/simulator.py`)
minimizando simultaneamente três objetivos operacionais:
1. `f1` (`mean_peak_queue`): pico médio de fila de espera por leito (segurança assistencial);
2. `f2` (`mean_deferred_electives`): volume médio de cirurgias/internações eletivas adiadas;
3. `f3` (`mean_surge_bed_days`): custo operacional de mobilização (leitos-dia de surge ativos).

Protocolo anti-viés em 2 estágios (Common Random Numbers):
- Estágio 1 (Busca NSGA-II): `R_search` replicações CRN (`seeds = search_seed_start..`);
- Estágio 2 (Auditoria da Fronteira): `R_audit` replicações CRN estritamente disjuntas
  (`seeds = audit_seed_start..`) comparando cada política da fronteira contra os baselines
  `reactive_h0` e `proactive_h2_v1_fixed`.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

from previsao_leitos.domain import DataContractError
from previsao_leitos.readiness.simulator import (
    CrnScenarioStream,
    HospitalReadinessConfig,
    PolicyMode,
    _resolve_exogenous_flood_days,
    evaluate_policy_on_crn,
    generate_crn_scenarios,
)

# Limites do espaço de decisão: [trigger_occ, hysteresis_close, elective_deferral, surge_beds_frac]
_LOWER_BOUNDS = np.array([0.78, 0.65, 0.05, 0.05], dtype=np.float64)
_UPPER_BOUNDS = np.array([1.15, 0.92, 0.90, 0.30], dtype=np.float64)


@dataclass(frozen=True)
class ContingencyPolicyDecision:
    """Vetor de decisão de uma política proativa `h = 2` de contingência hospitalar."""

    trigger_occ_threshold: float
    hysteresis_close_threshold: float
    elective_deferral_frac: float
    surge_beds_frac: float
    surge_beds: int


@dataclass(frozen=True)
class ParetoPolicyEvaluation:
    """Métricas auditadas (Estágio 2 CRN independente) de uma política de contingência."""

    label: str
    policy_mode: PolicyMode
    decision: ContingencyPolicyDecision
    mean_peak_queue: float
    mean_overload_days: float
    mean_deferred_electives: float
    mean_surge_bed_days: float
    mean_wait_hours: float
    queue_reduction_vs_reactive_pct: float
    elective_reduction_vs_v1_pct: float
    surge_cost_reduction_vs_v1_pct: float
    dominates_v1: bool


@dataclass(frozen=True)
class ParetoOptimizationResult:
    """Resultado completo da otimização NSGA-II + auditoria independente em 2 estágios CRN."""

    region_label: str
    r_search: int
    r_audit: int
    pop_size: int
    n_gen: int
    elapsed_seconds: float
    reactive_baseline: ParetoPolicyEvaluation
    v1_fixed_baseline: ParetoPolicyEvaluation
    pareto_front: tuple[ParetoPolicyEvaluation, ...]
    dominating_v1_count: int
    recommended_policy: ParetoPolicyEvaluation
    front_df: pl.DataFrame


def _decode_decision(x: np.ndarray, base_beds: int) -> ContingencyPolicyDecision:
    trig = float(np.clip(x[0], _LOWER_BOUNDS[0], _UPPER_BOUNDS[0]))
    # Garante histerese válida (fechamento <= acionamento - 0.03)
    close_hi = min(float(_UPPER_BOUNDS[1]), trig - 0.03)
    close = float(np.clip(x[1], float(_LOWER_BOUNDS[1]), max(float(_LOWER_BOUNDS[1]), close_hi)))
    defer = float(np.clip(x[2], _LOWER_BOUNDS[2], _UPPER_BOUNDS[2]))
    s_frac = float(np.clip(x[3], _LOWER_BOUNDS[3], _UPPER_BOUNDS[3]))
    s_beds = max(1, round(base_beds * s_frac))
    return ContingencyPolicyDecision(
        trigger_occ_threshold=trig,
        hysteresis_close_threshold=close,
        elective_deferral_frac=defer,
        surge_beds_frac=s_frac,
        surge_beds=s_beds,
    )


def _make_config_from_decision(
    base_config: HospitalReadinessConfig, decision: ContingencyPolicyDecision
) -> HospitalReadinessConfig:
    return HospitalReadinessConfig(
        base_beds=base_config.base_beds,
        surge_beds=decision.surge_beds,
        los_days=base_config.los_days,
        elective_share=base_config.elective_share,
        elective_deferral_frac=decision.elective_deferral_frac,
        surge_lead_time_days=base_config.surge_lead_time_days,
        polder_hospital_loss_frac=base_config.polder_hospital_loss_frac,
        trigger_occ_threshold=decision.trigger_occ_threshold,
        hysteresis_close_threshold=decision.hysteresis_close_threshold,
        region_label=base_config.region_label,
    )


_WORKER_CTX: dict[str, Any] = {}


def _init_pareto_worker(
    base_cfg: HospitalReadinessConfig,
    n_weeks: int,
    streams: tuple[CrnScenarioStream, ...],
    flood_days: set[int],
    pred_p90: tuple[float, ...],
) -> None:
    """Inicializa estado imutável do worker uma única vez (spawn-safe com Polars/Rayon)."""
    _WORKER_CTX["base_cfg"] = base_cfg
    _WORKER_CTX["sched"] = pl.DataFrame({"_w": list(range(n_weeks))})
    _WORKER_CTX["streams"] = streams
    _WORKER_CTX["flood_days"] = flood_days
    _WORKER_CTX["pred_p90"] = pred_p90


def _eval_candidate_from_ctx(
    item: tuple[tuple[float, float, float, float], PolicyMode],
) -> tuple[float, float, float, float, float]:
    """Avalia candidato usando o contexto pré-carregado no worker (IPC mínimo de 4 floats)."""
    x_tuple, mode = item
    return _eval_candidate_worker(
        (
            np.asarray(x_tuple, dtype=np.float64),
            _WORKER_CTX["base_cfg"],
            mode,
            _WORKER_CTX["sched"],
            _WORKER_CTX["streams"],
            _WORKER_CTX["flood_days"],
            _WORKER_CTX["pred_p90"],
        )
    )


def _eval_candidate_worker(
    payload: tuple[
        np.ndarray,
        HospitalReadinessConfig,
        PolicyMode,
        pl.DataFrame,
        tuple[CrnScenarioStream, ...],
        set[int],
        tuple[float, ...],
    ],
) -> tuple[float, float, float, float, float]:
    """Função top-level serializável para avaliação paralela multi-core no ProcessPoolExecutor."""
    x_vec, base_cfg, mode, sched, streams, flood_days, pred_p90 = payload
    dec = _decode_decision(x_vec, base_cfg.base_beds)
    cfg = _make_config_from_decision(base_cfg, dec)

    peaks: list[float] = []
    overloads: list[float] = []
    deferred: list[float] = []
    surge_days: list[float] = []
    waits: list[float] = []
    for st in streams:
        res = evaluate_policy_on_crn(
            weekly_schedule=sched,
            config=cfg,
            policy_mode=mode,
            stream=st,
            flood_days_precomputed=flood_days,
            pred_p90_weekly_precomputed=pred_p90,
        )
        peaks.append(float(res.peak_queue_patients))
        overloads.append(float(res.overload_days_gt95))
        deferred.append(float(res.deferred_electives))
        surge_days.append(float(res.surge_bed_days_active))
        waits.append(float(res.mean_wait_hours))

    return (
        float(np.mean(peaks)),
        float(np.mean(deferred)),
        float(np.mean(surge_days)),
        float(np.mean(overloads)),
        float(np.mean(waits)),
    )


def _non_dominated_indices(objectives: np.ndarray) -> list[int]:
    """Filtra índices estritamente não-dominados (minimização em todos os objetivos)."""
    n = objectives.shape[0]
    keep: list[int] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if np.all(objectives[j] <= objectives[i] + 1e-9) and np.any(
                objectives[j] < objectives[i] - 1e-9
            ):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


class _ContingencyNsga2Problem(Problem):  # type: ignore[misc]
    """Problema pymoo vetorizado (3 objetivos) para busca NSGA-II sobre replicações CRN."""

    def __init__(
        self,
        *,
        weekly_schedule: pl.DataFrame,
        base_config: HospitalReadinessConfig,
        search_streams: tuple[CrnScenarioStream, ...],
        flood_days: set[int],
        pred_p90: tuple[float, ...],
        executor: ProcessPoolExecutor | None = None,
    ) -> None:
        super().__init__(n_var=4, n_obj=3, xl=_LOWER_BOUNDS, xu=_UPPER_BOUNDS)
        self._sched = weekly_schedule
        self._cfg = base_config
        self._streams = search_streams
        self._flood_days = flood_days
        self._pred_p90 = pred_p90
        self._executor = executor

    def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        mode: PolicyMode = "proactive_h2"
        if self._executor is not None:
            items = [
                ((float(r[0]), float(r[1]), float(r[2]), float(r[3])), mode)
                for r in x
            ]
            results = list(self._executor.map(_eval_candidate_from_ctx, items))
        else:
            payloads = [
                (
                    np.asarray(row, dtype=np.float64),
                    self._cfg,
                    mode,
                    self._sched,
                    self._streams,
                    self._flood_days,
                    self._pred_p90,
                )
                for row in x
            ]
            results = [_eval_candidate_worker(p) for p in payloads]
        f_mat = np.array([[r[0], r[1], r[2]] for r in results], dtype=np.float64)
        out["F"] = f_mat


def _build_seeded_initial_population(
    base_config: HospitalReadinessConfig, pop_size: int, seed: int
) -> np.ndarray:
    """Inicializa população NSGA-II incluindo a política fixa V1 e vizinhança diversificada."""
    rng = np.random.default_rng(seed)
    v1_frac = float(np.clip(base_config.surge_beds / max(1, base_config.base_beds), 0.05, 0.30))
    anchors = np.array(
        [
            [
                base_config.trigger_occ_threshold,
                base_config.hysteresis_close_threshold,
                base_config.elective_deferral_frac,
                v1_frac,
            ],
            [0.95, 0.85, 0.35, v1_frac],
            [1.00, 0.86, 0.25, max(0.08, v1_frac * 0.85)],
            [0.92, 0.84, 0.45, min(0.28, v1_frac * 1.10)],
            [1.05, 0.88, 0.15, max(0.06, v1_frac * 0.70)],
            [0.88, 0.80, 0.60, min(0.28, v1_frac * 1.15)],
        ],
        dtype=np.float64,
    )
    anchors = np.clip(anchors, _LOWER_BOUNDS, _UPPER_BOUNDS)
    if pop_size <= anchors.shape[0]:
        return anchors[:pop_size]
    rand_rows = rng.uniform(_LOWER_BOUNDS, _UPPER_BOUNDS, size=(pop_size - anchors.shape[0], 4))
    return np.vstack([anchors, rand_rows])


def optimize_contingency_pareto_front(
    weekly_schedule: pl.DataFrame,
    base_config: HospitalReadinessConfig,
    *,
    r_search: int = 6,
    r_audit: int = 30,
    pop_size: int = 24,
    n_gen: int = 12,
    n_workers: int = 1,
    seed: int = 42,
    flood_indicator_col: str | None = "em_cheia_hidro",
) -> ParetoOptimizationResult:
    """Executa busca multiobjetivo NSGA-II (Estágio 1) e auditoria CRN independente (Estágio 2)."""
    if r_search < 1 or r_audit < 1:
        raise DataContractError("r_search e r_audit devem ser >= 1.")
    if pop_size < 4 or n_gen < 1:
        raise DataContractError("pop_size deve ser >= 4 e n_gen >= 1.")
    if n_workers < 1:
        raise DataContractError("n_workers deve ser >= 1.")

    t0 = time.perf_counter()
    flood_days = _resolve_exogenous_flood_days(
        weekly_schedule, flood_indicator_col=flood_indicator_col
    )
    pred_p90 = tuple(float(v) for v in weekly_schedule["previsao_p90_semanal"].to_list())

    # Estágio 1 vs Estágio 2: seeds estritamente disjuntas (zero viés de otimização na auditoria)
    search_seeds = [seed * 100 + i for i in range(r_search)]
    audit_seeds = [seed * 100 + 10_000 + i for i in range(r_audit)]
    search_streams = generate_crn_scenarios(weekly_schedule, base_config, seeds=search_seeds)
    audit_streams = generate_crn_scenarios(weekly_schedule, base_config, seeds=audit_seeds)

    init_pop = _build_seeded_initial_population(base_config, pop_size=pop_size, seed=seed)
    algo = NSGA2(
        pop_size=pop_size,
        sampling=init_pop if init_pop.shape[0] == pop_size else FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    if n_workers > 1:
        # `spawn` evita deadlock com o thread-pool Rust/Rayon do Polars no Linux (`fork`).
        mp_ctx = multiprocessing.get_context("spawn")
        n_weeks = weekly_schedule.height
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp_ctx,
            initializer=_init_pareto_worker,
            initargs=(base_config, n_weeks, search_streams, flood_days, pred_p90),
        ) as search_pool:
            problem = _ContingencyNsga2Problem(
                weekly_schedule=weekly_schedule,
                base_config=base_config,
                search_streams=search_streams,
                flood_days=flood_days,
                pred_p90=pred_p90,
                executor=search_pool,
            )
            res = minimize(problem, algo, ("n_gen", n_gen), seed=seed, verbose=False)
            cand_x = np.atleast_2d(np.asarray(res.X, dtype=np.float64))

        # Estágio 2: pool dedicado com R_audit streams CRN disjuntos
        v1_x = init_pop[0]
        v1_t = (float(v1_x[0]), float(v1_x[1]), float(v1_x[2]), float(v1_x[3]))
        audit_items: list[tuple[tuple[float, float, float, float], PolicyMode]] = [
            (v1_t, "reactive_h0"),
            (v1_t, "proactive_h2"),
            *[
                ((float(r[0]), float(r[1]), float(r[2]), float(r[3])), "proactive_h2")
                for r in cand_x
            ],
        ]
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp_ctx,
            initializer=_init_pareto_worker,
            initargs=(base_config, n_weeks, audit_streams, flood_days, pred_p90),
        ) as audit_pool:
            audit_metrics = list(audit_pool.map(_eval_candidate_from_ctx, audit_items))
    else:
        problem = _ContingencyNsga2Problem(
            weekly_schedule=weekly_schedule,
            base_config=base_config,
            search_streams=search_streams,
            flood_days=flood_days,
            pred_p90=pred_p90,
            executor=None,
        )
        res = minimize(problem, algo, ("n_gen", n_gen), seed=seed, verbose=False)
        cand_x = np.atleast_2d(np.asarray(res.X, dtype=np.float64))
        v1_x = init_pop[0]
        m_react: PolicyMode = "reactive_h0"
        m_pro: PolicyMode = "proactive_h2"
        audit_payloads: list[
            tuple[
                np.ndarray,
                HospitalReadinessConfig,
                PolicyMode,
                pl.DataFrame,
                tuple[CrnScenarioStream, ...],
                set[int],
                tuple[float, ...],
            ]
        ] = [
            (
                v1_x,
                base_config,
                m_react,
                weekly_schedule,
                audit_streams,
                flood_days,
                pred_p90,
            ),
            (
                v1_x,
                base_config,
                m_pro,
                weekly_schedule,
                audit_streams,
                flood_days,
                pred_p90,
            ),
            *[
                (
                    np.asarray(row, dtype=np.float64),
                    base_config,
                    m_pro,
                    weekly_schedule,
                    audit_streams,
                    flood_days,
                    pred_p90,
                )
                for row in cand_x
            ],
        ]
        audit_metrics = [_eval_candidate_worker(p) for p in audit_payloads]

    react_m = audit_metrics[0]
    v1_m = audit_metrics[1]
    cand_m = audit_metrics[2:]

    v1_dec = _decode_decision(v1_x, base_config.base_beds)

    def _to_eval(
        label: str,
        mode: PolicyMode,
        dec: ContingencyPolicyDecision,
        m: tuple[float, float, float, float, float],
    ) -> ParetoPolicyEvaluation:
        pk, df_e, sg_d, ov_d, wt_h = m
        q_red = 100.0 * (react_m[0] - pk) / react_m[0] if react_m[0] > 1e-9 else 0.0
        el_red = 100.0 * (v1_m[1] - df_e) / v1_m[1] if v1_m[1] > 1e-9 else 0.0
        sg_red = 100.0 * (v1_m[2] - sg_d) / v1_m[2] if v1_m[2] > 1e-9 else 0.0
        better_count = (
            int(pk < v1_m[0] - 1e-6) + int(df_e < v1_m[1] - 1e-6) + int(sg_d < v1_m[2] - 1e-6)
        )
        no_worse = (pk <= v1_m[0] + 1e-6) and (df_e <= v1_m[1] + 1e-6) and (sg_d <= v1_m[2] + 1e-6)
        return ParetoPolicyEvaluation(
            label=label,
            policy_mode=mode,
            decision=dec,
            mean_peak_queue=round(pk, 2),
            mean_overload_days=round(ov_d, 2),
            mean_deferred_electives=round(df_e, 1),
            mean_surge_bed_days=round(sg_d, 1),
            mean_wait_hours=round(wt_h, 2),
            queue_reduction_vs_reactive_pct=round(q_red, 2),
            elective_reduction_vs_v1_pct=round(el_red, 2),
            surge_cost_reduction_vs_v1_pct=round(sg_red, 2),
            dominates_v1=bool(no_worse and better_count >= 1),
        )

    reactive_eval = _to_eval("Reactive_H0", "reactive_h0", v1_dec, react_m)
    v1_eval = _to_eval("V1_Fixed_Proactive_H2", "proactive_h2", v1_dec, v1_m)

    # Dedup e filtro estrito de não-dominância sobre as métricas auditadas do Estágio 2
    cand_obj = np.array([[m[0], m[1], m[2]] for m in cand_m], dtype=np.float64)
    nd_idxs = _non_dominated_indices(cand_obj)
    front_evals: list[ParetoPolicyEvaluation] = []
    seen_keys: set[tuple[float, float, float]] = set()
    for rank_i, idx in enumerate(nd_idxs, start=1):
        key = (round(cand_m[idx][0], 2), round(cand_m[idx][1], 1), round(cand_m[idx][2], 1))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        dec = _decode_decision(cand_x[idx], base_config.base_beds)
        front_evals.append(_to_eval(f"Pareto_{rank_i:02d}", "proactive_h2", dec, cand_m[idx]))

    front_evals.sort(
        key=lambda e: (e.mean_peak_queue, e.mean_deferred_electives, e.mean_surge_bed_days)
    )

    # Joelho recomendado: prioriza políticas que dominam V1 em 3 objetivos; se não houver,
    # prioriza políticas seguras (fila <= V1 e eletivas <= V1) com menor distância à utopia.
    pool_rec = (
        [e for e in front_evals if e.dominates_v1]
        or [
            e
            for e in front_evals
            if e.mean_peak_queue <= v1_eval.mean_peak_queue + 1e-6
            and e.mean_deferred_electives <= v1_eval.mean_deferred_electives + 1e-6
        ]
        or [e for e in front_evals if e.mean_peak_queue <= v1_eval.mean_peak_queue + 1e-6]
        or front_evals
    )
    obj_pool = np.array(
        [[e.mean_peak_queue, e.mean_deferred_electives, e.mean_surge_bed_days] for e in pool_rec],
        dtype=np.float64,
    )
    mins = np.min(obj_pool, axis=0)
    spans = np.maximum(1e-6, np.max(obj_pool, axis=0) - mins)
    norm_d = np.linalg.norm((obj_pool - mins) / spans, axis=1)
    recommended = pool_rec[int(np.argmin(norm_d))]

    all_rows = [reactive_eval, v1_eval, *front_evals]
    front_df = pl.DataFrame(
        [
            {
                "politica": e.label,
                "trigger_occ": round(e.decision.trigger_occ_threshold, 3),
                "close_occ": round(e.decision.hysteresis_close_threshold, 3),
                "defer_frac": round(e.decision.elective_deferral_frac, 3),
                "surge_beds": e.decision.surge_beds,
                "pico_fila_medio": e.mean_peak_queue,
                "dias_sobrecarga": e.mean_overload_days,
                "eletivas_adiadas": e.mean_deferred_electives,
                "leitos_dia_surge": e.mean_surge_bed_days,
                "red_fila_vs_h0_pct": e.queue_reduction_vs_reactive_pct,
                "red_elet_vs_v1_pct": e.elective_reduction_vs_v1_pct,
                "red_surge_vs_v1_pct": e.surge_cost_reduction_vs_v1_pct,
                "domina_v1": e.dominates_v1,
            }
            for e in all_rows
        ]
    )

    return ParetoOptimizationResult(
        region_label=base_config.region_label,
        r_search=r_search,
        r_audit=r_audit,
        pop_size=pop_size,
        n_gen=n_gen,
        elapsed_seconds=round(time.perf_counter() - t0, 2),
        reactive_baseline=reactive_eval,
        v1_fixed_baseline=v1_eval,
        pareto_front=tuple(front_evals),
        dominating_v1_count=sum(1 for e in front_evals if e.dominates_v1),
        recommended_policy=recommended,
        front_df=front_df,
    )
