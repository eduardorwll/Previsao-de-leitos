"""Simulador de filas hospitalares em eventos discretos (SimPy) para planos de contingência."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl
import simpy

from previsao_leitos.domain import DataContractError

PolicyMode = Literal["reactive_h0", "proactive_h2"]


@dataclass(frozen=True)
class HospitalReadinessConfig:
    """Parâmetros físicos e operacionais da rede de hospitais-escola em uma região."""

    base_beds: int
    surge_beds: int
    los_days: float
    elective_share: float = 0.25
    elective_deferral_frac: float = 0.75
    surge_lead_time_days: int = 2
    polder_hospital_loss_frac: float = 0.0
    region_label: str = "Região Hospitalar RS"

    def __post_init__(self) -> None:
        if self.base_beds <= 0:
            raise DataContractError(f"base_beds deve ser > 0 (recebido {self.base_beds}).")
        if self.surge_beds < 0:
            raise DataContractError(f"surge_beds deve ser >= 0 (recebido {self.surge_beds}).")
        if self.los_days <= 0.0:
            raise DataContractError(f"los_days deve ser > 0 (recebido {self.los_days}).")
        if not (0.0 <= self.elective_share <= 1.0):
            raise DataContractError(
                f"elective_share deve estar em [0, 1] (recebido {self.elective_share})."
            )
        if not (0.0 <= self.elective_deferral_frac <= 1.0):
            raise DataContractError(
                f"elective_deferral_frac deve estar em [0, 1] "
                f"(recebido {self.elective_deferral_frac})."
            )
        if self.surge_lead_time_days < 0:
            raise DataContractError(
                f"surge_lead_time_days deve ser >= 0 (recebido {self.surge_lead_time_days})."
            )
        if not (0.0 <= self.polder_hospital_loss_frac < 0.8):
            raise DataContractError(
                f"polder_hospital_loss_frac fora de [0, 0.8): {self.polder_hospital_loss_frac}"
            )


@dataclass(frozen=True)
class PolicySimulationResult:
    """Trajetória diária e indicadores-chave de uma política de contingência no SimPy."""

    policy_mode: PolicyMode
    daily_df: pl.DataFrame
    peak_queue_patients: int
    overload_days_gt95: int
    mean_wait_hours: float
    deferred_electives: int


@dataclass(frozen=True)
class CounterfactualReadinessComparison:
    """Comparação contrafactual (Common Random Numbers): h=2 antecipado vs. h=0 reativo."""

    region_label: str
    reactive_h0: PolicySimulationResult
    proactive_h2: PolicySimulationResult
    queue_peak_reduction_pct: float
    overload_days_saved: int


def _resolve_exogenous_flood_days(
    weekly_schedule: pl.DataFrame,
    *,
    flood_indicator_col: str | None = "em_cheia_hidro",
) -> set[int]:
    """Determina dias de perda de pôlder usando indicador hidrológico exógeno quando disponível.

    Elimina circularidade com a demanda hospitalar (`Bug J`): se `flood_indicator_col`
    estiver presente em `weekly_schedule` (booleano ou 0/1), usa-o diretamente; caso
    contrário, recorre ao fallback de compatibilidade sobre `chegadas_semanais`.
    """
    flood_days: set[int] = set()
    if flood_indicator_col is not None and flood_indicator_col in weekly_schedule.columns:
        flags = weekly_schedule[flood_indicator_col].cast(pl.Float64).fill_null(0.0).to_list()
        for wk_idx, flag in enumerate(flags):
            if float(flag) > 0.5:
                for d in range(wk_idx * 7, (wk_idx + 1) * 7):
                    flood_days.add(d)
        return flood_days

    arrivals_weekly = weekly_schedule["chegadas_semanais"].to_list()
    median_arrivals = float(np.median(arrivals_weekly)) if arrivals_weekly else 0.0
    for wk_idx, arr_w in enumerate(arrivals_weekly):
        if float(arr_w) > 1.25 * max(median_arrivals, 1.0):
            for d in range(wk_idx * 7, (wk_idx + 1) * 7):
                flood_days.add(d)
    return flood_days


def _run_single_simpy_policy(
    *,
    weekly_schedule: pl.DataFrame,
    config: HospitalReadinessConfig,
    policy_mode: PolicyMode,
    daily_arrivals_urgent: list[int],
    daily_arrivals_elective: list[int],
    daily_los_samples: list[list[float]],
    flood_indicator_col: str | None = "em_cheia_hidro",
) -> PolicySimulationResult:
    """Executa simulação de eventos discretos em `simpy.Environment` com stream compartilhado."""
    n_weeks = weekly_schedule.height
    n_days = n_weeks * 7
    pred_p90_weekly = weekly_schedule["previsao_p90_semanal"].to_list()

    # Capacidade máxima no recurso SimPy = base + surge; leitos indisponíveis são bloqueados
    max_capacity = config.base_beds + config.surge_beds
    env = simpy.Environment()
    beds = simpy.PriorityResource(env, capacity=max_capacity)

    # Controle de tokens de bloqueio para modelar surge lead-time e perda em área aterrada/pôlder
    surge_active = False
    surge_requested_day: int | None = None
    polder_loss_beds = round(config.base_beds * config.polder_hospital_loss_frac)

    # Dias de cheia hidrológica exógena (ativa perda em pôlder sem proxy circular de demanda)
    flood_days = _resolve_exogenous_flood_days(
        weekly_schedule,
        flood_indicator_col=flood_indicator_col,
    )

    wait_times_days: list[float] = []
    deferred_count = 0
    in_bed_patients = 0
    waiting_patients = 0
    daily_records: list[dict[str, object]] = []

    def _patient_process(
        stay_days: float, priority: int
    ) -> Generator[simpy.Event, Any, None]:
        nonlocal in_bed_patients, waiting_patients
        arr_t = env.now
        waiting_patients += 1
        with beds.request(priority=priority) as req:
            yield req
            waiting_patients -= 1
            in_bed_patients += 1
            wait_times_days.append(float(env.now - arr_t))
            yield env.timeout(stay_days)
            in_bed_patients -= 1

    def _capacity_blocker(duration: float) -> Generator[simpy.Event, Any, None]:
        with beds.request(priority=-10) as req:
            yield req
            yield env.timeout(duration)

    def _hospital_driver() -> Generator[simpy.Event, Any, None]:
        nonlocal surge_active, surge_requested_day, deferred_count
        # Capacidade base semanal teórica (Lei de Little: base_beds * 7 / los_days)
        base_weekly_capacity = config.base_beds * 7.0 / config.los_days

        for day in range(n_days):
            wk_idx = day // 7
            pred_p90 = float(pred_p90_weekly[wk_idx])
            # Em modo proativo (h=2), antecipa o pedido de surge `surge_lead_time_days` antes
            lookahead_wk = min(n_weeks - 1, (day + config.surge_lead_time_days) // 7)
            pred_p90_lookahead = float(pred_p90_weekly[lookahead_wk])
            in_flood_day = day in flood_days
            active_polder_loss = polder_loss_beds if in_flood_day else 0

            # Capacidade efetiva antes da atualização de estado de hoje (para detectar crise)
            eff_beds_prev = max(
                1,
                config.base_beds - active_polder_loss + (config.surge_beds if surge_active else 0),
            )
            occ_now = in_bed_patients / eff_beds_prev

            # Política de acionamento de contingência com histerese segura sobre leitos BASE
            base_eff = max(1, config.base_beds - active_polder_loss)
            crisis_now = occ_now >= 0.95 or waiting_patients > 0
            forecast_alert_now = pred_p90 >= 0.90 * base_weekly_capacity
            forecast_alert_lead = pred_p90_lookahead >= 0.90 * base_weekly_capacity
            forecast_alert = forecast_alert_now or forecast_alert_lead
            below_base_safe = in_bed_patients < 0.85 * base_eff and waiting_patients == 0
            if policy_mode == "proactive_h2":
                trigger_on = forecast_alert or crisis_now
                safe_to_close = (not forecast_alert) and below_base_safe
            else:
                trigger_on = crisis_now
                safe_to_close = below_base_safe

            if trigger_on and not surge_active:
                if surge_requested_day is None:
                    surge_requested_day = day
                if day - surge_requested_day >= config.surge_lead_time_days:
                    surge_active = True
            elif not trigger_on and not surge_active:
                # Limpa pedido pendente se o alerta cessou antes de concluir o lead_time
                surge_requested_day = None
            elif safe_to_close and surge_active:
                surge_active = False
                surge_requested_day = None

            # Recalcula capacidade efetiva de hoje já refletindo ativação/desativação de surge
            eff_beds = max(
                1,
                config.base_beds - active_polder_loss + (config.surge_beds if surge_active else 0),
            )

            # Bloqueia leitos não-ativos durante o dia inteiro (surge inativo + perda por pôlder)
            blocked_today = (0 if surge_active else config.surge_beds) + active_polder_loss
            for _ in range(blocked_today):
                env.process(_capacity_blocker(1.0))

            # Chegadas urgentes (prioridade 1) + eletivas (prioridade 2, passíveis de adiamento)
            n_urg = daily_arrivals_urgent[day]
            n_elec_raw = daily_arrivals_elective[day]
            if trigger_on or surge_active:
                n_elec = round(n_elec_raw * (1.0 - config.elective_deferral_frac))
                deferred_count += max(0, n_elec_raw - n_elec)
            else:
                n_elec = n_elec_raw

            los_list = daily_los_samples[day]
            for i in range(n_urg):
                stay = los_list[i % len(los_list)] if los_list else config.los_days
                env.process(_patient_process(stay, priority=1))
            for i in range(n_elec):
                stay = los_list[(n_urg + i) % len(los_list)] if los_list else config.los_days
                env.process(_patient_process(stay, priority=2))

            # Amostra o estado no meio do dia (com bloqueios ativos) e conclui o passo diário
            yield env.timeout(0.5)
            occ_pct = 100.0 * min(1.25, in_bed_patients / max(1, eff_beds))
            daily_records.append(
                {
                    "dia": day + 1,
                    "semana_rel": wk_idx + 1,
                    "leitos_ativos": eff_beds,
                    "pacientes_internados": in_bed_patients,
                    "ocupacao_pct": occ_pct,
                    "fila_espera_pacientes": waiting_patients,
                }
            )
            yield env.timeout(0.5)

    env.process(_hospital_driver())
    env.run(until=n_days)

    daily_df = pl.DataFrame(daily_records)
    peak_q = int(daily_df.select(pl.col("fila_espera_pacientes").max()).item())
    overload_days = int(
        daily_df.filter(
            (pl.col("ocupacao_pct") >= 95.0) | (pl.col("fila_espera_pacientes") > 0)
        ).height
    )
    mean_wait_h = float(np.mean(wait_times_days) * 24.0) if wait_times_days else 0.0

    return PolicySimulationResult(
        policy_mode=policy_mode,
        daily_df=daily_df,
        peak_queue_patients=peak_q,
        overload_days_gt95=overload_days,
        mean_wait_hours=mean_wait_h,
        deferred_electives=deferred_count,
    )


def simulate_counterfactual_readiness(
    weekly_schedule: pl.DataFrame,
    config: HospitalReadinessConfig,
    *,
    seed: int = 42,
    flood_indicator_col: str | None = "em_cheia_hidro",
) -> CounterfactualReadinessComparison:
    """Executa comparação contrafactual `proactive_h2` vs. `reactive_h0` com Common Random Numbers.

    `weekly_schedule` deve conter colunas `chegadas_semanais` e `previsao_p90_semanal`
    (e opcionalmente `em_cheia_hidro` para ativação exógena de perda de pôlder).
    """
    required = ("chegadas_semanais", "previsao_p90_semanal")
    missing = [c for c in required if c not in weekly_schedule.columns]
    if missing:
        raise DataContractError(f"Colunas ausentes em weekly_schedule: {missing}")
    if weekly_schedule.is_empty():
        raise DataContractError("weekly_schedule vazio em simulate_counterfactual_readiness.")

    rng = np.random.default_rng(seed)
    daily_urgent: list[int] = []
    daily_elective: list[int] = []
    daily_los: list[list[float]] = []

    for arr_w in weekly_schedule["chegadas_semanais"].to_list():
        daily_rate = max(0.0, float(arr_w)) / 7.0
        for _ in range(7):
            n_urg = int(rng.poisson(daily_rate * (1.0 - config.elective_share)))
            n_elec = int(rng.poisson(daily_rate * config.elective_share))
            daily_urgent.append(n_urg)
            daily_elective.append(n_elec)
            n_tot = max(1, n_urg + n_elec)
            # LOS com distribuição Gamma (forma=2 -> cauda direita realista)
            samples = rng.gamma(shape=2.0, scale=config.los_days / 2.0, size=n_tot)
            daily_los.append([max(0.5, float(x)) for x in samples])

    reactive = _run_single_simpy_policy(
        weekly_schedule=weekly_schedule,
        config=config,
        policy_mode="reactive_h0",
        daily_arrivals_urgent=daily_urgent,
        daily_arrivals_elective=daily_elective,
        daily_los_samples=daily_los,
        flood_indicator_col=flood_indicator_col,
    )
    proactive = _run_single_simpy_policy(
        weekly_schedule=weekly_schedule,
        config=config,
        policy_mode="proactive_h2",
        daily_arrivals_urgent=daily_urgent,
        daily_arrivals_elective=daily_elective,
        daily_los_samples=daily_los,
        flood_indicator_col=flood_indicator_col,
    )

    if reactive.peak_queue_patients > 0:
        red_pct = (
            100.0
            * (reactive.peak_queue_patients - proactive.peak_queue_patients)
            / reactive.peak_queue_patients
        )
    else:
        red_pct = 0.0

    return CounterfactualReadinessComparison(
        region_label=config.region_label,
        reactive_h0=reactive,
        proactive_h2=proactive,
        queue_peak_reduction_pct=red_pct,
        overload_days_saved=reactive.overload_days_gt95 - proactive.overload_days_gt95,
    )
