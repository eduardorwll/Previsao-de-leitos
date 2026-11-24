"""Tabela auditável de gatilhos observáveis para pesquisa de prontidão hospitalar."""

from __future__ import annotations

from dataclasses import dataclass

_DISCLAIMER = (
    "Modelo exclusivamente de pesquisa e planejamento agregado "
    "(Not for patient care or live clinical use)."
)


@dataclass(frozen=True)
class ObservableTriggerAssessment:
    """Classificação agregada de estresse hidrológico-assistencial (uso em pesquisa)."""

    level: str
    occupancy_ratio_p90: float
    rationale: str
    disclaimer: str = _DISCLAIMER


def evaluate_observable_triggers(
    *,
    predicted_p90_beds: float,
    base_beds: int,
    upstream_api_mm: float,
    remanso_idx: float = 0.0,
) -> ObservableTriggerAssessment:
    """Classifica nível de estresse agregado a partir da previsão P90 e indicadores físicos."""
    ratio = predicted_p90_beds / max(1, base_beds)
    hydro_stress = upstream_api_mm >= 180.0 or remanso_idx >= 250.0

    if ratio >= 1.05 or (ratio >= 0.95 and hydro_stress):
        return ObservableTriggerAssessment(
            level="VERMELHO",
            occupancy_ratio_p90=ratio,
            rationale=(
                "Previsão P90 excede capacidade base sob estresse hidrológico/remanso "
                "(cenário de contingência máxima em simulação)."
            ),
        )
    if ratio >= 0.90 or (ratio >= 0.85 and hydro_stress):
        return ObservableTriggerAssessment(
            level="LARANJA",
            occupancy_ratio_p90=ratio,
            rationale=(
                "Previsão P90 acima de 90% da oferta base ou 85% sob onda a montante "
                "(cenário de pré-acionamento de eletivas e leitos extras h=2 sem.)."
            ),
        )
    if ratio >= 0.80:
        return ObservableTriggerAssessment(
            level="AMARELO",
            occupancy_ratio_p90=ratio,
            rationale="Previsão P90 entre 80% e 90% da oferta base (monitoramento reforçado).",
        )
    return ObservableTriggerAssessment(
        level="VERDE",
        occupancy_ratio_p90=ratio,
        rationale="Previsão P90 abaixo de 80% da oferta base (regime basal).",
    )
