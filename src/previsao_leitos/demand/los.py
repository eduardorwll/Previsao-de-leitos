"""Estimador causal de Tempo de Permanência (LOS) por especialidade e conversão em leitos/dia."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from previsao_leitos.domain import DataContractError


class SpecialtyLosEstimator:
    """Estima LOS médio causal (<= train_end_year) por especialidade e aplica a Lei de Little."""

    def __init__(self, stats_df: pl.DataFrame) -> None:
        self._stats = stats_df
        self._los_map: dict[str, float] = {
            str(r["especialidade"]): float(r["los_medio_dias"])
            for r in stats_df.iter_rows(named=True)
        }
        self._icu_map: dict[str, float] = {
            str(r["especialidade"]): float(r["diarias_uti_medias"])
            for r in stats_df.iter_rows(named=True)
        }

    @classmethod
    def fit_causal(
        cls, sih_df: pl.DataFrame, *, train_end_year: int = 2022
    ) -> SpecialtyLosEstimator:
        """Ajusta LOS médio estritamente sobre `ano_epi <= train_end_year` (zero vazamento)."""
        causal = sih_df.filter(pl.col("ano_epi") <= train_end_year)
        if causal.is_empty():
            raise DataContractError(f"Sem registros SIH até train_end_year={train_end_year}.")

        agg = (
            causal.group_by("especialidade")
            .agg(
                pl.col("internacoes").sum().cast(pl.Float64).alias("internacoes_total"),
                pl.col("dias_permanencia_total").sum().cast(pl.Float64).alias("dias_perm_total"),
                pl.col("diarias_uti_total").sum().cast(pl.Float64).alias("diarias_uti_total"),
            )
            .filter(pl.col("internacoes_total") > 0.0)
            .with_columns(
                (pl.col("dias_perm_total") / pl.col("internacoes_total")).alias("los_medio_dias"),
                (pl.col("diarias_uti_total") / pl.col("internacoes_total")).alias(
                    "diarias_uti_medias"
                ),
            )
            .sort("especialidade")
        )
        return cls(agg)

    @property
    def summary_table(self) -> pl.DataFrame:
        """Retorna tabela sumarizada de LOS por especialidade."""
        return self._stats.clone()

    def mean_los_days(self, especialidade: str) -> float:
        """Retorna LOS médio total (dias de permanência por internação) para a especialidade."""
        if especialidade not in self._los_map:
            raise DataContractError(f"Especialidade ausente no estimador LOS: {especialidade!r}")
        return self._los_map[especialidade]

    def mean_icu_days(self, especialidade: str) -> float:
        """Retorna diárias médias de UTI por internação para a especialidade."""
        if especialidade not in self._icu_map:
            raise DataContractError(f"Especialidade ausente no estimador LOS: {especialidade!r}")
        return self._icu_map[especialidade]

    @staticmethod
    def weekly_bed_days_to_occupied_beds(weekly_bed_days: float) -> float:
        """Converte diárias hospitalares semanais em média diária de leitos ocupados (L = D/7)."""
        return max(0.0, weekly_bed_days) / 7.0

    def admissions_to_occupied_beds(self, weekly_admissions: float, especialidade: str) -> float:
        """Converte chegadas semanais previstas em leitos ocupados/dia via Lei de Little."""
        return max(0.0, weekly_admissions) * self.mean_los_days(especialidade) / 7.0


@dataclass(frozen=True)
class AggregatedCompetingRisksSummary:
    """Resumo do hazard causa-específico (alta viva vs. óbito) por dia de internação."""

    bucket_dias: tuple[float, ...]
    hazard_discharge: tuple[float, ...]
    hazard_death: tuple[float, ...]
    cif_discharge: tuple[float, ...]
    cif_death: tuple[float, ...]
    mean_los_days: float
    mean_los_discharge_days: float
    mean_los_death_days: float
    n_internacoes: int


class AggregatedCompetingRisksLosEstimator:
    """Estimador de LOS com riscos competitivos em tempo discreto, sem `scikit-survival` (GPL).

    Ajusta um hazard causa-específico não-paramétrico por bucket de dia de internação
    (`h_k = eventos_k / em_risco`) e calcula a incidência acumulada de Aalen-Johansen
    (`CIF_k(d) = sum_{s<=d} S(s-1) * h_k(s)`), separando alta viva de óbito hospitalar.
    O LOS médio é obtido pela Lei de Little (`L = chegadas * LOS / 7`) no acoplamento `SimPy`.

    Entrada esperada (`risk_df`): agregado por `(bucket_dia_internacao)` com `em_risco`,
    `alta_viva` e `obito`, estritamente causal (`ano_epi <= train_end_year`).
    """

    def __init__(self, summary: AggregatedCompetingRisksSummary) -> None:
        self._summary = summary

    @property
    def summary(self) -> AggregatedCompetingRisksSummary:
        return self._summary

    @classmethod
    def fit_causal(
        cls,
        risk_df: pl.DataFrame,
        *,
        train_end_year: int = 2022,
        ano_col: str = "ano_epi",
        bucket_col: str = "bucket_dia_internacao",
        at_risk_col: str = "em_risco",
        discharge_col: str = "alta_viva",
        death_col: str = "obito",
    ) -> AggregatedCompetingRisksLosEstimator:
        """Ajusta o hazard causa-específico apenas sobre `ano_epi <= train_end_year`."""
        required = (ano_col, bucket_col, at_risk_col, discharge_col, death_col)
        missing = [c for c in required if c not in risk_df.columns]
        if missing:
            raise DataContractError(
                f"AggregatedCompetingRisksLosEstimator: colunas ausentes {missing}."
            )
        causal = risk_df.filter(pl.col(ano_col) <= train_end_year)
        if causal.is_empty():
            raise DataContractError(f"Sem registros até train_end_year={train_end_year}.")

        agg = (
            causal.group_by(bucket_col)
            .agg(
                pl.col(at_risk_col).sum().cast(pl.Float64).alias("em_risco"),
                pl.col(discharge_col).sum().cast(pl.Float64).alias("alta"),
                pl.col(death_col).sum().cast(pl.Float64).alias("obito"),
            )
            .filter(pl.col("em_risco") > 0.0)
            .sort(bucket_col)
        )
        if agg.is_empty():
            raise DataContractError(
                "AggregatedCompetingRisksLosEstimator: nenhum bucket com em_risco > 0."
            )

        buckets = agg[bucket_col].cast(pl.Float64).to_numpy()
        at_risk = agg["em_risco"].to_numpy()
        alta = agg["alta"].to_numpy()
        obito = agg["obito"].to_numpy()

        hazard_dis = np.clip(alta / at_risk, 0.0, 1.0)
        hazard_death = np.clip(obito / at_risk, 0.0, 1.0)
        # Garante h_alta + h_obito <= 1 (o restante é censura/persistência no bucket).
        overflow = hazard_dis + hazard_death > 1.0
        if overflow.any():
            norm = np.where(overflow, 1.0 / (hazard_dis + hazard_death), 1.0)
            hazard_dis = hazard_dis * norm
            hazard_death = hazard_death * norm

        n = len(buckets)
        cif_dis = np.zeros(n, dtype=np.float64)
        cif_death = np.zeros(n, dtype=np.float64)
        inc_dis = np.zeros(n, dtype=np.float64)
        inc_death = np.zeros(n, dtype=np.float64)
        surv_prev = 1.0
        cum_dis = 0.0
        cum_death = 0.0
        for i in range(n):
            inc_dis[i] = surv_prev * hazard_dis[i]
            inc_death[i] = surv_prev * hazard_death[i]
            cum_dis += inc_dis[i]
            cum_death += inc_death[i]
            cif_dis[i] = cum_dis
            cif_death[i] = cum_death
            surv_prev *= 1.0 - hazard_dis[i] - hazard_death[i]

        def _conditional_mean(incidence: np.ndarray, total: float) -> float:
            if total <= 1e-12:
                return float("nan")
            return float(np.sum(buckets * incidence) / total)

        mean_los_days = float(np.sum(buckets * (inc_dis + inc_death)))
        summary = AggregatedCompetingRisksSummary(
            bucket_dias=tuple(float(b) for b in buckets),
            hazard_discharge=tuple(float(v) for v in hazard_dis),
            hazard_death=tuple(float(v) for v in hazard_death),
            cif_discharge=tuple(float(v) for v in cif_dis),
            cif_death=tuple(float(v) for v in cif_death),
            mean_los_days=mean_los_days,
            mean_los_discharge_days=_conditional_mean(inc_dis, cum_dis),
            mean_los_death_days=_conditional_mean(inc_death, cum_death),
            n_internacoes=round(float(at_risk[0])),
        )
        return cls(summary)

    @property
    def mean_los_days(self) -> float:
        """LOS médio não-restrito (eventos combinados de alta e óbito)."""
        return self._summary.mean_los_days

    def admissions_to_occupied_beds(self, weekly_admissions: float) -> float:
        """Converte chegadas semanais em leitos ocupados/dia via Lei de Little."""
        return max(0.0, weekly_admissions) * self._summary.mean_los_days / 7.0
