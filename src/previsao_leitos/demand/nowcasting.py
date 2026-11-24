"""Nowcasting de atraso de processamento (Chain-Ladder) e LOS de riscos competitivos (F04.3).

Corrige a censura à direita nas competências recentes do SIH-SUS (ex.: queda
artificial na Semana 52 de 2025 antes do fechamento das competências de 2026)
e estima curvas de incidência acumulada (CIF) separando alta viva vs. óbito.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from previsao_leitos.domain import DataContractError


class ChainLadderNowcaster:
    """Nowcaster Chain-Ladder empírico para correção de atraso de notificação/processamento."""

    def __init__(self, *, max_delay_weeks: int = 8, min_completeness: float = 0.05) -> None:
        if max_delay_weeks < 1:
            raise DataContractError("max_delay_weeks deve ser >= 1.")
        self._max_delay = max_delay_weeks
        self._min_comp = min_completeness
        self._completeness: dict[int, float] = {}
        self._dispersion: float = 1.5
        self._fitted: bool = False

    def fit(
        self,
        triangle_df: pl.DataFrame,
        *,
        event_col: str = "event_week",
        delay_col: str = "delay_weeks",
        count_col: str = "cumulative_count",
    ) -> ChainLadderNowcaster:
        """Estima fatores de desenvolvimento cumulativos F(d) sobre o triângulo histórico."""
        for col in (event_col, delay_col, count_col):
            if col not in triangle_df.columns:
                raise DataContractError(f"ChainLadderNowcaster: coluna ausente '{col}'.")
        if triangle_df.is_empty():
            raise DataContractError("ChainLadderNowcaster: triângulo vazio.")

        # Calcula fatores passo-a-passo f_d = sum(C_{t, d+1}) / sum(C_{t, d})
        piv = triangle_df.pivot(
            on=delay_col,
            index=event_col,
            values=count_col,
            aggregate_function="sum",
        )
        step_factors: dict[int, float] = {}
        rel_errors: list[float] = []

        for d in range(self._max_delay):
            c_cur = str(d)
            c_nxt = str(d + 1)
            if c_cur not in piv.columns or c_nxt not in piv.columns:
                step_factors[d] = 1.0
                continue
            valid = piv.select([c_cur, c_nxt]).drop_nulls()
            cur_vals = valid[c_cur].cast(pl.Float64).to_numpy()
            nxt_vals = valid[c_nxt].cast(pl.Float64).to_numpy()
            denom = float(np.sum(cur_vals))
            if denom <= 1e-9:
                step_factors[d] = 1.0
                continue
            factor = max(1.0, float(np.sum(nxt_vals)) / denom)
            step_factors[d] = factor
            pred_nxt = cur_vals * factor
            pos = cur_vals > 1.0
            if np.any(pos):
                rel = (nxt_vals[pos] - pred_nxt[pos]) / np.sqrt(np.maximum(pred_nxt[pos], 1.0))
                rel_errors.extend(rel.tolist())

        # Completude cumulativa F(d) = 1 / prod_{k=d}^{D-1} f_k
        self._completeness = {self._max_delay: 1.0}
        running_factor = 1.0
        for d in range(self._max_delay - 1, -1, -1):
            running_factor *= step_factors.get(d, 1.0)
            self._completeness[d] = max(self._min_comp, min(1.0, 1.0 / running_factor))

        if len(rel_errors) > 2:
            self._dispersion = max(1.0, float(np.var(rel_errors, ddof=1)))
        else:
            self._dispersion = 1.5
        self._fitted = True
        return self

    def completeness_fraction(self, delay_weeks: int) -> float:
        """Retorna a fração esperada já processada após `delay_weeks` semanas."""
        if not self._fitted:
            raise DataContractError("ChainLadderNowcaster precisa ser ajustado antes de consultar.")
        if delay_weeks >= self._max_delay:
            return 1.0
        if delay_weeks <= 0:
            return self._completeness.get(0, self._min_comp)
        return self._completeness.get(delay_weeks, 1.0)

    def nowcast(
        self,
        recent_df: pl.DataFrame,
        *,
        delay_col: str = "delay_weeks",
        observed_col: str = "observed_count",
    ) -> pl.DataFrame:
        """Corrige contagens recentes censuradas à direita devolvendo P10/P50/P90."""
        if not self._fitted:
            raise DataContractError("ChainLadderNowcaster precisa ser ajustado antes do nowcast.")
        for col in (delay_col, observed_col):
            if col not in recent_df.columns:
                raise DataContractError(f"ChainLadderNowcaster: coluna ausente '{col}'.")

        delays = recent_df[delay_col].cast(pl.Int32).to_list()
        obs = recent_df[observed_col].cast(pl.Float64).to_numpy()
        fracs = np.array([self.completeness_fraction(int(d)) for d in delays], dtype=np.float64)

        p50 = np.maximum(obs, obs / np.maximum(fracs, self._min_comp))
        # Incerteza cresce proporcionalmente à parcela ainda não observada (1 - F(d))
        unobserved = np.maximum(0.0, p50 - obs)
        sd = np.sqrt(self._dispersion * np.maximum(unobserved, 0.0))
        p10 = np.maximum(obs, p50 - 1.2816 * sd)
        p90 = np.maximum(p50, p50 + 1.2816 * sd)

        return recent_df.with_columns(
            pl.Series("completeness_frac", fracs, dtype=pl.Float64),
            pl.Series("nowcast_p10", p10, dtype=pl.Float64),
            pl.Series("nowcast_p50", p50, dtype=pl.Float64),
            pl.Series("nowcast_p90", p90, dtype=pl.Float64),
        )


@dataclass(frozen=True)
class CompetingRisksSummary:
    """Resumo de incidência acumulada (CIF) para alta viva vs. óbito hospitalar."""

    especialidade: str
    horizon_days: int
    cif_discharge: float
    cif_death: float
    mean_los_discharge: float
    mean_los_death: float
    n_train_episodes: int


class CompetingRisksLosEstimator:
    """Estimador causal (<= train_end_year) de LOS sob riscos competitivos (alta vs. óbito)."""

    def __init__(self, *, train_end_year: int = 2022) -> None:
        self._train_end_year = train_end_year
        self._records: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._fitted: bool = False

    def fit(
        self,
        episodes_df: pl.DataFrame,
        *,
        year_col: str = "ano_epi",
        specialty_col: str = "especialidade",
        los_col: str = "los_days",
        outcome_col: str = "outcome",
    ) -> CompetingRisksLosEstimator:
        """Ajusta distribuições empíricas causais separando alta viva e óbito."""
        for col in (year_col, specialty_col, los_col, outcome_col):
            if col not in episodes_df.columns:
                raise DataContractError(f"CompetingRisksLosEstimator: coluna ausente '{col}'.")

        causal = episodes_df.filter(pl.col(year_col) <= self._train_end_year)
        if causal.is_empty():
            raise DataContractError(
                f"CompetingRisksLosEstimator: sem dados até train_end_year={self._train_end_year}."
            )

        self._records = {}
        for spec in sorted(causal[specialty_col].unique().to_list()):
            sub = causal.filter(pl.col(specialty_col) == spec)
            los_arr = sub[los_col].cast(pl.Float64).to_numpy()
            out_arr = sub[outcome_col].cast(pl.String).str.to_lowercase().to_numpy()
            self._records[str(spec)] = (los_arr, out_arr)

        self._fitted = True
        return self

    def summarize(self, especialidade: str, *, horizon_days: int = 14) -> CompetingRisksSummary:
        """Calcula CIF até `horizon_days` e LOS médio condicional por desfecho."""
        if not self._fitted or especialidade not in self._records:
            raise DataContractError(
                f"Especialidade '{especialidade}' não encontrada em CompetingRisksLosEstimator."
            )
        los_arr, out_arr = self._records[especialidade]
        n = max(int(los_arr.size), 1)
        is_disch = out_arr == "alta_viva"
        is_death = out_arr == "obito"

        cif_disch = float(np.sum(is_disch & (los_arr <= horizon_days))) / float(n)
        cif_death = float(np.sum(is_death & (los_arr <= horizon_days))) / float(n)
        mean_disch = float(np.mean(los_arr[is_disch])) if np.any(is_disch) else 0.0
        mean_death = float(np.mean(los_arr[is_death])) if np.any(is_death) else 0.0

        return CompetingRisksSummary(
            especialidade=especialidade,
            horizon_days=horizon_days,
            cif_discharge=round(cif_disch, 6),
            cif_death=round(cif_death, 6),
            mean_los_discharge=round(mean_disch, 4),
            mean_los_death=round(mean_death, 4),
            n_train_episodes=n,
        )
