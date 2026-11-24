"""Estimador causal de Tempo de Permanência (LOS) por especialidade e conversão em leitos/dia."""

from __future__ import annotations

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
