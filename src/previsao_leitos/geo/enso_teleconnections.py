"""Conector causal de teleconexões climáticas El Niño / ENSO e Atlântico Sul (F02.3).

Fornece parsers para séries NOAA CPC (`ONI` / `Niño 3.4`), classificação de
regimes (`el_nino_forte`, `el_nino`, `neutro`, `la_nina`) e projeção mensal ->
semanal ISO estritamente causal com lags de 1..24 semanas e atraso de publicação.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from typing import Literal

import polars as pl

from previsao_leitos.domain import DataContractError

EnsoRegime = Literal["el_nino_forte", "el_nino", "neutro", "la_nina"]

_NOAA_SEASON_TO_CENTER_MONTH: dict[str, int] = {
    "DJF": 1,
    "JFM": 2,
    "FMA": 3,
    "MAM": 4,
    "AMJ": 5,
    "MJJ": 6,
    "JJA": 7,
    "JAS": 8,
    "ASO": 9,
    "SON": 10,
    "OND": 11,
    "NDJ": 12,
}

ENSO_INDEX_COLUMNS: tuple[str, ...] = (
    "oni_c",
    "nino34_anom_c",
    "soi_std",
    "mei_v2",
    "tsa_anom_c",
    "sallj_proxy",
)


def classify_enso_regime(oni_c: float) -> EnsoRegime:
    """Classifica o regime ENSO pelo limiar operacional NOAA CPC ONI (°C)."""
    if oni_c >= 1.5:
        return "el_nino_forte"
    if oni_c >= 0.5:
        return "el_nino"
    if oni_c <= -0.5:
        return "la_nina"
    return "neutro"


def parse_noaa_oni_table(raw_text: str) -> pl.DataFrame:
    """Converte o formato ASCII oficial do NOAA CPC ONI (`SEAS YR TOTAL ANOM`) em DataFrame."""
    rows: list[dict[str, object]] = []
    for line in raw_text.splitlines():
        parts = line.strip().split()
        if len(parts) < 4 or parts[0].upper() == "SEAS":
            continue
        season = parts[0].upper()
        if season not in _NOAA_SEASON_TO_CENTER_MONTH:
            continue
        year = int(parts[1])
        anom = float(parts[3])
        month = _NOAA_SEASON_TO_CENTER_MONTH[season]
        rows.append({"data_ref": date(year, month, 1), "oni_c": anom})

    if not rows:
        raise DataContractError("Tabela NOAA ONI vazia ou fora do formato esperado.")
    return pl.DataFrame(rows).sort("data_ref")


def _align_monthly_to_weekly_causal(
    monthly_df: pl.DataFrame,
    *,
    value_cols: Sequence[str],
    publication_lag_days: int,
) -> pl.DataFrame:
    """Projeta colunas mensais para a grade semanal ISO com lag de publicação causal.

    Aplica deslocamento de disponibilidade (`publication_lag_days`) antes do
    `join_asof(..., strategy="backward")` para garantir que nenhum índice mensal
    fechado no futuro vaze para uma semana epidemiológica anterior.
    """
    if "data_ref" not in monthly_df.columns:
        raise DataContractError("monthly_df deve conter a coluna 'data_ref'.")
    if not value_cols:
        raise DataContractError("value_cols vazio para projeção semanal causal.")
    if any(c not in monthly_df.columns for c in value_cols):
        missing = [c for c in value_cols if c not in monthly_df.columns]
        raise DataContractError(f"Colunas ausentes para projeção semanal: {missing}.")

    norm = (
        monthly_df.select(["data_ref", *value_cols])
        .with_columns(
            pl.col("data_ref").cast(pl.Date),
            *[pl.col(c).cast(pl.Float64) for c in value_cols],
        )
        .with_columns(
            (pl.col("data_ref") + pl.duration(days=publication_lag_days)).alias("data_disponivel")
        )
        .sort("data_disponivel")
    )

    min_date = norm["data_ref"].min()
    max_date = norm["data_ref"].max()
    if not isinstance(min_date, date) or not isinstance(max_date, date):
        raise DataContractError("Datas inválidas em monthly_df.data_ref.")

    # Grade semanal (toda segunda-feira ISO) cobrindo o intervalo completo
    start_monday = min_date - timedelta(days=min_date.weekday())
    end_monday = max_date + timedelta(days=publication_lag_days + 7)
    week_dates: list[date] = []
    cur = start_monday
    while cur <= end_monday:
        week_dates.append(cur)
        cur += timedelta(days=7)

    week_grid = pl.DataFrame({"semana_inicio": week_dates}).with_columns(
        pl.col("semana_inicio").dt.iso_year().cast(pl.Int32).alias("ano_epi"),
        pl.col("semana_inicio").dt.week().cast(pl.Int8).alias("semana_epi"),
    )

    return (
        week_grid.join_asof(
            norm,
            left_on="semana_inicio",
            right_on="data_disponivel",
            strategy="backward",
        )
        .drop(["data_ref", "data_disponivel", "semana_inicio"])
    )


def _enso_regime_expr(col: str) -> pl.Expr:
    """Expressão de classificação de regime ENSO (4 classes) para uma coluna de anomalia."""
    return (
        pl.when(pl.col(col) >= 1.5)
        .then(pl.lit("el_nino_forte"))
        .when(pl.col(col) >= 0.5)
        .then(pl.lit("el_nino"))
        .when(pl.col(col) <= -0.5)
        .then(pl.lit("la_nina"))
        .otherwise(pl.lit("neutro"))
    )


def build_causal_enso_weekly_features(
    monthly_enso_df: pl.DataFrame,
    *,
    lags: Sequence[int] = (1, 4, 8, 12, 24),
    publication_lag_days: int = 32,
) -> pl.DataFrame:
    """Projeta índices mensais ENSO/Atlântico para grade semanal ISO estritamente causal.

    Aplica deslocamento de disponibilidade (`publication_lag_days`) antes do
    `join_asof(..., strategy="backward")` para garantir que nenhum índice mensal
    fechado no futuro vaze para uma semana epidemiológica anterior.
    """
    if "data_ref" not in monthly_enso_df.columns or "oni_c" not in monthly_enso_df.columns:
        raise DataContractError(
            "monthly_enso_df deve conter pelo menos as colunas 'data_ref' e 'oni_c'."
        )
    if monthly_enso_df.is_empty():
        raise DataContractError("monthly_enso_df vazio.")
    if any(lag < 1 for lag in lags):
        raise DataContractError("Todos os lags de teleconexão ENSO devem ser >= 1 semana.")

    present_indices = [c for c in ENSO_INDEX_COLUMNS if c in monthly_enso_df.columns]
    aligned = _align_monthly_to_weekly_causal(
        monthly_enso_df,
        value_cols=present_indices,
        publication_lag_days=publication_lag_days,
    ).with_columns(
        pl.col("oni_c").fill_null(0.0).clip(lower_bound=0.0).alias("el_nino_intensity"),
        _enso_regime_expr("oni_c").alias("enso_regime"),
    )

    lag_exprs: list[pl.Expr] = []
    for col in present_indices:
        for lag in sorted(set(lags)):
            lag_exprs.append(pl.col(col).shift(lag).alias(f"{col}_lag{lag}"))

    return aligned.with_columns(lag_exprs).sort(["ano_epi", "semana_epi"])


