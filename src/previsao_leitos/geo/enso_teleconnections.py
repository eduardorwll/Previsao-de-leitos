"""Conector causal de teleconexões climáticas El Niño / ENSO e Atlântico Sul (F02.3).

Fornece parsers para séries NOAA CPC (`ONI` / `Niño 3.4`), classificação de
regimes (`el_nino_forte`, `el_nino`, `neutro`, `la_nina`) e projeção mensal ->
semanal ISO estritamente causal com lags de 1..24 semanas e atraso de publicação.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date, timedelta
from typing import Literal

import polars as pl

from previsao_leitos.domain import DataContractError

EnsoRegime = Literal["el_nino_forte", "el_nino", "neutro", "la_nina"]

#: Um valor mensal do CPC (ex.: `-0.7`, `-999.9`); tolera o sentinela de falta colado.
_NOAA_MONTHLY_FLOAT = re.compile(r"-?\d+\.\d+")

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


def parse_noaa_roni_table(raw_text: str) -> pl.DataFrame:
    """Converte o formato ASCII oficial do NOAA CPC RONI (`SEAS YR ANOM`) em DataFrame.

    O RONI (índice relativo) é a versão do ONI com a tendência do cinturão tropical
    subtraída; o arquivo oficial tem 3 colunas (SEAS, YR, ANOM), diferentemente das
    4 do ONI (`SEAS YR TOTAL ANOM`).
    """
    rows: list[dict[str, object]] = []
    for line in raw_text.splitlines():
        parts = line.strip().split()
        if len(parts) != 3 or parts[0].upper() == "SEAS":
            continue
        season = parts[0].upper()
        if season not in _NOAA_SEASON_TO_CENTER_MONTH:
            continue
        year = int(parts[1])
        anom = float(parts[2])
        month = _NOAA_SEASON_TO_CENTER_MONTH[season]
        rows.append({"data_ref": date(year, month, 1), "roni_c": anom})

    if not rows:
        raise DataContractError("Tabela NOAA RONI vazia ou fora do formato esperado.")
    return pl.DataFrame(rows).sort("data_ref")


def parse_noaa_soi_table(raw_text: str) -> pl.DataFrame:
    """Converte o arquivo ASCII oficial do NOAA CPC SOI (bloco STANDARDIZED) em DataFrame.

    O arquivo tem dois blocos (`ANOMALY` e `STANDARDIZED DATA`); usamos o segundo
    (SOI padronizado Tahiti-Darwin). Valores de falta são `-999.9` e podem aparecer
    colados ao último valor válido na borda do mês corrente.
    """
    rows: list[dict[str, object]] = []
    in_standardized = False
    in_data = False
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        upper = stripped.upper()
        if "STANDARDIZED" in upper:
            in_standardized = True
            in_data = False
            continue
        if not in_standardized:
            continue
        if upper.startswith("YEAR"):
            in_data = True
            continue
        if not in_data:
            continue
        tokens = stripped.split()
        if not tokens or not tokens[0].isdigit():
            continue
        year = int(tokens[0])
        if year < 1951 or year > 2100:
            continue
        # Remove o token do ano e extrai os valores mensais (JAN..DEC) em ordem.
        rest = stripped[len(tokens[0]) :]
        values = [float(m.group()) for m in _NOAA_MONTHLY_FLOAT.finditer(rest)]
        for month_idx, val in enumerate(values[:12], start=1):
            if val <= -999.0:
                continue
            rows.append({"data_ref": date(year, month_idx, 1), "soi_std": val})

    if not rows:
        raise DataContractError("parse_noaa_soi_table: nenhum registro válido encontrado.")
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


def build_causal_roni_weekly_features(
    monthly_roni_df: pl.DataFrame,
    *,
    lags: Sequence[int] = (4, 12),
    publication_lag_days: int = 60,
) -> pl.DataFrame:
    """Projeta o RONI mensal para grade semanal ISO com lag de publicação de 2 meses.

    O RONI revisa por até dois meses após a publicação inicial (alta frequência do
    filtro ERSSTv6), portanto `publication_lag_days=60` é o padrão honesto-causal.
    Produz as features semanais `roni_c` (nível), `roni_regime` (classificação ±0,5/±1,5)
    e `roni_c_lag*`.
    """
    if "data_ref" not in monthly_roni_df.columns or "roni_c" not in monthly_roni_df.columns:
        raise DataContractError(
            "monthly_roni_df deve conter pelo menos as colunas 'data_ref' e 'roni_c'."
        )
    if monthly_roni_df.is_empty():
        raise DataContractError("monthly_roni_df vazio.")
    if any(lag < 1 for lag in lags):
        raise DataContractError("Todos os lags de RONI devem ser >= 1 semana.")

    aligned = _align_monthly_to_weekly_causal(
        monthly_roni_df,
        value_cols=("roni_c",),
        publication_lag_days=publication_lag_days,
    ).with_columns(
        _enso_regime_expr("roni_c").alias("roni_regime"),
    )

    lag_exprs: list[pl.Expr] = []
    for lag in sorted(set(lags)):
        lag_exprs.append(pl.col("roni_c").shift(lag).alias(f"roni_c_lag{lag}"))

    return aligned.with_columns(lag_exprs).sort(["ano_epi", "semana_epi"])


def build_causal_soi_weekly_features(
    monthly_soi_df: pl.DataFrame,
    *,
    lags: Sequence[int] = (4, 12),
    publication_lag_days: int = 30,
) -> pl.DataFrame:
    """Projeta o SOI mensal para grade semanal ISO com lag de publicação de 1 mês.

    Produz a feature semanal `soi_std` (nível, componente atmosférico do ENSO) e
    `soi_std_lag*`. O SOI é publicado com defasagem curta; o lag de 30 dias é o
    piso honesto-causal recomendado em F24.
    """
    if "data_ref" not in monthly_soi_df.columns or "soi_std" not in monthly_soi_df.columns:
        raise DataContractError(
            "monthly_soi_df deve conter pelo menos as colunas 'data_ref' e 'soi_std'."
        )
    if monthly_soi_df.is_empty():
        raise DataContractError("monthly_soi_df vazio.")
    if any(lag < 1 for lag in lags):
        raise DataContractError("Todos os lags de SOI devem ser >= 1 semana.")

    aligned = _align_monthly_to_weekly_causal(
        monthly_soi_df,
        value_cols=("soi_std",),
        publication_lag_days=publication_lag_days,
    )

    lag_exprs: list[pl.Expr] = []
    for lag in sorted(set(lags)):
        lag_exprs.append(pl.col("soi_std").shift(lag).alias(f"soi_std_lag{lag}"))

    return aligned.with_columns(lag_exprs).sort(["ano_epi", "semana_epi"])


def parse_bom_rnino_table(raw_text: str) -> pl.DataFrame:
    """Converte o CSV semanal do BOM Niño 3.4 relativo (`YYYYMMDD,YYYYMMDD,VALOR`) em DataFrame."""
    rows: list[dict[str, object]] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(",")
        if len(parts) != 3:
            continue
        start_raw, end_raw, val_raw = parts
        if (
            len(start_raw) != 8
            or len(end_raw) != 8
            or not start_raw.isdigit()
            or not end_raw.isdigit()
        ):
            continue
        week_start = date(int(start_raw[:4]), int(start_raw[4:6]), int(start_raw[6:8]))
        week_end = date(int(end_raw[:4]), int(end_raw[4:6]), int(end_raw[6:8]))
        rows.append(
            {"week_start": week_start, "week_end": week_end, "bom_rnino": float(val_raw)}
        )

    if not rows:
        raise DataContractError("parse_bom_rnino_table: nenhum registro válido encontrado.")
    return pl.DataFrame(rows).sort("week_start")


def build_causal_bom_rnino_weekly(
    bom_weekly_df: pl.DataFrame,
    *,
    publication_lag_weeks: int = 1,
) -> pl.DataFrame:
    """Projeta o BOM Niño 3.4 relativo semanal para a grade ISO com lag de publicação.

    O BOM já publica em frequência semanal (semana de segunda a domingo); ancoramos no
    `week_start` (segunda) para mapear à semana ISO e aplicamos `publication_lag_weeks=1`
    (o valor da semana W só está disponível na semana seguinte).
    """
    if "week_start" not in bom_weekly_df.columns or "bom_rnino" not in bom_weekly_df.columns:
        raise DataContractError(
            "bom_weekly_df deve conter pelo menos as colunas 'week_start' e 'bom_rnino'."
        )
    if bom_weekly_df.is_empty():
        raise DataContractError("bom_weekly_df vazio.")
    if publication_lag_weeks < 0:
        raise DataContractError("publication_lag_weeks deve ser >= 0.")

    return (
        bom_weekly_df.sort("week_start")
        .with_columns(
            pl.col("week_start").dt.iso_year().cast(pl.Int32).alias("ano_epi"),
            pl.col("week_start").dt.week().cast(pl.Int8).alias("semana_epi"),
        )
        .with_columns(pl.col("bom_rnino").shift(publication_lag_weeks).alias("bom_rnino"))
        .select(["ano_epi", "semana_epi", "bom_rnino"])
        .sort(["ano_epi", "semana_epi"])
    )
