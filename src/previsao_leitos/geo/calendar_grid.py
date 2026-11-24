"""Grade calendário diária e semanal ISO completa com API causal e defasagens distribuídas.

Regra anti-corrupção temporal: nenhum `rolling` ou `shift` opera sobre índices de linha crus
com dias/semanas ausentes. Lacunas permanecem explicitamente como `null`, nunca como `0.0`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np
import polars as pl

from previsao_leitos.domain import DataContractError


def build_complete_daily_grid(
    df: pl.DataFrame,
    *,
    entity_col: str,
    date_col: str = "data",
    value_cols: Sequence[str] = ("precip_dia_mm",),
) -> pl.DataFrame:
    """Expande cada entidade para todos os dias calendário contínuos entre min(data) e max(data)."""
    if df.is_empty():
        return df.clone()

    required = (entity_col, date_col, *value_cols)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataContractError(f"Colunas ausentes em build_complete_daily_grid: {missing}")

    typed = df.select(list(required)).with_columns(pl.col(date_col).cast(pl.Date))
    dups = typed.group_by([entity_col, date_col]).len().filter(pl.col("len") > 1)
    if dups.height > 0:
        raise DataContractError(
            f"Duplicatas encontradas em ({entity_col}, {date_col}): {dups.height} chave(s)."
        )

    bounds = typed.group_by(entity_col).agg(
        pl.col(date_col).min().alias("_min_d"),
        pl.col(date_col).max().alias("_max_d"),
    )

    grid = (
        bounds.with_columns(
            pl.date_ranges(pl.col("_min_d"), pl.col("_max_d"), interval="1d").alias(date_col)
        )
        .select([entity_col, date_col])
        .explode(date_col)
    )

    return grid.join(typed, on=[entity_col, date_col], how="left").sort([entity_col, date_col])


def compute_calendar_hydro_features(
    daily_df: pl.DataFrame,
    *,
    entity_col: str,
    date_col: str = "data",
    precip_col: str = "precip_dia_mm",
    k_decay: float = 0.88,
) -> pl.DataFrame:
    """Calcula API estritamente causal, acumulados em dias calendário e sazonalidade cíclica.

    Espera uma grade diária completa (produzida por `build_complete_daily_grid`).
    A chuva do dia `t` nunca entra em `api_indice_saturacao[t]` (usa apenas `t-1` e anteriores).
    """
    if not (0.0 < k_decay < 1.0):
        raise DataContractError(f"k_decay deve estar em (0, 1): recebido {k_decay}")
    if daily_df.is_empty():
        return daily_df.clone()

    ordered = daily_df.sort([entity_col, date_col])
    entities = ordered[entity_col].to_list()
    precip_raw = ordered[precip_col].cast(pl.Float64).to_numpy()
    dates = ordered[date_col].to_list()

    n = len(entities)
    api = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if entities[i] == entities[i - 1]:
            gap_days = (dates[i] - dates[i - 1]).days
            if gap_days <= 0:
                raise DataContractError(
                    f"Grade diária não estritamente crescente em {entities[i]}: "
                    f"{dates[i - 1]} -> {dates[i]}"
                )
            prev_p = 0.0 if np.isnan(precip_raw[i - 1]) else float(precip_raw[i - 1])
            # Decai 1 dia com a leitura de i-1 e, se houver salto >1 dia, decai os dias faltantes
            step_one = k_decay * api[i - 1] + prev_p
            api[i] = step_one * (k_decay ** (gap_days - 1))

    doy = ordered[date_col].dt.ordinal_day().cast(pl.Float64)
    two_pi_doy = (2.0 * np.pi / 365.25) * doy

    return (
        ordered.with_columns(
            pl.Series("api_indice_saturacao", api, dtype=pl.Float64),
            two_pi_doy.sin().alias("doy_sin"),
            two_pi_doy.cos().alias("doy_cos"),
            pl.col(precip_col).is_not_null().cast(pl.Int32).alias("_has_obs"),
            pl.col(precip_col).fill_null(0.0).alias("_p_zero"),
        )
        .with_columns(
            pl.col("_has_obs")
            .rolling_sum(window_size=3, min_samples=1)
            .over(entity_col)
            .alias("_obs_3d"),
            pl.col("_has_obs")
            .rolling_sum(window_size=7, min_samples=1)
            .over(entity_col)
            .alias("_obs_7d"),
            pl.col("_has_obs")
            .rolling_sum(window_size=15, min_samples=1)
            .over(entity_col)
            .alias("dias_obs_15d"),
            pl.col("_has_obs")
            .rolling_sum(window_size=30, min_samples=1)
            .over(entity_col)
            .alias("_obs_30d"),
            pl.col("_p_zero")
            .rolling_sum(window_size=3, min_samples=1)
            .over(entity_col)
            .alias("_sum_3d"),
            pl.col("_p_zero")
            .rolling_sum(window_size=7, min_samples=1)
            .over(entity_col)
            .alias("_sum_7d"),
            pl.col("_p_zero")
            .rolling_sum(window_size=15, min_samples=1)
            .over(entity_col)
            .alias("_sum_15d"),
            pl.col("_p_zero")
            .rolling_sum(window_size=30, min_samples=1)
            .over(entity_col)
            .alias("_sum_30d"),
        )
        .with_columns(
            pl.when(pl.col("_obs_3d") > 0)
            .then(pl.col("_sum_3d"))
            .otherwise(None)
            .alias("precip_acum_3d"),
            pl.when(pl.col("_obs_7d") > 0)
            .then(pl.col("_sum_7d"))
            .otherwise(None)
            .alias("precip_acum_7d"),
            pl.when(pl.col("dias_obs_15d") > 0)
            .then(pl.col("_sum_15d"))
            .otherwise(None)
            .alias("precip_acum_15d"),
            pl.when(pl.col("_obs_30d") > 0)
            .then(pl.col("_sum_30d"))
            .otherwise(None)
            .alias("precip_acum_30d"),
        )
        .drop(
            [
                "_has_obs",
                "_p_zero",
                "_obs_3d",
                "_obs_7d",
                "_obs_30d",
                "_sum_3d",
                "_sum_7d",
                "_sum_15d",
                "_sum_30d",
            ]
        )
    )


def aggregate_daily_to_iso_weekly(
    daily_df: pl.DataFrame,
    *,
    entity_col: str,
    date_col: str = "data",
    precip_col: str = "precip_dia_mm",
) -> pl.DataFrame:
    """Agrega grade diária para semanas ISO preservando semanas sem leitura como null."""
    if daily_df.is_empty():
        return daily_df.clone()

    with_iso = daily_df.with_columns(
        pl.col(date_col).dt.iso_year().cast(pl.Int32).alias("ano_epi"),
        pl.col(date_col).dt.week().cast(pl.Int8).alias("semana_epi"),
    )

    aggs: list[pl.Expr] = [
        pl.col(precip_col).is_not_null().sum().cast(pl.Int8).alias("dias_observados"),
        pl.when(pl.col(precip_col).is_not_null().sum() > 0)
        .then(pl.col(precip_col).sum())
        .otherwise(None)
        .alias("chuva_sem_mm"),
        pl.col(precip_col).max().alias("chuva_max_dia_mm"),
    ]
    if "api_indice_saturacao" in daily_df.columns:
        aggs.append(pl.col("api_indice_saturacao").max().alias("api_max_sem"))
        aggs.append(pl.col("api_indice_saturacao").mean().alias("api_media_sem"))
    if "umidade_solo_m3m3" in daily_df.columns:
        aggs.append(pl.col("umidade_solo_m3m3").mean().alias("umidade_solo_sem"))
    if "temp_media_c" in daily_df.columns:
        aggs.append(pl.col("temp_media_c").mean().alias("temp_media_sem_c"))

    return (
        with_iso.group_by([entity_col, "ano_epi", "semana_epi"])
        .agg(aggs)
        .sort([entity_col, "ano_epi", "semana_epi"])
    )


def _iso_week_monday(ano_epi: int, semana_epi: int) -> date:
    return date.fromisocalendar(ano_epi, semana_epi, 1)


def build_complete_iso_week_grid(
    weekly_df: pl.DataFrame,
    *,
    entity_col: str,
) -> pl.DataFrame:
    """Garante grade semanal ISO contínua (segunda-feira a segunda-feira) por entidade."""
    if weekly_df.is_empty():
        return weekly_df.clone()

    mondays = [
        _iso_week_monday(int(a), int(w))
        for a, w in zip(
            weekly_df["ano_epi"].to_list(), weekly_df["semana_epi"].to_list(), strict=True
        )
    ]
    with_monday = weekly_df.with_columns(pl.Series("_week_monday", mondays, dtype=pl.Date))

    bounds = with_monday.group_by(entity_col).agg(
        pl.col("_week_monday").min().alias("_min_w"),
        pl.col("_week_monday").max().alias("_max_w"),
    )

    grid_rows: list[tuple[object, date]] = []
    for ent, min_w, max_w in bounds.iter_rows():
        cur = min_w
        while cur <= max_w:
            grid_rows.append((ent, cur))
            cur += timedelta(days=7)

    grid = pl.DataFrame(
        {
            entity_col: [r[0] for r in grid_rows],
            "_week_monday": [r[1] for r in grid_rows],
        }
    ).with_columns(
        pl.col("_week_monday").dt.iso_year().cast(pl.Int32).alias("ano_epi"),
        pl.col("_week_monday").dt.week().cast(pl.Int8).alias("semana_epi"),
    )

    value_cols = [c for c in weekly_df.columns if c not in (entity_col, "ano_epi", "semana_epi")]
    return (
        grid.join(
            with_monday.select([entity_col, "_week_monday", *value_cols]),
            on=[entity_col, "_week_monday"],
            how="left",
        )
        .sort([entity_col, "_week_monday"])
        .drop("_week_monday")
    )


def build_ex_ante_distributed_lags(
    weekly_df: pl.DataFrame,
    *,
    entity_col: str,
    feature_cols: Sequence[str],
    lag_weeks: Sequence[int] = (0, 1, 2, 3, 4),
) -> pl.DataFrame:
    """Cria defasagens causais `lag_k` sobre grade semanal ISO completa (sem compressão de gaps)."""
    for k in lag_weeks:
        if k < 0:
            raise DataContractError(f"Lag negativo proibido (vazamento futuro): {k}")

    complete = build_complete_iso_week_grid(weekly_df, entity_col=entity_col)
    lag_exprs = [
        pl.col(col).shift(k).over(entity_col).alias(f"{col}_lag{k}")
        for col in feature_cols
        for k in lag_weeks
    ]
    return complete.with_columns(lag_exprs)
