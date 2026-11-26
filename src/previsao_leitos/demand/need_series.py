"""Série de necessidade de leitos (pré-registro `docs/preregistro/serie-necessidade-v1.md`).

A ocupação realizada é constrangida pela oferta (suspensão de eletivas em enchente).
Esta série estima a **necessidade** como baseline sazonal + excesso de internações
atribuível ao evento, ponderado pelo choque hidrológico ex-ante.

Fórmula congelada (v1):

    necessidade = baseline + w * max(0, a - b) * LOS

com `baseline`, `b` e `LOS` estritamente passados (`t-1..t-4` / anos anteriores).

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from previsao_leitos.domain import DataContractError, LeakageViolationError

_ENTITY = "nome_regiao_saude"
_YEAR = "ano_epi"
_WEEK = "semana_epi"
_SHOCK = "prob_enchente_f03_oof"
_HOLDOUT_YEAR = 2025
_BASELINE_YEARS = 3
_LOS_WINDOW = 4
_LEVEL_WINDOW = 12
_LOS_MIN, _LOS_MAX = 1.0, 60.0


@dataclass(frozen=True)
class NeedSpec:
    """Mapeamento alvo -> colunas de ocupação e internações (pré-registro §2.1)."""

    key: str
    occupancy_col: str
    admissions_col: str
    note: str = ""


NEED_SPECS: dict[str, NeedSpec] = {
    "uti_adulto": NeedSpec("uti_adulto", "uti_adulto_diarias", "uti_adulto_internacoes"),
    "uti_total": NeedSpec(
        "uti_total",
        "diarias_uti_total_ativas",
        "uti_adulto_internacoes",
        note="proxy: internações de UTI adulto (A5)",
    ),
    "clinico": NeedSpec("clinico", "clinico_pacientes_dia", "clinico_internacoes"),
    "respiratorio": NeedSpec(
        "respiratorio", "respiratorio_pacientes_dia", "respiratorio_internacoes"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _guard_holdout(panel: pl.DataFrame) -> None:
    max_year = panel[_YEAR].max()
    if max_year is not None and int(np.asarray(max_year).item()) >= _HOLDOUT_YEAR:
        raise LeakageViolationError(
            f"Painel contém ano_epi={int(np.asarray(max_year).item())} >= {_HOLDOUT_YEAR}; "
            "a série de necessidade só pode ser construída até 2024."
        )


def _seasonal_stat(
    panel: pl.DataFrame,
    columns: dict[str, str],
    *,
    how: str,
) -> pl.DataFrame:
    """Mediana (ou média) da mesma semana ISO nos 3 anos anteriores, por região."""
    current = panel.select(
        [
            pl.col(_ENTITY),
            pl.col(_YEAR).alias("current_year"),
            pl.col(_WEEK),
        ]
    )
    history = panel.select(
        [
            pl.col(_ENTITY),
            pl.col(_YEAR).alias("history_year"),
            pl.col(_WEEK),
            *[pl.col(src).alias(name) for name, src in columns.items()],
        ]
    )
    joined = current.join(history, on=[_ENTITY, _WEEK], how="inner").filter(
        (pl.col("history_year") >= pl.col("current_year") - _BASELINE_YEARS)
        & (pl.col("history_year") <= pl.col("current_year") - 1)
    )
    aggs = [
        getattr(pl.col(name), how)().alias(f"{name}_base") for name in columns
    ]
    aggs.append(pl.len().alias("history_n"))
    return joined.group_by([_ENTITY, "current_year", _WEEK]).agg(aggs)


def build_need_series(
    panel: pl.DataFrame,
    *,
    spec: NeedSpec,
    weighting: str = "shock",
    baseline_stat: str = "median",
    baseline_mode: str = "level_corrected",
) -> pl.DataFrame:
    """Constrói a série de necessidade (pré-registro v2 por padrão)."""
    _guard_holdout(panel)
    if weighting not in ("shock", "full"):
        raise DataContractError(f"weighting inválido: {weighting}")
    if baseline_stat not in ("median", "mean"):
        raise DataContractError(f"baseline_stat inválido: {baseline_stat}")
    if baseline_mode not in ("seasonal", "level_corrected"):
        raise DataContractError(f"baseline_mode inválido: {baseline_mode}")
    for col in (spec.occupancy_col, spec.admissions_col, _SHOCK):
        if col not in panel.columns:
            raise DataContractError(f"Coluna ausente no painel: {col}")

    ordered = panel.sort([_ENTITY, _YEAR, _WEEK]).with_columns(
        pl.col(spec.occupancy_col).cast(pl.Float64, strict=False).alias("_y"),
        pl.col(spec.admissions_col).cast(pl.Float64, strict=False).alias("_a"),
    )
    # LOS estritamente passado: média móvel de y/a nas 4 semanas anteriores.
    ordered = ordered.with_columns(
        (pl.col("_y") / pl.when(pl.col("_a") > 0).then(pl.col("_a")).otherwise(None))
        .clip(_LOS_MIN, _LOS_MAX)
        .shift(1)
        .rolling_mean(window_size=_LOS_WINDOW, min_samples=1)
        .over(_ENTITY)
        .alias("los_days")
    )

    seasonal = _seasonal_stat(
        ordered,
        {"_y": "_y", "_a": "_a"},
        how="median" if baseline_stat == "median" else "mean",
    ).rename({"_y_base": "baseline", "_a_base": "a_base", "current_year": _YEAR})

    enriched = ordered.join(seasonal, on=[_ENTITY, _YEAR, _WEEK], how="left").with_columns(
        pl.col("_y").alias(spec.occupancy_col),
        pl.col("_a").alias(spec.admissions_col),
        pl.col(_SHOCK).fill_null(0.0).clip(0.0, 1.0).alias("shock_weight"),
    )

    if baseline_mode == "level_corrected":
        # v2: fator de nível estimado no agregado estadual (30 regiões). No nível
        # regional a razão y/sazonal é ruidosa quando o alvo é pequeno (respiratório),
        # o que deixava a correção instável. A máscara usa apenas o desvio vs. o
        # sazonal e é independente de `w`, preservando a monotonicidade.
        state = (
            enriched.group_by([_YEAR, _WEEK])
            .agg(
                pl.col("_y").sum().alias("_state_y"),
                pl.col("baseline").sum().alias("_state_base"),
            )
            .sort([_YEAR, _WEEK])
            .with_columns(
                pl.when(pl.col("_state_base") > 0)
                .then(pl.col("_state_y") / pl.col("_state_base"))
                .otherwise(None)
                .alias("_state_ratio")
            )
            .with_columns(
                pl.when(
                    (pl.col("_state_ratio") >= 0.5) & (pl.col("_state_ratio") <= 1.5)
                )
                .then(pl.col("_state_ratio"))
                .otherwise(None)
                .shift(1)
                .rolling_median(window_size=_LEVEL_WINDOW, min_samples=3)
                .clip(0.5, 2.0)
                .fill_null(1.0)
                .fill_nan(1.0)
                .alias("level_factor")
            )
            .select([_YEAR, _WEEK, "level_factor"])
        )
        enriched = enriched.join(state, on=[_YEAR, _WEEK], how="left").with_columns(
            (pl.col("baseline") * pl.col("level_factor")).alias("baseline")
        )
    else:
        enriched = enriched.with_columns(pl.lit(1.0).alias("level_factor"))
    weight = (
        pl.col("shock_weight")
        if weighting == "shock"
        else pl.lit(1.0, dtype=pl.Float64)
    )
    enriched = enriched.with_columns(
        (weight * (pl.col("_a") - pl.col("a_base")).clip(lower_bound=0.0) * pl.col("los_days"))
        .fill_null(0.0)
        .fill_nan(0.0)
        .alias("excess"),
    ).with_columns(
        # A7 (v3): ocupação realizada é piso da necessidade.
        pl.max_horizontal(
            pl.col("_y"), pl.col("baseline") + pl.col("excess")
        ).alias("need")
    )
    # Sem histórico suficiente (>= 2 anos anteriores na mesma semana) a linha não é
    # definida: descartar é mais honesto do que usar a ocupação contemporânea.
    enriched = enriched.filter(
        pl.col("baseline").is_not_null() & (pl.col("history_n") >= 2)
    )

    return enriched.select(
        [
            pl.col(_ENTITY),
            pl.col(_YEAR),
            pl.col(_WEEK),
            pl.col("bacia_id"),
            pl.col(spec.occupancy_col).alias("occupancy"),
            pl.col(spec.admissions_col).alias("admissions"),
            pl.col("a_base").alias("admissions_baseline"),
            pl.col("baseline"),
            pl.col("level_factor"),
            pl.col("los_days"),
            pl.col("shock_weight"),
            pl.col("excess"),
            pl.col("need"),
        ]
    ).sort([_ENTITY, _YEAR, _WEEK])


def check_monotonic_in_shock(
    panel: pl.DataFrame,
    *,
    spec: NeedSpec,
    bump: float = 0.2,
    baseline_mode: str = "level_corrected",
) -> bool:
    """Teste estrutural: aumentar o choque não pode reduzir a necessidade.

    Recalcula a série com `w + bump` e compara linha a linha. Correlação temporal
    entre `need` e `w` **não** serve: semanas de enchente tendem a ter baseline
    sazonal menor, o que produz correlação negativa mesmo com a série correta.
    """
    keys = [_ENTITY, _YEAR, _WEEK]
    base = build_need_series(panel, spec=spec, baseline_mode=baseline_mode)
    bumped_panel = panel.with_columns(
        (pl.col(_SHOCK).fill_null(0.0) + bump).clip(0.0, 1.0).alias(_SHOCK)
    )
    bumped = build_need_series(bumped_panel, spec=spec, baseline_mode=baseline_mode)
    joined = base.select([*keys, "need"]).join(
        bumped.select([*keys, "need"]), on=keys, suffix="_bumped", how="inner"
    )
    if joined.height == 0:
        return False
    return bool((joined["need_bumped"] >= joined["need"] - 1e-9).all())


def validate_need_series(
    series: pl.DataFrame, *, event_weeks: tuple[int, int] = (18, 30)
) -> dict[str, Any]:
    """Aplica os cinco critérios de aceitação do pré-registro (§6)."""
    if series.height == 0:
        raise DataContractError("Série de necessidade vazia.")
    years = series[_YEAR]
    criteria: dict[str, Any] = {}
    criteria["no_holdout"] = bool(int(np.asarray(years.max()).item()) < _HOLDOUT_YEAR)
    criteria["non_negative"] = bool(float(np.asarray(series["need"].min()).item()) >= 0.0)
    criteria["above_baseline"] = bool(
        (series["need"] >= series["baseline"].fill_null(series["occupancy"]) - 1e-9).all()
    )
    criteria["at_least_occupancy"] = bool(
        (series["need"] >= series["occupancy"] - 1e-9).all()
    )

    event = series.filter(
        (pl.col(_YEAR) == 2024)
        & (pl.col(_WEEK) >= event_weeks[0])
        & (pl.col(_WEEK) <= event_weeks[1])
    )
    excess_by_target = float(event["excess"].sum()) if event.height else 0.0
    criteria["event_excess_2024"] = excess_by_target

    correlations: list[float] = []
    for _, group in series.group_by(_ENTITY):
        if group.height < 10:
            continue
        w = group["shock_weight"].to_numpy()
        need = group["need"].to_numpy()
        if np.std(w) < 1e-9 or np.std(need) < 1e-9:
            continue
        correlations.append(float(np.corrcoef(w, need)[0, 1]))
    # Diagnóstico (não é gate): correlação temporal é confundida pela sazonalidade.
    criteria["diagnostic_corr_with_shock"] = (
        float(np.median(correlations)) if correlations else 0.0
    )

    # C6: janela pré-evento após o aquecimento do fator de nível (t-1..t-4).
    pre_event = series.filter(
        (pl.col(_YEAR) == 2024)
        & (pl.col(_WEEK) >= _LOS_WINDOW + 1)
        & (pl.col(_WEEK) < event_weeks[0])
        & (pl.col("occupancy") > 0)
    )
    if pre_event.height:
        ratio = pre_event["need"] / pre_event["occupancy"]
        criteria["pre_event_level_ratio"] = float(np.median(ratio.to_numpy()))
    else:
        criteria["pre_event_level_ratio"] = float("nan")
    return criteria


def acceptance(
    series: pl.DataFrame,
    *,
    event_weeks: tuple[int, int] = (18, 30),
    monotone_in_shock: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """Combina os critérios em um veredito booleano com relatório."""
    criteria = validate_need_series(series, event_weeks=event_weeks)
    criteria["monotone_in_shock"] = bool(monotone_in_shock)
    ok = (
        bool(criteria["no_holdout"])
        and bool(criteria["non_negative"])
        and bool(criteria["above_baseline"])
        and bool(criteria["at_least_occupancy"])
        and float(criteria["event_excess_2024"]) > 0.0
        and bool(criteria["monotone_in_shock"])
        and 0.90 <= float(criteria["pre_event_level_ratio"]) <= 1.10
    )
    return ok, criteria


def save_need_series(
    series: pl.DataFrame,
    *,
    out_path: Path,
    panel_path: Path,
    target_key: str,
    criteria: dict[str, object],
) -> Path:
    """Grava a série e o manifesto de proveniência."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    series.write_parquet(out_path)
    manifest = {
        "artifact": out_path.name,
        "sha256": _sha256(out_path),
        "panel_path": str(panel_path),
        "panel_sha256": _sha256(panel_path),
        "target": target_key,
        "preregistration": "docs/preregistro/serie-necessidade-v1.md",
        "rows": series.height,
        "years": [int(v) for v in sorted(series[_YEAR].unique().to_list())],
        "criteria": criteria,
        "disclaimer": "Not for patient care or live clinical use",
    }
    out_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_path
