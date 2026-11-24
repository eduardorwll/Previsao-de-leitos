"""Sensores automáticos de higiene e invariantes de dados cruzados (Gates G1-G8).

Este módulo implementa 8 gates determinísticos que auditam a integridade de
cruzamentos hospitalares, hidrológicos e espaciais antes do treinamento e da
avaliação contrafactual.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

from previsao_leitos.domain.exceptions import DataContractError

HygieneStatus = Literal["PASS", "WARN", "FAIL"]

CLINICAL_DISCLAIMER = (
    "Pesquisa aberta e avaliação metodológica. "
    "Not for patient care or live clinical use."
)


@dataclass(frozen=True)
class HygieneGateResult:
    """Resultado auditável de um gate individual de higiene de dados."""

    gate_id: str
    name: str
    status: HygieneStatus
    observed_value: float
    threshold: float
    unit: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializa o resultado para dicionário compatível com JSON."""
        return asdict(self)


@dataclass(frozen=True)
class DataHygieneScorecard:
    """Scorecard consolidado dos gates de higiene de dados (G1-G8)."""

    gates: tuple[HygieneGateResult, ...]
    overall_status: HygieneStatus
    passed_count: int
    warn_count: int
    failed_count: int
    disclaimer: str = CLINICAL_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        """Serializa o scorecard para dicionário compatível com JSON."""
        return {
            "overall_status": self.overall_status,
            "passed_count": self.passed_count,
            "warn_count": self.warn_count,
            "failed_count": self.failed_count,
            "disclaimer": self.disclaimer,
            "gates": [gate.to_dict() for gate in self.gates],
        }

    def save_json(self, path: Path) -> Path:
        """Persiste o scorecard em JSON UTF-8 de forma atômica."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        tmp_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        tmp_path.replace(path)
        return path

    def require_all_pass(self, *, allow_warn: bool = True) -> None:
        """Lança DataContractError se algum gate reprovou (ou alertou se allow_warn=False)."""
        blocking = [
            gate
            for gate in self.gates
            if gate.status == "FAIL" or (not allow_warn and gate.status == "WARN")
        ]
        if blocking:
            ids = ", ".join(f"{g.gate_id} ({g.status}: {g.summary})" for g in blocking)
            raise DataContractError(f"Higiene de dados reprovada nos gates: {ids}")


def _validate_columns(df: pl.DataFrame, required: Sequence[str], context: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DataContractError(f"{context}: colunas ausentes no DataFrame: {missing}")


def check_g1_collinearity_vif(
    df: pl.DataFrame,
    feature_cols: Sequence[str],
    *,
    max_vif: float = 5.0,
    warn_vif: float = 3.5,
) -> HygieneGateResult:
    """G1: Calcula o Fator de Inflação de Variância (VIF) máximo entre covariáveis."""
    if len(feature_cols) < 2:
        raise DataContractError("G1 requer pelo menos 2 colunas para cálculo de VIF.")
    _validate_columns(df, feature_cols, "G1")

    clean_df = df.select(list(feature_cols)).drop_nulls()
    if clean_df.height <= len(feature_cols):
        raise DataContractError(
            "G1: número insuficiente de linhas completas para estimar VIF."
        )

    matrix = clean_df.to_numpy().astype(np.float64, copy=False)
    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0, ddof=1)

    vif_by_feature: dict[str, float] = {}
    for idx, col in enumerate(feature_cols):
        if stds[idx] <= 1e-12:
            vif_by_feature[col] = 1e6
            continue
        y = (matrix[:, idx] - means[idx]) / stds[idx]
        other_indices = [j for j in range(len(feature_cols)) if j != idx and stds[j] > 1e-12]
        if not other_indices:
            vif_by_feature[col] = 1.0
            continue
        x_others = (matrix[:, other_indices] - means[other_indices]) / stds[other_indices]
        beta, _, _, _ = np.linalg.lstsq(x_others, y, rcond=None)
        residuals = y - x_others @ beta
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum(y**2))
        r2 = max(0.0, min(1.0, 1.0 - ss_res / max(ss_tot, 1e-12)))
        vif_val = 1e6 if r2 >= 1.0 - 1e-6 else float(1.0 / (1.0 - r2))
        vif_by_feature[col] = round(vif_val, 4)

    max_observed = max(vif_by_feature.values())
    status: HygieneStatus
    if max_observed > max_vif:
        status = "FAIL"
    elif max_observed > warn_vif:
        status = "WARN"
    else:
        status = "PASS"

    return HygieneGateResult(
        gate_id="G1",
        name="Colinearidade de Covariáveis (VIF)",
        status=status,
        observed_value=round(max_observed, 4),
        threshold=max_vif,
        unit="VIF",
        summary=f"VIF máximo observado = {max_observed:.2f} (limite <= {max_vif:.2f}).",
        details={
            "n_rows": clean_df.height,
            "vif_by_feature": vif_by_feature,
        },
    )


def check_g2_join_loss(
    total_before: float,
    total_after: float,
    *,
    max_loss_pct: float = 0.1,
    warn_loss_pct: float = 0.01,
) -> HygieneGateResult:
    """G2: Verifica atrito/perda de registros ou massa em joins e detecta explosão cartesiana."""
    if total_before <= 0:
        raise DataContractError("G2: total_before deve ser estritamente positivo.")
    if total_after < 0:
        raise DataContractError("G2: total_after não pode ser negativo.")

    if total_after > total_before:
        inflation_pct = 100.0 * (total_after - total_before) / total_before
        return HygieneGateResult(
            gate_id="G2",
            name="Conservação de Join / Atrito Amostral",
            status="FAIL",
            observed_value=round(inflation_pct, 4),
            threshold=max_loss_pct,
            unit="%",
            summary=(
                f"Explosão cartesiana detectada no join: +{inflation_pct:.3f}% "
                f"({total_before:.0f} -> {total_after:.0f})."
            ),
            details={
                "total_before": total_before,
                "total_after": total_after,
                "cartesian_explosion": True,
            },
        )

    loss_pct = 100.0 * (total_before - total_after) / total_before
    status: HygieneStatus
    if loss_pct > max_loss_pct:
        status = "FAIL"
    elif loss_pct > warn_loss_pct:
        status = "WARN"
    else:
        status = "PASS"

    return HygieneGateResult(
        gate_id="G2",
        name="Conservação de Join / Atrito Amostral",
        status=status,
        observed_value=round(loss_pct, 6),
        threshold=max_loss_pct,
        unit="%",
        summary=(
            f"Perda no join = {loss_pct:.4f}% "
            f"({total_before - total_after:.0f} de {total_before:.0f} registros; "
            f"limite <= {max_loss_pct:.2f}%)."
        ),
        details={
            "total_before": total_before,
            "total_after": total_after,
            "lost_count": total_before - total_after,
            "cartesian_explosion": False,
        },
    )


def check_g3_census_capacity_conservation(
    df: pl.DataFrame,
    *,
    occupied_col: str,
    capacity_col: str,
    days_per_period: float = 7.0,
    max_occupancy_ratio: float = 1.30,
    warn_occupancy_ratio: float = 1.05,
) -> HygieneGateResult:
    """G3: Audita conservação física entre censo de pacientes-dia e capacidade instalada CNES."""
    if days_per_period <= 0:
        raise DataContractError("G3: days_per_period deve ser positivo.")
    _validate_columns(df, [occupied_col, capacity_col], "G3")
    if df.is_empty():
        raise DataContractError("G3: DataFrame vazio.")

    occ = df[occupied_col].cast(pl.Float64).to_numpy()
    cap = df[capacity_col].cast(pl.Float64).to_numpy()

    zero_cap_positive_occ = int(np.sum((cap <= 0.0) & (occ > 0.0)))
    valid_mask = (cap > 0.0) & np.isfinite(occ) & np.isfinite(cap)
    if not np.any(valid_mask):
        return HygieneGateResult(
            gate_id="G3",
            name="Conservação Censo Diário vs. Capacidade CNES",
            status="FAIL",
            observed_value=0.0,
            threshold=max_occupancy_ratio,
            unit="ratio",
            summary="Nenhuma linha com capacidade positiva para auditar ocupação.",
            details={"zero_capacity_with_positive_occupancy": zero_cap_positive_occ},
        )

    ratios = occ[valid_mask] / (cap[valid_mask] * days_per_period)
    max_ratio = float(np.max(ratios))
    p99_ratio = float(np.quantile(ratios, 0.99))
    exceed_count = int(np.sum(ratios > max_occupancy_ratio))

    status: HygieneStatus
    if max_ratio > max_occupancy_ratio or zero_cap_positive_occ > 0:
        status = "FAIL"
    elif max_ratio > warn_occupancy_ratio:
        status = "WARN"
    else:
        status = "PASS"

    return HygieneGateResult(
        gate_id="G3",
        name="Conservação Censo Diário vs. Capacidade CNES",
        status=status,
        observed_value=round(max_ratio, 4),
        threshold=max_occupancy_ratio,
        unit="ratio",
        summary=(
            f"Ocupação máxima = {max_ratio * 100:.1f}% da capacidade CNES "
            f"(limite <= {max_occupancy_ratio * 100:.1f}%; "
            f"capacidade zero com demanda = {zero_cap_positive_occ})."
        ),
        details={
            "p99_occupancy_ratio": round(p99_ratio, 4),
            "rows_exceeding_max": exceed_count,
            "zero_capacity_with_positive_occupancy": zero_cap_positive_occ,
            "days_per_period": days_per_period,
        },
    )


def check_g4_right_censoring(
    df: pl.DataFrame,
    *,
    time_col: str,
    value_col: str,
    tail_weeks: int = 4,
    reference_weeks: int = 12,
    max_drop_pct: float = 25.0,
    warn_drop_pct: float = 10.0,
) -> HygieneGateResult:
    """G4: Detecta queda abrupta nas semanas finais por censura à direita (atraso de registro)."""
    if tail_weeks < 1 or reference_weeks < 1:
        raise DataContractError("G4: tail_weeks e reference_weeks devem ser >= 1.")
    _validate_columns(df, [time_col, value_col], "G4")

    series_df = (
        df.group_by(time_col)
        .agg(pl.col(value_col).cast(pl.Float64).sum().alias("total_value"))
        .sort(time_col)
    )
    if series_df.height < tail_weeks + reference_weeks:
        raise DataContractError(
            "G4: histórico insuficiente para comparar janela de cauda e janela de referência."
        )

    values = series_df["total_value"].to_numpy()
    tail_slice = values[-tail_weeks:]
    ref_slice = values[-(tail_weeks + reference_weeks) : -tail_weeks]
    ref_median = float(np.median(ref_slice))
    min_tail = float(np.min(tail_slice))

    drop_pct = 0.0 if ref_median <= 1e-12 else max(0.0, 100.0 * (1.0 - min_tail / ref_median))

    status: HygieneStatus
    if drop_pct > max_drop_pct:
        status = "FAIL"
    elif drop_pct > warn_drop_pct:
        status = "WARN"
    else:
        status = "PASS"

    return HygieneGateResult(
        gate_id="G4",
        name="Detector de Censura à Direita (Atraso de Processamento)",
        status=status,
        observed_value=round(drop_pct, 4),
        threshold=max_drop_pct,
        unit="%",
        summary=(
            f"Queda máxima nas últimas {tail_weeks} semanas = {drop_pct:.1f}% "
            f"vs. mediana de referência ({ref_median:.1f}; limite <= {max_drop_pct:.1f}%)."
        ),
        details={
            "reference_median": round(ref_median, 4),
            "min_tail_value": round(min_tail, 4),
            "tail_weeks": tail_weeks,
            "reference_weeks": reference_weeks,
        },
    )


def check_g5_maup_dilution(
    raw_area_index: Sequence[float] | np.ndarray[Any, Any],
    weighted_index: Sequence[float] | np.ndarray[Any, Any],
    *,
    max_dilution_ratio: float = 2.0,
    warn_dilution_ratio: float = 1.5,
    eps: float = 1e-6,
) -> HygieneGateResult:
    """G5: Quantifica a razão máxima de diluição espacial (MAUP) entre índice bruto e ponderado."""
    raw = np.asarray(raw_area_index, dtype=np.float64)
    weighted = np.asarray(weighted_index, dtype=np.float64)
    if raw.shape != weighted.shape or raw.size == 0:
        raise DataContractError("G5: vetores raw_area_index e weighted_index incompatíveis.")

    ratios: list[float] = []
    for r_val, w_val in zip(raw.ravel(), weighted.ravel(), strict=True):
        if r_val <= eps and w_val <= eps:
            ratios.append(1.0)
            continue
        high = max(abs(float(r_val)), abs(float(w_val)))
        low = max(min(abs(float(r_val)), abs(float(w_val))), eps)
        ratios.append(high / low)

    max_ratio = max(ratios)
    median_ratio = float(np.median(ratios))

    status: HygieneStatus
    if max_ratio > max_dilution_ratio:
        status = "FAIL"
    elif max_ratio > warn_dilution_ratio:
        status = "WARN"
    else:
        status = "PASS"

    return HygieneGateResult(
        gate_id="G5",
        name="Razão de Diluição Espacial (MAUP)",
        status=status,
        observed_value=round(max_ratio, 4),
        threshold=max_dilution_ratio,
        unit="x",
        summary=(
            f"Distorção MAUP máxima = {max_ratio:.2f}x entre índice por área bruta e "
            f"índice ponderado por população/leitos (limite <= {max_dilution_ratio:.2f}x)."
        ),
        details={
            "median_ratio": round(median_ratio, 4),
            "n_units": int(raw.size),
        },
    )


def check_g6_spatial_grid_collision(
    df: pl.DataFrame,
    *,
    unit_col: str,
    grid_key_col: str,
    elevation_col: str | None = None,
    max_elevation_diff_m: float = 250.0,
    max_collision_rate_pct: float = 5.0,
) -> HygieneGateResult:
    """G6: Detecta colisão de unidades distintas na mesma célula de grade climática."""
    required = [unit_col, grid_key_col]
    if elevation_col is not None:
        required.append(elevation_col)
    _validate_columns(df, required, "G6")
    if df.is_empty():
        raise DataContractError("G6: DataFrame vazio.")

    dedup = df.select(required).unique(subset=[unit_col])
    total_units = dedup.height

    aggs: list[pl.Expr] = [pl.col(unit_col).n_unique().alias("n_units")]
    if elevation_col is not None:
        elev_expr = pl.col(elevation_col).cast(pl.Float64)
        aggs.append((elev_expr.max() - elev_expr.min()).alias("elev_diff_m"))

    grouped = dedup.group_by(grid_key_col).agg(aggs)
    colliding = grouped.filter(pl.col("n_units") > 1)
    colliding_units = (
        int(colliding["n_units"].sum()) if not colliding.is_empty() else 0
    )
    collision_rate_pct = 100.0 * colliding_units / max(total_units, 1)

    max_elev_diff = 0.0
    if elevation_col is not None and not colliding.is_empty():
        elev_arr = colliding["elev_diff_m"].cast(pl.Float64).to_numpy()
        finite_elev = elev_arr[np.isfinite(elev_arr)]
        if finite_elev.size > 0:
            max_elev_diff = float(np.max(finite_elev))

    elev_violation = elevation_col is not None and max_elev_diff > max_elevation_diff_m
    rate_violation = collision_rate_pct > max_collision_rate_pct

    status: HygieneStatus
    if elev_violation or rate_violation:
        status = "FAIL"
    elif colliding_units > 0:
        status = "WARN"
    else:
        status = "PASS"

    observed = max_elev_diff if elev_violation else collision_rate_pct
    threshold = max_elevation_diff_m if elev_violation else max_collision_rate_pct
    unit = "m" if elev_violation else "%"

    return HygieneGateResult(
        gate_id="G6",
        name="Detector de Colisão de Grade Espacial",
        status=status,
        observed_value=round(observed, 4),
        threshold=threshold,
        unit=unit,
        summary=(
            f"Unidades em colisão = {colliding_units}/{total_units} ({collision_rate_pct:.1f}%); "
            f"desnível altimétrico intra-célula máximo = {max_elev_diff:.1f} m."
        ),
        details={
            "total_units": total_units,
            "colliding_units": colliding_units,
            "collision_rate_pct": round(collision_rate_pct, 4),
            "max_elevation_diff_m": round(max_elev_diff, 4),
        },
    )


def check_g7_sutva_donor_purity(
    donor_df: pl.DataFrame,
    *,
    unit_col: str,
    shock_col: str,
    capacity_col: str | None = None,
    max_donor_shock: float = 100.0,
    min_donor_capacity: float = 1.0,
) -> HygieneGateResult:
    """G7: Verifica pureza SUTVA do pool de doadores (sem choque nem capacidade zero)."""
    required = [unit_col, shock_col]
    if capacity_col is not None:
        required.append(capacity_col)
    _validate_columns(donor_df, required, "G7")
    if donor_df.is_empty():
        raise DataContractError("G7: pool de doadores vazio.")

    shock_viol = donor_df.filter(pl.col(shock_col).cast(pl.Float64) > max_donor_shock)
    shock_units = sorted({str(u) for u in shock_viol[unit_col].to_list()})

    cap_units: list[str] = []
    if capacity_col is not None:
        cap_viol = donor_df.filter(pl.col(capacity_col).cast(pl.Float64) < min_donor_capacity)
        cap_units = sorted({str(u) for u in cap_viol[unit_col].to_list()})

    violating_units = sorted(set(shock_units) | set(cap_units))
    status: HygieneStatus = "FAIL" if violating_units else "PASS"

    return HygieneGateResult(
        gate_id="G7",
        name="Pureza de Controles Contrafactuais (SUTVA)",
        status=status,
        observed_value=float(len(violating_units)),
        threshold=0.0,
        unit="unidades",
        summary=(
            f"{len(violating_units)} doador(es) violando SUTVA "
            f"(choque > {max_donor_shock:.1f}: {shock_units}; "
            f"capacidade < {min_donor_capacity:.1f}: {cap_units})."
        ),
        details={
            "violating_units": violating_units,
            "shock_violating_units": shock_units,
            "zero_capacity_units": cap_units,
            "total_donors": donor_df[unit_col].n_unique(),
        },
    )


def check_g8_degenerate_surge_labels(
    y_true: Sequence[float] | np.ndarray[Any, Any],
    baseline: Sequence[float] | np.ndarray[Any, Any],
    surge_flags: Sequence[bool] | Sequence[int] | np.ndarray[Any, Any],
    *,
    min_positive_value: float = 1e-9,
) -> HygieneGateResult:
    """G8: Bloqueia rótulos degenerados de surto (ex.: 0.0 >= 1.10 * 0.0 marcado como True)."""
    y_arr = np.asarray(y_true, dtype=np.float64)
    base_arr = np.asarray(baseline, dtype=np.float64)
    flags_arr = np.asarray(surge_flags, dtype=bool)
    if not (y_arr.shape == base_arr.shape == flags_arr.shape) or y_arr.size == 0:
        raise DataContractError("G8: vetores y_true, baseline e surge_flags incompatíveis.")

    degenerate_mask = flags_arr & (y_arr <= min_positive_value)
    degenerate_count = int(np.sum(degenerate_mask))
    degenerate_rate_pct = 100.0 * degenerate_count / float(y_arr.size)

    status: HygieneStatus = "FAIL" if degenerate_count > 0 else "PASS"
    return HygieneGateResult(
        gate_id="G8",
        name="Trava Anti-Rótulo Degenerado de Surto (0 >= 0)",
        status=status,
        observed_value=float(degenerate_count),
        threshold=0.0,
        unit="casos",
        summary=(
            f"{degenerate_count} rótulo(s) de surto degenerados com y <= {min_positive_value} "
            f"({degenerate_rate_pct:.2f}% das observações)."
        ),
        details={
            "degenerate_count": degenerate_count,
            "degenerate_rate_pct": round(degenerate_rate_pct, 4),
            "total_observations": int(y_arr.size),
        },
    )


def compile_hygiene_scorecard(gates: Sequence[HygieneGateResult]) -> DataHygieneScorecard:
    """Consolida os resultados dos gates em um DataHygieneScorecard imutável."""
    if not gates:
        raise DataContractError("compile_hygiene_scorecard requer ao menos um gate.")

    passed = sum(1 for g in gates if g.status == "PASS")
    warned = sum(1 for g in gates if g.status == "WARN")
    failed = sum(1 for g in gates if g.status == "FAIL")

    overall: HygieneStatus
    if failed > 0:
        overall = "FAIL"
    elif warned > 0:
        overall = "WARN"
    else:
        overall = "PASS"

    return DataHygieneScorecard(
        gates=tuple(gates),
        overall_status=overall,
        passed_count=passed,
        warn_count=warned,
        failed_count=failed,
    )
