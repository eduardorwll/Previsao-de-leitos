"""Explicabilidade auditada (SHAP) do ponto F04.1 em 2 estágios.

Decompõe a predição pontual `(basal_rate + shock_rate) * capacidade` em:

- bloco basal (Estágio 1, linear): `capacidade * basal_rate`;
- bloco de choque (Estágio 2, GBDT monotônico): `capacidade * phi0` mais as
  contribuições `capacidade * phi_j` de cada covariável de choque ex-ante.

A margem conformal (`ConformalFloodCalibrator`) desloca apenas P10/P90/P95; ela é
reportada em estrutura separada e **nunca** participa da decomposição do ponto.

Fronteira obrigatória: Not for patient care or live clinical use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from previsao_leitos.demand.calibration import ConformalFloodCalibrator
from previsao_leitos.demand.two_stage import OrthogonalTwoStageOccupancyForecaster
from previsao_leitos.domain import DataContractError, LeakageViolationError

#: Tolerância do gate de aditividade (Task 1.2.6), em taxa relativa de ocupação.
ADDITIVITY_TOLERANCE: float = 1e-6

#: Ano máximo admitido como background/eval enquanto o holdout estiver lacrado.
_MAX_AUDIT_YEAR: int = 2024
_TRAIN_END_YEAR: int = 2022


def _as_int(value: object) -> int:
    """Converte escalar Polars (união ampla de tipos) para `int` de forma segura."""
    return int(np.asarray(value).item())


@dataclass(frozen=True)
class ConformalMargins:
    """Meia-larguras conformais em leitos/dia — explicitamente fora do SHAP."""

    low_10: np.ndarray
    high_90: np.ndarray
    high_95: np.ndarray


@dataclass(frozen=True)
class ShapPointDecomposition:
    """Decomposição aditiva auditável do ponto F04.1, em leitos ocupados/dia.

    Identidade publicada::

        pred_p50 = basal_contribution + base_value + Σ shap_values + median_shift

    `median_shift` é o intercepto de mediana dos resíduos padronizados (calibração do
    P50), reportado separadamente porque não é atribuição de feature. `conformal_margins`
    desloca apenas P10/P90/P95 e nunca entra nesta soma.
    """

    capacity: np.ndarray
    basal_contribution: np.ndarray
    base_value: np.ndarray
    shap_values: dict[str, np.ndarray] = field(default_factory=dict)
    pred_p50: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    median_shift: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    conformal_margins: ConformalMargins | None = None
    rate_residual: float = 0.0
    beds_residual: float = 0.0


def _iter_year_column(df: pl.DataFrame) -> pl.Series | None:
    for col in ("target_ano_epi", "ano_epi"):
        if col in df.columns:
            return df[col].cast(pl.Int64, strict=False)
    return None


def _guard_audit_year(df: pl.DataFrame, *, role: str, max_year: int) -> None:
    years = _iter_year_column(df)
    if years is None:
        return
    observed = years.max()
    if observed is not None and _as_int(observed) > max_year:
        raise LeakageViolationError(
            f"{role} contém ano(s) > {max_year} (máximo observado: {_as_int(observed)}); "
            "o background deve ser estritamente pré-choque e o holdout 2025+ permanece "
            "lacrado até a Task 1.3.0."
        )


def decompose_two_stage_point(
    model: OrthogonalTwoStageOccupancyForecaster,
    df: pl.DataFrame,
    *,
    background: pl.DataFrame,
    conformal: ConformalFloodCalibrator | None = None,
) -> ShapPointDecomposition:
    """Decompõe o ponto em blocos basal + choque com gate de aditividade `< 1e-6`.

    `background` é o painel de treino pré-choque (`ano_epi <= train_end_year`) e funciona
    como contrato anti-vazamento: o baseline do TreeSHAP é intrínseco às árvores
    (`tree_path_dependent`), e forçar um *masker* interventional com `background` quebra a
    aditividade exata (resíduo medido ~2e-1 em taxa). `df` é o conjunto auditado (<= 2024).
    O calibrador conformal, quando fornecido, deve já estar ajustado e é usado apenas para
    reportar a margem separada.
    """
    if background.height == 0 or df.height == 0:
        raise DataContractError("decompose_two_stage_point requer background e df não vazios.")
    _guard_audit_year(background, role="background", max_year=_TRAIN_END_YEAR)
    _guard_audit_year(df, role="df de auditoria", max_year=_MAX_AUDIT_YEAR)

    import shap  # lazy: preserva C-07 e o CI em `--extra cpu`

    comp = model.predict_stage_components(df)

    explainer = shap.TreeExplainer(
        model.shock_estimator,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )
    phi = np.asarray(explainer.shap_values(comp.shock_matrix), dtype=np.float64)
    base_rate = float(np.ravel(np.asarray(explainer.expected_value, dtype=np.float64))[0])

    capacity = comp.capacity
    rate_reconstructed = base_rate + phi.sum(axis=1)
    rate_residual = float(np.max(np.abs(comp.shock_rate - rate_reconstructed)))

    shap_beds = {
        col: phi[:, j] * capacity for j, col in enumerate(comp.shock_cols)
    }
    basal_contribution = comp.basal_rate * capacity
    base_value = np.full(df.height, base_rate, dtype=np.float64) * capacity
    point_beds = comp.point_beds
    published_p50 = model.predict_quantiles(df)["pred_p50"].to_numpy().astype(np.float64)
    median_shift = published_p50 - point_beds
    beds_residual = float(
        np.max(
            np.abs(
                published_p50
                - (
                    basal_contribution
                    + base_value
                    + rate_reconstructed * capacity
                    + median_shift
                )
            )
        )
    )

    margins: ConformalMargins | None = None
    if conformal is not None:
        base_q = model.predict_quantiles(df)
        conf_q = conformal.predict_quantiles(df)
        margins = ConformalMargins(
            low_10=base_q["pred_p10"].to_numpy().astype(np.float64)
            - conf_q["pred_p10"].to_numpy().astype(np.float64),
            high_90=conf_q["pred_p90"].to_numpy().astype(np.float64)
            - base_q["pred_p90"].to_numpy().astype(np.float64),
            high_95=conf_q["pred_p95"].to_numpy().astype(np.float64)
            - base_q["pred_p95"].to_numpy().astype(np.float64),
        )

    return ShapPointDecomposition(
        capacity=capacity,
        basal_contribution=basal_contribution,
        base_value=base_value,
        shap_values=shap_beds,
        pred_p50=published_p50,
        median_shift=median_shift,
        conformal_margins=margins,
        rate_residual=rate_residual,
        beds_residual=beds_residual,
    )


def additivity_residual(dec: ShapPointDecomposition) -> tuple[float, float]:
    """Retorna `(residuo_max_em_taxa, residuo_max_em_leitos)` da decomposição."""
    return dec.rate_residual, dec.beds_residual


def assert_additive(dec: ShapPointDecomposition, *, tol: float = ADDITIVITY_TOLERANCE) -> None:
    """Gate mecânico da Task 1.2.6: falha se a aditividade exceder `tol` em taxa."""
    if not np.isfinite(dec.rate_residual) or dec.rate_residual >= tol:
        raise DataContractError(
            f"Aditividade SHAP violada: resíduo em taxa {dec.rate_residual:.3e} >= {tol:.1e}."
        )
