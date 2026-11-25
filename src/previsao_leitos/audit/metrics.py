"""Métricas probabilísticas rigorosas e intervalos de confiança via block-bootstrap."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from previsao_leitos.domain import DataContractError


@dataclass(frozen=True)
class BootstrapCI:
    """Estimativa pontual e intervalo de confiança 95% por block-bootstrap."""

    point: float
    low: float
    high: float


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, *, alpha: float) -> float:
    """Calcula Pinball Loss (quantile loss) para o quantil `alpha` em (0, 1)."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


def empirical_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fração empírica de observações contidas no intervalo `[lower, upper]`."""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def block_bootstrap_ci(
    df: pl.DataFrame,
    *,
    block_col: str,
    metric_fn: Callable[[pl.DataFrame], float],
    n_boot: int = 400,
    ci_level: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Calcula intervalo de confiança reamostrando blocos completos (bacia/região/mês)."""
    point = float(metric_fn(df))
    blocks = sorted(df[block_col].unique().to_list())
    if len(blocks) < 2:
        return BootstrapCI(point=point, low=point, high=point)

    block_frames = [df.filter(pl.col(block_col) == b) for b in blocks]
    rng = np.random.default_rng(seed)
    n_blocks = len(block_frames)
    samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n_blocks, size=n_blocks)
        boot_df = pl.concat([block_frames[int(j)] for j in idx])
        samples[i] = float(metric_fn(boot_df))

    tail = (1.0 - ci_level) / 2.0
    return BootstrapCI(
        point=point,
        low=float(np.quantile(samples, tail)),
        high=float(np.quantile(samples, 1.0 - tail)),
    )


def evaluate_probabilistic_predictions(
    eval_df: pl.DataFrame,
    preds_df: pl.DataFrame,
    *,
    model_name: str,
    block_col: str = "nome_regiao_saude",
    n_boot: int = 200,
    seed: int = 42,
) -> dict[str, object]:
    """Avalia previsões P10/P50/P90/P95 contra Persistência e Sazonal-Ingênuo com IC 95%."""
    y_true = eval_df["target_y"].to_numpy().astype(np.float64)
    y_pers = eval_df["y_origin"].to_numpy().astype(np.float64)
    p10 = preds_df["pred_p10"].to_numpy().astype(np.float64)
    p50 = preds_df["pred_p50"].to_numpy().astype(np.float64)
    p90 = preds_df["pred_p90"].to_numpy().astype(np.float64)
    p95 = preds_df["pred_p95"].to_numpy().astype(np.float64)

    mae_mod = float(np.mean(np.abs(y_true - p50)))
    mae_pers = float(np.mean(np.abs(y_true - y_pers)))
    mase_pers = mae_mod / mae_pers if mae_pers > 1e-12 else float("inf")

    scored = eval_df.select([block_col, "target_y", "y_origin", "y_seasonal_naive"]).with_columns(
        pl.Series("pred_p10", p10),
        pl.Series("pred_p50", p50),
        pl.Series("pred_p90", p90),
        pl.Series("pred_p95", p95),
    )
    seas_valid = scored.drop_nulls(subset=["y_seasonal_naive"])
    if seas_valid.height > 0:
        s_true = seas_valid["target_y"].to_numpy().astype(np.float64)
        s_seas = seas_valid["y_seasonal_naive"].to_numpy().astype(np.float64)
        s_mod = seas_valid["pred_p50"].to_numpy().astype(np.float64)
        mae_seas = float(np.mean(np.abs(s_true - s_seas)))
        mase_seas = float(np.mean(np.abs(s_true - s_mod))) / mae_seas if mae_seas > 1e-12 else 0.0
    else:
        mae_seas = float("nan")
        mase_seas = float("nan")

    ci_mase = block_bootstrap_ci(
        scored,
        block_col=block_col,
        metric_fn=lambda d: (
            float(d.select((pl.col("target_y") - pl.col("pred_p50")).abs().mean()).item())
            / max(
                float(d.select((pl.col("target_y") - pl.col("y_origin")).abs().mean()).item()),
                1e-12,
            )
        ),
        n_boot=n_boot,
        seed=seed,
    )

    return {
        "modelo": model_name,
        "mae_p50": mae_mod,
        "mae_persistencia": mae_pers,
        "mae_sazonal": mae_seas,
        "mase_vs_pers": mase_pers,
        "mase_ci95_low": ci_mase.low,
        "mase_ci95_high": ci_mase.high,
        "mase_vs_saz": mase_seas,
        "pinball_p50": pinball_loss(y_true, p50, alpha=0.50),
        "pinball_p90": pinball_loss(y_true, p90, alpha=0.90),
        "pinball_p95": pinball_loss(y_true, p95, alpha=0.95),
        "cobertura_p10_p90": empirical_coverage(y_true, p10, p90),
    }


def expected_calibration_error(
    y_true_event: np.ndarray,
    pred_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Calcula Expected Calibration Error (ECE) em [0, 1] para probabilidades previstas."""
    y = np.asarray(y_true_event, dtype=np.float64)
    p = np.clip(np.asarray(pred_prob, dtype=np.float64), 0.0, 1.0)
    if len(y) == 0:
        raise DataContractError("Vetor vazio em expected_calibration_error.")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(len(y))
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        count = int(np.sum(mask))
        if count > 0:
            ece += (count / n) * abs(float(np.mean(y[mask])) - float(np.mean(p[mask])))
    return float(ece)


def brier_skill_score(y_true_event: np.ndarray, pred_prob: np.ndarray) -> float:
    """Calcula Brier Skill Score (BSS = 1 - BS / BS_climatologia; >0 supera prevalência)."""
    y = np.asarray(y_true_event, dtype=np.float64)
    p = np.clip(np.asarray(pred_prob, dtype=np.float64), 0.0, 1.0)
    bs_model = float(np.mean((p - y) ** 2))
    prev = float(np.mean(y))
    bs_ref = float(np.mean((prev - y) ** 2))
    if bs_ref <= 1e-12:
        return 0.0
    return float(1.0 - (bs_model / bs_ref))


def decision_curve_net_benefit(
    y_true_event: np.ndarray,
    pred_prob: np.ndarray,
    *,
    thresholds: Sequence[float] = (0.10, 0.20, 0.30, 0.40, 0.50),
) -> pl.DataFrame:
    """Análise de Curva de Decisão (Vickers & Elkin 2006) para acionamento de contingência."""
    y = np.asarray(y_true_event, dtype=np.float64) > 0.5
    p = np.clip(np.asarray(pred_prob, dtype=np.float64), 0.0, 1.0)
    n = float(len(y))
    if n <= 0.0:
        raise DataContractError("Vetor vazio em decision_curve_net_benefit.")

    prevalence = float(np.mean(y))
    rows: list[dict[str, float]] = []
    for pt in thresholds:
        if not (0.0 < pt < 1.0):
            raise DataContractError(f"Limiar pt deve estar em (0, 1): recebido {pt}")
        odds = pt / (1.0 - pt)
        flag = p >= pt
        tp = float(np.sum(flag & y))
        fp = float(np.sum(flag & (~y)))
        nb_model = (tp / n) - (fp / n) * odds
        nb_all = prevalence - (1.0 - prevalence) * odds
        rows.append(
            {
                "limiar_pt": float(pt),
                "net_benefit_modelo": nb_model,
                "net_benefit_tratar_todos": nb_all,
                "net_benefit_tratar_nenhum": 0.0,
            }
        )
    return pl.DataFrame(rows)


def estimate_surge_probability_from_quantiles(
    preds_df: pl.DataFrame,
    threshold: np.ndarray,
    *,
    baseline: np.ndarray | None = None,
    min_positive_value: float = 1e-9,
) -> np.ndarray:
    """Estima P(Y >= threshold) por interpolação monotônica linear dos quantis P10/P50/P90/P95.

    Aplica máscara estrutural anti-G8: regiões/linhas sem capacidade ou baseline nulo
    (`baseline <= min_positive_value` ou `threshold <= min_positive_value`) recebem `0.0`.
    """
    q_vals = np.column_stack(
        [
            preds_df["pred_p10"].to_numpy().astype(np.float64),
            preds_df["pred_p50"].to_numpy().astype(np.float64),
            preds_df["pred_p90"].to_numpy().astype(np.float64),
            preds_df["pred_p95"].to_numpy().astype(np.float64),
        ]
    )
    q_probs = np.array([0.10, 0.50, 0.90, 0.95], dtype=np.float64)
    base_arr = np.asarray(baseline, dtype=np.float64) if baseline is not None else None
    probs = np.empty(len(threshold), dtype=np.float64)
    for i, thr in enumerate(threshold):
        thr_val = float(thr)
        if thr_val <= min_positive_value or (
            base_arr is not None and float(base_arr[i]) <= min_positive_value
        ):
            probs[i] = 0.0
            continue
        row = q_vals[i]
        cdf = float(np.interp(thr_val, row, q_probs, left=0.05, right=0.98))
        probs[i] = float(np.clip(1.0 - cdf, 0.0, 1.0))
    return probs


def basin_cluster_bootstrap_ci(
    values: np.ndarray,
    clusters: Sequence[str],
    *,
    n_boot: int = 400,
    seed: int = 42,
) -> BootstrapCI:
    """Calcula IC 95% via cluster-bootstrap por bacia hidrográfica (respeitando dependência)."""
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0 or vals.size != len(clusters):
        raise DataContractError("values e clusters incompatíveis em basin_cluster_bootstrap_ci.")

    unique_clusters = sorted(set(clusters))
    cluster_indices = [
        np.array([i for i, c in enumerate(clusters) if c == cl], dtype=np.int64)
        for cl in unique_clusters
    ]
    n_clusters = len(cluster_indices)
    point = float(np.mean(vals))
    if n_clusters < 2:
        return BootstrapCI(point=point, low=point, high=point)

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([cluster_indices[int(c)] for c in sampled])
        boot_means[b] = float(np.mean(vals[idx]))

    return BootstrapCI(
        point=point,
        low=float(np.quantile(boot_means, 0.025)),
        high=float(np.quantile(boot_means, 0.975)),
    )
