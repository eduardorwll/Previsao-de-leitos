"""Métricas probabilísticas rigorosas e intervalos de confiança via block-bootstrap."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl


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
