"""Testes unitários da Task 4.3: cluster-bootstrap por bacia hidrográfica."""

from __future__ import annotations

import numpy as np
import pytest

from previsao_leitos.audit.metrics import basin_cluster_bootstrap_ci
from previsao_leitos.domain import DataContractError


def test_basin_cluster_bootstrap_ci_brackets_point_estimate_deterministically() -> None:
    vals = np.array([10.0, 12.0, 20.0, 22.0, 30.0, 32.0, 40.0, 42.0])
    basins = ["b1", "b1", "b2", "b2", "b3", "b3", "b4", "b4"]

    ci_a = basin_cluster_bootstrap_ci(vals, basins, n_boot=300, seed=42)
    ci_b = basin_cluster_bootstrap_ci(vals, basins, n_boot=300, seed=42)

    assert ci_a == ci_b
    assert ci_a.point == pytest.approx(26.0)
    assert ci_a.low < ci_a.point < ci_a.high

    with pytest.raises(DataContractError):
        basin_cluster_bootstrap_ci(vals, basins[:-1])
