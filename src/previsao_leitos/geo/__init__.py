"""Módulo geoespacial, grade calendário causal e propagação hidrológica de bacias (F02)."""

from previsao_leitos.geo.basin_graph import (
    RS_BASIN_EDGES,
    RS_REGION_PREFIX_TO_BASIN,
    BasinTopology,
    map_rs_health_region_to_basin,
)
from previsao_leitos.geo.calendar_grid import (
    aggregate_daily_to_iso_weekly,
    build_complete_daily_grid,
    build_complete_iso_week_grid,
    build_ex_ante_distributed_lags,
    compute_calendar_hydro_features,
)
from previsao_leitos.geo.hydro_sources import (
    HydroFixtureAdapter,
    HydroSourcePort,
    NasaPowerHydroAdapter,
    OpenMeteoEra5Adapter,
    RegionCentroid,
)
from previsao_leitos.geo.panel_builder import (
    SplitPolicy,
    TracerBulletReport,
    evaluate_distributed_lag_tracer_bullet,
    prepare_causal_weekly_panel,
)

__all__ = [
    "RS_BASIN_EDGES",
    "RS_REGION_PREFIX_TO_BASIN",
    "BasinTopology",
    "HydroFixtureAdapter",
    "HydroSourcePort",
    "NasaPowerHydroAdapter",
    "OpenMeteoEra5Adapter",
    "RegionCentroid",
    "SplitPolicy",
    "TracerBulletReport",
    "aggregate_daily_to_iso_weekly",
    "build_complete_daily_grid",
    "build_complete_iso_week_grid",
    "build_ex_ante_distributed_lags",
    "compute_calendar_hydro_features",
    "evaluate_distributed_lag_tracer_bullet",
    "map_rs_health_region_to_basin",
    "prepare_causal_weekly_panel",
]
