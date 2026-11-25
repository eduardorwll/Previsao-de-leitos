"""Módulo geoespacial e hidrológico 100% RS (grade calendário, bacias, aterros e W_ref)."""

from previsao_leitos.geo.basin_graph import (
    ESTUARINE_BACKWATER_BASINS,
    RS_BASIN_EDGES,
    RS_BASIN_WEIGHTED_EDGES,
    RS_REGION_PREFIX_TO_BASIN,
    BasinTopology,
    WeightedBasinEdge,
    compute_estuarine_backwater_index,
    map_rs_health_region_to_basin,
)
from previsao_leitos.geo.calendar_grid import (
    aggregate_daily_to_iso_weekly,
    build_complete_daily_grid,
    build_complete_iso_week_grid,
    build_ex_ante_distributed_lags,
    compute_calendar_hydro_features,
)
from previsao_leitos.geo.enso_teleconnections import (
    ENSO_INDEX_COLUMNS,
    EnsoRegime,
    build_causal_enso_weekly_features,
    classify_enso_regime,
    parse_noaa_oni_table,
)
from previsao_leitos.geo.fine_hydro_sources import (
    aggregate_ana_hidroweb_weekly,
    aggregate_merge_cptec_weekly,
    aggregate_sar_flood_extent,
)
from previsao_leitos.geo.hydro_sources import (
    HydroFixtureAdapter,
    HydroSourcePort,
    NasaPowerHydroAdapter,
    OpenMeteoEra5Adapter,
    RegionCentroid,
)
from previsao_leitos.geo.landfill_exposure import (
    RS_POLDER_MUNICIPALITIES,
    build_causal_landfill_exposure,
)
from previsao_leitos.geo.panel_builder import (
    SplitPolicy,
    TracerBulletReport,
    evaluate_distributed_lag_tracer_bullet,
    prepare_causal_weekly_panel,
)
from previsao_leitos.geo.referral_graph import (
    build_causal_referral_matrix,
    compute_catchment_flood_pressure,
)
from previsao_leitos.geo.v3_panel import build_rs_v3_panel_from_frames

__all__ = [
    "ENSO_INDEX_COLUMNS",
    "ESTUARINE_BACKWATER_BASINS",
    "RS_BASIN_EDGES",
    "RS_BASIN_WEIGHTED_EDGES",
    "RS_POLDER_MUNICIPALITIES",
    "RS_REGION_PREFIX_TO_BASIN",
    "BasinTopology",
    "EnsoRegime",
    "HydroFixtureAdapter",
    "HydroSourcePort",
    "NasaPowerHydroAdapter",
    "OpenMeteoEra5Adapter",
    "RegionCentroid",
    "SplitPolicy",
    "TracerBulletReport",
    "WeightedBasinEdge",
    "aggregate_ana_hidroweb_weekly",
    "aggregate_daily_to_iso_weekly",
    "aggregate_merge_cptec_weekly",
    "aggregate_sar_flood_extent",
    "build_causal_enso_weekly_features",
    "build_causal_landfill_exposure",
    "build_causal_referral_matrix",
    "build_complete_daily_grid",
    "build_complete_iso_week_grid",
    "build_ex_ante_distributed_lags",
    "build_rs_v3_panel_from_frames",
    "classify_enso_regime",
    "compute_calendar_hydro_features",
    "compute_catchment_flood_pressure",
    "compute_estuarine_backwater_index",
    "evaluate_distributed_lag_tracer_bullet",
    "map_rs_health_region_to_basin",
    "parse_noaa_oni_table",
    "prepare_causal_weekly_panel",
]
