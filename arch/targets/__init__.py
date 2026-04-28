"""Calibration target schema and query helpers."""

from db.schema import (
    DEFAULT_DB_PATH,
    DataSource,
    GeographicLevel,
    Jurisdiction,
    Stratum,
    StratumConstraint,
    Target,
    TargetType,
    get_engine,
    get_session,
    init_db,
)
from db.supabase_client import insert_targets_batch, query_strata, query_targets
from calibration.targets import TargetSpec, get_targets
from .microplex import (
    SOIAgingFactors,
    age_soi_targets,
    build_hierarchical_microplex_constraints,
    build_microplex_constraints,
    constraints_to_ipf_dicts,
    get_soi_aging_factors,
    load_microplex_targets,
)

__all__ = [
    "DEFAULT_DB_PATH",
    "DataSource",
    "GeographicLevel",
    "Jurisdiction",
    "Stratum",
    "StratumConstraint",
    "Target",
    "TargetSpec",
    "TargetType",
    "SOIAgingFactors",
    "age_soi_targets",
    "get_engine",
    "get_targets",
    "get_session",
    "init_db",
    "build_hierarchical_microplex_constraints",
    "build_microplex_constraints",
    "constraints_to_ipf_dicts",
    "get_soi_aging_factors",
    "insert_targets_batch",
    "load_microplex_targets",
    "query_strata",
    "query_targets",
]
