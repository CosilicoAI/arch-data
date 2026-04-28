"""Microplex adapters for Arch target inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from calibration.constraints import (
    Constraint,
    build_constraint_matrix,
    build_hierarchical_constraint_matrix,
)
from calibration.targets import TargetSpec, get_targets


def load_microplex_targets(
    db_path: Path | None = None,
    jurisdiction: str = "us",
    year: int | None = None,
    sources: list[str] | None = None,
    variables: list[str] | None = None,
) -> list[TargetSpec]:
    """Load Arch DB target inputs as ``TargetSpec`` objects for Microplex."""
    return get_targets(
        db_path=db_path,
        jurisdiction=jurisdiction,
        year=year,
        sources=sources,
        variables=variables,
    )


def build_microplex_constraints(
    microdata: pd.DataFrame,
    targets: list[TargetSpec] | None = None,
    *,
    db_path: Path | None = None,
    jurisdiction: str = "us",
    year: int | None = None,
    sources: list[str] | None = None,
    variables: list[str] | None = None,
    tolerance: float = 0.01,
    min_obs: int = 0,
) -> list[Constraint]:
    """
    Build flat Microplex calibration constraints from Arch DB target inputs.

    Args:
        microdata: One row per calibrated unit.
        targets: Optional preloaded target specs. If omitted, targets are loaded
            from the Arch SQLite database.
        db_path: Optional target database path.
        jurisdiction: Jurisdiction prefix to load from the database.
        year: Optional target period.
        sources: Optional data-source filters, such as ``["irs-soi"]``.
        variables: Optional target variable filters.
        tolerance: Default calibration tolerance.
        min_obs: Drop constraints with fewer non-zero indicator entries.

    Returns:
        Constraint objects accepted by the shared calibration methods.
    """
    if targets is None:
        targets = load_microplex_targets(
            db_path=db_path,
            jurisdiction=jurisdiction,
            year=year,
            sources=sources,
            variables=variables,
        )

    constraints = build_constraint_matrix(
        microdata=microdata,
        targets=targets,
        tolerance=tolerance,
    )
    return _filter_constraints_by_obs(constraints, min_obs=min_obs)


def build_hierarchical_microplex_constraints(
    hh_df: pd.DataFrame,
    person_df: pd.DataFrame,
    targets: list[TargetSpec] | None = None,
    *,
    db_path: Path | None = None,
    jurisdiction: str = "us",
    year: int | None = None,
    sources: list[str] | None = None,
    variables: list[str] | None = None,
    tolerance: float = 0.01,
    hh_id_col: str = "household_id",
    tax_unit_df: pd.DataFrame | None = None,
    min_obs: int = 0,
) -> list[Constraint]:
    """
    Build household-weighted Microplex constraints from Arch DB target inputs.

    This is the adapter for hierarchical microdata where household weights must
    satisfy person-level or tax-unit-level aggregate targets.
    """
    if targets is None:
        targets = load_microplex_targets(
            db_path=db_path,
            jurisdiction=jurisdiction,
            year=year,
            sources=sources,
            variables=variables,
        )

    constraints = build_hierarchical_constraint_matrix(
        hh_df=hh_df,
        person_df=person_df,
        targets=targets,
        tolerance=tolerance,
        hh_id_col=hh_id_col,
        tax_unit_df=tax_unit_df,
    )
    return _filter_constraints_by_obs(constraints, min_obs=min_obs)


def constraints_to_ipf_dicts(
    constraints: list[Constraint],
) -> list[dict[str, Any]]:
    """Convert shared ``Constraint`` objects to legacy IPF constraint dicts."""
    return [
        {
            "indicator": constraint.indicator,
            "target_value": constraint.target_value,
            "variable": constraint.variable,
            "target_type": constraint.target_type.value,
            "stratum": constraint.stratum_name,
            "n_obs": _count_nonzero_indicator(constraint.indicator),
        }
        for constraint in constraints
    ]


def _filter_constraints_by_obs(
    constraints: list[Constraint],
    min_obs: int,
) -> list[Constraint]:
    if min_obs <= 0:
        return constraints
    return [
        constraint
        for constraint in constraints
        if _count_nonzero_indicator(constraint.indicator) >= min_obs
    ]


def _count_nonzero_indicator(indicator: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(indicator)))
