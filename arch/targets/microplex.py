"""Microplex adapters for Arch target inputs."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
from db.schema import DataSource, TargetType


@dataclass(frozen=True)
class SOIAgingFactors:
    """Declared factors used to age SOI target inputs for Microplex."""

    source_year: int
    target_year: int
    count_factor: float
    amount_factor: float
    count_method: str
    amount_method: str


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


def get_soi_aging_factors(
    source_year: int,
    target_year: int,
    db_path: Path | None = None,
    jurisdiction: str = "us",
) -> SOIAgingFactors:
    """
    Resolve source-backed factors for aging SOI target inputs.

    Counts are scaled by labor force: BLS annual labor-force counts when
    available, then CBO labor-force projections for years beyond BLS coverage.
    Amounts are scaled by aggregate SOI adjusted gross income. If the target
    year is beyond available SOI AGI data, the last observed annual AGI growth
    rate is carried forward and declared in the method string.
    """
    if source_year == target_year:
        return SOIAgingFactors(
            source_year=source_year,
            target_year=target_year,
            count_factor=1.0,
            amount_factor=1.0,
            count_method="identity",
            amount_method="identity",
        )

    source_labor_force = _get_labor_force_target(
        year=source_year,
        db_path=db_path,
        jurisdiction=jurisdiction,
    )
    target_labor_force, count_source = _get_labor_force_target_with_source(
        year=target_year,
        db_path=db_path,
        jurisdiction=jurisdiction,
    )
    source_aggregate_income = _soi_total_agi_value(
        year=source_year,
        db_path=db_path,
        jurisdiction=jurisdiction,
    )
    target_aggregate_income, amount_method = _soi_total_agi_for_year(
        target_year=target_year,
        db_path=db_path,
        jurisdiction=jurisdiction,
    )

    return SOIAgingFactors(
        source_year=source_year,
        target_year=target_year,
        count_factor=target_labor_force / source_labor_force,
        amount_factor=target_aggregate_income / source_aggregate_income,
        count_method=f"{count_source}_labor_force_ratio",
        amount_method=amount_method,
    )


def age_soi_targets(
    targets: list[TargetSpec],
    target_year: int,
    *,
    db_path: Path | None = None,
    jurisdiction: str = "us",
    factors: SOIAgingFactors | None = None,
) -> list[TargetSpec]:
    """
    Age IRS SOI target inputs to a Microplex model year.

    This composes model-year targets from source records. It does not mutate
    Arch records in the database.
    """
    source_years = {
        target.period
        for target in targets
        if target.source == DataSource.IRS_SOI and target.period != target_year
    }
    if len(source_years) > 1:
        raise ValueError(
            "SOI target aging expects one SOI source year at a time; "
            f"got {sorted(source_years)}."
        )
    if not source_years:
        return targets

    source_year = source_years.pop()
    if factors is None:
        factors = get_soi_aging_factors(
            source_year=source_year,
            target_year=target_year,
            db_path=db_path,
            jurisdiction=jurisdiction,
        )
    elif factors.source_year != source_year or factors.target_year != target_year:
        raise ValueError(
            "SOI aging factors do not match target years: "
            f"targets are {source_year}->{target_year}, "
            f"factors are {factors.source_year}->{factors.target_year}."
        )

    aged_targets = []
    for target in targets:
        if target.source != DataSource.IRS_SOI or target.period == target_year:
            aged_targets.append(target)
            continue

        if target.target_type == TargetType.COUNT:
            factor = factors.count_factor
            method = factors.count_method
        elif target.target_type == TargetType.AMOUNT:
            factor = factors.amount_factor
            method = factors.amount_method
        else:
            factor = 1.0
            method = "rate_unchanged"

        aged_targets.append(
            replace(
                target,
                value=target.value * factor,
                period=target_year,
                stratum_name=_aged_stratum_name(
                    target.stratum_name,
                    source_year=target.period,
                    target_year=target_year,
                    method=method,
                ),
            )
        )

    return aged_targets


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


def _aged_stratum_name(
    stratum_name: str | None,
    *,
    source_year: int,
    target_year: int,
    method: str,
) -> str:
    base = stratum_name or "SOI target"
    return f"{base} (SOI aged {source_year}->{target_year}; {method})"


def _get_labor_force_target(
    *,
    year: int,
    db_path: Path | None,
    jurisdiction: str,
) -> float:
    value, _ = _get_labor_force_target_with_source(
        year=year,
        db_path=db_path,
        jurisdiction=jurisdiction,
    )
    return value


def _get_labor_force_target_with_source(
    *,
    year: int,
    db_path: Path | None,
    jurisdiction: str,
) -> tuple[float, str]:
    bls_value = _optional_target_value(
        db_path=db_path,
        jurisdiction=jurisdiction,
        year=year,
        source=DataSource.BLS,
        variable="labor_force_count",
    )
    if bls_value is not None:
        return bls_value, "bls"

    cbo_value = _optional_target_value(
        db_path=db_path,
        jurisdiction=jurisdiction,
        year=year,
        source=DataSource.CBO,
        variable="labor_force",
    )
    if cbo_value is not None:
        return cbo_value, "cbo"

    raise ValueError(f"No BLS/CBO labor-force target found for {year}.")


def _soi_total_agi_for_year(
    *,
    target_year: int,
    db_path: Path | None,
    jurisdiction: str,
) -> tuple[float, str]:
    target_agi = _optional_soi_total_agi_value(
        year=target_year,
        db_path=db_path,
        jurisdiction=jurisdiction,
    )
    if target_agi is not None:
        return target_agi, "soi_total_agi_ratio"

    available = _available_soi_total_agi_values(
        db_path=db_path,
        jurisdiction=jurisdiction,
        start_year=target_year - 20,
        end_year=target_year,
    )
    if len(available) < 2:
        raise ValueError(
            "Need at least two SOI total AGI years to extrapolate "
            f"aggregate income to {target_year}."
        )

    latest_year = max(available)
    previous_year = max(year for year in available if year < latest_year)
    annual_growth = available[latest_year] / available[previous_year]
    years_forward = target_year - latest_year
    projected = available[latest_year] * annual_growth**years_forward
    return (
        projected,
        "soi_total_agi_last_growth_extrapolation",
    )


def _available_soi_total_agi_values(
    *,
    db_path: Path | None,
    jurisdiction: str,
    start_year: int,
    end_year: int,
) -> dict[int, float]:
    values = {}
    for year in range(start_year, end_year + 1):
        value = _optional_soi_total_agi_value(
            year=year,
            db_path=db_path,
            jurisdiction=jurisdiction,
        )
        if value is not None:
            values[year] = value
    return values


def _soi_total_agi_value(
    *,
    year: int,
    db_path: Path | None,
    jurisdiction: str,
) -> float:
    value = _optional_soi_total_agi_value(
        year=year,
        db_path=db_path,
        jurisdiction=jurisdiction,
    )
    if value is None:
        raise ValueError(f"No SOI total AGI target found for {year}.")
    return value


def _optional_soi_total_agi_value(
    *,
    year: int,
    db_path: Path | None,
    jurisdiction: str,
) -> float | None:
    targets = get_targets(
        db_path=db_path,
        jurisdiction=jurisdiction,
        year=year,
        sources=[DataSource.IRS_SOI.value],
        variables=["adjusted_gross_income"],
    )
    if not targets:
        return None

    for target in targets:
        if target.stratum_name == "US All Filers":
            return target.value
    for target in targets:
        if target.constraints == [("is_tax_filer", "==", "1")]:
            return target.value
    return None


def _target_value(
    *,
    db_path: Path | None,
    jurisdiction: str,
    year: int,
    source: DataSource,
    variable: str,
) -> float:
    value = _optional_target_value(
        db_path=db_path,
        jurisdiction=jurisdiction,
        year=year,
        source=source,
        variable=variable,
    )
    if value is None:
        raise ValueError(f"No {source.value} {variable} target found for {year}.")
    return value


def _optional_target_value(
    *,
    db_path: Path | None,
    jurisdiction: str,
    year: int,
    source: DataSource,
    variable: str,
) -> float | None:
    targets = get_targets(
        db_path=db_path,
        jurisdiction=jurisdiction,
        year=year,
        sources=[source.value],
        variables=[variable],
    )
    if not targets:
        return None
    if len(targets) == 1:
        return targets[0].value

    unconstrained = [target for target in targets if not target.constraints]
    if len(unconstrained) == 1:
        return unconstrained[0].value
    return targets[0].value
