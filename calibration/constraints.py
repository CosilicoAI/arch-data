"""
Constraint matrix building for calibration.

Maps targets to microdata aggregations, building indicator vectors
for each constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from db.schema import TargetType
from .targets import TargetSpec


@dataclass
class Constraint:
    """
    A single calibration constraint.

    Represents the equation: sum(weights * indicator) = target_value

    Attributes:
        indicator: Vector of indicator values (length = microdata rows)
        target_value: Target aggregate to match
        variable: Arch target input variable ID
        target_type: COUNT or AMOUNT
        tolerance: Allowed deviation from target (fraction)
        stratum_name: Human-readable description
    """

    indicator: np.ndarray
    target_value: float
    variable: str
    target_type: TargetType
    tolerance: float = 0.01
    stratum_name: Optional[str] = None


def apply_stratum_constraints(
    microdata: pd.DataFrame,
    constraints: list[tuple[str, str, str]],
) -> pd.Series:
    """
    Return boolean mask for records matching stratum.

    Args:
        microdata: DataFrame with microdata
        constraints: List of (variable, operator, value) tuples

    Returns:
        Boolean Series indicating which rows match all constraints
    """
    mask = pd.Series(True, index=microdata.index)

    for variable, operator, value in constraints:
        if variable not in microdata.columns:
            # Skip constraints for variables not in microdata
            continue

        col = microdata[variable]

        # Parse value based on column dtype
        if pd.api.types.is_numeric_dtype(col):
            parsed_value = float(value)
        else:
            parsed_value = value

        if operator == "==":
            mask &= col == parsed_value
        elif operator == "!=":
            mask &= col != parsed_value
        elif operator == ">":
            mask &= col > parsed_value
        elif operator == ">=":
            mask &= col >= parsed_value
        elif operator == "<":
            mask &= col < parsed_value
        elif operator == "<=":
            mask &= col <= parsed_value
        elif operator == "in":
            # Value should be comma-separated list
            values = [v.strip() for v in value.split(",")]
            if pd.api.types.is_numeric_dtype(col):
                values = [float(v) for v in values]
            mask &= col.isin(values)
        else:
            raise ValueError(f"Unknown operator: {operator}")

    return mask


def build_constraint_matrix(
    microdata: pd.DataFrame,
    targets: list[TargetSpec],
    tolerance: float = 0.01,
) -> list[Constraint]:
    """
    Build constraint matrix from targets and microdata.

    For each target, creates an indicator vector:
    - COUNT type: indicator is 1 for matching rows, 0 otherwise
    - AMOUNT type: indicator is variable value for matching rows, 0 otherwise

    Args:
        microdata: DataFrame with microdata records
        targets: List of TargetSpec objects from get_targets()
        tolerance: Default allowed deviation (can be overridden per target)

    Returns:
        List of Constraint objects ready for calibration
    """
    constraints = []

    for target in targets:
        # Build stratum mask
        mask = apply_stratum_constraints(microdata, target.constraints)

        # Build indicator vector based on target type
        if target.target_type == TargetType.COUNT:
            # For counts, indicator is just the mask
            indicator = mask.astype(float).values
        elif target.target_type == TargetType.AMOUNT:
            # For amounts, indicator is variable * mask
            if target.variable in microdata.columns:
                indicator = (microdata[target.variable] * mask).values
            else:
                # If variable not in microdata, use zeros
                indicator = np.zeros(len(microdata))
        else:
            # RATE type - not commonly used for calibration
            indicator = mask.astype(float).values

        constraint = Constraint(
            indicator=indicator,
            target_value=target.value,
            variable=target.variable,
            target_type=target.target_type,
            tolerance=target.tolerance if target.tolerance else tolerance,
            stratum_name=target.stratum_name,
        )
        constraints.append(constraint)

    return constraints


def build_hierarchical_constraint_matrix(
    hh_df: pd.DataFrame,
    person_df: pd.DataFrame,
    targets: list[TargetSpec],
    tolerance: float = 0.01,
    hh_id_col: str = "household_id",
    tax_unit_df: Optional[pd.DataFrame] = None,
) -> list[Constraint]:
    """
    Build constraint matrix aggregating person/tax-unit targets to household level.

    For hierarchical microdata where weights are at the household level,
    this function handles person-level and tax-unit-level targets by
    aggregating (count or sum) to the household level.

    The key insight: since all calibration targets are sums, and all persons
    in a household share the household weight, we can pre-aggregate:
        sum(person_weight * is_age_6_10) = sum(hh_weight * n_age_6_10_in_hh)

    Args:
        hh_df: Household-level DataFrame (one row per household)
        person_df: Person-level DataFrame with hh_id_col linking to households
        targets: List of TargetSpec objects from get_targets()
        tolerance: Default allowed deviation
        hh_id_col: Column name for household ID in both DataFrames
        tax_unit_df: Optional tax-unit-level DataFrame

    Returns:
        List of Constraint objects with indicators at household level
    """
    from .variables import get_entity, infer_target_level

    constraints = []

    for target in targets:
        # Determine what level this target operates at
        # 1. Try to infer from the target variable reference
        # 2. Fall back to inferring from constraint variables
        try:
            if ":" in target.variable and "#" in target.variable:
                level = get_entity(target.variable)
            else:
                level = infer_target_level(target.constraints)
        except Exception:
            level = infer_target_level(target.constraints)

        # Build indicator based on level
        if level == "household":
            # Direct household-level constraint
            indicator = _build_household_indicator(
                hh_df, target, hh_id_col
            )
        elif level == "tax_unit" and tax_unit_df is not None:
            # Aggregate from tax unit to household
            indicator = _build_aggregated_indicator(
                hh_df, tax_unit_df, target, hh_id_col
            )
        else:
            # Aggregate from person to household (default)
            indicator = _build_aggregated_indicator(
                hh_df, person_df, target, hh_id_col
            )

        constraint = Constraint(
            indicator=indicator,
            target_value=target.value,
            variable=target.variable,
            target_type=target.target_type,
            tolerance=target.tolerance if target.tolerance else tolerance,
            stratum_name=target.stratum_name,
        )
        constraints.append(constraint)

    return constraints


def _build_household_indicator(
    hh_df: pd.DataFrame,
    target: TargetSpec,
    hh_id_col: str,
) -> np.ndarray:
    """Build indicator for household-level targets."""
    mask = apply_stratum_constraints(hh_df, target.constraints)

    if target.target_type == TargetType.COUNT:
        return mask.astype(float).values
    elif target.target_type == TargetType.AMOUNT:
        # Extract variable name from full reference if needed
        var_name = target.variable
        if "#" in var_name:
            var_name = var_name.split("#")[-1]

        if var_name in hh_df.columns:
            return (hh_df[var_name] * mask).values
        else:
            return np.zeros(len(hh_df))
    else:
        return mask.astype(float).values


def _build_aggregated_indicator(
    hh_df: pd.DataFrame,
    source_df: pd.DataFrame,
    target: TargetSpec,
    hh_id_col: str,
) -> np.ndarray:
    """
    Build indicator by aggregating from person/tax_unit to household.

    For COUNT targets: count matching records per household
    For AMOUNT targets: sum variable value for matching records per household
    """
    # Apply constraints to source (person or tax_unit) data
    mask = apply_stratum_constraints(source_df, target.constraints)
    filtered = source_df[mask]

    if len(filtered) == 0:
        # No matching records
        return np.zeros(len(hh_df))

    if target.target_type == TargetType.COUNT:
        # Count matching records per household
        agg = filtered.groupby(hh_id_col).size()
    elif target.target_type == TargetType.AMOUNT:
        # Sum variable for matching records per household
        var_name = target.variable
        if "#" in var_name:
            var_name = var_name.split("#")[-1]

        if var_name in filtered.columns:
            agg = filtered.groupby(hh_id_col)[var_name].sum()
        else:
            # Variable not found, return zeros
            return np.zeros(len(hh_df))
    else:
        # RATE type - treat as count
        agg = filtered.groupby(hh_id_col).size()

    # Map aggregated values back to household DataFrame order
    # Households with no matching persons get 0
    indicator = hh_df[hh_id_col].map(agg).fillna(0).values

    return indicator
