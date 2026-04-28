"""
Microplex pipeline: build calibrated microdata from Arch target inputs.

Reads CPS microdata and Arch target inputs, runs calibration, and writes
calibrated microplex output locally or back to Supabase.

Calibration methods:
- generalized-rake: bounded dual raking for count and amount constraints
- ipf: fast count-only iterative proportional fitting

Usage:
    python -m micro.us.pipeline --year 2024
    python -m micro.us.pipeline --year 2024 --dry-run
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from arch.client import get_supabase_client
from arch.microdata import query_cps_asec
from arch.targets import TargetSpec, TargetType, query_targets
from micro.us.targets import (
    MicroplexTargetProfile,
    TargetCompositionResult,
    compose_microplex_targets,
    load_microplex_targets,
)


@dataclass
class CalibrationResult:
    """Results from Microplex weight calibration."""
    original_weights: np.ndarray
    calibrated_weights: np.ndarray
    adjustment_factors: np.ndarray
    targets_before: Dict[str, Dict]
    targets_after: Dict[str, Dict]
    diagnostics: pd.DataFrame
    success: bool
    message: str
    l2_loss: float
    method: str


AGING_NOTE_RE = re.compile(
    r"SOI aged (?P<source_year>\d+)->(?P<target_year>\d+); "
    r"(?P<method>[^;)]+)(?:; factor x(?P<factor>[-+0-9.eE]+))?"
)
TARGET_VARIABLE_COLUMNS = {
    "adjusted_gross_income": "adjusted_gross_income",
    "employment_income": "wage_income",
}
SUPPORTED_TARGET_VARIABLES = {"tax_unit_count", *TARGET_VARIABLE_COLUMNS}
SUPPORTED_CONSTRAINT_VARIABLES = {
    "adjusted_gross_income",
    "is_tax_filer",
    "agi_bracket",
    "state_fips",
}


def load_cps_from_supabase(year: int, limit: int = 200000) -> pd.DataFrame:
    """Load raw CPS person data from Supabase."""
    print(f"Loading CPS ASEC {year} from Supabase...")
    df = query_cps_asec(year, table_type="person", limit=limit)
    print(f"  Loaded {len(df):,} person records")
    return df


def load_cps_from_local_file(
    year: int,
    path: Path | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load CPS person data from a local parquet file."""
    if path is None:
        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / "micro" / "us" / f"cps_{year}.parquet"

    print(f"Loading CPS ASEC {year} from {path}...")
    df = pd.read_parquet(path)
    if limit is not None:
        df = df.head(limit).copy()
    print(f"  Loaded {len(df):,} person records")
    return df


def load_targets_from_supabase(year: int) -> List[Dict[str, Any]]:
    """Load calibration targets from Supabase."""
    print(f"Loading targets for {year} from Supabase...")
    # Get targets for US jurisdictions (both "US" and "US_FEDERAL")
    all_targets = query_targets(year=year)
    targets = [t for t in all_targets
               if t.get("strata", {}).get("jurisdiction", "").startswith("US")]
    print(f"  Loaded {len(targets)} targets")
    return targets


def load_targets_from_db(
    year: int,
    db_path: Path | None = None,
    jurisdiction: str = "us",
    sources: list[str] | None = None,
    variables: list[str] | None = None,
) -> List[TargetSpec]:
    """Load calibration target inputs from the local Arch SQLite database."""
    print(f"Loading target inputs for {year} from Arch DB...")
    targets = load_microplex_targets(
        db_path=db_path,
        jurisdiction=jurisdiction,
        year=year,
        sources=sources,
        variables=variables,
    )
    print(f"  Loaded {len(targets)} target inputs")
    return targets


def build_tax_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build tax units from person-level CPS data.

    Simplified version - groups by household and assigns tax unit based on
    marital status and age. Full version would use relationship variables.
    """
    # Use raw_data if available for additional columns
    if "raw_data" in df.columns:
        # Extract key fields from raw_data
        for col in ["A_MARITL", "A_FAMREL", "TAX_ID"]:
            df[col.lower()] = df["raw_data"].apply(lambda x: x.get(col) if isinstance(x, dict) else None)

    if "tax_unit_id" in df.columns:
        return build_tax_units_from_census_tax_ids(df)

    # Calculate total income
    df["total_income"] = _numeric_first(
        df,
        ["ptotval", "total_person_income", "income"],
    )
    df["wage_income"] = _numeric_first(
        df,
        ["wsal_val", "wage_salary_income", "wage_income"],
    )
    df["self_employment_income"] = _numeric_first(
        df,
        ["semp_val", "self_employment_income"],
    ) + _numeric_first(df, ["frse_val", "farm_self_employment_income"])

    # Simple AGI estimate (wages + SE - 1/2 SE tax)
    se_tax = np.maximum(df["self_employment_income"] * 0.0765, 0)
    df["adjusted_gross_income"] = df["wage_income"] + df["self_employment_income"] - se_tax / 2

    # Weight
    if "marsupwt" in df.columns:
        df["weight"] = pd.to_numeric(df["marsupwt"], errors="coerce").fillna(0) / 100
    elif "weight" in df.columns:
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0)
    elif "march_supplement_weight" in df.columns:
        df["weight"] = pd.to_numeric(
            df["march_supplement_weight"],
            errors="coerce",
        ).fillna(0) / 100
    else:
        df["weight"] = 1.0

    if "state_fips" not in df.columns and "gestfips" in df.columns:
        df["state_fips"] = df["gestfips"]
    if "age" not in df.columns and "a_age" in df.columns:
        df["age"] = df["a_age"]
    if "household_id" not in df.columns and "ph_seq" in df.columns:
        df["household_id"] = df["ph_seq"]

    # Filter to likely filers
    filer_mask = (
        (df["total_income"] > 13850)
        | (df["wage_income"] > 0)
        | (df["self_employment_income"] > 0)
    )
    df = df[filer_mask].copy()
    df["is_tax_filer"] = 1
    print(f"  Filtered to {len(df):,} likely filers")

    return df


def build_tax_units_from_census_tax_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate person-level CPS records to Census-provided tax units."""
    df = df.copy()

    income_columns = [
        "total_person_income",
        "income",
        "wage_salary_income",
        "wage_income",
        "self_employment_income",
        "farm_self_employment_income",
        "interest_income",
        "dividend_income",
        "rental_income",
        "unemployment_compensation",
        "other_income",
    ]
    for column in income_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    if "weight" in df.columns:
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0)
    elif "march_supplement_weight" in df.columns:
        df["weight"] = pd.to_numeric(
            df["march_supplement_weight"],
            errors="coerce",
        ).fillna(0) / 100
    else:
        df["weight"] = 1.0

    if "line_number" in df.columns:
        sort_column = "line_number"
    elif "person_seq" in df.columns:
        sort_column = "person_seq"
    else:
        sort_column = None

    group_columns = ["tax_unit_id"]
    if "household_id" in df.columns:
        group_columns = ["household_id", "tax_unit_id"]

    rows = []
    for _, group in df.groupby(group_columns, sort=False, dropna=False):
        if sort_column is not None:
            group = group.sort_values(sort_column)
        head = group.iloc[0]

        wage_income = _sum_columns(group, ["wage_salary_income", "wage_income"])
        self_employment_income = _sum_columns(
            group,
            ["self_employment_income", "farm_self_employment_income"],
        )
        interest_income = _sum_columns(group, ["interest_income"])
        dividend_income = _sum_columns(group, ["dividend_income"])
        rental_income = _sum_columns(group, ["rental_income"])
        unemployment = _sum_columns(group, ["unemployment_compensation"])
        other_income = _sum_columns(group, ["other_income"])
        total_income = _sum_columns(group, ["total_person_income"])
        if total_income == 0:
            total_income = _sum_columns(group, ["income"])

        se_tax_adjustment = max(self_employment_income, 0) * 0.0765 / 2
        adjusted_gross_income = (
            wage_income
            + self_employment_income
            - se_tax_adjustment
            + interest_income
            + dividend_income
            + rental_income
            + unemployment
            + other_income
        )

        rows.append(
            {
                "tax_unit_id": head.get("tax_unit_id"),
                "household_id": head.get("household_id"),
                "person_id": head.get("person_id"),
                "person_count": len(group),
                "age": head.get("age", head.get("a_age")),
                "state_fips": head.get("state_fips", head.get("gestfips")),
                "weight": head["weight"],
                "total_income": total_income,
                "wage_income": wage_income,
                "self_employment_income": self_employment_income,
                "interest_income": interest_income,
                "dividend_income": dividend_income,
                "rental_income": rental_income,
                "unemployment_compensation": unemployment,
                "other_income": other_income,
                "adjusted_gross_income": adjusted_gross_income,
            }
        )

    tax_units = pd.DataFrame(rows)
    filer_mask = (
        (tax_units["total_income"] > 13_850)
        | (tax_units["wage_income"] > 0)
        | (tax_units["self_employment_income"] != 0)
    )
    tax_units = tax_units[filer_mask].copy()
    tax_units["is_tax_filer"] = 1
    print(f"  Aggregated to {len(tax_units):,} likely filing tax units")
    return tax_units


def _sum_columns(df: pd.DataFrame, columns: list[str]) -> float:
    total = 0.0
    for column in columns:
        if column in df.columns:
            total += float(pd.to_numeric(df[column], errors="coerce").fillna(0).sum())
    return total


def _base_target_diagnostic(
    target: TargetSpec,
    target_index: int,
) -> dict[str, Any]:
    aging = _parse_aging_note(target.stratum_name)
    return {
        "target_index": target_index,
        "source": _enum_value(target.source),
        "variable": target.variable,
        "target_type": _enum_value(target.target_type),
        "period": target.period,
        "source_period": aging.get("source_year", target.period),
        "stratum": target.stratum_name,
        "constraints": json.dumps(target.constraints, separators=(",", ":")),
        "target_value": target.value,
        "role": "",
        "status": "",
        "drop_reason": "",
        "n_obs": 0,
        "aging_method": aging.get("method"),
        "aging_factor": aging.get("factor"),
        "pre_value": np.nan,
        "pre_error": np.nan,
        "post_value": np.nan,
        "post_error": np.nan,
    }


def _base_dict_target_diagnostic(
    target: dict[str, Any],
    target_index: int,
) -> dict[str, Any]:
    stratum = target.get("strata", {})
    constraints = stratum.get("stratum_constraints", [])
    return {
        "target_index": target_index,
        "source": target.get("source"),
        "variable": target.get("variable"),
        "target_type": target.get("target_type"),
        "period": target.get("period"),
        "source_period": target.get("period"),
        "stratum": stratum.get("name", "unknown"),
        "constraints": json.dumps(constraints, separators=(",", ":"), default=str),
        "target_value": target.get("value"),
        "role": "",
        "status": "",
        "drop_reason": "",
        "n_obs": 0,
        "aging_method": None,
        "aging_factor": None,
        "pre_value": np.nan,
        "pre_error": np.nan,
        "post_value": np.nan,
        "post_error": np.nan,
    }


def _parse_aging_note(stratum_name: str | None) -> dict[str, Any]:
    if not stratum_name:
        return {}
    match = AGING_NOTE_RE.search(stratum_name)
    if not match:
        return {}
    factor = match.group("factor")
    return {
        "source_year": int(match.group("source_year")),
        "target_year": int(match.group("target_year")),
        "method": match.group("method"),
        "factor": float(factor) if factor is not None else np.nan,
    }


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _mark_unsupported(
    diagnostic: dict[str, Any],
    drop_reason: str,
) -> dict[str, Any]:
    diagnostic.update(
        role="unsupported",
        status="dropped",
        drop_reason=drop_reason,
    )
    return diagnostic


def _mark_diagnostic(
    diagnostic: dict[str, Any],
    drop_reason: str,
) -> dict[str, Any]:
    diagnostic.update(
        role="diagnostic",
        status="dropped",
        drop_reason=drop_reason,
    )
    return diagnostic


def _mark_holdout(diagnostic: dict[str, Any]) -> dict[str, Any]:
    diagnostic.update(role="holdout", status="dropped", drop_reason="holdout")
    return diagnostic


def _mark_active(diagnostic: dict[str, Any]) -> dict[str, Any]:
    diagnostic.update(role="active", status="used", drop_reason="")
    return diagnostic


def _numeric_first(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Return the first available numeric column, or zeros if none exist."""
    for column in columns:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").fillna(0)
    return pd.Series(0.0, index=df.index)


def assign_agi_bracket(agi: np.ndarray) -> np.ndarray:
    """Assign each record to an AGI bracket matching SOI data."""
    brackets = [
        ("under_1", -np.inf, 1),
        ("1_to_5k", 1, 5000),
        ("5k_to_10k", 5000, 10000),
        ("10k_to_15k", 10000, 15000),
        ("15k_to_20k", 15000, 20000),
        ("20k_to_25k", 20000, 25000),
        ("25k_to_30k", 25000, 30000),
        ("30k_to_40k", 30000, 40000),
        ("40k_to_50k", 40000, 50000),
        ("50k_to_75k", 50000, 75000),
        ("75k_to_100k", 75000, 100000),
        ("100k_to_200k", 100000, 200000),
        ("200k_to_500k", 200000, 500000),
        ("500k_to_1m", 500000, 1000000),
        ("1m_to_1_5m", 1000000, 1500000),
        ("1_5m_to_2m", 1500000, 2000000),
        ("2m_to_5m", 2000000, 5000000),
        ("5m_to_10m", 5000000, 10000000),
        ("10m_plus", 10000000, np.inf),
    ]

    result = np.empty(len(agi), dtype=object)
    for name, low, high in brackets:
        mask = (agi >= low) & (agi < high)
        result[mask] = name

    return result


def _build_microplex_constraint_dict(
    df: pd.DataFrame,
    *,
    variable: str,
    target_type: TargetType | str,
    target_value: float,
    stratum_name: str | None,
    stratum_constraints: list[tuple[str, str, str]],
) -> dict[str, Any] | None:
    mask = _stratum_mask(df, stratum_constraints)
    if mask is None:
        return None

    target_type_value = _enum_value(target_type)
    if variable == "tax_unit_count":
        indicator = mask.astype(float).to_numpy()
    else:
        column = TARGET_VARIABLE_COLUMNS.get(variable)
        if column is None or column not in df.columns:
            return None
        values = pd.to_numeric(df[column], errors="coerce").fillna(0).to_numpy()
        if target_type_value == TargetType.COUNT.value:
            indicator = (mask.to_numpy() & (values > 0)).astype(float)
        elif target_type_value == TargetType.AMOUNT.value:
            indicator = mask.astype(float).to_numpy() * values
        else:
            return None

    return {
        "indicator": indicator,
        "target_value": target_value,
        "variable": variable,
        "target_type": target_type_value,
        "stratum": stratum_name,
        "n_obs": int(np.count_nonzero(indicator)),
    }


def _stratum_mask(
    df: pd.DataFrame,
    stratum_constraints: list[tuple[str, str, str]],
) -> pd.Series | None:
    mask = pd.Series(True, index=df.index)
    for variable, operator, value in stratum_constraints:
        if variable == "is_tax_filer" and variable not in df.columns:
            col = pd.Series([1] * len(df), index=df.index)
        elif variable in df.columns:
            col = df[variable]
        else:
            return None

        parsed_value: float | int | str = value
        if pd.api.types.is_numeric_dtype(col):
            parsed_value = float(value)

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
        else:
            return None
    return mask


def build_constraints_from_target_specs(
    df: pd.DataFrame,
    targets: List[TargetSpec],
    min_obs: int = 100,
    include_amounts: bool = False,
    holdout_variables: tuple[str, ...] = (),
) -> List[Dict]:
    """
    Build legacy IPF constraint dicts from Arch DB ``TargetSpec`` objects.

    This keeps the old IPF pipeline working while moving the target source from
    Supabase-shaped dictionaries to the local Arch target database.
    """
    constraints, _, _ = build_constraints_and_diagnostics_from_target_specs(
        df,
        targets,
        min_obs=min_obs,
        include_amounts=include_amounts,
        holdout_variables=holdout_variables,
    )
    return constraints


def build_constraints_and_diagnostics_from_target_specs(
    df: pd.DataFrame,
    targets: List[TargetSpec],
    min_obs: int = 100,
    include_amounts: bool = False,
    holdout_variables: tuple[str, ...] = (),
) -> tuple[List[Dict], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build IPF constraints and one diagnostic row per loaded target."""
    df = df.copy()
    df["agi_bracket"] = assign_agi_bracket(df["adjusted_gross_income"].values)

    constraints = []
    evaluation_constraints = []
    diagnostics = []
    seen_keys = set()
    holdout_variable_set = set(holdout_variables)
    for target_index, target in enumerate(targets):
        diagnostic = _base_target_diagnostic(target, target_index)

        if target.variable not in SUPPORTED_TARGET_VARIABLES:
            _mark_unsupported(diagnostic, "unsupported_variable")
            diagnostics.append(diagnostic)
            continue
        if target.target_type == TargetType.RATE:
            _mark_unsupported(diagnostic, "rate_target")
            diagnostics.append(diagnostic)
            continue
        if target.target_type == TargetType.AMOUNT and target.value <= 0:
            _mark_unsupported(diagnostic, "nonpositive_amount_target")
            diagnostics.append(diagnostic)
            continue
        if any(
            variable not in SUPPORTED_CONSTRAINT_VARIABLES
            for variable, _, _ in target.constraints
        ):
            _mark_unsupported(diagnostic, "unsupported_constraint")
            diagnostics.append(diagnostic)
            continue

        constraint_dict = _build_microplex_constraint_dict(
            df,
            variable=target.variable,
            target_type=target.target_type,
            target_value=target.value,
            stratum_name=target.stratum_name,
            stratum_constraints=target.constraints,
        )
        if constraint_dict is None:
            _mark_unsupported(diagnostic, "could_not_build_constraint")
            diagnostics.append(diagnostic)
            continue

        diagnostic["n_obs"] = constraint_dict["n_obs"]
        key = (target.stratum_name, target.variable, target.target_type)
        drop_reason = ""
        is_holdout = target.variable in holdout_variable_set
        if is_holdout:
            drop_reason = "holdout"
        elif key in seen_keys:
            drop_reason = "duplicate_target"
        elif target.target_type == TargetType.AMOUNT and not include_amounts:
            drop_reason = "amount_targets_disabled"
        elif constraint_dict["n_obs"] < min_obs:
            drop_reason = "insufficient_obs"

        if drop_reason:
            if is_holdout:
                _mark_holdout(diagnostic)
            else:
                _mark_diagnostic(diagnostic, drop_reason)
            diagnostic_index = len(diagnostics)
            constraint_dict["diagnostic_index"] = diagnostic_index
            diagnostics.append(diagnostic)
            evaluation_constraints.append(constraint_dict)
            continue

        seen_keys.add(key)
        _mark_active(diagnostic)
        diagnostic_index = len(diagnostics)
        constraint_dict["diagnostic_index"] = diagnostic_index
        diagnostics.append(diagnostic)
        constraints.append(constraint_dict)

    print(f"  Built {len(constraints)} constraints (min {min_obs} obs each)")
    return constraints, diagnostics, evaluation_constraints


def build_constraints_from_targets(
    df: pd.DataFrame,
    targets: List[Dict[str, Any]] | List[TargetSpec],
    min_obs: int = 100,
    include_amounts: bool = False,
    holdout_variables: tuple[str, ...] = (),
    return_diagnostics: bool = False,
) -> List[Dict]:
    """
    Build calibration constraints from Supabase targets.

    Currently supports:
    - Tax unit counts by AGI bracket (adjusted_gross_income ranges)
    - AGI totals by bracket (if include_amounts=True)
    - Wage/employment income counts and totals (if include_amounts=True for totals)

    Skips unsupported targets:
    - Filing status (need CPS marital status mapping)
    - Program participation (SNAP, SSI, OASDI - need program vars)
    - Population counts (different universe than tax filers)
    """
    if targets and isinstance(targets[0], TargetSpec):
        constraints, diagnostics, evaluation_constraints = (
            build_constraints_and_diagnostics_from_target_specs(
                df,
                targets,
                min_obs=min_obs,
                include_amounts=include_amounts,
                holdout_variables=holdout_variables,
            )
        )
        if return_diagnostics:
            return constraints, diagnostics, evaluation_constraints
        return constraints

    constraints = []
    evaluation_constraints = []
    diagnostics = []
    seen_keys = set()
    holdout_variable_set = set(holdout_variables)
    df = df.copy()

    # Precompute AGI brackets
    df["agi_bracket"] = assign_agi_bracket(df["adjusted_gross_income"].values)

    for target_index, target in enumerate(targets):
        diagnostic = _base_dict_target_diagnostic(target, target_index)
        variable = target["variable"]
        value = target["value"]
        target_type = target.get("target_type")

        # Only calibrate on supported variables
        if variable not in SUPPORTED_TARGET_VARIABLES:
            _mark_unsupported(diagnostic, "unsupported_variable")
            diagnostics.append(diagnostic)
            continue

        # Skip rate targets; optionally skip amount targets
        if target_type == "rate":
            _mark_unsupported(diagnostic, "rate_target")
            diagnostics.append(diagnostic)
            continue
        if target_type == "amount" and value <= 0:
            _mark_unsupported(diagnostic, "nonpositive_amount_target")
            diagnostics.append(diagnostic)
            continue

        stratum = target.get("strata", {})
        stratum_name = stratum.get("name", "unknown")
        stratum_constraints = stratum.get("stratum_constraints", [])

        # Skip strata with unsupported constraint types
        # (filing_status, snap, ssi, oasdi, etc.)
        has_unsupported = False
        for c in stratum_constraints:
            var = c.get("variable")
            if var not in SUPPORTED_CONSTRAINT_VARIABLES:
                has_unsupported = True
                break
        if has_unsupported:
            _mark_unsupported(diagnostic, "unsupported_constraint")
            diagnostics.append(diagnostic)
            continue

        tuple_constraints = [
            (
                constraint.get("variable"),
                constraint.get("operator", "=="),
                constraint.get("value"),
            )
            for constraint in stratum_constraints
        ]
        constraint_dict = _build_microplex_constraint_dict(
            df,
            variable=variable,
            target_type=target_type,
            target_value=value,
            stratum_name=stratum_name,
            stratum_constraints=tuple_constraints,
        )
        if constraint_dict is None:
            _mark_unsupported(diagnostic, "could_not_build_constraint")
            diagnostics.append(diagnostic)
            continue

        n_obs = constraint_dict["n_obs"]
        diagnostic["n_obs"] = int(n_obs)
        key = (stratum_name, variable, target_type)
        drop_reason = ""
        is_holdout = variable in holdout_variable_set
        if is_holdout:
            drop_reason = "holdout"
        elif key in seen_keys:
            drop_reason = "duplicate_target"
        elif target_type == "amount" and not include_amounts:
            drop_reason = "amount_targets_disabled"
        elif n_obs < min_obs:
            drop_reason = "insufficient_obs"

        if drop_reason:
            if is_holdout:
                _mark_holdout(diagnostic)
            else:
                _mark_diagnostic(diagnostic, drop_reason)
            diagnostic_index = len(diagnostics)
            constraint_dict["diagnostic_index"] = diagnostic_index
            diagnostics.append(diagnostic)
            evaluation_constraints.append(constraint_dict)
            continue

        seen_keys.add(key)
        _mark_active(diagnostic)
        diagnostic_index = len(diagnostics)
        constraint_dict["diagnostic_index"] = diagnostic_index
        constraints.append(constraint_dict)
        diagnostics.append(diagnostic)

    print(f"  Built {len(constraints)} constraints (min {min_obs} obs each)")
    if return_diagnostics:
        return constraints, diagnostics, evaluation_constraints
    return constraints


def ipf_calibrate(
    original_weights: np.ndarray,
    constraints: List[Dict],
    bounds: tuple = (0.2, 5.0),
    max_iter: int = 100,
    damping: tuple = (0.9, 1.1),
    verbose: bool = True,
) -> tuple:
    """
    Calibrate weights using Iterative Proportional Fitting (IPF).

    IPF iteratively adjusts weights to match marginal totals.
    This is 486x faster than IPF+GREG with only 7% worse L2 loss.

    Args:
        original_weights: Initial survey weights
        constraints: List of constraint dicts with 'indicator' and 'target_value'
        bounds: Min/max weight adjustment factors (default 0.2-5.0)
        max_iter: Number of IPF iterations (default 100)
        damping: Min/max adjustment ratio per iteration (default 0.9-1.1)
        verbose: Print progress

    Returns:
        (calibrated_weights, success, l2_loss)
    """
    n = len(original_weights)
    m = len(constraints)

    if verbose:
        print(f"IPF calibration: {n:,} weights, {m} constraints, {max_iter} iterations")

    # Build constraint matrix
    A = np.zeros((m, n))
    targets = np.zeros(m)

    for j, c in enumerate(constraints):
        A[j, :] = c["indicator"]
        targets[j] = c["target_value"]

    w = original_weights.copy()

    for iteration in range(max_iter):
        for j in range(m):
            achieved = A[j] @ w
            if achieved > 0:
                # Damped ratio to ensure convergence
                ratio = np.clip(targets[j] / achieved, damping[0], damping[1])
                mask = A[j] != 0
                w[mask] *= ratio

        # Apply bounds after each full iteration
        adj = w / original_weights
        adj = np.clip(adj, bounds[0], bounds[1])
        w = original_weights * adj

    # Compute L2 loss (squared relative error)
    achieved = A @ w
    l2_loss = np.mean(((achieved - targets) / targets) ** 2)

    # Check convergence (all targets within 5%)
    max_error = np.max(np.abs((achieved - targets) / targets))
    success = max_error < 0.05

    if verbose:
        print(f"IPF converged: max error = {max_error:.1%}, L2 loss = {l2_loss:.6f}")

    return w, success, l2_loss


def generalized_rake_calibrate(
    original_weights: np.ndarray,
    constraints: List[Dict],
    bounds: tuple[float, float] = (0.1, 20.0),
    max_iter: int = 80,
    target_tolerance: float = 0.05,
    verbose: bool = True,
) -> tuple[np.ndarray, bool, float]:
    """
    Calibrate weights with bounded generalized raking.

    This solves the entropy/raking problem in the dual, where the number of
    variables is the number of calibration constraints rather than the number
    of microdata rows. It supports overlapping count and amount constraints
    such as SOI filer counts and AGI totals by bracket.

    Args:
        original_weights: Initial survey weights.
        constraints: Constraint dicts with ``indicator`` and ``target_value``.
        bounds: Min/max adjustment factors around ``original_weights``.
        max_iter: Maximum Newton iterations.
        target_tolerance: Success threshold for max relative target error.
        verbose: Print progress.

    Returns:
        (calibrated_weights, success, l2_loss)
    """
    lower, upper = bounds
    if not 0 < lower < 1 < upper:
        raise ValueError("generalized raking bounds must satisfy 0 < lower < 1 < upper")

    n = len(original_weights)
    m = len(constraints)
    if verbose:
        print(
            "Generalized raking calibration: "
            f"{n:,} weights, {m} constraints, bounds=[{lower}, {upper}]"
        )

    A = np.vstack([np.asarray(c["indicator"], dtype=float) for c in constraints])
    targets = np.asarray([c["target_value"] for c in constraints], dtype=float)
    if np.any(~np.isfinite(A)) or np.any(~np.isfinite(targets)):
        raise ValueError("Calibration constraints contain non-finite values")
    if np.any(targets == 0):
        raise ValueError("Generalized raking requires nonzero target values")

    current = A @ original_weights
    if np.any(current == 0):
        zero_constraints = [
            _constraint_key(c) for c, value in zip(constraints, current) if value == 0
        ]
        raise ValueError(
            "Cannot calibrate constraints with zero current support: "
            + ", ".join(zero_constraints)
        )

    # Row scaling keeps count and amount constraints on comparable numerical
    # footing without changing the equations being solved.
    scale = np.abs(current)
    B = A / scale[:, None]
    scaled_targets = targets / scale

    offset = np.log((1 - lower) / (upper - 1))
    dual = np.zeros(m)
    best_weights = original_weights.copy()
    best_error = np.inf

    for iteration in range(max_iter):
        adjustment, derivative = _bounded_rake_adjustment(
            B.T @ dual,
            offset=offset,
            lower=lower,
            upper=upper,
        )
        weights = original_weights * adjustment
        residual = B @ weights - scaled_targets
        relative_errors = (A @ weights - targets) / targets
        max_error = float(np.max(np.abs(relative_errors)))
        if max_error < best_error:
            best_error = max_error
            best_weights = weights.copy()

        if verbose and (iteration == 0 or (iteration + 1) % 10 == 0):
            print(
                f"  iter {iteration + 1}: max error={max_error:.2%}, "
                f"adjustment range=[{adjustment.min():.2f}, {adjustment.max():.2f}]"
            )

        jacobian = (B * (original_weights * derivative)) @ B.T
        ridge = max(float(np.trace(jacobian)) / max(m, 1) * 1e-9, 1e-12)
        step = np.linalg.lstsq(
            jacobian + np.eye(m) * ridge,
            residual,
            rcond=None,
        )[0]

        residual_norm = float(np.linalg.norm(residual))
        step_accepted = False
        small_improvement = False
        step_size = 1.0
        for _ in range(30):
            candidate_dual = dual - step_size * step
            candidate_adjustment, _ = _bounded_rake_adjustment(
                B.T @ candidate_dual,
                offset=offset,
                lower=lower,
                upper=upper,
            )
            candidate_residual = (
                B @ (original_weights * candidate_adjustment) - scaled_targets
            )
            candidate_norm = float(np.linalg.norm(candidate_residual))
            if (
                np.all(np.isfinite(candidate_residual))
                and candidate_norm < residual_norm
            ):
                dual = candidate_dual
                step_accepted = True
                small_improvement = (residual_norm - candidate_norm) <= max(
                    1e-12,
                    residual_norm * 1e-8,
                )
                break
            step_size *= 0.5

        if not step_accepted or small_improvement:
            break

    relative_errors = (A @ best_weights - targets) / targets
    l2_loss = float(np.mean(relative_errors**2))
    max_error = float(np.max(np.abs(relative_errors)))
    success = max_error < target_tolerance

    if verbose:
        print(
            f"Generalized raking complete: max error = {max_error:.1%}, "
            f"L2 loss = {l2_loss:.6f}"
        )

    return best_weights, success, l2_loss


def _bounded_rake_adjustment(
    linear_predictor: np.ndarray,
    *,
    offset: float,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bounded calibration factors and derivatives."""
    z = np.clip(offset + linear_predictor, -60, 60)
    probability = 1 / (1 + np.exp(-z))
    adjustment = lower + (upper - lower) * probability
    derivative = (adjustment - lower) * (upper - adjustment) / (upper - lower)
    return adjustment, derivative


def calibrate_weights(
    df: pd.DataFrame,
    targets: List[Dict[str, Any]] | List[TargetSpec],
    include_amounts: bool = True,
    min_obs: int = 100,
    calibration_method: str = "auto",
    weight_bounds: tuple[float, float] = (0.1, 20.0),
    holdout_variables: tuple[str, ...] = (),
    verbose: bool = True,
) -> CalibrationResult:
    """Calibrate Microplex weights to Arch target inputs."""
    original_weights = df["weight"].values.copy()

    if verbose:
        print(f"\nCalibrating {len(df):,} tax units...")
        print(f"Original weighted total: {original_weights.sum():,.0f}")

    constraints, diagnostics, evaluation_constraints = build_constraints_from_targets(
        df,
        targets,
        min_obs=min_obs,
        include_amounts=include_amounts,
        holdout_variables=holdout_variables,
        return_diagnostics=True,
    )
    diagnostics_df = pd.DataFrame(diagnostics)
    if not constraints:
        raise ValueError("No usable calibration constraints were built.")

    # Pre-scale weights to match total population target
    total_target = None
    for c in constraints:
        if c["variable"] == "tax_unit_count" and c["n_obs"] == len(df):
            total_target = c["target_value"]
            break

    if total_target:
        scale_factor = total_target / original_weights.sum()
        original_weights = original_weights * scale_factor
        if verbose:
            print(f"Pre-scaled weights by {scale_factor:.3f} to match total target {total_target:,.0f}")

    # Pre-calibration values
    targets_before = {}
    for c in constraints:
        current = np.dot(original_weights, c["indicator"])
        error = (
            (current - c["target_value"]) / c["target_value"]
            if c["target_value"] != 0
            else 0
        )
        diagnostic_index = c.get("diagnostic_index")
        if diagnostic_index is not None:
            diagnostics_df.loc[diagnostic_index, "pre_value"] = current
            diagnostics_df.loc[diagnostic_index, "pre_error"] = error

        key = _constraint_key(c)
        targets_before[key] = {
            "current": current,
            "target": c["target_value"],
            "error": error,
        }
    for c in evaluation_constraints:
        _update_diagnostic_values(
            diagnostics_df,
            c,
            weights=original_weights,
            value_column="pre_value",
            error_column="pre_error",
        )

    has_amount_constraints = any(c.get("target_type") == "amount" for c in constraints)
    method = calibration_method
    if method == "auto":
        method = "generalized-rake" if has_amount_constraints else "ipf"

    if method == "ipf":
        calibrated_weights, success, l2_loss = ipf_calibrate(
            original_weights,
            constraints,
            verbose=verbose,
        )
    elif method == "generalized-rake":
        calibrated_weights, success, l2_loss = generalized_rake_calibrate(
            original_weights,
            constraints,
            bounds=weight_bounds,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown calibration method: {calibration_method}")

    # Post-calibration values
    targets_after = {}
    max_error = 0
    for c in constraints:
        current = np.dot(calibrated_weights, c["indicator"])
        error = (current - c["target_value"]) / c["target_value"] if c["target_value"] != 0 else 0
        diagnostic_index = c.get("diagnostic_index")
        if diagnostic_index is not None:
            diagnostics_df.loc[diagnostic_index, "post_value"] = current
            diagnostics_df.loc[diagnostic_index, "post_error"] = error

        key = _constraint_key(c)
        targets_after[key] = {
            "current": current,
            "target": c["target_value"],
            "error": error,
        }
        max_error = max(max_error, abs(error))
    for c in evaluation_constraints:
        _update_diagnostic_values(
            diagnostics_df,
            c,
            weights=calibrated_weights,
            value_column="post_value",
            error_column="post_error",
        )

    adjustment_factors = calibrated_weights / original_weights

    if verbose:
        print(f"\nPost-calibration max error: {max_error:.1%}")
        print(f"Weight adjustments: mean={adjustment_factors.mean():.2f}, "
              f"range=[{adjustment_factors.min():.2f}, {adjustment_factors.max():.2f}]")
        print(f"L2 loss: {l2_loss:.6f}")

    return CalibrationResult(
        original_weights=original_weights,
        calibrated_weights=calibrated_weights,
        adjustment_factors=adjustment_factors,
        targets_before=targets_before,
        targets_after=targets_after,
        diagnostics=diagnostics_df,
        success=success,
        message="Converged" if success else "Did not converge",
        l2_loss=l2_loss,
        method=method,
    )


def _update_diagnostic_values(
    diagnostics_df: pd.DataFrame,
    constraint: dict[str, Any],
    *,
    weights: np.ndarray,
    value_column: str,
    error_column: str,
) -> None:
    diagnostic_index = constraint.get("diagnostic_index")
    if diagnostic_index is None:
        return
    current = float(np.dot(weights, constraint["indicator"]))
    target = constraint["target_value"]
    error = (current - target) / target if target != 0 else 0
    diagnostics_df.loc[diagnostic_index, value_column] = current
    diagnostics_df.loc[diagnostic_index, error_column] = error


def _constraint_key(constraint: dict[str, Any]) -> str:
    return (
        f"{constraint.get('variable')}|{constraint.get('target_type')}|"
        f"{constraint.get('stratum')}"
    )


def write_microplex_to_supabase(
    df: pd.DataFrame,
    year: int,
    chunk_size: int = 200,
) -> int:
    """Write calibrated microplex to Supabase."""
    client = get_supabase_client()
    table_name = f"us_microplex_{year}_person"

    print(f"Writing {len(df):,} records to {table_name}...")

    records = []
    for _, row in df.iterrows():
        record = {
            "source_person_id": int(row.get("id", 0)) if pd.notna(row.get("id")) else None,
            "household_id": int(row.get("ph_seq", 0)) if pd.notna(row.get("ph_seq")) else None,
            "age": int(row.get("a_age", 0)) if pd.notna(row.get("a_age")) else None,
            "state_fips": int(row.get("gestfips", 0)) if pd.notna(row.get("gestfips")) else None,
            "wage_income": float(row.get("wage_income", 0)),
            "self_employment_income": float(row.get("self_employment_income", 0)),
            "total_income": float(row.get("total_income", 0)),
            "adjusted_gross_income": float(row.get("adjusted_gross_income", 0)) if "adjusted_gross_income" in row else None,
            "original_weight": float(row.get("original_weight", 0)),
            "calibrated_weight": float(row.get("weight", 0)),
            "weight_adjustment": float(row.get("weight_adjustment", 1.0)),
            "agi_bracket": row.get("agi_bracket"),
        }
        records.append(record)

    total = 0
    for i in range(0, len(records), chunk_size):
        chunk = records[i : i + chunk_size]
        client.schema("microplex").table(table_name).insert(chunk).execute()
        total += len(chunk)
        if (i + chunk_size) % 5000 == 0:
            print(f"  Inserted {total:,} / {len(records):,}")

    print(f"  Done: {total:,} records")
    return total


def print_target_composition_diagnostics(
    composition: TargetCompositionResult,
) -> None:
    """Print how Arch target inputs became Microplex target candidates."""
    diagnostics = composition.diagnostics
    print("\nTarget input composition:")
    print(f"  Candidate targets: {len(composition.targets):,}")
    if composition.fallback_reason is not None:
        print(
            "  Fallback: "
            f"{composition.fallback_reason}; using {composition.fallback_year}"
        )
    if composition.soi_aging_factors is not None:
        factors = composition.soi_aging_factors
        print(
            "  Aged SOI targets "
            f"{factors.source_year}->{factors.target_year}: "
            f"counts x{factors.count_factor:.4f} "
            f"({factors.count_method}), "
            f"amounts x{factors.amount_factor:.4f} "
            f"({factors.amount_method})"
        )
    if diagnostics.empty:
        return
    action_counts = diagnostics["action"].value_counts()
    action_text = ", ".join(
        f"{action}={count}" for action, count in action_counts.items()
    )
    print(f"  Composition actions: {action_text}")


def run_pipeline(
    year: int = 2024,
    dry_run: bool = False,
    limit: int = 200000,
    target_source: str = "db",
    db_path: Path | None = None,
    microdata_source: str = "local",
    cps_path: Path | None = None,
    output_path: Path | None = None,
    age_soi: bool = True,
    include_amount_targets: bool = True,
    min_target_obs: int = 100,
    calibration_method: str = "auto",
    min_weight_factor: float = 0.1,
    max_weight_factor: float = 20.0,
    diagnostics_path: Path | None = None,
) -> pd.DataFrame:
    """Run the full microplex pipeline."""
    print("=" * 60)
    print("MICROPLEX PIPELINE")
    print("=" * 60)

    # Load data
    if microdata_source == "local":
        df = load_cps_from_local_file(year, path=cps_path, limit=limit)
    elif microdata_source == "supabase":
        df = load_cps_from_supabase(year, limit=limit)
    else:
        raise ValueError(f"Unknown microdata_source: {microdata_source}")

    if target_source == "db":
        target_profile = MicroplexTargetProfile(age_soi=age_soi)
        target_composition = compose_microplex_targets(
            target_year=year,
            db_path=db_path,
            profile=target_profile,
        )
        targets = target_composition.targets
        print_target_composition_diagnostics(target_composition)
    elif target_source == "supabase":
        targets = load_targets_from_supabase(year)
    else:
        raise ValueError(f"Unknown target_source: {target_source}")

    if target_source == "supabase" and len(targets) < 50:
        # Fall back to the latest available SOI targets when the model year
        # has insufficient usable tax targets.
        fallback_year = 2021
        fallback_reason = f"only {len(targets)} target inputs"
        print(f"  {year} has {fallback_reason}, trying {fallback_year}...")
        targets = load_targets_from_supabase(fallback_year)

    # Build tax units
    df = build_tax_units(df)

    # Calibrate
    result = calibrate_weights(
        df,
        targets,
        include_amounts=include_amount_targets,
        min_obs=min_target_obs,
        calibration_method=calibration_method,
        weight_bounds=(min_weight_factor, max_weight_factor),
        holdout_variables=target_profile.holdout_variables
        if target_source == "db"
        else (),
    )

    # Add calibrated weights to dataframe
    df["original_weight"] = result.original_weights
    df["weight"] = result.calibrated_weights
    df["weight_adjustment"] = result.adjustment_factors

    # Add AGI bracket for analysis
    df["agi_bracket"] = assign_agi_bracket(df["adjusted_gross_income"].values)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tax units: {len(df):,}")
    print(f"Original weighted: {result.original_weights.sum():,.0f}")
    print(f"Calibrated weighted: {result.calibrated_weights.sum():,.0f}")
    print(f"Calibration method: {result.method}")
    print(f"Calibration success: {result.success}")
    print_calibration_diagnostics(result.diagnostics)

    if diagnostics_path is not None:
        write_calibration_diagnostics(result.diagnostics, diagnostics_path)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"\nSaved microplex to {output_path}")
    elif not dry_run:
        write_microplex_to_supabase(df, year)
    else:
        print("\nDRY RUN - not writing to Supabase")

    return df


def print_calibration_diagnostics(diagnostics: pd.DataFrame, max_rows: int = 8) -> None:
    """Print a compact diagnostic summary for target usage and failures."""
    if diagnostics.empty:
        print("No calibration diagnostics available")
        return

    used = diagnostics[diagnostics["status"] == "used"].copy()
    dropped = diagnostics[diagnostics["status"] == "dropped"].copy()
    print("\nCalibration target diagnostics:")
    print(f"  Used targets: {len(used):,}")
    print(f"  Dropped targets: {len(dropped):,}")
    if "role" in diagnostics:
        roles = diagnostics["role"].fillna("unknown").value_counts()
        role_text = ", ".join(f"{role}={count}" for role, count in roles.items())
        print(f"  Target roles: {role_text}")
    if not dropped.empty:
        reasons = dropped["drop_reason"].value_counts()
        reason_text = ", ".join(f"{reason}={count}" for reason, count in reasons.items())
        print(f"  Drop reasons: {reason_text}")

    if used.empty or "post_error" not in used:
        return

    used["abs_post_error"] = used["post_error"].abs()
    top = used.sort_values("abs_post_error", ascending=False).head(max_rows)
    print("  Largest used-target errors:")
    for _, row in top.iterrows():
        print(
            "    "
            f"{row['variable']} {row['target_type']} | "
            f"{row['stratum']} | "
            f"target={row['target_value']:,.0f} "
            f"post={row['post_value']:,.0f} "
            f"error={row['post_error']:.1%} "
            f"n={int(row['n_obs']):,}"
        )


def write_calibration_diagnostics(diagnostics: pd.DataFrame, path: Path) -> None:
    """Write calibration diagnostics to CSV or Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        diagnostics.to_parquet(path, index=False)
    else:
        diagnostics.to_csv(path, index=False)
    print(f"Saved calibration diagnostics to {path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run microplex pipeline")
    parser.add_argument("--year", type=int, default=2024, help="Data year")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to Supabase")
    parser.add_argument("--limit", type=int, default=200000, help="Max records to load")
    parser.add_argument(
        "--microdata-source",
        choices=["local", "supabase"],
        default="local",
        help="CPS microdata source",
    )
    parser.add_argument(
        "--target-source",
        choices=["db", "supabase"],
        default="db",
        help="Calibration target source",
    )
    parser.add_argument("--db-path", type=Path, default=None, help="Arch SQLite DB path")
    parser.add_argument("--cps-path", type=Path, default=None, help="Local CPS parquet path")
    parser.add_argument("--output-path", type=Path, default=None, help="Local parquet output")
    parser.add_argument(
        "--age-soi-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Age fallback SOI target inputs to the requested model year",
    )
    parser.add_argument(
        "--include-amount-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include amount targets such as AGI totals in calibration",
    )
    parser.add_argument(
        "--min-target-obs",
        type=int,
        default=100,
        help="Minimum nonzero records required for a target constraint",
    )
    parser.add_argument(
        "--diagnostics-path",
        type=Path,
        default=None,
        help="Optional CSV or Parquet path for calibration target diagnostics",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["auto", "ipf", "generalized-rake"],
        default="auto",
        help="Calibration solver to use",
    )
    parser.add_argument(
        "--min-weight-factor",
        type=float,
        default=0.1,
        help="Minimum generalized-rake weight adjustment factor",
    )
    parser.add_argument(
        "--max-weight-factor",
        type=float,
        default=20.0,
        help="Maximum generalized-rake weight adjustment factor",
    )
    args = parser.parse_args()

    run_pipeline(
        year=args.year,
        dry_run=args.dry_run,
        limit=args.limit,
        target_source=args.target_source,
        db_path=args.db_path,
        microdata_source=args.microdata_source,
        cps_path=args.cps_path,
        output_path=args.output_path,
        age_soi=args.age_soi_targets,
        include_amount_targets=args.include_amount_targets,
        min_target_obs=args.min_target_obs,
        calibration_method=args.calibration_method,
        min_weight_factor=args.min_weight_factor,
        max_weight_factor=args.max_weight_factor,
        diagnostics_path=args.diagnostics_path,
    )


if __name__ == "__main__":
    main()
