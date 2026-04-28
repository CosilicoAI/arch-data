"""
Arch-to-Microplex Pipeline: Build calibrated microdata from Arch sources.

Reads CPS microdata and Arch target inputs, runs IPF calibration, and writes
calibrated microplex output locally or back to Supabase.

Calibration methods compared (L2 loss, runtime):
- IPF (100 iter): L2=0.040, 0.6s   <-- RECOMMENDED
- IPF+GREG (20):  L2=0.038, 301s   (486x slower for 7% better)
- Entropy:        L2=9.0, 0.04s    (doesn't converge)
- GD L2:          L2=0.77          (doesn't converge)

Usage:
    python -m arch.pipelines.microplex --year 2024
    python -m arch.pipelines.microplex --year 2024 --dry-run
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from arch.client import get_supabase_client
from arch.microdata import query_cps_asec
from arch.targets import DataSource, TargetSpec, TargetType, query_targets
from arch.targets.microplex import (
    age_soi_targets,
    build_microplex_constraints,
    constraints_to_ipf_dicts,
    get_soi_aging_factors,
    load_microplex_targets,
)


@dataclass
class CalibrationResult:
    """Results from IPF calibration."""
    original_weights: np.ndarray
    calibrated_weights: np.ndarray
    adjustment_factors: np.ndarray
    targets_before: Dict[str, Dict]
    targets_after: Dict[str, Dict]
    success: bool
    message: str
    l2_loss: float


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


def has_supported_tax_targets(
    targets: List[Dict[str, Any]] | List[TargetSpec],
) -> bool:
    """Return whether target inputs can produce current tax-unit constraints."""
    supported_variables = {"tax_unit_count", "adjusted_gross_income"}
    for target in targets:
        if isinstance(target, TargetSpec):
            if target.variable in supported_variables and target.target_type != TargetType.RATE:
                return True
        elif target.get("variable") in supported_variables and target.get("target_type") != "rate":
            return True
    return False


def latest_supported_soi_year(
    target_year: int,
    db_path: Path | None = None,
    jurisdiction: str = "us",
) -> int | None:
    """Find the latest SOI year at or before the model year with usable targets."""
    for candidate_year in range(target_year, 1989, -1):
        targets = load_microplex_targets(
            db_path=db_path,
            jurisdiction=jurisdiction,
            year=candidate_year,
            sources=[DataSource.IRS_SOI.value],
        )
        if has_supported_tax_targets(targets):
            return candidate_year
    return None


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
        ("1m_plus", 1000000, np.inf),
    ]

    result = np.empty(len(agi), dtype=object)
    for name, low, high in brackets:
        mask = (agi >= low) & (agi < high)
        result[mask] = name

    return result


def build_constraints_from_target_specs(
    df: pd.DataFrame,
    targets: List[TargetSpec],
    min_obs: int = 100,
    include_amounts: bool = False,
) -> List[Dict]:
    """
    Build legacy IPF constraint dicts from Arch DB ``TargetSpec`` objects.

    This keeps the old IPF pipeline working while moving the target source from
    Supabase-shaped dictionaries to the local Arch target database.
    """
    df = df.copy()
    df["agi_bracket"] = assign_agi_bracket(df["adjusted_gross_income"].values)

    supported_variables = {"tax_unit_count", "adjusted_gross_income"}
    supported_constraint_variables = {
        "adjusted_gross_income",
        "is_tax_filer",
        "agi_bracket",
    }

    filtered_targets = []
    seen_keys = set()
    for target in targets:
        if target.variable not in supported_variables:
            continue
        if target.target_type == TargetType.RATE:
            continue
        if target.target_type == TargetType.AMOUNT and not include_amounts:
            continue
        if any(
            variable not in supported_constraint_variables
            for variable, _, _ in target.constraints
        ):
            continue

        key = (target.stratum_name, target.variable, target.target_type)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        filtered_targets.append(target)

    constraints = build_microplex_constraints(
        df,
        targets=filtered_targets,
        min_obs=min_obs,
    )
    constraint_dicts = constraints_to_ipf_dicts(constraints)
    print(f"  Built {len(constraint_dicts)} constraints (min {min_obs} obs each)")
    return constraint_dicts


def build_constraints_from_targets(
    df: pd.DataFrame,
    targets: List[Dict[str, Any]] | List[TargetSpec],
    min_obs: int = 100,
    include_amounts: bool = False,
) -> List[Dict]:
    """
    Build calibration constraints from Supabase targets.

    Currently supports:
    - Tax unit counts by AGI bracket (adjusted_gross_income ranges)
    - AGI totals by bracket (if include_amounts=True)

    Skips unsupported targets:
    - Filing status (need CPS marital status mapping)
    - Program participation (SNAP, SSI, OASDI - need program vars)
    - Population counts (different universe than tax filers)
    """
    if targets and isinstance(targets[0], TargetSpec):
        return build_constraints_from_target_specs(
            df,
            targets,
            min_obs=min_obs,
            include_amounts=include_amounts,
        )

    constraints = []
    seen_keys = set()
    n = len(df)
    df = df.copy()

    # Precompute AGI brackets
    df["agi_bracket"] = assign_agi_bracket(df["adjusted_gross_income"].values)

    # Supported variables for tax filer calibration
    SUPPORTED_VARIABLES = {"tax_unit_count", "adjusted_gross_income"}

    for target in targets:
        variable = target["variable"]
        value = target["value"]
        target_type = target.get("target_type")

        # Only calibrate on supported variables
        if variable not in SUPPORTED_VARIABLES:
            continue

        # Skip rate targets; optionally skip amount targets
        if target_type == "rate":
            continue
        if target_type == "amount" and not include_amounts:
            continue

        stratum = target.get("strata", {})
        stratum_name = stratum.get("name", "unknown")
        stratum_constraints = stratum.get("stratum_constraints", [])

        # Skip strata with unsupported constraint types
        # (filing_status, snap, ssi, oasdi, etc.)
        has_unsupported = False
        for c in stratum_constraints:
            var = c.get("variable")
            if var not in {"adjusted_gross_income", "is_tax_filer", "agi_bracket"}:
                has_unsupported = True
                break
        if has_unsupported:
            continue

        # Deduplicate
        key = (stratum_name, variable, target_type)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        # Build indicator from stratum constraints
        indicator = np.ones(n, dtype=bool)

        for constraint in stratum_constraints:
            var = constraint.get("variable")
            op = constraint.get("operator", "==")
            val = constraint.get("value")

            if var == "adjusted_gross_income":
                col = df["adjusted_gross_income"]
                val = float(val)
            elif var == "agi_bracket":
                col = df["agi_bracket"]
            elif var == "is_tax_filer":
                # All records in our dataset are filers
                col = pd.Series([1] * n)
                val = int(val)
            else:
                continue

            # Apply operator
            if op == "==":
                indicator &= (col == val)
            elif op == ">=":
                indicator &= (col >= val)
            elif op == ">":
                indicator &= (col > val)
            elif op == "<=":
                indicator &= (col <= val)
            elif op == "<":
                indicator &= (col < val)
            elif op == "!=":
                indicator &= (col != val)

        indicator = indicator.astype(float)

        # For amount targets, multiply by AGI
        if target_type == "amount" and variable == "adjusted_gross_income":
            indicator = indicator * df["adjusted_gross_income"].values

        n_obs = (indicator > 0).sum()
        if n_obs >= min_obs:
            constraints.append({
                "indicator": indicator,
                "target_value": value,
                "variable": variable,
                "target_type": target_type,
                "stratum": stratum_name,
                "n_obs": n_obs,
            })

    print(f"  Built {len(constraints)} constraints (min {min_obs} obs each)")
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


def calibrate_weights(
    df: pd.DataFrame,
    targets: List[Dict[str, Any]] | List[TargetSpec],
    verbose: bool = True,
) -> CalibrationResult:
    """Calibrate weights using IPF (Iterative Proportional Fitting)."""
    original_weights = df["weight"].values.copy()

    if verbose:
        print(f"\nCalibrating {len(df):,} tax units...")
        print(f"Original weighted total: {original_weights.sum():,.0f}")

    constraints = build_constraints_from_targets(df, targets)

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
        targets_before[c["variable"]] = {
            "current": current,
            "target": c["target_value"],
            "error": (current - c["target_value"]) / c["target_value"] if c["target_value"] != 0 else 0,
        }

    # Run IPF calibration
    calibrated_weights, success, l2_loss = ipf_calibrate(
        original_weights, constraints, verbose=verbose
    )

    # Post-calibration values
    targets_after = {}
    max_error = 0
    for c in constraints:
        current = np.dot(calibrated_weights, c["indicator"])
        error = (current - c["target_value"]) / c["target_value"] if c["target_value"] != 0 else 0
        targets_after[c["variable"]] = {
            "current": current,
            "target": c["target_value"],
            "error": error,
        }
        max_error = max(max_error, abs(error))

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
        success=success,
        message="Converged" if success else "Did not converge",
        l2_loss=l2_loss,
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
        targets = load_targets_from_db(year, db_path=db_path)
    elif target_source == "supabase":
        targets = load_targets_from_supabase(year)
    else:
        raise ValueError(f"Unknown target_source: {target_source}")

    if len(targets) < 50 or not has_supported_tax_targets(targets):
        # Fall back to the latest available SOI targets when the model year
        # has insufficient usable tax targets.
        if target_source == "db":
            fallback_year = latest_supported_soi_year(year, db_path=db_path) or 2021
        else:
            fallback_year = 2021
        if len(targets) < 50:
            fallback_reason = f"only {len(targets)} target inputs"
        else:
            fallback_reason = "no supported current-year tax targets"
        print(f"  {year} has {fallback_reason}, trying {fallback_year}...")
        if target_source == "db":
            current_targets = targets
            targets = load_targets_from_db(
                fallback_year,
                db_path=db_path,
                sources=[DataSource.IRS_SOI.value],
            )
            needs_soi_aging = any(
                target.source == DataSource.IRS_SOI and target.period != year
                for target in targets
            )
            if age_soi and needs_soi_aging:
                factors = get_soi_aging_factors(
                    source_year=fallback_year,
                    target_year=year,
                    db_path=db_path,
                )
                targets = age_soi_targets(
                    targets,
                    target_year=year,
                    db_path=db_path,
                    factors=factors,
                )
                print(
                    "  Aged SOI targets "
                    f"{fallback_year}->{year}: "
                    f"counts x{factors.count_factor:.4f} "
                    f"({factors.count_method}), "
                    f"amounts x{factors.amount_factor:.4f} "
                    f"({factors.amount_method})"
                )
            targets = [
                target
                for target in current_targets
                if target.source != DataSource.IRS_SOI
            ] + targets
        else:
            targets = load_targets_from_supabase(fallback_year)

    # Build tax units
    df = build_tax_units(df)

    # Calibrate
    result = calibrate_weights(df, targets)

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
    print(f"Calibration success: {result.success}")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"\nSaved microplex to {output_path}")
    elif not dry_run:
        write_microplex_to_supabase(df, year)
    else:
        print("\nDRY RUN - not writing to Supabase")

    return df


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
    )


if __name__ == "__main__":
    main()
