"""
Hierarchical Microplex Pipeline: Calibrate household weights with person-level targets.

Reads hierarchical microdata (households + persons) and calibration targets,
aggregates person-level targets to household level, runs IPF calibration on
household weights, and writes results back to Supabase.

Key difference from flat pipeline:
- Weights are at household level (one weight per household)
- Targets can be at person level (e.g., count of children aged 0-17)
- Person-level targets are aggregated to household level before calibration

Usage:
    python -m arch.pipelines.hierarchical_microplex --year 2024
    python -m arch.pipelines.hierarchical_microplex --year 2024 --dry-run
    python -m arch.pipelines.hierarchical_microplex --year 2024 --use-mock
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# Lazy import of Supabase to avoid import errors when not available
_supabase_client = None

def _get_supabase_client():
    """Lazy import of Supabase client."""
    global _supabase_client
    if _supabase_client is None:
        try:
            from db.supabase_client import get_supabase_client
            _supabase_client = get_supabase_client()
        except ImportError:
            raise ImportError(
                "Supabase client not available. "
                "Install with: pip install supabase"
            )
    return _supabase_client


@dataclass
class HierarchicalCalibrationResult:
    """Results from hierarchical IPF calibration."""
    n_households: int
    n_persons: int
    original_weights: np.ndarray
    calibrated_weights: np.ndarray
    adjustment_factors: np.ndarray
    n_constraints: int
    success: bool
    message: str
    l2_loss: float
    max_error: float


def load_hierarchical_data_from_supabase(
    year: int = 2024,
    limit: int = 100000,
    use_mock: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load hierarchical microdata (households + persons) from Supabase.

    Args:
        year: Data year
        limit: Maximum households to load
        use_mock: Use mock data instead of Supabase

    Returns:
        Tuple of (household_df, person_df)
    """
    if use_mock:
        return _create_mock_hierarchical_data(limit)

    print(f"Loading hierarchical data for {year} from Supabase...")

    # Try microplex.households table first
    client = _get_supabase_client()

    try:
        # Query households
        print("  Loading households...")
        hh_result = client.schema("microplex").table("households").select("*").limit(limit).execute()
        hh_df = pd.DataFrame(hh_result.data)
        print(f"    Loaded {len(hh_df):,} households")

        if len(hh_df) == 0:
            raise ValueError("No household data found")

        # Query persons
        print("  Loading persons...")
        # Get household IDs to filter persons
        hh_ids = hh_df["household_id"].tolist()

        # Paginate person queries since there are many persons per household
        all_persons = []
        page_size = 1000
        offset = 0

        while True:
            person_result = (
                client.schema("microplex")
                .table("persons")
                .select("*")
                .in_("household_id", hh_ids)
                .range(offset, offset + page_size - 1)
                .execute()
            )
            if not person_result.data:
                break
            all_persons.extend(person_result.data)
            offset += len(person_result.data)
            if len(person_result.data) < page_size:
                break

        person_df = pd.DataFrame(all_persons)
        print(f"    Loaded {len(person_df):,} persons")

        return hh_df, person_df

    except Exception as e:
        print(f"  Supabase query failed: {e}")
        print("  Falling back to mock data")
        return _create_mock_hierarchical_data(limit)


def _create_mock_hierarchical_data(limit: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create mock hierarchical data for testing."""
    np.random.seed(42)

    n_households = min(limit, 1000)

    # Generate households
    hh_df = pd.DataFrame({
        "household_id": range(1, n_households + 1),
        "weight": np.random.uniform(50, 500, n_households),
        "state_fips": np.random.choice([6, 36, 48, 12, 17], n_households),  # CA, NY, TX, FL, IL
        "tenure": np.random.choice(["own", "rent"], n_households, p=[0.6, 0.4]),
    })

    # Generate persons (1-4 per household)
    persons = []
    person_id = 1
    for hh_id in hh_df["household_id"]:
        n_persons = np.random.randint(1, 5)
        for i in range(n_persons):
            age = np.random.choice(
                [5, 10, 15, 25, 35, 45, 55, 65, 75],
                p=[0.08, 0.08, 0.08, 0.15, 0.18, 0.15, 0.12, 0.10, 0.06],
            )
            income = max(0, np.random.normal(40000, 30000)) if age >= 18 else 0
            persons.append({
                "person_id": person_id,
                "household_id": hh_id,
                "age": age,
                "is_male": np.random.choice([True, False]),
                "employment_income": income,
                "is_snap_recipient": np.random.random() < 0.12,
                "is_medicaid_enrolled": np.random.random() < 0.20,
            })
            person_id += 1

    person_df = pd.DataFrame(persons)

    print(f"  Created mock data: {len(hh_df):,} households, {len(person_df):,} persons")
    return hh_df, person_df


def load_targets_from_supabase(
    year: int = 2024,
    scale_factor: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Load calibration targets from Supabase targets.* schema.

    Falls back to mock targets if not available.

    Args:
        year: Target year
        scale_factor: Scale factor for mock targets (based on sample size)

    Returns:
        List of target specifications
    """
    print(f"Loading targets for {year}...")

    try:
        client = _get_supabase_client()

        # Try to query targets from targets schema
        result = client.schema("microplex").table("targets").select(
            "*, strata(*, stratum_constraints(*))"
        ).eq("period", year).execute()

        if result.data and len(result.data) > 10:
            print(f"  Loaded {len(result.data)} targets from Supabase")
            return _convert_supabase_targets(result.data)

    except Exception as e:
        print(f"  Supabase targets query failed: {e}")

    # Fall back to mock targets (scaled to sample)
    print("  Using mock targets")
    return _create_mock_targets(scale_factor=scale_factor)


def _convert_supabase_targets(raw_targets: List[Dict]) -> List[Dict[str, Any]]:
    """Convert Supabase target format to internal format."""
    targets = []

    for t in raw_targets:
        strata = t.get("strata", {})
        constraints = []

        for c in strata.get("stratum_constraints", []):
            constraints.append((
                c.get("variable"),
                c.get("operator", "=="),
                c.get("value"),
            ))

        targets.append({
            "variable": t.get("variable"),
            "value": float(t.get("value", 0)),
            "target_type": t.get("target_type", "count"),
            "constraints": constraints,
            "stratum_name": strata.get("name"),
            "source": t.get("source"),
        })

    return targets


def _create_mock_targets(scale_factor: float = 1.0) -> List[Dict[str, Any]]:
    """Create mock targets for testing.

    Args:
        scale_factor: Multiplier for target values (use < 1 for smaller samples)

    Returns:
        List of mock target specifications
    """
    # Base targets (approximately proportional to US population)
    base_targets = [
        # Total population
        {
            "variable": "person_count",
            "value": 330000000 * scale_factor,
            "target_type": "count",
            "constraints": [],
            "stratum_name": "Total US Population",
        },
        # Children under 18 (~22%)
        {
            "variable": "person_count",
            "value": 73000000 * scale_factor,
            "target_type": "count",
            "constraints": [("age", ">=", "0"), ("age", "<", "18")],
            "stratum_name": "Children under 18",
        },
        # Adults 18-64 (~61%)
        {
            "variable": "person_count",
            "value": 200000000 * scale_factor,
            "target_type": "count",
            "constraints": [("age", ">=", "18"), ("age", "<", "65")],
            "stratum_name": "Working age adults",
        },
        # Seniors 65+ (~17%)
        {
            "variable": "person_count",
            "value": 57000000 * scale_factor,
            "target_type": "count",
            "constraints": [("age", ">=", "65")],
            "stratum_name": "Seniors 65+",
        },
        # SNAP recipients (~13%)
        {
            "variable": "person_count",
            "value": 42000000 * scale_factor,
            "target_type": "count",
            "constraints": [("is_snap_recipient", "==", "True")],
            "stratum_name": "SNAP recipients",
        },
        # Total household count
        {
            "variable": "household_count",
            "value": 130000000 * scale_factor,
            "target_type": "count",
            "constraints": [],
            "stratum_name": "Total households",
        },
    ]
    return base_targets


def build_hierarchical_constraints(
    hh_df: pd.DataFrame,
    person_df: pd.DataFrame,
    targets: List[Dict[str, Any]],
    hh_id_col: str = "household_id",
    min_obs: int = 10,
) -> List[Dict]:
    """
    Build calibration constraints aggregating person targets to household level.

    For each target, creates an indicator vector at the household level:
    - For household-level targets: indicator is 1/0 per household
    - For person-level targets: indicator is count or sum per household

    Args:
        hh_df: Household DataFrame with weight column
        person_df: Person DataFrame with household_id
        targets: List of target specifications
        hh_id_col: Column linking persons to households
        min_obs: Minimum observations for a constraint

    Returns:
        List of constraint dicts with 'indicator' and 'target_value'
    """
    constraints = []
    for target in targets:
        variable = target.get("variable", "")
        value = target.get("value", 0)
        target_type = target.get("target_type", "count")
        target_constraints = target.get("constraints", [])
        stratum_name = target.get("stratum_name", "unknown")

        # Determine if this is a household or person level target
        is_household_level = (
            "household" in variable.lower()
            or _all_constraints_are_household_level(target_constraints)
        )

        if is_household_level:
            # Direct household-level constraint
            indicator = _build_household_indicator(
                hh_df, target_constraints, variable, target_type, hh_id_col
            )
        else:
            # Aggregate from person to household
            indicator = _build_aggregated_indicator(
                hh_df, person_df, target_constraints, variable, target_type, hh_id_col
            )

        n_obs = np.sum(indicator > 0)
        if n_obs >= min_obs:
            constraints.append({
                "indicator": indicator,
                "target_value": value,
                "variable": variable,
                "target_type": target_type,
                "stratum": stratum_name,
                "n_obs": int(n_obs),
            })

    print(f"  Built {len(constraints)} constraints from {len(targets)} targets")
    return constraints


def _all_constraints_are_household_level(constraints: List[Tuple]) -> bool:
    """Check if all constraint variables are household-level.

    Returns False if empty (to default to person-level aggregation).
    """
    HOUSEHOLD_VARS = {"state_fips", "tenure", "household_size", "household_income"}
    # If no constraints, default to person level (more common case)
    if not constraints:
        return False
    for var, _, _ in constraints:
        if var not in HOUSEHOLD_VARS:
            return False
    return True


def _build_household_indicator(
    hh_df: pd.DataFrame,
    constraints: List[Tuple],
    variable: str,
    target_type: str,
    hh_id_col: str,
) -> np.ndarray:
    """Build indicator for household-level targets."""
    n = len(hh_df)
    mask = np.ones(n, dtype=bool)

    for var, op, val in constraints:
        if var not in hh_df.columns:
            continue

        col = hh_df[var]
        if pd.api.types.is_numeric_dtype(col):
            val = float(val)

        if op == "==":
            mask &= (col == val).values
        elif op == "!=":
            mask &= (col != val).values
        elif op == ">=":
            mask &= (col >= val).values
        elif op == ">":
            mask &= (col > val).values
        elif op == "<=":
            mask &= (col <= val).values
        elif op == "<":
            mask &= (col < val).values

    if target_type == "count":
        return mask.astype(float)
    elif target_type == "amount":
        # For amounts, multiply by variable value
        var_name = variable.split("#")[-1] if "#" in variable else variable
        if var_name in hh_df.columns:
            return (hh_df[var_name].values * mask).astype(float)
        return mask.astype(float)
    else:
        return mask.astype(float)


def _build_aggregated_indicator(
    hh_df: pd.DataFrame,
    person_df: pd.DataFrame,
    constraints: List[Tuple],
    variable: str,
    target_type: str,
    hh_id_col: str,
) -> np.ndarray:
    """
    Build indicator by aggregating from person to household.

    For COUNT targets: count matching persons per household
    For AMOUNT targets: sum variable value for matching persons per household
    """
    # Apply constraints to person data
    mask = pd.Series(True, index=person_df.index)

    for var, op, val in constraints:
        if var not in person_df.columns:
            continue

        col = person_df[var]

        # Handle boolean string values
        if val in ("True", "true", "1"):
            val = True
        elif val in ("False", "false", "0"):
            val = False
        elif pd.api.types.is_numeric_dtype(col):
            val = float(val)

        if op == "==":
            mask &= (col == val)
        elif op == "!=":
            mask &= (col != val)
        elif op == ">=":
            mask &= (col >= val)
        elif op == ">":
            mask &= (col > val)
        elif op == "<=":
            mask &= (col <= val)
        elif op == "<":
            mask &= (col < val)

    filtered = person_df[mask]

    if len(filtered) == 0:
        return np.zeros(len(hh_df))

    if target_type == "count":
        # Count matching persons per household
        agg = filtered.groupby(hh_id_col).size()
    elif target_type == "amount":
        # Sum variable value for matching persons per household
        var_name = variable.split("#")[-1] if "#" in variable else variable
        if var_name in filtered.columns:
            agg = filtered.groupby(hh_id_col)[var_name].sum()
        else:
            # Default to count if variable not found
            agg = filtered.groupby(hh_id_col).size()
    else:
        agg = filtered.groupby(hh_id_col).size()

    # Map back to household order (households with no matching persons get 0)
    indicator = hh_df[hh_id_col].map(agg).fillna(0).values

    return indicator.astype(float)


def run_hierarchical_ipf(
    original_weights: np.ndarray,
    constraints: List[Dict],
    bounds: Tuple[float, float] = (0.2, 5.0),
    max_iter: int = 100,
    damping: Tuple[float, float] = (0.9, 1.1),
    verbose: bool = True,
) -> Tuple[np.ndarray, bool, float]:
    """
    Run IPF calibration on household weights.

    Same algorithm as flat IPF but operates on household-level weights
    with constraints that may aggregate person-level targets.

    Args:
        original_weights: Initial household weights
        constraints: List of constraint dicts with 'indicator' and 'target_value'
        bounds: Min/max weight adjustment factors
        max_iter: Number of IPF iterations
        damping: Min/max adjustment ratio per iteration
        verbose: Print progress

    Returns:
        (calibrated_weights, success, l2_loss)
    """
    n = len(original_weights)
    m = len(constraints)

    if m == 0:
        if verbose:
            print("  No constraints, returning original weights")
        return original_weights.copy(), True, 0.0

    if verbose:
        print(f"  IPF calibration: {n:,} households, {m} constraints, {max_iter} iterations")

    # Build constraint matrix
    A = np.zeros((m, n))
    targets = np.zeros(m)

    for j, c in enumerate(constraints):
        A[j, :] = c["indicator"]
        targets[j] = c["target_value"]

    w = original_weights.copy()

    for iteration in range(max_iter):
        for j in range(m):
            achieved = np.dot(A[j], w)
            if np.isfinite(achieved) and achieved > 0 and targets[j] > 0:
                ratio = np.clip(targets[j] / achieved, damping[0], damping[1])
                mask = A[j] != 0
                w[mask] *= ratio

        # Apply bounds after each full iteration
        # Guard against division issues
        with np.errstate(divide='ignore', invalid='ignore'):
            adj = np.where(original_weights > 0, w / original_weights, 1.0)
        adj = np.clip(adj, bounds[0], bounds[1])
        w = original_weights * adj

        # Clean up any NaN/inf values
        w = np.where(np.isfinite(w), w, original_weights)

    # Compute L2 loss (squared relative error)
    achieved = A @ w
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_errors = np.where(targets != 0, (achieved - targets) / targets, 0)
    l2_loss = float(np.mean(rel_errors ** 2))

    # Check convergence (all targets within 5%)
    max_error = float(np.max(np.abs(rel_errors)))
    success = max_error < 0.05

    if verbose:
        print(f"  IPF converged: max error = {max_error:.1%}, L2 loss = {l2_loss:.6f}")

    return w, success, l2_loss


def write_households_to_supabase(
    hh_df: pd.DataFrame,
    year: int,
    chunk_size: int = 200,
) -> int:
    """Write calibrated household weights back to Supabase."""
    client = _get_supabase_client()

    print(f"Writing {len(hh_df):,} calibrated households...")

    # Update existing records with calibrated weights
    total = 0
    for i in range(0, len(hh_df), chunk_size):
        chunk = hh_df.iloc[i:i + chunk_size]

        for _, row in chunk.iterrows():
            hh_id = row["household_id"]
            calibrated_weight = float(row["calibrated_weight"])
            weight_adjustment = float(row.get("weight_adjustment", 1.0))

            try:
                client.schema("microplex").table("households").update({
                    "calibrated_weight": calibrated_weight,
                    "weight_adjustment": weight_adjustment,
                }).eq("household_id", hh_id).execute()
                total += 1
            except Exception as e:
                print(f"    Error updating household {hh_id}: {e}")

        if (i + chunk_size) % 1000 == 0:
            print(f"    Updated {total:,} / {len(hh_df):,}")

    print(f"  Done: {total:,} households updated")
    return total


def run_hierarchical_pipeline(
    year: int = 2024,
    dry_run: bool = False,
    use_mock: bool = False,
    limit: int = 100000,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full hierarchical microplex pipeline.

    Args:
        year: Data year
        dry_run: Don't write to Supabase
        use_mock: Use mock data instead of Supabase
        limit: Maximum households to process
        verbose: Print progress

    Returns:
        Calibrated household DataFrame
    """
    if verbose:
        print("=" * 60)
        print("HIERARCHICAL MICROPLEX PIPELINE")
        print("=" * 60)

    # Load hierarchical data
    hh_df, person_df = load_hierarchical_data_from_supabase(
        year=year, limit=limit, use_mock=use_mock
    )

    if verbose:
        print("\nData loaded:")
        print(f"  Households: {len(hh_df):,}")
        print(f"  Persons: {len(person_df):,}")
        print(f"  Avg persons/household: {len(person_df) / len(hh_df):.1f}")

    # Compute scale factor for mock targets (based on sample weight relative to US pop)
    # US has ~330M people and ~130M households
    sample_weight = hh_df["weight"].sum()
    scale_factor = sample_weight / 130000000.0 if use_mock else 1.0

    # Load targets
    targets = load_targets_from_supabase(year, scale_factor=scale_factor)

    if verbose:
        print(f"\nTargets loaded: {len(targets)}")

    # Build constraints
    constraints = build_hierarchical_constraints(
        hh_df, person_df, targets, hh_id_col="household_id"
    )

    if len(constraints) == 0:
        print("\nWARNING: No constraints built, skipping calibration")
        hh_df["calibrated_weight"] = hh_df["weight"]
        hh_df["weight_adjustment"] = 1.0
        return hh_df

    # Get original weights
    original_weights = hh_df["weight"].values.copy()

    if verbose:
        print(f"\nOriginal weighted households: {original_weights.sum():,.0f}")

    # Pre-scale to match total target if available
    for c in constraints:
        if c.get("variable") in ("household_count", "person_count") and c.get("n_obs") == len(hh_df):
            target_total = c["target_value"]
            scale = target_total / original_weights.sum()
            original_weights *= scale
            if verbose:
                print(f"Pre-scaled weights by {scale:.3f} to match {c['variable']} target")
            break

    # Run IPF calibration
    calibrated_weights, success, l2_loss = run_hierarchical_ipf(
        original_weights, constraints, verbose=verbose
    )

    # Compute max error
    max_error = 0.0
    for c in constraints:
        achieved = np.dot(calibrated_weights, c["indicator"])
        if c["target_value"] > 0:
            error = abs((achieved - c["target_value"]) / c["target_value"])
            max_error = max(max_error, error)

    # Add results to DataFrame
    hh_df["original_weight"] = hh_df["weight"]
    hh_df["calibrated_weight"] = calibrated_weights
    hh_df["weight_adjustment"] = calibrated_weights / hh_df["weight"].values

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Households: {len(hh_df):,}")
        print(f"Persons: {len(person_df):,}")
        print(f"Constraints: {len(constraints)}")
        print(f"Original weighted: {hh_df['original_weight'].sum():,.0f}")
        print(f"Calibrated weighted: {calibrated_weights.sum():,.0f}")
        print(f"Success: {success}")
        print(f"Max error: {max_error:.1%}")
        print(f"L2 loss: {l2_loss:.6f}")

        # Weight adjustment statistics
        adj = hh_df["weight_adjustment"]
        print("\nWeight adjustments:")
        print(f"  Mean: {adj.mean():.3f}")
        print(f"  Std: {adj.std():.3f}")
        print(f"  Min: {adj.min():.3f}")
        print(f"  Max: {adj.max():.3f}")

    # Write to Supabase
    if not dry_run:
        write_households_to_supabase(hh_df, year)
    else:
        if verbose:
            print("\nDRY RUN - not writing to Supabase")

    return hh_df


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run hierarchical microplex pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m arch.pipelines.hierarchical_microplex --year 2024 --dry-run
    python -m arch.pipelines.hierarchical_microplex --year 2024 --use-mock --dry-run
    python -m arch.pipelines.hierarchical_microplex --year 2024 --limit 10000
        """,
    )
    parser.add_argument("--year", type=int, default=2024, help="Data year (default: 2024)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to Supabase")
    parser.add_argument("--use-mock", action="store_true", help="Use mock data instead of Supabase")
    parser.add_argument("--limit", type=int, default=100000, help="Max households to load (default: 100000)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    run_hierarchical_pipeline(
        year=args.year,
        dry_run=args.dry_run,
        use_mock=args.use_mock,
        limit=args.limit,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
