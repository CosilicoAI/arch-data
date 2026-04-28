"""
Calibrate CPS tax unit weights to IRS SOI targets.

Uses entropy calibration (gradient descent on KL divergence) following
the architecture in docs/calibration-pipeline.md.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import minimize
import sys
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])  # Add parent to path

from calibration.constraints import Constraint
from arch.targets import TargetType


# IRS SOI 2021 Targets (from Statistics of Income, Table 1.1)
# Source: https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-returns-complete-report-publication-1304
# Note: Using consistent bracket data; total is computed as sum of brackets
IRS_SOI_2021_RETURNS_BY_AGI = {
    'no_agi': 13_992_100,       # Returns with no AGI (zero or not computed)
    'under_1': 1_686_440,       # $1 under $5,000 (negative AGI)
    '1_to_5k': 5_183_390,       # $1 - $5,000
    '5k_to_10k': 7_929_860,     # $5,000 - $10,000
    '10k_to_15k': 9_883_050,    # $10,000 - $15,000
    '15k_to_20k': 9_113_990,    # $15,000 - $20,000
    '20k_to_25k': 8_186_640,    # $20,000 - $25,000
    '25k_to_30k': 7_407_890,    # $25,000 - $30,000
    '30k_to_40k': 13_194_450,   # $30,000 - $40,000
    '40k_to_50k': 10_930_780,   # $40,000 - $50,000
    '50k_to_75k': 19_494_660,   # $50,000 - $75,000
    '75k_to_100k': 15_137_070,  # $75,000 - $100,000
    '100k_to_200k': 22_849_380, # $100,000 - $200,000
    '200k_to_500k': 7_167_290,  # $200,000 - $500,000
    '500k_to_1m': 1_106_040,    # $500,000 - $1,000,000
    '1m_plus': 664_340,         # $1,000,000 or more
}

# Total returns = sum of all brackets
IRS_SOI_2021_TOTAL_RETURNS = sum(IRS_SOI_2021_RETURNS_BY_AGI.values())  # ~153.9M

# AGI totals by bracket (in dollars)
IRS_SOI_2021_AGI_BY_BRACKET = {
    'no_agi': 0,
    'under_1': -94_000_000_000,      # Negative AGI returns
    '1_to_5k': 15_000_000_000,
    '5k_to_10k': 59_000_000_000,
    '10k_to_15k': 123_000_000_000,
    '15k_to_20k': 160_000_000_000,
    '20k_to_25k': 184_000_000_000,
    '25k_to_30k': 204_000_000_000,
    '30k_to_40k': 461_000_000_000,
    '40k_to_50k': 492_000_000_000,
    '50k_to_75k': 1_210_000_000_000,
    '75k_to_100k': 1_316_000_000_000,
    '100k_to_200k': 3_187_000_000_000,
    '200k_to_500k': 2_161_000_000_000,
    '500k_to_1m': 762_000_000_000,
    '1m_plus': 4_466_000_000_000,
}

IRS_SOI_2021_TOTAL_AGI = sum(IRS_SOI_2021_AGI_BY_BRACKET.values())  # ~$14.7T

AGI_BRACKETS = [
    ('no_agi', None, None),           # Special: AGI == 0 or not computed
    ('under_1', -np.inf, 1),          # Negative AGI
    ('1_to_5k', 1, 5000),
    ('5k_to_10k', 5000, 10000),
    ('10k_to_15k', 10000, 15000),
    ('15k_to_20k', 15000, 20000),
    ('20k_to_25k', 20000, 25000),
    ('25k_to_30k', 25000, 30000),
    ('30k_to_40k', 30000, 40000),
    ('40k_to_50k', 40000, 50000),
    ('50k_to_75k', 50000, 75000),
    ('75k_to_100k', 75000, 100000),
    ('100k_to_200k', 100000, 200000),
    ('200k_to_500k', 200000, 500000),
    ('500k_to_1m', 500000, 1000000),
    ('1m_plus', 1000000, np.inf),
]


@dataclass
class CalibrationResult:
    """Results from calibration."""
    original_weights: np.ndarray
    calibrated_weights: np.ndarray
    adjustment_factors: np.ndarray
    targets_before: dict
    targets_after: dict
    success: bool
    message: str
    kl_divergence: float


def assign_agi_bracket(agi: np.ndarray) -> np.ndarray:
    """Assign each record to an AGI bracket."""
    brackets = np.empty(len(agi), dtype=object)
    for name, low, high in AGI_BRACKETS:
        if name == 'no_agi':
            # Special case: AGI is exactly 0 or NaN
            mask = (agi == 0) | np.isnan(agi)
        else:
            mask = (agi >= low) & (agi < high)
        brackets[mask] = name
    return brackets


def build_constraints(df: pd.DataFrame, min_obs: int = 100) -> list[Constraint]:
    """
    Build calibration constraints from IRS SOI targets.

    Returns list of Constraint objects per architecture spec.
    Only uses bracket constraints (total is redundant since sum of brackets = total).
    """
    constraints = []

    df = df.copy()
    df['agi_bracket'] = assign_agi_bracket(df['adjusted_gross_income'].values)

    # Returns by AGI bracket (skip small strata)
    # Note: We don't add a total_returns constraint because it's redundant -
    # the sum of all bracket constraints implicitly constrains the total.
    for bracket_name, _, _ in AGI_BRACKETS:
        if bracket_name not in IRS_SOI_2021_RETURNS_BY_AGI:
            continue

        indicator = (df['agi_bracket'] == bracket_name).astype(float).values
        n_obs = indicator.sum()

        if n_obs >= min_obs:
            constraints.append(Constraint(
                indicator=indicator,
                target_value=IRS_SOI_2021_RETURNS_BY_AGI[bracket_name],
                variable=f'returns_{bracket_name}',
                target_type=TargetType.COUNT,
                tolerance=0.05,
                stratum_name=f'Filers AGI {bracket_name}',
            ))

    return constraints


def entropy_calibrate(
    original_weights: np.ndarray,
    constraints: list[Constraint],
    bounds: tuple[float, float] = (0.2, 5.0),
    max_iter: int = 200,
    tol: float = 1e-8,
    verbose: bool = True,
) -> tuple[np.ndarray, bool, float]:
    """
    Calibrate weights using entropy minimization (gradient descent).

    Uses the dual formulation: instead of optimizing n weights directly,
    we optimize m Lagrange multipliers (one per constraint) which is
    much more efficient.

    The optimal weights are: w_i = w0_i * exp(sum_j lambda_j * A_ij)
    where A_ij is the constraint matrix.
    """
    n = len(original_weights)
    m = len(constraints)

    if verbose:
        print(f"Entropy calibration: {n:,} weights, {m} constraints")

    # Build constraint matrix A (m x n)
    # Each row is an indicator/value vector for one constraint
    A = np.zeros((m, n))
    targets = np.zeros(m)

    for j, c in enumerate(constraints):
        A[j, :] = c.indicator
        targets[j] = c.target_value

    # Dual objective: find lambdas that minimize the dual
    # Dual = sum_i w0_i * exp(sum_j lambda_j * A_ji) - sum_j lambda_j * target_j
    def dual_objective(lambdas: np.ndarray) -> float:
        # Compute log adjustment: sum_j lambda_j * A_ji for each i
        log_adj = A.T @ lambdas  # (n,)

        # Clip for numerical stability
        log_adj = np.clip(log_adj, -10, 10)

        # Calibrated weights
        w = original_weights * np.exp(log_adj)

        # Dual value
        return w.sum() - lambdas @ targets

    def dual_gradient(lambdas: np.ndarray) -> np.ndarray:
        log_adj = A.T @ lambdas
        log_adj = np.clip(log_adj, -10, 10)
        w = original_weights * np.exp(log_adj)

        # Gradient: sum_i w_i * A_ji - target_j = achieved_j - target_j
        achieved = A @ w
        return achieved - targets

    # Initial lambdas: zeros (no adjustment)
    lambda0 = np.zeros(m)

    # Optimize using L-BFGS-B (gradient descent with bounds)
    result = minimize(
        dual_objective,
        lambda0,
        method='L-BFGS-B',
        jac=dual_gradient,
        options={
            'maxiter': max_iter,
            'ftol': tol,
            'gtol': 1e-6,
        }
    )

    if verbose:
        print(f"Optimization: {result.message}")
        print(f"Iterations: {result.nit}, Function evals: {result.nfev}")

    # Compute final weights
    log_adj = A.T @ result.x
    log_adj = np.clip(log_adj, np.log(bounds[0]), np.log(bounds[1]))
    calibrated_weights = original_weights * np.exp(log_adj)

    # Compute KL divergence
    w_safe = np.maximum(calibrated_weights, 1e-10)
    w0_safe = np.maximum(original_weights, 1e-10)
    kl_div = np.sum(w_safe * np.log(w_safe / w0_safe))

    return calibrated_weights, result.success, kl_div


def calibrate_weights(
    df: pd.DataFrame,
    bounds: tuple[float, float] = (0.2, 5.0),
    tolerance: float = 0.05,
    min_obs: int = 100,
    verbose: bool = True,
) -> CalibrationResult:
    """
    Calibrate weights using entropy minimization.

    Args:
        df: Tax unit DataFrame with 'weight' and 'adjusted_gross_income'
        bounds: (min_ratio, max_ratio) for weight adjustments
        tolerance: Allowed deviation from targets
        min_obs: Minimum observations for a constraint
        verbose: Print progress
    """
    original_weights = df['weight'].values.copy()
    n = len(original_weights)

    if verbose:
        print(f"Calibrating {n:,} tax units...")
        print(f"Original weighted total: {original_weights.sum():,.0f}")

    # Build constraints
    constraints = build_constraints(df, min_obs=min_obs)

    if verbose:
        print(f"Built {len(constraints)} constraints (min {min_obs} obs each)")

    # Compute pre-calibration values
    targets_before = {}
    for c in constraints:
        current = np.dot(original_weights, c.indicator)
        targets_before[c.variable] = {
            'current': current,
            'target': c.target_value,
            'error': (current - c.target_value) / c.target_value if c.target_value != 0 else 0,
        }

    if verbose:
        print("\nPre-calibration errors (sample brackets):")
        sample_brackets = ['returns_50k_to_75k', 'returns_100k_to_200k', 'returns_200k_to_500k']
        for name in sample_brackets:
            if name in targets_before:
                t = targets_before[name]
                print(f"  {name}: {t['error']:+.1%}")

    # Run entropy calibration
    calibrated_weights, success, kl_div = entropy_calibrate(
        original_weights,
        constraints,
        bounds=bounds,
        verbose=verbose,
    )

    # Compute post-calibration values
    targets_after = {}
    max_error = 0
    for c in constraints:
        current = np.dot(calibrated_weights, c.indicator)
        error = (current - c.target_value) / c.target_value if c.target_value != 0 else 0
        targets_after[c.variable] = {
            'current': current,
            'target': c.target_value,
            'error': error,
        }
        max_error = max(max_error, abs(error))

    adjustment_factors = calibrated_weights / original_weights

    sample_brackets = ['returns_50k_to_75k', 'returns_100k_to_200k', 'returns_200k_to_500k']
    if verbose:
        print(f"\nPost-calibration (max error: {max_error:.1%}):")
        for name in sample_brackets:
            if name in targets_after:
                t = targets_after[name]
                print(f"  {name}: {t['error']:+.1%}")

        # Calculate coverage
        calibrated_total = calibrated_weights.sum()
        irs_total = IRS_SOI_2021_TOTAL_RETURNS
        coverage = calibrated_total / irs_total
        print(f"\nCoverage: {calibrated_total:,.0f} / {irs_total:,.0f} = {coverage:.1%}")
        print(f"  (CPS underrepresents low/no-income filers by ~{(1-coverage)*100:.0f}%)")

        print(f"\nWeight adjustments: mean={adjustment_factors.mean():.2f}, "
              f"std={adjustment_factors.std():.2f}, "
              f"range=[{adjustment_factors.min():.2f}, {adjustment_factors.max():.2f}]")
        print(f"KL divergence: {kl_div:.2f}")

    return CalibrationResult(
        original_weights=original_weights,
        calibrated_weights=calibrated_weights,
        adjustment_factors=adjustment_factors,
        targets_before=targets_before,
        targets_after=targets_after,
        success=success and max_error < tolerance,
        message="Converged" if success else "Did not converge",
        kl_divergence=kl_div,
    )


def calibrate_and_run(year: int = 2024, filer_threshold: float = 0) -> pd.DataFrame:
    """Load data, calibrate weights, return calibrated DataFrame."""
    from tax_unit_builder import load_and_build_tax_units

    print("=" * 60)
    print("COSILICO MICRODATA CALIBRATION (Entropy Method)")
    print("=" * 60)

    print("\n1. Loading tax unit data...")
    df = load_and_build_tax_units(year)
    print(f"   Loaded {len(df):,} tax units")

    # Filter to likely filers (have income or already pay tax)
    # IRS filing threshold 2024: ~$13,850 for single, $27,700 for joint
    filer_mask = (
        (df['total_income'] > 13850) |  # Above single threshold
        (df['wage_income'] > 0) |        # Has wage income
        (df['self_employment_income'] > 0)  # Has SE income
    )
    df = df[filer_mask].copy()
    print(f"   Filtered to {len(df):,} likely filers")

    print("\n2. Calibrating to IRS SOI 2021 targets...")
    result = calibrate_weights(df)

    df['original_weight'] = result.original_weights
    df['weight'] = result.calibrated_weights
    df['weight_adjustment'] = result.adjustment_factors

    print("\n3. Summary:")
    print(f"   Original total: {result.original_weights.sum():,.0f}")
    print(f"   Calibrated total: {result.calibrated_weights.sum():,.0f}")
    print(f"   IRS SOI target: {IRS_SOI_2021_TOTAL_RETURNS:,.0f}")

    return df


if __name__ == "__main__":
    df = calibrate_and_run()

    print("\n" + "=" * 60)
    print("CALIBRATED DATA SUMMARY")
    print("=" * 60)
    print(f"\nTotal tax units: {len(df):,}")
    print(f"Weighted population: {df['weight'].sum():,.0f}")
    print(f"Total AGI: ${(df['adjusted_gross_income'] * df['weight']).sum():,.0f}")

    output_path = "tax_units_calibrated_2024.parquet"
    df.to_parquet(output_path)
    print(f"\nSaved to {output_path}")
