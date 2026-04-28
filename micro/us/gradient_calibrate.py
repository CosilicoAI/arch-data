"""
Gradient descent calibration for CPS microdata.

Minimizes squared relative error from targets, following the approach
in PolicyEngine-US-Data. No prior dependence - weights are determined
purely by targets.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
import sys

# Try to import torch, fall back to numpy-based optimization if not available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: torch not available, using scipy optimizer")

from scipy.optimize import minimize

sys.path.insert(0, str(__file__).rsplit('/', 3)[0])


@dataclass
class Target:
    """A calibration target."""
    name: str
    geographic_id: str  # "US", state FIPS, county FIPS, etc.
    variable: str       # e.g., "returns", "agi"
    bracket: str        # e.g., "50k_to_75k" or "all"
    value: float
    is_count: bool = True  # True for count, False for sum


@dataclass
class CalibrationResult:
    """Results from gradient descent calibration."""
    weights: np.ndarray
    original_weights: np.ndarray
    targets_df: pd.DataFrame
    initial_loss: float
    final_loss: float
    epochs: int
    target_errors: Dict[str, float]


# IRS SOI 2021 National Targets
IRS_SOI_2021_RETURNS = {
    'no_agi': 13_992_100,
    'under_1': 1_686_440,
    '1_to_5k': 5_183_390,
    '5k_to_10k': 7_929_860,
    '10k_to_15k': 9_883_050,
    '15k_to_20k': 9_113_990,
    '20k_to_25k': 8_186_640,
    '25k_to_30k': 7_407_890,
    '30k_to_40k': 13_194_450,
    '40k_to_50k': 10_930_780,
    '50k_to_75k': 19_494_660,
    '75k_to_100k': 15_137_070,
    '100k_to_200k': 22_849_380,
    '200k_to_500k': 7_167_290,
    '500k_to_1m': 1_106_040,
    '1m_plus': 664_340,
}

IRS_SOI_2021_AGI = {
    'no_agi': 0,
    'under_1': -94_000_000_000,
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

# IRS SOI 2021 State Targets (top 10 states by population)
# Source: IRS SOI Historic Table 2
IRS_SOI_2021_STATE_RETURNS = {
    '06': 17_847_450,  # California
    '48': 13_592_820,  # Texas
    '12': 11_214_320,  # Florida
    '36': 9_894_560,   # New York
    '42': 6_485_230,   # Pennsylvania
    '17': 6_317_890,   # Illinois
    '39': 5_897_420,   # Ohio
    '13': 5_234_670,   # Georgia
    '37': 5_178_340,   # North Carolina
    '26': 4_923_780,   # Michigan
}

AGI_BRACKETS = [
    ('no_agi', None, None),
    ('under_1', -np.inf, 1),
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


def assign_agi_bracket(agi: np.ndarray) -> np.ndarray:
    """Assign each record to an AGI bracket."""
    brackets = np.empty(len(agi), dtype=object)
    for name, low, high in AGI_BRACKETS:
        if name == 'no_agi':
            mask = (agi == 0) | np.isnan(agi)
        else:
            mask = (agi >= low) & (agi < high)
        brackets[mask] = name
    return brackets


def build_targets(include_states: bool = True) -> List[Target]:
    """Build list of calibration targets."""
    targets = []

    # National returns by AGI bracket
    for bracket, count in IRS_SOI_2021_RETURNS.items():
        targets.append(Target(
            name=f"US/returns/{bracket}",
            geographic_id="US",
            variable="returns",
            bracket=bracket,
            value=count,
            is_count=True,
        ))

    # National AGI by bracket
    for bracket, amount in IRS_SOI_2021_AGI.items():
        if amount != 0:  # Skip zero AGI bracket
            targets.append(Target(
                name=f"US/agi/{bracket}",
                geographic_id="US",
                variable="agi",
                bracket=bracket,
                value=amount,
                is_count=False,
            ))

    # State total returns
    if include_states:
        for state_fips, count in IRS_SOI_2021_STATE_RETURNS.items():
            targets.append(Target(
                name=f"{state_fips}/returns/all",
                geographic_id=state_fips,
                variable="returns",
                bracket="all",
                value=count,
                is_count=True,
            ))

    return targets


def build_indicator_matrix(
    df: pd.DataFrame,
    targets: List[Target],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build indicator matrix and target vector.

    Returns:
        A: Indicator matrix (n_targets x n_records)
        y: Target values (n_targets,)
    """
    n = len(df)
    m = len(targets)

    A = np.zeros((m, n), dtype=np.float32)
    y = np.zeros(m, dtype=np.float32)

    # Precompute brackets and state assignments
    df = df.copy()
    df['agi_bracket'] = assign_agi_bracket(df['adjusted_gross_income'].values)

    for j, target in enumerate(targets):
        y[j] = target.value

        # Geographic filter
        if target.geographic_id == "US":
            geo_mask = np.ones(n, dtype=bool)
        else:
            geo_mask = df['state_fips'].astype(str).str.zfill(2) == target.geographic_id

        # Bracket filter
        if target.bracket == "all":
            bracket_mask = np.ones(n, dtype=bool)
        else:
            bracket_mask = df['agi_bracket'] == target.bracket

        # Combined mask
        mask = geo_mask & bracket_mask

        # Value for this target
        if target.variable == "returns":
            A[j, mask] = 1.0
        elif target.variable == "agi":
            A[j, :] = mask * df['adjusted_gross_income'].values

    return A, y


def create_target_groups(targets: List[Target]) -> np.ndarray:
    """
    Assign group IDs to targets for loss normalization.

    Groups by (geographic_level, variable) so each group contributes
    equally to total loss.
    """
    groups = np.zeros(len(targets), dtype=int)
    group_id = 0

    # Determine geographic level
    def geo_level(geo_id: str) -> int:
        if geo_id == "US":
            return 0  # National
        elif len(geo_id) == 2:
            return 1  # State
        else:
            return 2  # County/ZIP

    # Group by (level, variable)
    seen_groups = {}
    for i, target in enumerate(targets):
        key = (geo_level(target.geographic_id), target.variable)
        if key not in seen_groups:
            seen_groups[key] = group_id
            group_id += 1
        groups[i] = seen_groups[key]

    return groups


def calibrate_torch(
    A: np.ndarray,
    y: np.ndarray,
    initial_weights: np.ndarray,
    groups: np.ndarray,
    epochs: int = 500,
    lr: float = 0.3,
    verbose: bool = True,
) -> Tuple[np.ndarray, float, float]:
    """
    Calibrate weights using PyTorch gradient descent.

    Returns:
        Tuple of (calibrated_weights, initial_loss, final_loss)
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not available")

    A_t = torch.tensor(A, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    groups_t = torch.tensor(groups, dtype=torch.int64)

    # Initialize log-weights
    log_weights = torch.tensor(
        np.log(initial_weights + 1e-10),
        dtype=torch.float32,
        requires_grad=True,
    )

    # Compute group normalization factors
    n_groups = groups.max() + 1
    group_sizes = torch.zeros(n_groups)
    for g in range(n_groups):
        group_sizes[g] = (groups_t == g).sum()

    optimizer = torch.optim.Adam([log_weights], lr=lr)
    initial_loss = None

    for epoch in range(epochs):
        optimizer.zero_grad()

        weights = torch.exp(log_weights)
        estimates = A_t @ weights

        # Squared relative error
        rel_errors = ((estimates - y_t) / (y_t + 1)) ** 2

        # Normalize within groups
        group_losses = torch.zeros(n_groups)
        for g in range(n_groups):
            mask = groups_t == g
            if mask.sum() > 0:
                group_losses[g] = rel_errors[mask].mean()

        loss = group_losses.mean()

        if epoch == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch:4d}: loss = {loss.item():.6f}")

    return torch.exp(log_weights).detach().numpy(), initial_loss, loss.item()


def calibrate_scipy(
    A: np.ndarray,
    y: np.ndarray,
    initial_weights: np.ndarray,
    groups: np.ndarray,
    max_iter: int = 500,
    verbose: bool = True,
) -> Tuple[np.ndarray, float, float]:
    """
    Calibrate weights using scipy L-BFGS-B.

    Returns:
        Tuple of (calibrated_weights, initial_loss, final_loss)
    """
    n_groups = groups.max() + 1

    def objective(log_weights: np.ndarray) -> float:
        weights = np.exp(log_weights)
        estimates = A @ weights
        rel_errors = ((estimates - y) / (y + 1)) ** 2

        # Normalize within groups
        group_losses = np.zeros(n_groups)
        for g in range(n_groups):
            mask = groups == g
            if mask.sum() > 0:
                group_losses[g] = rel_errors[mask].mean()

        return group_losses.mean()

    def gradient(log_weights: np.ndarray) -> np.ndarray:
        weights = np.exp(log_weights)
        estimates = A @ weights

        # d/d(log_w) of ((Aw - y)/(y+1))^2
        # = 2 * (Aw - y) / (y+1)^2 * A * w
        residuals = 2 * (estimates - y) / (y + 1) ** 2

        # Weight by group normalization
        group_weights = np.zeros(len(y))
        for g in range(n_groups):
            mask = groups == g
            if mask.sum() > 0:
                group_weights[mask] = 1.0 / (mask.sum() * n_groups)

        weighted_residuals = residuals * group_weights
        grad = A.T @ weighted_residuals * weights

        return grad

    log_w0 = np.log(initial_weights + 1e-10)

    # Compute initial loss
    initial_loss = objective(log_w0)

    result = minimize(
        objective,
        log_w0,
        method='L-BFGS-B',
        jac=gradient,
        options={'maxiter': max_iter, 'disp': verbose},
    )

    return np.exp(result.x), initial_loss, result.fun


def calibrate_weights(
    df: pd.DataFrame,
    include_states: bool = True,
    epochs: int = 500,
    lr: float = 0.3,
    verbose: bool = True,
) -> CalibrationResult:
    """
    Calibrate weights to IRS SOI targets.

    Args:
        df: DataFrame with weight, adjusted_gross_income, state_fips
        include_states: Whether to include state-level targets
        epochs: Number of optimization epochs
        lr: Learning rate (for torch optimizer)
        verbose: Print progress

    Returns:
        CalibrationResult with calibrated weights and diagnostics
    """
    original_weights = df['weight'].values.copy()

    # Build targets
    targets = build_targets(include_states=include_states)
    if verbose:
        print(f"Built {len(targets)} targets")

    # Build indicator matrix
    A, y = build_indicator_matrix(df, targets)
    if verbose:
        print(f"Indicator matrix: {A.shape}")

    # Create target groups
    groups = create_target_groups(targets)
    n_groups = groups.max() + 1
    if verbose:
        print(f"Target groups: {n_groups}")

    # Calibrate
    if HAS_TORCH:
        weights, initial_loss, final_loss = calibrate_torch(
            A, y, original_weights, groups, epochs, lr, verbose
        )
    else:
        weights, initial_loss, final_loss = calibrate_scipy(
            A, y, original_weights, groups, epochs, verbose
        )

    # Compute target errors
    estimates = A @ weights
    target_errors = {}
    for i, target in enumerate(targets):
        rel_error = (estimates[i] - y[i]) / (y[i] + 1)
        target_errors[target.name] = rel_error

    # Build targets dataframe
    targets_df = pd.DataFrame([
        {
            'name': t.name,
            'geographic_id': t.geographic_id,
            'variable': t.variable,
            'bracket': t.bracket,
            'target': t.value,
            'estimate': estimates[i],
            'rel_error': target_errors[t.name],
        }
        for i, t in enumerate(targets)
    ])

    return CalibrationResult(
        weights=weights,
        original_weights=original_weights,
        targets_df=targets_df,
        initial_loss=initial_loss,
        final_loss=final_loss,
        epochs=epochs,
        target_errors=target_errors,
    )


def calibrate_and_run(year: int = 2024) -> pd.DataFrame:
    """Load data, calibrate weights, return calibrated DataFrame."""
    from tax_unit_builder import load_and_build_tax_units

    print("=" * 70)
    print("COSILICO MICRODATA CALIBRATION (Gradient Descent)")
    print("=" * 70)

    print("\n1. Loading tax unit data...")
    df = load_and_build_tax_units(year)
    print(f"   Loaded {len(df):,} tax units")

    # Filter to likely filers
    filer_mask = (
        (df['total_income'] > 13850) |
        (df['wage_income'] > 0) |
        (df['self_employment_income'] > 0)
    )
    df = df[filer_mask].copy()
    print(f"   Filtered to {len(df):,} likely filers")

    print("\n2. Calibrating to IRS SOI 2021 targets...")
    result = calibrate_weights(df, include_states=True)

    df['original_weight'] = result.original_weights
    df['weight'] = result.weights
    df['weight_adjustment'] = result.weights / result.original_weights

    print("\n3. Summary:")
    print(f"   Original total: {result.original_weights.sum():,.0f}")
    print(f"   Calibrated total: {result.weights.sum():,.0f}")
    print(f"   Final loss: {result.final_loss:.6f}")

    # Show worst errors
    print("\n4. Worst target errors:")
    errors = result.targets_df.sort_values('rel_error', key=abs, ascending=False)
    for _, row in errors.head(10).iterrows():
        print(f"   {row['name']}: {row['rel_error']:+.1%}")

    print("\n5. Weight distribution:")
    adj = df['weight_adjustment']
    print(f"   Mean: {adj.mean():.2f}")
    print(f"   Std:  {adj.std():.2f}")
    print(f"   Range: [{adj.min():.2f}, {adj.max():.2f}]")

    return df


if __name__ == "__main__":
    df = calibrate_and_run()

    print("\n" + "=" * 70)
    print("CALIBRATED DATA SUMMARY")
    print("=" * 70)
    print(f"\nTotal tax units: {len(df):,}")
    print(f"Weighted population: {df['weight'].sum():,.0f}")
    print(f"Total AGI: ${(df['adjusted_gross_income'] * df['weight']).sum():,.0f}")

    output_path = "tax_units_calibrated_gradient_2024.parquet"
    df.to_parquet(output_path)
    print(f"\nSaved to {output_path}")
