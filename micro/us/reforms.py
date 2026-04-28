"""
Cosilico Reform System.

Supports both parametric and structural reforms for tax/benefit analysis.

Parametric reforms: Change parameter values (rates, thresholds, amounts)
Structural reforms: Change formulas (add/remove provisions, change logic)
"""

from dataclasses import dataclass
from typing import Callable
from copy import deepcopy

from cosilico_runner import PARAMS_2024, run_all_calculations
from tax_unit_builder import load_and_build_tax_units


@dataclass
class ParametricReform:
    """
    A reform that changes parameter values.

    Examples:
        - Increase EITC maximum credit by 10%
        - Raise CTC to $3,000 per child
        - Lower top income tax rate to 35%
    """
    name: str
    description: str
    parameter_changes: dict  # Nested dict of parameter paths to new values

    def apply(self, params: dict) -> dict:
        """Apply parameter changes to a copy of baseline params."""
        new_params = deepcopy(params)

        for path, value in self.parameter_changes.items():
            # Parse path like 'ctc.credit_per_child' or 'eitc.max_credit.3'
            keys = path.split('.')
            target = new_params

            for key in keys[:-1]:
                if key.isdigit():
                    key = int(key)
                target = target[key]

            final_key = keys[-1]
            if final_key.isdigit():
                final_key = int(final_key)

            target[final_key] = value

        return new_params


@dataclass
class StructuralReform:
    """
    A reform that changes calculation logic.

    Examples:
        - Make CTC fully refundable
        - Add new income threshold for NIIT
        - Create a new universal basic income
    """
    name: str
    description: str
    calculation_overrides: dict[str, Callable]  # variable -> new calculation function


# === Example Parametric Reforms ===

REFORMS = {
    # CTC Expansion (similar to ARPA 2021)
    'ctc_expansion': ParametricReform(
        name="CTC Expansion",
        description="Increase CTC to $3,000 per child ($3,600 under 6), fully refundable",
        parameter_changes={
            'ctc.credit_per_child': 3000,
            'ctc.refundable_max_per_child': 3000,  # Fully refundable
            'ctc.earned_income_threshold': 0,      # No earned income requirement
        }
    ),

    # EITC Boost for childless
    'eitc_childless_boost': ParametricReform(
        name="EITC Childless Boost",
        description="Triple EITC for childless workers",
        parameter_changes={
            'eitc.max_credit.0': 1896,  # 3x current $632
        }
    ),

    # Flat Tax
    'flat_tax_20': ParametricReform(
        name="20% Flat Tax",
        description="Replace progressive brackets with 20% flat tax",
        parameter_changes={
            'brackets_single': [(0, float('inf'), 0.20)],
            'brackets_joint': [(0, float('inf'), 0.20)],
        }
    ),

    # UBI Funding - Eliminate CTC/EITC
    'eliminate_credits': ParametricReform(
        name="Eliminate Refundable Credits",
        description="Zero out EITC and CTC (for UBI swap analysis)",
        parameter_changes={
            'eitc.max_credit.0': 0,
            'eitc.max_credit.1': 0,
            'eitc.max_credit.2': 0,
            'eitc.max_credit.3': 0,
            'ctc.credit_per_child': 0,
            'ctc.credit_per_other_dependent': 0,
        }
    ),

    # SS Taxability Reform
    'eliminate_ss_tax': ParametricReform(
        name="Eliminate SS Benefit Taxation",
        description="Make Social Security benefits fully tax-free",
        parameter_changes={
            'ss_taxability.tier1_rate': 0,
            'ss_taxability.tier2_rate': 0,
        }
    ),

    # Lower NIIT threshold
    'niit_lower_threshold': ParametricReform(
        name="Lower NIIT Threshold",
        description="Lower NIIT threshold to $100k single / $150k joint",
        parameter_changes={
            'niit.threshold_single': 100000,
            'niit.threshold_joint': 150000,
        }
    ),
}


def run_reform_analysis(
    reform: ParametricReform,
    baseline_params: dict = PARAMS_2024,
    year: int = 2024,
) -> dict:
    """
    Run a reform analysis comparing baseline to reformed system.

    Returns dict with baseline, reform, and difference statistics.
    """
    # Load data
    df = load_and_build_tax_units(year)

    # Run baseline
    baseline_df = run_all_calculations(df.copy(), year)

    # Apply reform and run
    reformed_params = reform.apply(baseline_params)

    # Need to modify run_all_calculations to accept params
    # For now, monkey-patch
    import cosilico_runner
    original_params = cosilico_runner.PARAMS_2024
    cosilico_runner.PARAMS_2024 = reformed_params

    reform_df = run_all_calculations(df.copy(), year)

    cosilico_runner.PARAMS_2024 = original_params

    # Calculate differences
    weight = df['weight'].values

    def wtotal(df, col):
        return (df[col] * weight).sum()

    results = {
        'reform': reform.name,
        'description': reform.description,
        'tax_units': len(df),
        'weighted_population': weight.sum(),
    }

    # Compare key variables (names match statute definitions)
    variables = [
        ('eitc', 'EITC'),
        ('total_child_tax_credit', 'CTC'),
        ('income_tax', 'Income Tax'),
        ('self_employment_tax', 'SE Tax'),
        ('niit', 'NIIT'),
    ]

    for var, label in variables:
        baseline_total = wtotal(baseline_df, var)
        reform_total = wtotal(reform_df, var)
        diff = reform_total - baseline_total

        results[f'{label}_baseline'] = baseline_total
        results[f'{label}_reform'] = reform_total
        results[f'{label}_change'] = diff

        # Winners and losers
        person_diff = reform_df[var] - baseline_df[var]
        results[f'{label}_winners'] = (person_diff > 1).sum()
        results[f'{label}_losers'] = (person_diff < -1).sum()

    # Net fiscal impact (positive = costs government money)
    results['net_fiscal_cost'] = (
        results['EITC_change'] +
        results['CTC_change'] -
        results['Income Tax_change'] -
        results['SE Tax_change'] -
        results['NIIT_change']
    )

    return results


def print_reform_analysis(results: dict):
    """Pretty print reform analysis results."""
    print("\n" + "=" * 70)
    print(f"REFORM: {results['reform']}")
    print(f"Description: {results['description']}")
    print("=" * 70)

    print(f"\nTax units analyzed: {results['tax_units']:,}")
    print(f"Weighted population: {results['weighted_population']:,.0f}")

    print("\n--- Fiscal Impact ---")
    for var in ['EITC', 'CTC', 'Income Tax', 'SE Tax', 'NIIT']:
        baseline = results[f'{var}_baseline']
        reform = results[f'{var}_reform']
        change = results[f'{var}_change']
        pct = change / baseline * 100 if baseline != 0 else 0

        print(f"{var:15} Baseline: ${baseline:>15,.0f}  Reform: ${reform:>15,.0f}  Change: ${change:>+15,.0f} ({pct:+.1f}%)")

    print(f"\n{'NET FISCAL COST':15} ${results['net_fiscal_cost']:>+15,.0f}")

    if results['net_fiscal_cost'] > 0:
        print("  (Reform costs government money - credits increase or taxes decrease)")
    else:
        print("  (Reform raises revenue - credits decrease or taxes increase)")

    print("\n--- Distributional Impact ---")
    for var in ['EITC', 'CTC']:
        winners = results[f'{var}_winners']
        losers = results[f'{var}_losers']
        print(f"{var}: {winners:,} winners, {losers:,} losers")


def run_calibrated_reform_analysis(
    reform: ParametricReform,
    baseline_params: dict = PARAMS_2024,
    year: int = 2024,
) -> dict:
    """
    Run reform analysis on CALIBRATED weights.
    """
    from calibrate import calibrate_and_run

    # Load calibrated data
    df = calibrate_and_run(year)

    # Run baseline
    baseline_df = run_all_calculations(df.copy(), year)

    # Apply reform
    import cosilico_runner
    original_params = cosilico_runner.PARAMS_2024
    cosilico_runner.PARAMS_2024 = reform.apply(baseline_params)
    reform_df = run_all_calculations(df.copy(), year)
    cosilico_runner.PARAMS_2024 = original_params

    # Calculate differences
    weight = df['weight'].values

    def wtotal(df, col):
        return (df[col] * weight).sum()

    results = {
        'reform': reform.name,
        'description': reform.description,
        'tax_units': len(df),
        'weighted_population': weight.sum(),
        'calibrated': True,
    }

    variables = [
        ('eitc', 'EITC'),
        ('total_child_tax_credit', 'CTC'),
        ('income_tax', 'Income Tax'),
        ('self_employment_tax', 'SE Tax'),
        ('niit', 'NIIT'),
    ]

    for var, label in variables:
        baseline_total = wtotal(baseline_df, var)
        reform_total = wtotal(reform_df, var)
        diff = reform_total - baseline_total

        results[f'{label}_baseline'] = baseline_total
        results[f'{label}_reform'] = reform_total
        results[f'{label}_change'] = diff

        person_diff = reform_df[var] - baseline_df[var]
        results[f'{label}_winners'] = (person_diff > 1).sum()
        results[f'{label}_losers'] = (person_diff < -1).sum()

    results['net_fiscal_cost'] = (
        results['EITC_change'] +
        results['CTC_change'] -
        results['Income Tax_change'] -
        results['SE Tax_change'] -
        results['NIIT_change']
    )

    return results


if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("COSILICO REFORM ANALYSIS")
    print("=" * 70)

    # Check for calibrated flag
    use_calibrated = '--calibrated' in sys.argv

    if use_calibrated:
        print("\n>>> Using CALIBRATED weights\n")
        # Run one reform with calibration
        reform = REFORMS['ctc_expansion']
        results = run_calibrated_reform_analysis(reform)
        print_reform_analysis(results)
    else:
        print("\n>>> Using RAW CPS weights (add --calibrated for calibrated)\n")
        # Run each reform
        for reform_key, reform in list(REFORMS.items())[:3]:  # Just first 3
            print(f"\n>>> Running: {reform.name}")
            results = run_reform_analysis(reform)
            print_reform_analysis(results)
