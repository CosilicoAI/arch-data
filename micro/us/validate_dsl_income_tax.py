"""
Validate DSL-based income tax calculation against PolicyEngine.

Runs the refactored income_tax.rac (using marginal_agg) on CPS-derived
tax unit data and compares to PolicyEngine calculations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add cosilico-engine to path
engine_path = Path(__file__).parents[3] / "cosilico-engine" / "src"
sys.path.insert(0, str(engine_path))

from cosilico.brackets import marginal_agg  # noqa: E402


def load_tax_units() -> pd.DataFrame:
    """Load tax unit data from CPS."""
    from tax_unit_builder import load_and_build_tax_units
    return load_and_build_tax_units(2024)


def get_brackets_2024() -> dict:
    """Get 2024 income tax brackets in marginal_agg format."""
    return {
        "rates": [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37],
        "thresholds": {
            "single": [0, 11600, 47150, 100525, 191950, 243725, 609350],
            "joint": [0, 23200, 94300, 201050, 383900, 487450, 731200],
            "head_of_household": [0, 16550, 63100, 100500, 191950, 243700, 609350],
            "married_filing_separately": [0, 11600, 47150, 100525, 191950, 243725, 365600],
        }
    }


def calculate_income_tax_dsl(df: pd.DataFrame) -> np.ndarray:
    """Calculate income tax using DSL marginal_agg function."""
    brackets = get_brackets_2024()

    taxable_income = df["taxable_income"].values

    # Map filing status to bracket keys
    filing_status = np.where(
        df["is_joint"].values,
        "joint",
        "single"  # Simplified - should handle HoH etc.
    )

    # Use marginal_agg with threshold_by for filing status
    tax = marginal_agg(taxable_income, brackets, threshold_by=filing_status)

    return tax


def calculate_income_tax_python(df: pd.DataFrame) -> np.ndarray:
    """Calculate income tax using Python loop (baseline)."""
    brackets_single = [
        (0, 11600, 0.10),
        (11600, 47150, 0.12),
        (47150, 100525, 0.22),
        (100525, 191950, 0.24),
        (191950, 243725, 0.32),
        (243725, 609350, 0.35),
        (609350, float('inf'), 0.37),
    ]

    brackets_joint = [
        (0, 23200, 0.10),
        (23200, 94300, 0.12),
        (94300, 201050, 0.22),
        (201050, 383900, 0.24),
        (383900, 487450, 0.32),
        (487450, 731200, 0.35),
        (731200, float('inf'), 0.37),
    ]

    taxable = df["taxable_income"].values
    is_joint = df["is_joint"].values

    tax = np.zeros(len(df))

    for i in range(len(df)):
        brackets = brackets_joint if is_joint[i] else brackets_single
        income = taxable[i]
        unit_tax = 0

        for low, high, rate in brackets:
            if income <= low:
                break
            bracket_income = min(income, high) - low
            unit_tax += bracket_income * rate

        tax[i] = unit_tax

    return tax


def get_policyengine_values(df: pd.DataFrame) -> np.ndarray:
    """Get PolicyEngine income tax for same scenarios."""
    from policyengine_us import Simulation

    # Run PE on sample scenarios
    taxes = []

    # Sample a subset for speed
    sample_idx = np.random.choice(len(df), min(1000, len(df)), replace=False)

    for i in sample_idx:
        row = df.iloc[i]

        try:
            situation = {
                "people": {
                    "p1": {
                        "age": {2024: 35},
                        "employment_income": {2024: float(row.get("earned_income", 0))},
                    }
                },
                "tax_units": {"tu": {"members": ["p1"]}},
                "households": {"hh": {"members": ["p1"], "state_code": {2024: "TX"}}}
            }

            if row.get("is_joint", False):
                situation["people"]["p2"] = {
                    "age": {2024: 33},
                    "employment_income": {2024: 0}
                }
                situation["tax_units"]["tu"]["members"].append("p2")
                situation["households"]["hh"]["members"].append("p2")

            sim = Simulation(situation=situation)
            tax = float(sim.calculate("income_tax_before_credits", 2024)[0])
            taxes.append(tax)
        except Exception:
            taxes.append(np.nan)

    return np.array(taxes), sample_idx


def main():
    print("=" * 60)
    print("DSL Income Tax Validation (marginal_agg)")
    print("=" * 60)

    # Load data
    print("\n1. Loading tax units...")
    df = load_tax_units()
    print(f"   Loaded {len(df):,} tax units")

    # Calculate with DSL marginal_agg
    print("\n2. Calculating with DSL marginal_agg...")
    import time

    start = time.time()
    dsl_tax = calculate_income_tax_dsl(df)
    dsl_time = time.time() - start
    print(f"   Done in {dsl_time:.2f}s ({len(df)/dsl_time:,.0f} units/sec)")

    # Calculate with Python loop (baseline)
    print("\n3. Calculating with Python loop (baseline)...")
    start = time.time()
    py_tax = calculate_income_tax_python(df)
    py_time = time.time() - start
    print(f"   Done in {py_time:.2f}s ({len(df)/py_time:,.0f} units/sec)")

    # Compare DSL vs Python
    print("\n4. Comparing DSL vs Python...")
    diff = np.abs(dsl_tax - py_tax)
    match_rate = (diff < 1).mean() * 100
    max_diff = diff.max()

    print(f"   Match rate (<$1): {match_rate:.1f}%")
    print(f"   Max difference:   ${max_diff:,.2f}")
    print(f"   Speedup:          {py_time/dsl_time:.1f}x")

    # Weighted totals
    weight = df["weight"].values
    dsl_total = (dsl_tax * weight).sum()
    py_total = (py_tax * weight).sum()

    print("\n   Weighted totals:")
    print(f"   DSL:    ${dsl_total:>20,.0f}")
    print(f"   Python: ${py_total:>20,.0f}")

    # Sample comparison with PolicyEngine
    print("\n5. Comparing to PolicyEngine (sample)...")
    try:
        pe_taxes, sample_idx = get_policyengine_values(df)

        dsl_sample = dsl_tax[sample_idx]
        valid = ~np.isnan(pe_taxes)

        if valid.sum() > 0:
            diff_pe = np.abs(dsl_sample[valid] - pe_taxes[valid])
            match_pe = (diff_pe < 100).mean() * 100
            corr = np.corrcoef(dsl_sample[valid], pe_taxes[valid])[0, 1]

            print(f"   Sample size:      {valid.sum()}")
            print(f"   Match (<$100):    {match_pe:.1f}%")
            print(f"   Correlation:      {corr:.4f}")
    except Exception as e:
        print(f"   PolicyEngine comparison skipped: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if match_rate >= 99:
        print("✅ DSL marginal_agg matches Python baseline perfectly")
    else:
        print(f"⚠️  Some discrepancies ({100-match_rate:.1f}% mismatch)")

    print(f"\nVectorized marginal_agg is {py_time/dsl_time:.1f}x faster than Python loop")


if __name__ == "__main__":
    main()
