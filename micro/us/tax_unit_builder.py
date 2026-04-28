"""
Build tax unit level data from person-level CPS ASEC.

Aggregates person records into tax units with derived variables
needed for tax calculations.
"""

import pandas as pd
from pathlib import Path


def build_tax_units(persons: pd.DataFrame, year: int = 2024) -> pd.DataFrame:
    """
    Build tax unit level data from person-level CPS records.

    Args:
        persons: Person-level CPS DataFrame with tax_unit_id
        year: Tax year for parameter lookups

    Returns:
        Tax unit level DataFrame with aggregated variables
    """
    # Validate required columns
    required = ['tax_unit_id', 'age', 'marital_status', 'weight']
    missing = set(required) - set(persons.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Group by tax unit
    tax_units = []

    for tu_id, group in persons.groupby('tax_unit_id'):
        tu = _process_tax_unit(group, tu_id, year)
        tax_units.append(tu)

    df = pd.DataFrame(tax_units)

    # Compute derived variables
    df = _compute_derived_variables(df, year)

    return df


def _process_tax_unit(group: pd.DataFrame, tu_id: int, year: int) -> dict:
    """Process a single tax unit's members."""

    # Sort by line number to identify head
    group = group.sort_values('line_number')
    head = group.iloc[0]

    # Get spouse if present (line_number referenced by spouse_line_number)
    spouse = None
    if head.get('spouse_line_number', 0) > 0:
        spouse_mask = group['line_number'] == head['spouse_line_number']
        if spouse_mask.any():
            spouse = group[spouse_mask].iloc[0]

    # Identify dependents (everyone else in the tax unit)
    dep_mask = group['line_number'] != head['line_number']
    if spouse is not None:
        dep_mask &= group['line_number'] != spouse['line_number']
    dependents = group[dep_mask]

    # Determine filing status
    filing_status = _determine_filing_status(head, spouse)

    # Count qualifying children (under 17 for CTC, under 19 for EITC)
    num_ctc_children = len(dependents[dependents['age'] < 17])
    num_eitc_children = len(dependents[
        (dependents['age'] < 19) |
        ((dependents['age'] < 24) & (dependents.get('employment_status', 0) == 0))  # Students
    ])
    num_dependents = len(dependents)

    # Aggregate income components
    wage_income = group['wage_salary_income'].sum()
    se_income = group['self_employment_income'].sum() + group.get('farm_self_employment_income', 0).sum()
    interest_income = group['interest_income'].sum()
    dividend_income = group['dividend_income'].sum()
    rental_income = group.get('rental_income', 0).sum()
    ss_income = group['social_security_income'].sum()
    unemployment = group.get('unemployment_compensation', 0).sum()
    other_income = group.get('other_income', 0).sum()

    # Earned income = wages + SE income (for EITC purposes)
    earned_income = wage_income + max(0, se_income)

    # Investment income (for NIIT and EITC disqualification)
    investment_income = interest_income + dividend_income + rental_income

    # Total income
    total_income = (wage_income + se_income + interest_income + dividend_income +
                    rental_income + ss_income + unemployment + other_income)

    # Use head's weight (tax unit weight)
    weight = head['weight']

    # Head's age (for additional standard deduction)
    head_age = head['age']

    # Geography (state FIPS from head)
    state_fips = head.get('state_fips', 0)
    spouse_age = spouse['age'] if spouse is not None else None

    return {
        'tax_unit_id': tu_id,
        'weight': weight,
        'state_fips': state_fips,
        'filing_status': filing_status,
        'head_age': head_age,
        'spouse_age': spouse_age,
        'num_dependents': num_dependents,
        'num_ctc_children': num_ctc_children,
        'num_eitc_children': num_eitc_children,
        'num_other_dependents': num_dependents - num_ctc_children,

        # Income components
        'wage_income': wage_income,
        'self_employment_income': se_income,
        'interest_income': interest_income,
        'dividend_income': dividend_income,
        'rental_income': rental_income,
        'social_security_income': ss_income,
        'unemployment_compensation': unemployment,
        'other_income': other_income,

        # Aggregates
        'earned_income': earned_income,
        'investment_income': investment_income,
        'total_income': total_income,

        # Tax amounts from CPS (for validation)
        'cps_federal_tax': group['federal_tax'].sum() if 'federal_tax' in group.columns else 0,
        'cps_eitc': group['eitc_received'].sum() if 'eitc_received' in group.columns else 0,
        'cps_ctc': group['ctc_received'].sum() if 'ctc_received' in group.columns else 0,
    }


def _determine_filing_status(head: pd.Series, spouse: pd.Series | None) -> str:
    """
    Determine filing status from CPS marital status.

    CPS marital status codes:
    1 = Married, spouse present
    2 = Married, spouse absent
    3 = Widowed
    4 = Divorced
    5 = Separated
    6 = Never married
    """
    marital = head.get('marital_status', 6)

    if spouse is not None or marital == 1:
        return 'JOINT'
    elif marital == 3:
        # Widowed - could be Qualifying Widow(er) if recent and has dependent child
        return 'SINGLE'  # Simplified - would need more info for QW
    elif marital in [4, 5, 6]:  # Divorced, separated, never married
        # Check if has dependents for Head of Household
        # This is simplified - real HoH has more requirements
        return 'SINGLE'
    else:
        return 'SINGLE'


def _compute_derived_variables(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Compute DATA-DERIVED variables only.

    NO POLICY CALCULATIONS HERE. All policy rules belong in .rac files:
    - adjusted_gross_income -> 26 USC § 62
    - standard_deduction -> 26 USC § 63(c)
    - taxable_income -> 26 USC § 63
    - is_head_of_household -> 26 USC § 2(b)

    This function only computes simple aggregations from raw CPS data.
    """
    # Simple income aggregations (not policy - just sums)
    df['earned_income'] = df['wage_income'] + df['self_employment_income']
    df['investment_income'] = df['interest_income'] + df['dividend_income']
    df['total_income'] = (
        df['wage_income'] +
        df['self_employment_income'] +
        df['interest_income'] +
        df['dividend_income'] +
        df['rental_income'] +
        df['social_security_income'] +
        df['unemployment_compensation'] +
        df['other_income']
    )

    # Filing status derived from CPS marital status (not policy)
    # is_joint = married spouse present per CPS definition
    df['is_joint'] = df['filing_status'] == 'JOINT'

    return df


def load_and_build_tax_units(year: int = 2024) -> pd.DataFrame:
    """
    Load CPS and build tax unit data.

    Args:
        year: Tax year

    Returns:
        Tax unit level DataFrame
    """
    data_dir = Path(__file__).parent
    cps_path = data_dir / f"cps_{year}.parquet"

    if not cps_path.exists():
        raise FileNotFoundError(f"CPS data not found at {cps_path}")

    persons = pd.read_parquet(cps_path)
    return build_tax_units(persons, year)


if __name__ == "__main__":
    # Test the builder
    print("Loading CPS 2024 and building tax units...")
    df = load_and_build_tax_units(2024)

    print(f"\nTax units: {len(df):,}")
    print(f"Weighted population: {df['weight'].sum():,.0f}")

    print("\nFiling status distribution:")
    for status, count in df['filing_status'].value_counts().items():
        weighted = df[df['filing_status'] == status]['weight'].sum()
        print(f"  {status}: {count:,} ({weighted:,.0f} weighted)")

    print("\nChildren distribution:")
    for n in range(4):
        count = (df['num_ctc_children'] == n).sum()
        print(f"  {n} children: {count:,}")
    print(f"  3+ children: {(df['num_ctc_children'] >= 3).sum():,}")

    print("\nIncome statistics (raw totals, not AGI):")
    print(f"  Mean total income: ${df['total_income'].mean():,.0f}")
    print(f"  Median total income: ${df['total_income'].median():,.0f}")
    print(f"  Total income: ${(df['total_income'] * df['weight']).sum():,.0f}")
