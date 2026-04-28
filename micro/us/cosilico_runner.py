"""
Run Cosilico encodings on CPS microdata.

Applies statute-based tax calculations to tax unit data.
"""

import numpy as np
import pandas as pd


# 2024 Tax Parameters (from Rev. Proc. 2023-34)
PARAMS_2024 = {
    # EITC parameters
    'eitc': {
        'max_credit': {0: 632, 1: 3995, 2: 6604, 3: 7430},
        'earned_income_threshold': {0: 7840, 1: 11750, 2: 16510, 3: 16510},
        'phaseout_start': {
            'single': {0: 9800, 1: 21560, 2: 21560, 3: 21560},
            'joint': {0: 16370, 1: 28120, 2: 28120, 3: 28120},
        },
        'phaseout_rate': {0: 0.0765, 1: 0.1598, 2: 0.2106, 3: 0.2106},
        'investment_income_limit': 11600,
    },

    # CTC parameters
    'ctc': {
        'credit_per_child': 2000,
        'credit_per_other_dependent': 500,
        'phaseout_threshold_joint': 400000,
        'phaseout_threshold_single': 200000,
        'phaseout_rate': 50,  # $50 per $1000
        'refundable_max_per_child': 1700,
        'earned_income_threshold': 2500,
        'earned_income_rate': 0.15,
    },

    # Self-employment tax
    'se_tax': {
        'oasdi_rate': 0.124,
        'hi_rate': 0.029,
        'ss_wage_base': 168600,
        'net_earnings_factor': 0.9235,
    },

    # Income tax brackets (single)
    'brackets_single': [
        (0, 11600, 0.10),
        (11600, 47150, 0.12),
        (47150, 100525, 0.22),
        (100525, 191950, 0.24),
        (191950, 243725, 0.32),
        (243725, 609350, 0.35),
        (609350, float('inf'), 0.37),
    ],

    # Income tax brackets (joint)
    'brackets_joint': [
        (0, 23200, 0.10),
        (23200, 94300, 0.12),
        (94300, 201050, 0.22),
        (201050, 383900, 0.24),
        (383900, 487450, 0.32),
        (487450, 731200, 0.35),
        (731200, float('inf'), 0.37),
    ],

    # Income tax brackets (head of household) per 26 USC § 1(b)
    'brackets_hoh': [
        (0, 16550, 0.10),
        (16550, 63100, 0.12),
        (63100, 100500, 0.22),
        (100500, 191950, 0.24),
        (191950, 243700, 0.32),
        (243700, 609350, 0.35),
        (609350, float('inf'), 0.37),
    ],

    # Social Security taxability thresholds (frozen since 1984)
    'ss_taxability': {
        'base_single': 25000,
        'base_joint': 32000,
        'adjusted_single': 34000,
        'adjusted_joint': 44000,
        'tier1_rate': 0.50,
        'tier2_rate': 0.85,
    },

    # NIIT
    'niit': {
        'rate': 0.038,
        'threshold_single': 200000,
        'threshold_joint': 250000,
    },

    # Standard deduction (26 USC § 63)
    'standard_deduction': {
        'basic_single': 14600,
        'basic_joint': 29200,
        'basic_hoh': 21900,
        'additional_single_or_hoh': 1950,  # 65+ or blind
        'additional_joint': 1550,  # per person 65+ or blind
        'dependent_minimum': 1300,
        'dependent_earned_addition': 450,
    },
}


def calculate_eitc(df: pd.DataFrame, params: dict) -> np.ndarray:
    """Calculate EITC per 26 USC § 32.

    Per the statute and IRS verification (EITC_VALIDATION_REPORT.md):
    - Phase-in rate (credit percentage) is DIFFERENT from phaseout rate
    - 0 children: 7.65% phase-in, 7.65% phaseout
    - 1 child: 34.00% phase-in, 15.98% phaseout
    - 2 children: 40.00% phase-in, 21.06% phaseout
    - 3+ children: 45.00% phase-in, 21.06% phaseout

    FIX (2025-12-27): Previously used phaseout_rate for phase-in calculation.
    Now correctly computes phase_in_rate = max_credit / earned_income_threshold.
    """
    p = params['eitc']

    # Cap children at 3 for EITC purposes
    n_children = np.minimum(df['num_eitc_children'].values, 3)

    # Check investment income limit
    disqualified = df['investment_income'].values > p['investment_income_limit']

    # Get max credit by number of children
    max_credit = np.array([p['max_credit'].get(n, p['max_credit'][3]) for n in n_children])

    # Get earned income threshold (where phase-in plateau begins)
    ei_threshold = np.array([p['earned_income_threshold'].get(n, p['earned_income_threshold'][3]) for n in n_children])

    # CORRECT: Phase-in rate = max_credit / earned_income_threshold
    # This is the "credit percentage" per 26 USC § 32(b)(1)
    phase_in_rate = max_credit / ei_threshold

    # Get phaseout start
    is_joint = df['is_joint'].values
    phaseout_start = np.where(
        is_joint,
        np.array([p['phaseout_start']['joint'].get(n, p['phaseout_start']['joint'][3]) for n in n_children]),
        np.array([p['phaseout_start']['single'].get(n, p['phaseout_start']['single'][3]) for n in n_children])
    )

    # Get phaseout rate (different from phase-in rate!)
    phaseout_rate = np.array([p['phaseout_rate'].get(n, p['phaseout_rate'][3]) for n in n_children])

    earned = df['earned_income'].values
    agi = df['adjusted_gross_income'].values

    # Phase-in: credit builds up at phase_in_rate until max_credit reached
    phase_in_credit = np.minimum(max_credit, earned * phase_in_rate)

    # Phase-out: credit reduces at phaseout_rate as income exceeds threshold
    phase_out_income = np.maximum(earned, agi)
    excess_income = np.maximum(0, phase_out_income - phaseout_start)
    phase_out_reduction = excess_income * phaseout_rate

    # Final credit
    eitc = np.maximum(0, phase_in_credit - phase_out_reduction)

    # Zero out if disqualified by investment income
    eitc = np.where(disqualified, 0, eitc)

    return eitc


def calculate_ctc(df: pd.DataFrame, params: dict, tax_before_credits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate Child Tax Credit per 26 USC § 24.

    The non-refundable CTC offsets tax liability first.
    The refundable ACTC (Additional CTC) is for the remaining amount,
    limited by the earned income formula.

    Args:
        df: Tax unit DataFrame
        params: Tax parameters
        tax_before_credits: Income tax before any credits (needed to limit non-refundable)

    Returns (nonrefundable_ctc, refundable_actc)
    """
    p = params['ctc']

    n_children = df['num_ctc_children'].values
    n_other_deps = df['num_other_dependents'].values

    # Credit before phaseout per § 24(a)
    credit_before = n_children * p['credit_per_child'] + n_other_deps * p['credit_per_other_dependent']

    # Phaseout per § 24(b)
    threshold = np.where(
        df['is_joint'].values,
        p['phaseout_threshold_joint'],
        p['phaseout_threshold_single']
    )

    # Phaseout reduction: $50 per $1000 over threshold
    agi = df['adjusted_gross_income'].values
    excess = np.maximum(0, agi - threshold)
    increments = np.ceil(excess / 1000)
    phaseout = increments * p['phaseout_rate']

    # Tentative credit after phaseout
    tentative = np.maximum(0, credit_before - phaseout)

    # Step 1: Non-refundable CTC limited by tax liability per § 24(a)
    # Can only offset tax, not go negative
    nonrefundable = np.minimum(tentative, np.maximum(0, tax_before_credits))

    # Step 2: Remaining credit that could be refundable
    remaining = tentative - nonrefundable

    # Step 3: ACTC (refundable) per § 24(d) - earned income formula
    earned = df['earned_income'].values
    excess_earned = np.maximum(0, earned - p['earned_income_threshold'])
    actc_earned_portion = p['earned_income_rate'] * excess_earned

    # Per-child cap on refundable amount per § 24(h)(5)
    actc_cap = n_children * p['refundable_max_per_child']
    actc_limit = np.minimum(actc_earned_portion, actc_cap)

    # ACTC is min of remaining credit and earned income limit
    actc = np.minimum(remaining, actc_limit)

    return nonrefundable, actc


def calculate_se_tax(df: pd.DataFrame, params: dict) -> np.ndarray:
    """Calculate Self-Employment Tax per 26 USC § 1401."""
    p = params['se_tax']

    se_income = df['self_employment_income'].values

    # Net earnings = 92.35% of SE income
    net_earnings = np.maximum(0, se_income * p['net_earnings_factor'])

    # Social Security portion (capped at wage base)
    ss_taxable = np.minimum(net_earnings, p['ss_wage_base'])
    ss_tax = ss_taxable * p['oasdi_rate']

    # Medicare portion (no cap)
    medicare_tax = net_earnings * p['hi_rate']

    return ss_tax + medicare_tax


def calculate_is_head_of_household(df: pd.DataFrame) -> np.ndarray:
    """Determine Head of Household status per 26 USC § 2(b).

    Per statute: unmarried individual who maintains household for qualifying person.
    Simplified: not married AND has at least one dependent.
    """
    is_joint = df['is_joint'].values
    has_dependents = df['num_dependents'].values > 0

    # HOH requires: (1) not married, (2) has qualifying person
    return (~is_joint) & has_dependents


def calculate_taxable_ss_for_agi(df: pd.DataFrame, params: dict) -> np.ndarray:
    """Calculate taxable Social Security for AGI calculation per 26 USC § 86.

    This is called during AGI calculation before AGI is finalized.
    Uses provisional income = (other income) + 50% * SS benefits.
    """
    p = params['ss_taxability']

    ss_benefits = df['social_security_income'].values
    is_joint = df['is_joint'].values

    # Provisional income for SS taxability (MAGI excluding SS + 50% SS)
    # Use total income minus SS as approximation of MAGI
    other_income = (
        df['wage_income'].values +
        df['self_employment_income'].values +
        df['interest_income'].values +
        df['dividend_income'].values +
        df['rental_income'].values +
        df['unemployment_compensation'].values +
        df.get('other_income', pd.Series(0, index=df.index)).values
    )

    provisional = other_income + 0.5 * ss_benefits

    # Get thresholds
    base = np.where(is_joint, p['base_joint'], p['base_single'])
    adjusted = np.where(is_joint, p['adjusted_joint'], p['adjusted_single'])

    # Tier 1: 50% of lesser of (benefits, excess over base)
    excess_base = np.maximum(0, provisional - base)
    tier1 = np.minimum(p['tier1_rate'] * ss_benefits, p['tier1_rate'] * excess_base)

    # Tier 2: 35% of excess over adjusted base
    excess_adjusted = np.maximum(0, provisional - adjusted)
    tier2_addition = (p['tier2_rate'] - p['tier1_rate']) * excess_adjusted

    # Total capped at 85%
    total = tier1 + tier2_addition
    max_taxable = p['tier2_rate'] * ss_benefits

    return np.minimum(total, max_taxable)


def calculate_adjusted_gross_income(df: pd.DataFrame, params: dict) -> np.ndarray:
    """Calculate Adjusted Gross Income per 26 USC § 62.

    AGI = Gross Income - Above-the-line deductions

    Above-the-line deductions include:
    - 50% of self-employment tax (§ 164(f))
    - Self-employed health insurance (§ 162(l)) - not modeled
    - IRA contributions (§ 219) - not modeled
    - Student loan interest (§ 221) - not modeled
    """
    # Gross income components
    wages = df['wage_income'].values
    se_income = df['self_employment_income'].values
    interest = df['interest_income'].values
    dividends = df['dividend_income'].values
    rental = df['rental_income'].values
    ui = df['unemployment_compensation'].values
    other = df.get('other_income', pd.Series(0, index=df.index)).values

    # Calculate taxable SS (uses provisional income, no circular dependency)
    taxable_ss = calculate_taxable_ss_for_agi(df, params)

    # Self-employment tax deduction (50% of SE tax)
    se_tax = calculate_se_tax(df, params)
    se_tax_deduction = se_tax * 0.5

    # Gross income
    gross_income = (
        wages +
        se_income +
        interest +
        dividends +
        rental +
        taxable_ss +  # Only taxable portion of SS
        ui +
        other
    )

    # AGI = Gross - above-the-line deductions
    agi = gross_income - se_tax_deduction

    return np.maximum(0, agi)


def calculate_standard_deduction_simple(df: pd.DataFrame, params: dict) -> np.ndarray:
    """Calculate standard deduction per 26 USC § 63(c).

    Basic standard deduction by filing status, plus additional amounts
    for age 65+ or blind.
    """
    p = params['standard_deduction']

    is_joint = df['is_joint'].values
    is_hoh = df['is_head_of_household'].values

    # Basic standard deduction by filing status
    basic = np.where(
        is_joint,
        p['basic_joint'],
        np.where(is_hoh, p['basic_hoh'], p['basic_single'])
    )

    # Additional deduction for age 65+
    head_age = df['head_age'].values
    spouse_age = df['spouse_age'].fillna(0).values

    head_65_plus = head_age >= 65
    spouse_65_plus = (spouse_age >= 65) & is_joint

    # Additional amount per condition
    additional_per = np.where(is_joint, p['additional_joint'], p['additional_single_or_hoh'])

    additional = head_65_plus.astype(int) * additional_per
    additional = additional + spouse_65_plus.astype(int) * p['additional_joint']

    return basic + additional


def calculate_taxable_income(df: pd.DataFrame, params: dict) -> np.ndarray:
    """Calculate taxable income per 26 USC § 63.

    Taxable income = AGI - (standard or itemized deduction)
    Simplified: always uses standard deduction.
    """
    agi = df['adjusted_gross_income'].values
    std_ded = df['standard_deduction'].values

    return np.maximum(0, agi - std_ded)


def calculate_income_tax(df: pd.DataFrame, params: dict) -> np.ndarray:
    """Calculate income tax using progressive brackets per 26 USC § 1."""
    taxable = df['taxable_income'].values
    is_joint = df['is_joint'].values
    is_hoh = df['is_head_of_household'].values

    tax = np.zeros(len(df))

    for i in range(len(df)):
        if is_joint[i]:
            brackets = params['brackets_joint']
        elif is_hoh[i]:
            brackets = params['brackets_hoh']
        else:
            brackets = params['brackets_single']

        income = taxable[i]
        unit_tax = 0

        for low, high, rate in brackets:
            if income <= low:
                break
            bracket_income = min(income, high) - low
            unit_tax += bracket_income * rate

        tax[i] = unit_tax

    return tax


def calculate_taxable_ss(df: pd.DataFrame, params: dict) -> np.ndarray:
    """Calculate taxable Social Security per 26 USC § 86."""
    p = params['ss_taxability']

    ss_benefits = df['social_security_income'].values

    # Combined income = MAGI + 50% of SS benefits
    # Using AGI as simplified MAGI
    combined = df['adjusted_gross_income'].values + 0.5 * ss_benefits

    is_joint = df['is_joint'].values

    # Get thresholds
    base = np.where(is_joint, p['base_joint'], p['base_single'])
    adjusted = np.where(is_joint, p['adjusted_joint'], p['adjusted_single'])

    # Tier 1: 50% of lesser of (benefits, excess over base)
    excess_base = np.maximum(0, combined - base)
    tier1 = np.minimum(p['tier1_rate'] * ss_benefits, p['tier1_rate'] * excess_base)

    # Tier 2: 35% of excess over adjusted base
    excess_adjusted = np.maximum(0, combined - adjusted)
    tier2_addition = (p['tier2_rate'] - p['tier1_rate']) * excess_adjusted

    # Total capped at 85%
    total = tier1 + tier2_addition
    max_taxable = p['tier2_rate'] * ss_benefits

    return np.minimum(total, max_taxable)


def calculate_niit(df: pd.DataFrame, params: dict) -> np.ndarray:
    """Calculate Net Investment Income Tax per 26 USC § 1411."""
    p = params['niit']

    threshold = np.where(
        df['is_joint'].values,
        p['threshold_joint'],
        p['threshold_single']
    )

    agi = df['adjusted_gross_income'].values
    investment = df['investment_income'].values

    # NIIT on lesser of (investment income, AGI over threshold)
    excess_agi = np.maximum(0, agi - threshold)
    niit_base = np.minimum(investment, excess_agi)

    return niit_base * p['rate']


def calculate_standard_deduction(df: pd.DataFrame, params: dict) -> np.ndarray:
    """Calculate standard deduction per 26 USC § 63.

    The standard deduction consists of:
    1. Basic standard deduction based on filing status
    2. Additional deduction for age 65+ or blind
    3. Special rules for dependents (limited to earned income + $450, min $1,300)
    """
    p = params['standard_deduction']

    is_joint = df['is_joint'].values
    is_hoh = df['is_head_of_household'].values
    is_dependent = df['is_dependent'].values
    age_head = df['age_head'].values
    age_spouse = df['age_spouse'].values
    is_blind_head = df['is_blind_head'].values
    is_blind_spouse = df['is_blind_spouse'].values
    earned_income = df['earned_income'].values

    # Basic standard deduction by filing status
    basic = np.where(
        is_joint,
        p['basic_joint'],
        np.where(is_hoh, p['basic_hoh'], p['basic_single'])
    )

    # Additional deduction for age 65+ or blind
    # Joint filers: $1,550 per person who is 65+ or blind
    # Single/HOH: $1,950 per condition (65+ or blind)
    head_65_plus = age_head >= 65
    spouse_65_plus = age_spouse >= 65

    # Count additional deductions
    additional_count_head = head_65_plus.astype(int) + is_blind_head.astype(int)
    additional_count_spouse = np.where(
        is_joint,
        spouse_65_plus.astype(int) + is_blind_spouse.astype(int),
        0
    )

    # Additional amount per condition
    additional_per_condition = np.where(
        is_joint,
        p['additional_joint'],
        p['additional_single_or_hoh']
    )

    additional = (additional_count_head + additional_count_spouse) * additional_per_condition

    # Total before dependent limitation
    total = basic + additional

    # Dependent limitation: min($1,300, earned + $450), capped at basic
    dependent_limit = np.maximum(
        p['dependent_minimum'],
        np.minimum(earned_income + p['dependent_earned_addition'], basic)
    )

    # Apply dependent limitation where applicable
    result = np.where(is_dependent, dependent_limit, total)

    return result


def run_all_calculations(df: pd.DataFrame, year: int = 2024) -> pd.DataFrame:
    """
    Run all Cosilico tax calculations on tax unit data.

    This is where POLICY is executed. The tax_unit_builder provides only
    raw data; all calculations per statute belong here.

    Output column names match statute variable definitions in cosilico-us.

    Args:
        df: Tax unit DataFrame from tax_unit_builder (raw data only)
        year: Tax year

    Returns:
        DataFrame with calculated tax variables added
    """
    params = PARAMS_2024

    df = df.copy()

    # ==========================================================================
    # FOUNDATION CALCULATIONS (required before tax calculations)
    # These implement the core statutory definitions
    # ==========================================================================

    # 26/2/b.rac::is_head_of_household - filing status determination
    df['is_head_of_household'] = calculate_is_head_of_household(df)

    # 26/62.rac::adjusted_gross_income
    df['adjusted_gross_income'] = calculate_adjusted_gross_income(df, params)

    # 26/63/c.rac::standard_deduction
    df['standard_deduction'] = calculate_standard_deduction_simple(df, params)

    # 26/63.rac::taxable_income
    df['taxable_income'] = calculate_taxable_income(df, params)

    # ==========================================================================
    # TAX CALCULATIONS
    # ==========================================================================

    # 26/1.rac::income_tax_before_credits - must calculate first for CTC
    df['income_tax_before_credits'] = calculate_income_tax(df, params)

    # 26/32.rac::eitc
    df['eitc'] = calculate_eitc(df, params)

    # 26/24.rac::non_refundable_ctc, refundable_ctc, total_child_tax_credit
    # Non-refundable CTC limited by tax liability
    df['non_refundable_ctc'], df['refundable_ctc'] = calculate_ctc(
        df, params, df['income_tax_before_credits'].values
    )
    df['total_child_tax_credit'] = df['non_refundable_ctc'] + df['refundable_ctc']

    # 26/1.rac::income_tax (after credits)
    df['income_tax'] = np.maximum(0,
        df['income_tax_before_credits']
        - df['non_refundable_ctc']
        - df['eitc']  # EITC is refundable but also offsets tax
    )

    # 26/1401/a.rac::self_employment_tax
    df['self_employment_tax'] = calculate_se_tax(df, params)

    # 26/86.rac::taxable_social_security
    df['taxable_social_security'] = calculate_taxable_ss_for_agi(df, params)

    # 26/1411/a.rac::niit
    df['niit'] = calculate_niit(df, params)

    return df


if __name__ == "__main__":
    from tax_unit_builder import load_and_build_tax_units

    print("Loading tax units...")
    df = load_and_build_tax_units(2024)

    print("Running Cosilico calculations...")
    df = run_all_calculations(df)

    print("\n=== Cosilico Calculation Results ===")
    print(f"Tax units: {len(df):,}")

    # Weighted totals
    def wtotal(col):
        return (df[col] * df['weight']).sum()

    print("\n--- Weighted Totals ---")
    print(f"EITC:           ${wtotal('eitc'):>20,.0f}")
    print(f"CTC Total:      ${wtotal('total_child_tax_credit'):>20,.0f}")
    print(f"  Nonrefundable:${wtotal('non_refundable_ctc'):>20,.0f}")
    print(f"  Refundable:   ${wtotal('refundable_ctc'):>20,.0f}")
    print(f"SE Tax:         ${wtotal('self_employment_tax'):>20,.0f}")
    print(f"Tax Before Cred:${wtotal('income_tax_before_credits'):>20,.0f}")
    print(f"Income Tax:     ${wtotal('income_tax'):>20,.0f}")
    print(f"Taxable SS:     ${wtotal('taxable_social_security'):>20,.0f}")
    print(f"NIIT:           ${wtotal('niit'):>20,.0f}")

    # Compare to CPS reported values
    print("\n--- CPS Reported vs Cosilico ---")
    print(f"EITC: CPS=${wtotal('cps_eitc'):,.0f}, Cosilico=${wtotal('eitc'):,.0f}")
    print(f"CTC:  CPS=${wtotal('cps_ctc'):,.0f}, Cosilico=${wtotal('total_child_tax_credit'):,.0f}")

    # Recipients
    print("\n--- Recipients (unweighted) ---")
    print(f"EITC recipients: {(df['eitc'] > 0).sum():,}")
    print(f"CTC recipients:  {(df['total_child_tax_credit'] > 0).sum():,}")
    print(f"SE Tax payers:   {(df['self_employment_tax'] > 0).sum():,}")
