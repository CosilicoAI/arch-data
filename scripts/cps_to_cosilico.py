#!/usr/bin/env python3
"""
CPS to Cosilico Converter

Converts CPS ASEC microdata to the Cosilico input format for microsimulation.
Maps CPS survey variables to statute input variables used by cosilico-us.

Usage:
    python cps_to_cosilico.py --year 2024 --output microsim_2024.parquet
    python cps_to_cosilico.py --year 2024 --calibrate --output microsim_2024_calibrated.parquet

Output format:
    - Person-level records with unique IDs
    - All income components mapped to statute variables
    - Weights (original and optionally calibrated)
    - Relationship data for qualifying child determination
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


# =============================================================================
# CPS Variable to Cosilico Statute Variable Mappings
# =============================================================================

# Income variables: CPS variable -> Cosilico statute variable
INCOME_MAPPINGS = {
    # Earned income (IRC 32(c)(2))
    "WSAL_VAL": "wages",
    "SEMP_VAL": "self_employment_income",
    "FRSE_VAL": "farm_self_employment_income",

    # Investment income (IRC 61(a)(4-7))
    "INT_VAL": "interest_income",
    "DIV_VAL": "dividend_income",
    "RNT_VAL": "rental_income",
    "CAP_VAL": "capital_gains",

    # Retirement income (IRC 402, 408, 86)
    "SS_VAL": "social_security_benefits",
    "PNSN_VAL": "pension_income",
    "ANN_VAL": "annuity_income",

    # Transfer income
    "SSI_VAL": "ssi_income",
    "UC_VAL": "unemployment_compensation",
    "VET_VAL": "veterans_benefits",
    "WC_VAL": "workers_compensation",
    "PAW_VAL": "public_assistance_income",
    "DIS_VAL1": "disability_income_1",
    "DIS_VAL2": "disability_income_2",

    # Other income
    "CSP_VAL": "child_support_received",
    "CHSP_VAL": "child_support_paid",
    "ALM_VAL": "alimony_received",
    "OI_VAL": "other_income",
}

# Tax unit level mappings
TAX_UNIT_MAPPINGS = {
    "AGI": "adjusted_gross_income",
    "TAX_INC": "taxable_income",
    "FEDTAX_AC": "federal_income_tax",
    "FEDTAX_BC": "federal_income_tax_before_credits",
    "EIT_CRED": "eitc",
    "CTC_CRD": "child_tax_credit",
    "ACTC_CRD": "additional_child_tax_credit",
    "FICA": "fica_tax",
    "STATETAX_A": "state_income_tax",
    "STATETAX_B": "state_income_tax_before_credits",
    "MARG_TAX": "marginal_tax_rate",
}

# Demographic mappings
DEMOGRAPHIC_MAPPINGS = {
    "A_AGE": "age",
    "A_SEX": "is_female",  # Transformed: 2 -> True
    "A_MARITL": "marital_status",
    "PRDTRACE": "race",
    "PRDTHSP": "is_hispanic",
    "PRCITSHP": "citizenship_status",
    "A_HSCOL": "education_level",
}

# Relationship mappings
RELATIONSHIP_MAPPINGS = {
    "A_FAMREL": "family_relationship",
    "PARENT": "parent_line_number",
    "A_LINENO": "line_number",
}

# Identifier mappings
ID_MAPPINGS = {
    "PH_SEQ": "household_id",
    "P_SEQ": "person_sequence",
    "TAX_ID": "tax_unit_id",
    "SPM_ID": "spm_unit_id",
    "GESTFIPS": "state_fips",
}

# Weight mappings
WEIGHT_MAPPINGS = {
    "MARSUPWT": "weight",
    "A_FNLWGT": "person_weight",
    "SPM_WEIGHT": "spm_weight",
}


# =============================================================================
# Core Conversion Functions
# =============================================================================

def load_cps_from_cache(year: int, cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load CPS data from cached parquet files."""
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "micro" / "us" / "census" / "cps-asec" / "raw_cache" / f"census_cps_{year}"

    person_path = cache_dir / "person.parquet"
    household_path = cache_dir / "household.parquet"

    if not person_path.exists():
        raise FileNotFoundError(
            f"CPS {year} not found at {person_path}. "
            f"Run: python micro/us/census/download_cps.py --year {year}"
        )

    print(f"Loading CPS {year} from {cache_dir}")
    person = pd.read_parquet(person_path)
    print(f"  Loaded {len(person):,} person records")

    # Merge household data for geography
    if household_path.exists():
        household = pd.read_parquet(
            household_path,
            columns=["H_SEQ", "GESTFIPS", "GEDIV", "GEREG"]
        )
        person = person.merge(
            household.rename(columns={"H_SEQ": "PH_SEQ"}),
            on="PH_SEQ",
            how="left"
        )

    return person


def transform_income_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Transform CPS income variables to Cosilico format."""
    result = pd.DataFrame(index=df.index)

    for cps_var, cos_var in INCOME_MAPPINGS.items():
        if cps_var in df.columns:
            result[cos_var] = df[cps_var].fillna(0).astype(float)
        else:
            # Variable not in data, set to 0
            result[cos_var] = 0.0

    # Derived income totals
    earned_income_cols = ["wages", "self_employment_income", "farm_self_employment_income"]
    result["earned_income"] = result[earned_income_cols].sum(axis=1)

    investment_income_cols = ["interest_income", "dividend_income", "capital_gains"]
    result["investment_income"] = result[investment_income_cols].sum(axis=1)

    # Total income (all sources)
    all_income_cols = [c for c in result.columns if c.endswith("_income") or c in ["wages", "capital_gains"]]
    result["total_income"] = result[all_income_cols].sum(axis=1)

    return result


def transform_tax_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Transform CPS tax unit variables to Cosilico format."""
    result = pd.DataFrame(index=df.index)

    for cps_var, cos_var in TAX_UNIT_MAPPINGS.items():
        if cps_var in df.columns:
            result[cos_var] = df[cps_var].fillna(0).astype(float)
        else:
            result[cos_var] = 0.0

    return result


def transform_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Transform CPS demographic variables to Cosilico format."""
    result = pd.DataFrame(index=df.index)

    for cps_var, cos_var in DEMOGRAPHIC_MAPPINGS.items():
        if cps_var in df.columns:
            result[cos_var] = df[cps_var].fillna(0)

    # Transform sex to is_female boolean
    if "is_female" in result.columns:
        result["is_female"] = result["is_female"] == 2

    # Transform Hispanic origin
    if "is_hispanic" in result.columns:
        result["is_hispanic"] = result["is_hispanic"] > 1  # 1 = not Hispanic

    return result


def transform_relationships(df: pd.DataFrame) -> pd.DataFrame:
    """Transform CPS relationship variables for qualifying child determination."""
    result = pd.DataFrame(index=df.index)

    for cps_var, cos_var in RELATIONSHIP_MAPPINGS.items():
        if cps_var in df.columns:
            result[cos_var] = df[cps_var].fillna(0).astype(int)

    return result


def transform_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Transform CPS identifier variables."""
    result = pd.DataFrame(index=df.index)

    for cps_var, cos_var in ID_MAPPINGS.items():
        if cps_var in df.columns:
            result[cos_var] = df[cps_var].fillna(0)

    # Create unique person ID
    if "household_id" in result.columns and "person_sequence" in result.columns:
        result["person_id"] = (
            result["household_id"].astype(str) + "_" +
            result["person_sequence"].astype(str)
        )

    return result


def transform_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Transform CPS weight variables (divide by 100)."""
    result = pd.DataFrame(index=df.index)

    for cps_var, cos_var in WEIGHT_MAPPINGS.items():
        if cps_var in df.columns:
            # CPS weights have 2 implied decimal places
            result[cos_var] = df[cps_var].fillna(0) / 100

    # Ensure primary weight exists
    if "weight" not in result.columns:
        if "person_weight" in result.columns:
            result["weight"] = result["person_weight"]
        else:
            result["weight"] = 1.0

    return result


def convert_cps_to_cosilico(
    year: int,
    output_path: Optional[Path] = None,
    calibrate: bool = False,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Convert CPS microdata to Cosilico input format.

    Args:
        year: Tax year (e.g., 2024)
        output_path: Path to save output parquet
        calibrate: Apply entropy calibration to IRS SOI targets
        cache_dir: Override default cache directory

    Returns:
        DataFrame in Cosilico input format
    """
    # Load raw CPS data
    cps = load_cps_from_cache(year, cache_dir)

    print("Transforming to Cosilico format...")

    # Transform all variable groups
    income = transform_income_variables(cps)
    tax = transform_tax_variables(cps)
    demographics = transform_demographics(cps)
    relationships = transform_relationships(cps)
    identifiers = transform_identifiers(cps)
    weights = transform_weights(cps)

    # Combine all transformations
    result = pd.concat([
        identifiers,
        weights,
        demographics,
        relationships,
        income,
        tax,
    ], axis=1)

    # Filter to positive weights
    result = result[result["weight"] > 0].copy()
    print(f"  {len(result):,} records with positive weights")

    # Apply calibration if requested
    if calibrate:
        result = apply_calibration(result, year)

    # Add metadata columns
    result["year"] = year
    result["source"] = "cps-asec"

    # Save output
    if output_path is None:
        output_dir = Path(__file__).parent.parent / "micro" / "us"
        output_path = output_dir / f"cosilico_input_{year}.parquet"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result.to_parquet(output_path, index=False)
    print(f"Saved {len(result):,} records to {output_path}")

    return result


def apply_calibration(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Apply entropy calibration to match IRS SOI targets."""
    try:
        from micro.us.calibrate import calibrate_weights
        print("Applying entropy calibration to IRS SOI targets...")

        # Calibration expects specific column names
        df_cal = df.copy()
        if "adjusted_gross_income" not in df_cal.columns:
            print("  Warning: AGI not found, skipping calibration")
            return df

        result = calibrate_weights(df_cal, verbose=True)

        if result.success:
            df["original_weight"] = df["weight"]
            df["weight"] = result.calibrated_weights
            df["weight_adjustment"] = result.adjustment_factors
            print(f"  Calibration successful (KL div: {result.kl_divergence:.2f})")
        else:
            print(f"  Calibration failed: {result.message}")

        return df

    except ImportError:
        print("  Warning: Calibration module not found, skipping")
        return df


def generate_summary(df: pd.DataFrame, year: int) -> dict:
    """Generate summary statistics for validation."""
    weight = df["weight"].values

    summary = {
        "year": year,
        "record_count": len(df),
        "weighted_population": weight.sum(),
        "totals": {},
    }

    # Income totals
    income_vars = [
        "wages", "self_employment_income", "interest_income",
        "dividend_income", "social_security_benefits", "unemployment_compensation",
    ]

    for var in income_vars:
        if var in df.columns:
            total = (df[var] * weight).sum()
            summary["totals"][var] = total

    # Tax totals
    tax_vars = ["eitc", "child_tax_credit", "federal_income_tax"]
    for var in tax_vars:
        if var in df.columns:
            total = (df[var] * weight).sum()
            summary["totals"][var] = total

    return summary


def print_summary(summary: dict) -> None:
    """Print formatted summary statistics."""
    print("\n" + "=" * 60)
    print(f"COSILICO INPUT SUMMARY - {summary['year']}")
    print("=" * 60)
    print(f"Records: {summary['record_count']:,}")
    print(f"Weighted population: {summary['weighted_population']:,.0f}")
    print("\nIncome Totals:")
    for var, total in summary["totals"].items():
        print(f"  {var}: ${total:,.0f}")


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert CPS ASEC microdata to Cosilico input format"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Tax year (default: 2024)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output parquet path"
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Apply entropy calibration to IRS SOI targets"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Override raw CPS cache directory"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics"
    )

    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    df = convert_cps_to_cosilico(
        year=args.year,
        output_path=output_path,
        calibrate=args.calibrate,
        cache_dir=cache_dir,
    )

    if args.summary:
        summary = generate_summary(df, args.year)
        print_summary(summary)


if __name__ == "__main__":
    main()
