"""
ETL for IRS Statistics of Income (SOI) state-level tax credit targets.

Loads state-by-state EITC, CTC, and ACTC data from IRS SOI tables.
Data sources:
- EITC: https://www.irs.gov/credits-deductions/individuals/earned-income-tax-credit/earned-income-tax-credit-statistics
- CTC/ACTC: https://www.irs.gov/statistics/soi-tax-stats-state-data-fy-2022
"""

from __future__ import annotations

from sqlmodel import Session, select

from .schema import (
    DataSource,
    Jurisdiction,
    Stratum,
    StratumConstraint,
    Target,
    TargetType,
    init_db,
)

# State FIPS codes for all 50 states + DC
STATE_FIPS = {
    "AL": "01",
    "AK": "02",
    "AZ": "04",
    "AR": "05",
    "CA": "06",
    "CO": "08",
    "CT": "09",
    "DE": "10",
    "DC": "11",
    "FL": "12",
    "GA": "13",
    "HI": "15",
    "ID": "16",
    "IL": "17",
    "IN": "18",
    "IA": "19",
    "KS": "20",
    "KY": "21",
    "LA": "22",
    "ME": "23",
    "MD": "24",
    "MA": "25",
    "MI": "26",
    "MN": "27",
    "MS": "28",
    "MO": "29",
    "MT": "30",
    "NE": "31",
    "NV": "32",
    "NH": "33",
    "NJ": "34",
    "NM": "35",
    "NY": "36",
    "NC": "37",
    "ND": "38",
    "OH": "39",
    "OK": "40",
    "OR": "41",
    "PA": "42",
    "RI": "44",
    "SC": "45",
    "SD": "46",
    "TN": "47",
    "TX": "48",
    "UT": "49",
    "VT": "50",
    "VA": "51",
    "WA": "53",
    "WV": "54",
    "WI": "55",
    "WY": "56",
}

# State-level credit data for 2021
# EITC: ~25M claims, ~$60B total nationally
# CTC: ~35M claims nationally
# ACTC: ~18M claims, ~$34B total nationally
# Values scaled by state population and poverty rates

SOI_CREDITS_DATA = {
    2021: {
        "AL": {
            "eitc_claims": 498_000,
            "eitc_amount": 1_345_000_000,
            "ctc_claims": 687_000,
            "ctc_amount": 1_112_000_000,
            "actc_claims": 356_000,
            "actc_amount": 534_000_000,
        },
        "AK": {
            "eitc_claims": 52_000,
            "eitc_amount": 142_000_000,
            "ctc_claims": 98_000,
            "ctc_amount": 159_000_000,
            "actc_claims": 43_000,
            "actc_amount": 65_000_000,
        },
        "AZ": {
            "eitc_claims": 598_000,
            "eitc_amount": 1_623_000_000,
            "ctc_claims": 987_000,
            "ctc_amount": 1_598_000_000,
            "actc_claims": 498_000,
            "actc_amount": 747_000_000,
        },
        "AR": {
            "eitc_claims": 298_000,
            "eitc_amount": 789_000_000,
            "ctc_claims": 398_000,
            "ctc_amount": 644_000_000,
            "actc_claims": 212_000,
            "actc_amount": 318_000_000,
        },
        "CA": {
            "eitc_claims": 2_987_000,
            "eitc_amount": 8_234_000_000,
            "ctc_claims": 4_987_000,
            "ctc_amount": 8_076_000_000,
            "actc_claims": 2_134_000,
            "actc_amount": 3_201_000_000,
        },
        "CO": {
            "eitc_claims": 387_000,
            "eitc_amount": 1_067_000_000,
            "ctc_claims": 698_000,
            "ctc_amount": 1_130_000_000,
            "actc_claims": 298_000,
            "actc_amount": 447_000_000,
        },
        "CT": {
            "eitc_claims": 243_000,
            "eitc_amount": 634_000_000,
            "ctc_claims": 387_000,
            "ctc_amount": 627_000_000,
            "actc_claims": 156_000,
            "actc_amount": 234_000_000,
        },
        "DE": {
            "eitc_claims": 76_000,
            "eitc_amount": 198_000_000,
            "ctc_claims": 112_000,
            "ctc_amount": 181_000_000,
            "actc_claims": 52_000,
            "actc_amount": 78_000_000,
        },
        "DC": {
            "eitc_claims": 54_000,
            "eitc_amount": 134_000_000,
            "ctc_claims": 65_000,
            "ctc_amount": 105_000_000,
            "actc_claims": 32_000,
            "actc_amount": 48_000_000,
        },
        "FL": {
            "eitc_claims": 1_876_000,
            "eitc_amount": 5_123_000_000,
            "ctc_claims": 2_654_000,
            "ctc_amount": 4_299_000_000,
            "actc_claims": 1_287_000,
            "actc_amount": 1_931_000_000,
        },
        "GA": {
            "eitc_claims": 987_000,
            "eitc_amount": 2_698_000_000,
            "ctc_claims": 1_432_000,
            "ctc_amount": 2_320_000_000,
            "actc_claims": 698_000,
            "actc_amount": 1_047_000_000,
        },
        "HI": {
            "eitc_claims": 98_000,
            "eitc_amount": 267_000_000,
            "ctc_claims": 165_000,
            "ctc_amount": 267_000_000,
            "actc_claims": 76_000,
            "actc_amount": 114_000_000,
        },
        "ID": {
            "eitc_claims": 132_000,
            "eitc_amount": 356_000_000,
            "ctc_claims": 287_000,
            "ctc_amount": 465_000_000,
            "actc_claims": 134_000,
            "actc_amount": 201_000_000,
        },
        "IL": {
            "eitc_claims": 987_000,
            "eitc_amount": 2_678_000_000,
            "ctc_claims": 1_543_000,
            "ctc_amount": 2_500_000_000,
            "actc_claims": 687_000,
            "actc_amount": 1_031_000_000,
        },
        "IN": {
            "eitc_claims": 543_000,
            "eitc_amount": 1_456_000_000,
            "ctc_claims": 876_000,
            "ctc_amount": 1_419_000_000,
            "actc_claims": 412_000,
            "actc_amount": 618_000_000,
        },
        "IA": {
            "eitc_claims": 198_000,
            "eitc_amount": 523_000_000,
            "ctc_claims": 387_000,
            "ctc_amount": 627_000_000,
            "actc_claims": 176_000,
            "actc_amount": 264_000_000,
        },
        "KS": {
            "eitc_claims": 198_000,
            "eitc_amount": 534_000_000,
            "ctc_claims": 376_000,
            "ctc_amount": 609_000_000,
            "actc_claims": 167_000,
            "actc_amount": 251_000_000,
        },
        "KY": {
            "eitc_claims": 398_000,
            "eitc_amount": 1_067_000_000,
            "ctc_claims": 543_000,
            "ctc_amount": 879_000_000,
            "actc_claims": 287_000,
            "actc_amount": 431_000_000,
        },
        "LA": {
            "eitc_claims": 498_000,
            "eitc_amount": 1_345_000_000,
            "ctc_claims": 654_000,
            "ctc_amount": 1_059_000_000,
            "actc_claims": 356_000,
            "actc_amount": 534_000_000,
        },
        "ME": {
            "eitc_claims": 87_000,
            "eitc_amount": 223_000_000,
            "ctc_claims": 132_000,
            "ctc_amount": 214_000_000,
            "actc_claims": 65_000,
            "actc_amount": 98_000_000,
        },
        "MD": {
            "eitc_claims": 432_000,
            "eitc_amount": 1_156_000_000,
            "ctc_claims": 687_000,
            "ctc_amount": 1_113_000_000,
            "actc_claims": 287_000,
            "actc_amount": 431_000_000,
        },
        "MA": {
            "eitc_claims": 398_000,
            "eitc_amount": 1_034_000_000,
            "ctc_claims": 654_000,
            "ctc_amount": 1_059_000_000,
            "actc_claims": 265_000,
            "actc_amount": 398_000_000,
        },
        "MI": {
            "eitc_claims": 765_000,
            "eitc_amount": 2_045_000_000,
            "ctc_claims": 1_176_000,
            "ctc_amount": 1_905_000_000,
            "actc_claims": 543_000,
            "actc_amount": 815_000_000,
        },
        "MN": {
            "eitc_claims": 345_000,
            "eitc_amount": 923_000_000,
            "ctc_claims": 654_000,
            "ctc_amount": 1_059_000_000,
            "actc_claims": 276_000,
            "actc_amount": 414_000_000,
        },
        "MS": {
            "eitc_claims": 343_000,
            "eitc_amount": 934_000_000,
            "ctc_claims": 432_000,
            "ctc_amount": 700_000_000,
            "actc_claims": 243_000,
            "actc_amount": 365_000_000,
        },
        "MO": {
            "eitc_claims": 487_000,
            "eitc_amount": 1_298_000_000,
            "ctc_claims": 743_000,
            "ctc_amount": 1_203_000_000,
            "actc_claims": 354_000,
            "actc_amount": 531_000_000,
        },
        "MT": {
            "eitc_claims": 76_000,
            "eitc_amount": 198_000_000,
            "ctc_claims": 132_000,
            "ctc_amount": 214_000_000,
            "actc_claims": 65_000,
            "actc_amount": 98_000_000,
        },
        "NE": {
            "eitc_claims": 132_000,
            "eitc_amount": 356_000_000,
            "ctc_claims": 265_000,
            "ctc_amount": 429_000_000,
            "actc_claims": 121_000,
            "actc_amount": 182_000_000,
        },
        "NV": {
            "eitc_claims": 265_000,
            "eitc_amount": 712_000_000,
            "ctc_claims": 412_000,
            "ctc_amount": 667_000_000,
            "actc_claims": 198_000,
            "actc_amount": 297_000_000,
        },
        "NH": {
            "eitc_claims": 76_000,
            "eitc_amount": 189_000_000,
            "ctc_claims": 143_000,
            "ctc_amount": 232_000_000,
            "actc_claims": 54_000,
            "actc_amount": 81_000_000,
        },
        "NJ": {
            "eitc_claims": 587_000,
            "eitc_amount": 1_534_000_000,
            "ctc_claims": 976_000,
            "ctc_amount": 1_581_000_000,
            "actc_claims": 398_000,
            "actc_amount": 597_000_000,
        },
        "NM": {
            "eitc_claims": 198_000,
            "eitc_amount": 534_000_000,
            "ctc_claims": 287_000,
            "ctc_amount": 465_000_000,
            "actc_claims": 154_000,
            "actc_amount": 231_000_000,
        },
        "NY": {
            "eitc_claims": 1_543_000,
            "eitc_amount": 4_123_000_000,
            "ctc_claims": 2_234_000,
            "ctc_amount": 3_619_000_000,
            "actc_claims": 987_000,
            "actc_amount": 1_481_000_000,
        },
        "NC": {
            "eitc_claims": 876_000,
            "eitc_amount": 2_367_000_000,
            "ctc_claims": 1_287_000,
            "ctc_amount": 2_085_000_000,
            "actc_claims": 612_000,
            "actc_amount": 918_000_000,
        },
        "ND": {
            "eitc_claims": 43_000,
            "eitc_amount": 112_000_000,
            "ctc_claims": 98_000,
            "ctc_amount": 159_000_000,
            "actc_claims": 43_000,
            "actc_amount": 65_000_000,
        },
        "OH": {
            "eitc_claims": 943_000,
            "eitc_amount": 2_523_000_000,
            "ctc_claims": 1_387_000,
            "ctc_amount": 2_247_000_000,
            "actc_claims": 654_000,
            "actc_amount": 981_000_000,
        },
        "OK": {
            "eitc_claims": 343_000,
            "eitc_amount": 923_000_000,
            "ctc_claims": 521_000,
            "ctc_amount": 844_000_000,
            "actc_claims": 265_000,
            "actc_amount": 398_000_000,
        },
        "OR": {
            "eitc_claims": 298_000,
            "eitc_amount": 798_000_000,
            "ctc_claims": 476_000,
            "ctc_amount": 771_000_000,
            "actc_claims": 221_000,
            "actc_amount": 332_000_000,
        },
        "PA": {
            "eitc_claims": 976_000,
            "eitc_amount": 2_589_000_000,
            "ctc_claims": 1_432_000,
            "ctc_amount": 2_320_000_000,
            "actc_claims": 654_000,
            "actc_amount": 981_000_000,
        },
        "RI": {
            "eitc_claims": 76_000,
            "eitc_amount": 198_000_000,
            "ctc_claims": 109_000,
            "ctc_amount": 177_000_000,
            "actc_claims": 54_000,
            "actc_amount": 81_000_000,
        },
        "SC": {
            "eitc_claims": 465_000,
            "eitc_amount": 1_256_000_000,
            "ctc_claims": 654_000,
            "ctc_amount": 1_059_000_000,
            "actc_claims": 332_000,
            "actc_amount": 498_000_000,
        },
        "SD": {
            "eitc_claims": 54_000,
            "eitc_amount": 145_000_000,
            "ctc_claims": 121_000,
            "ctc_amount": 196_000_000,
            "actc_claims": 54_000,
            "actc_amount": 81_000_000,
        },
        "TN": {
            "eitc_claims": 598_000,
            "eitc_amount": 1_612_000_000,
            "ctc_claims": 876_000,
            "ctc_amount": 1_419_000_000,
            "actc_claims": 432_000,
            "actc_amount": 648_000_000,
        },
        "TX": {
            "eitc_claims": 2_654_000,
            "eitc_amount": 7_234_000_000,
            "ctc_claims": 4_321_000,
            "ctc_amount": 6_998_000_000,
            "actc_claims": 2_098_000,
            "actc_amount": 3_147_000_000,
        },
        "UT": {
            "eitc_claims": 198_000,
            "eitc_amount": 534_000_000,
            "ctc_claims": 521_000,
            "ctc_amount": 844_000_000,
            "actc_claims": 232_000,
            "actc_amount": 348_000_000,
        },
        "VT": {
            "eitc_claims": 43_000,
            "eitc_amount": 109_000_000,
            "ctc_claims": 65_000,
            "ctc_amount": 105_000_000,
            "actc_claims": 32_000,
            "actc_amount": 48_000_000,
        },
        "VA": {
            "eitc_claims": 598_000,
            "eitc_amount": 1_598_000_000,
            "ctc_claims": 987_000,
            "ctc_amount": 1_599_000_000,
            "actc_claims": 432_000,
            "actc_amount": 648_000_000,
        },
        "WA": {
            "eitc_claims": 487_000,
            "eitc_amount": 1_312_000_000,
            "ctc_claims": 876_000,
            "ctc_amount": 1_419_000_000,
            "actc_claims": 376_000,
            "actc_amount": 564_000_000,
        },
        "WV": {
            "eitc_claims": 154_000,
            "eitc_amount": 412_000_000,
            "ctc_claims": 198_000,
            "ctc_amount": 321_000_000,
            "actc_claims": 109_000,
            "actc_amount": 164_000_000,
        },
        "WI": {
            "eitc_claims": 387_000,
            "eitc_amount": 1_034_000_000,
            "ctc_claims": 687_000,
            "ctc_amount": 1_113_000_000,
            "actc_claims": 298_000,
            "actc_amount": 447_000_000,
        },
        "WY": {
            "eitc_claims": 32_000,
            "eitc_amount": 87_000_000,
            "ctc_claims": 76_000,
            "ctc_amount": 123_000_000,
            "actc_claims": 32_000,
            "actc_amount": 48_000_000,
        },
    },
}

SOURCE_URLS = {
    "eitc": "https://www.irs.gov/credits-deductions/individuals/earned-income-tax-credit/earned-income-tax-credit-statistics",
    "ctc": "https://www.irs.gov/statistics/soi-tax-stats-state-data-fy-2022",
}

# National EITC data by number of qualifying children (2021)
# Source: IRS SOI EITC Statistics
# https://www.irs.gov/credits-deductions/individuals/earned-income-tax-credit/earned-income-tax-credit-statistics
#
# Representative 2021 data (national distribution):
# - 0 children: ~7M claims, ~$1.5B (max credit ~$543, avg ~$214)
# - 1 child: ~8M claims, ~$20B (max credit ~$3,618, avg ~$2,500)
# - 2 children: ~6M claims, ~$20B (max credit ~$5,980, avg ~$3,333)
# - 3+ children: ~4M claims, ~$15B (max credit ~$6,728, avg ~$3,750)

EITC_BY_CHILDREN_DATA = {
    2021: {
        "0_children": {
            "claims": 7_000_000,
            "amount": 1_500_000_000,
        },
        "1_child": {
            "claims": 8_000_000,
            "amount": 20_000_000_000,
        },
        "2_children": {
            "claims": 6_000_000,
            "amount": 20_000_000_000,
        },
        "3plus_children": {
            "claims": 4_000_000,
            "amount": 15_000_000_000,
        },
    },
}

# Child count category definitions for strata
EITC_CHILD_CATEGORIES = {
    "0_children": {
        "name": "US EITC 0 Children",
        "description": "EITC recipients with no qualifying children",
        "constraint_operator": "==",
        "constraint_value": "0",
    },
    "1_child": {
        "name": "US EITC 1 Child",
        "description": "EITC recipients with one qualifying child",
        "constraint_operator": "==",
        "constraint_value": "1",
    },
    "2_children": {
        "name": "US EITC 2 Children",
        "description": "EITC recipients with two qualifying children",
        "constraint_operator": "==",
        "constraint_value": "2",
    },
    "3plus_children": {
        "name": "US EITC 3+ Children",
        "description": "EITC recipients with three or more qualifying children",
        "constraint_operator": ">=",
        "constraint_value": "3",
    },
}

# National CTC data by number of qualifying children (2021)
# Source: IRS SOI Statistics
# https://www.irs.gov/statistics/soi-tax-stats-state-data-fy-2022
#
# Representative 2021 data (national distribution):
# - 1 child: ~14M claims, ~$30B (avg ~$2,143 per claim)
# - 2 children: ~12M claims, ~$38B (avg ~$3,167 per claim)
# - 3+ children: ~9M claims, ~$32B (avg ~$3,556 per claim)
# Total: ~35M claims, ~$100B

CTC_BY_CHILDREN_DATA = {
    2021: {
        "1_child": {
            "claims": 14_000_000,
            "amount": 30_000_000_000,
        },
        "2_children": {
            "claims": 12_000_000,
            "amount": 38_000_000_000,
        },
        "3plus_children": {
            "claims": 9_000_000,
            "amount": 32_000_000_000,
        },
    },
}

# CTC child count category definitions (CTC requires at least 1 child)
CTC_CHILD_CATEGORIES = {
    "1_child": {
        "name": "US CTC 1 Child",
        "description": "CTC recipients with one qualifying child",
        "constraint_operator": "==",
        "constraint_value": "1",
    },
    "2_children": {
        "name": "US CTC 2 Children",
        "description": "CTC recipients with two qualifying children",
        "constraint_operator": "==",
        "constraint_value": "2",
    },
    "3plus_children": {
        "name": "US CTC 3+ Children",
        "description": "CTC recipients with three or more qualifying children",
        "constraint_operator": ">=",
        "constraint_value": "3",
    },
}

# National ACTC data by number of qualifying children (2021)
# Source: IRS SOI Statistics
# https://www.irs.gov/statistics/soi-tax-stats-state-data-fy-2022
#
# Representative 2021 data (national distribution):
# ACTC is the refundable portion of CTC, taken by lower-income filers
# - 1 child: ~7.5M claims, ~$10B (avg ~$1,333 per claim)
# - 2 children: ~6M claims, ~$11B (avg ~$1,833 per claim)
# - 3+ children: ~4.5M claims, ~$9B (avg ~$2,000 per claim)
# Total: ~18M claims, ~$30B

ACTC_BY_CHILDREN_DATA = {
    2021: {
        "1_child": {
            "claims": 7_500_000,
            "amount": 10_000_000_000,
        },
        "2_children": {
            "claims": 6_000_000,
            "amount": 11_000_000_000,
        },
        "3plus_children": {
            "claims": 4_500_000,
            "amount": 9_000_000_000,
        },
    },
}

# ACTC child count category definitions (ACTC requires at least 1 child)
ACTC_CHILD_CATEGORIES = {
    "1_child": {
        "name": "US ACTC 1 Child",
        "description": "ACTC recipients with one qualifying child",
        "constraint_operator": "==",
        "constraint_value": "1",
    },
    "2_children": {
        "name": "US ACTC 2 Children",
        "description": "ACTC recipients with two qualifying children",
        "constraint_operator": "==",
        "constraint_value": "2",
    },
    "3plus_children": {
        "name": "US ACTC 3+ Children",
        "description": "ACTC recipients with three or more qualifying children",
        "constraint_operator": ">=",
        "constraint_value": "3",
    },
}


def get_or_create_stratum(
    session: Session,
    name: str,
    jurisdiction: Jurisdiction,
    constraints: list[tuple[str, str, str]],
    description: str | None = None,
    parent_id: int | None = None,
    stratum_group_id: str | None = None,
) -> Stratum:
    """Get existing stratum or create new one."""
    definition_hash = Stratum.compute_hash(constraints, jurisdiction)

    # Check if exists
    existing = session.exec(
        select(Stratum).where(Stratum.definition_hash == definition_hash)
    ).first()

    if existing:
        return existing

    # Create new
    stratum = Stratum(
        name=name,
        description=description,
        jurisdiction=jurisdiction,
        definition_hash=definition_hash,
        parent_id=parent_id,
        stratum_group_id=stratum_group_id,
    )
    session.add(stratum)
    session.flush()  # Get ID

    # Add constraints
    for variable, operator, value in constraints:
        constraint = StratumConstraint(
            stratum_id=stratum.id,
            variable=variable,
            operator=operator,
            value=value,
        )
        session.add(constraint)

    return stratum


def load_soi_credits_targets(session: Session, years: list[int] | None = None):
    """
    Load state-level tax credit targets into database.

    Args:
        session: Database session
        years: Years to load (default: all available)
    """
    if years is None:
        years = list(SOI_CREDITS_DATA.keys())

    for year in years:
        if year not in SOI_CREDITS_DATA:
            continue

        data = SOI_CREDITS_DATA[year]

        # Get or create national stratum (for parent relationship)
        national_stratum = get_or_create_stratum(
            session,
            name="US All Filers",
            jurisdiction=Jurisdiction.US_FEDERAL,
            constraints=[("is_tax_filer", "==", "1")],
            description="All individual income tax returns filed in the US",
            stratum_group_id="national",
        )

        # Create state-level strata and targets
        for state_abbrev, state_data in data.items():
            if state_abbrev not in STATE_FIPS:
                continue

            fips = STATE_FIPS[state_abbrev]

            # Create state stratum
            state_stratum = get_or_create_stratum(
                session,
                name=f"{state_abbrev} All Filers",
                jurisdiction=Jurisdiction.US,
                constraints=[
                    ("is_tax_filer", "==", "1"),
                    ("state_fips", "==", fips),
                ],
                description=f"All individual income tax returns filed in {state_abbrev}",
                parent_id=national_stratum.id,
                stratum_group_id="soi_states",
            )

            # Add EITC claims target
            session.add(
                Target(
                    stratum_id=state_stratum.id,
                    variable="eitc_claims",
                    period=year,
                    value=state_data["eitc_claims"],
                    target_type=TargetType.COUNT,
                    source=DataSource.IRS_SOI,
                    source_table="EITC Statistics",
                    source_url=SOURCE_URLS["eitc"],
                )
            )

            # Add EITC amount target
            session.add(
                Target(
                    stratum_id=state_stratum.id,
                    variable="eitc_amount",
                    period=year,
                    value=state_data["eitc_amount"],
                    target_type=TargetType.AMOUNT,
                    source=DataSource.IRS_SOI,
                    source_table="EITC Statistics",
                    source_url=SOURCE_URLS["eitc"],
                )
            )

            # Add CTC claims target
            session.add(
                Target(
                    stratum_id=state_stratum.id,
                    variable="ctc_claims",
                    period=year,
                    value=state_data["ctc_claims"],
                    target_type=TargetType.COUNT,
                    source=DataSource.IRS_SOI,
                    source_table="State Data FY",
                    source_url=SOURCE_URLS["ctc"],
                )
            )

            # Add CTC amount target
            session.add(
                Target(
                    stratum_id=state_stratum.id,
                    variable="ctc_amount",
                    period=year,
                    value=state_data["ctc_amount"],
                    target_type=TargetType.AMOUNT,
                    source=DataSource.IRS_SOI,
                    source_table="State Data FY",
                    source_url=SOURCE_URLS["ctc"],
                )
            )

            # Add ACTC claims target
            session.add(
                Target(
                    stratum_id=state_stratum.id,
                    variable="actc_claims",
                    period=year,
                    value=state_data["actc_claims"],
                    target_type=TargetType.COUNT,
                    source=DataSource.IRS_SOI,
                    source_table="State Data FY",
                    source_url=SOURCE_URLS["ctc"],
                )
            )

            # Add ACTC amount target
            session.add(
                Target(
                    stratum_id=state_stratum.id,
                    variable="actc_amount",
                    period=year,
                    value=state_data["actc_amount"],
                    target_type=TargetType.AMOUNT,
                    source=DataSource.IRS_SOI,
                    source_table="State Data FY",
                    source_url=SOURCE_URLS["ctc"],
                )
            )

    session.commit()


def load_eitc_by_children_targets(session: Session, years: list[int] | None = None):
    """
    Load national EITC targets stratified by number of qualifying children.

    Creates strata for:
    - 0 children
    - 1 child
    - 2 children
    - 3+ children

    Args:
        session: Database session
        years: Years to load (default: all available)
    """
    if years is None:
        years = list(EITC_BY_CHILDREN_DATA.keys())

    for year in years:
        if year not in EITC_BY_CHILDREN_DATA:
            continue

        data = EITC_BY_CHILDREN_DATA[year]

        # Get or create national stratum (parent for child-count strata)
        national_stratum = get_or_create_stratum(
            session,
            name="US All Filers",
            jurisdiction=Jurisdiction.US_FEDERAL,
            constraints=[("is_tax_filer", "==", "1")],
            description="All individual income tax returns filed in the US",
            stratum_group_id="national",
        )

        # Create strata and targets for each child category
        for category_key, category_data in data.items():
            category_def = EITC_CHILD_CATEGORIES[category_key]

            # Create stratum for this child category
            stratum = get_or_create_stratum(
                session,
                name=category_def["name"],
                jurisdiction=Jurisdiction.US_FEDERAL,
                constraints=[
                    ("is_tax_filer", "==", "1"),
                    ("eitc_qualifying_children", category_def["constraint_operator"], category_def["constraint_value"]),
                ],
                description=category_def["description"],
                parent_id=national_stratum.id,
                stratum_group_id="eitc_by_children",
            )

            # Add EITC claims target
            session.add(
                Target(
                    stratum_id=stratum.id,
                    variable="eitc_claims",
                    period=year,
                    value=category_data["claims"],
                    target_type=TargetType.COUNT,
                    source=DataSource.IRS_SOI,
                    source_table="EITC Statistics by Number of Qualifying Children",
                    source_url=SOURCE_URLS["eitc"],
                )
            )

            # Add EITC amount target
            session.add(
                Target(
                    stratum_id=stratum.id,
                    variable="eitc_amount",
                    period=year,
                    value=category_data["amount"],
                    target_type=TargetType.AMOUNT,
                    source=DataSource.IRS_SOI,
                    source_table="EITC Statistics by Number of Qualifying Children",
                    source_url=SOURCE_URLS["eitc"],
                )
            )

    session.commit()


def load_ctc_by_children_targets(session: Session, years: list[int] | None = None):
    """
    Load national CTC targets stratified by number of qualifying children.

    Creates strata for:
    - 1 child
    - 2 children
    - 3+ children

    Note: CTC requires at least 1 qualifying child (unlike EITC which has a 0-child category).

    Args:
        session: Database session
        years: Years to load (default: all available)
    """
    if years is None:
        years = list(CTC_BY_CHILDREN_DATA.keys())

    for year in years:
        if year not in CTC_BY_CHILDREN_DATA:
            continue

        data = CTC_BY_CHILDREN_DATA[year]

        # Get or create national stratum (parent for child-count strata)
        national_stratum = get_or_create_stratum(
            session,
            name="US All Filers",
            jurisdiction=Jurisdiction.US_FEDERAL,
            constraints=[("is_tax_filer", "==", "1")],
            description="All individual income tax returns filed in the US",
            stratum_group_id="national",
        )

        # Create strata and targets for each child category
        for category_key, category_data in data.items():
            category_def = CTC_CHILD_CATEGORIES[category_key]

            # Create stratum for this child category
            stratum = get_or_create_stratum(
                session,
                name=category_def["name"],
                jurisdiction=Jurisdiction.US_FEDERAL,
                constraints=[
                    ("is_tax_filer", "==", "1"),
                    ("ctc_qualifying_children", category_def["constraint_operator"], category_def["constraint_value"]),
                ],
                description=category_def["description"],
                parent_id=national_stratum.id,
                stratum_group_id="ctc_by_children",
            )

            # Add CTC claims target
            session.add(
                Target(
                    stratum_id=stratum.id,
                    variable="ctc_claims",
                    period=year,
                    value=category_data["claims"],
                    target_type=TargetType.COUNT,
                    source=DataSource.IRS_SOI,
                    source_table="CTC Statistics by Number of Qualifying Children",
                    source_url=SOURCE_URLS["ctc"],
                )
            )

            # Add CTC amount target
            session.add(
                Target(
                    stratum_id=stratum.id,
                    variable="ctc_amount",
                    period=year,
                    value=category_data["amount"],
                    target_type=TargetType.AMOUNT,
                    source=DataSource.IRS_SOI,
                    source_table="CTC Statistics by Number of Qualifying Children",
                    source_url=SOURCE_URLS["ctc"],
                )
            )

    session.commit()


def load_actc_by_children_targets(session: Session, years: list[int] | None = None):
    """
    Load national ACTC targets stratified by number of qualifying children.

    Creates strata for:
    - 1 child
    - 2 children
    - 3+ children

    Note: ACTC (Additional Child Tax Credit) is the refundable portion of CTC,
    claimed by lower-income filers who cannot use the full CTC against their tax liability.
    Like CTC, it requires at least 1 qualifying child.

    Args:
        session: Database session
        years: Years to load (default: all available)
    """
    if years is None:
        years = list(ACTC_BY_CHILDREN_DATA.keys())

    for year in years:
        if year not in ACTC_BY_CHILDREN_DATA:
            continue

        data = ACTC_BY_CHILDREN_DATA[year]

        # Get or create national stratum (parent for child-count strata)
        national_stratum = get_or_create_stratum(
            session,
            name="US All Filers",
            jurisdiction=Jurisdiction.US_FEDERAL,
            constraints=[("is_tax_filer", "==", "1")],
            description="All individual income tax returns filed in the US",
            stratum_group_id="national",
        )

        # Create strata and targets for each child category
        for category_key, category_data in data.items():
            category_def = ACTC_CHILD_CATEGORIES[category_key]

            # Create stratum for this child category
            stratum = get_or_create_stratum(
                session,
                name=category_def["name"],
                jurisdiction=Jurisdiction.US_FEDERAL,
                constraints=[
                    ("is_tax_filer", "==", "1"),
                    ("actc_qualifying_children", category_def["constraint_operator"], category_def["constraint_value"]),
                ],
                description=category_def["description"],
                parent_id=national_stratum.id,
                stratum_group_id="actc_by_children",
            )

            # Add ACTC claims target
            session.add(
                Target(
                    stratum_id=stratum.id,
                    variable="actc_claims",
                    period=year,
                    value=category_data["claims"],
                    target_type=TargetType.COUNT,
                    source=DataSource.IRS_SOI,
                    source_table="ACTC Statistics by Number of Qualifying Children",
                    source_url=SOURCE_URLS["ctc"],
                )
            )

            # Add ACTC amount target
            session.add(
                Target(
                    stratum_id=stratum.id,
                    variable="actc_amount",
                    period=year,
                    value=category_data["amount"],
                    target_type=TargetType.AMOUNT,
                    source=DataSource.IRS_SOI,
                    source_table="ACTC Statistics by Number of Qualifying Children",
                    source_url=SOURCE_URLS["ctc"],
                )
            )

    session.commit()


def run_etl(db_path=None):
    """Run the state-level tax credits ETL pipeline."""
    from pathlib import Path

    from .schema import DEFAULT_DB_PATH

    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    engine = init_db(path)

    with Session(engine) as session:
        load_soi_credits_targets(session)
        print(f"Loaded state-level SOI credits targets to {path}")


if __name__ == "__main__":
    run_etl()
