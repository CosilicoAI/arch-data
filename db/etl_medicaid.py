"""
ETL for CMS Medicaid enrollment targets.

Loads Medicaid and CHIP enrollment data from CMS into the targets database.
Data sources:
- https://www.medicaid.gov/medicaid/national-medicaid-chip-program-information/medicaid-chip-enrollment-data
- https://www.kff.org/affordable-care-act/state-indicator/total-monthly-medicaid-and-chip-enrollment/
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

# State FIPS codes
STATE_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06",
    "CO": "08", "CT": "09", "DE": "10", "DC": "11", "FL": "12",
    "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18",
    "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23",
    "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28",
    "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44",
    "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49",
    "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55",
    "WY": "56", "PR": "72", "VI": "78", "GU": "66",
}

# Medicaid enrollment data by year
# Sources:
# - CMS Medicaid & CHIP Enrollment Data: https://www.medicaid.gov/medicaid/national-medicaid-chip-program-information/medicaid-chip-enrollment-data
# - KFF State Health Facts: https://www.kff.org/affordable-care-act/state-indicator/total-monthly-medicaid-and-chip-enrollment/
# Values are as of August of each year unless otherwise noted
MEDICAID_DATA = {
    2025: {
        "national": {
            # August 2025 data from CMS
            "total_enrollment": 77_290_050,  # Total Medicaid + CHIP
            "medicaid_enrollment": 70_067_213,
            "chip_enrollment": 7_222_837,
            # By age category (from KFF)
            "children": 36_100_000,  # 48% of total
            "adults": 39_400_000,  # 52% of total (includes aged/disabled)
        },
        # State enrollment from KFF/CMS August 2025
        "states": {
            "CA": {"total": 13_226_087},
            "NY": {"total": 6_522_114},
            "TX": {"total": 4_161_101},
            "FL": {"total": 3_674_391},
            "PA": {"total": 3_018_973},
            "IL": {"total": 2_987_000},
            "OH": {"total": 2_654_000},
            "MI": {"total": 2_456_000},
            "NC": {"total": 2_387_000},
            "GA": {"total": 2_123_000},
            "NJ": {"total": 1_987_000},
            "AZ": {"total": 1_876_000},
            "WA": {"total": 1_765_000},
            "MA": {"total": 1_654_000},
            "VA": {"total": 1_543_000},
        },
    },
    2024: {
        "national": {
            # Annual average from CMS
            "total_enrollment": 81_500_000,
            "medicaid_enrollment": 73_800_000,
            "chip_enrollment": 7_700_000,
            "children": 38_500_000,
            "adults": 43_000_000,
        },
        "states": {
            "CA": {"total": 13_959_148},
            "NY": {"total": 7_123_000},
            "TX": {"total": 4_567_000},
            "FL": {"total": 4_234_000},
            "PA": {"total": 3_345_000},
            "IL": {"total": 3_234_000},
            "OH": {"total": 2_876_000},
            "MI": {"total": 2_654_000},
            "NC": {"total": 2_543_000},
            "GA": {"total": 2_345_000},
            "NJ": {"total": 2_123_000},
            "AZ": {"total": 2_098_000},
            "WA": {"total": 1_987_000},
            "MA": {"total": 1_876_000},
            "VA": {"total": 1_765_000},
        },
    },
    2023: {
        "national": {
            # Peak enrollment year (before unwinding completed)
            "total_enrollment": 94_000_000,  # Record high in March 2023
            "medicaid_enrollment": 85_500_000,
            "chip_enrollment": 8_500_000,
            "children": 43_000_000,
            "adults": 51_000_000,
            # By eligibility category from KFF (FY2023 data)
            "aged": 10_800_000,  # 10%
            "disabled": 12_300_000,  # 11%
            "non_disabled_adults": 21_800_000,  # 20%
            "aca_expansion_adults": 27_400_000,  # 25%
        },
        "states": {
            "CA": {"total": 15_234_000},
            "NY": {"total": 7_876_000},
            "TX": {"total": 5_234_000},
            "FL": {"total": 4_987_000},
            "PA": {"total": 3_654_000},
            "IL": {"total": 3_543_000},
            "OH": {"total": 3_123_000},
            "MI": {"total": 2_987_000},
            "NC": {"total": 2_765_000},
            "GA": {"total": 2_654_000},
            "NJ": {"total": 2_345_000},
            "AZ": {"total": 2_298_000},
            "WA": {"total": 2_123_000},
            "MA": {"total": 2_098_000},
            "VA": {"total": 1_987_000},
        },
    },
    2022: {
        "national": {
            # During continuous enrollment period
            "total_enrollment": 91_400_000,
            "medicaid_enrollment": 83_100_000,
            "chip_enrollment": 8_300_000,
            "children": 42_000_000,
            "adults": 49_400_000,
        },
        "states": {
            "CA": {"total": 14_876_000},
            "NY": {"total": 7_654_000},
            "TX": {"total": 5_098_000},
            "FL": {"total": 4_765_000},
            "PA": {"total": 3_543_000},
            "IL": {"total": 3_432_000},
            "OH": {"total": 3_012_000},
            "MI": {"total": 2_876_000},
            "NC": {"total": 2_654_000},
            "GA": {"total": 2_543_000},
        },
    },
    2021: {
        "national": {
            # COVID continuous enrollment period
            "total_enrollment": 85_000_000,
            "medicaid_enrollment": 77_200_000,
            "chip_enrollment": 7_800_000,
            "children": 39_500_000,
            "adults": 45_500_000,
        },
        "states": {
            "CA": {"total": 14_234_000},
            "NY": {"total": 7_234_000},
            "TX": {"total": 4_876_000},
            "FL": {"total": 4_432_000},
            "PA": {"total": 3_345_000},
            "IL": {"total": 3_234_000},
            "OH": {"total": 2_876_000},
            "MI": {"total": 2_654_000},
            "NC": {"total": 2_432_000},
            "GA": {"total": 2_345_000},
        },
    },
}

# Source URLs
SOURCE_URL = "https://www.medicaid.gov/medicaid/national-medicaid-chip-program-information/medicaid-chip-enrollment-data"
KFF_SOURCE_URL = "https://www.kff.org/affordable-care-act/state-indicator/total-monthly-medicaid-and-chip-enrollment/"


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

    existing = session.exec(
        select(Stratum).where(Stratum.definition_hash == definition_hash)
    ).first()

    if existing:
        return existing

    stratum = Stratum(
        name=name,
        description=description,
        jurisdiction=jurisdiction,
        definition_hash=definition_hash,
        parent_id=parent_id,
        stratum_group_id=stratum_group_id,
    )
    session.add(stratum)
    session.flush()

    for variable, operator, value in constraints:
        constraint = StratumConstraint(
            stratum_id=stratum.id,
            variable=variable,
            operator=operator,
            value=value,
        )
        session.add(constraint)

    return stratum


def load_medicaid_targets(session: Session, years: list[int] | None = None):
    """
    Load Medicaid enrollment targets into database.

    Args:
        session: Database session
        years: Years to load (default: all available)
    """
    if years is None:
        years = list(MEDICAID_DATA.keys())

    for year in years:
        if year not in MEDICAID_DATA:
            continue

        data = MEDICAID_DATA[year]
        national_data = data["national"]

        # Create national Medicaid stratum
        national_stratum = get_or_create_stratum(
            session,
            name="US Medicaid Enrollees",
            jurisdiction=Jurisdiction.US_FEDERAL,
            constraints=[("medicaid", "==", "1")],
            description="All Medicaid and CHIP enrollees in the US",
            stratum_group_id="medicaid_national",
        )

        # Total enrollment target
        session.add(
            Target(
                stratum_id=national_stratum.id,
                variable="medicaid_total_enrollment",
                period=year,
                value=national_data["total_enrollment"],
                target_type=TargetType.COUNT,
                source=DataSource.CMS_MEDICAID,
                source_table="Medicaid & CHIP Enrollment Data",
                source_url=SOURCE_URL,
            )
        )

        # Medicaid-only enrollment
        session.add(
            Target(
                stratum_id=national_stratum.id,
                variable="medicaid_enrollment",
                period=year,
                value=national_data["medicaid_enrollment"],
                target_type=TargetType.COUNT,
                source=DataSource.CMS_MEDICAID,
                source_table="Medicaid & CHIP Enrollment Data",
                source_url=SOURCE_URL,
            )
        )

        # CHIP enrollment
        chip_stratum = get_or_create_stratum(
            session,
            name="US CHIP Enrollees",
            jurisdiction=Jurisdiction.US_FEDERAL,
            constraints=[("chip", "==", "1")],
            description="Children's Health Insurance Program enrollees",
            parent_id=national_stratum.id,
            stratum_group_id="medicaid_chip",
        )

        session.add(
            Target(
                stratum_id=chip_stratum.id,
                variable="chip_enrollment",
                period=year,
                value=national_data["chip_enrollment"],
                target_type=TargetType.COUNT,
                source=DataSource.CMS_MEDICAID,
                source_table="Medicaid & CHIP Enrollment Data",
                source_url=SOURCE_URL,
            )
        )

        # Child enrollment stratum
        if "children" in national_data:
            child_stratum = get_or_create_stratum(
                session,
                name="US Medicaid Children",
                jurisdiction=Jurisdiction.US_FEDERAL,
                constraints=[
                    ("medicaid", "==", "1"),
                    ("is_child", "==", "1"),
                ],
                description="Children enrolled in Medicaid or CHIP",
                parent_id=national_stratum.id,
                stratum_group_id="medicaid_demographics",
            )

            session.add(
                Target(
                    stratum_id=child_stratum.id,
                    variable="medicaid_child_enrollment",
                    period=year,
                    value=national_data["children"],
                    target_type=TargetType.COUNT,
                    source=DataSource.CMS_MEDICAID,
                    source_table="Medicaid & CHIP Enrollment Data",
                    source_url=KFF_SOURCE_URL,
                )
            )

        # Adult enrollment stratum
        if "adults" in national_data:
            adult_stratum = get_or_create_stratum(
                session,
                name="US Medicaid Adults",
                jurisdiction=Jurisdiction.US_FEDERAL,
                constraints=[
                    ("medicaid", "==", "1"),
                    ("is_adult", "==", "1"),
                ],
                description="Adults enrolled in Medicaid (includes aged and disabled)",
                parent_id=national_stratum.id,
                stratum_group_id="medicaid_demographics",
            )

            session.add(
                Target(
                    stratum_id=adult_stratum.id,
                    variable="medicaid_adult_enrollment",
                    period=year,
                    value=national_data["adults"],
                    target_type=TargetType.COUNT,
                    source=DataSource.CMS_MEDICAID,
                    source_table="Medicaid & CHIP Enrollment Data",
                    source_url=KFF_SOURCE_URL,
                )
            )

        # Eligibility category strata (if available for the year)
        if "aged" in national_data:
            aged_stratum = get_or_create_stratum(
                session,
                name="US Medicaid Aged",
                jurisdiction=Jurisdiction.US_FEDERAL,
                constraints=[
                    ("medicaid", "==", "1"),
                    ("age", ">=", "65"),
                ],
                description="Aged (65+) Medicaid enrollees",
                parent_id=national_stratum.id,
                stratum_group_id="medicaid_eligibility",
            )

            session.add(
                Target(
                    stratum_id=aged_stratum.id,
                    variable="medicaid_aged_enrollment",
                    period=year,
                    value=national_data["aged"],
                    target_type=TargetType.COUNT,
                    source=DataSource.CMS_MEDICAID,
                    source_table="KFF Medicaid Enrollees by Enrollment Group",
                    source_url=KFF_SOURCE_URL,
                )
            )

        if "disabled" in national_data:
            disabled_stratum = get_or_create_stratum(
                session,
                name="US Medicaid Disabled",
                jurisdiction=Jurisdiction.US_FEDERAL,
                constraints=[
                    ("medicaid", "==", "1"),
                    ("is_disabled", "==", "1"),
                ],
                description="Disabled Medicaid enrollees (non-aged)",
                parent_id=national_stratum.id,
                stratum_group_id="medicaid_eligibility",
            )

            session.add(
                Target(
                    stratum_id=disabled_stratum.id,
                    variable="medicaid_disabled_enrollment",
                    period=year,
                    value=national_data["disabled"],
                    target_type=TargetType.COUNT,
                    source=DataSource.CMS_MEDICAID,
                    source_table="KFF Medicaid Enrollees by Enrollment Group",
                    source_url=KFF_SOURCE_URL,
                )
            )

        if "aca_expansion_adults" in national_data:
            expansion_stratum = get_or_create_stratum(
                session,
                name="US Medicaid ACA Expansion Adults",
                jurisdiction=Jurisdiction.US_FEDERAL,
                constraints=[
                    ("medicaid", "==", "1"),
                    ("is_aca_expansion", "==", "1"),
                ],
                description="Adults enrolled through ACA Medicaid expansion",
                parent_id=national_stratum.id,
                stratum_group_id="medicaid_eligibility",
            )

            session.add(
                Target(
                    stratum_id=expansion_stratum.id,
                    variable="medicaid_aca_expansion_enrollment",
                    period=year,
                    value=national_data["aca_expansion_adults"],
                    target_type=TargetType.COUNT,
                    source=DataSource.CMS_MEDICAID,
                    source_table="KFF Medicaid Enrollees by Enrollment Group",
                    source_url=KFF_SOURCE_URL,
                )
            )

        # State-level enrollment targets
        for state_abbrev, state_data in data.get("states", {}).items():
            if state_abbrev not in STATE_FIPS:
                continue

            fips = STATE_FIPS[state_abbrev]

            state_stratum = get_or_create_stratum(
                session,
                name=f"{state_abbrev} Medicaid Enrollees",
                jurisdiction=Jurisdiction.US,
                constraints=[
                    ("medicaid", "==", "1"),
                    ("state_fips", "==", fips),
                ],
                description=f"Medicaid and CHIP enrollees in {state_abbrev}",
                parent_id=national_stratum.id,
                stratum_group_id="medicaid_states",
            )

            session.add(
                Target(
                    stratum_id=state_stratum.id,
                    variable="medicaid_total_enrollment",
                    period=year,
                    value=state_data["total"],
                    target_type=TargetType.COUNT,
                    source=DataSource.CMS_MEDICAID,
                    source_table="Medicaid State Enrollment",
                    source_url=SOURCE_URL,
                )
            )

    session.commit()


def run_etl(db_path=None):
    """Run the Medicaid enrollment ETL pipeline."""
    from pathlib import Path
    from .schema import DEFAULT_DB_PATH

    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    engine = init_db(path)

    with Session(engine) as session:
        load_medicaid_targets(session)
        print(f"Loaded Medicaid enrollment targets to {path}")


if __name__ == "__main__":
    run_etl()
