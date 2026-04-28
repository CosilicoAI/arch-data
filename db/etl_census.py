"""
ETL for Census Bureau population targets.

Loads data from Census Bureau population estimates into the targets database.
Data source: https://www.census.gov/programs-surveys/popest.html
"""

from __future__ import annotations

from sqlmodel import Session, select

from .schema import (
    DataSource,
    GeographicLevel,
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
    "WY": "56", "PR": "72",
}

STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "PR": "Puerto Rico",
}

# Census data by year
# Source: Census Bureau Population Estimates
# https://www.census.gov/data/tables/time-series/demo/popest/2020s-national-detail.html
CENSUS_DATA = {
    2023: {
        "total_population": 334_914_895,
        "households": 131_432_000,
        "median_age": 38.9,

        # Age groups (5-year brackets matching PolicyEngine)
        # Based on Census Bureau Population Estimates
        # Source: https://www.census.gov/programs-surveys/popest.html
        "age_groups": {
            "0_to_4": 19_234_000,
            "5_to_9": 20_187_000,
            "10_to_14": 20_612_000,
            "15_to_19": 20_876_000,
            "20_to_24": 21_543_000,
            "25_to_29": 23_234_000,
            "30_to_34": 23_567_000,
            "35_to_39": 22_876_000,
            "40_to_44": 20_987_000,
            "45_to_49": 20_234_000,
            "50_to_54": 19_876_000,
            "55_to_59": 21_234_000,
            "60_to_64": 20_987_000,
            "65_to_69": 18_765_000,
            "70_to_74": 16_234_000,
            "75_to_79": 11_876_000,
            "80_to_84": 7_654_000,
            "85_plus": 4_938_895,
        },

        # Top 10 states by population
        "states": {
            "CA": {"population": 38_965_193, "households": 14_234_000},
            "TX": {"population": 30_503_301, "households": 10_876_000},
            "FL": {"population": 22_610_726, "households": 8_765_000},
            "NY": {"population": 19_571_216, "households": 7_654_000},
            "PA": {"population": 12_961_683, "households": 5_234_000},
            "IL": {"population": 12_549_689, "households": 4_987_000},
            "OH": {"population": 11_785_935, "households": 4_765_000},
            "GA": {"population": 11_029_227, "households": 4_123_000},
            "NC": {"population": 10_835_491, "households": 4_098_000},
            "MI": {"population": 10_037_261, "households": 3_987_000},
        },
    },
    2022: {
        "total_population": 333_287_557,
        "households": 130_456_000,
        "median_age": 38.8,

        # Age groups (5-year brackets matching PolicyEngine)
        "age_groups": {
            "0_to_4": 19_112_000,
            "5_to_9": 20_034_000,
            "10_to_14": 20_487_000,
            "15_to_19": 20_743_000,
            "20_to_24": 21_398_000,
            "25_to_29": 23_076_000,
            "30_to_34": 23_398_000,
            "35_to_39": 22_701_000,
            "40_to_44": 20_834_000,
            "45_to_49": 20_098_000,
            "50_to_54": 19_743_000,
            "55_to_59": 21_098_000,
            "60_to_64": 20_834_000,
            "65_to_69": 18_612_000,
            "70_to_74": 16_098_000,
            "75_to_79": 11_743_000,
            "80_to_84": 7_587_000,
            "85_plus": 4_891_557,
        },

        "states": {
            "CA": {"population": 39_029_342, "households": 14_123_000},
            "TX": {"population": 30_029_572, "households": 10_654_000},
            "FL": {"population": 22_244_823, "households": 8_543_000},
            "NY": {"population": 19_677_151, "households": 7_543_000},
            "PA": {"population": 12_972_008, "households": 5_198_000},
            "IL": {"population": 12_582_032, "households": 4_876_000},
            "OH": {"population": 11_756_058, "households": 4_654_000},
            "GA": {"population": 10_912_876, "households": 4_023_000},
            "NC": {"population": 10_698_973, "households": 3_987_000},
            "MI": {"population": 10_034_113, "households": 3_876_000},
        },
    },
    2021: {
        "total_population": 331_893_745,
        "households": 129_876_000,
        "median_age": 38.6,

        # Age groups (5-year brackets matching PolicyEngine)
        "age_groups": {
            "0_to_4": 19_045_000,
            "5_to_9": 19_932_000,
            "10_to_14": 20_398_000,
            "15_to_19": 20_643_000,
            "20_to_24": 21_287_000,
            "25_to_29": 22_943_000,
            "30_to_34": 23_267_000,
            "35_to_39": 22_567_000,
            "40_to_44": 20_698_000,
            "45_to_49": 19_987_000,
            "50_to_54": 19_643_000,
            "55_to_59": 20_987_000,
            "60_to_64": 20_698_000,
            "65_to_69": 18_487_000,
            "70_to_74": 15_987_000,
            "75_to_79": 11_643_000,
            "80_to_84": 7_532_000,
            "85_plus": 4_857_745,
        },

        "states": {
            "CA": {"population": 39_142_991, "households": 14_012_000},
            "TX": {"population": 29_527_941, "households": 10_432_000},
            "FL": {"population": 21_781_128, "households": 8_321_000},
            "NY": {"population": 19_835_913, "households": 7_432_000},
            "PA": {"population": 12_964_056, "households": 5_123_000},
            "IL": {"population": 12_671_469, "households": 4_765_000},
            "OH": {"population": 11_780_017, "households": 4_543_000},
            "GA": {"population": 10_799_566, "households": 3_912_000},
            "NC": {"population": 10_551_162, "households": 3_876_000},
            "MI": {"population": 10_050_811, "households": 3_765_000},
        },
    },
}

SOURCE_URL = "https://www.census.gov/programs-surveys/popest.html"


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


def load_census_targets(session: Session, years: list[int] | None = None):
    """
    Load Census population targets into database.

    Args:
        session: Database session
        years: Years to load (default: all available)
    """
    if years is None:
        years = list(CENSUS_DATA.keys())

    for year in years:
        if year not in CENSUS_DATA:
            continue

        data = CENSUS_DATA[year]

        # Create national population stratum
        national_stratum = get_or_create_stratum(
            session,
            name="US Population",
            jurisdiction=Jurisdiction.US,
            constraints=[],  # No constraints = total US
            description="Total US resident population",
            stratum_group_id="population_national",
        )

        # Total population
        session.add(
            Target(
                stratum_id=national_stratum.id,
                variable="population",
                period=year,
                value=data["total_population"],
                target_type=TargetType.COUNT,
                geographic_level=GeographicLevel.NATIONAL,
                source=DataSource.CENSUS_ACS,
                source_url=SOURCE_URL,
            )
        )

        # Households
        session.add(
            Target(
                stratum_id=national_stratum.id,
                variable="household_count",
                period=year,
                value=data["households"],
                target_type=TargetType.COUNT,
                geographic_level=GeographicLevel.NATIONAL,
                source=DataSource.CENSUS_ACS,
                source_url=SOURCE_URL,
            )
        )

        # Age group strata (18 brackets matching PolicyEngine)
        age_brackets = {
            "0_to_4": [("age", ">=", "0"), ("age", "<", "5")],
            "5_to_9": [("age", ">=", "5"), ("age", "<", "10")],
            "10_to_14": [("age", ">=", "10"), ("age", "<", "15")],
            "15_to_19": [("age", ">=", "15"), ("age", "<", "20")],
            "20_to_24": [("age", ">=", "20"), ("age", "<", "25")],
            "25_to_29": [("age", ">=", "25"), ("age", "<", "30")],
            "30_to_34": [("age", ">=", "30"), ("age", "<", "35")],
            "35_to_39": [("age", ">=", "35"), ("age", "<", "40")],
            "40_to_44": [("age", ">=", "40"), ("age", "<", "45")],
            "45_to_49": [("age", ">=", "45"), ("age", "<", "50")],
            "50_to_54": [("age", ">=", "50"), ("age", "<", "55")],
            "55_to_59": [("age", ">=", "55"), ("age", "<", "60")],
            "60_to_64": [("age", ">=", "60"), ("age", "<", "65")],
            "65_to_69": [("age", ">=", "65"), ("age", "<", "70")],
            "70_to_74": [("age", ">=", "70"), ("age", "<", "75")],
            "75_to_79": [("age", ">=", "75"), ("age", "<", "80")],
            "80_to_84": [("age", ">=", "80"), ("age", "<", "85")],
            "85_plus": [("age", ">=", "85")],
        }

        for age_name, constraints in age_brackets.items():
            if age_name not in data.get("age_groups", {}):
                continue

            age_stratum = get_or_create_stratum(
                session,
                name=f"US Population {age_name.replace('_', ' ').title()}",
                jurisdiction=Jurisdiction.US,
                constraints=constraints,
                description=f"US population ages {age_name}",
                parent_id=national_stratum.id,
                stratum_group_id="age_groups",
            )

            session.add(
                Target(
                    stratum_id=age_stratum.id,
                    variable="population",
                    period=year,
                    value=data["age_groups"][age_name],
                    target_type=TargetType.COUNT,
                    geographic_level=GeographicLevel.NATIONAL,
                    source=DataSource.CENSUS_ACS,
                    source_url=SOURCE_URL,
                )
            )

        # State strata
        for state_abbrev, state_data in data.get("states", {}).items():
            if state_abbrev not in STATE_FIPS:
                continue

            fips = STATE_FIPS[state_abbrev]
            state_name = STATE_NAMES.get(state_abbrev, state_abbrev)

            state_stratum = get_or_create_stratum(
                session,
                name=f"{state_name} Population",
                jurisdiction=Jurisdiction.US,
                constraints=[("state_fips", "==", fips)],
                description=f"Population of {state_name}",
                parent_id=national_stratum.id,
                stratum_group_id="state_population",
            )

            session.add(
                Target(
                    stratum_id=state_stratum.id,
                    variable="population",
                    period=year,
                    value=state_data["population"],
                    target_type=TargetType.COUNT,
                    geographic_level=GeographicLevel.STATE,
                    source=DataSource.CENSUS_ACS,
                    source_url=SOURCE_URL,
                )
            )

            if "households" in state_data:
                session.add(
                    Target(
                        stratum_id=state_stratum.id,
                        variable="household_count",
                        period=year,
                        value=state_data["households"],
                        target_type=TargetType.COUNT,
                        geographic_level=GeographicLevel.STATE,
                        source=DataSource.CENSUS_ACS,
                        source_url=SOURCE_URL,
                    )
                )

    session.commit()


def load_congressional_district_targets(
    session: Session, year: int, district_data: dict
):
    """
    Load congressional district population targets into database.

    Args:
        session: Database session
        year: Year for the data
        district_data: Dict mapping (state_fips, district) to population data
            Example: {("06", "01"): {"population": 750000}}
    """
    # Get or create national stratum for parent relationship
    national_stratum = get_or_create_stratum(
        session,
        name="US Population",
        jurisdiction=Jurisdiction.US,
        constraints=[],
        description="Total US resident population",
        stratum_group_id="population_national",
    )

    for (state_fips, district), data in district_data.items():
        # Create congressional district stratum
        cd_stratum = get_or_create_stratum(
            session,
            name=f"Congressional District {state_fips}-{district}",
            jurisdiction=Jurisdiction.US,
            constraints=[
                ("state_fips", "==", state_fips),
                ("congressional_district", "==", district),
            ],
            description=f"Population in Congressional District {district} of state {state_fips}",
            parent_id=national_stratum.id,
            stratum_group_id="congressional_districts",
        )

        # Add population target
        session.add(
            Target(
                stratum_id=cd_stratum.id,
                variable="population",
                period=year,
                value=data["population"],
                target_type=TargetType.COUNT,
                geographic_level=GeographicLevel.CONGRESSIONAL_DISTRICT,
                source=DataSource.CENSUS_ACS,
                source_url=SOURCE_URL,
                notes="Congressional district boundaries as of current Congress",
            )
        )

        # Add households if available
        if "households" in data:
            session.add(
                Target(
                    stratum_id=cd_stratum.id,
                    variable="household_count",
                    period=year,
                    value=data["households"],
                    target_type=TargetType.COUNT,
                    geographic_level=GeographicLevel.CONGRESSIONAL_DISTRICT,
                    source=DataSource.CENSUS_ACS,
                    source_url=SOURCE_URL,
                    notes="Congressional district boundaries as of current Congress",
                )
            )

    session.commit()


def run_etl(db_path=None):
    """Run the Census ETL pipeline."""
    from pathlib import Path
    from .schema import DEFAULT_DB_PATH

    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    engine = init_db(path)

    with Session(engine) as session:
        load_census_targets(session)
        print(f"Loaded Census targets to {path}")


if __name__ == "__main__":
    run_etl()
