"""
ETL for SSI (Supplemental Security Income) targets.

Loads SSI-specific data from SSA statistics into the targets database.
This is separate from the general SSA ETL to provide more detailed SSI data
including state-level breakdowns and federal vs state supplementation.

Data sources:
- https://www.ssa.gov/policy/docs/statcomps/ssi_sc/2023/
- https://www.ssa.gov/policy/docs/statcomps/ssi_asr/
- https://www.ssa.gov/policy/docs/statcomps/supplement/
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

# SSI national data by year
# Source: SSA Annual Statistical Supplement and SSI Annual Statistical Report
# https://www.ssa.gov/policy/docs/statcomps/ssi_asr/
# https://www.ssa.gov/policy/docs/statcomps/supplement/2024/highlights.html
SSI_DATA = {
    2023: {
        "national": {
            # Total recipients and payments
            "recipients": 7_425_331,
            "total_payments": 61_400_000_000,  # $61.4 billion
            "avg_monthly_payment": 675,  # dollars
            # By eligibility category
            "aged_recipients": 1_160_608,
            "blind_recipients": 63_877,
            "disabled_recipients": 6_199_817,
            # Payments by category
            "aged_payments": 7_294_542_000,
            "blind_payments": 544_835_000,
            "disabled_payments": 53_536_042_000,
            # Federal vs state supplementation
            "federal_payments": 58_200_000_000,  # Approximately
            "state_supplementation": 3_200_000_000,  # $3.2 billion
        },
        # State-level data (December 2023)
        # Source: https://www.ssa.gov/policy/docs/statcomps/ssi_sc/2023/
        "states": {
            "CA": {
                "recipients": 1_199_000,
                "aged": 305_000,
                "blind": 17_000,
                "disabled": 877_000,
                "payments": 12_800_000_000,  # Includes state supplement
            },
            "TX": {
                "recipients": 573_750,
                "aged": 100_791,
                "blind": 8_500,
                "disabled": 464_459,
                "payments": 4_200_000_000,
            },
            "NY": {
                "recipients": 557_200,
                "aged": 113_136,
                "blind": 7_800,
                "disabled": 436_264,
                "payments": 4_800_000_000,  # Includes state supplement
            },
            "FL": {
                "recipients": 538_984,
                "aged": 158_775,
                "blind": 6_200,
                "disabled": 374_009,
                "payments": 4_100_000_000,
            },
            "PA": {
                "recipients": 321_000,
                "aged": 48_000,
                "blind": 4_500,
                "disabled": 268_500,
                "payments": 2_400_000_000,
            },
            "OH": {
                "recipients": 278_000,
                "aged": 35_000,
                "blind": 3_800,
                "disabled": 239_200,
                "payments": 2_000_000_000,
            },
            "IL": {
                "recipients": 248_000,
                "aged": 42_000,
                "blind": 3_200,
                "disabled": 202_800,
                "payments": 1_850_000_000,
            },
            "GA": {
                "recipients": 245_000,
                "aged": 38_000,
                "blind": 3_100,
                "disabled": 203_900,
                "payments": 1_750_000_000,
            },
            "MI": {
                "recipients": 232_000,
                "aged": 28_000,
                "blind": 3_000,
                "disabled": 201_000,
                "payments": 1_700_000_000,
            },
            "NC": {
                "recipients": 218_000,
                "aged": 34_000,
                "blind": 2_900,
                "disabled": 181_100,
                "payments": 1_550_000_000,
            },
        },
    },
    2022: {
        "national": {
            "recipients": 7_542_222,
            "total_payments": 56_700_000_000,  # $56.7 billion
            "avg_monthly_payment": 622,
            "aged_recipients": 1_180_000,
            "blind_recipients": 66_000,
            "disabled_recipients": 6_296_222,
            "aged_payments": 6_800_000_000,
            "blind_payments": 520_000_000,
            "disabled_payments": 49_380_000_000,
            "federal_payments": 53_800_000_000,
            "state_supplementation": 2_900_000_000,
        },
        "states": {
            "CA": {
                "recipients": 1_215_000,
                "aged": 310_000,
                "blind": 17_500,
                "disabled": 887_500,
                "payments": 12_200_000_000,
            },
            "TX": {
                "recipients": 582_000,
                "aged": 102_000,
                "blind": 8_700,
                "disabled": 471_300,
                "payments": 4_000_000_000,
            },
            "NY": {
                "recipients": 568_000,
                "aged": 115_000,
                "blind": 8_000,
                "disabled": 445_000,
                "payments": 4_600_000_000,
            },
            "FL": {
                "recipients": 545_000,
                "aged": 160_000,
                "blind": 6_300,
                "disabled": 378_700,
                "payments": 3_900_000_000,
            },
            "PA": {
                "recipients": 328_000,
                "aged": 49_000,
                "blind": 4_600,
                "disabled": 274_400,
                "payments": 2_300_000_000,
            },
        },
    },
    2021: {
        "national": {
            "recipients": 7_800_000,
            "total_payments": 61_000_000_000,
            "avg_monthly_payment": 586,
            "aged_recipients": 1_200_000,
            "blind_recipients": 71_000,
            "disabled_recipients": 6_529_000,
            "aged_payments": 6_500_000_000,
            "blind_payments": 500_000_000,
            "disabled_payments": 54_000_000_000,
            "federal_payments": 57_800_000_000,
            "state_supplementation": 3_200_000_000,
        },
        "states": {
            "CA": {
                "recipients": 1_230_000,
                "aged": 315_000,
                "blind": 18_000,
                "disabled": 897_000,
                "payments": 12_000_000_000,
            },
            "TX": {
                "recipients": 590_000,
                "aged": 104_000,
                "blind": 9_000,
                "disabled": 477_000,
                "payments": 3_800_000_000,
            },
            "NY": {
                "recipients": 580_000,
                "aged": 118_000,
                "blind": 8_200,
                "disabled": 453_800,
                "payments": 4_400_000_000,
            },
            "FL": {
                "recipients": 552_000,
                "aged": 162_000,
                "blind": 6_500,
                "disabled": 383_500,
                "payments": 3_700_000_000,
            },
            "PA": {
                "recipients": 335_000,
                "aged": 50_000,
                "blind": 4_800,
                "disabled": 280_200,
                "payments": 2_200_000_000,
            },
        },
    },
}

SOURCE_URL = "https://www.ssa.gov/policy/docs/statcomps/ssi_sc/2023/"
SOURCE_URL_ASR = "https://www.ssa.gov/policy/docs/statcomps/ssi_asr/"


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


def load_ssi_targets(session: Session, years: list[int] | None = None):
    """
    Load SSI targets into database.

    Args:
        session: Database session
        years: Years to load (default: all available)
    """
    if years is None:
        years = list(SSI_DATA.keys())

    for year in years:
        if year not in SSI_DATA:
            continue

        data = SSI_DATA[year]
        national_data = data["national"]

        # Create national SSI stratum
        national_stratum = get_or_create_stratum(
            session,
            name="US SSI Recipients",
            jurisdiction=Jurisdiction.US_FEDERAL,
            constraints=[("ssi", "==", "1")],
            description="All SSI recipients in the United States",
            stratum_group_id="ssi_national",
        )

        # National recipient count
        session.add(
            Target(
                stratum_id=national_stratum.id,
                variable="ssi_recipients",
                period=year,
                value=national_data["recipients"],
                target_type=TargetType.COUNT,
                source=DataSource.SSA,
                source_table="SSI Annual Statistical Report",
                source_url=SOURCE_URL_ASR,
            )
        )

        # National total payments
        session.add(
            Target(
                stratum_id=national_stratum.id,
                variable="ssi_total_payments",
                period=year,
                value=national_data["total_payments"],
                target_type=TargetType.AMOUNT,
                source=DataSource.SSA,
                source_table="SSI Annual Statistical Report",
                source_url=SOURCE_URL_ASR,
            )
        )

        # Average monthly payment
        session.add(
            Target(
                stratum_id=national_stratum.id,
                variable="ssi_avg_monthly_payment",
                period=year,
                value=national_data["avg_monthly_payment"],
                target_type=TargetType.AMOUNT,
                source=DataSource.SSA,
                source_table="SSI Annual Statistical Report",
                source_url=SOURCE_URL_ASR,
            )
        )

        # Federal vs state supplementation
        session.add(
            Target(
                stratum_id=national_stratum.id,
                variable="ssi_federal_payments",
                period=year,
                value=national_data["federal_payments"],
                target_type=TargetType.AMOUNT,
                source=DataSource.SSA,
                source_table="SSI Annual Statistical Report",
                source_url=SOURCE_URL_ASR,
            )
        )

        session.add(
            Target(
                stratum_id=national_stratum.id,
                variable="ssi_state_supplementation",
                period=year,
                value=national_data["state_supplementation"],
                target_type=TargetType.AMOUNT,
                source=DataSource.SSA,
                source_table="SSI Annual Statistical Report",
                source_url=SOURCE_URL_ASR,
            )
        )

        # Create SSI Aged stratum
        aged_stratum = get_or_create_stratum(
            session,
            name="US SSI Aged Recipients",
            jurisdiction=Jurisdiction.US_FEDERAL,
            constraints=[
                ("ssi", "==", "1"),
                ("ssi_category", "==", "aged"),
            ],
            description="SSI recipients aged 65 or older",
            parent_id=national_stratum.id,
            stratum_group_id="ssi_categories",
        )

        session.add(
            Target(
                stratum_id=aged_stratum.id,
                variable="ssi_recipients",
                period=year,
                value=national_data["aged_recipients"],
                target_type=TargetType.COUNT,
                source=DataSource.SSA,
                source_table="SSI Annual Statistical Report",
                source_url=SOURCE_URL_ASR,
            )
        )

        session.add(
            Target(
                stratum_id=aged_stratum.id,
                variable="ssi_payments",
                period=year,
                value=national_data["aged_payments"],
                target_type=TargetType.AMOUNT,
                source=DataSource.SSA,
                source_table="SSI Annual Statistical Report",
                source_url=SOURCE_URL_ASR,
            )
        )

        # Create SSI Blind stratum
        blind_stratum = get_or_create_stratum(
            session,
            name="US SSI Blind Recipients",
            jurisdiction=Jurisdiction.US_FEDERAL,
            constraints=[
                ("ssi", "==", "1"),
                ("ssi_category", "==", "blind"),
            ],
            description="SSI recipients qualifying based on blindness",
            parent_id=national_stratum.id,
            stratum_group_id="ssi_categories",
        )

        session.add(
            Target(
                stratum_id=blind_stratum.id,
                variable="ssi_recipients",
                period=year,
                value=national_data["blind_recipients"],
                target_type=TargetType.COUNT,
                source=DataSource.SSA,
                source_table="SSI Annual Statistical Report",
                source_url=SOURCE_URL_ASR,
            )
        )

        session.add(
            Target(
                stratum_id=blind_stratum.id,
                variable="ssi_payments",
                period=year,
                value=national_data["blind_payments"],
                target_type=TargetType.AMOUNT,
                source=DataSource.SSA,
                source_table="SSI Annual Statistical Report",
                source_url=SOURCE_URL_ASR,
            )
        )

        # Create SSI Disabled stratum
        disabled_stratum = get_or_create_stratum(
            session,
            name="US SSI Disabled Recipients",
            jurisdiction=Jurisdiction.US_FEDERAL,
            constraints=[
                ("ssi", "==", "1"),
                ("ssi_category", "==", "disabled"),
            ],
            description="SSI recipients qualifying based on disability",
            parent_id=national_stratum.id,
            stratum_group_id="ssi_categories",
        )

        session.add(
            Target(
                stratum_id=disabled_stratum.id,
                variable="ssi_recipients",
                period=year,
                value=national_data["disabled_recipients"],
                target_type=TargetType.COUNT,
                source=DataSource.SSA,
                source_table="SSI Annual Statistical Report",
                source_url=SOURCE_URL_ASR,
            )
        )

        session.add(
            Target(
                stratum_id=disabled_stratum.id,
                variable="ssi_payments",
                period=year,
                value=national_data["disabled_payments"],
                target_type=TargetType.AMOUNT,
                source=DataSource.SSA,
                source_table="SSI Annual Statistical Report",
                source_url=SOURCE_URL_ASR,
            )
        )

        # Add state-level targets
        for state_abbrev, state_data in data.get("states", {}).items():
            if state_abbrev not in STATE_FIPS:
                continue

            fips = STATE_FIPS[state_abbrev]

            state_stratum = get_or_create_stratum(
                session,
                name=f"{state_abbrev} SSI Recipients",
                jurisdiction=Jurisdiction.US,
                constraints=[
                    ("ssi", "==", "1"),
                    ("state_fips", "==", fips),
                ],
                description=f"SSI recipients in {state_abbrev}",
                parent_id=national_stratum.id,
                stratum_group_id="ssi_states",
            )

            # Total state recipients
            session.add(
                Target(
                    stratum_id=state_stratum.id,
                    variable="ssi_recipients",
                    period=year,
                    value=state_data["recipients"],
                    target_type=TargetType.COUNT,
                    source=DataSource.SSA,
                    source_table="SSI Recipients by State and County",
                    source_url=SOURCE_URL,
                )
            )

            # State payments
            session.add(
                Target(
                    stratum_id=state_stratum.id,
                    variable="ssi_payments",
                    period=year,
                    value=state_data["payments"],
                    target_type=TargetType.AMOUNT,
                    source=DataSource.SSA,
                    source_table="SSI Recipients by State and County",
                    source_url=SOURCE_URL,
                )
            )

            # State-level category breakdowns
            # Aged recipients in state
            state_aged_stratum = get_or_create_stratum(
                session,
                name=f"{state_abbrev} SSI Aged Recipients",
                jurisdiction=Jurisdiction.US,
                constraints=[
                    ("ssi", "==", "1"),
                    ("ssi_category", "==", "aged"),
                    ("state_fips", "==", fips),
                ],
                description=f"SSI aged recipients in {state_abbrev}",
                parent_id=state_stratum.id,
                stratum_group_id="ssi_state_categories",
            )

            session.add(
                Target(
                    stratum_id=state_aged_stratum.id,
                    variable="ssi_recipients",
                    period=year,
                    value=state_data["aged"],
                    target_type=TargetType.COUNT,
                    source=DataSource.SSA,
                    source_table="SSI Recipients by State and County",
                    source_url=SOURCE_URL,
                )
            )

            # Blind recipients in state
            state_blind_stratum = get_or_create_stratum(
                session,
                name=f"{state_abbrev} SSI Blind Recipients",
                jurisdiction=Jurisdiction.US,
                constraints=[
                    ("ssi", "==", "1"),
                    ("ssi_category", "==", "blind"),
                    ("state_fips", "==", fips),
                ],
                description=f"SSI blind recipients in {state_abbrev}",
                parent_id=state_stratum.id,
                stratum_group_id="ssi_state_categories",
            )

            session.add(
                Target(
                    stratum_id=state_blind_stratum.id,
                    variable="ssi_recipients",
                    period=year,
                    value=state_data["blind"],
                    target_type=TargetType.COUNT,
                    source=DataSource.SSA,
                    source_table="SSI Recipients by State and County",
                    source_url=SOURCE_URL,
                )
            )

            # Disabled recipients in state
            state_disabled_stratum = get_or_create_stratum(
                session,
                name=f"{state_abbrev} SSI Disabled Recipients",
                jurisdiction=Jurisdiction.US,
                constraints=[
                    ("ssi", "==", "1"),
                    ("ssi_category", "==", "disabled"),
                    ("state_fips", "==", fips),
                ],
                description=f"SSI disabled recipients in {state_abbrev}",
                parent_id=state_stratum.id,
                stratum_group_id="ssi_state_categories",
            )

            session.add(
                Target(
                    stratum_id=state_disabled_stratum.id,
                    variable="ssi_recipients",
                    period=year,
                    value=state_data["disabled"],
                    target_type=TargetType.COUNT,
                    source=DataSource.SSA,
                    source_table="SSI Recipients by State and County",
                    source_url=SOURCE_URL,
                )
            )

    session.commit()


def run_etl(db_path=None):
    """Run the SSI ETL pipeline."""
    from pathlib import Path
    from .schema import DEFAULT_DB_PATH

    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    engine = init_db(path)

    with Session(engine) as session:
        load_ssi_targets(session)
        print(f"Loaded SSI targets to {path}")


if __name__ == "__main__":
    run_etl()
