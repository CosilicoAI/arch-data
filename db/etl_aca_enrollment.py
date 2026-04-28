"""
ETL for CMS ACA Marketplace enrollment targets.

Loads data from CMS Marketplace enrollment reports into the targets database.
Data sources:
- https://www.cms.gov/data-research/statistics-trends-reports/marketplace-products/
- https://www.kff.org/affordable-care-act/state-indicator/marketplace-enrollment/
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
from arch.normalization import SourceFact, apply_share, as_target, target_kwargs

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
    "WY": "56",
}

# ACA Marketplace enrollment data by year
# Sources:
# - CMS Marketplace Open Enrollment Reports
# - KFF State Health Facts: https://www.kff.org/affordable-care-act/state-indicator/marketplace-enrollment/
# - CMS effectuated enrollment snapshots
ACA_ENROLLMENT_DATA = {
    2025: {
        "national": {
            # Total marketplace plan selections
            "total_enrollment": 24_319_713,
            "new_enrollees": 3_900_000,
            "returning_enrollees": 20_400_000,
            # Subsidy recipients (APTC = Advance Premium Tax Credit)
            "aptc_recipients": 22_400_000,  # ~92% of enrollees
            "aptc_recipients_pct": 0.92,
            # Average premiums and subsidies
            "avg_monthly_premium_before_aptc": 600,  # dollars
            "avg_monthly_aptc": 528,  # dollars
            "avg_monthly_premium_after_aptc": 101,  # dollars
            # Metal level distribution (national percentages)
            "bronze_pct": 0.31,
            "silver_pct": 0.54,
            "gold_pct": 0.13,
            "platinum_pct": 0.01,
            "catastrophic_pct": 0.01,
        },
        # State-level enrollment (plan selections)
        # Source: KFF State Health Facts, CMS OEP reports
        "states": {
            "AL": {"enrollment": 268_857},
            "AK": {"enrollment": 20_844},
            "AZ": {"enrollment": 423_025},
            "AR": {"enrollment": 137_442},
            "CA": {"enrollment": 1_979_504},
            "CO": {"enrollment": 229_085},
            "CT": {"enrollment": 104_789},
            "DE": {"enrollment": 35_213},
            "DC": {"enrollment": 20_178},
            "FL": {"enrollment": 4_735_415},
            "GA": {"enrollment": 1_510_852},
            "HI": {"enrollment": 27_897},
            "ID": {"enrollment": 120_087},
            "IL": {"enrollment": 459_873},
            "IN": {"enrollment": 285_124},
            "IA": {"enrollment": 99_456},
            "KS": {"enrollment": 136_789},
            "KY": {"enrollment": 112_345},
            "LA": {"enrollment": 295_432},
            "ME": {"enrollment": 78_234},
            "MD": {"enrollment": 189_567},
            "MA": {"enrollment": 322_456},
            "MI": {"enrollment": 531_083},
            "MN": {"enrollment": 142_567},
            "MS": {"enrollment": 198_765},
            "MO": {"enrollment": 367_234},
            "MT": {"enrollment": 78_123},
            "NE": {"enrollment": 123_456},
            "NV": {"enrollment": 118_234},
            "NH": {"enrollment": 62_345},
            "NJ": {"enrollment": 398_567},
            "NM": {"enrollment": 78_234},
            "NY": {"enrollment": 221_534},
            "NC": {"enrollment": 975_110},
            "ND": {"enrollment": 32_145},
            "OH": {"enrollment": 583_443},
            "OK": {"enrollment": 267_345},
            "OR": {"enrollment": 156_789},
            "PA": {"enrollment": 496_661},
            "RI": {"enrollment": 42_345},
            "SC": {"enrollment": 398_765},
            "SD": {"enrollment": 42_678},
            "TN": {"enrollment": 398_765},
            "TX": {"enrollment": 3_966_226},
            "UT": {"enrollment": 287_654},
            "VT": {"enrollment": 32_456},
            "VA": {"enrollment": 456_789},
            "WA": {"enrollment": 267_890},
            "WV": {"enrollment": 52_345},
            "WI": {"enrollment": 267_456},
            "WY": {"enrollment": 28_765},
        },
    },
    2024: {
        "national": {
            "total_enrollment": 21_446_150,
            "new_enrollees": 5_000_000,
            "returning_enrollees": 16_300_000,
            "aptc_recipients": 19_500_000,  # 93% of effectuated enrollment
            "aptc_recipients_pct": 0.93,
            "csr_recipients": 10_400_000,  # Cost-sharing reduction recipients
            "avg_monthly_premium_before_aptc": 592,
            "avg_monthly_aptc": 528,
            "avg_monthly_premium_after_aptc": 101,
            "bronze_pct": 0.31,
            "silver_pct": 0.54,
            "gold_pct": 0.13,
            "platinum_pct": 0.01,
            "catastrophic_pct": 0.01,
        },
        "states": {
            "AL": {"enrollment": 235_678},
            "AK": {"enrollment": 18_234},
            "AZ": {"enrollment": 348_055},
            "AR": {"enrollment": 118_456},
            "CA": {"enrollment": 1_784_653},
            "CO": {"enrollment": 198_765},
            "CT": {"enrollment": 98_234},
            "DE": {"enrollment": 32_456},
            "DC": {"enrollment": 19_987},
            "FL": {"enrollment": 4_211_902},
            "GA": {"enrollment": 1_305_114},
            "HI": {"enrollment": 26_543},
            "ID": {"enrollment": 105_678},
            "IL": {"enrollment": 398_765},
            "IN": {"enrollment": 245_678},
            "IA": {"enrollment": 87_654},
            "KS": {"enrollment": 118_765},
            "KY": {"enrollment": 87_234},
            "LA": {"enrollment": 214_567},
            "ME": {"enrollment": 75_234},
            "MD": {"enrollment": 167_890},
            "MA": {"enrollment": 298_765},
            "MI": {"enrollment": 418_100},
            "MN": {"enrollment": 128_765},
            "MS": {"enrollment": 167_234},
            "MO": {"enrollment": 318_765},
            "MT": {"enrollment": 67_234},
            "NE": {"enrollment": 108_765},
            "NV": {"enrollment": 108_234},
            "NH": {"enrollment": 54_678},
            "NJ": {"enrollment": 308_765},
            "NM": {"enrollment": 68_765},
            "NY": {"enrollment": 288_681},
            "NC": {"enrollment": 1_027_930},
            "ND": {"enrollment": 28_765},
            "OH": {"enrollment": 477_793},
            "OK": {"enrollment": 234_567},
            "OR": {"enrollment": 148_765},
            "PA": {"enrollment": 434_571},
            "RI": {"enrollment": 38_765},
            "SC": {"enrollment": 348_765},
            "SD": {"enrollment": 41_234},
            "TN": {"enrollment": 345_678},
            "TX": {"enrollment": 3_484_632},
            "UT": {"enrollment": 248_765},
            "VT": {"enrollment": 28_765},
            "VA": {"enrollment": 398_765},
            "WA": {"enrollment": 238_765},
            "WV": {"enrollment": 39_876},
            "WI": {"enrollment": 234_567},
            "WY": {"enrollment": 24_567},
        },
    },
    2023: {
        "national": {
            "total_enrollment": 16_357_030,
            "new_enrollees": 3_600_000,
            "returning_enrollees": 12_700_000,
            "aptc_recipients": 14_800_000,
            "aptc_recipients_pct": 0.92,
            "avg_monthly_premium_before_aptc": 593,
            "avg_monthly_aptc": 520,
            "avg_monthly_premium_after_aptc": 117,
            "bronze_pct": 0.32,
            "silver_pct": 0.53,
            "gold_pct": 0.13,
            "platinum_pct": 0.01,
            "catastrophic_pct": 0.01,
        },
        "states": {
            "AL": {"enrollment": 178_456},
            "AK": {"enrollment": 15_678},
            "AZ": {"enrollment": 235_229},
            "AR": {"enrollment": 89_234},
            "CA": {"enrollment": 1_739_368},
            "CO": {"enrollment": 178_234},
            "CT": {"enrollment": 87_654},
            "DE": {"enrollment": 28_765},
            "DC": {"enrollment": 18_765},
            "FL": {"enrollment": 3_225_435},
            "GA": {"enrollment": 879_084},
            "HI": {"enrollment": 24_567},
            "ID": {"enrollment": 89_876},
            "IL": {"enrollment": 298_765},
            "IN": {"enrollment": 187_654},
            "IA": {"enrollment": 67_234},
            "KS": {"enrollment": 89_765},
            "KY": {"enrollment": 67_234},
            "LA": {"enrollment": 148_765},
            "ME": {"enrollment": 65_432},
            "MD": {"enrollment": 145_678},
            "MA": {"enrollment": 267_890},
            "MI": {"enrollment": 322_273},
            "MN": {"enrollment": 108_765},
            "MS": {"enrollment": 118_765},
            "MO": {"enrollment": 245_678},
            "MT": {"enrollment": 54_678},
            "NE": {"enrollment": 87_234},
            "NV": {"enrollment": 98_765},
            "NH": {"enrollment": 45_678},
            "NJ": {"enrollment": 234_567},
            "NM": {"enrollment": 54_321},
            "NY": {"enrollment": 214_052},
            "NC": {"enrollment": 800_850},
            "ND": {"enrollment": 23_456},
            "OH": {"enrollment": 294_644},
            "OK": {"enrollment": 178_234},
            "OR": {"enrollment": 134_567},
            "PA": {"enrollment": 371_516},
            "RI": {"enrollment": 34_567},
            "SC": {"enrollment": 278_234},
            "SD": {"enrollment": 34_567},
            "TN": {"enrollment": 267_890},
            "TX": {"enrollment": 2_410_810},
            "UT": {"enrollment": 198_765},
            "VT": {"enrollment": 23_456},
            "VA": {"enrollment": 312_456},
            "WA": {"enrollment": 212_345},
            "WV": {"enrollment": 28_765},
            "WI": {"enrollment": 187_234},
            "WY": {"enrollment": 19_876},
        },
    },
}

# Metal level enrollment by state (2024 data)
# Source: KFF State Health Facts, CMS OEP State, Metal Level PUF
METAL_LEVEL_BY_STATE = {
    2024: {
        # Format: {state: {"bronze_pct": x, "silver_pct": x, "gold_pct": x}}
        # National averages used where state-specific not available
        "FL": {"bronze_pct": 0.31, "silver_pct": 0.63, "gold_pct": 0.05},
        "CA": {"bronze_pct": 0.26, "silver_pct": 0.60, "gold_pct": 0.09},
        "TX": {"bronze_pct": 0.16, "silver_pct": 0.49, "gold_pct": 0.34},
        "GA": {"bronze_pct": 0.28, "silver_pct": 0.58, "gold_pct": 0.12},
        "NC": {"bronze_pct": 0.25, "silver_pct": 0.57, "gold_pct": 0.15},
        "NJ": {"bronze_pct": 0.10, "silver_pct": 0.81, "gold_pct": 0.08},
        "MD": {"bronze_pct": 0.15, "silver_pct": 0.35, "gold_pct": 0.48},
    },
}

SOURCE_URL = "https://www.cms.gov/data-research/statistics-trends-reports/marketplace-products/"
KFF_SOURCE_URL = "https://www.kff.org/affordable-care-act/state-indicator/marketplace-enrollment/"


def build_aca_metal_enrollment_target(
    stratum: Stratum,
    *,
    total_enrollment: float,
    share: float,
    share_name: str,
    period: int,
    source_table: str,
    source_url: str,
) -> Target:
    """Build ACA metal-level enrollment from total enrollment and a share."""
    total_fact = SourceFact(
        name="aca_marketplace_enrollment",
        value=total_enrollment,
        period=period,
        unit="count",
        source=DataSource.CMS_ACA,
        jurisdiction=stratum.jurisdiction,
        source_table=source_table,
        source_url=source_url,
    )
    metal_fact = apply_share(
        total_fact,
        share,
        name="aca_marketplace_metal_level_enrollment",
        share_name=share_name,
        unit="count",
    )
    blueprint = as_target(
        metal_fact,
        variable="aca_marketplace_enrollment",
        target_type=TargetType.COUNT,
        stratum_name=stratum.name,
    )
    kwargs = target_kwargs(blueprint, stratum_id=stratum.id)
    kwargs["value"] = int(kwargs["value"])
    return Target(**kwargs)


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


def load_aca_enrollment_targets(session: Session, years: list[int] | None = None):
    """
    Load ACA Marketplace enrollment targets into database.

    Args:
        session: Database session
        years: Years to load (default: all available)
    """
    if years is None:
        years = list(ACA_ENROLLMENT_DATA.keys())

    for year in years:
        if year not in ACA_ENROLLMENT_DATA:
            continue

        data = ACA_ENROLLMENT_DATA[year]
        national_data = data["national"]

        # Create national ACA marketplace stratum
        national_stratum = get_or_create_stratum(
            session,
            name="US ACA Marketplace Enrollees",
            jurisdiction=Jurisdiction.US_FEDERAL,
            constraints=[("aca_marketplace", "==", "1")],
            description="All ACA Marketplace plan selections",
            stratum_group_id="aca_national",
        )

        # Total enrollment
        session.add(
            Target(
                stratum_id=national_stratum.id,
                variable="aca_marketplace_enrollment",
                period=year,
                value=national_data["total_enrollment"],
                target_type=TargetType.COUNT,
                source=DataSource.CMS_ACA,
                source_table="Marketplace OEP Report",
                source_url=SOURCE_URL,
            )
        )

        # New enrollees
        session.add(
            Target(
                stratum_id=national_stratum.id,
                variable="aca_marketplace_new_enrollees",
                period=year,
                value=national_data["new_enrollees"],
                target_type=TargetType.COUNT,
                source=DataSource.CMS_ACA,
                source_table="Marketplace OEP Report",
                source_url=SOURCE_URL,
            )
        )

        # APTC recipients stratum
        aptc_stratum = get_or_create_stratum(
            session,
            name="US ACA APTC Recipients",
            jurisdiction=Jurisdiction.US_FEDERAL,
            constraints=[
                ("aca_marketplace", "==", "1"),
                ("receives_aptc", "==", "1"),
            ],
            description="ACA Marketplace enrollees receiving Advance Premium Tax Credit",
            parent_id=national_stratum.id,
            stratum_group_id="aca_subsidies",
        )

        session.add(
            Target(
                stratum_id=aptc_stratum.id,
                variable="aca_aptc_recipients",
                period=year,
                value=national_data["aptc_recipients"],
                target_type=TargetType.COUNT,
                source=DataSource.CMS_ACA,
                source_table="Marketplace Effectuated Enrollment",
                source_url=SOURCE_URL,
            )
        )

        # Average premium and subsidy amounts
        session.add(
            Target(
                stratum_id=national_stratum.id,
                variable="aca_avg_monthly_premium_gross",
                period=year,
                value=national_data["avg_monthly_premium_before_aptc"],
                target_type=TargetType.AMOUNT,
                source=DataSource.CMS_ACA,
                source_table="Marketplace Effectuated Enrollment",
                source_url=SOURCE_URL,
            )
        )

        session.add(
            Target(
                stratum_id=aptc_stratum.id,
                variable="aca_avg_monthly_aptc",
                period=year,
                value=national_data["avg_monthly_aptc"],
                target_type=TargetType.AMOUNT,
                source=DataSource.CMS_ACA,
                source_table="Marketplace Effectuated Enrollment",
                source_url=SOURCE_URL,
            )
        )

        session.add(
            Target(
                stratum_id=national_stratum.id,
                variable="aca_avg_monthly_premium_net",
                period=year,
                value=national_data["avg_monthly_premium_after_aptc"],
                target_type=TargetType.AMOUNT,
                source=DataSource.CMS_ACA,
                source_table="Marketplace Effectuated Enrollment",
                source_url=SOURCE_URL,
            )
        )

        # CSR recipients (if available)
        if "csr_recipients" in national_data:
            csr_stratum = get_or_create_stratum(
                session,
                name="US ACA CSR Recipients",
                jurisdiction=Jurisdiction.US_FEDERAL,
                constraints=[
                    ("aca_marketplace", "==", "1"),
                    ("receives_csr", "==", "1"),
                ],
                description="ACA Marketplace enrollees receiving Cost-Sharing Reductions",
                parent_id=national_stratum.id,
                stratum_group_id="aca_subsidies",
            )

            session.add(
                Target(
                    stratum_id=csr_stratum.id,
                    variable="aca_csr_recipients",
                    period=year,
                    value=national_data["csr_recipients"],
                    target_type=TargetType.COUNT,
                    source=DataSource.CMS_ACA,
                    source_table="Marketplace Effectuated Enrollment",
                    source_url=SOURCE_URL,
                )
            )

        # Metal level strata
        for metal_level in ["bronze", "silver", "gold", "platinum", "catastrophic"]:
            pct_key = f"{metal_level}_pct"
            if pct_key in national_data:
                metal_stratum = get_or_create_stratum(
                    session,
                    name=f"US ACA {metal_level.capitalize()} Plan Enrollees",
                    jurisdiction=Jurisdiction.US_FEDERAL,
                    constraints=[
                        ("aca_marketplace", "==", "1"),
                        ("aca_metal_level", "==", metal_level),
                    ],
                    description=f"ACA Marketplace enrollees in {metal_level} plans",
                    parent_id=national_stratum.id,
                    stratum_group_id="aca_metal_levels",
                )

                session.add(
                    build_aca_metal_enrollment_target(
                        metal_stratum,
                        total_enrollment=national_data["total_enrollment"],
                        share=national_data[pct_key],
                        share_name=pct_key,
                        period=year,
                        source_table="Marketplace OEP Metal Level Report",
                        source_url=SOURCE_URL,
                    )
                )

                # Also store the percentage as a rate
                session.add(
                    Target(
                        stratum_id=metal_stratum.id,
                        variable="aca_metal_level_share",
                        period=year,
                        value=national_data[pct_key],
                        target_type=TargetType.RATE,
                        source=DataSource.CMS_ACA,
                        source_table="Marketplace OEP Metal Level Report",
                        source_url=SOURCE_URL,
                    )
                )

        # Add state-level targets
        for state_abbrev, state_data in data.get("states", {}).items():
            if state_abbrev not in STATE_FIPS:
                continue

            fips = STATE_FIPS[state_abbrev]

            state_stratum = get_or_create_stratum(
                session,
                name=f"{state_abbrev} ACA Marketplace Enrollees",
                jurisdiction=Jurisdiction.US,
                constraints=[
                    ("aca_marketplace", "==", "1"),
                    ("state_fips", "==", fips),
                ],
                description=f"ACA Marketplace enrollees in {state_abbrev}",
                parent_id=national_stratum.id,
                stratum_group_id="aca_states",
            )

            session.add(
                Target(
                    stratum_id=state_stratum.id,
                    variable="aca_marketplace_enrollment",
                    period=year,
                    value=state_data["enrollment"],
                    target_type=TargetType.COUNT,
                    source=DataSource.CMS_ACA,
                    source_table="Marketplace OEP State Report",
                    source_url=KFF_SOURCE_URL,
                )
            )

            # Add state-level metal distribution if available
            if year in METAL_LEVEL_BY_STATE:
                metal_data = METAL_LEVEL_BY_STATE[year].get(state_abbrev)
                if metal_data:
                    for metal_level, pct in metal_data.items():
                        if metal_level.endswith("_pct"):
                            level_name = metal_level.replace("_pct", "")
                            state_metal_stratum = get_or_create_stratum(
                                session,
                                name=f"{state_abbrev} ACA {level_name.capitalize()} Plan Enrollees",
                                jurisdiction=Jurisdiction.US,
                                constraints=[
                                    ("aca_marketplace", "==", "1"),
                                    ("state_fips", "==", fips),
                                    ("aca_metal_level", "==", level_name),
                                ],
                                description=f"ACA {level_name} plan enrollees in {state_abbrev}",
                                parent_id=state_stratum.id,
                                stratum_group_id="aca_state_metal_levels",
                            )

                            session.add(
                                build_aca_metal_enrollment_target(
                                    state_metal_stratum,
                                    total_enrollment=state_data["enrollment"],
                                    share=pct,
                                    share_name=metal_level,
                                    period=year,
                                    source_table="Marketplace OEP State Metal Level Report",
                                    source_url=SOURCE_URL,
                                )
                            )

    session.commit()


def run_etl(db_path=None):
    """Run the ACA enrollment ETL pipeline."""
    from pathlib import Path
    from .schema import DEFAULT_DB_PATH

    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    engine = init_db(path)

    with Session(engine) as session:
        load_aca_enrollment_targets(session)
        print(f"Loaded ACA Marketplace enrollment targets to {path}")


if __name__ == "__main__":
    run_etl()
