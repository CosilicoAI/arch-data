"""
ETL for SSA (Social Security Administration) targets.

Loads data from SSA statistics into the targets database.
Data sources:
- https://www.ssa.gov/policy/docs/statcomps/supplement/
- https://www.ssa.gov/oact/STATS/table4a3.html
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

# SSA data by year
# Source: SSA Annual Statistical Supplement
# https://www.ssa.gov/policy/docs/statcomps/supplement/
SSA_DATA = {
    2023: {
        # OASDI (Old-Age, Survivors, and Disability Insurance)
        "total_beneficiaries": 67_500_000,
        "total_benefits": 1_415_000_000_000,  # $1.415 trillion annually

        # Retired workers
        "retired_workers": {
            "beneficiaries": 51_200_000,
            "benefits": 987_000_000_000,
            "avg_monthly_benefit": 1_907,  # dollars
        },

        # Disabled workers
        "disabled_workers": {
            "beneficiaries": 7_600_000,
            "benefits": 143_000_000_000,
            "avg_monthly_benefit": 1_537,
        },

        # Survivors
        "survivors": {
            "beneficiaries": 5_900_000,
            "benefits": 112_000_000_000,
            "avg_monthly_benefit": 1_498,
        },

        # Spouses and children
        "dependents": {
            "beneficiaries": 2_800_000,
            "benefits": 43_000_000_000,
        },

        # SSI (Supplemental Security Income)
        "ssi": {
            "recipients": 7_400_000,
            "payments": 59_800_000_000,
            "avg_monthly_payment": 674,
            "aged_recipients": 1_100_000,
            "blind_recipients": 67_000,
            "disabled_recipients": 6_233_000,
        },
    },
    2022: {
        "total_beneficiaries": 66_000_000,
        "total_benefits": 1_234_000_000_000,

        "retired_workers": {
            "beneficiaries": 50_100_000,
            "benefits": 876_000_000_000,
            "avg_monthly_benefit": 1_827,
        },

        "disabled_workers": {
            "beneficiaries": 7_800_000,
            "benefits": 134_000_000_000,
            "avg_monthly_benefit": 1_483,
        },

        "survivors": {
            "beneficiaries": 5_800_000,
            "benefits": 98_000_000_000,
            "avg_monthly_benefit": 1_456,
        },

        "dependents": {
            "beneficiaries": 2_300_000,
            "benefits": 38_000_000_000,
        },

        "ssi": {
            "recipients": 7_600_000,
            "payments": 56_700_000_000,
            "avg_monthly_payment": 621,
            "aged_recipients": 1_150_000,
            "blind_recipients": 69_000,
            "disabled_recipients": 6_381_000,
        },
    },
    2021: {
        "total_beneficiaries": 65_000_000,
        "total_benefits": 1_145_000_000_000,

        "retired_workers": {
            "beneficiaries": 49_000_000,
            "benefits": 812_000_000_000,
            "avg_monthly_benefit": 1_657,
        },

        "disabled_workers": {
            "beneficiaries": 8_000_000,
            "benefits": 128_000_000_000,
            "avg_monthly_benefit": 1_358,
        },

        "survivors": {
            "beneficiaries": 5_700_000,
            "benefits": 92_000_000_000,
            "avg_monthly_benefit": 1_387,
        },

        "dependents": {
            "beneficiaries": 2_300_000,
            "benefits": 35_000_000_000,
        },

        "ssi": {
            "recipients": 7_800_000,
            "payments": 54_800_000_000,
            "avg_monthly_payment": 586,
            "aged_recipients": 1_200_000,
            "blind_recipients": 71_000,
            "disabled_recipients": 6_529_000,
        },
    },
}

SOURCE_URL = "https://www.ssa.gov/policy/docs/statcomps/supplement/"


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


def load_ssa_targets(session: Session, years: list[int] | None = None):
    """
    Load SSA targets into database.

    Args:
        session: Database session
        years: Years to load (default: all available)
    """
    if years is None:
        years = list(SSA_DATA.keys())

    for year in years:
        if year not in SSA_DATA:
            continue

        data = SSA_DATA[year]

        # OASDI national stratum
        oasdi_stratum = get_or_create_stratum(
            session,
            name="US OASDI Beneficiaries",
            jurisdiction=Jurisdiction.US_FEDERAL,
            constraints=[("social_security", "==", "1")],
            description="All OASDI (Social Security) beneficiaries",
            stratum_group_id="ssa_national",
        )

        session.add(
            Target(
                stratum_id=oasdi_stratum.id,
                variable="oasdi_beneficiaries",
                period=year,
                value=data["total_beneficiaries"],
                target_type=TargetType.COUNT,
                source=DataSource.SSA,
                source_url=SOURCE_URL,
            )
        )

        session.add(
            Target(
                stratum_id=oasdi_stratum.id,
                variable="oasdi_benefits",
                period=year,
                value=data["total_benefits"],
                target_type=TargetType.AMOUNT,
                source=DataSource.SSA,
                source_url=SOURCE_URL,
            )
        )

        # Retired workers stratum
        retired_stratum = get_or_create_stratum(
            session,
            name="US Retired Workers",
            jurisdiction=Jurisdiction.US_FEDERAL,
            constraints=[("social_security_retired", "==", "1")],
            description="Social Security retired worker beneficiaries",
            parent_id=oasdi_stratum.id,
            stratum_group_id="ssa_categories",
        )

        retired_data = data["retired_workers"]
        session.add(
            Target(
                stratum_id=retired_stratum.id,
                variable="oasdi_beneficiaries",
                period=year,
                value=retired_data["beneficiaries"],
                target_type=TargetType.COUNT,
                source=DataSource.SSA,
                source_url=SOURCE_URL,
            )
        )

        session.add(
            Target(
                stratum_id=retired_stratum.id,
                variable="oasdi_benefits",
                period=year,
                value=retired_data["benefits"],
                target_type=TargetType.AMOUNT,
                source=DataSource.SSA,
                source_url=SOURCE_URL,
            )
        )

        session.add(
            Target(
                stratum_id=retired_stratum.id,
                variable="oasdi_avg_monthly_benefit",
                period=year,
                value=retired_data["avg_monthly_benefit"],
                target_type=TargetType.AMOUNT,
                source=DataSource.SSA,
                source_url=SOURCE_URL,
            )
        )

        # Disabled workers stratum
        disabled_stratum = get_or_create_stratum(
            session,
            name="US Disabled Workers",
            jurisdiction=Jurisdiction.US_FEDERAL,
            constraints=[("social_security_disabled", "==", "1")],
            description="Social Security disabled worker beneficiaries",
            parent_id=oasdi_stratum.id,
            stratum_group_id="ssa_categories",
        )

        disabled_data = data["disabled_workers"]
        session.add(
            Target(
                stratum_id=disabled_stratum.id,
                variable="oasdi_beneficiaries",
                period=year,
                value=disabled_data["beneficiaries"],
                target_type=TargetType.COUNT,
                source=DataSource.SSA,
                source_url=SOURCE_URL,
            )
        )

        session.add(
            Target(
                stratum_id=disabled_stratum.id,
                variable="oasdi_avg_monthly_benefit",
                period=year,
                value=disabled_data["avg_monthly_benefit"],
                target_type=TargetType.AMOUNT,
                source=DataSource.SSA,
                source_url=SOURCE_URL,
            )
        )

        # SSI stratum
        ssi_stratum = get_or_create_stratum(
            session,
            name="US SSI Recipients",
            jurisdiction=Jurisdiction.US_FEDERAL,
            constraints=[("ssi", "==", "1")],
            description="Supplemental Security Income recipients",
            stratum_group_id="ssa_ssi",
        )

        ssi_data = data["ssi"]
        session.add(
            Target(
                stratum_id=ssi_stratum.id,
                variable="ssi_recipients",
                period=year,
                value=ssi_data["recipients"],
                target_type=TargetType.COUNT,
                source=DataSource.SSA,
                source_url=SOURCE_URL,
            )
        )

        session.add(
            Target(
                stratum_id=ssi_stratum.id,
                variable="ssi_payments",
                period=year,
                value=ssi_data["payments"],
                target_type=TargetType.AMOUNT,
                source=DataSource.SSA,
                source_url=SOURCE_URL,
            )
        )

        session.add(
            Target(
                stratum_id=ssi_stratum.id,
                variable="ssi_avg_monthly_payment",
                period=year,
                value=ssi_data["avg_monthly_payment"],
                target_type=TargetType.AMOUNT,
                source=DataSource.SSA,
                source_url=SOURCE_URL,
            )
        )

    session.commit()


def run_etl(db_path=None):
    """Run the SSA ETL pipeline."""
    from pathlib import Path
    from .schema import DEFAULT_DB_PATH

    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    engine = init_db(path)

    with Session(engine) as session:
        load_ssa_targets(session)
        print(f"Loaded SSA targets to {path}")


if __name__ == "__main__":
    run_etl()
