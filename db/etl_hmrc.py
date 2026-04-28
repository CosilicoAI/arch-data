"""
ETL for UK HMRC (HM Revenue & Customs) targets.

Loads data from HMRC statistics into the targets database.
Data sources:
- https://www.gov.uk/government/statistics/income-tax-liabilities-statistics
- https://www.gov.uk/government/collections/national-insurance-contributions-statistics
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

# HMRC data by tax year
# Source: HMRC statistics publications
# Values in GBP (pounds sterling)
HMRC_DATA = {
    2022: {
        # Tax revenue (billions GBP converted to GBP)
        "income_tax": 248_800_000_000,  # £248.8bn
        "national_insurance": 176_500_000_000,  # £176.5bn
        "capital_gains_tax": 18_200_000_000,  # £18.2bn
        "inheritance_tax": 7_100_000_000,  # £7.1bn

        # Taxpayer counts
        "taxpayers": 34_100_000,
        "higher_rate_taxpayers": 6_100_000,
        "additional_rate_taxpayers": 629_000,

        # Total income
        "total_income": 1_534_000_000_000,  # £1.534tn

        # Benefits (DWP data, included here for convenience)
        "benefits": {
            "universal_credit": {
                "recipients": 6_200_000,
                "expenditure": 52_300_000_000,  # £52.3bn
            },
            "child_benefit": {
                "recipients": 7_100_000,  # families
                "expenditure": 12_100_000_000,
            },
            "state_pension": {
                "recipients": 12_700_000,
                "expenditure": 124_300_000_000,
            },
            "housing_benefit": {
                "recipients": 2_800_000,
                "expenditure": 18_400_000_000,
            },
            "pension_credit": {
                "recipients": 1_400_000,
                "expenditure": 5_600_000_000,
            },
        },
    },
    2021: {
        "income_tax": 224_700_000_000,
        "national_insurance": 157_800_000_000,
        "capital_gains_tax": 14_900_000_000,
        "inheritance_tax": 6_100_000_000,

        "taxpayers": 32_400_000,
        "higher_rate_taxpayers": 5_500_000,
        "additional_rate_taxpayers": 538_000,

        "total_income": 1_423_000_000_000,

        "benefits": {
            "universal_credit": {
                "recipients": 5_800_000,
                "expenditure": 44_700_000_000,
            },
            "child_benefit": {
                "recipients": 7_200_000,
                "expenditure": 11_900_000_000,
            },
            "state_pension": {
                "recipients": 12_500_000,
                "expenditure": 105_500_000_000,
            },
            "housing_benefit": {
                "recipients": 3_100_000,
                "expenditure": 20_100_000_000,
            },
            "pension_credit": {
                "recipients": 1_500_000,
                "expenditure": 5_400_000_000,
            },
        },
    },
}

SOURCE_URL_TAX = "https://www.gov.uk/government/statistics/income-tax-liabilities-statistics"
SOURCE_URL_DWP = "https://www.gov.uk/government/collections/dwp-benefits-statistics"


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


def load_hmrc_targets(session: Session, years: list[int] | None = None):
    """
    Load HMRC targets into database.

    Args:
        session: Database session
        years: Years to load (default: all available)
    """
    if years is None:
        years = list(HMRC_DATA.keys())

    for year in years:
        if year not in HMRC_DATA:
            continue

        data = HMRC_DATA[year]

        # Create national taxpayer stratum
        national_stratum = get_or_create_stratum(
            session,
            name="UK All Taxpayers",
            jurisdiction=Jurisdiction.UK,
            constraints=[("is_taxpayer", "==", "1")],  # Taxpayers only, not whole population
            description="All UK taxpayers",
            stratum_group_id="uk_national",
        )

        # Tax revenue targets
        session.add(
            Target(
                stratum_id=national_stratum.id,
                variable="income_tax",
                period=year,
                value=data["income_tax"],
                target_type=TargetType.AMOUNT,
                source=DataSource.HMRC,
                source_url=SOURCE_URL_TAX,
            )
        )

        session.add(
            Target(
                stratum_id=national_stratum.id,
                variable="national_insurance",
                period=year,
                value=data["national_insurance"],
                target_type=TargetType.AMOUNT,
                source=DataSource.HMRC,
                source_url=SOURCE_URL_TAX,
            )
        )

        session.add(
            Target(
                stratum_id=national_stratum.id,
                variable="capital_gains_tax",
                period=year,
                value=data["capital_gains_tax"],
                target_type=TargetType.AMOUNT,
                source=DataSource.HMRC,
                source_url=SOURCE_URL_TAX,
            )
        )

        session.add(
            Target(
                stratum_id=national_stratum.id,
                variable="inheritance_tax",
                period=year,
                value=data["inheritance_tax"],
                target_type=TargetType.AMOUNT,
                source=DataSource.HMRC,
                source_url=SOURCE_URL_TAX,
            )
        )

        # Taxpayer counts
        session.add(
            Target(
                stratum_id=national_stratum.id,
                variable="taxpayer_count",
                period=year,
                value=data["taxpayers"],
                target_type=TargetType.COUNT,
                source=DataSource.HMRC,
                source_url=SOURCE_URL_TAX,
            )
        )

        session.add(
            Target(
                stratum_id=national_stratum.id,
                variable="total_income",
                period=year,
                value=data["total_income"],
                target_type=TargetType.AMOUNT,
                source=DataSource.HMRC,
                source_url=SOURCE_URL_TAX,
            )
        )

        # Higher rate taxpayers stratum
        higher_rate_stratum = get_or_create_stratum(
            session,
            name="UK Higher Rate Taxpayers",
            jurisdiction=Jurisdiction.UK,
            constraints=[("income_tax_band", "==", "higher")],
            description="UK taxpayers in higher rate band",
            parent_id=national_stratum.id,
            stratum_group_id="uk_tax_bands",
        )

        session.add(
            Target(
                stratum_id=higher_rate_stratum.id,
                variable="taxpayer_count",
                period=year,
                value=data["higher_rate_taxpayers"],
                target_type=TargetType.COUNT,
                source=DataSource.HMRC,
                source_url=SOURCE_URL_TAX,
            )
        )

        # Additional rate taxpayers stratum
        additional_rate_stratum = get_or_create_stratum(
            session,
            name="UK Additional Rate Taxpayers",
            jurisdiction=Jurisdiction.UK,
            constraints=[("income_tax_band", "==", "additional")],
            description="UK taxpayers in additional rate band",
            parent_id=national_stratum.id,
            stratum_group_id="uk_tax_bands",
        )

        session.add(
            Target(
                stratum_id=additional_rate_stratum.id,
                variable="taxpayer_count",
                period=year,
                value=data["additional_rate_taxpayers"],
                target_type=TargetType.COUNT,
                source=DataSource.HMRC,
                source_url=SOURCE_URL_TAX,
            )
        )

        # Benefit targets
        for benefit_name, benefit_data in data.get("benefits", {}).items():
            benefit_display = benefit_name.replace("_", " ").title()

            benefit_stratum = get_or_create_stratum(
                session,
                name=f"UK {benefit_display} Recipients",
                jurisdiction=Jurisdiction.UK,
                constraints=[(benefit_name, "==", "1")],
                description=f"UK {benefit_display} recipients",
                stratum_group_id="uk_benefits",
            )

            session.add(
                Target(
                    stratum_id=benefit_stratum.id,
                    variable=f"{benefit_name}_recipients",
                    period=year,
                    value=benefit_data["recipients"],
                    target_type=TargetType.COUNT,
                    source=DataSource.DWP,
                    source_url=SOURCE_URL_DWP,
                )
            )

            session.add(
                Target(
                    stratum_id=benefit_stratum.id,
                    variable=f"{benefit_name}_expenditure",
                    period=year,
                    value=benefit_data["expenditure"],
                    target_type=TargetType.AMOUNT,
                    source=DataSource.DWP,
                    source_url=SOURCE_URL_DWP,
                )
            )

    session.commit()


def run_etl(db_path=None):
    """Run the HMRC ETL pipeline."""
    from pathlib import Path
    from .schema import DEFAULT_DB_PATH

    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    engine = init_db(path)

    with Session(engine) as session:
        load_hmrc_targets(session)
        print(f"Loaded HMRC targets to {path}")


if __name__ == "__main__":
    run_etl()
