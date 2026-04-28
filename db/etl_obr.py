"""
ETL for OBR (Office for Budget Responsibility) projections.

Loads UK fiscal and economic projections from OBR's Economic and Fiscal Outlook.
Data source: https://obr.uk/efo/economic-and-fiscal-outlook-march-2024/
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

# OBR projections data (5-year forecast, 2024-2029)
# Source: OBR Economic and Fiscal Outlook, March 2024
# https://obr.uk/efo/economic-and-fiscal-outlook-march-2024/
# GDP and budget figures in billions of GBP
# Rates as percentages
OBR_DATA = {
    # Fiscal Year 2024-25
    2024: {
        # GDP and public finances (billions GBP)
        "gdp": 2_791_000_000_000,  # £2.791 trillion
        "total_receipts": 1_139_000_000_000,
        "total_managed_expenditure": 1_226_000_000_000,
        "public_sector_net_borrowing": 87_000_000_000,
        "public_sector_net_debt": 2_654_000_000_000,
        # Economic indicators (percentages)
        "real_gdp_growth": 0.8,
        "unemployment_rate": 4.2,
        "cpi_inflation": 2.8,
        "rpi_inflation": 3.9,
        "bank_rate": 4.8,
        # Employment (thousands)
        "employment": 33_100_000,
    },
    2025: {
        "gdp": 2_929_000_000_000,
        "total_receipts": 1_207_000_000_000,
        "total_managed_expenditure": 1_296_000_000_000,
        "public_sector_net_borrowing": 89_000_000_000,
        "public_sector_net_debt": 2_783_000_000_000,
        "real_gdp_growth": 1.9,
        "unemployment_rate": 4.3,
        "cpi_inflation": 2.2,
        "rpi_inflation": 3.2,
        "bank_rate": 4.1,
        "employment": 33_200_000,
    },
    2026: {
        "gdp": 3_066_000_000_000,
        "total_receipts": 1_273_000_000_000,
        "total_managed_expenditure": 1_349_000_000_000,
        "public_sector_net_borrowing": 76_000_000_000,
        "public_sector_net_debt": 2_893_000_000_000,
        "real_gdp_growth": 2.0,
        "unemployment_rate": 4.1,
        "cpi_inflation": 1.8,
        "rpi_inflation": 2.8,
        "bank_rate": 3.6,
        "employment": 33_400_000,
    },
    2027: {
        "gdp": 3_199_000_000_000,
        "total_receipts": 1_337_000_000_000,
        "total_managed_expenditure": 1_399_000_000_000,
        "public_sector_net_borrowing": 62_000_000_000,
        "public_sector_net_debt": 2_985_000_000_000,
        "real_gdp_growth": 1.8,
        "unemployment_rate": 4.0,
        "cpi_inflation": 1.8,
        "rpi_inflation": 2.8,
        "bank_rate": 3.4,
        "employment": 33_500_000,
    },
    2028: {
        "gdp": 3_331_000_000_000,
        "total_receipts": 1_399_000_000_000,
        "total_managed_expenditure": 1_456_000_000_000,
        "public_sector_net_borrowing": 57_000_000_000,
        "public_sector_net_debt": 3_070_000_000_000,
        "real_gdp_growth": 1.7,
        "unemployment_rate": 4.0,
        "cpi_inflation": 2.0,
        "rpi_inflation": 3.0,
        "bank_rate": 3.3,
        "employment": 33_600_000,
    },
    2029: {
        "gdp": 3_464_000_000_000,
        "total_receipts": 1_459_000_000_000,
        "total_managed_expenditure": 1_512_000_000_000,
        "public_sector_net_borrowing": 53_000_000_000,
        "public_sector_net_debt": 3_148_000_000_000,
        "real_gdp_growth": 1.7,
        "unemployment_rate": 4.0,
        "cpi_inflation": 2.0,
        "rpi_inflation": 3.0,
        "bank_rate": 3.3,
        "employment": 33_700_000,
    },
}

SOURCE_URL = "https://obr.uk/efo/economic-and-fiscal-outlook-march-2024/"


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


def load_obr_targets(session: Session, years: list[int] | None = None):
    """
    Load OBR projections into database.

    Args:
        session: Database session
        years: Years to load (default: all available 2024-2029)
    """
    if years is None:
        years = list(OBR_DATA.keys())

    for year in years:
        if year not in OBR_DATA:
            continue

        data = OBR_DATA[year]

        # UK public finances stratum
        budget_stratum = get_or_create_stratum(
            session,
            name="UK Public Finances",
            jurisdiction=Jurisdiction.UK,
            constraints=[("sector", "==", "public")],  # Government sector
            description="UK public sector finances",
            stratum_group_id="obr_budget",
        )

        # Budget targets
        budget_vars = [
            ("gdp", TargetType.AMOUNT),
            ("total_receipts", TargetType.AMOUNT),
            ("total_managed_expenditure", TargetType.AMOUNT),
            ("public_sector_net_borrowing", TargetType.AMOUNT),
            ("public_sector_net_debt", TargetType.AMOUNT),
        ]

        for var_name, var_type in budget_vars:
            if var_name not in data:
                continue

            # Check for existing target
            existing = session.exec(
                select(Target).where(
                    Target.stratum_id == budget_stratum.id,
                    Target.variable == var_name,
                    Target.period == year,
                    Target.source == DataSource.OBR,
                )
            ).first()

            if existing:
                continue

            session.add(
                Target(
                    stratum_id=budget_stratum.id,
                    variable=var_name,
                    period=year,
                    value=data[var_name],
                    target_type=var_type,
                    source=DataSource.OBR,
                    source_url=SOURCE_URL,
                    is_preliminary=year > 2024,
                )
            )

        # UK Economy stratum for macro indicators
        economy_stratum = get_or_create_stratum(
            session,
            name="UK Economy",
            jurisdiction=Jurisdiction.UK,
            constraints=[],  # Whole economy, not a subset
            description="UK macroeconomic indicators",
            stratum_group_id="obr_economy",
        )

        # Economic rate targets
        rate_vars = [
            "real_gdp_growth",
            "unemployment_rate",
            "cpi_inflation",
            "rpi_inflation",
            "bank_rate",
        ]

        for var_name in rate_vars:
            if var_name not in data:
                continue

            existing = session.exec(
                select(Target).where(
                    Target.stratum_id == economy_stratum.id,
                    Target.variable == var_name,
                    Target.period == year,
                    Target.source == DataSource.OBR,
                )
            ).first()

            if existing:
                continue

            session.add(
                Target(
                    stratum_id=economy_stratum.id,
                    variable=var_name,
                    period=year,
                    value=data[var_name],
                    target_type=TargetType.RATE,
                    source=DataSource.OBR,
                    source_url=SOURCE_URL,
                    is_preliminary=year > 2024,
                )
            )

        # Employment count
        if "employment" in data:
            existing = session.exec(
                select(Target).where(
                    Target.stratum_id == economy_stratum.id,
                    Target.variable == "employment",
                    Target.period == year,
                    Target.source == DataSource.OBR,
                )
            ).first()

            if not existing:
                session.add(
                    Target(
                        stratum_id=economy_stratum.id,
                        variable="employment",
                        period=year,
                        value=data["employment"],
                        target_type=TargetType.COUNT,
                        source=DataSource.OBR,
                        source_url=SOURCE_URL,
                        is_preliminary=year > 2024,
                    )
                )

    session.commit()


def run_etl(db_path=None):
    """Run the OBR ETL pipeline."""
    from pathlib import Path
    from .schema import DEFAULT_DB_PATH

    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    engine = init_db(path)

    with Session(engine) as session:
        load_obr_targets(session)
        print(f"Loaded OBR projections to {path}")


if __name__ == "__main__":
    run_etl()
