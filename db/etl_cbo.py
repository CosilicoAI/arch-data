"""
ETL for CBO (Congressional Budget Office) projections.

Loads budget and economic projections from CBO's Budget and Economic Outlook.
Data source: https://www.cbo.gov/data/budget-economic-data
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

# CBO projections data (10-year baseline, 2024-2034)
# Source: CBO Budget and Economic Outlook, February 2024
# https://www.cbo.gov/publication/59710
# GDP and budget figures in billions of dollars
# Rates as percentages
CBO_DATA = {
    # Fiscal Year 2024
    2024: {
        # Budget (billions)
        "gdp": 28_269_000_000_000,  # $28.269 trillion
        "federal_revenue": 4_869_000_000_000,
        "federal_outlays": 6_751_000_000_000,
        "federal_deficit": 1_882_000_000_000,
        "debt_held_by_public": 28_166_000_000_000,
        # Economic indicators (rates as percentages)
        "unemployment_rate": 4.0,
        "cpi_inflation": 2.9,
        "interest_rate_10yr": 4.5,
        "real_gdp_growth": 2.0,
        # Labor force (thousands)
        "labor_force": 167_900_000,
    },
    2025: {
        "gdp": 29_458_000_000_000,
        "federal_revenue": 5_198_000_000_000,
        "federal_outlays": 7_025_000_000_000,
        "federal_deficit": 1_827_000_000_000,
        "debt_held_by_public": 30_065_000_000_000,
        "unemployment_rate": 4.4,
        "cpi_inflation": 2.3,
        "interest_rate_10yr": 4.2,
        "real_gdp_growth": 1.5,
        "labor_force": 168_500_000,
    },
    2026: {
        "gdp": 30_637_000_000_000,
        "federal_revenue": 5_551_000_000_000,
        "federal_outlays": 7_297_000_000_000,
        "federal_deficit": 1_746_000_000_000,
        "debt_held_by_public": 31_887_000_000_000,
        "unemployment_rate": 4.4,
        "cpi_inflation": 2.3,
        "interest_rate_10yr": 4.0,
        "real_gdp_growth": 2.1,
        "labor_force": 169_100_000,
    },
    2027: {
        "gdp": 31_854_000_000_000,
        "federal_revenue": 5_852_000_000_000,
        "federal_outlays": 7_556_000_000_000,
        "federal_deficit": 1_704_000_000_000,
        "debt_held_by_public": 33_662_000_000_000,
        "unemployment_rate": 4.3,
        "cpi_inflation": 2.3,
        "interest_rate_10yr": 3.9,
        "real_gdp_growth": 2.0,
        "labor_force": 169_600_000,
    },
    2028: {
        "gdp": 33_102_000_000_000,
        "federal_revenue": 6_131_000_000_000,
        "federal_outlays": 7_901_000_000_000,
        "federal_deficit": 1_770_000_000_000,
        "debt_held_by_public": 35_501_000_000_000,
        "unemployment_rate": 4.3,
        "cpi_inflation": 2.3,
        "interest_rate_10yr": 3.8,
        "real_gdp_growth": 2.0,
        "labor_force": 170_100_000,
    },
    2029: {
        "gdp": 34_388_000_000_000,
        "federal_revenue": 6_397_000_000_000,
        "federal_outlays": 8_241_000_000_000,
        "federal_deficit": 1_844_000_000_000,
        "debt_held_by_public": 37_413_000_000_000,
        "unemployment_rate": 4.3,
        "cpi_inflation": 2.3,
        "interest_rate_10yr": 3.8,
        "real_gdp_growth": 1.9,
        "labor_force": 170_500_000,
    },
    2030: {
        "gdp": 35_714_000_000_000,
        "federal_revenue": 6_667_000_000_000,
        "federal_outlays": 8_588_000_000_000,
        "federal_deficit": 1_921_000_000_000,
        "debt_held_by_public": 39_399_000_000_000,
        "unemployment_rate": 4.3,
        "cpi_inflation": 2.3,
        "interest_rate_10yr": 3.8,
        "real_gdp_growth": 1.9,
        "labor_force": 170_900_000,
    },
    2031: {
        "gdp": 37_080_000_000_000,
        "federal_revenue": 6_934_000_000_000,
        "federal_outlays": 8_960_000_000_000,
        "federal_deficit": 2_026_000_000_000,
        "debt_held_by_public": 41_488_000_000_000,
        "unemployment_rate": 4.3,
        "cpi_inflation": 2.3,
        "interest_rate_10yr": 3.8,
        "real_gdp_growth": 1.9,
        "labor_force": 171_200_000,
    },
    2032: {
        "gdp": 38_487_000_000_000,
        "federal_revenue": 7_212_000_000_000,
        "federal_outlays": 9_377_000_000_000,
        "federal_deficit": 2_165_000_000_000,
        "debt_held_by_public": 43_714_000_000_000,
        "unemployment_rate": 4.3,
        "cpi_inflation": 2.3,
        "interest_rate_10yr": 3.8,
        "real_gdp_growth": 1.8,
        "labor_force": 171_500_000,
    },
    2033: {
        "gdp": 39_934_000_000_000,
        "federal_revenue": 7_499_000_000_000,
        "federal_outlays": 9_822_000_000_000,
        "federal_deficit": 2_323_000_000_000,
        "debt_held_by_public": 46_096_000_000_000,
        "unemployment_rate": 4.3,
        "cpi_inflation": 2.3,
        "interest_rate_10yr": 3.8,
        "real_gdp_growth": 1.8,
        "labor_force": 171_700_000,
    },
    2034: {
        "gdp": 41_422_000_000_000,
        "federal_revenue": 7_796_000_000_000,
        "federal_outlays": 10_289_000_000_000,
        "federal_deficit": 2_493_000_000_000,
        "debt_held_by_public": 48_645_000_000_000,
        "unemployment_rate": 4.3,
        "cpi_inflation": 2.3,
        "interest_rate_10yr": 3.8,
        "real_gdp_growth": 1.8,
        "labor_force": 171_900_000,
    },
}

SOURCE_URL = "https://www.cbo.gov/data/budget-economic-data"


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


def load_cbo_targets(session: Session, years: list[int] | None = None):
    """
    Load CBO projections into database.

    Args:
        session: Database session
        years: Years to load (default: all available 2024-2034)
    """
    if years is None:
        years = list(CBO_DATA.keys())

    for year in years:
        if year not in CBO_DATA:
            continue

        data = CBO_DATA[year]

        # Federal budget stratum
        budget_stratum = get_or_create_stratum(
            session,
            name="US Federal Budget",
            jurisdiction=Jurisdiction.US_FEDERAL,
            constraints=[("sector", "==", "public")],  # Government sector
            description="US Federal Government budget",
            stratum_group_id="cbo_budget",
        )

        # Budget targets
        budget_vars = [
            ("gdp", TargetType.AMOUNT),
            ("federal_revenue", TargetType.AMOUNT),
            ("federal_outlays", TargetType.AMOUNT),
            ("federal_deficit", TargetType.AMOUNT),
            ("debt_held_by_public", TargetType.AMOUNT),
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
                    Target.source == DataSource.CBO,
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
                    source=DataSource.CBO,
                    source_url=SOURCE_URL,
                    is_preliminary=year > 2024,  # Projections are preliminary
                )
            )

        # US Economy stratum for macro indicators
        economy_stratum = get_or_create_stratum(
            session,
            name="US Economy",
            jurisdiction=Jurisdiction.US,
            constraints=[],  # Whole economy, not a subset
            description="US macroeconomic indicators",
            stratum_group_id="cbo_economy",
        )

        # Economic rate targets
        rate_vars = [
            "unemployment_rate",
            "cpi_inflation",
            "interest_rate_10yr",
            "real_gdp_growth",
        ]

        for var_name in rate_vars:
            if var_name not in data:
                continue

            existing = session.exec(
                select(Target).where(
                    Target.stratum_id == economy_stratum.id,
                    Target.variable == var_name,
                    Target.period == year,
                    Target.source == DataSource.CBO,
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
                    source=DataSource.CBO,
                    source_url=SOURCE_URL,
                    is_preliminary=year > 2024,
                )
            )

        # Labor force count
        if "labor_force" in data:
            existing = session.exec(
                select(Target).where(
                    Target.stratum_id == economy_stratum.id,
                    Target.variable == "labor_force",
                    Target.period == year,
                    Target.source == DataSource.CBO,
                )
            ).first()

            if not existing:
                session.add(
                    Target(
                        stratum_id=economy_stratum.id,
                        variable="labor_force",
                        period=year,
                        value=data["labor_force"],
                        target_type=TargetType.COUNT,
                        source=DataSource.CBO,
                        source_url=SOURCE_URL,
                        is_preliminary=year > 2024,
                    )
                )

    session.commit()


def run_etl(db_path=None):
    """Run the CBO ETL pipeline."""
    from pathlib import Path
    from .schema import DEFAULT_DB_PATH

    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    engine = init_db(path)

    with Session(engine) as session:
        load_cbo_targets(session)
        print(f"Loaded CBO projections to {path}")


if __name__ == "__main__":
    run_etl()
