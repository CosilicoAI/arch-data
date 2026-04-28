"""
ETL for BLS (Bureau of Labor Statistics) employment targets.

Loads data from BLS employment statistics into the targets database.
Data sources:
- https://www.bls.gov/cps/
- https://www.bls.gov/oes/
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

# BLS employment data by year
# Source: Current Population Survey (CPS) and Current Employment Statistics (CES)
# https://www.bls.gov/cps/tables.htm
BLS_DATA = {
    2023: {
        # Labor force (in thousands, converted to actual)
        "civilian_labor_force": 167_300_000,
        "employed": 161_100_000,
        "unemployed": 6_200_000,
        "not_in_labor_force": 99_800_000,

        # Rates (percentages stored as decimals)
        "unemployment_rate": 3.7,  # percent
        "labor_force_participation_rate": 62.6,  # percent
        "employment_population_ratio": 60.3,  # percent

        # Earnings
        "median_weekly_earnings": 1_145,  # dollars, full-time workers
        "median_hourly_wage": 23.40,  # dollars, all workers

        # By sector (nonfarm payroll, thousands)
        "total_nonfarm_employment": 157_000_000,
        "private_employment": 133_600_000,
        "government_employment": 23_400_000,
    },
    2022: {
        "civilian_labor_force": 164_300_000,
        "employed": 158_300_000,
        "unemployed": 6_000_000,
        "not_in_labor_force": 99_600_000,

        "unemployment_rate": 3.6,
        "labor_force_participation_rate": 62.2,
        "employment_population_ratio": 59.9,

        "median_weekly_earnings": 1_085,
        "median_hourly_wage": 22.00,

        "total_nonfarm_employment": 153_000_000,
        "private_employment": 130_200_000,
        "government_employment": 22_800_000,
    },
    2021: {
        "civilian_labor_force": 161_200_000,
        "employed": 152_600_000,
        "unemployed": 8_600_000,
        "not_in_labor_force": 100_000_000,

        "unemployment_rate": 5.4,
        "labor_force_participation_rate": 61.7,
        "employment_population_ratio": 58.4,

        "median_weekly_earnings": 1_013,
        "median_hourly_wage": 20.50,

        "total_nonfarm_employment": 146_100_000,
        "private_employment": 124_200_000,
        "government_employment": 21_900_000,
    },
}

SOURCE_URL = "https://www.bls.gov/cps/"


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


def load_bls_targets(session: Session, years: list[int] | None = None):
    """
    Load BLS employment targets into database.

    Args:
        session: Database session
        years: Years to load (default: all available)
    """
    if years is None:
        years = list(BLS_DATA.keys())

    for year in years:
        if year not in BLS_DATA:
            continue

        data = BLS_DATA[year]

        # Create national labor force stratum
        labor_force_stratum = get_or_create_stratum(
            session,
            name="US Labor Force",
            jurisdiction=Jurisdiction.US,
            constraints=[("in_labor_force", "==", "1")],
            description="US civilian labor force (16+)",
            stratum_group_id="bls_national",
        )

        # Employment count
        session.add(
            Target(
                stratum_id=labor_force_stratum.id,
                variable="employed",
                period=year,
                value=data["employed"],
                target_type=TargetType.COUNT,
                source=DataSource.BLS,
                source_url=SOURCE_URL,
            )
        )

        # Unemployment count
        session.add(
            Target(
                stratum_id=labor_force_stratum.id,
                variable="unemployed",
                period=year,
                value=data["unemployed"],
                target_type=TargetType.COUNT,
                source=DataSource.BLS,
                source_url=SOURCE_URL,
            )
        )

        # Unemployment rate
        session.add(
            Target(
                stratum_id=labor_force_stratum.id,
                variable="unemployment_rate",
                period=year,
                value=data["unemployment_rate"],
                target_type=TargetType.RATE,
                source=DataSource.BLS,
                source_url=SOURCE_URL,
            )
        )

        # Labor force participation rate
        session.add(
            Target(
                stratum_id=labor_force_stratum.id,
                variable="labor_force_participation_rate",
                period=year,
                value=data["labor_force_participation_rate"],
                target_type=TargetType.RATE,
                source=DataSource.BLS,
                source_url=SOURCE_URL,
            )
        )

        # Median weekly earnings
        session.add(
            Target(
                stratum_id=labor_force_stratum.id,
                variable="median_weekly_earnings",
                period=year,
                value=data["median_weekly_earnings"],
                target_type=TargetType.AMOUNT,
                source=DataSource.BLS,
                source_url=SOURCE_URL,
            )
        )

        # Total labor force count
        session.add(
            Target(
                stratum_id=labor_force_stratum.id,
                variable="labor_force_count",
                period=year,
                value=data["civilian_labor_force"],
                target_type=TargetType.COUNT,
                source=DataSource.BLS,
                source_url=SOURCE_URL,
            )
        )

    session.commit()


def run_etl(db_path=None):
    """Run the BLS ETL pipeline."""
    from pathlib import Path
    from .schema import DEFAULT_DB_PATH

    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    engine = init_db(path)

    with Session(engine) as session:
        load_bls_targets(session)
        print(f"Loaded BLS targets to {path}")


if __name__ == "__main__":
    run_etl()
