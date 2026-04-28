"""
ETL for BLS CPS (Current Population Survey) monthly employment targets.

Loads monthly employment status, unemployment rates, and labor force participation
data from the Bureau of Labor Statistics Current Population Survey.

Data sources:
- https://www.bls.gov/cps/tables.htm
- https://www.bls.gov/news.release/empsit.htm
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

# CPS monthly employment data
# Source: Current Population Survey Employment Situation Reports
# https://www.bls.gov/cps/tables.htm
# Format: {year: {month: {metrics}}}
# Values in thousands converted to actual counts, rates as percentages

CPS_MONTHLY_DATA = {
    2025: {
        11: {  # November 2025
            "civilian_labor_force": 168_500_000,
            "employed": 160_800_000,
            "unemployed": 7_700_000,
            "not_in_labor_force": 101_200_000,
            "unemployment_rate": 4.6,
            "labor_force_participation_rate": 62.5,
            "employment_population_ratio": 59.6,
        },
        9: {  # September 2025
            "civilian_labor_force": 168_600_000,
            "employed": 161_000_000,
            "unemployed": 7_600_000,
            "not_in_labor_force": 101_000_000,
            "unemployment_rate": 4.5,
            "labor_force_participation_rate": 62.5,
            "employment_population_ratio": 59.7,
        },
    },
    2024: {
        12: {  # December 2024
            "civilian_labor_force": 168_300_000,
            "employed": 161_300_000,
            "unemployed": 7_000_000,
            "not_in_labor_force": 100_500_000,
            "unemployment_rate": 4.2,
            "labor_force_participation_rate": 62.6,
            "employment_population_ratio": 60.0,
        },
        11: {  # November 2024
            "civilian_labor_force": 168_100_000,
            "employed": 161_100_000,
            "unemployed": 7_000_000,
            "not_in_labor_force": 100_400_000,
            "unemployment_rate": 4.2,
            "labor_force_participation_rate": 62.6,
            "employment_population_ratio": 60.0,
        },
        10: {  # October 2024
            "civilian_labor_force": 167_900_000,
            "employed": 160_800_000,
            "unemployed": 7_100_000,
            "not_in_labor_force": 100_300_000,
            "unemployment_rate": 4.2,
            "labor_force_participation_rate": 62.6,
            "employment_population_ratio": 60.0,
        },
        9: {  # September 2024
            "civilian_labor_force": 167_700_000,
            "employed": 160_500_000,
            "unemployed": 7_200_000,
            "not_in_labor_force": 100_200_000,
            "unemployment_rate": 4.3,
            "labor_force_participation_rate": 62.6,
            "employment_population_ratio": 59.9,
        },
    },
    2023: {
        12: {  # December 2023
            "civilian_labor_force": 167_300_000,
            "employed": 161_100_000,
            "unemployed": 6_200_000,
            "not_in_labor_force": 99_800_000,
            "unemployment_rate": 3.7,
            "labor_force_participation_rate": 62.6,
            "employment_population_ratio": 60.3,
        },
        6: {  # June 2023
            "civilian_labor_force": 166_700_000,
            "employed": 160_400_000,
            "unemployed": 6_300_000,
            "not_in_labor_force": 99_500_000,
            "unemployment_rate": 3.8,
            "labor_force_participation_rate": 62.6,
            "employment_population_ratio": 60.2,
        },
        1: {  # January 2023
            "civilian_labor_force": 165_900_000,
            "employed": 159_500_000,
            "unemployed": 6_400_000,
            "not_in_labor_force": 99_300_000,
            "unemployment_rate": 3.9,
            "labor_force_participation_rate": 62.4,
            "employment_population_ratio": 60.0,
        },
    },
}

SOURCE_URL = "https://www.bls.gov/cps/tables.htm"


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


def load_cps_targets(session: Session, years: list[int] | None = None, months: list[int] | None = None):
    """
    Load CPS monthly employment targets into database.

    Args:
        session: Database session
        years: Years to load (default: all available)
        months: Months to load (default: all available for selected years)
    """
    if years is None:
        years = list(CPS_MONTHLY_DATA.keys())

    for year in years:
        if year not in CPS_MONTHLY_DATA:
            continue

        year_data = CPS_MONTHLY_DATA[year]
        months_to_load = months if months is not None else list(year_data.keys())

        for month in months_to_load:
            if month not in year_data:
                continue

            data = year_data[month]

            # Period as YYYYMM format (e.g., 202311 for November 2023)
            period = year * 100 + month

            # Create national labor force stratum for this month
            labor_force_stratum = get_or_create_stratum(
                session,
                name="US Labor Force",
                jurisdiction=Jurisdiction.US,
                constraints=[("in_labor_force", "==", "1")],
                description="US civilian labor force (16+)",
                stratum_group_id="cps_monthly",
            )

            # Employment count
            session.add(
                Target(
                    stratum_id=labor_force_stratum.id,
                    variable="employed",
                    period=period,
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
                    period=period,
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
                    period=period,
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
                    period=period,
                    value=data["labor_force_participation_rate"],
                    target_type=TargetType.RATE,
                    source=DataSource.BLS,
                    source_url=SOURCE_URL,
                )
            )

            # Employment-population ratio
            session.add(
                Target(
                    stratum_id=labor_force_stratum.id,
                    variable="employment_population_ratio",
                    period=period,
                    value=data["employment_population_ratio"],
                    target_type=TargetType.RATE,
                    source=DataSource.BLS,
                    source_url=SOURCE_URL,
                )
            )

            # Total labor force count
            session.add(
                Target(
                    stratum_id=labor_force_stratum.id,
                    variable="labor_force_count",
                    period=period,
                    value=data["civilian_labor_force"],
                    target_type=TargetType.COUNT,
                    source=DataSource.BLS,
                    source_url=SOURCE_URL,
                )
            )

            # Not in labor force count
            session.add(
                Target(
                    stratum_id=labor_force_stratum.id,
                    variable="not_in_labor_force",
                    period=period,
                    value=data["not_in_labor_force"],
                    target_type=TargetType.COUNT,
                    source=DataSource.BLS,
                    source_url=SOURCE_URL,
                )
            )

    session.commit()


def run_etl(db_path=None):
    """Run the CPS ETL pipeline."""
    from pathlib import Path
    from .schema import DEFAULT_DB_PATH

    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    engine = init_db(path)

    with Session(engine) as session:
        load_cps_targets(session)
        print(f"Loaded CPS monthly targets to {path}")


if __name__ == "__main__":
    run_etl()
