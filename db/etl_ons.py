"""
ETL for ONS (Office for National Statistics) projections.

Loads UK population and household projections from ONS.
Data sources:
- National population projections: https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationprojections
- Household projections: https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationprojections/datasets/householdprojectionsforengland
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

# ONS 2022-based projections data (2024-2034)
# Source: ONS National Population Projections (2022-based)
# https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationprojections
# Population counts are for the UK as a whole
# Household projections based on ONS household projections data
ONS_DATA = {
    # Mid-2024 projections
    2024: {
        # Population by broad age groups (counts)
        "population_total": 68_265_000,
        "population_age_0_15": 12_410_000,  # Children
        "population_age_16_64": 42_638_000,  # Working age
        "population_age_65_plus": 13_217_000,  # Pension age
        # Household projections
        "households_total": 28_500_000,
        "average_household_size": 2.40,
    },
    2025: {
        "population_total": 68_626_000,
        "population_age_0_15": 12_448_000,
        "population_age_16_64": 42_703_000,
        "population_age_65_plus": 13_475_000,
        "households_total": 28_750_000,
        "average_household_size": 2.39,
    },
    2026: {
        "population_total": 68_978_000,
        "population_age_0_15": 12_485_000,
        "population_age_16_64": 42_759_000,
        "population_age_65_plus": 13_734_000,
        "households_total": 29_000_000,
        "average_household_size": 2.38,
    },
    2027: {
        "population_total": 69_321_000,
        "population_age_0_15": 12_521_000,
        "population_age_16_64": 42_808_000,
        "population_age_65_plus": 13_992_000,
        "households_total": 29_250_000,
        "average_household_size": 2.37,
    },
    2028: {
        "population_total": 69_656_000,
        "population_age_0_15": 12_555_000,
        "population_age_16_64": 42_852_000,
        "population_age_65_plus": 14_249_000,
        "households_total": 29_500_000,
        "average_household_size": 2.36,
    },
    2029: {
        "population_total": 69_984_000,
        "population_age_0_15": 12_588_000,
        "population_age_16_64": 42_891_000,
        "population_age_65_plus": 14_505_000,
        "households_total": 29_750_000,
        "average_household_size": 2.35,
    },
    2030: {
        "population_total": 70_306_000,
        "population_age_0_15": 12_619_000,
        "population_age_16_64": 42_927_000,
        "population_age_65_plus": 14_760_000,
        "households_total": 30_000_000,
        "average_household_size": 2.34,
    },
    2031: {
        "population_total": 70_622_000,
        "population_age_0_15": 12_649_000,
        "population_age_16_64": 42_960_000,
        "population_age_65_plus": 15_013_000,
        "households_total": 30_250_000,
        "average_household_size": 2.34,
    },
    2032: {
        "population_total": 70_933_000,
        "population_age_0_15": 12_678_000,
        "population_age_16_64": 42_991_000,
        "population_age_65_plus": 15_264_000,
        "households_total": 30_500_000,
        "average_household_size": 2.33,
    },
    2033: {
        "population_total": 71_240_000,
        "population_age_0_15": 12_706_000,
        "population_age_16_64": 43_020_000,
        "population_age_65_plus": 15_514_000,
        "households_total": 30_750_000,
        "average_household_size": 2.32,
    },
    2034: {
        "population_total": 71_543_000,
        "population_age_0_15": 12_733_000,
        "population_age_16_64": 43_047_000,
        "population_age_65_plus": 15_763_000,
        "households_total": 31_000_000,
        "average_household_size": 2.31,
    },
}

SOURCE_URL = "https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationprojections"


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


def load_ons_targets(session: Session, years: list[int] | None = None):
    """
    Load ONS population and household projections into database.

    Args:
        session: Database session
        years: Years to load (default: all available 2024-2034)
    """
    if years is None:
        years = list(ONS_DATA.keys())

    for year in years:
        if year not in ONS_DATA:
            continue

        data = ONS_DATA[year]

        # UK population stratum
        population_stratum = get_or_create_stratum(
            session,
            name="UK Population",
            jurisdiction=Jurisdiction.UK,
            constraints=[("entity_type", "==", "person")],  # Person-level data
            description="UK total population and age group breakdowns",
            stratum_group_id="ons_population",
        )

        # Population count targets
        population_vars = [
            ("population_total", TargetType.COUNT),
            ("population_age_0_15", TargetType.COUNT),
            ("population_age_16_64", TargetType.COUNT),
            ("population_age_65_plus", TargetType.COUNT),
        ]

        for var_name, var_type in population_vars:
            if var_name not in data:
                continue

            # Check for existing target
            existing = session.exec(
                select(Target).where(
                    Target.stratum_id == population_stratum.id,
                    Target.variable == var_name,
                    Target.period == year,
                    Target.source == DataSource.ONS,
                )
            ).first()

            if existing:
                continue

            session.add(
                Target(
                    stratum_id=population_stratum.id,
                    variable=var_name,
                    period=year,
                    value=data[var_name],
                    target_type=var_type,
                    source=DataSource.ONS,
                    source_url=SOURCE_URL,
                    is_preliminary=True,  # All projections are preliminary
                )
            )

        # UK households stratum
        households_stratum = get_or_create_stratum(
            session,
            name="UK Households",
            jurisdiction=Jurisdiction.UK,
            constraints=[("entity_type", "==", "household")],  # Household-level data
            description="UK household projections",
            stratum_group_id="ons_households",
        )

        # Household count targets
        if "households_total" in data:
            existing = session.exec(
                select(Target).where(
                    Target.stratum_id == households_stratum.id,
                    Target.variable == "households_total",
                    Target.period == year,
                    Target.source == DataSource.ONS,
                )
            ).first()

            if not existing:
                session.add(
                    Target(
                        stratum_id=households_stratum.id,
                        variable="households_total",
                        period=year,
                        value=data["households_total"],
                        target_type=TargetType.COUNT,
                        source=DataSource.ONS,
                        source_url=SOURCE_URL,
                        is_preliminary=True,
                    )
                )

        # Average household size (rate)
        if "average_household_size" in data:
            existing = session.exec(
                select(Target).where(
                    Target.stratum_id == households_stratum.id,
                    Target.variable == "average_household_size",
                    Target.period == year,
                    Target.source == DataSource.ONS,
                )
            ).first()

            if not existing:
                session.add(
                    Target(
                        stratum_id=households_stratum.id,
                        variable="average_household_size",
                        period=year,
                        value=data["average_household_size"],
                        target_type=TargetType.RATE,
                        source=DataSource.ONS,
                        source_url=SOURCE_URL,
                        is_preliminary=True,
                    )
                )

    session.commit()


def run_etl(db_path=None):
    """Run the ONS ETL pipeline."""
    from pathlib import Path
    from .schema import DEFAULT_DB_PATH

    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    engine = init_db(path)

    with Session(engine) as session:
        load_ons_targets(session)
        print(f"Loaded ONS projections to {path}")


if __name__ == "__main__":
    run_etl()
