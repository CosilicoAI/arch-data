"""CLI for managing Arch target input data."""

import argparse
import os
from pathlib import Path

from sqlmodel import Session, select

from .etl_soi import load_soi_targets
from .schema import DEFAULT_DB_PATH, Stratum, Target, init_db, get_engine


def cmd_init(args):
    """Initialize the database."""
    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH
    init_db(db_path)
    print(f"Initialized database at {db_path}")


def cmd_load(args):
    """Load targets from a source."""
    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH
    engine = init_db(db_path)

    with Session(engine) as session:
        if args.source == "soi" or args.source == "all":
            years = [int(y) for y in args.years.split(",")] if args.years else None
            load_soi_targets(session, years=years)
            print(f"Loaded SOI targets for years: {years or 'all available'}")

        if args.source == "soi-state" or args.source == "all":
            from .etl_soi_state import load_soi_state_targets
            years = [int(y) for y in args.years.split(",")] if args.years else None
            load_soi_state_targets(session, years=years)
            print(f"Loaded state-level SOI targets for years: {years or 'all available'}")

        if args.source == "soi-credits" or args.source == "all":
            from .etl_soi_credits import load_soi_credits_targets
            years = [int(y) for y in args.years.split(",")] if args.years else None
            load_soi_credits_targets(session, years=years)
            print(f"Loaded SOI credits targets for years: {years or 'all available'}")

        if args.source == "soi-income-sources" or args.source == "all":
            from .etl_soi_income_sources import load_soi_income_sources_targets
            years = [int(y) for y in args.years.split(",")] if args.years else None
            load_soi_income_sources_targets(session, years=years)
            print(f"Loaded SOI income sources targets for years: {years or 'all available'}")

        if args.source == "soi-deductions" or args.source == "all":
            from .etl_soi_deductions import load_soi_deductions_targets
            years = [int(y) for y in args.years.split(",")] if args.years else None
            load_soi_deductions_targets(session, years=years)
            print(f"Loaded SOI deductions targets for years: {years or 'all available'}")

        if args.source == "snap" or args.source == "all":
            from .etl_snap import load_snap_targets
            years = [int(y) for y in args.years.split(",")] if args.years else None
            load_snap_targets(session, years=years)
            print(f"Loaded SNAP targets for years: {years or 'all available'}")

        if args.source == "hmrc" or args.source == "all":
            from .etl_hmrc import load_hmrc_targets
            years = [int(y) for y in args.years.split(",")] if args.years else None
            load_hmrc_targets(session, years=years)
            print(f"Loaded HMRC targets for years: {years or 'all available'}")

        if args.source == "census" or args.source == "all":
            from .etl_census import load_census_targets
            years = [int(y) for y in args.years.split(",")] if args.years else None
            load_census_targets(session, years=years)
            print(f"Loaded Census targets for years: {years or 'all available'}")

        if args.source == "ssa" or args.source == "all":
            from .etl_ssa import load_ssa_targets
            years = [int(y) for y in args.years.split(",")] if args.years else None
            load_ssa_targets(session, years=years)
            print(f"Loaded SSA targets for years: {years or 'all available'}")

        if args.source == "bls" or args.source == "all":
            from .etl_bls import load_bls_targets
            years = [int(y) for y in args.years.split(",")] if args.years else None
            load_bls_targets(session, years=years)
            print(f"Loaded BLS targets for years: {years or 'all available'}")

        if args.source == "cps" or args.source == "all":
            from .etl_cps import load_cps_targets
            years = [int(y) for y in args.years.split(",")] if args.years else None
            load_cps_targets(session, years=years)
            print(f"Loaded CPS monthly targets for years: {years or 'all available'}")

        if args.source == "cbo" or args.source == "all":
            from .etl_cbo import load_cbo_targets
            years = [int(y) for y in args.years.split(",")] if args.years else None
            load_cbo_targets(session, years=years)
            print(f"Loaded CBO projections for years: {years or 'all available'}")

        if args.source == "obr" or args.source == "all":
            from .etl_obr import load_obr_targets
            years = [int(y) for y in args.years.split(",")] if args.years else None
            load_obr_targets(session, years=years)
            print(f"Loaded OBR projections for years: {years or 'all available'}")

        if args.source == "ons" or args.source == "all":
            from .etl_ons import load_ons_targets
            years = [int(y) for y in args.years.split(",")] if args.years else None
            load_ons_targets(session, years=years)
            print(f"Loaded ONS projections for years: {years or 'all available'}")


def cmd_stats(args):
    """Show database statistics."""
    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH

    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return

    engine = get_engine(db_path)
    with Session(engine) as session:
        strata_count = len(session.exec(select(Stratum)).all())
        targets_count = len(session.exec(select(Target)).all())

        # Get unique sources
        sources = session.exec(
            select(Target.source).distinct()
        ).all()

        # Get unique years
        years = session.exec(
            select(Target.period).distinct()
        ).all()

        print(f"Database: {db_path}")
        print(f"Strata: {strata_count}")
        print(f"Targets: {targets_count}")
        print(f"Sources: {', '.join(str(s) for s in sources)}")
        print(f"Years: {', '.join(str(y) for y in sorted(years))}")


def cmd_query(args):
    """Query targets."""
    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH

    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return

    engine = get_engine(db_path)
    with Session(engine) as session:
        query = select(Target, Stratum).join(Stratum)

        if args.variable:
            query = query.where(Target.variable == args.variable)
        if args.year:
            query = query.where(Target.period == int(args.year))
        if args.source:
            query = query.where(Target.source == args.source)

        results = session.exec(query).all()

        if not results:
            print("No targets found matching criteria")
            return

        print(f"{'Stratum':<30} {'Variable':<25} {'Year':<6} {'Value':>15} {'Source':<10}")
        print("-" * 90)
        for target, stratum in results[:args.limit]:
            print(f"{stratum.name:<30} {target.variable:<25} {target.period:<6} {target.value:>15,.0f} {target.source.value:<10}")


def main():
    parser = argparse.ArgumentParser(
        description="Manage Arch target input data"
    )
    parser.add_argument(
        "--db",
        help=f"Database path (default: {DEFAULT_DB_PATH})"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    init_parser = subparsers.add_parser("init", help="Initialize database")
    init_parser.set_defaults(func=cmd_init)

    # load
    load_parser = subparsers.add_parser("load", help="Load targets from source")
    load_parser.add_argument(
        "source",
        choices=["soi", "soi-state", "soi-credits", "soi-income-sources", "soi-deductions", "snap", "hmrc", "census", "ssa", "bls", "cps", "cbo", "obr", "ons", "all"],
        help="Data source to load"
    )
    load_parser.add_argument(
        "--years",
        help="Comma-separated years to load (default: all)"
    )
    load_parser.set_defaults(func=cmd_load)

    # stats
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.set_defaults(func=cmd_stats)

    # query
    query_parser = subparsers.add_parser("query", help="Query targets")
    query_parser.add_argument("--variable", help="Filter by variable name")
    query_parser.add_argument("--year", help="Filter by year")
    query_parser.add_argument("--source", help="Filter by source")
    query_parser.add_argument("--limit", type=int, default=20, help="Max results")
    query_parser.set_defaults(func=cmd_query)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
