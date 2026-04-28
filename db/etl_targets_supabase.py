"""
ETL for loading calibration targets to Supabase.

Migrates targets from local SQLite to Supabase microplex schema.
Handles strata and stratum_constraints alongside targets.

Usage:
    python -m db.etl_targets_supabase --source us
    python -m db.etl_targets_supabase --all
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .supabase_client import get_supabase_client, query_sources


def get_or_create_stratum(
    client,
    name: str,
    jurisdiction: str,
    constraints: List[Dict[str, str]],
    description: Optional[str] = None,
) -> str:
    """
    Get existing stratum or create new one in Supabase.

    Args:
        client: Supabase client
        name: Stratum name (must be unique per jurisdiction)
        jurisdiction: e.g., "us", "uk"
        constraints: List of {variable, operator, value} dicts
        description: Optional description

    Returns:
        Stratum UUID
    """
    # Check if exists by name + jurisdiction (unique constraint)
    result = client.schema("microplex").table("strata").select("id").eq("name", name).eq("jurisdiction", jurisdiction).execute()
    if result.data:
        return result.data[0]["id"]

    # Create new stratum
    stratum_data = {
        "name": name,
        "jurisdiction": jurisdiction,
    }
    if description:
        stratum_data["description"] = description

    result = client.schema("microplex").table("strata").insert(stratum_data).execute()
    stratum_id = result.data[0]["id"]

    # Add constraints
    for constraint in constraints:
        constraint_data = {
            "stratum_id": stratum_id,
            "variable": constraint["variable"],
            "operator": constraint["operator"],
            "value": constraint["value"],
        }
        client.schema("microplex").table("stratum_constraints").insert(constraint_data).execute()

    return stratum_id


def load_soi_targets_supabase(
    years: Optional[List[int]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Load IRS SOI targets to Supabase.

    Args:
        years: Years to load (default: 2018-2021)
        dry_run: If True, just return count without inserting

    Returns:
        Dict with counts and status
    """
    from .etl_soi import SOURCE_URL, available_soi_years, load_soi_table_1_1_data

    if years is None:
        years = available_soi_years()

    client = get_supabase_client()

    # Get or create IRS SOI source
    sources = query_sources(jurisdiction="us", institution="irs")
    soi_source = next((s for s in sources if s["dataset"] == "soi"), None)

    if not soi_source:
        result = client.schema("microplex").table("sources").insert({
            "jurisdiction": "us",
            "institution": "irs",
            "dataset": "soi",
            "name": "IRS Statistics of Income",
            "url": SOURCE_URL,
        }).execute()
        source_id = result.data[0]["id"]
    else:
        source_id = soi_source["id"]

    targets_loaded = 0
    strata_created = 0

    for year in years:
        if year not in available_soi_years():
            continue

        year_data = load_soi_table_1_1_data(year)

        # National totals
        if dry_run:
            targets_loaded += 2  # total_returns and total_agi
        else:
            stratum_id = get_or_create_stratum(
                client,
                name=f"US Tax Filers {year}",
                jurisdiction="us",
                constraints=[],
            )
            strata_created += 1

            client.schema("microplex").table("targets").insert({
                "source_id": source_id,
                "stratum_id": stratum_id,
                "variable": "tax_unit_count",
                "value": year_data["total_returns"],
                "target_type": "count",
                "period": year,
            }).execute()
            targets_loaded += 1

            client.schema("microplex").table("targets").insert({
                "source_id": source_id,
                "stratum_id": stratum_id,
                "variable": "agi_total",
                "value": year_data["total_agi"],
                "target_type": "amount",
                "period": year,
            }).execute()
            targets_loaded += 1

        # AGI bracket targets
        returns_by_bracket = year_data.get("returns_by_agi_bracket", {})
        agi_by_bracket = year_data.get("agi_by_bracket", {})

        for bracket, returns in returns_by_bracket.items():
            if dry_run:
                targets_loaded += 2
                continue

            stratum_name = f"US Tax Filers AGI {bracket} {year}"
            stratum_id = get_or_create_stratum(
                client,
                name=stratum_name,
                jurisdiction="us",
                constraints=[{"variable": "agi_bracket", "operator": "==", "value": bracket}],
            )
            strata_created += 1

            client.schema("microplex").table("targets").insert({
                "source_id": source_id,
                "stratum_id": stratum_id,
                "variable": "tax_unit_count",
                "value": returns,
                "target_type": "count",
                "period": year,
            }).execute()
            targets_loaded += 1

            if bracket in agi_by_bracket:
                client.schema("microplex").table("targets").insert({
                    "source_id": source_id,
                    "stratum_id": stratum_id,
                    "variable": "agi_total",
                    "value": agi_by_bracket[bracket],
                    "target_type": "amount",
                    "period": year,
                }).execute()
                targets_loaded += 1

    return {
        "targets_loaded": targets_loaded,
        "strata_created": strata_created,
        "years": years,
        "dry_run": dry_run,
    }


def load_snap_targets_supabase(
    years: Optional[List[int]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Load USDA SNAP targets to Supabase."""
    from .etl_snap import SNAP_DATA, SOURCE_URL

    if years is None:
        years = list(SNAP_DATA.keys())

    client = get_supabase_client()

    # Get or create USDA SNAP source
    sources = query_sources(jurisdiction="us", institution="usda")
    snap_source = next((s for s in sources if s["dataset"] == "snap"), None)

    if not snap_source:
        result = client.schema("microplex").table("sources").insert({
            "jurisdiction": "us",
            "institution": "usda",
            "dataset": "snap",
            "name": "USDA SNAP Statistics",
            "url": SOURCE_URL,
        }).execute()
        source_id = result.data[0]["id"]
    else:
        source_id = snap_source["id"]

    targets_loaded = 0

    for year in years:
        if year not in SNAP_DATA:
            continue

        data = SNAP_DATA[year]

        if dry_run:
            targets_loaded += 3  # participants, households, benefits
            continue

        # National SNAP stratum
        stratum_id = get_or_create_stratum(
            client,
            name=f"US SNAP Recipients {year}",
            jurisdiction="us",
            constraints=[{"variable": "snap_participation", "operator": "==", "value": "1"}],
        )

        # Participants
        client.schema("microplex").table("targets").insert({
            "source_id": source_id,
            "stratum_id": stratum_id,
            "variable": "snap_participants",
            "value": data["participants"],
            "target_type": "count",
            "period": year,
        }).execute()
        targets_loaded += 1

        # Households
        client.schema("microplex").table("targets").insert({
            "source_id": source_id,
            "stratum_id": stratum_id,
            "variable": "snap_households",
            "value": data["households"],
            "target_type": "count",
            "period": year,
        }).execute()
        targets_loaded += 1

        # Benefits
        client.schema("microplex").table("targets").insert({
            "source_id": source_id,
            "stratum_id": stratum_id,
            "variable": "snap_benefits_total",
            "value": data["benefits"],
            "target_type": "amount",
            "period": year,
        }).execute()
        targets_loaded += 1

    return {
        "targets_loaded": targets_loaded,
        "years": years,
        "dry_run": dry_run,
    }


def load_all_targets_supabase(dry_run: bool = False) -> Dict[str, Any]:
    """Load all available targets to Supabase."""
    results = {}

    print("Loading SOI targets...")
    results["soi"] = load_soi_targets_supabase(dry_run=dry_run)
    print(f"  Loaded {results['soi']['targets_loaded']} SOI targets")

    print("Loading SNAP targets...")
    results["snap"] = load_snap_targets_supabase(dry_run=dry_run)
    print(f"  Loaded {results['snap']['targets_loaded']} SNAP targets")

    total = sum(r["targets_loaded"] for r in results.values())
    results["total"] = total
    print(f"Total targets loaded: {total}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Load targets to Supabase")
    parser.add_argument("--source", type=str, choices=["soi", "snap", "all"], default="all")
    parser.add_argument("--years", type=int, nargs="+", help="Years to load")
    parser.add_argument("--dry-run", action="store_true", help="Show stats without inserting")
    args = parser.parse_args()

    if args.source == "all":
        load_all_targets_supabase(dry_run=args.dry_run)
    elif args.source == "soi":
        result = load_soi_targets_supabase(years=args.years, dry_run=args.dry_run)
        print(f"Loaded {result['targets_loaded']} SOI targets")
    elif args.source == "snap":
        result = load_snap_targets_supabase(years=args.years, dry_run=args.dry_run)
        print(f"Loaded {result['targets_loaded']} SNAP targets")


if __name__ == "__main__":
    main()
