"""
ETL for loading raw CPS ASEC microdata to Supabase.

Loads person, household, and family tables from locally cached parquet files
to the microplex schema in Supabase.

Table naming: microplex.us_census_cps_asec_{year}_{table_type}

Usage:
    python -m db.etl_cps_raw --year 2024
    python -m db.etl_cps_raw --year 2024 --dry-run
    python -m db.etl_cps_raw --year 2024 --method csv  # Export to CSV for dashboard import
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .supabase_client import get_supabase_client, get_table_name, register_dataset


# Path to local raw cache
CACHE_DIR = Path(__file__).parent.parent / "micro" / "us" / "census" / "raw_cache"


def get_raw_cache_dir(year: int) -> Path:
    """Get the directory for cached raw CPS data."""
    return CACHE_DIR / f"census_cps_{year}"


def get_cps_table_names(year: int) -> Dict[str, str]:
    """Get Supabase table names for CPS ASEC tables."""
    return {
        "person": get_table_name("us", "census", "cps_asec", year, "person"),
        "household": get_table_name("us", "census", "cps_asec", year, "household"),
        "family": get_table_name("us", "census", "cps_asec", year, "family"),
    }


# Key columns to extract as typed columns (rest goes in raw_data JSONB)
PERSON_KEY_COLUMNS = {
    "PH_SEQ": "ph_seq",
    "PPPOS": "pppos",
    "A_AGE": "a_age",
    "A_SEX": "a_sex",
    "PRDTRACE": "prdtrace",
    "PEHSPNON": "pehspnon",
    "PRCITSHP": "prcitshp",
    "GESTFIPS": "gestfips",
    "GTCO": "gtco",
    "PTOTVAL": "ptotval",
    "PEARNVAL": "pearnval",
    "WSAL_VAL": "wsal_val",
    "SEMP_VAL": "semp_val",
    "MARSUPWT": "marsupwt",
}

HOUSEHOLD_KEY_COLUMNS = {
    "H_SEQ": "h_seq",
    "H_NUMPER": "h_numper",
    "HH5TO18": "hh5to18",
    "HUNDER18": "hunder18",
    "GESTFIPS": "gestfips",
    "GTCO": "gtco",
    "HTOTVAL": "htotval",
    "HPROP_VAL": "hprop_val",
    "HSUP_WGT": "hsup_wgt",
}

FAMILY_KEY_COLUMNS = {
    "FH_SEQ": "fh_seq",
    "FFPOS": "ffpos",
    "FKIND": "fkind",
    "F_MV_FS": "f_mv_fs",
    "FHIP_VAL": "fhip_val",
    "FTOTVAL": "ftotval",
}


def _prepare_records(
    df: pd.DataFrame,
    key_columns: Dict[str, str],
    include_raw_data: bool = False,
) -> List[Dict[str, Any]]:
    """
    Prepare records for Supabase insert.

    Extracts key columns as typed fields. Optionally stores full row in raw_data JSONB.
    """
    records = []

    for _, row in df.iterrows():
        record = {}

        # Extract key columns
        for src_col, dest_col in key_columns.items():
            if src_col in df.columns:
                val = row[src_col]
                # Convert numpy types to Python types
                if pd.isna(val):
                    record[dest_col] = None
                elif hasattr(val, 'item'):
                    record[dest_col] = val.item()
                else:
                    record[dest_col] = val

        # Optionally store full row as raw_data JSONB (heavy, use sparingly)
        if include_raw_data:
            raw_data = {}
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    raw_data[col] = None
                elif hasattr(val, 'item'):
                    raw_data[col] = val.item()
                else:
                    raw_data[col] = val
            record["raw_data"] = raw_data

        records.append(record)

    return records


def prepare_person_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Prepare person records for Supabase insert."""
    return _prepare_records(df, PERSON_KEY_COLUMNS)


def prepare_household_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Prepare household records for Supabase insert."""
    return _prepare_records(df, HOUSEHOLD_KEY_COLUMNS)


def prepare_family_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Prepare family records for Supabase insert."""
    return _prepare_records(df, FAMILY_KEY_COLUMNS)


def load_cps_to_supabase(
    year: int,
    dry_run: bool = False,
    chunk_size: int = 200,
    limit: Optional[int] = None,
    truncate: bool = False,
    skip: int = 0,
) -> Dict[str, Any]:
    """
    Load CPS ASEC data from local cache to Supabase.

    Args:
        year: Data year (e.g., 2024)
        dry_run: If True, just return stats without inserting
        chunk_size: Records per batch insert (default 200 for stability)
        limit: Max records per table (for testing)
        truncate: If True, delete existing data before loading
        skip: Skip first N records (for resuming interrupted load)

    Returns:
        Dict with counts and status
    """
    cache_dir = get_raw_cache_dir(year)
    person_path = cache_dir / "person.parquet"
    household_path = cache_dir / "household.parquet"
    family_path = cache_dir / "family.parquet"

    if not person_path.exists():
        raise FileNotFoundError(
            f"CPS {year} not found at {cache_dir}. "
            f"Run: python micro/us/census/download_cps.py --year {year}"
        )

    # Load data
    print(f"Loading CPS ASEC {year} from {cache_dir}...")
    person_df = pd.read_parquet(person_path)
    household_df = pd.read_parquet(household_path) if household_path.exists() else pd.DataFrame()
    family_df = pd.read_parquet(family_path) if family_path.exists() else pd.DataFrame()

    # Apply skip and limit
    if skip > 0:
        person_df = person_df.iloc[skip:]
        # For household/family, skip proportionally (approximate)
        hh_skip = int(skip * len(household_df) / len(person_df)) if len(person_df) > 0 else 0
        fam_skip = int(skip * len(family_df) / len(person_df)) if len(person_df) > 0 else 0
        household_df = household_df.iloc[hh_skip:]
        family_df = family_df.iloc[fam_skip:]

    if limit:
        person_df = person_df.head(limit)
        household_df = household_df.head(limit)
        family_df = family_df.head(limit)

    result = {
        "year": year,
        "person_count": len(person_df),
        "household_count": len(household_df),
        "family_count": len(family_df),
        "person_columns": list(person_df.columns),
        "dry_run": dry_run,
    }

    if dry_run:
        print("DRY RUN - would load:")
        print(f"  {result['person_count']:,} person records")
        print(f"  {result['household_count']:,} household records")
        print(f"  {result['family_count']:,} family records")
        return result

    # Get Supabase client and table names
    client = get_supabase_client()
    table_names = get_cps_table_names(year)

    # Truncate existing data if requested
    if truncate:
        print("Truncating existing data...")
        for table_type, table_name in table_names.items():
            try:
                client.schema("microplex").table(table_name).delete().neq("id", 0).execute()
                print(f"  Truncated {table_name}")
            except Exception as e:
                print(f"  Warning: Could not truncate {table_name}: {e}")

    # Load person records
    print(f"Loading {len(person_df):,} person records...")
    person_records = prepare_person_records(person_df)
    _insert_batch(client, table_names["person"], person_records, chunk_size)

    # Load household records
    if len(household_df) > 0:
        print(f"Loading {len(household_df):,} household records...")
        household_records = prepare_household_records(household_df)
        _insert_batch(client, table_names["household"], household_records, chunk_size)

    # Load family records
    if len(family_df) > 0:
        print(f"Loading {len(family_df):,} family records...")
        family_records = prepare_family_records(family_df)
        _insert_batch(client, table_names["family"], family_records, chunk_size)

    # Update dataset registry
    register_dataset(
        jurisdiction="us",
        institution="census",
        dataset="cps_asec",
        year=year,
        table_type="person",
        row_count=len(person_df),
        columns=[{"name": c, "dtype": str(person_df[c].dtype)} for c in person_df.columns[:20]],
        source_url=f"https://www.census.gov/data/datasets/{year+1}/demo/cps/cps-asec-{year+1}.html",
    )

    result["status"] = "completed"
    print(f"Loaded CPS ASEC {year} to Supabase")
    return result


def _insert_batch(
    client,
    table_name: str,
    records: List[Dict[str, Any]],
    chunk_size: int,
) -> int:
    """Insert records in batches."""
    total = 0

    for i in range(0, len(records), chunk_size):
        chunk = records[i:i + chunk_size]
        client.schema("microplex").table(table_name).insert(chunk).execute()
        total += len(chunk)
        if (i + chunk_size) % 5000 == 0:
            print(f"  Inserted {total:,} / {len(records):,} records")

    return total


def export_cps_to_csv(
    year: int,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Export CPS data to CSV files for bulk import to Supabase.

    Much faster than record-by-record insert. Import via:
    - Supabase Dashboard: Table Editor > Import CSV
    - psql: \\copy microplex.us_census_cps_asec_2024_person FROM 'person.csv' CSV HEADER

    Args:
        year: Data year
        output_dir: Directory for CSV files (default: current dir)

    Returns:
        Dict of {table_type: csv_path}
    """
    cache_dir = get_raw_cache_dir(year)
    output_dir = output_dir or Path(".")

    paths = {}
    table_names = get_cps_table_names(year)

    # Person
    person_path = cache_dir / "person.parquet"
    if person_path.exists():
        print("Exporting person records...")
        df = pd.read_parquet(person_path)
        records = prepare_person_records(df)
        out_df = pd.DataFrame(records)
        csv_path = output_dir / f"cps_asec_{year}_person.csv"
        out_df.to_csv(csv_path, index=False)
        paths["person"] = csv_path
        print(f"  Exported {len(out_df):,} records to {csv_path}")

    # Household
    hh_path = cache_dir / "household.parquet"
    if hh_path.exists():
        print("Exporting household records...")
        df = pd.read_parquet(hh_path)
        records = prepare_household_records(df)
        out_df = pd.DataFrame(records)
        csv_path = output_dir / f"cps_asec_{year}_household.csv"
        out_df.to_csv(csv_path, index=False)
        paths["household"] = csv_path
        print(f"  Exported {len(out_df):,} records to {csv_path}")

    # Family
    fam_path = cache_dir / "family.parquet"
    if fam_path.exists():
        print("Exporting family records...")
        df = pd.read_parquet(fam_path)
        records = prepare_family_records(df)
        out_df = pd.DataFrame(records)
        csv_path = output_dir / f"cps_asec_{year}_family.csv"
        out_df.to_csv(csv_path, index=False)
        paths["family"] = csv_path
        print(f"  Exported {len(out_df):,} records to {csv_path}")

    print("\nTo import to Supabase:")
    print("  1. Go to Supabase Dashboard > Table Editor")
    print(f"  2. Select table (e.g., {table_names['person']})")
    print("  3. Click 'Insert' > 'Import data from CSV'")
    print("  4. Upload the CSV file")

    return paths


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Load raw CPS ASEC to Supabase")
    parser.add_argument("--year", type=int, default=2024, help="Data year")
    parser.add_argument("--dry-run", action="store_true", help="Show stats without inserting")
    parser.add_argument("--limit", type=int, help="Limit records per table (for testing)")
    parser.add_argument("--chunk-size", type=int, default=200, help="Batch insert size")
    parser.add_argument("--truncate", action="store_true", help="Delete existing data before loading")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N records (for resuming)")
    parser.add_argument("--method", type=str, choices=["api", "csv"], default="api",
                       help="Method: 'api' for record-by-record, 'csv' for export to CSV")
    parser.add_argument("--output-dir", type=str, help="Output directory for CSV export")
    args = parser.parse_args()

    if args.method == "csv":
        output_dir = Path(args.output_dir) if args.output_dir else None
        export_cps_to_csv(args.year, output_dir)
    else:
        load_cps_to_supabase(
            year=args.year,
            dry_run=args.dry_run,
            limit=args.limit,
            chunk_size=args.chunk_size,
            truncate=args.truncate,
            skip=args.skip,
        )


if __name__ == "__main__":
    main()
