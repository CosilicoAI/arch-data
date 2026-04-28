"""
CPS ASEC Microdata Downloader

Downloads Current Population Survey Annual Social and Economic Supplement
from Census Bureau and caches raw data locally for efficient variable extraction.

DESIGN PRINCIPLES:
1. Cache Full Raw Data: Download once, extract any variables later
2. Primary Inputs Only: Extract only raw survey responses, not derived values
3. Derive in Rules Engine: Qualifying children, tax liability, etc. computed in cosilico-us

The raw CPS is cached as HDF5 with all person/household/family tables.
This allows adding new variables without re-downloading (~200MB per year).

Usage:
    python download_cps.py [--year 2024] [--raw-only] [--output micro/us/cps_2024.parquet]
"""

import io
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    def tqdm(iterable=None, **kwargs):
        """Small fallback so downloads still work without tqdm installed."""
        if iterable is not None:
            return iterable
        return _NullProgress()


class _NullProgress:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def update(self, n: int) -> None:
        return None

# Storage paths
CACHE_DIR = Path(__file__).parent / "raw_cache"
OUTPUT_DIR = Path(__file__).parent.parent  # micro/us/

# Census Bureau CPS ASEC download URLs
# Data for year N is published in March of year N+1
CPS_URL_BY_YEAR = {
    2018: "https://www2.census.gov/programs-surveys/cps/datasets/2019/march/asecpub19csv.zip",
    2019: "https://www2.census.gov/programs-surveys/cps/datasets/2020/march/asecpub20csv.zip",
    2020: "https://www2.census.gov/programs-surveys/cps/datasets/2021/march/asecpub21csv.zip",
    2021: "https://www2.census.gov/programs-surveys/cps/datasets/2022/march/asecpub22csv.zip",
    2022: "https://www2.census.gov/programs-surveys/cps/datasets/2023/march/asecpub23csv.zip",
    2023: "https://www2.census.gov/programs-surveys/cps/datasets/2024/march/asecpub24csv.zip",
    2024: "https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asecpub25csv.zip",
}


def get_raw_cache_dir(year: int) -> Path:
    """Get directory for cached raw CPS data for a year."""
    return CACHE_DIR / f"census_cps_{year}"


def download_raw_cps(year: int, progress: bool = True, force: bool = False) -> Path:
    """Download and cache full raw CPS ASEC data.

    Downloads all tables (person, household, family) and stores as parquet.
    Adding new variables later doesn't require re-downloading.

    Args:
        year: Tax year (e.g., 2024)
        progress: Show download progress bar
        force: Re-download even if cached

    Returns:
        Path to cache directory containing parquet files
    """
    cache_dir = get_raw_cache_dir(year)
    person_path = cache_dir / "person.parquet"

    if person_path.exists() and not force:
        print(f"Using cached raw CPS {year} from {cache_dir}")
        return cache_dir

    if year not in CPS_URL_BY_YEAR:
        available = sorted(CPS_URL_BY_YEAR.keys())
        raise ValueError(f"Year {year} not available. Available: {available}")

    url = CPS_URL_BY_YEAR[year]
    print(f"Downloading CPS ASEC {year} from {url}")

    # Download ZIP
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 200_000_000))
    content = io.BytesIO()

    if progress:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Download") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                content.write(chunk)
                pbar.update(len(chunk))
    else:
        for chunk in response.iter_content(chunk_size=8192):
            content.write(chunk)

    # Extract and cache all tables as parquet
    cache_dir.mkdir(parents=True, exist_ok=True)
    yy = str(year + 1)[-2:]

    with zipfile.ZipFile(content) as zf:
        print(f"Files in ZIP: {zf.namelist()}")

        # Handle special path prefix for 2018
        file_prefix = "cpspb/asec/prod/data/2019/" if yy == "19" else ""

        # Person file (read ALL columns)
        person_file = f"{file_prefix}pppub{yy}.csv"
        print(f"Extracting {person_file} (all columns)...")
        with zf.open(person_file) as f:
            person = pd.read_csv(f, low_memory=False).fillna(0)
            person.to_parquet(cache_dir / "person.parquet", index=False)
            print(f"  → {len(person):,} person records, {len(person.columns)} columns")

        # Household file
        household_file = f"{file_prefix}hhpub{yy}.csv"
        print(f"Extracting {household_file}...")
        with zf.open(household_file) as f:
            household = pd.read_csv(f, low_memory=False).fillna(0)
            household.to_parquet(cache_dir / "household.parquet", index=False)
            print(f"  → {len(household):,} household records")

        # Family file
        family_file = f"{file_prefix}ffpub{yy}.csv"
        print(f"Extracting {family_file}...")
        with zf.open(family_file) as f:
            family = pd.read_csv(f, low_memory=False).fillna(0)
            family.to_parquet(cache_dir / "family.parquet", index=False)
            print(f"  → {len(family):,} family records")

    print(f"Cached raw CPS {year} to {cache_dir}")
    return cache_dir


def download_cps_zip(year: int, progress: bool = True) -> bytes:
    """Download a CPS ASEC ZIP and return its raw bytes."""
    if year not in CPS_URL_BY_YEAR:
        available = sorted(CPS_URL_BY_YEAR.keys())
        raise ValueError(f"Year {year} not available. Available: {available}")

    response = requests.get(CPS_URL_BY_YEAR[year], stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 200_000_000))
    content = io.BytesIO()

    if progress:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Download") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                content.write(chunk)
                pbar.update(len(chunk))
    else:
        for chunk in response.iter_content(chunk_size=8192):
            content.write(chunk)

    return content.getvalue()


def extract_person_data(
    zip_content: bytes,
    year: int,
    columns: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Extract person records from a CPS ASEC ZIP byte payload."""
    if columns is None:
        columns = PERSON_COLUMNS

    yy = str(year + 1)[-2:]
    person_file = f"pppub{yy}.csv"
    prefixed_person_file = f"cpspb/asec/prod/data/20{yy}/{person_file}"

    with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
        names = set(zf.namelist())
        if person_file in names:
            selected_file = person_file
        elif prefixed_person_file in names:
            selected_file = prefixed_person_file
        else:
            person_candidates = [
                name for name in names if Path(name).name.lower() == person_file
            ]
            if not person_candidates:
                raise FileNotFoundError(
                    f"Could not find {person_file} in CPS ZIP. "
                    f"Files: {sorted(names)}"
                )
            selected_file = person_candidates[0]

        with zf.open(selected_file) as f:
            person = pd.read_csv(f, low_memory=False).fillna(0)

    available = [c for c in columns if c in person.columns]
    df = person[available].copy()
    df = df.rename(columns={k: v for k, v in columns.items() if k in available})
    return process_cps_data(df)


def extract_cps_variables(
    year: int,
    columns: Optional[dict[str, str]] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Extract specific variables from cached raw CPS data.

    Args:
        year: Tax year
        columns: Dict mapping CPS column names to output names.
                 If None, uses default PERSON_COLUMNS.
        output_path: Path to save parquet (default: micro/us/cps_{year}.parquet)

    Returns:
        Processed DataFrame with requested columns
    """
    # Ensure raw data is cached
    cache_dir = get_raw_cache_dir(year)
    person_path = cache_dir / "person.parquet"
    if not person_path.exists():
        download_raw_cps(year)

    if columns is None:
        columns = PERSON_COLUMNS

    # Read from cache
    person = pd.read_parquet(person_path)

    # Merge household data for geography (GESTFIPS is in household file)
    household_path = cache_dir / "household.parquet"
    if household_path.exists():
        household = pd.read_parquet(household_path, columns=["H_SEQ", "GESTFIPS", "GEDIV", "GEREG"])
        # Merge on household ID (PH_SEQ in person = H_SEQ in household)
        person = person.merge(
            household.rename(columns={"H_SEQ": "PH_SEQ"}),
            on="PH_SEQ",
            how="left"
        )

    # Select and rename columns
    available = [c for c in columns.keys() if c in person.columns]
    missing = [c for c in columns.keys() if c not in person.columns]
    if missing:
        print(f"Warning: Missing columns: {missing}")

    df = person[available].copy()
    df = df.rename(columns={k: v for k, v in columns.items() if k in available})

    # Process
    df = _process_person_data(df)

    # Save
    if output_path is None:
        output_path = OUTPUT_DIR / f"cps_{year}.parquet"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df):,} records to {output_path}")

    return df


# Default columns to extract (PRIMARY INPUTS ONLY)
# See cps-asec/*.yaml for documentation of each variable
PERSON_COLUMNS = {
    # Identifiers
    "PH_SEQ": "household_id",
    "P_SEQ": "person_seq",
    "A_LINENO": "line_number",
    "TAX_ID": "tax_unit_id",
    "SPM_ID": "spm_unit_id",
    # Demographics
    "A_AGE": "age",
    "A_SEX": "sex",
    "PRDTRACE": "race",
    "A_MARITL": "marital_status",
    # Relationship (for qualifying child derivation)
    "A_FAMREL": "family_relationship",
    "PARENT": "parent_line_number",
    "A_SPOUSE": "spouse_line_number",
    # Weight
    "A_FNLWGT": "weight",
    "MARSUPWT": "march_supplement_weight",
    # Employment
    "A_CLSWKR": "class_of_worker",
    "A_WKSTAT": "work_status",
    "A_HRS1": "hours_worked",
    # Income (primary inputs)
    "WSAL_VAL": "wage_salary_income",
    "SEMP_VAL": "self_employment_income",
    "FRSE_VAL": "farm_self_employment_income",
    "INT_VAL": "interest_income",
    "DIV_VAL": "dividend_income",
    "RNT_VAL": "rental_income",
    "SS_VAL": "social_security_income",
    "SSI_VAL": "ssi_income",
    "PAW_VAL": "public_assistance_income",
    "UC_VAL": "unemployment_compensation",
    "VET_VAL": "veterans_benefits",
    "OI_VAL": "other_income",
    "PTOTVAL": "total_person_income",
    "PEARNVAL": "total_earnings",
    # Geography
    "GESTFIPS": "state_fips",
    # Tax/benefit validation targets (from TAXSIM)
    "FEDTAX_AC": "federal_tax",
    "FICA": "fica_tax",
    "EIT_CRED": "eitc_received",
    "ACTC_CRD": "actc_received",
    "CTC_CRD": "ctc_received",
}


def _process_person_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process raw person data into analysis-ready format."""
    # Create unique person ID
    if "household_id" in df.columns and "person_seq" in df.columns:
        df["person_id"] = df["household_id"].astype(str) + "_" + df["person_seq"].astype(str)

    # Employment status
    if "class_of_worker" in df.columns:
        df["employment_status"] = (df["class_of_worker"] > 0).astype(int)
    elif "work_status" in df.columns:
        df["employment_status"] = df["work_status"].isin([1, 2, 3, 4]).astype(int)

    # Total income from components
    income_cols = [
        "wage_salary_income", "self_employment_income", "farm_self_employment_income",
        "interest_income", "dividend_income", "rental_income",
    ]
    available = [c for c in income_cols if c in df.columns]
    if available:
        df["income"] = df[available].fillna(0).sum(axis=1)
    elif "total_person_income" in df.columns:
        df["income"] = df["total_person_income"].fillna(0)

    # Child presence indicator for simple calibration/synthesis tests.
    if "own_children_under_18" in df.columns:
        df["has_children"] = (df["own_children_under_18"].fillna(0) > 0).astype(int)
    elif "has_children" not in df.columns:
        df["has_children"] = 0

    # Apply weight scaling (CPS weights have 2 implied decimals)
    if "march_supplement_weight" in df.columns:
        df["weight"] = df["march_supplement_weight"].fillna(0) / 100
    elif "weight" in df.columns:
        df["weight"] = df["weight"].fillna(0) / 100
    else:
        df["weight"] = 1

    # Filter to positive weights
    df = df[df["weight"] > 0].copy()

    return df


def process_cps_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process raw or renamed CPS person records into analysis-ready columns."""
    return _process_person_data(df.copy())


# Legacy function for backwards compatibility
def download_and_process_cps(
    year: int,
    output_path: Optional[Path] = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Download CPS ASEC data and process to parquet.

    This is a convenience wrapper that downloads raw data (if not cached)
    and extracts default columns.
    """
    download_raw_cps(year, progress=progress)
    return extract_cps_variables(year, output_path=output_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download CPS ASEC microdata")
    parser.add_argument("--year", type=int, default=2024, help="Tax year")
    parser.add_argument("--output", type=str, help="Output parquet path")
    parser.add_argument("--raw-only", action="store_true", help="Only download raw data, don't extract")
    parser.add_argument("--force", action="store_true", help="Force re-download even if cached")
    args = parser.parse_args()

    if args.raw_only:
        download_raw_cps(args.year, force=args.force)
    else:
        output_path = Path(args.output) if args.output else None
        cache_dir = get_raw_cache_dir(args.year)
        if args.force or not (cache_dir / "person.parquet").exists():
            download_raw_cps(args.year, force=args.force)
        extract_cps_variables(args.year, output_path=output_path)


if __name__ == "__main__":
    main()
