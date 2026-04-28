"""
Supabase client for Arch.

Provides connection to Cosilico Supabase database for:
- Raw microdata tables (e.g., microplex.us_census_cps_asec_2024_person)
- Calibration targets
- Data sources metadata
- Dataset registry

Table naming pattern: {jurisdiction}_{institution}_{dataset}_{year}_{table_type}
Example: us_census_cps_asec_2024_person
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional

import pandas as pd
from supabase import create_client, Client


@dataclass
class SupabaseConfig:
    """Configuration for Supabase connection."""

    url: str
    secret_key: str

    @classmethod
    def from_env(cls) -> "SupabaseConfig":
        """
        Load configuration from environment variables.

        Required:
            COSILICO_SUPABASE_URL: Supabase project URL
            COSILICO_SUPABASE_SECRET_KEY: Service role key for full access

        Raises:
            ValueError: If required environment variables are missing
        """
        url = os.environ.get("COSILICO_SUPABASE_URL")
        if not url:
            raise ValueError(
                "COSILICO_SUPABASE_URL not set. "
                "Set this to your Supabase project URL."
            )

        secret_key = os.environ.get("COSILICO_SUPABASE_SECRET_KEY")
        if not secret_key:
            raise ValueError(
                "COSILICO_SUPABASE_SECRET_KEY not set. "
                "Set this to your service role key."
            )

        return cls(url=url, secret_key=secret_key)


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """
    Get a Supabase client instance.

    Uses service role key for full database access.
    Client is cached for reuse.

    Returns:
        Supabase client instance

    Raises:
        ValueError: If environment variables are not set
    """
    config = SupabaseConfig.from_env()
    return create_client(config.url, config.secret_key)


# =============================================================================
# Table naming helpers
# =============================================================================

def get_table_name(
    jurisdiction: str,
    institution: str,
    dataset: str,
    year: int,
    table_type: str,
) -> str:
    """
    Build table name from components.

    Args:
        jurisdiction: e.g., "us", "uk", "eu"
        institution: e.g., "census", "irs", "ons"
        dataset: e.g., "cps_asec", "puf", "frs"
        year: e.g., 2024
        table_type: e.g., "person", "household", "family"

    Returns:
        Table name like "us_census_cps_asec_2024_person"
    """
    return f"{jurisdiction}_{institution}_{dataset}_{year}_{table_type}"


# =============================================================================
# Sources and datasets
# =============================================================================

def query_sources(
    jurisdiction: Optional[str] = None,
    institution: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Query data sources from microplex.sources.

    Args:
        jurisdiction: Filter by jurisdiction (e.g., "us", "uk")
        institution: Filter by institution (e.g., "irs", "census")

    Returns:
        List of source records
    """
    client = get_supabase_client()
    query = client.schema("microplex").table("sources").select("*")

    if jurisdiction:
        query = query.eq("jurisdiction", jurisdiction)
    if institution:
        query = query.eq("institution", institution)

    result = query.execute()
    return result.data


def list_datasets(
    jurisdiction: Optional[str] = None,
    institution: Optional[str] = None,
    dataset: Optional[str] = None,
    year: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    List available raw microdata datasets.

    Args:
        jurisdiction: Filter by jurisdiction
        institution: Filter by institution
        dataset: Filter by dataset name
        year: Filter by year

    Returns:
        List of dataset records with table_name
    """
    client = get_supabase_client()
    query = client.schema("microplex").table("datasets").select("*")

    if jurisdiction:
        query = query.eq("jurisdiction", jurisdiction)
    if institution:
        query = query.eq("institution", institution)
    if dataset:
        query = query.eq("dataset", dataset)
    if year:
        query = query.eq("year", year)

    result = query.execute()
    return result.data


def register_dataset(
    jurisdiction: str,
    institution: str,
    dataset: str,
    year: int,
    table_type: str,
    row_count: Optional[int] = None,
    columns: Optional[List[Dict]] = None,
    source_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Register a new dataset in the datasets table.

    Args:
        jurisdiction: e.g., "us"
        institution: e.g., "census"
        dataset: e.g., "cps_asec"
        year: e.g., 2024
        table_type: e.g., "person"
        row_count: Number of rows
        columns: Column metadata [{name, dtype}, ...]
        source_url: URL to source data

    Returns:
        Created dataset record
    """
    client = get_supabase_client()

    data = {
        "jurisdiction": jurisdiction,
        "institution": institution,
        "dataset": dataset,
        "year": year,
        "table_type": table_type,
    }

    if row_count:
        data["row_count"] = row_count
    if columns:
        data["columns"] = columns
    if source_url:
        data["source_url"] = source_url

    result = client.schema("microplex").table("datasets").upsert(data, on_conflict="jurisdiction,institution,dataset,year,table_type").execute()
    return result.data[0] if result.data else {}


# =============================================================================
# Raw microdata queries
# =============================================================================

def query_microdata(
    jurisdiction: str,
    institution: str,
    dataset: str,
    year: int,
    table_type: str,
    columns: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 100000,
) -> pd.DataFrame:
    """
    Query raw microdata from a specific table.

    Uses pagination to handle large result sets (PostgREST default is 1000).

    Args:
        jurisdiction: e.g., "us"
        institution: e.g., "census"
        dataset: e.g., "cps_asec"
        year: e.g., 2024
        table_type: e.g., "person", "household"
        columns: Specific columns to select (default all)
        filters: Dict of {column: value} filters
        limit: Maximum records

    Returns:
        DataFrame with microdata records
    """
    client = get_supabase_client()
    table_name = get_table_name(jurisdiction, institution, dataset, year, table_type)

    select_cols = ",".join(columns) if columns else "*"
    page_size = 1000  # PostgREST default limit
    all_data = []
    offset = 0

    while offset < limit:
        fetch_limit = min(page_size, limit - offset)
        query = client.schema("microplex").table(table_name).select(select_cols)

        if filters:
            for col, val in filters.items():
                query = query.eq(col, val)

        query = query.range(offset, offset + fetch_limit - 1)
        result = query.execute()

        if not result.data:
            break

        all_data.extend(result.data)
        offset += len(result.data)

        if len(result.data) < fetch_limit:
            break  # No more data

    return pd.DataFrame(all_data)


def query_cps_asec(
    year: int,
    table_type: str = "person",
    state_fips: Optional[int] = None,
    columns: Optional[List[str]] = None,
    limit: int = 100000,
) -> pd.DataFrame:
    """
    Query CPS ASEC microdata (convenience wrapper).

    Args:
        year: Data year
        table_type: "person", "household", or "family"
        state_fips: Filter by state FIPS code
        columns: Specific columns
        limit: Maximum records

    Returns:
        DataFrame with CPS records
    """
    filters = {}
    if state_fips:
        filters["gestfips"] = state_fips

    return query_microdata(
        jurisdiction="us",
        institution="census",
        dataset="cps_asec",
        year=year,
        table_type=table_type,
        columns=columns,
        filters=filters,
        limit=limit,
    )


# =============================================================================
# Targets and strata
# =============================================================================

def query_strata(
    jurisdiction: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Query strata with their constraints from microplex.strata.

    Args:
        jurisdiction: Filter by jurisdiction

    Returns:
        List of strata records with nested constraints
    """
    client = get_supabase_client()
    query = client.schema("microplex").table("strata").select("*, stratum_constraints(*)")

    if jurisdiction:
        query = query.eq("jurisdiction", jurisdiction)

    result = query.execute()
    return result.data


def query_targets(
    jurisdiction: Optional[str] = None,
    year: Optional[int] = None,
    source_id: Optional[str] = None,
    variable: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Query calibration targets from microplex.targets.

    Args:
        jurisdiction: Filter by jurisdiction (via stratum join)
        year: Filter by period/year
        source_id: Filter by source UUID
        variable: Filter by variable name

    Returns:
        List of target records with stratum info
    """
    client = get_supabase_client()
    # Nested join: strata with their stratum_constraints
    query = client.schema("microplex").table("targets").select("*, strata(*, stratum_constraints(*)), sources(*)")

    if year:
        query = query.eq("period", year)
    if source_id:
        query = query.eq("source_id", source_id)
    if variable:
        query = query.eq("variable", variable)

    result = query.execute()

    # Filter by jurisdiction if specified (post-query since it's on joined table)
    data = result.data
    if jurisdiction:
        data = [t for t in data if t.get("strata", {}).get("jurisdiction") == jurisdiction]

    return data


# =============================================================================
# Insert operations
# =============================================================================

def insert_microdata_batch(
    jurisdiction: str,
    institution: str,
    dataset: str,
    year: int,
    table_type: str,
    records: List[Dict[str, Any]],
    chunk_size: int = 1000,
) -> int:
    """
    Insert microdata records in batches.

    Args:
        jurisdiction: e.g., "us"
        institution: e.g., "census"
        dataset: e.g., "cps_asec"
        year: e.g., 2024
        table_type: e.g., "person"
        records: List of record dicts
        chunk_size: Records per batch

    Returns:
        Number of records inserted
    """
    client = get_supabase_client()
    table_name = get_table_name(jurisdiction, institution, dataset, year, table_type)
    total = 0

    for i in range(0, len(records), chunk_size):
        chunk = records[i:i + chunk_size]
        client.schema("microplex").table(table_name).insert(chunk).execute()
        total += len(chunk)

    return total


def insert_targets_batch(
    targets: List[Dict[str, Any]],
    chunk_size: int = 100,
) -> int:
    """
    Insert target records in batches.

    Args:
        targets: List of target dicts
        chunk_size: Records per batch

    Returns:
        Number of records inserted
    """
    client = get_supabase_client()
    total = 0

    for i in range(0, len(targets), chunk_size):
        chunk = targets[i:i + chunk_size]
        client.schema("microplex").table("targets").insert(chunk).execute()
        total += len(chunk)

    return total
