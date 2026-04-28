"""
ETL for IRS Statistics of Income (SOI) targets.

Loads IRS SOI Publication 1304 Table 1.1 into the targets database.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from importlib.resources import files
from io import BytesIO
from typing import Any, TypedDict
from urllib.request import Request, urlopen

import pandas as pd
import yaml
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

SOURCE_URL = "https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics"
TABLE_1_1_SHEET_NAME = "TBL11"
TABLE_1_4_SHEET_NAME = "TBL14"
TABLE_1_1_MONEY_SCALE = 1_000
TABLE_1_4_MONEY_SCALE = 1_000
TABLE_1_1_PACKAGE_DIR = "data/irs_soi/table_1_1"
TABLE_1_4_PACKAGE_DIR = "data/irs_soi/table_1_4"
TABLE_1_1_MANIFEST = "manifest.yaml"
TABLE_1_4_MANIFEST = "manifest.yaml"

# AGI bracket definitions (lower, upper) in dollars.
AGI_BRACKETS = {
    "under_1": (float("-inf"), 1),
    "1_to_5k": (1, 5_000),
    "5k_to_10k": (5_000, 10_000),
    "10k_to_15k": (10_000, 15_000),
    "15k_to_20k": (15_000, 20_000),
    "20k_to_25k": (20_000, 25_000),
    "25k_to_30k": (25_000, 30_000),
    "30k_to_40k": (30_000, 40_000),
    "40k_to_50k": (40_000, 50_000),
    "50k_to_75k": (50_000, 75_000),
    "75k_to_100k": (75_000, 100_000),
    "100k_to_200k": (100_000, 200_000),
    "200k_to_500k": (200_000, 500_000),
    "500k_to_1m": (500_000, 1_000_000),
    "1m_to_1_5m": (1_000_000, 1_500_000),
    "1_5m_to_2m": (1_500_000, 2_000_000),
    "2m_to_5m": (2_000_000, 5_000_000),
    "5m_to_10m": (5_000_000, 10_000_000),
    "10m_plus": (10_000_000, float("inf")),
}

TABLE_1_1_AGI_LABEL_TO_BRACKET = {
    "No adjusted gross income": "under_1",
    "$1 under $5,000": "1_to_5k",
    "$5,000 under $10,000": "5k_to_10k",
    "$10,000 under $15,000": "10k_to_15k",
    "$15,000 under $20,000": "15k_to_20k",
    "$20,000 under $25,000": "20k_to_25k",
    "$25,000 under $30,000": "25k_to_30k",
    "$30,000 under $40,000": "30k_to_40k",
    "$40,000 under $50,000": "40k_to_50k",
    "$50,000 under $75,000": "50k_to_75k",
    "$75,000 under $100,000": "75k_to_100k",
    "$100,000 under $200,000": "100k_to_200k",
    "$200,000 under $500,000": "200k_to_500k",
    "$500,000 under $1,000,000": "500k_to_1m",
    "$1,000,000 under $1,500,000": "1m_to_1_5m",
    "$1,500,000 under $2,000,000": "1_5m_to_2m",
    "$2,000,000 under $5,000,000": "2m_to_5m",
    "$5,000,000 under $10,000,000": "5m_to_10m",
    "$10,000,000 or more": "10m_plus",
}


class SOITable11Data(TypedDict):
    source_url: str
    total_returns: int
    total_agi: int
    total_income_tax: int
    returns_by_agi_bracket: dict[str, int]
    agi_by_bracket: dict[str, int]
    income_tax_by_bracket: dict[str, int]


class SOITable14Data(TypedDict):
    source_url: str
    total_employment_income_returns: int
    total_employment_income: int
    employment_income_returns_by_agi_bracket: dict[str, int]
    employment_income_by_agi_bracket: dict[str, int]


@lru_cache(maxsize=1)
def _table_1_1_manifest() -> dict[str, Any]:
    manifest_path = files("db").joinpath(TABLE_1_1_PACKAGE_DIR, TABLE_1_1_MANIFEST)
    with manifest_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _table_1_4_manifest() -> dict[str, Any]:
    manifest_path = files("db").joinpath(TABLE_1_4_PACKAGE_DIR, TABLE_1_4_MANIFEST)
    with manifest_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def available_soi_years() -> list[int]:
    """Return SOI Table 1.1 years available in the source manifest."""
    return sorted(int(year) for year in _table_1_1_manifest()["files"])


def available_soi_table_1_4_years() -> list[int]:
    """Return SOI Table 1.4 years available in the source manifest."""
    return sorted(int(year) for year in _table_1_4_manifest()["files"])


def soi_table_1_1_source_url(year: int) -> str:
    """Return the IRS source URL for a Table 1.1 year."""
    return _table_1_1_file_spec(year)["source_url"]


def soi_table_1_4_source_url(year: int) -> str:
    """Return the IRS source URL for a Table 1.4 year."""
    return _table_1_4_file_spec(year)["source_url"]


def _table_1_1_file_spec(year: int) -> dict[str, str]:
    files_by_year = _table_1_1_manifest()["files"]
    try:
        return files_by_year[year]
    except KeyError:
        return files_by_year[str(year)]


def _table_1_4_file_spec(year: int) -> dict[str, str]:
    files_by_year = _table_1_4_manifest()["files"]
    try:
        return files_by_year[year]
    except KeyError:
        return files_by_year[str(year)]


# Compatibility manifest for older callers. Numeric SOI values are parsed from
# the source files at load time rather than embedded in Python.
SOI_DATA = {
    year: {"source_url": soi_table_1_1_source_url(year)}
    for year in available_soi_years()
}


@lru_cache(maxsize=None)
def _table_1_1_content(year: int) -> bytes:
    return _table_content(year, _table_1_1_file_spec, TABLE_1_1_PACKAGE_DIR)


@lru_cache(maxsize=None)
def _table_1_4_content(year: int) -> bytes:
    return _table_content(year, _table_1_4_file_spec, TABLE_1_4_PACKAGE_DIR)


def _table_content(year: int, file_spec_fn, package_dir: str) -> bytes:
    spec = file_spec_fn(year)
    package_path = files("db").joinpath(package_dir, spec["filename"])
    if package_path.is_file():
        content = package_path.read_bytes()
    else:
        request = Request(
            spec["source_url"],
            headers={"User-Agent": "cosilico-arch/0.1", "Accept": "*/*"},
        )
        with urlopen(request, timeout=120) as response:
            content = response.read()

    expected_sha = spec.get("sha256")
    if expected_sha:
        actual_sha = hashlib.sha256(content).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(
                f"SOI source file {year} checksum mismatch: "
                f"expected {expected_sha}, got {actual_sha}"
            )
    return content


@lru_cache(maxsize=None)
def _read_soi_table_1_1_frame(year: int) -> pd.DataFrame:
    return pd.read_excel(
        BytesIO(_table_1_1_content(year)),
        sheet_name=TABLE_1_1_SHEET_NAME,
        header=None,
        dtype=object,
        engine="xlrd",
    )


@lru_cache(maxsize=None)
def _read_soi_table_1_4_frame(year: int) -> pd.DataFrame:
    return pd.read_excel(
        BytesIO(_table_1_4_content(year)),
        sheet_name=TABLE_1_4_SHEET_NAME,
        header=None,
        dtype=object,
        engine="xlrd",
    )


def load_soi_table_1_1_data(year: int) -> SOITable11Data:
    """Parse one year of IRS SOI Publication 1304 Table 1.1."""
    return _parse_soi_table_1_1_frame(
        _read_soi_table_1_1_frame(year),
        source_url=soi_table_1_1_source_url(year),
    )


def load_soi_table_1_4_data(year: int) -> SOITable14Data:
    """Parse wage targets from IRS SOI Publication 1304 Table 1.4."""
    return _parse_soi_table_1_4_frame(
        _read_soi_table_1_4_frame(year),
        source_url=soi_table_1_4_source_url(year),
    )


def _clean_label(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).replace("\n", " ").split()).strip()


def _numeric_value(value: object) -> float:
    if value is None or pd.isna(value):
        raise ValueError("Expected numeric SOI cell, got empty value")
    if isinstance(value, str):
        clean = value.replace("$", "").replace(",", "").strip()
        if clean.startswith("["):
            raise ValueError(f"Expected numeric SOI cell, got footnote marker {value}")
        return float(clean)
    return float(value)


def _count_value(value: object) -> int:
    return int(round(_numeric_value(value)))


def _money_value(value: object) -> int:
    return int(round(_numeric_value(value) * TABLE_1_1_MONEY_SCALE))


def _table_1_4_money_value(value: object) -> int:
    return int(round(_numeric_value(value) * TABLE_1_4_MONEY_SCALE))


def _header_text(df: pd.DataFrame, column: int) -> str:
    pieces = [_clean_label(df.iat[row, column]) for row in range(min(8, len(df)))]
    return " ".join(piece for piece in pieces if piece).lower()


def _find_column(df: pd.DataFrame, *phrases: str) -> int:
    for column in range(df.shape[1]):
        header = _header_text(df, column)
        if all(phrase.lower() in header for phrase in phrases):
            return column
    joined = ", ".join(phrases)
    raise ValueError(f"Could not find SOI Table 1.1 column containing: {joined}")


def _table_1_1_size_rows(df: pd.DataFrame) -> pd.DataFrame:
    labels = df.iloc[:, 0].map(_clean_label)
    start_matches = labels[labels == "All returns"]
    if start_matches.empty:
        raise ValueError("Could not find SOI Table 1.1 'All returns' row")
    start = int(start_matches.index[0])

    end_label = "Accumulated from smallest size of adjusted gross income"
    end_matches = labels[(labels == end_label) & (labels.index > start)]
    end = int(end_matches.index[0]) if not end_matches.empty else len(df)
    return df.loc[start : end - 1]


def _table_1_4_size_rows(df: pd.DataFrame) -> pd.DataFrame:
    labels = df.iloc[:, 0].map(_clean_label)
    start_matches = labels[labels == "All returns, total"]
    if start_matches.empty:
        raise ValueError("Could not find SOI Table 1.4 'All returns, total' row")
    start = int(start_matches.index[0])

    end_matches = labels[(labels == "Taxable returns, total") & (labels.index > start)]
    end = int(end_matches.index[0]) if not end_matches.empty else len(df)
    return df.loc[start : end - 1]


def _row_by_label(rows: pd.DataFrame, label: str) -> pd.Series:
    labels = rows.iloc[:, 0].map(_clean_label)
    matches = rows[labels == label]
    if matches.empty:
        raise ValueError(f"Could not find SOI Table 1.1 row: {label}")
    return matches.iloc[0]


def _parse_soi_table_1_1_frame(
    df: pd.DataFrame,
    *,
    source_url: str,
) -> SOITable11Data:
    count_col = _find_column(df, "number of returns")
    agi_col = _find_column(df, "adjusted gross income less deficit", "amount")
    income_tax_col = _find_column(df, "total income tax", "amount")
    rows = _table_1_1_size_rows(df)
    all_returns = _row_by_label(rows, "All returns")

    returns_by_bracket = {}
    agi_by_bracket = {}
    income_tax_by_bracket = {}
    for source_label, bracket_name in TABLE_1_1_AGI_LABEL_TO_BRACKET.items():
        row = _row_by_label(rows, source_label)
        returns_by_bracket[bracket_name] = _count_value(row.iat[count_col])
        agi_by_bracket[bracket_name] = _money_value(row.iat[agi_col])
        income_tax_by_bracket[bracket_name] = _money_value(row.iat[income_tax_col])

    return {
        "source_url": source_url,
        "total_returns": _count_value(all_returns.iat[count_col]),
        "total_agi": _money_value(all_returns.iat[agi_col]),
        "total_income_tax": _money_value(all_returns.iat[income_tax_col]),
        "returns_by_agi_bracket": returns_by_bracket,
        "agi_by_bracket": agi_by_bracket,
        "income_tax_by_bracket": income_tax_by_bracket,
    }


def _parse_soi_table_1_4_frame(
    df: pd.DataFrame,
    *,
    source_url: str,
) -> SOITable14Data:
    wage_count_col = _find_column(df, "wages", "number of returns")
    wage_amount_col = wage_count_col + 1
    if "amount" not in _header_text(df, wage_amount_col):
        raise ValueError("Could not find SOI Table 1.4 wage amount column")
    rows = _table_1_4_size_rows(df)
    all_returns = _row_by_label(rows, "All returns, total")

    wage_returns_by_bracket = {}
    wage_amount_by_bracket = {}
    for source_label, bracket_name in TABLE_1_1_AGI_LABEL_TO_BRACKET.items():
        row = _row_by_label(rows, source_label)
        wage_returns_by_bracket[bracket_name] = _count_value(row.iat[wage_count_col])
        wage_amount_by_bracket[bracket_name] = _table_1_4_money_value(
            row.iat[wage_amount_col]
        )

    return {
        "source_url": source_url,
        "total_employment_income_returns": _count_value(
            all_returns.iat[wage_count_col]
        ),
        "total_employment_income": _table_1_4_money_value(
            all_returns.iat[wage_amount_col]
        ),
        "employment_income_returns_by_agi_bracket": wage_returns_by_bracket,
        "employment_income_by_agi_bracket": wage_amount_by_bracket,
    }


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


def _add_target(
    session: Session,
    *,
    stratum_id: int,
    variable: str,
    period: int,
    value: float,
    target_type: TargetType,
    source_url: str,
    source_table: str = "Table 1.1",
) -> None:
    session.add(
        Target(
            stratum_id=stratum_id,
            variable=variable,
            period=period,
            value=value,
            target_type=target_type,
            source=DataSource.IRS_SOI,
            source_table=source_table,
            source_url=source_url,
        )
    )


def load_soi_targets(session: Session, years: list[int] | None = None) -> None:
    """
    Load SOI targets into database.

    Args:
        session: Database session
        years: Years to load (default: all available)
    """
    if years is None:
        years = available_soi_years()

    for year in years:
        if year not in available_soi_years():
            continue

        data = load_soi_table_1_1_data(year)
        source_url = data["source_url"]
        table_1_4_data = None
        table_1_4_source_url = None
        if year in available_soi_table_1_4_years():
            table_1_4_data = load_soi_table_1_4_data(year)
            table_1_4_source_url = table_1_4_data["source_url"]

        national_stratum = get_or_create_stratum(
            session,
            name="US All Filers",
            jurisdiction=Jurisdiction.US_FEDERAL,
            constraints=[("is_tax_filer", "==", "1")],
            description="All individual income tax returns filed in the US",
            stratum_group_id="national",
        )

        _add_target(
            session,
            stratum_id=national_stratum.id,
            variable="tax_unit_count",
            period=year,
            value=data["total_returns"],
            target_type=TargetType.COUNT,
            source_url=source_url,
        )
        _add_target(
            session,
            stratum_id=national_stratum.id,
            variable="adjusted_gross_income",
            period=year,
            value=data["total_agi"],
            target_type=TargetType.AMOUNT,
            source_url=source_url,
        )
        _add_target(
            session,
            stratum_id=national_stratum.id,
            variable="income_tax_liability",
            period=year,
            value=data["total_income_tax"],
            target_type=TargetType.AMOUNT,
            source_url=source_url,
        )

        if table_1_4_data is not None and table_1_4_source_url is not None:
            _add_target(
                session,
                stratum_id=national_stratum.id,
                variable="employment_income",
                period=year,
                value=table_1_4_data["total_employment_income_returns"],
                target_type=TargetType.COUNT,
                source_url=table_1_4_source_url,
                source_table="Table 1.4",
            )
            _add_target(
                session,
                stratum_id=national_stratum.id,
                variable="employment_income",
                period=year,
                value=table_1_4_data["total_employment_income"],
                target_type=TargetType.AMOUNT,
                source_url=table_1_4_source_url,
                source_table="Table 1.4",
            )

        for bracket_name, (lower, upper) in AGI_BRACKETS.items():
            constraints = []
            if lower != float("-inf"):
                constraints.append(("adjusted_gross_income", ">=", str(lower)))
            if upper != float("inf"):
                constraints.append(("adjusted_gross_income", "<", str(upper)))

            bracket_stratum = get_or_create_stratum(
                session,
                name=f"US Filers AGI {bracket_name}",
                jurisdiction=Jurisdiction.US_FEDERAL,
                constraints=constraints,
                description=f"Tax filers with AGI in {bracket_name} bracket",
                parent_id=national_stratum.id,
                stratum_group_id="agi_brackets",
            )

            if bracket_name in data["returns_by_agi_bracket"]:
                _add_target(
                    session,
                    stratum_id=bracket_stratum.id,
                    variable="tax_unit_count",
                    period=year,
                    value=data["returns_by_agi_bracket"][bracket_name],
                    target_type=TargetType.COUNT,
                    source_url=source_url,
                )

            if bracket_name in data["agi_by_bracket"]:
                _add_target(
                    session,
                    stratum_id=bracket_stratum.id,
                    variable="adjusted_gross_income",
                    period=year,
                    value=data["agi_by_bracket"][bracket_name],
                    target_type=TargetType.AMOUNT,
                    source_url=source_url,
                )

            if bracket_name in data["income_tax_by_bracket"]:
                _add_target(
                    session,
                    stratum_id=bracket_stratum.id,
                    variable="income_tax_liability",
                    period=year,
                    value=data["income_tax_by_bracket"][bracket_name],
                    target_type=TargetType.AMOUNT,
                    source_url=source_url,
                )

            if table_1_4_data is not None and table_1_4_source_url is not None:
                if (
                    bracket_name
                    in table_1_4_data["employment_income_returns_by_agi_bracket"]
                ):
                    _add_target(
                        session,
                        stratum_id=bracket_stratum.id,
                        variable="employment_income",
                        period=year,
                        value=table_1_4_data[
                            "employment_income_returns_by_agi_bracket"
                        ][bracket_name],
                        target_type=TargetType.COUNT,
                        source_url=table_1_4_source_url,
                        source_table="Table 1.4",
                    )

                if bracket_name in table_1_4_data["employment_income_by_agi_bracket"]:
                    _add_target(
                        session,
                        stratum_id=bracket_stratum.id,
                        variable="employment_income",
                        period=year,
                        value=table_1_4_data["employment_income_by_agi_bracket"][
                            bracket_name
                        ],
                        target_type=TargetType.AMOUNT,
                        source_url=table_1_4_source_url,
                        source_table="Table 1.4",
                    )

    session.commit()


def run_etl(db_path=None) -> None:
    """Run the SOI ETL pipeline."""
    from pathlib import Path

    from .schema import DEFAULT_DB_PATH

    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    engine = init_db(path)

    with Session(engine) as session:
        load_soi_targets(session)
        print(f"Loaded SOI targets to {path}")


if __name__ == "__main__":
    run_etl()
