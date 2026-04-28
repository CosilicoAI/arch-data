"""PolicyEngine-US adapters for Microplex tax-unit calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SOI_INCOME_TAX_VARIABLE = "income_tax_before_credits"
DEFAULT_INCOME_TAX_COLUMN = "income_tax_liability"


class PolicyEngineNotAvailableError(ImportError):
    """Raised when PolicyEngine-US is needed but not installed."""


@dataclass(frozen=True)
class PolicyEngineTaxConfig:
    """Configuration for PolicyEngine-US tax calculations."""

    policyengine_variable: str = DEFAULT_SOI_INCOME_TAX_VARIABLE
    output_column: str = DEFAULT_INCOME_TAX_COLUMN
    batch_size: int = 1_000


def add_policyengine_income_tax(
    tax_units: pd.DataFrame,
    *,
    year: int,
    config: PolicyEngineTaxConfig | None = None,
) -> pd.DataFrame:
    """
    Add SOI-comparable income tax liability using PolicyEngine-US.

    SOI Publication 1304 Table 1.1's "total income tax" target corresponds to
    the PolicyEngine-US ``income_tax_before_credits`` variable in the
    PolicyEngine-US-data SOI utilities. Microplex keeps the Arch-facing target
    variable name, ``income_tax_liability``, and records the PE result there.
    """
    config = config or PolicyEngineTaxConfig()
    if config.batch_size <= 0:
        raise ValueError("PolicyEngine batch_size must be positive.")

    simulation_cls = _policyengine_simulation_cls()
    result = tax_units.copy()
    values = np.zeros(len(result), dtype=float)

    for start in range(0, len(result), config.batch_size):
        stop = min(start + config.batch_size, len(result))
        batch = result.iloc[start:stop]
        situation = _policyengine_situation(batch, year=year)
        simulation = simulation_cls(situation=situation)
        calculated = simulation.calculate(
            config.policyengine_variable,
            period=str(year),
        )
        values[start:stop] = np.asarray(calculated, dtype=float).reshape(-1)

    result[config.output_column] = values
    result[f"{config.output_column}_source"] = (
        f"policyengine_us:{config.policyengine_variable}"
    )
    return result


def add_policyengine_income_tax_from_persons(
    persons: pd.DataFrame,
    *,
    year: int,
    config: PolicyEngineTaxConfig | None = None,
) -> pd.DataFrame:
    """
    Calculate tax-unit income tax from a person/household hierarchy.

    This is the preferred bridge toward the PolicyEngine-US-data shape:
    Microplex starts from person rows grouped into households, derives entity
    links, builds a PE situation over the full household graph, and then returns
    one row per derived tax unit.
    """
    config = config or PolicyEngineTaxConfig()
    if config.batch_size <= 0:
        raise ValueError("PolicyEngine batch_size must be positive.")

    simulation_cls = _policyengine_simulation_cls()
    normalized = _normalize_person_hierarchy(persons)
    result = _tax_unit_rows_from_persons(normalized)
    values = np.zeros(len(result), dtype=float)

    key_columns = ["_household_entity_id", "_tax_unit_entity_id"]
    for start in range(0, len(result), config.batch_size):
        stop = min(start + config.batch_size, len(result))
        batch_units = result.iloc[start:stop][key_columns]
        batch_people = normalized.merge(batch_units, on=key_columns, how="inner")
        situation, tax_unit_order = _policyengine_situation_from_persons(
            batch_people,
            year=year,
        )
        simulation = simulation_cls(situation=situation)
        calculated = simulation.calculate(
            config.policyengine_variable,
            period=str(year),
        )
        calculated_values = np.asarray(calculated, dtype=float).reshape(-1)
        if len(calculated_values) != len(tax_unit_order):
            raise ValueError(
                "PolicyEngine returned a different number of tax-unit values "
                "than the person hierarchy provided."
            )
        value_by_entity = dict(zip(tax_unit_order, calculated_values))
        values[start:stop] = [
            value_by_entity[entity_id]
            for entity_id in result.iloc[start:stop]["_tax_unit_entity_id"]
        ]

    result[config.output_column] = values
    result[f"{config.output_column}_source"] = (
        f"policyengine_us:{config.policyengine_variable}"
    )
    return result.drop(columns=key_columns)


def policyengine_us_available() -> bool:
    """Return whether PolicyEngine-US can be imported."""
    try:
        _policyengine_simulation_cls()
    except PolicyEngineNotAvailableError:
        return False
    return True


def _policyengine_simulation_cls() -> Any:
    try:
        from policyengine_us import Simulation
    except ImportError as exc:
        raise PolicyEngineNotAvailableError(
            "PolicyEngine-US is required to calculate income_tax_liability. "
            "Install policyengine-us in the environment to enable these "
            "targets."
        ) from exc
    return Simulation


def _normalize_person_hierarchy(persons: pd.DataFrame) -> pd.DataFrame:
    if persons.empty:
        raise ValueError("Cannot build a PolicyEngine situation with no persons.")

    df = persons.copy().reset_index(drop=True)
    df["_row_position"] = np.arange(len(df))

    household_id = _first_identifier(df, ["household_id", "ph_seq", "PH_SEQ"])
    household_id = household_id.where(household_id.notna(), df["_row_position"])
    df["household_id"] = household_id
    df["_household_entity_id"] = "hh:" + household_id.map(_format_identifier)

    tax_unit_id = _first_identifier(df, ["tax_unit_id", "tax_id", "TAX_ID"])
    tax_unit_id = tax_unit_id.where(tax_unit_id.notna(), household_id)
    df["tax_unit_id"] = tax_unit_id
    df["_tax_unit_entity_id"] = (
        df["_household_entity_id"] + "|tu:" + tax_unit_id.map(_format_identifier)
    )

    person_id = _first_identifier(
        df,
        ["person_id", "a_lineno", "A_LINENO", "person_seq", "p_seq", "P_SEQ"],
    )
    person_id = person_id.where(person_id.notna(), df["_row_position"])
    df["person_id"] = person_id
    df["_person_entity_id"] = (
        df["_household_entity_id"] + "|p:" + person_id.map(_format_identifier)
    )

    spm_unit_id = _first_identifier(df, ["spm_unit_id", "spm_id", "SPM_ID"])
    spm_unit_id = spm_unit_id.where(spm_unit_id.notna(), household_id)
    df["spm_unit_id"] = spm_unit_id
    df["_spm_unit_entity_id"] = (
        df["_household_entity_id"] + "|spm:" + spm_unit_id.map(_format_identifier)
    )

    family_id = _first_identifier(
        df,
        ["family_id", "pf_seq", "PF_SEQ", "family_seq"],
    )
    family_id = family_id.where(family_id.notna(), household_id)
    df["family_id"] = family_id
    df["_family_entity_id"] = (
        df["_household_entity_id"] + "|fam:" + family_id.map(_format_identifier)
    )

    marital_unit_id = _first_identifier(df, ["marital_unit_id"])
    marital_unit_id = marital_unit_id.where(marital_unit_id.notna(), tax_unit_id)
    df["marital_unit_id"] = marital_unit_id
    df["_marital_unit_entity_id"] = (
        df["_household_entity_id"]
        + "|mu:"
        + marital_unit_id.map(_format_identifier)
    )

    df["age"] = _numeric_first_series(df, ["age", "a_age", "A_AGE"], default=40)
    df["state_fips"] = _numeric_first_series(
        df,
        ["state_fips", "gestfips", "GESTFIPS"],
        default=6,
    )
    df["weight"] = _normalized_weight(df)
    df["employment_income"] = _numeric_first_series(
        df,
        ["employment_income", "wage_income", "wage_salary_income", "wsal_val", "WSAL_VAL"],
    )
    df["self_employment_income"] = _numeric_first_series(
        df,
        ["self_employment_income", "semp_val", "SEMP_VAL"],
    )
    df["farm_operations_income"] = _numeric_first_series(
        df,
        ["farm_operations_income", "farm_self_employment_income", "frse_val", "FRSE_VAL"],
    )
    df["taxable_interest_income"] = _numeric_first_series(
        df,
        ["taxable_interest_income", "interest_income", "int_val", "INT_VAL"],
    )
    df["qualified_dividend_income"] = _numeric_first_series(
        df,
        ["qualified_dividend_income", "dividend_income", "div_val", "DIV_VAL"],
    )
    df["rental_income"] = _numeric_first_series(
        df,
        ["rental_income", "rnt_val", "RNT_VAL"],
    )
    df["unemployment_compensation"] = _numeric_first_series(
        df,
        ["unemployment_compensation", "uc_val", "UC_VAL"],
    )
    df["total_person_income"] = _numeric_first_series(
        df,
        ["total_person_income", "total_income", "income", "ptotval", "PTOTVAL"],
    )

    role_frame = _tax_unit_roles(df)
    df["is_tax_unit_head"] = role_frame["is_tax_unit_head"]
    df["is_tax_unit_spouse"] = role_frame["is_tax_unit_spouse"]
    df["is_tax_unit_dependent"] = role_frame["is_tax_unit_dependent"]
    return df


def _tax_unit_rows_from_persons(persons: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["_household_entity_id", "_tax_unit_entity_id"]
    for _, group in persons.groupby(group_columns, sort=False):
        head = group.iloc[0]
        self_employment_income = (
            group["self_employment_income"].sum()
            + group["farm_operations_income"].sum()
        )
        wage_income = group["employment_income"].sum()
        se_tax_adjustment = max(float(self_employment_income), 0.0) * 0.0765 / 2
        adjusted_gross_income = (
            wage_income
            + self_employment_income
            - se_tax_adjustment
            + group["taxable_interest_income"].sum()
            + group["qualified_dividend_income"].sum()
            + group["rental_income"].sum()
            + group["unemployment_compensation"].sum()
        )
        total_income = group["total_person_income"].sum()
        if total_income == 0:
            total_income = (
                wage_income
                + self_employment_income
                + group["taxable_interest_income"].sum()
                + group["qualified_dividend_income"].sum()
                + group["rental_income"].sum()
                + group["unemployment_compensation"].sum()
            )

        is_filer = (
            (total_income > 13_850)
            or (wage_income > 0)
            or (self_employment_income != 0)
        )
        rows.append(
            {
                "_household_entity_id": head["_household_entity_id"],
                "_tax_unit_entity_id": head["_tax_unit_entity_id"],
                "household_id": head["household_id"],
                "tax_unit_id": head["tax_unit_id"],
                "person_count": len(group),
                "age": float(head["age"]),
                "state_fips": int(head["state_fips"]),
                "weight": float(group["weight"].iloc[0]),
                "total_income": float(total_income),
                "wage_income": float(wage_income),
                "self_employment_income": float(self_employment_income),
                "interest_income": float(group["taxable_interest_income"].sum()),
                "dividend_income": float(group["qualified_dividend_income"].sum()),
                "rental_income": float(group["rental_income"].sum()),
                "unemployment_compensation": float(
                    group["unemployment_compensation"].sum()
                ),
                "adjusted_gross_income": float(adjusted_gross_income),
                "is_tax_filer": int(is_filer),
            }
        )
    return pd.DataFrame(rows)


def _policyengine_situation_from_persons(
    persons: pd.DataFrame,
    *,
    year: int,
) -> tuple[dict[str, Any], list[str]]:
    year_key = str(year)
    people: dict[str, dict[str, Any]] = {}
    tax_units: dict[str, dict[str, Any]] = {}
    households: dict[str, dict[str, Any]] = {}
    families: dict[str, dict[str, Any]] = {}
    spm_units: dict[str, dict[str, Any]] = {}
    marital_units: dict[str, dict[str, Any]] = {}

    for _, row in persons.iterrows():
        person_id = str(row["_person_entity_id"])
        people[person_id] = {
            "age": {year_key: int(row["age"])},
            "employment_income": {year_key: float(row["employment_income"])},
            "self_employment_income": {
                year_key: float(row["self_employment_income"])
            },
            "farm_operations_income": {
                year_key: float(row["farm_operations_income"])
            },
            "taxable_interest_income": {
                year_key: float(row["taxable_interest_income"])
            },
            "qualified_dividend_income": {
                year_key: float(row["qualified_dividend_income"])
            },
            "rental_income": {year_key: float(row["rental_income"])},
            "unemployment_compensation": {
                year_key: float(row["unemployment_compensation"])
            },
            "is_tax_unit_head": {year_key: bool(row["is_tax_unit_head"])},
            "is_tax_unit_spouse": {year_key: bool(row["is_tax_unit_spouse"])},
            "is_tax_unit_dependent": {
                year_key: bool(row["is_tax_unit_dependent"])
            },
        }
        _append_member(tax_units, str(row["_tax_unit_entity_id"]), person_id)
        _append_member(households, str(row["_household_entity_id"]), person_id)
        _append_member(families, str(row["_family_entity_id"]), person_id)
        _append_member(spm_units, str(row["_spm_unit_entity_id"]), person_id)
        _append_member(
            marital_units,
            str(row["_marital_unit_entity_id"]),
            person_id,
        )

    household_state = persons.groupby("_household_entity_id", sort=False)[
        "state_fips"
    ].first()
    for household_id, state_fips in household_state.items():
        households[str(household_id)]["state_fips"] = {year_key: int(state_fips)}

    for spm_unit in spm_units.values():
        spm_unit["snap"] = {year_key: 0}
        spm_unit["tanf"] = {year_key: 0}
        spm_unit["free_school_meals"] = {year_key: 0}
        spm_unit["reduced_price_school_meals"] = {year_key: 0}

    return (
        {
            "people": people,
            "tax_units": tax_units,
            "households": households,
            "families": families,
            "spm_units": spm_units,
            "marital_units": marital_units,
        },
        list(tax_units.keys()),
    )


def _policyengine_situation(batch: pd.DataFrame, *, year: int) -> dict[str, Any]:
    year_key = str(year)
    people: dict[str, dict[str, Any]] = {}
    tax_units: dict[str, dict[str, Any]] = {}
    households: dict[str, dict[str, Any]] = {}
    families: dict[str, dict[str, Any]] = {}
    spm_units: dict[str, dict[str, Any]] = {}
    marital_units: dict[str, dict[str, Any]] = {}

    for position, (_, row) in enumerate(batch.iterrows()):
        person_id = f"p{position}"
        tax_unit_id = f"tu{position}"
        household_id = f"hh{position}"
        family_id = f"fam{position}"
        spm_unit_id = f"spm{position}"
        marital_unit_id = f"mu{position}"
        members = [person_id]

        people[person_id] = {
            "age": {year_key: int(_row_number(row, ["age"], default=40))},
            "employment_income": {
                year_key: _row_number(row, ["wage_income", "employment_income"])
            },
            "self_employment_income": {
                year_key: _row_number(row, ["self_employment_income"])
            },
            "interest_income": {year_key: _row_number(row, ["interest_income"])},
            "dividend_income": {year_key: _row_number(row, ["dividend_income"])},
            "rental_income": {year_key: _row_number(row, ["rental_income"])},
            "unemployment_compensation": {
                year_key: _row_number(row, ["unemployment_compensation"])
            },
            "is_tax_unit_head": {year_key: True},
            "is_tax_unit_spouse": {year_key: False},
            "is_tax_unit_dependent": {year_key: False},
        }
        tax_units[tax_unit_id] = {"members": members}
        households[household_id] = {
            "members": members,
            "state_fips": {
                year_key: int(_row_number(row, ["state_fips"], default=6))
            },
        }
        families[family_id] = {"members": members}
        spm_units[spm_unit_id] = {
            "members": members,
            "snap": {year_key: 0},
            "tanf": {year_key: 0},
            "free_school_meals": {year_key: 0},
            "reduced_price_school_meals": {year_key: 0},
        }
        marital_units[marital_unit_id] = {"members": members}

    return {
        "people": people,
        "tax_units": tax_units,
        "households": households,
        "families": families,
        "spm_units": spm_units,
        "marital_units": marital_units,
    }


def _append_member(
    entities: dict[str, dict[str, Any]],
    entity_id: str,
    person_id: str,
) -> None:
    entities.setdefault(entity_id, {"members": []})["members"].append(person_id)


def _first_identifier(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=df.index, dtype="object")
    for column in columns:
        if column in df.columns:
            values = df[column]
            result = result.where(result.notna(), values)
    return result


def _format_identifier(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _numeric_first_series(
    df: pd.DataFrame,
    columns: list[str],
    *,
    default: float = 0.0,
) -> pd.Series:
    result = pd.Series(default, index=df.index, dtype=float)
    has_value = pd.Series(False, index=df.index)
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        take = ~has_value & values.notna()
        result.loc[take] = values.loc[take].astype(float)
        has_value.loc[take] = True
    return result.fillna(default)


def _normalized_weight(df: pd.DataFrame) -> pd.Series:
    if "weight" in df.columns:
        return pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    if "march_supplement_weight" in df.columns:
        return (
            pd.to_numeric(df["march_supplement_weight"], errors="coerce")
            .fillna(0.0)
            .div(100)
        )
    if "marsupwt" in df.columns:
        return (
            pd.to_numeric(df["marsupwt"], errors="coerce")
            .fillna(0.0)
            .div(100)
        )
    if "MARSUPWT" in df.columns:
        return (
            pd.to_numeric(df["MARSUPWT"], errors="coerce")
            .fillna(0.0)
            .div(100)
        )
    return pd.Series(1.0, index=df.index, dtype=float)


def _tax_unit_roles(df: pd.DataFrame) -> pd.DataFrame:
    explicit = [
        "is_tax_unit_head",
        "is_tax_unit_spouse",
        "is_tax_unit_dependent",
    ]
    if all(column in df.columns for column in explicit):
        return pd.DataFrame(
            {
                column: df[column].astype(bool)
                for column in explicit
            },
            index=df.index,
        )

    order = df.groupby("_tax_unit_entity_id", sort=False).cumcount()
    group_size = df.groupby("_tax_unit_entity_id", sort=False)[
        "_person_entity_id"
    ].transform("size")
    adult = df["age"] >= 18
    head = order == 0
    spouse = (order == 1) & adult & (group_size > 1)
    dependent = ~(head | spouse)
    return pd.DataFrame(
        {
            "is_tax_unit_head": head,
            "is_tax_unit_spouse": spouse,
            "is_tax_unit_dependent": dependent,
        },
        index=df.index,
    )


def _row_number(
    row: pd.Series,
    columns: list[str],
    *,
    default: float = 0.0,
) -> float:
    for column in columns:
        if column in row and pd.notna(row[column]):
            return float(row[column])
    return default
