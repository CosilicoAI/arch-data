"""Linked Microplex entity frames for US person-household microdata."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MicroplexEntityFrames:
    """Linked entity tables produced from person-level ASEC-like inputs."""

    households: pd.DataFrame
    persons: pd.DataFrame
    tax_units: pd.DataFrame


def build_microplex_entities(persons: pd.DataFrame) -> MicroplexEntityFrames:
    """
    Build household, person, and tax-unit frames from person-level microdata.

    Households and persons are the primitive inputs. Tax units are an assignment
    over persons inside households: Census ``TAX_ID`` is used when available,
    otherwise each person falls back to their own provisional tax unit.
    """
    normalized_persons = normalize_persons(persons)
    tax_units = build_tax_units_from_persons(normalized_persons)
    households = build_households_from_persons(normalized_persons, tax_units)
    return MicroplexEntityFrames(
        households=households,
        persons=normalized_persons,
        tax_units=tax_units,
    )


def normalize_persons(persons: pd.DataFrame) -> pd.DataFrame:
    """Return a person frame with stable entity IDs and normalized columns."""
    if persons.empty:
        raise ValueError("Cannot build Microplex entities from an empty person frame.")

    df = _unpack_raw_data(persons).reset_index(drop=True)
    df["_row_position"] = np.arange(len(df))

    household_id = _first_identifier(df, ["household_id", "ph_seq", "PH_SEQ"])
    household_id = household_id.where(household_id.notna(), df["_row_position"])
    df["household_id"] = household_id
    df["household_entity_id"] = "hh:" + household_id.map(_format_identifier)

    person_seq = _first_identifier(
        df,
        ["person_seq", "line_number", "a_lineno", "A_LINENO", "p_seq", "P_SEQ"],
    )
    person_seq = person_seq.where(person_seq.notna(), df["_row_position"] + 1)
    df["person_seq"] = person_seq

    person_id = _first_identifier(df, ["person_id"])
    person_id = person_id.where(
        person_id.notna(),
        df["household_entity_id"] + "|p:" + person_seq.map(_format_identifier),
    )
    df["person_id"] = person_id
    df["person_entity_id"] = (
        df["household_entity_id"] + "|p:" + person_id.map(_format_identifier)
    )

    tax_unit_id = _first_identifier(df, ["tax_unit_id", "tax_id", "TAX_ID"])
    has_source_tax_unit = tax_unit_id.notna()
    tax_unit_id = tax_unit_id.where(tax_unit_id.notna(), person_id)
    df["tax_unit_id"] = tax_unit_id
    df["tax_unit_assignment_source"] = np.where(
        has_source_tax_unit,
        "source",
        "person_fallback",
    )
    df["tax_unit_entity_id"] = (
        df["household_entity_id"] + "|tu:" + tax_unit_id.map(_format_identifier)
    )

    spm_unit_id = _first_identifier(df, ["spm_unit_id", "spm_id", "SPM_ID"])
    spm_unit_id = spm_unit_id.where(spm_unit_id.notna(), household_id)
    df["spm_unit_id"] = spm_unit_id
    df["spm_unit_entity_id"] = (
        df["household_entity_id"] + "|spm:" + spm_unit_id.map(_format_identifier)
    )

    family_id = _first_identifier(
        df,
        ["family_id", "pf_seq", "PF_SEQ", "family_seq"],
    )
    family_id = family_id.where(family_id.notna(), household_id)
    df["family_id"] = family_id
    df["family_entity_id"] = (
        df["household_entity_id"] + "|fam:" + family_id.map(_format_identifier)
    )

    df["age"] = _numeric_first(df, ["age", "a_age", "A_AGE"], default=40)
    df["state_fips"] = _numeric_first(
        df,
        ["state_fips", "gestfips", "GESTFIPS"],
        default=0,
    ).astype(int)
    df["weight"] = _normalized_weight(df)
    df["total_person_income"] = _numeric_first(
        df,
        ["total_person_income", "total_income", "income", "ptotval", "PTOTVAL"],
    )
    df["wage_income"] = _numeric_first(
        df,
        ["wage_income", "wage_salary_income", "wsal_val", "WSAL_VAL"],
    )
    df["self_employment_income"] = _numeric_first(
        df,
        ["self_employment_income", "semp_val", "SEMP_VAL"],
    )
    df["farm_self_employment_income"] = _numeric_first(
        df,
        ["farm_self_employment_income", "frse_val", "FRSE_VAL"],
    )
    df["interest_income"] = _numeric_first(
        df,
        ["interest_income", "int_val", "INT_VAL"],
    )
    df["dividend_income"] = _numeric_first(
        df,
        ["dividend_income", "div_val", "DIV_VAL"],
    )
    df["rental_income"] = _numeric_first(
        df,
        ["rental_income", "rnt_val", "RNT_VAL"],
    )
    df["unemployment_compensation"] = _numeric_first(
        df,
        ["unemployment_compensation", "uc_val", "UC_VAL"],
    )
    df["other_income"] = _numeric_first(df, ["other_income", "oi_val", "OI_VAL"])
    return df


def build_tax_units_from_persons(persons: pd.DataFrame) -> pd.DataFrame:
    """Build one tax-unit row per tax-unit assignment in a person frame."""
    rows: list[dict[str, Any]] = []
    for _, group in persons.groupby("tax_unit_entity_id", sort=False):
        head = group.iloc[0]
        wage_income = float(group["wage_income"].sum())
        self_employment_income = float(
            group["self_employment_income"].sum()
            + group["farm_self_employment_income"].sum()
        )
        interest_income = float(group["interest_income"].sum())
        dividend_income = float(group["dividend_income"].sum())
        rental_income = float(group["rental_income"].sum())
        unemployment = float(group["unemployment_compensation"].sum())
        other_income = float(group["other_income"].sum())
        total_income = float(group["total_person_income"].sum())
        if total_income == 0:
            total_income = (
                wage_income
                + self_employment_income
                + interest_income
                + dividend_income
                + rental_income
                + unemployment
                + other_income
            )

        se_tax_adjustment = max(self_employment_income, 0.0) * 0.0765 / 2
        adjusted_gross_income = (
            wage_income
            + self_employment_income
            - se_tax_adjustment
            + interest_income
            + dividend_income
            + rental_income
            + unemployment
            + other_income
        )
        is_tax_filer = (
            (total_income > 13_850)
            or (wage_income > 0)
            or (self_employment_income != 0)
        )

        rows.append(
            {
                "tax_unit_entity_id": head["tax_unit_entity_id"],
                "household_entity_id": head["household_entity_id"],
                "household_id": head["household_id"],
                "tax_unit_id": head["tax_unit_id"],
                "tax_unit_assignment_source": head["tax_unit_assignment_source"],
                "person_count": int(len(group)),
                "age": float(head["age"]),
                "state_fips": int(head["state_fips"]),
                "weight": float(group["weight"].iloc[0]),
                "total_income": total_income,
                "wage_income": wage_income,
                "self_employment_income": self_employment_income,
                "interest_income": interest_income,
                "dividend_income": dividend_income,
                "rental_income": rental_income,
                "unemployment_compensation": unemployment,
                "other_income": other_income,
                "adjusted_gross_income": float(adjusted_gross_income),
                "is_tax_filer": int(is_tax_filer),
            }
        )
    return pd.DataFrame(rows)


def build_households_from_persons(
    persons: pd.DataFrame,
    tax_units: pd.DataFrame,
) -> pd.DataFrame:
    """Build one household row per household assignment in a person frame."""
    person_grouped = persons.groupby("household_entity_id", sort=False)
    households = person_grouped.agg(
        household_id=("household_id", "first"),
        state_fips=("state_fips", "first"),
        weight=("weight", "first"),
        person_count=("person_entity_id", "size"),
    ).reset_index()
    tax_unit_counts = tax_units.groupby("household_entity_id", sort=False).size()
    households["tax_unit_count"] = (
        households["household_entity_id"].map(tax_unit_counts).fillna(0).astype(int)
    )
    return households


def with_household_weights(
    entities: MicroplexEntityFrames,
    households: pd.DataFrame,
) -> MicroplexEntityFrames:
    """Map calibrated household weights onto linked person and tax-unit frames."""
    weight_columns = [
        "household_entity_id",
        "original_weight",
        "weight",
        "calibrated_weight",
        "weight_adjustment",
    ]
    available_weight_columns = [
        column for column in weight_columns if column in households.columns
    ]
    household_weights = households[available_weight_columns].copy()

    persons = _replace_weight_columns(entities.persons, household_weights)
    tax_units = _replace_weight_columns(entities.tax_units, household_weights)
    return replace(entities, households=households, persons=persons, tax_units=tax_units)


def write_microplex_entities(
    entities: MicroplexEntityFrames,
    output_dir: Path,
) -> None:
    """Write linked Microplex entity frames to a local directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    entities.households.to_parquet(output_dir / "households.parquet", index=False)
    entities.persons.to_parquet(output_dir / "persons.parquet", index=False)
    entities.tax_units.to_parquet(output_dir / "tax_units.parquet", index=False)


def _replace_weight_columns(
    frame: pd.DataFrame,
    household_weights: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.drop(
        columns=[
            column
            for column in ["original_weight", "calibrated_weight", "weight_adjustment"]
            if column in frame.columns
        ],
    )
    if "weight" in output.columns:
        output = output.drop(columns=["weight"])
    return output.merge(household_weights, on="household_entity_id", how="left")


def _unpack_raw_data(persons: pd.DataFrame) -> pd.DataFrame:
    df = persons.copy()
    if "raw_data" not in df.columns:
        return df
    columns = [
        "A_AGE",
        "A_LINENO",
        "GESTFIPS",
        "MARSUPWT",
        "PH_SEQ",
        "P_SEQ",
        "PF_SEQ",
        "SPM_ID",
        "TAX_ID",
    ]
    for column in columns:
        lower_column = column.lower()
        if column not in df.columns and lower_column not in df.columns:
            df[lower_column] = df["raw_data"].apply(
                lambda value: value.get(column) if isinstance(value, dict) else None
            )
    return df


def _first_identifier(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=df.index, dtype="object")
    for column in columns:
        if column in df.columns:
            result = result.where(result.notna(), df[column])
    return result


def _numeric_first(
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
        return pd.to_numeric(df["marsupwt"], errors="coerce").fillna(0.0).div(100)
    if "MARSUPWT" in df.columns:
        return pd.to_numeric(df["MARSUPWT"], errors="coerce").fillna(0.0).div(100)
    return pd.Series(1.0, index=df.index, dtype=float)


def _format_identifier(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)
