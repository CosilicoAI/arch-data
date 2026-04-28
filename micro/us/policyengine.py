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
