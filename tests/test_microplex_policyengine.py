"""Tests for Microplex PolicyEngine-US tax adapters."""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd

from arch.targets import DataSource, TargetSpec, TargetType
from micro.us import pipeline as microplex
from micro.us.policyengine import (
    DEFAULT_SOI_INCOME_TAX_VARIABLE,
    PolicyEngineNotAvailableError,
    PolicyEngineTaxConfig,
    add_policyengine_income_tax,
    add_policyengine_income_tax_from_persons,
)


class FakeSimulation:
    """Small stand-in for policyengine_us.Simulation."""

    calls: list[tuple[dict, str, str | None]] = []

    def __init__(self, *, situation):
        self.situation = situation

    def calculate(self, variable, period=None):
        self.calls.append((self.situation, variable, period))
        values = []
        for tax_unit in self.situation["tax_units"].values():
            wages = 0.0
            self_employment = 0.0
            for member in tax_unit["members"]:
                person = self.situation["people"][member]
                wages += person["employment_income"][period]
                self_employment += person["self_employment_income"][period]
            values.append(wages * 0.10 + self_employment * 0.15)
        return np.array(values)


def install_fake_policyengine(monkeypatch):
    FakeSimulation.calls = []
    fake_module = types.ModuleType("policyengine_us")
    fake_module.Simulation = FakeSimulation
    monkeypatch.setitem(sys.modules, "policyengine_us", fake_module)


def test_add_policyengine_income_tax_uses_soi_comparable_variable(monkeypatch):
    install_fake_policyengine(monkeypatch)
    tax_units = pd.DataFrame(
        {
            "wage_income": [50_000.0, 0.0],
            "self_employment_income": [0.0, 20_000.0],
            "interest_income": [100.0, 0.0],
            "state_fips": [6, 48],
            "age": [35, 55],
        }
    )

    result = add_policyengine_income_tax(tax_units, year=2024, config=None)

    assert result["income_tax_liability"].tolist() == [5_000.0, 3_000.0]
    assert result["income_tax_liability_source"].unique().tolist() == [
        f"policyengine_us:{DEFAULT_SOI_INCOME_TAX_VARIABLE}"
    ]
    _, variable, period = FakeSimulation.calls[0]
    assert variable == "income_tax_before_credits"
    assert period == "2024"


def test_add_policyengine_income_tax_batches(monkeypatch):
    install_fake_policyengine(monkeypatch)
    tax_units = pd.DataFrame(
        {
            "wage_income": [10_000.0, 20_000.0, 30_000.0],
            "self_employment_income": [0.0, 0.0, 0.0],
        }
    )

    result = add_policyengine_income_tax(
        tax_units,
        year=2024,
        config=PolicyEngineTaxConfig(batch_size=2),
    )

    assert result["income_tax_liability"].tolist() == [1_000.0, 2_000.0, 3_000.0]
    assert len(FakeSimulation.calls) == 2


def test_add_policyengine_income_tax_from_persons_preserves_entities(monkeypatch):
    install_fake_policyengine(monkeypatch)
    persons = pd.DataFrame(
        {
            "household_id": [1, 1, 1, 2],
            "tax_unit_id": [10, 10, 11, 20],
            "spm_unit_id": [100, 100, 101, 200],
            "person_seq": [1, 2, 3, 1],
            "age": [40, 38, 16, 55],
            "state_fips": [6, 6, 6, 48],
            "weight": [100.0, 100.0, 100.0, 200.0],
            "wage_salary_income": [50_000.0, 20_000.0, 0.0, 0.0],
            "self_employment_income": [0.0, 0.0, 0.0, 20_000.0],
            "total_person_income": [50_000.0, 20_000.0, 0.0, 20_000.0],
        }
    )

    result = add_policyengine_income_tax_from_persons(persons, year=2024)

    assert result["tax_unit_id"].tolist() == [10, 11, 20]
    assert result["person_count"].tolist() == [2, 1, 1]
    assert result["income_tax_liability"].tolist() == [7_000.0, 0.0, 3_000.0]
    situation, variable, period = FakeSimulation.calls[0]
    assert variable == "income_tax_before_credits"
    assert period == "2024"
    assert [len(unit["members"]) for unit in situation["tax_units"].values()] == [
        2,
        1,
        1,
    ]
    assert [len(unit["members"]) for unit in situation["households"].values()] == [
        3,
        1,
    ]


def test_pipeline_merges_hierarchical_policyengine_tax(monkeypatch):
    install_fake_policyengine(monkeypatch)
    targets = [
        TargetSpec(
            variable="income_tax_liability",
            value=10_000.0,
            target_type=TargetType.AMOUNT,
            constraints=[("is_tax_filer", "==", "1")],
            source=DataSource.IRS_SOI,
            period=2024,
            stratum_name="Tax liability holdout",
        )
    ]
    tax_units = pd.DataFrame(
        {
            "household_id": [1, 2],
            "tax_unit_id": [10, 20],
            "wage_income": [70_000.0, 0.0],
            "self_employment_income": [0.0, 20_000.0],
        }
    )
    persons = pd.DataFrame(
        {
            "household_id": [1, 1, 2],
            "tax_unit_id": [10, 10, 20],
            "person_seq": [1, 2, 1],
            "age": [40, 38, 55],
            "state_fips": [6, 6, 48],
            "weight": [100.0, 100.0, 200.0],
            "wage_salary_income": [50_000.0, 20_000.0, 0.0],
            "self_employment_income": [0.0, 0.0, 20_000.0],
        }
    )

    result = microplex.maybe_add_policyengine_income_tax(
        tax_units,
        targets,
        year=2024,
        persons=persons,
    )

    assert result["income_tax_liability"].tolist() == [7_000.0, 3_000.0]
    assert len(FakeSimulation.calls) == 1
    situation = FakeSimulation.calls[0][0]
    assert [len(unit["members"]) for unit in situation["tax_units"].values()] == [
        2,
        1,
    ]


def test_policyengine_missing_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "policyengine_us", None)

    try:
        add_policyengine_income_tax(pd.DataFrame({"wage_income": [1.0]}), year=2024)
    except PolicyEngineNotAvailableError as exc:
        assert "PolicyEngine-US is required" in str(exc)
    else:
        raise AssertionError("Expected PolicyEngineNotAvailableError")


def test_run_pipeline_calculates_income_tax_when_targets_require_it(
    tmp_path,
    monkeypatch,
):
    install_fake_policyengine(monkeypatch)
    cps = pd.DataFrame(
        {
            "total_person_income": [50_000.0] * 150,
            "wage_salary_income": [50_000.0] * 150,
            "self_employment_income": [0.0] * 150,
            "weight": [100.0] * 150,
        }
    )
    cps_path = tmp_path / "cps.parquet"
    cps.to_parquet(cps_path, index=False)

    targets = [
        TargetSpec(
            variable="tax_unit_count",
            value=15_000.0,
            target_type=TargetType.COUNT,
            constraints=[("is_tax_filer", "==", "1")],
            source=DataSource.IRS_SOI,
            period=2024,
            stratum_name="All filers",
        ),
        TargetSpec(
            variable="income_tax_liability",
            value=750_000.0,
            target_type=TargetType.AMOUNT,
            constraints=[("is_tax_filer", "==", "1")],
            source=DataSource.IRS_SOI,
            period=2024,
            stratum_name="Tax liability holdout",
        ),
    ]
    monkeypatch.setattr(
        microplex,
        "compose_microplex_targets",
        lambda *args, **kwargs: microplex.TargetCompositionResult(
            targets=targets,
            diagnostics=pd.DataFrame(),
        ),
    )

    result = microplex.run_pipeline(
        year=2024,
        limit=150,
        cps_path=cps_path,
        dry_run=True,
        min_target_obs=1,
    )

    assert "income_tax_liability" in result
    assert result["income_tax_liability"].iloc[0] == 5_000.0


def test_calibrate_weights_evaluates_income_tax_holdout():
    df = pd.DataFrame(
        {
            "weight": [1.0, 1.0, 1.0],
            "is_tax_filer": [1, 1, 1],
            "adjusted_gross_income": [10_000.0, 20_000.0, 30_000.0],
            "income_tax_liability": [100.0, 200.0, 300.0],
        }
    )
    targets = [
        TargetSpec(
            variable="tax_unit_count",
            value=3.0,
            target_type=TargetType.COUNT,
            constraints=[("is_tax_filer", "==", "1")],
            source=DataSource.IRS_SOI,
            period=2024,
            stratum_name="All filers",
        ),
        TargetSpec(
            variable="income_tax_liability",
            value=600.0,
            target_type=TargetType.AMOUNT,
            constraints=[("is_tax_filer", "==", "1")],
            source=DataSource.IRS_SOI,
            period=2024,
            stratum_name="Tax liability holdout",
        ),
    ]

    result = microplex.calibrate_weights(
        df,
        targets,
        min_obs=1,
        holdout_variables=("income_tax_liability",),
        verbose=False,
    )

    diagnostics = result.diagnostics.set_index("stratum")
    assert diagnostics.loc["Tax liability holdout", "role"] == "holdout"
    assert diagnostics.loc["Tax liability holdout", "post_value"] == 600.0
