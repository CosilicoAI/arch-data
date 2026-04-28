"""Tests for the Microplex build pipeline."""

import pandas as pd

from arch.pipelines import microplex
from arch.targets import DataSource, TargetSpec, TargetType


def test_build_tax_units_accepts_local_cps_columns():
    cps = pd.DataFrame(
        {
            "total_person_income": [20_000, 0, 50_000],
            "wage_salary_income": [20_000, 0, 45_000],
            "self_employment_income": [0, 0, 5_000],
            "farm_self_employment_income": [0, 0, 0],
            "weight": [100.0, 200.0, 300.0],
            "state_fips": [6, 6, 36],
            "age": [30, 10, 40],
        }
    )

    tax_units = microplex.build_tax_units(cps)

    assert len(tax_units) == 2
    assert tax_units["weight"].tolist() == [100.0, 300.0]
    assert tax_units["is_tax_filer"].tolist() == [1, 1]
    assert tax_units["adjusted_gross_income"].tolist() == [20_000.0, 49_808.75]


def test_build_tax_units_aggregates_census_tax_unit_ids():
    cps = pd.DataFrame(
        {
            "household_id": [1, 1, 1],
            "tax_unit_id": [101, 101, 102],
            "person_seq": [1, 2, 3],
            "age": [40, 38, 10],
            "weight": [100.0, 100.0, 80.0],
            "total_person_income": [50_000, 20_000, 0],
            "wage_salary_income": [50_000, 20_000, 0],
            "self_employment_income": [0, 0, 0],
            "farm_self_employment_income": [0, 0, 0],
            "interest_income": [100, 50, 0],
            "dividend_income": [200, 100, 0],
            "rental_income": [0, 0, 0],
            "unemployment_compensation": [0, 0, 0],
            "other_income": [0, 0, 0],
        }
    )

    tax_units = microplex.build_tax_units(cps)

    assert len(tax_units) == 1
    assert tax_units.iloc[0]["tax_unit_id"] == 101
    assert tax_units.iloc[0]["weight"] == 100.0
    assert tax_units.iloc[0]["person_count"] == 2
    assert tax_units.iloc[0]["wage_income"] == 70_000
    assert tax_units.iloc[0]["adjusted_gross_income"] == 70_450


def test_run_pipeline_can_write_local_microplex(tmp_path, monkeypatch):
    n = 150
    cps = pd.DataFrame(
        {
            "total_person_income": [20_000] * n,
            "wage_salary_income": [20_000] * n,
            "self_employment_income": [0] * n,
            "weight": [100.0] * n,
        }
    )
    cps_path = tmp_path / "cps.parquet"
    output_path = tmp_path / "microplex.parquet"
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
        )
    ]
    monkeypatch.setattr(microplex, "load_targets_from_db", lambda *args, **kwargs: targets)
    monkeypatch.setattr(
        microplex,
        "get_soi_aging_factors",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("already-current SOI targets should not be aged")
        ),
    )

    result = microplex.run_pipeline(
        year=2024,
        limit=n,
        cps_path=cps_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert len(result) == n
    assert "weight_adjustment" in result.columns
    assert pd.read_parquet(output_path).shape[0] == n


def test_calibrate_weights_preserves_per_target_diagnostics():
    df = pd.DataFrame(
        {
            "weight": [1.0, 1.0, 1.0],
            "is_tax_filer": [1, 1, 1],
            "adjusted_gross_income": [10_000.0, 50_000.0, 100_000.0],
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
            variable="adjusted_gross_income",
            value=160_000.0,
            target_type=TargetType.AMOUNT,
            constraints=[("is_tax_filer", "==", "1")],
            source=DataSource.IRS_SOI,
            period=2024,
            stratum_name="All filers",
        ),
    ]

    result = microplex.calibrate_weights(
        df,
        targets,
        include_amounts=True,
        min_obs=1,
        verbose=False,
    )

    assert len(result.diagnostics) == 2
    assert result.diagnostics["status"].tolist() == ["used", "used"]
    assert set(result.targets_after) == {
        "tax_unit_count|count|All filers",
        "adjusted_gross_income|amount|All filers",
    }
