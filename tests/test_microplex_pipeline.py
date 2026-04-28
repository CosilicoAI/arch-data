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
