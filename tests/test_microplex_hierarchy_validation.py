"""Tests for flat-vs-household Microplex calibration comparison."""

import pandas as pd

from arch.targets import DataSource, TargetSpec, TargetType
from micro.us.hierarchy_validation import (
    compare_flat_vs_household_calibration,
    compare_target_diagnostics,
)
from micro.us.targets import TargetCompositionResult


def test_compare_target_diagnostics_aligns_targets():
    flat = pd.DataFrame(
        {
            "target_index": [0],
            "source": ["irs-soi"],
            "variable": ["tax_unit_count"],
            "target_type": ["count"],
            "period": [2024],
            "source_period": [2024],
            "stratum": ["All filers"],
            "constraints": ["[]"],
            "target_value": [2.0],
            "role": ["active"],
            "status": ["used"],
            "drop_reason": [""],
            "n_obs": [2],
            "pre_error": [0.2],
            "post_error": [0.01],
        }
    )
    household = flat.copy()
    household["post_error"] = -0.03

    comparison = compare_target_diagnostics(flat, household)

    assert len(comparison) == 1
    assert comparison.iloc[0]["flat_abs_post_error"] == 0.01
    assert comparison.iloc[0]["household_abs_post_error"] == 0.03
    assert abs(comparison.iloc[0]["abs_post_error_delta"] - 0.02) < 1e-12


def test_compare_flat_vs_household_calibration_writes_artifacts(
    tmp_path,
    monkeypatch,
):
    cps = pd.DataFrame(
        {
            "household_id": [1, 1, 2],
            "tax_unit_id": [10, 10, 20],
            "person_seq": [1, 2, 1],
            "age": [40, 38, 55],
            "state_fips": [6, 6, 48],
            "weight": [100.0, 100.0, 200.0],
            "total_person_income": [50_000.0, 20_000.0, 30_000.0],
            "wage_salary_income": [50_000.0, 20_000.0, 30_000.0],
            "self_employment_income": [0.0, 0.0, 0.0],
        }
    )
    cps_path = tmp_path / "cps.parquet"
    output_dir = tmp_path / "comparison"
    cps.to_parquet(cps_path, index=False)

    targets = [
        TargetSpec(
            variable="tax_unit_count",
            value=300.0,
            target_type=TargetType.COUNT,
            constraints=[("is_tax_filer", "==", "1")],
            source=DataSource.IRS_SOI,
            period=2024,
            stratum_name="All filers",
        )
    ]

    def fake_compose(*args, **kwargs):
        return TargetCompositionResult(targets=targets, diagnostics=pd.DataFrame())

    monkeypatch.setattr(
        "micro.us.hierarchy_validation.compose_microplex_targets",
        fake_compose,
    )

    result = compare_flat_vs_household_calibration(
        year=2024,
        cps_path=cps_path,
        limit=3,
        output_dir=output_dir,
        min_target_obs=1,
        add_policyengine_tax=False,
        verbose=False,
    )

    assert result.flat_result.calibration_unit == "tax_unit"
    assert result.household_result.calibration_unit == "household"
    assert len(result.target_comparison) == 1
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "target_comparison.csv").exists()
    assert set(result.summary["group"]) >= {"flat", "household", "delta"}
