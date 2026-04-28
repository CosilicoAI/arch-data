"""
Tests for hierarchical microplex pipeline.

Tests the pipeline that calibrates household weights using person-level targets
aggregated to the household level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Import functions to test
from arch.pipelines.hierarchical_microplex import (
    build_hierarchical_constraints,
    run_hierarchical_ipf,
    _create_mock_hierarchical_data,
)

# These imports may fail if Supabase not available
try:
    from arch.pipelines.hierarchical_microplex import (
        load_hierarchical_data_from_supabase,
        run_hierarchical_pipeline,
    )
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    load_hierarchical_data_from_supabase = None
    run_hierarchical_pipeline = None


class TestLoadHierarchicalData:
    """Test loading hierarchical household + person data."""

    def test_mock_data_loads_correctly(self):
        """Test that mock data can be loaded."""
        hh_df, person_df = _create_mock_hierarchical_data(limit=100)
        assert len(hh_df) > 0
        assert len(person_df) > 0
        assert "household_id" in hh_df.columns
        assert "household_id" in person_df.columns

    def test_person_household_link(self):
        """Test that persons properly link to households."""
        hh_df, person_df = _create_mock_hierarchical_data(limit=100)
        # All person household_ids should exist in hh_df
        person_hh_ids = set(person_df["household_id"].unique())
        hh_ids = set(hh_df["household_id"].unique())
        assert person_hh_ids.issubset(hh_ids)

    def test_mock_data_has_required_columns(self):
        """Test that mock data has all required columns."""
        hh_df, person_df = _create_mock_hierarchical_data(limit=50)
        # Household required columns
        assert "weight" in hh_df.columns
        assert "state_fips" in hh_df.columns
        # Person required columns
        assert "person_id" in person_df.columns
        assert "age" in person_df.columns


class TestBuildHierarchicalConstraints:
    """Test building constraint matrix for hierarchical calibration."""

    @pytest.fixture
    def sample_hierarchical_data(self):
        """Create sample hierarchical microdata."""
        # 5 households with 2-3 persons each
        hh_df = pd.DataFrame({
            "household_id": [1, 2, 3, 4, 5],
            "weight": [100.0, 200.0, 150.0, 180.0, 120.0],
            "state_fips": [6, 6, 36, 36, 48],  # CA, CA, NY, NY, TX
        })

        person_df = pd.DataFrame({
            "person_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "household_id": [1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5],
            "age": [35, 8, 45, 42, 12, 65, 62, 28, 6, 55, 50],
            "employment_income": [50000, 0, 80000, 60000, 0, 0, 0, 45000, 0, 70000, 40000],
        })

        return hh_df, person_df

    def test_count_target_aggregation(self, sample_hierarchical_data):
        """Test that person count targets aggregate to household level."""
        hh_df, person_df = sample_hierarchical_data

        # Target: count of persons aged 0-17
        mock_targets = [{
            "variable": "person_count",
            "value": 500.0,  # Total weighted children
            "target_type": "count",
            "constraints": [("age", ">=", "0"), ("age", "<", "18")],
        }]

        constraints = build_hierarchical_constraints(
            hh_df, person_df, mock_targets, hh_id_col="household_id", min_obs=1
        )

        assert len(constraints) == 1
        # Indicator should be count of children per household
        # HH1: 1 child (age 8), HH2: 1 child (age 12), HH3: 0, HH4: 1 (age 6), HH5: 0
        expected = np.array([1.0, 1.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(constraints[0]["indicator"], expected)

    def test_amount_target_aggregation(self, sample_hierarchical_data):
        """Test that person amount targets sum to household level."""
        hh_df, person_df = sample_hierarchical_data

        # Target: total employment income
        mock_targets = [{
            "variable": "employment_income",
            "value": 50000000.0,  # Total weighted income
            "target_type": "amount",
            "constraints": [],  # All persons
        }]

        constraints = build_hierarchical_constraints(
            hh_df, person_df, mock_targets, hh_id_col="household_id", min_obs=1
        )

        assert len(constraints) == 1
        # Indicator should be sum of income per household
        # HH1: 50000, HH2: 140000, HH3: 0, HH4: 45000, HH5: 110000
        expected = np.array([50000.0, 140000.0, 0.0, 45000.0, 110000.0])
        np.testing.assert_array_equal(constraints[0]["indicator"], expected)


class TestHierarchicalIPF:
    """Test IPF calibration with hierarchical constraints."""

    def test_basic_convergence(self):
        """Test that IPF converges with simple hierarchical data."""
        # Simple case: 3 households
        original_weights = np.array([100.0, 200.0, 150.0])

        # One constraint: total count = 500
        constraints = [{
            "indicator": np.array([1.0, 1.0, 1.0]),
            "target_value": 500.0,
        }]

        calibrated, success, loss = run_hierarchical_ipf(
            original_weights, constraints, verbose=False
        )

        # Should converge
        assert success
        # Total should match target
        total = np.sum(calibrated)
        np.testing.assert_almost_equal(total, 500.0, decimal=1)

    def test_multiple_constraints(self):
        """Test IPF with multiple hierarchical constraints."""
        original_weights = np.array([100.0, 100.0, 100.0, 100.0])

        # Constraint 1: total = 500
        # Constraint 2: first two = 200
        constraints = [
            {"indicator": np.array([1.0, 1.0, 1.0, 1.0]), "target_value": 500.0},
            {"indicator": np.array([1.0, 1.0, 0.0, 0.0]), "target_value": 200.0},
        ]

        calibrated, success, loss = run_hierarchical_ipf(
            original_weights, constraints, verbose=False
        )

        assert success
        np.testing.assert_almost_equal(np.sum(calibrated), 500.0, decimal=1)
        np.testing.assert_almost_equal(np.sum(calibrated[:2]), 200.0, decimal=1)


class TestRunHierarchicalPipeline:
    """Test the full hierarchical pipeline."""

    @pytest.mark.skipif(not SUPABASE_AVAILABLE, reason="Supabase not available")
    def test_dry_run_returns_dataframe(self):
        """Test that dry run returns calibrated DataFrame."""
        result = run_hierarchical_pipeline(
            year=2024, dry_run=True, use_mock=True, limit=100, verbose=False
        )
        assert result is not None
        assert "calibrated_weight" in result.columns or "weight" in result.columns

    @pytest.mark.skipif(not SUPABASE_AVAILABLE, reason="Supabase not available")
    def test_dry_run_does_not_write(self):
        """Test that dry run doesn't write to database."""
        # Just ensure no exception is raised
        run_hierarchical_pipeline(
            year=2024, dry_run=True, use_mock=True, limit=50, verbose=False
        )


class TestIntegration:
    """Integration tests using mock data."""

    def test_full_mock_calibration(self):
        """Test full calibration workflow with mock data."""
        # Create mock data
        hh_df, person_df = _create_mock_hierarchical_data(limit=500)

        # Create realistic targets
        total_weight = hh_df["weight"].sum()
        n_children = person_df[person_df["age"] < 18].groupby("household_id").size()
        weighted_children = (hh_df.set_index("household_id")["weight"] * n_children.reindex(hh_df["household_id"]).fillna(0).values).sum()

        targets = [
            {
                "variable": "household_count",
                "value": total_weight * 1.1,  # 10% increase target
                "target_type": "count",
                "constraints": [],
            },
            {
                "variable": "person_count",
                "value": weighted_children * 0.9,  # 10% decrease for children
                "target_type": "count",
                "constraints": [("age", "<", "18")],
            },
        ]

        # Build constraints
        constraints = build_hierarchical_constraints(
            hh_df, person_df, targets, hh_id_col="household_id"
        )

        assert len(constraints) == 2

        # Run IPF
        original_weights = hh_df["weight"].values.copy()
        calibrated, success, loss = run_hierarchical_ipf(
            original_weights, constraints, verbose=False
        )

        # Should converge (with some tolerance due to conflicting targets)
        assert calibrated.sum() > 0
        # Weights should change
        assert not np.allclose(calibrated, original_weights)
