"""Tests for calibration pipeline core."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlmodel import Session

from db.schema import (
    DataSource,
    TargetType,
    init_db,
)
from db.etl_soi import load_soi_targets


@pytest.fixture
def temp_db():
    """Create a temporary database with SOI targets for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_calibration.db"
        engine = init_db(db_path)
        with Session(engine) as session:
            load_soi_targets(session, years=[2021])
        yield db_path


@pytest.fixture
def sample_microdata():
    """Create sample microdata for testing constraint building."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "weight": np.random.uniform(100, 200, n),
        "is_tax_filer": np.random.choice([0, 1], n, p=[0.2, 0.8]),
        "adjusted_gross_income": np.random.lognormal(10, 1.5, n),
        "age": np.random.randint(18, 85, n),
        "state_fips": np.random.choice(["06", "36", "48"], n),  # CA, NY, TX
        "filing_status": np.random.choice(["1", "2", "3", "4"], n),  # single, mfj, mfs, hoh
    })


class TestTargetSpec:
    """Tests for TargetSpec dataclass."""

    def test_target_spec_creation(self):
        """TargetSpec should store target with constraints."""
        from calibration.targets import TargetSpec

        spec = TargetSpec(
            variable="tax_unit_count",
            value=153_774_296,
            target_type=TargetType.COUNT,
            constraints=[("is_tax_filer", "==", "1")],
            source=DataSource.IRS_SOI,
            period=2021,
        )

        assert spec.variable == "tax_unit_count"
        assert spec.value == 153_774_296
        assert spec.target_type == TargetType.COUNT
        assert len(spec.constraints) == 1

    def test_target_spec_with_multiple_constraints(self):
        """TargetSpec should support multiple constraints."""
        from calibration.targets import TargetSpec

        spec = TargetSpec(
            variable="tax_unit_count",
            value=18_892_456,
            target_type=TargetType.COUNT,
            constraints=[
                ("adjusted_gross_income", ">=", "50000"),
                ("adjusted_gross_income", "<", "75000"),
            ],
            source=DataSource.IRS_SOI,
            period=2021,
        )

        assert len(spec.constraints) == 2


class TestGetTargets:
    """Tests for get_targets() function."""

    def test_get_targets_returns_list(self, temp_db):
        """get_targets should return a list of TargetSpec objects."""
        from calibration.targets import get_targets, TargetSpec

        targets = get_targets(db_path=temp_db, jurisdiction="us", year=2021)

        assert isinstance(targets, list)
        assert len(targets) > 0
        assert all(isinstance(t, TargetSpec) for t in targets)

    def test_get_targets_filters_by_year(self, temp_db):
        """get_targets should filter by year."""
        from calibration.targets import get_targets

        targets = get_targets(db_path=temp_db, jurisdiction="us", year=2021)

        assert all(t.period == 2021 for t in targets)

    def test_get_targets_filters_by_source(self, temp_db):
        """get_targets should filter by source."""
        from calibration.targets import get_targets

        targets = get_targets(
            db_path=temp_db,
            jurisdiction="us",
            year=2021,
            sources=["irs-soi"],
        )

        assert all(t.source == DataSource.IRS_SOI for t in targets)

    def test_get_targets_filters_by_variable(self, temp_db):
        """get_targets should filter by variable name."""
        from calibration.targets import get_targets

        targets = get_targets(
            db_path=temp_db,
            jurisdiction="us",
            year=2021,
            variables=["tax_unit_count"],
        )

        assert all(t.variable == "tax_unit_count" for t in targets)

    def test_get_targets_includes_constraints(self, temp_db):
        """get_targets should include stratum constraints in TargetSpec."""
        from calibration.targets import get_targets

        targets = get_targets(db_path=temp_db, jurisdiction="us", year=2021)

        # Find a bracket target (should have AGI constraints)
        bracket_targets = [
            t for t in targets
            if any("adjusted_gross_income" in c[0] for c in t.constraints)
        ]

        assert len(bracket_targets) > 0

    def test_get_targets_empty_for_nonexistent_year(self, temp_db):
        """get_targets should return empty list for year with no data."""
        from calibration.targets import get_targets

        targets = get_targets(db_path=temp_db, jurisdiction="us", year=1900)

        assert targets == []


class TestBuildConstraintMatrix:
    """Tests for build_constraint_matrix() function."""

    def test_build_constraint_matrix_returns_constraints(self, sample_microdata):
        """build_constraint_matrix should return Constraint objects."""
        from calibration.targets import TargetSpec
        from calibration.constraints import build_constraint_matrix, Constraint

        targets = [
            TargetSpec(
                variable="tax_unit_count",
                value=800,
                target_type=TargetType.COUNT,
                constraints=[("is_tax_filer", "==", "1")],
                source=DataSource.IRS_SOI,
                period=2021,
            ),
        ]

        constraints = build_constraint_matrix(sample_microdata, targets)

        assert isinstance(constraints, list)
        assert len(constraints) == 1
        assert isinstance(constraints[0], Constraint)

    def test_constraint_has_indicator_vector(self, sample_microdata):
        """Constraint should have indicator vector matching microdata rows."""
        from calibration.targets import TargetSpec
        from calibration.constraints import build_constraint_matrix

        targets = [
            TargetSpec(
                variable="tax_unit_count",
                value=800,
                target_type=TargetType.COUNT,
                constraints=[("is_tax_filer", "==", "1")],
                source=DataSource.IRS_SOI,
                period=2021,
            ),
        ]

        constraints = build_constraint_matrix(sample_microdata, targets)

        assert len(constraints[0].indicator) == len(sample_microdata)
        assert constraints[0].indicator.dtype == np.float64

    def test_constraint_indicator_matches_stratum(self, sample_microdata):
        """Indicator vector should match stratum definition."""
        from calibration.targets import TargetSpec
        from calibration.constraints import build_constraint_matrix

        targets = [
            TargetSpec(
                variable="tax_unit_count",
                value=800,
                target_type=TargetType.COUNT,
                constraints=[("is_tax_filer", "==", "1")],
                source=DataSource.IRS_SOI,
                period=2021,
            ),
        ]

        constraints = build_constraint_matrix(sample_microdata, targets)

        # Indicator should be 1 where is_tax_filer == 1
        expected = (sample_microdata["is_tax_filer"] == 1).astype(float).values
        np.testing.assert_array_equal(constraints[0].indicator, expected)

    def test_constraint_with_range(self, sample_microdata):
        """Constraint should handle range conditions (>=, <)."""
        from calibration.targets import TargetSpec
        from calibration.constraints import build_constraint_matrix

        targets = [
            TargetSpec(
                variable="tax_unit_count",
                value=100,
                target_type=TargetType.COUNT,
                constraints=[
                    ("adjusted_gross_income", ">=", "50000"),
                    ("adjusted_gross_income", "<", "75000"),
                ],
                source=DataSource.IRS_SOI,
                period=2021,
            ),
        ]

        constraints = build_constraint_matrix(sample_microdata, targets)

        # Indicator should be 1 where AGI in range
        agi = sample_microdata["adjusted_gross_income"]
        expected = ((agi >= 50000) & (agi < 75000)).astype(float).values
        np.testing.assert_array_equal(constraints[0].indicator, expected)

    def test_constraint_stores_target_value(self, sample_microdata):
        """Constraint should store the target value."""
        from calibration.targets import TargetSpec
        from calibration.constraints import build_constraint_matrix

        targets = [
            TargetSpec(
                variable="tax_unit_count",
                value=800,
                target_type=TargetType.COUNT,
                constraints=[("is_tax_filer", "==", "1")],
                source=DataSource.IRS_SOI,
                period=2021,
            ),
        ]

        constraints = build_constraint_matrix(sample_microdata, targets)

        assert constraints[0].target_value == 800

    def test_constraint_amount_type_uses_variable(self, sample_microdata):
        """For AMOUNT type, indicator should be variable * mask."""
        from calibration.targets import TargetSpec
        from calibration.constraints import build_constraint_matrix

        targets = [
            TargetSpec(
                variable="adjusted_gross_income",
                value=1_000_000_000,
                target_type=TargetType.AMOUNT,
                constraints=[("is_tax_filer", "==", "1")],
                source=DataSource.IRS_SOI,
                period=2021,
            ),
        ]

        constraints = build_constraint_matrix(sample_microdata, targets)

        # For AMOUNT, indicator should be AGI * is_tax_filer
        mask = sample_microdata["is_tax_filer"] == 1
        expected = (sample_microdata["adjusted_gross_income"] * mask).values
        np.testing.assert_array_almost_equal(constraints[0].indicator, expected)

    def test_constraint_with_tolerance(self, sample_microdata):
        """Constraints should store tolerance."""
        from calibration.targets import TargetSpec
        from calibration.constraints import build_constraint_matrix

        targets = [
            TargetSpec(
                variable="tax_unit_count",
                value=800,
                target_type=TargetType.COUNT,
                constraints=[("is_tax_filer", "==", "1")],
                source=DataSource.IRS_SOI,
                period=2021,
            ),
        ]

        constraints = build_constraint_matrix(
            sample_microdata, targets, tolerance=0.05
        )

        assert constraints[0].tolerance == 0.05

    def test_multiple_constraints(self, sample_microdata):
        """Should build constraints for multiple targets."""
        from calibration.targets import TargetSpec
        from calibration.constraints import build_constraint_matrix

        targets = [
            TargetSpec(
                variable="tax_unit_count",
                value=800,
                target_type=TargetType.COUNT,
                constraints=[("is_tax_filer", "==", "1")],
                source=DataSource.IRS_SOI,
                period=2021,
            ),
            TargetSpec(
                variable="adjusted_gross_income",
                value=1_000_000_000,
                target_type=TargetType.AMOUNT,
                constraints=[("is_tax_filer", "==", "1")],
                source=DataSource.IRS_SOI,
                period=2021,
            ),
        ]

        constraints = build_constraint_matrix(sample_microdata, targets)

        assert len(constraints) == 2


class TestEntropyCalibrator:
    """Tests for EntropyCalibrator."""

    def test_entropy_calibrator_exists(self):
        """EntropyCalibrator class should exist."""
        from calibration.methods.entropy import EntropyCalibrator

        calibrator = EntropyCalibrator()
        assert calibrator is not None

    def test_entropy_calibrator_has_calibrate_method(self):
        """EntropyCalibrator should have calibrate method."""
        from calibration.methods.entropy import EntropyCalibrator

        calibrator = EntropyCalibrator()
        assert hasattr(calibrator, "calibrate")

    def test_calibrated_weights_sum_correctly(self):
        """Calibrated weights should sum to expected total."""
        from calibration.methods.entropy import EntropyCalibrator
        from calibration.constraints import Constraint

        # Simple case: 100 records, want total weight 1000
        n = 100
        original_weights = np.ones(n) * 10.0  # Sum = 1000

        # Create constraint: sum of all weights should be 1200
        indicator = np.ones(n)
        constraints = [
            Constraint(
                indicator=indicator,
                target_value=1200.0,
                variable="total",
                target_type=TargetType.COUNT,
            )
        ]

        calibrator = EntropyCalibrator()
        calibrated = calibrator.calibrate(original_weights, constraints)

        # Sum should match target
        np.testing.assert_allclose(calibrated.sum(), 1200.0, rtol=1e-4)

    def test_constraints_satisfied_within_tolerance(self):
        """Calibration should satisfy constraints within tolerance."""
        from calibration.methods.entropy import EntropyCalibrator
        from calibration.constraints import Constraint

        np.random.seed(42)
        n = 200
        original_weights = np.random.uniform(5, 15, n)

        # Two strata: first half and second half
        indicator1 = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
        indicator2 = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])

        constraints = [
            Constraint(
                indicator=indicator1,
                target_value=600.0,
                variable="stratum1",
                target_type=TargetType.COUNT,
                tolerance=0.01,
            ),
            Constraint(
                indicator=indicator2,
                target_value=800.0,
                variable="stratum2",
                target_type=TargetType.COUNT,
                tolerance=0.01,
            ),
        ]

        calibrator = EntropyCalibrator()
        calibrated = calibrator.calibrate(original_weights, constraints)

        # Check constraints are satisfied
        actual1 = (calibrated * indicator1).sum()
        actual2 = (calibrated * indicator2).sum()

        np.testing.assert_allclose(actual1, 600.0, rtol=0.01)
        np.testing.assert_allclose(actual2, 800.0, rtol=0.01)

    def test_weights_respect_bounds(self):
        """Calibrated weights should respect min/max bounds."""
        from calibration.methods.entropy import EntropyCalibrator
        from calibration.constraints import Constraint

        n = 100
        original_weights = np.ones(n) * 10.0

        indicator = np.ones(n)
        constraints = [
            Constraint(
                indicator=indicator,
                target_value=1500.0,
                variable="total",
                target_type=TargetType.COUNT,
            )
        ]

        # Set bounds: weights can be 0.5x to 2x original
        calibrator = EntropyCalibrator(bounds=(0.5, 2.0))
        calibrated = calibrator.calibrate(original_weights, constraints)

        # Check all weights are within bounds
        min_weight = original_weights * 0.5
        max_weight = original_weights * 2.0
        assert np.all(calibrated >= min_weight - 1e-6)
        assert np.all(calibrated <= max_weight + 1e-6)

    def test_minimal_deviation_from_original(self):
        """Calibration should minimize KL divergence."""
        from calibration.methods.entropy import EntropyCalibrator
        from calibration.constraints import Constraint

        np.random.seed(42)
        n = 100
        original_weights = np.random.uniform(8, 12, n)

        # Constraint that's already nearly satisfied
        indicator = np.ones(n)
        current_sum = original_weights.sum()
        target = current_sum * 1.05  # Just 5% adjustment needed

        constraints = [
            Constraint(
                indicator=indicator,
                target_value=target,
                variable="total",
                target_type=TargetType.COUNT,
            )
        ]

        calibrator = EntropyCalibrator()
        calibrated = calibrator.calibrate(original_weights, constraints)

        # Compute KL divergence: sum(w * log(w/w0))
        kl_div = np.sum(calibrated * np.log(calibrated / original_weights))

        # For a single uniform constraint, uniform scaling is optimal
        # and should give the same KL divergence
        uniform_scale = target / current_sum
        uniform_weights = original_weights * uniform_scale
        kl_uniform = np.sum(uniform_weights * np.log(uniform_weights / original_weights))

        # Our solution should be very close to uniform scaling
        np.testing.assert_allclose(kl_div, kl_uniform, rtol=0.01)

    def test_amount_constraint(self):
        """Should handle AMOUNT type constraints correctly."""
        from calibration.methods.entropy import EntropyCalibrator
        from calibration.constraints import Constraint

        np.random.seed(42)
        n = 100
        original_weights = np.ones(n) * 10.0
        income = np.random.lognormal(10, 1, n)

        # Constraint: total weighted income should be 5M
        indicator = income
        constraints = [
            Constraint(
                indicator=indicator,
                target_value=5_000_000.0,
                variable="income",
                target_type=TargetType.AMOUNT,
            )
        ]

        calibrator = EntropyCalibrator()
        calibrated = calibrator.calibrate(original_weights, constraints)

        # Check constraint is satisfied
        actual = (calibrated * income).sum()
        np.testing.assert_allclose(actual, 5_000_000.0, rtol=0.01)

    def test_multiple_amount_constraints(self):
        """Should handle multiple AMOUNT constraints simultaneously."""
        from calibration.methods.entropy import EntropyCalibrator
        from calibration.constraints import Constraint

        np.random.seed(42)
        n = 100
        original_weights = np.ones(n) * 10.0
        income = np.random.lognormal(10, 1, n)
        taxes = income * 0.2

        constraints = [
            Constraint(
                indicator=income,
                target_value=5_000_000.0,
                variable="income",
                target_type=TargetType.AMOUNT,
            ),
            Constraint(
                indicator=taxes,
                target_value=1_000_000.0,
                variable="taxes",
                target_type=TargetType.AMOUNT,
            ),
        ]

        calibrator = EntropyCalibrator()
        calibrated = calibrator.calibrate(original_weights, constraints)

        # Both constraints should be satisfied
        actual_income = (calibrated * income).sum()
        actual_taxes = (calibrated * taxes).sum()

        np.testing.assert_allclose(actual_income, 5_000_000.0, rtol=0.01)
        np.testing.assert_allclose(actual_taxes, 1_000_000.0, rtol=0.01)

    def test_infeasible_constraints_raises_error(self):
        """Should raise error if constraints are infeasible."""
        from calibration.methods.entropy import EntropyCalibrator
        from calibration.constraints import Constraint

        n = 10
        original_weights = np.ones(n) * 10.0

        # Infeasible: want sum = 500, but max possible is 10 * 10 * 10 = 1000
        # with bounds (0.1, 10) and tight second constraint
        indicator1 = np.ones(n)
        indicator2 = np.ones(n)

        constraints = [
            Constraint(
                indicator=indicator1,
                target_value=50.0,  # Need average weight of 5
                variable="total",
                target_type=TargetType.COUNT,
            ),
            Constraint(
                indicator=indicator2,
                target_value=150.0,  # Need average weight of 15
                variable="total2",
                target_type=TargetType.COUNT,
            ),
        ]

        calibrator = EntropyCalibrator(bounds=(0.5, 1.5))

        with pytest.raises((ValueError, RuntimeError)):
            calibrator.calibrate(original_weights, constraints)
