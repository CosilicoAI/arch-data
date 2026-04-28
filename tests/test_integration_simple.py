"""Simple integration tests for calibration pipeline.

Tests minimal end-to-end workflow with small synthetic data.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sqlmodel import Session

from calibration import (
    get_targets,
    build_constraint_matrix,
    EntropyCalibrator,
    TargetSpec,
)
from db.schema import (
    DataSource,
    Jurisdiction,
    Stratum,
    StratumConstraint,
    Target,
    TargetType,
    init_db,
)


class TestSimpleEndToEnd:
    """Simple integration tests with small data."""

    def test_load_and_calibrate_synthetic_data(self):
        """Test loading synthetic data and calibrating with simple constraints."""

        # Step 1: Create simple microdata with uniform weights
        np.random.seed(42)
        n = 100
        microdata = pd.DataFrame({
            "weight": np.ones(n) * 1000.0,  # Uniform weights
            "is_tax_filer": np.random.choice([0, 1], n, p=[0.2, 0.8]),
            "age": np.random.randint(18, 80, n),
        })

        # Step 2: Create simple target manually (no database needed)
        current_filers = (microdata["weight"] * microdata["is_tax_filer"]).sum()
        target_filers = current_filers * 1.1  # Want 10% more filers (modest change)

        targets = [
            TargetSpec(
                variable="tax_unit_count",
                value=target_filers,
                target_type=TargetType.COUNT,
                constraints=[("is_tax_filer", "==", "1")],
                source=DataSource.IRS_SOI,
                period=2021,
            )
        ]

        # Step 3: Build constraints
        constraints = build_constraint_matrix(microdata, targets)
        assert len(constraints) == 1

        # Step 4: Calibrate
        calibrator = EntropyCalibrator(bounds=(0.5, 3.0), max_iterations=100)
        original_weights = microdata["weight"].values
        calibrated_weights = calibrator.calibrate(original_weights, constraints)

        # Step 5: Verify
        assert len(calibrated_weights) == len(microdata)
        assert all(calibrated_weights > 0)

        # Check constraint is satisfied
        actual_filers = (calibrated_weights * microdata["is_tax_filer"]).sum()
        error = abs(actual_filers - target_filers) / target_filers
        assert error < 0.01, f"Error: {error:.2%}"

    def test_multiple_constraints_small_data(self):
        """Test calibration with multiple constraints on small dataset."""

        # Create small microdata
        np.random.seed(123)
        n = 50
        microdata = pd.DataFrame({
            "weight": np.ones(n) * 100.0,
            "is_tax_filer": np.random.choice([0, 1], n, p=[0.3, 0.7]),
            "age": np.random.randint(18, 80, n),
        })

        # Create two overlapping constraints
        targets = [
            # All filers
            TargetSpec(
                variable="tax_unit_count",
                value=4000.0,
                target_type=TargetType.COUNT,
                constraints=[("is_tax_filer", "==", "1")],
                source=DataSource.IRS_SOI,
                period=2021,
            ),
            # Young filers (age < 40)
            TargetSpec(
                variable="tax_unit_count",
                value=1500.0,
                target_type=TargetType.COUNT,
                constraints=[
                    ("is_tax_filer", "==", "1"),
                    ("age", "<", "40"),
                ],
                source=DataSource.IRS_SOI,
                period=2021,
            ),
        ]

        # Build and calibrate
        constraints = build_constraint_matrix(microdata, targets)
        calibrator = EntropyCalibrator(bounds=(0.5, 5.0), max_iterations=100)
        calibrated_weights = calibrator.calibrate(microdata["weight"].values, constraints)

        # Verify both constraints
        for i, constraint in enumerate(constraints):
            actual = (calibrated_weights * constraint.indicator).sum()
            error = abs(actual - constraint.target_value) / constraint.target_value
            assert error < 0.02, (
                f"Constraint {i} failed: target={constraint.target_value}, "
                f"actual={actual}, error={error:.2%}"
            )

    def test_amount_constraint_small_data(self):
        """Test AMOUNT type constraint with small dataset."""

        # Create microdata with income
        np.random.seed(789)
        n = 50
        microdata = pd.DataFrame({
            "weight": np.ones(n) * 100.0,
            "is_tax_filer": np.ones(n, dtype=int),
            "income": np.random.lognormal(10, 1, n),
        })

        # Target: adjust total income based on current value
        current_income = (microdata["weight"] * microdata["income"]).sum()
        target_income = current_income * 1.5  # Want 50% more total income

        targets = [
            TargetSpec(
                variable="income",
                value=target_income,
                target_type=TargetType.AMOUNT,
                constraints=[("is_tax_filer", "==", "1")],
                source=DataSource.IRS_SOI,
                period=2021,
            )
        ]

        # Calibrate (use wider bounds to accommodate 50% change)
        constraints = build_constraint_matrix(microdata, targets)
        calibrator = EntropyCalibrator(bounds=(0.1, 10.0), max_iterations=200)
        calibrated_weights = calibrator.calibrate(microdata["weight"].values, constraints)

        # Verify
        actual_income = (calibrated_weights * microdata["income"]).sum()
        error = abs(actual_income - target_income) / target_income
        assert error < 0.01, f"Income error: {error:.2%}"

    def test_database_integration(self):
        """Test querying targets from database and calibrating."""

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            engine = init_db(db_path)

            # Create simple target in database
            with Session(engine) as session:
                # Create stratum
                constraints_def = [("is_tax_filer", "==", "1")]
                stratum = Stratum(
                    name="all_filers",
                    jurisdiction=Jurisdiction.US,
                    description="All filers",
                    definition_hash=Stratum.compute_hash(constraints_def, Jurisdiction.US),
                )
                session.add(stratum)
                session.flush()

                # Add constraint
                constraint = StratumConstraint(
                    stratum_id=stratum.id,
                    variable="is_tax_filer",
                    operator="==",
                    value="1",
                )
                session.add(constraint)

                # Add target
                target = Target(
                    stratum_id=stratum.id,
                    variable="tax_unit_count",
                    value=4000.0,
                    target_type=TargetType.COUNT,
                    source=DataSource.IRS_SOI,
                    period=2021,
                )
                session.add(target)
                session.commit()

            # Query targets
            targets = get_targets(db_path=db_path, jurisdiction="us", year=2021)
            assert len(targets) == 1
            assert targets[0].value == 4000.0

            # Create microdata and calibrate
            np.random.seed(42)
            n = 50
            microdata = pd.DataFrame({
                "weight": np.ones(n) * 100.0,
                "is_tax_filer": np.random.choice([0, 1], n, p=[0.3, 0.7]),
            })

            constraints = build_constraint_matrix(microdata, targets)
            calibrator = EntropyCalibrator(bounds=(0.5, 5.0))
            calibrated_weights = calibrator.calibrate(microdata["weight"].values, constraints)

            # Verify constraint satisfied
            actual = (calibrated_weights * microdata["is_tax_filer"]).sum()
            error = abs(actual - 4000.0) / 4000.0
            assert error < 0.01, f"Error: {error:.2%}"
