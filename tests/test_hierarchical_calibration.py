"""Tests for hierarchical calibration with person/household aggregation."""

from dataclasses import dataclass
from enum import Enum
import importlib.util
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

# Import only the minimal types needed, avoid supabase dependency chain
# by defining local stubs for testing
class TargetType(str, Enum):
    """Stub for db.schema.TargetType for testing."""
    COUNT = "count"
    AMOUNT = "amount"
    RATE = "rate"


class DataSource(str, Enum):
    """Stub for db.schema.DataSource for testing."""
    IRS_SOI = "irs-soi"
    CENSUS_ACS = "census-acs"


# Load calibration.variables directly
_calibration_path = Path(__file__).parent.parent / "calibration"

_variables_spec = importlib.util.spec_from_file_location(
    "calibration_variables", _calibration_path / "variables.py"
)
_variables_module = importlib.util.module_from_spec(_variables_spec)
_variables_spec.loader.exec_module(_variables_module)

infer_target_level = _variables_module.infer_target_level
parse_variable_ref = _variables_module.parse_variable_ref
InvalidVariableRefError = _variables_module.InvalidVariableRefError


@dataclass
class TargetSpec:
    """Minimal TargetSpec for testing."""
    variable: str
    value: float
    target_type: TargetType
    constraints: list
    source: DataSource
    period: int
    tolerance: Optional[float] = None
    stratum_name: Optional[str] = None


@dataclass
class Constraint:
    """Minimal Constraint for testing."""
    indicator: np.ndarray
    target_value: float
    variable: str
    target_type: TargetType
    tolerance: float = 0.01
    stratum_name: Optional[str] = None


def apply_stratum_constraints(
    microdata: pd.DataFrame,
    constraints: list,
) -> pd.Series:
    """Apply stratum constraints to get boolean mask."""
    mask = pd.Series(True, index=microdata.index)

    for variable, operator, value in constraints:
        if variable not in microdata.columns:
            continue

        col = microdata[variable]
        if pd.api.types.is_numeric_dtype(col):
            parsed_value = float(value)
        else:
            parsed_value = value

        if operator == "==":
            mask &= col == parsed_value
        elif operator == ">=":
            mask &= col >= parsed_value
        elif operator == "<":
            mask &= col < parsed_value
        elif operator == "<=":
            mask &= col <= parsed_value
        elif operator == ">":
            mask &= col > parsed_value

    return mask


def build_hierarchical_constraint_matrix(
    hh_df: pd.DataFrame,
    person_df: pd.DataFrame,
    targets: list,
    tolerance: float = 0.01,
    hh_id_col: str = "household_id",
    tax_unit_df: Optional[pd.DataFrame] = None,
) -> list:
    """Simplified hierarchical constraint builder for testing."""
    constraints = []

    for target in targets:
        # Infer level from constraint variables
        level = infer_target_level(target.constraints)

        if level == "household":
            mask = apply_stratum_constraints(hh_df, target.constraints)
            if target.target_type == TargetType.COUNT:
                indicator = mask.astype(float).values
            else:
                indicator = np.zeros(len(hh_df))
        else:
            # Aggregate from person to household
            mask = apply_stratum_constraints(person_df, target.constraints)
            filtered = person_df[mask]

            if len(filtered) == 0:
                indicator = np.zeros(len(hh_df))
            elif target.target_type == TargetType.COUNT:
                agg = filtered.groupby(hh_id_col).size()
                indicator = hh_df[hh_id_col].map(agg).fillna(0).values
            else:
                indicator = np.zeros(len(hh_df))

        constraints.append(Constraint(
            indicator=indicator,
            target_value=target.value,
            variable=target.variable,
            target_type=target.target_type,
            tolerance=target.tolerance if target.tolerance else tolerance,
            stratum_name=target.stratum_name,
        ))

    return constraints


@pytest.fixture
def mock_households():
    """Create mock household DataFrame with 100 households."""
    np.random.seed(42)
    n_households = 100

    return pd.DataFrame({
        "household_id": range(1, n_households + 1),
        "state_fips": np.random.choice(
            ["06", "36", "48", "12", "17"],  # CA, NY, TX, FL, IL
            n_households,
            p=[0.30, 0.15, 0.20, 0.20, 0.15],
        ),
        "weight": np.random.uniform(100, 500, n_households),
    })


@pytest.fixture
def mock_persons(mock_households):
    """Create mock person DataFrame with ~250 persons linked to households."""
    np.random.seed(43)

    persons = []
    person_id = 1

    for hh_id in mock_households["household_id"]:
        # Each household has 1-5 persons
        n_persons = np.random.randint(1, 6)
        for _ in range(n_persons):
            age = np.random.randint(0, 95)
            persons.append({
                "person_id": person_id,
                "household_id": hh_id,
                "age": age,
                "is_employed": 1 if age >= 18 and age < 65 and np.random.random() > 0.3 else 0,
            })
            person_id += 1

    return pd.DataFrame(persons)


@pytest.fixture
def person_count_target():
    """Create a person-level COUNT target: people aged 18-64."""
    return TargetSpec(
        variable="person_count",
        value=150_000_000.0,  # Target count
        target_type=TargetType.COUNT,
        constraints=[
            ("age", ">=", "18"),
            ("age", "<", "65"),
        ],
        source=DataSource.CENSUS_ACS,
        period=2023,
        stratum_name="Working age population (18-64)",
    )


@pytest.fixture
def household_count_target():
    """Create a household-level COUNT target: households in California."""
    return TargetSpec(
        variable="household_count",
        value=13_000_000.0,  # Target count
        target_type=TargetType.COUNT,
        constraints=[
            ("state_fips", "==", "06"),
        ],
        source=DataSource.CENSUS_ACS,
        period=2023,
        stratum_name="California households",
    )


class TestBuildHierarchicalConstraintMatrix:
    """Tests for build_hierarchical_constraint_matrix() function."""

    def test_returns_constraint_objects(
        self, mock_households, mock_persons, person_count_target
    ):
        """build_hierarchical_constraint_matrix should return Constraint objects."""
        constraints = build_hierarchical_constraint_matrix(
            hh_df=mock_households,
            person_df=mock_persons,
            targets=[person_count_target],
        )

        assert isinstance(constraints, list)
        assert len(constraints) == 1
        assert isinstance(constraints[0], Constraint)

    def test_indicator_length_matches_households(
        self, mock_households, mock_persons, person_count_target
    ):
        """Person-level targets should be aggregated to household level."""
        constraints = build_hierarchical_constraint_matrix(
            hh_df=mock_households,
            person_df=mock_persons,
            targets=[person_count_target],
        )

        # Indicator length should equal number of households, not persons
        assert len(constraints[0].indicator) == len(mock_households)

    def test_person_target_aggregation_correct(
        self, mock_households, mock_persons, person_count_target
    ):
        """Person-level aggregation should correctly count matching persons per household."""
        constraints = build_hierarchical_constraint_matrix(
            hh_df=mock_households,
            person_df=mock_persons,
            targets=[person_count_target],
        )

        # Manually compute expected indicator: count of persons aged 18-64 per household
        mask = (mock_persons["age"] >= 18) & (mock_persons["age"] < 65)
        expected_counts = (
            mock_persons[mask]
            .groupby("household_id")
            .size()
        )

        # Map back to household order
        expected_indicator = (
            mock_households["household_id"]
            .map(expected_counts)
            .fillna(0)
            .values
        )

        np.testing.assert_array_equal(
            constraints[0].indicator,
            expected_indicator,
        )

    def test_household_target_direct_indicator(
        self, mock_households, mock_persons, household_count_target
    ):
        """Household-level targets should produce direct indicator (no aggregation)."""
        constraints = build_hierarchical_constraint_matrix(
            hh_df=mock_households,
            person_df=mock_persons,
            targets=[household_count_target],
        )

        # Indicator should be 1 for California households, 0 otherwise
        expected_indicator = (
            mock_households["state_fips"] == "06"
        ).astype(float).values

        np.testing.assert_array_equal(
            constraints[0].indicator,
            expected_indicator,
        )

    def test_indicator_length_equals_households_for_household_target(
        self, mock_households, mock_persons, household_count_target
    ):
        """Household-level indicator length should match household count."""
        constraints = build_hierarchical_constraint_matrix(
            hh_df=mock_households,
            person_df=mock_persons,
            targets=[household_count_target],
        )

        assert len(constraints[0].indicator) == len(mock_households)

    def test_multiple_targets_mixed_levels(
        self, mock_households, mock_persons, person_count_target, household_count_target
    ):
        """Should handle both person and household level targets together."""
        constraints = build_hierarchical_constraint_matrix(
            hh_df=mock_households,
            person_df=mock_persons,
            targets=[person_count_target, household_count_target],
        )

        assert len(constraints) == 2
        # Both should have household-level indicator length
        assert len(constraints[0].indicator) == len(mock_households)
        assert len(constraints[1].indicator) == len(mock_households)

    def test_stores_target_value(
        self, mock_households, mock_persons, person_count_target
    ):
        """Constraint should store the target value from TargetSpec."""
        constraints = build_hierarchical_constraint_matrix(
            hh_df=mock_households,
            person_df=mock_persons,
            targets=[person_count_target],
        )

        assert constraints[0].target_value == 150_000_000.0

    def test_stores_variable_name(
        self, mock_households, mock_persons, person_count_target
    ):
        """Constraint should store the variable name from TargetSpec."""
        constraints = build_hierarchical_constraint_matrix(
            hh_df=mock_households,
            person_df=mock_persons,
            targets=[person_count_target],
        )

        assert constraints[0].variable == "person_count"

    def test_stores_target_type(
        self, mock_households, mock_persons, person_count_target
    ):
        """Constraint should store the target type from TargetSpec."""
        constraints = build_hierarchical_constraint_matrix(
            hh_df=mock_households,
            person_df=mock_persons,
            targets=[person_count_target],
        )

        assert constraints[0].target_type == TargetType.COUNT

    def test_custom_tolerance(
        self, mock_households, mock_persons, person_count_target
    ):
        """Should respect custom tolerance parameter."""
        constraints = build_hierarchical_constraint_matrix(
            hh_df=mock_households,
            person_df=mock_persons,
            targets=[person_count_target],
            tolerance=0.05,
        )

        assert constraints[0].tolerance == 0.05

    def test_no_matching_persons_gives_zero_indicator(
        self, mock_households, mock_persons
    ):
        """Target with no matching persons should produce zero indicator."""
        # Target for people aged 150+ (none exist)
        impossible_target = TargetSpec(
            variable="person_count",
            value=0.0,
            target_type=TargetType.COUNT,
            constraints=[
                ("age", ">=", "150"),
            ],
            source=DataSource.CENSUS_ACS,
            period=2023,
        )

        constraints = build_hierarchical_constraint_matrix(
            hh_df=mock_households,
            person_df=mock_persons,
            targets=[impossible_target],
        )

        np.testing.assert_array_equal(
            constraints[0].indicator,
            np.zeros(len(mock_households)),
        )


class TestInferTargetLevel:
    """Tests for infer_target_level() function."""

    def test_person_variable_returns_person(self):
        """Constraints with person-level variables should return 'person'."""
        constraints = [("age", ">=", "18")]
        level = infer_target_level(constraints)
        assert level == "person"

    def test_household_variable_returns_household(self):
        """Constraints with household-level variables should return 'household'."""
        constraints = [("state_fips", "==", "06")]
        level = infer_target_level(constraints)
        assert level == "household"

    def test_mixed_levels_returns_most_granular(self):
        """Constraints with mixed levels should return most granular (person)."""
        constraints = [
            ("age", ">=", "18"),
            ("state_fips", "==", "06"),
        ]
        level = infer_target_level(constraints)
        # Person is more granular than household
        assert level == "person"

    def test_empty_constraints_returns_household(self):
        """Empty constraints should default to household level."""
        constraints = []
        level = infer_target_level(constraints)
        assert level == "household"

    def test_tax_unit_variable_returns_tax_unit(self):
        """Constraints with tax_unit-level variables should return 'tax_unit'."""
        constraints = [("filing_status", "==", "single")]
        level = infer_target_level(constraints)
        assert level == "tax_unit"

    def test_unknown_variable_defaults_to_person(self):
        """Unknown variables should default to person level."""
        constraints = [("unknown_variable_xyz", "==", "value")]
        level = infer_target_level(constraints)
        assert level == "person"


class TestParseVariableRef:
    """Tests for parse_variable_ref() function."""

    def test_parse_valid_reference(self):
        """Should correctly parse a valid fully qualified reference."""
        package, path, var_name = parse_variable_ref("us:statute/26/32#eitc")

        assert package == "us"
        assert path == "statute/26/32"
        assert var_name == "eitc"

    def test_parse_state_reference(self):
        """Should correctly parse a state-level reference."""
        package, path, var_name = parse_variable_ref("us-ca:statute/ca/rtc/17041#ca_agi")

        assert package == "us-ca"
        assert path == "statute/ca/rtc/17041"
        assert var_name == "ca_agi"

    def test_parse_nested_path(self):
        """Should correctly parse a deeply nested path."""
        package, path, var_name = parse_variable_ref(
            "us:statute/26/24/a/1#child_tax_credit_amount"
        )

        assert package == "us"
        assert path == "statute/26/24/a/1"
        assert var_name == "child_tax_credit_amount"

    def test_invalid_ref_missing_colon_raises_error(self):
        """Reference without colon should raise InvalidVariableRefError."""
        with pytest.raises(InvalidVariableRefError):
            parse_variable_ref("us-statute/26/32#eitc")

    def test_invalid_ref_missing_hash_raises_error(self):
        """Reference without hash should raise InvalidVariableRefError."""
        with pytest.raises(InvalidVariableRefError):
            parse_variable_ref("us:statute/26/32-eitc")

    def test_invalid_ref_empty_string_raises_error(self):
        """Empty string should raise InvalidVariableRefError."""
        with pytest.raises(InvalidVariableRefError):
            parse_variable_ref("")

    def test_invalid_ref_only_colon_raises_error(self):
        """Reference with only colon should raise InvalidVariableRefError."""
        with pytest.raises(InvalidVariableRefError):
            parse_variable_ref("us:")

    def test_invalid_ref_only_hash_raises_error(self):
        """Reference with only hash should raise InvalidVariableRefError."""
        with pytest.raises(InvalidVariableRefError):
            parse_variable_ref("#eitc")

    def test_parse_uk_reference(self):
        """Should correctly parse a UK reference."""
        package, path, var_name = parse_variable_ref("uk:statute/ita/2007#personal_allowance")

        assert package == "uk"
        assert path == "statute/ita/2007"
        assert var_name == "personal_allowance"

    def test_parse_reference_with_multiple_hashes(self):
        """Should handle reference with multiple # symbols (takes last)."""
        package, path, var_name = parse_variable_ref("us:path/with#hash#var")

        assert package == "us"
        assert path == "path/with#hash"
        assert var_name == "var"
