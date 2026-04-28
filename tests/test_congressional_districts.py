"""Tests for congressional district geographic level functionality."""

import tempfile
from pathlib import Path

import pytest
from sqlmodel import Session, select

from db.schema import (
    GeographicLevel,
    Stratum,
    Target,
    init_db,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_cd.db"
        engine = init_db(db_path)
        yield engine


class TestCongressionalDistrictGeographicLevel:
    """Tests for congressional district geographic level support."""

    def test_geographic_level_enum_exists(self):
        """GeographicLevel enum should include congressional_district."""
        assert GeographicLevel.CONGRESSIONAL_DISTRICT == "congressional_district"
        assert GeographicLevel.STATE == "state"
        assert GeographicLevel.NATIONAL == "national"

    def test_load_congressional_district_targets(self, temp_db):
        """Loading congressional district data should create proper targets."""
        from db.etl_census import load_congressional_district_targets

        district_data = {
            ("06", "01"): {"population": 750000, "households": 280000},
            ("06", "12"): {"population": 720000, "households": 265000},
            ("36", "15"): {"population": 710000, "households": 290000},
        }

        with Session(temp_db) as session:
            load_congressional_district_targets(session, 2023, district_data)

            # Should create 3 congressional district strata
            cd_strata = session.exec(
                select(Stratum).where(
                    Stratum.stratum_group_id == "congressional_districts"
                )
            ).all()

            assert len(cd_strata) == 3

    def test_congressional_district_target_has_correct_geographic_level(self, temp_db):
        """Congressional district targets should have correct geographic level."""
        from db.etl_census import load_congressional_district_targets

        district_data = {
            ("06", "01"): {"population": 750000},
        }

        with Session(temp_db) as session:
            load_congressional_district_targets(session, 2023, district_data)

            # Find the target
            target = session.exec(
                select(Target)
                .where(Target.variable == "population")
                .where(Target.period == 2023)
            ).first()

            assert target is not None
            assert target.geographic_level == GeographicLevel.CONGRESSIONAL_DISTRICT
            assert target.value == 750000

    def test_congressional_district_stratum_has_correct_constraints(self, temp_db):
        """Congressional district strata should have state and district constraints."""
        from db.etl_census import load_congressional_district_targets

        district_data = {
            ("06", "01"): {"population": 750000},
        }

        with Session(temp_db) as session:
            load_congressional_district_targets(session, 2023, district_data)

            # Find the stratum
            stratum = session.exec(
                select(Stratum).where(
                    Stratum.name == "Congressional District 06-01"
                )
            ).first()

            assert stratum is not None
            assert len(stratum.constraints) == 2

            # Check constraints
            constraint_vars = {c.variable for c in stratum.constraints}
            assert "state_fips" in constraint_vars
            assert "congressional_district" in constraint_vars

            # Check values
            state_constraint = next(
                c for c in stratum.constraints if c.variable == "state_fips"
            )
            district_constraint = next(
                c for c in stratum.constraints if c.variable == "congressional_district"
            )

            assert state_constraint.value == "06"
            assert state_constraint.operator == "=="
            assert district_constraint.value == "01"
            assert district_constraint.operator == "=="

    def test_congressional_district_with_households(self, temp_db):
        """Congressional district targets should include household data when available."""
        from db.etl_census import load_congressional_district_targets

        district_data = {
            ("06", "01"): {"population": 750000, "households": 280000},
        }

        with Session(temp_db) as session:
            load_congressional_district_targets(session, 2023, district_data)

            # Should have both population and household targets
            targets = session.exec(select(Target)).all()

            assert len(targets) == 2

            variables = {t.variable for t in targets}
            assert "population" in variables
            assert "household_count" in variables

            # Both should have congressional district geographic level
            for target in targets:
                assert target.geographic_level == GeographicLevel.CONGRESSIONAL_DISTRICT

    def test_multiple_districts_same_state(self, temp_db):
        """Should handle multiple congressional districts in the same state."""
        from db.etl_census import load_congressional_district_targets

        district_data = {
            ("06", "01"): {"population": 750000},
            ("06", "12"): {"population": 720000},
            ("06", "30"): {"population": 740000},
        }

        with Session(temp_db) as session:
            load_congressional_district_targets(session, 2023, district_data)

            # Should create 3 separate strata
            ca_districts = session.exec(
                select(Stratum).where(
                    Stratum.stratum_group_id == "congressional_districts"
                )
            ).all()

            assert len(ca_districts) == 3

            # All should have state_fips == "06"
            for stratum in ca_districts:
                state_constraint = next(
                    c for c in stratum.constraints if c.variable == "state_fips"
                )
                assert state_constraint.value == "06"

    def test_query_by_geographic_level(self, temp_db):
        """Should be able to query targets by geographic level."""
        from db.etl_census import load_census_targets, load_congressional_district_targets

        district_data = {
            ("06", "01"): {"population": 750000},
        }

        with Session(temp_db) as session:
            # Load both national and congressional district data
            load_census_targets(session, years=[2023])
            load_congressional_district_targets(session, 2023, district_data)

            # Query for congressional district targets only
            cd_targets = session.exec(
                select(Target).where(
                    Target.geographic_level == GeographicLevel.CONGRESSIONAL_DISTRICT
                )
            ).all()

            assert len(cd_targets) == 1
            assert cd_targets[0].value == 750000

            # Query for national targets
            national_targets = session.exec(
                select(Target).where(
                    Target.geographic_level == GeographicLevel.NATIONAL
                )
            ).all()

            assert len(national_targets) > 0

            # Query for state targets
            state_targets = session.exec(
                select(Target).where(
                    Target.geographic_level == GeographicLevel.STATE
                )
            ).all()

            assert len(state_targets) > 0
