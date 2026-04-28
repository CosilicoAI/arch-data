"""Tests for SNAP ETL."""

import tempfile
from pathlib import Path

import pytest
from sqlmodel import Session, select

from db.schema import (
    DataSource,
    Stratum,
    Target,
    TargetType,
    init_db,
)
from db.etl_snap import load_snap_targets, SNAP_DATA


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_snap.db"
        engine = init_db(db_path)
        yield engine


class TestSnapETL:
    """Tests for SNAP ETL loader."""

    def test_load_snap_creates_national_stratum(self, temp_db):
        """Loading SNAP data should create a national stratum."""
        with Session(temp_db) as session:
            load_snap_targets(session, years=[2023])

            national = session.exec(
                select(Stratum).where(Stratum.name == "US SNAP Recipients")
            ).first()

            assert national is not None
            assert national.stratum_group_id == "snap_national"

    def test_load_snap_creates_national_targets(self, temp_db):
        """Loading SNAP data should create national-level targets."""
        with Session(temp_db) as session:
            load_snap_targets(session, years=[2023])

            national = session.exec(
                select(Stratum).where(Stratum.name == "US SNAP Recipients")
            ).first()

            # Check household count
            hh_target = session.exec(
                select(Target)
                .where(Target.stratum_id == national.id)
                .where(Target.variable == "snap_household_count")
                .where(Target.period == 2023)
            ).first()

            assert hh_target is not None
            expected_hh = SNAP_DATA[2023]["national"]["households"] * 1000
            assert hh_target.value == expected_hh
            assert hh_target.target_type == TargetType.COUNT
            assert hh_target.source == DataSource.USDA_SNAP
            assert "convert_units" in hh_target.notes

    def test_load_snap_creates_participant_targets(self, temp_db):
        """Loading SNAP should create participant count targets."""
        with Session(temp_db) as session:
            load_snap_targets(session, years=[2023])

            national = session.exec(
                select(Stratum).where(Stratum.name == "US SNAP Recipients")
            ).first()

            participant_target = session.exec(
                select(Target)
                .where(Target.stratum_id == national.id)
                .where(Target.variable == "snap_participant_count")
                .where(Target.period == 2023)
            ).first()

            assert participant_target is not None
            expected = SNAP_DATA[2023]["national"]["participants"] * 1000
            assert participant_target.value == expected

    def test_load_snap_creates_benefit_targets(self, temp_db):
        """Loading SNAP should create benefit amount targets."""
        with Session(temp_db) as session:
            load_snap_targets(session, years=[2023])

            national = session.exec(
                select(Stratum).where(Stratum.name == "US SNAP Recipients")
            ).first()

            benefit_target = session.exec(
                select(Target)
                .where(Target.stratum_id == national.id)
                .where(Target.variable == "snap_benefits")
                .where(Target.period == 2023)
            ).first()

            assert benefit_target is not None
            expected = SNAP_DATA[2023]["national"]["benefits"] * 1_000_000
            assert benefit_target.value == expected
            assert benefit_target.target_type == TargetType.AMOUNT
            assert "convert_units" in benefit_target.notes

    def test_load_snap_creates_state_strata(self, temp_db):
        """Loading SNAP should create state-level strata."""
        with Session(temp_db) as session:
            load_snap_targets(session, years=[2023])

            state_strata = session.exec(
                select(Stratum).where(Stratum.stratum_group_id == "snap_states")
            ).all()

            # Should have strata for states in the data
            expected_states = len(SNAP_DATA[2023].get("states", {}))
            assert len(state_strata) == expected_states

    def test_load_snap_state_targets_correct(self, temp_db):
        """State-level SNAP targets should have correct values."""
        with Session(temp_db) as session:
            load_snap_targets(session, years=[2023])

            ca_stratum = session.exec(
                select(Stratum).where(Stratum.name == "CA SNAP Recipients")
            ).first()

            assert ca_stratum is not None

            ca_hh = session.exec(
                select(Target)
                .where(Target.stratum_id == ca_stratum.id)
                .where(Target.variable == "snap_household_count")
            ).first()

            expected_ca_hh = SNAP_DATA[2023]["states"]["CA"]["households"] * 1000
            assert ca_hh.value == expected_ca_hh

    def test_load_multiple_years(self, temp_db):
        """Loading multiple years should create targets for each."""
        with Session(temp_db) as session:
            load_snap_targets(session, years=[2021, 2022, 2023])

            national = session.exec(
                select(Stratum).where(Stratum.name == "US SNAP Recipients")
            ).first()

            targets = session.exec(
                select(Target)
                .where(Target.stratum_id == national.id)
                .where(Target.variable == "snap_household_count")
            ).all()

            years = {t.period for t in targets}
            assert years == {2021, 2022, 2023}

    def test_load_snap_idempotent(self, temp_db):
        """Loading SNAP twice should not duplicate data."""
        with Session(temp_db) as session:
            load_snap_targets(session, years=[2023])
            load_snap_targets(session, years=[2023])

            national_strata = session.exec(
                select(Stratum).where(Stratum.name == "US SNAP Recipients")
            ).all()

            # Should only have one national stratum
            assert len(national_strata) == 1

    def test_state_stratum_has_parent(self, temp_db):
        """State strata should have national stratum as parent."""
        with Session(temp_db) as session:
            load_snap_targets(session, years=[2023])

            national = session.exec(
                select(Stratum).where(Stratum.name == "US SNAP Recipients")
            ).first()

            ca = session.exec(
                select(Stratum).where(Stratum.name == "CA SNAP Recipients")
            ).first()

            assert ca.parent_id == national.id
