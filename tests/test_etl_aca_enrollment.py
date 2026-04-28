"""Tests for ACA Marketplace enrollment ETL."""

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
from db.etl_aca_enrollment import (
    load_aca_enrollment_targets,
    ACA_ENROLLMENT_DATA,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_aca.db"
        engine = init_db(db_path)
        yield engine


class TestAcaEnrollmentETL:
    """Tests for ACA Marketplace enrollment ETL loader."""

    def test_load_aca_creates_national_stratum(self, temp_db):
        """Loading ACA data should create a national stratum."""
        with Session(temp_db) as session:
            load_aca_enrollment_targets(session, years=[2024])

            national = session.exec(
                select(Stratum).where(Stratum.name == "US ACA Marketplace Enrollees")
            ).first()

            assert national is not None
            assert national.stratum_group_id == "aca_national"

    def test_load_aca_creates_enrollment_targets(self, temp_db):
        """Loading ACA data should create enrollment count targets."""
        with Session(temp_db) as session:
            load_aca_enrollment_targets(session, years=[2024])

            national = session.exec(
                select(Stratum).where(Stratum.name == "US ACA Marketplace Enrollees")
            ).first()

            enrollment_target = session.exec(
                select(Target)
                .where(Target.stratum_id == national.id)
                .where(Target.variable == "aca_marketplace_enrollment")
                .where(Target.period == 2024)
            ).first()

            assert enrollment_target is not None
            expected = ACA_ENROLLMENT_DATA[2024]["national"]["total_enrollment"]
            assert enrollment_target.value == expected
            assert enrollment_target.target_type == TargetType.COUNT
            assert enrollment_target.source == DataSource.CMS_ACA

    def test_load_aca_creates_aptc_stratum(self, temp_db):
        """Loading ACA data should create APTC recipient stratum."""
        with Session(temp_db) as session:
            load_aca_enrollment_targets(session, years=[2024])

            aptc_stratum = session.exec(
                select(Stratum).where(Stratum.name == "US ACA APTC Recipients")
            ).first()

            assert aptc_stratum is not None
            assert aptc_stratum.stratum_group_id == "aca_subsidies"

    def test_load_aca_creates_aptc_targets(self, temp_db):
        """Loading ACA should create APTC recipient and amount targets."""
        with Session(temp_db) as session:
            load_aca_enrollment_targets(session, years=[2024])

            aptc_stratum = session.exec(
                select(Stratum).where(Stratum.name == "US ACA APTC Recipients")
            ).first()

            # Check recipient count
            aptc_count = session.exec(
                select(Target)
                .where(Target.stratum_id == aptc_stratum.id)
                .where(Target.variable == "aca_aptc_recipients")
            ).first()

            assert aptc_count is not None
            expected = ACA_ENROLLMENT_DATA[2024]["national"]["aptc_recipients"]
            assert aptc_count.value == expected

            # Check average APTC amount
            aptc_amount = session.exec(
                select(Target)
                .where(Target.stratum_id == aptc_stratum.id)
                .where(Target.variable == "aca_avg_monthly_aptc")
            ).first()

            assert aptc_amount is not None
            assert aptc_amount.target_type == TargetType.AMOUNT

    def test_load_aca_creates_csr_stratum(self, temp_db):
        """Loading ACA data should create CSR recipient stratum for years with data."""
        with Session(temp_db) as session:
            load_aca_enrollment_targets(session, years=[2024])

            csr_stratum = session.exec(
                select(Stratum).where(Stratum.name == "US ACA CSR Recipients")
            ).first()

            assert csr_stratum is not None
            assert csr_stratum.stratum_group_id == "aca_subsidies"

    def test_load_aca_creates_metal_level_strata(self, temp_db):
        """Loading ACA data should create metal level strata."""
        with Session(temp_db) as session:
            load_aca_enrollment_targets(session, years=[2024])

            metal_levels = ["Bronze", "Silver", "Gold", "Platinum", "Catastrophic"]
            for level in metal_levels:
                stratum = session.exec(
                    select(Stratum).where(
                        Stratum.name == f"US ACA {level} Plan Enrollees"
                    )
                ).first()

                assert stratum is not None, f"Missing stratum for {level}"
                assert stratum.stratum_group_id == "aca_metal_levels"

    def test_load_aca_metal_level_enrollment_correct(self, temp_db):
        """Metal level enrollment should be calculated from percentages."""
        with Session(temp_db) as session:
            load_aca_enrollment_targets(session, years=[2024])

            silver_stratum = session.exec(
                select(Stratum).where(Stratum.name == "US ACA Silver Plan Enrollees")
            ).first()

            enrollment = session.exec(
                select(Target)
                .where(Target.stratum_id == silver_stratum.id)
                .where(Target.variable == "aca_marketplace_enrollment")
            ).first()

            national_data = ACA_ENROLLMENT_DATA[2024]["national"]
            expected = int(national_data["total_enrollment"] * national_data["silver_pct"])
            assert enrollment.value == expected
            assert "apply_share" in enrollment.notes

    def test_load_aca_creates_state_strata(self, temp_db):
        """Loading ACA should create state-level strata."""
        with Session(temp_db) as session:
            load_aca_enrollment_targets(session, years=[2024])

            state_strata = session.exec(
                select(Stratum).where(Stratum.stratum_group_id == "aca_states")
            ).all()

            # Should have strata for states in the data
            expected_states = len(ACA_ENROLLMENT_DATA[2024].get("states", {}))
            assert len(state_strata) == expected_states

    def test_load_aca_state_targets_correct(self, temp_db):
        """State-level ACA targets should have correct values."""
        with Session(temp_db) as session:
            load_aca_enrollment_targets(session, years=[2024])

            fl_stratum = session.exec(
                select(Stratum).where(Stratum.name == "FL ACA Marketplace Enrollees")
            ).first()

            assert fl_stratum is not None

            fl_enrollment = session.exec(
                select(Target)
                .where(Target.stratum_id == fl_stratum.id)
                .where(Target.variable == "aca_marketplace_enrollment")
            ).first()

            expected = ACA_ENROLLMENT_DATA[2024]["states"]["FL"]["enrollment"]
            assert fl_enrollment.value == expected

    def test_load_aca_state_metal_levels(self, temp_db):
        """States with metal level data should have metal level strata."""
        with Session(temp_db) as session:
            load_aca_enrollment_targets(session, years=[2024])

            # Florida has metal level data in 2024
            fl_silver = session.exec(
                select(Stratum).where(Stratum.name == "FL ACA Silver Plan Enrollees")
            ).first()

            assert fl_silver is not None
            assert fl_silver.stratum_group_id == "aca_state_metal_levels"

    def test_load_multiple_years(self, temp_db):
        """Loading multiple years should create targets for each."""
        with Session(temp_db) as session:
            load_aca_enrollment_targets(session, years=[2023, 2024, 2025])

            national = session.exec(
                select(Stratum).where(Stratum.name == "US ACA Marketplace Enrollees")
            ).first()

            targets = session.exec(
                select(Target)
                .where(Target.stratum_id == national.id)
                .where(Target.variable == "aca_marketplace_enrollment")
            ).all()

            years = {t.period for t in targets}
            assert years == {2023, 2024, 2025}

    def test_load_aca_idempotent(self, temp_db):
        """Loading ACA twice should not duplicate data."""
        with Session(temp_db) as session:
            load_aca_enrollment_targets(session, years=[2024])
            load_aca_enrollment_targets(session, years=[2024])

            national_strata = session.exec(
                select(Stratum).where(Stratum.name == "US ACA Marketplace Enrollees")
            ).all()

            # Should only have one national stratum
            assert len(national_strata) == 1

    def test_state_stratum_has_parent(self, temp_db):
        """State strata should have national stratum as parent."""
        with Session(temp_db) as session:
            load_aca_enrollment_targets(session, years=[2024])

            national = session.exec(
                select(Stratum).where(Stratum.name == "US ACA Marketplace Enrollees")
            ).first()

            ca = session.exec(
                select(Stratum).where(Stratum.name == "CA ACA Marketplace Enrollees")
            ).first()

            assert ca.parent_id == national.id

    def test_aptc_stratum_has_parent(self, temp_db):
        """APTC stratum should have national stratum as parent."""
        with Session(temp_db) as session:
            load_aca_enrollment_targets(session, years=[2024])

            national = session.exec(
                select(Stratum).where(Stratum.name == "US ACA Marketplace Enrollees")
            ).first()

            aptc = session.exec(
                select(Stratum).where(Stratum.name == "US ACA APTC Recipients")
            ).first()

            assert aptc.parent_id == national.id

    def test_premium_targets_created(self, temp_db):
        """Premium-related targets should be created."""
        with Session(temp_db) as session:
            load_aca_enrollment_targets(session, years=[2024])

            national = session.exec(
                select(Stratum).where(Stratum.name == "US ACA Marketplace Enrollees")
            ).first()

            # Check gross premium
            gross_premium = session.exec(
                select(Target)
                .where(Target.stratum_id == national.id)
                .where(Target.variable == "aca_avg_monthly_premium_gross")
            ).first()

            assert gross_premium is not None
            assert gross_premium.target_type == TargetType.AMOUNT

            # Check net premium
            net_premium = session.exec(
                select(Target)
                .where(Target.stratum_id == national.id)
                .where(Target.variable == "aca_avg_monthly_premium_net")
            ).first()

            assert net_premium is not None
            assert net_premium.target_type == TargetType.AMOUNT

    def test_metal_level_share_rate_target(self, temp_db):
        """Metal level share should be stored as a rate target."""
        with Session(temp_db) as session:
            load_aca_enrollment_targets(session, years=[2024])

            silver = session.exec(
                select(Stratum).where(Stratum.name == "US ACA Silver Plan Enrollees")
            ).first()

            share = session.exec(
                select(Target)
                .where(Target.stratum_id == silver.id)
                .where(Target.variable == "aca_metal_level_share")
            ).first()

            assert share is not None
            assert share.target_type == TargetType.RATE
            expected = ACA_ENROLLMENT_DATA[2024]["national"]["silver_pct"]
            assert share.value == expected
