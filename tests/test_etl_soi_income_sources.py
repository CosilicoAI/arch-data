"""Tests for IRS SOI income by source ETL."""

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
from db.etl_soi_income_sources import (
    load_soi_income_sources_targets,
    SOI_NATIONAL_INCOME_SOURCES_DATA,
    SOI_STATE_INCOME_SOURCES_DATA,
    STATE_FIPS,
    INCOME_SOURCES,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_soi_income_sources.db"
        engine = init_db(db_path)
        yield engine


class TestSoiIncomeSourcesETL:
    """Tests for IRS SOI income sources ETL loader."""

    def test_load_soi_income_sources_creates_national_stratum(self, temp_db):
        """Loading income source data should create/reference a national stratum."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            national = session.exec(
                select(Stratum).where(Stratum.name == "US All Filers")
            ).first()

            assert national is not None
            assert national.stratum_group_id == "national"

    def test_load_national_income_sources_targets(self, temp_db):
        """Loading should create national-level income source targets."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            national = session.exec(
                select(Stratum).where(Stratum.name == "US All Filers")
            ).first()

            # Check taxable interest target
            taxable_interest_target = session.exec(
                select(Target)
                .where(Target.stratum_id == national.id)
                .where(Target.variable == "taxable_interest_returns")
                .where(Target.period == 2021)
            ).first()

            assert taxable_interest_target is not None
            expected = SOI_NATIONAL_INCOME_SOURCES_DATA[2021]["taxable_interest_returns"]
            assert taxable_interest_target.value == expected
            assert taxable_interest_target.target_type == TargetType.COUNT
            assert taxable_interest_target.source == DataSource.IRS_SOI
            assert taxable_interest_target.source_table == "Publication 1304, Table 1.4"

    def test_load_soi_income_sources_creates_state_strata(self, temp_db):
        """Loading income source data should create state-level strata."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            state_strata = session.exec(
                select(Stratum).where(Stratum.stratum_group_id == "soi_states")
            ).all()

            # Should have strata for all 50 states + DC
            expected_states = len(STATE_FIPS)
            assert len(state_strata) == expected_states
            assert expected_states == 51  # 50 states + DC

    def test_load_taxable_interest_targets(self, temp_db):
        """Loading income sources should create taxable interest targets."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            ca_stratum = session.exec(
                select(Stratum).where(Stratum.name == "CA All Filers")
            ).first()

            assert ca_stratum is not None

            # Returns count target
            returns_target = session.exec(
                select(Target)
                .where(Target.stratum_id == ca_stratum.id)
                .where(Target.variable == "taxable_interest_returns")
                .where(Target.period == 2021)
            ).first()

            assert returns_target is not None
            expected = SOI_STATE_INCOME_SOURCES_DATA[2021]["CA"]["taxable_interest_returns"]
            assert returns_target.value == expected
            assert returns_target.target_type == TargetType.COUNT

            # Amount target
            amount_target = session.exec(
                select(Target)
                .where(Target.stratum_id == ca_stratum.id)
                .where(Target.variable == "taxable_interest_amount")
                .where(Target.period == 2021)
            ).first()

            assert amount_target is not None
            assert amount_target.target_type == TargetType.AMOUNT

    def test_load_tax_exempt_interest_targets(self, temp_db):
        """Loading income sources should create tax-exempt interest targets."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            fl_stratum = session.exec(
                select(Stratum).where(Stratum.name == "FL All Filers")
            ).first()

            target = session.exec(
                select(Target)
                .where(Target.stratum_id == fl_stratum.id)
                .where(Target.variable == "tax_exempt_interest_returns")
                .where(Target.period == 2021)
            ).first()

            assert target is not None
            expected = SOI_STATE_INCOME_SOURCES_DATA[2021]["FL"]["tax_exempt_interest_returns"]
            assert target.value == expected

    def test_load_ordinary_dividends_targets(self, temp_db):
        """Loading income sources should create ordinary dividends targets."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            tx_stratum = session.exec(
                select(Stratum).where(Stratum.name == "TX All Filers")
            ).first()

            target = session.exec(
                select(Target)
                .where(Target.stratum_id == tx_stratum.id)
                .where(Target.variable == "ordinary_dividends_amount")
                .where(Target.period == 2021)
            ).first()

            assert target is not None
            expected = SOI_STATE_INCOME_SOURCES_DATA[2021]["TX"]["ordinary_dividends_amount"]
            assert target.value == expected

    def test_load_qualified_dividends_targets(self, temp_db):
        """Loading income sources should create qualified dividends targets."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            ny_stratum = session.exec(
                select(Stratum).where(Stratum.name == "NY All Filers")
            ).first()

            target = session.exec(
                select(Target)
                .where(Target.stratum_id == ny_stratum.id)
                .where(Target.variable == "qualified_dividends_returns")
                .where(Target.period == 2021)
            ).first()

            assert target is not None
            expected = SOI_STATE_INCOME_SOURCES_DATA[2021]["NY"]["qualified_dividends_returns"]
            assert target.value == expected

    def test_load_short_term_capital_gains_targets(self, temp_db):
        """Loading income sources should create short-term capital gains targets."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            wa_stratum = session.exec(
                select(Stratum).where(Stratum.name == "WA All Filers")
            ).first()

            target = session.exec(
                select(Target)
                .where(Target.stratum_id == wa_stratum.id)
                .where(Target.variable == "short_term_capital_gains_amount")
                .where(Target.period == 2021)
            ).first()

            assert target is not None
            expected = SOI_STATE_INCOME_SOURCES_DATA[2021]["WA"]["short_term_capital_gains_amount"]
            assert target.value == expected

    def test_load_long_term_capital_gains_targets(self, temp_db):
        """Loading income sources should create long-term capital gains targets."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            nj_stratum = session.exec(
                select(Stratum).where(Stratum.name == "NJ All Filers")
            ).first()

            target = session.exec(
                select(Target)
                .where(Target.stratum_id == nj_stratum.id)
                .where(Target.variable == "long_term_capital_gains_returns")
                .where(Target.period == 2021)
            ).first()

            assert target is not None
            expected = SOI_STATE_INCOME_SOURCES_DATA[2021]["NJ"]["long_term_capital_gains_returns"]
            assert target.value == expected

    def test_load_state_local_refunds_targets(self, temp_db):
        """Loading income sources should create state/local refunds targets."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            ga_stratum = session.exec(
                select(Stratum).where(Stratum.name == "GA All Filers")
            ).first()

            target = session.exec(
                select(Target)
                .where(Target.stratum_id == ga_stratum.id)
                .where(Target.variable == "state_local_refunds_amount")
                .where(Target.period == 2021)
            ).first()

            assert target is not None
            expected = SOI_STATE_INCOME_SOURCES_DATA[2021]["GA"]["state_local_refunds_amount"]
            assert target.value == expected

    def test_load_alimony_received_targets(self, temp_db):
        """Loading income sources should create alimony received targets."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            ma_stratum = session.exec(
                select(Stratum).where(Stratum.name == "MA All Filers")
            ).first()

            target = session.exec(
                select(Target)
                .where(Target.stratum_id == ma_stratum.id)
                .where(Target.variable == "alimony_received_returns")
                .where(Target.period == 2021)
            ).first()

            assert target is not None
            expected = SOI_STATE_INCOME_SOURCES_DATA[2021]["MA"]["alimony_received_returns"]
            assert target.value == expected

    def test_load_schedule_c_income_targets(self, temp_db):
        """Loading income sources should create Schedule C business income targets."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            il_stratum = session.exec(
                select(Stratum).where(Stratum.name == "IL All Filers")
            ).first()

            target = session.exec(
                select(Target)
                .where(Target.stratum_id == il_stratum.id)
                .where(Target.variable == "schedule_c_income_amount")
                .where(Target.period == 2021)
            ).first()

            assert target is not None
            expected = SOI_STATE_INCOME_SOURCES_DATA[2021]["IL"]["schedule_c_income_amount"]
            assert target.value == expected

    def test_load_rental_royalty_income_targets(self, temp_db):
        """Loading income sources should create rental/royalty income targets."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            pa_stratum = session.exec(
                select(Stratum).where(Stratum.name == "PA All Filers")
            ).first()

            target = session.exec(
                select(Target)
                .where(Target.stratum_id == pa_stratum.id)
                .where(Target.variable == "rental_royalty_income_returns")
                .where(Target.period == 2021)
            ).first()

            assert target is not None
            expected = SOI_STATE_INCOME_SOURCES_DATA[2021]["PA"]["rental_royalty_income_returns"]
            assert target.value == expected

    def test_load_partnership_scorp_income_targets(self, temp_db):
        """Loading income sources should create partnership/S-corp income targets."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            oh_stratum = session.exec(
                select(Stratum).where(Stratum.name == "OH All Filers")
            ).first()

            target = session.exec(
                select(Target)
                .where(Target.stratum_id == oh_stratum.id)
                .where(Target.variable == "partnership_scorp_income_amount")
                .where(Target.period == 2021)
            ).first()

            assert target is not None
            expected = SOI_STATE_INCOME_SOURCES_DATA[2021]["OH"]["partnership_scorp_income_amount"]
            assert target.value == expected

    def test_load_soi_income_sources_stratum_has_state_constraint(self, temp_db):
        """State strata should have state_fips constraint."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            nc_stratum = session.exec(
                select(Stratum).where(Stratum.name == "NC All Filers")
            ).first()

            # Check constraints include state_fips
            state_constraint = None
            for constraint in nc_stratum.constraints:
                if constraint.variable == "state_fips":
                    state_constraint = constraint
                    break

            assert state_constraint is not None
            assert state_constraint.operator == "=="
            assert state_constraint.value == STATE_FIPS["NC"]

    def test_load_soi_income_sources_stratum_has_parent(self, temp_db):
        """State strata should have national stratum as parent."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            national = session.exec(
                select(Stratum).where(Stratum.name == "US All Filers")
            ).first()

            mi_stratum = session.exec(
                select(Stratum).where(Stratum.name == "MI All Filers")
            ).first()

            assert mi_stratum.parent_id == national.id

    def test_load_soi_income_sources_idempotent(self, temp_db):
        """Loading income sources twice should not duplicate strata."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])
            load_soi_income_sources_targets(session, years=[2021])

            ca_strata = session.exec(
                select(Stratum).where(Stratum.name == "CA All Filers")
            ).all()

            # Should only have one CA stratum
            assert len(ca_strata) == 1

    def test_all_states_loaded(self, temp_db):
        """All 50 states + DC should be loaded with income source targets."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            # 11 income sources x 2 (returns + amount) = 22 targets per state
            expected_targets_per_state = len(INCOME_SOURCES) * 2

            for state_abbrev in STATE_FIPS.keys():
                stratum = session.exec(
                    select(Stratum).where(Stratum.name == f"{state_abbrev} All Filers")
                ).first()

                assert stratum is not None, f"Missing stratum for {state_abbrev}"

                targets = session.exec(
                    select(Target).where(Target.stratum_id == stratum.id)
                ).all()

                assert len(targets) == expected_targets_per_state, (
                    f"Expected {expected_targets_per_state} targets for {state_abbrev}, "
                    f"got {len(targets)}"
                )

    def test_target_source_metadata(self, temp_db):
        """Targets should have correct source metadata."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            ca_stratum = session.exec(
                select(Stratum).where(Stratum.name == "CA All Filers")
            ).first()

            target = session.exec(
                select(Target)
                .where(Target.stratum_id == ca_stratum.id)
                .where(Target.variable == "taxable_interest_returns")
            ).first()

            assert target.source == DataSource.IRS_SOI
            assert target.source_table == "Individual Income Tax State Data"
            assert "individual-income-tax-statistics" in target.source_url

    def test_income_sources_list(self):
        """INCOME_SOURCES should include all expected income types."""
        expected_sources = [
            "taxable_interest",
            "tax_exempt_interest",
            "ordinary_dividends",
            "qualified_dividends",
            "short_term_capital_gains",
            "long_term_capital_gains",
            "state_local_refunds",
            "alimony_received",
            "schedule_c_income",
            "rental_royalty_income",
            "partnership_scorp_income",
        ]
        assert INCOME_SOURCES == expected_sources

    def test_national_totals_are_reasonable(self, temp_db):
        """National totals should match expected ranges."""
        with Session(temp_db) as session:
            load_soi_income_sources_targets(session, years=[2021])

            # Check national taxable interest returns - should be ~55M
            national_data = SOI_NATIONAL_INCOME_SOURCES_DATA[2021]
            assert 40_000_000 < national_data["taxable_interest_returns"] < 80_000_000

            # Check national long-term capital gains - should be ~$850B
            assert 500_000_000_000 < national_data["long_term_capital_gains_amount"] < 1_500_000_000_000

            # Alimony should be much lower than other income sources
            assert national_data["alimony_received_returns"] < national_data["taxable_interest_returns"]

    def test_qualified_dividends_less_than_ordinary(self, temp_db):
        """Qualified dividends should always be less than or equal to ordinary dividends."""
        for state_data in SOI_STATE_INCOME_SOURCES_DATA[2021].values():
            assert state_data["qualified_dividends_returns"] <= state_data["ordinary_dividends_returns"]
            assert state_data["qualified_dividends_amount"] <= state_data["ordinary_dividends_amount"]


class TestIncomeSourcesData:
    """Tests for the income sources data structure."""

    def test_all_states_have_data(self):
        """All 50 states + DC should have income sources data."""
        assert len(SOI_STATE_INCOME_SOURCES_DATA[2021]) == 51

        for state_abbrev in STATE_FIPS.keys():
            assert state_abbrev in SOI_STATE_INCOME_SOURCES_DATA[2021], (
                f"Missing data for {state_abbrev}"
            )

    def test_all_income_sources_present(self):
        """Each state should have all income source variables."""
        expected_vars = []
        for source in INCOME_SOURCES:
            expected_vars.append(f"{source}_returns")
            expected_vars.append(f"{source}_amount")

        for state_abbrev, state_data in SOI_STATE_INCOME_SOURCES_DATA[2021].items():
            for var in expected_vars:
                assert var in state_data, (
                    f"Missing {var} for {state_abbrev}"
                )
                assert isinstance(state_data[var], (int, float)), (
                    f"{var} for {state_abbrev} is not numeric"
                )

    def test_values_are_positive(self):
        """All values should be positive (or zero for some edge cases)."""
        for state_abbrev, state_data in SOI_STATE_INCOME_SOURCES_DATA[2021].items():
            for var, value in state_data.items():
                assert value >= 0, (
                    f"Negative value for {var} in {state_abbrev}: {value}"
                )

    def test_large_states_have_higher_values(self):
        """Larger states should have more income than smaller states."""
        # California (large) vs Wyoming (small)
        ca_data = SOI_STATE_INCOME_SOURCES_DATA[2021]["CA"]
        wy_data = SOI_STATE_INCOME_SOURCES_DATA[2021]["WY"]

        assert ca_data["taxable_interest_returns"] > wy_data["taxable_interest_returns"]
        assert ca_data["ordinary_dividends_amount"] > wy_data["ordinary_dividends_amount"]
        assert ca_data["long_term_capital_gains_amount"] > wy_data["long_term_capital_gains_amount"]
        assert ca_data["schedule_c_income_amount"] > wy_data["schedule_c_income_amount"]

        # Texas (large) vs Vermont (small)
        tx_data = SOI_STATE_INCOME_SOURCES_DATA[2021]["TX"]
        vt_data = SOI_STATE_INCOME_SOURCES_DATA[2021]["VT"]

        assert tx_data["taxable_interest_returns"] > vt_data["taxable_interest_returns"]
        assert tx_data["partnership_scorp_income_amount"] > vt_data["partnership_scorp_income_amount"]

    def test_national_data_structure(self):
        """National data should have all required fields."""
        national_data = SOI_NATIONAL_INCOME_SOURCES_DATA[2021]

        for source in INCOME_SOURCES:
            returns_var = f"{source}_returns"
            amount_var = f"{source}_amount"

            assert returns_var in national_data, f"Missing {returns_var} in national data"
            assert amount_var in national_data, f"Missing {amount_var} in national data"
            assert isinstance(national_data[returns_var], int)
            assert isinstance(national_data[amount_var], int)
