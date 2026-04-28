"""Tests for CPS raw microdata ETL to Supabase."""

import pandas as pd
from unittest.mock import MagicMock, patch


class TestCPSRawETL:
    """Test CPS raw data loading to Supabase."""

    def test_prepare_person_records(self):
        """Test preparing person records for Supabase insert."""
        from db.etl_cps_raw import prepare_person_records

        # Sample raw CPS data
        df = pd.DataFrame({
            "PH_SEQ": [1, 1, 2],
            "PPPOS": [1, 2, 1],
            "A_AGE": [35, 10, 42],
            "A_SEX": [1, 2, 2],
            "PRDTRACE": [1, 1, 2],
            "PEHSPNON": [2, 2, 1],
            "GESTFIPS": [6, 6, 36],
            "PTOTVAL": [50000.0, 0.0, 75000.0],
            "PEARNVAL": [48000.0, 0.0, 70000.0],
            "WSAL_VAL": [45000.0, 0.0, 65000.0],
            "MARSUPWT": [1500.0, 1500.0, 2000.0],
        })

        records = prepare_person_records(df)

        assert len(records) == 3
        assert records[0]["ph_seq"] == 1
        assert records[0]["a_age"] == 35
        assert records[0]["gestfips"] == 6
        assert records[0]["marsupwt"] == 1500.0
        # Should have raw_data with all columns
        assert "raw_data" in records[0]
        assert records[0]["raw_data"]["A_AGE"] == 35

    def test_prepare_household_records(self):
        """Test preparing household records for Supabase insert."""
        from db.etl_cps_raw import prepare_household_records

        df = pd.DataFrame({
            "H_SEQ": [1, 2],
            "H_NUMPER": [3, 2],
            "HH5TO18": [1, 0],
            "HUNDER18": [1, 0],
            "GESTFIPS": [6, 36],
            "HTOTVAL": [125000.0, 80000.0],
            "HSUP_WGT": [1500.0, 2000.0],
        })

        records = prepare_household_records(df)

        assert len(records) == 2
        assert records[0]["h_seq"] == 1
        assert records[0]["h_numper"] == 3
        assert records[1]["gestfips"] == 36

    def test_get_cps_table_names(self):
        """Test table name generation for CPS ASEC."""
        from db.etl_cps_raw import get_cps_table_names

        names = get_cps_table_names(2024)

        assert names["person"] == "us_census_cps_asec_2024_person"
        assert names["household"] == "us_census_cps_asec_2024_household"
        assert names["family"] == "us_census_cps_asec_2024_family"

    @patch("db.etl_cps_raw.get_supabase_client")
    @patch("db.etl_cps_raw.pd.read_parquet")
    @patch("db.etl_cps_raw.get_raw_cache_dir")
    def test_load_cps_to_supabase_dry_run(self, mock_cache_dir, mock_read, mock_client):
        """Test dry run mode returns stats without inserting."""
        from db.etl_cps_raw import load_cps_to_supabase

        # Mock cache directory with existing files
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_cache_dir.return_value = mock_path
        mock_cache_dir.return_value.__truediv__ = lambda self, x: mock_path

        # Mock parquet reading
        mock_read.return_value = pd.DataFrame({
            "PH_SEQ": [1, 2],
            "A_AGE": [35, 42],
            "MARSUPWT": [1500, 2000],
        })

        result = load_cps_to_supabase(2024, dry_run=True)

        assert "person_count" in result
        assert "household_count" in result
        assert "family_count" in result
        assert result["dry_run"] is True
        # In dry run, client should not be called
        mock_client.assert_not_called()
