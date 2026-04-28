"""Tests for CPS downloader."""

import io
import zipfile

import pandas as pd
import pytest

# Direct imports from download_cps to avoid calibration package import chain
from micro.us.census.download_cps import (
    CPS_URL_BY_YEAR,
    PERSON_COLUMNS,
    process_cps_data,
    download_cps_zip,
    extract_person_data,
)


class TestCPSDownloader:
    """Tests for CPS download functionality."""

    def test_cps_url_by_year_has_expected_years(self):
        """CPS URLs should be defined for recent years."""
        # Should have at least 2020-2024
        assert 2020 in CPS_URL_BY_YEAR
        assert 2021 in CPS_URL_BY_YEAR
        assert 2022 in CPS_URL_BY_YEAR
        assert 2023 in CPS_URL_BY_YEAR

    def test_cps_urls_are_census_gov(self):
        """All CPS URLs should be from census.gov."""
        for year, url in CPS_URL_BY_YEAR.items():
            assert "census.gov" in url, f"Year {year} URL not from census.gov"
            assert url.endswith(".zip"), f"Year {year} URL should be ZIP file"

    def test_person_columns_has_required_fields(self):
        """Person columns should include key demographic and income fields."""
        # Must have these for calibration
        required_targets = ["age", "weight", "state_fips"]
        for target in required_targets:
            assert target in PERSON_COLUMNS.values(), f"Missing {target}"

    def test_process_cps_data_creates_required_columns(self):
        """Processing should create all required columns."""
        # Create mock raw data
        raw_df = pd.DataFrame({
            "household_id": [1, 1, 2],
            "person_seq": [1, 2, 1],
            "age": [35, 10, 65],
            "march_supplement_weight": [100, 100, 150],
            "class_of_worker": [1, 0, 0],
            "wage_salary_income": [50000, 0, 20000],
            "own_children_under_18": [1, 0, 0],
            "state_fips": [6, 6, 36],
        })

        result = process_cps_data(raw_df)

        # Check required columns exist
        assert "weight" in result.columns
        assert "age" in result.columns
        assert "income" in result.columns
        assert "employment_status" in result.columns
        assert "has_children" in result.columns
        assert "state_fips" in result.columns

    def test_process_cps_data_scales_weights(self):
        """Processing should divide weights by 100 (CPS 2 implied decimals)."""
        raw_df = pd.DataFrame({
            "household_id": [1],
            "person_seq": [1],
            "age": [30],
            "march_supplement_weight": [10000],  # Raw weight
            "state_fips": [6],
        })

        result = process_cps_data(raw_df)

        # Weight should be divided by 100
        assert result.iloc[0]["weight"] == 100.0

    def test_process_cps_data_filters_positive_weights(self):
        """Processing should filter to positive weights only."""
        raw_df = pd.DataFrame({
            "household_id": [1, 2, 3],
            "person_seq": [1, 1, 1],
            "age": [30, 40, 50],
            "march_supplement_weight": [100, 0, -10],  # Only first is positive
            "state_fips": [1, 2, 3],
        })

        result = process_cps_data(raw_df)

        assert len(result) == 1
        assert result.iloc[0]["age"] == 30

    def test_process_cps_data_calculates_employment(self):
        """Employment status should be derived from class_of_worker."""
        raw_df = pd.DataFrame({
            "household_id": [1, 2, 3],
            "person_seq": [1, 1, 1],
            "age": [30, 40, 50],
            "march_supplement_weight": [100, 100, 100],
            "class_of_worker": [1, 0, 2],  # 1=employed, 0=not, 2=employed
            "state_fips": [1, 2, 3],
        })

        result = process_cps_data(raw_df)

        assert result.iloc[0]["employment_status"] == 1  # Employed
        assert result.iloc[1]["employment_status"] == 0  # Not employed
        assert result.iloc[2]["employment_status"] == 1  # Employed

    def test_download_cps_zip_rejects_invalid_year(self):
        """Should raise error for unavailable years."""
        with pytest.raises(ValueError, match="not available"):
            download_cps_zip(1990)  # Too old

    def test_extract_person_data_handles_missing_columns(self):
        """Should gracefully handle missing optional columns."""
        # Create a minimal ZIP with just required columns
        csv_content = "PH_SEQ,P_SEQ,A_AGE,A_FNLWGT,GESTFIPS\n1,1,30,100,6\n"

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("pppub24.csv", csv_content)

        zip_content = zip_buffer.getvalue()

        df = extract_person_data(zip_content, 2023)

        # Should have at least the columns we provided
        assert len(df) == 1
        assert "age" in df.columns or "A_AGE" in df.columns


class TestLoaderIntegration:
    """Tests for loader integration with CPS downloader.

    These tests require the calibration package which may have optional
    dependencies. They are skipped if the imports fail.
    """

    @pytest.fixture(autouse=True)
    def skip_if_calibration_unavailable(self):
        """Skip these tests if calibration module can't be imported."""
        pytest.importorskip("calibration")

    def test_load_microdata_loads_cps_with_correct_weights(self):
        """CPS data should have reasonable total weight (~330M for US pop)."""
        from calibration import load_microdata
        from pathlib import Path

        # Only run if CPS 2023 exists (it should after download)
        data_path = Path(__file__).parent.parent / "micro" / "us" / "cps_2023.parquet"
        if not data_path.exists():
            pytest.skip("CPS 2023 not downloaded")

        df = load_microdata(source="cps", year=2023)

        # Should have data
        assert len(df) > 100_000  # CPS has 100k+ person records
        assert "weight" in df.columns

        # Weight should sum to ~330M (US population), not 33B
        total_weight = df["weight"].sum()
        assert 300_000_000 < total_weight < 400_000_000, (
            f"Total weight {total_weight:,.0f} should be ~330M, not {total_weight/1e9:.1f}B"
        )

    def test_load_microdata_falls_back_to_synthetic(self):
        """Should fall back to synthetic when CPS unavailable."""
        from calibration import load_microdata
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = load_microdata(source="cps", year=2015)  # Old year, won't download

            # Should have data (synthetic fallback)
            assert len(df) > 0
            assert "weight" in df.columns

            # Should have warned about fallback
            assert any("falling back to synthetic" in str(warning.message).lower()
                      for warning in w)
