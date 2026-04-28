"""Tests for microdata loader."""

import pandas as pd
import pytest

from calibration.loader import load_microdata


class TestLoadMicrodataSynthetic:
    """Tests for synthetic microdata generation."""

    def test_returns_dataframe(self):
        """load_microdata should return a pandas DataFrame."""
        df = load_microdata(source="synthetic", year=2021)
        assert isinstance(df, pd.DataFrame)

    def test_has_weight_column(self):
        """Synthetic data should have a weight column."""
        df = load_microdata(source="synthetic", year=2021)
        assert "weight" in df.columns

    def test_has_required_columns(self):
        """Synthetic data should have demographic and income columns."""
        df = load_microdata(source="synthetic", year=2021)

        required_columns = [
            "weight",
            "age",
            "income",
            "employment_status",
            "has_children",
            "state_fips",
        ]

        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"

    def test_approximately_10000_rows(self):
        """Synthetic data should have approximately 10,000 rows."""
        df = load_microdata(source="synthetic", year=2021)
        assert 9000 <= len(df) <= 11000

    def test_age_range_realistic(self):
        """Ages should be in realistic range (0-120)."""
        df = load_microdata(source="synthetic", year=2021)
        assert df["age"].min() >= 0
        assert df["age"].max() <= 120

    def test_weights_positive(self):
        """All weights should be positive."""
        df = load_microdata(source="synthetic", year=2021)
        assert (df["weight"] > 0).all()

    def test_employment_status_values(self):
        """Employment status should be encoded correctly."""
        df = load_microdata(source="synthetic", year=2021)
        # Employment status: 0=not in labor force, 1=employed, 2=unemployed
        valid_values = {0, 1, 2}
        assert set(df["employment_status"].unique()).issubset(valid_values)

    def test_has_children_binary(self):
        """has_children should be binary (0 or 1)."""
        df = load_microdata(source="synthetic", year=2021)
        assert set(df["has_children"].unique()).issubset({0, 1})

    def test_state_fips_valid(self):
        """State FIPS codes should be valid US state codes."""
        df = load_microdata(source="synthetic", year=2021)
        # FIPS codes are 1-56 (with some gaps)
        fips = df["state_fips"].astype(int)
        assert fips.min() >= 1
        assert fips.max() <= 56

    def test_income_nonnegative(self):
        """Income should be non-negative."""
        df = load_microdata(source="synthetic", year=2021)
        assert (df["income"] >= 0).all()

    def test_reproducible_with_seed(self):
        """Same seed should produce same data."""
        df1 = load_microdata(source="synthetic", year=2021, seed=42)
        df2 = load_microdata(source="synthetic", year=2021, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_different_data(self):
        """Different seeds should produce different data."""
        df1 = load_microdata(source="synthetic", year=2021, seed=42)
        df2 = load_microdata(source="synthetic", year=2021, seed=123)
        assert not df1.equals(df2)


class TestLoadMicrodataVariables:
    """Tests for variable filtering."""

    def test_filter_variables(self):
        """Should return only requested variables plus weight."""
        df = load_microdata(
            source="synthetic",
            year=2021,
            variables=["age", "income"],
        )
        # Should have weight plus requested variables
        assert set(df.columns) == {"weight", "age", "income"}

    def test_weight_always_included(self):
        """Weight should always be included even if not specified."""
        df = load_microdata(
            source="synthetic",
            year=2021,
            variables=["age"],
        )
        assert "weight" in df.columns


class TestLoadMicrodataCPS:
    """Tests for CPS data loading (with fallback to synthetic)."""

    def test_cps_source_returns_data(self):
        """CPS source should return data (synthetic fallback if not available)."""
        df = load_microdata(source="cps", year=2021)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "weight" in df.columns

    def test_cps_invalid_year_raises(self):
        """Invalid year should raise ValueError."""
        with pytest.raises(ValueError, match="[Yy]ear"):
            load_microdata(source="cps", year=1800)


class TestLoadMicrodataEdgeCases:
    """Tests for edge cases."""

    def test_invalid_source_raises(self):
        """Invalid source should raise ValueError."""
        with pytest.raises(ValueError, match="[Ss]ource"):
            load_microdata(source="invalid_source", year=2021)

    def test_future_year_raises(self):
        """Future year should raise ValueError."""
        with pytest.raises(ValueError, match="[Yy]ear"):
            load_microdata(source="synthetic", year=2099)
