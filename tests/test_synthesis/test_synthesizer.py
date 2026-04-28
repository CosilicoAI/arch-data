"""
Tests for the complete TaxSynthesizer pipeline.

TDD: Integration tests that verify the full synthesis workflow.

Pipeline:
1. Load PUF microdata (training)
2. Fit transforms on PUF
3. Train discrete model for binary/categorical vars
4. Train normalizing flow for continuous vars
5. Generate synthetic tax variables for CPS demographics
6. Validate synthetic data quality
"""

import pytest
import numpy as np
import pandas as pd


class TestTaxSynthesizerInit:
    """Test TaxSynthesizer initialization."""

    def test_default_initialization(self):
        """Should initialize with default parameters."""
        from micro.us.synthesis import TaxSynthesizer

        synth = TaxSynthesizer()

        assert synth.continuous_vars is not None
        assert synth.discrete_vars is not None
        assert synth.demographic_vars is not None

    def test_custom_variable_specification(self):
        """Should accept custom variable lists."""
        from micro.us.synthesis import TaxSynthesizer

        synth = TaxSynthesizer(
            continuous_vars=['wages', 'capital_gains'],
            discrete_vars=['is_itemizer'],
            demographic_vars=['age', 'filing_status']
        )

        assert synth.continuous_vars == ['wages', 'capital_gains']
        assert synth.discrete_vars == ['is_itemizer']
        assert synth.demographic_vars == ['age', 'filing_status']


class TestTaxSynthesizerFit:
    """Test TaxSynthesizer training on PUF data."""

    @pytest.fixture
    def sample_puf_data(self):
        """Create sample PUF-like data for testing."""
        np.random.seed(42)
        n = 1000

        # Demographics
        age = np.random.randint(18, 80, n)
        filing_status = np.random.choice([1, 2, 3, 4], n, p=[0.4, 0.4, 0.1, 0.1])
        n_dependents = np.random.poisson(0.5, n)

        # Tax variables (with realistic correlations)
        base_income = np.random.lognormal(10, 1, n)
        wages = base_income * np.random.uniform(0.7, 1.0, n)
        wages[np.random.random(n) < 0.1] = 0  # 10% have no wages

        capital_gains = np.where(
            base_income > np.percentile(base_income, 70),
            np.random.lognormal(8, 2, n),
            0
        )

        is_itemizer = (base_income > np.percentile(base_income, 60)).astype(int)

        return pd.DataFrame({
            'age': age,
            'filing_status': filing_status,
            'n_dependents': n_dependents,
            'wages': wages,
            'capital_gains': capital_gains,
            'is_itemizer': is_itemizer,
            'weight': np.ones(n)
        })

    def test_fit_runs_without_error(self, sample_puf_data):
        """Fit should complete without errors."""
        from micro.us.synthesis import TaxSynthesizer

        synth = TaxSynthesizer(
            continuous_vars=['wages', 'capital_gains'],
            discrete_vars=['is_itemizer'],
            demographic_vars=['age', 'filing_status', 'n_dependents']
        )

        synth.fit(sample_puf_data, epochs=10)  # Short for testing

        assert synth.is_fitted_

    def test_fit_learns_transforms(self, sample_puf_data):
        """Fit should learn data transforms."""
        from micro.us.synthesis import TaxSynthesizer

        synth = TaxSynthesizer(
            continuous_vars=['wages', 'capital_gains'],
            discrete_vars=['is_itemizer'],
            demographic_vars=['age', 'filing_status', 'n_dependents']
        )

        synth.fit(sample_puf_data, epochs=10)

        assert hasattr(synth, 'transformer_')
        assert 'wages' in synth.transformer_.transformers_

    def test_fit_trains_flow_model(self, sample_puf_data):
        """Fit should train the normalizing flow."""
        from micro.us.synthesis import TaxSynthesizer

        synth = TaxSynthesizer(
            continuous_vars=['wages', 'capital_gains'],
            discrete_vars=['is_itemizer'],
            demographic_vars=['age', 'filing_status', 'n_dependents']
        )

        synth.fit(sample_puf_data, epochs=50)

        assert hasattr(synth, 'flow_model_')
        assert synth.training_loss_[-1] < synth.training_loss_[0]  # Loss decreased


class TestTaxSynthesizerGenerate:
    """Test synthetic data generation."""

    @pytest.fixture
    def fitted_synthesizer(self, sample_puf_data):
        """Return a fitted synthesizer."""
        from micro.us.synthesis import TaxSynthesizer

        synth = TaxSynthesizer(
            continuous_vars=['wages', 'capital_gains'],
            discrete_vars=['is_itemizer'],
            demographic_vars=['age', 'filing_status', 'n_dependents']
        )

        synth.fit(sample_puf_data, epochs=50)
        return synth

    @pytest.fixture
    def sample_puf_data(self):
        """Create sample PUF-like data."""
        np.random.seed(42)
        n = 1000

        age = np.random.randint(18, 80, n)
        filing_status = np.random.choice([1, 2, 3, 4], n, p=[0.4, 0.4, 0.1, 0.1])
        n_dependents = np.random.poisson(0.5, n)

        base_income = np.random.lognormal(10, 1, n)
        wages = base_income * np.random.uniform(0.7, 1.0, n)
        wages[np.random.random(n) < 0.1] = 0

        capital_gains = np.where(
            base_income > np.percentile(base_income, 70),
            np.random.lognormal(8, 2, n),
            0
        )

        is_itemizer = (base_income > np.percentile(base_income, 60)).astype(int)

        return pd.DataFrame({
            'age': age,
            'filing_status': filing_status,
            'n_dependents': n_dependents,
            'wages': wages,
            'capital_gains': capital_gains,
            'is_itemizer': is_itemizer,
            'weight': np.ones(n)
        })

    @pytest.fixture
    def sample_cps_demographics(self):
        """Create sample CPS demographics for generation."""
        np.random.seed(123)
        n = 500

        return pd.DataFrame({
            'age': np.random.randint(18, 80, n),
            'filing_status': np.random.choice([1, 2, 3, 4], n, p=[0.4, 0.4, 0.1, 0.1]),
            'n_dependents': np.random.poisson(0.5, n),
            'weight': np.random.uniform(1000, 5000, n)
        })

    def test_generate_returns_dataframe(self, fitted_synthesizer, sample_cps_demographics):
        """Generate should return a DataFrame."""
        result = fitted_synthesizer.generate(sample_cps_demographics)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_cps_demographics)

    def test_generate_includes_all_variables(self, fitted_synthesizer, sample_cps_demographics):
        """Generated data should include all tax variables."""
        result = fitted_synthesizer.generate(sample_cps_demographics)

        # Should have demographic vars (preserved)
        assert 'age' in result.columns
        assert 'filing_status' in result.columns

        # Should have generated tax vars
        assert 'wages' in result.columns
        assert 'capital_gains' in result.columns
        assert 'is_itemizer' in result.columns

    def test_generate_preserves_demographics(self, fitted_synthesizer, sample_cps_demographics):
        """Demographics should be preserved exactly."""
        result = fitted_synthesizer.generate(sample_cps_demographics)

        pd.testing.assert_series_equal(result['age'], sample_cps_demographics['age'])
        pd.testing.assert_series_equal(result['filing_status'], sample_cps_demographics['filing_status'])

    def test_generated_continuous_vars_are_numeric(self, fitted_synthesizer, sample_cps_demographics):
        """Continuous variables should be numeric and non-negative."""
        result = fitted_synthesizer.generate(sample_cps_demographics)

        assert result['wages'].dtype in [np.float32, np.float64]
        assert (result['wages'] >= 0).all()

        assert result['capital_gains'].dtype in [np.float32, np.float64]
        assert (result['capital_gains'] >= 0).all()

    def test_generated_discrete_vars_are_valid(self, fitted_synthesizer, sample_cps_demographics):
        """Discrete variables should have valid values."""
        result = fitted_synthesizer.generate(sample_cps_demographics)

        # Binary variable
        assert set(result['is_itemizer'].unique()).issubset({0, 1})

    def test_generate_is_stochastic(self, fitted_synthesizer, sample_cps_demographics):
        """Multiple generations should give different results."""
        result1 = fitted_synthesizer.generate(sample_cps_demographics)
        result2 = fitted_synthesizer.generate(sample_cps_demographics)

        # Should not be identical
        assert not np.allclose(result1['wages'].values, result2['wages'].values)

    def test_generate_with_seed_is_reproducible(self, fitted_synthesizer, sample_cps_demographics):
        """Generation with same seed should be reproducible."""
        result1 = fitted_synthesizer.generate(sample_cps_demographics, seed=42)
        result2 = fitted_synthesizer.generate(sample_cps_demographics, seed=42)

        np.testing.assert_array_equal(result1['wages'].values, result2['wages'].values)


class TestTaxSynthesizerQuality:
    """Test quality of synthetic data."""

    @pytest.fixture
    def sample_puf_data(self):
        """Create sample PUF-like data with realistic structure."""
        np.random.seed(42)
        n = 2000

        age = np.random.randint(18, 80, n)
        filing_status = np.random.choice([1, 2, 3, 4], n, p=[0.4, 0.4, 0.1, 0.1])
        n_dependents = np.random.poisson(0.5, n)

        # Correlated income structure
        base_income = np.random.lognormal(10.5, 1.2, n)
        wages = base_income * np.random.uniform(0.6, 1.0, n)
        wages[np.random.random(n) < 0.08] = 0

        # Capital gains correlate with high income
        has_cap_gains = (base_income > np.percentile(base_income, 65)) & (np.random.random(n) > 0.3)
        capital_gains = np.where(has_cap_gains, np.random.lognormal(9, 1.5, n), 0)

        # Itemization correlates with income
        is_itemizer = (base_income > np.percentile(base_income, 55)).astype(int)

        return pd.DataFrame({
            'age': age,
            'filing_status': filing_status,
            'n_dependents': n_dependents,
            'wages': wages,
            'capital_gains': capital_gains,
            'is_itemizer': is_itemizer,
            'weight': np.ones(n)
        })

    @pytest.fixture
    def fitted_synthesizer(self, sample_puf_data):
        """Return a well-trained synthesizer."""
        from micro.us.synthesis import TaxSynthesizer

        synth = TaxSynthesizer(
            continuous_vars=['wages', 'capital_gains'],
            discrete_vars=['is_itemizer'],
            demographic_vars=['age', 'filing_status', 'n_dependents'],
            flow_layers=6,
            hidden_dim=64
        )

        synth.fit(sample_puf_data, epochs=200)
        return synth

    def test_marginal_distribution_similarity(self, fitted_synthesizer, sample_puf_data):
        """Marginal distributions should be roughly similar to training data.

        Note: This is a soft quality test - strict quality validation is done
        by the validation framework, not unit tests. Here we just verify
        the synthesis is working (KS stat < 0.6 = some learning happening).
        """
        # Generate synthetic data with same demographics
        synthetic = fitted_synthesizer.generate(sample_puf_data[['age', 'filing_status', 'n_dependents', 'weight']])

        # Compare wages distribution
        from scipy import stats
        ks_stat, p_value = stats.ks_2samp(
            sample_puf_data['wages'],
            synthetic['wages']
        )

        # Soft threshold: KS < 0.6 means some learning is happening
        # (Strict quality testing is done in validation framework)
        assert ks_stat < 0.6, f"Wages distribution not learned at all: KS={ks_stat:.3f}"

    def test_correlation_preservation(self, fitted_synthesizer, sample_puf_data):
        """Correlations should be preserved."""
        synthetic = fitted_synthesizer.generate(sample_puf_data[['age', 'filing_status', 'n_dependents', 'weight']])

        # Check wage-capital gains correlation (should both be positive due to income effect)
        puf_mask = (sample_puf_data['wages'] > 0) & (sample_puf_data['capital_gains'] > 0)
        synth_mask = (synthetic['wages'] > 0) & (synthetic['capital_gains'] > 0)

        if puf_mask.sum() > 10 and synth_mask.sum() > 10:
            puf_corr = np.corrcoef(
                np.log1p(sample_puf_data.loc[puf_mask, 'wages']),
                np.log1p(sample_puf_data.loc[puf_mask, 'capital_gains'])
            )[0, 1]

            synth_corr = np.corrcoef(
                np.log1p(synthetic.loc[synth_mask, 'wages']),
                np.log1p(synthetic.loc[synth_mask, 'capital_gains'])
            )[0, 1]

            assert abs(puf_corr - synth_corr) < 0.25, \
                f"Correlation not preserved: puf={puf_corr:.3f}, synth={synth_corr:.3f}"

    def test_zero_fraction_similarity(self, fitted_synthesizer, sample_puf_data):
        """Zero fractions should be similar."""
        synthetic = fitted_synthesizer.generate(sample_puf_data[['age', 'filing_status', 'n_dependents', 'weight']])

        for var in ['wages', 'capital_gains']:
            puf_zero_frac = (sample_puf_data[var] == 0).mean()
            synth_zero_frac = (synthetic[var] == 0).mean()

            assert abs(puf_zero_frac - synth_zero_frac) < 0.10, \
                f"{var} zero fraction: puf={puf_zero_frac:.3f}, synth={synth_zero_frac:.3f}"


class TestSaveLoad:
    """Test model serialization."""

    @pytest.fixture
    def sample_puf_data(self):
        """Create sample data."""
        np.random.seed(42)
        n = 500
        return pd.DataFrame({
            'age': np.random.randint(18, 80, n),
            'filing_status': np.random.choice([1, 2, 3, 4], n),
            'n_dependents': np.random.poisson(0.5, n),
            'wages': np.random.lognormal(10, 1, n),
            'capital_gains': np.where(np.random.random(n) > 0.7, np.random.lognormal(8, 1, n), 0),
            'is_itemizer': (np.random.random(n) > 0.5).astype(int),
            'weight': np.ones(n)
        })

    def test_save_and_load(self, sample_puf_data, tmp_path):
        """Should save and load model correctly."""
        from micro.us.synthesis import TaxSynthesizer

        synth = TaxSynthesizer(
            continuous_vars=['wages', 'capital_gains'],
            discrete_vars=['is_itemizer'],
            demographic_vars=['age', 'filing_status', 'n_dependents']
        )
        synth.fit(sample_puf_data, epochs=20)

        # Save
        save_path = tmp_path / "model.pt"
        synth.save(save_path)

        # Load
        loaded = TaxSynthesizer.load(save_path)

        # Should generate same results with same seed
        demo_data = sample_puf_data[['age', 'filing_status', 'n_dependents', 'weight']].head(10)

        result1 = synth.generate(demo_data, seed=42)
        result2 = loaded.generate(demo_data, seed=42)

        np.testing.assert_array_almost_equal(result1['wages'].values, result2['wages'].values)


# =============================================================================
# Test __init__.py exports
# =============================================================================

def test_synthesis_module_exports():
    """Synthesis module should export main classes."""
    from micro.us.synthesis import TaxSynthesizer
    from micro.us.synthesis import ConditionalMAF
    from micro.us.synthesis import MultiVariableTransformer

    assert TaxSynthesizer is not None
    assert ConditionalMAF is not None
    assert MultiVariableTransformer is not None
