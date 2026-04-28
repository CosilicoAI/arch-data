"""
Tests for data transformations.

TDD: These tests define the expected behavior of transforms.
Transforms handle:
- Zero-inflated variables (many tax vars are 0 for most people)
- Heavy-tailed distributions (log transform)
- Standardization (for neural network training)
"""

import numpy as np
import torch


class TestZeroInflatedTransform:
    """Test zero-inflated variable handling."""

    def test_split_zeros_and_positives(self):
        """Should correctly split data into zero indicator and positive values."""
        from micro.us.synthesis.transforms import ZeroInflatedTransform

        transform = ZeroInflatedTransform()

        # Input: 50% zeros, 50% positive values
        x = np.array([0, 100, 0, 200, 0, 300, 0, 400])

        indicator, positive_values = transform.split(x)

        # Indicator should be binary
        assert indicator.dtype == np.float32 or indicator.dtype == np.float64
        np.testing.assert_array_equal(indicator, [0, 1, 0, 1, 0, 1, 0, 1])

        # Positive values should only contain non-zeros
        np.testing.assert_array_equal(positive_values, [100, 200, 300, 400])

    def test_recombine_zeros_and_positives(self):
        """Should correctly recombine indicator and values."""
        from micro.us.synthesis.transforms import ZeroInflatedTransform

        transform = ZeroInflatedTransform()

        indicator = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        positive_values = np.array([100, 200, 300, 400])

        result = transform.combine(indicator, positive_values)

        np.testing.assert_array_equal(result, [0, 100, 0, 200, 0, 300, 0, 400])

    def test_roundtrip(self):
        """Split then combine should give original data."""
        from micro.us.synthesis.transforms import ZeroInflatedTransform

        transform = ZeroInflatedTransform()

        original = np.array([0, 50.5, 0, 0, 123.7, 0, 999.9])
        indicator, positives = transform.split(original)
        result = transform.combine(indicator, positives)

        np.testing.assert_allclose(result, original, rtol=1e-5)

    def test_handles_all_zeros(self):
        """Should handle edge case of all zeros."""
        from micro.us.synthesis.transforms import ZeroInflatedTransform

        transform = ZeroInflatedTransform()

        x = np.array([0, 0, 0, 0])
        indicator, positives = transform.split(x)

        assert len(positives) == 0
        np.testing.assert_array_equal(indicator, [0, 0, 0, 0])

    def test_handles_no_zeros(self):
        """Should handle edge case of no zeros."""
        from micro.us.synthesis.transforms import ZeroInflatedTransform

        transform = ZeroInflatedTransform()

        x = np.array([1, 2, 3, 4])
        indicator, positives = transform.split(x)

        np.testing.assert_array_equal(indicator, [1, 1, 1, 1])
        np.testing.assert_array_equal(positives, [1, 2, 3, 4])


class TestLogTransform:
    """Test log transformation for heavy-tailed distributions."""

    def test_forward_transform(self):
        """Log transform should handle positive values."""
        from micro.us.synthesis.transforms import LogTransform

        transform = LogTransform()

        x = np.array([1.0, 10.0, 100.0, 1000.0])
        result = transform.forward(x)

        expected = np.log(x)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_inverse_transform(self):
        """Inverse should recover original values."""
        from micro.us.synthesis.transforms import LogTransform

        transform = LogTransform()

        x = np.array([1.0, 10.0, 100.0, 1000.0])
        log_x = transform.forward(x)
        result = transform.inverse(log_x)

        np.testing.assert_allclose(result, x, rtol=1e-5)

    def test_handles_small_values(self):
        """Should handle values close to zero with offset."""
        from micro.us.synthesis.transforms import LogTransform

        transform = LogTransform(offset=1.0)  # log(x + 1)

        x = np.array([0.0, 0.1, 1.0, 10.0])
        result = transform.forward(x)

        expected = np.log(x + 1.0)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_roundtrip_with_offset(self):
        """Roundtrip should work with offset."""
        from micro.us.synthesis.transforms import LogTransform

        transform = LogTransform(offset=1.0)

        original = np.array([0.0, 0.5, 1.0, 100.0, 10000.0])
        result = transform.inverse(transform.forward(original))

        np.testing.assert_allclose(result, original, rtol=1e-5)


class TestStandardization:
    """Test standardization for neural network training."""

    def test_fit_computes_mean_std(self):
        """Fit should compute weighted mean and std."""
        from micro.us.synthesis.transforms import Standardizer

        standardizer = Standardizer()

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        standardizer.fit(x, weights)

        assert hasattr(standardizer, 'mean_')
        assert hasattr(standardizer, 'std_')
        np.testing.assert_allclose(standardizer.mean_, 3.0, rtol=1e-5)

    def test_transform_standardizes(self):
        """Transform should produce zero mean, unit variance."""
        from micro.us.synthesis.transforms import Standardizer

        standardizer = Standardizer()

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.ones_like(x)

        standardizer.fit(x, weights)
        result = standardizer.transform(x)

        np.testing.assert_allclose(result.mean(), 0.0, atol=1e-5)
        np.testing.assert_allclose(result.std(), 1.0, rtol=1e-2)

    def test_inverse_transform(self):
        """Inverse should recover original values."""
        from micro.us.synthesis.transforms import Standardizer

        standardizer = Standardizer()

        original = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        weights = np.ones_like(original)

        standardizer.fit(original, weights)
        standardized = standardizer.transform(original)
        result = standardizer.inverse_transform(standardized)

        np.testing.assert_allclose(result, original, rtol=1e-5)

    def test_handles_weighted_data(self):
        """Should correctly compute weighted statistics."""
        from micro.us.synthesis.transforms import Standardizer

        standardizer = Standardizer()

        # Value 1 has weight 3, value 5 has weight 1
        # Weighted mean = (1*3 + 5*1) / 4 = 2
        x = np.array([1.0, 5.0])
        weights = np.array([3.0, 1.0])

        standardizer.fit(x, weights)

        np.testing.assert_allclose(standardizer.mean_, 2.0, rtol=1e-5)


class TestTaxVariableTransformer:
    """Test the complete transformation pipeline for tax variables."""

    def test_fit_transform_pipeline(self):
        """Full pipeline should handle tax variable quirks."""
        from micro.us.synthesis.transforms import TaxVariableTransformer

        transformer = TaxVariableTransformer(
            zero_inflated=True,
            log_transform=True,
            standardize=True
        )

        # Typical tax variable: many zeros, heavy tail
        x = np.array([0, 0, 0, 100, 0, 500, 0, 0, 1000, 50000])
        weights = np.ones_like(x, dtype=float)

        transformer.fit(x, weights)
        transformed = transformer.transform(x)

        # Should be standardized
        positive_mask = x > 0
        assert transformed[positive_mask].std() < 10  # Roughly standardized

    def test_inverse_transform_recovers_original(self):
        """Inverse should recover original values."""
        from micro.us.synthesis.transforms import TaxVariableTransformer

        transformer = TaxVariableTransformer(
            zero_inflated=True,
            log_transform=True,
            standardize=True
        )

        original = np.array([0, 0, 100, 0, 500, 0, 1000, 50000])
        weights = np.ones_like(original, dtype=float)

        transformer.fit(original, weights)
        transformed = transformer.transform(original)
        result = transformer.inverse_transform(transformed)

        np.testing.assert_allclose(result, original, rtol=1e-3)

    def test_torch_compatibility(self):
        """Should work with PyTorch tensors."""
        from micro.us.synthesis.transforms import TaxVariableTransformer

        transformer = TaxVariableTransformer(
            zero_inflated=True,
            log_transform=True,
            standardize=True
        )

        x_np = np.array([0, 100, 0, 500, 1000])
        weights = np.ones_like(x_np, dtype=float)

        transformer.fit(x_np, weights)

        x_torch = torch.tensor(x_np, dtype=torch.float32)
        transformed = transformer.transform(x_torch)

        assert isinstance(transformed, torch.Tensor)


class TestMultiVariableTransformer:
    """Test transformer that handles multiple tax variables at once."""

    def test_fit_multiple_variables(self):
        """Should fit separate transforms for each variable."""
        from micro.us.synthesis.transforms import MultiVariableTransformer

        transformer = MultiVariableTransformer(
            var_names=['wages', 'capital_gains', 'dividends']
        )

        data = {
            'wages': np.array([50000, 60000, 0, 70000]),
            'capital_gains': np.array([0, 0, 10000, 0]),
            'dividends': np.array([0, 500, 1000, 0]),
            'weight': np.array([1.0, 1.0, 1.0, 1.0])
        }

        transformer.fit(data)

        # Each variable should have its own transformer
        assert 'wages' in transformer.transformers_
        assert 'capital_gains' in transformer.transformers_
        assert 'dividends' in transformer.transformers_

    def test_transform_all_variables(self):
        """Should transform all variables together."""
        from micro.us.synthesis.transforms import MultiVariableTransformer

        transformer = MultiVariableTransformer(
            var_names=['wages', 'capital_gains']
        )

        data = {
            'wages': np.array([50000, 60000, 0, 70000]),
            'capital_gains': np.array([0, 0, 10000, 0]),
            'weight': np.array([1.0, 1.0, 1.0, 1.0])
        }

        transformer.fit(data)
        result = transformer.transform(data)

        assert 'wages' in result
        assert 'capital_gains' in result

    def test_roundtrip_multiple_variables(self):
        """Roundtrip should recover all original values."""
        from micro.us.synthesis.transforms import MultiVariableTransformer

        transformer = MultiVariableTransformer(
            var_names=['wages', 'capital_gains', 'dividends']
        )

        original = {
            'wages': np.array([50000, 60000, 0, 70000]),
            'capital_gains': np.array([0, 0, 10000, 0]),
            'dividends': np.array([0, 500, 1000, 0]),
            'weight': np.array([1.0, 1.0, 1.0, 1.0])
        }

        transformer.fit(original)
        transformed = transformer.transform(original)
        result = transformer.inverse_transform(transformed)

        for var in ['wages', 'capital_gains', 'dividends']:
            np.testing.assert_allclose(result[var], original[var], rtol=1e-3)
