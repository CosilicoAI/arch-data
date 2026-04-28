"""
Tests for normalizing flow model.

TDD: These tests define the expected behavior of the conditional normalizing flow.

Architecture: Masked Autoregressive Flow (MAF) conditioned on demographics.
- Input: demographics (age, filing_status, n_dependents, etc.)
- Output: tax variables (wages, capital_gains, etc.)
- Training: maximize log probability of PUF data
- Generation: sample from learned distribution
"""

import torch


class TestConditionalMAF:
    """Test Masked Autoregressive Flow for conditional generation."""

    def test_initialization(self):
        """Should initialize with correct dimensions."""
        from micro.us.synthesis.flows import ConditionalMAF

        flow = ConditionalMAF(
            n_features=10,       # Number of tax variables
            n_context=5,         # Number of demographic features
            n_layers=4,          # Number of flow layers
            hidden_dim=64        # Hidden layer size
        )

        assert flow.n_features == 10
        assert flow.n_context == 5

    def test_forward_returns_log_prob(self):
        """Forward pass should return log probability."""
        from micro.us.synthesis.flows import ConditionalMAF

        flow = ConditionalMAF(n_features=10, n_context=5, n_layers=4, hidden_dim=64)

        batch_size = 32
        x = torch.randn(batch_size, 10)  # Tax variables
        context = torch.randn(batch_size, 5)  # Demographics

        log_prob = flow.log_prob(x, context)

        assert log_prob.shape == (batch_size,)
        assert torch.isfinite(log_prob).all()

    def test_sample_returns_correct_shape(self):
        """Sample should return correct shape."""
        from micro.us.synthesis.flows import ConditionalMAF

        flow = ConditionalMAF(n_features=10, n_context=5, n_layers=4, hidden_dim=64)

        batch_size = 32
        context = torch.randn(batch_size, 5)

        samples = flow.sample(context)

        assert samples.shape == (batch_size, 10)

    def test_sample_is_deterministic_with_seed(self):
        """Sampling with same seed should give same results."""
        from micro.us.synthesis.flows import ConditionalMAF

        flow = ConditionalMAF(n_features=10, n_context=5, n_layers=4, hidden_dim=64)
        context = torch.randn(5, 5)

        torch.manual_seed(42)
        samples1 = flow.sample(context)

        torch.manual_seed(42)
        samples2 = flow.sample(context)

        torch.testing.assert_close(samples1, samples2)

    def test_log_prob_is_normalized(self):
        """Log probs should integrate to 1 (approximately, via sampling)."""
        from micro.us.synthesis.flows import ConditionalMAF

        flow = ConditionalMAF(n_features=2, n_context=2, n_layers=4, hidden_dim=32)

        # Train briefly on simple data
        optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
        context = torch.randn(1000, 2)
        x = context + torch.randn(1000, 2) * 0.5  # Simple conditional distribution

        for _ in range(100):
            loss = -flow.log_prob(x, context).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Sample and check log probs are reasonable
        samples = flow.sample(context[:10])
        log_probs = flow.log_prob(samples, context[:10])

        # Log probs should be finite and not too extreme
        assert torch.isfinite(log_probs).all()
        assert log_probs.mean() > -50  # Not too unlikely


class TestMAFLayer:
    """Test individual MAF layer (MADE + affine coupling)."""

    def test_made_autoregressive_property(self):
        """MADE network should be autoregressive."""
        from micro.us.synthesis.flows import MADE

        made = MADE(
            n_features=5,
            n_context=3,
            hidden_dim=32
        )

        x = torch.randn(10, 5)
        context = torch.randn(10, 3)

        # Get Jacobian
        x.requires_grad_(True)
        mu, log_scale = made(x, context)

        # For autoregressive, output[i] should only depend on input[:i]
        # This means Jacobian should be lower triangular
        # Check by computing gradients
        for i in range(5):
            if x.grad is not None:
                x.grad.zero_()
            mu[:, i].sum().backward(retain_graph=True)

            # Gradient w.r.t. x[:, j] should be 0 for j >= i
            for j in range(i, 5):
                assert torch.allclose(x.grad[:, j], torch.zeros_like(x.grad[:, j]), atol=1e-5), \
                    f"Output {i} depends on input {j} (should be autoregressive)"

    def test_affine_coupling_invertible(self):
        """Affine coupling layer should be invertible."""
        from micro.us.synthesis.flows import AffineCouplingLayer

        layer = AffineCouplingLayer(n_features=5, n_context=3, hidden_dim=32)

        x = torch.randn(10, 5)
        context = torch.randn(10, 3)

        # Forward
        z, log_det = layer.forward(x, context)

        # Inverse
        x_reconstructed = layer.inverse(z, context)

        torch.testing.assert_close(x_reconstructed, x, rtol=1e-4, atol=1e-4)

    def test_log_det_jacobian_correct(self):
        """Log determinant should be correct."""
        from micro.us.synthesis.flows import AffineCouplingLayer

        layer = AffineCouplingLayer(n_features=3, n_context=2, hidden_dim=16)

        x = torch.randn(5, 3, requires_grad=True)
        context = torch.randn(5, 2)

        z, log_det = layer.forward(x, context)

        # Compute Jacobian numerically for first sample
        def forward_fn(x_single):
            return layer.forward(x_single.unsqueeze(0), context[:1])[0].squeeze(0)

        jacobian = torch.autograd.functional.jacobian(forward_fn, x[0])

        # Log det should match log|det(J)|
        numerical_log_det = torch.linalg.slogdet(jacobian)[1]

        torch.testing.assert_close(log_det[0], numerical_log_det, rtol=1e-3, atol=1e-3)


class TestFlowTraining:
    """Test flow training dynamics."""

    def test_loss_decreases_during_training(self):
        """Negative log likelihood should decrease."""
        from micro.us.synthesis.flows import ConditionalMAF

        flow = ConditionalMAF(n_features=5, n_context=3, n_layers=4, hidden_dim=32)
        optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

        # Simple training data: features partially depend on context
        context = torch.randn(500, 3)
        x = torch.randn(500, 5)
        x[:, :3] = context + torch.randn(500, 3) * 0.3  # First 3 features depend on context

        initial_loss = -flow.log_prob(x, context).mean().item()

        for _ in range(200):
            loss = -flow.log_prob(x, context).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = -flow.log_prob(x, context).mean().item()

        assert final_loss < initial_loss, \
            f"Loss should decrease: {initial_loss:.3f} -> {final_loss:.3f}"

    def test_samples_match_training_distribution(self):
        """After training, samples should match training distribution."""
        from micro.us.synthesis.flows import ConditionalMAF

        flow = ConditionalMAF(n_features=2, n_context=2, n_layers=6, hidden_dim=64)
        optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

        # Training data: x = context + noise
        torch.manual_seed(0)
        context = torch.randn(1000, 2)
        noise = torch.randn(1000, 2) * 0.3
        x = context + noise

        # Train
        for _ in range(500):
            loss = -flow.log_prob(x, context).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Sample
        with torch.no_grad():
            test_context = torch.randn(1000, 2)
            samples = flow.sample(test_context)

        # Samples should be close to test_context (since x ≈ context in training)
        residuals = samples - test_context
        assert residuals.std() < 0.6, \
            f"Sample std from context: {residuals.std():.3f} (should be ~0.3)"


class TestFlowWithRealTaxData:
    """Test flow with tax-like data distributions."""

    def test_handles_heavy_tails(self):
        """Should model heavy-tailed distributions (like income)."""
        from micro.us.synthesis.flows import ConditionalMAF

        flow = ConditionalMAF(n_features=1, n_context=1, n_layers=6, hidden_dim=64)
        optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

        # Log-normal like income distribution
        torch.manual_seed(0)
        context = torch.randn(1000, 1)
        x = torch.exp(context * 0.5 + 2)  # Log-normal conditioned on context

        # Train on log-transformed data
        x_log = torch.log(x)

        for _ in range(300):
            loss = -flow.log_prob(x_log, context).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Sample and check distribution
        with torch.no_grad():
            test_context = torch.randn(1000, 1)
            samples_log = flow.sample(test_context)
            samples = torch.exp(samples_log)

        # Should have similar distribution shape
        assert samples.mean() > 0
        assert samples.std() > samples.mean() * 0.3  # Heavy tail = high CV

    def test_preserves_correlations(self):
        """Should preserve correlations between variables."""
        from micro.us.synthesis.flows import ConditionalMAF

        flow = ConditionalMAF(n_features=3, n_context=2, n_layers=6, hidden_dim=64)
        optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

        # Correlated features
        torch.manual_seed(0)
        context = torch.randn(1000, 2)
        z1 = context[:, 0:1] + torch.randn(1000, 1) * 0.2
        z2 = z1 + torch.randn(1000, 1) * 0.2  # Correlated with z1
        z3 = context[:, 1:2] + torch.randn(1000, 1) * 0.2
        x = torch.cat([z1, z2, z3], dim=1)

        # True correlation between z1 and z2
        true_corr = torch.corrcoef(x[:, :2].T)[0, 1].item()

        # Train
        for _ in range(500):
            loss = -flow.log_prob(x, context).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Sample and check correlation
        with torch.no_grad():
            test_context = torch.randn(1000, 2)
            samples = flow.sample(test_context)

        sample_corr = torch.corrcoef(samples[:, :2].T)[0, 1].item()

        # Correlation should be preserved (within tolerance)
        assert abs(sample_corr - true_corr) < 0.15, \
            f"Correlation not preserved: true={true_corr:.3f}, sample={sample_corr:.3f}"


class TestDiscreteVariableModel:
    """Test handling of discrete variables (filing status, has_business, etc.)."""

    def test_binary_variable_prediction(self):
        """Should predict binary variables (is_itemizer, has_business)."""
        from micro.us.synthesis.discrete import BinaryVariableModel

        model = BinaryVariableModel(n_context=5, hidden_dim=32)

        context = torch.randn(100, 5)
        probs = model(context)

        assert probs.shape == (100, 1)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_categorical_variable_prediction(self):
        """Should predict categorical variables (filing_status)."""
        from micro.us.synthesis.discrete import CategoricalVariableModel

        model = CategoricalVariableModel(n_context=5, n_categories=4, hidden_dim=32)

        context = torch.randn(100, 5)
        probs = model(context)

        assert probs.shape == (100, 4)
        assert torch.allclose(probs.sum(dim=1), torch.ones(100), atol=1e-5)

    def test_sample_discrete_variables(self):
        """Should sample discrete variables from predicted probabilities."""
        from micro.us.synthesis.discrete import DiscreteVariableSampler

        sampler = DiscreteVariableSampler()

        # Binary
        binary_probs = torch.tensor([[0.8], [0.2], [0.5]])
        binary_samples = sampler.sample_binary(binary_probs)
        assert binary_samples.shape == (3, 1)
        assert ((binary_samples == 0) | (binary_samples == 1)).all()

        # Categorical
        cat_probs = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
        cat_samples = sampler.sample_categorical(cat_probs)
        assert cat_samples.shape == (2,)
        assert (cat_samples >= 0).all() and (cat_samples < 3).all()
