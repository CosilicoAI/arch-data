"""
Tests for synthetic data evaluation metrics.

TDD: Write tests first, then implement to pass.

Compares Cosilico synthesis vs PolicyEngine Enhanced CPS vs IRS SOI ground truth.
"""

import pytest
import numpy as np


# =============================================================================
# TEST: Marginal Fidelity
# =============================================================================

class TestMarginalFidelity:
    """Test that synthetic marginal distributions match PUF."""

    def test_kl_divergence_per_variable(self, synthetic_data, puf_data, tax_vars):
        """KL divergence for each variable should be below threshold."""
        from micro.us.synthesis.evaluation import compute_kl_divergence

        max_kl = 0.1  # Threshold - synthetic should be close to PUF

        for var in tax_vars:
            kl = compute_kl_divergence(
                synthetic_data[var],
                puf_data[var],
                weights_p=synthetic_data['weight'],
                weights_q=puf_data['weight']
            )
            assert kl < max_kl, f"KL divergence for {var} is {kl:.3f}, exceeds {max_kl}"

    def test_zero_inflation_accuracy(self, synthetic_data, puf_data, tax_vars):
        """Fraction of zeros should match within tolerance."""
        from micro.us.synthesis.evaluation import compute_zero_fraction

        tolerance = 0.02  # Within 2 percentage points

        for var in tax_vars:
            synth_zero_frac = compute_zero_fraction(synthetic_data[var], synthetic_data['weight'])
            puf_zero_frac = compute_zero_fraction(puf_data[var], puf_data['weight'])

            assert abs(synth_zero_frac - puf_zero_frac) < tolerance, \
                f"Zero fraction for {var}: synthetic={synth_zero_frac:.3f}, puf={puf_zero_frac:.3f}"

    def test_quantile_coverage(self, synthetic_data, puf_data, tax_vars):
        """Quantiles should match at 10th, 50th, 90th, 99th percentiles."""
        from micro.us.synthesis.evaluation import compute_weighted_quantile

        quantiles = [0.1, 0.5, 0.9, 0.99]
        tolerance_pct = 0.10  # Within 10% of PUF quantile

        for var in tax_vars:
            for q in quantiles:
                synth_q = compute_weighted_quantile(synthetic_data[var], synthetic_data['weight'], q)
                puf_q = compute_weighted_quantile(puf_data[var], puf_data['weight'], q)

                if puf_q > 0:  # Only check non-zero quantiles
                    rel_error = abs(synth_q - puf_q) / puf_q
                    assert rel_error < tolerance_pct, \
                        f"{var} q{q}: synthetic={synth_q:.0f}, puf={puf_q:.0f}, error={rel_error:.1%}"


# =============================================================================
# TEST: Joint Fidelity
# =============================================================================

class TestJointFidelity:
    """Test that correlations and joint distributions are preserved."""

    def test_correlation_matrix_distance(self, synthetic_data, puf_data, tax_vars):
        """Correlation matrices should be similar (Frobenius norm)."""
        from micro.us.synthesis.evaluation import compute_weighted_correlation_matrix

        synth_corr = compute_weighted_correlation_matrix(synthetic_data[tax_vars], synthetic_data['weight'])
        puf_corr = compute_weighted_correlation_matrix(puf_data[tax_vars], puf_data['weight'])

        frobenius_dist = np.linalg.norm(synth_corr - puf_corr, 'fro')
        max_dist = 2.0  # Threshold for correlation matrix distance

        assert frobenius_dist < max_dist, \
            f"Correlation matrix Frobenius distance {frobenius_dist:.2f} exceeds {max_dist}"

    def test_pairwise_correlations(self, synthetic_data, puf_data):
        """Key economic correlations should be preserved."""
        from micro.us.synthesis.evaluation import compute_weighted_correlation

        # Important economic relationships
        pairs = [
            ('wages', 'retirement_contributions'),  # Workers save for retirement
            ('wages', 'state_taxes_paid'),  # Higher wages → higher state taxes
            ('business_income', 'self_employment_tax'),  # SE income → SE tax
            ('adjusted_gross_income', 'itemized_deductions'),  # High income → itemize
            ('capital_gains', 'dividends'),  # Investment income correlates
        ]

        tolerance = 0.1  # Correlation should be within 0.1

        for var1, var2 in pairs:
            synth_corr = compute_weighted_correlation(
                synthetic_data[var1], synthetic_data[var2], synthetic_data['weight']
            )
            puf_corr = compute_weighted_correlation(
                puf_data[var1], puf_data[var2], puf_data['weight']
            )

            assert abs(synth_corr - puf_corr) < tolerance, \
                f"Correlation ({var1}, {var2}): synthetic={synth_corr:.3f}, puf={puf_corr:.3f}"

    def test_conditional_distributions(self, synthetic_data, puf_data):
        """Conditional distributions should match (e.g., cap_gains | high_income)."""
        from micro.us.synthesis.evaluation import compute_conditional_mean

        # High income = top 10% AGI
        synth_high_income_mask = synthetic_data['adjusted_gross_income'] > \
            np.percentile(synthetic_data['adjusted_gross_income'], 90)
        puf_high_income_mask = puf_data['adjusted_gross_income'] > \
            np.percentile(puf_data['adjusted_gross_income'], 90)

        vars_to_check = ['capital_gains', 'dividends', 'itemized_deductions']
        tolerance_pct = 0.15

        for var in vars_to_check:
            synth_cond_mean = compute_conditional_mean(
                synthetic_data[var], synthetic_data['weight'], synth_high_income_mask
            )
            puf_cond_mean = compute_conditional_mean(
                puf_data[var], puf_data['weight'], puf_high_income_mask
            )

            if puf_cond_mean > 0:
                rel_error = abs(synth_cond_mean - puf_cond_mean) / puf_cond_mean
                assert rel_error < tolerance_pct, \
                    f"E[{var}|high_income]: synthetic={synth_cond_mean:.0f}, puf={puf_cond_mean:.0f}"


# =============================================================================
# TEST: Policy Utility
# =============================================================================

class TestPolicyUtility:
    """Test that synthetic data gives same policy answers as PUF."""

    def test_tax_liability_by_bracket(self, synthetic_data, puf_data, irs_soi_targets):
        """Total tax liability by AGI bracket should match."""
        from micro.us.synthesis.evaluation import compute_tax_by_bracket

        brackets = ['under_25k', '25k_to_50k', '50k_to_100k', '100k_to_200k', '200k_to_500k', '500k_plus']
        tolerance_pct = 0.05

        synth_tax = compute_tax_by_bracket(synthetic_data, brackets)
        puf_tax = compute_tax_by_bracket(puf_data, brackets)

        for bracket in brackets:
            if puf_tax[bracket] > 0:
                rel_error = abs(synth_tax[bracket] - puf_tax[bracket]) / puf_tax[bracket]
                assert rel_error < tolerance_pct, \
                    f"Tax in {bracket}: synthetic=${synth_tax[bracket]/1e9:.1f}B, puf=${puf_tax[bracket]/1e9:.1f}B"

    def test_credit_takeup_rates(self, synthetic_data, puf_data):
        """Credit recipient counts and amounts should match."""
        from micro.us.synthesis.evaluation import compute_credit_totals

        credits = ['eitc', 'ctc', 'cdcc', 'savers_credit', 'education_credits']
        tolerance_pct = 0.10

        for credit in credits:
            synth_recipients, synth_total = compute_credit_totals(synthetic_data, credit)
            puf_recipients, puf_total = compute_credit_totals(puf_data, credit)

            if puf_recipients > 0:
                recipient_error = abs(synth_recipients - puf_recipients) / puf_recipients
                assert recipient_error < tolerance_pct, \
                    f"{credit} recipients: synthetic={synth_recipients/1e6:.1f}M, puf={puf_recipients/1e6:.1f}M"

            if puf_total > 0:
                total_error = abs(synth_total - puf_total) / puf_total
                assert total_error < tolerance_pct, \
                    f"{credit} total: synthetic=${synth_total/1e9:.1f}B, puf=${puf_total/1e9:.1f}B"

    def test_reform_impact_similarity(self, synthetic_data, puf_data):
        """Policy reform impacts should be similar between synthetic and PUF."""
        from micro.us.synthesis.evaluation import simulate_reform_impact

        # Example reform: increase top marginal rate by 5pp
        reform = {'top_rate_increase': 0.05}

        synth_impact = simulate_reform_impact(synthetic_data, reform)
        puf_impact = simulate_reform_impact(puf_data, reform)

        # Revenue impact should be within 10%
        tolerance_pct = 0.10
        if puf_impact['revenue_change'] > 0:
            rel_error = abs(synth_impact['revenue_change'] - puf_impact['revenue_change']) / \
                       puf_impact['revenue_change']
            assert rel_error < tolerance_pct, \
                f"Reform revenue: synthetic=${synth_impact['revenue_change']/1e9:.1f}B, " \
                f"puf=${puf_impact['revenue_change']/1e9:.1f}B"


# =============================================================================
# TEST: Comparison vs PolicyEngine Enhanced CPS
# =============================================================================

class TestVsPolicyEngineECPS:
    """Compare Cosilico synthesis vs PolicyEngine Enhanced CPS."""

    def test_correlation_preservation_vs_pe(self, cosilico_data, pe_ecps_data, puf_data, tax_vars):
        """Cosilico should have better correlation preservation than PE ECPS."""
        from micro.us.synthesis.evaluation import compute_weighted_correlation_matrix

        puf_corr = compute_weighted_correlation_matrix(puf_data[tax_vars], puf_data['weight'])
        cosilico_corr = compute_weighted_correlation_matrix(cosilico_data[tax_vars], cosilico_data['weight'])
        pe_corr = compute_weighted_correlation_matrix(pe_ecps_data[tax_vars], pe_ecps_data['weight'])

        cosilico_dist = np.linalg.norm(cosilico_corr - puf_corr, 'fro')
        pe_dist = np.linalg.norm(pe_corr - puf_corr, 'fro')

        # Cosilico should be at least as good as PE
        assert cosilico_dist <= pe_dist * 1.1, \
            f"Cosilico correlation dist {cosilico_dist:.2f} worse than PE {pe_dist:.2f}"

    def test_joint_distribution_quality_vs_pe(self, cosilico_data, pe_ecps_data, puf_data):
        """Cosilico should have better joint distribution fidelity."""
        from micro.us.synthesis.evaluation import compute_joint_distribution_score

        cosilico_score = compute_joint_distribution_score(cosilico_data, puf_data)
        pe_score = compute_joint_distribution_score(pe_ecps_data, puf_data)

        # Higher score = better joint distribution match
        assert cosilico_score >= pe_score * 0.9, \
            f"Cosilico joint score {cosilico_score:.3f} vs PE {pe_score:.3f}"

    def test_calibration_parity(self, cosilico_data, pe_ecps_data, irs_soi_targets):
        """Both should match IRS SOI calibration targets similarly."""
        from micro.us.synthesis.evaluation import compute_calibration_errors

        cosilico_errors = compute_calibration_errors(cosilico_data, irs_soi_targets)
        pe_errors = compute_calibration_errors(pe_ecps_data, irs_soi_targets)

        # Cosilico should match calibration at least as well as PE
        cosilico_max_error = max(abs(e) for e in cosilico_errors.values())
        pe_max_error = max(abs(e) for e in pe_errors.values())

        assert cosilico_max_error <= pe_max_error * 1.2, \
            f"Cosilico max calibration error {cosilico_max_error:.1%} worse than PE {pe_max_error:.1%}"


# =============================================================================
# TEST: Privacy
# =============================================================================

class TestPrivacy:
    """Ensure synthetic data doesn't leak PUF records."""

    def test_minimum_nearest_neighbor_distance(self, synthetic_data, puf_data, tax_vars):
        """No synthetic record should be too close to any PUF record."""
        from micro.us.synthesis.evaluation import compute_nearest_neighbor_distances

        # Compute distance from each synthetic record to nearest PUF record
        distances = compute_nearest_neighbor_distances(
            synthetic_data[tax_vars].values,
            puf_data[tax_vars].values
        )

        # Minimum distance should be above threshold
        # (prevents verbatim copying or trivial perturbations)
        min_distance = distances.min()
        threshold = 0.01  # 1% of feature space diameter

        assert min_distance > threshold, \
            f"Nearest neighbor distance {min_distance:.4f} below privacy threshold {threshold}"

    def test_no_exact_matches(self, synthetic_data, puf_data, tax_vars):
        """No synthetic record should exactly match a PUF record."""
        from micro.us.synthesis.evaluation import find_exact_matches

        matches = find_exact_matches(
            synthetic_data[tax_vars].values,
            puf_data[tax_vars].values,
            tolerance=1e-6
        )

        assert len(matches) == 0, f"Found {len(matches)} exact matches between synthetic and PUF"


# =============================================================================
# FIXTURES (to be implemented)
# =============================================================================

@pytest.fixture
def tax_vars():
    """List of tax variables to evaluate."""
    return [
        'wages', 'interest', 'dividends', 'capital_gains', 'business_income',
        'rental_income', 'retirement_income', 'social_security',
        'adjusted_gross_income', 'itemized_deductions', 'mortgage_interest',
        'state_taxes_paid', 'charitable_contributions',
        'eitc', 'ctc', 'retirement_contributions',
    ]


@pytest.fixture
def puf_data():
    """Load PUF microdata (excluding aggregates)."""
    # TODO: Implement actual data loading
    pytest.skip("PUF data not yet configured")


@pytest.fixture
def synthetic_data(puf_data):
    """Generate synthetic data using trained model."""
    # TODO: Implement after model is built
    pytest.skip("Synthesis model not yet implemented")


@pytest.fixture
def pe_ecps_data():
    """Load PolicyEngine Enhanced CPS for comparison."""
    # TODO: Implement PE data loading
    pytest.skip("PE ECPS data not yet configured")


@pytest.fixture
def cosilico_data():
    """Load Cosilico synthetic data."""
    # TODO: Implement after synthesis is complete
    pytest.skip("Cosilico synthesis not yet implemented")


@pytest.fixture
def irs_soi_targets():
    """Load IRS SOI calibration targets."""
    # TODO: Load from calibration module
    pytest.skip("IRS SOI targets not yet configured")
