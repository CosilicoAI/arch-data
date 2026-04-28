"""
Validation framework for synthetic microdata.

Compares Cosilico synthesis against:
1. PolicyEngine Enhanced CPS (baseline)
2. IRS SOI targets (ground truth)
3. PUF (if available, for joint distribution)

Produces structured metrics that can drive iterative improvement.

Architecture follows the rules engine pattern:
- Validators produce metrics
- Metrics feed into synthesis improvements
- Dashboard shows progress over time
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import json
from datetime import datetime


@dataclass
class ValidationMetric:
    """A single validation metric."""
    name: str
    category: str  # marginal, joint, policy, privacy, calibration
    value: float
    target: Optional[float] = None  # Expected/ideal value
    threshold: Optional[float] = None  # Pass/fail threshold
    direction: str = "lower"  # "lower" or "higher" is better
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_passing(self) -> Optional[bool]:
        """Check if metric passes threshold."""
        if self.threshold is None:
            return None
        if self.direction == "lower":
            return self.value <= self.threshold
        return self.value >= self.threshold

    @property
    def distance_to_target(self) -> Optional[float]:
        """Distance from target (0 = perfect)."""
        if self.target is None:
            return None
        return abs(self.value - self.target)


@dataclass
class ValidationResult:
    """Complete validation results for a synthesis run."""
    timestamp: str
    synthesis_version: str
    metrics: List[ValidationMetric]
    comparison_baseline: str  # "pe_ecps" or "puf"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """Summarize validation results."""
        by_category = {}
        for m in self.metrics:
            if m.category not in by_category:
                by_category[m.category] = []
            by_category[m.category].append(m)

        return {
            "timestamp": self.timestamp,
            "version": self.synthesis_version,
            "n_metrics": len(self.metrics),
            "passing": sum(1 for m in self.metrics if m.is_passing is True),
            "failing": sum(1 for m in self.metrics if m.is_passing is False),
            "by_category": {
                cat: {
                    "count": len(metrics),
                    "mean_value": np.mean([m.value for m in metrics]),
                    "passing": sum(1 for m in metrics if m.is_passing is True),
                }
                for cat, metrics in by_category.items()
            }
        }

    def to_json(self) -> str:
        """Serialize to JSON for dashboard."""
        return json.dumps({
            "timestamp": self.timestamp,
            "synthesis_version": self.synthesis_version,
            "comparison_baseline": self.comparison_baseline,
            "metadata": self.metadata,
            "metrics": [
                {
                    "name": m.name,
                    "category": m.category,
                    "value": m.value,
                    "target": m.target,
                    "threshold": m.threshold,
                    "direction": m.direction,
                    "is_passing": m.is_passing,
                    "details": m.details,
                }
                for m in self.metrics
            ],
            "summary": self.summary()
        }, indent=2)


class SynthesisValidator:
    """
    Validates synthetic microdata quality.

    Produces metrics that can drive iterative improvement.
    """

    def __init__(
        self,
        irs_soi_targets: Optional[pd.DataFrame] = None,
        pe_ecps: Optional[pd.DataFrame] = None,
        puf: Optional[pd.DataFrame] = None,
    ):
        self.irs_soi_targets = irs_soi_targets
        self.pe_ecps = pe_ecps
        self.puf = puf

        # Tax variables to validate
        self.continuous_vars = [
            'wages', 'interest', 'dividends', 'capital_gains',
            'business_income', 'rental_income', 'social_security',
            'adjusted_gross_income', 'itemized_deductions',
            'mortgage_interest', 'state_taxes_paid', 'charitable',
        ]

        self.discrete_vars = [
            'is_itemizer', 'has_business', 'has_capital_gains',
        ]

        self.credit_vars = [
            'eitc', 'ctc', 'cdcc', 'education_credits',
        ]

    def validate(
        self,
        synthetic: pd.DataFrame,
        version: str = "dev",
    ) -> ValidationResult:
        """
        Run all validations on synthetic data.

        Returns structured metrics for dashboard and improvement.
        """
        metrics = []

        # 1. Marginal fidelity (vs PUF or PE ECPS)
        if self.puf is not None:
            metrics.extend(self._validate_marginals(synthetic, self.puf, "puf"))
            baseline = "puf"
        elif self.pe_ecps is not None:
            metrics.extend(self._validate_marginals(synthetic, self.pe_ecps, "pe_ecps"))
            baseline = "pe_ecps"
        else:
            baseline = "none"

        # 2. Joint distribution (correlations)
        if self.puf is not None:
            metrics.extend(self._validate_joints(synthetic, self.puf, "puf"))
        elif self.pe_ecps is not None:
            metrics.extend(self._validate_joints(synthetic, self.pe_ecps, "pe_ecps"))

        # 3. Calibration (vs IRS SOI)
        if self.irs_soi_targets is not None:
            metrics.extend(self._validate_calibration(synthetic))

        # 4. Policy utility
        if self.pe_ecps is not None:
            metrics.extend(self._validate_policy_utility(synthetic, self.pe_ecps))

        # 5. Privacy (if PUF available)
        if self.puf is not None:
            metrics.extend(self._validate_privacy(synthetic, self.puf))

        # 6. Comparison to PE ECPS (if available)
        if self.pe_ecps is not None and self.puf is not None:
            metrics.extend(self._compare_to_pe(synthetic, self.pe_ecps, self.puf))

        return ValidationResult(
            timestamp=datetime.now().isoformat(),
            synthesis_version=version,
            metrics=metrics,
            comparison_baseline=baseline,
            metadata={
                "n_synthetic_records": len(synthetic),
                "n_puf_records": len(self.puf) if self.puf is not None else None,
                "n_pe_records": len(self.pe_ecps) if self.pe_ecps is not None else None,
            }
        )

    def _validate_marginals(
        self,
        synthetic: pd.DataFrame,
        reference: pd.DataFrame,
        ref_name: str,
    ) -> List[ValidationMetric]:
        """Validate marginal distributions."""
        metrics = []

        for var in self.continuous_vars:
            if var not in synthetic.columns or var not in reference.columns:
                continue

            # KL divergence (binned approximation)
            kl = self._compute_kl_divergence(
                synthetic[var], reference[var],
                synthetic.get('weight', None), reference.get('weight', None)
            )

            metrics.append(ValidationMetric(
                name=f"kl_{var}",
                category="marginal",
                value=kl,
                target=0.0,
                threshold=0.1,  # KL < 0.1 is good
                direction="lower",
                details={"variable": var, "reference": ref_name}
            ))

            # Zero fraction difference
            zero_diff = self._compute_zero_fraction_diff(
                synthetic[var], reference[var],
                synthetic.get('weight', None), reference.get('weight', None)
            )

            metrics.append(ValidationMetric(
                name=f"zero_frac_diff_{var}",
                category="marginal",
                value=zero_diff,
                target=0.0,
                threshold=0.05,  # Within 5pp
                direction="lower",
                details={"variable": var, "reference": ref_name}
            ))

            # Quantile errors (median, 90th, 99th)
            for q in [0.5, 0.9, 0.99]:
                q_error = self._compute_quantile_error(
                    synthetic[var], reference[var],
                    synthetic.get('weight', None), reference.get('weight', None),
                    q
                )

                metrics.append(ValidationMetric(
                    name=f"q{int(q*100)}_{var}",
                    category="marginal",
                    value=q_error,
                    target=0.0,
                    threshold=0.15,  # Within 15%
                    direction="lower",
                    details={"variable": var, "quantile": q, "reference": ref_name}
                ))

        return metrics

    def _validate_joints(
        self,
        synthetic: pd.DataFrame,
        reference: pd.DataFrame,
        ref_name: str,
    ) -> List[ValidationMetric]:
        """Validate joint distributions (correlations)."""
        metrics = []

        # Key economic correlations to preserve
        correlation_pairs = [
            ('wages', 'adjusted_gross_income'),
            ('wages', 'state_taxes_paid'),
            ('capital_gains', 'dividends'),
            ('adjusted_gross_income', 'itemized_deductions'),
            ('business_income', 'adjusted_gross_income'),
        ]

        for var1, var2 in correlation_pairs:
            if var1 not in synthetic.columns or var2 not in synthetic.columns:
                continue
            if var1 not in reference.columns or var2 not in reference.columns:
                continue

            synth_corr = self._weighted_correlation(
                synthetic[var1], synthetic[var2], synthetic.get('weight', None)
            )
            ref_corr = self._weighted_correlation(
                reference[var1], reference[var2], reference.get('weight', None)
            )

            corr_diff = abs(synth_corr - ref_corr)

            metrics.append(ValidationMetric(
                name=f"corr_{var1}_{var2}",
                category="joint",
                value=corr_diff,
                target=0.0,
                threshold=0.1,  # Within 0.1 correlation units
                direction="lower",
                details={
                    "var1": var1,
                    "var2": var2,
                    "synthetic_corr": synth_corr,
                    "reference_corr": ref_corr,
                    "reference": ref_name,
                }
            ))

        # Correlation matrix Frobenius distance
        available_vars = [v for v in self.continuous_vars
                        if v in synthetic.columns and v in reference.columns]

        if len(available_vars) >= 3:
            synth_corr_mat = synthetic[available_vars].corr().values
            ref_corr_mat = reference[available_vars].corr().values

            frobenius = np.linalg.norm(synth_corr_mat - ref_corr_mat, 'fro')

            metrics.append(ValidationMetric(
                name="correlation_matrix_distance",
                category="joint",
                value=frobenius,
                target=0.0,
                threshold=2.0,
                direction="lower",
                details={"n_variables": len(available_vars), "reference": ref_name}
            ))

        return metrics

    def _validate_calibration(self, synthetic: pd.DataFrame) -> List[ValidationMetric]:
        """Validate against IRS SOI calibration targets."""
        metrics = []

        if self.irs_soi_targets is None:
            return metrics

        for _, target in self.irs_soi_targets.iterrows():
            target_name = target['name']
            target_value = target['target']

            # Compute estimated value from synthetic data
            estimated = self._compute_target_estimate(synthetic, target)

            if estimated is not None and target_value != 0:
                rel_error = abs(estimated - target_value) / abs(target_value)

                metrics.append(ValidationMetric(
                    name=f"calibration_{target_name}",
                    category="calibration",
                    value=rel_error,
                    target=0.0,
                    threshold=0.05,  # Within 5%
                    direction="lower",
                    details={
                        "target_name": target_name,
                        "target_value": target_value,
                        "estimated_value": estimated,
                    }
                ))

        return metrics

    def _validate_policy_utility(
        self,
        synthetic: pd.DataFrame,
        reference: pd.DataFrame,
    ) -> List[ValidationMetric]:
        """Validate policy simulation utility."""
        metrics = []

        # Compare credit totals
        for credit in self.credit_vars:
            if credit not in synthetic.columns or credit not in reference.columns:
                continue

            synth_total = (synthetic[credit] * synthetic.get('weight', 1)).sum()
            ref_total = (reference[credit] * reference.get('weight', 1)).sum()

            if ref_total != 0:
                rel_diff = abs(synth_total - ref_total) / abs(ref_total)

                metrics.append(ValidationMetric(
                    name=f"total_{credit}",
                    category="policy",
                    value=rel_diff,
                    target=0.0,
                    threshold=0.10,  # Within 10%
                    direction="lower",
                    details={
                        "credit": credit,
                        "synthetic_total": synth_total,
                        "reference_total": ref_total,
                    }
                ))

        return metrics

    def _validate_privacy(
        self,
        synthetic: pd.DataFrame,
        puf: pd.DataFrame,
    ) -> List[ValidationMetric]:
        """Validate privacy (no PUF records leaked)."""
        metrics = []

        # Check nearest neighbor distances
        available_vars = [v for v in self.continuous_vars
                        if v in synthetic.columns and v in puf.columns]

        if len(available_vars) >= 3:
            # Sample for efficiency
            synth_sample = synthetic[available_vars].sample(min(1000, len(synthetic)))
            puf_sample = puf[available_vars].sample(min(1000, len(puf)))

            # Standardize
            mean = puf_sample.mean()
            std = puf_sample.std() + 1e-8
            synth_norm = (synth_sample - mean) / std
            puf_norm = (puf_sample - mean) / std

            # Compute pairwise distances (expensive, use sample)
            from scipy.spatial.distance import cdist
            distances = cdist(synth_norm.values[:100], puf_norm.values[:100])
            min_distances = distances.min(axis=1)

            metrics.append(ValidationMetric(
                name="min_nn_distance",
                category="privacy",
                value=min_distances.min(),
                target=None,
                threshold=0.05,  # Should be > 0.05 normalized distance
                direction="higher",
                details={
                    "mean_min_distance": min_distances.mean(),
                    "n_vars": len(available_vars),
                }
            ))

        return metrics

    def _compare_to_pe(
        self,
        synthetic: pd.DataFrame,
        pe_ecps: pd.DataFrame,
        puf: pd.DataFrame,
    ) -> List[ValidationMetric]:
        """Compare Cosilico synthesis to PE ECPS using PUF as ground truth."""
        metrics = []

        available_vars = [v for v in self.continuous_vars
                        if v in synthetic.columns and v in pe_ecps.columns and v in puf.columns]

        if len(available_vars) >= 3:
            # Correlation matrix distances
            puf_corr = puf[available_vars].corr().values
            synth_corr = synthetic[available_vars].corr().values
            pe_corr = pe_ecps[available_vars].corr().values

            cosilico_dist = np.linalg.norm(synth_corr - puf_corr, 'fro')
            pe_dist = np.linalg.norm(pe_corr - puf_corr, 'fro')

            # Improvement ratio (< 1 means Cosilico is better)
            improvement = cosilico_dist / (pe_dist + 1e-8)

            metrics.append(ValidationMetric(
                name="correlation_vs_pe",
                category="comparison",
                value=improvement,
                target=1.0,
                threshold=1.0,  # Should be <= 1.0 (at least as good as PE)
                direction="lower",
                details={
                    "cosilico_distance": cosilico_dist,
                    "pe_distance": pe_dist,
                    "n_vars": len(available_vars),
                }
            ))

        return metrics

    # Helper methods

    def _compute_kl_divergence(
        self,
        p: pd.Series,
        q: pd.Series,
        weights_p: Optional[pd.Series] = None,
        weights_q: Optional[pd.Series] = None,
        n_bins: int = 50,
    ) -> float:
        """Compute KL divergence using histogram approximation."""
        # Determine bin edges from combined data
        all_data = pd.concat([p, q])
        bins = np.histogram_bin_edges(all_data[all_data > 0], bins=n_bins)

        # Add zero bin
        bins = np.concatenate([[-np.inf, 0], bins])

        # Compute histograms
        if weights_p is not None:
            hist_p, _ = np.histogram(p, bins=bins, weights=weights_p)
        else:
            hist_p, _ = np.histogram(p, bins=bins)

        if weights_q is not None:
            hist_q, _ = np.histogram(q, bins=bins, weights=weights_q)
        else:
            hist_q, _ = np.histogram(q, bins=bins)

        # Normalize
        hist_p = hist_p / hist_p.sum() + 1e-10
        hist_q = hist_q / hist_q.sum() + 1e-10

        # KL divergence
        return np.sum(hist_p * np.log(hist_p / hist_q))

    def _compute_zero_fraction_diff(
        self,
        p: pd.Series,
        q: pd.Series,
        weights_p: Optional[pd.Series] = None,
        weights_q: Optional[pd.Series] = None,
    ) -> float:
        """Compute difference in zero fractions."""
        if weights_p is not None:
            p_zero = ((p == 0) * weights_p).sum() / weights_p.sum()
        else:
            p_zero = (p == 0).mean()

        if weights_q is not None:
            q_zero = ((q == 0) * weights_q).sum() / weights_q.sum()
        else:
            q_zero = (q == 0).mean()

        return abs(p_zero - q_zero)

    def _compute_quantile_error(
        self,
        p: pd.Series,
        q: pd.Series,
        weights_p: Optional[pd.Series] = None,
        weights_q: Optional[pd.Series] = None,
        quantile: float = 0.5,
    ) -> float:
        """Compute relative error at a given quantile."""
        # Simple unweighted for now
        p_q = p.quantile(quantile)
        q_q = q.quantile(quantile)

        if q_q == 0:
            return 0 if p_q == 0 else 1.0

        return abs(p_q - q_q) / abs(q_q)

    def _weighted_correlation(
        self,
        x: pd.Series,
        y: pd.Series,
        weights: Optional[pd.Series] = None,
    ) -> float:
        """Compute weighted Pearson correlation."""
        if weights is None:
            return x.corr(y)

        # Weighted correlation
        mask = ~(x.isna() | y.isna())
        x, y, w = x[mask], y[mask], weights[mask]

        w_sum = w.sum()
        mean_x = (x * w).sum() / w_sum
        mean_y = (y * w).sum() / w_sum

        cov = ((x - mean_x) * (y - mean_y) * w).sum() / w_sum
        std_x = np.sqrt(((x - mean_x) ** 2 * w).sum() / w_sum)
        std_y = np.sqrt(((y - mean_y) ** 2 * w).sum() / w_sum)

        if std_x == 0 or std_y == 0:
            return 0.0

        return cov / (std_x * std_y)

    def _compute_target_estimate(
        self,
        df: pd.DataFrame,
        target: pd.Series,
    ) -> Optional[float]:
        """Compute estimated value for a calibration target."""
        # This would need to be implemented based on target structure
        # For now, return None
        return None


def run_validation_dashboard(
    synthetic_path: str,
    output_path: str,
    pe_ecps_path: Optional[str] = None,
    puf_path: Optional[str] = None,
    irs_targets_path: Optional[str] = None,
):
    """
    Run validation and output JSON for dashboard.

    This can be called from CI or manually to track synthesis quality over time.
    """
    # Load data
    synthetic = pd.read_parquet(synthetic_path)

    pe_ecps = pd.read_parquet(pe_ecps_path) if pe_ecps_path else None
    puf = pd.read_parquet(puf_path) if puf_path else None
    irs_targets = pd.read_csv(irs_targets_path) if irs_targets_path else None

    # Run validation
    validator = SynthesisValidator(
        irs_soi_targets=irs_targets,
        pe_ecps=pe_ecps,
        puf=puf,
    )

    result = validator.validate(synthetic)

    # Write results
    with open(output_path, 'w') as f:
        f.write(result.to_json())

    print(f"Validation complete. Results written to {output_path}")
    print(f"Summary: {result.summary()}")

    return result
