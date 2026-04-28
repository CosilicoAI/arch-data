"""
Entropy-based calibration method.

Minimizes Kullback-Leibler divergence from original weights
while satisfying calibration constraints.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from ..constraints import Constraint


@dataclass
class EntropyCalibrator:
    """
    Entropy-based weight calibrator.

    Finds new weights that minimize KL divergence from original weights
    while matching target constraints within tolerance.

    Attributes:
        bounds: (min_ratio, max_ratio) for weight adjustments
        max_iterations: Maximum solver iterations
        convergence_tol: Convergence tolerance for solver
    """

    bounds: tuple[float, float] = (0.1, 10.0)
    max_iterations: int = 100
    convergence_tol: float = 1e-6

    def calibrate(
        self,
        original_weights: np.ndarray,
        constraints: list[Constraint],
    ) -> np.ndarray:
        """
        Compute calibrated weights.

        Args:
            original_weights: Original survey weights (n,)
            constraints: List of Constraint objects

        Returns:
            Calibrated weights (n,)

        Raises:
            ValueError: If constraints are infeasible
            RuntimeError: If optimization fails to converge
        """
        n = len(original_weights)

        # Objective function: KL divergence
        def kl_divergence(w: np.ndarray) -> float:
            """Compute KL divergence: sum(w * log(w/w0))."""
            # Add small epsilon to avoid log(0)
            w_safe = np.maximum(w, 1e-10)
            w0_safe = np.maximum(original_weights, 1e-10)
            return np.sum(w_safe * np.log(w_safe / w0_safe))

        # Gradient of KL divergence
        def kl_gradient(w: np.ndarray) -> np.ndarray:
            """Gradient of KL divergence: log(w/w0) + 1."""
            w_safe = np.maximum(w, 1e-10)
            w0_safe = np.maximum(original_weights, 1e-10)
            return np.log(w_safe / w0_safe) + 1.0

        # Build constraint functions for scipy
        scipy_constraints = []
        for constraint in constraints:
            # Equality constraint: sum(w * indicator) - target = 0
            def constraint_fn(w: np.ndarray, ind=constraint.indicator,
                            target=constraint.target_value) -> float:
                return np.dot(w, ind) - target

            # Jacobian of constraint: indicator vector
            def constraint_jac(w: np.ndarray, ind=constraint.indicator) -> np.ndarray:
                return ind

            scipy_constraints.append({
                'type': 'eq',
                'fun': constraint_fn,
                'jac': constraint_jac,
            })

        # Bounds on individual weights
        weight_bounds = [
            (original_weights[i] * self.bounds[0],
             original_weights[i] * self.bounds[1])
            for i in range(n)
        ]

        # Initial guess: original weights
        x0 = original_weights.copy()

        # Solve optimization problem
        result: OptimizeResult = minimize(
            kl_divergence,
            x0,
            method='SLSQP',
            jac=kl_gradient,
            constraints=scipy_constraints,
            bounds=weight_bounds,
            options={
                'maxiter': self.max_iterations,
                'ftol': self.convergence_tol,
                'disp': False,
            }
        )

        # Check if optimization succeeded
        if not result.success:
            if 'Positive directional derivative for linesearch' in result.message:
                # This often means constraints are infeasible
                raise ValueError(
                    f"Optimization failed: constraints may be infeasible. "
                    f"Message: {result.message}"
                )
            else:
                raise RuntimeError(
                    f"Optimization failed to converge: {result.message}"
                )

        # Verify constraints are satisfied
        calibrated_weights = result.x
        for i, constraint in enumerate(constraints):
            actual = np.dot(calibrated_weights, constraint.indicator)
            error = abs(actual - constraint.target_value) / constraint.target_value
            if error > constraint.tolerance:
                raise RuntimeError(
                    f"Constraint {i} ({constraint.variable}) not satisfied: "
                    f"target={constraint.target_value:.2f}, "
                    f"actual={actual:.2f}, "
                    f"error={error:.4%} > tolerance={constraint.tolerance:.4%}"
                )

        return calibrated_weights
