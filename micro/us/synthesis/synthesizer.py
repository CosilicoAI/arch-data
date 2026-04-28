"""
TaxSynthesizer: Main class for synthesizing tax variables.

Pipeline:
1. Load PUF microdata (training)
2. Fit transforms on PUF
3. Train discrete model for binary/categorical vars
4. Train normalizing flow for continuous vars
5. Generate synthetic tax variables for CPS demographics
6. Validate synthetic data quality
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .transforms import MultiVariableTransformer
from .flows import ConditionalMAF
from .discrete import DiscreteModelCollection


@dataclass
class TaxSynthesizerConfig:
    """Configuration for TaxSynthesizer."""

    continuous_vars: List[str]
    discrete_vars: List[str]
    demographic_vars: List[str]

    # Model architecture
    flow_layers: int = 6
    hidden_dim: int = 64

    # Training
    batch_size: int = 256
    learning_rate: float = 1e-3


class TaxSynthesizer:
    """
    Synthesize tax variables conditioned on demographics.

    Uses normalizing flows for continuous variables and separate
    classifiers for discrete variables.
    """

    def __init__(
        self,
        continuous_vars: Optional[List[str]] = None,
        discrete_vars: Optional[List[str]] = None,
        demographic_vars: Optional[List[str]] = None,
        flow_layers: int = 6,
        hidden_dim: int = 64,
    ):
        """
        Initialize synthesizer.

        Args:
            continuous_vars: List of continuous tax variables to synthesize
            discrete_vars: List of binary/discrete tax variables
            demographic_vars: List of demographic variables to condition on
            flow_layers: Number of layers in normalizing flow
            hidden_dim: Hidden layer size
        """
        # Default variable lists
        self.continuous_vars = continuous_vars or [
            "wages",
            "interest",
            "dividends",
            "capital_gains",
            "business_income",
            "pension_income",
            "social_security",
            "other_income",
        ]

        self.discrete_vars = discrete_vars or [
            "is_itemizer",
            "has_business",
        ]

        self.demographic_vars = demographic_vars or [
            "age",
            "filing_status",
            "n_dependents",
        ]

        # Model parameters
        self.flow_layers = flow_layers
        self.hidden_dim = hidden_dim

        # Will be set during fit
        self.transformer_: Optional[MultiVariableTransformer] = None
        self.flow_model_: Optional[ConditionalMAF] = None
        self.discrete_model_: Optional[DiscreteModelCollection] = None
        self.is_fitted_: bool = False
        self.training_loss_: List[float] = []

    def fit(
        self,
        data: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        verbose: bool = True,
    ) -> "TaxSynthesizer":
        """
        Fit synthesizer on PUF data.

        Uses a two-stage approach:
        1. Binary models predict P(positive | demographics) for each variable
        2. Normalizing flow learns P(log_value | demographics, is_positive)

        Args:
            data: DataFrame with tax and demographic variables
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Optimizer learning rate
            verbose: Whether to print progress

        Returns:
            self
        """
        # Prepare data dict for transforms
        data_dict = {col: data[col].values for col in data.columns}

        # Fit transforms on continuous variables
        self.transformer_ = MultiVariableTransformer(self.continuous_vars)
        self.transformer_.fit(data_dict)

        # Transform continuous variables
        transformed = self.transformer_.transform(data_dict)

        # Prepare tensors
        n_context = len(self.demographic_vars)
        n_features = len(self.continuous_vars)

        # Context: demographic variables
        context_np = np.column_stack([
            data[var].values for var in self.demographic_vars
        ])
        context = torch.tensor(context_np, dtype=torch.float32)

        # Features: transformed continuous variables
        # Replace NaN (zeros) with 0 for training, we'll mask them
        features_list = []
        for var in self.continuous_vars:
            vals = transformed[var].copy()
            vals = np.nan_to_num(vals, nan=0.0)  # Replace NaN with 0
            features_list.append(vals)
        features_np = np.column_stack(features_list)
        features = torch.tensor(features_np, dtype=torch.float32)

        # Create mask for positive values (to weight training properly)
        positive_mask = torch.ones_like(features)
        for i, var in enumerate(self.continuous_vars):
            is_positive = (data[var].values > 0).astype(np.float32)
            positive_mask[:, i] = torch.tensor(is_positive)

        # Weights
        weights = torch.tensor(data["weight"].values, dtype=torch.float32)

        # Create normalizing flow
        self.flow_model_ = ConditionalMAF(
            n_features=n_features,
            n_context=n_context,
            n_layers=self.flow_layers,
            hidden_dim=self.hidden_dim,
        )

        # Train flow (with masking for zero values)
        self._train_flow(
            features, context, weights, positive_mask,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            verbose=verbose,
        )

        # Train zero/non-zero indicator models for each continuous variable
        self._train_zero_indicators(
            data, context,
            epochs=epochs // 2,
            learning_rate=learning_rate,
        )

        # Train discrete models if we have discrete variables
        if self.discrete_vars:
            self._train_discrete(
                data, context,
                epochs=epochs // 2,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )

        self.is_fitted_ = True
        return self

    def _train_flow(
        self,
        features: torch.Tensor,
        context: torch.Tensor,
        weights: torch.Tensor,
        positive_mask: torch.Tensor,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        verbose: bool,
    ):
        """Train the normalizing flow model.

        Uses only rows where ALL variables are positive to ensure
        the flow learns on clean data.
        """
        optimizer = torch.optim.Adam(
            self.flow_model_.parameters(), lr=learning_rate
        )

        # Only train on rows where all continuous variables are positive
        all_positive = positive_mask.all(dim=1)

        if all_positive.sum() < 10:
            # If very few rows have all positives, train on all data
            # (not ideal but prevents training failure)
            train_features = features
            train_context = context
            train_weights = weights
        else:
            train_features = features[all_positive]
            train_context = context[all_positive]
            train_weights = weights[all_positive]

        dataset = TensorDataset(train_features, train_context, train_weights)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.training_loss_ = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_features, batch_context, batch_weights in loader:
                optimizer.zero_grad()

                # Compute negative log likelihood
                log_prob = self.flow_model_.log_prob(batch_features, batch_context)

                # Weighted loss
                loss = -(log_prob * batch_weights).sum() / batch_weights.sum()

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.training_loss_.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    def _train_zero_indicators(
        self,
        data: pd.DataFrame,
        context: torch.Tensor,
        epochs: int,
        learning_rate: float,
    ):
        """Train binary models for zero/non-zero indicators."""
        from .discrete import BinaryVariableModel

        self.zero_indicators_ = nn.ModuleDict()

        for var in self.continuous_vars:
            # Create binary model
            model = BinaryVariableModel(
                n_context=len(self.demographic_vars),
                hidden_dim=self.hidden_dim // 2,
            )

            # Target: 1 if positive, 0 if zero
            target = torch.tensor(
                (data[var].values > 0).astype(np.float32)
            ).unsqueeze(-1)

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            for _ in range(epochs):
                optimizer.zero_grad()

                pred = model(context)
                loss = nn.functional.binary_cross_entropy(pred, target)

                loss.backward()
                optimizer.step()

            self.zero_indicators_[var] = model

    def _train_discrete(
        self,
        data: pd.DataFrame,
        context: torch.Tensor,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ):
        """Train discrete variable models."""
        # Identify binary vs categorical
        binary_vars = []
        categorical_vars = {}

        for var in self.discrete_vars:
            unique_vals = data[var].nunique()
            if unique_vals == 2:
                binary_vars.append(var)
            else:
                categorical_vars[var] = unique_vals

        self.discrete_model_ = DiscreteModelCollection(
            n_context=len(self.demographic_vars),
            binary_vars=binary_vars,
            categorical_vars=categorical_vars,
            hidden_dim=self.hidden_dim // 2,
        )

        # Prepare targets
        targets = {
            var: torch.tensor(data[var].values, dtype=torch.long)
            for var in self.discrete_vars
        }

        optimizer = torch.optim.Adam(
            self.discrete_model_.parameters(), lr=learning_rate
        )

        for epoch in range(epochs):
            optimizer.zero_grad()

            log_prob = self.discrete_model_.log_prob(context, targets)
            loss = -log_prob.mean()

            loss.backward()
            optimizer.step()

    def generate(
        self,
        demographics: pd.DataFrame,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic tax variables for given demographics.

        Two-stage generation:
        1. Sample zero/non-zero indicators for each continuous variable
        2. For non-zero cases, sample from flow and inverse transform

        Args:
            demographics: DataFrame with demographic variables
            seed: Random seed for reproducibility

        Returns:
            DataFrame with demographics + synthetic tax variables
        """
        if not self.is_fitted_:
            raise ValueError("Synthesizer not fitted. Call fit() first.")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Prepare context tensor
        context_np = np.column_stack([
            demographics[var].values for var in self.demographic_vars
        ])
        context = torch.tensor(context_np, dtype=torch.float32)

        # Sample from flow (will be masked by zero indicators)
        with torch.no_grad():
            samples = self.flow_model_.sample(context)

        # Convert to numpy
        samples_np = samples.numpy()

        # Create dict with transformed values (these are in log/standardized space)
        transformed_dict = {
            var: samples_np[:, i]
            for i, var in enumerate(self.continuous_vars)
        }

        # Inverse transform to original scale
        original_dict = self.transformer_.inverse_transform(transformed_dict)

        # Apply zero indicators: sample which values should be zero
        with torch.no_grad():
            for var in self.continuous_vars:
                if hasattr(self, 'zero_indicators_') and var in self.zero_indicators_:
                    # Get probability of being positive
                    prob_positive = self.zero_indicators_[var](context).squeeze(-1)

                    # Sample binary indicator
                    is_positive = torch.bernoulli(prob_positive).numpy()

                    # Zero out values where indicator is 0
                    original_dict[var] = np.where(
                        is_positive > 0.5,
                        original_dict[var],
                        0.0
                    )

        # Ensure non-negative values
        for var in self.continuous_vars:
            original_dict[var] = np.maximum(original_dict[var], 0)

        # Sample discrete variables if we have them
        if self.discrete_model_ is not None:
            with torch.no_grad():
                discrete_samples = self.discrete_model_.sample(context)

            for var in self.discrete_vars:
                original_dict[var] = discrete_samples[var].numpy().flatten()

        # Build result DataFrame
        result = demographics.copy()

        for var in self.continuous_vars:
            result[var] = original_dict[var]

        if self.discrete_model_ is not None:
            for var in self.discrete_vars:
                result[var] = original_dict[var]

        return result

    def save(self, path: Union[str, Path]) -> None:
        """
        Save fitted model to disk.

        Args:
            path: Path to save model
        """
        if not self.is_fitted_:
            raise ValueError("Synthesizer not fitted. Call fit() first.")

        path = Path(path)

        state = {
            "continuous_vars": self.continuous_vars,
            "discrete_vars": self.discrete_vars,
            "demographic_vars": self.demographic_vars,
            "flow_layers": self.flow_layers,
            "hidden_dim": self.hidden_dim,
            "transformer": self.transformer_,
            "flow_state_dict": self.flow_model_.state_dict(),
            "discrete_state_dict": (
                self.discrete_model_.state_dict()
                if self.discrete_model_ else None
            ),
            "zero_indicators_state_dict": (
                self.zero_indicators_.state_dict()
                if hasattr(self, 'zero_indicators_') else None
            ),
            "training_loss": self.training_loss_,
        }

        torch.save(state, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TaxSynthesizer":
        """
        Load fitted model from disk.

        Args:
            path: Path to saved model

        Returns:
            Loaded synthesizer
        """
        path = Path(path)
        state = torch.load(path, weights_only=False)

        synth = cls(
            continuous_vars=state["continuous_vars"],
            discrete_vars=state["discrete_vars"],
            demographic_vars=state["demographic_vars"],
            flow_layers=state["flow_layers"],
            hidden_dim=state["hidden_dim"],
        )

        synth.transformer_ = state["transformer"]
        synth.training_loss_ = state["training_loss"]

        # Reconstruct flow model
        n_features = len(state["continuous_vars"])
        n_context = len(state["demographic_vars"])

        synth.flow_model_ = ConditionalMAF(
            n_features=n_features,
            n_context=n_context,
            n_layers=state["flow_layers"],
            hidden_dim=state["hidden_dim"],
        )
        synth.flow_model_.load_state_dict(state["flow_state_dict"])

        # Reconstruct discrete model if present
        if state["discrete_state_dict"] is not None:
            # We need to infer the structure from the state dict
            # For now, assume binary vars only (simplification)
            synth.discrete_model_ = DiscreteModelCollection(
                n_context=n_context,
                binary_vars=state["discrete_vars"],
                categorical_vars={},
                hidden_dim=state["hidden_dim"] // 2,
            )
            synth.discrete_model_.load_state_dict(state["discrete_state_dict"])

        # Reconstruct zero indicators if present
        if state.get("zero_indicators_state_dict") is not None:
            from .discrete import BinaryVariableModel

            synth.zero_indicators_ = nn.ModuleDict()
            for var in state["continuous_vars"]:
                model = BinaryVariableModel(
                    n_context=n_context,
                    hidden_dim=state["hidden_dim"] // 2,
                )
                synth.zero_indicators_[var] = model

            synth.zero_indicators_.load_state_dict(state["zero_indicators_state_dict"])

        synth.is_fitted_ = True
        return synth
