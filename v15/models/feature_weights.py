"""
Explicit Feature Weights Layer.

Unlike dense projections that mix all features, this gives each feature
its own individual learnable weight and bias.
"""
import torch
import torch.nn as nn
from typing import Optional

class ExplicitFeatureWeights(nn.Module):
    """
    Applies learnable per-feature weights and optional bias.

    Each of the input features gets its own weight, allowing the model
    to learn feature importance directly.

    Args:
        n_features: Number of input features (e.g., 8632)
        use_bias: Whether to include learnable bias per feature
        init_weights: Initial weight value (default 1.0)
        init_bias: Initial bias value (default 0.0)
    """

    def __init__(
        self,
        n_features: int,
        use_bias: bool = True,
        init_weights: float = 1.0,
        init_bias: float = 0.0
    ):
        super().__init__()
        self.n_features = n_features
        self.use_bias = use_bias

        # Each feature gets its own learnable weight
        self.weights = nn.Parameter(torch.full((n_features,), init_weights))

        if use_bias:
            self.bias = nn.Parameter(torch.full((n_features,), init_bias))
        else:
            self.register_buffer('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply per-feature weights.

        Args:
            x: Input tensor [batch_size, n_features]

        Returns:
            Weighted features [batch_size, n_features]
        """
        if x.size(-1) != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {x.size(-1)}"
            )

        # Element-wise multiplication with weights
        out = x * self.weights

        if self.use_bias and self.bias is not None:
            out = out + self.bias

        return out

    def get_feature_importance(self) -> torch.Tensor:
        """Get absolute weight values as importance scores."""
        return torch.abs(self.weights.detach())

    def get_top_features(self, k: int = 100) -> torch.Tensor:
        """Get indices of top-k most important features."""
        importance = self.get_feature_importance()
        return torch.topk(importance, k).indices


class FeatureGating(nn.Module):
    """
    Learnable feature gating with sigmoid activation.

    Learns to gate (0-1) each feature, allowing complete suppression
    of irrelevant features.
    """

    def __init__(self, n_features: int, init_value: float = 0.0):
        super().__init__()
        # Initialize gates to sigmoid(0) = 0.5
        self.gate_logits = nn.Parameter(torch.full((n_features,), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_logits)
        return x * gates

    def get_active_features(self, threshold: float = 0.5) -> torch.Tensor:
        """Get indices of features with gate > threshold."""
        gates = torch.sigmoid(self.gate_logits.detach())
        return torch.where(gates > threshold)[0]
