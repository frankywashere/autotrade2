"""
Full V15 Channel Prediction Model.

Takes all 8,632+ features, applies explicit weights, encodes per-TF,
applies cross-TF attention, and produces predictions.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .feature_weights import ExplicitFeatureWeights, FeatureGating
from .tf_encoder import MultiTFEncoder
from .cross_tf_attention import CrossTFAttention, TFAggregator
from .prediction_heads import PredictionHeads
from ..config import (
    TOTAL_FEATURES, N_TIMEFRAMES, FEATURES_PER_TF,
    FEATURE_COUNTS, MODEL_CONFIG
)
from ..exceptions import ModelError


class V15Model(nn.Module):
    """
    Complete V15 Channel Prediction Model.

    Architecture:
        Input (8,665 features)
            ↓
        Feature Validation (check for NaN/Inf)
            ↓
        Explicit Feature Weights (8,665 learnable weights)
            ↓
        Feature Gating (optional, learns to suppress features)
            ↓
        Split into TF features (11 x 782) + Shared features (63)
            ↓
        Per-TF Encoders (11 encoders → 11 x 128 embeddings)
            ↓
        Cross-TF Attention (learns TF relationships)
            ↓
        TF Aggregator (11 x 128 → 256)
            ↓
        Prediction Heads (duration, direction, new_channel, confidence)
    """

    def __init__(
        self,
        input_dim: int = TOTAL_FEATURES,
        n_timeframes: int = N_TIMEFRAMES,
        features_per_tf: int = FEATURES_PER_TF,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        n_attention_heads: int = 8,
        dropout: float = 0.1,
        use_explicit_weights: bool = True,
        use_gating: bool = False,
        share_tf_weights: bool = False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_timeframes = n_timeframes
        self.features_per_tf = features_per_tf

        # Shared features = events + bar metadata
        self.shared_features_dim = (
            FEATURE_COUNTS['events_total'] +
            FEATURE_COUNTS['bar_metadata_per_tf'] * n_timeframes
        )

        # Validate dimensions
        expected_dim = features_per_tf * n_timeframes + self.shared_features_dim
        if input_dim != expected_dim:
            raise ModelError(
                f"Input dim mismatch: got {input_dim}, expected {expected_dim} "
                f"({features_per_tf} * {n_timeframes} + {self.shared_features_dim})"
            )

        # 1. Explicit Feature Weights
        if use_explicit_weights:
            self.feature_weights = ExplicitFeatureWeights(input_dim)
        else:
            self.feature_weights = None

        # 2. Optional Feature Gating
        if use_gating:
            self.feature_gating = FeatureGating(input_dim)
        else:
            self.feature_gating = None

        # 3. Per-TF Encoders
        self.tf_encoder = MultiTFEncoder(
            n_timeframes=n_timeframes,
            features_per_tf=features_per_tf,
            shared_features=self.shared_features_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            share_weights=share_tf_weights,
            dropout=dropout
        )

        # 4. Cross-TF Attention
        self.cross_tf_attention = CrossTFAttention(
            embed_dim=embed_dim,
            n_heads=n_attention_heads,
            dropout=dropout
        )

        # 5. TF Aggregator
        self.tf_aggregator = TFAggregator(
            embed_dim=embed_dim,
            n_timeframes=n_timeframes,
            strategy='attention',
            output_dim=hidden_dim
        )

        # 6. Prediction Heads
        self.prediction_heads = PredictionHeads(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2
        )

    def validate_input(self, x: torch.Tensor) -> None:
        """Check for NaN/Inf in input - LOUD failure."""
        if torch.isnan(x).any():
            nan_count = torch.isnan(x).sum().item()
            raise ModelError(f"Input contains {nan_count} NaN values")
        if torch.isinf(x).any():
            inf_count = torch.isinf(x).sum().item()
            raise ModelError(f"Input contains {inf_count} Inf values")

    def split_features(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split input into per-TF features and shared features.

        Args:
            x: [batch, input_dim]

        Returns:
            tf_features: [batch, n_timeframes, features_per_tf]
            shared_features: [batch, shared_features_dim]
        """
        batch_size = x.size(0)

        # TF features are first (n_tf * features_per_tf)
        tf_dim = self.n_timeframes * self.features_per_tf
        tf_flat = x[:, :tf_dim]
        tf_features = tf_flat.view(batch_size, self.n_timeframes, self.features_per_tf)

        # Shared features are last
        shared_features = x[:, tf_dim:]

        return tf_features, shared_features

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        validate: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [batch, input_dim] raw features
            return_attention: If True, include attention weights in output
            validate: If True, check for NaN/Inf (LOUD failure)

        Returns:
            Dict with predictions and optional attention weights
        """
        # 1. Validate input
        if validate:
            self.validate_input(x)

        # 2. Apply explicit feature weights
        if self.feature_weights is not None:
            x = self.feature_weights(x)

        # 3. Apply feature gating
        if self.feature_gating is not None:
            x = self.feature_gating(x)

        # 4. Split into TF and shared features
        tf_features, shared_features = self.split_features(x)

        # 5. Encode per-TF
        tf_embeddings = self.tf_encoder(tf_features, shared_features)

        # 6. Cross-TF attention
        tf_embeddings, attn_weights = self.cross_tf_attention(
            tf_embeddings, return_attention=return_attention
        )

        # 7. Aggregate
        aggregated, agg_weights = self.tf_aggregator(tf_embeddings)

        # 8. Predictions
        predictions = self.prediction_heads(aggregated)

        if return_attention:
            predictions['tf_attention_weights'] = attn_weights
            predictions['aggregation_weights'] = agg_weights

        return predictions

    def get_feature_importance(self) -> Optional[torch.Tensor]:
        """Get learned feature importance from explicit weights."""
        if self.feature_weights is not None:
            return self.feature_weights.get_feature_importance()
        return None


def create_model(config: Optional[Dict] = None) -> V15Model:
    """
    Create V15 model with config.

    Args:
        config: Optional config dict, defaults to MODEL_CONFIG

    Returns:
        Initialized V15Model
    """
    cfg = {**MODEL_CONFIG, **(config or {})}

    return V15Model(
        input_dim=cfg['input_dim'],
        hidden_dim=cfg['hidden_dim'],
        n_attention_heads=cfg['n_attention_heads'],
        dropout=cfg['dropout'],
        use_explicit_weights=cfg['use_explicit_weights'],
    )
