"""
Full V15 Channel Prediction Model.

Takes all 8,632+ features, applies explicit weights, encodes per-TF,
applies cross-TF attention, and produces predictions.
"""
import torch
import logging

logger = logging.getLogger(__name__)

import torch.nn as nn
from typing import Dict, Optional, Tuple

from .feature_weights import ExplicitFeatureWeights, FeatureGating
from .tf_encoder import MultiTFEncoder
from .cross_tf_attention import CrossTFAttention, TFAggregator
from .prediction_heads import PredictionHeads, PerTFPredictionHeads, PerTFPredictionHeadsV2
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
        Prediction Heads (duration, direction, new_channel)
            ↓
        [Optional] Window Selector (learned window selection)

    When use_window_selector=True, the model also predicts which of the 8
    lookback windows is optimal. This enables end-to-end training where
    the duration loss backpropagates through the window selection.
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
        share_tf_weights: bool = False,
        use_window_selector: bool = False,
        num_windows: int = 8,
        # Break scan label head flags
        enable_tsla_heads: bool = False,
        enable_spy_heads: bool = False,
        enable_cross_correlation_heads: bool = False,
        # Durability and RSI head flags
        enable_durability_heads: bool = False,
        enable_rsi_heads: bool = False,
        # Per-TF head version: 1 = original lightweight, 2 = with TF embedding + bigger
        per_tf_head_version: int = 1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_timeframes = n_timeframes
        self.features_per_tf = features_per_tf
        self.use_window_selector = use_window_selector
        self.num_windows = num_windows
        self.enable_tsla_heads = enable_tsla_heads
        self.enable_spy_heads = enable_spy_heads
        self.enable_cross_correlation_heads = enable_cross_correlation_heads
        self.enable_durability_heads = enable_durability_heads
        self.enable_rsi_heads = enable_rsi_heads

        # Shared features = events + bar metadata
        self.shared_features_dim = (
            FEATURE_COUNTS['events_total'] +
            FEATURE_COUNTS['bar_metadata_per_tf'] * n_timeframes
        )

        # Validate or compute features_per_tf from input_dim
        # This allows flexibility when C++ scanner produces different feature counts
        expected_dim = features_per_tf * n_timeframes + self.shared_features_dim
        if input_dim != expected_dim:
            # Compute features_per_tf from actual input_dim
            computed_per_tf = (input_dim - self.shared_features_dim) // n_timeframes
            if computed_per_tf * n_timeframes + self.shared_features_dim == input_dim:
                logger.info(
                    f"Adjusting features_per_tf: {features_per_tf} -> {computed_per_tf} "
                    f"(input_dim={input_dim})"
                )
                self.features_per_tf = computed_per_tf
            else:
                # Can't evenly divide, use input_dim directly as flat features
                logger.warning(
                    f"Input dim {input_dim} doesn't match expected structure. "
                    f"Using flat feature processing."
                )
                self.features_per_tf = (input_dim - self.shared_features_dim) // n_timeframes

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
            features_per_tf=self.features_per_tf,
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

        # 6. Prediction Heads (with optional window selector and break scan heads)
        self.prediction_heads = PredictionHeads(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            use_window_selector=use_window_selector,
            num_windows=num_windows,
            enable_tsla_heads=enable_tsla_heads,
            enable_spy_heads=enable_spy_heads,
            enable_cross_correlation_heads=enable_cross_correlation_heads,
            enable_durability_heads=enable_durability_heads,
            enable_rsi_heads=enable_rsi_heads,
        )

        # 7. Per-TF Prediction Heads for per-timeframe breakdown
        if per_tf_head_version == 2:
            self.per_tf_heads = PerTFPredictionHeadsV2(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim // 2,
                n_timeframes=n_timeframes,
            )
        else:
            self.per_tf_heads = PerTFPredictionHeads(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim // 4,
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
        return_per_tf: bool = False,
        validate: bool = True,
        window_selector_temperature: float = 1.0,
        window_selector_hard: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [batch, input_dim] raw features
            return_attention: If True, include attention weights in output
            return_per_tf: If True, include per-TF predictions under 'per_tf' key
            validate: If True, check for NaN/Inf (LOUD failure)
            window_selector_temperature: Temperature for window selection softmax (default: 1.0)
            window_selector_hard: If True, use argmax for window selection (inference mode)

        Returns:
            Dict with predictions and optional attention weights.
            If use_window_selector, also includes 'window_selection' dict.
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

        # 7. Per-TF predictions (before aggregation)
        if return_per_tf:
            per_tf_predictions = self.per_tf_heads(tf_embeddings)

        # 8. Aggregate
        aggregated, agg_weights = self.tf_aggregator(tf_embeddings)

        # 9. Predictions
        predictions = self.prediction_heads(
            aggregated,
            window_selector_temperature=window_selector_temperature,
            window_selector_hard=window_selector_hard,
        )

        if return_attention:
            predictions['tf_attention_weights'] = attn_weights
            predictions['aggregation_weights'] = agg_weights

        if return_per_tf:
            predictions['per_tf'] = per_tf_predictions

        return predictions

    def forward_with_per_tf(
        self,
        x: torch.Tensor,
        validate: bool = True,
        window_selector_temperature: float = 1.0,
        window_selector_hard: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass that also returns per-TF predictions.

        This method is useful for per-timeframe prediction breakdown.
        Returns both the normal (aggregated) predictions and per-TF predictions
        from lightweight heads run on embeddings after cross-TF attention.

        Args:
            x: [batch, input_dim] raw features
            validate: If True, check for NaN/Inf (LOUD failure)
            window_selector_temperature: Temperature for window selection softmax
            window_selector_hard: If True, use argmax for window selection

        Returns:
            Tuple of:
                - predictions: Dict with all aggregated prediction outputs (same as forward())
                - per_tf_predictions: Dict with per-TF predictions:
                    - 'duration_mean': [batch, n_timeframes]
                    - 'duration_log_std': [batch, n_timeframes]
                    - 'direction_logits': [batch, n_timeframes]
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
        tf_embeddings_attended, _ = self.cross_tf_attention(
            tf_embeddings, return_attention=False
        )

        # 7. Per-TF predictions (before aggregation)
        per_tf_predictions = self.per_tf_heads(tf_embeddings_attended)

        # 8. Aggregate
        aggregated, _ = self.tf_aggregator(tf_embeddings_attended)

        # 9. Aggregated predictions
        predictions = self.prediction_heads(
            aggregated,
            window_selector_temperature=window_selector_temperature,
            window_selector_hard=window_selector_hard,
        )

        return predictions, per_tf_predictions

    def get_feature_importance(self) -> Optional[torch.Tensor]:
        """Get learned feature importance from explicit weights."""
        if self.feature_weights is not None:
            return self.feature_weights.get_feature_importance()
        return None

    def has_window_selector(self) -> bool:
        """Check if this model has a learned window selector."""
        return self.prediction_heads.has_window_selector()

    def has_tsla_heads(self) -> bool:
        """Check if this model has TSLA break scan heads."""
        return self.prediction_heads.has_tsla_heads()

    def has_spy_heads(self) -> bool:
        """Check if this model has SPY break scan heads."""
        return self.prediction_heads.has_spy_heads()

    def has_cross_correlation_heads(self) -> bool:
        """Check if this model has cross-correlation heads."""
        return self.prediction_heads.has_cross_correlation_heads()

    def has_durability_heads(self) -> bool:
        """Check if this model has durability and bars-to-permanent heads."""
        return self.prediction_heads.has_durability_heads()

    def has_rsi_heads(self) -> bool:
        """Check if this model has RSI prediction heads."""
        return self.prediction_heads.has_rsi_heads()


def create_model(config: Optional[Dict] = None) -> V15Model:
    """
    Create V15 model with config.

    Args:
        config: Optional config dict, defaults to MODEL_CONFIG
            Supported keys:
            - input_dim: Total number of input features
            - hidden_dim: Hidden layer dimension
            - n_attention_heads: Number of attention heads
            - dropout: Dropout probability
            - use_explicit_weights: Whether to use explicit feature weights
            - use_window_selector: Whether to include learned window selection
            - num_windows: Number of windows for selector (default: 8)
            - enable_tsla_heads: Whether to include TSLA break scan heads (default: False)
            - enable_spy_heads: Whether to include SPY break scan heads (default: False)
            - enable_cross_correlation_heads: Whether to include cross-correlation heads (default: False)
            - enable_durability_heads: Whether to include durability heads (default: False)
            - enable_rsi_heads: Whether to include RSI prediction heads (default: False)

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
        use_window_selector=cfg.get('use_window_selector', False),
        num_windows=cfg.get('num_windows', 8),
        enable_tsla_heads=cfg.get('enable_tsla_heads', False),
        enable_spy_heads=cfg.get('enable_spy_heads', False),
        enable_cross_correlation_heads=cfg.get('enable_cross_correlation_heads', False),
        enable_durability_heads=cfg.get('enable_durability_heads', False),
        enable_rsi_heads=cfg.get('enable_rsi_heads', False),
        per_tf_head_version=cfg.get('per_tf_head_version', 1),
    )
