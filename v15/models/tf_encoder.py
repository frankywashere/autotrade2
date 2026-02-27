"""
Per-Timeframe Feature Encoder.

Each timeframe has its own encoder to process its features independently
before cross-TF attention.
"""
import torch
import torch.nn as nn
from typing import Optional

class TFEncoder(nn.Module):
    """
    Encoder for a single timeframe's features.

    Args:
        input_dim: Number of features for this TF (~782)
        hidden_dim: Hidden dimension for encoding
        output_dim: Output embedding dimension
        n_layers: Number of encoding layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []

        # Input projection
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))

        # Output projection
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode timeframe features.

        Args:
            x: [batch_size, input_dim] features for this TF

        Returns:
            [batch_size, output_dim] encoded representation
        """
        return self.encoder(x)


class MultiTFEncoder(nn.Module):
    """
    Encoder for all timeframes with shared or independent weights.

    Args:
        n_timeframes: Number of timeframes (11)
        features_per_tf: Features per timeframe (~782)
        shared_features: Number of shared features (events, etc.)
        hidden_dim: Hidden dimension
        output_dim: Output dimension per TF
        share_weights: If True, all TFs share encoder weights
    """

    def __init__(
        self,
        n_timeframes: int = 11,
        features_per_tf: int = 782,
        shared_features: int = 63,  # events + bar metadata
        hidden_dim: int = 256,
        output_dim: int = 128,
        share_weights: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_timeframes = n_timeframes
        self.features_per_tf = features_per_tf
        self.shared_features = shared_features

        # Total input per TF encoder = TF features + shared features
        encoder_input_dim = features_per_tf + shared_features

        if share_weights:
            # Single encoder shared across all TFs
            self.encoder = TFEncoder(
                encoder_input_dim, hidden_dim, output_dim, dropout=dropout
            )
            self.encoders = None
        else:
            # Independent encoder per TF
            self.encoder = None
            self.encoders = nn.ModuleList([
                TFEncoder(encoder_input_dim, hidden_dim, output_dim, dropout=dropout)
                for _ in range(n_timeframes)
            ])

    def forward(
        self,
        tf_features: torch.Tensor,
        shared_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode all timeframe features.

        Args:
            tf_features: [batch, n_timeframes, features_per_tf]
            shared_features: [batch, shared_features]

        Returns:
            [batch, n_timeframes, output_dim] encoded representations
        """
        batch_size = tf_features.size(0)
        outputs = []

        # Expand shared features for concatenation
        shared_expanded = shared_features.unsqueeze(1).expand(-1, self.n_timeframes, -1)

        for i in range(self.n_timeframes):
            # Concatenate TF features with shared features
            tf_input = torch.cat([tf_features[:, i], shared_expanded[:, i]], dim=-1)

            if self.encoders is not None:
                encoded = self.encoders[i](tf_input)
            else:
                encoded = self.encoder(tf_input)

            outputs.append(encoded)

        return torch.stack(outputs, dim=1)
