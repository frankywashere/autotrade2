"""
Cross-Timeframe Attention.

Learns which timeframes are most relevant for each prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class CrossTFAttention(nn.Module):
    """
    Multi-head attention over timeframe embeddings.

    Args:
        embed_dim: Dimension of TF embeddings
        n_heads: Number of attention heads
        dropout: Attention dropout
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply cross-TF attention.

        Args:
            x: [batch, n_timeframes, embed_dim]
            return_attention: If True, return attention weights

        Returns:
            output: [batch, n_timeframes, embed_dim]
            attention_weights: [batch, n_heads, n_tf, n_tf] if return_attention
        """
        # Self-attention over timeframes
        attn_out, attn_weights = self.attention(x, x, x, need_weights=return_attention)

        # Residual + norm
        x = self.norm1(x + attn_out)

        # FFN + residual + norm
        x = self.norm2(x + self.ffn(x))

        if return_attention:
            return x, attn_weights
        return x, None


class TFAggregator(nn.Module):
    """
    Aggregates timeframe embeddings into a single representation.

    Supports multiple aggregation strategies:
    - 'mean': Simple average
    - 'attention': Learned attention weights
    - 'concat': Concatenate all (increases dim)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_timeframes: int = 11,
        strategy: str = 'attention',
        output_dim: Optional[int] = None
    ):
        super().__init__()

        self.strategy = strategy
        self.embed_dim = embed_dim
        self.n_timeframes = n_timeframes

        if strategy == 'attention':
            self.attention_weights = nn.Linear(embed_dim, 1)
            self.output_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        elif strategy == 'concat':
            self.output_proj = nn.Linear(embed_dim * n_timeframes, output_dim or embed_dim)
        else:  # mean
            self.output_proj = nn.Linear(embed_dim, output_dim or embed_dim) if output_dim else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Aggregate timeframe embeddings.

        Args:
            x: [batch, n_timeframes, embed_dim]

        Returns:
            aggregated: [batch, output_dim]
            weights: [batch, n_timeframes] attention weights (if strategy='attention')
        """
        if self.strategy == 'mean':
            agg = x.mean(dim=1)
            if self.output_proj:
                agg = self.output_proj(agg)
            return agg, None

        elif self.strategy == 'attention':
            # Compute attention weights
            scores = self.attention_weights(x).squeeze(-1)  # [batch, n_tf]
            weights = F.softmax(scores, dim=-1)

            # Weighted sum
            agg = torch.einsum('bt,btd->bd', weights, x)
            agg = self.output_proj(agg)
            return agg, weights

        else:  # concat
            agg = x.view(x.size(0), -1)  # [batch, n_tf * embed_dim]
            agg = self.output_proj(agg)
            return agg, None
