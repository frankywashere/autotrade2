"""
Cross-Timeframe Attention.

Learns which timeframes are most relevant for each prediction.
Supports both global attention (all TFs in one pool) and
horizon-grouped attention (short/medium/long pools with cross-group exchange).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

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


class _AttentionBlock(nn.Module):
    """Single transformer block: self-attention + FFN with residuals."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class HorizonGroupedAttention(nn.Module):
    """
    Horizon-grouped cross-TF attention.

    Instead of mixing all 10 TFs in one attention pool, groups them into
    short/medium/long horizon pools. Each group attends within itself first,
    then a lightweight cross-group exchange allows controlled information flow.

    This prevents short-TF noise (5min) from leaking into long-TF (daily/weekly)
    predictions and vice versa.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 8,
        dropout: float = 0.1,
        horizon_indices: Optional[Dict[str, List[int]]] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Default: short=[0,1,2], medium=[3,4,5,6], long=[7,8,9]
        if horizon_indices is None:
            horizon_indices = {
                'short': [0, 1, 2],
                'medium': [3, 4, 5, 6],
                'long': [7, 8, 9],
            }
        self.horizon_indices = horizon_indices
        self.group_names = list(horizon_indices.keys())

        # Within-group attention (one block per group)
        self.group_attention = nn.ModuleDict({
            name: _AttentionBlock(embed_dim, n_heads, dropout)
            for name in self.group_names
        })

        # Cross-group exchange: each group produces a summary, all groups
        # read from all summaries via a small attention layer
        self.summary_proj = nn.ModuleDict({
            name: nn.Linear(embed_dim, embed_dim)
            for name in self.group_names
        })
        self.cross_group_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=min(n_heads, 4),
            dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(embed_dim)
        self.cross_gate = nn.Linear(embed_dim * 2, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch, n_timeframes, embed_dim]
        Returns:
            output: [batch, n_timeframes, embed_dim]
            attention_weights: None (not supported for grouped attention)
        """
        batch = x.size(0)
        output = torch.zeros_like(x)

        # Phase 1: Within-group self-attention
        group_summaries = []
        for name in self.group_names:
            idx = self.horizon_indices[name]
            idx_t = torch.tensor(idx, device=x.device)
            group_x = x[:, idx_t, :]                          # [B, group_size, D]
            group_x = self.group_attention[name](group_x)      # within-group attention
            output[:, idx_t, :] = group_x

            # Produce group summary (mean-pool then project)
            summary = self.summary_proj[name](group_x.mean(dim=1))  # [B, D]
            group_summaries.append(summary)

        # Phase 2: Cross-group exchange
        summaries = torch.stack(group_summaries, dim=1)  # [B, 3, D]
        cross_out, _ = self.cross_group_attn(summaries, summaries, summaries, need_weights=False)
        cross_out = self.cross_norm(summaries + cross_out)  # [B, 3, D]

        # Gate the cross-group info into each TF position
        for i, name in enumerate(self.group_names):
            idx = self.horizon_indices[name]
            idx_t = torch.tensor(idx, device=x.device)
            group_out = output[:, idx_t, :]                              # [B, group_size, D]
            cross_info = cross_out[:, i:i+1, :].expand_as(group_out)     # [B, group_size, D]
            gate_input = torch.cat([group_out, cross_info], dim=-1)      # [B, group_size, 2D]
            gate = torch.sigmoid(self.cross_gate(gate_input))            # [B, group_size, D]
            output[:, idx_t, :] = group_out + gate * cross_info

        return output, None


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
