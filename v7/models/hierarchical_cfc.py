"""
Hierarchical CfC Neural Network for Multi-Timeframe Channel Prediction (v7.0)

Architecture Overview:
====================
This model predicts channel break timing, direction, and post-break channel direction
using a hierarchical architecture that processes each timeframe independently before
combining insights across timeframes.

Input Features: 644 dimensions (TIMEFRAME-GROUPED ordering)
------------------------------------------------------------
The input tensor is ordered by timeframe, NOT alphabetically!
Each TF block contains: [tsla_{tf}, spy_{tf}, cross_{tf}] = 49 features
Followed by shared features at the end.

- TSLA per-TF features: 30 features × 11 timeframes = 330 total
  * Channel geometry: direction, position, width, slope, R²
  * Bounce metrics: count, cycles, bars since bounce
  * RSI: current, divergence, at upper/lower bounces
  * Exit tracking: exit counts, frequency, acceleration (10 features)
  * Break triggers: distance to longer TF boundaries (2 features)
  * Quality scores: channel_quality, rsi_confidence

- SPY per-TF features: 11 features × 11 timeframes = 121 total
  * Channel geometry and position
  * Bounce metrics and RSI

- Cross-asset: 8 features × 11 timeframes = 88 total
  * TSLA position in SPY channels
  * Alignment scores

- VIX regime: 6 features
  * Level, normalized level, trends, percentile, regime

- History features: 25 features (TSLA) + 25 features (SPY) = 50 total
  * Last 5 directions, durations, break directions (5+5+5 = 15)
  * Summary stats: avg duration, streak, counts, RSI stats (10)

- Alignment: 3 features
  * Direction match, both near upper/lower

- Events: 46 features (zeros if not provided)
  * Timing, earnings context, pre/post event drift

Architecture Flow:
=================
1. Input Layer (644 dims) → Feature Decomposition
   ├─ Per-TF features extracted for each of 11 timeframes (49 features each)
   ├─ Shared features (VIX, history, alignment, events) extracted from end

2. TF Branch Processing (11 parallel branches)
   ├─ Each branch: Linear projection → LayerNorm → CfC → Dropout
   ├─ Branch processes its TF + shared context
   ├─ Output: 64-dim embedding per TF

3. Cross-TF Attention
   ├─ Multi-head attention over 11 TF embeddings
   ├─ Learns which TFs are relevant for current prediction
   ├─ Output: Attended context vector (128 dims)

4. Prediction Heads (shared across all TFs)
   ├─ Duration Head: Predicts mean and std (Gaussian NLL loss)
   ├─ Break Direction Head: Binary classification (up/down)
   ├─ Next Channel Direction Head: 3-class (bear/sideways/bull)
   ├─ Confidence Head: Calibrated probability estimate

Outputs per Timeframe:
=====================
- duration_mean: Expected bars until break
- duration_log_std: Log of uncertainty in duration prediction
- direction_logits: Binary break direction (0=down, 1=up)
- next_channel_logits: [bear, sideways, bull]
- confidence: Calibrated prediction confidence [0, 1]

Training:
=========
- Duration: Gaussian NLL loss (predicts distribution, not point estimate)
- Break direction: Binary cross-entropy
- Next direction: Categorical cross-entropy
- Confidence: Calibration loss (Brier score or focal calibration)

Key Design Decisions:
====================
1. Hierarchical processing: Each TF gets its own CfC to capture temporal dynamics
2. Attention fusion: Cross-TF attention learns context-dependent TF importance
3. Shared heads: Same prediction logic across TFs (transfer learning within model)
4. Probabilistic duration: Gaussian distribution captures uncertainty
5. Confidence calibration: Network learns when to be uncertain
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

# Liquid Neural Network components
from ncps.torch import CfC
from ncps.wirings import AutoNCP


# =============================================================================
# Feature Configuration
# =============================================================================

# Import canonical feature dimensions from feature_ordering module
from v7.features.feature_ordering import (
    TSLA_PER_TF, SPY_PER_TF, CROSS_PER_TF,
    VIX_FEATURES, TSLA_HISTORY_FEATURES, SPY_HISTORY_FEATURES,
    ALIGNMENT_FEATURES, EVENT_FEATURES,
    PER_TF_FEATURES, SHARED_FEATURES, N_TIMEFRAMES, TOTAL_FEATURES,
    get_tf_index_range, get_shared_index_range
)


@dataclass
class FeatureConfig:
    """
    Configuration defining input feature dimensions.

    IMPORTANT: These values must match v7/features/feature_ordering.py!
    The canonical source of truth is feature_ordering.py. This dataclass
    uses those constants to ensure consistency.

    Feature Layout (TIMEFRAME-GROUPED ordering):
    - Per-TF block: [tsla_{tf}(30), spy_{tf}(11), cross_{tf}(8)] = 49 features
    - 11 timeframes × 49 = 539 per-TF features
    - Shared: [vix(6), tsla_history(25), spy_history(25), alignment(3), events(46)] = 105 features
    - Total: 539 + 105 = 644 features
    """

    # Per-timeframe feature counts (from feature_ordering.py)
    tsla_per_tf: int = TSLA_PER_TF    # 30 (18 base + 10 exit_tracking + 2 break_trigger)
    spy_per_tf: int = SPY_PER_TF      # 11 (channel metrics + RSI)
    cross_per_tf: int = CROSS_PER_TF  # 8 (TSLA-in-SPY containment)

    # Shared features (same across all TFs)
    vix_features: int = VIX_FEATURES                      # 6
    tsla_history_features: int = TSLA_HISTORY_FEATURES    # 25 (5+5+5 lists + 10 scalars)
    spy_history_features: int = SPY_HISTORY_FEATURES      # 25 (same structure)
    alignment_features: int = ALIGNMENT_FEATURES          # 3
    event_features: int = EVENT_FEATURES                  # 46 (zeros if not provided)

    # Number of timeframes
    n_timeframes: int = N_TIMEFRAMES  # 11: 5min, 15min, 30min, 1h, 2h, 3h, 4h, daily, weekly, monthly, 3month

    @property
    def total_features(self) -> int:
        """Calculate total input dimension."""
        per_tf = (self.tsla_per_tf + self.spy_per_tf + self.cross_per_tf) * self.n_timeframes
        shared = (self.vix_features + self.tsla_history_features +
                  self.spy_history_features + self.alignment_features + self.event_features)
        return per_tf + shared  # = (30+11+8)*11 + (6+25+25+3+46) = 539 + 105 = 644

    @property
    def shared_features(self) -> int:
        """Total shared feature dimension."""
        return (self.vix_features + self.tsla_history_features +
                self.spy_history_features + self.alignment_features + self.event_features)

    @property
    def per_tf_features(self) -> int:
        """Features per timeframe."""
        return self.tsla_per_tf + self.spy_per_tf + self.cross_per_tf

    def get_tf_slice(self, tf_idx: int) -> Tuple[int, int]:
        """
        Get start and end indices for a timeframe's features.

        Uses canonical indexing from feature_ordering module.

        Args:
            tf_idx: Timeframe index (0-10)

        Returns:
            (start_idx, end_idx) for slicing input tensor
        """
        return get_tf_index_range(tf_idx)

    def get_shared_slice(self) -> Tuple[int, int]:
        """
        Get start and end indices for shared features.

        Uses canonical indexing from feature_ordering module.

        Returns:
            (start_idx, end_idx) for slicing input tensor
        """
        return get_shared_index_range()


# =============================================================================
# Timeframe Branch (CfC Processing)
# =============================================================================

class TFBranch(nn.Module):
    """
    Processes features for a single timeframe using a Liquid Neural Network (CfC).

    Each branch:
    1. Takes per-TF features + shared context
    2. Projects to hidden dimension
    3. Processes through CfC (captures temporal dynamics)
    4. Outputs embedding for cross-TF attention

    The CfC layer is particularly well-suited here because:
    - It models continuous-time dynamics (natural for financial data)
    - Has strong extrapolation capabilities
    - Learns causal relationships between features
    - Compact parameter count vs LSTM/GRU
    """

    def __init__(
        self,
        per_tf_dim: int,
        shared_dim: int,
        hidden_dim: int = 64,
        cfc_units: int = 96,  # Must be > hidden_dim + 2
        dropout: float = 0.1
    ):
        """
        Initialize timeframe branch.

        Args:
            per_tf_dim: Dimension of per-timeframe features (51 for TSLA+SPY+cross)
            shared_dim: Dimension of shared features (65 for VIX+history+alignment)
            hidden_dim: Output embedding dimension
            cfc_units: Number of CfC units (neurons in liquid network, must be > hidden_dim + 2)
            dropout: Dropout probability
        """
        super().__init__()

        input_dim = per_tf_dim + shared_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # CfC (Liquid Neural Network) layer
        # AutoNCP creates a sparsely connected recurrent architecture
        # Note: cfc_units must be > hidden_dim + 2 for AutoNCP
        wiring = AutoNCP(cfc_units, hidden_dim)
        self.cfc = CfC(hidden_dim, wiring, batch_first=True)

        # Output processing
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.hidden_dim = hidden_dim

    def forward(
        self,
        per_tf_features: torch.Tensor,
        shared_features: torch.Tensor,
        hx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through timeframe branch.

        Args:
            per_tf_features: [batch_size, per_tf_dim] - features specific to this TF
            shared_features: [batch_size, shared_dim] - shared context (VIX, history, etc.)
            hx: Optional hidden state from previous timestep

        Returns:
            embedding: [batch_size, hidden_dim] - TF embedding for attention
            hx_new: Hidden state for next timestep
        """
        # Concatenate per-TF and shared features
        x = torch.cat([per_tf_features, shared_features], dim=-1)  # [batch, input_dim]

        # Project to hidden dimension
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.gelu(x)

        # Add sequence dimension for CfC (expects 3D input)
        x = x.unsqueeze(1)  # [batch, 1, hidden_dim]

        # Process through CfC
        x, hx_new = self.cfc(x, hx)

        # Remove sequence dimension
        x = x.squeeze(1)  # [batch, hidden_dim]

        # Output processing
        x = self.output_norm(x)
        x = self.dropout(x)

        return x, hx_new


# =============================================================================
# Cross-Timeframe Attention
# =============================================================================

class CrossTFAttention(nn.Module):
    """
    Multi-head attention over timeframe embeddings.

    This layer learns which timeframes are most relevant for the current prediction.
    For example:
    - When predicting 5min breaks, daily/weekly context might matter most
    - When predicting daily breaks, longer TFs (weekly/monthly) are key
    - During high volatility, shorter TFs might dominate

    The attention mechanism adaptively weights TF contributions based on current
    market state encoded in the TF embeddings.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize cross-TF attention.

        Args:
            embed_dim: Dimension of TF embeddings (from branches)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Output projection (doubles dimension for richer fusion)
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        tf_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Attend over timeframe embeddings.

        Args:
            tf_embeddings: [batch_size, n_timeframes, embed_dim]
            mask: Optional attention mask

        Returns:
            context: [batch_size, embed_dim * 2] - attended context vector
            attn_weights: [batch_size, n_timeframes, n_timeframes] - attention weights
        """
        # Self-attention over timeframes
        attn_out, attn_weights = self.attention(
            tf_embeddings,  # query
            tf_embeddings,  # key
            tf_embeddings,  # value
            attn_mask=mask
        )
        # attn_out: [batch, n_tf, embed_dim]
        # attn_weights: [batch, n_tf, n_tf]

        # Global pooling: mean over timeframes
        context = attn_out.mean(dim=1)  # [batch, embed_dim]

        # Project to higher dimension
        context = self.output_proj(context)  # [batch, embed_dim * 2]

        return context, attn_weights


# =============================================================================
# Prediction Heads
# =============================================================================

class DurationHead(nn.Module):
    """
    Predicts duration as a Gaussian distribution (mean and std).

    This is more principled than predicting a single value because:
    - Captures uncertainty (wide std = uncertain prediction)
    - Enables probabilistic forecasting
    - Trained with Gaussian NLL loss
    - Can sample from distribution at inference
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Initialize duration head.

        Args:
            input_dim: Input dimension (from attention context)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        # Separate heads for mean and log(std)
        self.mean_head = nn.Linear(hidden_dim // 2, 1)
        self.logstd_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [batch_size, input_dim]

        Returns:
            mean: [batch_size, 1] - predicted mean duration
            log_std: [batch_size, 1] - predicted log(standard deviation)
        """
        h = self.net(x)

        mean = self.mean_head(h)
        mean = F.softplus(mean) + 1.0  # Ensure positive, minimum 1 bar

        log_std = self.logstd_head(h)
        # Bounded for numerical stability. Range [-2, 4] gives std in [0.14, 54.6]
        # which is appropriate for duration data ranging from 1 to 500 bars
        log_std = log_std.clamp(-2, 4)

        return mean, log_std


class DirectionHead(nn.Module):
    """
    Predicts break direction (binary: up or down).

    Outputs logits for binary classification.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Initialize direction head.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)  # 1 logit for binary classification (UP probability)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch_size, input_dim]

        Returns:
            logits: [batch_size, 1] - single logit for BCEWithLogitsLoss
        """
        return self.net(x)


class NextChannelDirectionHead(nn.Module):
    """
    Predicts next channel direction (3-class: bear/sideways/bull).

    This predicts what type of channel forms AFTER the current one breaks.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Initialize next channel direction head.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3)  # 3 classes: bear, sideways, bull
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch_size, input_dim]

        Returns:
            logits: [batch_size, 3] - class logits
        """
        return self.net(x)


class ConfidenceHead(nn.Module):
    """
    Predicts calibrated confidence score [0, 1].

    This is trained to output well-calibrated probabilities:
    - confidence=0.8 means model is correct ~80% of the time
    - confidence=0.5 means model is uncertain

    Can be trained with Brier score or focal calibration loss.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Initialize confidence head.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch_size, input_dim]

        Returns:
            confidence: [batch_size, 1] - calibrated confidence
        """
        return self.net(x)


# =============================================================================
# Full Hierarchical Model
# =============================================================================

class HierarchicalCfCModel(nn.Module):
    """
    Complete hierarchical CfC model for multi-timeframe channel prediction.

    Architecture:
    1. Decompose 644-dim input into per-TF (49 features each) and shared features (105)
    2. Process each TF through dedicated CfC branch
    3. Attend over TF embeddings to create unified context
    4. Predict duration, break direction, next channel direction, confidence

    CRITICAL: Input must use TIMEFRAME-GROUPED ordering from feature_ordering.py!
    - Indices 0-48: TF0 (5min) = tsla_5min(30) + spy_5min(11) + cross_5min(8)
    - Indices 49-97: TF1 (15min) = tsla_15min(30) + spy_15min(11) + cross_15min(8)
    - ... (11 timeframes total)
    - Indices 539-643: Shared = vix(6) + tsla_history(25) + spy_history(25) + alignment(3) + events(46)

    This design allows the model to:
    - Learn timeframe-specific temporal dynamics
    - Adaptively weight timeframes based on context
    - Share prediction logic across scales
    - Provide calibrated uncertainty estimates
    """

    def __init__(
        self,
        feature_config: Optional[FeatureConfig] = None,
        hidden_dim: int = 64,
        cfc_units: int = 96,  # Must be > hidden_dim + 2
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        shared_heads: bool = True
    ):
        """
        Initialize hierarchical CfC model.

        Args:
            feature_config: Feature dimension configuration
            hidden_dim: Hidden dimension for branches and heads
            cfc_units: Number of units in each CfC layer (must be > hidden_dim + 2)
            num_attention_heads: Number of attention heads
            dropout: Dropout probability
            shared_heads: If True (default), use single shared heads for all TFs.
                         If False, create separate heads per timeframe (11x more head params).
        """
        super().__init__()

        self.config = feature_config or FeatureConfig()
        self.hidden_dim = hidden_dim
        self.n_timeframes = self.config.n_timeframes
        self.shared_heads = shared_heads

        # Validate total input dimension against canonical value from feature_ordering
        assert self.config.total_features == TOTAL_FEATURES, \
            f"Expected {TOTAL_FEATURES} features, got {self.config.total_features}"

        # Validate CfC configuration
        assert cfc_units > hidden_dim + 2, \
            f"cfc_units ({cfc_units}) must be > hidden_dim + 2 ({hidden_dim + 2})"

        # Create TF branches (11 parallel CfC processors)
        self.tf_branches = nn.ModuleList([
            TFBranch(
                per_tf_dim=self.config.per_tf_features,
                shared_dim=self.config.shared_features,
                hidden_dim=hidden_dim,
                cfc_units=cfc_units,
                dropout=dropout
            )
            for _ in range(self.n_timeframes)
        ])

        # Cross-timeframe attention
        self.cross_tf_attention = CrossTFAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )

        # Prediction heads (dual output design)
        context_dim = hidden_dim * 2  # From attention output projection

        # Per-timeframe prediction heads
        if self.shared_heads:
            # SHARED: Single set of heads for all TFs (default, fewer parameters)
            self.per_tf_duration_head = DurationHead(hidden_dim, hidden_dim // 2)
            self.per_tf_direction_head = DirectionHead(hidden_dim, hidden_dim // 2)
            self.per_tf_next_channel_head = NextChannelDirectionHead(hidden_dim, hidden_dim // 2)
            self.per_tf_confidence_head = ConfidenceHead(hidden_dim, hidden_dim // 2)
        else:
            # SEPARATE: 11 separate heads per prediction type (more parameters, TF-specific)
            self.per_tf_duration_heads = nn.ModuleList([
                DurationHead(hidden_dim, hidden_dim // 2)
                for _ in range(self.n_timeframes)
            ])
            self.per_tf_direction_heads = nn.ModuleList([
                DirectionHead(hidden_dim, hidden_dim // 2)
                for _ in range(self.n_timeframes)
            ])
            self.per_tf_next_channel_heads = nn.ModuleList([
                NextChannelDirectionHead(hidden_dim, hidden_dim // 2)
                for _ in range(self.n_timeframes)
            ])
            self.per_tf_confidence_heads = nn.ModuleList([
                ConfidenceHead(hidden_dim, hidden_dim // 2)
                for _ in range(self.n_timeframes)
            ])

        # Aggregate prediction heads (richer, use full attention context) - always shared
        self.agg_duration_head = DurationHead(context_dim, hidden_dim)
        self.agg_direction_head = DirectionHead(context_dim, hidden_dim)
        self.agg_next_channel_head = NextChannelDirectionHead(context_dim, hidden_dim)
        self.agg_confidence_head = ConfidenceHead(context_dim, hidden_dim)

        # Store hidden states for sequential processing
        self.register_buffer('_hx_states', None)

    def reset_hidden_states(self):
        """
        Reset hidden states for all CfC branches.

        Call this at the start of a new sequence or when you want
        stateless processing (e.g., during training on shuffled batches).
        """
        # Clear stored hidden states - next forward will start fresh
        self.hx_states = None

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical model.

        Args:
            x: [batch_size, 644] - full feature vector (TIMEFRAME-GROUPED ordering!)
            return_attention: If True, return attention weights

        Returns:
            Dictionary with:
                - duration_mean: [batch_size, 11]
                - duration_log_std: [batch_size, 11]
                - direction_logits: [batch_size, 11]
                - next_channel_logits: [batch_size, 11, 3]
                - confidence: [batch_size, 11]
                - attention_weights: [batch_size, n_tf, n_tf] (if return_attention=True)
        """
        batch_size = x.size(0)
        device = x.device

        # Validate input shape
        expected_features = self.config.total_features
        actual_features = x.size(1)
        if actual_features != expected_features:
            raise ValueError(
                f"Input feature dimension mismatch: expected {expected_features}, got {actual_features}. "
                f"Ensure features are concatenated using FEATURE_ORDER from v7/features/feature_ordering.py"
            )

        # Extract shared features (same for all TFs)
        shared_start, shared_end = self.config.get_shared_slice()
        shared_features = x[:, shared_start:shared_end]  # [batch, 105]

        # Process each timeframe through its branch
        tf_embeddings = []

        # Initialize hidden states if needed (list of None or previous states)
        # CRITICAL: For training on shuffled batches, we should NOT carry state between batches
        # For now, always start fresh (stateless processing)
        hx_states = [None] * self.n_timeframes

        new_hx_states = []

        for tf_idx in range(self.n_timeframes):
            # Extract per-TF features
            start, end = self.config.get_tf_slice(tf_idx)
            per_tf_features = x[:, start:end]  # [batch, 51]

            # Process through branch with NO hidden state (stateless for shuffled batches)
            embedding, new_hx = self.tf_branches[tf_idx](
                per_tf_features,
                shared_features,
                hx=hx_states[tf_idx]  # Always None for stateless training
            )

            tf_embeddings.append(embedding)
            new_hx_states.append(new_hx)

        # Don't store hidden states - each batch is independent
        # This prevents gradient graph from linking across batches
        # If you need stateful processing, detach states before storing:
        # self.hx_states = [h.detach() if h is not None else None for h in new_hx_states]
        self.hx_states = None

        # Stack embeddings: [batch, n_tf, hidden_dim]
        tf_embeddings_stacked = torch.stack(tf_embeddings, dim=1)

        # =====================================================================
        # PER-TIMEFRAME PREDICTIONS (one prediction per TF)
        # =====================================================================
        # Each TF embedding gets its own prediction
        per_tf_durations_mean = []
        per_tf_durations_log_std = []
        per_tf_directions = []
        per_tf_next_channels = []
        per_tf_confidences = []

        for tf_idx, embedding in enumerate(tf_embeddings):
            # Each timeframe makes independent prediction
            if self.shared_heads:
                # Single shared head for all TFs
                dur_mean, dur_log_std = self.per_tf_duration_head(embedding)
                dir_logits = self.per_tf_direction_head(embedding)
                next_ch = self.per_tf_next_channel_head(embedding)
                conf = self.per_tf_confidence_head(embedding)
            else:
                # Separate head per TF (TF-specific learned parameters)
                dur_mean, dur_log_std = self.per_tf_duration_heads[tf_idx](embedding)
                dir_logits = self.per_tf_direction_heads[tf_idx](embedding)
                next_ch = self.per_tf_next_channel_heads[tf_idx](embedding)
                conf = self.per_tf_confidence_heads[tf_idx](embedding)

            per_tf_durations_mean.append(dur_mean)
            per_tf_durations_log_std.append(dur_log_std)
            per_tf_directions.append(dir_logits)  # [batch, 1] - binary logit for UP
            per_tf_next_channels.append(next_ch)
            per_tf_confidences.append(conf)

        # Stack to [batch, num_timeframes] or [batch, num_timeframes, classes]
        duration_mean = torch.cat(per_tf_durations_mean, dim=1)           # [batch, 11]
        duration_log_std = torch.cat(per_tf_durations_log_std, dim=1)     # [batch, 11]
        direction_logits = torch.cat(per_tf_directions, dim=1)            # [batch, 11] - UP logits only
        next_channel_logits = torch.stack(per_tf_next_channels, dim=1)   # [batch, 11, 3]
        confidence = torch.cat(per_tf_confidences, dim=1)                 # [batch, 11]

        # =====================================================================
        # AGGREGATE PREDICTION (optional - for dashboard summary)
        # =====================================================================
        # Use cross-TF attention to create weighted aggregate
        context, attn_weights = self.cross_tf_attention(tf_embeddings_stacked)

        agg_dur_mean, agg_dur_log_std = self.agg_duration_head(context)
        agg_direction = self.agg_direction_head(context)
        agg_next_channel = self.agg_next_channel_head(context)
        agg_confidence = self.agg_confidence_head(context)

        # Build output dictionary (keys match CombinedLoss expectations)
        output = {
            # Per-timeframe predictions (primary - used for training)
            'duration_mean': duration_mean,                    # [batch, 11]
            'duration_log_std': duration_log_std,              # [batch, 11]
            'direction_logits': direction_logits,              # [batch, 11]
            'next_channel_logits': next_channel_logits,        # [batch, 11, 3]
            'confidence': confidence,                          # [batch, 11]

            # Aggregate predictions (optional - for dashboard summary)
            'aggregate': {
                'duration_mean': agg_dur_mean,                 # [batch, 1]
                'duration_log_std': agg_dur_log_std,           # [batch, 1]
                'direction_logits': agg_direction,             # [batch, 1] - binary logit
                'next_channel_logits': agg_next_channel,       # [batch, 3]
                'confidence': agg_confidence,                  # [batch, 1]
            }
        }

        if return_attention:
            output['attention_weights'] = attn_weights

        return output

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make predictions in evaluation mode with per-timeframe breakdown.

        Args:
            x: [batch_size, 644] - input features (TIMEFRAME-GROUPED ordering!)

        Returns:
            Dictionary with:
                - Per-timeframe predictions (duration, direction, confidence for each of 11 TFs)
                - Aggregate predictions (weighted combination)
                - Attention weights
                - Recommended timeframe (highest confidence)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, return_attention=True)

            # Convert per-timeframe logits to probabilities
            direction_probs = torch.sigmoid(outputs['direction_logits'])  # [batch, 11]
            next_channel_probs = F.softmax(outputs['next_channel_logits'], dim=-1)  # [batch, 11, 3]

            # Convert aggregate logits to probabilities
            agg_direction_probs = torch.sigmoid(outputs['aggregate']['direction_logits'])  # [batch, 1]
            agg_next_channel_probs = F.softmax(outputs['aggregate']['next_channel_logits'], dim=-1)  # [batch, 3]

            # Get class predictions for per-timeframe
            direction = (direction_probs > 0.5).long()                           # [batch, 11]
            next_channel = next_channel_probs.argmax(dim=-1)                     # [batch, 11]

            # Get aggregate class predictions
            agg_direction = (agg_direction_probs > 0.5).long()                   # [batch, 1]
            agg_next_channel = agg_next_channel_probs.argmax(dim=-1, keepdim=True)  # [batch, 1]

            # Compute std from log_std
            duration_std = torch.exp(outputs['duration_log_std'])                # [batch, 11]
            agg_duration_std = torch.exp(outputs['aggregate']['duration_log_std'])  # [batch, 1]

            # Find recommended timeframe (highest confidence)
            best_tf_idx = outputs['confidence'].argmax(dim=1)  # [batch]

            return {
                # Per-timeframe predictions (for dashboard table)
                'per_tf': {
                    'duration_mean': outputs['duration_mean'],          # [batch, 11]
                    'duration_std': duration_std,                       # [batch, 11]
                    'direction': direction,                             # [batch, 11]
                    'direction_probs': direction_probs,                 # [batch, 11]
                    'next_channel': next_channel,                       # [batch, 11]
                    'next_channel_probs': next_channel_probs,           # [batch, 11, 3]
                    'confidence': outputs['confidence'],                # [batch, 11]
                },

                # Aggregate prediction (for simple signal)
                'aggregate': {
                    'duration_mean': outputs['aggregate']['duration_mean'],
                    'duration_std': agg_duration_std,
                    'direction': agg_direction,                         # [batch, 1] - binary
                    'direction_probs': agg_direction_probs,             # [batch, 1]
                    'next_channel': agg_next_channel,
                    'next_channel_probs': agg_next_channel_probs,
                    'confidence': outputs['aggregate']['confidence'],
                },

                # Metadata
                'best_tf_idx': best_tf_idx,                             # [batch] - which TF to use
                'attention_weights': outputs['attention_weights']       # [batch, 11, 11]
            }

    def get_num_parameters(self) -> Dict[str, int]:
        """
        Count parameters in each component.

        Returns:
            Dictionary with parameter counts
        """
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        # Handle both shared and separate head architectures
        if self.shared_heads:
            per_tf_heads_count = (count_params(self.per_tf_duration_head) +
                                  count_params(self.per_tf_direction_head) +
                                  count_params(self.per_tf_next_channel_head) +
                                  count_params(self.per_tf_confidence_head))
        else:
            per_tf_heads_count = (sum(count_params(h) for h in self.per_tf_duration_heads) +
                                  sum(count_params(h) for h in self.per_tf_direction_heads) +
                                  sum(count_params(h) for h in self.per_tf_next_channel_heads) +
                                  sum(count_params(h) for h in self.per_tf_confidence_heads))

        return {
            'tf_branches': sum(count_params(branch) for branch in self.tf_branches),
            'cross_tf_attention': count_params(self.cross_tf_attention),
            'per_tf_heads': per_tf_heads_count,
            'aggregate_heads': (count_params(self.agg_duration_head) +
                              count_params(self.agg_direction_head) +
                              count_params(self.agg_next_channel_head) +
                              count_params(self.agg_confidence_head)),
            'total': count_params(self)
        }


# =============================================================================
# Loss Functions
# =============================================================================

class HierarchicalLoss(nn.Module):
    """
    Combined loss for hierarchical model.

    Computes weighted combination of:
    1. Gaussian NLL for duration
    2. Cross-entropy for break direction
    3. Cross-entropy for next channel direction
    4. Calibration loss for confidence
    """

    def __init__(
        self,
        duration_weight: float = 1.0,
        break_direction_weight: float = 1.0,
        next_direction_weight: float = 1.0,
        confidence_weight: float = 0.5
    ):
        """
        Initialize loss.

        Args:
            duration_weight: Weight for duration loss
            break_direction_weight: Weight for break direction loss
            next_direction_weight: Weight for next direction loss
            confidence_weight: Weight for confidence calibration loss
        """
        super().__init__()

        self.duration_weight = duration_weight
        self.break_direction_weight = break_direction_weight
        self.next_direction_weight = next_direction_weight
        self.confidence_weight = confidence_weight

        self.ce_loss = nn.CrossEntropyLoss()

    def gaussian_nll_loss(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Gaussian negative log-likelihood loss with numerical stability guards.

        Loss = 0.5 * (log(2π) + 2*log(std) + ((target - mean) / std)^2)

        Includes safety clamping to prevent NaN/Inf:
        - Clamps std to prevent division by zero
        - Clamps squared error to max=1000 (prevents explosion from outliers)
        - Matches CombinedLoss clamping strategy for consistency

        Args:
            mean: Predicted mean [batch, 1]
            std: Predicted std [batch, 1]
            target: True values [batch, 1]

        Returns:
            Scalar loss
        """
        # Guard 1: Clamp std to prevent division by zero and extreme log values
        eps = 1e-6
        std_clamped = torch.clamp(std, min=eps, max=1000.0)

        # Guard 2: Clamp squared error to prevent explosion
        squared_error = ((target - mean) / std_clamped) ** 2
        squared_error = torch.clamp(squared_error, max=1000.0)

        # Compute NLL with clamped values
        var = std_clamped ** 2
        loss = 0.5 * (torch.log(2 * torch.pi * var) + squared_error)
        return loss.mean()

    def confidence_calibration_loss(
        self,
        confidence: torch.Tensor,
        correct: torch.Tensor
    ) -> torch.Tensor:
        """
        Brier score for confidence calibration.

        Encourages confidence to match actual correctness probability.

        Args:
            confidence: Predicted confidence [batch, 1]
            correct: Binary correctness indicator [batch, 1]

        Returns:
            Scalar loss
        """
        return ((confidence - correct) ** 2).mean()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss.

        Args:
            outputs: Model outputs dict
            targets: Dict with:
                - duration: [batch, 1] - true duration
                - direction: [batch] - true break direction (0/1)
                - next_channel: [batch] - true next channel direction (0/1/2)
                - correct: [batch, 1] - optional correctness indicator for calibration

        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Individual loss components
        """
        # Duration loss (Gaussian NLL) - convert log_std to std
        duration_std = torch.exp(outputs['duration_log_std'])
        duration_loss = self.gaussian_nll_loss(
            outputs['duration_mean'],
            duration_std,
            targets['duration'].unsqueeze(-1) if targets['duration'].dim() == 1 else targets['duration']
        )

        # Direction loss (break direction)
        direction_loss = self.ce_loss(
            outputs['direction_logits'],
            targets['direction']
        )

        # Next channel direction loss
        next_channel_loss = self.ce_loss(
            outputs['next_channel_logits'],
            targets['next_channel']
        )

        # Confidence calibration loss (if correctness provided)
        if 'correct' in targets:
            conf_loss = self.confidence_calibration_loss(
                outputs['confidence'],
                targets['correct'].unsqueeze(-1) if targets['correct'].dim() == 1 else targets['correct']
            )
        else:
            # Approximate correctness from predictions using weighted average
            # (not requiring BOTH to be correct for partial confidence)
            direction_probs = torch.sigmoid(outputs['direction_logits'])
            direction_correct = ((direction_probs > 0.5).long().squeeze(-1) == targets['direction']).float()
            next_correct = (outputs['next_channel_logits'].argmax(dim=-1) == targets['next_channel']).float()
            # Weighted average: break direction is more critical (60/40 split)
            overall_correct = (0.6 * direction_correct + 0.4 * next_correct).unsqueeze(-1)
            conf_loss = self.confidence_calibration_loss(outputs['confidence'], overall_correct)

        # Total weighted loss
        total_loss = (
            self.duration_weight * duration_loss +
            self.break_direction_weight * direction_loss +
            self.next_direction_weight * next_channel_loss +
            self.confidence_weight * conf_loss
        )

        # Loss components for logging
        loss_dict = {
            'total': total_loss.item(),
            'duration': duration_loss.item(),
            'direction': direction_loss.item(),
            'next_channel': next_channel_loss.item(),
            'confidence': conf_loss.item()
        }

        return total_loss, loss_dict


# =============================================================================
# Utility Functions
# =============================================================================

def create_model(
    hidden_dim: int = 64,
    cfc_units: int = 96,  # Must be > hidden_dim + 2
    num_attention_heads: int = 4,
    dropout: float = 0.1,
    shared_heads: bool = True,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> HierarchicalCfCModel:
    """
    Factory function to create and initialize model.

    Args:
        hidden_dim: Hidden dimension for branches
        cfc_units: CfC units per branch (must be > hidden_dim + 2)
        num_attention_heads: Attention heads
        dropout: Dropout probability
        shared_heads: If True, use single shared heads for all TFs (default).
                     If False, create separate prediction heads per timeframe.
        device: Device to place model on

    Returns:
        Initialized model on specified device
    """
    model = HierarchicalCfCModel(
        feature_config=FeatureConfig(),
        hidden_dim=hidden_dim,
        cfc_units=cfc_units,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        shared_heads=shared_heads
    )

    model = model.to(device)

    # Print parameter counts
    param_counts = model.get_num_parameters()
    print("Model Parameter Counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")

    return model


def create_loss(
    duration_weight: float = 1.0,
    break_direction_weight: float = 1.0,
    next_direction_weight: float = 1.0,
    confidence_weight: float = 0.5
) -> HierarchicalLoss:
    """
    Factory function to create loss.

    Args:
        duration_weight: Weight for duration loss
        break_direction_weight: Weight for break direction loss
        next_direction_weight: Weight for next direction loss
        confidence_weight: Weight for confidence loss

    Returns:
        Loss module
    """
    return HierarchicalLoss(
        duration_weight=duration_weight,
        break_direction_weight=break_direction_weight,
        next_direction_weight=next_direction_weight,
        confidence_weight=confidence_weight
    )


if __name__ == '__main__':
    """
    Test script to verify model architecture.
    """
    print("=" * 80)
    print("Hierarchical CfC Model for Multi-Timeframe Channel Prediction")
    print("=" * 80)

    # Create model
    print("\n[1] Creating model...")
    model = create_model(hidden_dim=64, cfc_units=96, num_attention_heads=4)

    # Create dummy input
    print(f"\n[2] Creating dummy input (batch_size=4, features={TOTAL_FEATURES})...")
    batch_size = 4
    x_dummy = torch.randn(batch_size, TOTAL_FEATURES)

    # Forward pass
    print("\n[3] Running forward pass...")
    outputs = model(x_dummy, return_attention=True)

    print("\nOutput shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")

    # Test prediction
    print("\n[4] Testing prediction mode...")
    predictions = model.predict(x_dummy)

    print("\nPrediction outputs:")
    for key, value in predictions.items():
        if 'attention' not in key:
            print(f"  {key}: {value.shape}")

    # Test loss
    print("\n[5] Testing loss computation...")
    loss_fn = create_loss()

    targets = {
        'duration': torch.randint(10, 200, (batch_size,)).float(),
        'direction': torch.randint(0, 2, (batch_size,)),
        'next_channel': torch.randint(0, 3, (batch_size,))
    }

    total_loss, loss_dict = loss_fn(outputs, targets)

    print("\nLoss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
