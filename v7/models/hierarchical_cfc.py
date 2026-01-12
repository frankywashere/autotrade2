"""
Hierarchical CfC Neural Network for Multi-Timeframe Channel Prediction (v7.0)

Architecture Overview:
====================
This model predicts channel break timing, direction, and post-break channel direction
using a hierarchical architecture that processes each timeframe independently before
combining insights across timeframes.

Input Features: 776 dimensions (TIMEFRAME-GROUPED ordering)
------------------------------------------------------------
The input tensor is ordered by timeframe, NOT alphabetically!
Each TF block contains: [tsla_{tf}, spy_{tf}, cross_{tf}] = 56 features
Followed by shared features at the end.

- TSLA per-TF features: 35 features × 11 timeframes = 385 total
  * Channel geometry: direction, position, width, slope, R² (18 base features)
  * Bounce metrics: count, cycles, bars since bounce
  * RSI: current, divergence, at upper/lower bounces
  * Exit tracking: exit counts, frequency, acceleration (10 features)
  * Break triggers: distance to longer TF boundaries (2 features)
  * Return tracking: return_rate, resilience, duration after return (5 features)

- SPY per-TF features: 11 features × 11 timeframes = 121 total
  * Channel geometry and position
  * Bounce metrics and RSI

- Cross-asset: 10 features × 11 timeframes = 110 total
  * TSLA position in SPY channels (8 features)
  * RSI correlation features (2 features)

- VIX regime: 21 features
  * Level, normalized level, trends, percentile, regime (6 basic)
  * VIX-channel interaction features (15 features)

- History features: 25 features (TSLA) + 25 features (SPY) = 50 total
  * Last 5 directions, durations, break directions (5+5+5 = 15)
  * Summary stats: avg duration, streak, counts, RSI stats (10)

- Alignment: 3 features
  * Direction match, both near upper/lower

- Events: 46 features (zeros if not provided)
  * Timing, earnings context, pre/post event drift

- Window Scores: 40 features
  * 8 windows × 5 metrics (bounce_count, r_squared, quality, alternation_ratio, width)

Architecture Flow:
=================
1. Input Layer (776 dims) → Feature Decomposition
   ├─ Per-TF features extracted for each of 11 timeframes (56 features each)
   ├─ Shared features (VIX, history, alignment, events, window_scores) extracted from end
   ├─ Window scores (last 40 of shared) used for per-TF window selection

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
    ALIGNMENT_FEATURES, EVENT_FEATURES, WINDOW_SCORE_FEATURES,
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
    - Per-TF block: [tsla_{tf}(35), spy_{tf}(11), cross_{tf}(10)] = 56 features
    - 11 timeframes × 56 = 616 per-TF features
    - Shared: [vix(21), tsla_history(25), spy_history(25), alignment(3), events(46), window_scores(40)] = 160 features
    - Total: 616 + 160 = 776 features
    """

    # Per-timeframe feature counts (from feature_ordering.py)
    tsla_per_tf: int = TSLA_PER_TF    # 35 (18 base + 10 exit_tracking + 2 break_trigger + 5 return_tracking)
    spy_per_tf: int = SPY_PER_TF      # 11 (channel metrics + RSI)
    cross_per_tf: int = CROSS_PER_TF  # 10 (TSLA-in-SPY containment + 2 RSI correlation)

    # Shared features (same across all TFs)
    vix_features: int = VIX_FEATURES                      # 21 (6 basic + 15 channel interaction)
    tsla_history_features: int = TSLA_HISTORY_FEATURES    # 25 (5+5+5 lists + 10 scalars)
    spy_history_features: int = SPY_HISTORY_FEATURES      # 25 (same structure)
    alignment_features: int = ALIGNMENT_FEATURES          # 3
    event_features: int = EVENT_FEATURES                  # 46 (zeros if not provided)
    window_score_features: int = WINDOW_SCORE_FEATURES    # 40 (8 windows x 5 metrics)

    # Number of timeframes
    n_timeframes: int = N_TIMEFRAMES  # 11: 5min, 15min, 30min, 1h, 2h, 3h, 4h, daily, weekly, monthly, 3month

    # Window selection constants
    num_windows: int = 8              # Number of windows per timeframe
    window_metrics: int = 5           # Metrics per window (bounce_count, r_squared, quality, alternation_ratio, width)

    @property
    def total_features(self) -> int:
        """Calculate total input dimension."""
        per_tf = (self.tsla_per_tf + self.spy_per_tf + self.cross_per_tf) * self.n_timeframes
        shared = (self.vix_features + self.tsla_history_features +
                  self.spy_history_features + self.alignment_features +
                  self.event_features + self.window_score_features)
        return per_tf + shared  # = (35+11+10)*11 + (21+25+25+3+46+40) = 616 + 160 = 776

    @property
    def shared_features(self) -> int:
        """Total shared feature dimension."""
        return (self.vix_features + self.tsla_history_features +
                self.spy_history_features + self.alignment_features +
                self.event_features + self.window_score_features)

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
# Utility Functions
# =============================================================================

def hazard_to_duration_stats(duration_hazard: torch.Tensor, num_bins: int = 50, max_duration: float = 100.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert hazard predictions from survival loss to duration mean and std.

    Args:
        duration_hazard: Hazard predictions [batch, num_bins] or [batch, num_timeframes, num_bins]
        num_bins: Number of hazard bins
        max_duration: Maximum duration value

    Returns:
        duration_mean: Expected duration [batch] or [batch, num_timeframes]
        duration_std: Duration standard deviation [batch] or [batch, num_timeframes]
    """
    # Ensure hazard is 3D: [batch, num_timeframes, num_bins]
    if duration_hazard.dim() == 2:
        duration_hazard = duration_hazard.unsqueeze(1)  # [batch, 1, num_bins]

    batch_size, num_tfs, num_bins = duration_hazard.shape

    # Convert logits to probabilities
    hazard_probs = torch.sigmoid(duration_hazard)  # [batch, num_tfs, num_bins]

    # Compute survival probabilities: S(t) = prod(1 - h(k)) for k <= t
    # S(t) represents probability of survival beyond time t
    survival_probs = torch.cumprod(1 - hazard_probs, dim=-1)  # [batch, num_tfs, num_bins]

    # Prepend 1.0 for S(0) = 1 (certain to survive at t=0)
    ones = torch.ones(batch_size, num_tfs, 1, device=duration_hazard.device)
    survival_probs = torch.cat([ones, survival_probs], dim=-1)  # [batch, num_tfs, num_bins+1]

    # Bin edges (time points)
    bin_width = max_duration / num_bins
    bin_edges = torch.linspace(0, max_duration, num_bins + 1, device=duration_hazard.device)  # [num_bins+1]

    # Expected duration: E[T] = sum of S(t) * bin_width
    # This is the area under the survival curve
    duration_mean = (survival_probs * bin_width).sum(dim=-1)  # [batch, num_tfs]

    # Compute variance: Var[T] = E[T^2] - E[T]^2
    # For discrete survival analysis:
    # E[T^2] = sum((2*t + bin_width) * S(t) * bin_width)
    # This is the correct formula for discrete-time second moment
    t_values = bin_edges.unsqueeze(0).unsqueeze(0)  # [1, 1, num_bins+1]
    second_moment = ((2 * t_values + bin_width) * survival_probs * bin_width).sum(dim=-1)  # [batch, num_tfs]
    variance = second_moment - duration_mean ** 2
    variance = torch.clamp(variance, min=0.0)  # Ensure non-negative (numerical stability)
    duration_std = torch.sqrt(variance)  # [batch, num_tfs]

    return duration_mean, duration_std


# =============================================================================
# Squeeze-and-Excitation Block (Feature Reweighting)
# =============================================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for feature reweighting.

    SE-blocks learn channel-wise (feature-wise) attention weights to emphasize
    important features and suppress less relevant ones. Originally from SENet
    (Hu et al., 2018) for image classification, adapted here for tabular data.

    Architecture:
    -------------
    For input x of shape [batch, channels]:

        x → Excitation(x) → scale
              ↓
        FC(channels, channels/r) → ReLU → FC(channels/r, channels) → Sigmoid
              ↓
        x * scale → output

    The "squeeze" operation (global pooling) is implicit for 1D features since
    we already have per-feature values (no spatial dimensions to pool over).

    Benefits for Financial Data:
    ---------------------------
    1. **Adaptive feature selection**: Learns which features matter for each sample
       (e.g., RSI might matter more when near overbought/oversold)
    2. **Lightweight**: Only adds ~4K params per branch vs ~4M for full attention
    3. **Non-disruptive**: Multiplicative reweighting preserves original feature space
    4. **Sample-dependent**: Different samples can have different feature importance

    Example:
    --------
        >>> se = SEBlock(channels=128, reduction_ratio=8)
        >>> x = torch.randn(32, 128)  # batch=32, features=128
        >>> out = se(x)               # [32, 128] - reweighted features
        >>> # Each feature in out is x[i] * learned_weight[i]
    """

    def __init__(self, channels: int, reduction_ratio: int = 8):
        """
        Initialize SE-block.

        Args:
            channels: Number of input features (hidden_dim in our case)
            reduction_ratio: Bottleneck reduction factor (default 8)
                            channels/reduction_ratio = intermediate size
                            e.g., 128/8 = 16 intermediate neurons
        """
        super().__init__()

        # Ensure we don't reduce too much (minimum 4 neurons in bottleneck)
        reduced_channels = max(channels // reduction_ratio, 4)

        # Excitation network: learns feature importance weights
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=True),
            nn.Sigmoid()  # Output in [0, 1] for multiplicative scaling
        )

        # Store for logging/debugging
        self.channels = channels
        self.reduced_channels = reduced_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SE reweighting to input features.

        Args:
            x: [batch, channels] - input features

        Returns:
            [batch, channels] - reweighted features (x * scale)
        """
        # Compute feature importance weights
        scale = self.excitation(x)  # [batch, channels], values in [0, 1]

        # Apply multiplicative reweighting
        return x * scale


# =============================================================================
# Temporal Convolutional Network (TCN) Block
# =============================================================================

class Chomp1d(nn.Module):
    """Remove trailing padding to maintain causality."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    """Single temporal block with dilated causal convolution."""
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int,
                 dilation: int, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)  # Remove future padding for causality
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network block.
    Stacks multiple TemporalBlocks with exponentially increasing dilation.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2,
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()

        layers = []
        num_levels = num_layers
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_dim if i == 0 else hidden_dim
            layers.append(
                TemporalBlock(in_channels, hidden_dim, kernel_size, dilation, dropout)
            )

        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim] or [batch, input_dim]
        Returns:
            [batch, seq_len, hidden_dim] or [batch, hidden_dim]
        """
        # Handle both 2D and 3D inputs
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, input_dim]
            squeeze_output = True

        # Conv1d expects [batch, channels, seq_len]
        x = x.transpose(1, 2)  # [batch, input_dim, seq_len]
        x = self.network(x)
        x = x.transpose(1, 2)  # [batch, seq_len, hidden_dim]

        if squeeze_output:
            x = x.squeeze(1)  # [batch, hidden_dim]
        return x


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
    4. Optionally processes through TCN for additional temporal modeling
    5. Outputs embedding for cross-TF attention

    The CfC layer is particularly well-suited here because:
    - It models continuous-time dynamics (natural for financial data)
    - Has strong extrapolation capabilities
    - Learns causal relationships between features
    - Compact parameter count vs LSTM/GRU

    The optional TCN block adds:
    - Dilated causal convolutions for multi-scale temporal patterns
    - Residual connections for gradient flow
    - Complementary temporal modeling to CfC
    """

    def __init__(
        self,
        per_tf_dim: int,
        shared_dim: int,
        hidden_dim: int = 64,
        cfc_units: int = 96,  # Must be > hidden_dim + 2
        dropout: float = 0.1,
        use_se_blocks: bool = False,
        se_reduction_ratio: int = 8,
        use_tcn: bool = False,
        tcn_channels: int = 64,
        tcn_kernel_size: int = 3,
        tcn_layers: int = 2,
    ):
        """
        Initialize timeframe branch.

        Args:
            per_tf_dim: Dimension of per-timeframe features (56 for TSLA+SPY+cross)
            shared_dim: Dimension of shared features (65 for VIX+history+alignment)
            hidden_dim: Output embedding dimension
            cfc_units: Number of CfC units (neurons in liquid network, must be > hidden_dim + 2)
            dropout: Dropout probability
            use_se_blocks: If True, add SE-block for feature reweighting (default: False)
            se_reduction_ratio: Reduction ratio for SE-block bottleneck (default: 8)
            use_tcn: If True, add TCN block after CfC for additional temporal modeling (default: False)
            tcn_channels: Number of channels in TCN hidden layers (default: 64)
            tcn_kernel_size: Kernel size for TCN convolutions (default: 3)
            tcn_layers: Number of temporal blocks in TCN (default: 2)
        """
        super().__init__()

        input_dim = per_tf_dim + shared_dim
        self.use_se_blocks = use_se_blocks
        self.use_tcn = use_tcn

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Optional SE-block for feature reweighting (applied before CfC)
        # This learns which hidden features are important per sample
        if use_se_blocks:
            self.se_block = SEBlock(channels=hidden_dim, reduction_ratio=se_reduction_ratio)

        # CfC (Liquid Neural Network) layer
        # AutoNCP creates a sparsely connected recurrent architecture
        # Note: cfc_units must be > hidden_dim + 2 for AutoNCP
        wiring = AutoNCP(cfc_units, hidden_dim)
        self.cfc = CfC(hidden_dim, wiring, batch_first=True)

        # Optional TCN block for additional temporal modeling (applied after CfC)
        if use_tcn:
            self.tcn_block = TCNBlock(
                input_dim=hidden_dim,
                hidden_dim=tcn_channels,
                num_layers=tcn_layers,
                kernel_size=tcn_kernel_size,
                dropout=dropout
            )
            # Projection to match output dim if needed
            if tcn_channels != hidden_dim:
                self.tcn_proj = nn.Linear(tcn_channels, hidden_dim)
            else:
                self.tcn_proj = nn.Identity()

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

        # Apply SE-block for feature reweighting (if enabled)
        # This learns which features in the hidden representation are important
        if self.use_se_blocks:
            x = self.se_block(x)  # [batch, hidden_dim] - reweighted

        # Add sequence dimension for CfC (expects 3D input)
        x = x.unsqueeze(1)  # [batch, 1, hidden_dim]

        # Process through CfC
        cfc_out, hx_new = self.cfc(x, hx)

        # Remove sequence dimension
        cfc_out = cfc_out.squeeze(1)  # [batch, hidden_dim]

        # Apply TCN if enabled
        if self.use_tcn:
            tcn_out = self.tcn_block(cfc_out)
            tcn_out = self.tcn_proj(tcn_out)
            # Residual connection: combine CfC and TCN outputs
            x = cfc_out + tcn_out
        else:
            x = cfc_out

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
# Window Selection
# =============================================================================

class PerTFWindowSelector(nn.Module):
    """
    Per-timeframe window selector head.

    Learns to select the optimal lookback window for each timeframe based on:
    - TF embedding (learned representation of market state for this TF)
    - Window scores (8 windows × 5 metrics = 40 features)

    During training, uses soft selection (softmax) to allow gradient flow.
    During inference, uses hard selection (argmax) for discrete window choice.

    The window scores contain per-window metrics:
    - bounce_count: Number of bounces in the window
    - r_squared: Channel quality metric
    - quality: Overall channel quality
    - alternation_ratio: Upper/lower bounce alternation
    - width: Channel width
    """

    def __init__(self, hidden_dim: int = 64, num_windows: int = 8, window_metrics: int = 5):
        """
        Initialize window selector.

        Args:
            hidden_dim: Dimension of TF embeddings
            num_windows: Number of window options (default 8)
            window_metrics: Number of metrics per window (default 5)
        """
        super().__init__()

        self.num_windows = num_windows
        self.window_metrics = window_metrics

        # Input: TF embedding (hidden_dim) + window_scores (num_windows * window_metrics)
        input_dim = hidden_dim + num_windows * window_metrics

        self.selector = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_windows)
        )

    def forward(
        self,
        tf_embedding: torch.Tensor,
        window_scores: torch.Tensor,
        hard_select: bool = False
    ) -> torch.Tensor:
        """
        Compute window selection logits.

        Args:
            tf_embedding: [batch, hidden_dim] - embedding for one TF
            window_scores: [batch, num_windows * window_metrics] - flattened window scores

        Returns:
            window_logits: [batch, num_windows] - selection logits
        """
        x = torch.cat([tf_embedding, window_scores], dim=-1)
        logits = self.selector(x)
        return logits


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

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_hazard_bins: int = 0):
        """
        Initialize duration head.

        Args:
            input_dim: Input dimension (from attention context)
            hidden_dim: Hidden layer dimension
            num_hazard_bins: Number of bins for hazard prediction (0 = disabled)
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

        # Optional hazard head for survival analysis
        self.num_hazard_bins = num_hazard_bins
        if num_hazard_bins > 0:
            self.hazard_head = nn.Linear(hidden_dim // 2, num_hazard_bins)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: [batch_size, input_dim]

        Returns:
            mean: [batch_size, 1] - predicted mean duration
            log_std: [batch_size, 1] - predicted log(standard deviation)
            hazard: [batch_size, num_hazard_bins] or None - hazard logits for survival analysis
        """
        h = self.net(x)

        mean = self.mean_head(h)
        mean = F.softplus(mean) + 1.0  # Ensure positive, minimum 1 bar

        log_std = self.logstd_head(h)
        # Bounded for numerical stability. Range [-2, 4] gives std in [0.14, 54.6]
        # which is appropriate for duration data ranging from 1 to 500 bars
        log_std = log_std.clamp(-2, 4)

        # Compute hazard if enabled
        hazard = None
        if self.num_hazard_bins > 0:
            hazard = self.hazard_head(h)

        return mean, log_std, hazard


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


class TriggerTFHead(nn.Module):
    """
    Predicts which longer timeframe boundary triggered the channel break.

    21-class classification:
    - Class 0: NO_TRIGGER (break occurred without hitting a longer TF boundary)
    - Classes 1-20: TF + boundary combinations
      * 1-2: 15min (upper, lower)
      * 3-4: 30min (upper, lower)
      * 5-6: 1h (upper, lower)
      * 7-8: 2h (upper, lower)
      * 9-10: 3h (upper, lower)
      * 11-12: 4h (upper, lower)
      * 13-14: daily (upper, lower)
      * 15-16: weekly (upper, lower)
      * 17-18: monthly (upper, lower)
      * 19-20: 3month (upper, lower)

    This is an AGGREGATE-ONLY prediction (not per-TF) since it answers
    "which longer TF triggered the break for this channel?"
    """

    # Class constants for decoding predictions
    NUM_CLASSES = 21
    CLASS_NAMES = [
        'NO_TRIGGER',
        '15min_upper', '15min_lower',
        '30min_upper', '30min_lower',
        '1h_upper', '1h_lower',
        '2h_upper', '2h_lower',
        '3h_upper', '3h_lower',
        '4h_upper', '4h_lower',
        'daily_upper', 'daily_lower',
        'weekly_upper', 'weekly_lower',
        'monthly_upper', 'monthly_lower',
        '3month_upper', '3month_lower',
    ]

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Initialize trigger TF head.

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
            nn.Linear(hidden_dim // 2, self.NUM_CLASSES)  # 21 classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch_size, input_dim]

        Returns:
            logits: [batch_size, 21] - class logits for trigger TF
        """
        return self.net(x)

    @staticmethod
    def decode_prediction(class_idx: int) -> str:
        """
        Decode a class index to human-readable trigger TF string.

        Args:
            class_idx: Predicted class (0-20)

        Returns:
            String like 'NO_TRIGGER', '1h_upper', 'daily_lower', etc.
        """
        if 0 <= class_idx < TriggerTFHead.NUM_CLASSES:
            return TriggerTFHead.CLASS_NAMES[class_idx]
        return f'UNKNOWN_{class_idx}'


class MultiResolutionHead(nn.Module):
    """
    Multi-resolution prediction head that makes predictions at multiple temporal scales.
    Combines predictions via learned weighting or attention.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_resolutions: int = 3,
                 head_type: str = 'duration'):
        super().__init__()
        self.num_resolutions = num_resolutions
        self.head_type = head_type

        # Create separate heads for each resolution
        self.resolution_heads = nn.ModuleList()
        for i in range(num_resolutions):
            if head_type == 'duration':
                self.resolution_heads.append(DurationHead(input_dim, hidden_dim))
            elif head_type == 'direction':
                self.resolution_heads.append(DirectionHead(input_dim, hidden_dim))
            elif head_type == 'next_channel':
                self.resolution_heads.append(NextChannelDirectionHead(input_dim, hidden_dim))
            elif head_type == 'confidence':
                self.resolution_heads.append(ConfidenceHead(input_dim, hidden_dim))

        # Resolution-specific feature transforms
        self.resolution_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            for _ in range(num_resolutions)
        ])

        # Learned resolution weighting
        self.resolution_weights = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_resolutions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch, input_dim] - input features
        Returns:
            Dictionary with per-resolution predictions and combined prediction
        """
        outputs = {}

        # Get predictions at each resolution
        resolution_outputs = []
        for i, (transform, head) in enumerate(zip(self.resolution_transforms, self.resolution_heads)):
            res_features = transform(x)
            res_pred = head(res_features)
            resolution_outputs.append(res_pred)

            if self.head_type == 'duration':
                outputs[f'res{i}_mean'] = res_pred[0]
                outputs[f'res{i}_log_std'] = res_pred[1]
            else:
                outputs[f'res{i}_pred'] = res_pred

        # Compute resolution weights
        weights = self.resolution_weights(x)  # [batch, num_resolutions]
        outputs['resolution_weights'] = weights

        # Combine predictions (weighted average)
        if self.head_type == 'duration':
            # Weighted combination of means
            means = torch.stack([outputs[f'res{i}_mean'] for i in range(self.num_resolutions)], dim=-1)
            combined_mean = (means * weights.unsqueeze(1)).sum(dim=-1)
            outputs['combined_mean'] = combined_mean

            # Combine log_stds (use mean for simplicity)
            log_stds = torch.stack([outputs[f'res{i}_log_std'] for i in range(self.num_resolutions)], dim=-1)
            combined_log_std = (log_stds * weights.unsqueeze(1)).sum(dim=-1)
            outputs['combined_log_std'] = combined_log_std
        else:
            # Weighted combination of predictions
            preds = torch.stack([outputs[f'res{i}_pred'] for i in range(self.num_resolutions)], dim=-1)
            combined = (preds * weights.unsqueeze(1)).sum(dim=-1)
            outputs['combined_pred'] = combined

        return outputs


# =============================================================================
# Shared Window Encoder
# =============================================================================

class SharedWindowEncoder(nn.Module):
    """
    Encodes features from a single window into an embedding.

    This encoder is shared across all 8 windows, processing each window's
    761-dim features into a compact embedding. The shared weights ensure
    the model learns a consistent representation across different window sizes.

    Architecture:
    ------------
    The encoder uses a two-layer MLP with LayerNorm and GELU activation:

        Input (761 dims)
            -> Linear(761, 256)
            -> LayerNorm(256)
            -> GELU
            -> Dropout(0.1)
            -> Linear(256, embed_dim)
            -> LayerNorm(embed_dim)
            -> Output (embed_dim dims)

    Design Rationale:
    ----------------
    1. **Shared weights**: Using the same encoder for all windows enables:
       - Transfer learning across window sizes
       - Consistent feature extraction regardless of lookback period
       - Reduced parameter count (1 encoder vs 8 separate encoders)

    2. **LayerNorm**: Stabilizes training and helps with varying input distributions
       across different window sizes (smaller windows may have different statistics
       than larger ones).

    3. **GELU activation**: Smoother than ReLU, tends to perform better in
       transformer-style architectures.

    4. **Dropout**: Regularization to prevent overfitting, especially important
       since the same encoder processes all window types.

    Usage:
    ------
    The SharedWindowEncoder is used in the end-to-end window selection pipeline:

    1. For each of the 8 windows, extract 761-dim features
    2. Pass each window's features through this shared encoder
    3. Get 8 embeddings of shape [batch, embed_dim]
    4. Use these embeddings for window selection or downstream processing

    Example:
    --------
        >>> encoder = SharedWindowEncoder(input_dim=761, embed_dim=128)
        >>>
        >>> # Process features from one window
        >>> window_features = torch.randn(32, 761)  # batch=32
        >>> embedding = encoder(window_features)     # [32, 128]
        >>>
        >>> # Process all 8 windows
        >>> all_window_features = torch.randn(32, 8, 761)  # [batch, windows, features]
        >>> embeddings = []
        >>> for w in range(8):
        ...     emb = encoder(all_window_features[:, w, :])
        ...     embeddings.append(emb)
        >>> all_embeddings = torch.stack(embeddings, dim=1)  # [32, 8, 128]

    Attributes:
    ----------
    input_dim : int
        Dimension of input features (default: 776, matching TOTAL_FEATURES)
    embed_dim : int
        Dimension of output embeddings (default: 128)
    encoder : nn.Sequential
        The MLP encoder network
    """

    def __init__(self, input_dim: int = 776, embed_dim: int = 128):
        """
        Initialize the SharedWindowEncoder.

        Parameters:
        ----------
        input_dim : int, optional
            Dimension of input features per window. Default is 776, which matches
            the canonical TOTAL_FEATURES from v7/features/feature_ordering.py.
            This includes:
            - Per-TF features: 616 (56 features x 11 timeframes)
            - Shared features: 160 (VIX, history, alignment, events, window_scores)

        embed_dim : int, optional
            Dimension of output embeddings. Default is 128, providing a good
            balance between expressiveness and computational efficiency.
            Common choices:
            - 64: Lightweight, suitable for limited compute
            - 128: Balanced (default)
            - 256: Higher capacity, may need more regularization

        Note:
        ----
        The intermediate hidden dimension of 256 is hardcoded as it provides
        a good compression ratio (776 -> 256 -> embed_dim) while being
        computationally efficient.
        """
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode window features into a compact embedding.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor of shape [batch, input_dim] containing features
            from a single window. The features should follow the canonical
            ordering from v7/features/feature_ordering.py:
            - Indices 0-615: Per-timeframe features (11 TFs x 56 features)
            - Indices 616-760: Shared features

        Returns:
        -------
        torch.Tensor
            Output tensor of shape [batch, embed_dim] containing the
            encoded representation of the window.

        Raises:
        ------
        RuntimeError
            If input dimension does not match self.input_dim (implicit via Linear layer)

        Example:
        --------
            >>> encoder = SharedWindowEncoder(input_dim=761, embed_dim=128)
            >>> features = torch.randn(16, 761)
            >>> embedding = encoder(features)
            >>> print(embedding.shape)
            torch.Size([16, 128])
        """
        return self.encoder(x)


# =============================================================================
# Differentiable Window Selector (Phase 2b)
# =============================================================================

class DifferentiableWindowSelector(nn.Module):
    """
    Produces soft window selection probabilities based on window embeddings.

    This component enables end-to-end training where the duration prediction loss
    backpropagates through the window selection mechanism, allowing the model to
    learn which window is most predictive for each sample.

    Architecture:
    -------------
    The selector uses a context-based attention mechanism:

    1. **Context Aggregation**: Mean-pool all window embeddings to create a
       global context vector representing the overall state across all windows.

    2. **Per-Window Scoring**: For each window, concatenate its embedding with
       the context and pass through a scoring network to produce a scalar score.

    3. **Selection Mechanism**: Convert scores to probabilities via:
       - Softmax with temperature (default training)
       - Gumbel-softmax (optional, for differentiable discrete selection)
       - Argmax (inference, for hard discrete selection)

    4. **Weighted Combination**: Use probabilities to compute weighted sum of
       window embeddings: selected = sum(prob_i * embedding_i)

    Gradient Flow:
    -------------
    The key insight is that all operations are differentiable:

        Duration Loss
             |
             v
        duration_pred = DurationHead(selected_embedding)
             |
             v
        selected_embedding = einsum('bw,bwd->bd', probs, embeddings)
             |
             +---> probs = softmax(scores / temperature)
             |            |
             |            +-- gradient flows to score_net params
             |
             +---> embeddings (from SharedWindowEncoder)
                         |
                         +-- gradient flows to encoder params

    This allows the model to learn which windows minimize duration prediction error,
    rather than relying on heuristic window selection.

    Training Strategies:
    -------------------
    1. **Soft Selection (default)**: Standard softmax produces a probability
       distribution. All windows contribute to the output, weighted by probability.
       Most stable for gradient flow.

    2. **Gumbel-Softmax**: Adds Gumbel noise before softmax, approximating
       categorical sampling while maintaining differentiability. Useful when
       you want more discrete-like behavior during training.

    3. **Temperature Annealing**: Start with high temperature (soft, exploratory)
       and anneal to low temperature (sharp, near-discrete). This encourages
       exploration early and commitment late in training.

    Regularization:
    --------------
    The `compute_entropy()` method returns the entropy of the selection distribution:
    - High entropy = uncertain (probabilities spread across windows)
    - Low entropy = confident (probability concentrated on one window)

    Use entropy as a regularization term:
    - Minimize entropy to encourage decisive selection
    - Or maximize entropy early in training for exploration

    Example:
    --------
        >>> selector = DifferentiableWindowSelector(embed_dim=128, num_windows=8)
        >>>
        >>> # Training: soft selection
        >>> window_embeddings = torch.randn(32, 8, 128)  # [batch, windows, embed]
        >>> selected, probs = selector(window_embeddings, hard_select=False)
        >>> print(selected.shape)  # [32, 128]
        >>> print(probs.shape)     # [32, 8]
        >>>
        >>> # Inference: hard selection
        >>> selected, probs = selector(window_embeddings, hard_select=True)
        >>> print(probs.sum(dim=-1))  # All 1.0 (one-hot)
        >>>
        >>> # Entropy for regularization
        >>> entropy = selector.compute_entropy(probs)
        >>> print(entropy.shape)  # [32]

    Attributes:
    ----------
    embed_dim : int
        Dimension of window embeddings (default: 128)
    num_windows : int
        Number of windows to select from (default: 8)
    temperature : float
        Softmax temperature (default: 1.0). Lower = sharper distribution.
    use_gumbel : bool
        If True, use Gumbel-softmax during training (default: False)
    context_proj : nn.Linear
        Projects mean-pooled context
    score_net : nn.Sequential
        Scores each window given context
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_windows: int = 8,
        temperature: float = 1.0,
        use_gumbel: bool = False,
    ):
        """
        Initialize the DifferentiableWindowSelector.

        Parameters:
        ----------
        embed_dim : int, optional
            Dimension of window embeddings from SharedWindowEncoder. Default is 128.

        num_windows : int, optional
            Number of windows to select from. Default is 8, matching STANDARD_WINDOWS:
            [10, 20, 30, 40, 50, 75, 100, 150]

        temperature : float, optional
            Softmax temperature controlling distribution sharpness. Default is 1.0.
            - temperature > 1.0: softer distribution (more uniform)
            - temperature < 1.0: sharper distribution (more peaked)
            - temperature -> 0: approaches argmax

            Typical annealing schedule:
            - Start: 5.0 (exploratory)
            - End: 0.1 (near-discrete)

        use_gumbel : bool, optional
            If True, use Gumbel-softmax during training for stochastic discrete
            selection while maintaining differentiability. Default is False.

            Gumbel-softmax adds random noise to logits before softmax:
                probs = softmax((logits + gumbel_noise) / temperature)

            This approximates sampling from a categorical distribution.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_windows = num_windows
        self.temperature = temperature
        self.use_gumbel = use_gumbel

        # Context aggregation: project mean-pooled embeddings
        # This creates a "global view" of all windows to inform selection
        self.context_proj = nn.Linear(embed_dim, embed_dim)

        # Per-window scoring network
        # Input: [context (embed_dim), window_embed (embed_dim)] = embed_dim * 2
        # Output: scalar score for this window
        self.score_net = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        window_embeddings: torch.Tensor,
        hard_select: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute soft or hard window selection.

        This method produces a weighted combination of window embeddings based on
        learned selection probabilities. During training, gradients flow from the
        downstream loss (e.g., duration prediction) back through the selection
        weights to the scoring network.

        Parameters:
        ----------
        window_embeddings : torch.Tensor
            Tensor of shape [batch, num_windows, embed_dim] containing embeddings
            for each window, produced by SharedWindowEncoder.

        hard_select : bool, optional
            Selection mode. Default is False.
            - False: Soft selection (training) - returns weighted combination
            - True: Hard selection (inference) - returns one-hot selection

        Returns:
        -------
        selected_embedding : torch.Tensor
            Tensor of shape [batch, embed_dim] containing the selected/weighted
            window embedding. This feeds into downstream prediction heads.

        selection_probs : torch.Tensor
            Tensor of shape [batch, num_windows] containing selection probabilities.
            - Soft selection: probabilities sum to 1
            - Hard selection: one-hot vectors

        Notes:
        -----
        - During training (hard_select=False), all windows contribute to the output
          proportionally to their selection probability. This allows gradients to
          flow to the scoring network for all windows.

        - During inference (hard_select=True), only the highest-probability window
          contributes. This provides a discrete, interpretable window choice.

        - When use_gumbel=True and training, Gumbel noise is added to logits before
          softmax, approximating categorical sampling while remaining differentiable.

        Example:
        --------
            >>> selector = DifferentiableWindowSelector(embed_dim=128)
            >>> embeddings = torch.randn(16, 8, 128)
            >>>
            >>> # Soft selection for training
            >>> selected, probs = selector(embeddings, hard_select=False)
            >>> loss = criterion(prediction_head(selected), target)
            >>> loss.backward()  # Gradients flow to selector
            >>>
            >>> # Hard selection for inference
            >>> with torch.no_grad():
            ...     selected, probs = selector(embeddings, hard_select=True)
            ...     chosen_window = probs.argmax(dim=-1)  # Discrete choice
        """
        batch_size = window_embeddings.size(0)
        device = window_embeddings.device

        # Step 1: Create context from all windows (mean pooling)
        # context: [batch, embed_dim]
        context = window_embeddings.mean(dim=1)
        context = self.context_proj(context)

        # Step 2: Score each window using context + window embedding
        # This vectorized approach is more efficient than a loop
        # Expand context to match window dimension
        context_expanded = context.unsqueeze(1).expand(-1, self.num_windows, -1)
        # context_expanded: [batch, num_windows, embed_dim]

        # Concatenate context with each window embedding
        combined = torch.cat([context_expanded, window_embeddings], dim=-1)
        # combined: [batch, num_windows, embed_dim * 2]

        # Score all windows at once
        logits = self.score_net(combined).squeeze(-1)
        # logits: [batch, num_windows]

        # Step 3: Convert scores to selection probabilities
        if hard_select:
            # Discrete selection (inference)
            # Create one-hot vectors from argmax indices
            indices = logits.argmax(dim=-1)  # [batch]
            selection_probs = F.one_hot(indices, self.num_windows).float()
            # selection_probs: [batch, num_windows] (one-hot)

        elif self.use_gumbel and self.training:
            # Gumbel-softmax for differentiable discrete selection
            # This adds noise to encourage exploration while remaining differentiable
            selection_probs = F.gumbel_softmax(
                logits,
                tau=self.temperature,
                hard=False,  # Keep soft for gradient flow
                dim=-1
            )
            # selection_probs: [batch, num_windows]

        else:
            # Soft selection (standard training)
            selection_probs = F.softmax(logits / self.temperature, dim=-1)
            # selection_probs: [batch, num_windows]

        # Step 4: Weighted combination of embeddings
        # This is the key differentiable operation:
        # selected = sum_w(prob_w * embedding_w)
        # Equivalent to: bmm(probs.unsqueeze(1), embeddings).squeeze(1)
        selected_embedding = torch.einsum('bw,bwd->bd', selection_probs, window_embeddings)
        # selected_embedding: [batch, embed_dim]

        return selected_embedding, selection_probs

    def compute_entropy(self, selection_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute the entropy of selection probabilities.

        Entropy measures the uncertainty in window selection:
        - High entropy = model is uncertain (probabilities spread evenly)
        - Low entropy = model is confident (probability concentrated on one window)

        This can be used for regularization:
        - **Minimize entropy** to encourage decisive selection (commitment)
        - **Maximize entropy** early in training for exploration

        The entropy is computed as:
            H = -sum(p * log(p))

        where p is the selection probability for each window.

        Parameters:
        ----------
        selection_probs : torch.Tensor
            Tensor of shape [batch, num_windows] containing selection probabilities.
            Each row should sum to 1.

        Returns:
        -------
        entropy : torch.Tensor
            Tensor of shape [batch] containing per-sample entropy values.

            Bounds:
            - Minimum: 0 (one-hot, perfectly confident)
            - Maximum: log(num_windows) = log(8) = 2.08 (uniform, maximally uncertain)

        Example:
        --------
            >>> selector = DifferentiableWindowSelector(num_windows=8)
            >>> probs = torch.softmax(torch.randn(16, 8), dim=-1)
            >>> entropy = selector.compute_entropy(probs)
            >>> print(entropy.shape)  # [16]
            >>>
            >>> # Use for regularization (minimize entropy)
            >>> entropy_loss = entropy.mean()
            >>> total_loss = prediction_loss + 0.1 * entropy_loss
        """
        # Add small epsilon to prevent log(0) = -inf
        eps = 1e-10
        log_probs = (selection_probs + eps).log()
        entropy = -(selection_probs * log_probs).sum(dim=-1)
        # entropy: [batch]
        return entropy

    def get_selection_confidence(self, selection_probs: torch.Tensor) -> torch.Tensor:
        """
        Convert selection probabilities to a confidence score.

        This provides a normalized confidence measure [0, 1] based on how
        concentrated the selection distribution is:
        - Confidence 1.0 = one-hot (certain about window choice)
        - Confidence 0.0 = uniform (uncertain, all windows equally likely)

        The confidence is computed as:
            confidence = 1 - (entropy / max_entropy)

        where max_entropy = log(num_windows).

        Parameters:
        ----------
        selection_probs : torch.Tensor
            Tensor of shape [batch, num_windows] containing selection probabilities.

        Returns:
        -------
        confidence : torch.Tensor
            Tensor of shape [batch] containing confidence scores in [0, 1].

        Example:
        --------
            >>> selector = DifferentiableWindowSelector(num_windows=8)
            >>>
            >>> # High confidence (peaked distribution)
            >>> probs_peaked = torch.tensor([[0.9, 0.05, 0.02, 0.01, 0.01, 0.005, 0.003, 0.002]])
            >>> conf = selector.get_selection_confidence(probs_peaked)
            >>> print(conf)  # ~0.8
            >>>
            >>> # Low confidence (uniform distribution)
            >>> probs_uniform = torch.ones(1, 8) / 8
            >>> conf = selector.get_selection_confidence(probs_uniform)
            >>> print(conf)  # ~0.0
        """
        entropy = self.compute_entropy(selection_probs)
        max_entropy = torch.log(torch.tensor(float(self.num_windows), device=entropy.device))
        confidence = 1.0 - (entropy / max_entropy)
        return confidence.clamp(0.0, 1.0)

    def set_temperature(self, temperature: float) -> None:
        """
        Update the softmax temperature.

        This allows dynamic temperature annealing during training:
        - Start with high temperature (e.g., 5.0) for exploration
        - Anneal to low temperature (e.g., 0.1) for commitment

        Parameters:
        ----------
        temperature : float
            New temperature value. Must be positive.

        Example:
        --------
            >>> selector = DifferentiableWindowSelector(temperature=5.0)
            >>>
            >>> # Anneal temperature during training
            >>> for epoch in range(100):
            ...     progress = epoch / 100
            ...     temp = 5.0 * (0.1 / 5.0) ** progress  # Exponential decay
            ...     selector.set_temperature(temp)
            ...     # ... training step ...

        Raises:
        ------
        ValueError
            If temperature is not positive.
        """
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.temperature = temperature


# =============================================================================
# Full Hierarchical Model
# =============================================================================

class HierarchicalCfCModel(nn.Module):
    """
    Complete hierarchical CfC model for multi-timeframe channel prediction.

    Architecture:
    1. Decompose 761-dim input into per-TF (56 features each) and shared features (145)
    2. Process each TF through dedicated CfC branch
    3. Attend over TF embeddings to create unified context
    4. Predict duration, break direction, next channel direction, confidence
    5. Select optimal window per timeframe using PerTFWindowSelector

    CRITICAL: Input must use TIMEFRAME-GROUPED ordering from feature_ordering.py!
    - Indices 0-55: TF0 (5min) = tsla_5min(35) + spy_5min(11) + cross_5min(10)
    - Indices 56-111: TF1 (15min) = tsla_15min(35) + spy_15min(11) + cross_15min(10)
    - ... (11 timeframes total)
    - Indices 616-760: Shared = vix(6) + tsla_history(25) + spy_history(25) + alignment(3) + events(46) + window_scores(40)

    Window Selection:
    - Last 40 features of shared are window_scores (8 windows x 5 metrics)
    - PerTFWindowSelector uses TF embeddings + window_scores to select optimal window
    - Training: soft selection (softmax) for gradient flow
    - Inference: hard selection (argmax) for discrete choice

    This design allows the model to:
    - Learn timeframe-specific temporal dynamics
    - Adaptively weight timeframes based on context
    - Share prediction logic across scales
    - Provide calibrated uncertainty estimates
    - Dynamically select optimal lookback windows
    """

    def __init__(
        self,
        feature_config: Optional[FeatureConfig] = None,
        hidden_dim: int = 128,  # v9.2: Widened from 64 to preserve more feature info
        cfc_units: int = 192,   # v9.2: Increased from 96 (must be > hidden_dim + 2)
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        shared_heads: bool = True,
        use_se_blocks: bool = False,
        se_reduction_ratio: int = 8,
        use_multi_resolution: bool = False,
        resolution_levels: int = 3,
        use_tcn: bool = False,
        tcn_channels: int = 64,
        tcn_kernel_size: int = 3,
        tcn_layers: int = 2,
        num_hazard_bins: int = 0,
        max_duration: float = 100.0,
    ):
        """
        Initialize hierarchical CfC model.

        Args:
            feature_config: Feature dimension configuration
            hidden_dim: Hidden dimension for branches and heads (v9.2: default 128, was 64)
            cfc_units: Number of units in each CfC layer (must be > hidden_dim + 2)
            num_attention_heads: Number of attention heads
            dropout: Dropout probability
            shared_heads: If True (default), use single shared heads for all TFs.
                         If False, create separate heads per timeframe (11x more head params).
            use_se_blocks: If True, add SE-blocks to each TF branch for feature reweighting.
                          SE-blocks learn which features matter per sample. (default: False)
            se_reduction_ratio: Reduction ratio for SE-block bottleneck (default: 8).
                               hidden_dim/ratio = bottleneck size (e.g., 128/8 = 16 neurons)
            use_multi_resolution: If True, add multi-resolution prediction heads that make
                                 predictions at multiple temporal scales. (default: False)
            resolution_levels: Number of resolution levels for multi-resolution heads (default: 3)
            use_tcn: If True, add TCN block to each TF branch for additional temporal modeling (default: False)
            tcn_channels: Number of channels in TCN hidden layers (default: 64)
            tcn_kernel_size: Kernel size for TCN convolutions (default: 3)
            tcn_layers: Number of temporal blocks in TCN (default: 2)
            num_hazard_bins: Number of bins for hazard prediction in SurvivalLoss (default: 0 = disabled)
            max_duration: Maximum duration value for survival loss conversion (default: 100.0)
        """
        super().__init__()

        self.config = feature_config or FeatureConfig()
        self.hidden_dim = hidden_dim
        self.n_timeframes = self.config.n_timeframes
        self.shared_heads = shared_heads
        self.use_se_blocks = use_se_blocks
        self.use_tcn = use_tcn
        self.num_hazard_bins = num_hazard_bins
        self.max_duration = max_duration

        # Validate total input dimension against canonical value from feature_ordering
        assert self.config.total_features == TOTAL_FEATURES, \
            f"Expected {TOTAL_FEATURES} features, got {self.config.total_features}"

        # Validate CfC configuration
        assert cfc_units > hidden_dim + 2, \
            f"cfc_units ({cfc_units}) must be > hidden_dim + 2 ({hidden_dim + 2})"

        # Create TF branches (11 parallel CfC processors)
        # Each branch optionally includes SE-block for feature reweighting and/or TCN
        self.tf_branches = nn.ModuleList([
            TFBranch(
                per_tf_dim=self.config.per_tf_features,
                shared_dim=self.config.shared_features,
                hidden_dim=hidden_dim,
                cfc_units=cfc_units,
                dropout=dropout,
                use_se_blocks=use_se_blocks,
                se_reduction_ratio=se_reduction_ratio,
                use_tcn=use_tcn,
                tcn_channels=tcn_channels,
                tcn_kernel_size=tcn_kernel_size,
                tcn_layers=tcn_layers,
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

        # v9.2: Project context back to hidden_dim for per-TF enrichment
        # This allows cross-TF attention to influence per-TF duration predictions
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # v9.2.1: LayerNorm after residual addition to fix scale mismatch
        # (embedding is not normalized, context_projected is - this balances them)
        self.enriched_norm = nn.LayerNorm(hidden_dim)

        # Per-timeframe prediction heads
        if self.shared_heads:
            # SHARED: Single set of heads for all TFs (default, fewer parameters)
            self.per_tf_duration_head = DurationHead(hidden_dim, hidden_dim // 2, num_hazard_bins)
            self.per_tf_direction_head = DirectionHead(hidden_dim, hidden_dim // 2)
            self.per_tf_next_channel_head = NextChannelDirectionHead(hidden_dim, hidden_dim // 2)
            self.per_tf_confidence_head = ConfidenceHead(hidden_dim, hidden_dim // 2)
        else:
            # SEPARATE: 11 separate heads per prediction type (more parameters, TF-specific)
            self.per_tf_duration_heads = nn.ModuleList([
                DurationHead(hidden_dim, hidden_dim // 2, num_hazard_bins)
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
        self.agg_duration_head = DurationHead(context_dim, hidden_dim, num_hazard_bins)
        self.agg_direction_head = DirectionHead(context_dim, hidden_dim)
        self.agg_next_channel_head = NextChannelDirectionHead(context_dim, hidden_dim)
        self.agg_confidence_head = ConfidenceHead(context_dim, hidden_dim)
        self.agg_trigger_tf_head = TriggerTFHead(context_dim, hidden_dim)  # v9.0.0: 21-class trigger TF

        # Multi-resolution prediction heads (optional)
        self.use_multi_resolution = use_multi_resolution
        if use_multi_resolution:
            self.multi_res_duration = MultiResolutionHead(
                context_dim, hidden_dim // 2, resolution_levels, 'duration'
            )
            self.multi_res_direction = MultiResolutionHead(
                context_dim, hidden_dim // 2, resolution_levels, 'direction'
            )

        # Per-TF window selector (shared across all TFs)
        self.window_selector = PerTFWindowSelector(
            hidden_dim=hidden_dim,
            num_windows=self.config.num_windows,
            window_metrics=self.config.window_metrics
        )

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
            x: [batch_size, 761] - full feature vector (TIMEFRAME-GROUPED ordering!)
            return_attention: If True, return attention weights

        Returns:
            Dictionary with:
                - duration_mean: [batch_size, 11]
                - duration_log_std: [batch_size, 11]
                - duration_hazard: [batch_size, 11, num_hazard_bins] or None (if num_hazard_bins > 0)
                - direction_logits: [batch_size, 11]
                - next_channel_logits: [batch_size, 11, 3]
                - confidence: [batch_size, 11]
                - window_logits: [batch_size, 11, 8] - per-TF window selection logits
                - window_probs: [batch_size, 11, 8] - softmax of window_logits
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
        shared_features = x[:, shared_start:shared_end]  # [batch, 145]

        # Extract window scores (last 40 features of shared)
        # Window scores: 8 windows × 5 metrics = 40 features
        window_scores_start = shared_end - self.config.window_score_features
        window_scores = x[:, window_scores_start:shared_end]  # [batch, 40]

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
        # CROSS-TF ATTENTION (v9.2: moved BEFORE per-TF predictions)
        # =====================================================================
        # Compute cross-TF attention to get global context
        context, attn_weights = self.cross_tf_attention(tf_embeddings_stacked)

        # v9.2: Project context to hidden_dim for per-TF enrichment
        # This allows duration predictions to leverage cross-TF information
        context_projected = self.context_proj(context)  # [batch, hidden_dim]

        # =====================================================================
        # PER-TIMEFRAME PREDICTIONS (one prediction per TF)
        # =====================================================================
        # v9.2: Duration predictions use ENRICHED embeddings (TF + cross-TF context)
        # Other heads use raw embeddings to minimize parameter changes
        per_tf_durations_mean = []
        per_tf_durations_log_std = []
        per_tf_hazards = []  # Optional hazard outputs for SurvivalLoss
        per_tf_directions = []
        per_tf_next_channels = []
        per_tf_confidences = []

        for tf_idx, embedding in enumerate(tf_embeddings):
            # v9.2: Enrich embedding with cross-TF context for DURATION prediction
            # Uses residual addition so cross-TF info flows into duration heads
            # v9.2.1: Apply LayerNorm after addition to fix scale mismatch
            enriched_embedding = self.enriched_norm(embedding + context_projected)  # [batch, hidden_dim]

            # Each timeframe makes independent prediction
            if self.shared_heads:
                # Single shared head for all TFs
                # v9.2: Duration uses ENRICHED embedding (with cross-TF context)
                dur_mean, dur_log_std, dur_hazard = self.per_tf_duration_head(enriched_embedding)
                # Other heads use raw embedding (unchanged)
                dir_logits = self.per_tf_direction_head(embedding)
                next_ch = self.per_tf_next_channel_head(embedding)
                conf = self.per_tf_confidence_head(embedding)
            else:
                # Separate head per TF (TF-specific learned parameters)
                # v9.2: Duration uses ENRICHED embedding (with cross-TF context)
                dur_mean, dur_log_std, dur_hazard = self.per_tf_duration_heads[tf_idx](enriched_embedding)
                # Other heads use raw embedding (unchanged)
                dir_logits = self.per_tf_direction_heads[tf_idx](embedding)
                next_ch = self.per_tf_next_channel_heads[tf_idx](embedding)
                conf = self.per_tf_confidence_heads[tf_idx](embedding)

            per_tf_durations_mean.append(dur_mean)
            per_tf_durations_log_std.append(dur_log_std)
            if dur_hazard is not None:
                per_tf_hazards.append(dur_hazard)
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
        # AGGREGATE PREDICTION (uses full attention context)
        # =====================================================================

        agg_dur_mean, agg_dur_log_std, agg_dur_hazard = self.agg_duration_head(context)
        agg_direction = self.agg_direction_head(context)
        agg_next_channel = self.agg_next_channel_head(context)
        agg_confidence = self.agg_confidence_head(context)
        agg_trigger_tf = self.agg_trigger_tf_head(context)  # v9.0.0: 21-class trigger TF

        # =====================================================================
        # MULTI-RESOLUTION PREDICTIONS (optional)
        # =====================================================================
        multi_res_dur = None
        multi_res_dir = None
        if self.use_multi_resolution:
            # Use aggregate context for multi-resolution
            multi_res_dur = self.multi_res_duration(context)
            multi_res_dir = self.multi_res_direction(context)

            # Optionally override main predictions with combined multi-res
            # outputs['agg_duration_mean'] = multi_res_dur['combined_mean']
            # outputs['agg_duration_log_std'] = multi_res_dur['combined_log_std']

        # =====================================================================
        # WINDOW SELECTION (per-TF)
        # =====================================================================
        # After TF attention, use embeddings + window_scores to select optimal window
        window_logits_list = []
        for tf_idx, embedding in enumerate(tf_embeddings):
            # Each TF uses shared window selector with its own embedding
            logits = self.window_selector(embedding, window_scores)
            window_logits_list.append(logits)

        # Stack: [batch, num_tfs, num_windows]
        window_logits = torch.stack(window_logits_list, dim=1)  # [batch, 11, 8]
        window_probs = F.softmax(window_logits, dim=-1)         # [batch, 11, 8]

        # Build output dictionary (keys match CombinedLoss expectations)
        output = {
            # Per-timeframe predictions (primary - used for training)
            'duration_mean': duration_mean,                    # [batch, 11]
            'duration_log_std': duration_log_std,              # [batch, 11]
            'duration_hazard': torch.stack(per_tf_hazards, dim=1) if per_tf_hazards else None,  # [batch, 11, num_hazard_bins] or None
            'direction_logits': direction_logits,              # [batch, 11]
            'next_channel_logits': next_channel_logits,        # [batch, 11, 3]
            'confidence': confidence,                          # [batch, 11]

            # Window selection (per-TF)
            'window_logits': window_logits,                    # [batch, 11, 8]
            'window_probs': window_probs,                      # [batch, 11, 8]

            # Aggregate predictions (optional - for dashboard summary)
            'aggregate': {
                'duration_mean': agg_dur_mean,                 # [batch, 1]
                'duration_log_std': agg_dur_log_std,           # [batch, 1]
                'duration_hazard': agg_dur_hazard,             # [batch, num_hazard_bins] or None
                'direction_logits': agg_direction,             # [batch, 1] - binary logit
                'next_channel_logits': agg_next_channel,       # [batch, 3]
                'confidence': agg_confidence,                  # [batch, 1]
                'trigger_tf_logits': agg_trigger_tf,           # [batch, 21] - v9.0.0
            }
        }

        # Multi-resolution predictions
        if self.use_multi_resolution:
            output['multi_res_duration'] = multi_res_dur
            output['multi_res_direction'] = multi_res_dir

        if return_attention:
            output['attention_weights'] = attn_weights

        return output

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make predictions in evaluation mode with per-timeframe breakdown.

        Args:
            x: [batch_size, 761] - input features (TIMEFRAME-GROUPED ordering!)

        Returns:
            Dictionary with:
                - Per-timeframe predictions (duration, direction, confidence for each of 11 TFs)
                - Window selection (per-TF window choices)
                - Aggregate predictions (weighted combination)
                - Attention weights
                - Recommended timeframe (highest confidence)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, return_attention=True)

            # =====================================================================
            # SURVIVAL LOSS: Convert hazard outputs to mean/std
            # =====================================================================
            if self.num_hazard_bins > 0:
                # Per-timeframe: convert hazard to mean/std
                if 'duration_hazard' in outputs and outputs['duration_hazard'] is not None:
                    duration_mean, duration_std = hazard_to_duration_stats(
                        outputs['duration_hazard'],
                        num_bins=self.num_hazard_bins,
                        max_duration=self.max_duration
                    )
                    outputs['duration_mean'] = duration_mean
                    outputs['duration_std'] = duration_std
                    # Keep hazard for potential loss computation

                # Aggregate: convert hazard to mean/std
                if 'aggregate' in outputs and outputs['aggregate']['duration_hazard'] is not None:
                    agg_duration_mean, agg_duration_std = hazard_to_duration_stats(
                        outputs['aggregate']['duration_hazard'],
                        num_bins=self.num_hazard_bins,
                        max_duration=self.max_duration
                    )
                    outputs['aggregate']['duration_mean'] = agg_duration_mean
                    outputs['aggregate']['duration_std'] = agg_duration_std
                    # Keep hazard for potential loss computation
            elif 'duration_log_std' in outputs:
                # Gaussian NLL: convert log_std to std
                outputs['duration_std'] = torch.exp(outputs['duration_log_std'])
                if 'aggregate' in outputs and 'duration_log_std' in outputs['aggregate']:
                    outputs['aggregate']['duration_std'] = torch.exp(outputs['aggregate']['duration_log_std'])
            else:
                # Huber/MSE: no uncertainty, set to zeros
                if 'duration_mean' in outputs:
                    outputs['duration_std'] = torch.zeros_like(outputs['duration_mean'])
                if 'aggregate' in outputs and 'duration_mean' in outputs['aggregate']:
                    outputs['aggregate']['duration_std'] = torch.zeros_like(outputs['aggregate']['duration_mean'])

            # Convert per-timeframe logits to probabilities
            direction_probs = torch.sigmoid(outputs['direction_logits'])  # [batch, 11]
            next_channel_probs = F.softmax(outputs['next_channel_logits'], dim=-1)  # [batch, 11, 3]

            # Convert aggregate logits to probabilities
            agg_direction_probs = torch.sigmoid(outputs['aggregate']['direction_logits'])  # [batch, 1]
            agg_next_channel_probs = F.softmax(outputs['aggregate']['next_channel_logits'], dim=-1)  # [batch, 3]
            agg_trigger_tf_probs = F.softmax(outputs['aggregate']['trigger_tf_logits'], dim=-1)  # [batch, 21] v9.0.0

            # Get class predictions for per-timeframe
            direction = (direction_probs > 0.5).long()                           # [batch, 11]
            next_channel = next_channel_probs.argmax(dim=-1)                     # [batch, 11]

            # Get aggregate class predictions
            agg_direction = (agg_direction_probs > 0.5).long()                   # [batch, 1]
            agg_next_channel = agg_next_channel_probs.argmax(dim=-1, keepdim=True)  # [batch, 1]
            agg_trigger_tf = agg_trigger_tf_probs.argmax(dim=-1, keepdim=True)   # [batch, 1] v9.0.0

            # Get duration_std (already computed above based on loss type)
            duration_std = outputs.get('duration_std', torch.exp(outputs['duration_log_std']))  # [batch, 11]
            agg_duration_std = outputs['aggregate'].get('duration_std', torch.exp(outputs['aggregate']['duration_log_std']))  # [batch, 1]

            # Window selection: use argmax during inference (hard selection)
            window_selection = outputs['window_logits'].argmax(dim=-1)           # [batch, 11]

            # Extract channel_valid flags from input features (first element of each TF block)
            # Each TF block is 56 features, channel_valid is at position 0
            channel_valid_indices = [tf_idx * 56 for tf_idx in range(11)]
            channel_valid = x[:, channel_valid_indices]  # [batch, 11]

            # Find recommended timeframe (highest confidence)
            best_tf_idx = outputs['confidence'].argmax(dim=1)  # [batch]

            # Validate critical outputs - use safe defaults instead of hard assert
            if not torch.isfinite(outputs['duration_mean']).all():
                import warnings
                warnings.warn("NaN/Inf detected in duration predictions, using safe defaults")
                outputs['duration_mean'] = torch.full_like(outputs['duration_mean'], 25.0)
                duration_std = torch.full_like(duration_std, 10.0)

            if not torch.isfinite(outputs['aggregate']['duration_mean']).all():
                import warnings
                warnings.warn("NaN/Inf detected in aggregate duration predictions, using safe defaults")
                outputs['aggregate']['duration_mean'] = torch.full_like(outputs['aggregate']['duration_mean'], 25.0)
                agg_duration_std = torch.full_like(agg_duration_std, 10.0)

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
                    'channel_valid': channel_valid[0].cpu().numpy(),    # [11]
                },

                # Window selection (per-TF)
                'window_selection': {
                    'selected_window': window_selection,                # [batch, 11] - argmax indices
                    'window_probs': outputs['window_probs'],            # [batch, 11, 8] - softmax probs
                    'window_logits': outputs['window_logits'],          # [batch, 11, 8] - raw logits
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
                    # v9.0.0: Trigger TF predictions (21-class)
                    'trigger_tf': agg_trigger_tf,                       # [batch, 1] - class 0-20
                    'trigger_tf_probs': agg_trigger_tf_probs,           # [batch, 21] - probabilities
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
                              count_params(self.agg_confidence_head) +
                              count_params(self.agg_trigger_tf_head)),  # v9.0.0
            'window_selector': count_params(self.window_selector),
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
    hidden_dim: int = 128,   # v9.2: Widened from 64 to preserve more feature info
    cfc_units: int = 192,    # v9.2: Increased from 96 (must be > hidden_dim + 2)
    num_attention_heads: int = 4,
    dropout: float = 0.1,
    shared_heads: bool = True,
    use_se_blocks: bool = False,
    se_reduction_ratio: int = 8,
    use_multi_resolution: bool = False,
    resolution_levels: int = 3,
    use_tcn: bool = False,
    tcn_channels: int = 64,
    tcn_kernel_size: int = 3,
    tcn_layers: int = 2,
    num_hazard_bins: int = 0,
    max_duration: float = 100.0,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> HierarchicalCfCModel:
    """
    Factory function to create and initialize model.

    Args:
        hidden_dim: Hidden dimension for branches (v9.2: default 128, was 64)
        cfc_units: CfC units per branch (must be > hidden_dim + 2)
        num_attention_heads: Attention heads
        dropout: Dropout probability
        shared_heads: If True, use single shared heads for all TFs (default).
                     If False, create separate prediction heads per timeframe.
        use_se_blocks: If True, add SE-blocks for feature reweighting (default: False)
        se_reduction_ratio: Reduction ratio for SE-block bottleneck (default: 8)
        use_multi_resolution: If True, add multi-resolution prediction heads (default: False)
        resolution_levels: Number of resolution levels for multi-resolution heads (default: 3)
        use_tcn: If True, add TCN block to each TF branch for additional temporal modeling (default: False)
        tcn_channels: Number of channels in TCN hidden layers (default: 64)
        tcn_kernel_size: Kernel size for TCN convolutions (default: 3)
        tcn_layers: Number of temporal blocks in TCN (default: 2)
        num_hazard_bins: Number of bins for survival/hazard duration prediction (default: 0 = disabled)
        max_duration: Maximum duration value for survival loss conversion (default: 100.0)
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
        shared_heads=shared_heads,
        use_se_blocks=use_se_blocks,
        se_reduction_ratio=se_reduction_ratio,
        use_multi_resolution=use_multi_resolution,
        resolution_levels=resolution_levels,
        use_tcn=use_tcn,
        tcn_channels=tcn_channels,
        tcn_kernel_size=tcn_kernel_size,
        tcn_layers=tcn_layers,
        num_hazard_bins=num_hazard_bins,
        max_duration=max_duration,
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
    model = create_model(hidden_dim=128, cfc_units=192, num_attention_heads=4)

    # Create dummy input
    print(f"\n[2] Creating dummy input (batch_size=4, features={TOTAL_FEATURES})...")
    batch_size = 4
    x_dummy = torch.randn(batch_size, TOTAL_FEATURES)

    # Forward pass
    print("\n[3] Running forward pass...")
    outputs = model(x_dummy, return_attention=True)

    print("\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, dict):
            print(f"  {key}: (nested dict)")
            for k, v in value.items():
                print(f"    {k}: {v.shape}")
        elif isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    # Test prediction
    print("\n[4] Testing prediction mode...")
    predictions = model.predict(x_dummy)

    print("\nPrediction outputs:")
    for key, value in predictions.items():
        if isinstance(value, dict):
            print(f"  {key}: (nested dict)")
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: {v.shape}")
        elif isinstance(value, torch.Tensor) and 'attention' not in key:
            print(f"  {key}: {value.shape}")

    # Test window selection
    print("\n[5] Testing window selection...")
    print(f"  window_logits shape: {outputs['window_logits'].shape}")
    print(f"  window_probs shape: {outputs['window_probs'].shape}")
    print(f"  window_probs sum per TF: {outputs['window_probs'][0].sum(dim=-1)}")  # Should be 1.0 per TF

    # Test loss
    print("\n[6] Testing loss computation...")
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

    # ==========================================================================
    # Test DifferentiableWindowSelector (Phase 2b)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Testing DifferentiableWindowSelector (Phase 2b)")
    print("=" * 80)

    # Create selector and encoder
    print("\n[7] Creating SharedWindowEncoder and DifferentiableWindowSelector...")
    encoder = SharedWindowEncoder(input_dim=TOTAL_FEATURES, embed_dim=128)
    selector = DifferentiableWindowSelector(embed_dim=128, num_windows=8, temperature=1.0)

    print(f"  Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  Selector params: {sum(p.numel() for p in selector.parameters()):,}")

    # Create per-window features [batch, 8, 761]
    print(f"\n[8] Creating per-window features (batch=4, windows=8, features={TOTAL_FEATURES})...")
    per_window_features = torch.randn(batch_size, 8, TOTAL_FEATURES)

    # Encode each window
    print("\n[9] Encoding windows through SharedWindowEncoder...")
    window_embeddings = []
    for w in range(8):
        emb = encoder(per_window_features[:, w, :])
        window_embeddings.append(emb)
    window_embeddings = torch.stack(window_embeddings, dim=1)  # [batch, 8, embed_dim]
    print(f"  window_embeddings shape: {window_embeddings.shape}")

    # Test soft selection (training mode)
    print("\n[10] Testing SOFT selection (training)...")
    selector.train()
    selected_emb, probs = selector(window_embeddings, hard_select=False)
    print(f"  selected_embedding shape: {selected_emb.shape}")
    print(f"  selection_probs shape: {probs.shape}")
    print(f"  probs sum (should be 1.0): {probs.sum(dim=-1)}")
    print(f"  probs[0]: {probs[0].detach().numpy().round(3)}")

    # Test hard selection (inference mode)
    print("\n[11] Testing HARD selection (inference)...")
    selector.eval()
    selected_emb_hard, probs_hard = selector(window_embeddings, hard_select=True)
    print(f"  selected_embedding shape: {selected_emb_hard.shape}")
    print(f"  probs (one-hot): {probs_hard[0].detach().numpy()}")
    print(f"  selected window: {probs_hard[0].argmax().item()}")

    # Test entropy computation
    print("\n[12] Testing entropy computation...")
    entropy = selector.compute_entropy(probs)
    print(f"  entropy shape: {entropy.shape}")
    print(f"  entropy values: {entropy.detach().numpy().round(4)}")
    print(f"  max possible entropy: {torch.log(torch.tensor(8.0)).item():.4f}")

    # Test confidence computation
    print("\n[13] Testing selection confidence...")
    confidence = selector.get_selection_confidence(probs)
    print(f"  confidence shape: {confidence.shape}")
    print(f"  confidence values: {confidence.detach().numpy().round(4)}")

    # Test temperature annealing
    print("\n[14] Testing temperature annealing...")
    selector.set_temperature(5.0)
    _, probs_hot = selector(window_embeddings, hard_select=False)
    print(f"  temp=5.0: probs[0] = {probs_hot[0].detach().numpy().round(3)}")

    selector.set_temperature(0.1)
    _, probs_cold = selector(window_embeddings, hard_select=False)
    print(f"  temp=0.1: probs[0] = {probs_cold[0].detach().numpy().round(3)}")

    # Test gradient flow
    print("\n[15] Testing gradient flow (key for end-to-end training)...")
    selector.set_temperature(1.0)
    selector.train()

    # Create a simple prediction head
    prediction_head = nn.Linear(128, 1)

    # Forward pass
    selected_emb, probs = selector(window_embeddings, hard_select=False)
    prediction = prediction_head(selected_emb)
    target = torch.randn(batch_size, 1)
    loss = F.mse_loss(prediction, target)

    # Backward pass
    loss.backward()

    # Check gradients exist in selector
    has_grads = all(p.grad is not None for p in selector.parameters())
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradients exist in selector: {has_grads}")

    # Check gradient magnitudes
    total_grad_norm = 0
    for p in selector.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    print(f"  Total gradient norm in selector: {total_grad_norm:.6f}")

    # Test Gumbel-softmax mode
    print("\n[16] Testing Gumbel-softmax mode...")
    gumbel_selector = DifferentiableWindowSelector(
        embed_dim=128, num_windows=8, temperature=1.0, use_gumbel=True
    )
    gumbel_selector.train()
    _, gumbel_probs = gumbel_selector(window_embeddings, hard_select=False)
    print(f"  Gumbel probs[0]: {gumbel_probs[0].detach().numpy().round(3)}")
    print(f"  (Note: Gumbel adds stochasticity for exploration)")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
