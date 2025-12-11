"""
Hierarchical Liquid Neural Network for Multi-Scale Stock Prediction (v5.2)

Architecture:
- 11 CfC layers (one per timeframe: 5min, 15min, 30min, 1h, 2h, 3h, 4h, daily, weekly, monthly, 3month)
- VIX CfC layer: 90-day daily VIX sequence processing (v5.2)
- Event embedding: FOMC, earnings, deliveries (v5.2)
- Each layer receives NATIVE OHLC data at its timeframe (not downsampled 1-min)
- Bottom-up hidden state passing (fast → slow)
- Fusion Head: 33 layer predictions + 12 market_state = 45 dims

v5.2 Key Features:
- VIX CfC layer for regime-aware predictions
- Event embedding for catalyst-aware duration predictions
- Probabilistic duration (mean + std)
- Validity heads (forward-looking channel assessment)
- Multi-Phase Compositor (transition type + direction prediction)
- Dual output (raw geometric + adjusted)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from pathlib import Path
import json

# Import Liquid Neural Network components
from ncps.torch import CfC
from ncps.wirings import AutoNCP

from .base import ModelBase
from .physics_attention import (
    CoulombTimeframeAttention,
    MarketPhaseClassifier,
    TimeframeInteractionHierarchy,
    EnergyBasedConfidence
)

# v5.2: Import event system
try:
    from .live_events import EventEmbedding, VIXSequenceLoader
    HAS_EVENT_SYSTEM = True
except ImportError:
    HAS_EVENT_SYSTEM = False

# v5.3: Import hierarchical containment and RSI validation
try:
    from .hierarchical_containment import HierarchicalContainmentChecker, get_rsi_from_features
    from .rsi_validator import RSIDirectionValidator, RSIFeatureExtractor
    HAS_HIERARCHICAL_FEATURES = True
except ImportError:
    HAS_HIERARCHICAL_FEATURES = False

# Import config for timeframe settings
import sys
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))
import config as project_config


# =============================================================================
# v5.2 MULTI-PHASE COMPOSITOR
# =============================================================================

class MultiPhaseCompositor(nn.Module):
    """
    v5.2: Predict channel transitions and Phase 2 projections.

    Predicts what happens when current channel ends:
    - Transition type: CONTINUE, SWITCH_TF, REVERSE, SIDEWAYS
    - Direction: BULL, BEAR, SIDEWAYS
    - Phase 2 slope magnitude

    Analogy: Relay race handoff detector - who takes the baton next?
    """

    # Transition type constants
    TRANSITION_CONTINUE = 0   # Same channel extends
    TRANSITION_SWITCH_TF = 1  # Different TF's channel takes over
    TRANSITION_REVERSE = 2    # Same TF, opposite direction
    TRANSITION_SIDEWAYS = 3   # Same TF, enters consolidation

    # Direction constants
    DIRECTION_BULL = 0
    DIRECTION_BEAR = 1
    DIRECTION_SIDEWAYS = 2

    def __init__(
        self,
        hidden_size: int,
        n_timeframes: int = 11,
        vix_size: int = 128,
        event_size: int = 32
    ):
        """
        Initialize Multi-Phase Compositor.

        Args:
            hidden_size: Size of each TF hidden state
            n_timeframes: Number of timeframes (11)
            vix_size: Size of VIX hidden state
            event_size: Size of event embedding
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.n_timeframes = n_timeframes

        # Input: all TF hiddens + VIX + events
        input_dim = hidden_size * n_timeframes + vix_size + event_size

        # Transition type predictor (4 classes)
        self.transition_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # [continue, switch_tf, reverse, sideways]
        )

        # TF switch predictor (which TF to switch to, if switching)
        self.tf_switch_head = nn.Sequential(
            nn.Linear(hidden_size * n_timeframes, 128),
            nn.ReLU(),
            nn.Linear(128, n_timeframes),  # Probability per TF
        )

        # Phase 2 direction predictor (3 classes: bull, bear, sideways)
        # Uses selected TF hidden + VIX
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size + vix_size, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # [bull, bear, sideways]
        )

        # Phase 2 slope magnitude predictor
        self.phase2_slope_head = nn.Sequential(
            nn.Linear(hidden_size + vix_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        all_hidden: Dict[str, torch.Tensor],
        hidden_vix: torch.Tensor,
        event_embed: torch.Tensor,
        current_tf: str,
        timeframes: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Predict transition type and Phase 2 parameters.

        Args:
            all_hidden: Dict mapping timeframe -> hidden state [batch, hidden_size]
            hidden_vix: VIX hidden state [batch, vix_size]
            event_embed: Event embedding [batch, event_size]
            current_tf: Current selected timeframe name
            timeframes: List of timeframe names in order

        Returns:
            Dict with transition_probs, tf_switch_probs, direction_probs, phase2_slope
        """
        # Stack all hidden states
        h_all = torch.cat([all_hidden[tf] for tf in timeframes], dim=-1)

        # Full context including VIX and events
        context = torch.cat([h_all, hidden_vix, event_embed], dim=-1)

        # Transition type prediction
        transition_logits = self.transition_head(context)
        transition_probs = F.softmax(transition_logits, dim=-1)

        # TF switch probabilities
        tf_switch_logits = self.tf_switch_head(h_all)
        tf_switch_probs = F.softmax(tf_switch_logits, dim=-1)

        # Phase 2 direction (based on current TF hidden + VIX)
        current_hidden = all_hidden[current_tf]
        dir_context = torch.cat([current_hidden, hidden_vix], dim=-1)
        direction_logits = self.direction_head(dir_context)
        direction_probs = F.softmax(direction_logits, dim=-1)

        # Phase 2 slope magnitude
        phase2_slope = self.phase2_slope_head(dir_context)

        return {
            'transition_logits': transition_logits,      # [batch, 4] - for loss
            'transition_probs': transition_probs,        # [batch, 4] - softmax
            'tf_switch_logits': tf_switch_logits,        # [batch, 11] - for loss
            'tf_switch_probs': tf_switch_probs,          # [batch, 11] - softmax
            'direction_logits': direction_logits,        # [batch, 3] - for loss
            'direction_probs': direction_probs,          # [batch, 3] - softmax
            'phase2_slope': phase2_slope,                # [batch, 1]
        }


class ChannelProjectionExtractor(nn.Module):
    """
    v5.1: Selects best window projection by quality (no learned blending).

    For each timeframe, picks the single best window from 21 candidates
    based on quality score (composite of r², complete_cycles, etc.).
    Pure geometric selection - no neural network, fully interpretable.
    """

    def __init__(self, timeframe: str, hidden_size: int, num_windows: int = 21):
        """
        Args:
            timeframe: Timeframe name (e.g., '5min', '1h', 'daily')
            hidden_size: Size of CfC hidden state (unused in v5.1, kept for compatibility)
            num_windows: Number of window sizes (default: 21)
        """
        super().__init__()
        self.timeframe = timeframe
        self.hidden_size = hidden_size
        self.num_windows = num_windows

        # No learned parameters in v5.1 - pure quality-based selection

    def forward(
        self,
        hidden_state: torch.Tensor,
        projections: torch.Tensor,
        quality_scores: torch.Tensor,
        r_squared: torch.Tensor,
        complete_cycles: torch.Tensor,
        position: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Select best window projection by quality score (no blending).

        Args:
            hidden_state: [batch, hidden_size] - CfC layer output (unused in v5.1)
            projections: [batch, num_windows, 2] - (high, low) projections for each window
            quality_scores: [batch, num_windows] - Quality score per window
            r_squared: [batch, num_windows] - R² per window
            complete_cycles: [batch, num_windows] - Complete cycles per window
            position: [batch, num_windows] - Channel position per window

        Returns:
            dict with selected_high, selected_low, best_window_idx
        """
        batch_size = projections.shape[0]

        # v5.1: Simple selection - pick window with highest quality
        # Quality score already incorporates r², cycles, and other metrics
        best_window_idx = torch.argmax(quality_scores, dim=-1)  # [batch]

        # Gather projections from best window for each sample in batch
        batch_indices = torch.arange(batch_size, device=projections.device)
        selected_high = projections[batch_indices, best_window_idx, 0].unsqueeze(-1)  # [batch, 1]
        selected_low = projections[batch_indices, best_window_idx, 1].unsqueeze(-1)   # [batch, 1]

        # Create one-hot weights for interpretability (which window was selected)
        validity_weights = torch.zeros(batch_size, self.num_windows, device=projections.device)
        validity_weights[batch_indices, best_window_idx] = 1.0

        return {
            'weighted_high': selected_high,  # Keep same key names for compatibility
            'weighted_low': selected_low,
            'validity_weights': validity_weights,  # One-hot (1.0 for selected, 0.0 for others)
            'best_window_idx': best_window_idx  # For interpretability
        }


class ProjectionFeatureExtractor:
    """
    v5.0: Helper to extract channel projection and quality features from input tensor.

    Finds projection features (projected_high, projected_low) and corresponding
    quality metrics for each window, organized by timeframe.
    """

    def __init__(self, feature_names: List[str], num_windows: int = 21):
        """
        Build index mappings for fast feature extraction.

        Args:
            feature_names: List of all feature names in order
            num_windows: Number of window sizes (default: 21)
        """
        self.feature_names = feature_names
        self.num_windows = num_windows
        self.window_sizes = [168, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10]

        # Build index maps for each timeframe and symbol
        self.indices = {}

        for symbol in ['tsla', 'spy']:
            for tf in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
                key = f"{symbol}_{tf}"
                self.indices[key] = self._build_indices(symbol, tf)

    def _build_indices(self, symbol: str, tf: str) -> Dict[str, List[int]]:
        """Build feature indices for one symbol-timeframe combination."""
        indices = {
            'projected_high': [],
            'projected_low': [],
            'quality_score': [],
            'r_squared_avg': [],
            'complete_cycles': [],
            'position': []
        }

        # Find indices for each window
        for w in self.window_sizes:
            prefix = f"{symbol}_channel_{tf}_w{w}"

            # Find each feature's index
            for feat_name, feat_list in indices.items():
                full_name = f"{prefix}_{feat_name}"
                try:
                    idx = self.feature_names.index(full_name)
                    feat_list.append(idx)
                except ValueError:
                    # Feature not found (might be missing in legacy data)
                    feat_list.append(-1)  # Sentinel value

        return indices

    def extract(self, x: torch.Tensor, symbol: str, tf: str) -> Dict[str, torch.Tensor]:
        """
        Extract projection and quality features for one symbol-timeframe.

        Args:
            x: Input features [batch, seq_len, total_features] or [batch, total_features]
            symbol: 'tsla' or 'spy'
            tf: Timeframe name

        Returns:
            dict with tensors shaped [batch, num_windows] or [batch, num_windows, 2]
        """
        key = f"{symbol}_{tf}"
        idx_map = self.indices.get(key, {})

        # Handle sequence input (take last timestep)
        if x.dim() == 3:
            x = x[:, -1, :]  # [batch, features]

        batch_size = x.shape[0]
        device = x.device

        # Extract projections [batch, num_windows, 2]
        proj_high_list = []
        proj_low_list = []

        for idx_high, idx_low in zip(idx_map['projected_high'], idx_map['projected_low']):
            if idx_high >= 0:
                proj_high_list.append(x[:, idx_high:idx_high+1])
            else:
                proj_high_list.append(torch.zeros(batch_size, 1, device=device))

            if idx_low >= 0:
                proj_low_list.append(x[:, idx_low:idx_low+1])
            else:
                proj_low_list.append(torch.zeros(batch_size, 1, device=device))

        projections_high = torch.cat(proj_high_list, dim=-1)  # [batch, num_windows]
        projections_low = torch.cat(proj_low_list, dim=-1)    # [batch, num_windows]
        projections = torch.stack([projections_high, projections_low], dim=-1)  # [batch, num_windows, 2]

        # Extract quality metrics [batch, num_windows]
        quality_list = []
        r_squared_list = []
        cycles_list = []
        position_list = []

        for idx_q, idx_r, idx_c, idx_p in zip(
            idx_map['quality_score'],
            idx_map['r_squared_avg'],
            idx_map['complete_cycles'],
            idx_map['position']
        ):
            quality_list.append(x[:, idx_q:idx_q+1] if idx_q >= 0 else torch.zeros(batch_size, 1, device=device))
            r_squared_list.append(x[:, idx_r:idx_r+1] if idx_r >= 0 else torch.zeros(batch_size, 1, device=device))
            cycles_list.append(x[:, idx_c:idx_c+1] if idx_c >= 0 else torch.zeros(batch_size, 1, device=device))
            position_list.append(x[:, idx_p:idx_p+1] if idx_p >= 0 else torch.zeros(batch_size, 1, device=device))

        return {
            'projections': projections,  # [batch, num_windows, 2]
            'quality_scores': torch.cat(quality_list, dim=-1),  # [batch, num_windows]
            'r_squared': torch.cat(r_squared_list, dim=-1),
            'complete_cycles': torch.cat(cycles_list, dim=-1),
            'position': torch.cat(position_list, dim=-1)
        }


class HierarchicalLNN(nn.Module, ModelBase):
    """
    Hierarchical Liquid Neural Network with 11 temporal scales (v4.0).

    Each timeframe gets its own CfC layer with native OHLC data at that resolution.
    Hidden states flow from fast (5min) to slow (3month) layers.
    """

    # Class-level constants for architecture
    TIMEFRAMES = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']
    NUM_LAYERS = 11
    PREDICTIONS_PER_LAYER = 3  # high, low, confidence
    MARKET_STATE_DIM = 12
    FUSION_INPUT_DIM = NUM_LAYERS * PREDICTIONS_PER_LAYER + MARKET_STATE_DIM  # 33 + 12 = 45

    def __init__(
        self,
        input_sizes: Dict[str, int] = None,  # v4.0: Feature size per timeframe
        hidden_size: int = 128,
        internal_neurons_ratio: float = 2.0,  # Total neurons = hidden_size × ratio
        device: str = 'cpu',
        multi_task: bool = True,  # Enable multi-task heads
        use_fusion_head: bool = True,  # v4.1: Can disable for physics-only mode
        use_geometric_base: bool = True,  # v5.0: Use geometric projections or learned approximation
        information_flow: str = 'bottom_up',  # v5.3.2: independent, bottom_up, top_down, bidirectional_bottom, bidirectional_top
        # Backward compatibility
        input_size: int = None,  # Deprecated: use input_sizes dict
    ):
        """
        Initialize 11-layer hierarchical model.

        Args:
            input_sizes: Dict mapping timeframe name to feature count, e.g.:
                         {'5min': 900, '15min': 900, ..., '3month': 900}
                         If None, will be set on first forward pass
            hidden_size: Hidden state size / output neurons (default: 128)
            internal_neurons_ratio: Total neurons = hidden_size × ratio (default: 2.0 → 256 total)
            device: 'cuda', 'mps', or 'cpu'
            multi_task: Enable multi-task prediction heads
            use_fusion_head: If True, use fusion head for final predictions.
                           If False, use physics-based aggregation from Coulomb attention.
            use_geometric_base: If True, extract geometric projections from features as base.
                              If False, learn base approximations with neural nets.
            input_size: [DEPRECATED] Single input size (for backward compatibility only)
        """
        super().__init__()

        self.input_sizes = input_sizes or {}
        self.hidden_size = hidden_size
        self.internal_neurons_ratio = internal_neurons_ratio
        self.total_neurons = int(hidden_size * internal_neurons_ratio)
        self.device_type = device
        self.multi_task = multi_task
        self.use_fusion_head = use_fusion_head
        self.use_geometric_base = use_geometric_base  # v5.0: Geometric vs learned base
        self.use_channel_projections = use_geometric_base  # v5.0: Enable projection extractors if geometric
        self.information_flow = information_flow  # v5.3.1: Flow direction

        # Backward compatibility: if old-style single input_size provided
        if input_size is not None and not input_sizes:
            # Use same size for all timeframes (old behavior)
            self.input_sizes = {tf: input_size for tf in self.TIMEFRAMES}

        # Fusion output dimension scales with hidden_size
        self.fusion_output_dim = self.hidden_size // 2  # e.g., 128→64, 256→128

        # v5.0: Window sizes for projection extraction (21 windows)
        self.window_sizes = [168, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10]

        # =========================================================================
        # 11 CfC LAYERS (one per timeframe)
        # =========================================================================
        # Each layer receives: native OHLC features for its timeframe
        # First layer (5min): features + VIX + events
        # Subsequent layers: features + previous hidden + VIX + events
        #
        # v5.2: VIX hidden (128) + event embed (32) = 160 additional dims per layer

        self.timeframe_layers = nn.ModuleDict()
        self.timeframe_heads = nn.ModuleDict()

        # v5.2: Pre-declare dimensions for VIX/events (will be set properly in __init__)
        # These are used during layer creation before VIX layer is created
        _vix_hidden_size = 128
        _event_embed_dim = 32

        for i, tf in enumerate(self.TIMEFRAMES):
            tf_input_size = self.input_sizes.get(tf, 900)  # Default to 900 if not specified

            # v5.3.1: All layers same size (flow-independent)
            # Always allocate space for neighbor hidden (zero-padded if unused in flow)
            # This allows top_down/bidirectional without size mismatches
            layer_input_size = tf_input_size + hidden_size + _vix_hidden_size + _event_embed_dim
            # = 1104 + 128 + 128 + 32 = 1392 for all layers

            # v5.2: Increase total neurons to handle larger input (320 instead of 256)
            layer_total_neurons = int(hidden_size * 2.5)  # 128 * 2.5 = 320

            # Create CfC layer with AutoNCP wiring
            wiring = AutoNCP(layer_total_neurons, hidden_size)
            self.timeframe_layers[tf] = CfC(layer_input_size, wiring, batch_first=True)

            # Create output heads for this layer
            self.timeframe_heads[f'{tf}_high'] = nn.Linear(hidden_size, 1)
            self.timeframe_heads[f'{tf}_low'] = nn.Linear(hidden_size, 1)
            self.timeframe_heads[f'{tf}_conf'] = nn.Linear(hidden_size, 1)

        # =========================================================================
        # FUSION HEAD (v4.0: 45 dims instead of 790)
        # =========================================================================
        # Input: 33 layer predictions (11 TF × 3 outputs) + 12 market_state = 45 dims
        # No more news embeddings (removed)

        # v5.3: Keep fc1/fc2 for multi-task heads, remove prediction heads
        self.fusion_fc1 = nn.Linear(self.FUSION_INPUT_DIM, 128)
        self.fusion_fc2 = nn.Linear(128, self.fusion_output_dim)
        # fusion_fc_high, fusion_fc_low, fusion_fc_conf REMOVED (locked to Physics-Only)

        # NOTE: fusion_hidden is still created in forward() for multi-task heads

        # =========================================================================
        # MULTI-TASK HEADS
        # =========================================================================
        if multi_task:
            # Hit Band: Will price enter predicted band? (Binary classification)
            self.hit_band_head = nn.Sequential(
                nn.Linear(self.fusion_output_dim, self.fusion_output_dim // 2),
                nn.ReLU(),
                nn.Linear(self.fusion_output_dim // 2, 1),
                nn.Sigmoid()
            )

            # Hit Target: Will trade work (target before stop)? (Binary classification)
            self.hit_target_head = nn.Sequential(
                nn.Linear(self.fusion_output_dim, self.fusion_output_dim // 2),
                nn.ReLU(),
                nn.Linear(self.fusion_output_dim // 2, 1),
                nn.Sigmoid()
            )

            # Expected Return: Direct return prediction (Regression)
            self.expected_return_head = nn.Linear(self.fusion_output_dim, 1)

            # Overshoot: How far price overshoots band (Regression)
            self.overshoot_head = nn.Linear(self.fusion_output_dim, 1)

            # Continuation: Channel continuation duration and gain (Regression)
            # Legacy single-TF heads (for backward compatibility)
            self.continuation_duration_head = nn.Linear(self.fusion_output_dim, 1)
            self.continuation_gain_head = nn.Linear(self.fusion_output_dim, 1)
            self.continuation_confidence_head = nn.Sequential(
                nn.Linear(self.fusion_output_dim, self.fusion_output_dim // 2),
                nn.ReLU(),
                nn.Linear(self.fusion_output_dim // 2, 1),
                nn.Sigmoid()
            )

            # v4.3: Per-timeframe continuation prediction heads
            # Each TF has its own duration/gain/confidence predictions
            self.per_tf_cont_heads = nn.ModuleDict()
            for tf in self.TIMEFRAMES:
                # Each TF continuation head takes that TF's hidden state
                self.per_tf_cont_heads[f'{tf}_duration'] = nn.Linear(hidden_size, 1)
                self.per_tf_cont_heads[f'{tf}_gain'] = nn.Linear(hidden_size, 1)
                self.per_tf_cont_heads[f'{tf}_confidence'] = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, 1),
                    nn.Sigmoid()
                )

            # Adaptive horizon predictor
            self.adaptive_horizon_head = nn.Sequential(
                nn.Linear(self.fusion_output_dim, self.fusion_output_dim // 2),
                nn.ReLU(),
                nn.Linear(self.fusion_output_dim // 2, 1),
                nn.Sigmoid()
            )
            self.adaptive_conf_score_head = nn.Sequential(
                nn.Linear(self.fusion_output_dim, self.fusion_output_dim // 2),
                nn.ReLU(),
                nn.Linear(self.fusion_output_dim // 2, 1),
                nn.Sigmoid()
            )

            # Adaptive Projection: uses all 11 hidden states + 11 confidences
            # 11 layers × 128 hidden + 11 confs = 1408 + 11 = 1419
            self.adaptive_projection = nn.Sequential(
                nn.Linear(self.hidden_size * self.NUM_LAYERS + self.NUM_LAYERS, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 3)  # [price_change_pct, horizon_bars_log, confidence]
            )

            # Breakout Prediction Heads
            self.breakout_prob_head = nn.Sequential(
                nn.Linear(self.fusion_output_dim, self.fusion_output_dim // 2),
                nn.ReLU(),
                nn.Linear(self.fusion_output_dim // 2, 1),
                nn.Sigmoid()
            )

            self.breakout_direction_head = nn.Sequential(
                nn.Linear(self.fusion_output_dim, self.fusion_output_dim // 2),
                nn.ReLU(),
                nn.Linear(self.fusion_output_dim // 2, 1),
                nn.Sigmoid()
            )

            self.breakout_bars_head = nn.Sequential(
                nn.Linear(self.fusion_output_dim, self.fusion_output_dim // 2),
                nn.ReLU(),
                nn.Linear(self.fusion_output_dim // 2, 1)
            )

            self.breakout_confidence_head = nn.Sequential(
                nn.Linear(self.fusion_output_dim, self.fusion_output_dim // 2),
                nn.ReLU(),
                nn.Linear(self.fusion_output_dim // 2, 1),
                nn.Sigmoid()
            )

        # =========================================================================
        # PHYSICS-INSPIRED MODULES (GWC-based)
        # =========================================================================
        # Dynamic timeframe attention based on screened Coulomb potential
        self.coulomb_attention = CoulombTimeframeAttention(
            hidden_size=hidden_size,
            n_timeframes=len(self.TIMEFRAMES)
        )

        # Explicit V₁, V₂, V₃ interaction hierarchy
        self.interaction_hierarchy = TimeframeInteractionHierarchy(
            hidden_size=hidden_size,
            n_timeframes=len(self.TIMEFRAMES)
        )

        # Market phase classifier
        self.phase_classifier = MarketPhaseClassifier(
            hidden_size=hidden_size,
            n_timeframes=len(self.TIMEFRAMES)
        )

        # Energy-based confidence scorer
        self.energy_scorer = EnergyBasedConfidence(
            hidden_size=hidden_size,
            n_timeframes=len(self.TIMEFRAMES)
        )

        # =========================================================================
        # v5.0: CHANNEL PROJECTION EXTRACTORS (one per timeframe)
        # =========================================================================
        # Each extractor learns which of 21 window projections to trust
        self.projection_extractors = nn.ModuleDict({
            tf: ChannelProjectionExtractor(
                timeframe=tf,
                hidden_size=hidden_size,
                num_windows=21  # 21 window sizes per timeframe
            ) for tf in self.TIMEFRAMES
        })

        # v5.0: Adjustment networks (Option B: Channel base + learned corrections)
        # Small networks that learn when to adjust channel projections
        self.projection_adjusters = nn.ModuleDict({
            tf: nn.Sequential(
                nn.Linear(hidden_size + 2, 32),  # hidden + [base_high, base_low]
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 2)  # [adjustment_high, adjustment_low]
            ) for tf in self.TIMEFRAMES
        })

        # v5.0: Projection feature extractor (will be initialized on first forward)
        self.projection_feature_extractor = None

        # =========================================================================
        # v5.2: VIX CfC LAYER
        # =========================================================================
        # Processes 90 days of daily VIX data to capture regime information
        # Output is broadcast to all TF layers
        self.vix_input_size = 11  # OHLC (4) + derived (7)
        self.vix_hidden_size = 128
        self.vix_sequence_length = 90  # Days

        vix_wiring = AutoNCP(256, self.vix_hidden_size)
        self.vix_layer = CfC(self.vix_input_size, vix_wiring, batch_first=True)

        # =========================================================================
        # v5.2: EVENT EMBEDDING
        # =========================================================================
        self.event_embed_dim = 32
        if HAS_EVENT_SYSTEM:
            self.event_embedding = EventEmbedding(
                event_types=6,  # fomc, earnings, delivery, cpi, nfp, other
                embed_dim=self.event_embed_dim
            )
        else:
            # Fallback: simple learnable embedding
            self.event_embedding = nn.Sequential(
                nn.Linear(6, 16),  # 6 event type indicators
                nn.ReLU(),
                nn.Linear(16, self.event_embed_dim),
            )

        # =========================================================================
        # v5.2/v5.3: PROBABILISTIC DURATION HEADS (mean + log_std per TF)
        # =========================================================================
        # Each TF predicts duration with uncertainty
        # v5.3: Larger input to accommodate parent TF hiddens (up to 2 parents × 128)
        self.duration_heads = nn.ModuleDict()
        for tf in self.TIMEFRAMES:
            # v5.3 Context: hidden + parent_hiddens (up to 2×128) + VIX + events
            # Max: 128 + 256 + 128 + 32 = 544
            # Use max size, zero-pad when fewer parents available
            context_dim = hidden_size + (hidden_size * 2) + self.vix_hidden_size + self.event_embed_dim

            self.duration_heads[f'{tf}_mean'] = nn.Sequential(
                nn.Linear(context_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
                nn.Softplus(),  # Positive duration
            )
            self.duration_heads[f'{tf}_log_std'] = nn.Sequential(
                nn.Linear(context_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        # =========================================================================
        # v5.2: VALIDITY HEADS (forward-looking channel assessment)
        # =========================================================================
        # Predicts: Will this channel hold going forward?
        # Uses quality_score as ONE input (not the answer) + VIX + events + position
        self.validity_heads = nn.ModuleDict()
        for tf in self.TIMEFRAMES:
            # Input: hidden + VIX + events + [quality_score, position_in_channel]
            validity_input_dim = hidden_size + self.vix_hidden_size + self.event_embed_dim + 2

            self.validity_heads[tf] = nn.Sequential(
                nn.Linear(validity_input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()  # 0-1: probability channel holds
            )

        # =========================================================================
        # v5.2: MULTI-PHASE COMPOSITOR
        # =========================================================================
        self.compositor = MultiPhaseCompositor(
            hidden_size=hidden_size,
            n_timeframes=len(self.TIMEFRAMES),
            vix_size=self.vix_hidden_size,
            event_size=self.event_embed_dim
        )

        # v5.3: Hierarchical Containment & RSI (static methods, no parameters)
        # =========================================================================
        if HAS_HIERARCHICAL_FEATURES:
            self.containment_checker = HierarchicalContainmentChecker
            self.rsi_extractor = RSIFeatureExtractor
            self.rsi_validator = RSIDirectionValidator
        else:
            self.containment_checker = None
            self.rsi_extractor = None
            self.rsi_validator = None

        # v5.3.1: Refinement Networks (for bidirectional modes)
        # =========================================================================
        # Small networks that combine current hidden + neighbor hidden in Pass 2
        if 'bidirectional' in information_flow:
            self.refinement_nets = nn.ModuleDict()
            for tf in self.TIMEFRAMES:
                self.refinement_nets[tf] = nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size),  # Current + neighbor
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size, hidden_size)
                )
        else:
            self.refinement_nets = None

        # v5.2: Track VIX/event state for live inference
        self.cached_vix_hidden = None
        self.cached_event_embed = None

        # Move to device
        self.to(device)

        # Track last inputs for online learning
        self.last_inputs = {}
        self.last_market_state = None

    def _extract_projection_features_from_tensor(
        self,
        x_tf: torch.Tensor,
        tf: str,
        symbol: str = 'tsla'
    ) -> Dict[str, torch.Tensor]:
        """
        Extract channel projection and quality features from input tensor (v5.0).

        This is a simplified extractor that assumes feature order follows the standard pattern.
        For native TF mode where we don't have explicit feature names.

        Args:
            x_tf: Input features [batch, seq_len, features] or [batch, features]
            tf: Timeframe name
            symbol: 'tsla' or 'spy'

        Returns:
            dict with extracted projection and quality tensors
        """
        # Handle sequence input (take last timestep)
        if x_tf.dim() == 3:
            x = x_tf[:, -1, :]  # [batch, features]
        else:
            x = x_tf

        batch_size = x.shape[0]
        device = x.device

        # For now, we'll use the standard prediction heads as base
        # This will be enhanced when we have explicit feature name mapping
        # The base heads will learn to approximate geometric projections through training

        # Return empty dict - signals to use learned heads as base
        return None

    def forward(
        self,
        x: torch.Tensor,
        market_state: Optional[torch.Tensor] = None,
        hidden_states: Optional[Dict[str, torch.Tensor]] = None,
        vix_sequence: Optional[torch.Tensor] = None,  # v5.2: [batch, 90, 11] VIX data
        events: Optional[List[Dict]] = None,  # v5.2: Event list from LiveEventFetcher
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through 11 hierarchical layers.

        Args:
            x: Either a Dict[str, Tensor] mapping timeframe to features,
               OR a single Tensor [batch, seq_len, features] for backward compatibility
            market_state: Market regime features [batch, 12]
            hidden_states: Optional dict of initial hidden states per timeframe
            vix_sequence: v5.2 - VIX sequence [batch, 90, 11] for VIX CfC
            events: v5.2 - Event list for event embedding

        Returns:
            predictions: [batch, 3] - [predicted_high, predicted_low, confidence]
            output_dict: Dict with hidden states, layer predictions, multi-task outputs, v5.2 outputs
        """
        # Handle backward compatibility: single tensor input
        if isinstance(x, torch.Tensor):
            return self.forward_single_input(x, market_state, vix_sequence, events)

        # x is now timeframe_data dict
        timeframe_data = x

        if hidden_states is None:
            hidden_states = {}

        # Get batch size from first available tensor
        first_tf = next(iter(timeframe_data.keys()))
        batch_size = timeframe_data[first_tf].shape[0]
        device = timeframe_data[first_tf].device

        # Store inputs for online learning
        self.last_inputs = {tf: data.detach() for tf, data in timeframe_data.items()}

        # =========================================================================
        # v5.2: VIX CfC PROCESSING
        # =========================================================================
        if vix_sequence is not None:
            # Process VIX sequence through VIX CfC layer
            vix_out, _ = self.vix_layer(vix_sequence, None)
            hidden_vix = vix_out[:, -1, :]  # [batch, vix_hidden_size]
            self.cached_vix_hidden = hidden_vix.detach()
        elif self.cached_vix_hidden is not None:
            # Use cached VIX hidden from previous pass (live inference)
            hidden_vix = self.cached_vix_hidden.to(device)
            if hidden_vix.shape[0] != batch_size:
                hidden_vix = hidden_vix[:1].expand(batch_size, -1)
        else:
            # No VIX available - use zeros
            hidden_vix = torch.zeros(batch_size, self.vix_hidden_size, device=device)

        # =========================================================================
        # v5.2: EVENT EMBEDDING
        # =========================================================================
        if events is not None and HAS_EVENT_SYSTEM and isinstance(self.event_embedding, EventEmbedding):
            # v5.2: Events is a batch (list of lists), use forward_batch
            event_embed = self.event_embedding.forward_batch(events, device)
            self.cached_event_embed = event_embed.detach()
        elif self.cached_event_embed is not None:
            # Use cached event embedding (live inference)
            event_embed = self.cached_event_embed.to(device)
            if event_embed.shape[0] != batch_size:
                event_embed = event_embed[:1].expand(batch_size, -1)
        else:
            # No events - use zeros or no-event embedding
            if HAS_EVENT_SYSTEM and isinstance(self.event_embedding, EventEmbedding):
                event_embed = self.event_embedding([], batch_size, device)
            else:
                event_embed = torch.zeros(batch_size, self.event_embed_dim, device=device)

        # v5.0: Initialize projection feature extractor on first forward (need feature names from data)
        # This will be used to extract projection/quality features from input tensors
        if self.projection_feature_extractor is None and self.use_channel_projections:
            # NOTE: For native TF mode, we can't easily get feature names here
            # Will initialize lazily when needed, or require explicit initialization
            # For now, mark as "will initialize when feature names available"
            pass

        # =========================================================================
        # v5.3: TWO-PASS ARCHITECTURE
        # Pass 1: Build all hidden states (CfC processing)
        # Pass 2: Predict durations/validity with full hierarchical context
        # =========================================================================
        layer_predictions = []
        layer_hidden_states = {}
        layer_confidences = []
        all_hidden = []
        projection_metadata = {}  # v5.0: Store projection details for interpretability
        duration_outputs = {}  # v5.2: Probabilistic duration
        validity_outputs = {}  # v5.2: Forward-looking validity
        tf_hidden_dict = {}  # v5.3: Dict for easy parent access

        # =========================================================================
        # PASS 1: PROCESS ALL CFC LAYERS (build hidden states)
        # v5.3.1: Support 4 information flow modes
        # =========================================================================

        # Determine processing order based on information_flow
        if self.information_flow == 'top_down':
            tf_indices = list(reversed(range(len(self.TIMEFRAMES))))  # 10→0 (3month→5min)
        else:
            tf_indices = list(range(len(self.TIMEFRAMES)))  # 0→10 (5min→3month)

        # Process in chosen order
        for idx in tf_indices:
            i = idx  # Keep i for compatibility
            tf = self.TIMEFRAMES[i]

            # Get input for this timeframe
            if tf not in timeframe_data:
                # Skip if timeframe data not provided
                # Use zeros for predictions
                all_hidden.append(torch.zeros(batch_size, self.hidden_size, device=device))
                tf_hidden_dict[tf] = all_hidden[-1]
                continue

            x_tf = timeframe_data[tf]  # [batch, seq_len, features]
            seq_len = x_tf.shape[1]

            # v5.3.2: Get neighbor hidden (or zeros) based on flow direction
            neighbor_hidden = None

            if self.information_flow == 'independent':
                # Independent: each TF processes alone, no cross-TF hidden states
                neighbor_hidden = None  # Explicitly no neighbor

            elif self.information_flow == 'bottom_up':
                # Bottom-up: get previous (faster) TF
                if i > 0:
                    neighbor_hidden = tf_hidden_dict.get(self.TIMEFRAMES[i - 1])

            elif self.information_flow == 'top_down':
                # Top-down: get next (slower) TF
                if i < len(self.TIMEFRAMES) - 1:
                    neighbor_hidden = tf_hidden_dict.get(self.TIMEFRAMES[i + 1])

            else:  # Bidirectional modes - do bottom-up in Pass 1
                if i > 0:
                    neighbor_hidden = tf_hidden_dict.get(self.TIMEFRAMES[i - 1])

            # Always concat neighbor slot (zero-pad if no neighbor)
            if neighbor_hidden is not None:
                neighbor_expanded = neighbor_hidden.unsqueeze(1).expand(-1, seq_len, -1)
            else:
                # Zero-pad neighbor slot (flow-independent size)
                neighbor_expanded = torch.zeros(batch_size, seq_len, self.hidden_size, device=device)

            x_tf = torch.cat([x_tf, neighbor_expanded], dim=-1)

            # v5.2: Concatenate VIX hidden state to CfC input
            # This gives each TF layer awareness of volatility regime
            if hidden_vix is not None:
                vix_expanded = hidden_vix.unsqueeze(1).expand(-1, seq_len, -1)
                x_tf = torch.cat([x_tf, vix_expanded], dim=-1)

            # v5.2: Concatenate event embedding to CfC input
            # This gives each TF layer awareness of upcoming catalysts
            if event_embed is not None:
                event_expanded = event_embed.unsqueeze(1).expand(-1, seq_len, -1)
                x_tf = torch.cat([x_tf, event_expanded], dim=-1)

            # Get initial hidden state if provided
            h_init = hidden_states.get(tf, None)

            # Forward through CfC layer
            layer_out, h_new = self.timeframe_layers[tf](x_tf, h_init)

            # Take final hidden state
            hidden = layer_out[:, -1, :]  # [batch, hidden_size]
            layer_hidden_states[tf] = h_new
            all_hidden.append(hidden)
            tf_hidden_dict[tf] = hidden  # v5.3: Store in dict for parent access

        # =========================================================================
        # v5.3.1: BIDIRECTIONAL PASS 2 (if enabled)
        # Refine hidden states with opposite-direction context
        # =========================================================================
        if 'bidirectional' in self.information_flow and self.refinement_nets:
            # Determine Pass 2 direction (opposite of Pass 1)
            if self.information_flow == 'bidirectional_bottom':
                # Pass 1 was bottom-up, Pass 2 is top-down
                pass2_indices = list(reversed(range(len(self.TIMEFRAMES))))
            else:  # bidirectional_top
                # Pass 1 was top-down, Pass 2 is bottom-up
                pass2_indices = list(range(len(self.TIMEFRAMES)))

            # Refine each hidden with neighbor from opposite direction
            for idx in pass2_indices:
                i = idx
                tf = self.TIMEFRAMES[i]

                if tf not in tf_hidden_dict:
                    continue

                current_hidden = tf_hidden_dict[tf]

                # Get neighbor from opposite direction
                if self.information_flow == 'bidirectional_bottom':
                    # Top-down in Pass 2
                    if i < len(self.TIMEFRAMES) - 1:
                        neighbor_hidden = tf_hidden_dict.get(self.TIMEFRAMES[i + 1])
                    else:
                        neighbor_hidden = None
                else:  # bidirectional_top
                    # Bottom-up in Pass 2
                    if i > 0:
                        neighbor_hidden = tf_hidden_dict.get(self.TIMEFRAMES[i - 1])
                    else:
                        neighbor_hidden = None

                # Refine with neighbor context
                if neighbor_hidden is not None:
                    refinement_input = torch.cat([current_hidden, neighbor_hidden], dim=-1)
                    refined_hidden = self.refinement_nets[tf](refinement_input)
                    tf_hidden_dict[tf] = refined_hidden
                    # Update all_hidden as well (for downstream compatibility)
                    all_hidden[i] = refined_hidden

        # =========================================================================
        # PASS 2 (or PASS 3 for bidirectional): PREDICTIONS WITH HIERARCHICAL CONTEXT
        # Now all hidden states exist (and refined if bidirectional)
        # =========================================================================
        for i, tf in enumerate(self.TIMEFRAMES):
            # Get this TF's hidden state
            hidden = tf_hidden_dict.get(tf)
            if hidden is None:
                # TF was skipped
                continue

            # =====================================================================
            # v5.3: DURATION PREDICTION WITH HIERARCHICAL CONTEXT
            # =====================================================================
            duration_scale = 1.0  # Default scale
            if hasattr(self, 'duration_heads') and f'{tf}_mean' in self.duration_heads:
                # v5.3: Get parent TF hidden states (LARGER timeframes, flow-independent)
                # Use hierarchy position, not loop index (fixes top_down mode!)
                tf_hierarchy_idx = self.TIMEFRAMES.index(tf)  # Actual position (0-10)
                parent_hiddens = []

                # Parents are next 2 LARGER TFs in hierarchy (regardless of processing order)
                for offset in [1, 2]:
                    parent_hierarchy_idx = tf_hierarchy_idx + offset
                    if parent_hierarchy_idx < len(self.TIMEFRAMES):
                        parent_tf = self.TIMEFRAMES[parent_hierarchy_idx]
                        if parent_tf in tf_hidden_dict:
                            parent_hiddens.append(tf_hidden_dict[parent_tf])

                # v5.3: Build duration context with zero-padding for missing parents
                # Always use same input size (544) for all TFs
                if len(parent_hiddens) == 2:
                    parent_hidden_concat = torch.cat(parent_hiddens, dim=-1)  # [batch, 256]
                elif len(parent_hiddens) == 1:
                    # Pad with zeros for missing second parent
                    parent_hidden_concat = torch.cat([parent_hiddens[0], torch.zeros_like(parent_hiddens[0])], dim=-1)
                else:
                    # No parents - use zeros
                    parent_hidden_concat = torch.zeros(batch_size, self.hidden_size * 2, device=device)

                # Duration context: hidden + parents (zero-padded) + VIX + events
                duration_context = torch.cat([hidden, parent_hidden_concat, hidden_vix, event_embed], dim=-1)

                # Probabilistic duration (mean + std)
                duration_mean = self.duration_heads[f'{tf}_mean'](duration_context)
                # 🛡️ Clamp log_std to prevent variance collapse (exp(-6) to exp(6) = safe range)
                duration_log_std = self.duration_heads[f'{tf}_log_std'](duration_context).clamp(-3, 3)
                duration_std = torch.exp(duration_log_std).clamp(1, 20)

                # Three projection scenarios
                conservative = (duration_mean - duration_std).clamp(1, 48)
                expected = duration_mean.clamp(1, 48)
                aggressive = (duration_mean + duration_std).clamp(1, 48)

                # Duration confidence = inverse of relative uncertainty
                duration_confidence = 1.0 - (duration_std / (duration_mean + 1e-6)).clamp(0, 0.95)

                duration_outputs[tf] = {
                    'mean': duration_mean,
                    'log_std': duration_log_std,
                    'std': duration_std,
                    'conservative': conservative,
                    'expected': expected,
                    'aggressive': aggressive,
                    'confidence': duration_confidence,
                }

                # v5.2: Duration-aware projection scaling
                # If predicted duration < 24 bars, scale down projections proportionally
                # If > 24 bars, scale up (clamped to reasonable range)
                duration_scale = (duration_mean / 24.0).clamp(0.3, 2.0)  # [batch, 1]

            # v5.0: Channel-based predictions (Option B: Base + Adjustment)
            if self.use_channel_projections and hasattr(self, 'projection_adjusters'):
                # Try to extract geometric projections from features
                proj_features = self._extract_projection_features_from_tensor(
                    timeframe_data[tf], tf, symbol='tsla'
                )

                if proj_features is not None and hasattr(self, 'projection_extractors'):
                    # Use ChannelProjectionExtractor for validity-weighted geometric base
                    proj_output = self.projection_extractors[tf](
                        hidden_state=hidden,
                        projections=proj_features['projections'],
                        quality_scores=proj_features['quality_scores'],
                        r_squared=proj_features['r_squared'],
                        complete_cycles=proj_features['complete_cycles'],
                        position=proj_features['position']
                    )

                    # Base is weighted geometric projection
                    base_high = proj_output['weighted_high']
                    base_low = proj_output['weighted_low']
                    validity_weights = proj_output['validity_weights']
                else:
                    # Fallback: Use learned heads (will learn to approximate projections)
                    base_high = self.timeframe_heads[f'{tf}_high'](hidden)
                    base_low = self.timeframe_heads[f'{tf}_low'](hidden)
                    validity_weights = None

                # Adjustment network: learns corrections to base projection
                # Input: hidden state + base predictions
                adjuster_input = torch.cat([hidden, base_high, base_low], dim=-1)  # [batch, hidden_size + 2]
                adjustment = self.projection_adjusters[tf](adjuster_input)  # [batch, 2]

                # Final prediction = base + adjustment (OPTION B)
                pred_high = base_high + adjustment[:, 0:1]
                pred_low = base_low + adjustment[:, 1:2]

                # v5.2: Apply duration-aware scaling
                # Shorter expected duration → smaller price move
                # Longer expected duration → larger price move
                if isinstance(duration_scale, torch.Tensor):
                    pred_high = pred_high * duration_scale
                    pred_low = pred_low * duration_scale

                # v5.0: Store metadata for interpretability (only during eval, skip in training for speed)
                if not self.training:
                    metadata = {
                        'base_high': base_high.detach(),
                        'base_low': base_low.detach(),
                        'adjustment_high': adjustment[:, 0:1].detach(),
                        'adjustment_low': adjustment[:, 1:2].detach(),
                        'final_high': pred_high.detach(),
                        'final_low': pred_low.detach(),
                        'mode': 'geometric' if proj_features is not None else 'learned_approximation'
                    }

                    # v5.1: Add window selection info
                    if validity_weights is not None:
                        metadata['validity_weights'] = validity_weights.detach()
                        if 'best_window_idx' in proj_output:
                            metadata['best_window_idx'] = proj_output['best_window_idx'].detach()
                            # Map to actual window size for interpretability
                            window_sizes = [168, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10]
                            best_idx = proj_output['best_window_idx'][0].item()  # First sample
                            metadata['selected_window_size'] = window_sizes[best_idx] if best_idx < len(window_sizes) else None

                    projection_metadata[tf] = metadata
            else:
                # v4.x behavior: Direct neural net prediction
                pred_high = self.timeframe_heads[f'{tf}_high'](hidden)
                pred_low = self.timeframe_heads[f'{tf}_low'](hidden)

                # v5.2: Apply duration-aware scaling (same as projection case)
                if isinstance(duration_scale, torch.Tensor):
                    pred_high = pred_high * duration_scale
                    pred_low = pred_low * duration_scale

            # =====================================================================
            # v5.2: VALIDITY PREDICTION for this TF (compute BEFORE using as confidence)
            # (Duration already computed above before projections)
            # =====================================================================
            validity = None
            if hasattr(self, 'validity_heads') and tf in self.validity_heads:
                # Get quality score and position from projection features (or use defaults)
                if proj_features is not None:
                    quality_score = proj_features['quality_scores'].mean(dim=-1, keepdim=True)  # [batch, 1]
                    position = proj_features['position'].mean(dim=-1, keepdim=True)  # [batch, 1]
                else:
                    # Fallback: use old confidence head temporarily, will update below
                    old_conf = torch.sigmoid(self.timeframe_heads[f'{tf}_conf'](hidden))
                    quality_score = old_conf
                    position = torch.full((batch_size, 1), 0.5, device=device)

                # Validity input: hidden + VIX + events + [quality, position]
                validity_input = torch.cat([
                    hidden, hidden_vix, event_embed,
                    quality_score, position
                ], dim=-1)

                # Forward-looking validity prediction
                validity = self.validity_heads[tf](validity_input)
                validity_outputs[tf] = validity

            # v5.2: Use validity as confidence if available, otherwise old confidence head
            if validity is not None:
                pred_conf = validity  # NEW: Use forward-looking validity!
            else:
                pred_conf = torch.sigmoid(self.timeframe_heads[f'{tf}_conf'](hidden))  # Fallback: old confidence

            layer_predictions.extend([pred_high, pred_low, pred_conf])
            layer_confidences.append(pred_conf)

        # =========================================================================
        # PHYSICS-INSPIRED PROCESSING
        # =========================================================================
        # Build dict of final hidden states for physics modules
        tf_hidden_dict = {tf: all_hidden[i] for i, tf in enumerate(self.TIMEFRAMES) if i < len(all_hidden)}

        # Apply Coulomb attention - dynamic cross-timeframe attention
        if hasattr(self, 'coulomb_attention') and len(tf_hidden_dict) == len(self.TIMEFRAMES):
            tf_hidden_dict = self.coulomb_attention(tf_hidden_dict)
            # Update all_hidden with attended states
            all_hidden = [tf_hidden_dict[tf] for tf in self.TIMEFRAMES]

        # Apply interaction hierarchy - V₁, V₂, V₃ structured interactions
        if hasattr(self, 'interaction_hierarchy') and len(tf_hidden_dict) == len(self.TIMEFRAMES):
            tf_hidden_dict = self.interaction_hierarchy(tf_hidden_dict)
            all_hidden = [tf_hidden_dict[tf] for tf in self.TIMEFRAMES]

        # =========================================================================
        # FUSION HEAD vs PHYSICS-BASED AGGREGATION
        # =========================================================================
        # Concatenate all layer predictions: 11 × 3 = 33
        all_layer_preds = torch.cat(layer_predictions, dim=-1)  # [batch, 33]

        # Add market state
        if market_state is None:
            market_state = torch.zeros(batch_size, self.MARKET_STATE_DIM, device=device)

        self.last_market_state = market_state.detach()

        # v5.3: Locked to Physics-Only - SELECT best channel
        # Extract per-TF predictions from layer_predictions
        per_tf_highs = []
        per_tf_lows = []
        per_tf_confs = []
        for i in range(len(self.TIMEFRAMES)):
            idx = i * 3
            per_tf_highs.append(layer_predictions[idx])     # [batch, 1]
            per_tf_lows.append(layer_predictions[idx + 1])  # [batch, 1]
            per_tf_confs.append(layer_predictions[idx + 2]) # [batch, 1]

        per_tf_highs = torch.cat(per_tf_highs, dim=-1)  # [batch, 11]
        per_tf_lows = torch.cat(per_tf_lows, dim=-1)    # [batch, 11]
        per_tf_confs = torch.cat(per_tf_confs, dim=-1)  # [batch, 11]

        # SELECT the most confident TF (not weighted average!)
        best_tf_idx = torch.argmax(per_tf_confs, dim=-1)  # [batch]

        # Gather the best TF's predictions for each sample in batch
        batch_indices = torch.arange(per_tf_highs.shape[0], device=per_tf_highs.device)
        final_pred_high = per_tf_highs[batch_indices, best_tf_idx].unsqueeze(-1)  # [batch, 1]
        final_pred_low = per_tf_lows[batch_indices, best_tf_idx].unsqueeze(-1)    # [batch, 1]
        final_pred_conf = per_tf_confs[batch_indices, best_tf_idx].unsqueeze(-1)  # [batch, 1]

        # Store selection info for interpretability
        selection_info = {
            'best_tf_idx': best_tf_idx,
            'best_tf_name': [self.TIMEFRAMES[idx.item()] for idx in best_tf_idx],
            'per_tf_highs': per_tf_highs,
            'per_tf_lows': per_tf_lows,
            'per_tf_confs': per_tf_confs,
        }

        # For multi-task heads, create fusion_hidden from physics-enhanced states
        fusion_input = torch.cat([all_layer_preds, market_state], dim=-1)
        fusion_hidden = F.relu(self.fusion_fc1(fusion_input))
        fusion_hidden = F.relu(self.fusion_fc2(fusion_hidden))

        # =========================================================================
        # PHYSICS-INSPIRED OUTPUTS
        # =========================================================================
        # Phase classification
        if hasattr(self, 'phase_classifier') and len(tf_hidden_dict) == len(self.TIMEFRAMES):
            phase_output = self.phase_classifier(tf_hidden_dict)
        else:
            phase_output = None

        # Energy-based confidence adjustment
        if hasattr(self, 'energy_scorer') and len(tf_hidden_dict) == len(self.TIMEFRAMES):
            energy_output = self.energy_scorer(
                tf_hidden_dict,
                base_confidence=final_pred_conf.squeeze(-1)
            )
            # Use energy-adjusted confidence for final prediction
            adjusted_conf = energy_output['adjusted_confidence'].unsqueeze(-1)
            predictions = torch.cat([final_pred_high, final_pred_low, adjusted_conf], dim=-1)
        else:
            energy_output = None
            predictions = torch.cat([final_pred_high, final_pred_low, final_pred_conf], dim=-1)

        # Build output dict
        output_dict = {
            'hidden_states': layer_hidden_states,
            'layer_predictions': {},
        }

        # v5.0: Add projection metadata (interpretability)
        if projection_metadata:
            output_dict['projections'] = projection_metadata

        # Add physics outputs
        if phase_output is not None:
            output_dict['phase'] = phase_output
        if energy_output is not None:
            output_dict['energy'] = energy_output

        # v5.1: Add channel selection info (Physics-Only mode)
        if not self.use_fusion_head:
            output_dict['channel_selection'] = selection_info

        # =========================================================================
        # v5.2: DURATION, VALIDITY, AND COMPOSITOR OUTPUTS
        # =========================================================================
        # Add duration outputs (probabilistic)
        if duration_outputs:
            output_dict['duration'] = duration_outputs

        # Add validity outputs (forward-looking)
        if validity_outputs:
            output_dict['validity'] = validity_outputs

        # =========================================================================
        # v5.3: HIERARCHICAL CONTAINMENT ANALYSIS (interpretability)
        # =========================================================================
        if self.containment_checker and not self.training:
            # Build projections dict for containment checking
            all_projections_dict = {}
            all_validities_dict = {}

            for i, tf in enumerate(self.TIMEFRAMES):
                # Get predictions from layer_predictions
                if i * 3 + 2 < len(layer_predictions):
                    all_projections_dict[tf] = {
                        'high': layer_predictions[i*3][0, 0].item(),
                        'low': layer_predictions[i*3 + 1][0, 0].item(),
                    }
                if tf in validity_outputs:
                    all_validities_dict[tf] = validity_outputs[tf][0, 0].item()

            # Determine selected TF (use locals() not dir())
            if not self.use_fusion_head and 'selection_info' in locals():
                selected_tf = self.TIMEFRAMES[selection_info['best_tf_idx'][0].item()]
            else:
                # Use highest validity TF
                if all_validities_dict:
                    selected_tf = max(all_validities_dict, key=all_validities_dict.get)
                else:
                    selected_tf = 'daily'  # Fallback

            # Check containment
            containment_results = self.containment_checker.check_all_containments(
                selected_tf,
                all_projections_dict,
                all_validities_dict
            )
            output_dict['containment'] = containment_results

        # Add VIX hidden state
        output_dict['hidden_vix'] = hidden_vix

        # Add event embedding
        output_dict['event_embed'] = event_embed

        # Determine selected TF for compositor (use physics selection if available)
        if not self.use_fusion_head and 'selection_info' in locals():
            selected_tf = self.TIMEFRAMES[selection_info['best_tf_idx'][0].item()]
        else:
            # Use highest confidence TF
            conf_tensor = torch.cat(layer_confidences, dim=-1)  # [batch, 11]
            best_tf_idx = torch.argmax(conf_tensor, dim=-1)[0].item()
            selected_tf = self.TIMEFRAMES[best_tf_idx]

        # Run Multi-Phase Compositor
        if hasattr(self, 'compositor'):
            compositor_output = self.compositor(
                all_hidden=tf_hidden_dict,
                hidden_vix=hidden_vix,
                event_embed=event_embed,
                current_tf=selected_tf,
                timeframes=self.TIMEFRAMES
            )
            output_dict['compositor'] = compositor_output
            output_dict['selected_tf'] = selected_tf

        # Store per-layer predictions
        for i, tf in enumerate(self.TIMEFRAMES):
            idx = i * 3
            output_dict['layer_predictions'][tf] = torch.cat([
                layer_predictions[idx],
                layer_predictions[idx + 1],
                layer_predictions[idx + 2]
            ], dim=-1)

        # =========================================================================
        # MULTI-TASK PREDICTIONS
        # =========================================================================
        if self.multi_task:
            hit_band_pred = self.hit_band_head(fusion_hidden)
            hit_target_pred = self.hit_target_head(fusion_hidden)
            expected_return_pred = self.expected_return_head(fusion_hidden)
            overshoot_pred = self.overshoot_head(fusion_hidden)

            continuation_duration_pred = self.continuation_duration_head(fusion_hidden)
            continuation_gain_pred = self.continuation_gain_head(fusion_hidden)
            continuation_confidence_pred = self.continuation_confidence_head(fusion_hidden)

            adaptive_horizon_pred = self.adaptive_horizon_head(fusion_hidden)
            adaptive_conf_score_pred = self.adaptive_conf_score_head(fusion_hidden)

            # Adaptive Projection uses all 11 hidden states + confidences
            all_hidden_tensor = torch.cat(all_hidden, dim=-1)  # [batch, 11 * 128]
            all_conf_tensor = torch.cat(layer_confidences, dim=-1)  # [batch, 11]
            proj_input = torch.cat([all_hidden_tensor, all_conf_tensor], dim=-1)

            proj = self.adaptive_projection(proj_input)
            price_change_pct = proj[:, 0]
            horizon_bars_log = proj[:, 1]
            adaptive_confidence = torch.sigmoid(proj[:, 2])

            horizon_bars = torch.exp(horizon_bars_log) * 24
            horizon_bars = torch.clamp(horizon_bars, 24, 2016)

            # Determine dominant layer
            layer_weights = torch.cat(layer_confidences, dim=1)  # [batch, 11]
            dominant_layer_idx = torch.argmax(layer_weights, dim=1)

            output_dict['multi_task'] = {
                'hit_band': hit_band_pred,
                'hit_target': hit_target_pred,
                'expected_return': expected_return_pred,
                'overshoot': overshoot_pred,
                'continuation_duration': continuation_duration_pred,
                'continuation_gain': continuation_gain_pred,
                'continuation_confidence': continuation_confidence_pred,
                'adaptive_horizon': adaptive_horizon_pred,
                'adaptive_conf_score': adaptive_conf_score_pred,
                'price_change_pct': price_change_pct,
                'horizon_bars_log': horizon_bars_log,
                'horizon_bars': horizon_bars,
                'adaptive_confidence': adaptive_confidence,
                'dominant_layer_idx': dominant_layer_idx
            }

            # Breakout predictions
            if hasattr(self, 'breakout_prob_head'):
                breakout_prob = self.breakout_prob_head(fusion_hidden)
                breakout_direction = self.breakout_direction_head(fusion_hidden)
                breakout_bars_log = self.breakout_bars_head(fusion_hidden)
                breakout_confidence = self.breakout_confidence_head(fusion_hidden)

                breakout_bars = torch.exp(breakout_bars_log).clamp(1, 100)

                output_dict['breakout'] = {
                    'probability': breakout_prob,
                    'direction': breakout_direction,
                    'bars_until': breakout_bars,
                    'confidence': breakout_confidence,
                    'is_trained': False
                }

            # v4.3: Per-TF continuation predictions
            # Each timeframe gets duration/gain/confidence from its own hidden state
            if hasattr(self, 'per_tf_cont_heads'):
                per_tf_cont = {}
                for i, tf in enumerate(self.TIMEFRAMES):
                    if i < len(all_hidden):
                        tf_hidden = all_hidden[i]  # [batch, hidden_size]
                        duration = self.per_tf_cont_heads[f'{tf}_duration'](tf_hidden)
                        gain = self.per_tf_cont_heads[f'{tf}_gain'](tf_hidden)
                        confidence = self.per_tf_cont_heads[f'{tf}_confidence'](tf_hidden)

                        per_tf_cont[f'cont_{tf}_duration'] = duration
                        per_tf_cont[f'cont_{tf}_gain'] = gain
                        per_tf_cont[f'cont_{tf}_confidence'] = confidence
                    else:
                        # Placeholder if hidden not available for this TF
                        per_tf_cont[f'cont_{tf}_duration'] = torch.zeros(batch_size, 1, device=device)
                        per_tf_cont[f'cont_{tf}_gain'] = torch.zeros(batch_size, 1, device=device)
                        per_tf_cont[f'cont_{tf}_confidence'] = torch.full((batch_size, 1), 0.5, device=device)

                output_dict['per_tf_continuation'] = per_tf_cont

        return predictions, output_dict

    def forward_single_input(
        self,
        x: torch.Tensor,
        market_state: Optional[torch.Tensor] = None,
        vix_sequence: Optional[torch.Tensor] = None,  # v5.2
        events: Optional[List[Dict]] = None,  # v5.2
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Backward-compatible forward for single tensor input.

        Automatically distributes the same features to all timeframes.
        Used during training when dataset provides single feature tensor.

        Args:
            x: Input features [batch, seq_len, features]
            market_state: Market regime features [batch, 12]
            vix_sequence: v5.2 - VIX sequence [batch, 90, 11]
            events: v5.2 - Event list

        Returns:
            Same as forward()
        """
        # Create timeframe_data dict with same data for all (will be improved in dataset)
        # For now, use the same features - dataset restructuring will fix this
        timeframe_data = {}

        # Distribute to timeframes with appropriate sequence lengths
        seq_lens = getattr(project_config, 'TIMEFRAME_SEQUENCE_LENGTHS', {
            '5min': 200, '15min': 200, '30min': 200, '1h': 200,
            '2h': 100, '3h': 100, '4h': 100,
            'daily': 60, 'weekly': 52, 'monthly': 24, '3month': 12,
        })

        batch_size, full_seq_len, n_features = x.shape

        for tf in self.TIMEFRAMES:
            target_len = seq_lens.get(tf, 200)

            if full_seq_len >= target_len:
                # Take last target_len bars
                timeframe_data[tf] = x[:, -target_len:, :]
            else:
                # Pad with zeros if not enough data
                padding = torch.zeros(batch_size, target_len - full_seq_len, n_features, device=x.device)
                timeframe_data[tf] = torch.cat([padding, x], dim=1)

        return self.forward(timeframe_data, market_state, vix_sequence=vix_sequence, events=events)

    def clear_cached_states(self):
        """Clear cached hidden states to prevent memory accumulation."""
        self.last_inputs = {}
        self.last_market_state = None

    def predict(
        self,
        x: torch.Tensor,
        market_state: Optional[torch.Tensor] = None,
        h: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Generate predictions dict (ModelBase interface).

        Args:
            x: Input features [batch, seq_len, features] or Dict[str, Tensor]
            market_state: Optional market state [batch, 12]
            h: Optional hidden states dict

        Returns:
            predictions_dict: Dict with all prediction outputs
        """
        self.eval()

        with torch.no_grad():
            # Handle both dict and tensor input
            if isinstance(x, dict):
                predictions, output_dict = self.forward(x, market_state, h)
            else:
                predictions, output_dict = self.forward_single_input(x, market_state)

            # Extract predictions
            pred_high = predictions[0, 0].item()
            pred_low = predictions[0, 1].item()
            pred_conf = predictions[0, 2].item()

            result = {
                'predicted_high': pred_high,
                'predicted_low': pred_low,
                'confidence': pred_conf,
                'hidden_states': output_dict['hidden_states'],
            }

            # Add physics outputs if available
            if 'phase' in output_dict:
                result['phase'] = {
                    'id': output_dict['phase']['phase_id'][0].item(),
                    'name': output_dict['phase']['phase_names'][output_dict['phase']['phase_id'][0].item()],
                    'probs': output_dict['phase']['phase_probs'][0].cpu().numpy().tolist(),
                    'entropy': output_dict['phase']['phase_entropy'][0].item()
                }
            if 'energy' in output_dict:
                result['energy'] = {
                    'total': output_dict['energy']['energy'][0].item(),
                    'confidence': output_dict['energy']['energy_confidence'][0].item(),
                    'temperature': output_dict['energy']['temperature'].item()
                }

            # Add per-layer predictions
            for tf in self.TIMEFRAMES:
                if tf in output_dict['layer_predictions']:
                    layer_pred = output_dict['layer_predictions'][tf][0]
                    result[f'{tf}_pred_high'] = layer_pred[0].item()
                    result[f'{tf}_pred_low'] = layer_pred[1].item()
                    result[f'{tf}_pred_conf'] = layer_pred[2].item()

            # Add multi-task predictions
            if self.multi_task and 'multi_task' in output_dict:
                mt = output_dict['multi_task']
                result['hit_band_pred'] = mt['hit_band'][0, 0].item()
                result['hit_target_pred'] = mt['hit_target'][0, 0].item()
                result['expected_return_pred'] = mt['expected_return'][0, 0].item()
                result['overshoot_pred'] = mt['overshoot'][0, 0].item()

            # Add breakout predictions
            if 'breakout' in output_dict:
                bo = output_dict['breakout']
                result['breakout'] = {
                    'probability': bo['probability'][0, 0].item(),
                    'direction': bo['direction'][0, 0].item(),
                    'direction_label': 'up' if bo['direction'][0, 0].item() > 0.5 else 'down',
                    'bars_until': bo['bars_until'][0, 0].item(),
                    'confidence': bo['confidence'][0, 0].item(),
                    'is_trained': bo['is_trained']
                }

            return result

    def project_channel(
        self,
        x: torch.Tensor,
        current_price: float,
        horizons: List[int] = [15, 30, 60, 120, 240],
        min_confidence: float = 0.65,
        market_state: Optional[torch.Tensor] = None
    ) -> List[Dict[str, Any]]:
        """
        Project channel forward with confidence decay.

        Args:
            x: Current features [1, seq_len, features]
            current_price: Current TSLA price
            horizons: List of minutes ahead to project
            min_confidence: Minimum confidence threshold
            market_state: Optional market state [1, 12]

        Returns:
            projections: List of prediction dicts per horizon
        """
        from datetime import datetime, timedelta

        self.eval()
        projections = []

        with torch.no_grad():
            for horizon in horizons:
                pred_dict = self.predict(x, market_state)

                # Confidence decay factor
                decay_factor = np.exp(-horizon / 60.0)
                adjusted_confidence = pred_dict['confidence'] * decay_factor

                if adjusted_confidence >= min_confidence:
                    pred_high_pct = pred_dict['predicted_high']
                    pred_low_pct = pred_dict['predicted_low']

                    pred_high_price = current_price * (1 + pred_high_pct / 100.0)
                    pred_low_price = current_price * (1 + pred_low_pct / 100.0)

                    validation_time = datetime.now() + timedelta(minutes=horizon)

                    projection = {
                        'horizon_minutes': horizon,
                        'predicted_high': pred_high_pct,
                        'predicted_low': pred_low_pct,
                        'predicted_high_price': pred_high_price,
                        'predicted_low_price': pred_low_price,
                        'confidence': adjusted_confidence,
                        'confidence_original': pred_dict['confidence'],
                        'decay_factor': decay_factor,
                        'validation_time': validation_time.isoformat(),
                    }

                    # Add physics outputs if available
                    if 'phase' in pred_dict:
                        projection['phase'] = pred_dict['phase']
                    if 'energy' in pred_dict:
                        projection['energy'] = pred_dict['energy']

                    if 'breakout' in pred_dict:
                        projection['breakout'] = pred_dict['breakout']

                    projections.append(projection)
                else:
                    break

        return projections

    def update_online(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lr: float = 0.0001,
        layers: List[str] = None
    ):
        """
        Online learning update from prediction errors.

        Args:
            x: Input features (tensor or dict)
            y: True targets [batch, 2] - [actual_high, actual_low]
            lr: Learning rate for online update
            layers: List of timeframe names to update (None = all)
        """
        self.train()

        if layers is None:
            layers = self.TIMEFRAMES

        # Forward pass
        if isinstance(x, dict):
            predictions, _ = self.forward(x)
        else:
            predictions, _ = self.forward_single_input(x)

        # Calculate loss
        pred_high = predictions[:, 0]
        pred_low = predictions[:, 1]
        target_high = y[:, 0]
        target_low = y[:, 1]

        loss = F.mse_loss(pred_high, target_high) + F.mse_loss(pred_low, target_low)

        # Backward pass
        loss.backward()

        # Update specified layers
        with torch.no_grad():
            for tf in layers:
                if tf in self.timeframe_layers:
                    for param in self.timeframe_layers[tf].parameters():
                        if param.grad is not None:
                            param -= lr * param.grad

                    for head_name in [f'{tf}_high', f'{tf}_low', f'{tf}_conf']:
                        if head_name in self.timeframe_heads:
                            for param in self.timeframe_heads[head_name].parameters():
                                if param.grad is not None:
                                    param -= lr * param.grad

            # Always update fusion head
            for module in [self.fusion_fc1, self.fusion_fc2,
                          self.fusion_fc_high, self.fusion_fc_low, self.fusion_fc_conf]:
                for param in module.parameters():
                    if param.grad is not None:
                        param -= lr * param.grad

        self.zero_grad()

    def save_checkpoint(self, path: str, metadata: Dict = None):
        """Save model checkpoint with metadata."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_type': 'HierarchicalLNN',
            'model_version': '4.2',  # Updated for use_fusion_head flag
            'input_sizes': self.input_sizes,
            'hidden_size': self.hidden_size,
            'internal_neurons_ratio': self.internal_neurons_ratio,
            'total_neurons': self.total_neurons,
            'device_type': self.device_type,
            'multi_task': self.multi_task,
            'use_fusion_head': self.use_fusion_head,  # v4.2: Physics-only mode option
            'timeframes': self.TIMEFRAMES,
            'num_layers': self.NUM_LAYERS,
            'fusion_input_dim': self.FUSION_INPUT_DIM,
            'has_physics_modules': True,  # GWC-inspired physics attention
        }

        if metadata:
            checkpoint.update(metadata)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> Dict:
        """Load model checkpoint and return metadata."""
        checkpoint = torch.load(path, map_location=self.device_type)

        # Handle DataParallel checkpoints
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

        self.load_state_dict(state_dict)

        metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
        return metadata


def load_hierarchical_model(model_path: str, device: str = 'cpu') -> HierarchicalLNN:
    """
    Load a trained hierarchical model from checkpoint.

    Args:
        model_path: Path to .pth checkpoint
        device: 'cuda', 'mps', or 'cpu'

    Returns:
        model: Loaded HierarchicalLNN instance
    """
    checkpoint = torch.load(model_path, map_location=device)

    # Get parameters from checkpoint
    args = checkpoint.get('args', {})

    # v4.0: Use input_sizes dict if available
    input_sizes = checkpoint.get('input_sizes')

    # Backward compatibility: handle old single input_size
    input_size = checkpoint.get('input_size') or args.get('input_size')
    if input_size is not None and input_sizes is None:
        # Old format: create dict with same size for all
        input_sizes = {tf: input_size for tf in HierarchicalLNN.TIMEFRAMES}

    # Get other parameters
    hidden_size = checkpoint.get('hidden_size') or args.get('hidden_size', 128)
    internal_neurons_ratio = checkpoint.get('internal_neurons_ratio') or args.get('internal_neurons_ratio', 2.0)

    # v4.2: Get use_fusion_head (check args first, then top-level, default False for Physics-Only)
    use_fusion_head = args.get('use_fusion_head', checkpoint.get('use_fusion_head', False))

    # v5.0: Get use_geometric_base (check args first, then top-level, default True)
    use_geometric_base = args.get('use_geometric_base', checkpoint.get('use_geometric_base', True))

    # v5.3.1: Get information_flow (critical for correct CfC processing order!)
    information_flow = args.get('information_flow', checkpoint.get('information_flow', 'bottom_up'))

    # Create model
    model = HierarchicalLNN(
        input_sizes=input_sizes,
        hidden_size=hidden_size,
        internal_neurons_ratio=internal_neurons_ratio,
        device=device,
        multi_task=checkpoint.get('multi_task') or args.get('multi_task', True),
        use_fusion_head=use_fusion_head,
        use_geometric_base=use_geometric_base,
        information_flow=information_flow  # v5.3.1: Match training flow!
    )

    # Handle DataParallel checkpoints
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    # Load state dict with strict=False for compatibility
    incompatible = model.load_state_dict(state_dict, strict=False)

    if incompatible.missing_keys:
        non_critical = [k for k in incompatible.missing_keys if 'breakout' in k]
        critical = [k for k in incompatible.missing_keys if 'breakout' not in k]
        if critical:
            print(f"  ⚠️ Missing keys: {critical}")
        if non_critical:
            print(f"  ℹ️ New heads initialized randomly: {non_critical}")

    if incompatible.unexpected_keys:
        print(f"  ⚠️ Unexpected keys: {incompatible.unexpected_keys}")

    return model
