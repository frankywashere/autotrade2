"""
Hierarchical Liquid Neural Network for Multi-Scale Stock Prediction (v4.0)

Architecture:
- 11 CfC layers (one per timeframe: 5min, 15min, 30min, 1h, 2h, 3h, 4h, daily, weekly, monthly, 3month)
- Each layer receives NATIVE OHLC data at its timeframe (not downsampled 1-min)
- Bottom-up hidden state passing (fast → slow)
- Fusion Head: 33 layer predictions + 12 market_state = 45 dims

Key Features:
- Proper multi-timeframe learning (each layer sees meaningful bar counts)
- Market state integration (VIX, volatility regime, event proximity)
- Online learning support for continuous adaptation
- Channel projection with confidence decay
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

# Import config for timeframe settings
import sys
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))
import config as project_config


class ChannelProjectionExtractor(nn.Module):
    """
    v5.0: Extracts and weights channel projections from features.

    For each timeframe, combines 21 window projections weighted by learned validity.
    Each window's projection is a geometric channel projection (slope × horizon).
    Neural network learns which windows to trust based on quality metrics.
    """

    def __init__(self, timeframe: str, hidden_size: int, num_windows: int = 21):
        """
        Args:
            timeframe: Timeframe name (e.g., '5min', '1h', 'daily')
            hidden_size: Size of CfC hidden state
            num_windows: Number of window sizes (default: 21)
        """
        super().__init__()
        self.timeframe = timeframe
        self.hidden_size = hidden_size
        self.num_windows = num_windows

        # Validity predictor: learns which window projections to trust
        # Input: hidden_state (128) + quality_score (1) + r_squared (1) + complete_cycles (1) + position (1) = 132
        self.validity_net = nn.Sequential(
            nn.Linear(hidden_size + 4, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 0-1 validity score
        )

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
        Combine window projections weighted by learned validity.

        Args:
            hidden_state: [batch, hidden_size] - CfC layer output
            projections: [batch, num_windows, 2] - (high, low) projections for each window
            quality_scores: [batch, num_windows] - Quality score per window
            r_squared: [batch, num_windows] - R² per window
            complete_cycles: [batch, num_windows] - Complete cycles per window
            position: [batch, num_windows] - Channel position per window

        Returns:
            dict with weighted_high, weighted_low, validity_weights
        """
        batch_size = hidden_state.shape[0]

        # Calculate validity for each window
        validity_list = []
        for w in range(self.num_windows):
            # Combine hidden state with window-specific quality metrics
            window_context = torch.cat([
                hidden_state,
                quality_scores[:, w:w+1],
                r_squared[:, w:w+1],
                complete_cycles[:, w:w+1],
                position[:, w:w+1]
            ], dim=-1)  # [batch, hidden_size + 4]

            validity = self.validity_net(window_context)  # [batch, 1]
            validity_list.append(validity)

        # Stack validities: [batch, num_windows]
        validity_weights = torch.cat(validity_list, dim=-1)

        # Normalize to sum to 1 (softmax)
        validity_weights_norm = F.softmax(validity_weights, dim=-1)  # [batch, num_windows]

        # Weighted combination of projections
        # projections shape: [batch, num_windows, 2]
        weighted_high = (projections[:, :, 0] * validity_weights_norm).sum(dim=-1, keepdim=True)  # [batch, 1]
        weighted_low = (projections[:, :, 1] * validity_weights_norm).sum(dim=-1, keepdim=True)   # [batch, 1]

        return {
            'weighted_high': weighted_high,
            'weighted_low': weighted_low,
            'validity_weights': validity_weights_norm,  # For interpretability
            'raw_validities': validity_weights  # Before softmax
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
        self.use_channel_projections = True  # v5.0: Enable channel-based predictions

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
        # First layer (5min): just features
        # Subsequent layers: features + previous layer's hidden state

        self.timeframe_layers = nn.ModuleDict()
        self.timeframe_heads = nn.ModuleDict()

        for i, tf in enumerate(self.TIMEFRAMES):
            tf_input_size = self.input_sizes.get(tf, 900)  # Default to 900 if not specified

            # First layer takes only features, subsequent layers add hidden from previous
            if i == 0:
                layer_input_size = tf_input_size
            else:
                layer_input_size = tf_input_size + hidden_size  # Concat previous hidden

            # Create CfC layer with AutoNCP wiring
            wiring = AutoNCP(self.total_neurons, hidden_size)
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

        self.fusion_fc1 = nn.Linear(self.FUSION_INPUT_DIM, 128)
        self.fusion_fc2 = nn.Linear(128, self.fusion_output_dim)
        self.fusion_fc_high = nn.Linear(self.fusion_output_dim, 1)
        self.fusion_fc_low = nn.Linear(self.fusion_output_dim, 1)
        self.fusion_fc_conf = nn.Linear(self.fusion_output_dim, 1)

        # NOTE: Static fusion_weights removed - replaced by dynamic CoulombTimeframeAttention

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
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through 11 hierarchical layers.

        Args:
            x: Either a Dict[str, Tensor] mapping timeframe to features,
               OR a single Tensor [batch, seq_len, features] for backward compatibility
            market_state: Market regime features [batch, 12]
            hidden_states: Optional dict of initial hidden states per timeframe

        Returns:
            predictions: [batch, 3] - [predicted_high, predicted_low, confidence]
            output_dict: Dict with hidden states, layer predictions, multi-task outputs
        """
        # Handle backward compatibility: single tensor input
        if isinstance(x, torch.Tensor):
            return self.forward_single_input(x, market_state)

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

        # v5.0: Initialize projection feature extractor on first forward (need feature names from data)
        # This will be used to extract projection/quality features from input tensors
        if self.projection_feature_extractor is None and self.use_channel_projections:
            # NOTE: For native TF mode, we can't easily get feature names here
            # Will initialize lazily when needed, or require explicit initialization
            # For now, mark as "will initialize when feature names available"
            pass

        # =========================================================================
        # PROCESS EACH TIMEFRAME LAYER
        # =========================================================================
        layer_predictions = []
        layer_hidden_states = {}
        layer_confidences = []
        all_hidden = []
        projection_metadata = {}  # v5.0: Store projection details for interpretability

        prev_hidden = None
        for i, tf in enumerate(self.TIMEFRAMES):
            # Get input for this timeframe
            if tf not in timeframe_data:
                # Skip if timeframe data not provided
                # Use zeros for predictions
                layer_predictions.extend([
                    torch.zeros(batch_size, 1, device=device),
                    torch.zeros(batch_size, 1, device=device),
                    torch.zeros(batch_size, 1, device=device)
                ])
                layer_confidences.append(torch.zeros(batch_size, 1, device=device))
                all_hidden.append(torch.zeros(batch_size, self.hidden_size, device=device))
                continue

            x_tf = timeframe_data[tf]  # [batch, seq_len, features]

            # Concatenate previous hidden state if not first layer
            if i > 0 and prev_hidden is not None:
                # Expand hidden to match sequence length
                seq_len = x_tf.shape[1]
                prev_hidden_expanded = prev_hidden.unsqueeze(1).expand(-1, seq_len, -1)
                x_tf = torch.cat([x_tf, prev_hidden_expanded], dim=-1)

            # Get initial hidden state if provided
            h_init = hidden_states.get(tf, None)

            # Forward through CfC layer
            layer_out, h_new = self.timeframe_layers[tf](x_tf, h_init)

            # Take final hidden state
            hidden = layer_out[:, -1, :]  # [batch, hidden_size]
            layer_hidden_states[tf] = h_new
            all_hidden.append(hidden)
            prev_hidden = hidden

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

                # Store metadata for interpretability
                projection_metadata[tf] = {
                    'base_high': base_high.detach(),
                    'base_low': base_low.detach(),
                    'adjustment_high': adjustment[:, 0:1].detach(),
                    'adjustment_low': adjustment[:, 1:2].detach(),
                    'final_high': pred_high.detach(),
                    'final_low': pred_low.detach(),
                    'validity_weights': validity_weights.detach() if validity_weights is not None else None,
                    'mode': 'geometric' if proj_features is not None else 'learned_approximation'
                }
            else:
                # v4.x behavior: Direct neural net prediction
                pred_high = self.timeframe_heads[f'{tf}_high'](hidden)
                pred_low = self.timeframe_heads[f'{tf}_low'](hidden)

            pred_conf = torch.sigmoid(self.timeframe_heads[f'{tf}_conf'](hidden))

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

        if self.use_fusion_head:
            # Standard fusion head approach
            # Fusion input: 33 + 12 = 45
            fusion_input = torch.cat([all_layer_preds, market_state], dim=-1)

            # Fusion network
            fusion_hidden = F.relu(self.fusion_fc1(fusion_input))
            fusion_hidden = F.relu(self.fusion_fc2(fusion_hidden))

            # Primary predictions
            final_pred_high = self.fusion_fc_high(fusion_hidden)
            final_pred_low = self.fusion_fc_low(fusion_hidden)
            final_pred_conf = torch.sigmoid(self.fusion_fc_conf(fusion_hidden))
        else:
            # Physics-based aggregation: weighted average of per-TF predictions
            # Use per-TF confidences as attention weights (already transformed by physics modules)

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

            # Use confidence as attention weights
            weights = F.softmax(per_tf_confs, dim=-1)  # [batch, 11]

            # Weighted average
            final_pred_high = (per_tf_highs * weights).sum(dim=-1, keepdim=True)  # [batch, 1]
            final_pred_low = (per_tf_lows * weights).sum(dim=-1, keepdim=True)    # [batch, 1]
            final_pred_conf = (per_tf_confs * weights).sum(dim=-1, keepdim=True)  # [batch, 1]

            # For multi-task heads, we still need fusion_hidden
            # Create a simple representation from physics-enhanced hidden states
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
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Backward-compatible forward for single tensor input.

        Automatically distributes the same features to all timeframes.
        Used during training when dataset provides single feature tensor.

        Args:
            x: Input features [batch, seq_len, features]
            market_state: Market regime features [batch, 12]

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

        return self.forward(timeframe_data, market_state)

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

    # v4.2: Get use_fusion_head (default True for backward compatibility)
    use_fusion_head = checkpoint.get('use_fusion_head', True)

    # Create model
    model = HierarchicalLNN(
        input_sizes=input_sizes,
        hidden_size=hidden_size,
        internal_neurons_ratio=internal_neurons_ratio,
        device=device,
        multi_task=checkpoint.get('multi_task') or args.get('multi_task', True),
        use_fusion_head=use_fusion_head
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
