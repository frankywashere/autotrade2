"""
Hierarchical Liquid Neural Network for Multi-Scale Stock Prediction

Architecture:
- Fast Layer: 1-min bars → Learns intraday patterns (ping-pongs, RSI flips)
- Medium Layer: Downsampled to hourly → Learns swing patterns (1-4 hour channels, SPY correlation)
- Slow Layer: Downsampled to daily → Learns macro patterns (weekly/monthly cycles)
- Fusion Head: Adaptive combination based on market regime + news

Key Features:
- Bottom-up hidden state passing (fast → medium → slow)
- Dynamic downsampling via average pooling
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


class HierarchicalLNN(nn.Module, ModelBase):
    """
    Hierarchical Liquid Neural Network with 3 temporal scales.

    Processes 1-min data through fast → medium → slow layers,
    each learning patterns at different temporal resolutions.
    """

    def __init__(
        self,
        input_size: int = None,  # v3.13: Pass from training script (was 473, now ~12,936 with 21-window system)
        hidden_size: int = 128,
        internal_neurons_ratio: float = 2.0,  # Total neurons = hidden_size × ratio
        device: str = 'cpu',
        downsample_fast_to_medium: int = 5,  # 1min → 5min avg
        downsample_medium_to_slow: int = 12,  # 5min → 1hour avg (60/5)
        multi_task: bool = True,  # Enable multi-task heads
    ):
        """
        Initialize hierarchical model with multi-task heads.

        Args:
            input_size: Number of input features (None = auto-detect from first batch)
            hidden_size: Hidden state size / output neurons (default: 128)
            internal_neurons_ratio: Total neurons = hidden_size × ratio (default: 2.0 → 256 total)
            device: 'cuda', 'mps', or 'cpu'
            downsample_fast_to_medium: Downsampling ratio (fast → medium)
            downsample_medium_to_slow: Downsampling ratio (medium → slow)
            multi_task: Enable multi-task prediction heads
        """
        super().__init__()  # Calls nn.Module.__init__

        self.input_size = input_size  # Can be None initially
        self.hidden_size = hidden_size
        self.internal_neurons_ratio = internal_neurons_ratio
        self.total_neurons = int(hidden_size * internal_neurons_ratio)
        self.device_type = device
        self.downsample_fast_to_medium = downsample_fast_to_medium
        self.downsample_medium_to_slow = downsample_medium_to_slow
        self.multi_task = multi_task

        # Fusion output dimension scales with hidden_size (not hardcoded to 64)
        # Allows larger models to have proportionally larger fusion capacity
        self.fusion_output_dim = self.hidden_size // 2  # e.g., 128→64, 256→128

        # Calculate effective sequence lengths after downsampling
        # If input is 200 1-min bars:
        # - Fast: 200 bars (1-min)
        # - Medium: 200 / 5 = 40 bars (5-min)
        # - Slow: 40 / 12 ≈ 3-4 bars (1-hour)
        # Note: This is conceptual - actual downsampling happens in forward()

        # FAST LAYER (1-min scale)
        # Input: [batch, 200, 309] raw features
        # Output: [batch, 128] hidden state
        # total_neurons includes internal processing neurons for richer patterns
        wiring_fast = AutoNCP(self.total_neurons, hidden_size)  # e.g., AutoNCP(256, 128)
        self.fast_layer = CfC(input_size, wiring_fast, batch_first=True)

        # Fast layer output heads
        self.fast_fc_high = nn.Linear(hidden_size, 1)
        self.fast_fc_low = nn.Linear(hidden_size, 1)
        self.fast_fc_conf = nn.Linear(hidden_size, 1)

        # MEDIUM LAYER (5-min scale)
        # Input: [batch, 40, 309 + 128] (downsampled features + fast hidden state)
        # Output: [batch, 128] hidden state
        wiring_medium = AutoNCP(self.total_neurons, hidden_size)  # e.g., AutoNCP(256, 128)
        self.medium_layer = CfC(input_size + hidden_size, wiring_medium, batch_first=True)

        # Medium layer output heads
        self.medium_fc_high = nn.Linear(hidden_size, 1)
        self.medium_fc_low = nn.Linear(hidden_size, 1)
        self.medium_fc_conf = nn.Linear(hidden_size, 1)

        # SLOW LAYER (1-hour scale)
        # Input: [batch, 3-4, 309 + 128] (downsampled features + medium hidden state)
        # Output: [batch, 128] hidden state
        wiring_slow = AutoNCP(self.total_neurons, hidden_size)  # e.g., AutoNCP(256, 128)
        self.slow_layer = CfC(input_size + hidden_size, wiring_slow, batch_first=True)

        # Slow layer output heads
        self.slow_fc_high = nn.Linear(hidden_size, 1)
        self.slow_fc_low = nn.Linear(hidden_size, 1)
        self.slow_fc_conf = nn.Linear(hidden_size, 1)

        # ADAPTIVE FUSION HEAD
        # Combines predictions from all 3 layers
        # Input: 3 predictions (high, low, conf) × 3 layers + market_state + news
        fusion_input_size = 9 + 12 + 768 + 1  # 9 preds + 12 market state + 768 news + 1 news mask
        self.fusion_fc1 = nn.Linear(fusion_input_size, 128)
        self.fusion_fc2 = nn.Linear(128, self.fusion_output_dim)
        self.fusion_fc_high = nn.Linear(self.fusion_output_dim, 1)
        self.fusion_fc_low = nn.Linear(self.fusion_output_dim, 1)
        self.fusion_fc_conf = nn.Linear(self.fusion_output_dim, 1)

        # Learnable fusion weights (initialized equally)
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)  # [fast, medium, slow]

        # MULTI-TASK HEADS (Phase 3)
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
            self.continuation_duration_head = nn.Linear(self.fusion_output_dim, 1)
            self.continuation_gain_head = nn.Linear(self.fusion_output_dim, 1)
            self.continuation_confidence_head = nn.Sequential(
                nn.Linear(self.fusion_output_dim, self.fusion_output_dim // 2),
                nn.ReLU(),
                nn.Linear(self.fusion_output_dim // 2, 1),
                nn.Sigmoid()
            )

            # Adaptive horizon predictor (for adaptive continuation mode)
            # Predicts the optimal horizon (24-48 bars) based on market conditions
            self.adaptive_horizon_head = nn.Sequential(
                nn.Linear(self.fusion_output_dim, self.fusion_output_dim // 2),
                nn.ReLU(),
                nn.Linear(self.fusion_output_dim // 2, 1),
                nn.Sigmoid()  # Output 0-1, will be scaled to 24-48 bars
            )
            self.adaptive_conf_score_head = nn.Sequential(
                nn.Linear(self.fusion_output_dim, self.fusion_output_dim // 2),
                nn.ReLU(),
                nn.Linear(self.fusion_output_dim // 2, 1),
                nn.Sigmoid()  # Confidence score 0-1
            )

            # Adaptive Projection: Dynamic timescale selection and horizon prediction
            self.adaptive_projection = nn.Sequential(
                nn.Linear(self.hidden_size * 3 + 3, 256),  # 3 layers' hidden + 3 confs
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 3)  # [price_change_pct, horizon_bars_log, confidence]
            )

            # Breakout Prediction Heads (v3.21)
            # Uses fusion output to predict channel breakouts
            # Probability of breakout occurring in next N bars (0-1)
            self.breakout_prob_head = nn.Sequential(
                nn.Linear(self.fusion_output_dim, self.fusion_output_dim // 2),
                nn.ReLU(),
                nn.Linear(self.fusion_output_dim // 2, 1),
                nn.Sigmoid()
            )

            # Breakout direction if it occurs: 1=up, 0=down (probability of up-breakout)
            self.breakout_direction_head = nn.Sequential(
                nn.Linear(self.fusion_output_dim, self.fusion_output_dim // 2),
                nn.ReLU(),
                nn.Linear(self.fusion_output_dim // 2, 1),
                nn.Sigmoid()
            )

            # Expected bars until breakout (regression, log-transformed)
            # Output will be exp() transformed to get actual bars (1-100 range)
            self.breakout_bars_head = nn.Sequential(
                nn.Linear(self.fusion_output_dim, self.fusion_output_dim // 2),
                nn.ReLU(),
                nn.Linear(self.fusion_output_dim // 2, 1)
            )

            # Breakout confidence: separate from probability, indicates model certainty
            self.breakout_confidence_head = nn.Sequential(
                nn.Linear(self.fusion_output_dim, self.fusion_output_dim // 2),
                nn.ReLU(),
                nn.Linear(self.fusion_output_dim // 2, 1),
                nn.Sigmoid()
            )

        # Move to device
        self.to(device)

        # Track last inputs for online learning
        self.last_fast_input = None
        self.last_medium_input = None
        self.last_slow_input = None
        self.last_market_state = None
        self.last_news_vec = None

    def downsample_features(
        self,
        features: torch.Tensor,
        ratio: int,
        method: str = 'avg_pool'
    ) -> torch.Tensor:
        """
        Downsample feature tensor via average pooling.

        Args:
            features: [batch, seq_len, feature_dim]
            ratio: Downsampling ratio (e.g., 5 for 1min → 5min)
            method: 'avg_pool' or 'max_pool'

        Returns:
            downsampled: [batch, seq_len // ratio, feature_dim]
        """
        batch_size, seq_len, feature_dim = features.shape

        # Truncate to multiple of ratio
        truncated_len = (seq_len // ratio) * ratio
        features_truncated = features[:, :truncated_len, :]

        # Reshape for pooling: [batch, seq_len, feature_dim] → [batch, feature_dim, seq_len]
        features_transposed = features_truncated.transpose(1, 2)

        # Apply 1D pooling
        if method == 'avg_pool':
            pooled = F.avg_pool1d(features_transposed, kernel_size=ratio, stride=ratio)
        else:  # max_pool
            pooled, _ = F.max_pool1d(features_transposed, kernel_size=ratio, stride=ratio, return_indices=False)

        # Transpose back: [batch, feature_dim, seq_len_new] → [batch, seq_len_new, feature_dim]
        downsampled = pooled.transpose(1, 2)

        return downsampled

    def forward(
        self,
        x: torch.Tensor,
        market_state: Optional[torch.Tensor] = None,
        news_vec: Optional[torch.Tensor] = None,
        news_mask: Optional[torch.Tensor] = None,
        h_fast: Optional[torch.Tensor] = None,
        h_medium: Optional[torch.Tensor] = None,
        h_slow: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through hierarchical layers.

        Args:
            x: Input features [batch, 200, 299] (1-min bars)
            market_state: Market regime features [batch, 12] (optional)
            news_vec: News embeddings [batch, 768] (optional)
            news_mask: News availability mask [batch, 1] (optional)
            h_fast/h_medium/h_slow: Initial hidden states (optional)

        Returns:
            predictions: [batch, 3] - [predicted_high, predicted_low, confidence]
            hidden_states: Dict with 'fast', 'medium', 'slow', and layer predictions
        """
        batch_size = x.shape[0]

        # Store inputs for online learning
        self.last_fast_input = x.detach()

        # ===== FAST LAYER (1-min scale) =====
        # Process full 200 1-min bars
        fast_out, h_fast_new = self.fast_layer(x, h_fast)

        # Take final hidden state
        fast_hidden = fast_out[:, -1, :]  # [batch, 128]

        # Fast layer predictions
        fast_pred_high = self.fast_fc_high(fast_hidden)  # [batch, 1]
        fast_pred_low = self.fast_fc_low(fast_hidden)
        fast_pred_conf = torch.sigmoid(self.fast_fc_conf(fast_hidden))

        # ===== MEDIUM LAYER (5-min scale) =====
        # Downsample features: 200 → 40 bars
        x_medium = self.downsample_features(x, self.downsample_fast_to_medium)

        # Expand fast hidden state to match medium sequence length
        seq_len_medium = x_medium.shape[1]
        fast_hidden_expanded = fast_hidden.unsqueeze(1).expand(-1, seq_len_medium, -1)

        # Concatenate downsampled features with fast hidden state
        medium_input = torch.cat([x_medium, fast_hidden_expanded], dim=-1)
        self.last_medium_input = medium_input.detach()

        # Process through medium layer
        medium_out, h_medium_new = self.medium_layer(medium_input, h_medium)

        # Take final hidden state
        medium_hidden = medium_out[:, -1, :]  # [batch, 128]

        # Medium layer predictions
        medium_pred_high = self.medium_fc_high(medium_hidden)
        medium_pred_low = self.medium_fc_low(medium_hidden)
        medium_pred_conf = torch.sigmoid(self.medium_fc_conf(medium_hidden))

        # ===== SLOW LAYER (1-hour scale) =====
        # Downsample medium features further: 40 → 3-4 bars
        x_slow = self.downsample_features(x_medium, self.downsample_medium_to_slow)

        # Expand medium hidden state
        seq_len_slow = x_slow.shape[1]
        medium_hidden_expanded = medium_hidden.unsqueeze(1).expand(-1, seq_len_slow, -1)

        # Concatenate downsampled features with medium hidden state
        slow_input = torch.cat([x_slow, medium_hidden_expanded], dim=-1)
        self.last_slow_input = slow_input.detach()

        # Process through slow layer
        slow_out, h_slow_new = self.slow_layer(slow_input, h_slow)

        # Take final hidden state
        slow_hidden = slow_out[:, -1, :]  # [batch, 128]

        # Slow layer predictions
        slow_pred_high = self.slow_fc_high(slow_hidden)
        slow_pred_low = self.slow_fc_low(slow_hidden)
        slow_pred_conf = torch.sigmoid(self.slow_fc_conf(slow_hidden))

        # ===== ADAPTIVE FUSION =====
        # Combine all layer predictions
        layer_predictions = torch.cat([
            fast_pred_high, fast_pred_low, fast_pred_conf,
            medium_pred_high, medium_pred_low, medium_pred_conf,
            slow_pred_high, slow_pred_low, slow_pred_conf
        ], dim=-1)  # [batch, 9]

        # Add market state and news if available
        if market_state is None:
            market_state = torch.zeros(batch_size, 12, device=x.device)
        if news_vec is None:
            news_vec = torch.zeros(batch_size, 768, device=x.device)
        if news_mask is None:
            news_mask = torch.zeros(batch_size, 1, device=x.device)

        self.last_market_state = market_state.detach()
        self.last_news_vec = news_vec.detach()

        # Fusion input
        fusion_input = torch.cat([
            layer_predictions,
            market_state,
            news_vec,
            news_mask
        ], dim=-1)  # [batch, 9 + 12 + 768 + 1 = 790]

        # Fusion network
        fusion_hidden = F.relu(self.fusion_fc1(fusion_input))
        fusion_hidden = F.relu(self.fusion_fc2(fusion_hidden))  # [batch, self.fusion_output_dim] (scales with hidden_size)

        # Primary predictions
        final_pred_high = self.fusion_fc_high(fusion_hidden)
        final_pred_low = self.fusion_fc_low(fusion_hidden)
        final_pred_conf = torch.sigmoid(self.fusion_fc_conf(fusion_hidden))

        # Combine into output
        predictions = torch.cat([
            final_pred_high,
            final_pred_low,
            final_pred_conf
        ], dim=-1)  # [batch, 3]

        # Store hidden states and layer predictions
        hidden_states = {
            'fast': h_fast_new,
            'medium': h_medium_new,
            'slow': h_slow_new,
            'fast_pred': torch.cat([fast_pred_high, fast_pred_low, fast_pred_conf], dim=-1),
            'medium_pred': torch.cat([medium_pred_high, medium_pred_low, medium_pred_conf], dim=-1),
            'slow_pred': torch.cat([slow_pred_high, slow_pred_low, slow_pred_conf], dim=-1),
            'fusion_weights': F.softmax(self.fusion_weights, dim=0)
        }

        # MULTI-TASK PREDICTIONS (Phase 3)
        if self.multi_task:
            # Compute multi-task outputs from fusion_hidden
            hit_band_pred = self.hit_band_head(fusion_hidden)  # [batch, 1]
            hit_target_pred = self.hit_target_head(fusion_hidden)  # [batch, 1]
            expected_return_pred = self.expected_return_head(fusion_hidden)  # [batch, 1]
            overshoot_pred = self.overshoot_head(fusion_hidden)  # [batch, 1]

            # Continuation predictions
            continuation_duration_pred = self.continuation_duration_head(fusion_hidden)  # [batch, 1]
            continuation_gain_pred = self.continuation_gain_head(fusion_hidden)  # [batch, 1]
            continuation_confidence_pred = self.continuation_confidence_head(fusion_hidden)  # [batch, 1]

            # Adaptive horizon predictions (for adaptive mode)
            adaptive_horizon_pred = self.adaptive_horizon_head(fusion_hidden)  # [batch, 1], 0-1 range
            adaptive_conf_score_pred = self.adaptive_conf_score_head(fusion_hidden)  # [batch, 1], 0-1 range

            # Adaptive Projection: Dynamic timescale selection and horizon prediction
            fusion_input = torch.cat([
                fast_hidden, medium_hidden, slow_hidden,
                fast_pred_conf, medium_pred_conf, slow_pred_conf
            ], dim=-1)

            proj = self.adaptive_projection(fusion_input)
            price_change_pct = proj[:, 0]
            horizon_bars_log = proj[:, 1]
            adaptive_confidence = torch.sigmoid(proj[:, 2])

            horizon_bars = torch.exp(horizon_bars_log) * 24  # Base = 24 bars, can grow
            horizon_bars = torch.clamp(horizon_bars, 24, 2016)  # Max ~2 weeks (1-min bars)

            # Determine dominant layer (keep as tensor to avoid torch.compile graph break)
            layer_weights = torch.stack([fast_pred_conf, medium_pred_conf, slow_pred_conf], dim=1)
            dominant_layer_idx = torch.argmax(layer_weights, dim=1)
            # Note: To decode indices to strings: {0: "fast", 1: "medium", 2: "slow"}

            # Store in hidden_states
            hidden_states['multi_task'] = {
                'hit_band': hit_band_pred,
                'hit_target': hit_target_pred,
                'expected_return': expected_return_pred,
                'overshoot': overshoot_pred,
                'continuation_duration': continuation_duration_pred,
                'continuation_gain': continuation_gain_pred,
                'continuation_confidence': continuation_confidence_pred,
                'adaptive_horizon': adaptive_horizon_pred,  # 0-1 range, will be scaled to 24-48
                'adaptive_conf_score': adaptive_conf_score_pred,  # 0-1 confidence score
                'price_change_pct': price_change_pct,
                'horizon_bars_log': horizon_bars_log,
                'horizon_bars': horizon_bars,
                'adaptive_confidence': adaptive_confidence,
                'dominant_layer_idx': dominant_layer_idx  # Tensor indices (0=fast, 1=medium, 2=slow)
            }

            # Breakout predictions (v3.21) - check if heads exist for backward compat
            if hasattr(self, 'breakout_prob_head'):
                breakout_prob = self.breakout_prob_head(fusion_hidden)  # [batch, 1], 0-1
                breakout_direction = self.breakout_direction_head(fusion_hidden)  # [batch, 1], 0-1 (up prob)
                breakout_bars_log = self.breakout_bars_head(fusion_hidden)  # [batch, 1], log scale
                breakout_confidence = self.breakout_confidence_head(fusion_hidden)  # [batch, 1], 0-1

                # Transform breakout_bars from log to actual bars (1-100 range)
                breakout_bars = torch.exp(breakout_bars_log).clamp(1, 100)

                hidden_states['breakout'] = {
                    'probability': breakout_prob,  # Prob of breakout in next N bars
                    'direction': breakout_direction,  # Prob it's an up-breakout (vs down)
                    'bars_until': breakout_bars,  # Expected bars until breakout
                    'confidence': breakout_confidence,  # Model certainty
                    'is_trained': False  # Will be True after training with breakout labels
                }

        return predictions, hidden_states

    def clear_cached_states(self):
        """
        Clear cached hidden states to prevent memory accumulation.

        Call this periodically during training (e.g., every 10-50 batches)
        to free memory from accumulated detached tensors.
        """
        self.last_fast_input = None
        self.last_medium_input = None
        self.last_slow_input = None
        self.last_market_state = None
        self.last_news_vec = None

    def predict(
        self,
        x: torch.Tensor,
        market_state: Optional[torch.Tensor] = None,
        news_vec: Optional[torch.Tensor] = None,
        news_mask: Optional[torch.Tensor] = None,
        h: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Generate predictions dict (ModelBase interface).

        Args:
            x: Input features [batch, 200, 299]
            market_state: Optional market state [batch, 12]
            news_vec: Optional news embeddings [batch, 768]
            news_mask: Optional news mask [batch, 1]
            h: Optional hidden states dict

        Returns:
            predictions_dict: Dict with keys:
                - predicted_high: float (percentage)
                - predicted_low: float (percentage)
                - confidence: float (0-1)
                - fast_pred_high/low/conf: float
                - medium_pred_high/low/conf: float
                - slow_pred_high/low/conf: float
                - hidden_states: Dict for stateful inference
        """
        self.eval()

        with torch.no_grad():
            # Extract hidden states if provided
            h_fast = h.get('fast') if h else None
            h_medium = h.get('medium') if h else None
            h_slow = h.get('slow') if h else None

            # Forward pass
            predictions, hidden_states = self.forward(
                x, market_state, news_vec, news_mask,
                h_fast, h_medium, h_slow
            )

            # Extract predictions
            pred_high = predictions[0, 0].item()
            pred_low = predictions[0, 1].item()
            pred_conf = predictions[0, 2].item()

            # Extract layer predictions
            fast_preds = hidden_states['fast_pred'][0]
            medium_preds = hidden_states['medium_pred'][0]
            slow_preds = hidden_states['slow_pred'][0]

            result = {
                'predicted_high': pred_high,
                'predicted_low': pred_low,
                'confidence': pred_conf,
                'fast_pred_high': fast_preds[0].item(),
                'fast_pred_low': fast_preds[1].item(),
                'fast_pred_conf': fast_preds[2].item(),
                'medium_pred_high': medium_preds[0].item(),
                'medium_pred_low': medium_preds[1].item(),
                'medium_pred_conf': medium_preds[2].item(),
                'slow_pred_high': slow_preds[0].item(),
                'slow_pred_low': slow_preds[1].item(),
                'slow_pred_conf': slow_preds[2].item(),
                'hidden_states': hidden_states,
                'fusion_weights': hidden_states['fusion_weights'].cpu().numpy().tolist()
            }

            # Add multi-task predictions if enabled
            if self.multi_task and 'multi_task' in hidden_states:
                mt = hidden_states['multi_task']
                result['hit_band_pred'] = mt['hit_band'][0, 0].item()
                result['hit_target_pred'] = mt['hit_target'][0, 0].item()
                result['expected_return_pred'] = mt['expected_return'][0, 0].item()
                result['overshoot_pred'] = mt['overshoot'][0, 0].item()

            # Add breakout predictions if available
            if 'breakout' in hidden_states:
                bo = hidden_states['breakout']
                result['breakout'] = {
                    'probability': bo['probability'][0, 0].item(),  # 0-1
                    'direction': bo['direction'][0, 0].item(),  # 0-1 (up prob)
                    'direction_label': 'up' if bo['direction'][0, 0].item() > 0.5 else 'down',
                    'bars_until': bo['bars_until'][0, 0].item(),  # 1-100
                    'confidence': bo['confidence'][0, 0].item(),  # 0-1
                    'is_trained': bo['is_trained']  # False until trained
                }

            return result

    def project_channel(
        self,
        x: torch.Tensor,
        current_price: float,
        horizons: List[int] = [15, 30, 60, 120, 240],
        min_confidence: float = 0.65,
        market_state: Optional[torch.Tensor] = None,
        news_vec: Optional[torch.Tensor] = None
    ) -> List[Dict[str, Any]]:
        """
        Project channel forward with confidence decay.

        Makes predictions at multiple time horizons (e.g., 15min, 30min, 1hour, 2hour, 4hour)
        and returns only those with sufficient confidence.

        Args:
            x: Current features [1, 200, 299]
            current_price: Current TSLA price for converting % to absolute
            horizons: List of minutes ahead to project
            min_confidence: Minimum confidence threshold
            market_state: Optional market state [1, 12]
            news_vec: Optional news embeddings [1, 768]

        Returns:
            projections: List of dicts with:
                - horizon_minutes: int
                - predicted_high: float (percentage)
                - predicted_low: float (percentage)
                - predicted_high_price: float (absolute price)
                - predicted_low_price: float (absolute price)
                - confidence: float (with decay)
                - validation_time: str (for online learning)
        """
        from datetime import datetime, timedelta

        self.eval()
        projections = []

        with torch.no_grad():
            for horizon in horizons:
                # Make prediction
                pred_dict = self.predict(x, market_state, news_vec)

                # Confidence decay factor: exp(-horizon / 60)
                # Decays to ~37% after 1 hour, ~14% after 2 hours
                decay_factor = np.exp(-horizon / 60.0)
                adjusted_confidence = pred_dict['confidence'] * decay_factor

                if adjusted_confidence >= min_confidence:
                    # Calculate absolute prices
                    pred_high_pct = pred_dict['predicted_high']
                    pred_low_pct = pred_dict['predicted_low']

                    pred_high_price = current_price * (1 + pred_high_pct / 100.0)
                    pred_low_price = current_price * (1 + pred_low_pct / 100.0)

                    # Validation time (when to check this prediction)
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
                        'fast_layer_weight': pred_dict['fusion_weights'][0],
                        'medium_layer_weight': pred_dict['fusion_weights'][1],
                        'slow_layer_weight': pred_dict['fusion_weights'][2]
                    }

                    # Add breakout predictions if available
                    if 'breakout' in pred_dict:
                        projection['breakout'] = pred_dict['breakout']

                    projections.append(projection)
                else:
                    # Confidence too low - stop projecting further
                    break

        return projections

    def update_online(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lr: float = 0.0001,
        layer: str = 'all'
    ):
        """
        Online learning update from prediction errors.

        Args:
            x: Input features [batch, 200, 299]
            y: True targets [batch, 2] - [actual_high, actual_low]
            lr: Learning rate for online update
            layer: Which layer to update ('fast', 'medium', 'slow', 'all')
        """
        self.train()

        # Forward pass
        predictions, _ = self.forward(x)

        # Calculate loss
        pred_high = predictions[:, 0]
        pred_low = predictions[:, 1]
        target_high = y[:, 0]
        target_low = y[:, 1]

        loss = F.mse_loss(pred_high, target_high) + F.mse_loss(pred_low, target_low)

        # Backward pass with small learning rate
        loss.backward()

        # Update specified layers only
        with torch.no_grad():
            if layer in ['fast', 'all']:
                for param in self.fast_layer.parameters():
                    if param.grad is not None:
                        param -= lr * param.grad
                for param in [self.fast_fc_high, self.fast_fc_low, self.fast_fc_conf]:
                    for p in param.parameters():
                        if p.grad is not None:
                            p -= lr * p.grad

            if layer in ['medium', 'all']:
                for param in self.medium_layer.parameters():
                    if param.grad is not None:
                        param -= lr * param.grad
                for param in [self.medium_fc_high, self.medium_fc_low, self.medium_fc_conf]:
                    for p in param.parameters():
                        if p.grad is not None:
                            p -= lr * p.grad

            if layer in ['slow', 'all']:
                for param in self.slow_layer.parameters():
                    if param.grad is not None:
                        param -= lr * param.grad
                for param in [self.slow_fc_high, self.slow_fc_low, self.slow_fc_conf]:
                    for p in param.parameters():
                        if p.grad is not None:
                            p -= lr * p.grad

            # Always update fusion head
            for param in [self.fusion_fc1, self.fusion_fc2, self.fusion_fc_high,
                          self.fusion_fc_low, self.fusion_fc_conf]:
                for p in param.parameters():
                    if p.grad is not None:
                        p -= lr * p.grad

        # Zero gradients
        self.zero_grad()

    def save_checkpoint(self, path: str, metadata: Dict = None):
        """
        Save model checkpoint with metadata.

        Args:
            path: Save path
            metadata: Additional metadata dict
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_type': 'HierarchicalLNN',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'internal_neurons_ratio': self.internal_neurons_ratio,
            'total_neurons': self.total_neurons,
            'device_type': self.device_type,
            'downsample_fast_to_medium': self.downsample_fast_to_medium,
            'downsample_medium_to_slow': self.downsample_medium_to_slow,
            'multi_task': self.multi_task,
        }

        if metadata:
            checkpoint.update(metadata)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> Dict:
        """
        Load model checkpoint and return metadata.

        Args:
            path: Checkpoint path

        Returns:
            metadata: Dict containing model metadata
        """
        checkpoint = torch.load(path, map_location=self.device_type)

        # Handle DataParallel checkpoints: strip 'module.' prefix if present
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

        self.load_state_dict(state_dict)

        # Extract metadata
        metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}

        return metadata


def load_hierarchical_model(model_path: str, device: str = 'cpu') -> HierarchicalLNN:
    """
    Load a trained hierarchical model from checkpoint.

    Args:
        model_path: Path to .pth checkpoint
        device: 'cuda' or 'cpu'

    Returns:
        model: Loaded HierarchicalLNN instance
    """
    # Load checkpoint to get metadata
    checkpoint = torch.load(model_path, map_location=device)

    # Get parameters from checkpoint (check both top-level and args dict)
    args = checkpoint.get('args', {})

    # For input_size: infer from weights if not explicitly saved
    input_size = checkpoint.get('input_size') or args.get('input_size')
    if input_size is None:
        # Infer from first layer weight shape (accounting for wiring transform)
        state_dict = checkpoint['model_state_dict']
        first_weight = state_dict.get('fast_layer.rnn_cell.layer_0.ff1.weight')
        if first_weight is not None:
            # CfC adds internal connections, so actual input = weight.shape[1] - some offset
            # For AutoNCP wiring, the transform is: input -> input + hidden
            # So we approximate: input_size ≈ weight.shape[1] - hidden_size
            hidden_size = args.get('hidden_size', 128)
            input_size = first_weight.shape[1] - hidden_size + 51  # Empirical adjustment
            print(f"  ⚠️ Inferred input_size from weights: {input_size}")
        else:
            input_size = 14487  # Current feature count
            print(f"  ⚠️ Using current feature count: {input_size}")

    # Get other parameters
    hidden_size = checkpoint.get('hidden_size') or args.get('hidden_size', 128)
    internal_neurons_ratio = checkpoint.get('internal_neurons_ratio') or args.get('internal_neurons_ratio', 2.0)

    # Create model with saved/inferred parameters
    model = HierarchicalLNN(
        input_size=input_size,
        hidden_size=hidden_size,
        internal_neurons_ratio=internal_neurons_ratio,
        device=device,
        downsample_fast_to_medium=checkpoint.get('downsample_fast_to_medium') or args.get('downsample_fast_to_medium', 5),
        downsample_medium_to_slow=checkpoint.get('downsample_medium_to_slow') or args.get('downsample_medium_to_slow', 12),
        multi_task=checkpoint.get('multi_task') or args.get('multi_task', True)
    )

    # Handle DataParallel checkpoints: strip 'module.' prefix if present
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    # Load state dict (strict=False for backward compat with older checkpoints
    # that don't have breakout heads)
    incompatible = model.load_state_dict(state_dict, strict=False)

    # Log any missing keys (expected for new heads like breakout)
    if incompatible.missing_keys:
        # Only warn if keys are NOT breakout-related (expected to be missing)
        non_breakout_missing = [k for k in incompatible.missing_keys if 'breakout' not in k]
        if non_breakout_missing:
            print(f"  ⚠️ Unexpected missing keys: {non_breakout_missing}")
        else:
            print(f"  ℹ️ Breakout heads initialized randomly (not in checkpoint)")

    if incompatible.unexpected_keys:
        print(f"  ⚠️ Unexpected keys in checkpoint: {incompatible.unexpected_keys}")

    return model
