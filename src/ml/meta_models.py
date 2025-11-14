"""
Meta-models for combining multi-scale LNN predictions.

This module provides meta-learner models that take predictions from
multiple timeframe-specific LNN sub-models and combine them adaptively
based on market conditions and optionally news sentiment.

Architecture:
- MetaLNN: Liquid Neural Network meta-learner (default)
- MetaFTTransformer: Feature Tokenizer + Transformer (alternative)
- calculate_market_state(): Extract market regime features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from ncps.wirings import AutoNCP
from ncps.torch import CfC


class MetaLNN(nn.Module):
    """
    Meta-model using Liquid Neural Network to combine sub-model predictions.

    Learns adaptive weighting based on:
    - Sub-model predictions (high, low, confidence)
    - Market state features (volatility, jumps, correlations)
    - News embeddings (optional, LFM2-based)

    Input shape: [batch, num_submodels * 3 + market_state_dim + news_vec_dim + 1]
    Output: [predicted_high, predicted_low, confidence]
    """

    def __init__(self,
                 num_submodels=4,
                 market_state_dim=12,
                 news_vec_dim=768,
                 hidden_size=64):
        """
        Initialize Meta-LNN.

        Args:
            num_submodels: Number of sub-models (default: 4 for 15min/1h/4h/daily)
            market_state_dim: Dimension of market state features (default: 12)
            news_vec_dim: Dimension of news embeddings (default: 768 for LFM2)
            hidden_size: Hidden units in LNN (default: 64, small for fast inference)
        """
        super().__init__()

        self.num_submodels = num_submodels
        self.market_state_dim = market_state_dim
        self.news_vec_dim = news_vec_dim

        # Input dimensions
        subpreds_dim = num_submodels * 3  # Each model: [high, low, conf]
        total_input = subpreds_dim + market_state_dim + news_vec_dim + 1  # +1 for news_mask

        # Liquid Neural Network core
        self.wiring = AutoNCP(hidden_size, 3)  # 3 intermediate outputs
        self.lnn = CfC(total_input, self.wiring)

        # Output heads
        self.fc_high = nn.Linear(3, 1)
        self.fc_low = nn.Linear(3, 1)
        self.fc_conf = nn.Linear(3, 1)

    def forward(self, subpreds, market_state, news_vec, news_mask, h=None):
        """
        Forward pass.

        Args:
            subpreds: [batch, num_submodels, 3] - Sub-model predictions
            market_state: [batch, market_state_dim] - Market regime features
            news_vec: [batch, news_vec_dim] - News embeddings (or zeros)
            news_mask: [batch, 1] - 1 if news available, 0 otherwise
            h: Hidden state (optional)

        Returns:
            pred_high: [batch, 1]
            pred_low: [batch, 1]
            pred_conf: [batch, 1]
            hidden_state: For stateful inference
        """
        batch_size = subpreds.size(0)

        # Flatten sub-predictions: [batch, num_submodels, 3] → [batch, num_submodels * 3]
        subpreds_flat = subpreds.view(batch_size, -1)

        # Concatenate all inputs
        x = torch.cat([subpreds_flat, market_state, news_vec, news_mask], dim=1)

        # LNN processing (add time dimension: [batch, 1, features])
        x_seq = x.unsqueeze(1)
        out, h_new = self.lnn(x_seq, h)

        # Remove time dimension: [batch, 1, 3] → [batch, 3]
        out = out.squeeze(1)

        # Regression heads
        pred_high = self.fc_high(out)
        pred_low = self.fc_low(out)
        pred_conf = torch.sigmoid(self.fc_conf(out))

        return pred_high, pred_low, pred_conf, h_new


class MetaLNNWithModalityDropout(nn.Module):
    """
    Meta-LNN with modality dropout for robust training.

    During training, randomly zeros out news_vec with probability p
    to force model to handle both with-news and no-news scenarios.
    """

    def __init__(self,
                 num_submodels=4,
                 market_state_dim=12,
                 news_vec_dim=768,
                 hidden_size=64,
                 dropout_prob=0.4):
        """
        Initialize Meta-LNN with modality dropout.

        Args:
            dropout_prob: Probability of dropping news modality (default: 0.4)
        """
        super().__init__()

        self.meta_lnn = MetaLNN(num_submodels, market_state_dim, news_vec_dim, hidden_size)
        self.dropout_prob = dropout_prob

    def forward(self, subpreds, market_state, news_vec, news_mask, h=None):
        """
        Forward pass with modality dropout.

        During training, randomly zero out news with probability dropout_prob.
        """
        if self.training and torch.rand(1).item() < self.dropout_prob:
            # Drop news modality
            news_vec = torch.zeros_like(news_vec)
            news_mask = torch.zeros_like(news_mask)

        return self.meta_lnn(subpreds, market_state, news_vec, news_mask, h)


class MetaFTTransformer(nn.Module):
    """
    Alternative meta-model using Feature Tokenizer + Transformer.

    Better for pure tabular data without temporal dependencies.
    Uses feature tokenization to treat each input feature as a token.
    """

    def __init__(self,
                 num_submodels=4,
                 market_state_dim=12,
                 news_vec_dim=768,
                 d_model=128,
                 nhead=4,
                 num_layers=2):
        """
        Initialize FT-Transformer meta-model.

        Args:
            d_model: Transformer hidden dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
        """
        super().__init__()

        self.num_submodels = num_submodels
        self.market_state_dim = market_state_dim
        self.news_vec_dim = news_vec_dim

        # Total number of features
        subpreds_dim = num_submodels * 3
        total_features = subpreds_dim + market_state_dim + 1  # +1 for news_mask
        # Note: news_vec is treated separately

        # Feature tokenization (linear projection per feature)
        self.feature_tokenizers = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(total_features)
        ])

        # News embedding projection
        self.news_proj = nn.Linear(news_vec_dim, d_model)

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads
        self.fc_high = nn.Linear(d_model, 1)
        self.fc_low = nn.Linear(d_model, 1)
        self.fc_conf = nn.Linear(d_model, 1)

    def forward(self, subpreds, market_state, news_vec, news_mask, h=None):
        """
        Forward pass.

        Returns:
            pred_high, pred_low, pred_conf, None (no hidden state for transformer)
        """
        batch_size = subpreds.size(0)

        # Flatten sub-predictions
        subpreds_flat = subpreds.view(batch_size, -1)

        # Concatenate numeric features
        numeric_features = torch.cat([subpreds_flat, market_state, news_mask], dim=1)

        # Tokenize each feature: [batch, num_features] → [batch, num_features, d_model]
        tokens = []
        for i, tokenizer in enumerate(self.feature_tokenizers):
            feat = numeric_features[:, i:i+1]  # [batch, 1]
            token = tokenizer(feat)  # [batch, d_model]
            tokens.append(token)

        tokens = torch.stack(tokens, dim=1)  # [batch, num_features, d_model]

        # Project news embedding
        news_token = self.news_proj(news_vec).unsqueeze(1)  # [batch, 1, d_model]

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, d_model]

        # Concatenate all tokens
        all_tokens = torch.cat([cls_tokens, tokens, news_token], dim=1)

        # Transformer encoding
        encoded = self.transformer(all_tokens)  # [batch, seq_len, d_model]

        # Use CLS token for prediction
        cls_output = encoded[:, 0, :]  # [batch, d_model]

        # Output heads
        pred_high = self.fc_high(cls_output)
        pred_low = self.fc_low(cls_output)
        pred_conf = torch.sigmoid(self.fc_conf(cls_output))

        return pred_high, pred_low, pred_conf, None


def calculate_market_state(df: pd.DataFrame, current_idx: int, events_handler=None) -> torch.Tensor:
    """
    Calculate market regime features at prediction time.

    Features (12 total):
    1-3. Realized volatility: 5min, 30min, 1day
    4. Overnight return absolute value
    5. Intraday jump flag (|return| > 3σ)
    6. Volatility z-score
    7-8. Time of day (sin/cos encoding)
    9. Has earnings soon (within 7 days)
    10. Has macro event soon (FOMC/CPI/NFP within 3 days)
    11. SPY correlation regime
    12. VIX level (if available, else 0)

    Args:
        df: DataFrame with price data (must have 'returns', 'volatility', etc.)
        current_idx: Current bar index
        events_handler: EventsHandler instance (optional, for event features)

    Returns:
        Tensor of shape [12] with market state features
    """
    features = []

    # 1-3. Realized volatility (multiple horizons)
    # Use rolling std of returns as proxy for realized vol
    if 'returns' in df.columns:
        # 5-minute realized vol (5 bars for 1-min data)
        rv_5m = df['returns'].iloc[max(0, current_idx-5):current_idx].std()
        rv_5m = rv_5m if not np.isnan(rv_5m) else 0.0

        # 30-minute realized vol
        rv_30m = df['returns'].iloc[max(0, current_idx-30):current_idx].std()
        rv_30m = rv_30m if not np.isnan(rv_30m) else 0.0

        # 1-day realized vol (390 bars for 1-min data, 6.5 hours)
        rv_1d = df['returns'].iloc[max(0, current_idx-390):current_idx].std()
        rv_1d = rv_1d if not np.isnan(rv_1d) else 0.0
    else:
        rv_5m, rv_30m, rv_1d = 0.0, 0.0, 0.0

    features.extend([rv_5m, rv_30m, rv_1d])

    # 4-5. Jump detection
    if 'returns' in df.columns and current_idx > 20:
        current_ret = df['returns'].iloc[current_idx]
        sigma = df['returns'].iloc[max(0, current_idx-20):current_idx].std()
        sigma = sigma if not np.isnan(sigma) and sigma > 0 else 0.01

        jump_flag = 1.0 if abs(current_ret) > 3 * sigma else 0.0
        overnight_ret_abs = abs(df.get('overnight_return', df['returns']).iloc[current_idx])
    else:
        jump_flag = 0.0
        overnight_ret_abs = 0.0

    features.extend([overnight_ret_abs, jump_flag])

    # 6. Volatility regime (z-score)
    if 'volatility' in df.columns and current_idx > 100:
        vol_mean = df['volatility'].iloc[max(0, current_idx-100):current_idx].mean()
        vol_std = df['volatility'].iloc[max(0, current_idx-100):current_idx].std()
        current_vol = df['volatility'].iloc[current_idx]

        if vol_std > 0 and not np.isnan(vol_std):
            vol_zscore = (current_vol - vol_mean) / vol_std
        else:
            vol_zscore = 0.0
    else:
        vol_zscore = 0.0

    features.append(vol_zscore)

    # 7-8. Time of day (cyclical encoding)
    timestamp = df.index[current_idx]
    hour = timestamp.hour + timestamp.minute / 60.0
    time_sin = np.sin(2 * np.pi * hour / 24)
    time_cos = np.cos(2 * np.pi * hour / 24)

    features.extend([time_sin, time_cos])

    # 9-10. Event proximity
    if events_handler is not None:
        try:
            events = events_handler.get_events_for_date(
                str(timestamp.date()),
                lookback_days=7
            )
            has_earnings_soon = 1.0 if any(e.get('type') == 'earnings' for e in events) else 0.0

            # Check for macro events in next 3 days
            events_3d = events_handler.get_events_for_date(
                str(timestamp.date()),
                lookback_days=0,
                lookahead_days=3
            )
            has_macro_soon = 1.0 if any(e.get('type') == 'macro' for e in events_3d) else 0.0
        except:
            has_earnings_soon = 0.0
            has_macro_soon = 0.0
    else:
        has_earnings_soon = 0.0
        has_macro_soon = 0.0

    features.extend([has_earnings_soon, has_macro_soon])

    # 11. SPY correlation regime
    if 'correlation_50' in df.columns:
        spy_corr = df['correlation_50'].iloc[current_idx]
        spy_corr = spy_corr if not np.isnan(spy_corr) else 0.0
    else:
        spy_corr = 0.0

    features.append(spy_corr)

    # 12. VIX level (if available)
    # For now, placeholder (could fetch from separate data source)
    vix_level = 0.0
    features.append(vix_level)

    return torch.tensor(features, dtype=torch.float32)


def meta_loss(pred_high, pred_low, pred_conf, y_high, y_low, confidence_target=None):
    """
    Loss function for meta-model training.

    Uses:
    - Huber loss for price predictions (robust to outliers)
    - MSE or BCE for confidence calibration

    Args:
        pred_high: Predicted highs [batch]
        pred_low: Predicted lows [batch]
        pred_conf: Predicted confidence [batch]
        y_high: Actual highs [batch]
        y_low: Actual lows [batch]
        confidence_target: Optional target confidence (default: derived from accuracy)

    Returns:
        Total loss (scalar)
    """
    # Huber loss for percentages
    # delta=0.5 means errors >0.5 percentage points are treated as linear (robust to outliers)
    # Typical prediction range: -10% to +10%, so delta=0.5pp is reasonable
    loss_high = F.huber_loss(pred_high, y_high, delta=0.5)
    loss_low = F.huber_loss(pred_low, y_low, delta=0.5)

    # Confidence calibration
    if confidence_target is None:
        # Derive confidence target from prediction accuracy
        # Predictions and targets are now in percentage terms, so normalize differently
        # Assuming typical range of ±10%, we divide error by a reasonable scale
        high_accuracy = torch.exp(-torch.abs(pred_high - y_high) / 2.0)  # Exponential decay
        low_accuracy = torch.exp(-torch.abs(pred_low - y_low) / 2.0)
        confidence_target = (high_accuracy + low_accuracy) / 2.0
        confidence_target = torch.clamp(confidence_target, 0.0, 1.0)

    loss_conf = F.mse_loss(pred_conf, confidence_target)

    # Total loss (weight confidence less than prices)
    total_loss = loss_high + loss_low + 0.1 * loss_conf

    return total_loss


# Export main classes
__all__ = [
    'MetaLNN',
    'MetaLNNWithModalityDropout',
    'MetaFTTransformer',
    'calculate_market_state',
    'meta_loss'
]
