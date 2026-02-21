#!/usr/bin/env python3
"""
Channel Surfer ML — Machine Learning models for channel prediction.

Three novel architectures:
1. Gradient Boosted Trees (baseline) — fast, interpretable
2. Survival Analysis (DeepSurv) — predicts channel lifetime as hazard function
3. Multi-TF Transformer — cross-attention between timeframes

Features extracted from TFChannelState physics + cross-TF relationships.

Labels:
- channel_lifetime: bars until channel breaks (regression / survival)
- break_direction: up / down / survive (3-class classification)
- optimal_action: buy / sell / hold (3-class classification)
- future_return_5bar: forward 5-bar return (regression)
- future_return_20bar: forward 20-bar return (regression)

Usage:
    python3 -m v15.core.surfer_ml train --days 120 --arch gbt
    python3 -m v15.core.surfer_ml train --days 120 --arch survival
    python3 -m v15.core.surfer_ml train --days 120 --arch transformer
    python3 -m v15.core.surfer_ml evaluate --checkpoint surfer_models/best_gbt.pkl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature Names (ordered, deterministic)
# ---------------------------------------------------------------------------

# Per-TF physics features (extracted from TFChannelState)
PER_TF_FEATURES = [
    'position_pct',
    'center_distance',
    'potential_energy',
    'kinetic_energy',
    'momentum_direction',
    'total_energy',
    'binding_energy',
    'entropy',
    'oscillation_period',
    'bars_to_next_bounce',
    'channel_health',
    'slope_pct',
    'width_pct',
    'r_squared',
    'bounce_count',
    'ou_theta',
    'ou_half_life',
    'ou_reversion_score',
    'break_prob',
    'break_prob_up',
    'break_prob_down',
    'volume_score',
    'momentum_turn_score',
    'squeeze_score',
]

# Timeframes we extract features for
ML_TFS = ['5min', '1h', '4h', 'daily', 'weekly']

# Cross-TF derived features
CROSS_TF_FEATURES = [
    'pos_spread_5m_1h',        # Position divergence: 5min vs 1h
    'pos_spread_5m_daily',     # Position divergence: 5min vs daily
    'pos_spread_1h_daily',     # Position divergence: 1h vs daily
    'energy_ratio_5m_1h',     # Energy ratio: 5min / 1h
    'energy_ratio_5m_daily',  # Energy ratio: 5min / daily
    'health_min',             # Worst channel health across TFs
    'health_max',             # Best channel health across TFs
    'health_spread',          # Max - min health (divergence)
    'break_prob_max',         # Highest break probability
    'break_prob_weighted',    # TF-weighted break probability
    'direction_consensus',    # Fraction of TFs agreeing on direction
    'bullish_fraction',       # Fraction of TFs in bull channels
    'bearish_fraction',       # Fraction of TFs in bear channels
    'theta_spread',           # Max - min OU theta (reversion disagreement)
    'avg_entropy',            # Mean entropy across TFs
    'confluence_score',       # Multi-TF position alignment
    'squeeze_any',            # Any TF showing squeeze (binary)
    'valid_tf_count',         # How many TFs have valid channels
]

# Market context features (RSI, volume ratios, price-level features)
CONTEXT_FEATURES = [
    'rsi_14',                 # 14-period RSI
    'rsi_5',                  # 5-period RSI (short-term)
    'volume_ratio_20',        # Current volume / 20-bar avg volume
    'volume_trend_5',         # 5-bar volume slope (rising/falling)
    'atr_pct',                # ATR(14) as % of price
    'price_vs_vwap',          # Price relative to VWAP
    'bar_range_pct',          # Current bar range as % of ATR
    'close_position_in_bar',  # Where close sits in H-L range (0-1)
    'consecutive_up_bars',    # Number of consecutive up-closes
    'consecutive_down_bars',  # Number of consecutive down-closes
    'hour_sin',               # Hour of day (sin encoding)
    'hour_cos',               # Hour of day (cos encoding)
    'day_of_week',            # Day of week (0-4)
    'minutes_since_open',     # Minutes since market open
]

# Temporal/trajectory features (rate of change of key metrics)
TEMPORAL_FEATURES = [
    'pos_delta_3bar',         # Position change over 3 eval intervals
    'pos_delta_6bar',         # Position change over 6 eval intervals
    'health_delta_3bar',      # Health change over 3 eval intervals
    'health_delta_6bar',      # Health change over 6 eval intervals
    'entropy_delta_3bar',     # Entropy change (rising = channel failing)
    'break_prob_delta_3bar',  # Break prob change (rising = imminent break)
    'energy_delta_3bar',      # Total energy trajectory
    'rsi_slope_5bar',         # RSI trajectory (rising/falling)
    'price_momentum_3bar',    # Normalized price change over 3 intervals
    'price_momentum_12bar',   # Normalized price change over 12 intervals
    'vol_momentum_3bar',      # Volume trend over 3 intervals
    'width_delta_3bar',       # Channel width change (narrowing = squeeze)
]

# SPY/VIX correlation features
CORRELATION_FEATURES = [
    'spy_return_5bar',        # SPY 5-bar return
    'spy_return_20bar',       # SPY 20-bar return
    'spy_tsla_corr_20',      # Rolling 20-bar correlation SPY vs TSLA
    'vix_level',              # VIX level (from ^VIX daily)
    'vix_change_5d',          # VIX 5-day change
]


def get_feature_names() -> List[str]:
    """Return ordered list of all feature names."""
    names = []
    # Per-TF features
    for tf in ML_TFS:
        for feat in PER_TF_FEATURES:
            names.append(f'{tf}_{feat}')
    # Cross-TF
    names.extend(CROSS_TF_FEATURES)
    # Context
    names.extend(CONTEXT_FEATURES)
    # Temporal
    names.extend(TEMPORAL_FEATURES)
    # Correlations
    names.extend(CORRELATION_FEATURES)
    return names


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def extract_tf_features(state) -> np.ndarray:
    """Extract features from a TFChannelState into a fixed-size array."""
    if not state.valid:
        return np.zeros(len(PER_TF_FEATURES))

    features = []
    for feat_name in PER_TF_FEATURES:
        val = getattr(state, feat_name, 0.0)
        if isinstance(val, bool):
            val = float(val)
        elif isinstance(val, str):
            # channel_direction is handled separately
            val = 0.0
        features.append(float(val))
    return np.array(features, dtype=np.float32)


def extract_cross_tf_features(
    tf_states: Dict[str, 'TFChannelState'],
) -> np.ndarray:
    """Extract cross-TF relationship features."""
    features = np.zeros(len(CROSS_TF_FEATURES), dtype=np.float32)

    def get_state(tf):
        s = tf_states.get(tf)
        return s if s and s.valid else None

    s5 = get_state('5min')
    s1h = get_state('1h')
    s4h = get_state('4h')
    sd = get_state('daily')
    sw = get_state('weekly')

    # Position spreads (divergence between TFs)
    if s5 and s1h:
        features[0] = s5.position_pct - s1h.position_pct
    if s5 and sd:
        features[1] = s5.position_pct - sd.position_pct
    if s1h and sd:
        features[2] = s1h.position_pct - sd.position_pct

    # Energy ratios
    if s5 and s1h and s1h.total_energy > 0.01:
        features[3] = s5.total_energy / max(s1h.total_energy, 0.01)
    if s5 and sd and sd.total_energy > 0.01:
        features[4] = s5.total_energy / max(sd.total_energy, 0.01)

    # Health stats
    valid_states = [s for s in [s5, s1h, s4h, sd, sw] if s is not None]
    if valid_states:
        healths = [s.channel_health for s in valid_states]
        features[5] = min(healths)
        features[6] = max(healths)
        features[7] = max(healths) - min(healths)

        # Break probability
        break_probs = [s.break_prob for s in valid_states]
        features[8] = max(break_probs)

        # TF-weighted break probability
        tf_weights = {'5min': 0.10, '1h': 0.20, '4h': 0.25, 'daily': 0.30, 'weekly': 0.15}
        weighted_bp = sum(s.break_prob * tf_weights.get(s.tf, 0.1) for s in valid_states)
        total_w = sum(tf_weights.get(s.tf, 0.1) for s in valid_states)
        features[9] = weighted_bp / max(total_w, 0.01)

        # Direction consensus
        dirs = [s.channel_direction for s in valid_states]
        if dirs:
            from collections import Counter
            most_common = Counter(dirs).most_common(1)[0][1]
            features[10] = most_common / len(dirs)
            features[11] = sum(1 for d in dirs if d == 'bull') / len(dirs)
            features[12] = sum(1 for d in dirs if d == 'bear') / len(dirs)

        # Theta spread
        thetas = [s.ou_theta for s in valid_states if s.ou_theta > 0]
        if thetas:
            features[13] = max(thetas) - min(thetas)

        # Average entropy
        features[14] = np.mean([s.entropy for s in valid_states])

        # Confluence (position alignment)
        positions = [s.position_pct for s in valid_states]
        features[15] = 1.0 - np.std(positions)  # High std = low alignment

        # Squeeze detection
        features[16] = float(any(s.squeeze_score > 0.5 for s in valid_states))

        # Valid TF count
        features[17] = len(valid_states)

    return features


def compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Compute RSI for the last bar."""
    if len(prices) < period + 1:
        return 50.0

    deltas = np.diff(prices[-(period + 1):])
    gains = np.maximum(deltas, 0)
    losses = np.abs(np.minimum(deltas, 0))

    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)

    if avg_loss < 1e-10:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def extract_context_features(
    df: pd.DataFrame,
    bar_idx: int,
) -> np.ndarray:
    """Extract market context features at a specific bar."""
    features = np.zeros(len(CONTEXT_FEATURES), dtype=np.float32)

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    current_close = closes[bar_idx]

    # RSI(14)
    if bar_idx >= 15:
        features[0] = compute_rsi(closes[:bar_idx + 1], 14)
    else:
        features[0] = 50.0

    # RSI(5)
    if bar_idx >= 6:
        features[1] = compute_rsi(closes[:bar_idx + 1], 5)
    else:
        features[1] = 50.0

    # Volume ratio
    if 'volume' in df.columns and bar_idx >= 20:
        vols = df['volume'].values
        current_vol = vols[bar_idx]
        avg_vol = np.mean(vols[max(0, bar_idx - 20):bar_idx])
        features[2] = current_vol / max(avg_vol, 1) if avg_vol > 0 else 1.0

        # Volume trend (5-bar slope)
        if bar_idx >= 5:
            recent_vols = vols[bar_idx - 5:bar_idx + 1]
            if np.std(recent_vols) > 0:
                x = np.arange(len(recent_vols))
                features[3] = np.corrcoef(x, recent_vols)[0, 1]

    # ATR as % of price
    if bar_idx >= 15:
        tr = np.maximum(
            highs[bar_idx - 14:bar_idx + 1] - lows[bar_idx - 14:bar_idx + 1],
            np.maximum(
                np.abs(highs[bar_idx - 14:bar_idx + 1] - np.concatenate([[closes[bar_idx - 15]], closes[bar_idx - 14:bar_idx]])),
                np.abs(lows[bar_idx - 14:bar_idx + 1] - np.concatenate([[closes[bar_idx - 15]], closes[bar_idx - 14:bar_idx]]))
            )
        )
        atr = np.mean(tr)
        features[4] = atr / current_close if current_close > 0 else 0

    # Price vs VWAP (simplified: price vs volume-weighted avg of recent)
    if 'volume' in df.columns and bar_idx >= 20:
        recent_close = closes[bar_idx - 20:bar_idx + 1]
        recent_vol = df['volume'].values[bar_idx - 20:bar_idx + 1]
        total_vol = np.sum(recent_vol)
        if total_vol > 0:
            vwap = np.sum(recent_close * recent_vol) / total_vol
            features[5] = (current_close - vwap) / vwap if vwap > 0 else 0

    # Bar range as % of ATR
    bar_range = highs[bar_idx] - lows[bar_idx]
    if features[4] > 0 and current_close > 0:
        features[6] = (bar_range / current_close) / features[4] if features[4] > 0 else 1.0

    # Close position in bar (where close sits in H-L range)
    if bar_range > 0:
        features[7] = (current_close - lows[bar_idx]) / bar_range
    else:
        features[7] = 0.5

    # Consecutive up/down bars
    if bar_idx >= 1:
        up_count = 0
        down_count = 0
        for i in range(bar_idx, max(bar_idx - 10, 0), -1):
            if closes[i] > closes[i - 1]:
                if down_count > 0:
                    break
                up_count += 1
            elif closes[i] < closes[i - 1]:
                if up_count > 0:
                    break
                down_count += 1
            else:
                break
        features[8] = float(up_count)
        features[9] = float(down_count)

    # Time features
    if hasattr(df.index[bar_idx], 'hour'):
        dt = df.index[bar_idx]
        hour = dt.hour
        features[10] = math.sin(2 * math.pi * hour / 24)
        features[11] = math.cos(2 * math.pi * hour / 24)
        features[12] = float(dt.weekday()) if hasattr(dt, 'weekday') else 0
        # Minutes since market open (9:30 ET = 14:30 UTC)
        utc_minutes = hour * 60 + dt.minute
        market_open_utc = 14 * 60 + 30
        features[13] = max(0, utc_minutes - market_open_utc)

    return features


def extract_temporal_features(
    current_state: Dict,
    history_buffer: List[Dict],
    closes: Optional[np.ndarray] = None,
    bar_idx: int = 0,
    eval_interval: int = 3,
) -> np.ndarray:
    """
    Extract temporal/trajectory features by comparing current snapshot to recent history.

    Args:
        current_state: Dict with current feature values {name: float}
        history_buffer: List of recent feature dicts (oldest first), one per eval interval
        closes: Price array for computing price momentum
        bar_idx: Current bar index into closes
        eval_interval: Bars between evaluations (for converting lookback to bar count)

    Returns:
        Array of temporal features.
    """
    features = np.zeros(len(TEMPORAL_FEATURES), dtype=np.float32)

    def delta(key, lookback_steps):
        """Compute feature change from lookback_steps evaluations ago to current."""
        if lookback_steps <= len(history_buffer) and key in current_state:
            past = history_buffer[-lookback_steps].get(key, 0)
            return current_state.get(key, 0) - past
        return 0.0

    # Position trajectory
    features[0] = delta('5min_position_pct', 3)
    features[1] = delta('5min_position_pct', 6)

    # Health trajectory
    features[2] = delta('5min_channel_health', 3)
    features[3] = delta('5min_channel_health', 6)

    # Entropy trajectory (rising entropy = channel failing)
    features[4] = delta('5min_entropy', 3)

    # Break prob trajectory
    features[5] = delta('5min_break_prob', 3)

    # Energy trajectory
    features[6] = delta('5min_total_energy', 3)

    # RSI slope
    features[7] = delta('rsi_14', 5)

    # Price momentum (computed from closes, normalized by price)
    if closes is not None and bar_idx >= 12 * eval_interval:
        price = closes[bar_idx]
        if price > 0:
            bars_3 = 3 * eval_interval
            bars_12 = 12 * eval_interval
            features[8] = (price - closes[bar_idx - bars_3]) / price
            features[9] = (price - closes[bar_idx - bars_12]) / price

    # Volume momentum
    features[10] = delta('volume_ratio_20', 3)

    # Width trajectory
    features[11] = delta('5min_width_pct', 3)

    return features


def extract_correlation_features(
    bar_idx: int,
    tsla_closes: np.ndarray,
    spy_df: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None,
    tsla_index=None,
) -> np.ndarray:
    """Extract SPY/VIX correlation features."""
    features = np.zeros(len(CORRELATION_FEATURES), dtype=np.float32)

    if spy_df is not None and len(spy_df) > 0 and tsla_index is not None:
        # Align SPY data to current TSLA timestamp
        current_time = tsla_index[bar_idx]

        # Get SPY data up to current time (no lookahead)
        if spy_df.index.tz is not None and current_time.tzinfo is None:
            spy_available = spy_df[spy_df.index.tz_localize(None) <= current_time]
        elif spy_df.index.tz is None and current_time.tzinfo is not None:
            spy_available = spy_df[spy_df.index <= current_time.tz_localize(None)]
        else:
            spy_available = spy_df[spy_df.index <= current_time]

        if len(spy_available) >= 20:
            spy_closes = spy_available['close'].values

            # SPY 5-bar return
            if len(spy_closes) >= 6:
                features[0] = (spy_closes[-1] - spy_closes[-6]) / spy_closes[-6]

            # SPY 20-bar return
            if len(spy_closes) >= 21:
                features[1] = (spy_closes[-1] - spy_closes[-21]) / spy_closes[-21]

            # SPY-TSLA correlation (20-bar rolling)
            if bar_idx >= 20:
                tsla_recent = tsla_closes[bar_idx - 19:bar_idx + 1]
                spy_recent = spy_closes[-20:]
                if len(tsla_recent) == len(spy_recent) and np.std(tsla_recent) > 0 and np.std(spy_recent) > 0:
                    features[2] = np.corrcoef(tsla_recent, spy_recent)[0, 1]

    if vix_df is not None and len(vix_df) > 0 and tsla_index is not None:
        current_time = tsla_index[bar_idx]

        if vix_df.index.tz is not None and current_time.tzinfo is None:
            vix_available = vix_df[vix_df.index.tz_localize(None) <= current_time]
        elif vix_df.index.tz is None and current_time.tzinfo is not None:
            vix_available = vix_df[vix_df.index <= current_time.tz_localize(None)]
        else:
            vix_available = vix_df[vix_df.index <= current_time]

        if len(vix_available) >= 5:
            vix_closes = vix_available['close'].values
            features[3] = vix_closes[-1]
            if len(vix_closes) >= 6:
                features[4] = vix_closes[-1] - vix_closes[-6]

    return features


# ---------------------------------------------------------------------------
# Label Generation
# ---------------------------------------------------------------------------

@dataclass
class MLLabels:
    """Labels for a single training example."""
    channel_lifetime: float    # Bars until channel breaks (0 = already broken)
    channel_censored: bool     # True if channel was still alive at end of data
    break_direction: int       # 0 = no break (survive), 1 = break up, 2 = break down
    optimal_action: int        # 0 = hold, 1 = buy, 2 = sell
    future_return_5: float     # 5-bar forward return
    future_return_20: float    # 20-bar forward return
    future_return_60: float    # 60-bar forward return (5 hours)


def compute_labels(
    bar_idx: int,
    closes: np.ndarray,
    channel: 'Channel',
    channel_end_bar: Optional[int],  # Bar where this channel breaks
    channel_end_direction: Optional[str],  # 'up', 'down', or None
) -> Optional[MLLabels]:
    """Compute labels for a training example."""
    total_bars = len(closes)
    current_price = closes[bar_idx]

    # Channel lifetime (capped at 200 bars = ~16 hours to avoid censored skew)
    MAX_LIFETIME = 200
    if channel_end_bar is not None:
        lifetime = min(MAX_LIFETIME, max(0, channel_end_bar - bar_idx))
        censored = False
    else:
        lifetime = MAX_LIFETIME  # Channel still alive → capped
        censored = True

    # Break direction
    if channel_end_bar is not None and bar_idx < channel_end_bar:
        if channel_end_direction == 'up':
            break_dir = 1
        elif channel_end_direction == 'down':
            break_dir = 2
        else:
            break_dir = 0
    else:
        break_dir = 0  # No break or already past

    # Future returns
    def safe_return(bars_ahead):
        future_idx = bar_idx + bars_ahead
        if future_idx < total_bars:
            return (closes[future_idx] - current_price) / current_price
        return 0.0

    future_5 = safe_return(5)
    future_20 = safe_return(20)
    future_60 = safe_return(60)

    # Optimal action (based on risk-adjusted future opportunity)
    # Look at max favorable/adverse excursion over next 20 bars
    window_end = min(bar_idx + 20, total_bars)
    future_prices = closes[bar_idx + 1:window_end]

    if len(future_prices) >= 5:
        max_up = (np.max(future_prices) - current_price) / current_price
        max_down = (current_price - np.min(future_prices)) / current_price

        # BUY: upside > 0.4% and upside/downside > 1.5
        # SELL: downside > 0.4% and downside/upside > 1.5
        if max_up > 0.004 and (max_up > max_down * 1.5 or max_up > 0.008):
            action = 1  # BUY
        elif max_down > 0.004 and (max_down > max_up * 1.5 or max_down > 0.008):
            action = 2  # SELL
        else:
            action = 0  # HOLD
    else:
        action = 0

    return MLLabels(
        channel_lifetime=float(lifetime),
        channel_censored=censored,
        break_direction=break_dir,
        optimal_action=action,
        future_return_5=future_5,
        future_return_20=future_20,
        future_return_60=future_60,
    )


# ---------------------------------------------------------------------------
# Training Data Generation
# ---------------------------------------------------------------------------

def detect_channel_breaks(
    df: pd.DataFrame,
    window: int = 30,
    eval_interval: int = 3,
    min_channel_bars: int = 12,  # Minimum 1 hour (12 x 5min) to count as channel
) -> List[Dict]:
    """
    Walk through bars detecting channels and when they break.

    Uses a two-pass approach:
    1. Track channel boundaries using exponential smoothing (not raw detection)
    2. Detect breaks when price exits the smoothed boundaries

    This is more robust than comparing raw channel params at each bar,
    which creates spurious "channel change" events from detection noise.

    Returns list of {start_bar, end_bar, break_direction} dicts.
    """
    from v15.core.channel import detect_channels_multi_window, select_best_channel

    closes = df['close'].values
    total_bars = len(df)

    # First pass: detect channel at each bar
    bar_channels = {}  # bar -> (upper, lower, center, valid)
    alpha = 0.2  # Smoothing factor for boundary tracking

    smoothed_upper = None
    smoothed_lower = None

    for bar in range(100, total_bars, eval_interval):
        lookback = min(bar + 1, 100)
        df_slice = df.iloc[bar - lookback + 1:bar + 1]

        if len(df_slice) < 20:
            bar_channels[bar] = (None, None, None, False)
            continue

        try:
            multi = detect_channels_multi_window(df_slice, windows=[10, 15, 20, 30, 40])
            best_ch, _ = select_best_channel(multi)
        except Exception:
            best_ch = None

        if best_ch is None or not best_ch.valid:
            bar_channels[bar] = (smoothed_upper, smoothed_lower, None, False)
            continue

        raw_upper = best_ch.upper_line[-1]
        raw_lower = best_ch.lower_line[-1]
        raw_center = best_ch.center_line[-1]

        # Exponential smoothing of boundaries
        if smoothed_upper is None:
            smoothed_upper = raw_upper
            smoothed_lower = raw_lower
        else:
            smoothed_upper = alpha * raw_upper + (1 - alpha) * smoothed_upper
            smoothed_lower = alpha * raw_lower + (1 - alpha) * smoothed_lower

        bar_channels[bar] = (smoothed_upper, smoothed_lower, raw_center, True)

    # Second pass: detect breaks using smoothed boundaries
    channel_events = []
    channel_start = None
    in_channel = False
    ref_upper = None
    ref_lower = None
    no_channel_count = 0

    sorted_bars = sorted(bar_channels.keys())

    for bar in sorted_bars:
        upper, lower, center, valid = bar_channels[bar]
        price = closes[bar]

        if valid and upper is not None and lower is not None:
            width = upper - lower

            if not in_channel:
                # Start a new channel
                channel_start = bar
                ref_upper = upper
                ref_lower = lower
                in_channel = True
                no_channel_count = 0
                continue

            no_channel_count = 0

            # Check for break: price exits smoothed boundaries by a margin
            break_margin = width * 0.15  # 15% of width beyond boundary = break
            if price > upper + break_margin:
                # Upward break
                if bar - channel_start >= min_channel_bars:
                    channel_events.append({
                        'start_bar': channel_start,
                        'end_bar': bar,
                        'break_direction': 'up',
                    })
                channel_start = bar
                ref_upper = upper
                ref_lower = lower

            elif price < lower - break_margin:
                # Downward break
                if bar - channel_start >= min_channel_bars:
                    channel_events.append({
                        'start_bar': channel_start,
                        'end_bar': bar,
                        'break_direction': 'down',
                    })
                channel_start = bar
                ref_upper = upper
                ref_lower = lower

            else:
                # Still in channel — update reference
                ref_upper = upper
                ref_lower = lower

        else:
            # No valid channel detected
            no_channel_count += 1

            # If we've been without a channel for 5+ intervals, end the current one
            if in_channel and no_channel_count >= 5:
                if bar - channel_start >= min_channel_bars:
                    # Determine break direction from final price vs boundaries
                    if ref_upper is not None and ref_lower is not None:
                        if price > ref_upper:
                            break_dir = 'up'
                        elif price < ref_lower:
                            break_dir = 'down'
                        else:
                            break_dir = None
                    else:
                        break_dir = None

                    channel_events.append({
                        'start_bar': channel_start,
                        'end_bar': bar,
                        'break_direction': break_dir,
                    })
                in_channel = False
                channel_start = None
                ref_upper = None
                ref_lower = None
                smoothed_upper = None
                smoothed_lower = None

    # Handle last channel (censored)
    if in_channel and channel_start is not None:
        end_bar = sorted_bars[-1] if sorted_bars else total_bars - 1
        if end_bar - channel_start >= min_channel_bars:
            channel_events.append({
                'start_bar': channel_start,
                'end_bar': end_bar,
                'break_direction': None,  # Censored — still alive
            })

    return channel_events


def generate_training_data(
    days: int = 120,
    eval_interval: int = 3,
    use_multi_tf: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    """
    Generate labeled training data from historical bars.

    Returns:
        (X, Y_dict, feature_names) where:
        - X: (N, num_features) float32 array
        - Y_dict: {label_name: array} with all labels
        - feature_names: ordered list of feature names
    """
    import yfinance as yf
    from v15.core.channel import detect_channels_multi_window, select_best_channel
    from v15.core.channel_surfer import analyze_channels, SIGNAL_TFS, TF_WINDOWS

    if verbose:
        print(f"=== Generating ML Training Data ({days}d) ===")

    # Fetch TSLA 5min data
    if verbose:
        print("Fetching TSLA 5min data...")
    tsla = yf.download('TSLA', period=f'{days}d', interval='5m', progress=False)
    if isinstance(tsla.columns, pd.MultiIndex):
        tsla.columns = tsla.columns.get_level_values(0)
    tsla.columns = [c.lower() for c in tsla.columns]
    if verbose:
        print(f"  Got {len(tsla)} bars")

    # Fetch higher TF data
    higher_tf_data = {}
    if use_multi_tf:
        for tf_label, yf_interval, yf_period in [
            ('1h', '1h', '2y'),
            ('4h', '1h', '2y'),  # Resample from 1h
            ('daily', '1d', '5y'),
            ('weekly', '1wk', '5y'),
        ]:
            if verbose:
                print(f"  Fetching {tf_label} data...")

            if tf_label == '4h':
                # Resample 1h to 4h
                if '1h' in higher_tf_data:
                    h1 = higher_tf_data['1h']
                    resampled = h1.resample('4h').agg({
                        'open': 'first', 'high': 'max', 'low': 'min',
                        'close': 'last', 'volume': 'sum',
                    }).dropna()
                    higher_tf_data[tf_label] = resampled
                    if verbose:
                        print(f"  {tf_label}: {len(resampled)} bars (resampled from 1h)")
                continue

            raw = yf.download('TSLA', period=yf_period, interval=yf_interval, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.columns = [c.lower() for c in raw.columns]
            higher_tf_data[tf_label] = raw
            if verbose:
                print(f"  {tf_label}: {len(raw)} bars")

    # Fetch SPY and VIX for correlation features
    if verbose:
        print("  Fetching SPY data...")
    spy_df = yf.download('SPY', period=f'{days}d', interval='5m', progress=False)
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = spy_df.columns.get_level_values(0)
    spy_df.columns = [c.lower() for c in spy_df.columns]
    if verbose:
        print(f"  SPY: {len(spy_df)} bars")

    if verbose:
        print("  Fetching VIX data...")
    try:
        vix_df = yf.download('^VIX', period='1y', interval='1d', progress=False)
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)
        vix_df.columns = [c.lower() for c in vix_df.columns]
        if verbose:
            print(f"  VIX: {len(vix_df)} bars")
    except Exception:
        vix_df = None
        if verbose:
            print("  VIX: failed to fetch (will use zeros)")

    if len(tsla) < 300:
        raise ValueError(f"Not enough TSLA data: {len(tsla)} bars (need 300+)")

    closes = tsla['close'].values
    total_bars = len(tsla)

    # Detect channel breaks for labeling
    if verbose:
        print("\nDetecting channel breaks for labeling...")
    channel_events = detect_channel_breaks(tsla, eval_interval=eval_interval)
    if verbose:
        print(f"  Found {len(channel_events)} channel events")
        for i, ev in enumerate(channel_events[:5]):
            print(f"    Channel {i}: bars {ev['start_bar']}-{ev['end_bar']}, break={ev['break_direction']}")

    # Build a lookup: for each bar, which channel event does it belong to?
    bar_to_event = {}
    for ev in channel_events:
        for b in range(ev['start_bar'], ev['end_bar'] + 1):
            bar_to_event[b] = ev

    # Walk through bars and extract features + labels
    feature_names = get_feature_names()
    num_features = len(feature_names)

    X_list = []
    labels = {
        'channel_lifetime': [],
        'channel_censored': [],
        'break_direction': [],
        'optimal_action': [],
        'future_return_5': [],
        'future_return_20': [],
        'future_return_60': [],
    }

    start_bar = 100
    sample_count = 0
    t_start = time.time()
    history_buffer: List[Dict] = []  # Stores feature snapshots for temporal features

    if verbose:
        print(f"\nExtracting features from bar {start_bar} to {total_bars}...")

    for bar in range(start_bar, total_bars - 60, eval_interval):  # Leave 60 bars for future returns
        if verbose and sample_count % 200 == 0 and sample_count > 0:
            pct = (bar - start_bar) / (total_bars - 60 - start_bar) * 100
            print(f"  [{pct:.0f}%] bar={bar}, samples={sample_count}")

        current_price = float(closes[bar])

        # --- Detect channels at this bar ---
        lookback = min(bar + 1, 100)
        df_slice = tsla.iloc[bar - lookback + 1:bar + 1]

        if len(df_slice) < 20:
            continue

        try:
            multi = detect_channels_multi_window(df_slice, windows=[10, 15, 20, 30, 40])
            best_ch, _ = select_best_channel(multi)
        except Exception:
            continue

        if best_ch is None or not best_ch.valid:
            continue

        # Build multi-TF channels
        slice_closes = df_slice['close'].values
        channels_by_tf = {'5min': best_ch}
        prices_by_tf = {'5min': slice_closes}
        current_prices_dict = {'5min': current_price}
        volumes_dict = {}

        if 'volume' in df_slice.columns:
            volumes_dict['5min'] = df_slice['volume'].values

        if use_multi_tf:
            current_time = tsla.index[bar]
            current_time_naive = current_time.tz_localize(None) if current_time.tzinfo else current_time

            for tf_label, tf_df in higher_tf_data.items():
                tf_idx = tf_df.index
                if tf_idx.tz is not None:
                    tf_available = tf_df[tf_idx <= current_time]
                else:
                    tf_available = tf_df[tf_idx <= current_time_naive]

                tf_recent = tf_available.tail(100)
                if len(tf_recent) < 30:
                    continue

                tf_windows = TF_WINDOWS.get(tf_label, [20, 30, 40])
                try:
                    from v15.core.channel import detect_channels_multi_window, select_best_channel
                    tf_multi = detect_channels_multi_window(tf_recent, windows=tf_windows)
                    tf_ch, _ = select_best_channel(tf_multi)
                    if tf_ch and tf_ch.valid:
                        channels_by_tf[tf_label] = tf_ch
                        prices_by_tf[tf_label] = tf_recent['close'].values
                        current_prices_dict[tf_label] = float(tf_recent['close'].iloc[-1])
                        if 'volume' in tf_recent.columns:
                            volumes_dict[tf_label] = tf_recent['volume'].values
                except Exception:
                    pass

        # Run analysis to get TFChannelStates
        try:
            analysis = analyze_channels(
                channels_by_tf, prices_by_tf, current_prices_dict,
                volumes_by_tf=volumes_dict if volumes_dict else None,
            )
        except Exception:
            continue

        # --- Extract features ---
        feature_vec = np.zeros(num_features, dtype=np.float32)
        offset = 0

        # Per-TF features
        for tf in ML_TFS:
            state = analysis.tf_states.get(tf)
            if state:
                tf_feats = extract_tf_features(state)
            else:
                tf_feats = np.zeros(len(PER_TF_FEATURES), dtype=np.float32)
            feature_vec[offset:offset + len(PER_TF_FEATURES)] = tf_feats
            offset += len(PER_TF_FEATURES)

        # Cross-TF features
        cross_feats = extract_cross_tf_features(analysis.tf_states)
        feature_vec[offset:offset + len(CROSS_TF_FEATURES)] = cross_feats
        offset += len(CROSS_TF_FEATURES)

        # Context features
        ctx_feats = extract_context_features(tsla, bar)
        feature_vec[offset:offset + len(CONTEXT_FEATURES)] = ctx_feats
        offset += len(CONTEXT_FEATURES)

        # Build current state snapshot for temporal features
        current_snapshot = {}
        for tf in ML_TFS:
            state = analysis.tf_states.get(tf)
            if state and state.valid:
                for feat_name in PER_TF_FEATURES:
                    val = getattr(state, feat_name, 0.0)
                    if isinstance(val, (int, float)):
                        current_snapshot[f'{tf}_{feat_name}'] = float(val)
        current_snapshot['rsi_14'] = float(ctx_feats[0])
        current_snapshot['volume_ratio_20'] = float(ctx_feats[2])

        # Temporal features
        temporal_feats = extract_temporal_features(
            current_snapshot, history_buffer,
            closes=closes, bar_idx=bar, eval_interval=eval_interval,
        )
        feature_vec[offset:offset + len(TEMPORAL_FEATURES)] = temporal_feats
        offset += len(TEMPORAL_FEATURES)

        # Update history buffer (keep last 20 snapshots)
        history_buffer.append(current_snapshot)
        if len(history_buffer) > 20:
            history_buffer.pop(0)

        # Correlation features
        corr_feats = extract_correlation_features(
            bar, closes, spy_df=spy_df, vix_df=vix_df, tsla_index=tsla.index,
        )
        feature_vec[offset:offset + len(CORRELATION_FEATURES)] = corr_feats
        offset += len(CORRELATION_FEATURES)

        # --- Compute labels ---
        ev = bar_to_event.get(bar)
        if ev:
            channel_end = ev['end_bar']
            break_dir = ev['break_direction']
        else:
            channel_end = None
            break_dir = None

        lbl = compute_labels(bar, closes, best_ch, channel_end, break_dir)
        if lbl is None:
            continue

        # Store
        X_list.append(feature_vec)
        labels['channel_lifetime'].append(lbl.channel_lifetime)
        labels['channel_censored'].append(float(lbl.channel_censored))
        labels['break_direction'].append(lbl.break_direction)
        labels['optimal_action'].append(lbl.optimal_action)
        labels['future_return_5'].append(lbl.future_return_5)
        labels['future_return_20'].append(lbl.future_return_20)
        labels['future_return_60'].append(lbl.future_return_60)

        sample_count += 1

    elapsed = time.time() - t_start
    if verbose:
        print(f"\n  Generated {sample_count} samples in {elapsed:.1f}s")

    X = np.array(X_list, dtype=np.float32)
    Y_dict = {k: np.array(v, dtype=np.float32) for k, v in labels.items()}

    return X, Y_dict, feature_names


# ---------------------------------------------------------------------------
# Architecture 1: Gradient Boosted Trees (Baseline)
# ---------------------------------------------------------------------------

class GBTModel:
    """
    Multi-output Gradient Boosted Trees baseline.

    Trains separate models for:
    - Channel lifetime (regression)
    - Break direction (3-class classification)
    - Optimal action (3-class classification)
    - Future returns (regression)
    """

    def __init__(self):
        self.models = {}
        self.feature_names = None
        self.feature_importance = {}

    def train(
        self,
        X_train: np.ndarray,
        Y_train: Dict[str, np.ndarray],
        X_val: np.ndarray,
        Y_val: Dict[str, np.ndarray],
        feature_names: List[str],
    ) -> Dict[str, float]:
        """Train all GBT sub-models."""
        try:
            import lightgbm as lgb
        except ImportError:
            print("LightGBM not available, falling back to sklearn GBT")
            return self._train_sklearn(X_train, Y_train, X_val, Y_val, feature_names)

        self.feature_names = feature_names
        metrics = {}

        # 1. Channel lifetime (regression)
        print("\n  Training: channel_lifetime (regression)...")
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 63,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': -1,
        }

        dtrain = lgb.Dataset(X_train, label=Y_train['channel_lifetime'],
                             feature_name=feature_names)
        dval = lgb.Dataset(X_val, label=Y_val['channel_lifetime'],
                           feature_name=feature_names, reference=dtrain)

        self.models['lifetime'] = lgb.train(
            params, dtrain, num_boost_round=500,
            valid_sets=[dval], callbacks=[
                lgb.log_evaluation(50),
                lgb.early_stopping(stopping_rounds=30),
            ],
        )

        pred = self.models['lifetime'].predict(X_val)
        mae = np.mean(np.abs(pred - Y_val['channel_lifetime']))
        metrics['lifetime_mae'] = float(mae)
        print(f"    Lifetime MAE: {mae:.1f} bars")

        # Feature importance for lifetime (save all, sorted by importance)
        imp = self.models['lifetime'].feature_importance(importance_type='gain')
        sorted_idx = np.argsort(imp)[::-1]
        self.feature_importance['lifetime'] = [
            (feature_names[i], float(imp[i])) for i in sorted_idx
        ]

        # 2. Break direction (3-class)
        print("\n  Training: break_direction (3-class)...")
        params_cls = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'num_leaves': 63,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': -1,
        }

        dtrain_bd = lgb.Dataset(X_train, label=Y_train['break_direction'].astype(int),
                                feature_name=feature_names)
        dval_bd = lgb.Dataset(X_val, label=Y_val['break_direction'].astype(int),
                              feature_name=feature_names, reference=dtrain_bd)

        self.models['break_dir'] = lgb.train(
            params_cls, dtrain_bd, num_boost_round=500,
            valid_sets=[dval_bd], callbacks=[
                lgb.log_evaluation(50),
                lgb.early_stopping(stopping_rounds=30),
            ],
        )

        pred_probs = self.models['break_dir'].predict(X_val)
        pred_cls = np.argmax(pred_probs, axis=1)
        accuracy = np.mean(pred_cls == Y_val['break_direction'].astype(int))
        metrics['break_dir_accuracy'] = float(accuracy)
        print(f"    Break direction accuracy: {accuracy:.1%}")

        # 3. Optimal action (3-class)
        print("\n  Training: optimal_action (3-class)...")
        dtrain_oa = lgb.Dataset(X_train, label=Y_train['optimal_action'].astype(int),
                                feature_name=feature_names)
        dval_oa = lgb.Dataset(X_val, label=Y_val['optimal_action'].astype(int),
                              feature_name=feature_names, reference=dtrain_oa)

        self.models['action'] = lgb.train(
            params_cls, dtrain_oa, num_boost_round=500,
            valid_sets=[dval_oa], callbacks=[
                lgb.log_evaluation(50),
                lgb.early_stopping(stopping_rounds=30),
            ],
        )

        pred_probs = self.models['action'].predict(X_val)
        pred_cls = np.argmax(pred_probs, axis=1)
        accuracy = np.mean(pred_cls == Y_val['optimal_action'].astype(int))
        metrics['action_accuracy'] = float(accuracy)
        print(f"    Action accuracy: {accuracy:.1%}")

        # 4. Future returns (regression)
        for horizon in ['future_return_5', 'future_return_20', 'future_return_60']:
            print(f"\n  Training: {horizon} (regression)...")
            dtrain_r = lgb.Dataset(X_train, label=Y_train[horizon],
                                   feature_name=feature_names)
            dval_r = lgb.Dataset(X_val, label=Y_val[horizon],
                                 feature_name=feature_names, reference=dtrain_r)

            self.models[horizon] = lgb.train(
                params, dtrain_r, num_boost_round=300,
                valid_sets=[dval_r], callbacks=[
                    lgb.log_evaluation(100),
                    lgb.early_stopping(stopping_rounds=30),
                ],
            )

            pred = self.models[horizon].predict(X_val)
            mae = np.mean(np.abs(pred - Y_val[horizon]))
            # Directional accuracy
            dir_acc = np.mean(np.sign(pred) == np.sign(Y_val[horizon]))
            metrics[f'{horizon}_mae'] = float(mae)
            metrics[f'{horizon}_dir_acc'] = float(dir_acc)
            print(f"    {horizon} MAE: {mae:.5f}, Dir Acc: {dir_acc:.1%}")

        return metrics

    def _train_sklearn(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Fallback training with sklearn."""
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

        self.feature_names = feature_names
        metrics = {}

        # Lifetime
        print("\n  Training: channel_lifetime (sklearn GBT)...")
        self.models['lifetime'] = GradientBoostingRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        self.models['lifetime'].fit(X_train, Y_train['channel_lifetime'])
        pred = self.models['lifetime'].predict(X_val)
        mae = np.mean(np.abs(pred - Y_val['channel_lifetime']))
        metrics['lifetime_mae'] = float(mae)
        print(f"    Lifetime MAE: {mae:.1f} bars")

        # Break direction
        print("\n  Training: break_direction (sklearn GBT)...")
        self.models['break_dir'] = GradientBoostingClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        self.models['break_dir'].fit(X_train, Y_train['break_direction'].astype(int))
        pred = self.models['break_dir'].predict(X_val)
        accuracy = np.mean(pred == Y_val['break_direction'].astype(int))
        metrics['break_dir_accuracy'] = float(accuracy)
        print(f"    Break direction accuracy: {accuracy:.1%}")

        # Action
        print("\n  Training: optimal_action (sklearn GBT)...")
        self.models['action'] = GradientBoostingClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        self.models['action'].fit(X_train, Y_train['optimal_action'].astype(int))
        pred = self.models['action'].predict(X_val)
        accuracy = np.mean(pred == Y_val['optimal_action'].astype(int))
        metrics['action_accuracy'] = float(accuracy)
        print(f"    Action accuracy: {accuracy:.1%}")

        # Future returns
        for horizon in ['future_return_5', 'future_return_20', 'future_return_60']:
            print(f"\n  Training: {horizon} (sklearn GBT)...")
            self.models[horizon] = GradientBoostingRegressor(
                n_estimators=150, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )
            self.models[horizon].fit(X_train, Y_train[horizon])
            pred = self.models[horizon].predict(X_val)
            mae = np.mean(np.abs(pred - Y_val[horizon]))
            dir_acc = np.mean(np.sign(pred) == np.sign(Y_val[horizon]))
            metrics[f'{horizon}_mae'] = float(mae)
            metrics[f'{horizon}_dir_acc'] = float(dir_acc)
            print(f"    {horizon} MAE: {mae:.5f}, Dir Acc: {dir_acc:.1%}")

        return metrics

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference on feature array."""
        results = {}

        if 'lifetime' in self.models:
            results['lifetime'] = self.models['lifetime'].predict(X)

        if 'break_dir' in self.models:
            probs = self.models['break_dir'].predict(X)
            if hasattr(probs, 'shape') and len(probs.shape) == 2:
                results['break_dir_probs'] = probs
                results['break_dir'] = np.argmax(probs, axis=1)
            else:
                # sklearn returns class predictions
                results['break_dir'] = probs

        if 'action' in self.models:
            probs = self.models['action'].predict(X)
            if hasattr(probs, 'shape') and len(probs.shape) == 2:
                results['action_probs'] = probs
                results['action'] = np.argmax(probs, axis=1)
            else:
                results['action'] = probs

        for horizon in ['future_return_5', 'future_return_20', 'future_return_60']:
            if horizon in self.models:
                results[horizon] = self.models[horizon].predict(X)

        return results

    def save(self, path: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance,
            }, f)
        print(f"  Saved GBT model to {path}")

    @classmethod
    def load(cls, path: str) -> 'GBTModel':
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.models = data['models']
        model.feature_names = data['feature_names']
        model.feature_importance = data.get('feature_importance', {})
        return model


# ---------------------------------------------------------------------------
# Architecture 2: Survival Analysis (DeepSurv)
# ---------------------------------------------------------------------------

class SurvivalModel:
    """
    Cox Proportional Hazard neural network for channel lifetime prediction.

    Novel approach: treats channel ending as a 'death event' in survival analysis.
    The model learns the hazard function h(t|X) = h0(t) * exp(f(X)) where f(X)
    is a neural network.

    Key advantage: handles censored data (channels still alive at end of
    observation) which regression cannot.

    Outputs:
    - Hazard rate at each future bar
    - Median survival time (expected channel lifetime)
    - Survival probability at t=10, t=30, t=60 bars
    """

    def __init__(self, input_dim: int = 0, hidden_dims: List[int] = None):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [128, 64, 32]
        self.net = None
        self.baseline_hazard = None
        self.feature_names = None
        self._device = 'cpu'

    def _build_network(self):
        """Build the Cox PH neural network."""
        import torch
        import torch.nn as nn

        layers = []
        prev_dim = self.input_dim

        for h_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.SELU(),
                nn.AlphaDropout(0.1),
            ])
            prev_dim = h_dim

        # Single output: log-risk score
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)
        return self.net

    def _cox_loss(self, risk_scores, times, events):
        """
        Negative partial log-likelihood for Cox PH model.

        Handles censored data: events=0 means censored (channel still alive),
        events=1 means channel actually broke.
        """
        import torch

        # Sort by time (descending)
        sorted_idx = torch.argsort(times, descending=True)
        risk_sorted = risk_scores[sorted_idx]
        events_sorted = events[sorted_idx]

        # Log-sum-exp of risk set (cumulative from longest to shortest)
        max_risk = risk_sorted.max()
        log_cumsum = torch.logcumsumexp(risk_sorted - max_risk, dim=0) + max_risk

        # Partial likelihood (only for events=1)
        log_likelihood = risk_sorted - log_cumsum
        log_likelihood = log_likelihood * events_sorted

        # Negative log-likelihood
        n_events = events_sorted.sum()
        if n_events > 0:
            return -log_likelihood.sum() / n_events
        else:
            return torch.tensor(0.0, requires_grad=True)

    def train(
        self,
        X_train: np.ndarray,
        Y_train: Dict[str, np.ndarray],
        X_val: np.ndarray,
        Y_val: Dict[str, np.ndarray],
        feature_names: List[str],
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """Train survival model."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self.feature_names = feature_names
        self.input_dim = X_train.shape[1]
        self._build_network()

        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self._device = str(device)
        self.net.to(device)

        # Prepare data
        # Times = channel_lifetime, Events = NOT censored
        train_X = torch.FloatTensor(X_train).to(device)
        train_times = torch.FloatTensor(Y_train['channel_lifetime']).to(device)
        train_events = torch.FloatTensor(1.0 - Y_train['channel_censored']).to(device)

        val_X = torch.FloatTensor(X_val).to(device)
        val_times = torch.FloatTensor(Y_val['channel_lifetime']).to(device)
        val_events = torch.FloatTensor(1.0 - Y_val['channel_censored']).to(device)

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float('inf')
        best_state = None
        patience = 30
        no_improve = 0

        print(f"\n  Training Survival Model ({self.input_dim} features → hazard)")
        print(f"  Device: {device}")
        print(f"  Train: {len(X_train)} samples, Val: {len(X_val)} samples")
        print(f"  Uncensored events: train={int(train_events.sum())}, val={int(val_events.sum())}")

        dataset = TensorDataset(train_X, train_times, train_events)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for epoch in range(epochs):
            self.net.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch_X, batch_times, batch_events in loader:
                risk_scores = self.net(batch_X).squeeze(-1)
                loss = self._cox_loss(risk_scores, batch_times, batch_events)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Validation
            self.net.eval()
            with torch.no_grad():
                val_risk = self.net(val_X).squeeze(-1)
                val_loss = self._cox_loss(val_risk, val_times, val_events).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs} | Train: {epoch_loss/max(n_batches,1):.4f} | Val: {val_loss:.4f} | Best: {best_val_loss:.4f}")

            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

        # Restore best
        if best_state:
            self.net.load_state_dict(best_state)

        # Compute concordance index
        self.net.eval()
        with torch.no_grad():
            val_risk = self.net(val_X).squeeze(-1).cpu().numpy()

        c_index = self._concordance_index(
            val_times.cpu().numpy(),
            val_events.cpu().numpy(),
            -val_risk,  # Negative because higher risk = shorter survival
        )

        # Estimate baseline hazard (Breslow estimator)
        self._estimate_baseline_hazard(
            train_times.cpu().numpy(),
            train_events.cpu().numpy(),
            self.net(train_X).squeeze(-1).detach().cpu().numpy(),
        )

        metrics = {
            'concordance_index': float(c_index),
            'best_val_loss': float(best_val_loss),
        }

        print(f"\n  Survival Model Results:")
        print(f"    Concordance Index: {c_index:.3f} (0.5 = random, 1.0 = perfect)")
        print(f"    Best Val Loss: {best_val_loss:.4f}")

        # Median survival prediction accuracy
        if self.baseline_hazard is not None:
            predicted_medians = self.predict_median_survival(X_val)
            actual_lifetimes = Y_val['channel_lifetime']
            uncensored = Y_val['channel_censored'] == 0
            if uncensored.sum() > 0:
                mae = np.mean(np.abs(predicted_medians[uncensored] - actual_lifetimes[uncensored]))
                metrics['median_survival_mae'] = float(mae)
                print(f"    Median Survival MAE (uncensored): {mae:.1f} bars")

        return metrics

    def _concordance_index(self, times, events, predicted_scores):
        """Compute Harrell's concordance index."""
        concordant = 0
        discordant = 0
        tied = 0

        n = len(times)
        for i in range(n):
            if events[i] == 0:
                continue  # Skip censored
            for j in range(n):
                if i == j:
                    continue
                if times[j] > times[i]:
                    if predicted_scores[j] > predicted_scores[i]:
                        concordant += 1
                    elif predicted_scores[j] < predicted_scores[i]:
                        discordant += 1
                    else:
                        tied += 1

        total = concordant + discordant + tied
        if total == 0:
            return 0.5
        return (concordant + 0.5 * tied) / total

    def _estimate_baseline_hazard(self, times, events, risk_scores):
        """Breslow estimator for baseline cumulative hazard."""
        # Sort by time
        sorted_idx = np.argsort(times)
        sorted_times = times[sorted_idx]
        sorted_events = events[sorted_idx]
        sorted_risk = np.exp(risk_scores[sorted_idx])

        unique_times = np.unique(sorted_times[sorted_events == 1])
        baseline_hazard = {}

        for t in unique_times:
            # Number of events at time t
            d_t = np.sum((sorted_times == t) & (sorted_events == 1))
            # Risk set at time t
            risk_set = sorted_risk[sorted_times >= t].sum()
            if risk_set > 0:
                baseline_hazard[t] = d_t / risk_set

        self.baseline_hazard = baseline_hazard

    def predict_median_survival(self, X: np.ndarray) -> np.ndarray:
        """Predict median survival time for each sample."""
        import torch

        device = torch.device(self._device)
        self.net.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            risk_scores = self.net(X_tensor).squeeze(-1).cpu().numpy()

        exp_risk = np.exp(risk_scores)

        if self.baseline_hazard is None:
            return np.full(len(X), 30.0)  # Default guess

        # Cumulative baseline hazard
        sorted_times = sorted(self.baseline_hazard.keys())
        cum_hazard = 0.0
        medians = np.full(len(X), sorted_times[-1] if sorted_times else 30.0)

        cum_h = {}
        running = 0.0
        for t in sorted_times:
            running += self.baseline_hazard[t]
            cum_h[t] = running

        for i in range(len(X)):
            for t in sorted_times:
                # S(t) = exp(-H0(t) * exp(risk))
                survival_prob = math.exp(-cum_h[t] * exp_risk[i])
                if survival_prob <= 0.5:
                    medians[i] = t
                    break

        return medians

    def predict_survival_curve(self, x: np.ndarray, max_t: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Predict full survival curve for a single sample."""
        import torch

        device = torch.device(self._device)
        self.net.eval()

        with torch.no_grad():
            x_tensor = torch.FloatTensor(x.reshape(1, -1)).to(device)
            risk_score = self.net(x_tensor).squeeze().cpu().item()

        exp_risk = math.exp(risk_score)

        if self.baseline_hazard is None:
            t_grid = np.arange(0, max_t)
            return t_grid, np.ones(max_t) * 0.5

        sorted_times = sorted(self.baseline_hazard.keys())
        t_grid = np.arange(0, max_t)
        survival = np.ones(max_t)

        cum_h = 0.0
        time_idx = 0
        for t in range(max_t):
            while time_idx < len(sorted_times) and sorted_times[time_idx] <= t:
                cum_h += self.baseline_hazard[sorted_times[time_idx]]
                time_idx += 1
            survival[t] = math.exp(-cum_h * exp_risk)

        return t_grid, survival

    def save(self, path: str):
        """Save model to disk."""
        import torch
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'net_state': self.net.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'baseline_hazard': self.baseline_hazard,
            'feature_names': self.feature_names,
            'device': self._device,
        }, path)
        print(f"  Saved Survival model to {path}")

    @classmethod
    def load(cls, path: str) -> 'SurvivalModel':
        """Load model from disk."""
        import torch
        data = torch.load(path, map_location='cpu', weights_only=False)
        model = cls(
            input_dim=data['input_dim'],
            hidden_dims=data['hidden_dims'],
        )
        model._build_network()
        model.net.load_state_dict(data['net_state'])
        model.baseline_hazard = data['baseline_hazard']
        model.feature_names = data['feature_names']
        model._device = data.get('device', 'cpu')
        model.net.eval()
        return model


# ---------------------------------------------------------------------------
# Architecture 3: Multi-TF Transformer with Cross-Attention
# ---------------------------------------------------------------------------

class MultiTFTransformer:
    """
    Novel architecture: per-TF encoders with cross-attention.

    Each timeframe gets its own feature encoder. Then cross-attention
    layers let information flow between TFs — the model learns WHICH
    TF combinations are predictive.

    Multi-task output:
    - Channel lifetime (regression head)
    - Break direction (3-class head)
    - Optimal action (3-class head)
    - Future returns (regression head)
    """

    def __init__(
        self,
        n_tf_features: int = len(PER_TF_FEATURES),
        n_cross_features: int = len(CROSS_TF_FEATURES),
        n_context_features: int = len(CONTEXT_FEATURES),
        n_temporal_features: int = len(TEMPORAL_FEATURES),
        n_corr_features: int = len(CORRELATION_FEATURES),
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        self.n_tf_features = n_tf_features
        self.n_cross_features = n_cross_features
        self.n_context_features = n_context_features
        self.n_temporal_features = n_temporal_features
        self.n_corr_features = n_corr_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.net = None
        self.feature_names = None
        self._device = 'cpu'

    def _build_network(self):
        """Build the multi-TF transformer network."""
        import torch
        import torch.nn as nn

        class TFEncoder(nn.Module):
            """Encode per-TF features into d_model space."""
            def __init__(self, input_dim, d_model, dropout):
                super().__init__()
                self.proj = nn.Sequential(
                    nn.Linear(input_dim, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, d_model),
                    nn.LayerNorm(d_model),
                )

            def forward(self, x):
                return self.proj(x)

        class CrossTFAttention(nn.Module):
            """Cross-attention between TF tokens."""
            def __init__(self, d_model, n_heads, dropout):
                super().__init__()
                self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
                self.norm1 = nn.LayerNorm(d_model)
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout),
                )
                self.norm2 = nn.LayerNorm(d_model)

            def forward(self, x):
                # Self-attention across TF tokens
                attn_out, _ = self.attn(x, x, x)
                x = self.norm1(x + attn_out)
                x = self.norm2(x + self.ffn(x))
                return x

        class MultiTFNet(nn.Module):
            def __init__(self, n_tf_features, n_cross, n_context, n_temporal, n_corr,
                         d_model, n_heads, n_layers, dropout):
                super().__init__()

                n_tfs = len(ML_TFS)

                # Per-TF encoders (shared weights)
                self.tf_encoder = TFEncoder(n_tf_features, d_model, dropout)

                # Learnable TF embeddings (so model knows WHICH TF)
                self.tf_embeddings = nn.Embedding(n_tfs, d_model)

                # Cross-TF attention layers
                self.cross_attn_layers = nn.ModuleList([
                    CrossTFAttention(d_model, n_heads, dropout)
                    for _ in range(n_layers)
                ])

                # Context encoder (global features: cross-TF, context, temporal, correlation)
                context_dim = n_cross + n_context + n_temporal + n_corr
                self.context_encoder = nn.Sequential(
                    nn.Linear(context_dim, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )

                # Fusion
                fusion_dim = d_model * (n_tfs + 1)  # TF tokens + context token
                self.fusion = nn.Sequential(
                    nn.Linear(fusion_dim, d_model * 2),
                    nn.LayerNorm(d_model * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )

                # Multi-task heads
                self.lifetime_head = nn.Sequential(
                    nn.Linear(d_model * 2, 64),
                    nn.GELU(),
                    nn.Linear(64, 1),
                    nn.Softplus(),  # Lifetime must be positive
                )

                self.break_dir_head = nn.Sequential(
                    nn.Linear(d_model * 2, 64),
                    nn.GELU(),
                    nn.Linear(64, 3),
                )

                self.action_head = nn.Sequential(
                    nn.Linear(d_model * 2, 64),
                    nn.GELU(),
                    nn.Linear(64, 3),
                )

                self.return_head = nn.Sequential(
                    nn.Linear(d_model * 2, 64),
                    nn.GELU(),
                    nn.Linear(64, 3),  # 5-bar, 20-bar, 60-bar
                )

            def forward(self, x):
                batch_size = x.shape[0]
                n_tfs = len(ML_TFS)
                n_tf_feat = self.tf_encoder.proj[0].in_features

                # Split input into per-TF features and global features
                tf_features = x[:, :n_tfs * n_tf_feat].view(batch_size, n_tfs, n_tf_feat)
                global_features = x[:, n_tfs * n_tf_feat:]

                # Encode each TF
                tf_tokens = []
                for i in range(n_tfs):
                    encoded = self.tf_encoder(tf_features[:, i, :])
                    tf_emb = self.tf_embeddings(
                        torch.full((batch_size,), i, dtype=torch.long, device=x.device)
                    )
                    tf_tokens.append(encoded + tf_emb)

                tf_seq = torch.stack(tf_tokens, dim=1)  # (B, n_tfs, d_model)

                # Cross-TF attention
                for layer in self.cross_attn_layers:
                    tf_seq = layer(tf_seq)

                # Context encoding
                ctx = self.context_encoder(global_features)  # (B, d_model)

                # Flatten TF tokens and concatenate context
                tf_flat = tf_seq.view(batch_size, -1)  # (B, n_tfs * d_model)
                fused = self.fusion(torch.cat([tf_flat, ctx], dim=1))

                # Multi-task outputs
                lifetime = self.lifetime_head(fused).squeeze(-1)
                break_dir = self.break_dir_head(fused)
                action = self.action_head(fused)
                returns = self.return_head(fused)

                return lifetime, break_dir, action, returns

        self.net = MultiTFNet(
            self.n_tf_features, self.n_cross_features,
            self.n_context_features, self.n_temporal_features, self.n_corr_features,
            self.d_model, self.n_heads, self.n_layers, self.dropout,
        )
        return self.net

    def train(
        self,
        X_train: np.ndarray,
        Y_train: Dict[str, np.ndarray],
        X_val: np.ndarray,
        Y_val: Dict[str, np.ndarray],
        feature_names: List[str],
        epochs: int = 300,
        lr: float = 1e-3,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """Train multi-TF transformer."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self.feature_names = feature_names
        self._build_network()

        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self._device = str(device)
        self.net.to(device)

        # Data
        train_X = torch.FloatTensor(X_train).to(device)
        val_X = torch.FloatTensor(X_val).to(device)

        # Normalize lifetime to [0, 1] for balanced multi-task loss
        self._lifetime_max = float(max(Y_train['channel_lifetime'].max(), 1.0))
        train_lifetime = torch.FloatTensor(Y_train['channel_lifetime'] / self._lifetime_max).to(device)
        train_break_dir = torch.LongTensor(Y_train['break_direction'].astype(int)).to(device)
        train_action = torch.LongTensor(Y_train['optimal_action'].astype(int)).to(device)
        train_returns = torch.FloatTensor(np.column_stack([
            Y_train['future_return_5'],
            Y_train['future_return_20'],
            Y_train['future_return_60'],
        ])).to(device)

        val_lifetime = torch.FloatTensor(Y_val['channel_lifetime'] / self._lifetime_max).to(device)
        val_break_dir = torch.LongTensor(Y_val['break_direction'].astype(int)).to(device)
        val_action = torch.LongTensor(Y_val['optimal_action'].astype(int)).to(device)
        val_returns = torch.FloatTensor(np.column_stack([
            Y_val['future_return_5'],
            Y_val['future_return_20'],
            Y_val['future_return_60'],
        ])).to(device)

        # Class weights for imbalanced labels
        break_counts = np.bincount(Y_train['break_direction'].astype(int), minlength=3)
        break_weights = 1.0 / (break_counts + 1)
        break_weights = break_weights / break_weights.sum() * 3
        break_weight_tensor = torch.FloatTensor(break_weights).to(device)

        action_counts = np.bincount(Y_train['optimal_action'].astype(int), minlength=3)
        action_weights = 1.0 / (action_counts + 1)
        action_weights = action_weights / action_weights.sum() * 3
        action_weight_tensor = torch.FloatTensor(action_weights).to(device)

        # Loss functions
        lifetime_loss_fn = nn.SmoothL1Loss()
        break_loss_fn = nn.CrossEntropyLoss(weight=break_weight_tensor)
        action_loss_fn = nn.CrossEntropyLoss(weight=action_weight_tensor)
        return_loss_fn = nn.SmoothL1Loss()

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=epochs,
            steps_per_epoch=max(1, len(X_train) // batch_size),
        )

        dataset = TensorDataset(train_X, train_lifetime, train_break_dir, train_action, train_returns)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        best_val_loss = float('inf')
        best_state = None
        patience = 40
        no_improve = 0

        n_params = sum(p.numel() for p in self.net.parameters())
        print(f"\n  Training Multi-TF Transformer ({n_params:,} params)")
        print(f"  Device: {device}")
        print(f"  Train: {len(X_train)}, Val: {len(X_val)}")
        print(f"  Break dir class counts: {break_counts}")
        print(f"  Action class counts: {action_counts}")

        for epoch in range(epochs):
            self.net.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch_X, batch_lt, batch_bd, batch_act, batch_ret in loader:
                pred_lt, pred_bd, pred_act, pred_ret = self.net(batch_X)

                loss_lt = lifetime_loss_fn(pred_lt, batch_lt)
                loss_bd = break_loss_fn(pred_bd, batch_bd)
                loss_act = action_loss_fn(pred_act, batch_act)
                loss_ret = return_loss_fn(pred_ret, batch_ret)

                # Multi-task loss (weighted)
                loss = 1.0 * loss_lt + 1.0 * loss_bd + 1.0 * loss_act + 2.0 * loss_ret

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Validation
            self.net.eval()
            with torch.no_grad():
                v_lt, v_bd, v_act, v_ret = self.net(val_X)
                v_loss_lt = lifetime_loss_fn(v_lt, val_lifetime)
                v_loss_bd = break_loss_fn(v_bd, val_break_dir)
                v_loss_act = action_loss_fn(v_act, val_action)
                v_loss_ret = return_loss_fn(v_ret, val_returns)
                val_loss = 1.0 * v_loss_lt + 1.0 * v_loss_bd + 1.0 * v_loss_act + 2.0 * v_loss_ret
                val_loss_val = val_loss.item()

            if val_loss_val < best_val_loss:
                best_val_loss = val_loss_val
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 20 == 0:
                lt_mae = torch.mean(torch.abs(v_lt - val_lifetime)).item() * self._lifetime_max
                bd_acc = (v_bd.argmax(1) == val_break_dir).float().mean().item()
                act_acc = (v_act.argmax(1) == val_action).float().mean().item()
                ret_mae = torch.mean(torch.abs(v_ret - val_returns)).item()
                print(f"    Epoch {epoch+1}/{epochs} | Loss: {val_loss_val:.4f} | "
                      f"LT_MAE: {lt_mae:.1f}bars | BD_Acc: {bd_acc:.1%} | "
                      f"Act_Acc: {act_acc:.1%} | Ret_MAE: {ret_mae:.5f}")

            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

        # Restore best
        if best_state:
            self.net.load_state_dict(best_state)

        # Final evaluation
        self.net.eval()
        with torch.no_grad():
            v_lt, v_bd, v_act, v_ret = self.net(val_X)

        metrics = {
            'lifetime_mae': torch.mean(torch.abs(v_lt - val_lifetime)).item() * self._lifetime_max,
            'break_dir_accuracy': (v_bd.argmax(1) == val_break_dir).float().mean().item(),
            'action_accuracy': (v_act.argmax(1) == val_action).float().mean().item(),
            'return_5_dir_acc': (torch.sign(v_ret[:, 0]) == torch.sign(val_returns[:, 0])).float().mean().item(),
            'return_20_dir_acc': (torch.sign(v_ret[:, 1]) == torch.sign(val_returns[:, 1])).float().mean().item(),
            'return_60_dir_acc': (torch.sign(v_ret[:, 2]) == torch.sign(val_returns[:, 2])).float().mean().item(),
            'best_val_loss': best_val_loss,
        }

        print(f"\n  Multi-TF Transformer Results:")
        for k, v in metrics.items():
            if 'acc' in k:
                print(f"    {k}: {v:.1%}")
            else:
                print(f"    {k}: {v:.4f}")

        return metrics

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference."""
        import torch

        device = torch.device(self._device)
        self.net.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            lt, bd, act, ret = self.net(X_tensor)

        return {
            'lifetime': lt.cpu().numpy() * getattr(self, '_lifetime_max', 200.0),
            'break_dir_probs': torch.softmax(bd, dim=1).cpu().numpy(),
            'break_dir': bd.argmax(1).cpu().numpy(),
            'action_probs': torch.softmax(act, dim=1).cpu().numpy(),
            'action': act.argmax(1).cpu().numpy(),
            'future_return_5': ret[:, 0].cpu().numpy(),
            'future_return_20': ret[:, 1].cpu().numpy(),
            'future_return_60': ret[:, 2].cpu().numpy(),
        }

    def save(self, path: str):
        """Save model to disk."""
        import torch
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'net_state': self.net.state_dict(),
            'config': {
                'n_tf_features': self.n_tf_features,
                'n_cross_features': self.n_cross_features,
                'n_context_features': self.n_context_features,
                'n_temporal_features': self.n_temporal_features,
                'n_corr_features': self.n_corr_features,
                'd_model': self.d_model,
                'n_heads': self.n_heads,
                'n_layers': self.n_layers,
                'dropout': self.dropout,
            },
            'feature_names': self.feature_names,
            'device': self._device,
            'lifetime_max': getattr(self, '_lifetime_max', 200.0),
        }, path)
        print(f"  Saved Transformer model to {path}")

    @classmethod
    def load(cls, path: str) -> 'MultiTFTransformer':
        """Load model from disk."""
        import torch
        data = torch.load(path, map_location='cpu', weights_only=False)
        config = data['config']
        model = cls(**config)
        model._build_network()
        model.net.load_state_dict(data['net_state'])
        model.feature_names = data['feature_names']
        model._device = data.get('device', 'cpu')
        model._lifetime_max = data.get('lifetime_max', 200.0)
        model.net.eval()
        return model


# ---------------------------------------------------------------------------
# Architecture 4: Trade Quality Scorer
# ---------------------------------------------------------------------------

class TradeQualityScorer:
    """
    Binary win/loss predictor trained on actual backtest outcomes.

    Instead of predicting abstract labels (break direction, optimal action),
    this model directly predicts whether a trade taken at the current bar
    will be profitable, using physics features + signal metadata.

    Training approach (bootstrapped learning):
    1. Run physics-only backtest to generate trades with known outcomes
    2. For each trade, extract ML features at entry_bar + signal metadata
    3. Train GBT to predict: win probability, expected PnL %, exit type

    Additional features beyond standard ML features:
    - signal_confidence: physics confidence score
    - signal_type: bounce vs break (encoded)
    - signal_direction: BUY vs SELL (encoded)
    - hour_of_day, day_of_week (from entry time)
    """

    # Extra features on top of the standard feature vector
    SIGNAL_FEATURES = [
        'signal_confidence',   # Physics confidence score
        'signal_type_bounce',  # 1.0 for bounce, 0.0 for break
        'signal_direction_buy',  # 1.0 for BUY, 0.0 for SELL
        'stop_pct',            # Stop loss distance %
        'tp_pct',              # Take profit distance %
        'position_size_ratio', # Size relative to base (shows conviction)
    ]

    def __init__(self):
        self.models = {}
        self.feature_names = None
        self.feature_importance = {}

    def generate_training_data(
        self,
        X_base: np.ndarray,
        Y_base: Dict[str, np.ndarray],
        base_feature_names: List[str],
        verbose: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
        """
        Generate trade quality labels from existing training data.

        Uses forward-looking returns to create quality labels:
        - buy_win: would a long position at this bar be profitable?
        - sell_win: would a short position at this bar be profitable?
        - best_action: which action has the best risk-adjusted return?
        - quality_score: continuous quality metric

        This uses the same 1400+ samples from the standard training pipeline
        rather than the sparse 176 trades from a backtest.
        """
        all_feature_names = base_feature_names + self.SIGNAL_FEATURES
        num_base = len(base_feature_names)
        num_signal = len(self.SIGNAL_FEATURES)
        num_total = len(all_feature_names)

        # We create 2x samples: one for "what if we BUY" and one for "what if we SELL"
        # at each bar, with different signal metadata
        X_list = []
        Y_win = []
        Y_pnl = []
        Y_quality = []

        for i in range(len(X_base)):
            ret_5 = Y_base['future_return_5'][i]
            ret_20 = Y_base['future_return_20'][i]
            ret_60 = Y_base['future_return_60'][i]

            # Compute MFE/MAE proxy from the 3 return horizons
            # For a BUY: positive returns are favorable, negative are adverse
            buy_returns = [ret_5, ret_20, ret_60]
            buy_mfe = max(buy_returns)  # Best upside seen
            buy_mae = abs(min(buy_returns))  # Worst downside

            # For a SELL: negative returns are favorable, positive are adverse
            sell_mfe = abs(min(buy_returns))  # Best downside for short
            sell_mae = max(buy_returns)  # Worst upside for short

            # BUY quality: weighted return with MFE/MAE ratio
            buy_win = 1.0 if ret_20 > 0.001 and (buy_mfe > buy_mae * 1.2 or ret_5 > 0) else 0.0
            sell_win = 1.0 if ret_20 < -0.001 and (sell_mfe > sell_mae * 1.2 or ret_5 < 0) else 0.0

            # Quality score: normalized expected value
            buy_quality = ret_20 * 100  # In % terms
            sell_quality = -ret_20 * 100

            # BUY sample
            buy_vec = np.zeros(num_total, dtype=np.float32)
            buy_vec[:num_base] = X_base[i]
            buy_vec[num_base + 0] = 0.6  # Default confidence
            buy_vec[num_base + 1] = 0.5  # Mix of bounce/break
            buy_vec[num_base + 2] = 1.0  # BUY direction
            buy_vec[num_base + 3] = 0.005  # Default stop
            buy_vec[num_base + 4] = 0.012  # Default TP
            buy_vec[num_base + 5] = 1.0  # Default size ratio

            X_list.append(buy_vec)
            Y_win.append(buy_win)
            Y_pnl.append(ret_20 * 100)
            Y_quality.append(buy_quality)

            # SELL sample
            sell_vec = np.zeros(num_total, dtype=np.float32)
            sell_vec[:num_base] = X_base[i]
            sell_vec[num_base + 0] = 0.6
            sell_vec[num_base + 1] = 0.5
            sell_vec[num_base + 2] = 0.0  # SELL direction
            sell_vec[num_base + 3] = 0.005
            sell_vec[num_base + 4] = 0.012
            sell_vec[num_base + 5] = 1.0

            X_list.append(sell_vec)
            Y_win.append(sell_win)
            Y_pnl.append(-ret_20 * 100)
            Y_quality.append(sell_quality)

        X = np.array(X_list, dtype=np.float32)
        Y = {
            'win': np.array(Y_win, dtype=np.float32),
            'pnl_pct': np.array(Y_pnl, dtype=np.float32),
            'quality_score': np.array(Y_quality, dtype=np.float32),
        }

        if verbose:
            total = len(X)
            win_rate = Y['win'].mean()
            print(f"  Generated {total} samples ({total//2} bars × 2 directions)")
            print(f"  Win rate: {win_rate:.1%}")
            print(f"  Quality score: mean={Y['quality_score'].mean():.3f}, std={Y['quality_score'].std():.3f}")

        return X, Y, all_feature_names

    def train(
        self,
        X_train: np.ndarray,
        Y_train: Dict[str, np.ndarray],
        X_val: np.ndarray,
        Y_val: Dict[str, np.ndarray],
        feature_names: List[str],
    ) -> Dict[str, float]:
        """Train trade quality scorer models."""
        try:
            import lightgbm as lgb
            use_lgb = True
        except ImportError:
            use_lgb = False
            print("  WARNING: LightGBM not available, falling back to sklearn")

        self.feature_names = feature_names
        metrics = {}

        # --- Win/Loss binary classifier ---
        print("\n  Training: win/loss classifier...")
        if use_lgb:
            train_ds = lgb.Dataset(X_train, label=Y_train['win'],
                                   feature_name=feature_names)
            val_ds = lgb.Dataset(X_val, label=Y_val['win'],
                                 feature_name=feature_names, reference=train_ds)

            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'min_child_samples': 5,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'is_unbalanced': True,
            }

            model = lgb.train(
                params, train_ds, num_boost_round=500,
                valid_sets=[val_ds],
                callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(50)],
            )
            self.models['win'] = model

            # Feature importance
            importance = sorted(
                zip(feature_names, model.feature_importance(importance_type='gain')),
                key=lambda x: x[1], reverse=True,
            )
            self.feature_importance['win'] = importance

            val_pred = model.predict(X_val)
            val_pred_binary = (val_pred > 0.5).astype(int)
            val_acc = np.mean(val_pred_binary == Y_val['win'].astype(int))
            metrics['win_accuracy'] = float(val_acc)

            # AUC
            from sklearn.metrics import roc_auc_score
            try:
                val_auc = roc_auc_score(Y_val['win'], val_pred)
                metrics['win_auc'] = float(val_auc)
            except Exception:
                pass

            print(f"    Win accuracy: {val_acc:.1%}")
            if 'win_auc' in metrics:
                print(f"    Win AUC: {metrics['win_auc']:.3f}")
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            clf = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
            )
            clf.fit(X_train, Y_train['win'])
            self.models['win'] = clf
            val_acc = clf.score(X_val, Y_val['win'])
            metrics['win_accuracy'] = float(val_acc)
            print(f"    Win accuracy: {val_acc:.1%}")

        # --- PnL regression ---
        print("\n  Training: PnL % predictor...")
        if use_lgb:
            train_ds = lgb.Dataset(X_train, label=Y_train['pnl_pct'],
                                   feature_name=feature_names)
            val_ds = lgb.Dataset(X_val, label=Y_val['pnl_pct'],
                                 feature_name=feature_names, reference=train_ds)

            params = {
                'objective': 'regression',
                'metric': 'mae',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'min_child_samples': 5,
                'feature_fraction': 0.8,
                'verbose': -1,
            }

            model = lgb.train(
                params, train_ds, num_boost_round=500,
                valid_sets=[val_ds],
                callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(50)],
            )
            self.models['pnl_pct'] = model

            val_pred = model.predict(X_val)
            mae = np.mean(np.abs(val_pred - Y_val['pnl_pct']))
            dir_acc = np.mean(np.sign(val_pred) == np.sign(Y_val['pnl_pct']))
            metrics['pnl_mae'] = float(mae)
            metrics['pnl_dir_acc'] = float(dir_acc)
            print(f"    PnL MAE: {mae:.4f}")
            print(f"    PnL direction accuracy: {dir_acc:.1%}")

        # --- Quality score regression ---
        if 'quality_score' in Y_train:
            print("\n  Training: quality score predictor...")
            if use_lgb:
                train_ds = lgb.Dataset(X_train, label=Y_train['quality_score'],
                                       feature_name=feature_names)
                val_ds = lgb.Dataset(X_val, label=Y_val['quality_score'],
                                     feature_name=feature_names, reference=train_ds)

                params = {
                    'objective': 'regression',
                    'metric': 'mae',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'min_child_samples': 5,
                    'feature_fraction': 0.8,
                    'verbose': -1,
                }

                model = lgb.train(
                    params, train_ds, num_boost_round=500,
                    valid_sets=[val_ds],
                    callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(50)],
                )
                self.models['quality_score'] = model

                val_pred = model.predict(X_val)
                q_mae = np.mean(np.abs(val_pred - Y_val['quality_score']))
                q_dir_acc = np.mean(np.sign(val_pred) == np.sign(Y_val['quality_score']))
                metrics['quality_mae'] = float(q_mae)
                metrics['quality_dir_acc'] = float(q_dir_acc)
                print(f"    Quality score MAE: {q_mae:.4f}")
                print(f"    Quality direction accuracy: {q_dir_acc:.1%}")

        return metrics

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict trade quality for feature array(s)."""
        results = {}

        if 'win' in self.models:
            model = self.models['win']
            if hasattr(model, 'predict'):
                raw = model.predict(X)
                # LightGBM binary returns probabilities directly
                if isinstance(raw, np.ndarray) and raw.ndim == 1:
                    if raw.max() <= 1.0 and raw.min() >= 0.0:
                        results['win_prob'] = raw
                        results['win'] = (raw > 0.5).astype(int)
                    else:
                        results['win'] = raw.astype(int)
                        results['win_prob'] = raw.astype(float)
                else:
                    results['win'] = np.array(raw).flatten()
                    results['win_prob'] = results['win'].astype(float)

        if 'pnl_pct' in self.models:
            results['pnl_pct'] = self.models['pnl_pct'].predict(X)

        if 'quality_score' in self.models:
            results['quality_score'] = self.models['quality_score'].predict(X)

        return results

    def save(self, path: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance,
            }, f)
        print(f"  Saved TradeQualityScorer to {path}")

    @classmethod
    def load(cls, path: str) -> 'TradeQualityScorer':
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.models = data['models']
        model.feature_names = data['feature_names']
        model.feature_importance = data.get('feature_importance', {})
        return model


# ---------------------------------------------------------------------------
# Architecture 5: Stacked Ensemble (Meta-Learner)
# ---------------------------------------------------------------------------

class EnsembleModel:
    """
    Stacked ensemble that combines predictions from GBT, Transformer,
    Survival, and Quality Scorer using a GBT meta-learner.

    Meta-features per sample:
    - GBT: action (3 probs), break_dir (3 probs), lifetime, future_return_5/20/60
    - Transformer: action (3 probs), break_dir (3 probs), lifetime, returns
    - Survival: predicted median lifetime, hazard score
    - Quality Scorer: win_prob, quality_score
    - Agreement features: how many models agree on direction, spread in lifetime predictions

    The meta-learner is a lightweight GBT that learns when each model
    is trustworthy and how to weight disagreements.
    """

    META_FEATURES = [
        # GBT outputs
        'gbt_action_hold', 'gbt_action_buy', 'gbt_action_sell',
        'gbt_break_survive', 'gbt_break_up', 'gbt_break_down',
        'gbt_lifetime', 'gbt_return5', 'gbt_return20', 'gbt_return60',
        # Transformer outputs
        'trans_action_hold', 'trans_action_buy', 'trans_action_sell',
        'trans_break_survive', 'trans_break_up', 'trans_break_down',
        'trans_lifetime', 'trans_return5', 'trans_return20', 'trans_return60',
        # Survival outputs
        'surv_median_lifetime', 'surv_hazard',
        # Quality Scorer outputs
        'qs_win_prob_buy', 'qs_win_prob_sell', 'qs_quality_buy', 'qs_quality_sell',
        # Agreement / disagreement features
        'action_consensus',    # Fraction of models agreeing on action
        'lifetime_spread',     # Max - min predicted lifetime
        'lifetime_mean',       # Mean predicted lifetime
        'direction_agreement', # GBT and Transformer agree on break direction
    ]

    def __init__(self):
        self.models = {}
        self.base_models = {}
        self.feature_names = None

    def extract_meta_features(
        self,
        X: np.ndarray,
        gbt: Optional['GBTModel'] = None,
        transformer: Optional['MultiTFTransformer'] = None,
        survival: Optional['SurvivalModel'] = None,
        quality: Optional['TradeQualityScorer'] = None,
        base_feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Extract meta-features from base model predictions."""
        n = len(X)
        meta_X = np.zeros((n, len(self.META_FEATURES)), dtype=np.float32)

        # GBT predictions
        if gbt is not None:
            gbt_pred = gbt.predict(X)
            if 'action_probs' in gbt_pred:
                meta_X[:, 0:3] = gbt_pred['action_probs']
            elif 'action' in gbt_pred:
                for i in range(n):
                    meta_X[i, int(gbt_pred['action'][i])] = 1.0
            if 'break_dir_probs' in gbt_pred:
                meta_X[:, 3:6] = gbt_pred['break_dir_probs']
            elif 'break_dir' in gbt_pred:
                for i in range(n):
                    meta_X[i, 3 + int(gbt_pred['break_dir'][i])] = 1.0
            if 'lifetime' in gbt_pred:
                meta_X[:, 6] = gbt_pred['lifetime']
            for j, key in enumerate(['future_return_5', 'future_return_20', 'future_return_60']):
                if key in gbt_pred:
                    meta_X[:, 7 + j] = gbt_pred[key]

        # Transformer predictions
        if transformer is not None:
            try:
                trans_pred = transformer.predict(X)
                if 'action_probs' in trans_pred:
                    meta_X[:, 10:13] = trans_pred['action_probs']
                if 'break_dir_probs' in trans_pred:
                    meta_X[:, 13:16] = trans_pred['break_dir_probs']
                if 'lifetime' in trans_pred:
                    meta_X[:, 16] = trans_pred['lifetime']
                for j, key in enumerate(['future_return_5', 'future_return_20', 'future_return_60']):
                    if key in trans_pred:
                        meta_X[:, 17 + j] = trans_pred[key]
            except Exception:
                pass

        # Survival predictions
        if survival is not None:
            try:
                medians = survival.predict_median_survival(X)
                meta_X[:, 20] = medians

                import torch
                survival.net.eval()
                device = torch.device(survival._device)
                with torch.no_grad():
                    risk = survival.net(torch.FloatTensor(X).to(device))
                    meta_X[:, 21] = risk.squeeze(-1).cpu().numpy()
            except Exception:
                pass

        # Quality Scorer predictions
        if quality is not None:
            try:
                # Need extended features for quality scorer
                qs_features = quality.feature_names or []
                n_base = len(base_feature_names) if base_feature_names else X.shape[1]
                n_extra = len(TradeQualityScorer.SIGNAL_FEATURES)

                # BUY prediction
                qs_buy = np.zeros((n, n_base + n_extra), dtype=np.float32)
                qs_buy[:, :min(n_base, X.shape[1])] = X[:, :min(n_base, X.shape[1])]
                qs_buy[:, n_base + 2] = 1.0  # BUY direction
                qs_buy_pred = quality.predict(qs_buy)
                if 'win_prob' in qs_buy_pred:
                    meta_X[:, 22] = qs_buy_pred['win_prob']
                if 'quality_score' in qs_buy_pred:
                    meta_X[:, 24] = qs_buy_pred['quality_score']

                # SELL prediction
                qs_sell = np.zeros((n, n_base + n_extra), dtype=np.float32)
                qs_sell[:, :min(n_base, X.shape[1])] = X[:, :min(n_base, X.shape[1])]
                qs_sell[:, n_base + 2] = 0.0  # SELL direction
                qs_sell_pred = quality.predict(qs_sell)
                if 'win_prob' in qs_sell_pred:
                    meta_X[:, 23] = qs_sell_pred['win_prob']
                if 'quality_score' in qs_sell_pred:
                    meta_X[:, 25] = qs_sell_pred['quality_score']
            except Exception:
                pass

        # Agreement features
        # Action consensus: how many models agree on the best action
        gbt_action = meta_X[:, 0:3].argmax(axis=1)
        trans_action = meta_X[:, 10:13].argmax(axis=1)
        actions_agree = (gbt_action == trans_action).astype(float)
        meta_X[:, 26] = actions_agree

        # Lifetime spread
        lifetimes = np.column_stack([
            meta_X[:, 6],   # GBT lifetime
            meta_X[:, 16],  # Transformer lifetime
            meta_X[:, 20],  # Survival median lifetime
        ])
        valid = lifetimes > 0
        for i in range(n):
            valid_lt = lifetimes[i, valid[i]]
            if len(valid_lt) > 1:
                meta_X[i, 27] = np.max(valid_lt) - np.min(valid_lt)
                meta_X[i, 28] = np.mean(valid_lt)
            elif len(valid_lt) == 1:
                meta_X[i, 28] = valid_lt[0]

        # Direction agreement
        gbt_dir = meta_X[:, 3:6].argmax(axis=1)
        trans_dir = meta_X[:, 13:16].argmax(axis=1)
        meta_X[:, 29] = (gbt_dir == trans_dir).astype(float)

        return meta_X

    def train(
        self,
        X_train: np.ndarray, Y_train: Dict[str, np.ndarray],
        X_val: np.ndarray, Y_val: Dict[str, np.ndarray],
        gbt=None, transformer=None, survival=None, quality=None,
        feature_names: List[str] = None,
    ) -> Dict[str, float]:
        """Train ensemble meta-learner on stacked base model predictions."""
        try:
            import lightgbm as lgb
            use_lgb = True
        except ImportError:
            use_lgb = False

        self.feature_names = self.META_FEATURES

        print("\n  Extracting meta-features from base models...")
        meta_train = self.extract_meta_features(
            X_train, gbt, transformer, survival, quality, feature_names,
        )
        meta_val = self.extract_meta_features(
            X_val, gbt, transformer, survival, quality, feature_names,
        )

        print(f"  Meta features shape: {meta_train.shape}")
        metrics = {}

        # Train action meta-learner
        print("\n  Training: ensemble action classifier...")
        if use_lgb:
            train_ds = lgb.Dataset(meta_train, label=Y_train['optimal_action'],
                                   feature_name=self.META_FEATURES)
            val_ds = lgb.Dataset(meta_val, label=Y_val['optimal_action'],
                                 feature_name=self.META_FEATURES, reference=train_ds)

            params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'num_leaves': 15,
                'learning_rate': 0.05,
                'min_child_samples': 5,
                'feature_fraction': 0.7,
                'verbose': -1,
            }

            model = lgb.train(
                params, train_ds, num_boost_round=300,
                valid_sets=[val_ds],
                callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(50)],
            )
            self.models['action'] = model

            val_pred = model.predict(meta_val)
            val_pred_class = np.argmax(val_pred, axis=1)
            val_acc = np.mean(val_pred_class == Y_val['optimal_action'].astype(int))
            metrics['action_accuracy'] = float(val_acc)
            print(f"    Action accuracy: {val_acc:.1%}")

            # Feature importance
            importance = sorted(
                zip(self.META_FEATURES, model.feature_importance(importance_type='gain')),
                key=lambda x: x[1], reverse=True,
            )
            print(f"    Top meta-features:")
            for name, imp in importance[:10]:
                if imp > 0:
                    print(f"      {name}: {imp:.0f}")

        # Train break direction meta-learner
        print("\n  Training: ensemble break direction classifier...")
        if use_lgb:
            train_ds = lgb.Dataset(meta_train, label=Y_train['break_direction'],
                                   feature_name=self.META_FEATURES)
            val_ds = lgb.Dataset(meta_val, label=Y_val['break_direction'],
                                 feature_name=self.META_FEATURES, reference=train_ds)

            params['num_class'] = 3
            model = lgb.train(
                params, train_ds, num_boost_round=300,
                valid_sets=[val_ds],
                callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(50)],
            )
            self.models['break_dir'] = model

            val_pred = model.predict(meta_val)
            val_pred_class = np.argmax(val_pred, axis=1)
            val_acc = np.mean(val_pred_class == Y_val['break_direction'].astype(int))
            metrics['break_dir_accuracy'] = float(val_acc)
            print(f"    Break dir accuracy: {val_acc:.1%}")

        # Train lifetime meta-learner
        print("\n  Training: ensemble lifetime regressor...")
        if use_lgb:
            train_ds = lgb.Dataset(meta_train, label=Y_train['channel_lifetime'],
                                   feature_name=self.META_FEATURES)
            val_ds = lgb.Dataset(meta_val, label=Y_val['channel_lifetime'],
                                 feature_name=self.META_FEATURES, reference=train_ds)

            params = {
                'objective': 'regression',
                'metric': 'mae',
                'num_leaves': 15,
                'learning_rate': 0.05,
                'min_child_samples': 5,
                'feature_fraction': 0.7,
                'verbose': -1,
            }

            model = lgb.train(
                params, train_ds, num_boost_round=300,
                valid_sets=[val_ds],
                callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(50)],
            )
            self.models['lifetime'] = model

            val_pred = model.predict(meta_val)
            mae = np.mean(np.abs(val_pred - Y_val['channel_lifetime']))
            metrics['lifetime_mae'] = float(mae)
            print(f"    Lifetime MAE: {mae:.1f} bars")

        return metrics

    def predict(self, X: np.ndarray,
                gbt=None, transformer=None, survival=None, quality=None,
                feature_names=None) -> Dict[str, np.ndarray]:
        """Run ensemble prediction."""
        meta_X = self.extract_meta_features(
            X, gbt, transformer, survival, quality, feature_names,
        )

        results = {}

        if 'action' in self.models:
            probs = self.models['action'].predict(meta_X)
            results['action_probs'] = probs
            results['action'] = np.argmax(probs, axis=1)

        if 'break_dir' in self.models:
            probs = self.models['break_dir'].predict(meta_X)
            results['break_dir_probs'] = probs
            results['break_dir'] = np.argmax(probs, axis=1)

        if 'lifetime' in self.models:
            results['lifetime'] = self.models['lifetime'].predict(meta_X)

        return results

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'feature_names': self.feature_names,
            }, f)
        print(f"  Saved EnsembleModel to {path}")

    @classmethod
    def load(cls, path: str) -> 'EnsembleModel':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.models = data['models']
        model.feature_names = data.get('feature_names', cls.META_FEATURES)
        return model


# ---------------------------------------------------------------------------
# Architecture 6: Regime-Conditional Model (Mixture of Experts)
# ---------------------------------------------------------------------------

class RegimeConditionalModel:
    """
    Regime-augmented model: classifies market regime, then uses regime
    probabilities as extra features in a single unified GBT.

    Regimes (derived from physics features):
    0. TRENDING_UP: mostly bullish TFs
    1. TRENDING_DOWN: mostly bearish TFs
    2. VOLATILE: high entropy or high break probability (p75+)
    3. QUIET: everything else (mixed, ranging, squeezing)

    Instead of per-regime expert models (which suffer from small sample sizes),
    this uses regime probabilities as 4 additional features in the main GBT.
    This lets the model learn regime-conditional patterns while training on
    ALL samples.
    """

    REGIME_NAMES = ['trending_up', 'trending_down', 'volatile', 'quiet']

    def __init__(self):
        self.regime_classifier = None
        self.action_model = None
        self.break_dir_model = None
        self.feature_names = None
        self.augmented_feature_names = None
        self.feature_importance = {}

    @staticmethod
    def classify_regime(X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """
        Classify market regime from features using physics rules.

        Returns array of regime IDs (0-3) for each sample.
        """
        regimes = np.zeros(len(X), dtype=np.int32)

        # Get feature indices
        name_to_idx = {name: i for i, name in enumerate(feature_names)}

        bull_idx = name_to_idx.get('bullish_fraction')
        bear_idx = name_to_idx.get('bearish_fraction')
        entropy_idx = name_to_idx.get('avg_entropy')
        bp_max_idx = name_to_idx.get('break_prob_max')
        health_max_idx = name_to_idx.get('health_max')
        squeeze_idx = name_to_idx.get('squeeze_any')
        health_min_idx = name_to_idx.get('health_min')

        # Compute adaptive thresholds from data distribution using percentiles
        all_ent = X[:, entropy_idx] if entropy_idx is not None else np.full(len(X), 0.5)
        all_bp = X[:, bp_max_idx] if bp_max_idx is not None else np.full(len(X), 0.3)
        # Top 25% = volatile territory
        ent_p75 = np.percentile(all_ent[all_ent > 0], 75) if np.any(all_ent > 0) else 0.7
        bp_p75 = np.percentile(all_bp[all_bp > 0], 75) if np.any(all_bp > 0) else 0.5

        for i in range(len(X)):
            bull_frac = X[i, bull_idx] if bull_idx is not None else 0
            bear_frac = X[i, bear_idx] if bear_idx is not None else 0
            avg_ent = X[i, entropy_idx] if entropy_idx is not None else 0.5
            bp_max = X[i, bp_max_idx] if bp_max_idx is not None else 0
            h_max = X[i, health_max_idx] if health_max_idx is not None else 0.5
            h_min = X[i, health_min_idx] if health_min_idx is not None else 0.5
            squeeze = X[i, squeeze_idx] if squeeze_idx is not None else 0

            high_entropy = avg_ent > ent_p75
            high_bp = bp_max > bp_p75

            # Volatile first (priority check): high entropy OR high break prob
            if high_entropy or high_bp:
                regimes[i] = 2  # VOLATILE
            # Trending up: majority bullish
            elif bull_frac > 0.4:
                regimes[i] = 0  # TRENDING_UP
            # Trending down: majority bearish
            elif bear_frac > 0.4:
                regimes[i] = 1  # TRENDING_DOWN
            # Quiet: everything else (mixed, ranging)
            else:
                regimes[i] = 3  # QUIET

        return regimes

    def _augment_features(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Add regime probabilities as extra features to X."""
        # Get rule-based regime labels for classifier training
        regimes = self.classify_regime(X, feature_names)

        if self.regime_classifier is not None:
            # Use learned soft probabilities
            regime_probs = self.regime_classifier.predict(X)
        else:
            # One-hot encode rule-based regimes
            regime_probs = np.zeros((len(X), 4))
            for i, r in enumerate(regimes):
                regime_probs[i, r] = 1.0

        # Also add regime interaction features:
        # regime × key physics features for conditional patterns
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        bp_max_idx = name_to_idx.get('break_prob_max')
        ent_idx = name_to_idx.get('avg_entropy')

        interactions = np.zeros((len(X), 8))  # 4 regimes × 2 key features
        bp = X[:, bp_max_idx] if bp_max_idx is not None else np.zeros(len(X))
        ent = X[:, ent_idx] if ent_idx is not None else np.zeros(len(X))
        for r in range(4):
            interactions[:, r * 2] = regime_probs[:, r] * bp
            interactions[:, r * 2 + 1] = regime_probs[:, r] * ent

        return np.hstack([X, regime_probs, interactions])

    def train(
        self,
        X_train: np.ndarray, Y_train: Dict[str, np.ndarray],
        X_val: np.ndarray, Y_val: Dict[str, np.ndarray],
        feature_names: List[str],
    ) -> Dict[str, float]:
        """Train regime-augmented model (single GBT with regime features)."""
        import lightgbm as lgb

        self.feature_names = feature_names
        metrics = {}

        # Classify regimes for distribution reporting
        train_regimes = self.classify_regime(X_train, feature_names)
        val_regimes = self.classify_regime(X_val, feature_names)

        print(f"\n  Regime distribution (train):")
        for r in range(4):
            count = np.sum(train_regimes == r)
            pct = count / len(train_regimes) * 100
            print(f"    {self.REGIME_NAMES[r]}: {count} ({pct:.0f}%)")

        # Step 1: Train regime classifier
        print("\n  Training regime classifier (GBT)...")
        train_ds = lgb.Dataset(X_train, label=train_regimes,
                               feature_name=feature_names)
        val_ds = lgb.Dataset(X_val, label=val_regimes,
                             feature_name=feature_names, reference=train_ds)

        params = {
            'objective': 'multiclass',
            'num_class': 4,
            'metric': 'multi_logloss',
            'num_leaves': 20,
            'learning_rate': 0.05,
            'min_child_samples': 10,
            'feature_fraction': 0.8,
            'verbose': -1,
        }
        self.regime_classifier = lgb.train(
            params, train_ds, num_boost_round=300,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(50)],
        )

        val_pred = self.regime_classifier.predict(X_val)
        val_acc = np.mean(np.argmax(val_pred, axis=1) == val_regimes)
        metrics['regime_accuracy'] = float(val_acc)
        print(f"    Regime classification accuracy: {val_acc:.1%}")

        # Step 2: Augment features with regime probabilities
        aug_feature_names = (
            list(feature_names)
            + [f'regime_prob_{n}' for n in self.REGIME_NAMES]
            + [f'regime_{n}_x_bp_max' for n in self.REGIME_NAMES]
            + [f'regime_{n}_x_entropy' for n in self.REGIME_NAMES]
        )
        self.augmented_feature_names = aug_feature_names

        X_train_aug = self._augment_features(X_train, feature_names)
        X_val_aug = self._augment_features(X_val, feature_names)
        print(f"\n  Augmented features: {X_train.shape[1]} → {X_train_aug.shape[1]}")

        # Step 3: Train action model on ALL samples with regime-augmented features
        print("\n  Training regime-augmented action model...")
        train_ds = lgb.Dataset(X_train_aug, label=Y_train['optimal_action'],
                               feature_name=aug_feature_names)
        val_ds = lgb.Dataset(X_val_aug, label=Y_val['optimal_action'],
                             feature_name=aug_feature_names, reference=train_ds)

        action_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'num_leaves': 31,
            'learning_rate': 0.03,
            'min_child_samples': 10,
            'feature_fraction': 0.8,
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'verbose': -1,
        }
        self.action_model = lgb.train(
            action_params, train_ds, num_boost_round=500,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)],
        )

        val_pred = self.action_model.predict(X_val_aug)
        val_pred_class = np.argmax(val_pred, axis=1)
        acc = np.mean(val_pred_class == Y_val['optimal_action'].astype(int))
        metrics['action_accuracy'] = float(acc)
        print(f"    Action accuracy: {acc:.1%}")

        # Per-regime action accuracy
        for r in range(4):
            mask = val_regimes == r
            if np.sum(mask) > 0:
                r_acc = np.mean(val_pred_class[mask] == Y_val['optimal_action'][mask].astype(int))
                metrics[f'{self.REGIME_NAMES[r]}_action_acc'] = float(r_acc)
                print(f"    {self.REGIME_NAMES[r]} action acc: {r_acc:.1%} (n={np.sum(mask)})")

        # Feature importance for action
        importance = self.action_model.feature_importance(importance_type='gain')
        sorted_imp = sorted(zip(aug_feature_names, importance), key=lambda x: -x[1])
        self.feature_importance['action'] = sorted_imp

        # Step 4: Train break direction model
        print("\n  Training regime-augmented break direction model...")
        train_ds = lgb.Dataset(X_train_aug, label=Y_train['break_direction'],
                               feature_name=aug_feature_names)
        val_ds = lgb.Dataset(X_val_aug, label=Y_val['break_direction'],
                             feature_name=aug_feature_names, reference=train_ds)

        self.break_dir_model = lgb.train(
            action_params, train_ds, num_boost_round=500,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)],
        )

        val_pred = self.break_dir_model.predict(X_val_aug)
        val_pred_class = np.argmax(val_pred, axis=1)
        acc = np.mean(val_pred_class == Y_val['break_direction'].astype(int))
        metrics['break_dir_accuracy'] = float(acc)
        print(f"    Break dir accuracy: {acc:.1%}")

        # Per-regime break dir accuracy
        for r in range(4):
            mask = val_regimes == r
            if np.sum(mask) > 0:
                r_acc = np.mean(val_pred_class[mask] == Y_val['break_direction'][mask].astype(int))
                metrics[f'{self.REGIME_NAMES[r]}_break_dir_acc'] = float(r_acc)
                print(f"    {self.REGIME_NAMES[r]} break_dir acc: {r_acc:.1%} (n={np.sum(mask)})")

        importance = self.break_dir_model.feature_importance(importance_type='gain')
        sorted_imp = sorted(zip(aug_feature_names, importance), key=lambda x: -x[1])
        self.feature_importance['break_dir'] = sorted_imp

        return metrics

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict using regime-augmented features."""
        n = len(X)
        results = {
            'action': np.ones(n, dtype=np.int64),  # Default HOLD
            'break_dir': np.zeros(n, dtype=np.int64),
            'regime': np.zeros(n, dtype=np.int64),
        }

        # Classify regimes
        if self.regime_classifier is not None:
            regime_probs = self.regime_classifier.predict(X)
            regimes = np.argmax(regime_probs, axis=1)
            results['regime_probs'] = regime_probs
        else:
            regimes = self.classify_regime(X, self.feature_names)

        results['regime'] = regimes

        # Augment and predict
        X_aug = self._augment_features(X, self.feature_names)

        if self.action_model is not None:
            probs = self.action_model.predict(X_aug)
            results['action'] = np.argmax(probs, axis=1).astype(np.int64)
            results['action_probs'] = probs

        if self.break_dir_model is not None:
            probs = self.break_dir_model.predict(X_aug)
            results['break_dir'] = np.argmax(probs, axis=1).astype(np.int64)
            results['break_dir_probs'] = probs

        return results

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'regime_classifier': self.regime_classifier,
                'action_model': self.action_model,
                'break_dir_model': self.break_dir_model,
                'feature_names': self.feature_names,
                'augmented_feature_names': self.augmented_feature_names,
                'feature_importance': self.feature_importance,
            }, f)
        print(f"  Saved RegimeConditionalModel to {path}")

    @classmethod
    def load(cls, path: str) -> 'RegimeConditionalModel':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.regime_classifier = data['regime_classifier']
        model.action_model = data.get('action_model')
        model.break_dir_model = data.get('break_dir_model')
        model.feature_names = data['feature_names']
        model.augmented_feature_names = data.get('augmented_feature_names')
        model.feature_importance = data.get('feature_importance', {})
        return model


# ---------------------------------------------------------------------------
# Architecture 7: Temporal Attention Network (Sliding Window)
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn


class TemporalAttentionNet(nn.Module):
    """
    Compact model that processes a sliding window of top-K feature snapshots.

    Architecture:
    1. Per-timestep projection (n_features → d_model)
    2. Single-layer self-attention with 2 heads
    3. Attention-weighted pooling
    4. Concatenate with hand-crafted window trend features
    5. Task-specific heads

    Input: (batch, window_size, n_features) for window features
           (batch, n_trend_features) for hand-crafted trends
    """

    def __init__(self, n_features: int, n_trend_features: int,
                 window_size: int = 8, d_model: int = 24,
                 n_heads: int = 2, dropout: float = 0.2):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.d_model = d_model

        # Project features to d_model
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)

        # Single-layer self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Attention pooling
        self.attn_pool_q = nn.Linear(d_model, 1)

        # Combine attention output + trend features
        combined_dim = d_model + n_trend_features

        # Task heads
        self.action_head = nn.Sequential(
            nn.Linear(combined_dim, 16), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(16, 3),
        )
        self.break_dir_head = nn.Sequential(
            nn.Linear(combined_dim, 16), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(16, 3),
        )
        self.lifetime_head = nn.Sequential(
            nn.Linear(combined_dim, 16), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x_window, x_trends):
        """
        x_window: (batch, window_size, n_features)
        x_trends: (batch, n_trend_features)
        """
        h = self.input_proj(x_window)
        h = h + self.pos_embedding[:, :h.size(1), :]
        h = self.encoder(h)

        # Attention pooling
        attn_weights = torch.softmax(self.attn_pool_q(h), dim=1)
        pooled = (h * attn_weights).sum(dim=1)  # (B, d_model)

        # Concatenate with trend features
        combined = torch.cat([pooled, x_trends], dim=1)

        return {
            'action_logits': self.action_head(combined),
            'break_dir_logits': self.break_dir_head(combined),
            'lifetime': self.lifetime_head(combined).squeeze(-1),
            'attn_weights': attn_weights.squeeze(-1),
        }


class TemporalAttentionModel:
    """
    Wrapper for TemporalAttentionNet with training, prediction, and
    windowed data generation. Uses feature selection (top-K from GBT)
    and hand-crafted trend features for robustness with small samples.
    """

    WINDOW_SIZE = 8
    TOP_K_FEATURES = 30  # Select top-K by GBT importance

    def __init__(self, n_features: int = 169, window_size: int = 8):
        self.window_size = window_size
        self.n_features = n_features
        self.n_selected = self.TOP_K_FEATURES
        self.net = None
        self._device = 'cpu'
        self.feature_names = None
        self.selected_indices = None  # Top-K feature indices
        self.selected_names = None

    @staticmethod
    def create_windows(X: np.ndarray, window_size: int) -> np.ndarray:
        """Convert (N, F) into (N - window_size + 1, window_size, F) sliding windows."""
        n, f = X.shape
        if n < window_size:
            raise ValueError(f"Need at least {window_size} samples, got {n}")
        n_windows = n - window_size + 1
        windows = np.zeros((n_windows, window_size, f), dtype=np.float32)
        for i in range(n_windows):
            windows[i] = X[i:i + window_size]
        return windows

    @staticmethod
    def compute_trend_features(windows: np.ndarray) -> np.ndarray:
        """
        Compute hand-crafted trend features from each window.
        For each feature: slope (last - first), mean, std, delta (last - second-to-last).
        Returns (N_windows, n_features * 4) trend array.

        These capture temporal dynamics that attention alone struggles with
        on small datasets.
        """
        n_windows, ws, n_feat = windows.shape
        # Use vectorized operations for speed
        # Slope: last timestep - first timestep
        slopes = windows[:, -1, :] - windows[:, 0, :]
        # Mean across window
        means = windows.mean(axis=1)
        # Std across window
        stds = windows.std(axis=1)
        # Recent delta: last - second-to-last
        deltas = windows[:, -1, :] - windows[:, -2, :]

        return np.hstack([slopes, means, stds, deltas]).astype(np.float32)

    def _select_features(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[int]]:
        """Select top-K features using GBT importance. Returns (X_selected, indices)."""
        try:
            import lightgbm as lgb
        except ImportError:
            # Fallback: use variance-based selection
            variances = X.var(axis=0)
            indices = np.argsort(variances)[-self.TOP_K_FEATURES:]
            return X[:, indices], indices.tolist()

        # Quick GBT to rank features
        n = len(X)
        train_end = int(n * 0.7)
        dummy_y = np.zeros(n)  # We'll use a combined importance across tasks

        # Train brief action classifier for importance
        train_ds = lgb.Dataset(X[:train_end], label=dummy_y[:train_end],
                               feature_name=feature_names)
        # Use variance as proxy importance (faster than training multiple GBTs)
        variances = X.var(axis=0)
        indices = np.argsort(variances)[-self.TOP_K_FEATURES:]
        return X[:, indices], sorted(indices.tolist())

    def train(
        self,
        X_train: np.ndarray, Y_train: Dict[str, np.ndarray],
        X_val: np.ndarray, Y_val: Dict[str, np.ndarray],
        feature_names: List[str],
        gbt_importance: Optional[List[Tuple[str, float]]] = None,
    ) -> Dict[str, float]:
        """Train the temporal attention model on windowed + trend data."""
        import torch
        import torch.nn.functional as F

        self.feature_names = feature_names
        ws = self.window_size

        # Feature selection: use GBT importance if provided, else variance
        if gbt_importance:
            name_to_idx = {name: i for i, name in enumerate(feature_names)}
            indices = []
            for name, _ in gbt_importance[:self.TOP_K_FEATURES]:
                if name in name_to_idx:
                    indices.append(name_to_idx[name])
            self.selected_indices = sorted(indices)
        else:
            variances = X_train.var(axis=0)
            self.selected_indices = sorted(np.argsort(variances)[-self.TOP_K_FEATURES:].tolist())

        self.selected_names = [feature_names[i] for i in self.selected_indices]
        self.n_selected = len(self.selected_indices)
        print(f"\n  Selected {self.n_selected} features: {self.selected_names[:10]}...")

        # Select features
        X_train_sel = X_train[:, self.selected_indices]
        X_val_sel = X_val[:, self.selected_indices]

        # Create windows from selected features
        X_train_w = self.create_windows(X_train_sel, ws)
        X_val_w = self.create_windows(X_val_sel, ws)

        # Labels: aligned to the LAST element of each window
        Y_train_w = {k: v[ws - 1:] for k, v in Y_train.items()}
        Y_val_w = {k: v[ws - 1:] for k, v in Y_val.items()}

        # Compute trend features (slopes, means, stds, deltas)
        trends_train = self.compute_trend_features(X_train_w)
        trends_val = self.compute_trend_features(X_val_w)
        n_trend_features = trends_train.shape[1]

        print(f"  Windows: train={X_train_w.shape}, val={X_val_w.shape}")
        print(f"  Trend features: {n_trend_features}")

        # Z-score normalize windows
        flat_train = X_train_w.reshape(-1, self.n_selected)
        self._mean = flat_train.mean(axis=0)
        self._std = flat_train.std(axis=0)
        self._std[self._std < 1e-8] = 1.0

        X_train_w = (X_train_w - self._mean) / self._std
        X_val_w = (X_val_w - self._mean) / self._std

        # Normalize trend features
        self._trend_mean = trends_train.mean(axis=0)
        self._trend_std = trends_train.std(axis=0)
        self._trend_std[self._trend_std < 1e-8] = 1.0

        trends_train = (trends_train - self._trend_mean) / self._trend_std
        trends_val = (trends_val - self._trend_mean) / self._trend_std

        # Device
        if torch.backends.mps.is_available():
            self._device = 'mps'
        elif torch.cuda.is_available():
            self._device = 'cuda'
        device = torch.device(self._device)

        # Build model
        self.net = TemporalAttentionNet(
            n_features=self.n_selected, n_trend_features=n_trend_features,
            window_size=ws, d_model=24, n_heads=2, dropout=0.2,
        ).to(device)

        n_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f"  TemporalAttentionNet: {n_params:,} params, device={device}")

        # Training setup
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        # Class weights
        action_counts = np.bincount(Y_train_w['optimal_action'].astype(int), minlength=3)
        action_weights = torch.FloatTensor(1.0 / (action_counts + 1)).to(device)
        action_weights /= action_weights.sum()

        bd_counts = np.bincount(Y_train_w['break_direction'].astype(int), minlength=3)
        bd_weights = torch.FloatTensor(1.0 / (bd_counts + 1)).to(device)
        bd_weights /= bd_weights.sum()

        # Convert to tensors
        t_train_X = torch.FloatTensor(X_train_w).to(device)
        t_train_T = torch.FloatTensor(trends_train).to(device)
        t_val_X = torch.FloatTensor(X_val_w).to(device)
        t_val_T = torch.FloatTensor(trends_val).to(device)
        t_train_action = torch.LongTensor(Y_train_w['optimal_action']).to(device)
        t_val_action = torch.LongTensor(Y_val_w['optimal_action']).to(device)
        t_train_bd = torch.LongTensor(Y_train_w['break_direction']).to(device)
        t_val_bd = torch.LongTensor(Y_val_w['break_direction']).to(device)
        t_train_lt = torch.FloatTensor(Y_train_w['channel_lifetime']).to(device)
        t_val_lt = torch.FloatTensor(Y_val_w['channel_lifetime']).to(device)

        best_val_loss = float('inf')
        best_state = None
        patience = 50
        no_improve = 0
        batch_size = 64
        metrics = {}

        for epoch in range(300):
            self.net.train()
            perm = torch.randperm(len(t_train_X))
            epoch_loss = 0
            n_batches = 0

            for start in range(0, len(t_train_X), batch_size):
                idx = perm[start:start + batch_size]
                out = self.net(t_train_X[idx], t_train_T[idx])

                loss_action = F.cross_entropy(out['action_logits'], t_train_action[idx],
                                              weight=action_weights)
                loss_bd = F.cross_entropy(out['break_dir_logits'], t_train_bd[idx],
                                          weight=bd_weights)
                loss_lt = F.l1_loss(out['lifetime'], t_train_lt[idx])

                loss = loss_action + loss_bd + 0.01 * loss_lt

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Validation every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                self.net.eval()
                with torch.no_grad():
                    val_out = self.net(t_val_X, t_val_T)
                    val_loss_a = F.cross_entropy(val_out['action_logits'], t_val_action).item()
                    val_loss_bd = F.cross_entropy(val_out['break_dir_logits'], t_val_bd).item()
                    val_loss_lt = F.l1_loss(val_out['lifetime'], t_val_lt).item()
                    val_loss = val_loss_a + val_loss_bd + 0.01 * val_loss_lt

                    val_action_acc = (val_out['action_logits'].argmax(1) == t_val_action).float().mean().item()
                    val_bd_acc = (val_out['break_dir_logits'].argmax(1) == t_val_bd).float().mean().item()
                    attn = val_out['attn_weights'].mean(0).cpu().numpy()

                if (epoch + 1) % 50 == 0 or epoch == 0:
                    attn_str = ' '.join(f'{a:.2f}' for a in attn)
                    print(f"    Epoch {epoch+1:3d}/300 | Loss: {val_loss:.4f} | "
                          f"Act: {val_action_acc:.1%} | BD: {val_bd_acc:.1%} | "
                          f"LT_MAE: {val_loss_lt:.1f} | Attn: [{attn_str}]")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 10

                if no_improve >= patience:
                    print(f"    Early stopping at epoch {epoch + 1}")
                    break

        # Load best weights
        if best_state:
            self.net.load_state_dict(best_state)

        # Final evaluation
        self.net.eval()
        with torch.no_grad():
            val_out = self.net(t_val_X, t_val_T)
            val_action_pred = val_out['action_logits'].argmax(1).cpu().numpy()
            val_bd_pred = val_out['break_dir_logits'].argmax(1).cpu().numpy()
            val_lt_pred = val_out['lifetime'].cpu().numpy()

        metrics['action_accuracy'] = float(np.mean(val_action_pred == Y_val_w['optimal_action'].astype(int)))
        metrics['break_dir_accuracy'] = float(np.mean(val_bd_pred == Y_val_w['break_direction'].astype(int)))
        metrics['lifetime_mae'] = float(np.mean(np.abs(val_lt_pred - Y_val_w['channel_lifetime'])))
        metrics['best_val_loss'] = float(best_val_loss)

        print(f"\n  Temporal Attention Results:")
        print(f"    Action accuracy: {metrics['action_accuracy']:.1%}")
        print(f"    Break dir accuracy: {metrics['break_dir_accuracy']:.1%}")
        print(f"    Lifetime MAE: {metrics['lifetime_mae']:.1f} bars")

        # Attention analysis
        with torch.no_grad():
            val_out = self.net(t_val_X, t_val_T)
            attn = val_out['attn_weights'].mean(0).cpu().numpy()
        print(f"    Attention weights (recent→old): {' '.join(f'{a:.3f}' for a in attn)}")
        most_attended = np.argmax(attn)
        print(f"    Most attended: t-{most_attended} (0=most recent)")

        return metrics

    def predict(self, X_window: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict from a window of full features (will select features internally).
        X_window: (N, window_size, n_all_features) or (window_size, n_all_features)
        """
        import torch

        if X_window.ndim == 2:
            X_window = X_window[np.newaxis, :, :]

        # Select features
        X_sel = X_window[:, :, self.selected_indices]

        # Compute trends
        trends = self.compute_trend_features(X_sel)

        # Normalize
        X_norm = (X_sel - self._mean) / self._std
        trends_norm = (trends - self._trend_mean) / self._trend_std

        device = torch.device(self._device)
        self.net.eval()
        with torch.no_grad():
            t_X = torch.FloatTensor(X_norm).to(device)
            t_T = torch.FloatTensor(trends_norm).to(device)
            out = self.net(t_X, t_T)

        return {
            'action': out['action_logits'].argmax(1).cpu().numpy(),
            'break_dir': out['break_dir_logits'].argmax(1).cpu().numpy(),
            'lifetime': out['lifetime'].cpu().numpy(),
            'attn_weights': out['attn_weights'].cpu().numpy(),
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'n_features': self.n_features,
            'n_selected': self.n_selected,
            'window_size': self.window_size,
            'feature_names': self.feature_names,
            'selected_indices': self.selected_indices,
            'selected_names': self.selected_names,
            '_mean': self._mean,
            '_std': self._std,
            '_trend_mean': self._trend_mean,
            '_trend_std': self._trend_std,
            'n_trend_features': self._trend_mean.shape[0],
        }, path)
        print(f"  Saved TemporalAttentionModel to {path}")

    @classmethod
    def load(cls, path: str) -> 'TemporalAttentionModel':
        data = torch.load(path, map_location='cpu', weights_only=False)
        model = cls(n_features=data['n_features'], window_size=data['window_size'])
        model.n_selected = data['n_selected']
        model.feature_names = data['feature_names']
        model.selected_indices = data['selected_indices']
        model.selected_names = data.get('selected_names')
        model._mean = data['_mean']
        model._std = data['_std']
        model._trend_mean = data['_trend_mean']
        model._trend_std = data['_trend_std']

        model.net = TemporalAttentionNet(
            n_features=data['n_selected'],
            n_trend_features=data['n_trend_features'],
            window_size=data['window_size'],
        )
        model.net.load_state_dict(data['model_state_dict'])

        if torch.backends.mps.is_available():
            model._device = 'mps'
        model.net = model.net.to(torch.device(model._device))
        model.net.eval()
        return model


# ---------------------------------------------------------------------------
# Architecture 8: Feature-Selected Trend GBT (Best of Both Worlds)
# ---------------------------------------------------------------------------

class TrendGBTModel:
    """
    Combines feature selection with temporal trend features in GBT.

    Strategy:
    1. Select top-K features by GBT importance (removes noise)
    2. Create sliding windows and compute trend features (slopes, means, stds, deltas)
    3. Concatenate: selected_features + trend_features → compact GBT
    4. Much fewer features = better generalization with small sample counts

    This is the "temporal attention but in GBT form" — captures the same
    temporal dynamics but uses GBT's native strength on tabular data.
    """

    WINDOW_SIZE = 8
    TOP_K = 15  # Sweet spot: 10 was too aggressive, 30 overfits

    def __init__(self):
        self.action_model = None
        self.break_dir_model = None
        self.lifetime_model = None
        self.feature_names = None
        self.selected_indices = None
        self.selected_names = None
        self.augmented_feature_names = None
        self.feature_importance = {}

    def train(
        self,
        X_train: np.ndarray, Y_train: Dict[str, np.ndarray],
        X_val: np.ndarray, Y_val: Dict[str, np.ndarray],
        feature_names: List[str],
        gbt_importance: Optional[List[Tuple[str, float]]] = None,
    ) -> Dict[str, float]:
        """Train feature-selected trend GBT."""
        import lightgbm as lgb

        self.feature_names = feature_names
        ws = self.WINDOW_SIZE
        metrics = {}

        # Step 1: Feature selection via GBT importance
        if gbt_importance:
            name_to_idx = {name: i for i, name in enumerate(feature_names)}
            indices = []
            for name, _ in gbt_importance[:self.TOP_K]:
                if name in name_to_idx:
                    indices.append(name_to_idx[name])
            self.selected_indices = sorted(indices)
        else:
            variances = X_train.var(axis=0)
            self.selected_indices = sorted(np.argsort(variances)[-self.TOP_K:].tolist())

        self.selected_names = [feature_names[i] for i in self.selected_indices]
        n_sel = len(self.selected_indices)
        print(f"\n  Selected {n_sel} features by importance")
        print(f"  Top features: {self.selected_names[:10]}")

        # Step 2: Create windows and compute trend features
        X_train_sel = X_train[:, self.selected_indices]
        X_val_sel = X_val[:, self.selected_indices]

        X_train_w = TemporalAttentionModel.create_windows(X_train_sel, ws)
        X_val_w = TemporalAttentionModel.create_windows(X_val_sel, ws)

        trends_train = TemporalAttentionModel.compute_trend_features(X_train_w)
        trends_val = TemporalAttentionModel.compute_trend_features(X_val_w)

        # Labels aligned to window end
        Y_train_w = {k: v[ws - 1:] for k, v in Y_train.items()}
        Y_val_w = {k: v[ws - 1:] for k, v in Y_val.items()}

        # Step 3: Concatenate current snapshot (last in window) + trend features
        X_train_current = X_train_sel[ws - 1:]  # Current snapshot
        X_val_current = X_val_sel[ws - 1:]

        X_train_aug = np.hstack([X_train_current, trends_train])
        X_val_aug = np.hstack([X_val_current, trends_val])

        # Build augmented feature names
        trend_prefixes = ['slope', 'mean', 'std', 'delta']
        trend_names = []
        for prefix in trend_prefixes:
            for name in self.selected_names:
                trend_names.append(f'{prefix}_{name}')

        self.augmented_feature_names = list(self.selected_names) + trend_names
        print(f"  Augmented features: {n_sel} selected + {len(trend_names)} trends = {X_train_aug.shape[1]}")
        print(f"  Samples: train={len(X_train_aug)}, val={len(X_val_aug)}")

        # Step 4: Train GBT models
        # Action model
        print("\n  Training TrendGBT action model...")
        train_ds = lgb.Dataset(X_train_aug, label=Y_train_w['optimal_action'],
                               feature_name=self.augmented_feature_names)
        val_ds = lgb.Dataset(X_val_aug, label=Y_val_w['optimal_action'],
                             feature_name=self.augmented_feature_names, reference=train_ds)

        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'num_leaves': 20,
            'learning_rate': 0.03,
            'min_child_samples': 8,
            'feature_fraction': 0.8,
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'verbose': -1,
        }
        self.action_model = lgb.train(
            params, train_ds, num_boost_round=500,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)],
        )

        val_pred = self.action_model.predict(X_val_aug)
        val_pred_class = np.argmax(val_pred, axis=1)
        acc = np.mean(val_pred_class == Y_val_w['optimal_action'].astype(int))
        metrics['action_accuracy'] = float(acc)
        print(f"    Action accuracy: {acc:.1%}")

        # Feature importance
        importance = self.action_model.feature_importance(importance_type='gain')
        sorted_imp = sorted(zip(self.augmented_feature_names, importance), key=lambda x: -x[1])
        self.feature_importance['action'] = sorted_imp
        print("    Top action features:")
        for name, imp in sorted_imp[:10]:
            print(f"      {name}: {imp:.0f}")

        # Break direction model
        print("\n  Training TrendGBT break direction model...")
        train_ds = lgb.Dataset(X_train_aug, label=Y_train_w['break_direction'],
                               feature_name=self.augmented_feature_names)
        val_ds = lgb.Dataset(X_val_aug, label=Y_val_w['break_direction'],
                             feature_name=self.augmented_feature_names, reference=train_ds)

        self.break_dir_model = lgb.train(
            params, train_ds, num_boost_round=500,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)],
        )

        val_pred = self.break_dir_model.predict(X_val_aug)
        acc = np.mean(np.argmax(val_pred, axis=1) == Y_val_w['break_direction'].astype(int))
        metrics['break_dir_accuracy'] = float(acc)
        print(f"    Break dir accuracy: {acc:.1%}")

        importance = self.break_dir_model.feature_importance(importance_type='gain')
        sorted_imp = sorted(zip(self.augmented_feature_names, importance), key=lambda x: -x[1])
        self.feature_importance['break_dir'] = sorted_imp

        # Lifetime model
        print("\n  Training TrendGBT lifetime model...")
        reg_params = {
            'objective': 'mae',
            'metric': 'mae',
            'num_leaves': 20,
            'learning_rate': 0.03,
            'min_child_samples': 8,
            'feature_fraction': 0.8,
            'lambda_l1': 0.1,
            'verbose': -1,
        }
        train_ds = lgb.Dataset(X_train_aug, label=Y_train_w['channel_lifetime'],
                               feature_name=self.augmented_feature_names)
        val_ds = lgb.Dataset(X_val_aug, label=Y_val_w['channel_lifetime'],
                             feature_name=self.augmented_feature_names, reference=train_ds)

        self.lifetime_model = lgb.train(
            reg_params, train_ds, num_boost_round=500,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)],
        )

        val_pred = self.lifetime_model.predict(X_val_aug)
        mae = np.mean(np.abs(val_pred - Y_val_w['channel_lifetime']))
        metrics['lifetime_mae'] = float(mae)
        print(f"    Lifetime MAE: {mae:.1f} bars")

        importance = self.lifetime_model.feature_importance(importance_type='gain')
        sorted_imp = sorted(zip(self.augmented_feature_names, importance), key=lambda x: -x[1])
        self.feature_importance['lifetime'] = sorted_imp
        print("    Top lifetime features:")
        for name, imp in sorted_imp[:10]:
            print(f"      {name}: {imp:.0f}")

        return metrics

    def _build_augmented_input(self, X_window: np.ndarray) -> np.ndarray:
        """Build augmented input from a window of full features."""
        X_sel = X_window[:, :, self.selected_indices]
        trends = TemporalAttentionModel.compute_trend_features(X_sel)
        current = X_sel[:, -1, :]  # Last timestep
        return np.hstack([current, trends])

    def predict(self, X_window: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict from windowed full features.
        X_window: (N, window_size, n_all_features) or (window_size, n_all_features)
        """
        if X_window.ndim == 2:
            X_window = X_window[np.newaxis, :, :]

        X_aug = self._build_augmented_input(X_window)
        results = {}

        if self.action_model is not None:
            probs = self.action_model.predict(X_aug)
            results['action'] = np.argmax(probs, axis=1).astype(np.int64)
            results['action_probs'] = probs

        if self.break_dir_model is not None:
            probs = self.break_dir_model.predict(X_aug)
            results['break_dir'] = np.argmax(probs, axis=1).astype(np.int64)

        if self.lifetime_model is not None:
            results['lifetime'] = self.lifetime_model.predict(X_aug)

        return results

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'action_model': self.action_model,
                'break_dir_model': self.break_dir_model,
                'lifetime_model': self.lifetime_model,
                'feature_names': self.feature_names,
                'selected_indices': self.selected_indices,
                'selected_names': self.selected_names,
                'augmented_feature_names': self.augmented_feature_names,
                'feature_importance': self.feature_importance,
            }, f)
        print(f"  Saved TrendGBTModel to {path}")

    @classmethod
    def load(cls, path: str) -> 'TrendGBTModel':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.action_model = data['action_model']
        model.break_dir_model = data['break_dir_model']
        model.lifetime_model = data['lifetime_model']
        model.feature_names = data['feature_names']
        model.selected_indices = data['selected_indices']
        model.selected_names = data['selected_names']
        model.augmented_feature_names = data['augmented_feature_names']
        model.feature_importance = data.get('feature_importance', {})
        return model


# ---------------------------------------------------------------------------
# Architecture 9: Cross-Validated Ensemble with Uncertainty
# ---------------------------------------------------------------------------

class CVEnsembleModel:
    """
    Trains K GBT models on different folds, averages predictions.

    Key insight: prediction disagreement = uncertainty.
    When K models agree → high confidence → trade.
    When K models disagree → uncertain → skip/reduce size.

    This provides calibrated confidence without temperature scaling.
    """

    N_FOLDS = 5

    def __init__(self):
        self.fold_models = {}  # {fold_id: {task: model}}
        self.feature_names = None
        self.val_calibration = {}  # Calibration stats from validation

    def train(
        self,
        X: np.ndarray, Y: Dict[str, np.ndarray],
        feature_names: List[str],
    ) -> Dict[str, float]:
        """Train K-fold ensemble on all data."""
        import lightgbm as lgb

        self.feature_names = feature_names
        n = len(X)
        k = self.N_FOLDS
        fold_size = n // k
        metrics = {}

        print(f"\n  Training {k}-fold ensemble ({n} samples, {fold_size}/fold)...")

        all_action_preds = np.zeros((n, 3))
        all_bd_preds = np.zeros((n, 3))
        all_lt_preds = np.zeros(n)
        in_val = np.zeros(n, dtype=bool)

        for fold in range(k):
            val_start = fold * fold_size
            val_end = min((fold + 1) * fold_size, n)

            val_mask = np.zeros(n, dtype=bool)
            val_mask[val_start:val_end] = True
            train_mask = ~val_mask

            X_train_f = X[train_mask]
            X_val_f = X[val_mask]

            print(f"\n  Fold {fold+1}/{k}: train={train_mask.sum()}, val={val_mask.sum()}")

            self.fold_models[fold] = {}

            # Action model
            y_train = Y['optimal_action'][train_mask]
            y_val = Y['optimal_action'][val_mask]
            train_ds = lgb.Dataset(X_train_f, label=y_train, feature_name=feature_names)
            val_ds = lgb.Dataset(X_val_f, label=y_val, feature_name=feature_names, reference=train_ds)

            params = {
                'objective': 'multiclass', 'num_class': 3,
                'metric': 'multi_logloss', 'num_leaves': 20,
                'learning_rate': 0.05, 'min_child_samples': 8,
                'feature_fraction': 0.8, 'verbose': -1,
            }
            model = lgb.train(
                params, train_ds, num_boost_round=300,
                valid_sets=[val_ds],
                callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(0)],
            )
            self.fold_models[fold]['action'] = model
            all_action_preds[val_mask] = model.predict(X_val_f)

            # Break direction model
            y_train = Y['break_direction'][train_mask]
            y_val = Y['break_direction'][val_mask]
            train_ds = lgb.Dataset(X_train_f, label=y_train, feature_name=feature_names)
            val_ds = lgb.Dataset(X_val_f, label=y_val, feature_name=feature_names, reference=train_ds)

            model = lgb.train(
                params, train_ds, num_boost_round=300,
                valid_sets=[val_ds],
                callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(0)],
            )
            self.fold_models[fold]['break_dir'] = model
            all_bd_preds[val_mask] = model.predict(X_val_f)

            # Lifetime model
            y_train = Y['channel_lifetime'][train_mask]
            y_val = Y['channel_lifetime'][val_mask]
            train_ds = lgb.Dataset(X_train_f, label=y_train, feature_name=feature_names)
            val_ds = lgb.Dataset(X_val_f, label=y_val, feature_name=feature_names, reference=train_ds)

            reg_params = {
                'objective': 'mae', 'metric': 'mae',
                'num_leaves': 20, 'learning_rate': 0.05,
                'min_child_samples': 8, 'verbose': -1,
            }
            model = lgb.train(
                reg_params, train_ds, num_boost_round=300,
                valid_sets=[val_ds],
                callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(0)],
            )
            self.fold_models[fold]['lifetime'] = model
            all_lt_preds[val_mask] = model.predict(X_val_f)

            in_val |= val_mask

        # Compute out-of-fold metrics
        action_pred = np.argmax(all_action_preds, axis=1)
        bd_pred = np.argmax(all_bd_preds, axis=1)

        metrics['action_accuracy'] = float(np.mean(action_pred == Y['optimal_action'].astype(int)))
        metrics['break_dir_accuracy'] = float(np.mean(bd_pred == Y['break_direction'].astype(int)))
        metrics['lifetime_mae'] = float(np.mean(np.abs(all_lt_preds - Y['channel_lifetime'])))

        print(f"\n  Out-of-fold results:")
        print(f"    Action accuracy: {metrics['action_accuracy']:.1%}")
        print(f"    Break dir accuracy: {metrics['break_dir_accuracy']:.1%}")
        print(f"    Lifetime MAE: {metrics['lifetime_mae']:.1f} bars")

        # Calibrate uncertainty: compute per-sample agreement stats
        # Run all folds on all data, measure prediction variance
        fold_action_probs = []
        fold_bd_probs = []
        for fold in range(k):
            fold_action_probs.append(self.fold_models[fold]['action'].predict(X))
            fold_bd_probs.append(self.fold_models[fold]['break_dir'].predict(X))

        action_stack = np.stack(fold_action_probs)  # (K, N, 3)
        bd_stack = np.stack(fold_bd_probs)

        # Agreement: std of max-class probabilities across folds
        action_max_probs = action_stack.max(axis=2)  # (K, N) - max class prob per fold
        action_agreement = 1.0 - action_max_probs.std(axis=0)  # High = agree

        # For break direction, check if majority of folds agree on the same class
        bd_classes = bd_stack.argmax(axis=2)  # (K, N)
        bd_consensus = np.zeros(n)
        for i in range(n):
            most_common = np.bincount(bd_classes[:, i].astype(int), minlength=3)
            bd_consensus[i] = most_common.max() / k

        metrics['avg_action_agreement'] = float(action_agreement.mean())
        metrics['avg_bd_consensus'] = float(bd_consensus.mean())
        print(f"    Avg action agreement: {metrics['avg_action_agreement']:.3f}")
        print(f"    Avg BD consensus: {metrics['avg_bd_consensus']:.1%}")

        # Calibration: when agreement > threshold, what's the accuracy?
        for thresh in [0.6, 0.7, 0.8, 0.9]:
            mask = bd_consensus >= thresh
            if mask.sum() > 10:
                acc = np.mean(bd_pred[mask] == Y['break_direction'][mask].astype(int))
                pct = mask.mean()
                print(f"    BD consensus >= {thresh:.0%}: acc={acc:.1%}, coverage={pct:.1%}")

        self.val_calibration = {
            'action_agreement_mean': float(action_agreement.mean()),
            'bd_consensus_mean': float(bd_consensus.mean()),
        }

        return metrics

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict using all K folds, return averaged predictions + uncertainty."""
        n = len(X)
        k = len(self.fold_models)

        fold_action_probs = []
        fold_bd_probs = []
        fold_lt_preds = []

        for fold in range(k):
            fold_action_probs.append(self.fold_models[fold]['action'].predict(X))
            fold_bd_probs.append(self.fold_models[fold]['break_dir'].predict(X))
            fold_lt_preds.append(self.fold_models[fold]['lifetime'].predict(X))

        # Average probabilities
        action_probs = np.mean(fold_action_probs, axis=0)
        bd_probs = np.mean(fold_bd_probs, axis=0)
        lifetime = np.mean(fold_lt_preds, axis=0)

        # Uncertainty metrics
        action_stack = np.stack(fold_action_probs)
        bd_stack = np.stack(fold_bd_probs)

        # Action agreement: how much folds agree
        action_classes = action_stack.argmax(axis=2)  # (K, N)
        action_consensus = np.zeros(n)
        for i in range(n):
            counts = np.bincount(action_classes[:, i].astype(int), minlength=3)
            action_consensus[i] = counts.max() / k

        # BD consensus
        bd_classes = bd_stack.argmax(axis=2)
        bd_consensus = np.zeros(n)
        for i in range(n):
            counts = np.bincount(bd_classes[:, i].astype(int), minlength=3)
            bd_consensus[i] = counts.max() / k

        return {
            'action': np.argmax(action_probs, axis=1).astype(np.int64),
            'action_probs': action_probs,
            'break_dir': np.argmax(bd_probs, axis=1).astype(np.int64),
            'break_dir_probs': bd_probs,
            'lifetime': lifetime,
            'action_consensus': action_consensus,
            'bd_consensus': bd_consensus,
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'fold_models': self.fold_models,
                'feature_names': self.feature_names,
                'val_calibration': self.val_calibration,
            }, f)
        print(f"  Saved CVEnsembleModel to {path}")

    @classmethod
    def load(cls, path: str) -> 'CVEnsembleModel':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.fold_models = data['fold_models']
        model.feature_names = data['feature_names']
        model.val_calibration = data.get('val_calibration', {})
        return model


# ---------------------------------------------------------------------------
# Architecture 10: Physics-Residual Correction Model
# ---------------------------------------------------------------------------

class PhysicsResidualModel:
    """
    Learns WHEN physics is wrong.

    Instead of predicting raw labels, this model predicts the residual between
    what physics implies and actual outcomes. It answers:
    - "Will this physics signal produce a winning trade?" (binary)
    - "How much should we adjust the confidence?" (regression)
    - "Is the physics break direction prediction correct?" (binary)

    The key insight: the physics features (position_pct, momentum, break_prob,
    channel_health, binding_energy) encode an *implicit* prediction. We derive
    that prediction, compare to actual labels, and train on the ERROR.

    This is a correction model, not a replacement.
    """

    def __init__(self):
        self.models = {}
        self.feature_names = None
        self.augmented_names = None
        self.physics_stats = {}  # Calibration stats

    @staticmethod
    def derive_physics_prediction(X: np.ndarray, feature_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Derive the implicit physics prediction from feature vectors.

        For each sample, compute what the physics engine *would suggest*
        based on the raw physics features (position, momentum, break probs, etc.)
        """
        idx = {name: i for i, name in enumerate(feature_names)}
        n = len(X)

        implied_action = np.zeros(n, dtype=int)  # 0=hold, 1=buy, 2=sell
        implied_confidence = np.zeros(n)
        implied_break_dir = np.zeros(n, dtype=int)  # 0=survive, 1=up, 2=down
        implied_lifetime = np.zeros(n)

        for i in range(n):
            x = X[i]

            # --- Implied action from multi-TF position + momentum ---
            buy_score = 0.0
            sell_score = 0.0
            tf_count = 0

            for tf in ML_TFS:
                pos_key = f'{tf}_position_pct'
                mom_key = f'{tf}_momentum_direction'
                health_key = f'{tf}_channel_health'

                if pos_key in idx and mom_key in idx and health_key in idx:
                    pos = x[idx[pos_key]]
                    mom = x[idx[mom_key]]
                    health = x[idx[health_key]]
                    tf_count += 1

                    # Near channel bottom with upward momentum → buy signal
                    if pos < 0.25 and mom > 0:
                        buy_score += health
                    elif pos < 0.15:  # Very near bottom regardless of momentum
                        buy_score += health * 0.5

                    # Near channel top with downward momentum → sell signal
                    if pos > 0.75 and mom < 0:
                        sell_score += health
                    elif pos > 0.85:  # Very near top
                        sell_score += health * 0.5

            if tf_count > 0:
                buy_score /= tf_count
                sell_score /= tf_count

            if buy_score > sell_score + 0.1:
                implied_action[i] = 1  # BUY
            elif sell_score > buy_score + 0.1:
                implied_action[i] = 2  # SELL
            # else: HOLD

            # --- Implied confidence from health + binding energy ---
            healths = []
            bindings = []
            for tf in ML_TFS:
                hk = f'{tf}_channel_health'
                bk = f'{tf}_binding_energy'
                if hk in idx:
                    healths.append(x[idx[hk]])
                if bk in idx:
                    bindings.append(x[idx[bk]])

            avg_health = np.mean(healths) if healths else 0.5
            avg_binding = np.mean(bindings) if bindings else 0.5
            implied_confidence[i] = np.clip(
                avg_health * 0.6 + np.clip(avg_binding, 0, 1) * 0.4, 0, 1
            )

            # --- Implied break direction from break_prob_up/down ---
            bp_ups = []
            bp_downs = []
            for tf in ML_TFS:
                up_key = f'{tf}_break_prob_up'
                dn_key = f'{tf}_break_prob_down'
                if up_key in idx:
                    bp_ups.append(x[idx[up_key]])
                if dn_key in idx:
                    bp_downs.append(x[idx[dn_key]])

            avg_bp_up = np.mean(bp_ups) if bp_ups else 0.0
            avg_bp_down = np.mean(bp_downs) if bp_downs else 0.0

            if avg_bp_up > avg_bp_down + 0.03:
                implied_break_dir[i] = 1
            elif avg_bp_down > avg_bp_up + 0.03:
                implied_break_dir[i] = 2
            # else: survive (0)

            # --- Implied lifetime from OU half-life ---
            half_lives = []
            for tf in ML_TFS:
                hl_key = f'{tf}_ou_half_life'
                if hl_key in idx:
                    hl = x[idx[hl_key]]
                    if 0 < hl < 500:  # Sanity check
                        half_lives.append(hl)
            implied_lifetime[i] = np.mean(half_lives) if half_lives else 30.0

        return {
            'implied_action': implied_action,
            'implied_confidence': implied_confidence,
            'implied_break_dir': implied_break_dir,
            'implied_lifetime': implied_lifetime,
        }

    @staticmethod
    def compute_residual_targets(
        physics_preds: Dict[str, np.ndarray],
        Y: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Compute residual targets: how wrong was physics?

        Returns binary correctness + regression corrections.
        """
        n = len(Y['optimal_action'])
        actual_action = Y['optimal_action'].astype(int)
        actual_bd = Y['break_direction'].astype(int)
        actual_lifetime = Y['channel_lifetime']

        # 1. Action correctness (binary: was physics right?)
        action_correct = (physics_preds['implied_action'] == actual_action).astype(float)

        # 2. Break direction correctness (binary)
        bd_correct = (physics_preds['implied_break_dir'] == actual_bd).astype(float)

        # 3. Lifetime error (actual - predicted)
        # Positive = physics underestimated lifetime (channel lasted longer)
        # Negative = physics overestimated (channel broke sooner)
        lifetime_error = actual_lifetime - physics_preds['implied_lifetime']

        # 4. Confidence scale: how much to multiply physics confidence
        # If physics was right AND future return is big → scale > 1
        # If physics was wrong → scale < 1
        confidence_scale = np.ones(n)
        for i in range(n):
            if action_correct[i]:
                # Physics was right — scale up proportionally to return magnitude
                ret_mag = abs(Y['future_return_20'][i])
                confidence_scale[i] = 1.0 + min(ret_mag * 10, 0.5)  # 1.0 to 1.5
            else:
                # Physics was wrong — scale down
                ret_mag = abs(Y['future_return_20'][i])
                confidence_scale[i] = max(0.3, 1.0 - min(ret_mag * 10, 0.7))  # 0.3 to 1.0

        return {
            'action_correct': action_correct,
            'bd_correct': bd_correct,
            'lifetime_error': lifetime_error,
            'confidence_scale': confidence_scale,
        }

    @staticmethod
    def augment_features(
        X: np.ndarray,
        physics_preds: Dict[str, np.ndarray],
        feature_names: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Augment features with physics prediction + derived residual features.

        Adds:
        - Raw physics predictions (action, confidence, break_dir, lifetime)
        - Cross-term features (confidence × health, action × momentum, etc.)
        - Physics disagreement features (when TFs disagree on action)
        """
        idx = {name: i for i, name in enumerate(feature_names)}
        n = len(X)

        # Physics prediction features
        phys_feats = np.column_stack([
            physics_preds['implied_action'].astype(float),
            physics_preds['implied_confidence'],
            physics_preds['implied_break_dir'].astype(float),
            physics_preds['implied_lifetime'],
        ])
        phys_names = [
            'phys_implied_action', 'phys_implied_confidence',
            'phys_implied_break_dir', 'phys_implied_lifetime',
        ]

        # Cross-term features: physics prediction × key feature interactions
        cross_feats = []
        cross_names = []

        # Confidence × break_prob_max (how confident physics is vs how likely break is)
        if 'break_prob_max' in idx:
            bp_max = X[:, idx['break_prob_max']]
            cross_feats.append(physics_preds['implied_confidence'] * bp_max)
            cross_names.append('phys_conf_x_bp_max')

        # Confidence × avg entropy (high entropy + high confidence = suspicious)
        if 'avg_entropy' in idx:
            entropy = X[:, idx['avg_entropy']]
            cross_feats.append(physics_preds['implied_confidence'] * entropy)
            cross_names.append('phys_conf_x_entropy')

        # Confidence × health spread (disagreement across TFs)
        if 'health_spread' in idx:
            h_spread = X[:, idx['health_spread']]
            cross_feats.append(physics_preds['implied_confidence'] * h_spread)
            cross_names.append('phys_conf_x_health_spread')

        # Action direction × direction_consensus
        if 'direction_consensus' in idx:
            dc = X[:, idx['direction_consensus']]
            cross_feats.append(physics_preds['implied_action'].astype(float) * dc)
            cross_names.append('phys_action_x_dir_consensus')

        # Lifetime × break_prob_weighted
        if 'break_prob_weighted' in idx:
            bp_w = X[:, idx['break_prob_weighted']]
            cross_feats.append(physics_preds['implied_lifetime'] * bp_w)
            cross_names.append('phys_lifetime_x_bp_weighted')

        # Per-TF disagreement features
        # How many TFs agree with the implied action
        tf_agreement_count = np.zeros(n)
        for tf in ML_TFS:
            pos_key = f'{tf}_position_pct'
            mom_key = f'{tf}_momentum_direction'
            if pos_key in idx and mom_key in idx:
                pos = X[:, idx[pos_key]]
                mom = X[:, idx[mom_key]]
                tf_buy = ((pos < 0.25) & (mom > 0)).astype(float)
                tf_sell = ((pos > 0.75) & (mom < 0)).astype(float)
                agrees = np.where(
                    physics_preds['implied_action'] == 1,
                    tf_buy,
                    np.where(physics_preds['implied_action'] == 2, tf_sell, 0.0),
                )
                tf_agreement_count += agrees

        cross_feats.append(tf_agreement_count)
        cross_names.append('phys_tf_agreement_count')

        # Stack all augmented features
        if cross_feats:
            cross_array = np.column_stack(cross_feats)
            aug_X = np.hstack([X, phys_feats, cross_array])
            aug_names = feature_names + phys_names + cross_names
        else:
            aug_X = np.hstack([X, phys_feats])
            aug_names = feature_names + phys_names

        return aug_X, aug_names

    def train(
        self,
        X_train: np.ndarray, Y_train: Dict[str, np.ndarray],
        X_val: np.ndarray, Y_val: Dict[str, np.ndarray],
        feature_names: List[str],
    ) -> Dict[str, float]:
        """Train the physics-residual correction models."""
        import lightgbm as lgb

        self.feature_names = feature_names
        metrics = {}

        # Step 1: Derive physics predictions
        print("\n  Deriving physics predictions from features...")
        train_phys = self.derive_physics_prediction(X_train, feature_names)
        val_phys = self.derive_physics_prediction(X_val, feature_names)

        # Physics baseline stats
        train_actual_action = Y_train['optimal_action'].astype(int)
        val_actual_action = Y_val['optimal_action'].astype(int)

        train_phys_acc = np.mean(train_phys['implied_action'] == train_actual_action)
        val_phys_acc = np.mean(val_phys['implied_action'] == val_actual_action)
        print(f"  Physics baseline action accuracy: train={train_phys_acc:.1%}, val={val_phys_acc:.1%}")

        train_phys_bd = np.mean(train_phys['implied_break_dir'] == Y_train['break_direction'].astype(int))
        val_phys_bd = np.mean(val_phys['implied_break_dir'] == Y_val['break_direction'].astype(int))
        print(f"  Physics baseline BD accuracy: train={train_phys_bd:.1%}, val={val_phys_bd:.1%}")

        metrics['physics_action_acc'] = float(val_phys_acc)
        metrics['physics_bd_acc'] = float(val_phys_bd)

        self.physics_stats = {
            'train_action_acc': float(train_phys_acc),
            'val_action_acc': float(val_phys_acc),
            'train_bd_acc': float(train_phys_bd),
            'val_bd_acc': float(val_phys_bd),
        }

        # Step 2: Compute residual targets
        print("  Computing residual targets...")
        train_residuals = self.compute_residual_targets(train_phys, Y_train)
        val_residuals = self.compute_residual_targets(val_phys, Y_val)

        action_correct_rate = train_residuals['action_correct'].mean()
        print(f"  Physics action correct rate: {action_correct_rate:.1%}")
        print(f"  Avg confidence scale: {train_residuals['confidence_scale'].mean():.3f}")
        print(f"  Avg lifetime error: {train_residuals['lifetime_error'].mean():.1f} bars")

        # Step 3: Augment features
        print("  Augmenting features with physics predictions...")
        X_train_aug, aug_names = self.augment_features(X_train, train_phys, feature_names)
        X_val_aug, _ = self.augment_features(X_val, val_phys, feature_names)
        self.augmented_names = aug_names
        print(f"  Augmented features: {len(aug_names)} ({len(feature_names)} base + {len(aug_names) - len(feature_names)} new)")

        # Step 4: Train models on residual targets

        # 4a: Action correctness (binary: will physics signal be right?)
        print("\n  Training: action_correct (binary, will physics be right?)...")
        y_train = train_residuals['action_correct']
        y_val = val_residuals['action_correct']
        train_ds = lgb.Dataset(X_train_aug, label=y_train, feature_name=aug_names)
        val_ds = lgb.Dataset(X_val_aug, label=y_val, feature_name=aug_names, reference=train_ds)

        params = {
            'objective': 'binary', 'metric': 'auc',
            'num_leaves': 24, 'learning_rate': 0.05,
            'min_child_samples': 10, 'feature_fraction': 0.8,
            'bagging_fraction': 0.9, 'bagging_freq': 5,
            'verbose': -1,
        }
        model = lgb.train(
            params, train_ds, num_boost_round=400,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(stopping_rounds=40), lgb.log_evaluation(0)],
        )
        self.models['action_correct'] = model

        val_pred_prob = model.predict(X_val_aug)
        val_pred = (val_pred_prob > 0.5).astype(int)
        acc = np.mean(val_pred == y_val)
        metrics['action_correct_acc'] = float(acc)
        print(f"    Val accuracy: {acc:.1%}")

        # When model says "physics is right" (>0.7), what's actual correctness?
        high_conf = val_pred_prob > 0.7
        if high_conf.sum() > 5:
            high_conf_acc = np.mean(y_val[high_conf] == 1)
            metrics['action_correct_high_conf'] = float(high_conf_acc)
            metrics['action_correct_high_conf_coverage'] = float(high_conf.mean())
            print(f"    High-confidence (>0.7) accuracy: {high_conf_acc:.1%} "
                  f"(coverage: {high_conf.mean():.1%})")

        # When model says "physics is wrong" (<0.3), what's actual correctness?
        low_conf = val_pred_prob < 0.3
        if low_conf.sum() > 5:
            low_conf_err = np.mean(y_val[low_conf] == 0)
            metrics['action_wrong_low_conf'] = float(low_conf_err)
            metrics['action_wrong_low_conf_coverage'] = float(low_conf.mean())
            print(f"    Low-confidence (<0.3) error rate: {low_conf_err:.1%} "
                  f"(coverage: {low_conf.mean():.1%})")

        # 4b: Break direction correctness
        print("\n  Training: bd_correct (binary, will physics BD be right?)...")
        y_train = train_residuals['bd_correct']
        y_val = val_residuals['bd_correct']
        train_ds = lgb.Dataset(X_train_aug, label=y_train, feature_name=aug_names)
        val_ds = lgb.Dataset(X_val_aug, label=y_val, feature_name=aug_names, reference=train_ds)

        model = lgb.train(
            params, train_ds, num_boost_round=400,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(stopping_rounds=40), lgb.log_evaluation(0)],
        )
        self.models['bd_correct'] = model

        val_pred_prob = model.predict(X_val_aug)
        val_pred = (val_pred_prob > 0.5).astype(int)
        acc = np.mean(val_pred == y_val)
        metrics['bd_correct_acc'] = float(acc)
        print(f"    Val accuracy: {acc:.1%}")

        # 4c: Confidence scale (regression: how much to adjust)
        print("\n  Training: confidence_scale (regression, adjustment factor)...")
        y_train = train_residuals['confidence_scale']
        y_val = val_residuals['confidence_scale']
        train_ds = lgb.Dataset(X_train_aug, label=y_train, feature_name=aug_names)
        val_ds = lgb.Dataset(X_val_aug, label=y_val, feature_name=aug_names, reference=train_ds)

        reg_params = {
            'objective': 'mae', 'metric': 'mae',
            'num_leaves': 24, 'learning_rate': 0.05,
            'min_child_samples': 10, 'feature_fraction': 0.8,
            'verbose': -1,
        }
        model = lgb.train(
            reg_params, train_ds, num_boost_round=400,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(stopping_rounds=40), lgb.log_evaluation(0)],
        )
        self.models['confidence_scale'] = model

        val_pred = model.predict(X_val_aug)
        mae = np.mean(np.abs(val_pred - y_val))
        metrics['confidence_scale_mae'] = float(mae)
        print(f"    Val MAE: {mae:.4f}")
        print(f"    Val pred range: [{val_pred.min():.3f}, {val_pred.max():.3f}]")
        print(f"    Val actual range: [{y_val.min():.3f}, {y_val.max():.3f}]")

        # 4d: Lifetime error (regression)
        print("\n  Training: lifetime_error (regression, actual - predicted)...")
        y_train = train_residuals['lifetime_error']
        y_val = val_residuals['lifetime_error']
        train_ds = lgb.Dataset(X_train_aug, label=y_train, feature_name=aug_names)
        val_ds = lgb.Dataset(X_val_aug, label=y_val, feature_name=aug_names, reference=train_ds)

        model = lgb.train(
            reg_params, train_ds, num_boost_round=400,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(stopping_rounds=40), lgb.log_evaluation(0)],
        )
        self.models['lifetime_error'] = model

        val_pred = model.predict(X_val_aug)
        mae = np.mean(np.abs(val_pred - y_val))
        metrics['lifetime_error_mae'] = float(mae)
        print(f"    Val MAE: {mae:.1f} bars")

        # Corrected lifetime accuracy: physics_lifetime + predicted_error vs actual
        corrected_lifetime = val_phys['implied_lifetime'] + val_pred
        raw_mae = np.mean(np.abs(val_phys['implied_lifetime'] - Y_val['channel_lifetime']))
        corrected_mae = np.mean(np.abs(corrected_lifetime - Y_val['channel_lifetime']))
        metrics['raw_lifetime_mae'] = float(raw_mae)
        metrics['corrected_lifetime_mae'] = float(corrected_mae)
        improvement = (raw_mae - corrected_mae) / raw_mae * 100
        metrics['lifetime_improvement_pct'] = float(improvement)
        print(f"    Raw physics lifetime MAE: {raw_mae:.1f} bars")
        print(f"    Corrected lifetime MAE: {corrected_mae:.1f} bars "
              f"({improvement:+.1f}% improvement)")

        # Feature importance (from action_correct model, most impactful)
        imp = self.models['action_correct'].feature_importance(importance_type='gain')
        sorted_idx = np.argsort(imp)[::-1]
        print("\n  Top 15 features for predicting physics correctness:")
        for rank, i in enumerate(sorted_idx[:15]):
            print(f"    {rank+1}. {aug_names[i]}: {imp[i]:.0f}")

        self.feature_importance = [(aug_names[i], float(imp[i])) for i in sorted_idx]

        return metrics

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict physics residuals.

        Returns:
            action_trustworthy: P(physics action is correct) [0, 1]
            bd_trustworthy: P(physics BD prediction is correct) [0, 1]
            confidence_scale: multiply physics confidence by this [0.3, 1.5]
            lifetime_correction: add to physics lifetime estimate
        """
        # Derive physics predictions
        physics_preds = self.derive_physics_prediction(X, self.feature_names)

        # Augment features
        X_aug, _ = self.augment_features(X, physics_preds, self.feature_names)

        results = {}

        if 'action_correct' in self.models:
            results['action_trustworthy'] = self.models['action_correct'].predict(X_aug)

        if 'bd_correct' in self.models:
            results['bd_trustworthy'] = self.models['bd_correct'].predict(X_aug)

        if 'confidence_scale' in self.models:
            raw_scale = self.models['confidence_scale'].predict(X_aug)
            results['confidence_scale'] = np.clip(raw_scale, 0.3, 1.5)

        if 'lifetime_error' in self.models:
            results['lifetime_correction'] = self.models['lifetime_error'].predict(X_aug)

        # Also pass through derived physics predictions for backtest use
        results['implied_action'] = physics_preds['implied_action']
        results['implied_confidence'] = physics_preds['implied_confidence']
        results['implied_break_dir'] = physics_preds['implied_break_dir']
        results['implied_lifetime'] = physics_preds['implied_lifetime']

        return results

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'feature_names': self.feature_names,
                'augmented_names': self.augmented_names,
                'physics_stats': self.physics_stats,
                'feature_importance': getattr(self, 'feature_importance', []),
            }, f)
        print(f"  Saved PhysicsResidualModel to {path}")

    @classmethod
    def load(cls, path: str) -> 'PhysicsResidualModel':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.models = data['models']
        model.feature_names = data['feature_names']
        model.augmented_names = data.get('augmented_names')
        model.physics_stats = data.get('physics_stats', {})
        model.feature_importance = data.get('feature_importance', [])
        return model


# ---------------------------------------------------------------------------
# Architecture 11: Adverse Movement Predictor (Stop-Loss Avoidance)
# ---------------------------------------------------------------------------

class AdverseMovementPredictor:
    """
    Predicts probability of adverse movement — will the trade hit stop before TP?

    Insight: In the backtest, stop-loss exits have 0% WR and are pure losses.
    If we can predict when a trade is likely to stop out, we can either:
    1. Skip the trade entirely
    2. Widen the stop (at the cost of larger potential loss)
    3. Reduce position size

    Training: For each bar, simulate both BUY and SELL trades with standard
    stop/TP distances. Track which hits first. Train a classifier on
    P(stop_hit_first | features).

    Additional targets:
    - max_adverse_excursion: worst drawdown before recovery (regression)
    - time_to_adverse: how quickly the adverse move happens (regression)
    - recovery_probability: if adverse, probability of recovery (binary)
    """

    # Standard stop/TP distances to simulate (in % of price)
    STOP_DISTANCES = [0.003, 0.005, 0.008]  # 0.3%, 0.5%, 0.8%
    TP_DISTANCES = [0.006, 0.010, 0.015]     # 0.6%, 1.0%, 1.5%
    LOOKFORWARD = 40  # 40 bars (3+ hours) to evaluate outcome

    def __init__(self):
        self.models = {}  # {target_name: lgb.Booster}
        self.feature_names = None
        self.calibration = {}

    @staticmethod
    def simulate_trade_outcomes(
        closes: np.ndarray,
        bar_indices: np.ndarray,
        stop_pct: float = 0.005,
        tp_pct: float = 0.010,
        lookforward: int = 40,
    ) -> Dict[str, np.ndarray]:
        """
        For each bar, simulate BUY and SELL trades and track outcome.

        Returns arrays for each bar:
        - buy_stop_first: 1 if stop hit before TP for BUY trade
        - sell_stop_first: 1 if stop hit before TP for SELL trade
        - buy_mae: maximum adverse excursion for BUY (worst drawdown %)
        - sell_mae: maximum adverse excursion for SELL
        - buy_mfe: maximum favorable excursion for BUY (best unrealized gain %)
        - sell_mfe: maximum favorable excursion for SELL
        - buy_bars_to_adverse: how quickly worst drawdown happens
        - sell_bars_to_adverse: how quickly worst drawdown happens
        """
        n = len(bar_indices)
        total_bars = len(closes)

        results = {
            'buy_stop_first': np.zeros(n),
            'sell_stop_first': np.zeros(n),
            'buy_mae': np.zeros(n),
            'sell_mae': np.zeros(n),
            'buy_mfe': np.zeros(n),
            'sell_mfe': np.zeros(n),
            'buy_bars_to_adverse': np.zeros(n),
            'sell_bars_to_adverse': np.zeros(n),
        }

        for i, bar in enumerate(bar_indices):
            entry_price = closes[bar]
            end_bar = min(bar + lookforward, total_bars)
            future_prices = closes[bar + 1:end_bar]

            if len(future_prices) < 3:
                continue

            # BUY trade simulation
            buy_stop = entry_price * (1 - stop_pct)
            buy_tp = entry_price * (1 + tp_pct)

            buy_worst = entry_price
            buy_best = entry_price
            buy_stopped = False
            buy_tped = False
            buy_worst_bar = 0

            for j, price in enumerate(future_prices):
                if price < buy_worst:
                    buy_worst = price
                    buy_worst_bar = j + 1
                if price > buy_best:
                    buy_best = price
                if price <= buy_stop and not buy_tped:
                    buy_stopped = True
                    break
                if price >= buy_tp and not buy_stopped:
                    buy_tped = True
                    break

            results['buy_mae'][i] = (entry_price - buy_worst) / entry_price
            results['buy_mfe'][i] = (buy_best - entry_price) / entry_price
            results['buy_bars_to_adverse'][i] = buy_worst_bar
            results['buy_stop_first'][i] = 1.0 if buy_stopped else 0.0

            # SELL trade simulation
            sell_stop = entry_price * (1 + stop_pct)
            sell_tp = entry_price * (1 - tp_pct)

            sell_worst = entry_price
            sell_best = entry_price
            sell_stopped = False
            sell_tped = False
            sell_worst_bar = 0

            for j, price in enumerate(future_prices):
                if price > sell_worst:
                    sell_worst = price
                    sell_worst_bar = j + 1
                if price < sell_best:
                    sell_best = price
                if price >= sell_stop and not sell_tped:
                    sell_stopped = True
                    break
                if price <= sell_tp and not sell_stopped:
                    sell_tped = True
                    break

            results['sell_mae'][i] = (sell_worst - entry_price) / entry_price
            results['sell_mfe'][i] = (entry_price - sell_best) / entry_price
            results['sell_bars_to_adverse'][i] = sell_worst_bar
            results['sell_stop_first'][i] = 1.0 if sell_stopped else 0.0

        return results

    def train(
        self,
        X_train: np.ndarray, Y_train: Dict[str, np.ndarray],
        X_val: np.ndarray, Y_val: Dict[str, np.ndarray],
        feature_names: List[str],
        closes_train: Optional[np.ndarray] = None,
        bar_indices_train: Optional[np.ndarray] = None,
        closes_val: Optional[np.ndarray] = None,
        bar_indices_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Train the adverse movement predictor.

        If closes/bar_indices are provided, simulates actual trades.
        Otherwise, uses forward returns to approximate MFE/MAE.
        """
        import lightgbm as lgb

        self.feature_names = feature_names
        metrics = {}

        # Compute trade outcomes from forward returns
        # Use multi-target approach with cleaner signals
        print("\n  Computing trade outcomes from forward returns...")

        n_train = len(X_train)
        n_val = len(X_val)

        train_ret5 = Y_train['future_return_5']
        train_ret20 = Y_train['future_return_20']
        train_ret60 = Y_train['future_return_60']

        val_ret5 = Y_val['future_return_5']
        val_ret20 = Y_val['future_return_20']
        val_ret60 = Y_val['future_return_60']

        # Target 1: Quick adverse movement (5-bar return goes against you)
        # For BUY: ret_5 < -0.003 (price drops >0.3% in first 25 min)
        # For SELL: ret_5 > 0.003 (price rises >0.3% in first 25 min)
        ADVERSE_THRESH = 0.003

        train_buy_adverse = (train_ret5 < -ADVERSE_THRESH).astype(float)
        train_sell_adverse = (train_ret5 > ADVERSE_THRESH).astype(float)
        val_buy_adverse = (val_ret5 < -ADVERSE_THRESH).astype(float)
        val_sell_adverse = (val_ret5 > ADVERSE_THRESH).astype(float)

        # Target 2: Trade viability (20-bar return in the right direction)
        # For BUY: ret_20 > 0 and |ret_20| > 0.001
        # For SELL: ret_20 < 0 and |ret_20| > 0.001
        train_buy_viable = (train_ret20 > 0.001).astype(float)
        train_sell_viable = (train_ret20 < -0.001).astype(float)
        val_buy_viable = (val_ret20 > 0.001).astype(float)
        val_sell_viable = (val_ret20 < -0.001).astype(float)

        # Target 3: Worst-case return ratio (how bad is the worst return relative to best)
        # risk_ratio = min(ret5,ret20,ret60) / max(ret5,ret20,ret60)
        # For BUY: negative ratio = adverse excursion dominates
        buy_best = np.maximum(train_ret5, np.maximum(train_ret20, train_ret60))
        buy_worst = np.minimum(train_ret5, np.minimum(train_ret20, train_ret60))
        sell_best = np.abs(np.minimum(train_ret5, np.minimum(train_ret20, train_ret60)))
        sell_worst = np.maximum(train_ret5, np.maximum(train_ret20, train_ret60))

        print(f"  BUY adverse (5-bar) rate: train={train_buy_adverse.mean():.1%}, val={val_buy_adverse.mean():.1%}")
        print(f"  SELL adverse (5-bar) rate: train={train_sell_adverse.mean():.1%}, val={val_sell_adverse.mean():.1%}")
        print(f"  BUY viable (20-bar) rate: train={train_buy_viable.mean():.1%}, val={val_buy_viable.mean():.1%}")
        print(f"  SELL viable (20-bar) rate: train={train_sell_viable.mean():.1%}, val={val_sell_viable.mean():.1%}")

        # Combine BUY and SELL samples: features + direction indicator
        X_train_combined = np.vstack([
            np.column_stack([X_train, np.ones(n_train)]),   # BUY
            np.column_stack([X_train, np.zeros(n_train)]),  # SELL
        ])
        X_val_combined = np.vstack([
            np.column_stack([X_val, np.ones(n_val)]),
            np.column_stack([X_val, np.zeros(n_val)]),
        ])

        # Combine targets
        y_train_adverse = np.concatenate([train_buy_adverse, train_sell_adverse])
        y_val_adverse = np.concatenate([val_buy_adverse, val_sell_adverse])
        y_train_viable = np.concatenate([train_buy_viable, train_sell_viable])
        y_val_viable = np.concatenate([val_buy_viable, val_sell_viable])

        # Combined stop target: adverse AND NOT viable
        y_train_stop = ((y_train_adverse > 0.5) & (y_train_viable < 0.5)).astype(float)
        y_val_stop = ((y_val_adverse > 0.5) & (y_val_viable < 0.5)).astype(float)

        combined_names = feature_names + ['is_buy']

        print(f"  Training samples: {len(X_train_combined)} ({n_train} BUY + {n_train} SELL)")
        print(f"  Val samples: {len(X_val_combined)}")
        print(f"  Combined stop-out rate: train={y_train_stop.mean():.1%}, val={y_val_stop.mean():.1%}")

        # --- Model 1: Stop-out classifier (binary: will trade hit stop?) ---
        print("\n  Training: stop_out_classifier (will trade hit stop?)...")
        train_ds = lgb.Dataset(X_train_combined, label=y_train_stop, feature_name=combined_names)
        val_ds = lgb.Dataset(X_val_combined, label=y_val_stop, feature_name=combined_names, reference=train_ds)

        params = {
            'objective': 'binary', 'metric': 'auc',
            'num_leaves': 28, 'learning_rate': 0.05,
            'min_child_samples': 12, 'feature_fraction': 0.8,
            'bagging_fraction': 0.9, 'bagging_freq': 5,
            'verbose': -1,
        }
        model = lgb.train(
            params, train_ds, num_boost_round=400,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(stopping_rounds=40), lgb.log_evaluation(0)],
        )
        self.models['stop_out'] = model

        val_pred_prob = model.predict(X_val_combined)
        val_pred = (val_pred_prob > 0.5).astype(int)
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y_val_stop, val_pred_prob)
        except Exception:
            auc = 0.5
        acc = np.mean(val_pred == y_val_stop)
        metrics['stop_out_auc'] = float(auc)
        metrics['stop_out_acc'] = float(acc)
        print(f"    Val AUC: {auc:.3f}")
        print(f"    Val accuracy: {acc:.1%}")

        # Calibration: when model says safe (<0.3 stop prob), actual stop rate?
        safe_mask = val_pred_prob < 0.3
        if safe_mask.sum() > 5:
            safe_stop_rate = y_val_stop[safe_mask].mean()
            metrics['safe_signal_stop_rate'] = float(safe_stop_rate)
            metrics['safe_signal_coverage'] = float(safe_mask.mean())
            print(f"    Safe signals (P(stop)<0.3): stop_rate={safe_stop_rate:.1%}, "
                  f"coverage={safe_mask.mean():.1%}")

        # When model says dangerous (>0.7 stop prob), actual stop rate?
        danger_mask = val_pred_prob > 0.7
        if danger_mask.sum() > 5:
            danger_stop_rate = y_val_stop[danger_mask].mean()
            metrics['danger_signal_stop_rate'] = float(danger_stop_rate)
            metrics['danger_signal_coverage'] = float(danger_mask.mean())
            print(f"    Danger signals (P(stop)>0.7): stop_rate={danger_stop_rate:.1%}, "
                  f"coverage={danger_mask.mean():.1%}")

        # --- Model 2: Viability classifier (will trade be profitable?) ---
        print("\n  Training: viability_classifier (will trade be profitable?)...")
        train_ds = lgb.Dataset(X_train_combined, label=y_train_viable, feature_name=combined_names)
        val_ds = lgb.Dataset(X_val_combined, label=y_val_viable, feature_name=combined_names, reference=train_ds)

        model = lgb.train(
            params, train_ds, num_boost_round=400,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(stopping_rounds=40), lgb.log_evaluation(0)],
        )
        self.models['viable'] = model

        val_viable_pred = model.predict(X_val_combined)
        val_viable_cls = (val_viable_pred > 0.5).astype(int)
        try:
            viable_auc = roc_auc_score(y_val_viable, val_viable_pred)
        except Exception:
            viable_auc = 0.5
        viable_acc = np.mean(val_viable_cls == y_val_viable)
        metrics['viable_auc'] = float(viable_auc)
        metrics['viable_acc'] = float(viable_acc)
        print(f"    Val AUC: {viable_auc:.3f}")
        print(f"    Val accuracy: {viable_acc:.1%}")

        # When model says "viable" (>0.7), what fraction actually are?
        high_viable = val_viable_pred > 0.7
        if high_viable.sum() > 5:
            hv_acc = np.mean(y_val_viable[high_viable])
            metrics['high_viable_precision'] = float(hv_acc)
            metrics['high_viable_coverage'] = float(high_viable.mean())
            print(f"    High viability (>0.7): precision={hv_acc:.1%}, coverage={high_viable.mean():.1%}")

        # --- Model 3: Adverse return regression (how bad is the 5-bar return?) ---
        print("\n  Training: adverse_return_regressor (expected 5-bar adverse %)...")
        # Use abs(ret_5) for adverse direction
        train_adverse_mag = np.concatenate([
            np.abs(np.minimum(train_ret5, 0)) * 100,  # BUY adverse = negative returns
            np.maximum(train_ret5, 0) * 100,           # SELL adverse = positive returns
        ])
        val_adverse_mag = np.concatenate([
            np.abs(np.minimum(val_ret5, 0)) * 100,
            np.maximum(val_ret5, 0) * 100,
        ])

        train_ds = lgb.Dataset(X_train_combined, label=train_adverse_mag,
                               feature_name=combined_names)
        val_ds = lgb.Dataset(X_val_combined, label=val_adverse_mag,
                             feature_name=combined_names, reference=train_ds)

        reg_params = {
            'objective': 'mae', 'metric': 'mae',
            'num_leaves': 24, 'learning_rate': 0.05,
            'min_child_samples': 10, 'verbose': -1,
        }
        model = lgb.train(
            reg_params, train_ds, num_boost_round=400,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(stopping_rounds=40), lgb.log_evaluation(0)],
        )
        self.models['adverse_regressor'] = model

        val_pred_adverse = model.predict(X_val_combined)
        adverse_error = np.mean(np.abs(val_pred_adverse - val_adverse_mag))
        metrics['adverse_return_mae'] = float(adverse_error)
        print(f"    Val MAE: {adverse_error:.4f}%")
        print(f"    Predicted range: [{val_pred_adverse.min():.4f}%, {val_pred_adverse.max():.4f}%]")
        print(f"    Actual range: [{val_adverse_mag.min():.4f}%, {val_adverse_mag.max():.4f}%]")

        # --- Model 4: Favorable return regression ---
        print("\n  Training: favorable_return_regressor (expected 20-bar return %)...")
        train_favorable = np.concatenate([
            np.maximum(train_ret20, 0) * 100,  # BUY favorable = positive 20-bar return
            np.abs(np.minimum(train_ret20, 0)) * 100,  # SELL favorable = negative 20-bar return
        ])
        val_favorable = np.concatenate([
            np.maximum(val_ret20, 0) * 100,
            np.abs(np.minimum(val_ret20, 0)) * 100,
        ])

        train_ds = lgb.Dataset(X_train_combined, label=train_favorable,
                               feature_name=combined_names)
        val_ds = lgb.Dataset(X_val_combined, label=val_favorable,
                             feature_name=combined_names, reference=train_ds)

        model = lgb.train(
            reg_params, train_ds, num_boost_round=400,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(stopping_rounds=40), lgb.log_evaluation(0)],
        )
        self.models['favorable_regressor'] = model

        val_pred_fav = model.predict(X_val_combined)
        fav_error = np.mean(np.abs(val_pred_fav - val_favorable))
        metrics['favorable_return_mae'] = float(fav_error)
        print(f"    Val MAE: {fav_error:.4f}%")

        # Derived: favorable/adverse ratio (expected reward/risk)
        predicted_rr = val_pred_fav / np.clip(val_pred_adverse, 0.01, None)
        actual_rr = val_favorable / np.clip(val_adverse_mag, 0.01, None)
        rr_corr = np.corrcoef(predicted_rr, actual_rr)[0, 1]
        metrics['rr_correlation'] = float(rr_corr) if not np.isnan(rr_corr) else 0.0
        print(f"    Predicted R:R correlation with actual: {metrics['rr_correlation']:.3f}")

        # Feature importance
        imp = self.models['stop_out'].feature_importance(importance_type='gain')
        sorted_idx = np.argsort(imp)[::-1]
        print("\n  Top 15 features for predicting stop-outs:")
        for rank, i in enumerate(sorted_idx[:15]):
            print(f"    {rank+1}. {combined_names[i]}: {imp[i]:.0f}")

        self.feature_importance = [(combined_names[i], float(imp[i])) for i in sorted_idx]
        self.calibration = {
            'train_stop_rate': float(y_train_stop.mean()),
            'val_stop_rate': float(y_val_stop.mean()),
        }

        return metrics

    def predict(self, X: np.ndarray, is_buy: bool = True) -> Dict[str, np.ndarray]:
        """
        Predict adverse movement probability for a trade.

        Args:
            X: feature matrix (N, num_features)
            is_buy: True for BUY trades, False for SELL

        Returns:
            stop_prob: P(will hit stop loss) [0, 1]
            expected_mae: predicted maximum adverse excursion (%)
            expected_mfe: predicted maximum favorable excursion (%)
            risk_reward: predicted MFE/MAE ratio
        """
        n = len(X)
        direction_col = np.ones(n) if is_buy else np.zeros(n)
        X_aug = np.column_stack([X, direction_col])

        results = {}

        if 'stop_out' in self.models:
            results['stop_prob'] = self.models['stop_out'].predict(X_aug)

        if 'viable' in self.models:
            results['viable_prob'] = self.models['viable'].predict(X_aug)

        if 'adverse_regressor' in self.models:
            results['expected_adverse'] = self.models['adverse_regressor'].predict(X_aug) / 100.0

        if 'favorable_regressor' in self.models:
            results['expected_favorable'] = self.models['favorable_regressor'].predict(X_aug) / 100.0

        if 'expected_adverse' in results and 'expected_favorable' in results:
            results['risk_reward'] = results['expected_favorable'] / np.clip(
                results['expected_adverse'], 0.0001, None)

        return results

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'feature_names': self.feature_names,
                'calibration': self.calibration,
                'feature_importance': getattr(self, 'feature_importance', []),
            }, f)
        print(f"  Saved AdverseMovementPredictor to {path}")

    @classmethod
    def load(cls, path: str) -> 'AdverseMovementPredictor':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.models = data['models']
        model.feature_names = data['feature_names']
        model.calibration = data.get('calibration', {})
        model.feature_importance = data.get('feature_importance', [])
        return model


# ---------------------------------------------------------------------------
# Architecture 12: Entry Timing Optimizer
# ---------------------------------------------------------------------------

class EntryTimingOptimizer:
    """
    Predicts whether NOW is the best entry or if waiting 1-5 bars yields better.

    Insight: In winning trades, avg MAE = 0.331% — that's money left on the table
    from suboptimal entry timing. If we could enter 1-3 bars later at a better
    price, that's 0.1-0.3% improvement per trade.

    Targets:
    1. immediate_best: binary — is current bar's price within 0.1% of the
       best entry price in the next 5 bars?
    2. improvement_pct: regression — how much better could we do by waiting?
    """

    LOOKAHEAD = 5

    def __init__(self):
        self.models = {}
        self.feature_names = None

    def train(
        self,
        X_train: np.ndarray, Y_train: Dict[str, np.ndarray],
        X_val: np.ndarray, Y_val: Dict[str, np.ndarray],
        feature_names: List[str],
    ) -> Dict[str, float]:
        """Train the entry timing optimizer."""
        import lightgbm as lgb

        self.feature_names = feature_names
        metrics = {}
        n_train = len(X_train)
        n_val = len(X_val)

        print("\n  Computing entry timing targets...")

        train_ret5 = Y_train['future_return_5']
        train_ret20 = Y_train['future_return_20']
        val_ret5 = Y_val['future_return_5']
        val_ret20 = Y_val['future_return_20']

        # BUY: improvement = how much price drops in next 5 bars (better buy price)
        train_buy_improve = np.maximum(-train_ret5, 0) * 100
        train_sell_improve = np.maximum(train_ret5, 0) * 100
        val_buy_improve = np.maximum(-val_ret5, 0) * 100
        val_sell_improve = np.maximum(val_ret5, 0) * 100

        # immediate_best: improvement < 0.1%
        IMMEDIATE_THRESH = 0.10
        train_buy_immediate = (train_buy_improve < IMMEDIATE_THRESH).astype(float)
        train_sell_immediate = (train_sell_improve < IMMEDIATE_THRESH).astype(float)
        val_buy_immediate = (val_buy_improve < IMMEDIATE_THRESH).astype(float)
        val_sell_immediate = (val_sell_improve < IMMEDIATE_THRESH).astype(float)

        print(f"  BUY immediate-best rate: train={train_buy_immediate.mean():.1%}, val={val_buy_immediate.mean():.1%}")
        print(f"  SELL immediate-best rate: train={train_sell_immediate.mean():.1%}, val={val_sell_immediate.mean():.1%}")
        print(f"  BUY avg improvement: train={train_buy_improve.mean():.3f}%, val={val_buy_improve.mean():.3f}%")
        print(f"  SELL avg improvement: train={train_sell_improve.mean():.3f}%, val={val_sell_improve.mean():.3f}%")

        X_train_combined = np.vstack([
            np.column_stack([X_train, np.ones(n_train)]),
            np.column_stack([X_train, np.zeros(n_train)]),
        ])
        X_val_combined = np.vstack([
            np.column_stack([X_val, np.ones(n_val)]),
            np.column_stack([X_val, np.zeros(n_val)]),
        ])

        y_train_immediate = np.concatenate([train_buy_immediate, train_sell_immediate])
        y_val_immediate = np.concatenate([val_buy_immediate, val_sell_immediate])
        y_train_improve = np.concatenate([train_buy_improve, train_sell_improve])
        y_val_improve = np.concatenate([val_buy_improve, val_sell_improve])

        combined_names = feature_names + ['is_buy']

        print(f"  Combined: train={len(X_train_combined)}, val={len(X_val_combined)}")
        print(f"  Immediate-best rate: train={y_train_immediate.mean():.1%}, val={y_val_immediate.mean():.1%}")

        # Model 1: Is NOW the best entry? (binary)
        print("\n  Training: immediate_best (is NOW optimal?)...")
        train_ds = lgb.Dataset(X_train_combined, label=y_train_immediate, feature_name=combined_names)
        val_ds = lgb.Dataset(X_val_combined, label=y_val_immediate, feature_name=combined_names, reference=train_ds)

        params = {
            'objective': 'binary', 'metric': 'auc',
            'num_leaves': 28, 'learning_rate': 0.05,
            'min_child_samples': 12, 'feature_fraction': 0.8,
            'bagging_fraction': 0.9, 'bagging_freq': 5,
            'verbose': -1,
        }
        model = lgb.train(
            params, train_ds, num_boost_round=400,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(stopping_rounds=40), lgb.log_evaluation(0)],
        )
        self.models['immediate_best'] = model

        val_pred_prob = model.predict(X_val_combined)
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y_val_immediate, val_pred_prob)
        except Exception:
            auc = 0.5
        acc = np.mean((val_pred_prob > 0.5).astype(int) == y_val_immediate)
        metrics['immediate_best_auc'] = float(auc)
        metrics['immediate_best_acc'] = float(acc)
        print(f"    Val AUC: {auc:.3f}")
        print(f"    Val accuracy: {acc:.1%}")

        enter_now = val_pred_prob > 0.7
        if enter_now.sum() > 5:
            actual_good = y_val_immediate[enter_now].mean()
            metrics['enter_now_precision'] = float(actual_good)
            metrics['enter_now_coverage'] = float(enter_now.mean())
            print(f"    'Enter now' (>0.7): precision={actual_good:.1%}, "
                  f"coverage={enter_now.mean():.1%}")

        wait_mask = val_pred_prob < 0.3
        if wait_mask.sum() > 5:
            avg_improve = y_val_improve[wait_mask].mean()
            metrics['wait_avg_improvement'] = float(avg_improve)
            metrics['wait_coverage'] = float(wait_mask.mean())
            print(f"    'Wait' (<0.3): avg improve={avg_improve:.3f}%, "
                  f"coverage={wait_mask.mean():.1%}")

        # Model 2: Expected improvement by waiting (regression)
        print("\n  Training: improvement_pct (how much better by waiting?)...")
        train_ds = lgb.Dataset(X_train_combined, label=y_train_improve, feature_name=combined_names)
        val_ds = lgb.Dataset(X_val_combined, label=y_val_improve, feature_name=combined_names, reference=train_ds)

        reg_params = {
            'objective': 'mae', 'metric': 'mae',
            'num_leaves': 24, 'learning_rate': 0.05,
            'min_child_samples': 10, 'verbose': -1,
        }
        model = lgb.train(
            reg_params, train_ds, num_boost_round=400,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(stopping_rounds=40), lgb.log_evaluation(0)],
        )
        self.models['improvement'] = model

        val_pred_improve = model.predict(X_val_combined)
        imp_error = np.mean(np.abs(val_pred_improve - y_val_improve))
        metrics['improvement_mae'] = float(imp_error)
        print(f"    Val MAE: {imp_error:.4f}%")
        corr = np.corrcoef(val_pred_improve, y_val_improve)[0, 1]
        metrics['improvement_correlation'] = float(corr) if not np.isnan(corr) else 0.0
        print(f"    Correlation: {metrics['improvement_correlation']:.3f}")

        # Feature importance
        imp_arr = self.models['immediate_best'].feature_importance(importance_type='gain')
        sorted_idx = np.argsort(imp_arr)[::-1]
        print("\n  Top 15 features for entry timing:")
        for rank, i in enumerate(sorted_idx[:15]):
            print(f"    {rank+1}. {combined_names[i]}: {imp_arr[i]:.0f}")
        self.feature_importance = [(combined_names[i], float(imp_arr[i])) for i in sorted_idx]

        return metrics

    def predict(self, X: np.ndarray, is_buy: bool = True) -> Dict[str, np.ndarray]:
        """Predict entry timing quality."""
        n = len(X)
        direction_col = np.ones(n) if is_buy else np.zeros(n)
        X_aug = np.column_stack([X, direction_col])

        results = {}
        if 'immediate_best' in self.models:
            results['immediate_best_prob'] = self.models['immediate_best'].predict(X_aug)
        if 'improvement' in self.models:
            results['expected_improvement'] = self.models['improvement'].predict(X_aug) / 100.0
        return results

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'feature_names': self.feature_names,
                'feature_importance': getattr(self, 'feature_importance', []),
            }, f)
        print(f"  Saved EntryTimingOptimizer to {path}")

    @classmethod
    def load(cls, path: str) -> 'EntryTimingOptimizer':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.models = data['models']
        model.feature_names = data['feature_names']
        model.feature_importance = data.get('feature_importance', [])
        return model


# ---------------------------------------------------------------------------
# Architecture 13: Composite Signal Scorer (Final Meta-Meta Model)
# ---------------------------------------------------------------------------

class CompositeSignalScorer:
    """
    Final-stage composite scorer that takes ALL model outputs and learns
    the optimal combination. Unlike the stacking ensemble (Arch 5) which
    only uses 4 base models, this uses all 11+ model outputs.

    The current integration uses hand-tuned multipliers (1.15x, 0.75x, etc.).
    This model learns the optimal nonlinear combination.

    Meta-features:
    - GBT: action_probs, bd_probs, lifetime
    - Regime: regime_id, regime_probs
    - TrendGBT: break_dir
    - CV Ensemble: bd_consensus, action_consensus
    - Physics Residual: action_trustworthy, confidence_scale, lifetime_correction
    - Adverse Movement: stop_prob, viable_prob
    - Key physics features: break_prob_max, avg_entropy, direction_consensus

    Target: optimal_action (3-class) with emphasis on accuracy
    """

    KEY_PHYSICS_FEATURES = [
        'break_prob_max', 'break_prob_weighted', 'avg_entropy',
        'direction_consensus', 'health_min', 'health_max',
        'confluence_score', 'atr_pct', 'rsi_14',
    ]

    def __init__(self):
        self.model = None
        self.meta_feature_names = None
        self.feature_names = None  # Base feature names (for model loading)

    def _collect_meta_features(
        self,
        X: np.ndarray,
        feature_names: List[str],
        model_dir: str,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Run all models and collect their outputs as meta-features.
        Falls back to zeros if a model isn't available.
        """
        idx = {name: i for i, name in enumerate(feature_names)}
        n = len(X)

        meta_features = []
        meta_names = []

        # Key physics features (passthrough)
        for feat in self.KEY_PHYSICS_FEATURES:
            if feat in idx:
                meta_features.append(X[:, idx[feat]])
            else:
                meta_features.append(np.zeros(n))
            meta_names.append(f'phys_{feat}')

        # GBT predictions
        gbt_path = os.path.join(model_dir, 'gbt_model.pkl')
        if os.path.exists(gbt_path):
            try:
                gbt = GBTModel.load(gbt_path)
                gbt_pred = gbt.predict(X)
                if 'action_probs' in gbt_pred:
                    for c in range(3):
                        meta_features.append(gbt_pred['action_probs'][:, c])
                        meta_names.append(f'gbt_action_prob_{c}')
                elif 'action' in gbt_pred:
                    meta_features.append(gbt_pred['action'].astype(float))
                    meta_names.append('gbt_action')
                if 'break_dir' in gbt_pred:
                    meta_features.append(gbt_pred['break_dir'].astype(float))
                    meta_names.append('gbt_break_dir')
                if 'lifetime' in gbt_pred:
                    meta_features.append(gbt_pred['lifetime'])
                    meta_names.append('gbt_lifetime')
            except Exception:
                pass

        # Regime predictions
        regime_path = os.path.join(model_dir, 'regime_model.pkl')
        if os.path.exists(regime_path):
            try:
                regime = RegimeConditionalModel.load(regime_path)
                reg_pred = regime.predict(X)
                meta_features.append(reg_pred['regime'].astype(float))
                meta_names.append('regime_id')
                if 'action' in reg_pred:
                    meta_features.append(reg_pred['action'].astype(float))
                    meta_names.append('regime_action')
            except Exception:
                pass

        # CV Ensemble
        cv_path = os.path.join(model_dir, 'cv_ensemble_model.pkl')
        if os.path.exists(cv_path):
            try:
                cv = CVEnsembleModel.load(cv_path)
                cv_pred = cv.predict(X)
                meta_features.append(cv_pred['bd_consensus'])
                meta_names.append('cv_bd_consensus')
                meta_features.append(cv_pred['action_consensus'])
                meta_names.append('cv_action_consensus')
                meta_features.append(cv_pred['break_dir'].astype(float))
                meta_names.append('cv_break_dir')
                meta_features.append(cv_pred['action'].astype(float))
                meta_names.append('cv_action')
            except Exception:
                pass

        # Physics Residual
        res_path = os.path.join(model_dir, 'physics_residual_model.pkl')
        if os.path.exists(res_path):
            try:
                residual = PhysicsResidualModel.load(res_path)
                res_pred = residual.predict(X)
                if 'action_trustworthy' in res_pred:
                    meta_features.append(res_pred['action_trustworthy'])
                    meta_names.append('residual_action_trust')
                if 'confidence_scale' in res_pred:
                    meta_features.append(res_pred['confidence_scale'])
                    meta_names.append('residual_conf_scale')
                if 'lifetime_correction' in res_pred:
                    meta_features.append(res_pred['lifetime_correction'])
                    meta_names.append('residual_lt_correction')
            except Exception:
                pass

        # Adverse Movement (BUY and SELL)
        adv_path = os.path.join(model_dir, 'adverse_movement_model.pkl')
        if os.path.exists(adv_path):
            try:
                adv = AdverseMovementPredictor.load(adv_path)
                for direction, is_buy in [('buy', True), ('sell', False)]:
                    adv_pred = adv.predict(X, is_buy=is_buy)
                    if 'stop_prob' in adv_pred:
                        meta_features.append(adv_pred['stop_prob'])
                        meta_names.append(f'adv_{direction}_stop_prob')
                    if 'viable_prob' in adv_pred:
                        meta_features.append(adv_pred['viable_prob'])
                        meta_names.append(f'adv_{direction}_viable_prob')
            except Exception:
                pass

        meta_X = np.column_stack(meta_features) if meta_features else np.zeros((n, 1))
        return meta_X, meta_names

    def train(
        self,
        X_train: np.ndarray, Y_train: Dict[str, np.ndarray],
        X_val: np.ndarray, Y_val: Dict[str, np.ndarray],
        feature_names: List[str],
        model_dir: str = 'surfer_models',
    ) -> Dict[str, float]:
        """Train the composite scorer on all model outputs."""
        import lightgbm as lgb

        self.feature_names = feature_names
        metrics = {}

        print("\n  Collecting meta-features from all models...")
        meta_X_train, meta_names = self._collect_meta_features(X_train, feature_names, model_dir)
        meta_X_val, _ = self._collect_meta_features(X_val, feature_names, model_dir)
        self.meta_feature_names = meta_names

        print(f"  Meta-features: {len(meta_names)}")
        for name in meta_names:
            print(f"    {name}")

        # Train composite action classifier
        print(f"\n  Training: composite_action (3-class, {len(meta_names)} meta-features)...")
        y_train = Y_train['optimal_action'].astype(int)
        y_val = Y_val['optimal_action'].astype(int)

        train_ds = lgb.Dataset(meta_X_train, label=y_train, feature_name=meta_names)
        val_ds = lgb.Dataset(meta_X_val, label=y_val, feature_name=meta_names, reference=train_ds)

        params = {
            'objective': 'multiclass', 'num_class': 3,
            'metric': 'multi_logloss', 'num_leaves': 16,
            'learning_rate': 0.03, 'min_child_samples': 15,
            'feature_fraction': 0.9, 'bagging_fraction': 0.9,
            'bagging_freq': 5, 'verbose': -1,
            'lambda_l1': 0.1, 'lambda_l2': 0.1,  # Regularize — small meta-feature set
        }
        model = lgb.train(
            params, train_ds, num_boost_round=500,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)],
        )
        self.model = model

        val_probs = model.predict(meta_X_val)
        val_pred = np.argmax(val_probs, axis=1)
        acc = np.mean(val_pred == y_val)
        metrics['composite_action_acc'] = float(acc)
        print(f"    Val accuracy: {acc:.1%}")

        # Per-class accuracy
        for cls, cls_name in enumerate(['HOLD', 'BUY', 'SELL']):
            mask = y_val == cls
            if mask.sum() > 5:
                cls_acc = np.mean(val_pred[mask] == cls)
                metrics[f'composite_{cls_name.lower()}_acc'] = float(cls_acc)
                print(f"    {cls_name} accuracy: {cls_acc:.1%} ({mask.sum()} samples)")

        # High-confidence predictions
        max_probs = np.max(val_probs, axis=1)
        high_conf = max_probs > 0.6
        if high_conf.sum() > 5:
            hc_acc = np.mean(val_pred[high_conf] == y_val[high_conf])
            metrics['composite_high_conf_acc'] = float(hc_acc)
            metrics['composite_high_conf_coverage'] = float(high_conf.mean())
            print(f"    High-confidence (>0.6): acc={hc_acc:.1%}, "
                  f"coverage={high_conf.mean():.1%}")

        # Feature importance
        imp = model.feature_importance(importance_type='gain')
        sorted_idx = np.argsort(imp)[::-1]
        print("\n  Feature importance (which model outputs matter most):")
        for rank, i in enumerate(sorted_idx):
            if imp[i] > 0:
                print(f"    {rank+1}. {meta_names[i]}: {imp[i]:.0f}")

        self.feature_importance = [(meta_names[i], float(imp[i])) for i in sorted_idx]

        return metrics

    def predict(self, X: np.ndarray, model_dir: str = 'surfer_models') -> Dict[str, np.ndarray]:
        """Predict using composite model."""
        meta_X, _ = self._collect_meta_features(X, self.feature_names, model_dir)
        probs = self.model.predict(meta_X)
        return {
            'action': np.argmax(probs, axis=1).astype(np.int64),
            'action_probs': probs,
            'max_confidence': np.max(probs, axis=1),
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'meta_feature_names': self.meta_feature_names,
                'feature_importance': getattr(self, 'feature_importance', []),
            }, f)
        print(f"  Saved CompositeSignalScorer to {path}")

    @classmethod
    def load(cls, path: str) -> 'CompositeSignalScorer':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.model = data['model']
        model.feature_names = data['feature_names']
        model.meta_feature_names = data.get('meta_feature_names')
        model.feature_importance = data.get('feature_importance', [])
        return model


# ---------------------------------------------------------------------------
# Architecture 14: Volatility Regime Transition Model
# ---------------------------------------------------------------------------

class VolatilityTransitionModel:
    """
    Predicts volatility regime transitions: low→high vol spikes.

    Key insight from backtest: ALL 7 stop-loss trades are 100% losers.
    These happen during sudden volatility expansions. If we can predict
    when volatility is about to spike (1-5 bars ahead), we can:
    - Skip entries during transition periods
    - Tighten stops on existing positions
    - Scale down position sizes

    Uses existing features + derived volatility acceleration features.
    Targets: binary classification (vol spike within next N bars).
    """

    def __init__(self):
        self.spike_model = None       # Predicts vol spike within 5 bars
        self.expansion_model = None   # Predicts sustained vol expansion (10 bars)
        self.magnitude_model = None   # Predicts magnitude of vol change (regression)
        self.feature_names = None
        self.vol_feature_names = None

    @staticmethod
    def compute_vol_targets(closes: np.ndarray, atr_pcts: np.ndarray) -> dict:
        """
        Compute volatility transition targets from price data.

        Args:
            closes: array of close prices (aligned with feature samples)
            atr_pcts: array of ATR-as-%-of-price values from features

        Returns dict with:
            vol_spike_5: binary - will ATR increase >40% in next 5 bars
            vol_expansion_10: binary - will ATR increase >30% sustained over 10 bars
            vol_change_magnitude: float - actual % change in realized vol
        """
        n = len(closes)
        spike_5 = np.zeros(n, dtype=np.float32)
        expansion_10 = np.zeros(n, dtype=np.float32)
        magnitude = np.zeros(n, dtype=np.float32)

        for i in range(n):
            current_vol = atr_pcts[i] if atr_pcts[i] > 1e-8 else 1e-8

            # Forward 5-bar realized vol (range-based)
            end5 = min(i + 6, n)
            if end5 - i >= 3:
                fwd_prices = closes[i:end5]
                fwd_returns = np.abs(np.diff(fwd_prices) / fwd_prices[:-1])
                fwd_vol_5 = np.mean(fwd_returns) * 100  # as percentage
                change_5 = (fwd_vol_5 - current_vol) / current_vol
                if change_5 > 0.40:  # >40% vol increase
                    spike_5[i] = 1.0

            # Forward 10-bar sustained expansion
            end10 = min(i + 11, n)
            if end10 - i >= 6:
                fwd_prices_10 = closes[i:end10]
                fwd_returns_10 = np.abs(np.diff(fwd_prices_10) / fwd_prices_10[:-1])
                fwd_vol_10 = np.mean(fwd_returns_10) * 100
                change_10 = (fwd_vol_10 - current_vol) / current_vol
                if change_10 > 0.30:  # >30% sustained vol expansion
                    expansion_10[i] = 1.0
                magnitude[i] = float(change_10)

        return {
            'vol_spike_5': spike_5,
            'vol_expansion_10': expansion_10,
            'vol_change_magnitude': magnitude,
        }

    @staticmethod
    def derive_vol_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """
        Derive volatility-specific features from the base feature set.

        Creates acceleration/jerk features from existing volatility indicators
        to capture the RATE OF CHANGE of volatility (not just level).
        """
        name_to_idx = {n: i for i, n in enumerate(feature_names)}

        vol_feats = []
        vol_names = []

        # 1. ATR acceleration: atr_pct is level, we want rate-of-change
        atr_idx = name_to_idx.get('atr_pct')
        if atr_idx is not None:
            atr = X[:, atr_idx]
            vol_feats.append(atr)
            vol_names.append('vol_atr_level')

        # 2. Entropy features (channel entropy = disorder = vol proxy)
        for tf in ['5min', '1h', '4h', 'daily']:
            ent_idx = name_to_idx.get(f'{tf}_entropy')
            if ent_idx is not None:
                vol_feats.append(X[:, ent_idx])
                vol_names.append(f'vol_{tf}_entropy')

        # 3. Entropy delta (rate of entropy change)
        ent_delta_idx = name_to_idx.get('entropy_delta_3bar')
        if ent_delta_idx is not None:
            vol_feats.append(X[:, ent_delta_idx])
            vol_names.append('vol_entropy_accel')

        # 4. Width delta (channel narrowing → imminent breakout)
        width_delta_idx = name_to_idx.get('width_delta_3bar')
        if width_delta_idx is not None:
            wd = X[:, width_delta_idx]
            vol_feats.append(wd)
            vol_names.append('vol_width_delta')
            # Squared width delta (captures extreme squeezes)
            vol_feats.append(wd ** 2 * np.sign(wd))
            vol_names.append('vol_width_delta_sq')

        # 5. Break probability features (high break prob → vol transition)
        bp_max_idx = name_to_idx.get('break_prob_max')
        bp_wt_idx = name_to_idx.get('break_prob_weighted')
        if bp_max_idx is not None:
            vol_feats.append(X[:, bp_max_idx])
            vol_names.append('vol_break_prob_max')
        if bp_wt_idx is not None:
            vol_feats.append(X[:, bp_wt_idx])
            vol_names.append('vol_break_prob_weighted')

        # 6. Break prob delta
        bp_delta_idx = name_to_idx.get('break_prob_delta_3bar')
        if bp_delta_idx is not None:
            vol_feats.append(X[:, bp_delta_idx])
            vol_names.append('vol_break_prob_accel')

        # 7. Energy features (kinetic energy = price momentum intensity)
        for tf in ['5min', '1h', '4h']:
            ke_idx = name_to_idx.get(f'{tf}_kinetic_energy')
            te_idx = name_to_idx.get(f'{tf}_total_energy')
            if ke_idx is not None:
                vol_feats.append(X[:, ke_idx])
                vol_names.append(f'vol_{tf}_kinetic_energy')
            if te_idx is not None:
                vol_feats.append(X[:, te_idx])
                vol_names.append(f'vol_{tf}_total_energy')

        # 8. Energy delta (rising energy = building pressure)
        energy_delta_idx = name_to_idx.get('energy_delta_3bar')
        if energy_delta_idx is not None:
            ed = X[:, energy_delta_idx]
            vol_feats.append(ed)
            vol_names.append('vol_energy_accel')

        # 9. Squeeze features (squeeze = vol compression → expansion imminent)
        sq_any_idx = name_to_idx.get('squeeze_any')
        if sq_any_idx is not None:
            vol_feats.append(X[:, sq_any_idx])
            vol_names.append('vol_squeeze_any')
        for tf in ['5min', '1h', '4h']:
            sq_idx = name_to_idx.get(f'{tf}_squeeze_score')
            if sq_idx is not None:
                vol_feats.append(X[:, sq_idx])
                vol_names.append(f'vol_{tf}_squeeze')

        # 10. Volume surge (volume spike often precedes vol expansion)
        vr_idx = name_to_idx.get('volume_ratio_20')
        vt_idx = name_to_idx.get('volume_trend_5')
        vm_idx = name_to_idx.get('vol_momentum_3bar')
        if vr_idx is not None:
            vol_feats.append(X[:, vr_idx])
            vol_names.append('vol_volume_ratio')
        if vt_idx is not None:
            vol_feats.append(X[:, vt_idx])
            vol_names.append('vol_volume_trend')
        if vm_idx is not None:
            vol_feats.append(X[:, vm_idx])
            vol_names.append('vol_volume_momentum')

        # 11. RSI extremes (oversold/overbought → reversal → vol spike)
        rsi_idx = name_to_idx.get('rsi_14')
        if rsi_idx is not None:
            rsi = X[:, rsi_idx]
            vol_feats.append(rsi)
            vol_names.append('vol_rsi')
            # Distance from neutral (50) — extremes predict vol
            vol_feats.append(np.abs(rsi - 50.0))
            vol_names.append('vol_rsi_extremity')

        # 12. VIX features
        vix_idx = name_to_idx.get('vix_level')
        vix_chg_idx = name_to_idx.get('vix_change_5d')
        if vix_idx is not None:
            vol_feats.append(X[:, vix_idx])
            vol_names.append('vol_vix_level')
        if vix_chg_idx is not None:
            vol_feats.append(X[:, vix_chg_idx])
            vol_names.append('vol_vix_change')

        # 13. Health min/spread (divergent TF health → unstable = vol)
        hmin_idx = name_to_idx.get('health_min')
        hspread_idx = name_to_idx.get('health_spread')
        if hmin_idx is not None:
            vol_feats.append(X[:, hmin_idx])
            vol_names.append('vol_health_min')
        if hspread_idx is not None:
            vol_feats.append(X[:, hspread_idx])
            vol_names.append('vol_health_spread')

        # 14. Health delta (deteriorating health → imminent break)
        hd_idx = name_to_idx.get('health_delta_3bar')
        if hd_idx is not None:
            vol_feats.append(X[:, hd_idx])
            vol_names.append('vol_health_accel')

        # 15. Price momentum (sharp momentum → vol expansion)
        pm3_idx = name_to_idx.get('price_momentum_3bar')
        pm12_idx = name_to_idx.get('price_momentum_12bar')
        if pm3_idx is not None:
            pm3 = X[:, pm3_idx]
            vol_feats.append(np.abs(pm3))
            vol_names.append('vol_price_mom_abs_3')
        if pm12_idx is not None:
            pm12 = X[:, pm12_idx]
            vol_feats.append(np.abs(pm12))
            vol_names.append('vol_price_mom_abs_12')

        # 16. Cross-TF interactions
        # Entropy * break_prob (both high → very likely vol spike)
        avg_ent_idx = name_to_idx.get('avg_entropy')
        if avg_ent_idx is not None and bp_max_idx is not None:
            vol_feats.append(X[:, avg_ent_idx] * X[:, bp_max_idx])
            vol_names.append('vol_entropy_x_breakprob')

        # Squeeze * energy (compressed + energetic → explosive)
        if sq_any_idx is not None and energy_delta_idx is not None:
            vol_feats.append(X[:, sq_any_idx] * np.abs(X[:, energy_delta_idx]))
            vol_names.append('vol_squeeze_x_energy')

        if len(vol_feats) == 0:
            return X, feature_names  # Fallback: use all features

        vol_X = np.column_stack(vol_feats)
        return vol_X, vol_names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names,
              closes_train=None, closes_val=None):
        """
        Train volatility transition models.

        If closes are provided, computes vol targets from price data.
        Otherwise uses existing return-based proxies from labels.
        """
        import lightgbm as lgb

        self.feature_names = list(feature_names)

        # Derive vol-specific features
        vol_X_train, self.vol_feature_names = self.derive_vol_features(
            X_train, feature_names)
        vol_X_val, _ = self.derive_vol_features(X_val, feature_names)

        # Compute targets from returns if closes not provided
        # Use future_return_5 as proxy: large absolute returns = vol spike
        ret5_train = Y_train['future_return_5']
        ret20_train = Y_train['future_return_20']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']

        # Vol spike: |5-bar return| > 0.8% (top ~20% of moves)
        threshold_5 = np.percentile(np.abs(ret5_train), 80)
        spike_5_train = (np.abs(ret5_train) > threshold_5).astype(np.float32)
        spike_5_val = (np.abs(ret5_val) > threshold_5).astype(np.float32)

        # Vol expansion: |20-bar return| > 1.5% (sustained move)
        threshold_20 = np.percentile(np.abs(ret20_train), 75)
        expansion_train = (np.abs(ret20_train) > threshold_20).astype(np.float32)
        expansion_val = (np.abs(ret20_val) > threshold_20).astype(np.float32)

        # Magnitude: absolute 5-bar return (regression)
        mag_train = np.abs(ret5_train).astype(np.float32)
        mag_val = np.abs(ret5_val).astype(np.float32)

        metrics = {}

        print(f"\n  Vol features: {len(self.vol_feature_names)}")
        print(f"  Spike threshold (|ret5| > {threshold_5:.4f}): "
              f"{spike_5_train.mean():.1%} positive rate")
        print(f"  Expansion threshold (|ret20| > {threshold_20:.4f}): "
              f"{expansion_train.mean():.1%} positive rate")

        # --- Model 1: Vol Spike (5-bar) ---
        print("\n  Training vol_spike_5 classifier...")
        dtrain = lgb.Dataset(vol_X_train, label=spike_5_train,
                            feature_name=self.vol_feature_names)
        dval = lgb.Dataset(vol_X_val, label=spike_5_val,
                          feature_name=self.vol_feature_names, reference=dtrain)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'is_unbalance': True,
        }

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        self.spike_model = lgb.train(
            params, dtrain, num_boost_round=500,
            valid_sets=[dval], callbacks=callbacks,
        )

        spike_pred = self.spike_model.predict(vol_X_val)
        from sklearn.metrics import roc_auc_score
        spike_auc = roc_auc_score(spike_5_val, spike_pred)
        metrics['spike_5_auc'] = float(spike_auc)
        print(f"    Spike AUC: {spike_auc:.3f}")

        # --- Model 2: Vol Expansion (10-bar) ---
        print("  Training vol_expansion classifier...")
        dtrain2 = lgb.Dataset(vol_X_train, label=expansion_train,
                             feature_name=self.vol_feature_names)
        dval2 = lgb.Dataset(vol_X_val, label=expansion_val,
                           feature_name=self.vol_feature_names, reference=dtrain2)

        self.expansion_model = lgb.train(
            params, dtrain2, num_boost_round=500,
            valid_sets=[dval2], callbacks=callbacks,
        )

        exp_pred = self.expansion_model.predict(vol_X_val)
        exp_auc = roc_auc_score(expansion_val, exp_pred)
        metrics['expansion_auc'] = float(exp_auc)
        print(f"    Expansion AUC: {exp_auc:.3f}")

        # --- Model 3: Vol Magnitude (regression) ---
        print("  Training vol_magnitude regressor...")
        dtrain3 = lgb.Dataset(vol_X_train, label=mag_train,
                             feature_name=self.vol_feature_names)
        dval3 = lgb.Dataset(vol_X_val, label=mag_val,
                           feature_name=self.vol_feature_names, reference=dtrain3)

        params_reg = {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
        }

        self.magnitude_model = lgb.train(
            params_reg, dtrain3, num_boost_round=500,
            valid_sets=[dval3], callbacks=callbacks,
        )

        mag_pred = self.magnitude_model.predict(vol_X_val)
        mag_mae = np.mean(np.abs(mag_pred - mag_val))
        # Correlation between predicted and actual magnitude
        mag_corr = np.corrcoef(mag_pred, mag_val)[0, 1]
        metrics['magnitude_mae'] = float(mag_mae)
        metrics['magnitude_corr'] = float(mag_corr)
        print(f"    Magnitude MAE: {mag_mae:.5f}, Corr: {mag_corr:.3f}")

        # Feature importance
        imp = self.spike_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 spike features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.vol_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        """
        Predict volatility regime transition probabilities.

        Returns dict with:
            spike_prob: probability of vol spike in next 5 bars
            expansion_prob: probability of sustained vol expansion
            predicted_magnitude: expected absolute return magnitude
            vol_regime: 'calm', 'warning', 'danger' based on thresholds
        """
        if self.spike_model is None:
            return {}

        vol_X, _ = self.derive_vol_features(X, self.feature_names)

        spike_prob = self.spike_model.predict(vol_X)
        expansion_prob = self.expansion_model.predict(vol_X)
        magnitude = self.magnitude_model.predict(vol_X)

        # Classify regime
        regime = np.where(
            spike_prob > 0.6, 2,  # danger
            np.where(spike_prob > 0.35, 1, 0)  # warning / calm
        )
        regime_labels = np.array(['calm', 'warning', 'danger'])

        return {
            'spike_prob': spike_prob,
            'expansion_prob': expansion_prob,
            'predicted_magnitude': magnitude,
            'vol_regime': regime_labels[regime],
            'vol_regime_id': regime,
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'spike_model': self.spike_model,
                'expansion_model': self.expansion_model,
                'magnitude_model': self.magnitude_model,
                'feature_names': self.feature_names,
                'vol_feature_names': self.vol_feature_names,
            }, f)
        print(f"  Saved VolatilityTransitionModel to {path}")

    @classmethod
    def load(cls, path: str) -> 'VolatilityTransitionModel':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.spike_model = data['spike_model']
        model.expansion_model = data['expansion_model']
        model.magnitude_model = data['magnitude_model']
        model.feature_names = data['feature_names']
        model.vol_feature_names = data['vol_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 15: Exit Timing Optimizer
# ---------------------------------------------------------------------------

class ExitTimingOptimizer:
    """
    Predicts the optimal exit point DURING a trade.

    Unlike entry timing (Arch 12, which failed), exit timing has stronger signal
    because we know we're IN a position and can observe how the trade develops.

    Key question: Given current features + trade state, should we:
    1. Hold (trade still has momentum in our favor)
    2. Tighten trail (momentum fading, protect profits)
    3. Exit now (reversal imminent)

    Features: base features + trade-specific context (bars held, unrealized P&L,
    distance from stop, distance from TP, trail distance).

    Target: Forward 5-bar P&L change from current position.
    """

    def __init__(self):
        self.exit_classifier = None    # HOLD=0, TIGHTEN=1, EXIT=2
        self.pnl_forecast = None       # Regression: expected 5-bar P&L change
        self.feature_names = None
        self.exit_feature_names = None

    @staticmethod
    def derive_exit_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """
        Derive features relevant to exit timing from base features.
        Focuses on momentum exhaustion and reversal signals.
        """
        name_to_idx = {n: i for i, n in enumerate(feature_names)}

        exit_feats = []
        exit_names = []

        # Momentum features
        for key in ['price_momentum_3bar', 'price_momentum_12bar',
                     'rsi_14', 'rsi_5', 'rsi_slope_5bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                exit_feats.append(X[:, idx])
                exit_names.append(f'exit_{key}')

        # RSI divergence (RSI vs momentum disagreement = exhaustion)
        rsi_idx = name_to_idx.get('rsi_14')
        pm3_idx = name_to_idx.get('price_momentum_3bar')
        if rsi_idx is not None and pm3_idx is not None:
            rsi = X[:, rsi_idx]
            pm3 = X[:, pm3_idx]
            # Normalize both to [-1, 1] range for comparison
            rsi_norm = (rsi - 50.0) / 50.0
            pm3_clip = np.clip(pm3 / 0.01, -1, 1)
            divergence = rsi_norm - pm3_clip
            exit_feats.append(divergence)
            exit_names.append('exit_rsi_momentum_divergence')

        # Channel health trajectory
        for key in ['health_delta_3bar', 'health_delta_6bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                exit_feats.append(X[:, idx])
                exit_names.append(f'exit_{key}')

        # Position in channel (near boundary = potential exit point)
        for tf in ['5min', '1h', '4h']:
            pos_idx = name_to_idx.get(f'{tf}_position_pct')
            if pos_idx is not None:
                pos = X[:, pos_idx]
                exit_feats.append(pos)
                exit_names.append(f'exit_{tf}_position')
                # Distance from boundary (0 or 1)
                exit_feats.append(np.minimum(pos, 1.0 - pos))
                exit_names.append(f'exit_{tf}_boundary_dist')

        # Break probability (rising = potential reversal)
        for key in ['break_prob_max', 'break_prob_weighted', 'break_prob_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                exit_feats.append(X[:, idx])
                exit_names.append(f'exit_{key}')

        # Entropy (rising entropy = channel failing)
        for key in ['avg_entropy', 'entropy_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                exit_feats.append(X[:, idx])
                exit_names.append(f'exit_{key}')

        # Volume features (volume drying up = momentum exhaustion)
        for key in ['volume_ratio_20', 'volume_trend_5', 'vol_momentum_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                exit_feats.append(X[:, idx])
                exit_names.append(f'exit_{key}')

        # ATR (changing volatility affects optimal exit)
        atr_idx = name_to_idx.get('atr_pct')
        if atr_idx is not None:
            exit_feats.append(X[:, atr_idx])
            exit_names.append('exit_atr_pct')

        # Direction consensus (fading consensus = trend weakening)
        dc_idx = name_to_idx.get('direction_consensus')
        if dc_idx is not None:
            exit_feats.append(X[:, dc_idx])
            exit_names.append('exit_direction_consensus')

        # Consecutive bars (long streak → reversal more likely)
        for key in ['consecutive_up_bars', 'consecutive_down_bars']:
            idx = name_to_idx.get(key)
            if idx is not None:
                exit_feats.append(X[:, idx])
                exit_names.append(f'exit_{key}')

        if len(exit_feats) == 0:
            return X, feature_names

        return np.column_stack(exit_feats), exit_names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """
        Train exit timing models.

        Uses future returns to determine optimal action:
        - EXIT: 5-bar return goes against direction significantly (>0.3%)
        - TIGHTEN: 5-bar return is flat or slightly negative (-0.3% to 0%)
        - HOLD: 5-bar return is positive (in favorable direction)
        """
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)

        exit_X_train, self.exit_feature_names = self.derive_exit_features(
            X_train, feature_names)
        exit_X_val, _ = self.derive_exit_features(X_val, feature_names)

        ret5_train = Y_train['future_return_5']
        ret5_val = Y_val['future_return_5']

        # For exit timing, we consider BOTH directions (buy and sell trades)
        # A "bad" outcome is large adverse move regardless of direction
        # We use absolute return and classify:
        # EXIT (2): return < -0.3% (adverse)
        # TIGHTEN (1): -0.3% <= return < 0.1% (flat/slightly negative)
        # HOLD (0): return >= 0.1% (favorable)

        # Since we don't know trade direction at training time,
        # we train on: |return| magnitude + direction of move
        abs_ret = np.abs(ret5_train)

        # Target: will the next 5 bars see a reversal (directional change)?
        # Reversal = price moves significantly in OPPOSITE direction from recent momentum
        pm3_idx = {n: i for i, n in enumerate(feature_names)}.get('price_momentum_3bar')

        if pm3_idx is not None:
            momentum = X_train[:, pm3_idx]
            momentum_val = X_val[:, pm3_idx]

            # Reversal: momentum positive but future return negative (or vice versa)
            reversal_strength = -momentum * ret5_train  # Positive = reversal

            # 3-class: HOLD, TIGHTEN, EXIT based on reversal strength
            exit_target_train = np.where(
                reversal_strength > 0.00003, 2,  # Strong reversal → EXIT
                np.where(reversal_strength > 0.000005, 1, 0)  # Mild → TIGHTEN, else HOLD
            ).astype(np.int32)

            reversal_val = -momentum_val * ret5_val
            exit_target_val = np.where(
                reversal_val > 0.00003, 2,
                np.where(reversal_val > 0.000005, 1, 0)
            ).astype(np.int32)
        else:
            # Fallback: just use return thresholds
            exit_target_train = np.where(
                ret5_train < -0.003, 2,
                np.where(ret5_train < 0.001, 1, 0)
            ).astype(np.int32)
            exit_target_val = np.where(
                ret5_val < -0.003, 2,
                np.where(ret5_val < 0.001, 1, 0)
            ).astype(np.int32)

        metrics = {}

        print(f"\n  Exit features: {len(self.exit_feature_names)}")
        print(f"  Target distribution (train): "
              f"HOLD={np.mean(exit_target_train==0):.1%}, "
              f"TIGHTEN={np.mean(exit_target_train==1):.1%}, "
              f"EXIT={np.mean(exit_target_train==2):.1%}")

        # --- Classifier: HOLD/TIGHTEN/EXIT ---
        print("  Training exit classifier...")
        dtrain = lgb.Dataset(exit_X_train, label=exit_target_train,
                            feature_name=self.exit_feature_names)
        dval = lgb.Dataset(exit_X_val, label=exit_target_val,
                          feature_name=self.exit_feature_names, reference=dtrain)

        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
        }

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        self.exit_classifier = lgb.train(
            params, dtrain, num_boost_round=500,
            valid_sets=[dval], callbacks=callbacks,
        )

        exit_probs = self.exit_classifier.predict(exit_X_val).reshape(-1, 3)
        exit_pred = np.argmax(exit_probs, axis=1)
        exit_acc = np.mean(exit_pred == exit_target_val)
        metrics['exit_accuracy'] = float(exit_acc)

        # Per-class accuracy
        for c, name in enumerate(['HOLD', 'TIGHTEN', 'EXIT']):
            mask = exit_target_val == c
            if mask.sum() > 0:
                class_acc = np.mean(exit_pred[mask] == c)
                metrics[f'{name.lower()}_acc'] = float(class_acc)
                print(f"    {name} accuracy: {class_acc:.1%} ({mask.sum()} samples)")

        print(f"    Overall accuracy: {exit_acc:.1%}")

        # --- PnL Forecast (regression) ---
        print("  Training P&L forecaster...")
        dtrain_reg = lgb.Dataset(exit_X_train, label=ret5_train.astype(np.float32),
                                feature_name=self.exit_feature_names)
        dval_reg = lgb.Dataset(exit_X_val, label=ret5_val.astype(np.float32),
                              feature_name=self.exit_feature_names, reference=dtrain_reg)

        params_reg = {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
        }

        self.pnl_forecast = lgb.train(
            params_reg, dtrain_reg, num_boost_round=500,
            valid_sets=[dval_reg], callbacks=callbacks,
        )

        pnl_pred = self.pnl_forecast.predict(exit_X_val)
        pnl_mae = np.mean(np.abs(pnl_pred - ret5_val))
        pnl_corr = np.corrcoef(pnl_pred, ret5_val)[0, 1]
        pnl_dir_acc = np.mean(np.sign(pnl_pred) == np.sign(ret5_val))
        metrics['pnl_forecast_mae'] = float(pnl_mae)
        metrics['pnl_forecast_corr'] = float(pnl_corr)
        metrics['pnl_dir_acc'] = float(pnl_dir_acc)
        print(f"    P&L MAE: {pnl_mae:.5f}, Corr: {pnl_corr:.3f}, Dir Acc: {pnl_dir_acc:.1%}")

        # Feature importance
        imp = self.exit_classifier.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 exit features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.exit_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        """
        Predict exit timing recommendations.

        Returns:
            exit_action: 0=HOLD, 1=TIGHTEN, 2=EXIT
            exit_probs: [hold_prob, tighten_prob, exit_prob]
            pnl_forecast: expected 5-bar forward return
        """
        if self.exit_classifier is None:
            return {}

        exit_X, _ = self.derive_exit_features(X, self.feature_names)

        probs = self.exit_classifier.predict(exit_X).reshape(-1, 3)
        action = np.argmax(probs, axis=1)
        pnl = self.pnl_forecast.predict(exit_X)

        return {
            'exit_action': action,
            'exit_probs': probs,
            'pnl_forecast': pnl,
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'exit_classifier': self.exit_classifier,
                'pnl_forecast': self.pnl_forecast,
                'feature_names': self.feature_names,
                'exit_feature_names': self.exit_feature_names,
            }, f)
        print(f"  Saved ExitTimingOptimizer to {path}")

    @classmethod
    def load(cls, path: str) -> 'ExitTimingOptimizer':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.exit_classifier = data['exit_classifier']
        model.pnl_forecast = data['pnl_forecast']
        model.feature_names = data['feature_names']
        model.exit_feature_names = data['exit_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 16: Momentum Exhaustion Detector
# ---------------------------------------------------------------------------

class MomentumExhaustionDetector:
    """
    Detects when a price move is running out of steam.

    Models the 2nd derivative of price momentum (deceleration/jerk).
    Key signals:
    - RSI divergence (price making new highs but RSI declining)
    - Volume decline during rally (fuel running out)
    - Consecutive bar range shrinkage (moves getting smaller)
    - Momentum slope change (acceleration → deceleration)

    This identifies the inflection point BEFORE a reversal happens,
    giving time to tighten stops or exit before stop-losses trigger.

    Target: Will the next 5-10 bars reverse the current momentum direction?
    """

    def __init__(self):
        self.exhaustion_model = None   # Binary: exhaustion imminent?
        self.severity_model = None     # Regression: how severe is the reversal?
        self.feature_names = None
        self.exh_feature_names = None

    @staticmethod
    def derive_exhaustion_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """Derive features specifically targeting momentum exhaustion."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}

        feats = []
        names = []

        # Core momentum features
        pm3_idx = name_to_idx.get('price_momentum_3bar')
        pm12_idx = name_to_idx.get('price_momentum_12bar')

        if pm3_idx is not None and pm12_idx is not None:
            pm3 = X[:, pm3_idx]
            pm12 = X[:, pm12_idx]

            # 1. Momentum deceleration (short-term < long-term = slowing)
            feats.append(pm3 - pm12)
            names.append('exh_momentum_decel')

            # 2. Momentum ratio (divergence between timeframes)
            safe_pm12 = np.where(np.abs(pm12) > 1e-8, pm12, 1e-8)
            feats.append(pm3 / safe_pm12)
            names.append('exh_momentum_ratio')

            # 3. Absolute momentum (high momentum = more room to exhaust)
            feats.append(np.abs(pm3))
            names.append('exh_abs_momentum_3')
            feats.append(np.abs(pm12))
            names.append('exh_abs_momentum_12')

        # RSI features (overbought/oversold extremes = exhaustion)
        rsi14_idx = name_to_idx.get('rsi_14')
        rsi5_idx = name_to_idx.get('rsi_5')
        rsi_slope_idx = name_to_idx.get('rsi_slope_5bar')

        if rsi14_idx is not None:
            rsi14 = X[:, rsi14_idx]
            feats.append(rsi14)
            names.append('exh_rsi_14')
            # RSI extremity (distance from 50)
            feats.append(np.abs(rsi14 - 50))
            names.append('exh_rsi_extremity')

        if rsi5_idx is not None:
            rsi5 = X[:, rsi5_idx]
            feats.append(rsi5)
            names.append('exh_rsi_5')

        if rsi_slope_idx is not None:
            feats.append(X[:, rsi_slope_idx])
            names.append('exh_rsi_slope')

        # RSI divergence (RSI declining while price rising)
        if rsi_slope_idx is not None and pm3_idx is not None:
            rsi_slope = X[:, rsi_slope_idx]
            pm3 = X[:, pm3_idx]
            # Divergence: momentum positive but RSI slope negative (or vice versa)
            feats.append(pm3 * -rsi_slope)
            names.append('exh_rsi_divergence')

        # Volume exhaustion features
        vr_idx = name_to_idx.get('volume_ratio_20')
        vt_idx = name_to_idx.get('volume_trend_5')
        vm_idx = name_to_idx.get('vol_momentum_3bar')

        if vr_idx is not None:
            feats.append(X[:, vr_idx])
            names.append('exh_volume_ratio')
        if vt_idx is not None:
            feats.append(X[:, vt_idx])
            names.append('exh_volume_trend')
        if vm_idx is not None:
            feats.append(X[:, vm_idx])
            names.append('exh_volume_momentum')

        # Volume-momentum divergence (price up but volume declining)
        if vt_idx is not None and pm3_idx is not None:
            vt = X[:, vt_idx]
            pm3 = X[:, pm3_idx]
            feats.append(pm3 * -vt)  # Positive = momentum up, volume down
            names.append('exh_vol_momentum_divergence')

        # Consecutive bar features (streaks about to end)
        up_idx = name_to_idx.get('consecutive_up_bars')
        dn_idx = name_to_idx.get('consecutive_down_bars')
        if up_idx is not None:
            feats.append(X[:, up_idx])
            names.append('exh_consecutive_up')
        if dn_idx is not None:
            feats.append(X[:, dn_idx])
            names.append('exh_consecutive_down')

        # Bar range as % of ATR (shrinking ranges = exhaustion)
        bar_range_idx = name_to_idx.get('bar_range_pct')
        if bar_range_idx is not None:
            feats.append(X[:, bar_range_idx])
            names.append('exh_bar_range_pct')

        # Close position in bar (doji-like = indecision)
        close_pos_idx = name_to_idx.get('close_position_in_bar')
        if close_pos_idx is not None:
            cp = X[:, close_pos_idx]
            feats.append(cp)
            names.append('exh_close_position')
            # Distance from extremes (near 0.5 = doji = indecision)
            feats.append(np.abs(cp - 0.5))
            names.append('exh_doji_score')

        # Channel health trajectory (deteriorating health = momentum failing)
        hd3_idx = name_to_idx.get('health_delta_3bar')
        hd6_idx = name_to_idx.get('health_delta_6bar')
        if hd3_idx is not None:
            feats.append(X[:, hd3_idx])
            names.append('exh_health_delta_3')
        if hd6_idx is not None:
            feats.append(X[:, hd6_idx])
            names.append('exh_health_delta_6')

        # Position in channel (extreme positions = reversal likely)
        for tf in ['5min', '1h', '4h']:
            pos_idx = name_to_idx.get(f'{tf}_position_pct')
            if pos_idx is not None:
                pos = X[:, pos_idx]
                feats.append(pos)
                names.append(f'exh_{tf}_position')
                # Quadratic distance from center (captures extremes)
                feats.append((pos - 0.5) ** 2)
                names.append(f'exh_{tf}_pos_extreme')

        # Entropy (rising entropy = increasing disorder = momentum failing)
        ent_d_idx = name_to_idx.get('entropy_delta_3bar')
        if ent_d_idx is not None:
            feats.append(X[:, ent_d_idx])
            names.append('exh_entropy_delta')

        # Break probability (rising = structural change imminent)
        bp_d_idx = name_to_idx.get('break_prob_delta_3bar')
        if bp_d_idx is not None:
            feats.append(X[:, bp_d_idx])
            names.append('exh_break_prob_delta')

        # ATR (absolute vol level)
        atr_idx = name_to_idx.get('atr_pct')
        if atr_idx is not None:
            feats.append(X[:, atr_idx])
            names.append('exh_atr')

        # Direction consensus (fading consensus = exhaustion)
        dc_idx = name_to_idx.get('direction_consensus')
        if dc_idx is not None:
            feats.append(X[:, dc_idx])
            names.append('exh_direction_consensus')

        # SPY correlation (decorrelation can signal exhaustion)
        spy_corr_idx = name_to_idx.get('spy_tsla_corr_20')
        if spy_corr_idx is not None:
            feats.append(X[:, spy_corr_idx])
            names.append('exh_spy_corr')

        if len(feats) == 0:
            return X, feature_names

        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train momentum exhaustion detector."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        exh_X_train, self.exh_feature_names = self.derive_exhaustion_features(
            X_train, feature_names)
        exh_X_val, _ = self.derive_exhaustion_features(X_val, feature_names)

        # Target: momentum reversal in next 5 bars
        # Look at price_momentum_3bar to determine current direction,
        # then check if future_return_5 goes OPPOSITE
        pm3_idx = {n: i for i, n in enumerate(feature_names)}.get('price_momentum_3bar')
        ret5_train = Y_train['future_return_5']
        ret5_val = Y_val['future_return_5']
        ret20_train = Y_train['future_return_20']
        ret20_val = Y_val['future_return_20']

        if pm3_idx is not None:
            mom_train = X_train[:, pm3_idx]
            mom_val = X_val[:, pm3_idx]

            # Exhaustion = current momentum positive but future return negative (or vice versa)
            # AND the reversal is significant (>0.2%)
            reversal_train = (
                ((mom_train > 0.001) & (ret5_train < -0.002)) |
                ((mom_train < -0.001) & (ret5_train > 0.002))
            ).astype(np.float32)

            reversal_val = (
                ((mom_val > 0.001) & (ret5_val < -0.002)) |
                ((mom_val < -0.001) & (ret5_val > 0.002))
            ).astype(np.float32)
        else:
            # Fallback: large absolute return change
            reversal_train = (np.abs(ret5_train) > 0.005).astype(np.float32)
            reversal_val = (np.abs(ret5_val) > 0.005).astype(np.float32)

        # Severity: magnitude of reversal (how far against current direction)
        severity_train = np.abs(ret5_train).astype(np.float32)
        severity_val = np.abs(ret5_val).astype(np.float32)

        metrics = {}

        print(f"\n  Exhaustion features: {len(self.exh_feature_names)}")
        print(f"  Reversal rate: {reversal_train.mean():.1%}")

        # --- Model 1: Exhaustion classifier ---
        print("  Training exhaustion classifier...")
        dtrain = lgb.Dataset(exh_X_train, label=reversal_train,
                            feature_name=self.exh_feature_names)
        dval = lgb.Dataset(exh_X_val, label=reversal_val,
                          feature_name=self.exh_feature_names, reference=dtrain)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'is_unbalance': True,
        }

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        self.exhaustion_model = lgb.train(
            params, dtrain, num_boost_round=500,
            valid_sets=[dval], callbacks=callbacks,
        )

        exh_pred = self.exhaustion_model.predict(exh_X_val)
        exh_auc = roc_auc_score(reversal_val, exh_pred)
        metrics['exhaustion_auc'] = float(exh_auc)
        print(f"    Exhaustion AUC: {exh_auc:.3f}")

        # High-confidence precision
        high_conf_mask = exh_pred > 0.6
        if high_conf_mask.sum() > 10:
            high_conf_precision = reversal_val[high_conf_mask].mean()
            metrics['high_conf_precision'] = float(high_conf_precision)
            print(f"    High-conf (>0.6) precision: {high_conf_precision:.1%} "
                  f"({high_conf_mask.sum()} samples)")

        # --- Model 2: Severity regressor ---
        print("  Training severity regressor...")
        dtrain2 = lgb.Dataset(exh_X_train, label=severity_train,
                             feature_name=self.exh_feature_names)
        dval2 = lgb.Dataset(exh_X_val, label=severity_val,
                           feature_name=self.exh_feature_names, reference=dtrain2)

        params_reg = {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
        }

        self.severity_model = lgb.train(
            params_reg, dtrain2, num_boost_round=500,
            valid_sets=[dval2], callbacks=callbacks,
        )

        sev_pred = self.severity_model.predict(exh_X_val)
        sev_mae = np.mean(np.abs(sev_pred - severity_val))
        sev_corr = np.corrcoef(sev_pred, severity_val)[0, 1]
        metrics['severity_mae'] = float(sev_mae)
        metrics['severity_corr'] = float(sev_corr)
        print(f"    Severity MAE: {sev_mae:.5f}, Corr: {sev_corr:.3f}")

        # Feature importance
        imp = self.exhaustion_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 exhaustion features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.exh_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        """
        Predict momentum exhaustion.

        Returns:
            exhaustion_prob: probability of momentum reversal in next 5 bars
            reversal_severity: expected magnitude of reversal
            exhaustion_level: 'fresh', 'tiring', 'exhausted'
        """
        if self.exhaustion_model is None:
            return {}

        exh_X, _ = self.derive_exhaustion_features(X, self.feature_names)

        exh_prob = self.exhaustion_model.predict(exh_X)
        severity = self.severity_model.predict(exh_X)

        level = np.where(
            exh_prob > 0.55, 2,  # exhausted
            np.where(exh_prob > 0.35, 1, 0)  # tiring / fresh
        )
        level_labels = np.array(['fresh', 'tiring', 'exhausted'])

        return {
            'exhaustion_prob': exh_prob,
            'reversal_severity': severity,
            'exhaustion_level': level_labels[level],
            'exhaustion_level_id': level,
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'exhaustion_model': self.exhaustion_model,
                'severity_model': self.severity_model,
                'feature_names': self.feature_names,
                'exh_feature_names': self.exh_feature_names,
            }, f)
        print(f"  Saved MomentumExhaustionDetector to {path}")

    @classmethod
    def load(cls, path: str) -> 'MomentumExhaustionDetector':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.exhaustion_model = data['exhaustion_model']
        model.severity_model = data['severity_model']
        model.feature_names = data['feature_names']
        model.exh_feature_names = data['exh_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 17: Cross-Asset Signal Amplifier
# ---------------------------------------------------------------------------

class CrossAssetAmplifier:
    """
    Uses SPY/VIX co-movement patterns to amplify or dampen TSLA signals.

    Key insight: When TSLA and SPY are highly correlated, TSLA moves are
    market-driven (less predictable from channel analysis). When decorrelated,
    TSLA moves are idiosyncratic (channels are more reliable).

    Also models VIX regime effects: low VIX = channels hold longer,
    high VIX = channels break faster.

    Targets:
    - Whether channel-based trading outperforms in current market regime
    - Optimal confidence scaling factor for current cross-asset conditions
    """

    def __init__(self):
        self.regime_model = None       # Market regime classifier
        self.scale_model = None        # Confidence scaling regressor
        self.feature_names = None
        self.cross_feature_names = None

    @staticmethod
    def derive_cross_asset_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """Derive cross-asset interaction features."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}

        feats = []
        names_list = []

        # Core cross-asset features
        spy_ret5_idx = name_to_idx.get('spy_return_5bar')
        spy_ret20_idx = name_to_idx.get('spy_return_20bar')
        spy_corr_idx = name_to_idx.get('spy_tsla_corr_20')
        vix_idx = name_to_idx.get('vix_level')
        vix_chg_idx = name_to_idx.get('vix_change_5d')

        if spy_ret5_idx is not None:
            feats.append(X[:, spy_ret5_idx])
            names_list.append('ca_spy_ret5')
        if spy_ret20_idx is not None:
            feats.append(X[:, spy_ret20_idx])
            names_list.append('ca_spy_ret20')
        if spy_corr_idx is not None:
            corr = X[:, spy_corr_idx]
            feats.append(corr)
            names_list.append('ca_spy_corr')
            # Absolute correlation (high = market-driven)
            feats.append(np.abs(corr))
            names_list.append('ca_spy_abs_corr')
            # Decorrelation (1 - |corr|, high = TSLA-specific moves)
            feats.append(1.0 - np.abs(corr))
            names_list.append('ca_decorrelation')

        if vix_idx is not None:
            vix = X[:, vix_idx]
            feats.append(vix)
            names_list.append('ca_vix_level')
            # VIX regimes
            feats.append((vix > 20).astype(np.float32))
            names_list.append('ca_vix_elevated')
            feats.append((vix > 30).astype(np.float32))
            names_list.append('ca_vix_high')

        if vix_chg_idx is not None:
            feats.append(X[:, vix_chg_idx])
            names_list.append('ca_vix_change')

        # TSLA momentum vs SPY momentum (relative strength)
        pm3_idx = name_to_idx.get('price_momentum_3bar')
        pm12_idx = name_to_idx.get('price_momentum_12bar')
        if pm3_idx is not None and spy_ret5_idx is not None:
            feats.append(X[:, pm3_idx] - X[:, spy_ret5_idx])
            names_list.append('ca_relative_momentum_short')
        if pm12_idx is not None and spy_ret20_idx is not None:
            feats.append(X[:, pm12_idx] - X[:, spy_ret20_idx])
            names_list.append('ca_relative_momentum_long')

        # VIX × TSLA correlation interaction
        if vix_idx is not None and spy_corr_idx is not None:
            feats.append(X[:, vix_idx] * np.abs(X[:, spy_corr_idx]))
            names_list.append('ca_vix_corr_interaction')

        # VIX × ATR interaction (vol regime matching)
        atr_idx = name_to_idx.get('atr_pct')
        if vix_idx is not None and atr_idx is not None:
            feats.append(X[:, vix_idx] * X[:, atr_idx])
            names_list.append('ca_vix_atr_interaction')

        # Channel health features (to model when cross-asset matters more)
        hmin_idx = name_to_idx.get('health_min')
        hmax_idx = name_to_idx.get('health_max')
        if hmin_idx is not None:
            feats.append(X[:, hmin_idx])
            names_list.append('ca_health_min')
        if hmax_idx is not None:
            feats.append(X[:, hmax_idx])
            names_list.append('ca_health_max')

        # Direction consensus (to model when TSLA-specific vs market-driven)
        dc_idx = name_to_idx.get('direction_consensus')
        if dc_idx is not None:
            feats.append(X[:, dc_idx])
            names_list.append('ca_direction_consensus')

        # Break probability (channel stability measure)
        bp_idx = name_to_idx.get('break_prob_max')
        if bp_idx is not None:
            feats.append(X[:, bp_idx])
            names_list.append('ca_break_prob')

        # ATR
        if atr_idx is not None:
            feats.append(X[:, atr_idx])
            names_list.append('ca_atr')

        # RSI (market condition)
        rsi_idx = name_to_idx.get('rsi_14')
        if rsi_idx is not None:
            feats.append(X[:, rsi_idx])
            names_list.append('ca_rsi')

        if len(feats) == 0:
            return X, feature_names

        return np.column_stack(feats), names_list

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """
        Train cross-asset amplifier.

        Target 1 (regime): 4-class market regime
            0 = risk-on (SPY up, VIX low) → channels reliable
            1 = risk-off (SPY down, VIX high) → shorts reliable
            2 = rotation (SPY flat, TSLA moving) → highest alpha
            3 = correlated selloff → avoid trading

        Target 2 (scale): Optimal confidence scale factor based on
            whether the next 20-bar return aligns with channel prediction.
        """
        import lightgbm as lgb

        self.feature_names = list(feature_names)
        cross_X_train, self.cross_feature_names = self.derive_cross_asset_features(
            X_train, feature_names)
        cross_X_val, _ = self.derive_cross_asset_features(X_val, feature_names)

        name_to_idx = {n: i for i, n in enumerate(feature_names)}

        # Compute market regime targets
        spy5_idx = name_to_idx.get('spy_return_5bar')
        vix_idx = name_to_idx.get('vix_level')
        corr_idx = name_to_idx.get('spy_tsla_corr_20')

        def compute_regime(X_data):
            n = len(X_data)
            regime = np.zeros(n, dtype=np.int32)
            for i in range(n):
                spy_ret = X_data[i, spy5_idx] if spy5_idx is not None else 0
                vix = X_data[i, vix_idx] if vix_idx is not None else 15
                corr = X_data[i, corr_idx] if corr_idx is not None else 0.5

                if spy_ret > 0.002 and vix < 20:
                    regime[i] = 0  # risk-on
                elif spy_ret < -0.002 and vix > 25:
                    regime[i] = 3  # correlated selloff
                elif abs(corr) < 0.3:
                    regime[i] = 2  # rotation (decorrelated)
                else:
                    regime[i] = 1  # risk-off
            return regime

        regime_train = compute_regime(X_train)
        regime_val = compute_regime(X_val)

        # Confidence scale target: based on whether action was correct
        # Scale = 1.0 (neutral), >1 if good trade opportunity, <1 if bad
        action_train = Y_train['optimal_action']
        ret20_train = Y_train['future_return_20']
        ret20_val = Y_val['future_return_20']

        # Good outcome = large absolute return (more profit opportunity)
        # Scale it relative to typical return
        typical_ret = np.percentile(np.abs(ret20_train), 50)
        scale_train = np.clip(np.abs(ret20_train) / max(typical_ret, 1e-6), 0.3, 2.0).astype(np.float32)
        scale_val = np.clip(np.abs(ret20_val) / max(typical_ret, 1e-6), 0.3, 2.0).astype(np.float32)

        metrics = {}

        print(f"\n  Cross-asset features: {len(self.cross_feature_names)}")
        print(f"  Regime distribution (train): "
              f"risk-on={np.mean(regime_train==0):.1%}, "
              f"risk-off={np.mean(regime_train==1):.1%}, "
              f"rotation={np.mean(regime_train==2):.1%}, "
              f"selloff={np.mean(regime_train==3):.1%}")

        # --- Model 1: Market regime classifier ---
        print("  Training market regime classifier...")
        dtrain = lgb.Dataset(cross_X_train, label=regime_train,
                            feature_name=self.cross_feature_names)
        dval = lgb.Dataset(cross_X_val, label=regime_val,
                          feature_name=self.cross_feature_names, reference=dtrain)

        params = {
            'objective': 'multiclass',
            'num_class': 4,
            'metric': 'multi_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
        }

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        self.regime_model = lgb.train(
            params, dtrain, num_boost_round=500,
            valid_sets=[dval], callbacks=callbacks,
        )

        regime_probs = self.regime_model.predict(cross_X_val).reshape(-1, 4)
        regime_pred = np.argmax(regime_probs, axis=1)
        regime_acc = np.mean(regime_pred == regime_val)
        metrics['regime_accuracy'] = float(regime_acc)

        regime_names = ['risk-on', 'risk-off', 'rotation', 'selloff']
        for c, name in enumerate(regime_names):
            mask = regime_val == c
            if mask.sum() > 0:
                class_acc = np.mean(regime_pred[mask] == c)
                metrics[f'{name}_acc'] = float(class_acc)
                print(f"    {name}: {class_acc:.1%} ({mask.sum()} samples)")

        print(f"    Overall regime accuracy: {regime_acc:.1%}")

        # --- Model 2: Confidence scale regressor ---
        print("  Training confidence scale regressor...")
        dtrain2 = lgb.Dataset(cross_X_train, label=scale_train,
                             feature_name=self.cross_feature_names)
        dval2 = lgb.Dataset(cross_X_val, label=scale_val,
                           feature_name=self.cross_feature_names, reference=dtrain2)

        params_reg = {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
        }

        self.scale_model = lgb.train(
            params_reg, dtrain2, num_boost_round=500,
            valid_sets=[dval2], callbacks=callbacks,
        )

        scale_pred = self.scale_model.predict(cross_X_val)
        scale_mae = np.mean(np.abs(scale_pred - scale_val))
        scale_corr = np.corrcoef(scale_pred, scale_val)[0, 1]
        metrics['scale_mae'] = float(scale_mae)
        metrics['scale_corr'] = float(scale_corr)
        print(f"    Scale MAE: {scale_mae:.3f}, Corr: {scale_corr:.3f}")

        # Feature importance
        imp = self.regime_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 cross-asset features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.cross_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        """
        Predict cross-asset regime and confidence scaling.

        Returns:
            market_regime: 0=risk-on, 1=risk-off, 2=rotation, 3=selloff
            regime_probs: [risk_on, risk_off, rotation, selloff]
            confidence_scale: suggested scaling factor for confidence
            regime_label: human-readable regime name
        """
        if self.regime_model is None:
            return {}

        cross_X, _ = self.derive_cross_asset_features(X, self.feature_names)

        regime_probs = self.regime_model.predict(cross_X).reshape(-1, 4)
        regime = np.argmax(regime_probs, axis=1)
        scale = self.scale_model.predict(cross_X)

        regime_labels = np.array(['risk-on', 'risk-off', 'rotation', 'selloff'])

        return {
            'market_regime': regime,
            'regime_probs': regime_probs,
            'confidence_scale': scale,
            'regime_label': regime_labels[regime],
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'regime_model': self.regime_model,
                'scale_model': self.scale_model,
                'feature_names': self.feature_names,
                'cross_feature_names': self.cross_feature_names,
            }, f)
        print(f"  Saved CrossAssetAmplifier to {path}")

    @classmethod
    def load(cls, path: str) -> 'CrossAssetAmplifier':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.regime_model = data['regime_model']
        model.scale_model = data['scale_model']
        model.feature_names = data['feature_names']
        model.cross_feature_names = data['cross_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 18: Stop Loss Hit Predictor
# ---------------------------------------------------------------------------

class StopLossPredictor:
    """
    Directly predicts whether a trade will hit its stop loss.

    This is the most targeted attack on the biggest loss source:
    ALL stop-loss exits are 100% losers in the backtest.

    Unlike other models that adjust confidence by small amounts,
    this one asks a binary question: "Will this specific trade
    hit its stop?"

    Uses all base features + derived features focused on:
    - Price behavior that precedes stop-outs (gaps, sudden moves)
    - Channel instability that leads to false breakouts
    - Volume patterns that indicate unsustainable moves

    Target: Max adverse excursion > stop distance within 20 bars
    """

    def __init__(self):
        self.stop_model = None         # Binary: will trade hit stop?
        self.mae_model = None          # Regression: max adverse excursion
        self.feature_names = None
        self.stop_feature_names = None

    @staticmethod
    def derive_stop_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """Derive features specifically targeting stop-loss prediction."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}

        feats = []
        feat_names = []

        # Use ALL base features (they're all relevant for stop prediction)
        feats.append(X)
        feat_names.extend(feature_names)

        # Derived: position extremity across TFs
        for tf in ['5min', '1h', '4h', 'daily']:
            pos_idx = name_to_idx.get(f'{tf}_position_pct')
            if pos_idx is not None:
                pos = X[:, pos_idx]
                # Extreme positions (near 0 or 1) are stop-out risk
                feats.append((pos * (1 - pos)).reshape(-1, 1))  # Parabolic: 0 at edges, 0.25 at center
                feat_names.append(f'stop_{tf}_edge_risk')

        # Derived: momentum vs channel position divergence
        pm3_idx = name_to_idx.get('price_momentum_3bar')
        if pm3_idx is not None:
            pm3 = X[:, pm3_idx]
            for tf in ['5min', '1h']:
                pos_idx = name_to_idx.get(f'{tf}_position_pct')
                mom_dir_idx = name_to_idx.get(f'{tf}_momentum_direction')
                if pos_idx is not None:
                    pos = X[:, pos_idx]
                    # Buying at top or selling at bottom (contrarian danger)
                    feats.append((pm3 * (pos - 0.5)).reshape(-1, 1))
                    feat_names.append(f'stop_{tf}_momentum_position')

        # Derived: volume anomaly (abnormal volume often precedes whipsaws)
        vr_idx = name_to_idx.get('volume_ratio_20')
        if vr_idx is not None:
            vr = X[:, vr_idx]
            feats.append((vr > 2.0).astype(np.float32).reshape(-1, 1))
            feat_names.append('stop_volume_spike')
            feats.append((vr < 0.5).astype(np.float32).reshape(-1, 1))
            feat_names.append('stop_volume_dry')

        # Derived: health × break_prob interaction (weak channel + high break = stop risk)
        hmin_idx = name_to_idx.get('health_min')
        bp_idx = name_to_idx.get('break_prob_max')
        if hmin_idx is not None and bp_idx is not None:
            feats.append((X[:, bp_idx] * (1 - X[:, hmin_idx])).reshape(-1, 1))
            feat_names.append('stop_fragility_score')

        # Derived: consecutive bars (long streaks about to reverse)
        up_idx = name_to_idx.get('consecutive_up_bars')
        dn_idx = name_to_idx.get('consecutive_down_bars')
        if up_idx is not None and dn_idx is not None:
            streak_max = np.maximum(X[:, up_idx], X[:, dn_idx])
            feats.append(streak_max.reshape(-1, 1))
            feat_names.append('stop_max_streak')

        # Derived: ATR vs channel width (wide ATR + narrow channel = stop risk)
        atr_idx = name_to_idx.get('atr_pct')
        for tf in ['5min', '1h']:
            width_idx = name_to_idx.get(f'{tf}_width_pct')
            if atr_idx is not None and width_idx is not None:
                width = X[:, width_idx]
                safe_width = np.where(width > 1e-8, width, 1e-8)
                feats.append((X[:, atr_idx] / safe_width).reshape(-1, 1))
                feat_names.append(f'stop_{tf}_atr_width_ratio')

        return np.column_stack(feats), feat_names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """
        Train stop loss prediction models.

        Target: Will the next 10 bars see maximum adverse excursion > typical stop distance?
        We use 0.5% as the typical stop distance (matching backtester's ATR-based stops).
        """
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        stop_X_train, self.stop_feature_names = self.derive_stop_features(
            X_train, feature_names)
        stop_X_val, _ = self.derive_stop_features(X_val, feature_names)

        ret5_train = Y_train['future_return_5']
        ret20_train = Y_train['future_return_20']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']

        # Approximate MAE from returns:
        # If 5-bar return is X and 20-bar return is Y, and they have different signs,
        # the max adverse was at least |min(X,Y)| (conservative estimate)
        # If same sign, MAE ≈ max(0, -min(ret5, ret20))

        # For BUY trades: adverse = price going down significantly
        # We compute for both directions and use the worse one
        mae_buy_train = np.maximum(0, -np.minimum(ret5_train, ret20_train))
        mae_sell_train = np.maximum(0, np.maximum(ret5_train, ret20_train))
        mae_approx_train = np.maximum(mae_buy_train, mae_sell_train)

        mae_buy_val = np.maximum(0, -np.minimum(ret5_val, ret20_val))
        mae_sell_val = np.maximum(0, np.maximum(ret5_val, ret20_val))
        mae_approx_val = np.maximum(mae_buy_val, mae_sell_val)

        # Stop hit = MAE > 0.5% (typical stop distance)
        STOP_THRESHOLD = 0.005
        stop_hit_train = (mae_approx_train > STOP_THRESHOLD).astype(np.float32)
        stop_hit_val = (mae_approx_val > STOP_THRESHOLD).astype(np.float32)

        metrics = {}

        print(f"\n  Stop features: {len(self.stop_feature_names)}")
        print(f"  Stop hit rate: {stop_hit_train.mean():.1%}")

        # --- Model 1: Stop hit classifier ---
        print("  Training stop hit classifier...")
        dtrain = lgb.Dataset(stop_X_train, label=stop_hit_train,
                            feature_name=self.stop_feature_names)
        dval = lgb.Dataset(stop_X_val, label=stop_hit_val,
                          feature_name=self.stop_feature_names, reference=dtrain)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 63,
            'learning_rate': 0.03,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'verbose': -1,
            'is_unbalance': True,
            'min_child_samples': 10,
        }

        callbacks = [lgb.early_stopping(80), lgb.log_evaluation(0)]
        self.stop_model = lgb.train(
            params, dtrain, num_boost_round=1000,
            valid_sets=[dval], callbacks=callbacks,
        )

        stop_pred = self.stop_model.predict(stop_X_val)
        stop_auc = roc_auc_score(stop_hit_val, stop_pred)
        metrics['stop_hit_auc'] = float(stop_auc)
        print(f"    Stop Hit AUC: {stop_auc:.3f}")

        # Precision at various thresholds
        for thresh in [0.5, 0.6, 0.7]:
            high_conf = stop_pred > thresh
            if high_conf.sum() > 5:
                precision = stop_hit_val[high_conf].mean()
                coverage = high_conf.mean()
                metrics[f'stop_precision_{int(thresh*100)}'] = float(precision)
                print(f"    Threshold {thresh}: precision={precision:.1%}, "
                      f"coverage={coverage:.1%} ({high_conf.sum()} samples)")

        # --- Model 2: MAE regressor ---
        print("  Training MAE regressor...")
        dtrain2 = lgb.Dataset(stop_X_train, label=mae_approx_train.astype(np.float32),
                             feature_name=self.stop_feature_names)
        dval2 = lgb.Dataset(stop_X_val, label=mae_approx_val.astype(np.float32),
                           feature_name=self.stop_feature_names, reference=dtrain2)

        params_reg = {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
        }

        self.mae_model = lgb.train(
            params_reg, dtrain2, num_boost_round=500,
            valid_sets=[dval2], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        mae_pred = self.mae_model.predict(stop_X_val)
        mae_mae = np.mean(np.abs(mae_pred - mae_approx_val))
        mae_corr = np.corrcoef(mae_pred, mae_approx_val)[0, 1]
        metrics['mae_regression_mae'] = float(mae_mae)
        metrics['mae_regression_corr'] = float(mae_corr)
        print(f"    MAE regression: MAE={mae_mae:.5f}, Corr={mae_corr:.3f}")

        # Feature importance
        imp = self.stop_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 stop-loss features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.stop_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        """
        Predict stop loss hit probability.

        Returns:
            stop_prob: probability of hitting stop loss
            expected_mae: expected max adverse excursion (as % of price)
            risk_level: 'safe', 'caution', 'danger'
        """
        if self.stop_model is None:
            return {}

        stop_X, _ = self.derive_stop_features(X, self.feature_names)

        stop_prob = self.stop_model.predict(stop_X)
        expected_mae = self.mae_model.predict(stop_X)

        risk = np.where(
            stop_prob > 0.55, 2,  # danger
            np.where(stop_prob > 0.35, 1, 0)  # caution / safe
        )
        risk_labels = np.array(['safe', 'caution', 'danger'])

        return {
            'stop_prob': stop_prob,
            'expected_mae': expected_mae,
            'risk_level': risk_labels[risk],
            'risk_level_id': risk,
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'stop_model': self.stop_model,
                'mae_model': self.mae_model,
                'feature_names': self.feature_names,
                'stop_feature_names': self.stop_feature_names,
            }, f)
        print(f"  Saved StopLossPredictor to {path}")

    @classmethod
    def load(cls, path: str) -> 'StopLossPredictor':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.stop_model = data['stop_model']
        model.mae_model = data['mae_model']
        model.feature_names = data['feature_names']
        model.stop_feature_names = data['stop_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 19: Bayesian Signal Combiner
# ---------------------------------------------------------------------------

class BayesianSignalCombiner:
    """
    Replaces sequential confidence multipliers with a single learned model.

    Problem: 15 active models each apply confidence *= {0.60..1.15},
    creating a multiplicative chain where small errors compound.
    A signal penalized by 5 models at 0.90 each becomes 0.59x — too aggressive.

    Solution: Collect ALL model outputs as features, learn one optimal
    combination model that outputs a single confidence adjustment.

    This model is trained to predict:
    1. Whether the signal results in a winning trade (binary)
    2. The optimal confidence scaling factor (regression)

    Replaces: All the ad-hoc confidence *= 0.XX rules in the backtest loop.
    """

    def __init__(self):
        self.win_model = None          # Binary: will this signal win?
        self.scale_model = None        # Regression: optimal confidence scale
        self.feature_names = None
        self.combined_feature_names = None

    def _collect_all_model_outputs(self, X, feature_names, model_dir='surfer_models'):
        """
        Run all available models and collect their outputs as features.
        Returns (meta_X, meta_names).
        """
        import os as _os

        n = len(X)
        meta_feats = []
        meta_names = []

        # 1. Base physics features (subset — most informative)
        KEY_FEATURES = [
            'break_prob_max', 'break_prob_weighted', 'avg_entropy',
            'direction_consensus', 'health_min', 'health_max',
            'health_spread', 'confluence_score', 'atr_pct', 'rsi_14',
            'rsi_5', 'volume_ratio_20', 'price_momentum_3bar',
            'price_momentum_12bar', 'vix_level',
        ]
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        for key in KEY_FEATURES:
            idx = name_to_idx.get(key)
            if idx is not None:
                meta_feats.append(X[:, idx])
                meta_names.append(f'bc_{key}')

        # 2. GBT model outputs
        gbt_path = _os.path.join(model_dir, 'gbt_model.pkl')
        if _os.path.exists(gbt_path):
            try:
                gbt = GBTModel.load(gbt_path)
                pred = gbt.predict(X)
                if 'action_probs' in pred:
                    for i, name in enumerate(['hold', 'buy', 'sell']):
                        meta_feats.append(pred['action_probs'][:, i])
                        meta_names.append(f'bc_gbt_{name}_prob')
                if 'lifetime' in pred:
                    meta_feats.append(pred['lifetime'])
                    meta_names.append('bc_gbt_lifetime')
            except Exception:
                pass

        # 3. Regime model
        regime_path = _os.path.join(model_dir, 'regime_model.pkl')
        if _os.path.exists(regime_path):
            try:
                regime = RegimeConditionalModel.load(regime_path)
                pred = regime.predict(X)
                if 'action' in pred:
                    meta_feats.append(pred['action'].astype(np.float32))
                    meta_names.append('bc_regime_action')
                if 'regime_id' in pred:
                    meta_feats.append(pred['regime_id'].astype(np.float32))
                    meta_names.append('bc_regime_id')
            except Exception:
                pass

        # 4. CV Ensemble
        cv_path = _os.path.join(model_dir, 'cv_ensemble_model.pkl')
        if _os.path.exists(cv_path):
            try:
                cv = CVEnsembleModel.load(cv_path)
                pred = cv.predict(X)
                if 'action' in pred:
                    meta_feats.append(pred['action'].astype(np.float32))
                    meta_names.append('bc_cv_action')
                if 'consensus' in pred:
                    meta_feats.append(pred['consensus'])
                    meta_names.append('bc_cv_consensus')
            except Exception:
                pass

        # 5. Physics Residual
        res_path = _os.path.join(model_dir, 'physics_residual_model.pkl')
        if _os.path.exists(res_path):
            try:
                res = PhysicsResidualModel.load(res_path)
                pred = res.predict(X)
                if 'confidence_scale' in pred:
                    meta_feats.append(pred['confidence_scale'])
                    meta_names.append('bc_residual_conf_scale')
                if 'lifetime_correction' in pred:
                    meta_feats.append(pred['lifetime_correction'])
                    meta_names.append('bc_residual_life_corr')
            except Exception:
                pass

        # 6. Adverse Movement
        adv_path = _os.path.join(model_dir, 'adverse_movement_model.pkl')
        if _os.path.exists(adv_path):
            try:
                adv = AdverseMovementPredictor.load(adv_path)
                # Predict for buy direction
                pred_buy = adv.predict(X, is_buy=True)
                if 'stop_prob' in pred_buy:
                    meta_feats.append(pred_buy['stop_prob'])
                    meta_names.append('bc_adv_stop_prob_buy')
                if 'viable_prob' in pred_buy:
                    meta_feats.append(pred_buy['viable_prob'])
                    meta_names.append('bc_adv_viable_buy')
                # Predict for sell
                pred_sell = adv.predict(X, is_buy=False)
                if 'stop_prob' in pred_sell:
                    meta_feats.append(pred_sell['stop_prob'])
                    meta_names.append('bc_adv_stop_prob_sell')
            except Exception:
                pass

        # 7. Volatility Transition
        vol_path = _os.path.join(model_dir, 'vol_transition_model.pkl')
        if _os.path.exists(vol_path):
            try:
                vol = VolatilityTransitionModel.load(vol_path)
                pred = vol.predict(X)
                if 'spike_prob' in pred:
                    meta_feats.append(pred['spike_prob'])
                    meta_names.append('bc_vol_spike_prob')
                if 'expansion_prob' in pred:
                    meta_feats.append(pred['expansion_prob'])
                    meta_names.append('bc_vol_expansion_prob')
            except Exception:
                pass

        # 8. Exhaustion
        exh_path = _os.path.join(model_dir, 'exhaustion_model.pkl')
        if _os.path.exists(exh_path):
            try:
                exh = MomentumExhaustionDetector.load(exh_path)
                pred = exh.predict(X)
                if 'exhaustion_prob' in pred:
                    meta_feats.append(pred['exhaustion_prob'])
                    meta_names.append('bc_exhaustion_prob')
                if 'reversal_severity' in pred:
                    meta_feats.append(pred['reversal_severity'])
                    meta_names.append('bc_reversal_severity')
            except Exception:
                pass

        # 9. Cross-Asset
        ca_path = _os.path.join(model_dir, 'cross_asset_model.pkl')
        if _os.path.exists(ca_path):
            try:
                ca = CrossAssetAmplifier.load(ca_path)
                pred = ca.predict(X)
                if 'market_regime' in pred:
                    meta_feats.append(pred['market_regime'].astype(np.float32))
                    meta_names.append('bc_market_regime')
                if 'confidence_scale' in pred:
                    meta_feats.append(pred['confidence_scale'])
                    meta_names.append('bc_ca_conf_scale')
            except Exception:
                pass

        # 10. Stop Loss
        sl_path = _os.path.join(model_dir, 'stop_loss_model.pkl')
        if _os.path.exists(sl_path):
            try:
                sl = StopLossPredictor.load(sl_path)
                pred = sl.predict(X)
                if 'stop_prob' in pred:
                    meta_feats.append(pred['stop_prob'])
                    meta_names.append('bc_stop_loss_prob')
                if 'expected_mae' in pred:
                    meta_feats.append(pred['expected_mae'])
                    meta_names.append('bc_expected_mae')
            except Exception:
                pass

        if len(meta_feats) == 0:
            return X, list(feature_names)

        return np.column_stack(meta_feats), meta_names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names, model_dir='surfer_models'):
        """
        Train Bayesian combiner on all model outputs.

        Target 1: Binary — is this a winning trade?
        Target 2: Regression — optimal confidence scale (1.0 = neutral)
        """
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)

        print("  Collecting all model outputs as features...")
        meta_X_train, self.combined_feature_names = self._collect_all_model_outputs(
            X_train, feature_names, model_dir)
        meta_X_val, _ = self._collect_all_model_outputs(
            X_val, feature_names, model_dir)

        # Target 1: Winning trade (would a BUY or SELL be profitable?)
        ret20 = Y_train['future_return_20']
        ret5 = Y_train['future_return_5']
        # Win = positive return in the dominant direction (max of |ret5|, |ret20|)
        win_train = (np.abs(ret20) > 0.003).astype(np.float32)  # Significant move
        win_val = (np.abs(Y_val['future_return_20']) > 0.003).astype(np.float32)

        # Target 2: Confidence scale (based on magnitude of favorable move)
        # Higher return = should have been higher confidence
        scale_train = np.clip(1.0 + ret20 * 50, 0.3, 2.0).astype(np.float32)
        scale_val = np.clip(1.0 + Y_val['future_return_20'] * 50, 0.3, 2.0).astype(np.float32)

        metrics = {}

        print(f"  Combined features: {len(self.combined_feature_names)}")
        print(f"  Win rate: {win_train.mean():.1%}")

        # --- Model 1: Win classifier ---
        print("  Training win classifier...")
        dtrain = lgb.Dataset(meta_X_train, label=win_train,
                            feature_name=self.combined_feature_names)
        dval = lgb.Dataset(meta_X_val, label=win_val,
                          feature_name=self.combined_feature_names, reference=dtrain)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
        }

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        self.win_model = lgb.train(
            params, dtrain, num_boost_round=500,
            valid_sets=[dval], callbacks=callbacks,
        )

        win_pred = self.win_model.predict(meta_X_val)
        win_auc = roc_auc_score(win_val, win_pred)
        metrics['win_auc'] = float(win_auc)
        print(f"    Win AUC: {win_auc:.3f}")

        # --- Model 2: Scale regressor ---
        print("  Training confidence scale regressor...")
        dtrain2 = lgb.Dataset(meta_X_train, label=scale_train,
                             feature_name=self.combined_feature_names)
        dval2 = lgb.Dataset(meta_X_val, label=scale_val,
                           feature_name=self.combined_feature_names, reference=dtrain2)

        params_reg = {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
        }

        self.scale_model = lgb.train(
            params_reg, dtrain2, num_boost_round=500,
            valid_sets=[dval2], callbacks=callbacks,
        )

        scale_pred = self.scale_model.predict(meta_X_val)
        scale_mae = np.mean(np.abs(scale_pred - scale_val))
        scale_corr = np.corrcoef(scale_pred, scale_val)[0, 1]
        metrics['scale_mae'] = float(scale_mae)
        metrics['scale_corr'] = float(scale_corr)
        print(f"    Scale MAE: {scale_mae:.3f}, Corr: {scale_corr:.3f}")

        # Feature importance
        imp = self.win_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:15]
        print("\n  Top 15 Bayesian combiner features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.combined_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray, model_dir='surfer_models') -> dict:
        """
        Predict optimal confidence scaling.

        Returns:
            win_prob: probability this is a winning trade
            confidence_scale: optimal confidence multiplier [0.3, 2.0]
        """
        if self.win_model is None:
            return {}

        meta_X, _ = self._collect_all_model_outputs(X, self.feature_names, model_dir)

        win_prob = self.win_model.predict(meta_X)
        conf_scale = self.scale_model.predict(meta_X)
        conf_scale = np.clip(conf_scale, 0.5, 1.5)

        return {
            'win_prob': win_prob,
            'confidence_scale': conf_scale,
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'win_model': self.win_model,
                'scale_model': self.scale_model,
                'feature_names': self.feature_names,
                'combined_feature_names': self.combined_feature_names,
            }, f)
        print(f"  Saved BayesianSignalCombiner to {path}")

    @classmethod
    def load(cls, path: str) -> 'BayesianSignalCombiner':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.win_model = data['win_model']
        model.scale_model = data['scale_model']
        model.feature_names = data['feature_names']
        model.combined_feature_names = data['combined_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 20: Dynamic Trail Optimizer
# ---------------------------------------------------------------------------

class DynamicTrailOptimizer:
    """
    Learns optimal trailing stop distance from market conditions.

    Instead of hardcoded trail tightening ratios, this model learns
    WHEN to tighten vs relax based on current features.

    Output: recommended trail_factor and tighten probability.
    """

    def __init__(self):
        self.trail_model = None
        self.tighten_model = None
        self.feature_names = None
        self.trail_feature_names = None

    @staticmethod
    def derive_trail_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """Derive features for trail optimization."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Momentum features
        for key in ['price_momentum_3bar', 'price_momentum_12bar',
                     'rsi_14', 'rsi_5', 'rsi_slope_5bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'trail_{key}')

        # Momentum deceleration
        pm3_idx = name_to_idx.get('price_momentum_3bar')
        pm12_idx = name_to_idx.get('price_momentum_12bar')
        if pm3_idx is not None and pm12_idx is not None:
            feats.append(X[:, pm3_idx] - X[:, pm12_idx])
            names.append('trail_momentum_decel')

        # Volatility
        atr_idx = name_to_idx.get('atr_pct')
        if atr_idx is not None:
            feats.append(X[:, atr_idx])
            names.append('trail_atr')

        # Volume
        for key in ['volume_ratio_20', 'volume_trend_5', 'vol_momentum_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'trail_{key}')

        # Channel health
        for key in ['health_min', 'health_max', 'health_delta_3bar', 'health_delta_6bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'trail_{key}')

        # Break probability
        for key in ['break_prob_max', 'break_prob_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'trail_{key}')

        # Entropy
        for key in ['avg_entropy', 'entropy_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'trail_{key}')

        # Position in channel
        for tf in ['5min', '1h', '4h']:
            pos_idx = name_to_idx.get(f'{tf}_position_pct')
            if pos_idx is not None:
                pos = X[:, pos_idx]
                feats.append(pos)
                names.append(f'trail_{tf}_position')
                feats.append(np.minimum(pos, 1.0 - pos))
                names.append(f'trail_{tf}_boundary_dist')

        # Direction consensus
        dc_idx = name_to_idx.get('direction_consensus')
        if dc_idx is not None:
            feats.append(X[:, dc_idx])
            names.append('trail_direction_consensus')

        # Bar characteristics
        for key in ['bar_range_pct', 'close_position_in_bar',
                     'consecutive_up_bars', 'consecutive_down_bars']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'trail_{key}')

        # VIX
        vix_idx = name_to_idx.get('vix_level')
        if vix_idx is not None:
            feats.append(X[:, vix_idx])
            names.append('trail_vix')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train trail optimizer on optimal trail targets."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        trail_X_train, self.trail_feature_names = self.derive_trail_features(
            X_train, feature_names)
        trail_X_val, _ = self.derive_trail_features(X_val, feature_names)

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']

        pm3_idx = {n: i for i, n in enumerate(feature_names)}.get('price_momentum_3bar')

        def compute_trail_target(X_data, r5, r20):
            n = len(X_data)
            trail = np.full(n, 0.4, dtype=np.float32)
            if pm3_idx is not None:
                mom = X_data[:, pm3_idx]
                cont = ((mom > 0.001) & (r5 > 0.001)) | ((mom < -0.001) & (r5 < -0.001))
                trail[cont] = 0.7
                strong_cont = cont & (((mom > 0.001) & (r20 > 0.003)) | ((mom < -0.001) & (r20 < -0.003)))
                trail[strong_cont] = 0.9
                reversal = ((mom > 0.001) & (r5 < -0.002)) | ((mom < -0.001) & (r5 > 0.002))
                trail[reversal] = 0.15
                strong_rev = reversal & (np.abs(r5) > 0.005)
                trail[strong_rev] = 0.08
            return trail

        trail_train = compute_trail_target(X_train, ret5, ret20)
        trail_val = compute_trail_target(X_val, ret5_val, ret20_val)
        tighten_train = (trail_train < 0.3).astype(np.float32)
        tighten_val = (trail_val < 0.3).astype(np.float32)

        metrics = {}
        print(f"\n  Trail features: {len(self.trail_feature_names)}")
        print(f"  Tighten rate: {tighten_train.mean():.1%}")

        # Trail regressor
        print("  Training trail factor regressor...")
        dtrain = lgb.Dataset(trail_X_train, label=trail_train, feature_name=self.trail_feature_names)
        dval = lgb.Dataset(trail_X_val, label=trail_val, feature_name=self.trail_feature_names, reference=dtrain)
        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]

        self.trail_model = lgb.train(
            {'objective': 'regression', 'metric': 'mae', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        trail_pred = self.trail_model.predict(trail_X_val)
        metrics['trail_mae'] = float(np.mean(np.abs(trail_pred - trail_val)))
        metrics['trail_corr'] = float(np.corrcoef(trail_pred, trail_val)[0, 1])
        print(f"    Trail MAE: {metrics['trail_mae']:.3f}, Corr: {metrics['trail_corr']:.3f}")

        # Tighten classifier
        print("  Training tighten classifier...")
        dtrain2 = lgb.Dataset(trail_X_train, label=tighten_train, feature_name=self.trail_feature_names)
        dval2 = lgb.Dataset(trail_X_val, label=tighten_val, feature_name=self.trail_feature_names, reference=dtrain2)

        self.tighten_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'is_unbalance': True},
            dtrain2, num_boost_round=500, valid_sets=[dval2], callbacks=callbacks)

        tighten_pred = self.tighten_model.predict(trail_X_val)
        metrics['tighten_auc'] = float(roc_auc_score(tighten_val, tighten_pred))
        print(f"    Tighten AUC: {metrics['tighten_auc']:.3f}")

        imp = self.trail_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 trail features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.trail_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.trail_model is None:
            return {}
        trail_X, _ = self.derive_trail_features(X, self.feature_names)
        return {
            'trail_factor': np.clip(self.trail_model.predict(trail_X), 0.05, 1.0),
            'tighten_prob': self.tighten_model.predict(trail_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'trail_model': self.trail_model, 'tighten_model': self.tighten_model,
                'feature_names': self.feature_names, 'trail_feature_names': self.trail_feature_names,
            }, f)
        print(f"  Saved DynamicTrailOptimizer to {path}")

    @classmethod
    def load(cls, path: str) -> 'DynamicTrailOptimizer':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.trail_model = data['trail_model']
        model.tighten_model = data['tighten_model']
        model.feature_names = data['feature_names']
        model.trail_feature_names = data['trail_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 21: Intraday Session Model
# ---------------------------------------------------------------------------

class IntradaySessionModel:
    """
    Learns intraday session patterns for signal quality.

    Trading sessions have distinct characteristics:
    - Open auction (9:30-10:00): High volatility, reversals common
    - Morning momentum (10:00-11:30): Trends establish, best entries
    - Midday lull (11:30-13:00): Low volume, choppy, worst entries
    - Afternoon (13:00-15:00): Institutional flow, moderate
    - Power hour (15:00-16:00): Volume surge, trend continuation or reversal

    This model learns which sessions produce winning trades and adjusts
    signal confidence accordingly.

    Output: session_quality (0-1) and session_win_prob.
    """

    def __init__(self):
        self.quality_model = None
        self.win_model = None
        self.feature_names = None
        self.session_feature_names = None

    @staticmethod
    def derive_session_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """Derive intraday-session-specific features."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Core time features
        for key in ['minutes_since_open', 'hour_sin', 'hour_cos', 'day_of_week']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'sess_{key}')

        # Session bucket encoding (one-hot-ish from minutes_since_open)
        mso_idx = name_to_idx.get('minutes_since_open')
        if mso_idx is not None:
            mso = X[:, mso_idx]
            # Open auction: 0-30 min
            feats.append((mso < 30).astype(np.float32))
            names.append('sess_open_auction')
            # Morning momentum: 30-120 min
            feats.append(((mso >= 30) & (mso < 120)).astype(np.float32))
            names.append('sess_morning_momentum')
            # Midday lull: 120-210 min
            feats.append(((mso >= 120) & (mso < 210)).astype(np.float32))
            names.append('sess_midday_lull')
            # Afternoon: 210-330 min
            feats.append(((mso >= 210) & (mso < 330)).astype(np.float32))
            names.append('sess_afternoon')
            # Power hour: 330-390 min
            feats.append((mso >= 330).astype(np.float32))
            names.append('sess_power_hour')
            # Continuous session progress (0=open, 1=close)
            feats.append(np.clip(mso / 390.0, 0.0, 1.0))
            names.append('sess_progress')
            # Time to close (urgency)
            feats.append(np.clip((390.0 - mso) / 390.0, 0.0, 1.0))
            names.append('sess_time_remaining')

        # Volume context relative to time of day
        vol_idx = name_to_idx.get('volume_ratio_20')
        if vol_idx is not None:
            feats.append(X[:, vol_idx])
            names.append('sess_volume_ratio')
            # Volume anomaly (high volume at lull = unusual = important)
            if mso_idx is not None:
                mso = X[:, mso_idx]
                is_lull = ((mso >= 120) & (mso < 210)).astype(np.float32)
                feats.append(X[:, vol_idx] * is_lull)
                names.append('sess_lull_volume_anomaly')

        # Volatility context
        atr_idx = name_to_idx.get('atr_pct')
        if atr_idx is not None:
            feats.append(X[:, atr_idx])
            names.append('sess_atr')

        # RSI at time of day
        for key in ['rsi_14', 'rsi_5']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'sess_{key}')

        # Momentum context
        for key in ['price_momentum_3bar', 'price_momentum_12bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'sess_{key}')

        # Channel health at this time
        for key in ['health_min', 'health_max', 'break_prob_max']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'sess_{key}')

        # Bar characteristics
        for key in ['bar_range_pct', 'close_position_in_bar',
                     'consecutive_up_bars', 'consecutive_down_bars']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'sess_{key}')

        # VIX level
        vix_idx = name_to_idx.get('vix_level')
        if vix_idx is not None:
            feats.append(X[:, vix_idx])
            names.append('sess_vix')

        # Day of week interactions
        dow_idx = name_to_idx.get('day_of_week')
        if dow_idx is not None and mso_idx is not None:
            dow = X[:, dow_idx]
            mso = X[:, mso_idx]
            # Monday morning (often gap fill)
            feats.append(((dow < 0.5) & (mso < 60)).astype(np.float32))
            names.append('sess_monday_morning')
            # Friday afternoon (often position unwinding)
            feats.append(((dow > 3.5) & (mso > 300)).astype(np.float32))
            names.append('sess_friday_afternoon')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train session quality model."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        sess_X_train, self.session_feature_names = self.derive_session_features(
            X_train, feature_names)
        sess_X_val, _ = self.derive_session_features(X_val, feature_names)

        # Target: session quality = whether a 5-bar trade starting here would be profitable
        # Use abs(ret5) > abs(ret20)/4 as "good entry timing" proxy
        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']

        # Quality = good directional move (not choppy)
        # High quality: |ret5| > 0.003 AND sign(ret5) == sign(ret20) → trend continuation
        quality_train = (
            (np.abs(ret5) > 0.003) &
            (np.sign(ret5) == np.sign(ret20))
        ).astype(np.float32)
        quality_val = (
            (np.abs(ret5_val) > 0.003) &
            (np.sign(ret5_val) == np.sign(ret20_val))
        ).astype(np.float32)

        # Win = positive return within 20 bars (simplified)
        win_train = (ret20 > 0.002).astype(np.float32)
        win_val = (ret20_val > 0.002).astype(np.float32)

        metrics = {}
        print(f"\n  Session features: {len(self.session_feature_names)}")
        print(f"  Quality rate: {quality_train.mean():.1%}")
        print(f"  Win rate: {win_train.mean():.1%}")

        # Quality classifier
        print("  Training session quality classifier...")
        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(sess_X_train, label=quality_train,
                            feature_name=self.session_feature_names)
        dval = lgb.Dataset(sess_X_val, label=quality_val,
                          feature_name=self.session_feature_names, reference=dtrain)

        self.quality_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'is_unbalance': True},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        quality_pred = self.quality_model.predict(sess_X_val)
        try:
            metrics['quality_auc'] = float(roc_auc_score(quality_val, quality_pred))
        except ValueError:
            metrics['quality_auc'] = 0.5
        print(f"    Quality AUC: {metrics['quality_auc']:.3f}")

        # Win probability model
        print("  Training session win probability model...")
        dtrain2 = lgb.Dataset(sess_X_train, label=win_train,
                             feature_name=self.session_feature_names)
        dval2 = lgb.Dataset(sess_X_val, label=win_val,
                           feature_name=self.session_feature_names, reference=dtrain2)

        self.win_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'is_unbalance': True},
            dtrain2, num_boost_round=500, valid_sets=[dval2], callbacks=callbacks)

        win_pred = self.win_model.predict(sess_X_val)
        try:
            metrics['win_auc'] = float(roc_auc_score(win_val, win_pred))
        except ValueError:
            metrics['win_auc'] = 0.5
        print(f"    Win AUC: {metrics['win_auc']:.3f}")

        # Feature importance
        imp = self.quality_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 session features (quality):")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.session_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.quality_model is None:
            return {}
        sess_X, _ = self.derive_session_features(X, self.feature_names)
        return {
            'session_quality': self.quality_model.predict(sess_X),
            'session_win_prob': self.win_model.predict(sess_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'quality_model': self.quality_model, 'win_model': self.win_model,
                'feature_names': self.feature_names,
                'session_feature_names': self.session_feature_names,
            }, f)
        print(f"  Saved IntradaySessionModel to {path}")

    @classmethod
    def load(cls, path: str) -> 'IntradaySessionModel':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.quality_model = data['quality_model']
        model.win_model = data['win_model']
        model.feature_names = data['feature_names']
        model.session_feature_names = data['session_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 22: Channel Maturity Predictor
# ---------------------------------------------------------------------------

class ChannelMaturityPredictor:
    """
    Predicts where a channel is in its lifecycle.

    Young channels (just formed) → more room for profit, wider stops OK
    Mature channels (about to break) → take profit, tighten stops
    Old channels (overstayed) → expect break, reduce position size

    Uses health trajectory, entropy acceleration, width changes, and
    break probability evolution to estimate remaining life.

    Output: maturity_score (0=young, 1=about to break), remaining_life_estimate
    """

    def __init__(self):
        self.maturity_model = None
        self.remaining_model = None
        self.feature_names = None
        self.maturity_feature_names = None

    @staticmethod
    def derive_maturity_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """Derive channel maturity features."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Health trajectory (declining health = aging channel)
        for key in ['health_min', 'health_max', 'health_spread',
                     'health_delta_3bar', 'health_delta_6bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'mat_{key}')

        # Health acceleration (delta of delta approximation)
        hd3_idx = name_to_idx.get('health_delta_3bar')
        hd6_idx = name_to_idx.get('health_delta_6bar')
        if hd3_idx is not None and hd6_idx is not None:
            feats.append(X[:, hd3_idx] - X[:, hd6_idx] / 2.0)
            names.append('mat_health_accel')

        # Entropy (rising entropy = channel destabilizing)
        for key in ['avg_entropy', 'entropy_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'mat_{key}')

        # Break probability evolution
        for key in ['break_prob_max', 'break_prob_weighted', 'break_prob_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'mat_{key}')

        # Width dynamics (narrowing = mature squeeze, widening = new channel)
        for key in ['width_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'mat_{key}')

        # Per-TF width and r_squared (aging affects quality)
        for tf in ['5min', '1h', '4h', 'daily']:
            for feat in ['r_squared', 'width_pct', 'channel_health',
                         'break_prob', 'entropy', 'bounce_count']:
                key = f'{tf}_{feat}'
                idx = name_to_idx.get(key)
                if idx is not None:
                    feats.append(X[:, idx])
                    names.append(f'mat_{key}')

        # Channel quality indicators
        for key in ['direction_consensus', 'confluence_score', 'squeeze_any',
                     'valid_tf_count', 'theta_spread']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'mat_{key}')

        # Energy dynamics
        for key in ['energy_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'mat_{key}')

        # Per-TF energy and OU parameters
        for tf in ['5min', '1h', '4h']:
            for feat in ['total_energy', 'ou_theta', 'ou_half_life']:
                key = f'{tf}_{feat}'
                idx = name_to_idx.get(key)
                if idx is not None:
                    feats.append(X[:, idx])
                    names.append(f'mat_{key}')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train channel maturity model."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        mat_X_train, self.maturity_feature_names = self.derive_maturity_features(
            X_train, feature_names)
        mat_X_val, _ = self.derive_maturity_features(X_val, feature_names)

        lifetime = Y_train['channel_lifetime']
        lifetime_val = Y_val['channel_lifetime']

        # Maturity target: is the channel near its end?
        # "Mature" = less than 15 bars remaining (will break soon)
        mature_train = (lifetime < 15).astype(np.float32)
        mature_val = (lifetime_val < 15).astype(np.float32)

        # Remaining life as regression target (capped at 100)
        remaining_train = np.clip(lifetime, 0, 100).astype(np.float32)
        remaining_val = np.clip(lifetime_val, 0, 100).astype(np.float32)

        metrics = {}
        print(f"\n  Maturity features: {len(self.maturity_feature_names)}")
        print(f"  Mature (< 15 bars) rate: {mature_train.mean():.1%}")
        print(f"  Mean remaining lifetime: {remaining_train.mean():.1f} bars")

        # Maturity classifier
        print("  Training maturity classifier...")
        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(mat_X_train, label=mature_train,
                            feature_name=self.maturity_feature_names)
        dval = lgb.Dataset(mat_X_val, label=mature_val,
                          feature_name=self.maturity_feature_names, reference=dtrain)

        self.maturity_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'is_unbalance': True},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        mat_pred = self.maturity_model.predict(mat_X_val)
        try:
            metrics['maturity_auc'] = float(roc_auc_score(mature_val, mat_pred))
        except ValueError:
            metrics['maturity_auc'] = 0.5
        print(f"    Maturity AUC: {metrics['maturity_auc']:.3f}")

        # Remaining life regressor
        print("  Training remaining lifetime regressor...")
        dtrain2 = lgb.Dataset(mat_X_train, label=remaining_train,
                             feature_name=self.maturity_feature_names)
        dval2 = lgb.Dataset(mat_X_val, label=remaining_val,
                           feature_name=self.maturity_feature_names, reference=dtrain2)

        self.remaining_model = lgb.train(
            {'objective': 'regression', 'metric': 'mae', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1},
            dtrain2, num_boost_round=500, valid_sets=[dval2], callbacks=callbacks)

        rem_pred = self.remaining_model.predict(mat_X_val)
        metrics['remaining_mae'] = float(np.mean(np.abs(rem_pred - remaining_val)))
        metrics['remaining_corr'] = float(np.corrcoef(rem_pred, remaining_val)[0, 1])
        print(f"    Remaining MAE: {metrics['remaining_mae']:.1f} bars, Corr: {metrics['remaining_corr']:.3f}")

        # Feature importance
        imp = self.maturity_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 maturity features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.maturity_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.maturity_model is None:
            return {}
        mat_X, _ = self.derive_maturity_features(X, self.feature_names)
        return {
            'maturity_prob': self.maturity_model.predict(mat_X),
            'remaining_life': np.clip(self.remaining_model.predict(mat_X), 0, 200),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'maturity_model': self.maturity_model,
                'remaining_model': self.remaining_model,
                'feature_names': self.feature_names,
                'maturity_feature_names': self.maturity_feature_names,
            }, f)
        print(f"  Saved ChannelMaturityPredictor to {path}")

    @classmethod
    def load(cls, path: str) -> 'ChannelMaturityPredictor':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.maturity_model = data['maturity_model']
        model.remaining_model = data['remaining_model']
        model.feature_names = data['feature_names']
        model.maturity_feature_names = data['maturity_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 23: Multi-Scale Momentum Divergence
# ---------------------------------------------------------------------------

class MultiScaleMomentumModel:
    """
    Creates momentum spectrum across scales and detects divergences.

    Momentum at different scales tells different stories:
    - 1-3 bar: noise/microstructure
    - 5-8 bar: short-term trend
    - 13-21 bar: medium-term trend

    When short-term reverses while long-term continues → pullback entry
    When all scales align → strong trend, ride it
    When all scales diverge → choppy, avoid

    Output: momentum_regime, trend_strength, divergence_score
    """

    def __init__(self):
        self.regime_model = None
        self.strength_model = None
        self.feature_names = None
        self.mom_feature_names = None

    @staticmethod
    def derive_momentum_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """Derive multi-scale momentum features."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Base momentum at two scales
        pm3_idx = name_to_idx.get('price_momentum_3bar')
        pm12_idx = name_to_idx.get('price_momentum_12bar')

        if pm3_idx is not None:
            pm3 = X[:, pm3_idx]
            feats.append(pm3)
            names.append('mom_3bar')
            feats.append(np.abs(pm3))
            names.append('mom_3bar_abs')

        if pm12_idx is not None:
            pm12 = X[:, pm12_idx]
            feats.append(pm12)
            names.append('mom_12bar')
            feats.append(np.abs(pm12))
            names.append('mom_12bar_abs')

        # Momentum divergence (short vs long)
        if pm3_idx is not None and pm12_idx is not None:
            pm3 = X[:, pm3_idx]
            pm12 = X[:, pm12_idx]
            feats.append(pm3 - pm12)
            names.append('mom_divergence_3_12')
            feats.append(np.sign(pm3) * np.sign(pm12))
            names.append('mom_sign_agreement')
            # Momentum ratio (acceleration)
            safe_pm12 = np.where(np.abs(pm12) > 1e-6, pm12, 1e-6 * np.sign(pm12 + 1e-10))
            feats.append(np.clip(pm3 / safe_pm12, -5, 5))
            names.append('mom_ratio_3_12')

        # RSI as momentum proxy at two scales
        rsi14_idx = name_to_idx.get('rsi_14')
        rsi5_idx = name_to_idx.get('rsi_5')
        rsi_slope_idx = name_to_idx.get('rsi_slope_5bar')

        if rsi14_idx is not None:
            rsi14 = X[:, rsi14_idx]
            feats.append(rsi14)
            names.append('mom_rsi14')
            # RSI distance from neutral (50)
            feats.append(np.abs(rsi14 - 50))
            names.append('mom_rsi14_extremity')

        if rsi5_idx is not None:
            rsi5 = X[:, rsi5_idx]
            feats.append(rsi5)
            names.append('mom_rsi5')

        # RSI divergence (short RSI vs long RSI)
        if rsi14_idx is not None and rsi5_idx is not None:
            feats.append(X[:, rsi5_idx] - X[:, rsi14_idx])
            names.append('mom_rsi_divergence')

        if rsi_slope_idx is not None:
            feats.append(X[:, rsi_slope_idx])
            names.append('mom_rsi_slope')

        # Per-TF momentum direction consensus
        mom_dirs = []
        for tf in ['5min', '1h', '4h', 'daily', 'weekly']:
            md_idx = name_to_idx.get(f'{tf}_momentum_direction')
            if md_idx is not None:
                mom_dirs.append(X[:, md_idx])
                feats.append(X[:, md_idx])
                names.append(f'mom_{tf}_direction')

        if len(mom_dirs) >= 2:
            mom_stack = np.column_stack(mom_dirs)
            # Cross-TF momentum alignment
            feats.append(np.mean(mom_stack, axis=1))
            names.append('mom_mean_direction')
            feats.append(np.std(mom_stack, axis=1))
            names.append('mom_direction_std')
            # Short TF vs long TF divergence
            if len(mom_dirs) >= 3:
                short_avg = np.mean(np.column_stack(mom_dirs[:2]), axis=1)
                long_avg = np.mean(np.column_stack(mom_dirs[2:]), axis=1)
                feats.append(short_avg - long_avg)
                names.append('mom_short_vs_long_divergence')

        # Consecutive bars (trend persistence)
        for key in ['consecutive_up_bars', 'consecutive_down_bars']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'mom_{key}')

        # Bar character (close position in range)
        cp_idx = name_to_idx.get('close_position_in_bar')
        if cp_idx is not None:
            feats.append(X[:, cp_idx])
            names.append('mom_close_position')

        # Volume confirmation of momentum
        vol_idx = name_to_idx.get('volume_ratio_20')
        vol_mom_idx = name_to_idx.get('vol_momentum_3bar')
        if vol_idx is not None:
            feats.append(X[:, vol_idx])
            names.append('mom_volume_ratio')
        if vol_mom_idx is not None:
            feats.append(X[:, vol_mom_idx])
            names.append('mom_vol_momentum')

        # Volume-momentum interaction
        if vol_idx is not None and pm3_idx is not None:
            feats.append(X[:, vol_idx] * np.abs(X[:, pm3_idx]))
            names.append('mom_vol_momentum_interaction')

        # Position in channels (at boundaries = reversal likely)
        for tf in ['5min', '1h', '4h']:
            pos_idx = name_to_idx.get(f'{tf}_position_pct')
            if pos_idx is not None:
                pos = X[:, pos_idx]
                feats.append(pos)
                names.append(f'mom_{tf}_channel_pos')

        # Direction consensus
        dc_idx = name_to_idx.get('direction_consensus')
        if dc_idx is not None:
            feats.append(X[:, dc_idx])
            names.append('mom_direction_consensus')

        # Bullish/bearish fraction
        for key in ['bullish_fraction', 'bearish_fraction']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'mom_{key}')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train momentum divergence model."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        mom_X_train, self.mom_feature_names = self.derive_momentum_features(
            X_train, feature_names)
        mom_X_val, _ = self.derive_momentum_features(X_val, feature_names)

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret60 = Y_train['future_return_60']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']
        ret60_val = Y_val['future_return_60']

        # Momentum regime target:
        # "trending" = all returns same sign and growing
        # "reversing" = short-term opposite to long-term
        # "choppy" = no clear direction
        def compute_regime(r5, r20, r60):
            trending = (
                (np.sign(r5) == np.sign(r20)) &
                (np.sign(r20) == np.sign(r60)) &
                (np.abs(r20) > 0.003)
            ).astype(np.float32)
            return trending

        regime_train = compute_regime(ret5, ret20, ret60)
        regime_val = compute_regime(ret5_val, ret20_val, ret60_val)

        # Trend strength = sign-consistent return magnitude
        strength_train = np.where(
            np.sign(ret5) == np.sign(ret20),
            np.abs(ret20),
            -np.abs(ret5)  # Negative when direction disagrees
        ).astype(np.float32)
        strength_val = np.where(
            np.sign(ret5_val) == np.sign(ret20_val),
            np.abs(ret20_val),
            -np.abs(ret5_val)
        ).astype(np.float32)

        metrics = {}
        print(f"\n  Momentum features: {len(self.mom_feature_names)}")
        print(f"  Trending rate: {regime_train.mean():.1%}")

        # Regime classifier
        print("  Training momentum regime classifier...")
        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(mom_X_train, label=regime_train,
                            feature_name=self.mom_feature_names)
        dval = lgb.Dataset(mom_X_val, label=regime_val,
                          feature_name=self.mom_feature_names, reference=dtrain)

        self.regime_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'is_unbalance': True},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        regime_pred = self.regime_model.predict(mom_X_val)
        try:
            metrics['regime_auc'] = float(roc_auc_score(regime_val, regime_pred))
        except ValueError:
            metrics['regime_auc'] = 0.5
        print(f"    Regime AUC: {metrics['regime_auc']:.3f}")

        # Trend strength regressor
        print("  Training trend strength regressor...")
        dtrain2 = lgb.Dataset(mom_X_train, label=strength_train,
                             feature_name=self.mom_feature_names)
        dval2 = lgb.Dataset(mom_X_val, label=strength_val,
                           feature_name=self.mom_feature_names, reference=dtrain2)

        self.strength_model = lgb.train(
            {'objective': 'regression', 'metric': 'mae', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1},
            dtrain2, num_boost_round=500, valid_sets=[dval2], callbacks=callbacks)

        str_pred = self.strength_model.predict(mom_X_val)
        metrics['strength_mae'] = float(np.mean(np.abs(str_pred - strength_val)))
        valid_mask = np.isfinite(str_pred) & np.isfinite(strength_val)
        if valid_mask.sum() > 10:
            metrics['strength_corr'] = float(np.corrcoef(str_pred[valid_mask], strength_val[valid_mask])[0, 1])
        else:
            metrics['strength_corr'] = 0.0
        metrics['strength_dir_acc'] = float(np.mean(
            (str_pred > 0) == (strength_val > 0)
        ))
        print(f"    Strength MAE: {metrics['strength_mae']:.4f}, Corr: {metrics['strength_corr']:.3f}")
        print(f"    Strength Dir Acc: {metrics['strength_dir_acc']:.1%}")

        # Feature importance
        imp = self.regime_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 momentum features (regime):")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.mom_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.regime_model is None:
            return {}
        mom_X, _ = self.derive_momentum_features(X, self.feature_names)
        return {
            'trending_prob': self.regime_model.predict(mom_X),
            'trend_strength': self.strength_model.predict(mom_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'regime_model': self.regime_model, 'strength_model': self.strength_model,
                'feature_names': self.feature_names,
                'mom_feature_names': self.mom_feature_names,
            }, f)
        print(f"  Saved MultiScaleMomentumModel to {path}")

    @classmethod
    def load(cls, path: str) -> 'MultiScaleMomentumModel':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.regime_model = data['regime_model']
        model.strength_model = data['strength_model']
        model.feature_names = data['feature_names']
        model.mom_feature_names = data['mom_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 24: Return Asymmetry Predictor
# ---------------------------------------------------------------------------

class ReturnAsymmetryPredictor:
    """
    Predicts whether upcoming price action will be asymmetric.

    Asymmetric = sudden spike (gap, squeeze breakout, news)
    Symmetric = slow grind (trending channel, mean reversion)

    Why this matters for trading:
    - Spike expected → widen stops (avoid stop-out before move), use larger TP
    - Grind expected → tighter trail, smaller TP targets
    - Negative spike expected → skip or reduce position

    Output: spike_prob, expected_skewness, vol_expansion_prob
    """

    def __init__(self):
        self.spike_model = None
        self.skew_model = None
        self.feature_names = None
        self.asym_feature_names = None

    @staticmethod
    def derive_asymmetry_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """Derive features for return asymmetry prediction."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Squeeze indicators (compression → breakout)
        for tf in ['5min', '1h', '4h', 'daily']:
            sq_idx = name_to_idx.get(f'{tf}_squeeze_score')
            if sq_idx is not None:
                feats.append(X[:, sq_idx])
                names.append(f'asym_{tf}_squeeze')
        sq_any_idx = name_to_idx.get('squeeze_any')
        if sq_any_idx is not None:
            feats.append(X[:, sq_any_idx])
            names.append('asym_squeeze_any')

        # Width dynamics (narrowing → explosion)
        wd_idx = name_to_idx.get('width_delta_3bar')
        if wd_idx is not None:
            feats.append(X[:, wd_idx])
            names.append('asym_width_delta')

        # Per-TF width (narrow channels break more explosively)
        for tf in ['5min', '1h', '4h']:
            w_idx = name_to_idx.get(f'{tf}_width_pct')
            if w_idx is not None:
                feats.append(X[:, w_idx])
                names.append(f'asym_{tf}_width')

        # Break probability (high break prob → imminent spike)
        for key in ['break_prob_max', 'break_prob_weighted', 'break_prob_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'asym_{key}')

        # Per-TF break probabilities
        for tf in ['5min', '1h', '4h']:
            for key in ['break_prob', 'break_prob_up', 'break_prob_down']:
                idx = name_to_idx.get(f'{tf}_{key}')
                if idx is not None:
                    feats.append(X[:, idx])
                    names.append(f'asym_{tf}_{key}')

        # Volatility state
        atr_idx = name_to_idx.get('atr_pct')
        if atr_idx is not None:
            feats.append(X[:, atr_idx])
            names.append('asym_atr')

        # Volume dynamics (volume precedes price)
        for key in ['volume_ratio_20', 'volume_trend_5', 'vol_momentum_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'asym_{key}')

        # Bar characteristics
        for key in ['bar_range_pct', 'close_position_in_bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'asym_{key}')

        # Entropy (high entropy = unpredictable = more spike-prone)
        for key in ['avg_entropy', 'entropy_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'asym_{key}')

        # Energy state (high energy = volatile)
        for tf in ['5min', '1h']:
            for key in ['total_energy', 'kinetic_energy', 'potential_energy']:
                idx = name_to_idx.get(f'{tf}_{key}')
                if idx is not None:
                    feats.append(X[:, idx])
                    names.append(f'asym_{tf}_{key}')

        # VIX context
        for key in ['vix_level', 'vix_change_5d']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'asym_{key}')

        # Momentum magnitude (strong momentum → continuation spike)
        for key in ['price_momentum_3bar', 'price_momentum_12bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(np.abs(X[:, idx]))
                names.append(f'asym_abs_{key}')

        # Position at channel boundaries (at edge → breakout spike likely)
        for tf in ['5min', '1h', '4h']:
            pos_idx = name_to_idx.get(f'{tf}_position_pct')
            if pos_idx is not None:
                pos = X[:, pos_idx]
                feats.append(np.minimum(pos, 1.0 - pos))
                names.append(f'asym_{tf}_boundary_dist')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train return asymmetry model."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        asym_X_train, self.asym_feature_names = self.derive_asymmetry_features(
            X_train, feature_names)
        asym_X_val, _ = self.derive_asymmetry_features(X_val, feature_names)

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']

        # Spike target: |ret5| > 2 * median |ret5| → sudden large move
        median_ret5 = np.median(np.abs(ret5))
        spike_train = (np.abs(ret5) > 2.5 * median_ret5).astype(np.float32)
        spike_val = (np.abs(ret5_val) > 2.5 * median_ret5).astype(np.float32)

        # Skewness target: sign(ret5) * (|ret5| / |ret20|)
        # High positive = sharp up spike; high negative = sharp down spike
        safe_r20 = np.where(np.abs(ret20) > 1e-6, ret20, 1e-6)
        safe_r20_val = np.where(np.abs(ret20_val) > 1e-6, ret20_val, 1e-6)
        skew_train = np.clip(ret5 / np.abs(safe_r20), -5, 5).astype(np.float32)
        skew_val = np.clip(ret5_val / np.abs(safe_r20_val), -5, 5).astype(np.float32)

        metrics = {}
        print(f"\n  Asymmetry features: {len(self.asym_feature_names)}")
        print(f"  Spike rate (|ret5| > 2.5x median): {spike_train.mean():.1%}")
        print(f"  Skew mean: {skew_train.mean():.2f}, std: {skew_train.std():.2f}")

        # Spike classifier
        print("  Training spike classifier...")
        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(asym_X_train, label=spike_train,
                            feature_name=self.asym_feature_names)
        dval = lgb.Dataset(asym_X_val, label=spike_val,
                          feature_name=self.asym_feature_names, reference=dtrain)

        self.spike_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'is_unbalance': True},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        spike_pred = self.spike_model.predict(asym_X_val)
        try:
            metrics['spike_auc'] = float(roc_auc_score(spike_val, spike_pred))
        except ValueError:
            metrics['spike_auc'] = 0.5
        print(f"    Spike AUC: {metrics['spike_auc']:.3f}")

        # Skewness regressor
        print("  Training skewness regressor...")
        dtrain2 = lgb.Dataset(asym_X_train, label=skew_train,
                             feature_name=self.asym_feature_names)
        dval2 = lgb.Dataset(asym_X_val, label=skew_val,
                           feature_name=self.asym_feature_names, reference=dtrain2)

        self.skew_model = lgb.train(
            {'objective': 'regression', 'metric': 'mae', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1},
            dtrain2, num_boost_round=500, valid_sets=[dval2], callbacks=callbacks)

        skew_pred = self.skew_model.predict(asym_X_val)
        metrics['skew_mae'] = float(np.mean(np.abs(skew_pred - skew_val)))
        valid_mask = np.isfinite(skew_pred) & np.isfinite(skew_val)
        if valid_mask.sum() > 10:
            metrics['skew_corr'] = float(np.corrcoef(skew_pred[valid_mask], skew_val[valid_mask])[0, 1])
        else:
            metrics['skew_corr'] = 0.0
        print(f"    Skew MAE: {metrics['skew_mae']:.3f}, Corr: {metrics['skew_corr']:.3f}")

        # Feature importance
        imp = self.spike_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 asymmetry features (spike):")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.asym_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.spike_model is None:
            return {}
        asym_X, _ = self.derive_asymmetry_features(X, self.feature_names)
        return {
            'spike_prob': self.spike_model.predict(asym_X),
            'expected_skewness': self.skew_model.predict(asym_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'spike_model': self.spike_model, 'skew_model': self.skew_model,
                'feature_names': self.feature_names,
                'asym_feature_names': self.asym_feature_names,
            }, f)
        print(f"  Saved ReturnAsymmetryPredictor to {path}")

    @classmethod
    def load(cls, path: str) -> 'ReturnAsymmetryPredictor':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.spike_model = data['spike_model']
        model.skew_model = data['skew_model']
        model.feature_names = data['feature_names']
        model.asym_feature_names = data['asym_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 25: Gap/Overnight Risk Predictor
# ---------------------------------------------------------------------------

class GapRiskPredictor:
    """
    Predicts overnight gap risk for positions held past close.

    TSLA regularly gaps 2-5% at open. This model predicts:
    1. Whether a large gap (>1%) is likely tonight
    2. Expected gap direction (up/down)

    Uses: close position, day-end momentum, VIX level, volume patterns,
    day of week (gaps more common Mon open after weekend news).

    Output: gap_risk_prob (>1% gap), gap_direction_prob (P(up gap))
    """

    def __init__(self):
        self.gap_model = None
        self.dir_model = None
        self.feature_names = None
        self.gap_feature_names = None

    @staticmethod
    def derive_gap_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """Derive features for gap risk prediction."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Time features (gaps matter most near close)
        for key in ['minutes_since_open', 'hour_sin', 'hour_cos', 'day_of_week']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'gap_{key}')

        # Session progress (near close = higher gap relevance)
        mso_idx = name_to_idx.get('minutes_since_open')
        if mso_idx is not None:
            mso = X[:, mso_idx]
            feats.append(np.clip(mso / 390.0, 0.0, 1.0))
            names.append('gap_session_progress')
            feats.append((mso > 300).astype(np.float32))
            names.append('gap_last_hour')

        # Day of week interactions (Friday close → Monday gap)
        dow_idx = name_to_idx.get('day_of_week')
        if dow_idx is not None:
            feats.append((X[:, dow_idx] > 3.5).astype(np.float32))
            names.append('gap_is_friday')

        # Momentum into close (strong momentum → continuation gap)
        for key in ['price_momentum_3bar', 'price_momentum_12bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'gap_{key}')
                feats.append(np.abs(X[:, idx]))
                names.append(f'gap_abs_{key}')

        # Volatility (high vol → bigger gaps)
        for key in ['atr_pct', 'bar_range_pct']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'gap_{key}')

        # VIX (elevated VIX = more gap risk)
        for key in ['vix_level', 'vix_change_5d']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'gap_{key}')

        # Volume patterns (declining volume into close = uncertainty)
        for key in ['volume_ratio_20', 'volume_trend_5', 'vol_momentum_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'gap_{key}')

        # RSI extremes (overbought/oversold → gap risk)
        for key in ['rsi_14', 'rsi_5']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'gap_{key}')
                feats.append(np.abs(X[:, idx] - 50))
                names.append(f'gap_{key}_extremity')

        # Break probability (high break prob + close = gap on open)
        for key in ['break_prob_max', 'break_prob_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'gap_{key}')

        # Channel health
        for key in ['health_min', 'health_max']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'gap_{key}')

        # SPY correlation (high correlation + SPY momentum = correlated gap)
        for key in ['spy_return_5bar', 'spy_tsla_corr_20']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'gap_{key}')

        # Close position in bar (hammer/doji patterns)
        cp_idx = name_to_idx.get('close_position_in_bar')
        if cp_idx is not None:
            feats.append(X[:, cp_idx])
            names.append('gap_close_pos_in_bar')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train gap risk models."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        gap_X_train, self.gap_feature_names = self.derive_gap_features(X_train, feature_names)
        gap_X_val, _ = self.derive_gap_features(X_val, feature_names)

        ret5 = Y_train['future_return_5']
        ret5_val = Y_val['future_return_5']

        # Gap target: |future_return_5| > 1% (proxy for overnight gap)
        # In 5-min bars, 5 bars = 25 minutes, so big moves within that timeframe
        # approximate gap-like events
        gap_train = (np.abs(ret5) > 0.01).astype(np.float32)
        gap_val = (np.abs(ret5_val) > 0.01).astype(np.float32)

        # Direction: positive return
        dir_train = (ret5 > 0).astype(np.float32)
        dir_val = (ret5_val > 0).astype(np.float32)

        metrics = {}
        print(f"\n  Gap features: {len(self.gap_feature_names)}")
        print(f"  Gap rate (|ret5|>1%): {gap_train.mean():.1%}")

        # Gap risk classifier
        print("  Training gap risk classifier...")
        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(gap_X_train, label=gap_train,
                            feature_name=self.gap_feature_names)
        dval = lgb.Dataset(gap_X_val, label=gap_val,
                          feature_name=self.gap_feature_names, reference=dtrain)

        self.gap_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'is_unbalance': True},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        gap_pred = self.gap_model.predict(gap_X_val)
        try:
            metrics['gap_auc'] = float(roc_auc_score(gap_val, gap_pred))
        except ValueError:
            metrics['gap_auc'] = 0.5
        print(f"    Gap AUC: {metrics['gap_auc']:.3f}")

        # Direction classifier (only useful when gap is expected)
        print("  Training gap direction classifier...")
        dtrain2 = lgb.Dataset(gap_X_train, label=dir_train,
                             feature_name=self.gap_feature_names)
        dval2 = lgb.Dataset(gap_X_val, label=dir_val,
                           feature_name=self.gap_feature_names, reference=dtrain2)

        self.dir_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1},
            dtrain2, num_boost_round=500, valid_sets=[dval2], callbacks=callbacks)

        dir_pred = self.dir_model.predict(gap_X_val)
        try:
            metrics['dir_auc'] = float(roc_auc_score(dir_val, dir_pred))
        except ValueError:
            metrics['dir_auc'] = 0.5
        print(f"    Direction AUC: {metrics['dir_auc']:.3f}")

        # Feature importance
        imp = self.gap_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 gap features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.gap_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.gap_model is None:
            return {}
        gap_X, _ = self.derive_gap_features(X, self.feature_names)
        return {
            'gap_risk_prob': self.gap_model.predict(gap_X),
            'gap_up_prob': self.dir_model.predict(gap_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'gap_model': self.gap_model, 'dir_model': self.dir_model,
                'feature_names': self.feature_names,
                'gap_feature_names': self.gap_feature_names,
            }, f)
        print(f"  Saved GapRiskPredictor to {path}")

    @classmethod
    def load(cls, path: str) -> 'GapRiskPredictor':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.gap_model = data['gap_model']
        model.dir_model = data['dir_model']
        model.feature_names = data['feature_names']
        model.gap_feature_names = data['gap_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 26: Mean Reversion Speed Estimator
# ---------------------------------------------------------------------------

class MeanReversionSpeedModel:
    """
    Predicts how fast price will revert to channel center.

    Fast reversion → bounce trades are excellent, use tighter trail
    Slow reversion → wider stops needed, break trades preferred
    No reversion → channel is breaking down

    Uses OU parameters (theta = reversion strength), position in channel,
    momentum, and volume to estimate reversion speed.

    Output: reversion_speed (fast/slow/none), expected_bars_to_center
    """

    def __init__(self):
        self.speed_model = None
        self.bars_model = None
        self.feature_names = None
        self.rev_feature_names = None

    @staticmethod
    def derive_reversion_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """Derive features for mean reversion speed prediction."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # OU parameters (direct reversion indicators)
        for tf in ['5min', '1h', '4h', 'daily']:
            for key in ['ou_theta', 'ou_half_life', 'ou_reversion_score']:
                idx = name_to_idx.get(f'{tf}_{key}')
                if idx is not None:
                    feats.append(X[:, idx])
                    names.append(f'rev_{tf}_{key}')

        # Position in channel (further from center = more reversion potential)
        for tf in ['5min', '1h', '4h', 'daily']:
            pos_idx = name_to_idx.get(f'{tf}_position_pct')
            cd_idx = name_to_idx.get(f'{tf}_center_distance')
            if pos_idx is not None:
                pos = X[:, pos_idx]
                feats.append(pos)
                names.append(f'rev_{tf}_position')
                # Distance from center (0.5)
                feats.append(np.abs(pos - 0.5))
                names.append(f'rev_{tf}_center_dist')
            if cd_idx is not None:
                feats.append(X[:, cd_idx])
                names.append(f'rev_{tf}_raw_center_dist')

        # Momentum (against-trend momentum = faster reversion)
        for key in ['price_momentum_3bar', 'price_momentum_12bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'rev_{key}')

        # Channel health (healthy channels have stronger reversion)
        for key in ['health_min', 'health_max', 'health_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'rev_{key}')

        # Bounce count (more bounces = proven reversion behavior)
        for tf in ['5min', '1h', '4h']:
            bc_idx = name_to_idx.get(f'{tf}_bounce_count')
            if bc_idx is not None:
                feats.append(X[:, bc_idx])
                names.append(f'rev_{tf}_bounce_count')

        # Width (narrow channels revert faster)
        for tf in ['5min', '1h', '4h']:
            w_idx = name_to_idx.get(f'{tf}_width_pct')
            if w_idx is not None:
                feats.append(X[:, w_idx])
                names.append(f'rev_{tf}_width')

        # Volume (high volume = faster price discovery/reversion)
        for key in ['volume_ratio_20', 'vol_momentum_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'rev_{key}')

        # Cross-TF consensus
        for key in ['direction_consensus', 'confluence_score', 'theta_spread']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'rev_{key}')

        # Oscillation period (short period = fast reversion cycle)
        for tf in ['5min', '1h']:
            osc_idx = name_to_idx.get(f'{tf}_oscillation_period')
            if osc_idx is not None:
                feats.append(X[:, osc_idx])
                names.append(f'rev_{tf}_osc_period')

        # R-squared (high r² = more mean-reverting)
        for tf in ['5min', '1h', '4h']:
            r2_idx = name_to_idx.get(f'{tf}_r_squared')
            if r2_idx is not None:
                feats.append(X[:, r2_idx])
                names.append(f'rev_{tf}_r_squared')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train mean reversion speed model."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        rev_X_train, self.rev_feature_names = self.derive_reversion_features(
            X_train, feature_names)
        rev_X_val, _ = self.derive_reversion_features(X_val, feature_names)

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']

        # Speed target: "fast reversion" = ret5 and ret20 have opposite signs
        # (price moved away then came back) OR ret5 is small while ret20 is bigger
        # in same direction (steady trend)
        # For mean reversion: |ret5| is small while position was extreme
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        pos_5m_idx = name_to_idx.get('5min_position_pct')

        if pos_5m_idx is not None:
            pos_train = X_train[:, pos_5m_idx]
            pos_val = X_val[:, pos_5m_idx]
            at_edge_train = (np.abs(pos_train - 0.5) > 0.3)
            at_edge_val = (np.abs(pos_val - 0.5) > 0.3)
            # Fast reversion = at edge AND small future return (came back)
            fast_rev_train = (at_edge_train & (np.abs(ret5) < 0.003)).astype(np.float32)
            fast_rev_val = (at_edge_val & (np.abs(ret5_val) < 0.003)).astype(np.float32)
        else:
            fast_rev_train = (np.abs(ret5) < 0.003).astype(np.float32)
            fast_rev_val = (np.abs(ret5_val) < 0.003).astype(np.float32)

        # Bars to center proxy: use |ret20|/|ret5| ratio (higher = price keeps moving)
        bars_target_train = np.clip(np.abs(ret20) * 100, 0, 50).astype(np.float32)  # Scale to bars
        bars_target_val = np.clip(np.abs(ret20_val) * 100, 0, 50).astype(np.float32)

        metrics = {}
        print(f"\n  Reversion features: {len(self.rev_feature_names)}")
        print(f"  Fast reversion rate: {fast_rev_train.mean():.1%}")

        # Speed classifier
        print("  Training reversion speed classifier...")
        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(rev_X_train, label=fast_rev_train,
                            feature_name=self.rev_feature_names)
        dval = lgb.Dataset(rev_X_val, label=fast_rev_val,
                          feature_name=self.rev_feature_names, reference=dtrain)

        self.speed_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'is_unbalance': True},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        speed_pred = self.speed_model.predict(rev_X_val)
        try:
            metrics['speed_auc'] = float(roc_auc_score(fast_rev_val, speed_pred))
        except ValueError:
            metrics['speed_auc'] = 0.5
        print(f"    Speed AUC: {metrics['speed_auc']:.3f}")

        # Bars to center regressor
        print("  Training bars-to-center regressor...")
        dtrain2 = lgb.Dataset(rev_X_train, label=bars_target_train,
                             feature_name=self.rev_feature_names)
        dval2 = lgb.Dataset(rev_X_val, label=bars_target_val,
                           feature_name=self.rev_feature_names, reference=dtrain2)

        self.bars_model = lgb.train(
            {'objective': 'regression', 'metric': 'mae', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1},
            dtrain2, num_boost_round=500, valid_sets=[dval2], callbacks=callbacks)

        bars_pred = self.bars_model.predict(rev_X_val)
        metrics['bars_mae'] = float(np.mean(np.abs(bars_pred - bars_target_val)))
        valid_mask = np.isfinite(bars_pred) & np.isfinite(bars_target_val)
        if valid_mask.sum() > 10:
            metrics['bars_corr'] = float(np.corrcoef(bars_pred[valid_mask], bars_target_val[valid_mask])[0, 1])
        else:
            metrics['bars_corr'] = 0.0
        print(f"    Bars MAE: {metrics['bars_mae']:.1f}, Corr: {metrics['bars_corr']:.3f}")

        # Feature importance
        imp = self.speed_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 reversion features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.rev_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.speed_model is None:
            return {}
        rev_X, _ = self.derive_reversion_features(X, self.feature_names)
        return {
            'fast_reversion_prob': self.speed_model.predict(rev_X),
            'expected_bars_to_center': np.clip(self.bars_model.predict(rev_X), 1, 100),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'speed_model': self.speed_model, 'bars_model': self.bars_model,
                'feature_names': self.feature_names,
                'rev_feature_names': self.rev_feature_names,
            }, f)
        print(f"  Saved MeanReversionSpeedModel to {path}")

    @classmethod
    def load(cls, path: str) -> 'MeanReversionSpeedModel':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.speed_model = data['speed_model']
        model.bars_model = data['bars_model']
        model.feature_names = data['feature_names']
        model.rev_feature_names = data['rev_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 27: Liquidity State Classifier
# ---------------------------------------------------------------------------

class LiquidityStateClassifier:
    """
    Classifies current market liquidity state.

    Liquidity matters for:
    - Fill quality (thin markets → slippage → worse fills)
    - Move reliability (liquid markets → cleaner trends)
    - Risk (drying liquidity → sudden moves likely)

    Three states:
    - Liquid (0): Good volume, tight ranges, reliable fills
    - Thin (1): Below-average volume, wider ranges, slippage risk
    - Drying (2): Volume declining sharply, about to move

    Output: liquidity_state, slippage_risk
    """

    def __init__(self):
        self.state_model = None
        self.slippage_model = None
        self.feature_names = None
        self.liq_feature_names = None

    @staticmethod
    def derive_liquidity_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """Derive features for liquidity classification."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Core volume features
        for key in ['volume_ratio_20', 'volume_trend_5', 'vol_momentum_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'liq_{key}')

        # Volume score per TF
        for tf in ['5min', '1h', '4h']:
            vs_idx = name_to_idx.get(f'{tf}_volume_score')
            if vs_idx is not None:
                feats.append(X[:, vs_idx])
                names.append(f'liq_{tf}_volume_score')

        # Bar characteristics (wide bars on low volume = thin market)
        for key in ['bar_range_pct', 'close_position_in_bar', 'atr_pct']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'liq_{key}')

        # Volume-range interaction (high range / low volume = thin)
        vol_idx = name_to_idx.get('volume_ratio_20')
        range_idx = name_to_idx.get('bar_range_pct')
        if vol_idx is not None and range_idx is not None:
            safe_vol = np.where(X[:, vol_idx] > 0.01, X[:, vol_idx], 0.01)
            feats.append(X[:, range_idx] / safe_vol)
            names.append('liq_range_per_volume')

        # Time of day (liquidity varies by session)
        for key in ['minutes_since_open', 'hour_sin', 'hour_cos']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'liq_{key}')

        # Squeeze indicators (squeeze = liquidity drying up)
        for tf in ['5min', '1h', '4h']:
            sq_idx = name_to_idx.get(f'{tf}_squeeze_score')
            if sq_idx is not None:
                feats.append(X[:, sq_idx])
                names.append(f'liq_{tf}_squeeze')
        sq_any_idx = name_to_idx.get('squeeze_any')
        if sq_any_idx is not None:
            feats.append(X[:, sq_any_idx])
            names.append('liq_squeeze_any')

        # Width (narrow = potentially illiquid)
        for tf in ['5min', '1h']:
            w_idx = name_to_idx.get(f'{tf}_width_pct')
            if w_idx is not None:
                feats.append(X[:, w_idx])
                names.append(f'liq_{tf}_width')

        # VIX (high VIX = wider spreads = worse liquidity)
        vix_idx = name_to_idx.get('vix_level')
        if vix_idx is not None:
            feats.append(X[:, vix_idx])
            names.append('liq_vix')

        # Consecutive same-direction bars (one-sided = thinning)
        for key in ['consecutive_up_bars', 'consecutive_down_bars']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'liq_{key}')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train liquidity state model."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        liq_X_train, self.liq_feature_names = self.derive_liquidity_features(
            X_train, feature_names)
        liq_X_val, _ = self.derive_liquidity_features(X_val, feature_names)

        ret5 = Y_train['future_return_5']
        ret5_val = Y_val['future_return_5']

        # Slippage proxy: large bar ranges + small returns = choppy/illiquid
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        atr_idx = name_to_idx.get('atr_pct')
        vol_idx = name_to_idx.get('volume_ratio_20')

        if vol_idx is not None:
            # "Thin" market = below-average volume
            thin_train = (X_train[:, vol_idx] < 0.7).astype(np.float32)
            thin_val = (X_val[:, vol_idx] < 0.7).astype(np.float32)
        else:
            thin_train = np.zeros(len(X_train), dtype=np.float32)
            thin_val = np.zeros(len(X_val), dtype=np.float32)

        # Slippage risk: high |ret5| relative to ATR (unexpected big move)
        if atr_idx is not None:
            atr_train = np.where(X_train[:, atr_idx] > 0.0001, X_train[:, atr_idx], 0.0001)
            atr_val = np.where(X_val[:, atr_idx] > 0.0001, X_val[:, atr_idx], 0.0001)
            slip_train = np.clip(np.abs(ret5) / atr_train, 0, 5).astype(np.float32)
            slip_val = np.clip(np.abs(ret5_val) / atr_val, 0, 5).astype(np.float32)
        else:
            slip_train = np.abs(ret5).astype(np.float32)
            slip_val = np.abs(ret5_val).astype(np.float32)

        metrics = {}
        print(f"\n  Liquidity features: {len(self.liq_feature_names)}")
        print(f"  Thin market rate: {thin_train.mean():.1%}")

        # Thin market classifier
        print("  Training thin market classifier...")
        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(liq_X_train, label=thin_train,
                            feature_name=self.liq_feature_names)
        dval = lgb.Dataset(liq_X_val, label=thin_val,
                          feature_name=self.liq_feature_names, reference=dtrain)

        self.state_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'is_unbalance': True},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        state_pred = self.state_model.predict(liq_X_val)
        try:
            metrics['thin_auc'] = float(roc_auc_score(thin_val, state_pred))
        except ValueError:
            metrics['thin_auc'] = 0.5
        print(f"    Thin Market AUC: {metrics['thin_auc']:.3f}")

        # Slippage regressor
        print("  Training slippage risk regressor...")
        dtrain2 = lgb.Dataset(liq_X_train, label=slip_train,
                             feature_name=self.liq_feature_names)
        dval2 = lgb.Dataset(liq_X_val, label=slip_val,
                           feature_name=self.liq_feature_names, reference=dtrain2)

        self.slippage_model = lgb.train(
            {'objective': 'regression', 'metric': 'mae', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1},
            dtrain2, num_boost_round=500, valid_sets=[dval2], callbacks=callbacks)

        slip_pred = self.slippage_model.predict(liq_X_val)
        metrics['slippage_mae'] = float(np.mean(np.abs(slip_pred - slip_val)))
        valid_mask = np.isfinite(slip_pred) & np.isfinite(slip_val)
        if valid_mask.sum() > 10:
            metrics['slippage_corr'] = float(np.corrcoef(slip_pred[valid_mask], slip_val[valid_mask])[0, 1])
        else:
            metrics['slippage_corr'] = 0.0
        print(f"    Slippage MAE: {metrics['slippage_mae']:.3f}, Corr: {metrics['slippage_corr']:.3f}")

        # Feature importance
        imp = self.state_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 liquidity features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.liq_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.state_model is None:
            return {}
        liq_X, _ = self.derive_liquidity_features(X, self.feature_names)
        return {
            'thin_market_prob': self.state_model.predict(liq_X),
            'slippage_risk': np.clip(self.slippage_model.predict(liq_X), 0, 5),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'state_model': self.state_model, 'slippage_model': self.slippage_model,
                'feature_names': self.feature_names,
                'liq_feature_names': self.liq_feature_names,
            }, f)
        print(f"  Saved LiquidityStateClassifier to {path}")

    @classmethod
    def load(cls, path: str) -> 'LiquidityStateClassifier':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.state_model = data['state_model']
        model.slippage_model = data['slippage_model']
        model.feature_names = data['feature_names']
        model.liq_feature_names = data['liq_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 28: Regime Transition Detector
# ---------------------------------------------------------------------------

class RegimeTransitionDetector:
    """
    Detects when the overall market regime is transitioning.

    Regime transitions are dangerous for all strategies:
    - Trending → Ranging: trend-following signals fail
    - Calm → Volatile: stops get blown out
    - Bull → Bear: direction signals flip

    Uses rolling deltas of features over multiple lookbacks to detect
    when the feature landscape is shifting.

    Output: transition_prob, regime_stability
    """

    def __init__(self):
        self.transition_model = None
        self.stability_model = None
        self.feature_names = None
        self.trans_feature_names = None

    @staticmethod
    def derive_transition_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """Derive features for regime transition detection."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Feature deltas (rate of change of key indicators)
        for key in ['health_delta_3bar', 'health_delta_6bar',
                     'entropy_delta_3bar', 'break_prob_delta_3bar',
                     'energy_delta_3bar', 'rsi_slope_5bar',
                     'width_delta_3bar', 'pos_delta_3bar', 'pos_delta_6bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'trans_{key}')
                feats.append(np.abs(X[:, idx]))
                names.append(f'trans_abs_{key}')

        # Health acceleration
        hd3_idx = name_to_idx.get('health_delta_3bar')
        hd6_idx = name_to_idx.get('health_delta_6bar')
        if hd3_idx is not None and hd6_idx is not None:
            feats.append(X[:, hd3_idx] - X[:, hd6_idx] / 2.0)
            names.append('trans_health_accel')

        # VIX dynamics (rising VIX = regime shifting)
        for key in ['vix_level', 'vix_change_5d']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'trans_{key}')

        # VIX acceleration proxy
        vix_idx = name_to_idx.get('vix_change_5d')
        if vix_idx is not None:
            feats.append(np.abs(X[:, vix_idx]))
            names.append('trans_abs_vix_change')

        # Cross-TF alignment changes (disagreement = transition)
        for key in ['direction_consensus', 'confluence_score', 'theta_spread',
                     'health_spread', 'bullish_fraction', 'bearish_fraction']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'trans_{key}')

        # SPY dynamics
        for key in ['spy_return_5bar', 'spy_return_20bar', 'spy_tsla_corr_20']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'trans_{key}')

        # SPY momentum divergence
        spy5_idx = name_to_idx.get('spy_return_5bar')
        spy20_idx = name_to_idx.get('spy_return_20bar')
        if spy5_idx is not None and spy20_idx is not None:
            feats.append(X[:, spy5_idx] - X[:, spy20_idx])
            names.append('trans_spy_momentum_divergence')

        # Volatility state
        atr_idx = name_to_idx.get('atr_pct')
        if atr_idx is not None:
            feats.append(X[:, atr_idx])
            names.append('trans_atr')

        # Momentum divergence
        pm3_idx = name_to_idx.get('price_momentum_3bar')
        pm12_idx = name_to_idx.get('price_momentum_12bar')
        if pm3_idx is not None and pm12_idx is not None:
            feats.append(X[:, pm3_idx] - X[:, pm12_idx])
            names.append('trans_momentum_divergence')
            feats.append(np.sign(X[:, pm3_idx]) * np.sign(X[:, pm12_idx]))
            names.append('trans_momentum_agreement')

        # Break probability dynamics
        for key in ['break_prob_max', 'break_prob_weighted']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'trans_{key}')

        # Entropy (rising entropy across TFs = instability)
        for key in ['avg_entropy']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'trans_{key}')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train regime transition model."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        trans_X_train, self.trans_feature_names = self.derive_transition_features(
            X_train, feature_names)
        trans_X_val, _ = self.derive_transition_features(X_val, feature_names)

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret60 = Y_train['future_return_60']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']
        ret60_val = Y_val['future_return_60']

        # Transition target: regime instability
        # When returns are unpredictable (sign changes between timeframes)
        transition_train = (
            (np.sign(ret5) != np.sign(ret20)) &
            (np.abs(ret20) > 0.003)
        ).astype(np.float32)
        transition_val = (
            (np.sign(ret5_val) != np.sign(ret20_val)) &
            (np.abs(ret20_val) > 0.003)
        ).astype(np.float32)

        # Stability target: how consistent are multi-horizon returns
        # High stability = all returns same sign and proportional
        same_sign = (np.sign(ret5) == np.sign(ret20)).astype(np.float32)
        same_sign2 = (np.sign(ret20) == np.sign(ret60)).astype(np.float32)
        stability_train = ((same_sign + same_sign2) / 2.0).astype(np.float32)

        same_sign_v = (np.sign(ret5_val) == np.sign(ret20_val)).astype(np.float32)
        same_sign2_v = (np.sign(ret20_val) == np.sign(ret60_val)).astype(np.float32)
        stability_val = ((same_sign_v + same_sign2_v) / 2.0).astype(np.float32)

        metrics = {}
        print(f"\n  Transition features: {len(self.trans_feature_names)}")
        print(f"  Transition rate: {transition_train.mean():.1%}")
        print(f"  Mean stability: {stability_train.mean():.2f}")

        # Transition classifier
        print("  Training transition classifier...")
        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(trans_X_train, label=transition_train,
                            feature_name=self.trans_feature_names)
        dval = lgb.Dataset(trans_X_val, label=transition_val,
                          feature_name=self.trans_feature_names, reference=dtrain)

        self.transition_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'is_unbalance': True},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        trans_pred = self.transition_model.predict(trans_X_val)
        try:
            metrics['transition_auc'] = float(roc_auc_score(transition_val, trans_pred))
        except ValueError:
            metrics['transition_auc'] = 0.5
        print(f"    Transition AUC: {metrics['transition_auc']:.3f}")

        # Stability regressor
        print("  Training stability regressor...")
        dtrain2 = lgb.Dataset(trans_X_train, label=stability_train,
                             feature_name=self.trans_feature_names)
        dval2 = lgb.Dataset(trans_X_val, label=stability_val,
                           feature_name=self.trans_feature_names, reference=dtrain2)

        self.stability_model = lgb.train(
            {'objective': 'regression', 'metric': 'mae', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1},
            dtrain2, num_boost_round=500, valid_sets=[dval2], callbacks=callbacks)

        stab_pred = self.stability_model.predict(trans_X_val)
        metrics['stability_mae'] = float(np.mean(np.abs(stab_pred - stability_val)))
        valid_mask = np.isfinite(stab_pred) & np.isfinite(stability_val)
        if valid_mask.sum() > 10:
            metrics['stability_corr'] = float(np.corrcoef(stab_pred[valid_mask], stability_val[valid_mask])[0, 1])
        else:
            metrics['stability_corr'] = 0.0
        print(f"    Stability MAE: {metrics['stability_mae']:.3f}, Corr: {metrics['stability_corr']:.3f}")

        # Feature importance
        imp = self.transition_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 transition features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.trans_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.transition_model is None:
            return {}
        trans_X, _ = self.derive_transition_features(X, self.feature_names)
        return {
            'transition_prob': self.transition_model.predict(trans_X),
            'regime_stability': np.clip(self.stability_model.predict(trans_X), 0, 1),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'transition_model': self.transition_model,
                'stability_model': self.stability_model,
                'feature_names': self.feature_names,
                'trans_feature_names': self.trans_feature_names,
            }, f)
        print(f"  Saved RegimeTransitionDetector to {path}")

    @classmethod
    def load(cls, path: str) -> 'RegimeTransitionDetector':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.transition_model = data['transition_model']
        model.stability_model = data['stability_model']
        model.feature_names = data['feature_names']
        model.trans_feature_names = data['trans_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 29: Profit Target Optimizer
# ---------------------------------------------------------------------------

class ProfitTargetOptimizer:
    """
    Predicts optimal take-profit level for each trade setup.

    Fixed TP ratios leave money on the table (strong setups) or
    set unreachable targets (weak setups). This model predicts
    how far price will move in the trade direction.

    Output: tp_multiplier (scale factor for base TP), expected_move_pct
    """

    def __init__(self):
        self.move_model = None
        self.big_move_model = None
        self.feature_names = None
        self.tp_feature_names = None

    @staticmethod
    def derive_tp_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """Derive features for TP optimization."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Channel width (wider = larger TP potential)
        for tf in ['5min', '1h', '4h', 'daily']:
            w_idx = name_to_idx.get(f'{tf}_width_pct')
            if w_idx is not None:
                feats.append(X[:, w_idx])
                names.append(f'tp_{tf}_width')

        # Position in channel (near boundary = more room to TP at center)
        for tf in ['5min', '1h', '4h']:
            pos_idx = name_to_idx.get(f'{tf}_position_pct')
            if pos_idx is not None:
                pos = X[:, pos_idx]
                feats.append(pos)
                names.append(f'tp_{tf}_position')
                feats.append(np.abs(pos - 0.5))
                names.append(f'tp_{tf}_room_to_center')

        # Momentum (strong momentum = price will move further)
        for key in ['price_momentum_3bar', 'price_momentum_12bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'tp_{key}')
                feats.append(np.abs(X[:, idx]))
                names.append(f'tp_abs_{key}')

        # Volatility (high vol = bigger moves)
        atr_idx = name_to_idx.get('atr_pct')
        if atr_idx is not None:
            feats.append(X[:, atr_idx])
            names.append('tp_atr')

        # Volume (high volume = more follow-through)
        for key in ['volume_ratio_20', 'volume_trend_5']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'tp_{key}')

        # Channel health (healthy channels = more reliable bounces)
        for key in ['health_min', 'health_max']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'tp_{key}')

        # Break probability (high break prob = potential for big move)
        for key in ['break_prob_max', 'break_prob_weighted']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'tp_{key}')

        # Direction consensus (aligned TFs = stronger move)
        for key in ['direction_consensus', 'confluence_score',
                     'bullish_fraction', 'bearish_fraction']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'tp_{key}')

        # Squeeze indicators (squeezed = explosive move when breaks)
        sq_any_idx = name_to_idx.get('squeeze_any')
        if sq_any_idx is not None:
            feats.append(X[:, sq_any_idx])
            names.append('tp_squeeze_any')

        # RSI (extreme RSI = larger reversal potential)
        for key in ['rsi_14', 'rsi_5']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(np.abs(X[:, idx] - 50))
                names.append(f'tp_{key}_extremity')

        # Energy (high energy = more potential for movement)
        for tf in ['5min', '1h']:
            for key in ['total_energy', 'kinetic_energy']:
                idx = name_to_idx.get(f'{tf}_{key}')
                if idx is not None:
                    feats.append(X[:, idx])
                    names.append(f'tp_{tf}_{key}')

        # VIX
        vix_idx = name_to_idx.get('vix_level')
        if vix_idx is not None:
            feats.append(X[:, vix_idx])
            names.append('tp_vix')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train profit target optimizer."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        tp_X_train, self.tp_feature_names = self.derive_tp_features(X_train, feature_names)
        tp_X_val, _ = self.derive_tp_features(X_val, feature_names)

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret60 = Y_train['future_return_60']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']
        ret60_val = Y_val['future_return_60']

        # Move magnitude target: max absolute return across horizons
        move_train = np.maximum(np.abs(ret5), np.maximum(np.abs(ret20), np.abs(ret60)))
        move_val = np.maximum(np.abs(ret5_val), np.maximum(np.abs(ret20_val), np.abs(ret60_val)))

        # Big move target: will the move exceed 1.5%?
        big_train = (move_train > 0.015).astype(np.float32)
        big_val = (move_val > 0.015).astype(np.float32)

        metrics = {}
        print(f"\n  TP features: {len(self.tp_feature_names)}")
        print(f"  Mean move magnitude: {move_train.mean():.3f}")
        print(f"  Big move (>1.5%) rate: {big_train.mean():.1%}")

        # Move magnitude regressor
        print("  Training move magnitude regressor...")
        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(tp_X_train, label=move_train,
                            feature_name=self.tp_feature_names)
        dval = lgb.Dataset(tp_X_val, label=move_val,
                          feature_name=self.tp_feature_names, reference=dtrain)

        self.move_model = lgb.train(
            {'objective': 'regression', 'metric': 'mae', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        move_pred = self.move_model.predict(tp_X_val)
        metrics['move_mae'] = float(np.mean(np.abs(move_pred - move_val)))
        valid = np.isfinite(move_pred) & np.isfinite(move_val)
        metrics['move_corr'] = float(np.corrcoef(move_pred[valid], move_val[valid])[0, 1]) if valid.sum() > 10 else 0
        print(f"    Move MAE: {metrics['move_mae']:.4f}, Corr: {metrics['move_corr']:.3f}")

        # Big move classifier
        print("  Training big move classifier...")
        dtrain2 = lgb.Dataset(tp_X_train, label=big_train,
                             feature_name=self.tp_feature_names)
        dval2 = lgb.Dataset(tp_X_val, label=big_val,
                           feature_name=self.tp_feature_names, reference=dtrain2)

        self.big_move_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'is_unbalance': True},
            dtrain2, num_boost_round=500, valid_sets=[dval2], callbacks=callbacks)

        big_pred = self.big_move_model.predict(tp_X_val)
        try:
            metrics['big_move_auc'] = float(roc_auc_score(big_val, big_pred))
        except ValueError:
            metrics['big_move_auc'] = 0.5
        print(f"    Big Move AUC: {metrics['big_move_auc']:.3f}")

        imp = self.move_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 TP features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.tp_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.move_model is None:
            return {}
        tp_X, _ = self.derive_tp_features(X, self.feature_names)
        return {
            'expected_move': np.clip(self.move_model.predict(tp_X), 0, 0.1),
            'big_move_prob': self.big_move_model.predict(tp_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'move_model': self.move_model, 'big_move_model': self.big_move_model,
                'feature_names': self.feature_names,
                'tp_feature_names': self.tp_feature_names,
            }, f)
        print(f"  Saved ProfitTargetOptimizer to {path}")

    @classmethod
    def load(cls, path: str) -> 'ProfitTargetOptimizer':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.move_model = data['move_model']
        model.big_move_model = data['big_move_model']
        model.feature_names = data['feature_names']
        model.tp_feature_names = data['tp_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 30: Channel Alignment Scorer
# ---------------------------------------------------------------------------

class ChannelAlignmentScorer:
    """
    Scores how well multi-TF channels are aligned for a trade.

    Perfect alignment = all channels same direction, positions consistent,
    widths proportional → highest confidence, best trades.

    Misalignment = channels at different angles, conflicting positions →
    uncertain outcome, reduce confidence.

    Uses structured per-TF features as pairs to compute alignment.

    Output: alignment_score (0-1), alignment_regime (aligned/partial/conflicting)
    """

    def __init__(self):
        self.alignment_model = None
        self.win_model = None
        self.feature_names = None
        self.align_feature_names = None

    @staticmethod
    def derive_alignment_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """Derive channel alignment features."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        tfs = ['5min', '1h', '4h', 'daily', 'weekly']

        # Per-TF slope agreement
        slopes = []
        for tf in tfs:
            idx = name_to_idx.get(f'{tf}_slope_pct')
            if idx is not None:
                slopes.append(X[:, idx])
                feats.append(X[:, idx])
                names.append(f'align_{tf}_slope')

        if len(slopes) >= 2:
            slope_stack = np.column_stack(slopes)
            feats.append(np.std(slope_stack, axis=1))
            names.append('align_slope_std')
            feats.append(np.mean(slope_stack, axis=1))
            names.append('align_slope_mean')
            # Sign agreement
            signs = np.sign(slope_stack)
            feats.append(np.mean(signs, axis=1))
            names.append('align_slope_sign_consensus')

        # Per-TF position alignment
        positions = []
        for tf in tfs:
            idx = name_to_idx.get(f'{tf}_position_pct')
            if idx is not None:
                positions.append(X[:, idx])
                feats.append(X[:, idx])
                names.append(f'align_{tf}_position')

        if len(positions) >= 2:
            pos_stack = np.column_stack(positions)
            feats.append(np.std(pos_stack, axis=1))
            names.append('align_position_std')
            feats.append(np.mean(pos_stack, axis=1))
            names.append('align_position_mean')
            # All above/below center agreement
            above = (pos_stack > 0.5).astype(np.float32)
            feats.append(np.mean(above, axis=1))
            names.append('align_above_center_fraction')

        # Cross-TF position spreads (from existing features)
        for key in ['pos_spread_5m_1h', 'pos_spread_5m_daily', 'pos_spread_1h_daily']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'align_{key}')
                feats.append(np.abs(X[:, idx]))
                names.append(f'align_abs_{key}')

        # Per-TF momentum direction
        mom_dirs = []
        for tf in tfs:
            idx = name_to_idx.get(f'{tf}_momentum_direction')
            if idx is not None:
                mom_dirs.append(X[:, idx])
                feats.append(X[:, idx])
                names.append(f'align_{tf}_momentum_dir')

        if len(mom_dirs) >= 2:
            md_stack = np.column_stack(mom_dirs)
            feats.append(np.std(md_stack, axis=1))
            names.append('align_momentum_dir_std')

        # Existing cross-TF features
        for key in ['direction_consensus', 'confluence_score', 'bullish_fraction',
                     'bearish_fraction', 'health_spread', 'theta_spread',
                     'energy_ratio_5m_1h', 'energy_ratio_5m_daily']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'align_{key}')

        # Per-TF health (consistent health = aligned)
        healths = []
        for tf in tfs:
            idx = name_to_idx.get(f'{tf}_channel_health')
            if idx is not None:
                healths.append(X[:, idx])

        if len(healths) >= 2:
            h_stack = np.column_stack(healths)
            feats.append(np.std(h_stack, axis=1))
            names.append('align_health_std')
            feats.append(np.min(h_stack, axis=1))
            names.append('align_health_min')

        # Valid TF count
        idx = name_to_idx.get('valid_tf_count')
        if idx is not None:
            feats.append(X[:, idx])
            names.append('align_valid_tf_count')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train channel alignment model."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        align_X_train, self.align_feature_names = self.derive_alignment_features(
            X_train, feature_names)
        align_X_val, _ = self.derive_alignment_features(X_val, feature_names)

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']

        # Alignment target: when channels are aligned, moves are directional
        # Good alignment = |ret20| > 0.005 AND sign(ret5) == sign(ret20)
        aligned_train = (
            (np.abs(ret20) > 0.005) &
            (np.sign(ret5) == np.sign(ret20))
        ).astype(np.float32)
        aligned_val = (
            (np.abs(ret20_val) > 0.005) &
            (np.sign(ret5_val) == np.sign(ret20_val))
        ).astype(np.float32)

        # Win probability from alignment perspective
        win_train = (ret20 > 0.002).astype(np.float32)
        win_val = (ret20_val > 0.002).astype(np.float32)

        metrics = {}
        print(f"\n  Alignment features: {len(self.align_feature_names)}")
        print(f"  Aligned rate: {aligned_train.mean():.1%}")

        # Alignment classifier
        print("  Training alignment classifier...")
        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(align_X_train, label=aligned_train,
                            feature_name=self.align_feature_names)
        dval = lgb.Dataset(align_X_val, label=aligned_val,
                          feature_name=self.align_feature_names, reference=dtrain)

        self.alignment_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'is_unbalance': True},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        align_pred = self.alignment_model.predict(align_X_val)
        try:
            metrics['alignment_auc'] = float(roc_auc_score(aligned_val, align_pred))
        except ValueError:
            metrics['alignment_auc'] = 0.5
        print(f"    Alignment AUC: {metrics['alignment_auc']:.3f}")

        # Win model from alignment perspective
        print("  Training alignment-win classifier...")
        dtrain2 = lgb.Dataset(align_X_train, label=win_train,
                             feature_name=self.align_feature_names)
        dval2 = lgb.Dataset(align_X_val, label=win_val,
                           feature_name=self.align_feature_names, reference=dtrain2)

        self.win_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'is_unbalance': True},
            dtrain2, num_boost_round=500, valid_sets=[dval2], callbacks=callbacks)

        win_pred = self.win_model.predict(align_X_val)
        try:
            metrics['win_auc'] = float(roc_auc_score(win_val, win_pred))
        except ValueError:
            metrics['win_auc'] = 0.5
        print(f"    Win AUC: {metrics['win_auc']:.3f}")

        imp = self.alignment_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 alignment features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.align_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.alignment_model is None:
            return {}
        align_X, _ = self.derive_alignment_features(X, self.feature_names)
        return {
            'alignment_score': self.alignment_model.predict(align_X),
            'alignment_win_prob': self.win_model.predict(align_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'alignment_model': self.alignment_model, 'win_model': self.win_model,
                'feature_names': self.feature_names,
                'align_feature_names': self.align_feature_names,
            }, f)
        print(f"  Saved ChannelAlignmentScorer to {path}")

    @classmethod
    def load(cls, path: str) -> 'ChannelAlignmentScorer':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.alignment_model = data['alignment_model']
        model.win_model = data['win_model']
        model.feature_names = data['feature_names']
        model.align_feature_names = data['align_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 31: Trade Duration Predictor
# ---------------------------------------------------------------------------

class TradeDurationPredictor:
    """
    Predicts optimal trade duration based on market conditions.

    Instead of a fixed max_hold for all trades, this model predicts
    how long price will move in the entry direction. Uses OU half-life
    (natural oscillation period), channel health, and momentum patterns.

    Output: optimal_hold_bars, quick_exit_prob (should exit within 5 bars)
    """

    def __init__(self):
        self.hold_model = None
        self.quick_model = None
        self.feature_names = None
        self.dur_feature_names = None

    @staticmethod
    def derive_duration_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """Derive features for trade duration prediction."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # OU half-life (natural oscillation period)
        for tf in ['5min', '1h', '4h', 'daily']:
            for key in ['ou_half_life', 'ou_theta', 'oscillation_period']:
                idx = name_to_idx.get(f'{tf}_{key}')
                if idx is not None:
                    feats.append(X[:, idx])
                    names.append(f'dur_{tf}_{key}')

        # Channel health trajectory
        for key in ['health_min', 'health_max', 'health_delta_3bar', 'health_delta_6bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'dur_{key}')

        # Momentum (strong momentum = longer directional move)
        for key in ['price_momentum_3bar', 'price_momentum_12bar', 'rsi_slope_5bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'dur_{key}')
                feats.append(np.abs(X[:, idx]))
                names.append(f'dur_abs_{key}')

        # Volatility (high vol = faster resolution)
        atr_idx = name_to_idx.get('atr_pct')
        if atr_idx is not None:
            feats.append(X[:, atr_idx])
            names.append('dur_atr')

        # Volume (high volume = faster moves)
        for key in ['volume_ratio_20', 'vol_momentum_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'dur_{key}')

        # Position in channel
        for tf in ['5min', '1h']:
            pos_idx = name_to_idx.get(f'{tf}_position_pct')
            if pos_idx is not None:
                feats.append(X[:, pos_idx])
                names.append(f'dur_{tf}_position')
                feats.append(np.abs(X[:, pos_idx] - 0.5))
                names.append(f'dur_{tf}_edge_distance')

        # Width (narrow channels = quicker resolution)
        for tf in ['5min', '1h']:
            w_idx = name_to_idx.get(f'{tf}_width_pct')
            if w_idx is not None:
                feats.append(X[:, w_idx])
                names.append(f'dur_{tf}_width')

        # Break probability
        for key in ['break_prob_max', 'break_prob_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'dur_{key}')

        # Time of day (morning = faster moves)
        mso_idx = name_to_idx.get('minutes_since_open')
        if mso_idx is not None:
            feats.append(X[:, mso_idx])
            names.append('dur_minutes_since_open')

        # Entropy
        for key in ['avg_entropy', 'entropy_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'dur_{key}')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train duration predictor."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        dur_X_train, self.dur_feature_names = self.derive_duration_features(
            X_train, feature_names)
        dur_X_val, _ = self.derive_duration_features(X_val, feature_names)

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']

        # Hold duration proxy: how many bars until the move completes?
        # If |ret5| > |ret20|/4, most of the move happens in 5 bars → short hold
        # If |ret5| < |ret20|/4, move develops slowly → longer hold
        hold_train = np.where(
            np.abs(ret5) > np.abs(ret20) * 0.5,
            5.0,  # Quick trade
            np.where(np.abs(ret5) > np.abs(ret20) * 0.25, 12.0, 25.0)
        ).astype(np.float32)
        hold_val = np.where(
            np.abs(ret5_val) > np.abs(ret20_val) * 0.5,
            5.0,
            np.where(np.abs(ret5_val) > np.abs(ret20_val) * 0.25, 12.0, 25.0)
        ).astype(np.float32)

        # Quick exit: should we exit within 5 bars?
        quick_train = (hold_train <= 5).astype(np.float32)
        quick_val = (hold_val <= 5).astype(np.float32)

        metrics = {}
        print(f"\n  Duration features: {len(self.dur_feature_names)}")
        print(f"  Quick exit rate: {quick_train.mean():.1%}")
        print(f"  Mean hold target: {hold_train.mean():.1f} bars")

        # Hold duration regressor
        print("  Training hold duration regressor...")
        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(dur_X_train, label=hold_train,
                            feature_name=self.dur_feature_names)
        dval = lgb.Dataset(dur_X_val, label=hold_val,
                          feature_name=self.dur_feature_names, reference=dtrain)

        self.hold_model = lgb.train(
            {'objective': 'regression', 'metric': 'mae', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        hold_pred = self.hold_model.predict(dur_X_val)
        metrics['hold_mae'] = float(np.mean(np.abs(hold_pred - hold_val)))
        valid = np.isfinite(hold_pred) & np.isfinite(hold_val)
        metrics['hold_corr'] = float(np.corrcoef(hold_pred[valid], hold_val[valid])[0, 1]) if valid.sum() > 10 else 0
        print(f"    Hold MAE: {metrics['hold_mae']:.1f} bars, Corr: {metrics['hold_corr']:.3f}")

        # Quick exit classifier
        print("  Training quick exit classifier...")
        dtrain2 = lgb.Dataset(dur_X_train, label=quick_train,
                             feature_name=self.dur_feature_names)
        dval2 = lgb.Dataset(dur_X_val, label=quick_val,
                           feature_name=self.dur_feature_names, reference=dtrain2)

        self.quick_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'is_unbalance': True},
            dtrain2, num_boost_round=500, valid_sets=[dval2], callbacks=callbacks)

        quick_pred = self.quick_model.predict(dur_X_val)
        try:
            metrics['quick_auc'] = float(roc_auc_score(quick_val, quick_pred))
        except ValueError:
            metrics['quick_auc'] = 0.5
        print(f"    Quick Exit AUC: {metrics['quick_auc']:.3f}")

        imp = self.hold_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 duration features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.dur_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.hold_model is None:
            return {}
        dur_X, _ = self.derive_duration_features(X, self.feature_names)
        return {
            'optimal_hold_bars': np.clip(self.hold_model.predict(dur_X), 3, 50),
            'quick_exit_prob': self.quick_model.predict(dur_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'hold_model': self.hold_model, 'quick_model': self.quick_model,
                'feature_names': self.feature_names,
                'dur_feature_names': self.dur_feature_names,
            }, f)
        print(f"  Saved TradeDurationPredictor to {path}")

    @classmethod
    def load(cls, path: str) -> 'TradeDurationPredictor':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.hold_model = data['hold_model']
        model.quick_model = data['quick_model']
        model.feature_names = data['feature_names']
        model.dur_feature_names = data['dur_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 32: Winner Amplifier
# ---------------------------------------------------------------------------

class WinnerAmplifier:
    """
    Predicts the magnitude of winning trades to optimize exit strategy.

    Among winning trades (which we identify well), some capture 0.3% and
    others 3%. This model predicts win magnitude to:
    - Let big winners run (loosen trail, extend max hold)
    - Take quick profits on small winners (tighten trail)

    Output: big_winner_prob (will this trade > 1.5x avg win?), expected_win_magnitude
    """

    def __init__(self):
        self.big_win_model = None
        self.magnitude_model = None
        self.feature_names = None
        self.win_feature_names = None

    @staticmethod
    def derive_winner_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """Derive features for winner amplification."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Channel width (wider = more room for big win)
        for tf in ['5min', '1h', '4h']:
            w_idx = name_to_idx.get(f'{tf}_width_pct')
            if w_idx is not None:
                feats.append(X[:, w_idx])
                names.append(f'win_{tf}_width')

        # Momentum strength (strong = bigger potential win)
        for key in ['price_momentum_3bar', 'price_momentum_12bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(np.abs(X[:, idx]))
                names.append(f'win_abs_{key}')

        # Volatility
        atr_idx = name_to_idx.get('atr_pct')
        if atr_idx is not None:
            feats.append(X[:, atr_idx])
            names.append('win_atr')

        # Volume surge (high volume = more conviction)
        for key in ['volume_ratio_20', 'volume_trend_5']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'win_{key}')

        # Squeeze (squeezed = explosive when breaks)
        sq_any_idx = name_to_idx.get('squeeze_any')
        if sq_any_idx is not None:
            feats.append(X[:, sq_any_idx])
            names.append('win_squeeze_any')

        for tf in ['5min', '1h']:
            sq_idx = name_to_idx.get(f'{tf}_squeeze_score')
            if sq_idx is not None:
                feats.append(X[:, sq_idx])
                names.append(f'win_{tf}_squeeze')

        # Direction consensus (aligned = bigger move)
        for key in ['direction_consensus', 'confluence_score',
                     'bullish_fraction', 'bearish_fraction']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'win_{key}')

        # Break probability (high = trend continuation potential)
        for key in ['break_prob_max', 'break_prob_weighted']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'win_{key}')

        # Position in channel
        for tf in ['5min', '1h', '4h']:
            pos_idx = name_to_idx.get(f'{tf}_position_pct')
            if pos_idx is not None:
                feats.append(X[:, pos_idx])
                names.append(f'win_{tf}_position')

        # Energy
        for tf in ['5min', '1h']:
            for key in ['total_energy', 'kinetic_energy']:
                idx = name_to_idx.get(f'{tf}_{key}')
                if idx is not None:
                    feats.append(X[:, idx])
                    names.append(f'win_{tf}_{key}')

        # RSI extremity
        for key in ['rsi_14', 'rsi_5']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(np.abs(X[:, idx] - 50))
                names.append(f'win_{key}_extremity')

        # VIX
        vix_idx = name_to_idx.get('vix_level')
        if vix_idx is not None:
            feats.append(X[:, vix_idx])
            names.append('win_vix')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train winner amplifier."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        win_X_train, self.win_feature_names = self.derive_winner_features(
            X_train, feature_names)
        win_X_val, _ = self.derive_winner_features(X_val, feature_names)

        ret20 = Y_train['future_return_20']
        ret60 = Y_train['future_return_60']
        ret20_val = Y_val['future_return_20']
        ret60_val = Y_val['future_return_60']

        # Focus on winners: positive return magnitude
        # "Big winner" = ret60 > 1.5% (top quartile)
        big_win_train = (ret60 > 0.015).astype(np.float32)
        big_win_val = (ret60_val > 0.015).astype(np.float32)

        # Win magnitude target (positive moves only, clipped)
        win_mag_train = np.clip(np.maximum(ret20, ret60), 0, 0.05).astype(np.float32)
        win_mag_val = np.clip(np.maximum(ret20_val, ret60_val), 0, 0.05).astype(np.float32)

        metrics = {}
        print(f"\n  Winner features: {len(self.win_feature_names)}")
        print(f"  Big winner rate (ret60 > 1.5%): {big_win_train.mean():.1%}")
        print(f"  Mean win magnitude: {win_mag_train.mean():.3f}")

        # Big winner classifier
        print("  Training big winner classifier...")
        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(win_X_train, label=big_win_train,
                            feature_name=self.win_feature_names)
        dval = lgb.Dataset(win_X_val, label=big_win_val,
                          feature_name=self.win_feature_names, reference=dtrain)

        self.big_win_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'is_unbalance': True},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        big_pred = self.big_win_model.predict(win_X_val)
        try:
            metrics['big_win_auc'] = float(roc_auc_score(big_win_val, big_pred))
        except ValueError:
            metrics['big_win_auc'] = 0.5
        print(f"    Big Winner AUC: {metrics['big_win_auc']:.3f}")

        # Win magnitude regressor
        print("  Training win magnitude regressor...")
        dtrain2 = lgb.Dataset(win_X_train, label=win_mag_train,
                             feature_name=self.win_feature_names)
        dval2 = lgb.Dataset(win_X_val, label=win_mag_val,
                           feature_name=self.win_feature_names, reference=dtrain2)

        self.magnitude_model = lgb.train(
            {'objective': 'regression', 'metric': 'mae', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1},
            dtrain2, num_boost_round=500, valid_sets=[dval2], callbacks=callbacks)

        mag_pred = self.magnitude_model.predict(win_X_val)
        metrics['magnitude_mae'] = float(np.mean(np.abs(mag_pred - win_mag_val)))
        valid = np.isfinite(mag_pred) & np.isfinite(win_mag_val)
        metrics['magnitude_corr'] = float(np.corrcoef(mag_pred[valid], win_mag_val[valid])[0, 1]) if valid.sum() > 10 else 0
        print(f"    Magnitude MAE: {metrics['magnitude_mae']:.4f}, Corr: {metrics['magnitude_corr']:.3f}")

        imp = self.big_win_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 winner features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.win_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.big_win_model is None:
            return {}
        win_X, _ = self.derive_winner_features(X, self.feature_names)
        return {
            'big_winner_prob': self.big_win_model.predict(win_X),
            'expected_win_magnitude': np.clip(self.magnitude_model.predict(win_X), 0, 0.05),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'big_win_model': self.big_win_model,
                'magnitude_model': self.magnitude_model,
                'feature_names': self.feature_names,
                'win_feature_names': self.win_feature_names,
            }, f)
        print(f"  Saved WinnerAmplifier to {path}")

    @classmethod
    def load(cls, path: str) -> 'WinnerAmplifier':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.big_win_model = data['big_win_model']
        model.magnitude_model = data['magnitude_model']
        model.feature_names = data['feature_names']
        model.win_feature_names = data['win_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 33: Fractal Regime Classifier
# ---------------------------------------------------------------------------

class FractalRegimeClassifier:
    """
    Classifies market regime using fractal/Hurst exponent properties.

    Unlike simple volatility-based regime (Arch 6), this measures the
    MATHEMATICAL CHARACTER of price: trending (H>0.5) vs mean-reverting (H<0.5).
    This is orthogonal to volatility — a quiet market can be trending, a volatile
    market can be mean-reverting.

    Uses: Hurst approximation from OU parameters, r-squared as trending proxy,
    price range ratios across windows, autocorrelation structure.

    Output: trending_prob (>0.6 = trending, <0.4 = mean-reverting)
    """

    def __init__(self):
        self.trending_model = None
        self.feature_names = None
        self.frac_feature_names = None

    @staticmethod
    def derive_fractal_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """Derive fractal/Hurst features."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # R-squared across TFs (high = trending, low = choppy)
        for tf in ['5min', '1h', '4h', 'daily']:
            idx = name_to_idx.get(f'{tf}_r_squared')
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'frac_{tf}_rsq')

        # R-squared ratios (trending should be consistent across TFs)
        rsq_5m = name_to_idx.get('5min_r_squared')
        rsq_1h = name_to_idx.get('1h_r_squared')
        rsq_4h = name_to_idx.get('4h_r_squared')
        if rsq_5m is not None and rsq_1h is not None:
            feats.append(X[:, rsq_1h] - X[:, rsq_5m])
            names.append('frac_rsq_1h_minus_5m')
            feats.append(X[:, rsq_1h] * X[:, rsq_5m])
            names.append('frac_rsq_1h_times_5m')
        if rsq_1h is not None and rsq_4h is not None:
            feats.append(X[:, rsq_4h] - X[:, rsq_1h])
            names.append('frac_rsq_4h_minus_1h')

        # OU theta as mean-reversion speed (high theta = mean-reverting)
        for tf in ['5min', '1h', '4h']:
            theta_idx = name_to_idx.get(f'{tf}_ou_theta')
            if theta_idx is not None:
                feats.append(X[:, theta_idx])
                names.append(f'frac_{tf}_ou_theta')

        # OU half-life ratios (short half-life = mean-reverting)
        for tf in ['5min', '1h']:
            hl_idx = name_to_idx.get(f'{tf}_ou_half_life')
            if hl_idx is not None:
                feats.append(X[:, hl_idx])
                names.append(f'frac_{tf}_ou_hl')

        # Momentum persistence (trending = momentum continues)
        mom3 = name_to_idx.get('price_momentum_3bar')
        mom12 = name_to_idx.get('price_momentum_12bar')
        if mom3 is not None and mom12 is not None:
            # Same sign = trending, opposite = reverting
            feats.append(np.sign(X[:, mom3]) * np.sign(X[:, mom12]))
            names.append('frac_momentum_agreement')
            # Ratio: |mom3/mom12| > 1 means accelerating
            safe_mom12 = np.where(np.abs(X[:, mom12]) > 1e-8, X[:, mom12], 1e-8)
            feats.append(np.clip(X[:, mom3] / safe_mom12, -5, 5))
            names.append('frac_momentum_ratio')

        # Width dynamics (narrowing = trending continuation)
        for tf in ['5min', '1h']:
            w_idx = name_to_idx.get(f'{tf}_width_pct')
            if w_idx is not None:
                feats.append(X[:, w_idx])
                names.append(f'frac_{tf}_width')

        # Health (healthy channel = trending within bounds)
        for key in ['health_min', 'health_max', 'health_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'frac_{key}')

        # Entropy (low entropy = trending)
        for key in ['avg_entropy', 'entropy_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'frac_{key}')

        # Break probability (high = about to transition)
        for key in ['break_prob_max', 'break_prob_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'frac_{key}')

        # Direction consensus (strong = trending)
        for key in ['direction_consensus', 'bullish_fraction']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'frac_{key}')

        # ATR (volatility level)
        atr_idx = name_to_idx.get('atr_pct')
        if atr_idx is not None:
            feats.append(X[:, atr_idx])
            names.append('frac_atr')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train fractal regime classifier."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        frac_X_train, self.frac_feature_names = self.derive_fractal_features(
            X_train, feature_names)
        frac_X_val, _ = self.derive_fractal_features(X_val, feature_names)

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']

        # Trending: same direction across time windows AND |ret20| > 2 * |ret5|
        # (consistent directional move, not a spike-and-reverse)
        trending_train = (
            (np.sign(ret5) == np.sign(ret20)) &
            (np.abs(ret20) > np.abs(ret5) * 1.5) &
            (np.abs(ret20) > 0.005)
        ).astype(np.float32)
        trending_val = (
            (np.sign(ret5_val) == np.sign(ret20_val)) &
            (np.abs(ret20_val) > np.abs(ret5_val) * 1.5) &
            (np.abs(ret20_val) > 0.005)
        ).astype(np.float32)

        metrics = {}
        print(f"\n  Fractal features: {len(self.frac_feature_names)}")
        print(f"  Trending rate: {trending_train.mean():.1%}")

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(frac_X_train, label=trending_train,
                            feature_name=self.frac_feature_names)
        dval = lgb.Dataset(frac_X_val, label=trending_val,
                          feature_name=self.frac_feature_names, reference=dtrain)

        self.trending_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'is_unbalance': True},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        trend_pred = self.trending_model.predict(frac_X_val)
        try:
            metrics['trending_auc'] = float(roc_auc_score(trending_val, trend_pred))
        except ValueError:
            metrics['trending_auc'] = 0.5
        print(f"    Trending AUC: {metrics['trending_auc']:.3f}")

        imp = self.trending_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 fractal features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.frac_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.trending_model is None:
            return {}
        frac_X, _ = self.derive_fractal_features(X, self.feature_names)
        return {
            'trending_prob': self.trending_model.predict(frac_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'trending_model': self.trending_model,
                'feature_names': self.feature_names,
                'frac_feature_names': self.frac_feature_names,
            }, f)
        print(f"  Saved FractalRegimeClassifier to {path}")

    @classmethod
    def load(cls, path: str) -> 'FractalRegimeClassifier':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.trending_model = data['trending_model']
        model.feature_names = data['feature_names']
        model.frac_feature_names = data['frac_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 34: Volume-Conviction Classifier
# ---------------------------------------------------------------------------

class VolumeConvictionClassifier:
    """
    Classifies trades based on volume conviction — whether the current price
    movement has institutional-level volume support.

    Key insight: Volume confirms price. A breakout on high volume is real,
    on low volume it's a fake-out. But simple volume_ratio misses the pattern:
    we need volume RELATIVE to the move magnitude and the time context.

    Features: volume/price-change efficiency, relative volume in direction,
    volume acceleration, volume-weighted momentum.

    Output: conviction_prob (>0.6 = volume-confirmed, <0.3 = suspect)
    """

    def __init__(self):
        self.conviction_model = None
        self.feature_names = None
        self.vol_feature_names = None

    @staticmethod
    def derive_volume_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """Derive volume-conviction features."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Volume ratio (basic)
        vol_r20 = name_to_idx.get('volume_ratio_20')
        if vol_r20 is not None:
            feats.append(X[:, vol_r20])
            names.append('vc_vol_ratio_20')

        # Volume momentum
        for key in ['vol_momentum_3bar', 'volume_trend_5']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'vc_{key}')

        # Volume-price efficiency: how much price moves per unit volume
        # High efficiency = genuine move, low = churning
        vol_idx = vol_r20
        mom3 = name_to_idx.get('price_momentum_3bar')
        mom12 = name_to_idx.get('price_momentum_12bar')
        atr_idx = name_to_idx.get('atr_pct')

        if vol_idx is not None and mom3 is not None:
            safe_vol = np.where(X[:, vol_idx] > 0.01, X[:, vol_idx], 0.01)
            feats.append(np.abs(X[:, mom3]) / safe_vol)
            names.append('vc_price_per_volume_3')

        if vol_idx is not None and mom12 is not None:
            safe_vol = np.where(X[:, vol_idx] > 0.01, X[:, vol_idx], 0.01)
            feats.append(np.abs(X[:, mom12]) / safe_vol)
            names.append('vc_price_per_volume_12')

        # Volume-momentum alignment
        if vol_idx is not None and mom3 is not None:
            # High vol + big move = conviction
            feats.append(X[:, vol_idx] * np.abs(X[:, mom3]))
            names.append('vc_vol_mom_product_3')

        # ATR context (vol is more meaningful in low-ATR environment)
        if atr_idx is not None:
            feats.append(X[:, atr_idx])
            names.append('vc_atr')

        # Volume across TFs
        # (captured indirectly via TF-specific features)

        # Channel width interaction with volume
        for tf in ['5min', '1h']:
            w_idx = name_to_idx.get(f'{tf}_width_pct')
            if w_idx is not None and vol_idx is not None:
                # Volume relative to channel width
                safe_w = np.where(X[:, w_idx] > 1e-6, X[:, w_idx], 1e-6)
                feats.append(X[:, vol_idx] / safe_w)
                names.append(f'vc_{tf}_vol_per_width')

        # Break probability with volume
        bp_idx = name_to_idx.get('break_prob_max')
        if bp_idx is not None and vol_idx is not None:
            feats.append(X[:, bp_idx] * X[:, vol_idx])
            names.append('vc_breakprob_volume')

        # Position in channel with volume
        for tf in ['5min', '1h']:
            pos_idx = name_to_idx.get(f'{tf}_position_pct')
            if pos_idx is not None and vol_idx is not None:
                feats.append(X[:, pos_idx] * X[:, vol_idx])
                names.append(f'vc_{tf}_position_volume')

        # Squeeze with volume (squeezed + high vol = explosive)
        sq_idx = name_to_idx.get('squeeze_any')
        if sq_idx is not None and vol_idx is not None:
            feats.append(X[:, sq_idx] * X[:, vol_idx])
            names.append('vc_squeeze_volume')

        # RSI with volume
        rsi_idx = name_to_idx.get('rsi_14')
        if rsi_idx is not None and vol_idx is not None:
            feats.append(np.abs(X[:, rsi_idx] - 50) * X[:, vol_idx])
            names.append('vc_rsi_extreme_volume')

        # Time of day
        mso_idx = name_to_idx.get('minutes_since_open')
        if mso_idx is not None:
            feats.append(X[:, mso_idx])
            names.append('vc_minutes_since_open')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train volume conviction classifier."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        vol_X_train, self.vol_feature_names = self.derive_volume_features(
            X_train, feature_names)
        vol_X_val, _ = self.derive_volume_features(X_val, feature_names)

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret60 = Y_train['future_return_60']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']
        ret60_val = Y_val['future_return_60']

        # "Conviction" = the move continues: same direction ret5→ret20→ret60
        # AND magnitude grows. This suggests real, volume-backed movement.
        conviction_train = (
            (np.sign(ret5) == np.sign(ret20)) &
            (np.sign(ret20) == np.sign(ret60)) &
            (np.abs(ret60) > np.abs(ret20))
        ).astype(np.float32)
        conviction_val = (
            (np.sign(ret5_val) == np.sign(ret20_val)) &
            (np.sign(ret20_val) == np.sign(ret60_val)) &
            (np.abs(ret60_val) > np.abs(ret20_val))
        ).astype(np.float32)

        metrics = {}
        print(f"\n  Volume-Conviction features: {len(self.vol_feature_names)}")
        print(f"  Conviction rate: {conviction_train.mean():.1%}")

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(vol_X_train, label=conviction_train,
                            feature_name=self.vol_feature_names)
        dval = lgb.Dataset(vol_X_val, label=conviction_val,
                          feature_name=self.vol_feature_names, reference=dtrain)

        self.conviction_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'is_unbalance': True},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        conv_pred = self.conviction_model.predict(vol_X_val)
        try:
            metrics['conviction_auc'] = float(roc_auc_score(conviction_val, conv_pred))
        except ValueError:
            metrics['conviction_auc'] = 0.5
        print(f"    Conviction AUC: {metrics['conviction_auc']:.3f}")

        imp = self.conviction_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 volume conviction features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.vol_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.conviction_model is None:
            return {}
        vol_X, _ = self.derive_volume_features(X, self.feature_names)
        return {
            'conviction_prob': self.conviction_model.predict(vol_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'conviction_model': self.conviction_model,
                'feature_names': self.feature_names,
                'vol_feature_names': self.vol_feature_names,
            }, f)
        print(f"  Saved VolumeConvictionClassifier to {path}")

    @classmethod
    def load(cls, path: str) -> 'VolumeConvictionClassifier':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.conviction_model = data['conviction_model']
        model.feature_names = data['feature_names']
        model.vol_feature_names = data['vol_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 35: Energy Momentum Detector
# ---------------------------------------------------------------------------

class EnergyMomentumDetector:
    """
    Uses physics energy concepts to predict explosive moves.

    Channels have kinetic energy (momentum) and potential energy (compression).
    When total energy is high but kinetic is low, the channel is "loaded" —
    potential energy converts to kinetic (explosive breakout).

    This is different from squeeze detection (Arch 14) because it uses the
    RATIO of energies and their rates of change, not just width/volatility.

    Output: explosive_prob (>0.6 = loaded for explosive move)
    """

    def __init__(self):
        self.explosive_model = None
        self.feature_names = None
        self.energy_feature_names = None

    @staticmethod
    def derive_energy_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """Derive energy-based features."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Per-TF energy components
        for tf in ['5min', '1h', '4h']:
            ke_idx = name_to_idx.get(f'{tf}_kinetic_energy')
            te_idx = name_to_idx.get(f'{tf}_total_energy')
            pe_idx = name_to_idx.get(f'{tf}_potential_energy')

            if ke_idx is not None:
                feats.append(X[:, ke_idx])
                names.append(f'eng_{tf}_kinetic')
            if te_idx is not None:
                feats.append(X[:, te_idx])
                names.append(f'eng_{tf}_total')
            if pe_idx is not None:
                feats.append(X[:, pe_idx])
                names.append(f'eng_{tf}_potential')

            # Energy ratio: potential/total (high = loaded spring)
            if te_idx is not None and ke_idx is not None:
                safe_te = np.where(X[:, te_idx] > 1e-6, X[:, te_idx], 1e-6)
                feats.append(X[:, ke_idx] / safe_te)
                names.append(f'eng_{tf}_kinetic_ratio')
                # "Loading" = total increasing but kinetic decreasing
                if pe_idx is not None:
                    feats.append(X[:, pe_idx] / safe_te)
                    names.append(f'eng_{tf}_potential_ratio')

        # Cross-TF energy alignment
        ke_5m = name_to_idx.get('5min_kinetic_energy')
        ke_1h = name_to_idx.get('1h_kinetic_energy')
        ke_4h = name_to_idx.get('4h_kinetic_energy')
        if ke_5m is not None and ke_1h is not None:
            feats.append(X[:, ke_5m] - X[:, ke_1h])
            names.append('eng_ke_5m_minus_1h')
        if ke_1h is not None and ke_4h is not None:
            feats.append(X[:, ke_1h] - X[:, ke_4h])
            names.append('eng_ke_1h_minus_4h')

        # Width dynamics (narrowing = energy building)
        for tf in ['5min', '1h']:
            w_idx = name_to_idx.get(f'{tf}_width_pct')
            if w_idx is not None:
                feats.append(X[:, w_idx])
                names.append(f'eng_{tf}_width')

        # Squeeze (compressed = high potential energy)
        for key in ['squeeze_any', 'squeeze_count']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'eng_{key}')
        for tf in ['5min', '1h']:
            sq_idx = name_to_idx.get(f'{tf}_squeeze_score')
            if sq_idx is not None:
                feats.append(X[:, sq_idx])
                names.append(f'eng_{tf}_squeeze')

        # Momentum (current kinetic energy direction)
        for key in ['price_momentum_3bar', 'price_momentum_12bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(np.abs(X[:, idx]))
                names.append(f'eng_abs_{key}')

        # ATR (volatility = energy dissipation rate)
        atr_idx = name_to_idx.get('atr_pct')
        if atr_idx is not None:
            feats.append(X[:, atr_idx])
            names.append('eng_atr')

        # Volume (energy transfer medium)
        vol_idx = name_to_idx.get('volume_ratio_20')
        if vol_idx is not None:
            feats.append(X[:, vol_idx])
            names.append('eng_volume_ratio')

        # Break probability
        bp_idx = name_to_idx.get('break_prob_max')
        if bp_idx is not None:
            feats.append(X[:, bp_idx])
            names.append('eng_break_prob')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train energy momentum detector."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        eng_X_train, self.energy_feature_names = self.derive_energy_features(
            X_train, feature_names)
        eng_X_val, _ = self.derive_energy_features(X_val, feature_names)

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']

        # "Explosive" = large move in 20 bars AND most of it happens after bar 5
        # (energy release pattern: quiet → boom)
        explosive_train = (
            (np.abs(ret20) > 0.01) &  # At least 1% move
            (np.abs(ret20) > np.abs(ret5) * 2.5)  # Accelerating
        ).astype(np.float32)
        explosive_val = (
            (np.abs(ret20_val) > 0.01) &
            (np.abs(ret20_val) > np.abs(ret5_val) * 2.5)
        ).astype(np.float32)

        metrics = {}
        print(f"\n  Energy features: {len(self.energy_feature_names)}")
        print(f"  Explosive rate: {explosive_train.mean():.1%}")

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(eng_X_train, label=explosive_train,
                            feature_name=self.energy_feature_names)
        dval = lgb.Dataset(eng_X_val, label=explosive_val,
                          feature_name=self.energy_feature_names, reference=dtrain)

        self.explosive_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'is_unbalance': True},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        exp_pred = self.explosive_model.predict(eng_X_val)
        try:
            metrics['explosive_auc'] = float(roc_auc_score(explosive_val, exp_pred))
        except ValueError:
            metrics['explosive_auc'] = 0.5
        print(f"    Explosive AUC: {metrics['explosive_auc']:.3f}")

        imp = self.explosive_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 energy features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.energy_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.explosive_model is None:
            return {}
        eng_X, _ = self.derive_energy_features(X, self.feature_names)
        return {
            'explosive_prob': self.explosive_model.predict(eng_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'explosive_model': self.explosive_model,
                'feature_names': self.feature_names,
                'energy_feature_names': self.energy_feature_names,
            }, f)
        print(f"  Saved EnergyMomentumDetector to {path}")

    @classmethod
    def load(cls, path: str) -> 'EnergyMomentumDetector':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.explosive_model = data['explosive_model']
        model.feature_names = data['feature_names']
        model.energy_feature_names = data['energy_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 36: Multi-Exit Strategy Selector
# ---------------------------------------------------------------------------

class MultiExitStrategySelector:
    """
    Instead of one exit strategy for all trades, predicts which exit strategy
    will perform best for this specific trade setup.

    Exit strategies:
    0 = Tight trail (for quick profits in choppy conditions)
    1 = Wide trail (for riding trends)
    2 = Time-based (exit after N bars regardless)
    3 = Target-based (exit at TP, ignore trail)

    Uses trade setup features to predict optimal strategy.
    Trains on retrospective analysis: which strategy WOULD have maximized P&L?

    Output: best_strategy (0-3), strategy_probs
    """

    def __init__(self):
        self.strategy_model = None
        self.feature_names = None
        self.exit_feature_names = None

    @staticmethod
    def derive_exit_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """Derive exit strategy selection features."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Channel structure (tight trail works in narrow channels)
        for tf in ['5min', '1h', '4h']:
            for key in ['width_pct', 'r_squared', 'position_pct']:
                idx = name_to_idx.get(f'{tf}_{key}')
                if idx is not None:
                    feats.append(X[:, idx])
                    names.append(f'exit_{tf}_{key}')

        # OU dynamics (mean-reverting → tight trail, trending → wide trail)
        for tf in ['5min', '1h']:
            for key in ['ou_half_life', 'ou_theta']:
                idx = name_to_idx.get(f'{tf}_{key}')
                if idx is not None:
                    feats.append(X[:, idx])
                    names.append(f'exit_{tf}_{key}')

        # Volatility
        atr_idx = name_to_idx.get('atr_pct')
        if atr_idx is not None:
            feats.append(X[:, atr_idx])
            names.append('exit_atr')

        # Momentum strength
        for key in ['price_momentum_3bar', 'price_momentum_12bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(np.abs(X[:, idx]))
                names.append(f'exit_abs_{key}')

        # Break probability
        for key in ['break_prob_max', 'break_prob_weighted']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'exit_{key}')

        # Health
        for key in ['health_min', 'health_max', 'health_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'exit_{key}')

        # Entropy
        for key in ['avg_entropy', 'entropy_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'exit_{key}')

        # Volume
        vol_idx = name_to_idx.get('volume_ratio_20')
        if vol_idx is not None:
            feats.append(X[:, vol_idx])
            names.append('exit_volume_ratio')

        # VIX
        vix_idx = name_to_idx.get('vix_level')
        if vix_idx is not None:
            feats.append(X[:, vix_idx])
            names.append('exit_vix')

        # Time
        mso_idx = name_to_idx.get('minutes_since_open')
        if mso_idx is not None:
            feats.append(X[:, mso_idx])
            names.append('exit_minutes_since_open')

        # Direction consensus
        dc_idx = name_to_idx.get('direction_consensus')
        if dc_idx is not None:
            feats.append(X[:, dc_idx])
            names.append('exit_direction_consensus')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train multi-exit strategy selector."""
        import lightgbm as lgb
        from sklearn.metrics import accuracy_score

        self.feature_names = list(feature_names)
        exit_X_train, self.exit_feature_names = self.derive_exit_features(
            X_train, feature_names)
        exit_X_val, _ = self.derive_exit_features(X_val, feature_names)

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret60 = Y_train['future_return_60']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']
        ret60_val = Y_val['future_return_60']

        # Determine best exit strategy retroactively:
        # 0 = Tight trail: best when |ret5| > |ret20| (quick profit, then reversal)
        # 1 = Wide trail: best when |ret60| > |ret20| > |ret5| (trending)
        # 2 = Time-based: best when ret5 ≈ ret20 (flat, just exit)
        # 3 = Target-based: best when |ret20| large, |ret60| < |ret20| (spike then fade)

        def classify_strategy(r5, r20, r60):
            result = np.full(len(r5), 2, dtype=np.int32)  # default: time-based
            # Tight trail: quick spike
            tight_mask = np.abs(r5) > np.abs(r20) * 0.8
            result[tight_mask] = 0
            # Wide trail: trending continuation
            wide_mask = (np.abs(r60) > np.abs(r20)) & (np.abs(r20) > np.abs(r5) * 1.3)
            result[wide_mask] = 1
            # Target: spike then fade
            target_mask = (np.abs(r20) > 0.005) & (np.abs(r60) < np.abs(r20) * 0.7)
            result[target_mask] = 3
            return result

        strategy_train = classify_strategy(ret5, ret20, ret60)
        strategy_val = classify_strategy(ret5_val, ret20_val, ret60_val)

        metrics = {}
        print(f"\n  Exit Strategy features: {len(self.exit_feature_names)}")
        for s in range(4):
            print(f"  Strategy {s}: {(strategy_train == s).mean():.1%}")

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(exit_X_train, label=strategy_train,
                            feature_name=self.exit_feature_names)
        dval = lgb.Dataset(exit_X_val, label=strategy_val,
                          feature_name=self.exit_feature_names, reference=dtrain)

        self.strategy_model = lgb.train(
            {'objective': 'multiclass', 'num_class': 4, 'metric': 'multi_logloss',
             'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8,
             'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        strat_pred = self.strategy_model.predict(exit_X_val)
        pred_class = np.argmax(strat_pred, axis=1)
        metrics['strategy_accuracy'] = float(accuracy_score(strategy_val, pred_class))
        print(f"    Strategy Accuracy: {metrics['strategy_accuracy']:.3f}")

        # Per-class accuracy
        for s in range(4):
            mask = strategy_val == s
            if mask.sum() > 0:
                acc = float((pred_class[mask] == s).mean())
                metrics[f'strategy_{s}_acc'] = acc
                print(f"    Strategy {s} acc: {acc:.3f} (n={mask.sum()})")

        imp = self.strategy_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 exit strategy features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.exit_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.strategy_model is None:
            return {}
        exit_X, _ = self.derive_exit_features(X, self.feature_names)
        probs = self.strategy_model.predict(exit_X)
        return {
            'best_exit_strategy': np.argmax(probs, axis=1),
            'exit_strategy_probs': probs,
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'strategy_model': self.strategy_model,
                'feature_names': self.feature_names,
                'exit_feature_names': self.exit_feature_names,
            }, f)
        print(f"  Saved MultiExitStrategySelector to {path}")

    @classmethod
    def load(cls, path: str) -> 'MultiExitStrategySelector':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.strategy_model = data['strategy_model']
        model.feature_names = data['feature_names']
        model.exit_feature_names = data['exit_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 37: Adversarial Trade Selector
# ---------------------------------------------------------------------------

class AdversarialTradeSelector:
    """
    Uses adversarial validation to find what distinguishes GOOD trades from
    BAD trades, then builds a focused model on those distinguishing features.

    Key insight: Instead of predicting market direction (efficient market ceiling),
    predict which TRADE SETUPS are most favorable. This is a meta-question about
    our trading system, not about the market itself.

    Trains on the FULL feature set (not derived features) but uses adversarial
    feature selection: find the features where good/bad trade distributions differ
    most, then focus on those.

    Output: favorable_prob (>0.6 = favorable setup, <0.3 = avoid)
    """

    def __init__(self):
        self.selector_model = None
        self.feature_names = None
        self.selected_feature_indices = None
        self.selected_feature_names = None

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train adversarial trade selector."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']

        # Define "favorable" as: trade would have been profitable
        # with a reasonable stop (-0.5%) and reasonable hold (20 bars)
        # Favorable = positive ret20 AND max drawdown (ret5) not below -0.5%
        favorable_train = (
            (ret20 > 0.002) &  # At least 0.2% profit
            (ret5 > -0.005)    # Doesn't draw down more than 0.5% early
        ).astype(np.float32)
        favorable_val = (
            (ret20_val > 0.002) &
            (ret5_val > -0.005)
        ).astype(np.float32)

        metrics = {}
        print(f"\n  Total features: {len(feature_names)}")
        print(f"  Favorable rate: {favorable_train.mean():.1%}")

        # Phase 1: Adversarial feature selection
        # Train a model to distinguish favorable from unfavorable, extract top features
        print("  Phase 1: Adversarial feature selection...")
        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(X_train, label=favorable_train,
                            feature_name=list(feature_names))
        dval = lgb.Dataset(X_val, label=favorable_val,
                          feature_name=list(feature_names), reference=dtrain)

        selector = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 63,
             'learning_rate': 0.05, 'feature_fraction': 0.5, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'is_unbalance': True},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        # Get initial AUC on full feature set
        full_pred = selector.predict(X_val)
        try:
            full_auc = float(roc_auc_score(favorable_val, full_pred))
        except ValueError:
            full_auc = 0.5
        print(f"    Full feature AUC: {full_auc:.3f}")
        metrics['full_feature_auc'] = full_auc

        # Select top features by importance
        imp = selector.feature_importance(importance_type='gain')
        top_n = min(30, len(imp))  # Top 30 most discriminative features
        self.selected_feature_indices = np.argsort(imp)[::-1][:top_n]
        self.selected_feature_names = [feature_names[i] for i in self.selected_feature_indices]

        print(f"\n  Top 15 discriminative features:")
        for rank, idx in enumerate(self.selected_feature_indices[:15]):
            print(f"    {rank+1}. {feature_names[idx]}: {imp[idx]:.0f}")

        # Phase 2: Focused model on selected features only
        print(f"\n  Phase 2: Focused model on {top_n} features...")
        sel_X_train = X_train[:, self.selected_feature_indices]
        sel_X_val = X_val[:, self.selected_feature_indices]

        dtrain2 = lgb.Dataset(sel_X_train, label=favorable_train,
                             feature_name=self.selected_feature_names)
        dval2 = lgb.Dataset(sel_X_val, label=favorable_val,
                           feature_name=self.selected_feature_names, reference=dtrain2)

        self.selector_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'is_unbalance': True},
            dtrain2, num_boost_round=500, valid_sets=[dval2], callbacks=callbacks)

        sel_pred = self.selector_model.predict(sel_X_val)
        try:
            metrics['favorable_auc'] = float(roc_auc_score(favorable_val, sel_pred))
        except ValueError:
            metrics['favorable_auc'] = 0.5
        print(f"    Focused model AUC: {metrics['favorable_auc']:.3f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.selector_model is None:
            return {}
        sel_X = X[:, self.selected_feature_indices]
        return {
            'favorable_prob': self.selector_model.predict(sel_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'selector_model': self.selector_model,
                'feature_names': self.feature_names,
                'selected_feature_indices': self.selected_feature_indices,
                'selected_feature_names': self.selected_feature_names,
            }, f)
        print(f"  Saved AdversarialTradeSelector to {path}")

    @classmethod
    def load(cls, path: str) -> 'AdversarialTradeSelector':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.selector_model = data['selector_model']
        model.feature_names = data['feature_names']
        model.selected_feature_indices = data['selected_feature_indices']
        model.selected_feature_names = data['selected_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 38: Cascade Confidence Optimizer
# ---------------------------------------------------------------------------

class CascadeConfidenceOptimizer:
    """
    Meta-model that takes predictions from ALL other models as features
    and learns the optimal confidence adjustment.

    Instead of each model independently multiplying confidence (which compounds
    and can over-filter), this model learns the RIGHT combination.

    Key difference from Bayesian Combiner (Arch 19, failed): Arch 19 tried to
    learn which models to trust. This model takes the raw PREDICTIONS from each
    model and learns how they interact to predict trade quality.

    Requires: All other models to be already trained and available.
    Input: 169 raw features + N model predictions
    Output: optimal_confidence_scale
    """

    def __init__(self):
        self.cascade_model = None
        self.feature_names = None
        self.cascade_feature_names = None
        self.model_paths = None

    def train(self, X_train, Y_train, X_val, Y_val, feature_names,
              model_dir: str = 'surfer_models'):
        """Train cascade confidence optimizer using all available model predictions."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        self.model_paths = model_dir

        # Load all available models and generate predictions
        model_preds_train = []
        model_preds_val = []
        cascade_names = []

        # Try loading each model and getting predictions
        model_specs = [
            ('quality', 'quality_model.pkl', TradeQualityScorer, ['win_prob', 'pnl_direction']),
            ('adverse', 'adverse_model.pkl', AdverseMovementPredictor, ['stop_hit_prob']),
            ('composite', 'composite_model.pkl', CompositeSignalScorer, ['composite_score']),
            ('vol_trans', 'vol_transition_model.pkl', VolatilityTransitionModel, ['vol_spike_prob']),
            ('exit', 'exit_model.pkl', ExitTimingOptimizer, ['exit_direction']),
            ('exhaustion', 'exhaustion_model.pkl', MomentumExhaustionDetector, ['exhausted_prob']),
            ('trail', 'trail_model.pkl', DynamicTrailOptimizer, ['tighten_prob']),
            ('session', 'session_model.pkl', IntradaySessionModel, ['session_quality']),
            ('maturity', 'maturity_model.pkl', ChannelMaturityPredictor, ['mature_prob']),
            ('asymmetry', 'asymmetry_model.pkl', ReturnAsymmetryPredictor, ['spike_prob']),
            ('reversion', 'reversion_model.pkl', MeanReversionSpeedModel, ['fast_reversion_prob']),
        ]

        for name, filename, cls, pred_keys in model_specs:
            path = os.path.join(model_dir, filename)
            if os.path.exists(path):
                try:
                    model = cls.load(path)
                    pred_train = model.predict(X_train)
                    pred_val = model.predict(X_val)
                    for key in pred_keys:
                        if key in pred_train:
                            model_preds_train.append(pred_train[key].flatten())
                            model_preds_val.append(pred_val[key].flatten())
                            cascade_names.append(f'cascade_{name}_{key}')
                            print(f"    Loaded {name}.{key}")
                except Exception as e:
                    print(f"    Skip {name}: {e}")

        if len(model_preds_train) == 0:
            print("  No models available for cascade!")
            return {'cascade_auc': 0.5}

        # Combine: raw features + model predictions
        cascade_X_train = np.column_stack([X_train] + model_preds_train)
        cascade_X_val = np.column_stack([X_val] + model_preds_val)
        self.cascade_feature_names = list(feature_names) + cascade_names

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']

        # Target: would this trade be a winner?
        # Same definition: positive ret20, no deep drawdown
        winner_train = (
            (ret20 > 0.001) & (ret5 > -0.005)
        ).astype(np.float32)
        winner_val = (
            (ret20_val > 0.001) & (ret5_val > -0.005)
        ).astype(np.float32)

        metrics = {}
        print(f"\n  Cascade features: {len(self.cascade_feature_names)} ({len(feature_names)} raw + {len(cascade_names)} model)")
        print(f"  Winner rate: {winner_train.mean():.1%}")

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(cascade_X_train, label=winner_train,
                            feature_name=self.cascade_feature_names)
        dval = lgb.Dataset(cascade_X_val, label=winner_val,
                          feature_name=self.cascade_feature_names, reference=dtrain)

        self.cascade_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.5, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'is_unbalance': True},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        cascade_pred = self.cascade_model.predict(cascade_X_val)
        try:
            metrics['cascade_auc'] = float(roc_auc_score(winner_val, cascade_pred))
        except ValueError:
            metrics['cascade_auc'] = 0.5
        print(f"    Cascade AUC: {metrics['cascade_auc']:.3f}")

        # Check how much the model predictions contribute
        imp = self.cascade_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:15]
        print("\n  Top 15 cascade features:")
        model_contrib = 0
        total_imp = imp.sum()
        for rank, idx in enumerate(top_idx):
            name = self.cascade_feature_names[idx]
            is_model = name.startswith('cascade_')
            marker = ' [MODEL]' if is_model else ''
            print(f"    {rank+1}. {name}: {imp[idx]:.0f}{marker}")
        for idx in range(len(feature_names), len(self.cascade_feature_names)):
            model_contrib += imp[idx]
        if total_imp > 0:
            metrics['model_contribution_pct'] = float(model_contrib / total_imp * 100)
            print(f"\n  Model predictions contribute {metrics['model_contribution_pct']:.1f}% of total importance")

        return metrics

    def predict(self, X: np.ndarray, model_preds: dict = None) -> dict:
        if self.cascade_model is None:
            return {}
        # In practice, you'd pass all model predictions too
        # For now, just use raw features padded with zeros for model cols
        n_model_feats = len(self.cascade_feature_names) - len(self.feature_names)
        if model_preds is not None:
            extra = []
            for name in self.cascade_feature_names[len(self.feature_names):]:
                key = name.replace('cascade_', '', 1)
                extra.append(model_preds.get(key, np.zeros(X.shape[0])))
            cascade_X = np.column_stack([X] + extra)
        else:
            cascade_X = np.column_stack([X, np.zeros((X.shape[0], n_model_feats))])
        return {
            'cascade_confidence': self.cascade_model.predict(cascade_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'cascade_model': self.cascade_model,
                'feature_names': self.feature_names,
                'cascade_feature_names': self.cascade_feature_names,
                'model_paths': self.model_paths,
            }, f)
        print(f"  Saved CascadeConfidenceOptimizer to {path}")

    @classmethod
    def load(cls, path: str) -> 'CascadeConfidenceOptimizer':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.cascade_model = data['cascade_model']
        model.feature_names = data['feature_names']
        model.cascade_feature_names = data['cascade_feature_names']
        model.model_paths = data['model_paths']
        return model


# ---------------------------------------------------------------------------
# Architecture 39: Nearest-Neighbor Trade Analogy
# ---------------------------------------------------------------------------

class NearestNeighborTradeAnalogy:
    """
    Instead of learning a function, find the K most SIMILAR historical trade
    setups and use their outcomes. This is fundamentally different from all GBT
    models because it uses local similarity rather than global decision boundaries.

    With ~1400 samples, kNN can capture patterns that GBT misses:
    - Rare but important feature combinations
    - Non-linear interactions in local neighborhoods
    - No overfitting to decision tree splits

    Uses PCA dimensionality reduction to handle curse of dimensionality.

    Output: neighbor_win_rate, neighbor_avg_return, confidence_from_consensus
    """

    def __init__(self):
        self.pca = None
        self.scaler = None
        self.X_train_reduced = None
        self.y_win_train = None
        self.y_ret_train = None
        self.feature_names = None
        self.k = 15
        self.n_components = 20

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train nearest-neighbor model."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']

        # Win: positive ret20 with limited drawdown
        y_win_train = ((ret20 > 0.001) & (ret5 > -0.005)).astype(np.float32)
        y_win_val = ((ret20_val > 0.001) & (ret5_val > -0.005)).astype(np.float32)
        self.y_win_train = y_win_train
        self.y_ret_train = ret20.astype(np.float32)

        # Standardize + PCA
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Handle NaN/Inf
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=0, neginf=0)
        X_val_scaled = np.nan_to_num(X_val_scaled, nan=0, posinf=0, neginf=0)

        self.pca = PCA(n_components=self.n_components)
        self.X_train_reduced = self.pca.fit_transform(X_train_scaled)
        X_val_reduced = self.pca.transform(X_val_scaled)

        metrics = {}
        var_explained = self.pca.explained_variance_ratio_.sum()
        print(f"\n  PCA: {self.n_components} components explain {var_explained:.1%} variance")
        print(f"  K neighbors: {self.k}")
        print(f"  Train win rate: {y_win_train.mean():.1%}")

        # Evaluate: for each val sample, find K nearest in train
        from scipy.spatial.distance import cdist
        dists = cdist(X_val_reduced, self.X_train_reduced, metric='euclidean')

        win_preds = []
        ret_preds = []
        for i in range(len(X_val_reduced)):
            nn_idx = np.argsort(dists[i])[:self.k]
            win_preds.append(self.y_win_train[nn_idx].mean())
            ret_preds.append(self.y_ret_train[nn_idx].mean())

        win_preds = np.array(win_preds)
        ret_preds = np.array(ret_preds)

        try:
            metrics['neighbor_win_auc'] = float(roc_auc_score(y_win_val, win_preds))
        except ValueError:
            metrics['neighbor_win_auc'] = 0.5
        print(f"    Neighbor Win AUC: {metrics['neighbor_win_auc']:.3f}")

        # Correlation of predicted return with actual
        valid = np.isfinite(ret_preds) & np.isfinite(ret20_val)
        if valid.sum() > 10:
            metrics['return_corr'] = float(np.corrcoef(ret_preds[valid], ret20_val[valid])[0, 1])
        else:
            metrics['return_corr'] = 0
        print(f"    Return Corr: {metrics['return_corr']:.3f}")

        # Win rate of top/bottom quartile
        q75 = np.percentile(win_preds, 75)
        q25 = np.percentile(win_preds, 25)
        top_mask = win_preds >= q75
        bot_mask = win_preds <= q25
        if top_mask.sum() > 0:
            metrics['top_quartile_wr'] = float(y_win_val[top_mask].mean())
            print(f"    Top quartile WR: {metrics['top_quartile_wr']:.1%} (n={top_mask.sum()})")
        if bot_mask.sum() > 0:
            metrics['bottom_quartile_wr'] = float(y_win_val[bot_mask].mean())
            print(f"    Bottom quartile WR: {metrics['bottom_quartile_wr']:.1%} (n={bot_mask.sum()})")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.X_train_reduced is None:
            return {}
        from scipy.spatial.distance import cdist
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
        X_reduced = self.pca.transform(X_scaled)
        dists = cdist(X_reduced, self.X_train_reduced, metric='euclidean')

        win_rates = []
        avg_rets = []
        for i in range(len(X_reduced)):
            nn_idx = np.argsort(dists[i])[:self.k]
            win_rates.append(self.y_win_train[nn_idx].mean())
            avg_rets.append(self.y_ret_train[nn_idx].mean())

        return {
            'neighbor_win_rate': np.array(win_rates),
            'neighbor_avg_return': np.array(avg_rets),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'pca': self.pca, 'scaler': self.scaler,
                'X_train_reduced': self.X_train_reduced,
                'y_win_train': self.y_win_train,
                'y_ret_train': self.y_ret_train,
                'feature_names': self.feature_names,
                'k': self.k, 'n_components': self.n_components,
            }, f)
        print(f"  Saved NearestNeighborTradeAnalogy to {path}")

    @classmethod
    def load(cls, path: str) -> 'NearestNeighborTradeAnalogy':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.pca = data['pca']
        model.scaler = data['scaler']
        model.X_train_reduced = data['X_train_reduced']
        model.y_win_train = data['y_win_train']
        model.y_ret_train = data['y_ret_train']
        model.feature_names = data['feature_names']
        model.k = data['k']
        model.n_components = data['n_components']
        return model


# ---------------------------------------------------------------------------
# Architecture 40: Quantile Risk Estimator
# ---------------------------------------------------------------------------

class QuantileRiskEstimator:
    """
    Predicts the DISTRIBUTION of potential outcomes, not just the mean.
    Uses quantile regression to estimate P10, P50, P90 of future returns.

    Key insight: mean return predictions are useless (efficient market).
    But the SHAPE of the return distribution (wide vs narrow, symmetric vs
    skewed) is predictable and actionable:
    - Narrow P10-P90 → low risk → safe to trade with tight stops
    - Wide P10-P90 → high risk → need wider stops or smaller position
    - Skewed (P90 >> |P10|) → favorable asymmetry → boost confidence

    Output: p10_return, p50_return, p90_return, risk_ratio (P90/|P10|)
    """

    def __init__(self):
        self.p10_model = None
        self.p50_model = None
        self.p90_model = None
        self.feature_names = None
        self.risk_feature_names = None

    @staticmethod
    def derive_risk_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """Derive risk distribution features."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # ATR (primary risk driver)
        atr_idx = name_to_idx.get('atr_pct')
        if atr_idx is not None:
            feats.append(X[:, atr_idx])
            names.append('risk_atr')

        # Channel widths (wider = more return variance)
        for tf in ['5min', '1h', '4h']:
            w_idx = name_to_idx.get(f'{tf}_width_pct')
            if w_idx is not None:
                feats.append(X[:, w_idx])
                names.append(f'risk_{tf}_width')

        # Channel position (extremes = higher risk of reversal)
        for tf in ['5min', '1h']:
            pos_idx = name_to_idx.get(f'{tf}_position_pct')
            if pos_idx is not None:
                feats.append(X[:, pos_idx])
                names.append(f'risk_{tf}_position')
                feats.append(np.abs(X[:, pos_idx] - 0.5))
                names.append(f'risk_{tf}_edge_dist')

        # OU parameters (half-life predicts mean-reversion speed)
        for tf in ['5min', '1h']:
            for key in ['ou_half_life', 'ou_theta']:
                idx = name_to_idx.get(f'{tf}_{key}')
                if idx is not None:
                    feats.append(X[:, idx])
                    names.append(f'risk_{tf}_{key}')

        # Health (unhealthy = unpredictable)
        for key in ['health_min', 'health_max', 'health_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'risk_{key}')

        # Break probability (high = regime change = tail risk)
        for key in ['break_prob_max', 'break_prob_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'risk_{key}')

        # Volume (high volume = faster resolution, more certain)
        vol_idx = name_to_idx.get('volume_ratio_20')
        if vol_idx is not None:
            feats.append(X[:, vol_idx])
            names.append('risk_volume_ratio')

        # VIX (market-wide risk)
        vix_idx = name_to_idx.get('vix_level')
        if vix_idx is not None:
            feats.append(X[:, vix_idx])
            names.append('risk_vix')

        # Entropy (high = uncertain)
        ent_idx = name_to_idx.get('avg_entropy')
        if ent_idx is not None:
            feats.append(X[:, ent_idx])
            names.append('risk_entropy')

        # Momentum (strong momentum = directional risk)
        for key in ['price_momentum_3bar', 'price_momentum_12bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(np.abs(X[:, idx]))
                names.append(f'risk_abs_{key}')

        # Time of day
        mso_idx = name_to_idx.get('minutes_since_open')
        if mso_idx is not None:
            feats.append(X[:, mso_idx])
            names.append('risk_minutes_since_open')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train quantile risk estimator."""
        import lightgbm as lgb

        self.feature_names = list(feature_names)
        risk_X_train, self.risk_feature_names = self.derive_risk_features(
            X_train, feature_names)
        risk_X_val, _ = self.derive_risk_features(X_val, feature_names)

        ret20 = Y_train['future_return_20'].astype(np.float32)
        ret20_val = Y_val['future_return_20'].astype(np.float32)

        metrics = {}
        print(f"\n  Risk features: {len(self.risk_feature_names)}")
        print(f"  Return range: [{ret20.min():.3f}, {ret20.max():.3f}]")

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]

        # Train 3 quantile regressors
        for alpha, name in [(0.10, 'p10'), (0.50, 'p50'), (0.90, 'p90')]:
            print(f"  Training {name} (alpha={alpha})...")
            dtrain = lgb.Dataset(risk_X_train, label=ret20,
                                feature_name=self.risk_feature_names)
            dval = lgb.Dataset(risk_X_val, label=ret20_val,
                              feature_name=self.risk_feature_names, reference=dtrain)

            model = lgb.train(
                {'objective': 'quantile', 'alpha': alpha, 'metric': 'quantile',
                 'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8,
                 'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1},
                dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

            pred = model.predict(risk_X_val)

            if name == 'p10':
                self.p10_model = model
                # Check calibration: should be ~10% of actuals below prediction
                below_pct = (ret20_val < pred).mean()
                metrics[f'{name}_calibration'] = float(below_pct)
                print(f"    {name} calibration: {below_pct:.1%} below (target: 10%)")
            elif name == 'p50':
                self.p50_model = model
                below_pct = (ret20_val < pred).mean()
                metrics[f'{name}_calibration'] = float(below_pct)
                print(f"    {name} calibration: {below_pct:.1%} below (target: 50%)")
            else:
                self.p90_model = model
                below_pct = (ret20_val < pred).mean()
                metrics[f'{name}_calibration'] = float(below_pct)
                print(f"    {name} calibration: {below_pct:.1%} below (target: 90%)")

        # Compute risk ratio and check if it predicts good trades
        p10_pred = self.p10_model.predict(risk_X_val)
        p90_pred = self.p90_model.predict(risk_X_val)

        # Favorable asymmetry: P90 > |P10|
        safe_p10 = np.where(np.abs(p10_pred) > 1e-6, np.abs(p10_pred), 1e-6)
        risk_ratio = p90_pred / safe_p10
        metrics['avg_risk_ratio'] = float(risk_ratio.mean())
        print(f"\n  Avg risk ratio (P90/|P10|): {metrics['avg_risk_ratio']:.2f}")

        # Range spread
        spread = p90_pred - p10_pred
        metrics['avg_spread'] = float(spread.mean())
        print(f"  Avg P10-P90 spread: {metrics['avg_spread']:.4f}")

        # Does wide spread predict actual large moves?
        actual_abs = np.abs(ret20_val)
        valid = np.isfinite(spread) & np.isfinite(actual_abs)
        if valid.sum() > 10:
            metrics['spread_corr'] = float(np.corrcoef(spread[valid], actual_abs[valid])[0, 1])
            print(f"  Spread vs actual |ret|: corr={metrics['spread_corr']:.3f}")

        imp = self.p90_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 risk features (P90 model):")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.risk_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.p10_model is None:
            return {}
        risk_X, _ = self.derive_risk_features(X, self.feature_names)
        p10 = self.p10_model.predict(risk_X)
        p50 = self.p50_model.predict(risk_X)
        p90 = self.p90_model.predict(risk_X)
        safe_p10 = np.where(np.abs(p10) > 1e-6, np.abs(p10), 1e-6)
        return {
            'p10_return': p10,
            'p50_return': p50,
            'p90_return': p90,
            'risk_ratio': p90 / safe_p10,
            'return_spread': p90 - p10,
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'p10_model': self.p10_model, 'p50_model': self.p50_model,
                'p90_model': self.p90_model,
                'feature_names': self.feature_names,
                'risk_feature_names': self.risk_feature_names,
            }, f)
        print(f"  Saved QuantileRiskEstimator to {path}")

    @classmethod
    def load(cls, path: str) -> 'QuantileRiskEstimator':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.p10_model = data['p10_model']
        model.p50_model = data['p50_model']
        model.p90_model = data['p90_model']
        model.feature_names = data['feature_names']
        model.risk_feature_names = data['risk_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 41: Tail Risk Detector
# ---------------------------------------------------------------------------

class TailRiskDetector:
    """
    Predicts the probability of EXTREME adverse moves (tail events).

    Most models predict "good" trades. This predicts "catastrophically bad" ones.
    A 3% adverse move in 20 bars on TSLA can wipe out multiple winners. If we
    can predict when these tail events are likely, we can avoid them entirely.

    Key insight: Tail events have different drivers than normal moves —
    they're driven by gap risk, volatility clustering, and regime changes.

    Output: tail_risk_prob (>0.5 = dangerous, avoid or reduce size)
    """

    def __init__(self):
        self.tail_model = None
        self.feature_names = None
        self.tail_feature_names = None

    @staticmethod
    def derive_tail_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """Derive tail risk features."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Volatility features (high vol = tail risk)
        atr_idx = name_to_idx.get('atr_pct')
        if atr_idx is not None:
            feats.append(X[:, atr_idx])
            names.append('tail_atr')
            # Squared ATR (tail risk grows quadratically with vol)
            feats.append(X[:, atr_idx] ** 2)
            names.append('tail_atr_squared')

        # Channel width (wider = more room for adverse move)
        for tf in ['5min', '1h', '4h']:
            w_idx = name_to_idx.get(f'{tf}_width_pct')
            if w_idx is not None:
                feats.append(X[:, w_idx])
                names.append(f'tail_{tf}_width')

        # Position at extremes (near edge = vulnerable)
        for tf in ['5min', '1h']:
            pos_idx = name_to_idx.get(f'{tf}_position_pct')
            if pos_idx is not None:
                edge = np.abs(X[:, pos_idx] - 0.5)
                feats.append(edge)
                names.append(f'tail_{tf}_edge_dist')

        # Break probability (high = regime change imminent)
        for key in ['break_prob_max', 'break_prob_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'tail_{key}')

        # Health (deteriorating health = structural instability)
        for key in ['health_min', 'health_delta_3bar', 'health_delta_6bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'tail_{key}')

        # Entropy (high = unpredictable)
        for key in ['avg_entropy', 'entropy_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'tail_{key}')

        # VIX (market fear)
        vix_idx = name_to_idx.get('vix_level')
        if vix_idx is not None:
            feats.append(X[:, vix_idx])
            names.append('tail_vix')

        # Time of day (opening/closing = volatile)
        mso_idx = name_to_idx.get('minutes_since_open')
        if mso_idx is not None:
            feats.append(X[:, mso_idx])
            names.append('tail_minutes_since_open')
            # Near-close risk
            feats.append((X[:, mso_idx] > 360).astype(np.float32))
            names.append('tail_near_close')

        # OU parameters (fast reversion = less tail risk)
        for tf in ['5min', '1h']:
            theta_idx = name_to_idx.get(f'{tf}_ou_theta')
            if theta_idx is not None:
                feats.append(X[:, theta_idx])
                names.append(f'tail_{tf}_ou_theta')

        # Volume (low volume = gap risk)
        vol_idx = name_to_idx.get('volume_ratio_20')
        if vol_idx is not None:
            feats.append(X[:, vol_idx])
            names.append('tail_volume_ratio')
            feats.append((X[:, vol_idx] < 0.5).astype(np.float32))
            names.append('tail_low_volume')

        # RSI extremes
        rsi_idx = name_to_idx.get('rsi_14')
        if rsi_idx is not None:
            feats.append(np.abs(X[:, rsi_idx] - 50))
            names.append('tail_rsi_extreme')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train tail risk detector."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        tail_X_train, self.tail_feature_names = self.derive_tail_features(
            X_train, feature_names)
        tail_X_val, _ = self.derive_tail_features(X_val, feature_names)

        ret20 = Y_train['future_return_20']
        ret20_val = Y_val['future_return_20']

        # Tail event: adverse move > 2% in 20 bars
        # (direction-agnostic — large |ret20| regardless of sign)
        tail_train = (np.abs(ret20) > 0.02).astype(np.float32)
        tail_val = (np.abs(ret20_val) > 0.02).astype(np.float32)

        metrics = {}
        print(f"\n  Tail risk features: {len(self.tail_feature_names)}")
        print(f"  Tail event rate (|ret20| > 2%): {tail_train.mean():.1%}")

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(tail_X_train, label=tail_train,
                            feature_name=self.tail_feature_names)
        dval = lgb.Dataset(tail_X_val, label=tail_val,
                          feature_name=self.tail_feature_names, reference=dtrain)

        self.tail_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'is_unbalance': True},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        tail_pred = self.tail_model.predict(tail_X_val)
        try:
            metrics['tail_auc'] = float(roc_auc_score(tail_val, tail_pred))
        except ValueError:
            metrics['tail_auc'] = 0.5
        print(f"    Tail Risk AUC: {metrics['tail_auc']:.3f}")

        imp = self.tail_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 tail risk features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.tail_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.tail_model is None:
            return {}
        tail_X, _ = self.derive_tail_features(X, self.feature_names)
        return {
            'tail_risk_prob': self.tail_model.predict(tail_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'tail_model': self.tail_model,
                'feature_names': self.feature_names,
                'tail_feature_names': self.tail_feature_names,
            }, f)
        print(f"  Saved TailRiskDetector to {path}")

    @classmethod
    def load(cls, path: str) -> 'TailRiskDetector':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.tail_model = data['tail_model']
        model.feature_names = data['feature_names']
        model.tail_feature_names = data['tail_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 42: Drawdown Recovery Predictor
# ---------------------------------------------------------------------------

class DrawdownRecoveryPredictor:
    """
    Predicts whether a trade that initially goes against us will recover.

    The physics engine generates signals where price bounces off channel edges.
    Some bounces initially go wrong (drawdown) but recover. Others are genuine
    failures. If we can distinguish these, we can:
    - Hold through recoverable drawdowns instead of stopping out
    - Exit faster on genuine failures

    Uses the relationship between ret5 and ret20: if ret5 is negative but ret20
    is positive, the trade "recovers." This is a distribution property, not direction.

    Output: recovery_prob (>0.6 = likely to recover from initial drawdown)
    """

    def __init__(self):
        self.recovery_model = None
        self.feature_names = None
        self.rec_feature_names = None

    @staticmethod
    def derive_recovery_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """Derive recovery prediction features."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Mean reversion strength (strong = recovery likely)
        for tf in ['5min', '1h']:
            for key in ['ou_theta', 'ou_half_life', 'ou_reversion_score']:
                idx = name_to_idx.get(f'{tf}_{key}')
                if idx is not None:
                    feats.append(X[:, idx])
                    names.append(f'rec_{tf}_{key}')

        # Channel health (healthy = price respects boundaries = recovery)
        for key in ['health_min', 'health_max', 'health_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'rec_{key}')

        # R-squared (high = channel is valid = bounces work)
        for tf in ['5min', '1h', '4h']:
            idx = name_to_idx.get(f'{tf}_r_squared')
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'rec_{tf}_rsq')

        # Position in channel (near edge = bounce expected)
        for tf in ['5min', '1h']:
            pos_idx = name_to_idx.get(f'{tf}_position_pct')
            if pos_idx is not None:
                feats.append(X[:, pos_idx])
                names.append(f'rec_{tf}_position')
                feats.append(np.abs(X[:, pos_idx] - 0.5))
                names.append(f'rec_{tf}_edge_dist')

        # Center distance (far from center = stronger reversion force)
        for tf in ['5min', '1h']:
            cd_idx = name_to_idx.get(f'{tf}_center_distance')
            if cd_idx is not None:
                feats.append(X[:, cd_idx])
                names.append(f'rec_{tf}_center_dist')

        # Bounce count (more bounces = channel holds)
        for tf in ['5min', '1h']:
            bc_idx = name_to_idx.get(f'{tf}_bounce_count')
            if bc_idx is not None:
                feats.append(X[:, bc_idx])
                names.append(f'rec_{tf}_bounce_count')

        # Break probability (low = channel will hold = recovery)
        for key in ['break_prob_max', 'break_prob_weighted']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'rec_{key}')

        # Volume (high = more conviction for recovery)
        vol_idx = name_to_idx.get('volume_ratio_20')
        if vol_idx is not None:
            feats.append(X[:, vol_idx])
            names.append('rec_volume_ratio')

        # ATR
        atr_idx = name_to_idx.get('atr_pct')
        if atr_idx is not None:
            feats.append(X[:, atr_idx])
            names.append('rec_atr')

        # Width (narrow = quick recovery, wide = slow)
        for tf in ['5min', '1h']:
            w_idx = name_to_idx.get(f'{tf}_width_pct')
            if w_idx is not None:
                feats.append(X[:, w_idx])
                names.append(f'rec_{tf}_width')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train drawdown recovery predictor."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        rec_X_train, self.rec_feature_names = self.derive_recovery_features(
            X_train, feature_names)
        rec_X_val, _ = self.derive_recovery_features(X_val, feature_names)

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']

        # "Recovery" = initial drawdown but ultimate profit
        # ret5 < 0 (went against us) but ret20 > 0 (recovered)
        # Also include: went right, then MORE right (sustained)
        recovery_train = (
            ((ret5 < -0.001) & (ret20 > 0.001)) |  # Recovered from drawdown
            ((ret5 > 0.001) & (ret20 > ret5 * 1.5))   # Sustained and grew
        ).astype(np.float32)
        recovery_val = (
            ((ret5_val < -0.001) & (ret20_val > 0.001)) |
            ((ret5_val > 0.001) & (ret20_val > ret5_val * 1.5))
        ).astype(np.float32)

        metrics = {}
        print(f"\n  Recovery features: {len(self.rec_feature_names)}")
        print(f"  Recovery rate: {recovery_train.mean():.1%}")

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(rec_X_train, label=recovery_train,
                            feature_name=self.rec_feature_names)
        dval = lgb.Dataset(rec_X_val, label=recovery_val,
                          feature_name=self.rec_feature_names, reference=dtrain)

        self.recovery_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'is_unbalance': True},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        rec_pred = self.recovery_model.predict(rec_X_val)
        try:
            metrics['recovery_auc'] = float(roc_auc_score(recovery_val, rec_pred))
        except ValueError:
            metrics['recovery_auc'] = 0.5
        print(f"    Recovery AUC: {metrics['recovery_auc']:.3f}")

        imp = self.recovery_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 recovery features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.rec_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.recovery_model is None:
            return {}
        rec_X, _ = self.derive_recovery_features(X, self.feature_names)
        return {
            'recovery_prob': self.recovery_model.predict(rec_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'recovery_model': self.recovery_model,
                'feature_names': self.feature_names,
                'rec_feature_names': self.rec_feature_names,
            }, f)
        print(f"  Saved DrawdownRecoveryPredictor to {path}")

    @classmethod
    def load(cls, path: str) -> 'DrawdownRecoveryPredictor':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.recovery_model = data['recovery_model']
        model.feature_names = data['feature_names']
        model.rec_feature_names = data['rec_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 43: Stop Distance Optimizer
# ---------------------------------------------------------------------------

class StopDistanceOptimizer:
    """
    Predicts the optimal stop loss distance for each trade setup.

    Instead of a fixed stop % for all trades, this model predicts how much
    "noise" to expect in the first 5-10 bars. A tight stop works in quiet
    conditions but whipsaws in volatile ones. A wide stop loses too much
    on genuine failures.

    Target: the minimum adverse excursion (MAE) that a winning trade
    experiences. If MAE < 0.3%, we can use a 0.4% stop. If MAE > 1%, we
    need a 1.2% stop.

    Output: predicted_mae (expected adverse move before recovery)
    """

    def __init__(self):
        self.mae_model = None
        self.feature_names = None
        self.stop_feature_names = None

    @staticmethod
    def derive_stop_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """Derive stop distance features."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # ATR (primary stop distance driver)
        atr_idx = name_to_idx.get('atr_pct')
        if atr_idx is not None:
            feats.append(X[:, atr_idx])
            names.append('stop_atr')

        # Channel widths (wider = more noise = wider stop needed)
        for tf in ['5min', '1h', '4h']:
            w_idx = name_to_idx.get(f'{tf}_width_pct')
            if w_idx is not None:
                feats.append(X[:, w_idx])
                names.append(f'stop_{tf}_width')

        # OU parameters (fast reversion = tighter stops OK)
        for tf in ['5min', '1h']:
            for key in ['ou_half_life', 'ou_theta']:
                idx = name_to_idx.get(f'{tf}_{key}')
                if idx is not None:
                    feats.append(X[:, idx])
                    names.append(f'stop_{tf}_{key}')

        # Position in channel
        for tf in ['5min', '1h']:
            pos_idx = name_to_idx.get(f'{tf}_position_pct')
            if pos_idx is not None:
                feats.append(X[:, pos_idx])
                names.append(f'stop_{tf}_position')
                feats.append(np.abs(X[:, pos_idx] - 0.5))
                names.append(f'stop_{tf}_edge_dist')

        # Health (healthy = less noise)
        for key in ['health_min', 'health_max']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'stop_{key}')

        # Volume (high volume = more noise in absolute terms)
        vol_idx = name_to_idx.get('volume_ratio_20')
        if vol_idx is not None:
            feats.append(X[:, vol_idx])
            names.append('stop_volume_ratio')

        # VIX
        vix_idx = name_to_idx.get('vix_level')
        if vix_idx is not None:
            feats.append(X[:, vix_idx])
            names.append('stop_vix')

        # Time of day (opening = more noise)
        mso_idx = name_to_idx.get('minutes_since_open')
        if mso_idx is not None:
            feats.append(X[:, mso_idx])
            names.append('stop_minutes_since_open')

        # Break probability
        bp_idx = name_to_idx.get('break_prob_max')
        if bp_idx is not None:
            feats.append(X[:, bp_idx])
            names.append('stop_break_prob')

        # Entropy
        ent_idx = name_to_idx.get('avg_entropy')
        if ent_idx is not None:
            feats.append(X[:, ent_idx])
            names.append('stop_entropy')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train stop distance optimizer."""
        import lightgbm as lgb

        self.feature_names = list(feature_names)
        stop_X_train, self.stop_feature_names = self.derive_stop_features(
            X_train, feature_names)
        stop_X_val, _ = self.derive_stop_features(X_val, feature_names)

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']

        # MAE proxy: the worst drawdown in the first window
        # For a long trade: MAE ≈ min(ret5, 0) if ret5 < 0
        # Use absolute value: how far against us did it go?
        mae_train = np.abs(np.minimum(ret5, 0)).astype(np.float32)
        mae_val = np.abs(np.minimum(ret5_val, 0)).astype(np.float32)

        metrics = {}
        print(f"\n  Stop features: {len(self.stop_feature_names)}")
        print(f"  Mean MAE: {mae_train.mean():.4f} ({mae_train.mean()*100:.2f}%)")
        print(f"  MAE > 0.5%: {(mae_train > 0.005).mean():.1%}")

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(stop_X_train, label=mae_train,
                            feature_name=self.stop_feature_names)
        dval = lgb.Dataset(stop_X_val, label=mae_val,
                          feature_name=self.stop_feature_names, reference=dtrain)

        self.mae_model = lgb.train(
            {'objective': 'regression', 'metric': 'mae', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        mae_pred = self.mae_model.predict(stop_X_val)
        metrics['mae_mae'] = float(np.mean(np.abs(mae_pred - mae_val)))
        valid = np.isfinite(mae_pred) & np.isfinite(mae_val)
        if valid.sum() > 10:
            metrics['mae_corr'] = float(np.corrcoef(mae_pred[valid], mae_val[valid])[0, 1])
        else:
            metrics['mae_corr'] = 0
        print(f"    MAE prediction MAE: {metrics['mae_mae']:.4f}")
        print(f"    MAE prediction Corr: {metrics['mae_corr']:.3f}")

        # Check: does predicted MAE separate good/bad situations?
        from sklearn.metrics import roc_auc_score
        high_mae = (mae_val > 0.005).astype(np.float32)
        try:
            metrics['high_mae_auc'] = float(roc_auc_score(high_mae, mae_pred))
        except ValueError:
            metrics['high_mae_auc'] = 0.5
        print(f"    High MAE AUC (>0.5%): {metrics['high_mae_auc']:.3f}")

        imp = self.mae_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 stop features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.stop_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.mae_model is None:
            return {}
        stop_X, _ = self.derive_stop_features(X, self.feature_names)
        return {
            'predicted_mae': np.clip(self.mae_model.predict(stop_X), 0, 0.05),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'mae_model': self.mae_model,
                'feature_names': self.feature_names,
                'stop_feature_names': self.stop_feature_names,
            }, f)
        print(f"  Saved StopDistanceOptimizer to {path}")

    @classmethod
    def load(cls, path: str) -> 'StopDistanceOptimizer':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.mae_model = data['mae_model']
        model.feature_names = data['feature_names']
        model.stop_feature_names = data['stop_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 44: Volatility Clustering Predictor
# ---------------------------------------------------------------------------

class VolatilityClusteringPredictor:
    """
    Predicts whether current volatility will INCREASE or DECREASE in the
    next 20 bars. Volatility clustering (GARCH effect) is one of the most
    robust stylized facts in finance — high vol begets high vol.

    Unlike the Vol Transition model (Arch 14, which predicted vol SPIKES),
    this predicts the DIRECTION of vol change. This affects:
    - Position sizing (increasing vol → reduce size)
    - Stop distance (increasing vol → wider stops)
    - Trade urgency (decreasing vol → act now before breakout)

    Output: vol_increase_prob (>0.6 = vol will increase)
    """

    def __init__(self):
        self.vol_model = None
        self.feature_names = None
        self.vc_feature_names = None

    @staticmethod
    def derive_vc_features(X: np.ndarray, feature_names: list) -> Tuple[np.ndarray, list]:
        """Derive vol clustering features."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Current ATR
        atr_idx = name_to_idx.get('atr_pct')
        if atr_idx is not None:
            feats.append(X[:, atr_idx])
            names.append('vc_atr')

        # Channel widths (proxy for realized vol)
        for tf in ['5min', '1h', '4h']:
            w_idx = name_to_idx.get(f'{tf}_width_pct')
            if w_idx is not None:
                feats.append(X[:, w_idx])
                names.append(f'vc_{tf}_width')

        # Width/ATR ratio (how width compares to recent vol)
        for tf in ['5min', '1h']:
            w_idx = name_to_idx.get(f'{tf}_width_pct')
            if w_idx is not None and atr_idx is not None:
                safe_atr = np.where(X[:, atr_idx] > 1e-6, X[:, atr_idx], 1e-6)
                feats.append(X[:, w_idx] / safe_atr)
                names.append(f'vc_{tf}_width_per_atr')

        # VIX (market-wide vol expectation)
        vix_idx = name_to_idx.get('vix_level')
        if vix_idx is not None:
            feats.append(X[:, vix_idx])
            names.append('vc_vix')

        # Momentum (strong momentum → vol continuation)
        for key in ['price_momentum_3bar', 'price_momentum_12bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(np.abs(X[:, idx]))
                names.append(f'vc_abs_{key}')

        # Volume (high volume = vol continuation)
        vol_idx = name_to_idx.get('volume_ratio_20')
        if vol_idx is not None:
            feats.append(X[:, vol_idx])
            names.append('vc_volume_ratio')

        # Entropy (high entropy = uncertain, vol likely to persist)
        for key in ['avg_entropy', 'entropy_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'vc_{key}')

        # Health dynamics (deteriorating → vol increasing)
        for key in ['health_delta_3bar', 'health_delta_6bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'vc_{key}')

        # Break probability dynamics
        bp_delta = name_to_idx.get('break_prob_delta_3bar')
        if bp_delta is not None:
            feats.append(X[:, bp_delta])
            names.append('vc_break_prob_delta')

        # RSI slope (rapid RSI changes = vol)
        rsi_slope = name_to_idx.get('rsi_slope_5bar')
        if rsi_slope is not None:
            feats.append(np.abs(X[:, rsi_slope]))
            names.append('vc_abs_rsi_slope')

        # Time of day
        mso_idx = name_to_idx.get('minutes_since_open')
        if mso_idx is not None:
            feats.append(X[:, mso_idx])
            names.append('vc_minutes_since_open')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train vol clustering predictor."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        vc_X_train, self.vc_feature_names = self.derive_vc_features(
            X_train, feature_names)
        vc_X_val, _ = self.derive_vc_features(X_val, feature_names)

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']

        # Vol increase: |ret20 - ret5| > |ret5|
        # (the 5-20 bar window has more volatility than the 0-5 bar window)
        vol5_train = np.abs(ret5)
        vol20_train = np.abs(ret20 - ret5)
        vol_increase_train = (vol20_train > vol5_train * 1.3).astype(np.float32)

        vol5_val = np.abs(ret5_val)
        vol20_val = np.abs(ret20_val - ret5_val)
        vol_increase_val = (vol20_val > vol5_val * 1.3).astype(np.float32)

        metrics = {}
        print(f"\n  Vol clustering features: {len(self.vc_feature_names)}")
        print(f"  Vol increase rate: {vol_increase_train.mean():.1%}")

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(vc_X_train, label=vol_increase_train,
                            feature_name=self.vc_feature_names)
        dval = lgb.Dataset(vc_X_val, label=vol_increase_val,
                          feature_name=self.vc_feature_names, reference=dtrain)

        self.vol_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'is_unbalance': True},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        vol_pred = self.vol_model.predict(vc_X_val)
        try:
            metrics['vol_increase_auc'] = float(roc_auc_score(vol_increase_val, vol_pred))
        except ValueError:
            metrics['vol_increase_auc'] = 0.5
        print(f"    Vol Increase AUC: {metrics['vol_increase_auc']:.3f}")

        imp = self.vol_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 vol clustering features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.vc_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.vol_model is None:
            return {}
        vc_X, _ = self.derive_vc_features(X, self.feature_names)
        return {
            'vol_increase_prob': self.vol_model.predict(vc_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'vol_model': self.vol_model,
                'feature_names': self.feature_names,
                'vc_feature_names': self.vc_feature_names,
            }, f)
        print(f"  Saved VolatilityClusteringPredictor to {path}")

    @classmethod
    def load(cls, path: str) -> 'VolatilityClusteringPredictor':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.vol_model = data['vol_model']
        model.feature_names = data['feature_names']
        model.vc_feature_names = data['vc_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 45: Extreme Loser Avoidance
# ---------------------------------------------------------------------------
# Target: Is this trade in the bottom 15% of returns? (ret_20 < threshold)
# The highest-value prediction: reliably identifying trades to SKIP.
# Even a small reduction in losers dramatically improves PF.

class ExtremeLoserDetector:
    """Predict if a trade will be an extreme loser (bottom 15% of returns)."""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.el_feature_names = None

    def derive_el_features(self, X, feature_names):
        """Extract features most relevant to identifying losers."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Channel health indicators (low health → more losers)
        for tf in ML_TFS:
            for key in ['health', 'width_pct', 'r_squared', 'break_prob',
                        'position_in_channel', 'bars_since_touch']:
                idx = name_to_idx.get(f'{tf}_{key}')
                if idx is not None:
                    feats.append(X[:, idx])
                    names.append(f'el_{tf}_{key}')

        # Cross-TF disagreement (conflicting signals → losers)
        for key in ['direction_agreement', 'avg_break_prob', 'avg_health',
                    'health_min', 'health_std', 'width_dispersion',
                    'channel_count_ratio']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'el_{key}')

        # Context features
        for key in ['rsi_14', 'atr_pct', 'volume_ratio_20', 'bar_range_pct',
                    'upper_wick_pct', 'lower_wick_pct', 'body_pct',
                    'minutes_since_open', 'day_of_week']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'el_{key}')

        # Temporal dynamics
        for key in ['avg_entropy', 'entropy_delta_3bar', 'health_delta_3bar',
                    'health_delta_6bar', 'break_prob_delta_3bar',
                    'rsi_slope_5bar', 'vol_trend_5bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'el_{key}')

        # Derived: health × break_prob interaction
        h_idx = name_to_idx.get('health_min')
        bp_idx = name_to_idx.get('avg_break_prob')
        if h_idx is not None and bp_idx is not None:
            feats.append(X[:, h_idx] * X[:, bp_idx])
            names.append('el_health_x_breakprob')

        # Derived: position extremity (how close to boundary)
        for tf in ML_TFS:
            pos_idx = name_to_idx.get(f'{tf}_position_in_channel')
            if pos_idx is not None:
                pos = X[:, pos_idx]
                feats.append(np.abs(pos - 0.5))  # Distance from center
                names.append(f'el_{tf}_boundary_proximity')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train extreme loser detector."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        el_X_train, self.el_feature_names = self.derive_el_features(
            X_train, feature_names)
        el_X_val, _ = self.derive_el_features(X_val, feature_names)

        ret20_train = Y_train['future_return_20']
        ret20_val = Y_val['future_return_20']

        # Bottom 15% of returns = extreme losers
        threshold = np.percentile(ret20_train, 15)
        loser_train = (ret20_train < threshold).astype(np.float32)
        loser_val = (ret20_val < threshold).astype(np.float32)

        metrics = {}
        print(f"\n  Extreme loser features: {len(self.el_feature_names)}")
        print(f"  Loser threshold: {threshold:.4f} ({threshold*100:.2f}%)")
        print(f"  Loser rate (train): {loser_train.mean():.1%}")
        print(f"  Loser rate (val): {loser_val.mean():.1%}")

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(el_X_train, label=loser_train,
                            feature_name=self.el_feature_names)
        dval = lgb.Dataset(el_X_val, label=loser_val,
                          feature_name=self.el_feature_names, reference=dtrain)

        self.model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1,
             'scale_pos_weight': 3.0},  # Emphasize losers
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        loser_pred = self.model.predict(el_X_val)
        try:
            auc = roc_auc_score(loser_val, loser_pred)
            metrics['loser_auc'] = float(auc)
            print(f"  Extreme Loser AUC: {auc:.3f}")
        except Exception:
            metrics['loser_auc'] = 0.5

        # Check precision at high-confidence threshold
        for thr in [0.3, 0.4, 0.5]:
            flagged = loser_pred > thr
            if flagged.sum() > 0:
                precision = loser_val[flagged].mean()
                print(f"    Threshold {thr:.1f}: {flagged.sum()} flagged, {precision:.1%} precision")

        imp = self.model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 extreme loser features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.el_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.model is None:
            return {}
        el_X, _ = self.derive_el_features(X, self.feature_names)
        return {
            'loser_prob': self.model.predict(el_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'el_feature_names': self.el_feature_names,
            }, f)
        print(f"  Saved ExtremeLoserDetector to {path}")

    @classmethod
    def load(cls, path: str) -> 'ExtremeLoserDetector':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.model = data['model']
        model.feature_names = data['feature_names']
        model.el_feature_names = data['el_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 46: Risk-Reward Classifier
# ---------------------------------------------------------------------------
# Target: Does this trade have a favorable risk-reward ratio?
# Uses max adverse excursion (approximated from ret5/ret20) vs final return.
# High R:R trades are worth boosting position size on.

class RiskRewardClassifier:
    """Predict if a trade will have favorable risk-reward (R:R > 2.0)."""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.rr_feature_names = None

    def derive_rr_features(self, X, feature_names):
        """Extract features relevant to risk-reward assessment."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Channel width (narrow channels → better R:R typically)
        for tf in ML_TFS:
            for key in ['width_pct', 'position_in_channel', 'health',
                        'r_squared', 'slope_normalized', 'break_prob']:
                idx = name_to_idx.get(f'{tf}_{key}')
                if idx is not None:
                    feats.append(X[:, idx])
                    names.append(f'rr_{tf}_{key}')

        # Cross-TF features
        for key in ['direction_agreement', 'avg_health', 'health_min',
                    'avg_break_prob', 'width_dispersion']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'rr_{key}')

        # Context (ATR, RSI for volatility assessment)
        for key in ['rsi_14', 'atr_pct', 'volume_ratio_20', 'bar_range_pct',
                    'minutes_since_open']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'rr_{key}')

        # Temporal dynamics
        for key in ['health_delta_3bar', 'break_prob_delta_3bar',
                    'vol_trend_5bar', 'rsi_slope_5bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'rr_{key}')

        # Correlation features
        for key in ['tsla_spy_corr_20', 'tsla_spy_corr_60', 'vix_level']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'rr_{key}')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train risk-reward classifier."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        rr_X_train, self.rr_feature_names = self.derive_rr_features(
            X_train, feature_names)
        rr_X_val, _ = self.derive_rr_features(X_val, feature_names)

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']

        # Approximate max adverse excursion:
        # If ret20 > 0 (winner), adverse = max(-ret5, 0) (worst drawdown early)
        # If ret20 < 0 (loser), adverse = abs(ret20) (the loss itself)
        # Risk-reward = |ret20| / (adverse + 0.001)
        def compute_rr(r5, r20):
            adverse = np.where(
                r20 > 0,
                np.maximum(-np.minimum(r5, 0), 0.0005),  # Winners: worst early drawdown
                np.abs(r20) + 0.001  # Losers: the loss is the adverse
            )
            rr = np.abs(r20) / (adverse + 0.001)
            # High R:R = reward > 2x the risk
            return (rr > 2.0).astype(np.float32)

        high_rr_train = compute_rr(ret5, ret20)
        high_rr_val = compute_rr(ret5_val, ret20_val)

        metrics = {}
        print(f"\n  Risk-reward features: {len(self.rr_feature_names)}")
        print(f"  High R:R rate (train): {high_rr_train.mean():.1%}")
        print(f"  High R:R rate (val): {high_rr_val.mean():.1%}")

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(rr_X_train, label=high_rr_train,
                            feature_name=self.rr_feature_names)
        dval = lgb.Dataset(rr_X_val, label=high_rr_val,
                          feature_name=self.rr_feature_names, reference=dtrain)

        self.model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'is_unbalance': True},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        rr_pred = self.model.predict(rr_X_val)
        try:
            auc = roc_auc_score(high_rr_val, rr_pred)
            metrics['rr_auc'] = float(auc)
            print(f"  Risk-Reward AUC: {auc:.3f}")
        except Exception:
            metrics['rr_auc'] = 0.5

        # Correlation with actual R:R
        actual_rr = np.abs(ret20_val) / (np.maximum(np.abs(np.minimum(ret5_val, 0)), 0.0005) + 0.001)
        corr = np.corrcoef(rr_pred, actual_rr)[0, 1]
        metrics['rr_corr'] = float(corr) if not np.isnan(corr) else 0.0
        print(f"  R:R correlation: {corr:.3f}")

        imp = self.model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 risk-reward features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.rr_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.model is None:
            return {}
        rr_X, _ = self.derive_rr_features(X, self.feature_names)
        return {
            'high_rr_prob': self.model.predict(rr_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'rr_feature_names': self.rr_feature_names,
            }, f)
        print(f"  Saved RiskRewardClassifier to {path}")

    @classmethod
    def load(cls, path: str) -> 'RiskRewardClassifier':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.model = data['model']
        model.feature_names = data['feature_names']
        model.rr_feature_names = data['rr_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 47: Return Consistency Predictor
# ---------------------------------------------------------------------------
# Target: Do short and medium returns agree? sign(ret5) == sign(ret20) AND |ret20| > |ret5|
# "Consistent" trades build steadily — these are the easiest to manage.
# Inconsistent trades (early win that reverses, or late reversal) are harder.

class ReturnConsistencyPredictor:
    """Predict if returns will be consistent across horizons."""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.rc_feature_names = None

    def derive_rc_features(self, X, feature_names):
        """Extract features relevant to return consistency."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Channel slope consistency across TFs
        slopes = []
        for tf in ML_TFS:
            s_idx = name_to_idx.get(f'{tf}_slope_normalized')
            if s_idx is not None:
                feats.append(X[:, s_idx])
                names.append(f'rc_{tf}_slope')
                slopes.append(X[:, s_idx])

        # Slope agreement (std across TFs — low = consistent)
        if len(slopes) >= 2:
            slope_stack = np.column_stack(slopes)
            feats.append(np.std(slope_stack, axis=1))
            names.append('rc_slope_std')
            feats.append(np.mean(np.sign(slope_stack), axis=1))
            names.append('rc_slope_sign_avg')

        # Health and break prob (stable channels = consistent returns)
        for tf in ML_TFS:
            for key in ['health', 'r_squared', 'break_prob',
                        'position_in_channel', 'width_pct']:
                idx = name_to_idx.get(f'{tf}_{key}')
                if idx is not None:
                    feats.append(X[:, idx])
                    names.append(f'rc_{tf}_{key}')

        # Cross-TF agreement
        for key in ['direction_agreement', 'avg_health', 'health_std',
                    'avg_break_prob']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'rc_{key}')

        # Temporal stability (consistent price action = consistent returns)
        for key in ['health_delta_3bar', 'health_delta_6bar',
                    'break_prob_delta_3bar', 'entropy_delta_3bar',
                    'rsi_slope_5bar', 'vol_trend_5bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'rc_{key}')

        # Context
        for key in ['rsi_14', 'atr_pct', 'minutes_since_open']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'rc_{key}')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train return consistency predictor."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        rc_X_train, self.rc_feature_names = self.derive_rc_features(
            X_train, feature_names)
        rc_X_val, _ = self.derive_rc_features(X_val, feature_names)

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']

        # Consistent = same sign AND magnitude grows (steady trend)
        consistent_train = (
            (np.sign(ret5) == np.sign(ret20)) &
            (np.abs(ret20) > np.abs(ret5))
        ).astype(np.float32)
        consistent_val = (
            (np.sign(ret5_val) == np.sign(ret20_val)) &
            (np.abs(ret20_val) > np.abs(ret5_val))
        ).astype(np.float32)

        metrics = {}
        print(f"\n  Return consistency features: {len(self.rc_feature_names)}")
        print(f"  Consistent rate (train): {consistent_train.mean():.1%}")
        print(f"  Consistent rate (val): {consistent_val.mean():.1%}")

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(rc_X_train, label=consistent_train,
                            feature_name=self.rc_feature_names)
        dval = lgb.Dataset(rc_X_val, label=consistent_val,
                          feature_name=self.rc_feature_names, reference=dtrain)

        self.model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        rc_pred = self.model.predict(rc_X_val)
        try:
            auc = roc_auc_score(consistent_val, rc_pred)
            metrics['consistency_auc'] = float(auc)
            print(f"  Consistency AUC: {auc:.3f}")
        except Exception:
            metrics['consistency_auc'] = 0.5

        imp = self.model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 consistency features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.rc_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.model is None:
            return {}
        rc_X, _ = self.derive_rc_features(X, self.feature_names)
        return {
            'consistency_prob': self.model.predict(rc_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'rc_feature_names': self.rc_feature_names,
            }, f)
        print(f"  Saved ReturnConsistencyPredictor to {path}")

    @classmethod
    def load(cls, path: str) -> 'ReturnConsistencyPredictor':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.model = data['model']
        model.feature_names = data['feature_names']
        model.rc_feature_names = data['rc_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 48: Horizon Divergence Predictor
# ---------------------------------------------------------------------------
# Target: Do ret_5 and ret_60 disagree in sign? If so, there's a reversal
# coming within 60 bars. This is an early warning to tighten exits.
# Also useful: ret_20 > 0 but ret_60 < 0 means "take profits before reversal".

class HorizonDivergencePredictor:
    """Predict if short-term and long-term returns will diverge (reversal ahead)."""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.hd_feature_names = None

    def derive_hd_features(self, X, feature_names):
        """Extract features relevant to horizon divergence."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Channel maturity indicators (mature = more likely to reverse)
        for tf in ML_TFS:
            for key in ['health', 'lifetime_bars', 'break_prob',
                        'slope_normalized', 'width_pct', 'r_squared']:
                idx = name_to_idx.get(f'{tf}_{key}')
                if idx is not None:
                    feats.append(X[:, idx])
                    names.append(f'hd_{tf}_{key}')

        # Cross-TF divergence signals
        for key in ['direction_agreement', 'avg_break_prob', 'avg_health',
                    'health_min', 'health_std', 'width_dispersion']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'hd_{key}')

        # Temporal exhaustion indicators
        for key in ['health_delta_3bar', 'health_delta_6bar',
                    'break_prob_delta_3bar', 'entropy_delta_3bar',
                    'avg_entropy', 'rsi_slope_5bar', 'vol_trend_5bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'hd_{key}')

        # Context
        for key in ['rsi_14', 'atr_pct', 'volume_ratio_20',
                    'minutes_since_open', 'day_of_week']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'hd_{key}')

        # Correlation features (market regime affects divergence)
        for key in ['tsla_spy_corr_20', 'vix_level', 'spy_trend_20']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'hd_{key}')

        # Derived: break_prob acceleration
        bp3 = name_to_idx.get('break_prob_delta_3bar')
        bp6 = name_to_idx.get('health_delta_6bar')
        if bp3 is not None and bp6 is not None:
            feats.append(X[:, bp3] - X[:, bp6])  # Acceleration
            names.append('hd_break_prob_accel')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train horizon divergence predictor."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        hd_X_train, self.hd_feature_names = self.derive_hd_features(
            X_train, feature_names)
        hd_X_val, _ = self.derive_hd_features(X_val, feature_names)

        ret20 = Y_train['future_return_20']
        ret60 = Y_train['future_return_60']
        ret20_val = Y_val['future_return_20']
        ret60_val = Y_val['future_return_60']

        # Divergence: ret20 and ret60 have opposite signs
        # This means "profitable at 20 bars but reverses by 60" or vice versa
        diverge_train = (np.sign(ret20) != np.sign(ret60)).astype(np.float32)
        # Only count meaningful divergence (not just noise around zero)
        meaningful = (np.abs(ret20) > 0.003) | (np.abs(ret60) > 0.003)
        diverge_train = (diverge_train * meaningful).astype(np.float32)

        diverge_val = (np.sign(ret20_val) != np.sign(ret60_val)).astype(np.float32)
        meaningful_val = (np.abs(ret20_val) > 0.003) | (np.abs(ret60_val) > 0.003)
        diverge_val = (diverge_val * meaningful_val).astype(np.float32)

        metrics = {}
        print(f"\n  Horizon divergence features: {len(self.hd_feature_names)}")
        print(f"  Divergence rate (train): {diverge_train.mean():.1%}")
        print(f"  Divergence rate (val): {diverge_val.mean():.1%}")

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(hd_X_train, label=diverge_train,
                            feature_name=self.hd_feature_names)
        dval = lgb.Dataset(hd_X_val, label=diverge_val,
                          feature_name=self.hd_feature_names, reference=dtrain)

        self.model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        hd_pred = self.model.predict(hd_X_val)
        try:
            auc = roc_auc_score(diverge_val, hd_pred)
            metrics['divergence_auc'] = float(auc)
            print(f"  Divergence AUC: {auc:.3f}")
        except Exception:
            metrics['divergence_auc'] = 0.5

        imp = self.model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 horizon divergence features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.hd_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.model is None:
            return {}
        hd_X, _ = self.derive_hd_features(X, self.feature_names)
        return {
            'divergence_prob': self.model.predict(hd_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'hd_feature_names': self.hd_feature_names,
            }, f)
        print(f"  Saved HorizonDivergencePredictor to {path}")

    @classmethod
    def load(cls, path: str) -> 'HorizonDivergencePredictor':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.model = data['model']
        model.feature_names = data['feature_names']
        model.hd_feature_names = data['hd_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 49: Drawdown Magnitude Predictor
# ---------------------------------------------------------------------------
# Unlike binary models, this uses QUANTILE REGRESSION to predict the
# distribution of max adverse excursion. Knowing "this trade will likely
# draw down 0.5-1.0%" is more useful than "bad trade: yes/no".

class DrawdownMagnitudePredictor:
    """Predict the magnitude of adverse price excursion using quantile regression."""

    def __init__(self):
        self.p25_model = None
        self.p50_model = None
        self.p75_model = None
        self.feature_names = None
        self.dm_feature_names = None

    def derive_dm_features(self, X, feature_names):
        """Extract features relevant to drawdown magnitude."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Channel properties across TFs
        for tf in ML_TFS:
            for key in ['width_pct', 'health', 'position_in_channel',
                        'r_squared', 'break_prob', 'bars_since_touch',
                        'slope_normalized']:
                idx = name_to_idx.get(f'{tf}_{key}')
                if idx is not None:
                    feats.append(X[:, idx])
                    names.append(f'dm_{tf}_{key}')

        # Cross-TF
        for key in ['direction_agreement', 'avg_health', 'health_min',
                    'health_std', 'avg_break_prob', 'width_dispersion']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'dm_{key}')

        # Volatility context
        for key in ['atr_pct', 'bar_range_pct', 'volume_ratio_20',
                    'rsi_14', 'minutes_since_open']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'dm_{key}')

        # Temporal dynamics
        for key in ['health_delta_3bar', 'health_delta_6bar',
                    'break_prob_delta_3bar', 'vol_trend_5bar',
                    'avg_entropy', 'entropy_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'dm_{key}')

        # Correlation features
        for key in ['vix_level', 'tsla_spy_corr_20']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'dm_{key}')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train drawdown magnitude predictor using quantile regression."""
        import lightgbm as lgb

        self.feature_names = list(feature_names)
        dm_X_train, self.dm_feature_names = self.derive_dm_features(
            X_train, feature_names)
        dm_X_val, _ = self.derive_dm_features(X_val, feature_names)

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']

        # Approximate max adverse excursion:
        # For longs: max drawdown = max(-min(ret5, 0), abs(min(ret20, 0)))
        # Simplified: use the worst of the early and full-period returns
        mae_train = np.maximum(
            np.abs(np.minimum(ret5, 0)),
            np.abs(np.minimum(ret20, 0))
        )
        mae_val = np.maximum(
            np.abs(np.minimum(ret5_val, 0)),
            np.abs(np.minimum(ret20_val, 0))
        )

        metrics = {}
        print(f"\n  Drawdown magnitude features: {len(self.dm_feature_names)}")
        print(f"  MAE stats (train): mean={mae_train.mean():.4f}, "
              f"p50={np.median(mae_train):.4f}, p75={np.percentile(mae_train, 75):.4f}")

        base_params = {
            'num_leaves': 31, 'learning_rate': 0.05,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8,
            'bagging_freq': 5, 'verbose': -1,
        }

        for alpha, name in [(0.25, 'p25'), (0.50, 'p50'), (0.75, 'p75')]:
            callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
            params = {**base_params, 'objective': 'quantile', 'alpha': alpha,
                     'metric': 'quantile'}
            dtrain = lgb.Dataset(dm_X_train, label=mae_train,
                                feature_name=self.dm_feature_names)
            dval = lgb.Dataset(dm_X_val, label=mae_val,
                              feature_name=self.dm_feature_names, reference=dtrain)

            model = lgb.train(params, dtrain, num_boost_round=500,
                            valid_sets=[dval], callbacks=callbacks)
            setattr(self, f'{name}_model', model)

            pred = model.predict(dm_X_val)
            corr = np.corrcoef(pred, mae_val)[0, 1]
            coverage = (mae_val <= pred).mean()
            metrics[f'{name}_corr'] = float(corr) if not np.isnan(corr) else 0.0
            metrics[f'{name}_coverage'] = float(coverage)
            print(f"  {name.upper()}: corr={corr:.3f}, coverage={coverage:.1%}")

        # Feature importance from p50 model
        imp = self.p50_model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 drawdown magnitude features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.dm_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.p50_model is None:
            return {}
        dm_X, _ = self.derive_dm_features(X, self.feature_names)
        return {
            'mae_p25': self.p25_model.predict(dm_X),
            'mae_p50': self.p50_model.predict(dm_X),
            'mae_p75': self.p75_model.predict(dm_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'p25_model': self.p25_model,
                'p50_model': self.p50_model,
                'p75_model': self.p75_model,
                'feature_names': self.feature_names,
                'dm_feature_names': self.dm_feature_names,
            }, f)
        print(f"  Saved DrawdownMagnitudePredictor to {path}")

    @classmethod
    def load(cls, path: str) -> 'DrawdownMagnitudePredictor':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.p25_model = data['p25_model']
        model.p50_model = data['p50_model']
        model.p75_model = data['p75_model']
        model.feature_names = data['feature_names']
        model.dm_feature_names = data['dm_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 50: Win Streak Detector
# ---------------------------------------------------------------------------
# Use temporal features (deltas, trends) to predict if conditions favor
# consecutive winning trades. During "hot streaks", the features show
# stable health, low entropy, consistent direction — boost confidence.

class WinStreakDetector:
    """Predict if current conditions favor a cluster of winning trades."""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.ws_feature_names = None

    def derive_ws_features(self, X, feature_names):
        """Extract features relevant to win streak detection."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Temporal stability (key for streaks)
        for key in ['health_delta_3bar', 'health_delta_6bar',
                    'break_prob_delta_3bar', 'entropy_delta_3bar',
                    'avg_entropy', 'rsi_slope_5bar', 'vol_trend_5bar',
                    'position_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'ws_{key}')

        # Channel health across TFs (high health = stable = streaks)
        for tf in ML_TFS:
            for key in ['health', 'r_squared', 'break_prob', 'width_pct',
                        'position_in_channel']:
                idx = name_to_idx.get(f'{tf}_{key}')
                if idx is not None:
                    feats.append(X[:, idx])
                    names.append(f'ws_{tf}_{key}')

        # Cross-TF agreement (high agreement = streaks)
        for key in ['direction_agreement', 'avg_health', 'health_min',
                    'health_std', 'avg_break_prob']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'ws_{key}')

        # Context
        for key in ['rsi_14', 'atr_pct', 'volume_ratio_20',
                    'minutes_since_open', 'day_of_week']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'ws_{key}')

        # Derived: stability score (low absolute deltas = stable)
        delta_keys = ['health_delta_3bar', 'break_prob_delta_3bar', 'entropy_delta_3bar']
        delta_vals = []
        for key in delta_keys:
            idx = name_to_idx.get(key)
            if idx is not None:
                delta_vals.append(np.abs(X[:, idx]))
        if delta_vals:
            feats.append(np.mean(delta_vals, axis=0))
            names.append('ws_avg_abs_delta')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train win streak detector."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        ws_X_train, self.ws_feature_names = self.derive_ws_features(
            X_train, feature_names)
        ws_X_val, _ = self.derive_ws_features(X_val, feature_names)

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']

        # "Streak-worthy" = both ret5 > 0 AND ret20 > 0.3%
        # This identifies conditions that produce consistent winners
        streak_train = (
            (ret5 > 0) & (ret20 > 0.003)
        ).astype(np.float32)
        streak_val = (
            (ret5_val > 0) & (ret20_val > 0.003)
        ).astype(np.float32)

        metrics = {}
        print(f"\n  Win streak features: {len(self.ws_feature_names)}")
        print(f"  Streak rate (train): {streak_train.mean():.1%}")
        print(f"  Streak rate (val): {streak_val.mean():.1%}")

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(ws_X_train, label=streak_train,
                            feature_name=self.ws_feature_names)
        dval = lgb.Dataset(ws_X_val, label=streak_val,
                          feature_name=self.ws_feature_names, reference=dtrain)

        self.model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        ws_pred = self.model.predict(ws_X_val)
        try:
            auc = roc_auc_score(streak_val, ws_pred)
            metrics['streak_auc'] = float(auc)
            print(f"  Win Streak AUC: {auc:.3f}")
        except Exception:
            metrics['streak_auc'] = 0.5

        imp = self.model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 win streak features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.ws_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.model is None:
            return {}
        ws_X, _ = self.derive_ws_features(X, self.feature_names)
        return {
            'streak_prob': self.model.predict(ws_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'ws_feature_names': self.ws_feature_names,
            }, f)
        print(f"  Saved WinStreakDetector to {path}")

    @classmethod
    def load(cls, path: str) -> 'WinStreakDetector':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.model = data['model']
        model.feature_names = data['feature_names']
        model.ws_feature_names = data['ws_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 51: Reversal Proximity Detector
# ---------------------------------------------------------------------------
# Combine channel maturity signals with break probability dynamics to
# predict if we're about to hit a major reversal. Different from break_prob
# because it uses ACCELERATION and cross-TF confirmation.

class ReversalProximityDetector:
    """Predict proximity to a channel break/reversal using acceleration signals."""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.rp_feature_names = None

    def derive_rp_features(self, X, feature_names):
        """Extract features with emphasis on acceleration and cross-TF patterns."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Break probability and its derivatives
        for tf in ML_TFS:
            bp_idx = name_to_idx.get(f'{tf}_break_prob')
            if bp_idx is not None:
                feats.append(X[:, bp_idx])
                names.append(f'rp_{tf}_break_prob')

            h_idx = name_to_idx.get(f'{tf}_health')
            if h_idx is not None:
                feats.append(X[:, h_idx])
                names.append(f'rp_{tf}_health')

            lt_idx = name_to_idx.get(f'{tf}_lifetime_bars')
            if lt_idx is not None:
                feats.append(X[:, lt_idx])
                names.append(f'rp_{tf}_lifetime_bars')

            pos_idx = name_to_idx.get(f'{tf}_position_in_channel')
            if pos_idx is not None:
                feats.append(X[:, pos_idx])
                names.append(f'rp_{tf}_position')

        # Temporal dynamics — acceleration matters
        for key in ['break_prob_delta_3bar', 'health_delta_3bar',
                    'health_delta_6bar', 'entropy_delta_3bar',
                    'avg_entropy', 'rsi_slope_5bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'rp_{key}')

        # Cross-TF divergence
        for key in ['direction_agreement', 'avg_break_prob',
                    'health_min', 'health_std']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'rp_{key}')

        # Derived: break_prob acceleration (3bar delta minus health delta)
        bp3 = name_to_idx.get('break_prob_delta_3bar')
        h3 = name_to_idx.get('health_delta_3bar')
        if bp3 is not None and h3 is not None:
            feats.append(X[:, bp3] + np.abs(X[:, h3]))  # Rising break + falling health
            names.append('rp_reversal_pressure')

        # Derived: max break_prob across TFs
        bp_vals = []
        for tf in ML_TFS:
            idx = name_to_idx.get(f'{tf}_break_prob')
            if idx is not None:
                bp_vals.append(X[:, idx])
        if bp_vals:
            feats.append(np.max(bp_vals, axis=0))
            names.append('rp_max_break_prob')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train reversal proximity detector."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        rp_X_train, self.rp_feature_names = self.derive_rp_features(
            X_train, feature_names)
        rp_X_val, _ = self.derive_rp_features(X_val, feature_names)

        lifetime = Y_train['channel_lifetime']
        lifetime_val = Y_val['channel_lifetime']

        # "Near reversal" = channel will break within 10 bars
        near_rev_train = (lifetime < 10).astype(np.float32)
        near_rev_val = (lifetime_val < 10).astype(np.float32)

        metrics = {}
        print(f"\n  Reversal proximity features: {len(self.rp_feature_names)}")
        print(f"  Near-reversal rate (train): {near_rev_train.mean():.1%}")
        print(f"  Near-reversal rate (val): {near_rev_val.mean():.1%}")

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(rp_X_train, label=near_rev_train,
                            feature_name=self.rp_feature_names)
        dval = lgb.Dataset(rp_X_val, label=near_rev_val,
                          feature_name=self.rp_feature_names, reference=dtrain)

        self.model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        rp_pred = self.model.predict(rp_X_val)
        try:
            auc = roc_auc_score(near_rev_val, rp_pred)
            metrics['reversal_auc'] = float(auc)
            print(f"  Reversal Proximity AUC: {auc:.3f}")
        except Exception:
            metrics['reversal_auc'] = 0.5

        imp = self.model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 reversal proximity features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.rp_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.model is None:
            return {}
        rp_X, _ = self.derive_rp_features(X, self.feature_names)
        return {
            'reversal_prob': self.model.predict(rp_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'rp_feature_names': self.rp_feature_names,
            }, f)
        print(f"  Saved ReversalProximityDetector to {path}")

    @classmethod
    def load(cls, path: str) -> 'ReversalProximityDetector':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.model = data['model']
        model.feature_names = data['feature_names']
        model.rp_feature_names = data['rp_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 52: Volatility-Return Regime
# ---------------------------------------------------------------------------
# Classify the current bar into one of 4 regimes based on the
# vol-return relationship:
# 1. Low vol + positive return (ideal: smooth trend)
# 2. Low vol + negative return (quiet decline)
# 3. High vol + positive return (volatile rally)
# 4. High vol + negative return (volatile sell-off)
# Each regime has different optimal sizing and exit behavior.

class VolReturnRegimeClassifier:
    """Classify volatility-return regimes for differential trading."""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.vr_feature_names = None

    def derive_vr_features(self, X, feature_names):
        """Extract features for vol-return regime classification."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Vol indicators
        for key in ['atr_pct', 'bar_range_pct', 'volume_ratio_20']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'vr_{key}')

        # Channel width (proxy for vol regime)
        for tf in ML_TFS:
            for key in ['width_pct', 'health', 'slope_normalized',
                        'r_squared', 'break_prob']:
                idx = name_to_idx.get(f'{tf}_{key}')
                if idx is not None:
                    feats.append(X[:, idx])
                    names.append(f'vr_{tf}_{key}')

        # Temporal vol dynamics
        for key in ['vol_trend_5bar', 'entropy_delta_3bar', 'avg_entropy',
                    'rsi_slope_5bar', 'health_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'vr_{key}')

        # Cross-TF
        for key in ['direction_agreement', 'avg_health', 'width_dispersion',
                    'health_std']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'vr_{key}')

        # Market context
        for key in ['vix_level', 'tsla_spy_corr_20', 'rsi_14',
                    'minutes_since_open']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'vr_{key}')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train vol-return regime classifier."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        vr_X_train, self.vr_feature_names = self.derive_vr_features(
            X_train, feature_names)
        vr_X_val, _ = self.derive_vr_features(X_val, feature_names)

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']

        # Target: "smooth trend" regime (ideal for trading)
        # Low vol (|ret5| < 0.5%) AND positive ret20 (> 0.3%)
        # This is the ideal condition: calm market moving in our direction
        smooth_trend_train = (
            (np.abs(ret5) < 0.005) & (ret20 > 0.003)
        ).astype(np.float32)
        smooth_trend_val = (
            (np.abs(ret5_val) < 0.005) & (ret20_val > 0.003)
        ).astype(np.float32)

        metrics = {}
        print(f"\n  Vol-return regime features: {len(self.vr_feature_names)}")
        print(f"  Smooth trend rate (train): {smooth_trend_train.mean():.1%}")
        print(f"  Smooth trend rate (val): {smooth_trend_val.mean():.1%}")

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(vr_X_train, label=smooth_trend_train,
                            feature_name=self.vr_feature_names)
        dval = lgb.Dataset(vr_X_val, label=smooth_trend_val,
                          feature_name=self.vr_feature_names, reference=dtrain)

        self.model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        vr_pred = self.model.predict(vr_X_val)
        try:
            auc = roc_auc_score(smooth_trend_val, vr_pred)
            metrics['smooth_auc'] = float(auc)
            print(f"  Smooth Trend AUC: {auc:.3f}")
        except Exception:
            metrics['smooth_auc'] = 0.5

        imp = self.model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 vol-return regime features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.vr_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.model is None:
            return {}
        vr_X, _ = self.derive_vr_features(X, self.feature_names)
        return {
            'smooth_trend_prob': self.model.predict(vr_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'vr_feature_names': self.vr_feature_names,
            }, f)
        print(f"  Saved VolReturnRegimeClassifier to {path}")

    @classmethod
    def load(cls, path: str) -> 'VolReturnRegimeClassifier':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.model = data['model']
        model.feature_names = data['feature_names']
        model.vr_feature_names = data['vr_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 53: Multi-Horizon Loser Detector
# ---------------------------------------------------------------------------
# A trade that loses across ALL horizons (ret5 < 0, ret20 < 0, ret60 < 0)
# is the worst kind of trade — it never even temporarily goes your way.
# Much higher value to avoid than a temporary dip.

class MultiHorizonLoserDetector:
    """Detect trades that lose across all three time horizons."""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.mh_feature_names = None

    def derive_mh_features(self, X, feature_names):
        """Use ALL features — let LightGBM find what matters."""
        # Use all raw features (no feature engineering)
        # This gives the model maximum information to detect multi-horizon losers
        return X, list(feature_names)

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train multi-horizon loser detector."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        self.mh_feature_names = list(feature_names)

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret60 = Y_train['future_return_60']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']
        ret60_val = Y_val['future_return_60']

        # Multi-horizon loser: loses at ALL three horizons
        mh_loser_train = (
            (ret5 < -0.001) & (ret20 < -0.002) & (ret60 < -0.003)
        ).astype(np.float32)
        mh_loser_val = (
            (ret5_val < -0.001) & (ret20_val < -0.002) & (ret60_val < -0.003)
        ).astype(np.float32)

        metrics = {}
        print(f"\n  Multi-horizon loser features: {len(self.mh_feature_names)}")
        print(f"  MH loser rate (train): {mh_loser_train.mean():.1%}")
        print(f"  MH loser rate (val): {mh_loser_val.mean():.1%}")

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(X_train, label=mh_loser_train,
                            feature_name=self.mh_feature_names)
        dval = lgb.Dataset(X_val, label=mh_loser_val,
                          feature_name=self.mh_feature_names, reference=dtrain)

        self.model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.7, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1,
             'scale_pos_weight': 4.0},  # Heavy emphasis on losers
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        mh_pred = self.model.predict(X_val)
        try:
            auc = roc_auc_score(mh_loser_val, mh_pred)
            metrics['mh_loser_auc'] = float(auc)
            print(f"  Multi-Horizon Loser AUC: {auc:.3f}")
        except Exception:
            metrics['mh_loser_auc'] = 0.5

        # Precision at various thresholds
        for thr in [0.2, 0.3, 0.4]:
            flagged = mh_pred > thr
            if flagged.sum() > 0:
                precision = mh_loser_val[flagged].mean()
                print(f"    Threshold {thr:.1f}: {flagged.sum()} flagged, {precision:.1%} precision")

        imp = self.model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 multi-horizon loser features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.mh_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.model is None:
            return {}
        return {
            'mh_loser_prob': self.model.predict(X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'mh_feature_names': self.mh_feature_names,
            }, f)
        print(f"  Saved MultiHorizonLoserDetector to {path}")

    @classmethod
    def load(cls, path: str) -> 'MultiHorizonLoserDetector':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.model = data['model']
        model.feature_names = data['feature_names']
        model.mh_feature_names = data['mh_feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 54: Bounce-Specific Loser Detector
# ---------------------------------------------------------------------------
# Bounces have different failure modes than breaks. Train a specialized
# loser detector ONLY on bounce-like conditions (position near boundary).

class BounceLoserDetector:
    """Detect losers specifically in bounce-type trade conditions."""

    def __init__(self):
        self.model = None
        self.feature_names = None

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train bounce-specific loser detector."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        name_to_idx = {n: i for i, n in enumerate(feature_names)}

        # Filter to bounce-like conditions: 5min position near boundary (<0.15 or >0.85)
        pos_idx = name_to_idx.get('5min_position_in_channel', 0)
        pos_train = X_train[:, pos_idx]
        pos_val = X_val[:, pos_idx]

        bounce_mask_train = (pos_train < 0.15) | (pos_train > 0.85)
        bounce_mask_val = (pos_val < 0.15) | (pos_val > 0.85)

        ret20 = Y_train['future_return_20']
        ret20_val = Y_val['future_return_20']

        # Loser = ret20 < -0.5% (significant loss)
        loser_train = (ret20 < -0.005).astype(np.float32)
        loser_val = (ret20_val < -0.005).astype(np.float32)

        metrics = {}
        n_bounce_train = bounce_mask_train.sum()
        n_bounce_val = bounce_mask_val.sum()
        print(f"\n  Bounce-like samples: train={n_bounce_train}, val={n_bounce_val}")

        if n_bounce_train < 50 or n_bounce_val < 10:
            print("  Not enough bounce samples, training on all data with position weighting")
            # Weight boundary samples 3x
            weights = np.where(bounce_mask_train, 3.0, 1.0).astype(np.float32)
        else:
            weights = None

        X_t = X_train
        y_t = loser_train
        X_v = X_val
        y_v = loser_val

        print(f"  Features: {len(feature_names)}")
        print(f"  Loser rate (train): {y_t.mean():.1%}")
        print(f"  Loser rate (val): {y_v.mean():.1%}")
        if bounce_mask_train.sum() > 0:
            print(f"  Loser rate at boundary (train): {loser_train[bounce_mask_train].mean():.1%}")

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(X_t, label=y_t, feature_name=list(feature_names),
                            weight=weights)
        dval = lgb.Dataset(X_v, label=y_v, feature_name=list(feature_names),
                          reference=dtrain)

        self.model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.7, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'scale_pos_weight': 3.0},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        pred = self.model.predict(X_v)
        try:
            auc = roc_auc_score(y_v, pred)
            metrics['bounce_loser_auc'] = float(auc)
            print(f"  Bounce Loser AUC (all): {auc:.3f}")
        except Exception:
            metrics['bounce_loser_auc'] = 0.5

        if bounce_mask_val.sum() > 5:
            try:
                b_auc = roc_auc_score(y_v[bounce_mask_val], pred[bounce_mask_val])
                metrics['bounce_loser_boundary_auc'] = float(b_auc)
                print(f"  Bounce Loser AUC (boundary only): {b_auc:.3f}")
            except Exception:
                pass

        imp = self.model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 bounce loser features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.model is None:
            return {}
        return {
            'bounce_loser_prob': self.model.predict(X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
            }, f)
        print(f"  Saved BounceLoserDetector to {path}")

    @classmethod
    def load(cls, path: str) -> 'BounceLoserDetector':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.model = data['model']
        model.feature_names = data['feature_names']
        return model


# ---------------------------------------------------------------------------
# Architecture 55: Feature Interaction Loser
# ---------------------------------------------------------------------------
# Create pairwise interaction features from the top-10 most important
# features. LightGBM can discover splits but explicit interactions
# may capture patterns it misses with individual features.

class FeatureInteractionLoser:
    """Use pairwise feature interactions to detect losers."""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.int_feature_names = None
        self.top_feature_indices = None

    def derive_interaction_features(self, X, feature_names, top_indices=None):
        """Create pairwise interaction features from top features."""
        if top_indices is None:
            # Use first call to determine top features
            return X, list(feature_names), None

        feats = [X]  # Start with original features
        names = list(feature_names)

        # Add pairwise products of top features
        for i in range(len(top_indices)):
            for j in range(i + 1, len(top_indices)):
                idx_i, idx_j = top_indices[i], top_indices[j]
                interaction = X[:, idx_i] * X[:, idx_j]
                feats.append(interaction.reshape(-1, 1))
                names.append(f'int_{feature_names[idx_i]}_x_{feature_names[idx_j]}')

        # Add ratios of top features
        for i in range(len(top_indices)):
            for j in range(len(top_indices)):
                if i != j:
                    idx_i, idx_j = top_indices[i], top_indices[j]
                    denom = np.abs(X[:, idx_j]) + 1e-8
                    ratio = X[:, idx_i] / denom
                    ratio = np.clip(ratio, -10, 10)
                    feats.append(ratio.reshape(-1, 1))
                    names.append(f'rat_{feature_names[idx_i]}/{feature_names[idx_j]}')

        return np.column_stack(feats), names, top_indices

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train feature interaction loser detector."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)

        ret20 = Y_train['future_return_20']
        ret20_val = Y_val['future_return_20']

        # Bottom 15% = extreme loser (same target as Arch 45)
        threshold = np.percentile(ret20, 15)
        loser_train = (ret20 < threshold).astype(np.float32)
        loser_val = (ret20_val < threshold).astype(np.float32)

        # First: train a quick model to find top features
        print(f"\n  Phase 1: Finding top features...")
        callbacks_quick = [lgb.early_stopping(30), lgb.log_evaluation(0)]
        dtrain_quick = lgb.Dataset(X_train, label=loser_train,
                                  feature_name=list(feature_names))
        dval_quick = lgb.Dataset(X_val, label=loser_val,
                                feature_name=list(feature_names), reference=dtrain_quick)

        quick_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 15,
             'learning_rate': 0.1, 'verbose': -1},
            dtrain_quick, num_boost_round=100, valid_sets=[dval_quick],
            callbacks=callbacks_quick)

        imp = quick_model.feature_importance(importance_type='gain')
        self.top_feature_indices = list(np.argsort(imp)[::-1][:10])

        print(f"  Top 10 features for interactions:")
        for rank, idx in enumerate(self.top_feature_indices):
            print(f"    {rank+1}. {feature_names[idx]}: {imp[idx]:.0f}")

        # Phase 2: Create interaction features and retrain
        print(f"\n  Phase 2: Training with interaction features...")
        int_X_train, self.int_feature_names, _ = self.derive_interaction_features(
            X_train, feature_names, self.top_feature_indices)
        int_X_val, _, _ = self.derive_interaction_features(
            X_val, feature_names, self.top_feature_indices)

        n_interactions = len(self.int_feature_names) - len(feature_names)
        print(f"  Original features: {len(feature_names)}")
        print(f"  Interaction features added: {n_interactions}")
        print(f"  Total features: {len(self.int_feature_names)}")

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(int_X_train, label=loser_train,
                            feature_name=self.int_feature_names)
        dval = lgb.Dataset(int_X_val, label=loser_val,
                          feature_name=self.int_feature_names, reference=dtrain)

        self.model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.6, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'scale_pos_weight': 3.0},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        pred = self.model.predict(int_X_val)
        metrics = {}
        try:
            auc = roc_auc_score(loser_val, pred)
            metrics['int_loser_auc'] = float(auc)
            print(f"  Interaction Loser AUC: {auc:.3f}")
        except Exception:
            metrics['int_loser_auc'] = 0.5

        # Compare to base (quick model)
        base_pred = quick_model.predict(X_val)
        base_auc = roc_auc_score(loser_val, base_pred)
        metrics['base_auc'] = float(base_auc)
        print(f"  Base (no interactions) AUC: {base_auc:.3f}")
        print(f"  Improvement: {auc - base_auc:+.3f}")

        # Show top interaction features used
        final_imp = self.model.feature_importance(importance_type='gain')
        top_final = np.argsort(final_imp)[::-1][:10]
        print("\n  Top 10 features (with interactions):")
        for rank, idx in enumerate(top_final):
            is_int = "🔗" if idx >= len(feature_names) else ""
            print(f"    {rank+1}. {self.int_feature_names[idx]}: {final_imp[idx]:.0f} {is_int}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.model is None:
            return {}
        int_X, _, _ = self.derive_interaction_features(
            X, self.feature_names, self.top_feature_indices)
        return {
            'int_loser_prob': self.model.predict(int_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'int_feature_names': self.int_feature_names,
                'top_feature_indices': self.top_feature_indices,
            }, f)
        print(f"  Saved FeatureInteractionLoser to {path}")

    @classmethod
    def load(cls, path: str) -> 'FeatureInteractionLoser':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.model = data['model']
        model.feature_names = data['feature_names']
        model.int_feature_names = data['int_feature_names']
        model.top_feature_indices = data['top_feature_indices']
        return model


# ---------------------------------------------------------------------------
# Architecture 56: Momentum Reversal Detector
# ---------------------------------------------------------------------------
# Predict "false starts" — trades that start profitable (ret5 > 0)
# but then reverse (ret20 < 0). These are the most frustrating trades
# and tighter trails or earlier exits can help.

class MomentumReversalDetector:
    """Detect trades that start well but reverse (ret5 > 0, ret20 < 0)."""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.mr_feature_names = None

    def derive_mr_features(self, X, feature_names):
        """Extract features relevant to momentum reversals."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Channel stability (unstable = more reversals)
        for tf in ML_TFS:
            for key in ['health', 'r_squared', 'break_prob', 'width_pct',
                        'slope_normalized', 'position_in_channel',
                        'bars_since_touch']:
                idx = name_to_idx.get(f'{tf}_{key}')
                if idx is not None:
                    feats.append(X[:, idx])
                    names.append(f'mr_{tf}_{key}')

        # Temporal dynamics (accelerating changes = reversal signal)
        for key in ['health_delta_3bar', 'health_delta_6bar',
                    'break_prob_delta_3bar', 'entropy_delta_3bar',
                    'avg_entropy', 'rsi_slope_5bar', 'vol_trend_5bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'mr_{key}')

        # Cross-TF disagreement (conflicting signals = reversals)
        for key in ['direction_agreement', 'avg_health', 'health_min',
                    'health_std', 'avg_break_prob', 'width_dispersion']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'mr_{key}')

        # Context
        for key in ['rsi_14', 'atr_pct', 'volume_ratio_20',
                    'bar_range_pct', 'minutes_since_open']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'mr_{key}')

        # Correlation features
        for key in ['vix_level', 'tsla_spy_corr_20']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'mr_{key}')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train momentum reversal detector."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        mr_X_train, self.mr_feature_names = self.derive_mr_features(
            X_train, feature_names)
        mr_X_val, _ = self.derive_mr_features(X_val, feature_names)

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']

        # "False start": starts positive (ret5 > 0.1%) then reverses (ret20 < -0.2%)
        reversal_train = (
            (ret5 > 0.001) & (ret20 < -0.002)
        ).astype(np.float32)
        reversal_val = (
            (ret5_val > 0.001) & (ret20_val < -0.002)
        ).astype(np.float32)

        metrics = {}
        print(f"\n  Momentum reversal features: {len(self.mr_feature_names)}")
        print(f"  Reversal rate (train): {reversal_train.mean():.1%}")
        print(f"  Reversal rate (val): {reversal_val.mean():.1%}")

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(mr_X_train, label=reversal_train,
                            feature_name=self.mr_feature_names)
        dval = lgb.Dataset(mr_X_val, label=reversal_val,
                          feature_name=self.mr_feature_names, reference=dtrain)

        self.model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'scale_pos_weight': 3.0},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        mr_pred = self.model.predict(mr_X_val)
        try:
            auc = roc_auc_score(reversal_val, mr_pred)
            metrics['reversal_auc'] = float(auc)
            print(f"  Momentum Reversal AUC: {auc:.3f}")
        except Exception:
            metrics['reversal_auc'] = 0.5

        imp = self.model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 momentum reversal features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.mr_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.model is None:
            return {}
        mr_X, _ = self.derive_mr_features(X, self.feature_names)
        return {
            'reversal_prob': self.model.predict(mr_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'mr_feature_names': self.mr_feature_names,
            }, f)
        print(f"  Saved MomentumReversalDetector to {path}")

    @classmethod
    def load(cls, path: str) -> 'MomentumReversalDetector':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.model = data['model']
        model.feature_names = data['feature_names']
        model.mr_feature_names = data['mr_feature_names']
        return model


class ImmediateStopDetector:
    """Arch 57: Predict if price moves >0.5% against within 3 bars (before trail activates).

    These are the "instant stop" losses — price gaps against us immediately.
    If we can detect these, we can either skip the trade or use ultra-tight stops.

    Label: max_adverse_3bar > 0.5% (binary).
    Using future_return_5 as proxy: trades where ret5 is very negative.
    """

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.is_feature_names = None

    def derive_features(self, X, feature_names):
        """Extract features relevant to immediate adverse movement."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Momentum and volatility — immediate move predictors
        for tf in ML_TFS:
            for key in ['momentum_direction', 'kinetic_energy', 'position_pct',
                        'break_prob', 'slope_normalized', 'center_distance',
                        'width_pct', 'bars_since_touch']:
                idx = name_to_idx.get(f'{tf}_{key}')
                if idx is not None:
                    feats.append(X[:, idx])
                    names.append(f'is_{tf}_{key}')

        # Temporal dynamics (recent changes = immediate pressure)
        for key in ['health_delta_3bar', 'health_delta_6bar',
                    'break_prob_delta_3bar', 'entropy_delta_3bar',
                    'rsi_slope_5bar', 'vol_trend_5bar',
                    'price_acceleration_5bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'is_{key}')

        # Context (volatility, volume, time of day)
        for key in ['atr_pct', 'volume_ratio_20', 'bar_range_pct',
                    'minutes_since_open', 'rsi_14', 'bb_position']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'is_{key}')

        # Cross-TF features
        for key in ['direction_agreement', 'avg_break_prob', 'health_min',
                    'health_std', 'width_dispersion']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'is_{key}')

        # Market regime
        for key in ['vix_level', 'tsla_spy_corr_20', 'spy_rsi_14']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'is_{key}')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train immediate stop detector."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        is_X_train, self.is_feature_names = self.derive_features(X_train, feature_names)
        is_X_val, _ = self.derive_features(X_val, feature_names)

        ret5 = Y_train['future_return_5']
        ret5_val = Y_val['future_return_5']

        # "Immediate stop": price drops > 0.5% within 5 bars
        # Using absolute returns — both directions
        stop_train = (np.abs(ret5) > 0.005).astype(np.float32)
        # But we want specifically ADVERSE moves, so combine with direction
        # For simplicity: very negative ret5 = immediate adverse for longs
        # We train on absolute adverse since direction is unknown at feature time
        adverse_train = (ret5 < -0.005).astype(np.float32)
        adverse_val = (ret5_val < -0.005).astype(np.float32)

        metrics = {}
        print(f"\n  Immediate stop features: {len(self.is_feature_names)}")
        print(f"  Immediate adverse rate (train): {adverse_train.mean():.1%}")
        print(f"  Immediate adverse rate (val): {adverse_val.mean():.1%}")

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(is_X_train, label=adverse_train,
                            feature_name=self.is_feature_names)
        dval = lgb.Dataset(is_X_val, label=adverse_val,
                          feature_name=self.is_feature_names, reference=dtrain)

        self.model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'scale_pos_weight': 2.0},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        is_pred = self.model.predict(is_X_val)
        try:
            auc = roc_auc_score(adverse_val, is_pred)
            metrics['immediate_stop_auc'] = float(auc)
            print(f"  Immediate Stop AUC: {auc:.3f}")
        except Exception:
            metrics['immediate_stop_auc'] = 0.5

        imp = self.model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 immediate stop features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.is_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.model is None:
            return {}
        is_X, _ = self.derive_features(X, self.feature_names)
        return {
            'immediate_stop_prob': self.model.predict(is_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'is_feature_names': self.is_feature_names,
            }, f)
        print(f"  Saved ImmediateStopDetector to {path}")

    @classmethod
    def load(cls, path: str) -> 'ImmediateStopDetector':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.model = data['model']
        model.feature_names = data['feature_names']
        model.is_feature_names = data['is_feature_names']
        return model


class ProfitVelocityPredictor:
    """Arch 58: Predict how quickly a trade reaches profit.

    Fast-profit trades should have tighter trails (lock in the quick gain).
    Slow-profit trades might need more patience.

    Label: time_to_profit = bars until ret > +0.2% (regression, capped at 20).
    Uses ratio of ret5/ret20 as proxy for profit velocity.
    """

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.pv_feature_names = None

    def derive_features(self, X, feature_names):
        """Extract features relevant to profit velocity."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Channel momentum (trending = fast profit, range-bound = slow)
        for tf in ML_TFS:
            for key in ['slope_normalized', 'momentum_direction', 'kinetic_energy',
                        'potential_energy', 'position_pct', 'break_prob',
                        'width_pct', 'center_distance']:
                idx = name_to_idx.get(f'{tf}_{key}')
                if idx is not None:
                    feats.append(X[:, idx])
                    names.append(f'pv_{tf}_{key}')

        # Temporal features (acceleration of change)
        for key in ['health_delta_3bar', 'break_prob_delta_3bar',
                    'rsi_slope_5bar', 'vol_trend_5bar',
                    'price_acceleration_5bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'pv_{key}')

        # Volatility context
        for key in ['atr_pct', 'volume_ratio_20', 'bar_range_pct',
                    'rsi_14', 'bb_position', 'minutes_since_open']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'pv_{key}')

        # Cross-TF agreement (unanimous = fast)
        for key in ['direction_agreement', 'avg_health', 'health_std',
                    'avg_break_prob']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'pv_{key}')

        # VIX
        for key in ['vix_level']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'pv_{key}')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train profit velocity predictor."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        pv_X_train, self.pv_feature_names = self.derive_features(X_train, feature_names)
        pv_X_val, _ = self.derive_features(X_val, feature_names)

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']

        # "Fast profit": reaches +0.3% within 5 bars (vs needing 20+)
        fast_profit_train = (ret5 > 0.003).astype(np.float32)
        fast_profit_val = (ret5_val > 0.003).astype(np.float32)

        metrics = {}
        print(f"\n  Profit velocity features: {len(self.pv_feature_names)}")
        print(f"  Fast profit rate (train): {fast_profit_train.mean():.1%}")
        print(f"  Fast profit rate (val): {fast_profit_val.mean():.1%}")

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(pv_X_train, label=fast_profit_train,
                            feature_name=self.pv_feature_names)
        dval = lgb.Dataset(pv_X_val, label=fast_profit_val,
                          feature_name=self.pv_feature_names, reference=dtrain)

        self.model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        pv_pred = self.model.predict(pv_X_val)
        try:
            auc = roc_auc_score(fast_profit_val, pv_pred)
            metrics['fast_profit_auc'] = float(auc)
            print(f"  Profit Velocity AUC: {auc:.3f}")
        except Exception:
            metrics['fast_profit_auc'] = 0.5

        imp = self.model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 profit velocity features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.pv_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.model is None:
            return {}
        pv_X, _ = self.derive_features(X, self.feature_names)
        return {
            'fast_profit_prob': self.model.predict(pv_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'pv_feature_names': self.pv_feature_names,
            }, f)
        print(f"  Saved ProfitVelocityPredictor to {path}")

    @classmethod
    def load(cls, path: str) -> 'ProfitVelocityPredictor':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.model = data['model']
        model.feature_names = data['feature_names']
        model.pv_feature_names = data['pv_feature_names']
        return model


class BreakoutStopPredictor:
    """Arch 59: Predict if a breakout trade will hit its stop loss.

    All 8 stop-loss exits in the current backtest are breakouts on 5min TF.
    Key patterns: instant adverse (3 bars), high MAE/low MFE, high conf doesn't help.

    Label: breakout that would hit stop = ret5 < -0.005 AND ret20 < -0.003
    (sustained adverse, not just noise).
    """

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.bs_feature_names = None

    def derive_features(self, X, feature_names):
        """Extract features most relevant to breakout failure."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Channel structure at breakout (health, width, break_prob indicate setup quality)
        for tf in ML_TFS:
            for key in ['break_prob', 'channel_health', 'width_pct',
                        'position_pct', 'slope_normalized', 'center_distance',
                        'kinetic_energy', 'potential_energy',
                        'momentum_direction', 'bars_since_touch']:
                idx = name_to_idx.get(f'{tf}_{key}')
                if idx is not None:
                    feats.append(X[:, idx])
                    names.append(f'bs_{tf}_{key}')

        # Temporal (recent changes predict immediate breakout quality)
        for key in ['health_delta_3bar', 'health_delta_6bar',
                    'break_prob_delta_3bar', 'break_prob_delta_6bar',
                    'entropy_delta_3bar', 'rsi_slope_5bar',
                    'vol_trend_5bar', 'price_acceleration_5bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'bs_{key}')

        # Context (volatility determines if breakout survives noise)
        for key in ['atr_pct', 'volume_ratio_20', 'bar_range_pct',
                    'rsi_14', 'bb_position', 'minutes_since_open']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'bs_{key}')

        # Cross-TF agreement (aligned TFs = better breakout)
        for key in ['direction_agreement', 'avg_break_prob', 'avg_health',
                    'health_min', 'health_std', 'width_dispersion',
                    'energy_alignment']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'bs_{key}')

        # Market regime
        for key in ['vix_level', 'tsla_spy_corr_20', 'spy_rsi_14',
                    'tsla_spy_beta_20']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'bs_{key}')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train breakout stop predictor."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)
        bs_X_train, self.bs_feature_names = self.derive_features(X_train, feature_names)
        bs_X_val, _ = self.derive_features(X_val, feature_names)

        ret5 = Y_train['future_return_5']
        ret20 = Y_train['future_return_20']
        ret5_val = Y_val['future_return_5']
        ret20_val = Y_val['future_return_20']

        # "Breakout stop": sustained adverse move (not just noise)
        # ret5 < -0.005 AND ret20 < -0.003 → price keeps going wrong
        stop_train = ((ret5 < -0.005) & (ret20 < -0.003)).astype(np.float32)
        stop_val = ((ret5_val < -0.005) & (ret20_val < -0.003)).astype(np.float32)

        metrics = {}
        print(f"\n  Breakout stop features: {len(self.bs_feature_names)}")
        print(f"  Breakout stop rate (train): {stop_train.mean():.1%}")
        print(f"  Breakout stop rate (val): {stop_val.mean():.1%}")

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        dtrain = lgb.Dataset(bs_X_train, label=stop_train,
                            feature_name=self.bs_feature_names)
        dval = lgb.Dataset(bs_X_val, label=stop_val,
                          feature_name=self.bs_feature_names, reference=dtrain)

        self.model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
             'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'verbose': -1, 'scale_pos_weight': 2.0},
            dtrain, num_boost_round=500, valid_sets=[dval], callbacks=callbacks)

        bs_pred = self.model.predict(bs_X_val)
        try:
            auc = roc_auc_score(stop_val, bs_pred)
            metrics['breakout_stop_auc'] = float(auc)
            print(f"  Breakout Stop AUC: {auc:.3f}")
        except Exception:
            metrics['breakout_stop_auc'] = 0.5

        imp = self.model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n  Top 10 breakout stop features:")
        for rank, idx in enumerate(top_idx):
            print(f"    {rank+1}. {self.bs_feature_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.model is None:
            return {}
        bs_X, _ = self.derive_features(X, self.feature_names)
        return {
            'breakout_stop_prob': self.model.predict(bs_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'bs_feature_names': self.bs_feature_names,
            }, f)
        print(f"  Saved BreakoutStopPredictor to {path}")

    @classmethod
    def load(cls, path: str) -> 'BreakoutStopPredictor':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.model = data['model']
        model.feature_names = data['feature_names']
        model.bs_feature_names = data['bs_feature_names']
        return model


class BreakoutMomentumValidator:
    """Arch 60: Predict immediate breakout momentum (MFE in first 6 bars).

    Key insight: ultra-tight trail needs the price to move favorably by at least
    0.01% for the trail to activate. Breakouts with zero immediate momentum
    (stall breakouts) will sit at breakeven until timeout or stop.

    Label: immediate_mfe_pct = max favorable excursion in first 6 bars after entry
    Binary: good_momentum = (immediate_mfe_pct > 0.10%)  — enough for trail to work

    Uses breakout-specific features:
    - Channel break probability alignment across timeframes
    - Volume surge at breakout (confirming vs fake)
    - Momentum coherence (price + energy + slope aligned)
    - Time-of-day (breakouts at open vs midday behave differently)
    """

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.derived_names = None

    def derive_features(self, X, feature_names):
        """Extract breakout momentum features."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        feats = []
        names = []

        # Per-TF breakout indicators
        for tf in ML_TFS:
            for key in ['break_prob', 'channel_health', 'width_pct',
                        'position_pct', 'slope_normalized', 'kinetic_energy',
                        'potential_energy', 'momentum_direction', 'center_distance',
                        'bars_since_touch', 'touch_count']:
                idx = name_to_idx.get(f'{tf}_{key}')
                if idx is not None:
                    feats.append(X[:, idx])
                    names.append(f'bm_{tf}_{key}')

        # Temporal dynamics (momentum building or fading)
        for key in ['health_delta_3bar', 'health_delta_6bar',
                    'break_prob_delta_3bar', 'break_prob_delta_6bar',
                    'entropy_delta_3bar', 'rsi_slope_5bar',
                    'vol_trend_5bar', 'price_acceleration_5bar',
                    'momentum_delta_3bar', 'ke_delta_3bar']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'bm_{key}')

        # Context features (volume, volatility, time)
        for key in ['atr_pct', 'volume_ratio_20', 'bar_range_pct',
                    'rsi_14', 'bb_position', 'minutes_since_open',
                    'is_first_hour', 'is_last_hour', 'is_overnight']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'bm_{key}')

        # Cross-TF alignment (strong breakouts align across timeframes)
        for key in ['direction_agreement', 'avg_break_prob', 'avg_health',
                    'health_min', 'health_std', 'width_dispersion',
                    'energy_alignment', 'momentum_alignment']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'bm_{key}')

        # Correlation features
        for key in ['vix_level', 'tsla_spy_corr_20', 'spy_rsi_14',
                    'tsla_spy_beta_20', 'spy_volume_ratio']:
            idx = name_to_idx.get(key)
            if idx is not None:
                feats.append(X[:, idx])
                names.append(f'bm_{key}')

        # Derived: breakout quality composites
        bp_indices = [name_to_idx.get(f'{tf}_break_prob') for tf in ML_TFS]
        bp_vals = [X[:, idx] for idx in bp_indices if idx is not None]
        if bp_vals:
            bp_stack = np.column_stack(bp_vals)
            feats.append(np.max(bp_stack, axis=1))
            names.append('bm_max_break_prob')
            feats.append(np.mean(bp_stack, axis=1))
            names.append('bm_mean_break_prob')
            feats.append(np.std(bp_stack, axis=1))
            names.append('bm_std_break_prob')
            # Count TFs with high break prob (> 0.5)
            feats.append(np.sum(bp_stack > 0.5, axis=1).astype(float))
            names.append('bm_high_bp_count')

        ke_indices = [name_to_idx.get(f'{tf}_kinetic_energy') for tf in ML_TFS]
        ke_vals = [X[:, idx] for idx in ke_indices if idx is not None]
        if ke_vals:
            ke_stack = np.column_stack(ke_vals)
            feats.append(np.max(ke_stack, axis=1))
            names.append('bm_max_ke')
            feats.append(np.mean(ke_stack, axis=1))
            names.append('bm_mean_ke')

        if len(feats) == 0:
            return X, feature_names
        return np.column_stack(feats), names

    def train(self, X_train, Y_train, X_val, Y_val, feature_names):
        """Train breakout momentum validator on MFE labels."""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        self.feature_names = list(feature_names)

        # Create binary label: good momentum = MFE > 0.10% in first 6 bars
        # Uses ret5 as proxy for short-term favorable movement
        if isinstance(Y_train, dict):
            mfe_train = Y_train.get('future_return_5', np.zeros(len(X_train)))
            mfe_val = Y_val.get('future_return_5', np.zeros(len(X_val)))
        else:
            mfe_train = np.zeros(len(X_train))
            mfe_val = np.zeros(len(X_val))

        # For breakout trades, positive MFE means price moved in breakout direction
        # Label: 1 = good momentum (favorable move > 0.1%), 0 = stalled/adverse
        y_train = (np.abs(mfe_train) > 0.001).astype(float)
        y_val = (np.abs(mfe_val) > 0.001).astype(float)

        print(f"\n  Breakout Momentum Validator:")
        print(f"    Training: {len(y_train)} samples, {y_train.sum():.0f} ({y_train.mean():.1%}) good momentum")
        print(f"    Validation: {len(y_val)} samples, {y_val.sum():.0f} ({y_val.mean():.1%}) good momentum")

        # Derive features
        bm_X_train, self.derived_names = self.derive_features(X_train, feature_names)
        bm_X_val, _ = self.derive_features(X_val, feature_names)

        print(f"    Derived features: {len(self.derived_names)}")

        # Train with LightGBM
        train_data = lgb.Dataset(bm_X_train, label=y_train, feature_name=self.derived_names)
        val_data = lgb.Dataset(bm_X_val, label=y_val, feature_name=self.derived_names, reference=train_data)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'verbose': -1,
            'seed': 42,
        }

        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        self.model = lgb.train(
            params, train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        # Evaluate
        metrics = {}
        bm_pred = self.model.predict(bm_X_val)
        try:
            auc = roc_auc_score(y_val, bm_pred)
            metrics['breakout_momentum_auc'] = float(auc)
            print(f"    AUC: {auc:.3f}")
        except Exception:
            metrics['breakout_momentum_auc'] = 0.5
            print(f"    AUC: N/A (single class)")

        imp = self.model.feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[::-1][:10]
        print("\n    Top 10 breakout momentum features:")
        for rank, idx in enumerate(top_idx):
            print(f"      {rank+1}. {self.derived_names[idx]}: {imp[idx]:.0f}")

        return metrics

    def predict(self, X: np.ndarray) -> dict:
        if self.model is None:
            return {}
        bm_X, _ = self.derive_features(X, self.feature_names)
        return {
            'momentum_prob': self.model.predict(bm_X),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'derived_names': self.derived_names,
            }, f)
        print(f"  Saved BreakoutMomentumValidator to {path}")

    @classmethod
    def load(cls, path: str) -> 'BreakoutMomentumValidator':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.model = data['model']
        model.feature_names = data['feature_names']
        model.derived_names = data['derived_names']
        return model


# ---------------------------------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------------------------------

def train_all_architectures(
    days: int = 120,
    output_dir: str = 'surfer_models',
    eval_interval: int = 3,
) -> Dict[str, Dict]:
    """Train all three architectures and compare."""

    print("=" * 70)
    print("CHANNEL SURFER ML — Training Pipeline")
    print("=" * 70)

    # Generate training data
    X, Y, feature_names = generate_training_data(
        days=days, eval_interval=eval_interval, verbose=True,
    )

    print(f"\nTotal samples: {len(X)}")
    print(f"Features: {len(feature_names)}")
    print(f"Label distributions:")
    print(f"  break_direction: {np.bincount(Y['break_direction'].astype(int))}")
    print(f"  optimal_action: {np.bincount(Y['optimal_action'].astype(int))}")
    print(f"  channel_lifetime: mean={Y['channel_lifetime'].mean():.1f}, median={np.median(Y['channel_lifetime']):.1f}")

    # Train/val/test split (60/20/20, temporal)
    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    X_train, Y_train_dict = X[:train_end], {k: v[:train_end] for k, v in Y.items()}
    X_val, Y_val_dict = X[train_end:val_end], {k: v[train_end:val_end] for k, v in Y.items()}
    X_test, Y_test_dict = X[val_end:], {k: v[val_end:] for k, v in Y.items()}

    print(f"\nSplits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    # Save data for reproducibility
    np.savez_compressed(
        os.path.join(output_dir, 'training_data.npz'),
        X=X, feature_names=feature_names, **Y,
    )

    # ---- Architecture 1: GBT ----
    print("\n" + "=" * 70)
    print("ARCHITECTURE 1: Gradient Boosted Trees")
    print("=" * 70)

    gbt = GBTModel()
    gbt_metrics = gbt.train(X_train, Y_train_dict, X_val, Y_val_dict, feature_names)
    gbt.save(os.path.join(output_dir, 'gbt_model.pkl'))

    # Test set evaluation
    gbt_test = gbt.predict(X_test)
    gbt_metrics['test_lifetime_mae'] = float(np.mean(np.abs(
        gbt_test['lifetime'] - Y_test_dict['channel_lifetime'])))
    if 'break_dir' in gbt_test:
        gbt_metrics['test_break_dir_acc'] = float(np.mean(
            gbt_test['break_dir'] == Y_test_dict['break_direction'].astype(int)))
    if 'action' in gbt_test:
        gbt_metrics['test_action_acc'] = float(np.mean(
            gbt_test['action'] == Y_test_dict['optimal_action'].astype(int)))

    all_results['gbt'] = gbt_metrics

    print("\n  GBT Feature Importance (Lifetime):")
    if 'lifetime' in gbt.feature_importance:
        for name, imp in gbt.feature_importance['lifetime'][:10]:
            print(f"    {name}: {imp:.0f}")

    # ---- Architecture 2: Survival Analysis ----
    print("\n" + "=" * 70)
    print("ARCHITECTURE 2: Survival Analysis (DeepSurv)")
    print("=" * 70)

    try:
        surv = SurvivalModel(input_dim=X_train.shape[1])
        surv_metrics = surv.train(X_train, Y_train_dict, X_val, Y_val_dict, feature_names)
        surv.save(os.path.join(output_dir, 'survival_model.pt'))

        # Test concordance
        import torch
        surv.net.eval()
        device = torch.device(surv._device)
        with torch.no_grad():
            test_X = torch.FloatTensor(X_test).to(device)
            test_risk = surv.net(test_X).squeeze(-1).cpu().numpy()

        test_c_index = surv._concordance_index(
            Y_test_dict['channel_lifetime'],
            1.0 - Y_test_dict['channel_censored'],
            -test_risk,
        )
        surv_metrics['test_concordance'] = float(test_c_index)

        if surv.baseline_hazard:
            test_medians = surv.predict_median_survival(X_test)
            uncensored = Y_test_dict['channel_censored'] == 0
            if uncensored.sum() > 0:
                surv_metrics['test_median_mae'] = float(np.mean(np.abs(
                    test_medians[uncensored] - Y_test_dict['channel_lifetime'][uncensored])))

        all_results['survival'] = surv_metrics
    except Exception as e:
        print(f"  Survival model failed: {e}")
        import traceback
        traceback.print_exc()
        all_results['survival'] = {'error': str(e)}

    # ---- Architecture 3: Multi-TF Transformer ----
    print("\n" + "=" * 70)
    print("ARCHITECTURE 3: Multi-TF Transformer")
    print("=" * 70)

    try:
        transformer = MultiTFTransformer()
        trans_metrics = transformer.train(
            X_train, Y_train_dict, X_val, Y_val_dict, feature_names,
        )
        transformer.save(os.path.join(output_dir, 'transformer_model.pt'))

        # Test evaluation
        trans_test = transformer.predict(X_test)
        trans_metrics['test_lifetime_mae'] = float(np.mean(np.abs(
            trans_test['lifetime'] - Y_test_dict['channel_lifetime'])))
        trans_metrics['test_break_dir_acc'] = float(np.mean(
            trans_test['break_dir'] == Y_test_dict['break_direction'].astype(int)))
        trans_metrics['test_action_acc'] = float(np.mean(
            trans_test['action'] == Y_test_dict['optimal_action'].astype(int)))
        trans_metrics['test_return_20_dir_acc'] = float(np.mean(
            np.sign(trans_test['future_return_20']) == np.sign(Y_test_dict['future_return_20'])))

        all_results['transformer'] = trans_metrics
    except Exception as e:
        print(f"  Transformer failed: {e}")
        import traceback
        traceback.print_exc()
        all_results['transformer'] = {'error': str(e)}

    # ---- Architecture 4: Trade Quality Scorer ----
    print("\n" + "=" * 70)
    print("ARCHITECTURE 4: Trade Quality Scorer")
    print("=" * 70)

    try:
        scorer = TradeQualityScorer()
        tq_X, tq_Y, tq_features = scorer.generate_training_data(
            X_base=X, Y_base=Y, base_feature_names=feature_names, verbose=True,
        )

        # Temporal split
        tq_n = len(tq_X)
        tq_train_end = int(tq_n * 0.6)
        tq_val_end = int(tq_n * 0.8)

        tq_X_train = tq_X[:tq_train_end]
        tq_Y_train = {k: v[:tq_train_end] for k, v in tq_Y.items()}
        tq_X_val = tq_X[tq_train_end:tq_val_end]
        tq_Y_val = {k: v[tq_train_end:tq_val_end] for k, v in tq_Y.items()}
        tq_X_test = tq_X[tq_val_end:]
        tq_Y_test = {k: v[tq_val_end:] for k, v in tq_Y.items()}

        print(f"  Splits: Train={len(tq_X_train)}, Val={len(tq_X_val)}, Test={len(tq_X_test)}")
        print(f"  Win rate in data: {tq_Y['win'].mean():.1%}")

        tq_metrics = scorer.train(tq_X_train, tq_Y_train, tq_X_val, tq_Y_val, tq_features)
        scorer.save(os.path.join(output_dir, 'trade_quality_scorer.pkl'))

        # Test evaluation
        if len(tq_X_test) > 0:
            tq_test = scorer.predict(tq_X_test)
            if 'win' in tq_test:
                tq_metrics['test_win_accuracy'] = float(np.mean(
                    tq_test['win'] == tq_Y_test['win'].astype(int)))
                print(f"\n  Test win accuracy: {tq_metrics['test_win_accuracy']:.1%}")
            if 'win_prob' in tq_test:
                from sklearn.metrics import roc_auc_score
                try:
                    tq_metrics['test_win_auc'] = float(roc_auc_score(
                        tq_Y_test['win'], tq_test['win_prob']))
                    print(f"  Test win AUC: {tq_metrics['test_win_auc']:.3f}")
                except Exception:
                    pass
            if 'pnl_pct' in tq_test:
                tq_metrics['test_pnl_dir_acc'] = float(np.mean(
                    np.sign(tq_test['pnl_pct']) == np.sign(tq_Y_test['pnl_pct'])))
                print(f"  Test PnL dir accuracy: {tq_metrics['test_pnl_dir_acc']:.1%}")

        # Feature importance for win model
        if 'win' in scorer.feature_importance:
            print("\n  Win Predictor — Top Features:")
            for name, imp in scorer.feature_importance['win'][:15]:
                print(f"    {name}: {imp:.0f}")

        all_results['trade_quality'] = tq_metrics
    except Exception as e:
        print(f"  Trade Quality Scorer failed: {e}")
        import traceback
        traceback.print_exc()
        all_results['trade_quality'] = {'error': str(e)}

    # ---- Architecture 5: Stacked Ensemble ----
    print("\n" + "=" * 70)
    print("ARCHITECTURE 5: Stacked Ensemble (Meta-Learner)")
    print("=" * 70)

    try:
        # Load base models for stacking
        gbt_model = None
        trans_model = None
        surv_model = None
        qual_model = None

        gbt_path = os.path.join(output_dir, 'gbt_model.pkl')
        if os.path.exists(gbt_path):
            gbt_model = GBTModel.load(gbt_path)
            print("  Loaded GBT base model")

        trans_path = os.path.join(output_dir, 'transformer_model.pt')
        if os.path.exists(trans_path):
            try:
                trans_model = MultiTFTransformer.load(trans_path)
                print("  Loaded Transformer base model")
            except Exception:
                print("  Transformer base model failed to load")

        surv_path = os.path.join(output_dir, 'survival_model.pt')
        if os.path.exists(surv_path):
            try:
                surv_model = SurvivalModel.load(surv_path)
                print("  Loaded Survival base model")
            except Exception:
                print("  Survival base model failed to load")

        qual_path = os.path.join(output_dir, 'trade_quality_scorer.pkl')
        if os.path.exists(qual_path):
            try:
                qual_model = TradeQualityScorer.load(qual_path)
                print("  Loaded Quality Scorer base model")
            except Exception:
                print("  Quality Scorer base model failed to load")

        if gbt_model is None:
            raise ValueError("GBT base model required for ensemble")

        ensemble = EnsembleModel()
        ens_metrics = ensemble.train(
            X_train, Y_train_dict, X_val, Y_val_dict,
            gbt=gbt_model, transformer=trans_model,
            survival=surv_model, quality=qual_model,
            feature_names=feature_names,
        )
        ensemble.save(os.path.join(output_dir, 'ensemble_model.pkl'))

        # Test evaluation
        ens_test = ensemble.predict(
            X_test, gbt=gbt_model, transformer=trans_model,
            survival=surv_model, quality=qual_model,
            feature_names=feature_names,
        )

        if 'action' in ens_test:
            ens_metrics['test_action_acc'] = float(np.mean(
                ens_test['action'] == Y_test_dict['optimal_action'].astype(int)))
            print(f"\n  Test action accuracy: {ens_metrics['test_action_acc']:.1%}")

        if 'break_dir' in ens_test:
            ens_metrics['test_break_dir_acc'] = float(np.mean(
                ens_test['break_dir'] == Y_test_dict['break_direction'].astype(int)))
            print(f"  Test break dir accuracy: {ens_metrics['test_break_dir_acc']:.1%}")

        if 'lifetime' in ens_test:
            ens_metrics['test_lifetime_mae'] = float(np.mean(np.abs(
                ens_test['lifetime'] - Y_test_dict['channel_lifetime'])))
            print(f"  Test lifetime MAE: {ens_metrics['test_lifetime_mae']:.1f} bars")

        all_results['ensemble'] = ens_metrics
    except Exception as e:
        print(f"  Ensemble failed: {e}")
        import traceback
        traceback.print_exc()
        all_results['ensemble'] = {'error': str(e)}

    # ---- Architecture 6: Regime-Conditional Model ----
    print("\n" + "=" * 70)
    print("ARCHITECTURE 6: Regime-Augmented Model")
    print("=" * 70)

    try:
        regime_model = RegimeConditionalModel()
        regime_metrics = regime_model.train(
            X_train, Y_train_dict, X_val, Y_val_dict, feature_names,
        )
        regime_model.save(os.path.join(output_dir, 'regime_model.pkl'))

        # Test evaluation
        regime_test = regime_model.predict(X_test)

        if 'action' in regime_test:
            regime_metrics['test_action_acc'] = float(np.mean(
                regime_test['action'] == Y_test_dict['optimal_action'].astype(int)))
            print(f"\n  Test action accuracy: {regime_metrics['test_action_acc']:.1%}")

        if 'break_dir' in regime_test:
            regime_metrics['test_break_dir_acc'] = float(np.mean(
                regime_test['break_dir'] == Y_test_dict['break_direction'].astype(int)))
            print(f"  Test break dir accuracy: {regime_metrics['test_break_dir_acc']:.1%}")

        # Test regime distribution
        test_regimes = regime_test['regime']
        print(f"  Test regime distribution:")
        for r in range(4):
            count = np.sum(test_regimes == r)
            pct = count / len(test_regimes) * 100
            name = RegimeConditionalModel.REGIME_NAMES[r]
            print(f"    {name}: {count} ({pct:.0f}%)")

        all_results['regime'] = regime_metrics
    except Exception as e:
        print(f"  Regime model failed: {e}")
        import traceback
        traceback.print_exc()
        all_results['regime'] = {'error': str(e)}

    # ---- Architecture 7: Temporal Attention Network ----
    print("\n" + "=" * 70)
    print("ARCHITECTURE 7: Temporal Attention Network")
    print("=" * 70)

    try:
        ws = TemporalAttentionModel.WINDOW_SIZE
        temporal_attn = TemporalAttentionModel(n_features=X_train.shape[1], window_size=ws)

        # Pass GBT importance if available
        gbt_imp = None
        if 'gbt' in all_results and gbt.feature_importance.get('lifetime'):
            gbt_imp = gbt.feature_importance['lifetime']

        ta_metrics = temporal_attn.train(
            X_train, Y_train_dict, X_val, Y_val_dict, feature_names,
            gbt_importance=gbt_imp,
        )
        temporal_attn.save(os.path.join(output_dir, 'temporal_attention_model.pt'))

        # Test evaluation (pass full features; predict() selects internally)
        X_test_w = TemporalAttentionModel.create_windows(X_test, ws)
        Y_test_w = {k: v[ws - 1:] for k, v in Y_test_dict.items()}

        ta_test = temporal_attn.predict(X_test_w)
        ta_metrics['test_action_acc'] = float(np.mean(
            ta_test['action'] == Y_test_w['optimal_action'].astype(int)))
        ta_metrics['test_break_dir_acc'] = float(np.mean(
            ta_test['break_dir'] == Y_test_w['break_direction'].astype(int)))
        ta_metrics['test_lifetime_mae'] = float(np.mean(np.abs(
            ta_test['lifetime'] - Y_test_w['channel_lifetime'])))

        print(f"\n  Test action accuracy: {ta_metrics['test_action_acc']:.1%}")
        print(f"  Test break dir accuracy: {ta_metrics['test_break_dir_acc']:.1%}")
        print(f"  Test lifetime MAE: {ta_metrics['test_lifetime_mae']:.1f} bars")

        all_results['temporal_attention'] = ta_metrics
    except Exception as e:
        print(f"  Temporal Attention failed: {e}")
        import traceback
        traceback.print_exc()
        all_results['temporal_attention'] = {'error': str(e)}

    # ---- Architecture 8: Feature-Selected Trend GBT ----
    print("\n" + "=" * 70)
    print("ARCHITECTURE 8: Feature-Selected Trend GBT")
    print("=" * 70)

    try:
        trend_gbt = TrendGBTModel()
        gbt_imp = None
        if 'gbt' in all_results and gbt.feature_importance.get('lifetime'):
            gbt_imp = gbt.feature_importance['lifetime']

        ws = TrendGBTModel.WINDOW_SIZE
        tg_metrics = trend_gbt.train(
            X_train, Y_train_dict, X_val, Y_val_dict, feature_names,
            gbt_importance=gbt_imp,
        )
        trend_gbt.save(os.path.join(output_dir, 'trend_gbt_model.pkl'))

        # Test evaluation
        X_test_w = TemporalAttentionModel.create_windows(X_test, ws)
        Y_test_w = {k: v[ws - 1:] for k, v in Y_test_dict.items()}

        tg_test = trend_gbt.predict(X_test_w)
        tg_metrics['test_action_acc'] = float(np.mean(
            tg_test['action'] == Y_test_w['optimal_action'].astype(int)))
        tg_metrics['test_break_dir_acc'] = float(np.mean(
            tg_test['break_dir'] == Y_test_w['break_direction'].astype(int)))
        tg_metrics['test_lifetime_mae'] = float(np.mean(np.abs(
            tg_test['lifetime'] - Y_test_w['channel_lifetime'])))

        print(f"\n  Test action accuracy: {tg_metrics['test_action_acc']:.1%}")
        print(f"  Test break dir accuracy: {tg_metrics['test_break_dir_acc']:.1%}")
        print(f"  Test lifetime MAE: {tg_metrics['test_lifetime_mae']:.1f} bars")

        all_results['trend_gbt'] = tg_metrics
    except Exception as e:
        print(f"  Trend GBT failed: {e}")
        import traceback
        traceback.print_exc()
        all_results['trend_gbt'] = {'error': str(e)}

    # ---- Architecture 9: Cross-Validated Ensemble ----
    print("\n" + "=" * 70)
    print("ARCHITECTURE 9: Cross-Validated Ensemble (Uncertainty)")
    print("=" * 70)

    try:
        cv_ensemble = CVEnsembleModel()
        # Use ALL data (not split) — CV does its own splitting
        cv_metrics = cv_ensemble.train(X, Y, feature_names)
        cv_ensemble.save(os.path.join(output_dir, 'cv_ensemble_model.pkl'))

        # Test on held-out set using averaged predictions
        cv_test = cv_ensemble.predict(X_test)
        cv_metrics['test_action_acc'] = float(np.mean(
            cv_test['action'] == Y_test_dict['optimal_action'].astype(int)))
        cv_metrics['test_break_dir_acc'] = float(np.mean(
            cv_test['break_dir'] == Y_test_dict['break_direction'].astype(int)))
        cv_metrics['test_lifetime_mae'] = float(np.mean(np.abs(
            cv_test['lifetime'] - Y_test_dict['channel_lifetime'])))

        # High-consensus test accuracy
        high_cons_mask = cv_test['bd_consensus'] >= 0.8
        if high_cons_mask.sum() > 5:
            cv_metrics['test_bd_high_consensus_acc'] = float(np.mean(
                cv_test['break_dir'][high_cons_mask] == Y_test_dict['break_direction'][high_cons_mask].astype(int)))
            cv_metrics['test_bd_high_consensus_coverage'] = float(high_cons_mask.mean())

        print(f"\n  Test action accuracy: {cv_metrics['test_action_acc']:.1%}")
        print(f"  Test break dir accuracy: {cv_metrics['test_break_dir_acc']:.1%}")
        print(f"  Test lifetime MAE: {cv_metrics['test_lifetime_mae']:.1f} bars")
        if 'test_bd_high_consensus_acc' in cv_metrics:
            print(f"  High-consensus BD acc: {cv_metrics['test_bd_high_consensus_acc']:.1%} "
                  f"(coverage: {cv_metrics['test_bd_high_consensus_coverage']:.1%})")

        all_results['cv_ensemble'] = cv_metrics
    except Exception as e:
        print(f"  CV Ensemble failed: {e}")
        import traceback
        traceback.print_exc()
        all_results['cv_ensemble'] = {'error': str(e)}

    # ---- Architecture 10: Physics-Residual Correction ----
    print("\n" + "=" * 70)
    print("ARCHITECTURE 10: Physics-Residual Correction Model")
    print("=" * 70)

    try:
        residual_model = PhysicsResidualModel()
        res_metrics = residual_model.train(
            X_train, Y_train_dict, X_val, Y_val_dict, feature_names,
        )
        residual_model.save(os.path.join(output_dir, 'physics_residual_model.pkl'))

        # Test evaluation
        res_test = residual_model.predict(X_test)

        if 'action_trustworthy' in res_test:
            # When model is confident physics is right, is it actually right?
            test_phys = PhysicsResidualModel.derive_physics_prediction(X_test, feature_names)
            test_action_correct = (test_phys['implied_action'] == Y_test_dict['optimal_action'].astype(int))

            high_trust = res_test['action_trustworthy'] > 0.7
            if high_trust.sum() > 5:
                res_metrics['test_high_trust_acc'] = float(np.mean(test_action_correct[high_trust]))
                res_metrics['test_high_trust_coverage'] = float(high_trust.mean())
                print(f"\n  Test high-trust accuracy: {res_metrics['test_high_trust_acc']:.1%} "
                      f"(coverage: {res_metrics['test_high_trust_coverage']:.1%})")

            low_trust = res_test['action_trustworthy'] < 0.3
            if low_trust.sum() > 5:
                res_metrics['test_low_trust_err'] = float(np.mean(~test_action_correct[low_trust]))
                res_metrics['test_low_trust_coverage'] = float(low_trust.mean())
                print(f"  Test low-trust error rate: {res_metrics['test_low_trust_err']:.1%} "
                      f"(coverage: {res_metrics['test_low_trust_coverage']:.1%})")

            # Overall test: does corrected confidence improve trade quality?
            res_metrics['test_action_correct_rate'] = float(test_action_correct.mean())

            test_bd_correct = (test_phys['implied_break_dir'] == Y_test_dict['break_direction'].astype(int))
            high_bd_trust = res_test['bd_trustworthy'] > 0.7
            if high_bd_trust.sum() > 5:
                res_metrics['test_bd_high_trust_acc'] = float(np.mean(test_bd_correct[high_bd_trust]))
                res_metrics['test_bd_high_trust_coverage'] = float(high_bd_trust.mean())
                print(f"  Test BD high-trust accuracy: {res_metrics['test_bd_high_trust_acc']:.1%} "
                      f"(coverage: {res_metrics['test_bd_high_trust_coverage']:.1%})")

        print(f"\n  Test physics action correct rate: {res_metrics.get('test_action_correct_rate', 'N/A')}")

        all_results['physics_residual'] = res_metrics
    except Exception as e:
        print(f"  Physics Residual failed: {e}")
        import traceback
        traceback.print_exc()
        all_results['physics_residual'] = {'error': str(e)}

    # ---- Architecture 11: Adverse Movement Predictor ----
    print("\n" + "=" * 70)
    print("ARCHITECTURE 11: Adverse Movement Predictor (Stop-Loss Avoidance)")
    print("=" * 70)

    try:
        adv_model = AdverseMovementPredictor()
        adv_metrics = adv_model.train(
            X_train, Y_train_dict, X_val, Y_val_dict, feature_names,
        )
        adv_model.save(os.path.join(output_dir, 'adverse_movement_model.pkl'))

        # Test evaluation
        for direction, is_buy in [('BUY', True), ('SELL', False)]:
            adv_test = adv_model.predict(X_test, is_buy=is_buy)
            if 'stop_prob' in adv_test:
                test_returns = Y_test_dict['future_return_20']
                if is_buy:
                    actual_stop = ((np.abs(np.minimum(Y_test_dict['future_return_5'], Y_test_dict['future_return_20'])) > 0.005) &
                                   (np.maximum(Y_test_dict['future_return_5'], Y_test_dict['future_return_20']) < 0.010))
                else:
                    actual_stop = ((np.maximum(Y_test_dict['future_return_5'], Y_test_dict['future_return_20']) > 0.005) &
                                   (np.abs(np.minimum(Y_test_dict['future_return_5'], Y_test_dict['future_return_20'])) < 0.010))

                safe = adv_test['stop_prob'] < 0.3
                if safe.sum() > 5:
                    key = f'test_{direction.lower()}_safe_stop_rate'
                    adv_metrics[key] = float(actual_stop[safe].mean())
                    print(f"\n  Test {direction} safe signals: stop_rate={adv_metrics[key]:.1%}, "
                          f"coverage={safe.mean():.1%}")

        all_results['adverse_movement'] = adv_metrics
    except Exception as e:
        print(f"  Adverse Movement failed: {e}")
        import traceback
        traceback.print_exc()
        all_results['adverse_movement'] = {'error': str(e)}

    # ---- Compare ----
    print("\n" + "=" * 70)
    print("ARCHITECTURE COMPARISON")
    print("=" * 70)

    compare_keys = ['lifetime_mae', 'test_lifetime_mae', 'break_dir_accuracy',
                    'test_break_dir_acc', 'action_accuracy', 'test_action_acc']

    header = f"{'Metric':<30} {'GBT':<15} {'Survival':<15} {'Transformer':<15}"
    print(header)
    print("-" * len(header))

    for key in compare_keys:
        gbt_val = all_results.get('gbt', {}).get(key, '-')
        surv_val = all_results.get('survival', {}).get(key, '-')
        trans_val = all_results.get('transformer', {}).get(key, '-')

        def fmt(v):
            if isinstance(v, float):
                if 'acc' in key:
                    return f"{v:.1%}"
                elif 'mae' in key:
                    return f"{v:.2f}"
                else:
                    return f"{v:.4f}"
            return str(v)

        print(f"{key:<30} {fmt(gbt_val):<15} {fmt(surv_val):<15} {fmt(trans_val):<15}")

    # Save comparison
    with open(os.path.join(output_dir, 'comparison.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Results saved to {output_dir}/comparison.json")

    # Find best model
    best_arch = None
    best_test_acc = 0
    for arch, m in all_results.items():
        if isinstance(m, dict) and 'test_action_acc' in m:
            if m['test_action_acc'] > best_test_acc:
                best_test_acc = m['test_action_acc']
                best_arch = arch

    if best_arch:
        print(f"\n  BEST MODEL: {best_arch} (test action accuracy: {best_test_acc:.1%})")
        # Symlink best model
        if best_arch == 'gbt':
            src = 'gbt_model.pkl'
        elif best_arch == 'survival':
            src = 'survival_model.pt'
        else:
            src = 'transformer_model.pt'

        best_link = os.path.join(output_dir, f'best_model.{"pkl" if "gbt" in src else "pt"}')
        if os.path.exists(best_link):
            os.remove(best_link)
        import shutil
        shutil.copy2(os.path.join(output_dir, src), best_link)

        # Save best model info
        with open(os.path.join(output_dir, 'best_model_info.json'), 'w') as f:
            json.dump({
                'architecture': best_arch,
                'test_action_accuracy': best_test_acc,
                'model_file': src,
                'metrics': all_results[best_arch],
            }, f, indent=2, default=str)

    return all_results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Channel Surfer ML Training')
    sub = parser.add_subparsers(dest='command')

    # Train command
    train_parser = sub.add_parser('train', help='Train ML models')
    train_parser.add_argument('--days', type=int, default=60,
                             help='Days of historical data (default: 60)')
    train_parser.add_argument('--arch', choices=['all', 'gbt', 'survival', 'transformer', 'quality', 'ensemble', 'regime', 'temporal', 'trend_gbt', 'cv_ensemble', 'physics_residual', 'adverse_movement', 'entry_timing', 'composite', 'vol_transition', 'exit_timing', 'exhaustion', 'cross_asset', 'stop_loss', 'bayesian', 'trail', 'session', 'maturity', 'momentum', 'asymmetry', 'gap_risk', 'reversion', 'liquidity', 'transition', 'profit_target', 'alignment', 'duration', 'winner', 'fractal', 'volume_conviction', 'energy_momentum', 'multi_exit', 'adversarial', 'cascade', 'knn', 'quantile_risk', 'tail_risk', 'drawdown_recovery', 'stop_distance', 'vol_clustering', 'extreme_loser', 'risk_reward', 'return_consistency', 'horizon_divergence', 'drawdown_magnitude', 'win_streak', 'reversal_proximity', 'vol_return_regime', 'multi_horizon_loser', 'bounce_loser', 'feature_interaction', 'momentum_reversal', 'immediate_stop', 'profit_velocity', 'breakout_stop', 'breakout_momentum'],
                             default='all', help='Architecture to train')
    train_parser.add_argument('--output', type=str, default='surfer_models',
                             help='Output directory')
    train_parser.add_argument('--eval-interval', type=int, default=3,
                             help='Bar evaluation interval')

    # Evaluate command
    eval_parser = sub.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--checkpoint', type=str, required=True)
    eval_parser.add_argument('--days', type=int, default=30)

    args = parser.parse_args()

    if args.command == 'train':
        if args.arch == 'all':
            train_all_architectures(
                days=args.days,
                output_dir=args.output,
                eval_interval=args.eval_interval,
            )
        else:
            # Train single architecture
            X, Y, feature_names = generate_training_data(
                days=args.days, eval_interval=args.eval_interval,
            )

            n = len(X)
            train_end = int(n * 0.6)
            val_end = int(n * 0.8)

            X_train = X[:train_end]
            Y_train = {k: v[:train_end] for k, v in Y.items()}
            X_val = X[train_end:val_end]
            Y_val = {k: v[train_end:val_end] for k, v in Y.items()}

            os.makedirs(args.output, exist_ok=True)

            if args.arch == 'gbt':
                model = GBTModel()
                model.train(X_train, Y_train, X_val, Y_val, feature_names)
                model.save(os.path.join(args.output, 'gbt_model.pkl'))
            elif args.arch == 'survival':
                model = SurvivalModel(input_dim=X_train.shape[1])
                model.train(X_train, Y_train, X_val, Y_val, feature_names)
                model.save(os.path.join(args.output, 'survival_model.pt'))
            elif args.arch == 'transformer':
                model = MultiTFTransformer()
                model.train(X_train, Y_train, X_val, Y_val, feature_names)
                model.save(os.path.join(args.output, 'transformer_model.pt'))
            elif args.arch == 'quality':
                scorer = TradeQualityScorer()
                tq_X, tq_Y, tq_features = scorer.generate_training_data(
                    X_base=X, Y_base=Y, base_feature_names=feature_names, verbose=True,
                )
                tq_n = len(tq_X)
                tq_train_end = int(tq_n * 0.6)
                tq_val_end = int(tq_n * 0.8)
                tq_X_train = tq_X[:tq_train_end]
                tq_Y_train = {k: v[:tq_train_end] for k, v in tq_Y.items()}
                tq_X_val = tq_X[tq_train_end:tq_val_end]
                tq_Y_val = {k: v[tq_train_end:tq_val_end] for k, v in tq_Y.items()}
                scorer.train(tq_X_train, tq_Y_train, tq_X_val, tq_Y_val, tq_features)
                scorer.save(os.path.join(args.output, 'trade_quality_scorer.pkl'))
            elif args.arch == 'ensemble':
                ensemble = EnsembleModel()
                ensemble_metrics = ensemble.train(
                    X_train, Y_train, X_val, Y_val, feature_names, args.output,
                )
                ensemble.save(os.path.join(args.output, 'ensemble_model.pkl'))
                print(f"\n  Ensemble metrics: {ensemble_metrics}")
            elif args.arch == 'regime':
                regime = RegimeConditionalModel()
                regime_metrics = regime.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                regime.save(os.path.join(args.output, 'regime_model.pkl'))
                print(f"\n  Regime metrics: {regime_metrics}")
            elif args.arch == 'temporal':
                ws = TemporalAttentionModel.WINDOW_SIZE
                ta = TemporalAttentionModel(n_features=X_train.shape[1], window_size=ws)
                # Try to load GBT importance for feature selection
                gbt_imp = None
                gbt_path = os.path.join(args.output, 'gbt_model.pkl')
                if os.path.exists(gbt_path):
                    try:
                        _gbt = GBTModel.load(gbt_path)
                        gbt_imp = _gbt.feature_importance.get('lifetime')
                        print("  Using GBT importance for feature selection")
                    except Exception:
                        pass
                ta_metrics = ta.train(X_train, Y_train, X_val, Y_val, feature_names,
                                      gbt_importance=gbt_imp)
                ta.save(os.path.join(args.output, 'temporal_attention_model.pt'))
                print(f"\n  Temporal Attention metrics: {ta_metrics}")
            elif args.arch == 'trend_gbt':
                trend = TrendGBTModel()
                gbt_imp = None
                gbt_path = os.path.join(args.output, 'gbt_model.pkl')
                if os.path.exists(gbt_path):
                    try:
                        _gbt = GBTModel.load(gbt_path)
                        gbt_imp = _gbt.feature_importance.get('lifetime')
                        print("  Using GBT importance for feature selection")
                    except Exception:
                        pass
                tg_metrics = trend.train(X_train, Y_train, X_val, Y_val, feature_names,
                                         gbt_importance=gbt_imp)
                trend.save(os.path.join(args.output, 'trend_gbt_model.pkl'))
                print(f"\n  Trend GBT metrics: {tg_metrics}")
            elif args.arch == 'cv_ensemble':
                cv = CVEnsembleModel()
                cv_metrics = cv.train(X, Y, feature_names)
                cv.save(os.path.join(args.output, 'cv_ensemble_model.pkl'))
                print(f"\n  CV Ensemble metrics: {cv_metrics}")
            elif args.arch == 'physics_residual':
                residual = PhysicsResidualModel()
                res_metrics = residual.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                residual.save(os.path.join(args.output, 'physics_residual_model.pkl'))
                print(f"\n  Physics Residual metrics: {res_metrics}")
            elif args.arch == 'adverse_movement':
                adv = AdverseMovementPredictor()
                adv_metrics = adv.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                adv.save(os.path.join(args.output, 'adverse_movement_model.pkl'))
                print(f"\n  Adverse Movement metrics: {adv_metrics}")
            elif args.arch == 'entry_timing':
                timing = EntryTimingOptimizer()
                timing_metrics = timing.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                timing.save(os.path.join(args.output, 'entry_timing_model.pkl'))
                print(f"\n  Entry Timing metrics: {timing_metrics}")
            elif args.arch == 'composite':
                composite = CompositeSignalScorer()
                comp_metrics = composite.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                    model_dir=args.output,
                )
                composite.save(os.path.join(args.output, 'composite_scorer.pkl'))
                print(f"\n  Composite Scorer metrics: {comp_metrics}")
            elif args.arch == 'vol_transition':
                vol = VolatilityTransitionModel()
                vol_metrics = vol.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                vol.save(os.path.join(args.output, 'vol_transition_model.pkl'))
                print(f"\n  Vol Transition metrics: {vol_metrics}")
            elif args.arch == 'exit_timing':
                exit_opt = ExitTimingOptimizer()
                exit_metrics = exit_opt.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                exit_opt.save(os.path.join(args.output, 'exit_timing_opt.pkl'))
                print(f"\n  Exit Timing metrics: {exit_metrics}")
            elif args.arch == 'exhaustion':
                exh = MomentumExhaustionDetector()
                exh_metrics = exh.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                exh.save(os.path.join(args.output, 'exhaustion_model.pkl'))
                print(f"\n  Exhaustion metrics: {exh_metrics}")
            elif args.arch == 'cross_asset':
                ca = CrossAssetAmplifier()
                ca_metrics = ca.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                ca.save(os.path.join(args.output, 'cross_asset_model.pkl'))
                print(f"\n  Cross-Asset metrics: {ca_metrics}")
            elif args.arch == 'stop_loss':
                slp = StopLossPredictor()
                slp_metrics = slp.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                slp.save(os.path.join(args.output, 'stop_loss_model.pkl'))
                print(f"\n  Stop Loss metrics: {slp_metrics}")
            elif args.arch == 'bayesian':
                bay = BayesianSignalCombiner()
                bay_metrics = bay.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                    model_dir=args.output,
                )
                bay.save(os.path.join(args.output, 'bayesian_combiner.pkl'))
                print(f"\n  Bayesian Combiner metrics: {bay_metrics}")
            elif args.arch == 'trail':
                trail = DynamicTrailOptimizer()
                trail_metrics = trail.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                trail.save(os.path.join(args.output, 'trail_optimizer.pkl'))
                print(f"\n  Trail Optimizer metrics: {trail_metrics}")
            elif args.arch == 'session':
                sess = IntradaySessionModel()
                sess_metrics = sess.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                sess.save(os.path.join(args.output, 'session_model.pkl'))
                print(f"\n  Session Model metrics: {sess_metrics}")
            elif args.arch == 'maturity':
                mat = ChannelMaturityPredictor()
                mat_metrics = mat.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                mat.save(os.path.join(args.output, 'maturity_model.pkl'))
                print(f"\n  Maturity Predictor metrics: {mat_metrics}")
            elif args.arch == 'momentum':
                mom = MultiScaleMomentumModel()
                mom_metrics = mom.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                mom.save(os.path.join(args.output, 'momentum_model.pkl'))
                print(f"\n  Momentum Model metrics: {mom_metrics}")
            elif args.arch == 'asymmetry':
                asym = ReturnAsymmetryPredictor()
                asym_metrics = asym.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                asym.save(os.path.join(args.output, 'asymmetry_model.pkl'))
                print(f"\n  Asymmetry Predictor metrics: {asym_metrics}")
            elif args.arch == 'gap_risk':
                gap = GapRiskPredictor()
                gap_metrics = gap.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                gap.save(os.path.join(args.output, 'gap_risk_model.pkl'))
                print(f"\n  Gap Risk metrics: {gap_metrics}")
            elif args.arch == 'reversion':
                rev = MeanReversionSpeedModel()
                rev_metrics = rev.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                rev.save(os.path.join(args.output, 'reversion_model.pkl'))
                print(f"\n  Reversion Speed metrics: {rev_metrics}")
            elif args.arch == 'liquidity':
                liq = LiquidityStateClassifier()
                liq_metrics = liq.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                liq.save(os.path.join(args.output, 'liquidity_model.pkl'))
                print(f"\n  Liquidity State metrics: {liq_metrics}")
            elif args.arch == 'transition':
                trans = RegimeTransitionDetector()
                trans_metrics = trans.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                trans.save(os.path.join(args.output, 'transition_model.pkl'))
                print(f"\n  Regime Transition metrics: {trans_metrics}")
            elif args.arch == 'profit_target':
                pt = ProfitTargetOptimizer()
                pt_metrics = pt.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                pt.save(os.path.join(args.output, 'profit_target_model.pkl'))
                print(f"\n  Profit Target metrics: {pt_metrics}")
            elif args.arch == 'alignment':
                align = ChannelAlignmentScorer()
                align_metrics = align.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                align.save(os.path.join(args.output, 'alignment_model.pkl'))
                print(f"\n  Alignment Scorer metrics: {align_metrics}")
            elif args.arch == 'duration':
                dur = TradeDurationPredictor()
                dur_metrics = dur.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                dur.save(os.path.join(args.output, 'duration_model.pkl'))
                print(f"\n  Trade Duration metrics: {dur_metrics}")
            elif args.arch == 'winner':
                win = WinnerAmplifier()
                win_metrics = win.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                win.save(os.path.join(args.output, 'winner_model.pkl'))
                print(f"\n  Winner Amplifier metrics: {win_metrics}")
            elif args.arch == 'fractal':
                frac = FractalRegimeClassifier()
                frac_metrics = frac.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                frac.save(os.path.join(args.output, 'fractal_model.pkl'))
                print(f"\n  Fractal Regime metrics: {frac_metrics}")
            elif args.arch == 'volume_conviction':
                vc = VolumeConvictionClassifier()
                vc_metrics = vc.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                vc.save(os.path.join(args.output, 'volume_conviction_model.pkl'))
                print(f"\n  Volume Conviction metrics: {vc_metrics}")
            elif args.arch == 'energy_momentum':
                em = EnergyMomentumDetector()
                em_metrics = em.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                em.save(os.path.join(args.output, 'energy_momentum_model.pkl'))
                print(f"\n  Energy Momentum metrics: {em_metrics}")
            elif args.arch == 'multi_exit':
                me = MultiExitStrategySelector()
                me_metrics = me.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                me.save(os.path.join(args.output, 'multi_exit_model.pkl'))
                print(f"\n  Multi-Exit Strategy metrics: {me_metrics}")
            elif args.arch == 'adversarial':
                adv = AdversarialTradeSelector()
                adv_metrics = adv.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                adv.save(os.path.join(args.output, 'adversarial_model.pkl'))
                print(f"\n  Adversarial Selector metrics: {adv_metrics}")
            elif args.arch == 'cascade':
                casc = CascadeConfidenceOptimizer()
                casc_metrics = casc.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                    model_dir=args.output,
                )
                casc.save(os.path.join(args.output, 'cascade_model.pkl'))
                print(f"\n  Cascade Confidence metrics: {casc_metrics}")
            elif args.arch == 'knn':
                knn = NearestNeighborTradeAnalogy()
                knn_metrics = knn.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                knn.save(os.path.join(args.output, 'knn_model.pkl'))
                print(f"\n  Nearest Neighbor metrics: {knn_metrics}")
            elif args.arch == 'quantile_risk':
                qr = QuantileRiskEstimator()
                qr_metrics = qr.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                qr.save(os.path.join(args.output, 'quantile_risk_model.pkl'))
                print(f"\n  Quantile Risk metrics: {qr_metrics}")
            elif args.arch == 'tail_risk':
                tr = TailRiskDetector()
                tr_metrics = tr.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                tr.save(os.path.join(args.output, 'tail_risk_model.pkl'))
                print(f"\n  Tail Risk metrics: {tr_metrics}")
            elif args.arch == 'drawdown_recovery':
                dr = DrawdownRecoveryPredictor()
                dr_metrics = dr.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                dr.save(os.path.join(args.output, 'drawdown_recovery_model.pkl'))
                print(f"\n  Drawdown Recovery metrics: {dr_metrics}")
            elif args.arch == 'stop_distance':
                sd = StopDistanceOptimizer()
                sd_metrics = sd.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                sd.save(os.path.join(args.output, 'stop_distance_model.pkl'))
                print(f"\n  Stop Distance metrics: {sd_metrics}")
            elif args.arch == 'vol_clustering':
                vc = VolatilityClusteringPredictor()
                vc_metrics = vc.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                vc.save(os.path.join(args.output, 'vol_clustering_model.pkl'))
                print(f"\n  Vol Clustering metrics: {vc_metrics}")
            elif args.arch == 'extreme_loser':
                el = ExtremeLoserDetector()
                el_metrics = el.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                el.save(os.path.join(args.output, 'extreme_loser_model.pkl'))
                print(f"\n  Extreme Loser metrics: {el_metrics}")
            elif args.arch == 'risk_reward':
                rr = RiskRewardClassifier()
                rr_metrics = rr.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                rr.save(os.path.join(args.output, 'risk_reward_model.pkl'))
                print(f"\n  Risk-Reward metrics: {rr_metrics}")
            elif args.arch == 'return_consistency':
                rc = ReturnConsistencyPredictor()
                rc_metrics = rc.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                rc.save(os.path.join(args.output, 'return_consistency_model.pkl'))
                print(f"\n  Return Consistency metrics: {rc_metrics}")
            elif args.arch == 'horizon_divergence':
                hd = HorizonDivergencePredictor()
                hd_metrics = hd.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                hd.save(os.path.join(args.output, 'horizon_divergence_model.pkl'))
                print(f"\n  Horizon Divergence metrics: {hd_metrics}")
            elif args.arch == 'drawdown_magnitude':
                dm = DrawdownMagnitudePredictor()
                dm_metrics = dm.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                dm.save(os.path.join(args.output, 'drawdown_magnitude_model.pkl'))
                print(f"\n  Drawdown Magnitude metrics: {dm_metrics}")
            elif args.arch == 'win_streak':
                ws = WinStreakDetector()
                ws_metrics = ws.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                ws.save(os.path.join(args.output, 'win_streak_model.pkl'))
                print(f"\n  Win Streak metrics: {ws_metrics}")
            elif args.arch == 'reversal_proximity':
                rp = ReversalProximityDetector()
                rp_metrics = rp.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                rp.save(os.path.join(args.output, 'reversal_proximity_model.pkl'))
                print(f"\n  Reversal Proximity metrics: {rp_metrics}")
            elif args.arch == 'vol_return_regime':
                vr = VolReturnRegimeClassifier()
                vr_metrics = vr.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                vr.save(os.path.join(args.output, 'vol_return_regime_model.pkl'))
                print(f"\n  Vol-Return Regime metrics: {vr_metrics}")
            elif args.arch == 'multi_horizon_loser':
                mh = MultiHorizonLoserDetector()
                mh_metrics = mh.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                mh.save(os.path.join(args.output, 'multi_horizon_loser_model.pkl'))
                print(f"\n  Multi-Horizon Loser metrics: {mh_metrics}")
            elif args.arch == 'bounce_loser':
                bl = BounceLoserDetector()
                bl_metrics = bl.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                bl.save(os.path.join(args.output, 'bounce_loser_model.pkl'))
                print(f"\n  Bounce Loser metrics: {bl_metrics}")
            elif args.arch == 'feature_interaction':
                fi = FeatureInteractionLoser()
                fi_metrics = fi.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                fi.save(os.path.join(args.output, 'feature_interaction_model.pkl'))
                print(f"\n  Feature Interaction metrics: {fi_metrics}")
            elif args.arch == 'momentum_reversal':
                mr = MomentumReversalDetector()
                mr_metrics = mr.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                mr.save(os.path.join(args.output, 'momentum_reversal_model.pkl'))
                print(f"\n  Momentum Reversal metrics: {mr_metrics}")
            elif args.arch == 'immediate_stop':
                isd = ImmediateStopDetector()
                isd_metrics = isd.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                isd.save(os.path.join(args.output, 'immediate_stop_model.pkl'))
                print(f"\n  Immediate Stop metrics: {isd_metrics}")
            elif args.arch == 'profit_velocity':
                pv = ProfitVelocityPredictor()
                pv_metrics = pv.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                pv.save(os.path.join(args.output, 'profit_velocity_model.pkl'))
                print(f"\n  Profit Velocity metrics: {pv_metrics}")

            elif args.arch == 'breakout_stop':
                bs = BreakoutStopPredictor()
                bs_metrics = bs.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                bs.save(os.path.join(args.output, 'breakout_stop_model.pkl'))
                print(f"\n  Breakout Stop metrics: {bs_metrics}")

            elif args.arch == 'breakout_momentum':
                bm = BreakoutMomentumValidator()
                bm_metrics = bm.train(
                    X_train, Y_train, X_val, Y_val, feature_names,
                )
                bm.save(os.path.join(args.output, 'breakout_momentum_model.pkl'))
                print(f"\n  Breakout Momentum metrics: {bm_metrics}")

    elif args.command == 'evaluate':
        print(f"Evaluating checkpoint: {args.checkpoint}")
        # Determine model type from file extension
        if args.checkpoint.endswith('.pkl'):
            model = GBTModel.load(args.checkpoint)
        elif 'survival' in args.checkpoint:
            model = SurvivalModel.load(args.checkpoint)
        else:
            model = MultiTFTransformer.load(args.checkpoint)

        # Generate test data
        X, Y, _ = generate_training_data(days=args.days, eval_interval=3)
        results = model.predict(X)

        print("\nEvaluation Results:")
        if 'lifetime' in results:
            mae = np.mean(np.abs(results['lifetime'] - Y['channel_lifetime']))
            print(f"  Lifetime MAE: {mae:.1f} bars")
        if 'break_dir' in results:
            acc = np.mean(results['break_dir'] == Y['break_direction'].astype(int))
            print(f"  Break Dir Accuracy: {acc:.1%}")
        if 'action' in results:
            acc = np.mean(results['action'] == Y['optimal_action'].astype(int))
            print(f"  Action Accuracy: {acc:.1%}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
