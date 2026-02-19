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
    train_parser.add_argument('--arch', choices=['all', 'gbt', 'survival', 'transformer', 'quality', 'ensemble', 'regime', 'temporal', 'trend_gbt', 'cv_ensemble', 'physics_residual', 'adverse_movement', 'entry_timing', 'composite', 'vol_transition', 'exit_timing', 'exhaustion', 'cross_asset'],
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
