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

        # Feature importance for lifetime
        imp = self.models['lifetime'].feature_importance(importance_type='gain')
        top_idx = np.argsort(imp)[-10:][::-1]
        self.feature_importance['lifetime'] = [
            (feature_names[i], float(imp[i])) for i in top_idx
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
    train_parser.add_argument('--arch', choices=['all', 'gbt', 'survival', 'transformer', 'quality', 'ensemble'],
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
