"""
v15/features/channel_history.py - Channel History Pattern Features

Tracks the pattern of the LAST 5 CHANNELS for each timeframe for both TSLA and SPY.
This module extracts 67 features from the historical channel patterns to help the model
learn from channel sequences and regime changes.

Includes historical exit features from ChannelLabels:
- Exit counts, magnitudes, and bars outside
- Durability scores and return rates
- False break counts (bounces after return)
- Trends in exit behavior over recent channels

All features return valid floats with safe defaults (no NaN, no Inf).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, TYPE_CHECKING

from .utils import safe_float, safe_divide, safe_mean, safe_std, safe_min, safe_max

if TYPE_CHECKING:
    from v7.core.channel import Channel
    from v15.dtypes import ChannelLabels


# =============================================================================
# Helper Functions
# =============================================================================

def channel_to_history_dict(
    channel: "Channel",
    duration: int,
    break_direction: int,
    labels: Optional["ChannelLabels"] = None
) -> Dict:
    """
    Extract channel info from a Channel object for history tracking.

    Args:
        channel: Channel object from v7.core.channel
        duration: Duration of the channel in bars
        break_direction: Direction of break (-1=down, 0=no break, 1=up)
        labels: Optional ChannelLabels with exit metrics for historical exit features

    Returns:
        Dict with keys: duration, slope, direction, break_direction, r_squared, bounce_count,
                        exit_count, avg_exit_magnitude, avg_bars_outside, exit_return_rate,
                        durability_score, false_break_count
    """
    # Default exit metrics
    exit_metrics = {
        'exit_count': 0.0,
        'avg_exit_magnitude': 0.0,
        'avg_bars_outside': 0.0,
        'exit_return_rate': 0.0,
        'durability_score': 0.0,
        'false_break_count': 0.0,
    }

    # Extract exit metrics from labels if provided
    if labels is not None:
        exit_bars = getattr(labels, 'exit_bars', []) or []
        exit_magnitudes = getattr(labels, 'exit_magnitudes', []) or []
        exit_count = len(exit_bars)

        exit_metrics['exit_count'] = safe_float(exit_count, 0.0)
        exit_metrics['avg_exit_magnitude'] = safe_mean(exit_magnitudes, default=0.0) if exit_magnitudes else 0.0
        exit_metrics['avg_bars_outside'] = safe_float(getattr(labels, 'avg_bars_outside', 0.0), 0.0)
        exit_metrics['exit_return_rate'] = safe_float(getattr(labels, 'exit_return_rate', 0.0), 0.0)
        exit_metrics['durability_score'] = safe_float(getattr(labels, 'durability_score', 0.0), 0.0)
        exit_metrics['false_break_count'] = safe_float(getattr(labels, 'bounces_after_return', 0), 0.0)

    if channel is None:
        base_dict = {
            'duration': safe_float(duration, 50.0),
            'slope': 0.0,
            'direction': 1,  # sideways
            'break_direction': safe_float(break_direction, 0),
            'r_squared': 0.0,
            'bounce_count': 0.0,
        }
        base_dict.update(exit_metrics)
        return base_dict

    base_dict = {
        'duration': safe_float(duration, getattr(channel, 'window', 50)),
        'slope': safe_float(getattr(channel, 'slope', 0.0), 0.0),
        'direction': int(getattr(channel, 'direction', 1)),  # 0=bear, 1=sideways, 2=bull
        'break_direction': safe_float(break_direction, 0),
        'r_squared': safe_float(getattr(channel, 'r_squared', 0.0), 0.0),
        'bounce_count': safe_float(getattr(channel, 'bounce_count', 0), 0.0),
    }
    base_dict.update(exit_metrics)
    return base_dict


def _encode_direction_sequence(directions: List[int]) -> int:
    """
    Encode a sequence of channel directions to a single value.

    Args:
        directions: List of direction values (0=bear, 1=sideways, 2=bull)

    Returns:
        0 = all bear
        1 = mostly bear (>60% bear)
        2 = mixed
        3 = mostly bull (>60% bull)
        4 = all bull
    """
    if not directions:
        return 2  # mixed/unknown

    bull_count = sum(1 for d in directions if d == 2)
    bear_count = sum(1 for d in directions if d == 0)
    total = len(directions)

    if bear_count == total:
        return 0  # all bear
    elif bull_count == total:
        return 4  # all bull
    elif bear_count / total > 0.6:
        return 1  # mostly bear
    elif bull_count / total > 0.6:
        return 3  # mostly bull
    else:
        return 2  # mixed


def _encode_break_sequence(break_directions: List[int]) -> int:
    """
    Encode a sequence of break directions to a single value.

    Args:
        break_directions: List of break directions (-1=down, 0=no break, 1=up)

    Returns:
        0 = all down breaks
        1 = mostly down (>60%)
        2 = mixed
        3 = mostly up (>60%)
        4 = all up breaks
    """
    if not break_directions:
        return 2  # mixed/unknown

    up_count = sum(1 for b in break_directions if b > 0)
    down_count = sum(1 for b in break_directions if b < 0)
    total = len(break_directions)

    if down_count == total:
        return 0  # all down
    elif up_count == total:
        return 4  # all up
    elif down_count / total > 0.6:
        return 1  # mostly down
    elif up_count / total > 0.6:
        return 3  # mostly up
    else:
        return 2  # mixed


def _calculate_alternating_score(directions: List[int]) -> float:
    """
    Calculate how alternating the direction sequence is.

    Args:
        directions: List of direction values

    Returns:
        Score from 0 (no alternation) to 1 (perfect alternation)
    """
    if len(directions) < 2:
        return 0.0

    alternations = 0
    for i in range(1, len(directions)):
        if directions[i] != directions[i-1]:
            alternations += 1

    max_alternations = len(directions) - 1
    return safe_divide(alternations, max_alternations, 0.0)


def _calculate_trend(values: List[float]) -> float:
    """
    Calculate the trend in a sequence of values.

    Args:
        values: List of values

    Returns:
        Slope of linear fit, normalized: positive = increasing, negative = decreasing
    """
    if len(values) < 2:
        return 0.0

    try:
        x = np.arange(len(values))
        y = np.array(values, dtype=np.float64)

        # Filter out non-finite values
        valid_mask = np.isfinite(y)
        if np.sum(valid_mask) < 2:
            return 0.0

        x_valid = x[valid_mask]
        y_valid = y[valid_mask]

        slope, _ = np.polyfit(x_valid, y_valid, 1)

        # Normalize by mean of values
        mean_val = safe_mean(y_valid, default=1.0)
        if mean_val == 0:
            mean_val = 1.0

        return safe_float(safe_divide(slope, abs(mean_val), 0.0), 0.0)
    except Exception:
        return 0.0


def _calculate_momentum(directions: List[int]) -> float:
    """
    Calculate momentum based on recent vs older channel directions.

    Args:
        directions: List of direction values (0=bear, 1=sideways, 2=bull)

    Returns:
        Positive = trending bullish, negative = trending bearish
    """
    if len(directions) < 3:
        return 0.0

    # Compare recent (last 2) vs older (first 2-3)
    recent = directions[-2:]
    older = directions[:-2] if len(directions) > 3 else directions[:2]

    recent_score = sum(d - 1 for d in recent) / len(recent)  # -1 to 1 scale
    older_score = sum(d - 1 for d in older) / len(older)

    return safe_float(recent_score - older_score, 0.0)


def _calculate_regime_shift(history: List[Dict]) -> float:
    """
    Detect if there's a regime shift in the channel history.

    Args:
        history: List of channel history dicts

    Returns:
        Score from 0 (stable regime) to 1 (clear regime shift)
    """
    if len(history) < 3:
        return 0.0

    # Compare first half vs second half directions
    mid = len(history) // 2
    first_half = [h['direction'] for h in history[:mid]]
    second_half = [h['direction'] for h in history[mid:]]

    first_avg = safe_mean(first_half, default=1.0)
    second_avg = safe_mean(second_half, default=1.0)

    # Large difference = regime shift
    diff = abs(second_avg - first_avg)

    # Normalize: max diff is 2 (from 0 to 2 or vice versa)
    return safe_float(safe_divide(diff, 2.0, 0.0), 0.0)


def _extract_single_history_features(
    history: List[Dict],
    prefix: str
) -> Dict[str, float]:
    """
    Extract features from a single asset's channel history.

    Args:
        history: List of up to 5 channel history dicts
        prefix: Prefix for feature names (e.g., 'tsla_' or 'spy_')

    Returns:
        Dict of features
    """
    features: Dict[str, float] = {}

    # Handle empty or None history
    if not history:
        history = []

    # Ensure we have at most 5 channels
    history = history[-5:] if len(history) > 5 else history
    n_channels = len(history)

    # Extract sequences
    durations = [h.get('duration', 50) for h in history]
    slopes = [h.get('slope', 0.0) for h in history]
    directions = [h.get('direction', 1) for h in history]
    break_directions = [h.get('break_direction', 0) for h in history]
    r_squareds = [h.get('r_squared', 0.0) for h in history]
    bounce_counts = [h.get('bounce_count', 0) for h in history]

    # ==========================================================================
    # Core Last 5 Channel Features
    # ==========================================================================

    # 1. Average duration of last 5 channels
    features[f'{prefix}last5_avg_duration'] = safe_mean(durations, default=50.0)

    # 2. Average slope of last 5 channels
    features[f'{prefix}last5_avg_slope'] = safe_mean(slopes, default=0.0)

    # 3. Direction sequence pattern (encoded)
    features[f'{prefix}last5_direction_pattern'] = float(_encode_direction_sequence(directions))

    # 4. Break direction sequence pattern (encoded)
    features[f'{prefix}last5_break_pattern'] = float(_encode_break_sequence(break_directions))

    # 5. Average quality (r_squared) of last 5 channels
    features[f'{prefix}last5_avg_quality'] = safe_mean(r_squareds, default=0.0)

    # ==========================================================================
    # Trend and Momentum Features
    # ==========================================================================

    # 6. Channel momentum (trending up/down based on sequence)
    features[f'{prefix}channel_momentum'] = _calculate_momentum(directions)

    # 7. Slope trend (accelerating/decelerating)
    features[f'{prefix}last5_slope_trend'] = _calculate_trend(slopes)

    # 8. Duration trend (getting longer/shorter)
    features[f'{prefix}last5_duration_trend'] = _calculate_trend(durations)

    # 9. Quality trend (improving/deteriorating)
    features[f'{prefix}last5_quality_trend'] = _calculate_trend(r_squareds)

    # 10. Regime shift indicator
    features[f'{prefix}channel_regime_shift'] = _calculate_regime_shift(history)

    # ==========================================================================
    # Pattern Features
    # ==========================================================================

    # 11. Alternating pattern score
    features[f'{prefix}alternating_pattern'] = _calculate_alternating_score(directions)

    # 12. Break alternating pattern
    features[f'{prefix}break_alternating'] = _calculate_alternating_score(break_directions)

    # 13. Consecutive same direction count
    consecutive_same = 0
    if directions:
        last_dir = directions[-1]
        for d in reversed(directions):
            if d == last_dir:
                consecutive_same += 1
            else:
                break
    features[f'{prefix}consecutive_same_dir'] = float(consecutive_same)

    # 14. Consecutive same break direction count
    consecutive_same_break = 0
    if break_directions:
        last_break = break_directions[-1]
        for b in reversed(break_directions):
            if b == last_break:
                consecutive_same_break += 1
            else:
                break
    features[f'{prefix}consecutive_same_break'] = float(consecutive_same_break)

    # ==========================================================================
    # Statistical Features
    # ==========================================================================

    # 15. Duration standard deviation (channel length consistency)
    features[f'{prefix}last5_duration_std'] = safe_std(durations, default=0.0)

    # 16. Slope standard deviation (slope consistency)
    features[f'{prefix}last5_slope_std'] = safe_std(slopes, default=0.0)

    # 17. Quality standard deviation (quality consistency)
    features[f'{prefix}last5_quality_std'] = safe_std(r_squareds, default=0.0)

    # 18. Min duration in last 5
    features[f'{prefix}last5_min_duration'] = safe_min(durations, default=50.0)

    # 19. Max duration in last 5
    features[f'{prefix}last5_max_duration'] = safe_max(durations, default=50.0)

    # 20. Duration range (max - min)
    features[f'{prefix}last5_duration_range'] = features[f'{prefix}last5_max_duration'] - features[f'{prefix}last5_min_duration']

    # ==========================================================================
    # Recent vs Historical Comparison
    # ==========================================================================

    # 21. Most recent channel duration vs average
    if durations:
        recent_dur = durations[-1]
        avg_dur = features[f'{prefix}last5_avg_duration']
        features[f'{prefix}recent_vs_avg_duration'] = safe_divide(recent_dur - avg_dur, avg_dur + 1, 0.0)
    else:
        features[f'{prefix}recent_vs_avg_duration'] = 0.0

    # 22. Most recent channel slope vs average
    if slopes:
        recent_slope = slopes[-1]
        avg_slope = features[f'{prefix}last5_avg_slope']
        features[f'{prefix}recent_vs_avg_slope'] = safe_float(recent_slope - avg_slope, 0.0)
    else:
        features[f'{prefix}recent_vs_avg_slope'] = 0.0

    # 23. Most recent quality vs average
    if r_squareds:
        recent_qual = r_squareds[-1]
        avg_qual = features[f'{prefix}last5_avg_quality']
        features[f'{prefix}recent_vs_avg_quality'] = safe_float(recent_qual - avg_qual, 0.0)
    else:
        features[f'{prefix}recent_vs_avg_quality'] = 0.0

    # ==========================================================================
    # Directional Features
    # ==========================================================================

    # 24. Bull channel ratio in last 5
    bull_count = sum(1 for d in directions if d == 2)
    features[f'{prefix}bull_channel_ratio'] = safe_divide(bull_count, max(n_channels, 1), 0.0)

    # 25. Bear channel ratio in last 5
    bear_count = sum(1 for d in directions if d == 0)
    features[f'{prefix}bear_channel_ratio'] = safe_divide(bear_count, max(n_channels, 1), 0.0)

    # 26. Up break ratio in last 5
    up_break_count = sum(1 for b in break_directions if b > 0)
    features[f'{prefix}up_break_ratio'] = safe_divide(up_break_count, max(n_channels, 1), 0.0)

    # 27. Down break ratio in last 5
    down_break_count = sum(1 for b in break_directions if b < 0)
    features[f'{prefix}down_break_ratio'] = safe_divide(down_break_count, max(n_channels, 1), 0.0)

    # ==========================================================================
    # Composite Scores
    # ==========================================================================

    # 28. Channel stability score (consistent quality and duration)
    qual_consistency = 1.0 - min(features[f'{prefix}last5_quality_std'] / 0.5, 1.0)  # 0.5 is high std
    dur_consistency = 1.0 - min(features[f'{prefix}last5_duration_std'] / 50, 1.0)  # 50 bars is high std
    features[f'{prefix}channel_stability_score'] = safe_float((qual_consistency + dur_consistency) / 2, 0.5)

    # 29. Trend strength score (based on direction and slope consistency)
    dir_strength = abs(features[f'{prefix}bull_channel_ratio'] - features[f'{prefix}bear_channel_ratio'])
    slope_consistency = 1.0 - min(features[f'{prefix}last5_slope_std'] / 0.1, 1.0)  # Normalize
    features[f'{prefix}trend_strength_score'] = safe_float((dir_strength + slope_consistency) / 2, 0.0)

    # 30. Average bounce count
    features[f'{prefix}last5_avg_bounces'] = safe_mean(bounce_counts, default=0.0)

    # ==========================================================================
    # Historical Exit Features (from ChannelLabels exit metrics)
    # ==========================================================================

    # Extract exit metric sequences from history
    exit_counts = [h.get('exit_count', 0.0) for h in history]
    avg_exit_magnitudes = [h.get('avg_exit_magnitude', 0.0) for h in history]
    avg_bars_outside_list = [h.get('avg_bars_outside', 0.0) for h in history]
    exit_return_rates = [h.get('exit_return_rate', 0.0) for h in history]
    durability_scores = [h.get('durability_score', 0.0) for h in history]
    false_break_counts = [h.get('false_break_count', 0.0) for h in history]

    # 31. Average exits per channel in last 5
    features[f'{prefix}last5_avg_exit_count'] = safe_mean(exit_counts, default=0.0)

    # 32. Average exit magnitude across last 5 channels
    features[f'{prefix}last5_avg_exit_magnitude'] = safe_mean(avg_exit_magnitudes, default=0.0)

    # 33. Average time outside before return across last 5 channels
    features[f'{prefix}last5_avg_bars_outside'] = safe_mean(avg_bars_outside_list, default=0.0)

    # 34. Average exit return rate (how often exits return) across last 5
    features[f'{prefix}last5_avg_exit_return_rate'] = safe_mean(exit_return_rates, default=0.0)

    # 35. Average durability score across last 5 channels
    features[f'{prefix}last5_avg_durability'] = safe_mean(durability_scores, default=0.0)

    # 36. Average false breaks per channel in last 5
    features[f'{prefix}last5_avg_false_breaks'] = safe_mean(false_break_counts, default=0.0)

    # 37. Exit count trend - are channels getting more/fewer exits?
    features[f'{prefix}exit_count_trend'] = _calculate_trend(exit_counts)

    # 38. Durability trend - are channels getting more/less resilient?
    features[f'{prefix}durability_trend'] = _calculate_trend(durability_scores)

    # 39. Bars outside trend - are exits lasting longer/shorter?
    features[f'{prefix}bars_outside_trend'] = _calculate_trend(avg_bars_outside_list)

    # 40. Exit return rate trend - is return rate changing?
    features[f'{prefix}exit_return_rate_trend'] = _calculate_trend(exit_return_rates)

    return features


# =============================================================================
# Main Feature Extraction Function
# =============================================================================

def extract_channel_history_features(
    tsla_channel_history: List[Dict],
    spy_channel_history: List[Dict]
) -> Dict[str, float]:
    """
    Extract 67 channel history features from TSLA and SPY channel histories.

    This function analyzes the pattern of the last 5 channels for each asset,
    extracting features that capture momentum, regime shifts, cross-asset
    alignment patterns, and historical exit behavior.

    Args:
        tsla_channel_history: List of last 5 channel info dicts for TSLA
            Each dict should have: duration, slope, direction, break_direction, r_squared, bounce_count,
            exit_count, avg_exit_magnitude, avg_bars_outside, exit_return_rate, durability_score, false_break_count
        spy_channel_history: List of last 5 channel info dicts for SPY
            Each dict should have: duration, slope, direction, break_direction, r_squared, bounce_count,
            exit_count, avg_exit_magnitude, avg_bars_outside, exit_return_rate, durability_score, false_break_count

    Returns:
        Dict[str, float] with 67 features, all guaranteed to be valid floats
    """
    features: Dict[str, float] = {}

    # Handle None inputs
    tsla_history = tsla_channel_history if tsla_channel_history else []
    spy_history = spy_channel_history if spy_channel_history else []

    # ==========================================================================
    # TSLA Channel History Features (1-40)
    # ==========================================================================
    tsla_features = _extract_single_history_features(tsla_history, 'tsla_')
    features.update(tsla_features)

    # ==========================================================================
    # SPY Channel History Features (41-80, but we'll merge and reduce)
    # ==========================================================================
    spy_features = _extract_single_history_features(spy_history, 'spy_')
    features.update(spy_features)

    # ==========================================================================
    # Cross-Asset Alignment Features (to reach exactly 60)
    # ==========================================================================

    # Get direction patterns for alignment calculation
    tsla_directions = [h.get('direction', 1) for h in tsla_history[-5:]] if tsla_history else []
    spy_directions = [h.get('direction', 1) for h in spy_history[-5:]] if spy_history else []

    # 1. Channel alignment score (are TSLA and SPY showing similar direction patterns)
    if tsla_directions and spy_directions:
        min_len = min(len(tsla_directions), len(spy_directions))
        matches = sum(1 for i in range(min_len) if tsla_directions[-(i+1)] == spy_directions[-(i+1)])
        features['tsla_spy_channel_alignment'] = safe_divide(matches, min_len, 0.5)
    else:
        features['tsla_spy_channel_alignment'] = 0.5

    # 2. Momentum alignment (are both trending in same direction)
    tsla_mom = features.get('tsla_channel_momentum', 0.0)
    spy_mom = features.get('spy_channel_momentum', 0.0)
    if (tsla_mom > 0 and spy_mom > 0) or (tsla_mom < 0 and spy_mom < 0):
        features['channel_momentum_alignment'] = 1.0
    elif tsla_mom * spy_mom < 0:  # Opposite directions
        features['channel_momentum_alignment'] = 0.0
    else:
        features['channel_momentum_alignment'] = 0.5

    # 3. Break pattern alignment
    tsla_breaks = [h.get('break_direction', 0) for h in tsla_history[-5:]] if tsla_history else []
    spy_breaks = [h.get('break_direction', 0) for h in spy_history[-5:]] if spy_history else []

    if tsla_breaks and spy_breaks:
        min_len = min(len(tsla_breaks), len(spy_breaks))
        matches = sum(1 for i in range(min_len)
                     if (tsla_breaks[-(i+1)] > 0) == (spy_breaks[-(i+1)] > 0) or
                        (tsla_breaks[-(i+1)] < 0) == (spy_breaks[-(i+1)] < 0))
        features['break_pattern_alignment'] = safe_divide(matches, min_len, 0.5)
    else:
        features['break_pattern_alignment'] = 0.5

    # 4. Quality spread (TSLA quality - SPY quality)
    tsla_qual = features.get('tsla_last5_avg_quality', 0.0)
    spy_qual = features.get('spy_last5_avg_quality', 0.0)
    features['quality_spread'] = safe_float(tsla_qual - spy_qual, 0.0)

    # 5. Duration spread (TSLA avg duration - SPY avg duration), normalized
    tsla_dur = features.get('tsla_last5_avg_duration', 50.0)
    spy_dur = features.get('spy_last5_avg_duration', 50.0)
    features['duration_spread'] = safe_divide(tsla_dur - spy_dur, (tsla_dur + spy_dur) / 2 + 1, 0.0)

    # 6. Slope spread (TSLA avg slope - SPY avg slope)
    tsla_slope = features.get('tsla_last5_avg_slope', 0.0)
    spy_slope = features.get('spy_last5_avg_slope', 0.0)
    features['slope_spread'] = safe_float(tsla_slope - spy_slope, 0.0)

    # 7. Combined regime shift score
    tsla_regime = features.get('tsla_channel_regime_shift', 0.0)
    spy_regime = features.get('spy_channel_regime_shift', 0.0)
    features['combined_regime_shift'] = safe_float((tsla_regime + spy_regime) / 2, 0.0)

    # 8. Divergence indicator (TSLA vs SPY momentum difference)
    features['momentum_divergence'] = safe_float(abs(tsla_mom - spy_mom), 0.0)

    # 9. TSLA leading indicator (is TSLA regime shift leading SPY)
    # Positive = TSLA changing first, negative = SPY changing first
    features['tsla_leading_indicator'] = safe_float(tsla_regime - spy_regime, 0.0)

    # 10. Combined trend strength
    tsla_trend = features.get('tsla_trend_strength_score', 0.0)
    spy_trend = features.get('spy_trend_strength_score', 0.0)
    features['combined_trend_strength'] = safe_float((tsla_trend + spy_trend) / 2, 0.0)

    # ==========================================================================
    # Cross-Asset Exit Features (historical exit pattern comparison)
    # ==========================================================================

    # 11. Exit count spread - TSLA vs SPY exit count difference
    tsla_exit_count = features.get('tsla_last5_avg_exit_count', 0.0)
    spy_exit_count = features.get('spy_last5_avg_exit_count', 0.0)
    features['exit_count_spread'] = safe_float(tsla_exit_count - spy_exit_count, 0.0)

    # 12. Durability spread - TSLA vs SPY durability difference
    tsla_durability = features.get('tsla_last5_avg_durability', 0.0)
    spy_durability = features.get('spy_last5_avg_durability', 0.0)
    features['durability_spread_avg'] = safe_float(tsla_durability - spy_durability, 0.0)

    # 13. Exit alignment - do exit patterns correlate?
    # Calculate correlation of exit count trends
    tsla_exit_trend = features.get('tsla_exit_count_trend', 0.0)
    spy_exit_trend = features.get('spy_exit_count_trend', 0.0)
    # If both trending same direction (both positive or both negative), alignment is high
    if (tsla_exit_trend > 0 and spy_exit_trend > 0) or (tsla_exit_trend < 0 and spy_exit_trend < 0):
        features['exit_alignment'] = 1.0
    elif tsla_exit_trend * spy_exit_trend < 0:  # Opposite directions
        features['exit_alignment'] = 0.0
    else:
        features['exit_alignment'] = 0.5

    # ==========================================================================
    # Final Feature Selection (67 features)
    # ==========================================================================
    # We have 40 TSLA features + 40 SPY features + 13 cross-asset = 93
    # Selecting the most important 67 features

    # Define the final 67 features to keep
    final_feature_names = [
        # TSLA Core (5)
        'tsla_last5_avg_duration',
        'tsla_last5_avg_slope',
        'tsla_last5_direction_pattern',
        'tsla_last5_break_pattern',
        'tsla_last5_avg_quality',
        # SPY Core (5)
        'spy_last5_avg_duration',
        'spy_last5_avg_slope',
        'spy_last5_direction_pattern',
        'spy_last5_break_pattern',
        'spy_last5_avg_quality',
        # TSLA Trends (5)
        'tsla_channel_momentum',
        'tsla_last5_slope_trend',
        'tsla_last5_duration_trend',
        'tsla_last5_quality_trend',
        'tsla_channel_regime_shift',
        # SPY Trends (5)
        'spy_channel_momentum',
        'spy_last5_slope_trend',
        'spy_last5_duration_trend',
        'spy_last5_quality_trend',
        'spy_channel_regime_shift',
        # TSLA Patterns (5)
        'tsla_alternating_pattern',
        'tsla_consecutive_same_dir',
        'tsla_consecutive_same_break',
        'tsla_bull_channel_ratio',
        'tsla_bear_channel_ratio',
        # SPY Patterns (5)
        'spy_alternating_pattern',
        'spy_consecutive_same_dir',
        'spy_consecutive_same_break',
        'spy_bull_channel_ratio',
        'spy_bear_channel_ratio',
        # TSLA Stats (5)
        'tsla_last5_duration_std',
        'tsla_last5_slope_std',
        'tsla_up_break_ratio',
        'tsla_down_break_ratio',
        'tsla_channel_stability_score',
        # SPY Stats (5)
        'spy_last5_duration_std',
        'spy_last5_slope_std',
        'spy_up_break_ratio',
        'spy_down_break_ratio',
        'spy_channel_stability_score',
        # TSLA Historical Exit Features (5)
        'tsla_last5_avg_exit_count',
        'tsla_last5_avg_exit_magnitude',
        'tsla_last5_avg_bars_outside',
        'tsla_last5_avg_exit_return_rate',
        'tsla_last5_avg_durability',
        # SPY Historical Exit Features (5)
        'spy_last5_avg_exit_count',
        'spy_last5_avg_exit_magnitude',
        'spy_last5_avg_bars_outside',
        'spy_last5_avg_exit_return_rate',
        'spy_last5_avg_durability',
        # TSLA Exit Trends (2)
        'tsla_exit_count_trend',
        'tsla_durability_trend',
        # SPY Exit Trends (2)
        'spy_exit_count_trend',
        'spy_durability_trend',
        # Cross-Asset (10)
        'tsla_spy_channel_alignment',
        'channel_momentum_alignment',
        'break_pattern_alignment',
        'quality_spread',
        'duration_spread',
        'slope_spread',
        'combined_regime_shift',
        'momentum_divergence',
        'tsla_leading_indicator',
        'combined_trend_strength',
        # Cross-Asset Exit Features (3)
        'exit_count_spread',
        'durability_spread_avg',
        'exit_alignment',
    ]

    # Build final feature dict with exactly 67 features
    final_features: Dict[str, float] = {}
    for name in final_feature_names:
        final_features[name] = features.get(name, 0.0)

    # Final safety check - ensure all values are valid floats
    for key, value in final_features.items():
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            final_features[key] = 0.0
        else:
            final_features[key] = float(value)

    return final_features


# =============================================================================
# Utility Functions
# =============================================================================

def get_channel_history_feature_names() -> List[str]:
    """Get ordered list of all 67 channel history feature names."""
    return [
        # TSLA Core (5)
        'tsla_last5_avg_duration',
        'tsla_last5_avg_slope',
        'tsla_last5_direction_pattern',
        'tsla_last5_break_pattern',
        'tsla_last5_avg_quality',
        # SPY Core (5)
        'spy_last5_avg_duration',
        'spy_last5_avg_slope',
        'spy_last5_direction_pattern',
        'spy_last5_break_pattern',
        'spy_last5_avg_quality',
        # TSLA Trends (5)
        'tsla_channel_momentum',
        'tsla_last5_slope_trend',
        'tsla_last5_duration_trend',
        'tsla_last5_quality_trend',
        'tsla_channel_regime_shift',
        # SPY Trends (5)
        'spy_channel_momentum',
        'spy_last5_slope_trend',
        'spy_last5_duration_trend',
        'spy_last5_quality_trend',
        'spy_channel_regime_shift',
        # TSLA Patterns (5)
        'tsla_alternating_pattern',
        'tsla_consecutive_same_dir',
        'tsla_consecutive_same_break',
        'tsla_bull_channel_ratio',
        'tsla_bear_channel_ratio',
        # SPY Patterns (5)
        'spy_alternating_pattern',
        'spy_consecutive_same_dir',
        'spy_consecutive_same_break',
        'spy_bull_channel_ratio',
        'spy_bear_channel_ratio',
        # TSLA Stats (5)
        'tsla_last5_duration_std',
        'tsla_last5_slope_std',
        'tsla_up_break_ratio',
        'tsla_down_break_ratio',
        'tsla_channel_stability_score',
        # SPY Stats (5)
        'spy_last5_duration_std',
        'spy_last5_slope_std',
        'spy_up_break_ratio',
        'spy_down_break_ratio',
        'spy_channel_stability_score',
        # TSLA Historical Exit Features (5)
        'tsla_last5_avg_exit_count',
        'tsla_last5_avg_exit_magnitude',
        'tsla_last5_avg_bars_outside',
        'tsla_last5_avg_exit_return_rate',
        'tsla_last5_avg_durability',
        # SPY Historical Exit Features (5)
        'spy_last5_avg_exit_count',
        'spy_last5_avg_exit_magnitude',
        'spy_last5_avg_bars_outside',
        'spy_last5_avg_exit_return_rate',
        'spy_last5_avg_durability',
        # TSLA Exit Trends (2)
        'tsla_exit_count_trend',
        'tsla_durability_trend',
        # SPY Exit Trends (2)
        'spy_exit_count_trend',
        'spy_durability_trend',
        # Cross-Asset (10)
        'tsla_spy_channel_alignment',
        'channel_momentum_alignment',
        'break_pattern_alignment',
        'quality_spread',
        'duration_spread',
        'slope_spread',
        'combined_regime_shift',
        'momentum_divergence',
        'tsla_leading_indicator',
        'combined_trend_strength',
        # Cross-Asset Exit Features (3)
        'exit_count_spread',
        'durability_spread_avg',
        'exit_alignment',
    ]


def get_channel_history_feature_count() -> int:
    """Get total number of channel history features."""
    return 67


def get_default_channel_history_features() -> Dict[str, float]:
    """Get default feature values for when no history is available."""
    features = {}
    for name in get_channel_history_feature_names():
        if 'direction_pattern' in name or 'break_pattern' in name:
            features[name] = 2.0  # mixed pattern
        elif 'avg_duration' in name or 'min_duration' in name or 'max_duration' in name:
            features[name] = 50.0  # default duration
        elif 'alignment' in name or 'stability' in name:
            features[name] = 0.5  # neutral
        else:
            features[name] = 0.0
    return features


# =============================================================================
# TF-Prefixed Feature Extraction Functions
# =============================================================================

def extract_channel_history_features_tf(
    tsla_channel_history: List[Dict],
    spy_channel_history: List[Dict],
    tf: str
) -> Dict[str, float]:
    """
    Extract channel history features with TF prefix.

    Channel history should be tracked PER TIMEFRAME, not globally.
    This function wraps extract_channel_history_features and adds a TF prefix
    to all feature names.

    Args:
        tsla_channel_history: Last 5 TSLA channels for THIS TF
        spy_channel_history: Last 5 SPY channels for THIS TF
        tf: Timeframe name (e.g., 'daily', '1h', '5min')

    Returns:
        Dict with keys like 'daily_tsla_last5_avg_duration', '1h_spy_channel_momentum',
        'weekly_tsla_spy_channel_alignment'

    Example usage:
        channel_history_by_tf = {
            '5min': {'tsla': [...], 'spy': [...]},
            'daily': {'tsla': [...], 'spy': [...]},
            # etc.
        }

        for tf, histories in channel_history_by_tf.items():
            tf_features = extract_channel_history_features_tf(
                histories['tsla'], histories['spy'], tf
            )
            all_features.update(tf_features)
    """
    base_features = extract_channel_history_features(tsla_channel_history, spy_channel_history)
    prefix = f"{tf}_"
    return {f"{prefix}{k}": v for k, v in base_features.items()}


def get_channel_history_feature_names_tf(tf: str) -> List[str]:
    """
    Get feature names with TF prefix.

    Args:
        tf: Timeframe name (e.g., 'daily', '1h', '5min')

    Returns:
        List of feature names with TF prefix applied
        e.g., ['daily_tsla_last5_avg_duration', 'daily_tsla_last5_avg_slope', ...]
    """
    base_names = get_channel_history_feature_names()
    prefix = f"{tf}_"
    return [f"{prefix}{name}" for name in base_names]


def get_all_channel_history_feature_names() -> List[str]:
    """
    Get ALL channel history feature names across all TFs.

    Returns:
        List of all 670 feature names (67 features * 10 timeframes)
    """
    from v15.config import TIMEFRAMES

    all_names = []
    for tf in TIMEFRAMES:
        all_names.extend(get_channel_history_feature_names_tf(tf))
    return all_names


def get_total_channel_history_features() -> int:
    """
    Total channel history features: 67 * 10 TFs = 670

    Returns:
        670 (67 base features * 10 timeframes)
    """
    return 67 * 10


def get_default_channel_history_features_tf(tf: str) -> Dict[str, float]:
    """
    Get default feature values for a specific TF when no history is available.

    Args:
        tf: Timeframe name (e.g., 'daily', '1h', '5min')

    Returns:
        Dict with TF-prefixed feature names and default values
    """
    base_defaults = get_default_channel_history_features()
    prefix = f"{tf}_"
    return {f"{prefix}{k}": v for k, v in base_defaults.items()}


def get_all_default_channel_history_features() -> Dict[str, float]:
    """
    Get default feature values for ALL TFs when no history is available.

    Returns:
        Dict with all 670 TF-prefixed features set to default values
    """
    from v15.config import TIMEFRAMES

    all_defaults = {}
    for tf in TIMEFRAMES:
        all_defaults.update(get_default_channel_history_features_tf(tf))
    return all_defaults
