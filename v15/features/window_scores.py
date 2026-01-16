"""
v15/features/window_scores.py - Multi-Window Channel Score Features

Extracts 50 window score and alignment features from multi-window channel detection.
These features capture how different window sizes agree or disagree, and which
windows are showing the strongest signals.

Standard windows: [10, 20, 30, 40, 50, 60, 70, 80]

All features return valid floats with safe defaults (no NaN, no Inf).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, TYPE_CHECKING

from .utils import safe_float, safe_divide, safe_mean, safe_std, safe_min, safe_max

if TYPE_CHECKING:
    from v7.core.channel import Channel

# Standard window sizes for multi-window channel detection
STANDARD_WINDOWS: List[int] = [10, 20, 30, 40, 50, 60, 70, 80]
NUM_WINDOWS: int = len(STANDARD_WINDOWS)


def _get_channel_score(channel: Optional["Channel"]) -> float:
    """Calculate quality score for a channel: r_squared * bounce_count."""
    if channel is None:
        return 0.0

    is_valid = getattr(channel, 'valid', False)
    if not is_valid:
        return 0.0

    r_squared = safe_float(getattr(channel, 'r_squared', 0.0), 0.0)
    bounce_count = safe_float(getattr(channel, 'bounce_count', 0), 0.0)

    return safe_float(r_squared * bounce_count, 0.0)


def _is_channel_valid(channel: Optional["Channel"]) -> bool:
    """Check if a channel is valid."""
    if channel is None:
        return False
    return bool(getattr(channel, 'valid', False))


def _get_channel_r_squared(channel: Optional["Channel"]) -> float:
    """Get r_squared from channel, 0.0 if invalid."""
    if channel is None or not _is_channel_valid(channel):
        return 0.0
    return safe_float(getattr(channel, 'r_squared', 0.0), 0.0)


def _get_channel_slope(channel: Optional["Channel"]) -> float:
    """Get slope from channel, 0.0 if invalid."""
    if channel is None or not _is_channel_valid(channel):
        return 0.0
    return safe_float(getattr(channel, 'slope', 0.0), 0.0)


def _get_channel_bounce_count(channel: Optional["Channel"]) -> float:
    """Get bounce_count from channel, 0.0 if invalid."""
    if channel is None or not _is_channel_valid(channel):
        return 0.0
    return safe_float(getattr(channel, 'bounce_count', 0), 0.0)


def _get_channel_direction(channel: Optional["Channel"]) -> int:
    """Get direction from channel: 0=bear, 1=sideways, 2=bull."""
    if channel is None or not _is_channel_valid(channel):
        return 1  # Default to sideways
    direction = getattr(channel, 'direction', 1)
    return int(direction)


def _normalize_window_size(window: int) -> float:
    """Normalize window size to 0-1 range where 10=0, 80=1."""
    return safe_divide(window - 10, 70, 0.5)


def extract_window_score_features(
    channels: Dict[int, "Channel"],
    best_window: int
) -> Dict[str, float]:
    """
    Extract 50 window score and alignment features from multi-window channel detection.

    Args:
        channels: Dict mapping window size -> Channel object
        best_window: The window size selected as best

    Returns:
        Dict[str, float] with 50 features, all guaranteed to be valid floats
    """
    features: Dict[str, float] = {}

    # Handle None or empty channels dict
    if channels is None:
        channels = {}

    # Ensure best_window is valid
    if best_window not in STANDARD_WINDOWS:
        best_window = 50  # Default to middle window

    # Pre-compute validity and scores for all windows
    validity: Dict[int, bool] = {}
    scores: Dict[int, float] = {}
    r_squared_vals: Dict[int, float] = {}
    slopes: Dict[int, float] = {}
    bounce_counts: Dict[int, float] = {}
    directions: Dict[int, int] = {}

    for window in STANDARD_WINDOWS:
        channel = channels.get(window)
        validity[window] = _is_channel_valid(channel)
        scores[window] = _get_channel_score(channel)
        r_squared_vals[window] = _get_channel_r_squared(channel)
        slopes[window] = _get_channel_slope(channel)
        bounce_counts[window] = _get_channel_bounce_count(channel)
        directions[window] = _get_channel_direction(channel)

    # ==========================================================================
    # 1. PER-WINDOW VALIDITY (8 features)
    # ==========================================================================
    for window in STANDARD_WINDOWS:
        features[f'window_{window}_valid'] = 1.0 if validity[window] else 0.0

    # ==========================================================================
    # 2. PER-WINDOW SCORES (8 features)
    # ==========================================================================
    for window in STANDARD_WINDOWS:
        features[f'window_{window}_score'] = scores[window]

    # ==========================================================================
    # 3. ALIGNMENT FEATURES (15 features)
    # ==========================================================================

    # Count valid windows
    valid_windows = [w for w in STANDARD_WINDOWS if validity[w]]
    valid_count = len(valid_windows)

    # 3.1 valid_window_count
    features['valid_window_count'] = safe_float(valid_count, 0.0)

    # 3.2 valid_window_ratio
    features['valid_window_ratio'] = safe_divide(valid_count, NUM_WINDOWS, 0.0)

    # 3.3 all_windows_agree_direction
    if valid_count > 0:
        valid_directions = [directions[w] for w in valid_windows]
        all_same = all(d == valid_directions[0] for d in valid_directions)
        features['all_windows_agree_direction'] = 1.0 if all_same else 0.0
    else:
        features['all_windows_agree_direction'] = 0.0

    # 3.4 direction_consensus (% of valid windows with same direction as best)
    best_direction = directions[best_window]
    if valid_count > 0:
        same_direction_count = sum(1 for w in valid_windows if directions[w] == best_direction)
        features['direction_consensus'] = safe_divide(same_direction_count, valid_count, 0.0)
    else:
        features['direction_consensus'] = 0.0

    # 3.5 slope_consensus (std dev of slopes across valid windows)
    if valid_count >= 2:
        valid_slopes = [slopes[w] for w in valid_windows]
        features['slope_consensus'] = safe_std(valid_slopes, 0.0)
    else:
        features['slope_consensus'] = 0.0

    # 3.6 avg_r_squared_all_valid
    if valid_count > 0:
        valid_r_squared = [r_squared_vals[w] for w in valid_windows]
        features['avg_r_squared_all_valid'] = safe_mean(valid_r_squared, 0.0)
    else:
        features['avg_r_squared_all_valid'] = 0.0

    # 3.7 best_window_score_ratio (best score / avg score)
    all_scores = list(scores.values())
    avg_score = safe_mean(all_scores, 0.0)
    best_score = scores[best_window]
    features['best_window_score_ratio'] = safe_divide(best_score, avg_score, 1.0)

    # 3.8 window_spread (largest - smallest valid window)
    if valid_count >= 2:
        features['window_spread'] = safe_float(max(valid_windows) - min(valid_windows), 0.0)
    else:
        features['window_spread'] = 0.0

    # 3.9 small_windows_valid (10, 20, 30 valid count)
    small_windows = [10, 20, 30]
    small_valid_count = sum(1 for w in small_windows if validity[w])
    features['small_windows_valid'] = safe_float(small_valid_count, 0.0)

    # 3.10 large_windows_valid (60, 70, 80 valid count)
    large_windows = [60, 70, 80]
    large_valid_count = sum(1 for w in large_windows if validity[w])
    features['large_windows_valid'] = safe_float(large_valid_count, 0.0)

    # 3.11 small_vs_large_bias (which size range has more valid)
    # -1 = all small, 0 = balanced, 1 = all large
    total_extremes = small_valid_count + large_valid_count
    if total_extremes > 0:
        features['small_vs_large_bias'] = safe_divide(
            large_valid_count - small_valid_count, total_extremes, 0.0
        )
    else:
        features['small_vs_large_bias'] = 0.0

    # 3.12 consecutive_valid_windows (longest streak of consecutive valid windows)
    max_consecutive = 0
    current_consecutive = 0
    for window in STANDARD_WINDOWS:
        if validity[window]:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    features['consecutive_valid_windows'] = safe_float(max_consecutive, 0.0)

    # 3.13 window_gap_pattern (pattern of gaps in validity - encoded as ratio)
    # Count transitions from valid to invalid
    gap_count = 0
    for i in range(1, len(STANDARD_WINDOWS)):
        prev_valid = validity[STANDARD_WINDOWS[i - 1]]
        curr_valid = validity[STANDARD_WINDOWS[i]]
        if prev_valid != curr_valid:
            gap_count += 1
    # Normalize by max possible transitions (7)
    features['window_gap_pattern'] = safe_divide(gap_count, 7, 0.0)

    # 3.14 multi_scale_alignment (do small and large windows agree)
    small_directions = [directions[w] for w in small_windows if validity[w]]
    large_directions = [directions[w] for w in large_windows if validity[w]]
    if small_directions and large_directions:
        # Check if majority of each group agrees
        small_majority = max(set(small_directions), key=small_directions.count, default=1)
        large_majority = max(set(large_directions), key=large_directions.count, default=1)
        features['multi_scale_alignment'] = 1.0 if small_majority == large_majority else 0.0
    else:
        features['multi_scale_alignment'] = 0.5  # Neutral if not enough data

    # 3.15 fractal_score (self-similarity across scales)
    # Measure how consistent scores are across adjacent window sizes
    if valid_count >= 3:
        score_diffs = []
        for i in range(len(STANDARD_WINDOWS) - 1):
            w1, w2 = STANDARD_WINDOWS[i], STANDARD_WINDOWS[i + 1]
            if validity[w1] and validity[w2]:
                diff = abs(scores[w1] - scores[w2])
                score_diffs.append(diff)
        if score_diffs:
            avg_diff = safe_mean(score_diffs, 0.0)
            max_score = safe_max(all_scores, 1.0)
            # Lower diff relative to max = more fractal (self-similar)
            features['fractal_score'] = 1.0 - safe_divide(avg_diff, max_score, 0.0)
            features['fractal_score'] = max(0.0, features['fractal_score'])
        else:
            features['fractal_score'] = 0.0
    else:
        features['fractal_score'] = 0.0

    # ==========================================================================
    # 4. BEST WINDOW FEATURES (10 features)
    # ==========================================================================

    best_channel = channels.get(best_window)

    # 4.1 best_window_size (normalized 0-1 where 10=0, 80=1)
    features['best_window_size'] = _normalize_window_size(best_window)

    # 4.2 best_window_r_squared
    features['best_window_r_squared'] = r_squared_vals[best_window]

    # 4.3 best_window_bounce_count
    features['best_window_bounce_count'] = bounce_counts[best_window]

    # 4.4 best_window_slope
    features['best_window_slope'] = slopes[best_window]

    # 4.5 best_window_direction
    features['best_window_direction'] = safe_float(directions[best_window], 1.0)

    # 4.6 best_vs_second_best_score_gap
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) >= 2:
        score_gap = sorted_scores[0] - sorted_scores[1]
        features['best_vs_second_best_score_gap'] = safe_float(score_gap, 0.0)
    else:
        features['best_vs_second_best_score_gap'] = 0.0

    # 4.7 best_window_is_smallest
    features['best_window_is_smallest'] = 1.0 if best_window == min(STANDARD_WINDOWS) else 0.0

    # 4.8 best_window_is_largest
    features['best_window_is_largest'] = 1.0 if best_window == max(STANDARD_WINDOWS) else 0.0

    # 4.9 best_window_position (1-8 in sorted order by score)
    sorted_by_score = sorted(
        [(w, scores[w]) for w in STANDARD_WINDOWS],
        key=lambda x: x[1],
        reverse=True
    )
    best_position = next(
        (i + 1 for i, (w, _) in enumerate(sorted_by_score) if w == best_window),
        4  # Default to middle position
    )
    features['best_window_position'] = safe_float(best_position, 4.0)

    # 4.10 best_window_dominance (how much better than others)
    # Ratio of best score to sum of all other scores
    other_scores_sum = sum(s for w, s in scores.items() if w != best_window)
    features['best_window_dominance'] = safe_divide(best_score, other_scores_sum, 0.0)

    # ==========================================================================
    # 5. TREND ACROSS WINDOWS (9 features)
    # ==========================================================================

    # Prepare arrays for trend analysis
    window_indices = np.arange(NUM_WINDOWS, dtype=np.float64)
    slope_array = np.array([slopes[w] for w in STANDARD_WINDOWS], dtype=np.float64)
    r_squared_array = np.array([r_squared_vals[w] for w in STANDARD_WINDOWS], dtype=np.float64)
    bounce_array = np.array([bounce_counts[w] for w in STANDARD_WINDOWS], dtype=np.float64)
    score_array = np.array([scores[w] for w in STANDARD_WINDOWS], dtype=np.float64)

    # 5.1 slope_trend_across_windows (correlation of slope with window size)
    if valid_count >= 3:
        valid_indices = np.array([STANDARD_WINDOWS.index(w) for w in valid_windows], dtype=np.float64)
        valid_slope_array = np.array([slopes[w] for w in valid_windows], dtype=np.float64)
        if np.std(valid_slope_array) > 1e-10 and len(valid_indices) >= 2:
            try:
                corr = np.corrcoef(valid_indices, valid_slope_array)[0, 1]
                features['slope_trend_across_windows'] = safe_float(corr, 0.0)
            except Exception:
                features['slope_trend_across_windows'] = 0.0
        else:
            features['slope_trend_across_windows'] = 0.0
    else:
        features['slope_trend_across_windows'] = 0.0

    # 5.2 r_squared_trend_across_windows
    if valid_count >= 3:
        valid_r_array = np.array([r_squared_vals[w] for w in valid_windows], dtype=np.float64)
        if np.std(valid_r_array) > 1e-10 and len(valid_indices) >= 2:
            try:
                corr = np.corrcoef(valid_indices, valid_r_array)[0, 1]
                features['r_squared_trend_across_windows'] = safe_float(corr, 0.0)
            except Exception:
                features['r_squared_trend_across_windows'] = 0.0
        else:
            features['r_squared_trend_across_windows'] = 0.0
    else:
        features['r_squared_trend_across_windows'] = 0.0

    # 5.3 bounce_count_trend_across_windows
    if valid_count >= 3:
        valid_bounce_array = np.array([bounce_counts[w] for w in valid_windows], dtype=np.float64)
        if np.std(valid_bounce_array) > 1e-10 and len(valid_indices) >= 2:
            try:
                corr = np.corrcoef(valid_indices, valid_bounce_array)[0, 1]
                features['bounce_count_trend_across_windows'] = safe_float(corr, 0.0)
            except Exception:
                features['bounce_count_trend_across_windows'] = 0.0
        else:
            features['bounce_count_trend_across_windows'] = 0.0
    else:
        features['bounce_count_trend_across_windows'] = 0.0

    # 5.4 window_size_quality_correlation (correlation between window size and score)
    if valid_count >= 3:
        valid_score_array = np.array([scores[w] for w in valid_windows], dtype=np.float64)
        valid_sizes = np.array(valid_windows, dtype=np.float64)
        if np.std(valid_score_array) > 1e-10:
            try:
                corr = np.corrcoef(valid_sizes, valid_score_array)[0, 1]
                features['window_size_quality_correlation'] = safe_float(corr, 0.0)
            except Exception:
                features['window_size_quality_correlation'] = 0.0
        else:
            features['window_size_quality_correlation'] = 0.0
    else:
        features['window_size_quality_correlation'] = 0.0

    # 5.5 convergence_score (are all windows pointing to same price target)
    # Measure variance in slope * window combinations (projected endpoints)
    if valid_count >= 2:
        projected_targets = []
        for w in valid_windows:
            # Project where price would be based on slope
            target = slopes[w] * w  # Simplified projection
            projected_targets.append(target)

        if projected_targets:
            target_std = safe_std(projected_targets, 0.0)
            target_mean = abs(safe_mean(projected_targets, 1.0))
            # Lower std relative to mean = more convergence
            normalized_std = safe_divide(target_std, target_mean + 1.0, 1.0)
            features['convergence_score'] = max(0.0, 1.0 - normalized_std)
        else:
            features['convergence_score'] = 0.0
    else:
        features['convergence_score'] = 0.0

    # 5.6 divergence_warning (windows contradicting each other)
    # High when valid windows have opposite directions
    if valid_count >= 2:
        unique_directions = set(directions[w] for w in valid_windows)
        # 1.0 if we have both bullish and bearish, 0.5 if sideways involved
        if 0 in unique_directions and 2 in unique_directions:
            features['divergence_warning'] = 1.0
        elif len(unique_directions) > 1:
            features['divergence_warning'] = 0.5
        else:
            features['divergence_warning'] = 0.0
    else:
        features['divergence_warning'] = 0.0

    # 5.7 multi_timeframe_momentum
    # Average slope weighted by validity and score
    if valid_count > 0:
        weighted_slope_sum = sum(slopes[w] * scores[w] for w in valid_windows)
        score_sum = sum(scores[w] for w in valid_windows)
        features['multi_timeframe_momentum'] = safe_divide(weighted_slope_sum, score_sum, 0.0)
    else:
        features['multi_timeframe_momentum'] = 0.0

    # 5.8 window_regime (small-dominant, balanced, large-dominant)
    # Encoded: -1 = small-dominant, 0 = balanced, 1 = large-dominant
    if valid_count > 0:
        small_score_sum = sum(scores[w] for w in small_windows if validity[w])
        large_score_sum = sum(scores[w] for w in large_windows if validity[w])
        mid_windows = [40, 50]
        mid_score_sum = sum(scores[w] for w in mid_windows if validity[w])

        total_score = small_score_sum + mid_score_sum + large_score_sum
        if total_score > 0:
            # Calculate weighted position
            weighted_pos = safe_divide(
                large_score_sum - small_score_sum, total_score, 0.0
            )
            features['window_regime'] = safe_float(weighted_pos, 0.0)
        else:
            features['window_regime'] = 0.0
    else:
        features['window_regime'] = 0.0

    # 5.9 confidence_score (overall multi-window confidence)
    # Composite of: valid_ratio, direction_consensus, avg_r_squared, best_dominance
    confidence = (
        features['valid_window_ratio'] * 0.25 +
        features['direction_consensus'] * 0.25 +
        features['avg_r_squared_all_valid'] * 0.25 +
        min(features['best_window_dominance'], 1.0) * 0.25
    )
    features['confidence_score'] = safe_float(confidence, 0.0)

    # ==========================================================================
    # FINAL SAFETY CHECK
    # ==========================================================================
    for key, value in features.items():
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            features[key] = 0.0

    return features


def _get_default_features() -> Dict[str, float]:
    """Return default feature values for empty/invalid input."""
    features = {}

    # Per-window validity (8 features)
    for window in STANDARD_WINDOWS:
        features[f'window_{window}_valid'] = 0.0

    # Per-window scores (8 features)
    for window in STANDARD_WINDOWS:
        features[f'window_{window}_score'] = 0.0

    # Alignment features (15 features)
    features['valid_window_count'] = 0.0
    features['valid_window_ratio'] = 0.0
    features['all_windows_agree_direction'] = 0.0
    features['direction_consensus'] = 0.0
    features['slope_consensus'] = 0.0
    features['avg_r_squared_all_valid'] = 0.0
    features['best_window_score_ratio'] = 1.0
    features['window_spread'] = 0.0
    features['small_windows_valid'] = 0.0
    features['large_windows_valid'] = 0.0
    features['small_vs_large_bias'] = 0.0
    features['consecutive_valid_windows'] = 0.0
    features['window_gap_pattern'] = 0.0
    features['multi_scale_alignment'] = 0.5
    features['fractal_score'] = 0.0

    # Best window features (10 features)
    features['best_window_size'] = 0.5
    features['best_window_r_squared'] = 0.0
    features['best_window_bounce_count'] = 0.0
    features['best_window_slope'] = 0.0
    features['best_window_direction'] = 1.0
    features['best_vs_second_best_score_gap'] = 0.0
    features['best_window_is_smallest'] = 0.0
    features['best_window_is_largest'] = 0.0
    features['best_window_position'] = 4.0
    features['best_window_dominance'] = 0.0

    # Trend across windows (9 features)
    features['slope_trend_across_windows'] = 0.0
    features['r_squared_trend_across_windows'] = 0.0
    features['bounce_count_trend_across_windows'] = 0.0
    features['window_size_quality_correlation'] = 0.0
    features['convergence_score'] = 0.0
    features['divergence_warning'] = 0.0
    features['multi_timeframe_momentum'] = 0.0
    features['window_regime'] = 0.0
    features['confidence_score'] = 0.0

    return features


def get_window_score_feature_names() -> List[str]:
    """Get ordered list of all window score feature names."""
    return list(_get_default_features().keys())


def get_window_score_feature_count() -> int:
    """Get total number of window score features."""
    return len(_get_default_features())


# Feature name list for import
WINDOW_SCORE_FEATURE_NAMES: List[str] = get_window_score_feature_names()


# =============================================================================
# TF-PREFIXED FEATURE EXTRACTION (for multi-timeframe support)
# =============================================================================

def extract_window_score_features_tf(
    channels: Dict[int, "Channel"],
    best_window: int,
    tf: str
) -> Dict[str, float]:
    """
    Extract window score features with TF prefix.

    Args:
        channels: Dict mapping window size -> Channel object (detected on TF data)
        best_window: The best performing window for this TF
        tf: Timeframe name (e.g., 'daily', '1h', '5min')

    Returns:
        Dict with keys like 'daily_valid_window_count', '1h_direction_consensus', 'weekly_confidence_score'
    """
    base_features = extract_window_score_features(channels, best_window)
    prefix = f"{tf}_"
    return {f"{prefix}{k}": v for k, v in base_features.items()}


def get_window_score_feature_names_tf(tf: str) -> List[str]:
    """Get feature names with TF prefix."""
    base_names = get_window_score_feature_names()
    prefix = f"{tf}_"
    return [f"{prefix}{name}" for name in base_names]


def get_all_window_score_feature_names() -> List[str]:
    """Get ALL window score feature names across all TFs."""
    from v7.core.timeframe import TIMEFRAMES

    all_names = []
    for tf in TIMEFRAMES:
        all_names.extend(get_window_score_feature_names_tf(tf))
    return all_names


def get_total_window_score_features() -> int:
    """Total window score features: 50 * 11 TFs = 550"""
    return 50 * 11
