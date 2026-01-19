"""
TF-Aware Feature Extraction Orchestrator for x14 V15

This module provides comprehensive multi-timeframe feature extraction by:
1. Resampling base 5-min data to each of 10 timeframes
2. Detecting channels at all 8 windows on each TF's resampled data
3. Extracting ALL features from each TF's data with explicit TF prefixes

Total Features: 13,660
- 282 window-independent features * 10 TFs = 2,820
- 58 TSLA channel features * 8 windows * 10 TFs = 4,640
- 58 SPY channel features * 8 windows * 10 TFs = 4,640
- 50 channel correlation features * 10 TFs = 500
- 50 window score features * 10 TFs = 500
- 50 channel history features * 10 TFs = 500
- 30 event features (TF-independent) = 30
- 30 bar metadata features (3 per TF * 10 TFs) = 30

All features are explicitly named with TF prefix (e.g., 'daily_rsi_14', '1h_w50_channel_slope').
"""

from __future__ import annotations

import gc
import logging
from typing import Dict, List, Optional, TYPE_CHECKING, Any, Tuple

import numpy as np
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)

# =============================================================================
# Type checking imports
# =============================================================================
if TYPE_CHECKING:
    from v7.core.channel import Channel

# =============================================================================
# Core imports from v7 and v15
# =============================================================================

# Import timeframe utilities
from v15.config import TIMEFRAMES, BARS_PER_TF as CONFIG_BARS_PER_TF

try:
    from v7.core.timeframe import resample_ohlc
    _timeframe_available = True
except ImportError as e:
    logger.warning(f"Failed to import resample_ohlc from v7.core.timeframe: {e}")
    _timeframe_available = False

# Import partial bar resampling from v15
try:
    from ..data.resampler import resample_with_partial
    _partial_resampling_available = True
except ImportError as e:
    logger.warning(f"Failed to import partial resampling: {e}")
    _partial_resampling_available = False

# Use BARS_PER_TF from v15.config (imported above as CONFIG_BARS_PER_TF)
BARS_PER_TF = CONFIG_BARS_PER_TF

# Resample rule mapping
TF_TO_RESAMPLE_RULE = {
    '5min': '5min',
    '15min': '15min',
    '30min': '30min',
    '1h': '1h',
    '2h': '2h',
    '3h': '3h',
    '4h': '4h',
    'daily': '1D',
    'weekly': '1W',
    'monthly': '1MS',
}

# Import STANDARD_WINDOWS from v15.config
from v15.config import STANDARD_WINDOWS

# Import channel detection
try:
    from v7.core.channel import detect_channels_multi_window, select_best_channel
    _channel_available = True
except ImportError as e:
    logger.warning(f"Failed to import channel detection: {e}")
    _channel_available = False

# =============================================================================
# Feature extractor imports
# =============================================================================

from .utils import safe_float, ensure_finite

# Import individual feature extractors
try:
    from .tsla_price import extract_tsla_price_features
    _tsla_price_available = True
except ImportError as e:
    logger.warning(f"Failed to import tsla_price: {e}")
    _tsla_price_available = False

try:
    from .technical import extract_technical_features
    _technical_available = True
except ImportError as e:
    logger.warning(f"Failed to import technical: {e}")
    _technical_available = False

try:
    from .spy import extract_spy_features
    _spy_available = True
except ImportError as e:
    logger.warning(f"Failed to import spy: {e}")
    _spy_available = False

try:
    from .vix import extract_vix_features
    _vix_available = True
except ImportError as e:
    logger.warning(f"Failed to import vix: {e}")
    _vix_available = False

try:
    from .cross_asset import extract_cross_asset_features
    _cross_asset_available = True
except ImportError as e:
    logger.warning(f"Failed to import cross_asset: {e}")
    _cross_asset_available = False

try:
    from .tsla_channel import extract_tsla_channel_features
    _tsla_channel_available = True
except ImportError as e:
    logger.warning(f"Failed to import tsla_channel: {e}")
    _tsla_channel_available = False

try:
    from .spy_channel import extract_spy_channel_features
    _spy_channel_available = True
except ImportError as e:
    logger.warning(f"Failed to import spy_channel: {e}")
    _spy_channel_available = False

try:
    from .channel_correlation import extract_channel_correlation_features
    _channel_correlation_available = True
except ImportError as e:
    logger.warning(f"Failed to import channel_correlation: {e}")
    _channel_correlation_available = False

try:
    from .window_scores import extract_window_score_features
    _window_scores_available = True
except ImportError as e:
    logger.warning(f"Failed to import window_scores: {e}")
    _window_scores_available = False

try:
    from .channel_history import extract_channel_history_features
    _channel_history_available = True
except ImportError as e:
    logger.warning(f"Failed to import channel_history: {e}")
    _channel_history_available = False

try:
    from .events import extract_event_features
    _events_available = True
except ImportError as e:
    logger.warning(f"Failed to import events: {e}")
    _events_available = False


# =============================================================================
# Constants
# =============================================================================

# Feature counts per category
TF_FEATURE_COUNTS = {
    'price_per_tf': 60,               # tsla_price features
    'technical_per_tf': 77,           # technical indicators
    'spy_per_tf': 80,                 # SPY features
    'vix_per_tf': 25,                 # VIX features
    'cross_asset_per_tf': 40,         # correlations
    'channel_per_window': 58,         # TSLA channel (50 base + 8 excursion)
    'spy_channel_per_window': 58,     # SPY channel (50 base + 8 excursion)
    'channel_correlation_per_tf': 50, # Cross-correlation features
    'window_scores_per_tf': 50,       # window score features
    'channel_history_per_tf': 50,     # channel history features
    'events': 30,                     # TF-independent
}

# Calculated totals
TOTAL_PER_TF_WINDOW_INDEPENDENT = (
    TF_FEATURE_COUNTS['price_per_tf'] +
    TF_FEATURE_COUNTS['technical_per_tf'] +
    TF_FEATURE_COUNTS['spy_per_tf'] +
    TF_FEATURE_COUNTS['vix_per_tf'] +
    TF_FEATURE_COUNTS['cross_asset_per_tf']
)  # 282

# TSLA channel features per TF: 58 * 8 windows = 464
TOTAL_TSLA_CHANNEL_PER_TF = TF_FEATURE_COUNTS['channel_per_window'] * len(STANDARD_WINDOWS)  # 464

# SPY channel features per TF: 58 * 8 windows = 464
TOTAL_SPY_CHANNEL_PER_TF = TF_FEATURE_COUNTS['spy_channel_per_window'] * len(STANDARD_WINDOWS)  # 464

TOTAL_PER_TF_WINDOW_DEPENDENT = TOTAL_TSLA_CHANNEL_PER_TF + TOTAL_SPY_CHANNEL_PER_TF  # 928

TOTAL_PER_TF = (
    TOTAL_PER_TF_WINDOW_INDEPENDENT +
    TOTAL_PER_TF_WINDOW_DEPENDENT +
    TF_FEATURE_COUNTS['channel_correlation_per_tf'] +
    TF_FEATURE_COUNTS['window_scores_per_tf'] +
    TF_FEATURE_COUNTS['channel_history_per_tf']
)  # 1,310

TOTAL_FEATURES = TOTAL_PER_TF * len(TIMEFRAMES) + TF_FEATURE_COUNTS['events']  # 13,130 + 30 = 13,160


# =============================================================================
# Helper Functions
# =============================================================================

def _prefix_features(features: Dict[str, float], prefix: str) -> Dict[str, float]:
    """
    Add prefix to all feature names.

    Args:
        features: Dictionary of features
        prefix: Prefix to add to all keys

    Returns:
        Dictionary with prefixed keys
    """
    return {f"{prefix}{k}": v for k, v in features.items()}


def _prefix_tf(features: Dict[str, float], tf: str) -> Dict[str, float]:
    """
    Add TF prefix to features.

    Args:
        features: Dictionary of features
        tf: Timeframe name (e.g., 'daily', '1h')

    Returns:
        Dictionary with TF-prefixed keys (e.g., 'daily_rsi_14')
    """
    return _prefix_features(features, f"{tf}_")


def _prefix_tf_window(features: Dict[str, float], tf: str, window: int) -> Dict[str, float]:
    """
    Add TF and window prefix to features.

    Args:
        features: Dictionary of features
        tf: Timeframe name
        window: Window size

    Returns:
        Dictionary with TF+window prefixed keys (e.g., 'daily_w50_channel_slope')
    """
    return _prefix_features(features, f"{tf}_w{window}_")


def _resample_to_tf(
    df: pd.DataFrame,
    tf: str,
    source_bar_count: Optional[int] = None
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Resample DataFrame to target timeframe with partial bar support.

    CRITICAL: This keeps partial (incomplete) bars and calculates completion
    metadata based on where we are within the TF bar period.

    Args:
        df: Base OHLCV DataFrame (5-min data)
        tf: Target timeframe
        source_bar_count: Number of 5min bars from start of data to sample point.
                         Used to calculate bar_completion_pct based on position
                         within the TF bar. If None, uses len(df).

    Returns:
        Tuple of (resampled_df, metadata_dict) where metadata contains:
        - bar_completion_pct: 0.0-1.0, how complete the current TF bar is
        - bars_in_partial: Number of 5min bars in the partial bar
        - is_partial: Whether the last bar is incomplete
    """
    if source_bar_count is None:
        source_bar_count = len(df)

    bars_per_tf_bar = BARS_PER_TF.get(tf, 1)

    if tf == '5min':
        # No resampling needed - always complete
        return df, {
            'bar_completion_pct': 1.0,
            'bars_in_partial': 1,
            'expected_bars': 1,
            'is_partial': False,
            'total_bars': len(df),
            'source_bars': len(df),
        }

    # Calculate completion based on position within TF bar
    # At 5min index i, we're (i % bars_per_tf_bar) bars into the current TF bar
    bars_into_current = source_bar_count % bars_per_tf_bar
    if bars_into_current == 0:
        # Exactly at a TF boundary - bar is complete
        completion_pct = 1.0
        is_partial = False
    else:
        # Partial bar
        completion_pct = bars_into_current / bars_per_tf_bar
        is_partial = True

    # Use partial resampling if available
    if _partial_resampling_available:
        try:
            resample_rule = TF_TO_RESAMPLE_RULE.get(tf, tf)
            resampled, raw_metadata = resample_with_partial(df, resample_rule)

            # Override the completion_pct with our position-based calculation
            # This ensures bar_completion_pct reflects actual position within the bar
            metadata = {
                'bar_completion_pct': round(completion_pct, 4),
                'bars_in_partial': bars_into_current if is_partial else bars_per_tf_bar,
                'expected_bars': bars_per_tf_bar,
                'is_partial': is_partial,
                'total_bars': len(resampled),
                'source_bars': len(df),
            }

            if len(resampled) < 2:
                logger.debug(f"Not enough data after resampling to {tf}: {len(resampled)} bars")
                # Return empty dataframe to indicate insufficient data
                # DO NOT return original df - that causes channel detection to fail silently
                metadata['total_bars'] = len(resampled)
                metadata['insufficient_data'] = True
                return resampled, metadata

            return resampled, metadata

        except Exception as e:
            logger.warning(f"Failed to resample to {tf} with partial support: {e}")

    # Fallback to v7 resampling
    if _timeframe_available:
        try:
            resampled = resample_ohlc(df, tf)
            metadata = {
                'bar_completion_pct': round(completion_pct, 4),
                'bars_in_partial': bars_into_current if is_partial else bars_per_tf_bar,
                'expected_bars': bars_per_tf_bar,
                'is_partial': is_partial,
                'total_bars': len(resampled),
                'source_bars': len(df),
            }
            if len(resampled) < 2:
                logger.debug(f"Not enough data after resampling to {tf}: {len(resampled)} bars")
                # Return the resampled data (even if small) - not original df
                metadata['insufficient_data'] = True
                return resampled, metadata
            return resampled, metadata
        except Exception as e:
            logger.warning(f"Failed to resample to {tf}: {e}")

    # Return empty dataframe with default metadata to indicate resampling failed completely
    # DO NOT return original df - that would cause wrong data to be used for feature extraction
    logger.warning(f"All resampling methods failed for {tf}, returning empty dataframe")
    return pd.DataFrame(columns=df.columns), {
        'bar_completion_pct': round(completion_pct, 4),
        'bars_in_partial': bars_into_current if is_partial else bars_per_tf_bar,
        'expected_bars': bars_per_tf_bar,
        'is_partial': is_partial,
        'total_bars': 0,
        'source_bars': len(df),
        'insufficient_data': True,
    }


def _detect_channels_for_tf(
    df: pd.DataFrame,
    windows: List[int] = None
) -> Dict[int, "Channel"]:
    """
    Detect channels at multiple windows for a given timeframe's data.

    Args:
        df: OHLCV DataFrame (already resampled to target TF)
        windows: List of window sizes to detect at

    Returns:
        Dict mapping window size to Channel object
    """
    if windows is None:
        windows = STANDARD_WINDOWS

    if not _channel_available:
        logger.warning("Channel detection not available")
        return {}

    try:
        channels = detect_channels_multi_window(df, windows=windows)
        return channels
    except Exception as e:
        logger.warning(f"Channel detection failed: {e}")
        return {}


def _sanitize_features(features: Dict[str, float]) -> Dict[str, float]:
    """
    Ensure all feature values are valid floats.

    Args:
        features: Dictionary of features

    Returns:
        Dictionary with all valid float values (no NaN, no inf)
    """
    sanitized = {}
    for name, value in features.items():
        try:
            float_val = float(value)
            if np.isfinite(float_val):
                sanitized[name] = float_val
            else:
                sanitized[name] = 0.0
        except (TypeError, ValueError):
            sanitized[name] = 0.0
    return sanitized


# =============================================================================
# Per-TF Feature Extraction Functions
# =============================================================================

def _extract_window_independent_features_for_tf(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    tf: str
) -> Dict[str, float]:
    """
    Extract all window-independent features for a single TF.

    Features extracted:
    - TSLA price features (60)
    - Technical indicators (77)
    - SPY features (80)
    - VIX features (25)
    - Cross-asset features (40)

    Total: 282 features per TF

    Args:
        tsla_df: TSLA OHLCV DataFrame (already resampled to TF)
        spy_df: SPY OHLCV DataFrame (already resampled to TF)
        vix_df: VIX OHLCV DataFrame (already resampled to TF)
        tf: Timeframe name

    Returns:
        Dict with TF-prefixed features (e.g., 'daily_rsi_14')
    """
    features: Dict[str, float] = {}

    # 1. TSLA Price Features (60)
    if _tsla_price_available:
        try:
            price_feats = extract_tsla_price_features(tsla_df)
            features.update(_prefix_tf(price_feats, tf))
        except Exception as e:
            logger.debug(f"Failed to extract TSLA price features for {tf}: {e}")

    # 2. Technical Features (77)
    if _technical_available:
        try:
            tech_feats = extract_technical_features(tsla_df)
            features.update(_prefix_tf(tech_feats, tf))
        except Exception as e:
            logger.debug(f"Failed to extract technical features for {tf}: {e}")

    # 3. SPY Features (80)
    if _spy_available:
        try:
            spy_feats = extract_spy_features(spy_df)
            # SPY features are already prefixed with 'spy_', so we just add TF prefix
            features.update(_prefix_tf(spy_feats, tf))
        except Exception as e:
            logger.debug(f"Failed to extract SPY features for {tf}: {e}")

    # 4. VIX Features (25)
    if _vix_available:
        try:
            vix_feats = extract_vix_features(vix_df)
            # VIX features are already prefixed with 'vix_', so we just add TF prefix
            features.update(_prefix_tf(vix_feats, tf))
        except Exception as e:
            logger.debug(f"Failed to extract VIX features for {tf}: {e}")

    # 5. Cross-Asset Features (40)
    if _cross_asset_available:
        try:
            cross_feats = extract_cross_asset_features(tsla_df, spy_df, vix_df)
            features.update(_prefix_tf(cross_feats, tf))
        except Exception as e:
            logger.debug(f"Failed to extract cross-asset features for {tf}: {e}")

    return features


def _extract_channel_features_for_tf(
    channels_by_window: Dict[int, "Channel"],
    tf: str
) -> Dict[str, float]:
    """
    Extract TSLA channel features for all windows at a single TF.

    Features extracted:
    - 58 channel features per valid window * 8 windows = 464 max

    Args:
        channels_by_window: Dict mapping window size to Channel
        tf: Timeframe name

    Returns:
        Dict with TF+window prefixed features (e.g., 'daily_w50_channel_slope')
    """
    features: Dict[str, float] = {}

    if not _tsla_channel_available:
        return features

    for window in STANDARD_WINDOWS:
        channel = channels_by_window.get(window)

        try:
            channel_feats = extract_tsla_channel_features(channel)
            features.update(_prefix_tf_window(channel_feats, tf, window))
        except Exception as e:
            logger.debug(f"Failed to extract channel features for {tf} w{window}: {e}")

    return features


def _extract_spy_channel_features_for_tf(
    spy_df: pd.DataFrame,
    spy_channels_by_window: Dict[int, "Channel"],
    tf: str
) -> Dict[str, float]:
    """
    Extract SPY channel features for all windows at a single TF.

    Features extracted:
    - 58 SPY channel features per valid window * 8 windows = 464 max

    Args:
        spy_df: SPY OHLCV DataFrame (already resampled to TF)
        spy_channels_by_window: Dict mapping window size to SPY Channel
        tf: Timeframe name

    Returns:
        Dict with TF+window prefixed features (e.g., 'daily_w50_spy_channel_slope')
    """
    features: Dict[str, float] = {}

    if not _spy_channel_available:
        return features

    for window in STANDARD_WINDOWS:
        channel = spy_channels_by_window.get(window)

        try:
            # extract_spy_channel_features returns features with 'spy_' prefix
            spy_feats = extract_spy_channel_features(spy_df, channel, window, tf)
            # Add TF+window prefix: 'spy_position_in_channel' -> 'daily_w50_spy_position_in_channel'
            features.update(_prefix_tf_window(spy_feats, tf, window))
        except Exception as e:
            logger.debug(f"Failed to extract SPY channel features for {tf} w{window}: {e}")

    return features


def _extract_channel_correlation_for_tf(
    tsla_channels_by_window: Dict[int, "Channel"],
    spy_channels_by_window: Dict[int, "Channel"],
    tf: str
) -> Dict[str, float]:
    """
    Extract channel correlation features for a single TF.

    Compares TSLA and SPY channel features to create cross-correlation metrics.

    Features extracted:
    - 50 channel correlation features per TF

    Args:
        tsla_channels_by_window: Dict mapping window size to TSLA Channel
        spy_channels_by_window: Dict mapping window size to SPY Channel
        tf: Timeframe name

    Returns:
        Dict with TF-prefixed features (e.g., 'daily_position_in_channel_spread')
    """
    features: Dict[str, float] = {}

    if not _channel_correlation_available:
        return features

    # Use a representative window for correlation (best window or default to 50)
    # We'll use window 50 as the default comparison window
    default_window = 50

    tsla_channel = tsla_channels_by_window.get(default_window)
    spy_channel = spy_channels_by_window.get(default_window)

    # Extract base channel features for correlation comparison
    # We need the raw feature names (without prefixes) for the correlation extractor
    tsla_channel_feats = {}
    spy_channel_feats = {}

    if _tsla_channel_available and tsla_channel is not None:
        try:
            tsla_channel_feats = extract_tsla_channel_features(tsla_channel)
        except Exception as e:
            logger.debug(f"Failed to extract TSLA channel features for correlation {tf}: {e}")

    if _spy_channel_available and spy_channel is not None:
        try:
            # SPY channel features have 'spy_' prefix, need to remove for correlation
            raw_spy_feats = extract_spy_channel_features(None, spy_channel, default_window, tf)
            # Remove 'spy_' prefix to get base feature names for correlation
            spy_channel_feats = {k.replace('spy_', ''): v for k, v in raw_spy_feats.items()}
        except Exception as e:
            logger.debug(f"Failed to extract SPY channel features for correlation {tf}: {e}")

    try:
        # extract_channel_correlation_features adds the TF prefix itself
        correlation_feats = extract_channel_correlation_features(
            tsla_channel_feats, spy_channel_feats, tf
        )
        features.update(correlation_feats)
    except Exception as e:
        logger.debug(f"Failed to extract channel correlation features for {tf}: {e}")

    return features


def _extract_window_scores_for_tf(
    channels_by_window: Dict[int, "Channel"],
    best_window: int,
    tf: str
) -> Dict[str, float]:
    """
    Extract window score features for a single TF.

    Features extracted:
    - 50 window score features

    Args:
        channels_by_window: Dict mapping window size to Channel
        best_window: The best performing window for this TF
        tf: Timeframe name

    Returns:
        Dict with TF-prefixed features (e.g., 'daily_valid_window_count')
    """
    features: Dict[str, float] = {}

    if not _window_scores_available:
        return features

    try:
        window_feats = extract_window_score_features(channels_by_window, best_window)
        features.update(_prefix_tf(window_feats, tf))
    except Exception as e:
        logger.debug(f"Failed to extract window score features for {tf}: {e}")

    return features


def _extract_channel_history_for_tf(
    tsla_channel_history: Optional[List[Dict]],
    spy_channel_history: Optional[List[Dict]],
    tf: str
) -> Dict[str, float]:
    """
    Extract channel history features for a single TF.

    Features extracted:
    - 50 channel history features

    Args:
        tsla_channel_history: List of last 5 TSLA channel dicts for this TF
        spy_channel_history: List of last 5 SPY channel dicts for this TF
        tf: Timeframe name

    Returns:
        Dict with TF-prefixed features (e.g., 'daily_tsla_last5_avg_duration')
    """
    features: Dict[str, float] = {}

    if not _channel_history_available:
        return features

    try:
        history_feats = extract_channel_history_features(
            tsla_channel_history or [],
            spy_channel_history or []
        )
        features.update(_prefix_tf(history_feats, tf))
    except Exception as e:
        logger.debug(f"Failed to extract channel history features for {tf}: {e}")

    return features


# =============================================================================
# Main Extraction Function
# =============================================================================

def _extract_bar_metadata_features(
    metadata_by_tf: Dict[str, Dict[str, Any]]
) -> Dict[str, float]:
    """
    Extract bar metadata features for all timeframes.

    For each TF, extracts:
    - {tf}_bar_completion_pct: 0.0-1.0, how complete the current TF bar is
    - {tf}_bars_in_partial: Number of 5min bars in the partial bar
    - {tf}_complete_bars: Total number of complete bars

    Total: 3 features * 10 TFs = 30 features

    Args:
        metadata_by_tf: Dict mapping TF -> resampling metadata

    Returns:
        Dict of bar metadata features
    """
    features: Dict[str, float] = {}

    for tf in TIMEFRAMES:
        meta = metadata_by_tf.get(tf, {})

        # Bar completion percentage (0.0 to 1.0)
        completion = meta.get('bar_completion_pct', 1.0)
        features[f"{tf}_bar_completion_pct"] = float(completion)

        # Number of 5min bars in the partial bar
        bars_in_partial = meta.get('bars_in_partial', 0)
        features[f"{tf}_bars_in_partial"] = float(bars_in_partial)

        # Number of complete bars (total - 1 if partial, else total)
        total_bars = meta.get('total_bars', 0)
        is_partial = meta.get('is_partial', False)
        complete_bars = max(0, total_bars - 1) if is_partial else total_bars
        features[f"{tf}_complete_bars"] = float(complete_bars)

    return features


def extract_all_tf_features(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    timestamp: pd.Timestamp,
    channel_history_by_tf: Optional[Dict[str, Dict]] = None,
    source_bar_count: Optional[int] = None,
    include_bar_metadata: bool = True
) -> Dict[str, float]:
    """
    Extract ALL features for ALL timeframes with partial bar support.

    This is the main entry point for TF-aware feature extraction. It:
    1. Resamples data to each of 10 timeframes (keeping partial bars)
    2. Detects channels at all 8 windows on each TF for both TSLA and SPY
    3. Extracts all features with explicit TF prefixes
    4. Includes bar metadata features showing completion percentages

    CRITICAL: This supports partial bars to match live inference behavior.
    During training, bar_completion_pct will vary based on position within
    the TF bar, just like in live trading.

    Args:
        tsla_df: Base 5-min TSLA OHLCV DataFrame
        spy_df: Base 5-min SPY OHLCV DataFrame
        vix_df: Base 5-min VIX OHLCV DataFrame
        timestamp: Current timestamp for event features
        channel_history_by_tf: Optional dict mapping TF -> {'tsla': [...], 'spy': [...]}
            Each list contains the last 5 channel dicts for that TF
        source_bar_count: Number of 5min bars from start of data to sample point.
                         Used to calculate bar_completion_pct based on position
                         within the TF bar. If None, uses len(tsla_df).
        include_bar_metadata: If True, include 30 bar metadata features (default True)

    Returns:
        Dict[str, float] with ~13,660 features:
        - 282 window-independent features * 10 TFs = 2,820
        - 58 TSLA channel features * 8 windows * 10 TFs = 4,640
        - 58 SPY channel features * 8 windows * 10 TFs = 4,640
        - 50 channel correlation features * 10 TFs = 500
        - 50 window score features * 10 TFs = 500
        - 50 channel history features * 10 TFs = 500
        - 30 event features (TF-independent) = 30
        - 30 bar metadata features (if include_bar_metadata=True)

        All features are explicitly named with TF prefix.
        All values are guaranteed to be valid floats (no NaN, no inf).

    Example feature names:
        - 'daily_rsi_14'
        - '1h_macd_signal'
        - 'weekly_spy_momentum_5'
        - 'daily_w50_channel_slope'
        - 'daily_w50_spy_channel_slope'
        - 'daily_position_in_channel_spread'
        - '1h_w20_position_in_channel'
        - 'daily_valid_window_count'
        - 'daily_bar_completion_pct'  # New: varies 0.0-1.0 during training
        - 'is_monday' (TF-independent)
    """
    all_features: Dict[str, float] = {}
    metadata_by_tf: Dict[str, Dict[str, Any]] = {}

    # Initialize channel history if not provided
    if channel_history_by_tf is None:
        channel_history_by_tf = {}

    # Use provided source_bar_count or default to len(tsla_df)
    if source_bar_count is None:
        source_bar_count = len(tsla_df)

    # Track extraction statistics
    extraction_stats = {
        'timeframes_processed': 0,
        'features_extracted': 0,
        'errors': [],
    }

    # -------------------------------------------------------------------------
    # Process each timeframe
    # -------------------------------------------------------------------------
    for tf in TIMEFRAMES:
        try:
            # 1. Resample data to this TF with partial bar support
            # Pass source_bar_count to calculate correct bar_completion_pct
            tsla_tf, tsla_meta = _resample_to_tf(tsla_df, tf, source_bar_count)
            spy_tf, _ = _resample_to_tf(spy_df, tf, source_bar_count)
            vix_tf, _ = _resample_to_tf(vix_df, tf, source_bar_count)

            # Store metadata for bar metadata features
            metadata_by_tf[tf] = tsla_meta

            # Check if we have enough data
            if len(tsla_tf) < 10:
                logger.debug(f"Not enough data for {tf} (have {len(tsla_tf)} bars)")
                continue

            # 2. Detect channels at all windows for this TF (TSLA)
            tsla_channels_by_window = _detect_channels_for_tf(tsla_tf)

            # 3. Detect channels at all windows for SPY
            spy_channels_by_window = _detect_channels_for_tf(spy_tf)

            # Select best TSLA channel
            if _channel_available and tsla_channels_by_window:
                best_channel, best_window = select_best_channel(tsla_channels_by_window)
                if best_window is None:
                    best_window = 50
            else:
                best_window = 50

            # 4. Extract window-independent features (282 per TF)
            wi_features = _extract_window_independent_features_for_tf(
                tsla_tf, spy_tf, vix_tf, tf
            )
            all_features.update(wi_features)

            # 5. Extract per-window TSLA channel features (58 per window * 8 windows = 464 per TF)
            channel_features = _extract_channel_features_for_tf(tsla_channels_by_window, tf)
            all_features.update(channel_features)

            # 6. Extract per-window SPY channel features (58 per window * 8 windows = 464 per TF)
            spy_channel_features = _extract_spy_channel_features_for_tf(
                spy_tf, spy_channels_by_window, tf
            )
            all_features.update(spy_channel_features)

            # 7. Extract channel correlation features (50 per TF)
            correlation_features = _extract_channel_correlation_for_tf(
                tsla_channels_by_window, spy_channels_by_window, tf
            )
            all_features.update(correlation_features)

            # 8. Extract window score features (50 per TF)
            window_score_features = _extract_window_scores_for_tf(
                tsla_channels_by_window, best_window, tf
            )
            all_features.update(window_score_features)

            # 9. Extract channel history features (50 per TF)
            tf_history = channel_history_by_tf.get(tf, {})
            tsla_history = tf_history.get('tsla', [])
            spy_history = tf_history.get('spy', [])

            history_features = _extract_channel_history_for_tf(
                tsla_history, spy_history, tf
            )
            all_features.update(history_features)

            extraction_stats['timeframes_processed'] += 1

            # Explicit memory cleanup for this TF iteration
            # These large objects are no longer needed after feature extraction
            del tsla_tf, spy_tf, vix_tf
            del tsla_channels_by_window, spy_channels_by_window
            del wi_features, channel_features, spy_channel_features
            del correlation_features, window_score_features, history_features

        except Exception as e:
            logger.error(f"Error processing timeframe {tf}: {e}")
            extraction_stats['errors'].append(f"{tf}: {e}")

    # -------------------------------------------------------------------------
    # Extract event features (TF-independent, no prefix)
    # -------------------------------------------------------------------------
    if _events_available:
        try:
            # Event features need timestamp and df for some calculations
            event_feats = extract_event_features(timestamp, tsla_df)
            # Event features are NOT prefixed with TF - they're global
            all_features.update(event_feats)
        except Exception as e:
            logger.error(f"Error extracting event features: {e}")
            extraction_stats['errors'].append(f"events: {e}")

    # Trigger garbage collection after TF loop to clean up intermediate objects
    gc.collect()

    # -------------------------------------------------------------------------
    # Extract bar metadata features (30 features: 3 per TF * 10 TFs)
    # -------------------------------------------------------------------------
    if include_bar_metadata:
        bar_meta_features = _extract_bar_metadata_features(metadata_by_tf)
        all_features.update(bar_meta_features)

    # -------------------------------------------------------------------------
    # Sanitize all features
    # -------------------------------------------------------------------------
    all_features = _sanitize_features(all_features)

    extraction_stats['features_extracted'] = len(all_features)

    # Log summary
    logger.debug(
        f"TF feature extraction complete: {extraction_stats['features_extracted']} features, "
        f"{extraction_stats['timeframes_processed']}/{len(TIMEFRAMES)} TFs processed"
    )

    if extraction_stats['errors']:
        logger.warning(f"Extraction errors: {extraction_stats['errors']}")

    return all_features


# =============================================================================
# Feature Metadata Functions
# =============================================================================

def get_tf_feature_count() -> int:
    """
    Return total expected feature count.

    Returns:
        Expected total features (~13,660)
    """
    return TOTAL_FEATURES


def get_tf_feature_names() -> List[str]:
    """
    Return all feature names in consistent order.

    The order is:
    1. For each TF (in TIMEFRAMES order):
       a. Window-independent features (price, technical, spy, vix, cross_asset)
       b. Per-window TSLA channel features (for each window in STANDARD_WINDOWS)
       c. Per-window SPY channel features (for each window in STANDARD_WINDOWS)
       d. Channel correlation features
       e. Window score features
       f. Channel history features
    2. Event features (TF-independent)

    Returns:
        List of all feature names in consistent order
    """
    all_names = []

    # Get base feature names from each extractor
    try:
        from .tsla_price import get_tsla_price_feature_names
        price_names = get_tsla_price_feature_names()
    except ImportError:
        price_names = [f"price_feature_{i}" for i in range(60)]

    try:
        from .technical import get_technical_feature_names
        tech_names = get_technical_feature_names()
    except ImportError:
        tech_names = [f"tech_feature_{i}" for i in range(77)]

    try:
        from .spy import get_spy_feature_names
        spy_names = get_spy_feature_names()
    except ImportError:
        spy_names = [f"spy_feature_{i}" for i in range(80)]

    try:
        from .vix import get_vix_feature_names
        vix_names = get_vix_feature_names()
    except ImportError:
        vix_names = [f"vix_feature_{i}" for i in range(25)]

    try:
        from .cross_asset import get_cross_asset_feature_names
        cross_names = get_cross_asset_feature_names()
    except ImportError:
        cross_names = [f"cross_feature_{i}" for i in range(40)]

    try:
        from .tsla_channel import get_tsla_channel_feature_names
        channel_names = get_tsla_channel_feature_names()
    except ImportError:
        channel_names = [f"channel_feature_{i}" for i in range(58)]

    try:
        from .spy_channel import get_spy_channel_feature_names
        spy_channel_names = get_spy_channel_feature_names()
    except ImportError:
        spy_channel_names = [f"spy_channel_feature_{i}" for i in range(58)]

    try:
        from .channel_correlation import get_channel_correlation_feature_names
        correlation_names = get_channel_correlation_feature_names()
    except ImportError:
        correlation_names = [f"correlation_feature_{i}" for i in range(50)]

    try:
        from .window_scores import get_window_score_feature_names
        window_score_names = get_window_score_feature_names()
    except ImportError:
        window_score_names = [f"window_score_{i}" for i in range(50)]

    try:
        from .channel_history import get_channel_history_feature_names
        history_names = get_channel_history_feature_names()
    except ImportError:
        history_names = [f"history_feature_{i}" for i in range(50)]

    try:
        from .events import get_event_feature_names
        event_names = get_event_feature_names()
    except ImportError:
        event_names = [f"event_feature_{i}" for i in range(30)]

    # Build feature names for each TF
    for tf in TIMEFRAMES:
        tf_prefix = f"{tf}_"

        # Window-independent features
        for name in price_names:
            all_names.append(f"{tf_prefix}{name}")

        for name in tech_names:
            all_names.append(f"{tf_prefix}{name}")

        for name in spy_names:
            all_names.append(f"{tf_prefix}{name}")

        for name in vix_names:
            all_names.append(f"{tf_prefix}{name}")

        for name in cross_names:
            all_names.append(f"{tf_prefix}{name}")

        # Per-window TSLA channel features
        for window in STANDARD_WINDOWS:
            window_prefix = f"{tf}_w{window}_"
            for name in channel_names:
                all_names.append(f"{window_prefix}{name}")

        # Per-window SPY channel features
        for window in STANDARD_WINDOWS:
            window_prefix = f"{tf}_w{window}_"
            for name in spy_channel_names:
                all_names.append(f"{window_prefix}{name}")

        # Channel correlation features (TF-prefixed but not window-prefixed)
        for name in correlation_names:
            all_names.append(f"{tf_prefix}{name}")

        # Window score features
        for name in window_score_names:
            all_names.append(f"{tf_prefix}{name}")

        # Channel history features
        for name in history_names:
            all_names.append(f"{tf_prefix}{name}")

    # Event features (no TF prefix)
    all_names.extend(event_names)

    return all_names


def get_tf_feature_breakdown() -> Dict[str, int]:
    """
    Return feature count by category.

    Returns:
        Dictionary with category names and counts
    """
    return {
        'total_features': TOTAL_FEATURES,
        'num_timeframes': len(TIMEFRAMES),
        'num_windows': len(STANDARD_WINDOWS),
        'per_tf_window_independent': TOTAL_PER_TF_WINDOW_INDEPENDENT,
        'per_tf_window_dependent': TOTAL_PER_TF_WINDOW_DEPENDENT,
        'tsla_channel_per_tf': TOTAL_TSLA_CHANNEL_PER_TF,
        'spy_channel_per_tf': TOTAL_SPY_CHANNEL_PER_TF,
        'per_tf_total': TOTAL_PER_TF,
        'price_per_tf': TF_FEATURE_COUNTS['price_per_tf'],
        'technical_per_tf': TF_FEATURE_COUNTS['technical_per_tf'],
        'spy_per_tf': TF_FEATURE_COUNTS['spy_per_tf'],
        'vix_per_tf': TF_FEATURE_COUNTS['vix_per_tf'],
        'cross_asset_per_tf': TF_FEATURE_COUNTS['cross_asset_per_tf'],
        'tsla_channel_per_window': TF_FEATURE_COUNTS['channel_per_window'],
        'spy_channel_per_window': TF_FEATURE_COUNTS['spy_channel_per_window'],
        'channel_correlation_per_tf': TF_FEATURE_COUNTS['channel_correlation_per_tf'],
        'window_scores_per_tf': TF_FEATURE_COUNTS['window_scores_per_tf'],
        'channel_history_per_tf': TF_FEATURE_COUNTS['channel_history_per_tf'],
        'events_total': TF_FEATURE_COUNTS['events'],
    }


def get_features_for_tf(tf: str) -> List[str]:
    """
    Get all feature names for a specific timeframe.

    Args:
        tf: Timeframe name (e.g., 'daily', '1h')

    Returns:
        List of feature names for that TF
    """
    all_names = get_tf_feature_names()
    tf_prefix = f"{tf}_"

    # Get features that start with this TF's prefix
    return [name for name in all_names if name.startswith(tf_prefix)]


def get_features_for_window(tf: str, window: int) -> List[str]:
    """
    Get channel feature names for a specific TF and window.

    Args:
        tf: Timeframe name
        window: Window size

    Returns:
        List of channel feature names for that TF+window combination
    """
    window_prefix = f"{tf}_w{window}_"
    all_names = get_tf_feature_names()

    return [name for name in all_names if name.startswith(window_prefix)]


# =============================================================================
# Validation Functions
# =============================================================================

def validate_tf_features(features: Dict[str, float]) -> Dict[str, List[str]]:
    """
    Validate a feature dictionary and report issues.

    Args:
        features: Dictionary of features to validate

    Returns:
        Dictionary with 'missing', 'extra', 'invalid' lists
    """
    expected_names = set(get_tf_feature_names())
    actual_names = set(features.keys())

    issues = {
        'missing': sorted(expected_names - actual_names),
        'extra': sorted(actual_names - expected_names),
        'invalid': [],
    }

    # Check for invalid values
    for name, value in features.items():
        try:
            float_val = float(value)
            if not np.isfinite(float_val):
                issues['invalid'].append(name)
        except (TypeError, ValueError):
            issues['invalid'].append(name)

    return issues


def get_extraction_status() -> Dict[str, bool]:
    """
    Get availability status of all feature extraction modules.

    Returns:
        Dictionary mapping module name to availability status
    """
    return {
        'timeframe': _timeframe_available,
        'channel': _channel_available,
        'tsla_price': _tsla_price_available,
        'technical': _technical_available,
        'spy': _spy_available,
        'vix': _vix_available,
        'cross_asset': _cross_asset_available,
        'tsla_channel': _tsla_channel_available,
        'spy_channel': _spy_channel_available,
        'channel_correlation': _channel_correlation_available,
        'window_scores': _window_scores_available,
        'channel_history': _channel_history_available,
        'events': _events_available,
    }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_feature_vector(
    features: Dict[str, float],
    names: Optional[List[str]] = None
) -> np.ndarray:
    """
    Create a numpy array from feature dictionary in consistent order.

    Args:
        features: Dictionary of features
        names: Optional list of names to use (defaults to get_tf_feature_names())

    Returns:
        Numpy array of feature values in consistent order
    """
    if names is None:
        names = get_tf_feature_names()

    vector = np.zeros(len(names), dtype=np.float64)

    for i, name in enumerate(names):
        vector[i] = features.get(name, 0.0)

    return vector


def features_to_dataframe(
    features: Dict[str, float],
    timestamp: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Convert feature dictionary to a single-row DataFrame.

    Args:
        features: Dictionary of features
        timestamp: Optional timestamp for index

    Returns:
        DataFrame with features as columns
    """
    df = pd.DataFrame([features])

    if timestamp is not None:
        df.index = pd.DatetimeIndex([timestamp])

    return df
