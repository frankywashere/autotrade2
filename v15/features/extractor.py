"""
Feature Extractor - Main Orchestrator for x14 Feature System v15

This module is the single entry point for ALL feature extraction.
It orchestrates:
1. Partial bar resampling across all 10 timeframes (keeping incomplete bars)
2. TSLA channel detection at all 8 windows per timeframe
3. SPY channel detection at all 8 windows per timeframe (using same detect_channel)
4. Feature extraction from all modules including:
   - Window-independent features (TSLA price, technical, SPY, VIX, cross-asset)
   - TSLA channel features
   - SPY channel features
   - Channel correlation features (TSLA vs SPY)
   - Aggregated features (window scores, channel history)
5. Bar metadata features (completion_pct, bars_in_partial, complete_bars)
6. Strict validation - NO silent NaN allowed
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, TYPE_CHECKING, Any

import numpy as np
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)

# =============================================================================
# Type checking imports
# =============================================================================
if TYPE_CHECKING:
    from ..channel import Channel

# =============================================================================
# Configuration imports - Single source of truth
# =============================================================================
from ..config import (
    TIMEFRAMES,
    N_TIMEFRAMES,
    STANDARD_WINDOWS,
    N_WINDOWS,
    BARS_PER_TF,
    FEATURE_COUNTS,
    TOTAL_FEATURES,
    TF_LOOKBACK_5MIN,
    TF_FORWARD_5MIN,
)

from ..exceptions import (
    FeatureExtractionError,
    InvalidFeatureError,
    ResamplingError,
)

# =============================================================================
# Data resampling with partial bar support
# =============================================================================
# Use canonical resample module for OHLC resampling
from ..core.resample import resample_ohlc, BarMetadata, ResamplingError as CoreResamplingError
# Also import partial bar support from data.resampler (uses canonical under the hood)
from ..data.resampler import resample_with_partial

# =============================================================================
# Feature module imports - LOUD failures, no graceful degradation
# =============================================================================

from .utils import safe_float, ensure_finite
from .validation import validate_features as _validate_feature_dict

# Import all feature extractors - failure means we can't function
try:
    from .tsla_price import extract_tsla_price_features
except ImportError as e:
    raise FeatureExtractionError(f"Critical module tsla_price not available: {e}")

try:
    from .technical import extract_technical_features
except ImportError as e:
    raise FeatureExtractionError(f"Critical module technical not available: {e}")

try:
    from .spy import extract_spy_features
except ImportError as e:
    raise FeatureExtractionError(f"Critical module spy not available: {e}")

try:
    from .vix import extract_vix_features
except ImportError as e:
    raise FeatureExtractionError(f"Critical module vix not available: {e}")

try:
    from .cross_asset import extract_cross_asset_features
except ImportError as e:
    raise FeatureExtractionError(f"Critical module cross_asset not available: {e}")

try:
    from .tsla_channel import extract_tsla_channel_features
except ImportError as e:
    raise FeatureExtractionError(f"Critical module tsla_channel not available: {e}")

try:
    from .window_scores import extract_window_score_features
except ImportError as e:
    raise FeatureExtractionError(f"Critical module window_scores not available: {e}")

try:
    from .channel_history import extract_channel_history_features
except ImportError as e:
    raise FeatureExtractionError(f"Critical module channel_history not available: {e}")

try:
    from .events import extract_event_features
except ImportError as e:
    raise FeatureExtractionError(f"Critical module events not available: {e}")

try:
    from .spy_channel import extract_spy_channel_features
except ImportError as e:
    raise FeatureExtractionError(f"Critical module spy_channel not available: {e}")

try:
    from .channel_correlation import extract_channel_correlation_features
except ImportError as e:
    raise FeatureExtractionError(f"Critical module channel_correlation not available: {e}")


# =============================================================================
# Channel detection import
# =============================================================================
try:
    from v7.core.channel import detect_channels_multi_window, select_best_channel
    _channel_detection_available = True
except ImportError:
    logger.warning("v7.core.channel not available - using placeholder channel detection")
    _channel_detection_available = False


# =============================================================================
# Timeframe mapping for resampler
# =============================================================================
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
    'monthly': '1MS',   # Month Start (pandas format)
}


# =============================================================================
# Helper Functions
# =============================================================================

def _prefix_features(features: Dict[str, float], prefix: str) -> Dict[str, float]:
    """Add prefix to all feature names."""
    return {f"{prefix}{k}": v for k, v in features.items()}


def _prefix_tf(features: Dict[str, float], tf: str) -> Dict[str, float]:
    """Add TF prefix to features (e.g., 'daily_rsi_14')."""
    return _prefix_features(features, f"{tf}_")


def _prefix_tf_window(features: Dict[str, float], tf: str, window: int) -> Dict[str, float]:
    """Add TF and window prefix (e.g., 'daily_w50_channel_slope')."""
    return _prefix_features(features, f"{tf}_w{window}_")


def _resample_to_tf(
    df: pd.DataFrame,
    tf: str,
    source_tf: str = '5min',
    max_source_bars: Optional[int] = None,
    source_bar_count: Optional[int] = None
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Resample DataFrame to target timeframe using partial bar support.

    CRITICAL: This keeps partial (incomplete) bars instead of dropping them.
    In live trading, the current bar is always incomplete.

    Args:
        df: Source OHLCV DataFrame (typically 5-min data)
        tf: Target timeframe
        source_tf: Source timeframe (for metadata calculation)
        max_source_bars: If provided, only use last N source bars before resampling.
                        This improves efficiency by avoiding resampling unused data.
        source_bar_count: Number of 5min bars from start of data to sample point.
                         Used to calculate bar_completion_pct based on position
                         within the TF bar. If None, uses len(df).

    Returns:
        Tuple of (resampled_df, metadata_dict)

    Raises:
        ResamplingError: If resampling fails
    """
    # Apply efficient slicing BEFORE any processing
    if max_source_bars is not None and len(df) > max_source_bars:
        df = df.iloc[-max_source_bars:]

    # Use provided source_bar_count or default to len(df)
    if source_bar_count is None:
        source_bar_count = len(df)

    bars_per_tf_bar = BARS_PER_TF.get(tf, 1)

    if tf == '5min':
        # No resampling needed - return as-is with full completion
        return df, {
            'bar_completion_pct': 1.0,
            'bars_in_partial': BARS_PER_TF['5min'],
            'expected_bars': BARS_PER_TF['5min'],
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

    resample_rule = TF_TO_RESAMPLE_RULE.get(tf)
    if resample_rule is None:
        raise ResamplingError(f"Unknown timeframe: {tf}")

    try:
        resampled, raw_metadata = resample_with_partial(df, resample_rule, source_tf)

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

        return resampled, metadata
    except Exception as e:
        raise ResamplingError(f"Failed to resample to {tf}: {e}")


def _detect_channels(df: pd.DataFrame, windows: List[int] = None) -> Dict[int, Any]:
    """
    Detect channels at multiple windows.

    Args:
        df: OHLCV DataFrame (already resampled)
        windows: List of window sizes

    Returns:
        Dict mapping window -> Channel object
    """
    if windows is None:
        windows = STANDARD_WINDOWS

    if not _channel_detection_available:
        # Return empty dict - features will use defaults
        return {}

    try:
        channels = detect_channels_multi_window(df, windows=windows)
        return channels
    except Exception as e:
        logger.warning(f"Channel detection failed: {e}")
        return {}


def _detect_spy_channels(spy_df: pd.DataFrame, windows: List[int] = None) -> Dict[int, Any]:
    """
    Detect channels on SPY data at multiple windows.

    Uses the same detect_channels_multi_window function as TSLA.

    Args:
        spy_df: SPY OHLCV DataFrame (already resampled)
        windows: List of window sizes

    Returns:
        Dict mapping window -> Channel object for SPY
    """
    if windows is None:
        windows = STANDARD_WINDOWS

    if not _channel_detection_available:
        # Return empty dict - features will use defaults
        return {}

    try:
        channels = detect_channels_multi_window(spy_df, windows=windows)
        return channels
    except Exception as e:
        logger.warning(f"SPY channel detection failed: {e}")
        return {}


def _extract_bar_metadata_features(
    metadata_by_tf: Dict[str, Dict[str, Any]]
) -> Dict[str, float]:
    """
    Extract bar metadata features for all timeframes.

    For each TF, extracts:
    - {tf}_bar_completion_pct: 0.0-1.0, how complete the last bar is
    - {tf}_bars_in_partial: Number of source bars in the partial bar
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
        features[f"{tf}_bar_completion_pct"] = safe_float(completion, default=1.0, min_val=0.0, max_val=1.0)

        # Number of source bars in the partial bar
        bars_in_partial = meta.get('bars_in_partial', 0)
        features[f"{tf}_bars_in_partial"] = safe_float(bars_in_partial, default=0.0, min_val=0.0)

        # Number of complete bars (total - 1 if partial, else total)
        total_bars = meta.get('total_bars', 0)
        is_partial = meta.get('is_partial', False)
        complete_bars = max(0, total_bars - 1) if is_partial else total_bars
        features[f"{tf}_complete_bars"] = safe_float(complete_bars, default=0.0, min_val=0.0)

    return features


# =============================================================================
# Per-TF Feature Extraction
# =============================================================================

def _extract_window_independent_features_for_tf(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    tf: str,
    tsla_channels: Optional[Dict[int, "Channel"]] = None,
    spy_channels: Optional[Dict[int, "Channel"]] = None
) -> Dict[str, float]:
    """
    Extract all window-independent features for a single TF.

    Features:
    - TSLA price features (60)
    - Technical indicators (77)
    - SPY features (80)
    - VIX features (25)
    - Cross-asset features (40)

    Total: 282 features per TF

    Args:
        tsla_df: TSLA OHLCV (already resampled to TF)
        spy_df: SPY OHLCV (already resampled to TF)
        vix_df: VIX OHLCV (already resampled to TF)
        tf: Timeframe name
        tsla_channels: Optional dict of TSLA channels by window (for position_in_channel)
        spy_channels: Optional dict of SPY channels by window (for spy_position_in_channel)

    Returns:
        Dict with TF-prefixed features

    Raises:
        FeatureExtractionError: If any extraction fails
    """
    features: Dict[str, float] = {}

    # 1. TSLA Price Features (60)
    try:
        price_feats = extract_tsla_price_features(tsla_df)
        features.update(_prefix_tf(price_feats, tf))
    except Exception as e:
        raise FeatureExtractionError(f"TSLA price extraction failed for {tf}: {e}")

    # 2. Technical Features (77)
    try:
        tech_feats = extract_technical_features(tsla_df)
        features.update(_prefix_tf(tech_feats, tf))
    except Exception as e:
        raise FeatureExtractionError(f"Technical extraction failed for {tf}: {e}")

    # 3. SPY Features (80)
    try:
        spy_feats = extract_spy_features(spy_df)
        features.update(_prefix_tf(spy_feats, tf))
    except Exception as e:
        raise FeatureExtractionError(f"SPY extraction failed for {tf}: {e}")

    # 4. VIX Features (25)
    try:
        vix_feats = extract_vix_features(vix_df)
        features.update(_prefix_tf(vix_feats, tf))
    except Exception as e:
        raise FeatureExtractionError(f"VIX extraction failed for {tf}: {e}")

    # 5. Cross-Asset Features (59)
    # Pass in the already-extracted feature values for RSI/VIX correlation features
    try:
        # Get RSI values from price features
        tsla_rsi_14 = price_feats.get('rsi_14')
        spy_rsi_14 = spy_feats.get('spy_rsi_14')
        vix_level = vix_feats.get('vix_level')

        # Get position_in_channel from detected channels (use window 50 as default)
        position_in_channel = None
        spy_position_in_channel = None

        if tsla_channels:
            # Try window 50 first, then fall back to any available window
            tsla_channel = tsla_channels.get(50) or next(iter(tsla_channels.values()), None)
            if tsla_channel is not None:
                # Get position_in_channel from channel
                # CRITICAL: Use [-2] (previous bar) to avoid data leakage
                # Using [-1] would use current bar info which isn't available at prediction time
                upper_line = getattr(tsla_channel, 'upper_line', None)
                lower_line = getattr(tsla_channel, 'lower_line', None)
                if upper_line is not None and lower_line is not None and len(upper_line) > 1:
                    upper = upper_line[-2]  # Previous bar to avoid leakage
                    lower = lower_line[-2]  # Previous bar to avoid leakage
                    if upper != lower and len(tsla_df) > 1:
                        close = float(tsla_df['close'].iloc[-2])  # Previous bar close
                        position_in_channel = (close - lower) / (upper - lower)
                        position_in_channel = max(0.0, min(1.0, position_in_channel))

        if spy_channels:
            # Try window 50 first, then fall back to any available window
            spy_channel = spy_channels.get(50) or next(iter(spy_channels.values()), None)
            if spy_channel is not None:
                # Get position_in_channel from channel
                # CRITICAL: Use [-2] (previous bar) to avoid data leakage
                # Using [-1] would use current bar info which isn't available at prediction time
                upper_line = getattr(spy_channel, 'upper_line', None)
                lower_line = getattr(spy_channel, 'lower_line', None)
                if upper_line is not None and lower_line is not None and len(upper_line) > 1:
                    upper = upper_line[-2]  # Previous bar to avoid leakage
                    lower = lower_line[-2]  # Previous bar to avoid leakage
                    if upper != lower and len(spy_df) > 1:
                        close = float(spy_df['close'].iloc[-2])  # Previous bar close
                        spy_position_in_channel = (close - lower) / (upper - lower)
                        spy_position_in_channel = max(0.0, min(1.0, spy_position_in_channel))

        cross_feats = extract_cross_asset_features(
            tsla_df, spy_df, vix_df,
            tsla_rsi_14=tsla_rsi_14,
            spy_rsi_14=spy_rsi_14,
            position_in_channel=position_in_channel,
            spy_position_in_channel=spy_position_in_channel,
            vix_level=vix_level
        )
        features.update(_prefix_tf(cross_feats, tf))
    except Exception as e:
        raise FeatureExtractionError(f"Cross-asset extraction failed for {tf}: {e}")

    return features


def _extract_channel_features_for_tf(
    channels_by_window: Dict[int, "Channel"],
    tf: str
) -> Dict[str, float]:
    """
    Extract TSLA channel features for all windows at a single TF.

    Features: 50 per window * 8 windows = 400 per TF

    Args:
        channels_by_window: Dict mapping window -> Channel (TSLA)
        tf: Timeframe name

    Returns:
        Dict with TF+window prefixed features

    Raises:
        FeatureExtractionError: If extraction fails
    """
    features: Dict[str, float] = {}

    for window in STANDARD_WINDOWS:
        channel = channels_by_window.get(window)

        try:
            channel_feats = extract_tsla_channel_features(channel)
            features.update(_prefix_tf_window(channel_feats, tf, window))
        except Exception as e:
            raise FeatureExtractionError(
                f"Channel extraction failed for {tf} window {window}: {e}"
            )

    return features


def _extract_spy_channel_features_for_tf(
    spy_df: pd.DataFrame,
    spy_channels_by_window: Dict[int, "Channel"],
    tf: str
) -> Dict[str, float]:
    """
    Extract SPY channel features for all windows at a single TF.

    Features: 58 per window * 8 windows = 464 per TF

    Args:
        spy_df: SPY DataFrame for this TF
        spy_channels_by_window: Dict mapping window -> Channel (SPY)
        tf: Timeframe name

    Returns:
        Dict with TF+window prefixed features

    Raises:
        FeatureExtractionError: If extraction fails
    """
    features: Dict[str, float] = {}

    for window in STANDARD_WINDOWS:
        channel = spy_channels_by_window.get(window)

        try:
            # extract_spy_channel_features takes: spy_df, channel, window, tf
            spy_channel_feats = extract_spy_channel_features(spy_df, channel, window, tf)
            features.update(_prefix_tf_window(spy_channel_feats, tf, window))
        except Exception as e:
            raise FeatureExtractionError(
                f"SPY channel extraction failed for {tf} window {window}: {e}"
            )

    return features


def _extract_channel_correlation_features_for_tf(
    tsla_channels_by_window: Dict[int, "Channel"],
    spy_channels_by_window: Dict[int, "Channel"],
    tf: str
) -> Dict[str, float]:
    """
    Extract channel correlation features between TSLA and SPY channels for a TF.

    This computes cross-correlation metrics between TSLA and SPY channel features
    to capture market-relative behavior.

    Features: ~50 per TF

    Args:
        tsla_channels_by_window: Dict mapping window -> TSLA Channel
        spy_channels_by_window: Dict mapping window -> SPY Channel
        tf: Timeframe name

    Returns:
        Dict with TF-prefixed correlation features

    Raises:
        FeatureExtractionError: If extraction fails
    """
    # Get channel features for best window (or default window 50)
    best_window = 50

    # Try to get features from the best window
    tsla_channel = tsla_channels_by_window.get(best_window)
    spy_channel = spy_channels_by_window.get(best_window)

    # Extract base features for correlation (without prefixes)
    try:
        tsla_feats = extract_tsla_channel_features(tsla_channel) if tsla_channel else {}
        # For SPY, we need to get the base features without prefixes
        # The spy_channel features already have 'spy_' prefix
        spy_feats_raw = extract_spy_channel_features(
            pd.DataFrame(),  # We don't need spy_df for base features if channel is available
            spy_channel,
            best_window,
            tf
        ) if spy_channel else {}

        # Remove 'spy_' prefix from spy features for correlation computation
        spy_feats = {}
        for k, v in spy_feats_raw.items():
            # Remove 'spy_' prefix to align with tsla feature names
            new_key = k.replace('spy_', '') if k.startswith('spy_') else k
            spy_feats[new_key] = v

        # Extract correlation features
        corr_feats = extract_channel_correlation_features(tsla_feats, spy_feats, tf)
        return corr_feats

    except Exception as e:
        raise FeatureExtractionError(
            f"Channel correlation extraction failed for {tf}: {e}"
        )


def _extract_aggregated_features_for_tf(
    channels_by_window: Dict[int, "Channel"],
    tsla_channel_history: Optional[List[Dict]],
    spy_channel_history: Optional[List[Dict]],
    tf: str
) -> Dict[str, float]:
    """
    Extract aggregated features (window scores, channel history) for a TF.

    Features:
    - Window scores (50)
    - Channel history (50)

    Total: 100 per TF

    Args:
        channels_by_window: Dict mapping window -> Channel
        tsla_channel_history: Historical TSLA channel data
        spy_channel_history: Historical SPY channel data
        tf: Timeframe name

    Returns:
        Dict with TF-prefixed features

    Raises:
        FeatureExtractionError: If extraction fails
    """
    features: Dict[str, float] = {}

    # Determine best window
    best_window = 50  # Default
    if _channel_detection_available and channels_by_window:
        try:
            _, best_window = select_best_channel(channels_by_window)
            if best_window is None:
                best_window = 50
        except Exception:
            pass

    # 1. Window Score Features (50)
    try:
        window_feats = extract_window_score_features(channels_by_window, best_window)
        features.update(_prefix_tf(window_feats, tf))
    except Exception as e:
        raise FeatureExtractionError(f"Window score extraction failed for {tf}: {e}")

    # 2. Channel History Features (50)
    try:
        history_feats = extract_channel_history_features(
            tsla_channel_history or [],
            spy_channel_history or []
        )
        features.update(_prefix_tf(history_feats, tf))
    except Exception as e:
        raise FeatureExtractionError(f"Channel history extraction failed for {tf}: {e}")

    return features


# =============================================================================
# Main Extraction Function
# =============================================================================

def extract_all_features(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    timestamp: pd.Timestamp,
    channels_by_window: Dict[int, "Channel"],
    validate: bool = True,
    native_tf_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
    source_bar_count: Optional[int] = None
) -> Dict[str, float]:
    """
    Extract all features for a single sample with partial bar support.

    This is the MAIN ENTRY POINT for feature extraction. It:
    1. Resamples data to all 10 timeframes (keeping partial bars)
    2. Detects channels at all 8 windows per timeframe
    3. Extracts all features with explicit TF prefixes
    4. Adds bar metadata features with position-based completion percentages
    5. Validates all features (no silent NaN)

    CRITICAL: Supports partial bars to match live inference behavior.
    During training, bar_completion_pct will vary based on position within
    the TF bar, just like in live trading.

    Args:
        tsla_df: Base 5-min TSLA OHLCV DataFrame
        spy_df: Base 5-min SPY OHLCV DataFrame
        vix_df: Base 5-min VIX OHLCV DataFrame
        timestamp: Current timestamp for event features
        channels_by_window: Pre-computed channels (optional, will detect if empty)
        validate: If True, validate all features and raise on invalid
        native_tf_data: Optional dict of pre-loaded native TF data.
                       Format: {symbol: {tf: DataFrame}} where symbol is 'TSLA', 'SPY', or 'VIX'.
                       If provided for a TF, uses native data instead of resampling from 5-min.
                       This is more efficient and accurate for higher timeframes.
        source_bar_count: Number of 5min bars from start of data to sample point.
                         Used to calculate bar_completion_pct based on position
                         within the TF bar. If None, uses len(tsla_df).
                         Example for daily TF (78 bars per day):
                           At idx=100: bars_into_current = 100 % 78 = 22
                           completion_pct = 22 / 78 = 0.28 (28% complete)

    Returns:
        Dict with named features including:
        - Window-independent per TF: 282 * 10 = 2,820
        - TSLA channel per window per TF: 50 * 8 * 10 = 4,000
        - SPY channel per window per TF: 58 * 8 * 10 = 4,640
        - Channel correlation per TF: ~50 * 10 = 500
        - Aggregated per TF: 100 * 10 = 1,000
        - Events (global): 30
        - Bar metadata: 3 * 10 = 30

    Raises:
        FeatureExtractionError: If extraction fails
        InvalidFeatureError: If any feature is NaN/Inf (when validate=True)
    """
    # Validate inputs
    if tsla_df is None or len(tsla_df) == 0:
        raise FeatureExtractionError("TSLA DataFrame is empty or None")
    if spy_df is None or len(spy_df) == 0:
        raise FeatureExtractionError("SPY DataFrame is empty or None")
    if vix_df is None or len(vix_df) == 0:
        raise FeatureExtractionError("VIX DataFrame is empty or None")

    all_features: Dict[str, float] = {}
    metadata_by_tf: Dict[str, Dict[str, Any]] = {}

    # Use provided source_bar_count or default to len(tsla_df)
    if source_bar_count is None:
        source_bar_count = len(tsla_df)

    # =========================================================================
    # Process each timeframe
    # =========================================================================
    for tf in TIMEFRAMES:
        try:
            # Check if native TF data is available for this timeframe
            use_native = (
                native_tf_data is not None
                and native_tf_data.get('TSLA', {}).get(tf) is not None
                and native_tf_data.get('SPY', {}).get(tf) is not None
                and native_tf_data.get('VIX', {}).get(tf) is not None
            )

            if use_native:
                # Use pre-loaded native TF data (no resampling needed)
                tsla_tf = native_tf_data['TSLA'][tf]
                spy_tf = native_tf_data['SPY'][tf]
                vix_tf = native_tf_data['VIX'][tf]

                # Calculate completion based on position within TF bar
                # even for native data, to maintain consistency
                bars_per_tf_bar = BARS_PER_TF.get(tf, 1)
                bars_into_current = source_bar_count % bars_per_tf_bar
                if bars_into_current == 0:
                    completion_pct = 1.0
                    is_partial = False
                else:
                    completion_pct = bars_into_current / bars_per_tf_bar
                    is_partial = True

                tsla_meta = {
                    'bar_completion_pct': round(completion_pct, 4),
                    'bars_in_partial': bars_into_current if is_partial else bars_per_tf_bar,
                    'expected_bars': bars_per_tf_bar,
                    'is_partial': is_partial,
                    'total_bars': len(tsla_tf),
                    'source_bars': len(tsla_tf),
                }
            else:
                # Get TF-specific lookback requirement for efficient slicing
                lookback_bars = TF_LOOKBACK_5MIN.get(tf)  # None means use all data

                # 1. Resample data to this TF with efficient slicing (keeps partial bars)
                # Pass source_bar_count for position-based completion calculation
                tsla_tf, tsla_meta = _resample_to_tf(
                    tsla_df, tf, max_source_bars=lookback_bars,
                    source_bar_count=source_bar_count
                )
                spy_tf, _ = _resample_to_tf(
                    spy_df, tf, max_source_bars=lookback_bars,
                    source_bar_count=source_bar_count
                )
                vix_tf, _ = _resample_to_tf(
                    vix_df, tf, max_source_bars=lookback_bars,
                    source_bar_count=source_bar_count
                )

            # Store metadata for bar metadata features
            metadata_by_tf[tf] = tsla_meta

            # Check minimum data requirement
            if len(tsla_tf) < 10:
                raise FeatureExtractionError(
                    f"Insufficient data for {tf}: have {len(tsla_tf)} bars, need 10+"
                )

            # 2. Detect TSLA channels at all windows for this TF
            tf_channels = _detect_channels(tsla_tf)

            # 2b. Detect SPY channels at all windows for this TF
            # Uses the same detect_channel function as TSLA
            spy_tf_channels = _detect_spy_channels(spy_tf)

            # 3. Extract window-independent features (282 per TF)
            # Pass detected channels so cross-asset features can use position_in_channel
            wi_features = _extract_window_independent_features_for_tf(
                tsla_tf, spy_tf, vix_tf, tf,
                tsla_channels=tf_channels,
                spy_channels=spy_tf_channels
            )
            all_features.update(wi_features)

            # 4. Extract TSLA channel features (400 per TF)
            channel_features = _extract_channel_features_for_tf(tf_channels, tf)
            all_features.update(channel_features)

            # 4b. Extract SPY channel features (464 per TF = 58 features * 8 windows)
            spy_channel_features = _extract_spy_channel_features_for_tf(
                spy_tf, spy_tf_channels, tf
            )
            all_features.update(spy_channel_features)

            # 4c. Extract channel correlation features (~50 per TF)
            # Computes cross-correlation between TSLA and SPY channel features
            corr_features = _extract_channel_correlation_features_for_tf(
                tf_channels, spy_tf_channels, tf
            )
            all_features.update(corr_features)

            # 5. Extract aggregated features (100 per TF)
            # Note: For now, passing None for history - caller can provide
            aggregated_features = _extract_aggregated_features_for_tf(
                tf_channels,
                None,  # tsla_channel_history
                None,  # spy_channel_history
                tf
            )
            all_features.update(aggregated_features)

        except FeatureExtractionError:
            raise
        except Exception as e:
            raise FeatureExtractionError(f"Error processing timeframe {tf}: {e}")

    # =========================================================================
    # Extract global (TF-independent) features
    # =========================================================================

    # Event features (30)
    try:
        event_feats = extract_event_features(timestamp, tsla_df)
        all_features.update(event_feats)
    except Exception as e:
        raise FeatureExtractionError(f"Event extraction failed: {e}")

    # Bar metadata features (33)
    bar_meta_feats = _extract_bar_metadata_features(metadata_by_tf)
    all_features.update(bar_meta_feats)

    # =========================================================================
    # Validation
    # =========================================================================

    # Check feature count
    n_features = len(all_features)
    if n_features != TOTAL_FEATURES:
        if validate:
            raise FeatureExtractionError(
                f"Feature count mismatch: got {n_features}, expected {TOTAL_FEATURES}"
            )
        else:
            logger.warning(
                f"Feature count mismatch: got {n_features}, expected {TOTAL_FEATURES}"
            )

    # Validate all features (no NaN, no Inf)
    if validate:
        invalid_features = _validate_feature_dict(all_features, raise_on_invalid=False)
        if invalid_features:
            # Report first few invalid features
            sample = invalid_features[:5]
            sample_values = {k: all_features.get(k) for k in sample}
            raise InvalidFeatureError(
                feature_name=invalid_features[0],
                value=all_features.get(invalid_features[0]),
                message=(
                    f"Found {len(invalid_features)} invalid features. "
                    f"First 5: {sample_values}"
                )
            )

    logger.debug(f"Extracted {n_features} features successfully")
    return all_features


# =============================================================================
# Feature Metadata Functions
# =============================================================================

def get_feature_count() -> int:
    """Return total expected feature count (8,665)."""
    return TOTAL_FEATURES


def get_feature_names() -> List[str]:
    """
    Return all feature names in consistent order.

    Order:
    1. For each TF (in TIMEFRAMES order):
       a. Window-independent features (price, technical, spy, vix, cross_asset)
       b. Per-window channel features (for each window)
       c. Aggregated features (window_scores, channel_history)
    2. Event features (global)
    3. Bar metadata features (per TF)

    Returns:
        List of all 8,665 feature names
    """
    # This would be populated from the actual feature extractors
    # For now, return a computed list based on expected structure
    all_names: List[str] = []

    # Per-TF features
    for tf in TIMEFRAMES:
        # Window-independent (282)
        for i in range(FEATURE_COUNTS['tsla_price_per_tf']):
            all_names.append(f"{tf}_price_{i}")
        for i in range(FEATURE_COUNTS['technical_per_tf']):
            all_names.append(f"{tf}_technical_{i}")
        for i in range(FEATURE_COUNTS['spy_per_tf']):
            all_names.append(f"{tf}_spy_{i}")
        for i in range(FEATURE_COUNTS['vix_per_tf']):
            all_names.append(f"{tf}_vix_{i}")
        for i in range(FEATURE_COUNTS['cross_asset_per_tf']):
            all_names.append(f"{tf}_cross_{i}")

        # Per-window channel (400)
        for window in STANDARD_WINDOWS:
            for i in range(FEATURE_COUNTS['channel_per_window']):
                all_names.append(f"{tf}_w{window}_channel_{i}")

        # Aggregated (100)
        for i in range(FEATURE_COUNTS['window_scores_per_tf']):
            all_names.append(f"{tf}_wscore_{i}")
        for i in range(FEATURE_COUNTS['channel_history_per_tf']):
            all_names.append(f"{tf}_history_{i}")

    # Event features (30)
    for i in range(FEATURE_COUNTS['events_total']):
        all_names.append(f"event_{i}")

    # Bar metadata features (33)
    for tf in TIMEFRAMES:
        all_names.append(f"{tf}_bar_completion_pct")
        all_names.append(f"{tf}_bars_in_partial")
        all_names.append(f"{tf}_complete_bars")

    return all_names


def get_feature_breakdown() -> Dict[str, int]:
    """
    Return detailed feature count breakdown.

    Returns:
        Dict with category names and counts
    """
    return {
        'total_features': TOTAL_FEATURES,
        'n_timeframes': N_TIMEFRAMES,
        'n_windows': N_WINDOWS,

        # Per-TF breakdown
        'window_independent_per_tf': (
            FEATURE_COUNTS['tsla_price_per_tf'] +
            FEATURE_COUNTS['technical_per_tf'] +
            FEATURE_COUNTS['spy_per_tf'] +
            FEATURE_COUNTS['vix_per_tf'] +
            FEATURE_COUNTS['cross_asset_per_tf']
        ),  # 282
        'channel_per_tf': FEATURE_COUNTS['channel_per_window'] * N_WINDOWS,  # 400
        'aggregated_per_tf': (
            FEATURE_COUNTS['window_scores_per_tf'] +
            FEATURE_COUNTS['channel_history_per_tf']
        ),  # 100

        # Category details
        'tsla_price_per_tf': FEATURE_COUNTS['tsla_price_per_tf'],
        'technical_per_tf': FEATURE_COUNTS['technical_per_tf'],
        'spy_per_tf': FEATURE_COUNTS['spy_per_tf'],
        'vix_per_tf': FEATURE_COUNTS['vix_per_tf'],
        'cross_asset_per_tf': FEATURE_COUNTS['cross_asset_per_tf'],
        'channel_per_window': FEATURE_COUNTS['channel_per_window'],
        'window_scores_per_tf': FEATURE_COUNTS['window_scores_per_tf'],
        'channel_history_per_tf': FEATURE_COUNTS['channel_history_per_tf'],
        'events_total': FEATURE_COUNTS['events_total'],
        'bar_metadata_per_tf': FEATURE_COUNTS['bar_metadata_per_tf'],
        'bar_metadata_total': FEATURE_COUNTS['bar_metadata_per_tf'] * N_TIMEFRAMES,
    }


def get_feature_group_counts() -> Dict[str, int]:
    """
    Get the count of features in each group.

    Returns:
        Dict mapping group name to feature count
    """
    from ..config import FEATURE_COUNTS, N_TIMEFRAMES, N_WINDOWS

    return {
        'tsla_price': FEATURE_COUNTS['tsla_price_per_tf'] * N_TIMEFRAMES,
        'technical': FEATURE_COUNTS['technical_per_tf'] * N_TIMEFRAMES,
        'spy': FEATURE_COUNTS['spy_per_tf'] * N_TIMEFRAMES,
        'vix': FEATURE_COUNTS['vix_per_tf'] * N_TIMEFRAMES,
        'cross_asset': FEATURE_COUNTS['cross_asset_per_tf'] * N_TIMEFRAMES,
        'channel': FEATURE_COUNTS['channel_per_window'] * N_WINDOWS * N_TIMEFRAMES,
        'window_scores': FEATURE_COUNTS['window_scores_per_tf'] * N_TIMEFRAMES,
        'channel_history': FEATURE_COUNTS['channel_history_per_tf'] * N_TIMEFRAMES,
        'events': FEATURE_COUNTS['events_total'],
        'bar_metadata': FEATURE_COUNTS['bar_metadata_per_tf'] * N_TIMEFRAMES,
    }


def validate_features(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Validate a feature dictionary and report issues.

    Args:
        features: Dictionary of features to validate

    Returns:
        {
            'valid': bool,
            'n_features': int,
            'expected_features': int,
            'missing': List[str],
            'extra': List[str],
            'invalid': List[str],  # NaN/Inf values
        }

    Raises:
        InvalidFeatureError: If any feature is NaN/Inf
    """
    expected_names = set(get_feature_names())
    actual_names = set(features.keys())

    # Check for invalid values
    invalid = _validate_feature_dict(features, raise_on_invalid=False)

    result = {
        'valid': len(invalid) == 0 and actual_names == expected_names,
        'n_features': len(features),
        'expected_features': TOTAL_FEATURES,
        'missing': sorted(expected_names - actual_names),
        'extra': sorted(actual_names - expected_names),
        'invalid': invalid,
    }

    return result


# =============================================================================
# Convenience Functions
# =============================================================================

def create_feature_vector(
    features: Dict[str, float],
    names: Optional[List[str]] = None
) -> np.ndarray:
    """
    Create numpy array from feature dict in consistent order.

    Args:
        features: Dictionary of features
        names: Optional list of names (defaults to get_feature_names())

    Returns:
        Numpy array of shape (TOTAL_FEATURES,)
    """
    if names is None:
        names = get_feature_names()

    vector = np.zeros(len(names), dtype=np.float64)

    for i, name in enumerate(names):
        vector[i] = features.get(name, 0.0)

    return vector


def features_to_dataframe(
    features: Dict[str, float],
    timestamp: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Convert feature dict to single-row DataFrame.

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


# =============================================================================
# Legacy Wrapper (backwards compatibility)
# =============================================================================

def extract_features(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    channel: "Channel",
    window: int,
    channels_by_window: Optional[Dict[int, "Channel"]] = None
) -> Dict[str, float]:
    """
    Legacy interface for backwards compatibility.

    DEPRECATED: Use extract_all_features() instead.
    """
    import warnings
    warnings.warn(
        "extract_features() is deprecated. Use extract_all_features() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    if channels_by_window is None:
        channels_by_window = {window: channel}

    timestamp = tsla_df.index[-1] if len(tsla_df) > 0 else pd.Timestamp.now()

    return extract_all_features(
        tsla_df=tsla_df,
        spy_df=spy_df,
        vix_df=vix_df,
        timestamp=timestamp,
        channels_by_window=channels_by_window,
        validate=True,
    )


# =============================================================================
# Module Status
# =============================================================================

def get_available_modules() -> Dict[str, bool]:
    """Get availability status of all dependencies."""
    return {
        'channel_detection': _channel_detection_available,
        'resampler': True,  # Required, import would fail if not available
        'tsla_price': True,
        'technical': True,
        'spy': True,
        'vix': True,
        'cross_asset': True,
        'tsla_channel': True,
        'spy_channel': True,
        'channel_correlation': True,
        'window_scores': True,
        'channel_history': True,
        'events': True,
        'validation': True,
    }
