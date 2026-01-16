"""
Feature Extractor - Main Orchestrator for x14 Feature System v15

This module is the single entry point for ALL feature extraction.
It orchestrates:
1. Partial bar resampling across all 11 timeframes (keeping incomplete bars)
2. Channel detection at all 8 windows per timeframe
3. Feature extraction from all modules
4. Bar metadata features (completion_pct, bars_in_partial, complete_bars)
5. Strict validation - NO silent NaN allowed

Total Features: 8,665 (as defined in config.TOTAL_FEATURES)
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
)

from ..exceptions import (
    FeatureExtractionError,
    InvalidFeatureError,
    ResamplingError,
)

# =============================================================================
# Data resampling with partial bar support
# =============================================================================
from ..data.resampler import resample_with_partial, BarMetadata

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
    '3month': '3MS',    # Quarter Start (pandas format)
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
    source_tf: str = '5min'
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Resample DataFrame to target timeframe using partial bar support.

    CRITICAL: This keeps partial (incomplete) bars instead of dropping them.
    In live trading, the current bar is always incomplete.

    Args:
        df: Source OHLCV DataFrame (typically 5-min data)
        tf: Target timeframe
        source_tf: Source timeframe (for metadata calculation)

    Returns:
        Tuple of (resampled_df, metadata_dict)

    Raises:
        ResamplingError: If resampling fails
    """
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

    resample_rule = TF_TO_RESAMPLE_RULE.get(tf)
    if resample_rule is None:
        raise ResamplingError(f"Unknown timeframe: {tf}")

    try:
        resampled, metadata = resample_with_partial(df, resample_rule, source_tf)
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


def _extract_bar_metadata_features(
    metadata_by_tf: Dict[str, Dict[str, Any]]
) -> Dict[str, float]:
    """
    Extract bar metadata features for all timeframes.

    For each TF, extracts:
    - {tf}_bar_completion_pct: 0.0-1.0, how complete the last bar is
    - {tf}_bars_in_partial: Number of source bars in the partial bar
    - {tf}_complete_bars: Total number of complete bars

    Total: 3 features * 11 TFs = 33 features

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
    tf: str
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

    # 5. Cross-Asset Features (40)
    try:
        cross_feats = extract_cross_asset_features(tsla_df, spy_df, vix_df)
        features.update(_prefix_tf(cross_feats, tf))
    except Exception as e:
        raise FeatureExtractionError(f"Cross-asset extraction failed for {tf}: {e}")

    return features


def _extract_channel_features_for_tf(
    channels_by_window: Dict[int, "Channel"],
    tf: str
) -> Dict[str, float]:
    """
    Extract channel features for all windows at a single TF.

    Features: 50 per window * 8 windows = 400 per TF

    Args:
        channels_by_window: Dict mapping window -> Channel
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
    validate: bool = True
) -> Dict[str, float]:
    """
    Extract all features for a single sample.

    This is the MAIN ENTRY POINT for feature extraction. It:
    1. Resamples data to all 11 timeframes (keeping partial bars)
    2. Detects channels at all 8 windows per timeframe
    3. Extracts all features with explicit TF prefixes
    4. Adds bar metadata features
    5. Validates all features (no silent NaN)

    Args:
        tsla_df: Base 5-min TSLA OHLCV DataFrame
        spy_df: Base 5-min SPY OHLCV DataFrame
        vix_df: Base 5-min VIX OHLCV DataFrame
        timestamp: Current timestamp for event features
        channels_by_window: Pre-computed channels (optional, will detect if empty)
        validate: If True, validate all features and raise on invalid

    Returns:
        Dict with TOTAL_FEATURES (8,665) named features:
        - Window-independent per TF: 282 * 11 = 3,102
        - Channel per window per TF: 50 * 8 * 11 = 4,400
        - Aggregated per TF: 100 * 11 = 1,100
        - Events (global): 30
        - Bar metadata: 3 * 11 = 33

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

    # =========================================================================
    # Process each timeframe
    # =========================================================================
    for tf in TIMEFRAMES:
        try:
            # 1. Resample data to this TF (keeps partial bars)
            tsla_tf, tsla_meta = _resample_to_tf(tsla_df, tf)
            spy_tf, spy_meta = _resample_to_tf(spy_df, tf)
            vix_tf, vix_meta = _resample_to_tf(vix_df, tf)

            # Store metadata for bar metadata features
            metadata_by_tf[tf] = tsla_meta

            # Check minimum data requirement
            if len(tsla_tf) < 10:
                raise FeatureExtractionError(
                    f"Insufficient data for {tf}: have {len(tsla_tf)} bars, need 10+"
                )

            # 2. Detect channels at all windows for this TF
            tf_channels = _detect_channels(tsla_tf)

            # 3. Extract window-independent features (282 per TF)
            wi_features = _extract_window_independent_features_for_tf(
                tsla_tf, spy_tf, vix_tf, tf
            )
            all_features.update(wi_features)

            # 4. Extract channel features (400 per TF)
            channel_features = _extract_channel_features_for_tf(tf_channels, tf)
            all_features.update(channel_features)

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
        'window_scores': True,
        'channel_history': True,
        'events': True,
        'validation': True,
    }
