"""
V15 Features Package

Unified feature extraction system for x14 trading model.

Standard Mode (per sample, no TF awareness):
    extract_all_features() - Extract features from market data

TF-Aware Mode (full timeframe awareness):
    extract_all_tf_features() - Extract features across all timeframes

Feature Groups (Standard):
    - tsla_channel: Channel position, width, slope features
    - tsla_price: Price action, momentum, patterns
    - technical: RSI, MACD, Bollinger, etc.
    - spy: SPY market features and correlation
    - spy_channel: SPY channel features
    - channel_correlation: TSLA-SPY channel correlation
    - vix: Volatility index features
    - cross_asset: Cross-asset relationships
    - channel_history: Historical channel patterns
    - events: Calendar, earnings, macro events
    - window_scores: Window optimization scores

Feature Groups (TF-Aware):
    See config.py for current feature counts:
    - FEATURE_COUNTS: Per-category counts
    - WINDOW_INDEPENDENT_PER_TF: Features not dependent on window
    - WINDOW_DEPENDENT_PER_TF: Channel features per window
    - TOTAL_FEATURES: Total feature count
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Feature Count Constants
# =============================================================================

# Old counts (per sample, no TF awareness)
FEATURE_GROUPS = {
    'tsla_channel': 50,
    'tsla_price': 60,
    'technical': 77,
    'spy': 80,
    'spy_channel': 58,
    'channel_correlation': 50,
    'vix': 25,
    'cross_asset': 40,
    'channel_history': 50,
    'events': 30,
    'window_scores': 50,
}
TOTAL_FEATURES = 570  # Updated to include spy_channel and channel_correlation

# New counts (with full TF awareness)
TF_FEATURE_GROUPS = {
    'tsla_price': 60 * 10,                # 600
    'technical': 77 * 10,                  # 770
    'spy': 80 * 10,                        # 800
    'vix': 25 * 10,                        # 250
    'cross_asset': 40 * 10,                # 400
    'tsla_channel': 50 * 8 * 10,           # 4,000 (per window per TF)
    'spy_channel': 58 * 8 * 10,            # 4,640 (per window per TF)
    'channel_correlation': 50 * 10,        # 500 (per TF)
    'window_scores': 50 * 10,              # 500
    'channel_history': 50 * 10,            # 500
    'events': 30,                          # 30 (TF-independent)
    'bar_metadata': 3 * 10,                # 30 (per TF)
}
TOTAL_TF_FEATURES = 13020  # Updated to include spy_channel, channel_correlation, bar_metadata

# =============================================================================
# Main Extractor Imports
# =============================================================================

try:
    from .extractor import (
        extract_all_features,
        extract_features,  # Legacy wrapper for scanner compatibility
        get_feature_names,
        get_feature_count,
        get_feature_group_counts,
        get_available_modules,
        validate_features,
        create_feature_vector,
        features_to_dataframe,
    )
    _extractor_available = True
except ImportError as e:
    logger.warning(f"Failed to import extractor: {e}")
    _extractor_available = False

# =============================================================================
# TF-Aware Extractor Imports
# =============================================================================

try:
    from .tf_extractor import (
        extract_all_tf_features,
        get_tf_feature_count,
        get_tf_feature_names,
        get_tf_feature_breakdown,
        TF_FEATURE_COUNTS,
        TOTAL_FEATURES as _TF_TOTAL,  # Use local constant, import for validation
    )
    _tf_extractor_available = True
except ImportError as e:
    logger.debug(f"tf_extractor module not available: {e}")
    _tf_extractor_available = False

# =============================================================================
# Utility Imports
# =============================================================================

try:
    from .utils import (
        # Safe basic operations
        safe_float,
        safe_divide,
        safe_pct_change,
        safe_mean,
        safe_std,
        safe_min,
        safe_max,
        safe_sum,
        # Rolling calculations
        rolling_mean,
        rolling_std,
        rolling_correlation,
        # Technical indicators - moving averages
        calc_sma,
        calc_ema,
        # Technical indicators - momentum
        calc_rsi,
        calc_momentum,
        calc_roc,
        calc_stochastic,
        # Technical indicators - volatility
        calc_atr,
        calc_bollinger_bands,
        # Technical indicators - trend
        calc_macd,
        # Normalization and scoring
        normalize_values,
        zscore,
        crossover,
        # Utilities
        ensure_finite,
        get_last_valid,
        apply_to_columns,
        true_range,
        # Legacy aliases
        ema,
        sma,
        rsi,
        atr,
    )
    _utils_available = True
except ImportError as e:
    logger.warning(f"Failed to import utils: {e}")
    _utils_available = False

# =============================================================================
# Feature Module Imports (with graceful error handling)
# =============================================================================

# TSLA Channel Features
try:
    from .tsla_channel import (
        extract_tsla_channel_features,
        extract_tsla_channel_features_tf,
        get_tsla_channel_feature_names,
        get_all_tsla_channel_feature_names,
        get_tsla_channel_feature_count,
    )
    _tsla_channel_available = True
except ImportError as e:
    logger.debug(f"tsla_channel module not available: {e}")
    _tsla_channel_available = False

# TSLA Price Features
try:
    from .tsla_price import (
        extract_tsla_price_features,
        extract_tsla_price_features_tf,
        get_tsla_price_feature_names,
        get_all_tsla_price_feature_names,
        get_tsla_price_feature_count,
        get_total_price_features,
        TSLA_PRICE_FEATURE_NAMES,
    )
    _tsla_price_available = True
except ImportError as e:
    logger.debug(f"tsla_price module not available: {e}")
    _tsla_price_available = False

# Technical Features
try:
    from .technical import (
        extract_technical_features,
        extract_technical_features_tf,
        get_all_technical_feature_names,
    )
    _technical_available = True
except ImportError as e:
    logger.debug(f"technical module not available: {e}")
    _technical_available = False

# SPY Features
try:
    from .spy import (
        extract_spy_features,
        extract_spy_features_tf,
        get_spy_feature_names,
        get_all_spy_feature_names,
        get_spy_feature_count,
    )
    _spy_available = True
except ImportError as e:
    logger.debug(f"spy module not available: {e}")
    _spy_available = False

# VIX Features
try:
    from .vix import (
        extract_vix_features,
        extract_vix_features_tf,
        get_all_vix_feature_names,
    )
    _vix_available = True
except ImportError as e:
    logger.debug(f"vix module not available: {e}")
    _vix_available = False

# Cross-Asset Features
try:
    from .cross_asset import (
        extract_cross_asset_features,
        extract_cross_asset_features_tf,
        get_all_cross_asset_feature_names,
    )
    _cross_asset_available = True
except ImportError as e:
    logger.debug(f"cross_asset module not available: {e}")
    _cross_asset_available = False

# Channel History Features
try:
    from .channel_history import (
        extract_channel_history_features,
        extract_channel_history_features_tf,
        get_all_channel_history_feature_names,
    )
    _channel_history_available = True
except ImportError as e:
    logger.debug(f"channel_history module not available: {e}")
    _channel_history_available = False

# Event Features
try:
    from .events import (
        extract_event_features,
    )
    _events_available = True
except ImportError as e:
    logger.debug(f"events module not available: {e}")
    _events_available = False

# Window Score Features
try:
    from .window_scores import (
        extract_window_score_features,
        extract_window_score_features_tf,
        get_window_score_feature_names,
        get_all_window_score_feature_names,
        get_window_score_feature_count,
        WINDOW_SCORE_FEATURE_NAMES,
        STANDARD_WINDOWS,
    )
    _window_scores_available = True
except ImportError as e:
    logger.debug(f"window_scores module not available: {e}")
    _window_scores_available = False

# SPY Channel Features
try:
    from .spy_channel import (
        extract_spy_channel_features,
        extract_spy_channel_features_tf,
        get_spy_channel_feature_names,
        get_all_spy_channel_feature_names,
        get_spy_channel_feature_count,
        get_total_spy_channel_features,
    )
    _spy_channel_available = True
except ImportError as e:
    logger.debug(f"spy_channel module not available: {e}")
    _spy_channel_available = False

# Channel Correlation Features
try:
    from .channel_correlation import (
        extract_channel_correlation_features,
        get_channel_correlation_feature_names,
        get_channel_correlation_feature_names_tf,
        get_channel_correlation_feature_count,
        get_all_channel_correlation_feature_names,
        get_total_channel_correlation_features,
        extract_channel_correlation_features_multi_window,
    )
    _channel_correlation_available = True
except ImportError as e:
    logger.debug(f"channel_correlation module not available: {e}")
    _channel_correlation_available = False

# Validation Module
try:
    from .validation import (
        validate_features,
        validate_feature_matrix,
        analyze_correlations,
        check_for_constant_features,
        get_feature_stats,
        run_full_validation,
    )
    _validation_available = True
except ImportError as e:
    logger.debug(f"validation module not available: {e}")
    _validation_available = False


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Main extractor (standard mode)
    'extract_all_features',
    'extract_features',  # Legacy wrapper for scanner compatibility
    'get_feature_names',
    'get_feature_count',
    'get_feature_group_counts',
    'get_available_modules',
    'validate_features',
    'create_feature_vector',
    'features_to_dataframe',
    # Feature count constants (standard)
    'FEATURE_GROUPS',
    'TOTAL_FEATURES',
    # Feature count constants (TF-aware)
    'TF_FEATURE_GROUPS',
    'TOTAL_TF_FEATURES',
    # Utilities - safe operations
    'safe_float',
    'safe_divide',
    'safe_pct_change',
    'safe_mean',
    'safe_std',
    'safe_min',
    'safe_max',
    'safe_sum',
    # Utilities - rolling calculations
    'rolling_mean',
    'rolling_std',
    'rolling_correlation',
    # Utilities - technical indicators
    'calc_sma',
    'calc_ema',
    'calc_rsi',
    'calc_momentum',
    'calc_roc',
    'calc_stochastic',
    'calc_atr',
    'calc_bollinger_bands',
    'calc_macd',
    # Utilities - normalization
    'normalize_values',
    'zscore',
    'crossover',
    # Utilities - misc
    'ensure_finite',
    'get_last_valid',
    'apply_to_columns',
    'true_range',
    # Legacy aliases
    'ema',
    'sma',
    'rsi',
    'atr',
    # Standard feature extractors
    'extract_tsla_channel_features',
    'extract_tsla_price_features',
    'extract_technical_features',
    'extract_spy_features',
    'extract_vix_features',
    # Feature metadata (standard)
    'get_tsla_channel_feature_names',
    'get_tsla_channel_feature_count',
    'get_tsla_price_feature_names',
    'get_tsla_price_feature_count',
    'get_spy_feature_names',
    'get_spy_feature_count',
    'TSLA_PRICE_FEATURE_NAMES',
]

# Conditionally add TF extractor exports
if _tf_extractor_available:
    __all__.extend([
        'extract_all_tf_features',
        'get_tf_feature_count',
        'get_tf_feature_names',
        'get_tf_feature_breakdown',
        'TF_FEATURE_COUNTS',
    ])

# Conditionally add TF-aware module functions
if _tsla_price_available:
    __all__.extend([
        'extract_tsla_price_features_tf',
        'get_all_tsla_price_feature_names',
        'get_total_price_features',
    ])

if _technical_available:
    __all__.extend([
        'extract_technical_features_tf',
        'get_all_technical_feature_names',
    ])

if _spy_available:
    __all__.extend([
        'extract_spy_features_tf',
        'get_all_spy_feature_names',
    ])

if _vix_available:
    __all__.extend([
        'extract_vix_features_tf',
        'get_all_vix_feature_names',
    ])

if _cross_asset_available:
    __all__.extend([
        'extract_cross_asset_features',
        'extract_cross_asset_features_tf',
        'get_all_cross_asset_feature_names',
    ])

if _channel_history_available:
    __all__.extend([
        'extract_channel_history_features',
        'extract_channel_history_features_tf',
        'get_all_channel_history_feature_names',
    ])

if _events_available:
    __all__.append('extract_event_features')

if _window_scores_available:
    __all__.extend([
        'extract_window_score_features',
        'extract_window_score_features_tf',
        'get_window_score_feature_names',
        'get_all_window_score_feature_names',
        'get_window_score_feature_count',
        'WINDOW_SCORE_FEATURE_NAMES',
        'STANDARD_WINDOWS',
    ])

if _tsla_channel_available:
    __all__.extend([
        'extract_tsla_channel_features_tf',
        'get_all_tsla_channel_feature_names',
    ])

if _spy_channel_available:
    __all__.extend([
        'extract_spy_channel_features',
        'extract_spy_channel_features_tf',
        'get_spy_channel_feature_names',
        'get_all_spy_channel_feature_names',
        'get_spy_channel_feature_count',
        'get_total_spy_channel_features',
    ])

if _channel_correlation_available:
    __all__.extend([
        'extract_channel_correlation_features',
        'get_channel_correlation_feature_names',
        'get_channel_correlation_feature_names_tf',
        'get_channel_correlation_feature_count',
        'get_all_channel_correlation_feature_names',
        'get_total_channel_correlation_features',
        'extract_channel_correlation_features_multi_window',
    ])

if _validation_available:
    __all__.extend([
        'validate_features',
        'validate_feature_matrix',
        'analyze_correlations',
        'check_for_constant_features',
        'get_feature_stats',
        'run_full_validation',
    ])


# =============================================================================
# Module Status
# =============================================================================

def get_module_status() -> dict:
    """
    Get the availability status of all feature modules.

    Returns:
        Dictionary mapping module name to availability boolean.
    """
    return {
        'extractor': _extractor_available,
        'tf_extractor': _tf_extractor_available,
        'utils': _utils_available,
        'tsla_channel': _tsla_channel_available,
        'tsla_price': _tsla_price_available,
        'technical': _technical_available,
        'spy': _spy_available,
        'spy_channel': _spy_channel_available,
        'channel_correlation': _channel_correlation_available,
        'vix': _vix_available,
        'cross_asset': _cross_asset_available,
        'channel_history': _channel_history_available,
        'events': _events_available,
        'window_scores': _window_scores_available,
        'validation': _validation_available,
    }


# Log module status on import (debug level)
_status = get_module_status()
_available = sum(1 for v in _status.values() if v)
_total = len(_status)
logger.debug(f"v15.features loaded: {_available}/{_total} modules available")
