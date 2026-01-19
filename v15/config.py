"""
V15 Configuration - Single source of truth for all constants.
"""
from typing import Dict, List

# Timeframes (10 TFs - no 3month due to data limitations)
TIMEFRAMES: List[str] = [
    '5min', '15min', '30min', '1h', '2h', '3h', '4h',
    'daily', 'weekly', 'monthly'
]
N_TIMEFRAMES: int = len(TIMEFRAMES)  # 10

# Channel detection windows
STANDARD_WINDOWS: List[int] = [10, 20, 30, 40, 50, 60, 70, 80]
N_WINDOWS: int = len(STANDARD_WINDOWS)

# Bars per timeframe (5min bars per TF bar)
BARS_PER_TF: Dict[str, int] = {
    '5min': 1,
    '15min': 3,
    '30min': 6,
    '1h': 12,
    '2h': 24,
    '3h': 36,
    '4h': 48,
    'daily': 78,      # 6.5 hours * 12
    'weekly': 390,    # 5 days * 78
    'monthly': 1638,  # ~21 trading days * 78
}

# Per-TF lookback requirements (in 5-min bars)
# Based on max window (80 bars) * BARS_PER_TF + buffer for technical indicators
TF_LOOKBACK_5MIN: Dict[str, int] = {
    '5min': 200,        # 80 + buffer for RSI/MACD
    '15min': 400,       # 80 * 3 + buffer
    '30min': 700,       # 80 * 6 + buffer
    '1h': 1200,         # 80 * 12 + buffer
    '2h': 2200,         # 80 * 24 + buffer
    '3h': 3200,         # 80 * 36 + buffer
    '4h': 4200,         # 80 * 48 + buffer
    'daily': 7000,      # 80 * 78 + buffer
    'weekly': 32000,    # 80 * 390 + buffer
    'monthly': 132000,  # 80 * 1638 + buffer
}

# Per-TF forward requirements for label scanning (in 5-min bars)
# Based on TF_MAX_SCAN * BARS_PER_TF
TF_FORWARD_5MIN: Dict[str, int] = {
    '5min': 600,        # 500 + buffer
    '15min': 1400,      # 400 * 3 + buffer
    '30min': 2400,      # 350 * 6 + buffer
    '1h': 4000,         # 300 * 12 + buffer
    '2h': 6500,         # 250 * 24 + buffer
    '3h': 8000,         # 200 * 36 + buffer
    '4h': 8000,         # 150 * 48 + buffer
    'daily': 8500,      # 100 * 78 + buffer
    'weekly': 21000,    # 52 * 390 + buffer
    'monthly': 40000,   # 24 * 1638 + buffer
}

# Maximum lookback/forward for SCANNER (determines what positions can be scanned)
# NOTE: We use WEEKLY as the practical limit for scanning.
# Monthly TF will have invalid labels for most samples due to data limits.
# The model handles missing labels via masking.
#
# For a 440K bar dataset:
#   - Weekly lookback (32K) + forward (21K) = 53K (reasonable)
#   - Monthly (172K) exceeds our data
#
# Per-TF lookback values above are still used for EFFICIENT SLICING during
# feature extraction - each TF only resamples the data it needs.

# Practical limits for scanner (based on weekly TF)
SCANNER_LOOKBACK_5MIN = TF_LOOKBACK_5MIN['weekly']   # 32,000
SCANNER_FORWARD_5MIN = TF_FORWARD_5MIN['weekly']     # 21,000

# True maximums (for reference - may exceed available data)
MAX_LOOKBACK_5MIN = max(TF_LOOKBACK_5MIN.values())   # 132,000 (monthly)
MAX_FORWARD_5MIN = max(TF_FORWARD_5MIN.values())     # 40,000 (monthly)

# Feature counts per category
# UPDATE: Added SPY channel features (58 per window) and channel correlation (50 per TF)
FEATURE_COUNTS = {
    'tsla_price_per_tf': 60,
    'technical_per_tf': 77,
    'spy_per_tf': 80,
    'vix_per_tf': 25,
    'cross_asset_per_tf': 40,
    'channel_per_window': 58,           # TSLA channel features per window (50 base + 8 excursion)
    'spy_channel_per_window': 58,       # SPY channel features per window (50 base + 8 excursion)
    'channel_correlation_per_tf': 50,   # TSLA-SPY channel correlation features per TF
    'window_scores_per_tf': 50,
    'channel_history_per_tf': 50,
    'events_total': 30,
    'bar_metadata_per_tf': 3,  # completion_pct, bars_in_partial, complete_bars
}

# Calculated totals
WINDOW_INDEPENDENT_PER_TF = (
    FEATURE_COUNTS['tsla_price_per_tf'] +
    FEATURE_COUNTS['technical_per_tf'] +
    FEATURE_COUNTS['spy_per_tf'] +
    FEATURE_COUNTS['vix_per_tf'] +
    FEATURE_COUNTS['cross_asset_per_tf']
)  # 60 + 77 + 80 + 25 + 40 = 282

# TSLA channel + SPY channel, both per window
WINDOW_DEPENDENT_PER_TF = (
    FEATURE_COUNTS['channel_per_window'] +      # TSLA: 58 per window
    FEATURE_COUNTS['spy_channel_per_window']    # SPY: 58 per window
) * N_WINDOWS  # (58 + 58) * 8 = 928

AGGREGATED_PER_TF = (
    FEATURE_COUNTS['window_scores_per_tf'] +
    FEATURE_COUNTS['channel_history_per_tf'] +
    FEATURE_COUNTS['channel_correlation_per_tf']  # Cross-channel correlation features
)  # 50 + 50 + 50 = 150

BAR_METADATA_TOTAL = FEATURE_COUNTS['bar_metadata_per_tf'] * N_TIMEFRAMES  # 3 * 10 = 30

FEATURES_PER_TF = WINDOW_INDEPENDENT_PER_TF + WINDOW_DEPENDENT_PER_TF + AGGREGATED_PER_TF  # 282 + 928 + 150 = 1,360

TOTAL_TF_FEATURES = FEATURES_PER_TF * N_TIMEFRAMES  # 1,360 * 10 = 13,600
TOTAL_FEATURES = TOTAL_TF_FEATURES + FEATURE_COUNTS['events_total'] + BAR_METADATA_TOTAL  # 13,600 + 30 + 30 = 13,660

# Label scanning parameters per TF
TF_MAX_SCAN: Dict[str, int] = {
    '5min': 500,
    '15min': 400,
    '30min': 350,
    '1h': 300,
    '2h': 250,
    '3h': 200,
    '4h': 150,
    'daily': 100,
    'weekly': 26,   # ~6 months
    'monthly': 6,   # ~6 months
}

# Model configuration
MODEL_CONFIG = {
    'input_dim': TOTAL_FEATURES,
    'hidden_dim': 256,
    'n_attention_heads': 8,
    'dropout': 0.1,
    'use_explicit_weights': True,
}

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 64,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'warmup_steps': 1000,
    'max_epochs': 100,
}

# Scanner configuration
# Uses practical limits (weekly-based) to ensure we can scan with available data
SCANNER_CONFIG = {
    'step': 10,
    'warmup_bars': SCANNER_LOOKBACK_5MIN,   # 32,000 (weekly-based, practical)
    'forward_bars': SCANNER_FORWARD_5MIN,    # 21,000 (weekly-based, practical)
    'workers': 4,
}

# Validation thresholds
CORRELATION_THRESHOLD = 0.95  # Warn if features correlated above this
MAX_ALLOWED_NAN_RATIO = 0.0   # No NaN allowed (loud failure)
