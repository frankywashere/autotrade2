"""
V15 Configuration - Single source of truth for all constants.
"""
from typing import Dict, List

# Timeframes
TIMEFRAMES: List[str] = [
    '5min', '15min', '30min', '1h', '2h', '3h', '4h',
    'daily', 'weekly', 'monthly', '3month'
]
N_TIMEFRAMES: int = len(TIMEFRAMES)

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
    '3month': 4914,   # ~63 trading days * 78
}

# Feature counts per category
FEATURE_COUNTS = {
    'tsla_price_per_tf': 60,
    'technical_per_tf': 77,
    'spy_per_tf': 80,
    'vix_per_tf': 25,
    'cross_asset_per_tf': 40,
    'channel_per_window': 50,
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
)  # 282

WINDOW_DEPENDENT_PER_TF = FEATURE_COUNTS['channel_per_window'] * N_WINDOWS  # 400

AGGREGATED_PER_TF = (
    FEATURE_COUNTS['window_scores_per_tf'] +
    FEATURE_COUNTS['channel_history_per_tf']
)  # 100

BAR_METADATA_TOTAL = FEATURE_COUNTS['bar_metadata_per_tf'] * N_TIMEFRAMES  # 33

FEATURES_PER_TF = WINDOW_INDEPENDENT_PER_TF + WINDOW_DEPENDENT_PER_TF + AGGREGATED_PER_TF  # 782

TOTAL_TF_FEATURES = FEATURES_PER_TF * N_TIMEFRAMES  # 8,602
TOTAL_FEATURES = TOTAL_TF_FEATURES + FEATURE_COUNTS['events_total'] + BAR_METADATA_TOTAL  # 8,665

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
    'weekly': 52,
    'monthly': 24,
    '3month': 12,
}

TF_RETURN_THRESHOLD: Dict[str, int] = {
    '5min': 25,
    '15min': 20,
    '30min': 18,
    '1h': 15,
    '2h': 12,
    '3h': 10,
    '4h': 8,
    'daily': 5,
    'weekly': 3,
    'monthly': 2,
    '3month': 1,
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
SCANNER_CONFIG = {
    'step': 10,
    'warmup_bars': 32760,
    'forward_bars': 8000,
    'workers': 4,
}

# Validation thresholds
CORRELATION_THRESHOLD = 0.95  # Warn if features correlated above this
MAX_ALLOWED_NAN_RATIO = 0.0   # No NaN allowed (loud failure)
