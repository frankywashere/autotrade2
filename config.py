"""Configuration file for trading system."""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SHARD_DIR = DATA_DIR / "feature_cache"  # Directory for mmap shards and cached features

# API Keys
CLAUDE_API_KEY = "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"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7978931435:AAGdqdfcbK-GT8Q_BEw7dvmISkN9035FzZQ")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "7910666732")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "7958854e1a644a109cf28488af5b6d8c")  # NewsAPI.org
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "PYUPDHCVHQQT5I5L")  # For earnings calendar
FRED_API_KEY = os.getenv("FRED_API_KEY", "8e8fc56308f78390f4b44222c01fd449")  # For economic data (CPI, NFP, rates)
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "d4qh0u9r01quli1cimbgd4qh0u9r01quli1cimc0")  # For earnings calendar + company news

# Trading Parameters
DEFAULT_STOCK = "TSLA"
STOCKS = ["TSLA", "SPY"]

# Timeframes for analysis
TIMEFRAMES = {
    "1min": "1T",
    "1hour": "1h",
    "2hour": "2h",
    "3hour": "3h",
    "4hour": "4h",
    "daily": "1D",
    "weekly": "1W"
}

# RSI Parameters
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Linear Regression Channel Parameters
MIN_PING_PONGS_1H = 2  # Minimum bounces for 1-hour stability
MIN_PING_PONGS_4H = 1  # Minimum bounces for 4-hour stability
CHANNEL_LOOKBACK_HOURS = 168  # 1 week lookback
CHANNEL_STD_DEV = 2.0  # Standard deviations for channel width

# News Analysis
NEWS_REFRESH_MINUTES = 30
BS_SCORE_THRESHOLD = 70  # High BS score means ignore bearish news

# Signal Generation
MIN_CONFLUENCE_SCORE = 60  # Minimum score to trigger alert (0-100)

# GUI Settings
CHART_UPDATE_SECONDS = 60
DASHBOARD_PORT = 8501  # For Streamlit (legacy)

# Live Dashboard Settings (v2.0 - FastAPI)
PREDICTION_REFRESH_MINUTES = 15  # Auto-refresh interval for background predictions
ALERT_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to trigger Telegram alert
RUN_24_7 = True  # Run prediction loop outside market hours

# Live Data Settings
USE_LIVE_DATA = True  # Merge CSV with live data from yfinance
LIVE_DATA_DAYS_BACK = 7  # Number of days of live data to fetch (max 7 for 1-min)

# ======================================================================
# STAGE 2: ML MODEL CONFIGURATION
# ======================================================================

# Model Settings
ML_MODEL_TYPE = "LNN"  # LNN, LSTM, or Transformer
LNN_HIDDEN_SIZE = 128
LNN_NUM_LAYERS = 2
LNN_LEARNING_RATE = 0.001
ML_BATCH_SIZE = 16  # Reduced for memory efficiency
ML_SEQUENCE_LENGTH = 200  # Bars to look back (uniform across timeframes for multi-scale learning)

# Training Settings
ML_TRAIN_START_YEAR = 2015
ML_TRAIN_END_YEAR = 2023  # Train on 2015-2023
ML_TEST_YEAR = 2024  # Backtest on 2024
ML_EPOCHS = 50
ML_VALIDATION_SPLIT = 0.1

# Feature Settings
USE_CHANNEL_FEATURES = True
USE_RSI_FEATURES = True
USE_CORRELATION_FEATURES = True
USE_EVENT_FEATURES = True
USE_NEWS_FEATURES = True

# Event Settings
TSLA_EVENTS_FILE = DATA_DIR / "events.csv"  # Canonical events file (483 events, 2015-2025) - earnings have Alpha Vantage data
MACRO_EVENTS_API_KEY = os.getenv("MACRO_API_KEY", "8e8fc56308f78390f4b44222c01fd449")  # FRED API for economic calendar
EVENT_LOOKBACK_DAYS = 14  # Days before/after event to analyze (expanded for better pattern detection)
EVENT_LOOKAHEAD_DAYS = 14

# Prediction Settings
PREDICTION_HORIZON_HOURS = 24  # Forecast 24 hours ahead
PREDICTION_CONFIDENCE_THRESHOLD = 0.7  # Min confidence for alerts
PREDICTION_UPDATE_INTERVAL_HOURS = 4  # Recalculate every 4 hours

# Database Settings
ML_DB_PATH = DATA_DIR / "predictions.db"
ML_DB_TYPE = "sqlite"  # sqlite or postgresql

# Online Learning
ONLINE_LEARNING_ENABLED = True
ONLINE_LEARNING_LR = 0.0001  # Lower LR for stability
ONLINE_LEARNING_UPDATE_FREQUENCY = "daily"  # daily or weekly

# Backtesting
BACKTEST_NUM_SIMULATIONS = 100  # Random days/weeks to test
BACKTEST_RANDOM_SEED = 42

# ======================================================================
# DEVICE/GPU CONFIGURATION
# ======================================================================

# Device Selection
AUTO_SELECT_DEVICE = False  # Auto-select best device (CUDA > MPS > CPU)
FORCE_DEVICE = None  # Force specific device: 'cpu', 'cuda', 'mps', or None
INTERACTIVE_DEVICE_SELECTION = True  # Show interactive device selection menu

# Memory Management
ENABLE_MPS_FALLBACK = True  # Allow MPS to fallback to CPU for unsupported ops
MPS_MAX_MEMORY_MB = None  # Limit MPS memory usage (None = unlimited)
CUDA_MEMORY_FRACTION = 0.9  # Use 90% of GPU memory for CUDA

# Performance Settings
BENCHMARK_ON_STARTUP = False  # Run device benchmark on training start
LOG_DEVICE_OPERATIONS = False  # Log device transfer operations (debugging)

# ======================================================================
# PARALLEL PROCESSING CONFIGURATION
# ======================================================================

# Channel Calculation Parallelization
PARALLEL_CHANNEL_CALC = True  # Use joblib for parallel channel calculations
MAX_PARALLEL_WORKERS = 4      # Default: 4 workers (safe for 32-64GB RAM)
                              # Each worker uses ~15GB during feature extraction
                              # Recommended: RAM_GB / 15, minimum 1
                              # 16GB RAM -> 1 worker
                              # 32GB RAM -> 2 workers
                              # 48GB RAM -> 3 workers
                              # 64GB+ RAM -> 4+ workers
                              # Set to 0 for auto-detection based on available RAM
PARALLEL_BACKEND = 'loky'     # joblib backend: 'loky' (process), 'threading', 'multiprocessing'
PARALLEL_VERBOSE = 0           # joblib verbosity level (0=silent, 10=progress, 50=debug) - using tqdm instead

# ======================================================================
# CHUNKED FEATURE EXTRACTION CONFIGURATION
# ======================================================================

# Memory-efficient chunked processing for feature extraction
USE_CHUNKED_EXTRACTION = None  # None=auto-detect, True=force on, False=force off
CHUNK_SIZE_YEARS = 1           # Process features in 1-year chunks
CHUNK_OVERLAP_MONTHS = 6       # Overlap between chunks to ensure rolling features are complete

# ======================================================================
# HISTORICAL LOOKBACK CONFIGURATION
# ======================================================================

# Feature extraction minimum lookback requirements (v4.0: Updated for 14-window system)
# For 3-month timeframe with 10-bar window (smallest window, longest timeframe):
# 10 bars × 3 months/bar = 30 months = 2.5 years
# 660 trading days × 390 bars/day = 257,400 1-min bars
MIN_LOOKBACK_BARS = 257400     # 2.5 years of 1-min bars (was 25740 in v3.12)
MIN_LOOKBACK_MONTHS = 30       # 2.5 years = 30 months (was 3 in v3.12)

# Continuation analysis lookback requirements (in 1-min bars)
CONTINUATION_LOOKBACK_1H = 25740   # 3 months at 1-min bars (66 trading days × 390 bars/day)
CONTINUATION_LOOKBACK_4H = 98280   # 1 year at 1-min bars (252 trading days × 390 bars/day)

# v5.3.2: Adaptive rolling window for breakdown features (in 1-min bars)
# NOTE: v5.3.3+ uses ADAPTIVE_WINDOW_BARS_NATIVE (native TF resolution) for new calculations
# This remains for backward compatibility with v5.3.2 caches
# Largest window is 3month: 8 quarters × 66 days/quarter × 390 bars/day
MAX_ADAPTIVE_WINDOW_BARS = 206160  # 528 trading days for 3month adaptive window

# ======================================================================
# LIVE DATA FETCH LIMITS (v5.3.3)
# ======================================================================
# yfinance API limits as of December 2024
# IMPORTANT: If yfinance limits change, update here to affect all fetch operations
# Changing these values may require cache regeneration if adaptive windows are affected
YFINANCE_MAX_DAYS = {
    '1min': 7,           # 1m - hard 7-day limit (as of Dec 2024)
    'intraday': 60,      # 5m/15m/30m - 60-day limit (standard intraday)
    '1h': 730,           # Hourly data - 2 years available (730 days)
    'daily': 3650,       # Daily - 10 years (practically unlimited, using conservative 10yr)
    'weekly_monthly': 5475,  # Weekly/monthly - 15 years (adequate for all windows)
}

# Adaptive rolling windows for breakdown features (v5.3.3+)
# These are in NATIVE TF RESOLUTION (not 1-min bars!)
# Example: '5min': 1500 means 1500 5-minute bars (~19 trading days)
# Used by: _calculate_breakdown_at_native_tf() in features.py
ADAPTIVE_WINDOW_BARS_NATIVE = {
    '5min': 1500,     # 1500 5-min bars (~19 trading days)
    '15min': 400,     # 400 15-min bars (~15 trading days)
    '30min': 300,     # 300 30-min bars (~23 trading days)
    '1h': 200,        # 200 1-hour bars (~31 trading days)
    '2h': 100,        # 100 2-hour bars (~31 trading days)
    '3h': 80,         # 80 3-hour bars (~37 trading days)
    '4h': 60,         # 60 4-hour bars (~37 trading days)
    'daily': 100,     # 100 daily bars (100 trading days)
    'weekly': 20,     # 20 weekly bars (~100 trading days / 20 weeks)
    'monthly': 15,    # 15 monthly bars (~330 trading days / 15 months)
    '3month': 8,      # 8 quarters (~528 trading days / 2 years)
}

# Days required for each TF's adaptive window (for validation)
ADAPTIVE_WINDOW_DAYS_REQUIRED = {
    '5min': 19,      # 1500 5-min bars / (78 bars/day)
    '15min': 15,     # 400 15-min bars / (26 bars/day)
    '30min': 23,     # 300 30-min bars / (13 bars/day)
    '1h': 31,        # 200 1-hour bars / (6.5 bars/day)
    '2h': 31,        # 100 2-hour bars / (6.5/2 bars/day)
    '3h': 37,        # 80 3-hour bars
    '4h': 37,        # 60 4-hour bars
    'daily': 100,    # 100 daily bars
    'weekly': 100,   # 20 weekly bars
    'monthly': 330,  # 15 monthly bars
    '3month': 528,   # 8 quarters
}

def validate_live_data_compatibility():
    """
    Validate that adaptive windows can be satisfied by yfinance limits.
    Returns list of warnings for TFs where live data will be insufficient.

    Returns:
        List[str]: Warning messages for insufficient TFs (empty if all OK)
    """
    warnings = []

    for tf, days_needed in ADAPTIVE_WINDOW_DAYS_REQUIRED.items():
        # Determine which yfinance limit applies
        if tf in ['5min', '15min', '30min']:
            max_available = YFINANCE_MAX_DAYS['intraday']  # 60 days
        elif tf in ['1h', '2h', '3h', '4h']:
            max_available = YFINANCE_MAX_DAYS['1h']  # 730 days
        elif tf == 'daily':
            max_available = YFINANCE_MAX_DAYS['daily']  # 3650 days
        else:  # weekly, monthly, 3month
            max_available = YFINANCE_MAX_DAYS['weekly_monthly']  # 5475 days

        if days_needed > max_available:
            warnings.append(
                f"{tf}: needs {days_needed} days for adaptive window, "
                f"but yfinance only provides {max_available} days (use historical CSV supplement)"
            )

    return warnings

# Training dataset warmup settings
SKIP_WARMUP_PERIOD = True      # Automatically exclude insufficient-history timestamps
# Master warmup: max of all requirements (channel windows, adaptive windows, continuation)
WARMUP_BARS = max(MIN_LOOKBACK_BARS, MAX_ADAPTIVE_WINDOW_BARS, CONTINUATION_LOOKBACK_4H)
# = max(257400, 206160, 98280) = 257400 (channel requirement is largest)

# Buffer calculation
AUTO_CALCULATE_BUFFER = True   # Calculate buffer from feature requirements
HISTORICAL_BUFFER_YEARS = 3    # Years of data to load before training start (increased for multi-window)

# ======================================================================
# DATA REQUIREMENTS - IMPORTANT!
# ======================================================================
# MINIMUM DATA REQUIREMENTS BY TIMEFRAME:
#   Intraday (5min-4h):  6 months of data minimum
#   Daily:               1 year of data minimum
#   Weekly/Monthly:      3 years of data minimum (RECOMMENDED: 5 years)
#   3-Month:             3 years minimum (each bar = 3 months)
#
# IMPORTANT: Training will start AFTER the warmup period to ensure
# all timeframes have sufficient lookback data. With 11 timeframes
# including weekly/monthly/3month, you should have AT LEAST 3 YEARS
# of historical data. More is better!
#
# Example: Training on 2020-2023 requires data from 2017-2023 (warmup: 2017-2020)
# ======================================================================

# ======================================================================
# PRECISION CONFIGURATION (v3.13)
# ======================================================================
# Numerical precision for training
# - 'float64': Maximum precision (8 bytes), best for training, ~10% more memory
# - 'float32': Standard precision (4 bytes), faster, uses half memory, required for MPS
# - 'bfloat16': Brain floating point (2 bytes), fastest on Ampere+ GPUs, ~50% memory savings
#
# NOTE: MPS (Apple Silicon) requires float32. float64/bfloat16 unsupported on MPS devices.
# NOTE: bfloat16 best on NVIDIA Ampere+ (RTX 30/40 series, A100, H100)
TRAINING_PRECISION = 'float32'  # 'float64', 'float32', or 'bfloat16'

# Derived dtypes (auto-set from TRAINING_PRECISION - don't modify directly)
import numpy as np
# NumPy doesn't support bfloat16 natively - use float32 for data loading, bfloat16 for compute
NUMPY_DTYPE = np.float64 if TRAINING_PRECISION == 'float64' else np.float32

# Lazy-loaded torch dtype - avoids importing torch in multiprocessing workers
# Workers only need NUMPY_DTYPE, never TORCH_DTYPE
# This prevents macOS torch cleanup deadlock in spawn mode
_TORCH_DTYPE = None

def get_torch_dtype():
    """Lazy load torch dtype only when actually needed (training, not in workers)."""
    global _TORCH_DTYPE
    if _TORCH_DTYPE is None:
        import torch
        if TRAINING_PRECISION == 'float64':
            _TORCH_DTYPE = torch.float64
        elif TRAINING_PRECISION == 'bfloat16':
            _TORCH_DTYPE = torch.bfloat16
        else:
            _TORCH_DTYPE = torch.float32
    return _TORCH_DTYPE

def set_precision(precision: str):
    """Set training precision at runtime. Call before model creation."""
    global TRAINING_PRECISION, NUMPY_DTYPE, _TORCH_DTYPE
    import torch

    if precision not in ('float64', 'float32', 'bfloat16'):
        raise ValueError(f"Invalid precision: {precision}. Must be 'float64', 'float32', or 'bfloat16'")

    TRAINING_PRECISION = precision
    NUMPY_DTYPE = np.float64 if precision == 'float64' else np.float32

    if precision == 'float64':
        _TORCH_DTYPE = torch.float64
    elif precision == 'bfloat16':
        _TORCH_DTYPE = torch.bfloat16
    else:
        _TORCH_DTYPE = torch.float32

# ======================================================================
# CHANNEL WINDOW CONFIGURATION
# ======================================================================
# Window sizes for multi-window channel analysis (bars to look back)
# Used for ALL timeframes - same windows for consistency
# Model learns which windows are relevant for each timeframe
# v4.0: Reduced from 21 windows to 14 (10-100 range only, removed 110-168)
CHANNEL_WINDOW_SIZES = [100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10]

# Minimum data requirement (for 3-month timeframe with 10-bar window)
# 10 bars × 3 months/bar = 30 months = 2.5 years
MIN_DATA_YEARS = 2.5

# ======================================================================
# MULTI-TIMEFRAME ARCHITECTURE CONFIGURATION (v4.0)
# ======================================================================
# 11 timeframes for the new hierarchical architecture
# Each timeframe gets its own CfC layer with native OHLC data
MODEL_TIMEFRAMES = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']

# Sequence lengths per timeframe (how many bars each layer sees)
# SINGLE SOURCE OF TRUTH - features.py imports this
# These values determine both training data shape AND live inference buffer sizes
TIMEFRAME_SEQUENCE_LENGTHS = {
    '5min': 300,    # 1 day of 5-min bars (deep intraday patterns)
    '15min': 300,   # 3.1 days of 15-min bars (multi-day intraday context)
    '30min': 300,   # 6.2 days of 30-min bars (week-long intraday patterns)
    '1h': 500,      # 20.8 days of hourly bars (3-week cycles)
    '2h': 500,      # 41.7 days of 2-hour bars (6-week patterns)
    '3h': 500,      # 62.5 days of 3-hour bars (2-month patterns)
    '4h': 500,      # 83.3 days of 4-hour bars (quarterly patterns)
    'daily': 1200,  # 3.3 years of daily bars (multi-year trends, seasonal patterns)
    'weekly': 20,   # 20 weeks (~5 months, quarterly trends)
    'monthly': 12,  # 12 months (annual cycles)
    '3month': 8,    # 8 quarters (2 years, multi-year trends)
}

# VIX data file path
VIX_DATA_FILE = DATA_DIR / "VIX_History.csv"

# ======================================================================
# CONTINUATION LABEL CONFIGURATION
# ======================================================================
CONTINUATION_MODE = 'adaptive_labels'  # 'simple', 'adaptive_labels', 'adaptive_full'
# - simple: Fixed 24-bar horizon for ALL targets (baseline)
# - adaptive_labels: Fixed 24-bar for high/low, adaptive 20-40 for continuation (default)
# - adaptive_full: Adaptive 20-40 for ALL targets including high/low (experimental)

# Adaptive mode settings (used when CONTINUATION_MODE contains 'adaptive')
ADAPTIVE_MIN_HORIZON = 20  # Minimum prediction horizon (bars) - 20 minutes at 1-min resolution
ADAPTIVE_MAX_HORIZON = 40  # Maximum prediction horizon (bars) - 40 minutes at 1-min resolution
