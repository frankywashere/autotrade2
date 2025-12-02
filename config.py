"""Configuration file for trading system."""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# API Keys
CLAUDE_API_KEY = "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"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7978931435:AAGdqdfcbK-GT8Q_BEw7dvmISkN9035FzZQ")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "7910666732")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")  # Optional: add NewsAPI key
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")  # For earnings calendar (free: https://www.alphavantage.co/support/#api-key)
FRED_API_KEY = os.getenv("FRED_API_KEY", "")  # For economic data (free: https://fred.stlouisfed.org/docs/api/api_key.html)

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
DASHBOARD_PORT = 8501  # For Streamlit

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
TSLA_EVENTS_FILE = DATA_DIR / "tsla_events_REAL.csv"  # Complete earnings/deliveries + macro events (483 events, 2015-2025)
MACRO_EVENTS_API_KEY = os.getenv("MACRO_API_KEY", "")  # For FRED/economic calendar
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

# Feature extraction minimum lookback requirements (v3.13: Updated for 21-window system)
# For 3-month timeframe with 10-bar window (smallest window, longest timeframe):
# 10 bars × 3 months/bar = 30 months = 2.5 years
# 660 trading days × 390 bars/day = 257,400 1-min bars
MIN_LOOKBACK_BARS = 257400     # 2.5 years of 1-min bars (was 25740 in v3.12)
MIN_LOOKBACK_MONTHS = 30       # 2.5 years = 30 months (was 3 in v3.12)

# Continuation analysis lookback requirements (in 1-min bars)
CONTINUATION_LOOKBACK_1H = 25740   # 3 months at 1-min bars (66 trading days × 390 bars/day)
CONTINUATION_LOOKBACK_4H = 98280   # 1 year at 1-min bars (252 trading days × 390 bars/day)

# Training dataset warmup settings
SKIP_WARMUP_PERIOD = True      # Automatically exclude insufficient-history timestamps
WARMUP_BARS = MIN_LOOKBACK_BARS  # Can be overridden if needed

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
#
# NOTE: MPS (Apple Silicon) requires float32. float64 unsupported on MPS devices.
TRAINING_PRECISION = 'float32'  # 'float64' or 'float32' (MPS requires float32)

# Derived dtypes (auto-set from TRAINING_PRECISION - don't modify directly)
import numpy as np
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
        _TORCH_DTYPE = torch.float64 if TRAINING_PRECISION == 'float64' else torch.float32
    return _TORCH_DTYPE

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
TIMEFRAME_SEQUENCE_LENGTHS = {
    '5min': 200,    # 16.6 hours of 5-min data
    '15min': 200,   # 50 hours
    '30min': 200,   # 100 hours / 4.2 days
    '1h': 200,      # 8.3 days
    '2h': 100,      # 8.3 days
    '3h': 100,      # 12.5 days
    '4h': 100,      # 16.6 days
    'daily': 60,    # 60 trading days / 3 months
    'weekly': 52,   # 1 year
    'monthly': 24,  # 2 years
    '3month': 12,   # 3 years
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
