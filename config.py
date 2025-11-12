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
TSLA_EVENTS_FILE = DATA_DIR / "tsla_events.csv"  # User-provided earnings/deliveries
MACRO_EVENTS_API_KEY = os.getenv("MACRO_API_KEY", "")  # For FRED/economic calendar
EVENT_LOOKBACK_DAYS = 7  # Days before/after event to analyze
EVENT_LOOKAHEAD_DAYS = 7

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
