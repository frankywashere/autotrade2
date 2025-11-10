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
