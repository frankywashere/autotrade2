# Linear Regression Channel Trading System - Technical Specification

**Version:** 1.0 (Stage 1 Complete)
**Repository:** https://github.com/frankywashere/autotrade2
**Last Updated:** November 10, 2025

---

## Table of Contents
1. [System Overview](#system-overview)
2. [File Structure](#file-structure)
3. [Core Algorithms](#core-algorithms)
4. [Data Flow](#data-flow)
5. [Configuration](#configuration)
6. [API Integration](#api-integration)

---

## System Overview

An AI-powered stock trading analysis system that combines:
- **Linear regression channels** with ping-pong pattern detection
- **Multi-timeframe RSI analysis** with confluence scoring
- **Claude AI news sentiment** and BS detection
- **Automated Telegram alerts** for high-confidence signals
- **Interactive dashboard** with integrated monitoring

### Key Innovation: Intelligent Channel Selection
System evaluates ALL timeframes (1h, 2h, 3h, 4h, daily, weekly) and automatically selects the channel with:
- **70% Signal Confidence** (strength of trade setup)
- **25% RSI Confluence** (multi-timeframe alignment)
- **5% Channel Stability** (quality filter)

### Hybrid Data System
- Historical CSV data: 10+ years (2015-2025)
- Live data: Automatic merge with yfinance (last 7 days)
- Data freshness tracking: LIVE/RECENT/STALE/OUTDATED

---

## File Structure

### Root Directory (`/Users/frank/Desktop/CodingProjects/autotrade2/`)

```
autotrade2/
├── .git/                           # Git repository
├── .gitignore                      # Excludes /data folder
├── .env.example                    # Environment variable template
├── README.md                       # User documentation
├── SPEC.md                         # This technical specification
├── config.py                       # Central configuration
├── main.py                         # Main entry point
├── requirements.txt                # Python dependencies
├── run.sh                          # Quick start menu script
├── convert_data.py                 # Data conversion utility
├── data/                           # Stock data (gitignored)
│   ├── TSLAMin.txt                # Raw TSLA data
│   ├── SPYMin.txt                 # Raw SPY data
│   ├── TSLA_1min.csv              # Converted TSLA 1-min data
│   └── SPY_1min.csv               # Converted SPY 1-min data
└── src/                            # Source code modules
    ├── data_handler.py            # Data loading and resampling
    ├── live_data_fetcher.py       # Live data from yfinance
    ├── linear_regression.py       # Channel calculation
    ├── rsi_calculator.py          # RSI and confluence
    ├── news_analyzer.py           # AI news analysis
    ├── signal_generator.py        # Signal generation
    ├── telegram_bot.py            # Telegram alerts
    ├── gui_dashboard.py           # Basic dashboard (deprecated)
    └── gui_dashboard_enhanced.py  # Enhanced dashboard with monitoring
```

---

## Core Files - Detailed Breakdown

### 1. `/config.py` - Central Configuration

**Purpose:** Single source of truth for all system parameters

**Key Settings:**
```python
# API Keys
CLAUDE_API_KEY          # For news sentiment analysis
TELEGRAM_BOT_TOKEN      # For alerts (7978931435:AAGdqdfcbK...)
TELEGRAM_CHAT_ID        # Your chat ID (7910666732)

# Trading Parameters
DEFAULT_STOCK = "TSLA"
STOCKS = ["TSLA", "SPY"]

# Timeframes
TIMEFRAMES = {
    "1min": "1T", "1hour": "1h", "2hour": "2h",
    "3hour": "3h", "4hour": "4h", "daily": "1D", "weekly": "1W"
}

# RSI Parameters
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Channel Parameters
MIN_PING_PONGS_1H = 2
MIN_PING_PONGS_4H = 1
CHANNEL_LOOKBACK_HOURS = 168  # 1 week
CHANNEL_STD_DEV = 2.0

# Signal Generation
MIN_CONFLUENCE_SCORE = 60  # Alert threshold

# Live Data
USE_LIVE_DATA = True
LIVE_DATA_DAYS_BACK = 7
```

**Dependencies:** None (base configuration)

---

### 2. `/main.py` - Main Entry Point

**Purpose:** Command-line interface and entry point for all modes

**Functions:**
- `run_dashboard()` - Launch Streamlit GUI
- `generate_signal(stock, timeframe)` - Generate one-time signal
- `monitor_and_alert(stock, timeframe, interval)` - Continuous monitoring
- `test_components()` - Test all modules

**Usage:**
```bash
python main.py dashboard           # GUI mode
python main.py signal              # One-time signal
python main.py monitor             # Continuous monitoring
python main.py test                # Test all components
```

**Arguments:**
- `--stock TSLA|SPY` - Stock symbol
- `--timeframe 1hour|3hour|4hour|daily` - Force specific timeframe (default: auto)
- `--interval N` - Minutes between checks (monitor mode)

**Dependencies:**
- All src/ modules
- config.py

---

### 3. `/src/data_handler.py` - Data Loading and Resampling

**Purpose:** Load 1-minute data from CSV and resample to all timeframes

**Class:** `DataHandler`

**Key Methods:**
- `load_1min_data()` - Load CSV, merge with live data if enabled
- `resample_data(timeframe)` - Resample to 1h/2h/3h/4h/daily/weekly
- `get_data(timeframe)` - Get data for specific timeframe
- `get_all_timeframes()` - Get dict of all timeframes
- `get_latest_price()` - Current price
- `get_price_at_time(timestamp)` - Historical price lookup

**Data Pipeline:**
1. Load CSV file (`TSLA_1min.csv` or `SPY_1min.csv`)
2. Parse timestamps and OHLCV columns
3. If `use_live_data=True`:
   - Fetch last 7 days from yfinance
   - Merge with historical data
   - Track data freshness
4. Resample to all timeframes using pandas

**Attributes:**
- `data_1min` - 1-minute DataFrame
- `resampled_data` - Dict of timeframe DataFrames
- `data_freshness` - Metadata about data age

**Dependencies:**
- pandas, config
- `live_data_fetcher.py` (if live data enabled)

---

### 4. `/src/live_data_fetcher.py` - Live Data Integration

**Purpose:** Fetch real-time stock data from yfinance and merge with historical CSV

**Class:** `LiveDataFetcher`

**Key Methods:**
- `fetch_recent_data(days_back=7)` - Fetch 1-min data from yfinance
- `get_current_price()` - Get real-time price
- `merge_with_historical(historical_df)` - Merge CSV + live data
- `get_data_freshness(timestamp)` - Check data age

**Freshness Levels:**
- **LIVE** - Updated within 5 minutes
- **RECENT** - Within 1 hour
- **STALE** - Within 1 day
- **OUTDATED** - Older than 1 day

**Limitations:**
- yfinance provides 1-min data for last 7 days only
- Automatically removes timezone info to match CSV format

**Dependencies:**
- yfinance, pandas, config

---

### 5. `/src/linear_regression.py` - Channel Calculation

**Purpose:** Calculate linear regression channels with ping-pong detection and 24-hour predictions

**Class:** `LinearRegressionChannel`

**Data Class:** `ChannelData`
- `slope` - Regression line slope
- `intercept` - Regression line intercept
- `upper_line` - Upper channel boundary (center + 2σ)
- `lower_line` - Lower channel boundary (center - 2σ)
- `center_line` - Regression line
- `std_dev` - Standard deviation of residuals
- `r_squared` - Goodness of fit (0-1)
- `ping_pongs` - Number of bounces between lines
- `stability_score` - Overall channel quality (0-100)
- `predicted_high` - 24-hour expected high
- `predicted_low` - 24-hour expected low
- `predicted_center` - 24-hour center projection

**Key Methods:**

#### `calculate_channel(df, lookback_bars, timeframe)`
1. Extract close prices from DataFrame
2. Calculate linear regression (scipy.stats.linregress)
3. Compute residuals and standard deviation
4. Create upper/lower lines (center ± 2σ)
5. Detect ping-pongs (price bounces)
6. Calculate stability score
7. **Project 24 hours forward:**
   - Calculate bars for 24h (e.g., 3hour → 8 bars)
   - Project regression line forward
   - Find MAX(upper channel) = predicted_high
   - Find MIN(lower channel) = predicted_low

#### `_detect_ping_pongs(prices, upper, lower, threshold=0.02)`
Counts bounces between channel lines:
- Tracks when price touches upper (within 2%)
- Tracks when price touches lower (within 2%)
- Counts transitions: upper→lower or lower→upper
- Returns total bounce count

#### `_calculate_stability(r_squared, ping_pongs, n_bars)`
Composite stability score (0-100):
- **40 points:** R-squared × 40
- **40 points:** min(ping_pongs / 5, 1.0) × 40
- **20 points:** min(n_bars / 100, 1.0) × 20

#### `get_channel_position(price, channel)`
Returns current price position:
- `position` - 0.0 (lower) to 1.0 (upper)
- `zone` - lower_extreme, lower, middle, upper, upper_extreme
- `distance_to_upper_pct` - % to upper line
- `distance_to_lower_pct` - % to lower line

#### `analyze_multiple_timeframes(data_dict)`
Calculates channels for all timeframes with smart lookback:
- 1h/2h/3h: 168 hours (1 week)
- 4h: 42 bars (1 week)
- daily/weekly: All data

**Dependencies:**
- scipy.stats, numpy, pandas, config

---

### 6. `/src/rsi_calculator.py` - RSI and Confluence Analysis

**Purpose:** Calculate RSI across multiple timeframes and detect confluence

**Class:** `RSICalculator`

**Data Class:** `RSIData`
- `value` - RSI value (0-100)
- `oversold` - True if RSI < 30
- `overbought` - True if RSI > 70
- `signal` - 'buy', 'sell', or 'neutral'
- `history` - Full RSI Series

**Key Methods:**

#### `calculate_rsi(df, column='close')`
Standard RSI calculation:
1. Calculate price changes (delta)
2. Separate gains and losses
3. Calculate average gain/loss (14-period rolling)
4. RS = avg_gain / avg_loss
5. RSI = 100 - (100 / (1 + RS))

#### `get_rsi_data(df)`
Returns RSIData object with signal:
- RSI < 30 → 'buy' signal
- RSI > 70 → 'sell' signal
- Otherwise → 'neutral'

#### `analyze_multiple_timeframes(data_dict)`
Calculates RSI for all timeframes in parallel

#### `get_confluence_score(rsi_dict, primary_timeframe)`
**Confluence Scoring Algorithm:**
1. Check primary timeframe RSI signal
2. Look at higher timeframes for confirmation:
   - 1hour → check 2h/3h/4h/daily/weekly
   - 4hour → check daily/weekly
   - daily → check weekly
3. Count confirming timeframes
4. Calculate score:
   - Base: 40 points (primary signal)
   - Confirmation: (confirmations / total_checked) × 60

Returns:
```python
{
    "score": 0-100,
    "signal": "buy|sell|neutral",
    "primary_rsi": float,
    "confirming_timeframes": [list],
    "timeframes": {dict of all RSI values}
}
```

#### `detect_divergence(df, lookback=14)`
Detects RSI divergence:
- **Bullish:** Price lower lows, RSI higher lows
- **Bearish:** Price higher highs, RSI lower highs

**Dependencies:**
- pandas, numpy, config

---

### 7. `/src/news_analyzer.py` - AI News Analysis

**Purpose:** Fetch news and analyze sentiment + BS score using Claude AI

**Class:** `NewsAnalyzer`

**Data Class:** `NewsArticle`
- `title` - Article headline
- `description` - Article description
- `url` - Article URL
- `published_at` - Publication timestamp
- `source` - News source
- `sentiment` - positive/negative/neutral
- `sentiment_score` - -100 (bearish) to +100 (bullish)
- `bs_score` - 0 (factual) to 100 (BS/clickbait)
- `analysis` - AI explanation

**Key Methods:**

#### `fetch_news(hours_back=24)`
- Attempts to fetch from NewsAPI (if key configured)
- Falls back to mock articles for testing
- Returns list of NewsArticle objects

#### `analyze_article(article, stock_context)`
**Claude AI Analysis:**
1. Builds prompt with article + market context
2. Calls Claude API (model: claude-3-haiku-20240307)
3. Requests JSON response with:
   - Sentiment: positive/negative/neutral
   - Sentiment Score: -100 to +100
   - BS Score: 0-100
   - Analysis: 1-2 sentence explanation
4. Parses JSON response
5. Updates article object

**BS Scoring Criteria:**
- Rehashed old news → High BS
- Clickbait headline → High BS
- Contradicts historical patterns → High BS
- If bearish news historically led to rebounds → High BS
- Panic/FOMO content without substance → High BS

#### `analyze_all_news(stock_context)`
Analyzes all cached articles with AI

#### `get_overall_sentiment()`
Aggregates news sentiment:
```python
{
    "avg_sentiment_score": float,
    "avg_bs_score": float,
    "positive_count": int,
    "negative_count": int,
    "neutral_count": int,
    "high_bs_count": int,
    "signal": "positive|negative|neutral|ignore",
    "recommendation": str
}
```

**Signal Logic:**
- If avg_bs > 70 → "ignore" (high BS, disregard news)
- If avg_sentiment > 20 → "positive"
- If avg_sentiment < -20 → "negative"
- Otherwise → "neutral"

**Dependencies:**
- anthropic, requests, config

---

### 8. `/src/signal_generator.py` - Signal Generation Engine

**Purpose:** Combine channel + RSI + news analysis to generate trading signals

**Class:** `SignalGenerator`

**Data Class:** `TradingSignal`
```python
timestamp: datetime
stock: str
signal_type: str                # 'buy', 'sell', 'neutral'
confidence_score: float         # 0-100
current_price: float
channel_position: Dict          # Zone, position, distances
predicted_high: float           # 24-hour expected high
predicted_low: float            # 24-hour expected low
channel_stability: float        # 0-100
best_channel_timeframe: str     # Selected timeframe (e.g., '3hour')
best_channel_data: ChannelData  # Complete channel object
rsi_confluence: Dict            # RSI confluence data
primary_rsi: float              # RSI value
news_sentiment: Dict            # News analysis results
entry_price: float              # Recommended entry
target_price: float             # Take profit target
stop_loss: float                # Stop loss level
reasoning: str                  # Human-readable explanation
```

**Key Methods:**

#### `_evaluate_timeframe(timeframe, channel, current_price, rsi_dict, news_sentiment)`
Evaluates a single timeframe and returns:
```python
{
    'timeframe': str,
    'channel': ChannelData,
    'channel_position': Dict,
    'rsi_confluence': Dict,
    'signal_type': str,
    'confidence': float,
    'reasoning': str,
    'composite_score': float  # 70% Conf + 25% RSI + 5% Stab
}
```

#### `generate_signal(primary_timeframe=None)`
**Main Signal Generation Algorithm:**

**Step 1: Load Data**
- Get all timeframes from DataHandler
- Get current price

**Step 2: Analyze All Timeframes**
- Calculate channels for 1h, 2h, 3h, 4h, daily, weekly
- Calculate RSI for all timeframes
- Fetch and analyze news (once, same for all)

**Step 3: Evaluate Each Timeframe**
For each timeframe:
- Get channel position
- Get RSI confluence
- Calculate signal confidence
- Compute composite score:
  - **70%** Signal Confidence
  - **25%** RSI Confluence
  - **5%** Channel Stability

**Step 4: Select Best Timeframe**
- Pick timeframe with highest composite score
- Print all evaluations for transparency
- Extract best channel, RSI, and signal data

**Step 5: Calculate Trade Levels**
- Entry: Current price
- Target: Upper channel (buy) or lower channel (sell)
- Stop: Below lower (buy) or above upper (sell) with 2% buffer

**Step 6: Build Signal Object**
- Create TradingSignal with all data
- Cache in signal_history

#### `_calculate_signal(channel_pos, channel, rsi_conf, news)`
**Signal Confidence Scoring (0-100):**

**Channel Component (0-30 points):**
- In lower/lower_extreme zone → +20 (buy signal)
- In upper/upper_extreme zone → +20 (sell signal)
- Channel stability > 60 → +10

**RSI Component (0-40 points):**
- RSI signal matches channel → +(rsi_score × 0.4) up to 40
- RSI conflicts with channel → -15

**News Component (0-30 points):**
- High BS + bearish + buy signal → +15 (buy the dip!)
- Low BS + matches signal → +15
- Low BS + contradicts → -20
- High BS → +5 (ignore news)

**Final Adjustments:**
- Clamp to 0-100
- If confidence < MIN_CONFLUENCE_SCORE → signal = 'neutral'

#### `_calculate_levels(signal_type, current_price, channel, channel_pos)`
Buy signal:
- Entry: Current price
- Target: min(upper_value, predicted_high)
- Stop: lower_value × 0.98

Sell signal:
- Entry: Current price
- Target: max(lower_value, predicted_low)
- Stop: upper_value × 1.02

#### `get_signal_summary(signal)`
Formats signal as human-readable text with:
- Signal type and confidence
- Current market conditions
- 24-hour prediction with timeframe
- RSI confluence details
- News sentiment
- Trade levels if not neutral
- Full reasoning

**Dependencies:**
- All src/ modules, config

---

### 9. `/src/telegram_bot.py` - Telegram Alert System

**Purpose:** Send formatted trading alerts via Telegram

**Class:** `TelegramAlertBot`

**Key Methods:**

#### `send_signal_alert(signal)`
Async method that:
1. Formats signal as HTML message
2. Sends via Telegram API
3. Falls back to console if not configured

#### `_format_signal_message(signal)`
Creates HTML-formatted message:
```
🟢 TRADING ALERT: TSLA

Signal: BUY
Confidence: 85.5/100

📊 CURRENT MARKET
• Price: $445.26
• Channel: lower (18%)
• RSI: 47.2

🎯 24-HOUR FORECAST (3hour channel)
• Expected High: $503.06 (+13.0%)
• Expected Low: $434.89 (-2.3%)

💰 TRADE LEVELS
• Entry: $445.26
• Target: $503.06 (+13.0%)
• Stop: $426.40 (-4.2%)
• R/R: 3.07

📈 RSI CONFLUENCE
• Score: 80/100
• Confirmations: 2

📰 NEWS SENTIMENT
• Sentiment: -15
• BS Score: 85/100
• Signal: ignore

💡 REASONING
Price in lower zone | RSI oversold with 2 confirmations |
High BS bearish news - buy the dip
```

#### `test_connection()`
Verifies Telegram bot credentials and sends test message

**Dependencies:**
- python-telegram-bot, asyncio, config

---

### 10. `/src/gui_dashboard_enhanced.py` - Interactive Dashboard

**Purpose:** Streamlit web GUI with integrated monitoring and Telegram alerts

**Key Features:**

1. **Auto-Monitoring Controls (Sidebar)**
   - Start/Stop buttons
   - Configurable interval (15-120 minutes)
   - Live monitoring logs
   - Status indicator (active/inactive)

2. **Background Monitoring Thread**
   - Runs in daemon thread while dashboard is open
   - Generates signals at intervals
   - Sends Telegram alerts automatically
   - Updates session state with latest signal

3. **Main Chart (Left Column)**
   - Candlestick chart with OHLCV
   - Linear regression channel overlays
   - RSI subplot
   - Zoomed to relevant timeframe:
     - 1hour: Last 168 bars (7 days)
     - 3hour: Last 168 bars (21 days)
     - 4hour: Last 126 bars (21 days)
     - daily: Last 90 bars (3 months)
     - weekly: Last 52 bars (1 year)

4. **Signal Panel (Right Column)**
   - BUY/SELL/NEUTRAL indicator
   - Confidence score
   - Channel metadata (stability, ping-pongs, R²)
   - 24-hour forecast (high/low)
   - RSI analysis
   - Trade levels (entry/target/stop)
   - Reasoning

5. **Multi-Timeframe RSI Grid**
   - Shows RSI for all timeframes
   - Color-coded: green (oversold), red (overbought), blue (neutral)

6. **News Panel**
   - Overall sentiment metrics
   - Individual article cards with:
     - Sentiment score
     - BS score (color-coded)
     - AI analysis
     - Links to sources

7. **Data Freshness Indicator (Sidebar)**
   - 🟢 LIVE, 🟡 RECENT, 🟠 STALE, or 🔴 OUTDATED
   - Shows data age

**Session State:**
```python
monitoring: bool              # Monitoring active
monitor_thread: Thread        # Background thread
last_signal: TradingSignal    # Latest signal from monitor
monitor_logs: List[str]       # Last 10 log entries
```

**Caching:**
- `@st.cache_data(ttl=300)` - Data cached 5 minutes
- `@st.cache_resource` - Components cached

**Dependencies:**
- streamlit, plotly, all src/ modules, config

---

### 11. `/requirements.txt` - Python Dependencies

**Core Libraries:**
```
pandas>=2.0.0              # Data manipulation
numpy>=2.0.0,<2.3.0       # Numerical computing
scipy>=1.11.0              # Linear regression (stats)
scikit-learn>=1.3.0        # Future ML features
matplotlib>=3.7.0          # Plotting
plotly>=5.14.0             # Interactive charts
```

**Data & Indicators:**
```
yfinance>=0.2.0            # Live stock data
pandas-ta>=0.3.14b         # Technical indicators
```

**API Integration:**
```
anthropic>=0.18.0          # Claude AI
python-telegram-bot>=20.0  # Telegram alerts
requests>=2.31.0           # HTTP requests
```

**Dashboard:**
```
streamlit>=1.28.0          # Web GUI framework
```

**Utilities:**
```
python-dotenv>=1.0.0       # Environment variables
```

---

### 12. `/convert_data.py` - Data Conversion Utility

**Purpose:** One-time conversion of TSLAMin.txt and SPYMin.txt to CSV format

**Input Format:**
```
20150102 114000;223.29;223.29;223.29;223.29;175
```

**Output Format:**
```csv
timestamp,open,high,low,close,volume
2015-01-02 11:40:00,223.29,223.29,223.29,223.29,175
```

**Usage:**
```bash
python3 convert_data.py
```

Converts and saves:
- `data/TSLAMin.txt` → `data/TSLA_1min.csv`
- `data/SPYMin.txt` → `data/SPY_1min.csv`

---

### 13. `/run.sh` - Quick Start Menu

**Purpose:** Interactive menu for launching system

**Menu Options:**
1. Dashboard - Launch GUI with integrated monitoring
2. Signal - Generate current signal (one-time)
3. Monitor - Console monitoring only (no GUI)
4. Test - Test all components
5. Exit

**Usage:**
```bash
./run.sh
```

---

## Core Algorithms

### Algorithm 1: Intelligent Channel Selection

**Goal:** Select the timeframe with highest confidence in 24-hour predictions

**Process:**
```
For each timeframe (1h, 2h, 3h, 4h, daily, weekly):
    1. Calculate linear regression channel
    2. Get current price position in channel
    3. Calculate RSI confluence for that timeframe
    4. Generate signal and confidence score
    5. Calculate composite score:
       = Confidence × 0.70
       + RSI_Confluence × 0.25
       + Stability × 0.05

Select timeframe with MAX(composite_score)
```

**Weighting Rationale:**
- **70% Confidence** - Prioritizes actual trading opportunities
- **25% RSI** - Ensures multi-timeframe confirmation
- **5% Stability** - Basic quality filter only

**Example:**
```
2hour: Conf=30, RSI=0, Stab=60  → Composite = 21.0 + 0.0 + 3.0 = 24.0
3hour: Conf=30, RSI=0, Stab=79  → Composite = 21.0 + 0.0 + 4.0 = 25.0 ⭐
weekly: Conf=0, RSI=70, Stab=37 → Composite = 0.0 + 17.5 + 1.9 = 19.4

Winner: 3hour (highest composite)
```

---

### Algorithm 2: 24-Hour Prediction

**Goal:** Predict expected high and low for next 24 hours using selected channel

**Process:**
```
1. Select best channel (e.g., 3hour)

2. Calculate bars needed for 24 hours:
   bars_per_24h = {
       '1hour': 24,
       '2hour': 12,
       '3hour': 8,
       '4hour': 6,
       'daily': 1,
       'weekly': 1
   }

3. Project regression line forward N bars:
   future_x = [n, n+1, n+2, ..., n+N]
   future_center = slope × future_x + intercept
   future_upper = future_center + (2 × std_dev)
   future_lower = future_center - (2 × std_dev)

4. Find 24-hour range:
   predicted_high = MAX(future_upper)  # Highest point
   predicted_low = MIN(future_lower)   # Lowest point
```

**Example (3hour channel):**
```
Current time: 15:00
Project 8 bars forward (8 × 3h = 24h):
  Bar 1 (18:00): upper=$495, lower=$430
  Bar 2 (21:00): upper=$497, lower=$431
  Bar 3 (00:00): upper=$499, lower=$432
  Bar 4 (03:00): upper=$501, lower=$433
  Bar 5 (06:00): upper=$502, lower=$434
  Bar 6 (09:00): upper=$503, lower=$435
  Bar 7 (12:00): upper=$503, lower=$436
  Bar 8 (15:00): upper=$502, lower=$437

24h High: MAX = $503
24h Low: MIN = $430
```

---

### Algorithm 3: Signal Confidence Scoring

**Goal:** Calculate 0-100 confidence score for each timeframe

**Components:**

**1. Channel Analysis (0-30 points):**
```
If price in lower/lower_extreme zone:
    signal = 'buy'
    confidence += 20
    if stability > 60:
        confidence += 10

If price in upper/upper_extreme zone:
    signal = 'sell'
    confidence += 20
    if stability > 60:
        confidence += 10
```

**2. RSI Analysis (0-40 points):**
```
If RSI matches channel signal:
    confidence += min(rsi_confluence_score × 0.4, 40)
    # More higher timeframe confirmations = more points

If RSI conflicts:
    confidence -= 15
```

**3. News Analysis (0-30 points):**
```
If high BS (>70) + bearish + buy signal:
    confidence += 15  # "Buy the dip" opportunity

If low BS + matches signal:
    confidence += 15

If low BS + contradicts:
    confidence -= 20  # Genuine negative news

If high BS:
    confidence += 5  # Ignore noise
```

**4. Final Adjustments:**
```
confidence = clamp(confidence, 0, 100)

If confidence < MIN_CONFLUENCE_SCORE (60):
    signal = 'neutral'
```

---

### Algorithm 4: RSI Confluence Scoring

**Goal:** Measure multi-timeframe RSI alignment

**Process:**
```
1. Get primary timeframe RSI (e.g., 3hour)
2. Determine signal: oversold (<30), overbought (>70), neutral

3. Check higher timeframes:
   If primary = 3hour:
       check: daily, weekly
   If primary = 1hour:
       check: 2h, 3h, 4h, daily, weekly

4. Count confirmations:
   For buy signal: count higher TFs with RSI < 50
   For sell signal: count higher TFs with RSI > 50

5. Calculate score:
   Base: 40 points (primary signal exists)
   Confirmation: (confirmations / total_checked) × 60

Example:
   3hour RSI = 28 (oversold) → buy signal
   daily RSI = 35 → confirms (< 50)
   weekly RSI = 42 → confirms (< 50)

   Score = 40 + (2/2) × 60 = 100/100 (perfect confluence!)
```

---

## Data Flow

### Startup → Signal Generation

```
1. main.py
   ↓
2. SignalGenerator.__init__()
   - Creates DataHandler
   - Creates LinearRegressionChannel
   - Creates RSICalculator
   - Creates NewsAnalyzer
   ↓
3. generate_signal()
   ↓
4. DataHandler.get_all_timeframes()
   - Load CSV (1.85M rows historical)
   - Fetch yfinance (2.7K rows live, last 7 days)
   - Merge: 1.86M total rows
   - Resample to 1h/2h/3h/4h/daily/weekly
   ↓
5. FOR EACH TIMEFRAME:
   ↓
   5a. LinearRegressionChannel.calculate_channel()
       - Calculate regression on lookback period
       - Detect ping-pongs
       - Calculate stability score
       - Project 24 hours forward
       - Find predicted high/low
   ↓
   5b. RSICalculator.get_confluence_score()
       - Calculate RSI for this timeframe
       - Check higher timeframes for confirmation
       - Return confluence score (0-100)
   ↓
   5c. Calculate signal confidence (0-100)
       - Channel position score
       - RSI score
       - News score (fetched once)
   ↓
   5d. Calculate composite score
       = Confidence × 0.70
       + RSI × 0.25
       + Stability × 0.05
   ↓
6. SELECT BEST TIMEFRAME
   - Pick max(composite_score)
   ↓
7. BUILD TRADING SIGNAL
   - Use best channel's predictions
   - Calculate entry/target/stop
   - Generate reasoning
   ↓
8. OUTPUT
   - Print summary OR
   - Send Telegram alert OR
   - Display in dashboard
```

---

### Dashboard Monitoring Loop

```
1. User clicks "Start Monitor" in dashboard
   ↓
2. Background thread starts:
   ↓
   while monitoring:
       ↓
       3. SignalGenerator.generate_signal()
          - Evaluates all timeframes
          - Selects best by composite score
          - Returns TradingSignal
       ↓
       4. Check if alert-worthy:
          if confidence >= 60 AND
             signal != 'neutral' AND
             signal changed from last:
          ↓
          5. TelegramAlertBot.send_signal_alert()
             - Sends formatted message
             - Logs to monitor_logs
       ↓
       6. Update session_state.last_signal
       ↓
       7. Sleep for interval_minutes × 60
   ↓
8. Dashboard refreshes show latest signal
```

---

## Configuration

### API Keys

**Claude AI (Required):**
```python
CLAUDE_API_KEY = "sk-ant-api03-ljR7i4Eh5Aaiqsn6jsdOAMqt..."
```
Used for news sentiment and BS scoring (model: claude-3-haiku-20240307)

**Telegram (Optional):**
```python
TELEGRAM_BOT_TOKEN = "7978931435:AAGdqdfcbK-GT8Q_BEw7dvmISkN9035FzZQ"
TELEGRAM_CHAT_ID = "7910666732"
```
Get from @BotFather and @userinfobot

**NewsAPI (Optional):**
```python
NEWS_API_KEY = ""  # Sign up at newsapi.org
```
Falls back to mock articles if not configured

---

### Trading Parameters

**RSI Thresholds:**
```python
RSI_PERIOD = 14         # Standard RSI period
RSI_OVERSOLD = 30       # Buy signal threshold
RSI_OVERBOUGHT = 70     # Sell signal threshold
```

**Channel Parameters:**
```python
CHANNEL_LOOKBACK_HOURS = 168  # 1 week of data
CHANNEL_STD_DEV = 2.0         # 2 standard deviations
MIN_PING_PONGS_1H = 2         # Minimum bounces for 1h
MIN_PING_PONGS_4H = 1         # Minimum bounces for 4h
```

**Signal Generation:**
```python
MIN_CONFLUENCE_SCORE = 60     # Alert threshold (0-100)
BS_SCORE_THRESHOLD = 70       # High BS = ignore news
```

**Live Data:**
```python
USE_LIVE_DATA = True          # Enable live data merge
LIVE_DATA_DAYS_BACK = 7       # Days to fetch (max 7)
```

---

## API Integration

### 1. yfinance (Stock Data)

**Endpoint:** Yahoo Finance API (via yfinance library)

**Rate Limits:** None known, but fetching 1-min data limited to last 7 days

**Usage:**
```python
ticker = yf.Ticker('TSLA')
df = ticker.history(period='7d', interval='1m')
```

**Data Returned:**
- Open, High, Low, Close, Volume
- 1-minute intervals
- Last 7 days maximum

---

### 2. Claude AI API (News Analysis)

**Endpoint:** Anthropic Messages API

**Model:** claude-3-haiku-20240307 (fast, cost-effective)

**Rate Limits:** Per API key

**Request Format:**
```python
client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=500,
    messages=[{"role": "user", "content": prompt}]
)
```

**Prompt Structure:**
```
Analyze this news article about TSLA:

Title: [title]
Description: [description]
Current Market Context: [context]

Provide:
1. Sentiment: positive/negative/neutral
2. Sentiment Score: -100 to +100
3. BS Score: 0-100
4. Brief analysis

Format as JSON: {...}
```

**Cost:** ~$0.00025 per article (Haiku pricing)

---

### 3. Telegram Bot API

**Endpoint:** Telegram Bot API

**Method:** `sendMessage`

**Authentication:** Bot token in config

**Message Format:** HTML

**Features Used:**
- Bold tags: `<b>text</b>`
- Line breaks: newlines
- Emojis: Unicode

**Async:** Uses python-telegram-bot library with asyncio

---

### 4. NewsAPI (Optional)

**Endpoint:** newsapi.org/v2/everything

**Parameters:**
- `q`: Query (stock symbol + company name)
- `from`: Date to fetch from
- `sortBy`: publishedAt
- `language`: en
- `apiKey`: API key

**Returns:** Array of articles with title, description, URL, source, date

**Fallback:** Mock articles if API key not configured

---

## System Behavior

### Signal Generation Flow

**Input:** Stock symbol (TSLA/SPY)

**Process:**
1. Load historical CSV (1.85M rows)
2. Merge live data from yfinance (+2.7K rows)
3. Resample to 6 timeframes
4. Calculate 6 regression channels with 24h predictions
5. Calculate 6 RSI values with confluence
6. Fetch and analyze news (once)
7. Evaluate each timeframe with composite scoring
8. Select best timeframe
9. Build TradingSignal object

**Output:** TradingSignal with:
- Signal type (buy/sell/neutral)
- Confidence (0-100)
- Best channel timeframe
- 24-hour high/low forecast
- Entry/target/stop levels
- Full reasoning

**Performance:**
- Total time: ~5-10 seconds
- Data loading: ~2 seconds
- Channel calculations: ~1 second
- News analysis: ~2-5 seconds (Claude API)
- RSI calculations: <1 second

---

### Alert Triggering Logic

**Conditions for Telegram Alert:**
```python
if (signal.confidence_score >= 60 AND
    signal.signal_type != 'neutral' AND
    signal.signal_type != last_signal_type):

    send_telegram_alert(signal)
    last_signal_type = signal.signal_type
```

**Why These Conditions:**
1. **Confidence ≥ 60** - Only high-quality setups
2. **Not neutral** - Must be buy or sell signal
3. **Signal changed** - Prevents spam (no repeat alerts)

**Alert Frequency:**
- Monitor mode: Every N minutes (configurable)
- Dashboard mode: When monitoring enabled (user controls)
- Only sends on signal changes

---

### Dashboard Auto-Refresh

**Cache Strategy:**
```python
@st.cache_data(ttl=300)  # 5 minute cache
def load_data(stock):
    # Data loading cached for 5 minutes
```

**Refresh Triggers:**
1. Stock symbol changes
2. Manual refresh button clicked
3. 5-minute cache expires
4. Monitoring generates new signal

**Performance Optimization:**
- Components cached with `@st.cache_resource`
- Data cached with `@st.cache_data`
- Monitoring shares signal with dashboard (session_state)

---

## Current Limitations

1. **yfinance Data:** Only 7 days of 1-minute data available
2. **News Sources:** Limited to NewsAPI or mock articles
3. **Single Stock:** Monitor/dashboard shows one stock at a time
4. **No Backtesting:** Signal history not validated against historical performance
5. **No Database:** Signals not persisted
6. **No Paper Trading:** No simulation or execution
7. **Claude API Costs:** $0.00025 per article analyzed

---

## Future Enhancements (Stage 2+)

### Planned Features:
- **Backtesting Engine:** Validate signal accuracy on historical data
- **Signal History Database:** SQLite storage for performance tracking
- **Multi-Stock Portfolio:** Monitor multiple stocks simultaneously
- **Advanced Indicators:** MACD, Bollinger Bands, Volume Profile
- **Machine Learning:** Improve BS score calibration with historical accuracy
- **Paper Trading:** Simulated trades with P&L tracking
- **Performance Analytics:** Win rate, average R/R, drawdown metrics
- **Real-time Feeds:** WebSocket integration for true tick-by-tick data
- **Mobile App:** React Native or Flutter companion app

---

## Testing

### Test Suite (`python main.py test`)

**Tests Run:**
1. ✓ Data Handler - Load and resample data
2. ✓ Linear Regression - Channel calculation
3. ✓ RSI Calculator - Multi-timeframe RSI
4. ✓ News Analyzer - Fetch and analyze with Claude
5. ✓ Signal Generator - End-to-end signal generation
6. ✓ Telegram Bot - Connection and message sending

**Expected Output:**
```
======================================================================
TESTING ALL COMPONENTS
======================================================================

1. Testing Data Handler...
   ✓ Data handler working: 13032 bars of 4-hour data

2. Testing Linear Regression Channel...
   ✓ Channel calculator working
     - Stability: 79.1/100
     - Ping-pongs: 3
     - Predicted high: $503.06

3. Testing RSI Calculator...
   ✓ RSI calculator working: RSI = 47.3

4. Testing News Analyzer...
   ✓ News analyzer working: 2 articles fetched
     - Sentiment: neutral
     - BS Score: 50.0/100

5. Testing Signal Generator...
   ✓ Signal generator working
     - Signal: NEUTRAL
     - Confidence: 30.0/100

6. Testing Telegram Bot...
   ✓ Telegram bot connected: @judochopbot

======================================================================
TESTING COMPLETE
======================================================================
```

---

## Performance Metrics

### Data Volume (as of Nov 10, 2025):
- **TSLA Historical:** 1,854,183 rows (Jan 2015 - Sep 2025)
- **SPY Historical:** 2,144,644 rows (Jan 2015 - Sep 2025)
- **Live Data:** ~2,700 rows per stock (last 7 days)
- **Total Combined:** ~1.86M rows per stock

### Processing Speed:
- Data loading: 2-3 seconds
- Resampling to 6 timeframes: <1 second
- Channel calculations (6 timeframes): ~1 second
- RSI calculations (6 timeframes): <1 second
- News fetch + analysis: 2-5 seconds
- **Total signal generation:** 5-10 seconds

### Memory Usage:
- Full dataset in memory: ~500MB per stock
- Dashboard with cache: ~800MB
- Monitoring thread: +50MB

---

## Security Considerations

### API Keys:
- ⚠️ Claude API key stored in config.py (should use .env in production)
- ⚠️ Telegram credentials in config.py (should use .env in production)
- ✓ Data folder gitignored (protects proprietary data)
- ✓ Repository is private

### Recommendations:
1. Move all API keys to .env file
2. Use .env.example as template
3. Add config.py to .gitignore if keys are in file
4. Rotate keys if repository ever becomes public

---

## Deployment

### Local Development:
```bash
cd /Users/frank/Desktop/CodingProjects/autotrade2
pip install -r requirements.txt
python main.py test
python main.py dashboard
```

### Running 24/7:
```bash
# Terminal 1: Dashboard
python main.py dashboard

# Or use screen/tmux for background:
screen -S trading
python main.py monitor --interval 30
# Ctrl+A, D to detach
```

### System Requirements:
- Python 3.11+
- 2GB RAM minimum
- Internet connection for live data
- macOS/Linux/Windows

---

## Troubleshooting

### Common Issues:

**1. "Data file not found"**
- Run `python3 convert_data.py` first
- Check data/ folder has TSLA_1min.csv and SPY_1min.csv

**2. "ModuleNotFoundError: No module named 'config'"**
- Run from project root, not src/ directory
- Imports fixed with sys.path.insert(0, parent_dir)

**3. "Telegram connection failed"**
- Verify bot token and chat ID in config.py
- Test with: `python main.py test`

**4. "Claude API error 404"**
- Model updated to claude-3-haiku-20240307
- Check API key is valid

**5. "numpy version conflicts"**
- Install compatible version: pip install 'numpy>=2.0.0,<2.3.0'

**6. Chart showing 10 years**
- Fixed: Chart now zoomed to relevant window (168 bars for 3hour)

---

## Version History

### v1.0 - Stage 1 Complete (Nov 10, 2025)

**Git Commits:**
```
78f3d33 - Prioritize confidence and RSI confluence over stability
0557be9 - Fix predictions to always forecast next 24 hours
6873cd8 - Intelligent channel selection with composite scoring + chart zoom
f972f1b - Auto-select best channel by stability score
10d8152 - Add hybrid live + historical data system (Option C)
87fe02f - Add enhanced dashboard with integrated monitoring
1b508dd - Add quick start script and finalize setup
7f3f93d - Fix all errors and complete system setup
cf709ec - Update numpy and scipy version requirements
2d90353 - Add Telegram bot credentials to config
4fde7a7 - Initial commit: Linear Regression Channel Trading System
```

**Features Implemented:**
✅ Linear regression channels with ping-pong detection
✅ Multi-timeframe RSI (1h/2h/3h/4h/daily/weekly)
✅ AI news sentiment + BS scoring
✅ Telegram alerts
✅ Streamlit dashboard with monitoring
✅ Live data integration (yfinance)
✅ Intelligent channel selection (70% confidence weighted)
✅ 24-hour forecasting
✅ Chart zoom optimization

**Lines of Code:** 2,608

**Files:** 13 tracked files

**Test Status:** All components passing ✓

---

## Contact & Support

**Repository:** https://github.com/frankywashere/autotrade2 (Private)

**Author:** Built with Claude Code

**License:** Personal trading tool - Use at your own risk. Not financial advice.

---

## Appendix: Key Formulas

### Linear Regression:
```
center_line = slope × x + intercept
upper_line = center_line + (2 × σ)
lower_line = center_line - (2 × σ)

Where:
  σ = std_dev(residuals)
  residuals = actual_prices - center_line
```

### RSI:
```
RS = avg_gain(14) / avg_loss(14)
RSI = 100 - (100 / (1 + RS))
```

### Stability Score:
```
stability = (R² × 40) + (min(ping_pongs/5, 1) × 40) + (min(bars/100, 1) × 20)
```

### Composite Score:
```
composite = (confidence × 0.70) + (rsi_confluence × 0.25) + (stability × 0.05)
```

### 24-Hour Projection:
```
bars_24h = {'1hour': 24, '2hour': 12, '3hour': 8, '4hour': 6, 'daily': 1}
future_x = [n, n+1, n+2, ..., n+bars_24h]
predicted_high = MAX(slope × future_x + intercept + 2σ)
predicted_low = MIN(slope × future_x + intercept - 2σ)
```

---

**End of Specification**
