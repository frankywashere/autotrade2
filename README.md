# Linear Regression Channel Trading System

A Python-based trading tool for predicting daily highs and lows using linear regression channels, RSI indicators across multiple timeframes, and AI-powered news analysis.

## Features

### 1. Linear Regression Channels
- Calculate channels with upper, center, and lower deviation lines
- Detect "ping-pong" patterns (price bounces between channel lines)
- Extend channels backward for stability assessment
- Project forward to predict next period's high and low

### 2. Multi-Timeframe RSI Analysis
- RSI calculation across 1-hour, 2-hour, 3-hour, 4-hour, daily, and weekly timeframes
- Confluence scoring for multiple timeframe confirmations
- Oversold/overbought detection
- Divergence detection

### 3. AI-Powered News Analysis
- Automated news fetching every 30 minutes
- Claude AI sentiment analysis (positive/negative/neutral)
- BS scoring: Identifies clickbait, rehashed news, and overreactions
- Historical pattern comparison for signal adjustment

### 4. GUI Dashboard
- Real-time price charts with channel overlays
- Multi-timeframe RSI display
- News panel with sentiment and BS scores
- Signal log with confidence levels

### 5. Telegram Alerts
- Automated alerts for high-confluence trades
- Detailed signal information including predicted levels
- Entry, target, and stop-loss recommendations

## Installation

1. Clone or navigate to the repository:
```bash
cd /Users/frank/Desktop/CodingProjects/autotrade2
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API keys in `config.py`:
- Claude API key is already configured
- Add your Telegram bot token and chat ID (optional)
- Add NewsAPI key (optional)

## Data Requirements

Place your 1-minute CSV data files in the `/data` directory:
- `TSLA_1min.csv`
- `SPY_1min.csv`

CSV format should have columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`

The system will automatically resample to higher timeframes (1h, 4h, daily, weekly).

## Usage

The system has 4 operating modes:

### 1. Dashboard Mode (GUI)
Launch the interactive Streamlit dashboard:
```bash
python main.py dashboard
```

Access the dashboard at `http://localhost:8501`

### 2. Signal Mode (One-time Analysis)
Generate a single signal and print to console:
```bash
# Default: TSLA on 4-hour timeframe
python main.py signal

# Custom stock and timeframe
python main.py signal --stock SPY --timeframe 1hour
```

### 3. Monitor Mode (Continuous Alerts)
Continuously monitor and send Telegram alerts:
```bash
# Check every 60 minutes (default)
python main.py monitor

# Custom interval and stock
python main.py monitor --stock TSLA --timeframe 4hour --interval 30
```

### 4. Test Mode
Test all components:
```bash
python main.py test
```

## Configuration

Edit `config.py` to customize:

### Trading Parameters
- `DEFAULT_STOCK`: Default stock symbol (TSLA)
- `RSI_PERIOD`: RSI calculation period (14)
- `RSI_OVERSOLD`: Oversold threshold (30)
- `RSI_OVERBOUGHT`: Overbought threshold (70)

### Channel Parameters
- `MIN_PING_PONGS_1H`: Minimum bounces for 1-hour stability (2)
- `MIN_PING_PONGS_4H`: Minimum bounces for 4-hour stability (1)
- `CHANNEL_LOOKBACK_HOURS`: Lookback period (168 = 1 week)
- `CHANNEL_STD_DEV`: Standard deviations for channel width (2.0)

### Signal Generation
- `MIN_CONFLUENCE_SCORE`: Minimum score for Telegram alerts (60/100)
- `BS_SCORE_THRESHOLD`: High BS threshold for ignoring news (70/100)

## How It Works

### Signal Generation Process

1. **Channel Analysis** (0-30 points)
   - Price position within channel (lower/middle/upper)
   - Channel stability based on R-squared and ping-pongs
   - More stable channels = higher confidence

2. **RSI Confluence** (0-40 points)
   - Primary timeframe RSI signal (oversold/overbought)
   - Higher timeframe confirmations
   - Triple confirmation = maximum score

3. **News Analysis** (0-30 points)
   - AI sentiment scoring (-100 to +100)
   - BS detection (0-100, higher = more BS)
   - High BS bearish news during buy setup = opportunity
   - Low BS negative news = caution

4. **Total Confidence Score** (0-100)
   - Signals above `MIN_CONFLUENCE_SCORE` trigger alerts
   - Higher scores indicate stronger confluence

### Example Signal

```
Signal: BUY
Confidence: 85.5/100

REASONING:
Price in lower zone | High channel stability (75) |
RSI oversold with 2 confirmations |
High BS bearish news (85) - buy the dip

TRADE LEVELS:
Entry: $250.00
Target: $265.00 (+6.0%)
Stop Loss: $242.00 (-3.2%)
Risk/Reward: 1.88
```

## Project Structure

```
autotrade2/
├── main.py                 # Main entry point
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/                 # Stock data (gitignored)
│   ├── TSLA_1min.csv
│   └── SPY_1min.csv
└── src/                  # Source code
    ├── data_handler.py          # Data loading and resampling
    ├── linear_regression.py     # Channel calculation
    ├── rsi_calculator.py        # RSI and confluence
    ├── news_analyzer.py         # AI news analysis
    ├── signal_generator.py      # Signal generation
    ├── telegram_bot.py          # Telegram alerts
    └── gui_dashboard.py         # Streamlit dashboard
```

## Components

### DataHandler
Loads 1-minute CSV data and resamples to multiple timeframes.

### LinearRegressionChannel
Calculates regression channels with:
- Upper/lower deviation lines
- Ping-pong bounce detection
- Stability scoring
- Forward projection for predictions

### RSICalculator
Computes RSI across timeframes with:
- Oversold/overbought detection
- Multi-timeframe confluence scoring
- Divergence detection

### NewsAnalyzer
Uses Claude AI to:
- Score sentiment (-100 to +100)
- Detect BS/clickbait (0-100)
- Compare to historical patterns
- Generate trading recommendations

### SignalGenerator
Combines all analyses to:
- Generate buy/sell/neutral signals
- Calculate confidence scores (0-100)
- Provide entry/target/stop levels
- Explain reasoning

### TelegramAlertBot
Sends formatted alerts via Telegram with:
- Signal details and confidence
- Current market conditions
- Predicted levels
- Trade recommendations

## Tips for Best Results

1. **Channel Stability**: Look for channels with:
   - High R-squared (>0.7)
   - Multiple ping-pongs (2-3+)
   - Longer lookback periods

2. **RSI Confluence**: Strongest signals have:
   - Primary timeframe oversold/overbought
   - 2+ higher timeframe confirmations
   - Divergence supporting the signal

3. **News Analysis**:
   - High BS scores (>70) = ignore bearish panic
   - Low BS negative news (<40) = exercise caution
   - Use as confirmation, not primary signal

4. **Risk Management**:
   - Only trade signals with confidence >60
   - Use provided stop-loss levels
   - Higher R/R ratios are better (aim for >2)

## Troubleshooting

### "Data file not found" error
Ensure your CSV files are in the `/data` directory with correct naming:
- `TSLA_1min.csv`
- `SPY_1min.csv`

### Telegram not working
1. Get bot token from [@BotFather](https://t.me/botfather)
2. Get your chat ID from [@userinfobot](https://t.me/userinfobot)
3. Update `config.py` with your credentials
4. Run `python main.py test` to verify connection

### News analysis slow
- Claude API calls take 1-2 seconds per article
- Consider reducing article count in `news_analyzer.py`
- Articles are cached for 30 minutes

## Future Enhancements (Stage 2+)

- Backtesting engine with historical accuracy metrics
- Multiple stock monitoring in parallel
- Database for signal history
- Machine learning for BS score calibration
- Mobile app integration
- Real-time data feeds (vs CSV)

## License

This is a personal trading tool. Use at your own risk. Not financial advice.

## Support

For issues or questions, check the code comments or modify as needed.
