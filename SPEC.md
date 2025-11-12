# AutoTrade2 - Complete Technical Specification

**Version:** 3.2 (Multi-Scale LNN + Meta-LNN Coach + Dual Prediction Modes + Fixed Timeframe Loading)
**Repository:** https://github.com/frankywashere/autotrade2
**Last Updated:** November 12, 2025

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Quick Start](#quick-start)
3. [File Structure](#file-structure)
4. [Stage 1: Linear Regression Trading System](#stage-1-linear-regression-trading-system)
5. [Stage 2: Multi-Scale LNN System](#stage-2-multi-scale-lnn-system)
6. [Multi-Scale Architecture](#multi-scale-architecture)
7. [Meta-LNN Coach](#meta-lnn-coach)
8. [News Infrastructure](#news-infrastructure)
9. [Training Workflow](#training-workflow)
10. [Data Validation](#data-validation)
11. [Memory Optimization & GPU](#memory-optimization--gpu)
12. [Performance & Limitations](#performance--limitations)
13. [API Integration](#api-integration)
14. [Quick Reference](#quick-reference)

---

## System Overview

AutoTrade2 is a two-stage AI-powered stock trading analysis system:

### Stage 1: Technical Analysis & Alerts
- **Linear regression channels** with ping-pong pattern detection
- **Multi-timeframe RSI analysis** with confluence scoring
- **Claude AI news sentiment** and BS detection
- **Automated Telegram alerts** for high-confidence signals
- **Interactive Streamlit dashboard** with monitoring

### Stage 2: Multi-Scale LNN System
- **Multi-scale architecture**: 4 specialized LNN models (15min, 1hour, 4hour, daily)
- **Meta-LNN coach**: Adaptive combination based on market regime + news
- **135-feature extraction**: 11 timeframes × (7 channel + 3 RSI) + 25 base features
- **Real event integration**: TSLA earnings + macro events embedded in all models
- **Memory-efficient**: Lazy loading, ~2GB per model
- **News-ready**: LFM2-based headline encoding (optional)
- **Production-ready**: Ensemble prediction with SQLite logging

### Key Innovation: Intelligent Channel Selection
System evaluates ALL timeframes (1h, 2h, 3h, 4h, daily, weekly) and automatically selects the channel with:
- **70% Signal Confidence** (strength of trade setup)
- **25% RSI Confluence** (multi-timeframe alignment)
- **5% Channel Stability** (quality filter)

### Data System
- **Historical CSV data**: 10+ years (2015-2025), 1.35M aligned bars
- **Live data**: Automatic merge with yfinance (last 7 days)
- **Data validation**: Zero-tolerance for misaligned/fake data
- **Events**: 394 real events (TSLA earnings + macro)

---

## Quick Start

**For detailed step-by-step commands, see [QUICKSTART.md](QUICKSTART.md)**

### Stage 1 (Dashboard & Alerts)
```bash
pip install -r requirements.txt
python3 convert_data.py  # One-time data conversion
python main.py dashboard  # Launch Streamlit dashboard
```

### Stage 2 (Multi-Scale LNN Training)
```bash
# Install all dependencies
pip install -r requirements.txt

# Generate multi-scale CSVs (one-time)
python scripts/create_multiscale_csvs.py

# Train 4 specialized LNN models
python train_model_lazy.py --input_timeframe 15min --sequence_length 200 --epochs 50 --batch_size 128 --device cuda --output models/lnn_15min.pth
python train_model_lazy.py --input_timeframe 1hour --sequence_length 200 --epochs 50 --batch_size 128 --device cuda --output models/lnn_1hour.pth
python train_model_lazy.py --input_timeframe 4hour --sequence_length 200 --epochs 50 --batch_size 128 --device cuda --output models/lnn_4hour.pth
python train_model_lazy.py --input_timeframe daily --sequence_length 200 --epochs 50 --batch_size 128 --device cuda --output models/lnn_daily.pth

# Collect predictions
python backtest.py --model_path models/lnn_15min.pth --test_year 2023 --num_simulations 500
# (Repeat for other models...)

# Train Meta-LNN coach
python train_meta_lnn.py --mode backtest_no_news --epochs 100 --output models/meta_lnn.pth

# Validate on 2024 holdout
python backtest.py --model_path models/lnn_15min.pth --test_year 2024
```

**See [QUICKSTART.md](QUICKSTART.md) for complete command list.**

---

## File Structure

```
autotrade2/
├── .git/                                # Git repository
├── .gitignore                           # Excludes /data folder
├── README.md                            # User documentation
├── SPEC.md                              # This technical specification
├── QUICKSTART.md                        # Step-by-step command guide
├── config.py                            # Central configuration (all tunable parameters)
├── main.py                              # Stage 1 entry point
├── requirements.txt                     # Python dependencies (torch, ncps, transformers, etc.)
├── run.sh                               # Quick start menu
├── convert_data.py                      # Data conversion utility
│
├── data/                                # Stock data (gitignored)
│   ├── TSLAMin.txt                     # Raw TSLA data (original format)
│   ├── SPYMin.txt                      # Raw SPY data (original format)
│   ├── TSLA_1min.csv                   # Converted TSLA 1-minute (1.45M rows, ~93MB)
│   ├── SPY_1min.csv                    # Converted SPY 1-minute (1.79M rows, ~109MB)
│   ├── TSLA_5min.csv                   # Generated 5-minute TSLA bars
│   ├── TSLA_15min.csv                  # Generated 15-minute TSLA bars
│   ├── TSLA_30min.csv                  # Generated 30-minute TSLA bars
│   ├── TSLA_1hour.csv                  # Generated 1-hour TSLA bars
│   ├── TSLA_2hour.csv                  # Generated 2-hour TSLA bars
│   ├── TSLA_3hour.csv                  # Generated 3-hour TSLA bars
│   ├── TSLA_4hour.csv                  # Generated 4-hour TSLA bars
│   ├── TSLA_daily.csv                  # Generated daily TSLA bars
│   ├── TSLA_weekly.csv                 # Generated weekly TSLA bars
│   ├── TSLA_monthly.csv                # Generated monthly TSLA bars
│   ├── TSLA_3month.csv                 # Generated 3-month TSLA bars
│   ├── SPY_5min.csv                    # Generated 5-minute SPY bars
│   ├── SPY_15min.csv                   # Generated 15-minute SPY bars
│   ├── SPY_30min.csv                   # Generated 30-minute SPY bars
│   ├── SPY_1hour.csv                   # Generated 1-hour SPY bars
│   ├── SPY_2hour.csv                   # Generated 2-hour SPY bars
│   ├── SPY_3hour.csv                   # Generated 3-hour SPY bars
│   ├── SPY_4hour.csv                   # Generated 4-hour SPY bars
│   ├── SPY_daily.csv                   # Generated daily SPY bars
│   ├── SPY_weekly.csv                  # Generated weekly SPY bars
│   ├── SPY_monthly.csv                 # Generated monthly SPY bars
│   ├── SPY_3month.csv                  # Generated 3-month SPY bars
│   │                                   # (22 generated files, ~500MB total)
│   ├── tsla_events_REAL.csv            # Real events: earnings, deliveries (394 events)
│   ├── predictions.db                  # SQLite prediction database
│   └── news.db                         # SQLite news cache (optional, for live mode)
│
├── models/                              # Trained ML models
│   ├── lnn_15min.pth                   # 15-minute timeframe LNN (~1.4MB)
│   ├── lnn_1hour.pth                   # 1-hour timeframe LNN (~1.4MB)
│   ├── lnn_4hour.pth                   # 4-hour timeframe LNN (~1.4MB)
│   ├── lnn_daily.pth                   # Daily timeframe LNN (~1.4MB)
│   └── meta_lnn.pth                    # Meta-LNN coach (~0.6MB)
│
├── scripts/                             # Helper scripts
│   └── create_multiscale_csvs.py       # Generate multi-scale CSVs from 1-min data
│
├── src/                                 # Stage 1 source code
│   ├── data_handler.py                 # Data loading and resampling
│   ├── live_data_fetcher.py            # Live data from yfinance
│   ├── linear_regression.py            # Channel calculation
│   ├── rsi_calculator.py               # RSI and confluence
│   ├── news_analyzer.py                # AI news analysis
│   ├── signal_generator.py             # Signal generation
│   ├── telegram_bot.py                 # Telegram alerts
│   ├── gui_dashboard.py                # Basic dashboard (deprecated)
│   └── gui_dashboard_enhanced.py       # Enhanced dashboard with monitoring
│
└── src/ml/                              # Stage 2 ML system
    ├── __init__.py                     # ML module exports
    ├── base.py                         # Abstract interfaces (DataFeed, ModelBase, etc.)
    ├── data_feed.py                    # Data loading (CSVDataFeed, YFinanceDataFeed)
    ├── features.py                     # 135-feature extraction (11 timeframes)
    ├── features_lazy.py                # Feature extraction with progress bars
    ├── events.py                       # Event integration (earnings, macro)
    ├── model.py                        # LNN and LSTM model implementations
    ├── meta_models.py                  # Meta-LNN coach, market state features
    ├── ensemble.py                     # Multi-scale ensemble orchestrator
    ├── news_encoder.py                 # LFM2-based news headline encoding
    ├── fetch_news.py                   # News fetching (Google News RSS, Finnhub)
    ├── database.py                     # SQLite prediction logging (15 ensemble fields)
    ├── device_manager.py               # GPU/Metal hardware detection
    ├── interactive_params.py           # Parameter selection system (24 params)
    └── interactive_params_arrow.py     # Arrow-key navigation UI

# Stage 2 Training Scripts
├── train_model_lazy.py                  # Multi-scale LNN training (primary)
├── train_meta_lnn.py                    # Meta-LNN coach training
├── train_all_models.sh                  # Bash helper: train all 4 models sequentially
├── backtest.py                          # Walk-forward backtesting (reads metadata)
├── backtest_all_models.py               # Batch backtest all models
├── validate_results.py                  # Model validation reports
├── update_model.py                      # Online learning from errors
├── validate_data_alignment.py           # Data validation (zero-tolerance checking)
└── process_real_events.py               # Parse real events from RTF/JSON

# Test Scripts (for development/validation)
├── test_multiscale_features.py          # Test 135-feature extraction
├── test_meta_lnn.py                     # Test Meta-LNN architecture
├── test_news_system.py                  # Test news fetching/encoding
├── test_multiscale_system.py            # Comprehensive test suite
├── test_interactive_train_all.py        # Test "Train All 4" metadata flow
└── (10+ other test scripts)
```

---

## Stage 1: Linear Regression Trading System

### Core Components

#### 1. Data Handler (`src/data_handler.py`)
- Loads 1-minute CSV data
- Merges with live data from yfinance (last 7 days)
- Resamples to all timeframes (1h/2h/3h/4h/daily/weekly)
- Tracks data freshness (LIVE/RECENT/STALE/OUTDATED)

#### 2. Linear Regression Channel (`src/linear_regression.py`)
- Calculates regression channels with 2σ bands
- Detects ping-pong patterns (price bounces)
- Calculates stability score (R², ping-pongs, data points)
- **Projects 24 hours forward** for predicted high/low

**24-Hour Projection Algorithm:**
```python
# Calculate bars needed for 24 hours
bars_24h = {'1hour': 24, '3hour': 8, '4hour': 6, 'daily': 1}

# Project regression forward
future_x = [n, n+1, ..., n+bars_24h]
future_center = slope * future_x + intercept
future_upper = future_center + (2 * std_dev)
future_lower = future_center - (2 * std_dev)

# Find range
predicted_high = MAX(future_upper)
predicted_low = MIN(future_lower)
```

#### 3. RSI Calculator (`src/rsi_calculator.py`)
- Standard 14-period RSI
- Multi-timeframe confluence scoring
- Oversold (<30), Overbought (>70) detection
- Divergence detection

**RSI Confluence Algorithm:**
```python
# Primary timeframe RSI signal
if rsi < 30: signal = 'buy'
elif rsi > 70: signal = 'sell'

# Check higher timeframes for confirmation
confirmations = count_confirming_timeframes(signal, higher_tfs)

# Score: 40 (base) + 60 (confirmations/total)
confluence_score = 40 + (confirmations / total_checked) * 60
```

#### 4. News Analyzer (`src/news_analyzer.py`)
- Fetches news via NewsAPI (or mock data)
- Claude AI sentiment analysis (-100 to +100)
- **BS Score** (0-100): clickbait/rehash detection
- Recommendation: buy the dip if high BS + bearish

#### 5. Signal Generator (`src/signal_generator.py`)
**Main Algorithm:**
```
For each timeframe:
    1. Calculate channel position
    2. Calculate RSI confluence
    3. Generate signal confidence (0-100)
    4. Calculate composite score:
       = Confidence * 0.70 + RSI * 0.25 + Stability * 0.05

Select timeframe with MAX(composite_score)
Return TradingSignal with 24h forecast
```

**Signal Confidence Scoring:**
- Channel (0-30 pts): Price in lower/upper zone + stability
- RSI (0-40 pts): Confluence score * 0.4
- News (0-30 pts): High BS + bearish + buy signal = +15 pts

#### 6. Telegram Bot (`src/telegram_bot.py`)
- Sends formatted HTML alerts
- Triggers on: confidence ≥ 60, signal != neutral, signal changed
- Includes: signal, confidence, forecast, trade levels, reasoning

#### 7. Enhanced Dashboard (`src/gui_dashboard_enhanced.py`)
- Streamlit web GUI
- Candlestick chart with channel overlays
- Auto-zoomed to relevant timeframe
- Real-time monitoring with background thread
- Multi-timeframe RSI grid
- News panel with BS scores
- Integrated Telegram alerting

### Stage 1 Usage

**Dashboard Mode:**
```bash
python main.py dashboard
```

**One-Time Signal:**
```bash
python main.py signal --stock TSLA --timeframe 3hour
```

**Continuous Monitoring:**
```bash
python main.py monitor --stock TSLA --interval 30
```

**Test All Components:**
```bash
python main.py test
```

---

## Stage 2: Multi-Scale LNN System

### Architecture Overview

**Multi-Scale Approach:**
Stage 2 uses an ensemble of 4 specialized Liquid Neural Networks, each trained on a different timeframe, combined by a Meta-LNN "coach" that learns adaptive weighting based on market conditions.

```
TSLA_1min.csv + SPY_1min.csv (1.35M bars)
    ↓
Resample to multiple timeframes
    ├─ TSLA_15min.csv, SPY_15min.csv
    ├─ TSLA_1hour.csv, SPY_1hour.csv
    ├─ TSLA_4hour.csv, SPY_4hour.csv
    └─ TSLA_daily.csv, SPY_daily.csv
    ↓
Train 4 Specialized LNN Models (in parallel)
    ├─ LNN_15min → Learns 2-day intraday patterns
    ├─ LNN_1hour → Learns 8-day weekly patterns
    ├─ LNN_4hour → Learns 33-day swing patterns
    └─ LNN_daily → Learns 9-month seasonal patterns
    ↓
Each outputs: [predicted_high, predicted_low, confidence]
    ↓
Meta-LNN Coach
    ├─ Inputs: 4 predictions + 12 market-state features + 768 news embedding
    ├─ Learns: "Trust 15min during volatility, daily during trends"
    └─ Outputs: [final_high, final_low, final_confidence]
```

**Modular Design via Abstract Interfaces (`src/ml/base.py`):**
- `DataFeed` - Swappable data sources (CSV, IBKR, etc.)
- `FeatureExtractor` - Pluggable feature engineering
- `EventHandler` - Event data integration
- `ModelBase` - Swappable models (LNN, LSTM, Transformer)
- `PredictionDatabase` - Logging and analytics

### Core Components

#### 1. Data Feed (`src/ml/data_feed.py`)

**CSVDataFeed:**
- Loads SPY and TSLA 1-minute data
- **Inner join on timestamps** (zero tolerance for misalignment)
- Returns aligned DataFrame: 1,349,074 bars (75% of larger dataset)
- 100% quality, no nulls, no forward-fill

```python
feed = CSVDataFeed()
aligned_df = feed.load_aligned_data('2015-01-01', '2023-12-31')
# Returns DataFrame with: spy_open, spy_close, tsla_open, tsla_close, etc.
```

#### 2. Feature Extraction (`src/ml/features.py`)

**TradingFeatureExtractor** - Extracts 135 features across 11 timeframes:

| Category | Count | Features |
|----------|-------|----------|
| **Price** | 10 | SPY/TSLA: close, returns, log_returns, volatility_10, volatility_50 |
| **Channels** | 77 | **11 timeframes** (5min/15min/30min/1h/2h/3h/4h/daily/weekly/monthly/3month): position, upper_dist, lower_dist, slope, stability, ping_pongs, r_squared |
| **RSI** | 33 | **11 timeframes**: value, oversold, overbought |
| **Correlations** | 5 | correlation_10/50/200, divergence, divergence_magnitude |
| **Cycles** | 4 | distance_from_52w_high/low, within_mega_channel, mega_channel_position |
| **Volume** | 2 | tsla_volume_ratio, spy_volume_ratio |
| **Time** | 4 | hour_of_day, day_of_week, day_of_month, month_of_year (cyclical) |
| **Total** | **135** | Multi-scale temporal context at every bar |

**11 Timeframes for Channels & RSI:**
```
5min, 15min, 30min,        ← Intraday patterns
1h, 2h, 3h, 4h,            ← Hourly swing patterns
daily, weekly,              ← Daily/weekly trends
monthly, 3month             ← Long-term cycles
```

**Key Methods:**
```python
extractor = TradingFeatureExtractor()

# Extract features from aligned data
features_df = extractor.extract_features(aligned_df)  # Returns (N, 135)

# Create sequences for training (lazy loading)
dataset = LazyTradingDataset(features_df, sequence_length=200, target_horizon=24)
# Sequences created on-demand: (200, 135) per sample
# Targets: [predicted_high, predicted_low] for next 24 hours
```

**Multi-Scale Feature Calculation:**
- Resamples 1-minute data to each timeframe on-the-fly
- Calculates linear regression channel per timeframe
- Calculates RSI per timeframe
- Broadcasts features back to all 1-minute bars
- Each bar contains context from ALL timeframes simultaneously

**Leverages Stage 1:**
- Uses `LinearRegressionChannel` for all 11 timeframe channels
- Uses `RSICalculator` for all 11 timeframe RSI values
- Pure integration, no duplication

#### 3. Event Integration (`src/ml/events.py`)

**CombinedEventsHandler:**
- **TSLA events**: Earnings, deliveries from user CSV (86 events)
- **Macro events**: FOMC, CPI, NFP, Quad Witching (397 events)
- **Total**: 483 events, 394 in training range (2015-2023)

**Event Embeddings:**
```python
handler = CombinedEventsHandler('data/tsla_events_REAL.csv')

# Get events around a date
events = handler.get_events_for_date('2024-01-24', lookback_days=7)

# Convert to tensor
embedding = handler.embed_events(events)  # Returns (1, 21)
# 21 dimensions: TSLA one-hot (4) + days_until + beat/miss + macro one-hot (10) + days
```

#### 4. Sub-Models (`src/ml/model.py`)

**LNNTradingModel** - Liquid Neural Network (used for each timeframe):
```
Input (135 features, 200 timesteps)  ← Multi-scale features, uniform sequence length
  ↓
CfC Layer (Liquid Time-Constant, 128 hidden units)
  - Single layer only (CfC doesn't support stacking)
  - Sparse wiring via AutoNCP
  - Continuous-time dynamics
  - Emphasizes recent bars, integrates historical context
  ↓
Output Layer (2 units: high, low)
Confidence Head (1 unit: confidence score)
```

**Note:** The `num_layers` parameter in the menu **only affects LSTM**, not LNN. LNN always uses 1 CfC layer by design (the library doesn't support layer stacking). If you select LSTM as model_type, then num_layers will create stacked LSTM layers.

**4 Specialized Models (Each is an LNN):**
- **LNN_15min**: Trained on 15-minute data (200 bars = 50 hours)
- **LNN_1hour**: Trained on 1-hour data (200 bars = 8 days)
- **LNN_4hour**: Trained on 4-hour data (200 bars = 33 days)
- **LNN_daily**: Trained on daily data (200 bars = 9 months)

**Each sub-model:**
- Input size: 135 features (all 11 timeframes for multi-scale context)
- Hidden size: 128 units
- Sequence length: 200 bars (uniform across all models)
- Outputs: [predicted_high, predicted_low, confidence]

**Features:**
- **ncps library** - Closed-form Continuous-time (CfC) RNN
- **Sparse connections** - AutoNCP wiring for interpretability
- **Self-supervised pretraining** - 15% masking ratio
- **Event integration** - TSLA earnings + macro events embedded
- **Checkpoint system** - Save/load with timeframe metadata

**Usage:**
```python
# Each sub-model trains independently
model_15min = LNNTradingModel(input_size=135, hidden_size=128)
model_1hour = LNNTradingModel(input_size=135, hidden_size=128)
# etc.

# Each predicts independently
pred_15min = model_15min.predict(X_15min)
pred_1hour = model_1hour.predict(X_1hour)
# Returns: {
#   'predicted_high': array([250.5]),
#   'predicted_low': array([245.2]),
#   'confidence': array([0.85]),
#   'hidden_state': tensor(...)
# }

# Save model
model.save_checkpoint('models/lnn.pth', metadata={...})

# Online learning
model.update_online(X_new, y_actual, lr=0.0001)
```

**Alternative: LSTMTradingModel** - Traditional LSTM for comparison

#### 5. Database (`src/ml/database.py`)

**SQLitePredictionDB:**
- Logs every prediction with metadata
- Updates with actuals after 24 hours
- Calculates errors automatically
- Provides accuracy metrics

```python
db = SQLitePredictionDB('data/predictions.db')

# Log prediction
pred_id = db.log_prediction({
    'timestamp': '2024-01-24 14:00:00',
    'predicted_high': 250.5,
    'predicted_low': 245.2,
    'confidence': 0.85,
    'model_timeframe': '15min',  # Which model made prediction
    'is_ensemble': False,
    # ... 30+ fields
})

# Update with actuals
db.update_actual(pred_id, actual_high=252.1, actual_low=246.8)

# Get metrics
metrics = db.get_accuracy_metrics()
# Returns: mean_absolute_error, error by confidence bins, etc.
```

**Database Schema (predictions table):**

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| timestamp | DATETIME | When prediction was made |
| target_timestamp | DATETIME | When prediction is for |
| symbol | STRING | Stock symbol (TSLA) |
| timeframe | STRING | Prediction window (24h) |
| **model_timeframe** | STRING | Which model: '15min', '1hour', 'ensemble', etc. |
| **is_ensemble** | BOOLEAN | True if Meta-LNN prediction |
| **news_enabled** | BOOLEAN | True if news was used |
| predicted_high | FLOAT | Predicted high price |
| predicted_low | FLOAT | Predicted low price |
| predicted_center | FLOAT | Mid-point |
| predicted_range | FLOAT | High - Low |
| confidence | FLOAT | Model confidence (0-1) |
| actual_high | FLOAT | Actual high (filled later) |
| actual_low | FLOAT | Actual low (filled later) |
| actual_center | FLOAT | Actual mid-point |
| has_actuals | BOOLEAN | Whether actuals are filled |
| error_high | FLOAT | Prediction error (%) |
| error_low | FLOAT | Prediction error (%) |
| absolute_error | FLOAT | Average error |
| channel_position | FLOAT | Channel context |
| rsi_value | FLOAT | RSI context |
| spy_correlation | FLOAT | Correlation context |
| has_earnings | BOOLEAN | Earnings event |
| has_macro_event | BOOLEAN | Macro event (FOMC/CPI/NFP) |
| event_type | STRING | Event type |
| **sub_pred_15min_high** | FLOAT | 15min model prediction |
| **sub_pred_15min_low** | FLOAT | 15min model prediction |
| **sub_pred_15min_conf** | FLOAT | 15min model confidence |
| **sub_pred_1hour_high** | FLOAT | 1hour model prediction |
| **sub_pred_1hour_low** | FLOAT | 1hour model prediction |
| **sub_pred_1hour_conf** | FLOAT | 1hour model confidence |
| **sub_pred_4hour_high** | FLOAT | 4hour model prediction |
| **sub_pred_4hour_low** | FLOAT | 4hour model prediction |
| **sub_pred_4hour_conf** | FLOAT | 4hour model confidence |
| **sub_pred_daily_high** | FLOAT | daily model prediction |
| **sub_pred_daily_low** | FLOAT | daily model prediction |
| **sub_pred_daily_conf** | FLOAT | daily model confidence |
| model_version | STRING | Model version |
| feature_dim | INTEGER | Number of features |

**Total: 40+ fields** (boldface = ensemble-specific fields added in v3.0)

**News Database Schema (news.db):**

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| timestamp | TEXT | When article published |
| title | TEXT | Headline |
| source | TEXT | News source (Reuters, Bloomberg, etc.) |
| url | TEXT | Article URL |
| query_type | TEXT | 'TSLA' or 'MARKET' |
| created_at | TEXT | When fetched |

Indexed on timestamp and query_type for fast retrieval.
```

### Training Scripts

#### `train_model_lazy.py`
**Memory-efficient multi-scale training**

- Uses only **2-3 GB RAM** per model (lazy sequence loading)
- Supports all 11 timeframes via `--input_timeframe`
- Interactive mode with "Train All 4" option
- Stores timeframe + sequence_length in model metadata
- Progress bars for all phases

```bash
# Single model
python train_model_lazy.py --input_timeframe 15min --sequence_length 200 --epochs 50 --batch_size 128 --device cuda --output models/lnn_15min.pth

# Interactive (with multi-model option)
python train_model_lazy.py --interactive
```

#### `train_all_models.sh`
**Bash helper to train all 4 models sequentially**

- Configurable via environment variables
- Single command to train all models with same config
- Shows progress for each model

```bash
# Use defaults (epochs=50, batch_size=128, device=cuda)
./train_all_models.sh

# Or customize
EPOCHS=100 BATCH_SIZE=256 DEVICE=mps ./train_all_models.sh
```

#### `backtest.py`
**Walk-forward backtesting with metadata support**

- **Now reads model metadata** (input_timeframe, sequence_length)
- Automatically loads correct CSV files based on model's timeframe
- Tests model on unseen data (e.g., 2024)
- Random day/week simulation
- Logs predictions to database with model_timeframe tag

```bash
python backtest.py --model_path models/lnn_15min.pth --test_year 2024 --num_simulations 100
```

#### `backtest_all_models.py`
**Auto-backtest all trained models**

- Finds all models in models/ directory
- Runs backtests sequentially
- Shows combined comparison summary

```bash
python backtest_all_models.py --test_year 2023 --num_simulations 500
```

#### `train_meta_lnn.py`
**Train Meta-LNN coach**

- Loads predictions from database
- Purged K-fold CV (prevents leakage)
- Trains adaptive combination model

```bash
python train_meta_lnn.py --mode backtest_no_news --epochs 100 --output models/meta_lnn.pth
```

#### `validate_results.py`
**Model performance validation**

- Generates comprehensive report
- Error analysis by confidence bins
- Recommendations for improvement

#### `update_model.py`
**Online learning from prediction errors**

- Loads accumulated predictions from database
- Fine-tunes model on real errors
- Incremental learning

---

## Multi-Scale Architecture

### Why Multi-Scale?

**Problem with single-model approach:**
- To see monthly patterns on 1-min data: need sequence_length=50,000+ bars
- Memory explosion: 50,000 × 135 features × 4 bytes × batch_size = ~3.4GB per batch
- Training impractical on consumer hardware

**Multi-scale solution:**
- Train separate models on different timeframe CSVs
- Each model: sequence_length=200 bars (uniform, manageable)
- Each specializes in its temporal scale
- Combine predictions via Meta-LNN coach

### Temporal Coverage

| Model | Input Data | Sequence | Time Span | Specialization |
|-------|------------|----------|-----------|----------------|
| LNN_15min | TSLA_15min.csv | 200 bars | 50 hours (2 days) | Intraday volatility, gaps, reversals |
| LNN_1hour | TSLA_1hour.csv | 200 bars | 200 hours (8 days) | Weekly cycles, earnings momentum |
| LNN_4hour | TSLA_4hour.csv | 200 bars | 800 hours (33 days) | Swing patterns, multi-week trends |
| LNN_daily | TSLA_daily.csv | 200 bars | 200 days (9 months) | Seasonal effects, market cycles |

### Multi-Scale Features

**Key Insight:** All models receive 135 features per bar, calculated across 11 timeframes.

Even the 15min model sees:
```python
Bar at 2024-01-15 10:30:
  - channel_5min_position: 0.65     ← Where in 5-min channel
  - channel_15min_position: 0.73    ← Where in 15-min channel (native)
  - channel_1h_position: 0.81       ← Where in 1-hour channel
  - channel_4h_position: 0.45       ← Where in 4-hour channel
  - channel_daily_position: 0.82    ← Where in daily channel
  - channel_weekly_position: 0.58   ← Where in weekly channel
  - channel_monthly_position: 0.71  ← Where in monthly channel
  ... (135 features total with ALL timeframe context)
```

**This provides cross-timeframe context:**
- 15min model learns: "When daily channel oversold + 1h channel breaking up → bounce likely"
- 4hour model learns: "When weekly trend down but daily RSI extreme → short-term reversal"

### Memory Efficiency

| Approach | Sequence | Features | Memory per Model | Total Memory |
|----------|----------|----------|------------------|--------------|
| Single model (long seq) | 50,000 | 135 | ~25GB | ~25GB |
| **Multi-scale (4 models)** | **200** | **135** | **~2GB** | **~8GB** |

**Savings: 3x less memory, can train on consumer GPUs**

---

## Metadata System

### How Metadata Flows Through the Pipeline

**Critical for multi-scale**: Each model checkpoint contains metadata about how it was trained. This allows backtest.py to automatically use the correct data and configuration.

###Step 1: Training Saves Metadata

When you train a model:
```bash
python train_model_lazy.py --input_timeframe 15min --sequence_length 200 --epochs 50 --batch_size 128 --output models/lnn_15min.pth
```

**Metadata saved in lnn_15min.pth:**
```python
{
    'model_type': 'LNN',
    'input_size': 135,                    # Number of features
    'hidden_size': 128,
    'input_timeframe': '15min',           # ← CRITICAL: Which CSV was used
    'sequence_length': 200,               # ← CRITICAL: Bars to look back
    'train_start_year': 2015,
    'train_end_year': 2023,
    'epochs': 50,
    'pretrain_epochs': 10,
    'final_train_loss': 0.0234,
    'final_val_loss': 0.0289,
    'training_date': '2024-11-12T10:30:00',
    'feature_names': ['spy_close', 'tsla_close', ...],  # All 135 feature names
    'training_mode': 'lazy_loading',
    'peak_memory_mb': 2456.7,
    'device': 'cuda:0',
    'device_type': 'cuda'
}
```

### Step 2: Backtest Reads Metadata

When you backtest:
```bash
python backtest.py --model_path models/lnn_15min.pth --test_year 2024
```

**backtest.py automatically:**
1. Loads checkpoint: `torch.load('models/lnn_15min.pth')`
2. Extracts metadata: `metadata = checkpoint['metadata']`
3. Reads config:
   - `input_timeframe = metadata['input_timeframe']`  # '15min'
   - `sequence_length = metadata['sequence_length']`  # 200
4. **Loads correct CSVs:**
   - `CSVDataFeed(timeframe='15min')`  # Loads TSLA_15min.csv, SPY_15min.csv
5. **Uses correct sequence length:**
   - `sequence = features_df.tail(200).values`  # Not hardcoded 84!

**This ensures backtest uses EXACT same data/config as training!**

### Step 3: Metadata in Predictions Database

Each prediction logged to predictions.db includes:
- `model_timeframe`: '15min', '1hour', '4hour', 'daily', or 'ensemble'
- Allows querying: "Show me all 15min model predictions"
- Enables comparison: "Which timeframe is most accurate?"

### Interactive "Train All 4" Metadata Flow

When using interactive multi-model training:

```
User configures once:
  epochs: 50
  batch_size: 128
  sequence_length: 200
  device: cuda
  (all 24 parameters)
    ↓
Trains 4 models sequentially:
    ↓
Model 1 (15min):
  metadata['input_timeframe'] = '15min'
  metadata['sequence_length'] = 200      ← User's setting
  metadata['epochs'] = 50                ← User's setting
  metadata['batch_size'] = 128           ← User's setting
  Saved to: models/lnn_15min.pth
    ↓
Model 2 (1hour):
  metadata['input_timeframe'] = '1hour'  ← CHANGED
  metadata['sequence_length'] = 200      ← Same (user setting)
  metadata['epochs'] = 50                ← Same (user setting)
  Saved to: models/lnn_1hour.pth
    ↓
(Repeat for 4hour, daily)
```

**Result:** 4 model files, each with correct timeframe + shared user settings.

### Metadata Compatibility

**Forward compatibility:**
- Future versions can add new metadata fields
- Old models still load (missing fields use defaults)

**Backward compatibility:**
- Models without input_timeframe default to '1min'
- Models without sequence_length use config.ML_SEQUENCE_LENGTH

**Validation:**
- backtest.py warns if metadata seems incompatible
- Checks feature_dim matches extractor output

### Multi-Timeframe CSV Loading (v3.2 Fix)

**CRITICAL BUG FIX:** Prior to v3.2, all models trained on `TSLA_1min.csv` regardless of `--input_timeframe` parameter.

**How It Now Works:**

`load_and_prepare_data_lazy()` now receives the `timeframe` parameter and passes it to CSVDataFeed:

```python
# train_model_lazy.py line 154
def load_and_prepare_data_lazy(..., timeframe='1min'):
    ...
    data_feed = CSVDataFeed(timeframe=timeframe)  # ← Uses correct timeframe
    aligned_df = data_feed.load_aligned_data(start_date, end_date)
```

**CSV Selection Logic:**

- `CSVDataFeed(timeframe='15min')` → loads `data/SPY_15min.csv` + `data/TSLA_15min.csv`
- `CSVDataFeed(timeframe='1hour')` → loads `data/SPY_1hour.csv` + `data/TSLA_1hour.csv`
- `CSVDataFeed(timeframe='4hour')` → loads `data/SPY_4hour.csv` + `data/TSLA_4hour.csv`
- `CSVDataFeed(timeframe='daily')` → loads `data/SPY_daily.csv` + `data/TSLA_daily.csv`

**Path Construction (data_feed.py line 34):**
```python
csv_path = Path(self.data_dir) / f"{symbol}_{self.timeframe}.csv"
```

**Verification During Training:**

Print statement now shows correct timeframe:
```
✓ Loaded 123,456 aligned 15min bars
```
(not hardcoded "1-minute bars")

**Impact:**
- Each model now trains on its designated multi-timeframe CSV
- True multi-scale architecture finally functional
- Models learn different temporal patterns as designed
- **All models trained before v3.2 are invalid and should be retrained**

---

## Prediction Horizon (Critical Parameter)

### Understanding prediction_horizon

**IMPORTANT:** Despite the parameter name `prediction_horizon_hours`, this is measured in **BARS, NOT HOURS!**

The model predicts the high/low prices over the next N bars (where N = prediction_horizon).

### How It Works

```python
# Training creates targets (train_model_lazy.py line 107)
target_start = seq_end
target_end = seq_end + target_horizon  # target_horizon BARS ahead

future_prices = features_array[target_start:target_end, close_idx]
target_high = np.max(future_prices)  # Max price in next N bars
target_low = np.min(future_prices)   # Min price in next N bars
```

**Model learns:** "Given last 200 bars, what will be the high/low in NEXT 24 bars?"

### Prediction Modes (NEW in v3.1)

The system now supports two prediction modes:

#### 1. **Uniform Bars Mode** (Default)
All models use the same number of bars (e.g., 24), resulting in different time windows.

**With prediction_horizon=24 (default):**

| Model Timeframe | Bars Ahead | Actual Time Window |
|-----------------|------------|-------------------|
| 15min | 24 bars | 6 hours |
| 1hour | 24 bars | 24 hours (1 day) ✓ |
| 4hour | 24 bars | 4 days |
| Daily | 24 bars | 24 DAYS (not hours!) |

**Philosophy:** Each model learns at its natural temporal scale. LNNs handle temporal relationships through multi-scale features.

#### 2. **Uniform Time Mode**
All models predict the same absolute time window (24 hours), using different bar counts.

**For 24-hour prediction:**

| Model Timeframe | Bars Ahead | Actual Time Window |
|-----------------|------------|-------------------|
| 15min | 96 bars | 24 hours |
| 1hour | 24 bars | 24 hours |
| 4hour | 6 bars | 24 hours |
| Daily | 1 bar | 24 hours |

**Philosophy:** All models solve the same problem from different perspectives. Meta-LNN learns which timeframe is most accurate.

#### Selecting Prediction Mode

**Interactive Menu:**
```
╔════════════════════════════════════════════════╗
║ Prediction Horizon Mode                        ║
╠════════════════════════════════════════════════╣
║ [1] Uniform Bars (same bar count for all)      ║
║ [2] Uniform Time (24 hours for all models)     ║
╚════════════════════════════════════════════════╝
```

**Command Line:**
```bash
# Uniform bars mode (default)
python train_model_lazy.py --prediction_mode uniform_bars --prediction_horizon 24

# Uniform time mode (24 hours for all)
python train_model_lazy.py --prediction_mode uniform_time --prediction_horizon 24
```

**Multi-Model Training:**
- Selected mode automatically applies to all 4 models
- In uniform_time mode, prediction_horizon is automatically adjusted per timeframe
- Metadata stores both prediction_horizon and prediction_mode
- Backtesting correctly interprets based on saved mode

### Customizing Per Model

**You can now set different horizons per model:**

```bash
# Short-term intraday (6 hours out)
python train_model_lazy.py --input_timeframe 15min --prediction_horizon 24 ...

# Medium-term (2 days out)
python train_model_lazy.py --input_timeframe 1hour --prediction_horizon 48 ...

# Long-term swing (2 weeks out)
python train_model_lazy.py --input_timeframe 4hour --prediction_horizon 84 ...

# Very long-term (2 months out)
python train_model_lazy.py --input_timeframe daily --prediction_horizon 42 ...
```

### Recommendations

**By Timeframe:**
- **15min model:** prediction_horizon=24 (6 hours) - Intraday trading
- **1hour model:** prediction_horizon=24-48 (1-2 days) - Swing entries
- **4hour model:** prediction_horizon=42-84 (7-14 days) - Swing exits
- **Daily model:** prediction_horizon=21-63 (3 weeks - 3 months) - Position trading

**Trade-offs:**
- ✅ Shorter horizon: More accurate, easier to predict
- ✅ Longer horizon: More strategic, captures bigger moves
- ⚠️ Too short: Noise dominates
- ⚠️ Too long: Prediction becomes too uncertain

**In Interactive Menu:**
- Parameter shown as "Prediction horizon (hours)"
- **Actually means BARS**
- Hint clarifies: "24 bars = 6hrs(15min), 24hrs(1hour), 24 days(daily)"

---

## Meta-LNN Coach

### Purpose

The Meta-LNN "coach" learns to adaptively combine predictions from the 4 sub-models based on:
- Current market regime (volatility, jumps, events)
- Time of day / week (market microstructure)
- Optionally: News sentiment (LFM2-encoded headlines)

### Architecture (`src/ml/meta_models.py`)

```
Inputs:
  ├─ Sub-predictions: [4 models × 3 values] = 12 features
  │    ├─ 15min: [high, low, confidence]
  │    ├─ 1hour: [high, low, confidence]
  │    ├─ 4hour: [high, low, confidence]
  │    └─ daily: [high, low, confidence]
  │
  ├─ Market state: 12 features
  │    ├─ Realized volatility (5m, 30m, 1d)
  │    ├─ Overnight return absolute
  │    ├─ Intraday jump flag (|ret| > 3σ)
  │    ├─ Volatility z-score
  │    ├─ Time of day (sin/cos)
  │    ├─ Event proximity (earnings, macro)
  │    ├─ SPY correlation regime
  │    └─ VIX level
  │
  ├─ News embedding: 768 features (LFM2-350M, optional)
  └─ News mask: 1 feature (news available flag)

  Total input: 12 + 12 + 768 + 1 = 793 features
    ↓
Liquid Neural Network (64 hidden units)
  ├─ CfC layer with AutoNCP wiring
  ├─ Learns adaptive combination weights
  └─ ~160K parameters (tiny!)
    ↓
Output heads:
  ├─ fc_high: Predicted high
  ├─ fc_low: Predicted low
  └─ fc_conf: Final confidence
```

### Training

**Data source:** `data/predictions.db` (collected from Step 3 backtests)

**Training approach:**
- Purged K-fold CV (5 folds, 7-day embargo)
- Prevents temporal leakage
- Huber loss for prices (robust to outliers)
- MSE for confidence calibration

**Two modes:**
1. **backtest_no_news** (default): News disabled (zeros), pure numeric learning
2. **live_with_news** (future): News enabled, requires LFM2

### What Meta-LNN Learns

Example learned behaviors:
- "During high volatility (rv_5m > 0.02) → weight 15min higher (0.5 vs 0.15)"
- "During trending markets (vol_zscore < -1) → weight daily higher (0.6 vs 0.2)"
- "Around earnings (has_earnings_soon=1) → boost 1hour (captures momentum)"
- "With news spike (news_mask=1) → dynamically adjust based on sentiment"

**Not fixed rules - learned from data!**

---

## News Infrastructure

### Overview

Optional news integration using Liquid AI's LFM2 foundation model for headline encoding.

**Two operational modes:**
1. **backtest_no_news**: News disabled (for backtesting without news data)
2. **live_with_news**: News enabled (for live trading with real-time headlines)

### Components

#### 1. News Fetching (`src/ml/fetch_news.py`)

**Sources:**
- Google News RSS (no API key required)
- Finnhub API (optional, requires key)

**Queries:**
```python
TSLA_QUERY = "Tesla OR TSLA OR Cybertruck OR Giga OR FSD OR Autopilot..."
MARKET_QUERY = "FOMC OR Fed OR CPI OR NFP OR yields OR VIX OR S&P 500..."
```

**Filtering:**
- Whitelist: Reuters, Bloomberg, WSJ, FT, CNBC, MarketWatch, etc.
- Exclude: Reviews, coupons, rumors, promotional content

**Storage:** `data/news.db` with leak-safe timestamps

#### 2. News Encoding (`src/ml/news_encoder.py`)

**NewsEncoder class:**
- Uses LFM2-350M as frozen encoder (~1.5GB download)
- Encodes top-10 headlines into 768-dim embedding
- Mode-aware: Returns zeros in backtest mode

```python
encoder = NewsEncoder(mode='backtest_no_news')  # or 'live_with_news'
news_vec, news_mask = encoder.encode_headlines(headlines, timestamp)
# Returns: (768,) tensor and 0/1 mask
```

#### 3. Modality Dropout

During Meta-LNN training:
- 40% probability of zeroing out news_vec
- Forces model to handle both with-news and no-news states
- Prevents over-reliance on news signal
- Robust to missing news in production

### News Retrieval (Live Mode)

```python
# Fetch news for 2-hour window before prediction
news_articles = get_news_window(
    timestamp=current_time,
    lookback_minutes=120,
    query_types=['TSLA', 'MARKET']
)

# Returns top-k TSLA + top-k MARKET headlines (k=5 each)
# Encoded by LFM2 → 768-dim embedding → Fed to Meta-LNN
```

### Future Enhancements

- **FinBERT sentiment**: Add sentiment scores alongside LFM2 embeddings
- **News gate**: Learn dynamic weighting of news vs. numeric features
- **Multi-source ensemble**: Combine Google News, Finnhub, Twitter (if available)

---

## Training Workflow

### Quick Training Guide

For step-by-step commands, see **[QUICKSTART.md](QUICKSTART.md)** - Simple command guide without explanations.

### Step 0: Generate Multi-Scale CSVs (One-Time)

```bash
python scripts/create_multiscale_csvs.py
```

Resamples 1-minute data to 11 timeframes. Output: ~22 CSV files, ~500MB.

### Step 1: Train Sub-Models (4 Specialized LNNs)

```bash
# Train all 4 models (can run in parallel on GPU)
python train_model_lazy.py --input_timeframe 15min --sequence_length 200 --epochs 50 --batch_size 128 --device cuda --output models/lnn_15min.pth
python train_model_lazy.py --input_timeframe 1hour --sequence_length 200 --epochs 50 --batch_size 128 --device cuda --output models/lnn_1hour.pth
python train_model_lazy.py --input_timeframe 4hour --sequence_length 200 --epochs 50 --batch_size 128 --device cuda --output models/lnn_4hour.pth
python train_model_lazy.py --input_timeframe daily --sequence_length 200 --epochs 50 --batch_size 128 --device cuda --output models/lnn_daily.pth
```

**Time:** ~15-25 minutes per model on T4 GPU

### Step 2: Collect Predictions (Backtest)

```bash
# Run backtests to populate predictions.db
python backtest.py --model_path models/lnn_15min.pth --test_year 2023 --num_simulations 500
python backtest.py --model_path models/lnn_1hour.pth --test_year 2023 --num_simulations 500
python backtest.py --model_path models/lnn_4hour.pth --test_year 2023 --num_simulations 500
python backtest.py --model_path models/lnn_daily.pth --test_year 2023 --num_simulations 500
```

**Time:** ~30-60 minutes total

### Step 3: Train Meta-LNN Coach

```bash
python train_meta_lnn.py --mode backtest_no_news --epochs 100 --output models/meta_lnn.pth
```

**Time:** ~10-15 minutes

### Step 4: Validate on Holdout (2024)

```bash
# Test individual models
python backtest.py --model_path models/lnn_15min.pth --test_year 2024 --num_simulations 100
# Compare accuracies to find best model/ensemble
```

### Interactive Parameter Selection (Optional)

The training script supports an **interactive parameter selection system** with arrow-key navigation:

```bash
# Launch interactive mode
python3 train_model_lazy.py --interactive
```

**Multi-Model Training Option:**
When launching interactive mode, you'll be asked:
```
Training mode:
  1. Single model (choose one timeframe)
  2. All 4 models (15min, 1hour, 4hour, daily) - runs sequentially
```

**Select Option 2 to:**
- Configure parameters ONCE (epochs, batch_size, sequence_length, device, etc.)
- Train all 4 models automatically (15min → 1hour → 4hour → daily)
- Each model gets same user settings (epochs, batch_size, etc.) but different timeframe
- Metadata correctly saved with each model's timeframe + user settings
- Total time: ~60-100 minutes on GPU (sequential)

**Metadata Flow (Verified):**
- All 24 parameters you configure flow to ALL 4 models
- Only input_timeframe differs (15min, 1hour, 4hour, daily)
- Each model checkpoint contains correct metadata:
  - `input_timeframe`: Different for each ('15min', '1hour', etc.)
  - `sequence_length`: Same for all (your setting, e.g., 200)
  - `epochs`: Same for all (your setting, e.g., 50)
  - `batch_size`: Same for all (your setting, e.g., 128)
  - All other settings: Same for all (your configured values)

**Three Selection Modes (for single-model):**
1. **Quick Start** - Use defaults with auto-detected device (fastest)
2. **Arrow-Key Navigation** - Visual menu with ↑/↓ navigation (recommended)
3. **Number-Based Menu** - Type parameter numbers to select (legacy)

**Arrow-Key Navigation Features:**
- Navigate all 24 parameters with ↑/↓ arrow keys
- Press Enter to edit the highlighted parameter
- Visual pointer (`❯`) shows current selection
- Modified parameters marked with `*`
- Organized in 7 categories:
  - 📁 **Data Files**: SPY/TSLA data paths, events file, macro API key
  - 📅 **Training Period**: Start/end years
  - 🧠 **Model Architecture**: Model type, hidden size, layers (LSTM only), sequence length
  - ⚙️ **Training Parameters**: Epochs, batch size, learning rate, validation split
  - 📊 **Feature Flags**: Channel, RSI, correlation, event features
  - 🖥️ **Compute Device**: CPU/CUDA/MPS with auto-detection
  - 🚀 **GPU Optimization**: num_workers, pin_memory (auto-optimizes for GPU)

**RAM-Aware Batch Size Suggestions:**
```
💡 Batch size suggestions (based on 16.0 GB available RAM):
   Conservative : 128 (safe, slower)
   Balanced     : 256 (recommended)
   Aggressive   : 512 (fast, may OOM)
```

**Parameter-Specific Editors:**
- **Numbers**: Input with validation ranges (e.g., hidden_size: 32-1024)
- **Years**: Range-validated (2010-2024)
- **Booleans**: Yes/No confirmation
- **Device**: List selector showing available devices
- **Batch Size**: Special menu with RAM-based suggestions
- **Macro API**: Note that local macro data exists without API key

**Implementation:**
- `src/ml/interactive_params_arrow.py` - Arrow-key navigation system (InquirerPy-based)
- `src/ml/interactive_params.py` - Mode selection and routing
- `src/ml/device_manager.py` - Hardware detection for device selection
- Graceful fallback to number-based menu if InquirerPy not installed

**Dependencies:**
```bash
pip install inquirerpy  # For arrow-key navigation
```

### Step 1: Data Validation (MANDATORY!)

```bash
python3 validate_data_alignment.py --events_data data/tsla_events_REAL.csv
```

**What it checks:**
- ✓ Files exist and have required columns
- ✓ No nulls in price data
- ✓ No zeros in price data
- ✓ Timestamps sorted and unique
- ✓ SPY-TSLA alignment (inner join)
- ✓ Events within data range
- ✓ Sufficient data for training

**Expected output:**
```
✅ DATA VALIDATION PASSED - SAFE TO TRAIN!
  - 1,349,074 perfectly aligned SPY/TSLA bars
  - 394 events in training range
  - No nulls, zeros, or fake data
```

**If validation fails:** Fix errors before training! Never train on invalid data.

### Step 2: Training (Memory-Efficient)

**Quick Test (10-15 minutes):**
```bash
python3 train_model_lazy.py \
  --tsla_events data/tsla_events_REAL.csv \
  --start_year 2023 \
  --end_year 2023 \
  --epochs 10 \
  --pretrain_epochs 3 \
  --batch_size 8 \
  --output models/lnn_quick.pth
```

**Full Training (60-90 minutes):**
```bash
python3 train_model_lazy.py \
  --tsla_events data/tsla_events_REAL.csv \
  --epochs 50 \
  --pretrain_epochs 10 \
  --batch_size 16 \
  --output models/lnn_full.pth
```

**Training Process:**
1. **Load Data** (10 sec) - Load 1.35M aligned bars
2. **Extract Features** (3-5 min) - Calculate 56 features with progress
3. **Load Events** (2 sec) - Load 394 events
4. **Pretraining** (5-10 min/epoch) - Self-supervised learning with batch progress
5. **Training** (5-10 min/epoch) - Supervised learning with live metrics
6. **Save Model** - Checkpoint with metadata

**Progress Feedback:**
- Step-by-step feature extraction (7 steps visible)
- Batch-level progress during pretraining (84,000+ batches/epoch)
- Real-time loss, memory, and timing

### Step 3: Backtesting

```bash
python3 backtest.py \
  --model_path models/lnn_full.pth \
  --test_year 2024 \
  --num_simulations 100
```

**Output:**
```
Average Metrics:
  Mean Error (High): 3.12%
  Mean Error (Low): 2.89%
  Mean Confidence: 0.68

Error by Event Type:
  With Earnings: 4.23% (15 cases)
  No Events: 2.67% (85 cases)
```

**Good Results:** <5% mean absolute error
**Excellent Results:** <3% mean absolute error

### Step 4: Validation

```bash
python3 validate_results.py --model_path models/lnn_full.pth
```

Check `reports/validation_report.txt` for detailed analysis.

### Step 5: Online Learning (Optional)

After live predictions accumulate:

```bash
python3 update_model.py \
  --model_path models/lnn_full.pth \
  --output models/lnn_updated.pth
```

---

## Data Validation

### Zero-Tolerance Policy

**NO fake data, NO misalignment, NO approximations:**
- ❌ NO nulls allowed
- ❌ NO zeros in prices
- ❌ NO forward-filling gaps
- ❌ NO interpolation
- ✅ ONLY use exact timestamp matches

### Inner Join Alignment

```python
# SPY-TSLA alignment
common_timestamps = spy_df.index.intersection(tsla_df.index)
spy_aligned = spy_df.loc[common_timestamps]
tsla_aligned = tsla_df.loc[common_timestamps]

# Result: 1,349,074 perfectly aligned bars
# 75% of larger dataset (SPY), but 100% quality
```

**Why 75% alignment?**
- SPY trades extended hours (4am-8pm)
- TSLA has fewer total bars
- Inner join keeps ONLY overlapping timestamps
- **Result: 100% quality, 75% quantity** ✅

### Real Events Only

**Sources:**
- `data/earnings:P&D.rtf` - TSLA earnings/deliveries (user-provided)
- `data/historical_events.txt` - Macro events JSON (user-provided)

**Processing:**
```bash
python3 process_real_events.py
# Output: data/tsla_events_REAL.csv with 483 real events
```

**Validation:**
- Events matched to trading dates
- Beat/miss outcomes tracked
- Dates within data range

---

## Memory Optimization

### The Problem

**Original `train_model.py`:**
```python
# Creates ALL sequences upfront
X, y = create_sequences(features_df)  # 30.5 GB tensor!
# Process killed by OS
```

**For 1.35M bars:**
- Sequences: 1,349,892
- Sequence length: 84
- Features: 56
- Memory: 1,349,892 × 84 × 56 × 4 bytes = **30.5 GB**
- Peak (during numpy→torch): **60 GB**

### The Solution: Lazy Loading

**`train_model_lazy.py`:**
```python
class LazyTradingDataset(Dataset):
    def __init__(self, features_df, ...):
        self.features_df = features_df  # Store DataFrame (~2 GB)

    def __getitem__(self, idx):
        # Create ONE sequence on-demand
        seq = self.features_df.iloc[idx:idx+84].values
        return torch.tensor(seq)
```

**Memory Usage:**
- Features DataFrame: ~2 GB
- Batch (16 sequences): ~50 MB
- Total: **2-3 GB constant** (no spikes!)

**Benefits:**
- ✅ Trains on ALL 1.35M bars (no data loss)
- ✅ Works on normal hardware (8 GB RAM)
- ✅ Scales to any dataset size
- ✅ Same model quality

**Training Workflow:**
```
Load Features (2 GB)
  ↓
For each batch:
    Create 16 sequences on-the-fly
    Train on batch
    Garbage collect
    (Memory stays at 2-3 GB)
```

---

## Performance & Limitations

### Current Performance

**Stage 1 (Signal Generation):**
- Data loading: 2-3 seconds
- Channel calculations: ~1 second (6 timeframes)
- RSI calculations: <1 second
- News analysis: 2-5 seconds (Claude API)
- **Total:** 5-10 seconds per signal

**Stage 2 (Training):**
- Data loading: ~10 seconds
- Feature extraction: 3-5 minutes (1.35M bars)
- Pretraining: 5-10 min/epoch
- Training: 5-10 min/epoch
- **Full training:** 60-90 minutes (50 epochs)

**Memory:**
- Stage 1: ~500 MB per stock
- Stage 2 (lazy): ~2-3 GB constant
- Stage 2 (original): ~30+ GB (crashes)

### Limitations

#### 1. GPU/Metal Support
**Status:** ✅ IMPLEMENTED (v2.0)

The training scripts now support full GPU/Metal acceleration with automatic hardware detection and comprehensive error reporting:

**Supported Hardware:**
- **Apple Silicon (MPS)**: M1, M2, M3, M4, M5 Pro/Max/Ultra
- **NVIDIA CUDA**: Google Colab, local NVIDIA GPUs
- **CPU Fallback**: Compatible with any system

**Device Selection:**
```python
# Interactive mode (default) - prompts user
python train_model.py --tsla_events data/tsla_events_REAL.csv

# Auto-select best available device
python train_model.py --auto_device

# Force specific device
python train_model.py --device mps  # Apple Silicon
python train_model.py --device cuda  # NVIDIA GPU
python train_model.py --device cpu   # CPU only
```

**Performance Gains:**
- **Apple Silicon (M2 Max)**: ~3-5x speedup (300-500 seq/sec vs 100 CPU)
- **Google Colab T4**: ~4-6x speedup (400-600 seq/sec)
- **Training time**: 50 epochs in ~20-30 min (vs 90 min CPU)

**Implementation:**
- `src/ml/device_manager.py`: Hardware detection and device management
- Automatic device transfer for all tensors during training
- Cross-device model loading support
- Memory-efficient with lazy loading (~2-3GB usage)

**Testing:**
```bash
# Run comprehensive device compatibility tests
python test_device_compatibility.py
```

#### 2. Adding New Features Requires Code Changes

**Current Implementation:**
Feature extraction is hardcoded to SPY/TSLA:
```python
for symbol in ['spy', 'tsla']:  # Hardcoded!
    features_df[f'{symbol}_close'] = df[f'{symbol}_close']
```

**To add VXX (or any new symbol):**

**Required Changes:**
1. `src/ml/data_feed.py` - 3-way alignment (SPY/TSLA/VXX)
2. `src/ml/features.py` - Add VXX features, update feature names
3. Validate 3-way alignment
4. Retrain from scratch with new input size

**What stays the same:**
- Training scripts (agnostic to input size)
- Model architecture (handles any input size)
- Events system
- The core workflow

**Why it's not plug-and-play:**
The feature extraction expects specific column names and specific symbol pairs. This is why adding new data sources "messes it up" in typical implementations.

**Future:** Dynamic feature addition system

#### 3. Other Limitations

- **yfinance Data:** Only 7 days of 1-minute data available
- **Single Stock Focus:** Dashboard shows one stock at a time
- **No Paper Trading:** No simulation or execution
- **Claude API Costs:** $0.00025 per article analyzed
- **No Real-time Feeds:** WebSocket integration not yet implemented

---

## API Integration

### 1. yfinance (Stock Data)
**Endpoint:** Yahoo Finance API (via yfinance library)
**Rate Limits:** None known
**Data:** OHLCV, 1-minute intervals, last 7 days maximum

```python
ticker = yf.Ticker('TSLA')
df = ticker.history(period='7d', interval='1m')
```

### 2. Claude AI API (News Analysis)
**Endpoint:** Anthropic Messages API
**Model:** claude-3-haiku-20240307 (fast, cost-effective)
**Cost:** ~$0.00025 per article

```python
client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=500,
    messages=[{"role": "user", "content": prompt}]
)
```

### 3. Telegram Bot API
**Endpoint:** Telegram Bot API
**Method:** `sendMessage`
**Format:** HTML with bold tags and emojis
**Async:** python-telegram-bot library

### 4. NewsAPI (Optional)
**Endpoint:** newsapi.org/v2/everything
**Fallback:** Mock articles if API key not configured

---

## Future Improvements

### Planned Features

**Performance:**
- ✅ **GPU/Metal Support** - M1/M2/M3/M4/M5 acceleration (COMPLETED v2.0)
- **Quantization** - Model compression for edge deployment
- **Distributed Training** - Multi-GPU support

**Features:**
- ✅ **Dynamic Feature Addition** - Add symbols without code changes (HIGH PRIORITY)
- **Additional Indicators** - MACD, Bollinger Bands, Volume Profile
- **Transformer Models** - Alternative to LNN
- **Ensemble Predictions** - Combine multiple models

**System:**
- **Multi-Stock Portfolio** - Monitor multiple stocks simultaneously
- **Paper Trading** - Simulated trades with P&L tracking
- **Real-time Feeds** - WebSocket integration
- **Mobile App** - React Native companion app
- **Cloud Deployment** - Docker + AWS/GCP hosting

**Analytics:**
- **Performance Dashboard** - Win rate, R/R, drawdown metrics
- **Advanced Backtesting** - Walk-forward optimization
- **A/B Testing** - Compare model versions
- **Explainability** - SHAP values for predictions

---

## Quick Reference

### Common Commands

```bash
# Stage 1: Dashboard & Alerts
python main.py dashboard                        # Launch GUI
python main.py signal --stock TSLA              # One-time signal
python main.py monitor --interval 30            # Continuous monitoring
python main.py test                             # Test all components

# Stage 2: Data Processing
python3 process_real_events.py                  # Parse real events
python3 validate_data_alignment.py              # Validate data (MANDATORY!)

# Stage 2: Training
python3 train_model_lazy.py \                   # Quick test (10-15 min)
  --tsla_events data/tsla_events_REAL.csv \
  --start_year 2023 --end_year 2023 \
  --epochs 10 --pretrain_epochs 3 \
  --output models/quick.pth

python3 train_model_lazy.py \                   # Full training (60-90 min)
  --tsla_events data/tsla_events_REAL.csv \
  --epochs 50 --pretrain_epochs 10 \
  --output models/full.pth

# Stage 2: Validation & Testing
python3 backtest.py \                           # Backtest model
  --model_path models/full.pth \
  --test_year 2024 \
  --num_simulations 100

python3 validate_results.py \                   # Validate performance
  --model_path models/full.pth

python3 update_model.py \                       # Online learning
  --model_path models/full.pth \
  --output models/updated.pth
```

### Configuration

**Key Settings in `config.py`:**
```python
# Stage 1
MIN_CONFLUENCE_SCORE = 60          # Alert threshold
RSI_OVERSOLD = 30                  # Buy signal
RSI_OVERBOUGHT = 70                # Sell signal
CHANNEL_LOOKBACK_HOURS = 168       # 1 week
USE_LIVE_DATA = True               # Merge with yfinance

# Stage 2
ML_MODEL_TYPE = "LNN"              # LNN or LSTM
ML_BATCH_SIZE = 16                 # Reduced for memory
ML_SEQUENCE_LENGTH = 84            # Half week (3.5 days)
ML_EPOCHS = 50                     # Training epochs
PREDICTION_HORIZON_HOURS = 24      # Forecast 24 hours ahead
```

### Troubleshooting

**"Data file not found"**
```bash
python3 convert_data.py  # Convert raw data first
```

**"Memory error during training"**
```bash
# Use lazy version with smaller batch size
python3 train_model_lazy.py --batch_size 8
```

**"Validation failed"**
```bash
# Check data alignment
python3 validate_data_alignment.py --events_data data/tsla_events_REAL.csv
```

**"Stuck at feature extraction"**
```
This is normal! Takes 3-5 minutes for 1.35M bars.
Progress bars show step-by-step progress.
```

**"Pretraining at 0%"**
```
This is normal! Each epoch has 84,000+ batches.
Batch-level progress bars show activity.
```

---

## Key Formulas

### Linear Regression Channel:
```
center_line = slope × x + intercept
upper_line = center_line + (2 × σ)
lower_line = center_line - (2 × σ)

Where σ = std_dev(residuals)
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

### Composite Score (Intelligent Channel Selection):
```
composite = (confidence × 0.70) + (rsi_confluence × 0.25) + (stability × 0.05)
```

### 24-Hour Projection:
```
bars_24h = {'1hour': 24, '3hour': 8, '4hour': 6, 'daily': 1}
future_x = [n, n+1, ..., n+bars_24h]
predicted_high = MAX(slope × future_x + intercept + 2σ)
predicted_low = MIN(slope × future_x + intercept - 2σ)
```

---

## Version History

### v2.1 - Interactive Parameter Selection (Nov 11, 2025)
✅ Arrow-key navigation menu for parameter configuration
✅ Interactive parameter selection with 3 modes (quick start, arrow-key, number-based)
✅ RAM-aware batch size suggestions based on available memory
✅ Parameter-specific editors with validation
✅ Visual menu with 21 parameters in 6 categories
✅ Auto-device detection and selection
✅ Graceful fallback to number-based menu
✅ InquirerPy integration for terminal UI

### v2.0 - Stage 2 Complete + GPU/Metal Support (Nov 11, 2025)
✅ Liquid Neural Network implementation
✅ 56-feature extraction system
✅ Memory-efficient lazy loading
✅ Real event integration (394 events)
✅ Data validation system
✅ Progress bars and feedback
✅ SQLite prediction database
✅ Walk-forward backtesting
✅ Online learning system
✅ GPU/Metal acceleration (M1-M5, CUDA)
✅ Device manager with hardware detection
✅ Cross-device model loading

### v1.0 - Stage 1 Complete (Nov 10, 2025)
✅ Linear regression channels
✅ Multi-timeframe RSI
✅ AI news sentiment + BS scoring
✅ Telegram alerts
✅ Streamlit dashboard
✅ Live data integration
✅ Intelligent channel selection
✅ 24-hour forecasting

---

## Contact & Support

**Repository:** https://github.com/frankywashere/autotrade2 (Private)

**Author:** Built with Claude Code

**License:** Personal trading tool - Use at your own risk. Not financial advice.

---

**End of Specification**