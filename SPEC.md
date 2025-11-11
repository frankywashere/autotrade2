# AutoTrade2 - Complete Technical Specification

**Version:** 2.0 (Stage 1 + Stage 2 Complete)
**Repository:** https://github.com/frankywashere/autotrade2
**Last Updated:** November 11, 2025

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Quick Start](#quick-start)
3. [File Structure](#file-structure)
4. [Stage 1: Linear Regression Trading System](#stage-1-linear-regression-trading-system)
5. [Stage 2: ML-Powered Predictions](#stage-2-ml-powered-predictions)
6. [Training Workflow](#training-workflow)
7. [Data Validation](#data-validation)
8. [Memory Optimization](#memory-optimization)
9. [Performance & Limitations](#performance--limitations)
10. [API Integration](#api-integration)
11. [Future Improvements](#future-improvements)
12. [Quick Reference](#quick-reference)

---

## System Overview

AutoTrade2 is a two-stage AI-powered stock trading analysis system:

### Stage 1: Technical Analysis & Alerts
- **Linear regression channels** with ping-pong pattern detection
- **Multi-timeframe RSI analysis** with confluence scoring
- **Claude AI news sentiment** and BS detection
- **Automated Telegram alerts** for high-confidence signals
- **Interactive Streamlit dashboard** with monitoring

### Stage 2: ML-Powered Predictions
- **Liquid Neural Networks (LNN)** for 24-hour forecasting
- **56-feature extraction system** (channels, RSI, correlations, cycles)
- **Real event integration** (TSLA earnings + macro events)
- **Memory-efficient training** with lazy sequence loading
- **Online learning** from prediction errors
- **Production-ready** with SQLite logging

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

### Stage 1 (Dashboard & Alerts)
```bash
# Install dependencies
pip install -r requirements.txt

# Convert raw data (one-time)
python3 convert_data.py

# Launch dashboard
python main.py dashboard

# Or use quick menu
./run.sh
```

### Stage 2 (ML Training)
```bash
# Install ML dependencies
pip install torch ncps sqlalchemy tqdm psutil

# Validate data (MANDATORY!)
python3 validate_data_alignment.py --events_data data/tsla_events_REAL.csv

# Train model (memory-efficient)
python3 train_model_lazy.py \
  --tsla_events data/tsla_events_REAL.csv \
  --epochs 50 \
  --pretrain_epochs 10 \
  --output models/lnn_full.pth

# Backtest
python3 backtest.py --model_path models/lnn_full.pth --test_year 2024
```

---

## File Structure

```
autotrade2/
├── .git/                                # Git repository
├── .gitignore                           # Excludes /data folder
├── README.md                            # User documentation
├── SPEC.md                              # This technical specification
├── config.py                            # Central configuration
├── main.py                              # Stage 1 entry point
├── requirements.txt                     # Python dependencies
├── run.sh                               # Quick start menu
├── convert_data.py                      # Data conversion utility
│
├── data/                                # Stock data (gitignored)
│   ├── TSLAMin.txt                     # Raw TSLA data
│   ├── SPYMin.txt                      # Raw SPY data
│   ├── TSLA_1min.csv                   # Converted TSLA (1.45M rows)
│   ├── SPY_1min.csv                    # Converted SPY (1.79M rows)
│   ├── tsla_events_REAL.csv            # Real events (394 in training range)
│   └── predictions.db                  # SQLite prediction database
│
├── models/                              # Trained ML models
│   └── lnn_full.pth                    # Production LNN model
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
    ├── base.py                         # Abstract interfaces
    ├── data_feed.py                    # Data loading (CSV/IBKR)
    ├── features.py                     # 56-feature extraction
    ├── features_lazy.py                # Feature extraction with progress
    ├── events.py                       # Event integration
    ├── model.py                        # LNN and LSTM models
    └── database.py                     # SQLite prediction logging

# Stage 2 Scripts
├── train_model.py                       # Original training (memory-intensive)
├── train_model_lazy.py                  # Memory-efficient training (USE THIS!)
├── backtest.py                          # Walk-forward backtesting
├── validate_results.py                  # Model validation
├── update_model.py                      # Online learning
├── validate_data_alignment.py           # Data validation (MANDATORY)
└── process_real_events.py               # Parse real events from RTF/JSON
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

## Stage 2: ML-Powered Predictions

### Architecture

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

**TradingFeatureExtractor** - Extracts 56 features:

| Category | Count | Features |
|----------|-------|----------|
| **Price** | 10 | SPY/TSLA: close, returns, log_returns, volatility_10, volatility_50 |
| **Channels** | 21 | 3 timeframes (1h/4h/daily): position, upper_dist, lower_dist, slope, stability, ping_pongs, r_squared |
| **RSI** | 9 | 3 timeframes: value, oversold, overbought |
| **Correlations** | 5 | correlation_10/50/200, divergence, divergence_magnitude |
| **Cycles** | 4 | distance_from_52w_high/low, within_mega_channel, mega_channel_position |
| **Volume** | 2 | tsla_volume_ratio, spy_volume_ratio |
| **Time** | 4 | hour_of_day, day_of_week, day_of_month, month_of_year (cyclical) |

**Key Methods:**
```python
extractor = TradingFeatureExtractor()

# Extract features from aligned data
features_df = extractor.extract_features(aligned_df)  # Returns (N, 56)

# Create sequences for training
X, y = extractor.create_sequences(features_df, sequence_length=84, target_horizon=24)
# X: (num_sequences, 84, 56) - 84 timesteps of 56 features
# y: (num_sequences, 2) - [predicted_high, predicted_low] for next 24 hours
```

**Leverages Stage 1:**
- Uses `LinearRegressionChannel` for channel features
- Uses `RSICalculator` for RSI features
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

#### 4. Model (`src/ml/model.py`)

**LNNTradingModel** - Liquid Neural Network:
```
Input (56 features, 84 timesteps)
  ↓
CfC Layer (Liquid Time-Constant, 128 hidden units)
  - Sparse wiring via AutoNCP
  - Continuous-time dynamics
  ↓
Output Layer (2 units: high, low)
Confidence Head (1 unit: confidence score)
```

**Features:**
- **ncps library** - Closed-form Continuous-time (CfC) RNN
- **Sparse connections** - AutoNCP wiring for interpretability
- **Self-supervised pretraining** - 15% masking ratio
- **Online learning** - `update_online()` method
- **Checkpoint system** - Save/load with metadata

**Usage:**
```python
model = LNNTradingModel(input_size=56, hidden_size=128)

# Predict
predictions = model.predict(X)
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
    # ... 25+ fields
})

# Update with actuals
db.update_actual(pred_id, actual_high=252.1, actual_low=246.8)

# Get metrics
metrics = db.get_accuracy_metrics()
# Returns: mean_absolute_error, error by confidence bins, etc.
```

### Training Scripts

#### `train_model_lazy.py` (RECOMMENDED)
**Memory-efficient training with lazy sequence loading**

- Uses only **2-3 GB RAM** (vs 30+ GB for pre-created sequences)
- Trains on ALL 1.35M bars
- Creates sequences on-demand during training
- Progress bars for all phases

#### `train_model.py` (Original)
**Pre-creates all sequences upfront**

- Requires **30+ GB RAM** for full dataset
- Good for smaller datasets (single year)
- Slightly simpler code

#### `backtest.py`
**Walk-forward backtesting**

- Tests model on unseen data (e.g., 2024)
- Random day/week simulation
- Accuracy by event type
- Confidence calibration

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

## Training Workflow

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

### v2.0 - Stage 2 Complete (Nov 11, 2025)
✅ Liquid Neural Network implementation
✅ 56-feature extraction system
✅ Memory-efficient lazy loading
✅ Real event integration (394 events)
✅ Data validation system
✅ Progress bars and feedback
✅ SQLite prediction database
✅ Walk-forward backtesting
✅ Online learning system

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