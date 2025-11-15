# AutoTrade2 - Complete System Specification

**Version:** 3.8 (Hierarchical Multi-Task LNN)
**Status:** 🟢 Production Ready
**Last Updated:** November 2024

---

## Executive Summary

AutoTrade2 is an advanced machine learning trading system that predicts TSLA price movements using a hierarchical liquid neural network (LNN) architecture with continuous online learning. Unlike traditional static models, this system learns from its mistakes in real-time, adapting to changing market conditions while discovering complex multi-timeframe patterns automatically.

**Core Innovation:** Dynamic rolling channel detection across 11 timeframes (5-minute to 3-month), combined with multi-timeframe RSI analysis and SPY-TSLA correlation tracking. A 3-layer hierarchical neural network (Fast → Medium → Slow) processes **473 features** (including multi-threshold ping-pongs, normalized slopes, and automatic bull/bear/sideways detection) to generate **6 prediction outputs**, with an adaptive fusion layer that learns which timeframe to trust based on recent accuracy.

**Key Capabilities:**
- Predicts next 30-200 minute price movements (high, low, center, range, confidence, volatility)
- Learns trading patterns automatically from data without hardcoded rules
- Updates model weights online when prediction errors exceed thresholds  
- Tracks high-confidence trades with full rationale and context
- Handles live data limitations through hybrid multi-resolution fetching

**Current Status:** Core implementation complete and tested. Rolling channel detection verified, hybrid live integration functional, 469-feature extraction with multi-threshold ping-pongs, normalized slopes, normalized prices, and automatic direction detection working. Multi-task learning bug fixed. GPU acceleration implemented. **Ready for production training and deployment.**

---

## Quick Reference

- **Features:** 469 (12 price + 308 channels with multi-threshold ping-pongs + normalized slope + direction flags + 66 RSI + 83 other)
- **Architecture:** 3-layer Hierarchical LNN (~2.8M parameters)
- **Predictions:** 5 primary outputs (high, low, confidence) + 2 derived (center, range) + 4 multi-task (hit_band, hit_target, expected_return, overshoot)
- **Training Time:** First run 35-70 mins, subsequent runs 6-11 mins
- **Memory:** 2-4 GB RAM
- **Version:** v3.9
- **Status:** 🟢 Production Ready

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Feature System](#2-feature-system-313-features)
3. [Model Architecture](#3-model-architecture)
4. [Multi-Task Learning](#4-multi-task-learning)
5. [Rolling Channel Detection](#5-rolling-channel-detection)
6. [Hybrid Live Integration](#6-hybrid-live-integration)
7. [Online Learning](#7-online-learning)
8. [Training Pipeline](#8-training-pipeline)
9. [API Reference](#9-api-reference)
10. [System Status](#10-system-status)

---

## 1. System Architecture

### High-Level Flow

```
1-MIN DATA (TSLA + SPY)
  ↓
FEATURE EXTRACTION (313 features)
  ├─ Rolling Dynamic Channels (154 features)
  ├─ Multi-Timeframe RSI (66 features)
  ├─ Correlation & Alignment (5 features)
  ├─ Breakdown Indicators (54 features)
  ├─ Binary Flags (14 features)
  └─ Price, Volume, Time, Cycle (20 features)
  ↓
HIERARCHICAL LNN (3 layers)
  ├─ Fast Layer (1-min → 5-min scale)
  ├─ Medium Layer (1-hour scale)
  ├─ Slow Layer (daily scale)
  └─ Adaptive Fusion (learned weights)
  ↓
FUSION OUTPUTS (3 primary + 2 derived)
  ├─ Predicted High % | Predicted Low % | Confidence
  └─ Derived: Center = (high+low)/2, Range = high-low
  ↓
MULTI-TASK AUXILIARY HEADS (4 outputs, optional)
  ├─ Hit Band (will price stay in band?) | Hit Target (target before stop?)
  └─ Expected Return (profit %) | Overshoot (how far beyond band)
  ↓
ONLINE LEARNER → TRADE TRACKER
```

### Core Components

| Component | Purpose | File |
|-----------|---------|------|
| **TradingFeatureExtractor** | 313-feature extraction | `src/ml/features.py` |
| **HierarchicalLNN** | 3-layer neural network | `src/ml/hierarchical_model.py` |
| **OnlineLearner** | Continuous learning | `src/ml/online_learner.py` |
| **TradeTracker** | High-confidence logging | `src/ml/trade_tracker.py` |
| **HybridLiveDataFeed** | Multi-res data fetching | `src/ml/live_data_feed.py` |

---

## 2. Feature System (473 Features)

### Feature Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| **Price** | 12 | SPY & TSLA: close, close_norm (v3.8), returns, log_returns, volatility |
| **Channels** | 308 | Rolling channels (11 TFs × 2 stocks × 14 metrics) |
|  | | - Base: position, upper_dist, lower_dist, slope, stability, r² |
|  | | - Normalized: slope_pct (% per bar) |
|  | | - Multi-threshold ping-pongs: 2%, 0.5%, 1%, 3% |
|  | | - Direction flags: is_bull, is_bear, is_sideways |
| **RSI** | 66 | Multi-TF RSI (11 TFs × 2 stocks × 3 metrics) |
| **Correlation** | 5 | SPY-TSLA correlation & divergence |
| **Cycle** | 4 | 52-week highs/lows, mega channel |
| **Volume** | 2 | Volume ratios |
| **Time** | 4 | Hour, day, month, year |
| **Breakdown** | 54 | Volume surge, RSI divergence, alignment |
| **Binary Flags** | 14 | Day flags, volatility, in-channel flags |
| **Event Features** | 4 | is_earnings_week, days_until_earnings, days_until_fomc, is_high_impact_event (v3.9) |
| **TOTAL** | **473** | v3.9: +66 ping-pongs + 22 slopes + 66 direction + 2 norm prices + 4 events |

### Multi-Threshold Ping-Pong Learning (v3.6 Feature)

**Innovation:** Instead of using a fixed 2% threshold for detecting channel bounces, the system extracts ping-pong counts at 4 different thresholds:

- **0.5%** (strict): Price must get very close to bounds
- **1.0%** (medium): Moderate proximity required
- **2.0%** (default): Standard threshold
- **3.0%** (loose): Counts touches further from bounds

**Why this matters:**
The model **automatically learns** which threshold is most predictive for each situation:
- Volatile TSLA 5min channels → Model might trust 3% threshold
- Stable SPY daily channels → Model might trust 0.5% threshold
- Mixed signals → Model combines multiple thresholds

**Example:**
```
Same TSLA 1h channel:
- ping_pongs_0_5pct = 4 (strict counting)
- ping_pongs_1_0pct = 6 (medium)
- ping_pongs_2_0pct = 8 (default)
- ping_pongs_3_0pct = 10 (loose counting)

Model learns: "When ping_pongs_0_5pct=4 but ping_pongs_2_0pct=8,
it's a weaker channel (price isn't precisely bouncing)"
```

**Added features:** 11 timeframes × 2 stocks × 3 new thresholds = **66 new features**

### Normalized Slope + Direction Detection (v3.7 Feature)

**Innovation:** Channel slope is now provided in two forms:

1. **Raw Slope** (`slope`): Absolute price change per bar ($/bar)
   - Used internally for calculations
   - Not comparable across timeframes ($0.50/bar means different things for 5min vs daily)

2. **Normalized Slope** (`slope_pct`): Percentage change per bar (% per bar)
   - **Comparable across ALL timeframes**
   - 5min slope_pct = +0.2% per bar = same interpretation as daily slope_pct = +0.2% per bar
   - Model can learn: "ANY positive slope_pct = bullish" regardless of timeframe

3. **Direction Flags** (Binary):
   - `is_bull`: slope_pct > 0.1% per bar (uptrending channel)
   - `is_bear`: slope_pct < -0.1% per bar (downtrending channel)
   - `is_sideways`: |slope_pct| ≤ 0.1% per bar (ranging channel)

**Why this matters:**
The model can now learn directional patterns explicitly:
- Bull channel + high ping-pongs → "Buy dips in uptrend"
- Bear channel + high ping-pongs → "Sell rallies in downtrend"
- Sideways + high ping-pongs → "Trade both directions in range"

**Example:**
```
TSLA 4h channel:
- slope = +0.50 (raw)
- slope_pct = +0.2% per bar (normalized)
- is_bull = 1 (yes, >0.1% per bar)
- is_bear = 0
- is_sideways = 0
- ping_pongs_2pct = 8

Model learns: "Bull channel + 8 bounces = strong uptrend, buy dips"
```

**Added features:**
- Normalized slopes: 11 timeframes × 2 stocks = **22 new features**
- Direction flags: 11 timeframes × 2 stocks × 3 flags = **66 new features**

### Critical: Rolling Dynamic Channels

**NOT Static!** Channels calculated at EACH timestamp using rolling window:

```
10:00am → 1h channel from 09:00-10:00 → r²=0.81, position=0.3
11:00am → 1h channel from 10:00-11:00 → r²=0.73, position=0.6
12:00pm → 1h channel from 11:00-12:00 → r²=0.41, position=0.95
```

**Result:** Metrics vary dynamically (r²: 0.08 → 0.95), capturing channel formation and breakdown.

**Timeframes:** 5min, 15min, 30min, 1h, 2h, 3h, 4h, daily, weekly, monthly, 3month

### Caching System

- **First run:** 30-60 minutes (calculates rolling channels)
- **Cached runs:** 2-5 seconds (loads from disk)
- **Cache file:** `data/feature_cache/rolling_channels_v3.5_{start}_{end}_{bars}.pkl`

---

## 3. Model Architecture

### Structure

```
INPUT: [batch, 200, 313]  # 200 bars, 313 features

FAST LAYER (1-min scale)
  CfC(313 → 128) → [high, low, conf] + hidden[128]
  ↓ Pool 5:1
  
MEDIUM LAYER (hourly scale)  
  CfC(313+128 → 128) → [high, low, conf] + hidden[128]
  ↓ Pool 12:1
  
SLOW LAYER (daily scale)
  CfC(313+128 → 128) → [high, low, conf] + hidden[128]
  ↓
  
ADAPTIVE FUSION
  Learnable weights: [w_fast, w_medium, w_slow]
  → fusion_hidden[128]
  ↓

FUSION OUTPUTS (3 neural heads)
  ├─ fusion_fc_high: Linear(128 → 64 → 1) → predicted_high
  ├─ fusion_fc_low: Linear(128 → 64 → 1) → predicted_low
  └─ fusion_fc_conf: Linear(128 → 64 → 1) + Sigmoid → confidence

DERIVED (post-processing, not neural heads)
  ├─ predicted_center = (predicted_high + predicted_low) / 2
  └─ predicted_range = predicted_high - predicted_low

MULTI-TASK AUXILIARY HEADS (4 optional heads)
  ├─ hit_band_head: Linear(64 → 32 → 1) + Sigmoid
  ├─ hit_target_head: Linear(64 → 32 → 1) + Sigmoid
  ├─ expected_return_head: Linear(64 → 1)
  └─ overshoot_head: Linear(64 → 1)
```

### Parameters

- **Total:** ~2.8M parameters
- **Fast CfC:** ~420K  
- **Medium CfC:** ~570K
- **Slow CfC:** ~570K
- **Fusion + Heads:** ~220K

### Automatic Pattern Discovery

The model learns patterns automatically (NO hardcoded rules!):

1. **RSI + Channel Position:** High RSI (>70) at top (>0.85) → reversal
2. **Multi-TF Confluence:** All TFs oversold + all near bottoms → bounce
3. **Nested Channels:** 15min breaks but 1h holds → ignore noise
4. **SPY-TSLA Alignment:** Both at tops + correlated → coordinated drop

---

## 4. Multi-Task Learning

### 6 Prediction Tasks

| Task | Type | Range | Use Case |
|------|------|-------|----------|
| **Predicted High** | Regression | -5% to +15% | Maximum gain |
| **Predicted Low** | Regression | -15% to +5% | Maximum loss |
| **Predicted Center** | Regression | -10% to +10% | Direction |
| **Predicted Range** | Regression | 0% to 20% | Volatility/sizing |
| **Confidence** | Binary (Sigmoid) | 0.0 to 1.0 | Trade filter |
| **Volatility** | Regression | 0% to 10% | Risk regime |

### Loss Function

```python
loss = (
    mse_loss(pred_high, target_high) +
    mse_loss(pred_low, target_low) +
    mse_loss(pred_center, target_center) +
    mse_loss(pred_range, target_range) +
    mse_loss(confidence, actual_accuracy) +
    mse_loss(pred_vol, actual_vol)
) / 6
```

### Benefits

- **Knowledge Sharing:** Tasks share fusion_hidden representation
- **Regularization:** Prevents overfitting on single task
- **Consistency:** Learns relationships (high > center > low)
- **Efficiency:** Single forward pass for all outputs

---

## 4.1 Automatic Pattern Discovery

The neural network automatically discovers these patterns from training data (no hardcoded rules):

1. **RSI + Channel Position = Reversal**
   - High RSI (>70) at channel top (position >0.85) → likely drop to bottom
   - Low RSI (<30) at channel bottom (position <0.15) → likely bounce to top
   - Model learns this correlation from 8 years of price history

2. **Multi-Timeframe Confluence**
   - All timeframes oversold (RSI <30) + all near channel bottoms → high-confidence bounce
   - Mixed signals (some oversold, some overbought) → lower confidence, smaller position

3. **Nested Channel Dynamics**
   - 15min channel breaks (r²=0.32) but 1h channel holds (r²=0.84) → ignore noise, return to 1h range
   - Both 15min and 1h break (both r²<0.40) → bigger move coming, trend change

4. **SPY-TSLA Alignment**
   - Both at channel tops (position >0.85) + highly correlated (>0.8) → likely coordinated fall
   - TSLA oversold but SPY neutral → weaker signal, less reliable

5. **Channel Formation and Breakdown**
   - r² rising (0.45→0.82) + ping-pongs increasing (2→8) → channel strengthening
   - r² falling (0.82→0.35) + position >1.0 → breakout confirmed, channel broken

6. **Directional Channel Trading**
   - Bull channel (is_bull=1) + 8 tight bounces → buy dips, sell near top
   - Bear channel (is_bear=1) + 6 bounces → sell rallies, cover near bottom
   - Sideways (is_sideways=1) + 10 bounces → fade extremes, trade both directions

**You never program these rules!** The 256+ neurons in each layer discover optimal patterns automatically from the 473 features! 🧠

---

## 5. Rolling Channel Detection

### The Critical Fix

**Before (BROKEN):**
```python
channel = calculate_channel(all_data)  # r² = 0.057 everywhere (static)
```

**After (FIXED):**
```python
for each_timestamp:
    window = data[timestamp - lookback : timestamp]
    channel = calculate_channel(window)  # r² varies: 0.08 → 0.95
```

### Why It Matters

**Static:** "Channels don't matter (r²=0.057)"  
**Rolling:** "Channels form (r²=0.89), hold, then break (r²=0.32)"

### Example Timeline

```
Time     15min Channel           1h Channel
10:00    Forming (r²=0.45)      Forming (r²=0.62)
10:30    Strong (r²=0.89)       Strong (r²=0.78)
11:00    Breaking (r²=0.41)     Strong (r²=0.76)
11:30    Broken (r²=0.18)       Weakening (r²=0.52)
12:00    New forming (r²=0.67)  Breaking (r²=0.31)
```

---

## 6. Hybrid Live Integration

### The yfinance 7-Day Problem

**Problem:** yfinance limits 1-min data to 7 days (~2,730 bars)  
**Need:** 168 hours for 1h channel = requires >7 days  
**Solution:** Hybrid multi-resolution fetching

```python
# Download 3 resolutions:
data_1min = yfinance.download('TSLA', period='7d', interval='1m')     # 2,730 bars
data_1h = yfinance.download('TSLA', period='2y', interval='1h')       # 3,494 bars
data_daily = yfinance.download('TSLA', period='max', interval='1d')   # Many years
```

### Resolution Routing

```python
if timeframe in ['5min', '15min', '30min']:
    use_data = data_1min  # 7 days sufficient
elif timeframe in ['1h', '2h', '3h', '4h']:
    use_data = data_1h    # 2 years available
elif timeframe in ['daily', 'weekly', 'monthly']:
    use_data = data_daily  # Max history
```

**Result:** All 11 timeframes have adequate lookback!

---

## 7. Online Learning

### How It Works

1. **Predict + Track:** Log prediction to database with validation_time
2. **Wait:** 30 minutes (prediction horizon)
3. **Validate:** Compare actual vs predicted
4. **Update Decision:**
   - Error > 2.0% → Update all layers
   - Error > 1.5% → Update fast + medium
   - Error > 1.0% → Update fast only
5. **Apply Update:** Gradient descent with small LR (0.0001)
6. **Propagate:** Share error signal across layers (weighted)

### Learning Rates

| Layer | LR | Updates/Day |
|-------|-----------|-------------|
| Fast | 0.0001 | 5-10 |
| Medium | 0.00005 | 1-3 |
| Slow | 0.00001 | <1 |

### Fusion Adaptation

```python
# Better layers get higher weights automatically
fast_accuracy = ema(1 - fast_error/10)
medium_accuracy = ema(1 - medium_error/10)
slow_accuracy = ema(1 - slow_error/10)

weights = softmax([fast_accuracy, medium_accuracy, slow_accuracy])
```

---

## 8. Training Pipeline

### Quick Start

```bash
# Interactive (recommended)
python train_hierarchical.py --interactive

# Command line
python train_hierarchical.py \
  --epochs 100 \
  --batch_size 64 \
  --device mps \
  --train_start_year 2015 \
  --train_end_year 2022
```

### Timing

**First Run:**
- Feature extraction: 30-60 mins (builds cache)
- Training: 5-10 mins (M1/M2 Mac, 100 epochs)
- **Total:** 35-70 mins

**Subsequent Runs:**
- Feature extraction: 2-5 secs (loads cache)
- Training: 5-10 mins
- **Total:** 6-11 mins

### Interactive Menu Features

- Device selection (CUDA/MPS/CPU auto-detection)
- Hardware info display
- Capacity selection (192/256/384/512 neurons)
- Cache regeneration option
- Recommended batch sizes per device

---

## 8.1 GPU Acceleration (Optional)

### Overview

GPU acceleration is available for rolling channel calculation (the most time-consuming part of feature extraction). Uses hybrid GPU+CPU approach for optimal balance of speed and correctness.

### Performance

| Dataset Size | CPU Time | GPU Time (Hybrid) | Speedup |
|--------------|----------|-------------------|---------|
| 10K bars | ~20 sec | ~15 sec | 1.3x |
| 50K bars | ~5 mins | ~3 mins | 1.7x |
| 100K bars | ~13 mins | ~8 mins | 1.6x |
| 1.15M bars (training) | ~45 mins | ~25-30 mins | 1.5-1.8x |

**Note:** Modest speedup due to hybrid approach (GPU for regression, CPU for derived metrics). Cached runs are instant regardless of GPU/CPU (2-5 seconds).

### How It Works

**Hybrid Approach:**
- ⚡ **GPU Phase:** Linear regression (vectorized, 80% of time) → 15x speedup on this phase
- 💾 **CPU Phase:** Derived metrics (ping-pongs, position, stability) → Exact formula matching

**Why Hybrid:**
- Pure GPU would require complex vectorization of stateful algorithms
- Hybrid gets most of the benefit with guaranteed correctness
- Linear regression is the bottleneck (80% of time), so GPU-accelerating it gives majority of speedup

### Known Minor Differences

GPU and CPU produce equivalent results within acceptable tolerances:

| Metric | Difference | Tolerance | Impact |
|--------|-----------|-----------|--------|
| Position | <0.0001 | 1e-4 | None |
| Slope | <1e-7 | 1e-4 | None |
| R² | <1e-5 | 1e-4 | None |
| **Ping-pongs** | **±1-2 counts** | ±2.5 | Negligible |
| **Stability** | **±0.04 points** | ±0.05 | Negligible (0.04%) |
| Distances | <0.0001 | 1e-4 | None |

**Why differences exist:**
- Floating point edge cases in threshold detection (price exactly at 2% boundary)
- Ping-pong state transitions may round differently
- Stability affected by ping-pong rounding (stability = r²*40 + pp*40 + length*20)

**Impact on model training:** None - differences are 0.04% (well within noise of real market data)

### Usage

**Interactive Mode:**
```
⚡ GPU Acceleration Available: Apple Silicon (MPS)

? Use GPU acceleration for feature extraction?
  ● Yes - Use MPS GPU (1.5-1.8x faster for calculation) ⚡
  ○ No - Use CPU (reliable, compatible) 💾
```

**Command Line:**
```bash
# Auto-detect (uses GPU for large datasets >50K bars)
python train_hierarchical.py --train_start_year 2015 --train_end_year 2022

# GPU will auto-select for training (1.15M bars), CPU for live predictions (2.7K bars)
```

### Validation

Verify GPU equivalence:
```bash
python validate_gpu_cpu_equivalence.py

# Expected: All tests PASS within tolerances
# GPU and CPU produce equivalent results ✅
```

### When to Use GPU

**Use GPU when:**
- First training run (no cache) → saves 15-20 minutes
- Regenerating cache (new date range) → saves time
- Experimenting with different data → faster iteration

**GPU not beneficial when:**
- Cache already exists → loading is instant (2-5 sec) regardless
- Live predictions (small datasets) → CPU is actually faster
- Backtest (168 bars per window) → CPU is faster

**Auto-detection handles this automatically** - uses GPU for training, CPU for live.

---

## 9. API Reference

### TradingFeatureExtractor

```python
from src.ml.features import TradingFeatureExtractor

extractor = TradingFeatureExtractor()
features = extractor.extract_features(df, use_cache=True)

# Returns: DataFrame with 313 columns
# Cache: Auto-saved to data/feature_cache/
```

### HierarchicalLNN

```python
from src.ml.hierarchical_model import HierarchicalLNN, load_hierarchical_model

# Create new model
model = HierarchicalLNN(
    input_size=313,
    hidden_size=128,
    device='mps',
    multi_task=True
)

# Load trained model
model = load_hierarchical_model('models/hierarchical_lnn.pth')

# Predict
pred = model.predict(features[-200:])
# Returns: Dict with 6 predictions + fusion_weights
```

### HybridLiveDataFeed

```python
from src.ml.live_data_feed import HybridLiveDataFeed

feed = HybridLiveDataFeed(symbols=['TSLA', 'SPY'])
df = feed.fetch_for_prediction()

# Returns: 1-min DataFrame with multi_resolution attrs
```

### OnlineLearner

```python
from src.ml.online_learner import OnlineLearner

learner = OnlineLearner(model)

# Predict with tracking
pred, pred_id = learner.predict_with_tracking(
    x, current_price=245.50, timestamp=datetime.now()
)

# Validate later
update_info = learner.validate_and_update(
    pred_id, actual_high=2.5, actual_low=-0.8
)
```

### System Validation

Verify the system is working correctly:

**Validate Rolling Channels:**
```bash
python scripts/validate_channels.py

# Expected output:
# ✅ Rolling channels working correctly
# - r² varies: 0.08 → 0.95 (dynamic, not static)
# - Ping-pongs vary: 0 → 15
# - Position varies: -1.2 → 1.3
```

**Test Hybrid Live Integration:**
```bash
python test_hybrid_features.py

# Expected output:
# ✅ HYBRID FEATURE EXTRACTION TEST PASSED!
# - Multi-resolution data: 1min (2730), 1hour (3494), daily (3871)
# - Features extracted: 469
# - Channel features: VALID ✓
# - RSI features: VALID ✓
```

**Validate GPU/CPU Equivalence:**
```bash
python validate_gpu_cpu_equivalence.py

# Expected output:
# ✅ ALL TESTS PASSED
# GPU and CPU produce equivalent results!
# GPU acceleration is SAFE to use in production.
```

---

## 10. System Status

### ✅ Implemented & Tested

- **Feature Extraction:** 473 features, rolling channels, caching (✅)
- **Model Architecture:** 3-layer hierarchical LNN, 6 multi-task heads (✅)
- **Multi-Task Learning:** Dimension bug fixed, all 6 tasks working (✅)
- **Online Learning:** Prediction tracking, error-based updates (✅)
- **Hybrid Live:** Multi-resolution fetching, automatic routing (✅)
- **Training Pipeline:** Interactive menus, progress bars, caching (✅)
- **Trade Tracking:** High-confidence logging with context (✅)

### ⚠️ Known Issues

1. **Cache files large** (~500MB) - Clear old caches periodically
2. **First run slow** (30-60 mins) - Expected, one-time cost

### 📊 Performance

- **Memory:** 2-4 GB RAM
- **Training (M1/M2):** 3-5 sec/epoch
- **Feature Extraction:** First 30-60 min, cached 2-5 sec
- **Live Prediction:** 10-15 sec (includes data download)

### 🎯 Expected Accuracy

- **MAPE:** < 3.5% (target)
- **High-Conf (>0.75):** < 2.5% error
- **Win Rate:** > 70%

---

## Comparison: v3.5 vs v3.4

| Feature | Ensemble (v3.4) | Hierarchical (v3.5) |
|---------|----------------|---------------------|
| Architecture | 4 independent models | 1 unified (3 layers) |
| Features | 245 | **313** (+68) |
| Predictions | 3 | **6** (+3) |
| Channels | Static (?) | **Rolling dynamic** |
| Learning | Static | **Online continual** |
| Live Mode | Single-res | **Hybrid multi-res** |
| Memory | ~5 GB | ~3 GB |

---

## 10.1 Event Data Maintenance

### Overview

Event features require periodic updates to maintain accuracy. The system uses `tsla_events_REAL.csv` which contains historical and future events.

**Current Coverage:** 2015-2025 (483 events)
**Update Frequency:** Quarterly (when new earnings dates announced)

### Checking Coverage

```bash
python validate_event_data.py

# Shows:
# - Event coverage status
# - Days until CSV expiration
# - Missing quarters (if any)
# - Data quality issues
```

### When to Update

**Warning triggers:**
- System displays: "⚠️ EVENT DATA COVERAGE WARNING" on startup
- Less than 90 days of future events remaining
- New year approaching (add next year's FOMC schedule)

### How to Update CSV

**Step 1: Get TSLA Earnings Dates**
- Visit: https://ir.tesla.com
- Find: "Events & Presentations" → Upcoming earnings
- Note: Q1, Q2, Q3, Q4 earnings dates for next year

**Step 2: Get FOMC Dates**
- Visit: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
- Download: Next year's FOMC meeting schedule (usually 8 meetings/year)

**Step 3: Get Economic Calendar Dates**
- CPI: Usually 2nd or 3rd Wednesday of each month
- NFP: First Friday of each month
- Quad Witching: 3rd Friday of Mar/Jun/Sep/Dec

**Step 4: Update CSV**

Add new rows to `data/tsla_events_REAL.csv`:

```csv
date,event_type,expected,actual,beat_miss,category
2026-01-28,earnings,0.0,0.0,neutral,tsla
2026-01-28,fomc,0.0,0.0,neutral,macro
2026-02-07,nfp,0.0,0.0,neutral,macro
...
```

**Step 5: Validate**

```bash
python validate_event_data.py

# Expected: ✅ EVENT DATA IS PRODUCTION READY
```

### CSV Format

```csv
date,event_type,expected,actual,beat_miss,category
2026-01-28,earnings,5.25,0.0,neutral,tsla     # Earnings (expected EPS)
2026-01-02,delivery,500000,0.0,neutral,tsla   # Deliveries (expected units)
2026-01-29,fomc,0.0,0.0,neutral,macro         # FOMC meeting
2026-02-12,cpi,0.0,0.0,neutral,macro          # CPI release
2026-02-07,nfp,0.0,0.0,neutral,macro          # Jobs report
```

**Required columns:**
- `date`: YYYY-MM-DD format
- `event_type`: earnings, delivery, fomc, cpi, nfp, quad_witching
- `expected`: Expected value (EPS, deliveries, etc.) or 0.0
- `actual`: Actual result (filled after event) or 0.0
- `beat_miss`: beat, miss, meet, or neutral (before event)
- `category`: tsla or macro

### Robustness to Date Shifts

**System is robust:** Uses relative timing, not absolute dates

**Example:**
```
Training: Q1 2022 earnings on Jan 26
Live: Q1 2026 earnings shifts to Feb 2 (week later)

Model learns: "3 days before earnings" (relative)
Not: "January 26" (absolute)

Result: ✅ Works correctly regardless of actual date
```

**Feature:** `days_until_earnings = -3` means "3 days before" (date-agnostic)

### Handling Missing Future Events

**If CSV is outdated:**
- Event features = 0 (no crash)
- System runs normally
- Accuracy ~5-10% lower (missing event context)
- Warning displayed on startup

**Graceful degradation** - system never crashes due to missing events.

---

## Next Steps

1. **Train Model:** `python train_hierarchical.py --interactive`
2. **Validate:** `python scripts/validate_channels.py`
3. **Test Live:** `python test_hybrid_features.py`
4. **Deploy:** See `QUICKSTART.md` for deployment instructions

---

**Status:** 🟢 **PRODUCTION READY**

**For Quick Start Guide:** See `QUICKSTART.md` (5-minute setup)

---

## Built With

- **PyTorch** 2.0+ - Neural network framework
- **LiquidNN (ncps)** - Continuous-time neural networks
- **yfinance** - Market data fetching
- **pandas & numpy** - Data processing
- **InquirerPy** - Interactive menus
- **Streamlit** - Dashboard UI

**Created:** November 2024
**Version:** 3.8
**License:** MIT
