# Technical Specification: Adaptive Channel Prediction System v3.15
*Last Updated: November 19, 2025*
*Status: Production Ready - Chunked Extraction, macOS Multiprocessing Fixed*

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Critical Design Decisions](#3-critical-design-decisions)
4. [Technical Implementation](#4-technical-implementation)
5. [Feature Engineering](#5-feature-engineering)
6. [Installation & Setup](#6-installation--setup)
7. [Usage Guide](#7-usage-guide)
8. [Known Issues & Solutions](#8-known-issues--solutions)
9. [Performance Optimization](#9-performance-optimization)
10. [News Integration Roadmap](#10-news-integration-roadmap)
11. [Testing & Validation](#11-testing--validation)
12. [Troubleshooting Guide](#12-troubleshooting-guide)
13. [API Reference](#13-api-reference)
14. [Future Enhancements](#14-future-enhancements)

---

## 1. Executive Summary

AutoTrade2 v3.0 is a sophisticated **Adaptive Channel Prediction System** that dynamically analyzes market structure across multiple timeframes to predict optimal entry/exit points with variable time horizons. The system uses a hierarchical liquid neural network (LNN) that "reads the ocean layers" of market data - from fast intraday ripples to slow macro tides - to make intelligent predictions.

### Core Analogy: Reading the Ocean Layers
Just as an oceanographer studies different water layers to understand currents, our system analyzes market timeframes as "layers":
- **Fast Layer**: Intraday ripples (hours) - RSI warnings and short-term volatility
- **Medium Layer**: Swings (days) - Channel alignment and trend confirmation
- **Slow Layer**: Macro tides (weeks+) - Long-term support/resistance and fundamental drivers

The model dynamically selects the most confident layer and projects forward accordingly, using higher layers as confirmation for longer holds.

### Current State (v3.15 - November 19, 2025)
- ✅ Training pipeline complete and tested
- ✅ **12,936 features** with 21-window multi-OHLC system
- ✅ **Chunked feature extraction with HDF5** - Never OOM, constant 500MB RAM
- ✅ **macOS multiprocessing fixed** - Lazy torch loading + forkserver mode
- ✅ **Parallel processing** with real-time multi-progress bars (8 workers)
- ✅ **OHLC integration** - Separate regressions for high/low/close prices
- ✅ **Rolling statistics optimization** - 45x speedup for channel calculations
- ✅ **Vectorized ping-pong detection** - 10x speedup
- ✅ **MEMORY LEAKS ELIMINATED** - Comprehensive fixes across 6 subsystems
- ✅ **Precision control** - Full float64/float32 toggle via interactive menu
- ✅ **Peak memory: <4 GB during extraction, 8-12 GB during training** - NO SWAP!
- ✅ **Worker cleanup optimized** - Collect results first, cleanup after (240s → 2s)
- ✅ Continuation labels bug FIXED (duplicate resampling issue)
- ✅ Event system with 483 events (2015-2025)
- ✅ Multi-task learning with 12 prediction heads
- ✅ Dashboard fully implemented with layer interplay visualization
- ✅ Dynamic progress messages (show actual 21 windows, not hardcoded 6)
- ❌ News sentiment integration (NEXT PRIORITY - detailed plan included)
- ❌ Hierarchical backtester (to be implemented)

---

## 2. System Architecture

### 2.1 Current File Structure (Post-Cleanup November 2024)
```
autotrade2/
├── train_hierarchical.py           # Main training script with interactive mode
├── hierarchical_dashboard.py       # Streamlit prediction dashboard
├── config.py                       # System configuration
├── config/
│   └── hierarchical_config.yaml   # Model hyperparameters
├── src/
│   ├── ml/
│   │   ├── hierarchical_model.py  # Hierarchical LNN architecture
│   │   ├── hierarchical_dataset.py # Dataset with adaptive targets
│   │   ├── features.py            # Feature extraction (495+ features)
│   │   ├── features_lazy.py       # Lazy feature extraction with progress
│   │   ├── data_feed.py          # CSV data loading and validation
│   │   ├── channel_features.py   # Channel-specific calculations
│   │   ├── events.py             # Event handling and integration
│   │   └── base.py               # Abstract base classes
│   ├── linear_regression.py      # Linear regression channel calculations
│   └── rsi_calculator.py         # RSI analysis and signals
├── data/                         # Data directory
│   ├── TSLA_1min.csv            # Tesla 1-minute OHLC data
│   ├── SPY_1min.csv             # S&P 500 1-minute OHLC data
│   ├── tsla_events_REAL.csv    # Event calendar (483 events)
│   └── feature_cache/           # Cached feature calculations
├── models/                      # Saved model checkpoints
├── deprecated/                  # Legacy code (50+ old files moved here)
└── Technical_Specification_v3.md # This document
```

### 2.2 Component Architecture

```mermaid
graph TD
    A[1-Min Data Feed] --> B[Feature Extractor<br/>495+ Features]
    B --> C[Hierarchical LNN]
    C --> D[Fast Layer<br/>5min-1hour]
    C --> E[Medium Layer<br/>4hour-daily]
    C --> F[Slow Layer<br/>weekly+]
    D --> G[Fusion Module]
    E --> G
    F --> G
    G --> H[Multi-Task Outputs<br/>16 Heads]
    H --> I[Dashboard Display]
```

---

## 3. Critical Design Decisions

### 3.1 Price-Agnostic Architecture

**Problem:** Model trained on 2015 data ($10 TSLA) must work on 2024 data ($250 TSLA)

**Solution Implemented:**
- All channel metrics use **PERCENTAGES** not dollar amounts
  - `upper_dist = (upper_band - price) / price * 100`
  - `ping_pong_threshold = bound_width * 0.02` (2% of channel width)
- Slopes normalized to **% per bar** for cross-timeframe comparison
  - `slope_pct = slope / price * 100`
- Prices normalized to 0-1 yearly range
  - `close_norm = (price - yearly_min) / (yearly_max - yearly_min)`
- All targets in percentages
  - `target_high_pct = (future_high - current_price) / current_price * 100`

**Result:** Model works at ANY price level (98% of features are price-agnostic)

### 3.2 Rolling Channel Detection

**Problem:** Static channels across entire dataset had r²=0.057 (useless)

**Solution:** Calculate new channel at EACH timestamp
```python
for timestamp in data:
    lookback_window = data[timestamp - 168 bars : timestamp]
    channel = fit_linear_regression(lookback_window)
    features = extract_channel_metrics(channel)
```

**Result:** Dynamic r² values from 0.08 to 0.95, captures channel formation/breakdown in real-time

### 3.3 Multi-Threshold Ping-Pongs

**Problem:** Single 2% threshold may not be optimal for all timeframes/conditions

**Solution:** Extract ping-pongs at 4 thresholds simultaneously
- 0.5% - Strict (tight bounces, high-quality channels)
- 1.0% - Medium
- 2.0% - Default
- 3.0% - Loose (distant touches, volatile conditions)

**Learning:** Model automatically learns optimal threshold per context
- "For TSLA 5min volatility > 3%, use 3% threshold"
- "For SPY daily with r² > 0.9, use 0.5% threshold"

### 3.4 Hybrid GPU+CPU Approach

**Problem:** Pure GPU gave 10-20x speedup but formulas didn't match CPU exactly

**Solution:** Split computation
- **GPU (80% of time):** Linear regression calculation (vectorized, 15x speedup)
- **CPU (20% of time):** Derived metrics (exact formula matching)

```python
# GPU: Fast regression
slopes, intercepts = torch_linear_regression_batch(prices_gpu)

# CPU: Exact metrics
for i, (slope, intercept) in enumerate(zip(slopes, intercepts)):
    ping_pongs = calculate_exact_ping_pongs_cpu(prices[i], slope, intercept)
```

**Result:** 1.5-1.8x total speedup with perfect accuracy

### 3.5 Event System Design

**Problem:** Event dates shift year-to-year (earnings, FOMC meetings)

**Solution:** Use RELATIVE timing instead of absolute dates
```python
# Features use relative days
days_until_earnings = (earnings_date - current_date).days  # Can be -14 to +14
is_earnings_week = abs(days_until_earnings) <= 7

# NOT absolute dates
# BAD: features['is_jan_26'] = (date == 'Jan 26')
```

**Result:** Model learns patterns not dates, robust to schedule changes

### 3.6 Continuation Label Strategy

**Problem:** Predicting fixed 24-bar horizon doesn't capture variable trade durations

**Solution:** Multi-timeframe continuation analysis
```python
def generate_continuation_labels():
    # Pull 1h and 4h OHLC chunks
    # Calculate RSI for both timeframes
    # Check slope alignment
    # Score continuation probability:
    score = 0
    score += 1 if rsi_1h < 40  # Room to run
    score += 1 if rsi_4h < 40  # Broader support
    score += 1 if slopes_aligned  # Trend agreement
    score += 2 if strong_channel_support

    # Look ahead for actual continuation
    actual_continuation = measure_actual_move(future_data)
    return score, actual_continuation
```

---

## 4. Technical Implementation

### 4.1 Model Architecture - Hierarchical LNN with 12 Heads

#### Layer Structure
```python
class HierarchicalLNN(nn.Module):
    def __init__(self):
        # Three time-scale layers
        self.fast_layer = LiquidLayer(64 neurons)    # 5min-1hour patterns
        self.medium_layer = LiquidLayer(128 neurons) # 4hour-daily patterns
        self.slow_layer = LiquidLayer(256 neurons)   # weekly+ patterns

        # 12 prediction heads total
        self.heads = {
            # Primary predictions (2 heads)
            'high', 'low',

            # Multi-task auxiliary (10 heads)
            'hit_band', 'hit_target', 'expected_return', 'overshoot',
            'continuation_duration', 'continuation_gain', 'continuation_confidence',
            'price_change_pct', 'horizon_bars_log', 'adaptive_confidence'
        }
```

#### Liquid Neurons
- Learnable time constants for temporal dynamics
- Skip connections between layers
- Attention mechanism for layer weighting
- Dropout: 0.2 for regularization
- Total parameters: ~2.8M

### 4.2 Training Pipeline

#### Multi-Task Loss Function
```python
loss = (
    0.4 * price_loss +           # Main task: predict high/low
    0.2 * volatility_loss +       # Predict range width
    0.2 * continuation_loss +     # Predict trend continuation
    0.1 * confidence_loss +       # Calibrate confidence
    0.1 * auxiliary_losses        # Hit band, expected return, etc.
)
```

#### Training Configuration
- Batch size: 32-64 (memory dependent)
- Learning rate: 0.001 with cosine annealing
- Early stopping: Patience 10 epochs
- Validation split: 10%
- Hardware: Auto-detect GPU/MPS/CPU

---

## 5. Feature Engineering

### 5.1 Complete Feature Breakdown (v3.13: 12,936 Features)

**Feature Explosion:** v3.13 introduces multi-window OHLC system:
- **v3.12:** 495 features (single best window per timeframe)
- **v3.13:** 12,936 features (21 windows × OHLC × quality scores)

#### Price Features (12 total - UNCHANGED)
**Per stock (TSLA, SPY):**
```python
'close', 'close_norm', 'returns', 'log_returns', 'volatility_10', 'volatility_50'
```

#### Channel Features (v3.13: 12,936 total)

**NEW: Multi-Window Architecture**
Instead of picking "best" window, model sees ALL windows with quality scores:

**Windows Tested:** [168, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10]
- **21 windows** for EVERY timeframe
- **Same windows** for all timeframes (consistent)
- **No filtering** - all windows calculated, bad ones scored low

**Per Window Features (28 total):**

**OHLC Regression Slopes (6):**
```python
'{symbol}_channel_{tf}_w{window}_close_slope'       # Trend direction
'{symbol}_channel_{tf}_w{window}_close_slope_pct'   # % per bar
'{symbol}_channel_{tf}_w{window}_high_slope'        # Resistance trend
'{symbol}_channel_{tf}_w{window}_high_slope_pct'
'{symbol}_channel_{tf}_w{window}_low_slope'         # Support trend
'{symbol}_channel_{tf}_w{window}_low_slope_pct'
```

**OHLC R-Squared (4):**
```python
'{symbol}_channel_{tf}_w{window}_close_r_squared'   # Close fit quality
'{symbol}_channel_{tf}_w{window}_high_r_squared'    # High fit quality
'{symbol}_channel_{tf}_w{window}_low_r_squared'     # Low fit quality
'{symbol}_channel_{tf}_w{window}_r_squared_avg'     # Average quality
```

**Position Metrics (3):**
```python
'{symbol}_channel_{tf}_w{window}_position'          # Where price sits in channel (0-1)
'{symbol}_channel_{tf}_w{window}_upper_dist'        # % to upper band
'{symbol}_channel_{tf}_w{window}_lower_dist'        # % to lower band
```

**Channel Structure (3):**
```python
'{symbol}_channel_{tf}_w{window}_channel_width_pct' # Channel width as % of close
'{symbol}_channel_{tf}_w{window}_slope_convergence' # High/low slope divergence
'{symbol}_channel_{tf}_w{window}_stability'         # Composite quality score
```

**Multi-Threshold Ping-Pongs (4):**
```python
'{symbol}_channel_{tf}_w{window}_ping_pongs'        # 2% threshold (default)
'{symbol}_channel_{tf}_w{window}_ping_pongs_0_5pct' # Strict (0.5%)
'{symbol}_channel_{tf}_w{window}_ping_pongs_1_0pct' # Medium (1%)
'{symbol}_channel_{tf}_w{window}_ping_pongs_3_0pct' # Loose (3%)
```

**Direction Flags (3):**
```python
'{symbol}_channel_{tf}_w{window}_is_bull'           # Uptrend (slope_pct > 0.1%)
'{symbol}_channel_{tf}_w{window}_is_bear'           # Downtrend (slope_pct < -0.1%)
'{symbol}_channel_{tf}_w{window}_is_sideways'       # Ranging
```

**Quality Indicators (4 - NEW in v3.13):**
```python
'{symbol}_channel_{tf}_w{window}_quality_score'     # 0-1: (r²*0.7 + ping_pongs*0.3)
'{symbol}_channel_{tf}_w{window}_is_valid'          # 1.0 if ping_pongs >= 3
'{symbol}_channel_{tf}_w{window}_insufficient_data' # 1.0 if window > available bars
'{symbol}_channel_{tf}_w{window}_duration'          # Actual bars in window
```

**Total Channel Features:**
- 28 features/window × 21 windows × 11 timeframes × 2 stocks = **12,936 features**

**Example Feature Names:**
```
tsla_channel_5min_w168_close_slope
tsla_channel_5min_w168_quality_score
tsla_channel_5min_w90_high_slope
spy_channel_daily_w45_insufficient_data
```

**Timeframes:** 5min, 15min, 30min, 1h, 2h, 3h, 4h, daily, weekly, monthly, 3-month

#### RSI Features (66 total = 3 metrics × 11 timeframes × 2 stocks)
```python
'rsi_value'       # 0-100 RSI value
'rsi_oversold'    # Binary flag: RSI < 30
'rsi_overbought'  # Binary flag: RSI > 70
```

#### Correlation Features (5)
```python
'correlation_10'      # 10-bar SPY-TSLA correlation
'correlation_50'      # 50-bar correlation
'correlation_200'     # 200-bar correlation
'divergence'          # Binary: opposite directions
'divergence_magnitude'# Strength of divergence
```

#### Breakdown Features (54)
**Volume Surge:**
```python
'tsla_volume_surge'   # Volume > 2x average
```

**RSI Divergence (4):**
```python
'tsla_rsi_divergence_15min'
'tsla_rsi_divergence_1h'
'tsla_rsi_divergence_4h'
'tsla_rsi_divergence_daily'
```

**Channel Duration (3):**
```python
'tsla_channel_duration_ratio_1h'    # Time in channel / total time
'tsla_channel_duration_ratio_4h'
'tsla_channel_duration_ratio_daily'
```

**SPY-TSLA Alignment (2):**
```python
'channel_alignment_spy_tsla_1h'     # Channels pointing same direction
'channel_alignment_spy_tsla_4h'
```

**Time in Channel (22):**
```python
'{tsla,spy}_time_in_channel_{timeframe}'  # For all 11 timeframes
```

**Enhanced Positions (22):**
```python
'{tsla,spy}_channel_position_norm_{timeframe}'  # Normalized positions
```

#### Cycle Features (4)
```python
'distance_from_52w_high'
'distance_from_52w_low'
'within_mega_channel'     # Long-term channel detection
'mega_channel_position'
```

#### Volume Features (2)
```python
'tsla_volume_ratio'       # Current / average volume
'spy_volume_ratio'
```

#### Time Features (4)
```python
'hour_of_day'            # 0-23
'day_of_week'            # 0-6
'day_of_month'           # 1-31
'month_of_year'          # 1-12
```

#### Binary Flags (14)
```python
'is_monday', 'is_friday'
'is_volatile_now'        # Volatility > threshold
'{tsla,spy}_in_channel_{1h,4h,daily}'  # 6 flags
'is_high_impact_event'   # Within ±3 days of major event
```

#### Event Features (4)
```python
'is_earnings_week'       # Within ±14 days of earnings
'days_until_earnings'    # -14 to +14 (0 = day of)
'days_until_fomc'        # -14 to +14
'is_high_impact_event'   # Major event within 3 days
```

### 5.2 Continuation Labels

Generated using multi-timeframe analysis:
1. Extract 1h and 4h OHLC chunks
2. Calculate RSI for both timeframes
3. Check slope alignment
4. Score continuation probability (0-5)
5. Look ahead to validate actual continuation

---

## 6. Installation & Setup

### 6.1 Environment Setup
```bash
# Clone repository
git clone https://github.com/yourusername/autotrade2.git
cd autotrade2

# Create virtual environment
python -m venv myenv
source myenv/bin/activate  # Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or manually:
pip install torch torchvision pandas numpy streamlit plotly
pip install tqdm psutil pyyaml scikit-learn tables
```

### 6.2 Data Preparation

#### Required Data Structure
```
data/
├── TSLA_1min.csv    # Columns: timestamp,open,high,low,close,volume
├── SPY_1min.csv     # Same format as TSLA
└── tsla_events_REAL.csv  # Optional: date,type,importance,description
```

#### Data Validation
```bash
# Test data loading
python test_data_loading.py

# Output should show:
# ✓ Loaded 2350 rows of TSLA data
# ✓ Loaded 3587 rows of SPY data
```

### 6.3 Configuration

Edit `config.py`:
```python
# Key settings to adjust
DATA_DIR = "data"                    # Path to data files
ML_BATCH_SIZE = 32                   # Reduce if out of memory
ML_TRAIN_START_YEAR = 2015          # Start of training data
ML_TRAIN_END_YEAR = 2022            # End of training data
ML_TEST_YEAR = 2023                 # Validation year

# v3.13: Precision configuration
TRAINING_PRECISION = 'float64'       # 'float64' or 'float32'
CHANNEL_WINDOW_SIZES = [168, 160, 150, ..., 10]  # 21 windows
MIN_DATA_YEARS = 2.5                 # Minimum data for multi-window system

# v3.15: Chunked extraction configuration
USE_CHUNKED_EXTRACTION = None        # None=auto-detect, True=force, False=disable
CHUNK_SIZE_YEARS = 1                 # Process in 1-year chunks
CHUNK_OVERLAP_MONTHS = 6             # Overlap for rolling features
```

#### Precision Configuration (v3.13)

**Centralized Precision Control:**
The system uses a single `TRAINING_PRECISION` setting that controls numerical precision throughout the entire pipeline:

**config.py settings (v3.15 - Lazy Torch Loading):**
```python
TRAINING_PRECISION = 'float64'  # Master control
NUMPY_DTYPE = np.float64        # Auto-derived (don't modify)

# Lazy-loaded to prevent torch import in multiprocessing workers
def get_torch_dtype():          # Call this instead of TORCH_DTYPE
    import torch                # Only imported when actually needed
    return torch.float64
```

**Precision is applied to 87 locations:**
- **NumPy arrays (58)**: Feature storage, result arrays, channel calculations
  - `parallel_channel_extraction.py`: 27 array allocations
  - `features.py`: 31 array allocations (sequential + legacy features)
- **PyTorch tensors (29)**: Input features, target labels, embeddings
  - `hierarchical_dataset.py`: 27 tensor creations
  - `events.py`: 2 news embedding tensors

**How it works:**
```python
# Instead of hardcoded:
results['slope'] = np.zeros(n, dtype=np.float32)  # OLD

# Now centralized:
results['slope'] = np.zeros(n, dtype=config.NUMPY_DTYPE)  # NEW
```

**Changing precision:**
1. **Interactive mode**: Select in menu → updates config automatically
2. **Manual**: Edit `config.py` → set `TRAINING_PRECISION = 'float32'`
3. **Effect**: All 87 locations update automatically

**Recommendation:** Use `float64` for training, quantize to `int8` or `float16` for deployment (4x faster inference)

---

## 7. Usage Guide

### 7.1 Training

#### Interactive Mode (Recommended - v3.13 Enhanced)
```bash
python train_hierarchical.py --interactive
```

**New in v3.13:** Interactive precision selection
- **float64** (8 bytes): Maximum precision, recommended for training
- **float32** (4 bytes): Half memory, standard ML practice

**Selection controls ALL 87 dtype locations:**
- 58 NumPy array allocations (features.py, parallel_channel_extraction.py)
- 29 PyTorch tensor creations (hierarchical_dataset.py, events.py)
- Centralized via `config.TRAINING_PRECISION`

**Memory Impact:**
- float64: 30-40 GB (8 workers), 20-25 GB (4 workers)
- float32: 20-30 GB (8 workers), 15-18 GB (4 workers)

**Interactive Date Range Analysis (v3.14):**
When users select their training dates in interactive mode, the system immediately provides comprehensive data sufficiency analysis:

```
? Training data start year: 2015
? Training data end year: 2022

📅 Training Date Range Analysis:
   Requested: 2015-2022 (7 years)
   Warmup required: 2.5 years (257,400 bars for 21-window system)

   ⚠️  IF your CSV starts at 2015:
       Effective training: 2017.5-2022 (4.5 years)
       → First 2.5 years used for warmup (ensures complete feature history)

   💡 To train from 2015, you need CSV data from 2012.5

   ✓ Good! 4.5 years of quality training data after warmup

If insufficient (<2 years after warmup):
   ⚠️  Warning: Only 1.2 years of usable training data after warmup!
   Continue with 2015-2017 anyway? [y/N]:
```

**Key Analysis Components:**
- **Warmup Period**: 2.5 years required for 21-window multi-OHLC system (257,400 1-min bars)
- **Effective Training**: Years after warmup period (ensures complete feature history)
- **Data Requirements**: Automatic calculation of minimum CSV start date needed
- **Quality Assessment**: Validates sufficient training data for robust model convergence
- **User Guidance**: Clear warnings and recommendations for data sufficiency

#### Quick Test (1 epoch)
```bash
python train_hierarchical.py --epochs 1 --batch_size 32 --device cpu
```

#### Full Training
```bash
python train_hierarchical.py \
    --epochs 100 \
    --batch_size 64 \
    --device auto \
    --lr 0.001 \
    --patience 10 \
    --multi_task
```

#### Memory-Constrained Systems (v3.15 - Chunked Extraction)

**NEW: Chunked Feature Extraction**
For systems with limited RAM (<32GB), use chunked extraction to prevent OOM:

```bash
# Interactive mode (recommended):
python train_hierarchical.py --interactive
# When prompted:
#   "Use chunked feature extraction? ⭐ Recommended for your system"
#   Select: Yes - Process in 1-year chunks (500MB-1GB RAM, ~10% slower)

# Force chunking via CLI:
python train_hierarchical.py --use-chunking

# Disable chunking (if you have 64GB+ RAM):
python train_hierarchical.py --no-chunking
```

**How Chunked Extraction Works:**
- Processes data in 1-year chunks with 6-month overlap
- Uses HDF5 incremental appending (no memory accumulation)
- Each chunk: ~500MB peak RAM
- Final combine: ~3.5GB brief spike (acceptable with swap)

**Auto-Detection (if not specified):**
- Enables chunking if system RAM < 64GB
- Shows clear recommendation in interactive menu
- Based on `psutil.virtual_memory()` detection

**Memory Impact with Chunking:**
- **Extraction peak:** 500MB-1GB constant (regardless of data size!)
- **Training peak (float64):** 8-12 GB
- **Training peak (float32):** 4-6 GB
- **Minimum RAM required:** 16GB (with float32 + chunking)

**Parallel Workers (Additional Control):**
```bash
# Reduce workers further if needed:
python train_hierarchical.py --use-chunking --feature_workers 4  # 4 cores
python train_hierarchical.py --use-chunking --feature_workers 2  # 2 cores (ultra-conservative)
```

#### Training Timeline (v3.13 - Massively Improved)
- **First run:** 40-60 seconds with rolling statistics (was 45-60 minutes!)
- **Subsequent runs:** 2-5 seconds (using cache)
- **Per epoch:** 2-5 minutes depending on hardware

### 7.2 Dashboard

```bash
streamlit run hierarchical_dashboard.py
```
Then open http://localhost:8501 in browser

Features:
- Real-time predictions with confidence scores
- Layer interplay visualization
- Channel overlay on price chart
- Auto-refresh every 30 minutes

### 7.3 Backtesting

```bash
# Not yet implemented for hierarchical model
# TODO: Create backtest_hierarchical.py
python backtest_hierarchical.py --year 2023 --model models/hierarchical_lnn.pth
```

---

## 8. Known Issues & Solutions

### 8.1 Continuation Labels Bug (✅ FIXED in v3.0)
**Issue:** KeyError: 'close' when generating continuation labels
**Cause:** Duplicate resampling code with incorrect column references
**Solution:** Removed duplicate code block at `features.py:1575-1600`
**Status:** ✅ FIXED

### 8.2 Zombie Code Import (✅ FIXED in v3.13)
**Issue:** `ChannelFeatureExtractor` imported but never used
**Cause:** Refactoring debt - logic moved to `LinearRegressionChannel` but import remained
**Impact:** Wasted ~1-2ms per worker creation, unnecessary dependency
**Solution:** Removed imports from `features.py:24, 54` and `parallel_channel_extraction.py:724`
**Status:** ✅ FIXED

### 8.3 Progress Bar Positioning Conflicts (✅ FIXED in v3.13)
**Issue:** Triple-nested progress bars (position=0,1,2) caused terminal corruption/flickering
**Cause:** Multiple tqdm bars with explicit position parameters conflicting
**Symptoms:** Progress bars freeze, overlap, or display garbled text
**Solution:** Removed ALL `position=` parameters, let tqdm handle positioning automatically
**Status:** ✅ FIXED

### 8.4 Silent Fallback to Sequential Mode (✅ FIXED in v3.13)
**Issue:** Parallel mode disabled without notification
**Cause:** Conditions failed (GPU mode, <3 cores, etc.) but user not informed
**Impact:** User expects 7min execution, gets 55min with no explanation
**Solution:** Added notification showing reason and time estimate
**Status:** ✅ FIXED
**Location:** `features.py:519-536`

### 8.5 Worker Process Print Collision (✅ FIXED in v3.13)
**Issue:** Multiple worker processes printing simultaneously caused garbled output
**Example:** `[Worker 123] Proc[Worker 456] Proessing tsla_1hcessing spy_4h`
**Cause:** 8 separate processes writing to stdout without synchronization
**Solution:** Removed worker print statements, use rich progress bars instead
**Status:** ✅ FIXED

### 8.6 DataFrame Column Naming Bug in Parallel Worker (✅ FIXED in v3.13)
**Issue:** Workers crashed silently with KeyError: 'open'
**Cause:** `pd.DataFrame(ohlcv_data)` created columns [0,1,2,3,4] instead of ['open','high','low','close','volume']
**Impact:** Resample aggregation failed, workers returned empty results
**Solution:** Added explicit column names: `pd.DataFrame(ohlcv_data, columns=['open','high','low','close','volume'])`
**Status:** ✅ FIXED
**Location:** `parallel_channel_extraction.py:81-85`

### 8.7 Worker Infinite Loop on Empty Queue (✅ FIXED in v3.13)
**Issue:** Workers never exited after completing tasks, causing `worker.join()` to hang forever
**Cause:** `except: continue` caught queue timeouts indefinitely
**Impact:** Progress bars showed 100% but process hung, never completed
**Solution:** Added `empty_count` tracker - exit after 10 consecutive timeouts (~5 seconds)
**Status:** ✅ FIXED
**Location:** `parallel_channel_extraction.py:49-62`

### 8.8 Result Collection Race Condition (✅ FIXED in v3.13)
**Issue:** Only 5-20 of 22 results collected, causing KeyError when sorting
**Cause:** `while not result_queue.empty()` unreliable in multiprocessing - checked empty too early
**Impact:** Missing results, crashes with KeyError
**Solution:** Use exact-count blocking get: `for i in range(len(tasks)): result_queue.get(timeout=1)`
**Status:** ✅ FIXED
**Location:** `parallel_channel_extraction.py:501-521`

### 8.9 Worker Memory Leak (✅ FIXED in v3.13)
**Issue:** Workers accumulated 2 GB per task, causing OOM crashes after 3-4 tasks
**Cause:** `results` dict from previous tasks never freed between tasks
**Impact:** System crash with "MemoryError" or OS kill
**Solution:** Explicit `del` statements + `gc.collect()` after each task
**Status:** ✅ FIXED
**Location:** `parallel_channel_extraction.py:309-318, 334-344`

### 8.10 Main Process results_dict Accumulation (✅ FIXED in v3.13)
**Issue:** 26 GB of duplicate data in main process after merging results
**Cause:** `results_dict` kept in memory after copying to `all_channel_data`
**Impact:** 52 GB peak memory (26 GB original + 26 GB copy)
**Solution:** `del result_dict` + `gc.collect()` after merge
**Status:** ✅ FIXED
**Location:** `features.py:617-620`

### 8.11 Sequential Path Memory Fragmentation (✅ FIXED in v3.13)
**Issue:** 231 million individual array assignments caused memory fragmentation
**Cause:** `for idx in indices:` loop assigning one value at a time
**Impact:** Memory usage grew exponentially, 30+ GB fragmentation overhead
**Solution:** Vectorized assignments: `array[indices] = value` (all at once)
**Status:** ✅ FIXED
**Location:** `features.py:694-719`

### 8.12 Dashboard Feature Count Issue (⚠️ SUPERSEDED)
**Note:** Now generating 12,936 features (was 495), so previous count mismatch is irrelevant

### 8.13 Memory Issues (✅ FULLY RESOLVED in v3.15 - Nov 19, 2025)

**Previous Issues (v3.13):** Out of memory, swap usage, gradual memory accumulation during training

**Comprehensive Memory Solutions (v3.14-v3.15):**

#### Fix #1: Model Hidden State Accumulation (CRITICAL)
**Location:** `hierarchical_model.py:408-419`
**Issue:** Model cached 5 detached tensors per forward pass (lines 244, 268, 291, 320, 321) that NEVER cleared
**Impact:** 5-10 GB memory leak accumulating over training
**Solution:** New `clear_cached_states()` method called every 10 batches
```python
model.clear_cached_states()  # Clears last_fast_input, last_medium_input, etc.
```

#### Fix #2: Training Loop Tensor Accumulation (CRITICAL)
**Location:** `train_hierarchical.py:266-284`
**Issue:** 15+ tensors created per batch without explicit cleanup
**Impact:** 1.6-3.2 GB gradual accumulation
**Solution:** Periodic cleanup every 10 batches with gc + GPU cache clearing
```python
if batch_idx % 10 == 0:
    del predictions, hidden_states
    torch.cuda.empty_cache() / torch.mps.empty_cache()
    gc.collect()
```

#### Fix #3: GPU Memory Fragmentation (HIGH PRIORITY)
**Location:** `features.py:1446-1458`
**Issue:** Tensors deleted before GPU operations completed (async execution)
**Impact:** 500 MB - 1 GB GPU fragmentation
**Solution:** Synchronize before cleanup
```python
torch.cuda.synchronize()  # Wait for GPU to finish
del windows_batch, regression_results
torch.cuda.empty_cache()
```

#### Fix #4: Hardcoded dtype References (PRECISION CONTROL)
**Locations:** 5 files with hardcoded float32/float64
**Issue:** Mixed precision caused inconsistencies and prevented float32 memory savings
**Solution:** All dtypes now use `config.TORCH_DTYPE` and `config.NUMPY_DTYPE`
- `hierarchical_dataset.py:186` - x_tensor creation
- `features.py:1199` - GPU linear regression
- `features.py:1335` - GPU batch processing
- `features.py:2158-2159` - Continuation labels
- `data_feed.py:91` - CSV loading with explicit dtype

#### Fix #5: Preload Mode OOM Prevention (USER SAFETY)
**Location:** `hierarchical_dataset.py:345-356`
**Issue:** Preload mode tried to allocate 100-200 GB for 12,936 features
**Solution:** Memory estimation + user warning before allocation
```python
if estimated_gb > 50:
    print(f"⚠️  WARNING: Estimated memory usage: {estimated_gb:.1f} GB")
    response = input("Continue with preload? (y/n): ")
```

#### Fix #6: dtype Validation (DATA INTEGRITY)
**Location:** `hierarchical_dataset.py:72-85`
**Issue:** DataFrame dtype could mismatch config, causing silent errors
**Solution:** Automatic validation and conversion with warnings
```python
if self.features_array.dtype != config.NUMPY_DTYPE:
    print(f"⚠️  Converting {self.features_array.dtype} → {config.NUMPY_DTYPE}")
    self.features_array = self.features_array.astype(config.NUMPY_DTYPE)
```

#### Fix #7: Chunked Extraction Memory Accumulation (CRITICAL - v3.15)
**Location:** `features.py:784-914`
**Issue:** Chunked extraction accumulated ALL chunks in memory before combining
**Impact:** Chunk 1: 500MB → Chunk 5: 2.5GB → OOM kill by chunk 5-6
**Solution:** HDF5 incremental appending
```python
# OLD (accumulated in memory):
chunk_results = []
chunk_results.append(chunk1)  # 500MB
chunk_results.append(chunk2)  # 1GB total
# ... crashes at chunk 5

# NEW (HDF5 append):
for chunk in chunks:
    if first:
        chunk.to_hdf(temp_file, mode='w')
    else:
        chunk.to_hdf(temp_file, mode='a', append=True)
    del chunk  # ← Freed immediately!
combined_df = pd.read_hdf(temp_file)  # Load final result
```
**Result:** Constant 500MB RAM per chunk, no accumulation

#### Fix #8: macOS Torch Multiprocessing Deadlock (CRITICAL - v3.15)
**Location:** `config.py:199-210`, `train_hierarchical.py:791-799`
**Issue:** Workers hung for 30s after completing tasks before forced termination
**Cause:** `import torch` at module level → torch background threads deadlock on process exit (spawn mode on macOS)
**Impact:** 240s wasted waiting for 8 workers × 30s each
**Solution:** Lazy torch loading + forkserver mode
```python
# config.py - OLD:
import torch
TORCH_DTYPE = torch.float64

# config.py - NEW (lazy):
def get_torch_dtype():
    import torch  # Only imported when needed
    return torch.float64

# train_hierarchical.py:
mp.set_start_method('forkserver', force=True)  # Safer than spawn on macOS
```
**Result:** Workers exit cleanly, no hangs

#### Fix #9: Worker Cleanup Order (PERFORMANCE - v3.15)
**Location:** `parallel_channel_extraction.py:540-656`
**Issue:** Waited for workers to exit (240s) BEFORE collecting results
**Impact:** Results already in queue, but waited 30s per worker × 8 = 240s wasted
**Solution:** Reordered operations
```python
# OLD order:
1. Wait for workers to exit (240s)
2. Collect results from queue

# NEW order:
1. Collect results immediately (<2s)
2. Terminate workers (don't wait)
```
**Result:** 240s → 2s for cleanup phase

**Memory Requirements After All Fixes (v3.15):**
- **Chunked extraction (any precision):** 500MB-1GB constant (no accumulation!)
- **Training (float64):** 8-12 GB peak (reduced from 25-40 GB)
- **Training (float32):** 4-6 GB peak (reduced from 12-20 GB)
- **NO MORE SWAP USAGE!** ✅
- **NO MORE OOM KILLS!** ✅

**Comparison:**
| Version | Extraction Peak | Training Peak | Swap Usage | Notes |
|---------|----------------|---------------|------------|-------|
| v3.13 | 48-90 GB | N/A | Heavy | Frequent OOM |
| v3.14 | 25-40 GB | 25-40 GB | None | Still risky on <64GB RAM |
| v3.15 (float64) | 500MB-1GB | 8-12 GB | None | Works on 16GB+ systems |
| v3.15 (float32) | 500MB-1GB | 4-6 GB | None | Works on 8GB+ systems |

### 8.14 Slow Initial Feature Extraction (✅ MASSIVELY IMPROVED in v3.13)
**Issue:** First run took ~55 minutes
**Previous Solution:** Caching (second run instant)
**New Solutions (v3.13):**
- Rolling statistics optimization: 45x speedup
- Vectorized ping-pong detection: 10x speedup
- **Combined:** 55 minutes → **~40-60 seconds** (even first run!)
**Cache still used:** Second run loads in ~2-5 seconds

---

## 9. Performance Optimization

### 9.1 Hardware Acceleration

```python
Device Priority:
1. CUDA (NVIDIA GPU) - Fastest, ~3-5x speedup
2. MPS (Apple Silicon) - Fast for M1/M2/M3, ~2x speedup
3. CPU - Fallback, slower but reliable

Auto-detection:
python train_hierarchical.py --device auto
```

### 9.2 Memory Management (Updated v3.15 - Chunked Extraction)

**System Memory Requirements (with Chunked Extraction):**
- **Extraction (any workers, any precision):** 500MB-1GB constant
- **Training 8 workers (float64):** 8-12 GB RAM - Works on 16GB+ systems
- **Training 8 workers (float32):** 4-6 GB RAM - Works on 8GB+ systems
- **Training 4 workers (float64):** 6-10 GB RAM - Works on 16GB systems
- **Training 4 workers (float32):** 3-5 GB RAM - Works on 8GB systems

**Without Chunking (legacy mode, not recommended):**
- **Extraction 8 workers (float64):** 25-40 GB RAM - Requires 64GB+ system
- **Extraction 8 workers (float32):** 12-20 GB RAM - Requires 32GB+ system

**Training Batch Sizes:**
| Batch Size | RAM Usage | GPU VRAM | Speed |
|------------|-----------|----------|-------|
| 16         | ~4GB      | ~2GB     | Slow  |
| 32         | ~8GB      | ~4GB     | Good  |
| 64         | ~16GB     | ~8GB     | Fast  |
| 128        | ~32GB     | ~12GB    | Fastest |

### 9.3 Chunked Feature Extraction (NEW in v3.15)

**Problem:** Large datasets (2015-2022, ~1M bars) caused memory accumulation during feature extraction, leading to OOM kills on systems with <64GB RAM.

**Solution:** Time-based chunking with HDF5 incremental appending

**How It Works:**
```python
# Process in 1-year chunks with 6-month overlap
Chunk 1 (2015-2016): 500MB → Write to HDF5
Chunk 2 (2016-2017): 500MB → Append to HDF5
Chunk 3 (2017-2018): 500MB → Append to HDF5
...
Chunk 7 (2021-2022): 500MB → Append to HDF5

Final: Load HDF5 → pandas DataFrame → Save to cache
```

**Memory Profile:**
- **Peak during chunks:** 500MB-1GB constant (never accumulates)
- **Peak during final load:** 3.5GB (brief, acceptable with swap)
- **Total peak:** <4GB vs previous 25-40GB

**Configuration:**
```bash
# Auto-detect (recommended):
python train_hierarchical.py --interactive
# Enables chunking if RAM < 64GB

# Force enable:
python train_hierarchical.py --use-chunking

# Disable (for 64GB+ systems):
python train_hierarchical.py --no-chunking
```

**Performance:**
- Overhead: ~10% slower than non-chunked (worth it for stability)
- Per-chunk time: ~40-60 seconds (same as non-chunked for that chunk)
- Total for 7 chunks: ~5-7 minutes first run
- Cached runs: Still instant (2-5 seconds)

**Requirements:**
- `tables>=3.8.0` package (HDF5 support)
- Disk space: ~1GB temp file during processing

### 9.4 Feature Caching

```python
# Cache structure
feature_cache/
├── rolling_channels_*.pkl           # Cached channel features (final result)
├── temp_chunked_*.h5               # Temp HDF5 (deleted after processing)
└── continuation_labels_*.pkl        # Continuation analysis

# Cache size: ~500MB-1GB
# Speedup: 30-60 minutes → 2-5 seconds
```

### 9.5 Parallel Processing (v3.13-v3.15: Custom Multi-Progress Implementation)

#### Multi-Window Parallel Extraction
Uses **custom multiprocessing** with rich progress bars for real-time visibility:

**How it works:**
- 22 tasks (11 timeframes × 2 stocks) distributed across worker processes
- Each worker calculates 21 windows per task using rolling statistics
- Real-time progress bars show each timeframe individually
- Results collected via Queue with proper synchronization

**Configuration (config.py):**
```python
PARALLEL_CHANNEL_CALC = True   # Enable/disable parallelization
MAX_PARALLEL_WORKERS = 4       # Recommended: 4 for 16-32GB RAM, 8 for 64GB+
CHANNEL_WINDOW_SIZES = [168, 160, 150, ..., 20, 15, 10]  # 21 windows
MIN_DATA_YEARS = 2.5           # Minimum data for 3-month TF with 10-bar window
```

**Performance Improvements (v3.13):**
| Method | Time | Speedup | Memory Usage | Features |
|--------|------|---------|--------------|----------|
| Sequential (old) | ~55 minutes | 1x | ~2-4 GB | 495 |
| Sequential (v3.13) | ~2-3 minutes | 27x | ~8-12 GB | 12,936 |
| Parallel 4 cores (v3.13) | ~40-60 seconds | 55-82x | ~20-25 GB | 12,936 |
| Parallel 8 cores (v3.13) | ~30-40 seconds | 82-110x | ~30-40 GB | 12,936 |

**Key Optimizations:**
1. **Rolling Statistics** (45x): Incremental sum updates instead of recalculating regression
2. **Vectorized Ping-Pongs** (10x): NumPy array operations instead of Python loops
3. **OHLC Integration**: Separate regressions for high/low/close (richer features)
4. **Memory Cleanup**: Explicit garbage collection prevents accumulation
5. **Multi-Progress Display**: Rich library shows real-time progress per timeframe

**When Parallelization is Used:**
- CPU mode (GPU requires sequential for context preservation)
- Any number of cores (even 1 core uses parallel framework)
- Not in live trading mode (for stability)

**Fallback to Sequential:**
- GPU mode active (CUDA/MPS context can't cross processes)
- Live trading mode (stability)
- Config: `PARALLEL_CHANNEL_CALC = False`

**macOS-Specific Optimizations (v3.15):**
- **Forkserver mode:** Replaces spawn for faster, safer process creation
- **Lazy torch loading:** config.py doesn't import torch (prevents worker deadlock)
- **Result collection first:** Collect from queue while workers flush, then cleanup
- **Impact:** 240s worker shutdown → 2s, no more forced kills

#### Future Scaling with Ray (Roadmap)
For massive-scale distributed computing:

**Potential Applications:**
- **Multi-machine clusters**: Scale to 100s/1000s of cores
- **Hyperparameter search**: Train multiple model configurations in parallel
- **Cross-validation**: Parallel fold processing
- **Batch inference**: Process thousands of predictions simultaneously

**Why Ray over joblib for future:**
- True distributed computing across network
- Better resource management and scheduling
- Built-in dashboard for monitoring
- Fault tolerance and automatic retries

**Note on Training Epochs:**
Individual training epochs **cannot be parallelized** as each epoch depends on the weights updated by the previous epoch. This is a fundamental constraint of gradient descent. However, within each epoch:
- **Batch processing** can be distributed
- **Data loading** can be parallelized
- **Multiple models** with different hyperparameters can train simultaneously

---

## 10. News Integration Roadmap

### User Requirements
*"I want the system to learn: Leading into earnings, sentiment is bad → stock did X. Headlines say crash, article says minor → BS score high, buy the dip"*

### Infrastructure Status (80% Built!)
✅ **src/news_analyzer.py** - Claude-powered sentiment analysis
✅ **src/ml/news_encoder.py** - LFM2-350M embeddings (768-dim)
✅ **src/ml/fetch_news.py** - RSS fetching & storage

❌ **Missing:** Historical news database (2015-2022)
❌ **Missing:** News features in model
❌ **Missing:** Pattern matching system

### Implementation Plan

#### PHASE 1: Acquire Historical News (2-3 hours, $200-500)

**Options:**
- **Benzinga API:** ~$200-500 for historical archive
- **Finnhub Premium:** Similar pricing, good coverage
- **Alpha Vantage Premium:** Check availability

**Required Format:**
```csv
date,headline,full_text,source
2018-03-15,"Tesla recalls vehicles","Full article...",Reuters
```

#### PHASE 2: Score Headlines with Claude (10-15 hours, ~$500)

```python
import pandas as pd
from src.news_analyzer import NewsAnalyzer

analyzer = NewsAnalyzer()
historical_news = pd.read_csv('historical_news.csv')

for _, row in historical_news.iterrows():
    result = analyzer.analyze_headline(
        row['headline'], row['full_text']
    )
    # Store: sentiment (-100 to +100), bs_score (0-100)
```

#### PHASE 3: Aggregate Daily Sentiment (2-3 hours)

```python
daily_sentiment = scored_news.groupby('date').agg({
    'sentiment': 'mean',
    'bs_score': 'mean',
    'headline': 'count'
})
```

#### PHASE 4: Add News Features (5-8 hours)

```python
def _extract_news_features():
    features = {
        'news_sentiment_24h': last_24h_sentiment,
        'news_sentiment_7d': rolling_7d_average,
        'news_bs_score_24h': bs_level,
        'news_count_24h': headline_count,
        'news_momentum': sentiment_change,
        'headline_article_discrepancy': headline_vs_article
    }
    return features
```

#### PHASE 5: Integration & Testing (3-4 hours)

1. Update feature extractor
2. Retrain model from scratch
3. Compare accuracy with/without news
4. Expected improvement: 10-20% around events

#### PHASE 6: Live Collection (2-3 hours, $10/month)

```bash
# Cron job for hourly news fetching
*/60 * * * * python src/ml/fetch_news.py --store-db
```

### Total Effort
- **Time:** 24-36 hours
- **Cost:** $700-1000 one-time + $10/month
- **Blocker:** Historical news acquisition

---

## 11. Testing & Validation

### 11.1 Data Validation
```bash
# Validate data files exist and are formatted correctly
python test_data_loading.py
```

### 11.2 Feature Validation
```bash
# Test feature extraction
python test_continuation_fix.py

# Validate all features extracting correctly
python validate_features.py
```

### 11.3 Model Validation
```bash
# Check trained model exists
ls -la models/hierarchical_lnn.pth

# Test inference
python -c "from src.ml.hierarchical_model import load_hierarchical_model; model = load_hierarchical_model('models/hierarchical_lnn.pth'); print('Model loaded successfully')"
```

### 11.4 Performance Benchmarks
- **Baseline (Buy & Hold):** ~15% annual return
- **Linear Regression Channels:** ~52% directional accuracy
- **Target (Hierarchical LNN):** >65% directional accuracy
- **High-Confidence Trades:** >75% accuracy when confidence > 0.8
- **Inference Speed:** <100ms per prediction

---

## 12. Troubleshooting Guide

### Common Issues & Solutions

#### "No data loaded" / FileNotFoundError
```bash
# Check data files exist
ls -la data/*.csv

# Verify format (should have timestamp,open,high,low,close,volume)
head -n 5 data/TSLA_1min.csv

# Check date ranges
python -c "import pandas as pd; df = pd.read_csv('data/TSLA_1min.csv', parse_dates=['timestamp'], index_col='timestamp'); print(f'Date range: {df.index.min()} to {df.index.max()}')"
```

#### CUDA/MPS Out of Memory
```bash
# Option 1: Reduce batch size
python train_hierarchical.py --batch_size 16

# Option 2: Use CPU instead
python train_hierarchical.py --device cpu

# Option 3: Clear GPU memory
# Restart Python or use: torch.cuda.empty_cache()
```

#### Module Not Found Errors
```bash
# Ensure you're in project root
cd /Users/frank/Desktop/CodingProjects/autotrade2

# Activate virtual environment
source myenv/bin/activate

# Verify Python path includes project
python -c "import sys; print('\n'.join(sys.path))"
```

#### Progress Bars Not Showing
```bash
# Use unbuffered output
python -u train_hierarchical.py

# Or set environment variable
export PYTHONUNBUFFERED=1
```

#### Feature Cache Corrupted
```bash
# Clear cache to force regeneration
rm -rf data/feature_cache/

# Features will be recalculated on next run
```

---

## 13. API Reference

### 13.1 Model Interface

```python
from src.ml.hierarchical_model import HierarchicalLNN, load_hierarchical_model

# Load trained model
model = load_hierarchical_model('models/hierarchical_lnn.pth', device='cpu')
model.eval()

# Prepare input
features = torch.tensor(...)  # Shape: [batch, sequence_length, num_features]

# Get predictions
with torch.no_grad():
    outputs = model(features)
    # Returns dict with: high, low, confidence, layer_outputs
```

### 13.2 Feature Extraction

```python
from src.ml.features import TradingFeatureExtractor

# Initialize
extractor = TradingFeatureExtractor()

# Extract features
features_df = extractor.extract_features(
    df,                    # OHLCV DataFrame
    timestamps,            # List of timestamps
    sequence_length=200,   # Lookback window
    device='auto'          # auto/cuda/mps/cpu
)
```

### 13.3 Data Feed

```python
from src.ml.data_feed import CSVDataFeed

# Initialize
feed = CSVDataFeed(data_dir='data', timeframe='1min')

# Load data
tsla_df = feed.load_data('TSLA', start_date='2024-01-01', end_date='2024-01-31')
spy_df = feed.load_data('SPY', start_date='2024-01-01', end_date='2024-01-31')
```

### 13.4 Continuation Labels

```python
from src.ml.features import TradingFeatureExtractor

extractor = TradingFeatureExtractor()
labels_df = extractor.generate_continuation_labels(
    df,                      # Full OHLC DataFrame
    timestamps,              # Timestamps to process
    prediction_horizon=24,   # Bars to look ahead
    debug=True              # Enable debug output
)
```

---

## 14. Future Enhancements

### Near-term Priorities (Next 1-3 months)

1. **News Sentiment Integration** (24-36 hours)
   - Historical news acquisition
   - Sentiment scoring pipeline
   - Model retraining with news features
   - Expected: 10-20% accuracy improvement

2. **Hierarchical Backtester** (3-4 hours)
   - Implement `backtest_hierarchical.py`
   - Simulate trading on 2023-2024 data
   - Generate performance metrics

3. **Dashboard Completion** (2-3 hours)
   - Fix feature count issue (490 vs 495)
   - Add trade execution interface
   - Improve layer visualization

### Long-term Vision (3-12 months)

1. **Advanced Architectures**
   - Transformer-based attention mechanisms
   - Graph neural networks for market structure
   - Ensemble methods combining multiple models

2. **Live Trading Integration**
   - Broker API connections (Interactive Brokers, Alpaca)
   - Position sizing algorithms
   - Risk management system
   - Real-time execution monitoring

3. **Multi-Asset Expansion**
   - Add more stocks beyond TSLA/SPY
   - Cryptocurrency support
   - Commodity futures

4. **Advanced Features**
   - Options flow analysis
   - Social sentiment from Reddit/Twitter
   - Satellite data integration
   - Alternative data sources

5. **Production Deployment**
   - Cloud deployment (AWS/GCP)
   - Distributed training
   - Real-time inference pipeline
   - Monitoring and alerting

---

## Version History

- **v3.15** (Nov 19, 2025):
  - **CRITICAL:** HDF5-based chunked feature extraction eliminates memory accumulation
  - **CRITICAL:** Fixed macOS torch multiprocessing deadlock with lazy loading + forkserver
  - **CRITICAL:** Reordered worker cleanup - collect results first (240s → 2s speedup)
  - Chunked extraction: Constant 500MB-1GB RAM regardless of dataset size
  - Interactive menu option for chunked extraction with auto-detection (<64GB RAM)
  - Workers exit cleanly without 30s hangs (torch background threads fixed)
  - Added `tables>=3.8.0` dependency for HDF5 support
  - **Memory reduction:** 25-40 GB → <4 GB for extraction phase
  - **Minimum RAM:** Now works on 16GB systems (float64 + chunking) or 8GB (float32 + chunking)
  - **No more OOM kills** during feature extraction
  - **CLI arguments:** `--use-chunking` / `--no-chunking` for manual control
- **v3.14** (Nov 18, 2024):
  - **CRITICAL:** Eliminated all memory leaks across 6 subsystems
  - **CRITICAL:** Fixed 5 hardcoded dtype references for full float64/float32 control
  - Model hidden state cleanup (prevents 5-10 GB accumulation)
  - Training loop periodic cleanup (prevents 1.6-3.2 GB accumulation)
  - GPU synchronization before cache clearing (prevents fragmentation)
  - Preload mode OOM prevention with user warnings
  - dtype validation throughout data pipeline
  - Dynamic progress messages (show actual 21 windows, not hardcoded 6)
  - **Peak memory:** 25-40 GB (float64) or 12-20 GB (float32) - NO SWAP!
  - **Memory reduction:** 48-90 GB → 25-40 GB (45-55% savings)
- **v3.13** (Nov 18, 2024):
  - **MAJOR:** Multi-window OHLC system (21 windows: 10-168 bars) with 12,936 features
  - **MAJOR:** Rolling statistics optimization (45x speedup)
  - **MAJOR:** Vectorized ping-pong detection (10x speedup)
  - **MAJOR:** Custom multiprocessing with rich multi-progress bars
  - Fixed 11 critical bugs (memory leaks, race conditions, worker hangs)
  - float64 precision throughout training pipeline
  - Memory optimization: 60-98 GB → 20-40 GB with cleanup
  - Performance: 55 minutes → 30-60 seconds (82-110x speedup)
- **v3.12** (Nov 17, 2024): Added joblib parallelization for 5.5-7.8x speedup in channel calculations
- **v3.11** (Nov 17, 2024): Feature expansion to 495 features, dynamic channel duration detection
- **v2.0** (Nov 15, 2024): Added multi-task learning, event integration, GPU acceleration
- **v1.0** (Nov 10, 2024): Initial hierarchical LNN implementation

---

## Important Notes

1. **Price-Agnostic Design**: The system works at any TSLA price level due to percentage-based features
2. **Multi-Task Learning**: Provides better regularization - keep enabled
3. **Event Window**: Currently ±14 days for event features (expanded from ±7)
4. **GPU Acceleration**: Uses hybrid GPU+CPU for speed with accuracy (GPU mode uses sequential)
5. **Parallel Processing (v3.13)**: Custom multiprocessing with 82-110x speedup, real-time multi-progress bars
6. **Cache Management**: FEATURE_VERSION="v3.13_multiwindow_21" invalidates old cache (intentional)
7. **Feature Count (v3.13)**: **12,936 features** (21 windows × 28 OHLC features × 22 combinations)
8. **Memory Requirements (v3.15)**:
   - **With chunking (recommended):** <4 GB extraction, 8-12 GB training (float64) or 4-6 GB (float32)
   - **Without chunking:** 25-40 GB extraction (float64) - only for 64GB+ systems
   - **Minimum system:** 16GB RAM (with float32 + chunking) - NO SWAP!
9. **Multi-Window System**: ALL windows calculated with quality scores - no filtering, model learns relevance
10. **Configurable Precision (v3.13)**:
    - **Centralized control**: `config.TRAINING_PRECISION` controls all 87 dtype locations
    - **Interactive menu**: Select float64 or float32 during setup
    - **Recommendation**: float64 for training (maximum precision), quantize to int8/float16 for deployment
    - **Architecture**: All NumPy arrays, PyTorch tensors, and model operations respect precision setting
11. **Data Requirements**: Minimum 2.5 years for 3-month TF with 10-bar lookback (257,400 1-min bars)
12. **Continuation Labels**: Fixed duplicate resampling bug in v3.0
13. **News Priority**: User's top priority - infrastructure 80% ready
14. **macOS Compatibility (v3.15)**:
    - **Lazy torch loading**: Prevents worker deadlocks on exit
    - **Forkserver mode**: Safer than spawn for multiprocessing
    - **Result collection first**: No waiting for background thread cleanup
    - **Workers exit instantly**: No more 30s hangs or forced kills

---

*For questions or issues, refer to the troubleshooting guide or create an issue in the project repository.*

---

## Summary of v3.15 Improvements

**Memory Optimization:**
- Chunked extraction: 25-40GB → <4GB (87-90% reduction)
- HDF5 appending: No accumulation, constant RAM
- Works on 16GB systems (was 64GB minimum)

**macOS Compatibility:**
- Lazy torch loading: Workers don't import torch
- Forkserver mode: Safer than spawn
- Worker cleanup: 240s → 2s (99% reduction)
- No more deadlocks or forced kills

**User Experience:**
- Interactive chunking menu with smart defaults
- Clear memory recommendations based on system RAM
- Automatic fallback for low-memory systems
- Detailed progress during chunked processing

**Reliability:**
- No more OOM kills during extraction
- No more worker hangs on macOS
- Stable on systems with as little as 8GB RAM (float32 + chunking)

*END OF TECHNICAL SPECIFICATION v3.15*