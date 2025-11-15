# Quick Start Guide - Hierarchical Trading System

**System Status:** 🟢 Production Ready (November 2024)

This guide covers everything you need to train, validate, and deploy the hierarchical trading system.

---

## 📋 Table of Contents

1. [Installation](#installation)
2. [Training the Model](#training-the-model)
3. [Validation & Testing](#validation--testing)
4. [Live Predictions](#live-predictions)
5. [Cache Management](#cache-management)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

---

## 🔧 Installation

### Prerequisites
- Python 3.10+
- 8GB+ RAM
- GPU recommended (but not required - M1/M2 Mac works great!)

### Setup
```bash
# Clone repository (if not already done)
cd /path/to/autotrade2

# Create virtual environment
python3 -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from src.ml.hierarchical_model import HierarchicalLNN; print('✓ Model imports')"
```

---

## 🎓 Training the Model

### Interactive Mode (Recommended)
```bash
python train_hierarchical.py --interactive
```

**Interactive menus guide you through:**
1. Device selection (CUDA/MPS/CPU)
2. Capacity selection (128/256/512 neurons)
3. Training parameters (epochs, batch size)
4. Model output location

**First Run Timeline:**
```
1. Loading data... (10-20 seconds)
2. Extracting features... (30-60 MINUTES - builds cache)
   🔄 Calculating ROLLING channels...
   💾 Saving to cache
3. Creating datasets... (10 seconds)
4. Training... (5-10 minutes for 50 epochs on M1/M2)
5. Saving model... ✓

Total: ~40-70 minutes
```

**Subsequent Runs (Cache Exists):**
```
1. Loading data... (10-20 seconds)
2. Extracting features... (2-5 SECONDS - loads from cache!)
3. Creating datasets... (10 seconds)
4. Training... (5-10 minutes)
5. Saving model... ✓

Total: ~6-11 minutes (much faster!)
```

### Command-Line Mode
```bash
python train_hierarchical.py \
  --epochs 100 \
  --batch_size 64 \
  --device auto \
  --train_start_year 2015 \
  --train_end_year 2022 \
  --output models/hierarchical_lnn.pth
```

**Training Time Estimates:**
- **M1/M2 Mac (MPS):** 3-5 seconds/epoch → 5-8 minutes for 100 epochs
- **RTX 3090 (CUDA):** 1-2 seconds/epoch → 2-3 minutes for 100 epochs
- **CPU:** 15-30 seconds/epoch → 25-50 minutes for 100 epochs

---

## ✅ Validation & Testing

### 1. Validate Channel Detection (Before Training)

**Check if rolling channels are working correctly:**
```bash
python scripts/validate_channels.py --timeframe 1h --year 2023
```

**What to Look For:**
```
1h Channel Metrics:
  R-Squared (Goodness of Fit):
    Mean: 0.612    ✓ Good average
    Median: 0.658
    Min: 0.08      ✓ MUST VARY! (if constant = bug!)
    Max: 0.95      ✓ MUST VARY!

  Ping-Pongs:
    Mean: 4.3      ✓ Channels are validated
    Median: 4
    Min: 0         ✓ MUST VARY!
    Max: 15        ✓ MUST VARY!

✅ VERDICT: Channels are SOLID and DYNAMIC
```

**⚠️ Red Flags:**
- If r² is CONSTANT (e.g., all values = 0.057) → Channels are still static (BUG!)
- If r² mean < 0.3 → Channels are weak for this timeframe
- Expected: r² should vary showing channel formation/breakdown

**Test Other Timeframes:**
```bash
python scripts/validate_channels.py --timeframe 4h --year 2023
python scripts/validate_channels.py --timeframe daily --year 2022
```

**Output:** Saves `channel_quality_{timeframe}.png` with distribution plots

---

### 2. Test Hybrid Live Integration

**Verify live predictions can fetch and extract features:**
```bash
python test_hybrid_features.py
```

**Expected Output:**
```
✅ HYBRID FEATURE EXTRACTION TEST PASSED!

Summary:
  - Live data fetched: 2730 bars (1-min)
  - Multi-resolution data:
      • 1min: 2730 bars (7 days)
      • 1hour: 3494 bars (2 years!)
      • daily: 3871 bars (15 years!)
  - Features extracted: 309 features
  - Multi-resolution mode: WORKING ✓
  - Channel features: VALID ✓
    • tsla_channel_4h_r_squared: 0.7637 (strong!)
  - RSI features: VALID ✓
    • tsla_rsi_1h: 27.33 (oversold)
```

---

### 3. Analyze Feature Importance (After Training)

**See which features the model learned to trust:**
```bash
python scripts/analyze_feature_importance.py --model_path models/hierarchical_lnn.pth
```

**What It Shows:**
- Top 20 most important features
- Importance by category (channels, RSI, volume, etc.)
- Which timeframes matter most
- Whether model trusts your trading concepts

**Example Output:**
```
TOP 20 MOST IMPORTANT FEATURES:
 1. tsla_channel_1h_position        ████████ 0.034521
 2. tsla_rsi_1h                     ███████  0.028934
 3. spy_correlation_10              ██████   0.024512
 4. tsla_channel_4h_r_squared       ██████   0.022134
 5. tsla_channel_1h_upper_dist      █████    0.019823

Importance by Category:
   Channel Features: 0.2841  ← Model TRUSTS channels!
   RSI Features: 0.2134      ← Model TRUSTS RSI!
   Correlation: 0.1523       ← Learns SPY relationship
   Volume: 0.0834
```

**Output File:** `feature_importance_report.txt`

---

## 🚀 Live Predictions

### Simple Prediction (One-Time)

```python
from src.ml.live_data_feed import HybridLiveDataFeed
from src.ml.hierarchical_model import load_hierarchical_model
from src.ml.features import TradingFeatureExtractor
import torch

# Initialize (one-time setup)
feed = HybridLiveDataFeed(symbols=['TSLA', 'SPY'])
model = load_hierarchical_model('models/hierarchical_lnn.pth', device='mps')
extractor = TradingFeatureExtractor()

# Fetch live data
print("Fetching live data...")
df = feed.fetch_for_prediction()
# Downloads:
#   - 1-min: 7 days (~2,730 bars)
#   - 1-hour: 2 years (~3,494 bars)
#   - Daily: Max history (~3,871+ bars)

# Extract features (same 309 as training)
print("Extracting features...")
features_df = extractor.extract_features(df)

# Prepare for prediction (last 200 bars)
x = torch.tensor(features_df.values[-200:], dtype=torch.float32).unsqueeze(0)

# Predict
with torch.no_grad():
    pred = model.predict(x)

print(f"\n📊 Live Prediction:")
print(f"  Predicted High: {pred['predicted_high']:.2f}%")
print(f"  Predicted Low: {pred['predicted_low']:.2f}%")
print(f"  Confidence: {pred['confidence']:.2f}")
print(f"  Fusion Weights: {pred['fusion_weights']}")  # [fast, medium, slow]
```

---

### Continuous Prediction Loop

```python
import time
from datetime import datetime

# Initialize once
feed = HybridLiveDataFeed(symbols=['TSLA', 'SPY'])
model = load_hierarchical_model('models/hierarchical_lnn.pth', device='mps')
extractor = TradingFeatureExtractor()

print("🔄 Starting continuous prediction loop...")
print("   Press Ctrl+C to stop\n")

while True:
    try:
        # Fetch latest data
        df = feed.fetch_for_prediction()

        # Extract features
        features_df = extractor.extract_features(df)

        # Predict
        x = torch.tensor(features_df.values[-200:], dtype=torch.float32).unsqueeze(0)
        pred = model.predict(x)

        # Display
        print(f"{datetime.now().strftime('%H:%M:%S')} | "
              f"High: {pred['predicted_high']:+.2f}% | "
              f"Low: {pred['predicted_low']:+.2f}% | "
              f"Conf: {pred['confidence']:.2f}")

        # Your trading logic here...
        if pred['confidence'] > 0.75:
            print(f"   🔥 HIGH CONFIDENCE SETUP!")
            # Execute trade, send alert, etc.

        # Wait 1 minute
        time.sleep(60)

    except KeyboardInterrupt:
        print("\n\n✋ Stopped by user")
        break
    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        time.sleep(60)  # Wait and retry
```

---

## 💾 Cache Management

### Understanding the Cache

**The cache stores pre-computed rolling channels to dramatically speed up feature extraction.**

#### First Run (No Cache):
```bash
python train_hierarchical.py --interactive

# Output:
2. Extracting features...
   🔄 Calculating ROLLING channels (this will take ~30-60 mins first time)...
   💡 Results will be cached for instant loading next time

   Rolling channels: 100%|████████| 22/22 [45:23<00:00]
   💾 Saving to cache: rolling_channels_20150102_20221231_1150051.pkl

   ✓ Extracted 309 features (TOTAL TIME: ~45 minutes)
```

#### Subsequent Runs (Cache Exists):
```bash
python train_hierarchical.py --interactive

# Output:
2. Extracting features...
   ✓ Loading channel features from cache: rolling_channels_20150102_20221231_1150051.pkl
   ✓ Extracted 309 features (TOTAL TIME: ~3 seconds!)
```

### Cache Location & Structure

```
data/feature_cache/
├── rolling_channels_20150102_20221231_1150051.pkl  (~500MB)
├── rolling_channels_20230101_20231231_198995.pkl   (~200MB)
└── rolling_channels_20251106_20251114_2730.pkl     (~50MB - live data)
```

**Cache filename format:** `rolling_channels_{start_date}_{end_date}_{num_bars}.pkl`

### Cache Management Commands

**View cache:**
```bash
ls -lh data/feature_cache/
```

**Clear all caches (forces recalculation):**
```bash
rm -rf data/feature_cache/
```

**Clear specific cache:**
```bash
rm data/feature_cache/rolling_channels_20150102_20221231_1150051.pkl
```

**When to clear cache:**
- ✅ Data source changed (new CSV files)
- ✅ Channel calculation logic modified
- ✅ Debugging feature extraction issues
- ❌ Just retraining with different epochs/batch size (keep cache!)

### Cache Benefits

| Aspect | First Run | Cached Run |
|--------|-----------|------------|
| Feature extraction time | 30-60 minutes | 2-5 seconds |
| Total training time | 40-70 minutes | 6-11 minutes |
| Disk space used | +500MB cache | 500MB cache |
| Can change model params? | Yes | Yes ✓ |
| Can change epochs? | Yes | Yes ✓ |

**💡 Pro Tip:** After first successful training, feature extraction is instant! Experiment freely with different model architectures, epochs, batch sizes without re-computing features.

---

## 🔧 Advanced Usage

### Online Learning (Continual Improvement)

```python
from src.ml.online_learner import OnlineLearner
from src.ml.hierarchical_model import load_hierarchical_model

# Initialize
model = load_hierarchical_model('models/hierarchical_lnn.pth')
learner = OnlineLearner(model, learning_rate=0.0001)

# Make prediction with tracking
pred, pred_id = learner.predict_with_tracking(
    x,
    current_price=245.50,
    timestamp=datetime.now()
)

print(f"Prediction ID: {pred_id}")
print(f"Predicted High: {pred['predicted_high']:.2f}%")

# 30 minutes later... validate and update model
actual_high = 2.5  # Actual high was +2.5%
actual_low = -0.8  # Actual low was -0.8%

learner.validate_and_update(pred_id, actual_high, actual_low)
print("✓ Model updated with actual results")
```

---

### High-Confidence Trade Tracking

```python
from src.ml.trade_tracker import TradeTracker

# Initialize tracker
tracker = TradeTracker(
    confidence_threshold=0.75,
    db_path='data/trades.db'
)

# Log high-confidence trade
if pred['confidence'] > 0.75:
    trade_id = tracker.log_trade(
        timestamp=datetime.now(),
        model_type='hierarchical',
        confidence=pred['confidence'],
        predicted_high=pred['predicted_high'],
        predicted_low=pred['predicted_low'],
        current_price=245.50,
        features_dict=features.to_dict(),
        layer_weights=pred['fusion_weights']
    )

    print(f"✓ Logged trade {trade_id}")

    # Later, update with actuals
    tracker.update_actual(trade_id, actual_high=2.5, actual_low=-0.8)

# Get performance stats
stats = tracker.get_stats()
print(f"High-Confidence Trades:")
print(f"  Win Rate: {stats['win_rate']:.1f}%")
print(f"  Total Trades: {stats['total_trades']}")
print(f"  Avg Confidence: {stats['avg_confidence']:.2f}")
```

---

## ❓ Troubleshooting

### Issue: "Channels are all constant (r² doesn't vary)"

**Symptom:**
```bash
python scripts/validate_channels.py --timeframe 1h --year 2023

# Output shows:
R-Squared: Mean: 0.057, Min: 0.057, Max: 0.057  ← ALL SAME!
```

**Cause:** Rolling channel detection not working (static channels)

**Fix:**
1. Check you're using latest `features.py` (should have `_calculate_rolling_channels()`)
2. Clear cache and retrain:
   ```bash
   rm -rf data/feature_cache/
   python train_hierarchical.py --interactive
   ```
3. Verify fix worked:
   ```bash
   python scripts/validate_channels.py --timeframe 1h --year 2023
   # Should now show: Min: 0.08, Max: 0.95 (VARIES!)
   ```

---

### Issue: "Live predictions fail with insufficient data"

**Symptom:**
```python
features_df = extractor.extract_features(df)
# KeyError or insufficient data errors
```

**Cause:** Not using HybridLiveDataFeed (using old data feed)

**Fix:**
```python
# ❌ OLD (doesn't work for live):
from src.ml.data_feed import CSVDataFeed
feed = CSVDataFeed()

# ✅ NEW (works for live):
from src.ml.live_data_feed import HybridLiveDataFeed
feed = HybridLiveDataFeed(symbols=['TSLA', 'SPY'])
```

---

### Issue: "Feature extraction takes 60 minutes every time"

**Symptom:** Cache not being used

**Causes & Fixes:**
1. **Data range changing:** If you change date ranges, cache won't match
   - Solution: Use consistent date ranges, cache will build per range

2. **Cache cleared:** Someone deleted `data/feature_cache/`
   - Solution: Let it rebuild (one-time 60 mins)

3. **Different data:** Training on 2015-2022, then 2016-2023
   - Solution: Normal - different data = different cache file

**Verify cache is working:**
```bash
# First run
python train_hierarchical.py --interactive
# Should show: "🔄 Calculating ROLLING channels..."

# Second run (immediately after)
python train_hierarchical.py --interactive
# Should show: "✓ Loading channel features from cache" (instant!)
```

---

### Issue: "Out of memory during training"

**Solutions:**
1. **Reduce batch size:**
   ```bash
   python train_hierarchical.py --batch_size 32  # Instead of 64
   ```

2. **Use smaller capacity:**
   ```bash
   # In interactive mode, choose 128 neurons instead of 256/512
   ```

3. **Use CPU instead of GPU:**
   ```bash
   python train_hierarchical.py --device cpu
   ```

4. **Reduce lookback window:**
   - Edit `train_hierarchical.py`: change `lookback=200` to `lookback=100`

---

## 📚 Next Steps

**After successful training:**

1. ✅ **Validate channels work** → `python scripts/validate_channels.py`
2. ✅ **Analyze feature importance** → `python scripts/analyze_feature_importance.py`
3. ✅ **Test live integration** → `python test_hybrid_features.py`
4. ✅ **Run backtest** (if available)
5. ✅ **Paper trade** before live deployment

**For more details:**
- [`README.md`](README.md) - System overview
- [`HIERARCHICAL_SPEC.md`](HIERARCHICAL_SPEC.md) - Complete technical specification
- [`COMPLETE_SYSTEM_STATUS.md`](COMPLETE_SYSTEM_STATUS.md) - Final status and what the model learns

---

**You're ready to train! 🚀**

```bash
python train_hierarchical.py --interactive
```
