# AutoTrade2 v3.5 - Quick Start Guide

**Get up and running in 5 minutes!**

---

## Prerequisites

- Python 3.10+
- 8GB+ RAM (16GB recommended for M1/M2 Mac)
- GPU recommended (CUDA/MPS) but not required

---

## Installation

```bash
# Clone repository
cd /path/to/autotrade2

# Create virtual environment
python3 -m venv myenv
source myenv/bin/activate  # On Mac/Linux
# myenv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

**Key Dependencies:**
- PyTorch 2.0+ (with CUDA/MPS support)
- ncps (Liquid Neural Networks)
- pandas, numpy, yfinance
- InquirerPy (interactive menus)

---

## Quick Training

### Option 1: Interactive Mode (Recommended)

```bash
python train_hierarchical.py --interactive
```

**Interactive menu will guide you through:**
1. Device selection (CUDA/MPS/CPU auto-detected)
2. Training years (default: 2015-2022)
3. Cache management (use existing or regenerate)
4. Model capacity (192/256/384/512 neurons)
5. Batch size and learning rate
6. Number of epochs and early stopping

**Example session:**
```
? Select compute device: Apple Silicon GPU (MPS) - Fast 🍎 [Detected]
? Training data start year: 2015
? Training data end year: 2022

📂 Feature Cache Found:
   💾 Size: 487.3 MB
   📅 Created: 2024-11-14 15:32:41
   📊 Version: v3.5

? Use existing cache or regenerate features? Use cache (fast - loads in ~5 seconds) ⭐

? Model capacity: Standard (256 total, 128 output) - Recommended ⭐
? Number of epochs: 100
? Batch size: 64
? Learning rate: 0.001

? Start training with these settings? Yes

✅ Training started...
```

### Option 2: Command Line

```bash
python train_hierarchical.py \
  --epochs 100 \
  --batch_size 64 \
  --device mps \
  --sequence_length 200 \
  --prediction_horizon 24 \
  --train_start_year 2015 \
  --train_end_year 2022 \
  --val_split 0.1 \
  --lr 0.001 \
  --output models/hierarchical_lnn.pth
```

### What to Expect

**First Training Run:**
```
1. Loading 1-min data...
   Loaded 1150051 bars (2015-01-02 to 2022-12-31)

2. Extracting features...
   Feature extraction:  14%|████▌  | 1/7 [00:00<00:00]
   🔄 Calculating ROLLING channels (30-60 mins first time)...
   📊 Processing 22 calculations (11 timeframes × 2 stocks)
   ⏱️  Estimated time: ~55 minutes
   
   Rolling channels (SPY + TSLA):   5%|▌  | 1/22 [02:34<48:23]
      TSLA 5min:  41%|████▏  | 8,234/20,000 [00:45<01:02, 189 bars/s]
   
   ✓ Extracted 473 features

3. Creating datasets...
   Train: 1034546 samples, Val: 115051 samples

4. Creating model...
   HierarchicalLNN: 2.8M parameters

5. Training...
   Training Progress:   2%|█▌  | 2/100 [04:41<3:50:12]
   
   Epoch 2/100
   ----------------------------------------------------------------------
     Epoch 2 [Train]: 100%|████| 1234/1234 [02:14<00:00, loss=0.0198]
     Validating: 100%|████| 137/137 [00:12<00:00, loss=0.0187]
     ✓ New best model (val_loss: 0.0187)

TRAINING COMPLETE
Best val loss: 0.0187
Model saved to: models/hierarchical_lnn.pth
```

**Subsequent Runs (cache exists):**
```
2. Extracting features...
   Loading cache: 100%|██████| 1/1 [00:03<00:00]
   ✓ Loaded channel features from cache
   ✓ Extracted 473 features
   
[Training proceeds as above, ~5-10 minutes total]
```

---

## Making Predictions

### Live Prediction Script

```python
from src.ml.live_data_feed import HybridLiveDataFeed
from src.ml.hierarchical_model import load_hierarchical_model
from src.ml.features import TradingFeatureExtractor
from datetime import datetime

# Load components
feed = HybridLiveDataFeed(symbols=['TSLA', 'SPY'])
model = load_hierarchical_model('models/hierarchical_lnn.pth')
extractor = TradingFeatureExtractor()

# Fetch live data (downloads 1-min, 1-hour, daily)
print("📡 Fetching live data...")
df = feed.fetch_for_prediction()

# Extract features (uses multi-resolution data automatically)
print("🔧 Extracting features...")
features = extractor.extract_features(df, use_cache=True)

# Make prediction
print("🔮 Making prediction...")
pred = model.predict(features[-200:])  # Last 200 bars

# Display results
print(f"\n{'='*60}")
print(f"PREDICTION @ {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'='*60}")
print(f"Predicted High:    {pred['predicted_high']:+.2f}%")
print(f"Predicted Low:     {pred['predicted_low']:+.2f}%")
print(f"Predicted Center:  {pred['predicted_center']:+.2f}%")
print(f"Predicted Range:   {pred['predicted_range']:.2f}%")
print(f"Confidence:        {pred['confidence']:.1%}")
print(f"Volatility:        {pred['predicted_volatility']:.2f}%")
print(f"\nFusion Weights:")
print(f"  Fast:   {pred['fusion_weights']['fast']:.2f}")
print(f"  Medium: {pred['fusion_weights']['medium']:.2f}")
print(f"  Slow:   {pred['fusion_weights']['slow']:.2f}")
print(f"{'='*60}")
```

**Example Output:**
```
📡 Fetching live data...
   ✓ 1-min: 2700 bars
   ✓ 1-hour: 3494 bars
   ✓ Daily: 3871 bars

🔧 Extracting features...
   Loading cache: 100%|██████| 1/1 [00:03<00:00]
   ✓ Extracted 473 features

🔮 Making prediction...

============================================================
PREDICTION @ 2024-11-14 15:30
============================================================
Predicted High:    +2.85%
Predicted Low:     -0.62%
Predicted Center:  +1.12%
Predicted Range:   3.47%
Confidence:        82.3%
Volatility:        2.10%

Fusion Weights:
  Fast:   0.35
  Medium: 0.48
  Slow:   0.17
============================================================
```

---

## Cache Management

### Understanding the Cache

**What's cached:** Rolling channel calculations (30-60 mins first time)  
**Cache location:** `data/feature_cache/rolling_channels_v3.5_*.pkl`  
**Cache size:** ~500MB per date range  

### Cache Operations

```bash
# View cache files
ls -lh data/feature_cache/

# Clear old caches (free disk space)
rm data/feature_cache/rolling_channels_v*.pkl

# Force regenerate cache (interactive mode)
python train_hierarchical.py --interactive
# Select "Regenerate cache" when prompted

# Force regenerate cache (command line)
# Delete cache file manually, then run training
```

### When to Regenerate Cache

1. **Version change:** Feature calculation logic updated (FEATURE_VERSION bump)
2. **Data change:** Using different date range
3. **Corruption:** Cache file corrupted or incomplete
4. **Testing:** Want to verify fresh calculation

---

## GPU Acceleration

### Overview

GPU acceleration speeds up rolling channel calculation by ~1.5-1.8x on first run. Uses hybrid GPU+CPU approach for correctness.

**Performance gains:**
- 10K bars: 20 sec → 15 sec (1.3x)
- 50K bars: 5 mins → 3 mins (1.7x)
- 1.15M bars (training): 45 mins → 25-30 mins (1.5-1.8x)
- **Cached runs:** 2-5 seconds (same for GPU and CPU)

### Supported Hardware

- ✅ **Apple Silicon** (M1/M2/M3) - via MPS
- ✅ **NVIDIA GPUs** - via CUDA
- ⚠️ **Not supported:** AMD GPUs, older Macs

### Accuracy

GPU produces **nearly identical results** to CPU:
- Most features: Exact match (within 0.01%)
- Ping-pongs: May differ by ±1-2 counts (floating point edge cases)
- Stability: May differ by ±0.04 points out of 100 (0.04% difference)

**Impact:** Negligible for model training - learns patterns, not exact counts.

### Usage

**Interactive menu (recommended):**
```
? Use GPU acceleration for feature extraction?
  ● Yes - Use MPS GPU ⚡
  ○ No - Use CPU 💾
```

**Note:** If cache exists, GPU only applies if you choose "Regenerate cache"

### Validation

Verify GPU is working correctly:
```bash
python validate_gpu_cpu_equivalence.py

# Tests GPU vs CPU equivalence
# Should show: ✅ ALL TESTS PASSED
```

### When to Use

**GPU beneficial:**
- First training run (no cache)
- Regenerating cache (new date range)
- Development/experimentation

**CPU fine:**
- Cache already exists (instant either way)
- Live predictions (auto-uses CPU)

---

## Validation & Testing

### Validate Rolling Channels

```bash
python scripts/validate_channels.py --timeframe 1h --year 2023

# Expected output:
# ✅ Rolling channels working correctly
# - r² varies: 0.08 → 0.95 (dynamic, not static)
# - Ping-pongs vary: 0 → 15
# - Position varies: -1.2 → 1.3
```

### Test Hybrid Live Integration

```bash
python test_hybrid_features.py

# Expected output:
# ✅ HYBRID FEATURE EXTRACTION TEST PASSED!
# - Multi-resolution data: 1min (2730), 1hour (3494), daily (3871)
# - Features extracted: 313
# - Channel features: VALID ✓
# - RSI features: VALID ✓
```

### Validate Features

```bash
python validate_features.py

# Checks:
# - Feature count (313)
# - No NaN/Inf values
# - Reasonable ranges
# - Column names match spec
```

---

## Common Commands

### Training

```bash
# Standard training (2015-2022)
python train_hierarchical.py --interactive

# Custom date range
python train_hierarchical.py \
  --train_start_year 2018 \
  --train_end_year 2023 \
  --epochs 50

# CPU-only training
python train_hierarchical.py --device cpu --batch_size 32

# Resume from checkpoint
python train_hierarchical.py --resume models/hierarchical_lnn.pth
```

### Prediction

```bash
# Single prediction
python predict.py --model models/hierarchical_lnn.pth

# Continuous predictions (every 30 mins)
python predict_loop.py --interval 30

# Backtest on historical data
python backtest.py \
  --model models/hierarchical_lnn.pth \
  --start_date 2023-01-01 \
  --end_date 2023-12-31
```

### Analysis

```bash
# View training history
python scripts/plot_training_history.py models/hierarchical_lnn.pth

# Analyze feature importance
python scripts/analyze_feature_importance.py models/hierarchical_lnn.pth

# Check high-confidence trades
python scripts/view_trades.py --min_confidence 0.75
```

---

## Troubleshooting

### Issue: "Out of memory" during training

**Solution:**
```bash
# Reduce batch size
python train_hierarchical.py --batch_size 32  # or 16

# Use CPU instead of GPU
python train_hierarchical.py --device cpu

# Disable preload mode
python train_hierarchical.py --preload false
```

### Issue: "Cache file too small, regenerating..."

**Cause:** Corrupted or incomplete cache  
**Solution:** Let it regenerate automatically (wait 30-60 mins)

### Issue: First run takes very long

**Expected!** Rolling channel calculation takes 30-60 minutes first time.  
**Solution:** Be patient. Subsequent runs will be instant (2-5 seconds).

### Issue: "Dimension mismatch" error

**Cause:** Old model file incompatible with new code  
**Solution:**
```bash
# Delete old model
rm models/hierarchical_lnn.pth

# Retrain from scratch
python train_hierarchical.py --interactive
```

### Issue: Progress bars look wrong

**Cause:** Terminal doesn't support ANSI colors  
**Solution:** Use `--no-progress` flag (if available) or ignore visual artifacts

---

## Performance Tips

### For Faster Training

1. **Use GPU:** MPS (Mac) or CUDA (NVIDIA) >> CPU
2. **Increase batch size:** 64-128 (if memory allows)
3. **Use cache:** Don't regenerate unless necessary
4. **Reduce epochs:** 50-100 is often enough

### For Better Accuracy

1. **More training data:** 2015-2022 (7+ years)
2. **Longer sequences:** 200-300 bars
3. **Higher capacity:** 384 or 512 total neurons
4. **More epochs:** 100-200 with early stopping

### For Live Predictions

1. **Pre-warm cache:** Run training once to build cache
2. **Keep model loaded:** Don't reload for each prediction
3. **Use multi-resolution:** HybridLiveDataFeed handles it automatically

---

## File Structure

```
autotrade2/
├── src/ml/
│   ├── features.py              # 313-feature extraction
│   ├── hierarchical_model.py    # Model architecture
│   ├── hierarchical_dataset.py  # Dataset preparation
│   ├── online_learner.py        # Online learning
│   ├── trade_tracker.py         # Trade logging
│   ├── live_data_feed.py        # Hybrid data fetching
│   └── prediction_scheduler.py  # Adaptive scheduling
├── train_hierarchical.py        # Training script
├── predict.py                   # Single prediction
├── test_hybrid_features.py      # Live integration test
├── scripts/
│   ├── validate_channels.py     # Channel validation
│   └── analyze_feature_importance.py
├── data/
│   ├── SPY_1min.csv            # Historical SPY data
│   ├── TSLA_1min.csv           # Historical TSLA data
│   └── feature_cache/          # Cached rolling channels
├── models/
│   └── hierarchical_lnn.pth    # Trained model
├── config/
│   └── hierarchical_config.yaml # Configuration
├── SPEC.md                      # Complete specification
└── QUICKSTART.md                # This guide
```

---

## Next Steps

1. ✅ **Train your first model:** `python train_hierarchical.py --interactive`
2. ✅ **Make a prediction:** Create prediction script using example above
3. ✅ **Explore features:** Run `validate_features.py` to see all 473 features
4. ✅ **Read spec:** See `SPEC.md` for complete technical details

---

## Resources

- **Technical Specification:** `SPEC.md`
- **Code Documentation:** Inline comments in source files
- **Test Scripts:** `test_*.py` and `scripts/validate_*.py`
- **Configuration:** `config/hierarchical_config.yaml`

---

## Getting Help

1. **Check logs:** Look for error messages in console output
2. **Validate setup:** Run test scripts to verify installation
3. **Read spec:** `SPEC.md` has detailed technical information
4. **Check code:** All functions have docstrings

---

**Status:** 🟢 Ready to train and predict!

**Estimated time to first prediction:** 35-70 minutes (first run) or 6-11 minutes (cached)


  How to Add Delivery Dates

  Open the file and add rows like this:

  date,event_type,expected,actual,beat_miss,category
  2026-01-02,delivery,500000,0.0,neutral,tsla
  2026-04-02,delivery,520000,0.0,neutral,tsla
  2026-07-02,delivery,540000,0.0,neutral,tsla
  2026-10-02,delivery,560000,0.0,neutral,tsla

  ---
  Format Explanation

  | Column     | What to Put                        | Example    |
  |------------|------------------------------------|------------|
  | date       | YYYY-MM-DD format                  | 2026-01-02 |
  | event_type | Always "delivery"                  | delivery   |
  | expected   | Expected deliveries (estimate)     | 500000     |
  | actual     | Leave as 0.0 (fill after report)   | 0.0        |
  | beat_miss  | Leave as "neutral" (before report) | neutral    |
  | category   | Always "tsla"                      | tsla       |


