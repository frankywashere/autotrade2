# Stage 2 Quick Start Guide

Get started with the ML-powered predictive model in 5 minutes!

---

## Prerequisites

1. **Stage 1 must be working** (data files exist, dependencies installed)
2. **Python 3.11+** installed
3. **8GB+ RAM** recommended

---

## Quick Setup (5 Minutes)

### 1. Install New Dependencies

```bash
pip install torch ncps sqlalchemy
```

Or update everything:

```bash
pip install -r requirements.txt
```

### 2. Process Real Events Data (NO FAKE DATA!)

```bash
python3 process_real_events.py
```

This parses your REAL data files:
- `data/earnings:P&D.rtf` → 86 TSLA events (earnings + deliveries)
- `data/historical_events.txt` → 397 macro events (FOMC, CPI, NFP)

**Output:** `data/tsla_events_REAL.csv` with 483 real events

### 3. Validate Data Alignment (CRITICAL!)

```bash
python3 validate_data_alignment.py --events_data data/tsla_events_REAL.csv
```

**This is MANDATORY before training!**

Expected output:
```
✅ DATA VALIDATION PASSED - SAFE TO TRAIN!
  - 1,349,074 perfectly aligned SPY/TSLA bars
  - 394 events in training range
  - No nulls, zeros, or fake data
```

**If you see errors:** Fix them before training! Never train on invalid data.

### 4. Train the Model (Fast Mode)

For quick testing, train on just 1 year with reduced epochs:

```bash
python3 train_model.py \
  --tsla_events data/tsla_events_REAL.csv \
  --start_year 2023 \
  --end_year 2023 \
  --epochs 10 \
  --pretrain_epochs 3 \
  --batch_size 16 \
  --output models/lnn_quick.pth
```

**Time:** ~10-15 minutes
**Data:** ~250K aligned bars, ~45 events

For full training (recommended for production):

```bash
python3 train_model.py \
  --tsla_events data/tsla_events_REAL.csv \
  --epochs 50 \
  --pretrain_epochs 10 \
  --output models/lnn_full.pth
```

**Time:** ~60-90 minutes
**Data:** 1.35M aligned bars, 394 events

### 4. Run Backtest

Test on recent data (2024):

```bash
python backtest.py \
  --model_path models/lnn_quick.pth \
  --test_year 2024 \
  --num_simulations 20
```

**Time:** ~2-3 minutes

### 5. Validate Results

```bash
python validate_results.py \
  --model_path models/lnn_quick.pth \
  --output_dir reports/
```

Check `reports/validation_report.txt` for accuracy metrics!

---

## Expected Results

### Training Output

```
======================================================================
LOADING AND PREPARING DATA
======================================================================
1. Loading SPY and TSLA data from 2023 to 2023...
   Loaded 252,000 aligned 1-minute bars

2. Extracting features (channels, RSI, correlations, cycles)...
   Extracted 56 features

3. Loading events data...
   Loaded 18 events (TSLA + macro)

4. Creating sequences for training...
   Created 1,456 sequences

======================================================================
SELF-SUPERVISED PRETRAINING
======================================================================
Epoch 1/3 - Pretraining Loss: 0.2145
Epoch 2/3 - Pretraining Loss: 0.1823
Epoch 3/3 - Pretraining Loss: 0.1567
Pretraining completed!

======================================================================
SUPERVISED TRAINING
======================================================================
Epoch 1/10 - Train Loss: 3.2451 | Val Loss: 3.0987
Epoch 5/10 - Train Loss: 1.8234 | Val Loss: 1.7456
   → New best validation loss! Saving checkpoint...
Epoch 10/10 - Train Loss: 1.2345 | Val Loss: 1.3456

Training completed! Best validation loss: 1.3456
```

### Backtest Output

```
======================================================================
BACKTEST RESULTS SUMMARY
======================================================================
Completed simulations: 20/20

Average Metrics:
  Mean Error (High): 3.12%
  Mean Error (Low): 2.89%
  Mean Absolute Error: 3.01%
  Mean Confidence: 0.68

Error by Event Type:
  With Earnings: 4.23% (3 cases)
  No Events: 2.67% (17 cases)
```

**Good results:** <5% mean absolute error
**Excellent results:** <3% mean absolute error

---

## Troubleshooting Quick Fixes

### "ModuleNotFoundError: No module named 'ncps'"

```bash
pip install ncps
```

### "FileNotFoundError: data/tsla_events_REAL.csv"

```bash
python3 process_real_events.py
python3 validate_data_alignment.py
```

### "Not enough data"

- Use a shorter date range: `--start_year 2023 --end_year 2023`
- Or reduce sequence length in `config.py`: `ML_SEQUENCE_LENGTH = 84` (half week)

### "CUDA out of memory" or "Killed"

- Reduce batch size: `--batch_size 8`
- Models run on CPU by default (no GPU needed)
- Close other applications to free RAM

### Low accuracy (>10% error)

- Train on more data: use full 2015-2023 range
- Increase epochs: `--epochs 50`
- Increase model capacity: `--hidden_size 256`
- ✅ You're using `data/tsla_events_REAL.csv` with validated data!

---

## Next Steps

### 1. Full Training

Once quick test works, run full training:

```bash
python train_model.py \
  --start_year 2015 \
  --end_year 2023 \
  --epochs 50 \
  --pretrain_epochs 10 \
  --output models/lnn_full.pth
```

### 2. Comprehensive Backtesting

```bash
python backtest.py \
  --model_path models/lnn_full.pth \
  --test_year 2024 \
  --num_simulations 100
```

### 3. Online Learning

After live predictions accumulate:

```bash
python update_model.py \
  --model_path models/lnn_full.pth \
  --output models/lnn_updated.pth
```

### 4. Integration (Coming Soon)

- Stage 1 dashboard will show ML predictions
- Telegram alerts will include ML probabilities
- Real-time inference pipeline

---

## Understanding the Output

### Training Metrics

- **Pretraining Loss:** Should decrease (0.3 → 0.1)
- **Train Loss:** Should decrease steadily
- **Val Loss:** Should decrease; if it increases, model is overfitting

### Backtest Metrics

- **Mean Error:** Average % error on predictions
  - <3%: Excellent
  - 3-5%: Good
  - 5-10%: Fair
  - >10%: Needs improvement

- **Confidence:** Model's self-assessment
  - >0.7: High confidence
  - 0.5-0.7: Medium
  - <0.5: Low (model uncertain)

### Validation Report

Check `reports/validation_report.txt`:
- Model metadata
- Accuracy metrics
- Recommendations for improvement

---

## Command Cheat Sheet

```bash
# Process real events (FIRST!)
python3 process_real_events.py

# Validate data alignment (MANDATORY!)
python3 validate_data_alignment.py --events_data data/tsla_events_REAL.csv

# Quick test (10-15 min)
python3 train_model.py --tsla_events data/tsla_events_REAL.csv --start_year 2023 --end_year 2023 --epochs 10 --output models/quick.pth

# Full training (60-90 min)
python3 train_model.py --tsla_events data/tsla_events_REAL.csv --epochs 50 --output models/full.pth

# Backtest
python3 backtest.py --model_path models/full.pth --test_year 2024 --num_simulations 100

# Validate
python3 validate_results.py --model_path models/full.pth

# Online learning
python3 update_model.py --model_path models/full.pth --output models/updated.pth
```

---

## Getting Help

1. Check `README_STAGE2.md` for detailed documentation
2. Review `SPEC.md` for technical details
3. Inspect logs for error messages
4. Verify data files exist in `data/` directory

---

**Ready to predict the future?** 🚀

Start with the quick test, then scale up to full training when you're comfortable!
