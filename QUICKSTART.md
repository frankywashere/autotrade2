# AutoTrade2 - Quick Start Guide

**Simple step-by-step commands to train the multi-scale LNN system.**

Download the official Python 3.12 installer from the Python website: https://www.python.org/downloads/release/python-3127/ (or the latest 3.12.x release). Choose the macOS installer for your architecture.
Run the installer—it will place Python 3.12 in /Library/Frameworks/Python.framework/Versions/3.12/.
Create a new virtual environment using Python 3.12:text/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv myenv
source myenv/bin/activate

Running Scripts or Commands Without Activation: If you don't want to activate every time, you can run things directly using the full path to the venv's Python, like /path/to/myenv/bin/python your_script.py or /path/to/myenv/bin/pip install something. This works from any terminal without activation.

---

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

**Required:**
- Python 3.8+
- 8GB+ RAM (16GB recommended)
- GPU optional (10-20x faster training)

**Data:**
- TSLA_1min.csv (~93 MB)
- SPY_1min.csv (~109 MB)
- tsla_events_REAL.csv (~37 KB)

All included in `data/` directory.

---

## Step 1: Generate Multi-Scale CSVs (One-Time Setup)

```bash
python3 scripts/create_multiscale_csvs.py
```

**What it does:**
- Resamples 1-minute data to 11 timeframes
- Creates: 5min, 15min, 30min, 1h, 2h, 3h, 4h, daily, weekly, monthly, 3month
- Output: ~22 CSV files, ~500MB total
- **Time: ~5 minutes**

---

## Step 2: Train Sub-Models (4 Specialized LNNs)

**Option A: Train all 4 models in parallel** (if you have GPU)

```bash
# Run all 4 in parallel (GPU recommended)
python train_model_lazy.py --input_timeframe 15min --sequence_length 200 --epochs 50 --batch_size 128 --device cuda --output models/lnn_15min.pth &
python train_model_lazy.py --input_timeframe 1hour --sequence_length 200 --epochs 50 --batch_size 128 --device cuda --output models/lnn_1hour.pth &
python train_model_lazy.py --input_timeframe 4hour --sequence_length 200 --epochs 50 --batch_size 128 --device cuda --output models/lnn_4hour.pth &
python train_model_lazy.py --input_timeframe daily --sequence_length 200 --epochs 50 --batch_size 128 --device cuda --output models/lnn_daily.pth &
wait
```

**Time: ~15-25 minutes each on T4 GPU (can run in parallel)**

**Option B: Train one at a time** (if limited resources)

```bash
# 15-minute model
python train_model_lazy.py --input_timeframe 15min --sequence_length 200 --epochs 50 --batch_size 128 --device cuda --output models/lnn_15min.pth

# 1-hour model
python train_model_lazy.py --input_timeframe 1hour --sequence_length 200 --epochs 50 --batch_size 128 --device cuda --output models/lnn_1hour.pth

# 4-hour model
python train_model_lazy.py --input_timeframe 4hour --sequence_length 200 --epochs 50 --batch_size 128 --device cuda --output models/lnn_4hour.pth

# Daily model
python train_model_lazy.py --input_timeframe daily --sequence_length 200 --epochs 50 --batch_size 128 --device cuda --output models/lnn_daily.pth
```

**Option C: Interactive mode** (menu-based configuration)

```bash
python train_model_lazy.py --interactive
```

Then select:
- Parameter 1: Input timeframe → Choose: 15min, 1hour, 4hour, or daily
- Parameter 9: Sequence length → Keep default 200
- Parameter 11: Batch size → Set 128 for GPU
- Parameter 21: Device → Select cuda
- Run training, then repeat for other timeframes

---

## Step 3: Collect Predictions (Backtest Sub-Models)

```bash
# Backtest each model to populate predictions database
python backtest.py --model_path models/lnn_15min.pth --test_year 2023 --num_simulations 500
python backtest.py --model_path models/lnn_1hour.pth --test_year 2023 --num_simulations 500
python backtest.py --model_path models/lnn_4hour.pth --test_year 2023 --num_simulations 500
python backtest.py --model_path models/lnn_daily.pth --test_year 2023 --num_simulations 500
```

**What it does:**
- Runs each model on random dates in 2023
- Logs predictions + actuals to `data/predictions.db`
- Needed for training Meta-LNN coach
- **Time: ~30-60 minutes total**

---

## Step 4: Train Meta-LNN Coach

```bash
python train_meta_lnn.py --mode backtest_no_news --epochs 100 --output models/meta_lnn.pth
```

**What it does:**
- Loads all sub-model predictions from database
- Trains Meta-LNN to combine them adaptively
- Uses purged K-fold CV (prevents leakage)
- Learns market-regime-dependent weighting
- **Time: ~10-15 minutes**

---

## Step 5: Validate Performance

**Test individual models:**

```bash
# Test each model on 2024 holdout
python backtest.py --model_path models/lnn_15min.pth --test_year 2024 --num_simulations 100
python backtest.py --model_path models/lnn_1hour.pth --test_year 2024 --num_simulations 100
python backtest.py --model_path models/lnn_4hour.pth --test_year 2024 --num_simulations 100
python backtest.py --model_path models/lnn_daily.pth --test_year 2024 --num_simulations 100
```

**Analyze results:**
```bash
# Compare which timeframe is most accurate
sqlite3 data/predictions.db "SELECT model_timeframe, AVG(absolute_error) FROM predictions GROUP BY model_timeframe"
```

---

## Optional: Enable News Mode (Future)

### Install transformers:
```bash
pip install transformers datasets
```

### Fetch news and store:
```bash
python -m src.ml.fetch_news --tsla_max 20 --market_max 20
```

### Fine-tune Meta-LNN with news:
```bash
python train_meta_lnn.py --mode live_with_news --resume models/meta_lnn.pth --epochs 50
```

### Use in live predictions:
```bash
python backtest.py --ensemble --mode live_with_news --test_year 2024
```

---

## Troubleshooting

### "CUDA out of memory"
Reduce batch size:
```bash
python train_model_lazy.py --input_timeframe 1hour --batch_size 64 ...
```

### "No such file: data/TSLA_15min.csv"
Run CSV generation first:
```bash
python scripts/create_multiscale_csvs.py
```

### "Module 'ncps' not found"
Install ML dependencies:
```bash
pip install torch ncps sqlalchemy tqdm
```

### "Not enough predictions in database"
Run backtests to collect predictions (Step 3)

---

## Quick Commands Summary

```bash
# Full pipeline (copy-paste ready)
python scripts/create_multiscale_csvs.py
python train_model_lazy.py --input_timeframe 15min --sequence_length 200 --epochs 50 --batch_size 128 --device cuda --output models/lnn_15min.pth
python train_model_lazy.py --input_timeframe 1hour --sequence_length 200 --epochs 50 --batch_size 128 --device cuda --output models/lnn_1hour.pth
python train_model_lazy.py --input_timeframe 4hour --sequence_length 200 --epochs 50 --batch_size 128 --device cuda --output models/lnn_4hour.pth
python train_model_lazy.py --input_timeframe daily --sequence_length 200 --epochs 50 --batch_size 128 --device cuda --output models/lnn_daily.pth
python backtest.py --model_path models/lnn_15min.pth --test_year 2023 --num_simulations 500
python backtest.py --model_path models/lnn_1hour.pth --test_year 2023 --num_simulations 500
python backtest.py --model_path models/lnn_4hour.pth --test_year 2023 --num_simulations 500
python backtest.py --model_path models/lnn_daily.pth --test_year 2023 --num_simulations 500
python train_meta_lnn.py --mode backtest_no_news --epochs 100 --output models/meta_lnn.pth
```

---

For complete technical details, see **SPEC.md**.
