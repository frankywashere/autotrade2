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

**Option A: Bash script** (EASIEST - runs all 4 sequentially)

```bash
./train_all_models.sh
```

**What it does:**
- Trains all 4 models (15min, 1hour, 4hour, daily) automatically
- Uses defaults: epochs=50, batch_size=128, sequence_length=200, device=cuda
- **Time: ~60-100 minutes on T4 GPU (sequential)**

**Customize:**
```bash
EPOCHS=100 BATCH_SIZE=256 DEVICE=mps ./train_all_models.sh
```

**Option B: Interactive "Train All 4"** (menu-based, fully functional)

```bash
python train_model_lazy.py --interactive
```

Then:
1. **Select training mode: 2** (All 4 models)
2. **Configure parameters once** via interactive menu:
   - Sequence length: 200 (or customize)
   - Epochs: 50 (or customize)
   - Batch size: 128 for GPU
   - Device: cuda
   - (All 24 parameters)
3. **Confirm** configuration
4. **All 4 models train automatically** (15min → 1hour → 4hour → daily)
5. **Each model saves with correct metadata** (timeframe + your settings)

**Time: ~60-100 minutes on GPU (sequential)**

**Metadata verified:** All your settings (epochs, batch_size, sequence_length, etc.) flow correctly to all 4 models!

**Option C: Manual CLI** (full control, one at a time)

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

**Recommendation: Use Option A (bash script) for simplicity!**

---

## Step 3: Collect Predictions (Backtest Sub-Models)

**Option A: Auto-backtest all models** (EASIEST)

```bash
python backtest_all_models.py --test_year 2023 --num_simulations 500
```

**What it does:**
- Auto-finds all trained models in models/ directory
- Backtests each sequentially
- Logs predictions + actuals to `data/predictions.db`
- Shows comparison summary at end
- **Time: ~30-60 minutes total**

**Option B: Manual (one-by-one)**

```bash
# Backtest each model individually
python backtest.py --model_path models/lnn_15min.pth --test_year 2023 --num_simulations 500
python backtest.py --model_path models/lnn_1hour.pth --test_year 2023 --num_simulations 500
python backtest.py --model_path models/lnn_4hour.pth --test_year 2023 --num_simulations 500
python backtest.py --model_path models/lnn_daily.pth --test_year 2023 --num_simulations 500
```

**Recommendation: Use Option A (auto-backtest script)!**

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

## Step 6: Test Ensemble (Meta-LNN Coach)

```bash
# Test ensemble on 2024 holdout
python3 ensemble_backtest.py --test_year 2024 --num_simulations 50
```

**What it does:**
- Tests Meta-LNN coach combining all 4 models
- Compares ensemble vs best individual model
- Logs to database with model_timeframe='ensemble'
- **Time: ~15-20 minutes**

**Analyze ensemble vs individual:**
```bash
sqlite3 data/predictions.db "
  SELECT
    model_timeframe,
    COUNT(*) as n,
    ROUND(AVG(absolute_error), 2) as error
  FROM predictions
  WHERE simulation_date >= '2024-01-01'
  GROUP BY model_timeframe
  ORDER BY error ASC
"
```

---

## Step 7: Deploy Live Trading System

```bash
# Launch ML dashboard (all-in-one: predictions + alerts + monitoring)
streamlit run ml_dashboard.py
```

**What it does:**
- Opens browser dashboard at http://localhost:8501
- Select models to display (checkboxes in sidebar)
- Fetches live data via yfinance
- Makes predictions at correct bar-close times:
  * 15min: Updates every 15 minutes (:00, :15, :30, :45)
  * 1hour: Updates hourly (top of hour)
  * 4hour: Updates every 4 hours
  * daily: Updates at market close (4pm ET)
  * ensemble: Updates hourly
- Sends Telegram alerts for high-confidence predictions
- Shows countdown timers to next update
- Displays recent 30-day performance

**Optional: Setup actuals updater (cron job)**
```bash
crontab -e
# Add this line:
0 * * * * cd /path/to/autotrade2 && /path/to/myenv/bin/python3 update_live_actuals.py
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

## Quick Commands Summary (EASIEST METHOD)

```bash
# Full pipeline using helper scripts (copy-paste ready)

# Step 1: Generate CSVs
python3 scripts/create_multiscale_csvs.py

# Step 2: Train all 4 models
./train_all_models.sh

# Step 3: Backtest all models
python3 backtest_all_models.py --test_year 2023 --num_simulations 500

# Step 4: Train Meta-LNN coach
python3 train_meta_lnn.py --mode backtest_no_news --epochs 100 --output models/meta_lnn.pth

# Step 5: Validate on 2024
python3 backtest_all_models.py --test_year 2024 --num_simulations 100
```

**Total time: ~2-3 hours end-to-end on T4 GPU**

---

For complete technical details, see **SPEC.md**.
