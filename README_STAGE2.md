# Stage 2: Advanced ML-Powered Predictive Model

**Version:** 2.0
**Last Updated:** November 10, 2025

---

## Overview

Stage 2 builds on the foundation of Stage 1 by implementing a robust **Liquid Neural Network (LNN)** trained on 10 years of 1-minute SPY and TSLA data. The system incorporates:

- **Multi-timeframe channel patterns** (from Stage 1)
- **RSI analysis** across timeframes
- **SPY-TSLA correlations** and divergence detection
- **Event integration** (TSLA earnings/deliveries + macro events like FOMC, CPI, NFP)
- **Self-supervised pretraining** with masked sequence reconstruction
- **Online learning** from prediction errors
- **Comprehensive backtesting** with walk-forward validation

---

## Architecture

### Modular Design

```
src/ml/
├── base.py              # Abstract interfaces (DataFeed, FeatureExtractor, etc.)
├── data_feed.py         # CSV and YFinance data loaders
├── events.py            # TSLA and macro events handlers
├── features.py          # Comprehensive feature extraction
├── model.py             # LNN and LSTM implementations
└── database.py          # Prediction logging with SQLite

Scripts:
├── train_model.py       # Initial training with self-supervised pretraining
├── backtest.py          # Walk-forward validation on holdout year
├── update_model.py      # Online learning from errors
└── validate_results.py  # Post-training analysis and reporting
```

### Key Features

1. **Plug-and-Play Components:**
   - Swap data sources (CSV → IBKR)
   - Swap models (LNN ↔ LSTM)
   - Add new event types easily

2. **Event-Aware Training:**
   - TSLA earnings, deliveries, production (user-provided CSV)
   - Macro events: FOMC, CPI, NFP, GDP, etc. (hardcoded + API extensible)
   - Learns patterns like "FOMC + Earnings beat = 3-day rebound"

3. **Prediction Database:**
   - Logs all predictions with timestamps
   - Tracks actuals vs predictions
   - Calculates errors for online learning
   - Provides accuracy metrics

---

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

New dependencies for Stage 2:
- `torch>=2.0.0` - PyTorch for deep learning
- `ncps>=0.0.1` - Liquid Neural Networks library
- `sqlalchemy>=2.0.0` - Database ORM

### 2. Create TSLA Events File

```bash
python create_sample_events.py --output data/tsla_events.csv
```

**Edit** `data/tsla_events.csv` with accurate historical data:

```csv
date,event_type,expected,actual,beat_miss
2024-01-24,earnings,0.74,0.71,miss
2024-04-02,delivery,449000,386810,miss
2024-07-23,earnings,0.60,0.52,miss
2024-10-23,earnings,0.58,0.72,beat
```

Event types: `earnings`, `delivery`, `production`
Beat/Miss: `beat`, `meet`, `miss`

---

## Usage Guide

### Step 1: Train the Model

Train on 2015-2023 data with self-supervised pretraining:

```bash
python train_model.py \
  --spy_data data/SPY_1min.csv \
  --tsla_data data/TSLA_1min.csv \
  --tsla_events data/tsla_events.csv \
  --epochs 50 \
  --pretrain_epochs 10 \
  --output models/lnn_model.pth
```

**Parameters:**
- `--start_year 2015` / `--end_year 2023` - Training period
- `--model_type LNN` or `LSTM` - Model architecture
- `--hidden_size 128` - Model capacity
- `--batch_size 32` - Training batch size
- `--lr 0.001` - Learning rate

**What it does:**
1. Loads and aligns SPY/TSLA data (inner join on timestamps)
2. Extracts 50+ features (channels, RSI, correlations, cycles)
3. Loads TSLA and macro events
4. Self-supervised pretraining (10 epochs) with masked reconstruction
5. Supervised training (50 epochs) to predict 24-hour high/low
6. Saves model to `models/lnn_model.pth`

**Expected Output:**
```
======================================================================
LOADING AND PREPARING DATA
======================================================================
1. Loading SPY and TSLA data from 2015 to 2023...
   Loaded 1,854,183 aligned 1-minute bars
2. Extracting features (channels, RSI, correlations, cycles)...
   Extracted 56 features
3. Loading events data...
   Loaded 87 events (TSLA + macro)
4. Creating sequences for training...
   Created 11,024 sequences

======================================================================
SELF-SUPERVISED PRETRAINING
======================================================================
Epoch 1/10 - Pretraining Loss: 0.1523
...
Pretraining completed!

======================================================================
SUPERVISED TRAINING
======================================================================
Epoch 1/50 - Train Loss: 2.3451 | Val Loss: 2.1987
...
   → New best validation loss! Saving checkpoint...
```

---

### Step 2: Backtest on Holdout Year

Test on 2024 data with random day/week simulation:

```bash
python backtest.py \
  --model_path models/lnn_model.pth \
  --test_year 2024 \
  --num_simulations 100 \
  --db_path data/predictions.db
```

**What it does:**
1. Loads trained model
2. Randomly selects 100 dates in 2024
3. For each date:
   - Loads 1 week of prior context
   - Makes prediction for next 24 hours
   - Compares prediction to actual high/low
   - Logs to database
4. Generates summary metrics

**Expected Output:**
```
[1/100] Simulating 2024-01-15...
   ✓ Predicted: [245.32 - 252.18]
   ✓ Actual: [247.89 - 254.31]
   ✓ Error: 1.23% | Confidence: 0.78

======================================================================
BACKTEST RESULTS SUMMARY
======================================================================
Completed simulations: 100/100

Average Metrics:
  Mean Error (High): 2.45%
  Mean Error (Low): 2.18%
  Mean Absolute Error: 2.32%
  Mean Confidence: 0.72

Error by Event Type:
  With Earnings: 3.45% (12 cases)
  With Macro Event: 2.89% (23 cases)
  No Events: 1.98% (65 cases)
```

Detailed results saved to: `models/backtest_results_2024.csv`

---

### Step 3: Validate Results

Analyze performance and generate reports:

```bash
python validate_results.py \
  --model_path models/lnn_model.pth \
  --db_path data/predictions.db \
  --output_dir reports/
```

**What it does:**
1. Loads model metadata
2. Analyzes database metrics (accuracy by confidence, event type)
3. Generates visualizations:
   - Error distribution histogram
   - Confidence vs Error scatter plot
   - Error by event type bar chart
   - Error over time line plot
4. Creates comprehensive report

**Outputs:**
- `reports/validation_report.txt` - Text summary
- `reports/error_distribution.png`
- `reports/confidence_vs_error.png`
- `reports/error_by_event.png`
- `reports/error_over_time.png`

---

### Step 4: Online Learning (Optional)

Update model incrementally from recent errors:

```bash
python update_model.py \
  --model_path models/lnn_model.pth \
  --db_path data/predictions.db \
  --output models/lnn_model_updated.pth \
  --error_threshold 10.0 \
  --epochs 5
```

**What it does:**
1. Fetches predictions with >10% error from database
2. Re-extracts features for those dates
3. Performs focused training on error samples (5 epochs)
4. Saves updated model

**When to use:**
- After backtesting reveals systematic errors
- Weekly/monthly updates with live data
- After major market events (earnings, FOMC)

---

## Feature Extraction Details

The system extracts **56 features** per timestamp:

### Price Features (10)
- `spy_close`, `tsla_close`
- Returns, log returns
- Volatility (10-bar, 50-bar)

### Channel Features (21)
For each timeframe (1h, 4h, daily):
- Position in channel (0-1)
- Distance to upper/lower (%)
- Slope, stability, ping-pongs, R²

### RSI Features (9)
For each timeframe (1h, 4h, daily):
- RSI value
- Oversold/overbought flags

### Correlation Features (5)
- SPY-TSLA correlation (10, 50, 200-bar)
- Divergence flag
- Divergence magnitude

### Cycle Features (4)
- Distance from 52-week high/low
- Within mega-channel (3-4 year)
- Mega-channel position

### Volume Features (2)
- TSLA/SPY volume ratio vs 20-bar average

### Time Features (4)
- Hour of day, day of week, day of month, month of year (cyclical encoding)

### Event Embeddings (21)
- TSLA events: one-hot type (4), days until, beat/miss, surprise magnitude
- Macro events: one-hot type (10), days until

---

## Database Schema

**Predictions Table:**

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| timestamp | DateTime | When prediction was made |
| target_timestamp | DateTime | When prediction is for |
| symbol | String | TSLA or SPY |
| timeframe | String | 1h, 4h, daily, 24h |
| predicted_high | Float | Predicted high |
| predicted_low | Float | Predicted low |
| predicted_center | Float | Midpoint |
| confidence | Float | 0-1 confidence score |
| actual_high | Float | Actual high (filled later) |
| actual_low | Float | Actual low (filled later) |
| error_high | Float | % error on high |
| error_low | Float | % error on low |
| absolute_error | Float | Average error |
| has_earnings | Boolean | TSLA event flag |
| has_macro_event | Boolean | Macro event flag |
| model_version | String | Model identifier |

**Usage:**

```python
from src.ml.database import SQLitePredictionDB

db = SQLitePredictionDB('data/predictions.db')

# Log prediction
pred_id = db.log_prediction({
    'predicted_high': 250.5,
    'predicted_low': 245.2,
    'confidence': 0.85,
    ...
})

# Update with actuals
db.update_actual(pred_id, actual_high=252.1, actual_low=246.8)

# Get metrics
metrics = db.get_accuracy_metrics()
print(f"Mean error: {metrics['mean_absolute_error']:.2f}%")
```

---

## Configuration

All settings in `config.py`:

```python
# Model Settings
ML_MODEL_TYPE = "LNN"  # or "LSTM"
LNN_HIDDEN_SIZE = 128
LNN_LEARNING_RATE = 0.001
ML_BATCH_SIZE = 32
ML_SEQUENCE_LENGTH = 168  # 1 week of hourly bars

# Training Settings
ML_TRAIN_START_YEAR = 2015
ML_TRAIN_END_YEAR = 2023
ML_TEST_YEAR = 2024
ML_EPOCHS = 50

# Event Settings
TSLA_EVENTS_FILE = DATA_DIR / "tsla_events.csv"
EVENT_LOOKBACK_DAYS = 7

# Prediction Settings
PREDICTION_HORIZON_HOURS = 24
PREDICTION_CONFIDENCE_THRESHOLD = 0.7

# Online Learning
ONLINE_LEARNING_ENABLED = True
ONLINE_LEARNING_LR = 0.0001
```

---

## Integration with Stage 1

*(Coming Soon)*

The ML predictions will be integrated into the Stage 1 dashboard:

1. **Enhanced Signal Panel:**
   - Shows ML predicted high/low for next 24 hours
   - Displays confidence score
   - Highlights event-aware predictions

2. **Telegram Alerts:**
   - Includes ML probabilities: "80% chance of break to $250 in 24h"
   - Factors in upcoming events

3. **Real-Time Updates:**
   - Model inference every 4 hours (configurable)
   - Logs predictions to database automatically

---

## Troubleshooting

### "Data file not found"
- Run `python convert_data.py` to create CSV files
- Ensure `data/SPY_1min.csv` and `data/TSLA_1min.csv` exist

### "Events file not found"
- Run `python create_sample_events.py`
- Edit `data/tsla_events.csv` with real data

### "Not enough data for training"
- Sequence length (168) + prediction horizon (24) requires at least 192 bars
- Check date ranges align with available data

### "CUDA out of memory"
- Reduce `--batch_size` (try 16 or 8)
- Use CPU: model runs on CPU by default

### "Model accuracy too low (>15% error)"
- Run more pretraining epochs (`--pretrain_epochs 20`)
- Increase model capacity (`--hidden_size 256`)
- Check data quality (nulls, zeros, alignment)
- Verify events file is accurate

---

## Performance Benchmarks

**Training (on 2015-2023 data):**
- Data loading: ~2 minutes
- Feature extraction: ~3 minutes
- Pretraining (10 epochs): ~15 minutes
- Supervised training (50 epochs): ~45 minutes
- **Total:** ~65 minutes

**Backtesting (100 simulations):**
- ~10 minutes

**Inference (single prediction):**
- <1 second

**Hardware:**
- CPU: M1/M2 Mac or equivalent
- RAM: 8GB minimum, 16GB recommended
- Disk: 5GB for data + models

---

## Advanced Usage

### Custom Feature Engineering

Add new features in `src/ml/features.py`:

```python
def _extract_custom_features(self, df, features_df):
    # Your custom indicator
    features_df['my_indicator'] = calculate_my_indicator(df)
    return features_df
```

Update `_build_feature_names()` to include new feature names.

### Custom Events

Add new event handler in `src/ml/events.py`:

```python
class CustomEventsHandler(EventHandler):
    def load_events(self, start_date, end_date):
        # Load from API or file
        pass
```

### Model Swapping

Switch to LSTM:

```bash
python train_model.py --model_type LSTM ...
```

Or create custom model by subclassing `ModelBase` in `src/ml/base.py`.

---

## Roadmap

### Completed ✅
- [x] Modular architecture with abstract interfaces
- [x] LNN model with ncps library
- [x] Self-supervised pretraining
- [x] Event integration (TSLA + macro)
- [x] Comprehensive feature extraction
- [x] Prediction database with SQLite
- [x] Training, backtesting, online learning scripts
- [x] Validation and reporting

### Upcoming 🚀
- [ ] Integration with Stage 1 dashboard
- [ ] Telegram alerts with ML probabilities
- [ ] Real-time inference pipeline
- [ ] IBKR data feed integration
- [ ] PostgreSQL database option
- [ ] Hyperparameter tuning with Optuna
- [ ] Ensemble models (LNN + LSTM + Transformer)
- [ ] Attention mechanisms for event weighting
- [ ] Multi-stock support (add NVDA, AAPL, etc.)

---

## License

Personal trading tool - Use at your own risk. Not financial advice.

---

## Credits

Built with:
- **ncps** - Neural Circuit Policies library for LNN
- **PyTorch** - Deep learning framework
- **SQLAlchemy** - Database ORM
- **Stage 1** - Foundation for features and indicators

---

**End of Stage 2 Documentation**
