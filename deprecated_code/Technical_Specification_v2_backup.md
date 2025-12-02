# Technical Specification: Adaptive Channel Prediction System v2.0
*Last Updated: November 17, 2024*

## 1. Executive Summary

Our trading system has evolved into a sophisticated **Adaptive Channel Prediction System** that dynamically analyzes market structure across multiple timeframes to predict optimal entry/exit points with variable time horizons. The system uses a hierarchical neural network that "reads the ocean layers" of market data - from fast intraday ripples to slow macro tides - to make intelligent predictions.

### Core Analogy: Reading the Ocean Layers
Just as an oceanographer studies different water layers to understand currents, our system analyzes market timeframes as "layers":
- **Fast Layer**: Intraday ripples (hours) - RSI warnings and short-term volatility
- **Medium Layer**: Swings (days) - Channel alignment and trend confirmation
- **Slow Layer**: Macro tides (weeks+) - Long-term support/resistance and fundamental drivers

The model dynamically selects the most confident layer and projects forward accordingly, using higher layers as confirmation for longer holds.

## 2. System Architecture

### 2.1 Current File Structure (Post-Cleanup)
```
autotrade2/
├── train_hierarchical.py           # Main training script
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
│   ├── tsla_events_REAL.csv    # Event calendar
│   └── feature_cache/           # Cached feature calculations
├── models/                      # Saved model checkpoints
└── deprecated/                  # Legacy code (50+ old files)
```

### 2.2 Component Descriptions

#### Training Engine (`train_hierarchical.py`)
- **Purpose**: Orchestrates model training with adaptive projection
- **Key Features**:
  - Multi-task learning (price, volatility, continuation, horizons)
  - Dynamic batch sizing based on available memory
  - Automatic device selection (CUDA > MPS > CPU)
  - Progress tracking with clean terminal output
  - Checkpoint saving with early stopping

#### Dashboard (`hierarchical_dashboard.py`)
- **Purpose**: Real-time prediction visualization
- **Key Features**:
  - Layer confidence "debate" display
  - Adaptive time horizon visualization
  - Interactive price charts with projections
  - Channel overlay visualization
  - Auto-refresh every 30 minutes

#### Feature Extraction (`src/ml/features.py`)
- **Purpose**: Extract 495+ technical indicators
- **Components**:
  - Channel features across 11 timeframes
  - RSI calculations with multiple periods
  - Continuation label generation
  - Volume-weighted features
  - Cross-correlation metrics

## 3. Technical Implementation

### 3.1 Model Architecture

#### Hierarchical LNN Structure
```python
Layer Configuration:
- Fast Layer: 64 liquid neurons (5-min to 1-hour patterns)
- Medium Layer: 128 liquid neurons (4-hour to daily patterns)
- Slow Layer: 256 liquid neurons (weekly+ patterns)

Multi-Head Outputs:
1. Price Prediction Head (next close price)
2. Volatility Band Head (expected high/low range)
3. Continuation Head (trend continuation probability)
4. Adaptive Horizon Head (optimal holding period)

Internal Architecture:
- Liquid neurons with learnable time constants
- Skip connections between layers
- Attention mechanism for layer weighting
- Dropout: 0.2 for regularization
```

### 3.2 Feature Engineering (495 Features)

#### Channel Features (220 features)
- **Timeframes**: 5min, 15min, 30min, 1h, 2h, 3h, 4h, daily, weekly, monthly, 3-month
- **Per Timeframe** (10 metrics × 2 stocks):
  - Channel position (0-1)
  - Channel slope (normalized)
  - Channel R-squared
  - Ping-pong counts (3 thresholds)
  - Distance to upper/lower bands
  - Channel width (volatility)

#### RSI Features (44 features)
- **Periods**: 14, 21, 50
- **Calculations**:
  - Raw RSI value
  - RSI slope
  - Divergence indicators
  - Oversold/overbought flags

#### Continuation Labels (Dynamic)
```python
Scoring Algorithm:
- Pull 1h and 4h OHLC chunks
- Calculate RSI for both timeframes
- Check slope alignment
- Apply scoring:
  +1: RSI < 40 on 1h (room to run)
  +1: RSI < 40 on 4h (broader support)
  +1: Slopes align (both bullish/bearish)
  +2: Strong channel support detected
- Look ahead for actual continuation validation
```

#### Additional Features (231 features)
- Volume indicators
- Price momentum
- Correlation metrics (SPY vs TSLA)
- Time-based features (hour, day, month effects)
- Event proximity indicators

### 3.3 Data Pipeline

#### Input Requirements
```yaml
Data Format: CSV with columns [timestamp, open, high, low, close, volume]
Minimum History: 2 years (for continuation analysis)
Sampling Rate: 1-minute bars
Alignment: SPY and TSLA data must overlap temporally
```

#### Processing Pipeline
1. **Load & Validate**: Check for missing data, outliers
2. **Historical Buffer**: Add 2-year lookback for continuation
3. **Feature Extraction**: Calculate all 495+ features
4. **Caching**: Store computed features for reuse
5. **Batching**: Create sequences of 200 bars
6. **Normalization**: Z-score normalization per feature

## 4. Installation & Setup

### 4.1 Environment Setup
```bash
# Create virtual environment
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install dependencies
pip install torch torchvision pandas numpy streamlit plotly
pip install tqdm psutil pyyaml scikit-learn
```

### 4.2 Data Preparation
```bash
# Expected data structure
data/
├── TSLA_1min.csv    # Columns: timestamp,open,high,low,close,volume
├── SPY_1min.csv     # Same format as TSLA
└── tsla_events_REAL.csv  # Optional: event calendar

# Verify data
python test_data_loading.py
```

### 4.3 Configuration
Edit `config.py` for customization:
```python
# Key settings
DATA_DIR = "data"
ML_BATCH_SIZE = 32  # Adjust based on memory
ML_TRAIN_START_YEAR = 2015
ML_TRAIN_END_YEAR = 2022
```

## 5. Usage Guide

### 5.1 Training

#### Quick Start (1 epoch test)
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
    --patience 10
```

#### Training Parameters
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32, reduce if OOM)
- `--device`: Device selection (auto/cuda/mps/cpu)
- `--lr`: Learning rate (default: 0.001)
- `--patience`: Early stopping patience (default: 10)
- `--preload`: Preload all data to memory (faster but uses more RAM)
- `--multi_task`: Enable multi-task learning (recommended)
- `--interactive`: Interactive parameter selection

### 5.2 Running the Dashboard
```bash
streamlit run hierarchical_dashboard.py
```
Then open http://localhost:8501 in your browser.

### 5.3 Backtesting
```bash
python backtest_hierarchical.py --year 2024 --model models/hierarchical_lnn.pth
```

## 6. Known Issues & Solutions

### 6.1 Continuation Labels Bug (FIXED in v2.0)
**Issue**: KeyError: 'close' when generating continuation labels
**Cause**: Duplicate resampling code with incorrect column references
**Solution**: Removed duplicate code block at `features.py:1575-1600`

### 6.2 Memory Issues
**Issue**: Out of memory during feature extraction
**Solutions**:
- Use lazy loading mode (default)
- Reduce batch size
- Clear cache: `rm -rf data/feature_cache/`

### 6.3 Slow Feature Extraction
**Issue**: First run takes ~55 minutes
**Solution**: Features are cached after first extraction. Subsequent runs are instant.

## 7. Performance Optimization

### 7.1 Hardware Acceleration
```python
Device Priority:
1. CUDA (NVIDIA GPU) - Fastest
2. MPS (Apple Silicon) - Fast for M1/M2/M3
3. CPU - Fallback, slower but works everywhere

Auto-detection:
python train_hierarchical.py --device auto
```

### 7.2 Memory Management
- **Lazy Loading**: Default mode, loads data as needed
- **Feature Caching**: Automatic, saves ~55 minutes on subsequent runs
- **Batch Size Tuning**: Start with 32, increase if memory allows

### 7.3 Training Speed
- Initial feature extraction: ~55 minutes (first time only)
- Per epoch: ~2-5 minutes (depends on device)
- Total training (100 epochs): ~4-8 hours

## 8. Testing & Validation

### 8.1 Data Validation
```bash
# Test data loading
python test_data_loading.py

# Validate features
python test_continuation_fix.py
```

### 8.2 Model Validation
- **Train/Val Split**: 90/10 by default
- **Metrics**: MSE for price, accuracy for continuation
- **Backtesting**: Separate test on 2023-2024 data

### 8.3 Performance Benchmarks
- **Baseline (Linear Regression)**: ~52% directional accuracy
- **Target (Hierarchical LNN)**: >65% directional accuracy
- **Inference Speed**: <100ms per prediction

## 9. Troubleshooting Guide

### Common Issues

#### "No data loaded"
- Check data files exist in `data/` directory
- Verify CSV format matches requirements
- Check date ranges in your command

#### "CUDA out of memory"
```bash
# Reduce batch size
python train_hierarchical.py --batch_size 16

# Or use CPU
python train_hierarchical.py --device cpu
```

#### "Module not found"
```bash
# Ensure you're in project root
cd /path/to/autotrade2

# Activate virtual environment
source myenv/bin/activate
```

#### Progress bars not showing
```bash
# Use unbuffered output
python -u train_hierarchical.py
```

## 10. API Reference

### Model Prediction Interface
```python
from src.ml.hierarchical_model import load_hierarchical_model

# Load model
model = load_hierarchical_model('models/hierarchical_lnn.pth')

# Make prediction
features = torch.tensor(...)  # Shape: [batch, sequence, features]
price, volatility, continuation, horizon = model(features)
```

### Data Feed Interface
```python
from src.ml.data_feed import CSVDataFeed

# Initialize
feed = CSVDataFeed(timeframe='1min')

# Load data
df = feed.load_data('TSLA', start_date='2024-01-01', end_date='2024-01-31')
```

## 11. Future Enhancements

### Planned Features (v3.0)
- **News Sentiment Integration**: Real-time news analysis via NLP
- **Options Flow Analysis**: Integrate unusual options activity
- **Multi-Asset Support**: Extend beyond TSLA/SPY
- **Live Trading Interface**: Connect to broker APIs
- **Ensemble Methods**: Combine multiple models for voting

### Research Directions
- Transformer architecture experiments
- Reinforcement learning for position sizing
- Graph neural networks for market structure
- Federated learning for privacy-preserving updates

## 12. Appendix

### A. Feature Cache Structure
```
data/feature_cache/
├── channels_TSLA_5min_HASH.pkl
├── channels_TSLA_15min_HASH.pkl
├── continuation_labels_HASH.pkl
└── features_complete_HASH.pkl
```

### B. Model Checkpoint Format
```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
    'val_metrics': {...},
    'config': {...}
}
```

### C. Event Data Format
```csv
date,type,importance,description
2024-01-24,earnings,high,Q4 2023 Earnings
2024-02-15,deliveries,medium,January Deliveries
```

## Version History
- **v2.0** (Nov 17, 2024): Major cleanup, fixed continuation labels bug
- **v1.0** (Nov 10, 2024): Initial hierarchical implementation

---
*For questions or issues, please refer to the troubleshooting guide or create an issue in the project repository.*