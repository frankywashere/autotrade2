# Channel Prediction System v7 - Architecture Specification

## Overview

A pattern recognition system that predicts:
1. **Duration**: How many bars until the current channel breaks
2. **Direction**: Which direction price will break (up/down)
3. **Next Channel**: What type of channel follows (bull/bear/sideways)

## Core Concepts

### What is a Channel?
A price channel is a linear regression trend with upper/lower bounds (±2σ) where price "bounces" between the boundaries. A valid channel has at least 1 complete cycle (lower→upper→lower or upper→lower→upper).

### The Prediction Task
When we detect a valid channel, we want to predict:
- How long will it last? (duration in bars)
- When it breaks, which direction?
- What patterns led to this outcome?

## Data Sources

### Primary Data (5-min base resolution)
| Source | Resolution | Fields |
|--------|------------|--------|
| TSLA | 5-min OHLCV | open, high, low, close, volume |
| SPY | 5-min OHLCV | open, high, low, close, volume |
| VIX | Daily | open, high, low, close |

### Events Data
See: `/docs/EVENTS_IMPLEMENTATION_PLAN.md`
- TSLA Earnings
- TSLA Delivery Reports
- FOMC Decisions
- CPI Releases
- NFP Releases
- Quad Witching

## Timeframes

11 hierarchical timeframes, all derived from 5-min base:
```
5min → 15min → 30min → 1h → 2h → 3h → 4h → daily → weekly → monthly → 3month
```

Each timeframe:
- Has its own channel detection
- Checks containment against ALL longer timeframes
- Tracks its own bounce patterns and RSI

## Channel Detection Algorithm

### Step 1: Fit Linear Regression
```python
# For a window of N bars:
x = [0, 1, 2, ..., N-1]
y = close_prices

slope, intercept = linear_regression(x, y)
center_line = slope * x + intercept

residuals = close_prices - center_line
std_dev = std(residuals)
```

### Step 2: Calculate Bounds
```python
upper_line = center_line + 2.0 * std_dev
lower_line = center_line - 2.0 * std_dev
```

### Step 3: Detect Bounces
```python
# Key insight: Use HIGHS for upper touches, LOWS for lower touches
for each bar:
    if HIGH is within 10% of channel_width from upper_line:
        touch = 'upper'
    elif LOW is within 10% of channel_width from lower_line:
        touch = 'lower'
```

### Step 4: Count Complete Cycles
```python
# A cycle is a full round-trip:
# lower → upper → lower  OR  upper → lower → upper
touches = [lower, upper, lower, upper, lower]  # Example
cycles = 2  # (L→U→L) and (U→L→U)
```

### Step 5: Validate Channel
```python
is_valid = complete_cycles >= 1
direction = BULL if slope > 0.05% else BEAR if slope < -0.05% else SIDEWAYS
```

## Feature Extraction

### Per-Bar Features (for each symbol, each timeframe)

#### Channel State Features
```
channel_valid          # 0 or 1
channel_direction      # 0=bear, 1=sideways, 2=bull
channel_position       # 0.0 (at lower) to 1.0 (at upper)
channel_upper_dist     # % distance to upper bound
channel_lower_dist     # % distance to lower bound
channel_width_pct      # channel width as % of price
channel_slope_pct      # slope as % per bar
channel_r_squared      # linear fit quality
channel_bounce_count   # bounces so far
channel_bars_since_bounce  # bars since last touch
channel_last_touch     # 0=lower, 1=upper
channel_duration       # bars in current channel
```

#### RSI Features
```
rsi_14                 # Standard 14-period RSI
rsi_at_last_upper      # RSI value when price last touched upper
rsi_at_last_lower      # RSI value when price last touched lower
rsi_divergence         # Price making new high but RSI lower? (or vice versa)
```

#### Multi-TF Containment Features
For each timeframe, check against ALL longer timeframes:
```
# Example for 5min checking against 1h:
near_1h_upper          # Is 5min price within 10% of 1h upper bound?
near_1h_lower          # Is 5min price within 10% of 1h lower bound?
dist_to_1h_upper       # % distance to 1h upper
dist_to_1h_lower       # % distance to 1h lower
```

### Channel History Features
```
last_5_directions      # [bull, bull, bear, sideways, bear]
last_5_durations       # [45, 23, 67, 12, 89] bars
last_5_break_dirs      # [up, up, down, down, up]
avg_duration_recent    # mean of last 5 durations
direction_streak       # consecutive same-direction channels
bounce_count_trend     # are channels getting more/fewer bounces?
```

### Cross-Asset Features
```
# TSLA vs SPY alignment
tsla_spy_dir_match     # Both same direction?
spy_channel_direction  # SPY's current channel
spy_channel_position   # Where is SPY in its channel?
spy_near_boundary      # Is SPY near upper/lower?

# VIX features
vix_level              # Current VIX (fear gauge)
vix_trend              # Rising/falling over last N days
vix_percentile         # Where is VIX in its historical range?
```

### Event Features
See `/docs/EVENTS_IMPLEMENTATION_PLAN.md` for full specification:
```
days_until_earnings
days_until_fomc
days_until_cpi
is_high_impact_period
pre_event_drift
post_event_drift
...
```

## Label Generation

### Duration Label
```python
# Scan forward from current bar until channel breaks
for future_bar in range(1, MAX_SCAN):
    if price_outside_channel(future_bar):
        duration_label = future_bar
        break
```

### Direction Label
```python
# When channel breaks, which way did it go?
if break_price > upper_line:
    direction_label = UP
elif break_price < lower_line:
    direction_label = DOWN
```

### Transition Label
```python
# What type of channel followed?
next_channel = detect_next_channel(after_break)
transition_label = next_channel.direction  # bull/bear/sideways
```

## Model Architecture

### Input: Feature Sequences
- Last N bars of features for each timeframe
- Shape: `[batch, timeframes, sequence_length, features]`

### Network: Hierarchical CfC (Liquid Neural Network)
```
11 parallel CfC layers (one per timeframe)
    ↓
Cross-timeframe attention
    ↓
Duration head (regression)
Direction head (classification)
Transition head (classification)
```

### Output
```python
{
    'duration_mean': float,      # Predicted bars until break
    'duration_std': float,       # Uncertainty
    'direction_prob': [up, down],  # Break direction probabilities
    'transition_prob': [bull, bear, sideways],  # Next channel type
    'confidence': float          # Overall prediction confidence
}
```

## Directory Structure

```
v7/
├── core/
│   ├── __init__.py
│   ├── channel.py          # Channel detection algorithm
│   ├── bounce.py           # Bounce detection
│   └── timeframe.py        # Timeframe resampling
├── features/
│   ├── __init__.py
│   ├── channel_features.py # Per-bar channel features
│   ├── rsi.py              # RSI calculations
│   ├── containment.py      # Multi-TF containment
│   ├── history.py          # Channel history features
│   ├── cross_asset.py      # TSLA/SPY/VIX correlations
│   └── events.py           # Event features
├── models/
│   ├── __init__.py
│   ├── cfc_layer.py        # Single CfC layer
│   ├── hierarchical.py     # Full hierarchical model
│   └── heads.py            # Duration/Direction/Transition heads
├── training/
│   ├── __init__.py
│   ├── dataset.py          # PyTorch dataset
│   ├── labels.py           # Label generation
│   ├── trainer.py          # Training loop
│   └── losses.py           # Loss functions
├── data/
│   └── (symlinks to main data/)
├── tools/
│   ├── visualize.py        # Channel visualization
│   └── inspect.py          # Data inspection
└── docs/
    ├── ARCHITECTURE.md     # This file
    └── EVENTS.md           # Events specification
```

## Configuration

```python
# config.py

# Timeframes
TIMEFRAMES = ['5min', '15min', '30min', '1h', '2h', '3h', '4h',
              'daily', 'weekly', 'monthly', '3month']

# Channel detection
CHANNEL_STD_MULTIPLIER = 2.0     # ±2σ for bounds
CHANNEL_TOUCH_THRESHOLD = 0.10   # 10% of channel width
MIN_COMPLETE_CYCLES = 1          # Minimum for valid channel

# Window sizes to try
WINDOW_SIZES = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]

# RSI
RSI_PERIOD = 14

# Sequence lengths for model input
SEQUENCE_LENGTHS = {
    '5min': 300,    # ~1 day
    '15min': 200,   # ~2 days
    '30min': 150,   # ~3 days
    '1h': 120,      # ~5 days
    '2h': 100,      # ~8 days
    '3h': 80,       # ~10 days
    '4h': 60,       # ~10 days
    'daily': 252,   # 1 year
    'weekly': 52,   # 1 year
    'monthly': 24,  # 2 years
    '3month': 16,   # 4 years
}
```

## Training Strategy

1. **Data Split**
   - Train: 2015-2022
   - Validation: 2023
   - Test: 2024+

2. **Loss Function**
   - Duration: Gaussian NLL (predict mean + std)
   - Direction: Cross-entropy
   - Transition: Cross-entropy
   - Combined with learnable weights

3. **Optimization**
   - Optimizer: AdamW
   - Learning rate: 1e-4 with cosine decay
   - Batch size: 64-256 depending on GPU memory
   - Gradient clipping: 1.0

## Inference Pipeline

```python
# Live prediction flow
def predict(current_bar):
    # 1. Get latest data
    tsla_data = fetch_latest('TSLA', '5min')
    spy_data = fetch_latest('SPY', '5min')
    vix_data = fetch_latest('VIX', 'daily')

    # 2. Detect channels at all timeframes
    channels = {}
    for tf in TIMEFRAMES:
        channels[tf] = detect_channel(resample(tsla_data, tf))

    # 3. Extract features
    features = extract_all_features(tsla_data, spy_data, vix_data, channels)

    # 4. Run model
    prediction = model(features)

    # 5. Return trading signal
    return {
        'channel_valid': channels['5min'].is_valid,
        'duration_bars': prediction['duration_mean'],
        'break_direction': 'UP' if prediction['direction_prob'][0] > 0.5 else 'DOWN',
        'confidence': prediction['confidence']
    }
```
