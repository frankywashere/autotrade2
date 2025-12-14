# Learned Channel Projection Plan

## Status: IMPLEMENTED (v5.6.0)

## Overview

~~Currently, `projected_high` and `projected_low` are calculated as fixed geometric extrapolations (24 hours forward). This is wrong.~~

**Implemented**: The model learns to predict channel continuation duration, then projections are calculated using the PREDICTED duration.

```
Current (Wrong):
  Features (incl. fixed 24h projection) → Model → Predictions

Correct:
  Features → Model → Predicted Duration → Calculate Projected High/Low
```

---

## Current State Analysis

### What We Have

1. **Continuation Labels** (generated in `features.py`):
   - `duration_bars` - How many bars until channel broke
   - `max_gain` - Maximum gain before break
   - `break_direction` - Which way it broke (up/down)
   - `confidence` - Label quality score

2. **Model Outputs** (in `hierarchical_dataset.py`):
   - Model is trained to predict these continuation labels
   - Per-TF predictions: `cont_{tf}_duration`, `cont_{tf}_gain`, etc.

3. **Fixed Projection Features** (wrong):
   - `{symbol}_channel_{tf}_w{window}_projected_high`
   - `{symbol}_channel_{tf}_w{window}_projected_low`
   - Currently: geometric projection by fixed 24-hour bars
   - Problem: doesn't use learned information

### What's Missing

1. **Inference pipeline** that:
   - Takes model's predicted duration
   - Projects channel forward by that duration
   - Returns predicted price targets

2. **Post-processing step** that converts:
   - `predicted_duration_bars` → `projected_high_price`, `projected_low_price`

---

## Implementation Plan

### Task 1: Understand Current Model Architecture

**Files to examine:**
- `src/ml/hierarchical_dataset.py` - What targets does the model predict?
- `src/ml/model.py` or similar - Model architecture and outputs
- `src/ml/train.py` or similar - Training loop and loss functions

**Questions to answer:**
- Does the model output `predicted_duration` per TF?
- What's the output format (regression value? classification bins?)
- How is inference currently done?

---

### Task 2: Define Projection Calculation

**Input** (from model prediction):
```python
predicted_duration_bars = model.predict(features)['cont_weekly_duration']
# e.g., 23.5 bars
```

**Channel state** (from features):
```python
current_price = 245.00
upper_bound = 248.00  # Current channel upper
lower_bound = 242.00  # Current channel lower
high_slope = 0.15     # $/bar slope of upper bound
low_slope = 0.12      # $/bar slope of lower bound
```

**Projection calculation**:
```python
# Project channel forward by PREDICTED duration
projected_upper = upper_bound + (high_slope * predicted_duration_bars)
projected_lower = lower_bound + (low_slope * predicted_duration_bars)

# As percentages from current price
projected_high_pct = (projected_upper - current_price) / current_price * 100
projected_low_pct = (projected_lower - current_price) / current_price * 100
```

**Output**:
```python
{
    'predicted_duration_bars': 23.5,
    'projected_high_price': 251.45,
    'projected_low_price': 244.82,
    'projected_high_pct': 2.63,   # +2.63% from current
    'projected_low_pct': -0.07,   # -0.07% from current
}
```

---

### Task 3: Remove Fixed Projection Features

**File:** `src/ml/partial_channel_calc_vectorized.py`

**Current** (lines ~477-506):
```python
# Fixed 24-hour projection (WRONG)
forecast_bars = BARS_PER_24H.get(tf, 1)
future_upper = ...
projected_high = (future_upper - current_price) / current_price * 100
```

**New approach - Option A: Remove entirely**
- Delete `projected_high` and `projected_low` from features
- They become post-inference calculations only
- Reduces feature count by 2 per window (2 × 14 windows × 11 TFs × 2 symbols = 616 fewer features)

**New approach - Option B: Keep as geometric features**
- Rename to `geometric_projected_high_24h` to clarify it's not learned
- Keep as a feature (model might find it useful)
- Add learned projection as separate post-inference step

**Recommendation**: Option A - remove them. The model already has slope, width, and duration predictions. Fixed projections add no information.

---

### Task 4: Update Feature Version

**File:** `src/ml/features.py`

```python
FEATURE_VERSION = "v5.6.0"  # Removed fixed projection features
PARTIAL_BAR_VERSION = "v4"  # Cache invalidation
```

Update feature count expectations:
```python
features_per_window = 32  # Was 34, removed projected_high/low
```

---

### Task 5: Create Inference Projection Module

**New file:** `src/ml/projection_calculator.py`

```python
"""
Learned Channel Projection Calculator

Takes model predictions (duration) and current channel state,
returns projected price targets.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ChannelProjection:
    """Projected channel based on learned duration prediction."""
    timeframe: str
    window: int

    # Model prediction
    predicted_duration_bars: float
    prediction_confidence: float

    # Current channel state
    current_price: float
    current_upper: float
    current_lower: float
    high_slope: float
    low_slope: float

    # Calculated projections
    projected_upper: float
    projected_lower: float
    projected_high_pct: float
    projected_low_pct: float

    # Trading signals
    upside_potential: float  # % to projected upper
    downside_risk: float     # % to projected lower
    risk_reward_ratio: float


def calculate_projection(
    predicted_duration: float,
    current_price: float,
    channel_upper: float,
    channel_lower: float,
    high_slope: float,
    low_slope: float,
    confidence: float = 1.0
) -> ChannelProjection:
    """
    Calculate projected price targets using learned duration.

    Args:
        predicted_duration: Model's predicted bars until channel break
        current_price: Current close price
        channel_upper: Current upper bound of channel
        channel_lower: Current lower bound of channel
        high_slope: Slope of upper bound ($/bar)
        low_slope: Slope of lower bound ($/bar)
        confidence: Model's confidence in prediction

    Returns:
        ChannelProjection with calculated targets
    """
    # Project channel forward by predicted duration
    projected_upper = channel_upper + (high_slope * predicted_duration)
    projected_lower = channel_lower + (low_slope * predicted_duration)

    # Convert to percentages
    projected_high_pct = (projected_upper - current_price) / current_price * 100
    projected_low_pct = (projected_lower - current_price) / current_price * 100

    # Calculate trading signals
    upside = (projected_upper - current_price) / current_price * 100
    downside = (current_price - projected_lower) / current_price * 100
    rr_ratio = upside / downside if downside > 0 else float('inf')

    return ChannelProjection(
        predicted_duration_bars=predicted_duration,
        prediction_confidence=confidence,
        current_price=current_price,
        current_upper=channel_upper,
        current_lower=channel_lower,
        high_slope=high_slope,
        low_slope=low_slope,
        projected_upper=projected_upper,
        projected_lower=projected_lower,
        projected_high_pct=projected_high_pct,
        projected_low_pct=projected_low_pct,
        upside_potential=upside,
        downside_risk=downside,
        risk_reward_ratio=rr_ratio,
        timeframe="",  # Set by caller
        window=0,      # Set by caller
    )


def calculate_all_projections(
    model_predictions: Dict[str, float],
    channel_features: Dict[str, float],
    current_price: float,
    timeframes: list = None,
    windows: list = None,
) -> Dict[str, ChannelProjection]:
    """
    Calculate projections for all TF/window combinations.

    Args:
        model_predictions: Dict with keys like 'cont_weekly_duration'
        channel_features: Dict with channel feature values
        current_price: Current close price
        timeframes: List of timeframes to process
        windows: List of window sizes to process

    Returns:
        Dict of ChannelProjection objects keyed by '{tf}_w{window}'
    """
    if timeframes is None:
        timeframes = ['5min', '15min', '30min', '1h', '2h', '3h', '4h',
                      'daily', 'weekly', 'monthly', '3month']
    if windows is None:
        windows = [100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10]

    projections = {}

    for tf in timeframes:
        # Get predicted duration for this TF
        duration_key = f'cont_{tf}_duration'
        if duration_key not in model_predictions:
            continue

        predicted_duration = model_predictions[duration_key]
        confidence = model_predictions.get(f'cont_{tf}_confidence', 1.0)

        for window in windows:
            prefix = f'tsla_channel_{tf}_w{window}'

            # Get channel state from features
            try:
                channel_upper = channel_features.get(f'{prefix}_upper_bound',
                                current_price * 1.01)  # Fallback
                channel_lower = channel_features.get(f'{prefix}_lower_bound',
                                current_price * 0.99)  # Fallback
                high_slope = channel_features.get(f'{prefix}_high_slope', 0)
                low_slope = channel_features.get(f'{prefix}_low_slope', 0)

                projection = calculate_projection(
                    predicted_duration=predicted_duration,
                    current_price=current_price,
                    channel_upper=channel_upper,
                    channel_lower=channel_lower,
                    high_slope=high_slope,
                    low_slope=low_slope,
                    confidence=confidence,
                )
                projection.timeframe = tf
                projection.window = window

                projections[f'{tf}_w{window}'] = projection

            except Exception as e:
                # Skip if features missing
                continue

    return projections
```

---

### Task 6: Update Inference Pipeline

**File:** `src/ml/inference.py` (or wherever live prediction happens)

```python
from .projection_calculator import calculate_all_projections

def predict_with_projections(model, features, current_price):
    """
    Run model inference and calculate price projections.

    Returns:
        dict with model predictions AND calculated projections
    """
    # Step 1: Get model predictions (duration, confidence, etc.)
    raw_predictions = model.predict(features)

    # Step 2: Calculate projections using predicted durations
    projections = calculate_all_projections(
        model_predictions=raw_predictions,
        channel_features=features,  # Current channel state
        current_price=current_price,
    )

    # Step 3: Combine into final output
    result = {
        'predictions': raw_predictions,
        'projections': projections,
        'summary': {
            'best_upside': max(p.upside_potential for p in projections.values()),
            'best_rr_ratio': max(p.risk_reward_ratio for p in projections.values()),
        }
    }

    return result
```

---

### Task 7: Verify Continuation Labels

**File:** `src/ml/features.py` - `generate_hierarchical_continuation_labels_5min`

Ensure labels include everything needed:

```python
# Required label fields:
{
    'duration_bars': int,      # Bars until channel broke
    'max_gain_pct': float,     # Max gain before break
    'break_direction': int,    # 1=up, -1=down, 0=sideways
    'final_position': float,   # Where in channel at break
    'confidence': float,       # Label quality
}
```

The model learns to predict `duration_bars`, which is then used for projection.

---

### Task 8: Update Training to Emphasize Duration

**File:** Training configuration

Ensure duration prediction is weighted appropriately:

```python
loss_weights = {
    'duration': 2.0,      # Most important for projections
    'max_gain': 1.0,
    'direction': 0.5,
    'confidence': 0.5,
}
```

Consider adding duration-specific metrics:
- MAE for duration prediction
- % predictions within 10% of actual
- Correlation between predicted and actual duration

---

## Summary of Changes

| File | Change |
|------|--------|
| `partial_channel_calc_vectorized.py` | Remove `projected_high/low` features |
| `parallel_channel_extraction.py` | Remove `projected_high/low` features (live mode) |
| `features.py` | Update version, feature count |
| `projection_calculator.py` | NEW - Calculate projections from predictions |
| `inference.py` | Add projection calculation step |
| `hierarchical_dataset.py` | Verify duration labels are included |

## Feature Count Impact

**Before:**
- 34 features per window
- 34 × 14 windows × 11 TFs × 2 symbols = 10,472 features

**After:**
- 32 features per window (removed projected_high/low)
- 32 × 14 × 11 × 2 = 9,856 features
- Savings: 616 features

---

## Testing Checklist

1. [ ] Verify model outputs `cont_{tf}_duration` predictions
2. [ ] Verify channel slope features are available at inference
3. [ ] Test projection calculation with known values
4. [ ] Validate projections make sense (steep channel = larger projection)
5. [ ] Compare predicted vs actual duration on test set
6. [ ] End-to-end: features → model → duration → projection → price target

---

## Implementation Order

1. ✅ **Task 1**: Examine current model architecture (research only)
2. ✅ **Task 7**: Verify continuation labels are correct
3. ✅ **Task 3**: Remove fixed projection features from partial_channel_calc_vectorized.py
4. ✅ **Task 4**: Update feature version (v5.6.0)
5. ✅ **Task 5**: Create projection_calculator.py
6. ✅ **Task 6**: Update inference pipeline (added `predict_with_projections` method)
7. ⏳ **Task 8**: Verify training weights duration appropriately (optional)
8. ⏳ **Testing**: Full validation (optional)

---

## Notes

- The fixed 24-hour projection was a misunderstanding of the system design
- The correct flow: Model predicts duration → Project by predicted duration
- This aligns with the "continuation labels" training approach
- Projections become post-inference calculations, not input features
