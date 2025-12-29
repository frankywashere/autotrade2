# v6.0 Duration-Primary Architecture - Quick Start Guide

## Overview

v6.0 fundamentally changes what the model predicts:

**v5.9 (Old):**
- Predicts: High/low price targets directly
- Problem: Model learns to predict prices, not channel behavior

**v6.0 (New):**
- Predicts: Duration (how long until channel breaks)
- Computes: High/low bounds from duration × channel geometry
- Result: Model learns channel behavior, projections are just math

## Step-by-Step Setup

### Step 1: Generate v6 Cache

You need v6 labels before training. These labels include:
- Duration per window (when does each window's channel break?)
- Break tracking (did price return after breaking?)
- Window quality (R² scores for each window)
- Transition labels (what happens after break?)

```bash
# Generate v6 cache from existing v5 features
python scripts/generate_v6_cache.py \
  --features-path data/feature_cache/tf_meta_*.json \
  --output data/feature_cache_v6 \
  --validate

# This takes ~30-60 minutes for full dataset
# Output: data/feature_cache_v6/tf_{timeframe}_6.0.0.npz (one per TF)
```

### Step 2: Train (v6 is Default)

```bash
# Train with v6 duration-primary architecture (DEFAULT)
python train_hierarchical.py \
  --epochs 50 \
  --device cuda \
  --native-timeframes \
  --tf-meta data/feature_cache/tf_meta_*.json

# v6.0 is now the default architecture!
# No --v6 flag needed

# Optional flags:
# --v6-warmup-epochs 10: Custom warmup (default: 10)
# --v5-legacy: Use old v5.x architecture (for comparison only)
```

### Step 3: Monitor Training

v6 loss components:
```
E5 | total=2.3456 | dur=1.2345 | win=0.1234 | tf=0.0987 | cont=0.5432 | trans=0.3210 | ret_bonus=0.0123
```

Where:
- `dur`: Duration NLL (PRIMARY, always 1.0 weight)
- `win`: Window selection loss (punish bad window choices)
- `tf`: TF selection loss (punish trusting bad timeframes)
- `cont`: Containment loss (validates duration via price bounds, ramps up)
- `trans`: Transition loss (punish wrong predictions, ramps up)
- `ret_bonus`: Return bonus (rewards temporary breaks that return)

## What Changed in the Code

### Model Changes (`src/ml/hierarchical_model.py`)

**Removed:**
- `timeframe_heads[f'{tf}_high']` - No longer learns high targets
- `timeframe_heads[f'{tf}_low']` - No longer learns low targets
- `hit_band_head` - No longer predicts band hits
- `hit_target_head` - No longer predicts target hits
- `expected_return_head` - No longer predicts returns

**Enhanced:**
- Window selector now uses Gumbel-Softmax with temperature annealing
- Separate temperatures for TF selection and window selection

**Added:**
- `set_selection_temperatures()` - Unified temperature annealing
- `get_v6_output_dict()` - Extracts outputs for v6 loss

**Kept:**
- `duration_heads` - Now PRIMARY output (was secondary)
- `validity_heads` - Forward-looking channel validity
- `window_selectors` - Enhanced with R² guidance
- `compositor` - Transition predictions
- `hit_probability_heads` - Where price will go within channel

### Loss Changes (`src/ml/loss_v6.py`)

New loss structure:
1. **Duration NLL** (weight 1.0) - Gaussian NLL for probabilistic duration
2. **Window Selection** (weight 0.3) - Punish low R² / short duration windows
3. **TF Selection** (weight 0.3) - Punish trusting TFs that broke early
4. **Containment** (ramps 0→1.0) - Check if price stays in projected bounds
5. **Breakout Timing** (weight 0.5) - Penalize if breaks before predicted
6. **Return Bonus** (weight -0.2) - Reward temporary breaks that return
7. **Transition** (ramps 0→0.5) - Punish wrong transition type/direction

### Dataset Changes (`src/ml/hierarchical_dataset.py`)

**Added:**
- `v6_cache_dir` parameter - Path to v6 cache
- `_init_v6_cache_mode()` - Load v6 .npz files
- `_get_v6_labels_for_sample()` - Extract labels for one sample
- v6 labels automatically added to targets dict

**Cache Format:**
```
data/feature_cache_v6/
├── tf_5min_6.0.0.npz
├── tf_15min_6.0.0.npz
├── ...
└── cache_meta_6.0.0.json
```

Each .npz contains:
- `timestamps`: int64 nanosecond timestamps
- `w{window}_*`: Per-window labels (14 windows × 17 metrics each)
  - `valid`, `r_squared`, `slope`, `width`
  - `first_break_bar`, `final_duration`
  - `break_direction`, `returned`, `bars_to_return`
  - `bars_outside`, `max_consecutive_outside`
  - `hit_upper`, `hit_midline`, `hit_lower`
  - `bars_to_upper`, `bars_to_midline`, `bars_to_lower`
- `transition_type`, `transition_direction`, `transition_next_tf`

## Configuration (`config.py`)

New v6.0 settings:
```python
V6_CACHE_DIR = DATA_DIR / "feature_cache_v6"
V6_CACHE_VERSION = "6.0.0"
V6_MAX_SCAN_BARS = 500
V6_RETURN_THRESHOLD_BARS = 3

V6_LOSS_WEIGHTS = {
    'duration': 1.0,
    'window_selection': 0.3,
    'tf_selection': 0.3,
    'containment_final': 1.0,
    'breakout_timing': 0.5,
    'return_bonus': 0.2,
    'transition_final': 0.5,
}

V6_WARMUP_EPOCHS = 10

V6_TEMPERATURE = {
    'tf_start': 2.0,
    'tf_end': 0.5,
    'window_start': 2.0,
    'window_end': 0.5,
}
```

## Training Warmup Schedule

v6 uses warmup to prevent noisy losses early in training:

```
Epoch:    1    2    3    4    5    6    7    8    9   10   11+
          ├────────────────────┼────────────────────┼─────────
               Phase 1              Phase 2             Phase 3
          Duration Focus      Add Containment      Full Training

Loss Weights:
Duration:       1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
Window Select:  0.3  0.3  0.3  0.3  0.3  0.3  0.3  0.3  0.3  0.3  0.3
TF Select:      0.3  0.3  0.3  0.3  0.3  0.3  0.3  0.3  0.3  0.3  0.3
Containment:    0.0  0.04 0.08 0.16 0.25 0.36 0.49 0.64 0.81 1.0  1.0
Breakout:       0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5
Return Bonus:  -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2 -0.2
Transition:     0.0  0.05 0.1  0.15 0.2  0.25 0.3  0.35 0.4  0.5  0.5

Temperature (Gumbel-Softmax):
TF Selection:   2.0  1.8  1.6  1.4  1.2  1.0  0.9  0.8  0.7  0.6  0.5
Window Select:  2.0  1.8  1.6  1.4  1.2  1.0  0.9  0.8  0.7  0.6  0.5
```

## Common Issues

### 1. "Missing v6 cache"

**Solution:** Generate v6 cache first:
```bash
python scripts/generate_v6_cache.py --features-path data/feature_cache/tf_meta_*.json
```

### 2. "No module named 'src.ml.loss_v6'"

**Solution:** Make sure you're running from the project root:
```bash
cd /Users/frank/Desktop/CodingProjects/exp
python train_hierarchical.py --v6 ...
```

### 3. "v6 targets missing"

**Problem:** v6 cache wasn't loaded or doesn't have labels for this timestamp

**Check:**
```bash
python -m src.ml.cache_v6 validate --dir data/feature_cache_v6
```

### 4. "High/low predictions are zero"

**This is expected!** In v6.0:
- High/low are COMPUTED from duration × channel geometry
- They come from geometric_predictions dict, not learned heads
- If duration head isn't trained yet, projections will be minimal

## Backward Compatibility

v6.0 mode is **opt-in**:
- Without `--v6` flag: Uses v5.x loss (high/low MSE)
- With `--v6` flag: Uses duration-primary loss

The model can load v5.x checkpoints and continue training in v6 mode (though high/low heads will be unused).

## Expected Results

After warmup (epoch 10+):
- Duration predictions converge to actual channel lifetimes
- Window selection favors high R² windows
- TF selection trusts timeframes with longer-lasting channels
- Containment rate improves (price stays within projected bounds)
- Transition predictions become more accurate

## Next Steps

1. Generate v6 cache
2. Train for 20-50 epochs with `--v6`
3. Monitor loss components (duration should decrease consistently)
4. Check containment rate improvement (printed every 5 epochs)
5. Validate on test set

## Philosophy

**v5.9 asked:** "Where will the price go?"
**v6.0 asks:** "How long will this channel last?"

If you know the duration, the projection is just geometry. This is more learnable and interpretable.
