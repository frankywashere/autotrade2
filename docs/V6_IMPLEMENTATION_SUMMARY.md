# v6.0 Duration-Primary Architecture - Implementation Summary

**Date:** December 28, 2025
**Status:** ✅ Complete - Ready for Testing
**Branch:** `x3` (current)

---

## What Was Implemented

### Core Philosophy Change

**v5.9 Architecture (Old):**
```
Input → CfC Layers → High/Low Prediction Heads → Loss (MSE on price targets)
```

**v6.0 Architecture (New):**
```
Input → CfC Layers → Duration Prediction → Compute High/Low from Geometry → Loss (Duration NLL + Containment)
```

The model now learns **"how long will this channel last?"** instead of **"where will the price go?"**

---

## Files Created

### 1. `src/ml/loss_v6.py` (448 lines)

Complete v6 loss system:

**Loss Functions:**
- `compute_duration_nll()` - Gaussian NLL for probabilistic duration prediction
- `compute_window_selection_loss()` - Punishes placing weight on low R² windows
- `compute_tf_selection_loss()` - Punishes trusting timeframes that broke early
- `compute_containment_loss()` - Validates duration by checking if price stayed in bounds
- `compute_breakout_timing_loss()` - Penalizes optimistic predictions (breaks before predicted)
- `compute_return_bonus()` - Rewards channels that returned after temporary breaks (negative loss)
- `compute_transition_loss()` - Punishes wrong transition type/direction/next_tf predictions

**Utilities:**
- `compute_v6_loss()` - Main loss combining all components with warmup
- `V6LossConfig` - Configuration class
- `get_warmup_weight()` - Quadratic warmup scheduler
- `get_temperature()` - Gumbel-Softmax temperature annealing
- `format_loss_log()` - Pretty loss logging

### 2. `src/ml/cache_v6.py` (566 lines)

Unified cache format and label generation:

**Label Generation:**
- `detect_break_with_return()` - Detects when channel breaks AND tracks if price returns
  - Scans up to 500 bars forward
  - Tracks: first_break_bar, returned, bars_to_return, bars_outside, max_consecutive_outside
  - Hit tracking: upper, midline, lower bounds
- `compute_channel_state()` - Computes slope, intercept, R², residual_std for a window
- `detect_transition()` - Determines transition type/direction/next_tf after break

**Cache Management:**
- `generate_v6_cache()` - Generates .npz files per timeframe
- `generate_tf_labels()` - Processes all windows (14) for one timeframe
- `load_v6_cache()` - Loads cache into memory
- `validate_v6_cache()` - Validates cache integrity

**Cache Format:**
```
data/feature_cache_v6/
├── tf_5min_6.0.0.npz       # 418,635 bars
├── tf_15min_6.0.0.npz      # 154,407 bars
├── tf_30min_6.0.0.npz      # 81,492 bars
├── tf_1h_6.0.0.npz         # 43,120 bars
├── tf_2h_6.0.0.npz         # 23,197 bars
├── tf_3h_6.0.0.npz         # 17,045 bars
├── tf_4h_6.0.0.npz         # 12,885 bars
├── tf_daily_6.0.0.npz      # 3,160 bars
├── tf_weekly_6.0.0.npz     # 561 bars
├── tf_monthly_6.0.0.npz    # 129 bars
├── tf_3month_6.0.0.npz     # 44 bars
└── cache_meta_6.0.0.json   # Metadata
```

### 3. `scripts/generate_v6_cache.py` (160 lines)

CLI tool for cache generation:
```bash
python scripts/generate_v6_cache.py \
  --features-path data/feature_cache/tf_meta_*.json \
  --output data/feature_cache_v6 \
  --max-scan-bars 500 \
  --return-threshold 3 \
  --validate
```

### 4. `docs/V6_QUICKSTART.md`

User guide with:
- Step-by-step setup instructions
- Training commands
- Loss interpretation
- Troubleshooting
- Expected results

---

## Files Modified

### 1. `src/ml/hierarchical_model.py`

**Removed (lines 507-509):**
```python
# v6.0: REMOVED high/low heads
# self.timeframe_heads[f'{tf}_high'] = nn.Linear(hidden_size, 1)  # DELETED
# self.timeframe_heads[f'{tf}_low'] = nn.Linear(hidden_size, 1)   # DELETED
# Only keeping conf head for validity
```

**Removed (lines 534-540):**
```python
# v6.0: REMOVED multi-task heads that depend on price predictions
# self.hit_band_head - DELETED
# self.hit_target_head - DELETED
# self.expected_return_head - DELETED
```

**Enhanced (lines 676-695):**
```python
# v6.0: Enhanced window selector with Gumbel-Softmax
self.window_selectors = nn.ModuleDict({...})
self.window_selection_temperature = 2.0  # Annealed 2.0 → 0.5
```

**Updated (lines 910-922):**
```python
# v6.0: Gumbel-Softmax with temperature annealing
window_weights = F.gumbel_softmax(
    window_logits,
    tau=self.window_selection_temperature,
    hard=False
)
```

**Updated (lines 1330-1349):**
```python
# v6.0: Get high/low from geometric projection (computed, not learned)
if geo_result is not None:
    pred_high = geo_result['high']  # From geometry
    pred_low = geo_result['low']    # From geometry
else:
    pred_high = torch.zeros(...)    # Fallback
```

**Added Methods:**
- `set_selection_temperatures()` (lines 1768-1791) - Unified temperature annealing
- `get_v6_output_dict()` (lines 1793-1839) - Extract outputs for v6 loss

**Constants Updated (lines 421-425):**
```python
PREDICTIONS_PER_LAYER = 1  # Only confidence (high/low removed)
FUSION_INPUT_DIM = NUM_LAYERS * 3 + MARKET_STATE_DIM  # 11 confs + 11 durations + 11 validities
```

### 2. `src/ml/hierarchical_dataset.py`

**Added Parameters (line 79):**
```python
v6_cache_dir: str = None  # Path to v6 cache directory
```

**Added Methods:**
- `_init_v6_cache_mode()` (lines 647-685) - Load v6 .npz files
- `_get_v6_labels_for_sample()` (lines 687-729) - Extract labels for one timestamp/TF

**Updated `_getitem_native_timeframe()` (lines 1462-1501):**
```python
# v6.0: Add duration-primary labels if v6 cache is loaded
if getattr(self, 'use_v6_cache', False) and self._v6_labels:
    # Extract v6 labels for all TFs
    # Add to targets dict with tf prefix
    # Aggregate window labels into arrays for loss
```

**Updated `create_hierarchical_dataset()` (line 2653):**
```python
v6_cache_dir: str = None  # v6.0 parameter added
```

All dataset creation calls updated to pass `v6_cache_dir` parameter.

### 3. `train_hierarchical.py`

**Added Imports (lines 52-55):**
```python
from src.ml.loss_v6 import (
    compute_v6_loss, V6LossConfig, get_warmup_weight, get_temperature, format_loss_log
)
```

**Added Arguments (lines 5307-5311):**
```python
--v6                    # Enable v6 mode
--v6-warmup-epochs N    # Warmup epochs (default: 10)
```

**Added v6 Configuration (lines 4120-4146):**
```python
USE_V6_LOSS = getattr(args, 'v6', False)
V6_WARMUP_EPOCHS = ...
v6_loss_config = V6LossConfig(...)
```

**Updated Temperature Annealing (lines 4197-4224):**
```python
if USE_V6_LOSS:
    model_core.set_selection_temperatures(
        epoch=epoch,
        warmup_epochs=V6_WARMUP_EPOCHS,
        tf_start=2.0, tf_end=0.5,
        window_start=2.0, window_end=0.5,
    )
```

**Added v6 Loss Path (lines 4335-4374):**
```python
if USE_V6_LOSS:
    # Extract v6 predictions
    v6_predictions = model_core.get_v6_output_dict(hidden_states)

    # Prepare v6 targets
    v6_targets = {...}

    # Compute v6 loss
    loss, loss_components = compute_v6_loss(...)
else:
    # v5.x legacy path
```

**Updated Legacy Loss Guards:**
- Multi-task losses: Skip hit_band/hit_target/expected_return in v6 mode
- Duration loss: Skip in v6 mode (handled by compute_v6_loss)
- Validity loss: Skip in v6 mode (handled by compute_v6_loss)
- Transition loss: Skip in v6 mode (handled by compute_v6_loss)

**Updated Collate (lines 467-473):**
```python
# v6.0: Handle window_r_squared and window_durations (lists → 2D tensors)
elif '_window_r_squared' in k or '_window_durations' in k:
    values = [t[k] for t in targets_list]
    targets_batch[k] = torch.tensor(values, dtype=torch_dtype)
```

**Updated Dataset Creation (lines 3646-3672):**
```python
# v6.0: Get v6 cache directory if v6 mode enabled
v6_cache_dir_path = str(project_config.V6_CACHE_DIR) if args.v6 else None

train_dataset, val_dataset, test_dataset = create_hierarchical_dataset(
    ...
    v6_cache_dir=v6_cache_dir_path,
)
```

### 4. `config.py`

**Added v6 Configuration Section (lines 362-414):**
```python
# v6.0 DURATION-PRIMARY ARCHITECTURE CONFIGURATION
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

V6_WINDOWS = CHANNEL_WINDOW_SIZES
V6_TRANSITION_TYPES = {'CONTINUE': 0, 'SWITCH_TF': 1, 'REVERSE': 2, 'SIDEWAYS': 3}
V6_DIRECTIONS = {'BEAR': 0, 'BULL': 1, 'SIDEWAYS': 2}
```

---

## Architecture Changes Summary

### What Was Removed
✂️ High/low prediction heads per TF (33 heads total)
✂️ hit_band_head (binary classifier)
✂️ hit_target_head (binary classifier)
✂️ expected_return_head (regression)
✂️ Direct price prediction losses (MSE on high/low)

### What Was Enhanced
🔧 Window selectors: Now use Gumbel-Softmax with temperature annealing
🔧 Window selection: R² guidance with soft→hard selection over training
🔧 Selection temperatures: Separate for TF and window, annealed 2.0→0.5

### What Was Kept (Now Primary)
✅ Duration heads (mean + log_std) - NOW PRIMARY OUTPUT
✅ Validity heads - Forward-looking channel trust
✅ Window selectors - Which lookback window to use
✅ Transition compositor - Type/direction/next_tf predictions
✅ Hit probability heads - Where price goes within channel

### What Is Computed (Not Learned)
📐 Projected high/low bounds from duration × channel geometry
📐 `geo_high = upper_dist + (high_slope_pct × duration)`
📐 `geo_low = -lower_dist + (low_slope_pct × duration)`

---

## Loss Structure Comparison

### v5.9 Loss (Old)
```
Total = 1.0 × MSE(high, low)              # Primary
      + 0.1 × multi_tf_loss               # All TF contributions
      + 0.05 × entropy_regularization     # Prevent mode collapse
      + 0.3 × duration_nll                # Secondary (warmup)
      + 0.2 × validity_loss               # Secondary (warmup)
      + 0.5 × transition_loss             # Secondary (warmup)
      + 0.1 × containment_loss            # Validation
```

### v6.0 Loss (New)
```
Total = 1.0 × duration_nll                # PRIMARY (always)
      + 0.3 × window_selection_loss       # Punish bad windows
      + 0.3 × tf_selection_loss           # Punish bad TF trust
      + [0→1.0] × containment_loss        # Ramps up (warmup)
      + 0.5 × breakout_timing_loss        # Penalize early breaks
      - 0.2 × return_bonus                # Reward returns (negative)
      + [0→0.5] × transition_loss         # Ramps up (warmup)
```

---

## Model Output Changes

### v5.9 Outputs
```python
predictions = [batch, 3]  # [high, low, confidence]
output_dict = {
    'duration': {...},              # Secondary
    'geometric_predictions': {...}, # Secondary
    'direct_predictions': {...},    # PRIMARY (learned high/low)
}
```

### v6.0 Outputs
```python
predictions = [batch, 3]  # [geo_high, geo_low, validity]
output_dict = {
    'duration': {...},              # PRIMARY (mean, log_std, std, confidence)
    'geometric_predictions': {...}, # Computed from duration × geometry
    'validity': {...},              # Per-TF forward-looking trust
    'window_weights': {...},        # Soft selection over 14 windows
    'compositor': {...},            # Transition predictions
}
```

---

## Training Workflow

### 1. Generate v6 Cache (One-Time Setup)
```bash
python scripts/generate_v6_cache.py \
  --features-path data/feature_cache/tf_meta_v5.9.1_*.json \
  --output data/feature_cache_v6 \
  --validate

# Takes ~30-60 minutes
# Generates ~2-3 GB of .npz files
```

### 2. Train (v6 is Default)
```bash
# v6.0 is now the DEFAULT architecture (no flag needed)
python train_hierarchical.py \
  --epochs 50 \
  --batch_size 128 \
  --device cuda \
  --native-timeframes \
  --tf-meta data/feature_cache/tf_meta_*.json

# Use --v5-legacy ONLY if you want old v5.x behavior for comparison

# Output:
# E1 | total=3.2156 | dur=2.8901 | win=0.1234 | tf=0.0987 | ...
# E10 | total=1.4321 | dur=1.2345 | win=0.0678 | tf=0.0543 | cont=0.0789 | ...
```

### 3. Monitor Key Metrics
- **Duration loss**: Should decrease steadily (model learning channel lifetimes)
- **Window selection**: Should converge to high R² windows
- **Containment**: Should increase after warmup (price stays in projected bounds)
- **Temperature**: Anneals from 2.0 → 0.5 (soft → hard selection)

---

## Label Format

### Per-Window Labels (14 windows × 17 metrics = 238 labels per TF)

For each window (w100, w90, ..., w10):
```python
{
    # Quality
    'w100_valid': 1,              # Is this window's channel valid?
    'w100_r_squared': 0.85,       # Fit quality [0-1]
    'w100_slope': 0.05,           # Channel slope (price/bar)
    'w100_width': 2.5,            # Channel width (4σ)

    # Duration
    'w100_first_break_bar': 47,   # Bar when first broke (or 500 if none)
    'w100_final_duration': 120,   # Effective duration (with returns)

    # Break tracking
    'w100_break_direction': 1,    # -1=below, 0=none, 1=above
    'w100_returned': 1,           # Did price return? (0/1)
    'w100_bars_to_return': 8,     # Bars until returned
    'w100_bars_outside': 15,      # Total bars spent outside
    'w100_max_consecutive_outside': 4,  # Longest streak

    # Hit tracking
    'w100_hit_upper': 1,          # Hit upper bound?
    'w100_hit_midline': 1,        # Hit midline?
    'w100_hit_lower': 0,          # Hit lower bound?
    'w100_bars_to_upper': 23,     # Bars until hit upper
    'w100_bars_to_midline': 12,   # Bars until hit midline
    'w100_bars_to_lower': 500,    # Bars until hit lower (didn't hit)
}
```

### Transition Labels (3 per sample)
```python
{
    'transition_type': 1,      # 0=continue, 1=switch_tf, 2=reverse, 3=sideways
    'transition_direction': 1, # 0=bear, 1=bull, 2=sideways
    'transition_next_tf': 3,   # Next TF index (0-10)
}
```

---

## Training Warmup Schedule

### Loss Weight Progression

| Epoch | Duration | Window | TF | Containment | Breakout | Return | Transition |
|-------|----------|--------|-----|-------------|----------|--------|------------|
| 1     | 1.0      | 0.3    | 0.3 | 0.00        | 0.5      | -0.2   | 0.00       |
| 2     | 1.0      | 0.3    | 0.3 | 0.04        | 0.5      | -0.2   | 0.05       |
| 5     | 1.0      | 0.3    | 0.3 | 0.25        | 0.5      | -0.2   | 0.20       |
| 10    | 1.0      | 0.3    | 0.3 | 1.00        | 0.5      | -0.2   | 0.50       |
| 11+   | 1.0      | 0.3    | 0.3 | 1.00        | 0.5      | -0.2   | 0.50       |

### Temperature Progression

| Epoch | TF Selection | Window Selection |
|-------|--------------|------------------|
| 1     | 2.0          | 2.0              |
| 5     | 1.2          | 1.2              |
| 10    | 0.5          | 0.5              |
| 11+   | 0.5          | 0.5              |

High temp = soft selection (explore all options)
Low temp = hard selection (commit to best)

---

## v6.0 is Now the Default

✅ **Default behavior**: v6.0 duration-primary architecture
✅ **No flag needed**: Just run `train_hierarchical.py` normally
✅ **Legacy mode**: Use `--v5-legacy` flag only if you need old v5.x behavior
⚠️ **predictions tensor**: Still returned by forward() but NOT used by v6 training/validation/test
  - High/low are GEOMETRIC (computed from duration × channel geometry)
  - Only used for inference/dashboards
  - NOT fed into v6 loss function

---

## Testing Checklist

Before production use:

- [ ] Generate v6 cache for full dataset
- [ ] Validate cache integrity (`--validate` flag)
- [ ] Train for 20+ epochs with `--v6`
- [ ] Verify duration loss decreases
- [ ] Verify containment rate improves after epoch 10
- [ ] Check window selection converges to high R² windows
- [ ] Test inference with geometric projections
- [ ] Compare duration predictions to actual breakouts
- [ ] Validate transition predictions on test set

---

## File Change Summary

| File | Lines Changed | Status |
|------|--------------|--------|
| `src/ml/loss_v6.py` | +448 (new) | ✅ Created |
| `src/ml/cache_v6.py` | +566 (new) | ✅ Created |
| `scripts/generate_v6_cache.py` | +160 (new) | ✅ Created |
| `docs/V6_QUICKSTART.md` | +233 (new) | ✅ Created |
| `src/ml/hierarchical_model.py` | ~150 modified | ✅ Updated |
| `src/ml/hierarchical_dataset.py` | ~120 modified | ✅ Updated |
| `train_hierarchical.py` | ~180 modified | ✅ Updated |
| `config.py` | +53 added | ✅ Updated |

**Total:** ~1,910 lines of new/modified code

---

## Next Steps

1. **Generate v6 cache:**
   ```bash
   python scripts/generate_v6_cache.py \
     --features-path data/feature_cache/tf_meta_*.json \
     --output data/feature_cache_v6 \
     --validate
   ```

2. **Train v6 model:**
   ```bash
   python train_hierarchical.py \
     --v6 \
     --epochs 50 \
     --device cuda \
     --native-timeframes \
     --tf-meta data/feature_cache/tf_meta_*.json
   ```

3. **Monitor training:**
   - Watch duration loss decrease
   - Check containment improves after epoch 10
   - Verify no NaN/Inf issues

4. **Evaluate:**
   - Compare duration predictions vs actual
   - Check projection accuracy (geometric bounds)
   - Validate transition predictions

---

**Implementation Status:** ✅ Complete
**Ready for:** Cache Generation → Training → Evaluation
