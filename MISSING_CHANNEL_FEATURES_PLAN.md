# Missing Channel Features Implementation Plan

## Overview

The new partial bar channel calculation (`partial_channel_calc_vectorized.py`) is missing 22 features that exist in the old `LinearRegressionChannel` system. This plan details how to add them while maintaining compatibility with the existing feature extraction pipeline.

---

## Integration with Full Feature Pipeline

The feature extraction pipeline has these stages:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Base Features (8 parallel extraction steps)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Price features (OHLCV)                                                   │
│ 2. Channel features ◄──── THIS PLAN AFFECTS THIS                           │
│    - Bollinger bands, linear regression channels                            │
│    - 11 timeframes × 14 windows × 2 symbols × 34 features = 10,472 columns │
│ 3. RSI features (11 TFs × 3 metrics × 2 symbols)                            │
│ 4. Correlation features                                                     │
│ 5. Cycle features                                                           │
│ 6. Volume features                                                          │
│ 7. Time features                                                            │
│ 8. VIX features                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ STAGE 2: Breakdown Features                                                 │
│ - Uses channel_w50_stability and channel_w50_position ✅ (exist)            │
│ - duration_ratio, alignment, time_in_channel, position_norm                 │
│ - Calculated at 5min resolution from channel features                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ STAGE 3: Native TF Generation (if enabled)                                  │
│ - Resamples all 11 timeframes                                               │
│ - Will automatically include new channel features                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ STAGE 4: Labels                                                             │
│ - Continuation labels (per-TF hierarchical)                                 │
│ - Transition labels (v5.2+ compositor)                                      │
│ - Use channel features for break detection ✅ (position/stability exist)    │
└─────────────────────────────────────────────────────────────────────────────┘
```

**What changes:**
- Channel features in Stage 1 go from 12 → 34 features per window
- Total columns: 3,696 → 10,472 (matches old system)

**What stays the same:**
- Breakdown features (only use w50 stability/position)
- Labels (only use position/stability for break detection)
- Native TF generation (just resamples whatever features exist)

---

## Current State

### Feature Count Per Window

| System | Features/Window | Total (14 windows × 11 TFs × 2 symbols) |
|--------|-----------------|----------------------------------------|
| OLD (LinearRegressionChannel) | 34 | 10,472 |
| NEW (partial_channel_calc_vectorized) | 12 | 3,696 |
| **Missing** | **22** | **6,776** |

### Existing Features (12) ✅

```python
# Already implemented in partial_channel_calc_vectorized.py
close_slope, close_slope_pct, close_r_squared
high_slope, low_slope
position, upper_dist, lower_dist
channel_width_pct, stability
is_valid, insufficient_data
```

### Missing Features (22) ❌

```python
# EASY - Simple calculations from existing data
high_slope_pct          # high_slope / price * 100
low_slope_pct           # low_slope / price * 100
high_r_squared          # R² of high regression
low_r_squared           # R² of low regression
r_squared_avg           # (close_r_squared + high_r_squared + low_r_squared) / 3
slope_convergence       # 1 - abs(high_slope - low_slope) / (abs(close_slope) + 1e-10)
is_bull                 # close_slope_pct > 0.1
is_bear                 # close_slope_pct < -0.1
is_sideways             # abs(close_slope_pct) <= 0.1
quality_score           # cycles × (0.5 + 0.5 × r_squared)
duration                # actual_duration (bars since channel start)
projected_high          # high regression extrapolated to current bar
projected_low           # low regression extrapolated to current bar

# HARD - Require tracking state across bars
ping_pongs              # Alternating upper/lower touches (2% threshold)
ping_pongs_0_5pct       # At 0.5% threshold
ping_pongs_1_0pct       # At 1.0% threshold
ping_pongs_3_0pct       # At 3.0% threshold
complete_cycles         # Full round-trips (lower→upper→lower or upper→lower→upper)
complete_cycles_0_5pct  # At 0.5% threshold
complete_cycles_1_0pct  # At 1.0% threshold
complete_cycles_3_0pct  # At 3.0% threshold
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/ml/partial_channel_calc_vectorized.py` | Add 22 missing features to both paths |
| `src/ml/features.py` | Bump `PARTIAL_BAR_VERSION` to invalidate cache |
| `config.py` | No changes needed |

---

## Downstream Dependencies

### Breakdown Features (lines 4440-4515 in features.py)

Uses only these channel features (all currently exist ✅):
- `{symbol}_channel_{tf}_w50_stability` ✅
- `{symbol}_channel_{tf}_w50_position` ✅

**Status: Safe - breakdown will continue to work.**

### Labels (continuation labels)

Uses channel features for break detection. The core features (position, stability) exist.

**Status: Safe - labels will continue to work.**

### Native TF Generation

Resamples features to coarser timeframes. Will automatically include new features.

**Status: Safe - will pick up new features automatically.**

---

## Implementation Tasks

### Task 1: Add EASY Features (12 features)

**File:** `src/ml/partial_channel_calc_vectorized.py`

These features are simple calculations from already-computed values.

#### 1.1 Regular Path (non-5min TFs)

**Location:** `calculate_channel_features_vectorized()` function

**Current state (lines 173-216):**
- Already computes `high_slope`, `low_slope`, `high_intercept`, `low_intercept`
- Only computes R² for close (lines 212-216), NOT for high/low
- Missing high_r_squared, low_r_squared calculation

**To compute high/low R² (add after line 190):**
```python
# Compute high R² (same pattern as close)
high_predicted = high_slope[:, None] * x_hist[None, :] + high_intercept[:, None]
high_residuals = hist_highs[None, :] - high_predicted
high_ss_res = np.sum(high_residuals ** 2, axis=1)
high_ss_tot = n_hist * np.var(hist_highs)
high_r_squared = np.where(high_ss_tot > 1e-10, 1 - high_ss_res / high_ss_tot, 0)
high_r_squared = np.clip(high_r_squared, 0, 1)

# Compute low R² (same pattern as close)
low_predicted = low_slope[:, None] * x_hist[None, :] + low_intercept[:, None]
low_residuals = hist_lows[None, :] - low_predicted
low_ss_res = np.sum(low_residuals ** 2, axis=1)
low_ss_tot = n_hist * np.var(hist_lows)
low_r_squared = np.where(low_ss_tot > 1e-10, 1 - low_ss_res / low_ss_tot, 0)
low_r_squared = np.clip(low_r_squared, 0, 1)

# Average R²
r_squared_avg = (close_r_squared + high_r_squared + low_r_squared) / 3
```

**Add to output dict (line 75-88):**
```python
# Initialize output arrays
prefix = f'{symbol}_channel_{tf}_w{window}'
output = {
    # ... existing 12 features ...

    # NEW: Slope percentage variants
    f'{prefix}_high_slope_pct': np.zeros(n_5min, dtype=np.float32),
    f'{prefix}_low_slope_pct': np.zeros(n_5min, dtype=np.float32),

    # NEW: R-squared variants
    f'{prefix}_high_r_squared': np.zeros(n_5min, dtype=np.float32),
    f'{prefix}_low_r_squared': np.zeros(n_5min, dtype=np.float32),
    f'{prefix}_r_squared_avg': np.zeros(n_5min, dtype=np.float32),

    # NEW: Derived metrics
    f'{prefix}_slope_convergence': np.zeros(n_5min, dtype=np.float32),
    f'{prefix}_quality_score': np.zeros(n_5min, dtype=np.float32),
    f'{prefix}_duration': np.zeros(n_5min, dtype=np.float32),

    # NEW: Direction flags
    f'{prefix}_is_bull': np.zeros(n_5min, dtype=np.float32),
    f'{prefix}_is_bear': np.zeros(n_5min, dtype=np.float32),
    f'{prefix}_is_sideways': np.zeros(n_5min, dtype=np.float32),

    # NEW: Projections
    f'{prefix}_projected_high': np.zeros(n_5min, dtype=np.float32),
    f'{prefix}_projected_low': np.zeros(n_5min, dtype=np.float32),
}
```

**Calculate and fill values (inside the period loop, after computing existing features):**
```python
# After computing close_slope, high_slope, low_slope, current_price...

# Slope percentages
high_slope_pct = high_slope / current_prices_safe * 100
low_slope_pct = low_slope / current_prices_safe * 100

# R-squared for high/low (need to compute separately - see calculation below)
high_r_squared = _compute_r_squared(high_regression_residuals)
low_r_squared = _compute_r_squared(low_regression_residuals)
r_squared_avg = (close_r_squared + high_r_squared + low_r_squared) / 3

# Slope convergence: how parallel are the channel lines
slope_range = np.abs(high_slope - low_slope)
slope_convergence = 1 - slope_range / (np.abs(close_slope) + 1e-10)
slope_convergence = np.clip(slope_convergence, 0, 1)

# Direction flags (based on close_slope_pct)
is_bull = (close_slope_pct > 0.1).astype(np.float32)
is_bear = (close_slope_pct < -0.1).astype(np.float32)
is_sideways = (np.abs(close_slope_pct) <= 0.1).astype(np.float32)

# Projections (extrapolate regression lines to current x position)
x_current = n_bars - 1  # Current position in window
projected_high = high_slope * x_current + high_intercept
projected_low = low_slope * x_current + low_intercept

# Duration (bars since window start - increases with each bar)
duration = np.arange(n_5min, dtype=np.float32)

# Quality score (cycles × (0.5 + 0.5 × r²)) - requires cycles from Task 2
# Initially set to r_squared_avg as placeholder until cycles implemented
quality_score = r_squared_avg
```

#### 1.2 5min Fast Path

**Location:** `_calculate_5min_channel_features_rolling()` function

**Same additions as 1.1 but using vectorized sliding window approach.**

The 5min path already computes slopes and r_squared. Need to add:
1. high_r_squared, low_r_squared (from high_windows/low_windows regression)
2. slope_pct variants
3. Direction flags
4. Projections
5. Duration

---

### Task 2: Add HARD Features (10 features - Ping-Pongs and Cycles)

**Complexity:** These require tracking price touches across the channel boundaries.

#### 2.1 Understanding the Algorithm

From `linear_regression.py` lines 86-171:

**Ping-pongs:** Count alternating touches of upper/lower boundaries
```
Price touches upper → then touches lower → ping_pong += 1
Price touches lower → then touches upper → ping_pong += 1
```

**Complete cycles:** Count full round-trips
```
Lower → Upper → Lower = 1 cycle
Upper → Lower → Upper = 1 cycle
```

**Thresholds:** A "touch" is when price is within X% of boundary
- 2% (default): `ping_pongs`, `complete_cycles`
- 0.5%: `ping_pongs_0_5pct`, `complete_cycles_0_5pct`
- 1.0%: `ping_pongs_1_0pct`, `complete_cycles_1_0pct`
- 3.0%: `ping_pongs_3_0pct`, `complete_cycles_3_0pct`

#### 2.2 Vectorized Implementation Strategy

**Challenge:** Ping-pongs/cycles depend on sequential state (last touch position).

**Key Insight for Partial Bar Context:**
- For non-5min TFs, we have historical complete bars + 1 partial bar
- Ping-pongs should be computed on the COMPLETE bars only (not partial)
- This matches the old behavior where channels were computed on resampled data

**Solution for Regular Path (non-5min):**
```python
# In the period loop, after computing channel boundaries...
# hist_closes, hist_highs, hist_lows already available
# Need to compute upper/lower lines at each historical bar position

# Compute full channel lines over historical window
x_hist = np.arange(n_hist)
close_line = close_slope[:, None] * x_hist[None, :] + close_intercept[:, None]  # (n_bars, n_hist)
high_line = high_slope[:, None] * x_hist[None, :] + high_intercept[:, None]
low_line = low_slope[:, None] * x_hist[None, :] + low_intercept[:, None]

# Adjust with residual std (same as partial bar position)
upper_line = np.maximum(high_line, close_line + 2.0 * residual_std[:, None])
lower_line = np.minimum(low_line, close_line - 2.0 * residual_std[:, None])

# Now compute ping-pongs for each bar in this period
for j, bar_idx in enumerate(bar_indices):
    pp = _compute_ping_pongs_single(
        hist_closes, upper_line[j], lower_line[j], threshold=0.02
    )
    output[f'{prefix}_ping_pongs'][bar_idx] = pp
    # ... other thresholds
```

**Helper function (Numba-optimized):**
```python
@numba.jit(nopython=True, fastmath=True)
def _compute_ping_pongs_single(prices, upper, lower, threshold=0.02):
    """Compute ping-pongs for a single window."""
    bounces = 0
    last_touch = 0  # 0=none, 1=upper, 2=lower

    for i in range(len(prices)):
        p = prices[i]
        u = upper[i]
        l = lower[i]

        upper_dist = abs(p - u) / u if u > 0 else 1.0
        lower_dist = abs(p - l) / abs(l) if l != 0 else 1.0

        if upper_dist <= threshold:
            if last_touch == 2:
                bounces += 1
            last_touch = 1
        elif lower_dist <= threshold:
            if last_touch == 1:
                bounces += 1
            last_touch = 2

    return bounces
```

**Solution for 5min Fast Path:**
```python
# In _calculate_5min_channel_features_rolling()
# Already have sliding windows: close_windows, high_windows, low_windows (n_windows, window_size)
# Already computed: upper, lower arrays (n_windows,) at each window's end position

# Need to compute full upper/lower lines for each window position
# This requires computing the regression line at each x position within the window

# For each window i, channel boundaries are:
#   upper[i, j] = high_slope[i] * j + high_intercept[i]  (adjusted with residual_std)
#   lower[i, j] = low_slope[i] * j + low_intercept[i]  (adjusted with residual_std)

# Then call ping-pong detection on each window
ping_pongs = np.zeros(n_windows, dtype=np.int32)
for i in range(n_windows):
    upper_line = high_slope[i] * x + high_intercept[i]
    lower_line = low_slope[i] * x + low_intercept[i]
    upper_line = np.maximum(upper_line, close_intercept[i] + close_slope[i] * x + 2 * residual_std[i])
    lower_line = np.minimum(lower_line, close_intercept[i] + close_slope[i] * x - 2 * residual_std[i])

    ping_pongs[i] = _compute_ping_pongs_single(
        close_windows[i], upper_line, lower_line, threshold=0.02
    )
```

**Performance Note:**
- Regular path: O(periods × window_size) - acceptable since periods << 5min bars
- 5min path: O(n_windows × window_size) - may need Numba batch processing for speed

#### 2.3 Integration Points

**Regular path:** After computing channel boundaries, call ping-pong/cycle functions.

**5min fast path:** Same approach using sliding_window_view data.

#### 2.4 Add to Output Dict

```python
# Ping-pongs (4 thresholds)
f'{prefix}_ping_pongs': np.zeros(n_5min, dtype=np.float32),
f'{prefix}_ping_pongs_0_5pct': np.zeros(n_5min, dtype=np.float32),
f'{prefix}_ping_pongs_1_0pct': np.zeros(n_5min, dtype=np.float32),
f'{prefix}_ping_pongs_3_0pct': np.zeros(n_5min, dtype=np.float32),

# Complete cycles (4 thresholds)
f'{prefix}_complete_cycles': np.zeros(n_5min, dtype=np.float32),
f'{prefix}_complete_cycles_0_5pct': np.zeros(n_5min, dtype=np.float32),
f'{prefix}_complete_cycles_1_0pct': np.zeros(n_5min, dtype=np.float32),
f'{prefix}_complete_cycles_3_0pct': np.zeros(n_5min, dtype=np.float32),
```

---

### Task 3: Update quality_score After Cycles Implemented

Once `complete_cycles` is computed, update quality_score calculation:

```python
# From linear_regression.py line 68:
# quality_score = cycles × (0.5 + 0.5 × r²)
quality_score = complete_cycles * (0.5 + 0.5 * r_squared_avg)
```

---

### Task 4: Bump Version for Cache Invalidation

**File:** `src/ml/features.py` line 38

```python
# OLD
PARTIAL_BAR_VERSION = "pb1.0"

# NEW
PARTIAL_BAR_VERSION = "pb2.0"  # Added 22 missing channel features
```

This will invalidate all caches and force regeneration with new features.

---

### Task 5: Verify Feature Count

After implementation, verify:

```python
# Expected features per window: 34
# Windows: 14 (from config.CHANNEL_WINDOW_SIZES)
# Timeframes: 11 (5min, 15min, 30min, 1h, 2h, 3h, 4h, daily, weekly, monthly, 3month)
# Symbols: 2 (tsla, spy)
# Total: 34 × 14 × 11 × 2 = 10,472 channel features
```

---

## Implementation Order

1. **Task 1.1:** Add EASY features to regular path (non-5min TFs)
2. **Task 1.2:** Add EASY features to 5min fast path
3. **Task 2:** Add ping-pongs and complete_cycles (HARD features)
4. **Task 3:** Update quality_score calculation
5. **Task 4:** Bump PARTIAL_BAR_VERSION
6. **Run test extraction** to verify:
   - Feature count matches expected
   - No NaN values
   - Values are reasonable (not all zeros)
7. **Run full pipeline** to verify:
   - Breakdown features still work
   - Labels generate correctly
   - Training runs without errors

---

## Testing Checklist

After implementation:

- [ ] Feature count = 34 per window (was 12)
- [ ] All TFs produce features (5min through monthly)
- [ ] Both symbols (tsla, spy) have features
- [ ] No NaN in any feature column
- [ ] `position` values vary (not all 0.5)
- [ ] `ping_pongs` values are integers >= 0
- [ ] `complete_cycles` values are integers >= 0
- [ ] `is_bull + is_bear + is_sideways ≈ 1` (mutually exclusive)
- [ ] `projected_high > projected_low` (channel makes sense)
- [ ] Breakdown features calculate without errors
- [ ] Labels generate without errors
- [ ] Training batch 0 completes without crash

---

## Estimated Effort

| Task | Complexity | Estimated Time |
|------|------------|----------------|
| Task 1.1 (EASY features, regular path) | Medium | 45 min |
| Task 1.2 (EASY features, 5min path) | Medium | 30 min |
| Task 2 (ping-pongs/cycles) | Hard | 90 min |
| Task 3 (quality_score) | Easy | 5 min |
| Task 4 (version bump) | Easy | 2 min |
| Testing | Medium | 30 min |
| **Total** | | **~3.5 hours** |

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Ping-pong vectorization slow | Medium | Use Numba JIT (already proven in linear_regression.py) |
| Memory increase | Low | Features are float32, minimal impact |
| Cache invalidation issues | Low | Version bump forces regeneration |
| Breakdown features break | Very Low | They only use stability/position which exist |

---

## Appendix: Feature Name Reference

For exact naming compatibility, features must match this pattern:
```
{symbol}_channel_{tf}_w{window}_{feature_name}

Examples:
tsla_channel_daily_w50_position
spy_channel_weekly_w100_ping_pongs
tsla_channel_5min_w20_complete_cycles_1_0pct
```

This matches the existing naming in both old and new systems.
