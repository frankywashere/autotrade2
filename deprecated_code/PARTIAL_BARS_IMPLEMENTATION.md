# Partial Bars Implementation Plan (v5.4)

## Overview

This document contains the complete implementation plan for adding partial bar support to the channel/feature calculation pipeline. At each 5min timestamp, features for coarser TFs (daily, weekly, monthly, 3month) will include the "in-progress" bar data.

**Core modules already implemented:**
- `src/ml/partial_bars.py` - Computes partial OHLCV at each 5min timestamp
- `src/ml/partial_channel_calc_vectorized.py` - Vectorized channel calculation with partial bars

## Remaining Tasks

### Task 1: Integrate Partial Channels into features.py

**File:** `src/ml/features.py`

**What to change:** Replace `_extract_channel_features` to use the vectorized partial bar approach.

**Key insight:** The current function resamples to TF bars, calculates channels, then ffills back. The new approach calculates directly at 5min with partial bars included.

**Implementation:**

```python
# In _extract_channel_features, add a new branch for partial bar mode:

def _extract_channel_features(self, df: pd.DataFrame, use_cache: bool = True,
                               multi_res_data: dict = None, use_gpu: bool = False,
                               cache_suffix: str = None,
                               use_partial_bars: bool = True) -> pd.DataFrame:  # NEW PARAM
    """
    Extract channel features.
    If use_partial_bars=True, uses vectorized partial bar calculation.
    """
    # ... cache loading logic stays same ...

    if use_partial_bars and not is_live_mode:
        # NEW: Use vectorized partial bar calculation
        from .partial_channel_calc_vectorized import calculate_all_channel_features_vectorized

        all_results = []
        for symbol in ['tsla', 'spy']:
            for tf_name, tf_rule in timeframes.items():
                result = calculate_all_channel_features_vectorized(
                    df, symbol, tf_name, tf_rule,
                    windows=config.CHANNEL_WINDOW_SIZES,
                    show_progress=True
                )
                all_results.append(result)

        channel_features = pd.concat(all_results, axis=1)

        # Save to cache
        if use_cache:
            with open(cache_file, 'wb') as f:
                pickle.dump(channel_features, f)

        return channel_features

    else:
        # Original ffill-based approach (for live mode or backward compat)
        # ... existing code ...
```

**Cache key:** Update `FEATURE_VERSION` to `v5.4.0` to invalidate old caches.

---

### Task 2: RSI with Partial Bars

**File:** `src/ml/features.py` function `_extract_rsi_features`

**Current problem:** RSI is calculated on complete TF bars, then ffilled. Need to include partial bar.

**Solution:** Similar to channels - compute partial bar OHLCV, append to history, calculate RSI.

**Implementation:**

```python
def _extract_rsi_features(self, df: pd.DataFrame, multi_res_data: dict = None,
                          use_partial_bars: bool = True) -> pd.DataFrame:
    """Extract RSI features with optional partial bar inclusion."""
    from .partial_bars import compute_partial_bars

    rsi_features = {}
    num_rows = len(df)

    for symbol in ['tsla', 'spy']:
        symbol_df = df[[c for c in df.columns if c.startswith(f'{symbol}_')]].copy()
        symbol_df.columns = [c.replace(f'{symbol}_', '') for c in symbol_df.columns]

        for tf_name, tf_rule in timeframes.items():
            prefix = f'{symbol}_rsi'

            if use_partial_bars and tf_name != '5min':
                # Compute partial bar state
                partial_state = compute_partial_bars(symbol_df, tf_name)

                # Get complete TF bars
                resampled = symbol_df.resample(tf_rule).agg({...}).dropna()

                # For each 5min bar, find complete bars before it + partial
                # Calculate RSI on that series
                # This is similar to channel calculation logic

                rsi_values = np.zeros(num_rows)
                # ... vectorized calculation similar to channels ...

                rsi_features[f'{prefix}_{tf_name}'] = rsi_values
            else:
                # Original approach for 5min or backward compat
                rsi_series = self.rsi_calc.calculate_rsi(resampled)
                rsi_aligned = rsi_series.reindex(df.index, method='ffill').bfill().fillna(50.0)
                rsi_features[f'{prefix}_{tf_name}'] = rsi_aligned.values

            # Oversold/overbought flags
            rsi_features[f'{prefix}_{tf_name}_oversold'] = (rsi_features[f'{prefix}_{tf_name}'] < 30).astype(float)
            rsi_features[f'{prefix}_{tf_name}_overbought'] = (rsi_features[f'{prefix}_{tf_name}'] > 70).astype(float)

    return pd.DataFrame(rsi_features, index=df.index)
```

---

### Task 3: Breakdown Features at 5min Resolution

**File:** `src/ml/features.py`

**Current:** `_calculate_breakdown_at_native_tf` calculates at TF resolution, then `_precompute_timeframe_sequences` broadcasts via ffill (Pass 2).

**New:** Calculate breakdown at 5min resolution directly. No ffill needed.

**Changes:**

1. **Remove Pass 2** in `_precompute_timeframe_sequences` - no more cross-TF broadcast
2. **Calculate breakdown at 5min** using the new channel features that already include partial bars

```python
# In _precompute_timeframe_sequences, simplify:

# OLD Pass 2: Add cross-TF breakdown features to each file
# DELETE THIS - no longer needed

# NEW: Breakdown is calculated from 5min channel features directly
# Since channels now include partial bars, breakdown inherits that
```

The breakdown features (duration_ratio, alignment, time_in_channel, etc.) can be calculated from the 5min channel features which already have partial bar data.

---

### Task 4: Labels at 5min Resolution

**File:** `src/ml/features.py` function `generate_hierarchical_continuation_labels`

**Current:** Labels are generated per TF bar (e.g., 561 weekly labels), then looked up by timestamp.

**New:** Generate labels at 5min resolution. Each 5min bar gets its own continuation label based on its rolling channel.

**Key semantic change:**
- OLD: "Will LAST WEEK's complete channel continue?"
- NEW: "Will the channel I'm currently in (including today's data) continue?"

**Implementation:**

```python
def generate_hierarchical_continuation_labels_5min(
    self,
    df: pd.DataFrame,  # 5min data
    timeframes: list = None,
    output_dir: Path = None,
    cache_suffix: str = None
) -> Dict[str, Path]:
    """
    Generate continuation labels at 5min resolution.
    Each 5min bar gets labels for each TF's rolling channel.
    """
    from .partial_bars import compute_partial_bars

    for tf in timeframes:
        # For each 5min bar:
        # 1. Get the channel that includes partial bar (already calculated)
        # 2. Look forward to find when channel breaks
        # 3. Store duration, gain, confidence

        partial_state = compute_partial_bars(df, tf)

        # Use precomputed channel features (with partial)
        # Determine break points (when price exits channel bounds)
        # Calculate label values

        labels = []
        for i in range(len(df)):
            # Get current channel bounds (from features)
            # Scan forward until break
            # Record duration, max gain, etc.
            pass

        # Save at 5min resolution
        output_path = output_dir / f"continuation_labels_5min_{tf}_{cache_suffix}.pkl"
```

**Note:** This is the most complex change because it affects the semantic meaning of labels.

---

### Task 5: Update Warmup Configuration

**File:** `src/ml/features.py` or `config.py`

**Current warmup:** ~3 months (enough for 200-bar lookback at 1min)

**New warmup needed:** ~10 months (need 5148 5min bars for 3month w50 rolling window)

**Calculation:**
- 3month = ~66 trading days × 78 5min bars = 5148 bars
- Window = 50 bars of 3month = 50 × 5148 = 257,400 5min bars
- That's ~330 trading days = ~15 months!

Actually, the window is in TF bars, not 5min bars. So:
- 3month w50 = 50 complete 3-month bars + 1 partial
- Each 3-month period = ~66 trading days
- Need 50 × 66 = 3300 trading days = ~13 years of history!

**Wait** - this is the current approach too. The difference is:
- Current: ffill means first valid label appears after 50 complete 3month bars
- New: same requirement, but values update within each period

The warmup requirement doesn't actually change for training - we still need enough history for the longest window. The difference is that within each period, values now evolve.

**Action:** Keep current warmup, but ensure early bars have `insufficient_data=1` flag set.

---

### Task 6: Update hierarchical_dataset.py

**File:** `src/ml/hierarchical_dataset.py`

**Current:** Labels are loaded per-TF, then looked up using `searchsorted` by timestamp.

**New:** Labels are at 5min resolution, direct index lookup.

**Changes:**

```python
# In _load_hierarchical_continuation_labels:

# OLD: Load per-TF labels, build ts_to_idx lookup
# NEW: Load 5min labels directly

def _load_hierarchical_continuation_labels(self, labels_dir: Path, cache_suffix: str):
    """Load continuation labels at 5min resolution."""
    for tf in HIERARCHICAL_TIMEFRAMES:
        # NEW: Load 5min-resolution labels
        label_file = labels_dir / f"continuation_labels_5min_{tf}_{cache_suffix}.pkl"
        if label_file.exists():
            with open(label_file, 'rb') as f:
                labels = pickle.load(f)
            self._per_tf_continuation[tf] = labels
            # No ts_to_idx needed - direct index access

# In __getitem__:

# OLD:
#   tf_idx = np.searchsorted(tf_timestamps, ts_5min, side='right') - 1
#   row_idx = self._per_tf_ts_to_idx[tf].get(int(tf_ts))

# NEW:
#   row_idx = data_idx_5min  # Direct index!
#   targets[f'cont_{tf}_duration'] = self._per_tf_continuation[tf]['duration_bars'][row_idx]
```

---

## Implementation Order

1. ✅ **Update FEATURE_VERSION** to `v5.4.0` in features.py
2. ✅ **Integrate channels** - `_extract_channel_features` now uses `calculate_all_channel_features_vectorized`
3. ✅ **Update RSI** - `_extract_rsi_features` with `_calculate_rsi_with_partial_bars`
4. ✅ **Simplify breakdown** - New `_calculate_all_breakdown_at_5min`, removed Pass 2
5. ✅ **Regenerate labels** - New `generate_hierarchical_continuation_labels_5min`
6. ✅ **Update dataset** - `_load_hierarchical_continuation_labels` and `__getitem__` support 5min labels
7. **Delete old cache** - Force regeneration with new approach
8. **Full integration test** - Run training, verify no NaN, check loss converges

---

## Testing Checkpoints

After each major change, verify:

1. **Channels:** Position values change during the day (not constant like ffill)
2. **RSI:** RSI values evolve within each TF period
3. **Breakdown:** No NaN in breakdown features
4. **Labels:** Labels exist at 5min resolution
5. **Training:** No crash at batch 0, loss decreases

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/ml/features.py` | Channel extraction, RSI, breakdown, labels, version |
| `src/ml/hierarchical_dataset.py` | Label loading, lookup |
| `config.py` | Maybe warmup settings |

## Files Already Created

| File | Purpose |
|------|---------|
| `src/ml/partial_bars.py` | Compute partial OHLCV at each 5min |
| `src/ml/partial_channel_calc_vectorized.py` | Vectorized channel calc with partial |

---

## Quick Reference: Key Functions

```python
# Compute partial bars
from src.ml.partial_bars import compute_partial_bars
partial_state = compute_partial_bars(df, 'weekly')
# Returns: PartialBarState with partial_open, partial_high, partial_low, partial_close

# Vectorized channel calculation
from src.ml.partial_channel_calc_vectorized import calculate_all_channel_features_vectorized
result = calculate_all_channel_features_vectorized(df, 'tsla', 'weekly', '1W', windows=[50])
# Returns: DataFrame at 5min resolution with channel features
```

---

## Estimated Time

| Task | Time |
|------|------|
| Integrate channels | 30 min |
| Update RSI | 30 min |
| Simplify breakdown | 20 min |
| Regenerate labels | 45 min |
| Update dataset | 30 min |
| Testing | 30 min |
| **Total** | ~3 hours |
