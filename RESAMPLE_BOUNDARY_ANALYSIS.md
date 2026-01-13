# Resample Boundary Issue - Detailed Analysis

## Executive Summary

**CRITICAL BUG CONFIRMED**: The parallel optimization in `v7/training/labels.py` introduces lookahead bias by including future data in resampled timeframes when channels end mid-day.

**Impact**: Walk-forward validation results are contaminated with future information, making performance metrics unreliable.

**Root Cause**: The `_try_get_precomputed_slice()` function uses `searchsorted(side='right')` to slice precomputed daily (and other timeframe) bars, which includes complete daily bars even when the channel ends mid-day. This complete bar contains data from later in the day that hasn't occurred yet from the perspective of the channel end position.

---

## Question 1: When does a daily bar "close"?

**Answer**: Daily bars are labeled at **MIDNIGHT (00:00)** of each calendar day.

The pandas `resample('1D')` function:
- Groups all data from a calendar day (00:00:00 to 23:59:59)
- Labels the aggregated bar with the day's midnight timestamp
- The 'close' field of the daily bar is the 'close' from the **last 5-minute bar** in that day

**Example**:
```python
# 5min data from 2024-01-02 14:30 to 20:55 (78 bars, full trading day)
# Resamples to:
# 2024-01-02 00:00:00 - Daily bar (close = value from 20:55)
```

**Key Point**: The daily bar timestamp (00:00) comes BEFORE the actual data (14:30-20:55). This is standard pandas behavior.

---

## Question 2: If historical data ends at 14:30, what does resample_ohlc() include in that day's bar?

**Answer**: It creates a **PARTIAL daily bar** containing only data up to 14:30.

**Sequential Approach Example**:
```python
# Data ends at position i=50 (2024-01-02 18:40)
df_partial = df[:51]  # Includes only bars 0-50

# Resample to daily
daily_partial = df_partial.resample('1D').agg({...})

# Result:
# 2024-01-02 00:00:00 - Daily bar (close = value from 18:40, NOT 20:55)
```

**Behavior**:
- The daily bar still has timestamp 2024-01-02 00:00:00
- But the 'close' value comes from 18:40 (last available bar)
- The daily bar represents only the partial day up to 18:40

---

## Question 3: What does the precomputed full resample include?

**Answer**: It includes **COMPLETE daily bars** with all data from the full trading day.

**Parallel Approach (Precomputed)**:
```python
# Full data for entire day (bars 0-77, ending at 20:55)
df_full = df  # All bars

# Precompute resample ONCE
daily_full = df_full.resample('1D').agg({...})

# Result:
# 2024-01-02 00:00:00 - Daily bar (close = value from 20:55)
```

**Critical Difference**:
- The precomputed daily bar uses the LAST bar from the day (20:55)
- This is different from the sequential approach at position i=50 (18:40)
- The precomputed bar contains data from bars 51-77, which are **FUTURE** relative to position 50

---

## Question 4: How does searchsorted with side='right' work exactly?

**Answer**: `searchsorted(value, side='right')` returns the insertion point **AFTER** any existing values equal to `value`.

**Concrete Example**:
```python
daily_index = [
    '2024-01-02 00:00',  # idx 0
    '2024-01-03 00:00',  # idx 1
    '2024-01-04 00:00',  # idx 2
]

# Test different end timestamps:
daily_index.searchsorted('2024-01-02 14:30', side='right')  # Returns 1
daily_index.searchsorted('2024-01-02 23:59', side='right')  # Returns 1
daily_index.searchsorted('2024-01-03 00:00', side='right')  # Returns 2 (AFTER match)
daily_index.searchsorted('2024-01-03 14:30', side='right')  # Returns 2
```

**Key Insight**: When slicing `[:idx]`, we include indices `0` to `idx-1`.

**Problem**: For a channel ending at 2024-01-02 18:40:
- `searchsorted('2024-01-02 18:40', side='right')` returns 1
- Slice `[:1]` includes daily bar at index 0
- This bar has timestamp '2024-01-02 00:00' but contains data up to 20:55
- **We include future data** (18:40 to 20:55) in the resampled timeframe

---

## Question 5: Is there a concrete example where parallel includes future data that sequential doesn't?

**YES - Confirmed with Concrete Timestamps**

### Scenario Setup

**Position**: `i=50` (channel ends at this position)
**Timestamp**: `2024-01-02 18:40:00` (mid-day, not at market close)
**Full day**: 78 bars from 14:30 to 20:55 (bars 0-77)

### Sequential Approach (CORRECT)

```python
df_sequential = df[:51]  # Bars 0-50
daily_sequential = df_sequential.resample('1D').agg({...})

# Result:
# 2024-01-02 00:00:00
# close: 101.0000 (from bar 50 at 18:40)
```

**Data included**: Only bars 0-50 (up to and including position i=50)

### Parallel Approach (LOOKAHEAD BUG)

```python
# Precompute full resample
daily_full = df.resample('1D').agg({...})

# Result:
# 2024-01-02 00:00:00
# close: 101.2700 (from bar 77 at 20:55)

# Slice by timestamp
end_timestamp = df.index[50]  # 2024-01-02 18:40
idx = daily_full.index.searchsorted(end_timestamp, side='right')  # Returns 1
daily_parallel = daily_full.iloc[:1]

# Result:
# 2024-01-02 00:00:00
# close: 101.2700 (from bar 77 at 20:55) ← FUTURE DATA!
```

**Data included**: Bars 0-77 (includes bars 51-77 which are FUTURE relative to position 50)

### The Difference

| Metric | Sequential | Parallel | Difference |
|--------|-----------|----------|------------|
| Daily close | 101.0000 | 101.2700 | **0.2700** |
| Last bar used | 50 (18:40) | 77 (20:55) | **27 bars of future data** |
| Lookahead? | No | **YES** | **CRITICAL BUG** |

**Impact**: The parallel approach includes 27 bars (2 hours and 15 minutes) of future price data that hasn't occurred yet from the perspective of the channel at position i=50.

---

## Code Analysis

### Location of Bug

**File**: `/Users/frank/Desktop/CodingProjects/x9/v7/training/labels.py`
**Function**: `_try_get_precomputed_slice()` (lines 209-245)

### Problematic Code

```python
def _try_get_precomputed_slice(df: pd.DataFrame, timeframe: str) -> Optional[pd.DataFrame]:
    """..."""
    # Check if pre-computed TSLA data is available
    precomputed_tsla = getattr(_precomputed_local, 'tsla', None)
    if precomputed_tsla is not None and timeframe in precomputed_tsla:
        precomputed_df = precomputed_tsla[timeframe]
        if precomputed_df is not None and len(precomputed_df) > 0 and len(df) > 0:
            # Get end timestamp from input df
            end_timestamp = df.index[-1]

            # Find all bars whose start time is <= end_timestamp
            # Use 'right' to get the position after the last matching timestamp
            idx = precomputed_df.index.searchsorted(end_timestamp, side='right')

            if idx > 0:
                return precomputed_df.iloc[:idx]  # ← BUG: Includes complete bars with future data

    return None
```

### Why It's Wrong

The comment says "Find all bars whose start time is <= end_timestamp", but this is **incorrect** for partial periods:

1. Daily bars have timestamp at 00:00 (start of day)
2. When `end_timestamp = 18:40`, the daily bar's timestamp (00:00) is indeed `<=` 18:40
3. But the daily bar's **data** extends to 20:55, which is **>** 18:40
4. By including this bar, we include future data

### Existing Comment Acknowledges Issue

Lines 216-224 contain a comment that acknowledges the problem but dismisses it:

```python
# IMPORTANT: The pre-computed approach is mathematically equivalent to fresh
# resampling for all COMPLETE time periods. For the last partial period,
# there may be a minor difference if the pre-computed data includes more
# bars in that period. This is acceptable for the use case (channel detection
# and feature extraction) where slight differences in the final partial bar
# don't materially affect results.
```

**This justification is WRONG for walk-forward validation**:
- The "minor difference" is actually **future data leakage**
- For daily bars, this can be 2+ hours of future price movement
- This is NOT acceptable for backtesting or validation
- The comment assumes the use case doesn't require strict time boundaries, but walk-forward validation DOES

---

## Impact on Walk-Forward Validation

### Where This Matters

The bug affects **label generation** in walk-forward validation:

1. We're at position `i` in the 5min data (channel end)
2. We generate features using data up to position `i` (correct)
3. We generate labels by scanning forward from `i` (correct intent)
4. BUT: We resample to longer timeframes using precomputed data
5. The precomputed daily bar at position `i` includes future data
6. This future data leaks into features derived from longer timeframes

### Affected Features

Any feature that uses resampled timeframes is affected:
- Daily/weekly/monthly OHLC values
- Multi-timeframe channel detection
- Longer timeframe momentum/volatility calculations
- Break trigger TF detection (which uses longer timeframe channels)

### Example Contamination Path

```
Position i=50 (2024-01-02 18:40)
↓
Generate features for 1h timeframe
↓
Resample to 1h using precomputed data
↓
Precomputed 1h bar includes data beyond 18:40
↓
Features use future price information
↓
Model learns to "predict" using data it shouldn't have
↓
Inflated validation performance
```

---

## Solutions

### Option 1: Disable Parallel Optimization (Safest, Slow)

```python
def _try_get_precomputed_slice(df: pd.DataFrame, timeframe: str) -> Optional[pd.DataFrame]:
    # DISABLE: Always return None to force fresh resampling
    return None
```

**Pros**:
- Guarantees no lookahead
- Simple, one-line fix
- Matches sequential behavior exactly

**Cons**:
- Loses performance optimization
- Redundant resampling across positions

### Option 2: Smart Hybrid Slicing (Fast, Complex)

```python
def _try_get_precomputed_slice(df: pd.DataFrame, timeframe: str) -> Optional[pd.DataFrame]:
    """..."""
    if timeframe == '5min':
        return None

    precomputed_tsla = getattr(_precomputed_local, 'tsla', None)
    if precomputed_tsla is not None and timeframe in precomputed_tsla:
        precomputed_df = precomputed_tsla[timeframe]
        if precomputed_df is not None and len(precomputed_df) > 0 and len(df) > 0:
            end_timestamp = df.index[-1]

            # Check if end_timestamp is exactly on a timeframe boundary
            # For daily: check if it's at market close (e.g., 16:00 ET)
            # If not on boundary, need to re-resample the partial period

            # Get the resampled bar that would contain end_timestamp
            idx = precomputed_df.index.searchsorted(end_timestamp, side='right')

            if idx > 0:
                # Check if this is a partial bar (end_timestamp is mid-period)
                bar_timestamp = precomputed_df.index[idx - 1]

                # Use side='left' to find the NEXT bar boundary
                next_bar_idx = precomputed_df.index.searchsorted(end_timestamp, side='left')

                # If searchsorted left and right give different results, we're mid-period
                if next_bar_idx != idx:
                    # Partial period - need to re-resample the last bar
                    complete_bars = precomputed_df.iloc[:idx-1] if idx > 1 else pd.DataFrame()

                    # Re-resample just the partial period
                    partial_df = df[df.index.date == end_timestamp.date()]
                    partial_resampled = resample_ohlc(partial_df, timeframe)

                    # Combine complete bars + partial bar
                    if len(complete_bars) > 0:
                        result = pd.concat([complete_bars, partial_resampled])
                    else:
                        result = partial_resampled

                    return result
                else:
                    # On exact boundary - safe to use precomputed
                    return precomputed_df.iloc[:idx]

    return None
```

**Pros**:
- Preserves performance optimization for complete periods
- Correct behavior for partial periods
- No lookahead bias

**Cons**:
- Complex logic
- Requires careful testing for edge cases
- Boundary detection must be timezone-aware

### Option 3: Boundary-Only Optimization (Compromise)

```python
def _try_get_precomputed_slice(df: pd.DataFrame, timeframe: str) -> Optional[pd.DataFrame]:
    """..."""
    if timeframe == '5min':
        return None

    precomputed_tsla = getattr(_precomputed_local, 'tsla', None)
    if precomputed_tsla is not None and timeframe in precomputed_tsla:
        precomputed_df = precomputed_tsla[timeframe]
        if precomputed_df is not None and len(precomputed_df) > 0 and len(df) > 0:
            end_timestamp = df.index[-1]

            # Only use precomputed if end_timestamp is exactly on a timeframe boundary
            idx = precomputed_df.index.searchsorted(end_timestamp, side='right')

            if idx > 0:
                # Check if end_timestamp matches a timeframe boundary exactly
                if idx < len(precomputed_df) and precomputed_df.index[idx - 1] == end_timestamp:
                    # Exact match - safe to use precomputed
                    return precomputed_df.iloc[:idx]

            # Not on boundary - fall back to fresh resampling
            return None

    return None
```

**Pros**:
- Simple to implement
- Preserves optimization for many cases
- Safe fallback for edge cases

**Cons**:
- Still does redundant resampling for non-boundary positions
- Not as fast as full optimization

---

## Recommendation

**Immediate Action**: Disable precomputed optimization (Option 1)
- Change line 243 in `labels.py` to `return None`
- Verify walk-forward results change
- Document the difference in validation metrics

**Future Enhancement**: Implement smart hybrid (Option 2)
- After confirming the bug impact
- With comprehensive test coverage
- Include timezone handling

**Testing Strategy**:
1. Create unit test with concrete timestamps (see investigation script)
2. Compare sequential vs parallel label generation
3. Assert equal values for all positions
4. Test daily/hourly/weekly timeframes
5. Test boundary and mid-period positions

---

## Verification Steps

To confirm this bug affects your results:

1. Run walk-forward validation with current code (parallel enabled)
2. Disable precomputed optimization (return None)
3. Re-run walk-forward validation (now fully sequential)
4. Compare metrics - if different, confirms lookahead contamination

**Expected Impact**:
- Performance metrics should **decrease** when bug is fixed
- The decrease represents the removal of lookahead information
- Larger decrease = more significant contamination

---

## Related Code Locations

1. **Bug Location**: `v7/training/labels.py:209-245` (`_try_get_precomputed_slice()`)
2. **Used By**: `v7/training/labels.py:157-206` (`cached_resample_ohlc()`)
3. **Initialized In**: `v7/training/scanning.py:42-66` (`_init_scan_worker()`)
4. **Called From**: Dataset generation with `parallel=True`

---

## Timestamp Evidence

See `/Users/frank/Desktop/CodingProjects/x9/investigate_resample_boundaries.py` for:
- Concrete examples with real timestamps
- Step-by-step demonstration of the bug
- Comparison of sequential vs parallel approaches
- Exact difference in values (0.27 in the example)

**Key Output**:
```
Position: i=50 (timestamp: 2024-01-02 18:40:00)

SEQUENTIAL daily 'close': 101.0000
PARALLEL daily 'close':   101.2700

Difference: 0.2700

*** LOOKAHEAD DETECTED! ***
*** Parallel includes future data that sequential doesn't! ***
```

This is a reproducible, deterministic demonstration of future data leakage.

---

## Conclusion

The parallel optimization introduces **measurable lookahead bias** by including complete timeframe bars that contain future data. This affects walk-forward validation results and any evaluation that depends on strict time boundaries.

**Action Required**: Disable or fix the optimization before trusting validation metrics.
