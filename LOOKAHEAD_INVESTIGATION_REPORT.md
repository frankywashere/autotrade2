# Precomputed Resampling Lookahead Risk Investigation

## Executive Summary

**VERDICT: CONFIRMED LOOKAHEAD BUG**

The precomputed resampling optimization in `v7/training/labels.py:209` (`_try_get_precomputed_slice`) contains a real lookahead bug that includes future data in partial daily bars. This is NOT a false alarm.

## Investigation Results

### 1. How `_try_get_precomputed_slice()` Uses `searchsorted`

**Location:** `/Users/frank/Desktop/CodingProjects/x9/v7/training/labels.py:209-245`

```python
def _try_get_precomputed_slice(df: pd.DataFrame, timeframe: str) -> Optional[pd.DataFrame]:
    # ...
    # Line 236: Get end timestamp from input df
    end_timestamp = df.index[-1]

    # Line 239-240: Find all bars whose start time is <= end_timestamp
    # Use 'right' to get the position after the last matching timestamp
    idx = precomputed_df.index.searchsorted(end_timestamp, side='right')

    # Line 242-243: Return slice
    if idx > 0:
        return precomputed_df.iloc[:idx]
```

**How it works:**
1. Takes the last timestamp from historical data (e.g., `2024-01-02 14:30:00`)
2. Uses `searchsorted(side='right')` to find position in precomputed daily index
3. Returns all daily bars up to that position using slice `[:idx]`

### 2. What `side='right'` Means for `searchsorted`

`searchsorted(side='right')` finds the insertion point **AFTER** all elements ≤ query.

**Example with daily bars:**
```
Daily index: [2024-01-02 00:00:00, 2024-01-03 00:00:00, 2024-01-04 00:00:00]
Query: 2024-01-02 14:30:00 (midday timestamp)

side='right': idx=1 (position after Jan 2 bar)
side='left':  idx=1 (position at Jan 2 bar)

Slice [:1] includes: Jan 2 bar at 00:00:00
```

Both `side='right'` and `side='left'` return idx=1 when the query (14:30) falls **between** index elements.

### 3. Concrete Example: Historical Data Ends at 14:30

**Scenario:** We're detecting a channel at 5-min bar 60 (14:30 on Jan 2).

#### Historical Data
- **Time range:** 9:30 to 14:30
- **Bars included:** 61 bars (0-60)
- **Trading day progress:** 78% complete

#### Method 1: Direct Resample (CORRECT)
```python
df_historical = df_5min.iloc[:61]  # 9:30 to 14:30
df_daily = resample_ohlc(df_historical, 'daily')
```

**Result for Jan 2:**
- Open:   99.98 (first 5-min bar at 9:30)
- High:   101.41 (max from 9:30 to 14:30)
- Low:    98.89 (min from 9:30 to 14:30)
- Close:  99.55 (close at 14:30)
- Volume: 371,981 (sum from 9:30 to 14:30)

**Data used:** Only bars 0-60 (9:30 to 14:30) ✓ NO LOOKAHEAD

#### Method 2: Precomputed Slice (LOOKAHEAD BUG)
```python
df_daily_full = resample_ohlc(df_5min, 'daily')  # Precompute full days
end_timestamp = df_historical.index[-1]  # 14:30
idx = df_daily_full.index.searchsorted(end_timestamp, side='right')  # idx=1
df_daily = df_daily_full.iloc[:idx]  # Include Jan 2 bar
```

**Result for Jan 2:**
- Open:   99.98 (first 5-min bar at 9:30)
- High:   110.06 (max from 9:30 to **16:00**)
- Low:    98.89 (min from 9:30 to **16:00**)
- Close:  109.90 (close at **16:00**)
- Volume: 469,248 (sum from 9:30 to **16:00**)

**Data used:** All bars 0-77 (9:30 to 16:00) ✗ INCLUDES FUTURE DATA

#### Lookahead Data
**Future bars included:** 61-77 (14:35 to 16:00) - **17 bars of future data**

**Differences (Precomputed - Historical):**
- Open:   +0.00 (same, uses first bar)
- High:   **+8.65** (includes afternoon high)
- Low:    +0.00 (morning was the low)
- Close:  **+10.35** (uses closing price at 16:00 instead of 14:30)
- Volume: **+97,267** (includes afternoon volume)

### 4. Comparison: Resampling `df[:i]` vs Using Precomputed Slice

| Aspect | Direct Resample `df[:i]` | Precomputed Slice |
|--------|--------------------------|-------------------|
| **Data Source** | Only historical bars (0 to i) | Full dataset (all bars) |
| **Partial Bar** | Correctly truncated at time i | Includes complete period |
| **High** | max(9:30 to 14:30) | max(9:30 to 16:00) ✗ |
| **Low** | min(9:30 to 14:30) | min(9:30 to 16:00) ✗ |
| **Close** | close at 14:30 | close at 16:00 ✗ |
| **Volume** | sum(9:30 to 14:30) | sum(9:30 to 16:00) ✗ |
| **Lookahead Risk** | ✓ None | ✗ Yes - 17 bars of future data |

### 5. Is This Actual Lookahead or False Alarm?

**VERDICT: ACTUAL LOOKAHEAD BUG**

#### Evidence

1. **Different OHLCV values:**
   - Close differs by 10.35 points (10.4% move)
   - High differs by 8.65 points
   - Volume differs by 97,267 shares (26% increase)

2. **Future data included:**
   - Historical: 61 bars (78% of trading day)
   - Precomputed: 78 bars (100% of trading day)
   - Lookahead: 17 bars (22% of trading day)

3. **Impact on model:**
   - **Channel detection:** Daily high/low range is wider with lookahead
   - **Channel std_dev:** Will be larger, affecting channel bounds
   - **Break detection:** May incorrectly classify breaks based on future data
   - **Features:** Any daily-derived features include future information

#### Why the Code Comment is Wrong

The comment at lines 216-224 states:
> "The pre-computed approach is mathematically equivalent to fresh resampling for all COMPLETE time periods. For the last partial period, there may be a minor difference if the pre-computed data includes more bars in that period."

This is **incorrect**. The difference is not "minor" - it's **future data contamination**:
- It's not just "more bars in that period"
- It's **all remaining bars until market close**
- This fundamentally changes the OHLC values
- This is textbook lookahead bias

## Where This Code is Used

### Call Chain

1. **Parallel workers initialized:** `v7/training/scanning.py:58`
   ```python
   set_precomputed_resampled_data(precomputed_tsla, precomputed_spy)
   ```

2. **Called during label generation:** `v7/training/labels.py:174`
   ```python
   def cached_resample_ohlc(df, timeframe):
       precomputed_slice = _try_get_precomputed_slice(df, timeframe)
       if precomputed_slice is not None:
           return precomputed_slice  # LOOKAHEAD BUG HERE
   ```

3. **Used in multiple places:**
   - Line 810: `get_longer_tf_channels()` - channel detection at longer timeframes
   - Line 1221: `generate_labels_per_tf()` - channel detection for historical data
   - Line 1224: `generate_labels_per_tf()` - forward scanning data
   - Line 1356: `generate_labels_multi_window()` - multi-window channel detection

### When Active

The precomputed optimization is active when:
- Running parallel position scanning
- `set_precomputed_resampled_data()` has been called with non-None data
- Processing TSLA or SPY data (the two symbols that have precomputed data)

## Impact Assessment

### Severity: HIGH

This affects:
1. **Training data quality:** Labels computed with lookahead bias
2. **Model reliability:** Trained on impossible-to-obtain information
3. **Backtesting accuracy:** Results are overly optimistic
4. **Production deployment:** Model won't perform as expected

### Affected Components

- ✗ Channel detection on daily/weekly/monthly timeframes (uses future high/low)
- ✗ Break direction classification (may see breaks that don't exist yet)
- ✗ Duration prediction (future volatility affects when breaks occur)
- ✗ Trigger TF detection (longer timeframe boundaries use future data)
- ✗ Feature extraction (any features derived from daily+ timeframes)

### Magnitude

- **Partial bars:** ~22% of trading day data is lookahead (14:30 to 16:00)
- **Impact varies by volatility:** Higher in afternoon session
- **Systematic bias:** Affects ALL samples except those at exact market close

## Recommendations

### Immediate Action Required

The precomputed optimization should be **disabled** or **fixed** before using this system for production training.

### Fix Options

**Option A: Disable Optimization (Safest)**
```python
def _try_get_precomputed_slice(df, timeframe):
    return None  # Always fall back to fresh resampling
```

**Option B: Fix Partial Bar Handling (Complex)**
- Detect when query timestamp is within a period (not at period start)
- Fall back to fresh resampling for partial bars
- Only use precomputed for complete periods

**Option C: Pre-compute Correctly (Expensive)**
- Pre-compute partial bars for every historical position
- Massive memory overhead (78 partial bars per day per position)
- Not practical

### Recommended Solution: Option A

Disable the optimization entirely:
1. **Correctness over speed:** Training data quality is paramount
2. **Minimal code changes:** One-line fix
3. **No edge cases:** Eliminates all lookahead risk
4. **Performance impact:** Slower but correct

The speed benefit is not worth the contaminated training data.

## Testing

Two test files created to demonstrate the bug:

1. **`test_searchsorted_lookahead.py`** - Basic searchsorted behavior
2. **`test_lookahead_impact.py`** - Full simulation of actual code flow

Both tests show concrete evidence of:
- Different OHLCV values between methods
- Future data inclusion in precomputed slice
- Impact on channel detection and features

Run with:
```bash
python3 test_searchsorted_lookahead.py
python3 test_lookahead_impact.py
```

## Conclusion

**This is a CONFIRMED lookahead bug, not a false alarm.**

The precomputed resampling optimization trades correctness for performance, introducing systematic lookahead bias into the training data. The partial daily bar includes data from 14:35 to 16:00 when historical data only goes to 14:30, contaminating channel detection, labels, and features with future information that would not be available in real-time trading.

**Action Required:** Disable or fix this optimization before using the system for production model training.
