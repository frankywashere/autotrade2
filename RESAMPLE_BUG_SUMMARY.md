# Resample Boundary Bug - Executive Summary

## The Problem in One Sentence

**The parallel optimization includes complete daily bars with future data when channels end mid-day, causing lookahead bias in walk-forward validation.**

---

## Visual Example

```
Timeline of a trading day (2024-01-02):

09:30  10:00  11:00  12:00  13:00  14:00  15:00  16:00
|------|------|------|------|------|------|------|------|
                            ^
                            |
                    Channel ends here (i=50, 18:40)
                            |
                            └─ SEQUENTIAL: Daily bar uses data up to HERE ✓

                    But full day continues...

                            |.........................|
                            └─ PARALLEL: Daily bar uses data up to HERE ✗
                                                      (includes 2.25 hours of future data)
```

---

## Concrete Numbers

**Test Case**: Channel at position i=50 (timestamp 2024-01-02 18:40:00)

| Approach | Daily Close | Last Bar Used | Future Data? |
|----------|-------------|---------------|--------------|
| Sequential | 101.0000 | Bar 50 (18:40) | No ✓ |
| Parallel | 101.2700 | Bar 77 (20:55) | **Yes - 27 bars** ✗ |

**Difference**: 0.27 points (0.27% of price) from 2.25 hours of future data

---

## How It Happens

### Sequential (Correct)
1. Channel ends at position i=50 (18:40)
2. Slice data: `df[:51]` (bars 0-50)
3. Resample to daily: Creates **partial day bar** with data up to 18:40
4. Daily close = 101.0000 (from bar 50)

### Parallel (Bug)
1. Channel ends at position i=50 (18:40)
2. Pre-resampled daily bar already computed with **full day data** (bars 0-77)
3. Slice by timestamp using searchsorted: Includes the complete daily bar
4. Daily close = 101.2700 (from bar 77 at 20:55)
5. **Bug**: Bars 51-77 are future data relative to position 50

---

## Why searchsorted Doesn't Help

```python
# Daily bars have timestamps at midnight
daily_bars = [
    '2024-01-02 00:00',  # Bar contains data from 09:30-16:00
    '2024-01-03 00:00',  # Bar contains data from 09:30-16:00
]

# Channel ends mid-day
end_time = '2024-01-02 18:40'

# searchsorted finds: Which daily bar?
idx = daily_bars.searchsorted(end_time, side='right')  # Returns 1

# Slice includes bar 0
daily_bars[:1]  # ['2024-01-02 00:00']

# But this bar contains data BEYOND 18:40!
# The bar timestamp (00:00) is BEFORE end_time (18:40)
# But the bar's DATA extends to 20:55 (AFTER end_time)
```

**The Issue**: Daily bar timestamps are at midnight, but the bar's data spans the entire day. Slicing by timestamp doesn't account for this.

---

## Impact on Model Training

### Contamination Path

```
Position i=50 (18:40)
    ↓
Generate features using longer timeframes
    ↓
Resample to daily (uses parallel optimization)
    ↓
Daily bar includes data up to 20:55 (FUTURE)
    ↓
Features use future price information
    ↓
Model learns patterns that require future knowledge
    ↓
Validation metrics are artificially inflated
```

### Affected Features

Any feature derived from resampled timeframes:
- Daily/weekly/monthly OHLC values
- Multi-timeframe channel boundaries
- Longer timeframe momentum indicators
- Break trigger TF classification
- Volatility measures

---

## Where in the Code

**File**: `/Users/frank/Desktop/CodingProjects/x9/v7/training/labels.py`

**Function**: `_try_get_precomputed_slice()` (lines 209-245)

**Problematic Lines**:
```python
end_timestamp = df.index[-1]
idx = precomputed_df.index.searchsorted(end_timestamp, side='right')
return precomputed_df.iloc[:idx]  # ← BUG: Includes complete bars with future data
```

**Used By**:
- `cached_resample_ohlc()` → Called throughout label generation
- Affects all parallel dataset generation
- Enabled by default when `parallel=True`

---

## The Fix

### Quick Fix (Recommended First)

**Disable the optimization**:

```python
# In _try_get_precomputed_slice(), line 243:
# OLD:
return precomputed_df.iloc[:idx]

# NEW:
return None  # Force fresh resampling - no lookahead
```

**Result**: Slower but correct. No lookahead bias.

### Proper Fix (Future)

Implement smart slicing that:
1. Checks if end_timestamp is on a timeframe boundary
2. If on boundary: Use precomputed (safe)
3. If mid-period: Re-resample just the partial period
4. Combine complete periods + partial period

**Result**: Fast AND correct.

---

## How to Verify

### Test Impact

```bash
# 1. Run current walk-forward validation
python v7/training/prepare_walkforward.py --parallel

# 2. Apply fix (return None in _try_get_precomputed_slice)

# 3. Re-run validation
python v7/training/prepare_walkforward.py --parallel

# 4. Compare metrics
# If metrics decrease → confirms lookahead contamination
# Decrease magnitude → how much contamination existed
```

### Expected Results

- Performance metrics should **decrease** after fix
- The decrease shows how much the model was using future data
- Larger decrease = more significant contamination

---

## Root Cause Analysis

### Design Assumption (Wrong)

The optimization was designed with this assumption:

> "For the last partial period, there may be a minor difference if the pre-computed data includes more bars in that period. This is acceptable for the use case (channel detection and feature extraction) where slight differences in the final partial bar don't materially affect results."

**Why This is Wrong**:
- "Minor difference" = future data leakage in reality
- For daily bars, this is 2+ hours of future price action
- NOT acceptable for backtesting/validation
- The assumption works for some use cases but NOT for strict time-series validation

### The Comment That Should Have Been a Red Flag

Lines 216-224 in `labels.py`:
```python
# IMPORTANT: The pre-computed approach is mathematically equivalent to fresh
# resampling for all COMPLETE time periods. For the last partial period,
# there may be a minor difference if the pre-computed data includes more
# bars in that period. This is acceptable for the use case...
```

**Translation**: "We know this might include future data in partial periods, but we think it's okay."

**Reality**: It's not okay for walk-forward validation.

---

## Key Insights

### 1. Daily Bars are Backward-Looking Labels

Daily bar timestamp = **00:00** (start of day)
Daily bar data = **09:30 to 16:00** (entire trading day)

The timestamp doesn't reflect when the data becomes available.

### 2. Pandas Resample is Correct

Pandas `resample()` works correctly for both:
- Partial periods (sequential): Creates partial bar
- Complete periods (precomputed): Creates complete bar

The bug is in **how we slice the precomputed bars**.

### 3. searchsorted Isn't Enough

`searchsorted()` finds bars by **timestamp**, not by **data content**.

A daily bar with timestamp `00:00` contains data up to `16:00`, but searchsorted only sees `00:00`.

### 4. This Only Matters for Time-Series

If you're training on random samples (no time ordering), this bug doesn't matter.

But for walk-forward validation (strict time boundaries), this is **critical**.

---

## Questions Answered

### Q1: When does a daily bar "close"?
**A**: Daily bars are labeled at midnight, but their 'close' value comes from the last bar of the trading day (~16:00 ET for US markets).

### Q2: If historical data ends at 14:30, what does resample_ohlc() include?
**A**: Sequential creates a partial bar with data only up to 14:30. Precomputed has a complete bar with data up to 16:00.

### Q3: What does the precomputed full resample include?
**A**: Complete daily bars with all data from each full trading day.

### Q4: How does searchsorted with side='right' work?
**A**: Returns position AFTER any matching values. For mid-day timestamps, this includes the daily bar that contains future data.

### Q5: Is there a concrete example of lookahead?
**A**: Yes. See investigation script. Sequential gives 101.0000, parallel gives 101.2700 - a difference of 0.27 from 27 bars (2.25 hours) of future data.

---

## Files Created

1. `/Users/frank/Desktop/CodingProjects/x9/investigate_resample_boundaries.py`
   - Executable script demonstrating the bug
   - Concrete timestamps and values
   - Run with: `python investigate_resample_boundaries.py`

2. `/Users/frank/Desktop/CodingProjects/x9/RESAMPLE_BOUNDARY_ANALYSIS.md`
   - Detailed technical analysis
   - All questions answered with examples
   - Multiple solution approaches

3. `/Users/frank/Desktop/CodingProjects/x9/RESAMPLE_BUG_SUMMARY.md` (this file)
   - Executive summary
   - Quick reference
   - Key insights

---

## Next Steps

1. **Verify Impact**: Run walk-forward with and without the fix
2. **Apply Quick Fix**: Return None in `_try_get_precomputed_slice()`
3. **Re-validate**: Check if metrics change
4. **Document**: Record the difference in performance
5. **Future**: Implement smart slicing if performance is critical

---

## Bottom Line

**The parallel optimization is broken for walk-forward validation.**

It includes future data when channels end mid-day. This contaminates features and inflates validation metrics. The fix is simple: disable the optimization. The cost is performance, but the benefit is correct results.

**Recommendation**: Fix immediately before trusting any validation metrics.
