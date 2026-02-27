# Feature Lookup Debug Summary

## Executive Summary

**The feature names being used in the code ARE CORRECT and DO EXIST in the samples.**

However, **channels are only being successfully detected for 3 out of 10 timeframes**:
- ✓ 5min (8/8 windows detected)
- ✓ weekly (5/8 windows detected)
- ✓ monthly (1/8 windows detected)
- ✗ 15min, 30min, 1h, 2h, 3h, 4h, daily (0/8 windows detected)

## Detailed Findings

### 1. Feature Naming is Correct ✓

The code in `dual_inspector.py` looks for:
- TSLA: `{tf}_w{window}_channel_slope`, `{tf}_w{window}_channel_intercept`, `{tf}_w{window}_std_dev_ratio`
- SPY: `{tf}_w{window}_spy_channel_slope`, `{tf}_w{window}_spy_channel_intercept`, `{tf}_w{window}_spy_std_dev_ratio`

These features exist in the samples with exactly these names.

### 2. Window Selection is Correct ✓

The code uses `display_window` which is set to either:
- The manually selected window index, OR
- `sample.best_window` (which is 20 for the test sample)

All 8 standard windows (10, 20, 30, 40, 50, 60, 70, 80) have features present.

### 3. Channel Detection Statistics

**Out of 80 total TF/window combinations:**
- TSLA channels detected: 14 (17.5%)
- SPY channels detected: 13 (16.2%)

**Per Timeframe (TSLA):**
```
TF        w10    w20    w30    w40    w50    w60    w70    w80
5min       ✓      ✓      ✓      ✓      ✓      ✓      ✓      ✓   (8/8)
15min      -      -      -      -      -      -      -      -   (0/8)
30min      -      -      -      -      -      -      -      -   (0/8)
1h         -      -      -      -      -      -      -      -   (0/8)
2h         -      -      -      -      -      -      -      -   (0/8)
3h         -      -      -      -      -      -      -      -   (0/8)
4h         -      -      -      -      -      -      -      -   (0/8)
daily      -      -      -      -      -      -      -      -   (0/8)
weekly     ✓      ✓      ✓      ✓      ✓      -      -      -   (5/8)
monthly    ✓      -      -      -      -      -      -      -   (1/8)
```

### 4. Example Values

**5min w50 TSLA (VALID):**
- `channel_slope: -0.08070`
- `channel_intercept: 193.95294`
- `std_dev_ratio: 0.00452`
- `channel_valid: 1.0`

**15min w50 TSLA (ZERO):**
- `channel_slope: 0.0`
- `channel_intercept: 0.0`
- `std_dev_ratio: 0.0`
- `channel_valid: 0.0`

## Root Cause Analysis

The issue is **NOT** in the feature lookup code. The issue is in the **feature extraction pipeline**.

### Where the Problem Occurs

In `v15/features/tf_extractor.py`, the `extract_all_tf_features()` function:

1. Resamples 5min data to each of 10 timeframes
2. Calls `_detect_channels_for_tf()` to detect channels on the resampled data
3. Extracts channel features from detected channels

**The channel detection is failing for 15min, 30min, 1h, 2h, 3h, 4h, and daily timeframes.**

### Likely Causes

1. **Insufficient data after resampling**
   - Higher timeframes may not have enough bars after resampling
   - Channel detection requires minimum bar counts
   - Early in dataset (2016-01-27), there may not be enough history

2. **Channel detection thresholds**
   - The channel detection algorithm may have strict validity requirements
   - R-squared, touch counts, or other metrics may fail threshold checks

3. **Silent failures**
   - Errors during channel detection may be caught and suppressed
   - The code falls back to zero values without logging the reason

## Impact on dual_inspector.py

The `_reconstruct_channel_from_features()` function:

1. Checks if features exist (they do) ✓
2. Checks if values are non-zero
3. Returns `None` when all values are zero

This causes:
- 5min panels: Show channel bounds correctly
- 15min, 30min, 1h, 2h, 3h, 4h, daily panels: Fall back to arrow display
- weekly panels: Show channel bounds for some windows
- monthly panels: Show channel bounds for w10 only

## Verification Test Case

Sample 0 from `test_samples.pkl`:
- Timestamp: 2016-01-27 15:40:00
- Best window: 20
- Total features: 13,660

All feature keys exist, but only 5min (and some weekly/monthly) have valid channel parameters.

## Recommendations

### Immediate Action: Add Debug Logging

In `v15/features/tf_extractor.py`, add logging to `_detect_channels_for_tf()`:

```python
logger.info(f"TF={tf}: After resampling, have {len(df)} bars")
logger.info(f"TF={tf}: Detected {len(channels)} valid channels")
for window, channel in channels.items():
    if channel and channel.valid:
        logger.info(f"TF={tf} w{window}: slope={channel.slope:.6f}, r2={channel.r_squared:.3f}")
```

### Investigation Steps

1. **Check bar counts after resampling**
   - Log the number of bars available for each TF after resampling
   - Verify against minimum requirements for channel detection

2. **Check channel detection parameters**
   - Review thresholds in `v7/core/channel.py`
   - Consider relaxing requirements for higher TFs

3. **Check for errors**
   - Look for try/except blocks that may be hiding failures
   - Add explicit error logging

4. **Test with later data**
   - Try samples from later in the dataset (more history available)
   - See if channel detection improves with more data

### Long-term Fix

Options:
1. **Lower thresholds for higher TFs** - May be appropriate since higher TFs are naturally less noisy
2. **Require more historical data** - Only create samples when sufficient history is available
3. **Use fallback logic** - If higher TF channel fails, use projected 5min channel
4. **Flag samples** - Mark samples where channel detection failed for investigation

## Conclusion

**The feature lookup code is working correctly.**

The problem is that channel detection is failing for most timeframes during feature extraction, resulting in zero values being stored in the features dictionary. The inspector code correctly identifies these as invalid and falls back to arrow display.

To fix this, investigate why `_detect_channels_for_tf()` is only succeeding for 5min (and occasionally weekly/monthly) timeframes.

## Files for Reference

- Feature extraction: `/Users/frank/Desktop/CodingProjects/x14/v15/features/tf_extractor.py`
- Channel detection: `/Users/frank/Desktop/CodingProjects/x14/v7/core/channel.py`
- Inspector code: `/Users/frank/Desktop/CodingProjects/x14/v15/dual_inspector.py`
- Test samples: `/Users/frank/Desktop/CodingProjects/x14/test_samples.pkl`

## Debug Scripts Created

1. `debug_features.py` - Analyzes feature structure and grouping
2. `debug_channel_detection.py` - Shows channel detection success rates per TF/window
3. `FEATURE_DEBUG_REPORT.md` - Detailed analysis document
4. `FEATURE_LOOKUP_DEBUG_SUMMARY.md` - This file
