# Feature Lookup Debug Report

## Summary

The feature names being used in `dual_inspector.py` **DO EXIST** in the samples, but there's a critical issue: **channels are only being detected for the 5min timeframe, not for higher timeframes**.

## Key Findings

### 1. Feature Naming Convention is Correct

The code is looking for features with the correct naming pattern:
- TSLA: `{tf}_w{window}_channel_slope`, `{tf}_w{window}_channel_intercept`, `{tf}_w{window}_std_dev_ratio`
- SPY: `{tf}_w{window}_spy_channel_slope`, `{tf}_w{window}_spy_channel_intercept`, `{tf}_w{window}_spy_std_dev_ratio`

### 2. Features Exist for 5min Timeframe

For `5min` timeframe with window 50, all required features exist with valid values:

**TSLA:**
- `5min_w50_channel_slope: -0.080700`
- `5min_w50_channel_intercept: 193.952941`
- `5min_w50_std_dev_ratio: 0.004523`

**SPY:**
- `5min_w50_spy_channel_slope: -0.013745`
- `5min_w50_spy_channel_intercept: 189.881765`
- `5min_w50_spy_std_dev_ratio: 0.001597`

### 3. Features Are All Zeros for Higher Timeframes

For `15min`, `1h`, `2h`, `daily`, etc., the channel features exist but are **all zeros**:

**15min w50 TSLA:**
- `15min_w50_channel_slope: 0.000000`
- `15min_w50_channel_intercept: 0.000000`
- `15min_w50_std_dev_ratio: 0.000000`

**1h w30 TSLA:**
- `1h_w30_channel_slope: 0.000000`
- `1h_w30_channel_intercept: 0.000000`
- `1h_w30_std_dev_ratio: 0.000000`

## Root Cause

The issue is in the **feature extraction pipeline** (`v15/features/tf_extractor.py`):

1. Features are extracted for all 10 timeframes (5min, 15min, 30min, 1h, 2h, 3h, 4h, daily, weekly, monthly)
2. For each timeframe, the data is resampled and channels should be detected
3. However, **channel detection is failing or not happening** for timeframes other than 5min
4. When channels aren't detected, the feature extractor fills in zeros for all channel-related features

## Impact on dual_inspector.py

The `_reconstruct_channel_from_features()` function in `dual_inspector.py`:

1. **Works correctly for 5min timeframe** - all features exist with valid values
2. **Returns None for other timeframes** - because slope/intercept/std_dev are all zero
3. This causes the inspector to fall back to the arrow display instead of showing channel bounds

## Window Usage

The code is using the correct window:
- `display_window = sample.best_window` (defaults to 50 if window_idx is not set)
- For the test sample, `best_window = 20`
- However, features exist for all windows (10, 20, 30, 40, 50, 60, 70, 80)

## Feature Count Breakdown

Total features per sample: **13,660**

Per timeframe (10 timeframes):
- Window-independent features: 435
- Window-dependent features (8 windows × 58 TSLA + 58 SPY): 928
- **Total per TF: 1,363**

Global (no TF prefix):
- Event features: 30

## Recommendations

### 1. Debug Channel Detection
Check why `_detect_channels_for_tf()` in `tf_extractor.py` is not detecting valid channels for higher timeframes:
- Is there insufficient data after resampling?
- Are the detection thresholds too strict?
- Is there an error during channel detection that's being silently caught?

### 2. Add Logging
Add debug logging in `tf_extractor.py` to show:
- How many bars available after resampling each TF
- Whether channel detection succeeds or fails for each TF/window
- What the actual channel parameters are (slope, intercept, std_dev)

### 3. Verify Resampling
Check that resampling is working correctly:
- Are higher timeframes getting enough data?
- Is the resampling logic preserving data correctly?

### 4. Fallback Logic
The current fallback in `dual_inspector.py` is appropriate:
- When features are missing or zero, it falls back to arrow display
- This prevents crashes but hides the channel visualization

## Test Case

Looking at sample 0 from `test_samples.pkl`:
- Timestamp: 2016-01-27 15:40:00
- Best window: 20
- 5min features: Valid with non-zero channel parameters
- 15min, 1h, etc. features: All zeros

This suggests the scanner is only successfully detecting channels on the 5min timeframe and not resampling/detecting on higher timeframes.

## Next Steps

1. Check `v15/scanner.py` to see if it's only extracting features for 5min
2. Look at `extract_all_tf_features()` to verify it's processing all 10 timeframes
3. Add debug output to see channel detection success rates per TF
4. Verify that resampling is producing enough data for channel detection
