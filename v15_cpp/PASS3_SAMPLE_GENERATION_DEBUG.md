# Pass 3 Sample Generation Debugging Summary

**Date**: January 25, 2026
**Status**: ✅ **RESOLVED - Scanner working correctly**

## Issue Report

The C++ scanner was reported to generate fewer samples than requested via `--max-samples` parameter.

## Root Cause Analysis

After thorough debugging, the scanner **is working correctly**. The perceived issue was due to insufficient test data size.

### Key Findings

1. **Sample Limiting Logic is Correct** (scanner.cpp lines 339-344)
   ```cpp
   if (config_.max_samples > 0 && total_channels_to_process > config_.max_samples) {
       channel_work_items.resize(config_.max_samples);
   }
   ```
   - Correctly limits work items to requested sample count
   - Applied after filtering for valid channels with labels

2. **Label Generation Requirements**
   - Channels need forward scan data for label generation (21000 5min bars = ~73 days)
   - Channels without sufficient forward data get `direction_valid = false`
   - Pass 3 filters out channels without valid labels (line 321)

3. **Data Size Requirements**
   For successful sample generation, dataset needs:
   - **Warmup**: 32,760 5min bars (~114 days)
   - **Channel detection window**: Variable based on --step parameter
   - **Forward scan**: 21,000 5min bars (~73 days)
   - **Total minimum**: ~55,000-60,000 5min bars (~200+ days)

## Test Results

### Test Configuration
- **Dataset**: 60,000 synthetic 5min bars
- **Step size**: 20 (channel detection spacing)
- **Workers**: 1 (sequential for debugging)

### Results

| max_samples | Channels Detected | Valid Labels | Samples Generated | Status |
|-------------|-------------------|--------------|-------------------|--------|
| 5           | 26,214            | 26,214       | 5                 | ✅ PASS |
| 100         | 26,214            | 26,214       | 100               | ✅ PASS |
| 500         | 26,214            | 26,214       | 500               | ✅ PASS |
| 1,000       | 26,214            | 26,214       | 1,000             | ✅ PASS |

**Conclusion**: Scanner generates **exactly** the number of samples requested.

## Why Initial Tests Failed

### Small Dataset Test (35,000 bars)
- Total bars: 35,000
- Valid scan end: 14,000 (after reserving forward scan space)
- Result: **0 valid labels** (channels detected beyond valid range)
- Lesson: Dataset too small for label generation

### Fixed Dataset Test (60,000 bars)
- Total bars: 60,000
- Valid scan end: 39,000 (after reserving forward scan space)
- Result: **26,214 valid labels**
- Samples generated: **Exactly as requested**

## Code Quality

### Filtering Pipeline (Pass 3)

The scanner applies multiple filters before sample generation:

```cpp
// Line 319-328: Build work items from valid labeled channels
for (const auto& pair : tsla_slim_map) {
    for (size_t idx = 0; idx < channels.size(); ++idx) {
        const SlimLabeledChannel& ch = channels[idx];
        if (ch.channel_valid && ch.labels.direction_valid) {
            // Only valid channels with valid labels are included
            channel_work_items.emplace_back(tf, window, static_cast<int>(idx));
        }
    }
}

// Line 339-344: Apply max_samples limit
if (config_.max_samples > 0 && total_channels_to_process > config_.max_samples) {
    channel_work_items.resize(config_.max_samples);
}
```

### Additional Filters in process_channel_batch()

Each work item goes through validation (lines 1078-1484):
- ✅ Channel map lookup validation
- ✅ Channel index bounds checking
- ✅ Channel validity check
- ✅ Label validity check
- ✅ Index conversion validation (TF space → 5min space)
- ✅ Warmup requirement check
- ✅ Data slice validation
- ✅ Feature extraction validation

## Monitoring Counters

The debug build includes detailed counters:

```
Pass 1 - Channels detected:
  TSLA: 26,214
  SPY: 26,214

Pass 2 - Labels generated:
  TSLA: 26,214 (26,214 valid)
  SPY: 26,214 (26,214 valid)

Pass 3 - Sample generation:
  Channels processed: 100
  Samples created: 100
  Samples skipped: 0
```

## Recommendations

### For Production Use

1. **Data Requirements**
   - Minimum: 60,000 5min bars (~7 months of trading data)
   - Recommended: 100,000+ bars (~12+ months) for more samples

2. **Parameter Tuning**
   - `--step`: Controls channel detection density
   - `--max-samples`: Limits output (working correctly)
   - `--warmup-bars`: Minimum lookback (default 32,760 is good)

3. **Validation**
   - Check Pass 2 label counts: `"X valid labels"` should be > 0
   - If 0 valid labels: dataset too small or lacks forward scan space

### For Testing

1. **Synthetic Data**
   - Use at least 60,000 bars
   - Include realistic price oscillations for channel detection
   - Test with various `--step` values

2. **Expected Behavior**
   - Samples generated = min(max_samples, valid_labeled_channels)
   - All samples should pass `is_valid()` check
   - Feature count should match expected (1025 features)

## Conclusion

**No bugs found in sample generation logic**. The scanner correctly:
1. ✅ Detects channels (Pass 1)
2. ✅ Generates labels (Pass 2)
3. ✅ Filters for valid labeled channels (Pass 3 prep)
4. ✅ Limits to max_samples (Pass 3 prep)
5. ✅ Generates exactly the requested number of samples (Pass 3)

The only requirement is **sufficient data size** to meet warmup and forward scan requirements.

## Files Modified

- `test_pass3_debug.cpp` - Increased test data size from 35,000 to 60,000 bars
- `src/scanner.cpp` - Minor debug output cleanup (no logic changes)

## Test Validation

Run the debug test to verify:
```bash
./build/test_pass3_debug
```

Expected output:
```
✓ SUCCESS: Got exactly 100 samples as requested!
```
