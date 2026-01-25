# C++ Scanner Validation Report

**Date:** 2026-01-25
**Validation Type:** C++ vs Python Baseline Comparison
**Target:** >95% of features match within 1e-6 tolerance

## Executive Summary

**STATUS: CRITICAL FAILURE**

The C++ scanner is producing **0 valid samples** compared to Python's 100 samples. This is a fundamental correctness issue that prevents any meaningful comparison.

## Test Configuration

```bash
# Python Baseline
python3 -m v15.scanner \
  --data-dir data \
  --step 200 \
  --max-samples 100 \
  --workers 1 \
  --output /tmp/python_baseline_100.pkl

# C++ Implementation
./build/v15_scanner \
  --data-dir ../data \
  --step 200 \
  --max-samples 100 \
  --workers 1 \
  --output /tmp/cpp_samples_100.bin
```

## Results

### Sample Count

|Scanner|Channels Detected|Valid Labels|Samples Generated|
|-------|----------------|------------|-----------------|
|Python|~29,200|~29,200|100|
|C++|55,785 (step=200)|0|0|
|C++|1,112,321 (step=10)|0|0|

### Root Cause Analysis

The C++ scanner has a **critical bug in channel position tracking**:

1. **Pass 1 (Channel Detection):** Successfully detects channels (e.g., 556,675 TSLA channels with step=10)

2. **Pass 2 (Label Generation):** Fails to generate ANY valid labels (0 valid out of 1,112,321 channels)

3. **Root Cause:** All detected channels have `end_idx` at or near the last bar of the dataset, leaving no forward data for label scanning:
   ```cpp
   int available_forward = n_bars - end_idx - 1;
   // For ALL channels: available_forward <= 0
   ```

4. **Consequence:** Label generator returns early with all validity flags set to `false`:
   ```cpp
   if (n_forward <= 0) {
       labels.duration_valid = false;
       labels.direction_valid = false;
       labels.next_channel_valid = false;
       labels.break_scan_valid = false;
       return labels;
   }
   ```

5. **Pass 3 Filter:** Samples are only created for channels with `direction_valid=true`, so 0 samples are generated.

### Channel Detection Logic Bug

The channel detector appears to have one or more of these issues:

1. **Incorrect index tracking:** Channel `end_idx` is being set to the wrong value (likely always set to last bar index or window end)

2. **Missing position parameter:** The `pos` parameter in `detect_channel()` may not be properly propagated to the Channel struct's indices

3. **Timestamp vs index confusion:** Possible mismatch between bar indices and timestamps when creating Channel objects

### Evidence

```
From C++ scanner output (step=10):
  [PASS 1] Total: 1,112,321 channels detected
  [PASS 2] Processed 556675 channels, 0 with valid labels
  [PASS 3] Channels to process: 0

From Python scanner output (step=200):
  [PASS 1] 29,212 TSLA channels detected
  [PASS 2] 29,212 labels generated (29,198 valid)
  [PASS 3] 100 samples generated
```

## Detailed Findings

### Channel Detection Behavior

**Python (Working):**
- Detects channels at regular intervals based on `--step`
- Each channel has valid forward data for label generation
- Channels distributed throughout the dataset timeline

**C++ (Broken):**
- Detects many channels (even more than Python when using same step size)
- **ALL channels end at or near dataset boundary**
- Zero forward data available for any channel
- This suggests channels are "sliding" to the end of available data

### Label Generation Impact

Due to the channel position bug:
- 0% of channels have valid labels (target: ~100%)
- 0 samples generated (target: 100+)
- Cannot proceed with feature validation
- Cannot measure prediction accuracy

## Comparison with Python (NOT POSSIBLE)

Cannot perform feature/label comparison because C++ produces 0 samples.

**Comparison Prerequisites:**
- [x] Python baseline generated: 100 samples (/tmp/python_baseline_100.pkl)
- [x] Python loader implemented: load_samples.py
- [x] Comparison script created: compare_scanners.py
- [ ] C++ samples generated: **FAILED - 0 samples**

## Recommended Fixes

### Priority 1: Fix Channel Position Tracking

The Channel struct needs to properly store the detection position:

```cpp
// Current (suspected bug):
// Channel end_idx is likely being set to window end or last bar

// Required fix:
// Channel end_idx must equal the 'pos' parameter passed to detect_channel()
// Verify in channel_detector.cpp that Channel object is created with:
//   channel.end_idx = pos
//   channel.end_timestamp_ms = timestamps[pos]
```

### Priority 2: Add Validation

Add assertions in Pass 1 to verify:
```cpp
assert(channel.end_idx >= window_size);
assert(channel.end_idx < n_bars - min_forward_bars);
assert(channel.start_idx < channel.end_idx);
```

### Priority 3: Add Debug Output

Log first 10 channels in Pass 1:
```cpp
std::cout << "Channel #" << i << ": "
          << "start_idx=" << ch.start_idx << " "
          << "end_idx=" << ch.end_idx << " "
          << "n_bars=" << n_bars << " "
          << "available_forward=" << (n_bars - ch.end_idx - 1) << "\n";
```

## Next Steps

1. **URGENT:** Debug and fix channel end_idx calculation in `detect_channel()`
2. Verify fix by checking that detected channels have `end_idx` < `n_bars - 100`
3. Re-run validation with fixed code
4. Proceed with feature/label comparison once samples are generated

## Files

- Python baseline: `/tmp/python_baseline_100.pkl` (61MB, 100 samples)
- C++ output: `/tmp/cpp_samples_100.bin` (NOT GENERATED - 0 samples)
- Loader: `/Users/frank/Desktop/CodingProjects/x14/v15_cpp/load_samples.py`
- Comparator: `/Users/frank/Desktop/CodingProjects/x14/v15_cpp/compare_scanners.py`

## Conclusion

**The C++ scanner implementation has a critical bug that prevents it from generating any valid samples.** The channel detection logic is placing all channels at the end of the dataset, leaving no forward data for label generation. This must be fixed before any validation can proceed.

**Status: BLOCKED on channel position tracking bug fix**
