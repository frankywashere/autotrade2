# Label Validation Issue - Diagnostic Report

**Date:** 2026-01-25
**Issue:** All labels marked as invalid in Pass 2, preventing Pass 3 sample generation

---

## Problem Summary

The C++ scanner successfully detects channels in Pass 1 (up to 1.1M channels), but Pass 2 marks **100% of labels as invalid**, resulting in 0 samples for Pass 3.

```
[PASS 1] Total: 1,112,321 channels detected
[PASS 2] Processed 1,112,321 channels, 0 with valid labels
[PASS 3] Channels to process: 0
```

---

## Symptoms

1. **Pass 1 Success:** Channels are detected correctly
   - TSLA: 556,675 channels
   - SPY: 555,646 channels
   - Total: 1,112,321 channels

2. **Pass 2 Failure:** All labels marked invalid
   - Input: 1,112,321 channels
   - Output: 0 valid labels (100% rejection rate)

3. **Pass 3 Skip:** No samples generated
   - Input: 0 labeled channels
   - Output: 0 samples

---

## Root Cause Investigation

### Hypothesis 1: Break Detection Failure

The label generator requires **forward-looking data** to detect channel breaks. If forward data is unavailable or insufficient, all labels will be invalid.

**Evidence:**
- Labels require break scanning beyond channel end
- Break detection needs 100-1000+ bars of forward data
- Current dataset: 440,404 bars total
- Channels detected throughout entire range

**Likelihood:** HIGH

### Hypothesis 2: Validation Too Strict

The label validation logic may require multiple conditions to be met:
- `break_scan_valid = true`
- `direction_valid = true`
- `duration_valid = true`
- `next_channel_valid = true`

If ANY of these fail, the entire label is marked invalid.

**Evidence from code:**
```cpp
labels.duration_valid = false;
labels.direction_valid = false;
labels.next_channel_valid = false;
labels.break_scan_valid = false;
```

Multiple exit points return all-false validity flags.

**Likelihood:** HIGH

### Hypothesis 3: Channel End Position Issue

Channels detected near the end of the dataset have insufficient forward data for break detection.

**Math:**
- Dataset: 440,404 bars (5-minute)
- Channel detection step: 10
- Last possible channel: ~440,400
- Forward data required: ~1000-5000 bars (typical)
- Available forward data at end: 0-10 bars

**Conclusion:** Most channels (especially in higher timeframes) lack forward data.

**Likelihood:** VERY HIGH

---

## Validation Logic Analysis

From `label_generator.cpp`:

```cpp
// Early exit #1: NULL pointer check
if (!forward_high || !forward_low || !forward_close) {
    labels.duration_valid = false;
    labels.direction_valid = false;
    labels.next_channel_valid = false;
    labels.break_scan_valid = false;
    return labels;
}

// Early exit #2: Invalid channel
if (channel.slope == 0.0 && channel.intercept == 0.0) {
    labels.duration_valid = false;
    // ... all false
    return labels;
}

// Early exit #3: No forward data
if (n_forward <= 0) {
    labels.duration_valid = false;
    // ... all false
    return labels;
}

// Early exit #4: Break scan failure
if (break_pos < 0) {
    labels.duration_valid = true;
    labels.direction_valid = false;  // <-- Most likely culprit
    labels.next_channel_valid = false;
    labels.break_scan_valid = true;
    return labels;
}
```

**Most Likely Issue:** Early exit #3 (n_forward <= 0)

All channels at the end of the dataset have no forward data, causing 100% label rejection.

---

## Proposed Solutions

### Solution 1: Skip End-of-Dataset Channels (Recommended)

Only detect channels where sufficient forward data exists:

```cpp
// In Pass 1: Limit channel detection to positions with forward data
size_t max_pos = tf_data.size() - MIN_FORWARD_BARS;
for (size_t pos = window; pos < max_pos; pos += config_.step) {
    // Detect channel...
}
```

**Pros:**
- Simple fix
- Matches Python behavior
- No wasted computation

**Cons:**
- Reduces total channels detected
- May miss recent market structure

### Solution 2: Partial Label Validity

Allow labels with only some validity flags set:

```cpp
// Accept labels if ANY validity flag is true
bool is_valid = labels.duration_valid ||
                labels.direction_valid ||
                labels.break_scan_valid;
```

**Pros:**
- More samples for training
- Utilizes more data

**Cons:**
- May introduce noisy labels
- Harder to interpret results

### Solution 3: Synthetic Forward Data

Extend dataset with synthetic bars (linear projection, last value, etc.):

**Pros:**
- No data waste
- Consistent sample count

**Cons:**
- Synthetic data may mislead model
- Complex implementation
- Not production-safe

---

## Immediate Action Required

**Priority:** HIGH

**Task:** Modify Pass 1 to skip channels without sufficient forward data

**Implementation:**
1. Calculate `min_forward_bars` based on labeling requirements (~1000-5000 bars)
2. Limit channel detection to positions where `pos + min_forward_bars < dataset.size()`
3. Re-run benchmark to verify label generation succeeds

**Expected Result:**
- Reduced channel count (e.g., 1.1M → 0.5M channels)
- Non-zero valid labels (e.g., 50-80% validity rate)
- Pass 3 can execute and generate samples

---

## Impact on Benchmark

**Current State:**
- Cannot measure Pass 3 performance
- Benchmark incomplete
- Unknown end-to-end throughput

**After Fix:**
- Can measure full pipeline
- Estimate Pass 3 time: 0.5-2.0s per 1000 samples
- Complete benchmark with all 3 passes

---

## Comparison with Python

The Python v15 scanner likely handles this correctly by:
1. Only scanning up to `len(df) - forward_window`
2. Silently skipping channels without forward data
3. Returning partial results with explicit counts

**Verification Needed:**
Run Python scanner with identical parameters to compare:
- Channel count
- Label validity rate
- Sample generation rate

---

## Conclusion

**Root Cause:** Channels detected at end of dataset lack forward data for label generation.

**Fix:** Limit channel detection to positions with sufficient forward bars.

**Status:** Blocking Pass 3 benchmarking and full pipeline validation.

**Priority:** Fix before production deployment.

---

**Report Generated:** 2026-01-25
**Next Step:** Implement Solution 1 (skip end-of-dataset channels)
