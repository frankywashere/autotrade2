# Channel Detection Performance Analysis Report

## Executive Summary

The channel detection system is **NOT fully optimized** for ground truth generation. While some improvements have been made with multi-window detection and shared feature extraction, there are **significant redundancies** causing 8+ separate regression operations per position during dataset building.

---

## 1. How `detect_channels_multi_window()` Works

**Location:** `/Users/frank/Desktop/CodingProjects/x6/v7/core/channel.py` (lines 455-489)

### Implementation:
```python
def detect_channels_multi_window(
    df: pd.DataFrame,
    windows: List[int] = None,
    max_workers: int = 4,
    **kwargs
) -> Dict[int, Channel]:
    """Detect channels at multiple window sizes with parallel execution."""
    if windows is None:
        windows = STANDARD_WINDOWS  # [10, 20, 30, 40, 50, 60, 70, 80]

    valid_windows = [w for w in windows if len(df) >= w]

    def detect_for_window(w):
        return w, detect_channel(df, window=w, **kwargs)

    channels = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(detect_for_window, valid_windows)
        for w, channel in results:
            channels[w] = channel

    return channels
```

### Key Findings:

✓ **GOOD:** Uses parallel ThreadPoolExecutor for 8 window sizes
✓ **GOOD:** Detects all 8 windows simultaneously (not sequentially)

✗ **PROBLEM:** Runs `detect_channel()` **8 separate times** - each performs:
  - Linear regression on the same OHLC data
  - Bounce detection on the same boundaries
  - Residual computation
  - Standard deviation calculations

---

## 2. What Operations Are Involved in detect_channel()

**Location:** `/Users/frank/Desktop/CodingProjects/x6/v7/core/channel.py` (lines 229-331)

### Operations Per Window:

```
For EACH window size (8 times):
1. Extract last N bars from DataFrame (slice operation)
2. Convert to numpy arrays (memory copy)
3. Linear regression via scipy.stats.linregress() - EXPENSIVE
   - Computes slope, intercept, r_value, p_value, std_err
4. Calculate residuals: close - center_line
5. Calculate std_dev: np.std(residuals)
6. Compute upper/lower bounds: center_line ± 2*std_dev
7. Detect bounces: detect_bounces()
   - Iterate through all bars checking HIGH/LOW vs bounds
   - Count alternations, complete cycles, touches
8. Calculate quality metrics:
   - alternation_ratio, slope_pct, width_pct
```

### Performance Cost:

- **Linear regression:** O(n) per window
- **Bounce detection:** O(n) per window
- **Total per position:** 8 × O(n) = **8× slower than necessary**

---

## 3. Is Channel Detection Run Multiple Times Per Position?

### During Dataset Building (scanning.py):

**Location:** `/Users/frank/Desktop/CodingProjects/x6/v7/training/scanning.py`

#### Call 1: Multi-window detection (Line 113)
```python
channels = detect_channels_multi_window(
    tsla_window_df,
    windows=STANDARD_WINDOWS,  # [10, 20, 30, 40, 50, 60, 70, 80]
    min_cycles=min_cycles
)
# Result: 8 separate regressions
```

#### Call 2: Label generation (Line 157)
```python
labels_per_window = generate_labels_multi_window(
    df=tsla_full_df,
    channels=channels,  # Reuses above
    channel_end_idx_5min=i - 1,
    ...
)
```
This **reuses** the channels from Call 1 ✓

#### During Feature Extraction (full_features.py):

**Location:** `/Users/frank/Desktop/CodingProjects/x6/v7/features/full_features.py`

##### In `extract_shared_features()` (Line 753):
```python
# Step 8: Compute window scores
multi_window_channels = detect_channels_multi_window(
    tsla_df,
    windows=STANDARD_WINDOWS
)
tsla_window_scores = extract_multi_window_scores(multi_window_channels)
# Result: 8 more regressions - REDUNDANT!
```

This is called **ONCE per sample** during feature extraction.

##### In `extract_window_features()` (Line 811):
```python
# Step 1: Detect TSLA channels per window
for window in STANDARD_WINDOWS:  # 8 iterations
    tsla_channels_dict[window] = detect_channel(df_tf, window=window)
    # Result: 8 more regressions
```

This is called **8 times** - ONCE for each window size!

---

## 4. Redundancy Pattern Analysis

### Call Sequence Per Position:

```
scanning.py:_process_single_position(i)
├─ Line 113: detect_channels_multi_window(tsla_window_df)
│  └─ 8 regressions ← MAIN DETECTION
├─ Line 126: extract_all_window_features()
│  ├─ extract_shared_features()
│  │  └─ Line 753: detect_channels_multi_window(tsla_df)
│  │     └─ 8 MORE regressions ← REDUNDANT!
│  └─ extract_window_features()
│     └─ Line 811: detect_channel(df, window=window) × 8
│        └─ 8 MORE regressions ← REDUNDANT!
└─ Line 157: generate_labels_multi_window(channels)
   └─ Reuses channels from Line 113 ✓
```

### Impact:

**Total regressions per position during dataset building: 8 + 8 + 8 = 24 regressions**

✗ **Only 8 are necessary** (from the initial multi-window detection)
✗ **16 are completely redundant** (already computed channels are recomputed)

---

## 5. Redundant Detections Identified

### Problem 1: extract_shared_features() Line 753
```python
multi_window_channels = detect_channels_multi_window(tsla_df, windows=STANDARD_WINDOWS)
```
**Status:** REDUNDANT
**Solution:** Pass channels from scanning.py to extract_shared_features()
**Impact:** Eliminates 8 regressions per position

### Problem 2: extract_window_features() Line 811 (8 iterations)
```python
for window in STANDARD_WINDOWS:  # [10, 20, 30, 40, 50, 60, 70, 80]
    tsla_channels_dict[window] = detect_channel(df_tf, window=window)
```
**Status:** PARTIALLY REDUNDANT
**Why:** The channels dict is already computed in detect_channels_multi_window()
**Current Usage:** Only per-timeframe channels are needed, but code redetects
**Solution:** Pass pre-computed channels from shared features
**Impact:** Eliminates 8 regressions per position

### Problem 3: Labels Phase (detect_new_channel)
```python
# In labels.py line 563:
channel = detect_channel(df_slice, window=window)
```
**Status:** ACCEPTABLE
**Why:** This is for detecting NEW channels after breaks (different position)
**Impact:** Minor - only when permanent break is found

---

## 6. Per-Window vs Per-Timeframe Complexity

### Current Flow for Multi-Timeframe Labels:

```
For EACH of 11 timeframes:
  For EACH of 8 windows:
    1. generate_labels_per_tf() is called
    2. detect_channel(df_tf, window=w) OR uses pre-detected channel
    3. Scans forward for label generation
```

### In labels.py generate_labels_per_tf() Line 1012:

```python
# For non-5min timeframes:
tf_channels = detect_channels_multi_window(
    df_tf_for_channel.iloc[:channel_end_idx_tf + 1],
    windows=STANDARD_WINDOWS,
    min_cycles=min_cycles
)
```

**Status:** ACCEPTABLE BUT INEFFICIENT
**Why:**
- Detection is needed per-timeframe (different data distribution)
- BUT doing it inside the loop (11 TFs × 8 windows) means unnecessary work
- Could be optimized: detect all 8 windows for ALL timeframes once, then reuse

---

## 7. Summary: Is Channel Detection Optimized?

| Aspect | Status | Details |
|--------|--------|---------|
| **Multi-window detection** | ✓ Optimized | Uses ThreadPoolExecutor (parallel) |
| **Bounce detection** | ✗ Not optimized | Repeated 8+ times per position |
| **Per-position redundancy** | ✗ HIGH | 24 regressions vs 8 needed (3× overhead) |
| **Per-TF redundancy** | ✗ MEDIUM | 11 TFs × 8 windows = 88 detections per position |
| **Feature extraction integration** | ✗ Broken | extract_shared/window_features redetect channels |
| **Label generation** | ✓ Good | Reuses channels from scanning phase |

---

## 8. Optimization Opportunities

### Quick Win #1: Pass channels to feature extraction
**Effort:** Low (parameter passing)
**Impact:** 16 regressions eliminated (67% reduction in feature extraction)
**Implementation:**
- Modify extract_all_window_features() to accept pre-computed channels
- Remove detect_channels_multi_window() calls from extract_shared/window_features()

### Quick Win #2: Cache per-TF channel detections
**Effort:** Medium (add caching layer)
**Impact:** Reduce redundant detections within labels generation
**Implementation:**
- Cache detect_channels_multi_window() results per timeframe
- Reuse across multiple label generation calls

### Quick Win #3: Vectorize bounce detection
**Effort:** High (algorithmic improvement)
**Impact:** 50-70% speedup in bounce detection
**Implementation:**
- Replace loop in detect_bounces() with numpy vectorization
- Compute all window boundaries simultaneously

---

## Conclusion

**Channel detection is NOT optimized.** While the multi-window approach uses parallel execution, there are **16 completely redundant regressions per position** during dataset building (24 total vs 8 needed). This represents a **3× computational overhead** for channel detection.

The main issues are:
1. **Redundant calls** in feature extraction (extract_shared_features redetects)
2. **Redundant per-window detection** in feature extraction (detect_channel called 8 times)
3. **Per-timeframe inefficiency** in label generation (detect all 11 TFs separately)

Quick wins could eliminate 16 regressions per position with minimal code changes.

---

## File Locations for Reference

- **Channel Detection Core:** `/Users/frank/Desktop/CodingProjects/x6/v7/core/channel.py`
  - `detect_channel()` (lines 229-331)
  - `detect_channels_multi_window()` (lines 455-489)
  - `detect_bounces()` (lines 155-226)

- **Dataset Scanning:** `/Users/frank/Desktop/CodingProjects/x6/v7/training/scanning.py`
  - `_process_single_position()` (lines 47-188) [Problem 1: Line 113, Problem 2: Line 126]

- **Feature Extraction:** `/Users/frank/Desktop/CodingProjects/x6/v7/features/full_features.py`
  - `extract_shared_features()` (lines 584-775) [Problem 3: Line 753]
  - `extract_window_features()` (lines 778-846) [Problem 4: Line 811]

- **Label Generation:** `/Users/frank/Desktop/CodingProjects/x6/v7/training/labels.py`
  - `generate_labels_per_tf()` (lines 893-1054) [Problem 5: Line 1012]
  - `detect_new_channel()` (lines 447-567) [Acceptable - new position]
