# Label Generation Bottleneck Analysis

## Summary
**YES - Forward scanning IS a significant bottleneck, but it's more nuanced than initially suspected.**

The label generation system scans forward through market data, but the actual computational cost is much higher than a simple 44,000 bar scans per position due to per-timeframe channel detection overhead.

---

## 1. Forward Scanning Loop in `generate_labels()`

### Code Location
File: `/Users/frank/Desktop/CodingProjects/x6/v7/training/labels.py`
Function: `generate_labels()` (lines 630-759)

### The Forward Scanning Pattern

```python
# Lines 707-709 in labels.py
break_idx, break_direction = find_permanent_break(
    df_forward, upper_proj, lower_proj, return_threshold
)
```

The `find_permanent_break()` function (lines 361-444):
- Uses vectorized numpy operations to check all bars at once
- Creates boolean masks for upward/downward breaks (highly optimized)
- Tracks state with minimal loop overhead

### Max Scan Values Per Timeframe
From `TF_MAX_SCAN` dict (lines 231-243):

```python
'5min': 100,      # ~8 hours
'15min': 100,     # ~25 hours
'30min': 50,      # ~25 hours
'1h': 50,         # ~50 hours (~2 days)
'2h': 50,         # ~100 hours (~4 days)
'3h': 50,         # ~150 hours (~6 days)
'4h': 50,         # ~200 hours (~8 days)
'daily': 50,      # ~50 trading days (~2.5 months)
'weekly': 50,     # ~50 weeks (~1 year)
'monthly': 10,    # ~10 months
'3month': 10,     # ~30 months (~2.5 years)
```

**Total: 100+100+50+50+50+50+50+50+50+10+10 = 620 bars scanned across all TFs**

---

## 2. Multi-Window and Multi-TF Complexity

### Dataset Call Pattern
File: `/Users/frank/Desktop/CodingProjects/x6/v7/training/dataset.py` (line 1233)

```python
labels_per_window = generate_labels_multi_window(
    df=tsla_df.iloc[:i + max_forward_5min_bars],
    channels=channels,  # Dict with 8 window sizes
    channel_end_idx_5min=i - 1,
    ...
)
```

### Window Sizes
From `STANDARD_WINDOWS` in `/Users/frank/Desktop/CodingProjects/x6/v7/core/channel.py` (line 18):

```python
STANDARD_WINDOWS = [10, 20, 30, 40, 50, 60, 70, 80]
```

**8 window sizes detected**

### Timeframes
From `TIMEFRAMES` in `/Users/frank/Desktop/CodingProjects/x6/v7/core/timeframe.py` (lines 10-13):

```python
TIMEFRAMES = [
    '5min', '15min', '30min', '1h', '2h', '3h', '4h',
    'daily', 'weekly', 'monthly', '3month'
]
```

**11 timeframes total**

---

## 3. The Real Computational Cost

### Architecture Flow

For **EACH position** in the dataset:

```
generate_labels_multi_window()
  ├─ FOR each of 8 window sizes:
  │   └─ generate_labels_per_tf()
  │       ├─ FOR each of 11 timeframes:
  │       │   ├─ resample_ohlc() [CACHED]
  │       │   ├─ detect_channels_multi_window() [EXPENSIVE!]
  │       │   │   └─ FOR each of 8 window sizes:
  │       │   │       └─ detect_channel() - linear regression
  │       │   ├─ generate_labels()
  │       │   │   ├─ project_channel_bounds() - numpy array operations
  │       │   │   ├─ find_permanent_break() - vectorized scan [620 bar scans total]
  │       │   │   └─ detect_new_channel() - candidate scanning [up to max_scan]
  │       │   └─ find_break_trigger_tf()
  │       │       ├─ get_longer_tf_channels()
  │       │       └─ check_containment()
```

### Calculation: Total Operations Per Position

#### A) Forward Scanning (find_permanent_break)
- **Total bars scanned**: 100+100+50+50+50+50+50+50+50+10+10 = **620 bars**
- **Per-timeframe**: Varies (50-100)
- **Vectorized**: Uses numpy masked arrays (fast)

#### B) Channel Detection Overhead (THE REAL COST)
This is where the bottleneck actually lies:

For **each of 8 windows × 11 timeframes = 88 combinations**:
- `detect_channels_multi_window()` is called
- This function detects channels at **all 8 window sizes** for each timeframe
- That's **8 × 88 = 704 full channel detection passes per position**

Each channel detection involves:
- Linear regression (slope/intercept calculation)
- Residuals computation
- Standard deviation calculation
- Bounce detection (touch counting and alternation analysis)

#### C) New Channel Detection
For each timeframe (11 total):
- `detect_new_channel()` scans forward for up to max_scan bars
- Pre-extracts numpy arrays and performs vectorized variance checks
- Calls `detect_channel()` on candidate windows
- This is somewhat optimized with early termination

#### D) Trigger TF Detection
For each timeframe:
- `find_break_trigger_tf()` calls `get_longer_tf_channels()`
- This resamples and detects channels at all longer timeframes
- Uses cached resampling (efficient)

### The Math:

**Channel Detection Cost** (PRIMARY BOTTLENECK):
```
Per Position:
  8 windows × 11 timeframes × 8 window sizes per TF = 704 channel detections
  Each detection: ~O(window_size) linear regression
```

**Forward Scanning Cost** (SECONDARY):
```
Per Position:
  620 bars across all timeframes
  Vectorized with numpy (fast)
  But done AFTER expensive channel detection
```

**Total Cost Per Position:**
- Channel detection: ~704 × O(avg_window) = dominant cost
- Forward scanning: 620 vectorized bar checks = minor cost
- New channel detection: variable, optimized with early exit
- Trigger TF detection: 11 × (cached resampling + channel detection)

---

## 4. Current Optimizations

### Resampling Cache
**Status: ACTIVE** (lines 37-158 in labels.py)
- Thread-local cache stores resampled DataFrames
- Avoids redundant resampling within a single label generation call
- Cache key: `(id(df), len(df), timeframe)`

### Vectorized Forward Scanning
**Status: ACTIVE** (lines 385-437 in labels.py)
- Uses numpy masked arrays in `find_permanent_break()`
- Breaks computation computed once for all bars
- State tracking with minimal loop overhead

### Early Termination in New Channel Detection
**Status: ACTIVE** (lines 494-567 in labels.py)
- Pre-computes variance threshold
- Skips positions with insufficient price movement
- Stops as soon as valid channel is found

### Caching Stats Available
**Status: OPTIONAL** (lines 60-124 in labels.py)
- Can enable cache monitoring: `labels.ENABLE_CACHE_STATS = True`
- Provides hit/miss statistics for tuning

---

## 5. Where The Bottleneck Actually Is

### PRIMARY BOTTLENECK: Channel Detection
**Location**: `generate_labels_per_tf()` lines 1003-1019

```python
# For EACH timeframe:
tf_channels = detect_channels_multi_window(
    df_tf_for_channel.iloc[:channel_end_idx_tf + 1],
    windows=STANDARD_WINDOWS,  # All 8 windows!
    min_cycles=min_cycles
)
tf_channel, best_tf_window = select_best_channel(tf_channels)
```

This is called **11 times per position** (once per timeframe).

Each call detects channels at **all 8 window sizes** = **704 total channel detections**.

### SECONDARY BOTTLENECK: New Channel Detection
**Location**: `generate_labels()` lines 736-741

```python
new_channel = detect_new_channel(
    df,
    start_idx=break_absolute_idx + return_threshold,
    window=window,
    max_scan=max_scan - break_idx  # Variable, but can be large
)
```

Called **11 times per position** (once per timeframe).

Scans forward for new channels with optimized variance checks.

---

## 6. Is It 44,000 Bar Scans Per Position?

### The Original Hypothesis
8 windows × 11 TFs × 500 bars = 44,000 bar scans

### Reality
- **Forward bars scanned**: ~620 bars total (not 44,000)
- **Timeframes**: 11 (correct)
- **Window sizes**: 8 (correct)
- **BUT**: The 620 bar forward scans are **vectorized** and fast

The actual bottleneck is **704 channel detection operations** (8 windows × 11 timeframes × 8 windows per detection), each involving O(window) linear regression operations.

### True Complexity Per Position
```
Dominant cost: 704 × O(window) = Channel Detection
Secondary cost: 11 × O(max_scan) = New Channel Detection
Minor cost: 620 vectorized bar checks = Forward Scanning
```

---

## 7. Performance Characteristics

### What's Fast
- ✓ Forward bar scanning (vectorized numpy)
- ✓ Resampling (cached)
- ✓ Numpy operations (highly optimized)

### What's Slow
- ✗ Linear regression × 704 times per position
- ✗ Touch detection and bounce counting × 88 times per position
- ✗ New channel detection scanning (though optimized)

### Scalability Issues
- Dataset building with **10,000+ positions** × **704 channel detections** = **7+ million regressions**
- Single resampled DataFrame cached and reused (good)
- But per-position channel detection redone for each window size (bad)

---

## 8. Recommendations for Optimization

### Quick Wins (2-3x speedup)
1. **Cache channel detection results per timeframe**
   - Store channels detected at all window sizes for a given timeframe
   - Reuse across window iterations in `generate_labels_multi_window()`

2. **Vectorize touch detection**
   - Current implementation iterates through touch positions
   - Could use numpy diff() for alternation detection

3. **Reduce window sizes for longer timeframes**
   - TF-specific window sets (larger TFs don't need size-10 windows)

### Medium Effort (5-10x speedup)
4. **Lazy channel detection**
   - Only detect channels for timeframes that have valid price data
   - Skip 3month if only 1 month of data available

5. **Pre-compute break scenarios**
   - Cache break probability models per timeframe
   - Use heuristics instead of exhaustive search in some cases

### Long Term Architecture
6. **Separate concerns**
   - Channel detection (expensive, parallelizable)
   - Label generation (quick, depends on channels)
   - Run channel detection once, store results, apply labels separately

---

## Conclusion

**Forward scanning of 500 bars is NOT the bottleneck.**

**The actual bottleneck is 704 channel detections per position** (8 windows × 11 timeframes, with 8-window detection per TF), each requiring O(window) linear regression operations.

The forward bar scanning is:
- Only 620 bars total (not 44,000)
- Highly vectorized (fast)
- A minor contributor to overall runtime

To optimize label generation speed, focus on reducing channel detection overhead, not forward scanning overhead.

### Current Optimization Status
- ✓ Forward scanning: Already vectorized (good)
- ✓ Resampling: Already cached (good)
- ✗ Channel detection: Redundantly performed (bad - main target for optimization)
- ~ New channel detection: Optimized with variance checks (acceptable)
