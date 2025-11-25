# V3.16 Feature Extraction Performance Optimization Plan

## Overview

**Goal**: Dramatically reduce feature extraction time from 30-60 minutes to 6-12 minutes (70-80% speedup)

**Timeline**: 4-6 weeks, comprehensive implementation

**Approach**: Aggressive speed optimizations leveraging CUDA GPU acceleration, vectorization, and algorithmic improvements

**Current State**:
- 14,487 features (14,322 channel + 165 non-channel)
- Extraction time: 30-60 minutes (first run, no cache)
- Memory usage: 20-40GB standard, 2-5GB with chunking
- GPU: CUDA available but underutilized

**Target State**:
- Extraction time: 6-12 minutes (70-80% speedup)
- Memory usage: 10-16GB (as secondary benefit)
- Full CUDA acceleration for rolling operations and channel calculations
- Vectorized nested loops

---

## Critical Performance Bottlenecks (Prioritized)

1. **Nested loops in channel mapping** (parallel_channel_extraction.py:229-324)
   - Current: O(n × 21 windows) with scalar operations
   - Impact: 500k bars × 21 windows = 10.5M iterations
   - **Optimization potential: 40-50% speedup**

2. **CPU-only rolling statistics** (features.py:520-533, 2293-2300)
   - Current: 6+ separate pandas rolling operations on CPU
   - Impact: Multiple passes over entire dataset
   - **Optimization potential: 20-30% speedup with CUDA**

3. **DataFrame copies and concatenations** (data_feed.py, features.py)
   - Current: 4-6 DataFrame copies during alignment and feature assembly
   - Impact: Memory churn and copy overhead
   - **Optimization potential: 10-15% speedup**

4. **Sequential ping-pong detection** (linear_regression.py)
   - Current: CPU-only numpy operations
   - Impact: Per-window calculations not GPU-accelerated
   - **Optimization potential: 15-20% speedup with CUDA**

---

## Phase 1: Foundation (Week 1-2) - Target 35% speedup

### 1.1 Vectorize Nested Loops in Channel Extraction ⚡ HIGHEST IMPACT

**File**: `src/ml/parallel_channel_extraction.py` (lines 229-324)

**Problem**: Triple nested structure processes each window sequentially with scalar operations.

**Solution**: Vectorize window loop to process all 21 windows simultaneously.

**Implementation**:

1. **Pre-allocate feature matrix**:
   ```python
   # Instead of dict of arrays, use 2D matrix
   num_features_per_window = 31  # position, slopes, r-squared, etc.
   num_windows = len(window_sizes)

   # Shape: [num_bars, num_windows * num_features_per_window]
   results_matrix = np.zeros((len(original_timestamps), num_windows * num_features_per_window),
                             dtype=config.NUMPY_DTYPE)
   ```

2. **Stack channel data for vectorization**:
   ```python
   # For each bar, process all windows together
   for i in range(len(resampled)):
       # Stack all channel data for this bar
       channels_at_bar = [all_windows_channels[w][i] for w in window_sizes]

       # Vectorized slope calculations (21 divisions → 1 operation)
       all_close_slopes = np.array([ch.close_slope for ch in channels_at_bar if ch])
       all_slope_pcts = (all_close_slopes / current_price) * 100  # Vectorized

       # Batch all feature assignments
       results_matrix[indices, feature_slice] = stacked_features
   ```

3. **Eliminate feature-by-feature assignments**:
   - Current: 31 assignments per window (31 × 21 = 651 per bar)
   - Target: 1 assignment per bar (stacked feature vector)

**Expected Gain**: 30-40% speedup

**Risk**: Medium (requires careful indexing)

**Validation**: Compare outputs with `np.allclose(rtol=1e-7)`

---

### 1.2 Eliminate Redundant DataFrame Copies

**Files**:
- `src/ml/data_feed.py` (lines 155-156, 202-203)
- `src/ml/features.py` (line 891)
- `src/ml/hierarchical_dataset.py` (lines 1649, 1671)

**Problem**: Multiple DataFrame copies during alignment and slicing.

**Solution**: Use views where possible, copy only once at end.

**Implementation**:

1. **In data_feed.py:align_symbols()**:
   ```python
   # BEFORE (2 copies):
   spy_aligned = spy_df.loc[common_timestamps].copy()
   tsla_aligned = tsla_df.loc[common_timestamps].copy()

   # AFTER (views, 1 copy at end):
   spy_view = spy_df.loc[common_timestamps]
   tsla_view = tsla_df.loc[common_timestamps]
   # Validate with views
   # Copy only after column renaming
   spy_aligned = spy_view.rename(columns=lambda x: f'spy_{x}')
   tsla_aligned = tsla_view.rename(columns=lambda x: f'tsla_{x}')
   ```

2. **In features.py:_extract_channel_features_live()**:
   ```python
   # BEFORE: 22 copies (2 symbols × 11 timeframes)
   symbol_df = source_data[[c for c in source_data.columns if c.startswith(f'{symbol}_')]].copy()

   # AFTER: Use view, copy once
   columns = [c for c in source_data.columns if c.startswith(f'{symbol}_')]
   symbol_view = source_data[columns]
   # Work with view, copy only if modified
   ```

3. **Enable pandas copy-on-write**:
   ```python
   # In config.py or at import
   pd.options.mode.copy_on_write = True  # Requires pandas >= 1.5.0
   ```

**Expected Gain**: 5-10% speedup, 10-15% memory reduction

**Risk**: Low

---

### 1.3 Consolidate DataFrame Concatenations

**File**: `src/ml/features.py` (lines 445-470)

**Problem**: 3-stage concatenation creates intermediate DataFrame copies.

**Solution**: Build feature list once, concatenate once.

**Implementation**:
```python
# BEFORE (3 concatenations):
base_features_df = pd.concat([price_df, rsi_df, ...], axis=1)  # Concat 1
# ... later
features_df = pd.concat([base_features_df, breakdown_df], axis=1)  # Concat 2

# AFTER (1 concatenation):
all_features = [price_df, rsi_df, correlation_df, cycle_df,
                volume_df, time_df, breakdown_df]
features_df = pd.concat(all_features, axis=1, copy=False)
```

**Expected Gain**: 3-5% speedup, 5-10% memory reduction

**Risk**: Very Low

---

## Phase 2: CUDA Acceleration (Week 3-4) - Target 55% additional speedup

### 2.1 GPU-Accelerated Rolling Statistics ⚡ MAJOR IMPACT

**File**: New file `src/ml/gpu_rolling.py` + modifications to `src/ml/features.py`

**Problem**: All rolling operations (min, max, std, correlation) run on CPU via pandas.

**Solution**: Implement CUDA-accelerated rolling functions using PyTorch.

**Implementation**:

1. **Create GPU rolling statistics module**:
   ```python
   # src/ml/gpu_rolling.py
   import torch
   import numpy as np

   class CUDARollingStats:
       def __init__(self, device='cuda'):
           self.device = device

       def rolling_stats(self, data: np.ndarray, windows: list[int],
                        stats: list[str] = ['min', 'max', 'std']):
           """
           Compute multiple rolling statistics in parallel on GPU.

           Args:
               data: 1D numpy array
               windows: List of window sizes (e.g., [10, 50, 252])
               stats: Statistics to compute ['min', 'max', 'std', 'mean']

           Returns:
               Dict mapping 'stat_window' -> numpy array
           """
           # Convert to GPU tensor
           x = torch.from_numpy(data).float().to(self.device)
           results = {}

           for window in windows:
               # Use unfold for sliding windows (zero-copy view)
               # Shape: [n_windows, window_size]
               windows_view = x.unfold(0, window, 1)

               # Compute all stats in parallel on GPU
               if 'min' in stats:
                   results[f'min_{window}'] = windows_view.min(dim=1)[0].cpu().numpy()
               if 'max' in stats:
                   results[f'max_{window}'] = windows_view.max(dim=1)[0].cpu().numpy()
               if 'std' in stats:
                   results[f'std_{window}'] = windows_view.std(dim=1).cpu().numpy()
               if 'mean' in stats:
                   results[f'mean_{window}'] = windows_view.mean(dim=1).cpu().numpy()

               # Pad to match original length
               pad_size = window - 1
               for key in list(results.keys()):
                   if key.endswith(f'_{window}'):
                       padded = np.full(len(data), np.nan, dtype=np.float32)
                       padded[pad_size:] = results[key]
                       results[key] = padded

           return results

       def rolling_correlation(self, x: np.ndarray, y: np.ndarray,
                               windows: list[int]) -> dict:
           """GPU-accelerated rolling correlation."""
           x_t = torch.from_numpy(x).float().to(self.device)
           y_t = torch.from_numpy(y).float().to(self.device)
           results = {}

           for window in windows:
               x_win = x_t.unfold(0, window, 1)
               y_win = y_t.unfold(0, window, 1)

               # Compute correlation using covariance formula
               x_mean = x_win.mean(dim=1, keepdim=True)
               y_mean = y_win.mean(dim=1, keepdim=True)

               x_centered = x_win - x_mean
               y_centered = y_win - y_mean

               numerator = (x_centered * y_centered).sum(dim=1)
               denominator = torch.sqrt((x_centered ** 2).sum(dim=1) *
                                       (y_centered ** 2).sum(dim=1))

               corr = numerator / (denominator + 1e-8)

               # Pad and store
               padded = np.full(len(x), np.nan, dtype=np.float32)
               padded[window-1:] = corr.cpu().numpy()
               results[f'corr_{window}'] = padded

           return results
   ```

2. **Integrate into feature extraction**:
   ```python
   # In features.py:_extract_price_features()
   if self.gpu_available and self.device == 'cuda':
       gpu_roller = CUDARollingStats(device='cuda')

       # Batch compute all rolling statistics
       rolling_results = gpu_roller.rolling_stats(
           data=df[close_col].values,
           windows=[10, 50, 252],
           stats=['min', 'max', 'std']
       )

       # Extract results
       df[f'{col}_volatility_10'] = rolling_results['std_10']
       df[f'{col}_volatility_50'] = rolling_results['std_50']
       high_52w = rolling_results['max_252']
       low_52w = rolling_results['min_252']
   else:
       # Fallback to pandas
       df[f'{col}_volatility_10'] = returns.rolling(10).std()
       # ... etc
   ```

3. **GPU correlation batching**:
   ```python
   # In features.py:_extract_correlation_features()
   if self.gpu_available and self.device == 'cuda':
       gpu_roller = CUDARollingStats(device='cuda')

       corr_results = gpu_roller.rolling_correlation(
           x=spy_returns.values,
           y=tsla_returns.values,
           windows=[10, 50, 200]
       )

       return pd.DataFrame({
           'correlation_10': corr_results['corr_10'],
           'correlation_50': corr_results['corr_50'],
           'correlation_200': corr_results['corr_200'],
       }, index=spy_returns.index)
   ```

**Expected Gain**: 40-50% speedup (CUDA 10-20x faster than pandas for rolling ops)

**Risk**: Medium (GPU memory management, numerical precision)

**Validation**:
- Compare with pandas: `np.allclose(gpu_result, pandas_result, rtol=1e-5, atol=1e-6)`
- Test on various dataset sizes
- Monitor GPU memory usage

---

### 2.2 GPU-Accelerated Channel Calculations

**File**: `src/linear_regression.py` (channel calculation methods)

**Problem**: Linear regression calculations run on CPU with numpy.

**Solution**: Port to PyTorch for GPU acceleration.

**Implementation**:

1. **GPU linear regression**:
   ```python
   # In LinearRegressionChannel class
   def calculate_multi_window_rolling_gpu(self, df, window_sizes, device='cuda'):
       """GPU-accelerated multi-window channel calculation."""
       # Convert data to GPU tensors
       close = torch.from_numpy(df['close'].values).float().to(device)
       high = torch.from_numpy(df['high'].values).float().to(device)
       low = torch.from_numpy(df['low'].values).float().to(device)

       n = len(close)
       results = {}

       for window in window_sizes:
           # Create rolling windows on GPU
           close_windows = close.unfold(0, window, 1)  # [n_windows, window]

           # Vectorized linear regression (least squares)
           x = torch.arange(window, dtype=torch.float32, device=device)
           x_mean = x.mean()

           # Broadcast and compute slopes for all windows in parallel
           y_mean = close_windows.mean(dim=1, keepdim=True)

           numerator = ((x - x_mean) * (close_windows - y_mean)).sum(dim=1)
           denominator = ((x - x_mean) ** 2).sum()

           slopes = numerator / denominator  # All slopes computed in parallel
           intercepts = y_mean.squeeze() - slopes * x_mean

           # Compute R-squared
           y_pred = slopes.unsqueeze(1) * x + intercepts.unsqueeze(1)
           ss_res = ((close_windows - y_pred) ** 2).sum(dim=1)
           ss_tot = ((close_windows - y_mean) ** 2).sum(dim=1)
           r_squared = 1 - (ss_res / (ss_tot + 1e-8))

           # Store results (transfer back to CPU)
           results[f'slope_{window}'] = slopes.cpu().numpy()
           results[f'r_squared_{window}'] = r_squared.cpu().numpy()

       return results
   ```

2. **Batch process all window sizes**:
   - Instead of sequential processing, stack all windows
   - Compute all regressions in single GPU kernel launch

**Expected Gain**: 15-25% speedup

**Risk**: Medium-High (complex GPU implementation)

---

## Phase 3: Advanced Optimizations (Week 5-6) - Target 10-15% additional speedup

### 3.1 Numba JIT Compilation for CPU Fallbacks

**Files**:
- `src/linear_regression.py` (ping-pong detection, cycle counting)
- `src/ml/parallel_channel_extraction.py` (position calculations)

**Problem**: Python loops not JIT-compiled.

**Solution**: Use Numba for CPU-intensive functions.

**Implementation**:
```python
import numba

@numba.jit(nopython=True, parallel=True, fastmath=True)
def detect_ping_pongs_fast(prices, upper_bounds, lower_bounds, threshold):
    """JIT-compiled ping-pong detection."""
    n = len(prices)
    transitions = 0
    last_state = 0  # 0=neutral, 1=upper, -1=lower

    for i in numba.prange(n):
        if prices[i] >= upper_bounds[i] * (1 - threshold):
            if last_state != 1:
                transitions += 1
                last_state = 1
        elif prices[i] <= lower_bounds[i] * (1 + threshold):
            if last_state != -1:
                transitions += 1
                last_state = -1

    return transitions

@numba.jit(nopython=True, parallel=True)
def vectorized_position_calculation(prices, slopes, intercepts, window_size):
    """Fast position calculation for all windows."""
    n = len(prices)
    positions = np.zeros(n, dtype=np.float32)

    for i in numba.prange(n):
        # Vectorized position logic
        predicted = slopes[i] * window_size + intercepts[i]
        positions[i] = (prices[i] - predicted) / predicted

    return positions
```

**Expected Gain**: 10-15% speedup for CPU operations

**Risk**: Low-Medium (numba compatibility issues)

---

### 3.2 Optimize Type Conversions

**File**: `src/ml/hierarchical_dataset.py` (lines 140, 175-188)

**Problem**: Multiple `.astype()` conversions creating full array copies.

**Solution**: Check dtype before converting, set correct dtype at creation.

**Implementation**:
```python
# Check before converting
if features_array.dtype != expected_dtype:
    features_array = features_array.astype(expected_dtype)
else:
    # Already correct dtype, skip conversion
    pass

# Set dtype at DataFrame creation
features_df = pd.DataFrame(data, dtype=config.NUMPY_DTYPE)

# Check contiguity before copying
if not array.flags['C_CONTIGUOUS']:
    array = np.ascontiguousarray(array)
```

**Expected Gain**: 3-5% speedup, 5-8% memory reduction

**Risk**: Very Low

---

### 3.3 Parallel Processing Tuning

**File**: `src/ml/parallel_channel_extraction.py`

**Optimization**: Tune worker count and batch sizes for optimal CUDA utilization.

**Implementation**:
- Reduce CPU workers when using GPU (4-6 workers vs 8-12)
- Larger batch sizes for GPU operations (10000 → 50000)
- Better queue management to reduce IPC overhead

**Expected Gain**: 5-8% speedup

---

## Implementation Strategy

### Week-by-Week Breakdown

**Week 1**:
- Day 1-2: Set up benchmarking harness, baseline metrics
- Day 3-4: Implement 1.1 (vectorize nested loops)
- Day 5: Implement 1.2 (eliminate copies) + 1.3 (consolidate concat)
- Validate: Target 30-35% speedup

**Week 2**:
- Day 1-2: Validate Week 1 changes, fix bugs
- Day 3-5: Refine vectorization, optimize indexing
- Validate: Confirm 35% speedup, prepare for GPU phase

**Week 3**:
- Day 1-3: Implement 2.1 (GPU rolling statistics)
- Day 4-5: Test on CUDA, validate numerical accuracy
- Validate: Measure GPU speedup (target 40-50%)

**Week 4**:
- Day 1-3: Implement 2.2 (GPU channel calculations)
- Day 4-5: Integration testing, end-to-end validation
- Validate: Cumulative 60-70% speedup

**Week 5**:
- Day 1-3: Implement 3.1 (Numba JIT) for CPU fallbacks
- Day 4-5: Implement 3.2 (type conversions) + 3.3 (parallel tuning)
- Validate: Target 70-80% cumulative speedup

**Week 6**:
- Day 1-3: Performance tuning, profiling, optimization
- Day 4-5: Final validation, documentation, benchmarking
- Deliverable: Production-ready optimized extraction pipeline

---

## Expected Outcomes

### Performance Targets

| Phase | Timeline | Speedup | Extraction Time | Memory | Risk |
|-------|----------|---------|-----------------|--------|------|
| Baseline | - | 0% | 30-60 min | 20-40 GB | - |
| Phase 1 | Week 1-2 | 35% | 19-39 min | 15-30 GB | Med |
| Phase 2 | Week 3-4 | 65% | 10-21 min | 12-20 GB | Med-High |
| Phase 3 | Week 5-6 | 75% | 7-15 min | 10-16 GB | Med |
| **Final Target** | **Week 6** | **80%** | **6-12 min** | **10-16 GB** | **Med** |

### Success Metrics
- ✅ Feature extraction completes in <12 minutes (80% speedup)
- ✅ Peak memory usage <16 GB
- ✅ All 14,487 features numerically equivalent (rtol=1e-5)
- ✅ GPU utilization >70% during channel/rolling operations
- ✅ Zero performance regression on subsequent cached runs
- ✅ Comprehensive test coverage for all optimizations

---

## Critical Files for Implementation

### Priority 1 (Highest Impact):
1. **`src/ml/parallel_channel_extraction.py`** (lines 229-324)
   - Vectorize nested loops
   - Batch feature assignments

2. **`src/ml/gpu_rolling.py`** (new file)
   - CUDA rolling statistics
   - GPU correlation functions

3. **`src/ml/features.py`**
   - Integrate GPU rolling (lines 520-533, 2293-2300)
   - Consolidate concatenations (lines 445-470)
   - Eliminate copies (line 891)

### Priority 2 (Medium Impact):
4. **`src/ml/data_feed.py`** (lines 147-171)
   - Optimize alignment copies

5. **`src/linear_regression.py`**
   - GPU channel calculations
   - Numba JIT for ping-pong detection

6. **`src/ml/hierarchical_dataset.py`** (lines 140, 175-188)
   - Optimize type conversions

### Priority 3 (Supporting):
7. **`config.py`**
   - Add feature flags for optimizations
   - GPU batch size tuning

8. **`benchmark_extraction.py`** (new file)
   - Comprehensive benchmarking suite

---

## Validation Strategy

### Numerical Validation
```python
# Compare old vs new implementations
old_features = old_extractor.extract_features(df)
new_features = new_extractor.extract_features(df)

# Strict tolerance for CPU operations
assert np.allclose(old_features, new_features, rtol=1e-7, atol=1e-9)

# Relaxed tolerance for GPU (float32)
assert np.allclose(gpu_features, cpu_features, rtol=1e-5, atol=1e-6)

# Check for NaN/Inf
assert not np.any(np.isnan(new_features))
assert not np.any(np.isinf(new_features))
```

### Performance Validation
- Benchmark on small dataset (10k bars): Target <10s
- Benchmark on medium dataset (100k bars): Target <2min
- Benchmark on full dataset (500k+ bars): Target <12min
- Track memory usage throughout

### Regression Testing
- Save baseline feature outputs from current implementation
- Compare all 14,487 features after each optimization
- Alert on any discrepancies > tolerance
- Test on multiple date ranges and market conditions

---

## Risk Mitigation

### Git Workflow
- Create feature branch: `optimize/feature-extraction-speed`
- Sub-branches for each phase: `optimize/phase1-vectorization`, etc.
- Commit after each working optimization
- Tag stable versions: `v3.19-optimized-phase1`, etc.

### Feature Flags
```python
# In config.py
USE_GPU_ROLLING = True  # Toggle GPU acceleration
USE_VECTORIZED_MAPPING = True  # Toggle vectorized loops
USE_NUMBA_JIT = True  # Toggle JIT compilation

# Fallback to original implementation if False
```

### Rollback Plan
- Keep original code paths with feature flags
- Document all changes in CHANGELOG.md
- Maintain backward compatibility until validation complete
- Keep performance metrics for comparison

---

## Next Steps

1. ✅ Review and approve this plan
2. Create feature branch: `optimize/feature-extraction-speed`
3. Set up benchmarking infrastructure (Day 1)
4. Begin Phase 1: Vectorize nested loops (Day 2-4)
5. Weekly progress reviews and adjustments
6. Final validation and deployment (Week 6)

---

**Status**: Plan archived for future reference. Not currently being implemented.

**Note**: This optimization plan is for the existing V3.16 feature extraction system. It does not include the V4 CNN architecture proposal which would largely eliminate feature extraction overhead.
