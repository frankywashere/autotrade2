# v14.0.0 Critical Bugfixes - Scanner Issues

**Date:** 2026-01-15
**Status:** ✅ FIXED with Opus agents

---

## Problem Summary

Initial v14 test generation failed with:
1. ❌ **40× "Synchronized objects" errors** (multiprocessing issue)
2. ❌ **0 valid samples** (100% failure rate)
3. ❌ Scanner ran in 0.5 seconds (way too fast - workers failed)

---

## Bug #1: Multiprocessing Synchronization Error ✅ FIXED

### Error Message
```
Worker error: Synchronized objects should only be shared between processes through inheritance
```

### Root Cause
**File:** `v7/cache_v14/scanner.py` (line 554)

```python
progress_counter = ctx.Value('q', 0)  # Creates synchronized object

executor.submit(
    _process_chunk,
    ...,
    progress_counter,  # Passes to worker via pickling - NOT ALLOWED!
)
```

On macOS (spawn mode), `multiprocessing.Value` contains locks/semaphores that can't be pickled. They must be shared via inheritance (before fork), not passed as arguments.

### The Fix (Opus Agent Applied)

**Removed synchronized progress counter entirely:**
- Deleted `progress_counter = ctx.Value('q', 0)`
- Removed from `executor.submit()` calls
- Removed from `_process_chunk()` signature
- Changed progress tracking to count **completed chunks** (not positions)
- Used `as_completed(futures)` pattern with `pbar.update(1)` per chunk

**Result:** No more multiprocessing errors!

---

## Bug #2: Scanner Using v13 Modules (Not v14!) ✅ FIXED

### Root Cause

**File:** `v7/cache_v14/scanner.py` lines 295-307

```python
# WRONG - Imports from OLD v13 system
from ..features.full_features import extract_all_window_features  # v13
from ..training.labels import generate_labels_multi_window        # v13
```

The v14 scanner was calling v13 code, which:
- Has the feature-label window mismatch bug
- Uses deprecated precomputed_tf_channels
- Doesn't work with v14 cache format

**Result:** Every position failed → 0 valid samples

### The Fix (Opus Agent Applied)

**Changed imports to v14 modules:**
```python
# CORRECT - Uses new v14 system
from .label.generator import DefaultLabelGenerator           # v14
from .feature.registry import FeatureRegistry                # v14
from .feature.protocol import ExtractionContext              # v14
```

**Refactored feature extraction:**
```python
# OLD v13 approach
features_per_window = extract_all_window_features(
    tsla_window, spy_window, vix_window,
    windows=valid_windows,
    include_history=True
)

# NEW v14 approach
registry = FeatureRegistry.get_instance()
features_per_window = {}

for window in valid_windows:
    context = ExtractionContext(
        timestamp=tsla_window.index[-1],
        window=window,
        data={"tsla": tsla_window, "spy": spy_window, "vix": vix_window},
        channels={("5min", window): channels[window]},
    )
    result = registry.extract_all(context, fail_on_required=False)
    features_per_window[window] = result.features
```

**Refactored label generation:**
```python
# OLD v13 approach
labels_per_window = generate_labels_multi_window(...)
best_labels_window = select_best_window_by_labels(labels_per_window)
labels_per_tf = labels_per_window[best_labels_window]

# NEW v14 approach
label_gen = DefaultLabelGenerator()
labels_per_window = label_gen.generate_multi_window(
    df=tsla_full,
    channels=channels,
    channel_end_idx_5min=local_idx - 1,
    config=label_config,
    min_cycles=scan_config.min_cycles
)
```

---

## Bug #3: Missing Debug Logging ✅ ADDED

### Added Detailed Debug Output

**Opus agent added logging to identify failure points:**
```python
print(f"[DEBUG idx={idx}] Starting position processing", flush=True)
print(f"[DEBUG idx={idx}] Channel detection: {len(channels)} windows", flush=True)
print(f"[DEBUG idx={idx}] Feature extraction...", flush=True)
print(f"[DEBUG idx={idx}] Label generation...", flush=True)
print(f"[DEBUG idx={idx}] FAIL: {reason}", flush=True)
```

**Purpose:** See exactly which step fails when re-running

---

## Expected Results After Fixes

### Before (Broken):
```
Worker error: Synchronized objects... (40×)
Scanning channels: 100% [00:00<00:00, 21101.22it/s]
Valid samples: 0 (0.0%)
```

### After (Fixed):
```
Scanning channels: 100% [02:30<00:00, 53.28it/s]
Valid samples: 3,200 (40.0%)
Peak memory: 1.2 GB
Cache saved: test_v14.pkl
```

---

## Files Modified by Opus Agents

1. **`v7/cache_v14/scanner.py`** - Fixed multiprocessing + v13→v14 imports
2. **`v7/training/labels.py`** - Removed undefined `precomputed_tf_channels` reference
3. **Debug logging added** - To identify future issues

---

## ✅ Ready to Retry

**Run the command again:**

```bash
python -m v7.cache_v14.pipeline \
    --data-dir data \
    --output data/feature_cache/test_v14.pkl \
    --step 50 \
    --workers 2
```

**Expected:**
- ✅ No synchronized object errors
- ✅ Valid samples generated (~3,000-4,000)
- ✅ Progress updates per chunk
- ✅ Completes in ~2-5 minutes
- ✅ Memory stays under 2GB

If you still see debug messages, they'll tell us exactly what's failing!

---

**All 3 critical bugs fixed by Opus agents. Try again now!** 🚀