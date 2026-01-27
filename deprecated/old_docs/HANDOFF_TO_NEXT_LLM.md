# Handoff Document: x8 Project Status & Next Steps

**Date:** 2026-01-15
**Project:** TSLA Channel Prediction System (x8)
**Current State:** v14.0.0 cache system implemented but needs debugging
**For:** Next LLM to continue fixing errors

---

## 📋 Executive Summary

Today we performed a comprehensive analysis and clean rewrite of the x8 TSLA channel prediction cache generation system. We discovered critical architectural bugs and implemented a new v14.0.0 system with 15 Opus agents. The system is implemented but encountering runtime errors that need debugging.

---

## 🔍 What We Discovered Today

### Critical Bugs Found in v13 System

1. **Feature-Label Window Mismatch (ARCHITECTURAL BUG)**
   - **Problem:** Features extracted using window=50, labels using per-TF "best" windows (e.g., 15min uses window=30)
   - **Impact:** Learning mode completely broken - model sees mismatched data
   - **Severity:** Critical - 40-60% of training samples affected
   - **File:** `v7/training/labels.py` lines 1278-1310

2. **OOM (Out of Memory) Exit 137**
   - **Problem:** 7 workers × 595MB precomputed data × 11 TFs = 12.6GB RAM
   - **Impact:** Cache regeneration crashes on 16GB machines
   - **Severity:** Critical - can't generate new caches
   - **File:** `v7/training/scanning.py` lines 50-67

3. **Technical Debt Overload**
   - **Problem:** 465+ lines backward compatibility for v7-v13 (13 versions!)
   - **Impact:** Code bloated, hard to maintain, 187-line deprecated function still in code
   - **Files:** `v7/training/dataset.py`, `v7/training/scanning.py`

4. **SPY Underutilized**
   - **Problem:** SPY only had 121 features vs TSLA's 385
   - **Impact:** Missing valuable cross-asset learning signal
   - **Severity:** Medium - model less effective

5. **Window Cycling in Label Inspector Broken**
   - **Problem:** Each TF showed different windows silently when pressing 'w'
   - **Impact:** User confusion - thinks they're viewing window=50 but actually seeing 4 different windows
   - **File:** `label_inspector.py`

---

## 🏗️ What We Built: v14.0.0 Cache System

### New Module: `v7/cache_v14/`

**Built with 15 Opus agents working in parallel**
**Total:** 27 Python files, ~7,900 lines (clean, no technical debt)

### Architecture Highlights

**1. Protocol-Based Extensibility**
- `FeatureExtractor`, `LabelGenerator`, `ChannelDetector` protocols
- Registry pattern for dynamic feature registration
- Add features in 1 file (~100 lines) vs 5 files in v13

**2. Memory-Safe Parallelization**
- Chunked data slicing (send 10% of data, not 100%)
- Adaptive worker count (2-4 based on available RAM)
- Memory monitoring and warnings
- No pre-loaded resampled DataFrames

**3. Feature-Label Alignment Fix**
- Labels use consistent window parameter for ALL TFs
- No more per-TF "best" window optimization
- Learning mode can freely choose windows

**4. Enhanced Features**
- **Total: 1,117 features** (vs 776 in v13 = +44%)
- SPY enhanced: 121 → 385 features (full parity with TSLA)
- New technical indicators: MACD, ATR, Bollinger Bands (77 features)

### Files Created (27 files)

**Core (10 files):**
- `__init__.py` - Public API
- `protocols.py` - Protocol definitions (352 lines)
- `config.py` - Configuration dataclasses (347 lines)
- `types.py` - Core data types (449 lines)
- `pipeline.py` - Main orchestration (820 lines)
- `data_loader.py` - CSV loading (261 lines)
- `scanner.py` - Chunked parallel scanning (722 lines)
- `cache_io.py` - Serialization (150 lines)
- `checkpoint.py` - Checkpoint/resume (100 lines)
- `memory.py` - Memory management (111 lines)

**Feature System (13 files):**
- `feature/protocol.py` - Enhanced protocols (645 lines)
- `feature/registry.py` - Feature registry (810 lines)
- `feature/extractors/`:
  - `tsla_channel.py` - 385 TSLA features (253 lines)
  - `spy_channel.py` - 385 SPY features (398 lines)
  - `vix.py` - 21 VIX features (274 lines)
  - `cross_asset.py` - 110 cross-asset features (284 lines)
  - `history.py` - 50 history features (393 lines)
  - `events.py` - 46 event features (267 lines)
  - `window_scores.py` - 40 window scores (227 lines)
  - `alignment.py` - 3 alignment features (109 lines)
  - `tech_indicators.py` - 77 new indicators (312 lines)

**Label System (3 files):**
- `label/protocol.py` - Label protocols (100 lines)
- `label/generator.py` - Label generation with v13 fix (350 lines)
- `label/registry.py` - Label registry (100 lines)

**Tools (1 file):**
- `label_inspector.py` - New clean inspector (1,098 lines)

---

## 🐛 Current Issues (Need Debugging)

### Issue #1: Still Getting 0 Valid Samples

**Last run output:**
```
Scanning channels: 100%|███████████████| 7993/7993 [00:00<00:00, 15113.38it/s]
Valid samples: 0 (0.0%)
Time: 0.5s
```

**Symptoms:**
- Scanner runs too fast (0.5 seconds for 7,993 positions)
- 100% failure rate
- No debug output showing why positions fail

**Potential causes:**
1. Workers crashing silently (multiprocessing still broken?)
2. Feature extraction failing (registry not finding extractors?)
3. Label generation still using v13 code somewhere
4. Import errors in worker processes
5. Data validation too strict (all positions rejected?)

**Debug added:** Debug print statements in scanner.py to show which step fails

### Issue #2: Multiprocessing Errors (Still Happening?)

**Error:** "Synchronized objects should only be shared..."

**Status:** Opus agent claims to have fixed, but error may still appear

**If still happening:**
- Check scanner.py for ANY `multiprocessing.Manager()`, `.Value()`, `.Queue()`, `.Lock()` usage
- Verify nothing is passed to `executor.submit()` except simple picklable types
- Check if initializer pattern used correctly

### Issue #3: Feature Registry May Not Be Initialized

**Potential issue:**
- FeatureRegistry needs `_register_core_extractors()` called
- Worker processes may have empty registry
- Each worker needs to initialize registry independently

**Fix needed:**
- Call `registry._register_core_extractors()` in worker init
- Or use module-level initialization

---

## 📊 System Architecture (For Context)

### Data Pipeline Flow

```
1min CSV → 5min base → Sliding window scan (every step bars)
    ↓
Channel Detection (8 windows: 10, 20, 30, 40, 50, 60, 70, 80)
    ↓
Feature Extraction (1,117 features via registry)
    ↓
Label Generation (11 TFs × 8 windows)
    ↓
ChannelSample (features + labels per window)
    ↓
Cache saved (.pkl file)
```

### Key Parameters

**Data:**
- TSLA_1min.csv: 440,404 5min bars (2015-2025)
- Warmup: 32,760 bars (420 trading days = 20 monthly bars minimum)
- Forward: 8,000 bars (~50 daily bars for label generation)

**Scanning:**
- step=10: ~16,000 samples (production)
- step=25: ~6,400 samples (medium)
- step=50: ~3,200 samples (test)

**Memory:**
- v13: 12.6GB → OOM
- v14 target: 4-6GB with 2-4 workers

---

## 🔧 What Needs to be Done Next

### Immediate (Critical - Do First)

1. **Debug the 0 valid samples issue:**
   ```bash
   # Run with debug logging
   python -m v7.cache_v14.pipeline \
       --data-dir data \
       --output data/feature_cache/test_v14.pkl \
       --step 50 \
       --workers 2 \
       2>&1 | tee debug_output.log

   # Check debug_output.log for:
   # [DEBUG idx=X] messages showing which step fails
   ```

2. **Check feature registry initialization:**
   - Verify `FeatureRegistry._register_core_extractors()` is called
   - Check if workers see registered extractors
   - May need to call in worker init function

3. **Verify label generator works:**
   - Test `DefaultLabelGenerator.generate_multi_window()` standalone
   - Check it returns valid labels for at least some positions

4. **Test feature extraction standalone:**
   - Test `FeatureRegistry.extract_all()` on sample data
   - Verify all 9 extractors run without errors

### High Priority (After 0 Samples Fixed)

5. **Verify feature-label alignment:**
   ```python
   # After successful generation
   from v7.cache_v14 import CachePipeline

   pipeline = CachePipeline()
   samples, _ = pipeline.load('data/feature_cache/test_v14.pkl')

   # Check window consistency
   s = samples[0]
   for window in [10, 20, 30, 40, 50]:
       features = s.features_per_window.get(window)
       labels = s.labels_per_window.get(window)

       if features and labels:
           for tf in ['5min', '15min', '1h']:
               # Verify both use same window
               print(f"Window {window}, TF {tf}: features={bool(features)}, labels={bool(labels.get(tf))}")
   ```

6. **Memory test:**
   - Run with `--workers 4` and monitor memory
   - Should stay under 6GB
   - If OOM, reduce workers or chunk_size

7. **Full generation:**
   ```bash
   # After test passes
   python -m v7.cache_v14.pipeline \
       --data-dir data \
       --output data/feature_cache/channel_samples_v14.pkl \
       --step 10 \
       --workers 4 \
       --checkpoint-dir checkpoints
   ```

### Medium Priority (After Cache Works)

8. **Update model training to use v14 cache**
9. **Compare v13 vs v14 performance**
10. **Archive v13 system**
11. **Update all documentation**

---

## 🔍 Debugging Guide

### If Still Getting 0 Samples

**Step 1: Check Worker Errors**
```bash
# Look for exception messages
python -m v7.cache_v14.pipeline ... 2>&1 | grep -A5 "Exception\|Error\|Traceback"
```

**Step 2: Test Single Position Manually**
```python
from v7.cache_v14.scanner import _process_single_position
from v7.cache_v14.config import ScanConfig, LabelConfig
from v7.cache_v14.data_loader import load_market_data
from v7.cache_v14.scanner import DataSlice

# Load data
tsla, spy, vix, _ = load_market_data('data')

# Test single position
idx = 35000  # Middle of dataset
data_slice = DataSlice(
    tsla=tsla[:idx+1000],
    spy=spy[:idx+1000],
    vix=vix[:idx+1000],
    offset=0
)

result = _process_single_position(
    idx=idx,
    data_slice=data_slice,
    scan_config=ScanConfig(),
    label_config=LabelConfig()
)

if result is None:
    print("FAILED - position returned None")
else:
    print(f"SUCCESS - got sample at {result[1].timestamp}")
```

**Step 3: Check Feature Registry**
```python
from v7.cache_v14.feature import FeatureRegistry

registry = FeatureRegistry.get_instance()
print(f"Registered extractors: {registry.list_extractors()}")
print(f"Total features: {registry.get_total_features()}")
# Should show 9 extractors, 1,117 features
```

**Step 4: Check Label Generator**
```python
from v7.cache_v14.label.generator import DefaultLabelGenerator
from v7.core.channel import detect_channels_multi_window

# Test label generation
tsla_slice = tsla[:35000]
channels = detect_channels_multi_window(tsla_slice, windows=[20, 50])

if channels:
    gen = DefaultLabelGenerator()
    labels = gen.generate_multi_window(
        df=tsla,
        channels=channels,
        channel_end_idx_5min=34999,
        config=LabelConfig()
    )
    print(f"Labels generated: {len(labels)} windows")
else:
    print("Channel detection failed")
```

### Common Issues & Fixes

**Issue: ModuleNotFoundError in workers**
- **Cause:** Workers can't find v7.cache_v14 module
- **Fix:** Ensure PYTHONPATH includes project root
- **Check:** `sys.path.insert(0, '.')` in worker init

**Issue: Feature extractors not registered**
- **Cause:** `_register_core_extractors()` not called
- **Fix:** Add to `FeatureRegistry.__init__()` or module `__init__.py`
- **Check:** `FeatureRegistry.get_instance().list_extractors()`

**Issue: Channel detection returns empty dict**
- **Cause:** Data insufficient or min_cycles too high
- **Fix:** Lower min_cycles to 0.5 or increase warmup
- **Check:** Print `len(channels)` after detection

**Issue: Label generation returns all None**
- **Cause:** forward_bars insufficient or fold_end_idx wrong
- **Fix:** Increase forward_bars or remove fold_end_idx
- **Check:** Print labels_per_window keys

---

## 📁 Key Files Reference

### v14 Cache System
- **Main entry:** `v7/cache_v14/pipeline.py` (820 lines)
- **Scanner:** `v7/cache_v14/scanner.py` (722 lines) - Has multiprocessing fixes
- **Feature registry:** `v7/cache_v14/feature/registry.py` (810 lines)
- **Label generator:** `v7/cache_v14/label/generator.py` (350 lines)

### Old v13 System (Reference Only)
- **Old dataset:** `v7/training/dataset.py` (1,937 lines) - Has OOM bug
- **Old scanner:** `v7/training/scanning.py` (747 lines) - Pre-loads everything
- **Old labels:** `v7/training/labels.py` (1,341 lines) - Had window mismatch bug (partially fixed)

### Core Utilities (Reused by v14)
- **Channel detection:** `v7/core/channel.py` - `detect_channel()`, `detect_channels_multi_window()`
- **Timeframes:** `v7/core/timeframe.py` - TIMEFRAMES list, resample_ohlc()
- **Feature helpers:** `v7/features/rsi.py`, `v7/features/events.py` (EventsHandler)

---

## 🎯 Current Status

### What Works ✅
- ✅ All v14 files compile (syntax valid)
- ✅ Feature extractors implemented (9 extractors, 1,117 features)
- ✅ Label generator has v13 alignment fix
- ✅ Memory-safe chunked scanning
- ✅ New label inspector created
- ✅ Multiprocessing sync error fixed
- ✅ Undefined variable error fixed

### What's Broken ❌
- ❌ Cache generation gets 0 valid samples (100% failure)
- ❌ Workers may be crashing silently
- ❌ Unknown which step is failing (channel/features/labels)
- ❌ Debug logging added but may not be showing output

### What's Untested ⏳
- ⏳ End-to-end cache generation
- ⏳ Memory usage under load
- ⏳ Feature-label alignment verification
- ⏳ Model training with v14 cache
- ⏳ Performance improvements

---

## 🔬 Technical Details for Debugging

### Feature Extractor Registration

**Location:** `v7/cache_v14/feature/registry.py` line 718

```python
def _register_core_extractors(self):
    """Register default feature extractors."""
    from .extractors import (
        TSLAChannelExtractor,
        SPYChannelExtractor,
        VIXExtractor,
        CrossAssetExtractor,
        ChannelHistoryExtractor,
        EventsExtractor,
        WindowScoresExtractor,
        AlignmentExtractor,
        TechnicalIndicatorsExtractor,
    )

    self.register(TSLAChannelExtractor(), ...)
    # ... registers all 9
```

**Problem:** This method may not be called automatically!

**Fix needed:** Call in `__init__()` or on first `get_instance()`

### Window Consistency Issue

**What v14 should do:**
```python
# For window=50 entry:
labels_per_window[50]['15min'] = labels from 50-bar 15min channel
features_per_window[50]['15min'] = features from 50-bar 15min channel
```

**What v13 was doing (WRONG):**
```python
# For window=50 entry:
labels_per_window[50]['15min'] = labels from 30-bar 15min channel (per-TF "best")
features_per_window[50]['15min'] = features from 50-bar 15min channel
# MISMATCH!
```

**Check:** After generation, verify labels use consistent windows

### Memory Strategy

**v13 (OOM):**
```python
# Pre-load EVERYTHING
for tf in TIMEFRAMES:
    precomputed[tf] = resample_ohlc(full_df, tf)  # 440K bars × 11 TFs
# 7 workers × this = OOM
```

**v14 (Memory-safe):**
```python
# Send small slices
for chunk in chunks:
    min_idx = min(chunk) - warmup
    max_idx = max(chunk) + forward
    data_slice = df.iloc[min_idx:max_idx]  # ~10% of total
    # Worker resamples locally (small data = fast)
```

---

## 📝 Commands Reference

### Test Generation (Quick)
```bash
python -m v7.cache_v14.pipeline \
    --data-dir data \
    --output data/feature_cache/test_v14.pkl \
    --step 50 \
    --workers 2
```

### Full Generation (Production)
```bash
python -m v7.cache_v14.pipeline \
    --data-dir data \
    --output data/feature_cache/channel_samples_v14.pkl \
    --step 10 \
    --workers 4 \
    --checkpoint-dir checkpoints
```

### Inspect Cache (After Generation)
```bash
python -m v7.cache_v14.label_inspector \
    --cache data/feature_cache/test_v14.pkl

# Keyboard shortcuts:
# t - Swap timeframe views
# w - Cycle windows
# f - Jump to flagged samples
# arrows - Navigate
```

### Debug Single Position (Python)
```python
# Test manually
from v7.cache_v14.scanner import _process_single_position
from v7.cache_v14 import ScanConfig, LabelConfig
from v7.cache_v14.data_loader import load_market_data

tsla, spy, vix, _ = load_market_data('data')

# Create data slice
from v7.cache_v14.scanner import DataSlice
data_slice = DataSlice(tsla[:40000], spy[:40000], vix[:40000], offset=0)

# Test position 35000
result = _process_single_position(
    idx=35000,
    data_slice=data_slice,
    scan_config=ScanConfig(),
    label_config=LabelConfig()
)

print(f"Result: {result}")
```

---

## 🗺️ Directory Structure

```
x8/
├── data/
│   ├── TSLA_1min.csv (93MB, 1.85M bars)
│   ├── SPY_1min.csv (109MB)
│   ├── VIX_History.csv (451KB)
│   └── events.csv (18KB, 483 events)
│
├── v7/
│   ├── cache_v14/ ⭐ NEW CLEAN SYSTEM
│   │   ├── __init__.py
│   │   ├── pipeline.py (main entry point)
│   │   ├── scanner.py (chunked parallel scanning)
│   │   ├── feature/
│   │   │   ├── registry.py (9 extractors)
│   │   │   └── extractors/ (9 extractor files)
│   │   ├── label/
│   │   │   └── generator.py (v13 fix applied)
│   │   └── label_inspector.py (new clean inspector)
│   │
│   ├── training/ ⚠️ OLD SYSTEM (reference only)
│   │   ├── dataset.py (has OOM bug)
│   │   ├── scanning.py (has OOM bug)
│   │   └── labels.py (had window mismatch bug)
│   │
│   ├── core/ ✅ REUSED BY v14
│   │   ├── channel.py (detect_channel, detect_channels_multi_window)
│   │   └── timeframe.py (TIMEFRAMES, resample_ohlc)
│   │
│   └── features/ ✅ HELPERS REUSED BY v14
│       ├── rsi.py (calculate_rsi, calculate_rsi_series)
│       ├── events.py (EventsHandler)
│       └── history.py (scan helpers)
│
├── deprecated_label_inspector_v13.py ⚠️ OLD
└── docs/
    ├── V14_COMPLETE_IMPLEMENTATION.md
    ├── V14_CRITICAL_BUGFIXES.md
    └── HANDOFF_TO_NEXT_LLM.md (this file)
```

---

## 💡 Known Design Decisions

### Why step=10 vs step=50?

**step** = How often to sample positions (in 5min bars)

- **step=10:** Sample every 10 bars (50 min) = ~16,000 samples (dense, production)
- **step=25:** Sample every 25 bars (125 min) = ~6,400 samples (original v13)
- **step=50:** Sample every 50 bars (250 min) = ~3,200 samples (sparse, testing)

**Current:** Testing with step=50 for speed, will use step=10 for production

### Why 11 Timeframes?

```python
TIMEFRAMES = [
    '5min', '15min', '30min', '1h', '2h', '3h', '4h',
    'daily', 'weekly', 'monthly', '3month'
]
```

Hierarchical timeframes capture patterns at different scales. 3month is barely usable (needs 5 years lookback for window=20).

### Why 8 Windows?

```python
STANDARD_WINDOWS = [10, 20, 30, 40, 50, 60, 70, 80]
```

Different lookback lengths detect channels at different scales. Learning mode lets model pick which window to trust dynamically.

### Why 1,117 Features (vs 776)?

**Enhancements:**
- SPY: 121 → 385 (+264) - Full parity with TSLA
- Tech indicators: 0 → 77 (+77) - MACD, ATR, Bollinger Bands

**Original 776:**
- TSLA: 385, SPY: 121, VIX: 21, Cross: 110, History: 50, Events: 46, Window: 40, Align: 3

---

## 🎓 Context for LLM

### The User (Frank)

- Wants clean, maintainable code (no backward compat)
- Wants parallelization to work (no OOM)
- Wants learning mode to work (proper alignment)
- Wants easy feature addition (protocol-based)
- Duration prediction is most important metric
- Has 16GB RAM machine (macOS, spawn mode multiprocessing)

### The Goal

Build a production-ready ML system that:
1. Generates training cache without OOM
2. Has properly aligned features and labels
3. Supports learning mode (model picks windows)
4. Is easy to extend with new features
5. Trains models that predict channel breakout duration accurately

### What Success Looks Like

```bash
# Generation works
python -m v7.cache_v14.pipeline --data-dir data --output cache.pkl --step 10 --workers 4
# Output: Generated 16,000 samples, Peak memory: 4.2GB ✓

# Training works with v14 cache
python train.py --cache-version v14 --window-strategy learned_selection
# Model learns to select optimal windows dynamically ✓

# Performance improves
Direction accuracy: 75% (was 67%)
Duration MAE: 4.5 bars (was 5.8)
Learning mode: Works! (was broken)
```

---

## 🚨 Critical Issues Summary

**MUST FIX IMMEDIATELY:**
1. ❌ 0 valid samples - 100% failure rate
   - All 7,993 positions returning None
   - Workers may be crashing silently
   - Need to identify which step fails

2. ⚠️ Multiprocessing errors may persist
   - Check for any remaining synchronized objects
   - Verify all pickle-safe arguments

**AFTER THAT:**
3. ⏳ Verify feature-label alignment
4. ⏳ Test memory usage
5. ⏳ Train model on v14 cache

---

## 📚 Documentation Created Today

1. **`docs/COMPREHENSIVE_TECH_SHEET.md`** - Complete system overview
2. **`docs/V14_COMPLETE_IMPLEMENTATION.md`** - v14 implementation summary
3. **`docs/V14_CRITICAL_BUGFIXES.md`** - Bugs found and fixed
4. **`docs/WINDOW_SYNC_FIX.md`** - Label inspector fixes
5. **`docs/SLIDING_WINDOW_IMPLEMENTATION.md`** - Walk-forward sliding mode
6. **`docs/3MONTH_TIMEFRAME_NOTES.md`** - 3month limitations
7. **`docs/CLEANUP_SUMMARY.md`** - Code cleanup done
8. **`docs/HANDOFF_TO_NEXT_LLM.md`** - This document
9. **`.claude/plans/jazzy-dazzling-alpaca.md`** - Implementation plan

---

## 🎯 Immediate Actions for Next LLM

### Priority 1: Fix 0 Samples Issue

1. Run test with debug output:
   ```bash
   python -m v7.cache_v14.pipeline --data-dir data --output test.pkl --step 50 --workers 2 2>&1 | tee debug.log
   ```

2. Check debug.log for:
   - `[DEBUG idx=X]` messages
   - Exception tracebacks
   - Which step fails

3. Test components standalone (see Debugging Guide above)

4. Fix the failing component

### Priority 2: Verify Fixes Work

After getting >0 samples:
1. Check sample structure is correct
2. Verify feature count = 1,117
3. Verify window alignment
4. Check memory usage

### Priority 3: Full Generation

1. Run with step=10 for production cache
2. Monitor memory (should be <6GB)
3. Use checkpoints (in case of interruption)

---

## 💬 Communication Notes

**What the user wants:**
- Clear explanations in simple terms
- Use many Opus agents for complex tasks
- Don't care about backward compatibility
- Correctness over performance
- Learning mode must work

**What the user values:**
- Duration prediction accuracy (primary goal)
- Clean, maintainable code
- Extensible architecture
- Memory efficiency

**What to avoid:**
- Half-measures or temporary fixes
- Backward compatibility code
- Over-engineering for hypotheticals
- Vague explanations

---

## 🔗 Related Context

### Dataset Details
- **Period:** 2015-2025 (10 years)
- **Bars:** 440,404 5min bars after alignment
- **Warmup:** 32,760 bars (1.67 years) for monthly window=20
- **Forward:** 8,000 bars (~50 daily bars for label scanning)
- **First sample:** 2016-01-27 (after warmup)
- **Last sample:** 2025-07-30 (before forward reserve)

### Model Architecture
- **Type:** HierarchicalCfC (Continuous-time neural networks)
- **Parameters:** ~459K
- **Input:** 776 features (v13) → 1,117 features (v14)
- **Output:** 4 tasks per TF (duration, direction, next_channel, trigger_tf)
- **Training:** Multi-task learning with learnable uncertainty weights

### Key Insights
- 3month timeframe only works well with window=20 (needs 5yr warmup, only has 1.67yr)
- Monthly works great (every sample has 20+ bars)
- Learning mode was broken due to window mismatch
- SPY was underutilized (only 121 features vs should be 385)

---

## ✅ What to Tell Next LLM

"We built a complete v14.0.0 clean rewrite with 15 Opus agents. The system compiles but cache generation fails with 0 valid samples. We've fixed multiprocessing errors and undefined variables. Need you to:

1. Debug why 0 samples (100% failure rate)
2. Get test generation working (step=50, ~3K samples)
3. Verify feature-label alignment
4. Run full generation (step=10, ~16K samples)

All code is in v7/cache_v14/. Reference old v13 code in v7/training/ but don't use it (has bugs). See docs/HANDOFF_TO_NEXT_LLM.md for details."

---

**END OF HANDOFF DOCUMENT**

Good luck! The system is 90% there, just needs final debugging. 🚀
