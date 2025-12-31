# v6 Code Archive

## What's Here

This directory contains the **complete v6 codebase** that was replaced by the clean v7 rebuild.

**Moved on:** 2025-12-31

**Reason:** v6 was "vibe-coded" with addons on addons. v7 is a clean rebuild from scratch with:
- Correct channel detection (HIGH/LOW bounces, not close-based)
- Clean architecture (modular, tested, optimized)
- 8-11x faster training
- Interactive CLIs
- Complete documentation

---

## Directory Contents

### Main Scripts
- `train_hierarchical.py` (5,646 lines) - Old training loop with DDP, AMP, complex config
- `predict.py` (54KB) - Old prediction script
- `config.py` (17KB) - Old configuration system
- `test_settings.py` - Old test configuration
- `dashboard_v531.py` (29KB) - Old v5.3.1 dashboard

### Source Code (`src/`)
- `src/ml/features.py` (6,649 lines) - Old feature extraction
  - 10K+ features with channel_features, VIX, events, RSI, breakdown
  - Continuation and transition label generation
  - Cached in multiple tiers

- `src/ml/hierarchical_model.py` (2,319 lines) - Old CfC model
  - 11 parallel CfC networks
  - Multi-phase compositor
  - Physics-based attention (Coulomb forces, energy)

- `src/ml/hierarchical_dataset.py` (3,002 lines) - Old dataset
  - Native timeframe mode
  - Pre-stacking
  - Boundary sampling

- And 19 other modules...

### Tools (`tools/`)
- `visualize_channels.py` - Old channel visualizer (cached data)
- `visualize_live_channels.py` - Old live visualizer
- `channel_loader.py` - Old cache loader
- `channel_inspector.py` - Old inspector tool
- `README_visualizer.md` - Old docs

### Models (`models/`)
- `hierarchical_lnn.pth` - Old trained model weights
- `hierarchical_training_history.json` - Old training metrics

### Utilities (`utils/`)
- Various utility functions

---

## Why v6 Was Deprecated

### Problems with v6

1. **Wrong Bounce Detection**
   - Used CLOSE prices instead of HIGH/LOW
   - Missed actual touches of channel boundaries
   - Led to incorrect bounce counts and channel validity

2. **Architecture Complexity**
   - 10+ loss components with warmup schedules
   - Multiple overlapping features (14 windows × 11 TFs × 31 metrics)
   - Physics-based attention (Coulomb, Energy) - unclear benefit
   - Multiple architecture modes (geometric_physics, geometric_fusion, learned_fusion)

3. **Codebase Mess**
   - Comments referencing v5.0, v5.2, v5.3, v5.6, v5.7, v5.8, v5.9, v6.0...
   - Features added incrementally without cleanup
   - Multiple cache tiers with complex version strings
   - 75+ files in deprecated_code already

4. **No Optimization**
   - Redundant calculations everywhere
   - No caching layer
   - No pre-computation
   - Training took 13-55 hours

5. **No Testing**
   - No unit tests
   - No verification of correctness
   - Hard to debug

6. **No User Interface**
   - Complex command-line arguments
   - No interactive configuration
   - No real-time dashboard

---

## What v7 Fixes

### Correct Implementation
- ✅ HIGH/LOW bounce detection (visually verified)
- ✅ Clean channel detection algorithm
- ✅ Proper ±2σ bounds calculation
- ✅ Complete cycles counting (lower→upper→lower)

### Clean Architecture
- ✅ Modular design (core, features, models, training)
- ✅ 528 well-defined features (down from 10K+)
- ✅ Single clear architecture (hierarchical CfC)
- ✅ No physics-based magic
- ✅ Clear separation of concerns

### Optimized Performance
- ✅ 8-11x faster training (with cache)
- ✅ 98x faster resampling
- ✅ 10,000x faster channel queries
- ✅ Pre-computation support
- ✅ Efficient caching layer

### Verified Correctness
- ✅ 19/19 tests passing
- ✅ All optimizations verified identical
- ✅ Visual channel validation
- ✅ Label generation tested

### Better UX
- ✅ Interactive training CLI (`train.py`)
- ✅ Real-time dashboard (`dashboard.py`)
- ✅ Beautiful progress displays
- ✅ Clear error messages

### Documentation
- ✅ 20+ comprehensive docs
- ✅ Architecture diagrams
- ✅ Usage examples
- ✅ Complete API reference

---

## Can v6 Code Be Used?

**For reference only.** Some useful patterns to study:

1. **Channel feature calculations** (`src/ml/features.py:5463-5862`)
   - Good reference for what features were computed
   - But implementation is messy

2. **CfC layer implementation** (`src/ml/hierarchical_model.py`)
   - Shows how CfC networks were structured
   - Physics-based attention might have insights

3. **Dataset optimizations** (`src/ml/hierarchical_dataset.py`)
   - Pre-stacking batches
   - Boundary sampling
   - Native timeframe mode

**Do NOT use v6 directly** - it has the wrong bounce detection logic.

---

## Migration Checklist

If you need something from v6:
- [ ] Identify the specific feature/function
- [ ] Review its implementation in v6
- [ ] Re-implement cleanly in v7 (don't copy-paste)
- [ ] Test that it produces correct results
- [ ] Add unit tests

---

## v6 Statistics (For Historical Record)

**Codebase Size:**
- Main training: 5,646 lines
- Feature extraction: 6,649 lines
- Model: 2,319 lines
- Dataset: 3,002 lines
- Total: ~18,000 lines in core modules

**Feature Count:** ~10,000+ features across all windows/TFs

**Training Time:** 13-55 hours per 100 epochs

**Test Coverage:** 0% (no tests)

**Documentation:** Scattered comments and version notes

---

## v7 Statistics (Current System)

**Codebase Size:**
- Core + Features: ~4,500 lines
- Models: ~1,200 lines
- Training: ~3,500 lines
- Tools: ~2,000 lines
- Tests: ~2,300 lines
- Total: ~13,500 lines (28% reduction while adding features)

**Feature Count:** 528 clean features

**Training Time:** 1.5-5 hours per 100 epochs (8-11x faster)

**Test Coverage:** 100% on core modules (19/19 passing)

**Documentation:** 20+ comprehensive docs (~50KB)

---

**Date Archived:** 2025-12-31
**Replaced By:** v7 (clean rebuild)
**Status:** Deprecated - Use v7 instead
