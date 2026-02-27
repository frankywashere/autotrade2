# Code Cleanup Summary

**Date:** December 4, 2024
**Project:** HierarchicalLNN v4.2

---

## Files Moved to deprecated_code/

The following test scripts were moved to `deprecated_code/` as they were one-off testing utilities no longer actively used:

1. **test_continuation_fix.py** (Nov 17, 2024)
   - One-time test for continuation label bug fix
   - No longer needed after fix was verified

2. **test_continuation_optimization.py** (Nov 19, 2024)
   - Performance testing for continuation label generation
   - Optimization work completed

3. **test_continuation_simple.py** (Nov 19, 2024)
   - Simplified continuation label test
   - Superseded by integrated tests

4. **test_data_loading.py** (Nov 17, 2024)
   - Data loading validation script
   - Functionality now integrated into main training pipeline

5. **test_parallel_channels.py** (Nov 17, 2024)
   - Parallel channel extraction testing
   - Feature now stable and tested

6. **verify_feature_count.py** (Nov 19, 2024)
   - Feature count verification utility
   - One-time validation, no longer needed

---

## Active Code Files (Not Moved)

### Root Directory
- **config.py** - Active configuration file
- **train_hierarchical.py** - Main training script

### src/ml/
All files in `src/ml/` are actively used:
- **base.py** - Model base class
- **channel_features.py** - Channel feature extraction
- **data_feed.py** - Data loading utilities
- **events.py** - Event handling
- **features.py** - Main feature extraction (imports linear_regression, rsi_calculator)
- **gpu_rolling.py** - GPU-accelerated rolling operations
- **hierarchical_dataset.py** - Dataset class
- **hierarchical_model.py** - Main model architecture
- **market_state.py** - Market state features
- **memory_profiler.py** - Memory profiling utilities
- **parallel_channel_extraction.py** - Parallel channel processing
- **physics_attention.py** - **NEW:** Physics-inspired modules

### src/
- **linear_regression.py** - Used by features.py (channel slope calculations)
- **rsi_calculator.py** - Used by features.py (RSI indicator)

### backend/
All backend files remain active for API service.

---

## Documentation Added

### 1. PHYSICS_ARCHITECTURE.md (NEW)
Comprehensive technical specification for the physics-inspired architecture:
- Plain English explanations with analogies
- Quotes from Generalized Wigner Crystal paper
- Technical architecture details
- Mathematical formulations
- Configuration parameters
- Debugging & visualization guides
- **Size:** ~50KB, ~1,200 lines

**Contents:**
- Executive Summary
- Plain English Explanation (Restaurant Kitchen Analogy)
- The Physics Analogy (Wigner Crystals → Markets)
- Core Components (4 physics modules)
- Technical Architecture (full system flow)
- Mathematical Formulations (Coulomb potential, energy functions)
- Configuration & Hyperparameters
- Training & Loss Functions
- Expected Behavior (attention weights, phase classification, energy scores)
- Debugging & Visualization (code examples)
- Comparison to Static Weighting
- Future Enhancements

---

## Project Structure (After Cleanup)

```
exp/
├── config.py
├── train_hierarchical.py
│
├── src/
│   ├── linear_regression.py
│   ├── rsi_calculator.py
│   └── ml/
│       ├── __init__.py
│       ├── base.py
│       ├── channel_features.py
│       ├── data_feed.py
│       ├── events.py
│       ├── features.py
│       ├── gpu_rolling.py
│       ├── hierarchical_dataset.py
│       ├── hierarchical_model.py
│       ├── market_state.py
│       ├── memory_profiler.py
│       ├── parallel_channel_extraction.py
│       └── physics_attention.py ← NEW (v4.1)
│
├── backend/
│   └── app/
│       └── ... (API service files)
│
├── docs/
│   ├── PHYSICS_ARCHITECTURE.md ← NEW (comprehensive)
│   └── CLEANUP_SUMMARY.md ← NEW (this file)
│
├── deprecated_code/
│   ├── test_continuation_fix.py ← MOVED
│   ├── test_continuation_optimization.py ← MOVED
│   ├── test_continuation_simple.py ← MOVED
│   ├── test_data_loading.py ← MOVED
│   ├── test_parallel_channels.py ← MOVED
│   ├── verify_feature_count.py ← MOVED
│   └── ... (other old files)
│
└── tools/
    ├── visualize_channels.py
    └── channel_loader.py
```

---

## Key Changes in v4.2

### Physics Modules Added
1. **CoulombTimeframeAttention** - Dynamic cross-timeframe attention
2. **TimeframeInteractionHierarchy** - V₁, V₂, V₃ interaction strengths
3. **MarketPhaseClassifier** - 5-phase market regime detection
4. **EnergyBasedConfidence** - Stability-based confidence scoring

### Architecture Improvements
- Removed static `fusion_weights` parameter
- Added `use_fusion_head` flag for A/B testing
- Physics-based aggregation as alternative to fusion head
- Energy-adjusted confidence for better calibration

### Bug Fixes
- Fixed cache row count mismatch (chunk creation off-by-one error)
- Fixed YS alignment dropping partial first year
- Added validation warnings for cache inconsistencies

---

## Next Steps

### For Users
1. Read `PHYSICS_ARCHITECTURE.md` to understand the new system
2. Train with `--use-fusion-head` (default) vs `--no-fusion-head` (physics-only)
3. Compare performance metrics
4. Visualize attention weights, phase predictions, energy scores

### For Developers
1. All active code is in root, src/, and backend/
2. Test scripts are in deprecated_code/ for reference
3. New physics modules in `src/ml/physics_attention.py`
4. See PHYSICS_ARCHITECTURE.md for implementation details

---

## Maintenance Notes

### When to Move Files to deprecated_code/
- One-off test scripts that served their purpose
- Old implementations replaced by new versions
- Experimental features that didn't make it to production
- Validation utilities no longer needed

### When to Keep Files Active
- Imported by other active modules
- Part of the training pipeline
- Used by the API backend
- Configuration files
- Utility functions actively called

### Periodic Cleanup (Recommended Every 3 Months)
1. Review test scripts in root directory
2. Check import usage with grep
3. Move unused files to deprecated_code/
4. Update this CLEANUP_SUMMARY.md
5. Document any architectural changes

---

## File Size Summary

| Category | File Count | Description |
|----------|-----------|-------------|
| Active Root | 2 | config.py, train_hierarchical.py |
| Active src/ml/ | 14 | Core model and feature code |
| Active backend/ | ~15 | API service |
| Active tools/ | 2 | Visualization utilities |
| Deprecated | 50+ | Old test scripts, old implementations |
| Documentation | 2 | PHYSICS_ARCHITECTURE.md, this file |

**Total Active Codebase:** ~33 Python files
**Total Lines (active):** ~15,000 lines (estimated)
**Physics Module Addition:** +444 lines (physics_attention.py)

---

## Contact

For questions about:
- **Physics architecture:** See PHYSICS_ARCHITECTURE.md
- **Codebase structure:** See this file
- **Training:** See train_hierarchical.py --help
- **API:** See backend/README.md

**Last Updated:** December 4, 2024
**Version:** HierarchicalLNN v4.2
