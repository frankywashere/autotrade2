# Migration Notice: v5.3.3 → v5.9.3

**Date:** December 25, 2025

## Documentation Update

The technical specification has been updated to reflect the current stable version (v5.9.3).

**Old Specification:** `Technical_Specification_v5.3.md` (this directory)
**New Specification:** `../Technical_Specification_v5.9.md` (project root)

## Major Changes Since v5.3.3

1. **v5.4-v5.5:** Enhanced channel features (1027 → 1049 features)
2. **v5.6:** Removed fixed geometric projections, now computed at inference
3. **v5.7:** Dual prediction mode (Direct + Geometric), loss warmup, selection temperature annealing
4. **v5.8:** SPY volatility regime feature
5. **v5.9:** Event-aware architecture with RTH-based anchoring, partial window support
6. **v5.9.2:** Layered cache validation system
7. **v5.9.3:** Multi-GPU DDP with TF32, parallel optimization

## Breaking Changes

### Cache Regeneration Required
- Feature version changed from `v5.3.3_bdv2` to `v5.9.1_projv2_bdv3_pbv4_contv2.1`
- Must delete and regenerate all cache files

### Training Configuration
- **CRITICAL:** `num_workers=0` only (multiprocessing hang in v5.9.3)
- FP32 precision only (FP16 AMP unstable)
- TF32 auto-enabled on CUDA

### Event System
- Events now RTH-aligned (9:30 AM ET)
- Timestamps may shift by hours

## Migration Steps

```bash
# 1. Backup old cache (optional)
mv data/feature_cache data/feature_cache_v533_backup

# 2. Update to new branch
git checkout stable-training

# 3. Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# 4. Regenerate cache
python train_hierarchical.py --interactive
# Select "Regenerate cache"

# 5. Train with new settings
# Menu selections:
#   - num_workers: 0
#   - Precision: FP32
#   - Base: Geometric projections
#   - Aggregation: Physics-Only
```

## Known Issues in v5.9.3

1. **Multiprocessing hang:** `num_workers > 0` causes DataLoader to freeze
   - Root cause: Mmap + spawn method incompatibility
   - Workaround: Use `num_workers=0`
   - Fix attempted in v5.9.4 (unstable, on `alignment-backup` branch)

2. **FP16 instability:** AMP causes NaN in duration loss
   - Use FP32 (TF32 auto-enabled on CUDA)

3. **Missing 3month labels:** Often not enough bars for generation
   - Training accepts 10/11 transition label files
   - v5.9.1 partial window support helps

## Performance Impact

| Metric | v5.3.3 | v5.9.3 |
|--------|--------|--------|
| Throughput (workers=4 vs 0) | 3.2 batch/sec | 0.9 batch/sec |
| GPU utilization | 85% | 45-75% (DDP) |
| Epoch time (single GPU) | 45 min | 180 min |
| Epoch time (2 GPU DDP+TF32) | N/A | 70 min |

**Note:** Worker limitation significantly impacts training speed. Multi-GPU DDP + TF32 partially compensates.

## Backward Compatibility

### Models
- v5.3.3 models **NOT compatible** with v5.9.3 (architecture changes)
- Must retrain from scratch

### Caches
- v5.3.3 caches **NOT compatible** with v5.9.3 (feature version mismatch)
- Must regenerate

### Predictions
- Output format similar but with additional fields:
  - `v52_duration` (probabilistic)
  - `v52_validity` (forward-looking)
  - `v52_compositor` (transition predictions)

## Questions?

See main specification: `../Technical_Specification_v5.9.md`

Or GitHub issues: https://github.com/frankywashere/autotrade2/issues
