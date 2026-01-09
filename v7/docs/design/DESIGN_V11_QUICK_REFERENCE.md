# v11.0.0 Quick Reference Card

**Status:** Design Complete - Ready for Implementation
**Timeline:** 4 days implementation + 1 week testing + 2 weeks rollout
**Risk:** Low-Medium

---

## 🎯 Goal

Store features from **all 8 windows** (10, 20, 30, 40, 50, 60, 70, 80) per sample, enabling the model to learn optimal window selection.

---

## 📊 Key Metrics

| Metric | v10.0.0 | v11.0.0 | Mitigation |
|--------|---------|---------|------------|
| Storage (100k samples) | 510 MB | 3.3 GB | Compress → 2.1 GB |
| Extraction (per sample) | 50 ms | 370 ms | 8 cores → ~50 ms |
| Memory (batch=128) | 415 KB | 3.2 MB | Reduce to 64 |
| Implementation | - | 32 hours | 4 days |

---

## 🏗️ Architecture

```
ChannelSample (v11.0.0)
├── per_window_features: Dict[int, PerWindowFeatures]
│   ├── 10: PerWindowFeatures (616 features)
│   ├── 20: PerWindowFeatures (616 features)
│   ├── ... (30-70)
│   └── 80: PerWindowFeatures (616 features)
│
└── shared_features: SharedFeatures (145 features)
    ├── VIX (6)
    ├── History (50)
    ├── Events (46)
    └── Alignment (3)
```

**Total:** 616 × 8 + 145 = **5,073 features per sample**

---

## 🔑 Core Data Structures

### PerWindowFeatures (NEW)
```python
@dataclass
class PerWindowFeatures:
    window: int  # 10, 20, 30, 40, 50, 60, 70, or 80
    tsla: Dict[str, TSLAChannelFeatures]  # 385 features
    spy: Dict[str, SPYFeatures]  # 121 features
    cross_containment: Dict[str, CrossAssetContainment]  # 110 features
    # Total: 616 features
```

### SharedFeatures (NEW)
```python
@dataclass
class SharedFeatures:
    vix: VIXFeatures  # 6
    tsla_history: ChannelHistoryFeatures  # 25
    spy_history: ChannelHistoryFeatures  # 25
    events: Optional[EventFeatures]  # 46
    tsla_window_scores: np.ndarray  # 40
    # + 3 alignment booleans
    # Total: 145 features
```

### ChannelSample (EXTENDED)
```python
@dataclass
class ChannelSample:
    # v10 fields (backward compatible)
    features: FullFeatures
    channel: Channel
    labels: Dict[str, ChannelLabels]
    channels: Dict[int, Channel]
    best_window: int
    labels_per_window: Dict[int, Dict[str, ChannelLabels]]

    # v11 NEW
    per_window_features: Dict[int, PerWindowFeatures]
    shared_features: SharedFeatures
```

---

## 🛠️ Key APIs

### Feature Extraction
```python
# NEW in v11
per_win, shared = extract_full_features_multi_window(
    tsla_df, spy_df, vix_df,
    windows=[10, 20, 30, 40, 50, 60, 70, 80]
)
```

### Cache Loading
```python
# Auto-migrates v10 → v11
samples = load_cache_with_auto_migration(
    cache_path,
    auto_migrate=True
)
```

### Sample Access
```python
# Get features for specific window
features_20 = sample.get_features_for_window(20)

# Get all windows
all_features = sample.get_all_window_features()

# Check multi-window support
has_all = sample.has_multi_window_features()
```

### Dataset Loading
```python
dataset = ChannelDataset(samples)
features, labels = dataset[0]
# features.shape = [8, 761]  # 8 windows × 761 features
```

---

## ✅ Implementation Checklist

### Phase 1: Data Structures (2h)
- [ ] `PerWindowFeatures` dataclass
- [ ] `SharedFeatures` dataclass
- [ ] `FullFeatures.from_split_features()`
- [ ] `ChannelSample` extensions

### Phase 2: Feature Extraction (8h)
- [ ] `extract_shared_features()`
- [ ] `extract_per_window_features()`
- [ ] `extract_full_features_multi_window()`
- [ ] Update parallel scanner

### Phase 3: Dataset Loading (6h)
- [ ] `_get_multi_window_features()`
- [ ] `concatenate_features_dict()`
- [ ] Tensor shape validation

### Phase 4: Cache Management (4h)
- [ ] Update `CACHE_VERSION = "v11.0.0"`
- [ ] `migrate_cache_v10_to_v11()`
- [ ] `load_cache_with_auto_migration()`

### Phase 5: Testing (8h)
- [ ] Unit: Feature extraction
- [ ] Unit: Tensor construction
- [ ] Integration: End-to-end
- [ ] Backward compat: v10 → v11
- [ ] Performance: Benchmark

### Phase 6: Documentation (4h)
- [ ] API docs
- [ ] Migration guide
- [ ] Usage examples

**Total: 32 hours**

---

## 📈 Performance Optimization

### Extraction
```python
# Use parallel scanning (8 cores)
samples = scan_valid_channels(
    tsla_df, spy_df, vix_df,
    use_parallel=True,
    num_workers=8
)
# Result: ~1.3 hours for 100k samples (vs 10.3 sequential)
```

### Storage
```python
# Use compression
with gzip.open('cache.pkl.gz', 'wb') as f:
    pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
# Result: 3.3 GB → 2.1 GB
```

### Memory
```python
# Reduce batch size
dataloader = DataLoader(
    dataset,
    batch_size=64,  # Was 128 in v10
    shuffle=True
)
# Or use gradient accumulation
```

---

## 🔄 Migration Paths

### Path 1: Auto-Migrate (Quick)
```python
samples = load_cache_with_auto_migration(
    'data/channels_v10.pkl',
    auto_migrate=True
)
# Result: Limited multi-window (only best_window)
# Time: ~1 minute
```

### Path 2: Full Rebuild (Recommended)
```python
samples = scan_valid_channels(
    tsla_df, spy_df, vix_df,
    use_parallel=True,
    num_workers=8
)
# Result: Full multi-window (all 8 windows)
# Time: ~1.3 hours for 100k samples
```

---

## ⚠️ Common Pitfalls

### ❌ DON'T
```python
# Don't assume all windows present
features = sample.per_window_features[30]  # May KeyError

# Don't forget to check multi-window support
if sample.per_window_features:  # Wrong - always truthy dict
```

### ✅ DO
```python
# Check if window available
if 30 in sample.per_window_features:
    features = sample.per_window_features[30]

# Or use helper method
features = sample.get_features_for_window(30)  # Returns None if missing

# Check multi-window support
if sample.has_multi_window_features():
    # Use multi-window features
```

---

## 🧪 Testing Examples

### Unit Test: Feature Extraction
```python
def test_split_features():
    per_win, shared = extract_full_features_multi_window(...)

    # Check all windows extracted
    assert len(per_win) == 8
    assert all(w in per_win for w in [10, 20, 30, 40, 50, 60, 70, 80])

    # Check shared features
    assert shared.vix is not None
    assert shared.tsla_history is not None

    # Check per-window correctness
    assert per_win[20].window == 20
    assert len(per_win[20].tsla) <= 11  # Up to 11 TFs
```

### Integration Test: End-to-End
```python
def test_e2e_v11():
    # Scan
    samples = scan_valid_channels(...)

    # Save
    with open('test_v11.pkl', 'wb') as f:
        pickle.dump(samples, f)

    # Load
    loaded = load_cache_with_auto_migration('test_v11.pkl')

    # Verify
    assert len(loaded) == len(samples)
    assert loaded[0].has_multi_window_features()

    # Use in dataset
    dataset = ChannelDataset(loaded)
    features, labels = dataset[0]
    assert features.shape == (8, 761)
```

---

## 📚 Documentation Links

| Document | Purpose |
|----------|---------|
| [DESIGN_V11_SUMMARY.md](./DESIGN_V11_SUMMARY.md) | Master summary (start here) |
| [DESIGN_V11_MULTI_WINDOW_CACHE.md](./DESIGN_V11_MULTI_WINDOW_CACHE.md) | Full design specification |
| [DESIGN_V11_API_SPECIFICATION.md](./DESIGN_V11_API_SPECIFICATION.md) | Detailed API reference |
| [DESIGN_V11_DECISION_MATRIX.md](./DESIGN_V11_DECISION_MATRIX.md) | Trade-off analysis |
| This file | Quick reference |

---

## 🚀 Quick Start Commands

```bash
# 1. Review design
cat v7/DESIGN_V11_SUMMARY.md

# 2. Implement data structures
# Edit: v7/features/full_features.py
# Edit: v7/training/types.py

# 3. Test data structures
python -m pytest v7/tests/test_v11_data_structures.py

# 4. Implement feature extraction
# Edit: v7/features/full_features.py

# 5. Test extraction
python -m pytest v7/tests/test_v11_extraction.py

# 6. Update dataset
# Edit: v7/training/dataset.py

# 7. Integration test
python -m pytest v7/tests/test_v11_integration.py

# 8. Benchmark
python v7/tools/benchmark_v11.py

# 9. Generate production cache
python v7/tools/generate_v11_cache.py --parallel --workers 8

# 10. Train model
python v7/training/example_training.py --cache data/channels_v11.pkl
```

---

## 💡 Pro Tips

1. **Start Small:** Test with 1k samples before full dataset
2. **Use Parallel:** Always use `num_workers=8` for scanning
3. **Compress Caches:** Use `pickle.HIGHEST_PROTOCOL` + optional gzip
4. **Monitor Memory:** Reduce batch size if OOM during training
5. **Backup v10:** Keep v10 caches until v11 proven in production
6. **Check Multi-Window:** Use `has_multi_window_features()` to detect migrated vs full caches
7. **Profile First:** Benchmark before optimizing
8. **Document Changes:** Update metadata with rebuild reasons

---

## 🎓 Key Concepts

### Window-Dependent Features
Features that **change** based on window size:
- Channel boundaries (upper/lower lines shift)
- Channel position (% from bottom to top)
- Channel metrics (width, slope, bounces)
- All per-TF features (TSLA, SPY, cross)

### Window-Independent Features
Features that **don't change** with window size:
- VIX regime (market volatility)
- Historical patterns (past channel behavior)
- Event calendar (earnings, FOMC dates)
- Cross-asset alignment (high-level directional match)

### Why Split?
- **Efficiency:** Extract shared features once, reuse 8 times
- **Clarity:** Clear separation of concerns
- **Storage:** Avoid 8x duplication of shared features
- **Correctness:** Ensure shared features truly shared

---

## 🔍 Debugging Tips

### Cache Won't Load
```python
# Check version
metadata = get_cache_metadata(cache_path)
print(metadata['cache_version'])  # Should be "v11.0.0"

# Check metadata
print(metadata['full_multi_window'])  # True = full, False = migrated
```

### Tensor Shape Mismatch
```python
# Validate features
from v7.features.feature_ordering import validate_feature_dict

features_dict = features_to_tensor_dict(features)
errors = validate_feature_dict(features_dict, raise_on_error=False)
print(errors)  # List of dimension mismatches
```

### Missing Windows
```python
# Check which windows available
print(sample.per_window_features.keys())

# Get window count
print(sample.get_window_count())

# Safe access
features = sample.get_features_for_window(30)
if features is None:
    print("Window 30 not available")
```

---

## ✨ Success Criteria

- [ ] All unit tests passing
- [ ] Integration test: scan→save→load→train works
- [ ] Performance: 100k samples in <2 hours (8 cores)
- [ ] Storage: <5 GB per 100k samples (compressed)
- [ ] Memory: batch_size=64 works without OOM
- [ ] Backward compat: v10 caches load and work
- [ ] Migration: v10→v11 conversion successful

---

**Last Updated:** 2026-01-06
**Version:** 1.0
**Status:** ✅ Design Complete - Ready for Implementation
