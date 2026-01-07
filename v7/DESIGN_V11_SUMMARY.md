# v11.0.0 Multi-Window Cache - Design Summary

## Documentation Index

This is the master summary document. For detailed specifications, see:

1. **[DESIGN_V11_MULTI_WINDOW_CACHE.md](./DESIGN_V11_MULTI_WINDOW_CACHE.md)** - Complete design rationale, data structures, and implementation plan
2. **[DESIGN_V11_API_SPECIFICATION.md](./DESIGN_V11_API_SPECIFICATION.md)** - Detailed API docs, usage examples, and performance analysis
3. **[DESIGN_V11_DECISION_MATRIX.md](./DESIGN_V11_DECISION_MATRIX.md)** - Comparison table, trade-off analysis, and decision justification

---

## Executive Summary

### Problem
v10.0.0 selects the best window **before** feature extraction, so the model never sees features from alternative windows. This limits the model's ability to learn which window provides the best signal for current market conditions.

### Solution
v11.0.0 extends the cache to store features from **all 8 windows** (10, 20, 30, 40, 50, 60, 70, 80) per sample, enabling the model to:
1. See complete feature context from all windows
2. Learn optimal window selection from data
3. Make window selection part of the prediction task

### Design Choice
**Split Architecture:** Separate features into PerWindowFeatures (616 features × 8 windows) + SharedFeatures (145 features × 1 copy)

### Key Metrics
- **Storage:** 6.6x increase (~3.3 GB per 100k samples, compressed to ~2.1 GB)
- **Extraction Speed:** 7.4x slower (mitigated to ~1x with 8-core parallelization)
- **Memory:** 7.7x per batch (reduce batch size 128→64 or use gradient accumulation)
- **Implementation Time:** ~4 days (32 hours estimated)
- **Risk Level:** Low-Medium (well-defined scope, clear mitigation)

---

## Core Data Structures

### PerWindowFeatures (New in v11.0.0)
```python
@dataclass
class PerWindowFeatures:
    """Window-dependent features (616 total)."""
    timestamp: pd.Timestamp
    window: int  # 10, 20, 30, 40, 50, 60, 70, or 80

    tsla: Dict[str, TSLAChannelFeatures]  # 35 features × 11 TFs = 385
    spy: Dict[str, SPYFeatures]  # 11 features × 11 TFs = 121
    cross_containment: Dict[str, CrossAssetContainment]  # 10 features × 11 TFs = 110
```

### SharedFeatures (New in v11.0.0)
```python
@dataclass
class SharedFeatures:
    """Window-independent features (145 total)."""
    timestamp: pd.Timestamp

    vix: VIXFeatures  # 6 features
    tsla_history: ChannelHistoryFeatures  # 25 features
    spy_history: ChannelHistoryFeatures  # 25 features
    tsla_spy_direction_match: bool  # 1
    both_near_upper: bool  # 1
    both_near_lower: bool  # 1
    events: Optional[EventFeatures]  # 46 features
    tsla_window_scores: Optional[np.ndarray]  # 40 features (8×5)
```

### ChannelSample (Extended in v11.0.0)
```python
@dataclass
class ChannelSample:
    # v10.0.0 fields (backward compatible)
    timestamp: pd.Timestamp
    channel_end_idx: int
    channel: Channel
    features: FullFeatures
    labels: Dict[str, ChannelLabels]
    channels: Dict[int, Channel]
    best_window: int
    labels_per_window: Dict[int, Dict[str, ChannelLabels]]

    # NEW in v11.0.0
    per_window_features: Dict[int, PerWindowFeatures]  # {10: features, 20: features, ...}
    shared_features: SharedFeatures  # Single shared instance
```

---

## Key API Functions

### Feature Extraction
```python
# NEW: Extract features for all windows
per_window, shared = extract_full_features_multi_window(
    tsla_df, spy_df, vix_df,
    windows=STANDARD_WINDOWS,  # [10, 20, 30, 40, 50, 60, 70, 80]
    include_history=True,
    events_handler=events
)

# Shared features extracted once
shared = extract_shared_features(tsla_df, spy_df, vix_df)

# Per-window features extracted for each window
per_win_20 = extract_per_window_features(tsla_df, spy_df, window=20, timestamp=now)
```

### Dataset Loading
```python
# Load with auto-migration
samples = load_cache_with_auto_migration(
    Path('data/channels_v10.pkl'),
    auto_migrate=True  # Converts v10 → v11 automatically
)

# Dataset returns multi-window features
features, labels = dataset[0]
# features.shape = [num_windows, num_features] = [8, 761]
```

### Sample Inspection
```python
sample = samples[0]

# Access features for specific window
features_20 = sample.get_features_for_window(20)

# Access all windows
all_features = sample.get_all_window_features()

# Check if multi-window
is_v11 = sample.has_multi_window_features()
```

---

## Implementation Roadmap

### Phase 1: Data Structures (2 hours)
- [ ] Define `PerWindowFeatures` dataclass
- [ ] Define `SharedFeatures` dataclass
- [ ] Update `FullFeatures.from_split_features()` classmethod
- [ ] Extend `ChannelSample` with v11 fields
- [ ] Add helper methods (`get_features_for_window`, etc.)

### Phase 2: Feature Extraction (8 hours)
- [ ] Implement `extract_shared_features()`
- [ ] Implement `extract_per_window_features()`
- [ ] Implement `extract_full_features_multi_window()`
- [ ] Update parallel scanner to call new functions
- [ ] Add resample caching optimization

### Phase 3: Dataset Loading (6 hours)
- [ ] Update `ChannelDataset.__getitem__()` for multi-window
- [ ] Implement `_get_multi_window_features()`
- [ ] Implement `concatenate_features_dict()`
- [ ] Update batch collation functions
- [ ] Add tensor shape validation

### Phase 4: Cache Management (4 hours)
- [ ] Update `CACHE_VERSION = "v11.0.0"`
- [ ] Implement `migrate_cache_v10_to_v11()`
- [ ] Update `load_cache_with_auto_migration()`
- [ ] Add v11 metadata fields
- [ ] Update cache validation functions

### Phase 5: Testing (8 hours)
- [ ] Unit test: Split feature extraction correctness
- [ ] Unit test: Multi-window tensor construction
- [ ] Integration test: End-to-end scan→save→load
- [ ] Backward compat test: Load v10 in v11 code
- [ ] Migration test: v10→v11 conversion
- [ ] Performance benchmark: v10 vs v11 speed

### Phase 6: Documentation (4 hours)
- [ ] Update feature extraction docs
- [ ] Update dataset docs
- [ ] Add migration guide
- [ ] Add performance tuning guide
- [ ] Add usage examples

**Total: 32 hours (~4 days)**

---

## Performance Specifications

### Storage Requirements

| Dataset Size | v10.0.0 | v11.0.0 Raw | v11.0.0 Compressed |
|--------------|---------|-------------|-------------------|
| 50k samples | 255 MB | 1.64 GB | 1.05 GB |
| 100k samples | 510 MB | 3.28 GB | 2.10 GB |
| 200k samples | 1.02 GB | 6.56 GB | 4.20 GB |

**Recommendation:** Use pickle HIGHEST_PROTOCOL + optional gzip compression

### Extraction Performance

| Operation | v10.0.0 | v11.0.0 Sequential | v11.0.0 Parallel (8 cores) |
|-----------|---------|-------------------|---------------------------|
| Per sample | 50 ms | 370 ms (7.4x) | ~50 ms (1.1x) |
| 100k samples | 1.4 hours | 10.3 hours | **1.3 hours** |

**Recommendation:** Use parallel scanning with 8 workers

### Training Memory

| Batch Size | v10.0.0 | v11.0.0 | Recommendation |
|-----------|---------|---------|----------------|
| 128 | 415 KB | 3.2 MB (7.7x) | Reduce to 64 or use grad accumulation |
| 64 | 208 KB | 1.6 MB | ✅ Recommended |
| 32 | 104 KB | 800 KB | Use if memory constrained |

**Note:** Even RTX 3090 (24 GB) easily handles 3.2 MB batches

---

## Migration Strategy

### Automatic Migration (Recommended)
```python
# Load v10 cache with auto-migration to v11
samples = load_cache_with_auto_migration(
    Path('data/channels_v10.pkl'),
    auto_migrate=True  # Creates channels_v10_v11.pkl
)
```

**Result:**
- Creates new v11-format cache with single window
- Original v10 cache unchanged
- Limited multi-window features (only best_window)
- Backward compatible with v10 code

### Full Rebuild (Recommended for Production)
```python
# Scan data with v11 feature extraction
samples = scan_valid_channels(
    tsla_df, spy_df, vix_df,
    window=20, step=10,
    include_history=True,
    use_parallel=True,
    num_workers=8
)
```

**Result:**
- Complete multi-window features (all 8 windows)
- Full v11 capabilities
- Takes ~1.3 hours for 100k samples (8 cores)

### Version Detection
```python
# Cache metadata includes version info
metadata = get_cache_metadata(Path('data/channels.pkl'))
print(metadata['cache_version'])  # "v11.0.0"
print(metadata['full_multi_window'])  # True for full rebuild, False for migrated
```

---

## Trade-off Summary

### Pros ✅
1. **Better Learning:** Model sees all windows, learns optimal selection
2. **Maximum Flexibility:** Easy to add new features, extend architecture
3. **Clean Code:** Clear separation of window-dependent vs shared features
4. **Strong Typing:** Dataclasses provide validation and IDE support
5. **Backward Compatible:** v10 code works unchanged, gradual migration
6. **Future-Proof:** Easy to optimize (lazy loading, compression, etc.)

### Cons ❌
1. **Storage:** 6.6x increase (mitigated: ~4x with compression, ~$0.40 per 100k samples)
2. **Extraction Speed:** 7.4x slower (mitigated: ~1x with 8-core parallelization)
3. **Memory:** 7.7x per batch (mitigated: reduce batch size or gradient accumulation)
4. **Complexity:** More code, more testing (mitigated: well-defined scope, 4-day timeline)

### Net Assessment: **Strongly Positive** ✅

The benefits (model improvement potential) far outweigh the costs (storage/compute), especially with mitigations applied.

---

## Risk Mitigation

### Technical Risks

| Risk | Mitigation |
|------|------------|
| Shared features not truly independent | Careful analysis of each feature, unit tests comparing windows |
| Tensor shape mismatches | Comprehensive validation in `concatenate_features_dict()` |
| OOM during training | Document recommended batch sizes, add memory monitoring |
| Feature extraction bugs | Extensive unit tests, compare outputs to v10 for best_window |

### Operational Risks

| Risk | Mitigation |
|------|------------|
| Disk space exhaustion | Document requirements (4 GB per 100k), monitor usage |
| Long rebuild times | Expected and acceptable, can run overnight |
| User confusion | Clear docs, auto-migration, backward compatibility |
| Breaking changes | v10 code works unchanged, `features` field maintained |

### Overall Risk: **LOW** ✅

All risks have clear mitigation strategies and are well-understood.

---

## Success Metrics

### Implementation Success
- [ ] All phases completed within 4 days
- [ ] All unit tests passing
- [ ] Integration test: scan→save→load→train works end-to-end
- [ ] Backward compat test: v10 cache loads and works in v11 code
- [ ] Performance: 100k sample extraction in <2 hours (8 cores)

### Model Success (Post-Implementation)
- [ ] Model can select different windows for different samples
- [ ] Window selection head converges during training
- [ ] Prediction accuracy improves vs v10 (baseline: same best_window)
- [ ] Attention over windows shows interpretable patterns

### Operational Success
- [ ] Caches build successfully on production data
- [ ] Storage usage within budget (<5 GB per 100k samples)
- [ ] Training memory usage manageable (batch_size=64 works)
- [ ] Users can migrate v10→v11 smoothly

---

## Usage Examples

### Generate v11 Cache
```python
from v7.training.scanning import scan_valid_channels

samples = scan_valid_channels(
    tsla_df, spy_df, vix_df,
    window=20, step=10,
    include_history=True,
    use_parallel=True,
    num_workers=8
)

# Save
with open('data/channels_v11.pkl', 'wb') as f:
    pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
```

### Load and Inspect
```python
from v7.training.dataset import load_cache_with_auto_migration

samples = load_cache_with_auto_migration(
    Path('data/channels_v11.pkl'),
    auto_migrate=True
)

# Inspect multi-window features
sample = samples[0]
print(f"Windows available: {list(sample.per_window_features.keys())}")

for window in [10, 20, 40, 80]:
    features = sample.get_features_for_window(window)
    pos = features.tsla['5min'].position
    print(f"Window {window}: position={pos:.3f}")
```

### Train Model
```python
from v7.training.dataset import ChannelDataset
from torch.utils.data import DataLoader

dataset = ChannelDataset(samples=samples)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

for features, labels in dataloader:
    # features.shape = [64, 8, 761]
    # Process all windows with window selection head
    window_scores = model.window_selector(features)  # [64, 8]
    selected = window_scores.argmax(dim=1)  # [64]

    # Or use attention over windows
    attended = model.window_attention(features)  # [64, 761]
```

---

## Next Steps

### Immediate (This Week)
1. ✅ Review and approve design documents
2. 📋 Create detailed task breakdown in project tracker
3. 🔨 Implement Phase 1 (data structures)
4. 🧪 Test Phase 1 with small sample

### Short-term (Next Week)
1. 🔨 Implement Phases 2-4 (extraction, dataset, cache)
2. 🧪 Comprehensive testing (unit + integration)
3. 📊 Performance benchmarking
4. 📚 Documentation updates

### Medium-term (2-3 Weeks)
1. 🚀 Roll out to production pipeline
2. 🔄 Rebuild production caches with v11
3. 🤖 Update model architecture for multi-window input
4. 📈 Train and evaluate v11 models

### Long-term (1-2 Months)
1. 🎯 Analyze window selection patterns
2. 🔬 A/B test v11 vs v10 predictions
3. 🚀 Deploy to production if successful
4. 📊 Monitor performance and optimize

---

## Questions & Answers

### Q: Can I still use v10 code with v11 caches?
**A:** Yes! The `features`, `channel`, and `labels` fields are maintained for backward compatibility. v10 code will use the best_window features automatically.

### Q: What happens if I load a v10 cache in v11 code?
**A:** With `auto_migrate=True`, it's automatically converted to v11 format (limited - only best_window). Or rebuild from scratch for full multi-window support.

### Q: How much will this cost in storage?
**A:** ~2 GB per 100k samples (compressed). At $0.02/GB for HDD, that's $0.04 per 100k samples. Negligible.

### Q: Will this slow down training?
**A:** Slightly higher memory usage per batch (reduce batch size), but training speed should be similar since it's still the same total compute per epoch.

### Q: Can I add more windows later?
**A:** Yes! The design supports any set of windows. Just rebuild cache with new `STANDARD_WINDOWS` list.

### Q: What if I only want to use 4 windows instead of 8?
**A:** Extract all 8, then subsample during dataset loading. Or modify `STANDARD_WINDOWS` and rebuild.

### Q: How do I know if my cache has full v11 multi-window features?
**A:** Check `sample.has_multi_window_features()` or inspect metadata `full_multi_window` field.

---

## Conclusion

v11.0.0 multi-window cache design provides a **clean, flexible, and backward-compatible** architecture for storing features from all detection windows. The split into PerWindowFeatures + SharedFeatures achieves optimal storage efficiency while maintaining code clarity.

**Key strengths:**
- Well-defined scope and implementation plan
- Clear performance trade-offs with practical mitigations
- Strong backward compatibility
- Future-proof and extensible

**Recommendation:** **Proceed with implementation** as specified in the three design documents.

**Estimated Timeline:** 1 week for implementation + testing, 2-3 weeks for full production rollout.

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-06 | Claude | Initial design |

## References

1. [DESIGN_V11_MULTI_WINDOW_CACHE.md](./DESIGN_V11_MULTI_WINDOW_CACHE.md) - Full design specification
2. [DESIGN_V11_API_SPECIFICATION.md](./DESIGN_V11_API_SPECIFICATION.md) - API details and examples
3. [DESIGN_V11_DECISION_MATRIX.md](./DESIGN_V11_DECISION_MATRIX.md) - Decision analysis
4. Current v10.0.0 implementation:
   - `/Users/frank/Desktop/CodingProjects/x6/v7/features/full_features.py`
   - `/Users/frank/Desktop/CodingProjects/x6/v7/training/types.py`
   - `/Users/frank/Desktop/CodingProjects/x6/v7/training/dataset.py`
   - `/Users/frank/Desktop/CodingProjects/x6/v7/training/scanning.py`
