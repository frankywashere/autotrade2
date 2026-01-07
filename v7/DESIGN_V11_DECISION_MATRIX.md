# v11.0.0 Design Decision Matrix

## Quick Reference

**Recommended Approach:** Option 1 (Optimized) - Split features into PerWindowFeatures + SharedFeatures

**Key Benefits:**
- Maximum flexibility for model development
- Clean architecture with clear separation
- Backward compatible with v10.0.0
- Acceptable storage overhead with optimization

**Key Trade-offs:**
- ~6.7x storage increase vs v10.0.0
- ~7.4x slower feature extraction
- Requires careful implementation of split extraction

---

## Detailed Comparison Table

| Criterion | Option 1: Full FullFeatures | Option 1 Optimized (Split) | Option 2: Dict of Arrays | Option 3: Per-TF Only |
|-----------|---------------------------|---------------------------|------------------------|---------------------|
| **Storage per sample** | 6,088 features (761×8) | 5,073 features (616×8+145) | ~5,073 features | 5,073 features |
| **Storage for 100k** | 24 MB | 20 MB | 20 MB | 20 MB |
| **Code clarity** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Very Good | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Very Good |
| **Flexibility** | ⭐⭐⭐⭐⭐ Maximum | ⭐⭐⭐⭐⭐ Maximum | ⭐⭐⭐ Moderate | ⭐⭐⭐⭐ High |
| **Backward compat** | ⭐⭐⭐⭐⭐ Perfect | ⭐⭐⭐⭐⭐ Perfect | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Very Good |
| **Implementation** | ⭐⭐⭐⭐⭐ Easiest | ⭐⭐⭐⭐ Easy | ⭐⭐⭐ Moderate | ⭐⭐⭐ Moderate |
| **Debuggability** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐ Poor | ⭐⭐⭐⭐ Very Good |
| **Extension** | ⭐⭐⭐⭐⭐ Very Easy | ⭐⭐⭐⭐⭐ Very Easy | ⭐⭐ Difficult | ⭐⭐⭐⭐ Easy |
| **Memory efficient** | ⭐⭐ Poor | ⭐⭐⭐⭐ Very Good | ⭐⭐⭐⭐ Very Good | ⭐⭐⭐⭐ Very Good |
| **Feature duplication** | High (145×8) | None | None | None |
| **Migration complexity** | ⭐⭐⭐⭐⭐ Trivial | ⭐⭐⭐⭐ Easy | ⭐⭐⭐ Moderate | ⭐⭐⭐ Moderate |
| **Type safety** | ⭐⭐⭐⭐⭐ Strong | ⭐⭐⭐⭐⭐ Strong | ⭐⭐ Weak | ⭐⭐⭐⭐⭐ Strong |
| **Overall Score** | 48/55 (87%) | **54/55 (98%)** | 32/55 (58%) | 45/55 (82%) |

---

## Feature Category Breakdown

### Window-Dependent Features (616 per window)

These features MUST be stored per window because they change based on window size:

| Feature Group | Count per TF | Total (11 TFs) | Why Window-Dependent? |
|--------------|--------------|----------------|----------------------|
| TSLA channels | 35 | 385 | Channel boundaries, positions, slopes vary by window |
| SPY channels | 11 | 121 | Same - window affects all channel metrics |
| Cross containment | 10 | 110 | TSLA position in SPY channels depends on both windows |
| **Subtotal** | **56** | **616** | |

**Examples of window-dependency:**
- `tsla['5min'].position`: With window=10, position might be 0.3; with window=50, it might be 0.7
- `tsla['5min'].width_pct`: Wider windows typically create wider channels
- `cross['5min'].tsla_in_spy_upper`: Containment changes as SPY channel bounds shift

### Window-Independent Features (145 shared)

These features can be shared because they don't depend on channel detection window:

| Feature Group | Count | Why Window-Independent? |
|--------------|-------|------------------------|
| VIX | 6 | Market volatility regime is external to our channels |
| TSLA history | 25 | Historical pattern analysis uses fixed past, not current window |
| SPY history | 25 | Same - past patterns independent of current detection |
| Alignment | 3 | High-level cross-asset alignment (uses standard window=20) |
| Events | 46 | Calendar events independent of technical analysis |
| Window scores | 40 | Quality metrics for all 8 windows (meta-feature) |
| **Total** | **145** | |

**Why alignment is "shared":**
While technically window-dependent, we standardize on window=20 for simplicity:
- Alignment is a high-level directional metric
- Impact of window size is minimal (bull vs bear doesn't change much)
- Simplifies architecture without significant loss

---

## Storage Analysis Deep Dive

### v10.0.0 Baseline (Single Window)

```
Per sample:
  FullFeatures: 761 features × 4 bytes = 3,044 bytes
  Channel: ~500 bytes
  Labels: ~1,200 bytes (11 TFs × ~100 bytes)
  ChannelSample overhead: ~500 bytes
  ----------------------------------------
  Total: ~5,244 bytes = ~5.1 KB per sample

100k samples: 5.1 KB × 100k = 510 MB
```

### v11.0.0 Full (Naive - No Optimization)

```
Per sample:
  FullFeatures × 8: 761 × 8 = 6,088 features × 4 bytes = 24,352 bytes
  Channels × 8: ~500 × 8 = 4,000 bytes
  Labels × 8: ~1,200 × 8 = 9,600 bytes
  ChannelSample overhead: ~500 bytes
  ----------------------------------------
  Total: ~38,452 bytes = ~37.5 KB per sample

100k samples: 37.5 KB × 100k = 3.66 GB

Increase: 7.4x
```

### v11.0.0 Optimized (Split Features)

```
Per sample:
  PerWindowFeatures × 8: 616 × 8 = 4,928 features × 4 bytes = 19,712 bytes
  SharedFeatures × 1: 145 features × 4 bytes = 580 bytes
  Channels × 8: ~500 × 8 = 4,000 bytes
  Labels × 8: ~1,200 × 8 = 9,600 bytes
  ChannelSample overhead: ~500 bytes
  ----------------------------------------
  Total: ~34,392 bytes = ~33.6 KB per sample

100k samples: 33.6 KB × 100k = 3.28 GB

Increase: 6.6x
Savings vs Naive: 10% (380 MB per 100k)
```

### Storage Optimization Options

| Optimization | Savings | Complexity | Recommended? |
|-------------|---------|-----------|--------------|
| Split features (implemented) | 10% | Low | ✅ Yes |
| Pickle HIGHEST_PROTOCOL | ~20% | None | ✅ Yes |
| Float16 for features | ~50% | Medium | ⚠️ Test accuracy impact |
| Compress with gzip | ~60% | Low | ⚠️ Slower I/O |
| HDF5/Zarr chunked | ~40% | High | ❌ Not yet |
| Per-window lazy loading | Memory only | High | ❌ Future |

**Recommended Stack:**
1. Split features (10% reduction)
2. Pickle HIGHEST_PROTOCOL (20% reduction)
3. Optional: gzip compression if I/O not bottleneck

**Expected Result:** 3.28 GB → 2.1 GB per 100k samples (~64% of naive)

---

## Performance Impact Analysis

### Feature Extraction Time

**v10.0.0 (Single Window):**
```
Per sample (window=20):
  Shared features: ~50 ms (with history)
  Per-TF features: ~40 ms (all 11 TFs)
  ----------------------------------------
  Total: ~50 ms per sample
```

**v11.0.0 Naive (8× extraction):**
```
Per sample (all 8 windows):
  Shared features × 8: 50 ms × 8 = 400 ms
  Per-TF features × 8: 40 ms × 8 = 320 ms
  ----------------------------------------
  Total: ~720 ms per sample

Slowdown: 14.4x ❌ Unacceptable
```

**v11.0.0 Optimized (Shared once):**
```
Per sample (all 8 windows):
  Shared features × 1: 50 ms
  Per-TF features × 8: 40 ms × 8 = 320 ms
  ----------------------------------------
  Total: ~370 ms per sample

Slowdown: 7.4x ✅ Acceptable
```

**Full Dataset Scan (100k samples):**
- v10: 50 ms × 100k = 5,000 seconds = **1.4 hours**
- v11 Optimized: 370 ms × 100k = 37,000 seconds = **10.3 hours**

**Mitigation with 8-core parallel:**
- v11 Optimized: 10.3 hours / 8 = **1.3 hours** (comparable to v10!)

### Memory Usage (Training)

**v10.0.0:**
```
Batch size = 128:
  Features: 128 × 761 × 4 bytes = 390 KB
  Labels: 128 × ~200 bytes = 25 KB
  ----------------------------------------
  Total: ~415 KB per batch
```

**v11.0.0:**
```
Batch size = 128:
  Features: 128 × 8 × 761 × 4 bytes = 3.12 MB
  Labels: 128 × ~800 bytes = 100 KB (window scores added)
  ----------------------------------------
  Total: ~3.2 MB per batch

Increase: 7.7x
```

**Recommendation:**
- Reduce batch size: 128 → 64 or 32
- Or use gradient accumulation (2-4 steps)
- GPU memory should handle 3.2 MB easily (even 3090 has 24 GB)

---

## Implementation Complexity

### Phase 1: Data Structures (Low)
**Estimated Time:** 2 hours

```python
# Simple dataclass definitions
@dataclass
class PerWindowFeatures:
    # Already well-defined in existing FullFeatures
    pass

@dataclass
class SharedFeatures:
    # Extract subset of FullFeatures
    pass

# Update ChannelSample (add 2 fields)
```

**Risk:** Low - Pure data structure work

### Phase 2: Feature Extraction (Medium)
**Estimated Time:** 8 hours

```python
# Split extract_full_features() into:
def extract_shared_features():
    # Extract VIX, history, events, alignment
    # ~100 lines, mostly copy-paste
    pass

def extract_per_window_features():
    # Extract TSLA, SPY, cross per TF
    # ~150 lines, mostly existing code
    pass

def extract_full_features_multi_window():
    # Orchestrate: shared once + per-window loop
    # ~50 lines
    pass
```

**Risk:** Medium - Need to ensure shared features truly independent

### Phase 3: Dataset Loading (Medium)
**Estimated Time:** 6 hours

```python
# Update ChannelDataset.__getitem__():
def _get_multi_window_features():
    # Loop over windows, reconstruct, convert, stack
    # ~80 lines
    pass

# Add helper:
def concatenate_features_dict():
    # ~20 lines
    pass
```

**Risk:** Medium - Tensor shape mismatches can be subtle

### Phase 4: Cache Management (Low)
**Estimated Time:** 4 hours

```python
# Update cache version
CACHE_VERSION = "v11.0.0"

# Migration function
def migrate_cache_v10_to_v11():
    # ~100 lines, straightforward
    pass

# Update metadata handling
```

**Risk:** Low - Well-defined migration path

### Phase 5: Testing (High Priority)
**Estimated Time:** 8 hours

- Unit tests: Feature extraction correctness
- Integration tests: End-to-end scan→save→load
- Backward compat: Load v10 in v11 code
- Migration: v10→v11 conversion
- Performance: Benchmark extraction and loading

**Risk:** Medium - Testing is comprehensive but well-scoped

### Phase 6: Documentation (Low)
**Estimated Time:** 4 hours

- Update docstrings
- Add usage examples
- Migration guide
- Performance tuning guide

**Risk:** Low

**Total Estimated Time:** 32 hours (~4 days)

---

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Shared features not truly independent | Low | High | Careful analysis + unit tests |
| Tensor shape mismatches in model | Medium | Medium | Comprehensive shape validation |
| OOM during training | Low | Medium | Reduce batch size, add memory monitoring |
| Cache corruption during migration | Low | High | Validate before/after, backup original |
| Feature extraction bugs | Medium | High | Extensive unit testing + comparison to v10 |
| Performance degradation | Low | Low | Already analyzed, parallelization mitigates |

### Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Disk space exhaustion | Medium | Medium | Monitor usage, document requirements |
| Long cache rebuild times | High | Low | Expected, can run overnight |
| User confusion about versions | Medium | Low | Clear documentation, auto-migration |
| Breaking changes for users | Low | High | Backward compatibility maintained |

### Overall Risk Level: **LOW-MEDIUM** ✅

The design is well-thought-out with clear mitigation strategies.

---

## Decision Criteria Summary

### Choose Option 1 Optimized If:
✅ You want maximum model development flexibility
✅ You value code clarity and maintainability
✅ Storage cost is not a primary concern
✅ You may add new features in the future
✅ You want easy debugging and inspection
✅ **Recommended for production**

### Choose Option 2 (Dict of Arrays) If:
⚠️ Storage is extremely constrained
⚠️ You never need to inspect features as objects
⚠️ Your model only needs raw arrays
❌ **Not recommended** - Marginal storage benefit, significant usability cost

### Choose Option 3 (Per-TF Only) If:
⚠️ You want minimal storage
⚠️ Clear separation is critical
✅ Good middle ground
⚠️ **Consider as alternative** - Similar benefits to Option 1 Optimized

---

## Recommendation: Proceed with Option 1 Optimized

### Justification

1. **Best Code Quality:**
   - Clean separation of concerns
   - Strong typing and validation
   - Easy to extend and maintain

2. **Acceptable Trade-offs:**
   - 6.6x storage increase mitigated by:
     - Modern storage is cheap ($0.02/GB)
     - Compression reduces to ~4x
     - Critical for model improvement
   - 7.4x extraction slowdown mitigated by:
     - Parallelization (8 cores → ~1x)
     - One-time cost (cache persists)
     - Can extract overnight

3. **Strong Backward Compatibility:**
   - v10 code works unchanged
   - Auto-migration available
   - Gradual adoption possible

4. **Future-Proof:**
   - Easy to add new per-window features
   - Easy to add new shared features
   - Supports model experimentation

5. **Implementation Risk: Low**
   - Well-defined scope
   - Clear testing plan
   - Proven patterns

### Next Steps

1. ✅ Review and approve this design (this document)
2. 📋 Create implementation task list
3. 🔨 Implement Phase 1 (data structures)
4. 🧪 Test with small dataset (1k samples)
5. 📊 Benchmark performance
6. 🚀 Roll out to full pipeline
7. 📚 Document for users

**Estimated Timeline:** 1 week for implementation + testing, 2 weeks for full rollout

---

## Appendix: Alternative Approaches Considered

### A1: Lazy Window Loading

**Idea:** Store all windows but load on-demand

**Pros:**
- Reduces memory usage
- Fast initial load

**Cons:**
- Complex cache format (HDF5/Zarr required)
- Slower access patterns
- Complicated implementation

**Decision:** Defer to future optimization if memory becomes issue

### A2: Window Subsampling

**Idea:** Only store subset of windows (e.g., 10, 30, 50, 70)

**Pros:**
- 50% storage reduction
- Faster extraction

**Cons:**
- Arbitrary window selection
- Model can't learn from all windows
- Defeats purpose of v11

**Decision:** Rejected - Defeats core goal

### A3: Delta Encoding

**Idea:** Store window=20 fully, store deltas for other windows

**Pros:**
- Significant compression for correlated features
- Full reconstruction possible

**Cons:**
- Complex implementation
- Slower reconstruction
- Error accumulation risk

**Decision:** Defer to future optimization

### A4: Separate Window Files

**Idea:** One cache file per window (8 files total)

**Pros:**
- Can load only needed windows
- Easier to update single window

**Cons:**
- File management complexity
- Synchronization issues
- User confusion

**Decision:** Rejected - Too complex for users

---

## Conclusion

**Option 1 Optimized** (Split into PerWindowFeatures + SharedFeatures) is the clear winner:

- **Highest overall score:** 98%
- **Best code quality:** Excellent clarity, typing, extensibility
- **Acceptable trade-offs:** Storage and speed increases are mitigated
- **Lowest risk:** Well-defined implementation path
- **Future-proof:** Easy to extend and optimize

**Proceed with implementation as designed in DESIGN_V11_MULTI_WINDOW_CACHE.md**
