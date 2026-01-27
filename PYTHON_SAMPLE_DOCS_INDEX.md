# Python Sample Structure Documentation Index

This directory contains comprehensive documentation of the Python sample structure for implementing a C++ loader.

---

## Documents

### 1. PYTHON_SAMPLE_STRUCTURE.md
**Complete technical specification of Python sample structure**

Covers:
- ChannelSample class definition and all fields
- ChannelLabels nested structure
- labels_per_window structure (NEW and OLD formats)
- Inspector usage patterns and requirements
- Dataset usage patterns and feature extraction
- Type conversions between C++ and Python
- Validation checklist
- Example Python code

Use this for: Understanding the complete structure and implementation requirements

---

### 2. SAMPLE_STRUCTURE_SUMMARY.md
**High-level overview and implementation strategy**

Covers:
- Key findings summary
- Inspector and Dataset access patterns
- Feature structure and naming conventions
- C++ loader implementation options
- Critical type conversions
- Performance considerations
- Validation tests
- Next steps

Use this for: Quick reference and implementation planning

---

### 3. CHANNEL_LABELS_FIELDS.md
**Complete field reference for ChannelLabels dataclass**

Covers:
- All 100+ fields organized by category
- Field types and default values
- C++ type mappings
- List fields (need special handling)
- Timestamp fields (need conversion)
- Summary statistics

Use this for: Implementing the ChannelLabels C++ struct

---

## Quick Start

1. **Understanding Structure**: Read SAMPLE_STRUCTURE_SUMMARY.md first
2. **Implementation Details**: Reference PYTHON_SAMPLE_STRUCTURE.md
3. **Field Definitions**: Use CHANNEL_LABELS_FIELDS.md as checklist

---

## Key Insights

### Sample Structure
```
ChannelSample
├── timestamp: pd.Timestamp (nanoseconds since epoch)
├── channel_end_idx: int (index in 5min DataFrame)
├── tf_features: Dict[str, float] (~7,880 features)
├── labels_per_window: Dict[int, Dict[str, Dict[str, ChannelLabels]]]
│   └── window -> asset -> timeframe -> ChannelLabels
├── bar_metadata: Dict[str, Dict[str, float]]
└── best_window: int (10-80)
```

### Critical Requirements
1. Must be picklable (for inspector.py)
2. Must be Python list of ChannelSample objects
3. timestamp must be pd.Timestamp (for DataFrame operations)
4. Labels must be ChannelLabels dataclass instances (for getattr())
5. Features must maintain alphabetical ordering when converted to numpy

### Inspector vs Dataset Usage
- **Inspector**: Recomputes channels fresh, uses labels for display only
- **Dataset**: Extracts features and labels, converts to numpy arrays
- **Both**: Load via pickle.load(), expect List[ChannelSample]

---

## Implementation Paths

### Option A: Pure Python Conversion (Simplest)
1. Load binary data in C++
2. Convert to Python dicts/lists
3. Call Python ChannelSample() constructor
4. Return Python list

**Pros**: Simple, guaranteed compatibility
**Cons**: Slower, requires Python types throughout

### Option B: pybind11 Bindings (Recommended)
1. Define C++ ChannelSample/ChannelLabels classes
2. Expose via pybind11 with proper pickle support
3. Load binary data in C++
4. Construct Python objects via pybind11

**Pros**: Fast, native Python objects, type safety
**Cons**: More complex setup, requires pybind11 expertise

### Option C: Hybrid (Balanced)
1. Fast C++ loading to native C++ structs
2. Lazy Python conversion only when accessed
3. Memory mapping for large files
4. Batch conversion for efficiency

**Pros**: Best performance, flexible
**Cons**: Most complex, potential memory issues

---

## File Locations in Source

### Python Source Files
- **ChannelSample Definition**: `/Users/frank/Desktop/CodingProjects/x14/v15/dtypes.py:294-315`
- **ChannelLabels Definition**: `/Users/frank/Desktop/CodingProjects/x14/v15/dtypes.py:57-291`
- **Inspector Usage**: `/Users/frank/Desktop/CodingProjects/x14/v15/inspector.py`
- **Dataset Usage**: `/Users/frank/Desktop/CodingProjects/x14/v15/training/dataset.py`
- **Sample Creation**: `/Users/frank/Desktop/CodingProjects/x14/v15/scanner.py:448-455`
- **Constants**: `/Users/frank/Desktop/CodingProjects/x14/v15/dtypes.py:18-51`

### C++ Implementation (to be created)
- **Sample Loader**: `v15_cpp/loader/sample_loader.cpp`
- **Python Bindings**: `v15_cpp/bindings/python_bindings.cpp`
- **Data Types**: `v15_cpp/include/sample_types.hpp`
- **Binary Format**: `v15_cpp/include/binary_format.hpp`

---

## Testing Strategy

### Phase 1: Structure Validation
- [ ] Load Python pickle samples
- [ ] Verify ChannelSample structure
- [ ] Check ChannelLabels fields
- [ ] Validate nested dictionaries

### Phase 2: C++ Loader Implementation
- [ ] Define binary format
- [ ] Implement C++ structs
- [ ] Write serialization code
- [ ] Test roundtrip (Python -> C++ -> Python)

### Phase 3: Python Bindings
- [ ] Implement pybind11 bindings
- [ ] Test pickle compatibility
- [ ] Verify inspector can load C++ samples
- [ ] Benchmark performance

### Phase 4: Integration Testing
- [ ] Load samples with inspector
- [ ] Train model with C++ loaded samples
- [ ] Compare accuracy with Python loaded samples
- [ ] Performance benchmarking

---

## Common Pitfalls

1. **Timestamp Conversion**: Must use pd.Timestamp(nanoseconds), not seconds
2. **Feature Ordering**: Must be alphabetically sorted when converting to numpy
3. **Labels Structure**: Must handle both NEW (asset -> tf) and OLD (tf only) formats
4. **List Fields**: exit_bars, exit_magnitudes, etc. need special handling
5. **Pickle Compatibility**: C++ objects must be picklable or converted to Python types
6. **getattr() Access**: Labels must support getattr(labels, 'field_name', default)
7. **Dictionary Access**: Must support both dict[key] and .attribute access

---

## Performance Targets

### Current Python Performance
- Loading 10,000 samples: ~5-10 seconds
- Memory: ~500MB-1GB
- Bottleneck: pickle deserialization + Python object creation

### Target C++ Performance
- Loading 10,000 samples: <1 second (5-10x faster)
- Memory: ~300-500MB (more efficient structs)
- Bottleneck: Python object conversion (if using pure Python)

### Optimization Strategies
1. Memory mapping for large files
2. Parallel loading/conversion
3. Lazy label loading (load only when accessed)
4. Compressed storage (zstd, lz4)
5. Batch Python object creation

---

## Next Steps

1. Review all three documents thoroughly
2. Decide on implementation path (A/B/C)
3. Define binary serialization format
4. Implement C++ data structures
5. Write pybind11 bindings (if Option B)
6. Create test suite
7. Benchmark and optimize

---

## Contact / Questions

For questions about:
- **Sample Structure**: See PYTHON_SAMPLE_STRUCTURE.md
- **Field Definitions**: See CHANNEL_LABELS_FIELDS.md
- **Implementation**: See SAMPLE_STRUCTURE_SUMMARY.md
- **Source Code**: Check file locations above

---

Generated: 2026-01-26
Source Codebase: x14/v15
