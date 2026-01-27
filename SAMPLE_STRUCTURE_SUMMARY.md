# Sample Structure Analysis Summary

## Key Findings

### 1. ChannelSample Structure (v15/dtypes.py:294-315)
```python
@dataclass
class ChannelSample:
    timestamp: pd.Timestamp           # Channel end timestamp
    channel_end_idx: int              # Index in 5min DataFrame
    tf_features: Dict[str, float]     # ~7,880 features (flat dict)
    labels_per_window: Dict[int, Dict[str, Any]]  # Nested: window -> asset -> tf -> ChannelLabels
    bar_metadata: Dict[str, Dict[str, float]]     # TF bar completion info
    best_window: int                  # Optimal window (10-80)
```

### 2. ChannelLabels Structure (v15/dtypes.py:57-291)
Large dataclass with 100+ fields organized into categories:
- **Prediction Targets**: duration_bars, break_direction, next_channel_direction, permanent_break
- **Break Scan Features**: bars_to_first_break, break_magnitude, returned_to_channel, etc.
- **Source Channel Params**: slope, intercept, std_dev, r_squared, direction, bounce_count
- **Exit Events (Lists)**: exit_bars, exit_magnitudes, exit_durations, exit_types, exit_returned
- **SPY Fields**: All TSLA fields mirrored with spy_ prefix
- **Validity Flags**: duration_valid, direction_valid, break_scan_valid
- **RSI Labels**: rsi_at_first_break, rsi_at_permanent_break, etc. (TSLA + SPY)
- **Next Channel Labels**: best_next_channel_direction, bars_away, bounce_count, etc.

### 3. labels_per_window Structure

**NEW FORMAT** (current):
```python
{
    10: {  # window
        'tsla': {
            '5min': ChannelLabels(...),
            '15min': ChannelLabels(...),
            # ... all 10 timeframes
        },
        'spy': {
            '5min': ChannelLabels(...),
            # ... all 10 timeframes
        }
    },
    20: { ... },  # next window
    # ... windows: 10, 20, 30, 40, 50, 60, 70, 80
}
```

**OLD FORMAT** (backward compatible):
```python
{
    10: {  # window
        '5min': ChannelLabels(...),  # TSLA only
        '15min': ChannelLabels(...),
        # ... all 10 timeframes
    },
    20: { ... },
}
```

### 4. Inspector Usage (inspector.py)

**Loading**:
```python
with open(samples_path, 'rb') as f:
    samples = pickle.load(f)  # Expects List[ChannelSample]
```

**Accessing**:
```python
sample = self.samples[idx]
sample.timestamp                    # pd.Timestamp
sample.channel_end_idx              # int
sample.labels_per_window[50]['tsla']['daily']  # ChannelLabels

# Uses getattr() for labels
getattr(labels, 'bars_to_first_break', 0)
```

**Critical Requirements**:
1. Must be picklable
2. Must be a Python list
3. timestamp must be pd.Timestamp (for .searchsorted())
4. Labels must be ChannelLabels dataclass (for getattr())
5. Supports attribute access (sample.timestamp, labels.break_direction)

### 5. Dataset Usage (training/dataset.py)

**Loading**:
```python
with open(path, 'rb') as f:
    samples = pickle.load(f)

# Validation
if not hasattr(first, 'tf_features') or not hasattr(first, 'labels_per_window'):
    raise DataLoadError(...)
```

**Feature Extraction**:
```python
# Features sorted alphabetically
if self.feature_names is None:
    self.feature_names = sorted(sample.tf_features.keys())

feature_array = np.array([
    sample.tf_features.get(name, 0.0) for name in self.feature_names
], dtype=np.float32)
```

**Label Extraction**:
```python
# Handles both old and new structure
window_labels = sample.labels_per_window.get(window, {})

if 'tsla' in window_labels:
    # New structure
    tsla_labels_dict = window_labels.get('tsla', {})
    tf_labels = tsla_labels_dict.get(tf)
else:
    # Old structure
    tf_labels = window_labels.get(tf)
```

### 6. Feature Structure

**Format**: `{timeframe}_{feature_name}_{window_offset}`

**Examples**:
- `5min_price_pct_0`: 5-minute timeframe, price percent change, window 0
- `daily_rsi_14_0`: daily timeframe, RSI-14, window 0
- `monthly_volume_zscore_2`: monthly timeframe, volume z-score, window 2

**Count**: ~7,880 features total across:
- 10 timeframes (5min, 15min, 30min, 1h, 2h, 3h, 4h, daily, weekly, monthly)
- 8 standard windows (10, 20, 30, 40, 50, 60, 70, 80)
- Multiple feature types per TF/window

**Ordering**: Alphabetically sorted when converted to numpy array

### 7. Constants (v15/dtypes.py)

```python
TIMEFRAMES = [
    '5min', '15min', '30min', '1h', '2h', '3h', '4h',
    'daily', 'weekly', 'monthly'
]

STANDARD_WINDOWS = [10, 20, 30, 40, 50, 60, 70, 80]

BARS_PER_TF = {
    '5min': 1, '15min': 3, '30min': 6, '1h': 12,
    '2h': 24, '3h': 36, '4h': 48, 'daily': 78,
    'weekly': 390, 'monthly': 1638
}
```

---

## C++ Loader Implementation Strategy

### Option 1: Native Python Objects (Recommended)
Use pybind11 to create Python-compatible objects:
```cpp
// Define ChannelLabels as Python class
py::class_<ChannelLabels>(m, "ChannelLabels")
    .def(py::init<>())
    .def_readwrite("duration_bars", &ChannelLabels::duration_bars)
    .def_readwrite("break_direction", &ChannelLabels::break_direction)
    // ... all 100+ fields
    .def(py::pickle(...));  // Make picklable

// Define ChannelSample as Python class
py::class_<ChannelSample>(m, "ChannelSample")
    .def(py::init<>())
    .def_readwrite("timestamp", &ChannelSample::timestamp)  // Convert to pd.Timestamp
    .def_readwrite("tf_features", &ChannelSample::tf_features)  // py::dict
    .def(py::pickle(...));  // Make picklable
```

### Option 2: Convert to Python Types
Load in C++, convert to Python native types:
```python
# In Python wrapper
def load_samples_cpp(path):
    # Load binary via C++
    cpp_samples = cpp_loader.load(path)
    
    # Convert to Python ChannelSample objects
    python_samples = []
    for cpp_sample in cpp_samples:
        python_samples.append(ChannelSample(
            timestamp=pd.Timestamp(cpp_sample.timestamp_ns),
            channel_end_idx=cpp_sample.channel_end_idx,
            tf_features=dict(cpp_sample.tf_features),
            labels_per_window=convert_labels(cpp_sample.labels),
            bar_metadata=dict(cpp_sample.bar_metadata),
            best_window=cpp_sample.best_window
        ))
    
    return python_samples
```

---

## Critical Type Conversions

### Timestamp
```python
# C++ -> Python
timestamp = pd.Timestamp(cpp_timestamp_nanoseconds)

# Python -> C++
nanoseconds = timestamp.value  # Gets int64 nanoseconds
```

### Features Dict
```python
# C++ vector<pair<string, float>> -> Python dict
tf_features = {name: value for name, value in cpp_features}

# Must maintain alphabetical order when converting to numpy
feature_names = sorted(tf_features.keys())
```

### Labels Structure
```python
# NEW structure (nested dict)
labels_per_window = {
    window: {
        'tsla': {
            tf: convert_to_ChannelLabels(cpp_labels)
            for tf in timeframes
        },
        'spy': { ... }
    }
    for window in windows
}
```

---

## Validation Tests

After loading from C++, verify:
```python
import pickle
from v15.dtypes import ChannelSample, ChannelLabels

def validate_sample(sample):
    # Type checks
    assert isinstance(sample, ChannelSample)
    assert isinstance(sample.timestamp, pd.Timestamp)
    assert isinstance(sample.tf_features, dict)
    assert isinstance(sample.labels_per_window, dict)
    
    # Feature checks
    assert len(sample.tf_features) > 0
    assert all(isinstance(k, str) for k in sample.tf_features.keys())
    assert all(isinstance(v, float) for v in sample.tf_features.values())
    
    # Labels structure
    for window, window_dict in sample.labels_per_window.items():
        assert isinstance(window, int)
        if 'tsla' in window_dict:
            # New structure
            for tf, labels in window_dict['tsla'].items():
                assert isinstance(labels, ChannelLabels)
        else:
            # Old structure
            for tf, labels in window_dict.items():
                assert isinstance(labels, ChannelLabels)
    
    # Pickle test
    pickled = pickle.dumps(sample)
    loaded = pickle.loads(pickled)
    assert loaded.timestamp == sample.timestamp
    
    return True
```

---

## Performance Considerations

### Memory
- Each sample: ~7,880 features × 4 bytes = ~31KB (features only)
- Full sample with labels: ~50-100KB per sample
- For 10,000 samples: ~500MB-1GB total

### Loading Speed
- Python pickle: ~50-100MB/s (CPU bottleneck)
- C++ binary: Could be 5-10x faster if properly optimized
- Bottleneck will be Python object creation, not I/O

### Recommendations
1. Use memory mapping for large files
2. Lazy load labels (only when accessed)
3. Consider compression (zstd, lz4) for storage
4. Batch convert C++ -> Python for efficiency

---

## Next Steps

1. **Define Binary Format**: Document exact byte layout for C++ serialization
2. **Implement pybind11 Bindings**: Create Python-compatible C++ classes
3. **Test Pickle Compatibility**: Ensure C++ objects can be pickled/unpickled
4. **Benchmark Performance**: Compare C++ vs Python loading speed
5. **Validate with Inspector**: Load C++ samples and verify inspector works
6. **Test with Dataset**: Ensure training code can use C++ loaded samples

---

## References

- **ChannelSample Definition**: `/Users/frank/Desktop/CodingProjects/x14/v15/dtypes.py:294-315`
- **ChannelLabels Definition**: `/Users/frank/Desktop/CodingProjects/x14/v15/dtypes.py:57-291`
- **Inspector Usage**: `/Users/frank/Desktop/CodingProjects/x14/v15/inspector.py`
- **Dataset Usage**: `/Users/frank/Desktop/CodingProjects/x14/v15/training/dataset.py`
- **Sample Creation**: `/Users/frank/Desktop/CodingProjects/x14/v15/scanner.py:448-455`
