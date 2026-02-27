# Python Sample Structure Documentation

## Purpose
This document defines the exact structure of `ChannelSample` objects that the Python inspector and training code expect. This ensures the C++ loader creates compatible Python objects.

---

## Core Python Class: `ChannelSample`

**Location**: `/Users/frank/Desktop/CodingProjects/x14/v15/dtypes.py` (lines 294-315)

```python
@dataclass
class ChannelSample:
    """
    A complete sample for V15 channel prediction.
    
    ARCHITECTURE: Samples are created ONLY at channel end positions.
    Therefore: sample_position = channel_end_idx = the point of prediction.
    """
    timestamp: pd.Timestamp = None
    channel_end_idx: int = 0
    tf_features: Dict[str, float] = field(default_factory=dict)
    labels_per_window: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    bar_metadata: Dict[str, Dict[str, float]] = field(default_factory=dict)
    best_window: int = 50
```

---

## Field Descriptions

### 1. `timestamp` (pd.Timestamp)
- **Type**: `pandas.Timestamp`
- **Description**: The timestamp at the channel end position (= sample position = prediction point)
- **Example**: `Timestamp('2024-01-15 14:30:00')`
- **C++ Equivalent**: `int64_t` (nanoseconds since epoch)
- **Conversion**: 
  ```python
  # Python to C++: timestamp.value (gets nanoseconds)
  # C++ to Python: pd.Timestamp(nanoseconds)
  ```

### 2. `channel_end_idx` (int)
- **Type**: `int`
- **Description**: Index in the 5-minute TSLA DataFrame where the channel ends
- **Purpose**: Used by inspector to locate the sample position in resampled data
- **Example**: `15432`

### 3. `tf_features` (Dict[str, float])
- **Type**: `Dict[str, float]` - flat dictionary of feature names to values
- **Description**: ALL features for the sample, with TF-prefixed keys
- **Structure**:
  ```python
  {
      '5min_price_pct_0': 0.0234,
      '5min_price_pct_1': -0.0123,
      '5min_volume_zscore_0': 1.45,
      # ... ~7,880 total features across all TFs and windows
      'daily_momentum_14_0': 0.567,
      'monthly_rsi_14_0': 52.3,
  }
  ```
- **Key Pattern**: `{timeframe}_{feature_name}_{window_offset}`
- **Feature Count**: Approximately 7,880 features total (from `config.TOTAL_FEATURES`)
- **Usage**: 
  - Inspector: NOT used (inspector computes fresh)
  - Dataset: Converted to ordered numpy array for training
- **Ordering**: 
  ```python
  # In dataset.py line 141:
  if self.feature_names is None:
      self.feature_names = sorted(sample.tf_features.keys())
  feature_array = np.array([
      sample.tf_features.get(name, 0.0) for name in self.feature_names
  ], dtype=np.float32)
  ```

### 4. `labels_per_window` (Dict[int, Dict[str, Any]])
- **Type**: Nested dictionary structure
- **Description**: Labels for each window/asset/timeframe combination
- **Structure** (NEW format):
  ```python
  {
      10: {  # window size
          'tsla': {  # asset
              '5min': ChannelLabels(...),    # timeframe
              '15min': ChannelLabels(...),
              # ... all 10 timeframes
          },
          'spy': {  # asset
              '5min': ChannelLabels(...),
              '15min': ChannelLabels(...),
              # ... all 10 timeframes
          }
      },
      20: { ... },  # next window
      # ... all 8 windows (10, 20, 30, 40, 50, 60, 70, 80)
  }
  ```
- **OLD format** (backward compatible):
  ```python
  {
      10: {  # window size
          '5min': ChannelLabels(...),   # timeframe -> TSLA labels only
          '15min': ChannelLabels(...),
          # ... all 10 timeframes
      },
      20: { ... },
      # ... all 8 windows
  }
  ```
- **Usage**:
  - Inspector: Uses to display label information in info panel
  - Dataset: Extracts labels for target timeframe/window for training
- **Access Pattern** (dataset.py lines 235-250):
  ```python
  window_labels = sample.labels_per_window.get(window, {})
  
  # Handle both structures:
  if 'tsla' in window_labels:
      # New structure
      tsla_labels_dict = window_labels.get('tsla', {})
      spy_labels_dict = window_labels.get('spy', {})
      tf_labels = tsla_labels_dict.get(tf)
      spy_tf_labels = spy_labels_dict.get(tf)
  else:
      # Old structure
      tf_labels = window_labels.get(tf)
      spy_tf_labels = None
  ```

### 5. `bar_metadata` (Dict[str, Dict[str, float]])
- **Type**: Nested dictionary with timeframe info
- **Description**: Partial bar completion information per timeframe
- **Structure**:
  ```python
  {
      '5min': {
          'bar_completion_pct': 1.0,
          'bars_in_partial': 0,
          'total_bars': 245,
          'is_partial': False
      },
      '15min': {
          'bar_completion_pct': 0.67,
          'bars_in_partial': 2,
          'total_bars': 82,
          'is_partial': True
      },
      # ... for all timeframes
  }
  ```
- **Usage**: Metadata only, not actively used by inspector or training

### 6. `best_window` (int)
- **Type**: `int`
- **Description**: The optimal window size for this sample
- **Values**: One of `[10, 20, 30, 40, 50, 60, 70, 80]`
- **Selection Criteria** (from scanner.py):
  - The window of the PRIMARY channel that triggered sample creation
  - Based on channel quality metrics (bounce_count, r_squared)
- **Usage**:
  - Inspector: Fallback for window selection
  - Dataset: Used when no specific window strategy is specified

---

## Nested Type: `ChannelLabels`

**Location**: `/Users/frank/Desktop/CodingProjects/x14/v15/dtypes.py` (lines 57-291)

This is a large dataclass with ~100+ fields. Key fields the inspector uses:

### Core Prediction Targets
```python
duration_bars: int = 0              # How long until channel breaks
break_direction: int = 0            # Which bound breached FIRST (0=DOWN, 1=UP)
next_channel_direction: int = 1     # Direction after break (0=BEAR, 1=SIDE, 2=BULL)
permanent_break: bool = False       # Whether break sticks
```

### Break Scan Features (Inspector Display)
```python
# TSLA break dynamics
bars_to_first_break: int = 0
break_magnitude: float = 0.0
returned_to_channel: bool = False
bounces_after_return: int = 0
permanent_break_direction: int = -1
bars_to_permanent_break: int = -1
durability_score: float = 0.0

# SPY break dynamics (same fields with spy_ prefix)
spy_bars_to_first_break: int = 0
spy_break_direction: int = 0
# ... etc
```

### Source Channel Parameters (Inspector Visualization)
```python
source_channel_slope: float = 0.0
source_channel_intercept: float = 0.0
source_channel_std_dev: float = 0.0
source_channel_r_squared: float = 0.0
source_channel_direction: int = -1     # 0=BEAR, 1=SIDE, 2=BULL
source_channel_bounce_count: int = 0

# SPY source channel (same fields with spy_ prefix)
spy_source_channel_slope: float = 0.0
# ... etc
```

### Exit Events (Lists)
```python
exit_bars: List[int] = []            # Bar indices when each exit occurred
exit_magnitudes: List[float] = []    # Magnitude of each exit
exit_durations: List[int] = []       # Bars outside before return
exit_types: List[int] = []           # 0=lower, 1=upper
exit_returned: List[bool] = []       # Whether each exit returned

# SPY exit events (same fields with spy_ prefix)
spy_exit_bars: List[int] = []
# ... etc
```

### Validity Flags
```python
duration_valid: bool = False
direction_valid: bool = False
break_scan_valid: bool = False
```

### RSI Labels (Phase 7)
```python
# TSLA RSI
rsi_at_first_break: float = 50.0
rsi_at_permanent_break: float = 50.0
rsi_at_channel_end: float = 50.0
rsi_overbought_at_break: bool = False
rsi_oversold_at_break: bool = False

# SPY RSI (same fields with spy_ prefix)
spy_rsi_at_first_break: float = 50.0
# ... etc
```

### Next Channel Labels (Phase 6)
```python
best_next_channel_direction: int = -1
best_next_channel_bars_away: int = -1
best_next_channel_bounce_count: int = 0
# ... etc (SPY versions with spy_ prefix)
```

---

## Inspector Usage Patterns

### How Inspector Loads Samples (inspector.py lines 883-886)
```python
print(f"Loading samples from {samples_path}...")
with open(samples_path, 'rb') as f:
    samples = pickle.load(f)
print(f"Loaded {len(samples)} samples")
```

### How Inspector Accesses Sample Data (inspector.py lines 36-37, 228-232)
```python
def __init__(self, samples: List[ChannelSample], tsla_df: pd.DataFrame, ...):
    self.samples = samples
    self.tsla_df = tsla_df
    
# Later in _draw_panel:
sample = self.samples[self.current_idx]
sample_ts = sample.timestamp
channel_end_idx = sample.channel_end_idx

# Access labels:
if window in sample.labels_per_window:
    if 'tsla' in sample.labels_per_window[window]:
        labels = sample.labels_per_window[window]['tsla'].get(tf)
```

### Critical Inspector Requirements
1. **Sample must be picklable**: Inspector loads via `pickle.load()`
2. **Must be a list**: `samples = pickle.load(f)` expects `List[ChannelSample]`
3. **Must have correct field names**: Inspector accesses via attribute names
4. **Timestamps must be pandas.Timestamp**: Inspector uses `.index.searchsorted()` and `.index.get_loc()`
5. **Labels must be ChannelLabels dataclass instances**: Inspector accesses via `getattr()`

---

## Dataset Usage Patterns

### How Dataset Loads Samples (dataset.py lines 1207-1240)
```python
def load_samples(path: str) -> List[ChannelSample]:
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"Samples file not found: {path}")
    
    with open(path, 'rb') as f:
        samples = pickle.load(f)
    
    if not isinstance(samples, list):
        raise DataLoadError(f"Expected list of samples, got {type(samples)}")
    
    # Validate first sample
    if samples:
        first = samples[0]
        if not hasattr(first, 'tf_features') or not hasattr(first, 'labels_per_window'):
            raise DataLoadError(
                f"Samples missing required attributes. "
                f"Expected ChannelSample with tf_features and labels_per_window"
            )
    
    return samples
```

### How Dataset Extracts Features (dataset.py lines 133-145)
```python
for sample in self.samples:
    # Get tf_features dict directly
    if not sample.tf_features:
        raise ValidationError(f"Sample at {sample.timestamp} has empty tf_features")
    
    # Convert dict to ordered array
    if self.feature_names is None:
        self.feature_names = sorted(sample.tf_features.keys())
    
    feature_array = np.array([
        sample.tf_features.get(name, 0.0) for name in self.feature_names
    ], dtype=np.float32)
    
    features_list.append(feature_array)
```

### How Dataset Extracts Labels (dataset.py lines 207-487)
```python
def _extract_labels(self, sample: ChannelSample, tf: str, window: int) -> Dict[str, Any]:
    # Get labels from structure
    window_labels = sample.labels_per_window.get(window, {})
    
    # Handle both old and new structure
    if 'tsla' in window_labels:
        # New structure
        tsla_labels_dict = window_labels.get('tsla', {})
        spy_labels_dict = window_labels.get('spy', {})
        tf_labels = tsla_labels_dict.get(tf)
        spy_tf_labels = spy_labels_dict.get(tf)
    else:
        # Old structure
        tf_labels = window_labels.get(tf)
        spy_tf_labels = None
    
    # Extract all label fields using getattr()
    labels = {
        'tsla_bars_to_first_break': getattr(tf_labels, 'bars_to_first_break', 0),
        'tsla_break_direction': getattr(tf_labels, 'break_direction', 0),
        # ... ~100+ label fields
    }
```

---

## C++ Loader Requirements

### Pickle Compatibility
The C++ loader must create Python objects that are pickle-compatible:
1. **Use pybind11**: Define ChannelSample and ChannelLabels as Python classes
2. **Register with pickle**: Implement `__getstate__` and `__setstate__` or use pybind11's pickle support
3. **Match Python types exactly**: 
   - `timestamp` -> `pd.Timestamp` (convert int64_t nanoseconds)
   - `tf_features` -> `dict[str, float]`
   - `labels_per_window` -> `dict[int, dict[str, dict[str, ChannelLabels]]]`

### Field Access
Inspector and dataset use both dictionary access and attribute access:
```python
# Dictionary access
sample.tf_features['5min_price_pct_0']
sample.labels_per_window[50]['tsla']['daily']

# Attribute access
sample.timestamp
sample.channel_end_idx
labels.bars_to_first_break
```

### Type Conversions
```python
# Timestamp: C++ int64_t -> pd.Timestamp
timestamp = pd.Timestamp(nanoseconds_since_epoch)

# Features: C++ flat array -> Python dict
tf_features = {feature_names[i]: values[i] for i in range(len(feature_names))}

# Labels: C++ struct -> Python dataclass
labels = ChannelLabels(
    duration_bars=cpp_labels.duration_bars,
    break_direction=cpp_labels.break_direction,
    # ... all fields
)
```

---

## Validation Checklist

Before using C++ loaded samples with Python code, verify:

- [ ] Samples are a Python list: `isinstance(samples, list)`
- [ ] Each sample is a ChannelSample: `isinstance(sample, ChannelSample)`
- [ ] Timestamp is pd.Timestamp: `isinstance(sample.timestamp, pd.Timestamp)`
- [ ] Features is dict: `isinstance(sample.tf_features, dict)`
- [ ] Feature keys are strings: `all(isinstance(k, str) for k in sample.tf_features.keys())`
- [ ] Feature values are floats: `all(isinstance(v, float) for v in sample.tf_features.values())`
- [ ] Labels structure is correct: `isinstance(sample.labels_per_window, dict)`
- [ ] Labels are ChannelLabels: Check nested structure
- [ ] Sample is picklable: `pickle.dumps(sample)` succeeds
- [ ] Feature count matches: `len(sample.tf_features) == TOTAL_FEATURES`

---

## Example Sample Creation (Python)

```python
import pandas as pd
from v15.dtypes import ChannelSample, ChannelLabels

# Create labels
tsla_labels = ChannelLabels(
    duration_bars=25,
    break_direction=1,
    break_magnitude=2.3,
    bars_to_first_break=5,
    returned_to_channel=True,
    source_channel_r_squared=0.85,
    source_channel_direction=2,
    break_scan_valid=True,
    timeframe='daily'
)

# Create sample
sample = ChannelSample(
    timestamp=pd.Timestamp('2024-01-15 14:30:00'),
    channel_end_idx=15432,
    tf_features={
        '5min_price_pct_0': 0.0234,
        '5min_volume_zscore_0': 1.45,
        # ... ~7,880 features
    },
    labels_per_window={
        50: {  # window
            'tsla': {
                'daily': tsla_labels,
                # ... other timeframes
            },
            'spy': {
                # ... SPY labels
            }
        },
        # ... other windows
    },
    bar_metadata={
        '5min': {
            'bar_completion_pct': 1.0,
            'total_bars': 245,
            'is_partial': False
        },
        # ... other timeframes
    },
    best_window=50
)

# Verify picklable
import pickle
pickled = pickle.dumps(sample)
loaded = pickle.loads(pickled)
assert loaded.timestamp == sample.timestamp
```

---

## Summary

The C++ loader must create `List[ChannelSample]` where each sample:
1. Has exact field names matching the dataclass
2. Uses correct Python types (pd.Timestamp, dict, ChannelLabels)
3. Is picklable (for saving/loading)
4. Supports both attribute and dictionary access
5. Matches the nested structure for labels_per_window

The key challenge is creating proper Python objects from C++ that are indistinguishable from Python-created samples.
