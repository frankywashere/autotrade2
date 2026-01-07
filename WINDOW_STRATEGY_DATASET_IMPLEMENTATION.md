# Window Selection Strategy Implementation in ChannelDataset

**Version**: v11.0.0
**Date**: 2026-01-06
**Author**: Claude Sonnet 4.5

## Overview

Successfully implemented window selection strategy support in `v7/training/dataset.py`, enabling the ChannelDataset class to dynamically select the best window using configurable strategies from the window selection framework.

## Implementation Summary

### 1. Added Strategy Parameter to `__init__()`

**Location**: `ChannelDataset.__init__()` (lines 439-477)

**Changes**:
- Added `strategy` parameter (default: `"bounce_first"`)
- Added `**strategy_kwargs` for custom strategy parameters
- Implemented string-to-enum conversion for strategy names
- Initialized `self.strategy` using `get_strategy()` factory

**Supported Strategies**:
- `"bounce_first"`: Use sample.best_window (v7-v9 default behavior)
- `"label_validity"`: Select window with most valid TF labels
- `"balanced_score"`: Weighted combination (40% bounce, 60% labels)
- `"quality_score"`: Use pre-computed channel.quality_score

**Example Usage**:
```python
# Default bounce-first strategy
dataset = ChannelDataset(samples)

# Label validity strategy
dataset = ChannelDataset(samples, strategy="label_validity")

# Balanced strategy with custom weights
dataset = ChannelDataset(
    samples,
    strategy="balanced_score",
    bounce_weight=0.3,
    label_weight=0.7
)
```

### 2. Implemented `_is_window_valid()` Method

**Location**: Lines 486-522

**Purpose**: Validate that a window has:
- Valid channel (`channel.valid == True`)
- At least one valid label across all timeframes

**Signature**:
```python
def _is_window_valid(
    self,
    window_size: int,
    channels: Dict[int, Channel],
    labels_per_window: Optional[Dict[int, Dict[str, ChannelLabels]]]
) -> bool
```

**Returns**: `True` if window has valid channel and labels, `False` otherwise

### 3. Implemented `_select_window()` Method

**Location**: Lines 524-611

**Purpose**: Select the best window using the configured strategy with full backward compatibility.

**Signature**:
```python
def _select_window(
    self,
    sample: ChannelSample
) -> Tuple[Optional[int], Optional[Channel], Optional[Dict[str, ChannelLabels]]]
```

**Returns**: `(window_size, channel, labels_per_tf)` or `(None, None, None)` if no valid window

**Key Features**:
- **Backward Compatibility**: Handles legacy single-window samples (v7-v9 caches)
- **Multi-Window Support**: Uses strategy framework for multi-window samples (v10+)
- **Fallback Logic**: Gracefully falls back to `sample.best_window` if strategy fails
- **Validation**: Filters to valid windows before applying strategy
- **Error Handling**: Catches and warns on strategy failures

**Logic Flow**:
```
1. Check if sample.channels is None (old format)
   └─> Yes: Return (best_window, sample.channel, sample.labels)
   └─> No: Continue to multi-window logic

2. Filter channels to valid windows only

3. Apply strategy.select_window() to get best window

4. Validate selection and apply fallbacks if needed

5. Return (selected_window, selected_channel, selected_labels)
```

### 4. Modified `__getitem__()` Method

**Location**: Lines 613-842

**Changes**:

#### 4.1 Window Selection (lines 652-662)
Added window selection at the beginning of `__getitem__()`:
```python
# Select the best window using the configured strategy
selected_window_size, selected_channel, selected_labels = self._select_window(sample)

# Fallback if selection failed
if selected_window_size is None or selected_labels is None:
    selected_labels = sample.labels if sample.labels else {}
    selected_window_size = sample.best_window if sample.best_window else STANDARD_WINDOWS[0]
```

#### 4.2 Label Extraction (lines 677-750)
Modified to use `selected_labels` instead of `sample.labels`:
```python
# OLD:
for tf in TIMEFRAMES:
    tf_labels = sample.labels.get(tf)

# NEW:
for tf in TIMEFRAMES:
    tf_labels = selected_labels.get(tf)
```

#### 4.3 Multi-Window Labels (lines 792-800, 835-836)
Added `selected_window` index to labels_dict:
```python
# New field in labels_dict (v11.0.0)
labels_dict['selected_window'] = torch.tensor(selected_window_idx, dtype=torch.long)
```

**New labels_dict Structure**:
```python
{
    # Existing per-TF labels [11]
    'duration': ...,
    'direction': ...,
    'next_channel': ...,
    'trigger_tf': ...,

    # Existing validity masks [11]
    'duration_valid': ...,
    'direction_valid': ...,
    'next_channel_valid': ...,
    'trigger_tf_valid': ...,

    # Existing aggregate labels
    'duration_bars': ...,
    'permanent_break': ...,

    # Existing multi-window labels
    'window_scores': ...,   # [num_windows, 3]
    'window_valid': ...,    # [num_windows]
    'best_window': ...,     # scalar (0-7)

    # NEW: Selected window (v11.0.0)
    'selected_window': ...  # scalar (0-7) - window chosen by strategy
}
```

### 5. Updated `collate_fn()` Function

**Location**: Line 1377

**Changes**: Added batching for `selected_window` field:
```python
'selected_window': torch.stack([l['selected_window'] for l in labels_list]),  # v11.0.0
```

## Backward Compatibility

### v10.0.0 Caches (Single Window)
- Automatically detected via `sample.channels is None`
- Uses `sample.channel` and `sample.labels` directly
- No strategy selection needed (already pre-selected)
- `selected_window == best_window` for these samples

### v11.0.0+ Caches (Multi-Window)
- Uses full strategy framework
- Selects window dynamically based on strategy
- `selected_window` may differ from `best_window`

## Testing

Comprehensive test suite in `test_dataset_strategy_simple.py`:

### Test Results
```
Test 1: Old Cache Format (Single Channel) ................... ✓ PASSED
Test 2: Multi-Window Bounce-First Strategy .................. ✓ PASSED
Test 3: Multi-Window Label Validity Strategy ................ ✓ PASSED
Test 4: Multi-Window Balanced Score Strategy ................ ✓ PASSED
Test 5: Custom Strategy Weights ............................. ✓ PASSED
```

### Test Coverage
- ✓ Old cache format (v10.0.0) compatibility
- ✓ Multi-window bounce-first strategy
- ✓ Multi-window label validity strategy
- ✓ Multi-window balanced score strategy
- ✓ Custom strategy weights
- ✓ Fallback logic for invalid selections
- ✓ `selected_window` field in labels_dict

## Usage Examples

### Example 1: Default Bounce-First Strategy
```python
from v7.training.dataset import prepare_dataset_from_scratch

train_samples, val_samples, test_samples = prepare_dataset_from_scratch(
    data_dir=data_dir,
    cache_dir=cache_dir,
    window=50,
    step=25
)

# Create dataset with default strategy
train_dataset = ChannelDataset(train_samples)  # Uses "bounce_first"
```

### Example 2: Label Validity Strategy
```python
# Prioritize label completeness over bounce quality
dataset = ChannelDataset(
    train_samples,
    strategy="label_validity"
)
```

### Example 3: Balanced Strategy with Custom Weights
```python
# 70% weight to bounces, 30% to labels
dataset = ChannelDataset(
    train_samples,
    strategy="balanced_score",
    bounce_weight=0.7,
    label_weight=0.3
)
```

### Example 4: Dynamic Strategy Selection
```python
def create_dataset_for_experiment(samples, experiment_config):
    """Create dataset with strategy from config."""
    strategy = experiment_config.get('window_strategy', 'bounce_first')
    strategy_params = experiment_config.get('strategy_params', {})

    return ChannelDataset(
        samples,
        strategy=strategy,
        **strategy_params
    )
```

## Integration with Training

The implementation is transparent to training code:

```python
# Training loop - no changes needed
train_loader, val_loader, test_loader = create_dataloaders(
    train_samples, val_samples, test_samples,
    batch_size=32
)

for features, labels in train_loader:
    # labels['selected_window'] tells you which window was used
    # labels['best_window'] is the original best window from cache

    # Model can learn from both:
    # 1. Which window was selected (labels['selected_window'])
    # 2. What the "ground truth" best window was (labels['best_window'])

    outputs = model(features)
    loss = criterion(outputs, labels)
```

## Cache Version Update

Updated `CACHE_VERSION` to `"v11.0.0"`:
```python
# v11.0.0: Multi-window cache architecture with per-window feature extraction.
#          Supports both single-window (v10.0.0 compatible) and multi-window modes.
CACHE_VERSION = "v11.0.0"

# Backward compatible with:
COMPATIBLE_CACHE_VERSIONS = ["v10.0.0", "v9.0.0", "v8.0.0", "v7.3.0", "v7.2.0", "v7.1.0"]
```

## Benefits

1. **Flexibility**: Easily switch between different window selection strategies
2. **Experimentation**: Test different strategies without regenerating caches
3. **Backward Compatibility**: Works seamlessly with old single-window caches
4. **Extensibility**: Add custom strategies via the framework
5. **Transparency**: `selected_window` field tracks which window was used
6. **Performance**: Strategy selection is fast (O(n) where n = # windows, typically 8)

## Future Enhancements

Potential improvements:
1. **Per-sample strategies**: Allow different samples to use different strategies
2. **Confidence filtering**: Skip samples where strategy confidence is low
3. **Strategy logging**: Track which strategy/window was used for analysis
4. **Ensemble strategies**: Combine multiple strategies with voting
5. **Adaptive strategies**: Learn optimal strategy from model performance

## Files Modified

- `/Users/frank/Desktop/CodingProjects/x6/v7/training/dataset.py`
  - Added imports for window strategy framework
  - Modified `ChannelDataset.__init__()`
  - Added `_is_window_valid()` method
  - Added `_select_window()` method
  - Modified `__getitem__()` method
  - Updated `collate_fn()` function
  - Updated `CACHE_VERSION` to v11.0.0

## Files Created

- `/Users/frank/Desktop/CodingProjects/x6/test_dataset_strategy_simple.py`
  - Comprehensive test suite for strategy functionality
  - 5 test cases covering all major scenarios
  - All tests passing

## Dependencies

- `v7.core.window_strategy` (SelectionStrategy, get_strategy, WindowSelectionStrategy)
- Existing v7 core modules (channel, timeframe, labels, features)

## Conclusion

Window selection strategy support has been successfully implemented in the ChannelDataset class with:
- ✅ Full backward compatibility with v10.0.0 caches
- ✅ Support for all 4 built-in strategies
- ✅ Custom strategy parameters
- ✅ Comprehensive test coverage (100% passing)
- ✅ Clean, well-documented code
- ✅ Transparent integration with existing training pipeline

The implementation is production-ready and can be used immediately in training experiments.
