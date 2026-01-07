# Window Selection Strategy Framework - Quick Reference

**Version**: v10.0.0 | **Location**: `v7/core/window_strategy.py` | **Tests**: 41/41 ✅

---

## Import

```python
from v7.core.window_strategy import (
    get_strategy,                      # Factory function
    SelectionStrategy,                 # Strategy enum
    select_best_window_bounce_first,   # Convenience: bounce-first
    select_best_window_by_labels,      # Convenience: label validity
    select_best_window_balanced,       # Convenience: balanced
)
```

---

## Quick Start

```python
# 1. Get a strategy
strategy = get_strategy(SelectionStrategy.BOUNCE_FIRST)

# 2. Select best window
best_window, confidence = strategy.select_window(channels, labels_per_window)

# 3. Use the result
if confidence >= 0.7:
    best_channel = channels[best_window]
```

---

## Available Strategies

| Strategy | Selection Criteria | Best For |
|----------|-------------------|----------|
| **BOUNCE_FIRST** | max(bounce_count) → r² | Production (v7-v9 compat) |
| **LABEL_VALIDITY** | max(valid_labels) | Research, data quality |
| **BALANCED_SCORE** | 0.4×bounce + 0.6×labels | Production (optimized) |
| **QUALITY_SCORE** | max(quality_score) | Simple quality-based |

---

## Common Patterns

### Pattern 1: Basic Selection
```python
strategy = get_strategy(SelectionStrategy.BOUNCE_FIRST)
best_window, confidence = strategy.select_window(channels)
```

### Pattern 2: With Labels
```python
strategy = get_strategy(SelectionStrategy.BALANCED_SCORE)
best_window, confidence = strategy.select_window(channels, labels_per_window)
```

### Pattern 3: Custom Weights
```python
strategy = get_strategy(
    SelectionStrategy.BALANCED_SCORE,
    bounce_weight=0.3,
    label_weight=0.7
)
best_window, confidence = strategy.select_window(channels, labels_per_window)
```

### Pattern 4: Confidence Filter
```python
MIN_CONFIDENCE = 0.7

best_window, confidence = strategy.select_window(channels, labels_per_window)
if confidence >= MIN_CONFIDENCE:
    # Use this sample
    process_sample(channels[best_window])
else:
    # Skip or use fallback
    logger.warning(f"Low confidence: {confidence:.3f}")
```

### Pattern 5: Fallback Strategy
```python
# Try primary strategy
strategy = get_strategy(SelectionStrategy.BALANCED_SCORE)
best_window, confidence = strategy.select_window(channels, labels_per_window)

# Fall back if low confidence
if confidence < 0.7:
    fallback = get_strategy(SelectionStrategy.BOUNCE_FIRST)
    best_window, confidence = fallback.select_window(channels)
```

---

## Convenience Functions

### Bounce-First (v7-v9 Default)
```python
best_window, confidence = select_best_window_bounce_first(channels)
```

### Label Validity
```python
# With channels
best_window, confidence = select_best_window_by_labels(labels_per_window, channels)

# Without channels (creates dummy channels)
best_window, confidence = select_best_window_by_labels(labels_per_window)
```

### Balanced Score
```python
best_window, confidence = select_best_window_balanced(
    channels, labels_per_window,
    bounce_weight=0.4,
    label_weight=0.6
)
```

---

## Confidence Scores

| Confidence | Meaning | Action |
|------------|---------|--------|
| **1.0** | Clear winner | Use with high confidence |
| **0.7-0.9** | Moderate confidence | Use, but monitor |
| **0.5-0.6** | Close tie | Consider fallback |
| **0.0** | No valid windows | Skip sample |

---

## Edge Cases

All strategies handle these automatically:

```python
# Empty channels
strategy.select_window({})
# Returns: (None, 0.0)

# All invalid channels
strategy.select_window(invalid_channels)
# Returns: (None, 0.0)

# No labels (LabelValidity falls back to BounceFirst)
label_strategy.select_window(channels, None)
# Returns: bounce-first result

# All labels None
strategy.select_window(channels, all_none_labels)
# Returns: (None, 0.0)
```

---

## Custom Strategy

```python
from dataclasses import dataclass

@dataclass
class MyCustomStrategy:
    param: float = 1.0

    def select_window(self, channels, labels_per_window=None):
        if not channels:
            return None, 0.0

        # Your logic here
        best_window = min(channels.keys())
        confidence = 0.8

        return best_window, confidence

# Register
from v7.core.window_strategy import register_strategy
register_strategy("my_custom", MyCustomStrategy)

# Use
strategy = get_strategy(SelectionStrategy.MY_CUSTOM, param=2.0)
```

---

## Backward Compatibility

### Old API (v7-v9)
```python
from v7.core.channel import select_best_channel
best_channel, best_window = select_best_channel(channels)
```

### New API (v10+)
```python
# Option 1: Compatibility wrapper
from v7.core.window_strategy import select_best_channel_v7
best_channel, best_window = select_best_channel_v7(channels)

# Option 2: New API (recommended)
from v7.core.window_strategy import select_best_window_bounce_first
best_window, confidence = select_best_window_bounce_first(channels)
best_channel = channels[best_window]
```

---

## Production Config

```python
# config.py
WINDOW_SELECTION_STRATEGY = SelectionStrategy.BALANCED_SCORE
WINDOW_SELECTION_PARAMS = {'bounce_weight': 0.4, 'label_weight': 0.6}
MIN_SELECTION_CONFIDENCE = 0.7
FALLBACK_STRATEGY = SelectionStrategy.BOUNCE_FIRST

# usage.py
from config import *

def select_window_for_sample(channels, labels_per_window):
    # Primary strategy
    strategy = get_strategy(WINDOW_SELECTION_STRATEGY, **WINDOW_SELECTION_PARAMS)
    best_window, confidence = strategy.select_window(channels, labels_per_window)

    # Confidence filter
    if confidence < MIN_SELECTION_CONFIDENCE:
        # Try fallback
        fallback = get_strategy(FALLBACK_STRATEGY)
        fb_window, fb_conf = fallback.select_window(channels, labels_per_window)
        if fb_conf > confidence:
            best_window, confidence = fb_window, fb_conf

    return best_window, confidence
```

---

## Testing

```bash
# Run all tests
pytest v7/core/test_window_strategy.py -v

# Run specific test
pytest v7/core/test_window_strategy.py::test_bounce_first_selects_most_bounces -v

# Run with coverage
pytest v7/core/test_window_strategy.py --cov=v7.core.window_strategy
```

---

## Common Issues

### Issue: Returns (None, 0.0)
**Cause**: No valid channels or all labels None
**Fix**: Check channel validity and label completeness

### Issue: Unexpected window selected
**Cause**: Strategy weights may not match expectations
**Fix**: Debug with logging:
```python
for w, ch in channels.items():
    print(f"Window {w}: bounces={ch.bounce_count}, r²={ch.r_squared:.3f}")
```

### Issue: Low confidence
**Cause**: Multiple windows have similar scores (genuine tie)
**Fix**: This is expected - consider using fallback or accepting the result

---

## Performance

- **Complexity**: O(n) where n = windows (typically 8)
- **Memory**: Minimal overhead
- **Thread-safe**: Yes
- **Execution time**: <1ms per selection

---

## Files

| File | Purpose | Lines |
|------|---------|-------|
| `window_strategy.py` | Core implementation | 794 |
| `test_window_strategy.py` | Test suite | 723 |
| `WINDOW_STRATEGY_GUIDE.md` | Full documentation | 630 |
| `WINDOW_STRATEGY_SUMMARY.md` | Implementation summary | 400+ |
| `window_strategy_example.py` | Usage examples | 498 |
| `WINDOW_STRATEGY_QUICKREF.md` | This file | ~200 |

---

## More Information

- **Full Guide**: `v7/core/WINDOW_STRATEGY_GUIDE.md`
- **Implementation Summary**: `v7/core/WINDOW_STRATEGY_SUMMARY.md`
- **Examples**: `v7/core/window_strategy_example.py`
- **Tests**: `v7/core/test_window_strategy.py`

---

**Last Updated**: 2026-01-06 | **Version**: v10.0.0 | **Status**: Production-Ready ✅
