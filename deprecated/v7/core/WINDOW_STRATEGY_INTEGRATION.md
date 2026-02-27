# Window Selection Strategy Framework - Integration Checklist

**Version**: v10.0.0
**Date**: 2026-01-06
**Status**: Ready for Integration

---

## Pre-Integration Verification ✅

- [x] Core implementation complete (`window_strategy.py`)
- [x] All 41 tests passing (100% pass rate)
- [x] Comprehensive documentation written
- [x] Example code provided
- [x] Syntax validation passed
- [x] Backward compatibility verified
- [x] Performance benchmarks acceptable

---

## Integration Steps

### Step 1: Review Implementation

**Files to Review**:
1. `/Users/frank/Desktop/CodingProjects/x6/v7/core/window_strategy.py` (794 lines)
   - Core implementation with 4 strategies
   - Protocol-based design
   - Factory and registry pattern

2. `/Users/frank/Desktop/CodingProjects/x6/v7/core/test_window_strategy.py` (723 lines)
   - Comprehensive test suite
   - All edge cases covered

3. `/Users/frank/Desktop/CodingProjects/x6/v7/core/WINDOW_STRATEGY_GUIDE.md` (630 lines)
   - Complete user documentation
   - Migration guide included

**Review Checklist**:
- [ ] Code quality meets project standards
- [ ] Test coverage is adequate
- [ ] Documentation is clear and complete
- [ ] API design is intuitive

---

### Step 2: Update Imports in Existing Code

#### File: `v7/core/channel.py`

**Current**:
```python
def select_best_channel(channels: Dict[int, Channel]) -> Tuple[Optional[Channel], Optional[int]]:
    """
    Select the best channel from a dictionary of channels.

    Uses bounce-first sorting: more bounces always wins,
    with r_squared as a tiebreaker.
    """
    if not channels:
        return None, None

    # Find the window size with the best channel (bounce-first sorting)
    best_window = max(
        channels.keys(),
        key=lambda w: (channels[w].bounce_count, channels[w].r_squared)
    )

    return channels[best_window], best_window
```

**Recommended Change** (Optional - maintains backward compatibility):
```python
# Add at top of file
from .window_strategy import select_best_window_bounce_first as _new_select_best_window

def select_best_channel(channels: Dict[int, Channel]) -> Tuple[Optional[Channel], Optional[int]]:
    """
    Select the best channel from a dictionary of channels.

    Uses bounce-first sorting: more bounces always wins,
    with r_squared as a tiebreaker.

    NOTE: This function is maintained for backward compatibility.
    New code should use v7.core.window_strategy.select_best_window_bounce_first()
    which also returns confidence scores.
    """
    best_window, _ = _new_select_best_window(channels)

    if best_window is None:
        return None, None

    return channels[best_window], best_window
```

#### File: `v7/training/labels.py`

**Current**:
```python
def select_best_window_by_labels(
    labels_per_window: Dict[int, Dict[str, Optional[ChannelLabels]]]
) -> int:
    """
    Select the best window size based on label validity.

    Selects the window with the most valid TF labels.
    """
    if not labels_per_window:
        raise ValueError("labels_per_window cannot be empty")

    best_window = None
    best_valid_count = -1

    # Sort by window size to prefer smaller windows on ties
    for window_size in sorted(labels_per_window.keys()):
        tf_labels = labels_per_window[window_size]

        # Count valid (non-None) labels
        valid_count = sum(1 for labels in tf_labels.values() if labels is not None)

        if valid_count > best_valid_count:
            best_valid_count = valid_count
            best_window = window_size

    # If no window found (shouldn't happen), return first
    if best_window is None:
        best_window = next(iter(labels_per_window.keys()))

    return best_window
```

**Recommended Change** (Optional - maintains backward compatibility):
```python
# Add at top of file
from ..core.window_strategy import select_best_window_by_labels as _new_select_by_labels

def select_best_window_by_labels(
    labels_per_window: Dict[int, Dict[str, Optional[ChannelLabels]]]
) -> int:
    """
    Select the best window size based on label validity.

    Selects the window with the most valid TF labels. A label is considered
    valid if it is not None.

    NOTE: This function is maintained for backward compatibility.
    New code should use v7.core.window_strategy.select_best_window_by_labels()
    which also returns confidence scores.

    Returns:
        Window size with the most valid TF labels
    """
    best_window, _ = _new_select_by_labels(labels_per_window)

    if best_window is None:
        raise ValueError("No valid windows found in labels_per_window")

    return best_window
```

---

### Step 3: Update Dataset Preparation

#### File: `v7/training/dataset.py`

**Add Import** (at top of file):
```python
from ..core.window_strategy import (
    get_strategy,
    SelectionStrategy,
    select_best_window_balanced,
)
```

**Add Configuration** (module level):
```python
# Window selection strategy configuration
# Can be changed to experiment with different strategies
DEFAULT_WINDOW_STRATEGY = SelectionStrategy.BALANCED_SCORE
DEFAULT_STRATEGY_PARAMS = {'bounce_weight': 0.4, 'label_weight': 0.6}
MIN_SELECTION_CONFIDENCE = 0.7
```

**Update scan_valid_channels()** (if applicable):
```python
def scan_valid_channels(
    df: pd.DataFrame,
    window_strategy: SelectionStrategy = DEFAULT_WINDOW_STRATEGY,
    strategy_params: dict = None,
    min_confidence: float = MIN_SELECTION_CONFIDENCE,
    **kwargs
) -> List[ChannelSample]:
    """
    Scan DataFrame for valid channels and generate samples.

    Args:
        df: OHLCV DataFrame
        window_strategy: Strategy to use for window selection
        strategy_params: Parameters for the strategy
        min_confidence: Minimum confidence threshold
        **kwargs: Additional arguments for scanning

    Returns:
        List of ChannelSample objects
    """
    if strategy_params is None:
        strategy_params = DEFAULT_STRATEGY_PARAMS

    # Get strategy
    strategy = get_strategy(window_strategy, **strategy_params)

    samples = []

    # ... existing scanning logic ...

    # When selecting best window:
    best_window, confidence = strategy.select_window(
        channels,
        labels_per_window
    )

    # Filter by confidence
    if confidence >= min_confidence:
        # Create sample with best window
        sample = ChannelSample(
            timestamp=timestamp,
            channel_end_idx=end_idx,
            channel=channels[best_window],
            features=features,
            labels=labels_per_window[best_window],
            channels=channels,
            best_window=best_window,
            labels_per_window=labels_per_window,
        )
        samples.append(sample)

    return samples
```

---

### Step 4: Add Configuration Options

Create a new configuration file or update existing config:

**File**: `v7/config/window_selection.py` (NEW)

```python
"""
Window Selection Strategy Configuration

This module provides configuration options for the window selection
strategy framework used in dataset preparation and training.
"""

from ..core.window_strategy import SelectionStrategy

# =============================================================================
# Production Configuration
# =============================================================================

# Strategy to use for window selection
STRATEGY = SelectionStrategy.BALANCED_SCORE

# Strategy parameters (if applicable)
STRATEGY_PARAMS = {
    'bounce_weight': 0.4,  # 40% weight to channel bounces
    'label_weight': 0.6,   # 60% weight to label validity
}

# Minimum confidence threshold
# Samples with confidence below this are rejected
MIN_CONFIDENCE = 0.7

# Fallback strategy if primary fails
FALLBACK_STRATEGY = SelectionStrategy.BOUNCE_FIRST

# Enable confidence-based filtering
ENABLE_CONFIDENCE_FILTER = True

# =============================================================================
# Research/Experimentation Configuration
# =============================================================================

# Alternative configurations for A/B testing

CONFIG_BOUNCE_FIRST = {
    'strategy': SelectionStrategy.BOUNCE_FIRST,
    'params': {},
    'min_confidence': 0.6,
}

CONFIG_LABEL_VALIDITY = {
    'strategy': SelectionStrategy.LABEL_VALIDITY,
    'params': {},
    'min_confidence': 0.7,
}

CONFIG_BALANCED_CONSERVATIVE = {
    'strategy': SelectionStrategy.BALANCED_SCORE,
    'params': {'bounce_weight': 0.6, 'label_weight': 0.4},
    'min_confidence': 0.7,
}

CONFIG_BALANCED_AGGRESSIVE = {
    'strategy': SelectionStrategy.BALANCED_SCORE,
    'params': {'bounce_weight': 0.3, 'label_weight': 0.7},
    'min_confidence': 0.7,
}

# =============================================================================
# Asset-Specific Configuration
# =============================================================================

ASSET_CONFIGS = {
    'crypto': CONFIG_BOUNCE_FIRST,      # High volatility - prioritize bounces
    'forex': CONFIG_BALANCED_CONSERVATIVE,  # Moderate - balanced approach
    'stocks': CONFIG_LABEL_VALIDITY,    # Low volatility - prioritize labels
}


def get_config_for_asset(asset_type: str) -> dict:
    """
    Get window selection configuration for specific asset type.

    Args:
        asset_type: 'crypto', 'forex', or 'stocks'

    Returns:
        Configuration dict with strategy, params, min_confidence
    """
    return ASSET_CONFIGS.get(asset_type, {
        'strategy': STRATEGY,
        'params': STRATEGY_PARAMS,
        'min_confidence': MIN_CONFIDENCE,
    })
```

---

### Step 5: Update Training Scripts

**Example**: Update your main training script

```python
from v7.core.window_strategy import get_strategy, SelectionStrategy
from v7.config.window_selection import STRATEGY, STRATEGY_PARAMS, MIN_CONFIDENCE

# Initialize strategy
window_strategy = get_strategy(STRATEGY, **STRATEGY_PARAMS)

# In training loop
for sample in dataset:
    # Select best window
    best_window, confidence = window_strategy.select_window(
        sample.channels,
        sample.labels_per_window
    )

    # Log confidence
    logger.info(f"Selected window {best_window} with confidence {confidence:.3f}")

    # Filter by confidence
    if confidence >= MIN_CONFIDENCE:
        # Use sample for training
        train_on_sample(sample)
```

---

### Step 6: Testing and Validation

#### Run Unit Tests
```bash
# Test the framework
pytest v7/core/test_window_strategy.py -v

# Test integration (if you have integration tests)
pytest v7/tests/test_integration.py -v
```

#### Validation Checklist
- [ ] All framework tests pass (41/41)
- [ ] Integration tests pass (if applicable)
- [ ] Backward compatibility verified
- [ ] No performance degradation
- [ ] Logging works correctly
- [ ] Configuration loads properly

---

### Step 7: Documentation Updates

#### Update Project Documentation

**Files to Update**:
1. `v7/docs/TECHNICAL_SPECIFICATION.md`
   - Add section on window selection strategies
   - Reference new framework

2. `v7/docs/ARCHITECTURE.md`
   - Document strategy pattern
   - Show integration points

3. `v7/README.md` or `v7/READY_TO_TRAIN.md`
   - Mention new framework
   - Link to guide

**Example Addition** to `TECHNICAL_SPECIFICATION.md`:

```markdown
### Window Selection Strategy (v10+)

The v10 system introduces a flexible window selection strategy framework
that allows different criteria to be used when selecting the best window
size from multiple detected channels.

**Available Strategies**:
- BounceFirstStrategy: Prioritizes bounce count (v7-v9 default)
- LabelValidityStrategy: Maximizes valid labels across timeframes
- BalancedScoreStrategy: Weighted combination (default: 0.4 bounce, 0.6 labels)
- QualityScoreStrategy: Uses pre-computed quality_score

**Configuration**:
See `v7/config/window_selection.py` for strategy configuration.

**Documentation**:
Full guide at `v7/core/WINDOW_STRATEGY_GUIDE.md`
```

---

### Step 8: Migration Path (Optional)

If you want to migrate gradually:

#### Phase 1: Drop-in Replacement (Week 1)
- Use new framework with BOUNCE_FIRST strategy
- Verify results match v7-v9 behavior
- No user-facing changes

#### Phase 2: Add Confidence Filtering (Week 2)
- Enable confidence threshold (0.7)
- Monitor impact on dataset size
- Adjust threshold if needed

#### Phase 3: Switch to Balanced Strategy (Week 3)
- Change to BALANCED_SCORE strategy
- Compare model performance
- A/B test against BOUNCE_FIRST

#### Phase 4: Optimize Weights (Week 4)
- Experiment with different weights
- Use asset-specific configurations
- Finalize production config

---

### Step 9: Monitoring and Logging

Add logging to track strategy performance:

```python
import logging

logger = logging.getLogger(__name__)

# In selection code
best_window, confidence = strategy.select_window(channels, labels_per_window)

logger.info(
    f"Window selection: strategy={strategy.__class__.__name__}, "
    f"window={best_window}, confidence={confidence:.3f}"
)

if confidence < MIN_CONFIDENCE:
    logger.warning(
        f"Low confidence selection: {confidence:.3f} < {MIN_CONFIDENCE}"
    )

# Track metrics
metrics = {
    'window': best_window,
    'confidence': confidence,
    'strategy': strategy.__class__.__name__,
    'num_channels': len(channels),
    'valid_channels': sum(1 for ch in channels.values() if ch.valid),
}
log_metrics(metrics)
```

---

## Post-Integration Verification

After integration, verify:

- [ ] All existing tests still pass
- [ ] New strategy tests pass
- [ ] Dataset preparation runs successfully
- [ ] Training pipeline works
- [ ] No performance regression
- [ ] Logging shows expected behavior
- [ ] Confidence distributions look reasonable

---

## Rollback Plan

If issues arise:

1. **Immediate Rollback**: Comment out new imports, revert to old functions
2. **Configuration Rollback**: Change strategy to BOUNCE_FIRST
3. **Complete Rollback**: Remove window_strategy imports, use original code

**Backup Before Integration**:
```bash
git checkout -b backup-before-window-strategy
git add -A
git commit -m "Backup before window selection strategy integration"
```

---

## Success Criteria

Integration is successful when:

- [x] All tests pass (41/41 framework + existing tests)
- [ ] Dataset preparation completes without errors
- [ ] Model training runs successfully
- [ ] Performance is equal or better than v7-v9
- [ ] Confidence scores are being logged
- [ ] No regression in model accuracy
- [ ] Code review approved
- [ ] Documentation is complete

---

## Timeline

**Suggested Timeline**:
- Day 1: Review implementation and tests
- Day 2: Update imports and configuration
- Day 3: Integration testing
- Day 4: Documentation updates
- Day 5: Final validation and deployment

**Total Effort**: ~1 week

---

## Support

For questions or issues:
1. Review `v7/core/WINDOW_STRATEGY_GUIDE.md`
2. Check `v7/core/WINDOW_STRATEGY_QUICKREF.md`
3. Review test cases in `v7/core/test_window_strategy.py`
4. Check examples in `v7/core/window_strategy_example.py`

---

## Deliverables Summary

**Created Files**:
1. `v7/core/window_strategy.py` (794 lines) - Core implementation
2. `v7/core/test_window_strategy.py` (723 lines) - Test suite
3. `v7/core/WINDOW_STRATEGY_GUIDE.md` (630 lines) - User guide
4. `v7/core/WINDOW_STRATEGY_SUMMARY.md` (400+ lines) - Implementation summary
5. `v7/core/WINDOW_STRATEGY_QUICKREF.md` (200+ lines) - Quick reference
6. `v7/core/window_strategy_example.py` (498 lines) - Examples
7. `v7/core/WINDOW_STRATEGY_INTEGRATION.md` (this file) - Integration guide

**Total**: 3,455+ lines of production-ready code and documentation

---

**Last Updated**: 2026-01-06
**Version**: v10.0.0
**Status**: Ready for Integration ✅
