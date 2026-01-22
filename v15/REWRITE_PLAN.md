# V15 Complete System Reference

## Overview

The V15 channel prediction system is a complete rewrite addressing all architectural issues from previous versions. This document serves as the comprehensive reference for the fully implemented system.

**Status: COMPLETE**

## Critical Issues Resolved

### Issue 1: Feature Count Mismatch (RESOLVED)
- **Problem:** V15 generates 8,632 features but V7 model expects 776.
- **Solution:** New model architecture accepts all 8,665 features with explicit per-feature weights.

### Issue 2: No Explicit Feature Weights (RESOLVED)
- **Problem:** Features were compressed through dense layers, losing individual importance.
- **Solution:** `ExplicitFeatureWeights` layer with learnable weight per feature. `FeatureGating` available for feature suppression.

### Issue 3: Rolling Sample / Stale TF Problem (RESOLVED)
- **Problem:** `dropna()` discards incomplete bars, causing higher TF features to be identical for many consecutive samples.
- **Solution:** `resample_with_partial()` keeps partial bars with metadata (completion_pct, bars_in_partial, complete_bars).

### Issue 4: No Cross-Correlation Checks (RESOLVED)
- **Problem:** Redundant features passed unchanged, wasting model capacity.
- **Solution:** `analyze_correlations()` in features/validation.py identifies highly correlated pairs and suggests drops.

### Issue 5: Silent Failures (RESOLVED)
- **Problem:** Errors caught and ignored silently.
- **Solution:** Custom exception hierarchy in exceptions.py with LOUD failures. No silent try/except blocks.

---

## Implementation Status

### Phase 1: Foundation - COMPLETE
- [x] `exceptions.py` with custom exception classes (9 exception types)
- [x] `config.py` with all constants and configuration
- [x] `types.py` with data structures (ChannelSample, ChannelLabels)

### Phase 2: Data Pipeline - COMPLETE
- [x] `data/loader.py` with validation
- [x] `data/resampler.py` with partial bar support and `BarMetadata`
- [x] Bar completion metadata features (33 features)

### Phase 3: Feature Extraction - COMPLETE
- [x] All feature modules with TF prefixes:
  - `features/tsla_price.py` (60 features per TF)
  - `features/technical.py` (77 features per TF)
  - `features/spy.py` (80 features per TF)
  - `features/vix.py` (25 features per TF)
  - `features/cross_asset.py` (40 features per TF)
  - `features/tsla_channel.py` (50 features per window)
  - `features/window_scores.py` (50 features per TF)
  - `features/channel_history.py` (50 features per TF)
  - `features/events.py` (30 global features)
- [x] `features/extractor.py` - main orchestrator
- [x] `features/tf_extractor.py` - TF-aware extraction
- [x] `features/validation.py` - correlation analysis tools
- [x] `features/utils.py` - safe_float, ensure_finite helpers

### Phase 4: Model Architecture - COMPLETE
- [x] `models/feature_weights.py` - ExplicitFeatureWeights and FeatureGating
- [x] `models/tf_encoder.py` - MultiTFEncoder for 782 features each
- [x] `models/cross_tf_attention.py` - TF relationship learning
- [x] `models/prediction_heads.py` - duration, direction, new_channel, confidence
- [x] `models/full_model.py` - V15Model with 8,665 input features

### Phase 5: Training Pipeline - COMPLETE
- [x] `training/dataset.py` - ChannelDataset for 8,665 features
- [x] `training/trainer.py` - full training loop with mixed precision
- [x] `training/metrics.py` - MetricsTracker and evaluation metrics

### Phase 6: Integration & Testing - COMPLETE
- [x] `pipeline.py` - CLI entry point
- [x] `scanner.py` - parallel channel scanning
- [x] `inspector.py` - terminal-based sample inspection
- [x] Visual inspection tools (deprecated)

---

## Directory Structure

```
v15/
├── REWRITE_PLAN.md          # This file
├── __init__.py              # Clean exports
├── config.py                # All configuration in one place
├── exceptions.py            # Custom exceptions (loud failures)
├── types.py                 # Data structures
├── data/
│   ├── __init__.py
│   ├── loader.py            # Data loading with validation
│   └── resampler.py         # TF resampling with partial bar support
├── features/
│   ├── __init__.py
│   ├── utils.py             # Helper functions (safe_float, etc.)
│   ├── validation.py        # Feature validation & correlation
│   ├── tsla_price.py        # TSLA price features
│   ├── tsla_channel.py      # Channel features per window
│   ├── technical.py         # Technical indicators
│   ├── spy.py               # SPY features
│   ├── vix.py               # VIX features
│   ├── cross_asset.py       # Cross-asset correlations
│   ├── window_scores.py     # Window score aggregation
│   ├── channel_history.py   # Channel history features
│   ├── events.py            # Time/calendar events
│   ├── tf_extractor.py      # TF-aware extraction
│   └── extractor.py         # Main orchestrator
├── models/
│   ├── __init__.py          # create_model() factory
│   ├── feature_weights.py   # Explicit feature weighting layer
│   ├── tf_encoder.py        # Per-TF feature encoder
│   ├── cross_tf_attention.py # Cross-timeframe attention
│   ├── prediction_heads.py  # Output heads
│   └── full_model.py        # Complete model (8,665 -> predictions)
├── training/
│   ├── __init__.py
│   ├── dataset.py           # ChannelDataset for PyTorch
│   ├── trainer.py           # Training loop with validation
│   └── metrics.py           # Evaluation metrics
├── labels.py                # Label generation
├── scanner.py               # Parallel data scanning pipeline
├── pipeline.py              # CLI entry point
└── inspector.py             # Terminal-based sample inspector
```

---

## Feature Architecture (8,665 features)

### Timeframes and Windows
```
11 Timeframes: [5min, 15min, 30min, 1h, 2h, 3h, 4h, daily, weekly, monthly, 3month]
8 Windows: [10, 20, 30, 40, 50, 60, 70, 80]
```

### Feature Breakdown

| Category | Per Unit | Units | Total |
|----------|----------|-------|-------|
| TSLA Price | 60 | 11 TFs | 660 |
| Technical | 77 | 11 TFs | 847 |
| SPY | 80 | 11 TFs | 880 |
| VIX | 25 | 11 TFs | 275 |
| Cross-Asset | 40 | 11 TFs | 440 |
| **Window-Independent Subtotal** | **282** | **11 TFs** | **3,102** |
| Channel (per window) | 50 | 8 windows x 11 TFs | 4,400 |
| Window Scores | 50 | 11 TFs | 550 |
| Channel History | 50 | 11 TFs | 550 |
| **Aggregated Subtotal** | **100** | **11 TFs** | **1,100** |
| Events (global) | 30 | 1 | 30 |
| Bar Metadata | 3 | 11 TFs | 33 |
| **TOTAL** | | | **8,665** |

### Bar Metadata Features (per TF)
```
{tf}_bar_completion_pct   # 0.0-1.0, how complete is current bar
{tf}_bars_in_partial      # Source bars in partial bar
{tf}_complete_bars        # Total complete bars
```

---

## Model Architecture

```
Input: 8,665 features
    |
[Feature Validation Layer] - Check for NaN/Inf, raise if found
    |
[Explicit Feature Weights] - 8,665 learnable weights (element-wise)
    |
[Feature Gating] (optional) - sigmoid gates for feature suppression
    |
[Split Features]
    |-> TF Features: [batch, 11, 782]
    |-> Shared Features: [batch, 63] (events + bar_metadata)
    |
[MultiTFEncoder] x 11 - Each processes its 782 features -> 128-dim embedding
    |
[CrossTFAttention] - Multi-head attention across TF embeddings
    |
[TFAggregator] - Attention-weighted combination -> 256-dim
    |
[Prediction Heads]
    |-> Duration Head (Gaussian: mean + log_std)
    |-> Direction Head (binary logits: up/down)
    |-> New Channel Head (3-class logits: bear/sideways/bull)
    |-> Confidence Head (calibrated probability)
```

### Model Configuration (from config.py)
```python
MODEL_CONFIG = {
    'input_dim': 8665,
    'hidden_dim': 256,
    'n_attention_heads': 8,
    'dropout': 0.1,
    'use_explicit_weights': True,
}
```

---

## Training Configuration

```python
TRAINING_CONFIG = {
    'batch_size': 64,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'warmup_steps': 1000,
    'max_epochs': 100,
}

SCANNER_CONFIG = {
    'step': 10,
    'warmup_bars': 32760,
    'forward_bars': 8000,
    'workers': 4,
}
```

### Training Features
- Mixed precision training (FP16 on CUDA)
- Gradient clipping (default: 1.0)
- OneCycleLR scheduler with warmup
- Early stopping with patience
- Automatic checkpointing (latest.pt, best.pt)

---

## Commands Reference

### Data Scanning
```bash
# Generate samples with parallel processing
python -m v15.pipeline scan --data-dir data --output samples.pkl

# Options
python -m v15.pipeline scan \
    --data-dir data \
    --output samples.pkl \
    --step 10 \           # Bars between samples (default: 10)
    --warmup 32760 \      # Minimum warmup bars (default: 32760)
    --forward 8000 \      # Forward bars for labels (default: 8000)
    --workers 4           # Parallel workers (default: 4)
```

### Model Training
```bash
# Train model
python -m v15.pipeline train --samples samples.pkl --output checkpoints/

# Options
python -m v15.pipeline train \
    --samples samples.pkl \
    --output checkpoints/ \
    --batch-size 64 \     # Batch size (default: 64)
    --lr 0.0001 \         # Learning rate (default: 1e-4)
    --epochs 100 \        # Max epochs (default: 100)
    --target-tf daily     # Target TF for labels (default: daily)
```

### Feature Analysis
```bash
# Analyze feature correlations
python -m v15.pipeline analyze --samples samples.pkl --output report.json
```

### Sample Inspection
```bash
# Terminal-based inspector
python -m v15.inspector --cache samples.pkl

# Navigation:
#   Enter/n: Next sample
#   p: Previous sample
#   w: Change window
#   t: Change timeframe
#   q: Quit
#   [number]: Jump to sample
```

### Direct Module Execution
```bash
# Run scanner directly with test settings
python -m v15.scanner

# Run with custom data directory
python -c "
from v15.data import load_market_data
from v15.scanner import scan_channels

tsla, spy, vix = load_market_data('data')
samples = scan_channels(tsla, spy, vix, step=100)
print(f'Generated {len(samples)} samples')
"
```

---

## Exception Hierarchy

```python
V15Error                    # Base exception
├── FeatureExtractionError  # Feature extraction failures
├── InvalidFeatureError     # NaN, Inf, or invalid type
├── DataLoadError           # Data loading failures
├── ResamplingError         # Timeframe resampling failures
├── ChannelDetectionError   # Channel detection failures
├── LabelGenerationError    # Label generation failures
├── ModelError              # Model forward pass failures
├── ConfigurationError      # Invalid configuration
└── ValidationError         # Validation check failures

FeatureCorrelationWarning   # Warning for highly correlated features
```

---

## Inference Module

### Loading a Trained Model
```python
from v15.models import create_model
import torch

# Load model
model = create_model()
checkpoint = torch.load('checkpoints/best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Run inference
with torch.no_grad():
    predictions = model(features_tensor)

    duration_mean = predictions['duration_mean']
    duration_std = torch.exp(predictions['duration_log_std'])
    direction_prob = torch.sigmoid(predictions['direction_logits'])
    new_channel_probs = torch.softmax(predictions['new_channel_logits'], dim=-1)
```

### Feature Extraction for Inference
```python
from v15.features.extractor import extract_all_features
from v15.data import load_market_data, resample_with_partial

# Load current market data
tsla_df, spy_df, vix_df = load_market_data('data')

# Extract features (handles partial bars automatically)
features = extract_all_features(
    tsla_df=tsla_df,
    spy_df=spy_df,
    vix_df=vix_df,
    timestamp=tsla_df.index[-1],
    channels_by_window={},  # Will detect automatically
    validate=True
)

# Convert to tensor
feature_names = sorted(features.keys())
feature_array = torch.tensor([features[n] for n in feature_names], dtype=torch.float32)
```

### Getting Feature Importance
```python
# After training, extract learned feature importance
importance = model.get_feature_importance()  # Returns tensor of 8,665 weights

# Get top-k most important features
top_indices = model.feature_weights.get_top_features(k=100)

# Map to feature names
feature_names = sorted(samples[0].tf_features.keys())
top_features = [feature_names[i] for i in top_indices]
```

---

## Dashboard Integration

### Real-Time Predictions
```python
import streamlit as st
from v15.models import create_model
from v15.features.extractor import extract_all_features

@st.cache_resource
def load_model():
    model = create_model()
    model.load_state_dict(torch.load('best.pt')['model_state_dict'])
    model.eval()
    return model

def get_prediction(tsla_df, spy_df, vix_df):
    model = load_model()
    features = extract_all_features(tsla_df, spy_df, vix_df,
                                    tsla_df.index[-1], {})
    # ... convert and predict
```

### Visualization Helpers
```python
from v15.features.extractor import get_feature_breakdown, get_feature_group_counts

# Show feature counts by category
breakdown = get_feature_breakdown()
group_counts = get_feature_group_counts()

# Display bar metadata for partial bar awareness
for tf in TIMEFRAMES:
    completion = features[f'{tf}_bar_completion_pct']
    st.metric(f"{tf} Bar Completion", f"{completion:.1%}")
```

---

## Live Trading Integration

### Key Considerations

1. **Partial Bar Handling**: The system is designed for live trading where the current bar is always incomplete. Bar metadata features tell the model how complete each TF's latest bar is.

2. **Feature Consistency**: Always use `extract_all_features()` to ensure consistent feature ordering and validation.

3. **Prediction Uncertainty**: Duration predictions come with uncertainty (log_std). Use this for position sizing:
   ```python
   duration_mean = predictions['duration_mean']
   duration_std = torch.exp(predictions['duration_log_std'])

   # Conservative estimate (1 std below mean)
   conservative_duration = duration_mean - duration_std
   ```

4. **Direction Confidence**: Use sigmoid probability for trade signals:
   ```python
   direction_prob = torch.sigmoid(predictions['direction_logits'])

   # High confidence threshold
   if direction_prob > 0.7:  # Strong up signal
       ...
   elif direction_prob < 0.3:  # Strong down signal
       ...
   ```

### Example Integration
```python
class V15TradingSignal:
    def __init__(self, model_path: str):
        self.model = create_model()
        self.model.load_state_dict(torch.load(model_path)['model_state_dict'])
        self.model.eval()

    def generate_signal(self, tsla_df, spy_df, vix_df):
        # Extract features
        features = extract_all_features(
            tsla_df, spy_df, vix_df,
            tsla_df.index[-1], {}
        )

        # Convert to tensor
        feature_names = sorted(features.keys())
        x = torch.tensor([features[n] for n in feature_names]).unsqueeze(0)

        # Predict
        with torch.no_grad():
            pred = self.model(x)

        return {
            'expected_duration': pred['duration_mean'].item(),
            'duration_uncertainty': torch.exp(pred['duration_log_std']).item(),
            'up_probability': torch.sigmoid(pred['direction_logits']).item(),
            'new_channel_probs': torch.softmax(pred['new_channel_logits'], dim=-1).tolist(),
            'bar_completion': {
                tf: features[f'{tf}_bar_completion_pct']
                for tf in TIMEFRAMES
            }
        }
```

---

## Deprecation Guide

### Deprecated Functions

| Deprecated | Replacement | Notes |
|------------|-------------|-------|
| `extract_features()` | `extract_all_features()` | Legacy wrapper still works but emits warning |
| Direct channel detection | `_detect_channels()` | Now handled internally by extractor |
| `dropna()` resampling | `resample_with_partial()` | Critical change for partial bar support |

### Migration Steps

1. **Update imports**:
   ```python
   # Old
   from v15.features import extract_features

   # New
   from v15.features.extractor import extract_all_features
   ```

2. **Update function calls**:
   ```python
   # Old
   features = extract_features(tsla, spy, vix, channel, window)

   # New
   features = extract_all_features(
       tsla_df=tsla,
       spy_df=spy,
       vix_df=vix,
       timestamp=tsla.index[-1],
       channels_by_window={},
       validate=True
   )
   ```

3. **Handle new feature count**: Model now expects 8,665 features instead of 776. Update any hardcoded dimensions.

4. **Use new data structures**: Replace custom dicts with `ChannelSample` and `ChannelLabels` from `types.py`.

---

## Troubleshooting

### Common Errors

#### InvalidFeatureError: Feature 'xxx' has invalid value
**Cause**: NaN or Inf value in extracted features.
**Solution**: Check input data for gaps. Ensure sufficient warmup bars. The system requires clean continuous data.

```python
# Debug: Check which features are invalid
from v15.features.validation import validate_features
invalid = validate_features(features, raise_on_invalid=False)
print(f"Invalid features: {invalid[:10]}")
```

#### ResamplingError: Cannot resample empty DataFrame
**Cause**: Input DataFrame is empty or None.
**Solution**: Verify data loading. Check date range has data.

#### ModelError: Input contains NaN values
**Cause**: Model received invalid input tensor.
**Solution**: Ensure feature extraction validation is enabled (`validate=True`).

#### DataLoadError: Not enough data
**Cause**: Dataset too small for warmup_bars + forward_bars requirement.
**Solution**: Use a larger dataset or reduce warmup_bars/forward_bars.

### Memory Issues

#### Out of Memory during scanning
- Reduce `workers` count
- Increase `step` value (fewer samples)
- Process in smaller date chunks

#### Out of Memory during training
- Reduce `batch_size`
- Enable gradient checkpointing
- Use `pin_memory=False` if CPU memory limited

### Performance Issues

#### Slow feature extraction
- Ensure parallel scanning (`workers > 1`)
- Use `step > 1` to reduce sample count
- Profile with: `python -m cProfile -m v15.scanner`

#### Slow training
- Enable mixed precision (automatic on CUDA)
- Reduce model `hidden_dim`
- Increase `batch_size` if memory allows

---

## Performance Notes

### Feature Extraction
- **Serial extraction**: ~50 samples/minute
- **Parallel (4 workers)**: ~180 samples/minute
- **Parallel (8 workers)**: ~300 samples/minute

### Model Training
- **CPU**: ~2 epochs/hour (batch_size=64)
- **CUDA (mixed precision)**: ~20 epochs/hour (batch_size=64)
- **MPS (Apple Silicon)**: ~10 epochs/hour (batch_size=64)

### Memory Requirements
- **Feature extraction**: ~4GB RAM per worker
- **Model training**: ~8GB GPU memory (batch_size=64)
- **Full dataset (100K samples)**: ~10GB RAM

### Recommended Hardware
- **Minimum**: 16GB RAM, 8-core CPU
- **Recommended**: 32GB RAM, 16+ cores, CUDA GPU with 8GB+ VRAM
- **Apple Silicon**: M1 Pro/Max with 16GB+ unified memory

---

## Success Criteria Checklist

- [x] All 8,665 features used - Model input_dim = 8665
- [x] Explicit weights - Each feature has learnable weight
- [x] No stale TF data - Partial bars included with completion %
- [x] Correlation analysis - Report generated on feature redundancy
- [x] Loud failures - No silent try/except, explicit exceptions
- [x] Clean architecture - Single responsibility per module
- [x] Parallel scanning - Multi-worker feature extraction
- [x] Training pipeline - Full trainer with validation and checkpointing
- [x] Inspection tools - Terminal and visual inspectors
- [x] Documentation - Comprehensive reference (this file)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 15.0.0 | Initial | Complete rewrite of channel prediction system |

---

## Related Documentation

- `v7/README.md` - Original channel detection implementation
- `docs/` - Additional implementation notes
- Code comments - Extensive inline documentation in all modules
