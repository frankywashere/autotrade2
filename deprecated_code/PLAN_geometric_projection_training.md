# v5.7 Dual Prediction Mode - Implementation Complete

## Summary

Trains BOTH prediction methods simultaneously:
- **Direct**: Model learns high/low directly from neural net
- **Geometric**: Model learns duration → calculates high/low from channel geometry

Dashboard shows both with comparison view.

---

## What Was Implemented

### 1. ChannelFeatureIndexer (`src/ml/hierarchical_model.py:67-187`)

Maps channel feature names to tensor indices for fast extraction during forward pass.

```python
class ChannelFeatureIndexer:
    WINDOWS = [100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10]

    def extract_all_windows(self, x, tf, symbol='tsla'):
        """Extract channel features for all 14 windows in a TF."""
        # Returns: {
        #   'high_slope_pct': [batch, 14],
        #   'low_slope_pct': [batch, 14],
        #   'upper_dist': [batch, 14],
        #   'lower_dist': [batch, 14],
        #   'quality_score': [batch, 14],
        # }
```

### 2. Window Selectors (`src/ml/hierarchical_model.py:680-697`)

Per-TF networks that learn which window to trust for geometric projection.

```python
self.window_selectors = nn.ModuleDict({
    tf: nn.Sequential(
        nn.Linear(hidden_size + 14, 64),  # hidden + quality scores
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 14),  # logits for 14 windows
    ) for tf in self.TIMEFRAMES
})
```

### 3. Geometric Projection Method (`src/ml/hierarchical_model.py:825-927`)

```python
def _compute_geometric_projection(self, x_tf, hidden, tf, duration_mean):
    # 1. Extract channel features for all windows
    # 2. Select best window (soft during training, hard during inference)
    # 3. Calculate: geo_high = upper_dist + (high_slope_pct × duration)
    # 4. Calculate: geo_low = -lower_dist + (low_slope_pct × duration)
    return {
        'high': geo_high,
        'low': geo_low,
        'duration': duration_mean,
        'window_weights': window_weights,
        'channel_state': {...}
    }
```

### 4. Forward Pass Updates (`src/ml/hierarchical_model.py:1234-1300`)

Both paths computed for each TF:

```python
# Direct path (learned)
pred_high = self.timeframe_heads[f'{tf}_high'](hidden)
pred_low = self.timeframe_heads[f'{tf}_low'](hidden)

# Geometric path (calculated)
geo_result = self._compute_geometric_projection(x_tf, hidden, tf, duration_mean)

# Store both
direct_predictions[tf] = {'high': pred_high, 'low': pred_low, 'conf': pred_conf}
geometric_predictions[tf] = geo_result
```

### 5. Loss Function (`train_hierarchical.py:3792-3818`)

```python
# Geometric price loss (validates that geometry produces correct prices)
geo_price_loss_total = 0.0
for tf, geo_data in hidden_states['geometric_predictions'].items():
    geo_high = geo_data['high'].squeeze()
    geo_low = geo_data['low'].squeeze()

    geo_high_loss = F.mse_loss(geo_high, target_high)
    geo_low_loss = F.mse_loss(geo_low, target_low)

    # Weight: 0.4 (duration is 0.3, totals ~0.7 for geometric path)
    geo_price_loss_total += 0.4 * (geo_high_loss + geo_low_loss) / 2
```

### 6. Checkpoint Updates (`train_hierarchical.py:4086-4089`)

```python
'model_version': '5.7',
'has_geometric_projection': True,
'feature_columns': getattr(model_to_save, '_feature_columns', None),
```

### 7. Predictor Updates (`predict.py:1195-1274`)

Returns comprehensive `all_timeframes` list:

```python
result['all_timeframes'] = [
    {
        'timeframe': 'weekly',
        'rank': 1,
        'direct': {'high': 3.2, 'low': -2.1, 'confidence': 0.85},
        'geometric': {
            'high': 3.1, 'low': -2.0,
            'duration_bars': 47,
            'channel_state': {'high_slope_pct': 0.05, ...}
        },
        'validity': 0.87,
    },
    ...
]
```

### 8. Dashboard Updates (`dashboard_v531.py:258-369, 414-536`)

- View mode selector: Direct / Geometric / Compare
- Side-by-side comparison with agreement indicator
- All Channels table with D.High/D.Low and G.High/G.Low/Dur columns

---

## Loss Breakdown

| Component | Weight | What It Trains |
|-----------|--------|----------------|
| primary | 1.0 | Direct high/low predictions |
| duration | 0.3 | Duration prediction (NLL) |
| geo_price | 0.4 | Geometric projection accuracy |
| validity | 0.2 | Channel validity prediction |
| transition | 0.3 | Phase transition prediction |

**Diagnostic Matrix:**

| Duration Loss | Geo Price Loss | Interpretation |
|---------------|----------------|----------------|
| Low ✓ | Low ✓ | Everything works |
| Low ✓ | High ✗ | Channel features noisy |
| High ✗ | Low ✓ | Got lucky |
| High ✗ | High ✗ | Duration needs work |

---

## Next Steps (Planned)

### 1. Label Verification

Verify that extracted labels are correct:
- Channel detection (slopes, bounds, position)
- Transition labels (continue, switch_tf, reverse, sideways)
- Continuation labels (duration, gain)
- Break detection

**Approach:** Use visualizer to inspect random samples visually.

### 2. Visualizer Updates

Modify existing visualizer to:
- Show detected channels overlaid on price
- Mark transition points
- Display continuation metrics
- Allow random sample inspection

### 3. Extraction Optimization

After verifying correctness:
- Profile slow paths
- Vectorize remaining loops
- Consider numba/cython for hot paths
- Maintain exact calculation parity (test suite)

---

## Files Modified

| File | Lines Changed |
|------|---------------|
| `src/ml/hierarchical_model.py` | +250 |
| `train_hierarchical.py` | +35 |
| `predict.py` | +80 |
| `dashboard_v531.py` | +120 |
