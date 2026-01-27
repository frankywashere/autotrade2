# Dual Output Design - Per-Timeframe + Aggregate

## The Solution: Output BOTH

You can have your cake and eat it too - output both per-timeframe predictions AND an aggregate.

---

## Architecture

```
11 TF Branches (64-dim each)
       │
       ├─────────────────────────────────────┐
       │                                     │
       ▼                                     ▼
PER-TIMEFRAME HEADS                  CROSS-TF ATTENTION
(11 separate predictions)            (aggregate context)
       │                                     │
       │                                     ▼
       │                              AGGREGATE HEADS
       │                              (1 combined prediction)
       │                                     │
       └─────────────────┬───────────────────┘
                         ▼
                 COMBINED OUTPUT
```

---

## Output Structure

```python
output = {
    # PER-TIMEFRAME (for dashboard breakdown)
    'per_tf': {
        'duration_mean': [batch, 11],        # Each TF's prediction
        'duration_log_std': [batch, 11],
        'direction_logits': [batch, 11],     # Binary per TF
        'next_channel_logits': [batch, 11, 3],
        'confidence': [batch, 11]
    },

    # AGGREGATE (for simple trading signal)
    'aggregate': {
        'duration_mean': [batch, 1],         # Best estimate
        'duration_log_std': [batch, 1],
        'direction_logits': [batch, 2],      # Binary classes
        'next_channel_logits': [batch, 3],   # 3 classes
        'confidence': [batch, 1]             # Overall confidence
    }
}
```

---

## How It Works

### 1. Per-Timeframe Predictions (What Each TF Thinks)

```python
# For each of 11 timeframes
for tf_idx, embedding in enumerate(tf_embeddings):
    # Each TF makes its own prediction
    per_tf_predictions[tf_idx] = {
        'duration': duration_head(embedding),    # What this TF predicts
        'direction': direction_head(embedding),
        'next_channel': next_channel_head(embedding),
        'confidence': confidence_head(embedding)
    }
```

**Result:** 11 separate predictions
- 5min thinks: "12 bars, DOWN, 62% confident"
- 1hr thinks: "23 bars, UP, 89% confident"
- 4hr thinks: "5 bars, UP, 78% confident"

### 2. Aggregate Prediction (Best Combined Estimate)

**Method 1: Use Highest Confidence TF**
```python
# Simple approach
best_tf_idx = per_tf_confidences.argmax(dim=1)  # Which TF is most confident?

aggregate = {
    'duration': per_tf_durations[best_tf_idx],
    'direction': per_tf_directions[best_tf_idx],
    'confidence': per_tf_confidences[best_tf_idx]
}
```

**Method 2: Confidence-Weighted Average**
```python
# Sophisticated approach
weights = F.softmax(per_tf_confidences, dim=1)  # [batch, 11] confidence weights

aggregate_duration = (per_tf_durations * weights).sum(dim=1)  # Weighted average
# Direction: pick from highest confidence
aggregate_direction = per_tf_directions[weights.argmax(dim=1)]
```

**Method 3: Use Cross-TF Attention (Current)**
```python
# What the model currently does
context = cross_tf_attention(tf_embeddings)  # [batch, 128]

aggregate = {
    'duration': duration_head(context),
    'direction': direction_head(context),
    'confidence': confidence_head(context)
}
```

---

## Dashboard Display

### Multi-Timeframe Table (From per_tf predictions)

```
Current Timeframe Analysis:
┌──────────┬──────────┬───────────┬────────────┬──────┐
│ TF       │ DURATION │ DIRECTION │ CONFIDENCE │ USE? │
├──────────┼──────────┼───────────┼────────────┼──────┤
│ 5min     │ 12 bars  │ BEAR→BULL │    62%     │      │
│ 15min    │ 8 bars   │ BEAR→SIDE │    71%     │      │
│ 1hr      │ 23 bars  │ SIDE→BULL │    89%     │  ⭐  │
│ 4hr      │ 5 bars   │ BULL      │    78%     │      │
│ daily    │ 3 bars   │ BULL      │    55%     │      │
└──────────┴──────────┴───────────┴────────────┴──────┘
```

### Aggregate Signal (Simple one-liner)

```
┌─────────────────────────────────────────────────────┐
│  RECOMMENDED TRADE: LONG @ $345.67                  │
│  Timeframe: 1hr | Duration: 23 bars | Conf: 89%    │
└─────────────────────────────────────────────────────┘
```

**OR** if you want both signals side-by-side:

```
┌───────────────────────── SIGNALS ─────────────────────────┐
│                                                            │
│  AGGREGATE (Weighted):    LONG @ $345.67                  │
│  Duration: 18 bars | Break: UP | Confidence: 82%          │
│                                                            │
│  BEST TIMEFRAME (1hr):    LONG @ $345.67                  │
│  Duration: 23 bars | Break: UP | Confidence: 89% ⭐       │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## Implementation

### Model Changes (hierarchical_cfc.py)

```python
def forward(self, x):
    # ... (11 TF branches, 11 embeddings)

    # PER-TIMEFRAME PREDICTIONS
    per_tf_preds = self._predict_per_timeframe(tf_embeddings)

    # AGGREGATE PREDICTION (using attention)
    context = self.cross_tf_attention(tf_embeddings_tensor)
    aggregate_preds = self._predict_aggregate(context)

    return {
        'per_tf': per_tf_preds,      # [batch, 11] or [batch, 11, 3]
        'aggregate': aggregate_preds  # [batch, 1] or [batch, 2/3]
    }
```

### Loss Calculation (trainer.py)

```python
# Use BOTH in training
loss_per_tf, _ = self.per_tf_loss(predictions['per_tf'], targets_per_tf)
loss_aggregate, _ = self.aggregate_loss(predictions['aggregate'], targets_aggregate)

# Combined
total_loss = loss_per_tf + 0.5 * loss_aggregate
```

**OR** just use per-timeframe and derive aggregate at inference:

```python
# Training: Only train per-timeframe
loss, loss_dict = self.criterion(predictions['per_tf'], targets)

# Inference: Derive aggregate from per-timeframe
best_idx = predictions['per_tf']['confidence'].argmax()
aggregate = extract_best_prediction(predictions['per_tf'], best_idx)
```

---

## Dashboard Code

```python
# Get predictions
predictions = model(features)

# Show per-timeframe table
for tf_idx, tf_name in enumerate(TIMEFRAMES):
    duration = predictions['per_tf']['duration_mean'][0, tf_idx].item()
    confidence = predictions['per_tf']['confidence'][0, tf_idx].item()
    direction = predictions['per_tf']['direction_logits'][0, tf_idx].argmax()

    print(f"{tf_name:8s} | {duration:3.0f} bars | {'UP' if direction else 'DOWN'} | {confidence:.0%}")

# Show aggregate
agg_duration = predictions['aggregate']['duration_mean'][0, 0].item()
agg_confidence = predictions['aggregate']['confidence'][0, 0].item()
print(f"\nAggregate: {agg_duration:.0f} bars, Confidence: {agg_confidence:.0%}")
```

**Output:**
```
5min     | 12 bars | DOWN | 62%
15min    | 8 bars  | UP   | 71%
1hr      | 23 bars | UP   | 89% ⭐
4hr      | 5 bars  | UP   | 78%

Aggregate: 18 bars, Confidence: 82%
Recommended: Trade 1hr (highest individual confidence)
```

---

## Recommended Approach

**Best of both worlds:**

1. **Train on per-timeframe predictions** (primary)
   - Each TF learns its own patterns
   - Produces 11 confidence scores
   - Matches your vision

2. **Derive aggregate at inference time**
   - Don't train a separate aggregate head
   - Just pick the highest confidence TF's prediction
   - Or compute weighted average

**Why this is best:**
- ✅ Matches your dashboard vision
- ✅ Leverages all 11 timeframes
- ✅ No extra training complexity
- ✅ Simple: aggregate = best(per_tf)
- ✅ Compatible with CombinedLoss

---

## Code Changes Needed

**Only modify model (hierarchical_cfc.py):**
- Change prediction heads to output per-TF predictions
- Remove aggregate heads (or keep for optional weighted average)
- Return proper shapes: [batch, 11] for most outputs

**Effort:** 1-2 hours

**Result:** Dashboard shows all 11 timeframes with confidence, highlights best one.

---

**ANSWER TO YOUR QUESTION:**

Yes, you can show BOTH:
- Per-timeframe table (11 rows)
- Aggregate recommendation (pick highest confidence)

The aggregate is just derived from per-timeframe, not separately trained.
