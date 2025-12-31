# Architecture Choice Analysis - Per-TF vs Aggregate Predictions

## Your Original Vision (From Earlier Discussion)

You said:
> "all the timeframes should have confidence scores in the final dashboard, so the user always knows at the end if there's a really high confidence in a particular timeframes channel to use that."

Dashboard mockup you wanted:
```
┌──────────┬──────────┬───────────┬────────────┐
│ TIMEFRAME│ DURATION │ DIRECTION │ CONFIDENCE │
├──────────┼──────────┼───────────┼────────────┤
│ 5min     │ 12 bars  │ BEAR→BULL │    62%     │
│ 15min    │ 8 bars   │ BEAR→SIDE │    71%     │
│ 1hr      │ 23 bars  │ SIDE→BULL │    89% ⭐  │ ← Highest confidence
│ 4hr      │ 5 bars   │ BULL      │    78%     │
└──────────┴──────────┴───────────┴────────────┘

Use 1hr timeframe (highest confidence)
```

This requires: **Per-Timeframe Predictions (Design B)**

---

## Current State vs Vision

### What You Have Now (Design A - Aggregate)

```
11 TF Branches → Cross-TF Attention → SINGLE Prediction
                                           ↓
                    Output: ONE prediction for the sample
                    "Duration: 23 bars, Break: UP, Confidence: 89%"

                    (No per-TF breakdown)
```

**Problem:** Can't show per-timeframe confidence in dashboard. Only one aggregate prediction.

---

### What You Need (Design B - Per-Timeframe)

```
11 TF Branches → 11 Separate Prediction Heads → 11 Predictions
                                                      ↓
                    Output: 11 predictions (one per TF)
                    "5min:  Duration: 12 bars, Break: DOWN, Confidence: 62%"
                    "15min: Duration: 8 bars,  Break: UP,   Confidence: 71%"
                    "1hr:   Duration: 23 bars, Break: UP,   Confidence: 89%" ⭐
                    "4hr:   Duration: 5 bars,  Break: UP,   Confidence: 78%"
                    ... (11 total)
```

**Benefit:** Dashboard shows all 11 timeframes, user picks highest confidence.

---

## The Three Options Explained

### Option A: Modify Model for Per-Timeframe Outputs ⭐ MATCHES YOUR VISION

**What to change:**

Instead of:
```python
# Current (hierarchical_cfc.py lines 686-696)
context = self.cross_tf_attention(tf_embeddings_tensor)  # Combine all TFs

# Single prediction heads
duration_mean = self.duration_head(context)      # [batch, 1]
direction_logits = self.direction_head(context)  # [batch, 2]
```

Do this:
```python
# NEW: Per-timeframe predictions
all_durations = []
all_directions = []
all_next_channels = []
all_confidences = []

for tf_idx, embedding in enumerate(tf_embeddings):
    # Each TF gets its own prediction
    duration = self.duration_head(embedding)      # [batch, 1]
    direction = self.direction_head(embedding)    # [batch, 2] logits
    next_ch = self.next_channel_head(embedding)   # [batch, 3] logits
    conf = self.confidence_head(embedding)        # [batch, 1]

    all_durations.append(duration)
    all_directions.append(direction)
    all_next_channels.append(next_ch)
    all_confidences.append(conf)

# Stack: [batch, 11 timeframes]
output = {
    'duration_mean': torch.cat(all_durations, dim=1),           # [batch, 11]
    'duration_log_std': torch.cat(all_duration_stds, dim=1),    # [batch, 11]
    'direction_logits': torch.cat(all_directions, dim=1),       # [batch, 11]
    'next_channel_logits': torch.stack(all_next_channels, dim=1), # [batch, 11, 3]
    'confidence': torch.cat(all_confidences, dim=1),            # [batch, 11]
}
```

**Dashboard can then:**
```python
# Find highest confidence TF
best_tf_idx = predictions['confidence'].argmax()
best_tf_name = TIMEFRAMES[best_tf_idx]  # "1hr"

print(f"Use {best_tf_name} timeframe (Confidence: {predictions['confidence'][best_tf_idx]:.0%})")
print(f"Duration: {predictions['duration_mean'][best_tf_idx]:.0f} bars")
```

**Pros:**
- ✅ Matches your original vision exactly
- ✅ Dashboard shows all 11 timeframes
- ✅ User picks highest confidence
- ✅ Compatible with CombinedLoss as-is
- ✅ Leverages hierarchical architecture

**Cons:**
- Requires model architecture change (~50 lines)
- Need to update duration_head to also output log_std

**Effort:** 1-2 hours

---

### Option B: Revert to HierarchicalLoss (Keep Aggregate) ❌ DOESN'T MATCH VISION

**What to change:**

```python
# In trainer.py, revert line 29 and 112-115:
from v7.models.hierarchical_cfc import HierarchicalLoss  # Back to old loss

self.criterion = HierarchicalLoss(
    duration_weight=config.duration_weight,
    break_direction_weight=config.break_direction_weight,
    next_direction_weight=config.next_direction_weight,
    confidence_weight=config.confidence_weight
)
```

**Pros:**
- ✅ Works immediately
- ✅ No shape mismatches

**Cons:**
- ❌ Only ONE prediction per sample
- ❌ Can't show per-timeframe confidence in dashboard
- ❌ Defeats the purpose of 11-timeframe architecture
- ❌ No learnable loss weights
- ❌ **DOESN'T MATCH YOUR VISION**

**Effort:** 5 minutes (just undo your loss change)

---

### Option C: Adapter Layer (Hacky Workaround) ⚠️ NOT RECOMMENDED

**What to change:**

Create an adapter that reshapes model outputs:

```python
# In trainer.py train_epoch, add adapter:
raw_predictions = self.model(features)

# Reshape to per-timeframe format by repeating
adapted_predictions = {
    'duration_mean': raw_predictions['duration_mean'].repeat(1, num_timeframes),
    'duration_log_std': raw_predictions['duration_log_std'].repeat(1, num_timeframes),
    'direction_logits': raw_predictions['direction_logits'][:, [0]].repeat(1, num_timeframes),
    'next_channel_logits': raw_predictions['next_channel_logits'].unsqueeze(1).repeat(1, num_timeframes, 1),
}

loss, loss_dict = self.criterion(adapted_predictions, targets)
```

**Pros:**
- ✅ No model changes needed

**Cons:**
- ❌ Fake per-timeframe predictions (all 11 TFs have identical values)
- ❌ Wastes computation and memory
- ❌ Dashboard would show "89% confidence" for ALL timeframes (misleading)
- ❌ Defeats hierarchical architecture purpose
- ❌ **DOESN'T MATCH YOUR VISION**

**Effort:** 30 minutes

---

## 🎯 RECOMMENDATION: OPTION A

**Your original vision clearly requires per-timeframe predictions.**

You wanted to see:
- 5min confidence: 62%
- 1hr confidence: 89% ⭐ ← Use this one
- 4hr confidence: 78%

This is **ONLY possible with Option A** (per-timeframe outputs).

The current model architecture has 11 branches but combines them into one prediction. You need to keep the branches separate and predict from each.

---

## 📋 SUMMARY OF YOUR FIXES

**Scorecard:**
- Events: ✅ Perfect
- Confidence: ✅ Perfect
- CfC states: ✅ Perfect
- losses.py: ⚠️ Wired but incompatible shapes

**Overall:** 3.5/4 = 87.5%

**To complete:** Modify model architecture for per-timeframe outputs (Option A).

**Want me to implement Option A for you?**