# Training Fixes Plan: Mode Collapse & Training Stability

**Created:** 2025-12-15
**Revised:** 2025-12-15 (after peer review)
**Status:** PENDING APPROVAL
**Backwards Compatibility:** NOT REQUIRED

---

## Executive Summary

Four critical bugs prevent effective training:

1. **BCE Mismatch**: Using BCEWithLogits on sigmoid outputs (double-sigmoid)
2. **Mode Collapse**: Hard `argmax()` selection → 100% 4h timeframe
3. **Geo-price Explosion**: Duration not clamped before geometric projection
4. **Validity Masks Ignored**: Training on placeholder/invalid labels

**NOT a bug:** The "scale mismatch" (predictions=-0.019 vs targets=0.33) is normal untrained behavior. The code uses percentage-points consistently throughout.

---

## Phase 1: Fix BCE/Sigmoid Mismatch (Critical Bug)

### Problem
```python
# Model outputs PROBABILITIES (after sigmoid):
self.hit_band_head = nn.Sequential(..., nn.Sigmoid())  # 0-1 range

# Training expects LOGITS (before sigmoid):
F.binary_cross_entropy_with_logits(mt['hit_band'], ...)  # Applies sigmoid internally!

# Result: sigmoid(sigmoid(x)) — completely wrong gradients
```

### Fix 1A: Remove Sigmoid from Heads (Keep BCEWithLogits)

**File:** `src/ml/hierarchical_model.py`

**Location:** Lines 529-542 (hit_band_head, hit_target_head definitions)

```python
# BEFORE:
self.hit_band_head = nn.Sequential(
    nn.Linear(hidden_size, hidden_size // 2),
    nn.ReLU(),
    nn.Linear(hidden_size // 2, 1),
    nn.Sigmoid()  # ← REMOVE THIS
)

# AFTER:
self.hit_band_head = nn.Sequential(
    nn.Linear(hidden_size, hidden_size // 2),
    nn.ReLU(),
    nn.Linear(hidden_size // 2, 1)
    # No sigmoid - BCEWithLogits handles it
)
```

**Apply to:**
- `hit_band_head` (line ~529)
- `hit_target_head` (line ~535)
- Any other heads using sigmoid + BCEWithLogits

### Fix 1B: Update Inference to Apply Sigmoid

**File:** `src/ml/hierarchical_model.py`

Where these heads are used for inference (not training), apply sigmoid:

```python
# During inference:
hit_band_prob = torch.sigmoid(self.hit_band_head(hidden))
hit_target_prob = torch.sigmoid(self.hit_target_head(hidden))
```

**Note:** Training uses raw logits (BCEWithLogits applies sigmoid). Inference needs explicit sigmoid.

---

## Phase 2: Fix Loss Display (Visibility)

### Problem
```
Logged:  Primary=0.328, Duration=13.658, Total=14.8
Actual:  Total=2215 (geo_price=~2200 NOT SHOWN)
```

### Fix 2A: Log geo_price Loss

**File:** `train_hierarchical.py`

| Line | Change |
|------|--------|
| ~3936 | Add: `print(f"   Geo-price: {loss_components.get('geo_price', 0):.3f}")` |

```python
# Around line 3936, after other loss prints:
print(f"   Calibration: {loss_components.get('calibration', 0):.3f}")
print(f"   Geo-price: {loss_components.get('geo_price', 0):.3f}")  # ADD THIS
print(f"   ─────────────────────")
```

### Fix 2B: Log Adaptive Projection Losses

**File:** `train_hierarchical.py`

Currently lines 3740-3746 add losses without tracking:
```python
loss = loss + 0.4 * F.mse_loss(mt['price_change_pct'].squeeze(), ...)
loss = loss + 0.3 * F.mse_loss(mt['horizon_bars_log'].squeeze(), ...)
```

**Change:** Track these in `loss_components`:
```python
adaptive_price_loss = 0.4 * F.mse_loss(mt['price_change_pct'].squeeze(), ...)
adaptive_horizon_loss = 0.3 * F.mse_loss(mt['horizon_bars_log'].squeeze(), ...)
loss = loss + adaptive_price_loss + adaptive_horizon_loss
loss_components['adaptive_price'] = adaptive_price_loss.item()
loss_components['adaptive_horizon'] = adaptive_horizon_loss.item()
```

---

## Phase 3: Loss Warmup (Stabilization)

### Problem
```
Epoch 1: geo_price_loss = 2200 (dominates everything)
         primary_loss = 0.3 (ignored)
         Model learns: "minimize geo_price at all costs"
```

### Fix 3A: Implement Warmup Function

**File:** `train_hierarchical.py`

Add new function before training loop (~line 3350):

```python
def get_loss_warmup_weight(epoch: int, warmup_epochs: int, final_weight: float) -> float:
    """Quadratic warmup: 0 → final_weight over warmup_epochs."""
    if epoch >= warmup_epochs:
        return final_weight
    progress = epoch / warmup_epochs
    return final_weight * (progress ** 2)
```

### Fix 3B: Apply Warmup to Secondary Losses

**File:** `train_hierarchical.py`

Modify loss computation section (~lines 3784-3826):

```python
# Configuration
WARMUP_EPOCHS = 5

# Get current warmup weights
duration_weight = get_loss_warmup_weight(epoch, WARMUP_EPOCHS, 0.3)
geo_price_weight = get_loss_warmup_weight(epoch, WARMUP_EPOCHS, 0.4)
validity_weight = get_loss_warmup_weight(epoch, WARMUP_EPOCHS, 0.2)
transition_weight = get_loss_warmup_weight(epoch, WARMUP_EPOCHS, 0.5)

# Apply warmup weights (replace hardcoded values)
if duration_loss_total > 0:
    loss = loss + duration_weight * duration_loss_total  # Was: 0.3 *

# ... similar for geo_price, validity, transition
```

### Warmup Schedule

| Epoch | Primary | Duration | Geo-price | Validity | Transition |
|-------|---------|----------|-----------|----------|------------|
| 1 | 1.0 | 0.012 | 0.016 | 0.008 | 0.020 |
| 2 | 1.0 | 0.048 | 0.064 | 0.032 | 0.080 |
| 3 | 1.0 | 0.108 | 0.144 | 0.072 | 0.180 |
| 4 | 1.0 | 0.192 | 0.256 | 0.128 | 0.320 |
| 5+ | 1.0 | 0.300 | 0.400 | 0.200 | 0.500 |

---

## Phase 4: Fix Geo-Price Explosion (Clamp Duration)

### Problem
```python
# Duration is clamped AFTER geo projection (too late!):
geo_high = upper_dist + (high_slope * duration_mean)  # duration could be 500+!
duration_mean = duration_mean.clamp(1, 48)  # Clamped here, but geo already computed

# If duration=500, high_slope=0.1%/bar:
# geo_high = 2.5 + (0.1 × 500) = 52.5%
# Target = 0.33%
# MSE = (52.5 - 0.33)² = 2721  ← THIS IS THE 2200 LOSS!
```

### Fix 4A: Clamp Duration BEFORE Geometric Projection

**File:** `src/ml/hierarchical_model.py`

**Location:** Lines 911-912 (geometric projection calculation)

```python
# BEFORE:
geo_high = upper_dist + (high_slope * duration_mean)
geo_low = -lower_dist + (low_slope * duration_mean)

# AFTER:
# Clamp duration for geo projection to prevent explosion
duration_for_geo = duration_mean.clamp(1, 48)  # Same bounds as reporting
geo_high = upper_dist + (high_slope * duration_for_geo)
geo_low = -lower_dist + (low_slope * duration_for_geo)
```

**Why:** The raw duration_mean from the duration head can be arbitrarily large early in training. Clamping before geo projection prevents the 2200+ loss explosion.

---

## Phase 5: Fix Mode Collapse (Soft Selection)

### Problem
```
Current:  argmax(confidences) → ONE winner → 100% gradients to winner
Result:   4h selected 100%, other TFs never train
```

### Fix 5A: Replace Hard Selection with Gumbel-Softmax (Using LOGITS)

**File:** `src/ml/hierarchical_model.py`

**CRITICAL:** `per_tf_confs` are sigmoid outputs (probabilities). Gumbel-Softmax needs logits!

**Location:** ~line 1347

```python
# BEFORE:
best_tf_idx = torch.argmax(per_tf_confs, dim=-1)
batch_indices = torch.arange(per_tf_highs.shape[0], device=per_tf_highs.device)
final_pred_high = per_tf_highs[batch_indices, best_tf_idx].unsqueeze(-1)
final_pred_low = per_tf_lows[batch_indices, best_tf_idx].unsqueeze(-1)

# AFTER:
# Convert probabilities to logits for Gumbel-Softmax
# per_tf_confs are sigmoid outputs [0, 1], need logits [-inf, inf]
per_tf_logits = torch.log(per_tf_confs.clamp(1e-6, 1-1e-6) / (1 - per_tf_confs.clamp(1e-6, 1-1e-6)))

# Soft selection with temperature annealing
temperature = getattr(self, 'selection_temperature', 1.0)
tf_weights = F.gumbel_softmax(per_tf_logits, tau=temperature, hard=False)  # [batch, 11]

# Weighted combination of all TF predictions
final_pred_high = (tf_weights * per_tf_highs).sum(dim=-1, keepdim=True)  # [batch, 1]
final_pred_low = (tf_weights * per_tf_lows).sum(dim=-1, keepdim=True)    # [batch, 1]
final_pred_conf = (tf_weights * per_tf_confs).sum(dim=-1, keepdim=True)  # [batch, 1]

# Store weights for loss computation
self._last_tf_weights = tf_weights
```

**Alternative (cleaner):** Modify confidence heads to output logits instead of applying sigmoid, then apply sigmoid only where probabilities are needed.

### Fix 5B: Add Temperature Annealing

**File:** `src/ml/hierarchical_model.py`

Add to `__init__`:
```python
self.selection_temperature = 2.0  # Start soft, anneal to hard
```

**File:** `train_hierarchical.py`

Add to epoch loop:
```python
# Anneal temperature: 2.0 → 0.5 over 10 epochs
if epoch < 10:
    model.selection_temperature = 2.0 - (epoch * 0.15)
else:
    model.selection_temperature = 0.5
```

### Fix 5C: Compute Per-TF Compositor Outputs During Training

**File:** `src/ml/hierarchical_model.py`

**Location:** ~line 1476-1494

**Problem:** Compositor requires a real `current_tf` (used at line 309: `all_hidden[current_tf]`). Setting `selected_tf = None` would crash.

**Solution:** During training, compute compositor for ALL timeframes. During inference, use argmax as before.

```python
# BEFORE:
best_tf_idx = torch.argmax(conf_tensor, dim=-1)[0].item()
selected_tf = self.TIMEFRAMES[best_tf_idx]

if hasattr(self, 'compositor'):
    compositor_output = self.compositor(
        all_hidden=tf_hidden_dict,
        hidden_vix=hidden_vix,
        event_embed=event_embed,
        current_tf=selected_tf,
        timeframes=self.TIMEFRAMES
    )
    output_dict['compositor'] = compositor_output
    output_dict['selected_tf'] = selected_tf

# AFTER:
if self.training:
    # Training: compute compositor for ALL timeframes (for multi-TF loss in Phase 9)
    if hasattr(self, 'compositor'):
        for tf in self.TIMEFRAMES:
            compositor_output = self.compositor(
                all_hidden=tf_hidden_dict,
                hidden_vix=hidden_vix,
                event_embed=event_embed,
                current_tf=tf,  # Each TF gets its own compositor output
                timeframes=self.TIMEFRAMES
            )
            output_dict[f'compositor_{tf}'] = compositor_output

        # Also store the "best" one for backwards compat with existing code
        best_tf_idx = torch.argmax(conf_tensor, dim=-1)[0].item()
        output_dict['selected_tf'] = self.TIMEFRAMES[best_tf_idx]
        output_dict['compositor'] = output_dict[f'compositor_{self.TIMEFRAMES[best_tf_idx]}']
else:
    # Inference: use argmax for discrete selection (original behavior)
    best_tf_idx = torch.argmax(conf_tensor, dim=-1)[0].item()
    selected_tf = self.TIMEFRAMES[best_tf_idx]

    if hasattr(self, 'compositor'):
        compositor_output = self.compositor(
            all_hidden=tf_hidden_dict,
            hidden_vix=hidden_vix,
            event_embed=event_embed,
            current_tf=selected_tf,
            timeframes=self.TIMEFRAMES
        )
        output_dict['compositor'] = compositor_output
        output_dict['selected_tf'] = selected_tf
```

**Note:** This adds 11 compositor forward passes during training. Each is small (~256 + 128 + 64 params), so overhead is minimal.

---

## Phase 6: Multi-TF Loss Contribution

### Problem
```
Current:  Only selected TF's predictions contribute to loss
Result:   10 out of 11 TFs get zero gradient each batch
```

### Fix 6A: Train All Timeframes Every Batch

**File:** `train_hierarchical.py`

Add after primary loss computation (~line 3595):

```python
# Multi-TF auxiliary loss: all timeframes contribute
if 'per_tf_predictions' in hidden_states:
    per_tf_highs = hidden_states['per_tf_predictions']['highs']  # [batch, 11]
    per_tf_lows = hidden_states['per_tf_predictions']['lows']    # [batch, 11]
    tf_weights = hidden_states.get('tf_weights', None)           # [batch, 11]

    if tf_weights is not None:
        # Confidence-weighted loss for each TF
        multi_tf_loss = 0.0
        for i in range(11):
            tf_high_loss = F.mse_loss(per_tf_highs[:, i], target_tensor[:, 0], reduction='none')
            tf_low_loss = F.mse_loss(per_tf_lows[:, i], target_tensor[:, 1], reduction='none')
            # Weight by confidence (detached to not affect confidence learning)
            weight = tf_weights[:, i].detach()
            multi_tf_loss += (weight * (tf_high_loss + tf_low_loss) / 2).mean()

        loss = loss + 0.1 * multi_tf_loss
        loss_components['multi_tf'] = multi_tf_loss.item()
```

### Fix 6B: Expose Per-TF Predictions in Hidden States

**File:** `src/ml/hierarchical_model.py`

Add to forward return (~line 1390):

```python
hidden_states['per_tf_predictions'] = {
    'highs': per_tf_highs,  # [batch, 11]
    'lows': per_tf_lows,    # [batch, 11]
    'confs': per_tf_confs,  # [batch, 11]
}
hidden_states['tf_weights'] = self._last_tf_weights  # From Gumbel-Softmax
```

---

## Phase 7: Entropy Regularization

### Problem
```
Even with soft selection, model may still prefer one TF
Need explicit encouragement for diversity
```

### Fix 7A: Add Entropy Loss Term

**File:** `train_hierarchical.py`

Add after multi-TF loss (~line 3610):

```python
# Entropy regularization: encourage diverse TF selection
if 'tf_weights' in hidden_states:
    tf_weights = hidden_states['tf_weights']  # [batch, 11]
    # Entropy = -sum(p * log(p)), higher = more uniform
    entropy = -(tf_weights * torch.log(tf_weights + 1e-8)).sum(dim=-1).mean()
    # Maximize entropy (subtract from loss)
    entropy_weight = get_loss_warmup_weight(epoch, WARMUP_EPOCHS, 0.05)
    loss = loss - entropy_weight * entropy
    loss_components['entropy'] = entropy.item()
```

---

## Phase 8: Validity Masks Fix (Critical Bug)

### Problem
```python
# Dataset provides validity flags:
'trans_5min_valid': 1,  # 1 = valid label, 0 = placeholder/invalid
'cont_5min_valid': 0,   # This label should NOT be trained on!

# Training ignores these flags:
transition_labels_dict[tf] = {
    'transition_type': targets[f'trans_{tf}_type'],  # Could be invalid!
    ...
}
```

Training on invalid/placeholder labels corrupts the compositor and biases selection.

### Fix 8A: Check Validity Before Building Label Dict

**File:** `train_hierarchical.py`

**Location:** ~lines 3774-3781

```python
# BEFORE:
for i, tf in enumerate(TIMEFRAMES):
    trans_type_key = f'trans_{tf}_type'
    if trans_type_key in targets:
        transition_labels_dict[tf] = {
            'transition_type': int(targets[trans_type_key][0].item()),
            'new_direction': int(targets.get(f'trans_{tf}_direction', ...)[0].item()),
        }

# AFTER:
for i, tf in enumerate(TIMEFRAMES):
    trans_type_key = f'trans_{tf}_type'
    trans_valid_key = f'trans_{tf}_valid'

    # Only include if label is valid
    if trans_type_key in targets and targets.get(trans_valid_key, torch.tensor([0]))[0].item() == 1:
        transition_labels_dict[tf] = {
            'transition_type': int(targets[trans_type_key][0].item()),
            'new_direction': int(targets.get(f'trans_{tf}_direction', ...)[0].item()),
        }
```

### Fix 8B: Same for Continuation Labels

```python
# Also check cont_*_valid flags when using continuation labels
cont_valid_key = f'cont_{tf}_valid'
if targets.get(cont_valid_key, torch.tensor([0]))[0].item() == 1:
    # Use continuation label
```

---

## Phase 9: Transition Compositor Fix

### Problem
```
Current:  Compositor only trains when selected_tf matches label
Result:   ~50% of batches have zero transition gradient
```

### Fix 9A: Always Train on Available Labels

**File:** `train_hierarchical.py`

**Location:** ~lines 3847-3873

```python
# BEFORE:
if selected_tf in transition_labels_dict:
    # Train only on selected TF
    ...

# AFTER:
# Train on ALL available transition labels (not just selected TF)
for tf, labels in transition_labels_dict.items():
    if f'compositor_{tf}' in hidden_states:
        compositor = hidden_states[f'compositor_{tf}']
        trans_type = labels['transition_type']
        direction = labels['new_direction']

        # Weight by TF confidence (more confident = more gradient)
        tf_idx = TIMEFRAMES.index(tf)
        tf_weight = hidden_states['tf_weights'][:, tf_idx].mean().detach()

        trans_loss = tf_weight * F.cross_entropy(compositor['transition_logits'], ...)
        dir_loss = tf_weight * F.cross_entropy(compositor['direction_logits'], ...)

        loss = loss + 0.3 * trans_loss + 0.2 * dir_loss
```

### Fix 9B: Per-TF Compositor Outputs Already Handled

**Note:** This is now handled by Fix 5C above. During training, we call:

```python
for tf in self.TIMEFRAMES:
    compositor_output = self.compositor(
        all_hidden=tf_hidden_dict,
        hidden_vix=hidden_vix,
        event_embed=event_embed,
        current_tf=tf,
        timeframes=self.TIMEFRAMES
    )
    output_dict[f'compositor_{tf}'] = compositor_output
```

The training loop in Phase 9A can then access `hidden_states[f'compositor_{tf}']` for any timeframe.

---

## Implementation Order

| Order | Phase | Description | Est. Time | Risk | Dependency |
|-------|-------|-------------|-----------|------|------------|
| 1 | Phase 1 | BCE/Sigmoid mismatch fix | 15 min | Low | None |
| 2 | Phase 2 | Log geo_price loss | 5 min | None | None |
| 3 | Phase 4 | Clamp duration for geo projection | 10 min | Low | None |
| 4 | Phase 8 | Validity masks fix | 15 min | Low | None |
| 5 | Phase 3 | Loss warmup | 20 min | Low | None |
| 6 | Phase 5 | Soft selection (Gumbel-Softmax) | 45 min | Medium | None |
| 7 | Phase 6 | Multi-TF loss | 20 min | Low | Phase 5 |
| 8 | Phase 7 | Entropy regularization | 10 min | Low | Phase 5 |
| 9 | Phase 9 | Compositor multi-TF training | 30 min | Medium | Phase 5 |

**Total estimated time:** ~2.5-3 hours

---

## Verification Checklist

After implementation, verify:

- [ ] No more sigmoid(sigmoid(x)) - BCE loss uses logits correctly
- [ ] Loss components sum ≈ total loss (no hidden components)
- [ ] Geo-price loss is reasonable (< 10, not 2200+) in epoch 1
- [ ] TF selection diversity > 0% for non-4h timeframes by epoch 3
- [ ] Entropy value logged and positive
- [ ] Invalid labels (trans_*_valid=0) are NOT trained on
- [ ] Training loss decreases over epochs
- [ ] No NaN/Inf in any loss component

---

## Rollback Plan

If issues arise:
1. Git revert to pre-change commit
2. All changes are in 2 files: `hierarchical_model.py`, `train_hierarchical.py`
3. No data format changes (shards unchanged)
4. No config changes required

---

## Files Modified

| File | Changes |
|------|---------|
| `src/ml/hierarchical_model.py` | Remove sigmoid from heads, clamp duration, soft selection, per-TF storage (~60 lines) |
| `train_hierarchical.py` | Warmup, logging, validity checks, multi-TF loss, entropy (~100 lines) |

---

## Key Corrections From Peer Review

| Original Diagnosis | Corrected Understanding |
|-------------------|------------------------|
| "Scale mismatch (100×)" | NOT a bug - code uses percentage-points consistently |
| Gumbel-Softmax on probabilities | WRONG - needs logits, added conversion |
| Missing: BCE mismatch | FOUND - sigmoid heads + BCEWithLogits = double sigmoid |
| Missing: Validity masks | FOUND - training ignores *_valid flags |
| Missing: Duration clamping | FOUND - unclamped duration causes geo explosion |

---

## Expected Outcomes

| Metric | Before | After |
|--------|--------|-------|
| BCE/Sigmoid bug | Double sigmoid | Correct logits |
| Geo-price loss epoch 1 | ~2200 (explosion) | ~5-20 (clamped) |
| Loss visibility | 6 of 8 components shown | All 8+ shown |
| Loss warmup | All losses full from epoch 1 | Gradual ramp-up |
| TF diversity | 0% (100% 4h) | 30-70% multi-TF |
| Training stability | Erratic | Smooth convergence |

---

## Approval Required

This plan modifies core training logic. Please review and confirm before implementation.

**To approve:** Reply with "approved" or provide feedback.
