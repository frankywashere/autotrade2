# Experiment Tracking Document
> This document tracks all hyperparameter experiments and configurations for the x7 trading model.
> Created: 2026-01-10

## Current Best Configuration (EXP8)

```python
# Best performing configuration as of 2026-01-10
best_config = {
    # Architecture
    "hidden_dim": 64,
    "cfc_units": 96,
    "num_attention_heads": 4,
    "dropout": 0.2,
    "use_se_blocks": True,
    "se_reduction_ratio": 8,

    # Training
    "epochs": 20,
    "batch_size": 64,
    "learning_rate": 0.001,
    "optimizer": "adamw",
    "scheduler": "cosine_restarts",
    "weight_decay": 0.0001,
    "gradient_clip": 1.0,

    # Loss Weights (key finding: direction weight matters!)
    "weight_duration": 2.5,
    "weight_direction": 4.0,  # Increased from 1.0 - major improvement
    "weight_next_channel": 0.8,
    "weight_trigger_tf": 1.5,
    "weight_calibration": 0.5,

    # Data
    "step": 25,
    "windows": [10, 20, 30, 40, 50, 60, 70, 80],
}
```

**Result: 62.64% direction accuracy** (vs 54.57% baseline)

---

## All Experiment Results (2026-01-10)

| Exp | Name | Changes from Baseline | Dir Acc | Val Loss | Best Epoch | Notes |
|-----|------|----------------------|---------|----------|------------|-------|
| EXP2 | Baseline | HD=64, LR=0.001, dropout=0.1, dir_weight=1.0 | 54.57% | 7.7183 | 8/10 | Reference |
| EXP3 | Higher LR | LR=0.002 | 55.24% | 7.7470 | 4/10 | Slightly better, faster convergence |
| EXP4 | Larger Model | HD=128, heads=8 | 52.89% | 7.7813 | 1/10 | Overfit immediately - too much capacity |
| EXP5 | SE-blocks | use_se_blocks=True | 58.18% | 7.7658 | 8/10 | Significant improvement! |
| EXP6 | Higher Dropout | dropout=0.2 | 57.50% | ~7.88 | 6/10 | Helps regularization |
| EXP7 | Combo v1 | SE + dropout=0.2 + dir_weight=2.0 | 59.0% | - | - | Good improvement |
| **EXP8** | **Combo v2** | **SE + dropout=0.2 + dir_weight=4.0** | **62.64%** | 7.7818 | 5/20 | **BEST** |

---

## Key Findings

### What Works:
1. **SE-blocks (Squeeze-and-Excitation)**: +3.6% direction accuracy over baseline
2. **Higher direction weight (4.0x)**: Major improvement when combined with SE-blocks
3. **Dropout 0.2**: Better than 0.1 for regularization
4. **Early stopping around epoch 5-8**: Models peak early, then overfit

### What Doesn't Work:
1. **Larger models (HD=128)**: Overfit at epoch 1 - capacity not the bottleneck
2. **Higher LR alone**: Minor improvement, not enough

### Trade-offs Identified:
- Direction accuracy improved but val loss stayed similar (~7.7-7.8)
- Suggests multi-task objectives conflict (duration vs direction)
- Duration prediction may be undertrained when direction weight is high

---

## Codex Analysis & Recommendations (2026-01-10)

### For Improving Both Direction AND Duration:

1. **Shared encoder + dual heads**: Keep shared CfC/attention encoder, use specialized heads
2. **Temporal density block**: Add TCN/dilated conv to capture break dynamics
3. **Uncertainty-aware fusion**: Learned task weights (Kendall & Gal method)
4. **Multi-resolution**: Duration head attends to finer windows, direction to longer context

### For Duration Specifically:

1. **Survival/hazard modeling**: Treat channel break as time-to-event
2. **Quantile regression**: Predict P10/P50/P90 time-to-break
3. **Auxiliary targets**: "distance to boundary", "volatility-adjusted distance"
4. **Event-focused sampling**: Oversample windows where breaks occur soon

### Loss Function Recommendations:

- **Direction**: Cross-entropy or focal loss (if class imbalance)
- **Duration**: Huber loss or survival NLL (better than MSE for heavy tails)
- **Dynamic weighting**: GradNorm or PCGrad to reduce gradient conflict

---

## Model Architecture Reference

```
EndToEndWindowModel
├── window_encoder (per window)
│   ├── Feature projection
│   ├── CfC layers (Closed-form Continuous-time)
│   ├── SE-blocks (Squeeze-and-Excitation) [RECOMMENDED]
│   └── Multi-head attention
├── window_selector
│   └── Learns which windows are most informative
├── hierarchical_model
│   └── Combines window representations
└── prediction_heads (separate per timeframe)
    ├── duration_head
    ├── direction_head
    ├── next_channel_head
    └── trigger_tf_head
```

---

## Data Configuration

- **Cache location**: `data/feature_cache/channel_samples.pkl`
- **Sample date range**: 2016-01-27 to 2025-07-30
- **Total samples**: ~15,965 cached (after filtering)
- **Train/Val split**: train_end=2024-01-01, val_end=2024-12-31
- **Windows**: [10, 20, 30, 40, 50, 60, 70, 80] minute lookbacks
- **Step**: 25 (sample every 25 bars)

---

## Next Experiments to Try

1. [ ] **direction_weight=8.0**: See if even higher helps
2. [ ] **Survival loss for duration**: Replace MSE with hazard-based loss
3. [ ] **GradNorm**: Dynamic task weighting to balance direction/duration
4. [ ] **Two-stage training**: Pretrain on direction, then fine-tune jointly
5. [ ] **Auxiliary duration targets**: Add distance-to-boundary features
6. [ ] **Full walk-forward validation**: Test best config on proper temporal splits

---

## Commands to Reproduce Best Result

```bash
# On vast.ai instance
cd /workspace/autotrade2
python3 train.py --no-interactive \
    --run-name best_config \
    --hidden-dim 64 \
    --cfc-units 96 \
    --attention-heads 4 \
    --se-blocks \
    --se-ratio 8 \
    --dropout 0.2 \
    --epochs 20 \
    --batch-size 64 \
    --lr 0.001 \
    --weight-direction 4.0 \
    --step 25 \
    --train-end 2024-01-01 \
    --val-end 2024-12-31
```

---

## Version History

| Date | Changes |
|------|---------|
| 2026-01-10 | Initial experiments, found SE-blocks + dir_weight=4.0 optimal |
