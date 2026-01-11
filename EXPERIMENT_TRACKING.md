# Experiment Tracking Document
> This document tracks all hyperparameter experiments and configurations for the x7 trading model.
> Created: 2026-01-10

> **Note (2026-01-11):** New multi-task learning features have been added including gradient balancing methods (GradNorm, PCGrad), two-stage training, advanced loss functions, and architecture options (TCN, multi-resolution heads). See the "New Multi-Task Learning Features" section below.

## New Multi-Task Learning Features (2026-01-11)

### Gradient Balancing Methods

Control how gradients from multiple tasks (direction, duration, etc.) are balanced during training:

| Method | Flag | Description |
|--------|------|-------------|
| `none` | `--gradient-balancing none` | Standard gradient descent - all task gradients summed directly |
| `gradnorm` | `--gradient-balancing gradnorm --gradnorm-alpha 1.5` | GradNorm adaptive task weighting. Dynamically adjusts task weights based on training rates. Alpha parameter (default 1.5) controls how aggressively weights are balanced |
| `pcgrad` | `--gradient-balancing pcgrad` | PCGrad conflicting gradient projection. Projects conflicting gradients to reduce interference between tasks |

### Two-Stage Training

Enables pretraining on a single task before joint multi-task training:

| Parameter | Flag | Description |
|-----------|------|-------------|
| `stage1_epochs` | `--stage1-epochs N` | Number of epochs for stage 1 pretraining (default: 0, disabled) |
| `stage1_task` | `--stage1-task [direction\|duration]` | Primary task for stage 1 (default: direction) |

**How it works:**
1. Stage 1: Train only on the specified task (e.g., direction) for N epochs
2. Stage 2: Continue with full multi-task training using all loss components
3. This can help establish a strong feature representation before balancing multiple objectives

### Loss Function Options

#### Duration Loss Functions
| Loss | Flag | Description |
|------|------|-------------|
| `gaussian_nll` | `--duration-loss gaussian_nll` | **Default.** Gaussian negative log-likelihood - learns both mean and variance |
| `huber` | `--duration-loss huber --huber-delta 1.0` | Huber loss - robust to outliers. Delta parameter controls transition from L2 to L1 |
| `survival` | `--duration-loss survival` | Hazard-based survival loss - treats channel break as time-to-event prediction |

#### Direction Loss Functions
| Loss | Flag | Description |
|------|------|-------------|
| `bce` | `--direction-loss bce` | **Default.** Binary cross-entropy |
| `focal` | `--direction-loss focal --focal-gamma 2.0` | Focal loss - down-weights easy examples, focuses on hard cases. Gamma controls focusing strength |

### Architecture Options

#### TCN (Temporal Convolutional Network) Block
Adds dilated causal convolutions to better capture temporal patterns in channel dynamics:

| Parameter | Flag | Description |
|-----------|------|-------------|
| `use_tcn` | `--use-tcn` | Enable TCN block in the encoder |
| `tcn_channels` | `--tcn-channels 64` | Number of TCN hidden channels (default: 64) |
| `tcn_kernel_size` | `--tcn-kernel-size 3` | Kernel size for TCN convolutions (default: 3) |
| `tcn_layers` | `--tcn-layers 4` | Number of TCN layers (default: 4) |

#### Multi-Resolution Heads
Enables task heads to attend to different temporal resolutions:

| Parameter | Flag | Description |
|-----------|------|-------------|
| `use_multi_resolution` | `--use-multi-resolution` | Enable multi-resolution attention in prediction heads |
| `resolution_levels` | `--resolution-levels 3` | Number of resolution levels (default: 3) |

**How it works:** Duration head attends more to fine-grained (short window) features for precise timing, while direction head focuses on longer context for trend identification.

### Example Command with New Features

```bash
# Full-featured training with GradNorm, two-stage training, and advanced options
python3 train.py --no-interactive \
    --run-name mtl_experiment \
    --hidden-dim 64 \
    --cfc-units 96 \
    --attention-heads 4 \
    --se-blocks \
    --se-ratio 8 \
    --dropout 0.2 \
    --epochs 30 \
    --batch-size 64 \
    --lr 0.001 \
    --weight-direction 4.0 \
    --gradient-balancing gradnorm \
    --gradnorm-alpha 1.5 \
    --stage1-epochs 5 \
    --stage1-task direction \
    --duration-loss huber \
    --huber-delta 1.0 \
    --direction-loss focal \
    --focal-gamma 2.0 \
    --use-tcn \
    --tcn-channels 64 \
    --tcn-layers 4 \
    --use-multi-resolution \
    --resolution-levels 3 \
    --step 25 \
    --train-end 2024-01-01 \
    --val-end 2024-12-31
```

---

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

## Duration-Focused Experiments (2026-01-10)

Experiments to improve duration prediction while maintaining direction accuracy.

| Exp | Name | Config | Dir Acc | Duration Val | Notes |
|-----|------|--------|---------|--------------|-------|
| EXP17b | Huber dur_w=4.0 | `--duration-loss huber --weight-duration 4.0` | 60.7% | 5.65 | Huber slightly worse than Gaussian NLL |
| EXP18b | Survival Loss | `--duration-loss survival` | FAILED | - | Tensor shape mismatch - model needs duration_hazard output |
| EXP19c | Two-Stage (dur first) | `--two-stage-training --stage1-task duration --stage1-epochs 5` | 59.3% | 2.68 | Duration pretraining didn't help |
| **EXP20d** | **Huber dur_w=5.0** | `--duration-loss huber --weight-duration 5.0` | **62.6%** | 5.59 | **Matches best! Duration focus doesn't hurt direction** |

### Key Findings (Duration Experiments):

1. **Huber loss with duration_weight=5.0 matches best direction accuracy (62.6%)** while potentially improving duration
2. **Two-stage duration pretraining hurts performance** - 59.3% vs 62.6% baseline
3. **Survival loss needs model changes** - requires duration_hazard logits output
4. **Duration weight up to 5.0 doesn't hurt direction** - can focus on duration without sacrificing direction accuracy

### Recommended Duration Configuration:

```bash
python3 train.py --no-interactive \
    --run-name duration_optimized \
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
    --weight-duration 5.0 \
    --duration-loss huber \
    --huber-delta 1.0 \
    --step 25 \
    --train-end 2024-01-01 \
    --val-end 2024-12-31
```

---

## Next Experiments to Try

1. [x] **Huber loss for duration**: Better than Gaussian NLL? - Tested, works well
2. [x] **Two-stage training**: Pretrain on duration - Tested, doesn't help
3. [x] **Survival loss**: Needs model changes for hazard output
4. [ ] **Duration weight=6.0-8.0**: Can we push duration focus further?
5. [ ] **Lower huber delta (0.5)**: More aggressive outlier handling
6. [ ] **Auxiliary duration targets**: Add distance-to-boundary features
7. [ ] **Full walk-forward validation**: Test best config on proper temporal splits

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
| 2026-01-11 | Added multi-task learning features: gradient balancing (GradNorm, PCGrad), two-stage training, loss functions (Huber, survival, focal), TCN blocks, multi-resolution heads |
| 2026-01-10 | Initial experiments, found SE-blocks + dir_weight=4.0 optimal |
