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
| EXP18b | Survival Loss | `--duration-loss survival` | FIXED | - | Now working after adding duration_hazard output |
| EXP19c | Two-Stage (dur first) | `--two-stage-training --stage1-task duration --stage1-epochs 5` | 59.3% | 2.68 | Duration pretraining didn't help |
| EXP20d | Huber dur_w=5.0 | `--duration-loss huber --weight-duration 5.0` | 62.6% | 5.59 | Good but not best |
| EXP21 | Baseline (Gaussian NLL) | `--duration-loss gaussian_nll --weight-duration 2.5` | 61.8% | 2.54 | Baseline with duration metrics |
| EXP22 | Survival Loss | `--duration-loss survival --weight-duration 5.0` | 63.7% | 2.39 | Great baseline with survival |
| EXP23 | Huber delta=0.5 | `--duration-loss huber --huber-delta 0.5 --weight-duration 5.0` | 62.3% | 2.85 | Smaller delta didn't help |
| EXP24 | Huber dur_w=8.0 | `--duration-loss huber --weight-duration 8.0` | 63.3% | 5.56 | High weight decent direction but worse duration |
| EXP26 | Survival + TCN | `--duration-loss survival --use-tcn --tcn-kernel-size 5 --tcn-layers 3` | 64.0% | 2.38 | TCN helps direction slightly |
| **EXP27** | **Survival + Learnable** | `--duration-loss survival --weight-mode learnable --min-duration-precision 0.4` | **64.8%** | **2.39** | **BUGGY CODE** (before survival fix) |
| EXP27_S1-5 | Multi-seed (fixed code) | 5 seeds with fixed survival NLL | **60.7% ± 1.7%** | **6.28 ± 0.06 bars MAE** | **True stable performance** |
| EXP28 | Survival + TCN + Learnable | Combined EXP26 + EXP27 | 64.2% | 2.40 | Worse than learnable alone - redundancy |
| EXP29 | Early Stopping pat=2 | `--early-stopping 2 --epochs 8` | 63.9% | 2.37 | Stopped at epoch 5, slightly worse |
| EXP30 | Lower LR | `--lr 0.0003 --epochs 40` | CRASHED | - | Memory/segfault |
| EXP31 | Shared + PCGrad | `--shared-heads --gradient-balancing pcgrad` | 43.4% | - | Complete failure |
| WF_W1 | Walk-forward Window 1 (buggy) | EXP27 config, 3-month val window | 61.2% | 2.31 | Crashed after Window 1 |
| **WF_FINAL** | **Walk-forward 3 windows (fixed)** | **Batch=32, expanding windows** | **67.45% ± 4.24%** | **5.82 ± 0.18 bars MAE** | **PRIMARY ESTIMATE** |

### Walk-Forward Detailed Results (Fixed Survival Loss):

| Window | Val Period | Direction Acc | Duration MAE | Duration Loss | Best Epoch |
|--------|------------|---------------|--------------|---------------|------------|
| 1      | 2024-12 → 2025-03 | 63.5% | 5.59 bars | 1.96 | 10 |
| 2      | 2025-03 → 2025-06 | 73.3% | 6.02 bars | 1.95 | 19 |
| 3      | 2025-06 → 2025-09 | 65.5% | 5.86 bars | 1.92 | 5 |
| **Average** | **All windows** | **67.45% ± 4.24%** | **5.82 ± 0.18 bars** | **1.94 ± 0.02** | **11.3** |

### Key Findings (Duration Experiments):

1. **Walk-forward is PRIMARY estimate: 67.45% ± 4.24% direction, 5.82 ± 0.18 bars MAE**
2. **Multi-seed stable performance: 60.7% ± 1.7% direction** (single split)
3. **Critical bug fixed**: Survival NLL now uses S(t-1) for correct likelihood
4. **Duration metrics now meaningful**: MAE computed from hazard-derived expected duration
5. **Survival loss models time-to-event better than Huber/Gaussian**
6. **Learnable task weights auto-balance better than fixed weights**
7. **TCN adds temporal patterns: 64.0% direction** - but combining with learnable hurts (redundancy)
8. **Shared heads + PCGrad completely failed** - 43.4% direction (do NOT use)
9. **Two-stage duration pretraining hurts performance**
10. **Models peak very early (epoch 2-11) then overfit** - use early stopping
11. **Walk-forward variance is higher** - samples different market regimes (more realistic)

### Multi-Seed Results (EXP27 with Fixed Survival Loss):

| Seed | Direction Acc | Duration MAE | Duration Loss | Best Epoch |
|------|---------------|--------------|---------------|------------|
| 42   | 59.6%         | 6.30 bars    | 2.0411        | 3          |
| 123  | 63.8%         | 6.23 bars    | 2.0430        | 5          |
| 456  | 59.7%         | 6.38 bars    | 2.0461        | 2          |
| 789  | 59.5%         | 6.26 bars    | 2.0419        | 3          |
| 999  | 61.1%         | 6.23 bars    | 2.0437        | 3          |
| **Mean** | **60.7% ± 1.7%** | **6.28 ± 0.06 bars** | **2.0432 ± 0.002** | **3.2** |

**Bug Fix Impact:**
- Original EXP27 (buggy survival NLL): 64.8% direction, 2.39 duration loss (no MAE)
- Fixed survival NLL: 60.7% direction, 2.04 duration loss, **6.28 bars MAE**
- Trade-off: Fixing the bug made duration better but direction worse
- Why: Fixed NLL strengthens duration gradients, trades off with direction in shared capacity

### Weight Grid Results (min_duration_precision Tuning):

| min_dur_precision | Direction Acc | Duration MAE | Best Epoch | vs Baseline Dir | vs Baseline MAE |
|-------------------|---------------|--------------|------------|-----------------|-----------------|
| 0.3 | 61.1% | 6.36 bars | 1 | +0.4% | +0.08 bars |
| **0.4** | **63.7%** | 6.43 bars | 4 | **+3.0%** | +0.15 bars |
| 0.5 | 59.1% | 6.31 bars | 3 | -1.6% | +0.03 bars |
| 0.6 | 58.3% | 6.27 bars | 4 | -2.4% | -0.01 bars |

**Finding:** min_duration_precision=0.4 gives **best direction (63.7%)** with acceptable duration MAE trade-off (+0.15 bars)

**Codex Insight:** Clear direction/duration trade-off curve. 0.6 minimizes MAE but hurts direction. 0.4 is optimal for direction-focused use case. Need multi-seed validation to confirm (single runs have ~1.7% noise).

### Codex Recommendations for Next Steps:

1. [x] **Multi-seed validation** - DONE: 60.7% ± 1.7% stable (not 64.8%)
2. [x] **Walk-forward validation with batch=32** - DONE: 67.45% ± 4.24% (PRIMARY)
3. [x] **Tune task weights** - DONE: 0.4 optimal for direction
4. [ ] **Multi-seed for 0.4** - Verify 63.7% holds across seeds
5. [ ] **Try intermediate values (0.35, 0.45)** - Find precise knee of curve

### Recommended Duration Configuration (BEST - EXP27):

**Full Configuration (from run_config.json on server):**

```bash
python3 train.py --no-interactive \
    --run-name exp27_survival_learnable \
    --mode standard \
    --device cuda \
    --hidden-dim 64 \
    --cfc-units 96 \
    --attention-heads 4 \
    --se-blocks \
    --se-ratio 8 \
    --dropout 0.2 \
    --shared-heads false \
    --use-tcn false \
    --tcn-channels 64 \
    --tcn-layers 2 \
    --tcn-kernel-size 3 \
    --epochs 20 \
    --batch-size 64 \
    --lr 0.001 \
    --optimizer adamw \
    --weight-decay 0.0001 \
    --gradient-clip 1.0 \
    --scheduler cosine_restarts \
    --duration-loss survival \
    --direction-loss bce \
    --weight-mode learnable \
    --min-duration-precision 0.4 \
    --calibration-mode brier_per_tf \
    --uncertainty-penalty 0.1 \
    --focal-gamma 2.0 \
    --huber-delta 1.0 \
    --gradient-balancing none \
    --early-stopping 15 \
    --early-stopping-metric duration \
    --two-stage-training false \
    --step 25 \
    --train-end 2024-01-01 \
    --val-end 2024-12-31 \
    --window-selection-strategy learned_selection
```

**Results:** 64.8% direction accuracy, 2.39 duration loss (best epoch 3/20) - BUGGY CODE

---

### Walk-Forward Validation Configuration (PRIMARY ESTIMATE):

**Full Configuration (from run_config.json on server):**

```bash
python3 train.py --no-interactive \
    --mode walk-forward \
    --run-name exp_wf_fixed_batch32 \
    --device cuda \
    --wf-windows 3 \
    --wf-val-months 3 \
    --wf-type expanding \
    --hidden-dim 64 \
    --cfc-units 96 \
    --attention-heads 4 \
    --se-blocks \
    --se-ratio 8 \
    --dropout 0.2 \
    --shared-heads false \
    --use-tcn false \
    --tcn-channels 64 \
    --tcn-layers 2 \
    --tcn-kernel-size 3 \
    --batch-size 32 \
    --epochs 20 \
    --lr 0.001 \
    --optimizer adamw \
    --weight-decay 0.0001 \
    --gradient-clip 1.0 \
    --scheduler cosine_restarts \
    --duration-loss survival \
    --direction-loss bce \
    --weight-mode learnable \
    --min-duration-precision 0.4 \
    --calibration-mode brier_per_tf \
    --uncertainty-penalty 0.1 \
    --focal-gamma 2.0 \
    --huber-delta 1.0 \
    --gradient-balancing none \
    --early-stopping 15 \
    --early-stopping-metric duration \
    --two-stage-training false \
    --step 25 \
    --window-selection-strategy learned_selection
```

**Results:** 67.45% ± 4.24% direction, 5.82 ± 0.18 bars MAE (3 windows avg)

**Key Differences from Single-Split:**
- Mode: `walk-forward` (not `standard`)
- Batch size: `32` (not `64`) - to avoid memory crashes
- Validation: 3 rolling windows of 3 months each
- Window type: `expanding` (training data grows each window)

---

## Next Experiments to Try

1. [x] **Huber loss for duration**: Tested - not better than survival
2. [x] **Two-stage training**: Tested - doesn't help
3. [x] **Survival loss**: 63.7% direction, 2.39 duration
4. [x] **Duration weight=8.0**: Tested - helps direction but not duration
5. [x] **Lower huber delta (0.5)**: Tested - didn't help
6. [x] **Survival + TCN**: 64.0% direction - helps!
7. [x] **Survival + learnable weights**: **64.8% direction - BEST!**
8. [ ] **Survival + TCN + learnable**: Combine both improvements
9. [ ] **Full walk-forward validation**: Test best config on proper temporal splits

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
