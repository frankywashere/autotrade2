# CLI Training Guide

Non-interactive command-line training for use on Colab, remote servers, or automated pipelines.

## Quick Start

```bash
# View all options
python train.py --help

# Quick test (10 epochs, small model)
python train.py --mode quick --no-interactive

# Standard training (50 epochs, balanced)
python train.py --mode standard --no-interactive

# Full training (100 epochs, large model)
python train.py --preset full --no-interactive
```

## Presets

| Preset | Epochs | Hidden | CfC | Batch | LR |
|--------|--------|--------|-----|-------|------|
| `quick` | 10 | 64 | 96 | 32 | 0.001 |
| `standard` | 50 | 128 | 192 | 64 | 0.0005 |
| `full` | 100 | 256 | 384 | 128 | 0.0003 |

## Common Usage Patterns

### Basic Training with Preset
```bash
python train.py --mode standard --no-interactive
```

### Override Preset Values
```bash
python train.py --mode quick --epochs 20 --batch-size 128 --no-interactive
```

### Walk-Forward Validation
```bash
python train.py --mode walk-forward \
  --wf-windows 5 \
  --wf-val-months 3 \
  --no-interactive
```

### Custom Model Architecture
```bash
python train.py --no-interactive \
  --hidden-dim 256 \
  --cfc-units 384 \
  --attention-heads 8 \
  --se-blocks \
  --epochs 100
```

### Full Custom (Colab Example)
```bash
python train.py --no-interactive \
  --mode standard \
  --run-name "colab_exp_001" \
  --epochs 100 \
  --batch-size 64 \
  --lr 0.0005 \
  --hidden-dim 128 \
  --cfc-units 192 \
  --attention-heads 8 \
  --device cuda \
  --early-stopping 15 \
  --early-stopping-metric duration
```

### Advanced Multi-Task Learning
```bash
# With GradNorm gradient balancing and focal loss
python train.py --no-interactive \
  --run-name "mtl_gradnorm" \
  --hidden-dim 64 \
  --se-blocks \
  --gradient-balancing gradnorm \
  --gradnorm-alpha 1.5 \
  --direction-loss focal \
  --focal-gamma 2.0 \
  --epochs 50
```

### Two-Stage Training
```bash
# Pretrain on direction for 10 epochs, then joint training
python train.py --no-interactive \
  --run-name "two_stage_exp" \
  --two-stage-training \
  --stage1-epochs 10 \
  --stage1-task direction \
  --epochs 50
```

### With TCN and Multi-Resolution Heads
```bash
python train.py --no-interactive \
  --run-name "tcn_multires" \
  --hidden-dim 64 \
  --se-blocks \
  --use-tcn \
  --tcn-channels 64 \
  --tcn-layers 3 \
  --use-multi-resolution \
  --resolution-levels 3 \
  --epochs 50
```

### Best Known Configuration (from experiments)
```bash
python train.py --no-interactive \
  --run-name "best_config" \
  --hidden-dim 64 \
  --cfc-units 96 \
  --attention-heads 4 \
  --se-blocks \
  --dropout 0.2 \
  --weight-direction 4.0 \
  --epochs 20 \
  --batch-size 64 \
  --lr 0.001 \
  --step 25 \
  --train-end 2024-01-01 \
  --val-end 2024-12-31
```

## Key Arguments Reference

### Mode & Run
| Argument | Description |
|----------|-------------|
| `--mode` | `quick`, `standard`, `full`, `walk-forward`, `custom` |
| `--preset` | Alias for mode (same options) |
| `--run-name NAME` | Optional name for the run directory |
| `--no-interactive` | **Required** to skip menus |

### Model Architecture
| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden-dim` | 128 | Hidden dimension (must be divisible by attention heads) |
| `--cfc-units` | 192 | CfC units (must be > hidden_dim + 2) |
| `--attention-heads` | 8 | Choices: 2, 4, 8, 16 |
| `--dropout` | 0.1 | Choices: 0.0, 0.1, 0.2, 0.3 |
| `--se-blocks` | off | Enable Squeeze-and-Excitation blocks |
| `--se-ratio` | 8 | SE-block reduction ratio: 4, 8, or 16 |
| `--shared-heads` | off | Use shared prediction heads (fewer params) |

### TCN (Temporal Convolutional Network)
| Argument | Default | Description |
|----------|---------|-------------|
| `--use-tcn` | off | Enable TCN block after CfC layer |
| `--tcn-channels` | 64 | Number of TCN hidden channels |
| `--tcn-kernel-size` | 3 | TCN convolution kernel size |
| `--tcn-layers` | 2 | Number of TCN layers (dilations: 1, 2, 4, ...) |

### Multi-Resolution Heads
| Argument | Default | Description |
|----------|---------|-------------|
| `--use-multi-resolution` | off | Enable multi-resolution prediction heads |
| `--resolution-levels` | 3 | Number of resolution levels (2-4) |

### Training
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 64 | Choices: 16, 32, 64, 128, 256 |
| `--lr` | 0.001 | Learning rate |
| `--optimizer` | adamw | Choices: adam, adamw, sgd |
| `--scheduler` | cosine_restarts | Choices: cosine_restarts, cosine, step, plateau, none |
| `--early-stopping` | 15 | Patience (0 to disable) |
| `--early-stopping-metric` | duration | Choices: duration, total, direction_acc, next_channel_acc |
| `--use-amp` | off | Enable mixed precision (faster but less stable) |

### Gradient Balancing (Multi-Task Learning)
| Argument | Default | Description |
|----------|---------|-------------|
| `--gradient-balancing` | none | Choices: none, gradnorm, pcgrad |
| `--gradnorm-alpha` | 1.5 | GradNorm alpha (higher = more aggressive balancing) |

**Methods:**
- `none`: Standard gradient descent - all task gradients summed directly
- `gradnorm`: Dynamically adjusts task weights based on training rates
- `pcgrad`: Projects conflicting gradients to reduce task interference

### Two-Stage Training
| Argument | Default | Description |
|----------|---------|-------------|
| `--two-stage-training` | off | Enable two-stage training |
| `--stage1-epochs` | 5 | Number of epochs for stage 1 pretraining |
| `--stage1-task` | direction | Primary task for stage 1: direction or duration |

**How it works:** Train primarily on one task first, then continue with full multi-task training.

### Loss Functions
| Argument | Default | Description |
|----------|---------|-------------|
| `--duration-loss` | gaussian_nll | Choices: gaussian_nll, huber, survival |
| `--huber-delta` | 1.0 | Delta for Huber loss (transition point) |
| `--direction-loss` | bce | Choices: bce, focal |
| `--focal-gamma` | 2.0 | Gamma for Focal loss (focusing strength) |

**Duration losses:**
- `gaussian_nll`: Default. Learns both mean and variance
- `huber`: Robust to outliers
- `survival`: Hazard-based time-to-event modeling

**Direction losses:**
- `bce`: Default. Binary cross-entropy
- `focal`: Down-weights easy examples, focuses on hard cases

### Loss Weights
| Argument | Default | Description |
|----------|---------|-------------|
| `--weight-mode` | fixed_duration_focus | Choices: learnable, fixed_duration_focus, fixed_balanced, fixed_custom |
| `--weight-duration` | 2.5 | Duration task weight (with fixed_custom) |
| `--weight-direction` | 1.0 | Direction task weight (with fixed_custom) |
| `--weight-next-channel` | 0.8 | Next channel task weight (with fixed_custom) |
| `--weight-trigger-tf` | 1.5 | Trigger TF task weight (with fixed_custom) |
| `--weight-calibration` | 0.5 | Calibration task weight (with fixed_custom) |

**Tip:** Increasing `--weight-direction` (e.g., to 4.0) improved direction accuracy significantly in experiments.

### Data
| Argument | Default | Description |
|----------|---------|-------------|
| `--step` | 25 | Sliding window step (1-100) |
| `--train-end` | 70% | Training split end date (YYYY-MM-DD) |
| `--val-end` | 85% | Validation split end date (YYYY-MM-DD) |
| `--window-strategy` | learned_selection | Choices: learned_selection, bounce_first, label_validity, balanced_score, quality_score |
| `--include-history` | on | Include channel history features |

### Walk-Forward
| Argument | Default | Description |
|----------|---------|-------------|
| `--wf-windows` | 3 | Number of validation windows (2-10) |
| `--wf-val-months` | 3 | Validation period per window (1-12 months) |
| `--wf-type` | expanding | Choices: expanding, sliding |
| `--wf-train-months` | 12 | Training window for sliding type (3-36 months) |

### Device
| Argument | Default | Description |
|----------|---------|-------------|
| `--device` | auto | Choices: cuda, mps, cpu |

## Colab Cell Example

```python
# In a Colab notebook cell:
!python train.py --no-interactive \
  --preset standard \
  --run-name "colab_run" \
  --epochs 50 \
  --device cuda
```

## Interactive Mode

Running without `--no-interactive` launches the original menu-based interface:

```bash
python train.py  # Opens interactive menus
```

## Output

Training runs are saved to `runs/{timestamp}_{run_name}/` with:
- `run_config.json` - Full configuration
- `windows/` - Model checkpoints
- `logs/` - Training logs
