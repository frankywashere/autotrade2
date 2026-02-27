# Channel Prediction Training Pipeline

Complete PyTorch training pipeline for the channel prediction system.

## Overview

This training pipeline provides:

1. **ChannelDataset** - PyTorch Dataset for loading channel samples
2. **Data Preparation** - Scanning, caching, and splitting historical data
3. **Trainer** - Complete training loop with multi-task learning
4. **Example Scripts** - Ready-to-run training examples

## Architecture

```
training/
├── dataset.py           # PyTorch Dataset and data preparation
├── trainer.py           # Training loop and utilities
├── labels.py            # Label generation (already exists)
├── example_training.py  # Complete working example
└── README.md           # This file
```

## Quick Start

### 1. Prepare Data

The pipeline expects data in `/Volumes/NVME2/x6/data/`:
- `TSLA_1min.csv` - TSLA 1-minute OHLCV data
- `SPY_1min.csv` - SPY 1-minute OHLCV data
- `VIX_History.csv` - VIX daily data

### 2. Run Example Training

```bash
cd /Volumes/NVME2/x6/v7/training
python example_training.py
```

This will:
1. Load and resample data to 5-minute bars
2. Scan for valid channels (cached for speed)
3. Split into train/val/test (2015-2022 / 2023 / 2024+)
4. Train a simple MLP model
5. Save best checkpoint and training history

### 3. Monitor Training

If TensorBoard is enabled:
```bash
tensorboard --logdir /Volumes/NVME2/x6/logs
```

## Detailed Usage

### Dataset Preparation

```python
from pathlib import Path
from training.dataset import prepare_dataset_from_scratch, create_dataloaders

# Prepare dataset
train_samples, val_samples, test_samples = prepare_dataset_from_scratch(
    data_dir=Path("../data"),
    cache_dir=Path("../data/feature_cache"),
    window=50,              # Channel detection window
    step=25,                # Sliding window step (smaller = more samples)
    min_cycles=1,           # Minimum cycles for valid channel
    train_end="2022-12-31", # End of training period
    val_end="2023-12-31",   # End of validation period
    include_history=False,  # Include channel history features (slower)
    force_rebuild=False     # Force rebuild cache
)

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    train_samples,
    val_samples,
    test_samples,
    batch_size=32,
    num_workers=4,          # Parallel data loading workers
    augment_train=True,     # Data augmentation for training
    pin_memory=True         # Faster GPU transfer
)
```

### Model Definition

Define your model as a PyTorch `nn.Module`:

```python
import torch.nn as nn

class YourModel(nn.Module):
    def __init__(self, feature_dims):
        super().__init__()
        # Define your architecture
        self.backbone = nn.Sequential(...)
        self.duration_head = nn.Sequential(...)
        self.break_direction_head = nn.Sequential(...)
        self.new_direction_head = nn.Sequential(...)
        self.permanent_break_head = nn.Sequential(...)

    def forward(self, features):
        """
        Args:
            features: Dict of tensors with keys like 'tsla_5min', 'spy_5min', etc.

        Returns:
            Dict with predictions:
                - 'duration': [batch_size, 1]
                - 'break_direction': [batch_size, 2] (UP/DOWN logits)
                - 'new_direction': [batch_size, 3] (BEAR/SIDEWAYS/BULL logits)
                - 'permanent_break': [batch_size, 1] (logit)
        """
        # Your implementation
        return predictions
```

### Training

```python
from training.trainer import Trainer, TrainingConfig

# Configure training
config = TrainingConfig(
    # Training hyperparameters
    num_epochs=50,
    learning_rate=0.001,
    weight_decay=0.0001,
    batch_size=32,
    gradient_clip=1.0,

    # Loss weights
    duration_weight=1.0,
    break_direction_weight=2.0,
    new_direction_weight=1.0,
    permanent_break_weight=1.0,

    # Optimization
    optimizer='adam',       # 'adam', 'adamw', 'sgd'
    scheduler='cosine',     # 'cosine', 'step', 'plateau', 'none'
    use_amp=True,          # Mixed precision training

    # Early stopping
    early_stopping_patience=10,
    early_stopping_metric='val_loss',

    # Checkpointing
    save_dir=Path("./checkpoints"),
    save_best_only=True,

    # Device
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Create trainer
trainer = Trainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader
)

# Train
history = trainer.train()

# Load best checkpoint
checkpoint = torch.load(config.save_dir / 'best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Data Format

### Input Features

Each sample contains features from `full_features.py`:

```python
features_dict = {
    # TSLA features per timeframe
    'tsla_5min': [18 features],
    'tsla_15min': [18 features],
    'tsla_30min': [18 features],
    'tsla_1h': [18 features],
    # ... up to 'tsla_3month'

    # SPY features per timeframe
    'spy_5min': [11 features],
    'spy_15min': [11 features],
    # ... up to 'spy_3month'

    # Cross-asset containment
    'cross_5min': [8 features],
    'cross_15min': [8 features],
    # ... up to 'cross_3month'

    # VIX regime
    'vix': [6 features],

    # Channel history
    'tsla_history': [26 features],
    'spy_history': [26 features],

    # Alignment
    'alignment': [3 features]
}
```

**Total: ~300+ features** (varies by number of timeframes with valid channels)

### Labels

Each sample has labels from `labels.py`:

```python
labels_dict = {
    'duration_bars': int,          # Bars until permanent break
    'break_direction': int,        # 0=DOWN, 1=UP
    'break_trigger_tf': str,       # e.g., "1h_upper", "daily_lower"
    'new_channel_direction': int,  # 0=BEAR, 1=SIDEWAYS, 2=BULL
    'permanent_break': bool        # Whether break was found
}
```

## Multi-Task Learning

The model predicts 4 related tasks:

1. **Duration** (Regression)
   - How many bars until channel breaks
   - Loss: MSE or Huber

2. **Break Direction** (Binary Classification)
   - Will it break UP (1) or DOWN (0)?
   - Loss: CrossEntropy

3. **New Channel Direction** (3-class Classification)
   - Next channel will be BEAR (0), SIDEWAYS (1), or BULL (2)?
   - Loss: CrossEntropy

4. **Permanent Break** (Binary Classification)
   - Will a permanent break occur in scan window?
   - Loss: BCEWithLogits

Total loss is weighted sum:
```
L_total = w1*L_duration + w2*L_break_dir + w3*L_new_dir + w4*L_perm_break
```

## Data Caching

To speed up repeated runs, samples are cached to disk:

```python
# First run - scans data and builds cache
train_samples, val_samples, test_samples = prepare_dataset_from_scratch(
    data_dir=data_dir,
    cache_dir=cache_dir,
    force_rebuild=False  # Use cache if exists
)
# Saves to: cache_dir/channel_samples.pkl

# Subsequent runs - loads from cache (~instant)
train_samples, val_samples, test_samples = prepare_dataset_from_scratch(
    data_dir=data_dir,
    cache_dir=cache_dir,
    force_rebuild=False  # Loads from cache
)
```

Cache includes metadata JSON:
```json
{
  "window": 50,
  "step": 25,
  "min_cycles": 1,
  "num_samples": 15234,
  "start_date": "2015-01-02",
  "end_date": "2024-12-31",
  "created_at": "2024-01-15 10:30:45"
}
```

## Data Augmentation

Training augmentation (optional):

1. **Gaussian Noise** - Add small noise to continuous features
   ```python
   augment_noise_std=0.01  # 1% noise
   ```

2. **Time Shifts** (Future)
   - Shift window start by small amount
   - Requires regenerating features

Enable in dataset:
```python
train_dataset = ChannelDataset(
    train_samples,
    augment=True,
    augment_noise_std=0.01
)
```

## Memory Efficiency

For large datasets:

1. **Streaming** - Dataset loads samples on-demand (already implemented)
2. **Num Workers** - Parallel data loading
   ```python
   DataLoader(..., num_workers=4)
   ```
3. **Pin Memory** - Faster GPU transfer
   ```python
   DataLoader(..., pin_memory=True)
   ```
4. **Smaller Batch Size** - Reduce GPU memory
   ```python
   batch_size=16  # Instead of 32
   ```

## Advanced Features

### Custom Transforms

```python
def my_transform(features):
    # Custom feature transformation
    features['tsla_5min'] = torch.log1p(features['tsla_5min'])
    return features

dataset = ChannelDataset(samples, transform=my_transform)
```

### Learning Rate Scheduling

```python
config = TrainingConfig(
    scheduler='cosine',
    scheduler_kwargs={'T_max': 50, 'eta_min': 1e-6}
)

# Or step scheduler
config = TrainingConfig(
    scheduler='step',
    scheduler_kwargs={'step_size': 10, 'gamma': 0.5}
)

# Or plateau scheduler
config = TrainingConfig(
    scheduler='plateau',
    scheduler_kwargs={'patience': 5, 'factor': 0.5}
)
```

### Resume Training

```python
# Load checkpoint
trainer.load_checkpoint(Path('./checkpoints/checkpoint_epoch_25.pt'))

# Continue training
history = trainer.train()
```

## Performance Tips

1. **Use GPU** - 10-50x faster training
   ```python
   config.device = 'cuda'
   ```

2. **Mixed Precision** - 2x faster, less memory
   ```python
   config.use_amp = True
   ```

3. **Batch Size** - Larger batches = faster (if fits in memory)
   ```python
   config.batch_size = 64  # Or 128
   ```

4. **Num Workers** - Parallel data loading
   ```python
   num_workers = 4  # Or 8
   ```

5. **Cache Dataset** - First run is slow, subsequent runs are instant
   ```python
   force_rebuild = False  # Use cache
   ```

## Troubleshooting

### Out of Memory (OOM)

- Reduce batch size: `batch_size=16`
- Disable mixed precision: `use_amp=False`
- Reduce num workers: `num_workers=0`
- Reduce model size: `hidden_dim=128`

### Slow Training

- Enable GPU: `device='cuda'`
- Enable mixed precision: `use_amp=True`
- Increase batch size: `batch_size=64`
- Increase num workers: `num_workers=8`
- Use cached dataset: `force_rebuild=False`

### Poor Convergence

- Adjust learning rate: Try 0.0001 or 0.01
- Tune loss weights: Increase weight on important tasks
- Add regularization: Increase `weight_decay`
- Try different optimizer: 'adamw' or 'sgd'
- Add/reduce dropout

### Cache Issues

- Force rebuild: `force_rebuild=True`
- Delete cache: `rm -rf data/feature_cache/channel_samples.pkl`
- Check metadata: `cat data/feature_cache/channel_samples.json`

## Example Results

Typical training on full dataset (2015-2024):

```
Training samples: ~12,000
Validation samples: ~2,000
Test samples: ~1,500

Epoch 1/50 (45s):
  Train Loss: 2.4567
  Val Loss: 2.1234
  Val Accuracies: Break=0.612, NewDir=0.445, PermBreak=0.892

...

Epoch 35/50 (45s):
  Train Loss: 0.8234
  Val Loss: 1.0123
  Val Accuracies: Break=0.724, NewDir=0.581, PermBreak=0.923

Best validation val_loss: 0.9876
```

## Citation

If you use this training pipeline, please cite:

```
Channel Prediction Training Pipeline v7.0
Author: [Your Name]
Date: 2024
```

## License

[Your License]
