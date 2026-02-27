# Training Pipeline Architecture Diagram

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA PREPARATION PHASE                      │
└─────────────────────────────────────────────────────────────────┘

    Raw CSV Files (data/)
    ├── TSLA_1min.csv (97 MB)
    ├── SPY_1min.csv (114 MB)
    └── VIX_History.csv (462 KB)
            ↓
    load_market_data()
    - Parse timestamps
    - Resample 1min → 5min (base TF)
    - Align SPY/VIX with TSLA
            ↓
    ┌─────────────────────────────┐
    │ TSLA: 5min OHLCV (2015-2024)│
    │ SPY:  5min OHLCV (2015-2024)│
    │ VIX:  Daily (2015-2024)     │
    └─────────────────────────────┘
            ↓
    scan_valid_channels()
    - Sliding window (step=25 bars)
    - Detect channels (window=50)
    - Filter valid (min_cycles=1)
    - Extract features (~300 dims)
    - Generate labels (4 tasks)
            ↓
    ┌─────────────────────────────┐
    │ ~15,000 ChannelSample       │
    │ - timestamp                 │
    │ - channel (Channel obj)     │
    │ - features (FullFeatures)   │
    │ - labels (ChannelLabels)    │
    └─────────────────────────────┘
            ↓
    cache_samples()
    → data/feature_cache/channel_samples.pkl
    → data/feature_cache/channel_samples.json (metadata)
            ↓
    split_by_date()
            ↓
    ┌──────────────────┬──────────────────┬──────────────────┐
    │   Train (2015-   │    Val (2023)    │  Test (2024+)    │
    │     2022)        │                  │                  │
    │   ~12,000        │     ~2,000       │     ~1,500       │
    └──────────────────┴──────────────────┴──────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING PHASE                              │
└─────────────────────────────────────────────────────────────────┘

    Train/Val/Test Samples
            ↓
    ChannelDataset (PyTorch Dataset)
    - __getitem__: Load sample
    - Convert features to tensors
    - Apply augmentation (optional)
    - Return (features_dict, labels_dict)
            ↓
    DataLoader (batch_size=32)
    - Multi-worker loading
    - Batch collation
    - GPU memory pinning
            ↓
    ┌─────────────────────────────┐
    │        BATCHED DATA         │
    │                             │
    │ features: {                 │
    │   'tsla_5min': [32, 18],    │
    │   'tsla_15min': [32, 18],   │
    │   'spy_5min': [32, 11],     │
    │   'vix': [32, 6],           │
    │   ... (~20 keys)            │
    │ }                           │
    │                             │
    │ labels: {                   │
    │   'duration_bars': [32],    │
    │   'break_direction': [32],  │
    │   'new_direction': [32],    │
    │   'permanent_break': [32]   │
    │ }                           │
    └─────────────────────────────┘
            ↓
    ┌─────────────────────────────────────────────────────────┐
    │                    MODEL FORWARD                         │
    │                                                          │
    │  Input: features_dict                                    │
    │    ↓                                                     │
    │  Concatenate all features → [32, ~300]                   │
    │    ↓                                                     │
    │  Shared Backbone (MLP)                                   │
    │    ↓                                                     │
    │  ┌──────────┬──────────┬──────────┬──────────┐          │
    │  │ Duration │  Break   │   New    │ Permanent│          │
    │  │   Head   │   Dir    │   Dir    │  Break   │          │
    │  │          │   Head   │   Head   │   Head   │          │
    │  └──────────┴──────────┴──────────┴──────────┘          │
    │       ↓          ↓          ↓          ↓                │
    │    [32,1]     [32,2]     [32,3]     [32,1]              │
    │                                                          │
    │  Output: predictions_dict                               │
    └─────────────────────────────────────────────────────────┘
            ↓
    ┌─────────────────────────────────────────────────────────┐
    │                 MULTI-TASK LOSS                          │
    │                                                          │
    │  L_duration = MSE(pred_duration, true_duration)          │
    │  L_break = CrossEntropy(pred_break, true_break)          │
    │  L_new = CrossEntropy(pred_new, true_new)                │
    │  L_perm = BCE(pred_perm, true_perm)                      │
    │                                                          │
    │  L_total = w1*L_duration + w2*L_break +                  │
    │           w3*L_new + w4*L_perm                           │
    └─────────────────────────────────────────────────────────┘
            ↓
    Backward Pass
    - Gradient computation
    - Gradient clipping (max_norm=1.0)
    - Optimizer step (Adam/AdamW/SGD)
    - LR scheduler step
            ↓
    Validation Loop (every epoch)
    - Forward pass (no gradients)
    - Calculate losses + accuracies
    - Track metrics
            ↓
    Checkpoint & Early Stopping
    - Save best model (lowest val_loss)
    - Save periodic checkpoints
    - Check early stopping patience
            ↓
    ┌─────────────────────────────┐
    │    SAVED ARTIFACTS          │
    │                             │
    │ checkpoints/                │
    │ ├── best_model.pt           │
    │ ├── checkpoint_epoch_5.pt   │
    │ ├── checkpoint_epoch_10.pt  │
    │ └── ...                     │
    │                             │
    │ logs/                       │
    │ ├── training_summary.json   │
    │ └── tensorboard/ (optional) │
    └─────────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                      INFERENCE PHASE                             │
└─────────────────────────────────────────────────────────────────┘

    Load Best Checkpoint
            ↓
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
            ↓
    New Data (live or test)
            ↓
    Extract Features (full_features.py)
            ↓
    features_dict = features_to_tensor_dict(features)
            ↓
    predictions = model(features_dict)
            ↓
    ┌─────────────────────────────┐
    │      PREDICTIONS            │
    │                             │
    │ Duration: 47.3 bars         │
    │ Break Dir: UP (0.87)        │
    │ New Dir: BULL (0.65)        │
    │ Perm Break: YES (0.92)      │
    └─────────────────────────────┘
            ↓
    Trading Decision
```

## Feature Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION                         │
└──────────────────────────────────────────────────────────────┘

Market Data (TSLA 5min)
    ↓
┌───────────────────────────────────────────────┐
│  Multi-Timeframe Channel Detection            │
│                                               │
│  For each TF in [5min, 15min, 30min, 1h,      │
│                 2h, 3h, 4h, daily, weekly,    │
│                 monthly, 3month]:             │
│                                               │
│    1. Resample to TF                          │
│    2. Detect channel (window=50)              │
│    3. Extract TSLA features:                  │
│       - Channel geometry (18 features)        │
│       - RSI indicators                        │
│       - Bounce patterns                       │
│       - Exit tracking (10 features)           │
│       - Break triggers (2 features)           │
│       → Total: ~18 per TF                     │
│                                               │
│    4. Extract SPY features:                   │
│       - Channel geometry (11 features)        │
│       → Total: ~11 per TF                     │
│                                               │
│    5. Cross-asset containment:                │
│       - TSLA position in SPY channel          │
│       - Distance to SPY bounds                │
│       → Total: ~8 per TF                      │
└───────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────┐
│  VIX Regime Features                          │
│  - Current level                              │
│  - Normalized (0-100)                         │
│  - 5-day / 20-day trends                      │
│  - 252-day percentile                         │
│  - Regime classification                      │
│  → Total: 6 features                          │
└───────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────┐
│  Channel History Features                     │
│  TSLA:                                        │
│  - Last 5 channel directions                  │
│  - Last 5 durations                           │
│  - Last 5 break directions                    │
│  - Patterns & statistics                      │
│  → Total: 26 features                         │
│                                               │
│  SPY:                                         │
│  - Same structure                             │
│  → Total: 26 features                         │
└───────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────┐
│  Alignment Features                           │
│  - TSLA/SPY direction match                   │
│  - Both near upper                            │
│  - Both near lower                            │
│  → Total: 3 features                          │
└───────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│           COMPLETE FEATURE VECTOR               │
│                                                 │
│  Total Dimensions: ~300-400                     │
│                                                 │
│  Organized as dict of tensors:                  │
│  {                                              │
│    'tsla_5min': [18],                           │
│    'tsla_15min': [18],                          │
│    ...                                          │
│    'spy_5min': [11],                            │
│    ...                                          │
│    'cross_5min': [8],                           │
│    ...                                          │
│    'vix': [6],                                  │
│    'tsla_history': [26],                        │
│    'spy_history': [26],                         │
│    'alignment': [3]                             │
│  }                                              │
└─────────────────────────────────────────────────┘
```

## Training Loop Diagram

```
┌──────────────────────────────────────────────────────────┐
│                   TRAINING LOOP                          │
└──────────────────────────────────────────────────────────┘

for epoch in range(num_epochs):

    ┌────────────────────────────┐
    │      TRAINING EPOCH        │
    └────────────────────────────┘

    model.train()

    for batch in train_loader:
        │
        ├─ features, labels = batch
        │
        ├─ predictions = model(features)
        │
        ├─ loss, loss_dict = criterion(predictions, labels)
        │   │
        │   ├─ L_duration (MSE)
        │   ├─ L_break_dir (CrossEntropy)
        │   ├─ L_new_dir (CrossEntropy)
        │   └─ L_perm_break (BCE)
        │
        ├─ loss.backward()  # Compute gradients
        │
        ├─ clip_grad_norm_(parameters, max_norm=1.0)
        │
        ├─ optimizer.step()  # Update weights
        │
        └─ optimizer.zero_grad()

    ┌────────────────────────────┐
    │     VALIDATION EPOCH       │
    └────────────────────────────┘

    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            │
            ├─ predictions = model(features)
            │
            ├─ loss, loss_dict = criterion(predictions, labels)
            │
            └─ track metrics (loss, accuracy)

    ┌────────────────────────────┐
    │    CHECKPOINT & METRICS    │
    └────────────────────────────┘

    ├─ scheduler.step(val_loss)
    │
    ├─ if val_loss < best_val_loss:
    │   ├─ best_val_loss = val_loss
    │   ├─ save_checkpoint('best_model.pt')
    │   └─ epochs_without_improvement = 0
    │  else:
    │   └─ epochs_without_improvement += 1
    │
    ├─ if epochs_without_improvement >= patience:
    │   └─ EARLY STOP
    │
    └─ if epoch % save_every == 0:
        └─ save_checkpoint(f'checkpoint_{epoch}.pt')
```

## File Structure

```
v7/
├── core/
│   ├── channel.py          # Channel detection
│   └── timeframe.py        # Timeframe utilities
│
├── features/
│   ├── full_features.py    # Feature extraction
│   ├── rsi.py              # RSI indicators
│   ├── containment.py      # Multi-TF containment
│   ├── cross_asset.py      # SPY/VIX features
│   ├── history.py          # Channel history
│   ├── exit_tracking.py    # Exit pattern tracking
│   └── break_trigger.py    # Break trigger detection
│
├── training/               # ← NEW MODULE
│   ├── __init__.py         # Module exports
│   ├── dataset.py          # PyTorch Dataset (21 KB)
│   ├── trainer.py          # Training loop (21 KB)
│   ├── labels.py           # Label generation (14 KB)
│   ├── example_training.py # Complete example (11 KB)
│   ├── quick_start.py      # Minimal example (3 KB)
│   ├── test_pipeline.py    # Pipeline test (6 KB)
│   ├── README.md           # Documentation (11 KB)
│   ├── IMPLEMENTATION_SUMMARY.md  # This summary (11 KB)
│   └── PIPELINE_DIAGRAM.md # Architecture diagrams
│
└── data/
    ├── TSLA_1min.csv       # Raw data
    ├── SPY_1min.csv
    ├── VIX_History.csv
    └── feature_cache/      # Cached samples
        ├── channel_samples.pkl      (~500 MB)
        └── channel_samples.json     (metadata)
```

## Data Caching Flow

```
First Run (Cold Cache):
    ┌──────────────────┐
    │ Raw CSV Files    │
    └──────────────────┘
           ↓
    load_market_data()  (~5 sec)
           ↓
    scan_valid_channels()  (~10-15 min)
    - Sliding window over entire dataset
    - For each position:
      • Detect channel
      • Extract features (300+ dims)
      • Generate labels (scan forward)
    - Creates ~15,000 samples
           ↓
    cache_samples()  (~1 sec)
    - Pickle to disk
    - Save metadata JSON
           ↓
    ┌──────────────────────────┐
    │ channel_samples.pkl      │
    │ channel_samples.json     │
    └──────────────────────────┘

Subsequent Runs (Hot Cache):
    ┌──────────────────────────┐
    │ channel_samples.pkl      │
    └──────────────────────────┘
           ↓
    load_cached_samples()  (~1 sec)
           ↓
    Ready to train!
```

## Key Components Interaction

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   User Code                                                 │
│   ┌─────────────────────────────────────────────┐           │
│   │ from training import (                      │           │
│   │     prepare_dataset_from_scratch,           │           │
│   │     create_dataloaders,                     │           │
│   │     Trainer, TrainingConfig                 │           │
│   │ )                                           │           │
│   │                                             │           │
│   │ # Prepare data                              │           │
│   │ train, val, test = prepare_dataset_...()    │           │
│   │                                             │           │
│   │ # Create loaders                            │           │
│   │ loaders = create_dataloaders(...)           │           │
│   │                                             │           │
│   │ # Train                                     │           │
│   │ trainer = Trainer(model, config, ...)       │           │
│   │ history = trainer.train()                   │           │
│   └─────────────────────────────────────────────┘           │
│                          ↓                                  │
│   ┌──────────────────────────────────────────────┐          │
│   │           training/dataset.py                │          │
│   │  - ChannelDataset (PyTorch Dataset)          │          │
│   │  - Data loading & caching                    │          │
│   │  - Batch collation                           │          │
│   │  - Train/val/test splitting                  │          │
│   └──────────────────────────────────────────────┘          │
│                          ↓                                  │
│   ┌──────────────────────────────────────────────┐          │
│   │           training/trainer.py                │          │
│   │  - Training loop                             │          │
│   │  - Multi-task loss                           │          │
│   │  - Checkpointing                             │          │
│   │  - Early stopping                            │          │
│   │  - LR scheduling                             │          │
│   └──────────────────────────────────────────────┘          │
│                          ↓                                  │
│   ┌──────────────────────────────────────────────┐          │
│   │      features/full_features.py               │          │
│   │  - extract_full_features()                   │          │
│   │  - features_to_tensor_dict()                 │          │
│   └──────────────────────────────────────────────┘          │
│                          ↓                                  │
│   ┌──────────────────────────────────────────────┐          │
│   │         training/labels.py                   │          │
│   │  - generate_labels()                         │          │
│   │  - labels_to_dict()                          │          │
│   └──────────────────────────────────────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
