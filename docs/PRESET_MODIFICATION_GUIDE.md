# Preset Modification Guide

## Overview

The v7 training system includes an interactive preset-based configuration system that allows users to quickly start training with sensible defaults while retaining the flexibility to modify individual parameters. This guide explains how the preset system works and how to customize configurations.

## Table of Contents

1. [How the Preset System Works](#how-the-preset-system-works)
2. [Using Confirmation Screens](#using-confirmation-screens)
3. [Modifying Individual Preset Values](#modifying-individual-preset-values)
4. [Available Presets](#available-presets)
5. [Modifiable Parameters](#modifiable-parameters)
6. [Common Modification Examples](#common-modification-examples)

---

## How the Preset System Works

The preset system provides three predefined training configurations plus a custom mode:

### Preset Selection Flow

1. **Mode Selection**: When you run `python train.py`, you are presented with mode choices
2. **Preset Application**: If you select a preset (Quick Start, Standard, or Full Training), base parameters are automatically set
3. **Partial Override**: Even with a preset, you can still modify additional parameters that aren't in the preset
4. **Configuration Review**: All settings are displayed in a summary screen before training begins
5. **Final Confirmation**: You must confirm before training starts

### Preset Architecture

Presets are defined as dictionaries containing core training parameters:

```python
PRESETS = {
    "Quick Start": {
        "desc": "Fast training for testing (small window, few epochs)",
        "window": 50,
        "step": 50,
        "hidden_dim": 64,
        "cfc_units": 96,
        "num_epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
    },
    # ... other presets
}
```

When a preset is selected, these values are automatically applied to skip interactive prompts for those specific parameters.

---

## Using Confirmation Screens

### Configuration Summary Display

Before training begins, the system displays a comprehensive configuration summary organized into three sections:

#### 1. Data Configuration
- Window size (for channel detection)
- Step size (sliding window increment)
- Training period end date
- Validation period end date
- Channel history feature inclusion

#### 2. Model Configuration
- Hidden dimension
- CfC (Continuous-time Recurrent Neural Network) units
- Number of attention heads
- Dropout rate

#### 3. Training Configuration
- Number of epochs
- Batch size
- Learning rate
- Optimizer type
- Scheduler type
- Device (CPU/CUDA/MPS)

### Reviewing the Summary

The summary is displayed in a rich terminal layout with color-coded panels:
- **Cyan**: Data configuration
- **Magenta**: Model configuration
- **Green**: Training configuration

### Confirmation Prompt

After reviewing the summary, you'll see:
```
Proceed with training? (Y/n)
```

**Options**:
- Press `Enter` or type `Y` to proceed with training
- Type `N` to cancel and exit without training

**Important**: This is your last chance to verify settings before potentially hours of training begin.

---

## Modifying Individual Preset Values

### How Preset Override Works

The preset system uses **partial overrides**, meaning:

1. **Preset values are used silently**: Parameters defined in the preset are not prompted for
2. **Additional parameters are interactive**: All other parameters still require user input
3. **No direct modification during setup**: You cannot change preset values during the initial configuration

### When Preset Values Are Applied

```python
# Data Configuration
if preset:
    window = preset["window"]           # Used directly
    step = preset["step"]              # Used directly
    # Date ranges still prompted
    # History features still prompted

# Model Configuration
if preset:
    hidden_dim = preset["hidden_dim"]  # Used directly
    cfc_units = preset["cfc_units"]    # Used directly
    # Attention heads still prompted
    # Dropout still prompted

# Training Configuration
if preset:
    num_epochs = preset["num_epochs"]          # Used directly
    batch_size = preset["batch_size"]          # Used directly
    learning_rate = preset["learning_rate"]    # Used directly
    # Optimizer still prompted
    # Scheduler still prompted
```

### Workaround: Modifying Preset Parameters

To modify preset parameters, you have two options:

**Option 1: Use Custom Mode**
- Select "Custom - Configure everything manually"
- Manually enter all parameters including those normally in presets

**Option 2: Edit the Preset Definition**
Before running `train.py`, edit the `PRESETS` dictionary in `train.py` (lines 69-100):

```python
PRESETS = {
    "Standard": {
        "desc": "Balanced configuration for typical training",
        "window": 50,        # Change this value
        "step": 25,          # Change this value
        "hidden_dim": 128,   # Change this value
        "cfc_units": 192,    # Change this value
        "num_epochs": 50,    # Change this value
        "batch_size": 64,    # Change this value
        "learning_rate": 0.0005,  # Change this value
    },
}
```

---

## Available Presets

### Quick Start
**Purpose**: Fast training for testing and experimentation

**Characteristics**:
- Small model size (hidden_dim=64)
- Large step size (faster data processing)
- Few epochs (10)
- Minimal training time
- Good for: Testing pipelines, debugging, rapid iteration

**Parameters**:
```
window: 50
step: 50
hidden_dim: 64
cfc_units: 96
num_epochs: 10
batch_size: 32
learning_rate: 0.001
```

### Standard
**Purpose**: Balanced configuration for typical production training

**Characteristics**:
- Medium model size (hidden_dim=128)
- Moderate step size (balance of speed and data richness)
- Reasonable epoch count (50)
- Good performance/time tradeoff
- Good for: Standard model training, most use cases

**Parameters**:
```
window: 50
step: 25
hidden_dim: 128
cfc_units: 192
num_epochs: 50
batch_size: 64
learning_rate: 0.0005
```

### Full Training
**Purpose**: Maximum quality for final production models

**Characteristics**:
- Large model size (hidden_dim=256)
- Small step size (maximum data samples)
- Many epochs (100)
- Requires powerful GPU
- Long training time
- Good for: Final model training, maximum accuracy

**Parameters**:
```
window: 50
step: 10
hidden_dim: 256
cfc_units: 384
num_epochs: 100
batch_size: 128
learning_rate: 0.0003
```

---

## Modifiable Parameters

### Always Modifiable (Not in Presets)

These parameters are **always interactive**, regardless of preset selection:

#### Data Parameters
- **Date range**: Start and end dates for dataset
- **Train/Val split**: Training period end date (default: 2022-12-31)
- **Validation end**: Validation period end date (default: 2023-12-31)
- **Channel history**: Whether to include channel history features

#### Model Parameters
- **Attention heads**: Number of attention heads (choices: 2, 4, 8)
- **Dropout**: Dropout rate (0.0, 0.1, 0.2, 0.3)

#### Training Parameters
- **Optimizer**: adam, adamw, or sgd
- **Scheduler**: cosine, step, plateau, or none
- **Advanced options** (if enabled):
  - Weight decay
  - Gradient clipping
  - Duration loss weight
  - Break direction loss weight
  - New direction loss weight
  - Confidence loss weight

#### System Parameters
- **Device**: CPU, CUDA, or MPS (auto-detected, user selects from available)

### Preset-Locked Parameters

These parameters are **set by preset** and require Custom mode or preset editing to change:

- `window`: Channel detection window size
- `step`: Sliding window step size
- `hidden_dim`: Model hidden dimension
- `cfc_units`: CfC recurrent units
- `num_epochs`: Number of training epochs
- `batch_size`: Training batch size
- `learning_rate`: Initial learning rate

---

## Common Modification Examples

### Example 1: Standard Preset with Different Scheduler

**Goal**: Use Standard preset but with plateau scheduler instead of cosine

**Steps**:
1. Select "Standard - Balanced configuration"
2. Accept default date ranges
3. When prompted for scheduler, select "plateau"
4. Review configuration summary
5. Confirm and proceed

**Result**: Standard preset with plateau scheduler

---

### Example 2: Quick Start with History Features

**Goal**: Fast training but include channel history features

**Steps**:
1. Select "Quick Start - Fast training for testing"
2. Accept default date ranges
3. When prompted "Include channel history features?", select **Yes**
4. Continue with other prompts
5. Review and confirm

**Result**: Quick Start preset with richer feature set

---

### Example 3: Full Training with Custom Date Range

**Goal**: Full quality training on specific date range (2020-2022)

**Steps**:
1. Select "Full Training - Maximum quality (slow)"
2. When prompted "Use full dataset?", select **No**
3. Enter start date: 2020-01-01
4. Enter end date: 2022-12-31
5. Set train end: 2021-12-31
6. Set val end: 2022-06-30
7. Continue with remaining prompts
8. Review and confirm

**Result**: Full Training preset on custom 2020-2022 dataset

---

### Example 4: Standard Preset with Advanced Loss Weights

**Goal**: Standard training but custom loss weights for better direction prediction

**Steps**:
1. Select "Standard - Balanced configuration"
2. Proceed through standard prompts
3. When asked "Configure advanced options?", select **Yes**
4. Set break_direction_weight: 2.0 (double importance)
5. Set new_direction_weight: 2.0 (double importance)
6. Set duration_weight: 1.0 (normal)
7. Set confidence_weight: 0.3 (reduced)
8. Review and confirm

**Result**: Standard preset with emphasis on direction prediction accuracy

---

### Example 5: Completely Custom Configuration

**Goal**: Full control over all parameters

**Steps**:
1. Select "Custom - Configure everything manually"
2. Manually specify every parameter:
   - Window: 75
   - Step: 15
   - Hidden dim: 192
   - CfC units: 320
   - Epochs: 75
   - Batch size: 96
   - Learning rate: 0.0008
   - Attention heads: 4
   - Dropout: 0.15
   - Optimizer: adamw
   - Scheduler: cosine
3. Review comprehensive summary
4. Confirm

**Result**: Fully customized configuration not matching any preset

---

## Best Practices

### When to Use Presets

- **Quick Start**: Development, debugging, pipeline testing
- **Standard**: Most production training, balanced performance
- **Full Training**: Final models, maximum accuracy required, GPU available
- **Custom**: Experimental configurations, research, hyperparameter tuning

### Parameter Modification Guidelines

1. **Start with a preset**: Use the closest preset to your needs
2. **Modify incrementally**: Change only what's necessary
3. **Review carefully**: Always check the configuration summary
4. **Document changes**: Save training configs for reproducibility
5. **Test first**: Try Quick Start before committing to Full Training

### Configuration File

After training, your exact configuration is saved to:
```
checkpoints/training_config.json
```

This allows you to reproduce training runs exactly or use them as templates for future modifications.
