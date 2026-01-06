"""
Training module for channel break prediction.

This module provides:
1. Label generation for supervised learning
2. PyTorch Dataset for loading channel samples
3. Training loop with multi-task learning
4. Data preparation and caching utilities
"""

from .labels import (
    ChannelLabels,
    BreakDirection,
    NewChannelDirection,
    generate_labels,
    generate_labels_batch,
    labels_to_dict,
    labels_to_array,
)

from .dataset import (
    ChannelDataset,
    ChannelSample,
    load_market_data,
    cache_samples,
    load_cached_samples,
    split_by_date,
    collate_fn,
    create_dataloaders,
    prepare_dataset_from_scratch,
    validate_date_range,
    get_data_date_range,
)

# Use parallel scanner from scanning.py (not the sequential one in dataset.py)
from .scanning import scan_valid_channels

from .trainer import (
    Trainer,
    TrainingConfig,
)

from .losses import (
    CombinedLoss,
    GaussianNLLLoss,
    DirectionLoss,
    NextChannelDirectionLoss,
    MetricsCalculator,
)

from .walk_forward import (
    WalkForwardWindow,
    generate_walk_forward_windows,
    split_samples_by_window,
    validate_windows,
)

from .walk_forward_results import (
    WindowMetrics,
    WalkForwardResults,
)

__all__ = [
    # Labels
    'ChannelLabels',
    'BreakDirection',
    'NewChannelDirection',
    'generate_labels',
    'generate_labels_batch',
    'labels_to_dict',
    'labels_to_array',

    # Dataset
    'ChannelDataset',
    'ChannelSample',
    'load_market_data',
    'scan_valid_channels',
    'cache_samples',
    'load_cached_samples',
    'split_by_date',
    'collate_fn',
    'create_dataloaders',
    'prepare_dataset_from_scratch',
    'validate_date_range',
    'get_data_date_range',

    # Trainer
    'Trainer',
    'TrainingConfig',

    # Losses
    'CombinedLoss',
    'GaussianNLLLoss',
    'DirectionLoss',
    'NextChannelDirectionLoss',
    'MetricsCalculator',

    # Walk-Forward Validation
    'WalkForwardWindow',
    'generate_walk_forward_windows',
    'split_samples_by_window',
    'validate_windows',

    # Walk-Forward Results
    'WindowMetrics',
    'WalkForwardResults',
]
