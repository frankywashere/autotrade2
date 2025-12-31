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
    scan_valid_channels,
    cache_samples,
    load_cached_samples,
    split_by_date,
    collate_fn,
    create_dataloaders,
    prepare_dataset_from_scratch,
)

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

    # Trainer
    'Trainer',
    'TrainingConfig',

    # Losses
    'CombinedLoss',
    'GaussianNLLLoss',
    'DirectionLoss',
    'NextChannelDirectionLoss',
    'MetricsCalculator',
]
