"""
PyTorch Dataset for V15 Channel Prediction.

Handles the 8,665 features with proper validation.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import pickle
from pathlib import Path

from ..config import TOTAL_FEATURES, TIMEFRAMES, N_TIMEFRAMES
from ..exceptions import DataLoadError, ValidationError
from ..features.validation import validate_feature_matrix, analyze_correlations, check_for_constant_features
import warnings


class ChannelDataset(Dataset):
    """
    Dataset for channel break prediction.

    Each sample contains:
        - features: [TOTAL_FEATURES] tensor
        - labels: Dict with duration, direction, new_channel, etc.
        - metadata: timestamp, window, etc.
    """

    def __init__(
        self,
        samples: List[Any],  # List of ChannelSample objects
        feature_names: Optional[List[str]] = None,
        validate: bool = True,
        target_tf: str = 'daily',  # Which TF's labels to use
        analyze_correlations: bool = True,
    ):
        """
        Args:
            samples: List of ChannelSample objects from scanner
            feature_names: Optional list of feature names for validation
            validate: If True, validate features on load
            target_tf: Target timeframe for labels
        """
        self.samples = samples
        self.feature_names = feature_names
        self.target_tf = target_tf
        self.analyze_correlations_flag = analyze_correlations
        self.correlation_info = None

        # Extract and validate features
        self._prepare_data(validate)

    def _prepare_data(self, validate: bool):
        """Convert samples to tensors with validation."""
        features_list = []
        labels_list = []

        for sample in self.samples:
            # Get tf_features dict and convert to array
            if hasattr(sample, 'tf_features') and sample.tf_features:
                features = sample.tf_features
            else:
                raise ValidationError(
                    f"Sample at {sample.timestamp} has no tf_features"
                )

            # Convert dict to ordered array
            if self.feature_names is None:
                self.feature_names = sorted(features.keys())

            feature_array = np.array([
                features.get(name, 0.0) for name in self.feature_names
            ], dtype=np.float32)

            features_list.append(feature_array)

            # Extract labels for target TF
            labels = self._extract_labels(sample, self.target_tf)
            labels_list.append(labels)

        # Stack into matrices
        self.features = np.stack(features_list)  # [n_samples, n_features]

        # Validate if requested
        if validate:
            validation_result = validate_feature_matrix(
                self.features, self.feature_names, raise_on_invalid=True
            )

        # Perform correlation analysis if requested
        if self.analyze_correlations_flag:
            self.correlation_info = analyze_correlations(
                self.features, self.feature_names
            )
            n_correlated = len(self.correlation_info.get('highly_correlated_pairs', []))
            if n_correlated > 0:
                warnings.warn(
                    f"Found {n_correlated} highly correlated feature pairs. "
                    "Use get_correlation_report() for details."
                )

            # Check for constant features
            constant_features = check_for_constant_features(
                self.features, self.feature_names
            )
            if constant_features:
                warnings.warn(
                    f"Found {len(constant_features)} constant features: "
                    f"{constant_features[:5]}{'...' if len(constant_features) > 5 else ''}"
                )

        # Convert to tensors
        self.features_tensor = torch.from_numpy(self.features)
        self.labels = labels_list

    def _extract_labels(self, sample, tf: str) -> Dict[str, Any]:
        """Extract labels for a specific timeframe."""
        # Get labels from best window for target TF
        best_window = sample.best_window

        if hasattr(sample, 'labels_per_window'):
            window_labels = sample.labels_per_window.get(best_window, {})
            tf_labels = window_labels.get(tf)

            if tf_labels is None:
                return {
                    'duration': 0,
                    'direction': 0,
                    'new_channel': 1,
                    'valid': False,
                }

            return {
                'duration': tf_labels.duration_bars,
                'direction': tf_labels.break_direction,
                'new_channel': tf_labels.new_channel_direction,
                'permanent_break': tf_labels.permanent_break,
                'duration_valid': tf_labels.duration_valid,
                'direction_valid': tf_labels.direction_valid,
                'valid': True,
            }

        return {'duration': 0, 'direction': 0, 'new_channel': 1, 'valid': False}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        features = self.features_tensor[idx]
        labels = self.labels[idx]

        # Convert labels to tensors
        label_tensors = {
            'duration': torch.tensor(labels['duration'], dtype=torch.float32),
            'direction': torch.tensor(labels['direction'], dtype=torch.long),
            'new_channel': torch.tensor(labels['new_channel'], dtype=torch.long),
            'valid': torch.tensor(labels['valid'], dtype=torch.bool),
        }

        return features, label_tensors

    def get_correlation_report(self) -> Optional[Dict[str, Any]]:
        """
        Get correlation analysis report.

        Returns:
            Dict containing correlation info if analysis was performed, None otherwise.
            Keys may include:
                - 'highly_correlated_pairs': List of (feature1, feature2, correlation) tuples
                - 'correlation_matrix': Full correlation matrix if computed
                - 'threshold': Correlation threshold used for flagging pairs
        """
        return self.correlation_info


def create_dataloaders(
    train_samples: List[Any],
    val_samples: Optional[List[Any]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders.

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = ChannelDataset(train_samples, **kwargs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = None
    if val_samples:
        val_dataset = ChannelDataset(
            val_samples,
            feature_names=train_dataset.feature_names,
            **kwargs
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return train_loader, val_loader


def load_samples(path: str) -> List[Any]:
    """Load samples from pickle file."""
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"Samples file not found: {path}")

    with open(path, 'rb') as f:
        samples = pickle.load(f)

    if not isinstance(samples, list):
        raise DataLoadError(f"Expected list of samples, got {type(samples)}")

    return samples
