"""
PyTorch Dataset for V15 Channel Prediction.

Handles the 7,880 features with proper validation.

ChannelSample structure:
    - tf_features: Dict[str, float] - ~7,880 TF-prefixed features
    - labels_per_window: Dict[int, Dict[str, ChannelLabels]] - labels per window/TF
    - bar_metadata: Dict[str, Dict[str, float]] - partial bar completion info
"""
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import TIMEFRAMES, TOTAL_FEATURES
from ..core.window_strategy import SelectionStrategy, get_strategy
from ..exceptions import DataLoadError, ValidationError
from ..features.validation import (
    analyze_correlations,
    check_for_constant_features,
    validate_feature_matrix,
)
from ..types import ChannelLabels, ChannelSample


class ChannelDataset(Dataset):
    """
    Dataset for channel break prediction.

    Each sample contains:
        - features: [TOTAL_FEATURES] tensor (~7,880 features)
        - labels: Dict with duration, direction, new_channel, etc.
        - metadata: timestamp, window, bar_metadata

    Expects ChannelSample objects with:
        - tf_features: Dict[str, float] - all features keyed by name
        - labels_per_window: Dict[int, Dict[str, ChannelLabels]]
        - bar_metadata: Dict[str, Dict[str, float]]
        - best_window: int
    """

    def __init__(
        self,
        samples: List[ChannelSample],
        feature_names: Optional[List[str]] = None,
        validate: bool = True,
        target_tf: str = 'daily',
        target_window: Optional[int] = None,
        strategy: str = 'bounce_first',
        strategy_kwargs: Optional[Dict] = None,
        analyze_correlations: bool = True,
    ):
        """
        Args:
            samples: List of ChannelSample objects from scanner
            feature_names: Optional list of feature names for validation
            validate: If True, validate features on load
            target_tf: Target timeframe for labels (must be in TIMEFRAMES)
            target_window: Specific window to use for labels (None = use strategy)
            strategy: Window selection strategy name ('bounce_first', 'label_validity',
                     'balanced_score', 'quality_score', 'learned')
            strategy_kwargs: Optional kwargs to pass to the strategy constructor
            analyze_correlations: If True, analyze feature correlations
        """
        if target_tf not in TIMEFRAMES:
            raise ValidationError(f"Invalid target_tf '{target_tf}'. Must be one of {TIMEFRAMES}")

        self.samples = samples
        self.feature_names = feature_names
        self.target_tf = target_tf
        self.target_window = target_window
        self.analyze_correlations_flag = analyze_correlations
        self.correlation_info = None

        # Initialize window selection strategy
        self.strategy_name = strategy
        strategy_kwargs = strategy_kwargs or {}
        try:
            strategy_enum = SelectionStrategy(strategy)
            self.strategy = get_strategy(strategy_enum, **strategy_kwargs)
        except ValueError:
            raise ValidationError(
                f"Invalid strategy '{strategy}'. Must be one of: "
                f"{[s.value for s in SelectionStrategy]}"
            )

        # Extract and validate features
        self._prepare_data(validate)

    def _prepare_data(self, validate: bool):
        """Convert samples to tensors with validation."""
        features_list = []
        labels_list = []
        metadata_list = []

        for sample in self.samples:
            # Get tf_features dict directly (no backwards compatibility)
            if not sample.tf_features:
                raise ValidationError(
                    f"Sample at {sample.timestamp} has empty tf_features"
                )

            # Convert dict to ordered array
            if self.feature_names is None:
                self.feature_names = sorted(sample.tf_features.keys())

            feature_array = np.array([
                sample.tf_features.get(name, 0.0) for name in self.feature_names
            ], dtype=np.float32)

            features_list.append(feature_array)

            # Extract labels for target TF and window
            # Priority: target_window > strategy > best_window
            if self.target_window is not None:
                window = self.target_window
            elif self.strategy_name == 'learned':
                # For learned mode, we'll handle this differently (pass all windows)
                # Use best_window as fallback for label extraction during prep
                window = sample.best_window
            else:
                # Use strategy to select window
                window = self.strategy.select_window(sample)

            labels = self._extract_labels(sample, self.target_tf, window)
            labels_list.append(labels)

            # Store bar_metadata for this sample
            metadata_list.append({
                'timestamp': sample.timestamp,
                'window': window,
                'bar_metadata': sample.bar_metadata,
            })

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
        self.metadata = metadata_list

    def _extract_labels(self, sample: ChannelSample, tf: str, window: int) -> Dict[str, Any]:
        """
        Extract labels for a specific timeframe and window.

        Args:
            sample: ChannelSample with labels_per_window
            tf: Target timeframe (e.g., 'daily')
            window: Window size to extract labels for

        Returns:
            Dict with label values and validity flags
        """
        # Get labels from labels_per_window structure
        window_labels = sample.labels_per_window.get(window, {})
        tf_labels: Optional[ChannelLabels] = window_labels.get(tf)

        if tf_labels is None:
            return {
                'duration': 0,
                'direction': 0,
                'new_channel': 1,
                'break_trigger_tf': 0,
                'permanent_break': False,
                'break_return': 0.0,
                'valid': False,
                'duration_valid': False,
                'direction_valid': False,
            }

        return {
            'duration': tf_labels.duration_bars,
            'direction': tf_labels.break_direction,
            'new_channel': tf_labels.new_channel_direction,
            'break_trigger_tf': tf_labels.break_trigger_tf,
            'permanent_break': tf_labels.permanent_break,
            'break_return': tf_labels.break_return,
            'valid': tf_labels.duration_valid or tf_labels.direction_valid,
            'duration_valid': tf_labels.duration_valid,
            'direction_valid': tf_labels.direction_valid,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a single sample.

        Returns:
            Tuple of (features, labels) where:
                - features: [n_features] tensor (or [8, features_per_window] for learned mode)
                - labels: Dict with duration, direction, new_channel, etc.
                        For learned mode, also includes 'per_window_features' and 'all_window_labels'
        """
        features = self.features_tensor[idx]
        labels = self.labels[idx]

        # Convert labels to tensors
        label_tensors = {
            'duration': torch.tensor(labels['duration'], dtype=torch.float32),
            'direction': torch.tensor(labels['direction'], dtype=torch.long),
            'new_channel': torch.tensor(labels['new_channel'], dtype=torch.long),
            'break_trigger_tf': torch.tensor(labels['break_trigger_tf'], dtype=torch.long),
            'permanent_break': torch.tensor(labels['permanent_break'], dtype=torch.bool),
            'break_return': torch.tensor(labels['break_return'], dtype=torch.float32),
            'valid': torch.tensor(labels['valid'], dtype=torch.bool),
            'duration_valid': torch.tensor(labels['duration_valid'], dtype=torch.bool),
            'direction_valid': torch.tensor(labels['direction_valid'], dtype=torch.bool),
        }

        # For learned mode, add per-window features so model can learn to select
        if self.strategy_name == 'learned':
            sample = self.samples[idx]
            per_window_features = self._extract_per_window_features(sample)
            label_tensors['per_window_features'] = per_window_features
            label_tensors['all_window_labels'] = self._extract_all_window_labels(sample)
            label_tensors['best_window_idx'] = torch.tensor(
                self._get_best_window_index(sample), dtype=torch.long
            )

        return features, label_tensors

    def _extract_per_window_features(self, sample: ChannelSample) -> torch.Tensor:
        """
        Extract features specific to each window for learned window selection.

        For each of the 8 windows (10, 20, 30, 40, 50, 60, 70, 80), extracts:
            - Label validity counts across TFs
            - Window-specific channel quality metrics (if available)

        Returns:
            Tensor of shape [8, features_per_window] where features_per_window
            includes validity flags and quality metrics per window.
        """
        from ..types import STANDARD_WINDOWS

        n_windows = len(STANDARD_WINDOWS)
        n_tfs = len(TIMEFRAMES)

        # Features per window: validity per TF (n_tfs) + summary stats (3)
        # Summary stats: total_valid_ratio, has_target_tf, window_normalized
        features_per_window = n_tfs + 3
        per_window_features = torch.zeros(n_windows, features_per_window)

        for i, window in enumerate(STANDARD_WINDOWS):
            window_labels = sample.labels_per_window.get(window, {})

            # Per-TF validity flags
            for j, tf in enumerate(TIMEFRAMES):
                tf_label = window_labels.get(tf)
                if tf_label is not None:
                    # Check if label has valid duration or direction
                    is_valid = getattr(tf_label, 'duration_valid', False) or \
                               getattr(tf_label, 'direction_valid', False)
                    per_window_features[i, j] = 1.0 if is_valid else 0.0

            # Summary stats
            valid_count = per_window_features[i, :n_tfs].sum().item()
            per_window_features[i, n_tfs] = valid_count / n_tfs  # valid_ratio

            # Has target TF valid
            target_label = window_labels.get(self.target_tf)
            if target_label is not None:
                has_target = getattr(target_label, 'duration_valid', False) or \
                             getattr(target_label, 'direction_valid', False)
                per_window_features[i, n_tfs + 1] = 1.0 if has_target else 0.0

            # Window size normalized (smaller is closer to 1)
            per_window_features[i, n_tfs + 2] = 1.0 - (window - 10) / 70.0

        return per_window_features

    def _extract_all_window_labels(self, sample: ChannelSample) -> torch.Tensor:
        """
        Extract labels for all windows to support learned window selection.

        Returns:
            Tensor of shape [8, n_label_fields] with labels per window.
        """
        from ..types import STANDARD_WINDOWS

        n_windows = len(STANDARD_WINDOWS)
        # Fields: duration, direction, valid
        n_fields = 3
        all_labels = torch.zeros(n_windows, n_fields)

        for i, window in enumerate(STANDARD_WINDOWS):
            labels = self._extract_labels(sample, self.target_tf, window)
            all_labels[i, 0] = labels['duration']
            all_labels[i, 1] = labels['direction']
            all_labels[i, 2] = 1.0 if labels['valid'] else 0.0

        return all_labels

    def _get_best_window_index(self, sample: ChannelSample) -> int:
        """
        Get the index of the best window in STANDARD_WINDOWS.

        Returns:
            Index (0-7) of the best window in the standard windows list.
        """
        from ..types import STANDARD_WINDOWS

        best_window = sample.best_window
        if best_window in STANDARD_WINDOWS:
            return STANDARD_WINDOWS.index(best_window)
        # Fallback to middle window
        return 4  # window 50

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a sample (timestamp, window, bar_metadata)."""
        return self.metadata[idx]

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
    train_samples: List[ChannelSample],
    val_samples: Optional[List[ChannelSample]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    target_tf: str = 'daily',
    target_window: Optional[int] = None,
    strategy: str = 'bounce_first',
    strategy_kwargs: Optional[Dict] = None,
    validate: bool = True,
    analyze_correlations: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders.

    Args:
        train_samples: List of ChannelSample objects for training
        val_samples: Optional list of ChannelSample objects for validation
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        target_tf: Target timeframe for labels
        target_window: Specific window to use (None = use strategy)
        strategy: Window selection strategy ('bounce_first', 'label_validity',
                 'balanced_score', 'quality_score', 'learned')
        strategy_kwargs: Optional kwargs to pass to the strategy constructor
        validate: Whether to validate features
        analyze_correlations: Whether to analyze feature correlations

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = ChannelDataset(
        train_samples,
        validate=validate,
        target_tf=target_tf,
        target_window=target_window,
        strategy=strategy,
        strategy_kwargs=strategy_kwargs,
        analyze_correlations=analyze_correlations,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_samples:
        val_dataset = ChannelDataset(
            val_samples,
            feature_names=train_dataset.feature_names,  # Use same feature ordering
            validate=validate,
            target_tf=target_tf,
            target_window=target_window,
            strategy=strategy,
            strategy_kwargs=strategy_kwargs,
            analyze_correlations=False,  # Already analyzed on train
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader


def load_samples(path: str) -> List[ChannelSample]:
    """
    Load ChannelSample objects from pickle file.

    Args:
        path: Path to pickle file containing list of ChannelSample objects

    Returns:
        List of ChannelSample objects

    Raises:
        DataLoadError: If file not found or invalid format
    """
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"Samples file not found: {path}")

    with open(path, 'rb') as f:
        samples = pickle.load(f)

    if not isinstance(samples, list):
        raise DataLoadError(f"Expected list of samples, got {type(samples)}")

    # Validate first sample has expected structure
    if samples:
        first = samples[0]
        if not hasattr(first, 'tf_features') or not hasattr(first, 'labels_per_window'):
            raise DataLoadError(
                f"Samples missing required attributes. "
                f"Expected ChannelSample with tf_features and labels_per_window"
            )

    return samples
