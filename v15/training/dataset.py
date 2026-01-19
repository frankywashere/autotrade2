"""
PyTorch Dataset for V15 Channel Prediction.

Handles the 7,880 features with proper validation.

ChannelSample structure:
    - tf_features: Dict[str, float] - ~7,880 TF-prefixed features
    - labels_per_window: Labels per window/asset/TF
        New structure: {window: {'tsla': {tf: ChannelLabels}, 'spy': {tf: ChannelLabels}}}
        Old structure: {window: {tf: ChannelLabels}} (backward compatible)
    - bar_metadata: Dict[str, Dict[str, float]] - partial bar completion info

Label types extracted (all use explicit prefixes):
    - Core labels: duration, direction, new_channel, permanent_break
    - TSLA break scan (tsla_ prefix): tsla_bars_to_first_break, tsla_break_direction,
                       tsla_break_magnitude, tsla_returned_to_channel, tsla_bounces_after_return,
                       tsla_channel_continued, tsla_break_scan_valid
    - SPY break scan (spy_ prefix): spy_bars_to_first_break, spy_break_direction,
                       spy_break_magnitude, spy_returned_to_channel, spy_bounces_after_return,
                       spy_channel_continued, spy_break_scan_valid
    - Cross-correlation (cross_ prefix): cross_direction_aligned, cross_tsla_broke_first,
                         cross_spy_broke_first, cross_break_lag_bars, cross_magnitude_spread,
                         cross_both_returned, cross_both_permanent, cross_return_pattern_aligned,
                         cross_continuation_aligned, cross_who_broke_first, cross_valid
    - Validity masks: valid, duration_valid, direction_valid
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
from ..types import ChannelLabels, ChannelSample, CrossCorrelationLabels
from ..labels import compute_cross_correlation_labels


class ChannelDataset(Dataset):
    """
    Dataset for channel break prediction.

    Each sample contains:
        - features: [TOTAL_FEATURES] tensor (~7,880 features)
        - labels: Dict with all label types as tensors (using explicit prefixes):
            Core: duration, direction, new_channel, permanent_break
            TSLA break scan (tsla_ prefix): tsla_bars_to_first_break, tsla_break_direction,
                             tsla_break_magnitude, tsla_returned_to_channel, tsla_bounces_after_return,
                             tsla_channel_continued, tsla_break_scan_valid
            SPY break scan (spy_ prefix): spy_bars_to_first_break, spy_break_direction,
                            spy_break_magnitude, spy_returned_to_channel, spy_bounces_after_return,
                            spy_channel_continued, spy_break_scan_valid
            Cross-correlation (cross_ prefix): cross_direction_aligned, cross_tsla_broke_first,
                              cross_spy_broke_first, cross_break_lag_bars, cross_magnitude_spread,
                              cross_both_returned, cross_both_permanent, cross_return_pattern_aligned,
                              cross_continuation_aligned, cross_who_broke_first, cross_valid
            Validity masks: valid, duration_valid, direction_valid
        - metadata: timestamp, window, bar_metadata

    Expects ChannelSample objects with:
        - tf_features: Dict[str, float] - all features keyed by name
        - labels_per_window: Labels per window/asset/TF
            New structure: {window: {'tsla': {tf: ChannelLabels}, 'spy': {tf: ChannelLabels}}}
            Old structure: {window: {tf: ChannelLabels}} (backward compatible)
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

        Handles both old and new labels_per_window structures:
        - Old: {window: {tf: ChannelLabels}}
        - New: {window: {'tsla': {tf: ChannelLabels}, 'spy': {tf: ChannelLabels}}}

        All labels use explicit prefixes for clarity:
        - TSLA fields: tsla_bars_to_first_break, tsla_break_direction, etc.
        - SPY fields: spy_bars_to_first_break, spy_break_direction, etc.
        - Cross fields: cross_direction_aligned, cross_tsla_broke_first, etc.

        Args:
            sample: ChannelSample with labels_per_window
            tf: Target timeframe (e.g., 'daily')
            window: Window size to extract labels for

        Returns:
            Dict with label values and validity flags including:
            - Core labels: duration, direction, new_channel, permanent_break
            - TSLA break scan labels (tsla_ prefix): tsla_bars_to_first_break, etc.
            - SPY break scan labels (spy_ prefix): spy_bars_to_first_break, etc.
            - Cross-correlation labels (cross_ prefix): cross_direction_aligned, etc.
            - Validity masks: valid, duration_valid, direction_valid, tsla_break_scan_valid,
              spy_break_scan_valid, cross_valid
        """
        # Get labels from labels_per_window structure
        window_labels = sample.labels_per_window.get(window, {})

        # Handle both old and new structure
        # New structure has 'tsla' and 'spy' keys
        # Old structure has TF keys directly
        if 'tsla' in window_labels:
            # New structure: {window: {'tsla': {tf: ChannelLabels}, 'spy': {tf: ChannelLabels}}}
            tsla_labels_dict = window_labels.get('tsla', {})
            spy_labels_dict = window_labels.get('spy', {})
            tf_labels: Optional[ChannelLabels] = tsla_labels_dict.get(tf)
            spy_tf_labels: Optional[ChannelLabels] = spy_labels_dict.get(tf)
        else:
            # Old structure: {window: {tf: ChannelLabels}}
            tf_labels: Optional[ChannelLabels] = window_labels.get(tf)
            spy_tf_labels = None

        # Default labels for invalid/missing data
        default_labels = {
            # Core labels
            'duration': 0,
            'direction': 0,
            'new_channel': 1,
            'permanent_break': False,
            'valid': False,
            'duration_valid': False,
            'direction_valid': False,
            # TSLA break scan labels (explicit tsla_ prefix)
            'tsla_bars_to_first_break': 0,
            'tsla_break_direction': 0,
            'tsla_break_magnitude': 0.0,
            'tsla_returned_to_channel': False,
            'tsla_bounces_after_return': 0,
            'tsla_channel_continued': False,
            'tsla_break_scan_valid': False,
            # SPY break scan labels (spy_ prefix)
            'spy_bars_to_first_break': 0,
            'spy_break_direction': 0,
            'spy_break_magnitude': 0.0,
            'spy_returned_to_channel': False,
            'spy_bounces_after_return': 0,
            'spy_channel_continued': False,
            'spy_break_scan_valid': False,
            # Cross-correlation labels (cross_ prefix)
            'cross_direction_aligned': False,
            'cross_tsla_broke_first': False,
            'cross_spy_broke_first': False,
            'cross_break_lag_bars': 0,
            'cross_magnitude_spread': 0.0,
            'cross_both_returned': False,
            'cross_both_permanent': False,
            'cross_return_pattern_aligned': False,
            'cross_continuation_aligned': False,
            'cross_who_broke_first': 0,  # 0=simultaneous, 1=TSLA first, 2=SPY first
            'cross_valid': False,
        }

        if tf_labels is None:
            return default_labels

        # Build labels dict with core labels
        labels = {
            # Core labels
            'duration': tf_labels.duration_bars,
            'direction': tf_labels.break_direction,
            'new_channel': tf_labels.next_channel_direction,
            'permanent_break': tf_labels.permanent_break,
            'valid': tf_labels.duration_valid or tf_labels.direction_valid,
            'duration_valid': tf_labels.duration_valid,
            'direction_valid': tf_labels.direction_valid,
            # TSLA break scan labels (explicit tsla_ prefix)
            'tsla_bars_to_first_break': getattr(tf_labels, 'bars_to_first_break', 0),
            'tsla_break_direction': getattr(tf_labels, 'break_direction', 0),
            'tsla_break_magnitude': getattr(tf_labels, 'break_magnitude', 0.0),
            'tsla_returned_to_channel': getattr(tf_labels, 'returned_to_channel', False),
            'tsla_bounces_after_return': getattr(tf_labels, 'bounces_after_return', 0),
            'tsla_channel_continued': getattr(tf_labels, 'channel_continued', False),
            'tsla_break_scan_valid': getattr(tf_labels, 'break_scan_valid', False),
        }

        # SPY break scan labels (from SPY's ChannelLabels if available)
        if spy_tf_labels is not None:
            labels.update({
                'spy_bars_to_first_break': getattr(spy_tf_labels, 'bars_to_first_break', 0),
                'spy_break_direction': getattr(spy_tf_labels, 'break_direction', 0),
                'spy_break_magnitude': getattr(spy_tf_labels, 'break_magnitude', 0.0),
                'spy_returned_to_channel': getattr(spy_tf_labels, 'returned_to_channel', False),
                'spy_bounces_after_return': getattr(spy_tf_labels, 'bounces_after_return', 0),
                'spy_channel_continued': getattr(spy_tf_labels, 'channel_continued', False),
                'spy_break_scan_valid': getattr(spy_tf_labels, 'break_scan_valid', False),
            })
        else:
            # Try to get SPY fields from TSLA labels (old structure with spy_ prefix on same object)
            labels.update({
                'spy_bars_to_first_break': getattr(tf_labels, 'spy_bars_to_first_break', 0),
                'spy_break_direction': getattr(tf_labels, 'spy_break_direction', 0),
                'spy_break_magnitude': getattr(tf_labels, 'spy_break_magnitude', 0.0),
                'spy_returned_to_channel': getattr(tf_labels, 'spy_returned_to_channel', False),
                'spy_bounces_after_return': getattr(tf_labels, 'spy_bounces_after_return', 0),
                'spy_channel_continued': getattr(tf_labels, 'spy_channel_continued', False),
                'spy_break_scan_valid': getattr(tf_labels, 'break_scan_valid', False),  # Same validity for old structure
            })

        # Compute cross-correlation labels
        cross_labels = self._extract_cross_correlation_labels(tf_labels, spy_tf_labels, tf)
        labels.update(cross_labels)

        return labels

    def _extract_cross_correlation_labels(
        self,
        tsla_labels: Optional[ChannelLabels],
        spy_labels: Optional[ChannelLabels],
        tf: str
    ) -> Dict[str, Any]:
        """
        Extract cross-correlation labels comparing TSLA and SPY break behavior.

        All cross-correlation fields use the cross_ prefix for clarity.

        Args:
            tsla_labels: ChannelLabels for TSLA (may be None)
            spy_labels: ChannelLabels for SPY (may be None, or same object as tsla_labels for old structure)
            tf: Target timeframe

        Returns:
            Dict with cross-correlation label values (all with cross_ prefix):
            - cross_direction_aligned: Both broke in the same direction
            - cross_tsla_broke_first: TSLA broke before SPY
            - cross_spy_broke_first: SPY broke before TSLA
            - cross_who_broke_first: 0=simultaneous, 1=TSLA first, 2=SPY first
            - cross_break_lag_bars: Bars between TSLA and SPY breaks
            - cross_magnitude_spread: TSLA magnitude minus SPY magnitude
            - cross_both_returned: Both returned to channel
            - cross_both_permanent: Neither returned to channel
            - cross_return_pattern_aligned: Both returned or both permanent
            - cross_continuation_aligned: Both continued or both didn't
            - cross_valid: Cross-correlation data is valid
        """
        # Default cross-correlation labels (all with cross_ prefix)
        default_cross = {
            'cross_direction_aligned': False,
            'cross_tsla_broke_first': False,
            'cross_spy_broke_first': False,
            'cross_break_lag_bars': 0,
            'cross_magnitude_spread': 0.0,
            'cross_both_returned': False,
            'cross_both_permanent': False,
            'cross_return_pattern_aligned': False,
            'cross_continuation_aligned': False,
            'cross_who_broke_first': 0,  # 0=simultaneous, 1=TSLA first, 2=SPY first
            'cross_valid': False,
        }

        if tsla_labels is None:
            return default_cross

        # Check if we have separate SPY labels (new structure) or combined labels (old structure)
        if spy_labels is not None and spy_labels is not tsla_labels:
            # New structure: separate TSLA and SPY ChannelLabels
            cross = compute_cross_correlation_labels(tsla_labels, spy_labels, tf)
            # Compute who_broke_first: 0=simultaneous, 1=TSLA first, 2=SPY first
            if cross.tsla_broke_first:
                who_broke_first = 1
            elif cross.spy_broke_first:
                who_broke_first = 2
            else:
                who_broke_first = 0
            return {
                'cross_direction_aligned': cross.break_direction_aligned,
                'cross_tsla_broke_first': cross.tsla_broke_first,
                'cross_spy_broke_first': cross.spy_broke_first,
                'cross_break_lag_bars': cross.break_lag_bars,
                'cross_magnitude_spread': cross.magnitude_spread,
                'cross_both_returned': cross.both_returned,
                'cross_both_permanent': cross.both_permanent,
                'cross_return_pattern_aligned': cross.return_pattern_aligned,
                'cross_continuation_aligned': cross.continuation_aligned,
                'cross_who_broke_first': who_broke_first,
                'cross_valid': cross.cross_valid,
            }
        else:
            # Old structure: SPY fields are on the same ChannelLabels object
            # Check if TSLA labels have valid break scan data for both assets
            tsla_valid = getattr(tsla_labels, 'break_scan_valid', False)
            # For old structure, SPY validity is implicit if spy_ fields are populated
            spy_bars = getattr(tsla_labels, 'spy_bars_to_first_break', 0)
            spy_valid = tsla_valid and spy_bars > 0

            if not tsla_valid or not spy_valid:
                return default_cross

            # Compute cross-correlation from the single ChannelLabels object
            tsla_break_dir = getattr(tsla_labels, 'break_direction', 0)
            spy_break_dir = getattr(tsla_labels, 'spy_break_direction', 0)
            direction_aligned = tsla_break_dir == spy_break_dir

            tsla_bars = getattr(tsla_labels, 'bars_to_first_break', 0)
            spy_bars = getattr(tsla_labels, 'spy_bars_to_first_break', 0)
            tsla_broke_first = tsla_bars < spy_bars
            spy_broke_first = spy_bars < tsla_bars
            break_lag_bars = abs(tsla_bars - spy_bars)

            # Compute who_broke_first: 0=simultaneous, 1=TSLA first, 2=SPY first
            if tsla_broke_first:
                who_broke_first = 1
            elif spy_broke_first:
                who_broke_first = 2
            else:
                who_broke_first = 0

            tsla_mag = getattr(tsla_labels, 'break_magnitude', 0.0)
            spy_mag = getattr(tsla_labels, 'spy_break_magnitude', 0.0)
            magnitude_spread = tsla_mag - spy_mag

            tsla_returned = getattr(tsla_labels, 'returned_to_channel', False)
            spy_returned = getattr(tsla_labels, 'spy_returned_to_channel', False)
            both_returned = tsla_returned and spy_returned
            both_permanent = (not tsla_returned) and (not spy_returned)
            return_pattern_aligned = both_returned or both_permanent

            tsla_continued = getattr(tsla_labels, 'channel_continued', False)
            spy_continued = getattr(tsla_labels, 'spy_channel_continued', False)
            continuation_aligned = tsla_continued == spy_continued

            return {
                'cross_direction_aligned': direction_aligned,
                'cross_tsla_broke_first': tsla_broke_first,
                'cross_spy_broke_first': spy_broke_first,
                'cross_break_lag_bars': break_lag_bars,
                'cross_magnitude_spread': magnitude_spread,
                'cross_both_returned': both_returned,
                'cross_both_permanent': both_permanent,
                'cross_return_pattern_aligned': return_pattern_aligned,
                'cross_continuation_aligned': continuation_aligned,
                'cross_who_broke_first': who_broke_first,
                'cross_valid': True,
            }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a single sample.

        Returns:
            Tuple of (features, labels) where:
                - features: [n_features] tensor (or [8, features_per_window] for learned mode)
                - labels: Dict with explicit prefixes (tsla_, spy_, cross_) for all fields.
                        For learned mode, also includes 'per_window_features' and 'all_window_labels'
        """
        features = self.features_tensor[idx]
        labels = self.labels[idx]

        # Convert labels to tensors
        label_tensors = {
            # Core labels
            'duration': torch.tensor(labels['duration'], dtype=torch.float32),
            'direction': torch.tensor(labels['direction'], dtype=torch.long),
            'new_channel': torch.tensor(labels['new_channel'], dtype=torch.long),
            'permanent_break': torch.tensor(labels['permanent_break'], dtype=torch.bool),
            'valid': torch.tensor(labels['valid'], dtype=torch.bool),
            'duration_valid': torch.tensor(labels['duration_valid'], dtype=torch.bool),
            'direction_valid': torch.tensor(labels['direction_valid'], dtype=torch.bool),

            # TSLA break scan labels (explicit tsla_ prefix)
            'tsla_bars_to_first_break': torch.tensor(labels['tsla_bars_to_first_break'], dtype=torch.float32),
            'tsla_break_direction': torch.tensor(labels['tsla_break_direction'], dtype=torch.long),
            'tsla_break_magnitude': torch.tensor(labels['tsla_break_magnitude'], dtype=torch.float32),
            'tsla_returned_to_channel': torch.tensor(labels['tsla_returned_to_channel'], dtype=torch.bool),
            'tsla_bounces_after_return': torch.tensor(labels['tsla_bounces_after_return'], dtype=torch.float32),
            'tsla_channel_continued': torch.tensor(labels['tsla_channel_continued'], dtype=torch.bool),
            'tsla_break_scan_valid': torch.tensor(labels['tsla_break_scan_valid'], dtype=torch.bool),

            # SPY break scan labels (spy_ prefix)
            'spy_bars_to_first_break': torch.tensor(labels['spy_bars_to_first_break'], dtype=torch.float32),
            'spy_break_direction': torch.tensor(labels['spy_break_direction'], dtype=torch.long),
            'spy_break_magnitude': torch.tensor(labels['spy_break_magnitude'], dtype=torch.float32),
            'spy_returned_to_channel': torch.tensor(labels['spy_returned_to_channel'], dtype=torch.bool),
            'spy_bounces_after_return': torch.tensor(labels['spy_bounces_after_return'], dtype=torch.float32),
            'spy_channel_continued': torch.tensor(labels['spy_channel_continued'], dtype=torch.bool),
            'spy_break_scan_valid': torch.tensor(labels['spy_break_scan_valid'], dtype=torch.bool),

            # Cross-correlation labels (cross_ prefix)
            'cross_direction_aligned': torch.tensor(labels['cross_direction_aligned'], dtype=torch.bool),
            'cross_tsla_broke_first': torch.tensor(labels['cross_tsla_broke_first'], dtype=torch.bool),
            'cross_spy_broke_first': torch.tensor(labels['cross_spy_broke_first'], dtype=torch.bool),
            'cross_break_lag_bars': torch.tensor(labels['cross_break_lag_bars'], dtype=torch.float32),
            'cross_magnitude_spread': torch.tensor(labels['cross_magnitude_spread'], dtype=torch.float32),
            'cross_both_returned': torch.tensor(labels['cross_both_returned'], dtype=torch.bool),
            'cross_both_permanent': torch.tensor(labels['cross_both_permanent'], dtype=torch.bool),
            'cross_return_pattern_aligned': torch.tensor(labels['cross_return_pattern_aligned'], dtype=torch.bool),
            'cross_continuation_aligned': torch.tensor(labels['cross_continuation_aligned'], dtype=torch.bool),
            'cross_who_broke_first': torch.tensor(labels['cross_who_broke_first'], dtype=torch.long),
            'cross_valid': torch.tensor(labels['cross_valid'], dtype=torch.bool),
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

        Handles both old and new labels_per_window structures:
        - Old: {window: {tf: ChannelLabels}}
        - New: {window: {'tsla': {tf: ChannelLabels}, 'spy': {tf: ChannelLabels}}}

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

            # Handle both old and new structure
            if 'tsla' in window_labels:
                # New structure: get TSLA labels dict
                tf_labels_dict = window_labels.get('tsla', {})
            else:
                # Old structure: labels are directly by TF
                tf_labels_dict = window_labels

            # Per-TF validity flags
            for j, tf in enumerate(TIMEFRAMES):
                tf_label = tf_labels_dict.get(tf)
                if tf_label is not None:
                    # Check if label has valid duration or direction
                    is_valid = getattr(tf_label, 'duration_valid', False) or \
                               getattr(tf_label, 'direction_valid', False)
                    per_window_features[i, j] = 1.0 if is_valid else 0.0

            # Summary stats
            valid_count = per_window_features[i, :n_tfs].sum().item()
            per_window_features[i, n_tfs] = valid_count / n_tfs  # valid_ratio

            # Has target TF valid
            target_label = tf_labels_dict.get(self.target_tf)
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
