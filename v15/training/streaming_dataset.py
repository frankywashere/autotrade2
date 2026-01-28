"""
Chunked Streaming Dataset for memory-efficient training on large datasets.

This module provides a PyTorch Dataset that loads samples in chunks,
enabling training on datasets much larger than available RAM.

Memory usage: ~2.8GB for 15K sample chunks (well under 4GB target)

Usage:
    from v15.training.streaming_dataset import ChunkedStreamingDataset

    dataset = ChunkedStreamingDataset(
        binary_path='/path/to/samples.bin',
        feature_names=feature_names,
        chunk_size=15000,
        target_tf='daily'
    )

    # Use with DataLoader (shuffle handled by sampler)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Tuple
import threading
from pathlib import Path

from ..binary_index import get_or_build_index
from ..binary_loader import ChannelSample, read_channel_sample
from ..config import TIMEFRAMES


class ChunkedStreamingDataset(Dataset):
    """
    Memory-efficient dataset that loads samples in chunks.

    Instead of loading all samples into RAM, this dataset:
    1. Builds/loads a byte-offset index (~8MB for 1M samples)
    2. Loads chunks of samples on-demand (~2.8GB per 15K samples)
    3. Caches the current chunk for efficient batch access
    4. Optionally prefetches the next chunk in background

    Memory budget (15K chunk):
        - Offset index: ~8 MB
        - Feature names: ~0.5 MB
        - Current chunk:
            - Python objects: ~1.8 GB
            - Feature tensors: ~900 MB
            - Label tensors: ~150 MB
        Total: ~2.86 GB (well under 4GB target)
    """

    def __init__(
        self,
        binary_path: str,
        feature_names: Optional[List[str]] = None,
        chunk_size: int = 15000,
        target_tf: str = 'daily',
        target_window: Optional[int] = None,
        prefetch: bool = True,
    ):
        """
        Initialize streaming dataset.

        Args:
            binary_path: Path to binary sample file
            feature_names: List of feature names (extracted from file if None)
            chunk_size: Number of samples per chunk (default 15000 = ~2.8GB)
            target_tf: Target timeframe for labels (default 'daily')
            target_window: Specific window to use (None = use sample's best_window)
            prefetch: Whether to prefetch next chunk in background
        """
        self.binary_path = binary_path
        self.chunk_size = chunk_size
        self.target_tf = target_tf
        self.target_window = target_window
        self.prefetch = prefetch

        # Build or load index
        print(f"Loading offset index for {binary_path}...")
        self.offsets, self.version, self.feature_table = get_or_build_index(binary_path)
        self.num_samples = len(self.offsets)

        # Get feature names from file if not provided
        if feature_names is None:
            if self.feature_table is not None:
                self.feature_names = self.feature_table
            else:
                raise ValueError(
                    "feature_names must be provided for v2 format files, "
                    "or use v3 format which includes feature names"
                )
        else:
            self.feature_names = feature_names

        self.num_features = len(self.feature_names)

        # Chunk cache state
        self.current_chunk_idx = -1
        self.current_chunk_features: Optional[torch.Tensor] = None
        self.current_chunk_labels: Optional[List[Dict[str, Any]]] = None
        self.current_chunk_start = 0
        self.current_chunk_end = 0

        # Prefetch state
        self.prefetch_thread: Optional[threading.Thread] = None
        self.prefetch_chunk_idx = -1
        self.prefetch_chunk_features: Optional[torch.Tensor] = None
        self.prefetch_chunk_labels: Optional[List[Dict[str, Any]]] = None
        self.prefetch_lock = threading.Lock()

        # Open file handle (will be reopened for each chunk load for thread safety)
        self._file_lock = threading.Lock()

        print(f"ChunkedStreamingDataset initialized:")
        print(f"  Total samples: {self.num_samples:,}")
        print(f"  Chunk size: {self.chunk_size:,}")
        print(f"  Num chunks: {(self.num_samples + self.chunk_size - 1) // self.chunk_size}")
        print(f"  Num features: {self.num_features:,}")
        print(f"  Target TF: {self.target_tf}")
        print(f"  Estimated RAM per chunk: ~{self.chunk_size * 190 / 1024:.1f} MB")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a single sample.

        Loads chunk if needed, then returns sample from cached chunk.

        Args:
            idx: Sample index (0 to num_samples-1)

        Returns:
            Tuple of (features, labels) where:
                - features: [num_features] float tensor
                - labels: Dict of label tensors
        """
        # Determine which chunk this index belongs to
        chunk_idx = idx // self.chunk_size

        # Load chunk if needed
        if chunk_idx != self.current_chunk_idx:
            self._ensure_chunk_loaded(chunk_idx)

        # Get sample from cached chunk
        local_idx = idx - self.current_chunk_start

        # Bounds check
        if local_idx < 0 or local_idx >= len(self.current_chunk_labels):
            raise IndexError(
                f"Index {idx} (local {local_idx}) out of range for chunk {chunk_idx} "
                f"(size {len(self.current_chunk_labels)})"
            )

        features = self.current_chunk_features[local_idx]
        labels = self.current_chunk_labels[local_idx]

        return features, labels

    def _ensure_chunk_loaded(self, chunk_idx: int):
        """Load chunk, using prefetched data if available."""
        # Check if prefetch has this chunk ready
        with self.prefetch_lock:
            if self.prefetch_chunk_idx == chunk_idx and self.prefetch_chunk_features is not None:
                print(f"Using prefetched chunk {chunk_idx}")
                self.current_chunk_features = self.prefetch_chunk_features
                self.current_chunk_labels = self.prefetch_chunk_labels
                self.current_chunk_idx = chunk_idx
                self.current_chunk_start = chunk_idx * self.chunk_size
                self.current_chunk_end = min(self.current_chunk_start + self.chunk_size, self.num_samples)

                # Clear prefetch
                self.prefetch_chunk_features = None
                self.prefetch_chunk_labels = None
                self.prefetch_chunk_idx = -1
            else:
                # Load synchronously
                self._load_chunk(chunk_idx)

        # Start prefetching next chunk
        if self.prefetch:
            next_chunk_idx = chunk_idx + 1
            if next_chunk_idx * self.chunk_size < self.num_samples:
                self._start_prefetch(next_chunk_idx)

    def _load_chunk(self, chunk_idx: int):
        """Load a chunk of samples into memory."""
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.num_samples)
        chunk_size = end_idx - start_idx

        print(f"Loading chunk {chunk_idx}: samples [{start_idx:,}:{end_idx:,}]")

        # Load samples from binary file
        samples = []
        with open(self.binary_path, 'rb') as f:
            for i in range(start_idx, end_idx):
                offset = self.offsets[i]
                f.seek(offset)
                sample = read_channel_sample(f, version=self.version, feature_table=self.feature_table)
                samples.append(sample)

        # Convert to tensors
        features, labels = self._convert_to_tensors(samples)

        # Update cache
        self.current_chunk_features = features
        self.current_chunk_labels = labels
        self.current_chunk_idx = chunk_idx
        self.current_chunk_start = start_idx
        self.current_chunk_end = end_idx

        print(f"  Loaded {len(samples):,} samples, features shape: {features.shape}")

    def _start_prefetch(self, chunk_idx: int):
        """Start background thread to prefetch next chunk."""
        # Don't prefetch if already prefetching this chunk
        with self.prefetch_lock:
            if self.prefetch_chunk_idx == chunk_idx:
                return

        def prefetch_worker():
            try:
                start_idx = chunk_idx * self.chunk_size
                end_idx = min(start_idx + self.chunk_size, self.num_samples)

                # Load samples
                samples = []
                with open(self.binary_path, 'rb') as f:
                    for i in range(start_idx, end_idx):
                        offset = self.offsets[i]
                        f.seek(offset)
                        sample = read_channel_sample(f, version=self.version, feature_table=self.feature_table)
                        samples.append(sample)

                # Convert to tensors
                features, labels = self._convert_to_tensors(samples)

                # Store prefetched data
                with self.prefetch_lock:
                    self.prefetch_chunk_features = features
                    self.prefetch_chunk_labels = labels
                    self.prefetch_chunk_idx = chunk_idx

            except Exception as e:
                print(f"Prefetch error for chunk {chunk_idx}: {e}")

        # Start background thread
        self.prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        self.prefetch_thread.start()

    def _convert_to_tensors(
        self, samples: List[ChannelSample]
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Convert samples to feature tensors and label dicts.

        This mirrors the logic in ChannelDataset but operates on a chunk.
        """
        features_list = []
        labels_list = []

        for sample in samples:
            # Extract features as ordered array
            feature_array = np.array([
                sample.tf_features.get(name, 0.0)
                for name in self.feature_names
            ], dtype=np.float32)
            features_list.append(feature_array)

            # Extract labels
            labels = self._extract_labels(sample)
            labels_list.append(labels)

        # Stack features into tensor
        features_tensor = torch.from_numpy(np.stack(features_list))

        return features_tensor, labels_list

    def _extract_labels(self, sample: ChannelSample) -> Dict[str, torch.Tensor]:
        """
        Extract labels for a sample (mirrors ChannelDataset._extract_labels).

        Uses target_window if specified, otherwise sample.best_window.
        """
        # Determine window to use
        window = self.target_window if self.target_window is not None else sample.best_window

        # Get labels from labels_per_window structure
        window_labels = sample.labels_per_window.get(window, {})

        # Handle both old and new structure
        if 'tsla' in window_labels:
            # New structure: {window: {'tsla': {tf: ChannelLabels}, 'spy': {tf: ChannelLabels}}}
            tsla_labels_dict = window_labels.get('tsla', {})
            spy_labels_dict = window_labels.get('spy', {})
            tf_labels = tsla_labels_dict.get(self.target_tf)
            spy_tf_labels = spy_labels_dict.get(self.target_tf)
        else:
            # Old structure: {window: {tf: ChannelLabels}}
            tf_labels = window_labels.get(self.target_tf)
            spy_tf_labels = None

        # Build label tensor dict (same structure as ChannelDataset)
        if tf_labels is None:
            # Return default labels
            return self._get_default_labels()

        labels = {
            # Core labels
            'duration': torch.tensor(tf_labels.duration_bars, dtype=torch.float32),
            'direction': torch.tensor(tf_labels.break_direction, dtype=torch.long),
            'new_channel': torch.tensor(tf_labels.next_channel_direction, dtype=torch.long),
            'permanent_break': torch.tensor(tf_labels.permanent_break, dtype=torch.bool),
            'valid': torch.tensor(
                tf_labels.duration_valid or tf_labels.direction_valid, dtype=torch.bool
            ),
            'duration_valid': torch.tensor(tf_labels.duration_valid, dtype=torch.bool),
            'direction_valid': torch.tensor(tf_labels.direction_valid, dtype=torch.bool),

            # TSLA break scan labels
            'tsla_bars_to_first_break': torch.tensor(
                getattr(tf_labels, 'bars_to_first_break', 0), dtype=torch.float32
            ),
            'tsla_break_direction': torch.tensor(
                getattr(tf_labels, 'break_direction', 0), dtype=torch.long
            ),
            'tsla_break_magnitude': torch.tensor(
                getattr(tf_labels, 'break_magnitude', 0.0), dtype=torch.float32
            ),
            'tsla_returned_to_channel': torch.tensor(
                getattr(tf_labels, 'returned_to_channel', False), dtype=torch.bool
            ),
            'tsla_bounces_after_return': torch.tensor(
                getattr(tf_labels, 'bounces_after_return', 0), dtype=torch.float32
            ),
            'tsla_channel_continued': torch.tensor(
                getattr(tf_labels, 'channel_continued', False), dtype=torch.bool
            ),
            'tsla_break_scan_valid': torch.tensor(
                getattr(tf_labels, 'break_scan_valid', False), dtype=torch.bool
            ),
            'tsla_duration_to_permanent': torch.tensor(
                getattr(tf_labels, 'duration_to_permanent', -1), dtype=torch.float32
            ),
            'tsla_avg_bars_outside': torch.tensor(
                getattr(tf_labels, 'avg_bars_outside', 0.0), dtype=torch.float32
            ),
            'tsla_total_bars_outside': torch.tensor(
                getattr(tf_labels, 'total_bars_outside', 0), dtype=torch.float32
            ),
            'tsla_durability_score': torch.tensor(
                getattr(tf_labels, 'durability_score', 0.0), dtype=torch.float32
            ),
            'tsla_first_break_returned': torch.tensor(
                getattr(tf_labels, 'first_break_returned', False), dtype=torch.bool
            ),
            'tsla_exit_return_rate': torch.tensor(
                getattr(tf_labels, 'exit_return_rate', 0.0), dtype=torch.float32
            ),
            'tsla_exits_returned_count': torch.tensor(
                getattr(tf_labels, 'exits_returned_count', 0), dtype=torch.float32
            ),
            'tsla_exits_stayed_out_count': torch.tensor(
                getattr(tf_labels, 'exits_stayed_out_count', 0), dtype=torch.float32
            ),
            'tsla_scan_timed_out': torch.tensor(
                getattr(tf_labels, 'scan_timed_out', False), dtype=torch.bool
            ),
            'tsla_bars_verified_permanent': torch.tensor(
                getattr(tf_labels, 'bars_verified_permanent', 0), dtype=torch.float32
            ),

            # TSLA RSI labels
            'tsla_rsi_at_first_break': torch.tensor(
                getattr(tf_labels, 'rsi_at_first_break', 50.0), dtype=torch.float32
            ),
            'tsla_rsi_at_permanent_break': torch.tensor(
                getattr(tf_labels, 'rsi_at_permanent_break', 50.0), dtype=torch.float32
            ),
            'tsla_rsi_at_channel_end': torch.tensor(
                getattr(tf_labels, 'rsi_at_channel_end', 50.0), dtype=torch.float32
            ),
            'tsla_rsi_overbought_at_break': torch.tensor(
                int(getattr(tf_labels, 'rsi_overbought_at_break', False)), dtype=torch.long
            ),
            'tsla_rsi_oversold_at_break': torch.tensor(
                int(getattr(tf_labels, 'rsi_oversold_at_break', False)), dtype=torch.long
            ),
            'tsla_rsi_divergence_at_break': torch.tensor(
                getattr(tf_labels, 'rsi_divergence_at_break', 0), dtype=torch.long
            ),
            'tsla_rsi_trend_in_channel': torch.tensor(
                getattr(tf_labels, 'rsi_trend_in_channel', 0), dtype=torch.long
            ),
            'tsla_rsi_range_in_channel': torch.tensor(
                getattr(tf_labels, 'rsi_range_in_channel', 0.0), dtype=torch.float32
            ),
        }

        # SPY labels (from separate spy_labels object or from same object with spy_ prefix)
        if spy_tf_labels is not None:
            labels.update(self._extract_spy_labels(spy_tf_labels))
        else:
            labels.update(self._extract_spy_labels_from_combined(tf_labels))

        # Cross-correlation labels
        labels.update(self._extract_cross_correlation_labels(tf_labels, spy_tf_labels))

        # Per-TF duration labels
        per_tf_duration, per_tf_duration_valid = self._extract_per_tf_duration_labels(sample)
        labels['per_tf_duration'] = per_tf_duration
        labels['per_tf_duration_valid'] = per_tf_duration_valid

        return labels

    def _extract_spy_labels(self, spy_labels) -> Dict[str, torch.Tensor]:
        """Extract SPY labels from separate SPY ChannelLabels object."""
        return {
            'spy_bars_to_first_break': torch.tensor(
                getattr(spy_labels, 'bars_to_first_break', 0), dtype=torch.float32
            ),
            'spy_break_direction': torch.tensor(
                getattr(spy_labels, 'break_direction', 0), dtype=torch.long
            ),
            'spy_break_magnitude': torch.tensor(
                getattr(spy_labels, 'break_magnitude', 0.0), dtype=torch.float32
            ),
            'spy_returned_to_channel': torch.tensor(
                getattr(spy_labels, 'returned_to_channel', False), dtype=torch.bool
            ),
            'spy_bounces_after_return': torch.tensor(
                getattr(spy_labels, 'bounces_after_return', 0), dtype=torch.float32
            ),
            'spy_channel_continued': torch.tensor(
                getattr(spy_labels, 'channel_continued', False), dtype=torch.bool
            ),
            'spy_break_scan_valid': torch.tensor(
                getattr(spy_labels, 'break_scan_valid', False), dtype=torch.bool
            ),
            'spy_duration_to_permanent': torch.tensor(
                getattr(spy_labels, 'duration_to_permanent', -1), dtype=torch.float32
            ),
            'spy_avg_bars_outside': torch.tensor(
                getattr(spy_labels, 'avg_bars_outside', 0.0), dtype=torch.float32
            ),
            'spy_total_bars_outside': torch.tensor(
                getattr(spy_labels, 'total_bars_outside', 0), dtype=torch.float32
            ),
            'spy_durability_score': torch.tensor(
                getattr(spy_labels, 'durability_score', 0.0), dtype=torch.float32
            ),
            'spy_first_break_returned': torch.tensor(
                getattr(spy_labels, 'first_break_returned', False), dtype=torch.bool
            ),
            'spy_exit_return_rate': torch.tensor(
                getattr(spy_labels, 'exit_return_rate', 0.0), dtype=torch.float32
            ),
            'spy_exits_returned_count': torch.tensor(
                getattr(spy_labels, 'exits_returned_count', 0), dtype=torch.float32
            ),
            'spy_exits_stayed_out_count': torch.tensor(
                getattr(spy_labels, 'exits_stayed_out_count', 0), dtype=torch.float32
            ),
            'spy_scan_timed_out': torch.tensor(
                getattr(spy_labels, 'scan_timed_out', False), dtype=torch.bool
            ),
            'spy_bars_verified_permanent': torch.tensor(
                getattr(spy_labels, 'bars_verified_permanent', 0), dtype=torch.float32
            ),
            # SPY RSI labels
            'spy_rsi_at_first_break': torch.tensor(
                getattr(spy_labels, 'rsi_at_first_break', 50.0), dtype=torch.float32
            ),
            'spy_rsi_at_permanent_break': torch.tensor(
                getattr(spy_labels, 'rsi_at_permanent_break', 50.0), dtype=torch.float32
            ),
            'spy_rsi_at_channel_end': torch.tensor(
                getattr(spy_labels, 'rsi_at_channel_end', 50.0), dtype=torch.float32
            ),
            'spy_rsi_overbought_at_break': torch.tensor(
                int(getattr(spy_labels, 'rsi_overbought_at_break', False)), dtype=torch.long
            ),
            'spy_rsi_oversold_at_break': torch.tensor(
                int(getattr(spy_labels, 'rsi_oversold_at_break', False)), dtype=torch.long
            ),
            'spy_rsi_divergence_at_break': torch.tensor(
                getattr(spy_labels, 'rsi_divergence_at_break', 0), dtype=torch.long
            ),
            'spy_rsi_trend_in_channel': torch.tensor(
                getattr(spy_labels, 'rsi_trend_in_channel', 0), dtype=torch.long
            ),
            'spy_rsi_range_in_channel': torch.tensor(
                getattr(spy_labels, 'rsi_range_in_channel', 0.0), dtype=torch.float32
            ),
        }

    def _extract_spy_labels_from_combined(self, tf_labels) -> Dict[str, torch.Tensor]:
        """Extract SPY labels from combined TSLA object (old format with spy_ prefix)."""
        return {
            'spy_bars_to_first_break': torch.tensor(
                getattr(tf_labels, 'spy_bars_to_first_break', 0), dtype=torch.float32
            ),
            'spy_break_direction': torch.tensor(
                getattr(tf_labels, 'spy_break_direction', 0), dtype=torch.long
            ),
            'spy_break_magnitude': torch.tensor(
                getattr(tf_labels, 'spy_break_magnitude', 0.0), dtype=torch.float32
            ),
            'spy_returned_to_channel': torch.tensor(
                getattr(tf_labels, 'spy_returned_to_channel', False), dtype=torch.bool
            ),
            'spy_bounces_after_return': torch.tensor(
                getattr(tf_labels, 'spy_bounces_after_return', 0), dtype=torch.float32
            ),
            'spy_channel_continued': torch.tensor(
                getattr(tf_labels, 'spy_channel_continued', False), dtype=torch.bool
            ),
            'spy_break_scan_valid': torch.tensor(
                getattr(tf_labels, 'break_scan_valid', False), dtype=torch.bool
            ),
            'spy_duration_to_permanent': torch.tensor(
                getattr(tf_labels, 'spy_duration_to_permanent', -1), dtype=torch.float32
            ),
            'spy_avg_bars_outside': torch.tensor(
                getattr(tf_labels, 'spy_avg_bars_outside', 0.0), dtype=torch.float32
            ),
            'spy_total_bars_outside': torch.tensor(
                getattr(tf_labels, 'spy_total_bars_outside', 0), dtype=torch.float32
            ),
            'spy_durability_score': torch.tensor(
                getattr(tf_labels, 'spy_durability_score', 0.0), dtype=torch.float32
            ),
            'spy_first_break_returned': torch.tensor(
                getattr(tf_labels, 'spy_first_break_returned', False), dtype=torch.bool
            ),
            'spy_exit_return_rate': torch.tensor(
                getattr(tf_labels, 'spy_exit_return_rate', 0.0), dtype=torch.float32
            ),
            'spy_exits_returned_count': torch.tensor(
                getattr(tf_labels, 'spy_exits_returned_count', 0), dtype=torch.float32
            ),
            'spy_exits_stayed_out_count': torch.tensor(
                getattr(tf_labels, 'spy_exits_stayed_out_count', 0), dtype=torch.float32
            ),
            'spy_scan_timed_out': torch.tensor(
                getattr(tf_labels, 'spy_scan_timed_out', False), dtype=torch.bool
            ),
            'spy_bars_verified_permanent': torch.tensor(
                getattr(tf_labels, 'spy_bars_verified_permanent', 0), dtype=torch.float32
            ),
            # SPY RSI labels from combined object
            'spy_rsi_at_first_break': torch.tensor(
                getattr(tf_labels, 'spy_rsi_at_first_break', 50.0), dtype=torch.float32
            ),
            'spy_rsi_at_permanent_break': torch.tensor(
                getattr(tf_labels, 'spy_rsi_at_permanent_break', 50.0), dtype=torch.float32
            ),
            'spy_rsi_at_channel_end': torch.tensor(
                getattr(tf_labels, 'spy_rsi_at_channel_end', 50.0), dtype=torch.float32
            ),
            'spy_rsi_overbought_at_break': torch.tensor(
                int(getattr(tf_labels, 'spy_rsi_overbought_at_break', False)), dtype=torch.long
            ),
            'spy_rsi_oversold_at_break': torch.tensor(
                int(getattr(tf_labels, 'spy_rsi_oversold_at_break', False)), dtype=torch.long
            ),
            'spy_rsi_divergence_at_break': torch.tensor(
                getattr(tf_labels, 'spy_rsi_divergence_at_break', 0), dtype=torch.long
            ),
            'spy_rsi_trend_in_channel': torch.tensor(
                getattr(tf_labels, 'spy_rsi_trend_in_channel', 0), dtype=torch.long
            ),
            'spy_rsi_range_in_channel': torch.tensor(
                getattr(tf_labels, 'spy_rsi_range_in_channel', 0.0), dtype=torch.float32
            ),
        }

    def _extract_cross_correlation_labels(self, tsla_labels, spy_labels) -> Dict[str, torch.Tensor]:
        """Extract cross-correlation labels."""
        # Default cross-correlation labels
        cross = {
            'cross_direction_aligned': torch.tensor(False, dtype=torch.bool),
            'cross_tsla_broke_first': torch.tensor(False, dtype=torch.bool),
            'cross_spy_broke_first': torch.tensor(False, dtype=torch.bool),
            'cross_break_lag_bars': torch.tensor(0, dtype=torch.float32),
            'cross_magnitude_spread': torch.tensor(0.0, dtype=torch.float32),
            'cross_both_returned': torch.tensor(False, dtype=torch.bool),
            'cross_both_permanent': torch.tensor(False, dtype=torch.bool),
            'cross_return_pattern_aligned': torch.tensor(False, dtype=torch.bool),
            'cross_continuation_aligned': torch.tensor(False, dtype=torch.bool),
            'cross_who_broke_first': torch.tensor(0, dtype=torch.long),
            'cross_valid': torch.tensor(False, dtype=torch.bool),
            'cross_tsla_permanent_first': torch.tensor(False, dtype=torch.bool),
            'cross_spy_permanent_first': torch.tensor(False, dtype=torch.bool),
            'cross_permanent_duration_lag_bars': torch.tensor(0, dtype=torch.float32),
            'cross_permanent_duration_spread': torch.tensor(0, dtype=torch.float32),
            'cross_durability_spread': torch.tensor(0.0, dtype=torch.float32),
            'cross_avg_bars_outside_spread': torch.tensor(0.0, dtype=torch.float32),
            'cross_total_bars_outside_spread': torch.tensor(0, dtype=torch.float32),
            'cross_both_high_durability': torch.tensor(False, dtype=torch.bool),
            'cross_both_low_durability': torch.tensor(False, dtype=torch.bool),
            'cross_durability_aligned': torch.tensor(False, dtype=torch.bool),
            'cross_tsla_more_durable': torch.tensor(False, dtype=torch.bool),
            'cross_spy_more_durable': torch.tensor(False, dtype=torch.bool),
            'cross_permanent_dynamics_valid': torch.tensor(False, dtype=torch.bool),
            'cross_exit_return_rate_spread': torch.tensor(0.0, dtype=torch.float32),
            'cross_exit_return_rate_aligned': torch.tensor(False, dtype=torch.bool),
            'cross_tsla_more_resilient': torch.tensor(False, dtype=torch.bool),
            'cross_spy_more_resilient': torch.tensor(False, dtype=torch.bool),
            'cross_exits_returned_spread': torch.tensor(0, dtype=torch.float32),
            'cross_exits_stayed_out_spread': torch.tensor(0, dtype=torch.float32),
            'cross_total_exits_spread': torch.tensor(0, dtype=torch.float32),
            'cross_both_scan_timed_out': torch.tensor(False, dtype=torch.bool),
            'cross_scan_timeout_aligned': torch.tensor(False, dtype=torch.bool),
            'cross_bars_verified_spread': torch.tensor(0, dtype=torch.float32),
            'cross_both_first_returned_then_permanent': torch.tensor(False, dtype=torch.bool),
            'cross_both_never_returned': torch.tensor(False, dtype=torch.bool),
            'cross_exit_verification_valid': torch.tensor(False, dtype=torch.bool),
            'cross_exit_timing_correlation': torch.tensor(0.0, dtype=torch.float32),
            'cross_exit_timing_lag_mean': torch.tensor(0.0, dtype=torch.float32),
            'cross_exit_direction_agreement': torch.tensor(0.0, dtype=torch.float32),
            'cross_exit_count_spread': torch.tensor(0, dtype=torch.float32),
            'cross_lead_lag_exits': torch.tensor(0, dtype=torch.float32),
            'cross_exit_magnitude_correlation': torch.tensor(0.0, dtype=torch.float32),
            'cross_mean_magnitude_spread': torch.tensor(0.0, dtype=torch.float32),
            'cross_exit_duration_correlation': torch.tensor(0.0, dtype=torch.float32),
            'cross_mean_duration_spread': torch.tensor(0.0, dtype=torch.float32),
            'cross_simultaneous_exit_count': torch.tensor(0, dtype=torch.float32),
            'cross_exit_cross_correlation_valid': torch.tensor(0, dtype=torch.long),
            'cross_divergence_predicts_reversal': torch.tensor(0, dtype=torch.long),
            'cross_permanent_break_matches_next': torch.tensor(0, dtype=torch.long),
            'cross_next_channel_direction_aligned': torch.tensor(0, dtype=torch.long),
            'cross_next_channel_quality_aligned': torch.tensor(0, dtype=torch.long),
            'cross_best_next_channel_tsla_vs_spy': torch.tensor(0, dtype=torch.long),
            'cross_rsi_aligned_at_break': torch.tensor(0, dtype=torch.long),
            'cross_rsi_divergence_aligned': torch.tensor(0, dtype=torch.long),
            'cross_tsla_rsi_higher_at_break': torch.tensor(0, dtype=torch.long),
            'cross_rsi_spread_at_break': torch.tensor(0.0, dtype=torch.float32),
            'cross_overbought_predicts_down_break': torch.tensor(0, dtype=torch.long),
            'cross_oversold_predicts_up_break': torch.tensor(0, dtype=torch.long),
        }

        if tsla_labels is None:
            return cross

        # Compute cross-correlation if both labels available
        tsla_valid = getattr(tsla_labels, 'break_scan_valid', False)

        if spy_labels is not None and spy_labels is not tsla_labels:
            # New structure: separate objects
            spy_valid = getattr(spy_labels, 'break_scan_valid', False)
            if tsla_valid and spy_valid:
                cross = self._compute_cross_labels(tsla_labels, spy_labels)
        else:
            # Old structure: spy_ prefixed fields on same object
            spy_bars = getattr(tsla_labels, 'spy_bars_to_first_break', 0)
            spy_valid = tsla_valid and spy_bars > 0
            if tsla_valid and spy_valid:
                cross = self._compute_cross_labels_from_combined(tsla_labels)

        return cross

    def _compute_cross_labels(self, tsla_labels, spy_labels) -> Dict[str, torch.Tensor]:
        """Compute cross-correlation labels from separate TSLA and SPY labels."""
        tsla_dir = getattr(tsla_labels, 'break_direction', 0)
        spy_dir = getattr(spy_labels, 'break_direction', 0)
        tsla_bars = getattr(tsla_labels, 'bars_to_first_break', 0)
        spy_bars = getattr(spy_labels, 'bars_to_first_break', 0)

        direction_aligned = tsla_dir == spy_dir
        tsla_broke_first = tsla_bars < spy_bars
        spy_broke_first = spy_bars < tsla_bars
        who_broke_first = 1 if tsla_broke_first else (2 if spy_broke_first else 0)

        tsla_returned = getattr(tsla_labels, 'returned_to_channel', False)
        spy_returned = getattr(spy_labels, 'returned_to_channel', False)

        return {
            'cross_direction_aligned': torch.tensor(direction_aligned, dtype=torch.bool),
            'cross_tsla_broke_first': torch.tensor(tsla_broke_first, dtype=torch.bool),
            'cross_spy_broke_first': torch.tensor(spy_broke_first, dtype=torch.bool),
            'cross_break_lag_bars': torch.tensor(abs(tsla_bars - spy_bars), dtype=torch.float32),
            'cross_magnitude_spread': torch.tensor(
                getattr(tsla_labels, 'break_magnitude', 0.0) -
                getattr(spy_labels, 'break_magnitude', 0.0), dtype=torch.float32
            ),
            'cross_both_returned': torch.tensor(tsla_returned and spy_returned, dtype=torch.bool),
            'cross_both_permanent': torch.tensor(not tsla_returned and not spy_returned, dtype=torch.bool),
            'cross_return_pattern_aligned': torch.tensor(
                (tsla_returned and spy_returned) or (not tsla_returned and not spy_returned), dtype=torch.bool
            ),
            'cross_continuation_aligned': torch.tensor(
                getattr(tsla_labels, 'channel_continued', False) ==
                getattr(spy_labels, 'channel_continued', False), dtype=torch.bool
            ),
            'cross_who_broke_first': torch.tensor(who_broke_first, dtype=torch.long),
            'cross_valid': torch.tensor(True, dtype=torch.bool),
            # Additional fields with defaults
            'cross_tsla_permanent_first': torch.tensor(False, dtype=torch.bool),
            'cross_spy_permanent_first': torch.tensor(False, dtype=torch.bool),
            'cross_permanent_duration_lag_bars': torch.tensor(0, dtype=torch.float32),
            'cross_permanent_duration_spread': torch.tensor(0, dtype=torch.float32),
            'cross_durability_spread': torch.tensor(
                getattr(tsla_labels, 'durability_score', 0.0) -
                getattr(spy_labels, 'durability_score', 0.0), dtype=torch.float32
            ),
            'cross_avg_bars_outside_spread': torch.tensor(
                getattr(tsla_labels, 'avg_bars_outside', 0.0) -
                getattr(spy_labels, 'avg_bars_outside', 0.0), dtype=torch.float32
            ),
            'cross_total_bars_outside_spread': torch.tensor(
                getattr(tsla_labels, 'total_bars_outside', 0) -
                getattr(spy_labels, 'total_bars_outside', 0), dtype=torch.float32
            ),
            'cross_both_high_durability': torch.tensor(False, dtype=torch.bool),
            'cross_both_low_durability': torch.tensor(False, dtype=torch.bool),
            'cross_durability_aligned': torch.tensor(False, dtype=torch.bool),
            'cross_tsla_more_durable': torch.tensor(
                getattr(tsla_labels, 'durability_score', 0.0) >
                getattr(spy_labels, 'durability_score', 0.0), dtype=torch.bool
            ),
            'cross_spy_more_durable': torch.tensor(
                getattr(spy_labels, 'durability_score', 0.0) >
                getattr(tsla_labels, 'durability_score', 0.0), dtype=torch.bool
            ),
            'cross_permanent_dynamics_valid': torch.tensor(False, dtype=torch.bool),
            'cross_exit_return_rate_spread': torch.tensor(0.0, dtype=torch.float32),
            'cross_exit_return_rate_aligned': torch.tensor(False, dtype=torch.bool),
            'cross_tsla_more_resilient': torch.tensor(False, dtype=torch.bool),
            'cross_spy_more_resilient': torch.tensor(False, dtype=torch.bool),
            'cross_exits_returned_spread': torch.tensor(0, dtype=torch.float32),
            'cross_exits_stayed_out_spread': torch.tensor(0, dtype=torch.float32),
            'cross_total_exits_spread': torch.tensor(0, dtype=torch.float32),
            'cross_both_scan_timed_out': torch.tensor(False, dtype=torch.bool),
            'cross_scan_timeout_aligned': torch.tensor(False, dtype=torch.bool),
            'cross_bars_verified_spread': torch.tensor(0, dtype=torch.float32),
            'cross_both_first_returned_then_permanent': torch.tensor(False, dtype=torch.bool),
            'cross_both_never_returned': torch.tensor(False, dtype=torch.bool),
            'cross_exit_verification_valid': torch.tensor(False, dtype=torch.bool),
            'cross_exit_timing_correlation': torch.tensor(0.0, dtype=torch.float32),
            'cross_exit_timing_lag_mean': torch.tensor(0.0, dtype=torch.float32),
            'cross_exit_direction_agreement': torch.tensor(0.0, dtype=torch.float32),
            'cross_exit_count_spread': torch.tensor(0, dtype=torch.float32),
            'cross_lead_lag_exits': torch.tensor(0, dtype=torch.float32),
            'cross_exit_magnitude_correlation': torch.tensor(0.0, dtype=torch.float32),
            'cross_mean_magnitude_spread': torch.tensor(0.0, dtype=torch.float32),
            'cross_exit_duration_correlation': torch.tensor(0.0, dtype=torch.float32),
            'cross_mean_duration_spread': torch.tensor(0.0, dtype=torch.float32),
            'cross_simultaneous_exit_count': torch.tensor(0, dtype=torch.float32),
            'cross_exit_cross_correlation_valid': torch.tensor(0, dtype=torch.long),
            'cross_divergence_predicts_reversal': torch.tensor(0, dtype=torch.long),
            'cross_permanent_break_matches_next': torch.tensor(0, dtype=torch.long),
            'cross_next_channel_direction_aligned': torch.tensor(0, dtype=torch.long),
            'cross_next_channel_quality_aligned': torch.tensor(0, dtype=torch.long),
            'cross_best_next_channel_tsla_vs_spy': torch.tensor(0, dtype=torch.long),
            'cross_rsi_aligned_at_break': torch.tensor(0, dtype=torch.long),
            'cross_rsi_divergence_aligned': torch.tensor(0, dtype=torch.long),
            'cross_tsla_rsi_higher_at_break': torch.tensor(0, dtype=torch.long),
            'cross_rsi_spread_at_break': torch.tensor(0.0, dtype=torch.float32),
            'cross_overbought_predicts_down_break': torch.tensor(0, dtype=torch.long),
            'cross_oversold_predicts_up_break': torch.tensor(0, dtype=torch.long),
        }

    def _compute_cross_labels_from_combined(self, tf_labels) -> Dict[str, torch.Tensor]:
        """Compute cross-correlation labels from combined object (old format)."""
        tsla_dir = getattr(tf_labels, 'break_direction', 0)
        spy_dir = getattr(tf_labels, 'spy_break_direction', 0)
        tsla_bars = getattr(tf_labels, 'bars_to_first_break', 0)
        spy_bars = getattr(tf_labels, 'spy_bars_to_first_break', 0)

        direction_aligned = tsla_dir == spy_dir
        tsla_broke_first = tsla_bars < spy_bars
        spy_broke_first = spy_bars < tsla_bars
        who_broke_first = 1 if tsla_broke_first else (2 if spy_broke_first else 0)

        tsla_returned = getattr(tf_labels, 'returned_to_channel', False)
        spy_returned = getattr(tf_labels, 'spy_returned_to_channel', False)

        return {
            'cross_direction_aligned': torch.tensor(direction_aligned, dtype=torch.bool),
            'cross_tsla_broke_first': torch.tensor(tsla_broke_first, dtype=torch.bool),
            'cross_spy_broke_first': torch.tensor(spy_broke_first, dtype=torch.bool),
            'cross_break_lag_bars': torch.tensor(abs(tsla_bars - spy_bars), dtype=torch.float32),
            'cross_magnitude_spread': torch.tensor(
                getattr(tf_labels, 'break_magnitude', 0.0) -
                getattr(tf_labels, 'spy_break_magnitude', 0.0), dtype=torch.float32
            ),
            'cross_both_returned': torch.tensor(tsla_returned and spy_returned, dtype=torch.bool),
            'cross_both_permanent': torch.tensor(not tsla_returned and not spy_returned, dtype=torch.bool),
            'cross_return_pattern_aligned': torch.tensor(
                (tsla_returned and spy_returned) or (not tsla_returned and not spy_returned), dtype=torch.bool
            ),
            'cross_continuation_aligned': torch.tensor(
                getattr(tf_labels, 'channel_continued', False) ==
                getattr(tf_labels, 'spy_channel_continued', False), dtype=torch.bool
            ),
            'cross_who_broke_first': torch.tensor(who_broke_first, dtype=torch.long),
            'cross_valid': torch.tensor(True, dtype=torch.bool),
            # Fill remaining with defaults
            'cross_tsla_permanent_first': torch.tensor(False, dtype=torch.bool),
            'cross_spy_permanent_first': torch.tensor(False, dtype=torch.bool),
            'cross_permanent_duration_lag_bars': torch.tensor(0, dtype=torch.float32),
            'cross_permanent_duration_spread': torch.tensor(0, dtype=torch.float32),
            'cross_durability_spread': torch.tensor(0.0, dtype=torch.float32),
            'cross_avg_bars_outside_spread': torch.tensor(0.0, dtype=torch.float32),
            'cross_total_bars_outside_spread': torch.tensor(0, dtype=torch.float32),
            'cross_both_high_durability': torch.tensor(False, dtype=torch.bool),
            'cross_both_low_durability': torch.tensor(False, dtype=torch.bool),
            'cross_durability_aligned': torch.tensor(False, dtype=torch.bool),
            'cross_tsla_more_durable': torch.tensor(False, dtype=torch.bool),
            'cross_spy_more_durable': torch.tensor(False, dtype=torch.bool),
            'cross_permanent_dynamics_valid': torch.tensor(False, dtype=torch.bool),
            'cross_exit_return_rate_spread': torch.tensor(0.0, dtype=torch.float32),
            'cross_exit_return_rate_aligned': torch.tensor(False, dtype=torch.bool),
            'cross_tsla_more_resilient': torch.tensor(False, dtype=torch.bool),
            'cross_spy_more_resilient': torch.tensor(False, dtype=torch.bool),
            'cross_exits_returned_spread': torch.tensor(0, dtype=torch.float32),
            'cross_exits_stayed_out_spread': torch.tensor(0, dtype=torch.float32),
            'cross_total_exits_spread': torch.tensor(0, dtype=torch.float32),
            'cross_both_scan_timed_out': torch.tensor(False, dtype=torch.bool),
            'cross_scan_timeout_aligned': torch.tensor(False, dtype=torch.bool),
            'cross_bars_verified_spread': torch.tensor(0, dtype=torch.float32),
            'cross_both_first_returned_then_permanent': torch.tensor(False, dtype=torch.bool),
            'cross_both_never_returned': torch.tensor(False, dtype=torch.bool),
            'cross_exit_verification_valid': torch.tensor(False, dtype=torch.bool),
            'cross_exit_timing_correlation': torch.tensor(0.0, dtype=torch.float32),
            'cross_exit_timing_lag_mean': torch.tensor(0.0, dtype=torch.float32),
            'cross_exit_direction_agreement': torch.tensor(0.0, dtype=torch.float32),
            'cross_exit_count_spread': torch.tensor(0, dtype=torch.float32),
            'cross_lead_lag_exits': torch.tensor(0, dtype=torch.float32),
            'cross_exit_magnitude_correlation': torch.tensor(0.0, dtype=torch.float32),
            'cross_mean_magnitude_spread': torch.tensor(0.0, dtype=torch.float32),
            'cross_exit_duration_correlation': torch.tensor(0.0, dtype=torch.float32),
            'cross_mean_duration_spread': torch.tensor(0.0, dtype=torch.float32),
            'cross_simultaneous_exit_count': torch.tensor(0, dtype=torch.float32),
            'cross_exit_cross_correlation_valid': torch.tensor(0, dtype=torch.long),
            'cross_divergence_predicts_reversal': torch.tensor(0, dtype=torch.long),
            'cross_permanent_break_matches_next': torch.tensor(0, dtype=torch.long),
            'cross_next_channel_direction_aligned': torch.tensor(0, dtype=torch.long),
            'cross_next_channel_quality_aligned': torch.tensor(0, dtype=torch.long),
            'cross_best_next_channel_tsla_vs_spy': torch.tensor(0, dtype=torch.long),
            'cross_rsi_aligned_at_break': torch.tensor(0, dtype=torch.long),
            'cross_rsi_divergence_aligned': torch.tensor(0, dtype=torch.long),
            'cross_tsla_rsi_higher_at_break': torch.tensor(0, dtype=torch.long),
            'cross_rsi_spread_at_break': torch.tensor(0.0, dtype=torch.float32),
            'cross_overbought_predicts_down_break': torch.tensor(0, dtype=torch.long),
            'cross_oversold_predicts_up_break': torch.tensor(0, dtype=torch.long),
        }

    def _extract_per_tf_duration_labels(
        self, sample: ChannelSample
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract duration labels for all TFs from the sample's best window."""
        n_tfs = len(TIMEFRAMES)
        per_tf_duration = torch.zeros(n_tfs, dtype=torch.float32)
        per_tf_duration_valid = torch.zeros(n_tfs, dtype=torch.bool)

        window = sample.best_window
        window_labels = sample.labels_per_window.get(window, {})

        # Handle both structures
        if 'tsla' in window_labels:
            tf_labels_dict = window_labels.get('tsla', {})
        else:
            tf_labels_dict = window_labels

        for tf_idx, tf in enumerate(TIMEFRAMES):
            tf_label = tf_labels_dict.get(tf)
            if tf_label is not None:
                duration = getattr(tf_label, 'duration_bars', 0)
                is_valid = getattr(tf_label, 'duration_valid', False)
                per_tf_duration[tf_idx] = float(duration) if duration is not None else 0.0
                per_tf_duration_valid[tf_idx] = bool(is_valid)

        return per_tf_duration, per_tf_duration_valid

    def _get_default_labels(self) -> Dict[str, torch.Tensor]:
        """Get default labels for invalid/missing data."""
        n_tfs = len(TIMEFRAMES)
        return {
            'duration': torch.tensor(0, dtype=torch.float32),
            'direction': torch.tensor(0, dtype=torch.long),
            'new_channel': torch.tensor(1, dtype=torch.long),
            'permanent_break': torch.tensor(False, dtype=torch.bool),
            'valid': torch.tensor(False, dtype=torch.bool),
            'duration_valid': torch.tensor(False, dtype=torch.bool),
            'direction_valid': torch.tensor(False, dtype=torch.bool),
            # TSLA labels
            'tsla_bars_to_first_break': torch.tensor(0, dtype=torch.float32),
            'tsla_break_direction': torch.tensor(0, dtype=torch.long),
            'tsla_break_magnitude': torch.tensor(0.0, dtype=torch.float32),
            'tsla_returned_to_channel': torch.tensor(False, dtype=torch.bool),
            'tsla_bounces_after_return': torch.tensor(0, dtype=torch.float32),
            'tsla_channel_continued': torch.tensor(False, dtype=torch.bool),
            'tsla_break_scan_valid': torch.tensor(False, dtype=torch.bool),
            'tsla_duration_to_permanent': torch.tensor(-1, dtype=torch.float32),
            'tsla_avg_bars_outside': torch.tensor(0.0, dtype=torch.float32),
            'tsla_total_bars_outside': torch.tensor(0, dtype=torch.float32),
            'tsla_durability_score': torch.tensor(0.0, dtype=torch.float32),
            'tsla_first_break_returned': torch.tensor(False, dtype=torch.bool),
            'tsla_exit_return_rate': torch.tensor(0.0, dtype=torch.float32),
            'tsla_exits_returned_count': torch.tensor(0, dtype=torch.float32),
            'tsla_exits_stayed_out_count': torch.tensor(0, dtype=torch.float32),
            'tsla_scan_timed_out': torch.tensor(False, dtype=torch.bool),
            'tsla_bars_verified_permanent': torch.tensor(0, dtype=torch.float32),
            'tsla_rsi_at_first_break': torch.tensor(50.0, dtype=torch.float32),
            'tsla_rsi_at_permanent_break': torch.tensor(50.0, dtype=torch.float32),
            'tsla_rsi_at_channel_end': torch.tensor(50.0, dtype=torch.float32),
            'tsla_rsi_overbought_at_break': torch.tensor(0, dtype=torch.long),
            'tsla_rsi_oversold_at_break': torch.tensor(0, dtype=torch.long),
            'tsla_rsi_divergence_at_break': torch.tensor(0, dtype=torch.long),
            'tsla_rsi_trend_in_channel': torch.tensor(0, dtype=torch.long),
            'tsla_rsi_range_in_channel': torch.tensor(0.0, dtype=torch.float32),
            # SPY labels
            'spy_bars_to_first_break': torch.tensor(0, dtype=torch.float32),
            'spy_break_direction': torch.tensor(0, dtype=torch.long),
            'spy_break_magnitude': torch.tensor(0.0, dtype=torch.float32),
            'spy_returned_to_channel': torch.tensor(False, dtype=torch.bool),
            'spy_bounces_after_return': torch.tensor(0, dtype=torch.float32),
            'spy_channel_continued': torch.tensor(False, dtype=torch.bool),
            'spy_break_scan_valid': torch.tensor(False, dtype=torch.bool),
            'spy_duration_to_permanent': torch.tensor(-1, dtype=torch.float32),
            'spy_avg_bars_outside': torch.tensor(0.0, dtype=torch.float32),
            'spy_total_bars_outside': torch.tensor(0, dtype=torch.float32),
            'spy_durability_score': torch.tensor(0.0, dtype=torch.float32),
            'spy_first_break_returned': torch.tensor(False, dtype=torch.bool),
            'spy_exit_return_rate': torch.tensor(0.0, dtype=torch.float32),
            'spy_exits_returned_count': torch.tensor(0, dtype=torch.float32),
            'spy_exits_stayed_out_count': torch.tensor(0, dtype=torch.float32),
            'spy_scan_timed_out': torch.tensor(False, dtype=torch.bool),
            'spy_bars_verified_permanent': torch.tensor(0, dtype=torch.float32),
            'spy_rsi_at_first_break': torch.tensor(50.0, dtype=torch.float32),
            'spy_rsi_at_permanent_break': torch.tensor(50.0, dtype=torch.float32),
            'spy_rsi_at_channel_end': torch.tensor(50.0, dtype=torch.float32),
            'spy_rsi_overbought_at_break': torch.tensor(0, dtype=torch.long),
            'spy_rsi_oversold_at_break': torch.tensor(0, dtype=torch.long),
            'spy_rsi_divergence_at_break': torch.tensor(0, dtype=torch.long),
            'spy_rsi_trend_in_channel': torch.tensor(0, dtype=torch.long),
            'spy_rsi_range_in_channel': torch.tensor(0.0, dtype=torch.float32),
            # Cross-correlation
            **self._extract_cross_correlation_labels(None, None),
            # Per-TF
            'per_tf_duration': torch.zeros(n_tfs, dtype=torch.float32),
            'per_tf_duration_valid': torch.zeros(n_tfs, dtype=torch.bool),
        }

    def get_feature_names(self) -> List[str]:
        """Get the list of feature names."""
        return self.feature_names

    def get_num_features(self) -> int:
        """Get the number of features."""
        return self.num_features


def create_streaming_dataloaders(
    binary_path: str,
    batch_size: int = 128,
    chunk_size: int = 15000,
    target_tf: str = 'daily',
    val_split: float = 0.2,
    num_workers: int = 0,
    prefetch: bool = True,
) -> Tuple['torch.utils.data.DataLoader', 'torch.utils.data.DataLoader', int]:
    """
    Create train and validation dataloaders for streaming dataset.

    Note: Validation uses a subset of indices, not separate chunks.

    Args:
        binary_path: Path to binary sample file
        batch_size: Batch size for dataloaders
        chunk_size: Samples per chunk for streaming
        target_tf: Target timeframe for labels
        val_split: Fraction of data for validation
        num_workers: DataLoader workers (0 recommended for streaming)
        prefetch: Whether to prefetch next chunk

    Returns:
        Tuple of (train_loader, val_loader, num_features)
    """
    from torch.utils.data import DataLoader, Subset
    import random

    # Create dataset
    dataset = ChunkedStreamingDataset(
        binary_path=binary_path,
        chunk_size=chunk_size,
        target_tf=target_tf,
        prefetch=prefetch,
    )

    # Split indices
    n_samples = len(dataset)
    indices = list(range(n_samples))
    random.shuffle(indices)

    val_size = int(n_samples * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    print(f"Train samples: {len(train_indices):,}")
    print(f"Val samples: {len(val_indices):,}")

    # Create subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, dataset.get_num_features()
