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
import os
import random
import torch
from torch.utils.data import Dataset, Sampler
from typing import Any, Dict, List, Optional, Tuple
import threading
from pathlib import Path

from ..binary_index import get_or_build_index
from ..binary_loader import ChannelSample, read_channel_sample
from ..config import TIMEFRAMES


class ChunkOrderedSampler(Sampler):
    """
    Sampler that groups indices by chunk for sequential disk I/O.

    Instead of random access (which causes chunk thrashing — loading a new
    15K-sample chunk for nearly every sample), this sampler ensures all
    samples from one chunk are processed before moving to the next.

    Randomness is preserved at two levels:
    - Chunk order is shuffled each epoch
    - Sample order within each chunk is shuffled

    This eliminates chunk thrashing while maintaining training randomness.
    """

    def __init__(
        self,
        subset_indices: List[int],
        chunk_size: int,
        shuffle_chunks: bool = True,
        shuffle_within: bool = True,
    ):
        """
        Args:
            subset_indices: Original dataset indices used by the Subset
                (e.g., train_indices from the train/val split)
            chunk_size: Chunk size of the streaming dataset
            shuffle_chunks: Shuffle chunk visit order each epoch
            shuffle_within: Shuffle sample order within each chunk
        """
        self.chunk_size = chunk_size
        self.shuffle_chunks = shuffle_chunks
        self.shuffle_within = shuffle_within

        # Group logical indices (0..N-1) by which chunk their original index maps to
        self.chunk_groups: Dict[int, List[int]] = {}
        for logical_idx, original_idx in enumerate(subset_indices):
            chunk_id = original_idx // chunk_size
            if chunk_id not in self.chunk_groups:
                self.chunk_groups[chunk_id] = []
            self.chunk_groups[chunk_id].append(logical_idx)

        self.n_samples = len(subset_indices)

    def __iter__(self):
        chunk_ids = list(self.chunk_groups.keys())
        if self.shuffle_chunks:
            random.shuffle(chunk_ids)

        for chunk_id in chunk_ids:
            indices = list(self.chunk_groups[chunk_id])
            if self.shuffle_within:
                random.shuffle(indices)
            yield from indices

    def __len__(self):
        return self.n_samples


class DistributedChunkOrderedSampler(Sampler):
    """
    Distributed variant of ChunkOrderedSampler for DDP training.

    Assigns whole chunks to ranks via round-robin to preserve chunk locality
    (important for streaming datasets that load chunks from disk).

    Pads smaller ranks to match the largest rank's sample count to prevent
    DDP hangs (all ranks must have the same number of batches).
    """

    def __init__(
        self,
        subset_indices: List[int],
        chunk_size: int,
        shuffle_chunks: bool = True,
        shuffle_within: bool = True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 42,
    ):
        import torch.distributed as dist

        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle_chunks = shuffle_chunks
        self.shuffle_within = shuffle_within
        self.seed = seed
        self.epoch = 0

        # Group logical indices by chunk
        self.chunk_groups: Dict[int, List[int]] = {}
        for logical_idx, original_idx in enumerate(subset_indices):
            chunk_id = original_idx // chunk_size
            if chunk_id not in self.chunk_groups:
                self.chunk_groups[chunk_id] = []
            self.chunk_groups[chunk_id].append(logical_idx)

        # Assign chunks to this rank via round-robin
        all_chunk_ids = sorted(self.chunk_groups.keys())
        self.my_chunk_ids = [
            cid for i, cid in enumerate(all_chunk_ids) if i % self.num_replicas == self.rank
        ]

        # Count samples for this rank
        self.my_n_samples = sum(len(self.chunk_groups[cid]) for cid in self.my_chunk_ids)

        # Compute padded length (all ranks must return same number of samples)
        all_counts = []
        for r in range(self.num_replicas):
            r_chunks = [cid for i, cid in enumerate(all_chunk_ids) if i % self.num_replicas == r]
            all_counts.append(sum(len(self.chunk_groups[cid]) for cid in r_chunks))
        self.total_size = max(all_counts) if all_counts else 0

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling across ranks."""
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        chunk_ids = list(self.my_chunk_ids)
        if self.shuffle_chunks:
            rng.shuffle(chunk_ids)

        indices = []
        for chunk_id in chunk_ids:
            chunk_indices = list(self.chunk_groups[chunk_id])
            if self.shuffle_within:
                rng.shuffle(chunk_indices)
            indices.extend(chunk_indices)

        # Pad to total_size by wrapping
        if len(indices) < self.total_size and len(indices) > 0:
            while len(indices) < self.total_size:
                indices.append(indices[len(indices) % self.my_n_samples])

        return iter(indices[:self.total_size])

    def __len__(self):
        return self.total_size


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
        if int(os.environ.get('RANK', 0)) == 0:
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

        # Pre-build name→index map for fast feature extraction (avoids 222M dict lookups)
        self._feature_name_to_idx = {name: i for i, name in enumerate(self.feature_names)}

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
        ram_mb = self.chunk_size * self.num_features * 4 / (1024 * 1024)  # float32 = 4 bytes
        print(f"  Estimated RAM per chunk: ~{ram_mb:.0f} MB (features only)")

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
        chunk_size = self.current_chunk_end - self.current_chunk_start
        if local_idx < 0 or local_idx >= chunk_size:
            raise IndexError(
                f"Index {idx} (local {local_idx}) out of range for chunk {chunk_idx} "
                f"(size {chunk_size})"
            )

        features = self.current_chunk_features[local_idx]
        # Labels stored as Dict[str, Tensor[N, ...]] — index into each
        labels = {k: v[local_idx] for k, v in self.current_chunk_labels.items()}

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
        import time as _time
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.num_samples)
        chunk_size = end_idx - start_idx

        print(f"Loading chunk {chunk_idx}: samples [{start_idx:,}:{end_idx:,}] ({chunk_size:,} samples)")

        # Load samples from binary file
        samples = []
        t0 = _time.perf_counter()
        with open(self.binary_path, 'rb') as f:
            for i in range(start_idx, end_idx):
                offset = self.offsets[i]
                f.seek(offset)
                sample = read_channel_sample(f, version=self.version, feature_table=self.feature_table)
                samples.append(sample)
                loaded = len(samples)
                if loaded % 5000 == 0:
                    elapsed = _time.perf_counter() - t0
                    rate = loaded / elapsed if elapsed > 0 else 0
                    print(f"  Read {loaded:,}/{chunk_size:,} samples ({elapsed:.1f}s, {rate:.0f} samples/s)")

        read_elapsed = _time.perf_counter() - t0
        print(f"  Read complete: {len(samples):,} samples in {read_elapsed:.1f}s")

        # Convert to tensors
        t1 = _time.perf_counter()
        features, labels = self._convert_to_tensors(samples)
        convert_elapsed = _time.perf_counter() - t1
        print(f"  Tensor conversion: {convert_elapsed:.1f}s")

        # Update cache
        self.current_chunk_features = features
        self.current_chunk_labels = labels
        self.current_chunk_idx = chunk_idx
        self.current_chunk_start = start_idx
        self.current_chunk_end = end_idx

        total_elapsed = _time.perf_counter() - t0
        print(f"  Chunk {chunk_idx} ready: {features.shape}, total {total_elapsed:.1f}s")

    def _start_prefetch(self, chunk_idx: int):
        """Start background thread to prefetch next chunk."""
        # Don't prefetch if already prefetching this chunk
        with self.prefetch_lock:
            if self.prefetch_chunk_idx == chunk_idx:
                return

        def prefetch_worker():
            import time as _time
            try:
                start_idx = chunk_idx * self.chunk_size
                end_idx = min(start_idx + self.chunk_size, self.num_samples)
                chunk_size = end_idx - start_idx

                # Load samples
                samples = []
                t0 = _time.perf_counter()
                with open(self.binary_path, 'rb') as f:
                    for i in range(start_idx, end_idx):
                        offset = self.offsets[i]
                        f.seek(offset)
                        sample = read_channel_sample(f, version=self.version, feature_table=self.feature_table)
                        samples.append(sample)
                        loaded = len(samples)
                        if loaded % 5000 == 0:
                            elapsed = _time.perf_counter() - t0
                            rate = loaded / elapsed if elapsed > 0 else 0
                            print(f"  [prefetch] Read {loaded:,}/{chunk_size:,} ({elapsed:.1f}s, {rate:.0f}/s)")

                read_elapsed = _time.perf_counter() - t0

                # Convert to tensors
                t1 = _time.perf_counter()
                features, labels = self._convert_to_tensors(samples)
                convert_elapsed = _time.perf_counter() - t1

                # Store prefetched data
                with self.prefetch_lock:
                    self.prefetch_chunk_features = features
                    self.prefetch_chunk_labels = labels
                    self.prefetch_chunk_idx = chunk_idx

                total = _time.perf_counter() - t0
                print(f"  [prefetch] Chunk {chunk_idx} ready: read {read_elapsed:.1f}s + convert {convert_elapsed:.1f}s = {total:.1f}s")

            except Exception as e:
                print(f"Prefetch error for chunk {chunk_idx}: {e}")

        # Start background thread
        self.prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        self.prefetch_thread.start()

    def _convert_to_tensors(
        self, samples: List[ChannelSample]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Convert samples to feature tensors and batched label tensors.

        Optimized:
        - Features: pre-allocates [N, num_features] array, fills via name→index map
        - Labels: pre-allocates numpy arrays per label key with defaults,
          fills raw values in one pass (no per-sample torch.tensor() calls),
          batch-converts to tensors once at the end.
        """
        import time as _time
        n = len(samples)
        n_tfs = len(TIMEFRAMES)

        # --- FEATURES: pre-allocate and fill in-place ---
        t0 = _time.perf_counter()
        features = np.zeros((n, self.num_features), dtype=np.float32)
        name_to_idx = self._feature_name_to_idx

        for i, sample in enumerate(samples):
            for name, value in sample.tf_features.items():
                idx = name_to_idx.get(name)
                if idx is not None:
                    features[i, idx] = value

            if (i + 1) % 5000 == 0:
                elapsed = _time.perf_counter() - t0
                print(f"    Features: {i+1:,}/{n:,} samples ({elapsed:.1f}s)")

        features_tensor = torch.from_numpy(features)
        feat_elapsed = _time.perf_counter() - t0

        # --- LABELS: pre-allocate numpy arrays with defaults, fill raw values ---
        t1 = _time.perf_counter()

        # Pre-allocate all label arrays with their default values
        # Float32 scalars
        L_duration = np.zeros(n, dtype=np.float32)
        L_tsla_bars_to_first_break = np.zeros(n, dtype=np.float32)
        L_tsla_break_magnitude = np.zeros(n, dtype=np.float32)
        L_tsla_bounces_after_return = np.zeros(n, dtype=np.float32)
        L_tsla_duration_to_permanent = np.full(n, -1.0, dtype=np.float32)
        L_tsla_avg_bars_outside = np.zeros(n, dtype=np.float32)
        L_tsla_total_bars_outside = np.zeros(n, dtype=np.float32)
        L_tsla_durability_score = np.zeros(n, dtype=np.float32)
        L_tsla_exit_return_rate = np.zeros(n, dtype=np.float32)
        L_tsla_exits_returned_count = np.zeros(n, dtype=np.float32)
        L_tsla_exits_stayed_out_count = np.zeros(n, dtype=np.float32)
        L_tsla_bars_verified_permanent = np.zeros(n, dtype=np.float32)
        L_tsla_rsi_at_first_break = np.full(n, 50.0, dtype=np.float32)
        L_tsla_rsi_at_permanent_break = np.full(n, 50.0, dtype=np.float32)
        L_tsla_rsi_at_channel_end = np.full(n, 50.0, dtype=np.float32)
        L_tsla_rsi_range_in_channel = np.zeros(n, dtype=np.float32)
        # SPY float32 scalars
        L_spy_bars_to_first_break = np.zeros(n, dtype=np.float32)
        L_spy_break_magnitude = np.zeros(n, dtype=np.float32)
        L_spy_bounces_after_return = np.zeros(n, dtype=np.float32)
        L_spy_duration_to_permanent = np.full(n, -1.0, dtype=np.float32)
        L_spy_avg_bars_outside = np.zeros(n, dtype=np.float32)
        L_spy_total_bars_outside = np.zeros(n, dtype=np.float32)
        L_spy_durability_score = np.zeros(n, dtype=np.float32)
        L_spy_exit_return_rate = np.zeros(n, dtype=np.float32)
        L_spy_exits_returned_count = np.zeros(n, dtype=np.float32)
        L_spy_exits_stayed_out_count = np.zeros(n, dtype=np.float32)
        L_spy_bars_verified_permanent = np.zeros(n, dtype=np.float32)
        L_spy_rsi_at_first_break = np.full(n, 50.0, dtype=np.float32)
        L_spy_rsi_at_permanent_break = np.full(n, 50.0, dtype=np.float32)
        L_spy_rsi_at_channel_end = np.full(n, 50.0, dtype=np.float32)
        L_spy_rsi_range_in_channel = np.zeros(n, dtype=np.float32)
        # Cross float32 scalars
        L_cross_break_lag_bars = np.zeros(n, dtype=np.float32)
        L_cross_magnitude_spread = np.zeros(n, dtype=np.float32)
        L_cross_permanent_duration_lag_bars = np.zeros(n, dtype=np.float32)
        L_cross_permanent_duration_spread = np.zeros(n, dtype=np.float32)
        L_cross_durability_spread = np.zeros(n, dtype=np.float32)
        L_cross_avg_bars_outside_spread = np.zeros(n, dtype=np.float32)
        L_cross_total_bars_outside_spread = np.zeros(n, dtype=np.float32)
        L_cross_exit_return_rate_spread = np.zeros(n, dtype=np.float32)
        L_cross_exits_returned_spread = np.zeros(n, dtype=np.float32)
        L_cross_exits_stayed_out_spread = np.zeros(n, dtype=np.float32)
        L_cross_total_exits_spread = np.zeros(n, dtype=np.float32)
        L_cross_bars_verified_spread = np.zeros(n, dtype=np.float32)
        L_cross_exit_timing_correlation = np.zeros(n, dtype=np.float32)
        L_cross_exit_timing_lag_mean = np.zeros(n, dtype=np.float32)
        L_cross_exit_direction_agreement = np.zeros(n, dtype=np.float32)
        L_cross_exit_count_spread = np.zeros(n, dtype=np.float32)
        L_cross_lead_lag_exits = np.zeros(n, dtype=np.float32)
        L_cross_exit_magnitude_correlation = np.zeros(n, dtype=np.float32)
        L_cross_mean_magnitude_spread = np.zeros(n, dtype=np.float32)
        L_cross_exit_duration_correlation = np.zeros(n, dtype=np.float32)
        L_cross_mean_duration_spread = np.zeros(n, dtype=np.float32)
        L_cross_simultaneous_exit_count = np.zeros(n, dtype=np.float32)
        L_cross_rsi_spread_at_break = np.zeros(n, dtype=np.float32)

        # Long scalars
        L_direction = np.zeros(n, dtype=np.int64)
        L_new_channel = np.ones(n, dtype=np.int64)  # default=1 (sideways)
        L_tsla_break_direction = np.zeros(n, dtype=np.int64)
        L_tsla_rsi_overbought_at_break = np.zeros(n, dtype=np.int64)
        L_tsla_rsi_oversold_at_break = np.zeros(n, dtype=np.int64)
        L_tsla_rsi_divergence_at_break = np.zeros(n, dtype=np.int64)
        L_tsla_rsi_trend_in_channel = np.zeros(n, dtype=np.int64)
        L_spy_break_direction = np.zeros(n, dtype=np.int64)
        L_spy_rsi_overbought_at_break = np.zeros(n, dtype=np.int64)
        L_spy_rsi_oversold_at_break = np.zeros(n, dtype=np.int64)
        L_spy_rsi_divergence_at_break = np.zeros(n, dtype=np.int64)
        L_spy_rsi_trend_in_channel = np.zeros(n, dtype=np.int64)
        # Cross long scalars
        L_cross_who_broke_first = np.zeros(n, dtype=np.int64)
        L_cross_exit_cross_correlation_valid = np.zeros(n, dtype=np.int64)
        L_cross_divergence_predicts_reversal = np.zeros(n, dtype=np.int64)
        L_cross_permanent_break_matches_next = np.zeros(n, dtype=np.int64)
        L_cross_next_channel_direction_aligned = np.zeros(n, dtype=np.int64)
        L_cross_next_channel_quality_aligned = np.zeros(n, dtype=np.int64)
        L_cross_best_next_channel_tsla_vs_spy = np.zeros(n, dtype=np.int64)
        L_cross_rsi_aligned_at_break = np.zeros(n, dtype=np.int64)
        L_cross_rsi_divergence_aligned = np.zeros(n, dtype=np.int64)
        L_cross_tsla_rsi_higher_at_break = np.zeros(n, dtype=np.int64)
        L_cross_overbought_predicts_down_break = np.zeros(n, dtype=np.int64)
        L_cross_oversold_predicts_up_break = np.zeros(n, dtype=np.int64)

        # Bool scalars
        L_permanent_break = np.zeros(n, dtype=np.bool_)
        L_valid = np.zeros(n, dtype=np.bool_)
        L_duration_valid = np.zeros(n, dtype=np.bool_)
        L_direction_valid = np.zeros(n, dtype=np.bool_)
        L_tsla_returned_to_channel = np.zeros(n, dtype=np.bool_)
        L_tsla_channel_continued = np.zeros(n, dtype=np.bool_)
        L_tsla_break_scan_valid = np.zeros(n, dtype=np.bool_)
        L_tsla_first_break_returned = np.zeros(n, dtype=np.bool_)
        L_tsla_scan_timed_out = np.zeros(n, dtype=np.bool_)
        L_spy_returned_to_channel = np.zeros(n, dtype=np.bool_)
        L_spy_channel_continued = np.zeros(n, dtype=np.bool_)
        L_spy_break_scan_valid = np.zeros(n, dtype=np.bool_)
        L_spy_first_break_returned = np.zeros(n, dtype=np.bool_)
        L_spy_scan_timed_out = np.zeros(n, dtype=np.bool_)
        # Cross bool scalars
        L_cross_direction_aligned = np.zeros(n, dtype=np.bool_)
        L_cross_tsla_broke_first = np.zeros(n, dtype=np.bool_)
        L_cross_spy_broke_first = np.zeros(n, dtype=np.bool_)
        L_cross_both_returned = np.zeros(n, dtype=np.bool_)
        L_cross_both_permanent = np.zeros(n, dtype=np.bool_)
        L_cross_return_pattern_aligned = np.zeros(n, dtype=np.bool_)
        L_cross_continuation_aligned = np.zeros(n, dtype=np.bool_)
        L_cross_valid = np.zeros(n, dtype=np.bool_)
        L_cross_tsla_permanent_first = np.zeros(n, dtype=np.bool_)
        L_cross_spy_permanent_first = np.zeros(n, dtype=np.bool_)
        L_cross_both_high_durability = np.zeros(n, dtype=np.bool_)
        L_cross_both_low_durability = np.zeros(n, dtype=np.bool_)
        L_cross_durability_aligned = np.zeros(n, dtype=np.bool_)
        L_cross_tsla_more_durable = np.zeros(n, dtype=np.bool_)
        L_cross_spy_more_durable = np.zeros(n, dtype=np.bool_)
        L_cross_permanent_dynamics_valid = np.zeros(n, dtype=np.bool_)
        L_cross_exit_return_rate_aligned = np.zeros(n, dtype=np.bool_)
        L_cross_tsla_more_resilient = np.zeros(n, dtype=np.bool_)
        L_cross_spy_more_resilient = np.zeros(n, dtype=np.bool_)
        L_cross_both_scan_timed_out = np.zeros(n, dtype=np.bool_)
        L_cross_scan_timeout_aligned = np.zeros(n, dtype=np.bool_)
        L_cross_both_first_returned_then_permanent = np.zeros(n, dtype=np.bool_)
        L_cross_both_never_returned = np.zeros(n, dtype=np.bool_)
        L_cross_exit_verification_valid = np.zeros(n, dtype=np.bool_)

        # Per-TF vectors
        L_per_tf_duration = np.zeros((n, n_tfs), dtype=np.float32)
        L_per_tf_duration_valid = np.zeros((n, n_tfs), dtype=np.bool_)

        # --- Fill raw values per sample (no torch.tensor calls) ---
        for i, sample in enumerate(samples):
            window = self.target_window if self.target_window is not None else sample.best_window
            window_labels = sample.labels_per_window.get(window, {})

            # Detect old vs new structure
            if 'tsla' in window_labels:
                tsla_labels_dict = window_labels.get('tsla', {})
                spy_labels_dict = window_labels.get('spy', {})
                tf_labels = tsla_labels_dict.get(self.target_tf)
                spy_tf_labels = spy_labels_dict.get(self.target_tf)
            else:
                tf_labels = window_labels.get(self.target_tf)
                spy_tf_labels = None

            if tf_labels is None:
                # defaults already set in array initialization
                # Per-TF also defaults to zeros
                continue

            # Core labels
            L_duration[i] = tf_labels.duration_bars
            L_direction[i] = tf_labels.break_direction
            L_new_channel[i] = tf_labels.next_channel_direction
            L_permanent_break[i] = tf_labels.permanent_break
            L_valid[i] = tf_labels.duration_valid or tf_labels.direction_valid
            L_duration_valid[i] = tf_labels.duration_valid
            L_direction_valid[i] = tf_labels.direction_valid

            # TSLA break scan labels
            L_tsla_bars_to_first_break[i] = getattr(tf_labels, 'bars_to_first_break', 0)
            L_tsla_break_direction[i] = getattr(tf_labels, 'break_direction', 0)
            L_tsla_break_magnitude[i] = getattr(tf_labels, 'break_magnitude', 0.0)
            L_tsla_returned_to_channel[i] = getattr(tf_labels, 'returned_to_channel', False)
            L_tsla_bounces_after_return[i] = getattr(tf_labels, 'bounces_after_return', 0)
            L_tsla_channel_continued[i] = getattr(tf_labels, 'channel_continued', False)
            L_tsla_break_scan_valid[i] = getattr(tf_labels, 'break_scan_valid', False)
            L_tsla_duration_to_permanent[i] = getattr(tf_labels, 'duration_to_permanent', -1)
            L_tsla_avg_bars_outside[i] = getattr(tf_labels, 'avg_bars_outside', 0.0)
            L_tsla_total_bars_outside[i] = getattr(tf_labels, 'total_bars_outside', 0)
            L_tsla_durability_score[i] = getattr(tf_labels, 'durability_score', 0.0)
            L_tsla_first_break_returned[i] = getattr(tf_labels, 'first_break_returned', False)
            L_tsla_exit_return_rate[i] = getattr(tf_labels, 'exit_return_rate', 0.0)
            L_tsla_exits_returned_count[i] = getattr(tf_labels, 'exits_returned_count', 0)
            L_tsla_exits_stayed_out_count[i] = getattr(tf_labels, 'exits_stayed_out_count', 0)
            L_tsla_scan_timed_out[i] = getattr(tf_labels, 'scan_timed_out', False)
            L_tsla_bars_verified_permanent[i] = getattr(tf_labels, 'bars_verified_permanent', 0)
            # TSLA RSI
            L_tsla_rsi_at_first_break[i] = getattr(tf_labels, 'rsi_at_first_break', 50.0)
            L_tsla_rsi_at_permanent_break[i] = getattr(tf_labels, 'rsi_at_permanent_break', 50.0)
            L_tsla_rsi_at_channel_end[i] = getattr(tf_labels, 'rsi_at_channel_end', 50.0)
            L_tsla_rsi_overbought_at_break[i] = int(getattr(tf_labels, 'rsi_overbought_at_break', False))
            L_tsla_rsi_oversold_at_break[i] = int(getattr(tf_labels, 'rsi_oversold_at_break', False))
            L_tsla_rsi_divergence_at_break[i] = getattr(tf_labels, 'rsi_divergence_at_break', 0)
            L_tsla_rsi_trend_in_channel[i] = getattr(tf_labels, 'rsi_trend_in_channel', 0)
            L_tsla_rsi_range_in_channel[i] = getattr(tf_labels, 'rsi_range_in_channel', 0.0)

            # SPY labels
            spy_src = spy_tf_labels if spy_tf_labels is not None else None
            if spy_src is not None:
                # New structure: separate SPY object
                L_spy_bars_to_first_break[i] = getattr(spy_src, 'bars_to_first_break', 0)
                L_spy_break_direction[i] = getattr(spy_src, 'break_direction', 0)
                L_spy_break_magnitude[i] = getattr(spy_src, 'break_magnitude', 0.0)
                L_spy_returned_to_channel[i] = getattr(spy_src, 'returned_to_channel', False)
                L_spy_bounces_after_return[i] = getattr(spy_src, 'bounces_after_return', 0)
                L_spy_channel_continued[i] = getattr(spy_src, 'channel_continued', False)
                L_spy_break_scan_valid[i] = getattr(spy_src, 'break_scan_valid', False)
                L_spy_duration_to_permanent[i] = getattr(spy_src, 'duration_to_permanent', -1)
                L_spy_avg_bars_outside[i] = getattr(spy_src, 'avg_bars_outside', 0.0)
                L_spy_total_bars_outside[i] = getattr(spy_src, 'total_bars_outside', 0)
                L_spy_durability_score[i] = getattr(spy_src, 'durability_score', 0.0)
                L_spy_first_break_returned[i] = getattr(spy_src, 'first_break_returned', False)
                L_spy_exit_return_rate[i] = getattr(spy_src, 'exit_return_rate', 0.0)
                L_spy_exits_returned_count[i] = getattr(spy_src, 'exits_returned_count', 0)
                L_spy_exits_stayed_out_count[i] = getattr(spy_src, 'exits_stayed_out_count', 0)
                L_spy_scan_timed_out[i] = getattr(spy_src, 'scan_timed_out', False)
                L_spy_bars_verified_permanent[i] = getattr(spy_src, 'bars_verified_permanent', 0)
                L_spy_rsi_at_first_break[i] = getattr(spy_src, 'rsi_at_first_break', 50.0)
                L_spy_rsi_at_permanent_break[i] = getattr(spy_src, 'rsi_at_permanent_break', 50.0)
                L_spy_rsi_at_channel_end[i] = getattr(spy_src, 'rsi_at_channel_end', 50.0)
                L_spy_rsi_overbought_at_break[i] = int(getattr(spy_src, 'rsi_overbought_at_break', False))
                L_spy_rsi_oversold_at_break[i] = int(getattr(spy_src, 'rsi_oversold_at_break', False))
                L_spy_rsi_divergence_at_break[i] = getattr(spy_src, 'rsi_divergence_at_break', 0)
                L_spy_rsi_trend_in_channel[i] = getattr(spy_src, 'rsi_trend_in_channel', 0)
                L_spy_rsi_range_in_channel[i] = getattr(spy_src, 'rsi_range_in_channel', 0.0)
            else:
                # Old structure: spy_ prefixed fields on tf_labels
                L_spy_bars_to_first_break[i] = getattr(tf_labels, 'spy_bars_to_first_break', 0)
                L_spy_break_direction[i] = getattr(tf_labels, 'spy_break_direction', 0)
                L_spy_break_magnitude[i] = getattr(tf_labels, 'spy_break_magnitude', 0.0)
                L_spy_returned_to_channel[i] = getattr(tf_labels, 'spy_returned_to_channel', False)
                L_spy_bounces_after_return[i] = getattr(tf_labels, 'spy_bounces_after_return', 0)
                L_spy_channel_continued[i] = getattr(tf_labels, 'spy_channel_continued', False)
                L_spy_break_scan_valid[i] = getattr(tf_labels, 'break_scan_valid', False)
                L_spy_duration_to_permanent[i] = getattr(tf_labels, 'spy_duration_to_permanent', -1)
                L_spy_avg_bars_outside[i] = getattr(tf_labels, 'spy_avg_bars_outside', 0.0)
                L_spy_total_bars_outside[i] = getattr(tf_labels, 'spy_total_bars_outside', 0)
                L_spy_durability_score[i] = getattr(tf_labels, 'spy_durability_score', 0.0)
                L_spy_first_break_returned[i] = getattr(tf_labels, 'spy_first_break_returned', False)
                L_spy_exit_return_rate[i] = getattr(tf_labels, 'spy_exit_return_rate', 0.0)
                L_spy_exits_returned_count[i] = getattr(tf_labels, 'spy_exits_returned_count', 0)
                L_spy_exits_stayed_out_count[i] = getattr(tf_labels, 'spy_exits_stayed_out_count', 0)
                L_spy_scan_timed_out[i] = getattr(tf_labels, 'spy_scan_timed_out', False)
                L_spy_bars_verified_permanent[i] = getattr(tf_labels, 'spy_bars_verified_permanent', 0)
                L_spy_rsi_at_first_break[i] = getattr(tf_labels, 'spy_rsi_at_first_break', 50.0)
                L_spy_rsi_at_permanent_break[i] = getattr(tf_labels, 'spy_rsi_at_permanent_break', 50.0)
                L_spy_rsi_at_channel_end[i] = getattr(tf_labels, 'spy_rsi_at_channel_end', 50.0)
                L_spy_rsi_overbought_at_break[i] = int(getattr(tf_labels, 'spy_rsi_overbought_at_break', False))
                L_spy_rsi_oversold_at_break[i] = int(getattr(tf_labels, 'spy_rsi_oversold_at_break', False))
                L_spy_rsi_divergence_at_break[i] = getattr(tf_labels, 'spy_rsi_divergence_at_break', 0)
                L_spy_rsi_trend_in_channel[i] = getattr(tf_labels, 'spy_rsi_trend_in_channel', 0)
                L_spy_rsi_range_in_channel[i] = getattr(tf_labels, 'spy_rsi_range_in_channel', 0.0)

            # Cross-correlation labels
            tsla_valid = getattr(tf_labels, 'break_scan_valid', False)
            _spy_for_cross = spy_tf_labels
            if _spy_for_cross is not None and _spy_for_cross is not tf_labels:
                spy_valid = getattr(_spy_for_cross, 'break_scan_valid', False)
            else:
                spy_bars = getattr(tf_labels, 'spy_bars_to_first_break', 0)
                spy_valid = tsla_valid and spy_bars > 0
                _spy_for_cross = None  # use combined path

            if tsla_valid and spy_valid:
                if _spy_for_cross is not None:
                    # Separate objects
                    _tsla = tf_labels
                    _spy = _spy_for_cross
                    tsla_dir = getattr(_tsla, 'break_direction', 0)
                    spy_dir = getattr(_spy, 'break_direction', 0)
                    tsla_bars = getattr(_tsla, 'bars_to_first_break', 0)
                    spy_bars_v = getattr(_spy, 'bars_to_first_break', 0)
                    tsla_returned = getattr(_tsla, 'returned_to_channel', False)
                    spy_returned = getattr(_spy, 'returned_to_channel', False)
                    tsla_mag = getattr(_tsla, 'break_magnitude', 0.0)
                    spy_mag = getattr(_spy, 'break_magnitude', 0.0)
                    tsla_dur_score = getattr(_tsla, 'durability_score', 0.0)
                    spy_dur_score = getattr(_spy, 'durability_score', 0.0)
                    tsla_avg_bars_out = getattr(_tsla, 'avg_bars_outside', 0.0)
                    spy_avg_bars_out = getattr(_spy, 'avg_bars_outside', 0.0)
                    tsla_total_bars_out = getattr(_tsla, 'total_bars_outside', 0)
                    spy_total_bars_out = getattr(_spy, 'total_bars_outside', 0)
                    tsla_cont = getattr(_tsla, 'channel_continued', False)
                    spy_cont = getattr(_spy, 'channel_continued', False)
                else:
                    # Combined object
                    tsla_dir = getattr(tf_labels, 'break_direction', 0)
                    spy_dir = getattr(tf_labels, 'spy_break_direction', 0)
                    tsla_bars = getattr(tf_labels, 'bars_to_first_break', 0)
                    spy_bars_v = getattr(tf_labels, 'spy_bars_to_first_break', 0)
                    tsla_returned = getattr(tf_labels, 'returned_to_channel', False)
                    spy_returned = getattr(tf_labels, 'spy_returned_to_channel', False)
                    tsla_mag = getattr(tf_labels, 'break_magnitude', 0.0)
                    spy_mag = getattr(tf_labels, 'spy_break_magnitude', 0.0)
                    tsla_dur_score = getattr(tf_labels, 'durability_score', 0.0)
                    spy_dur_score = getattr(tf_labels, 'spy_durability_score', 0.0)
                    tsla_avg_bars_out = getattr(tf_labels, 'avg_bars_outside', 0.0)
                    spy_avg_bars_out = getattr(tf_labels, 'spy_avg_bars_outside', 0.0)
                    tsla_total_bars_out = getattr(tf_labels, 'total_bars_outside', 0)
                    spy_total_bars_out = getattr(tf_labels, 'spy_total_bars_outside', 0)
                    tsla_cont = getattr(tf_labels, 'channel_continued', False)
                    spy_cont = getattr(tf_labels, 'spy_channel_continued', False)

                direction_aligned = tsla_dir == spy_dir
                tsla_broke_first = tsla_bars < spy_bars_v
                spy_broke_first = spy_bars_v < tsla_bars
                who_broke_first = 1 if tsla_broke_first else (2 if spy_broke_first else 0)

                L_cross_direction_aligned[i] = direction_aligned
                L_cross_tsla_broke_first[i] = tsla_broke_first
                L_cross_spy_broke_first[i] = spy_broke_first
                L_cross_break_lag_bars[i] = abs(tsla_bars - spy_bars_v)
                L_cross_magnitude_spread[i] = tsla_mag - spy_mag
                L_cross_both_returned[i] = tsla_returned and spy_returned
                L_cross_both_permanent[i] = not tsla_returned and not spy_returned
                L_cross_return_pattern_aligned[i] = (
                    (tsla_returned and spy_returned) or (not tsla_returned and not spy_returned)
                )
                L_cross_continuation_aligned[i] = tsla_cont == spy_cont
                L_cross_who_broke_first[i] = who_broke_first
                L_cross_valid[i] = True
                L_cross_durability_spread[i] = tsla_dur_score - spy_dur_score
                L_cross_avg_bars_outside_spread[i] = tsla_avg_bars_out - spy_avg_bars_out
                L_cross_total_bars_outside_spread[i] = tsla_total_bars_out - spy_total_bars_out
                L_cross_tsla_more_durable[i] = tsla_dur_score > spy_dur_score
                L_cross_spy_more_durable[i] = spy_dur_score > tsla_dur_score

            # Per-TF duration labels
            ptf_window = sample.best_window
            ptf_window_labels = sample.labels_per_window.get(ptf_window, {})
            if 'tsla' in ptf_window_labels:
                ptf_dict = ptf_window_labels.get('tsla', {})
            else:
                ptf_dict = ptf_window_labels
            for tf_idx, tf in enumerate(TIMEFRAMES):
                tf_label = ptf_dict.get(tf)
                if tf_label is not None:
                    dur = getattr(tf_label, 'duration_bars', 0)
                    L_per_tf_duration[i, tf_idx] = float(dur) if dur is not None else 0.0
                    L_per_tf_duration_valid[i, tf_idx] = bool(getattr(tf_label, 'duration_valid', False))

            if (i + 1) % 5000 == 0:
                elapsed = _time.perf_counter() - t1
                print(f"    Labels: {i+1:,}/{n:,} samples ({elapsed:.1f}s)")

        # --- Batch convert all arrays to tensors (one allocation per key) ---
        labels_batched = {
            # Core
            'duration': torch.from_numpy(L_duration),
            'direction': torch.from_numpy(L_direction),
            'new_channel': torch.from_numpy(L_new_channel),
            'permanent_break': torch.from_numpy(L_permanent_break),
            'valid': torch.from_numpy(L_valid),
            'duration_valid': torch.from_numpy(L_duration_valid),
            'direction_valid': torch.from_numpy(L_direction_valid),
            # TSLA
            'tsla_bars_to_first_break': torch.from_numpy(L_tsla_bars_to_first_break),
            'tsla_break_direction': torch.from_numpy(L_tsla_break_direction),
            'tsla_break_magnitude': torch.from_numpy(L_tsla_break_magnitude),
            'tsla_returned_to_channel': torch.from_numpy(L_tsla_returned_to_channel),
            'tsla_bounces_after_return': torch.from_numpy(L_tsla_bounces_after_return),
            'tsla_channel_continued': torch.from_numpy(L_tsla_channel_continued),
            'tsla_break_scan_valid': torch.from_numpy(L_tsla_break_scan_valid),
            'tsla_duration_to_permanent': torch.from_numpy(L_tsla_duration_to_permanent),
            'tsla_avg_bars_outside': torch.from_numpy(L_tsla_avg_bars_outside),
            'tsla_total_bars_outside': torch.from_numpy(L_tsla_total_bars_outside),
            'tsla_durability_score': torch.from_numpy(L_tsla_durability_score),
            'tsla_first_break_returned': torch.from_numpy(L_tsla_first_break_returned),
            'tsla_exit_return_rate': torch.from_numpy(L_tsla_exit_return_rate),
            'tsla_exits_returned_count': torch.from_numpy(L_tsla_exits_returned_count),
            'tsla_exits_stayed_out_count': torch.from_numpy(L_tsla_exits_stayed_out_count),
            'tsla_scan_timed_out': torch.from_numpy(L_tsla_scan_timed_out),
            'tsla_bars_verified_permanent': torch.from_numpy(L_tsla_bars_verified_permanent),
            'tsla_rsi_at_first_break': torch.from_numpy(L_tsla_rsi_at_first_break),
            'tsla_rsi_at_permanent_break': torch.from_numpy(L_tsla_rsi_at_permanent_break),
            'tsla_rsi_at_channel_end': torch.from_numpy(L_tsla_rsi_at_channel_end),
            'tsla_rsi_overbought_at_break': torch.from_numpy(L_tsla_rsi_overbought_at_break),
            'tsla_rsi_oversold_at_break': torch.from_numpy(L_tsla_rsi_oversold_at_break),
            'tsla_rsi_divergence_at_break': torch.from_numpy(L_tsla_rsi_divergence_at_break),
            'tsla_rsi_trend_in_channel': torch.from_numpy(L_tsla_rsi_trend_in_channel),
            'tsla_rsi_range_in_channel': torch.from_numpy(L_tsla_rsi_range_in_channel),
            # SPY
            'spy_bars_to_first_break': torch.from_numpy(L_spy_bars_to_first_break),
            'spy_break_direction': torch.from_numpy(L_spy_break_direction),
            'spy_break_magnitude': torch.from_numpy(L_spy_break_magnitude),
            'spy_returned_to_channel': torch.from_numpy(L_spy_returned_to_channel),
            'spy_bounces_after_return': torch.from_numpy(L_spy_bounces_after_return),
            'spy_channel_continued': torch.from_numpy(L_spy_channel_continued),
            'spy_break_scan_valid': torch.from_numpy(L_spy_break_scan_valid),
            'spy_duration_to_permanent': torch.from_numpy(L_spy_duration_to_permanent),
            'spy_avg_bars_outside': torch.from_numpy(L_spy_avg_bars_outside),
            'spy_total_bars_outside': torch.from_numpy(L_spy_total_bars_outside),
            'spy_durability_score': torch.from_numpy(L_spy_durability_score),
            'spy_first_break_returned': torch.from_numpy(L_spy_first_break_returned),
            'spy_exit_return_rate': torch.from_numpy(L_spy_exit_return_rate),
            'spy_exits_returned_count': torch.from_numpy(L_spy_exits_returned_count),
            'spy_exits_stayed_out_count': torch.from_numpy(L_spy_exits_stayed_out_count),
            'spy_scan_timed_out': torch.from_numpy(L_spy_scan_timed_out),
            'spy_bars_verified_permanent': torch.from_numpy(L_spy_bars_verified_permanent),
            'spy_rsi_at_first_break': torch.from_numpy(L_spy_rsi_at_first_break),
            'spy_rsi_at_permanent_break': torch.from_numpy(L_spy_rsi_at_permanent_break),
            'spy_rsi_at_channel_end': torch.from_numpy(L_spy_rsi_at_channel_end),
            'spy_rsi_overbought_at_break': torch.from_numpy(L_spy_rsi_overbought_at_break),
            'spy_rsi_oversold_at_break': torch.from_numpy(L_spy_rsi_oversold_at_break),
            'spy_rsi_divergence_at_break': torch.from_numpy(L_spy_rsi_divergence_at_break),
            'spy_rsi_trend_in_channel': torch.from_numpy(L_spy_rsi_trend_in_channel),
            'spy_rsi_range_in_channel': torch.from_numpy(L_spy_rsi_range_in_channel),
            # Cross-correlation
            'cross_direction_aligned': torch.from_numpy(L_cross_direction_aligned),
            'cross_tsla_broke_first': torch.from_numpy(L_cross_tsla_broke_first),
            'cross_spy_broke_first': torch.from_numpy(L_cross_spy_broke_first),
            'cross_break_lag_bars': torch.from_numpy(L_cross_break_lag_bars),
            'cross_magnitude_spread': torch.from_numpy(L_cross_magnitude_spread),
            'cross_both_returned': torch.from_numpy(L_cross_both_returned),
            'cross_both_permanent': torch.from_numpy(L_cross_both_permanent),
            'cross_return_pattern_aligned': torch.from_numpy(L_cross_return_pattern_aligned),
            'cross_continuation_aligned': torch.from_numpy(L_cross_continuation_aligned),
            'cross_who_broke_first': torch.from_numpy(L_cross_who_broke_first),
            'cross_valid': torch.from_numpy(L_cross_valid),
            'cross_tsla_permanent_first': torch.from_numpy(L_cross_tsla_permanent_first),
            'cross_spy_permanent_first': torch.from_numpy(L_cross_spy_permanent_first),
            'cross_permanent_duration_lag_bars': torch.from_numpy(L_cross_permanent_duration_lag_bars),
            'cross_permanent_duration_spread': torch.from_numpy(L_cross_permanent_duration_spread),
            'cross_durability_spread': torch.from_numpy(L_cross_durability_spread),
            'cross_avg_bars_outside_spread': torch.from_numpy(L_cross_avg_bars_outside_spread),
            'cross_total_bars_outside_spread': torch.from_numpy(L_cross_total_bars_outside_spread),
            'cross_both_high_durability': torch.from_numpy(L_cross_both_high_durability),
            'cross_both_low_durability': torch.from_numpy(L_cross_both_low_durability),
            'cross_durability_aligned': torch.from_numpy(L_cross_durability_aligned),
            'cross_tsla_more_durable': torch.from_numpy(L_cross_tsla_more_durable),
            'cross_spy_more_durable': torch.from_numpy(L_cross_spy_more_durable),
            'cross_permanent_dynamics_valid': torch.from_numpy(L_cross_permanent_dynamics_valid),
            'cross_exit_return_rate_spread': torch.from_numpy(L_cross_exit_return_rate_spread),
            'cross_exit_return_rate_aligned': torch.from_numpy(L_cross_exit_return_rate_aligned),
            'cross_tsla_more_resilient': torch.from_numpy(L_cross_tsla_more_resilient),
            'cross_spy_more_resilient': torch.from_numpy(L_cross_spy_more_resilient),
            'cross_exits_returned_spread': torch.from_numpy(L_cross_exits_returned_spread),
            'cross_exits_stayed_out_spread': torch.from_numpy(L_cross_exits_stayed_out_spread),
            'cross_total_exits_spread': torch.from_numpy(L_cross_total_exits_spread),
            'cross_both_scan_timed_out': torch.from_numpy(L_cross_both_scan_timed_out),
            'cross_scan_timeout_aligned': torch.from_numpy(L_cross_scan_timeout_aligned),
            'cross_bars_verified_spread': torch.from_numpy(L_cross_bars_verified_spread),
            'cross_both_first_returned_then_permanent': torch.from_numpy(L_cross_both_first_returned_then_permanent),
            'cross_both_never_returned': torch.from_numpy(L_cross_both_never_returned),
            'cross_exit_verification_valid': torch.from_numpy(L_cross_exit_verification_valid),
            'cross_exit_timing_correlation': torch.from_numpy(L_cross_exit_timing_correlation),
            'cross_exit_timing_lag_mean': torch.from_numpy(L_cross_exit_timing_lag_mean),
            'cross_exit_direction_agreement': torch.from_numpy(L_cross_exit_direction_agreement),
            'cross_exit_count_spread': torch.from_numpy(L_cross_exit_count_spread),
            'cross_lead_lag_exits': torch.from_numpy(L_cross_lead_lag_exits),
            'cross_exit_magnitude_correlation': torch.from_numpy(L_cross_exit_magnitude_correlation),
            'cross_mean_magnitude_spread': torch.from_numpy(L_cross_mean_magnitude_spread),
            'cross_exit_duration_correlation': torch.from_numpy(L_cross_exit_duration_correlation),
            'cross_mean_duration_spread': torch.from_numpy(L_cross_mean_duration_spread),
            'cross_simultaneous_exit_count': torch.from_numpy(L_cross_simultaneous_exit_count),
            'cross_exit_cross_correlation_valid': torch.from_numpy(L_cross_exit_cross_correlation_valid),
            'cross_divergence_predicts_reversal': torch.from_numpy(L_cross_divergence_predicts_reversal),
            'cross_permanent_break_matches_next': torch.from_numpy(L_cross_permanent_break_matches_next),
            'cross_next_channel_direction_aligned': torch.from_numpy(L_cross_next_channel_direction_aligned),
            'cross_next_channel_quality_aligned': torch.from_numpy(L_cross_next_channel_quality_aligned),
            'cross_best_next_channel_tsla_vs_spy': torch.from_numpy(L_cross_best_next_channel_tsla_vs_spy),
            'cross_rsi_aligned_at_break': torch.from_numpy(L_cross_rsi_aligned_at_break),
            'cross_rsi_divergence_aligned': torch.from_numpy(L_cross_rsi_divergence_aligned),
            'cross_tsla_rsi_higher_at_break': torch.from_numpy(L_cross_tsla_rsi_higher_at_break),
            'cross_rsi_spread_at_break': torch.from_numpy(L_cross_rsi_spread_at_break),
            'cross_overbought_predicts_down_break': torch.from_numpy(L_cross_overbought_predicts_down_break),
            'cross_oversold_predicts_up_break': torch.from_numpy(L_cross_oversold_predicts_up_break),
            # Per-TF
            'per_tf_duration': torch.from_numpy(L_per_tf_duration),
            'per_tf_duration_valid': torch.from_numpy(L_per_tf_duration_valid),
        }

        label_elapsed = _time.perf_counter() - t1
        print(f"    Conversion breakdown: features {feat_elapsed:.1f}s + labels {label_elapsed:.1f}s")

        return features_tensor, labels_batched

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
    sorted_reads: bool = False,
    max_samples: Optional[int] = None,
    distributed: bool = False,
    seed: int = 42,
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
        sorted_reads: Group samples by chunk for sequential disk I/O.
            Prevents chunk thrashing (loading new 15K-sample chunk per sample).
            Chunks and samples within chunks are still shuffled for training.
        max_samples: If set, cap total samples used (train + val) to this number.

    Returns:
        Tuple of (train_loader, val_loader, num_features)
    """
    from torch.utils.data import DataLoader, Subset

    # Disable prefetch when sorted_reads is on.
    # Prefetch assumes sequential chunk order (N → N+1), but ChunkOrderedSampler
    # visits chunks in shuffled order, so prefetch almost never hits (<1%).
    # Disabling it frees ~2.8GB RAM (one full chunk), allowing larger chunk sizes
    # which is a bigger throughput win (~50%) than working prefetch could provide (~6%).
    if sorted_reads:
        prefetch = False

    # Create dataset
    dataset = ChunkedStreamingDataset(
        binary_path=binary_path,
        chunk_size=chunk_size,
        target_tf=target_tf,
        prefetch=prefetch,
    )

    # Split indices
    n_samples = len(dataset)
    if max_samples is not None and max_samples < n_samples:
        n_samples = max_samples
    indices = list(range(n_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_size = int(n_samples * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    if int(os.environ.get('RANK', 0)) == 0:
        print(f"Train samples: {len(train_indices):,}")
        print(f"Val samples: {len(val_indices):,}")

    # Create subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # Create dataloaders
    if distributed:
        # DDP: chunk-ordered sampling with rank-based chunk assignment
        train_sampler = DistributedChunkOrderedSampler(
            train_indices, chunk_size, shuffle_chunks=True, shuffle_within=True, seed=seed,
        )
        val_sampler = DistributedChunkOrderedSampler(
            val_indices, chunk_size, shuffle_chunks=False, shuffle_within=False, seed=seed,
        )

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    elif sorted_reads:
        if int(os.environ.get('RANK', 0)) == 0:
            print(f"Sorted reads: ON (chunk-ordered sampling, {len(set(i // chunk_size for i in train_indices))} train chunks, prefetch OFF)")
        train_sampler = ChunkOrderedSampler(train_indices, chunk_size)
        val_sampler = ChunkOrderedSampler(val_indices, chunk_size, shuffle_chunks=False)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
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
