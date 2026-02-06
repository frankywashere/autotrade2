"""
Flat format dataset for instant loading of pre-converted training data.

The .bin format stores variable-length records that require per-sample Python
parsing (~4 hours for 996K samples). This module provides:

1. convert_bin_to_flat(): One-time conversion from .bin to .flat directory
   - Reads .bin in chunks (streaming, low RAM)
   - Writes features as a single contiguous .npy file (mmap-able)
   - Writes each label as a separate .npy file
   - Saves feature names and metadata

2. FlatDataset: PyTorch Dataset that loads .flat directory
   - Memory-maps features (instant, OS handles paging)
   - Loads labels into RAM (small)
   - __getitem__ returns same format as ChunkedStreamingDataset

Usage:
    # One-time conversion
    python3 -m v15.pipeline convert --samples data.bin --output data.flat

    # Training (auto-detects .flat directory)
    python3 -m v15.pipeline train --samples data.flat ...
"""

import json
import numpy as np
import random
import time
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple


def convert_bin_to_flat(
    bin_path: str,
    output_dir: str,
    chunk_size: int = 15000,
    target_tf: str = 'daily',
):
    """
    Convert .bin sample file to .flat directory for instant loading.

    Reads the .bin in chunks using the streaming infrastructure (low RAM),
    converts features and labels, writes to .npy files.

    Args:
        bin_path: Path to .bin sample file
        output_dir: Path to output .flat directory
        chunk_size: Samples per chunk during conversion
        target_tf: Target timeframe for label extraction
    """
    from .streaming_dataset import ChunkedStreamingDataset
    from ..binary_loader import load_samples
    import struct

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = output_dir / 'labels'
    labels_dir.mkdir(exist_ok=True)

    # Check file version - v2 needs feature_names, v3 has them embedded
    with open(bin_path, 'rb') as f:
        f.read(8)  # magic
        version = struct.unpack('<I', f.read(4))[0]

    feature_names = None
    if version < 3:
        # v2 format: need to provide feature names from config
        from ..features import get_tf_feature_names
        feature_names = get_tf_feature_names()
        print(f"  Note: v2 format detected, using {len(feature_names)} feature names from config")

    # Create streaming dataset for reading (no prefetch needed)
    dataset = ChunkedStreamingDataset(
        binary_path=bin_path,
        chunk_size=chunk_size,
        target_tf=target_tf,
        prefetch=False,
        feature_names=feature_names,  # None for v3 (uses embedded), list for v2
    )

    N = dataset.num_samples
    F = dataset.num_features
    num_chunks = (N + chunk_size - 1) // chunk_size

    print(f"\nConverting {bin_path} -> {output_dir}")
    print(f"  Samples: {N:,}")
    print(f"  Features: {F:,}")
    print(f"  Target TF: {target_tf}")
    print(f"  Chunks: {num_chunks} (size {chunk_size:,})")

    # Create features memmap (.npy format, supports np.load with mmap_mode)
    features_path = str(output_dir / 'features.npy')
    print(f"\n  Creating features array: [{N:,} x {F:,}] float32 ({N * F * 4 / 1e9:.1f} GB)")
    features_mm = np.lib.format.open_memmap(
        features_path, mode='w+', dtype=np.float32, shape=(N, F)
    )

    # Label memmaps will be created on first chunk (need to see the keys)
    label_memmaps: Dict[str, np.memmap] = {}

    t_total = time.perf_counter()

    for chunk_idx in range(num_chunks):
        t_chunk = time.perf_counter()
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, N)
        actual_size = end - start

        # Load and convert chunk (reuses streaming dataset's optimized code)
        dataset._load_chunk(chunk_idx)

        # Write features
        features_mm[start:end] = dataset.current_chunk_features.numpy()

        # Initialize label memmaps on first chunk
        if chunk_idx == 0:
            for key, tensor in dataset.current_chunk_labels.items():
                arr = tensor.numpy()
                shape = (N,) + arr.shape[1:]
                label_path = str(labels_dir / f'{key}.npy')
                label_memmaps[key] = np.lib.format.open_memmap(
                    label_path, mode='w+', dtype=arr.dtype, shape=shape
                )

        # Write labels
        for key, tensor in dataset.current_chunk_labels.items():
            arr = tensor.numpy()
            label_memmaps[key][start:end] = arr[:actual_size]

        # Flush periodically
        if (chunk_idx + 1) % 5 == 0:
            features_mm.flush()
            for mm in label_memmaps.values():
                mm.flush()

        chunk_elapsed = time.perf_counter() - t_chunk
        total_elapsed = time.perf_counter() - t_total
        rate = end / total_elapsed if total_elapsed > 0 else 0
        print(f"  Chunk {chunk_idx + 1}/{num_chunks}: {end:,}/{N:,} samples "
              f"({chunk_elapsed:.1f}s, overall {rate:.0f} samples/s)")

    # Final flush
    features_mm.flush()
    del features_mm
    for mm in label_memmaps.values():
        mm.flush()
    del label_memmaps

    # Save feature names
    with open(output_dir / 'feature_names.json', 'w') as f:
        json.dump(dataset.feature_names, f)

    # Save metadata
    meta = {
        'num_samples': N,
        'num_features': F,
        'target_tf': target_tf,
        'source': str(bin_path),
        'format_version': 1,
    }
    with open(output_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    total_elapsed = time.perf_counter() - t_total
    features_size_gb = N * F * 4 / 1e9

    # Calculate total label size
    label_size = 0
    for npy_file in labels_dir.glob('*.npy'):
        label_size += npy_file.stat().st_size
    label_size_gb = label_size / 1e9

    print(f"\nConversion complete in {total_elapsed:.1f}s")
    print(f"  Features: {features_size_gb:.1f} GB ({features_path})")
    print(f"  Labels: {label_size_gb:.2f} GB ({len(list(labels_dir.glob('*.npy')))} arrays)")
    print(f"  Output: {output_dir}")


class FlatDataset(Dataset):
    """
    Dataset that loads pre-converted .flat directory.

    Features are memory-mapped (instant load, OS handles paging).
    Labels are loaded into RAM (small).
    """

    def __init__(self, flat_dir: str):
        flat_dir = Path(flat_dir)

        # Load metadata
        with open(flat_dir / 'meta.json') as f:
            self.meta = json.load(f)

        with open(flat_dir / 'feature_names.json') as f:
            self.feature_names = json.load(f)

        self.num_samples = self.meta['num_samples']
        self.num_features = self.meta['num_features']

        # Memory-map features (instant, no RAM used until accessed)
        self.features = np.load(
            str(flat_dir / 'features.npy'), mmap_mode='r'
        )

        # Load labels into RAM (small relative to features)
        self.labels: Dict[str, np.ndarray] = {}
        labels_dir = flat_dir / 'labels'
        for npy_file in sorted(labels_dir.glob('*.npy')):
            self.labels[npy_file.stem] = np.load(str(npy_file))

        # Fix 1D validity arrays that should be [N, n_cols] to match their companion data arrays
        for key in list(self.labels.keys()):
            if key.endswith('_valid') and self.labels[key].ndim == 1:
                data_key = key.replace('_valid', '')
                if data_key in self.labels and self.labels[data_key].ndim == 2:
                    n_cols = self.labels[data_key].shape[1]
                    if self.labels[key].shape[0] == self.num_samples * n_cols:
                        self.labels[key] = self.labels[key].reshape(self.num_samples, n_cols)
                        print(f"  Reshaped {key}: [{self.num_samples * n_cols}] -> [{self.num_samples}, {n_cols}]")

        print(f"FlatDataset loaded: {self.num_samples:,} samples, "
              f"{self.num_features:,} features, {len(self.labels)} label arrays")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # .copy() needed because mmap returns read-only view
        features = torch.from_numpy(self.features[idx].copy())
        # torch.tensor handles both numpy scalars and arrays
        labels = {k: torch.tensor(v[idx]) for k, v in self.labels.items()}
        return features, labels

    def get_num_features(self) -> int:
        return self.num_features


def create_flat_dataloaders(
    flat_dir: str,
    batch_size: int = 128,
    val_split: float = 0.2,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
    distributed: bool = False,
    seed: int = 42,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
    """
    Create train and validation dataloaders from a .flat directory.

    Args:
        flat_dir: Path to .flat directory
        batch_size: Batch size for dataloaders
        val_split: Fraction of data for validation
        num_workers: DataLoader workers
        max_samples: Cap total samples used

    Returns:
        Tuple of (train_loader, val_loader, num_features)
    """
    from torch.utils.data import DataLoader, Subset

    dataset = FlatDataset(flat_dir)

    n_samples = len(dataset)
    if max_samples is not None and max_samples < n_samples:
        n_samples = max_samples

    indices = list(range(n_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_size = int(n_samples * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    print(f"Train samples: {len(train_indices):,}")
    print(f"Val samples: {len(val_indices):,}")

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_subset, shuffle=True)
        val_sampler = DistributedSampler(val_subset, shuffle=False)
    else:
        val_sampler = None

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, dataset.get_num_features()
