"""
Hierarchical Dataset for 1-min Data Loading

Optimized lazy loading dataset for training HierarchicalLNN.
Loads 1-min data and dynamically creates training sequences with:
- 200 1-min bars as input
- Target high/low in next 24 bars (prediction horizon)
- Percentage-based targets (not absolute prices)

v4.1: Added native timeframe mode where each CfC layer receives
its timeframe's features at native resolution (5min layer sees 5-min bars, etc.)
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict, Union
from pathlib import Path
import sys
import json

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

import config  # For precision configuration

from src.ml.features import (
    TradingFeatureExtractor,
    TIMEFRAME_SEQUENCE_LENGTHS,
    HIERARCHICAL_TIMEFRAMES
)


class HierarchicalDataset(Dataset):
    """
    Lazy loading dataset for hierarchical training.

    Loads 1-min data on-demand, caches column indices for performance.
    Now includes raw OHLC data and continuation prediction labels.
    """

    def __init__(
        self,
        features_df: pd.DataFrame = None,
        raw_ohlc_df: pd.DataFrame = None,
        continuation_labels_df: pd.DataFrame = None,  # Legacy: single DataFrame
        continuation_labels_dir: str = None,  # v4.3: Directory with per-TF label files
        sequence_length: int = 200,
        prediction_horizon: int = 24,
        mode: str = 'uniform_bars',
        cache_indices: bool = True,
        include_continuation: bool = False,
        mmap_meta_path: str = None,
        profiler=None,
        preload_to_ram: bool = False,
        # v4.1: Native timeframe mode
        use_native_timeframes: bool = False,
        tf_meta_path: str = None,
    ):
        """
        Initialize dataset.

        Args:
            features_df: Features dataframe (165 non-channel + optional mmap 12,474 channel = 12,639 total)
            raw_ohlc_df: Raw OHLC data for input sequences
            continuation_labels_df: [LEGACY] DataFrame with continuation labels (single TF)
            continuation_labels_dir: [v4.3] Directory containing per-TF continuation label files
            sequence_length: Input sequence length (200 1-min bars) - ignored if use_native_timeframes=True
            prediction_horizon: How many bars ahead to predict (24 = 24 minutes)
            mode: 'uniform_bars' (fixed # bars ahead)
            cache_indices: Cache column lookups for speed
            include_continuation: Whether to include continuation prediction targets
            profiler: Optional MemoryProfiler for logging RAM usage
            preload_to_ram: If True, load all mmap data into RAM at startup for fast access
                           (requires ~90GB RAM but avoids 400ms/sample disk I/O)
            use_native_timeframes: If True, return Dict[str, Tensor] where each timeframe
                                   gets its native resolution features (5min layer sees 5-min bars)
            tf_meta_path: Path to tf_meta_*.json file with timeframe sequence metadata
        """
        self._preload_to_ram = preload_to_ram
        self._profiler = profiler
        self.features_df = features_df
        self.raw_ohlc_df = raw_ohlc_df
        self.continuation_labels_df = continuation_labels_df
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.mode = mode
        self.include_continuation = include_continuation

        # v4.1: Native timeframe mode
        self.use_native_timeframes = use_native_timeframes
        self.tf_meta_path = tf_meta_path

        # Diagnostic tracking for timestamp mismatches (always initialize)
        self._missing_label_count = 0
        self._timestamp_deltas = []  # Track time differences in ms
        self._logged_mismatches = 0  # Count of logged mismatches (limit to first 5)

        # Cache which optional continuation fields exist (for consistent dict keys)
        if self.continuation_labels_df is not None:
            self.has_adaptive_horizon = 'adaptive_horizon' in self.continuation_labels_df.columns
            self.has_conf_score = 'conf_score' in self.continuation_labels_df.columns
            self.has_channel_1h_cycles = 'channel_1h_cycles' in self.continuation_labels_df.columns
            self.has_channel_4h_cycles = 'channel_4h_cycles' in self.continuation_labels_df.columns
            self.has_channel_1h_valid = 'channel_1h_valid' in self.continuation_labels_df.columns
            self.has_channel_4h_valid = 'channel_4h_valid' in self.continuation_labels_df.columns
            self.has_channel_1h_r_squared = 'channel_1h_r_squared' in self.continuation_labels_df.columns
            self.has_channel_4h_r_squared = 'channel_4h_r_squared' in self.continuation_labels_df.columns

            # Build lookup dict for O(1) timestamp lookups
            self._build_continuation_lookup()

            # Pre-extract numpy arrays for O(1) access (avoid pandas iloc overhead)
            # This eliminates ~1ms per sample from iloc + torch.tensor() calls
            self._cont_duration = self.continuation_labels_df['duration_hours'].values.astype(config.NUMPY_DTYPE)
            self._cont_gain = self.continuation_labels_df['projected_gain'].values.astype(config.NUMPY_DTYPE)
            self._cont_confidence = self.continuation_labels_df['confidence'].values.astype(config.NUMPY_DTYPE)

            # Optional fields (check existence, extract if present)
            if self.has_adaptive_horizon:
                self._cont_adaptive_horizon = self.continuation_labels_df['adaptive_horizon'].values.astype(config.NUMPY_DTYPE)
            if self.has_conf_score:
                self._cont_conf_score = self.continuation_labels_df['conf_score'].values.astype(config.NUMPY_DTYPE)
            if self.has_channel_1h_cycles:
                self._cont_channel_1h_cycles = self.continuation_labels_df['channel_1h_cycles'].values.astype(config.NUMPY_DTYPE)
            if self.has_channel_4h_cycles:
                self._cont_channel_4h_cycles = self.continuation_labels_df['channel_4h_cycles'].values.astype(config.NUMPY_DTYPE)
            if self.has_channel_1h_valid:
                self._cont_channel_1h_valid = self.continuation_labels_df['channel_1h_valid'].values.astype(config.NUMPY_DTYPE)
            if self.has_channel_4h_valid:
                self._cont_channel_4h_valid = self.continuation_labels_df['channel_4h_valid'].values.astype(config.NUMPY_DTYPE)
            if self.has_channel_1h_r_squared:
                self._cont_channel_1h_r_squared = self.continuation_labels_df['channel_1h_r_squared'].values.astype(config.NUMPY_DTYPE)
            if self.has_channel_4h_r_squared:
                self._cont_channel_4h_r_squared = self.continuation_labels_df['channel_4h_r_squared'].values.astype(config.NUMPY_DTYPE)
        else:
            self.has_adaptive_horizon = False
            self.has_conf_score = False
            self.has_channel_1h_cycles = False
            self.has_channel_4h_cycles = False
            self.has_channel_1h_valid = False
            self.has_channel_4h_valid = False
            self.has_channel_1h_r_squared = False
            self.has_channel_4h_r_squared = False
            self._continuation_ts_to_idx = {}

        # v4.3: Load per-TF hierarchical continuation labels
        self.continuation_labels_dir = continuation_labels_dir
        self._per_tf_continuation = {}  # Dict[tf, Dict] with arrays for each TF
        self._per_tf_ts_to_idx = {}  # Dict[tf, Dict[int64_ns, row_idx]]

        if continuation_labels_dir is not None:
            self._load_hierarchical_continuation_labels(continuation_labels_dir)

        # Dtype validation - ensure data matches config precision
        expected_dtype = config.NUMPY_DTYPE

        # v4.1: Native timeframe mode - load per-TF sequences
        if self.use_native_timeframes and tf_meta_path is not None:
            self._init_native_timeframe_mode(tf_meta_path, raw_ohlc_df, expected_dtype)
            return  # Skip legacy loading paths

        # Memory-mapped shard loading (zero RAM spike!)
        if mmap_meta_path is not None:
            import json
            import numpy as np
            from pathlib import Path

            print(f"  📂 Loading memory-mapped channel shards...")
            meta = json.load(open(mmap_meta_path))
            cache_base_dir = Path(mmap_meta_path).parent  # Base for resolving relative paths

            # Helper to resolve paths (handles both relative and legacy absolute paths)
            def resolve_shard_path(p):
                path = Path(p)
                if path.is_absolute():
                    return path  # Legacy absolute path
                return cache_base_dir / path  # New relative path

            # Load channel feature shards as memory-maps
            self.channel_mmaps = []
            self.channel_cumulative_rows = [0]
            self.premerged_channel_mmaps = None  # Optional: merged main + monthly for fast slicing

            for info in meta['chunk_info']:
                shard_path = resolve_shard_path(info['path'])
                mmap_array = np.load(str(shard_path), mmap_mode='r')
                self.channel_mmaps.append(mmap_array)
                self.channel_cumulative_rows.append(self.channel_cumulative_rows[-1] + info['rows'])

            # Log after loading channel mmaps
            if self._profiler:
                total_mmap_bytes = sum(m.nbytes for m in self.channel_mmaps)
                self._profiler.log_info(f"CHANNEL_MMAPS_LOADED | shards={len(self.channel_mmaps)} | total_rows={self.channel_cumulative_rows[-1]:,} | virtual_size_gb={total_mmap_bytes/1e9:.2f}")
                self._profiler.snapshot("post_channel_mmaps_load", 0, force_log=True)

            # Load timestamps from shards
            all_timestamps = []
            for info in meta['chunk_info']:
                index_path = resolve_shard_path(info['index_path'])
                idx_array = np.load(str(index_path), mmap_mode='r')
                all_timestamps.append(idx_array)
            self.timestamps = np.concatenate(all_timestamps)

            # Pre-convert timestamps to int64 nanoseconds for fast lookup
            # Avoids pd.Timestamp() creation per sample (~50µs savings each)
            self._timestamps_ns = self.timestamps.astype('datetime64[ns]').astype(np.int64)

            # v3.19: Load monthly/3month separate shard if present
            # NOTE: Monthly shard may have more rows than chunks (includes 2015 data)
            # We store _monthly_offset to apply when accessing
            self.monthly_3month_mmap = None
            self._monthly_offset = 0
            if 'monthly_3month_shard' in meta and meta['monthly_3month_shard'] is not None:
                monthly_shard_info = meta['monthly_3month_shard']
                monthly_path = resolve_shard_path(monthly_shard_info['path'])

                if monthly_path.exists():
                    self.monthly_3month_mmap = np.load(str(monthly_path), mmap_mode='r')
                    monthly_rows = self.monthly_3month_mmap.shape[0]
                    chunk_rows = self.channel_cumulative_rows[-1]

                    # Check if monthly shard needs alignment
                    if monthly_rows > chunk_rows:
                        self._monthly_offset = monthly_rows - chunk_rows
                        print(f"     ✓ Loaded monthly/3month shard: {monthly_rows:,} rows × {self.monthly_3month_mmap.shape[1]} cols")
                        print(f"        ⚠️  Monthly shard offset: {self._monthly_offset:,} (will slice to align with chunks)")
                    else:
                        print(f"     ✓ Loaded monthly/3month shard: {monthly_rows:,} rows × {self.monthly_3month_mmap.shape[1]} cols")

                    # Log after loading monthly shard
                    if self._profiler:
                        monthly_size_gb = self.monthly_3month_mmap.nbytes / 1e9
                        self._profiler.log_info(f"MONTHLY_SHARD_LOADED | rows={monthly_rows:,} | cols={self.monthly_3month_mmap.shape[1]} | virtual_size_gb={monthly_size_gb:.2f} | offset={self._monthly_offset}")
                        self._profiler.snapshot("post_monthly_shard_load", 0, force_log=True)
                else:
                    print(f"     ⚠️  Monthly/3month shard not found: {monthly_path}")
                    print(f"        Expected: {monthly_shard_info['cols']} features, will use zeros")
            else:
                self.premerged_channel_mmaps = None

            # Load non-channel features normally (these are small - ~165 base features)
            if features_df is not None:
                # CRITICAL: Align non-channel features with mmap chunks!
                # Chunks may start from a later date (e.g., 2016) while non-channel
                # starts earlier (e.g., 2015). We need to slice to match.
                chunk_total_rows = self.channel_cumulative_rows[-1]
                nc_total_rows = len(features_df)

                if nc_total_rows > chunk_total_rows:
                    # Non-channel has more rows (starts earlier) - slice to align
                    # The offset is at the START (chunks skip early dates)
                    offset = nc_total_rows - chunk_total_rows
                    print(f"     ⚠️  Index alignment: non-channel has {nc_total_rows:,} rows, chunks have {chunk_total_rows:,}")
                    print(f"        Slicing non-channel from index {offset:,} to align with chunks")

                    # Validate alignment by checking timestamps
                    chunk_first_ts = pd.Timestamp(self.timestamps[0])
                    nc_timestamps = features_df.index

                    # Find where non-channel matches first chunk timestamp
                    nc_aligned_idx = nc_timestamps.get_indexer([chunk_first_ts], method='nearest')[0]
                    if nc_aligned_idx != offset:
                        print(f"        ℹ️  Timestamp-based offset: {nc_aligned_idx}, row-count offset: {offset}")
                        # Use timestamp-based alignment (more accurate)
                        offset = nc_aligned_idx

                    # Slice features_df to match chunks
                    features_df = features_df.iloc[offset:]
                    print(f"        Aligned non-channel: {len(features_df):,} rows (matches chunks: {chunk_total_rows:,})")

                    # Store offset for debugging
                    self._nc_offset = offset
                elif nc_total_rows < chunk_total_rows:
                    raise ValueError(
                        f"Non-channel features ({nc_total_rows:,} rows) has fewer rows than "
                        f"mmap chunks ({chunk_total_rows:,} rows). Data is inconsistent!"
                    )
                else:
                    self._nc_offset = 0

                # Optimize: check dtype before converting
                temp_array = features_df.values
                if temp_array.dtype != config.NUMPY_DTYPE:
                    self.non_channel_array = temp_array.astype(config.NUMPY_DTYPE)
                else:
                    self.non_channel_array = temp_array
                # Log after loading non-channel array
                if self._profiler:
                    nc_size_mb = self.non_channel_array.nbytes / 1e6
                    self._profiler.log_info(f"NON_CHANNEL_LOADED | rows={self.non_channel_array.shape[0]:,} | cols={self.non_channel_array.shape[1]} | size_mb={nc_size_mb:.1f}")
                    self._profiler.snapshot("post_non_channel_load", 0, force_log=True)
            else:
                self.non_channel_array = None
                self._nc_offset = 0

            self.using_mmaps = True
            self.num_channel_features = meta['num_features']
            print(f"     ✓ Loaded {len(self.channel_mmaps)} channel shards ({self.num_channel_features:,} features)")
            print(f"     ✓ Loaded {self.non_channel_array.shape[1] if self.non_channel_array is not None else 0} non-channel features")
            print(f"     ✓ Total rows: {meta['total_rows']:,}")

            # Initialize preloaded arrays as None
            self.preloaded_main = None
            self.preloaded_monthly = None

            if self._preload_to_ram:
                # Preload all mmap data into RAM for fast access (avoids 400ms/sample disk I/O)
                print(f"  📦 Preloading all channel data to RAM...")
                import time
                preload_start = time.perf_counter()

                total_rows = self.channel_cumulative_rows[-1]
                main_cols = self.channel_mmaps[0].shape[1]

                # Allocate single contiguous array for main channel features
                self.preloaded_main = np.empty((total_rows, main_cols), dtype=config.NUMPY_DTYPE)

                # Copy from mmap shards
                for i, mmap_arr in enumerate(self.channel_mmaps):
                    start = self.channel_cumulative_rows[i]
                    end = self.channel_cumulative_rows[i + 1]
                    self.preloaded_main[start:end] = mmap_arr[:]
                    if self._profiler:
                        self._profiler.snapshot(f"preload_shard_{i}", 0, force_log=True)

                # Copy monthly/3month shard if present (apply offset to align with chunks)
                if self.monthly_3month_mmap is not None:
                    monthly_offset = getattr(self, '_monthly_offset', 0)
                    if monthly_offset > 0:
                        # Slice to align with chunks (skip early rows)
                        self.preloaded_monthly = np.array(self.monthly_3month_mmap[monthly_offset:])
                        print(f"        Sliced monthly shard from {monthly_offset:,} for alignment")
                    else:
                        self.preloaded_monthly = np.array(self.monthly_3month_mmap)

                preload_time = time.perf_counter() - preload_start

                # Log memory usage
                main_gb = self.preloaded_main.nbytes / 1e9
                monthly_gb = self.preloaded_monthly.nbytes / 1e9 if self.preloaded_monthly is not None else 0
                total_gb = main_gb + monthly_gb

                print(f"     ✓ Loaded {total_rows:,} rows into RAM ({total_gb:.1f} GB in {preload_time:.1f}s)")
                if self._profiler:
                    self._profiler.log_info(f"PRELOAD_COMPLETE | rows={total_rows:,} | main_gb={main_gb:.2f} | monthly_gb={monthly_gb:.2f} | time_sec={preload_time:.1f}")
                    self._profiler.snapshot("post_preload_complete", 0, force_log=True)

                # Clear mmap references to save virtual memory (optional, mmaps are lightweight)
                # Keep them around as backup in case we need them
                # self.channel_mmaps = None
                # self.monthly_3month_mmap = None
            else:
                # mmap-only mode: rely on OS page cache for memory efficiency
                print(f"     ✓ Using mmap-only mode (no pre-merge, OS page cache manages hot data)")

            self.premerged_channel_mmaps = None  # Legacy field
            self.premerged_shard_indices = set()  # Legacy field

        else:
            # Normal path - load everything into RAM
            self.using_mmaps = False
            self.features_array = features_df.values
            self.timestamps = features_df.index.values

            # Pre-convert timestamps to int64 nanoseconds for fast lookup
            self._timestamps_ns = self.timestamps.astype('datetime64[ns]').astype(np.int64)

            # Validate and convert feature array dtype if needed
            if self.features_array.dtype != expected_dtype:
                print(f"  ⚠️  Feature dtype mismatch: {self.features_array.dtype} != {expected_dtype}")
                print(f"     Converting to {expected_dtype} (may use extra memory temporarily)")
                self.features_array = self.features_array.astype(expected_dtype)

        if raw_ohlc_df is not None:
            # CRITICAL: Apply same offset as non_channel to keep alignment
            # Without this, raw_ohlc indices don't match chunk/non_channel indices
            if hasattr(self, '_nc_offset') and self._nc_offset > 0:
                raw_ohlc_df = raw_ohlc_df.iloc[self._nc_offset:]
                print(f"     ✓ Aligned raw_ohlc: sliced {self._nc_offset:,} rows to match chunks")
            self.raw_ohlc_array = raw_ohlc_df[['tsla_open', 'tsla_high', 'tsla_low', 'tsla_close']].values
            # Optimize: ensure OHLC matches dtype (already checks before converting)
            if self.raw_ohlc_array.dtype != expected_dtype:
                self.raw_ohlc_array = self.raw_ohlc_array.astype(expected_dtype)
            # Free the DataFrame - we only need the extracted array
            # This prevents COW memory multiplication when DataLoader forks workers
            self.raw_ohlc_df = None
        else:
            self.raw_ohlc_array = None

        # Free features_df - we've extracted all needed arrays
        # This prevents COW memory multiplication when DataLoader forks workers
        self.features_df = None

        # Cached torch scalars to avoid per-sample tensor creation
        self._torch_dtype = config.get_torch_dtype()
        self._const_zero = torch.tensor(0.0, dtype=self._torch_dtype)
        self._const_half = torch.tensor(0.5, dtype=self._torch_dtype)
        self._const_one = torch.tensor(1.0, dtype=self._torch_dtype)
        self._const_default_horizon = torch.tensor(24.0, dtype=self._torch_dtype)

        # Cache column indices (CRITICAL for performance)
        if cache_indices:
            if self.using_mmaps:
                # For mmap mode, tsla_close is in non_channel_array
                if self.non_channel_array is not None:
                    self.feature_names = features_df.columns.tolist()
                    self.close_idx = self.feature_names.index('tsla_close')
                    # v3.17: high_idx/low_idx unused (we use raw_ohlc_array for actual extremes)
                    self.high_idx = None  # Deprecated - not used
                    self.low_idx = None   # Deprecated - not used
                else:
                    self.close_idx = 0  # Fallback
            else:
                self.feature_names = features_df.columns.tolist()
                self.close_idx = self.feature_names.index('tsla_close')
                # v3.17: high_idx/low_idx unused (we use raw_ohlc_array for actual extremes)
                self.high_idx = None  # Deprecated - not used
                self.low_idx = None   # Deprecated - not used
        else:
            self.close_idx = None

        # Calculate valid indices
        # Need: sequence_length bars for input + prediction_horizon bars for target
        self.min_context = sequence_length
        self.total_required = sequence_length + prediction_horizon

        # Get total length (from mmap or features_array)
        if self.using_mmaps:
            total_len = self.channel_cumulative_rows[-1]  # Total rows across all shards
        else:
            total_len = len(self.features_array)

        # Valid start indices
        self.valid_indices = list(range(
            self.min_context,
            total_len - prediction_horizon
        ))

        if len(self.valid_indices) == 0:
            raise ValueError(
                f"Not enough data. Need at least {self.total_required} bars, "
                f"but have {total_len}"
            )

    def _init_native_timeframe_mode(
        self,
        tf_meta_path: str,
        raw_ohlc_df: pd.DataFrame,
        expected_dtype
    ):
        """
        Initialize native timeframe mode - each layer gets features at its native resolution.

        Args:
            tf_meta_path: Path to tf_meta_*.json metadata file
            raw_ohlc_df: Raw OHLC DataFrame for target calculation
            expected_dtype: Expected numpy dtype for features
        """
        print(f"\n  📂 Loading native timeframe sequences...")

        # Load metadata
        with open(tf_meta_path) as f:
            self.tf_meta = json.load(f)

        cache_dir = Path(tf_meta_path).parent
        cache_key = self.tf_meta['cache_key']

        # Load per-timeframe arrays and timestamps
        self.tf_mmaps = {}
        self.tf_timestamps = {}  # For index conversion
        self.tf_columns = self.tf_meta['timeframe_columns']
        self.tf_sequence_lengths = self.tf_meta['sequence_lengths']
        self.timeframe_feature_counts = {}  # For model input_sizes

        for tf in HIERARCHICAL_TIMEFRAMES:
            # Load feature array
            mmap_path = cache_dir / f"tf_sequence_{tf}_{cache_key}.npy"
            if mmap_path.exists():
                self.tf_mmaps[tf] = np.load(str(mmap_path), mmap_mode='r')
                self.timeframe_feature_counts[tf] = self.tf_mmaps[tf].shape[1]

                # Load timestamps
                ts_path = cache_dir / f"tf_timestamps_{tf}_{cache_key}.npy"
                if ts_path.exists():
                    self.tf_timestamps[tf] = np.load(str(ts_path), mmap_mode='r')

                shape = self.tf_mmaps[tf].shape
                seq_len = self.tf_sequence_lengths[tf]
                print(f"     {tf}: {shape[0]:,} bars × {shape[1]} features (seq_len={seq_len})")
            else:
                raise FileNotFoundError(f"Missing timeframe sequence file: {mmap_path}")

        # Store raw OHLC for target calculation
        if raw_ohlc_df is not None:
            self.raw_ohlc_array = raw_ohlc_df[['tsla_open', 'tsla_high', 'tsla_low', 'tsla_close']].values
            if self.raw_ohlc_array.dtype != expected_dtype:
                self.raw_ohlc_array = self.raw_ohlc_array.astype(expected_dtype)
            # Store timestamps for Priority 1b: timestamp-based index conversion
            self.raw_ohlc_timestamps = raw_ohlc_df.index.values.astype('datetime64[ns]').astype('int64')
        else:
            self.raw_ohlc_array = None
            self.raw_ohlc_timestamps = np.array([], dtype='int64')

        # Use 1-min timestamps from 5min array (first timeframe, most granular)
        # Actually we need 1-min timestamps for index conversion
        # The total_rows_1min tells us how many 1-min bars there are
        self.total_1min_rows = self.tf_meta['total_rows_1min']

        # Calculate valid indices based on:
        # 1. Minimum context required by longest timeframe sequence
        # 2. Future bars needed for target calculation
        max_seq_len = max(self.tf_sequence_lengths.values())
        # Convert to 1-min index: need enough 1-min bars that all timeframes have their seq_len
        # For 5min with seq_len=200, we need 200 * 5 = 1000 1-min bars
        # For 3month with seq_len=8, we need ~8 * 90 days * 6.5 hours * 60 min ≈ 280,800 1-min bars
        # But this is handled by the resampled arrays themselves
        # FIX (Priority 2): Handle missing/insufficient timeframes gracefully
        available_tfs = [tf for tf in HIERARCHICAL_TIMEFRAMES if tf in self.tf_mmaps]
        if not available_tfs:
            raise ValueError("No valid timeframe sequences found!")

        missing_tfs = [tf for tf in HIERARCHICAL_TIMEFRAMES if tf not in self.tf_mmaps]
        if missing_tfs:
            print(f"     ⚠️  Warning: Missing timeframes (insufficient data): {', '.join(missing_tfs)}")
            print(f"     📊 Will use available timeframes: {', '.join(available_tfs)}")

        # Valid indices are based on the minimum of available timeframe array lengths
        min_tf_rows = min(self.tf_mmaps[tf].shape[0] for tf in available_tfs)

        # We use 5min bars as the reference for indexing (since it's our finest resampled resolution)
        # Each sample corresponds to a 5min bar index
        self.min_context = max(self.tf_sequence_lengths.values())
        self.total_required = self.min_context + (self.prediction_horizon // 5 + 1)  # Convert to 5-min bars

        # Valid indices are positions in the 5min array
        self.valid_indices = list(range(
            self.min_context,
            self.tf_mmaps['5min'].shape[0] - (self.prediction_horizon // 5 + 1)
        ))

        print(f"     ✓ Loaded {len(HIERARCHICAL_TIMEFRAMES)} timeframe sequences")
        print(f"     ✓ Valid samples: {len(self.valid_indices):,}")

        # Set flags
        self.using_mmaps = False  # Not using legacy mmap mode
        self.using_native_timeframes = True

        # Cache torch dtype
        self._torch_dtype = config.get_torch_dtype()

        # We need to find tsla_close index in the 5min features for current price lookup
        if 'tsla_close' in self.tf_columns['5min']:
            self.close_idx_5min = self.tf_columns['5min'].index('tsla_close')
        else:
            # Search for it
            for i, col in enumerate(self.tf_columns['5min']):
                if 'tsla_close' in col:
                    self.close_idx_5min = i
                    break
            else:
                raise ValueError("Cannot find tsla_close in 5min features")

    def _calculate_targets_from_future(self, current_price: float, future_prices: np.ndarray,
                                       seq_start: int, seq_end: int,
                                       past_prices: np.ndarray = None) -> dict:
        """
        Calculate multi-task targets from future price data.

        This method implements the full target calculation logic used in legacy mode.
        It's shared between __getitem__ (legacy) and _getitem_native_timeframe (native).

        Args:
            current_price: Current close price
            future_prices: Array of future prices
            seq_start: Start index of historical sequence (for breakout detection)
            seq_end: End index of historical sequence / current index
            past_prices: Optional past prices for channel-based breakout detection

        Returns:
            Dictionary with all target labels
        """
        # Defensive check
        if current_price <= 0 or np.isnan(current_price) or np.isinf(current_price):
            raise ValueError(f"Invalid current_price: {current_price}")

        # Get high/low from future prices
        future_high_actual = np.max(future_prices)
        future_low_actual = np.min(future_prices)

        # Convert to percentage change
        target_high_pct = (future_high_actual - current_price) / current_price * 100.0
        target_low_pct = (future_low_actual - current_price) / current_price * 100.0

        # Label 1: Hit Band
        ideal_band_high = future_high_actual * 1.02
        ideal_band_low = future_low_actual * 0.98
        prices_in_ideal_band = (future_prices >= ideal_band_low) & (future_prices <= ideal_band_high)
        hit_band_label = float(prices_in_ideal_band.sum() / len(prices_in_ideal_band) > 0.8)

        # Label 2: Hit Target Before Stop
        target_price = future_high_actual
        stop_price = current_price * (1 + target_low_pct/100 - 0.02)
        hit_target_label = float(self._check_target_sequence(
            future_prices, current_price, target_price, stop_price
        ))

        # Label 3: Expected Return
        expected_return_label = self._simulate_trade_execution(
            future_prices, current_price, target_price, stop_price
        )

        # Label 4: Overshoot
        band_range = abs(target_high_pct - target_low_pct)
        if band_range > 0:
            overshoot_high = max(0, future_high_actual - ideal_band_high) / current_price * 100
            overshoot_low = max(0, ideal_band_low - future_low_actual) / current_price * 100
            overshoot_label = (overshoot_high + overshoot_low) / band_range
        else:
            overshoot_label = 0.0

        # Adaptive targets
        actual_max_idx = future_prices.argmax()
        bars_to_peak = actual_max_idx
        adaptive_price_change = target_high_pct if target_high_pct > abs(target_low_pct) else target_low_pct
        adaptive_horizon_log = np.log(bars_to_peak / 24 + 1e-6)
        adaptive_confidence = 1.0 if bars_to_peak > 48 else 0.5

        targets = {
            'high': target_high_pct,
            'low': target_low_pct,
            'hit_band': hit_band_label,
            'hit_target': hit_target_label,
            'expected_return': expected_return_label,
            'overshoot': overshoot_label,
            'continuation_duration': 0.0,   # Placeholder (requires continuation_labels_df)
            'continuation_gain': 0.0,       # Placeholder
            'continuation_confidence': 0.5, # Placeholder
            'price_change_pct': adaptive_price_change,
            'horizon_bars_log': adaptive_horizon_log,
            'adaptive_confidence': adaptive_confidence,
        }

        # Breakout labels
        try:
            if past_prices is not None:
                breakout_labels = self._detect_channel_breakout(
                    past_prices=past_prices,
                    future_prices=future_prices,
                    current_price=current_price,
                    lookback=60,
                    channel_std=2.0,
                    breakout_threshold=1.0
                )
                targets['breakout_occurred'] = breakout_labels['breakout_occurred']
                targets['breakout_direction'] = breakout_labels['breakout_direction']
                targets['breakout_bars_log'] = breakout_labels['breakout_bars_log']
                targets['breakout_magnitude'] = breakout_labels['breakout_magnitude']
            else:
                # Defaults if no past prices available
                targets['breakout_occurred'] = 0.0
                targets['breakout_direction'] = 0.5
                targets['breakout_bars_log'] = np.log(self.prediction_horizon + 1)
                targets['breakout_magnitude'] = 0.0
        except Exception:
            # Fallback if breakout detection fails
            targets['breakout_occurred'] = 0.0
            targets['breakout_direction'] = 0.5
            targets['breakout_bars_log'] = np.log(self.prediction_horizon + 1)
            targets['breakout_magnitude'] = 0.0

        return targets

    def _getitem_native_timeframe(self, idx: int) -> Tuple[Dict[str, np.ndarray], dict]:
        """
        Get sample in native timeframe mode - returns Dict[str, np.ndarray].

        Args:
            idx: Sample index

        Returns:
            timeframe_data: Dict mapping timeframe -> features [seq_len, features]
            targets: Dict with prediction targets
        """
        # Get 5min array index (our reference timeframe)
        data_idx_5min = self.valid_indices[idx]

        # Get the 5min timestamp at this index for cross-timeframe alignment
        ts_5min = self.tf_timestamps['5min'][data_idx_5min]

        # Build features dict for each timeframe
        timeframe_data = {}

        for tf in HIERARCHICAL_TIMEFRAMES:
            # FIX (Priority 2): Skip missing/unavailable timeframes
            if tf not in self.tf_mmaps:
                continue

            seq_len = self.tf_sequence_lengths[tf]

            # Find index in this timeframe's array that corresponds to our 5min timestamp
            # Binary search for the closest timestamp <= ts_5min
            tf_timestamps = self.tf_timestamps[tf]
            tf_idx = np.searchsorted(tf_timestamps, ts_5min, side='right') - 1
            tf_idx = max(seq_len, tf_idx)  # Ensure we have enough history

            # Extract sequence
            start = tf_idx - seq_len
            end = tf_idx

            tf_features = self.tf_mmaps[tf][start:end, :]
            timeframe_data[tf] = np.ascontiguousarray(tf_features)

        # Get current price from 5min features
        current_price = self.tf_mmaps['5min'][data_idx_5min - 1, self.close_idx_5min]

        # Get future prices for target calculation
        horizon_5min = self.prediction_horizon // 5 + 1

        if self.raw_ohlc_array is not None:
            # Use raw 1-min OHLC for more accurate targets
            # FIX (Priority 1b): Use timestamp-based lookup instead of approximation
            # Get the 5min timestamp at current data_idx_5min
            if '5min' in self.tf_timestamps and len(self.tf_timestamps['5min']) > data_idx_5min:
                ts_5min = self.tf_timestamps['5min'][data_idx_5min]

                # Use binary search to find corresponding 1-min index by timestamp
                # This handles market gaps (weekends, holidays) correctly
                if hasattr(self, 'raw_ohlc_timestamps') and len(self.raw_ohlc_timestamps) > 0:
                    approx_1min_idx = np.searchsorted(self.raw_ohlc_timestamps, ts_5min, side='right') - 1
                    approx_1min_idx = max(0, min(approx_1min_idx, len(self.raw_ohlc_array) - 1))
                else:
                    # Fallback if timestamps not available: use approximation
                    approx_1min_idx = data_idx_5min * 5
            else:
                # Fallback if 5min timestamps not available
                approx_1min_idx = data_idx_5min * 5

            future_start = approx_1min_idx
            future_end = min(approx_1min_idx + self.prediction_horizon, len(self.raw_ohlc_array))

            if future_end > future_start:
                future_ohlc = self.raw_ohlc_array[future_start:future_end]
                future_high = np.max(future_ohlc[:, 1])
                future_low = np.min(future_ohlc[:, 2])
                future_prices = future_ohlc[:, 3]
            else:
                future_high = current_price
                future_low = current_price
                future_prices = np.array([current_price])
        else:
            # Fallback: Use 5min closes
            future_5min_end = min(data_idx_5min + horizon_5min, len(self.tf_mmaps['5min']))
            future_5min = self.tf_mmaps['5min'][data_idx_5min:future_5min_end, self.close_idx_5min]
            future_high = np.max(future_5min)
            future_low = np.min(future_5min)
            future_prices = future_5min

        # Get past prices for breakout detection
        past_5min_start = max(0, data_idx_5min - 200)
        past_prices = self.tf_mmaps['5min'][past_5min_start:data_idx_5min, self.close_idx_5min]

        # FIX (Priority 1): Call new helper method to calculate proper targets
        targets = self._calculate_targets_from_future(
            current_price=current_price,
            future_prices=future_prices,
            seq_start=past_5min_start,
            seq_end=data_idx_5min,
            past_prices=past_prices
        )

        # v4.3: Add per-TF hierarchical continuation prediction targets
        # Each timeframe gets its own duration/gain/confidence predictions
        if self.include_continuation and len(self._per_tf_continuation) > 0:
            for tf in HIERARCHICAL_TIMEFRAMES:
                if tf not in self._per_tf_continuation or tf not in self._per_tf_ts_to_idx:
                    # No labels for this TF - use placeholders
                    targets[f'cont_{tf}_duration'] = 0.0
                    targets[f'cont_{tf}_gain'] = 0.0
                    targets[f'cont_{tf}_confidence'] = 0.5
                    continue

                try:
                    # Get timestamp for this timeframe
                    if tf in self.tf_timestamps:
                        # Find the corresponding index in this TF's array
                        tf_timestamps = self.tf_timestamps[tf]
                        tf_idx = np.searchsorted(tf_timestamps, ts_5min, side='right') - 1
                        tf_idx = max(0, min(tf_idx, len(tf_timestamps) - 1))
                        tf_ts = tf_timestamps[tf_idx]

                        # Lookup in per-TF dict
                        row_idx = self._per_tf_ts_to_idx[tf].get(int(tf_ts))

                        if row_idx is not None:
                            cont_data = self._per_tf_continuation[tf]
                            targets[f'cont_{tf}_duration'] = float(cont_data['duration_bars'][row_idx])
                            targets[f'cont_{tf}_gain'] = float(cont_data['max_gain_pct'][row_idx])
                            targets[f'cont_{tf}_confidence'] = float(cont_data['confidence'][row_idx])
                        else:
                            # Timestamp not found - use placeholders
                            targets[f'cont_{tf}_duration'] = 0.0
                            targets[f'cont_{tf}_gain'] = 0.0
                            targets[f'cont_{tf}_confidence'] = 0.5
                    else:
                        # TF timestamps not available
                        targets[f'cont_{tf}_duration'] = 0.0
                        targets[f'cont_{tf}_gain'] = 0.0
                        targets[f'cont_{tf}_confidence'] = 0.5
                except Exception:
                    # Error - use placeholders
                    targets[f'cont_{tf}_duration'] = 0.0
                    targets[f'cont_{tf}_gain'] = 0.0
                    targets[f'cont_{tf}_confidence'] = 0.5

        # Legacy fallback: single continuation_labels_df (backward compatibility)
        elif self.include_continuation and self.continuation_labels_df is not None:
            try:
                ts_5min = self.tf_timestamps['5min'][data_idx_5min]
                ts_aligned_ns = int(ts_5min) + (4 * 60 * 1_000_000_000)
                row_idx = self._continuation_ts_to_idx.get(ts_aligned_ns)
                if row_idx is None:
                    row_idx = self._continuation_ts_to_idx.get(int(ts_5min))

                if row_idx is not None:
                    targets['continuation_duration'] = float(self._cont_duration[row_idx])
                    targets['continuation_gain'] = float(self._cont_gain[row_idx])
                    targets['continuation_confidence'] = float(self._cont_confidence[row_idx])
            except Exception:
                pass

        return timeframe_data, targets

    def __len__(self) -> int:
        """Return number of valid sequences."""
        return len(self.valid_indices)

    def _build_continuation_lookup(self):
        """Build dict for O(1) timestamp lookups into continuation labels.

        This method can be called after __init__ if continuation_labels_df is
        assigned post-creation (e.g., in the mmap path of create_hierarchical_dataset).
        """
        if self.continuation_labels_df is None:
            self._continuation_ts_to_idx = {}
            return

        if 'timestamp' not in self.continuation_labels_df.columns:
            self._continuation_ts_to_idx = {}
            return

        # Build dict mapping int64 nanoseconds → row index
        # This avoids pandas type coercion issues between numpy.datetime64 and pd.Timestamp
        self._continuation_ts_to_idx = {}
        for i, ts in enumerate(self.continuation_labels_df['timestamp'].values):
            ts_ns = pd.Timestamp(ts).value
            self._continuation_ts_to_idx[ts_ns] = i

        # Debug output
        if len(self._continuation_ts_to_idx) > 0:
            sample_keys = list(self._continuation_ts_to_idx.keys())[:3]
            sample_ts = [pd.Timestamp(k) for k in sample_keys]
            print(f"     ℹ️  Continuation lookup dict: {len(self._continuation_ts_to_idx):,} entries")
            print(f"        Sample timestamps: {sample_ts[0]}, {sample_ts[1]}, {sample_ts[2]}")

    def _load_hierarchical_continuation_labels(self, labels_dir: str):
        """
        Load per-timeframe hierarchical continuation labels from directory.

        v4.3: Each timeframe has its own continuation labels file with:
        - timestamp: Bar timestamp at native TF resolution
        - duration_bars: How many bars until channel break (at native resolution)
        - max_gain_pct: Maximum favorable price move before break
        - confidence: Channel quality score (0-1)

        Args:
            labels_dir: Directory containing per-TF label files
                       (e.g., continuation_labels_5min_*.pkl, continuation_labels_1h_*.pkl)
        """
        import pickle
        from pathlib import Path

        labels_path = Path(labels_dir)
        if not labels_path.exists():
            print(f"     ⚠️  Continuation labels directory not found: {labels_dir}")
            return

        print(f"\n  📂 Loading hierarchical continuation labels from {labels_dir}...")

        loaded_count = 0
        for tf in HIERARCHICAL_TIMEFRAMES:
            # Look for label files matching pattern
            pattern = f"continuation_labels_{tf}_*.pkl"
            matching_files = list(labels_path.glob(pattern))

            if not matching_files:
                # Try without cache suffix
                pattern_simple = f"continuation_labels_{tf}.pkl"
                simple_file = labels_path / pattern_simple
                if simple_file.exists():
                    matching_files = [simple_file]

            if not matching_files:
                continue

            # Use most recent file if multiple exist
            label_file = sorted(matching_files)[-1]

            try:
                with open(label_file, 'rb') as f:
                    labels_df = pickle.load(f)

                if isinstance(labels_df, pd.DataFrame) and len(labels_df) > 0:
                    # Extract arrays for O(1) access
                    self._per_tf_continuation[tf] = {
                        'duration_bars': labels_df['duration_bars'].values.astype(config.NUMPY_DTYPE),
                        'max_gain_pct': labels_df['max_gain_pct'].values.astype(config.NUMPY_DTYPE),
                        'confidence': labels_df['confidence'].values.astype(config.NUMPY_DTYPE),
                    }

                    # Build timestamp -> row_idx lookup dict
                    # Note: timestamp is the INDEX, not a column
                    self._per_tf_ts_to_idx[tf] = {}
                    for i, ts in enumerate(labels_df.index):
                        ts_ns = pd.Timestamp(ts).value
                        self._per_tf_ts_to_idx[tf][ts_ns] = i

                    loaded_count += 1
                    print(f"     {tf}: {len(labels_df):,} labels loaded from {label_file.name}")

            except Exception as e:
                print(f"     ⚠️  Failed to load {tf} labels: {e}")

        if loaded_count > 0:
            print(f"     ✓ Loaded continuation labels for {loaded_count}/{len(HIERARCHICAL_TIMEFRAMES)} timeframes")
        else:
            print(f"     ⚠️  No continuation label files found in {labels_dir}")

    def get_label_mismatch_summary(self) -> dict:
        """
        Get diagnostic summary of timestamp mismatches in continuation labels.

        Returns:
            dict with:
            - missing_count: Number of samples missing continuation labels
            - missing_pct: Percentage of samples missing labels
            - avg_delta_ms: Average time delta to closest label (if found)
            - max_delta_ms: Maximum time delta to closest label
            - diagnosis: String interpretation of the mismatch pattern
        """
        if not self.include_continuation or self.continuation_labels_df is None:
            return {'status': 'No continuation labels loaded'}

        total_samples = len(self.valid_indices)
        missing_pct = (self._missing_label_count / total_samples * 100) if total_samples > 0 else 0

        result = {
            'missing_count': self._missing_label_count,
            'missing_pct': missing_pct,
            'total_samples': total_samples,
        }

        if self._timestamp_deltas:
            result['avg_delta_ms'] = sum(self._timestamp_deltas) / len(self._timestamp_deltas)
            result['max_delta_ms'] = max(self._timestamp_deltas)
            result['min_delta_ms'] = min(self._timestamp_deltas)

            # Diagnosis
            avg = result['avg_delta_ms']
            if avg < 1.0:
                result['diagnosis'] = f"Precision issue: {avg:.3f}ms avg delta - timestamps off by <1ms (need fuzzy matching with ±{int(avg*2)}ms tolerance)"
            elif avg < 100:
                result['diagnosis'] = f"Minor misalignment: {avg:.1f}ms avg delta - may need fuzzy matching with ±{int(avg*2)}ms tolerance"
            else:
                result['diagnosis'] = f"Data pipeline issue: {avg:.0f}ms avg delta - check if label generation aligned with features"
        else:
            result['avg_delta_ms'] = 0
            result['diagnosis'] = "All missing labels are at dataset edges (expected behavior - last 40 bars have no future data)"

        return result

    def _get_channel_sequence_from_shards(self, start: int, end: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get a sequence of rows from memory-mapped shards (or preloaded RAM).

        Always returns (main_channels, monthly_channels) as separate arrays.
        Merging happens in __getitem__/__getitems__ for efficiency.

        Args:
            start: Start row index (global)
            end: End row index (global)

        Returns:
            (main_channels, monthly_or_None) where:
            - main has channel cols (11718)
            - monthly has monthly cols (2604) or None
        """
        # FAST PATH: Use preloaded RAM data if available (avoids 400ms/sample disk I/O)
        # NOTE: preloaded arrays are already aligned (sliced during preload)
        if self.preloaded_main is not None:
            main_result = self.preloaded_main[start:end]
            monthly_result = self.preloaded_monthly[start:end] if self.preloaded_monthly is not None else None
            return main_result, monthly_result

        # SLOW PATH: Memory-mapped disk access
        import bisect

        # Find which shards contain our range
        start_shard = bisect.bisect_right(self.channel_cumulative_rows, start) - 1
        end_shard = bisect.bisect_right(self.channel_cumulative_rows, end - 1) - 1

        # Apply offset for monthly shard (it may have more rows from earlier dates)
        monthly_offset = getattr(self, '_monthly_offset', 0)

        if start_shard == end_shard:
            # Entire sequence in one shard (common case - 99%+)
            local_start = start - self.channel_cumulative_rows[start_shard]
            local_end = end - self.channel_cumulative_rows[start_shard]

            main_result = self.channel_mmaps[start_shard][local_start:local_end]
            # Apply offset when accessing monthly shard
            if self.monthly_3month_mmap is not None:
                monthly_result = self.monthly_3month_mmap[start + monthly_offset:end + monthly_offset, :]
            else:
                monthly_result = None

            return main_result, monthly_result
        else:
            # Sequence spans multiple shards (rare - at shard boundaries)
            main_cols = self.channel_mmaps[0].shape[1]
            main_result = np.empty((end - start, main_cols), dtype=self.channel_mmaps[0].dtype)
            pos = 0

            for shard_idx in range(start_shard, end_shard + 1):
                shard_start = max(start, self.channel_cumulative_rows[shard_idx])
                shard_end = min(end, self.channel_cumulative_rows[shard_idx + 1])

                local_start = shard_start - self.channel_cumulative_rows[shard_idx]
                local_end = shard_end - self.channel_cumulative_rows[shard_idx]
                length = local_end - local_start

                main_result[pos:pos + length] = self.channel_mmaps[shard_idx][local_start:local_end]
                pos += length

            # Apply offset when accessing monthly shard
            if self.monthly_3month_mmap is not None:
                monthly_result = self.monthly_3month_mmap[start + monthly_offset:end + monthly_offset, :]
            else:
                monthly_result = None
            return main_result, monthly_result

    def __getitem__(self, idx: int) -> Union[Tuple[tuple, dict], Tuple[Dict[str, np.ndarray], dict]]:
        """
        Get a single training sample with multi-task labels.

        Args:
            idx: Sample index

        Returns:
            If use_native_timeframes=False (legacy mode):
                x: Tuple(main_channels, monthly_channels_or_None, non_channel_sequence_or_None) each as np.ndarray [200, feat_dim]
            If use_native_timeframes=True (v4.1 mode):
                x: Dict[str, np.ndarray] mapping timeframe -> features [seq_len, feat_dim]

            targets: Dict with:
                - high: target_high % (regression)
                - low: target_low % (regression)
                - hit_band: 0/1 (classification) - NO CIRCULARITY
                - hit_target: 0/1 (classification) - NO CIRCULARITY
                - expected_return: % (regression) - NO CIRCULARITY
                - overshoot: ratio (regression) - NO CIRCULARITY
        """
        import time
        import sys
        _getitem_start = time.perf_counter()

        # v4.1: Native timeframe mode - return Dict[str, np.ndarray]
        if getattr(self, 'using_native_timeframes', False):
            return self._getitem_native_timeframe(idx)

        # Get actual data index
        data_idx = self.valid_indices[idx]

        # Extract input sequence
        seq_start = data_idx - self.sequence_length
        seq_end = data_idx

        # Handle memory-mapped shards vs normal array
        if self.using_mmaps:
            # Get channel features from shards (mmap - minimal RAM)
            main_channel_sequence, monthly_sequence = self._get_channel_sequence_from_shards(seq_start, seq_end)

            # Make contiguous here (one-time cost per sample, avoids collate copies)
            if main_channel_sequence is not None:
                main_channel_sequence = np.ascontiguousarray(main_channel_sequence)
            if monthly_sequence is not None:
                monthly_sequence = np.ascontiguousarray(monthly_sequence)

            # Get non-channel features (small, already in RAM)
            non_channel_sequence = self.non_channel_array[seq_start:seq_end, :] if self.non_channel_array is not None else None
            if non_channel_sequence is not None:
                non_channel_sequence = np.ascontiguousarray(non_channel_sequence)

            future_window = None
            if self.raw_ohlc_array is None:
                # Only build future window when we don't have raw OHLC (fallback path)
                future_start = seq_end
                future_end = seq_end + self.prediction_horizon
                main_future, monthly_future = self._get_channel_sequence_from_shards(future_start, future_end)

                parts = [main_future]
                if monthly_future is not None:
                    parts.append(monthly_future)

                if self.non_channel_array is not None:
                    non_channel_future = self.non_channel_array[future_start:future_end, :]
                    non_channel_future = np.ascontiguousarray(non_channel_future)
                    parts.append(non_channel_future)

                future_window = np.concatenate(parts, axis=1)
                # Concatenate may create non-contiguous result, ensure contiguous
                future_window = np.ascontiguousarray(future_window)
        else:
            # Normal path - single array
            main_channel_sequence = np.ascontiguousarray(self.features_array[seq_start:seq_end, :])  # [200, 299]
            monthly_sequence = None
            non_channel_sequence = None
            if self.raw_ohlc_array is None:
                future_start = seq_end
                future_end = seq_end + self.prediction_horizon
                future_window = np.ascontiguousarray(self.features_array[future_start:future_end, :])
            else:
                future_window = None

        # Calculate target (percentage change from current price)
        if self.using_mmaps and self.non_channel_array is not None:
            # tsla_close is in non_channel_array
            current_price = self.non_channel_array[seq_end - 1, self.close_idx]
        else:
            current_price = self.features_array[seq_end - 1, self.close_idx]

        # v3.18: Three adaptive modes:
        # - simple: All targets over fixed 24 bars
        # - adaptive_labels: high/low over 24, continuation over adaptive 20-40 (default)
        # - adaptive_full: ALL targets over adaptive 20-40 (sliced below if enabled)

        # Use actual OHLC high/low from raw_ohlc_array (not min/max of closes!)
        if self.raw_ohlc_array is not None:
            # Get future OHLC window [prediction_horizon, 4] where 4 = [open, high, low, close]
            future_ohlc = self.raw_ohlc_array[seq_end:seq_end + self.prediction_horizon]

            # Use ACTUAL intraday extremes (not just closes)
            future_high_actual = np.max(future_ohlc[:, 1])  # Column 1 = high
            future_low_actual = np.min(future_ohlc[:, 2])   # Column 2 = low
            future_prices = future_ohlc[:, 3]  # Column 3 = close (needed for multi-task labels)
        else:
            # Fallback: Use min/max of close prices (old simplified method)
            if self.using_mmaps and self.non_channel_array is not None:
                future_prices = future_window[:, self.num_channel_features + self.close_idx]
            else:
                future_prices = future_window[:, self.close_idx]

            future_high_actual = np.max(future_prices)
            future_low_actual = np.min(future_prices)

        # v3.18: Adaptive Full mode - slice ALL targets to adaptive horizon
        if config.CONTINUATION_MODE == 'adaptive_full' and self.include_continuation and self.continuation_labels_df is not None:
            try:
                # Get continuation label for this timestamp (O(1) lookup via dict)
                ts_ns = pd.Timestamp(self.timestamps[seq_end - 1]).value
                row_idx = self._continuation_ts_to_idx.get(ts_ns)

                if row_idx is not None and 'adaptive_horizon' in self.continuation_labels_df.columns:
                    adaptive_horizon = int(self.continuation_labels_df.iloc[row_idx]['adaptive_horizon'])

                    # Slice to adaptive horizon (cap at available length)
                    slice_len = min(adaptive_horizon, len(future_prices))

                    if slice_len < len(future_prices):
                        # Slice and recalculate ALL targets on adaptive window
                        if self.raw_ohlc_array is not None:
                            future_ohlc_sliced = future_ohlc[:slice_len]
                            future_high_actual = np.max(future_ohlc_sliced[:, 1])
                            future_low_actual = np.min(future_ohlc_sliced[:, 2])
                            future_prices = future_ohlc_sliced[:, 3]
                        else:
                            future_prices = future_prices[:slice_len]
                            future_high_actual = np.max(future_prices)
                            future_low_actual = np.min(future_prices)
            except:
                pass  # Fall back to fixed horizon if any issues

        # Convert to percentage change
        # Defensive check: prevent division by zero or invalid prices
        if current_price <= 0 or np.isnan(current_price) or np.isinf(current_price):
            raise ValueError(
                f"Invalid current_price at idx={idx}, data_idx={data_idx}, seq_end={seq_end}: "
                f"current_price={current_price}. This indicates a data alignment issue."
            )
        target_high_pct = (future_high_actual - current_price) / current_price * 100.0
        target_low_pct = (future_low_actual - current_price) / current_price * 100.0

        # ===== MULTI-TASK LABELS (NO CIRCULARITY - ALL FROM GROUND TRUTH) =====

        # Label 1: Hit Band (Did price respect ideal band?)
        # Ideal band = actual high/low with tolerance (NOT predicted band!)
        ideal_band_high = future_high_actual * 1.02  # +2% tolerance
        ideal_band_low = future_low_actual * 0.98    # -2% tolerance

        prices_in_ideal_band = (future_prices >= ideal_band_low) & (future_prices <= ideal_band_high)
        hit_band_label = float(prices_in_ideal_band.sum() / len(prices_in_ideal_band) > 0.8)

        # Label 2: Hit Target Before Stop
        # Simulate trade: entry at current_price, target at actual high, stop at 2% below actual low
        target_price = future_high_actual
        stop_price = current_price * (1 + target_low_pct/100 - 0.02)  # 2% below predicted low
        hit_target_label = float(self._check_target_sequence(
            future_prices, current_price, target_price, stop_price
        ))

        # Label 3: Expected Return (simulate actual trade execution)
        expected_return_label = self._simulate_trade_execution(
            future_prices, current_price, target_price, stop_price
        )

        # Label 4: Overshoot Ratio (how far actual exceeded ideal band)
        band_range = abs(target_high_pct - target_low_pct)
        if band_range > 0:
            overshoot_high = max(0, future_high_actual - ideal_band_high) / current_price * 100
            overshoot_low = max(0, ideal_band_low - future_low_actual) / current_price * 100
            overshoot_label = (overshoot_high + overshoot_low) / band_range
        else:
            overshoot_label = 0.0

        # Calculate adaptive targets
        actual_max_idx = future_prices.argmax()
        bars_to_peak = actual_max_idx  # Index directly represents bars into the future

        # Simple adaptive targets (can be enhanced with channel bounds calculation)
        adaptive_price_change = target_high_pct if target_high_pct > abs(target_low_pct) else target_low_pct
        adaptive_horizon_log = torch.log(torch.tensor(bars_to_peak / 24 + 1e-6, dtype=config.get_torch_dtype()))  # Normalized log with proper dtype
        adaptive_confidence = 1.0 if bars_to_peak > 48 else 0.5  # Simple confidence

        targets = {
            'high': target_high_pct,
            'low': target_low_pct,
            'hit_band': hit_band_label,
            'hit_target': hit_target_label,
            'expected_return': expected_return_label,
            'overshoot': overshoot_label,
            'continuation_duration': 0.0,   # Placeholder
            'continuation_gain': 0.0,       # Placeholder
            'continuation_confidence': 0.5, # Placeholder
            'price_change_pct': adaptive_price_change,
            'horizon_bars_log': adaptive_horizon_log,
            'adaptive_confidence': adaptive_confidence
        }

        # ===== BREAKOUT LABELS (v3.21) =====
        # Detect channel breakout using past prices for channel definition
        # and future prices for breakout detection
        try:
            # Get past close prices for channel calculation
            if self.raw_ohlc_array is not None:
                past_ohlc = self.raw_ohlc_array[seq_start:seq_end]
                past_prices_for_channel = past_ohlc[:, 3]  # Close prices
            elif self.using_mmaps and self.non_channel_array is not None:
                past_prices_for_channel = self.non_channel_array[seq_start:seq_end, self.close_idx]
            else:
                past_prices_for_channel = self.features_array[seq_start:seq_end, self.close_idx]

            breakout_labels = self._detect_channel_breakout(
                past_prices=past_prices_for_channel,
                future_prices=future_prices,
                current_price=current_price,
                lookback=60,  # 1 hour channel
                channel_std=2.0,
                breakout_threshold=1.0
            )

            targets['breakout_occurred'] = breakout_labels['breakout_occurred']
            targets['breakout_direction'] = breakout_labels['breakout_direction']
            targets['breakout_bars_log'] = breakout_labels['breakout_bars_log']
            targets['breakout_magnitude'] = breakout_labels['breakout_magnitude']

        except Exception as e:
            # Fallback values if breakout detection fails
            targets['breakout_occurred'] = 0.0
            targets['breakout_direction'] = 0.5
            targets['breakout_bars_log'] = np.log(self.prediction_horizon + 1)
            targets['breakout_magnitude'] = 0.0

        # Add continuation prediction targets if enabled
        if self.include_continuation and self.continuation_labels_df is not None:
            try:
                # Fast timestamp lookup using pre-converted int64 nanoseconds
                # (avoids pd.Timestamp() creation per sample - ~50µs savings)
                ts_ns = self._timestamps_ns[seq_end - 1]
                row_idx = self._continuation_ts_to_idx.get(ts_ns)

                if row_idx is not None:
                    # Found exact match - use pre-extracted numpy arrays (O(1) access)
                    # Return scalars, collate handles tensor conversion once per batch
                    targets['continuation_duration'] = float(self._cont_duration[row_idx])
                    targets['continuation_gain'] = float(self._cont_gain[row_idx])
                    targets['continuation_confidence'] = float(self._cont_confidence[row_idx])

                    # Add optional fields if they exist
                    if self.has_adaptive_horizon:
                        targets['adaptive_horizon'] = float(self._cont_adaptive_horizon[row_idx])
                    if self.has_conf_score:
                        targets['conf_score'] = float(self._cont_conf_score[row_idx])
                    if self.has_channel_1h_cycles:
                        targets['channel_1h_cycles'] = float(self._cont_channel_1h_cycles[row_idx])
                    if self.has_channel_4h_cycles:
                        targets['channel_4h_cycles'] = float(self._cont_channel_4h_cycles[row_idx])
                    if self.has_channel_1h_valid:
                        targets['channel_1h_valid'] = float(self._cont_channel_1h_valid[row_idx])
                    if self.has_channel_4h_valid:
                        targets['channel_4h_valid'] = float(self._cont_channel_4h_valid[row_idx])
                    if self.has_channel_1h_r_squared:
                        targets['channel_1h_r_squared'] = float(self._cont_channel_1h_r_squared[row_idx])
                    if self.has_channel_4h_r_squared:
                        targets['channel_4h_r_squared'] = float(self._cont_channel_4h_r_squared[row_idx])
                else:
                    # No exact match - use fallback values (scalars)
                    self._missing_label_count += 1

                    # Log first few mismatches with diagnostic info
                    if self._logged_mismatches < 5:
                        ts = pd.Timestamp(self.timestamps[seq_end - 1])
                        print(f"     ⚠️  Sample {idx}: Continuation label missing for {ts}")
                        print(f"        Lookup key (ns): {ts_ns}")
                        print(f"        Dict size: {len(self._continuation_ts_to_idx)}")
                        # Find nearest key for diagnosis
                        if len(self._continuation_ts_to_idx) > 0:
                            import bisect
                            sorted_keys = sorted(self._continuation_ts_to_idx.keys())
                            pos = bisect.bisect_left(sorted_keys, ts_ns)
                            if pos < len(sorted_keys):
                                nearest = sorted_keys[pos]
                                delta_sec = (nearest - ts_ns) / 1e9
                                print(f"        Nearest key: {pd.Timestamp(nearest)} (delta: {delta_sec:.1f}s)")
                        self._logged_mismatches += 1

                    # Use fallback values (scalars, not tensors)
                    targets['continuation_duration'] = 0.0
                    targets['continuation_gain'] = 0.0
                    targets['continuation_confidence'] = 0.5

                    # Add fallback for all optional fields to maintain dict consistency
                    if self.has_adaptive_horizon:
                        targets['adaptive_horizon'] = 24.0
                    if self.has_conf_score:
                        targets['conf_score'] = 0.5
                    if self.has_channel_1h_cycles:
                        targets['channel_1h_cycles'] = 0.0
                    if self.has_channel_4h_cycles:
                        targets['channel_4h_cycles'] = 0.0
                    if self.has_channel_1h_valid:
                        targets['channel_1h_valid'] = 0.0
                    if self.has_channel_4h_valid:
                        targets['channel_4h_valid'] = 0.0
                    if self.has_channel_1h_r_squared:
                        targets['channel_1h_r_squared'] = 0.0
                    if self.has_channel_4h_r_squared:
                        targets['channel_4h_r_squared'] = 0.0

            except Exception as e:
                # Exception occurred - use fallback values (scalars)
                self._missing_label_count += 1

                targets['continuation_duration'] = 0.0
                targets['continuation_gain'] = 0.0
                targets['continuation_confidence'] = 0.5

                # Add fallback for all optional fields to maintain dict consistency
                if self.has_adaptive_horizon:
                    targets['adaptive_horizon'] = 24.0
                if self.has_conf_score:
                    targets['conf_score'] = 0.5
                if self.has_channel_1h_cycles:
                    targets['channel_1h_cycles'] = 0.0
                if self.has_channel_4h_cycles:
                    targets['channel_4h_cycles'] = 0.0
                if self.has_channel_1h_valid:
                    targets['channel_1h_valid'] = 0.0
                if self.has_channel_4h_valid:
                    targets['channel_4h_valid'] = 0.0
                if self.has_channel_1h_r_squared:
                    targets['channel_1h_r_squared'] = 0.0
                if self.has_channel_4h_r_squared:
                    targets['channel_4h_r_squared'] = 0.0

        # Performance logging for slow samples (diagnose mmap read bottlenecks)
        _getitem_elapsed_ms = (time.perf_counter() - _getitem_start) * 1000
        if _getitem_elapsed_ms > 50:  # Log if >50ms (should be <10ms with premerge)
            print(f"[SLOW_GETITEM] idx={idx} took {_getitem_elapsed_ms:.0f}ms", file=sys.stderr, flush=True)

        return (main_channel_sequence, monthly_sequence, non_channel_sequence), targets

    def _check_target_sequence(
        self,
        prices: np.ndarray,
        entry_price: float,
        target_price: float,
        stop_price: float
    ) -> bool:
        """
        Check if target was hit before stop in price sequence.

        Args:
            prices: Future price sequence (ground truth)
            entry_price: Entry price
            target_price: Target price to hit
            stop_price: Stop loss price

        Returns:
            1.0 if hit target before stop, 0.0 otherwise
        """
        for price in prices:
            if price >= target_price:
                return True  # Hit target first
            if price <= stop_price:
                return False  # Hit stop first
        return False  # Neither hit within horizon

    def _simulate_trade_execution(
        self,
        prices: np.ndarray,
        entry_price: float,
        target_price: float,
        stop_price: float
    ) -> float:
        """
        Simulate trade execution and return realized return %.

        Args:
            prices: Future price sequence (ground truth)
            entry_price: Entry price
            target_price: Target price
            stop_price: Stop loss price

        Returns:
            Realized return percentage
        """
        for price in prices:
            if price >= target_price:
                # Hit target - exit with profit
                return (target_price - entry_price) / entry_price * 100.0
            if price <= stop_price:
                # Hit stop - exit with loss
                return (stop_price - entry_price) / entry_price * 100.0

        # Neither hit - hold to end of horizon
        final_price = prices[-1]
        return (final_price - entry_price) / entry_price * 100.0

    def _detect_channel_breakout(
        self,
        past_prices: np.ndarray,
        future_prices: np.ndarray,
        current_price: float,
        lookback: int = 60,
        channel_std: float = 2.0,
        breakout_threshold: float = 1.0
    ) -> dict:
        """
        Detect channel breakout in future prices based on past channel bounds.

        Uses linear regression channel from past prices and detects when future
        prices break out of the channel. This is a FORWARD-LOOKING label (no leakage
        since we only use it for training targets, not features).

        Args:
            past_prices: Historical close prices (input sequence) - shape [seq_len]
            future_prices: Future close prices (label window) - shape [horizon]
            current_price: Current price at prediction time
            lookback: Bars to use for channel calculation (default 60 = 1 hour)
            channel_std: Channel width in std deviations (default 2.0)
            breakout_threshold: Std deviations beyond channel for breakout (default 1.0)

        Returns:
            Dict with:
                - breakout_occurred: 1.0 if breakout happened, 0.0 otherwise
                - breakout_direction: 1.0 if upward, 0.0 if downward (or no breakout)
                - breakout_bars: Bars until breakout (log-scaled), or max horizon
                - breakout_magnitude: How far price moved beyond channel (% of channel width)
        """
        # Default values (no breakout)
        result = {
            'breakout_occurred': 0.0,
            'breakout_direction': 0.5,  # Neutral
            'breakout_bars_log': np.log(len(future_prices) + 1),  # Max horizon (log-scaled)
            'breakout_magnitude': 0.0
        }

        # Need enough past data for channel calculation
        if len(past_prices) < lookback or len(past_prices) < 10:
            return result

        # Calculate channel from past prices (last `lookback` bars)
        y = past_prices[-lookback:]
        X = np.arange(lookback)

        # Fit linear regression
        X_mean = X.mean()
        y_mean = y.mean()
        slope = np.sum((X - X_mean) * (y - y_mean)) / (np.sum((X - X_mean) ** 2) + 1e-10)
        intercept = y_mean - slope * X_mean

        # Calculate residuals and channel width
        fitted = slope * X + intercept
        residuals = y - fitted
        channel_width = np.std(residuals) + 1e-10  # Avoid division by zero

        # Project channel forward
        # The channel at bar i (future) extends the regression line
        upper_bounds = []
        lower_bounds = []
        for i in range(len(future_prices)):
            future_bar_idx = lookback + i  # Continue from end of past sequence
            projected_center = slope * future_bar_idx + intercept
            upper_bounds.append(projected_center + channel_std * channel_width)
            lower_bounds.append(projected_center - channel_std * channel_width)

        upper_bounds = np.array(upper_bounds)
        lower_bounds = np.array(lower_bounds)

        # Detect breakout
        breakout_threshold_dist = breakout_threshold * channel_width

        for i, price in enumerate(future_prices):
            # Check for upward breakout
            if price > upper_bounds[i] + breakout_threshold_dist:
                result['breakout_occurred'] = 1.0
                result['breakout_direction'] = 1.0  # Up
                result['breakout_bars_log'] = np.log(i + 1 + 1e-6)  # Log-scaled bars (+1 to avoid log(0))
                result['breakout_magnitude'] = (price - upper_bounds[i]) / channel_width
                break

            # Check for downward breakout
            elif price < lower_bounds[i] - breakout_threshold_dist:
                result['breakout_occurred'] = 1.0
                result['breakout_direction'] = 0.0  # Down
                result['breakout_bars_log'] = np.log(i + 1 + 1e-6)
                result['breakout_magnitude'] = (lower_bounds[i] - price) / channel_width
                break

        return result

    def get_sample_info(self, idx: int) -> dict:
        """
        Get metadata about a sample (for debugging).

        Args:
            idx: Sample index

        Returns:
            info: Dict with timestamp, price, targets, etc.
        """
        data_idx = self.valid_indices[idx]
        seq_end = data_idx

        current_price = self.features_array[seq_end - 1, self.close_idx]
        timestamp = self.timestamps[seq_end - 1]

        # Get targets
        _, targets = self.__getitem__(idx)

        return {
            'idx': idx,
            'data_idx': data_idx,
            'timestamp': pd.Timestamp(timestamp),
            'current_price': current_price,
            'target_high_pct': targets['high'].item(),
            'target_low_pct': targets['low'].item(),
            'hit_band': targets['hit_band'].item(),
            'hit_target': targets['hit_target'].item(),
            'expected_return': targets['expected_return'].item(),
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon
        }

    def __getitems__(self, indices):
        """
        Batch-optimized data loading with sorted access for mmap locality.

        PyTorch DataLoader calls this when available (PyTorch 2.0+).
        Key optimization: Sort indices by data position for sequential mmap access,
        which dramatically improves OS page cache hit rate for spinning disks and
        reduces random I/O for SSDs.

        Args:
            indices: List of sample indices to load

        Returns:
            List of ((main, monthly, nc), targets) tuples in original index order
        """
        import time
        import sys
        _batch_start = time.perf_counter()

        batch_size = len(indices)

        # Map to (original_position, sample_idx, data_idx) for sorting
        indexed = [(i, idx, self.valid_indices[idx]) for i, idx in enumerate(indices)]

        # Sort by data_idx for sequential mmap access (key optimization)
        # BUT: Only sort when using disk mmap - RAM has O(1) random access
        # - mmap mode + not preloaded: Sort (disk I/O optimization)
        # - mmap mode + preloaded: Don't sort (data copied to RAM)
        # - non-mmap mode: Don't sort (data already in self.features_array RAM)
        if self.using_mmaps and not self._preload_to_ram:
            sorted_indexed = sorted(indexed, key=lambda x: x[2])
        else:
            sorted_indexed = indexed  # Keep original shuffled order for RAM mode

        # Pre-allocate results list
        results = [None] * batch_size

        # Load samples in sorted order (sequential access pattern)
        for orig_pos, sample_idx, _ in sorted_indexed:
            results[orig_pos] = self.__getitem__(sample_idx)

        # Performance logging for slow batches (only if debug mode enabled)
        import os
        _batch_elapsed_ms = (time.perf_counter() - _batch_start) * 1000
        if _batch_elapsed_ms > 500 and os.environ.get('TRAIN_DEBUG', '0') == '1':  # Log if batch takes >500ms
            per_sample_ms = _batch_elapsed_ms / batch_size
            print(f"[SLOW_GETITEMS] {batch_size} samples took {_batch_elapsed_ms:.0f}ms "
                  f"({per_sample_ms:.1f}ms/sample)", file=sys.stderr, flush=True)

        return results


class PreloadHierarchicalDataset(Dataset):
    """
    Preloaded version of HierarchicalDataset.

    Loads all sequences into memory at initialization.
    Faster training but requires more RAM (~30-40 GB for full dataset).
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        raw_ohlc_df: pd.DataFrame = None,
        continuation_labels_df: pd.DataFrame = None,
        sequence_length: int = 200,
        prediction_horizon: int = 24,
        mode: str = 'uniform_bars'
    ):
        """Initialize preloaded dataset."""
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.mode = mode

        print(f"Preloading dataset with {len(features_df)} bars...")

        # Memory warning for large datasets
        estimated_samples = len(features_df) - sequence_length - prediction_horizon
        estimated_gb = (estimated_samples * sequence_length * features_df.shape[1] *
                       (8 if config.get_torch_dtype() == torch.float64 else 4)) / 1e9

        # Dynamic memory check based on actual available RAM
        # IMPORTANT: psutil can misread container RAM (sees host instead of cgroup limit)
        import os
        container_ram_gb = float(os.environ.get('CONTAINER_RAM_GB', '0'))

        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024**3)
            total_gb = psutil.virtual_memory().total / (1024**3)

            # Detect container or use override
            if container_ram_gb > 0:
                total_gb = container_ram_gb
                available_gb = container_ram_gb * 0.8
                print(f"    ℹ️  Using container RAM override: {container_ram_gb}GB")
            elif total_gb > 200:  # Likely seeing host RAM
                print(f"    ⚠️  Container detected: psutil sees {total_gb:.0f}GB (host RAM)")
                total_gb = 46  # Conservative default
                available_gb = 40

            warn_threshold = available_gb * 0.8  # Warn if using >80% of available RAM
        except ImportError:
            available_gb = 50  # Fallback assumption
            total_gb = 64
            warn_threshold = 50

        if estimated_gb > warn_threshold:
            print(f"⚠️  WARNING: Estimated memory usage: {estimated_gb:.1f} GB")
            print(f"    Available RAM: {available_gb:.1f} GB (of {total_gb:.1f} GB total)")
            if estimated_gb > available_gb:
                print(f"    This WILL cause OOM! Use lazy loading (preload=False) instead.")
            else:
                print(f"    This may cause swap usage or OOM errors!")
            print(f"    Consider using lazy loading (preload=False) for datasets this large.")
            response = input("    Continue with preload? (y/n): ")
            if response.lower() != 'y':
                raise MemoryError("Preload cancelled by user due to memory constraints")

        # Create lazy dataset first
        lazy_dataset = HierarchicalDataset(
            features_df,
            raw_ohlc_df,
            continuation_labels_df,
            sequence_length,
            prediction_horizon,
            mode,
            cache_indices=True,
            include_continuation=continuation_labels_df is not None
        )

        # Preload all samples
        num_samples = len(lazy_dataset)
        self.X = torch.zeros((num_samples, sequence_length, features_df.shape[1]), dtype=config.get_torch_dtype())

        # Multi-task targets (store separately)
        self.targets = {
            'high': torch.zeros(num_samples, dtype=config.get_torch_dtype()),
            'low': torch.zeros(num_samples, dtype=config.get_torch_dtype()),
            'hit_band': torch.zeros(num_samples, dtype=config.get_torch_dtype()),
            'hit_target': torch.zeros(num_samples, dtype=config.get_torch_dtype()),
            'expected_return': torch.zeros(num_samples, dtype=config.get_torch_dtype()),
            'overshoot': torch.zeros(num_samples, dtype=config.get_torch_dtype()),
            'continuation_duration': torch.zeros(num_samples, dtype=config.get_torch_dtype()),
            'continuation_gain': torch.zeros(num_samples, dtype=config.get_torch_dtype()),
            'continuation_confidence': torch.zeros(num_samples, dtype=config.get_torch_dtype())
        }

        print(f"Loading {num_samples} sequences...")

        from tqdm import tqdm
        for i in tqdm(range(num_samples), desc="  Preloading", ncols=100):
            x, targets_dict = lazy_dataset[i]
            self.X[i] = x

            # Store each target
            for key in self.targets.keys():
                self.targets[key][i] = targets_dict[key]

        print(f"Preload complete. Memory usage: ~{self.X.element_size() * self.X.nelement() / 1e9:.2f} GB")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """Get preloaded sample."""
        targets_dict = {
            key: self.targets[key][idx]
            for key in self.targets.keys()
        }
        return self.X[idx], targets_dict


def create_hierarchical_dataset(
    features_df: pd.DataFrame,
    raw_ohlc_df: pd.DataFrame = None,
    continuation_labels_df: pd.DataFrame = None,
    continuation_labels_dir: str = None,  # v4.3: Directory with per-TF label files
    sequence_length: int = 200,
    prediction_horizon: int = 24,
    mode: str = 'uniform_bars',
    preload: bool = False,
    validation_split: Optional[float] = None,
    test_split: Optional[float] = None,  # v4.4: Support for 3-way split
    include_continuation: bool = False,
    mmap_meta_path: str = None,
    profiler=None,
    preload_to_ram: bool = False,
    use_native_timeframes: bool = False,
    tf_meta_path: str = None
) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
    """
    Factory function to create hierarchical dataset(s).

    Args:
        features_df: Features DataFrame
        continuation_labels_df: [Legacy] Single DataFrame with continuation labels
        continuation_labels_dir: [v4.3] Directory with per-TF label files
        sequence_length: Input sequence length
        prediction_horizon: Prediction horizon in bars
        mode: 'uniform_bars'
        preload: If True, use PreloadHierarchicalDataset (legacy)
        validation_split: If provided, split data into train/val or train/val/test
        test_split: If provided, create held-out test set (v4.4)
        profiler: Optional MemoryProfiler for logging RAM usage
        preload_to_ram: If True, load all mmap data into RAM at startup
                       (requires ~90GB RAM but avoids 400ms/sample disk I/O)

    Returns:
        train_dataset: Training dataset
        val_dataset: Validation dataset (if validation_split provided)
        test_dataset: Test dataset (if test_split provided)
    """
    # v4.4: Support for 3-way split (train/val/test)
    if test_split is not None and validation_split is not None:
        # 3-way split (e.g., 85/10/5)
        total_len = len(features_df)
        train_split_ratio = 1.0 - validation_split - test_split
        train_end_idx = int(total_len * train_split_ratio)
        val_end_idx = int(total_len * (train_split_ratio + validation_split))

        # Calculate date ranges for logging
        train_start_date = features_df.index[0].date()
        train_end_date = features_df.index[min(train_end_idx - 1, total_len - 1)].date()
        val_start_date = features_df.index[train_end_idx].date()
        val_end_date = features_df.index[min(val_end_idx - 1, total_len - 1)].date()
        test_start_date = features_df.index[val_end_idx].date()
        test_end_date = features_df.index[total_len - 1].date()

        print(f"\n📊 3-Way Split:")
        print(f"   Train: {train_end_idx:,} samples ({train_start_date} to {train_end_date}) [{train_split_ratio*100:.0f}%]")
        print(f"   Val:   {val_end_idx - train_end_idx:,} samples ({val_start_date} to {val_end_date}) [{validation_split*100:.0f}%]")
        print(f"   Test:  {total_len - val_end_idx:,} samples ({test_start_date} to {test_end_date}) [{test_split*100:.0f}%]")
        print(f"   ⚠️  Test set will NOT be used during training (held-out evaluation only)\n")

        # All datasets get full continuation labels (lookup by timestamp)
        train_continuation_df = continuation_labels_df
        val_continuation_df = continuation_labels_df
        test_continuation_df = continuation_labels_df

        if mmap_meta_path:
            # When using mmaps, create ONE base dataset with FULL data,
            # then restrict valid_indices for each split
            import copy

            base_dataset = HierarchicalDataset(
                features_df,  # Full non-channel features
                raw_ohlc_df,
                None,  # Continuation handled separately below
                continuation_labels_dir=continuation_labels_dir,
                sequence_length=sequence_length,
                prediction_horizon=prediction_horizon,
                mode=mode,
                include_continuation=include_continuation,
                mmap_meta_path=mmap_meta_path,
                profiler=profiler,
                preload_to_ram=preload_to_ram,
                use_native_timeframes=use_native_timeframes,
                tf_meta_path=tf_meta_path
            )

            # Calculate index ranges for 3-way split
            all_valid = base_dataset.valid_indices
            train_valid = [i for i in all_valid if i < train_end_idx]
            val_valid = [i for i in all_valid if train_end_idx <= i < val_end_idx]
            test_valid = [i for i in all_valid if i >= val_end_idx]

            # Train dataset (modify in place)
            train_dataset = base_dataset
            train_dataset.valid_indices = train_valid
            train_dataset.continuation_labels_df = train_continuation_df
            train_dataset.include_continuation = include_continuation
            train_dataset._build_continuation_lookup()

            # Val dataset (shallow copy)
            val_dataset = copy.copy(base_dataset)
            val_dataset.valid_indices = val_valid
            val_dataset.continuation_labels_df = val_continuation_df
            val_dataset.include_continuation = include_continuation
            val_dataset._build_continuation_lookup()

            # Test dataset (shallow copy)
            test_dataset = copy.copy(base_dataset)
            test_dataset.valid_indices = test_valid
            test_dataset.continuation_labels_df = test_continuation_df
            test_dataset.include_continuation = include_continuation
            test_dataset._build_continuation_lookup()

        else:
            # No mmaps - traditional split
            train_df = features_df.iloc[:train_end_idx]
            val_df = features_df.iloc[train_end_idx:val_end_idx]
            test_df = features_df.iloc[val_end_idx:]

            # Slice raw_ohlc_df to match features
            train_raw_ohlc = raw_ohlc_df.iloc[:train_end_idx] if raw_ohlc_df is not None else None
            val_raw_ohlc = raw_ohlc_df.iloc[train_end_idx:val_end_idx] if raw_ohlc_df is not None else None
            test_raw_ohlc = raw_ohlc_df.iloc[val_end_idx:] if raw_ohlc_df is not None else None

            if preload:
                train_dataset = PreloadHierarchicalDataset(
                    train_df, train_raw_ohlc, train_continuation_df,
                    sequence_length, prediction_horizon, mode
                )
                val_dataset = PreloadHierarchicalDataset(
                    val_df, val_raw_ohlc, val_continuation_df,
                    sequence_length, prediction_horizon, mode
                )
                test_dataset = PreloadHierarchicalDataset(
                    test_df, test_raw_ohlc, test_continuation_df,
                    sequence_length, prediction_horizon, mode
                )
            else:
                train_dataset = HierarchicalDataset(
                    train_df, train_raw_ohlc, train_continuation_df,
                    continuation_labels_dir=continuation_labels_dir,
                    sequence_length=sequence_length,
                    prediction_horizon=prediction_horizon,
                    mode=mode,
                    include_continuation=include_continuation,
                    profiler=profiler,
                    preload_to_ram=preload_to_ram,
                    use_native_timeframes=use_native_timeframes,
                    tf_meta_path=tf_meta_path
                )
                val_dataset = HierarchicalDataset(
                    val_df, val_raw_ohlc, val_continuation_df,
                    continuation_labels_dir=continuation_labels_dir,
                    sequence_length=sequence_length,
                    prediction_horizon=prediction_horizon,
                    mode=mode,
                    include_continuation=include_continuation,
                    profiler=profiler,
                    preload_to_ram=preload_to_ram,
                    use_native_timeframes=use_native_timeframes,
                    tf_meta_path=tf_meta_path
                )
                test_dataset = HierarchicalDataset(
                    test_df, test_raw_ohlc, test_continuation_df,
                    continuation_labels_dir=continuation_labels_dir,
                    sequence_length=sequence_length,
                    prediction_horizon=prediction_horizon,
                    mode=mode,
                    include_continuation=include_continuation,
                    profiler=profiler,
                    preload_to_ram=preload_to_ram,
                    use_native_timeframes=use_native_timeframes,
                    tf_meta_path=tf_meta_path
                )

        return train_dataset, val_dataset, test_dataset

    elif validation_split is not None:
        # 2-way split (backward compatible)
        split_idx = int(len(features_df) * (1 - validation_split))
        print(f"Split data: {split_idx:,} train rows, {len(features_df) - split_idx:,} val rows")

        # Give FULL continuation labels to both datasets
        # (lookup is by timestamp, so each dataset only uses what it needs)
        # Previously we split by timestamp, but valid_indices are split by position,
        # causing misalignment when data isn't perfectly time-sorted
        train_continuation_df = continuation_labels_df
        val_continuation_df = continuation_labels_df

        if mmap_meta_path:
            # When using mmaps, create ONE base dataset with FULL data,
            # then restrict valid_indices to avoid duplicating mmap references
            import copy

            base_dataset = HierarchicalDataset(
                features_df,  # Full non-channel features
                raw_ohlc_df,
                None,  # Continuation handled separately below
                continuation_labels_dir=continuation_labels_dir,  # v4.3: Per-TF labels
                sequence_length=sequence_length,
                prediction_horizon=prediction_horizon,
                mode=mode,
                include_continuation=include_continuation,
                mmap_meta_path=mmap_meta_path,
                profiler=profiler,
                preload_to_ram=preload_to_ram,
                use_native_timeframes=use_native_timeframes,
                tf_meta_path=tf_meta_path
            )

            # Calculate index ranges (valid_indices already has built-in buffers)
            all_valid = base_dataset.valid_indices
            train_valid = [i for i in all_valid if i < split_idx]
            val_valid = [i for i in all_valid if i >= split_idx]

            # Train dataset (modify in place)
            train_dataset = base_dataset
            train_dataset.valid_indices = train_valid
            train_dataset.continuation_labels_df = train_continuation_df
            train_dataset.include_continuation = include_continuation
            train_dataset._build_continuation_lookup()  # Rebuild dict for legacy labels

            # Val dataset (shallow copy, shares mmap/arrays but different indices)
            # NOTE: per-TF labels are shared (lookup is by timestamp, so safe)
            val_dataset = copy.copy(base_dataset)
            val_dataset.valid_indices = val_valid
            val_dataset.continuation_labels_df = val_continuation_df
            val_dataset.include_continuation = include_continuation
            val_dataset._build_continuation_lookup()  # Rebuild dict for legacy labels

        else:
            # No mmaps - traditional split
            train_df = features_df.iloc[:split_idx]
            val_df = features_df.iloc[split_idx:]

            # Also slice raw_ohlc_df to match features (critical for correct target calculation)
            train_raw_ohlc = raw_ohlc_df.iloc[:split_idx] if raw_ohlc_df is not None else None
            val_raw_ohlc = raw_ohlc_df.iloc[split_idx:] if raw_ohlc_df is not None else None

            if preload:
                train_dataset = PreloadHierarchicalDataset(
                    train_df, train_raw_ohlc, train_continuation_df,
                    sequence_length, prediction_horizon, mode
                )
                val_dataset = PreloadHierarchicalDataset(
                    val_df, val_raw_ohlc, val_continuation_df,
                    sequence_length, prediction_horizon, mode
                )
            else:
                train_dataset = HierarchicalDataset(
                    train_df, train_raw_ohlc, train_continuation_df,
                    continuation_labels_dir=continuation_labels_dir,
                    sequence_length=sequence_length,
                    prediction_horizon=prediction_horizon,
                    mode=mode,
                    include_continuation=include_continuation,
                    profiler=profiler,
                    preload_to_ram=preload_to_ram,
                    use_native_timeframes=use_native_timeframes,
                    tf_meta_path=tf_meta_path
                )
                val_dataset = HierarchicalDataset(
                    val_df, val_raw_ohlc, val_continuation_df,
                    continuation_labels_dir=continuation_labels_dir,
                    sequence_length=sequence_length,
                    prediction_horizon=prediction_horizon,
                    mode=mode,
                    include_continuation=include_continuation,
                    profiler=profiler,
                    preload_to_ram=preload_to_ram,
                    use_native_timeframes=use_native_timeframes,
                    tf_meta_path=tf_meta_path
                )

        return train_dataset, val_dataset, None  # v4.4: Return None for test_dataset (2-way split)
    else:
        # No validation split
        if preload:
            dataset = PreloadHierarchicalDataset(
                features_df, raw_ohlc_df, continuation_labels_df,
                sequence_length, prediction_horizon, mode
            )
        else:
            dataset = HierarchicalDataset(
                features_df, raw_ohlc_df, continuation_labels_df,
                continuation_labels_dir=continuation_labels_dir,
                sequence_length=sequence_length,
                prediction_horizon=prediction_horizon,
                mode=mode,
                include_continuation=include_continuation,
                mmap_meta_path=mmap_meta_path,
                profiler=profiler,
                preload_to_ram=preload_to_ram,
                use_native_timeframes=use_native_timeframes,
                tf_meta_path=tf_meta_path
            )

        return dataset, None, None  # v4.4: Return None for val_dataset and test_dataset (no split)


def test_hierarchical_dataset():
    """
    Test function for dataset.

    Loads a small sample and verifies output shapes.
    """
    from src.ml.data_feed import CSVDataFeed

    print("Testing HierarchicalDataset...")

    # Load 1-min data
    data_feed = CSVDataFeed(timeframe='1min')
    df = data_feed.load_aligned_data(
        start_date='2023-01-01',
        end_date='2023-12-31'
    )

    print(f"Loaded {len(df)} bars")

    # Extract features
    extractor = TradingFeatureExtractor()
    features_df, _ = extractor.extract_features(df)

    print(f"Extracted {len(features_df.columns)} features")

    # Create dataset
    dataset = HierarchicalDataset(
        features_df,
        sequence_length=200,
        prediction_horizon=24
    )

    print(f"Dataset size: {len(dataset)} sequences")

    # Test sample
    x, y = dataset[0]
    print(f"Sample 0:")
    print(f"  X shape: {x.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Target high: {y[0].item():.2f}%")
    print(f"  Target low: {y[1].item():.2f}%")

    # Test sample info
    info = dataset.get_sample_info(0)
    print(f"Sample info:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("\nTest passed! ✓")


if __name__ == '__main__':
    test_hierarchical_dataset()
