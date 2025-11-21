"""
Hierarchical Dataset for 1-min Data Loading

Optimized lazy loading dataset for training HierarchicalLNN.
Loads 1-min data and dynamically creates training sequences with:
- 200 1-min bars as input
- Target high/low in next 24 bars (prediction horizon)
- Percentage-based targets (not absolute prices)
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

import config  # For precision configuration

from src.ml.features import TradingFeatureExtractor


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
        continuation_labels_df: pd.DataFrame = None,
        sequence_length: int = 200,
        prediction_horizon: int = 24,
        mode: str = 'uniform_bars',
        cache_indices: bool = True,
        include_continuation: bool = False,
        mmap_meta_path: str = None
    ):
        """
        Initialize dataset.

        Args:
            features_df: Features dataframe (165 non-channel + optional mmap 12,474 channel = 12,639 total)
            raw_ohlc_df: Raw OHLC data for input sequences
            continuation_labels_df: DataFrame with continuation labels
            sequence_length: Input sequence length (200 1-min bars)
            prediction_horizon: How many bars ahead to predict (24 = 24 minutes)
            mode: 'uniform_bars' (fixed # bars ahead)
            cache_indices: Cache column lookups for speed
            include_continuation: Whether to include continuation prediction targets
        """
        self.features_df = features_df
        self.raw_ohlc_df = raw_ohlc_df
        self.continuation_labels_df = continuation_labels_df
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.mode = mode
        self.include_continuation = include_continuation

        # Dtype validation - ensure data matches config precision
        expected_dtype = config.NUMPY_DTYPE

        # Memory-mapped shard loading (zero RAM spike!)
        if mmap_meta_path is not None:
            import json
            import numpy as np
            from pathlib import Path

            print(f"  📂 Loading memory-mapped channel shards...")
            meta = json.load(open(mmap_meta_path))

            # Load channel feature shards as memory-maps
            self.channel_mmaps = []
            self.channel_cumulative_rows = [0]

            for info in meta['chunk_info']:
                mmap_array = np.load(info['path'], mmap_mode='r')
                self.channel_mmaps.append(mmap_array)
                self.channel_cumulative_rows.append(self.channel_cumulative_rows[-1] + info['rows'])

            # Load timestamps from shards
            all_timestamps = []
            for info in meta['chunk_info']:
                idx_array = np.load(info['index_path'], mmap_mode='r')
                all_timestamps.append(idx_array)
            self.timestamps = np.concatenate(all_timestamps)

            # Load non-channel features normally (these are small - ~181 features)
            if features_df is not None:
                self.non_channel_array = features_df.values.astype(config.NUMPY_DTYPE)
            else:
                self.non_channel_array = None

            self.using_mmaps = True
            self.num_channel_features = meta['num_features']
            print(f"     ✓ Loaded {len(self.channel_mmaps)} channel shards ({self.num_channel_features:,} features)")
            print(f"     ✓ Loaded {self.non_channel_array.shape[1] if self.non_channel_array is not None else 0} non-channel features")
            print(f"     ✓ Total rows: {meta['total_rows']:,}")

        else:
            # Normal path - load everything into RAM
            self.using_mmaps = False
            self.features_array = features_df.values
            self.timestamps = features_df.index.values

            # Validate and convert feature array dtype if needed
            if self.features_array.dtype != expected_dtype:
                print(f"  ⚠️  Feature dtype mismatch: {self.features_array.dtype} != {expected_dtype}")
                print(f"     Converting to {expected_dtype} (may use extra memory temporarily)")
                self.features_array = self.features_array.astype(expected_dtype)

        if raw_ohlc_df is not None:
            self.raw_ohlc_array = raw_ohlc_df[['tsla_open', 'tsla_high', 'tsla_low', 'tsla_close']].values
            # Ensure OHLC also matches dtype
            if self.raw_ohlc_array.dtype != expected_dtype:
                self.raw_ohlc_array = self.raw_ohlc_array.astype(expected_dtype)
        else:
            self.raw_ohlc_array = None

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

    def __len__(self) -> int:
        """Return number of valid sequences."""
        return len(self.valid_indices)

    def _get_channel_sequence_from_shards(self, start: int, end: int) -> np.ndarray:
        """
        Get a sequence of rows from memory-mapped shards (zero copy, minimal RAM).

        Args:
            start: Start row index (global)
            end: End row index (global)

        Returns:
            Array of shape [end-start, num_channel_features]
        """
        import bisect

        # Find which shards contain our range
        start_shard = bisect.bisect_right(self.channel_cumulative_rows, start) - 1
        end_shard = bisect.bisect_right(self.channel_cumulative_rows, end - 1) - 1

        if start_shard == end_shard:
            # Entire sequence in one shard (common case - fast, zero-copy!)
            local_start = start - self.channel_cumulative_rows[start_shard]
            local_end = end - self.channel_cumulative_rows[start_shard]
            return self.channel_mmaps[start_shard][local_start:local_end]
        else:
            # Sequence spans multiple shards (rare - happens at shard boundaries)
            # Pre-allocate contiguous array instead of vstack (avoids extra copy)
            result = np.empty((end - start, self.num_channel_features), dtype=self.channel_mmaps[0].dtype)
            pos = 0

            for shard_idx in range(start_shard, end_shard + 1):
                shard_start = max(start, self.channel_cumulative_rows[shard_idx])
                shard_end = min(end, self.channel_cumulative_rows[shard_idx + 1])

                local_start = shard_start - self.channel_cumulative_rows[shard_idx]
                local_end = shard_end - self.channel_cumulative_rows[shard_idx]

                length = local_end - local_start
                result[pos:pos + length] = self.channel_mmaps[shard_idx][local_start:local_end]
                pos += length

            return result

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """
        Get a single training sample with multi-task labels.

        Args:
            idx: Sample index

        Returns:
            x: Input features [200, 299]
            targets: Dict with:
                - high: target_high % (regression)
                - low: target_low % (regression)
                - hit_band: 0/1 (classification) - NO CIRCULARITY
                - hit_target: 0/1 (classification) - NO CIRCULARITY
                - expected_return: % (regression) - NO CIRCULARITY
                - overshoot: ratio (regression) - NO CIRCULARITY
        """
        # Get actual data index
        data_idx = self.valid_indices[idx]

        # Extract input sequence
        seq_start = data_idx - self.sequence_length
        seq_end = data_idx

        # Handle memory-mapped shards vs normal array
        if self.using_mmaps:
            # Get channel features from shards (mmap - minimal RAM)
            channel_sequence = self._get_channel_sequence_from_shards(seq_start, seq_end)

            # Get non-channel features (small, already in RAM)
            if self.non_channel_array is not None:
                non_channel_sequence = self.non_channel_array[seq_start:seq_end, :]
                # Concatenate: [200, 12936] + [200, ~181] = [200, ~13117]
                x = np.concatenate([channel_sequence, non_channel_sequence], axis=1)
            else:
                x = channel_sequence  # Only channels

            # Extract future window for target (also from shards)
            future_start = seq_end
            future_end = seq_end + self.prediction_horizon
            channel_future = self._get_channel_sequence_from_shards(future_start, future_end)

            if self.non_channel_array is not None:
                non_channel_future = self.non_channel_array[future_start:future_end, :]
                future_window = np.concatenate([channel_future, non_channel_future], axis=1)
            else:
                future_window = channel_future
        else:
            # Normal path - single array
            x = self.features_array[seq_start:seq_end, :]  # [200, 299]
            future_start = seq_end
            future_end = seq_end + self.prediction_horizon
            future_window = self.features_array[future_start:future_end, :]

        # Calculate target (percentage change from current price)
        if self.using_mmaps and self.non_channel_array is not None:
            # tsla_close is in non_channel_array
            current_price = self.non_channel_array[seq_end - 1, self.close_idx]
        else:
            current_price = self.features_array[seq_end - 1, self.close_idx]

        # v3.17: Use actual OHLC high/low from raw_ohlc_array (not min/max of closes!)
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

        # Convert to percentage change
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

        # Convert to tensors (dtype from config for precision flexibility)
        x_tensor = torch.tensor(x, dtype=config.get_torch_dtype())

        # Calculate adaptive targets
        actual_max_idx = future_prices.argmax()
        bars_to_peak = actual_max_idx  # Index directly represents bars into the future

        # Simple adaptive targets (can be enhanced with channel bounds calculation)
        adaptive_price_change = target_high_pct if target_high_pct > abs(target_low_pct) else target_low_pct
        adaptive_horizon_log = torch.log(torch.tensor(bars_to_peak / 24 + 1e-6))  # Normalized log
        adaptive_confidence = 1.0 if bars_to_peak > 48 else 0.5  # Simple confidence

        targets = {
            'high': torch.tensor(target_high_pct, dtype=config.get_torch_dtype()),
            'low': torch.tensor(target_low_pct, dtype=config.get_torch_dtype()),
            'hit_band': torch.tensor(hit_band_label, dtype=config.get_torch_dtype()),
            'hit_target': torch.tensor(hit_target_label, dtype=config.get_torch_dtype()),
            'expected_return': torch.tensor(expected_return_label, dtype=config.get_torch_dtype()),
            'overshoot': torch.tensor(overshoot_label, dtype=config.get_torch_dtype()),
            'continuation_duration': torch.tensor(0.0, dtype=config.get_torch_dtype()),  # Placeholder
            'continuation_gain': torch.tensor(0.0, dtype=config.get_torch_dtype()),     # Placeholder
            'continuation_confidence': torch.tensor(0.5, dtype=config.get_torch_dtype()), # Placeholder
            'price_change_pct': torch.tensor(adaptive_price_change, dtype=config.get_torch_dtype()),
            'horizon_bars_log': adaptive_horizon_log,
            'adaptive_confidence': torch.tensor(adaptive_confidence, dtype=config.get_torch_dtype())
        }

        # Add continuation prediction targets if enabled
        if self.include_continuation and self.continuation_labels_df is not None:
            try:
                # Find continuation label for this timestamp
                ts = pd.Timestamp(self.timestamps[seq_end - 1])
                cont_row = self.continuation_labels_df[self.continuation_labels_df['timestamp'] == ts]
                if not cont_row.empty:
                    targets['continuation_duration'] = torch.tensor(cont_row['duration_hours'].iloc[0], dtype=config.get_torch_dtype())
                    targets['continuation_gain'] = torch.tensor(cont_row['projected_gain'].iloc[0], dtype=config.get_torch_dtype())
                    targets['continuation_confidence'] = torch.tensor(cont_row['confidence'].iloc[0], dtype=config.get_torch_dtype())

                    # Add adaptive fields if they exist (when using adaptive mode)
                    if 'adaptive_horizon' in cont_row.columns:
                        targets['adaptive_horizon'] = torch.tensor(cont_row['adaptive_horizon'].iloc[0], dtype=config.get_torch_dtype())
                    if 'conf_score' in cont_row.columns:
                        targets['conf_score'] = torch.tensor(cont_row['conf_score'].iloc[0], dtype=config.get_torch_dtype())

                    # v3.17: Channel quality fields for timeframe switching
                    if 'channel_1h_cycles' in cont_row.columns:
                        targets['channel_1h_cycles'] = torch.tensor(cont_row['channel_1h_cycles'].iloc[0], dtype=config.get_torch_dtype())
                    if 'channel_4h_cycles' in cont_row.columns:
                        targets['channel_4h_cycles'] = torch.tensor(cont_row['channel_4h_cycles'].iloc[0], dtype=config.get_torch_dtype())
                    if 'channel_1h_valid' in cont_row.columns:
                        targets['channel_1h_valid'] = torch.tensor(cont_row['channel_1h_valid'].iloc[0], dtype=config.get_torch_dtype())
                    if 'channel_4h_valid' in cont_row.columns:
                        targets['channel_4h_valid'] = torch.tensor(cont_row['channel_4h_valid'].iloc[0], dtype=config.get_torch_dtype())
                    if 'channel_1h_r_squared' in cont_row.columns:
                        targets['channel_1h_r_squared'] = torch.tensor(cont_row['channel_1h_r_squared'].iloc[0], dtype=config.get_torch_dtype())
                    if 'channel_4h_r_squared' in cont_row.columns:
                        targets['channel_4h_r_squared'] = torch.tensor(cont_row['channel_4h_r_squared'].iloc[0], dtype=config.get_torch_dtype())
            except:
                # Fallback values
                targets['continuation_duration'] = torch.tensor(0.0, dtype=config.get_torch_dtype())
                targets['continuation_gain'] = torch.tensor(0.0, dtype=config.get_torch_dtype())
                targets['continuation_confidence'] = torch.tensor(0.5, dtype=config.get_torch_dtype())
                # Fallback for adaptive fields if expected but not found
                if self.continuation_labels_df is not None and 'adaptive_horizon' in self.continuation_labels_df.columns:
                    targets['adaptive_horizon'] = torch.tensor(24.0, dtype=config.get_torch_dtype())  # Default to min horizon
                if self.continuation_labels_df is not None and 'conf_score' in self.continuation_labels_df.columns:
                    targets['conf_score'] = torch.tensor(0.5, dtype=config.get_torch_dtype())  # Default to neutral confidence

        return x_tensor, targets

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

        if estimated_gb > 50:
            print(f"⚠️  WARNING: Estimated memory usage: {estimated_gb:.1f} GB")
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
    sequence_length: int = 200,
    prediction_horizon: int = 24,
    mode: str = 'uniform_bars',
    preload: bool = False,
    validation_split: Optional[float] = None,
    include_continuation: bool = False,
    mmap_meta_path: str = None
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Factory function to create hierarchical dataset(s).

    Args:
        features_df: Features DataFrame
        sequence_length: Input sequence length
        prediction_horizon: Prediction horizon in bars
        mode: 'uniform_bars'
        preload: If True, preload all data into memory
        validation_split: If provided, split data into train/val

    Returns:
        train_dataset: Training dataset
        val_dataset: Validation dataset (if validation_split provided)
    """
    if validation_split is not None:
        # Calculate split index
        split_idx = int(len(features_df) * (1 - validation_split))
        print(f"Split data: {split_idx:,} train rows, {len(features_df) - split_idx:,} val rows")

        # Split continuation labels if provided
        train_continuation_df = None
        val_continuation_df = None
        if continuation_labels_df is not None:
            # Split continuation labels by timestamp
            split_timestamp = features_df.index[split_idx - 1]
            train_continuation_df = continuation_labels_df[
                continuation_labels_df['timestamp'] <= split_timestamp
            ].copy()
            val_continuation_df = continuation_labels_df[
                continuation_labels_df['timestamp'] > split_timestamp
            ].copy()

        if mmap_meta_path:
            # When using mmaps, create ONE base dataset with FULL data,
            # then restrict valid_indices to avoid duplicating mmap references
            import copy

            base_dataset = HierarchicalDataset(
                features_df,  # Full non-channel features
                raw_ohlc_df,
                None,  # Continuation handled separately below
                sequence_length, prediction_horizon, mode,
                include_continuation=False,
                mmap_meta_path=mmap_meta_path
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

            # Val dataset (shallow copy, shares mmap/arrays but different indices)
            val_dataset = copy.copy(base_dataset)
            val_dataset.valid_indices = val_valid
            val_dataset.continuation_labels_df = val_continuation_df
            val_dataset.include_continuation = include_continuation

        else:
            # No mmaps - traditional split
            train_df = features_df.iloc[:split_idx]
            val_df = features_df.iloc[split_idx:]

            if preload:
                train_dataset = PreloadHierarchicalDataset(
                    train_df, raw_ohlc_df, train_continuation_df,
                    sequence_length, prediction_horizon, mode
                )
                val_dataset = PreloadHierarchicalDataset(
                    val_df, raw_ohlc_df, val_continuation_df,
                    sequence_length, prediction_horizon, mode
                )
            else:
                train_dataset = HierarchicalDataset(
                    train_df, raw_ohlc_df, train_continuation_df,
                    sequence_length, prediction_horizon, mode,
                    include_continuation=include_continuation
                )
                val_dataset = HierarchicalDataset(
                    val_df, raw_ohlc_df, val_continuation_df,
                    sequence_length, prediction_horizon, mode,
                    include_continuation=include_continuation
                )

        return train_dataset, val_dataset
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
                sequence_length, prediction_horizon, mode,
                include_continuation=include_continuation,
                mmap_meta_path=mmap_meta_path
            )

        return dataset, None


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
