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
        features_df: pd.DataFrame,
        raw_ohlc_df: pd.DataFrame = None,
        continuation_labels_df: pd.DataFrame = None,
        sequence_length: int = 200,
        prediction_horizon: int = 24,
        mode: str = 'uniform_bars',
        cache_indices: bool = True,
        include_continuation: bool = False
    ):
        """
        Initialize dataset.

        Args:
            features_df: Features dataframe with 495+ columns
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

        # Convert to numpy for speed
        self.features_array = features_df.values
        self.timestamps = features_df.index.values

        # Dtype validation - ensure data matches config precision
        expected_dtype = config.NUMPY_DTYPE
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
            self.feature_names = features_df.columns.tolist()
            self.close_idx = self.feature_names.index('tsla_close')
            self.high_idx = self.feature_names.index('tsla_close')  # Will calc from close + returns
            self.low_idx = self.feature_names.index('tsla_close')
        else:
            self.close_idx = None

        # Calculate valid indices
        # Need: sequence_length bars for input + prediction_horizon bars for target
        self.min_context = sequence_length
        self.total_required = sequence_length + prediction_horizon

        # Valid start indices
        self.valid_indices = list(range(
            self.min_context,
            len(self.features_array) - prediction_horizon
        ))

        if len(self.valid_indices) == 0:
            raise ValueError(
                f"Not enough data. Need at least {self.total_required} bars, "
                f"but have {len(self.features_array)}"
            )

    def __len__(self) -> int:
        """Return number of valid sequences."""
        return len(self.valid_indices)

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

        x = self.features_array[seq_start:seq_end, :]  # [200, 299]

        # Extract future window for target
        future_start = seq_end
        future_end = seq_end + self.prediction_horizon

        future_window = self.features_array[future_start:future_end, :]

        # Calculate target (percentage change from current price)
        current_price = self.features_array[seq_end - 1, self.close_idx]

        # Get future prices (GROUND TRUTH)
        future_prices = future_window[:, self.close_idx]

        # Calculate high and low (primary targets)
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
            except:
                # Fallback values
                targets['continuation_duration'] = torch.tensor(0.0, dtype=config.get_torch_dtype())
                targets['continuation_gain'] = torch.tensor(0.0, dtype=config.get_torch_dtype())
                targets['continuation_confidence'] = torch.tensor(0.5, dtype=config.get_torch_dtype())

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
    include_continuation: bool = False
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
        # Split into train/val
        split_idx = int(len(features_df) * (1 - validation_split))

        train_df = features_df.iloc[:split_idx]
        val_df = features_df.iloc[split_idx:]

        print(f"Split data: {len(train_df)} train, {len(val_df)} val")

        # Split continuation labels if provided
        train_continuation_df = None
        val_continuation_df = None
        if continuation_labels_df is not None:
            # Split continuation labels by timestamp
            split_timestamp = train_df.index[-1]
            train_continuation_df = continuation_labels_df[
                continuation_labels_df['timestamp'] <= split_timestamp
            ].copy()
            val_continuation_df = continuation_labels_df[
                continuation_labels_df['timestamp'] > split_timestamp
            ].copy()

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
                include_continuation=include_continuation
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
