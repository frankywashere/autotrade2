"""
Channel data loader for visualization.

Loads channel features from memory-mapped shards and raw OHLC data.
Supports both local and external drive storage locations.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import config


class ChannelLoader:
    """Load channel data from cached mmap shards for visualization."""

    def __init__(self, shard_path: Path):
        """
        Initialize loader with shard storage path.

        Args:
            shard_path: Path to feature_cache directory (local or external drive)
        """
        self.shard_path = Path(shard_path)

        if not self.shard_path.exists():
            raise FileNotFoundError(f"Shard path does not exist: {shard_path}")

        # Load metadata
        meta_files = list(self.shard_path.glob('features_mmap_meta_*.json'))

        if not meta_files:
            raise FileNotFoundError(
                f"No shard metadata found in {shard_path}\n"
                f"Expected: features_mmap_meta_*.json\n"
                f"Run feature extraction first to generate shards"
            )

        self.meta_file = meta_files[0]
        with open(self.meta_file) as f:
            self.meta = json.load(f)

        print(f"✓ Loaded shard metadata: {self.meta_file.name}")
        print(f"  Shards: {len(self.meta['chunk_info'])}")
        print(f"  Total rows: {self.meta['total_rows']:,}")
        print(f"  Features: {self.meta['num_features']:,}")
        print(f"  Dtype: {self.meta['dtype']}")

        # Load all shards as memory-maps
        self.channel_mmaps = []
        self.timestamps_mmaps = []
        self.cumulative_rows = [0]

        for chunk_info in self.meta['chunk_info']:
            # Load channel features
            mmap_array = np.load(chunk_info['path'], mmap_mode='r')
            self.channel_mmaps.append(mmap_array)

            # Load timestamps
            ts_array = np.load(chunk_info['index_path'], mmap_mode='r')
            self.timestamps_mmaps.append(ts_array)

            # Track cumulative rows
            self.cumulative_rows.append(self.cumulative_rows[-1] + chunk_info['rows'])

        # Concatenate all timestamps
        self.all_timestamps = np.concatenate(self.timestamps_mmaps)

        print(f"✓ Loaded {len(self.channel_mmaps)} shards into memory-map mode")

        # Build feature name index for quick lookups
        self._build_feature_index()

    def _build_feature_index(self):
        """Build index mapping feature names to column indices."""
        # Reconstruct feature names matching extraction code
        from src.ml.features import TradingFeatureExtractor

        extractor = TradingFeatureExtractor()
        all_feature_names = extractor.get_feature_names()

        # Channel features only (first 8,778 in v3.17)
        self.channel_feature_names = [
            name for name in all_feature_names
            if '_channel_' in name
        ]

        # Create lookup dict
        self.feature_to_idx = {name: idx for idx, name in enumerate(self.channel_feature_names)}

        print(f"✓ Indexed {len(self.channel_feature_names)} channel features")

    def _find_timestamp_index(self, timestamp: pd.Timestamp) -> Optional[int]:
        """Find index of timestamp in mmap."""
        # Convert numpy datetime64 to timestamp for comparison
        ts_index = pd.DatetimeIndex(self.all_timestamps)

        try:
            idx = ts_index.get_indexer([timestamp], method='nearest')[0]
            if idx >= 0:
                return idx
        except:
            pass

        return None

    def _find_shard_for_index(self, global_idx: int) -> Tuple[int, int]:
        """Find which shard contains this global index."""
        import bisect
        shard_idx = bisect.bisect_right(self.cumulative_rows, global_idx) - 1
        local_idx = global_idx - self.cumulative_rows[shard_idx]
        return shard_idx, local_idx

    def get_channel_metrics(
        self,
        timestamp: pd.Timestamp,
        symbol: str = 'tsla',
        timeframe: str = '1h',
        window: int = 168
    ) -> Dict:
        """
        Get all channel metrics for a specific channel.

        Args:
            timestamp: Timestamp to query
            symbol: 'tsla' or 'spy'
            timeframe: '5min', '1h', '4h', 'daily', etc.
            window: Window size (10, 20, ..., 168)

        Returns:
            Dictionary with all channel metrics
        """
        # Find timestamp index
        idx = self._find_timestamp_index(timestamp)

        if idx is None:
            raise ValueError(f"Timestamp {timestamp} not found in shards")

        # Find shard
        shard_idx, local_idx = self._find_shard_for_index(idx)

        # Extract all metrics for this channel
        prefix = f'{symbol}_channel_{timeframe}'
        metrics = [
            'position', 'upper_dist', 'lower_dist', 'slope', 'slope_pct', 'stability',
            'ping_pongs', 'ping_pongs_0_5pct', 'ping_pongs_1_0pct', 'ping_pongs_3_0pct',
            'complete_cycles', 'complete_cycles_0_5pct', 'complete_cycles_1_0pct', 'complete_cycles_3_0pct',
            'r_squared', 'is_bull', 'is_bear', 'is_sideways', 'duration',
            'quality_score', 'is_valid'
        ]

        result = {'timestamp': timestamp, 'symbol': symbol, 'timeframe': timeframe, 'window': window}

        for metric in metrics:
            feature_name = f'{prefix}_{metric}_w{window}'

            if feature_name in self.feature_to_idx:
                col_idx = self.feature_to_idx[feature_name]
                value = self.channel_mmaps[shard_idx][local_idx, col_idx]
                result[metric] = float(value)

        return result

    def get_raw_ohlc_window(
        self,
        timestamp: pd.Timestamp,
        symbol: str = 'tsla',
        window_bars: int = 168,
        timeframe: str = '1h'
    ) -> pd.DataFrame:
        """
        Load raw OHLC data for the window used in channel calculation.

        Args:
            timestamp: End timestamp of channel
            symbol: 'tsla' or 'spy'
            window_bars: Number of bars in window
            timeframe: Timeframe of the channel

        Returns:
            DataFrame with OHLC data for the window
        """
        # Load from CSV (or could use mmap if available)
        from src.ml.data_feed import CSVDataFeed

        # Determine which CSV to load based on timeframe
        if timeframe in ['5min', '15min', '30min', '1h', '2h', '3h', '4h']:
            feed = CSVDataFeed(timeframe='1min')  # Load 1-min, will resample
        else:
            feed = CSVDataFeed(timeframe='1hour')  # Load hourly for daily/weekly

        # Load around timestamp (with buffer)
        # Calculate approximate date range needed
        import datetime
        end_date = timestamp
        start_date = timestamp - datetime.timedelta(days=30)  # 30-day buffer

        df = feed.load_data(
            symbol.upper(),
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

        # Resample if needed
        if timeframe != '1min':
            timeframe_map = {
                '5min': '5T', '15min': '15T', '30min': '30T',
                '1h': '1h', '2h': '2h', '3h': '3h', '4h': '4h',
                'daily': '1D', 'weekly': '1W', 'monthly': '1M'
            }
            resample_rule = timeframe_map.get(timeframe, '1h')

            df = df.resample(resample_rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

        # Get window ending at timestamp
        try:
            end_idx = df.index.get_indexer([timestamp], method='nearest')[0]
            start_idx = max(0, end_idx - window_bars + 1)

            window_df = df.iloc[start_idx:end_idx+1].copy()

            if len(window_df) < window_bars // 2:
                raise ValueError(f"Insufficient data: Got {len(window_df)} bars, need ~{window_bars}")

            return window_df

        except Exception as e:
            raise ValueError(f"Could not extract window: {e}")

    def find_high_quality_timestamps(
        self,
        symbol: str = 'tsla',
        timeframe: str = '1h',
        window: int = 168,
        min_quality: float = 0.8,
        limit: int = 100
    ) -> list:
        """
        Find timestamps with high-quality channels.

        Args:
            symbol: 'tsla' or 'spy'
            timeframe: Timeframe to check
            window: Window size
            min_quality: Minimum quality_score (0-1)
            limit: Max results to return

        Returns:
            List of (timestamp, quality_score) tuples
        """
        feature_name = f'{symbol}_channel_{timeframe}_quality_score_w{window}'

        if feature_name not in self.feature_to_idx:
            raise ValueError(f"Feature {feature_name} not found in shards")

        col_idx = self.feature_to_idx[feature_name]

        results = []

        # Search through shards
        for shard_idx, mmap_array in enumerate(self.channel_mmaps):
            quality_values = mmap_array[:, col_idx]

            # Find indices above threshold
            high_quality_indices = np.where(quality_values >= min_quality)[0]

            # Get timestamps for these indices
            for local_idx in high_quality_indices:
                global_idx = self.cumulative_rows[shard_idx] + local_idx
                ts = pd.Timestamp(self.all_timestamps[global_idx])
                quality = float(quality_values[local_idx])

                results.append((ts, quality))

                if len(results) >= limit:
                    break

            if len(results) >= limit:
                break

        # Sort by quality descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:limit]

    def get_summary_stats(self) -> Dict:
        """Get summary statistics across all channels."""
        # Sample 1000 random timestamps
        sample_size = min(1000, len(self.all_timestamps))
        sample_indices = np.random.choice(len(self.all_timestamps), sample_size, replace=False)

        stats = {
            'total_timestamps': len(self.all_timestamps),
            'sample_size': sample_size,
            'metrics': {}
        }

        # For each metric, calculate distribution
        metrics_to_analyze = ['quality_score', 'complete_cycles', 'ping_pongs', 'r_squared', 'is_valid']

        for metric in metrics_to_analyze:
            # Get feature for tsla_1h_w168 as example
            feature_name = f'tsla_channel_1h_{metric}_w168'

            if feature_name in self.feature_to_idx:
                col_idx = self.feature_to_idx[feature_name]

                # Sample values
                values = []
                for idx in sample_indices:
                    shard_idx, local_idx = self._find_shard_for_index(idx)
                    val = self.channel_mmaps[shard_idx][local_idx, col_idx]
                    values.append(float(val))

                stats['metrics'][metric] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }

        return stats
