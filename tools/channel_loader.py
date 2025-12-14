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
        self.monthly_mmap = None

        for chunk_info in self.meta['chunk_info']:
            # Load channel features (prepend shard_path for relative paths)
            chunk_path = self.shard_path / chunk_info['path']
            mmap_array = np.load(chunk_path, mmap_mode='r')
            self.channel_mmaps.append(mmap_array)

            # Load timestamps
            index_path = self.shard_path / chunk_info['index_path']
            ts_array = np.load(index_path, mmap_mode='r')
            self.timestamps_mmaps.append(ts_array)

            # Track cumulative rows
            self.cumulative_rows.append(self.cumulative_rows[-1] + chunk_info['rows'])

        # Load monthly/3month shard if present
        if 'monthly_3month_shard' in self.meta and self.meta['monthly_3month_shard']:
            m_info = self.meta['monthly_3month_shard']
            m_path = Path(m_info['path'])
            # Prepend shard_path if path is relative
            if not m_path.is_absolute():
                m_path = self.shard_path / m_path
            if m_path.exists():
                self.monthly_mmap = np.load(str(m_path), mmap_mode='r')
                print(f"  ✓ Loaded monthly/3month shard: {self.monthly_mmap.shape[0]:,} rows × {self.monthly_mmap.shape[1]} cols")
            else:
                print(f"  ⚠️ Monthly/3month shard missing at {m_path}")

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

        # Channel features only
        channel_feature_names = [name for name in all_feature_names if '_channel_' in name]

        # Split main vs monthly/3month for hybrid shards
        def is_monthly(name: str) -> bool:
            return '_channel_monthly_' in name or '_channel_3month_' in name

        self.main_channel_features = [n for n in channel_feature_names if not is_monthly(n)]
        self.monthly_channel_features = [n for n in channel_feature_names if is_monthly(n)]

        self.main_feature_to_idx = {name: idx for idx, name in enumerate(self.main_channel_features)}
        self.monthly_feature_to_idx = {name: idx for idx, name in enumerate(self.monthly_channel_features)}

        # Create unified feature_to_idx for backward compatibility
        # Note: Main features use main shard, monthly use monthly shard
        self.feature_to_idx = {**self.main_feature_to_idx}

        print(f"✓ Indexed {len(self.main_channel_features)} main channel features")
        if self.monthly_channel_features:
            print(f"✓ Indexed {len(self.monthly_channel_features)} monthly/3month features")

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
            'position', 'upper_dist', 'lower_dist',
            'close_slope', 'high_slope', 'low_slope',
            'close_slope_pct', 'high_slope_pct', 'low_slope_pct',
            'close_r_squared', 'high_r_squared', 'low_r_squared', 'r_squared_avg',
            'channel_width_pct', 'slope_convergence', 'stability',
            'ping_pongs', 'ping_pongs_0_5pct', 'ping_pongs_1_0pct', 'ping_pongs_3_0pct',
            'complete_cycles', 'complete_cycles_0_5pct', 'complete_cycles_1_0pct', 'complete_cycles_3_0pct',
            'is_bull', 'is_bear', 'is_sideways',
            'quality_score', 'is_valid', 'insufficient_data',
            'duration'
        ]

        result = {'timestamp': timestamp, 'symbol': symbol, 'timeframe': timeframe, 'window': window}

        # Select which mmap to use (monthly/3month vs main)
        if timeframe in ['monthly', '3month'] and self.monthly_mmap is not None:
            feature_map = self.monthly_feature_to_idx
            shard_array = self.monthly_mmap
        else:
            feature_map = self.main_feature_to_idx
            shard_array = self.channel_mmaps[shard_idx]

        for metric in metrics:
            feature_name = f'{prefix}_{metric}_w{window}'
            if feature_name in feature_map:
                col_idx = feature_map[feature_name]
                value = shard_array[local_idx, col_idx]
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

    # =========================================================================
    # v5.7: LABEL LOADING FOR VERIFICATION
    # =========================================================================

    def _load_labels(self):
        """Load continuation and transition labels if available."""
        if hasattr(self, '_labels_loaded') and self._labels_loaded:
            return

        self._continuation_labels = {}  # tf -> {'timestamps': array, 'duration_bars': array, 'max_gain_pct': array}
        self._transition_labels = {}    # tf -> {'timestamps': array, 'transition_type': array, 'new_direction': array}

        # Look for labels directory
        labels_dir = self.shard_path / 'continuation_labels'
        if not labels_dir.exists():
            # Try parent directory
            labels_dir = self.shard_path.parent / 'continuation_labels'

        if not labels_dir.exists():
            print(f"  ⚠️ Labels directory not found at {labels_dir}")
            self._labels_loaded = True
            return

        # Load continuation labels for each TF
        timeframes = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']

        for tf in timeframes:
            # Continuation labels
            cont_file = labels_dir / f'{tf}_labels.parquet'
            if cont_file.exists():
                try:
                    df = pd.read_parquet(cont_file)
                    self._continuation_labels[tf] = {
                        'timestamps': df.index.values,
                        'duration_bars': df['duration_bars'].values if 'duration_bars' in df.columns else None,
                        'max_gain_pct': df['max_gain_pct'].values if 'max_gain_pct' in df.columns else None,
                    }
                except Exception as e:
                    print(f"  ⚠️ Could not load {cont_file.name}: {e}")

            # Transition labels
            trans_file = labels_dir / f'transition_{tf}_labels.parquet'
            if trans_file.exists():
                try:
                    df = pd.read_parquet(trans_file)
                    self._transition_labels[tf] = {
                        'timestamps': df.index.values,
                        'transition_type': df['transition_type'].values if 'transition_type' in df.columns else None,
                        'new_direction': df['new_direction'].values if 'new_direction' in df.columns else None,
                    }
                except Exception as e:
                    print(f"  ⚠️ Could not load {trans_file.name}: {e}")

        loaded_cont = len(self._continuation_labels)
        loaded_trans = len(self._transition_labels)
        if loaded_cont > 0 or loaded_trans > 0:
            print(f"  ✓ Loaded labels: {loaded_cont} continuation, {loaded_trans} transition TFs")

        self._labels_loaded = True

    def get_continuation_label(
        self,
        timestamp: pd.Timestamp,
        timeframe: str = '1h'
    ) -> Optional[Dict]:
        """
        Get continuation label for a timestamp/timeframe.

        Args:
            timestamp: Timestamp to query
            timeframe: Timeframe

        Returns:
            Dict with 'duration_bars', 'max_gain_pct' or None if not found
        """
        self._load_labels()

        if timeframe not in self._continuation_labels:
            return None

        labels = self._continuation_labels[timeframe]
        if labels['timestamps'] is None:
            return None

        # Find closest timestamp
        ts_index = pd.DatetimeIndex(labels['timestamps'])
        try:
            idx = ts_index.get_indexer([timestamp], method='nearest')[0]
            if idx < 0:
                return None

            # Check if timestamp is close enough (within 1 hour)
            time_diff = abs((ts_index[idx] - timestamp).total_seconds())
            if time_diff > 3600:  # More than 1 hour away
                return None

            result = {'timestamp': ts_index[idx]}
            if labels['duration_bars'] is not None:
                result['duration_bars'] = int(labels['duration_bars'][idx])
            if labels['max_gain_pct'] is not None:
                result['max_gain_pct'] = float(labels['max_gain_pct'][idx])

            return result

        except Exception:
            return None

    def get_transition_label(
        self,
        timestamp: pd.Timestamp,
        timeframe: str = '1h'
    ) -> Optional[Dict]:
        """
        Get transition label for a timestamp/timeframe.

        Args:
            timestamp: Timestamp to query
            timeframe: Timeframe

        Returns:
            Dict with 'transition_type', 'new_direction' or None if not found
        """
        self._load_labels()

        if timeframe not in self._transition_labels:
            return None

        labels = self._transition_labels[timeframe]
        if labels['timestamps'] is None:
            return None

        # Find closest timestamp
        ts_index = pd.DatetimeIndex(labels['timestamps'])
        try:
            idx = ts_index.get_indexer([timestamp], method='nearest')[0]
            if idx < 0:
                return None

            # Check if timestamp is close enough (within 1 hour)
            time_diff = abs((ts_index[idx] - timestamp).total_seconds())
            if time_diff > 3600:
                return None

            result = {'timestamp': ts_index[idx]}
            if labels['transition_type'] is not None:
                result['transition_type'] = int(labels['transition_type'][idx])
                result['transition_name'] = ['CONTINUE', 'SWITCH_TF', 'REVERSE', 'SIDEWAYS'][result['transition_type']]
            if labels['new_direction'] is not None:
                result['new_direction'] = int(labels['new_direction'][idx])
                result['direction_name'] = ['BEAR', 'BULL', 'SIDEWAYS'][result['new_direction']] if result['new_direction'] < 3 else 'UNKNOWN'

            return result

        except Exception:
            return None

    def find_timestamps_by_transition_type(
        self,
        transition_type: int,
        timeframe: str = '1h',
        limit: int = 100
    ) -> list:
        """
        Find timestamps with a specific transition type.

        Args:
            transition_type: 0=continue, 1=switch_tf, 2=reverse, 3=sideways
            timeframe: Timeframe to search
            limit: Max results

        Returns:
            List of timestamps
        """
        self._load_labels()

        if timeframe not in self._transition_labels:
            return []

        labels = self._transition_labels[timeframe]
        if labels['transition_type'] is None:
            return []

        # Find matching indices
        matches = np.where(labels['transition_type'] == transition_type)[0]

        # Convert to timestamps
        results = []
        for idx in matches[:limit]:
            ts = pd.Timestamp(labels['timestamps'][idx])
            results.append(ts)

        return results

    def get_future_price_data(
        self,
        timestamp: pd.Timestamp,
        symbol: str = 'tsla',
        bars_forward: int = 50,
        timeframe: str = '1h'
    ) -> Optional[pd.DataFrame]:
        """
        Get price data AFTER the timestamp to verify breaks.

        Args:
            timestamp: Starting timestamp
            symbol: 'tsla' or 'spy'
            bars_forward: Number of bars to fetch after timestamp
            timeframe: Timeframe

        Returns:
            DataFrame with OHLC data after timestamp
        """
        from src.ml.data_feed import CSVDataFeed

        # Determine which CSV to load based on timeframe
        if timeframe in ['5min', '15min', '30min', '1h', '2h', '3h', '4h']:
            feed = CSVDataFeed(timeframe='1min')
        else:
            feed = CSVDataFeed(timeframe='1hour')

        # Load data after timestamp
        import datetime
        start_date = timestamp
        end_date = timestamp + datetime.timedelta(days=30)

        try:
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

            # Get bars after timestamp
            try:
                start_idx = df.index.get_indexer([timestamp], method='nearest')[0]
                end_idx = min(start_idx + bars_forward, len(df))

                return df.iloc[start_idx:end_idx].copy()
            except:
                return None

        except Exception as e:
            print(f"  ⚠️ Could not load future data: {e}")
            return None
