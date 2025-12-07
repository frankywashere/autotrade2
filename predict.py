"""
Live Prediction System for Hierarchical LNN v5.0

This module provides real-time prediction capabilities using a trained model.
It uses the EXACT SAME feature extraction pipeline as training to ensure
feature consistency.

Key Architecture:
1. LiveDataBuffer: Maintains OHLCV history for SPY + TSLA at multiple resolutions
2. LiveFeatureExtractor: Uses TradingFeatureExtractor in "live mode"
3. LivePredictor: Coordinates buffers, extraction, and model inference

Usage:
    from predict import LivePredictor

    predictor = LivePredictor(
        model_path='models/hierarchical_lnn.pth',
        tf_meta_path='data/feature_cache/tf_meta_*.json'
    )

    # Update with new data (must include both SPY and TSLA)
    predictor.update_bars(spy_bar, tsla_bar)

    # Get prediction
    result = predictor.predict()
    print(f"Predicted high: {result['predicted_high']:.4f}")
    print(f"Predicted low: {result['predicted_low']:.4f}")
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime, timedelta
from collections import deque
import warnings
from glob import glob as file_glob

# Import project modules
import config
from src.ml.hierarchical_model import HierarchicalLNN, load_hierarchical_model
from src.ml.features import (
    TradingFeatureExtractor,
    TIMEFRAME_SEQUENCE_LENGTHS,
    TIMEFRAME_RESAMPLE_RULES,
    HIERARCHICAL_TIMEFRAMES,
    load_vix_data,
)


def fetch_live_vix(days: int = 30) -> pd.DataFrame:
    """
    Fetch recent VIX data from yfinance.

    Args:
        days: Number of days of history to fetch (default: 30)

    Returns:
        DataFrame with columns: vix_open, vix_high, vix_low, vix_close
        Index: DatetimeIndex (date only)
    """
    try:
        import yfinance as yf

        # Fetch VIX data
        vix = yf.download("^VIX", period=f"{days}d", interval="1d", progress=False)

        if vix is None or len(vix) == 0:
            warnings.warn("No VIX data returned from yfinance")
            return pd.DataFrame()

        # Handle MultiIndex columns (yfinance 0.2+)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)

        # Create output DataFrame with renamed columns
        vix_df = pd.DataFrame(index=vix.index)
        vix_df['vix_open'] = vix['Open'].values
        vix_df['vix_high'] = vix['High'].values
        vix_df['vix_low'] = vix['Low'].values
        vix_df['vix_close'] = vix['Close'].values

        # Normalize index to date only (no time, no timezone)
        vix_df.index = pd.to_datetime(vix_df.index.date)

        return vix_df

    except ImportError:
        warnings.warn("yfinance not installed. Run: pip install yfinance")
        return pd.DataFrame()
    except Exception as e:
        warnings.warn(f"Failed to fetch live VIX: {e}")
        return pd.DataFrame()


def combine_vix_data(
    historical_vix: pd.DataFrame,
    live_vix: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine historical VIX (from CSV) with live VIX (from yfinance).

    Live data takes precedence for overlapping dates.

    Args:
        historical_vix: VIX data loaded from CSV
        live_vix: Recent VIX data from yfinance

    Returns:
        Combined DataFrame with all VIX history + recent data
    """
    if historical_vix is None or len(historical_vix) == 0:
        return live_vix

    if live_vix is None or len(live_vix) == 0:
        return historical_vix

    # Combine, with live data taking precedence
    combined = pd.concat([historical_vix, live_vix])

    # Remove duplicates, keeping the last (live) value
    combined = combined[~combined.index.duplicated(keep='last')]

    # Sort by date
    combined = combined.sort_index()

    return combined


def fetch_live_ohlcv(
    symbols: List[str] = None,
    days: int = 7,
    interval: str = '1m'
) -> Dict[str, pd.DataFrame]:
    """
    Fetch live OHLCV data from yfinance for multiple symbols.

    Args:
        symbols: List of ticker symbols (default: ['SPY', 'TSLA'])
        days: Number of days of history (default: 7, max ~30 for 1m data)
        interval: Data interval (default: '1m')

    Returns:
        Dict mapping symbol -> DataFrame with OHLCV data

    Note:
        yfinance limits 1-minute data to ~30 days of history.
        For longer history, use daily data or a paid data provider.
    """
    if symbols is None:
        symbols = ['SPY', 'TSLA']

    try:
        import yfinance as yf

        result = {}
        for symbol in symbols:
            print(f"   Fetching {symbol} ({interval}, {days} days)...")

            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d", interval=interval)

            if df is None or len(df) == 0:
                warnings.warn(f"No data returned for {symbol}")
                continue

            # Standardize column names to lowercase
            df.columns = df.columns.str.lower()

            # Keep only OHLCV columns
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            df = df[[c for c in ohlcv_cols if c in df.columns]]

            result[symbol] = df
            print(f"   ✓ {symbol}: {len(df):,} bars ({df.index[0]} to {df.index[-1]})")

        return result

    except ImportError:
        warnings.warn("yfinance not installed. Run: pip install yfinance")
        return {}
    except Exception as e:
        warnings.warn(f"Failed to fetch OHLCV data: {e}")
        return {}


def align_spy_tsla(
    spy_df: pd.DataFrame,
    tsla_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Align SPY and TSLA DataFrames into single combined DataFrame.

    Args:
        spy_df: SPY OHLCV DataFrame
        tsla_df: TSLA OHLCV DataFrame

    Returns:
        Combined DataFrame with spy_* and tsla_* columns
        Only includes timestamps where both have data
    """
    # Rename columns with prefix
    spy_renamed = spy_df.rename(columns={
        'open': 'spy_open', 'high': 'spy_high', 'low': 'spy_low',
        'close': 'spy_close', 'volume': 'spy_volume'
    })
    tsla_renamed = tsla_df.rename(columns={
        'open': 'tsla_open', 'high': 'tsla_high', 'low': 'tsla_low',
        'close': 'tsla_close', 'volume': 'tsla_volume'
    })

    # Inner join - only keep bars where both exist
    combined = spy_renamed.join(tsla_renamed, how='inner')

    # Remove any rows with NaN
    combined = combined.dropna()

    print(f"   Aligned: {len(combined):,} bars (SPY: {len(spy_df):,}, TSLA: {len(tsla_df):,})")

    return combined


class LiveDataBuffer:
    """
    Manages rolling OHLC data buffers for SPY and TSLA.

    The buffer maintains enough history at each resolution to support:
    - Feature extraction (needs lookback for channels)
    - Sequence generation (model input)

    Supported intervals (matching yfinance):
        - 1min: Raw tick data
        - 5min, 15min, 30min: Intraday
        - 1hour: Hourly
        - daily: Daily bars
        - weekly: Weekly bars
        - monthly: Monthly bars
    """

    # Required history per interval (bars needed for feature extraction + seq_len)
    # Formula: max_channel_window(168) + seq_len + buffer
    REQUIRED_HISTORY = {
        '1min': 50000,     # ~35 days (for resampling to higher TFs)
        '5min': 10000,     # ~35 days
        '15min': 4000,     # ~42 days
        '30min': 2000,     # ~42 days
        '1hour': 1000,     # ~42 days
        'daily': 500,      # ~2 years
        'weekly': 200,     # ~4 years
        'monthly': 60,     # ~5 years
    }

    def __init__(self):
        """Initialize empty data buffers."""
        self.buffers: Dict[str, pd.DataFrame] = {}
        self.last_update: Dict[str, datetime] = {}

        # Initialize empty DataFrames for each interval
        for interval in self.REQUIRED_HISTORY:
            self.buffers[interval] = pd.DataFrame()

    def load_from_dataframe(self, df: pd.DataFrame, interval: str = '1min') -> None:
        """
        Load historical data from a DataFrame.

        Args:
            df: DataFrame with columns for SPY and TSLA OHLCV
                Expected columns: spy_open, spy_high, spy_low, spy_close, spy_volume,
                                  tsla_open, tsla_high, tsla_low, tsla_close, tsla_volume
                Index: DatetimeIndex
            interval: Data interval ('1min', '5min', 'daily', etc.)
        """
        required_cols = [
            'spy_open', 'spy_high', 'spy_low', 'spy_close', 'spy_volume',
            'tsla_open', 'tsla_high', 'tsla_low', 'tsla_close', 'tsla_volume'
        ]

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Store the data
        buffer_df = df[required_cols].copy()

        # Normalize timezone - strip timezone info to avoid comparison issues
        # yfinance returns timezone-aware (America/New_York), but feature extraction expects naive
        if hasattr(buffer_df.index, 'tz') and buffer_df.index.tz is not None:
            # Convert to naive by replacing timezone info (keeps wall clock time)
            buffer_df.index = pd.DatetimeIndex(buffer_df.index.strftime('%Y-%m-%d %H:%M:%S'))

        self.buffers[interval] = buffer_df

        if len(df) > 0:
            self.last_update[interval] = buffer_df.index[-1]

        print(f"   Loaded {len(buffer_df):,} {interval} bars ({buffer_df.index[0]} to {buffer_df.index[-1]})")

    def update_bar(self, bar: Dict[str, Any], interval: str = '1min') -> None:
        """
        Add a new bar to the buffer.

        Args:
            bar: Dict with timestamp + SPY/TSLA OHLCV data
                 {timestamp, spy_open, spy_high, ..., tsla_open, tsla_high, ...}
            interval: Data interval
        """
        timestamp = pd.Timestamp(bar['timestamp'])

        # Create row from bar
        row = pd.DataFrame([{
            k: v for k, v in bar.items() if k != 'timestamp'
        }], index=[timestamp])

        # Append to buffer
        if len(self.buffers[interval]) == 0:
            self.buffers[interval] = row
        else:
            self.buffers[interval] = pd.concat([self.buffers[interval], row])

        # Trim to max size
        max_size = self.REQUIRED_HISTORY.get(interval, 10000)
        if len(self.buffers[interval]) > max_size:
            self.buffers[interval] = self.buffers[interval].iloc[-max_size:]

        self.last_update[interval] = timestamp

    def resample_from_1min(self, skip_daily_and_longer: bool = False) -> None:
        """
        Resample 1-min data to higher intraday timeframes.

        Args:
            skip_daily_and_longer: If True, don't resample to daily/weekly/monthly
                                   (useful when those are loaded separately with more history)
        """
        if '1min' not in self.buffers or len(self.buffers['1min']) == 0:
            raise ValueError("No 1-min data to resample from")

        df = self.buffers['1min']

        # Resample rules - intraday only
        resample_rules = {
            '5min': '5min',
            '15min': '15min',
            '30min': '30min',
            '1hour': '1h',
        }

        # Optionally include daily and longer (only if not loaded separately)
        if not skip_daily_and_longer:
            resample_rules.update({
                'daily': '1D',
                'weekly': '1W',
                'monthly': '1ME',
            })

        for interval, rule in resample_rules.items():
            # Don't overwrite if we already have more data
            existing = self.buffers.get(interval)
            resampled = self._resample_ohlcv(df, rule)

            if len(resampled) > 0:
                # Only overwrite if resampled has MORE data than existing
                if existing is None or len(existing) == 0 or len(resampled) > len(existing):
                    self.buffers[interval] = resampled
                    self.last_update[interval] = resampled.index[-1]
                    print(f"   Resampled to {interval}: {len(resampled):,} bars")
                else:
                    print(f"   Keeping existing {interval}: {len(existing):,} bars (more than resampled {len(resampled)})")

    def _resample_ohlcv(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Resample OHLCV data to a different timeframe."""
        agg_dict = {}
        for symbol in ['spy', 'tsla']:
            agg_dict[f'{symbol}_open'] = 'first'
            agg_dict[f'{symbol}_high'] = 'max'
            agg_dict[f'{symbol}_low'] = 'min'
            agg_dict[f'{symbol}_close'] = 'last'
            agg_dict[f'{symbol}_volume'] = 'sum'

        return df.resample(rule).agg(agg_dict).dropna()

    def get_multi_res_data(self) -> Dict[str, pd.DataFrame]:
        """
        Get data in multi_res_data format for TradingFeatureExtractor.

        Returns:
            Dict mapping interval names (as expected by feature extractor) to DataFrames
        """
        return {
            '1min': self.buffers.get('1min', pd.DataFrame()),
            '5min': self.buffers.get('5min', pd.DataFrame()),
            '15min': self.buffers.get('15min', pd.DataFrame()),
            '30min': self.buffers.get('30min', pd.DataFrame()),
            '1hour': self.buffers.get('1hour', pd.DataFrame()),
            'daily': self.buffers.get('daily', pd.DataFrame()),
            'weekly': self.buffers.get('weekly', pd.DataFrame()),
            'monthly': self.buffers.get('monthly', pd.DataFrame()),
        }

    def get_status(self) -> Dict[str, Dict]:
        """Get buffer status for all intervals."""
        status = {}
        for interval, df in self.buffers.items():
            status[interval] = {
                'count': len(df),
                'required': self.REQUIRED_HISTORY.get(interval, 0),
                'ready': len(df) >= self.REQUIRED_HISTORY.get(interval, 0) * 0.5,
                'last_update': self.last_update.get(interval)
            }
        return status


class LiveFeatureExtractor:
    """
    Extracts features using the SAME pipeline as training.

    This ensures exact feature consistency between training and inference.
    Uses TradingFeatureExtractor in "live mode" with multi_res_data.
    """

    def __init__(
        self,
        tf_meta_path: str = None,
        vix_data: pd.DataFrame = None,
        fetch_live_vix: bool = True
    ):
        """
        Initialize the feature extractor.

        Args:
            tf_meta_path: Path to tf_meta_*.json (contains feature columns per TF)
            vix_data: Optional VIX historical data for regime features
            fetch_live_vix: Whether to fetch live VIX from yfinance (default: True)
        """
        # Load tf_meta to get exact feature columns
        self.tf_meta = self._load_tf_meta(tf_meta_path)
        self.sequence_lengths = self.tf_meta.get('sequence_lengths', TIMEFRAME_SEQUENCE_LENGTHS)
        self.feature_columns = self.tf_meta.get('timeframe_columns', {})

        # Initialize the trading feature extractor
        self.extractor = TradingFeatureExtractor()

        # Load VIX data (historical + live)
        self.vix_data = vix_data
        if self.vix_data is None:
            self._load_vix_with_live(fetch_live_vix)

    def _load_vix_with_live(self, fetch_live: bool = True) -> None:
        """
        Load VIX data: historical from CSV + optionally live from yfinance.

        The model needs VIX features (15 total) including:
        - vix_level, vix_regime, vix_spike
        - vix_percentile_20d, vix_percentile_252d
        - vix_tsla_corr_20d, vix_spy_corr_20d
        - etc.

        For accurate predictions, we need current VIX values.
        """
        # Load historical VIX from CSV
        historical_vix = None
        try:
            historical_vix = load_vix_data()
            print(f"   Loaded historical VIX: {len(historical_vix):,} days")
        except Exception as e:
            warnings.warn(f"Could not load historical VIX: {e}")

        # Fetch live VIX if requested
        live_vix = None
        if fetch_live:
            print("   Fetching live VIX from yfinance...")
            live_vix = fetch_live_vix(days=30)
            if len(live_vix) > 0:
                print(f"   Fetched live VIX: {len(live_vix)} days ({live_vix.index[-1].date()})")
            else:
                warnings.warn("Failed to fetch live VIX - using historical only")

        # Combine historical + live
        self.vix_data = combine_vix_data(historical_vix, live_vix)

        if self.vix_data is not None and len(self.vix_data) > 0:
            print(f"   VIX data ready: {self.vix_data.index[0].date()} to {self.vix_data.index[-1].date()}")

    def refresh_vix(self) -> None:
        """
        Refresh VIX data by fetching latest from yfinance.
        Call this periodically (e.g., daily) to keep VIX current.
        """
        print("   Refreshing VIX data...")
        live_vix = fetch_live_vix(days=30)
        if len(live_vix) > 0:
            self.vix_data = combine_vix_data(self.vix_data, live_vix)
            print(f"   VIX refreshed: latest date {self.vix_data.index[-1].date()}")
        else:
            warnings.warn("Failed to refresh VIX data")

    def _load_tf_meta(self, path: str = None) -> Dict:
        """Load tf_meta.json file."""
        if path is None:
            # Find the most recent tf_meta file
            pattern = 'data/feature_cache/tf_meta_*.json'
            files = sorted(file_glob(pattern))
            if not files:
                warnings.warn(f"No tf_meta files found at {pattern}")
                return {}
            path = files[-1]

        with open(path) as f:
            meta = json.load(f)

        print(f"   Loaded tf_meta: {Path(path).name}")
        print(f"   Feature version: {meta.get('feature_version', 'unknown')}")

        return meta

    def extract_features(
        self,
        data_buffer: LiveDataBuffer,
        target_tf: str = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features for all (or specific) timeframes.

        This uses the SAME extraction pipeline as training by:
        1. Passing multi_res_data to TradingFeatureExtractor (triggers "live mode")
        2. Getting the full feature DataFrame
        3. Resampling to each TF
        4. Selecting the exact columns used in training
        5. Taking the last seq_len bars

        Args:
            data_buffer: LiveDataBuffer with SPY+TSLA data
            target_tf: Optional specific TF to extract (None = all TFs)

        Returns:
            Dict mapping TF name -> feature tensor [1, seq_len, num_features]
        """
        # Get multi-resolution data
        multi_res = data_buffer.get_multi_res_data()

        # Check we have enough data - find first non-empty DataFrame
        base_df = None
        for key in ['1min', '5min', 'daily']:
            candidate = multi_res.get(key)
            if candidate is not None and len(candidate) > 0:
                base_df = candidate
                break

        if base_df is None:
            raise ValueError("No data in buffers")

        # Prepare the DataFrame for feature extraction
        # Use the 1-min (or finest available) as base
        df = base_df.copy()
        df.attrs['multi_resolution'] = multi_res

        # Extract features using the SAME pipeline as training
        # use_cache=False forces fresh extraction (no mmap)
        # skip_native_tf_generation=True prevents creating new tf_meta files
        print("   Extracting features in LIVE mode...")
        extraction_result = self.extractor.extract_features(
            df,
            use_cache=False,
            use_gpu=False,  # CPU for stability in live
            continuation=False,
            use_chunking=False,
            skip_native_tf_generation=True,  # Don't create new tf_meta files!
            vix_data=self.vix_data
        )

        # Handle variable return values (2 or 3 depending on cache state)
        if isinstance(extraction_result, tuple):
            features_df = extraction_result[0]
        else:
            features_df = extraction_result

        # Now we have full features at 1-min resolution
        # Resample to each TF and select correct columns
        tf_features = {}
        timeframes = [target_tf] if target_tf else HIERARCHICAL_TIMEFRAMES

        for tf in timeframes:
            try:
                tf_tensor = self._extract_for_tf(features_df, tf)
                tf_features[tf] = tf_tensor
            except Exception as e:
                warnings.warn(f"Feature extraction failed for {tf}: {e}")
                # Create zero tensor as fallback
                seq_len = self.sequence_lengths.get(tf, 100)
                num_features = 1048 if tf in ['monthly', '3month'] else 1104
                tf_features[tf] = torch.zeros(1, seq_len, num_features)

        return tf_features

    def _extract_for_tf(self, features_df: pd.DataFrame, tf: str) -> torch.Tensor:
        """
        Extract features for a specific timeframe.

        Args:
            features_df: Full features DataFrame at 1-min resolution
            tf: Target timeframe

        Returns:
            Tensor of shape [1, seq_len, num_features]
        """
        # Get resample rule
        resample_rule = TIMEFRAME_RESAMPLE_RULES.get(tf, '1D')

        # Resample features to this TF
        resampled = features_df.resample(resample_rule).last().dropna()

        # Get the columns for this TF (from tf_meta)
        if tf in self.feature_columns:
            columns = self.feature_columns[tf]
        else:
            # Fallback: use all available columns
            columns = list(resampled.columns)

        # Select only the columns we need (in exact order)
        available = [c for c in columns if c in resampled.columns]
        if len(available) < len(columns):
            missing = set(columns) - set(available)
            warnings.warn(f"{tf}: Missing {len(missing)} columns (will use zeros)")

            # Add missing columns as zeros (efficient concat instead of loop)
            missing_cols = [c for c in columns if c not in resampled.columns]
            if missing_cols:
                missing_df = pd.DataFrame(0.0, index=resampled.index, columns=missing_cols)
                resampled = pd.concat([resampled, missing_df], axis=1)

        # Select columns in exact order
        tf_features = resampled[columns].values

        # Get sequence length
        seq_len = self.sequence_lengths.get(tf, len(tf_features))

        # Take last seq_len rows
        if len(tf_features) >= seq_len:
            sequence = tf_features[-seq_len:]
        else:
            # Pad with zeros at the beginning
            pad_size = seq_len - len(tf_features)
            padding = np.zeros((pad_size, tf_features.shape[1]))
            sequence = np.vstack([padding, tf_features])

        # Convert to tensor [1, seq_len, features]
        tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

        return tensor


class LivePredictor:
    """
    Main prediction class that coordinates data buffers, feature extraction,
    and model inference.

    Usage:
        predictor = LivePredictor('models/hierarchical_lnn.pth')

        # Load historical data
        predictor.load_historical_data(spy_df, tsla_df)

        # Or update with new bars
        predictor.update_bar(spy_bar, tsla_bar)

        # Make prediction
        result = predictor.predict()
    """

    def __init__(
        self,
        model_path: str,
        tf_meta_path: str = None,
        device: str = 'auto'
    ):
        """
        Initialize the live predictor.

        Args:
            model_path: Path to trained model checkpoint (.pth)
            tf_meta_path: Path to tf_meta_*.json (auto-detected if None)
            device: Device for inference ('auto', 'cpu', 'cuda', 'mps')
        """
        # Determine device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        print(f"🔮 Initializing LivePredictor (device: {self.device})")

        # Load the model
        print(f"   Loading model: {model_path}")
        self.model = load_hierarchical_model(model_path, device=self.device)
        self.model.eval()

        # Initialize data buffer
        self.data_buffer = LiveDataBuffer()

        # Initialize feature extractor
        self.feature_extractor = LiveFeatureExtractor(tf_meta_path)

        # Store metadata
        self.sequence_lengths = self.feature_extractor.sequence_lengths
        self.model_path = model_path

        # Cache for extracted features (prevents re-extraction on every prediction)
        self._cached_features = None
        self._cache_invalidated = True  # Flag to know when to re-extract

        print(f"   ✓ Ready for inference")
        print(f"   Sequence lengths: {self.sequence_lengths}")

    def load_historical_data(
        self,
        df: pd.DataFrame,
        interval: str = '1min',
        resample: bool = True
    ) -> None:
        """
        Load historical OHLCV data.

        Args:
            df: DataFrame with spy_* and tsla_* columns
            interval: Data interval
            resample: Whether to resample to all timeframes (default: True)
        """
        print(f"\n📊 Loading historical data...")
        self.data_buffer.load_from_dataframe(df, interval)

        if resample and interval == '1min':
            print("   Resampling to all timeframes...")
            self.data_buffer.resample_from_1min()

        # Invalidate cache when data is loaded
        self._cache_invalidated = True

    def load_separate_data(
        self,
        spy_df: pd.DataFrame,
        tsla_df: pd.DataFrame,
        interval: str = '1min'
    ) -> None:
        """
        Load separate SPY and TSLA DataFrames.

        Args:
            spy_df: SPY OHLCV DataFrame with columns: open, high, low, close, volume
            tsla_df: TSLA OHLCV DataFrame with same structure
            interval: Data interval
        """
        # Align and combine
        spy_renamed = spy_df.rename(columns={
            'open': 'spy_open', 'high': 'spy_high', 'low': 'spy_low',
            'close': 'spy_close', 'volume': 'spy_volume'
        })
        tsla_renamed = tsla_df.rename(columns={
            'open': 'tsla_open', 'high': 'tsla_high', 'low': 'tsla_low',
            'close': 'tsla_close', 'volume': 'tsla_volume'
        })

        # Inner join on index (only keep bars where both exist)
        combined = spy_renamed.join(tsla_renamed, how='inner')

        self.load_historical_data(combined, interval)

    def update_bar(
        self,
        spy_bar: Dict,
        tsla_bar: Dict,
        interval: str = '1min'
    ) -> None:
        """
        Update with a new bar for both SPY and TSLA.

        Args:
            spy_bar: SPY bar {timestamp, open, high, low, close, volume}
            tsla_bar: TSLA bar {timestamp, open, high, low, close, volume}
            interval: Data interval
        """
        # Combine into single bar
        combined_bar = {
            'timestamp': spy_bar['timestamp'],
            'spy_open': spy_bar['open'],
            'spy_high': spy_bar['high'],
            'spy_low': spy_bar['low'],
            'spy_close': spy_bar['close'],
            'spy_volume': spy_bar['volume'],
            'tsla_open': tsla_bar['open'],
            'tsla_high': tsla_bar['high'],
            'tsla_low': tsla_bar['low'],
            'tsla_close': tsla_bar['close'],
            'tsla_volume': tsla_bar['volume'],
        }

        self.data_buffer.update_bar(combined_bar, interval)

        # Invalidate cache when data is updated
        self._cache_invalidated = True

    def fetch_live_data(
        self,
        intraday_days: int = 60,
        daily_days: int = 400,
        longer_days: int = 5475,  # ~15 years for weekly/monthly (ensures w168 coverage)
        refresh_vix: bool = True
    ) -> None:
        """
        Fetch live SPY, TSLA, and VIX data from yfinance using NATIVE intervals.

        This fetches each timeframe directly from yfinance (no resampling needed)
        to get maximum data availability:
        - Intraday (5m, 15m, 30m, 1h): Up to 60 days from yfinance
        - Daily: ~400 days (~1.5 years)
        - Weekly/Monthly: ~15 years (ensures w168 window coverage)

        Fetches:
        - Native intervals: 5m, 15m, 30m, 1h, 1d, 1wk, 1mo, 3mo
        - Resampled: 2h, 3h, 4h (from 1h, since yfinance doesn't have these)
        - VIX daily data

        Args:
            intraday_days: Days of intraday data (default: 60, max for yfinance)
            daily_days: Days of daily data (default: 400, ~1.5 years)
            longer_days: Days for weekly/monthly (default: 5475 = 15 years)
            refresh_vix: Also refresh VIX data (default: True)

        Example:
            predictor = LivePredictor('models/hierarchical_lnn.pth')
            predictor.fetch_live_data()
            result = predictor.predict()
        """
        print(f"\n📡 Fetching live data from yfinance (native intervals)...")

        # Invalidate cached features since we're loading new data
        self._cache_invalidated = True

        # Map our TFs to yfinance intervals with appropriate history lengths
        native_intervals = {
            '5min': ('5m', min(intraday_days, 60)),     # Intraday limit
            '15min': ('15m', min(intraday_days, 60)),
            '30min': ('30m', min(intraday_days, 60)),
            '1hour': ('1h', min(intraday_days, 60)),
            'daily': ('1d', daily_days),
            'weekly': ('1wk', longer_days),             # Need ~15 years for w168
            'monthly': ('1mo', longer_days),
            '3month': ('3mo', longer_days),
        }

        # Fetch each native interval
        for our_tf, (yf_interval, days) in native_intervals.items():
            print(f"\n   --- {our_tf.upper()} ({yf_interval}, {days} days) ---")

            # Fetch from yfinance
            data = fetch_live_ohlcv(['SPY', 'TSLA'], days=days, interval=yf_interval)

            if 'SPY' not in data or 'TSLA' not in data:
                print(f"   ⚠️ Failed to fetch {our_tf}, will try resampling later")
                continue

            # Align and load
            combined = align_spy_tsla(data['SPY'], data['TSLA'])
            self.data_buffer.load_from_dataframe(combined, interval=our_tf)

        # Resample 2h, 3h, 4h from 1h (yfinance doesn't have these natively)
        print(f"\n   --- Resampling 2h, 3h, 4h from 1h ---")
        self._resample_hourly_to_multihour()

        # Refresh VIX if requested
        if refresh_vix:
            self.refresh_vix()

        print(f"\n✓ Live data ready!")
        self._print_buffer_summary()

    def _resample_hourly_to_multihour(self) -> None:
        """Resample 1h data to 2h, 3h, 4h (yfinance doesn't have these)."""
        hourly_df = self.data_buffer.buffers.get('1hour')
        if hourly_df is None or len(hourly_df) == 0:
            print("   ⚠️ No 1h data to resample from")
            return

        # Resample to 2h, 3h, 4h
        for interval, rule in [('2h', '2h'), ('3h', '3h'), ('4h', '4h')]:
            resampled = self.data_buffer._resample_ohlcv(hourly_df, rule)
            if len(resampled) > 0:
                self.data_buffer.buffers[interval] = resampled
                print(f"   Resampled to {interval}: {len(resampled):,} bars")

    def _resample_daily_to_longer(self) -> None:
        """Resample daily data to weekly and monthly."""
        daily_df = self.data_buffer.buffers.get('daily')
        if daily_df is None or len(daily_df) == 0:
            return

        # Resample to weekly
        weekly = self.data_buffer._resample_ohlcv(daily_df, '1W')
        if len(weekly) > 0:
            self.data_buffer.buffers['weekly'] = weekly
            print(f"   Resampled to weekly: {len(weekly):,} bars")

        # Resample to monthly
        monthly = self.data_buffer._resample_ohlcv(daily_df, '1ME')
        if len(monthly) > 0:
            self.data_buffer.buffers['monthly'] = monthly
            print(f"   Resampled to monthly: {len(monthly):,} bars")

    def _print_buffer_summary(self) -> None:
        """Print summary of all data buffers."""
        print("\n   Buffer Status:")
        for interval, df in self.data_buffer.buffers.items():
            if len(df) > 0:
                status = "✓" if len(df) >= self.data_buffer.REQUIRED_HISTORY.get(interval, 0) * 0.3 else "⚠"
                print(f"   {status} {interval:8s}: {len(df):6,} bars")

    @torch.no_grad()
    def predict(self) -> Dict[str, Any]:
        """
        Make a prediction using current data.

        Returns:
            Dict with prediction results:
            - predicted_high: Predicted high % move (from best TF)
            - predicted_low: Predicted low % move (from best TF)
            - confidence: Model confidence
            - selected_tf: Which timeframe was selected as most confident
            - all_channels: All 11 TF predictions sorted by confidence
            - timestamp: Prediction timestamp
            - buffer_status: Status of data buffers
        """
        # Check buffer status
        buffer_status = self.data_buffer.get_status()

        # Extract features (or use cached if data hasn't changed)
        if self._cache_invalidated or self._cached_features is None:
            print("   🔄 Extracting features (data changed)...")
            features = self.feature_extractor.extract_features(self.data_buffer)
            self._cached_features = features
            self._cache_invalidated = False
        else:
            features = self._cached_features

        # Move features to device
        features_device = {
            tf: tensor.to(self.device)
            for tf, tensor in features.items()
        }

        # v5.2: Load VIX sequence for inference
        vix_sequence = None
        events = None

        try:
            from src.ml.live_events import VIXSequenceLoader, LiveEventFetcher
            from datetime import date

            # Load VIX sequence (90 days)
            vix_loader = VIXSequenceLoader('data/VIX_History.csv')
            vix_seq = vix_loader.get_sequence(as_of_date=date.today(), sequence_length=90)
            vix_sequence = torch.tensor(vix_seq, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Fetch upcoming events
            event_fetcher = LiveEventFetcher()
            events = event_fetcher.fetch_upcoming_events()
        except Exception as e:
            print(f"   ⚠️  v5.2 VIX/events not available: {e}")

        # Run inference
        predictions, output_dict = self.model(features_device, vix_sequence=vix_sequence, events=events)

        # Extract primary results (from selected best TF)
        pred_high = predictions[0, 0].item()
        pred_low = predictions[0, 1].item()
        pred_conf = predictions[0, 2].item() if predictions.shape[1] > 2 else None

        # Build result
        result = {
            'predicted_high': pred_high,
            'predicted_low': pred_low,
            'confidence': pred_conf,
            'timestamp': datetime.now(),
            'buffer_status': buffer_status,
            'device': self.device,
        }

        # Extract channel selection info (Physics-Only mode)
        if 'channel_selection' in output_dict:
            selection = output_dict['channel_selection']
            result['selected_tf'] = selection['best_tf_name'][0]  # First sample in batch

            # Build all channels list sorted by confidence
            timeframes = self.model.TIMEFRAMES
            per_tf_highs = selection['per_tf_highs'][0].cpu().numpy()  # [11]
            per_tf_lows = selection['per_tf_lows'][0].cpu().numpy()    # [11]
            per_tf_confs = selection['per_tf_confs'][0].cpu().numpy()  # [11]

            all_channels = []
            for i, tf in enumerate(timeframes):
                all_channels.append({
                    'timeframe': tf,
                    'high': float(per_tf_highs[i]),
                    'low': float(per_tf_lows[i]),
                    'confidence': float(per_tf_confs[i]),
                })

            # Sort by confidence descending
            all_channels.sort(key=lambda x: x['confidence'], reverse=True)
            result['all_channels'] = all_channels

        # Also include per-TF predictions from layer_predictions
        if 'layer_predictions' in output_dict:
            result['layer_predictions'] = {
                tf: {
                    'high': preds[0, 0].item(),
                    'low': preds[0, 1].item(),
                    'confidence': preds[0, 2].item(),
                }
                for tf, preds in output_dict['layer_predictions'].items()
            }

        # v5.2: Add duration predictions
        if 'duration' in output_dict:
            result['v52_duration'] = {}
            for tf, dur_data in output_dict['duration'].items():
                result['v52_duration'][tf] = {
                    'expected': dur_data['expected'][0, 0].item(),
                    'conservative': dur_data['conservative'][0, 0].item(),
                    'aggressive': dur_data['aggressive'][0, 0].item(),
                    'confidence': dur_data['confidence'][0, 0].item(),
                }

        # v5.2: Add validity predictions
        if 'validity' in output_dict:
            result['v52_validity'] = {
                tf: val[0, 0].item()
                for tf, val in output_dict['validity'].items()
            }

        # v5.2: Add compositor predictions
        if 'compositor' in output_dict:
            compositor = output_dict['compositor']
            trans_probs = compositor['transition_probs'][0].cpu().numpy()
            dir_probs = compositor['direction_probs'][0].cpu().numpy()
            tf_switch_probs = compositor['tf_switch_probs'][0].cpu().numpy()

            result['v52_compositor'] = {
                'transition': {
                    'continue': float(trans_probs[0]),
                    'switch_tf': float(trans_probs[1]),
                    'reverse': float(trans_probs[2]),
                    'sideways': float(trans_probs[3]),
                },
                'direction': {
                    'bull': float(dir_probs[0]),
                    'bear': float(dir_probs[1]),
                    'sideways': float(dir_probs[2]),
                },
                'tf_switch_probs': {tf: float(tf_switch_probs[i]) for i, tf in enumerate(self.model.TIMEFRAMES)},
                'phase2_slope': compositor['phase2_slope'][0, 0].item(),
            }

        # v5.2: Add events info
        if events:
            result['v52_events'] = events

        return result

    def get_buffer_status(self) -> Dict[str, Dict]:
        """Get status of all data buffers."""
        return self.data_buffer.get_status()

    def refresh_vix(self) -> None:
        """
        Refresh VIX data by fetching latest from yfinance.

        Call this periodically (e.g., at start of each trading day)
        to ensure the model has current VIX values.

        The model uses 15 VIX features including vix_level, vix_regime,
        vix_spike, etc. - these need current data for accurate predictions.
        """
        self.feature_extractor.refresh_vix()

    def get_vix_status(self) -> Dict[str, Any]:
        """Get current VIX data status."""
        vix = self.feature_extractor.vix_data
        if vix is None or len(vix) == 0:
            return {'loaded': False, 'latest_date': None, 'latest_value': None}

        return {
            'loaded': True,
            'total_days': len(vix),
            'latest_date': vix.index[-1].date(),
            'latest_value': vix['vix_close'].iloc[-1],
            'date_range': f"{vix.index[0].date()} to {vix.index[-1].date()}"
        }


def load_sample_data() -> pd.DataFrame:
    """
    Load sample data for testing from existing training data.

    Returns:
        DataFrame with SPY and TSLA OHLCV data
    """
    # Try to load from existing parquet files
    data_dir = Path('data')

    # Look for aligned data
    aligned_file = data_dir / 'aligned_spy_tsla.parquet'
    if aligned_file.exists():
        print(f"   Loading from {aligned_file}")
        df = pd.read_parquet(aligned_file)
        return df

    # Try pickle
    aligned_pkl = data_dir / 'aligned_spy_tsla.pkl'
    if aligned_pkl.exists():
        print(f"   Loading from {aligned_pkl}")
        df = pd.read_pickle(aligned_pkl)
        return df

    raise FileNotFoundError(
        "No sample data found. Please provide SPY/TSLA data "
        "or run data collection first."
    )


def main():
    """Test the live prediction system."""
    import argparse

    parser = argparse.ArgumentParser(description='Live Prediction System')
    parser.add_argument('--model', type=str, default='models/hierarchical_lnn.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto/cpu/cuda/mps)')
    parser.add_argument('--test', action='store_true',
                       help='Run test prediction with sample data')
    args = parser.parse_args()

    print("=" * 60)
    print("LIVE PREDICTION SYSTEM v5.0")
    print("=" * 60)

    # Initialize predictor
    predictor = LivePredictor(
        model_path=args.model,
        device=args.device
    )

    if args.test:
        print("\n📊 Loading sample data for testing...")
        try:
            sample_data = load_sample_data()

            # Use last 50,000 bars (or all if less)
            n_bars = min(50000, len(sample_data))
            test_data = sample_data.tail(n_bars)

            predictor.load_historical_data(test_data, interval='1min')

            print("\n🔮 Making prediction...")
            result = predictor.predict()

            print("\n" + "=" * 40)
            print("PREDICTION RESULTS")
            print("=" * 40)
            print(f"  Predicted High: {result['predicted_high']:.4f}")
            print(f"  Predicted Low:  {result['predicted_low']:.4f}")
            print(f"  Timestamp:      {result['timestamp']}")
            print(f"  Device:         {result['device']}")

            print("\n📈 Buffer Status:")
            for interval, status in result['buffer_status'].items():
                ready = "✓" if status['ready'] else "✗"
                print(f"  {interval:8s}: {status['count']:6d} bars [{ready}]")

        except FileNotFoundError as e:
            print(f"\n❌ {e}")
            print("\nTo test, you need historical SPY/TSLA data.")
            print("Options:")
            print("  1. Run data collection: python collect_data.py")
            print("  2. Provide your own data via API")

    else:
        print("\n📈 Ready for live data.")
        print("Use predictor.update_bar(spy_bar, tsla_bar) to add new bars")
        print("Use predictor.predict() to get predictions")


if __name__ == '__main__':
    main()
