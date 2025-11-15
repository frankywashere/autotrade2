"""
Feature extraction system for ML model
Leverages existing Stage 1 components (channels, RSI) plus new features
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple
from pathlib import Path
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

import config
from src.linear_regression import LinearRegressionChannel
from src.rsi_calculator import RSICalculator
from .base import FeatureExtractor
from .channel_features import ChannelFeatureExtractor

# Feature cache version - increment when calculation logic changes
FEATURE_VERSION = "v3.5"


class TradingFeatureExtractor(FeatureExtractor):
    """
    Comprehensive feature extractor for stock trading
    Combines technical indicators, channel patterns, correlations, and cycles
    """

    def __init__(self):
        self.channel_calc = LinearRegressionChannel()
        self.rsi_calc = RSICalculator()
        self.channel_features_calc = ChannelFeatureExtractor()
        self.feature_names = []
        self._build_feature_names()

    def _build_feature_names(self):
        """Build list of all feature names"""
        features = []

        # Price features (both SPY and TSLA)
        for symbol in ['spy', 'tsla']:
            features.extend([
                f'{symbol}_close',
                f'{symbol}_returns',
                f'{symbol}_log_returns',
                f'{symbol}_volatility_10',
                f'{symbol}_volatility_50',
            ])

        # TSLA Channel features (multi-timeframe) - RENAMED for consistency
        for tf in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
            features.extend([
                f'tsla_channel_{tf}_position',  # 0-1 position in channel
                f'tsla_channel_{tf}_upper_dist',  # Distance to upper
                f'tsla_channel_{tf}_lower_dist',  # Distance to lower
                f'tsla_channel_{tf}_slope',
                f'tsla_channel_{tf}_stability',
                f'tsla_channel_{tf}_ping_pongs',
                f'tsla_channel_{tf}_r_squared',
            ])

        # SPY Channel features (multi-timeframe) - NEW in v3.4
        for tf in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
            features.extend([
                f'spy_channel_{tf}_position',  # 0-1 position in channel
                f'spy_channel_{tf}_upper_dist',  # Distance to upper
                f'spy_channel_{tf}_lower_dist',  # Distance to lower
                f'spy_channel_{tf}_slope',
                f'spy_channel_{tf}_stability',
                f'spy_channel_{tf}_ping_pongs',
                f'spy_channel_{tf}_r_squared',
            ])

        # TSLA RSI features (multi-timeframe) - RENAMED for consistency
        for tf in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
            features.extend([
                f'tsla_rsi_{tf}',
                f'tsla_rsi_{tf}_oversold',  # Binary
                f'tsla_rsi_{tf}_overbought',  # Binary
            ])

        # SPY RSI features (multi-timeframe) - NEW in v3.4
        for tf in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
            features.extend([
                f'spy_rsi_{tf}',
                f'spy_rsi_{tf}_oversold',  # Binary
                f'spy_rsi_{tf}_overbought',  # Binary
            ])

        # SPY-TSLA correlation features
        features.extend([
            'correlation_10',  # 10-bar rolling correlation
            'correlation_50',
            'correlation_200',
            'divergence',  # SPY up, TSLA down (or vice versa)
            'divergence_magnitude',
        ])

        # Larger cycle features
        features.extend([
            'distance_from_52w_high',
            'distance_from_52w_low',
            'within_mega_channel',  # Binary: in 3-4 year channel
            'mega_channel_position',
        ])

        # Volume features
        features.extend([
            'tsla_volume_ratio',  # Current / average
            'spy_volume_ratio',
        ])

        # Time features
        features.extend([
            'hour_of_day',
            'day_of_week',
            'day_of_month',
            'month_of_year',
        ])

        # Breakdown indicator features (NEW for hierarchical model)
        features.append('tsla_volume_surge')
        for tf in ['15min', '1h', '4h', 'daily']:
            features.append(f'tsla_rsi_divergence_{tf}')
        for tf in ['1h', '4h', 'daily']:
            features.append(f'tsla_channel_duration_ratio_{tf}')
        for tf in ['1h', '4h']:
            features.append(f'channel_alignment_spy_tsla_{tf}')

        # Time-in-channel features (additional breakdown indicators)
        for tf in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
            features.append(f'tsla_time_in_channel_{tf}')
            features.append(f'spy_time_in_channel_{tf}')

        # Enhanced channel position features (normalized -1 to +1)
        for tf in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
            features.append(f'tsla_channel_position_norm_{tf}')
            features.append(f'spy_channel_position_norm_{tf}')

        # Binary feature flags (Phase 4)
        features.extend(['is_monday', 'is_friday', 'is_volatile_now', 'is_earnings_week'])

        # In-channel binary flags
        for tf in ['1h', '4h', 'daily']:
            features.append(f'tsla_in_channel_{tf}')
            features.append(f'spy_in_channel_{tf}')

        self.feature_names = features

    def get_feature_names(self) -> List[str]:
        """Return list of feature names"""
        return self.feature_names

    def get_feature_dim(self) -> int:
        """Return total number of features"""
        return len(self.feature_names)

    def extract_features(self, df: pd.DataFrame, use_cache: bool = True, **kwargs) -> pd.DataFrame:
        """
        Extract all 313 features from aligned SPY-TSLA data (v3.5 - Hierarchical Multi-Task).

        Args:
            df: DataFrame with SPY and TSLA OHLCV columns
            use_cache: Whether to use cached rolling channels (default: True). Set to False to force regeneration.
            **kwargs: Additional arguments (reserved for future use)

        df should have columns: spy_open, spy_high, spy_low, spy_close, spy_volume,
                                tsla_open, tsla_high, tsla_low, tsla_close, tsla_volume

        Returns DataFrame with 313 columns:
        - 10 price features
        - 154 channel features (77 TSLA + 77 SPY)
        - 66 RSI features (33 TSLA + 33 SPY)
        - 5 correlation features
        - 4 cycle features
        - 2 volume features
        - 4 time features
        - 54 breakdown/channel enhancement features
        - 14 binary feature flags (NO LEAKAGE - is_monday, is_friday, is_volatile_now, is_earnings_week, in_channel flags)
        """
        from tqdm import tqdm

        # Extract multi-resolution data if present (for live mode) and remove from attrs to prevent deep copy recursion
        multi_res_data = df.attrs.pop('multi_resolution', None)

        # PASS 1: Extract base features
        print("   Extracting base features...")
        with tqdm(total=7, desc="   Feature extraction", leave=True, position=0, ncols=100) as pbar:
            price_df = self._extract_price_features(df)
            pbar.update(1)

            channel_df = self._extract_channel_features(df, multi_res_data=multi_res_data, use_cache=use_cache)
            pbar.update(1)

            rsi_df = self._extract_rsi_features(df, multi_res_data=multi_res_data)
            pbar.update(1)

            correlation_df = self._extract_correlation_features(df)
            pbar.update(1)

            cycle_df = self._extract_cycle_features(df)
            pbar.update(1)

            volume_df = self._extract_volume_features(df)
            pbar.update(1)

            time_df = self._extract_time_features(df)
            pbar.update(1)

        # Concat base features FIRST
        base_features_df = pd.concat([
            price_df,
            channel_df,
            rsi_df,
            correlation_df,
            cycle_df,
            volume_df,
            time_df
        ], axis=1)

        # PASS 2: Extract breakdown features (needs base features)
        with tqdm(total=1, desc="   Breakdown features", leave=False, ncols=100) as pbar:
            breakdown_df = self._extract_breakdown_features(base_features_df, df)
            pbar.update(1)

        # Final concat
        features_df = pd.concat([base_features_df, breakdown_df], axis=1)

        # Fill NaNs
        features_df = features_df.bfill().fillna(0)

        print(f"   ✓ Extracted {len(features_df.columns)} features")
        return features_df

    def _extract_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic price features. Returns DataFrame with 10 columns."""
        price_features = {}

        for symbol in ['spy', 'tsla']:
            close_col = f'{symbol}_close'
            price_features[close_col] = df[close_col]

            # Returns
            returns = df[close_col].pct_change()
            price_features[f'{symbol}_returns'] = returns
            price_features[f'{symbol}_log_returns'] = np.log(df[close_col] / df[close_col].shift(1))

            # Volatility (rolling std of returns)
            price_features[f'{symbol}_volatility_10'] = returns.rolling(10).std()
            price_features[f'{symbol}_volatility_50'] = returns.rolling(50).std()

        return pd.DataFrame(price_features, index=df.index)

    def _extract_channel_features(self, df: pd.DataFrame, use_cache: bool = True, multi_res_data: dict = None) -> pd.DataFrame:
        """
        Extract ROLLING linear regression channel features for multiple timeframes.

        CRITICAL: Channels are calculated at EACH timestamp using a rolling lookback window.
        This captures channel dynamics (formation, strength, breakdown) over time.

        NOW PROCESSES BOTH TSLA AND SPY (v3.4)
        Returns DataFrame with 154 columns (77 TSLA + 77 SPY).

        Args:
            df: OHLCV DataFrame
            use_cache: If True, load from cache or save to cache (recommended)
        """
        from tqdm import tqdm
        import hashlib
        import pickle

        # Check cache first
        if use_cache:
            cache_dir = Path('data/feature_cache')
            cache_dir.mkdir(exist_ok=True)

            # Create cache key from version + data range (version ensures cache invalidation when logic changes)
            cache_key = f"{FEATURE_VERSION}_{df.index[0].strftime('%Y%m%d')}_{df.index[-1].strftime('%Y%m%d')}_{len(df)}"
            cache_file = cache_dir / f'rolling_channels_{cache_key}.pkl'

            if cache_file.exists():
                try:
                    # Validate cache file before loading
                    file_size = cache_file.stat().st_size

                    if file_size < 1000:  # Less than 1KB is too small
                        print(f"   ⚠️  Cache file too small ({file_size} bytes), regenerating...")
                        cache_file.unlink()
                    else:
                        # Load cache with progress bar
                        with tqdm(total=1, desc=f"   Loading cache", leave=False, position=1, ncols=100) as pbar:
                            with open(cache_file, 'rb') as f:
                                result = pickle.load(f)
                            pbar.update(1)

                        # Validate loaded data
                        expected_cols = 154  # 77 features × 2 stocks (TSLA + SPY)
                        if len(result.columns) != expected_cols:
                            print(f"   ⚠️  Cache has {len(result.columns)} columns, expected {expected_cols}")
                            print(f"   ⚠️  Regenerating cache...")
                        elif len(result) != len(df):
                            print(f"   ⚠️  Cache has {len(result)} rows, expected {len(df)}")
                            print(f"   ⚠️  Regenerating cache...")
                        else:
                            # Cache is valid!
                            print(f"   ✓ Loaded channel features from cache: {cache_file.name}")
                            return result

                except Exception as e:
                    print(f"   ⚠️  Cache load failed ({type(e).__name__}: {e}), regenerating...")
                    if cache_file.exists():
                        cache_file.unlink()

        # No cache - calculate rolling channels
        # Check if multi-resolution data was provided (live mode)
        is_live_mode = multi_res_data is not None

        channel_features = {}
        num_rows = len(df)

        # Resample to different timeframes
        timeframes = {
            '5min': '5min',
            '15min': '15min',
            '30min': '30min',
            '1h': '1h',
            '2h': '2h',
            '3h': '3h',
            '4h': '4h',
            'daily': '1D',
            'weekly': '1W',
            'monthly': '1ME',
            '3month': '3ME'
        }

        # Process both TSLA and SPY
        total_calcs = len(timeframes) * 2  # 11 timeframes × 2 stocks

        # Print status messages with detailed info
        if is_live_mode:
            print(f"   🔄 Extracting channel features in LIVE mode (using multi-resolution data)...")
            print(f"   📊 Processing {total_calcs} calculations (11 timeframes × 2 stocks: SPY + TSLA)")
        else:
            print(f"   🔄 Calculating ROLLING channels (this will take ~30-60 mins first time)...")
            print(f"   📊 Processing {total_calcs} calculations (11 timeframes × 2 stocks: SPY + TSLA)")
            print(f"   ⏱️  Estimated time: ~{total_calcs * 2.5:.0f} minutes")
            print(f"   💡 Results will be cached for instant loading next time")

        calc_progress = tqdm(total=total_calcs, desc="   Rolling channels (SPY + TSLA)", ncols=100, leave=False, position=1)

        for symbol in ['tsla', 'spy']:
            for tf_name, tf_rule in timeframes.items():
                # HYBRID DATA SELECTION: Use appropriate resolution for live mode
                if is_live_mode:
                    # Live mode: Get data from appropriate resolution
                    if tf_name in ['5min', '15min', '30min']:
                        # Use 1-min data (sufficient history)
                        source_data = multi_res_data['1min']
                    elif tf_name in ['1h', '2h', '3h', '4h']:
                        # Use hourly data (2 years of history)
                        source_data = multi_res_data['1hour']
                    else:  # daily, weekly, monthly, 3month
                        # Use daily data (max history)
                        source_data = multi_res_data['daily']

                    # Extract symbol columns
                    symbol_df = source_data[[c for c in source_data.columns if c.startswith(f'{symbol}_')]].copy()
                else:
                    # Training mode: Use input dataframe
                    symbol_df = df[[c for c in df.columns if c.startswith(f'{symbol}_')]].copy()

                symbol_df.columns = [c.replace(f'{symbol}_', '') for c in symbol_df.columns]

                # Resample to target timeframe
                resampled = symbol_df.resample(tf_rule).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()

                prefix = f'{symbol}_channel'

                if len(resampled) < 20:
                    # Not enough data - fill with zeros
                    for feat in ['position', 'upper_dist', 'lower_dist', 'slope', 'stability', 'ping_pongs', 'r_squared']:
                        channel_features[f'{prefix}_{tf_name}_{feat}'] = np.zeros(num_rows)
                    calc_progress.update(1)
                    continue

                # ROLLING CHANNEL CALCULATION
                lookback = min(168, len(resampled) // 2)  # Use half of data or 168, whichever is smaller

                rolling_results = self._calculate_rolling_channels(
                    resampled, lookback, tf_name, symbol, df.index
                )

                # Store rolling results (each row has unique values!)
                for feat_name, values in rolling_results.items():
                    channel_features[f'{prefix}_{tf_name}_{feat_name}'] = values

                calc_progress.update(1)

        calc_progress.close()

        result_df = pd.DataFrame(channel_features, index=df.index)

        # Save to cache (atomic write to prevent corruption)
        if use_cache:
            print(f"   💾 Saving to cache: {cache_file.name}")
            # Write to temp file first (atomic operation)
            temp_file = cache_file.with_suffix('.pkl.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(result_df, f)
            # Atomic rename (ensures no corruption even if interrupted)
            temp_file.rename(cache_file)
            print(f"   ✓ Cache saved successfully")

        return result_df

    def _calculate_rolling_channels(
        self,
        resampled_df: pd.DataFrame,
        lookback: int,
        tf_name: str,
        symbol: str,
        original_index: pd.DatetimeIndex
    ) -> dict:
        """
        Calculate channels at each timestamp using rolling window.

        Args:
            resampled_df: Resampled OHLCV data at target timeframe
            lookback: Number of bars to look back
            tf_name: Timeframe name
            symbol: 'tsla' or 'spy'
            original_index: Original 1-min index (for alignment)

        Returns:
            Dictionary with arrays for each channel feature
        """
        num_original_rows = len(original_index)

        # Initialize result arrays
        results = {
            'position': np.zeros(num_original_rows),
            'upper_dist': np.zeros(num_original_rows),
            'lower_dist': np.zeros(num_original_rows),
            'slope': np.zeros(num_original_rows),
            'stability': np.zeros(num_original_rows),
            'ping_pongs': np.zeros(num_original_rows),
            'r_squared': np.zeros(num_original_rows)
        }

        # Calculate channel at each timestamp
        for i in range(lookback, len(resampled_df)):
            try:
                # Rolling window
                window = resampled_df.iloc[i-lookback:i]

                # Calculate channel for this window
                channel = self.channel_calc.calculate_channel(window, lookback, tf_name)
                current_price = resampled_df['close'].iloc[i]
                position_data = self.channel_calc.get_channel_position(current_price, channel)

                # Map resampled timestamp to original 1-min index
                timestamp = resampled_df.index[i]

                # Find all 1-min bars that map to this resampled bar
                # (All 1-min bars between this timestamp and next)
                if i < len(resampled_df) - 1:
                    next_timestamp = resampled_df.index[i + 1]
                    mask = (original_index >= timestamp) & (original_index < next_timestamp)
                else:
                    # Last bar
                    mask = original_index >= timestamp

                # Assign channel metrics to all 1-min bars in this window
                results['position'][mask] = position_data['position']
                results['upper_dist'][mask] = position_data['distance_to_upper_pct']
                results['lower_dist'][mask] = position_data['distance_to_lower_pct']
                results['slope'][mask] = channel.slope
                results['stability'][mask] = channel.stability_score
                results['ping_pongs'][mask] = channel.ping_pongs
                results['r_squared'][mask] = channel.r_squared

            except Exception as e:
                # If calculation fails, leave as zeros
                continue

        return results

    def _extract_rsi_features(self, df: pd.DataFrame, multi_res_data: dict = None) -> pd.DataFrame:
        """
        Extract RSI features for multiple timeframes.
        NOW PROCESSES BOTH TSLA AND SPY (v3.4)
        Returns DataFrame with 66 columns (33 TSLA + 33 SPY).
        Supports HYBRID mode for live predictions (uses multi-resolution data).
        """
        rsi_features = {}
        num_rows = len(df)

        # Check if multi-resolution data was provided (live mode)
        is_live_mode = multi_res_data is not None

        timeframes = {
            '5min': '5min',
            '15min': '15min',
            '30min': '30min',
            '1h': '1h',
            '2h': '2h',
            '3h': '3h',
            '4h': '4h',
            'daily': '1D',
            'weekly': '1W',
            'monthly': '1ME',
            '3month': '3ME'
        }

        # Process both TSLA and SPY
        for symbol in ['tsla', 'spy']:
            for tf_name, tf_rule in timeframes.items():
                # HYBRID DATA SELECTION: Use appropriate resolution for live mode
                if is_live_mode:
                    # Live mode: Get data from appropriate resolution
                    if tf_name in ['5min', '15min', '30min']:
                        # Use 1-min data (sufficient history)
                        source_data = multi_res_data['1min']
                    elif tf_name in ['1h', '2h', '3h', '4h']:
                        # Use hourly data (2 years of history)
                        source_data = multi_res_data['1hour']
                    else:  # daily, weekly, monthly, 3month
                        # Use daily data (max history)
                        source_data = multi_res_data['daily']

                    # Extract symbol columns
                    symbol_df = source_data[[c for c in source_data.columns if c.startswith(f'{symbol}_')]].copy()
                else:
                    # Training mode: Use input dataframe
                    symbol_df = df[[c for c in df.columns if c.startswith(f'{symbol}_')]].copy()

                symbol_df.columns = [c.replace(f'{symbol}_', '') for c in symbol_df.columns]

                # Resample to target timeframe
                resampled = symbol_df.resample(tf_rule).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()

                prefix = f'{symbol}_rsi'

                if len(resampled) < 20:
                    # Not enough data - fill with defaults
                    rsi_features[f'{prefix}_{tf_name}'] = np.full(num_rows, 50.0)
                    rsi_features[f'{prefix}_{tf_name}_oversold'] = np.zeros(num_rows)
                    rsi_features[f'{prefix}_{tf_name}_overbought'] = np.zeros(num_rows)
                    continue

                try:
                    rsi_data = self.rsi_calc.get_rsi_data(resampled)
                    rsi_value = rsi_data.value if rsi_data.value is not None else 50.0

                    # Store features (broadcast scalar to all rows)
                    rsi_features[f'{prefix}_{tf_name}'] = np.full(num_rows, rsi_value)
                    rsi_features[f'{prefix}_{tf_name}_oversold'] = np.full(num_rows, 1.0 if rsi_data.oversold else 0.0)
                    rsi_features[f'{prefix}_{tf_name}_overbought'] = np.full(num_rows, 1.0 if rsi_data.overbought else 0.0)

                except Exception:
                    # Fill with defaults
                    rsi_features[f'{prefix}_{tf_name}'] = np.full(num_rows, 50.0)
                    rsi_features[f'{prefix}_{tf_name}_oversold'] = np.zeros(num_rows)
                    rsi_features[f'{prefix}_{tf_name}_overbought'] = np.zeros(num_rows)

        return pd.DataFrame(rsi_features, index=df.index)

    def _extract_correlation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract SPY-TSLA correlation and divergence features. Returns DataFrame with 5 columns."""
        spy_returns = df['spy_close'].pct_change()
        tsla_returns = df['tsla_close'].pct_change()

        correlation_features = {
            'correlation_10': spy_returns.rolling(10).corr(tsla_returns),
            'correlation_50': spy_returns.rolling(50).corr(tsla_returns),
            'correlation_200': spy_returns.rolling(200).corr(tsla_returns),
            'divergence': (((spy_returns > 0) & (tsla_returns < 0)) |
                          ((spy_returns < 0) & (tsla_returns > 0))).astype(float),
            'divergence_magnitude': abs(spy_returns - tsla_returns)
        }

        return pd.DataFrame(correlation_features, index=df.index)

    def _extract_cycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract larger cycle features (52-week highs/lows, mega channels). Returns DataFrame with 4 columns."""
        tsla_close = df['tsla_close']
        high_52w = tsla_close.rolling(252, min_periods=50).max()
        low_52w = tsla_close.rolling(252, min_periods=50).min()

        cycle_features = {
            'distance_from_52w_high': (high_52w - tsla_close) / high_52w,
            'distance_from_52w_low': (tsla_close - low_52w) / low_52w
        }

        # Mega channel: 3-4 year channels
        if len(df) > 756:  # 3 years of daily data (approx)
            lookback = min(1008, len(df))
            mega_high = tsla_close.rolling(lookback, min_periods=252).max()
            mega_low = tsla_close.rolling(lookback, min_periods=252).min()

            cycle_features['within_mega_channel'] = ((tsla_close >= mega_low * 0.9) &
                                                      (tsla_close <= mega_high * 1.1)).astype(float)
            cycle_features['mega_channel_position'] = (tsla_close - mega_low) / (mega_high - mega_low + 1e-8)
        else:
            cycle_features['within_mega_channel'] = np.zeros(len(df))
            cycle_features['mega_channel_position'] = np.full(len(df), 0.5)

        return pd.DataFrame(cycle_features, index=df.index)

    def _extract_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract volume-based features. Returns DataFrame with 2 columns."""
        volume_features = {
            'tsla_volume_ratio': df['tsla_volume'] / df['tsla_volume'].rolling(20, min_periods=1).mean(),
            'spy_volume_ratio': df['spy_volume'] / df['spy_volume'].rolling(20, min_periods=1).mean()
        }

        return pd.DataFrame(volume_features, index=df.index)

    def _extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features. Returns DataFrame with 4 columns."""
        time_features = {
            'hour_of_day': df.index.hour / 24.0,
            'day_of_week': df.index.dayofweek / 7.0,
            'day_of_month': df.index.day / 31.0,
            'month_of_year': df.index.month / 12.0
        }

        return pd.DataFrame(time_features, index=df.index)

    def _extract_breakdown_features(
        self,
        features_df: pd.DataFrame,  # Base features already extracted (has tsla_volatility_10, etc.)
        raw_df: pd.DataFrame        # Original OHLCV data (for index.dayofweek)
    ) -> pd.DataFrame:
        """
        Extract channel breakdown and enhancement features. Returns DataFrame with 68 columns.

        Args:
            features_df: Base features DataFrame (245 features already extracted)
            raw_df: Original OHLCV DataFrame (for timestamp index)

        Features:
        - Volume surge indicator
        - RSI divergence from channel position
        - Channel duration vs historical average
        - SPY-TSLA channel alignment
        - Time in channel (bars since last break)
        - Normalized channel position (-1 to +1)
        - Binary flags (day of week, volatility, in_channel)
        """
        breakdown_features = {}
        num_rows = len(features_df)

        # Extract necessary data from RAW df (OHLCV)
        tsla_prices = raw_df['tsla_close'].values
        spy_prices = raw_df['spy_close'].values
        tsla_volume = raw_df['tsla_volume'].values if 'tsla_volume' in raw_df.columns else np.zeros(num_rows)

        # 1. Volume surge (recent vs historical)
        if len(tsla_volume) >= 60:
            recent_vol = pd.Series(tsla_volume).rolling(10, min_periods=1).mean()
            historical_vol = pd.Series(tsla_volume).rolling(60, min_periods=10).mean().shift(10)
            volume_surge = ((recent_vol - historical_vol) / (historical_vol + 1e-8)).fillna(0)
            breakdown_features['tsla_volume_surge'] = volume_surge.values
        else:
            breakdown_features['tsla_volume_surge'] = np.zeros(num_rows)

        # 2. RSI divergence from channel position (4 timeframes)
        for tf_name in ['15min', '1h', '4h', 'daily']:
            # Get RSI value from already calculated base features
            rsi_col = f'tsla_rsi_{tf_name}'
            channel_pos_col = f'tsla_channel_{tf_name}_position'

            if rsi_col in features_df.columns and channel_pos_col in features_df.columns:
                rsi_normalized = features_df[rsi_col] / 100.0  # 0-1 range
                channel_pos = features_df[channel_pos_col]  # 0-1 range

                # Divergence: High RSI + low position = potential reversal
                divergence = rsi_normalized - channel_pos
                breakdown_features[f'tsla_rsi_divergence_{tf_name}'] = divergence.values
            else:
                breakdown_features[f'tsla_rsi_divergence_{tf_name}'] = np.zeros(num_rows)

        # 3. Channel duration vs historical average (3 timeframes)
        # Use stability score as proxy for channel duration
        for tf_name in ['1h', '4h', 'daily']:
            stability_col = f'tsla_channel_{tf_name}_stability'

            if stability_col in features_df.columns:
                stability = features_df[stability_col]
                # Duration ratio = current stability vs rolling average
                avg_stability = stability.rolling(50, min_periods=10).mean()
                duration_ratio = (stability / (avg_stability + 1e-8)).fillna(1.0)
                breakdown_features[f'tsla_channel_duration_ratio_{tf_name}'] = duration_ratio.values
            else:
                breakdown_features[f'tsla_channel_duration_ratio_{tf_name}'] = np.ones(num_rows)

        # 4. SPY-TSLA channel alignment (2 timeframes)
        for tf_name in ['1h', '4h']:
            tsla_pos_col = f'tsla_channel_{tf_name}_position'
            spy_pos_col = f'spy_channel_{tf_name}_position'

            if tsla_pos_col in features_df.columns and spy_pos_col in features_df.columns:
                tsla_pos = features_df[tsla_pos_col] * 2 - 1  # Convert 0-1 to -1 to +1
                spy_pos = features_df[spy_pos_col] * 2 - 1

                # Alignment: both at top (+1) or both at bottom (-1) = high alignment
                alignment = tsla_pos * spy_pos  # -1 to +1
                breakdown_features[f'channel_alignment_spy_tsla_{tf_name}'] = alignment.values
            else:
                breakdown_features[f'channel_alignment_spy_tsla_{tf_name}'] = np.zeros(num_rows)

        # 5. Time in channel features (11 timeframes × 2 stocks)
        # Use channel stability as proxy (higher stability = longer time in channel)
        for tf_name in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
            for symbol in ['tsla', 'spy']:
                stability_col = f'{symbol}_channel_{tf_name}_stability'

                if stability_col in features_df.columns:
                    # Normalize stability to represent "time in channel" score
                    stability = features_df[stability_col]
                    time_in_channel = np.clip(stability * 100, 0, 100)  # 0-100 scale
                    breakdown_features[f'{symbol}_time_in_channel_{tf_name}'] = time_in_channel.values
                else:
                    breakdown_features[f'{symbol}_time_in_channel_{tf_name}'] = np.zeros(num_rows)

        # 6. Enhanced normalized channel position (11 timeframes × 2 stocks)
        for tf_name in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
            for symbol in ['tsla', 'spy']:
                pos_col = f'{symbol}_channel_{tf_name}_position'

                if pos_col in features_df.columns:
                    # Convert 0-1 position to -1 to +1 (bottom to top)
                    position_norm = features_df[pos_col] * 2 - 1
                    breakdown_features[f'{symbol}_channel_position_norm_{tf_name}'] = position_norm.values
                else:
                    breakdown_features[f'{symbol}_channel_position_norm_{tf_name}'] = np.zeros(num_rows)

        # 7. Binary feature flags (NO LEAKAGE - past data only)

        # Day of week flags (known at prediction time - OK) - Use raw_df index
        breakdown_features['is_monday'] = (raw_df.index.dayofweek == 0).astype(float)
        breakdown_features['is_friday'] = (raw_df.index.dayofweek == 4).astype(float)

        # Volatility regime (uses PAST volatility only - NO LEAKAGE) - Use features_df
        current_vol_10 = features_df['tsla_volatility_10']  # From base features
        historical_avg_vol = current_vol_10.rolling(200, min_periods=20).mean()  # Past 200 bars

        breakdown_features['is_volatile_now'] = (
            current_vol_10 > historical_avg_vol * 1.5
        ).fillna(0).astype(float)

        # In channel binary flags (for key timeframes)
        # Note: time_in_channel features were just created above in section #5
        for tf_name in ['1h', '4h', 'daily']:
            for symbol in ['tsla', 'spy']:
                time_col = f'{symbol}_time_in_channel_{tf_name}'

                # Check if we just created this feature (it's in breakdown_features dict)
                if time_col in breakdown_features:
                    # Binary: in channel if time_in_channel > 5 bars
                    in_channel = (breakdown_features[time_col] > 5).astype(float)
                    breakdown_features[f'{symbol}_in_channel_{tf_name}'] = in_channel
                else:
                    # Fallback to zeros if not found
                    breakdown_features[f'{symbol}_in_channel_{tf_name}'] = np.zeros(num_rows)

        # Earnings proximity (scheduled dates are public - NO LEAKAGE)
        # Note: Would need events_handler passed in for full implementation
        # For now, placeholder
        breakdown_features['is_earnings_week'] = np.zeros(num_rows)  # TODO: Implement with events

        # Debug: Check breakdown feature count
        num_breakdown = len(breakdown_features)
        expected_breakdown = 64  # 10 indicators + 22 time_in_channel + 22 positions + 10 binary
        if num_breakdown != expected_breakdown:
            print(f"   ⚠️  Breakdown features: {num_breakdown} (expected {expected_breakdown})")
            print(f"   Missing/Extra: {num_breakdown - expected_breakdown} features")
        else:
            print(f"   ✓ Breakdown features: {num_breakdown} (correct)")

        return pd.DataFrame(breakdown_features, index=raw_df.index)

    def create_sequences(self, features_df: pd.DataFrame, sequence_length: int = 168,
                        target_horizon: int = 24) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create sequences for training
        Returns (X, y) where:
        - X: (num_sequences, sequence_length, num_features)
        - y: (num_sequences, 2) - [predicted_high, predicted_low] for next target_horizon bars
        """
        features = features_df.values
        num_samples = len(features) - sequence_length - target_horizon

        if num_samples <= 0:
            raise ValueError(f"Not enough data. Need at least {sequence_length + target_horizon} bars")

        X = []
        y = []

        for i in range(num_samples):
            # Input sequence
            seq = features[i:i + sequence_length]
            X.append(seq)

            # Target: high and low in next target_horizon bars
            future_window = features_df.iloc[i + sequence_length:i + sequence_length + target_horizon]
            target_high = future_window['tsla_close'].max()
            target_low = future_window['tsla_close'].min()
            y.append([target_high, target_low])

        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32)

        return X, y
