"""
Feature extraction with progress feedback for lazy loading
This is a wrapper around the existing feature extractor with progress reporting
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import time

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from src.ml.features import TradingFeatureExtractor


class TradingFeatureExtractorWithProgress(TradingFeatureExtractor):
    """
    Extends TradingFeatureExtractor to add progress feedback during extraction.
    """

    def extract_features(self, df: pd.DataFrame, **kwargs) -> tuple:
        """
        Extract all features from aligned SPY-TSLA data with progress feedback.
        Returns (features_df, continuation_df) if continuation=True, else (features_df, None)
        """
        features_df = pd.DataFrame(index=df.index)
        total_steps = 8  # Number of feature extraction steps
        continuation_enabled = kwargs.get('continuation', False)

        print(f"  Extracting features from {len(df):,} bars...")

        with tqdm(total=total_steps, desc="    Feature extraction", unit="step",
                  ncols=80, ascii=True, bar_format="{l_bar}{bar:30}{r_bar}") as pbar:

            # 1. Price features
            pbar.set_postfix_str("Price features (returns, volatility)")
            features_df = self._extract_price_features(df, features_df)
            pbar.update(1)
            time.sleep(0.01)  # Minimal pause to prevent interference

            # 2. Channel features (slowest - multi-timeframe)
            pbar.set_postfix_str("Channel features (3 timeframes) - this takes time...")
            start_channels = time.time()
            features_df = self._extract_channel_features(df, features_df)
            channel_time = time.time() - start_channels
            pbar.set_postfix_str(f"Channels done ({channel_time:.1f}s)")
            pbar.update(1)

            # 3. RSI features
            pbar.set_postfix_str("RSI indicators (3 timeframes)")
            features_df = self._extract_rsi_features(df, features_df)
            pbar.update(1)

            # 4. Correlation features
            pbar.set_postfix_str("SPY-TSLA correlations")
            features_df = self._extract_correlation_features(df, features_df)
            pbar.update(1)

            # 5. Cycle features
            pbar.set_postfix_str("52-week highs/lows, mega channels")
            features_df = self._extract_cycle_features(df, features_df)
            pbar.update(1)

            # 6. Volume features
            pbar.set_postfix_str("Volume ratios")
            features_df = self._extract_volume_features(df, features_df)
            pbar.update(1)

            # 7. Time features
            pbar.set_postfix_str("Time encoding")
            features_df = self._extract_time_features(df, features_df)
            pbar.update(1)

        # 8. Generate continuation labels (optional)
        if continuation_enabled:
            pbar.set_postfix_str("Generating continuation labels")
            timestamps = df.index.tolist()
            continuation_df = self.generate_continuation_labels(df, timestamps, prediction_horizon=24)
            pbar.update(1)
        else:
            continuation_df = None
            pbar.update(1)

        # Drop any NaN rows (from rolling calculations)
        features_df = features_df.bfill().fillna(0)

        return features_df, continuation_df