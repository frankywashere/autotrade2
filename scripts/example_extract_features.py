#!/usr/bin/env python3
"""
Example: Extract Features with AutoTrade v7.0

Shows how to use the modular feature extractors with real or mock data.

Usage:
    python3 scripts/example_extract_features.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_mock_data(n_bars=500):
    """Create mock OHLCV data for testing"""
    print("Creating mock data...")

    dates = pd.date_range('2024-01-01 09:30', periods=n_bars, freq='5min')

    # Generate realistic price movements
    tsla_base = 250
    spy_base = 450

    tsla_returns = np.random.randn(n_bars) * 0.02  # 2% volatility
    spy_returns = np.random.randn(n_bars) * 0.01   # 1% volatility

    tsla_prices = tsla_base * (1 + tsla_returns).cumprod()
    spy_prices = spy_base * (1 + spy_returns).cumprod()

    df = pd.DataFrame({
        'tsla_open': tsla_prices * (1 + np.random.randn(n_bars) * 0.001),
        'tsla_high': tsla_prices * (1 + np.abs(np.random.randn(n_bars)) * 0.005),
        'tsla_low': tsla_prices * (1 - np.abs(np.random.randn(n_bars)) * 0.005),
        'tsla_close': tsla_prices,
        'tsla_volume': np.random.randint(1000000, 5000000, n_bars),
        'spy_open': spy_prices * (1 + np.random.randn(n_bars) * 0.0005),
        'spy_high': spy_prices * (1 + np.abs(np.random.randn(n_bars)) * 0.003),
        'spy_low': spy_prices * (1 - np.abs(np.random.randn(n_bars)) * 0.003),
        'spy_close': spy_prices,
        'spy_volume': np.random.randint(10000000, 30000000, n_bars),
    }, index=dates)

    # Ensure high >= close >= low
    for symbol in ['tsla', 'spy']:
        df[f'{symbol}_high'] = df[[f'{symbol}_open', f'{symbol}_high',
                                    f'{symbol}_low', f'{symbol}_close']].max(axis=1)
        df[f'{symbol}_low'] = df[[f'{symbol}_open', f'{symbol}_high',
                                   f'{symbol}_low', f'{symbol}_close']].min(axis=1)

    print(f"✓ Created {len(df)} bars of mock data")
    return df


def main():
    """Extract features using the modular extractors"""
    print("\n" + "=" * 80)
    print("AutoTrade v7.0 - Feature Extraction Example")
    print("=" * 80)

    # Import extractors
    from config import get_feature_config
    from src.features import (
        MarketFeatureExtractor,
        VIXFeatureExtractor,
        EventFeatureExtractor,
    )

    # Load config
    config = get_feature_config()
    print(f"\n✓ Config loaded: {config.version}")
    print(f"  - Channel windows: {config.channel_windows}")
    print(f"  - RSI timeframes: {config.rsi_timeframes}")

    # Create or load data
    print("\n" + "-" * 80)
    print("STEP 1: Load Data")
    print("-" * 80)

    # Try to load real data, fall back to mock
    data_path = Path('data/tsla_1min_data.csv')
    if data_path.exists():
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"✓ Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    else:
        print("No real data found, creating mock data...")
        df = create_mock_data(n_bars=500)

    # Extract features
    print("\n" + "-" * 80)
    print("STEP 2: Extract Market Features")
    print("-" * 80)

    market_extractor = MarketFeatureExtractor(config)
    market_features = market_extractor.extract(
        df,
        symbols=['tsla', 'spy'],
        mode='batch'
    )
    print(f"✓ Extracted {market_features.shape[1]} market features")
    print(f"  Sample columns: {list(market_features.columns[:5])}")

    print("\n" + "-" * 80)
    print("STEP 3: Extract VIX Features (with graceful fallback)")
    print("-" * 80)

    vix_extractor = VIXFeatureExtractor(config)
    vix_features = vix_extractor.extract(df, vix_data=None, mode='batch')
    print(f"✓ Extracted {vix_features.shape[1]} VIX features")
    print(f"  Sample columns: {list(vix_features.columns[:5])}")

    print("\n" + "-" * 80)
    print("STEP 4: Extract Event Features (with graceful fallback)")
    print("-" * 80)

    event_extractor = EventFeatureExtractor(config)
    event_features = event_extractor.extract(
        df,
        earnings_dates=None,
        fomc_dates=None,
        mode='batch'
    )
    print(f"✓ Extracted {event_features.shape[1]} event features")
    print(f"  Columns: {list(event_features.columns)}")

    # Combine all features
    print("\n" + "-" * 80)
    print("STEP 5: Combine All Features")
    print("-" * 80)

    all_features = pd.concat([
        market_features,
        vix_features,
        event_features
    ], axis=1)

    print(f"\n✓ Combined feature DataFrame:")
    print(f"  Shape: {all_features.shape}")
    print(f"  Index: {all_features.index[0]} to {all_features.index[-1]}")
    print(f"  Total features: {all_features.shape[1]}")

    # Show sample
    print("\n" + "-" * 80)
    print("STEP 6: Sample Feature Values (last 5 rows)")
    print("-" * 80)

    # Show a few interesting features
    sample_cols = [
        'tsla_close_price',
        'tsla_returns_1bar',
        'tsla_volatility_20bar',
        'tsla_rsi_5min_value',
        'vix_close',
        'days_to_next_earnings',
    ]

    available_cols = [c for c in sample_cols if c in all_features.columns]
    if available_cols:
        print(all_features[available_cols].tail())
    else:
        print(all_features.iloc[:, :6].tail())

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Successfully extracted {all_features.shape[1]} features")
    print(f"✓ Data range: {len(all_features)} bars")
    print(f"✓ All extractors working with graceful degradation")
    print("\nFeature Breakdown:")
    print(f"  • Market features: {market_features.shape[1]}")
    print(f"  • VIX features: {vix_features.shape[1]}")
    print(f"  • Event features: {event_features.shape[1]}")

    print("\nNext Steps:")
    print("  - Add channel feature extraction (requires resampling)")
    print("  - Add channel history features (requires transition labels)")
    print("  - Add breakdown features (requires channel features)")
    print("  - Save features for training")

    return 0


if __name__ == "__main__":
    sys.exit(main())
