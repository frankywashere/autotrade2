#!/usr/bin/env python3
"""
Test script for AutoTrade v7.0 Clean Architecture

Demonstrates the new modular architecture:
- Config-driven feature selection
- Error handling with graceful degradation
- Structured logging
- Metrics tracking

Usage:
    python3 scripts/test_v7_pipeline.py
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_feature_config
from src.features import FeaturePipeline
from src.monitoring import setup_logging, get_logger
from src.errors import InsufficientDataError
import pandas as pd
import numpy as np


def main():
    # Setup logging
    setup_logging(log_level="INFO")
    logger = get_logger(__name__)

    logger.info("=" * 80)
    logger.info("AutoTrade v7.0 Pipeline Test")
    logger.info("=" * 80)

    # Load config
    logger.info("Loading feature configuration...")
    config = get_feature_config()

    logger.info(
        "Config loaded",
        version=config.version,
        windows=config.channel_windows,
        rsi_timeframes=config.rsi_timeframes,
    )

    # Show feature counts
    counts = config.count_features()
    logger.info("Expected feature counts:")
    for category, count in counts.items():
        logger.info(f"  {category}: {count:,}")

    # Initialize pipeline
    logger.info("Initializing feature pipeline...")
    pipeline = FeaturePipeline(config)

    # Create mock data for testing
    logger.info("Creating mock OHLC data...")
    n_bars = 1000
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='1min')

    # Realistic price movement
    tsla_close = 200 + np.cumsum(np.random.randn(n_bars) * 0.5)
    spy_close = 450 + np.cumsum(np.random.randn(n_bars) * 0.3)

    df = pd.DataFrame({
        'timestamp': dates,
        'tsla_close': tsla_close,
        'tsla_high': tsla_close * 1.002,
        'tsla_low': tsla_close * 0.998,
        'tsla_volume': np.random.randint(1_000_000, 10_000_000, n_bars),
        'spy_close': spy_close,
        'spy_high': spy_close * 1.001,
        'spy_low': spy_close * 0.999,
        'spy_volume': np.random.randint(10_000_000, 50_000_000, n_bars),
    }).set_index('timestamp')

    logger.info(f"Mock data created: {len(df)} bars")

    # Test insufficient data error
    logger.info("Testing error handling (insufficient data)...")
    try:
        small_df = df.head(50)  # Only 50 bars
        pipeline.extract(small_df)
    except InsufficientDataError as e:
        logger.info(f"✓ Error handling works: {e}")

    # Extract features
    logger.info("Extracting features...")
    try:
        features = pipeline.extract(df, mode='batch')

        logger.info(
            "✓ Feature extraction successful",
            rows=len(features),
            columns=len(features.columns)
        )

        # Show sample features
        logger.info("Sample channel features (first 5):")
        channel_cols = [col for col in features.columns if 'channel' in col][:5]
        for col in channel_cols:
            logger.info(f"  {col}: {features[col].iloc[-1]:.4f}")

        # Show metrics
        logger.info("Performance metrics:")
        metrics_summary = pipeline.metrics.get_all_metrics()
        for metric, stats in metrics_summary.items():
            logger.info(
                f"  {metric}: {stats['mean']:.2f}ms (p95: {stats['p95']:.2f}ms)"
            )

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}", exc_info=True)
        return 1

    logger.info("=" * 80)
    logger.info("✅ All tests passed! Clean architecture working.")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
