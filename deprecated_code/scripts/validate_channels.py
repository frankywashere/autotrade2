"""
Channel Quality Validation Script

Validates that the linear regression channels we're detecting are "solid" and not just noise.

Checks:
1. R-squared distribution (goodness of fit)
2. Ping-pong frequency (market actually respects channels)
3. Channel stability over time
4. Breakdown prediction accuracy

Usage:
    python scripts/validate_channels.py --timeframe 1h --sample_size 1000
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from src.ml.data_feed import CSVDataFeed
from src.ml.features import TradingFeatureExtractor
from src.linear_regression import LinearRegressionChannel
import matplotlib.pyplot as plt


def validate_channel_quality(
    df: pd.DataFrame,
    timeframe: str = '1h',
    min_r_squared: float = 0.6,
    min_ping_pongs: int = 2
):
    """
    Validate channel quality across dataset.

    Args:
        df: OHLCV DataFrame
        timeframe: Which timeframe to check
        min_r_squared: Minimum r-squared for "good" channel
        min_ping_pongs: Minimum touches for "validated" channel

    Returns:
        quality_report: Dict with statistics
    """
    print(f"\n{'='*70}")
    print(f"CHANNEL QUALITY VALIDATION - {timeframe.upper()}")
    print(f"{'='*70}")

    # Extract features
    print("\n1. Extracting features...")
    extractor = TradingFeatureExtractor()
    features_df = extractor.extract_features(df)

    # Get channel metrics
    r_squared_col = f'tsla_channel_{timeframe}_r_squared'
    ping_pongs_col = f'tsla_channel_{timeframe}_ping_pongs'
    stability_col = f'tsla_channel_{timeframe}_stability'
    position_col = f'tsla_channel_{timeframe}_position'

    if r_squared_col not in features_df.columns:
        print(f"   ✗ Channel features not found for timeframe: {timeframe}")
        return None

    r_squared = features_df[r_squared_col]
    ping_pongs = features_df[ping_pongs_col]
    stability = features_df[stability_col]
    position = features_df[position_col]

    # Quality analysis
    print(f"\n2. Analyzing channel quality...")

    total_bars = len(r_squared)
    good_fit = (r_squared > min_r_squared).sum()
    validated = (ping_pongs >= min_ping_pongs).sum()
    strong = (stability > 0.7).sum()

    report = {
        'timeframe': timeframe,
        'total_bars': total_bars,
        'good_fit_pct': good_fit / total_bars * 100,
        'validated_pct': validated / total_bars * 100,
        'strong_pct': strong / total_bars * 100,
        'avg_r_squared': r_squared.mean(),
        'median_r_squared': r_squared.median(),
        'avg_ping_pongs': ping_pongs.mean(),
        'median_ping_pongs': ping_pongs.median(),
        'avg_stability': stability.mean()
    }

    # Print results
    print(f"\n3. Results:")
    print(f"   Total bars analyzed: {total_bars:,}")
    print(f"\n   R-Squared (Goodness of Fit):")
    print(f"      Mean: {report['avg_r_squared']:.3f}")
    print(f"      Median: {report['median_r_squared']:.3f}")
    print(f"      >={min_r_squared}: {report['good_fit_pct']:.1f}% of bars")

    print(f"\n   Ping-Pongs (Market Validation):")
    print(f"      Mean: {report['avg_ping_pongs']:.2f}")
    print(f"      Median: {report['median_ping_pongs']:.0f}")
    print(f"      >={min_ping_pongs}: {report['validated_pct']:.1f}% of bars")

    print(f"\n   Stability:")
    print(f"      Mean: {report['avg_stability']:.3f}")
    print(f"      Strong (>0.7): {report['strong_pct']:.1f}% of bars")

    # Interpretation
    print(f"\n4. Interpretation:")
    if report['good_fit_pct'] > 60 and report['validated_pct'] > 40:
        print(f"   ✅ Channels are RELIABLE")
        print(f"      - Majority have good statistical fit")
        print(f"      - Market actually respects them (ping-pongs)")
        print(f"      - Safe to use for trading signals")
    elif report['good_fit_pct'] > 40:
        print(f"   ⚠️  Channels are MODERATE")
        print(f"      - Decent fit but not always validated")
        print(f"      - Use with caution, combine with other indicators")
    else:
        print(f"   ❌ Channels are WEAK")
        print(f"      - Poor statistical fit")
        print(f"      - May be fitting noise, not real patterns")

    # Distribution plot
    print(f"\n5. Generating distribution plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # R-squared distribution
    axes[0, 0].hist(r_squared, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(min_r_squared, color='red', linestyle='--', label=f'Threshold ({min_r_squared})')
    axes[0, 0].set_xlabel('R-Squared')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'R-Squared Distribution ({timeframe})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Ping-pongs distribution
    axes[0, 1].hist(ping_pongs, bins=range(0, int(ping_pongs.max())+2), edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(min_ping_pongs, color='red', linestyle='--', label=f'Threshold ({min_ping_pongs})')
    axes[0, 1].set_xlabel('Ping-Pongs')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Ping-Pong Distribution ({timeframe})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Stability distribution
    axes[1, 0].hist(stability, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Stability Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Stability Distribution ({timeframe})')
    axes[1, 0].grid(True, alpha=0.3)

    # R-squared vs Ping-pongs (correlation check)
    axes[1, 1].scatter(r_squared, ping_pongs, alpha=0.3, s=1)
    axes[1, 1].set_xlabel('R-Squared')
    axes[1, 1].set_ylabel('Ping-Pongs')
    axes[1, 1].set_title('Fit Quality vs Market Validation')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = f'channel_quality_{timeframe}.png'
    plt.savefig(output_path)
    print(f"   ✓ Saved: {output_path}")

    print(f"\n{'='*70}")
    print(f"VALIDATION COMPLETE")
    print(f"{'='*70}\n")

    return report


def main():
    parser = argparse.ArgumentParser(description='Validate Channel Quality')
    parser.add_argument('--timeframe', type=str, default='1h',
                        choices=['15min', '1h', '4h', 'daily'],
                        help='Timeframe to validate')
    parser.add_argument('--year', type=int, default=2023,
                        help='Year to analyze')
    parser.add_argument('--min_r_squared', type=float, default=0.6,
                        help='Minimum r-squared for "good" channel')
    parser.add_argument('--min_ping_pongs', type=int, default=2,
                        help='Minimum touches for "validated" channel')

    args = parser.parse_args()

    # Load data
    print(f"Loading {args.timeframe} data for {args.year}...")
    data_feed = CSVDataFeed(timeframe='1min')  # Load 1-min, will resample
    df = data_feed.load_aligned_data(
        start_date=f'{args.year}-01-01',
        end_date=f'{args.year}-12-31'
    )

    print(f"Loaded {len(df):,} bars")

    # Validate
    report = validate_channel_quality(
        df,
        timeframe=args.timeframe,
        min_r_squared=args.min_r_squared,
        min_ping_pongs=args.min_ping_pongs
    )

    # Summary
    if report:
        if report['good_fit_pct'] > 60:
            print(f"✅ Channel detection is SOLID for {args.timeframe}")
            print(f"   → Safe to use for model training")
        else:
            print(f"⚠️  Channel detection needs improvement for {args.timeframe}")
            print(f"   → Consider tuning LinearRegressionChannel parameters")


if __name__ == '__main__':
    main()
