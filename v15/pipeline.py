"""
v15/pipeline.py - Main entry point for cache generation.

Simple CLI that orchestrates the cache generation pipeline:
1. Load market data
2. Calculate scan range
3. Run channel scanner
4. Save results to pickle
"""

import argparse
import pickle
from pathlib import Path

from v15.data import load_market_data
from v15 import scanner


def generate_cache(
    data_dir: str,
    output_path: str,
    step: int = 10,
    workers: int = 4,
    warmup_bars: int = 32760,
    forward_bars: int = 8000
) -> None:
    """
    Generate channel cache from market data.

    Args:
        data_dir: Directory containing market data CSVs
        output_path: Path to save the pickle output
        step: Step size between samples (default 10)
        workers: Number of parallel workers (default 4)
        warmup_bars: Warmup period for channel detection (default 32760)
        forward_bars: Forward bars for label generation (default 8000)
    """
    # 1. Load market data
    print(f"Loading market data from {data_dir}...")
    tsla_df, spy_df, vix_df = load_market_data(data_dir)

    # 2. Print data summary
    print(f"Data loaded:")
    print(f"  Bars: {len(tsla_df):,}")
    print(f"  Date range: {tsla_df.index[0]} to {tsla_df.index[-1]}")

    # 3. Calculate scan range
    start_idx = warmup_bars
    end_idx = len(tsla_df) - forward_bars
    n_positions = (end_idx - start_idx) // step

    if end_idx <= start_idx:
        raise ValueError(
            f"Insufficient data: need at least {warmup_bars + forward_bars} bars, "
            f"got {len(tsla_df)}"
        )

    print(f"Scan range: index {start_idx} to {end_idx}")
    print(f"Estimated positions: ~{n_positions:,} (step={step})")

    # 4. Call scanner
    print(f"Scanning channels (workers={workers})...")
    results = scanner.scan_channels(
        tsla_df=tsla_df,
        spy_df=spy_df,
        vix_df=vix_df,
        step=step,
        warmup_bars=warmup_bars,
        forward_bars=forward_bars,
        workers=workers,
        progress=True
    )

    # 5. Save results to pickle
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving results to {output_path}...")
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    # 6. Print summary
    print(f"Complete!")
    print(f"  Samples generated: {len(results):,}")
    print(f"  Output file: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate channel cache from market data'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing market data CSVs'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save the pickle output'
    )
    parser.add_argument(
        '--step',
        type=int,
        default=10,
        help='Step size between samples (default: 10)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=32760,
        help='Warmup bars for channel detection (default: 32760)'
    )
    parser.add_argument(
        '--forward',
        type=int,
        default=8000,
        help='Forward bars for label generation (default: 8000)'
    )

    args = parser.parse_args()

    generate_cache(
        data_dir=args.data_dir,
        output_path=args.output,
        step=args.step,
        workers=args.workers,
        warmup_bars=args.warmup,
        forward_bars=args.forward
    )
