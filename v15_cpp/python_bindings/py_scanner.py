"""
Python wrapper for v15scanner C++ backend.

Provides the same interface as v15.scanner but uses the high-performance C++
implementation when available. Falls back to pure Python if C++ module is not
built or import fails.

Usage:
    from v15_cpp.python_bindings.py_scanner import scan_channels_two_pass

    samples = scan_channels_two_pass(
        tsla_df, spy_df, vix_df,
        step=10,
        workers=8,
        max_samples=10000,
        output_path="samples.pkl"
    )

Command-line interface (same as v15.scanner):
    python -m v15_cpp.python_bindings.py_scanner --step 10 --workers 8 --output samples.pkl
"""

import argparse
import os
import pickle
import sys
import time
from typing import List, Optional

import pandas as pd

# Try to import C++ backend
try:
    import v15scanner_cpp as cpp_backend
    CPP_AVAILABLE = True
    BACKEND = "cpp"
except ImportError:
    CPP_AVAILABLE = False
    BACKEND = "python"
    # Fallback to Python implementation
    try:
        from v15.scanner import (
            scan_channels_two_pass as py_scan_channels_two_pass,
            EXPECTED_FEATURE_COUNT
        )
    except ImportError:
        py_scan_channels_two_pass = None
        EXPECTED_FEATURE_COUNT = None

# Try to import data loader
try:
    from v15.data import load_market_data
except ImportError:
    load_market_data = None


# =============================================================================
# Version and Backend Detection
# =============================================================================

def get_backend():
    """Get the current backend being used ('cpp' or 'python')."""
    return BACKEND


def is_cpp_available():
    """Check if C++ backend is available."""
    return CPP_AVAILABLE


def get_version():
    """Get version string with backend info."""
    if CPP_AVAILABLE:
        return f"v15scanner {cpp_backend.__version__} (C++ backend)"
    else:
        return "v15scanner (Python backend)"


# =============================================================================
# ChannelSample Wrapper for Pickle Compatibility
# =============================================================================

class ChannelSample:
    """
    Python wrapper for ChannelSample that provides pickle compatibility.

    This class wraps the C++ ChannelSample or provides a pure Python
    implementation, ensuring samples can be saved/loaded with pickle.
    """

    def __init__(self, data_dict=None, **kwargs):
        """
        Create ChannelSample from dict or keyword arguments.

        Args:
            data_dict: Dictionary with sample data (from C++ backend or pickle)
            **kwargs: Individual fields (timestamp, channel_end_idx, etc.)
        """
        if data_dict is not None:
            # Initialize from dict (C++ backend or unpickle)
            self.timestamp = data_dict.get('timestamp')
            self.channel_end_idx = data_dict.get('channel_end_idx', 0)
            self.best_window = data_dict.get('best_window', 50)
            self.tf_features = data_dict.get('tf_features', {})
            self.labels_per_window = data_dict.get('labels_per_window', {})
            self.bar_metadata = data_dict.get('bar_metadata', {})
        else:
            # Initialize from kwargs
            self.timestamp = kwargs.get('timestamp')
            self.channel_end_idx = kwargs.get('channel_end_idx', 0)
            self.best_window = kwargs.get('best_window', 50)
            self.tf_features = kwargs.get('tf_features', {})
            self.labels_per_window = kwargs.get('labels_per_window', {})
            self.bar_metadata = kwargs.get('bar_metadata', {})

    def __repr__(self):
        return (f"<ChannelSample idx={self.channel_end_idx} "
                f"features={len(self.tf_features)} "
                f"window={self.best_window}>")

    def __getstate__(self):
        """Support for pickle."""
        return {
            'timestamp': self.timestamp,
            'channel_end_idx': self.channel_end_idx,
            'best_window': self.best_window,
            'tf_features': self.tf_features,
            'labels_per_window': self.labels_per_window,
            'bar_metadata': self.bar_metadata,
        }

    def __setstate__(self, state):
        """Support for unpickle."""
        self.timestamp = state['timestamp']
        self.channel_end_idx = state['channel_end_idx']
        self.best_window = state['best_window']
        self.tf_features = state['tf_features']
        self.labels_per_window = state['labels_per_window']
        self.bar_metadata = state['bar_metadata']


# =============================================================================
# Main Scanner Function
# =============================================================================

def scan_channels_two_pass(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    step: int = 10,
    warmup_bars: int = 32760,
    max_samples: Optional[int] = None,
    workers: int = 4,
    batch_size: int = 8,
    progress: bool = True,
    strict: bool = True,
    output_path: Optional[str] = None,
    incremental_path: Optional[str] = None,
    incremental_chunk: int = 1000
) -> List[ChannelSample]:
    """
    Scan for channels using the two-pass labeling system.

    Uses C++ backend if available, otherwise falls back to Python implementation.

    CHANNEL-END SAMPLING ARCHITECTURE:
    1. PASS 1: detect_all_channels() - find all channels
    2. PASS 2: generate_all_labels() - compute labels at channel END
    3. SCAN: Iterate over detected channels - each channel = ONE sample

    Each detected channel produces EXACTLY ONE sample at its end position.
    The --step parameter controls channel detection spacing in Pass 1.

    Args:
        tsla_df: TSLA OHLCV DataFrame with DatetimeIndex
        spy_df: SPY OHLCV DataFrame aligned to TSLA
        vix_df: VIX OHLCV DataFrame aligned to TSLA
        step: Step size for channel detection in Pass 1 (default 10)
        warmup_bars: Minimum 5min bars before first sample (default 32760)
        max_samples: Maximum samples to generate (None = unlimited)
        workers: Number of parallel workers (default 4)
        batch_size: Channels per batch for parallel processing (default 8)
        progress: Show progress bar (default True)
        strict: Raise on errors (default True)
        output_path: Output file path for saving results
        incremental_path: Temp file for incremental writes (C++ not supported)
        incremental_chunk: Samples to buffer before writing (C++ not supported)

    Returns:
        List of ChannelSample objects
    """
    print(f"\n[BACKEND] Using {get_version()}")

    if CPP_AVAILABLE:
        # Use C++ backend
        return _scan_cpp(
            tsla_df, spy_df, vix_df,
            step=step,
            warmup_bars=warmup_bars,
            max_samples=max_samples,
            workers=workers,
            batch_size=batch_size,
            progress=progress,
            strict=strict,
            output_path=output_path
        )
    else:
        # Fallback to Python implementation
        if py_scan_channels_two_pass is None:
            raise ImportError(
                "Neither C++ backend nor Python scanner is available. "
                "Please install the C++ module or ensure v15.scanner is available."
            )

        print("[FALLBACK] C++ backend not available, using Python implementation")
        return py_scan_channels_two_pass(
            tsla_df, spy_df, vix_df,
            step=step,
            warmup_bars=warmup_bars,
            max_samples=max_samples,
            workers=workers,
            batch_size=batch_size,
            progress=progress,
            strict=strict,
            output_path=output_path,
            incremental_path=incremental_path,
            incremental_chunk=incremental_chunk
        )


def _scan_cpp(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    step: int = 10,
    warmup_bars: int = 32760,
    max_samples: Optional[int] = None,
    workers: int = 4,
    batch_size: int = 8,
    progress: bool = True,
    strict: bool = True,
    output_path: Optional[str] = None
) -> List[ChannelSample]:
    """
    Run scanner using C++ backend.

    Internal function - use scan_channels_two_pass() instead.
    """
    # Create configuration
    config = cpp_backend.ScannerConfig()
    config.step = step
    config.warmup_bars = warmup_bars
    config.max_samples = max_samples if max_samples is not None else 0
    config.workers = workers if workers > 0 else 0
    config.batch_size = batch_size
    config.progress = progress
    config.verbose = True
    config.strict = strict
    config.output_path = output_path if output_path else ""

    # Create scanner
    scanner = cpp_backend.Scanner(config)

    # Run scan
    print(f"\n[C++] Starting scan with {workers} workers...")
    start_time = time.time()

    try:
        # C++ scan returns list of dicts (pickle-compatible)
        sample_dicts = scanner.scan(tsla_df, spy_df, vix_df)

        # Wrap in ChannelSample objects
        samples = [ChannelSample(data_dict=d) for d in sample_dicts]

        # Get statistics
        stats = scanner.get_stats()

        # Print summary
        elapsed = time.time() - start_time
        print(f"\n[C++] Scan complete in {elapsed:.1f}s")
        print(f"  Samples created: {stats.samples_created}")
        print(f"  Samples skipped: {stats.samples_skipped}")
        print(f"  Errors: {stats.errors_encountered}")
        print(f"  Throughput: {stats.samples_per_second:.2f} samples/sec")

        # Save to file if requested
        if output_path:
            print(f"\n[SAVE] Writing {len(samples)} samples to {output_path}...")
            with open(output_path, 'wb') as f:
                pickle.dump(samples, f)
            print(f"[SAVE] Saved successfully!")

        return samples

    except Exception as e:
        print(f"\n[ERROR] C++ scan failed: {e}")
        if strict:
            raise
        return []


# =============================================================================
# Command-Line Interface
# =============================================================================

def main():
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description='V15 Channel Scanner (C++ backend when available)'
    )
    parser.add_argument('--step', type=int, default=10,
                        help='Step size for CHANNEL DETECTION in Pass 1 (default: 10)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to generate')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path for samples (pickle format)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: auto-detect)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Channels per batch for parallel processing (default: 8)')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel processing')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Data directory path (default: data)')
    parser.add_argument('--backend', type=str, choices=['cpp', 'python', 'auto'],
                        default='auto',
                        help='Force specific backend (default: auto)')
    parser.add_argument('--version', action='store_true',
                        help='Show version and backend info')

    args = parser.parse_args()

    # Show version and exit
    if args.version:
        print(get_version())
        print(f"Backend: {get_backend()}")
        print(f"C++ available: {is_cpp_available()}")
        return

    # Check for data loader
    if load_market_data is None:
        print("ERROR: v15.data.load_market_data not available")
        print("Please ensure v15 package is installed")
        sys.exit(1)

    # Worker configuration
    if args.no_parallel:
        workers = 1
        batch_size = 1
    else:
        import multiprocessing
        workers = args.workers if args.workers is not None else max(1, multiprocessing.cpu_count() - 1)
        batch_size = args.batch_size

    print("=" * 60)
    print("V15 Channel Scanner")
    print("=" * 60)
    print(f"\nBackend: {get_version()}")
    print(f"\nConfiguration:")
    print(f"  Channel detection step: {args.step}")
    print(f"  Max samples: {args.max_samples if args.max_samples else 'unlimited'}")
    print(f"  Output file: {args.output if args.output else 'none'}")
    print(f"  Workers: {workers}")
    print(f"  Batch size: {batch_size}")

    # Load data
    print(f"\nLoading market data from {args.data_dir}...")
    tsla, spy, vix = load_market_data(args.data_dir)
    print(f"Loaded {len(tsla)} bars")
    print(f"Date range: {tsla.index[0]} to {tsla.index[-1]}")

    # Run scanner
    print(f"\nRunning scanner (step={args.step})...")
    samples = scan_channels_two_pass(
        tsla, spy, vix,
        step=args.step,
        max_samples=args.max_samples,
        workers=workers,
        batch_size=batch_size,
        progress=True,
        strict=True,
        output_path=args.output
    )

    # Summary
    print(f"\n{'=' * 60}")
    print(f"SCAN COMPLETE")
    print(f"{'=' * 60}")
    print(f"Generated {len(samples)} samples")

    if samples:
        print(f"\nFirst sample: {samples[0].timestamp}")
        print(f"Last sample: {samples[-1].timestamp}")
        print(f"Features per sample: {len(samples[0].tf_features)}")

        # Show sample features
        if samples[0].tf_features:
            print(f"\nSample feature names (first 10):")
            for i, (name, value) in enumerate(list(samples[0].tf_features.items())[:10]):
                print(f"  - {name}: {value:.4f}")


# Alias for backward compatibility
scan_channels = scan_channels_two_pass


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
