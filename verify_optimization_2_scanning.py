"""
Verify Optimization 2: DataFrame slicing vs numpy array conversion in scanning.py

Tests whether using .iloc slicing on pre-sliced DataFrames produces identical
results to converting DataFrames to numpy arrays and reconstructing them.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))


def simulate_dataframe_slicing(df, position):
    """
    Simulate the OPTIMIZED approach: Pre-slice DataFrame once, then use .iloc
    This is what scanning.py does now.
    """
    # Pre-slice to position (like tsla_presliced = tsla_df.iloc[:end_idx])
    df_presliced = df.iloc[:position]

    # Later, slice again to get window (like tsla_window = tsla_presliced.iloc[:i])
    df_window = df_presliced.iloc[:position]

    # Extract values for computation
    close_values = df_window['close'].values
    high_values = df_window['high'].values
    low_values = df_window['low'].values

    # Simple computation: mean, std, and range
    result = {
        'mean_close': np.mean(close_values),
        'std_close': np.std(close_values),
        'range_high_low': np.mean(high_values - low_values),
        'first_close': close_values[0] if len(close_values) > 0 else 0.0,
        'last_close': close_values[-1] if len(close_values) > 0 else 0.0,
    }

    return result


def simulate_numpy_conversion(df, position):
    """
    Simulate an ALTERNATIVE approach: Convert to numpy arrays and reconstruct
    This would be similar to what some parallel implementations might do.
    """
    # Convert full DataFrame to numpy arrays (as if serializing for multiprocessing)
    close_arr = df['close'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    index_arr = df.index.values

    # Slice arrays to position
    close_slice = close_arr[:position]
    high_slice = high_arr[:position]
    low_slice = low_arr[:position]
    index_slice = index_arr[:position]

    # Reconstruct DataFrame (as a worker process might do)
    df_reconstructed = pd.DataFrame({
        'close': close_slice,
        'high': high_slice,
        'low': low_slice
    }, index=pd.DatetimeIndex(index_slice))

    # Extract values for computation
    close_values = df_reconstructed['close'].values
    high_values = df_reconstructed['high'].values
    low_values = df_reconstructed['low'].values

    # Same computation as above
    result = {
        'mean_close': np.mean(close_values),
        'std_close': np.std(close_values),
        'range_high_low': np.mean(high_values - low_values),
        'first_close': close_values[0] if len(close_values) > 0 else 0.0,
        'last_close': close_values[-1] if len(close_values) > 0 else 0.0,
    }

    return result


def test_dataframe_vs_numpy():
    """
    Test if DataFrame .iloc slicing produces identical results to numpy conversion.
    """
    print("=" * 80)
    print("OPTIMIZATION 2: DataFrame slicing vs Numpy arrays (scanning.py)")
    print("=" * 80)
    print()

    # Generate synthetic test data
    np.random.seed(42)
    n_bars = 1000
    dates = pd.date_range('2023-01-01', periods=n_bars, freq='5min')

    close = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
    high = close + np.abs(np.random.randn(n_bars) * 0.5)
    low = close - np.abs(np.random.randn(n_bars) * 0.5)
    open_price = close + np.random.randn(n_bars) * 0.3
    volume = np.random.randint(1000, 10000, n_bars)

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    print("Test data: {} bars".format(len(df)))
    print()

    # Test multiple positions
    test_positions = [50, 100, 500, 900]
    all_match = True

    for pos in test_positions:
        print("Testing position {}:".format(pos))

        # Approach 1: DataFrame slicing (optimized)
        result_df = simulate_dataframe_slicing(df, pos)

        # Approach 2: Numpy conversion (alternative)
        result_np = simulate_numpy_conversion(df, pos)

        # Compare results
        for key in result_df.keys():
            val_df = result_df[key]
            val_np = result_np[key]

            # Check if values are identical (within floating point tolerance)
            if np.isnan(val_df) and np.isnan(val_np):
                match = True
            else:
                match = np.allclose(val_df, val_np, rtol=1e-15, atol=1e-15)

            status = "MATCH" if match else "DIFFER"
            print("  {}: {:.10f} vs {:.10f} [{}]".format(key, val_df, val_np, status))

            if not match:
                all_match = False

        print()

    print("-" * 80)
    if all_match:
        print("RESULT: DataFrame slicing produces IDENTICAL results to numpy arrays")
        return True
    else:
        print("RESULT: DataFrame slicing produces DIFFERENT results from numpy arrays")
        return False


if __name__ == '__main__':
    try:
        result = test_dataframe_vs_numpy()
        sys.exit(0 if result else 1)
    except Exception as e:
        print("\nFATAL ERROR: {}".format(e))
        import traceback
        traceback.print_exc()
        sys.exit(2)
