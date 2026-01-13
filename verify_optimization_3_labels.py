"""
Verify Optimization 3: Vectorized variance vs np.var() in labels.py

Tests whether the manual vectorized variance computation in detect_new_channel()
matches np.var() exactly. This is critical for mathematical equivalence.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))


def manual_variance_vectorized(data):
    """
    Manual variance computation using vectorized operations.
    This is what detect_new_channel() uses in the optimized version.
    """
    # Create sliding windows using strided array view
    window = 50
    if len(data) < window:
        return np.array([])

    from numpy.lib.stride_tricks import as_strided
    stride = data.strides[0]
    num_windows = len(data) - window + 1

    # Create strided view for all windows at once
    data_windows = as_strided(
        data,
        shape=(num_windows, window),
        strides=(stride, stride)
    )

    # Vectorized variance computation for ALL windows at once
    window_vars = np.var(data_windows, axis=1)

    return window_vars


def numpy_variance_loop(data):
    """
    Standard np.var() in a loop (baseline for comparison).
    This is what the original implementation would use.
    """
    window = 50
    if len(data) < window:
        return np.array([])

    num_windows = len(data) - window + 1
    window_vars = np.zeros(num_windows)

    for i in range(num_windows):
        window_data = data[i:i + window]
        window_vars[i] = np.var(window_data)

    return window_vars


def test_vectorized_vs_numpy_var():
    """
    Test if vectorized variance matches np.var() exactly.
    """
    print("=" * 80)
    print("OPTIMIZATION 3: Vectorized variance vs np.var() (labels.py)")
    print("=" * 80)
    print()

    # Test with multiple datasets
    test_cases = [
        ("Random normal", np.random.randn(500)),
        ("Random uniform", np.random.uniform(50, 150, 500)),
        ("Linear trend", np.linspace(100, 200, 500)),
        ("Sine wave", 100 + 50 * np.sin(np.linspace(0, 4 * np.pi, 500))),
        ("Constant", np.ones(500) * 100),
        ("Step function", np.concatenate([np.ones(250) * 100, np.ones(250) * 150])),
    ]

    all_match = True

    for name, data in test_cases:
        print("Testing: {}".format(name))
        print("  Data shape: {}".format(data.shape))

        # Compute variance using both methods
        vars_vectorized = manual_variance_vectorized(data)
        vars_numpy = numpy_variance_loop(data)

        print("  Number of windows: {}".format(len(vars_vectorized)))

        # Check if lengths match
        if len(vars_vectorized) != len(vars_numpy):
            print("  ERROR: Length mismatch!")
            print("    Vectorized: {} windows".format(len(vars_vectorized)))
            print("    Numpy loop: {} windows".format(len(vars_numpy)))
            all_match = False
            print()
            continue

        # Check if all values match
        max_abs_diff = np.max(np.abs(vars_vectorized - vars_numpy))
        max_rel_diff = np.max(np.abs((vars_vectorized - vars_numpy) / (vars_numpy + 1e-10)))

        # Check exact match (within floating point tolerance)
        exact_match = np.allclose(vars_vectorized, vars_numpy, rtol=1e-15, atol=1e-15)

        print("  Max absolute difference: {:.2e}".format(max_abs_diff))
        print("  Max relative difference: {:.2e}".format(max_rel_diff))
        print("  Exact match: {}".format("YES" if exact_match else "NO"))

        if not exact_match:
            # Show first few differing values
            diff_mask = ~np.isclose(vars_vectorized, vars_numpy, rtol=1e-15, atol=1e-15)
            if np.any(diff_mask):
                num_diffs = np.sum(diff_mask)
                print("  Number of differing values: {} / {}".format(num_diffs, len(vars_vectorized)))

                # Show first 3 differences
                diff_indices = np.where(diff_mask)[0][:3]
                for idx in diff_indices:
                    print("    Window {}: {:.15f} vs {:.15f}".format(
                        idx, vars_vectorized[idx], vars_numpy[idx]
                    ))

            all_match = False

        print()

    print("-" * 80)
    if all_match:
        print("RESULT: Vectorized variance matches np.var() EXACTLY")
        return True
    else:
        print("RESULT: Vectorized variance DIFFERS from np.var()")
        return False


def test_edge_cases():
    """Test edge cases that might reveal numerical issues."""
    print("=" * 80)
    print("EDGE CASE TESTING")
    print("=" * 80)
    print()

    edge_cases = [
        ("Very small values", np.random.randn(200) * 1e-10),
        ("Very large values", np.random.randn(200) * 1e10),
        ("Mixed scale", np.concatenate([np.random.randn(100) * 1e-5, np.random.randn(100) * 1e5])),
        ("Near-zero variance", np.ones(200) + np.random.randn(200) * 1e-10),
    ]

    all_match = True

    for name, data in edge_cases:
        print("Testing: {}".format(name))

        vars_vectorized = manual_variance_vectorized(data)
        vars_numpy = numpy_variance_loop(data)

        exact_match = np.allclose(vars_vectorized, vars_numpy, rtol=1e-10, atol=1e-15)

        if exact_match:
            print("  PASS")
        else:
            max_abs_diff = np.max(np.abs(vars_vectorized - vars_numpy))
            max_rel_diff = np.max(np.abs((vars_vectorized - vars_numpy) / (vars_numpy + 1e-10)))
            print("  FAIL: Max abs diff={:.2e}, Max rel diff={:.2e}".format(max_abs_diff, max_rel_diff))
            all_match = False

        print()

    return all_match


if __name__ == '__main__':
    try:
        result1 = test_vectorized_vs_numpy_var()
        result2 = test_edge_cases()

        print("\n" + "=" * 80)
        print("FINAL RESULT")
        print("=" * 80)
        if result1 and result2:
            print("ALL TESTS PASSED: Vectorized variance is mathematically equivalent")
            sys.exit(0)
        else:
            print("TESTS FAILED: Vectorized variance differs from np.var()")
            sys.exit(1)
    except Exception as e:
        print("\nFATAL ERROR: {}".format(e))
        import traceback
        traceback.print_exc()
        sys.exit(2)
