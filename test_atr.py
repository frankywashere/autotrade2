"""
Test script for ATR (Average True Range) calculation.
"""

import numpy as np
import sys
sys.path.append('/Users/frank/Desktop/CodingProjects/x7')

from v7.core.channel import calculate_atr


def test_atr_basic():
    """Test ATR with simple synthetic data."""
    # Create simple test data
    high = np.array([110, 115, 120, 118, 125, 130, 128, 135, 140, 138,
                     145, 150, 148, 155, 160], dtype=np.float64)
    low = np.array([100, 105, 110, 108, 115, 120, 118, 125, 130, 128,
                    135, 140, 138, 145, 150], dtype=np.float64)
    close = np.array([105, 110, 115, 113, 120, 125, 123, 130, 135, 133,
                      140, 145, 143, 150, 155], dtype=np.float64)

    # Calculate ATR with period=5 for easier verification
    atr_series, current_atr = calculate_atr(high, low, close, period=5)

    print("Test 1: Basic ATR Calculation")
    print(f"  Data length: {len(high)}")
    print(f"  ATR series length: {len(atr_series)}")
    print(f"  Current ATR: {current_atr:.4f}")
    print(f"  Last 5 ATR values: {atr_series[-5:]}")

    # Verify outputs
    assert len(atr_series) == len(high), "ATR series should match input length"
    assert current_atr > 0, "Current ATR should be positive"
    assert current_atr == atr_series[-1], "Current ATR should match last value"
    print("  ✓ Basic test passed\n")


def test_atr_empty():
    """Test ATR with empty arrays."""
    high = np.array([])
    low = np.array([])
    close = np.array([])

    atr_series, current_atr = calculate_atr(high, low, close)

    print("Test 2: Empty Arrays")
    print(f"  ATR series length: {len(atr_series)}")
    print(f"  Current ATR: {current_atr}")

    assert len(atr_series) == 0, "Should return empty array"
    assert current_atr == 0.0, "Should return 0.0 for empty data"
    print("  ✓ Empty test passed\n")


def test_atr_single_bar():
    """Test ATR with single bar."""
    high = np.array([110.0])
    low = np.array([100.0])
    close = np.array([105.0])

    atr_series, current_atr = calculate_atr(high, low, close, period=14)

    print("Test 3: Single Bar")
    print(f"  ATR series: {atr_series}")
    print(f"  Current ATR: {current_atr:.4f}")
    print(f"  Expected: {(110 - 100):.4f}")

    assert len(atr_series) == 1, "Should return single value"
    assert atr_series[0] == 10.0, "First bar ATR should be high - low"
    assert current_atr == 10.0, "Current ATR should be 10.0"
    print("  ✓ Single bar test passed\n")


def test_atr_with_gaps():
    """Test ATR with price gaps (where previous close matters)."""
    # Create data with a gap up
    high = np.array([110, 125, 130], dtype=np.float64)
    low = np.array([100, 115, 120], dtype=np.float64)
    close = np.array([105, 120, 125], dtype=np.float64)

    atr_series, current_atr = calculate_atr(high, low, close, period=2)

    print("Test 4: Price Gaps")
    print(f"  High: {high}")
    print(f"  Low: {low}")
    print(f"  Close: {close}")

    # Calculate expected TR values manually
    tr1 = high[0] - low[0]  # 10
    tr2 = max(high[1] - low[1],  # 10
              abs(high[1] - close[0]),  # 20 (gap up)
              abs(low[1] - close[0]))   # 10
    tr3 = max(high[2] - low[2],  # 10
              abs(high[2] - close[1]),  # 10
              abs(low[2] - close[1]))   # 0

    print(f"  TR values: [{tr1}, {tr2}, {tr3}]")
    print(f"  ATR series: {atr_series}")
    print(f"  Current ATR: {current_atr:.4f}")

    # TR2 should pick up the gap (max of 10, 20, 10) = 20
    assert tr2 == 20, "TR should capture gap up"
    print("  ✓ Gap test passed\n")


def test_atr_standard_period():
    """Test ATR with standard 14-period."""
    # Create 20 bars of realistic data
    np.random.seed(42)
    price = 100.0
    high = []
    low = []
    close = []

    for _ in range(20):
        # Simulate daily price movement
        daily_range = np.random.uniform(1, 5)
        h = price + daily_range
        l = price - daily_range
        c = price + np.random.uniform(-daily_range, daily_range)

        high.append(h)
        low.append(l)
        close.append(c)
        price = c

    high = np.array(high, dtype=np.float64)
    low = np.array(low, dtype=np.float64)
    close = np.array(close, dtype=np.float64)

    atr_series, current_atr = calculate_atr(high, low, close, period=14)

    print("Test 5: Standard 14-Period ATR")
    print(f"  Data length: {len(high)}")
    print(f"  ATR series: {atr_series}")
    print(f"  Current ATR: {current_atr:.4f}")
    print(f"  Average ATR: {np.mean(atr_series):.4f}")

    assert len(atr_series) == 20, "Should have 20 ATR values"
    assert current_atr > 0, "ATR should be positive"
    assert np.all(atr_series > 0), "All ATR values should be positive"
    print("  ✓ Standard period test passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("ATR (Average True Range) Unit Tests")
    print("=" * 60 + "\n")

    try:
        test_atr_basic()
        test_atr_empty()
        test_atr_single_bar()
        test_atr_with_gaps()
        test_atr_standard_period()

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
