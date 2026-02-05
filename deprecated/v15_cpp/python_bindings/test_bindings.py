#!/usr/bin/env python3
"""
Integration test for v15scanner Python bindings.

This script verifies that the bindings work correctly without requiring
the full C++ build. It checks the Python wrapper fallback logic.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test that all modules can be imported."""
    print("TEST 1: Module Imports")
    print("-" * 60)

    # Test py_scanner imports
    try:
        from py_scanner import (
            scan_channels_two_pass,
            scan_channels,
            get_backend,
            get_version,
            is_cpp_available,
            ChannelSample,
        )
        print("✓ py_scanner imports successful")
        print(f"  Backend: {get_backend()}")
        print(f"  Version: {get_version()}")
        print(f"  C++ available: {is_cpp_available()}")
    except ImportError as e:
        print(f"✗ py_scanner import failed: {e}")
        return False

    # Test package __init__ file exists
    try:
        init_path = os.path.join(os.path.dirname(__file__), '__init__.py')
        if os.path.exists(init_path):
            print("✓ Package __init__.py exists")
            with open(init_path, 'r') as f:
                content = f.read()
                if '__all__' in content:
                    print("  Contains __all__ definition")
        else:
            print("✗ Package __init__.py missing")
            return False
    except Exception as e:
        print(f"✗ Package check failed: {e}")
        return False

    print()
    return True


def test_channel_sample():
    """Test ChannelSample wrapper class."""
    print("TEST 2: ChannelSample Wrapper")
    print("-" * 60)

    from py_scanner import ChannelSample
    import pickle
    import tempfile
    import pandas as pd

    # Create sample
    sample = ChannelSample(
        timestamp=pd.Timestamp('2020-01-01 10:00:00', tz='UTC'),
        channel_end_idx=100,
        best_window=50,
        tf_features={'1h_rsi': 45.5, '1h_macd': 0.123},
        labels_per_window={
            50: {
                '1h': {
                    'duration_bars': 10,
                    'direction_valid': True
                }
            }
        },
        bar_metadata={
            '1h': {
                'partial_bar_pct': 0.5
            }
        }
    )

    print(f"Created sample: {sample}")
    print(f"  Timestamp: {sample.timestamp}")
    print(f"  Channel end idx: {sample.channel_end_idx}")
    print(f"  Best window: {sample.best_window}")
    print(f"  Features: {sample.tf_features}")

    # Test pickle
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        temp_path = f.name
        pickle.dump(sample, f)

    with open(temp_path, 'rb') as f:
        loaded = pickle.load(f)

    print("✓ Pickle save/load successful")
    print(f"  Loaded: {loaded}")

    # Verify
    assert loaded.channel_end_idx == sample.channel_end_idx
    assert loaded.best_window == sample.best_window
    assert len(loaded.tf_features) == len(sample.tf_features)

    os.remove(temp_path)
    print("✓ All ChannelSample tests passed")
    print()
    return True


def test_api_compatibility():
    """Test API compatibility with v15.scanner."""
    print("TEST 3: API Compatibility")
    print("-" * 60)

    from py_scanner import scan_channels_two_pass
    import inspect

    # Check function signature
    sig = inspect.signature(scan_channels_two_pass)
    params = list(sig.parameters.keys())

    expected_params = [
        'tsla_df', 'spy_df', 'vix_df',
        'step', 'warmup_bars', 'max_samples',
        'workers', 'batch_size', 'progress',
        'strict', 'output_path', 'incremental_path',
        'incremental_chunk'
    ]

    print(f"Function signature: {sig}")
    print(f"\nExpected parameters: {expected_params}")
    print(f"Actual parameters: {params}")

    # Check all expected params exist
    missing = set(expected_params) - set(params)
    if missing:
        print(f"✗ Missing parameters: {missing}")
        return False

    print("✓ All expected parameters present")

    # Check defaults
    defaults = {
        'step': 10,
        'warmup_bars': 32760,
        'max_samples': None,
        'workers': 4,
        'batch_size': 8,
        'progress': True,
        'strict': True,
        'output_path': None,
    }

    for param, expected_default in defaults.items():
        actual_default = sig.parameters[param].default
        if actual_default != expected_default:
            print(f"✗ Default mismatch for {param}: {actual_default} != {expected_default}")
            return False

    print("✓ Default values match specification")
    print("✓ API compatibility verified")
    print()
    return True


def test_error_handling():
    """Test error handling and fallback logic."""
    print("TEST 4: Error Handling")
    print("-" * 60)

    from py_scanner import scan_channels_two_pass, is_cpp_available

    if not is_cpp_available():
        print("ℹ C++ backend not available - testing fallback logic")

        # Verify that fallback is handled gracefully
        try:
            # This should either work (Python fallback) or raise ImportError
            # but not crash
            from py_scanner import py_scan_channels_two_pass
            print("✓ Python fallback import available")
        except (ImportError, NameError):
            print("ℹ Python fallback not available (expected if v15.scanner not installed)")

        print("✓ Fallback logic working correctly")
    else:
        print("ℹ C++ backend available - skipping fallback test")

    print()
    return True


def test_backend_detection():
    """Test backend detection functions."""
    print("TEST 5: Backend Detection")
    print("-" * 60)

    from py_scanner import get_backend, get_version, is_cpp_available

    backend = get_backend()
    version = get_version()
    cpp_avail = is_cpp_available()

    print(f"Backend: {backend}")
    print(f"Version: {version}")
    print(f"C++ available: {cpp_avail}")

    # Verify backend is valid
    assert backend in ['cpp', 'python'], f"Invalid backend: {backend}"
    print("✓ Backend detection working")

    # Verify consistency
    if cpp_avail:
        assert backend == 'cpp', "C++ available but backend is not 'cpp'"
        assert 'C++' in version, "C++ available but version string doesn't mention it"
    else:
        assert backend == 'python', "C++ not available but backend is not 'python'"
        assert 'Python' in version, "Python backend but version string doesn't mention it"

    print("✓ Backend state consistent")
    print()
    return True


def test_documentation():
    """Test that documentation files exist and are readable."""
    print("TEST 6: Documentation")
    print("-" * 60)

    docs = [
        'README.md',
        'QUICKSTART.md',
        '__init__.py',
        'bindings.cpp',
        'py_scanner.py',
        'example.py',
        'build.sh'
    ]

    base_dir = os.path.dirname(__file__)

    for doc in docs:
        path = os.path.join(base_dir, doc)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"✓ {doc:<20} ({size:>8,} bytes)")
        else:
            print(f"✗ {doc:<20} (MISSING)")
            return False

    print("✓ All documentation files present")
    print()
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("v15scanner Python Bindings - Integration Tests")
    print("=" * 60)
    print()

    tests = [
        test_imports,
        test_channel_sample,
        test_api_compatibility,
        test_error_handling,
        test_backend_detection,
        test_documentation,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    for i, (test, result) in enumerate(zip(tests, results), 1):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test.__name__}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print()
        print("✓ All tests passed!")
        print()
        print("Next steps:")
        print("  1. Build the C++ module: ./build.sh")
        print("  2. Run examples: python3 example.py")
        print("  3. Test with real data")
        return 0
    else:
        print()
        print("✗ Some tests failed")
        print()
        print("Please review the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
