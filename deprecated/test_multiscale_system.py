#!/usr/bin/env python3
"""
Comprehensive test suite for multi-scale LNN system.

Runs all tests in sequence:
1. Feature extraction (136 features)
2. CSV generation validation
3. Meta-LNN architecture
4. News system
5. Database schema
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import individual test modules
import test_multiscale_features
import test_csv_generation
import test_meta_lnn
import test_news_system


def main():
    """Run all multi-scale system tests."""
    print("\n" + "=" * 80)
    print(" " * 20 + "MULTI-SCALE LNN SYSTEM TEST SUITE")
    print("=" * 80)
    print("\nThis will test:")
    print("  1. Multi-scale feature extraction (56 → 136 features)")
    print("  2. CSV generation for all timeframes")
    print("  3. Meta-LNN architecture and forward pass")
    print("  4. News fetching and encoding system")
    print("\n" + "=" * 80)

    all_passed = True

    # Test 1: Features
    print("\n\n" + "█" * 80)
    print("TEST 1: MULTI-SCALE FEATURE EXTRACTION")
    print("█" * 80)
    try:
        test_multiscale_features.test_multiscale_features()
        print("✅ Test 1 PASSED")
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
        all_passed = False

    # Test 2: CSV Generation
    print("\n\n" + "█" * 80)
    print("TEST 2: CSV GENERATION VALIDATION")
    print("█" * 80)
    try:
        test_csv_generation.test_csv_generation()
        print("✅ Test 2 PASSED")
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        all_passed = False

    # Test 3: Meta-LNN
    print("\n\n" + "█" * 80)
    print("TEST 3: META-LNN ARCHITECTURE")
    print("█" * 80)
    try:
        test_meta_lnn.test_meta_lnn_architecture()
        test_meta_lnn.test_market_state_calculation()
        test_meta_lnn.test_meta_loss()
        test_meta_lnn.test_modality_dropout()
        print("✅ Test 3 PASSED")
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")
        all_passed = False

    # Test 4: News System
    print("\n\n" + "█" * 80)
    print("TEST 4: NEWS SYSTEM")
    print("█" * 80)
    try:
        test_news_system.test_news_db_init()
        test_news_system.test_news_encoder_backtest_mode()
        test_news_system.test_news_encoder_live_mode()
        test_news_system.test_news_window_retrieval()
        print("✅ Test 4 PASSED")
    except Exception as e:
        print(f"❌ Test 4 FAILED: {e}")
        all_passed = False

    # Final summary
    print("\n\n" + "=" * 80)
    print(" " * 25 + "FINAL TEST RESULTS")
    print("=" * 80)

    if all_passed:
        print("\n✅ ALL TESTS PASSED!")
        print("\nThe multi-scale LNN system is ready for training!")
        print("\nNext steps:")
        print("  1. Generate multi-scale CSVs:")
        print("     python scripts/create_multiscale_csvs.py")
        print("\n  2. Train sub-models:")
        print("     python train_model_lazy.py --input_timeframe 15min --sequence_length 500 --output models/lnn_15min.pth")
        print("     python train_model_lazy.py --input_timeframe 1hour --sequence_length 500 --output models/lnn_1hour.pth")
        print("     python train_model_lazy.py --input_timeframe 4hour --sequence_length 500 --output models/lnn_4hour.pth")
        print("     python train_model_lazy.py --input_timeframe daily --sequence_length 500 --output models/lnn_daily.pth")
        print("\n  3. Run backtests to collect predictions in database")
        print("\n  4. Train meta-LNN coach:")
        print("     python train_meta_lnn.py --mode backtest_no_news")
        print("\n  5. Use ensemble for predictions:")
        print("     python backtest.py --ensemble --mode backtest_no_news")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("\nPlease fix the failing tests before proceeding.")

    print("\n" + "=" * 80)
    print()


if __name__ == '__main__':
    main()
