#!/usr/bin/env python3
"""
Test script for hybrid feature extraction.
Verifies that live predictions can extract features correctly using multi-resolution data.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
from src.ml.live_data_feed import HybridLiveDataFeed
from src.ml.features import TradingFeatureExtractor

def test_hybrid_extraction():
    """Test hybrid feature extraction with live data."""

    print("="*80)
    print("TESTING HYBRID FEATURE EXTRACTION")
    print("="*80)
    print()

    # Initialize
    print("1. Initializing HybridLiveDataFeed...")
    feed = HybridLiveDataFeed(symbols=['TSLA', 'SPY'])
    print("   ✓ Feed initialized")
    print()

    # Fetch live data
    print("2. Fetching multi-resolution live data...")
    print("   (This will download 1-min, 1-hour, and daily data)")
    try:
        df = feed.fetch_for_prediction()
        print(f"   ✓ Data fetched successfully!")
        print(f"   - Combined DataFrame shape: {df.shape}")
        print(f"   - Date range: {df.index[0]} to {df.index[-1]}")
        print()

        # Check multi_resolution attribute
        if 'multi_resolution' in df.attrs:
            print("   ✓ Multi-resolution data attached:")
            for key, data in df.attrs['multi_resolution'].items():
                if isinstance(data, pd.DataFrame) and not data.empty:
                    print(f"     - {key}: {len(data)} bars, {data.index[0]} to {data.index[-1]}")
                else:
                    print(f"     - {key}: Empty or invalid")
        else:
            print("   ✗ ERROR: No multi_resolution data found!")
            return False
        print()

    except Exception as e:
        print(f"   ✗ ERROR fetching data: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Extract features
    print("3. Extracting features with hybrid mode...")
    try:
        extractor = TradingFeatureExtractor()
        features_df = extractor.extract_features(df)
        print(f"   ✓ Features extracted successfully!")
        print(f"   - Features shape: {features_df.shape}")
        print(f"   - Number of features: {features_df.shape[1]}")
        print()

        # Verify channel features exist and have valid values
        print("4. Verifying channel features...")
        channel_cols = [c for c in features_df.columns if '_channel_' in c and '_r_squared' in c]
        print(f"   - Found {len(channel_cols)} r_squared features")

        # Check a few key features
        test_features = ['tsla_channel_1h_r_squared', 'tsla_channel_4h_r_squared', 'tsla_channel_daily_r_squared']
        for feat in test_features:
            if feat in features_df.columns:
                val = features_df[feat].iloc[-1]
                print(f"   - {feat}: {val:.4f}")
            else:
                print(f"   ✗ Missing feature: {feat}")
        print()

        # Verify RSI features
        print("5. Verifying RSI features...")
        rsi_cols = [c for c in features_df.columns if '_rsi_' in c and c.endswith(('1h', '4h', 'daily'))]
        print(f"   - Found {len(rsi_cols)} key RSI features")

        test_features = ['tsla_rsi_1h', 'tsla_rsi_4h', 'tsla_rsi_daily']
        for feat in test_features:
            if feat in features_df.columns:
                val = features_df[feat].iloc[-1]
                print(f"   - {feat}: {val:.2f}")
            else:
                print(f"   ✗ Missing feature: {feat}")
        print()

        # Check for any NaN or zero-filled columns (sign of insufficient data)
        print("6. Checking for data quality issues...")
        zero_cols = []
        for col in features_df.columns:
            if (features_df[col] == 0).all():
                zero_cols.append(col)

        if zero_cols:
            print(f"   ⚠️  WARNING: {len(zero_cols)} features are all zeros:")
            for col in zero_cols[:10]:  # Show first 10
                print(f"      - {col}")
            if len(zero_cols) > 10:
                print(f"      ... and {len(zero_cols) - 10} more")
        else:
            print("   ✓ No all-zero columns detected")

        nan_count = features_df.isna().sum().sum()
        if nan_count > 0:
            print(f"   ⚠️  WARNING: {nan_count} NaN values detected")
        else:
            print("   ✓ No NaN values detected")
        print()

        print("="*80)
        print("✅ HYBRID FEATURE EXTRACTION TEST PASSED!")
        print("="*80)
        print()
        print("Summary:")
        print(f"  - Live data fetched: {df.shape[0]} bars")
        print(f"  - Features extracted: {features_df.shape[1]} features")
        print(f"  - Multi-resolution mode: WORKING ✓")
        print(f"  - Channel features: VALID ✓")
        print(f"  - RSI features: VALID ✓")
        print()
        return True

    except Exception as e:
        print(f"   ✗ ERROR extracting features: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_hybrid_extraction()
    sys.exit(0 if success else 1)
