#!/usr/bin/env python3
"""
Test script to verify binary loader samples are compatible with inspector.py access patterns.

Key access patterns from inspector.py:
- sample.timestamp (used with df.index.searchsorted)
- sample.channel_end_idx
- sample.labels_per_window[window]['tsla'][tf].attribute
"""

import sys
sys.path.insert(0, '/Users/frank/Desktop/CodingProjects/x14/v15')

import pandas as pd
import numpy as np
from binary_loader import load_samples_simple, ChannelSample, ChannelLabels

def print_section(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)

def test_basic_loading():
    """Test 1: Basic loading of samples"""
    print_section("TEST 1: Basic Loading")
    
    try:
        samples = load_samples_simple('/Users/frank/Desktop/CodingProjects/x14/v15_cpp/baseline_test.bin')
        print(f"[PASS] Loaded {len(samples)} samples successfully")
        return samples
    except Exception as e:
        print(f"[FAIL] Failed to load samples: {e}")
        return None

def test_timestamp_compatibility(samples):
    """Test 2: Timestamp compatibility with pd.Index.searchsorted"""
    print_section("TEST 2: Timestamp Compatibility (pd.Index.searchsorted)")
    
    if not samples:
        print("[SKIP] No samples to test")
        return False
    
    sample = samples[0]
    
    # Check timestamp type
    print(f"  sample.timestamp type: {type(sample.timestamp)}")
    print(f"  sample.timestamp value: {sample.timestamp}")
    
    # Create a mock DataFrame index like inspector does
    try:
        # Create a DatetimeIndex with sample timestamps
        timestamps = [s.timestamp for s in samples[:10]]
        df_index = pd.DatetimeIndex(timestamps)
        
        # Test searchsorted (key inspector operation)
        test_ts = sample.timestamp
        idx = df_index.searchsorted(test_ts)
        
        print(f"  Created DatetimeIndex with {len(df_index)} entries")
        print(f"  searchsorted({test_ts}) returned: {idx}")
        print("[PASS] Timestamp works with pd.Index.searchsorted")
        return True
    except Exception as e:
        print(f"[FAIL] Timestamp incompatible: {e}")
        return False

def test_channel_end_idx(samples):
    """Test 3: channel_end_idx attribute access"""
    print_section("TEST 3: channel_end_idx Attribute")
    
    if not samples:
        print("[SKIP] No samples to test")
        return False
    
    sample = samples[0]
    
    try:
        idx = sample.channel_end_idx
        print(f"  sample.channel_end_idx = {idx}")
        print(f"  Type: {type(idx)}")
        
        # Verify it's usable as an integer index
        if isinstance(idx, (int, np.integer)):
            print("[PASS] channel_end_idx is integer-compatible")
            return True
        else:
            print(f"[FAIL] channel_end_idx is not int: {type(idx)}")
            return False
    except AttributeError as e:
        print(f"[FAIL] Missing attribute: {e}")
        return False

def test_labels_per_window_structure(samples):
    """Test 4: labels_per_window[window]['tsla'][tf].attribute access pattern"""
    print_section("TEST 4: labels_per_window Access Pattern")
    
    if not samples:
        print("[SKIP] No samples to test")
        return False
    
    sample = samples[0]
    issues = []
    
    # Test structure exists
    if not hasattr(sample, 'labels_per_window'):
        print("[FAIL] Missing labels_per_window attribute")
        return False
    
    print(f"  Available windows: {list(sample.labels_per_window.keys())}")
    
    if not sample.labels_per_window:
        print("[FAIL] labels_per_window is empty")
        return False
    
    # Pick first window for testing
    window = list(sample.labels_per_window.keys())[0]
    window_data = sample.labels_per_window[window]
    
    print(f"  Testing window {window}:")
    print(f"    Keys in window: {list(window_data.keys())}")
    
    # Test 'tsla' key exists
    if 'tsla' not in window_data:
        issues.append("Missing 'tsla' key in labels_per_window[window]")
    else:
        tsla_data = window_data['tsla']
        print(f"    Timeframes for TSLA: {list(tsla_data.keys())}")
        
        # Test accessing a timeframe's labels
        if tsla_data:
            tf = list(tsla_data.keys())[0]
            labels = tsla_data[tf]
            print(f"\n  Testing labels_per_window[{window}]['tsla']['{tf}']:")
            print(f"    Type: {type(labels)}")
            
            # Test common label attributes used by inspector
            test_attrs = [
                'duration_bars',
                'break_direction', 
                'break_magnitude',
                'permanent_break',
                'bars_to_first_break',
                'source_channel_slope',
                'source_channel_r_squared',
                'source_channel_direction',
                'best_next_channel_direction',
                'best_next_channel_bars_away',
                'rsi_at_first_break',
                'duration_valid',
                'break_scan_valid'
            ]
            
            print(f"\n  Checking label attributes:")
            for attr in test_attrs:
                if hasattr(labels, attr):
                    val = getattr(labels, attr)
                    print(f"    [OK] {attr} = {val}")
                else:
                    issues.append(f"Missing attribute: {attr}")
                    print(f"    [MISSING] {attr}")
    
    # Test 'spy' key exists (if applicable)
    if 'spy' not in window_data:
        print("\n  Note: 'spy' key not present (may be expected)")
    else:
        spy_data = window_data['spy']
        print(f"\n    Timeframes for SPY: {list(spy_data.keys())}")
        
        if spy_data:
            tf = list(spy_data.keys())[0]
            labels = spy_data[tf]
            print(f"    SPY labels type: {type(labels)}")
    
    if issues:
        print(f"\n[FAIL] Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n[PASS] All label access patterns work correctly")
        return True

def test_all_samples_consistency(samples, limit=100):
    """Test 5: Verify consistency across multiple samples"""
    print_section(f"TEST 5: Consistency Check (first {limit} samples)")
    
    if not samples:
        print("[SKIP] No samples to test")
        return False
    
    test_count = min(len(samples), limit)
    issues = []
    
    for i in range(test_count):
        sample = samples[i]
        
        # Check timestamp
        if not isinstance(sample.timestamp, pd.Timestamp):
            issues.append(f"Sample {i}: timestamp is {type(sample.timestamp)}, not pd.Timestamp")
        
        # Check channel_end_idx
        if not isinstance(sample.channel_end_idx, (int, np.integer)):
            issues.append(f"Sample {i}: channel_end_idx is {type(sample.channel_end_idx)}")
        
        # Check labels structure
        for window, window_data in sample.labels_per_window.items():
            if 'tsla' not in window_data:
                issues.append(f"Sample {i}, window {window}: missing 'tsla' key")
            else:
                for tf, labels in window_data['tsla'].items():
                    if not isinstance(labels, ChannelLabels):
                        issues.append(f"Sample {i}, window {window}, tf {tf}: labels is {type(labels)}")
    
    if issues:
        print(f"[FAIL] Found {len(issues)} consistency issues (showing first 10):")
        for issue in issues[:10]:
            print(f"  - {issue}")
        return False
    else:
        print(f"[PASS] All {test_count} samples have consistent structure")
        return True

def test_inspector_simulation(samples):
    """Test 6: Simulate actual inspector access patterns"""
    print_section("TEST 6: Inspector Access Simulation")
    
    if not samples:
        print("[SKIP] No samples to test")
        return False
    
    try:
        # Create a mock price DataFrame like inspector would have
        # Use the actual number of samples we have
        num_samples = len(samples)
        timestamps = [s.timestamp for s in samples]
        df = pd.DataFrame({
            'close': np.random.random(num_samples) * 100 + 400
        }, index=pd.DatetimeIndex(timestamps))
        
        print(f"  Created mock price DataFrame with {num_samples} rows")
        
        # Simulate inspector's sample navigation
        sample = samples[0]
        
        # Pattern 1: Find bar index using searchsorted
        bar_idx = df.index.searchsorted(sample.timestamp)
        print(f"  df.index.searchsorted(sample.timestamp) = {bar_idx}")
        
        # Pattern 2: Access channel_end_idx for slicing
        start_idx = sample.channel_end_idx - 10
        end_idx = sample.channel_end_idx
        print(f"  Channel slice range: {start_idx} to {end_idx}")
        
        # Pattern 3: Access nested labels for all windows/assets/timeframes
        access_count = 0
        for window in sample.labels_per_window:
            for asset in ['tsla', 'spy']:
                if asset in sample.labels_per_window[window]:
                    for tf, labels in sample.labels_per_window[window][asset].items():
                        # Access various label attributes (as inspector would)
                        _ = labels.duration_bars
                        _ = labels.break_direction
                        _ = labels.break_magnitude
                        _ = labels.permanent_break
                        _ = labels.bars_to_first_break
                        _ = labels.source_channel_slope
                        _ = labels.source_channel_r_squared
                        _ = labels.source_channel_direction
                        _ = labels.best_next_channel_direction
                        _ = labels.rsi_at_first_break
                        _ = labels.duration_valid
                        _ = labels.break_scan_valid
                        access_count += 1
        
        print(f"  Successfully accessed {access_count} label combinations")
        
        # Pattern 4: Iterate through samples like inspector's next/prev navigation
        for i, s in enumerate(samples[:5]):
            _ = s.timestamp
            _ = s.channel_end_idx
            _ = s.best_window
            for w, w_data in s.labels_per_window.items():
                _ = w_data['tsla']
        
        print("  Successfully iterated through first 5 samples")
        
        print("[PASS] Inspector access simulation successful")
        return True
        
    except Exception as e:
        print(f"[FAIL] Inspector simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_additional_attributes(samples):
    """Test 7: Check additional attributes that might be used"""
    print_section("TEST 7: Additional Attributes")
    
    if not samples:
        print("[SKIP] No samples to test")
        return False
    
    sample = samples[0]
    issues = []
    
    # Check tf_features
    if hasattr(sample, 'tf_features'):
        print(f"  tf_features: {len(sample.tf_features)} entries")
        if sample.tf_features:
            key = list(sample.tf_features.keys())[0]
            print(f"    Example: {key} = {sample.tf_features[key]}")
    else:
        issues.append("Missing tf_features attribute")
    
    # Check best_window
    if hasattr(sample, 'best_window'):
        print(f"  best_window: {sample.best_window}")
    else:
        issues.append("Missing best_window attribute")
    
    # Check bar_metadata
    if hasattr(sample, 'bar_metadata'):
        print(f"  bar_metadata: {len(sample.bar_metadata)} timeframes")
        if sample.bar_metadata:
            tf = list(sample.bar_metadata.keys())[0]
            print(f"    {tf}: {list(sample.bar_metadata[tf].keys())[:5]}...")
    else:
        issues.append("Missing bar_metadata attribute")
    
    # Check additional ChannelLabels attributes (source channel, next channel, RSI)
    window = list(sample.labels_per_window.keys())[0]
    tf = list(sample.labels_per_window[window]['tsla'].keys())[0]
    labels = sample.labels_per_window[window]['tsla'][tf]
    
    print(f"\n  Checking extended label attributes:")
    extended_attrs = [
        # Source channel
        'source_channel_slope', 'source_channel_intercept', 
        'source_channel_std_dev', 'source_channel_r_squared',
        'source_channel_bounce_count', 'source_channel_start_ts', 'source_channel_end_ts',
        # Next channel
        'best_next_channel_direction', 'best_next_channel_bars_away',
        'best_next_channel_duration', 'best_next_channel_r_squared',
        'shortest_next_channel_direction', 'small_channels_before_best',
        # RSI
        'rsi_at_first_break', 'rsi_at_permanent_break', 'rsi_at_channel_end',
        'rsi_overbought_at_break', 'rsi_oversold_at_break', 'rsi_divergence_at_break',
        # Exit events
        'exit_bars', 'exit_magnitudes', 'exit_durations', 'exit_types', 'exit_returned'
    ]
    
    for attr in extended_attrs:
        if hasattr(labels, attr):
            val = getattr(labels, attr)
            if isinstance(val, list):
                print(f"    [OK] {attr} = list({len(val)} items)")
            else:
                print(f"    [OK] {attr} = {val}")
        else:
            issues.append(f"Missing extended attribute: {attr}")
            print(f"    [MISSING] {attr}")
    
    if issues:
        print(f"\n[FAIL] Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n[PASS] All additional attributes present")
        return True

def main():
    print("\n" + "="*60)
    print(" INSPECTOR COMPATIBILITY TEST SUITE")
    print(" Binary Loader -> Inspector.py Integration")
    print("="*60)
    
    results = {}
    
    # Run tests
    samples = test_basic_loading()
    results['basic_loading'] = samples is not None
    
    results['timestamp'] = test_timestamp_compatibility(samples)
    results['channel_end_idx'] = test_channel_end_idx(samples)
    results['labels_structure'] = test_labels_per_window_structure(samples)
    results['consistency'] = test_all_samples_consistency(samples)
    results['simulation'] = test_inspector_simulation(samples)
    results['additional_attrs'] = test_additional_attributes(samples)
    
    # Summary
    print_section("SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "[PASS]" if passed_test else "[FAIL]"
        print(f"  {status} {test_name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  STATUS: FULLY COMPATIBLE")
        print("  Binary loader samples can be used directly with inspector.py")
    else:
        print("\n  STATUS: COMPATIBILITY ISSUES FOUND")
        print("  Some inspector functionality may not work correctly")
    
    return 0 if passed == total else 1

if __name__ == '__main__':
    sys.exit(main())
