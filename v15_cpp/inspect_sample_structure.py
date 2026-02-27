#!/usr/bin/env python3
"""
Inspect the complete structure of samples loaded from binary file.
"""

import sys
sys.path.insert(0, '/Users/frank/Desktop/CodingProjects/x14/v15_cpp')

from load_samples import load_samples, ChannelSample, ChannelLabels

def inspect_structure(filename: str):
    """Load samples and print complete structure of first sample."""
    
    print("=" * 70)
    print(f"Loading: {filename}")
    print("=" * 70)
    
    version, num_samples, num_features, samples = load_samples(filename)
    
    print(f"\nFile Header:")
    print(f"  version: {version}")
    print(f"  num_samples: {num_samples}")
    print(f"  num_features: {num_features}")
    print(f"  actual loaded: {len(samples)}")
    
    if not samples:
        print("\nNo samples loaded!")
        return
    
    sample = samples[0]
    
    print("\n" + "=" * 70)
    print("SAMPLE STRUCTURE (first sample)")
    print("=" * 70)
    
    # Top-level attributes
    print("\n1. TOP-LEVEL ATTRIBUTES:")
    print(f"   sample.timestamp: {sample.timestamp} (type: {type(sample.timestamp).__name__})")
    print(f"   sample.channel_end_idx: {sample.channel_end_idx} (type: {type(sample.channel_end_idx).__name__})")
    print(f"   sample.best_window: {sample.best_window} (type: {type(sample.best_window).__name__})")
    print(f"   sample.tf_features: dict with {len(sample.tf_features)} keys")
    print(f"   sample.labels_per_window: dict with {len(sample.labels_per_window)} keys")
    print(f"   sample.bar_metadata: dict with {len(sample.bar_metadata)} keys")
    
    # tf_features structure
    print("\n2. TF_FEATURES STRUCTURE:")
    print(f"   Type: {type(sample.tf_features).__name__}")
    if sample.tf_features:
        feature_keys = list(sample.tf_features.keys())
        print(f"   Total keys: {len(feature_keys)}")
        print(f"   First 10 keys: {feature_keys[:10]}")
        print(f"   Sample values:")
        for key in feature_keys[:5]:
            print(f"      {key}: {sample.tf_features[key]} (type: {type(sample.tf_features[key]).__name__})")
    
    # labels_per_window structure
    print("\n3. LABELS_PER_WINDOW STRUCTURE:")
    print(f"   Type: {type(sample.labels_per_window).__name__}")
    print(f"   Window keys: {list(sample.labels_per_window.keys())}")
    
    for window_key, tf_dict in sample.labels_per_window.items():
        print(f"\n   Window {window_key}:")
        print(f"      Type: {type(tf_dict).__name__}")
        print(f"      Timeframe keys: {list(tf_dict.keys())}")
        
        for tf_key, labels in tf_dict.items():
            print(f"\n      Timeframe '{tf_key}':")
            print(f"         Type: {type(labels).__name__}")
            
            # Get all attributes of ChannelLabels
            attrs = [attr for attr in dir(labels) if not attr.startswith('_')]
            print(f"         Total attributes: {len(attrs)}")
            
            # Print ALL ChannelLabels fields with their values
            print(f"\n         ALL ChannelLabels fields:")
            for attr in attrs:
                value = getattr(labels, attr)
                if not callable(value):
                    print(f"            {attr}: {value} ({type(value).__name__})")
            
            # Only show first timeframe as example
            break
        # Only show first window as example
        break
    
    # bar_metadata structure
    print("\n4. BAR_METADATA STRUCTURE:")
    print(f"   Type: {type(sample.bar_metadata).__name__}")
    print(f"   Timeframe keys: {list(sample.bar_metadata.keys())}")
    
    for tf_key, meta_dict in sample.bar_metadata.items():
        print(f"\n   Timeframe '{tf_key}':")
        print(f"      Type: {type(meta_dict).__name__}")
        print(f"      Metadata keys: {list(meta_dict.keys())}")
        print(f"      Sample values:")
        for meta_key, value in list(meta_dict.items())[:5]:
            print(f"         {meta_key}: {value} ({type(value).__name__})")
        # Only show first timeframe as example
        break
    
    # Summary of all windows and timeframes
    print("\n5. COMPLETE LABELS_PER_WINDOW HIERARCHY:")
    for window_key in sample.labels_per_window.keys():
        tfs = list(sample.labels_per_window[window_key].keys())
        print(f"   Window {window_key}: {len(tfs)} timeframes: {tfs}")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

if __name__ == '__main__':
    inspect_structure('/Users/frank/Desktop/CodingProjects/x14/v15_cpp/baseline_test.bin')
