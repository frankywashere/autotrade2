#!/usr/bin/env python3
"""
Test feature extraction with ATR to verify dimensions
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data_module import ChannelDataModule

def test_feature_extraction():
    """Test that features are extracted with correct dimensions including ATR"""
    
    print("=" * 60)
    print("Testing Feature Extraction with ATR")
    print("=" * 60)
    
    # Initialize data module
    print("\n1. Initializing data module...")
    dm = ChannelDataModule(
        data_path=str(project_root / "data" / "labeled_channels_v4.parquet"),
        window_size=20,
        step=50,
        batch_size=32,
        include_history=True,
        split_mode='time',
        train_end='2023-06-30',
        val_end='2023-09-30'
    )
    
    # Setup data
    print("2. Loading and preparing data...")
    dm.setup()
    
    # Get a batch
    print("3. Getting a training batch...")
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    # Extract features
    features = batch['features']
    print(f"\n4. Feature tensor shape: {features.shape}")
    print(f"   - Batch size: {features.shape[0]}")
    print(f"   - Sequence length: {features.shape[1]}")
    print(f"   - Feature dimension: {features.shape[2]}")
    
    # Verify dimension
    expected_dim = 809  # With ATR features
    actual_dim = features.shape[2]
    
    print(f"\n5. Dimension check:")
    print(f"   Expected (with ATR): {expected_dim}")
    print(f"   Actual: {actual_dim}")
    
    if actual_dim == expected_dim:
        print("   ✓ Dimension matches! ATR features included.")
    elif actual_dim == 776:
        print("   ✗ Dimension is 776 - ATR features MISSING!")
        return False
    else:
        print(f"   ✗ Unexpected dimension: {actual_dim}")
        return False
    
    # Check for non-zero ATR values
    print("\n6. Checking ATR values...")
    
    # ATR features should be in specific columns
    # For each timeframe (D/W/M), we have 11 ATR features at the end
    # Total: 33 ATR features (11 per timeframe)
    
    # Extract last 33 features (ATR columns)
    atr_features = features[0, :, -33:]  # First sample, all timesteps, last 33 features
    
    print(f"   ATR features shape: {atr_features.shape}")
    print(f"   ATR features min: {atr_features.min().item():.6f}")
    print(f"   ATR features max: {atr_features.max().item():.6f}")
    print(f"   ATR features mean: {atr_features.mean().item():.6f}")
    print(f"   ATR features std: {atr_features.std().item():.6f}")
    
    # Check if ATR values are non-zero
    non_zero_count = (atr_features != 0).sum().item()
    total_count = atr_features.numel()
    non_zero_pct = (non_zero_count / total_count) * 100
    
    print(f"   Non-zero ATR values: {non_zero_count}/{total_count} ({non_zero_pct:.1f}%)")
    
    if non_zero_pct > 50:
        print("   ✓ ATR features have non-zero values!")
    else:
        print("   ✗ Most ATR features are zero - possible issue!")
        return False
    
    print("\n" + "=" * 60)
    print("Feature extraction test PASSED! ✓")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_feature_extraction()
    sys.exit(0 if success else 1)
