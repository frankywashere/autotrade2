"""
Quick test to verify the training pipeline works.

Tests:
1. Data loading
2. Dataset creation
3. Batch collation
4. Model forward pass
5. Loss calculation

Run this before starting full training to catch issues early.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import load_market_data, scan_valid_channels, ChannelDataset, collate_fn
from training.trainer import MultiTaskLoss


class DummyModel(nn.Module):
    """Minimal model for testing."""

    def __init__(self, input_dim=300):
        super().__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.duration_head = nn.Linear(128, 1)
        self.break_direction_head = nn.Linear(128, 2)
        self.new_direction_head = nn.Linear(128, 3)
        self.permanent_break_head = nn.Linear(128, 1)

    def forward(self, features):
        # Concatenate all features
        x = torch.cat([v for v in features.values()], dim=1)
        x = self.fc(x)
        return {
            'duration': self.duration_head(x),
            'break_direction': self.break_direction_head(x),
            'new_direction': self.new_direction_head(x),
            'permanent_break': self.permanent_break_head(x)
        }


def test_pipeline():
    """Run quick pipeline test."""
    print("=" * 80)
    print("TRAINING PIPELINE TEST")
    print("=" * 80)

    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"

    # Test 1: Load data
    print("\n[1/6] Testing data loading...")
    try:
        tsla_df, spy_df, vix_df = load_market_data(
            data_dir,
            start_date="2022-01-01",
            end_date="2022-01-31"  # Just 1 month for quick test
        )
        print(f"  TSLA: {len(tsla_df)} bars")
        print(f"  SPY: {len(spy_df)} bars")
        print(f"  VIX: {len(vix_df)} bars")
        print("  ✓ Data loading works")
    except Exception as e:
        print(f"  ✗ Data loading failed: {e}")
        return False

    # Test 2: Scan channels
    print("\n[2/6] Testing channel scanning...")
    try:
        samples = scan_valid_channels(
            tsla_df,
            spy_df,
            vix_df,
            window=50,
            step=50,  # Large step for speed
            min_cycles=1,
            max_scan=100,  # Small scan for speed
            include_history=False,
            progress=False
        )
        print(f"  Found {len(samples)} valid channels")
        if len(samples) > 0:
            print("  ✓ Channel scanning works")
        else:
            print("  ✗ No channels found (might need more data)")
            return False
    except Exception as e:
        print(f"  ✗ Channel scanning failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Create dataset
    print("\n[3/6] Testing dataset creation...")
    try:
        dataset = ChannelDataset(samples[:10], augment=False)  # First 10 samples
        print(f"  Dataset size: {len(dataset)}")
        print("  ✓ Dataset creation works")
    except Exception as e:
        print(f"  ✗ Dataset creation failed: {e}")
        return False

    # Test 4: Get single sample
    print("\n[4/6] Testing single sample loading...")
    try:
        features, labels = dataset[0]
        print(f"  Features keys: {list(features.keys())[:5]}... ({len(features)} total)")
        print(f"  Labels keys: {list(labels.keys())}")
        print(f"  Sample feature shape: {features['tsla_5min'].shape}")
        print("  ✓ Sample loading works")
    except Exception as e:
        print(f"  ✗ Sample loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 5: Batch collation
    print("\n[5/6] Testing batch collation...")
    try:
        batch = [dataset[i] for i in range(min(4, len(dataset)))]
        batched_features, batched_labels = collate_fn(batch)
        print(f"  Batch size: {len(batch)}")
        print(f"  Batched feature shape: {batched_features['tsla_5min'].shape}")
        print(f"  Batched label shape: {batched_labels['duration_bars'].shape}")
        print("  ✓ Batch collation works")
    except Exception as e:
        print(f"  ✗ Batch collation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 6: Model forward pass and loss
    print("\n[6/6] Testing model forward pass and loss...")
    try:
        # Infer input dimension
        total_dim = sum(v.shape[-1] for v in batched_features.values())
        model = DummyModel(input_dim=total_dim)

        # Forward pass
        predictions = model(batched_features)
        print(f"  Predictions keys: {list(predictions.keys())}")
        print(f"  Duration pred shape: {predictions['duration'].shape}")
        print(f"  Break direction pred shape: {predictions['break_direction'].shape}")

        # Calculate loss
        criterion = MultiTaskLoss()
        loss, loss_dict = criterion(predictions, batched_labels)
        print(f"  Total loss: {loss.item():.4f}")
        print(f"  Loss components: {loss_dict}")

        # Backward pass
        loss.backward()
        print("  ✓ Model forward/backward works")

    except Exception as e:
        print(f"  ✗ Model/loss failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # All tests passed
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
    print("\nThe training pipeline is ready to use!")
    print("Run example_training.py to start full training.")

    return True


if __name__ == '__main__':
    success = test_pipeline()
    sys.exit(0 if success else 1)
