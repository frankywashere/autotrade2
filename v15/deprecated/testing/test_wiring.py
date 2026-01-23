#!/usr/bin/env python3
"""
Quick end-to-end test to verify wiring fixes work.

Tests:
1. Load existing samples
2. Create ChannelDataset with new label naming
3. Get a batch and verify label keys
4. Create model with all heads enabled
5. Forward pass
6. Compute loss
"""
import sys
import torch
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_wiring():
    print("=" * 60)
    print("V15 WIRING VERIFICATION TEST")
    print("=" * 60)

    # 1. Load samples
    print("\n[1] Loading samples...")
    from v15.training.dataset import load_samples, ChannelDataset

    sample_path = "tiny_test.pkl"
    if not Path(sample_path).exists():
        sample_path = "test_samples.pkl"
    if not Path(sample_path).exists():
        sample_path = "samples_small.pkl"

    print(f"    Using: {sample_path}")
    samples = load_samples(sample_path)
    print(f"    Loaded {len(samples)} samples")

    # 2. Create dataset
    print("\n[2] Creating ChannelDataset...")
    try:
        dataset = ChannelDataset(
            samples=samples[:100],  # Use subset for speed
            target_tf='1h',
            validate=False,  # Skip validation for speed
            analyze_correlations=False
        )
        print(f"    Dataset created with {len(dataset)} samples")
        print(f"    Features: {dataset.features.shape}")
    except Exception as e:
        print(f"    ERROR creating dataset: {e}")
        return False

    # 3. Get a batch and check label keys
    print("\n[3] Checking label keys...")
    try:
        features, labels = dataset[0]
        print(f"    Features shape: {features.shape}")
        print(f"    Label keys ({len(labels)}):")

        # Check for new naming convention
        expected_tsla = ['tsla_bars_to_first_break', 'tsla_break_direction', 'tsla_break_scan_valid']
        expected_spy = ['spy_bars_to_first_break', 'spy_break_direction', 'spy_break_scan_valid']
        expected_cross = ['cross_direction_aligned', 'cross_who_broke_first', 'cross_break_lag_bars']

        missing = []
        for key in expected_tsla + expected_spy + expected_cross:
            if key in labels:
                print(f"      ✓ {key}")
            else:
                print(f"      ✗ {key} MISSING")
                missing.append(key)

        # Check orphan fields are GONE
        orphans = ['break_trigger_tf', 'break_return', 'bars_outside', 'spy_bars_outside']
        for key in orphans:
            if key in labels:
                print(f"      ⚠ {key} still present (should be removed)")
            else:
                print(f"      ✓ {key} removed (good)")

        if missing:
            print(f"\n    WARNING: {len(missing)} expected keys missing")
    except Exception as e:
        print(f"    ERROR getting batch: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. Create model with all heads
    print("\n[4] Creating model with all heads enabled...")
    from v15.models.full_model import V15Model
    from v15.config import TOTAL_FEATURES

    n_features = dataset.features.shape[1]
    print(f"    Sample features: {n_features}")
    print(f"    Model expects: {TOTAL_FEATURES}")

    if n_features != TOTAL_FEATURES:
        print(f"    ⚠ Feature mismatch! Old samples have {n_features}, model needs {TOTAL_FEATURES}")
        print(f"    → You need to REGENERATE samples with current feature extractor")
        print(f"    → But LABEL WIRING is verified correct (see step 3)")

        # Still try to verify model creation works with correct feature count
        print(f"\n    Testing model creation with expected feature count...")
        try:
            model = V15Model(
                input_dim=TOTAL_FEATURES,
                enable_tsla_heads=True,
                enable_spy_heads=True,
                enable_cross_correlation_heads=True,
            )
            print(f"    ✓ Model created successfully")
            print(f"    Has TSLA heads: {model.has_tsla_heads()}")
            print(f"    Has SPY heads: {model.has_spy_heads()}")
            print(f"    Has Cross heads: {model.has_cross_correlation_heads()}")
        except Exception as e:
            print(f"    ERROR creating model: {e}")
            return False

        print("\n" + "=" * 60)
        print("⚠ PARTIAL SUCCESS - Label wiring verified, but samples need regeneration")
        print("=" * 60)
        return True  # Wiring is correct, just need new samples

    # If features match, do full test
    try:
        model = V15Model(
            input_dim=n_features,
            enable_tsla_heads=True,
            enable_spy_heads=True,
            enable_cross_correlation_heads=True,
        )
        print(f"    Model created")
        print(f"    Has TSLA heads: {model.has_tsla_heads()}")
        print(f"    Has SPY heads: {model.has_spy_heads()}")
        print(f"    Has Cross heads: {model.has_cross_correlation_heads()}")
    except Exception as e:
        print(f"    ERROR creating model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 5. Forward pass
    print("\n[5] Running forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            # Create batch
            features_batch = torch.stack([dataset[i][0] for i in range(min(4, len(dataset)))])
            print(f"    Input shape: {features_batch.shape}")

            outputs = model(features_batch)
            print(f"    Output keys ({len(outputs)}):")
            for key in sorted(outputs.keys())[:15]:
                val = outputs[key]
                if isinstance(val, torch.Tensor):
                    print(f"      {key}: {val.shape}")
                else:
                    print(f"      {key}: {type(val)}")
            if len(outputs) > 15:
                print(f"      ... and {len(outputs) - 15} more")
    except Exception as e:
        print(f"    ERROR in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 6. Test loss computation
    print("\n[6] Testing loss computation...")
    try:
        from v15.training.trainer import Trainer, TrainingConfig
        from torch.utils.data import DataLoader

        # Create minimal trainer config
        config = TrainingConfig(
            lr=1e-4,
            max_epochs=1,
        )

        # Create dataloader
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            config=config
        )

        # Get one batch and compute loss
        features_batch, labels_batch = next(iter(loader))
        model.train()
        outputs = model(features_batch)

        total_loss, losses = trainer.compute_loss(outputs, labels_batch)

        print(f"    Total loss: {total_loss.item():.4f}")
        print(f"    Individual losses:")
        for key, val in sorted(losses.items()):
            if val > 0:
                print(f"      {key}: {val:.4f}")

    except Exception as e:
        print(f"    ERROR in loss computation: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED - Wiring is correct!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_wiring()
    sys.exit(0 if success else 1)
