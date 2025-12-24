#!/usr/bin/env python3
"""
Test script for Mamba2 integration.
Run this to verify Mamba2 can be instantiated and perform forward passes.

Usage:
    python test_mamba.py

Expected output:
    - MambaWrapper initialization success
    - Forward pass success with correct output shapes
    - HierarchicalLNN with Mamba initialization success
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_mamba_wrapper():
    """Test MambaWrapper standalone."""
    print("=" * 60)
    print("Testing MambaWrapper (standalone)")
    print("=" * 60)

    try:
        from src.ml.mamba_wrapper import MambaWrapper, find_optimal_chunk_size
        print("✓ MambaWrapper import successful")
    except ImportError as e:
        print(f"✗ MambaWrapper import failed: {e}")
        return False

    # Test chunk size calculation
    print("\nChunk size calculations:")
    for seq_len in [75, 90, 128, 200, 300]:
        chunk = find_optimal_chunk_size(seq_len)
        print(f"  seq_len={seq_len} -> chunk_size={chunk} (chunks={seq_len//chunk})")

    # Test initialization
    print("\nInitializing MambaWrapper...")
    try:
        wrapper = MambaWrapper(
            input_size=1392,  # Same as your CfC input size
            units=128,        # Hidden size
            d_state=128,
            d_conv=4,
            expand=2,
            n_layers=1,
        )
        print(f"✓ MambaWrapper created")
        print(f"  Parameters: {sum(p.numel() for p in wrapper.parameters()):,}")
    except Exception as e:
        print(f"✗ MambaWrapper creation failed: {e}")
        return False

    # Test forward pass
    print("\nTesting forward pass...")
    try:
        batch_size = 2
        seq_len = 75
        input_size = 1392

        x = torch.randn(batch_size, seq_len, input_size)
        output, hidden = wrapper(x)

        print(f"✓ Forward pass successful")
        print(f"  Input shape:  {list(x.shape)}")
        print(f"  Output shape: {list(output.shape)}")
        print(f"  Hidden shape: {list(hidden.shape)}")

        assert output.shape == (batch_size, seq_len, 128), f"Output shape mismatch: {output.shape}"
        assert hidden.shape == (batch_size, 128), f"Hidden shape mismatch: {hidden.shape}"
        print(f"✓ Shape assertions passed")

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_hierarchical_lnn_mamba():
    """Test HierarchicalLNN with Mamba enabled."""
    print("\n" + "=" * 60)
    print("Testing HierarchicalLNN with use_mamba=True")
    print("=" * 60)

    try:
        from src.ml.hierarchical_model import HierarchicalLNN, HAS_MAMBA
        print(f"✓ HierarchicalLNN import successful")
        print(f"  HAS_MAMBA = {HAS_MAMBA}")
    except ImportError as e:
        print(f"✗ HierarchicalLNN import failed: {e}")
        return False

    if not HAS_MAMBA:
        print("✗ Mamba not available in hierarchical_model (HAS_MAMBA=False)")
        return False

    # Create input_sizes dict matching your config
    input_sizes = {
        '5min': 1104, '15min': 1104, '30min': 1104,
        '1h': 1104, '2h': 1104, '3h': 1104, '4h': 1104,
        'daily': 1104, 'weekly': 1104, 'monthly': 1104, '3month': 1104
    }

    print("\nInitializing HierarchicalLNN with Mamba...")
    try:
        model = HierarchicalLNN(
            input_sizes=input_sizes,
            hidden_size=128,
            use_mamba=True,
            device='cpu',
        )
        print(f"✓ HierarchicalLNN created with Mamba")
        print(f"  use_mamba = {model.use_mamba}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
    except Exception as e:
        print(f"✗ HierarchicalLNN creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test forward pass with dummy data
    print("\nTesting forward pass with dummy data...")
    try:
        batch_size = 2
        seq_len = 75

        # Create timeframe data dict
        timeframe_data = {}
        for tf in model.TIMEFRAMES:
            tf_features = input_sizes.get(tf, 1104)
            timeframe_data[tf] = torch.randn(batch_size, seq_len, tf_features)

        # Create VIX and event data
        vix_sequence = torch.randn(batch_size, 90, 11)
        event_data = torch.zeros(batch_size, 6)

        output = model(
            timeframe_data,
            vix_sequence=vix_sequence,
            event_data=event_data,
        )

        print(f"✓ Forward pass successful")
        print(f"  Output keys: {list(output.keys())[:5]}...")

        if 'predictions' in output:
            print(f"  Predictions shape: {list(output['predictions'].shape)}")

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def compare_cfc_vs_mamba():
    """Compare CfC and Mamba parameter counts."""
    print("\n" + "=" * 60)
    print("Comparing CfC vs Mamba parameter counts")
    print("=" * 60)

    input_sizes = {
        '5min': 1104, '15min': 1104, '30min': 1104,
        '1h': 1104, '2h': 1104, '3h': 1104, '4h': 1104,
        'daily': 1104, 'weekly': 1104, 'monthly': 1104, '3month': 1104
    }

    try:
        from src.ml.hierarchical_model import HierarchicalLNN

        # CfC model
        print("\nCreating CfC model...")
        model_cfc = HierarchicalLNN(
            input_sizes=input_sizes,
            hidden_size=128,
            use_mamba=False,
            device='cpu',
        )
        cfc_params = sum(p.numel() for p in model_cfc.parameters())
        print(f"  CfC parameters: {cfc_params:,}")

        # Mamba model
        print("Creating Mamba model...")
        model_mamba = HierarchicalLNN(
            input_sizes=input_sizes,
            hidden_size=128,
            use_mamba=True,
            device='cpu',
        )
        mamba_params = sum(p.numel() for p in model_mamba.parameters())
        print(f"  Mamba parameters: {mamba_params:,}")

        print(f"\nDifference: {mamba_params - cfc_params:+,} parameters")
        print(f"Ratio: {mamba_params / cfc_params:.2f}x")

    except Exception as e:
        print(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()


def benchmark_speed():
    """Simple speed comparison between CfC and Mamba."""
    print("\n" + "=" * 60)
    print("Speed benchmark (single forward pass)")
    print("=" * 60)

    import time

    input_sizes = {
        '5min': 1104, '15min': 1104, '30min': 1104,
        '1h': 1104, '2h': 1104, '3h': 1104, '4h': 1104,
        'daily': 1104, 'weekly': 1104, 'monthly': 1104, '3month': 1104
    }

    try:
        from src.ml.hierarchical_model import HierarchicalLNN

        batch_size = 4
        seq_len = 75

        # Create dummy data
        timeframe_data = {}
        for tf in HierarchicalLNN.TIMEFRAMES:
            tf_features = input_sizes.get(tf, 1104)
            timeframe_data[tf] = torch.randn(batch_size, seq_len, tf_features)
        vix_sequence = torch.randn(batch_size, 90, 11)
        event_data = torch.zeros(batch_size, 6)

        # Warmup and time CfC
        print("\nTiming CfC...")
        model_cfc = HierarchicalLNN(input_sizes=input_sizes, hidden_size=128, use_mamba=False, device='cpu')
        model_cfc.eval()

        with torch.no_grad():
            # Warmup
            _ = model_cfc(timeframe_data, vix_sequence=vix_sequence, event_data=event_data)

            start = time.time()
            for _ in range(5):
                _ = model_cfc(timeframe_data, vix_sequence=vix_sequence, event_data=event_data)
            cfc_time = (time.time() - start) / 5

        print(f"  CfC: {cfc_time*1000:.1f}ms per forward pass")

        # Warmup and time Mamba
        print("Timing Mamba...")
        model_mamba = HierarchicalLNN(input_sizes=input_sizes, hidden_size=128, use_mamba=True, device='cpu')
        model_mamba.eval()

        with torch.no_grad():
            # Warmup
            _ = model_mamba(timeframe_data, vix_sequence=vix_sequence, event_data=event_data)

            start = time.time()
            for _ in range(5):
                _ = model_mamba(timeframe_data, vix_sequence=vix_sequence, event_data=event_data)
            mamba_time = (time.time() - start) / 5

        print(f"  Mamba: {mamba_time*1000:.1f}ms per forward pass")
        print(f"\nSpeedup: {cfc_time/mamba_time:.2f}x {'faster' if mamba_time < cfc_time else 'slower'}")

    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print("Mamba2 Integration Test Suite")
    print("=" * 60)

    # Run tests
    wrapper_ok = test_mamba_wrapper()
    model_ok = test_hierarchical_lnn_mamba() if wrapper_ok else False

    if wrapper_ok and model_ok:
        compare_cfc_vs_mamba()
        benchmark_speed()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  MambaWrapper test:    {'PASS' if wrapper_ok else 'FAIL'}")
    print(f"  HierarchicalLNN test: {'PASS' if model_ok else 'FAIL'}")

    if wrapper_ok and model_ok:
        print("\n✓ All tests passed! Mamba2 integration is ready.")
        print("\nTo use Mamba2 in training:")
        print("  python train_hierarchical.py --use-mamba")
        print("  OR select 'Yes' when prompted in interactive mode")
    else:
        print("\n✗ Some tests failed. Check errors above.")
        sys.exit(1)
