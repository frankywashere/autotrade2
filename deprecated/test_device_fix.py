#!/usr/bin/env python3
"""
Quick test to verify the device mismatch fix for SelfSupervisedPretrainer
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.model import LNNTradingModel, SelfSupervisedPretrainer
from src.ml.device_manager import DeviceManager


def test_pretrainer_device_compatibility():
    """Test that SelfSupervisedPretrainer works on different devices"""

    print("="*70)
    print("Testing SelfSupervisedPretrainer Device Compatibility")
    print("="*70)

    # Initialize device manager
    dm = DeviceManager()

    # Get available devices
    devices = dm.get_available_devices()

    # Model parameters
    input_size = 50
    hidden_size = 32
    batch_size = 4
    seq_len = 84

    for device_name in devices.keys():
        if not devices[device_name]['available']:
            print(f"\nSkipping {device_name} (not available)")
            continue

        print(f"\n--- Testing on {device_name} ---")

        try:
            device = torch.device(device_name)

            # Create model and move to device
            model = LNNTradingModel(input_size, hidden_size)
            model = model.to(device)
            print(f"✓ Model created and moved to {device}")

            # Create pretrainer (this should now handle device correctly)
            pretrainer = SelfSupervisedPretrainer(model, mask_ratio=0.15)
            print(f"✓ Pretrainer created")

            # Check reconstruction head is on correct device
            reconstruction_device = next(pretrainer.reconstruction_head.parameters()).device
            print(f"  Reconstruction head device: {reconstruction_device}")

            # Create optimizer
            optimizer = torch.optim.Adam(
                list(model.parameters()) +
                list(pretrainer.reconstruction_head.parameters()),
                lr=0.001
            )

            # Create dummy data on device
            x = torch.randn(batch_size, seq_len, input_size).to(device)
            print(f"✓ Test data created on {device}")

            # Test pretrain step
            loss = pretrainer.pretrain_step(x, optimizer)
            print(f"✓ Pretrain step successful! Loss: {loss:.4f}")

            # Verify all components are on same device
            model_device = next(model.parameters()).device
            recon_device = next(pretrainer.reconstruction_head.parameters()).device

            if model_device == recon_device == device:
                print(f"✅ SUCCESS: All components on {device_name}")
            else:
                print(f"❌ WARNING: Device mismatch detected!")
                print(f"   Model: {model_device}")
                print(f"   Reconstruction: {recon_device}")
                print(f"   Expected: {device}")

        except Exception as e:
            print(f"❌ FAILED on {device_name}")
            print(f"   Error: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("Test Complete!")
    print("="*70)


if __name__ == '__main__':
    test_pretrainer_device_compatibility()