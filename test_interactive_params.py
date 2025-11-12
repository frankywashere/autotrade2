#!/usr/bin/env python3
"""
Test script to verify the interactive parameter selection system
without running actual training.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.interactive_params import calculate_optimal_batch_size
from src.ml.device_manager import DeviceManager


def test_batch_size_calculation():
    """Test the batch size calculation function."""
    print("\n" + "=" * 70)
    print("📊 TESTING BATCH SIZE CALCULATION")
    print("=" * 70)

    # Test with different RAM amounts and devices
    test_cases = [
        (8.0, 'cpu', 84, 50, 128),    # 8GB RAM, CPU
        (16.0, 'mps', 84, 50, 128),   # 16GB RAM, MPS
        (32.0, 'cuda', 84, 50, 256),  # 32GB RAM, CUDA, larger model
        (4.0, 'cpu', 168, 50, 64),    # 4GB RAM, longer sequence
    ]

    for available_ram, device_type, seq_len, features, hidden in test_cases:
        suggestions = calculate_optimal_batch_size(
            device_type=device_type,
            available_ram_gb=available_ram,
            sequence_length=seq_len,
            num_features=features,
            hidden_size=hidden
        )

        print(f"\nTest case:")
        print(f"  RAM: {available_ram} GB")
        print(f"  Device: {device_type}")
        print(f"  Sequence: {seq_len}")
        print(f"  Features: {features}")
        print(f"  Hidden size: {hidden}")
        print(f"\nSuggestions:")
        print(f"  Conservative: {suggestions['conservative']}")
        print(f"  Balanced: {suggestions['balanced']}")
        print(f"  Aggressive: {suggestions['aggressive']}")
        print(f"  Memory per sample: {suggestions['memory_per_sample_mb']:.2f} MB")
        print(f"  Max safe: {suggestions['max_safe']}")


def test_ram_detection():
    """Test RAM detection capabilities."""
    print("\n" + "=" * 70)
    print("💾 TESTING RAM DETECTION")
    print("=" * 70)

    dm = DeviceManager()

    try:
        total_ram = dm.get_system_ram_gb()
        available_ram = dm.get_available_ram_gb()

        print(f"\nSystem RAM:")
        print(f"  Total: {total_ram:.1f} GB")
        print(f"  Available: {available_ram:.1f} GB")
        print(f"  Used: {total_ram - available_ram:.1f} GB")
        print(f"  Usage: {(1 - available_ram/total_ram)*100:.1f}%")
        print("\n✓ RAM detection working!")
    except Exception as e:
        print(f"\n✗ RAM detection failed: {e}")


def test_device_detection():
    """Test device detection capabilities."""
    print("\n" + "=" * 70)
    print("🖥️  TESTING DEVICE DETECTION")
    print("=" * 70)

    dm = DeviceManager()
    hardware_info = dm.detect_hardware()

    print(f"\nHardware detected:")
    print(f"  Platform: {hardware_info['platform']}")
    print(f"  Processor: {hardware_info['processor']}")
    print(f"  PyTorch: {hardware_info['pytorch_version']}")

    if hardware_info.get('cuda_available'):
        print(f"  CUDA: ✓ Available")
        print(f"    Device: {hardware_info.get('cuda_device_name', 'Unknown')}")
    else:
        print(f"  CUDA: ✗ Not available")

    if hardware_info.get('mps_available'):
        print(f"  MPS: ✓ Available")
        print(f"    Chip: {hardware_info.get('apple_chip', 'Unknown')}")
    else:
        print(f"  MPS: ✗ Not available")

    # Get available devices
    devices = dm.get_available_devices()
    print(f"\nAvailable devices for training:")
    for device_name, info in devices.items():
        if info['available']:
            print(f"  - {device_name}")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("🧪 INTERACTIVE PARAMETER SYSTEM TEST")
    print("=" * 70)
    print("\nThis test verifies the components without running full interactive mode.")

    test_ram_detection()
    test_device_detection()
    test_batch_size_calculation()

    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 70)
    print("\nTo test the full interactive mode, run:")
    print("  python train_model.py --interactive")
    print("  python train_model_lazy.py --interactive")
    print("\nYou can cancel at any time with Ctrl+C")
    print("=" * 70)


if __name__ == '__main__':
    main()