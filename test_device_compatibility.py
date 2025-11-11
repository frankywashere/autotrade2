#!/usr/bin/env python3
"""
Test script to verify device compatibility and GPU/Metal support
Tests: CPU, CUDA (if available), MPS (if available)
"""

import torch
import sys
from pathlib import Path
import time
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.model import LNNTradingModel, LSTMTradingModel
from src.ml.device_manager import DeviceManager


def test_tensor_operations(device_name: str):
    """Test basic tensor operations on device"""
    print(f"\n{'='*70}")
    print(f"🧪 Testing Tensor Operations on: {device_name}")
    print('='*70)

    results = []

    try:
        device = torch.device(device_name)

        # Test 1: Tensor creation and basic math
        print("  1. Tensor creation and basic operations...")
        try:
            a = torch.randn(1000, 1000).to(device)
            b = torch.randn(1000, 1000).to(device)
            c = a @ b
            d = torch.nn.functional.relu(c)
            results.append("✓ Basic tensor ops")
            print("     ✓ Success")
        except Exception as e:
            results.append(f"✗ Basic tensor ops: {e}")
            print(f"     ✗ Failed: {e}")
            return False

        # Test 2: Gradient computation
        print("  2. Gradient computation...")
        try:
            x = torch.randn(100, 100, requires_grad=True).to(device)
            y = (x ** 2).sum()
            y.backward()
            if x.grad is not None:
                results.append("✓ Gradient computation")
                print("     ✓ Success")
            else:
                results.append("✗ Gradient computation: No gradients")
                print("     ✗ Failed: No gradients computed")
        except Exception as e:
            results.append(f"✗ Gradient computation: {e}")
            print(f"     ✗ Failed: {e}")

        # Test 3: Neural network layers
        print("  3. Neural network layers...")
        try:
            layer = torch.nn.Linear(100, 50).to(device)
            input_tensor = torch.randn(32, 100).to(device)
            output = layer(input_tensor)
            loss = output.sum()
            loss.backward()
            results.append("✓ Neural network layers")
            print("     ✓ Success")
        except Exception as e:
            results.append(f"✗ Neural network layers: {e}")
            print(f"     ✗ Failed: {e}")

        # Test 4: Memory allocation
        print("  4. Large tensor allocation...")
        try:
            # Try to allocate ~400MB tensor
            large_tensor = torch.randn(10000, 10000).to(device)
            results.append("✓ Large tensor allocation")
            print("     ✓ Success (10000x10000 tensor)")
        except Exception as e:
            results.append(f"✗ Large tensor allocation: {e}")
            print(f"     ✗ Failed: {e}")

        print(f"\n  Summary for {device_name}:")
        for result in results:
            print(f"    {result}")

        return all("✓" in r for r in results)

    except Exception as e:
        print(f"\n  ❌ Device initialization failed: {e}")
        return False


def test_model_on_device(device_name: str, model_type: str = "LNN"):
    """Test LNN/LSTM model on specific device"""
    print(f"\n{'='*70}")
    print(f"🤖 Testing {model_type} Model on: {device_name}")
    print('='*70)

    try:
        device = torch.device(device_name)

        # Model parameters
        input_size = 50
        hidden_size = 32
        batch_size = 4
        seq_len = 84

        # Create model
        print(f"  Creating {model_type} model...")
        if model_type == "LNN":
            model = LNNTradingModel(input_size, hidden_size)
        else:
            model = LSTMTradingModel(input_size, hidden_size)

        model = model.to(device)
        print(f"  ✓ Model created and moved to {device}")

        # Create dummy data
        print(f"  Creating test data (batch={batch_size}, seq={seq_len}, features={input_size})...")
        x = torch.randn(batch_size, seq_len, input_size).to(device)
        print(f"  ✓ Data created on {device}")

        # Test 1: Forward pass
        print(f"  Testing forward pass...")
        start_time = time.time()
        predictions, hidden = model.forward(x)
        forward_time = time.time() - start_time
        print(f"  ✓ Forward pass successful ({forward_time:.3f}s)")
        print(f"    Output shape: {predictions.shape}")
        print(f"    Output device: {predictions.device}")

        # Test 2: Backward pass
        print(f"  Testing backward pass...")
        start_time = time.time()
        loss = predictions.sum()
        loss.backward()
        backward_time = time.time() - start_time
        print(f"  ✓ Backward pass successful ({backward_time:.3f}s)")

        # Test 3: Predict method
        print(f"  Testing predict method...")
        model.eval()
        with torch.no_grad():
            result = model.predict(x)
        print(f"  ✓ Predict method successful")
        print(f"    Predictions shape: {result['predictions'].shape}")
        print(f"    Confidence shape: {result['confidence'].shape}")

        # Test 4: Performance benchmark
        print(f"\n  Running performance benchmark (10 iterations)...")
        model.train()
        times = []
        for i in range(10):
            x_test = torch.randn(batch_size, seq_len, input_size).to(device)

            start = time.time()
            pred, _ = model.forward(x_test)
            loss = pred.sum()
            loss.backward()
            times.append(time.time() - start)

        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"  ✓ Benchmark complete")
        print(f"    Average iteration: {avg_time*1000:.1f}ms ± {std_time*1000:.1f}ms")
        print(f"    Throughput: {1/avg_time:.1f} iterations/sec")

        print(f"\n  ✅ All tests passed for {model_type} on {device_name}!")
        return True

    except Exception as e:
        print(f"\n  ❌ Test failed for {model_type} on {device_name}")
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_devices():
    """Compare performance across available devices"""
    print(f"\n{'='*70}")
    print(f"📊 DEVICE PERFORMANCE COMPARISON")
    print('='*70)

    dm = DeviceManager()
    available_devices = dm.get_available_devices()

    # Test parameters
    batch_size = 16
    seq_len = 84
    input_size = 50
    hidden_size = 128
    iterations = 20

    results = {}

    for device_name in available_devices.keys():
        if not available_devices[device_name]['available']:
            print(f"\n  Skipping {device_name} (not available)")
            continue

        print(f"\n  Testing {device_name}...")

        try:
            device = torch.device(device_name)
            model = LNNTradingModel(input_size, hidden_size).to(device)

            # Warmup
            x = torch.randn(batch_size, seq_len, input_size).to(device)
            for _ in range(3):
                pred, _ = model.forward(x)

            # Benchmark
            times = []
            for _ in range(iterations):
                x = torch.randn(batch_size, seq_len, input_size).to(device)

                start = time.time()
                pred, _ = model.forward(x)
                loss = pred.sum()
                loss.backward()
                times.append(time.time() - start)

            avg_time = np.mean(times) * 1000  # Convert to ms
            throughput = batch_size / np.mean(times)  # sequences/sec

            results[device_name] = {
                'avg_time_ms': avg_time,
                'throughput': throughput,
                'speedup': 1.0  # Will calculate relative to CPU
            }

            print(f"    Average time: {avg_time:.1f}ms")
            print(f"    Throughput: {throughput:.1f} sequences/sec")

        except Exception as e:
            print(f"    Failed: {e}")
            results[device_name] = None

    # Calculate speedup relative to CPU
    if 'cpu' in results and results['cpu']:
        cpu_time = results['cpu']['avg_time_ms']
        for device in results:
            if results[device]:
                results[device]['speedup'] = cpu_time / results[device]['avg_time_ms']

    # Print summary
    print(f"\n{'='*70}")
    print(f"📈 PERFORMANCE SUMMARY")
    print('='*70)
    print(f"\nConfiguration:")
    print(f"  Model: LNN with {hidden_size} hidden units")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Features: {input_size}")

    print(f"\nResults:")
    for device, result in results.items():
        if result:
            print(f"\n  {device.upper()}:")
            print(f"    Time per batch: {result['avg_time_ms']:.1f}ms")
            print(f"    Throughput: {result['throughput']:.1f} seq/sec")
            print(f"    Speedup vs CPU: {result['speedup']:.2f}x")
        else:
            print(f"\n  {device.upper()}: Failed")

    return results


def main():
    print("\n" + "="*70)
    print("🧪 DEVICE COMPATIBILITY TEST SUITE")
    print("="*70)

    # Initialize device manager
    dm = DeviceManager()

    # 1. Show hardware detection
    print("\n" + "="*70)
    print("1️⃣  HARDWARE DETECTION")
    print("="*70)
    hw_info = dm.detect_hardware()

    print("\nSystem Information:")
    for key, value in hw_info.items():
        if not key.endswith('_error') and not key.endswith('_warning'):
            print(f"  {key}: {value}")

    if 'cuda_error' in hw_info:
        print(f"\n  ⚠️  CUDA Issue: {hw_info['cuda_error']}")
    if 'mps_error' in hw_info:
        print(f"\n  ⚠️  MPS Issue: {hw_info['mps_error']}")
    if 'mps_warning' in hw_info:
        print(f"\n  ⚠️  MPS Warning: {hw_info['mps_warning']}")

    # 2. Test available devices
    print("\n" + "="*70)
    print("2️⃣  DEVICE AVAILABILITY")
    print("="*70)

    available = dm.get_available_devices()
    for device_name, info in available.items():
        if info['available']:
            print(f"\n  {device_name.upper()}: ✅ Available")
            if info['tensor_creation'] and info['forward_pass'] and info['backward_pass']:
                print(f"    All operations supported")
            else:
                if not info['tensor_creation']:
                    print(f"    ⚠️  Tensor creation failed")
                if not info['forward_pass']:
                    print(f"    ⚠️  Forward pass failed")
                if not info['backward_pass']:
                    print(f"    ⚠️  Backward pass failed")
            if info['errors']:
                for error in info['errors']:
                    print(f"    ⚠️  {error}")
        else:
            print(f"\n  {device_name.upper()}: ❌ Not available")
            if info['errors']:
                for error in info['errors']:
                    print(f"    {error}")

    # 3. Test tensor operations
    print("\n" + "="*70)
    print("3️⃣  TENSOR OPERATIONS TEST")
    print("="*70)

    test_results = {}
    for device_name in available.keys():
        if available[device_name]['available']:
            test_results[device_name] = test_tensor_operations(device_name)

    # 4. Test models
    print("\n" + "="*70)
    print("4️⃣  MODEL COMPATIBILITY TEST")
    print("="*70)

    model_results = {}
    for device_name in available.keys():
        if available[device_name]['available'] and available[device_name]['forward_pass']:
            # Test LNN
            lnn_result = test_model_on_device(device_name, "LNN")
            # Test LSTM
            lstm_result = test_model_on_device(device_name, "LSTM")
            model_results[device_name] = {
                'LNN': lnn_result,
                'LSTM': lstm_result
            }

    # 5. Performance benchmark
    print("\n" + "="*70)
    print("5️⃣  PERFORMANCE BENCHMARK")
    print("="*70)
    benchmark_results = benchmark_devices()

    # 6. Final summary
    print("\n" + "="*70)
    print("📊 FINAL TEST SUMMARY")
    print("="*70)

    print("\nDevice Compatibility:")
    for device in test_results:
        tensor_status = "✅" if test_results[device] else "❌"
        model_status = "✅" if device in model_results and all(model_results[device].values()) else "❌"
        print(f"  {device.upper()}:")
        print(f"    Tensor Operations: {tensor_status}")
        print(f"    Model Training: {model_status}")
        if device in benchmark_results and benchmark_results[device]:
            print(f"    Performance: {benchmark_results[device]['speedup']:.2f}x vs CPU")

    print("\n" + "="*70)
    print("✅ DEVICE COMPATIBILITY TEST COMPLETE")
    print("="*70)

    # Recommendation
    print("\n💡 RECOMMENDATION:")
    if 'cuda' in available and available['cuda']['available'] and available['cuda']['backward_pass']:
        print("  Use CUDA for best performance (NVIDIA GPU)")
    elif 'mps' in available and available['mps']['available'] and available['mps']['forward_pass']:
        print("  Use MPS for Apple Silicon acceleration (3-5x faster than CPU)")
        if not available['mps']['backward_pass']:
            print("  Note: Some operations may fallback to CPU")
    else:
        print("  Use CPU (no GPU acceleration available)")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()