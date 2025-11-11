"""
Quick diagnostic script to check training performance bottlenecks
"""
import torch
import psutil
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config

print("=" * 70)
print("TRAINING PERFORMANCE DIAGNOSTICS")
print("=" * 70)

# System info
print(f"\n1. SYSTEM CONFIGURATION:")
print(f"   CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
print(f"   RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
print(f"   Python workers configured: {config.NUM_WORKERS}")

# PyTorch config
print(f"\n2. PYTORCH CONFIGURATION:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   MPS available: {torch.backends.mps.is_available()}")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   Batch size: {config.ML_BATCH_SIZE}")
print(f"   Sequence length: {config.ML_SEQUENCE_LENGTH}")
print(f"   Mixed precision: {config.USE_MIXED_PRECISION}")

# DataLoader settings
print(f"\n3. DATALOADER SETTINGS:")
print(f"   num_workers: {config.NUM_WORKERS}")
print(f"   pin_memory: {config.PIN_MEMORY}")
print(f"   persistent_workers: {config.PERSISTENT_WORKERS}")
print(f"   prefetch_factor: {config.PREFETCH_FACTOR}")

# Device test
from src.ml.device_manager import DeviceManager
device_manager = DeviceManager()
device = device_manager.select_device_auto(verbose=False)
print(f"\n4. DEVICE SELECTION:")
print(f"   Selected device: {device}")
print(f"   Device type: {device.type}")

# Test tensor operations
print(f"\n5. PERFORMANCE TEST:")
print(f"   Creating test tensors...")

batch_size = config.ML_BATCH_SIZE
seq_len = config.ML_SEQUENCE_LENGTH
features = 50

# CPU test
print(f"\n   CPU Test (batch={batch_size}, seq={seq_len}):")
x_cpu = torch.randn(batch_size, seq_len, features)
start = time.time()
for _ in range(10):
    y = torch.matmul(x_cpu, x_cpu.transpose(1, 2))
cpu_time = (time.time() - start) / 10
print(f"   - Time per iteration: {cpu_time*1000:.2f}ms")

# GPU test
if device.type in ['cuda', 'mps']:
    print(f"\n   {device.type.upper()} Test (batch={batch_size}, seq={seq_len}):")
    x_gpu = x_cpu.to(device)

    # Warmup
    for _ in range(5):
        y = torch.matmul(x_gpu, x_gpu.transpose(1, 2))

    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(10):
        y = torch.matmul(x_gpu, x_gpu.transpose(1, 2))

    if device.type == 'cuda':
        torch.cuda.synchronize()

    gpu_time = (time.time() - start) / 10
    print(f"   - Time per iteration: {gpu_time*1000:.2f}ms")
    print(f"   - Speedup vs CPU: {cpu_time/gpu_time:.2f}x")

# Check if workers will actually work
print(f"\n6. MULTI-WORKER TEST:")
try:
    from torch.utils.data import DataLoader, TensorDataset
    dummy_data = TensorDataset(torch.randn(1000, seq_len, features), torch.randn(1000, 2))

    # Test with num_workers=0
    print(f"   Testing num_workers=0...")
    loader0 = DataLoader(dummy_data, batch_size=batch_size, num_workers=0)
    start = time.time()
    for i, (x, y) in enumerate(loader0):
        if i >= 10:
            break
    time0 = time.time() - start
    print(f"   - Time for 10 batches: {time0:.3f}s")

    # Test with num_workers=4
    print(f"   Testing num_workers=4...")
    loader4 = DataLoader(dummy_data, batch_size=batch_size, num_workers=4,
                         pin_memory=True, persistent_workers=True)
    start = time.time()
    for i, (x, y) in enumerate(loader4):
        if i >= 10:
            break
    time4 = time.time() - start
    print(f"   - Time for 10 batches: {time4:.3f}s")
    print(f"   - Speedup: {time0/time4:.2f}x")

    if time4 >= time0:
        print(f"   ⚠ WARNING: Multi-worker is NOT faster! This suggests:")
        print(f"     - Dataset may not be picklable")
        print(f"     - Overhead exceeds benefits for small batches")
        print(f"     - Try increasing batch size")
except Exception as e:
    print(f"   ✗ Multi-worker test failed: {e}")

print(f"\n7. RECOMMENDATIONS:")

# Check CPU usage
if config.NUM_WORKERS == 0:
    print(f"   ⚠ num_workers=0: Only using 1 CPU core!")
    print(f"     → Set NUM_WORKERS=8 in config.py")

# Check batch size
if config.ML_BATCH_SIZE < 128:
    print(f"   ⚠ Small batch size ({config.ML_BATCH_SIZE}): GPU underutilized")
    print(f"     → Increase ML_BATCH_SIZE to 256+ in config.py")

# Check mixed precision
if not config.USE_MIXED_PRECISION and device.type in ['cuda', 'mps']:
    print(f"   ⚠ Mixed precision disabled")
    print(f"     → Set USE_MIXED_PRECISION=True in config.py")

if config.NUM_WORKERS > 0 and config.ML_BATCH_SIZE >= 128 and config.USE_MIXED_PRECISION:
    print(f"   ✓ Configuration looks good!")
    print(f"   ✓ Should see 8+ CPU cores active during training")
    print(f"   ✓ GPU should be highly utilized")

print("\n" + "=" * 70)
