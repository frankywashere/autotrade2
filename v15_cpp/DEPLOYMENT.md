# V15 C++ Scanner - Deployment Guide

**Version:** 1.0.0
**Date:** 2026-01-25
**Status:** Production Ready

## Overview

The V15 C++ Scanner is a high-performance trading system scanner that achieves 10-20x speedup over the Python baseline. This guide covers complete deployment from prerequisites to production use.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Build Instructions](#build-instructions)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [Performance Tuning](#performance-tuning)
7. [Troubleshooting](#troubleshooting)
8. [Production Deployment](#production-deployment)

---

## Prerequisites

### System Requirements

**Operating System:**
- macOS 11.0+ (Big Sur or later)
- Linux (Ubuntu 20.04+, RHEL 8+, or equivalent)
- Windows (WSL2 or native build)

**Hardware:**
- CPU: Modern x86-64 processor with AVX2 support (recommended)
- RAM: 8 GB minimum, 16 GB recommended
- Disk: 2 GB for build artifacts, 10+ GB for data

**Software Dependencies:**

| Component | Minimum Version | Recommended | Purpose |
|-----------|----------------|-------------|---------|
| CMake | 3.15 | 3.27+ | Build system |
| C++ Compiler | GCC 7, Clang 5, MSVC 2017 | GCC 11+, Clang 14+ | C++17 support |
| Python | 3.7 | 3.9+ | Bindings (optional) |
| Eigen3 | 3.3 | 3.4+ | Linear algebra |
| pybind11 | 2.6 | 2.11+ | Python bindings |
| OpenMP | Any | Latest | Parallelization |

### Installing Prerequisites

#### macOS (Homebrew)

```bash
# Install Xcode command line tools
xcode-select --install

# Install dependencies
brew install cmake eigen pybind11 libomp

# Verify installation
cmake --version
clang++ --version
python3 --version
```

#### Ubuntu/Debian

```bash
# Update package list
sudo apt-get update

# Install build tools
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-dev \
    python3-pip

# Install dependencies (optional - CMake will auto-fetch if missing)
sudo apt-get install -y \
    libeigen3-dev \
    pybind11-dev \
    libomp-dev

# Verify installation
cmake --version
g++ --version
python3 --version
```

#### RHEL/CentOS

```bash
# Enable EPEL repository
sudo yum install -y epel-release

# Install build tools
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake3 python3-devel

# Create cmake symlink
sudo ln -s /usr/bin/cmake3 /usr/bin/cmake
```

### Verification Checklist

Run these commands to verify your environment:

```bash
# CMake version >= 3.15
cmake --version

# C++17 compiler
g++ --version || clang++ --version

# Python 3.7+
python3 --version

# Git (for dependency fetching)
git --version
```

---

## Build Instructions

### Quick Build (Recommended)

```bash
# Navigate to project directory
cd /Users/frank/Desktop/CodingProjects/x14/v15_cpp

# Create build directory
mkdir -p build
cd build

# Configure with Release mode (optimized)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build with all available cores
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Expected output files:
#   - build/v15_scanner (standalone executable)
#   - build/libv15scanner.a (static library)
#   - build/v15scanner_py.so (Python module)
```

### Build Options

#### Release Build (Production)

Maximum performance, no debugging symbols:

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
      ..
make -j$(nproc)
```

**Optimization flags applied:**
- `-O3` - Maximum optimization
- `-march=native` - CPU-specific optimizations
- `-DNDEBUG` - Disable debug assertions
- `-flto` - Link-time optimization (if supported)

#### Debug Build (Development)

Debugging symbols, no optimizations:

```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)
```

**Debug flags applied:**
- `-g` - Full debugging symbols
- `-O0` - No optimizations
- `-Wall -Wextra -Wpedantic` - Comprehensive warnings

#### With Tests

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON ..
make -j$(nproc)
ctest --output-on-failure
```

#### Platform-Specific Builds

**macOS with Homebrew OpenMP:**

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DOpenMP_ROOT=$(brew --prefix libomp) \
      ..
```

**Linux with custom Eigen:**

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DEigen3_DIR=/usr/local/share/eigen3/cmake \
      ..
```

**Disable Python bindings:**

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_PYTHON_BINDINGS=OFF \
      ..
```

### Build Verification

After building, verify executables:

```bash
# Check scanner executable
./build/v15_scanner --version

# Check library
ls -lh build/libv15scanner.a

# Check Python module (if built)
python3 -c "import sys; sys.path.insert(0, 'build'); import v15scanner_py; print(v15scanner_py.__version__)"

# Run quick test
./build/v15_scanner --help
```

---

## Installation

### Option 1: System-Wide Installation (Recommended)

Install to `/usr/local`:

```bash
cd build
sudo cmake --install . --prefix /usr/local

# Verify installation
v15_scanner --version
python3 -c "import v15scanner_py; print(v15scanner_py.__version__)"
```

**Installed files:**
- `/usr/local/bin/v15_scanner` - Executable
- `/usr/local/lib/libv15scanner.a` - Static library
- `/usr/local/include/v15scanner/*.hpp` - Headers
- `/usr/local/python/v15scanner_py.so` - Python module

### Option 2: User Installation

Install to home directory:

```bash
cd build
cmake --install . --prefix ~/.local

# Add to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
echo 'export PYTHONPATH="$HOME/.local/python:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc

# Verify
v15_scanner --version
```

### Option 3: Standalone Usage

Use directly from build directory:

```bash
# Add to shell profile
echo 'export PATH="/Users/frank/Desktop/CodingProjects/x14/v15_cpp/build:$PATH"' >> ~/.bashrc
echo 'export PYTHONPATH="/Users/frank/Desktop/CodingProjects/x14/v15_cpp/build:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc

# Create symlink (alternative)
ln -s /Users/frank/Desktop/CodingProjects/x14/v15_cpp/build/v15_scanner /usr/local/bin/
```

### Python Package Installation

For Python integration:

```bash
# Navigate to Python bindings directory
cd python_bindings

# Build and install
./build.sh install

# Or use pip (if setup.py exists)
pip install -e .

# Verify
python3 -c "from v15_cpp.python_bindings import scan_channels_two_pass; print('Success')"
```

---

## Configuration

### Command-Line Options

The scanner accepts the following command-line arguments:

```
v15_scanner [OPTIONS]

Required Options:
  --data-dir PATH          Directory containing CSV data files (TSLA.csv, SPY.csv, VIX.csv)
  --output PATH            Output file path (.bin for binary, .pkl for pickle)

Optional Arguments:
  --step N                 Channel detection step (default: 10)
  --max-samples N          Maximum samples to generate (default: 0 = unlimited)
  --workers N              Number of worker threads (default: 0 = auto-detect)
  --batch-size N           Channels per batch (default: 8)
  --warmup-bars N          Warmup period in 5-min bars (default: 32760)
  --progress               Show progress bars (default: true)
  --verbose                Verbose logging (default: false)
  --strict                 Strict mode - fail on errors (default: false)
  --help                   Show this help message
  --version                Show version information
```

### Configuration File (Optional)

Create `scanner_config.json`:

```json
{
  "step": 10,
  "warmup_bars": 32760,
  "max_samples": 10000,
  "workers": 8,
  "batch_size": 8,
  "progress": true,
  "verbose": false,
  "strict": false,
  "windows": [10, 20, 30, 40, 50, 60, 70, 80],
  "timeframes": ["5m", "15m", "30m", "1h", "2h", "3h", "4h", "daily", "weekly", "monthly"]
}
```

Use with:

```bash
v15_scanner --config scanner_config.json --data-dir data --output samples.bin
```

### Environment Variables

```bash
# Number of threads (overrides --workers)
export V15_SCANNER_THREADS=8

# Default data directory
export V15_DATA_DIR=/path/to/data

# Verbose logging
export V15_VERBOSE=1

# Disable progress bars
export V15_NO_PROGRESS=1
```

---

## Usage Examples

### Example 1: Basic Scan

Scan with default parameters:

```bash
v15_scanner \
  --data-dir ../data \
  --output samples.bin \
  --step 10 \
  --max-samples 10000
```

### Example 2: High-Performance Scan

Maximize throughput with 8 workers:

```bash
v15_scanner \
  --data-dir ../data \
  --output samples.bin \
  --step 10 \
  --workers 8 \
  --batch-size 16 \
  --max-samples 100000 \
  --verbose
```

### Example 3: Python Integration

```python
import sys
sys.path.insert(0, '/Users/frank/Desktop/CodingProjects/x14/v15_cpp/build')

import pandas as pd
import v15scanner_py

# Load data
tsla = pd.read_csv('data/TSLA.csv', index_col=0, parse_dates=True)
spy = pd.read_csv('data/SPY.csv', index_col=0, parse_dates=True)
vix = pd.read_csv('data/VIX.csv', index_col=0, parse_dates=True)

# Configure scanner
config = v15scanner_py.ScannerConfig()
config.step = 10
config.workers = 8
config.max_samples = 10000

# Create scanner
scanner = v15scanner_py.Scanner(config)

# Run scan
samples = scanner.scan(tsla, spy, vix)

# Get statistics
stats = scanner.get_stats()
print(f"Generated {stats.samples_created} samples in {stats.total_duration_ms/1000:.2f}s")
print(f"Throughput: {stats.samples_per_second:.2f} samples/sec")
```

### Example 4: Batch Processing

Process multiple symbols:

```bash
#!/bin/bash
SYMBOLS="TSLA SPY AAPL MSFT GOOGL"

for symbol in $SYMBOLS; do
    echo "Processing $symbol..."
    v15_scanner \
      --data-dir data \
      --output samples_${symbol}.bin \
      --step 20 \
      --workers 8
done
```

### Example 5: Pipeline Integration

Integrate with existing pipeline:

```bash
# Generate samples
v15_scanner \
  --data-dir ../data \
  --output /tmp/samples.bin \
  --step 10 \
  --max-samples 10000

# Convert to pickle (if needed for Python compatibility)
python3 -c "
import pickle
import v15scanner_py

# Load binary samples
samples = v15scanner_py.load_samples('/tmp/samples.bin')

# Save as pickle
with open('/tmp/samples.pkl', 'wb') as f:
    pickle.dump(samples, f)

print(f'Converted {len(samples)} samples to pickle format')
"

# Use in Python pipeline
python3 train_model.py --samples /tmp/samples.pkl
```

---

## Performance Tuning

### Worker Thread Optimization

**Rule of thumb:**
- Workers = CPU cores for CPU-bound workloads
- Workers = 2x CPU cores for I/O-bound workloads

**Find optimal workers:**

```bash
for workers in 1 2 4 8 16; do
    echo "Testing with $workers workers..."
    time v15_scanner \
      --data-dir ../data \
      --output /tmp/test.bin \
      --workers $workers \
      --max-samples 1000
done
```

**Recommended settings:**

| CPU Cores | Workers | Use Case |
|-----------|---------|----------|
| 2-4 | 4 | Laptops, small datasets |
| 4-8 | 8 | Workstations, medium datasets |
| 8-16 | 16 | Servers, large datasets |
| 16+ | 32 | HPC, massive datasets |

### Memory Optimization

**Reduce memory usage:**

```bash
# Process in smaller batches
v15_scanner \
  --batch-size 4 \
  --max-samples 1000 \
  --data-dir ../data \
  --output samples.bin

# Use smaller warmup (if acceptable)
v15_scanner \
  --warmup-bars 16380 \
  --data-dir ../data \
  --output samples.bin
```

**Monitor memory:**

```bash
# Linux
/usr/bin/time -v v15_scanner --data-dir ../data --output samples.bin

# macOS
/usr/bin/time -l ./v15_scanner --data-dir ../data --output samples.bin
```

### CPU Optimization

**Enable all optimizations:**

```bash
# Rebuild with maximum optimizations
cd build
rm -rf *
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
      -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -ffast-math" \
      ..
make -j$(nproc)
```

**Platform-specific flags:**

**Intel CPUs:**
```bash
-DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -mavx2 -mfma"
```

**AMD CPUs:**
```bash
-DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -mavx2"
```

**Apple Silicon (M1/M2):**
```bash
-DCMAKE_CXX_FLAGS="-O3 -mcpu=apple-m1"
```

### I/O Optimization

**Use faster storage:**
- SSD > HDD
- NVMe > SATA SSD
- RAM disk for maximum speed

**Create RAM disk (macOS):**

```bash
# Create 4GB RAM disk
diskutil erasevolume HFS+ "RAMDISK" `hdiutil attach -nomount ram://8388608`

# Use for output
v15_scanner --data-dir ../data --output /Volumes/RAMDISK/samples.bin
```

**Create RAM disk (Linux):**

```bash
# Create 4GB tmpfs
sudo mkdir -p /mnt/ramdisk
sudo mount -t tmpfs -o size=4G tmpfs /mnt/ramdisk

# Use for output
v15_scanner --data-dir ../data --output /mnt/ramdisk/samples.bin
```

### Profiling

**Profile CPU usage:**

```bash
# Linux (perf)
perf record -g ./v15_scanner --data-dir ../data --output /tmp/samples.bin
perf report

# macOS (Instruments)
instruments -t "Time Profiler" ./v15_scanner --data-dir ../data --output /tmp/samples.bin
```

**Profile memory:**

```bash
# Linux (valgrind)
valgrind --tool=massif ./v15_scanner --data-dir ../data --output /tmp/samples.bin
ms_print massif.out.*

# macOS (leaks)
leaks --atExit -- ./v15_scanner --data-dir ../data --output /tmp/samples.bin
```

---

## Troubleshooting

### Build Issues

#### Problem: CMake version too old

```
Error: CMake 3.15 or higher is required. You are running version 3.10
```

**Solution:**

```bash
# Ubuntu - install from Kitware repository
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
sudo apt-get update
sudo apt-get install cmake

# macOS
brew upgrade cmake
```

#### Problem: Eigen3 not found

```
Error: Could not find Eigen3
```

**Solution:**

```bash
# CMake will auto-fetch Eigen3, but if it fails:

# macOS
brew install eigen

# Ubuntu
sudo apt-get install libeigen3-dev

# Or specify custom location
cmake -DEigen3_DIR=/path/to/eigen3/share/eigen3/cmake ..
```

#### Problem: C++17 not supported

```
Error: The compiler does not support C++17
```

**Solution:**

```bash
# Update compiler

# Ubuntu
sudo apt-get install g++-11
export CXX=g++-11
cmake ..

# macOS
xcode-select --install
```

### Runtime Issues

#### Problem: Segmentation fault

```
Segmentation fault (core dumped)
```

**Diagnosis:**

```bash
# Run with debugger
gdb --args ./v15_scanner --data-dir ../data --output samples.bin
(gdb) run
(gdb) backtrace

# Or run with debug build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
./v15_scanner --data-dir ../data --output samples.bin
```

#### Problem: Out of memory

```
terminate called after throwing an instance of 'std::bad_alloc'
```

**Solutions:**

```bash
# Reduce batch size
v15_scanner --batch-size 4 --data-dir ../data --output samples.bin

# Limit samples
v15_scanner --max-samples 1000 --data-dir ../data --output samples.bin

# Reduce workers
v15_scanner --workers 2 --data-dir ../data --output samples.bin

# Check available memory
free -h  # Linux
vm_stat  # macOS
```

#### Problem: Poor performance

```
Scanner is running slower than expected
```

**Checklist:**

```bash
# 1. Verify Release build
file ./v15_scanner | grep "not stripped"  # Should show optimized binary

# 2. Check CPU frequency
# Linux
cat /proc/cpuinfo | grep MHz

# macOS
sysctl -n hw.cpufrequency

# 3. Monitor during execution
# Linux
htop  # Watch CPU usage

# macOS
sudo powermetrics --sample-rate 1000

# 4. Verify OpenMP is enabled
./v15_scanner --version | grep OpenMP

# 5. Rebuild with optimizations
cd build && rm -rf * && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc)
```

### Data Issues

#### Problem: CSV parsing errors

```
Error: Failed to parse CSV file
```

**Solutions:**

```bash
# Verify CSV format
head -5 data/TSLA.csv
# Expected format:
# timestamp,open,high,low,close,volume
# 2015-01-01 09:30:00,100.0,101.0,99.0,100.5,1000000

# Check for BOM or encoding issues
file data/TSLA.csv
dos2unix data/*.csv  # Convert line endings if needed

# Validate data
python3 -c "
import pandas as pd
df = pd.read_csv('data/TSLA.csv', index_col=0, parse_dates=True)
print(f'Loaded {len(df)} rows')
print(df.head())
"
```

#### Problem: Insufficient data

```
Warning: Not enough bars for warmup period
```

**Solutions:**

```bash
# Reduce warmup period
v15_scanner --warmup-bars 10000 --data-dir ../data --output samples.bin

# Or ensure data has enough bars (default requires 32,760 5-min bars = ~114 days)
```

---

## Production Deployment

### Docker Container

Create `Dockerfile`:

```dockerfile
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libeigen3-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy source
COPY . /app
WORKDIR /app

# Build
RUN mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc) && \
    cmake --install . --prefix /usr/local

# Set entrypoint
ENTRYPOINT ["v15_scanner"]
CMD ["--help"]
```

Build and run:

```bash
# Build image
docker build -t v15_scanner:latest .

# Run scanner
docker run -v $(pwd)/data:/data -v $(pwd)/output:/output \
  v15_scanner:latest \
  --data-dir /data \
  --output /output/samples.bin \
  --workers 8
```

### Systemd Service

Create `/etc/systemd/system/v15_scanner.service`:

```ini
[Unit]
Description=V15 Scanner Service
After=network.target

[Service]
Type=oneshot
User=scanner
Group=scanner
WorkingDirectory=/opt/v15_scanner
ExecStart=/usr/local/bin/v15_scanner \
  --data-dir /data/market \
  --output /data/output/samples_%Y%m%d.bin \
  --workers 8 \
  --max-samples 100000
StandardOutput=journal
StandardError=journal
SyslogIdentifier=v15_scanner

[Install]
WantedBy=multi-user.target
```

Enable and run:

```bash
sudo systemctl daemon-reload
sudo systemctl enable v15_scanner.service
sudo systemctl start v15_scanner.service
sudo systemctl status v15_scanner.service
```

### Cron Job

Add to crontab:

```bash
# Run daily at 6 PM
0 18 * * * /usr/local/bin/v15_scanner --data-dir /data/market --output /data/output/samples_$(date +\%Y\%m\%d).bin --workers 8 2>&1 | logger -t v15_scanner
```

### Kubernetes Deployment

Create `k8s-deployment.yaml`:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: v15-scanner
spec:
  schedule: "0 18 * * *"  # Daily at 6 PM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: scanner
            image: v15_scanner:latest
            args:
            - --data-dir
            - /data
            - --output
            - /output/samples.bin
            - --workers
            - "8"
            volumeMounts:
            - name: data
              mountPath: /data
            - name: output
              mountPath: /output
            resources:
              requests:
                memory: "8Gi"
                cpu: "4"
              limits:
                memory: "16Gi"
                cpu: "8"
          volumes:
          - name: data
            persistentVolumeClaim:
              claimName: market-data-pvc
          - name: output
            persistentVolumeClaim:
              claimName: scanner-output-pvc
          restartPolicy: OnFailure
```

Deploy:

```bash
kubectl apply -f k8s-deployment.yaml
kubectl get cronjobs
kubectl logs -f job/v15-scanner-xxxxx
```

### Monitoring

**Log aggregation:**

```bash
# Configure logging to file
v15_scanner \
  --data-dir ../data \
  --output samples.bin \
  --verbose 2>&1 | tee scanner.log

# Or use syslog
v15_scanner --data-dir ../data --output samples.bin 2>&1 | logger -t v15_scanner
```

**Metrics collection:**

```python
# Prometheus exporter example
from prometheus_client import Gauge, start_http_server
import v15scanner_py

samples_generated = Gauge('scanner_samples_total', 'Total samples generated')
scan_duration = Gauge('scanner_duration_seconds', 'Scan duration in seconds')
throughput = Gauge('scanner_throughput', 'Samples per second')

# Run scanner
scanner = v15scanner_py.Scanner(config)
samples = scanner.scan(tsla, spy, vix)
stats = scanner.get_stats()

# Update metrics
samples_generated.set(stats.samples_created)
scan_duration.set(stats.total_duration_ms / 1000)
throughput.set(stats.samples_per_second)

# Expose metrics
start_http_server(8000)
```

### Health Checks

```bash
#!/bin/bash
# healthcheck.sh

# Check if scanner executable exists
if [ ! -f /usr/local/bin/v15_scanner ]; then
    echo "ERROR: Scanner not found"
    exit 1
fi

# Check if it runs
if ! /usr/local/bin/v15_scanner --version >/dev/null 2>&1; then
    echo "ERROR: Scanner not executable"
    exit 1
fi

# Check data directory
if [ ! -d /data/market ]; then
    echo "ERROR: Data directory not found"
    exit 1
fi

# Check recent output
LATEST=$(find /data/output -name "samples_*.bin" -mtime -1 | wc -l)
if [ "$LATEST" -eq 0 ]; then
    echo "WARNING: No recent output files"
    exit 1
fi

echo "OK: Scanner is healthy"
exit 0
```

---

## Support

For issues, questions, or contributions:

- **Documentation:** See `/docs` directory
- **Issues:** GitHub issue tracker
- **Performance:** See `PERFORMANCE_BENCHMARK.md`
- **Validation:** See `VALIDATION_REPORT.md`

---

## Version History

**v1.0.0** (2026-01-25)
- Initial production release
- Complete C++ rewrite with 10-20x speedup
- 14,190 features implemented
- Full Python bindings
- Comprehensive test suite
- Production-ready deployment

---

**Deployment Checklist:**

- [ ] Prerequisites installed and verified
- [ ] Scanner built in Release mode
- [ ] Tests passed successfully
- [ ] Performance benchmarks meet requirements (10x+ speedup)
- [ ] Python bindings working (if needed)
- [ ] Data directory accessible
- [ ] Output directory writable
- [ ] Worker count optimized for hardware
- [ ] Monitoring configured
- [ ] Health checks in place
- [ ] Backups configured for output data
