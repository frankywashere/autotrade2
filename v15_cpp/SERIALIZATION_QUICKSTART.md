# Binary Serialization - Quick Start Guide

## TL;DR

```bash
# Generate binary samples
./build_manual/bin/v15_scanner --output samples.bin --max-samples 10000

# Load in Python
python3 load_samples.py samples.bin
```

## Basic Usage

### 1. Generate Samples (C++)

```bash
./build_manual/bin/v15_scanner \
    --step 10 \
    --max-samples 50000 \
    --output training_samples.bin \
    --workers 8
```

Output:
```
Saving 50000 samples to training_samples.bin...
  Successfully saved in 5.2s
  File verification: 50000 samples, 247 avg features, version 1
```

### 2. Load Samples (Python)

```python
from load_samples import load_samples

version, num_samples, num_features, samples = load_samples("training_samples.bin")

print(f"Loaded {len(samples)} samples")
print(f"Average features: {num_features}")

# Access first sample
sample = samples[0]
print(f"Timestamp: {sample.timestamp}")
print(f"Features: {len(sample.tf_features)}")

# Get specific feature
rsi = sample.tf_features.get("1h_rsi", None)
if rsi is not None:
    print(f"1h RSI: {rsi:.2f}")

# Get labels for best window
if sample.best_window in sample.labels_per_window:
    labels_dict = sample.labels_per_window[sample.best_window]
    for tf, labels in labels_dict.items():
        print(f"{tf} duration: {labels.duration_bars} bars")
```

### 3. Convert to Pandas DataFrame (Python)

```python
import pandas as pd
from load_samples import load_samples

_, _, _, samples = load_samples("training_samples.bin")

# Extract features
data = []
for sample in samples:
    row = {
        'timestamp': sample.timestamp,
        'channel_end_idx': sample.channel_end_idx,
        'best_window': sample.best_window,
    }
    # Add all features
    row.update(sample.tf_features)

    # Add labels from best window
    if sample.best_window in sample.labels_per_window:
        for tf, labels in sample.labels_per_window[sample.best_window].items():
            row[f'{tf}_duration'] = labels.duration_bars
            row[f'{tf}_direction'] = labels.next_channel_direction
            row[f'{tf}_break_mag'] = labels.break_magnitude

    data.append(row)

df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

print(df.head())
```

### 4. Extract for ML Training (Python)

```python
import numpy as np
from load_samples import load_samples

_, _, _, samples = load_samples("training_samples.bin")

# Extract feature matrix (X) and target (y)
feature_names = sorted(samples[0].tf_features.keys())
X = np.array([[s.tf_features[f] for f in feature_names] for s in samples])

# Extract duration as target (from 1h timeframe, best window)
y = []
for sample in samples:
    labels = sample.labels_per_window.get(sample.best_window, {}).get("1h")
    if labels:
        y.append(labels.duration_bars)
    else:
        y.append(0)
y = np.array(y)

print(f"X shape: {X.shape}")  # (num_samples, num_features)
print(f"y shape: {y.shape}")  # (num_samples,)
print(f"Features: {feature_names[:5]}")  # First 5 feature names
```

## File Format

### Magic Bytes
All files start with: `V15SAMP\0` (8 bytes)

### Verification

```bash
# Check magic bytes
hexdump -C samples.bin | head -1
# Should show: 56 31 35 53 41 4d 50 00 (V15SAMP\0)

# Check file size (roughly)
# Formula: ~3 KB per sample
ls -lh samples.bin
```

### Python Quick Check

```python
with open("samples.bin", "rb") as f:
    magic = f.read(8)
    if magic == b'V15SAMP\x00':
        print("Valid V15 binary file")
    else:
        print(f"Invalid magic bytes: {magic}")
```

## Common Operations

### Count Samples Without Loading

```python
import struct

with open("samples.bin", "rb") as f:
    f.read(8)  # Skip magic
    version = struct.unpack('<I', f.read(4))[0]
    num_samples = struct.unpack('<Q', f.read(8))[0]
    num_features = struct.unpack('<I', f.read(4))[0]

print(f"File contains {num_samples} samples with ~{num_features} features each")
```

### Batch Processing

```python
def process_samples_in_batches(filename, batch_size=1000):
    _, _, _, samples = load_samples(filename)

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        # Process batch
        print(f"Processing batch {i//batch_size + 1}: {len(batch)} samples")
        yield batch

# Usage
for batch in process_samples_in_batches("samples.bin", batch_size=5000):
    # Train model, compute stats, etc.
    pass
```

### Filter Samples

```python
from load_samples import load_samples

_, _, _, samples = load_samples("training_samples.bin")

# Filter by timestamp
import time
cutoff = int(time.mktime(time.strptime("2023-01-01", "%Y-%m-%d")) * 1000)
recent_samples = [s for s in samples if s.timestamp >= cutoff]

# Filter by feature value
high_rsi_samples = [s for s in samples if s.tf_features.get("1h_rsi", 0) > 70]

# Filter by label
long_duration = [s for s in samples
                 if 50 in s.labels_per_window
                 and "1h" in s.labels_per_window[50]
                 and s.labels_per_window[50]["1h"].duration_bars > 100]

print(f"Recent: {len(recent_samples)}")
print(f"High RSI: {len(high_rsi_samples)}")
print(f"Long duration: {len(long_duration)}")
```

## Performance Tips

### C++ Side
- Use `--workers 8` for parallel processing
- Larger `--step` = faster but fewer samples
- `--max-samples` limits output size

### Python Side
- Load entire file once if possible (it's fast)
- Use generators for very large files
- Convert to numpy/pandas for analysis
- Cache processed features if doing multiple passes

## Troubleshooting

### File Not Found
```bash
# Make sure you're in the project directory
cd /path/to/v15_cpp
python3 load_samples.py samples.bin
```

### Magic Bytes Error
```python
# File is not a valid V15 binary
# Regenerate with v15_scanner --output
```

### EOF Error
```python
# File is incomplete (write was interrupted)
# Regenerate the file
```

### Version Mismatch
```python
# Format version not supported
# Update load_samples.py or regenerate with current scanner
```

## Next Steps

1. **Generate training data**: Run scanner with `--output`
2. **Load in Python**: Use `load_samples.py`
3. **Build ML pipeline**: Extract features and labels
4. **Train model**: Use scikit-learn, PyTorch, etc.
5. **Backtest**: Use channel predictions

## Files Reference

- `include/serialization.hpp` - C++ API
- `src/serialization.cpp` - C++ implementation
- `load_samples.py` - Python loader
- `test_serialization.cpp` - C++ test
- `verify_serialization.sh` - Full test suite
- `SERIALIZATION_README.md` - Detailed docs
