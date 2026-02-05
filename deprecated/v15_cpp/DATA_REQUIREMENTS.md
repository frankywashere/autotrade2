# V15 Scanner Data Requirements

## Overview

The V15 Scanner requires sufficient historical data to generate samples with valid labels. This document explains the data requirements and why they exist.

## Minimum Data Requirements

### Formula
```
Total bars needed = WARMUP + SCANNABLE_WINDOW + FORWARD_SCAN
```

### Constants (5min timeframe)
- **WARMUP**: 32,760 bars (~114 days)
  - Required for feature extraction
  - Cannot extract features without sufficient lookback

- **FORWARD_SCAN**: 21,000 bars (~73 days)
  - Required for label generation
  - Scanner needs to observe channel behavior after detection

- **SCANNABLE_WINDOW**: Depends on --step parameter
  - With step=10: ~6,000 bars minimum
  - With step=20: ~3,000 bars minimum

### Recommended Minimums

| Use Case | Minimum Bars | Days | Months |
|----------|--------------|------|--------|
| Quick test | 60,000 | ~208 | ~7 |
| Development | 80,000 | ~278 | ~9 |
| Production | 100,000+ | ~347+ | ~12+ |

## Why These Requirements?

### 1. Feature Extraction (WARMUP)
```
To extract features at position N, we need:
- Historical data from position (N - 32,760) to N
- Without this, features are incomplete/invalid
```

Example:
- Sample at bar 40,000
- Feature extraction looks back to bar 7,240 (40,000 - 32,760)
- If data starts at bar 10,000 → **INSUFFICIENT** → Sample skipped

### 2. Label Generation (FORWARD_SCAN)
```
To generate labels for a channel ending at position N:
- Need forward data from N+1 to N+21,000
- Observe break patterns, next channel formation, etc.
```

Example:
- Channel ends at bar 60,000
- Label generation scans bars 60,001 to 81,000
- If data ends at bar 70,000 → **INSUFFICIENT** → Invalid labels

### 3. Valid Scan Window
```
Valid positions for samples:
  START = WARMUP = 32,760
  END = TOTAL_BARS - FORWARD_SCAN = TOTAL_BARS - 21,000

Valid window size = END - START
```

Example with 60,000 bars:
- START: 32,760
- END: 39,000 (60,000 - 21,000)
- **Valid window**: 6,240 bars (~22 days)

Example with 100,000 bars:
- START: 32,760
- END: 79,000 (100,000 - 21,000)
- **Valid window**: 46,240 bars (~161 days)

## Checking Your Dataset

### Using the Scanner Output

Run the scanner with `--verbose` to see scan bounds:

```bash
./v15_scanner --verbose --data-dir /path/to/data
```

Look for this output:
```
[SCAN BOUNDS] Total 5min bars: 100000
[SCAN BOUNDS] Scanner forward requirement: 21000 bars
[SCAN BOUNDS] Valid scan end (5min): 79000
```

### Interpreting Results

**Good dataset:**
```
Total bars: 100,000
Valid scan end: 79,000
Valid window: 46,240 bars
✓ Plenty of room for sample generation
```

**Marginal dataset:**
```
Total bars: 60,000
Valid scan end: 39,000
Valid window: 6,240 bars
⚠️ Limited sample space - may not hit max_samples
```

**Insufficient dataset:**
```
Total bars: 40,000
Valid scan end: 19,000
Valid window: -13,760 bars (NEGATIVE!)
✗ Cannot generate any samples
```

## Pass 2 Label Validation

After Pass 2, check the label counts:

```
Pass 2 - Labels generated:
  TSLA: 26,214 (26,214 valid)
  SPY: 26,214 (26,214 valid)
```

**Healthy**: `valid labels = total labels`
- All channels have sufficient forward data
- Ready for sample generation

**Problematic**: `valid labels = 0` or very low
- Dataset too small
- Need more historical data

## Troubleshooting

### Problem: "0 samples generated"

**Check 1: Total bars**
```bash
# Should be 60,000+
grep "Total 5min bars" <scanner_output>
```

**Check 2: Valid labels**
```bash
# Should be > 0
grep "valid labels" <scanner_output>
```

**Solution**: Get more historical data

### Problem: "Fewer samples than requested"

**Scenario 1: Not enough valid channels**
```
max_samples = 1000
Valid channels with labels = 500
Samples generated = 500  # Correct behavior
```

**Scenario 2: Dataset boundaries**
```
Total bars = 60,000
Valid window = 6,240 bars
Step = 10
Possible channels ≈ 624
Samples generated ≤ 624  # Cannot exceed available
```

**Solution**:
- Increase dataset size, OR
- Decrease --step parameter (finds more channels), OR
- Lower --max-samples to match available

### Problem: "All labels invalid"

**Cause**: Dataset too small for forward scan

**Check**:
```
[SCAN BOUNDS] Valid scan end (5min): X
Total bars: Y
Forward requirement: 21000

If (Y - X) < 21000 → PROBLEM
```

**Solution**: Need dataset with at least 60,000 bars

## Recommendations by Use Case

### Development/Testing
```bash
# Minimum viable dataset
Total bars: 60,000
--step: 20
--max-samples: 500
```

### Production Training Data
```bash
# Full dataset
Total bars: 100,000+
--step: 10
--max-samples: 10,000+
```

### Quick Validation
```bash
# Just verify it works
Total bars: 60,000
--step: 50  # Fewer channels, faster
--max-samples: 100
```

## Data Format

The scanner expects aligned 5min OHLCV data:
- TSLA_1min.csv → resampled to 5min
- SPY_1min.csv → resampled to 5min
- VIX_1min.csv → resampled to 5min

All three must have:
- Same number of bars
- Aligned timestamps
- No gaps or missing data

## Performance Notes

### Larger datasets = Better quality
- More diverse market conditions
- Better representation of channel patterns
- More valid samples for training

### But also = Slower processing
- Pass 1: ~300K channels/sec
- Pass 2: ~140K labels/sec
- Pass 3: ~25 samples/sec

Example timing (100K bars):
- Pass 1: ~30 seconds
- Pass 2: ~40 seconds
- Pass 3: ~400 seconds (for 10K samples)
- **Total**: ~8 minutes

## Summary

| Requirement | Bars | Purpose |
|-------------|------|---------|
| WARMUP | 32,760 | Feature extraction lookback |
| FORWARD_SCAN | 21,000 | Label generation forward scan |
| MINIMUM TOTAL | 60,000 | Viable dataset |
| RECOMMENDED | 100,000+ | Production use |

**Quick Check**: If your dataset has fewer than 60,000 5min bars, you'll likely get 0 samples or invalid labels.
