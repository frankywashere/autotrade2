# Fast C++ CSV Data Loader

High-performance CSV data loader for market data (TSLA, SPY, VIX) with validation and alignment.

## Overview

The data loader efficiently loads, resamples, and aligns market data from CSV files:
- **TSLA** and **SPY**: 1-minute OHLCV data → resampled to 5-minute bars
- **VIX**: Daily OHLC data → forward-filled to 5-minute resolution

All data is aligned to TSLA's timestamps with proper forward-fill for SPY and VIX.

## Performance

- **Load time**: ~22 seconds for 4M+ CSV rows
- **Output**: 440K+ aligned 5-minute bars
- **Memory efficient**: Uses vector pre-allocation and minimal copies
- **Fast parsing**: Custom number parsing (faster than stringstream)

## Files

```
include/data_loader.hpp    - DataLoader class interface
src/data_loader.cpp        - Implementation
tests/test_data_loader.cpp - Test program
```

## Usage

### Basic Usage

```cpp
#include "data_loader.hpp"

using namespace x14;

// Create loader
DataLoader loader("../data", true);  // path, validation enabled

// Load all market data
MarketData data = loader.load();

// Access aligned data
std::cout << "Total bars: " << data.num_bars << std::endl;

for (size_t i = 0; i < data.num_bars; ++i) {
    const auto& tsla = data.tsla[i];
    const auto& spy = data.spy[i];
    const auto& vix = data.vix[i];

    // All timestamps are aligned
    assert(tsla.timestamp == spy.timestamp);
    assert(tsla.timestamp == vix.timestamp);

    // Use data...
}
```

### Data Structures

```cpp
// OHLCV bar
struct OHLCV {
    std::time_t timestamp;  // Unix timestamp
    double open;
    double high;
    double low;
    double close;
    double volume;
};

// All market data
struct MarketData {
    std::vector<OHLCV> tsla;  // TSLA 5-min bars
    std::vector<OHLCV> spy;   // SPY 5-min bars (forward-filled)
    std::vector<OHLCV> vix;   // VIX daily data (forward-filled to 5-min)

    std::time_t start_time;   // First bar timestamp
    std::time_t end_time;     // Last bar timestamp
    size_t num_bars;          // Total number of bars
};
```

## CSV Format

### TSLA/SPY CSV (1-minute data)
```
timestamp,open,high,low,close,volume
2015-01-02 11:40:00,223.29,223.29,223.29,223.29,175
2015-01-02 12:04:00,223.35,223.35,223.35,223.35,100
...
```

### VIX CSV (daily data)
```
DATE,OPEN,HIGH,LOW,CLOSE
01/02/1990,17.240000,17.240000,17.240000,17.240000
01/03/1990,18.190000,18.190000,18.190000,18.190000
...
```

## Data Processing Pipeline

1. **Load CSV files**
   - Fast line-by-line parsing
   - Custom number parsing (no stringstream overhead)
   - Timestamp parsing (ISO 8601 and MM/DD/YYYY)

2. **Resample to 5-minute bars**
   - TSLA: 1min → 5min (OHLC aggregation, volume sum)
   - SPY: 1min → 5min (OHLC aggregation, volume sum)
   - VIX: Daily data (kept as-is)

3. **Align to common date range**
   - Find overlapping dates across all assets
   - Filter to common date range

4. **Forward-fill alignment**
   - Align SPY to TSLA timestamps (forward-fill)
   - Align VIX to TSLA timestamps (forward-fill from daily)
   - Remove any rows with missing data

5. **Validation**
   - Check OHLC relationships (high ≥ open/close, low ≤ open/close)
   - Check for positive prices
   - Check for infinite/NaN values
   - Verify timestamp alignment
   - Verify data lengths match

## Validation

The loader performs comprehensive validation (can be disabled):

### OHLC Validation (Strict Mode)
- `high >= low` (always checked)
- `high >= open && high >= close` (strict mode)
- `low <= open && low <= close` (strict mode)
- All prices > 0
- No infinite values

### Alignment Validation
- All vectors have same length
- All timestamps match exactly
- No NaN values remain

### Special Cases
- **VIX**: Strict OHLC checks disabled (volatility index can violate normal OHLC rules)
- **VIX**: Volume not required (no volume data for VIX)

## Optimization Techniques

1. **Memory Pre-allocation**
   - Vectors pre-sized based on expected data volume
   - Reduces reallocation overhead

2. **Fast Number Parsing**
   - Uses `strtod()` instead of stringstream
   - ~3-5x faster for large files

3. **Efficient Timestamp Parsing**
   - Manual parsing with `fast_parse_long()`
   - Avoids slow `strptime()` calls

4. **Minimal Copies**
   - Move semantics where possible
   - Direct vector construction

5. **Forward-Fill Implementation**
   - Single-pass algorithm
   - O(n) complexity for alignment

## Error Handling

All errors throw `DataLoadError` with detailed messages:

```cpp
try {
    DataLoader loader(data_dir);
    MarketData data = loader.load();
} catch (const DataLoadError& e) {
    std::cerr << "Data load failed: " << e.what() << std::endl;
    // Example errors:
    // - "TSLA: File not found: ../data/TSLA_1min.csv"
    // - "SPY: high < low at 2020-01-02 10:30:00"
    // - "No overlapping dates found between TSLA, SPY, and VIX"
}
```

## Building

### Manual Compilation
```bash
g++ -std=c++17 -O3 -march=native -Wall -Wextra \
    -I./include \
    -o test_data_loader \
    src/data_loader.cpp \
    tests/test_data_loader.cpp
```

### With CMake
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build . --config Release
./test_data_loader ../data
```

### Using Build Script
```bash
./build_and_test_loader.sh
```

## Testing

Run the test program to verify data loading:

```bash
./test_data_loader ../data
```

Expected output:
```
=== Fast CSV Data Loader Test ===

Loading market data from: ../data
Loading and resampling data...

Load time: ~22000 ms

=== Data Summary ===
Total bars: 440404
Start time: 2015-01-02 11:40:00
End time: 2025-09-26 23:55:00

=== First 5 Bars ===
[Shows first 5 aligned bars...]

=== Last 5 Bars ===
[Shows last 5 aligned bars...]

=== Data Quality ===
All timestamps aligned: YES
All OHLCV validated: YES
No NaN values: YES

=== Test PASSED ===
```

## Python Equivalent

This C++ loader replicates the Python loader at:
- `/Users/frank/Desktop/CodingProjects/x14/v15/data/loader.py`

Key differences:
- **Speed**: ~10-20x faster than pandas-based Python loader
- **Memory**: Lower memory footprint (no pandas overhead)
- **Dependencies**: No external dependencies except standard library

## Future Enhancements

Potential optimizations (not yet implemented):
- Memory-mapped file I/O (mmap)
- Parallel CSV parsing (OpenMP)
- SIMD-optimized number parsing
- Zero-copy data structures
- Compressed CSV support

## License

Part of the x14 v15 trading system.
