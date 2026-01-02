# Live Module Test Suite

Comprehensive test suite for validating the live data module functionality in the v7 system.

## Overview

The `test_live_module.py` script provides automated testing for all critical components of the live data pipeline, ensuring reliable operation in production environments.

## Test Coverage

### Test 1: Fetch TSLA Data
- **Purpose**: Validate data loading from CSV files (simulating live data fetch)
- **Checks**:
  - File existence and accessibility
  - Required columns present (open, high, low, close, volume)
  - Data integrity (no negative prices, high >= low)
  - No NaN values
  - Proper timestamp indexing
- **Output**: Data statistics and date range

### Test 2: Multi-Resolution Alignment
- **Purpose**: Verify resampling across all 11 timeframes
- **Timeframes Tested**:
  - 5min, 15min, 30min, 1h, 2h, 3h, 4h
  - daily, weekly, monthly, 3month
- **Checks**:
  - Successful resampling to each timeframe
  - No NaN values in resampled data
  - OHLC data integrity maintained
  - Temporal consistency (shorter TF has more bars)
- **Output**: Bar counts and latest prices for each timeframe

### Test 3: VIX Integration
- **Purpose**: Test VIX regime detection and feature extraction
- **Checks**:
  - Sufficient historical data (252+ days)
  - Normalized level in valid range (0-1)
  - Percentile calculation (0-100)
  - Regime classification (0=Low, 1=Normal, 2=High, 3=Extreme)
  - Trend calculations (5-day, 20-day)
- **Output**: Current VIX metrics and regime status

### Test 4: SPY Alignment
- **Purpose**: Validate SPY data alignment with TSLA timestamps
- **Checks**:
  - Forward-fill alignment to TSLA index
  - Index equality after alignment
  - Length matching
  - Channel detection on SPY data
  - SPY feature extraction across multiple timeframes
- **Output**: SPY channel features (direction, position, RSI)

### Test 5: Cross-Asset Features
- **Purpose**: Test integrated cross-asset feature extraction
- **Checks**:
  - SPY features for all timeframes
  - Cross-containment calculations (TSLA vs SPY channels)
  - VIX integration with intraday data
  - Feature completeness
- **Output**: Feature counts and sample cross-containment metrics

### Test 6: Data Caching
- **Purpose**: Verify data loading with date filtering
- **Checks**:
  - Date range filtering (start_date, end_date)
  - Alignment of filtered data
  - Proper intersection of date ranges across assets
  - NaN handling after alignment
- **Output**: Filtered data statistics

### Test 7: Error Handling
- **Purpose**: Test robustness against edge cases
- **Scenarios Tested**:
  1. Invalid date ranges (future dates)
  2. Invalid timeframe strings
  3. Empty dataframes
  4. Insufficient historical data
  5. NaN values in channel detection
- **Output**: Pass/fail for each error scenario

### Test 8: Live Data Simulation
- **Purpose**: Simulate real-time data updates
- **Checks**:
  - Appending new bars to historical data
  - Timestamp continuity
  - Channel re-detection with updated data
  - Multi-timeframe resampling after update
- **Output**: Updated data metrics and channel status

## Usage

### Basic Run
```bash
# From project root
source myenv/bin/activate
python v7/tests/test_live_module.py
```

### Expected Output
```
================================================================================
LIVE MODULE TEST SUITE
================================================================================
Data directory: /path/to/data
Start time: 2026-01-02 09:41:52

[... test execution ...]

================================================================================
TEST SUMMARY
================================================================================
✓ Fetch TSLA Data: PASS
✓ Multi-Resolution Alignment: PASS
✓ VIX Integration: PASS
✓ SPY Alignment: PASS
✓ Cross-Asset Features: PASS
✓ Data Caching: PASS
✓ Error Handling: PASS
✓ Live Data Simulation: PASS

Total: 8 tests | Passed: 8 | Failed: 0
Pass Rate: 100.0%
================================================================================

All tests passed! Live module is ready for deployment.
```

## Requirements

### Data Files
The test requires the following CSV files in the `data/` directory:
- `TSLA_1min.csv` - TSLA 1-minute OHLCV data
- `SPY_1min.csv` - SPY 1-minute OHLCV data
- `VIX_History.csv` - VIX daily historical data

### Python Dependencies
- pandas
- numpy
- v7.core.timeframe
- v7.core.channel
- v7.features.cross_asset

## Exit Codes
- `0` - All tests passed
- `1` - One or more tests failed

## Troubleshooting

### FileNotFoundError
**Problem**: CSV files not found
**Solution**: Ensure data files exist in the `data/` directory with correct names

### Import Errors
**Problem**: Module import failures
**Solution**:
- Verify virtual environment is activated
- Check v7 package structure is intact
- Run from project root directory

### Data Quality Issues
**Problem**: Tests fail due to data integrity
**Solution**:
- Check for corrupted CSV files
- Verify date ranges in data files
- Ensure OHLCV columns are properly formatted

## Integration with CI/CD

The test script can be integrated into continuous integration pipelines:

```yaml
# Example GitHub Actions workflow
- name: Test Live Module
  run: |
    source myenv/bin/activate
    python v7/tests/test_live_module.py
```

## Future Enhancements

Potential additions to the test suite:
- Performance benchmarking (data loading speed)
- Memory usage profiling
- Concurrent data fetching tests
- Network error simulation (for future API integration)
- Historical data backfill validation
- Real-time websocket connection tests

## Related Documentation

- `/v7/core/timeframe.py` - Timeframe resampling utilities
- `/v7/core/channel.py` - Channel detection algorithms
- `/v7/features/cross_asset.py` - Cross-asset feature extraction
- `/v7/training/dataset.py` - Data loading and caching

## Support

For issues or questions about the test suite:
1. Check test output for specific error messages
2. Review related module documentation
3. Verify data file integrity
4. Check Python environment setup

---

**Last Updated**: 2026-01-02
**Test Suite Version**: 1.0
**Compatibility**: v7 system architecture
