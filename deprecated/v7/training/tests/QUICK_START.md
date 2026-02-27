# Quick Start Guide - Parallel Scanning Integration Tests

## TL;DR

```bash
cd /Users/frank/Desktop/CodingProjects/x6
PYTHONPATH=. python3 -m pytest v7/training/tests/test_parallel_scanning_integration.py -v -s -k "not slow"
```

Expected: ✅ 8 passed in ~4-5 minutes

---

## Prerequisites

1. **Install pytest**:
   ```bash
   pip3 install pytest
   ```

2. **Ensure v7 is a package** (already done):
   ```bash
   ls v7/__init__.py  # Should exist
   ```

3. **Have test data** (already present):
   ```bash
   ls data/TSLA_1min.csv data/SPY_1min.csv data/VIX_History.csv
   ```

---

## Run All Tests

```bash
cd /Users/frank/Desktop/CodingProjects/x6
PYTHONPATH=. python3 -m pytest v7/training/tests/test_parallel_scanning_integration.py -v -s -k "not slow"
```

**Expected Output:**
```
8 passed in 276.09s (0:04:36)

✓ Sequential vs parallel test passed: 13 samples identical
✓ Worker count test passed for 2 workers: 13 samples identical
✓ Worker count test passed for 4 workers: 13 samples identical
✓ Worker count test passed for 8 workers: 13 samples identical
✓ Label generation test passed: Found labels for 10 timeframes
✓ Multi-window test passed: 9 samples with consistent window selection
✓ Determinism test passed: 3 runs produced identical 9 samples
✓ Custom return thresholds test passed: 9 samples identical
```

---

## Success Criteria

All tests should:
- ✅ Complete without errors
- ✅ Show "PASSED" status
- ✅ Display "X samples identical" message
- ✅ Take 4-5 minutes total
- ✅ Report no race conditions or exceptions
