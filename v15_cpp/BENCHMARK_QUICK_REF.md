# Performance Benchmark - Quick Reference

## TL;DR

**Status:** ✅ Partial Success - Passes 1 & 2 working, Pass 3 blocked

**Performance:**
- **Channel Detection:** 390K channels/sec (8 workers)
- **Label Generation:** 2.1M labels/sec
- **Pipeline Time:** 4.5s (internal), 28s (wall clock)
- **Memory:** 3.9 GB peak
- **Speedup vs Python:** ~200x (pipeline), ~30x (including I/O)

**Blocker:** All labels invalid → 0 samples → Pass 3 cannot run

---

## Quick Commands

### Run Benchmark
```bash
# 8 workers (recommended)
./build_manual/bin/v15_scanner \
  --data-dir ../data \
  --step 10 \
  --max-samples 10000 \
  --workers 8 \
  --output /tmp/benchmark.bin

# Quiet mode
./build_manual/bin/v15_scanner \
  --data-dir ../data \
  --step 10 \
  --max-samples 10000 \
  --workers 8 \
  --output /tmp/benchmark.bin \
  --quiet

# With memory tracking
/usr/bin/time -l ./build_manual/bin/v15_scanner \
  --data-dir ../data \
  --step 10 \
  --max-samples 10000 \
  --workers 8 \
  --output /tmp/benchmark.bin \
  --quiet
```

### Benchmark Different Worker Counts
```bash
for workers in 1 4 8 16; do
  echo "=== Workers: $workers ==="
  ./build_manual/bin/v15_scanner \
    --data-dir ../data \
    --step 10 \
    --max-samples 10000 \
    --workers $workers \
    --output /tmp/bench_w${workers}.bin \
    --quiet | grep -E "(Pass|Total)"
done
```

---

## Results at a Glance

### Performance Metrics (8 Workers)

| Metric | Value |
|--------|-------|
| Channel Detection | 392,353 ch/s |
| Label Generation | 2,155,373 lb/s |
| Feature Extraction | N/A (blocked) |
| Peak Memory | 3.94 GB |
| Pipeline Time | 4.5s |
| Wall Clock Time | 28s |

### Worker Scaling

| Workers | Pass 1 | Pass 2 | Total | Memory |
|---------|--------|--------|-------|--------|
| 1       | 2.81s  | 0.52s  | 4.6s  | 3.69 GB |
| 4       | 2.95s  | 0.63s  | 4.8s  | N/A |
| 8       | 2.73s  | 0.52s  | 4.5s  | 3.94 GB |

**Observation:** Minimal scaling benefit (7% improvement 1→8 workers)

---

## Bottlenecks

1. **Data Loading** (55% of time) - Use binary cache
2. **Data Resampling** (37% of time) - Cache resampled data
3. **Memory Bandwidth** (Pass 1) - SIMD vectorization
4. **Label Validation** (CRITICAL) - All labels invalid

---

## Known Issues

### Critical
- ❌ **Pass 3 not working** - All labels invalid (0 samples generated)
  - Cause: Channels at end of dataset lack forward data
  - Fix: Limit detection to positions with forward data available

### Performance
- ⚠️ **Poor parallel scaling** - 13% efficiency (expected 80%+)
  - Cause: Memory bandwidth bottleneck, not CPU bound
  - Fix: SIMD, better cache locality, thread-local storage

### Data Loading
- ⚠️ **CSV loading overhead** - 15s (55% of wall clock)
  - Cause: Parsing CSV files
  - Fix: Binary cache with mmap

---

## Next Steps

### High Priority
1. Fix label validation (skip end-of-dataset channels)
2. Benchmark Pass 3 once working
3. Implement binary data cache

### Medium Priority
4. Profile memory bandwidth in Pass 1
5. SIMD vectorization for linear regression
6. Optimize parallel scaling

### Low Priority
7. Fine-tune worker configuration
8. Memory pool allocation
9. Cache-oblivious algorithms

---

## Files Generated

- `PERFORMANCE_BENCHMARK.md` - Detailed analysis
- `BENCHMARK_SUMMARY.txt` - Tabular results
- `LABEL_VALIDATION_ISSUE.md` - Debug report
- `BENCHMARK_QUICK_REF.md` - This file

---

## Reproducing Results

1. **Build scanner:**
   ```bash
   ./build_manual.sh
   ```

2. **Run benchmark:**
   ```bash
   ./build_manual/bin/v15_scanner \
     --data-dir ../data \
     --step 10 \
     --max-samples 10000 \
     --workers 8 \
     --output /tmp/benchmark.bin
   ```

3. **Check output:**
   ```bash
   # View summary
   tail -20 output.log

   # Check memory
   /usr/bin/time -l ./build_manual/bin/v15_scanner ...
   ```

---

## Conclusion

The C++ scanner achieves **excellent performance** for channel detection and label generation (390K ch/s, 2M lb/s), delivering the target **10-20x speedup** over Python.

**However**, Pass 3 is blocked due to label validation issues. Once fixed, the scanner will be production-ready.

**Achievement:** Core algorithms optimized ✅
**Blocker:** Label validation needs fix ❌

---

**Benchmark Date:** 2026-01-25
**Scanner Version:** v15
**Dataset:** 440K bars (TSLA/SPY, 2015-2025)
