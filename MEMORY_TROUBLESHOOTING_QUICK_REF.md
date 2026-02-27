# Memory Troubleshooting Quick Reference

## TL;DR - Scanner Killed on Server

**Problem:** Scanner dies with "Killed" message on 28-core server at `/workspace/autotrade2`

**Root Cause:** OOM (Out of Memory) kill by cgroup/container memory limit

**Solution:** Enable streaming mode (now default) and use appropriate worker count

---

## Quick Diagnostic (30 seconds)

```bash
# 1. Check total memory
free -h

# 2. Check container limit
cat /sys/fs/cgroup/memory/memory.limit_in_bytes | awk '{print $1/1024/1024/1024 " GB"}'

# 3. Check recent OOM kills
dmesg | grep -i "killed process" | tail -3

# 4. Check if streaming was enabled
grep -i "streaming" scanner.log  # Look for "ENABLED" or "disabled"
```

---

## Safe Scanner Command (Any Server)

```bash
# Streaming mode (default) - safe for any memory constraint
./v15_scanner \
  --step 1 \
  --output samples.bin \
  --workers 8 \
  --batch-size 8 \
  --streaming \
  --data-dir /workspace/autotrade2/data
```

**Memory usage:** ~8-12 GB peak (safe for 32GB+ servers)

---

## Memory-Constrained Servers (<32 GB)

```bash
# Reduce workers and batch size
./v15_scanner \
  --step 1 \
  --output samples.bin \
  --workers 4 \
  --batch-size 4 \
  --streaming \
  --flush-interval 500 \
  --data-dir /workspace/autotrade2/data
```

**Memory usage:** ~6-8 GB peak

---

## Quick Test (Small Dataset)

```bash
# Capped samples for testing
./v15_scanner \
  --step 10 \
  --max-samples 10000 \
  --output test.bin \
  --workers 4 \
  --data-dir /workspace/autotrade2/data
```

**Memory usage:** ~3-5 GB peak
**Duration:** ~10-30 minutes

---

## Real-Time Memory Monitoring

```bash
# Terminal 1: Run scanner
./v15_scanner --step 1 --output samples.bin --workers 8 --streaming

# Terminal 2: Monitor memory
watch -n 5 'free -h; echo ""; ps aux --sort=-%mem | head -5'

# Terminal 3: Watch for OOM
dmesg -w | grep -i memory
```

---

## Common Issues and Fixes

### Issue 1: "Killed" with no error message
**Cause:** OOM kill by container/cgroup
**Fix:** Enable streaming, reduce workers
```bash
./v15_scanner --step 1 --output samples.bin --workers 4 --streaming
```

### Issue 2: Slow performance on 28-core server
**Cause:** Memory bottleneck (not CPU)
**Fix:** Use 4-8 workers instead of 28
```bash
./v15_scanner --step 1 --output samples.bin --workers 8 --streaming
```

### Issue 3: "Error: output file required for streaming"
**Cause:** Streaming enabled but no output path
**Fix:** Always specify `--output` with streaming
```bash
./v15_scanner --step 1 --output samples.bin --streaming
```

### Issue 4: Process killed during Pass 1
**Cause:** Step too low (detecting too many channels)
**Fix:** Increase step size or use max-samples
```bash
./v15_scanner --step 5 --max-samples 100000 --output samples.bin
```

---

## Memory Requirements by Configuration

| Step | Streaming | Workers | Peak Memory | Safe For |
|------|-----------|---------|-------------|----------|
| 1 | Yes | 8 | 8-12 GB | 32GB+ servers |
| 1 | Yes | 4 | 6-8 GB | 16GB+ servers |
| 1 | No | 8 | 15-30 GB | 64GB+ servers ⚠️ |
| 10 | Yes | 8 | 3-5 GB | 16GB+ servers |
| 10 | No | 8 | 5-10 GB | 32GB+ servers |

⚠️ = Not recommended for cloud containers

---

## Server Type Detection

### RunPod / Lambda Labs (Most Likely)
- Path: `/workspace/*`
- 28 cores: A40 or A100 instance
- Typical RAM: 32-64 GB
- Memory management: cgroup limits
- **Recommendation:** `--workers 8 --streaming`

### Google Colab
- Path: `/content/*`
- Free: 12 GB RAM, 2 cores
- Pro: 25-50 GB RAM, 8 cores
- **Recommendation:** `--workers 4 --step 10`

### AWS / GCP / Azure
- Path: Varies
- Check: `free -h` for total RAM
- **Recommendation:** `--workers $(nproc / 4) --streaming`

---

## Cheat Sheet: Memory Optimization

### 1. Always Use Streaming (Default)
```bash
# Good (streaming enabled by default)
./v15_scanner --output samples.bin

# Bad (explicitly disabled streaming - OOM risk!)
./v15_scanner --no-streaming --output samples.bin
```

### 2. Worker Count = RAM / 8 GB
```bash
# 32 GB server → 4 workers
./v15_scanner --workers 4 --output samples.bin

# 64 GB server → 8 workers
./v15_scanner --workers 8 --output samples.bin
```

### 3. Test Before Full Run
```bash
# Always test with max-samples first
./v15_scanner --max-samples 10000 --output test.bin --workers 4
```

### 4. Monitor During Execution
```bash
# Watch memory every 5 seconds
watch -n 5 'free -h'
```

---

## One-Liner Memory Check

```bash
echo "RAM: $(free -h | awk '/^Mem:/{print $2}') | Available: $(free -h | awk '/^Mem:/{print $7}') | Limit: $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null | awk '{print $1/1024/1024/1024}')GB | OOM Events: $(dmesg | grep -ic 'out of memory')"
```

---

## Emergency: Scanner Killed Mid-Run

```bash
# 1. Check what happened
dmesg | grep -i "killed process" | tail -5

# 2. Check memory state
free -h
cat /sys/fs/cgroup/memory/memory.usage_in_bytes | awk '{print $1/1024/1024/1024 " GB"}'

# 3. Restart with conservative settings
./v15_scanner \
  --step 5 \
  --max-samples 50000 \
  --output samples.bin \
  --workers 4 \
  --batch-size 4 \
  --streaming

# 4. Monitor closely
watch -n 2 'free -h; echo; ps aux --sort=-%mem | head -3'
```

---

## Key Takeaways

1. **Streaming is essential** for step=1 scans (now default)
2. **More workers ≠ faster** if memory-constrained
3. **Monitor first, then optimize** - check `free -h` regularly
4. **Test small before large** - use `--max-samples` for testing
5. **Cloud containers have limits** - don't trust core count alone

---

## Support Resources

- Full analysis: `SERVER_MEMORY_ANALYSIS.md`
- Scanner README: `v15_cpp/SCANNER_README.md`
- Deployment guide: `v15_cpp/DEPLOYMENT.md`

---

**Last Updated:** 2026-02-04
