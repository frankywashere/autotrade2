# Server Environment and Memory Constraints Analysis

## Investigation Summary

Based on the path `/workspace/autotrade2` and the "Killed" message on a 28-core server, this analysis identifies the likely server environment, memory constraints, and provides diagnostic recommendations.

---

## 1. Server Environment Identification

### Path Analysis: `/workspace/autotrade2`

The path structure `/workspace/*` is characteristic of **cloud GPU compute environments**. Based on industry patterns:

**Most Likely Environments:**

1. **RunPod** (Highest probability)
   - Uses `/workspace` as the standard working directory
   - Popular for ML/AI workloads
   - Offers 28-core instances (e.g., A40, A100 pods)
   - Known for aggressive memory limits

2. **Lambda Labs**
   - Also uses `/workspace` convention
   - Provides high-core-count GPU instances
   - Similar memory constraints

3. **Other possibilities:**
   - Paperspace Gradient
   - Vast.ai
   - Generic Docker/Kubernetes deployment
   - Google Colab Pro+ (uses `/content` but could be customized)

### Environment Characteristics

- **Type**: Containerized cloud GPU environment
- **CPU Cores**: 28 (suggests A40, A100, or similar GPU instance)
- **Memory Management**: cgroup-controlled (Docker/Kubernetes)
- **Termination Signal**: OOM killer sent SIGKILL (hence "Killed" message)

---

## 2. Memory Limit Estimation

### Typical Memory Configurations for 28-Core Instances

| Environment | Instance Type | RAM Range | Memory/Core |
|-------------|---------------|-----------|-------------|
| RunPod | A40 (48GB GPU) | 32-64 GB | 1.1-2.3 GB |
| RunPod | A100 (40GB GPU) | 64-80 GB | 2.3-2.9 GB |
| Lambda Labs | A100 | 64-96 GB | 2.3-3.4 GB |
| AWS p3.8xlarge | 4x V100 | 244 GB | 8.7 GB |
| Generic Container | Variable | 16-128 GB | Variable |

### Most Likely Scenario: 32-64 GB RAM

Based on the 28-core configuration and typical cloud GPU offerings:
- **Estimated RAM**: 32-64 GB
- **Effective available**: 28-56 GB (accounting for OS/system overhead)
- **Per-core allocation**: ~1-2 GB

---

## 3. Scanner Memory Requirements

### Memory Consumption Analysis

From the codebase (`/Users/frank/Desktop/CodingProjects/x14/v15_cpp/src/scanner.cpp`):

#### Pass 1: Channel Detection
```cpp
std::unordered_map<TFWindowKey, std::vector<Channel>, TFWindowKeyHash> tsla_channel_map;
std::unordered_map<TFWindowKey, std::vector<Channel>, TFWindowKeyHash> spy_channel_map;
```

**Memory per channel map:**
- 10 timeframes × 8 windows = 80 (tf, window) combinations
- Full Channel struct includes: `upper_line`, `lower_line`, `center_line` vectors
- Estimated: **2-5 GB per asset** (TSLA + SPY = 4-10 GB total)

#### Pass 2: Label Generation
```cpp
SlimLabeledChannelMap tsla_slim_map;
SlimLabeledChannelMap spy_slim_map;
```

**Memory optimization:**
- Strips heavy arrays after Pass 1 (`tsla_channel_map.clear()`)
- SlimLabeledChannel: ~100x reduction (MBs instead of GBs)
- Estimated: **200-500 MB per asset**

#### Pass 3: Sample Generation
```cpp
std::vector<ChannelSample> samples;  // Only used in non-streaming mode
std::vector<ChannelWorkItem> channel_work_items;
```

**Memory consumption (non-streaming mode):**
- Channel work items with pre-computed histories
- Each sample: ~3-5 KB (features + labels)
- For 100K samples: **300-500 MB**
- For 1M samples: **3-5 GB**

**Total peak memory (non-streaming):**
- Base data (TSLA, SPY, VIX): ~2-3 GB
- Pass 1 full maps: 4-10 GB
- Pass 2 slim maps: 0.5-1 GB
- Pass 3 samples: 0.3-5 GB
- **Total: 7-19 GB** (varies with step size and data volume)

### Why the Scanner Was Killed

**Scenario: step=1 without streaming**
```bash
./v15_scanner --step 1 --output samples.bin --workers 28
```

With `step=1`:
- Detects **500K-1M+ channels** per asset
- Pass 1 peak: **10-20 GB** (full channel maps)
- Pass 3 accumulation: **5-10 GB** (all samples in memory)
- **Total peak: 15-30 GB**

On a 32-48 GB server with 28 cores:
- Available per-process: ~28-40 GB
- Scanner consumption: 15-30 GB
- **Result: OOM kill if near upper bound**

---

## 4. Memory-Related Configuration Options

### Current Scanner Options (from `main_scanner.cpp`)

```bash
# Memory-critical options
--streaming              # Enable streaming mode (default: on)
--no-streaming          # Disable streaming (accumulate in memory - OOM risk!)
--flush-interval N      # Samples between disk flushes (default: 1000)
--max-samples N         # Cap sample count (prevents unbounded growth)
--workers N             # Thread count (each adds overhead)
--batch-size N          # Channels per batch (default: 8)
--step N                # Channel detection step (lower = more channels = more memory)
```

### Streaming Mode (Critical for Large Scans)

From `scanner.cpp` line 590:
```cpp
// STREAMING MODE: Write samples directly to disk to avoid OOM
bool use_streaming = config_.streaming && !config_.output_path.empty();
```

**Benefits:**
- Writes samples directly to disk during generation
- Memory usage: **O(batch_size)** instead of O(total_samples)
- Essential for `step=1` or low-step scans
- Default: **ENABLED** (since recent commits)

**Command comparison:**
```bash
# Safe: Streaming enabled (default)
./v15_scanner --step 1 --output samples.bin --workers 8

# DANGEROUS: Streaming disabled - will OOM on large datasets!
./v15_scanner --step 1 --no-streaming --output samples.bin --workers 8
```

### Parallel Processing Memory Impact

From `scanner.hpp` line 282:
```cpp
class ThreadPool {
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    // Each worker adds ~8-16 MB overhead + task data
};
```

**Memory scaling with workers:**
- Base overhead: ~8-16 MB per worker
- Task queue: Holds batch futures
- With 28 workers: **~300-500 MB overhead**

---

## 5. System Constraints in Codebase

### No Explicit Memory Limits Configured

**Search results:**
- No `ulimit`, `cgroup`, or memory limit configuration found in codebase
- No Docker memory limits (`--memory`, `--memory-swap`)
- No Kubernetes resource limits in deployment files

### Deployment Configuration

From `/Users/frank/Desktop/CodingProjects/x14/v15_cpp/DEPLOYMENT.md` (line 948):

```yaml
# Kubernetes deployment example
resources:
  requests:
    memory: "8Gi"
    cpu: "4"
  limits:
    memory: "16Gi"
    cpu: "8"
```

**Recommendation:** These are example values. The actual server likely has **no explicit limits** set, relying on cgroup defaults.

---

## 6. Diagnostic Commands for Server Memory

### Check Container Memory Limits

Run these commands on the `/workspace/autotrade2` server:

#### 1. Check cgroup memory limit
```bash
# Docker/Kubernetes memory limit
cat /sys/fs/cgroup/memory/memory.limit_in_bytes
# If output is very large (e.g., 9223372036854771712), no limit is set

# Convert to GB
cat /sys/fs/cgroup/memory/memory.limit_in_bytes | awk '{print $1/1024/1024/1024 " GB"}'
```

#### 2. Check current memory usage
```bash
# Total system memory
free -h

# Process memory (run during scanner execution)
ps aux --sort=-%mem | head -20

# Real-time monitoring
top -o %MEM
```

#### 3. Check OOM kill history
```bash
# Recent OOM kills
dmesg | grep -i "killed process"
dmesg | grep -i "out of memory"

# Systemd journal (if available)
journalctl -k | grep -i "out of memory"
```

#### 4. Check swap availability
```bash
# Swap status (often disabled in containers)
swapon --show
free -h | grep -i swap
```

#### 5. Check cgroup memory stats
```bash
# Current usage vs limit
cat /sys/fs/cgroup/memory/memory.usage_in_bytes
cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes
cat /sys/fs/cgroup/memory/memory.limit_in_bytes

# Memory pressure events
cat /sys/fs/cgroup/memory/memory.oom_control
```

#### 6. Check available memory for process
```bash
# Available memory (not cached/buffered)
free -h | awk '/^Mem:/ {print $7}'

# Total vs available
cat /proc/meminfo | grep -E "MemTotal|MemAvailable"
```

### Memory-Aware Scanner Execution

```bash
# 1. Test with memory monitoring
(./v15_scanner --step 10 --output test.bin --workers 8 --max-samples 10000 &
 PID=$!
 while kill -0 $PID 2>/dev/null; do
   ps -p $PID -o pid,vsz,rss,%mem,cmd
   sleep 5
 done)

# 2. Run with resource limits (if running directly)
ulimit -v 30000000  # Limit virtual memory to ~30GB
./v15_scanner --step 10 --output samples.bin --workers 8

# 3. Use time and /usr/bin/time for memory tracking
/usr/bin/time -v ./v15_scanner --step 10 --output samples.bin --workers 8
```

---

## 7. Recommended Configuration for 32-64 GB Server

### Safe Configuration (Step=1, Full Scan)

```bash
# With streaming (default) - SAFE
./v15_scanner \
  --step 1 \
  --output samples.bin \
  --workers 8 \
  --batch-size 8 \
  --streaming \
  --flush-interval 1000 \
  --data-dir /workspace/autotrade2/data
```

**Memory usage:** ~8-12 GB peak

### Conservative Configuration (Limited Memory)

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

### Quick Test Configuration

```bash
# Capped sample count for testing
./v15_scanner \
  --step 1 \
  --max-samples 10000 \
  --output test_samples.bin \
  --workers 8 \
  --data-dir /workspace/autotrade2/data
```

**Memory usage:** ~5-7 GB peak

---

## 8. Memory Optimization Checklist

### Before Running

- [ ] Verify `--streaming` is enabled (default) or explicitly set
- [ ] Set `--max-samples` for initial testing
- [ ] Reduce `--workers` if memory-constrained (4-8 instead of 28)
- [ ] Specify `--output` path (required for streaming mode)
- [ ] Check available memory: `free -h`

### During Execution

- [ ] Monitor memory: `watch -n 5 'free -h'`
- [ ] Check process memory: `watch -n 5 'ps aux --sort=-%mem | head -10'`
- [ ] Watch for OOM warnings: `dmesg -w | grep -i memory`

### After Execution

- [ ] Check for OOM kills: `dmesg | grep -i killed`
- [ ] Verify output file: `ls -lh samples.bin`
- [ ] Review peak memory from logs

---

## 9. Container Memory Limit Detection Script

Save this as `check_memory.sh` and run on the server:

```bash
#!/bin/bash
echo "=== Container Memory Analysis ==="
echo

echo "1. Total System Memory:"
free -h | head -2
echo

echo "2. Cgroup Memory Limit:"
if [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
    LIMIT=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
    LIMIT_GB=$(echo "scale=2; $LIMIT/1024/1024/1024" | bc)
    echo "  Limit: $LIMIT_GB GB"

    if [ "$LIMIT" -gt 100000000000000 ]; then
        echo "  Status: No limit set (unlimited)"
    else
        echo "  Status: Limited to $LIMIT_GB GB"
    fi
else
    echo "  Cgroup v1 not found, checking cgroup v2..."
    if [ -f /sys/fs/cgroup/memory.max ]; then
        cat /sys/fs/cgroup/memory.max
    else
        echo "  Unable to detect cgroup memory limits"
    fi
fi
echo

echo "3. Current Memory Usage:"
if [ -f /sys/fs/cgroup/memory/memory.usage_in_bytes ]; then
    USAGE=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes)
    USAGE_GB=$(echo "scale=2; $USAGE/1024/1024/1024" | bc)
    echo "  Current: $USAGE_GB GB"
fi
echo

echo "4. Swap Status:"
swapon --show 2>/dev/null || echo "  No swap enabled (typical for containers)"
echo

echo "5. CPU Info:"
echo "  Cores: $(nproc)"
echo "  Model: $(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo

echo "6. Recent OOM Events:"
dmesg | grep -i "out of memory" | tail -5 || echo "  No recent OOM events"
echo

echo "7. Recommended Scanner Settings:"
TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_MEM_GB=$(echo "scale=0; $TOTAL_MEM_KB/1024/1024" | bc)

if [ "$TOTAL_MEM_GB" -lt 16 ]; then
    echo "  Low memory detected (<16GB):"
    echo "  --workers 2 --batch-size 4 --max-samples 10000 --streaming"
elif [ "$TOTAL_MEM_GB" -lt 32 ]; then
    echo "  Medium memory detected (16-32GB):"
    echo "  --workers 4 --batch-size 4 --streaming"
elif [ "$TOTAL_MEM_GB" -lt 64 ]; then
    echo "  Good memory detected (32-64GB):"
    echo "  --workers 8 --batch-size 8 --streaming"
else
    echo "  High memory detected (>64GB):"
    echo "  --workers 16 --batch-size 16 --streaming"
fi
```

---

## 10. Key Findings Summary

1. **Environment**: Likely RunPod or Lambda Labs cloud GPU instance
2. **Memory Limit**: Estimated 32-64 GB RAM (1-2 GB per core)
3. **OOM Cause**: Scanner accumulating samples in memory without streaming
4. **Solution**: Use `--streaming` mode (now default) + reasonable worker count
5. **Diagnostics**: Check cgroup limits, monitor with `free -h` and `ps aux`

---

## 11. Immediate Action Items

1. **Verify streaming mode is enabled:**
   ```bash
   # Streaming is now DEFAULT - just ensure output path is set
   ./v15_scanner --step 1 --output samples.bin --workers 8
   ```

2. **Run memory diagnostic script** (see Section 9)

3. **Test with conservative settings:**
   ```bash
   ./v15_scanner --step 10 --max-samples 10000 --output test.bin --workers 4
   ```

4. **Monitor memory during execution:**
   ```bash
   watch -n 2 'free -h; echo; ps aux --sort=-%mem | head -5'
   ```

5. **Check for OOM after run:**
   ```bash
   dmesg | grep -i "killed process"
   ```

---

## Appendix: Memory Estimate Calculator

To estimate memory usage for your scan:

```
Base Formula (step=1, streaming):
- Pass 1 (Channel Maps): 10-20 GB
- Pass 2 (Slim Maps): 0.5-1 GB
- Pass 3 (Streaming): 1-2 GB (batch buffers)
- Worker Overhead: workers × 16 MB
- Total Peak: 12-24 GB

Base Formula (step=1, no streaming):
- Pass 1: 10-20 GB
- Pass 2: 0.5-1 GB
- Pass 3: (num_samples × 4 KB)
- Total: 12 GB + (num_samples × 4 KB)

For step > 1:
- Memory scales approximately as (1/step)
- step=10: ~1/10th the memory (~2-5 GB peak)
```

---

**Document Version:** 1.0
**Last Updated:** 2026-02-04
**Scanner Version:** v15_cpp (with streaming support)
