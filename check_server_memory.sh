#!/bin/bash
#
# Server Memory Diagnostic Script
# For use on /workspace/autotrade2 or similar cloud GPU environments
#
# Usage: bash check_server_memory.sh
#

set -e

echo "=============================================="
echo "  Server Memory Diagnostic Report"
echo "=============================================="
echo "Generated: $(date)"
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo "PWD: $(pwd)"
echo

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_section() {
    echo
    echo "----------------------------------------"
    echo "  $1"
    echo "----------------------------------------"
}

print_warning() {
    echo -e "${YELLOW}⚠ WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
}

print_ok() {
    echo -e "${GREEN}✓ $1${NC}"
}

# 1. System Information
print_section "1. System Information"

echo "Kernel: $(uname -r)"
echo "OS: $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d'"' -f2 || echo 'Unknown')"
echo "Architecture: $(uname -m)"
echo "CPU Cores: $(nproc)"
echo "CPU Model: $(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs || echo 'Unknown')"

# 2. Total System Memory
print_section "2. Total System Memory"

free -h
echo
TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_MEM_GB=$(echo "scale=2; $TOTAL_MEM_KB/1024/1024" | bc)
AVAILABLE_MEM_KB=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
AVAILABLE_MEM_GB=$(echo "scale=2; $AVAILABLE_MEM_KB/1024/1024" | bc)

echo "Total Memory: ${TOTAL_MEM_GB} GB"
echo "Available Memory: ${AVAILABLE_MEM_GB} GB"

if (( $(echo "$AVAILABLE_MEM_GB < 8" | bc -l) )); then
    print_warning "Low available memory (<8GB) - scanner may fail"
elif (( $(echo "$AVAILABLE_MEM_GB < 16" | bc -l) )); then
    print_warning "Limited memory (<16GB) - use conservative settings"
else
    print_ok "Sufficient memory available (${AVAILABLE_MEM_GB} GB)"
fi

# 3. Container/Cgroup Memory Limits
print_section "3. Container Memory Limits"

CGROUP_V1_LIMIT=""
CGROUP_V2_LIMIT=""

# Check cgroup v1
if [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
    CGROUP_V1_LIMIT=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
    CGROUP_V1_LIMIT_GB=$(echo "scale=2; $CGROUP_V1_LIMIT/1024/1024/1024" | bc)

    echo "Cgroup v1 detected:"
    echo "  Limit: ${CGROUP_V1_LIMIT_GB} GB"

    # Check if limit is effectively unlimited (very large number)
    if [ "$CGROUP_V1_LIMIT" -gt 100000000000000 ]; then
        print_ok "No effective memory limit (unlimited)"
    else
        echo "  Current Usage: $(cat /sys/fs/cgroup/memory/memory.usage_in_bytes | awk '{print $1/1024/1024/1024}') GB"
        echo "  Max Usage: $(cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes | awk '{print $1/1024/1024/1024}') GB"
        print_warning "Container has memory limit: ${CGROUP_V1_LIMIT_GB} GB"
    fi
fi

# Check cgroup v2
if [ -f /sys/fs/cgroup/memory.max ]; then
    CGROUP_V2_LIMIT=$(cat /sys/fs/cgroup/memory.max)
    echo "Cgroup v2 detected:"
    echo "  Limit: $CGROUP_V2_LIMIT"

    if [ "$CGROUP_V2_LIMIT" = "max" ]; then
        print_ok "No effective memory limit (unlimited)"
    else
        CGROUP_V2_LIMIT_GB=$(echo "scale=2; $CGROUP_V2_LIMIT/1024/1024/1024" | bc)
        print_warning "Container has memory limit: ${CGROUP_V2_LIMIT_GB} GB"
    fi
fi

if [ -z "$CGROUP_V1_LIMIT" ] && [ -z "$CGROUP_V2_LIMIT" ]; then
    echo "No cgroup limits detected (may not be containerized)"
fi

# 4. Swap Status
print_section "4. Swap Configuration"

SWAP_TOTAL=$(free -h | grep Swap | awk '{print $2}')
if [ "$SWAP_TOTAL" = "0B" ] || [ -z "$SWAP_TOTAL" ]; then
    print_warning "No swap enabled (typical for containers)"
    echo "  Scanner will be killed immediately if it exceeds memory limit"
else
    echo "Swap available: $SWAP_TOTAL"
    swapon --show 2>/dev/null || echo "  (unable to show swap details)"
fi

# 5. Recent OOM Events
print_section "5. Out-of-Memory Events"

OOM_COUNT=$(dmesg 2>/dev/null | grep -ic "out of memory" || echo "0")
if [ "$OOM_COUNT" -gt 0 ]; then
    print_error "Found $OOM_COUNT OOM events in kernel log"
    echo
    echo "Recent OOM kills:"
    dmesg 2>/dev/null | grep -i "killed process" | tail -5 || echo "  (no details available)"
    echo
    echo "Last OOM event:"
    dmesg 2>/dev/null | grep -i "out of memory" | tail -3 || echo "  (no details available)"
else
    print_ok "No OOM events detected"
fi

# 6. Current Memory Usage
print_section "6. Current Memory Usage (Top Processes)"

echo "Top 5 memory consumers:"
ps aux --sort=-%mem | head -6 | tail -5

# 7. Disk Space (for output files)
print_section "7. Disk Space"

echo "Current directory: $(pwd)"
df -h . | tail -1
DISK_AVAIL=$(df . | tail -1 | awk '{print $4}')
DISK_AVAIL_GB=$(echo "scale=2; $DISK_AVAIL/1024/1024" | bc)

echo "Available disk space: ${DISK_AVAIL_GB} GB"

if (( $(echo "$DISK_AVAIL_GB < 10" | bc -l) )); then
    print_error "Low disk space (<10GB) - may not be sufficient for output"
elif (( $(echo "$DISK_AVAIL_GB < 50" | bc -l) )); then
    print_warning "Limited disk space (<50GB)"
else
    print_ok "Sufficient disk space available"
fi

# 8. Recommended Scanner Configuration
print_section "8. Recommended Scanner Configuration"

RECOMMENDED_WORKERS=8
RECOMMENDED_BATCH=8
RECOMMENDED_STEP=1
ADDITIONAL_FLAGS=""

if (( $(echo "$TOTAL_MEM_GB < 16" | bc -l) )); then
    RECOMMENDED_WORKERS=2
    RECOMMENDED_BATCH=4
    RECOMMENDED_STEP=10
    ADDITIONAL_FLAGS="--max-samples 10000"
    print_warning "Low memory system detected"
elif (( $(echo "$TOTAL_MEM_GB < 32" | bc -l) )); then
    RECOMMENDED_WORKERS=4
    RECOMMENDED_BATCH=4
    RECOMMENDED_STEP=5
    print_warning "Medium memory system detected"
elif (( $(echo "$TOTAL_MEM_GB < 64" | bc -l) )); then
    RECOMMENDED_WORKERS=8
    RECOMMENDED_BATCH=8
    RECOMMENDED_STEP=1
    print_ok "Good memory system detected"
else
    RECOMMENDED_WORKERS=16
    RECOMMENDED_BATCH=16
    RECOMMENDED_STEP=1
    print_ok "High memory system detected"
fi

echo
echo "Recommended command:"
echo
echo "./v15_scanner \\"
echo "  --step $RECOMMENDED_STEP \\"
echo "  --output samples.bin \\"
echo "  --workers $RECOMMENDED_WORKERS \\"
echo "  --batch-size $RECOMMENDED_BATCH \\"
echo "  --streaming \\"
echo "  --data-dir /workspace/autotrade2/data $ADDITIONAL_FLAGS"
echo

# 9. Environment Detection
print_section "9. Environment Detection"

if [ -d /workspace ]; then
    echo "Path structure: /workspace detected"
    if [ -f /.dockerenv ]; then
        echo "Environment: Docker container"
    elif [ -d /var/run/secrets/kubernetes.io ]; then
        echo "Environment: Kubernetes pod"
    else
        echo "Environment: Likely RunPod or Lambda Labs"
    fi
else
    echo "Path structure: Not a standard /workspace environment"
fi

if command -v nvidia-smi &> /dev/null; then
    echo
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    echo "No NVIDIA GPU detected (or nvidia-smi not available)"
fi

# 10. Quick Memory Test
print_section "10. Memory Allocation Test"

echo "Testing memory allocation (allocating 1GB)..."
python3 -c "
import sys
try:
    # Try to allocate 1GB
    test = bytearray(1024 * 1024 * 1024)
    print('✓ Successfully allocated 1GB')
    del test
except MemoryError:
    print('✗ Failed to allocate 1GB - severe memory constraint!')
    sys.exit(1)
" || print_error "Memory allocation test failed"

# Summary
print_section "Summary and Action Items"

echo "1. Total Memory: ${TOTAL_MEM_GB} GB"
echo "2. Available Memory: ${AVAILABLE_MEM_GB} GB"
echo "3. Container Limit: ${CGROUP_V1_LIMIT_GB:-unlimited} GB"
echo "4. OOM Events: $OOM_COUNT"
echo "5. Recommended Workers: $RECOMMENDED_WORKERS"
echo

if [ "$OOM_COUNT" -gt 0 ]; then
    echo "⚠ ACTION REQUIRED: OOM events detected!"
    echo "  - Use streaming mode (default)"
    echo "  - Reduce workers to $RECOMMENDED_WORKERS"
    echo "  - Monitor with: watch -n 5 'free -h'"
fi

if (( $(echo "$AVAILABLE_MEM_GB < 16" | bc -l) )); then
    echo "⚠ ACTION REQUIRED: Low memory detected!"
    echo "  - Use conservative settings (see section 8)"
    echo "  - Enable streaming mode"
    echo "  - Consider increasing step size to reduce memory usage"
fi

echo
echo "=============================================="
echo "  Report Complete"
echo "=============================================="
echo
echo "To monitor memory during scanner execution:"
echo "  watch -n 5 'free -h; echo; ps aux --sort=-%mem | head -5'"
echo
echo "To check for new OOM events:"
echo "  dmesg | grep -i \"out of memory\" | tail -5"
echo
