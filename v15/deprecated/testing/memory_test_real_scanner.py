#!/usr/bin/env python3
"""
Memory test using the REAL scanner code path.
Monitors memory at each phase to identify where growth occurs.
"""
import gc
import os
import sys
import time
import threading
import psutil

def get_mem():
    """Get memory for main process and all children."""
    proc = psutil.Process()
    main = proc.memory_info().rss / (1024**2)
    children = proc.children(recursive=True)
    child_mem = sum(c.memory_info().rss / (1024**2) for c in children)
    return main, child_mem, len(children)

def log_mem(label):
    main, child, n = get_mem()
    total = main + child
    print(f"[MEM] {label}: Main={main:.0f}MB, Children={child:.0f}MB ({n} procs), Total={total:.0f}MB")
    return total

class MemoryWatcher(threading.Thread):
    """Background thread that logs memory every N seconds."""
    def __init__(self, interval=2.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.running = True
        self.peak_total = 0
        self.samples = []

    def run(self):
        while self.running:
            main, child, n = get_mem()
            total = main + child
            self.peak_total = max(self.peak_total, total)
            self.samples.append((time.time(), total, n))
            time.sleep(self.interval)

    def stop(self):
        self.running = False


def main():
    print("="*70)
    print("REAL SCANNER MEMORY TEST")
    print("="*70)
    print(f"PID: {os.getpid()}")

    # Start background memory watcher
    watcher = MemoryWatcher(interval=1.0)
    watcher.start()

    log_mem("START")

    # Import and run the scanner
    from v15.data import load_market_data
    from v15.scanner import scan_channels

    print("\n[1] Loading market data...")
    tsla, spy, vix = load_market_data("data")
    print(f"    Loaded {len(tsla)} bars")
    log_mem("AFTER_DATA_LOAD")

    # Run with limited workers and samples for testing
    NUM_WORKERS = 8  # Enough to see multiprocessing behavior
    MAX_SAMPLES = 200  # Limit for test
    STEP = 100  # Larger step = fewer positions

    print(f"\n[2] Running scan_channels...")
    print(f"    Workers: {NUM_WORKERS}")
    print(f"    Max samples: {MAX_SAMPLES}")
    print(f"    Step: {STEP}")

    start_time = time.time()

    try:
        samples = scan_channels(
            tsla_df=tsla,
            spy_df=spy,
            vix_df=vix,
            step=STEP,
            warmup_bars=5000,
            workers=NUM_WORKERS,
            max_samples=MAX_SAMPLES,
            progress=True,
            output_path=None,
            incremental_path=None,
        )

        elapsed = time.time() - start_time
        log_mem("AFTER_SCAN")
        print(f"\n    Generated {len(samples)} samples in {elapsed:.1f}s")

    except Exception as e:
        log_mem(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Stop watcher
    watcher.stop()

    # Cleanup
    gc.collect()
    log_mem("AFTER_GC")

    # Print memory timeline
    print("\n" + "="*70)
    print("MEMORY TIMELINE (sampled every 1s)")
    print("="*70)

    if watcher.samples:
        start_ts = watcher.samples[0][0]
        prev_total = 0
        for ts, total, n_procs in watcher.samples:
            elapsed = ts - start_ts
            delta = total - prev_total if prev_total > 0 else 0
            delta_str = f" (+{delta:.0f})" if delta > 50 else ""
            bar = "#" * int(total / 100)
            print(f"  {elapsed:6.1f}s | {total:6.0f}MB | {n_procs:2d} procs | {bar}{delta_str}")
            prev_total = total

        print(f"\n  Peak total memory: {watcher.peak_total:.0f}MB")

        # Check for growth
        if len(watcher.samples) > 10:
            first_10_avg = sum(s[1] for s in watcher.samples[:10]) / 10
            last_10_avg = sum(s[1] for s in watcher.samples[-10:]) / 10
            growth = last_10_avg - first_10_avg
            if growth > 100:
                print(f"  WARNING: Memory grew by {growth:.0f}MB during scan!")
            else:
                print(f"  Memory relatively stable (delta: {growth:.0f}MB)")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
