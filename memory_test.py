#!/usr/bin/env python3
"""
Memory monitoring test for v15 scanner.

Runs a small scan with memory tracking at each phase to identify
where memory is growing.
"""
import gc
import os
import sys
import time
import threading
import psutil
from dataclasses import dataclass
from typing import List, Optional

# Memory tracking
@dataclass
class MemSnapshot:
    label: str
    timestamp: float
    main_rss_mb: float
    total_rss_mb: float
    worker_count: int

def get_memory_snapshot(label: str) -> MemSnapshot:
    """Get current memory usage for main process and all children."""
    proc = psutil.Process()
    main_rss = proc.memory_info().rss / (1024 * 1024)

    children = proc.children(recursive=True)
    children_rss = sum(c.memory_info().rss / (1024 * 1024) for c in children)

    return MemSnapshot(
        label=label,
        timestamp=time.time(),
        main_rss_mb=main_rss,
        total_rss_mb=main_rss + children_rss,
        worker_count=len(children)
    )

class MemoryMonitor:
    """Background memory monitor that samples every N seconds."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.snapshots: List[MemSnapshot] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._sample_count = 0

    def _monitor_loop(self):
        while self._running:
            self._sample_count += 1
            snap = get_memory_snapshot(f"sample_{self._sample_count}")
            with self._lock:
                self.snapshots.append(snap)
            time.sleep(self.interval)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def mark(self, label: str):
        """Add a labeled snapshot."""
        snap = get_memory_snapshot(label)
        with self._lock:
            self.snapshots.append(snap)
        print(f"  [MEM] {label}: Main={snap.main_rss_mb:.1f}MB, Total={snap.total_rss_mb:.1f}MB, Workers={snap.worker_count}")

    def print_report(self):
        """Print memory report showing growth."""
        print("\n" + "="*70)
        print("MEMORY MONITORING REPORT")
        print("="*70)

        # Get labeled snapshots (not samples)
        labeled = [s for s in self.snapshots if not s.label.startswith("sample_")]
        samples = [s for s in self.snapshots if s.label.startswith("sample_")]

        print(f"\nLabeled checkpoints ({len(labeled)}):")
        print("-" * 70)
        print(f"{'Label':<35} {'Main MB':>10} {'Total MB':>10} {'Workers':>8}")
        print("-" * 70)

        prev_main = 0
        for snap in labeled:
            delta = snap.main_rss_mb - prev_main if prev_main > 0 else 0
            delta_str = f"(+{delta:.0f})" if delta > 10 else ""
            print(f"{snap.label:<35} {snap.main_rss_mb:>10.1f} {snap.total_rss_mb:>10.1f} {snap.worker_count:>8} {delta_str}")
            prev_main = snap.main_rss_mb

        if samples:
            # Find peak during sampling
            peak_total = max(s.total_rss_mb for s in samples)
            peak_main = max(s.main_rss_mb for s in samples)
            print(f"\nPeak during sampling ({len(samples)} samples):")
            print(f"  Main process peak: {peak_main:.1f} MB")
            print(f"  Total (with workers) peak: {peak_total:.1f} MB")

            # Show memory over time
            print(f"\nMemory timeline (sampled every {self.interval}s):")
            print("-" * 50)
            for i, s in enumerate(samples[::max(1, len(samples)//20)]):  # Show ~20 samples
                bar_len = int(s.total_rss_mb / 100)  # Scale: 100MB = 1 char
                bar = "#" * min(bar_len, 50)
                print(f"  {s.total_rss_mb:>8.1f}MB |{bar}")


def run_memory_test():
    """Run scanner with memory monitoring."""
    from v15.data import load_market_data
    from v15.scanner import scan_channels

    monitor = MemoryMonitor(interval=0.5)

    print("="*70)
    print("V15 SCANNER MEMORY TEST")
    print("="*70)
    print(f"PID: {os.getpid()}")
    print(f"Python: {sys.version}")

    monitor.mark("START")

    # Load data
    print("\n[1] Loading market data...")
    data_dir = "data"
    tsla, spy, vix = load_market_data(data_dir)
    print(f"    TSLA: {len(tsla)} bars")
    print(f"    SPY: {len(spy)} bars")
    print(f"    VIX: {len(vix)} bars")
    monitor.mark("AFTER_DATA_LOAD")

    gc.collect()
    monitor.mark("AFTER_GC_1")

    # Run scan with limited samples and workers
    print("\n[2] Starting scan (limited test)...")
    print("    Workers: 4")
    print("    Max samples: 100")
    print("    Step: 50")

    monitor.start()  # Start background sampling

    try:
        samples = scan_channels(
            tsla_df=tsla,
            spy_df=spy,
            vix_df=vix,
            step=50,           # Large step = fewer positions
            warmup_bars=5000,
            workers=4,         # Small number of workers for test
            max_samples=100,   # Limit samples
            progress=True,
            output_path=None,
            incremental_path=None,
        )

        monitor.mark("AFTER_SCAN")
        print(f"\n    Generated {len(samples)} samples")

    except Exception as e:
        monitor.mark(f"ERROR: {e}")
        raise
    finally:
        monitor.stop()

    gc.collect()
    monitor.mark("AFTER_GC_2")

    # Print report
    monitor.print_report()

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    run_memory_test()
