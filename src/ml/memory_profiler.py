"""
Memory Profiler for Training Debugging

Tracks RAM and GPU memory usage throughout training, logs to file for post-hoc analysis.
Detects memory spikes and provides summaries for debugging OOM issues.

Usage:
    python train_hierarchical.py --device cuda --memory-profile
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import gc


class MemoryProfiler:
    """
    Lightweight memory tracking for training debugging.

    Logs memory snapshots to a rotating file and detects spikes.
    Designed to have minimal performance impact on training.
    """

    def __init__(
        self,
        log_path: str = "logs/memory_debug.log",
        device: str = "cpu",
        log_every_n: int = 10,
        spike_threshold_mb: int = 500
    ):
        """
        Initialize the memory profiler.

        Args:
            log_path: Path to the log file (will create directory if needed)
            device: Device type ('cuda', 'mps', 'cpu')
            log_every_n: Log memory every N batches (default: 10)
            spike_threshold_mb: Alert if memory increases by more than this (default: 500MB)
        """
        self.device = device
        self.log_every_n = log_every_n
        self.spike_threshold_mb = spike_threshold_mb

        # Ensure log directory exists
        log_dir = Path(log_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        # Setup rotating file logger (max 10MB, keep 3 backups)
        self.logger = logging.getLogger("memory_profiler")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear any existing handlers

        handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=3
        )
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

        # Memory tracking state
        self.prev_ram_mb = 0
        self.prev_gpu_mb = 0
        self.peak_ram_mb = 0
        self.peak_gpu_mb = 0
        self.baseline_ram_mb = 0
        self.baseline_gpu_mb = 0
        self.spike_count = 0
        self.spike_locations = []
        self.total_snapshots = 0
        self.current_epoch = 0
        self.current_batch = 0

        # Import device-specific memory functions
        self._setup_memory_functions()

        # Record baseline
        self._record_baseline()

        # Log session start
        self._log(f"SESSION_START | device={device} | log_every_n={log_every_n} | spike_threshold={spike_threshold_mb}MB")

    def _setup_memory_functions(self):
        """Setup device-specific memory query functions."""
        self._get_gpu_memory = lambda: (0, 0)  # Default: no GPU

        if self.device == 'cuda':
            try:
                import torch
                if torch.cuda.is_available():
                    def get_cuda_memory():
                        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                        reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
                        return allocated, reserved
                    self._get_gpu_memory = get_cuda_memory
            except ImportError:
                pass

        elif self.device == 'mps':
            try:
                import torch
                if torch.backends.mps.is_available():
                    def get_mps_memory():
                        # MPS doesn't have detailed memory API, estimate from allocated
                        allocated = torch.mps.current_allocated_memory() / (1024 ** 2)
                        return allocated, allocated
                    self._get_gpu_memory = get_mps_memory
            except (ImportError, AttributeError):
                pass

        # RAM monitoring
        try:
            import psutil
            self._psutil_available = True
        except ImportError:
            self._psutil_available = False

    def _get_ram_memory(self) -> tuple:
        """Get RAM usage (used_mb, total_mb)."""
        if self._psutil_available:
            import psutil
            mem = psutil.virtual_memory()
            used_mb = (mem.total - mem.available) / (1024 ** 2)
            total_mb = mem.total / (1024 ** 2)
            return used_mb, total_mb
        return 0, 0

    def _record_baseline(self):
        """Record baseline memory at profiler start."""
        ram_used, _ = self._get_ram_memory()
        gpu_alloc, _ = self._get_gpu_memory()
        self.baseline_ram_mb = ram_used
        self.baseline_gpu_mb = gpu_alloc
        self.prev_ram_mb = ram_used
        self.prev_gpu_mb = gpu_alloc

    def _log(self, message: str):
        """Write timestamped message to log file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.logger.info(f"{timestamp} | {message}")

    def set_epoch(self, epoch: int):
        """Set current epoch for logging context."""
        self.current_epoch = epoch
        self._log(f"EPOCH_START | epoch={epoch}")

    def snapshot(self, phase: str, batch_idx: int = 0, force_log: bool = False) -> Dict:
        """
        Capture memory snapshot and optionally log it.

        Args:
            phase: Training phase (e.g., "forward_end", "backward_end")
            batch_idx: Current batch index
            force_log: Force logging even if not at log interval

        Returns:
            Dict with memory stats for progress bar display
        """
        self.current_batch = batch_idx
        self.total_snapshots += 1

        # Get current memory
        ram_used, ram_total = self._get_ram_memory()
        gpu_alloc, gpu_reserved = self._get_gpu_memory()

        # Update peaks
        self.peak_ram_mb = max(self.peak_ram_mb, ram_used)
        self.peak_gpu_mb = max(self.peak_gpu_mb, gpu_alloc)

        # Check for spikes (always, regardless of logging interval)
        ram_delta = ram_used - self.prev_ram_mb
        gpu_delta = gpu_alloc - self.prev_gpu_mb

        if ram_delta > self.spike_threshold_mb:
            self.spike_count += 1
            self.spike_locations.append({
                'epoch': self.current_epoch,
                'batch': batch_idx,
                'phase': phase,
                'type': 'RAM',
                'delta_mb': ram_delta
            })
            self._log(f"SPIKE | +{ram_delta:.0f}MB RAM in {phase} | epoch={self.current_epoch} | batch={batch_idx}")

        if gpu_delta > self.spike_threshold_mb:
            self.spike_count += 1
            self.spike_locations.append({
                'epoch': self.current_epoch,
                'batch': batch_idx,
                'phase': phase,
                'type': 'GPU',
                'delta_mb': gpu_delta
            })
            self._log(f"SPIKE | +{gpu_delta:.0f}MB GPU in {phase} | epoch={self.current_epoch} | batch={batch_idx}")

        # Log at intervals or when forced
        should_log = force_log or (batch_idx % self.log_every_n == 0)
        if should_log:
            self._log(
                f"MEMORY | RAM: {ram_used:.0f}/{ram_total:.0f}MB | "
                f"GPU: {gpu_alloc:.0f}/{gpu_reserved:.0f}MB | "
                f"phase={phase} | epoch={self.current_epoch} | batch={batch_idx}"
            )

        # Update previous values for delta calculation
        self.prev_ram_mb = ram_used
        self.prev_gpu_mb = gpu_alloc

        # Return compact dict for progress bar
        return {
            'ram_mb': ram_used,
            'ram_total_mb': ram_total,
            'gpu_mb': gpu_alloc,
            'gpu_reserved_mb': gpu_reserved
        }

    def log_phase(self, phase: str, details: Optional[Dict] = None):
        """
        Log a training phase transition.

        Args:
            phase: Phase name (e.g., "forward_start", "cleanup_end")
            details: Optional additional details to log
        """
        detail_str = ""
        if details:
            detail_str = " | " + " | ".join(f"{k}={v}" for k, v in details.items())

        self._log(f"PHASE | {phase} | epoch={self.current_epoch} | batch={self.current_batch}{detail_str}")

    def log_cleanup(self, freed_gpu_mb: float = 0, freed_ram_mb: float = 0):
        """Log memory freed during cleanup."""
        self._log(
            f"CLEANUP | freed_gpu={freed_gpu_mb:.0f}MB | freed_ram={freed_ram_mb:.0f}MB | "
            f"epoch={self.current_epoch} | batch={self.current_batch}"
        )

    def get_summary(self) -> Dict:
        """
        Get summary statistics for the profiling session.

        Returns:
            Dict with peak memory, spike info, and session stats
        """
        return {
            'device': self.device,
            'peak_ram_mb': self.peak_ram_mb,
            'peak_gpu_mb': self.peak_gpu_mb,
            'baseline_ram_mb': self.baseline_ram_mb,
            'baseline_gpu_mb': self.baseline_gpu_mb,
            'ram_growth_mb': self.peak_ram_mb - self.baseline_ram_mb,
            'gpu_growth_mb': self.peak_gpu_mb - self.baseline_gpu_mb,
            'spike_count': self.spike_count,
            'spike_threshold_mb': self.spike_threshold_mb,
            'spike_locations': self.spike_locations[-10:],  # Last 10 spikes
            'total_snapshots': self.total_snapshots,
            'log_every_n': self.log_every_n
        }

    def print_summary(self):
        """Print a human-readable summary to console."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("MEMORY PROFILING SUMMARY")
        print("=" * 60)
        print(f"Device: {summary['device']}")
        print(f"Peak RAM: {summary['peak_ram_mb']:.0f} MB (grew {summary['ram_growth_mb']:.0f} MB from baseline)")
        print(f"Peak GPU: {summary['peak_gpu_mb']:.0f} MB (grew {summary['gpu_growth_mb']:.0f} MB from baseline)")
        print(f"Spikes detected: {summary['spike_count']} (threshold: {summary['spike_threshold_mb']} MB)")

        if summary['spike_locations']:
            print("\nRecent spikes:")
            for spike in summary['spike_locations'][-5:]:
                print(f"  - Epoch {spike['epoch']}, Batch {spike['batch']}: "
                      f"+{spike['delta_mb']:.0f}MB {spike['type']} in {spike['phase']}")

        print("=" * 60 + "\n")

    def close(self):
        """Close the profiler and flush logs."""
        self._log(f"SESSION_END | total_snapshots={self.total_snapshots} | spike_count={self.spike_count}")
        for handler in self.logger.handlers:
            handler.flush()
            handler.close()
