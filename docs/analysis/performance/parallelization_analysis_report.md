# Parallelization Analysis Report

## Scope
Files reviewed for parallelization, training, data loading, and model execution:
- `train.py`
- `v7/training/scanning.py`
- `v7/training/dataset.py`
- `v7/training/trainer.py`
- `v7/models/hierarchical_cfc.py`
- `v7/core/channel.py`
- `v7/training/labels.py`
- `v7/training/example_training.py`
- `v7/training/quick_start.py`
- `v7/features/full_features.py`

## Summary
- The only true multi-process path is channel scanning via `ProcessPoolExecutor` and it is structurally correct (top-level worker functions, initializer, deterministic ordering).
- DataLoader parallelism is enabled only for CUDA; CPU and MPS default to single-worker loading.
- Model execution is single-process; "parallel branches" are logical, not thread-level.
- MPS is supported in the CLI device selection and DataLoader config, but defaults and examples do not select MPS automatically.
- One correctness issue exists in the scan heartbeat timeout: the timeout exception does not propagate, and progress tracking is not updated by per-position progress events.

## 1. Multi-threaded / multi-process correctness

### 1.1 ProcessPoolExecutor for channel scanning
- `_scan_parallel` uses `ProcessPoolExecutor` with an initializer to set worker globals and avoid per-task pickling of large arrays (`v7/training/scanning.py:372-463`).
- Workers run `_process_position_batch` -> `_process_single_position`, reconstructing DataFrames from NumPy arrays and producing `ChannelSample`s (`v7/training/scanning.py:42-219`).
- Results are sorted by index to maintain deterministic order (`v7/training/scanning.py:541-546`).

Correctness: This is a valid multiprocessing design. The main risk is memory overhead because each worker process receives full copies of the arrays.

### 1.2 ThreadPoolExecutor inside each worker
- `detect_channels_multi_window` uses `ThreadPoolExecutor` for parallel window detection (`v7/core/channel.py:455-489`).
- When `_scan_parallel` is enabled, each process also spawns threads, creating nested parallelism.

Correctness: Likely OK because the DataFrame is read-only, but this can oversubscribe CPU cores and reduce performance.

### 1.3 DataLoader workers
- Device-aware DataLoader config enables workers only for CUDA, and uses `persistent_workers` when `num_workers > 0` (`v7/training/dataset.py:1218-1288`).
- Dataset processing occurs in PyTorch DataLoader worker processes.

Correctness: This is standard PyTorch usage. No clear race conditions are introduced by the dataset code.

### 1.4 No distributed or multi-GPU training
- Training is single-process; no DDP or `DataParallel` usage (`v7/training/trainer.py:220-323`).

## 2. MPS compatibility (Apple Silicon)

### Supported paths
- CLI device selection includes MPS (`train.py:1252-1282`).
- DataLoader auto-detects MPS and sets `num_workers=0` (`v7/training/dataset.py:1210-1243`).

### Gaps / risks
- `TrainingConfig` defaults to CUDA/CPU only (no MPS check) (`v7/training/trainer.py:84-86`).
- `create_model()` default device is CUDA/CPU only (`v7/models/hierarchical_cfc.py:1350-1356`).
- Example/quick-start flows ignore MPS (`v7/training/example_training.py:247-267`, `v7/training/quick_start.py:103-107`).
- AMP on MPS: `GradScaler(self.device.type)` and `autocast(self.device.type)` are used when `use_amp=True` (`v7/training/trainer.py:137-141`, `v7/training/trainer.py:271-276`). PyTorch does not reliably support `GradScaler` on MPS, so enabling AMP on MPS can throw or silently disable scaling depending on version.

## 3. CPU-only and other non-MPS backends
- CPU-only is supported via device auto-detect in `train.py` (`train.py:1252-1265`).
- DataLoader defaults to `num_workers=0` for CPU (`v7/training/dataset.py:1241-1243`), which is safe but disables parallel loading.
- Non-MPS accelerators (e.g., XPU) are not auto-detected; users must manually pass device strings.

## 4. Device placement and data movement
- Model and loss are moved to the configured device during Trainer initialization (`v7/training/trainer.py:117-129`).
- Each batch moves features and labels to the device, then concatenates features to build the model input (`v7/training/trainer.py:237-267`).
- Pin-memory is enabled only for CUDA (not MPS/CPU), which is correct (`v7/training/dataset.py:1245-1287`).
- Feature extraction and scanning operate on CPU; there is no device-aware data movement at that stage.

## 5. Thread safety and race conditions

### 5.1 Thread-safe caches
- The core cache uses an `RLock` (thread-safe) but is process-local; each process has its own cache instance (`v7/core/cache.py`).

### 5.2 Label resampling cache is not thread-safe
- `labels.py` maintains a module-level `_resample_cache` with no locking (`v7/training/labels.py:35-73`).
- It is safe in current use (single-threaded per process), but would be unsafe if label generation is invoked from multiple threads in the same process.

### 5.3 Heartbeat timeout logic does not fail the scan
- The heartbeat thread raises `TimeoutError` inside a daemon thread, which does not propagate to the main thread (`v7/training/scanning.py:488-511`).
- `last_progress_time` is only updated when futures complete, not when progress updates are received, so long-running chunks can trigger false warnings/timeouts (`v7/training/scanning.py:514-525`).

## 6. Recommendations (non-code)
- If MPS use is desired outside `train.py`, update device defaults in `TrainingConfig` and `create_model()` to include MPS or ensure callers pass `device='mps'`.
- Keep `use_amp=False` on MPS unless verified with the current PyTorch version.
- If scan timeouts matter, rework the heartbeat to update on progress-queue updates or to signal the main thread rather than raising in a daemon thread.
