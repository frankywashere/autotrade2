"""
Run Management for Training Experiments

This module provides utilities for organizing, tracking, and managing training runs:
- Unique timestamped run IDs and directories
- Configuration saving and loading
- Experiment index maintenance for comparing runs
- Thread-safe updates to shared experiment index

Example usage:
    from v7.training.run_manager import RunManager, generate_run_id

    manager = RunManager(base_dir=Path("runs"))
    run_dir = manager.create_run(name="baseline_model")

    # Save configuration
    manager.save_run_config(run_dir, {"learning_rate": 0.001, ...})

    # After training, update the experiment index
    manager.update_experiments_index(run_dir, {
        "best_val_loss": 0.45,
        "best_direction_acc": 0.72,
        "best_epoch": 15,
        "num_windows": 12,
    })

    # Load all experiments for comparison
    experiments = manager.load_experiments_index()
"""

import json
import os
import fcntl
import tempfile
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


def generate_run_id() -> str:
    """
    Generate a unique run ID based on current timestamp.

    Format: "YYYYMMDD_HHMMSS"

    Returns:
        str: Timestamp-based run ID like "20250109_143022"

    Example:
        >>> run_id = generate_run_id()
        >>> print(run_id)  # "20250109_143022"
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class ExperimentEntry:
    """
    Dataclass representing a single experiment in the experiments index.

    Stores metadata about a training run including performance metrics,
    configuration settings, and status information.

    Attributes:
        run_id: Unique identifier for the run (timestamp-based)
        name: Human-readable name for the experiment
        timestamp: ISO format timestamp when run was created
        status: Current status ("running", "completed", "failed")
        best_val_loss: Best validation loss achieved (None if not yet computed)
        best_direction_acc: Best direction accuracy achieved (None if not yet computed)
        best_epoch: Epoch number where best metrics were achieved
        num_windows: Number of walk-forward windows in the run
        settings: Dictionary containing all model and training settings
        path: Absolute path to the run directory
    """
    run_id: str
    name: str
    timestamp: str
    status: str  # "running", "completed", "failed"
    best_val_loss: Optional[float] = None
    best_direction_acc: Optional[float] = None
    best_epoch: Optional[int] = None
    num_windows: int = 0
    settings: Dict[str, Any] = field(default_factory=dict)
    path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentEntry":
        """Create ExperimentEntry from dictionary."""
        return cls(
            run_id=data.get("run_id", ""),
            name=data.get("name", ""),
            timestamp=data.get("timestamp", ""),
            status=data.get("status", "unknown"),
            best_val_loss=data.get("best_val_loss"),
            best_direction_acc=data.get("best_direction_acc"),
            best_epoch=data.get("best_epoch"),
            num_windows=data.get("num_windows", 0),
            settings=data.get("settings", {}),
            path=data.get("path", ""),
        )


class RunManager:
    """
    Manages training run directories and experiment tracking.

    Provides functionality for:
    - Creating timestamped run directories with standard subdirectory structure
    - Saving and loading run configurations
    - Maintaining a thread-safe experiments index for comparing runs

    Attributes:
        base_dir: Base directory where all runs are stored
        index_file: Path to the experiments_index.json file

    Example:
        >>> manager = RunManager(base_dir=Path("runs"))
        >>> run_dir = manager.create_run(name="test_run")
        >>> print(run_dir)  # Path("runs/20250109_143022_test_run")
    """

    def __init__(self, base_dir: Path = Path("runs")):
        """
        Initialize the RunManager.

        Args:
            base_dir: Base directory for storing runs. Will be created if it
                      doesn't exist. Defaults to "runs" in current directory.
        """
        self.base_dir = Path(base_dir)
        self.index_file = self.base_dir / "experiments_index.json"

        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_run(self, name: str = "") -> Path:
        """
        Create a new run directory with standard subdirectory structure.

        Creates a timestamped directory with the format:
        - "YYYYMMDD_HHMMSS_name/" if name is provided
        - "YYYYMMDD_HHMMSS/" if no name is provided

        Subdirectories created:
        - windows/: For storing walk-forward window checkpoints
        - config/: For storing configuration files
        - logs/: For storing training logs and TensorBoard events

        Args:
            name: Optional human-readable name to append to the run ID.
                  Will be sanitized (spaces replaced with underscores).

        Returns:
            Path: Absolute path to the created run directory

        Example:
            >>> manager = RunManager()
            >>> run_dir = manager.create_run(name="baseline")
            >>> print(run_dir)  # /path/to/runs/20250109_143022_baseline
            >>> print(list(run_dir.iterdir()))
            # [Path('.../windows'), Path('.../config'), Path('.../logs')]
        """
        run_id = generate_run_id()

        # Sanitize name (replace spaces with underscores, remove special chars)
        if name:
            safe_name = name.replace(" ", "_")
            # Keep only alphanumeric, underscores, and hyphens
            safe_name = "".join(c for c in safe_name if c.isalnum() or c in "_-")
            dir_name = f"{run_id}_{safe_name}"
        else:
            dir_name = run_id

        run_dir = self.base_dir / dir_name

        # Create run directory and subdirectories
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "windows").mkdir(exist_ok=True)
        (run_dir / "config").mkdir(exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)

        return run_dir.resolve()

    def save_run_config(self, run_dir: Path, config: Dict[str, Any]) -> Path:
        """
        Save the full configuration for a run.

        Saves the configuration dictionary to run_dir/run_config.json.
        Handles non-serializable objects by converting them to strings.

        Args:
            run_dir: Path to the run directory
            config: Dictionary containing all configuration settings.
                    Can include nested dicts, lists, and primitive types.
                    Non-serializable objects will be converted to strings.

        Returns:
            Path: Path to the saved configuration file

        Example:
            >>> config = {
            ...     "learning_rate": 0.001,
            ...     "batch_size": 32,
            ...     "model_class": "TransformerModel",
            ... }
            >>> manager.save_run_config(run_dir, config)
        """
        config_path = Path(run_dir) / "run_config.json"

        # Make config JSON-serializable
        serializable_config = self._make_serializable(config)

        with open(config_path, "w") as f:
            json.dump(serializable_config, f, indent=2, sort_keys=True)

        return config_path

    def _make_serializable(self, obj: Any) -> Any:
        """
        Recursively convert an object to a JSON-serializable format.

        Args:
            obj: Any Python object

        Returns:
            JSON-serializable version of the object
        """
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(k): self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, "__dict__"):
            # Handle dataclasses and other objects with __dict__
            return self._make_serializable(vars(obj))
        else:
            # Fallback: convert to string representation
            return str(obj)

    def update_experiments_index(self, run_dir: Path, summary: Dict[str, Any]) -> None:
        """
        Update the experiments index with information about a run.

        Creates or updates an entry in runs/experiments_index.json for the
        specified run. Uses file locking for thread-safe updates.

        The summary dict should contain:
        - best_val_loss: Best validation loss (optional)
        - best_direction_acc: Best direction accuracy (optional)
        - best_epoch: Epoch of best metrics (optional)
        - num_windows: Number of walk-forward windows (optional)
        - status: "running", "completed", or "failed" (optional, defaults to "completed")
        - settings: Dict of model/training settings (optional)

        Args:
            run_dir: Path to the run directory
            summary: Dictionary containing run summary information

        Example:
            >>> summary = {
            ...     "best_val_loss": 0.45,
            ...     "best_direction_acc": 0.72,
            ...     "best_epoch": 15,
            ...     "num_windows": 12,
            ...     "status": "completed",
            ... }
            >>> manager.update_experiments_index(run_dir, summary)
        """
        run_dir = Path(run_dir).resolve()

        # Extract run_id and name from directory name
        dir_name = run_dir.name
        parts = dir_name.split("_", 2)  # Split into at most 3 parts

        if len(parts) >= 2:
            run_id = f"{parts[0]}_{parts[1]}"
            name = parts[2] if len(parts) > 2 else ""
        else:
            run_id = dir_name
            name = ""

        # Create experiment entry
        entry = ExperimentEntry(
            run_id=run_id,
            name=name,
            timestamp=datetime.now().isoformat(),
            status=summary.get("status", "completed"),
            best_val_loss=summary.get("best_val_loss"),
            best_direction_acc=summary.get("best_direction_acc"),
            best_epoch=summary.get("best_epoch"),
            num_windows=summary.get("num_windows", 0),
            settings=summary.get("settings", {}),
            path=str(run_dir),
        )

        # Thread-safe update using file locking
        self._update_index_atomic(entry)

    def _update_index_atomic(self, entry: ExperimentEntry) -> None:
        """
        Atomically update the experiments index with a new or updated entry.

        Uses file locking to prevent race conditions and atomic write
        (write to temp file, then rename) for data integrity.

        Args:
            entry: ExperimentEntry to add or update
        """
        # Load existing entries
        entries = self.load_experiments_index()

        # Find and update existing entry or add new one
        entry_dict = entry.to_dict()
        updated = False

        for i, existing in enumerate(entries):
            if existing.get("path") == entry.path or existing.get("run_id") == entry.run_id:
                entries[i] = entry_dict
                updated = True
                break

        if not updated:
            entries.append(entry_dict)

        # Sort by timestamp descending (newest first)
        entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        # Atomic write: write to temp file, then rename
        try:
            # Create temp file in the same directory for atomic rename
            fd, temp_path = tempfile.mkstemp(
                suffix=".json",
                prefix=".experiments_index_",
                dir=self.base_dir
            )

            try:
                with os.fdopen(fd, "w") as f:
                    # Acquire exclusive lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        json.dump(entries, f, indent=2)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

                # Atomic rename
                shutil.move(temp_path, self.index_file)

            except Exception:
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise

        except Exception as e:
            # Fallback: direct write with locking if atomic write fails
            self._update_index_direct(entries)

    def _update_index_direct(self, entries: List[Dict[str, Any]]) -> None:
        """
        Directly write to the index file with file locking.

        Fallback method if atomic write fails.

        Args:
            entries: List of experiment entry dictionaries
        """
        with open(self.index_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(entries, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def load_experiments_index(self) -> List[Dict[str, Any]]:
        """
        Load all experiments from the index file.

        Returns experiments sorted by timestamp descending (newest first).
        Handles missing or corrupted index files gracefully.

        Returns:
            List[dict]: List of experiment dictionaries, each containing
                        all fields from ExperimentEntry. Returns empty list
                        if index doesn't exist or is corrupted.

        Example:
            >>> experiments = manager.load_experiments_index()
            >>> for exp in experiments:
            ...     print(f"{exp['run_id']}: {exp['best_val_loss']}")
        """
        if not self.index_file.exists():
            return []

        try:
            with open(self.index_file, "r") as f:
                # Acquire shared lock for reading
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    content = f.read()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            if not content.strip():
                return []

            entries = json.loads(content)

            if not isinstance(entries, list):
                # Corrupted format, return empty
                return []

            # Ensure sorted by timestamp descending
            entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

            return entries

        except (json.JSONDecodeError, IOError, OSError):
            # Corrupted or unreadable file
            return []

    def get_run_summary(self, run_dir: Path) -> Dict[str, Any]:
        """
        Extract summary information from a run directory.

        Reads the run_config.json and any available metrics files
        to construct a summary dictionary suitable for update_experiments_index.

        Args:
            run_dir: Path to the run directory

        Returns:
            Dict containing summary information:
            - config: Full configuration dictionary (if available)
            - best_val_loss: Best validation loss (if available)
            - best_direction_acc: Best direction accuracy (if available)
            - best_epoch: Best epoch number (if available)
            - num_windows: Number of window subdirectories
            - status: Inferred status based on available files

        Example:
            >>> summary = manager.get_run_summary(run_dir)
            >>> print(f"Best loss: {summary.get('best_val_loss')}")
        """
        run_dir = Path(run_dir)
        summary: Dict[str, Any] = {}

        # Load configuration if available
        config_path = run_dir / "run_config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    summary["settings"] = json.load(f)
            except (json.JSONDecodeError, IOError):
                summary["settings"] = {}
        else:
            summary["settings"] = {}

        # Count windows
        windows_dir = run_dir / "windows"
        if windows_dir.exists():
            summary["num_windows"] = len([
                d for d in windows_dir.iterdir()
                if d.is_dir()
            ])
        else:
            summary["num_windows"] = 0

        # Try to extract best metrics from training_history.json if it exists
        history_path = run_dir / "training_history.json"
        if history_path.exists():
            try:
                with open(history_path, "r") as f:
                    history = json.load(f)

                # Extract best metrics from history
                if "best_val_loss" in history:
                    summary["best_val_loss"] = history["best_val_loss"]
                if "best_direction_acc" in history:
                    summary["best_direction_acc"] = history["best_direction_acc"]
                if "best_epoch" in history:
                    summary["best_epoch"] = history["best_epoch"]

            except (json.JSONDecodeError, IOError):
                pass

        # Try to load metrics from individual window results
        if "best_val_loss" not in summary and windows_dir.exists():
            all_val_losses = []
            all_direction_accs = []

            for window_dir in windows_dir.iterdir():
                if not window_dir.is_dir():
                    continue

                metrics_path = window_dir / "best_metrics.json"
                if metrics_path.exists():
                    try:
                        with open(metrics_path, "r") as f:
                            metrics = json.load(f)
                        if "val_loss" in metrics:
                            all_val_losses.append(metrics["val_loss"])
                        if "direction_acc" in metrics:
                            all_direction_accs.append(metrics["direction_acc"])
                    except (json.JSONDecodeError, IOError):
                        pass

            if all_val_losses:
                summary["best_val_loss"] = min(all_val_losses)
            if all_direction_accs:
                summary["best_direction_acc"] = max(all_direction_accs)

        # Infer status
        completed_marker = run_dir / "completed"
        failed_marker = run_dir / "failed"

        if failed_marker.exists():
            summary["status"] = "failed"
        elif completed_marker.exists():
            summary["status"] = "completed"
        elif any((run_dir / "windows").iterdir()) if (run_dir / "windows").exists() else False:
            summary["status"] = "running"
        else:
            summary["status"] = "unknown"

        return summary

    def list_runs(self) -> List[Path]:
        """
        List all run directories in the base directory.

        Returns:
            List[Path]: List of run directory paths, sorted by name descending
                        (newest first, assuming timestamp-based naming)
        """
        if not self.base_dir.exists():
            return []

        runs = [
            d for d in self.base_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

        return sorted(runs, reverse=True)

    def get_latest_run(self) -> Optional[Path]:
        """
        Get the most recent run directory.

        Returns:
            Path: Path to the most recent run, or None if no runs exist
        """
        runs = self.list_runs()
        return runs[0] if runs else None
