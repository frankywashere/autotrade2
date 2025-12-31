"""
Walk-Forward Validation Results Tracking

This module provides dataclasses for tracking and analyzing walk-forward
validation results across multiple training windows.

Classes:
    WindowMetrics: Metrics and metadata for a single training window
    WalkForwardResults: Aggregated results across all windows with statistics

Features:
    - Complete tracking of training history per window
    - Automatic computation of aggregated statistics
    - JSON serialization/deserialization support
    - Checkpoint path tracking for model recovery
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import json
import numpy as np


@dataclass
class WindowMetrics:
    """
    Metrics and metadata for a single walk-forward training window.

    Attributes:
        window_id: Unique identifier for this window (e.g., 0, 1, 2, ...)
        train_start: Start date of training period (YYYY-MM-DD)
        train_end: End date of training period (YYYY-MM-DD)
        val_start: Start date of validation period (YYYY-MM-DD)
        val_end: End date of validation period (YYYY-MM-DD)
        best_val_metric: Best validation metric achieved during training
        best_val_metric_name: Name of the metric (e.g., 'val_loss', 'val_accuracy')
        epochs_trained: Total number of epochs trained
        best_epoch: Epoch number where best validation metric was achieved
        train_history: Per-epoch training metrics (list of dicts)
        val_history: Per-epoch validation metrics (list of dicts)
        checkpoint_path: Path to saved model checkpoint for this window
        training_time_seconds: Total wall-clock time for training this window
        metadata: Additional window-specific metadata
    """

    window_id: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    best_val_metric: float
    best_val_metric_name: str = 'val_loss'
    epochs_trained: int = 0
    best_epoch: int = 0
    train_history: List[Dict[str, float]] = field(default_factory=list)
    val_history: List[Dict[str, float]] = field(default_factory=list)
    checkpoint_path: Optional[str] = None
    training_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert WindowMetrics to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WindowMetrics':
        """Create WindowMetrics from dictionary."""
        return cls(**data)

    def get_final_train_metrics(self) -> Dict[str, float]:
        """Get training metrics from the final epoch."""
        if not self.train_history:
            return {}
        return self.train_history[-1]

    def get_final_val_metrics(self) -> Dict[str, float]:
        """Get validation metrics from the final epoch."""
        if not self.val_history:
            return {}
        return self.val_history[-1]

    def get_best_val_metrics(self) -> Dict[str, float]:
        """Get validation metrics from the best epoch."""
        if not self.val_history or self.best_epoch >= len(self.val_history):
            return {}
        return self.val_history[self.best_epoch]


@dataclass
class WalkForwardResults:
    """
    Aggregated results from walk-forward validation across multiple windows.

    This class tracks all window results and computes aggregate statistics
    to assess model performance stability and generalization across time.

    Attributes:
        num_windows: Total number of walk-forward windows
        window_type: Type of windowing scheme ('expanding', 'rolling', 'anchored')
        window_metrics: List of WindowMetrics objects, one per window
        best_metric_name: Name of metric used for selecting best models
        creation_timestamp: ISO timestamp when results were created
        metadata: Additional global metadata
    """

    num_windows: int = 0
    window_type: str = 'rolling'  # 'rolling', 'expanding', or 'anchored'
    window_metrics: List[WindowMetrics] = field(default_factory=list)
    best_metric_name: str = 'val_loss'
    creation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_window(self, window: WindowMetrics) -> None:
        """
        Add a new window's metrics to the results.

        Args:
            window: WindowMetrics object for the completed window
        """
        self.window_metrics.append(window)
        self.num_windows = len(self.window_metrics)

    def get_aggregated_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute aggregated statistics across all windows.

        Returns dictionary with mean, std, min, max, and median for each metric.
        Metrics are aggregated from the best epoch of each window.

        Returns:
            Dict mapping metric names to their statistics:
            {
                'val_loss': {'mean': 0.5, 'std': 0.1, 'min': 0.4, 'max': 0.6, 'median': 0.5},
                'val_accuracy': {...},
                ...
            }
        """
        if not self.window_metrics:
            return {}

        # Collect metrics from all windows (use best epoch metrics)
        all_metrics: Dict[str, List[float]] = {}

        for window in self.window_metrics:
            best_metrics = window.get_best_val_metrics()
            for metric_name, metric_value in best_metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(metric_value)

        # Compute statistics for each metric
        aggregated = {}
        for metric_name, values in all_metrics.items():
            values_array = np.array(values)
            aggregated[metric_name] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'median': float(np.median(values_array)),
            }

        return aggregated

    def get_training_time_stats(self) -> Dict[str, float]:
        """
        Get statistics about training time across windows.

        Returns:
            Dict with mean, std, min, max training times in seconds
        """
        if not self.window_metrics:
            return {}

        times = [w.training_time_seconds for w in self.window_metrics]
        times_array = np.array(times)

        return {
            'mean': float(np.mean(times_array)),
            'std': float(np.std(times_array)),
            'min': float(np.min(times_array)),
            'max': float(np.max(times_array)),
            'total': float(np.sum(times_array)),
        }

    def get_best_window(self, mode: str = 'min') -> Optional[WindowMetrics]:
        """
        Find the window with the best validation metric.

        Args:
            mode: 'min' for metrics where lower is better, 'max' for higher is better

        Returns:
            WindowMetrics object for the best window, or None if no windows
        """
        if not self.window_metrics:
            return None

        if mode == 'min':
            return min(self.window_metrics, key=lambda w: w.best_val_metric)
        else:
            return max(self.window_metrics, key=lambda w: w.best_val_metric)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert WalkForwardResults to dictionary for serialization.

        Returns:
            Dictionary representation of all results
        """
        return {
            'num_windows': self.num_windows,
            'window_type': self.window_type,
            'window_metrics': [w.to_dict() for w in self.window_metrics],
            'best_metric_name': self.best_metric_name,
            'creation_timestamp': self.creation_timestamp,
            'metadata': self.metadata,
            'aggregated_metrics': self.get_aggregated_metrics(),
            'training_time_stats': self.get_training_time_stats(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WalkForwardResults':
        """
        Create WalkForwardResults from dictionary.

        Args:
            data: Dictionary representation of results

        Returns:
            WalkForwardResults object
        """
        # Extract window metrics and convert to objects
        window_dicts = data.pop('window_metrics', [])
        windows = [WindowMetrics.from_dict(w) for w in window_dicts]

        # Remove computed fields that shouldn't be in constructor
        data.pop('aggregated_metrics', None)
        data.pop('training_time_stats', None)

        # Create object
        results = cls(**data)
        results.window_metrics = windows

        return results

    def save(self, filepath: Path) -> None:
        """
        Save results to JSON file.

        Args:
            filepath: Path where JSON file should be saved
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        print(f"Walk-forward results saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'WalkForwardResults':
        """
        Load results from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            WalkForwardResults object
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)

    def print_summary(self) -> None:
        """Print a human-readable summary of walk-forward results."""
        print("\n" + "=" * 60)
        print("WALK-FORWARD VALIDATION RESULTS")
        print("=" * 60)
        print(f"Window Type: {self.window_type}")
        print(f"Number of Windows: {self.num_windows}")
        print(f"Best Metric: {self.best_metric_name}")
        print()

        # Per-window summary
        print("Per-Window Results:")
        print("-" * 60)
        for window in self.window_metrics:
            print(f"Window {window.window_id}:")
            print(f"  Train: {window.train_start} to {window.train_end}")
            print(f"  Val:   {window.val_start} to {window.val_end}")
            print(f"  Best {self.best_metric_name}: {window.best_val_metric:.4f} (epoch {window.best_epoch})")
            print(f"  Epochs: {window.epochs_trained}, Time: {window.training_time_seconds:.1f}s")
            if window.checkpoint_path:
                print(f"  Checkpoint: {window.checkpoint_path}")
            print()

        # Aggregated statistics
        print("Aggregated Statistics:")
        print("-" * 60)
        agg_metrics = self.get_aggregated_metrics()
        for metric_name, stats in agg_metrics.items():
            print(f"{metric_name}:")
            print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  Median: {stats['median']:.4f}")
        print()

        # Training time statistics
        time_stats = self.get_training_time_stats()
        if time_stats:
            print("Training Time:")
            print(f"  Total: {time_stats['total']:.1f}s ({time_stats['total']/60:.1f} min)")
            print(f"  Per Window: {time_stats['mean']:.1f}s ± {time_stats['std']:.1f}s")

        print("=" * 60 + "\n")
