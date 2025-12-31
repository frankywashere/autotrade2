"""
Metrics tracking for AutoTrade v7.0

Tracks performance metrics for training and inference.

Features:
- In-memory metrics storage
- Context managers for timing
- TODO (Week 11): Prometheus integration for production

Usage:
    from src.monitoring import MetricsTracker

    metrics = MetricsTracker()

    # Record metric
    metrics.record('prediction_latency_ms', 42.5)

    # Timing context manager
    with metrics.timer('feature_extraction'):
        features = extractor.extract(data)

    # Get metrics
    stats = metrics.get_stats('prediction_latency_ms')
    print(f"Avg latency: {stats['mean']:.2f}ms")
"""

from collections import defaultdict
from contextlib import contextmanager
import time
from typing import Dict, List, Optional
import numpy as np


class MetricsTracker:
    """
    Track and report metrics.

    In-memory metrics storage for training and inference.
    Prometheus integration added in Week 11.
    """

    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self._prometheus_enabled = False  # TODO: Enable in Week 11

    def record(self, metric_name: str, value: float, tags: Optional[dict] = None):
        """
        Record a metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value
            tags: Optional tags (for filtering/grouping)

        Example:
            metrics.record('duration_mae', 2.5, tags={'epoch': 10})
        """
        # In-memory storage
        self.metrics[metric_name].append(value)

        # TODO (Week 11): Send to Prometheus
        # if self._prometheus_enabled:
        #     self._send_to_prometheus(metric_name, value, tags)

    @contextmanager
    def timer(self, name: str):
        """
        Context manager for timing operations.

        Usage:
            with metrics.timer('feature_extraction'):
                features = extractor.extract(data)

            # Automatically records 'feature_extraction_duration_ms'

        Args:
            name: Name of the operation being timed

        Yields:
            None
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self.record(f'{name}_duration_ms', duration_ms)

    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """
        Get statistics for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Dict with mean, std, min, max, p50, p95, p99, count

        Example:
            stats = metrics.get_stats('prediction_latency_ms')
            print(f"P95 latency: {stats['p95']:.2f}ms")
        """
        values = self.metrics.get(metric_name, [])

        if not values:
            return {
                'count': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'p50': 0.0,
                'p95': 0.0,
                'p99': 0.0,
            }

        values_array = np.array(values)

        return {
            'count': len(values),
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'p50': float(np.percentile(values_array, 50)),
            'p95': float(np.percentile(values_array, 95)),
            'p99': float(np.percentile(values_array, 99)),
        }

    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all tracked metrics.

        Returns:
            Dict mapping metric name to stats dict

        Example:
            all_stats = metrics.get_all_metrics()
            for metric, stats in all_stats.items():
                print(f"{metric}: {stats['mean']:.2f} ± {stats['std']:.2f}")
        """
        return {
            name: self.get_stats(name)
            for name in self.metrics.keys()
        }

    def reset(self, metric_name: Optional[str] = None):
        """
        Reset metrics (clear history).

        Args:
            metric_name: If specified, reset only this metric. Otherwise, reset all.

        Example:
            # Reset single metric
            metrics.reset('prediction_latency_ms')

            # Reset all metrics
            metrics.reset()
        """
        if metric_name:
            if metric_name in self.metrics:
                del self.metrics[metric_name]
        else:
            self.metrics.clear()

    def summary(self) -> str:
        """
        Get human-readable summary of all metrics.

        Returns:
            Formatted string with all metrics

        Example:
            print(metrics.summary())
        """
        lines = ["Metrics Summary:", "=" * 80]

        for metric_name in sorted(self.metrics.keys()):
            stats = self.get_stats(metric_name)
            lines.append(
                f"{metric_name:40s} | "
                f"mean: {stats['mean']:8.2f} | "
                f"std: {stats['std']:8.2f} | "
                f"p95: {stats['p95']:8.2f} | "
                f"count: {stats['count']}"
            )

        lines.append("=" * 80)
        return "\n".join(lines)


# Example usage:
#
# # Training metrics
# metrics = MetricsTracker()
#
# for epoch in range(100):
#     with metrics.timer('epoch'):
#         train_loss = train_epoch(model, loader)
#         metrics.record('train_loss', train_loss)
#
#     # Every 10 epochs, print summary
#     if epoch % 10 == 0:
#         print(metrics.summary())
#
# # Inference metrics
# metrics = MetricsTracker()
#
# for request in requests:
#     with metrics.timer('prediction'):
#         prediction = model.predict(request)
#
#     metrics.record('confidence', prediction.confidence)
#
# # Check latency
# latency_stats = metrics.get_stats('prediction_duration_ms')
# assert latency_stats['p95'] < 100, "P95 latency exceeds 100ms!"
