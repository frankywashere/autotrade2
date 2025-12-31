"""
Monitoring and observability for AutoTrade v7.0

Provides:
- Structured logging (loguru)
- Prometheus metrics
- Health checks
- Alerting
"""

from .logger import get_logger, setup_logging
from .metrics_tracker import MetricsTracker
# from .health_checks import HealthChecker  # TODO: Week 11
# from .alerts import send_alert  # TODO: Week 11

__all__ = [
    'get_logger',
    'setup_logging',
    'MetricsTracker',
]
