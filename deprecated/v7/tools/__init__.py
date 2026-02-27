"""
Tools module for v7 channel prediction system.

Contains utilities for:
- Label inspection and visualization
- Cache precomputation and validation
- Channel visualization
"""

from .label_inspector import (
    detect_suspicious_sample,
    detect_suspicious_samples,
    SuspiciousFlag,
    SuspiciousResult,
)

__all__ = [
    'detect_suspicious_sample',
    'detect_suspicious_samples',
    'SuspiciousFlag',
    'SuspiciousResult',
]
