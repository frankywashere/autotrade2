"""
Global error handlers for AutoTrade v7.0

Provides consistent error handling across the codebase.
"""

from contextlib import contextmanager
from typing import Callable, Optional
import logging

from .exceptions import (
    InsufficientDataError,
    FeatureExtractionError,
    PredictionError,
)


logger = logging.getLogger(__name__)


def setup_error_handlers():
    """
    Setup global exception handlers.

    Call this once at application startup.
    """
    # TODO: Setup global exception hooks
    pass


@contextmanager
def handle_errors(
    alert_fn: Optional[Callable] = None,
    critical: bool = False
):
    """
    Context manager for consistent error handling.

    Usage:
        with handle_errors():
            features = extractor.extract(data)

        with handle_errors(alert_fn=send_telegram, critical=True):
            prediction = model.predict(features)

    Args:
        alert_fn: Optional function to call on errors (e.g., send_telegram)
        critical: If True, re-raise exceptions. If False, log and continue.

    Yields:
        None
    """
    try:
        yield

    except InsufficientDataError as e:
        # Expected - just log and continue
        logger.info(f"Insufficient data: {e}")
        if not critical:
            return
        raise

    except FeatureExtractionError as e:
        # Serious but recoverable
        logger.error(f"Feature extraction failed: {e}", exc_info=True)
        if alert_fn:
            alert_fn(severity='high', message=f"Feature extraction failed: {e}")
        if critical:
            raise

    except PredictionError as e:
        # Prediction failed
        logger.error(f"Prediction failed: {e}", exc_info=True)
        if alert_fn:
            alert_fn(severity='critical', message=f"Prediction failed: {e}")
        if critical:
            raise

    except Exception as e:
        # Unexpected error
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        if alert_fn:
            alert_fn(severity='critical', message=f"Unexpected error: {e}")
        raise
