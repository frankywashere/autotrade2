"""
Structured logging for AutoTrade v7.0

Uses loguru for clean, structured logging with automatic context.

Features:
- Automatic JSON formatting
- Log rotation (1 GB files, 30 day retention)
- Performance tracking (log timing automatically)
- Error context (stack traces with locals)

Usage:
    from src.monitoring import get_logger

    logger = get_logger(__name__)

    logger.info("Training started", epoch=1, batch_size=256, lr=0.001)
    logger.error("Batch failed", batch_idx=123, error=str(e))

    with logger.contextualize(request_id=req_id):
        logger.info("Processing request")  # Automatically includes request_id
"""

from loguru import logger
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    rotation: str = "1 GB",
    retention: str = "30 days",
    enable_json: bool = False,
):
    """
    Setup structured logging with loguru.

    Call this once at application startup.

    Args:
        log_dir: Directory for log files
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: When to rotate log files (size or time)
        retention: How long to keep old logs
        enable_json: If True, use JSON format (for log aggregation)

    Example:
        # Training (detailed logging)
        setup_logging(log_level="DEBUG")

        # Production (JSON for aggregation)
        setup_logging(log_level="INFO", enable_json=True)
    """
    # Remove default handler
    logger.remove()

    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Console handler (human-readable)
    if not enable_json:
        logger.add(
            sys.stderr,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            level=log_level,
            colorize=True,
        )
    else:
        # JSON format for production log aggregation
        logger.add(
            sys.stderr,
            format="{message}",  # JSON already includes timestamp, level, etc.
            level=log_level,
            serialize=True,  # Output as JSON
        )

    # File handler (detailed logs)
    logger.add(
        log_path / "autotrade_{time:YYYY-MM-DD}.log",
        rotation=rotation,
        retention=retention,
        level=log_level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        ),
        enqueue=True,  # Async logging (doesn't block)
        backtrace=True,  # Include full traceback
        diagnose=True,  # Include variable values in traceback
    )

    # Error file (errors only)
    logger.add(
        log_path / "autotrade_error_{time:YYYY-MM-DD}.log",
        rotation=rotation,
        retention=retention,
        level="ERROR",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} | {message}\n{exception}"
        ),
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    logger.info("Logging setup complete", log_dir=str(log_path), level=log_level)


def get_logger(name: Optional[str] = None):
    """
    Get logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance

    Usage:
        logger = get_logger(__name__)
        logger.info("Message", key=value)
    """
    if name:
        return logger.bind(module=name)
    return logger


# Example usage patterns:
#
# 1. Basic logging:
#    logger = get_logger(__name__)
#    logger.info("Training started", epoch=1, lr=0.001)
#
# 2. Error logging:
#    try:
#        risky_operation()
#    except Exception as e:
#        logger.error("Operation failed", error=str(e), exc_info=True)
#
# 3. Contextual logging:
#    with logger.contextualize(request_id="abc123"):
#        logger.info("Processing")  # Includes request_id automatically
#
# 4. Performance timing:
#    import time
#    start = time.perf_counter()
#    expensive_operation()
#    duration = time.perf_counter() - start
#    logger.info("Operation complete", duration_ms=duration * 1000)
#
# 5. Structured data:
#    logger.info(
#        "Model prediction",
#        duration_bars=12.5,
#        confidence=0.87,
#        selected_tf='4h'
#    )
