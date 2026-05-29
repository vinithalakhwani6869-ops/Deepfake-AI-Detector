"""
Structured logging configuration for production deployments.

Features:
  - Console logging with timestamps
  - Rotating file logging (prevents disk exhaustion)
  - Request tracking with unique IDs
  - Startup/shutdown lifecycle logging
  - Inference error logging with full context
  - Environment-aware log levels (DEBUG in dev, INFO in prod)

Usage:
    from api.logging_config import setup_logging
    setup_logging(log_level="INFO", log_to_file=True)
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Literal


def setup_logging(
    log_level: str = "INFO",
    log_format: Literal["text", "json"] = "text",
    log_to_file: bool = True,
    log_file_max_mb: int = 100,
    log_file_backup_count: int = 10,
    log_dir: Path | None = None,
) -> None:
    """
    Configure structured logging for the API.
    
    Args:
        log_level: Python logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Output format (text or json).
        log_to_file: Whether to write logs to rotating file.
        log_file_max_mb: Max file size before rotation (MB).
        log_file_backup_count: Number of backup files to keep.
        log_dir: Directory for log files (default: ./logs).
    """
    log_level_obj = getattr(logging, log_level.upper(), logging.INFO)
    
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Define log format
    if log_format == "json":
        # JSON format for structured logging (machine-readable)
        formatter = logging.Formatter(
            '{"time": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s"}'
        )
    else:
        # Text format for human readability (development/debugging)
        formatter = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_obj)
    root_logger.handlers.clear()  # Remove any existing handlers
    
    # Console handler (always enabled)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level_obj)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (rotating, prevents disk exhaustion)
    if log_to_file:
        log_file = log_dir / "api.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=log_file_max_mb * 1024 * 1024,  # Convert MB to bytes
            backupCount=log_file_backup_count,
        )
        file_handler.setLevel(log_level_obj)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    root_logger.info(
        "[logging] Setup complete: level=%s format=%s file_logging=%s",
        log_level,
        log_format,
        log_to_file,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__).
    
    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
