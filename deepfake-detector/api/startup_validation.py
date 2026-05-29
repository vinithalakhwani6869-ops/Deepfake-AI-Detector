"""
Startup and runtime validation utilities for production safety.

Validates:
  - Model checkpoint existence and integrity
  - Configuration parameters
  - Environment variables
  - GPU availability and CUDA setup
  - Disk space for temp files
  - Log directory permissions

Raises:
    RuntimeError: If critical validation fails.
    Warning: Logged if non-critical issues detected.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class StartupValidationError(Exception):
    """Raised when startup validation fails critically."""
    pass


def validate_model_checkpoint(weights_path: Path) -> bool:
    """
    Validate model checkpoint file existence and basic integrity.
    
    Args:
        weights_path: Path to model weights file.
    
    Returns:
        True if checkpoint is valid and readable.
    
    Raises:
        RuntimeError: If checkpoint is missing or unreadable.
    """
    if not weights_path.exists():
        msg = f"Model checkpoint not found: {weights_path}"
        logger.error("[validate] %s", msg)
        raise RuntimeError(msg)
    
    if not weights_path.is_file():
        msg = f"Model checkpoint is not a file: {weights_path}"
        logger.error("[validate] %s", msg)
        raise RuntimeError(msg)
    
    # Check file size (safeguard against truncated downloads)
    file_size_mb = weights_path.stat().st_size / (1024 * 1024)
    if file_size_mb < 1:
        logger.warning(
            "[validate] Model file is suspiciously small: %.2f MB",
            file_size_mb,
        )
    
    # Check read permissions
    if not os.access(weights_path, os.R_OK):
        msg = f"Model checkpoint is not readable: {weights_path}"
        logger.error("[validate] %s", msg)
        raise RuntimeError(msg)
    
    logger.info(
        "[validate] Model checkpoint validated: %s (%.2f MB)",
        weights_path,
        file_size_mb,
    )
    return True


def validate_directory_writable(dir_path: Path, purpose: str) -> bool:
    """
    Validate that a directory exists and is writable.
    
    Creates directory if it doesn't exist.
    
    Args:
        dir_path: Path to directory.
        purpose: Purpose description (e.g., "logs", "temp uploads").
    
    Returns:
        True if directory is writable.
    
    Raises:
        RuntimeError: If directory cannot be written to.
    """
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        msg = f"Cannot create {purpose} directory: {dir_path} ({exc})"
        logger.error("[validate] %s", msg)
        raise RuntimeError(msg) from exc
    
    # Test write permission
    test_file = dir_path / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
    except OSError as exc:
        msg = f"{purpose} directory is not writable: {dir_path} ({exc})"
        logger.error("[validate] %s", msg)
        raise RuntimeError(msg) from exc
    
    logger.info("[validate] Directory writable: %s (%s)", dir_path, purpose)
    return True


def validate_disk_space(dir_path: Path, min_free_mb: int = 100) -> bool:
    """
    Validate that disk has minimum free space.
    
    Args:
        dir_path: Path to check disk space on.
        min_free_mb: Minimum free space required (MB).
    
    Returns:
        True if disk has sufficient free space.
    
    Raises:
        RuntimeError: If disk space is insufficient.
    """
    stat_result = shutil.disk_usage(dir_path)
    free_mb = stat_result.free / (1024 * 1024)
    
    if free_mb < min_free_mb:
        msg = (
            f"Insufficient disk space: {free_mb:.1f} MB free "
            f"(need {min_free_mb} MB)"
        )
        logger.error("[validate] %s", msg)
        raise RuntimeError(msg)
    
    logger.info(
        "[validate] Disk space OK: %.1f MB free (need %d MB)",
        free_mb,
        min_free_mb,
    )
    return True


def validate_gpu_availability() -> bool:
    """
    Validate GPU availability and log device information.
    
    Returns:
        True if CUDA is available, False if CPU-only.
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info("[validate] CUDA available: %d GPU(s)", num_gpus)
        for i in range(num_gpus):
            device_name = torch.cuda.get_device_name(i)
            capability = torch.cuda.get_device_capability(i)
            logger.info(
                "[validate] GPU %d: %s (capability %d.%d)",
                i,
                device_name,
                capability[0],
                capability[1],
            )
        return True
    else:
        logger.warning("[validate] CUDA not available, using CPU (slower)")
        return False


def validate_environment(
    weights_path: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    temp_dir: Optional[Path] = None,
    min_disk_space_mb: int = 100,
) -> dict[str, bool]:
    """
    Run all startup validation checks.
    
    Args:
        weights_path: Path to model weights (if None, skipped).
        log_dir: Path to logs directory (if None, uses ./logs).
        temp_dir: Path to temp uploads directory (if None, uses ./temp_uploads).
        min_disk_space_mb: Minimum required disk space (MB).
    
    Returns:
        Dict with validation results.
    
    Raises:
        StartupValidationError: If critical validation fails.
    """
    results = {}
    
    if log_dir is None:
        log_dir = Path("logs")
    if temp_dir is None:
        temp_dir = Path("temp_uploads")
    
    logger.info("[validate] Starting startup validation...")
    
    # Validate model checkpoint
    if weights_path is not None:
        try:
            results["model_checkpoint"] = validate_model_checkpoint(weights_path)
        except RuntimeError as exc:
            logger.error("[validate] Model checkpoint validation failed: %s", exc)
            raise StartupValidationError(f"Model validation failed: {exc}") from exc
    
    # Validate directories
    try:
        results["log_dir"] = validate_directory_writable(log_dir, "logs")
        results["temp_dir"] = validate_directory_writable(temp_dir, "temp uploads")
    except RuntimeError as exc:
        raise StartupValidationError(f"Directory validation failed: {exc}") from exc
    
    # Validate disk space
    try:
        results["disk_space"] = validate_disk_space(
            log_dir,
            min_disk_space_mb,
        )
    except RuntimeError as exc:
        logger.warning("[validate] Disk space check failed (non-critical): %s", exc)
        results["disk_space"] = False
    
    # Validate GPU
    results["gpu_available"] = validate_gpu_availability()
    
    logger.info("[validate] Startup validation complete: %s", results)
    return results


def safe_shutdown() -> None:
    """
    Gracefully shutdown resources.
    
    Called on SIGTERM or application shutdown.
    """
    logger.info("[shutdown] Initiating graceful shutdown...")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("[shutdown] GPU cache cleared")
    
    # Sync filesystem
    import subprocess
    try:
        subprocess.run(["sync"], check=False, timeout=5)
        logger.info("[shutdown] Filesystem synced")
    except Exception as exc:
        logger.warning("[shutdown] Filesystem sync failed (non-critical): %s", exc)
    
    logger.info("[shutdown] Graceful shutdown complete")
