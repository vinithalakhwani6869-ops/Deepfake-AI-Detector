"""
Shared FastAPI dependencies — detector singleton injection.

Production enhancements:
  - Thread-safe singleton pattern
  - Lazy initialization support
  - Clear error propagation to API layer
  - Logging for dependency injection events
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Optional

from core.detector import DeepfakeDetector

logger = logging.getLogger(__name__)

# Loaded once at import time (same as original app.py startup behaviour).
_detector: Optional[DeepfakeDetector] = None
_detector_lock = Lock()  # Thread-safe initialization


def get_detector() -> DeepfakeDetector:
    """
    Return the shared DeepfakeDetector instance (singleton).
    
    Thread-safe lazy initialization:
      - First call creates and initializes detector
      - Subsequent calls return cached instance
      - Lock prevents race conditions in multi-threaded servers
    
    Returns:
        DeepfakeDetector instance.
    
    Raises:
        RuntimeError: If detector initialization fails.
    """
    global _detector
    
    if _detector is not None:
        return _detector
    
    with _detector_lock:
        # Double-check after acquiring lock
        if _detector is not None:
            return _detector
        
        logger.info("[dependencies] Initializing detector singleton...")
        try:
            _detector = DeepfakeDetector(lazy_load=True)
            logger.info("[dependencies] Detector singleton created")
        except Exception as exc:
            logger.error(
                "[dependencies] Failed to create detector: %s",
                exc,
            )
            raise RuntimeError(
                f"Detector initialization failed: {exc}"
            ) from exc
    
    return _detector
