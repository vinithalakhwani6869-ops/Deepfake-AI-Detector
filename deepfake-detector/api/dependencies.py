"""
Shared FastAPI dependencies — detector singleton injection.
"""

from __future__ import annotations

from core.detector import DeepfakeDetector

# Loaded once at import time (same as original app.py startup behaviour).
_detector: DeepfakeDetector | None = None


def get_detector() -> DeepfakeDetector:
    """Return the shared DeepfakeDetector instance."""
    global _detector
    if _detector is None:
        _detector = DeepfakeDetector()
    return _detector
