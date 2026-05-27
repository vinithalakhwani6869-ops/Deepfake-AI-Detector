"""
Backward-compatible detector module.

Re-exports DeepfakeDetector from the refactored core package.
"""

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent / "deepfake-detector"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from core.detector import DeepfakeDetector  # noqa: F401, E402

__all__ = ["DeepfakeDetector"]
