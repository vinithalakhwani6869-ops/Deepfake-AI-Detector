"""
Backward-compatible FastAPI entry point.

Delegates to deepfake-detector/api/app.py — all routes and behaviour unchanged.
Run:  cd deepfake-api && uvicorn app:app --reload --host 127.0.0.1 --port 8000
"""

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent / "deepfake-detector"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from api.app import app  # noqa: F401, E402

__all__ = ["app"]
