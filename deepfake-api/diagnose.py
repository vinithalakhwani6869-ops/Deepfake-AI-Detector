"""
Backward-compatible diagnostic script.

Prefer:  python ../deepfake-detector/scripts/diagnose.py
"""

import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "deepfake-detector" / "scripts" / "diagnose.py"

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, str(_SCRIPT)]))
