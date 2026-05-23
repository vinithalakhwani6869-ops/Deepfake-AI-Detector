"""
Full diagnostic — migrated from deepfake-api/diagnose.py.

Run from deepfake-detector/:
    python scripts/diagnose.py
"""

import sys
from pathlib import Path

# Ensure package root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Re-use original diagnostic logic with updated import paths
import traceback

PASS = "✅"
FAIL = "❌"
results = []


def check(label, fn):
    try:
        msg = fn()
        print(f"  {PASS}  {label}: {msg}")
        results.append((label, True, msg))
        return True
    except Exception as e:
        print(f"  {FAIL}  {label}: {e}")
        traceback.print_exc()
        results.append((label, False, str(e)))
        return False


print("\n" + "=" * 60)
print("  DEEPFAKE DETECTOR — FULL DIAGNOSTIC")
print("=" * 60)

print("\n[1] ENVIRONMENT")
check("Python version", lambda: sys.version)

print("\n[2] IMPORTS")
torch_ok = check("torch", lambda: __import__("torch").__version__)
tv_ok = check("torchvision", lambda: __import__("torchvision").__version__)
pil_ok = check("PIL", lambda: __import__("PIL").__version__)
check("fastapi", lambda: __import__("fastapi").__version__)

print("\n[3] TORCH DEVICE")
if torch_ok:
    import torch

    check("CUDA available", lambda: str(torch.cuda.is_available()))
    check(
        "Device",
        lambda: str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    )

print("\n[4] MODEL FILE")
from core.model_registry import resolve_weights_path

MODEL_PATH = resolve_weights_path()
check("weights path resolved", lambda: str(MODEL_PATH))
check("weights exist", lambda: str(MODEL_PATH.exists()))
if MODEL_PATH.exists():
    check("file size", lambda: f"{MODEL_PATH.stat().st_size / 1024 / 1024:.2f} MB")

print("\n[10] DeepfakeDetector MODULE")
check(
    "import DeepfakeDetector",
    lambda: str(__import__("core.detector", fromlist=["DeepfakeDetector"]).DeepfakeDetector),
)

print("\n[11] DeepfakeDetector() INSTANTIATION")
try:
    from core.detector import DeepfakeDetector

    def _instantiate():
        d = DeepfakeDetector()
        return f"device={d.device} model_loaded={d.model is not None}"

    check("DeepfakeDetector()", _instantiate)
except Exception as e:
    print(f"  {FAIL}  Could not instantiate: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
failed = [r for r in results if not r[1]]
if not failed:
    print(f"  {PASS}  Core checks passed.")
else:
    print(f"  {FAIL}  {len(failed)} check(s) failed.")
print("=" * 60 + "\n")
