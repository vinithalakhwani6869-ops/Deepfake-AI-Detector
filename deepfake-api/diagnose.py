"""
diagnose.py — DeepFake Detector full diagnostic
Run from your project root:  python diagnose.py
Paste the output back to get the exact fix.
"""

import sys
import traceback
from pathlib import Path

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

print("\n" + "="*60)
print("  DEEPFAKE DETECTOR — FULL DIAGNOSTIC")
print("="*60)

# ── 1. Environment ─────────────────────────────────────────
print("\n[1] ENVIRONMENT")
check("Python version", lambda: sys.version)

# ── 2. Imports ─────────────────────────────────────────────
print("\n[2] IMPORTS")
torch_ok   = check("torch",       lambda: __import__("torch").__version__)
tv_ok      = check("torchvision", lambda: __import__("torchvision").__version__)
pil_ok     = check("PIL",         lambda: __import__("PIL").__version__)
fastapi_ok = check("fastapi",     lambda: __import__("fastapi").__version__)

# ── 3. Device ──────────────────────────────────────────────
print("\n[3] TORCH DEVICE")
if torch_ok:
    import torch
    check("CUDA available", lambda: str(torch.cuda.is_available()))
    check("Device",         lambda: str(torch.device("cuda" if torch.cuda.is_available() else "cpu")))

# ── 4. Model file ──────────────────────────────────────────
print("\n[4] MODEL FILE")
BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "deepfake_model.pth"
check("model/ dir exists",         lambda: str((BASE_DIR / "model").exists()))
check("deepfake_model.pth exists", lambda: str(MODEL_PATH.exists()))
if MODEL_PATH.exists():
    check("file size", lambda: f"{MODEL_PATH.stat().st_size / 1024 / 1024:.2f} MB")

# ── 5. Raw checkpoint load ────────────────────────────────
print("\n[5] CHECKPOINT LOADING")
if torch_ok and MODEL_PATH.exists():
    def load_checkpoint():
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            keys = list(ckpt.keys())[:5]
            return f"type=dict  top_level_keys={keys}"
        return f"type={type(ckpt).__name__}"
    ck_ok = check("torch.load()", load_checkpoint)

    if ck_ok:
        def inspect_state_dict():
            ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                sd = (ckpt.get("model_state_dict")
                      or ckpt.get("state_dict")
                      or ckpt)
            else:
                sd = ckpt
            keys        = list(sd.keys())
            has_prefix  = any(k.startswith("model.") for k in keys)
            sample_key  = keys[0]
            sample_shape = tuple(sd[sample_key].shape)
            return (f"total_keys={len(keys)}  "
                    f"has_model_prefix={has_prefix}  "
                    f"first_3={keys[:3]}  "
                    f"last_3={keys[-3:]}  "
                    f"sample_shape['{sample_key}']={sample_shape}")
        check("state_dict inspection", inspect_state_dict)

# ── 6. Architecture build ────────────────────────────────
print("\n[6] ARCHITECTURE — EfficientNet-B0")
if torch_ok and tv_ok:
    import torch.nn as nn
    from torchvision import models

    def build_b0():
        m    = models.efficientnet_b0(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_f, 2),
        )
        total = sum(p.numel() for p in m.parameters())
        return f"in_features={in_f}  total_params={total:,}"
    arch_ok = check("build EfficientNet-B0", build_b0)

    # ── 7. Load weights into model ────────────────────────
    print("\n[7] LOAD WEIGHTS INTO MODEL")
    if arch_ok and MODEL_PATH.exists():
        def full_load():
            m    = models.efficientnet_b0(weights=None)
            in_f = m.classifier[1].in_features
            m.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_f, 2),
            )
            ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                sd = (ckpt.get("model_state_dict")
                      or ckpt.get("state_dict")
                      or ckpt)
            else:
                sd = ckpt
            sd = {(k[6:] if k.startswith("model.") else k): v
                  for k, v in sd.items()}
            m.load_state_dict(sd, strict=True)
            m.eval()
            return "OK — strict=True passed"
        load_ok = check("load_state_dict strict=True", full_load)

        if not load_ok:
            def loose_load():
                m    = models.efficientnet_b0(weights=None)
                in_f = m.classifier[1].in_features
                m.classifier = nn.Sequential(
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(in_f, 2),
                )
                ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict):
                    sd = (ckpt.get("model_state_dict")
                          or ckpt.get("state_dict")
                          or ckpt)
                else:
                    sd = ckpt
                sd = {(k[6:] if k.startswith("model.") else k): v
                      for k, v in sd.items()}
                missing, unexpected = m.load_state_dict(sd, strict=False)
                return (f"missing_keys={missing[:5]}  "
                        f"unexpected_keys={unexpected[:5]}")
            check("load_state_dict strict=False (mismatch details)", loose_load)

# ── 8. PIL + transform pipeline ──────────────────────────
print("\n[8] PIL + TRANSFORM")
if pil_ok:
    from PIL import Image, ImageOps
    import io

    def make_jpeg():
        img = Image.new("RGB", (32, 32), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        return buf

    def pil_test():
        img = Image.open(make_jpeg()).convert("RGB")
        img = ImageOps.exif_transpose(img)
        return f"size={img.size}  mode={img.mode}"
    pil_ok2 = check("PIL open + exif_transpose", pil_test)

    if pil_ok2 and torch_ok and tv_ok:
        from torchvision import transforms
        T = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std =[0.229,0.224,0.225]),
        ])
        def transform_test():
            img = Image.open(make_jpeg()).convert("RGB")
            t   = T(img).unsqueeze(0)
            return f"shape={tuple(t.shape)}  dtype={t.dtype}"
        check("TRANSFORM → tensor", transform_test)

# ── 9. Full end-to-end inference ─────────────────────────
print("\n[9] FULL INFERENCE (end-to-end)")
if torch_ok and tv_ok and pil_ok and MODEL_PATH.exists():
    from PIL import Image, ImageOps
    from torchvision import transforms
    import io

    T2 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std =[0.229,0.224,0.225]),
    ])

    def e2e():
        m    = models.efficientnet_b0(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_f, 2),
        )
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            sd = (ckpt.get("model_state_dict")
                  or ckpt.get("state_dict")
                  or ckpt)
        else:
            sd = ckpt
        sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
        m.load_state_dict(sd, strict=True)
        m.eval()
        img = Image.new("RGB", (224, 224), color=(100, 100, 100))
        t   = T2(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(m(t), dim=1)
        real_p = probs[0, 0].item()
        fake_p = probs[0, 1].item()
        label  = "Fake" if fake_p >= real_p else "Real"
        return f"result={label}  real={real_p:.4f}  fake={fake_p:.4f}"
    check("end-to-end inference on synthetic image", e2e)

# ── 10. detector.py module import ────────────────────────
print("\n[10] detector.py MODULE")
check("import DeepfakeDetector",
      lambda: str(__import__("detector").DeepfakeDetector))

# ── 11. DeepfakeDetector() instantiation ─────────────────
print("\n[11] DeepfakeDetector() INSTANTIATION")
try:
    from detector import DeepfakeDetector
    def inst():
        d = DeepfakeDetector()
        return f"device={d.device}  model_type={type(d.model).__name__}"
    check("DeepfakeDetector()", inst)
except Exception as e:
    print(f"  {FAIL}  Could not even import: {e}")
    traceback.print_exc()

# ── Summary ───────────────────────────────────────────────
print("\n" + "="*60)
print("  SUMMARY")
print("="*60)
failed = [r for r in results if not r[1]]
if not failed:
    print(f"  {PASS}  All checks passed.")
    print("         The 500 is NOT from model/inference.")
    print("         Share your uvicorn terminal output for the next step.")
else:
    print(f"  {FAIL}  {len(failed)} check(s) failed:\n")
    for name, _, msg in failed:
        print(f"       • {name}")
        print(f"         → {msg}\n")
print("="*60 + "\n")