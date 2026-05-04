import os
import uuid
import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from detector import DeepfakeDetector

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DeepFake Detector AI",
    description="Forensic-grade deepfake image detection API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten this to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ──────────────────────────────────────────────────────────────────
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
ALLOWED_EXTENSIONS    = {".jpg", ".jpeg", ".png"}
UPLOAD_DIR            = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Load detector once at startup ──────────────────────────────────────────────
detector = DeepfakeDetector()


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "DeepFake Detector API is running."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": detector.model is not None,
        "device": str(detector.device),
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Accepts a JPG or PNG image and returns a deepfake prediction.

    Returns:
        {
            "result":     "Real" | "Fake",
            "confidence": float  (0–100, percentage),
            "filename":   str
        }
    """

    # ── 1. Validate content type ───────────────────────────────────────────────
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Accepted: JPEG, PNG.",
        )

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported extension '{suffix}'. Accepted: .jpg, .jpeg, .png",
        )

    # ── 2. Save file to a temp path ────────────────────────────────────────────
    temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {exc}",
        )

    # ── 3. Run inference ───────────────────────────────────────────────────────
    try:
        result, confidence = detector.predict(str(temp_path))
    except ValueError as exc:
        # Corrupt or unreadable image
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        # Model-level failure
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        # Always clean up the temp file
        if temp_path.exists():
            temp_path.unlink()

    return JSONResponse(
        content={
            "result":     result,
            "confidence": round(confidence, 2),
            "filename":   file.filename,
        }
    )