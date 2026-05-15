import uuid
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ──────────────────────────────────────────────────────────────────
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "application/octet-stream",  # fallback — some browsers/mobile send this
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE_MB   = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

UPLOAD_DIR = Path("temp_uploads")
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
        "status":       "ok",
        "model_loaded": detector.model is not None,
        "device":       str(detector.device),
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Accepts a JPG, PNG, or WebP image and returns a deepfake prediction.

    Returns:
        {
            "result":     "Real" | "Fake",
            "confidence": float  (0–100),
            "filename":   str
        }
    """

    print(f"[detect] filename={file.filename!r}  content_type={file.content_type!r}")

    # ── 1. Normalise and validate content type ─────────────────────────────────
    # file.content_type can be None when no Content-Type header is sent.
    # Some mobile browsers and curl send "application/octet-stream".
    # In both cases we fall back to trusting the file extension.
    content_type = (file.content_type or "").lower().strip()
    suffix       = Path(file.filename or "").suffix.lower()

    if content_type in ("", "application/octet-stream"):
        # No usable MIME — trust extension only
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Cannot determine file type from Content-Type header. "
                    f"Extension '{suffix}' is not accepted. "
                    f"Please upload a .jpg, .jpeg, .png, or .webp file."
                ),
            )
    else:
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file type '{content_type}'. "
                    f"Accepted types: JPEG, PNG, WebP."
                ),
            )
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file extension '{suffix}'. "
                    f"Accepted: .jpg, .jpeg, .png, .webp"
                ),
            )

    # ── 2. Read file bytes safely ──────────────────────────────────────────────
    # Always seek(0) first — FastAPI's SpooledTemporaryFile pointer position
    # is not guaranteed to be at the start after multipart parsing.
    await file.seek(0)
    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB} MB.",
        )

    # ── 3. Write to temp file ──────────────────────────────────────────────────
    temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    try:
        temp_path.write_bytes(file_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {exc}",
        )

    # ── 4. Run inference ───────────────────────────────────────────────────────
    try:
        result, confidence = detector.predict(str(temp_path))

    except ValueError as exc:
        # Corrupt or unreadable image — client error
        raise HTTPException(status_code=422, detail=str(exc))

    except RuntimeError as exc:
        # Model or inference failure — server error
        print(f"[detect] RUNTIME ERROR: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    except Exception as exc:
        print(f"[detect] UNEXPECTED ERROR: {exc}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")

    finally:
        # Always clean up — even if inference raised
        if temp_path.exists():
            temp_path.unlink()

    print(f"[detect] result={result}  confidence={confidence:.2f}%")

    return JSONResponse(
        content={
            "result":     result,
            "confidence": round(confidence, 2),
            "filename":   file.filename,
        }
    )