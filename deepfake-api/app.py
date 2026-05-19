"""
app.py — DeepFake Detector AI
Production-hardened FastAPI backend

Validation pipeline (executed in this exact order on every request):
  ① MIME type check          → HTTP 415  (invalid content-type or extension)
  ② File size check          → HTTP 413  (over 50 MB)
  ③ Empty file check         → HTTP 422  (zero bytes)
  ④ Decompression bomb cap   → HTTP 413  (pixel count > 64 MP)
  ⑤ PIL integrity verify     → HTTP 422  (corrupt / truncated file)
  ⑥ Resolution check         → HTTP 413  (width or height > 8 000 px)
  ⑦ Pre-inference resize     → in-memory (image scaled to 512×512 before save)
  ⑧ EXIF + RGB normalise     → in-memory
  ⑨ Inference error wrap     → HTTP 500  (clean JSON, no traceback leak)
  ⑩ Structured logging       → terminal  (every step logged)

Frontend response contract (unchanged):
  { "result": "Real"|"Fake", "confidence": float, "filename": str }
"""

import io
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from detector import DeepfakeDetector

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════
# Single consistent format for every log line.
# asctime gives human-readable timestamp; levelname is padded for alignment.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("deepfake_api")


# ══════════════════════════════════════════════════════════════════════════════
# APP + CORS
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="DeepFake Detector AI",
    description="Production-grade deepfake image detection API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# ① Accepted MIME types
ALLOWED_MIME_TYPES: frozenset[str] = frozenset({
    "image/jpeg",
    "image/png",
    "image/webp",
})

# ① Accepted file extensions (used when MIME is missing or generic)
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".webp",
})

# ② Maximum file size — 50 MB
MAX_FILE_BYTES: int = 50 * 1024 * 1024

# ④ Decompression bomb pixel cap — 64 MP (8 000 × 8 000)
#    PIL default is 178 MP; we lower it to limit RAM exposure.
#    Setting to None would disable the check — never do that in production.
SAFE_MAX_PIXELS: int = 8_000 * 8_000

# ⑥ Maximum image dimensions in pixels
MAX_IMAGE_WIDTH:  int = 8_000
MAX_IMAGE_HEIGHT: int = 8_000

# ⑦ Pre-inference resize target.
#    Images larger than this are resized BEFORE being written to disk and
#    passed to the model. This prevents huge tensors and RAM spikes.
#    The model's own TRANSFORM will resize again to 224×224 — this is an
#    upstream safety cap, not a replacement for the model's preprocessing.
PRE_INFERENCE_MAX_PX: int = 512

# Temp directory for inference files
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL — loaded once at startup, reused for every request
# ══════════════════════════════════════════════════════════════════════════════
detector = DeepfakeDetector()


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE VALIDATION HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _validate_and_prepare_image(file_bytes: bytes, filename: str) -> Image.Image:
    """
    Validate raw file bytes and return a safe, RGB-normalised PIL Image.

    Performs (in order):
      ④ Decompression bomb protection  — caps Image.MAX_IMAGE_PIXELS
      ⑤ PIL integrity verification     — verify() rejects corrupt / truncated files
      ⑥ Resolution check               — rejects images wider/taller than 8 000 px
      ⑦ Pre-inference resize           — downscales to PRE_INFERENCE_MAX_PX if needed
      ⑧ EXIF orientation correction    — exif_transpose() fixes phone camera rotation
      ⑧ RGB conversion                 — normalises RGBA, palette, greyscale, etc.

    Args:
        file_bytes: Raw bytes from the uploaded file. Read only once in the caller.
        filename:   Original filename, used only for log messages.

    Returns:
        PIL Image in RGB mode, safe to save and pass to the model.

    Raises:
        HTTPException 413 — pixel count or dimensions exceed limits
        HTTPException 422 — corrupt, truncated, or unreadable image
    """
    # ── ④ Apply decompression bomb cap globally BEFORE any Image.open() ──────
    # PIL raises DecompressionBombError automatically when the pixel count
    # of a freshly-opened image exceeds MAX_IMAGE_PIXELS.
    Image.MAX_IMAGE_PIXELS = SAFE_MAX_PIXELS

    # ── ⑤ Open image (first pass — for verify()) ─────────────────────────────
    # verify() inspects the raw bytes for corruption / truncation but
    # leaves the stream in an unusable state afterwards, so we open twice:
    #   pass 1 → verify()
    #   pass 2 → actual use
    # Both passes use fresh BytesIO instances constructed from the same bytes
    # variable — the original bytes are NOT re-read from the network.
    try:
        verify_buf = io.BytesIO(file_bytes)
        probe      = Image.open(verify_buf)
        probe.verify()                          # raises on corrupt / truncated
    except Image.DecompressionBombError:
        logger.warning("[validate] ④ Decompression bomb rejected: %s", filename)
        raise HTTPException(
            status_code=413,
            detail=(
                "Image pixel count exceeds the safe processing limit. "
                "Please upload a smaller image."
            ),
        )
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        logger.warning("[validate] ⑤ Corrupt / unreadable image '%s': %s", filename, exc)
        raise HTTPException(
            status_code=422,
            detail=(
                f"The uploaded file is not a valid image or appears to be corrupt. "
                f"Error: {exc}"
            ),
        )

    # ── ⑤ Re-open for actual use (verify() leaves the image unusable) ─────────
    try:
        img = Image.open(io.BytesIO(file_bytes))
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        logger.warning("[validate] ⑤ Re-open failed '%s': %s", filename, exc)
        raise HTTPException(
            status_code=422,
            detail=f"Could not load image for processing. Error: {exc}",
        )

    # ── ⑥ Resolution check ────────────────────────────────────────────────────
    width, height = img.size
    logger.info(
        "[validate] ⑥ Dimensions: %d × %d px — %s", width, height, filename
    )

    if width > MAX_IMAGE_WIDTH or height > MAX_IMAGE_HEIGHT:
        logger.warning(
            "[validate] ⑥ Resolution rejected: %d×%d — %s", width, height, filename
        )
        raise HTTPException(
            status_code=413,
            detail=(
                f"Image resolution {width}×{height} px exceeds the maximum "
                f"allowed {MAX_IMAGE_WIDTH}×{MAX_IMAGE_HEIGHT} px. "
                f"Please resize and upload again."
            ),
        )

    # ── ⑧ EXIF orientation correction ────────────────────────────────────────
    # Phone cameras store rotation in EXIF metadata but do NOT rotate pixels.
    # exif_transpose() physically rotates the pixel data to match the metadata
    # so the model always receives an upright image.
    img = ImageOps.exif_transpose(img)

    # ── ⑧ RGB normalisation ───────────────────────────────────────────────────
    # Handles RGBA (transparency), palette (P), greyscale (L), CMYK, etc.
    # The model expects 3-channel RGB tensors.
    img = img.convert("RGB")

    # ── ⑦ Pre-inference resize ───────────────────────────────────────────────
    # If either dimension exceeds PRE_INFERENCE_MAX_PX, downscale with
    # LANCZOS (high-quality anti-aliased downsampling).
    # thumbnail() modifies in-place and preserves aspect ratio.
    # This caps the size of the temp file and prevents RAM spikes from
    # large tensors during model preprocessing.
    if img.width > PRE_INFERENCE_MAX_PX or img.height > PRE_INFERENCE_MAX_PX:
        original_size = img.size
        img.thumbnail((PRE_INFERENCE_MAX_PX, PRE_INFERENCE_MAX_PX), Image.LANCZOS)
        logger.info(
            "[validate] ⑦ Pre-resize: %s → %s — %s",
            original_size, img.size, filename,
        )

    return img


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    """Health ping — confirms the server is reachable."""
    return {"status": "ok", "message": "DeepFake Detector API is running."}


@app.get("/health")
def health():
    """Detailed health check — reports model load status and compute device."""
    return {
        "status":       "ok",
        "model_loaded": detector.model is not None,
        "device":       str(detector.device),
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Deepfake detection endpoint.

    Accepts: JPEG, PNG, or WebP image (≤ 50 MB, ≤ 8 000 × 8 000 px)
    Returns:
        {
            "result":     "Real" | "Fake",
            "confidence": float,   # 0.00 – 100.00
            "filename":   str
        }
    """

    # ── Log incoming request ──────────────────────────────────────────────────
    logger.info(
        "[detect] ▶ filename=%r  content_type=%r",
        file.filename,
        file.content_type,
    )

    # ── ① MIME type + extension validation ───────────────────────────────────
    # file.content_type can be:
    #   • None                      → header absent; trust extension
    #   • "application/octet-stream"→ generic binary; trust extension
    #   • a real MIME string        → validate directly
    content_type: str = (file.content_type or "").lower().strip()
    suffix: str       = Path(file.filename or "untitled").suffix.lower()

    if content_type in ("", "application/octet-stream"):
        # No usable MIME — fall back to extension check
        if suffix not in ALLOWED_EXTENSIONS:
            logger.warning(
                "[detect] ① Rejected (no MIME, bad ext): %r — %r",
                content_type, suffix,
            )
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Cannot determine file type from the Content-Type header. "
                    f"Extension '{suffix}' is not accepted. "
                    f"Please upload a .jpg, .jpeg, .png, or .webp file."
                ),
            )
    else:
        # Real MIME present — validate it
        if content_type not in ALLOWED_MIME_TYPES:
            logger.warning("[detect] ① Rejected (MIME): %r", content_type)
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file type '{content_type}'. "
                    f"Accepted types: image/jpeg, image/png, image/webp."
                ),
            )
        # Also validate extension to catch mismatched MIME/ext pairs
        if suffix not in ALLOWED_EXTENSIONS:
            logger.warning("[detect] ① Rejected (ext mismatch): %r", suffix)
            raise HTTPException(
                status_code=415,
                detail=(
                    f"File extension '{suffix}' is not accepted. "
                    f"Use .jpg, .jpeg, .png, or .webp."
                ),
            )

    logger.info("[detect] ① MIME/ext OK: %r / %r", content_type, suffix)

    # ── ② + ③ Read bytes ONCE ─────────────────────────────────────────────────
    # seek(0) guarantees the pointer is at the start.
    # FastAPI's SpooledTemporaryFile position is not guaranteed after multipart
    # parsing, so we always seek before reading.
    # file_bytes is reused for all subsequent steps — no second await file.read().
    await file.seek(0)
    file_bytes: bytes = await file.read()

    file_size_mb: float = len(file_bytes) / 1_048_576
    logger.info("[detect] ② File size: %.2f MB — %r", file_size_mb, file.filename)

    if len(file_bytes) == 0:
        logger.warning("[detect] ③ Empty file rejected: %r", file.filename)
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    if len(file_bytes) > MAX_FILE_BYTES:
        logger.warning(
            "[detect] ② File too large: %.2f MB — %r", file_size_mb, file.filename
        )
        raise HTTPException(
            status_code=413,
            detail=(
                f"File size {file_size_mb:.1f} MB exceeds the {MAX_FILE_BYTES // 1_048_576} MB limit."
            ),
        )

    # ── ④ ⑤ ⑥ ⑦ ⑧ Validate and prepare image ─────────────────────────────────
    # All image-level checks happen inside this helper.
    # It raises HTTPException on any failure, so we never reach inference
    # with an invalid image.
    img: Image.Image = _validate_and_prepare_image(
        file_bytes, file.filename or "upload"
    )

    logger.info(
        "[detect] ✓ Image ready: %d×%d RGB — %r",
        img.width, img.height, file.filename,
    )

    # ── Write validated image to temp file ───────────────────────────────────
    # detector.predict() operates on a file path (preserving the existing
    # contract). We save as PNG — a lossless format that avoids any JPEG
    # re-compression artefacts and is always readable by PIL.
    temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.png"

    try:
        img.save(temp_path, format="PNG")
    except Exception as exc:
        logger.error("[detect] Temp file write failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Server failed to prepare image for analysis: {exc}",
        )

    # ── ⑨ Model inference ────────────────────────────────────────────────────
    result: str    = ""
    confidence: float = 0.0

    try:
        result, confidence = detector.predict(str(temp_path))
        logger.info(
            "[detect] ✓ Inference: result=%s  confidence=%.2f%%  file=%r",
            result, confidence, file.filename,
        )

    except ValueError as exc:
        # Raised by _preprocess() — image was valid enough to pass validation
        # but failed during tensor conversion (edge case).
        logger.warning("[detect] ⑨ Inference ValueError: %s", exc)
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    except RuntimeError as exc:
        # Model failure (not loaded, CUDA OOM, tensor shape mismatch, etc.)
        # Log the full error server-side; return a clean message to the client.
        logger.error("[detect] ⑨ Inference RuntimeError: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Model inference failed. Check server logs for details.",
        )

    except Exception as exc:
        # Catch-all — prevents any unhandled exception from leaking a traceback
        # to the client.
        logger.error("[detect] ⑨ Unexpected inference error (%s): %s",
                     type(exc).__name__, exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during analysis. Please try again.",
        )

    finally:
        # Always delete the temp file — even if inference raised.
        # We use a separate try so a deletion failure never masks the real error.
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError as exc:
                logger.warning(
                    "[detect] Could not delete temp file %s: %s", temp_path, exc
                )

    # ── ⑩ Return result — exact format the frontend expects ──────────────────
    return JSONResponse(
        content={
            "result":     result,
            "confidence": round(confidence, 2),
            "filename":   file.filename,   # unchanged original filename
        }
    )