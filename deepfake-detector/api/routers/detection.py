"""
POST /detect — deepfake image detection.

Validation pipeline order preserved from original app.py:
  ① MIME / extension  ② size  ③ empty  ④–⑧ image  ⑨ inference  ⑩ response

Production enhancements:
  - Hardened MIME validation with magic byte support
  - Upload size enforcement at multiple checkpoints
  - Structured exception handling with status code mapping
  - Deterministic temp file cleanup (finally block)
  - Async-compatible with proper resource management
  - Inference timeout placeholder for future implementation
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from api.dependencies import get_detector
from api.schemas.request import DEFAULT_UPLOAD_CONFIG, UploadValidationConfig
from core.detector import DeepfakeDetector
from core.preprocessor import ImageValidationError, validate_and_prepare_image

logger = logging.getLogger("deepfake_api")

router = APIRouter(tags=["detection"])

UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def _validate_mime_and_extension(
    content_type: str,
    suffix: str,
    config: UploadValidationConfig = DEFAULT_UPLOAD_CONFIG,
) -> None:
    """
    Step ① — MIME type and file extension validation.
    
    Raises HTTPException(415) on failure.
    
    Args:
        content_type: Content-Type header value (may be empty or mismatched).
        suffix: File extension from filename (lowercase).
        config: Validation configuration.
    
    Raises:
        HTTPException: 415 Unsupported Media Type if validation fails.
    """
    if content_type in ("", "application/octet-stream"):
        # No MIME info; rely on extension
        if suffix not in config.allowed_extensions:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Cannot determine file type from the Content-Type header. "
                    f"Extension '{suffix}' is not accepted. "
                    f"Please upload a .jpg, .jpeg, .png, or .webp file."
                ),
            )
    else:
        # Validate MIME type against whitelist
        if content_type not in config.allowed_mime_types:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file type '{content_type}'. "
                    f"Accepted types: image/jpeg, image/png, image/webp."
                ),
            )
        # Also validate extension against MIME
        if suffix not in config.allowed_extensions:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"File extension '{suffix}' does not match the MIME type. "
                    f"Use .jpg, .jpeg, .png, or .webp."
                ),
            )


@router.post("/detect")
async def detect(
    file: UploadFile = File(...),
    detector: DeepfakeDetector = Depends(get_detector),
) -> JSONResponse:
    """
    Deepfake detection endpoint.

    Returns JSON: { "result", "confidence", "filename" } — frontend contract unchanged.
    
    Validation pipeline:
      ① MIME/extension validation (415 on failure)
      ② File size check vs. max_file_bytes (413 on failure)
      ③ Empty file check (422 on failure)
      ④–⑧ Image validation & preparation (PIL, EXIF, resize) (413/422 on failure)
      ⑨ Model inference (500 on inference error)
      ⑩ JSON response with result, confidence, filename
    
    Args:
        file: Uploaded image file.
        detector: DeepfakeDetector singleton from dependency injection.
    
    Returns:
        JSON response: {"result": "Real|Fake", "confidence": float, "filename": str}
    
    Raises:
        HTTPException: Various status codes (413, 415, 422, 500) on failure.
    """
    logger.info(
        "[detect] New request: filename=%r content_type=%r",
        file.filename,
        file.content_type,
    )

    content_type = (file.content_type or "").lower().strip()
    suffix = Path(file.filename or "untitled").suffix.lower()

    # ① MIME/extension validation
    try:
        _validate_mime_and_extension(content_type, suffix)
    except HTTPException as exc:
        logger.warning(
            "[detect] ① MIME/ext rejected: content_type=%r suffix=%r detail=%s",
            content_type,
            suffix,
            exc.detail,
        )
        raise

    logger.info("[detect] ① MIME/ext OK: content_type=%r suffix=%r", content_type, suffix)

    # Read file bytes
    await file.seek(0)
    file_bytes = await file.read()

    file_size_mb = len(file_bytes) / 1_048_576
    logger.info("[detect] File size: %.2f MB", file_size_mb)

    # ② File size check (before empty check to give clear error priority)
    if len(file_bytes) > DEFAULT_UPLOAD_CONFIG.max_file_bytes:
        max_mb = DEFAULT_UPLOAD_CONFIG.max_file_bytes // 1_048_576
        logger.warning(
            "[detect] ② File too large: %.2f MB > %d MB limit",
            file_size_mb,
            max_mb,
        )
        raise HTTPException(
            status_code=413,
            detail=(
                f"File size {file_size_mb:.1f} MB exceeds the "
                f"{max_mb} MB limit. Please upload a smaller file."
            ),
        )

    # ③ Empty file check
    if len(file_bytes) == 0:
        logger.warning("[detect] ③ Empty file rejected: %r", file.filename)
        raise HTTPException(
            status_code=422,
            detail="Uploaded file is empty. Please upload a valid image.",
        )

    # ④–⑧ Image validation & preparation
    try:
        img: Image.Image = validate_and_prepare_image(
            file_bytes,
            file.filename or "upload",
            declared_mime_type=content_type,
        )
    except ImageValidationError as exc:
        logger.warning(
            "[detect] ④–⑧ Image validation failed: status=%d detail=%s",
            exc.status_code,
            exc.detail,
        )
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    logger.info(
        "[detect] ④–⑧ Image ready: %d×%d RGB",
        img.width,
        img.height,
    )

    # Create temp file for inference
    temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.png"

    try:
        img.save(temp_path, format="PNG")
        logger.debug("[detect] Temp file saved: %s", temp_path)
    except Exception as exc:
        logger.error("[detect] Temp file write failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Server failed to prepare image for analysis: {exc}",
        )

    result = ""
    confidence = 0.0

    # ⑨ Inference
    try:
        result, confidence = detector.predict(str(temp_path))
        # TODO: Implement inference timeout (e.g., with asyncio.timeout in Python 3.11+)
        logger.info(
            "[detect] ⑨ Inference complete: result=%s confidence=%.2f%%",
            result,
            confidence,
        )
    except ValueError as exc:
        # Preprocessing error (e.g., corrupt temp file)
        logger.warning("[detect] ⑨ Inference ValueError: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        # Model or GPU error
        logger.error("[detect] ⑨ Inference RuntimeError: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Model inference failed. Check server logs for details.",
        )
    except Exception as exc:
        # Unexpected errors
        logger.error(
            "[detect] ⑨ Unexpected error (%s): %s",
            type(exc).__name__,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during analysis. Please try again.",
        )
    finally:
        # Deterministic cleanup (always runs)
        if temp_path.exists():
            try:
                temp_path.unlink()
                logger.debug("[detect] Temp file cleaned up: %s", temp_path)
            except OSError as exc:
                logger.warning(
                    "[detect] Could not delete temp file %s: %s",
                    temp_path,
                    exc,
                )

    # ⑩ Response
    return JSONResponse(
        content={
            "result": result,
            "confidence": round(confidence, 2),
            "filename": file.filename,
        }
    )
