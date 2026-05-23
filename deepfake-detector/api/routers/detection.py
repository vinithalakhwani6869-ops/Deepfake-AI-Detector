"""
POST /detect — deepfake image detection.

Validation pipeline order preserved from original app.py:
  ① MIME / extension  ② size  ③ empty  ④–⑧ image  ⑨ inference  ⑩ response
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
    """Step ① — raises HTTP 415 on failure."""
    if content_type in ("", "application/octet-stream"):
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
        if content_type not in config.allowed_mime_types:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file type '{content_type}'. "
                    f"Accepted types: image/jpeg, image/png, image/webp."
                ),
            )
        if suffix not in config.allowed_extensions:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"File extension '{suffix}' is not accepted. "
                    f"Use .jpg, .jpeg, .png, or .webp."
                ),
            )


@router.post("/detect")
async def detect(
    file: UploadFile = File(...),
    detector: DeepfakeDetector = Depends(get_detector),
):
    """
    Deepfake detection endpoint.

    Returns JSON: { "result", "confidence", "filename" } — frontend contract unchanged.
    """
    logger.info(
        "[detect] filename=%r content_type=%r",
        file.filename,
        file.content_type,
    )

    content_type = (file.content_type or "").lower().strip()
    suffix = Path(file.filename or "untitled").suffix.lower()

    try:
        _validate_mime_and_extension(content_type, suffix)
    except HTTPException:
        logger.warning("[detect] ① Rejected: %r / %r", content_type, suffix)
        raise

    logger.info("[detect] ① MIME/ext OK: %r / %r", content_type, suffix)

    await file.seek(0)
    file_bytes = await file.read()

    file_size_mb = len(file_bytes) / 1_048_576
    logger.info("[detect] ② File size: %.2f MB — %r", file_size_mb, file.filename)

    if len(file_bytes) == 0:
        logger.warning("[detect] ③ Empty file rejected: %r", file.filename)
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    if len(file_bytes) > DEFAULT_UPLOAD_CONFIG.max_file_bytes:
        logger.warning(
            "[detect] ② File too large: %.2f MB — %r", file_size_mb, file.filename
        )
        raise HTTPException(
            status_code=413,
            detail=(
                f"File size {file_size_mb:.1f} MB exceeds the "
                f"{DEFAULT_UPLOAD_CONFIG.max_file_bytes // 1_048_576} MB limit."
            ),
        )

    try:
        img: Image.Image = validate_and_prepare_image(
            file_bytes, file.filename or "upload"
        )
    except ImageValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    logger.info(
        "[detect] Image ready: %d×%d RGB — %r",
        img.width,
        img.height,
        file.filename,
    )

    temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.png"

    try:
        img.save(temp_path, format="PNG")
    except Exception as exc:
        logger.error("[detect] Temp file write failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Server failed to prepare image for analysis: {exc}",
        )

    result = ""
    confidence = 0.0

    try:
        result, confidence = detector.predict(str(temp_path))
        logger.info(
            "[detect] Inference: result=%s confidence=%.2f%% file=%r",
            result,
            confidence,
            file.filename,
        )
    except ValueError as exc:
        logger.warning("[detect] Inference ValueError: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        logger.error("[detect] Inference RuntimeError: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Model inference failed. Check server logs for details.",
        )
    except Exception as exc:
        logger.error(
            "[detect] Unexpected inference error (%s): %s",
            type(exc).__name__,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during analysis. Please try again.",
        )
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError as exc:
                logger.warning(
                    "[detect] Could not delete temp file %s: %s", temp_path, exc
                )

    return JSONResponse(
        content={
            "result": result,
            "confidence": round(confidence, 2),
            "filename": file.filename,
        }
    )
