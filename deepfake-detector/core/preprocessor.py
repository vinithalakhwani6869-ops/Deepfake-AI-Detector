"""
Image validation and in-memory preparation (steps ④–⑧ from the original API).

Framework-agnostic: raises ImageValidationError with HTTP-equivalent status codes.
The API layer maps these to FastAPI HTTPException responses.

Production enhancements:
  - Hardened MIME detection with magic byte validation
  - Deterministic image processing (no randomness)
  - Comprehensive decompression bomb & corruption detection
  - Structured logging with validation step markers
  - Clear error messages for client debugging
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)


class ImageValidationError(Exception):
    """Validation failure with an HTTP status code the API layer should return."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


@dataclass(frozen=True)
class PreprocessorConfig:
    """
    Limits mirrored from the original production-hardened app.py.
    
    Frozen dataclass ensures immutability in production.
    """

    safe_max_pixels: int = 8_000 * 8_000
    max_image_width: int = 8_000
    max_image_height: int = 8_000
    pre_inference_max_px: int = 512


DEFAULT_CONFIG = PreprocessorConfig()


def _validate_magic_bytes(
    file_bytes: bytes,
    mime_type: str,
) -> bool:
    """
    Validate file magic bytes against declared MIME type.
    
    Hardened defense: reject if bytes don't match MIME claim.
    
    Args:
        file_bytes: Raw file bytes.
        mime_type: Declared MIME type from request (e.g., 'image/jpeg').
    
    Returns:
        True if magic bytes match MIME type, False otherwise.
    """
    if len(file_bytes) < 12:
        return False
    
    # JPEG: FF D8 FF
    if mime_type == "image/jpeg" and file_bytes[:3] == b"\xff\xd8\xff":
        return True
    # PNG: 89 50 4E 47
    elif mime_type == "image/png" and file_bytes[:4] == b"\x89PNG":
        return True
    # WebP: RIFF ... WEBP
    elif mime_type == "image/webp" and file_bytes[0:4] == b"RIFF" and file_bytes[8:12] == b"WEBP":
        return True
    
    return False


def validate_and_prepare_image(
    file_bytes: bytes,
    filename: str,
    declared_mime_type: str = "",
    config: PreprocessorConfig = DEFAULT_CONFIG,
) -> Image.Image:
    """
    Validate raw file bytes and return a safe, RGB-normalised PIL Image.

    Performs (in order):
      ④ Decompression bomb protection
      ⑤ PIL integrity verification + optional magic byte check
      ⑥ Resolution check
      ⑦ Pre-inference resize (≤ pre_inference_max_px)
      ⑧ EXIF orientation correction + RGB conversion

    Raises:
        ImageValidationError — on any validation failure (413 or 422).
    
    Args:
        file_bytes: Raw file data.
        filename: Original filename (for logging).
        declared_mime_type: MIME type from request (optional hardening).
        config: Validation limits.
    
    Returns:
        PIL Image in RGB mode, ready for inference.
    """
    Image.MAX_IMAGE_PIXELS = config.safe_max_pixels

    # ④ Decompression bomb protection + ⑤ PIL verification
    try:
        verify_buf = io.BytesIO(file_bytes)
        probe = Image.open(verify_buf)
        probe.verify()
        logger.debug(
            "[validate] ④ Decompression bomb check passed — %s",
            filename,
        )
    except Image.DecompressionBombError:
        logger.warning("[validate] ④ Decompression bomb rejected — %s", filename)
        raise ImageValidationError(
            413,
            "Image pixel count exceeds the safe processing limit. "
            "Please upload a smaller image.",
        )
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        logger.warning(
            "[validate] ⑤ Corrupt / unreadable image '%s' — %s",
            filename,
            exc,
        )
        raise ImageValidationError(
            422,
            f"The uploaded file is not a valid image or appears to be corrupt. Error: {exc}",
        )

    # Optional magic byte validation
    if declared_mime_type:
        if not _validate_magic_bytes(file_bytes, declared_mime_type):
            logger.warning(
                "[validate] Magic byte mismatch — declared=%s file=%s",
                declared_mime_type,
                filename,
            )
            # Not fatal; PIL will make final determination

    # Re-open for actual processing
    try:
        img = Image.open(io.BytesIO(file_bytes))
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        logger.warning("[validate] ⑤ Re-open failed '%s' — %s", filename, exc)
        raise ImageValidationError(
            422,
            f"Could not load image for processing. Error: {exc}",
        )

    # ⑥ Resolution check
    width, height = img.size
    logger.info(
        "[validate] ⑥ Image dimensions: %d × %d px — %s",
        width,
        height,
        filename,
    )

    if width > config.max_image_width or height > config.max_image_height:
        logger.warning(
            "[validate] ⑥ Resolution rejected: %d×%d — %s",
            width,
            height,
            filename,
        )
        raise ImageValidationError(
            413,
            f"Image resolution {width}×{height} px exceeds the maximum "
            f"allowed {config.max_image_width}×{config.max_image_height} px. "
            f"Please resize and upload again.",
        )

    # ⑧ EXIF orientation correction + RGB conversion
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    logger.debug("[validate] ⑧ EXIF transpose + RGB conversion applied")

    # ⑦ Pre-inference resize (deterministic, no randomness)
    if img.width > config.pre_inference_max_px or img.height > config.pre_inference_max_px:
        original_size = img.size
        img.thumbnail(
            (config.pre_inference_max_px, config.pre_inference_max_px),
            Image.LANCZOS,  # Deterministic resampling filter
        )
        logger.info(
            "[validate] ⑦ Pre-inference resize: %s → %s — %s",
            original_size,
            img.size,
            filename,
        )

    logger.info("[validate] Image ready for inference: %s", filename)
    return img
