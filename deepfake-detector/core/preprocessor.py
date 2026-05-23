"""
Image validation and in-memory preparation (steps ④–⑧ from the original API).

Framework-agnostic: raises ImageValidationError with HTTP-equivalent status codes.
The API layer maps these to FastAPI HTTPException responses.
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
    """Limits mirrored from the original production-hardened app.py."""

    safe_max_pixels: int = 8_000 * 8_000
    max_image_width: int = 8_000
    max_image_height: int = 8_000
    pre_inference_max_px: int = 512


DEFAULT_CONFIG = PreprocessorConfig()


def validate_and_prepare_image(
    file_bytes: bytes,
    filename: str,
    config: PreprocessorConfig = DEFAULT_CONFIG,
) -> Image.Image:
    """
    Validate raw file bytes and return a safe, RGB-normalised PIL Image.

    Performs (in order):
      ④ Decompression bomb protection
      ⑤ PIL integrity verification
      ⑥ Resolution check
      ⑧ EXIF orientation correction + RGB conversion
      ⑦ Pre-inference resize (≤ pre_inference_max_px)

    Raises:
        ImageValidationError — on any validation failure (413 or 422).
    """
    Image.MAX_IMAGE_PIXELS = config.safe_max_pixels

    try:
        verify_buf = io.BytesIO(file_bytes)
        probe = Image.open(verify_buf)
        probe.verify()
    except Image.DecompressionBombError:
        logger.warning("[validate] ④ Decompression bomb rejected: %s", filename)
        raise ImageValidationError(
            413,
            "Image pixel count exceeds the safe processing limit. "
            "Please upload a smaller image.",
        )
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        logger.warning("[validate] ⑤ Corrupt / unreadable image '%s': %s", filename, exc)
        raise ImageValidationError(
            422,
            f"The uploaded file is not a valid image or appears to be corrupt. Error: {exc}",
        )

    try:
        img = Image.open(io.BytesIO(file_bytes))
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        logger.warning("[validate] ⑤ Re-open failed '%s': %s", filename, exc)
        raise ImageValidationError(
            422,
            f"Could not load image for processing. Error: {exc}",
        )

    width, height = img.size
    logger.info("[validate] ⑥ Dimensions: %d × %d px — %s", width, height, filename)

    if width > config.max_image_width or height > config.max_image_height:
        logger.warning(
            "[validate] ⑥ Resolution rejected: %d×%d — %s", width, height, filename
        )
        raise ImageValidationError(
            413,
            f"Image resolution {width}×{height} px exceeds the maximum "
            f"allowed {config.max_image_width}×{config.max_image_height} px. "
            f"Please resize and upload again.",
        )

    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    if img.width > config.pre_inference_max_px or img.height > config.pre_inference_max_px:
        original_size = img.size
        img.thumbnail(
            (config.pre_inference_max_px, config.pre_inference_max_px),
            Image.LANCZOS,
        )
        logger.info(
            "[validate] ⑦ Pre-resize: %s → %s — %s",
            original_size,
            img.size,
            filename,
        )

    return img
