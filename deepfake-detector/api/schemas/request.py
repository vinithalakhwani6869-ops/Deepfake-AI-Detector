"""
Upload request validation constants (MIME, size, extensions).

Used by the detection router — mirrors original app.py validation (steps ①–③).

Production enhancements:
  - Immutable frozen dataclass
  - Clear inline documentation
  - Reasonable production defaults
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UploadValidationConfig:
    """
    Upload validation configuration (immutable).
    
    Attributes:
        allowed_mime_types: MIME types accepted (image/jpeg, image/png, image/webp).
        allowed_extensions: File extensions accepted (.jpg, .jpeg, .png, .webp).
        max_file_bytes: Maximum file size (50 MB).
    """

    allowed_mime_types: frozenset[str] = frozenset({
        "image/jpeg",
        "image/png",
        "image/webp",
    })
    allowed_extensions: frozenset[str] = frozenset({
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    })
    max_file_bytes: int = 50 * 1024 * 1024  # 50 MB


DEFAULT_UPLOAD_CONFIG = UploadValidationConfig()
