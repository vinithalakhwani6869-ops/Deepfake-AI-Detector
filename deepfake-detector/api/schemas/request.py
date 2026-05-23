"""
Upload request validation constants (MIME, size, extensions).

Used by the detection router — mirrors original app.py validation (steps ①–③).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UploadValidationConfig:
    allowed_mime_types: frozenset[str] = frozenset({
        "image/jpeg",
        "image/png",
        "image/webp",
    })
    allowed_extensions: frozenset[str] = frozenset({
        ".jpg", ".jpeg", ".png", ".webp",
    })
    max_file_bytes: int = 50 * 1024 * 1024


DEFAULT_UPLOAD_CONFIG = UploadValidationConfig()
