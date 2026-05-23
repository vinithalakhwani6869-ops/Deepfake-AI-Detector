"""Unit tests for core.preprocessor."""

import io

import pytest
from PIL import Image

from core.preprocessor import ImageValidationError, validate_and_prepare_image


def _jpeg_bytes(size=(64, 64)):
    img = Image.new("RGB", size, color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_valid_jpeg_returns_rgb():
    img = validate_and_prepare_image(_jpeg_bytes(), "test.jpg")
    assert img.mode == "RGB"
    assert img.size == (64, 64)


def test_empty_bytes_raises():
    with pytest.raises(ImageValidationError) as exc:
        validate_and_prepare_image(b"", "empty.jpg")
    assert exc.value.status_code == 422
