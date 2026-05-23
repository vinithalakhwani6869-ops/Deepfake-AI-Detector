"""Integration tests for POST /detect."""

import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from api.app import app

client = TestClient(app)


def _make_upload_file():
    img = Image.new("RGB", (128, 128), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return ("test.jpg", buf, "image/jpeg")


@pytest.mark.slow
def test_detect_returns_contract():
    name, data, mime = _make_upload_file()
    r = client.post("/detect", files={"file": (name, data, mime)})
    assert r.status_code == 200
    body = r.json()
    assert body["result"] in ("Real", "Fake")
    assert "confidence" in body
    assert body["filename"] == "test.jpg"
