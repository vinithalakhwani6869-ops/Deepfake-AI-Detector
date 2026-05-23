"""Integration tests for health endpoints."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from api.app import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "model_loaded" in data
    assert "device" in data
