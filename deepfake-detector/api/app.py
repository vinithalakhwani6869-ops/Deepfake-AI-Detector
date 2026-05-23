"""
FastAPI application — instance, middleware, router registration.

Run from deepfake-detector/:
    uvicorn api.app:app --reload --host 127.0.0.1 --port 8000

Legacy entry (deepfake-api/app.py) re-exports this app for backward compatibility.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import detection, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(
    title="DeepFake Detector AI",
    description="Production-grade deepfake image detection API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(detection.router)
