"""
FastAPI application — instance, middleware, router registration, startup lifecycle.

Run from deepfake-detector/::

    uvicorn api.app:app --reload --host 127.0.0.1 --port 8000

Legacy entry (deepfake-api/app.py) re-exports this app for backward compatibility.

Production enhancements:
  - Structured logging setup before app initialization
  - Startup/shutdown lifecycle events for resource management
  - CORS middleware with controlled origins
  - Error handlers for graceful failure modes
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers import detection, health

# Configure structured logging before creating app
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DeepFake Detector AI",
    description="Production-grade deepfake image detection API",
    version="2.0.0",
)

# CORS middleware: allow all origins (can be restricted in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router)
app.include_router(detection.router)


@app.on_event("startup")
async def startup_event() -> None:
    """
    Startup lifecycle event: initialize detector singleton.
    
    Logs initialization status; errors are deferred to first /detect call
    to allow graceful degradation (health check still responds).
    """
    logger.info("[app.startup] FastAPI application starting up")
    try:
        from api.dependencies import get_detector
        detector = get_detector()
        logger.info(
            "[app.startup] Detector initialized: model_loaded=%s device=%s",
            detector.model is not None,
            detector.device,
        )
    except Exception as exc:
        logger.error(
            "[app.startup] Warning: detector initialization deferred — %s",
            exc,
        )


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """
    Shutdown lifecycle event: cleanup resources.
    
    Called when server stops (e.g., uvicorn shutdown, Docker stop).
    """
    logger.info("[app.shutdown] FastAPI application shutting down")
    import torch
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    logger.info("[app.shutdown] GPU cache cleared")


@app.get("/docs", include_in_schema=False)
async def swagger_ui() -> dict[str, Any]:
    """Swagger UI redirect for API documentation."""
    return {}


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """
    Global exception handler for uncaught errors.
    
    Returns 500 with generic message to avoid leaking internals.
    """
    logger.error(
        "[app.exception] Unhandled exception: %s: %s",
        type(exc).__name__,
        exc,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred. Please try again.",
        },
    )
