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
  - Environment-aware configuration loading
  - Startup validation for production safety
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.logging_config import setup_logging
from api.routers import detection, health
from api.startup_validation import safe_shutdown, validate_environment

# ════════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP (before app creation)
# ════════════════════════════════════════════════════════════════════════════════

log_level = os.getenv("LOG_LEVEL", "INFO")
log_format = os.getenv("LOG_FORMAT", "text")
log_to_file = os.getenv("LOG_TO_FILE", "true").lower() == "true"
log_file_max_mb = int(os.getenv("LOG_FILE_MAX_SIZE_MB", "100"))

setup_logging(
    log_level=log_level,
    log_format=log_format,
    log_to_file=log_to_file,
    log_file_max_mb=log_file_max_mb,
)

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# FASTAPI APP INITIALIZATION
# ════════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="DeepFake Detector AI",
    description="Production-grade deepfake image detection API",
    version="2.0.0",
)

# ════════════════════════════════════════════════════════════════════════════════
# MIDDLEWARE CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

# CORS middleware: configurable origins (can be restricted in production)
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
if cors_origins == ["*"]:
    cors_origins = ["*"]
else:
    cors_origins = [origin.strip() for origin in cors_origins if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ════════════════════════════════════════════════════════════════════════════════
# ROUTER REGISTRATION
# ════════════════════════════════════════════════════════════════════════════════

app.include_router(health.router)
app.include_router(detection.router)

# ════════════════════════════════════════════════════════════════════════════════
# LIFECYCLE EVENTS
# ════════════════════════════════════════════════════════════════════════════════


@app.on_event("startup")
async def startup_event() -> None:
    """
    Startup lifecycle event: initialize detector singleton and validate environment.
    
    Logs initialization status; errors are deferred to first /detect call
    to allow graceful degradation (health check still responds).
    """
    logger.info("[app.startup] FastAPI application starting up")
    logger.info("[app.startup] Environment: %s", os.getenv("ENVIRONMENT", "development"))
    
    # Validate startup environment
    try:
        weights_file = os.getenv("WEIGHTS_FILE", "checkpoints/deepfake_model.pth")
        weights_path = Path(weights_file)
        
        validation_results = validate_environment(
            weights_path=weights_path,
            log_dir=Path("logs"),
            temp_dir=Path("temp_uploads"),
            min_disk_space_mb=100,
        )
        logger.info("[app.startup] Environment validation: %s", validation_results)
    except Exception as exc:
        logger.warning(
            "[app.startup] Environment validation failed (non-critical): %s",
            exc,
        )
    
    # Initialize detector singleton
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
    
    logger.info("[app.startup] Startup complete")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """
    Shutdown lifecycle event: cleanup resources and gracefully shutdown.
    
    Called when server stops (e.g., uvicorn shutdown, Docker stop).
    """
    logger.info("[app.shutdown] FastAPI application shutting down")
    safe_shutdown()
    logger.info("[app.shutdown] Shutdown complete")

# ════════════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ════════════════════════════════════════════════════════════════════════════════


@app.get("/docs", include_in_schema=False)
async def swagger_ui() -> dict[str, Any]:
    """Swagger UI redirect for API documentation."""
    return {}


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
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
