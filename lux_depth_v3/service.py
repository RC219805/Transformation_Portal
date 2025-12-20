"""FastAPI service for Depth Anything 3 inference.

Provides REST API for depth estimation with security hardening.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List
import io
import os
import re
import time

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image

from lux_depth_v3.config import (
    DA3Config,
    ModelVariant,
    Preset,
    ExportFormat,
)
from lux_depth_v3.input_manager import ImageInput
from lux_depth_v3.inference import DA3InferenceEngine
from lux_depth_v3.postprocessing import Postprocessor
from lux_depth_v3.export import Exporter


# Security configuration
MAX_FILE_SIZE_MB = 50
MAX_IMAGE_DIMENSION = 4096
RATE_LIMIT_REQUESTS_PER_MINUTE = 60

# Request models


class DepthRequest(BaseModel):
    """Depth estimation request parameters."""

    model_variant: ModelVariant = ModelVariant.METRIC_LARGE
    preset: Optional[Preset] = None
    metric_scaling: bool = False
    export_formats: List[ExportFormat] = [ExportFormat.PNG]


class DepthResponse(BaseModel):
    """Depth estimation response."""

    depth_range: tuple[float, float] = Field(..., description="Min and max depth values")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_variant: str = Field(..., description="Model variant used")
    export_paths: dict[str, str] = Field(..., description="Exported file paths")


# Create FastAPI app
app = FastAPI(
    title="Lux Depth V3 Service",
    description="Depth Anything 3 depth estimation service with security hardening",
    version="0.1.0",
)

# CORS middleware (configure as needed for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
inference_engine: Optional[DA3InferenceEngine] = None
output_dir: Path = Path("service_output")


# Rate limiting
request_timestamps: List[float] = []


def check_rate_limit(client_ip: str) -> bool:
    """Check if request is within rate limit.

    Args:
        client_ip: Client IP address

    Returns:
        True if within limit, False otherwise
    """
    global request_timestamps

    current_time = time.time()

    # Remove timestamps older than 1 minute
    request_timestamps = [
        ts for ts in request_timestamps
        if current_time - ts < 60
    ]

    # Check limit
    if len(request_timestamps) >= RATE_LIMIT_REQUESTS_PER_MINUTE:
        return False

    request_timestamps.append(current_time)
    return True


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    global inference_engine

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize inference engine with default model
    config = DA3Config.from_preset(Preset.PHOTO_REALISTIC)
    config.export.output_dir = output_dir

    inference_engine = DA3InferenceEngine(config)
    inference_engine.load_model()

    print("Service initialized successfully")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Lux Depth V3",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": inference_engine is not None,
    }


@app.post("/depth/estimate", response_model=DepthResponse)
async def estimate_depth(
    request: Request,
    file: UploadFile = File(...),
    model_variant: Optional[ModelVariant] = None,
    metric_scaling: bool = False,
):
    """Estimate depth from uploaded image.

    Args:
        request: FastAPI request
        file: Uploaded image file
        model_variant: Model variant to use
        metric_scaling: Apply metric scaling

    Returns:
        Depth estimation response

    Raises:
        HTTPException: If validation fails
    """
    # Rate limiting
    client_ip = request.client.host
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Max 60 requests per minute.",
        )

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Must be image.",
        )

    # Read file
    contents = await file.read()

    # Validate file size
    file_size_mb = len(contents) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File size ({file_size_mb:.1f}MB) exceeds limit ({MAX_FILE_SIZE_MB}MB)",
        )

    # Load image
    try:
        image = Image.open(io.BytesIO(contents))

        # Validate dimensions
        if max(image.size) > MAX_IMAGE_DIMENSION:
            raise HTTPException(
                status_code=400,
                detail=f"Image dimension ({max(image.size)}) exceeds limit ({MAX_IMAGE_DIMENSION})",
            )

        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        image_array = np.array(image)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load image: {str(e)}",
        )

    # Create input
    img_input = ImageInput(array=image_array)

    # Update config if needed
    if model_variant is not None and model_variant != inference_engine.config.model_variant:
        # Reload model with new variant
        inference_engine.config.model_variant = model_variant
        inference_engine.load_model(force_reload=True)

    # Configure postprocessing
    inference_engine.config.postprocessing.apply_metric_scaling = metric_scaling

    # Run inference
    start_time = time.perf_counter()

    try:
        result = inference_engine.inference(img_input)

        # Apply postprocessing
        postprocessor = Postprocessor(inference_engine.config.postprocessing)
        result = postprocessor.process(result)

        # Export results
        exporter = Exporter(inference_engine.config.export)
        filename_base = f"depth_{int(time.time() * 1000)}"
        exported = exporter.export(result, filename_base)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}",
        )

    processing_time = (time.perf_counter() - start_time) * 1000  # ms

    # Create response
    depth_range = result.get_depth_range()

    response = DepthResponse(
        depth_range=depth_range,
        processing_time_ms=processing_time,
        model_variant=inference_engine.config.model_variant.value,
        export_paths={fmt: str(path) for fmt, path in exported.items()},
    )

    return response


@app.get("/depth/download/{filename}")
async def download_depth(filename: str):
    """Download exported depth file.

    Args:
        filename: Filename to download

    Returns:
        File response
    
    Security:
        - Validates filename to prevent directory traversal (CWE-22)
        - Ensures path stays within configured output directory
        - Only serves regular files (not directories or special files)
    """
    # Security: ALLOWLIST validation - CodeQL-recognized sanitizer pattern
    # Restrict to safe filename pattern (alphanumeric + common separators)
    # This is the canonical pattern for preventing path traversal attacks
    SAFE_FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
    
    if not filename or not SAFE_FILENAME_PATTERN.match(filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # Additional protection: reject special directory names
    if filename in ('.', '..'):
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # Build safe path from trusted base directory
    output_dir_resolved = output_dir.resolve()
    
    # Construct path - safe because filename passed allowlist validation
    try:
        safe_file_path = output_dir_resolved / filename
        safe_file_path = safe_file_path.resolve(strict=False)
    except (RuntimeError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Verify containment within output directory (defense in depth)
    try:
        # Use is_relative_to (Python 3.9+) or fallback to relative_to
        if hasattr(Path, 'is_relative_to'):
            if not safe_file_path.is_relative_to(output_dir_resolved):
                raise HTTPException(status_code=400, detail="Invalid filename")
        else:
            # Fallback for Python 3.8
            try:
                safe_file_path.relative_to(output_dir_resolved)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid filename")
    except (TypeError, OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # Verify file exists and is regular file (not directory/device/symlink to outside)
    try:
        if not safe_file_path.exists() or not safe_file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
    except (OSError, RuntimeError):
        raise HTTPException(status_code=404, detail="File not found")

    # Serve file - safe_file_path is validated by allowlist and containment checks
    return FileResponse(
        path=safe_file_path,
        filename=safe_file_path.name,
        media_type="application/octet-stream",
    )


@app.get("/models/list")
async def list_models():
    """List available model variants."""
    return {
        "models": [variant.value for variant in ModelVariant],
        "presets": [preset.value for preset in Preset],
    }


def create_app(
    model_variant: ModelVariant = ModelVariant.METRIC_LARGE,
    output_dir_path: Path = Path("service_output"),
) -> FastAPI:
    """Create FastAPI app with custom configuration.

    Args:
        model_variant: Default model variant
        output_dir_path: Output directory path

    Returns:
        Configured FastAPI app
    """
    global output_dir
    # Resolve output directory to an absolute, normalized path for security checks
    output_dir = output_dir_path.resolve()
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8088,
        log_level="info",
    )
