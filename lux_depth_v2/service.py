from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Optional

from .config import PipelineConfig
from .logging_utils import setup_logging
from .pipeline import LuxPipelineV2


def validate_filepath(filename: str) -> None:
    """Validate uploaded filename to prevent path traversal attacks."""
    if not filename:
        raise ValueError("Filename cannot be empty")
    
    # Check for path traversal attempts
    if ".." in filename or "/" in filename or "\\" in filename:
        raise ValueError(f"Invalid filename: path traversal detected in '{filename}'")
    
    # Check for suspicious characters
    if any(c in filename for c in ['\x00', '\n', '\r']):
        raise ValueError(f"Invalid filename: null or newline characters in '{filename}'")
    
    # Verify extension is allowed
    allowed_extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    ext = Path(filename).suffix.lower()
    if ext not in allowed_extensions:
        raise ValueError(f"Invalid file extension: {ext}. Allowed: {allowed_extensions}")


def run_service(cfg: PipelineConfig, host: str = "0.0.0.0", port: int = 8088, logger=None) -> None:
    """Run a FastAPI service around the pipeline (persistent models, low latency)."""
    logger = logger or setup_logging("INFO")

    try:
        import uvicorn  # type: ignore
        from fastapi import FastAPI, File, UploadFile, HTTPException, Request  # type: ignore
        from fastapi.responses import JSONResponse  # type: ignore
        from slowapi import Limiter, _rate_limit_exceeded_handler  # type: ignore
        from slowapi.util import get_remote_address  # type: ignore
        from slowapi.errors import RateLimitExceeded  # type: ignore
    except ImportError as e:
        missing = str(e).split("'")[-2] if "'" in str(e) else "unknown"
        raise RuntimeError(
            f"Service mode requires fastapi, uvicorn, and slowapi. "
            f"Install: pip install fastapi 'uvicorn[standard]' slowapi"
        ) from e

    # Rate limiting (10 requests per minute per IP)
    limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
    app = FastAPI(title="LuxDepthV2", version="2.0")
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Max upload size (100MB default)
    MAX_UPLOAD_SIZE = int(os.environ.get("MAX_UPLOAD_SIZE", 100 * 1024 * 1024))

    pipe = LuxPipelineV2(cfg, logger=logger)
    sem = asyncio.Semaphore(int(cfg.service.max_concurrency) if cfg.service else 1)

    incoming_dir = Path(cfg.output_dir) / "_incoming"
    incoming_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/health")
    async def health():
        return {"ok": True, "version": "2.0"}

    @app.post("/v2/process")
    @limiter.limit("10/minute")
    async def process(
        request: Request,
        image: UploadFile = File(...),
        depth: Optional[UploadFile] = File(None)
    ):
        async with sem:
            # Validate filenames
            try:
                validate_filepath(image.filename or "image.png")
                if depth:
                    validate_filepath(depth.filename or "depth.png")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

            # Check file size
            img_data = await image.read()
            if len(img_data) > MAX_UPLOAD_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"Image too large: {len(img_data)} bytes (max {MAX_UPLOAD_SIZE})"
                )

            req_id = uuid.uuid4().hex
            # Use sanitized filename
            safe_img_name = Path(image.filename or "image.png").name
            img_path = incoming_dir / f"{req_id}_{safe_img_name}"
            
            with open(img_path, "wb") as f:
                f.write(img_data)

            depth_path = None
            if depth is not None:
                depth_data = await depth.read()
                if len(depth_data) > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Depth too large: {len(depth_data)} bytes (max {MAX_UPLOAD_SIZE})"
                    )
                safe_depth_name = Path(depth.filename or "depth.png").name
                depth_path = incoming_dir / f"{req_id}_{safe_depth_name}"
                with open(depth_path, "wb") as f:
                    f.write(depth_data)

            try:
                rep = pipe.process_one(img_path, depth_path=depth_path)
                return JSONResponse(rep)
            except Exception as e:
                logger.exception(f"request failed: {req_id}: {e}")
                return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

    logger.info(f"Starting service on {host}:{port} | output_dir={cfg.output_dir}")
    logger.info(f"Security: Rate limiting enabled (10/min), max upload size: {MAX_UPLOAD_SIZE} bytes")
    uvicorn.run(app, host=host, port=int(port))
