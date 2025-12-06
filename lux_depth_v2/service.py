from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Optional

from .config import PipelineConfig
from .logging_utils import setup_logging
from .pipeline import LuxPipelineV2


def run_service(cfg: PipelineConfig, host: str = "0.0.0.0", port: int = 8088, logger=None) -> None:
    """Run a FastAPI service around the pipeline (persistent models, low latency)."""
    logger = logger or setup_logging("INFO")

    try:
        import uvicorn  # type: ignore
        from fastapi import FastAPI, File, UploadFile  # type: ignore
        from fastapi.responses import JSONResponse  # type: ignore
    except Exception as e:
        raise RuntimeError("Service mode requires fastapi and uvicorn. Install: fastapi uvicorn[standard]") from e

    app = FastAPI(title="LuxDepthV2", version="2.0")

    pipe = LuxPipelineV2(cfg, logger=logger)
    sem = asyncio.Semaphore(int(cfg.service.max_concurrency) if cfg.service else 1)

    incoming_dir = Path(cfg.output_dir) / "_incoming"
    incoming_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/v2/process")
    async def process(image: UploadFile = File(...), depth: Optional[UploadFile] = File(None)):
        async with sem:
            req_id = uuid.uuid4().hex
            img_path = incoming_dir / f"{req_id}_{Path(image.filename or 'image').name}"
            with open(img_path, "wb") as f:
                f.write(await image.read())

            depth_path = None
            if depth is not None:
                depth_path = incoming_dir / f"{req_id}_{Path(depth.filename or 'depth').name}"
                with open(depth_path, "wb") as f:
                    f.write(await depth.read())

            try:
                rep = pipe.process_one(img_path, depth_path=depth_path)
                return JSONResponse(rep)
            except Exception as e:
                logger.exception(f"request failed: {req_id}: {e}")
                return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

    logger.info(f"Starting service on {host}:{port} | output_dir={cfg.output_dir}")
    uvicorn.run(app, host=host, port=int(port))
