from __future__ import annotations

from fastapi import FastAPI

from .fastapi import install_observability

app = FastAPI(title="Lux Depth V2 Observability Smoke App")

install_observability(app, service_name="lux_depth_v2_smoke")


@app.get("/health")
def health():
    return {"status": "ok"}
