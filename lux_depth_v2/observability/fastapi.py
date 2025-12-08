from __future__ import annotations

import os
from typing import Optional

from .json_logging import configure_structured_logging
from .metrics import get_metrics


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def install_observability(
    app,
    *,
    service_name: str = "lux_depth_v2",
    enable_metrics: Optional[bool] = None,
    enable_json_logging: Optional[bool] = None,
) -> None:
    """
    Additive installer for FastAPI apps:
    - Structured logging (JSON by default here)
    - Correlation + metrics middleware
    - /metrics endpoint (Prometheus text exposition)

    No changes to existing routes required.
    """
    # Defer heavy imports; keeps non-service contexts clean.
    from fastapi import Response
    from fastapi.responses import PlainTextResponse

    from .middleware import ObservabilityMiddleware

    if enable_json_logging is None:
        enable_json_logging = True  # service install defaults to JSON
    configure_structured_logging(force_json=bool(enable_json_logging))

    # Middleware (request id + metrics + access logging)
    app.add_middleware(ObservabilityMiddleware, service_name=service_name)

    # /metrics endpoint
    @app.get("/metrics", include_in_schema=False, response_class=Response)
    async def metrics_endpoint():
        m = get_metrics()
        body, content_type, status = m.render()
        return Response(content=body, media_type=content_type, status_code=status)
