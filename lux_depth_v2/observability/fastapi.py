from __future__ import annotations

import os
import secrets
from typing import TYPE_CHECKING, Optional

# Import Request at module level for type annotations in nested functions
# (FastAPI's eval_str=True requires it in globals)
from fastapi import Request  # noqa: F401

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
    - /metrics endpoint (Prometheus text exposition with optional authentication)

    No changes to existing routes required.

    Args:
        app: FastAPI application instance
        service_name: Service name for logging and metrics
        enable_metrics: Enable Prometheus metrics (unused - controlled by LUX_METRICS_ENABLED env var)
        enable_json_logging: Enable JSON logging format (default: True for service mode)
    """
    # Defer heavy imports; keeps non-service contexts clean.
    from fastapi import HTTPException, Response

    from .middleware import ObservabilityMiddleware

    if enable_json_logging is None:
        enable_json_logging = True  # service install defaults to JSON
    configure_structured_logging(force_json=bool(enable_json_logging))

    # Middleware (request id + metrics + access logging)
    app.add_middleware(ObservabilityMiddleware, service_name=service_name)

    # /metrics endpoint with optional bearer token authentication
    @app.get("/metrics", include_in_schema=False, response_class=Response)
    async def metrics_endpoint(request: Request):
        m = get_metrics()
        # Check if authentication is required
        if m.auth.token:
            auth_header = request.headers.get("authorization", "")
            if not auth_header.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
            token = auth_header[7:]  # Remove "Bearer " prefix
            # Use constant-time comparison to prevent timing attacks
            if not secrets.compare_digest(token, m.auth.token):
                raise HTTPException(status_code=403, detail="Invalid metrics token")
        body, content_type, status = m.render()
        return Response(content=body, media_type=content_type, status_code=status)
