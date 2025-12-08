"""
Lux Depth V2 Observability (additive)

Features:
- Prometheus /metrics endpoint (Prometheus text exposition format)
- Structured JSON logging (one JSON object per line)
- Request correlation (X-Request-ID propagation + context binding)

All behavior is additive and gated via environment variables:
- LUX_METRICS_ENABLED=1|0 (default: 1)
- LUX_METRICS_TOKEN=<bearer token> (optional, enables /metrics auth when set)
- LUX_LOG_FORMAT=json|text (default: json for service installs, otherwise unchanged)
- LUX_HTTP_ACCESS_LOG=1|0 (default: 1)
"""

from .fastapi import install_observability
from .context import get_request_id, new_request_id
from .json_logging import configure_structured_logging
from .metrics import get_metrics, PrometheusMetrics

__all__ = [
    "install_observability",
    "configure_structured_logging",
    "get_metrics",
    "PrometheusMetrics",
    "get_request_id",
    "new_request_id",
]
