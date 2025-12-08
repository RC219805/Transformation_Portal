from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

PROM_AVAILABLE = True
try:
    from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
except Exception:  # pragma: no cover
    PROM_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; charset=utf-8"
    CollectorRegistry = object  # type: ignore
    Counter = Gauge = Histogram = object  # type: ignore
    generate_latest = None  # type: ignore


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class MetricsAuth:
    token: Optional[str] = None  # If set, /metrics requires Authorization: Bearer <token>


class PrometheusMetrics:
    """
    Prometheus metrics (safe defaults; avoids high-cardinality labels).
    """

    def __init__(self) -> None:
        self.enabled: bool = PROM_AVAILABLE and _env_bool("LUX_METRICS_ENABLED", True)
        self.auth = MetricsAuth(token=os.getenv("LUX_METRICS_TOKEN") or None)

        self.registry = None
        self.http_requests_total = None
        self.http_request_duration = None
        self.http_inflight = None
        self.pipeline_stage_duration = None
        self.pipeline_total_duration = None
        self.exceptions_total = None

        if not self.enabled:
            return

        self.registry = CollectorRegistry(auto_describe=True)

        self.http_requests_total = Counter(
            "lux_http_requests_total",
            "Total HTTP requests processed.",
            ["method", "route", "status_code"],
            registry=self.registry,
        )
        self.http_request_duration = Histogram(
            "lux_http_request_duration_seconds",
            "HTTP request duration in seconds.",
            ["method", "route"],
            registry=self.registry,
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 60),
        )
        self.http_inflight = Gauge(
            "lux_http_inflight_requests",
            "In-flight HTTP requests.",
            registry=self.registry,
        )

        self.pipeline_stage_duration = Histogram(
            "lux_pipeline_stage_duration_seconds",
            "Per-stage pipeline duration in seconds.",
            ["stage"],
            registry=self.registry,
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 60, 120),
        )
        self.pipeline_total_duration = Histogram(
            "lux_pipeline_total_duration_seconds",
            "End-to-end pipeline duration in seconds.",
            registry=self.registry,
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 60, 120, 300),
        )

        self.exceptions_total = Counter(
            "lux_exceptions_total",
            "Total unhandled exceptions.",
            ["where", "exc_type"],
            registry=self.registry,
        )

    def observe_http(self, *, method: str, route: str, status_code: int, duration_s: float) -> None:
        if not self.enabled:
            return
        assert self.http_requests_total is not None
        assert self.http_request_duration is not None
        assert self.http_inflight is not None
        self.http_requests_total.labels(method=method, route=route, status_code=str(status_code)).inc()
        self.http_request_duration.labels(method=method, route=route).observe(duration_s)

    def inflight_inc(self) -> None:
        if not self.enabled:
            return
        assert self.http_inflight is not None
        self.http_inflight.inc()

    def inflight_dec(self) -> None:
        if not self.enabled:
            return
        assert self.http_inflight is not None
        self.http_inflight.dec()

    def observe_pipeline_timings(self, timing_s: Mapping[str, float]) -> None:
        """
        timing_s is expected to be small, fixed-key dict (e.g., stage breakdown).
        Do NOT pass request_id or filenames here (cardinality explosion).
        """
        if not self.enabled:
            return
        assert self.pipeline_stage_duration is not None
        assert self.pipeline_total_duration is not None

        total = None
        for stage, sec in timing_s.items():
            if not isinstance(sec, (int, float)):
                continue
            if stage == "total":
                total = float(sec)
            self.pipeline_stage_duration.labels(stage=str(stage)).observe(float(sec))

        if total is not None:
            self.pipeline_total_duration.observe(total)

    def observe_exception(self, *, where: str, exc_type: str) -> None:
        if not self.enabled:
            return
        assert self.exceptions_total is not None
        self.exceptions_total.labels(where=where, exc_type=exc_type).inc()

    def render(self) -> Tuple[bytes, str, int]:
        """
        Returns (body, content_type, status_code)
        """
        if not PROM_AVAILABLE:
            return (b"prometheus_client not installed\n", CONTENT_TYPE_LATEST, 503)
        if not self.enabled:
            return (b"metrics disabled\n", CONTENT_TYPE_LATEST, 404)
        assert self.registry is not None
        assert generate_latest is not None
        return (generate_latest(self.registry), CONTENT_TYPE_LATEST, 200)


_METRICS_SINGLETON: Optional[PrometheusMetrics] = None


def get_metrics() -> PrometheusMetrics:
    global _METRICS_SINGLETON
    if _METRICS_SINGLETON is None:
        _METRICS_SINGLETON = PrometheusMetrics()
    return _METRICS_SINGLETON
