from __future__ import annotations

import asyncio
import hmac
import json
import logging
import mimetypes
import os
import re
import sys
import tempfile
import time
import uuid
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Deque, Dict, List, Mapping, Optional
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler as fastapi_request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import Response, StreamingResponse

# ----------------------------
# In-memory job store (MVP)
# ----------------------------

LOGGER = logging.getLogger(__name__)


def _env_csv(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    values = [item.strip() for item in raw.split(",")]
    return [value for value in values if value]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return max(minimum, parsed)


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        return default
    return max(minimum, parsed)


PORTAL_HTML = Path(__file__).resolve().parent / "portal.html"
REPO_ROOT = Path(__file__).resolve().parent
ARCHIVE_GOVERNANCE_SCRIPT = REPO_ROOT / "tools" / "archive_governance.py"


def _normalize_root_path(value: str | Path) -> Path:
    raw = str(value or "").strip()
    if not raw or raw.startswith("~") or "\x00" in raw:
        raise ValueError("Invalid path root")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return Path(os.path.realpath(candidate))


def _default_allowed_path_roots() -> List[Path]:
    roots: List[Path] = [REPO_ROOT]
    candidate_paths: List[Path] = [Path(tempfile.gettempdir()).resolve()]
    if os.name != "nt":
        # Accept the common POSIX temp aliases used by operators and local tooling.
        candidate_paths.extend([Path("/tmp"), Path("/private/tmp")])

    for candidate in candidate_paths:
        try:
            normalized = _normalize_root_path(candidate)
        except (OSError, RuntimeError, ValueError):
            continue
        if normalized not in roots:
            roots.append(normalized)
    return roots


def _env_path_roots(name: str, default: List[Path]) -> List[Path]:
    raw = os.getenv(name)
    if raw is None:
        values = [str(path) for path in default]
    else:
        values = [item.strip() for item in raw.split(",") if item.strip()]

    roots: List[Path] = []
    invalid_values: List[str] = []
    for value in values:
        try:
            root = _normalize_root_path(value)
        except (OSError, RuntimeError, ValueError):
            invalid_values.append(value)
            continue
        if root not in roots:
            roots.append(root)
    if invalid_values:
        LOGGER.warning(
            "%s ignored invalid roots: %s",
            name,
            ", ".join(sorted(set(invalid_values))),
        )
    if raw is not None and not roots:
        raise RuntimeError(f"{name} is set but contains no valid roots")
    if not roots:
        raise RuntimeError(f"{name} resolved to an empty root allowlist")
    return roots


def _resolve_untrusted_request_path(path_value: str) -> Path:
    raw = str(path_value or "").strip()
    if not raw or raw.startswith("~") or "\x00" in raw:
        raise ValueError("Invalid path value")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return Path(os.path.realpath(candidate))


def _is_within_allowed_roots(
    candidate: Path,
    allowed_roots: List[Path],
) -> bool:
    candidate_real = os.path.realpath(candidate)
    for root in allowed_roots:
        root_real = os.path.realpath(root)
        try:
            if os.path.commonpath([candidate_real, root_real]) == root_real:
                return True
        except ValueError:
            # Mixed absolute/relative or drive mismatch on non-POSIX platforms.
            continue
    return False


def _validate_path_against_roots(
    path_value: str,
    allowed_roots: List[Path],
) -> str:
    try:
        resolved = _resolve_untrusted_request_path(path_value)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("Invalid path value") from exc

    if not _is_within_allowed_roots(resolved, allowed_roots):
        raise ValueError("Path outside allowed roots")
    return str(resolved)


LOG_TAIL_LIMIT = 2000
STATUS_LOG_LIMIT = 250
EVENT_QUEUE_MAXSIZE = 512
HEARTBEAT_SECONDS = 15
JOB_RETENTION_SECONDS = _env_int("TP_JOB_RETENTION_SECONDS", 3600, minimum=1)
CLEANUP_INTERVAL_SECONDS = _env_int(
    "TP_CLEANUP_INTERVAL_SECONDS",
    60,
    minimum=1,
)
CANCEL_GRACE_SECONDS = _env_float(
    "TP_CANCEL_GRACE_SECONDS",
    5.0,
    minimum=0.1,
)
JOB_LIST_LIMIT = _env_int("TP_JOB_LIST_LIMIT", 200, minimum=1)
MAX_INDEXED_ARTIFACTS = _env_int("TP_MAX_INDEXED_ARTIFACTS", 200, minimum=1)
PROGRESS_RE = re.compile(r"progress=(\d{1,3})%")
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
]
ALLOWED_ORIGINS = _env_csv(
    "TP_ALLOWED_ORIGINS",
    DEFAULT_ALLOWED_ORIGINS,
)
TRUSTED_HOSTS = _env_csv(
    "TP_TRUSTED_HOSTS",
    ["localhost", "127.0.0.1", "::1", "testserver"],
)
ENABLE_TRUSTED_HOSTS = _env_bool("TP_ENABLE_TRUSTED_HOSTS", True)
API_KEY_HEADER = os.getenv("TP_API_KEY_HEADER", "x-api-key").strip().lower() or "x-api-key"
API_KEY_SECRET = os.getenv("TP_API_KEY", "").strip()
ENFORCE_JOB_API_KEY = _env_bool("TP_ENFORCE_JOB_API_KEY", True)
ALLOW_SSE_QUERY_API_KEY = _env_bool("TP_ALLOW_SSE_QUERY_API_KEY", False)
TRUST_X_FORWARDED_FOR = _env_bool("TP_TRUST_X_FORWARDED_FOR", False)
TRUSTED_PROXY_IPS = set(_env_csv("TP_TRUSTED_PROXY_IPS", []))
MAX_REQUEST_BYTES = _env_int("TP_MAX_REQUEST_BYTES", 1024 * 1024, minimum=1024)
RATE_LIMIT_PER_MINUTE = _env_int("TP_RATE_LIMIT_PER_MINUTE", 60, minimum=0)
MAX_CONCURRENT_JOBS = _env_int("TP_MAX_CONCURRENT_JOBS", 4, minimum=1)
RATE_LIMIT_WINDOW_SECONDS = 60.0
DEFAULT_ALLOWED_PATH_ROOTS = _default_allowed_path_roots()
ALLOWED_INPUT_ROOTS = _env_path_roots(
    "TP_ALLOWED_INPUT_ROOTS",
    DEFAULT_ALLOWED_PATH_ROOTS,
)
ALLOWED_OUTPUT_ROOTS = _env_path_roots(
    "TP_ALLOWED_OUTPUT_ROOTS",
    DEFAULT_ALLOWED_PATH_ROOTS,
)
ALLOWED_PATH_ROOTS = list(dict.fromkeys([*ALLOWED_INPUT_ROOTS, *ALLOWED_OUTPUT_ROOTS]))
ENABLE_API_DOCS = _env_bool("TP_ENABLE_API_DOCS", False)
READY_VERBOSE = _env_bool("TP_READY_VERBOSE", False)
DEFAULT_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com data:; "
    "img-src 'self' data: blob:; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "frame-ancestors 'none'; "
    "form-action 'self';"
)
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "X-Permitted-Cross-Domain-Policies": "none",
    "Content-Security-Policy": os.getenv("TP_CSP", DEFAULT_CSP).strip() or DEFAULT_CSP,
}

PRESET_CATALOG: Dict[str, List[Dict[str, Any]]] = {
    "lux-depth-v3": [
        {
            "name": "premium",
            "label": "premium (Stable)",
            "stability": "stable",
            "description": "Balanced production quality preset",
            "is_research": False,
            "recommended_args": {
                "quality_tier": "premium",
                "depth_backend": "da3",
                "enable_segmentation": True,
                "segmentation_backend": "efficientsam",
                "strict_segmentation": True,
                "materials_v3": True,
                "pbr": True,
                "emit_master16": True,
                "emit_upscaled16": True,
                "emit_report": True,
                "emit_run_card": True,
                "emit_marketing": False,
                "enable_v2": False,
                "enable_reconstruction": False,
            },
            "advanced_sections": [],
        },
        {
            "name": "default",
            "label": "default (Canary)",
            "stability": "canary",
            "description": "Canary preset for iterative validation",
            "is_research": False,
            "recommended_args": {
                "quality_tier": "standard",
                "depth_backend": "da3",
                "enable_segmentation": False,
                "segmentation_backend": "stub",
                "strict_segmentation": False,
                "materials_v3": False,
                "pbr": False,
                "emit_master16": True,
                "emit_upscaled16": False,
                "emit_report": True,
                "emit_run_card": True,
                "emit_marketing": False,
                "enable_v2": False,
                "enable_reconstruction": False,
            },
            "advanced_sections": ["advanced"],
        },
        {
            "name": "depth-anything-v3.1-research-m4",
            "label": "v3.1-m4 (Experimental)",
            "stability": "experimental",
            "description": "Research-only preset" " requiring non-commercial" " acknowledgments",
            "is_research": True,
            "recommended_args": {
                "quality_tier": "apex",
                "depth_backend": "depth_pro",
                "enable_segmentation": True,
                "segmentation_backend": "sam2",
                "strict_segmentation": True,
                "materials_v3": True,
                "pbr": True,
                "emit_master16": True,
                "emit_upscaled16": True,
                "emit_report": True,
                "emit_run_card": True,
                "emit_marketing": False,
                "enable_v2": True,
                "v2_preset": "default",
                "enable_reconstruction": False,
            },
            "advanced_sections": ["governance", "advanced"],
        },
    ],
    "archive-gate-a": [
        {
            "name": "default",
            "label": "default (Stable)",
            "stability": "stable",
            "description": "Manifest and provenance assembly",
            "is_research": False,
            "recommended_args": {"dedup": True, "sign": True},
            "advanced_sections": [],
        }
    ],
    "archive-gate-b": [
        {
            "name": "default",
            "label": "default (Stable)",
            "stability": "stable",
            "description": "BagIt packaging and validation workflow",
            "is_research": False,
            "recommended_args": {"dedup": True, "sign": True},
            "advanced_sections": [],
        }
    ],
    "archive-gate-c": [
        {
            "name": "default",
            "label": "default (Stable)",
            "stability": "stable",
            "description": "METS/PROV/STAC export workflow",
            "is_research": False,
            "recommended_args": {"dedup": True, "sign": True},
            "advanced_sections": [],
        }
    ],
}


@dataclass
class Job:
    id: str
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    done_published_at: Optional[float] = None  # Set after 'done' event is published
    state: str = "queued"  # queued|running|succeeded|failed|canceled
    progress: int = 0
    exit_code: Optional[int] = None
    request: Dict[str, Any] = field(default_factory=dict)
    logs_tail: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    proc: Optional[asyncio.subprocess.Process] = None
    terminate_task: Optional[asyncio.Task[None]] = None
    cancel_requested: bool = False
    error: Optional[Dict[str, Any]] = None

    def add_log(self, line: str, limit: int = LOG_TAIL_LIMIT) -> None:
        self.logs_tail.append(line)
        if len(self.logs_tail) > limit:
            self.logs_tail = self.logs_tail[-limit:]


JOBS: Dict[str, Job] = {}
EVENT_SUBSCRIBERS: Dict[str, Dict[str, "asyncio.Queue[Dict[str, Any]]"]] = {}
RATE_LIMIT_BUCKETS: Dict[str, Deque[float]] = {}
JOB_ADMISSION_LOCK = asyncio.Lock()

# Gate pipelines integrated directly
ARCHIVE_GATE_PIPELINES = {"archive-gate-a", "archive-gate-b", "archive-gate-c"}
ALLOWED_PIPELINES = {"lux-depth-v3", *ARCHIVE_GATE_PIPELINES}
ARCHIVE_GATE_DEFAULT_COMMANDS = {
    "archive-gate-a": "fixity-scan",
    "archive-gate-b": "bag-build",
    "archive-gate-c": "mets-export",
}
ARCHIVE_GATE_ALLOWED_COMMANDS = {
    "archive-gate-a": {
        "fixity-scan",
        "fixity-verify",
        "manifest-build",
        "rights-apply",
    },
    "archive-gate-b": {"bag-build", "bag-validate", "dedup-plan"},
    "archive-gate-c": {"mets-export", "prov-export", "stac-export"},
}
ALLOWED_QUALITY = {"standard", "premium", "apex"}
ALLOWED_BACKENDS = {"da3", "depth_pro"}
ALLOWED_SEGMENTATION_BACKENDS = {"stub", "efficientsam", "sam2"}
ALLOWED_SAM2_MODEL_SIZES = {"base", "large"}
ALLOWED_GROUPING_MODES = {"single", "parent_dir"}
ALLOWED_RECONSTRUCTION_TIERS = {
    "apex_research",
    "apex_research_ultra",
    "experimental",
}
ALLOWED_RAW_INGEST_MODES = {"auto", "force_rawpy", "force_preview"}
ALLOWED_RAW_WB_MODES = {"camera"}
ALLOWED_RAW_DEMOSAIC = {"AHD"}
ALLOWED_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}
DEPTH_BACKEND_ALIASES = {
    "depth_anything_v3": "da3",
    "depth-anything-v3": "da3",
}
VALIDATION_REASON_CODES = {
    "Unsupported pipeline": "unsupported_pipeline",
    "input_dir and output_dir are required": "missing_required_paths",
    "Invalid path value": "invalid_path_value",
    "Path outside allowed roots": "path_outside_allowed_roots",
    "Invalid quality_tier": "invalid_quality_tier",
    "Invalid depth_backend": "invalid_depth_backend",
    "Invalid segmentation_backend": "invalid_segmentation_backend",
    "Invalid sam2_model_size": "invalid_sam2_model_size",
    "Invalid reconstruction_tier": "invalid_reconstruction_tier",
    "Invalid raw_ingest_mode": "invalid_raw_ingest_mode",
    "Invalid raw_wb_mode": "invalid_raw_wb_mode",
    "Invalid raw_demosaic": "invalid_raw_demosaic",
    "Invalid log_level": "invalid_log_level",
    "verbose and quiet are mutually exclusive": "conflicting_log_verbosity_flags",
    "Archive governance runner unavailable": "archive_runner_unavailable",
    "Invalid archive_command": "invalid_archive_command",
    "Invalid archive integer option": "invalid_archive_integer_option",
}


def _now() -> float:
    return time.time()


def _sse(event: str, data: Dict[str, Any]) -> str:
    # SSE payload: event type + JSON data, double newline.
    payload = json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"event: {event}\ndata: {payload}\n\n"


def _error_obj(
    code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {"code": code, "message": message, "details": details or {}}


def _api_envelope(
    schema: str,
    *,
    success: bool,
    data: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {"schema": schema, "success": success, "data": data, "error": error}


def _error_response(
    status_code: int,
    *,
    code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    schema: str = "tp.orchestrator.error.v1",
    headers: Optional[Mapping[str, str]] = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=_api_envelope(
            schema,
            success=False,
            data=None,
            error=_error_obj(code, message, details),
        ),
        headers=headers,
    )


HTTP_STATUS_ERROR_CODES = {
    400: "INVALID_ARGUMENT",
    401: "UNAUTHORIZED",
    404: "NOT_FOUND",
    413: "REQUEST_TOO_LARGE",
    429: "RATE_LIMITED",
}


def _is_api_v1_path(path: str) -> bool:
    return path.startswith("/v1/")


def _http_status_error_code(status_code: int) -> str:
    return HTTP_STATUS_ERROR_CODES.get(status_code, "HTTP_ERROR")


def _cleanup_expired_jobs(now: float) -> None:
    expired = [
        job_id
        for job_id, job in JOBS.items()
        if job.finished_at is not None and now - job.finished_at >= JOB_RETENTION_SECONDS
    ]
    for job_id in expired:
        JOBS.pop(job_id, None)
        EVENT_SUBSCRIBERS.pop(job_id, None)


def _active_job_count() -> int:
    return sum(1 for job in JOBS.values() if job.state in {"queued", "running"})


def _cleanup_rate_limit_buckets(now: float) -> None:
    if RATE_LIMIT_PER_MINUTE <= 0:
        return

    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    stale_ips: List[str] = []
    for client_ip, timestamps in RATE_LIMIT_BUCKETS.items():
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()
        if not timestamps:
            stale_ips.append(client_ip)
    for client_ip in stale_ips:
        RATE_LIMIT_BUCKETS.pop(client_ip, None)


async def _cleanup_loop() -> None:
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        now = _now()
        _cleanup_expired_jobs(now)
        _cleanup_rate_limit_buckets(now)


def _pick(args: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = args.get(key)
        if value is not None:
            return value
    return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _extract_progress_percent(line: str) -> Optional[int]:
    match = PROGRESS_RE.search(line)
    if not match:
        return None
    try:
        return max(0, min(100, int(match.group(1))))
    except ValueError:
        return None


def _canonical_depth_backend(value: Any) -> str:
    backend = str(value or "").strip().lower()
    if not backend:
        return ""
    return DEPTH_BACKEND_ALIASES.get(backend, backend)


def _is_mutating_job_endpoint(method: str, path: str) -> bool:
    if method != "POST":
        return False
    if path == "/v1/jobs":
        return True
    return bool(re.fullmatch(r"/v1/jobs/[^/]+/cancel", path))


def _is_job_events_endpoint(path: str) -> bool:
    return bool(re.fullmatch(r"/v1/jobs/[^/]+/events", path))


def _is_protected_job_endpoint(path: str) -> bool:
    return path == "/v1/jobs" or path.startswith("/v1/jobs/")


def _job_api_key_enforced() -> bool:
    return ENFORCE_JOB_API_KEY or bool(API_KEY_SECRET)


def _extract_client_ip(request: Request) -> str:
    peer_ip = request.client.host if request.client and request.client.host else None
    trust_forwarded = TRUST_X_FORWARDED_FOR or (peer_ip in TRUSTED_PROXY_IPS if peer_ip else False)

    if trust_forwarded:
        forwarded_for = request.headers.get("x-forwarded-for", "")
        if forwarded_for:
            first = forwarded_for.split(",")[0].strip()
            if first:
                return first

    if peer_ip:
        return peer_ip
    return "unknown"


def _is_rate_limited(client_ip: str, now: float) -> bool:
    if RATE_LIMIT_PER_MINUTE <= 0:
        return False

    timestamps = RATE_LIMIT_BUCKETS.get(client_ip)
    if timestamps is None:
        timestamps = deque()
        RATE_LIMIT_BUCKETS[client_ip] = timestamps

    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    while timestamps and timestamps[0] < cutoff:
        timestamps.popleft()

    if len(timestamps) >= RATE_LIMIT_PER_MINUTE:
        return True

    timestamps.append(now)
    return False


def _has_valid_api_key(request: Request) -> bool:
    if not API_KEY_SECRET:
        return True

    provided = request.headers.get(API_KEY_HEADER, "")
    if not provided:
        authorization = request.headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            provided = authorization[7:].strip()
    if not provided and ALLOW_SSE_QUERY_API_KEY and _is_job_events_endpoint(request.url.path):
        provided = request.query_params.get("api_key", "").strip()

    if not provided:
        return False
    return hmac.compare_digest(provided, API_KEY_SECRET)


def _enforce_content_length_limit(request: Request) -> Optional[JSONResponse]:
    if request.method not in {"POST", "PUT", "PATCH"}:
        return None

    content_length = request.headers.get("content-length")
    if not content_length:
        return None
    try:
        size = int(content_length)
    except ValueError:
        if _is_api_v1_path(request.url.path):
            return _error_response(
                400,
                code="INVALID_ARGUMENT",
                message=("invalid Content-Length header. " "Please ensure the header is a valid integer."),
                details={
                    "path": request.url.path,
                    "field": "content-length",
                },
            )
        return JSONResponse(
            status_code=400,
            content={"detail": "invalid Content-Length header"},
        )

    if size > MAX_REQUEST_BYTES:
        if _is_api_v1_path(request.url.path):
            return _error_response(
                413,
                code="REQUEST_TOO_LARGE",
                message=(f"request body too large" f" (max {MAX_REQUEST_BYTES} bytes)"),
                details={
                    "path": request.url.path,
                    "max_request_bytes": MAX_REQUEST_BYTES,
                },
            )
        return JSONResponse(
            status_code=413,
            content={
                "detail": (f"request body too large" f" (max {MAX_REQUEST_BYTES} bytes)"),
            },
        )
    return None


def _install_stream_body_limit(request: Request) -> None:
    if request.method not in {"POST", "PUT", "PATCH"}:
        return
    if getattr(
        request.state,
        "_tp_body_limit_installed",
        False,
    ):
        return

    original_receive = getattr(
        request,
        "_receive",
        None,
    )
    if original_receive is None:
        return

    async def limited_receive() -> Dict[str, Any]:
        message = await original_receive()
        if message.get("type") == "http.request":
            body = message.get("body", b"") or b""
            consumed = getattr(
                request.state,
                "_tp_body_bytes_received",
                0,
            )
            consumed += len(body)
            request.state._tp_body_bytes_received = consumed
            if consumed > MAX_REQUEST_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(f"request body too large" f" (max {MAX_REQUEST_BYTES} bytes)"),
                )
        return message

    setattr(request, "_receive", limited_receive)
    request.state._tp_body_limit_installed = True


def _sanitized_child_env() -> Dict[str, str]:
    child_env = os.environ.copy()
    sensitive_exact = {
        "TP_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    }
    sensitive_suffixes = (
        "_TOKEN",
        "_SECRET",
        "_PASSWORD",
        "_API_KEY",
        "_ACCESS_KEY",
        "_PRIVATE_KEY",
    )
    for key in list(child_env.keys()):
        upper = key.upper()
        if upper in sensitive_exact or upper.endswith(sensitive_suffixes):
            child_env.pop(key, None)
    return child_env


async def _publish_event(
    job_id: str,
    event: str,
    data: Dict[str, Any],
) -> None:
    subscribers = EVENT_SUBSCRIBERS.get(job_id)
    if not subscribers:
        return

    payload = {"event": event, "data": data}
    stale_subscribers: List[str] = []
    for subscriber_id, queue in list(subscribers.items()):
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            continue
        except RuntimeError:
            stale_subscribers.append(subscriber_id)

    for subscriber_id in stale_subscribers:
        subscribers.pop(subscriber_id, None)


async def _terminate_process(
    proc: asyncio.subprocess.Process,
    grace_seconds: float = CANCEL_GRACE_SECONDS,
) -> None:
    if proc.returncode is not None:
        return
    try:
        proc.terminate()
    except ProcessLookupError:
        return
    except Exception:
        return

    try:
        await asyncio.wait_for(proc.wait(), timeout=grace_seconds)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            return
        except Exception:
            return
        await proc.wait()


async def _request_cancel(job: Job) -> None:
    job.cancel_requested = True
    if job.proc is None or job.proc.returncode is not None:
        return

    if job.terminate_task is None or job.terminate_task.done():
        job.terminate_task = asyncio.create_task(_terminate_process(job.proc))


def _job_output_dir(job: Job) -> Optional[Path]:
    args = job.request.get("args")
    if not isinstance(args, dict):
        return None
    output_dir = str(
        _pick(args, "output_dir", "outputDir", default=""),
    ).strip()
    if not output_dir:
        return None
    return Path(output_dir).expanduser()


def _infer_artifact_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {
        ".json",
        ".yaml",
        ".yml",
        ".txt",
        ".md",
        ".log",
        ".csv",
    }:
        return "metadata"
    if suffix in {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".webp",
        ".exr",
    }:
        return "image"
    if suffix in {".zip", ".tar", ".gz", ".tgz", ".bag"}:
        return "archive"
    return "file"


def _artifact_content_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def _artifact_media_kind(path: Path) -> str:
    artifact_type = _infer_artifact_type(path)
    if artifact_type == "image":
        return "image"
    if artifact_type == "metadata":
        return "metadata"
    if artifact_type == "archive":
        return "archive"
    return "file"


def _artifact_is_previewable(path: Path) -> bool:
    return _artifact_media_kind(path) == "image" and _artifact_content_type(path).startswith("image/")


def _artifact_url(job_id: str, relative_path: str) -> str:
    return f"/v1/jobs/{quote(str(job_id), safe='')}" f"/artifacts/{quote(relative_path, safe='/')}"


def _serialize_indexed_artifact(
    *,
    job_id: str,
    relative_path: str,
    path: Path,
) -> Dict[str, Any]:
    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = None

    content_type = _artifact_content_type(path)
    return {
        "artifact_type": _infer_artifact_type(path),
        "media_kind": _artifact_media_kind(path),
        "previewable": _artifact_is_previewable(path),
        "content_type": content_type,
        "url": _artifact_url(job_id, relative_path),
        # Do not expose absolute server paths in API/SSE payloads.
        "path": relative_path,
        "relative_path": relative_path,
        "size_bytes": size_bytes,
    }


def _resolve_job_artifact_path(
    job: Job,
    artifact_path: str,
) -> tuple[Path, Path, str]:
    output_dir = _job_output_dir(job)
    if output_dir is None:
        raise FileNotFoundError("job_output_dir_missing")

    output_dir = Path(os.path.realpath(output_dir.expanduser()))
    if not output_dir.exists() or not output_dir.is_dir():
        raise FileNotFoundError("job_output_dir_missing")

    raw = str(artifact_path or "").strip()
    if not raw or raw.startswith("~") or "\x00" in raw:
        raise ValueError("invalid_artifact_path")

    candidate = Path(raw)
    if candidate.is_absolute():
        raise ValueError("absolute_artifact_path")

    resolved = Path(os.path.realpath(output_dir / candidate))
    try:
        relative_path = str(resolved.relative_to(output_dir))
    except ValueError as exc:
        raise ValueError("artifact_path_outside_job_output_dir") from exc

    return output_dir, resolved, relative_path


def _index_job_artifacts(job: Job) -> List[Dict[str, Any]]:
    output_dir = _job_output_dir(job)
    if output_dir is None:
        job.artifacts = {
            "output_dir": None,
            "items": [],
            "indexed_count": 0,
            "truncated": False,
        }
        return []
    if not output_dir.exists() or not output_dir.is_dir():
        job.artifacts = {
            "output_dir": str(output_dir),
            "items": [],
            "indexed_count": 0,
            "truncated": False,
        }
        return []

    selected: List[tuple[tuple[str, str], str, Path]] = []
    selected_keys: List[tuple[str, str]] = []
    total_files = 0
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        total_files += 1
        try:
            relative_path = str(path.relative_to(output_dir))
        except Exception:
            relative_path = path.name
        key = (relative_path.casefold(), relative_path)

        if len(selected) < MAX_INDEXED_ARTIFACTS:
            insert_at = bisect_left(selected_keys, key)
            selected_keys.insert(insert_at, key)
            selected.insert(insert_at, (key, relative_path, path))
            continue

        if key >= selected_keys[-1]:
            continue

        insert_at = bisect_left(selected_keys, key)
        selected_keys.insert(insert_at, key)
        selected.insert(insert_at, (key, relative_path, path))
        selected_keys.pop()
        selected.pop()

    truncated = total_files > MAX_INDEXED_ARTIFACTS

    items: List[Dict[str, Any]] = []
    for _, relative_path, path in selected:
        items.append(
            _serialize_indexed_artifact(
                job_id=job.id,
                relative_path=relative_path,
                path=path,
            )
        )

    job.artifacts = {
        "output_dir": str(output_dir),
        "items": items,
        "indexed_count": len(items),
        "truncated": truncated,
    }
    return items


def _serialize_job(job: Job, *, include_logs: bool = True) -> Dict[str, Any]:
    data = {
        "id": job.id,
        "pipeline": str(job.request.get("pipeline") or ""),
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "state": job.state,
        "progress": job.progress,
        "exit_code": job.exit_code,
        "events_url": f"/v1/jobs/{job.id}/events",
        "artifacts": job.artifacts,
        "error": job.error,
    }
    if include_logs:
        data["logs_tail"] = job.logs_tail[-STATUS_LOG_LIMIT:]
    return data


def _path_arg(
    args: Dict[str, Any],
    *keys: str,
    default: str,
    allowed_roots: List[Path],
) -> str:
    value = _pick(args, *keys, default=default)
    text = str(value or "").strip()
    return _validate_path_against_roots(text or default, allowed_roots)


def _int_arg(
    args: Dict[str, Any],
    *keys: str,
    default: int,
    minimum: int = 0,
) -> int:
    value = _pick(args, *keys, default=None)
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid archive integer option") from exc
    if parsed < minimum:
        raise ValueError("Invalid archive integer option")
    return parsed


def _archive_gate_argv(
    pipeline: str,
    args: Dict[str, Any],
    input_dir: str,
    output_dir: str,
) -> List[str]:
    if not ARCHIVE_GOVERNANCE_SCRIPT.is_file():
        raise ValueError("Archive governance runner unavailable")

    default_command = ARCHIVE_GATE_DEFAULT_COMMANDS[pipeline]
    command = str(
        _pick(
            args,
            "archive_command",
            "archiveCommand",
            default=default_command,
        )
        or ""
    ).strip()
    if command not in ARCHIVE_GATE_ALLOWED_COMMANDS[pipeline]:
        raise ValueError("Invalid archive_command")

    manifest_default = str(Path(output_dir) / "archive_manifest_v2.jsonl")
    rights_manifest_default = str(Path(output_dir) / "archive_manifest_v2.rights.jsonl")
    argv = [sys.executable, str(ARCHIVE_GOVERNANCE_SCRIPT), "--json", command]

    if command == "fixity-scan":
        archive_index = _path_arg(
            args,
            "archive_index",
            "archiveIndex",
            default=str(Path(input_dir) / "archive_index_normalized.csv.gz"),
            allowed_roots=ALLOWED_INPUT_ROOTS,
        )
        archive_root = _path_arg(
            args,
            "archive_root",
            "archiveRoot",
            default=input_dir,
            allowed_roots=ALLOWED_INPUT_ROOTS,
        )
        out_dir = _path_arg(
            args,
            "out_dir",
            "outDir",
            default=output_dir,
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        workers = _int_arg(args, "workers", default=1, minimum=1)

        argv.extend(
            [
                "--archive-index",
                archive_index,
                "--archive-root",
                archive_root,
                "--out-dir",
                out_dir,
                "--workers",
                str(workers),
            ]
        )
        strict = _pick(args, "strict")
        if strict is not None:
            argv.append("--strict" if _as_bool(strict) else "--no-strict")
        strict_identity = _pick(args, "strict_identity", "strictIdentity")
        if strict_identity is not None:
            flag = "--strict-identity" if _as_bool(strict_identity) else "--no-strict-identity"
            argv.append(flag)
        validate_schemas = _pick(args, "validate_schemas", "validateSchemas")
        if validate_schemas is not None:
            flag = "--validate-schemas" if _as_bool(validate_schemas, default=True) else "--no-validate-schemas"
            argv.append(flag)

    elif command == "fixity-verify":
        hash_manifest = _path_arg(
            args,
            "hash_manifest",
            "hashManifest",
            default=str(Path(output_dir) / "hash_manifest.csv.gz"),
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        archive_root = _path_arg(
            args,
            "archive_root",
            "archiveRoot",
            default=input_dir,
            allowed_roots=ALLOWED_INPUT_ROOTS,
        )
        report_path = _path_arg(
            args,
            "report_path",
            "reportPath",
            default=str(Path(output_dir) / "verification_report.json"),
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        verify_sample = _int_arg(
            args,
            "verify_sample",
            "verifySample",
            default=0,
            minimum=0,
        )
        workers = _int_arg(args, "workers", default=1, minimum=1)

        argv.extend(
            [
                "--hash-manifest",
                hash_manifest,
                "--archive-root",
                archive_root,
                "--report-path",
                report_path,
                "--verify-sample",
                str(verify_sample),
                "--workers",
                str(workers),
            ]
        )

    elif command == "manifest-build":
        archive_index = _path_arg(
            args,
            "archive_index",
            "archiveIndex",
            default=str(Path(input_dir) / "archive_index_normalized.csv.gz"),
            allowed_roots=ALLOWED_INPUT_ROOTS,
        )
        hash_manifest = _path_arg(
            args,
            "hash_manifest",
            "hashManifest",
            default=str(Path(output_dir) / "hash_manifest.csv.gz"),
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        archive_root = _path_arg(
            args,
            "archive_root",
            "archiveRoot",
            default=input_dir,
            allowed_roots=ALLOWED_INPUT_ROOTS,
        )
        out_jsonl = _path_arg(
            args,
            "out_jsonl",
            "outJsonl",
            default=manifest_default,
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        out_summary = _path_arg(
            args,
            "out_summary",
            "outSummary",
            default=str(Path(output_dir) / "archive_manifest_v2.summary.json"),
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        collection_id = str(
            _pick(
                args,
                "collection_id",
                "collectionId",
                default="UNSPECIFIED",
            )
            or "UNSPECIFIED"
        ).strip()
        owner = str(_pick(args, "owner", default="UNSPECIFIED") or "UNSPECIFIED").strip()

        argv.extend(
            [
                "--archive-index",
                archive_index,
                "--hash-manifest",
                hash_manifest,
                "--archive-root",
                archive_root,
                "--out-jsonl",
                out_jsonl,
                "--out-summary",
                out_summary,
                "--collection-id",
                collection_id or "UNSPECIFIED",
                "--owner",
                owner or "UNSPECIFIED",
            ]
        )
        rights_jsonl = _pick(args, "rights_jsonl", "rightsJsonl")
        if rights_jsonl:
            validated = _validate_path_against_roots(
                str(rights_jsonl),
                ALLOWED_INPUT_ROOTS,
            )
            argv.extend(["--rights-jsonl", validated])

    elif command == "rights-apply":
        manifest_jsonl = _path_arg(
            args,
            "manifest_jsonl",
            "manifestJsonl",
            default=manifest_default,
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        policy_yaml = _path_arg(
            args,
            "policy_yaml",
            "policyYaml",
            default=str(REPO_ROOT / "policy" / "archive" / "rights_flags.yml"),
            allowed_roots=ALLOWED_INPUT_ROOTS,
        )
        out_jsonl = _path_arg(
            args,
            "out_jsonl",
            "outJsonl",
            default=rights_manifest_default,
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        out_summary = _path_arg(
            args,
            "out_summary",
            "outSummary",
            default=str(Path(output_dir) / "asset_rights.summary.json"),
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )

        argv.extend(
            [
                "--manifest-jsonl",
                manifest_jsonl,
                "--policy-yaml",
                policy_yaml,
                "--out-jsonl",
                out_jsonl,
                "--out-summary",
                out_summary,
            ]
        )

    elif command == "bag-build":
        manifest_jsonl = _path_arg(
            args,
            "manifest_jsonl",
            "manifestJsonl",
            default=rights_manifest_default,
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        archive_root = _path_arg(
            args,
            "archive_root",
            "archiveRoot",
            default=input_dir,
            allowed_roots=ALLOWED_INPUT_ROOTS,
        )
        bag_dir = _path_arg(
            args,
            "bag_dir",
            "bagDir",
            default=str(Path(output_dir) / "bag"),
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        report_json = _path_arg(
            args,
            "report_json",
            "reportJson",
            default=str(Path(output_dir) / "bag_build_report.json"),
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        source_organization = str(
            _pick(
                args,
                "source_organization",
                "sourceOrganization",
                default="UNSPECIFIED",
            )
            or "UNSPECIFIED"
        ).strip()
        validate_with_bagit = _pick(
            args,
            "validate_with_bagit_python",
            "validateWithBagitPython",
        )
        if validate_with_bagit is None:
            validate_with_bagit = _pick(args, "sign")

        argv.extend(
            [
                "--manifest-jsonl",
                manifest_jsonl,
                "--archive-root",
                archive_root,
                "--bag-dir",
                bag_dir,
                "--report-json",
                report_json,
                "--source-organization",
                source_organization or "UNSPECIFIED",
            ]
        )
        if _as_bool(validate_with_bagit, default=False):
            argv.append("--validate-with-bagit-python")

    elif command == "bag-validate":
        bag_dir = _path_arg(
            args,
            "bag_dir",
            "bagDir",
            default=str(Path(output_dir) / "bag"),
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        report_json = _path_arg(
            args,
            "report_json",
            "reportJson",
            default=str(Path(output_dir) / "bag_validate_report.json"),
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        validate_with_bagit = _pick(
            args,
            "validate_with_bagit_python",
            "validateWithBagitPython",
        )
        if validate_with_bagit is None:
            validate_with_bagit = _pick(args, "sign")

        argv.extend(
            [
                "--bag-dir",
                bag_dir,
                "--report-json",
                report_json,
            ]
        )
        if _as_bool(validate_with_bagit, default=False):
            argv.append("--validate-with-bagit-python")

    elif command == "dedup-plan":
        manifest_jsonl = _path_arg(
            args,
            "manifest_jsonl",
            "manifestJsonl",
            default=rights_manifest_default,
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        out_ledger = _path_arg(
            args,
            "out_ledger",
            "outLedger",
            default=str(Path(output_dir) / "dedup_ledger.csv"),
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        out_summary = _path_arg(
            args,
            "out_summary",
            "outSummary",
            default=str(Path(output_dir) / "dedup_summary.json"),
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        approver = str(_pick(args, "approver", default="UNSPECIFIED") or "UNSPECIFIED").strip()

        argv.extend(
            [
                "--manifest-jsonl",
                manifest_jsonl,
                "--out-ledger",
                out_ledger,
                "--out-summary",
                out_summary,
                "--approver",
                approver or "UNSPECIFIED",
            ]
        )

    elif command == "mets-export":
        manifest_jsonl = _path_arg(
            args,
            "manifest_jsonl",
            "manifestJsonl",
            default=rights_manifest_default,
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        out_xml = _path_arg(
            args,
            "out_xml",
            "outXml",
            default=str(Path(output_dir) / "mets_export.xml"),
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        out_summary = _path_arg(
            args,
            "out_summary",
            "outSummary",
            default=str(Path(output_dir) / "mets_summary.json"),
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        href_prefix = str(
            _pick(
                args,
                "href_prefix",
                "hrefPrefix",
                default="data",
            )
            or "data"
        ).strip()

        argv.extend(
            [
                "--manifest-jsonl",
                manifest_jsonl,
                "--out-xml",
                out_xml,
                "--out-summary",
                out_summary,
                "--href-prefix",
                href_prefix or "data",
            ]
        )

    elif command == "prov-export":
        manifest_jsonl = _path_arg(
            args,
            "manifest_jsonl",
            "manifestJsonl",
            default=rights_manifest_default,
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        out_prov_jsonld = _path_arg(
            args,
            "out_prov_jsonld",
            "outProvJsonld",
            default=str(Path(output_dir) / "prov.jsonld"),
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        out_summary = _path_arg(
            args,
            "out_summary",
            "outSummary",
            default=str(Path(output_dir) / "prov_summary.json"),
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        datetime_field = str(
            _pick(
                args,
                "datetime_field",
                "datetimeField",
                default="modified_utc",
            )
            or "modified_utc"
        ).strip()

        argv.extend(
            [
                "--manifest-jsonl",
                manifest_jsonl,
                "--out-prov-jsonld",
                out_prov_jsonld,
                "--out-summary",
                out_summary,
                "--datetime-field",
                datetime_field or "modified_utc",
            ]
        )

    elif command == "stac-export":
        manifest_jsonl = _path_arg(
            args,
            "manifest_jsonl",
            "manifestJsonl",
            default=rights_manifest_default,
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        out_prov_jsonld = _path_arg(
            args,
            "out_prov_jsonld",
            "outProvJsonld",
            default=str(Path(output_dir) / "prov.jsonld"),
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        out_stac_catalog = _path_arg(
            args,
            "out_stac_catalog",
            "outStacCatalog",
            default=str(Path(output_dir) / "catalog.json"),
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        out_stac_items_dir = _path_arg(
            args,
            "out_stac_items_dir",
            "outStacItemsDir",
            default=str(Path(output_dir) / "stac_items"),
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        out_summary = _path_arg(
            args,
            "out_summary",
            "outSummary",
            default=str(Path(output_dir) / "stac_summary.json"),
            allowed_roots=ALLOWED_OUTPUT_ROOTS,
        )
        datetime_field = str(
            _pick(
                args,
                "datetime_field",
                "datetimeField",
                default="modified_utc",
            )
            or "modified_utc"
        ).strip()
        require_stac = _pick(
            args,
            "require_stac",
            "requireStac",
        )

        argv.extend(
            [
                "--manifest-jsonl",
                manifest_jsonl,
                "--out-prov-jsonld",
                out_prov_jsonld,
                "--out-stac-catalog",
                out_stac_catalog,
                "--out-stac-items-dir",
                out_stac_items_dir,
                "--out-summary",
                out_summary,
                "--datetime-field",
                datetime_field or "modified_utc",
            ]
        )
        if require_stac is not None:
            flag = "--require-stac" if _as_bool(require_stac, default=False) else "--no-require-stac"
            argv.append(flag)

    return argv


def _argv_from_request(payload: Dict[str, Any]) -> List[str]:
    """
    Build argv securely (no shell).
    Input validation: allowlist pipeline/backend/quality, require paths.
    """
    pipeline = payload.get("pipeline")
    args = payload.get("args")
    if not isinstance(args, dict):
        args = {}

    if pipeline not in ALLOWED_PIPELINES:
        raise ValueError("Unsupported pipeline")

    input_dir_raw = str(
        _pick(args, "input_dir", "inputDir", default=""),
    ).strip()
    output_dir_raw = str(
        _pick(args, "output_dir", "outputDir", default=""),
    ).strip()
    if not input_dir_raw or not output_dir_raw:
        raise ValueError("input_dir and output_dir are required")
    input_dir = _validate_path_against_roots(
        input_dir_raw,
        ALLOWED_INPUT_ROOTS,
    )
    output_dir = _validate_path_against_roots(
        output_dir_raw,
        ALLOWED_OUTPUT_ROOTS,
    )

    def onoff(b: Any) -> str:
        return "on" if _as_bool(b) else "off"

    argv = [
        pipeline,
        "--input-dir",
        input_dir,
        "--output-dir",
        output_dir,
    ]

    if _as_bool(_pick(args, "overwrite", default=False)):
        argv.append("--overwrite")

    # Pipeline-specific argument building
    if pipeline == "lux-depth-v3":
        quality = (
            str(
                _pick(
                    args,
                    "quality_tier",
                    "qualityTier",
                    default="standard",
                )
                or "standard"
            )
            .strip()
            .lower()
        )
        backend = _canonical_depth_backend(
            _pick(
                args,
                "depth_backend",
                "depthBackend",
                default="da3",
            )
        )
        preset = str(_pick(args, "preset", default="premium") or "premium").strip() or "premium"
        depth_device_raw = _pick(args, "depth_device", "depthDevice")
        depth_device = str(depth_device_raw).strip() if depth_device_raw is not None else ""
        save_float_depth = _pick(
            args,
            "save_float_depth",
            "saveFloatDepth",
            default=False,
        )
        segmentation_backend = (
            str(
                _pick(
                    args,
                    "segmentation_backend",
                    "segmentationBackend",
                    default="stub",
                )
                or "stub"
            )
            .strip()
            .lower()
        )
        sam2_model_size = (
            str(
                _pick(
                    args,
                    "sam2_model_size",
                    "sam2ModelSize",
                    default="base",
                )
                or "base"
            )
            .strip()
            .lower()
        )
        sam2_checkpoint_path_raw = _pick(
            args,
            "sam2_checkpoint_path",
            "sam2CheckpointPath",
        )
        enable_segmentation = _pick(
            args,
            "enable_segmentation",
            "enableSegmentation",
            default=False,
        )
        strict_segmentation = _pick(
            args,
            "strict_segmentation",
            "strictSegmentation",
            default=False,
        )
        enable_reconstruction = _pick(
            args,
            "enable_reconstruction",
            "enableReconstruction",
            default=False,
        )
        grouping_mode = (
            str(
                _pick(
                    args,
                    "grouping_mode",
                    "groupingMode",
                    default="single",
                )
                or "single"
            )
            .strip()
            .lower()
        )
        cameras_sidecar_path_raw = _pick(
            args,
            "cameras_sidecar_path",
            "camerasSidecarPath",
        )
        reconstruction_iterations_raw = _pick(
            args,
            "reconstruction_iterations",
            "reconstructionIterations",
        )
        reconstruction_tier = _pick(
            args,
            "reconstruction_tier",
            "reconstructionTier",
        )
        emit_scene_debug_bundle = _pick(
            args,
            "emit_scene_debug_bundle",
            "emitSceneDebugBundle",
            default=False,
        )
        raw_ingest_mode = _pick(args, "raw_ingest_mode", "rawIngestMode")
        raw_wb_mode = _pick(args, "raw_wb_mode", "rawWbMode")
        raw_demosaic = _pick(args, "raw_demosaic", "rawDemosaic")
        max_workers_raw = _pick(args, "max_workers", "maxWorkers")
        max_gpu_workers_raw = _pick(args, "max_gpu_workers", "maxGpuWorkers")
        log_level_raw = _pick(args, "log_level", "logLevel")

        if quality not in ALLOWED_QUALITY:
            raise ValueError("Invalid quality_tier")
        if backend not in ALLOWED_BACKENDS:
            raise ValueError("Invalid depth_backend")
        if segmentation_backend not in ALLOWED_SEGMENTATION_BACKENDS:
            raise ValueError("Invalid segmentation_backend")
        if segmentation_backend == "sam2" and sam2_model_size not in ALLOWED_SAM2_MODEL_SIZES:
            raise ValueError("Invalid sam2_model_size")
        if grouping_mode not in ALLOWED_GROUPING_MODES:
            raise ValueError("Invalid grouping_mode")

        reconstruction_tier_value = ""
        if reconstruction_tier is not None and str(reconstruction_tier).strip():
            reconstruction_tier_value = str(reconstruction_tier).strip().lower()
            if reconstruction_tier_value not in ALLOWED_RECONSTRUCTION_TIERS:
                raise ValueError("Invalid reconstruction_tier")

        raw_ingest_mode_value = ""
        if raw_ingest_mode is not None and str(raw_ingest_mode).strip():
            raw_ingest_mode_value = str(raw_ingest_mode).strip().lower()
            if raw_ingest_mode_value not in ALLOWED_RAW_INGEST_MODES:
                raise ValueError("Invalid raw_ingest_mode")

        raw_wb_mode_value = ""
        if raw_wb_mode is not None and str(raw_wb_mode).strip():
            raw_wb_mode_value = str(raw_wb_mode).strip().lower()
            if raw_wb_mode_value not in ALLOWED_RAW_WB_MODES:
                raise ValueError("Invalid raw_wb_mode")

        raw_demosaic_value = ""
        if raw_demosaic is not None and str(raw_demosaic).strip():
            raw_demosaic_value = str(raw_demosaic).strip().upper()
            if raw_demosaic_value not in ALLOWED_RAW_DEMOSAIC:
                raise ValueError("Invalid raw_demosaic")

        log_level_value = ""
        if log_level_raw is not None and str(log_level_raw).strip():
            log_level_value = str(log_level_raw).strip().upper()
            if log_level_value not in ALLOWED_LOG_LEVELS:
                raise ValueError("Invalid log_level")

        def _parse_optional_positive_int(
            value: Any,
            field_name: str,
        ) -> Optional[int]:
            if value is None or (isinstance(value, str) and not value.strip()):
                return None
            try:
                parsed = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid {field_name}") from exc
            if parsed < 1:
                raise ValueError(f"Invalid {field_name}")
            return parsed

        reconstruction_iterations = _parse_optional_positive_int(
            reconstruction_iterations_raw,
            "reconstruction_iterations",
        )
        max_workers = _parse_optional_positive_int(
            max_workers_raw,
            "max_workers",
        )
        max_gpu_workers = _parse_optional_positive_int(
            max_gpu_workers_raw,
            "max_gpu_workers",
        )

        sam2_checkpoint_path = ""
        if sam2_checkpoint_path_raw is not None and str(sam2_checkpoint_path_raw).strip():
            sam2_checkpoint_path = _validate_path_against_roots(
                str(sam2_checkpoint_path_raw),
                ALLOWED_INPUT_ROOTS,
            )

        cameras_sidecar_path = ""
        if cameras_sidecar_path_raw is not None and str(cameras_sidecar_path_raw).strip():
            cameras_sidecar_path = _validate_path_against_roots(
                str(cameras_sidecar_path_raw),
                ALLOWED_INPUT_ROOTS,
            )

        argv.extend(
            [
                "--preset",
                preset,
                "--quality-tier",
                quality,
                "--depth-backend",
                backend,
                "--enable-segmentation",
                onoff(enable_segmentation),
                "--segmentation-backend",
                segmentation_backend,
                "--materials-v3",
                onoff(
                    _pick(
                        args,
                        "materials_v3",
                        "materials",
                        default=False,
                    )
                ),
                "--pbr",
                onoff(_pick(args, "pbr", default=False)),
                "--save-float-depth",
                onoff(save_float_depth),
                "--cache-depth",
                onoff(
                    _pick(
                        args,
                        "cache_depth",
                        "cacheDepth",
                        default=False,
                    )
                ),
            ]
        )
        if segmentation_backend == "sam2":
            argv.extend(["--sam2-model-size", sam2_model_size])
        if segmentation_backend == "sam2" and sam2_checkpoint_path:
            argv.extend(["--sam2-checkpoint-path", sam2_checkpoint_path])
        if _as_bool(strict_segmentation, default=False):
            argv.append("--strict-segmentation")

        if depth_device:
            argv.extend(["--depth-device", str(depth_device)])

        argv.extend(
            [
                "--emit-master16",
                onoff(
                    _pick(
                        args,
                        "emit_master16",
                        "emitMaster16",
                        default=True,
                    )
                ),
                "--emit-upscaled16",
                onoff(
                    _pick(
                        args,
                        "emit_upscaled16",
                        "emitUpscaled16",
                        default=True,
                    )
                ),
                "--emit-marketing",
                onoff(
                    _pick(
                        args,
                        "emit_marketing",
                        "emitMarketing",
                        default=False,
                    )
                ),
                "--emit-report",
                onoff(
                    _pick(
                        args,
                        "emit_report",
                        "emitReport",
                        default=True,
                    )
                ),
                "--emit-run-card",
                onoff(
                    _pick(
                        args,
                        "emit_run_card",
                        "emitRunCard",
                        default=True,
                    )
                ),
            ]
        )

        enable_v2_value = _pick(args, "enable_v2", "enableV2")
        enable_v2_specified = enable_v2_value is not None
        enable_v2 = _as_bool(enable_v2_value, default=True)
        if enable_v2_specified:
            argv.extend(["--enable-v2", onoff(enable_v2)])
        if enable_v2:
            v2_preset = _pick(args, "v2_preset", "v2Preset")
            if v2_preset:
                argv.extend(["--v2-preset", str(v2_preset)])

        if _as_bool(
            _pick(
                args,
                "non_commercial_ok",
                "nonCommercialOk",
                default=False,
            )
        ):
            argv.extend(["--non-commercial-ok", "true"])
        if _as_bool(
            _pick(
                args,
                "accept_apple_depth_pro_research_license",
                "acceptAppleDepthProResearchLicense",
                default=False,
            )
        ):
            argv.extend(
                [
                    "--accept-apple-depth-pro-research-license",
                    "true",
                ]
            )
        if _as_bool(
            _pick(
                args,
                "accept_research_tools_license",
                "acceptResearchToolsLicense",
                default=False,
            )
        ):
            argv.extend(["--accept-research-tools-license", "true"])

        argv.extend(
            [
                "--enable-reconstruction",
                onoff(enable_reconstruction),
            ]
        )
        argv.extend(["--grouping-mode", grouping_mode])
        if cameras_sidecar_path:
            argv.extend(["--cameras-sidecar-path", cameras_sidecar_path])
        if reconstruction_iterations is not None:
            argv.extend(
                [
                    "--reconstruction-iterations",
                    str(reconstruction_iterations),
                ]
            )
        if reconstruction_tier_value:
            argv.extend(
                [
                    "--reconstruction-tier",
                    reconstruction_tier_value,
                ]
            )
        argv.extend(
            [
                "--emit-scene-debug-bundle",
                onoff(emit_scene_debug_bundle),
            ]
        )

        if _as_bool(
            _pick(
                args,
                "force_depth",
                "forceDepth",
                default=False,
            )
        ):
            argv.append("--force-depth")
        if _as_bool(
            _pick(
                args,
                "strict_inputs",
                "strictInputs",
                default=False,
            )
        ):
            argv.append("--strict-inputs")
        if _as_bool(
            _pick(
                args,
                "verify_images",
                "verifyImages",
                default=False,
            )
        ):
            argv.append("--verify-images")
        if _as_bool(
            _pick(
                args,
                "allow_semantic_fallback",
                "allowSemanticFallback",
                default=False,
            )
        ):
            argv.append("--allow-semantic-fallback")

        if raw_ingest_mode_value:
            argv.extend(["--raw-ingest-mode", raw_ingest_mode_value])
        if raw_wb_mode_value:
            argv.extend(["--raw-wb-mode", raw_wb_mode_value])
        if raw_demosaic_value:
            argv.extend(["--raw-demosaic", raw_demosaic_value])

        if max_workers is not None:
            argv.extend(["--max-workers", str(max_workers)])
        if max_gpu_workers is not None:
            argv.extend(["--max-gpu-workers", str(max_gpu_workers)])

        verbose_enabled = _as_bool(_pick(args, "verbose", default=False))
        quiet_enabled = _as_bool(_pick(args, "quiet", default=False))
        if verbose_enabled and quiet_enabled:
            print(
                "Error: --verbose and --quiet cannot both be enabled.\n"
                "Use --verbose for detailed output"
                " or --quiet for minimal output.",
                file=sys.stderr,
            )
            raise ValueError("verbose and quiet are mutually exclusive")
        if verbose_enabled:
            argv.append("--verbose")
        if quiet_enabled:
            argv.append("--quiet")
        if log_level_value:
            argv.extend(["--log-level", log_level_value])
    elif pipeline in ARCHIVE_GATE_PIPELINES:
        argv = _archive_gate_argv(
            str(pipeline),
            args,
            input_dir,
            output_dir,
        )

    return argv


app = FastAPI(
    title="Transformation Portal Orchestrator",
    version="0.3.0",
    docs_url="/docs" if ENABLE_API_DOCS else None,
    redoc_url="/redoc" if ENABLE_API_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_API_DOCS else None,
)
app.state.cleanup_task = None

if ENABLE_TRUSTED_HOSTS:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=TRUSTED_HOSTS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "Accept",
        "Authorization",
        API_KEY_HEADER,
    ],
)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    if _is_api_v1_path(request.url.path):
        detail = exc.detail
        message = detail if isinstance(detail, str) and detail.strip() else "request failed"
        details = {"path": request.url.path}
        return _error_response(
            exc.status_code,
            code=_http_status_error_code(exc.status_code),
            message=message,
            details=details,
            headers=exc.headers,
        )

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def request_validation_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    if _is_api_v1_path(request.url.path):
        return _error_response(
            400,
            code="INVALID_ARGUMENT",
            message="request validation failed",
            details={
                "path": request.url.path,
                "errors": exc.errors(),
            },
        )
    return await fastapi_request_validation_exception_handler(
        request,
        exc,
    )


@app.on_event("startup")
async def startup() -> None:
    if _job_api_key_enforced() and not API_KEY_SECRET:
        LOGGER.warning(
            "TP_ENFORCE_JOB_API_KEY is enabled but"
            " TP_API_KEY is unset; /v1/jobs endpoints"
            " will return AUTH_CONFIGURATION_ERROR."
        )
    cleanup_task = getattr(app.state, "cleanup_task", None)
    if cleanup_task is None or cleanup_task.done():
        app.state.cleanup_task = asyncio.create_task(_cleanup_loop())


@app.on_event("shutdown")
async def shutdown() -> None:
    cleanup_task = getattr(app.state, "cleanup_task", None)
    if cleanup_task is not None:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        app.state.cleanup_task = None


@app.middleware("http")
async def security_layer(
    request: Request,
    call_next: Callable[[Request], Any],
) -> Response:
    maybe_error = _enforce_content_length_limit(request)
    if maybe_error is not None:
        return maybe_error

    _install_stream_body_limit(request)

    if _is_protected_job_endpoint(request.url.path) and _job_api_key_enforced():
        if not API_KEY_SECRET:
            return _error_response(
                503,
                code="AUTH_CONFIGURATION_ERROR",
                message=("job endpoint authentication is" " enforced but TP_API_KEY is not" " configured"),
                details={"path": request.url.path, "env": "TP_API_KEY"},
            )
        if not _has_valid_api_key(request):
            return _error_response(
                401,
                code="UNAUTHORIZED",
                message="invalid or missing API key",
                details={"path": request.url.path},
            )

    client_ip = _extract_client_ip(request)
    if _is_rate_limited(client_ip, _now()):
        return _error_response(
            429,
            code="RATE_LIMITED",
            message="rate limit exceeded",
            details={"client_ip": client_ip},
        )

    response = await call_next(request)
    for name, value in SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    if request.url.path.startswith("/v1/"):
        response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.get("/")
async def serve_ui() -> Response:
    """Serves the single-file UI."""
    if not PORTAL_HTML.exists():
        raise HTTPException(
            status_code=500,
            detail="portal.html is missing",
        )
    return FileResponse(str(PORTAL_HTML))


@app.get("/portal/bootstrap")
async def portal_bootstrap() -> JSONResponse:
    """Expose standalone portal auth mode for direct backend debugging."""
    return JSONResponse(
        {
            "authMode": "direct_debug",
            "csrfToken": None,
            "actor": None,
            "features": {
                "apiKeyInput": True,
                "directDebug": True,
            },
        },
        headers={"Cache-Control": "no-store"},
    )


@app.get("/ready")
async def ready() -> Dict[str, Any]:
    response: Dict[str, Any] = {
        "ok": True,
        "time": _now(),
        "version": "0.3.0",
    }
    if READY_VERBOSE:
        from shutil import which

        response["cli"] = {
            "lux-depth-v3": bool(which("lux-depth-v3")),
            "archive-governance": ARCHIVE_GOVERNANCE_SCRIPT.is_file(),
            "python": sys.version.split()[0],
        }
        response["jobs"] = {
            "active": _active_job_count(),
            "total": len(JOBS),
        }
        response["security"] = {
            "api_key_enforced_for_jobs": _job_api_key_enforced(),
            "rate_limit_per_minute": RATE_LIMIT_PER_MINUTE,
            "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
            "max_request_bytes": MAX_REQUEST_BYTES,
            "trusted_hosts_enabled": ENABLE_TRUSTED_HOSTS,
            "trust_x_forwarded_for": TRUST_X_FORWARDED_FOR,
            "trusted_proxy_ips_count": len(TRUSTED_PROXY_IPS),
            "allowed_input_roots_count": len(ALLOWED_INPUT_ROOTS),
            "allowed_output_roots_count": len(ALLOWED_OUTPUT_ROOTS),
            "allow_sse_query_api_key": ALLOW_SSE_QUERY_API_KEY,
            "docs_enabled": ENABLE_API_DOCS,
        }
    return response


@app.get("/v1/presets")
async def list_presets(pipeline: Optional[str] = None) -> JSONResponse:
    if pipeline is not None and pipeline not in PRESET_CATALOG:
        return _error_response(
            400,
            code="INVALID_ARGUMENT",
            message=f"Unsupported pipeline '{pipeline}'",
            details={
                "field": "pipeline",
                "allowed": sorted(PRESET_CATALOG.keys()),
            },
        )

    data: Dict[str, Any]
    if pipeline is None:
        data = {
            "pipelines": [
                {
                    "pipeline": pipeline_name,
                    "presets": presets,
                }
                for pipeline_name, presets in PRESET_CATALOG.items()
            ],
        }
    else:
        data = {
            "pipeline": pipeline,
            "presets": PRESET_CATALOG[pipeline],
        }

    return JSONResponse(
        _api_envelope(
            "tp.orchestrator.presets.v1",
            success=True,
            data=data,
            error=None,
        )
    )


@app.post("/v1/jobs")
async def create_job(payload: Dict[str, Any]) -> JSONResponse:
    try:
        argv = _argv_from_request(payload)
    except ValueError as exc:
        reason_code = VALIDATION_REASON_CODES.get(
            str(exc),
            "invalid_request",
        )
        return _error_response(
            400,
            code="INVALID_ARGUMENT",
            message="invalid job request",
            details={"field": "payload", "reason": reason_code},
        )

    async with JOB_ADMISSION_LOCK:
        _cleanup_expired_jobs(_now())
        active_jobs = _active_job_count()
        if active_jobs >= MAX_CONCURRENT_JOBS:
            return _error_response(
                429,
                code="RATE_LIMITED",
                message="too many active jobs; try again later",
                details={
                    "active_jobs": active_jobs,
                    "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
                },
            )
        jid = "job_" + uuid.uuid4().hex[:8]
        job = Job(id=jid, created_at=_now(), request=payload)
        JOBS[jid] = job
        EVENT_SUBSCRIBERS[jid] = {}

    asyncio.create_task(_run_job(job, argv))

    return JSONResponse(
        _api_envelope(
            "tp.orchestrator.job.v1",
            success=True,
            data={
                "id": jid,
                "state": job.state,
                "events_url": f"/v1/jobs/{jid}/events",
            },
            error=None,
        )
    )


@app.get("/v1/jobs")
async def list_jobs(limit: int = JOB_LIST_LIMIT) -> JSONResponse:
    _cleanup_expired_jobs(_now())
    bounded_limit = max(1, min(limit, JOB_LIST_LIMIT))
    jobs_sorted = sorted(
        JOBS.values(),
        key=lambda item: item.created_at,
        reverse=True,
    )
    serialized = [_serialize_job(job) for job in jobs_sorted[:bounded_limit]]

    return JSONResponse(
        _api_envelope(
            "tp.orchestrator.jobs.v1",
            success=True,
            data={
                "jobs": serialized,
                "total": len(JOBS),
                "returned": len(serialized),
            },
            error=None,
        )
    )


@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str) -> JSONResponse:
    _cleanup_expired_jobs(_now())
    job = JOBS.get(job_id)
    if not job:
        return _error_response(
            404,
            code="NOT_FOUND",
            message="job not found",
            details={"job_id": job_id},
        )
    return JSONResponse(
        _api_envelope(
            "tp.orchestrator.job_status.v1",
            success=True,
            data=_serialize_job(job),
            error=None,
        )
    )


@app.get("/v1/jobs/{job_id}/artifacts/{artifact_path:path}")
async def get_job_artifact(job_id: str, artifact_path: str) -> Response:
    _cleanup_expired_jobs(_now())
    job = JOBS.get(job_id)
    if not job:
        return _error_response(
            404,
            code="NOT_FOUND",
            message="job not found",
            details={"job_id": job_id},
        )

    try:
        _, resolved_artifact, relative_path = _resolve_job_artifact_path(job, artifact_path)
    except ValueError as exc:
        reason = str(exc)
        return _error_response(
            400,
            code="INVALID_ARGUMENT",
            message="invalid artifact path",
            details={"job_id": job_id, "reason": reason},
        )
    except FileNotFoundError:
        return _error_response(
            404,
            code="NOT_FOUND",
            message="artifact not found",
            details={"job_id": job_id, "path": str(artifact_path or "").strip()},
        )

    if not resolved_artifact.exists() or not resolved_artifact.is_file():
        return _error_response(
            404,
            code="NOT_FOUND",
            message="artifact not found",
            details={"job_id": job_id, "path": relative_path},
        )

    return FileResponse(
        resolved_artifact,
        media_type=_artifact_content_type(resolved_artifact),
        filename=resolved_artifact.name,
        headers={"Cache-Control": "no-store"},
    )


@app.post("/v1/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> JSONResponse:
    job = JOBS.get(job_id)
    if not job:
        return _error_response(
            404,
            code="NOT_FOUND",
            message="job not found",
            details={"job_id": job_id},
        )
    await _request_cancel(job)
    return JSONResponse(
        _api_envelope(
            "tp.orchestrator.job.v1",
            success=True,
            data={"id": job_id, "state": job.state},
            error=None,
        )
    )


@app.get("/v1/jobs/{job_id}/events")
async def job_events(
    request: Request,
    job_id: str,
) -> Response:
    job = JOBS.get(job_id)
    if not job:
        return _error_response(
            404,
            code="NOT_FOUND",
            message="job not found",
            details={"job_id": job_id},
        )

    subscribers = EVENT_SUBSCRIBERS.setdefault(job_id, {})
    subscriber_id = uuid.uuid4().hex
    q: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(
        maxsize=EVENT_QUEUE_MAXSIZE,
    )
    subscribers[subscriber_id] = q

    async def gen() -> AsyncGenerator[str, None]:
        try:
            yield _sse(
                "state",
                {
                    "id": job_id,
                    "state": job.state,
                    "progress": job.progress,
                },
            )
            if job.done_published_at is not None:
                # The 'done' event has already been published to all subscribers.
                # For a late-connecting client, we can safely drain any queued
                # events (artifact, done, etc.) and synthesize if needed.
                while True:
                    try:
                        ev = q.get_nowait()
                        yield _sse(ev["event"], ev["data"])
                        if ev["event"] == "done":
                            return
                    except asyncio.QueueEmpty:
                        break
                # No 'done' event was queued; generate a synthetic one from job state.
                yield _sse(
                    "done",
                    {
                        "id": job.id,
                        "state": job.state,
                        "exit_code": job.exit_code,
                        "error": job.error,
                        "artifacts": job.artifacts,
                    },
                )
                return
            elif job.finished_at is not None:
                # Job processing finished (state is terminal) but done_published_at
                # is not set yet. This means artifact indexing and event publication
                # are still in progress. Wait for real events rather than synthesizing.
                pass  # Fall through to the normal event loop below

            last_beat = _now()
            while True:
                if await request.is_disconnected():
                    break

                # Heartbeat comment line keeps intermediaries alive.
                now = _now()
                if now - last_beat > HEARTBEAT_SECONDS:
                    yield ": heartbeat\n\n"
                    last_beat = now

                try:
                    ev = await asyncio.wait_for(q.get(), timeout=1.0)
                    yield _sse(ev["event"], ev["data"])
                    if ev["event"] == "done":
                        break
                except asyncio.TimeoutError:
                    continue
        finally:
            subscribers_for_job = EVENT_SUBSCRIBERS.get(job_id)
            if subscribers_for_job is not None:
                subscribers_for_job.pop(subscriber_id, None)
                if not subscribers_for_job and JOBS.get(job_id) and JOBS[job_id].finished_at is not None:
                    EVENT_SUBSCRIBERS.pop(job_id, None)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


async def _run_job(job: Job, argv: List[str]) -> None:
    job.state = "running"
    job.started_at = _now()
    await _publish_event(
        job.id,
        "state",
        {"id": job.id, "state": job.state},
    )

    try:
        # NO SHELL EXECUTION.
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=_sanitized_child_env(),
        )
        job.proc = proc

        if proc.stdout is None:
            raise RuntimeError("failed to capture subprocess stdout")

        while True:
            if job.cancel_requested and proc.returncode is None and (job.terminate_task is None or job.terminate_task.done()):
                job.terminate_task = asyncio.create_task(
                    _terminate_process(proc),
                )

            raw_line = await proc.stdout.readline()
            if not raw_line:
                break

            line = raw_line.decode(
                "utf-8",
                errors="replace",
            ).rstrip("\n")
            job.add_log(line)
            await _publish_event(
                job.id,
                "log",
                {"id": job.id, "line": line},
            )

            pct = _extract_progress_percent(line)
            if pct is not None and pct != job.progress:
                job.progress = pct
                await _publish_event(
                    job.id,
                    "progress",
                    {"id": job.id, "progress": job.progress},
                )

        rc = await proc.wait()
        job.exit_code = int(rc)
        if job.cancel_requested:
            job.state = "canceled"
        else:
            job.state = "succeeded" if rc == 0 else "failed"
            if rc != 0:
                job.error = _error_obj(
                    "RUNNER_EXIT_NONZERO",
                    f"runner exited with code {rc}",
                    {"exit_code": int(rc)},
                )

    except FileNotFoundError:
        job.state = "failed"
        job.exit_code = 127
        job.error = _error_obj(
            "RUNNER_NOT_FOUND",
            f"Command '{argv[0]}' not found in PATH.",
            {"command": argv[0]},
        )
        msg = f"runner_error: {job.error['message']}"
        job.add_log(msg)
        await _publish_event(
            job.id,
            "log",
            {"id": job.id, "line": msg},
        )
    except Exception as exc:
        LOGGER.exception(
            "Unhandled runner exception for job %s",
            job.id,
        )
        job.state = "failed"
        job.exit_code = 1
        job.error = _error_obj(
            "RUNNER_ERROR",
            "unexpected runner failure",
            {"exception_type": type(exc).__name__},
        )
        msg = "runner_error: unexpected runner failure"
        job.add_log(msg)
        await _publish_event(
            job.id,
            "log",
            {"id": job.id, "line": msg},
        )
    finally:
        if job.terminate_task is not None:
            try:
                await job.terminate_task
            except Exception:
                pass

        # Index artifacts and publish terminal events BEFORE setting finished_at.
        # This ensures late-connecting SSE clients can deterministically check
        # done_published_at to know if they need to wait for real events or can
        # safely synthesize a 'done' from job state.
        indexed_artifacts = _index_job_artifacts(job)
        for artifact in indexed_artifacts:
            await _publish_event(
                job.id,
                "artifact",
                {"id": job.id, **artifact},
            )

        await _publish_event(
            job.id,
            "done",
            {
                "id": job.id,
                "state": job.state,
                "exit_code": job.exit_code,
                "error": job.error,
                "artifacts": job.artifacts,
            },
        )
        # Mark timestamps AFTER all events are published, so SSE endpoint knows
        # it's safe to synthesize 'done' if done_published_at is set.
        job.done_published_at = _now()
        job.finished_at = job.done_published_at
        _cleanup_expired_jobs(_now())
