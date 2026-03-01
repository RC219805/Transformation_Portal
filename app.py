from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import re
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Deque, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import StreamingResponse

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
LOG_TAIL_LIMIT = 2000
STATUS_LOG_LIMIT = 250
EVENT_QUEUE_MAXSIZE = 512
HEARTBEAT_SECONDS = 15
JOB_RETENTION_SECONDS = _env_int("TP_JOB_RETENTION_SECONDS", 3600, minimum=1)
CLEANUP_INTERVAL_SECONDS = _env_int("TP_CLEANUP_INTERVAL_SECONDS", 60, minimum=1)
CANCEL_GRACE_SECONDS = _env_float("TP_CANCEL_GRACE_SECONDS", 5.0, minimum=0.1)
JOB_LIST_LIMIT = _env_int("TP_JOB_LIST_LIMIT", 200, minimum=1)
MAX_INDEXED_ARTIFACTS = _env_int("TP_MAX_INDEXED_ARTIFACTS", 200, minimum=1)
PROGRESS_RE = re.compile(r"progress=(\d{1,3})%")
DEFAULT_ALLOWED_ORIGINS = ["http://localhost", "http://localhost:3000", "http://127.0.0.1:8000"]
ALLOWED_ORIGINS = _env_csv("TP_ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS)
TRUSTED_HOSTS = _env_csv("TP_TRUSTED_HOSTS", ["localhost", "127.0.0.1", "::1", "testserver"])
ENABLE_TRUSTED_HOSTS = _env_bool("TP_ENABLE_TRUSTED_HOSTS", True)
API_KEY_HEADER = os.getenv("TP_API_KEY_HEADER", "x-api-key").strip().lower() or "x-api-key"
API_KEY_SECRET = os.getenv("TP_API_KEY", "").strip()
TRUST_X_FORWARDED_FOR = _env_bool("TP_TRUST_X_FORWARDED_FOR", False)
TRUSTED_PROXY_IPS = set(_env_csv("TP_TRUSTED_PROXY_IPS", []))
MAX_REQUEST_BYTES = _env_int("TP_MAX_REQUEST_BYTES", 1024 * 1024, minimum=1024)
RATE_LIMIT_PER_MINUTE = _env_int("TP_RATE_LIMIT_PER_MINUTE", 0, minimum=0)
RATE_LIMIT_WINDOW_SECONDS = 60.0
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
        },
        {
            "name": "default",
            "label": "default (Canary)",
            "stability": "canary",
            "description": "Canary preset for iterative validation",
            "is_research": False,
        },
        {
            "name": "depth-anything-v3.1-research-m4",
            "label": "v3.1-m4 (Experimental)",
            "stability": "experimental",
            "description": "Research-only preset requiring non-commercial acknowledgments",
            "is_research": True,
        },
    ],
    "archive-gate-a": [
        {
            "name": "default",
            "label": "default (Stable)",
            "stability": "stable",
            "description": "Manifest and provenance assembly",
            "is_research": False,
        }
    ],
    "archive-gate-b": [
        {
            "name": "default",
            "label": "default (Stable)",
            "stability": "stable",
            "description": "BagIt packaging and validation workflow",
            "is_research": False,
        }
    ],
    "archive-gate-c": [
        {
            "name": "default",
            "label": "default (Stable)",
            "stability": "stable",
            "description": "METS/PROV/STAC export workflow",
            "is_research": False,
        }
    ],
}


@dataclass
class Job:
    id: str
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
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

# Gate pipelines integrated directly
ALLOWED_PIPELINES = {"lux-depth-v3", "archive-gate-a", "archive-gate-b", "archive-gate-c"}
ALLOWED_QUALITY = {"standard", "premium", "apex"}
ALLOWED_BACKENDS = {"da3", "depth_pro"}
DEPTH_BACKEND_ALIASES = {
    "depth_anything_v3": "da3",
    "depth-anything-v3": "da3",
}
VALIDATION_REASON_CODES = {
    "Unsupported pipeline": "unsupported_pipeline",
    "input_dir and output_dir are required": "missing_required_paths",
    "Invalid quality_tier": "invalid_quality_tier",
    "Invalid depth_backend": "invalid_depth_backend",
}


def _now() -> float:
    return time.time()


def _sse(event: str, data: Dict[str, Any]) -> str:
    # SSE payload format: event type + JSON data, terminated by double newline
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _error_obj(code: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=_api_envelope(schema, success=False, data=None, error=_error_obj(code, message, details)),
    )


def _cleanup_expired_jobs(now: float) -> None:
    expired = [
        job_id
        for job_id, job in JOBS.items()
        if job.finished_at is not None and now - job.finished_at >= JOB_RETENTION_SECONDS
    ]
    for job_id in expired:
        JOBS.pop(job_id, None)
        EVENT_SUBSCRIBERS.pop(job_id, None)


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
    backend = str(value or "").strip()
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
    if not provided and _is_job_events_endpoint(request.url.path):
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
        return JSONResponse(status_code=400, content={"detail": "invalid Content-Length header"})

    if size > MAX_REQUEST_BYTES:
        return JSONResponse(
            status_code=413,
            content={"detail": f"request body too large (max {MAX_REQUEST_BYTES} bytes)"},
        )
    return None


def _install_stream_body_limit(request: Request) -> None:
    if request.method not in {"POST", "PUT", "PATCH"}:
        return
    if getattr(request.state, "_tp_body_limit_installed", False):
        return

    original_receive = getattr(request, "_receive", None)
    if original_receive is None:
        return

    async def limited_receive() -> Dict[str, Any]:
        message = await original_receive()
        if message.get("type") == "http.request":
            body = message.get("body", b"") or b""
            consumed = getattr(request.state, "_tp_body_bytes_received", 0)
            consumed += len(body)
            request.state._tp_body_bytes_received = consumed
            if consumed > MAX_REQUEST_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"request body too large (max {MAX_REQUEST_BYTES} bytes)",
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


async def _publish_event(job_id: str, event: str, data: Dict[str, Any]) -> None:
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


async def _terminate_process(proc: asyncio.subprocess.Process, grace_seconds: float = CANCEL_GRACE_SECONDS) -> None:
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
    output_dir = str(_pick(args, "output_dir", "outputDir", default="")).strip()
    if not output_dir:
        return None
    return Path(output_dir).expanduser()


def _infer_artifact_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".json", ".yaml", ".yml", ".txt", ".md", ".log", ".csv"}:
        return "metadata"
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".exr"}:
        return "image"
    if suffix in {".zip", ".tar", ".gz", ".tgz", ".bag"}:
        return "archive"
    return "file"


def _index_job_artifacts(job: Job) -> List[Dict[str, Any]]:
    output_dir = _job_output_dir(job)
    if output_dir is None:
        job.artifacts = {"output_dir": None, "items": [], "indexed_count": 0, "truncated": False}
        return []
    if not output_dir.exists() or not output_dir.is_dir():
        job.artifacts = {"output_dir": str(output_dir), "items": [], "indexed_count": 0, "truncated": False}
        return []

    collected: List[tuple[str, Path]] = []
    truncated = False
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            relative_path = str(path.relative_to(output_dir))
        except Exception:
            relative_path = path.name
        collected.append((relative_path, path))
        if len(collected) > MAX_INDEXED_ARTIFACTS:
            truncated = True
            break

    collected.sort(key=lambda item: item[0].lower())
    if truncated:
        collected = collected[:MAX_INDEXED_ARTIFACTS]

    items: List[Dict[str, Any]] = []
    for relative_path, path in collected:
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = None
        items.append(
            {
                "artifact_type": _infer_artifact_type(path),
                # Do not expose absolute server paths in API/SSE payloads.
                "path": relative_path,
                "relative_path": relative_path,
                "size_bytes": size_bytes,
            }
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

    input_dir = str(_pick(args, "input_dir", "inputDir", default="")).strip()
    output_dir = str(_pick(args, "output_dir", "outputDir", default="")).strip()
    if not input_dir or not output_dir:
        raise ValueError("input_dir and output_dir are required")

    def onoff(b: Any) -> str:
        return "on" if _as_bool(b) else "off"

    argv = [pipeline, "--input-dir", input_dir, "--output-dir", output_dir]

    if _as_bool(_pick(args, "overwrite", default=False)):
        argv.append("--overwrite")

    # Pipeline-specific argument building
    if pipeline == "lux-depth-v3":
        quality = _pick(args, "quality_tier", "qualityTier", default="standard")
        backend = _canonical_depth_backend(_pick(args, "depth_backend", "depthBackend", default="da3"))
        preset = _pick(args, "preset", default="premium")
        depth_device = _pick(args, "depth_device", "depthDevice")

        if quality not in ALLOWED_QUALITY:
            raise ValueError("Invalid quality_tier")
        if backend not in ALLOWED_BACKENDS:
            raise ValueError("Invalid depth_backend")

        argv.extend(
            [
                "--preset",
                preset,
                "--quality-tier",
                quality,
                "--depth-backend",
                backend,
                "--materials-v3",
                onoff(_pick(args, "materials_v3", "materials", default=False)),
                "--pbr",
                onoff(_pick(args, "pbr", default=False)),
                "--cache-depth",
                onoff(_pick(args, "cache_depth", "cacheDepth", default=False)),
            ]
        )

        if depth_device:
            argv.extend(["--depth-device", str(depth_device)])

        argv.extend(
            [
                "--emit-master16",
                onoff(_pick(args, "emit_master16", "emitMaster16", default=True)),
                "--emit-upscaled16",
                onoff(_pick(args, "emit_upscaled16", "emitUpscaled16", default=True)),
                "--emit-marketing",
                onoff(_pick(args, "emit_marketing", "emitMarketing", default=False)),
                "--emit-report",
                onoff(_pick(args, "emit_report", "emitReport", default=True)),
                "--emit-run-card",
                onoff(_pick(args, "emit_run_card", "emitRunCard", default=True)),
            ]
        )

        enable_v2 = _as_bool(_pick(args, "enable_v2", "enableV2", default=False))
        if enable_v2:
            argv.extend(["--enable-v2", "on"])
            v2_preset = _pick(args, "v2_preset", "v2Preset")
            if v2_preset:
                argv.extend(["--v2-preset", str(v2_preset)])

        if _as_bool(_pick(args, "non_commercial_ok", "nonCommercialOk", default=False)):
            argv.extend(["--non-commercial-ok", "true"])
        if _as_bool(
            _pick(
                args,
                "accept_apple_depth_pro_research_license",
                "acceptAppleDepthProResearchLicense",
                default=False,
            )
        ):
            argv.extend(["--accept-apple-depth-pro-research-license", "true"])
    elif pipeline in ["archive-gate-a", "archive-gate-b", "archive-gate-c"]:
        if _as_bool(_pick(args, "dedup", default=False)):
            argv.append("--dedup")
            argv.append("on")
        if _as_bool(_pick(args, "sign", default=False)):
            argv.append("--sign")
            argv.append("true")

    return argv


app = FastAPI(title="Transformation Portal Orchestrator", version="0.3.0")
app.state.cleanup_task = None

if ENABLE_TRUSTED_HOSTS:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=TRUSTED_HOSTS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Authorization", API_KEY_HEADER],
)


@app.on_event("startup")
async def startup() -> None:
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
async def security_layer(request: Request, call_next):
    maybe_error = _enforce_content_length_limit(request)
    if maybe_error is not None:
        return maybe_error

    _install_stream_body_limit(request)

    if API_KEY_SECRET and _is_protected_job_endpoint(request.url.path) and not _has_valid_api_key(request):
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
async def serve_ui():
    """Serves the single-file UI."""
    if not PORTAL_HTML.exists():
        raise HTTPException(status_code=500, detail="portal.html is missing")
    return FileResponse(str(PORTAL_HTML))


@app.get("/ready")
async def ready() -> Dict[str, Any]:
    from shutil import which

    return {
        "ok": True,
        "time": _now(),
        "version": "0.3.0",
        "cli": {
            "lux-depth-v3": bool(which("lux-depth-v3")),
            "python": sys.version.split()[0],
        },
        "jobs": {
            "active": sum(1 for job in JOBS.values() if job.state in {"queued", "running"}),
            "total": len(JOBS),
        },
        "security": {
            "api_key_required_for_mutations": bool(API_KEY_SECRET),
            "rate_limit_per_minute": RATE_LIMIT_PER_MINUTE,
            "max_request_bytes": MAX_REQUEST_BYTES,
            "trusted_hosts_enabled": ENABLE_TRUSTED_HOSTS,
            "trust_x_forwarded_for": TRUST_X_FORWARDED_FOR,
            "trusted_proxy_ips_count": len(TRUSTED_PROXY_IPS),
            "api_key_protects_job_reads": bool(API_KEY_SECRET),
        },
    }


@app.get("/v1/presets")
async def list_presets(pipeline: Optional[str] = None) -> JSONResponse:
    if pipeline is not None and pipeline not in PRESET_CATALOG:
        return _error_response(
            400,
            code="INVALID_ARGUMENT",
            message=f"Unsupported pipeline '{pipeline}'",
            details={"field": "pipeline", "allowed": sorted(PRESET_CATALOG.keys())},
        )

    if pipeline is None:
        data = {
            "pipelines": [{"pipeline": pipeline_name, "presets": presets} for pipeline_name, presets in PRESET_CATALOG.items()]
        }
    else:
        data = {"pipeline": pipeline, "presets": PRESET_CATALOG[pipeline]}

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
        reason_code = VALIDATION_REASON_CODES.get(str(exc), "invalid_request")
        return _error_response(
            400,
            code="INVALID_ARGUMENT",
            message="invalid job request",
            details={"field": "payload", "reason": reason_code},
        )

    _cleanup_expired_jobs(_now())
    jid = "job_" + uuid.uuid4().hex[:8]
    job = Job(id=jid, created_at=_now(), request=payload)
    JOBS[jid] = job
    EVENT_SUBSCRIBERS[jid] = {}

    asyncio.create_task(_run_job(job, argv))

    return JSONResponse(
        _api_envelope(
            "tp.orchestrator.job.v1",
            success=True,
            data={"id": jid, "state": job.state, "events_url": f"/v1/jobs/{jid}/events"},
            error=None,
        )
    )


@app.get("/v1/jobs")
async def list_jobs(limit: int = JOB_LIST_LIMIT) -> JSONResponse:
    _cleanup_expired_jobs(_now())
    bounded_limit = max(1, min(limit, JOB_LIST_LIMIT))
    jobs_sorted = sorted(JOBS.values(), key=lambda item: item.created_at, reverse=True)
    serialized = [_serialize_job(job) for job in jobs_sorted[:bounded_limit]]

    return JSONResponse(
        _api_envelope(
            "tp.orchestrator.jobs.v1",
            success=True,
            data={"jobs": serialized, "total": len(JOBS), "returned": len(serialized)},
            error=None,
        )
    )


@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str) -> JSONResponse:
    _cleanup_expired_jobs(_now())
    job = JOBS.get(job_id)
    if not job:
        return _error_response(404, code="NOT_FOUND", message="job not found", details={"job_id": job_id})
    return JSONResponse(
        _api_envelope(
            "tp.orchestrator.job_status.v1",
            success=True,
            data=_serialize_job(job),
            error=None,
        )
    )


@app.post("/v1/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> JSONResponse:
    job = JOBS.get(job_id)
    if not job:
        return _error_response(404, code="NOT_FOUND", message="job not found", details={"job_id": job_id})
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
async def job_events(request: Request, job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return _error_response(404, code="NOT_FOUND", message="job not found", details={"job_id": job_id})

    subscribers = EVENT_SUBSCRIBERS.setdefault(job_id, {})
    subscriber_id = uuid.uuid4().hex
    q: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=EVENT_QUEUE_MAXSIZE)
    subscribers[subscriber_id] = q

    async def gen() -> AsyncGenerator[str, None]:
        try:
            yield _sse("state", {"id": job_id, "state": job.state, "progress": job.progress})
            if job.finished_at is not None:
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
                if not subscribers_for_job and JOBS.get(job_id, None) and JOBS[job_id].finished_at is not None:
                    EVENT_SUBSCRIBERS.pop(job_id, None)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


async def _run_job(job: Job, argv: List[str]) -> None:
    job.state = "running"
    job.started_at = _now()
    await _publish_event(job.id, "state", {"id": job.id, "state": job.state})

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
                job.terminate_task = asyncio.create_task(_terminate_process(proc))

            raw_line = await proc.stdout.readline()
            if not raw_line:
                break

            line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
            job.add_log(line)
            await _publish_event(job.id, "log", {"id": job.id, "line": line})

            pct = _extract_progress_percent(line)
            if pct is not None and pct != job.progress:
                job.progress = pct
                await _publish_event(job.id, "progress", {"id": job.id, "progress": job.progress})

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
        await _publish_event(job.id, "log", {"id": job.id, "line": msg})
    except Exception as exc:
        LOGGER.exception("Unhandled runner exception for job %s", job.id)
        job.state = "failed"
        job.exit_code = 1
        job.error = _error_obj(
            "RUNNER_ERROR",
            "unexpected runner failure",
            {"exception_type": type(exc).__name__},
        )
        msg = "runner_error: unexpected runner failure"
        job.add_log(msg)
        await _publish_event(job.id, "log", {"id": job.id, "line": msg})
    finally:
        if job.terminate_task is not None:
            try:
                await job.terminate_task
            except Exception:
                pass

        job.finished_at = _now()
        indexed_artifacts = _index_job_artifacts(job)
        for artifact in indexed_artifacts:
            await _publish_event(job.id, "artifact", {"id": job.id, **artifact})

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
        _cleanup_expired_jobs(_now())
