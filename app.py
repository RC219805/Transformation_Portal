from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import math
import os
import re
import shlex
import signal
import stat
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from email.parser import BytesParser
from email.policy import default as email_policy
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Deque, Dict, List, Mapping, NamedTuple, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler as fastapi_request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import Response, StreamingResponse

from transformation_portal.api.v1 import (
    ConfigMetadataEnvelope,
    ConfigPreviewEnvelope,
    HealthzResponse,
    JobBriefData,
    JobEnvelope,
    JobsListData,
    JobsListEnvelope,
    JobStatusData,
    JobStatusEnvelope,
    PortalEventEnvelope,
    PortalRumIngestEnvelope,
    PresetsEnvelope,
    ReadinessEnvelope,
    ReadyResponse,
    UploadStagingEnvelope,
)
from transformation_portal.determinism.trace import get_or_create_trace_context
from transformation_portal.ingest.upload_staging import (
    DEFAULT_CAPTURE_METADATA_CONFIG_PATH,
    DEFAULT_CAPTURE_METADATA_SCHEMA_PATH,
    IncomingUpload,
    UploadStagingError,
    cleanup_expired_batches,
    parse_client_manifest_relative_paths,
    stage_upload_batch,
)
from transformation_portal.lux_depth_v3.model_registry import resolve_model_spec, resolve_registry_key, visible_cli_model_specs
from transformation_portal.orchestrator import get_job_repository
from transformation_portal.orchestrator.queue import (
    JobEnqueueRequest,
    QueueBrokerError,
    get_queue_broker,
)
from transformation_portal.orchestrator.recovery import sweep_orphaned_jobs
from transformation_portal.orchestrator.worker import WorkerConfig, run_worker_forever
from transformation_portal.portal import archive_index_preflight as _portal_archive_index_preflight
from transformation_portal.portal import asset_bundle as _portal_asset_bundle
from transformation_portal.portal import job_artifacts as _portal_job_artifacts
from transformation_portal.portal import path_security as _portal_path_security
from transformation_portal.portal import sam2_checkpoint_security as _portal_sam2_checkpoint_security
from transformation_portal.portal.asset_bundle import (
    PORTAL_ASSETS_DIR,
    PORTAL_ASSETS_DIR_REAL,
    PORTAL_ASSET_CACHE_CONTROL,
    PORTAL_ASSET_FINGERPRINT_LENGTH,
    PORTAL_ASSET_FINGERPRINT_PARAM,
    PORTAL_ASSET_MANIFEST,
    PORTAL_ASSET_MANIFEST_PATH,
    PORTAL_ASSET_MEDIA_TYPES,
    PORTAL_ASSET_PATHS,
    PORTAL_CSS_TEMPLATE_PATH,
    PORTAL_CSS_TEMPLATE_TOKENS,
    PORTAL_DIRECT_FINGERPRINT_ASSET_NAMES,
    PORTAL_HTML,
    PORTAL_HTML_TEMPLATE_TOKENS,
    PORTAL_IMMUTABLE_ASSET_CACHE_CONTROL,
    PortalAssetBundle,
    PortalAssetSpec,
    PortalRenderedTextAsset,
)
from transformation_portal.portal.job_artifacts import (
    AbsoluteArtifactPathError,
    ArtifactPathOutsideJobOutputDirError,
    ArtifactPathValidationError,
    InvalidArtifactPathError,
    JobRunMetadata,
)
from transformation_portal.portal.sam2_checkpoint_security import (
    ManagedSam2CheckpointValidationResult,
    _ManagedSam2BoundedChecksumCache,
    _ManagedSam2ChecksumCacheEntry,
    _Sam2CacheKey,
)
from transformation_portal.vlm_captioning.fastvlm_runtime import (
    FASTVLM_CHECKPOINT_DIRS,
    default_fastvlm_runtime_root,
    resolve_fastvlm_model_path,
)

# ----------------------------
# In-memory job store (MVP)
# ----------------------------

LOGGER = logging.getLogger(__name__)
_PORTAL_EVENT_LOG_WRITE_LOCK = threading.Lock()
_MANAGED_SAM2_CHECKSUM_CACHE_LOCK = threading.Lock()

# Optionally suppress successful /healthz and /ready access-log lines so
# operator log streams stay focused on actionable failures. Errors are never
# suppressed. Controlled via TP_LOG_HEALTHCHECKS=0 (default keeps the lines).
try:
    from transformation_portal.core.observability.log_classification import (
        install_healthcheck_log_filter as _install_healthcheck_log_filter,
    )

    _install_healthcheck_log_filter()
except Exception:  # pragma: no cover - observability never blocks startup
    LOGGER.debug("healthcheck log filter unavailable", exc_info=True)


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


def _env_rollout_percent(name: str, default: int = 0) -> int:
    return min(100, _env_int(name, default, minimum=0))


def _stable_rollout_bucket(key: str) -> int:
    normalized = str(key or "").strip().lower()
    if not normalized:
        return 100
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def _portal_rollout_cohort_key(
    actor: Optional[Mapping[str, Any]] = None,
    *,
    default: str = "direct-debug",
) -> str:
    actor_mapping = actor if isinstance(actor, Mapping) else {}
    return (
        str(
            actor_mapping.get("username")
            or actor_mapping.get("accessEmail")
            or actor_mapping.get("role")
            or os.getenv("TP_PORTAL_DIRECT_DEBUG_COHORT_KEY", default)
        )
        .strip()
        .lower()
    )


def _portal_artifact_viewer_modal_enabled(actor: Optional[Mapping[str, Any]] = None) -> bool:
    return _portal_rollout_enabled("TP_PORTAL_ARTIFACT_VIEWER_MODAL_ROLLOUT_PERCENT", actor)


def _portal_review_surface_deferred_enabled(actor: Optional[Mapping[str, Any]] = None) -> bool:
    return _portal_rollout_enabled("TP_PORTAL_REVIEW_SURFACE_DEFER_ROLLOUT_PERCENT", actor)


def _portal_rollout_enabled(env_name: str, actor: Optional[Mapping[str, Any]] = None) -> bool:
    rollout_percent = _env_rollout_percent(env_name, 0)
    if rollout_percent <= 0:
        return False
    cohort_key = _portal_rollout_cohort_key(actor)
    if not cohort_key:
        return False
    return _stable_rollout_bucket(cohort_key) < rollout_percent


def _portal_rum_enabled(actor: Optional[Mapping[str, Any]] = None) -> bool:
    if not _env_bool("TP_PORTAL_RUM_ENABLED", False):
        return False
    return _portal_rollout_enabled("TP_PORTAL_RUM_ROLLOUT_PERCENT", actor)


def _portal_staged_uploads_enabled(actor: Optional[Mapping[str, Any]] = None) -> bool:
    if not _env_bool("TP_PORTAL_UPLOAD_STAGING_ENABLED", False):
        return False
    return _portal_rollout_enabled("TP_PORTAL_STAGED_UPLOADS_ROLLOUT_PERCENT", actor)


def _portal_fastvlm_captioning_enabled(actor: Optional[Mapping[str, Any]] = None) -> bool:
    if not _env_bool("TP_PORTAL_FASTVLM_CAPTIONING_ENABLED", False):
        return False
    return _portal_rollout_enabled("TP_PORTAL_FASTVLM_CAPTIONING_ROLLOUT_PERCENT", actor)


REPO_ROOT = Path(__file__).resolve().parent
PORTAL_VIDEO_ASSET_NAME = "dna-portal-video-2.mp4"
PORTAL_VIDEO_PATH = REPO_ROOT / "public" / "video" / PORTAL_VIDEO_ASSET_NAME

_PORTAL_ASSET_BUNDLE_APP_GLOBALS = (
    "REPO_ROOT",
    "PORTAL_HTML",
    "PORTAL_ASSETS_DIR",
    "PORTAL_ASSETS_DIR_REAL",
    "PORTAL_ASSET_MANIFEST_PATH",
    "PORTAL_ASSET_CACHE_CONTROL",
    "PORTAL_IMMUTABLE_ASSET_CACHE_CONTROL",
    "PORTAL_ASSET_FINGERPRINT_PARAM",
    "PORTAL_ASSET_FINGERPRINT_LENGTH",
    "PORTAL_CSS_TEMPLATE_PATH",
    "PORTAL_DIRECT_FINGERPRINT_ASSET_NAMES",
    "PORTAL_CSS_TEMPLATE_TOKENS",
    "PORTAL_HTML_TEMPLATE_TOKENS",
    "PORTAL_ASSET_MANIFEST",
    "PORTAL_ASSET_PATHS",
    "PORTAL_ASSET_MEDIA_TYPES",
)


@contextmanager
def _portal_asset_bundle_app_context():
    saved = {name: getattr(_portal_asset_bundle, name) for name in _PORTAL_ASSET_BUNDLE_APP_GLOBALS}
    try:
        for name in _PORTAL_ASSET_BUNDLE_APP_GLOBALS:
            setattr(_portal_asset_bundle, name, globals()[name])
        yield
    finally:
        for name, value in saved.items():
            setattr(_portal_asset_bundle, name, value)


def _copy_portal_cache_api(wrapper: Callable[..., Any], cached: Callable[..., Any]) -> None:
    for attr_name in ("cache_clear", "cache_info", "cache_parameters"):
        if hasattr(cached, attr_name):
            setattr(wrapper, attr_name, getattr(cached, attr_name))


def _load_portal_asset_manifest() -> Dict[str, PortalAssetSpec]:
    with _portal_asset_bundle_app_context():
        return _portal_asset_bundle._load_portal_asset_manifest()


def _portal_asset_signature(path: Path) -> Tuple[str, int, int]:
    return _portal_asset_bundle._portal_asset_signature(path)


def _fingerprint_bytes(payload: bytes) -> str:
    return _portal_asset_bundle._fingerprint_bytes(payload)


def _portal_asset_route_path(asset_name: str) -> str:
    with _portal_asset_bundle_app_context():
        return _portal_asset_bundle._portal_asset_route_path(asset_name)


def _portal_asset_versioned_url(asset_name: str, fingerprint: str) -> str:
    with _portal_asset_bundle_app_context():
        return _portal_asset_bundle._portal_asset_versioned_url(asset_name, fingerprint)


def _render_portal_template(template_text: str, replacements: Mapping[str, str], *, template_name: str) -> str:
    return _portal_asset_bundle._render_portal_template(template_text, replacements, template_name=template_name)


def _portal_direct_asset_signature(asset_name: str) -> Tuple[str, int, int]:
    with _portal_asset_bundle_app_context():
        return _portal_asset_bundle._portal_direct_asset_signature(asset_name)


def _build_portal_direct_asset_fingerprint(asset_name: str, signature: Tuple[str, int, int]) -> str:
    with _portal_asset_bundle_app_context():
        return _portal_asset_bundle._build_portal_direct_asset_fingerprint(asset_name, signature)


def _get_portal_direct_asset_fingerprint(asset_name: str) -> str:
    with _portal_asset_bundle_app_context():
        return _portal_asset_bundle._get_portal_direct_asset_fingerprint(asset_name)


def _portal_css_dependency_asset_names() -> Tuple[str, ...]:
    with _portal_asset_bundle_app_context():
        return _portal_asset_bundle._portal_css_dependency_asset_names()


def _portal_css_signature() -> Tuple[object, ...]:
    with _portal_asset_bundle_app_context():
        return _portal_asset_bundle._portal_css_signature()


def _build_portal_css_asset(signature: Tuple[object, ...]) -> PortalRenderedTextAsset:
    with _portal_asset_bundle_app_context():
        return _portal_asset_bundle._build_portal_css_asset(signature)


def _get_portal_css_asset() -> PortalRenderedTextAsset:
    with _portal_asset_bundle_app_context():
        return _portal_asset_bundle._get_portal_css_asset()


def _portal_html_signature() -> Tuple[object, ...]:
    with _portal_asset_bundle_app_context():
        return _portal_asset_bundle._portal_html_signature()


def _build_portal_asset_bundle(signature: Tuple[object, ...]) -> PortalAssetBundle:
    with _portal_asset_bundle_app_context():
        return _portal_asset_bundle._build_portal_asset_bundle(signature)


def _get_portal_asset_bundle() -> PortalAssetBundle:
    with _portal_asset_bundle_app_context():
        return _portal_asset_bundle._get_portal_asset_bundle()


def _requested_portal_asset_fingerprint(request: Request) -> str:
    with _portal_asset_bundle_app_context():
        return _portal_asset_bundle._requested_portal_asset_fingerprint(request)


def _portal_asset_cache_control(current_fingerprint: str, requested_fingerprint: str) -> str:
    with _portal_asset_bundle_app_context():
        return _portal_asset_bundle._portal_asset_cache_control(current_fingerprint, requested_fingerprint)


def _portal_asset_etag(fingerprint: str) -> str:
    with _portal_asset_bundle_app_context():
        return _portal_asset_bundle._portal_asset_etag(fingerprint)


def _portal_asset_request_etag_matches(request: Request, current_etag: str) -> bool:
    with _portal_asset_bundle_app_context():
        return _portal_asset_bundle._portal_asset_request_etag_matches(request, current_etag)


def _portal_asset_not_modified_response(*, etag: str, cache_control: str) -> Response:
    with _portal_asset_bundle_app_context():
        return _portal_asset_bundle._portal_asset_not_modified_response(etag=etag, cache_control=cache_control)


_copy_portal_cache_api(_build_portal_direct_asset_fingerprint, _portal_asset_bundle._build_portal_direct_asset_fingerprint)
_copy_portal_cache_api(_build_portal_css_asset, _portal_asset_bundle._build_portal_css_asset)
_copy_portal_cache_api(_build_portal_asset_bundle, _portal_asset_bundle._build_portal_asset_bundle)


ARCHIVE_GOVERNANCE_SCRIPT = REPO_ROOT / "tools" / "archive_governance.py"
LUX_DEPTH_MODULE = "transformation_portal.lux_depth_v3"
APP_VERSION = "0.3.0"


def _normalize_root_path(value: str | Path) -> Path:
    return _portal_path_security._normalize_root_path(value, repo_root=REPO_ROOT)


def _default_allowed_path_roots() -> List[Path]:
    return _portal_path_security._default_allowed_path_roots(repo_root=REPO_ROOT)


def _env_path_roots(name: str, default: List[Path]) -> List[Path]:
    return _portal_path_security._env_path_roots(name, default, repo_root=REPO_ROOT, logger=LOGGER)


def _portal_telemetry_sink_path_from_env(name: str) -> Optional[Path]:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return None
    try:
        return _portal_path_security.validate_portal_telemetry_sink_path(raw_value, repo_root=REPO_ROOT)
    except _portal_path_security.PathSecurityValidationError as exc:
        raise RuntimeError(f"{name} violates portal telemetry sink path policy: {exc}") from exc


def _lux_depth_runner_command() -> List[str]:
    return [sys.executable, "-m", LUX_DEPTH_MODULE]


def _lux_depth_runner_available() -> bool:
    try:
        module_spec = find_spec(LUX_DEPTH_MODULE)
        if module_spec is None:
            return False
        if module_spec.submodule_search_locations is not None:
            return find_spec(f"{LUX_DEPTH_MODULE}.__main__") is not None
        return True
    except (ImportError, ValueError):
        return False


def _resolve_untrusted_request_path(path_value: str) -> Path:
    return _portal_path_security._resolve_untrusted_request_path(path_value, repo_root=REPO_ROOT)


def _portal_path_security_error(exc: _portal_path_security.PathSecurityValidationError) -> "_PortalValidationReasonError":
    return _PortalValidationReasonError(str(exc), reason=exc.reason)


def _validate_path_against_roots(
    path_value: str,
    allowed_roots: List[Path],
) -> str:
    return str(_resolve_allowed_request_path(path_value, allowed_roots))


def _resolve_allowed_request_path(
    path_value: str,
    allowed_roots: List[Path],
) -> Path:
    try:
        return _portal_path_security._resolve_allowed_request_path(path_value, allowed_roots, repo_root=REPO_ROOT)
    except _portal_path_security.PathSecurityValidationError as exc:
        raise _portal_path_security_error(exc) from None


def _resolved_portal_upload_root() -> Path:
    return _resolve_allowed_request_path(str(PORTAL_UPLOAD_ROOT), ALLOWED_INPUT_ROOTS)


def _path_is_within_root(resolved_path: Path, root: Path) -> bool:
    return _portal_path_security._path_is_within_root(resolved_path, root)


def _trusted_allowed_entry(
    resolved_path: Path,
    allowed_roots: List[Path],
) -> Optional[Path]:
    return _portal_path_security._trusted_allowed_entry(resolved_path, allowed_roots)


_UNSAFE_PATH_SEGMENT_RE = _portal_path_security._UNSAFE_PATH_SEGMENT_RE


def _trusted_existing_dir(value: str, allowed_roots: List[Path]) -> Optional[Path]:
    """Return ``value`` as a trusted existing directory inside an allowed root.

    Resolves the user-supplied string against the allowlist and then walks
    each segment via :func:`_trusted_allowed_entry` (``iterdir`` based). When
    the full path resolves to an existing directory, the trusted ``Path`` is
    returned; otherwise ``None``. Callers must not pass any other user-derived
    string to subsequent filesystem APIs.
    """

    return _portal_path_security._trusted_existing_dir(value, allowed_roots, repo_root=REPO_ROOT)


def _trusted_creatable_dir(value: str, allowed_roots: List[Path]) -> Optional[Path]:
    """Return ``value`` as a trusted directory path safe to ``mkdir`` later.

    Walks each existing segment via ``Path.iterdir()`` (so pre-existing
    components are validated against the parent's children rather than
    by-string) and validates the names of any non-existent trailing segments
    against :data:`_UNSAFE_PATH_SEGMENT_RE` to defeat traversal injection.
    """

    return _portal_path_security._trusted_creatable_dir(value, allowed_roots, repo_root=REPO_ROOT)


def _ensure_safe_regular_file_path(path_value: Path, allowed_roots: List[Path]) -> Path:
    try:
        return _portal_path_security._ensure_safe_regular_file_path(path_value, allowed_roots, repo_root=REPO_ROOT)
    except _portal_path_security.PathSecurityValidationError as exc:
        raise _portal_path_security_error(exc) from exc.__cause__


_MANAGED_SAM2_REASON_MESSAGES = _portal_sam2_checkpoint_security._MANAGED_SAM2_REASON_MESSAGES
_MANAGED_SAM2_CHECKSUM_CACHE: _ManagedSam2BoundedChecksumCache = _ManagedSam2BoundedChecksumCache()


def _managed_sam2_reason_message(reason: str) -> str:
    return _portal_sam2_checkpoint_security._managed_sam2_reason_message(reason)


def _managed_sam2_checksum_cache_key(path: Path) -> _Sam2CacheKey:
    return _portal_sam2_checkpoint_security._managed_sam2_checksum_cache_key(path)


def _clear_managed_sam2_checksum_cache() -> None:
    _portal_sam2_checkpoint_security._clear_managed_sam2_checksum_cache(
        _MANAGED_SAM2_CHECKSUM_CACHE,
        _MANAGED_SAM2_CHECKSUM_CACHE_LOCK,
    )


def _hash_file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    return _portal_sam2_checkpoint_security._hash_file_sha256(path, chunk_size)


def _cached_managed_sam2_checksum_result(path: Path) -> _ManagedSam2ChecksumCacheEntry:
    try:
        return _portal_sam2_checkpoint_security._cached_managed_sam2_checksum_result(
            path,
            trusted_sha256=MANAGED_SAM2_TRUSTED_SHA256,
            checksum_max_bytes=MANAGED_SAM2_CHECKSUM_MAX_BYTES,
            checksum_cache=_MANAGED_SAM2_CHECKSUM_CACHE,
            checksum_cache_lock=_MANAGED_SAM2_CHECKSUM_CACHE_LOCK,
            hash_file_sha256=_hash_file_sha256,
        )
    except _portal_sam2_checkpoint_security.Sam2CheckpointValidationError as exc:
        raise _PortalValidationReasonError(str(exc), reason=exc.reason) from exc.__cause__


def _resolve_managed_sam2_checkpoint_validation(path_value: str) -> ManagedSam2CheckpointValidationResult:
    return _portal_sam2_checkpoint_security._resolve_managed_sam2_checkpoint_validation(
        path_value,
        allowed_input_roots=ALLOWED_INPUT_ROOTS,
        trusted_roots=MANAGED_SAM2_TRUSTED_ROOTS,
        trusted_sha256=MANAGED_SAM2_TRUSTED_SHA256,
        checksum_max_bytes=MANAGED_SAM2_CHECKSUM_MAX_BYTES,
        checksum_cache=_MANAGED_SAM2_CHECKSUM_CACHE,
        checksum_cache_lock=_MANAGED_SAM2_CHECKSUM_CACHE_LOCK,
        repo_root=REPO_ROOT,
        hash_file_sha256=_hash_file_sha256,
    )


def _validate_managed_sam2_checkpoint_path(path_value: str) -> str:
    try:
        return _portal_sam2_checkpoint_security._validate_managed_sam2_checkpoint_path(
            path_value,
            allowed_input_roots=ALLOWED_INPUT_ROOTS,
            trusted_roots=MANAGED_SAM2_TRUSTED_ROOTS,
            trusted_sha256=MANAGED_SAM2_TRUSTED_SHA256,
            checksum_max_bytes=MANAGED_SAM2_CHECKSUM_MAX_BYTES,
            checksum_cache=_MANAGED_SAM2_CHECKSUM_CACHE,
            checksum_cache_lock=_MANAGED_SAM2_CHECKSUM_CACHE_LOCK,
            repo_root=REPO_ROOT,
            hash_file_sha256=_hash_file_sha256,
        )
    except _portal_sam2_checkpoint_security.Sam2CheckpointValidationError as exc:
        raise _PortalValidationReasonError(str(exc), reason=exc.reason) from exc.__cause__


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
# Keep fingerprinting inexpensive by default. Up to MAX_INDEXED_ARTIFACTS files
# are hashed per job (off the event loop via asyncio.to_thread); deployments
# that need copy-friendly hashes for larger artifacts can opt in via the env
# override.
ARTIFACT_FINGERPRINT_MAX_BYTES = _env_int(
    "TP_ARTIFACT_FINGERPRINT_MAX_BYTES",
    8 * 1024 * 1024,
    minimum=1024,
)
_ARTIFACT_FINGERPRINT_CHUNK_BYTES = 1024 * 1024
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
MAX_UPLOAD_REQUEST_BYTES = _env_int("TP_PORTAL_MAX_UPLOAD_REQUEST_BYTES", MAX_REQUEST_BYTES, minimum=1024)
PORTAL_UPLOAD_MAX_FILES = _env_int("TP_PORTAL_UPLOAD_MAX_FILES", 256, minimum=1)
PORTAL_UPLOAD_MAX_FIELDS = _env_int("TP_PORTAL_UPLOAD_MAX_FIELDS", 32, minimum=1)
PORTAL_UPLOAD_MAX_PART_BYTES = _env_int("TP_PORTAL_UPLOAD_MAX_PART_BYTES", MAX_UPLOAD_REQUEST_BYTES, minimum=1024)
RATE_LIMIT_PER_MINUTE = _env_int("TP_RATE_LIMIT_PER_MINUTE", 60, minimum=0)
MAX_CONCURRENT_JOBS = _env_int("TP_MAX_CONCURRENT_JOBS", 4, minimum=1)
RATE_LIMIT_WINDOW_SECONDS = 60.0
# Default Retry-After (seconds) returned when the job-admission concurrency cap
# (MAX_CONCURRENT_JOBS) rejects a request. Not tied to a deterministic schedule
# because jobs free their slot when they finish, so this is a conservative
# best-effort hint rather than a precise reset.
JOB_ADMISSION_RETRY_AFTER_SECONDS = 5
# Phase 2.C wiring (Phase 2.D promoted to default-on). The orchestrator
# enqueues via the QueueBroker and an in-process WorkerRunner pool consumes
# leases; cancellation flows through ``broker.cancel`` rather than directly
# poking ``Job.cancel_requested``. The legacy in-band
# ``asyncio.create_task(_run_job(...))`` path remains available behind
# ``TP_ORCHESTRATOR_USE_QUEUE_BROKER=0`` as an explicit opt-out for any
# single-process deployment or test fixture that needs the pre-Phase-2.C
# semantics. A subsequent cleanup PR will remove that fallback.
USE_QUEUE_BROKER = _env_bool("TP_ORCHESTRATOR_USE_QUEUE_BROKER", True)


def _worker_env_float(canonical: str, legacy: str, default: float, *, minimum: float) -> float:
    """Read a worker tunable that accepts both ``TP_WORKER_*`` and the legacy
    ``TP_ORCHESTRATOR_WORKER_*`` env names introduced in Phase 2.C.

    Phase 2.D unifies the names: ``TP_WORKER_*`` (matching
    ``worker.py``'s ``_config_from_env``) is the canonical surface, and
    the legacy ``TP_ORCHESTRATOR_WORKER_*`` form is honored as a
    fallback so existing deployments do not have to flip env vars
    twice. The canonical name wins when both are set.
    """
    if os.getenv(canonical) is not None:
        return _env_float(canonical, default, minimum=minimum)
    return _env_float(legacy, default, minimum=minimum)


# Per-worker lease + heartbeat tunables. Defaults intentionally generous so
# the in-process worker pool never reclaims its own jobs under realistic
# single-process load; production multi-host deployments tighten these via env.
WORKER_LEASE_SECONDS = _worker_env_float("TP_WORKER_LEASE_SECONDS", "TP_ORCHESTRATOR_WORKER_LEASE_SECONDS", 300.0, minimum=1.0)
WORKER_HEARTBEAT_INTERVAL_SECONDS = _worker_env_float(
    "TP_WORKER_HEARTBEAT_SECONDS", "TP_ORCHESTRATOR_WORKER_HEARTBEAT_SECONDS", 30.0, minimum=0.1
)
WORKER_POLL_INTERVAL_SECONDS = _worker_env_float(
    "TP_WORKER_POLL_SECONDS", "TP_ORCHESTRATOR_WORKER_POLL_SECONDS", 0.05, minimum=0.001
)
# Grace period the lifespan shutdown waits for the worker pool to drain
# cleanly before escalating to ``task.cancel()``. Bounded so a stuck
# executor cannot block the broker / repository close paths.
WORKER_SHUTDOWN_GRACE_SECONDS = _env_float("TP_ORCHESTRATOR_WORKER_SHUTDOWN_GRACE_SECONDS", 5.0, minimum=0.1)
# Phase 2.D - broker reclaim reconciler. Periodically calls
# ``broker.reclaim_expired_leases(now=server_time())`` so any lease whose
# heartbeat fell behind (worker died, event-loop blocked, multi-host fleet
# lost a node) gets the dispatch payload reclaimed; the orchestrator then
# transitions the in-process ``Job`` and the repository row to
# ``worker_lost`` with ``error.retriable=True`` and publishes the terminal
# ``done`` event so SSE late-clients see closure. Interval is bounded
# below to keep test runs fast and above to keep production overhead small.
RECLAIM_SWEEP_INTERVAL_SECONDS = _env_float("TP_ORCHESTRATOR_RECLAIM_SWEEP_INTERVAL_SECONDS", 5.0, minimum=0.01)
WORKER_LOST_REASON_RECLAIMED = "worker_lost_via_lease_reclaim"
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
FASTVLM_RUNTIME_ALLOWED_ROOTS = list(
    dict.fromkeys(
        [
            *ALLOWED_INPUT_ROOTS,
            Path(os.path.realpath(default_fastvlm_runtime_root())),
        ]
    )
)
MANAGED_SAM2_TRUSTED_ROOTS = [
    Path(os.path.realpath(REPO_ROOT / "models" / "sam2")),
    Path(os.path.realpath(REPO_ROOT / "checkpoints")),
]
MANAGED_SAM2_TRUSTED_SHA256 = frozenset(
    {
        "d0bb7f236400a49669ffdd1be617959a8b1d1065081789d7bbff88eded3a8071",
        "7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b",
    }
)
MANAGED_SAM2_CHECKSUM_MAX_BYTES = _env_int(
    "TP_MANAGED_SAM2_CHECKSUM_MAX_BYTES",
    1024 * 1024 * 1024,
    minimum=1024,
)
PORTAL_UPLOAD_ROOT = Path(
    os.getenv("TP_PORTAL_UPLOAD_ROOT", str(Path(tempfile.gettempdir()) / "transformation-portal" / "uploads"))
).expanduser()
PORTAL_UPLOAD_TTL_SECONDS = _env_int("TP_PORTAL_UPLOAD_TTL_SECONDS", 24 * 60 * 60, minimum=1)


def _allowed_roots_for_scope(scope: str) -> List[Path]:
    if scope == PATH_SCOPE_INPUT:
        return ALLOWED_INPUT_ROOTS
    if scope == PATH_SCOPE_OUTPUT:
        return ALLOWED_OUTPUT_ROOTS
    return ALLOWED_PATH_ROOTS


def _repo_top_level_entries() -> set[str]:
    try:
        return {entry.name for entry in REPO_ROOT.iterdir()}
    except OSError:
        return set()


def _is_single_leading_slash_path(raw_value: str) -> bool:
    return raw_value.startswith("/") and not raw_value.startswith("//")


def _is_valid_allowed_absolute_path(path_value: str, allowed_roots: List[Path]) -> bool:
    raw = str(path_value or "").strip()
    if not raw:
        return False
    try:
        if not Path(raw).is_absolute():
            return False
    except (OSError, RuntimeError, ValueError):
        return False
    try:
        _resolve_allowed_request_path(raw, allowed_roots)
    except ValueError:
        return False
    return True


def _normalize_repo_relative_display_path(raw_value: str, resolved_path: Path) -> str:
    raw = str(raw_value or "").strip()
    try:
        if Path(raw).is_absolute():
            return str(resolved_path)
    except (OSError, RuntimeError, ValueError):
        return str(resolved_path)

    repo_real = Path(os.path.realpath(REPO_ROOT))
    try:
        relative = resolved_path.relative_to(repo_real).as_posix()
    except ValueError:
        return str(resolved_path)
    return "." if not relative else f"./{relative}"


def _attempt_repo_local_path_repair(
    path_value: str,
    *,
    allowed_roots: List[Path],
    repo_entries: set[str],
) -> tuple[Optional[str], Optional[str]]:
    raw = str(path_value or "").strip()
    if not _is_single_leading_slash_path(raw):
        return None, None
    if _is_valid_allowed_absolute_path(raw, allowed_roots):
        return None, None

    candidate = raw[1:]
    if not candidate:
        return None, None

    segments = candidate.split("/")
    if segments and segments[-1] == "":
        segments = segments[:-1]
    if not segments:
        return None, None
    if any(segment in {".", ".."} for segment in segments):
        return None, PATH_SHORTHAND_TRAVERSAL_DISALLOWED_CODE
    if any(not segment for segment in segments):
        return None, None
    if segments[0] not in repo_entries:
        return None, None
    return "./" + "/".join(segments), None


def _normalize_operator_payload_paths(
    pipeline: str,
    args: Dict[str, Any],
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if pipeline not in ALLOWED_PIPELINES:
        return dict(args or {}), [], []

    normalized_args = dict(args or {})
    warnings: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    repo_entries = _repo_top_level_entries()

    for canonical_field, aliases, scope in PATH_FIELD_SPECS:
        raw_value = _pick(normalized_args, *aliases, default=None)
        text = str(raw_value or "").strip()
        if not text or text.startswith("~") or "\x00" in text:
            continue

        repaired, error_code = _attempt_repo_local_path_repair(
            text,
            allowed_roots=_allowed_roots_for_scope(scope),
            repo_entries=repo_entries,
        )
        if error_code == PATH_SHORTHAND_TRAVERSAL_DISALLOWED_CODE:
            errors.append(
                _portal_issue(
                    canonical_field,
                    PATH_SHORTHAND_TRAVERSAL_DISALLOWED_CODE,
                    f"{canonical_field} cannot use repo-local shorthand with '.' or '..' segments.",
                    suggestion="Use a direct workspace-relative path without traversal segments.",
                )
            )
            continue
        if not repaired:
            continue

        for alias in aliases:
            if alias in normalized_args:
                normalized_args[alias] = repaired
        warnings.append(
            _portal_issue(
                canonical_field,
                REPO_LOCAL_PATH_REPAIRED_CODE,
                f"{canonical_field} was normalized to a workspace-relative path.",
                suggestion="The portal repaired repo-local leading-slash shorthand to canonical ./... form.",
            )
        )

    return normalized_args, warnings, errors


ENABLE_API_DOCS = _env_bool("TP_ENABLE_API_DOCS", False)
READY_VERBOSE = _env_bool("TP_READY_VERBOSE", False)
DEFAULT_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "font-src 'self'; "
    "img-src 'self' data: blob:; "
    "media-src 'self'; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "frame-ancestors 'none'; "
    "form-action 'self';"
)
PORTAL_VIDEO_CACHE_CONTROL = "public, max-age=86400"
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
                "model_key": "da3-metric",
                "enable_segmentation": True,
                "segmentation_backend": "efficientsam",
                "strict_segmentation": True,
                "materials_v3": True,
                "pbr": True,
                "emit_master16": True,
                "emit_upscaled16": True,
                "emit_report": True,
                "emit_run_card": True,
                "run_card_version": "v1",
                "run_card_include_proofs": False,
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
                "model_key": "da3-metric",
                "enable_segmentation": False,
                "segmentation_backend": "stub",
                "strict_segmentation": False,
                "materials_v3": False,
                "pbr": False,
                "emit_master16": True,
                "emit_upscaled16": False,
                "emit_report": True,
                "emit_run_card": True,
                "run_card_version": "v1",
                "run_card_include_proofs": False,
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
                "depth_backend": "da3",
                "model_key": "da3-research",
                "enable_segmentation": True,
                "segmentation_backend": "sam2",
                "strict_segmentation": True,
                "materials_v3": True,
                "pbr": True,
                "emit_master16": True,
                "emit_upscaled16": True,
                "emit_report": True,
                "emit_run_card": True,
                "run_card_version": "v2",
                "run_card_include_proofs": False,
                "emit_marketing": False,
                "enable_v2": True,
                "v2_preset": "default",
                "enable_reconstruction": False,
            },
            "advanced_sections": ["governance", "advanced"],
        },
        {
            "name": "depth-pro-research-m4",
            "label": "Depth Pro research (Experimental)",
            "stability": "experimental",
            "description": "Depth Pro research-only preset requiring non-commercial and Apple acknowledgments",
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
                "run_card_version": "v2",
                "run_card_include_proofs": False,
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
    last_event_at: Optional[float] = None
    state: str = "queued"  # queued|running|succeeded|partial|failed|canceled|worker_lost
    progress: int = 0
    exit_code: Optional[int] = None
    request: Dict[str, Any] = dataclass_field(default_factory=dict)
    effective_request: Dict[str, Any] = dataclass_field(default_factory=dict)
    logs_tail: List[str] = dataclass_field(default_factory=list)
    artifacts: Dict[str, Any] = dataclass_field(default_factory=dict)
    artifact_lookup: Dict[str, Path] = dataclass_field(default_factory=dict)
    run_summary: Dict[str, Any] = dataclass_field(default_factory=dict)
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
ACTIVE_JOB_STATES = {"queued", "running"}
# Phase 2.D - terminal states a job can reach. ``worker_lost`` joins the
# existing terminal states so the broker-reclaim reconciler, the executor's
# terminal-state guard, and operator tooling all agree on what "done"
# means.
_TERMINAL_JOB_STATES = {"succeeded", "partial", "failed", "canceled", "worker_lost"}
JOB_RUN_SUMMARY_MAX_BYTES = 1024 * 1024

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
ARCHIVE_INDEX_REQUIRED_COLUMNS = _portal_archive_index_preflight.ARCHIVE_INDEX_REQUIRED_COLUMNS
ARCHIVE_INDEX_PREFLIGHT_EXAMPLE_LIMIT = _portal_archive_index_preflight.ARCHIVE_INDEX_PREFLIGHT_EXAMPLE_LIMIT
ARCHIVE_INDEX_PREFLIGHT_PREVIEW_ROW_LIMIT = _portal_archive_index_preflight.ARCHIVE_INDEX_PREFLIGHT_PREVIEW_ROW_LIMIT
ARCHIVE_INDEX_PREFLIGHT_CACHE_MAX = _portal_archive_index_preflight.ARCHIVE_INDEX_PREFLIGHT_CACHE_MAX
ARCHIVE_INDEX_PREFLIGHT_SCAN_MODES = _portal_archive_index_preflight.ARCHIVE_INDEX_PREFLIGHT_SCAN_MODES
_ARCHIVE_INDEX_PREFLIGHT_CACHE_LOCK = _portal_archive_index_preflight._ARCHIVE_INDEX_PREFLIGHT_CACHE_LOCK
_ARCHIVE_INDEX_PREFLIGHT_CACHE = _portal_archive_index_preflight._ARCHIVE_INDEX_PREFLIGHT_CACHE
ALLOWED_QUALITY = {"standard", "premium", "apex"}
ALLOWED_BACKENDS = {"da3", "depth_pro"}
ALLOWED_DEPTH_DEVICES = {"cpu", "cuda", "mps"}
ALLOWED_RUN_CARD_VERSIONS = {"v1", "v2"}
ALLOWED_SEGMENTATION_BACKENDS = {"stub", "efficientsam", "sam2"}
ALLOWED_SEGMENTATION_CACHE_POLICIES = {"off", "read_write"}
ALLOWED_SAM2_MODEL_SIZES = {"base", "large"}
ALLOWED_GROUPING_MODES = {"single", "parent_dir"}
ALLOWED_RECONSTRUCTION_TIERS = {
    "apex_research",
    "apex_research_ultra",
    "experimental",
}
ALLOWED_RAW_INGEST_MODES = {"auto", "force_rawpy", "force_preview"}
ALLOWED_RAW_WB_MODES = {"camera"}
# Demosaic name validation: orchestrator-side syntactic gate only. The
# subprocess RAW worker holds the authoritative semantic gate
# (resolve_demosaic_algorithm), which reflects the installed LibRaw build's
# actual rawpy.DemosaicAlgorithm members and fails closed on unknown names.
# Curating a hard-coded allowlist here would artificially block valid members
# in newer/older LibRaw builds (e.g. AFD, VCD, VCD_MODIFIED_AHD).
try:
    from transformation_portal.core.raw_runtime import (
        is_valid_demosaic_name as _is_valid_demosaic_name,
    )
except ImportError:  # pragma: no cover - defensive fallback if core import fails
    import re as _re

    _DEMOSAIC_NAME_RE = _re.compile(r"^[A-Z][A-Z0-9_]{0,31}$")

    def _is_valid_demosaic_name(name: object) -> bool:
        return isinstance(name, str) and bool(_DEMOSAIC_NAME_RE.fullmatch(name.strip().upper()))


ALLOWED_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}
ALLOWED_VLM_CAPTIONING_BACKENDS = {"fastvlm"}
FASTVLM_RUN_STATUS_VALUES = {
    "off",
    "requested",
    "succeeded",
    "failed",
    "skipped",
    "missing_runtime",
    "invalid_config",
    "unsupported_backend",
}
FASTVLM_RUNTIME_STATUS_ALIASES = {
    "ok": "succeeded",
    "success": "succeeded",
    "successful": "succeeded",
    "succeeded": "succeeded",
    "error": "failed",
    "failed": "failed",
    "failure": "failed",
    "proxy_error": "failed",
    "timeout": "failed",
    "missing_model": "missing_runtime",
    "missing_runtime": "missing_runtime",
    "invalid_config": "invalid_config",
    "unsupported_backend": "unsupported_backend",
    "skipped": "skipped",
    "disabled": "off",
    "off": "off",
    "requested": "requested",
}
ALLOWED_VLM_CAPTIONING_PROXY_FORMATS = {"png", "jpeg"}
ALLOWED_VLM_CAPTIONING_MODEL_ROLES = frozenset(FASTVLM_CHECKPOINT_DIRS.keys())
PORTAL_DEFAULT_DA3_MODEL_KEY = "da3-metric"
PORTAL_DA3_MODEL_KEY_BY_REGISTRY_KEY = {
    "da3_metric": "da3-metric",
    "da3_research": "da3-research",
}
DEPTH_BACKEND_ALIASES = {
    "depth_anything_v3": "da3",
    "depth-anything-v3": "da3",
}
VALIDATION_REASON_CODES = {
    "Unsupported pipeline": "unsupported_pipeline",
    "input_dir and output_dir are required": "missing_required_paths",
    "Invalid path value": "invalid_path_value",
    "Path shorthand traversal disallowed": "path_shorthand_traversal_disallowed",
    "Path outside allowed roots": "path_outside_allowed_roots",
    "SAM2 checkpoint path is not trusted": "untrusted_checkpoint_path",
    "SAM2 checkpoint path exceeds checksum verification size limit": "checkpoint_file_too_large",
    "Invalid quality_tier": "invalid_quality_tier",
    "Invalid depth_backend": "invalid_depth_backend",
    "Invalid model_key": "invalid_model_key",
    "Invalid segmentation_backend": "invalid_segmentation_backend",
    "Invalid sam2_model_size": "invalid_sam2_model_size",
    "Invalid reconstruction_tier": "invalid_reconstruction_tier",
    "Invalid raw_ingest_mode": "invalid_raw_ingest_mode",
    "Invalid raw_wb_mode": "invalid_raw_wb_mode",
    "Invalid raw_demosaic": "invalid_raw_demosaic",
    "Invalid log_level": "invalid_log_level",
    "Invalid vlm_captioning_backend": "invalid_vlm_captioning_backend",
    "Invalid vlm_captioning_model": "invalid_vlm_captioning_model",
    "Invalid vlm_captioning_proxy_format": "invalid_vlm_captioning_proxy_format",
    "Invalid vlm_captioning_max_side_px": "invalid_vlm_captioning_max_side_px",
    "Invalid fastvlm_timeout_seconds": "invalid_fastvlm_timeout_seconds",
    "verbose and quiet are mutually exclusive": "conflicting_log_verbosity_flags",
    "Archive governance runner unavailable": "archive_runner_unavailable",
    "Invalid archive_command": "invalid_archive_command",
    "Invalid archive integer option": "invalid_archive_integer_option",
}
PORTAL_SAFE_ERROR_MESSAGES = {
    "archive_index_root_mismatch": "Archive index rows must resolve under the selected archive root.",
    "archive_index_required": "An archive index artifact is required before dispatch.",
    "archive_runner_unavailable": "The selected archive command is unavailable in this environment.",
    "bag_dir_required": "A bag directory is required before dispatch.",
    "input_dir_required": "An existing input directory is required before dispatch.",
    "output_dir_unwritable": "The output directory or its parent is not writable.",
    "invalid_depth_device": "The selected compute device is not supported.",
    "invalid_preset": "The selected preset is not supported.",
    "invalid_run_card_version": "The selected run card version is not supported.",
    "checkpoint_file_too_large": "Managed SAM2 checkpoint overrides exceed the checksum verification size limit.",
    "conflicting_log_verbosity_flags": "Verbose and quiet mode cannot both be enabled.",
    "hash_manifest_required": "A hash manifest artifact is required before dispatch.",
    "invalid_archive_command": "The selected archive command is not supported.",
    "invalid_archive_integer_option": "One or more archive numeric options are invalid.",
    "invalid_depth_backend": "The selected depth backend is not supported.",
    "invalid_event_type": "The telemetry event type is not supported.",
    "invalid_field": "The telemetry field is not supported.",
    "invalid_log_level": "The selected log level is not supported.",
    "invalid_model_key": "The selected DA3 model is not supported.",
    "invalid_path_value": "One or more configured paths are invalid.",
    "invalid_pipeline": "The selected pipeline is not supported.",
    "invalid_quality_tier": "The selected quality tier is not supported.",
    "invalid_raw_demosaic": "The selected RAW demosaic mode is not supported.",
    "invalid_raw_ingest_mode": "The selected RAW ingest mode is not supported.",
    "invalid_raw_wb_mode": "The selected RAW white-balance mode is not supported.",
    "invalid_reconstruction_tier": "The selected reconstruction tier is not supported.",
    "invalid_request": "The request contains invalid values.",
    "invalid_sam2_model_size": "The selected SAM2 model size is not supported.",
    "invalid_segmentation_backend": "The selected segmentation backend is not supported.",
    "invalid_surface": "The telemetry surface is not supported.",
    "invalid_vlm_captioning_backend": "The selected captioning backend is not supported.",
    "invalid_vlm_captioning_model": "The selected captioning model is not supported.",
    "invalid_vlm_captioning_proxy_format": "The selected captioning proxy format is not supported.",
    "invalid_vlm_captioning_max_side_px": "The selected captioning proxy size is invalid.",
    "invalid_fastvlm_timeout_seconds": "The selected FastVLM timeout is invalid.",
    "captioning_feature_disabled": "FastVLM captioning is not enabled for this portal cohort.",
    "manifest_jsonl_required": "A manifest JSONL artifact is required before dispatch.",
    "missing_required_paths": "Input and output paths are required.",
    "path_outside_allowed_roots": "Configured paths must stay within the allowed workspace roots.",
    "path_shorthand_traversal_disallowed": "Repo-local shorthand paths cannot include traversal segments.",
    "policy_yaml_required": "A rights policy YAML file is required before dispatch.",
    "rights_manifest_required": "A rights-manifest JSONL artifact is required before dispatch.",
    "runner_unavailable": "The selected pipeline runner is unavailable in this environment.",
    "da3_runtime_unavailable": "The selected DA3 runtime is unavailable.",
    "da3_model_non_commercial_required": "The selected DA3 model requires non-commercial acknowledgment.",
    "untrusted_checkpoint_path": "Managed SAM2 checkpoint overrides must use a repo-controlled or checksum-verified file.",
    "unsafe_path": "Configured paths must stay within the allowed workspace roots.",
    "unsupported_pipeline": "The selected pipeline is not supported.",
}
PORTAL_EVENT_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,63}$")
PORTAL_ALLOWED_EVENT_TYPES = {
    "field_commit",
    "toggle_change",
    "preview_error_seen",
    "effective_config_opened",
    "config_exported",
    "step_completed",
    "job_submitted",
    "job_selected",
    "artifact_opened",
    "artifact_viewer_opened",
    "artifact_viewer_fallback",
    "artifact_compared",
    "run_details_opened",
    "cancel_requested",
    "stream_reconnected",
    "dispatch_blocked",
    "debug_bundle_guardrail_seen",
}
PORTAL_ALLOWED_EVENT_SURFACES = {
    "mission_control",
    "reconstruction_runtime",
    "effective_config",
    "dispatch",
    "build_stepper",
    "job_queue",
    "job_inspector",
    "artifact_review",
    "stream_transport",
}
PORTAL_ALLOWED_EVENT_FIELDS = {
    "accept_apple_depth_pro_research_license",
    "accept_research_tools_license",
    "debug_bundle_acknowledged",
    "depth_backend",
    "enable_reconstruction",
    "emit_scene_debug_bundle",
    "grouping_mode",
    "log_level",
    "max_gpu_workers",
    "max_workers",
    "max_gpu_workers_mode",
    "max_workers_mode",
    "model_key",
    "non_commercial_ok",
    "quality_tier",
    "raw_ingest_mode",
    "reconstruction_iterations",
    "reconstruction_tier",
    "sam2_crop_n_layers",
    "sam2_global_pass_longest_side",
    "sam2_max_concurrency",
    "sam2_model_size",
    "sam2_overlap_px",
    "sam2_points_per_batch",
    "sam2_points_per_side",
    "sam2_pred_iou_thresh",
    "sam2_stability_score_thresh",
    "sam2_tile_size_px",
    "sam2_tiling_enabled",
    "run_card_include_proofs",
    "run_card_version",
    "segmentation_backend",
    "segmentation_cache",
    "strict_segmentation",
    "vlm_captioning_enabled",
    "vlm_captioning_backend",
    "vlm_captioning_model",
    "vlm_captioning_proxy_format",
    "vlm_captioning_max_side_px",
    "fastvlm_python_executable",
    "fastvlm_mlx_vlm_dir",
    "fastvlm_timeout_seconds",
}
PORTAL_ALLOWED_RUM_EVENT_TYPES = {
    "portal_shell_rendered",
    "bootstrap_ready",
    "first_view_interactive",
    "core_web_vital",
    "queue_request",
    "sse_reconnect",
    "landing_rendered",
    "login_rendered",
    "login_submit_attempt",
    "login_submit_success",
    "login_submit_failure",
    "logout_submit_attempt",
    "logout_submit_success",
    "logout_submit_failure",
}
PORTAL_ALLOWED_RUM_ROUTES = {"/portal", "/", "/login", "/logout"}
PORTAL_ALLOWED_RUM_VIEWS = {
    "overview",
    "build",
    "operate",
    "review",
    "landing",
    "login",
}
PORTAL_ALLOWED_RUM_UNITS = {"ms", "score", "count"}
PORTAL_ALLOWED_RUM_METRICS = {
    "bootstrap_ready": {"duration"},
    "core_web_vital": {"cls", "inp", "lcp"},
    "first_view_interactive": {"duration"},
    "portal_shell_rendered": {"duration"},
    "queue_request": {"cancel", "submit"},
    "sse_reconnect": set(),
    "landing_rendered": {"duration"},
    "login_rendered": {"duration"},
    "login_submit_attempt": {"count"},
    "login_submit_success": {"duration"},
    "login_submit_failure": {"duration"},
    "logout_submit_attempt": {"count"},
    "logout_submit_success": {"duration"},
    "logout_submit_failure": {"duration"},
}
PATH_SCOPE_INPUT = "input"
PATH_SCOPE_OUTPUT = "output"
PATH_SCOPE_ANY = "path"
REPO_LOCAL_PATH_REPAIRED_CODE = "repo_local_path_repaired"
PATH_SHORTHAND_TRAVERSAL_DISALLOWED_CODE = "path_shorthand_traversal_disallowed"
PATH_FIELD_SPECS: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("input_dir", ("input_dir", "inputDir"), PATH_SCOPE_INPUT),
    ("output_dir", ("output_dir", "outputDir"), PATH_SCOPE_OUTPUT),
    ("sam2_checkpoint_path", ("sam2_checkpoint_path", "sam2CheckpointPath"), PATH_SCOPE_INPUT),
    ("cameras_sidecar_path", ("cameras_sidecar_path", "camerasSidecarPath"), PATH_SCOPE_INPUT),
    ("fastvlm_python_executable", ("fastvlm_python_executable", "fastvlmPythonExecutable"), PATH_SCOPE_INPUT),
    ("fastvlm_mlx_vlm_dir", ("fastvlm_mlx_vlm_dir", "fastvlmMlxVlmDir"), PATH_SCOPE_INPUT),
    ("archive_index", ("archive_index", "archiveIndex"), PATH_SCOPE_ANY),
    ("manifest_jsonl", ("manifest_jsonl", "manifestJsonl"), PATH_SCOPE_OUTPUT),
    ("archive_root", ("archive_root", "archiveRoot"), PATH_SCOPE_INPUT),
    ("out_dir", ("out_dir", "outDir"), PATH_SCOPE_OUTPUT),
    ("hash_manifest", ("hash_manifest", "hashManifest"), PATH_SCOPE_OUTPUT),
    ("report_path", ("report_path", "reportPath"), PATH_SCOPE_OUTPUT),
    ("out_jsonl", ("out_jsonl", "outJsonl"), PATH_SCOPE_OUTPUT),
    ("out_summary", ("out_summary", "outSummary"), PATH_SCOPE_OUTPUT),
    ("policy_yaml", ("policy_yaml", "policyYaml"), PATH_SCOPE_INPUT),
    ("bag_dir", ("bag_dir", "bagDir"), PATH_SCOPE_OUTPUT),
    ("report_json", ("report_json", "reportJson"), PATH_SCOPE_OUTPUT),
    ("out_ledger", ("out_ledger", "outLedger"), PATH_SCOPE_OUTPUT),
    ("out_xml", ("out_xml", "outXml"), PATH_SCOPE_OUTPUT),
    ("out_prov_jsonld", ("out_prov_jsonld", "outProvJsonld"), PATH_SCOPE_OUTPUT),
    ("out_stac_catalog", ("out_stac_catalog", "outStacCatalog"), PATH_SCOPE_OUTPUT),
    ("out_stac_items_dir", ("out_stac_items_dir", "outStacItemsDir"), PATH_SCOPE_OUTPUT),
    ("rights_jsonl", ("rights_jsonl", "rightsJsonl"), PATH_SCOPE_INPUT),
)
LUX_PORTAL_DEFAULT_ARGS: Dict[str, Any] = {
    "preset": "premium",
    "quality_tier": "apex",
    "depth_backend": "da3",
    "model_key": "da3-metric",
    "depth_device": "cpu",
    "enable_segmentation": False,
    "segmentation_backend": "stub",
    "segmentation_cache": "read_write",
    "sam2_model_size": "base",
    "sam2_tiling_enabled": False,
    "sam2_tile_size_px": 1536,
    "sam2_overlap_px": 256,
    "sam2_global_pass_longest_side": 1280,
    "sam2_max_concurrency": 1,
    "sam2_points_per_side": 32,
    "sam2_points_per_batch": 64,
    "sam2_pred_iou_thresh": 0.88,
    "sam2_stability_score_thresh": 0.85,
    "sam2_crop_n_layers": 1,
    "strict_segmentation": False,
    "materials_v3": True,
    "pbr": True,
    "save_float_depth": False,
    "cache_depth": True,
    "enable_v2": False,
    "v2_preset": "default",
    "emit_master16": True,
    "emit_upscaled16": True,
    "emit_marketing": False,
    "emit_report": True,
    "emit_run_card": True,
    "run_card_version": "v1",
    "run_card_include_proofs": False,
    "non_commercial_ok": False,
    "accept_apple_depth_pro_research_license": False,
    "accept_research_tools_license": False,
    "enable_reconstruction": False,
    "grouping_mode": "single",
    "reconstruction_iterations": 1000,
    "reconstruction_tier": "apex_research",
    "emit_scene_debug_bundle": False,
    "force_depth": False,
    "strict_inputs": False,
    "verify_images": False,
    "allow_semantic_fallback": False,
    "raw_ingest_mode": "auto",
    "raw_wb_mode": "camera",
    "raw_demosaic": "AHD",
    "vlm_captioning_enabled": False,
    "vlm_captioning_backend": "fastvlm",
    "vlm_captioning_model": "default",
    "vlm_captioning_proxy_format": "png",
    "vlm_captioning_max_side_px": 1600,
    "fastvlm_python_executable": "",
    "fastvlm_mlx_vlm_dir": "",
    "fastvlm_timeout_seconds": 180,
    "verbose": False,
    "quiet": False,
    "overwrite": False,
}
LUX_RECONSTRUCTION_INACTIVE_FIELDS = (
    "grouping_mode",
    "cameras_sidecar_path",
    "reconstruction_iterations",
    "reconstruction_tier",
    "emit_scene_debug_bundle",
)
LUX_RECONSTRUCTION_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "grouping_mode": ("grouping_mode", "groupingMode"),
    "cameras_sidecar_path": ("cameras_sidecar_path", "camerasSidecarPath"),
    "reconstruction_iterations": ("reconstruction_iterations", "reconstructionIterations"),
    "reconstruction_tier": ("reconstruction_tier", "reconstructionTier"),
    "emit_scene_debug_bundle": ("emit_scene_debug_bundle", "emitSceneDebugBundle"),
}
LUX_DEBUG_BUNDLE_DESTINATION_TEMPLATE = "reconstruction/<scene-fingerprint>/debug"

PORTAL_EVENT_LOG_PATH = _portal_telemetry_sink_path_from_env("TP_PORTAL_EVENT_LOG_PATH")
PORTAL_RUM_LOG_PATH = _portal_telemetry_sink_path_from_env("TP_PORTAL_RUM_LOG_PATH")


class JobPreflightError(RuntimeError):
    """Raised when a job fails readiness preflight before argv construction."""

    def __init__(
        self,
        reason: str,
        *,
        field: Optional[str] = None,
        message: str = "job blocked by readiness preflight",
        status_code: int = 400,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.field = field
        self.message = message
        self.status_code = status_code
        self.extra = extra or {}

    @property
    def details(self) -> Dict[str, Any]:
        details: Dict[str, Any] = {"reason": self.reason}
        if self.field:
            details["field"] = self.field
        details.update(self.extra)
        return details


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
    *,
    retriable: Optional[bool] = None,
) -> Dict[str, Any]:
    """Construct a wire-shape error dict for ``ApiEnvelope.error`` / ``Job.error``.

    Phase 2.D adds the optional ``retriable`` field so callers can
    distinguish executor-level failures (``retriable=False`` —
    re-running the same job will fail the same way) from broker-level
    failures (``retriable=True`` — ``worker_lost`` etc., the job
    payload is intact and a retry can succeed).

    Emission rules:

    - Callers that pass ``retriable=False`` / ``retriable=True``
      embed the field on the wire. Phase 2.D opts the executor
      failure paths (``RUNNER_EXIT_NONZERO`` / ``RUNNER_NOT_FOUND``
      / ``RUNNER_ERROR``) into ``retriable=False``, so those
      payloads now grow a ``retriable`` key — clients that did not
      previously see the field will start seeing it. The change is
      additive (no existing key changes type or value), so JSON
      consumers that ignore unknown fields are unaffected.
    - Callers that omit the argument (``retriable=None``) get the
      pre-Phase-2.D shape with no ``retriable`` key. The
      orchestrator's many non-executor errors (HTTP-level
      validation, request-parse failures, etc.) still use this
      path, so a typical 4xx envelope on the wire is byte-identical
      to its pre-2.D form.
    """
    payload: Dict[str, Any] = {"code": code, "message": message, "details": details or {}}
    if retriable is not None:
        payload["retriable"] = retriable
    return payload


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


def _auth_mode() -> str:
    return "direct_debug"


def _readiness_issue(
    reason: str,
    *,
    severity: str,
    message: str,
    field: Optional[str] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    issue: Dict[str, Any] = {
        "reason": reason,
        "severity": severity,
        "message": message,
    }
    if field:
        issue["field"] = field
    if path:
        issue["path"] = path
    return issue


def _module_available(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _resolve_lux_depth_canary_runtime() -> Optional[Path]:
    configured = os.getenv("TRANSFORMATION_PORTAL_DA3_PYTHON", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        if candidate.exists():
            return candidate.resolve()

    repo_local = REPO_ROOT / ".runtime" / "Depth-Anything-3" / ".venv-da3" / "bin" / "python"
    if repo_local.exists():
        return repo_local.resolve()
    return None


def _validate_existing_path(
    raw_value: Any,
    *,
    field: str,
    allowed_roots: List[Path],
    missing_reason: str,
    missing_message: str,
    expected_type: str,
    required: bool,
) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    text = str(raw_value or "").strip()
    if not text:
        if required:
            return None, _readiness_issue(
                missing_reason,
                severity="blocked",
                message=missing_message,
                field=field,
            )
        return None, None

    try:
        candidate_real = _resolve_allowed_request_path(text, allowed_roots)
    except _PortalValidationReasonError:
        return None, _readiness_issue(
            "unsafe_path",
            severity="blocked",
            message=f"{field} must stay within allowed roots.",
            field=field,
            path=text,
        )
    except (OSError, RuntimeError, ValueError):
        return None, _readiness_issue(
            "unsafe_path",
            severity="blocked",
            message=f"{field} must stay within allowed roots.",
            field=field,
            path=text,
        )

    resolved = str(candidate_real)
    trusted_entry = _trusted_allowed_entry(candidate_real, allowed_roots)
    if trusted_entry is None:
        return resolved, _readiness_issue(
            missing_reason,
            severity="blocked",
            message=missing_message,
            field=field,
            path=resolved,
        )

    try:
        if expected_type == "file" and not trusted_entry.is_file():
            return resolved, _readiness_issue(
                missing_reason,
                severity="blocked",
                message=missing_message,
                field=field,
                path=resolved,
            )
        if expected_type == "dir" and not trusted_entry.is_dir():
            return resolved, _readiness_issue(
                missing_reason,
                severity="blocked",
                message=missing_message,
                field=field,
                path=resolved,
            )
    except OSError:
        return resolved, _readiness_issue(
            missing_reason,
            severity="blocked",
            message=missing_message,
            field=field,
            path=resolved,
        )
    return resolved, None


_archive_index_preflight_example = _portal_archive_index_preflight._archive_index_preflight_example
_archive_index_preflight_result = _portal_archive_index_preflight._archive_index_preflight_result
_archive_index_preflight_message = _portal_archive_index_preflight._archive_index_preflight_message
_archive_index_preflight_primary_reason = _portal_archive_index_preflight._archive_index_preflight_primary_reason
_archive_index_preflight_root_reason = _portal_archive_index_preflight._archive_index_preflight_root_reason
_is_drive_prefixed_relpath = _portal_archive_index_preflight._is_drive_prefixed_relpath
_archive_index_preflight_scan_mode = _portal_archive_index_preflight._archive_index_preflight_scan_mode
_archive_index_preflight_row_limit = _portal_archive_index_preflight._archive_index_preflight_row_limit
_copy_archive_index_preflight_result = _portal_archive_index_preflight._copy_archive_index_preflight_result
_archive_index_preflight_cache_key = _portal_archive_index_preflight._archive_index_preflight_cache_key


def _trusted_existing_entry_without_realpath(
    path_value: Any,
    allowed_roots: List[Path],
) -> Optional[Path]:
    return _portal_archive_index_preflight._trusted_existing_entry_without_realpath(
        path_value,
        allowed_roots,
        repo_root=REPO_ROOT,
    )


_validate_archive_index_relpath = _portal_archive_index_preflight._validate_archive_index_relpath


def _validate_archive_index_against_root(
    archive_index: Path,
    archive_root: Path,
    *,
    scan_mode: str = "full",
) -> Dict[str, Any]:
    return _portal_archive_index_preflight._validate_archive_index_against_root(
        archive_index,
        archive_root,
        scan_mode=scan_mode,
        allowed_path_roots=ALLOWED_PATH_ROOTS,
        allowed_input_roots=ALLOWED_INPUT_ROOTS,
        repo_root=REPO_ROOT,
        cache=_ARCHIVE_INDEX_PREFLIGHT_CACHE,
        cache_lock=_ARCHIVE_INDEX_PREFLIGHT_CACHE_LOCK,
        cache_max_entries=ARCHIVE_INDEX_PREFLIGHT_CACHE_MAX,
        relpath_validator=_validate_archive_index_relpath,
    )


def _lux_depth_readiness(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    runner_available = _lux_depth_runner_available()
    canary_runtime = _resolve_lux_depth_canary_runtime()
    canary_status = (
        "ready"
        if canary_runtime is not None
        else "degraded" if (_module_available("torch") and _module_available("transformers")) else "unavailable"
    )

    issues: List[Dict[str, Any]] = []
    notes = [
        "Base readiness covers runner invocation, path safety, and orchestrator preflight.",
        "Canary readiness is reported separately; selected DA3 dispatch requires a runnable DA3 runtime.",
    ]
    selected_model: Dict[str, Any] = {
        "backend": "",
        "model_key": "",
        "canonical_model_key": "",
        "repo_id": "",
        "license_id": "",
        "requires_non_commercial_ok": False,
        "runtime_available": None,
        "status": "not_selected",
    }
    if not runner_available:
        issues.append(
            _readiness_issue(
                "runner_unavailable",
                severity="blocked",
                message="Lux Depth runner module is not importable in the active environment.",
            )
        )
    if canary_status == "ready":
        notes.append(f"Repo-local DA3 canary runtime resolved at {canary_runtime}.")
    elif canary_status == "degraded":
        notes.append("ML libraries are present in the active environment, but no isolated DA3 runtime contract was found.")
    else:
        notes.append("No DA3 canary runtime contract was found; model execution remains optional and unverified.")

    if args is not None:
        selected_backend = _canonical_depth_backend(_pick(args, "depth_backend", "depthBackend", default="da3")) or "da3"
        if selected_backend == "da3":
            selected_model_key = _canonical_da3_model_key(
                _pick(args, "model_key", "modelKey", default=PORTAL_DEFAULT_DA3_MODEL_KEY)
            )
            selected_spec = _da3_model_spec_for_portal_key(selected_model_key)
            selected_model.update(
                {
                    "backend": "da3",
                    "model_key": selected_model_key,
                    "canonical_model_key": getattr(selected_spec, "key", "") if selected_spec is not None else "",
                    "repo_id": getattr(selected_spec, "repo_id", "") if selected_spec is not None else "",
                    "license_id": getattr(selected_spec, "license_id", "") if selected_spec is not None else "",
                    "requires_non_commercial_ok": (
                        bool(getattr(selected_spec, "requires_non_commercial_ok", False))
                        if selected_spec is not None
                        else False
                    ),
                    "runtime_available": canary_runtime is not None,
                    "status": "ready" if canary_runtime is not None and selected_spec is not None else "blocked",
                }
            )
            if selected_spec is None:
                issues.append(
                    _readiness_issue(
                        "invalid_model_key",
                        severity="blocked",
                        message="The selected DA3 model is not supported.",
                        field="model_key",
                    )
                )
            if (
                selected_spec is not None
                and bool(getattr(selected_spec, "requires_non_commercial_ok", False))
                and not _as_bool(
                    _pick(args, "non_commercial_ok", "nonCommercialOk", default=False),
                    default=False,
                )
            ):
                issues.append(
                    _readiness_issue(
                        "da3_model_non_commercial_required",
                        severity="blocked",
                        message="The selected DA3 research model requires a non-commercial acknowledgment.",
                        field="non_commercial_ok",
                    )
                )
            if canary_runtime is None:
                issues.append(
                    _readiness_issue(
                        "da3_runtime_unavailable",
                        severity="blocked",
                        message=(
                            "The selected DA3 backend requires the repo-local DA3 runtime before dispatch. "
                            "Run ./scripts/setup/install_da3_runtime.sh or set TRANSFORMATION_PORTAL_DA3_PYTHON."
                        ),
                        field="depth_backend",
                    )
                )
        elif selected_backend == "depth_pro":
            selected_model.update(
                {
                    "backend": "depth_pro",
                    "model_key": "apple/ml-depth-pro",
                    "canonical_model_key": "depth_pro",
                    "repo_id": "apple/ml-depth-pro",
                    "license_id": "apple-depth-pro-research",
                    "requires_non_commercial_ok": True,
                    "runtime_available": None,
                    "status": "research_ack_required",
                }
            )
        for field, keys, roots in (
            ("input_dir", ("input_dir", "inputDir"), ALLOWED_INPUT_ROOTS),
            ("output_dir", ("output_dir", "outputDir"), ALLOWED_OUTPUT_ROOTS),
        ):
            raw_value = _pick(args, *keys, default="")
            text = str(raw_value or "").strip()
            if not text:
                continue
            try:
                _validate_path_against_roots(text, roots)
            except ValueError:
                issues.append(
                    _readiness_issue(
                        "unsafe_path",
                        severity="blocked",
                        message=f"{field} must stay within allowed roots.",
                        field=field,
                        path=text,
                    )
                )

    status = "blocked" if any(item["severity"] == "blocked" for item in issues) else "ready"
    return {
        "status": status,
        "canonical_command": "lux-depth-v3",
        "missing_prerequisites": issues,
        "runner_details": {
            "type": "python_module",
            "available": runner_available,
            "module": LUX_DEPTH_MODULE,
            "command": _lux_depth_runner_command(),
            "python_executable": sys.executable,
        },
        "notes": notes,
        "canary_status": canary_status,
        "selected_model": selected_model,
    }


def _archive_gate_readiness(
    pipeline: str,
    args: Optional[Dict[str, Any]] = None,
    *,
    require_dispatch_inputs: bool,
    archive_index_scan_mode: str = "full",
) -> Dict[str, Any]:
    command = ARCHIVE_GATE_DEFAULT_COMMANDS[pipeline]
    if args:
        candidate = str(_pick(args, "archive_command", "archiveCommand", default=command) or "").strip()
        if candidate:
            command = candidate

    issues: List[Dict[str, Any]] = []
    notes: List[str] = []
    runner_available = ARCHIVE_GOVERNANCE_SCRIPT.is_file()
    if not runner_available:
        issues.append(
            _readiness_issue(
                "runner_unavailable",
                severity="blocked",
                message="Archive governance runner script is missing.",
            )
        )

    if command not in ARCHIVE_GATE_ALLOWED_COMMANDS[pipeline]:
        issues.append(
            _readiness_issue(
                "invalid_archive_command",
                severity="blocked",
                message=f"{command!r} is not allowed for {pipeline}.",
                field="archive_command",
            )
        )

    def _append_issue(issue: Optional[Dict[str, Any]]) -> None:
        if issue is not None:
            issues.append(issue)

    if args is not None:
        for field, keys, roots in (
            ("input_dir", ("input_dir", "inputDir"), ALLOWED_INPUT_ROOTS),
            ("output_dir", ("output_dir", "outputDir"), ALLOWED_OUTPUT_ROOTS),
        ):
            raw_value = _pick(args, *keys, default="")
            text = str(raw_value or "").strip()
            if not text:
                continue
            try:
                _validate_path_against_roots(text, roots)
            except ValueError:
                issues.append(
                    _readiness_issue(
                        "unsafe_path",
                        severity="blocked",
                        message=f"{field} must stay within allowed roots.",
                        field=field,
                        path=text,
                    )
                )

    if pipeline == "archive-gate-a":
        notes.append("Canonical archive-gate-a dispatch expects fixity-scan with an existing archive index.")
        if command == "fixity-scan":
            archive_index_value = _pick(args or {}, "archive_index", "archiveIndex", default="")
            if require_dispatch_inputs:
                archive_index_path, issue = _validate_existing_path(
                    archive_index_value,
                    field="archive_index",
                    allowed_roots=ALLOWED_PATH_ROOTS,
                    missing_reason="archive_index_required",
                    missing_message="Provide an existing archive index before dispatch.",
                    expected_type="file",
                    required=True,
                )
                _append_issue(issue)
                archive_root_value = _pick(
                    args or {},
                    "archive_root",
                    "archiveRoot",
                    default=_pick(args or {}, "input_dir", "inputDir", default=""),
                )
                archive_root_field = (
                    "archive_root"
                    if _pick(args or {}, "archive_root", "archiveRoot", default=None) is not None
                    else "input_dir"
                )
                archive_root_path, archive_root_issue = _validate_existing_path(
                    archive_root_value,
                    field=archive_root_field,
                    allowed_roots=ALLOWED_INPUT_ROOTS,
                    missing_reason="input_dir_required",
                    missing_message="Provide an existing archive root before dispatch.",
                    expected_type="dir",
                    required=True,
                )
                _append_issue(archive_root_issue)
                if archive_index_path and archive_root_path and issue is None and archive_root_issue is None:
                    index_preflight = _validate_archive_index_against_root(
                        Path(str(archive_index_value)),
                        Path(str(archive_root_value)),
                        scan_mode=archive_index_scan_mode,
                    )
                    root_reason = _archive_index_preflight_root_reason(index_preflight)
                    if root_reason is not None:
                        issues.append(
                            _readiness_issue(
                                "input_dir_required" if root_reason != "archive_root_symlink" else "unsafe_path",
                                severity="blocked",
                                message=(
                                    "Archive root must be an existing directory before dispatch."
                                    if root_reason != "archive_root_symlink"
                                    else "Archive root must be a real directory, not a symlink."
                                ),
                                field=archive_root_field,
                                path=archive_root_path,
                            )
                        )
                    elif not index_preflight["ok"]:
                        issues.append(
                            _readiness_issue(
                                "archive_index_root_mismatch",
                                severity="blocked",
                                message=_archive_index_preflight_message(index_preflight),
                                field="archive_index",
                                path=archive_index_path,
                            )
                        )
            else:
                issues.append(
                    _readiness_issue(
                        "archive_index_required",
                        severity="degraded",
                        message="An existing archive index is required at dispatch time.",
                        field="archive_index",
                    )
                )
        elif require_dispatch_inputs and command == "fixity-verify":
            _, issue = _validate_existing_path(
                _pick(args or {}, "hash_manifest", "hashManifest", default=""),
                field="hash_manifest",
                allowed_roots=ALLOWED_OUTPUT_ROOTS,
                missing_reason="hash_manifest_required",
                missing_message="Provide an existing hash manifest before dispatch.",
                expected_type="file",
                required=True,
            )
            _append_issue(issue)
        elif require_dispatch_inputs and command == "manifest-build":
            _, issue = _validate_existing_path(
                _pick(args or {}, "archive_index", "archiveIndex", default=""),
                field="archive_index",
                allowed_roots=ALLOWED_PATH_ROOTS,
                missing_reason="archive_index_required",
                missing_message="Provide an existing archive index before dispatch.",
                expected_type="file",
                required=True,
            )
            _append_issue(issue)
            _, issue = _validate_existing_path(
                _pick(args or {}, "hash_manifest", "hashManifest", default=""),
                field="hash_manifest",
                allowed_roots=ALLOWED_OUTPUT_ROOTS,
                missing_reason="hash_manifest_required",
                missing_message="Provide an existing hash manifest before dispatch.",
                expected_type="file",
                required=True,
            )
            _append_issue(issue)
        elif require_dispatch_inputs and command == "rights-apply":
            _, issue = _validate_existing_path(
                _pick(args or {}, "manifest_jsonl", "manifestJsonl", default=""),
                field="manifest_jsonl",
                allowed_roots=ALLOWED_OUTPUT_ROOTS,
                missing_reason="manifest_jsonl_required",
                missing_message="Provide an existing manifest JSONL before dispatch.",
                expected_type="file",
                required=True,
            )
            _append_issue(issue)
            _, issue = _validate_existing_path(
                _pick(args or {}, "policy_yaml", "policyYaml", default=""),
                field="policy_yaml",
                allowed_roots=ALLOWED_INPUT_ROOTS,
                missing_reason="policy_yaml_required",
                missing_message="Provide an existing rights policy YAML before dispatch.",
                expected_type="file",
                required=True,
            )
            _append_issue(issue)
    elif pipeline == "archive-gate-b":
        notes.append("Canonical dispatch for this archive stage requires a prior rights-manifest artifact.")
        if command == "bag-build":
            if require_dispatch_inputs:
                _, issue = _validate_existing_path(
                    _pick(args or {}, "manifest_jsonl", "manifestJsonl", default=""),
                    field="manifest_jsonl",
                    allowed_roots=ALLOWED_OUTPUT_ROOTS,
                    missing_reason="rights_manifest_required",
                    missing_message="Provide an existing rights-manifest JSONL artifact before dispatch.",
                    expected_type="file",
                    required=True,
                )
                _append_issue(issue)
            else:
                issues.append(
                    _readiness_issue(
                        "rights_manifest_required",
                        severity="blocked",
                        message="A rights-manifest JSONL artifact from a prior archive stage is required.",
                        field="manifest_jsonl",
                    )
                )
        elif require_dispatch_inputs and command == "bag-validate":
            _, issue = _validate_existing_path(
                _pick(args or {}, "bag_dir", "bagDir", default=""),
                field="bag_dir",
                allowed_roots=ALLOWED_OUTPUT_ROOTS,
                missing_reason="bag_dir_required",
                missing_message="Provide an existing bag directory before dispatch.",
                expected_type="dir",
                required=True,
            )
            _append_issue(issue)
        elif require_dispatch_inputs and command == "dedup-plan":
            _, issue = _validate_existing_path(
                _pick(args or {}, "manifest_jsonl", "manifestJsonl", default=""),
                field="manifest_jsonl",
                allowed_roots=ALLOWED_OUTPUT_ROOTS,
                missing_reason="rights_manifest_required",
                missing_message="Provide an existing rights-manifest JSONL artifact before dispatch.",
                expected_type="file",
                required=True,
            )
            _append_issue(issue)
    else:
        notes.append("Canonical dispatch for this archive stage requires a prior rights-manifest artifact.")
        if command == "mets-export":
            if require_dispatch_inputs:
                _, issue = _validate_existing_path(
                    _pick(args or {}, "manifest_jsonl", "manifestJsonl", default=""),
                    field="manifest_jsonl",
                    allowed_roots=ALLOWED_OUTPUT_ROOTS,
                    missing_reason="rights_manifest_required",
                    missing_message="Provide an existing rights-manifest JSONL artifact before dispatch.",
                    expected_type="file",
                    required=True,
                )
                _append_issue(issue)
            else:
                issues.append(
                    _readiness_issue(
                        "rights_manifest_required",
                        severity="blocked",
                        message="A rights-manifest JSONL artifact from a prior archive stage is required.",
                        field="manifest_jsonl",
                    )
                )
        elif require_dispatch_inputs and command in {"prov-export", "stac-export"}:
            _, issue = _validate_existing_path(
                _pick(args or {}, "manifest_jsonl", "manifestJsonl", default=""),
                field="manifest_jsonl",
                allowed_roots=ALLOWED_OUTPUT_ROOTS,
                missing_reason="rights_manifest_required",
                missing_message="Provide an existing rights-manifest JSONL artifact before dispatch.",
                expected_type="file",
                required=True,
            )
            _append_issue(issue)

    blocked = any(item["severity"] == "blocked" for item in issues)
    degraded = any(item["severity"] == "degraded" for item in issues)
    status = "blocked" if blocked else "degraded" if degraded else "ready"
    return {
        "status": status,
        "canonical_command": ARCHIVE_GATE_DEFAULT_COMMANDS[pipeline],
        "missing_prerequisites": issues,
        "runner_details": {
            "type": "python_script",
            "available": runner_available,
            "script_path": str(ARCHIVE_GOVERNANCE_SCRIPT),
            "python_executable": sys.executable,
            "command": [sys.executable, str(ARCHIVE_GOVERNANCE_SCRIPT), "--json", command],
        },
        "notes": notes,
    }


def _evaluate_pipeline_readiness(
    pipeline: str,
    args: Optional[Dict[str, Any]] = None,
    *,
    require_dispatch_inputs: bool = False,
    archive_index_scan_mode: str = "full",
) -> Dict[str, Any]:
    if pipeline not in ALLOWED_PIPELINES:
        return {
            "status": "blocked",
            "canonical_command": "",
            "missing_prerequisites": [
                _readiness_issue(
                    "unsupported_pipeline",
                    severity="blocked",
                    message=f"Unsupported pipeline {pipeline!r}.",
                    field="pipeline",
                )
            ],
            "runner_details": {},
            "notes": [],
        }

    normalized_args = args
    normalization_errors: List[Dict[str, Any]] = []
    if isinstance(args, dict):
        normalized_args, _, normalization_errors = _normalize_operator_payload_paths(pipeline, args)

    if pipeline == "lux-depth-v3":
        readiness_payload = _lux_depth_readiness(normalized_args)
    else:
        readiness_payload = _archive_gate_readiness(
            pipeline,
            normalized_args,
            require_dispatch_inputs=require_dispatch_inputs,
            archive_index_scan_mode=archive_index_scan_mode,
        )

    if normalization_errors:
        synthesized = [
            _readiness_issue(
                _portal_reason_code(issue.get("code")),
                severity="blocked",
                message=str(issue.get("message") or "Configured path shorthand is invalid."),
                field=str(issue.get("field") or "payload"),
            )
            for issue in normalization_errors
        ]
        readiness_payload["missing_prerequisites"] = synthesized + list(readiness_payload.get("missing_prerequisites") or [])
        readiness_payload["status"] = "blocked"
    return readiness_payload


def _enforce_job_readiness_preflight(
    pipeline: str,
    readiness_snapshot: Dict[str, Any],
) -> None:
    if readiness_snapshot.get("status") != "blocked":
        return

    issues = readiness_snapshot.get("missing_prerequisites") or []
    first_issue = issues[0] if issues else {}
    raise JobPreflightError(
        str(first_issue.get("reason") or "invalid_request"),
        field=str(first_issue.get("field") or "payload"),
        message=str(first_issue.get("message") or "job blocked by readiness preflight"),
        status_code=400,
        extra={"pipeline": pipeline},
    )


def _enforce_dispatch_value_preflight(
    pipeline: str,
    execution_args: Dict[str, Any],
) -> None:
    """Validate enum-style fields at dispatch to match preview validation.

    Preview already rejects invalid preset / depth_device / run_card_version,
    but dispatch historically trusted execution_args and happily forwarded
    arbitrary strings to the runner. This closes that gap so preview and
    dispatch reject the same inputs.
    """

    if pipeline == "lux-depth-v3":
        allowed_presets = _allowed_preset_names(pipeline)
        preset_value = str(_pick(execution_args, "preset", default="") or "").strip()
        if preset_value and allowed_presets and preset_value not in allowed_presets:
            raise JobPreflightError(
                "invalid_preset",
                field="preset",
                message="The selected preset is not supported.",
                status_code=400,
                extra={"pipeline": pipeline},
            )

        device_value = str(_pick(execution_args, "depth_device", "depthDevice", default="") or "").strip().lower()
        if device_value and device_value not in ALLOWED_DEPTH_DEVICES:
            raise JobPreflightError(
                "invalid_depth_device",
                field="depth_device",
                message="The selected compute device is not supported.",
                status_code=400,
                extra={"pipeline": pipeline},
            )

        run_card_value = str(_pick(execution_args, "run_card_version", "runCardVersion", default="") or "").strip().lower()
        if run_card_value and run_card_value not in ALLOWED_RUN_CARD_VERSIONS:
            raise JobPreflightError(
                "invalid_run_card_version",
                field="run_card_version",
                message="Run card version must be v1 or v2.",
                status_code=400,
                extra={"pipeline": pipeline},
            )


def _enforce_dispatch_filesystem_preflight(
    pipeline: str,
    execution_args: Dict[str, Any],
) -> Optional[Path]:
    """Read-only validation of dispatch filesystem inputs.

    Verifies that ``input_dir`` exists inside an allowed root and returns the
    trusted ``output_dir`` Path that the caller should ``mkdir`` *after* the
    job admission gate succeeds (so a 429 reject does not leave behind empty
    directories under load). Each path component is walked via
    :func:`_trusted_existing_dir` / :func:`_trusted_creatable_dir`, which
    iterate ``Path.iterdir()`` rather than passing user-controlled strings to
    filesystem APIs — this also satisfies the CodeQL ``py/path-injection``
    detector.

    Raises :class:`JobPreflightError` with a portal-safe reason on failure.
    """

    input_dir_raw = str(_pick(execution_args, "input_dir", "inputDir", default="")).strip()
    if input_dir_raw:
        input_dir_trusted = _trusted_existing_dir(input_dir_raw, ALLOWED_INPUT_ROOTS)
        if input_dir_trusted is None:
            raise JobPreflightError(
                "input_dir_required",
                field="input_dir",
                message="An existing input directory is required before dispatch.",
                status_code=400,
                extra={"pipeline": pipeline},
            )

    output_dir_raw = str(_pick(execution_args, "output_dir", "outputDir", default="")).strip()
    if not output_dir_raw:
        return None

    output_dir_trusted = _trusted_creatable_dir(output_dir_raw, ALLOWED_OUTPUT_ROOTS)
    if output_dir_trusted is None:
        raise JobPreflightError(
            "output_dir_unwritable",
            field="output_dir",
            message="The output directory or its parent is not writable.",
            status_code=400,
            extra={"pipeline": pipeline},
        )
    # Confirm the deepest existing ancestor is writable without creating
    # anything yet; the actual mkdir happens after admission succeeds.
    existing_ancestor = output_dir_trusted
    while not existing_ancestor.exists():
        parent = existing_ancestor.parent
        if parent == existing_ancestor:
            break
        existing_ancestor = parent
    if not os.access(str(existing_ancestor), os.W_OK):
        raise JobPreflightError(
            "output_dir_unwritable",
            field="output_dir",
            message="The output directory or its parent is not writable.",
            status_code=400,
            extra={"pipeline": pipeline},
        )
    return output_dir_trusted


def _materialize_dispatch_output_dir(
    pipeline: str,
    output_dir_trusted: Optional[Path],
) -> None:
    """Create the trusted output directory after job admission succeeds.

    Splitting mkdir from preflight prevents accumulation of empty directories
    when dispatch is rejected post-preflight (e.g. by the concurrency gate).
    The path here was validated component-by-component via iterdir(); only
    the final ``mkdir`` is necessary to materialise it.
    """

    if output_dir_trusted is None:
        return
    try:
        output_dir_trusted.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise JobPreflightError(
            "output_dir_unwritable",
            field="output_dir",
            message="The output directory or its parent is not writable.",
            status_code=400,
            extra={"pipeline": pipeline},
        ) from None


HTTP_STATUS_ERROR_CODES = {
    400: "INVALID_ARGUMENT",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    413: "REQUEST_TOO_LARGE",
    429: "RATE_LIMITED",
    500: "INTERNAL_ERROR",
    503: "SERVICE_UNAVAILABLE",
}

PUBLIC_HTTP_ERROR_MESSAGES = {
    400: "invalid request",
    401: "unauthorized",
    403: "forbidden",
    404: "not found",
    405: "method not allowed",
    413: "request body too large",
    429: "rate limit exceeded",
    500: "internal server error",
    503: "service unavailable",
}


def _is_versioned_api_path(path: str) -> bool:
    return path.startswith(("/v1/", "/v2/"))


def _http_status_error_code(status_code: int) -> str:
    return HTTP_STATUS_ERROR_CODES.get(status_code, "HTTP_ERROR")


def _is_upload_staging_endpoint(path: str) -> bool:
    return (path.rstrip("/") or "/") == "/v1/uploads/staging"


def _request_body_limit_bytes(path: str) -> int:
    return MAX_UPLOAD_REQUEST_BYTES if _is_upload_staging_endpoint(path) else MAX_REQUEST_BYTES


def _request_too_large_message(path: str) -> str:
    return f"request body too large (max {_request_body_limit_bytes(path)} bytes)"


@dataclass
class _ParsedPortalUploadPayload:
    uploads: List[IncomingUpload]
    client_manifest_raw: Optional[str]

    def close(self) -> None:
        for upload in self.uploads:
            close = getattr(upload.stream, "close", None)
            if callable(close):
                close()


@dataclass
class _PortalMultipartPart:
    field_name: str
    filename: Optional[str]
    content_type: str
    charset: str
    file_stream: Any = None
    value_bytes: bytearray = dataclass_field(default_factory=bytearray)
    size_bytes: int = 0

    def close(self) -> None:
        if self.file_stream is None:
            return
        self.file_stream.close()


def _portal_upload_boundary_bytes(content_type: str) -> bytes:
    header_message = BytesParser(policy=email_policy).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
    )
    if header_message.get_content_type() != "multipart/form-data":
        raise UploadStagingError(
            "invalid_content_type",
            "staged uploads require multipart/form-data",
            field="content-type",
        )
    boundary = str(header_message.get_param("boundary", header="content-type") or "").strip()
    if not boundary:
        raise UploadStagingError(
            "invalid_content_type",
            "multipart/form-data boundary is required",
            field="content-type",
        )
    return boundary.encode("utf-8")


def _parse_portal_upload_part_headers(header_bytes: bytes) -> _PortalMultipartPart:
    header_message = BytesParser(policy=email_policy).parsebytes(header_bytes + b"\r\n\r\n")
    if header_message.get_content_disposition() != "form-data":
        raise UploadStagingError(
            "invalid_multipart_payload",
            "multipart parts must use form-data content disposition",
        )
    field_name = str(header_message.get_param("name", header="content-disposition") or "").strip()
    if not field_name:
        raise UploadStagingError(
            "invalid_multipart_payload",
            "multipart field name is required",
        )
    filename = header_message.get_filename()
    if filename is not None:
        return _PortalMultipartPart(
            field_name=field_name,
            filename=str(filename or ""),
            content_type=str(header_message.get_content_type() or ""),
            charset=str(header_message.get_content_charset("utf-8") or "utf-8"),
            file_stream=tempfile.SpooledTemporaryFile(
                max_size=min(PORTAL_UPLOAD_MAX_PART_BYTES, 1024 * 1024),
                mode="w+b",
            ),
        )
    return _PortalMultipartPart(
        field_name=field_name,
        filename=None,
        content_type=str(header_message.get_content_type() or ""),
        charset=str(header_message.get_content_charset("utf-8") or "utf-8"),
    )


def _write_portal_multipart_part_content(part: _PortalMultipartPart, payload: bytes) -> None:
    if not payload:
        return
    part.size_bytes += len(payload)
    if part.size_bytes > PORTAL_UPLOAD_MAX_PART_BYTES:
        raise UploadStagingError(
            "multipart_part_too_large",
            "multipart field exceeds the per-part size limit",
            field=part.field_name,
            status_code=413,
        )
    if part.file_stream is not None:
        part.file_stream.write(payload)
        return
    part.value_bytes.extend(payload)


def _finalize_portal_multipart_part(
    part: _PortalMultipartPart,
    *,
    uploads: List[IncomingUpload],
    client_manifest: Optional[str],
    field_count: int,
) -> tuple[Optional[str], int]:
    if part.filename is not None:
        if part.field_name != "files":
            raise UploadStagingError(
                "unexpected_field",
                "unexpected multipart field in staged upload payload",
                field=part.field_name,
            )
        part.file_stream.seek(0)
        uploads.append(
            IncomingUpload(
                filename=part.filename,
                stream=part.file_stream,
                content_type=part.content_type,
            )
        )
        if len(uploads) > PORTAL_UPLOAD_MAX_FILES:
            raise UploadStagingError(
                "too_many_files",
                "too many upload files in staged upload payload",
                field="files",
            )
        part.file_stream = None
        return client_manifest, field_count

    field_count += 1
    if field_count > PORTAL_UPLOAD_MAX_FIELDS:
        raise UploadStagingError(
            "too_many_fields",
            "too many form fields in staged upload payload",
        )
    if part.field_name != "client_manifest":
        raise UploadStagingError(
            "unexpected_field",
            "unexpected multipart field in staged upload payload",
            field=part.field_name,
        )
    if client_manifest is not None:
        raise UploadStagingError(
            "duplicate_client_manifest",
            "client_manifest must be provided at most once",
            field="client_manifest",
        )
    try:
        client_manifest = bytes(part.value_bytes).decode(part.charset)
    except (LookupError, UnicodeDecodeError) as exc:
        raise UploadStagingError(
            "invalid_client_manifest",
            "client_manifest must be valid UTF-8 JSON",
            field="client_manifest",
        ) from exc
    return client_manifest, field_count


async def _parse_portal_upload_multipart(request: Request) -> _ParsedPortalUploadPayload:
    boundary = _portal_upload_boundary_bytes(request.headers.get("content-type", ""))
    opening_boundary = b"--" + boundary
    body_boundary = b"\r\n" + opening_boundary
    buffer = bytearray()
    state = "preamble"
    current_part: Optional[_PortalMultipartPart] = None
    uploads: List[IncomingUpload] = []
    client_manifest: Optional[str] = None
    field_count = 0

    def _cleanup() -> None:
        nonlocal current_part
        if current_part is not None:
            current_part.close()
            current_part = None
        for upload in uploads:
            close = getattr(upload.stream, "close", None)
            if callable(close):
                close()
        uploads.clear()

    def _consume_after_boundary(*, eof: bool) -> bool:
        nonlocal state
        if len(buffer) < 2:
            if eof:
                raise UploadStagingError(
                    "invalid_multipart_payload",
                    "multipart payload terminated unexpectedly",
                )
            return False
        if buffer[:2] == b"--":
            del buffer[:2]
            if buffer[:2] == b"\r\n":
                del buffer[:2]
            if buffer and bytes(buffer).strip():
                raise UploadStagingError(
                    "invalid_multipart_payload",
                    "multipart payload contains unexpected trailing bytes",
                )
            state = "done"
            return False
        if buffer[:2] != b"\r\n":
            raise UploadStagingError(
                "invalid_multipart_payload",
                "multipart payload uses an invalid boundary separator",
            )
        del buffer[:2]
        state = "headers"
        return True

    def _advance(*, eof: bool) -> None:
        nonlocal state, current_part, client_manifest, field_count
        while True:
            if state == "done":
                if buffer and bytes(buffer).strip():
                    raise UploadStagingError(
                        "invalid_multipart_payload",
                        "multipart payload contains unexpected trailing bytes",
                    )
                return

            if state == "preamble":
                boundary_index = buffer.find(opening_boundary)
                if boundary_index < 0:
                    if eof:
                        raise UploadStagingError(
                            "invalid_multipart_payload",
                            "staged upload payload must be a valid multipart form submission",
                        )
                    trim_bytes = max(0, len(buffer) - (len(opening_boundary) + 4))
                    if trim_bytes:
                        del buffer[:trim_bytes]
                    return
                del buffer[: boundary_index + len(opening_boundary)]
                state = "after_boundary"
                continue

            if state == "after_boundary":
                if not _consume_after_boundary(eof=eof):
                    return
                continue

            if state == "headers":
                header_index = buffer.find(b"\r\n\r\n")
                if header_index < 0:
                    if eof:
                        raise UploadStagingError(
                            "invalid_multipart_payload",
                            "multipart part headers terminated unexpectedly",
                        )
                    return
                current_part = _parse_portal_upload_part_headers(bytes(buffer[:header_index]))
                del buffer[: header_index + 4]
                state = "body"
                continue

            if state == "body":
                assert current_part is not None
                boundary_index = buffer.find(body_boundary)
                if boundary_index < 0:
                    if eof:
                        raise UploadStagingError(
                            "invalid_multipart_payload",
                            "multipart payload terminated before the closing boundary",
                        )
                    flush_limit = len(buffer) - (len(body_boundary) + 4)
                    if flush_limit > 0:
                        _write_portal_multipart_part_content(current_part, bytes(buffer[:flush_limit]))
                        del buffer[:flush_limit]
                    return

                _write_portal_multipart_part_content(current_part, bytes(buffer[:boundary_index]))
                del buffer[: boundary_index + len(body_boundary)]
                client_manifest, field_count = _finalize_portal_multipart_part(
                    current_part,
                    uploads=uploads,
                    client_manifest=client_manifest,
                    field_count=field_count,
                )
                current_part = None
                state = "after_boundary"
                continue

            raise UploadStagingError(
                "invalid_multipart_payload",
                "multipart payload entered an invalid parser state",
            )

    try:
        async for chunk in request.stream():
            if chunk:
                buffer.extend(chunk)
            _advance(eof=False)
        _advance(eof=True)
    except Exception:
        _cleanup()
        raise

    return _ParsedPortalUploadPayload(uploads=uploads, client_manifest_raw=client_manifest)


def _public_http_error_message(status_code: int, path: str = "") -> str:
    if status_code == 413:
        return _request_too_large_message(path)
    return PUBLIC_HTTP_ERROR_MESSAGES.get(status_code, "request failed")


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
    return sum(1 for job in JOBS.values() if job.state in ACTIVE_JOB_STATES)


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


def _retained_staged_input_dirs() -> set[str]:
    retained: set[str] = set()
    for job in JOBS.values():
        effective_request = (
            job.effective_request if isinstance(job.effective_request, dict) and job.effective_request else job.request
        )
        if not isinstance(effective_request, dict):
            continue
        args = effective_request.get("args")
        if not isinstance(args, dict):
            continue
        input_dir = args.get("input_dir") or args.get("inputDir")
        if not input_dir:
            continue
        try:
            retained.add(str(Path(os.path.realpath(str(input_dir)))))
        except (OSError, RuntimeError, ValueError):
            continue

    return retained


def _cleanup_expired_upload_batches(now: float) -> None:
    try:
        upload_root = _resolved_portal_upload_root()
    except _PortalValidationReasonError:
        LOGGER.warning("Skipping staged upload cleanup because TP_PORTAL_UPLOAD_ROOT is outside allowed input roots")
        return

    removed = cleanup_expired_batches(
        upload_root,
        now=now,
        ttl_seconds=PORTAL_UPLOAD_TTL_SECONDS,
        retained_input_dirs=_retained_staged_input_dirs(),
    )
    if removed:
        LOGGER.info("Removed %d expired staged upload batches", len(removed))


async def _cleanup_loop() -> None:
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        now = _now()
        _cleanup_expired_jobs(now)
        _cleanup_expired_upload_batches(now)
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


# Secret-shaped substrings we refuse to persist in runner logs or forward over
# SSE. These are intentionally conservative: the runner environment is already
# sanitized, but third-party tools and misconfigured callers can still echo
# credentials into stdout (e.g. HTTP tracing, library debug output).
_LOG_REDACT_KV_KEYS = (
    "api_key",
    "api-key",
    "apikey",
    "access_token",
    "access-token",
    "refresh_token",
    "refresh-token",
    "authorization",
    "auth_token",
    "auth-token",
    "token",
    "secret",
    "password",
    "passwd",
    "private_key",
    "private-key",
    "client_secret",
    "client-secret",
    "session_token",
    "session-token",
    "aws_secret_access_key",
    "aws-secret-access-key",
)

_LOG_REDACTION_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"(?i)(authorization|proxy-authorization)\s*:\s*" r"(?:bearer|basic|digest|token|apikey)\s+\S+"),
        r"\1: <redacted>",
    ),
    (
        re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+"),
        "Bearer <redacted>",
    ),
    (
        re.compile(
            r"(?i)(^|[^A-Za-z0-9])(" + "|".join(re.escape(key) for key in _LOG_REDACT_KV_KEYS) + r")(\s*[:=]\s*)([^\s,;]+)"
        ),
        r"\1\2\3<redacted>",
    ),
)


def _redact_log_line(line: str) -> str:
    if not line:
        return line
    redacted = line
    for pattern, replacement in _LOG_REDACTION_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _canonical_depth_backend(value: Any) -> str:
    backend = str(value or "").strip().lower()
    if not backend:
        return ""
    return DEPTH_BACKEND_ALIASES.get(backend, backend)


def _portal_da3_model_key_for_registry_key(registry_key: str) -> str:
    return PORTAL_DA3_MODEL_KEY_BY_REGISTRY_KEY.get(registry_key, registry_key)


def _canonical_da3_model_key(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return PORTAL_DEFAULT_DA3_MODEL_KEY
    registry_key = resolve_registry_key(raw)
    if registry_key is None or registry_key not in PORTAL_DA3_MODEL_KEY_BY_REGISTRY_KEY:
        return ""
    return _portal_da3_model_key_for_registry_key(registry_key)


def _da3_model_spec_for_portal_key(model_key: str) -> Optional[Any]:
    if not model_key:
        return None
    spec = resolve_model_spec(model_key)
    if spec is None or getattr(spec, "family", "") != "da3":
        return None
    return spec


def _lux_da3_model_options() -> List[Dict[str, Any]]:
    options: List[Dict[str, Any]] = []
    for spec in visible_cli_model_specs():
        if getattr(spec, "family", "") != "da3":
            continue
        value = _portal_da3_model_key_for_registry_key(str(spec.key))
        usage_class = getattr(getattr(spec, "usage_class", None), "value", getattr(spec, "usage_class", "unknown"))
        requires_non_commercial = bool(getattr(spec, "requires_non_commercial_ok", False))
        options.append(
            {
                "value": value,
                "canonical_key": spec.key,
                "label": "DA3 Research" if requires_non_commercial else "DA3 Metric",
                "repo_id": spec.repo_id,
                "license_id": spec.license_id,
                "usage_class": str(usage_class),
                "requires_non_commercial_ok": requires_non_commercial,
                "policy_posture": "research_only" if requires_non_commercial else "commercial_ok",
            }
        )
    return options


def _preset_descriptor(pipeline: str, preset_name: str) -> Optional[Dict[str, Any]]:
    presets = PRESET_CATALOG.get(pipeline) or []
    for preset in presets:
        if str(preset.get("name") or "") == str(preset_name or ""):
            return preset
    return None


def _allowed_preset_names(pipeline: str) -> set[str]:
    presets = PRESET_CATALOG.get(pipeline) or []
    names = {str(entry.get("name") or "").strip() for entry in presets if entry.get("name")}
    if pipeline == "lux-depth-v3":
        names.add("custom")
    return names


def _portal_issue(
    field: str,
    code: str,
    message: str,
    *,
    suggestion: str = "",
) -> Dict[str, Any]:
    issue = {
        "field": field,
        "code": code,
        "message": message,
    }
    if suggestion:
        issue["suggestion"] = suggestion
    return issue


def _portal_soft_cpu_worker_cap() -> int:
    return max(2, min(32, os.cpu_count() or 8))


def _portal_soft_gpu_worker_cap() -> int:
    return 4


def _portal_is_token(value: str) -> bool:
    return bool(PORTAL_EVENT_TOKEN_RE.fullmatch(str(value or "").strip().lower()))


def _portal_estimate_band(score: int) -> str:
    if score <= 1:
        return "low"
    if score == 2:
        return "medium"
    return "high"


def _portal_reason_code(value: Any) -> str:
    raw = str(value or "").strip()
    if raw in VALIDATION_REASON_CODES:
        return VALIDATION_REASON_CODES[raw]
    token = raw.lower()
    if _portal_is_token(token):
        return token
    return "invalid_request"


def _portal_safe_error_message(reason: str, *, field: str = "payload") -> str:
    return PORTAL_SAFE_ERROR_MESSAGES.get(reason, f"{field} contains invalid values.")


def _portal_issue_public_message(issue: Any, *, field: str = "payload") -> str:
    issue_dict = issue if isinstance(issue, dict) else {}
    reason = _portal_reason_code(issue_dict.get("code"))
    message = _portal_safe_error_message(reason, field=field)
    generic_message = f"{field} contains invalid values."
    if message != generic_message:
        return message

    issue_message = issue_dict.get("message")
    if not isinstance(issue_message, str):
        return message
    text = issue_message.strip()
    if not text or len(text) > 300:
        return message
    if "\n" in text or "\r" in text or "\x00" in text:
        return message
    return text


def _portal_next_best_action_label(field: Any, default: str) -> str:
    field_name = str(field or "").strip().replace("_", " ")
    if not field_name or field_name == "payload":
        return default
    return f"Resolve {field_name}"


def _portal_next_best_action_detail(issue: Mapping[str, Any], fallback: str) -> str:
    message = str(issue.get("message") or fallback).strip()
    suggestion = str(issue.get("suggestion") or "").strip()
    if suggestion and suggestion not in message:
        message = f"{message} {suggestion}".strip()
    return message or fallback


def _preview_next_best_action(
    *,
    pipeline: str,
    errors: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    readiness_snapshot: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    readiness_issues = (
        readiness_snapshot.get("missing_prerequisites")
        if isinstance(readiness_snapshot, Mapping) and isinstance(readiness_snapshot.get("missing_prerequisites"), list)
        else []
    )
    blocked_issue = next(
        (
            issue
            for issue in readiness_issues
            if isinstance(issue, Mapping) and str(issue.get("severity") or "").strip().lower() == "blocked"
        ),
        None,
    )
    readiness_warning = next(
        (
            issue
            for issue in readiness_issues
            if isinstance(issue, Mapping) and str(issue.get("severity") or "").strip().lower() != "blocked"
        ),
        None,
    )

    if errors:
        issue = errors[0] if isinstance(errors[0], Mapping) else {}
        field = str(issue.get("field") or "").strip()
        return {
            "action": "resolve_validation_error",
            "field": field,
            "label": _portal_next_best_action_label(field, "Resolve configuration issue"),
            "detail": _portal_next_best_action_detail(issue, "Resolve the current configuration issue before dispatch."),
            "tone": "blocked",
        }

    if blocked_issue is not None:
        field = str(blocked_issue.get("field") or "").strip()
        return {
            "action": "resolve_readiness",
            "field": field,
            "label": _portal_next_best_action_label(field, "Resolve readiness prerequisite"),
            "detail": _portal_next_best_action_detail(
                blocked_issue,
                "A dispatch prerequisite is still missing.",
            ),
            "tone": "blocked",
        }

    if warnings:
        issue = warnings[0] if isinstance(warnings[0], Mapping) else {}
        field = str(issue.get("field") or "").strip()
        return {
            "action": "review_warning",
            "field": field,
            "label": _portal_next_best_action_label(field, "Review warning before dispatch"),
            "detail": _portal_next_best_action_detail(issue, "Review the current warning before dispatch."),
            "tone": "warning",
        }

    if readiness_warning is not None:
        field = str(readiness_warning.get("field") or "").strip()
        return {
            "action": "review_readiness_warning",
            "field": field,
            "label": _portal_next_best_action_label(field, "Review readiness warning"),
            "detail": _portal_next_best_action_detail(
                readiness_warning,
                "Review the current readiness warning before dispatch.",
            ),
            "tone": "warning",
        }

    canonical_command = (
        str(readiness_snapshot.get("canonical_command") or "").strip() if isinstance(readiness_snapshot, Mapping) else ""
    )
    if pipeline == "lux-depth-v3":
        return {
            "action": "dispatch_ready",
            "field": "run_job",
            "label": "Execute the Lux run",
            "detail": "Preview-backed validation is ready. Review the expected outputs and dispatch when satisfied.",
            "tone": "ready",
        }
    command_detail = (
        f"Canonical command {canonical_command} is ready. Review the expected outputs and dispatch when satisfied."
        if canonical_command
        else "Archive readiness is clear. Review the expected outputs and dispatch when satisfied."
    )
    return {
        "action": "dispatch_ready",
        "field": "run_job",
        "label": "Dispatch the archive stage",
        "detail": command_detail,
        "tone": "ready",
    }


class _PortalValidationReasonError(ValueError):
    def __init__(self, message: str, *, reason: Optional[str] = None) -> None:
        cleaned_message = str(message or "").strip() or "invalid request"
        self.reason = _portal_reason_code(reason or cleaned_message)
        super().__init__(cleaned_message)


def _portal_reason_from_exception(exc: BaseException, *, default: str = "invalid_request") -> str:
    if isinstance(exc, _PortalValidationReasonError):
        return exc.reason
    return default


def _portal_payload_has_any_key(payload: Any, keys: Tuple[str, ...]) -> bool:
    if not isinstance(payload, dict):
        return False
    return any(key in payload for key in keys)


def _portal_inactive_reconstruction_field_value(
    request_args: Dict[str, Any],
    defaults: Dict[str, Any],
    field_name: str,
    value: Any,
) -> Optional[Dict[str, Any]]:
    if value in (None, "", False):
        return None
    aliases = LUX_RECONSTRUCTION_FIELD_ALIASES.get(field_name, (field_name,))
    explicit = _portal_payload_has_any_key(request_args, aliases)
    default_value = defaults.get(field_name)
    if not explicit and value == default_value:
        return None
    if explicit and value == default_value:
        return None
    return {
        "field": field_name,
        "value": value,
        "reason": "enable_reconstruction_disabled",
        "message": "Preserved for later, but ignored while reconstruction is off.",
    }


def _format_argv_preview(argv: List[str]) -> str:
    return " ".join(shlex.quote(str(token)) for token in argv)


def _lux_portal_defaults(args: Dict[str, Any]) -> Dict[str, Any]:
    preset_name = str(
        _pick(args, "preset", default=LUX_PORTAL_DEFAULT_ARGS["preset"]) or LUX_PORTAL_DEFAULT_ARGS["preset"]
    ).strip()
    defaults = dict(LUX_PORTAL_DEFAULT_ARGS)
    descriptor = _preset_descriptor("lux-depth-v3", preset_name)
    if descriptor is not None:
        recommended = descriptor.get("recommended_args")
        if isinstance(recommended, dict):
            defaults.update(recommended)
    defaults["preset"] = preset_name or str(defaults["preset"])
    return defaults


def _normalize_optional_positive_int(
    value: Any,
    field: str,
    errors: List[Dict[str, Any]],
) -> Optional[int]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        errors.append(
            _portal_issue(
                field,
                f"invalid_{field}",
                f"{field} must be an integer greater than or equal to 1.",
                suggestion="Use Auto or enter an integer greater than or equal to 1.",
            )
        )
        return None
    if parsed < 1:
        errors.append(
            _portal_issue(
                field,
                f"invalid_{field}",
                f"{field} must be greater than or equal to 1.",
                suggestion="Use Auto or enter an integer greater than or equal to 1.",
            )
        )
        return None
    return parsed


def _normalize_optional_non_negative_int(
    value: Any,
    field: str,
    errors: List[Dict[str, Any]],
) -> Optional[int]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        errors.append(
            _portal_issue(
                field,
                f"invalid_{field}",
                f"{field} must be an integer greater than or equal to 0.",
            )
        )
        return None
    if parsed < 0:
        errors.append(
            _portal_issue(
                field,
                f"invalid_{field}",
                f"{field} must be greater than or equal to 0.",
            )
        )
        return None
    return parsed


def _normalize_optional_probability(
    value: Any,
    field: str,
    errors: List[Dict[str, Any]],
) -> Optional[float]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        errors.append(
            _portal_issue(
                field,
                f"invalid_{field}",
                f"{field} must be a number in the range [0, 1].",
            )
        )
        return None
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        errors.append(
            _portal_issue(
                field,
                f"invalid_{field}",
                f"{field} must be in the range [0, 1].",
            )
        )
        return None
    return parsed


def _normalize_portal_path_arg(
    value: Any,
    field: str,
    allowed_roots: List[Path],
    errors: List[Dict[str, Any]],
    *,
    required: bool = False,
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
) -> str:
    text = str(value or "").strip()
    if not text:
        if required:
            errors.append(
                _portal_issue(
                    field,
                    "required",
                    f"{field} is required.",
                    suggestion=f"Provide a valid {field} within the allowed workspace roots.",
                )
            )
        return ""
    if text.startswith("~") or "\x00" in text:
        errors.append(
            _portal_issue(
                field,
                "invalid_path_value",
                f"{field} contains an invalid path value.",
                suggestion=(
                    f"Choose a valid {field} path under the configured repository or temp roots "
                    f"without invalid characters or path expansion syntax."
                ),
            )
        )
        return ""
    try:
        resolved_path = _resolve_allowed_request_path(text, allowed_roots)
    except ValueError as exc:
        reason = _portal_reason_from_exception(exc, default="invalid_path_value")
        del exc
        if reason == "path_outside_allowed_roots":
            errors.append(
                _portal_issue(
                    field,
                    "path_outside_allowed_roots",
                    f"{field} must stay within the allowed workspace roots.",
                    suggestion=f"Choose a {field} path under the configured repository or temp roots.",
                )
            )
        else:
            errors.append(
                _portal_issue(
                    field,
                    "invalid_path_value",
                    f"{field} contains an invalid path value.",
                    suggestion=(
                        f"Choose a valid {field} path under the configured repository or temp roots "
                        f"without invalid characters or path expansion syntax."
                    ),
                )
            )
        return ""
    trusted_entry: Optional[Path] = None
    if must_exist or must_be_file or must_be_dir:
        trusted_entry = _trusted_allowed_entry(resolved_path, allowed_roots)
    if must_exist and trusted_entry is None:
        errors.append(
            _portal_issue(
                field,
                "missing",
                f"{field} does not exist.",
                suggestion=f"Choose an existing {field} under the configured repository or temp roots.",
            )
        )
        return ""
    if must_be_file and trusted_entry is not None and not trusted_entry.is_file():
        errors.append(
            _portal_issue(
                field,
                "not_a_file",
                f"{field} must be a file.",
                suggestion=f"Choose a file path for {field} under the configured repository or temp roots.",
            )
        )
        return ""
    if must_be_dir and trusted_entry is not None and not trusted_entry.is_dir():
        errors.append(
            _portal_issue(
                field,
                "not_a_directory",
                f"{field} must be a directory.",
                suggestion=f"Choose a directory path for {field} under the configured repository or temp roots.",
            )
        )
        return ""
    display_path = trusted_entry if trusted_entry is not None else resolved_path
    return _normalize_repo_relative_display_path(text, display_path)


def _preview_path_errors_by_field(issues: Optional[List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for issue in issues or []:
        if not isinstance(issue, dict):
            continue
        field_name = str(issue.get("field") or "").strip()
        if not field_name:
            continue
        grouped.setdefault(field_name, []).append(issue)
    return grouped


def _normalize_preview_path_field(
    args: Dict[str, Any],
    field: str,
    keys: Tuple[str, ...],
    allowed_roots: List[Path],
    errors: List[Dict[str, Any]],
    path_errors_by_field: Dict[str, List[Dict[str, Any]]],
    *,
    required: bool = False,
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
) -> str:
    existing_errors = path_errors_by_field.get(field) or []
    if existing_errors:
        errors.extend(existing_errors)
        return ""
    return _normalize_portal_path_arg(
        _pick(args, *keys, default=""),
        field,
        allowed_roots,
        errors,
        required=required,
        must_exist=must_exist,
        must_be_file=must_be_file,
        must_be_dir=must_be_dir,
    )


def _lux_config_metadata() -> Dict[str, Any]:
    cpu_cap = _portal_soft_cpu_worker_cap()
    gpu_cap = _portal_soft_gpu_worker_cap()
    return {
        "pipeline": "lux-depth-v3",
        "advanced_sections": ["advanced", "governance", "reconstruction"],
        "estimate_bands": {
            "runtime": ["low", "medium", "high"],
            "gpu_pressure": ["low", "medium", "high"],
            "research_risk": ["none", "research_only", "experimental"],
        },
        "backend_catalog": {
            "da3": {
                "label": "DA3",
                "kind": "depth_backend",
                "operator_summary": (
                    "Default managed depth backend for standard Lux runs. The portal default uses "
                    "the Apache-2.0 DA3 metric model unless a research model is selected explicitly."
                ),
                "policy_posture": {
                    "code": "governed_default",
                    "label": "Governed default",
                    "detail": (
                        "Treat the selected DA3 model key and backend-owned release policy as the operator source of truth."
                    ),
                },
                "required_acknowledgments": [],
                "checkpoint_expectation": {
                    "required": False,
                    "field": None,
                    "detail": (
                        "Prefers the repo-local canary runtime when it is"
                        " available, but base readiness is evaluated"
                        " separately."
                    ),
                },
                "model_provider_label": "Depth Anything",
                "model_display_label": "Depth Anything v3",
                "default_model_key": PORTAL_DEFAULT_DA3_MODEL_KEY,
                "model_selector_field": "model_key",
            },
            "depth_pro": {
                "label": "Depth Pro",
                "kind": "depth_backend",
                "operator_summary": (
                    "Metric depth backend reserved for research-oriented runs" " and higher-cost validation."
                ),
                "policy_posture": {
                    "code": "research_only",
                    "label": "Research only",
                    "detail": (
                        "This backend stays behind explicit" " non-commercial and Apple research-license" " acknowledgments."
                    ),
                },
                "required_acknowledgments": [
                    {
                        "field": "non_commercial_ok",
                        "label": "Non-commercial acknowledgment",
                    },
                    {
                        "field": "accept_apple_depth_pro_research_license",
                        "label": "Apple Depth Pro research license",
                    },
                ],
                "checkpoint_expectation": {
                    "required": True,
                    "field": None,
                    "detail": ("Requires a local Depth Pro checkpoint in the active" " runtime before execution."),
                },
                "model_provider_label": "Apple",
                "model_display_label": "Depth Pro",
            },
            "efficientsam": {
                "label": "EfficientSAM",
                "kind": "segmentation_backend",
                "operator_summary": (
                    "Lighter segmentation backend for managed runs that need" " masks without the heaviest research posture."
                ),
                "policy_posture": {
                    "code": "managed_optional",
                    "label": "Managed optional",
                    "detail": (
                        "Suitable for standard segmentation coverage when the" " run contract does not require the SAM2 path."
                    ),
                },
                "required_acknowledgments": [],
                "checkpoint_expectation": {
                    "required": False,
                    "field": None,
                    "detail": "No explicit checkpoint path is required.",
                },
                "model_provider_label": "EfficientSAM",
                "model_display_label": "EfficientSAM",
            },
            "sam2": {
                "label": "SAM2",
                "kind": "segmentation_backend",
                "operator_summary": ("Highest-fidelity segmentation path for runs that need" " stronger scene coverage."),
                "policy_posture": {
                    "code": "experimental_segmentation",
                    "label": "Experimental",
                    "detail": (
                        "Use when the run benefits from deeper segmentation"
                        " coverage and the higher runtime cost is acceptable."
                    ),
                },
                "required_acknowledgments": [],
                "checkpoint_expectation": {
                    "required": False,
                    "field": "sam2_checkpoint_path",
                    "detail": (
                        "A local checkpoint path is optional. Managed flows only"
                        " accept repo-controlled SAM2 paths or files whose"
                        " checksum matches the governed SAM2 manifest."
                    ),
                },
                "model_provider_label": "Meta",
                "model_display_label": "SAM2 Hiera",
            },
            "stub": {
                "label": "Stub",
                "kind": "segmentation_backend",
                "operator_summary": ("Deterministic no-model fallback used for contract checks" " and low-risk iteration."),
                "policy_posture": {
                    "code": "deterministic_fallback",
                    "label": "Deterministic fallback",
                    "detail": ("Keeps segmentation semantics explicit without adding" " a model dependency."),
                },
                "required_acknowledgments": [],
                "checkpoint_expectation": {
                    "required": False,
                    "field": None,
                    "detail": "No checkpoint is used.",
                },
                "model_provider_label": "Built-in",
                "model_display_label": "Portal stub",
            },
        },
        "model_catalog": {
            "da3": _lux_da3_model_options(),
        },
        "debug_bundle_policy": {
            "acknowledgement_required": True,
            "destination_template": LUX_DEBUG_BUNDLE_DESTINATION_TEMPLATE,
            "includes": [
                "scene_manifest",
                "camera_payload",
                "input_image_copies",
                "segmentation_overlays",
                "reprojection_preview",
            ],
            "sensitivity": "camera_metadata_and_source_images",
        },
        "fields": {
            "model_key": {
                "label": "DA3 Model",
                "kind": "enum",
                "default": PORTAL_DEFAULT_DA3_MODEL_KEY,
                "helper_text": (
                    "Selects the DA3 registry model. DA3 Metric is Apache-2.0; DA3 Research requires "
                    "a non-commercial acknowledgment."
                ),
                "options": _lux_da3_model_options(),
            },
            "reconstruction_tier": {
                "label": "Reconstruction Tier",
                "kind": "enum",
                "default": "apex_research",
                "helper_text": "Selects the research reconstruction posture for the next run.",
                "options": [
                    {
                        "value": "apex_research",
                        "label": "APEX Research",
                        "description": "Balanced research reconstruction with moderate runtime.",
                        "runtime_band": "medium",
                        "research_risk": "research_only",
                    },
                    {
                        "value": "apex_research_ultra",
                        "label": "APEX Research Ultra",
                        "description": "Higher-cost research tier with heavier runtime pressure.",
                        "runtime_band": "high",
                        "research_risk": "research_only",
                    },
                    {
                        "value": "experimental",
                        "label": "Experimental",
                        "description": "Highest-risk research tier reserved for experimentation.",
                        "runtime_band": "high",
                        "research_risk": "experimental",
                    },
                ],
            },
            "reconstruction_iterations": {
                "label": "Reconstruction Iterations",
                "kind": "integer",
                "default": 1000,
                "min": 1,
                "recommended": {"fast": 250, "balanced": 1000, "high_quality": 2000},
                "warning_threshold": 2000,
                "helper_text": "Higher iterations improve optimization quality but increase runtime.",
            },
            "grouping_mode": {
                "label": "Grouping Mode",
                "kind": "enum",
                "default": "single",
                "options": [
                    {
                        "value": "single",
                        "label": "Single",
                        "description": "Treat the full input set as one scene.",
                    },
                    {
                        "value": "parent_dir",
                        "label": "Parent Directory",
                        "description": "Group images by parent directory for multi-view scenes.",
                    },
                ],
            },
            "raw_ingest_mode": {
                "label": "RAW Ingest Mode",
                "kind": "enum",
                "default": "auto",
                "options": [
                    {"value": "auto", "label": "Auto"},
                    {"value": "force_rawpy", "label": "Force rawpy"},
                    {"value": "force_preview", "label": "Force preview"},
                ],
                "helper_text": "Controls how RAW files are decoded before downstream stages.",
            },
            "raw_wb_mode": {
                "label": "RAW WB Mode",
                "kind": "locked",
                "default": "camera",
                "display_value": "camera",
                "helper_text": "The backend currently supports only camera white balance.",
            },
            "raw_demosaic": {
                "label": "RAW Demosaic",
                "kind": "locked",
                "default": "AHD",
                "display_value": "AHD",
                "helper_text": "The backend currently supports only AHD demosaic.",
            },
            "max_workers": {
                "label": "Max Workers",
                "kind": "optional_integer",
                "min": 1,
                "soft_max": cpu_cap,
                "default_mode": "auto",
                "helper_text": "Auto lets the runtime choose a safe CPU worker cap for the current environment.",
            },
            "max_gpu_workers": {
                "label": "Max GPU Workers",
                "kind": "optional_integer",
                "min": 1,
                "soft_max": gpu_cap,
                "default_mode": "auto",
                "helper_text": "Auto keeps GPU parallelism conservative to reduce VRAM contention.",
            },
            "log_level": {
                "label": "Log Level",
                "kind": "enum",
                "default": "",
                "options": [
                    {"value": "", "label": "Default"},
                    {"value": "DEBUG", "label": "DEBUG"},
                    {"value": "INFO", "label": "INFO"},
                    {"value": "WARNING", "label": "WARNING"},
                    {"value": "ERROR", "label": "ERROR"},
                ],
            },
            "vlm_captioning_model": {
                "label": "FastVLM Model",
                "kind": "enum_or_path",
                "default": "default",
                "options": [
                    {"value": "default", "label": "Default"},
                    {"value": "review", "label": "Review"},
                    {"value": "smoke", "label": "Smoke"},
                ],
                "helper_text": "Advisory caption model role or repo-safe checkpoint path.",
            },
            "vlm_captioning_proxy_format": {
                "label": "FastVLM Proxy Format",
                "kind": "enum",
                "default": "png",
                "options": [
                    {"value": "png", "label": "PNG"},
                    {"value": "jpeg", "label": "JPEG"},
                ],
                "helper_text": "Image proxy format for the isolated FastVLM subprocess.",
            },
            "vlm_captioning_max_side_px": {
                "label": "FastVLM Proxy Max Side",
                "kind": "integer",
                "default": 1600,
                "min": 1,
                "helper_text": "Largest proxy side length in pixels.",
            },
            "fastvlm_timeout_seconds": {
                "label": "FastVLM Timeout",
                "kind": "integer",
                "default": 180,
                "min": 1,
                "helper_text": "Subprocess timeout for advisory captioning.",
            },
        },
    }


def _lux_estimate_summary(normalized_args: Dict[str, Any]) -> Dict[str, Any]:
    runtime_score = 0
    gpu_score = 0
    research_risk = "none"
    reasons: List[str] = []

    if str(normalized_args.get("quality_tier") or "") == "apex":
        runtime_score += 1
    if str(normalized_args.get("depth_backend") or "") == "depth_pro":
        runtime_score += 1
        gpu_score += 1
        research_risk = "research_only"
        reasons.append("depth_pro_research_backend")
    if str(normalized_args.get("depth_backend") or "") == "da3":
        spec = _da3_model_spec_for_portal_key(str(normalized_args.get("model_key") or PORTAL_DEFAULT_DA3_MODEL_KEY))
        if spec is not None and bool(getattr(spec, "requires_non_commercial_ok", False)):
            research_risk = "research_only"
            reasons.append("da3_research_model")
    if str(normalized_args.get("segmentation_backend") or "") == "sam2" and _as_bool(
        normalized_args.get("enable_segmentation"),
        False,
    ):
        runtime_score += 1
        gpu_score += 1
        reasons.append("sam2_segmentation")

    if _as_bool(normalized_args.get("enable_reconstruction"), False):
        runtime_score += 1
        gpu_score += 1
        research_risk = "research_only"
        reasons.append("scene_reconstruction")
        iterations = int(normalized_args.get("reconstruction_iterations") or 1000)
        if iterations >= 2000:
            runtime_score += 1
            reasons.append("high_iteration_count")
        if iterations >= 3000:
            gpu_score += 1
        tier = str(normalized_args.get("reconstruction_tier") or "apex_research")
        if tier == "apex_research_ultra":
            runtime_score += 1
            gpu_score += 1
            reasons.append("ultra_reconstruction_tier")
        elif tier == "experimental":
            runtime_score += 1
            gpu_score += 1
            research_risk = "experimental"
            reasons.append("experimental_reconstruction_tier")

    if str(normalized_args.get("raw_ingest_mode") or "") == "force_rawpy":
        runtime_score += 1
        reasons.append("forced_rawpy_ingest")
    if _as_bool(normalized_args.get("emit_scene_debug_bundle"), False):
        runtime_score += 1
        reasons.append("debug_bundle_emission")
    if _as_bool(normalized_args.get("vlm_captioning_enabled"), False):
        runtime_score += 1
        reasons.append("fastvlm_captioning")

    max_gpu_workers = normalized_args.get("max_gpu_workers")
    if isinstance(max_gpu_workers, int) and max_gpu_workers >= 2:
        gpu_score += 1
        reasons.append("gpu_worker_override")
    max_workers = normalized_args.get("max_workers")
    if isinstance(max_workers, int) and max_workers > _portal_soft_cpu_worker_cap():
        runtime_score += 1
        reasons.append("cpu_worker_override")

    runtime_band = _portal_estimate_band(runtime_score)
    gpu_band = _portal_estimate_band(gpu_score)
    if research_risk == "none" and str(normalized_args.get("preset") or "").lower().find("research") >= 0:
        research_risk = "research_only"

    return {
        "runtime_band": runtime_band,
        "gpu_pressure": gpu_band,
        "research_risk": research_risk,
        "reasons": reasons,
        "summary_label": (
            f"{runtime_band.title()} runtime"
            f" · {gpu_band.title()} GPU pressure"
            f" · {research_risk.replace('_', ' ').title()} posture"
        ),
    }


def _lux_debug_bundle_summary(normalized_args: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = str(normalized_args.get("output_dir") or "").strip()
    enabled = _as_bool(normalized_args.get("emit_scene_debug_bundle"), False)
    return {
        "enabled": enabled,
        "requires_acknowledgement": enabled,
        "output_root": output_dir,
        "destination": LUX_DEBUG_BUNDLE_DESTINATION_TEMPLATE,
        "includes": [
            "scene_manifest",
            "camera_payload",
            "input_image_copies",
            "segmentation_overlays",
            "reprojection_preview",
        ],
        "sensitivity": "camera_metadata_and_source_images",
        "notes": [
            "Debug bundles may copy source imagery and camera metadata into the output tree.",
            "Portal dispatch requires an explicit acknowledgement before enabling debug bundle emission.",
        ],
    }


def _resolve_fastvlm_model_selector_path(selector: str) -> Path:
    normalized = str(selector or "default").strip()
    role = normalized.lower()
    if role == "review":
        env_path = os.getenv("TP_FASTVLM_REVIEW_MODEL", "").strip()
    elif role == "default":
        env_path = os.getenv("TP_FASTVLM_MODEL", "").strip()
    else:
        env_path = ""
    if env_path:
        return resolve_fastvlm_model_path(env_path, allowed_roots=tuple(FASTVLM_RUNTIME_ALLOWED_ROOTS))
    return resolve_fastvlm_model_path(normalized, allowed_roots=tuple(FASTVLM_RUNTIME_ALLOWED_ROOTS))


def _portal_existing_path_status(path_text: str, allowed_roots: List[Path], *, expected_type: str) -> Dict[str, Any]:
    text = str(path_text or "").strip()
    status = {
        "path": text,
        "configured": bool(text),
        "exists": False,
        "expected_type": expected_type,
        "status": "missing",
    }
    if not text:
        return status
    try:
        resolved_path = _resolve_allowed_request_path(text, allowed_roots)
    except (OSError, RuntimeError, ValueError, _PortalValidationReasonError):
        status["status"] = "invalid_path"
        return status
    trusted_entry = _trusted_allowed_entry(resolved_path, allowed_roots)
    if trusted_entry is None:
        return status
    try:
        if expected_type == "file":
            exists = trusted_entry.is_file()
        elif expected_type == "dir":
            exists = trusted_entry.is_dir()
        else:
            exists = trusted_entry.exists()
    except OSError:
        exists = False
    status["exists"] = bool(exists)
    status["path"] = _normalize_repo_relative_display_path(text, trusted_entry)
    status["status"] = "ready" if exists else "missing"
    return status


def _normalize_fastvlm_runtime_path_arg(
    value: Any,
    field: str,
    allowed_roots: List[Path],
    errors: List[Dict[str, Any]],
) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        resolved_path = _resolve_allowed_request_path(text, allowed_roots)
    except ValueError as exc:
        reason = _portal_reason_from_exception(exc, default="invalid_path_value")
        errors.append(
            _portal_issue(
                field,
                reason,
                f"{field} must stay within the allowed workspace roots.",
                suggestion=f"Choose a {field} path under the configured repository or temp roots.",
            )
        )
        return ""
    return _normalize_repo_relative_display_path(text, resolved_path)


def _portal_fastvlm_runtime_path_status(path_text: str, allowed_roots: List[Path], *, expected_type: str) -> Dict[str, Any]:
    text = str(path_text or "").strip()
    status = {
        "path": text,
        "configured": bool(text),
        "exists": False,
        "expected_type": expected_type,
        "status": "missing",
    }
    if not text:
        return status
    try:
        resolved_path = _resolve_allowed_request_path(text, allowed_roots)
    except (OSError, RuntimeError, ValueError, _PortalValidationReasonError):
        status["status"] = "invalid_path"
        return status
    trusted_entry = _trusted_allowed_entry(resolved_path, allowed_roots)
    if trusted_entry is None:
        status["path"] = _normalize_repo_relative_display_path(text, resolved_path)
        return status
    try:
        if expected_type == "file":
            exists = trusted_entry.is_file()
        elif expected_type == "dir":
            exists = trusted_entry.is_dir()
        else:
            exists = trusted_entry.exists()
    except OSError:
        exists = False
    status["exists"] = bool(exists)
    status["path"] = _normalize_repo_relative_display_path(text, trusted_entry)
    status["status"] = "ready" if exists else "missing"
    return status


def _portal_default_fastvlm_python_status(default_python_path: Path) -> Dict[str, Any]:
    """Return readiness for the managed default FastVLM venv Python path.

    Python venv executables are commonly symlinks to the system interpreter.
    Keep explicit user-provided paths on the strict realpath-contained path,
    but let the managed default venv path report ready when the symlink itself
    is present under `.runtime/fastvlm`.
    """
    status = {
        "path": str(default_python_path),
        "configured": True,
        "exists": False,
        "expected_type": "file",
        "status": "missing",
    }
    runtime_root = Path(os.path.realpath(default_fastvlm_runtime_root()))
    lexical_path = default_python_path.expanduser()
    if not lexical_path.is_absolute():
        lexical_path = REPO_ROOT / lexical_path
    try:
        lexical_path.relative_to(runtime_root)
    except ValueError:
        status["status"] = "invalid_path"
        return status
    exists = lexical_path.is_file()
    status["exists"] = bool(exists)
    status["path"] = _normalize_repo_relative_display_path(str(default_python_path), lexical_path)
    status["status"] = "ready" if exists else "missing"
    return status


def _fastvlm_runtime_remediation(check_name: str, status: str) -> str:
    if status == "ready":
        return "Ready for advisory FastVLM captioning."
    if status == "invalid_path":
        return "Choose a repo-safe path under the configured FastVLM runtime roots."
    if check_name == "python_executable":
        return "Run make install-fastvlm-runtime or provide the isolated FastVLM Python path."
    if check_name == "mlx_vlm_dir":
        return "Run make install-fastvlm-runtime or provide the pinned mlx-vlm checkout path."
    if check_name == "model_path":
        return "Install the selected FastVLM model role or choose an installed checkpoint path."
    return "Install or configure the optional FastVLM runtime."


def _fastvlm_readiness_check(
    check_name: str,
    path_status: Mapping[str, Any],
    *,
    required: bool,
    invalid_path: str = "",
) -> Dict[str, Any]:
    raw_status = str(path_status.get("status") or "missing").strip().lower()
    status = raw_status if raw_status in {"ready", "missing", "invalid_path"} else "missing"
    path = str(path_status.get("path") or "")
    if invalid_path:
        status = "invalid_path"
        path = invalid_path
    return {
        "status": status,
        "path": path,
        "expected_type": str(path_status.get("expected_type") or ""),
        "required": bool(required),
        "remediation": _fastvlm_runtime_remediation(check_name, status),
    }


def _fastvlm_runtime_readiness(
    *,
    enabled: bool,
    python_status: Mapping[str, Any],
    mlx_status: Mapping[str, Any],
    model_status: Mapping[str, Any],
    invalid_paths: Mapping[str, str],
) -> Dict[str, Any]:
    checks = {
        "python_executable": _fastvlm_readiness_check(
            "python_executable",
            python_status,
            required=enabled,
            invalid_path=str(invalid_paths.get("fastvlm_python_executable") or ""),
        ),
        "mlx_vlm_dir": _fastvlm_readiness_check(
            "mlx_vlm_dir",
            mlx_status,
            required=enabled,
            invalid_path=str(invalid_paths.get("fastvlm_mlx_vlm_dir") or ""),
        ),
        "model_path": _fastvlm_readiness_check(
            "model_path",
            model_status,
            required=enabled,
            invalid_path=str(invalid_paths.get("vlm_captioning_model") or ""),
        ),
    }
    if not enabled:
        status = "off"
    elif any(check["status"] == "invalid_path" for check in checks.values()):
        status = "invalid_config"
    elif all(check["status"] == "ready" for check in checks.values()):
        status = "ready"
    else:
        status = "missing_runtime"
    return {
        "status": status,
        "checks": checks,
        "verification_scope": "path-existence",
    }


def _normalize_vlm_captioning_model(
    raw_model: Any,
    errors: List[Dict[str, Any]],
) -> str:
    model = str(raw_model or "default").strip() or "default"
    role = model.lower()
    if role in ALLOWED_VLM_CAPTIONING_MODEL_ROLES:
        return role

    path_like = model.startswith(".") or "/" in model or "\\" in model or Path(model).is_absolute()
    if not path_like:
        errors.append(
            _portal_issue(
                "vlm_captioning_model",
                "invalid_vlm_captioning_model",
                "FastVLM model must be one of default, review, smoke, or a repo-safe model path.",
                suggestion="Choose a model role or a checkpoint path under the configured workspace roots.",
            )
        )
        return "default"

    normalized_model = _normalize_portal_path_arg(
        model,
        "vlm_captioning_model",
        FASTVLM_RUNTIME_ALLOWED_ROOTS,
        errors,
        required=False,
        must_exist=False,
    )
    return normalized_model or "default"


def _captioning_summary(
    normalized_args: Dict[str, Any],
    *,
    feature_enabled: bool,
    invalid_paths: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    enabled = _as_bool(normalized_args.get("vlm_captioning_enabled"), False)
    invalid_path_values = invalid_paths or {}
    default_python_path = default_fastvlm_runtime_root() / ".venv-fastvlm" / "bin" / "python"
    default_mlx_vlm_dir = default_fastvlm_runtime_root() / "mlx-vlm"
    explicit_python_path = str(normalized_args.get("fastvlm_python_executable") or "")
    python_status = _portal_fastvlm_runtime_path_status(
        str(explicit_python_path or default_python_path),
        FASTVLM_RUNTIME_ALLOWED_ROOTS,
        expected_type="file",
    )
    if not explicit_python_path and python_status.get("status") == "invalid_path":
        python_status = _portal_default_fastvlm_python_status(default_python_path)
    mlx_status = _portal_fastvlm_runtime_path_status(
        str(normalized_args.get("fastvlm_mlx_vlm_dir") or default_mlx_vlm_dir),
        FASTVLM_RUNTIME_ALLOWED_ROOTS,
        expected_type="dir",
    )
    try:
        model_path = _resolve_fastvlm_model_selector_path(str(normalized_args.get("vlm_captioning_model") or "default"))
    except ValueError:
        model_status = {
            "path": str(normalized_args.get("vlm_captioning_model") or "default"),
            "configured": True,
            "exists": False,
            "expected_type": "dir",
            "status": "invalid_path",
        }
    else:
        model_status = _portal_existing_path_status(
            str(model_path),
            FASTVLM_RUNTIME_ALLOWED_ROOTS,
            expected_type="dir",
        )
    runtime_readiness = _fastvlm_runtime_readiness(
        enabled=enabled,
        python_status=python_status,
        mlx_status=mlx_status,
        model_status=model_status,
        invalid_paths=invalid_path_values,
    )
    if not enabled:
        runtime_status = "off"
    elif runtime_readiness["status"] == "ready":
        runtime_status = "ready"
    else:
        runtime_status = "missing_runtime"
    return {
        "feature_enabled": bool(feature_enabled),
        "enabled": enabled,
        "backend": str(normalized_args.get("vlm_captioning_backend") or "fastvlm"),
        "model": str(normalized_args.get("vlm_captioning_model") or "default"),
        "proxy_format": str(normalized_args.get("vlm_captioning_proxy_format") or "png"),
        "max_side_px": int(normalized_args.get("vlm_captioning_max_side_px") or 1600),
        "fastvlm_python_executable": str(normalized_args.get("fastvlm_python_executable") or ""),
        "fastvlm_mlx_vlm_dir": str(normalized_args.get("fastvlm_mlx_vlm_dir") or ""),
        "timeout_seconds": int(normalized_args.get("fastvlm_timeout_seconds") or 180),
        "runtime_path_status": {
            "python_executable": python_status,
            "mlx_vlm_dir": mlx_status,
            "model_path": model_status,
        },
        "runtime_readiness": runtime_readiness,
        "runtime_status": runtime_status,
        "role": "advisory",
        "used_for_quality_gate": False,
    }


def _build_lux_config_preview(
    args: Dict[str, Any],
    *,
    readiness_snapshot: Optional[Dict[str, Any]] = None,
    portal_actor: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    args, path_warnings, path_errors = _normalize_operator_payload_paths("lux-depth-v3", args)
    defaults = _lux_portal_defaults(args)
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = list(path_warnings)
    inactive_fields: List[Dict[str, Any]] = []
    path_errors_by_field = _preview_path_errors_by_field(path_errors)

    normalized_args: Dict[str, Any] = {}
    preset_raw = str(_pick(args, "preset", default=defaults["preset"]) or defaults["preset"]).strip()
    allowed_preset_names = _allowed_preset_names("lux-depth-v3")
    if allowed_preset_names and preset_raw and preset_raw not in allowed_preset_names:
        errors.append(
            _portal_issue(
                "preset",
                "invalid_preset",
                "The selected preset is not supported.",
                suggestion="Select a preset from the catalog.",
            )
        )
        preset_raw = str(defaults["preset"])
    normalized_args["preset"] = preset_raw or defaults["preset"]
    quality = (
        str(_pick(args, "quality_tier", "qualityTier", default=defaults["quality_tier"]) or defaults["quality_tier"])
        .strip()
        .lower()
    )
    if quality not in ALLOWED_QUALITY:
        errors.append(
            _portal_issue(
                "quality_tier",
                "invalid_quality_tier",
                "Quality tier is not supported.",
                suggestion="Choose standard, premium, or apex.",
            )
        )
        quality = str(defaults["quality_tier"])
    normalized_args["quality_tier"] = quality

    depth_backend = _canonical_depth_backend(_pick(args, "depth_backend", "depthBackend", default=defaults["depth_backend"]))
    if depth_backend not in ALLOWED_BACKENDS:
        errors.append(
            _portal_issue(
                "depth_backend",
                "invalid_depth_backend",
                "Depth backend is not supported.",
                suggestion="Choose da3 or depth_pro.",
            )
        )
        depth_backend = str(defaults["depth_backend"])
    normalized_args["depth_backend"] = depth_backend

    da3_model_key = ""
    da3_model_spec: Optional[Any] = None
    if depth_backend == "da3":
        da3_model_key = _canonical_da3_model_key(_pick(args, "model_key", "modelKey", default=defaults["model_key"]))
        da3_model_spec = _da3_model_spec_for_portal_key(da3_model_key)
        if da3_model_spec is None:
            errors.append(
                _portal_issue(
                    "model_key",
                    "invalid_model_key",
                    "The selected DA3 model is not supported.",
                    suggestion="Choose da3-metric for Apache-2.0 usage or da3-research with a non-commercial acknowledgment.",
                )
            )
            da3_model_key = str(defaults["model_key"])
            da3_model_spec = _da3_model_spec_for_portal_key(da3_model_key)
        normalized_args["model_key"] = da3_model_key

    depth_device = str(
        _pick(args, "depth_device", "depthDevice", default=defaults["depth_device"]) or defaults["depth_device"]
    ).strip()
    if depth_device:
        device_token = depth_device.lower()
        if device_token not in ALLOWED_DEPTH_DEVICES:
            errors.append(
                _portal_issue(
                    "depth_device",
                    "invalid_depth_device",
                    "The selected compute device is not supported.",
                    suggestion="Choose cpu, cuda, or mps.",
                )
            )
            depth_device = str(defaults["depth_device"])
        normalized_args["depth_device"] = depth_device

    normalized_args["input_dir"] = _normalize_preview_path_field(
        args,
        "input_dir",
        ("input_dir", "inputDir"),
        ALLOWED_INPUT_ROOTS,
        errors,
        path_errors_by_field,
        required=True,
    )
    normalized_args["output_dir"] = _normalize_preview_path_field(
        args,
        "output_dir",
        ("output_dir", "outputDir"),
        ALLOWED_OUTPUT_ROOTS,
        errors,
        path_errors_by_field,
        required=True,
    )

    segmentation_enabled = _as_bool(
        _pick(args, "enable_segmentation", "enableSegmentation", default=defaults["enable_segmentation"]),
        default=bool(defaults["enable_segmentation"]),
    )
    normalized_args["enable_segmentation"] = segmentation_enabled

    segmentation_backend = (
        str(
            _pick(args, "segmentation_backend", "segmentationBackend", default=defaults["segmentation_backend"])
            or defaults["segmentation_backend"]
        )
        .strip()
        .lower()
    )
    if segmentation_backend not in ALLOWED_SEGMENTATION_BACKENDS:
        errors.append(
            _portal_issue(
                "segmentation_backend",
                "invalid_segmentation_backend",
                "Segmentation backend is not supported.",
                suggestion="Choose stub, efficientsam, or sam2.",
            )
        )
        segmentation_backend = str(defaults["segmentation_backend"])
    normalized_args["segmentation_backend"] = segmentation_backend

    segmentation_cache = (
        str(
            _pick(
                args,
                "segmentation_cache",
                "segmentationCache",
                default=defaults["segmentation_cache"],
            )
            or defaults["segmentation_cache"]
        )
        .strip()
        .lower()
    )
    if segmentation_cache not in ALLOWED_SEGMENTATION_CACHE_POLICIES:
        errors.append(
            _portal_issue(
                "segmentation_cache",
                "invalid_segmentation_cache",
                "Segmentation cache policy is not supported.",
                suggestion="Choose off or read_write.",
            )
        )
        segmentation_cache = str(defaults["segmentation_cache"])
    normalized_args["segmentation_cache"] = segmentation_cache

    sam2_model_size = (
        str(
            _pick(args, "sam2_model_size", "sam2ModelSize", default=defaults["sam2_model_size"]) or defaults["sam2_model_size"]
        )
        .strip()
        .lower()
    )
    if sam2_model_size not in ALLOWED_SAM2_MODEL_SIZES:
        errors.append(
            _portal_issue(
                "sam2_model_size",
                "invalid_sam2_model_size",
                "SAM2 model size is not supported.",
                suggestion="Choose base or large.",
            )
        )
        sam2_model_size = str(defaults["sam2_model_size"])
    normalized_args["sam2_model_size"] = sam2_model_size
    sam2_checkpoint_path = _normalize_preview_path_field(
        args,
        "sam2_checkpoint_path",
        ("sam2_checkpoint_path", "sam2CheckpointPath"),
        ALLOWED_INPUT_ROOTS,
        errors,
        path_errors_by_field,
        required=False,
        must_exist=False,
    )
    if sam2_checkpoint_path:
        if segmentation_backend == "sam2":
            validation = _resolve_managed_sam2_checkpoint_validation(sam2_checkpoint_path)
            if validation.reason == "untrusted_checkpoint_path":
                errors.append(
                    _portal_issue(
                        "sam2_checkpoint_path",
                        "untrusted_checkpoint_path",
                        "SAM2 checkpoint overrides must use a repo-controlled or checksum-verified file.",
                        suggestion=(
                            "Use ./models/sam2/... or ./checkpoints/... for repo-managed checkpoints, "
                            "or provide a file whose SHA-256 matches the governed SAM2 checkpoint manifest."
                        ),
                    )
                )
            elif validation.reason is not None:
                message = _portal_safe_error_message(
                    validation.reason,
                    field="sam2_checkpoint_path",
                )
                errors.append(
                    _portal_issue(
                        "sam2_checkpoint_path",
                        validation.reason,
                        message,
                        suggestion=("Choose an existing checkpoint file under the configured repository or temp roots."),
                    )
                )
            else:
                normalized_args["sam2_checkpoint_path"] = str(validation.normalized_path or sam2_checkpoint_path)
        else:
            normalized_args["sam2_checkpoint_path"] = sam2_checkpoint_path
    normalized_args["sam2_tiling_enabled"] = _as_bool(
        _pick(args, "sam2_tiling_enabled", "sam2TilingEnabled", default=defaults["sam2_tiling_enabled"]),
        default=bool(defaults["sam2_tiling_enabled"]),
    )
    for field_name in (
        "sam2_tile_size_px",
        "sam2_global_pass_longest_side",
        "sam2_max_concurrency",
        "sam2_points_per_side",
        "sam2_points_per_batch",
    ):
        value = _normalize_optional_positive_int(
            _pick(args, field_name, default=defaults[field_name]),
            field_name,
            errors,
        )
        normalized_args[field_name] = int(defaults[field_name]) if value is None else value
    for field_name in ("sam2_overlap_px", "sam2_crop_n_layers"):
        value = _normalize_optional_non_negative_int(
            _pick(args, field_name, default=defaults[field_name]),
            field_name,
            errors,
        )
        normalized_args[field_name] = int(defaults[field_name]) if value is None else value
    for field_name in ("sam2_pred_iou_thresh", "sam2_stability_score_thresh"):
        value = _normalize_optional_probability(
            _pick(args, field_name, default=defaults[field_name]),
            field_name,
            errors,
        )
        normalized_args[field_name] = float(defaults[field_name]) if value is None else value
    if normalized_args["sam2_overlap_px"] >= normalized_args["sam2_tile_size_px"]:
        errors.append(
            _portal_issue(
                "sam2_overlap_px",
                "invalid_sam2_overlap_px",
                "sam2_overlap_px must be smaller than sam2_tile_size_px.",
            )
        )
    normalized_args["strict_segmentation"] = _as_bool(
        _pick(args, "strict_segmentation", "strictSegmentation", default=defaults["strict_segmentation"]),
        default=bool(defaults["strict_segmentation"]),
    )

    for field_name in (
        "materials_v3",
        "pbr",
        "save_float_depth",
        "cache_depth",
        "enable_v2",
        "emit_master16",
        "emit_upscaled16",
        "emit_marketing",
        "emit_report",
        "emit_run_card",
        "non_commercial_ok",
        "accept_apple_depth_pro_research_license",
        "accept_research_tools_license",
        "force_depth",
        "strict_inputs",
        "verify_images",
        "allow_semantic_fallback",
        "verbose",
        "quiet",
        "overwrite",
    ):
        normalized_args[field_name] = _as_bool(
            _pick(args, field_name, default=defaults[field_name]), default=bool(defaults[field_name])
        )
    normalized_args["run_card_version"] = (
        str(_pick(args, "run_card_version", "runCardVersion", default=defaults["run_card_version"]) or "v1").strip().lower()
        or "v1"
    )
    if normalized_args["run_card_version"] not in ALLOWED_RUN_CARD_VERSIONS:
        errors.append(
            _portal_issue(
                "run_card_version",
                "invalid_run_card_version",
                "Run card version must be v1 or v2.",
            )
        )
        normalized_args["run_card_version"] = str(defaults["run_card_version"])
    normalized_args["run_card_include_proofs"] = _as_bool(
        _pick(
            args,
            "run_card_include_proofs",
            "runCardIncludeProofs",
            default=defaults["run_card_include_proofs"],
        ),
        default=bool(defaults["run_card_include_proofs"]),
    )

    normalized_args["v2_preset"] = str(
        _pick(args, "v2_preset", "v2Preset", default=defaults["v2_preset"]) or defaults["v2_preset"]
    ).strip() or str(defaults["v2_preset"])
    normalized_args["enable_reconstruction"] = _as_bool(
        _pick(args, "enable_reconstruction", "enableReconstruction", default=defaults["enable_reconstruction"]),
        default=bool(defaults["enable_reconstruction"]),
    )
    grouping_mode = (
        str(_pick(args, "grouping_mode", "groupingMode", default=defaults["grouping_mode"]) or defaults["grouping_mode"])
        .strip()
        .lower()
    )
    if grouping_mode not in ALLOWED_GROUPING_MODES:
        errors.append(
            _portal_issue(
                "grouping_mode",
                "invalid_grouping_mode",
                "Grouping mode is not supported.",
                suggestion="Choose single or parent_dir.",
            )
        )
        grouping_mode = str(defaults["grouping_mode"])
    normalized_args["grouping_mode"] = grouping_mode

    cameras_sidecar_path = _normalize_preview_path_field(
        args,
        "cameras_sidecar_path",
        ("cameras_sidecar_path", "camerasSidecarPath"),
        ALLOWED_INPUT_ROOTS,
        errors,
        path_errors_by_field,
        required=False,
        must_exist=bool(normalized_args["enable_reconstruction"]),
        must_be_file=True,
    )
    if cameras_sidecar_path:
        normalized_args["cameras_sidecar_path"] = cameras_sidecar_path

    reconstruction_iterations = _normalize_optional_positive_int(
        _pick(args, "reconstruction_iterations", "reconstructionIterations", default=defaults["reconstruction_iterations"]),
        "reconstruction_iterations",
        errors,
    )
    if reconstruction_iterations is None:
        reconstruction_iterations = int(defaults["reconstruction_iterations"])
    normalized_args["reconstruction_iterations"] = reconstruction_iterations

    reconstruction_tier = (
        str(
            _pick(args, "reconstruction_tier", "reconstructionTier", default=defaults["reconstruction_tier"])
            or defaults["reconstruction_tier"]
        )
        .strip()
        .lower()
    )
    if reconstruction_tier not in ALLOWED_RECONSTRUCTION_TIERS:
        errors.append(
            _portal_issue(
                "reconstruction_tier",
                "invalid_reconstruction_tier",
                "Reconstruction tier is not supported.",
                suggestion="Choose apex_research, apex_research_ultra, or experimental.",
            )
        )
        reconstruction_tier = str(defaults["reconstruction_tier"])
    normalized_args["reconstruction_tier"] = reconstruction_tier
    normalized_args["emit_scene_debug_bundle"] = _as_bool(
        _pick(args, "emit_scene_debug_bundle", "emitSceneDebugBundle", default=defaults["emit_scene_debug_bundle"]),
        default=bool(defaults["emit_scene_debug_bundle"]),
    )

    raw_ingest_mode = (
        str(
            _pick(args, "raw_ingest_mode", "rawIngestMode", default=defaults["raw_ingest_mode"]) or defaults["raw_ingest_mode"]
        )
        .strip()
        .lower()
    )
    if raw_ingest_mode not in ALLOWED_RAW_INGEST_MODES:
        errors.append(
            _portal_issue(
                "raw_ingest_mode",
                "invalid_raw_ingest_mode",
                "RAW ingest mode is not supported.",
                suggestion="Choose auto, force_rawpy, or force_preview.",
            )
        )
        raw_ingest_mode = str(defaults["raw_ingest_mode"])
    normalized_args["raw_ingest_mode"] = raw_ingest_mode

    raw_wb_mode = (
        str(_pick(args, "raw_wb_mode", "rawWbMode", default=defaults["raw_wb_mode"]) or defaults["raw_wb_mode"])
        .strip()
        .lower()
    )
    if raw_wb_mode not in ALLOWED_RAW_WB_MODES:
        errors.append(
            _portal_issue(
                "raw_wb_mode",
                "invalid_raw_wb_mode",
                "RAW white-balance mode is not supported.",
                suggestion="The current backend supports only camera.",
            )
        )
        raw_wb_mode = str(defaults["raw_wb_mode"])
    normalized_args["raw_wb_mode"] = raw_wb_mode

    raw_demosaic = (
        str(_pick(args, "raw_demosaic", "rawDemosaic", default=defaults["raw_demosaic"]) or defaults["raw_demosaic"])
        .strip()
        .upper()
    )
    if not _is_valid_demosaic_name(raw_demosaic):
        errors.append(
            _portal_issue(
                "raw_demosaic",
                "invalid_raw_demosaic",
                "RAW demosaic mode is not supported.",
                suggestion=(
                    "Provide a rawpy.DemosaicAlgorithm member name (uppercase letters, digits, "
                    "underscores; must start with a letter). The decode step verifies the name "
                    "against the installed LibRaw build and fails closed for unknown values."
                ),
            )
        )
        raw_demosaic = str(defaults["raw_demosaic"])
    normalized_args["raw_demosaic"] = raw_demosaic

    captioning_feature_enabled = _portal_fastvlm_captioning_enabled(portal_actor)
    vlm_captioning_requested = _as_bool(
        _pick(
            args,
            "vlm_captioning_enabled",
            "vlmCaptioningEnabled",
            default=defaults["vlm_captioning_enabled"],
        ),
        default=bool(defaults["vlm_captioning_enabled"]),
    )
    if vlm_captioning_requested and not captioning_feature_enabled:
        errors.append(
            _portal_issue(
                "vlm_captioning_enabled",
                "captioning_feature_disabled",
                "FastVLM captioning is not enabled for this portal cohort.",
                suggestion="Disable FastVLM captioning or enable the portal captioning feature flag for this cohort.",
            )
        )
        vlm_captioning_requested = False
    normalized_args["vlm_captioning_enabled"] = bool(vlm_captioning_requested)

    vlm_captioning_backend = (
        str(
            _pick(
                args,
                "vlm_captioning_backend",
                "vlmCaptioningBackend",
                default=defaults["vlm_captioning_backend"],
            )
            or defaults["vlm_captioning_backend"]
        )
        .strip()
        .lower()
    )
    if vlm_captioning_backend not in ALLOWED_VLM_CAPTIONING_BACKENDS:
        errors.append(
            _portal_issue(
                "vlm_captioning_backend",
                "invalid_vlm_captioning_backend",
                "FastVLM is the only supported advisory captioning backend.",
                suggestion="Choose fastvlm.",
            )
        )
        vlm_captioning_backend = str(defaults["vlm_captioning_backend"])
    normalized_args["vlm_captioning_backend"] = vlm_captioning_backend

    captioning_invalid_paths: Dict[str, str] = {}

    def _captioning_invalid_display_value(value: Any) -> str:
        return str(value or "").replace("\x00", "").strip()

    def _record_captioning_error_path(field: str, raw_value: Any, new_errors: List[Dict[str, Any]]) -> None:
        if any(str(issue.get("field") or "") == field for issue in new_errors):
            captioning_invalid_paths[field] = _captioning_invalid_display_value(raw_value)

    raw_captioning_model = _pick(
        args,
        "vlm_captioning_model",
        "vlmCaptioningModel",
        default=defaults["vlm_captioning_model"],
    )
    before_captioning_model_errors = len(errors)
    normalized_args["vlm_captioning_model"] = _normalize_vlm_captioning_model(raw_captioning_model, errors)
    _record_captioning_error_path(
        "vlm_captioning_model",
        raw_captioning_model,
        errors[before_captioning_model_errors:],
    )

    vlm_captioning_proxy_format = (
        str(
            _pick(
                args,
                "vlm_captioning_proxy_format",
                "vlmCaptioningProxyFormat",
                default=defaults["vlm_captioning_proxy_format"],
            )
            or defaults["vlm_captioning_proxy_format"]
        )
        .strip()
        .lower()
    )
    if vlm_captioning_proxy_format not in ALLOWED_VLM_CAPTIONING_PROXY_FORMATS:
        errors.append(
            _portal_issue(
                "vlm_captioning_proxy_format",
                "invalid_vlm_captioning_proxy_format",
                "FastVLM proxy format must be png or jpeg.",
                suggestion="Choose png or jpeg.",
            )
        )
        vlm_captioning_proxy_format = str(defaults["vlm_captioning_proxy_format"])
    normalized_args["vlm_captioning_proxy_format"] = vlm_captioning_proxy_format

    max_side_px = _normalize_optional_positive_int(
        _pick(
            args,
            "vlm_captioning_max_side_px",
            "vlmCaptioningMaxSidePx",
            default=defaults["vlm_captioning_max_side_px"],
        ),
        "vlm_captioning_max_side_px",
        errors,
    )
    normalized_args["vlm_captioning_max_side_px"] = (
        int(defaults["vlm_captioning_max_side_px"]) if max_side_px is None else max_side_px
    )

    def _captioning_path_value(field: str, aliases: Tuple[str, ...]) -> str:
        existing_errors = path_errors_by_field.get(field) or []
        raw_value = _pick(args, *aliases, default="")
        if existing_errors:
            errors.extend(existing_errors)
            captioning_invalid_paths[field] = _captioning_invalid_display_value(raw_value)
            return ""
        field_supplied = raw_value is not None and bool(str(raw_value).strip())
        if not field_supplied:
            return ""
        before_path_errors = len(errors)
        normalized_path = _normalize_fastvlm_runtime_path_arg(
            raw_value,
            field,
            FASTVLM_RUNTIME_ALLOWED_ROOTS,
            errors,
        )
        _record_captioning_error_path(field, raw_value, errors[before_path_errors:])
        return normalized_path

    normalized_args["fastvlm_python_executable"] = _captioning_path_value(
        "fastvlm_python_executable",
        ("fastvlm_python_executable", "fastvlmPythonExecutable"),
    )
    normalized_args["fastvlm_mlx_vlm_dir"] = _captioning_path_value(
        "fastvlm_mlx_vlm_dir",
        ("fastvlm_mlx_vlm_dir", "fastvlmMlxVlmDir"),
    )
    fastvlm_timeout_seconds = _normalize_optional_positive_int(
        _pick(
            args,
            "fastvlm_timeout_seconds",
            "fastvlmTimeoutSeconds",
            default=defaults["fastvlm_timeout_seconds"],
        ),
        "fastvlm_timeout_seconds",
        errors,
    )
    normalized_args["fastvlm_timeout_seconds"] = (
        int(defaults["fastvlm_timeout_seconds"]) if fastvlm_timeout_seconds is None else fastvlm_timeout_seconds
    )
    captioning_summary = _captioning_summary(
        normalized_args,
        feature_enabled=captioning_feature_enabled,
        invalid_paths=captioning_invalid_paths,
    )
    if normalized_args["vlm_captioning_enabled"]:
        warnings.append(
            _portal_issue(
                "vlm_captioning_enabled",
                "vlm_captioning_advisory_only",
                "FastVLM captions are advisory sidecar metadata and are not used for quality gates.",
                suggestion="Review captions as operator context only.",
            )
        )
        runtime_readiness = captioning_summary.get("runtime_readiness") or {}
        if captioning_summary.get("runtime_status") == "missing_runtime":
            warnings.append(
                _portal_issue(
                    "fastvlm_python_executable",
                    "fastvlm_runtime_missing",
                    "FastVLM runtime paths are not fully present; caption sidecars may be skipped at run time.",
                    suggestion="Install the optional FastVLM runtime under ./.runtime/fastvlm when captions are needed.",
                )
            )
        if isinstance(runtime_readiness, dict):
            readiness_checks = runtime_readiness.get("checks") or {}
            if isinstance(readiness_checks, dict):
                check_fields = {
                    "python_executable": "fastvlm_python_executable",
                    "mlx_vlm_dir": "fastvlm_mlx_vlm_dir",
                    "model_path": "vlm_captioning_model",
                }
                for check_name, check in readiness_checks.items():
                    if not isinstance(check, dict) or check.get("status") != "missing":
                        continue
                    warnings.append(
                        _portal_issue(
                            check_fields.get(str(check_name), "vlm_captioning_model"),
                            f"fastvlm_runtime_{check_name}_missing",
                            f"FastVLM {check_name.replace('_', ' ')} is not ready for advisory captioning.",
                            suggestion=str(check.get("remediation") or "Install or configure the optional FastVLM runtime."),
                        )
                    )

    max_workers = _normalize_optional_positive_int(_pick(args, "max_workers", "maxWorkers"), "max_workers", errors)
    max_gpu_workers = _normalize_optional_positive_int(
        _pick(args, "max_gpu_workers", "maxGpuWorkers"), "max_gpu_workers", errors
    )
    if max_workers is not None:
        normalized_args["max_workers"] = max_workers
    if max_gpu_workers is not None:
        normalized_args["max_gpu_workers"] = max_gpu_workers

    log_level = str(_pick(args, "log_level", "logLevel", default="") or "").strip().upper()
    if log_level and log_level not in ALLOWED_LOG_LEVELS:
        errors.append(
            _portal_issue(
                "log_level",
                "invalid_log_level",
                "Log level is not supported.",
                suggestion="Choose DEBUG, INFO, WARNING, or ERROR.",
            )
        )
        log_level = ""
    if log_level:
        normalized_args["log_level"] = log_level

    if normalized_args["verbose"] and normalized_args["quiet"]:
        errors.append(
            _portal_issue(
                "verbose",
                "conflicting_log_verbosity_flags",
                "verbose and quiet cannot both be enabled.",
                suggestion="Disable either verbose or quiet before dispatch.",
            )
        )

    if depth_backend == "depth_pro":
        if not normalized_args["non_commercial_ok"]:
            errors.append(
                _portal_issue(
                    "non_commercial_ok",
                    "depth_pro_non_commercial_required",
                    "Depth Pro requires a non-commercial acknowledgment before dispatch.",
                    suggestion="Acknowledge non-commercial use to continue with Depth Pro.",
                )
            )
        if not normalized_args["accept_apple_depth_pro_research_license"]:
            errors.append(
                _portal_issue(
                    "accept_apple_depth_pro_research_license",
                    "depth_pro_license_required",
                    "Depth Pro requires the Apple research license acknowledgment before dispatch.",
                    suggestion="Acknowledge the Apple Depth Pro research license to continue.",
                )
            )

    if (
        depth_backend == "da3"
        and da3_model_spec is not None
        and bool(getattr(da3_model_spec, "requires_non_commercial_ok", False))
        and not normalized_args["non_commercial_ok"]
    ):
        errors.append(
            _portal_issue(
                "non_commercial_ok",
                "da3_model_non_commercial_required",
                "The selected DA3 research model requires a non-commercial acknowledgment.",
                suggestion="Acknowledge non-commercial use or switch the DA3 model to da3-metric.",
            )
        )

    if "v3.1" in str(normalized_args["preset"]).lower() and not normalized_args["non_commercial_ok"]:
        errors.append(
            _portal_issue(
                "non_commercial_ok",
                "research_preset_non_commercial_required",
                "The selected research preset requires a non-commercial acknowledgment.",
                suggestion="Acknowledge non-commercial use or switch to a non-research preset.",
            )
        )

    if quality == "apex" and normalized_args["materials_v3"]:
        if not segmentation_enabled:
            errors.append(
                _portal_issue(
                    "enable_segmentation",
                    "apex_materials_requires_segmentation",
                    "APEX with Materials V3 requires segmentation to be enabled.",
                    suggestion="Enable segmentation or disable Materials V3.",
                )
            )
        if segmentation_backend == "stub":
            errors.append(
                _portal_issue(
                    "segmentation_backend",
                    "apex_materials_requires_real_segmentation",
                    "APEX with Materials V3 cannot use the stub segmentation backend.",
                    suggestion="Choose efficientsam or sam2.",
                )
            )
        if not normalized_args["strict_segmentation"]:
            errors.append(
                _portal_issue(
                    "strict_segmentation",
                    "apex_materials_requires_strict_segmentation",
                    "APEX with Materials V3 requires strict segmentation.",
                    suggestion="Enable strict segmentation or disable Materials V3.",
                )
            )

    if normalized_args["enable_reconstruction"]:
        cameras_sidecar_has_error = any(str(item.get("field") or "").strip() == "cameras_sidecar_path" for item in errors)
        if not normalized_args["non_commercial_ok"]:
            errors.append(
                _portal_issue(
                    "non_commercial_ok",
                    "reconstruction_non_commercial_required",
                    "Scene reconstruction requires a non-commercial acknowledgment.",
                    suggestion="Acknowledge non-commercial use before dispatch.",
                )
            )
        if not normalized_args["accept_research_tools_license"]:
            errors.append(
                _portal_issue(
                    "accept_research_tools_license",
                    "reconstruction_license_required",
                    "Scene reconstruction requires the research-tools license acknowledgment.",
                    suggestion="Acknowledge the research-tools license before dispatch.",
                )
            )
        if "cameras_sidecar_path" not in normalized_args and not cameras_sidecar_has_error:
            warnings.append(
                _portal_issue(
                    "cameras_sidecar_path",
                    "camera_sidecar_missing",
                    "Camera sidecar path is missing; reconstruction may fail for multi-view scenes.",
                    suggestion="Provide a camera sidecar file when available.",
                )
            )
        if grouping_mode == "single":
            warnings.append(
                _portal_issue(
                    "grouping_mode",
                    "reconstruction_single_grouping",
                    'Reconstruction is enabled with grouping mode "single"; overlap may be weak.',
                    suggestion="Use parent_dir for typical multi-view scene folders.",
                )
            )
    else:
        for field_name in LUX_RECONSTRUCTION_INACTIVE_FIELDS:
            value = normalized_args.get(field_name)
            inactive_field = _portal_inactive_reconstruction_field_value(
                args,
                defaults,
                field_name,
                value,
            )
            if inactive_field is not None:
                inactive_fields.append(inactive_field)

    if raw_ingest_mode == "force_rawpy":
        warnings.append(
            _portal_issue(
                "raw_ingest_mode",
                "force_rawpy_runtime_warning",
                "force_rawpy may increase runtime and memory pressure.",
                suggestion="Use auto unless a RAW decode mismatch requires force_rawpy.",
            )
        )

    if isinstance(max_workers, int) and max_workers > _portal_soft_cpu_worker_cap():
        warnings.append(
            _portal_issue(
                "max_workers",
                "cpu_workers_above_recommended_cap",
                "Max workers is above the recommended Portal cap for typical local runs.",
                suggestion=f"Consider { _portal_soft_cpu_worker_cap() } or Auto unless profiling shows a benefit.",
            )
        )
    if isinstance(max_gpu_workers, int):
        if depth_device not in {"cuda", "mps"}:
            warnings.append(
                _portal_issue(
                    "max_gpu_workers",
                    "gpu_workers_without_gpu_device",
                    "Max GPU workers is set while the selected depth device is not GPU-backed.",
                    suggestion="Use Auto or switch to a GPU-backed depth device.",
                )
            )
        elif max_gpu_workers > _portal_soft_gpu_worker_cap():
            warnings.append(
                _portal_issue(
                    "max_gpu_workers",
                    "gpu_workers_above_recommended_cap",
                    "Max GPU workers is above the recommended Portal cap.",
                    suggestion=f"Consider {_portal_soft_gpu_worker_cap()} or Auto to reduce VRAM contention.",
                )
            )

    if normalized_args["emit_scene_debug_bundle"]:
        warnings.append(
            _portal_issue(
                "emit_scene_debug_bundle",
                "debug_bundle_sensitive_output",
                "Debug bundle emission copies source images and camera metadata into the output tree.",
                suggestion="Review the debug bundle acknowledgment before dispatch.",
            )
        )

    effective_args = dict(normalized_args)
    if not normalized_args["enable_reconstruction"]:
        for field_name in LUX_RECONSTRUCTION_INACTIVE_FIELDS:
            effective_args.pop(field_name, None)

    if readiness_snapshot is None:
        readiness_args = dict(normalized_args)
        readiness_snapshot = _evaluate_pipeline_readiness(
            "lux-depth-v3",
            readiness_args,
            require_dispatch_inputs=True,
        )

    argv_preview = ""
    if not errors:
        argv_preview = _format_argv_preview(
            _argv_from_request(
                {
                    "pipeline": "lux-depth-v3",
                    "args": normalized_args,
                },
                execution_args=normalized_args,
            )
        )

    return {
        "pipeline": "lux-depth-v3",
        "normalized_args": effective_args,
        "execution_args": dict(normalized_args),
        "argv_preview": argv_preview,
        "field_errors": errors,
        "field_warnings": warnings,
        "inactive_fields": inactive_fields,
        "readiness": readiness_snapshot,
        "estimate_summary": _lux_estimate_summary(normalized_args),
        "debug_bundle_summary": _lux_debug_bundle_summary(normalized_args),
        "captioning_summary": captioning_summary,
        "next_best_action": _preview_next_best_action(
            pipeline="lux-depth-v3",
            errors=errors,
            warnings=warnings,
            readiness_snapshot=readiness_snapshot,
        ),
    }


def _build_archive_config_preview(
    pipeline: str,
    args: Dict[str, Any],
    *,
    readiness_snapshot: Optional[Dict[str, Any]] = None,
    archive_index_scan_mode: str = "preview",
) -> Dict[str, Any]:
    args, path_warnings, path_errors = _normalize_operator_payload_paths(pipeline, args)
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = list(path_warnings)
    path_errors_by_field = _preview_path_errors_by_field(path_errors)
    normalized_args = dict(args)
    handled_path_fields: set[str] = set()

    def _set_archive_path_field(
        field: str,
        keys: Tuple[str, ...],
        *,
        required: bool = False,
        must_exist: bool = False,
        must_be_file: bool = False,
        must_be_dir: bool = False,
    ) -> str:
        allowed_roots = _allowed_roots_for_scope(
            next((scope for canonical, _, scope in PATH_FIELD_SPECS if canonical == field), PATH_SCOPE_ANY)
        )
        value = _normalize_preview_path_field(
            args,
            field,
            keys,
            allowed_roots,
            errors,
            path_errors_by_field,
            required=required,
            must_exist=must_exist or must_be_dir,
            must_be_file=must_be_file,
            must_be_dir=must_be_dir,
        )
        handled_path_fields.add(field)
        for key in keys:
            normalized_args.pop(key, None)
        if value or required:
            normalized_args[field] = value
        else:
            normalized_args.pop(field, None)
        return value

    _set_archive_path_field("input_dir", ("input_dir", "inputDir"), required=True)
    _set_archive_path_field("output_dir", ("output_dir", "outputDir"), required=True)

    default_command = ARCHIVE_GATE_DEFAULT_COMMANDS[pipeline]
    archive_command = str(_pick(args, "archive_command", "archiveCommand", default=default_command) or "").strip()
    if not archive_command:
        archive_command = default_command
    normalized_args["archive_command"] = archive_command
    if archive_command and archive_command not in ARCHIVE_GATE_ALLOWED_COMMANDS[pipeline]:
        errors.append(
            _portal_issue(
                "payload",
                "invalid_archive_command",
                _portal_safe_error_message("invalid_archive_command"),
            )
        )
    command = archive_command or default_command

    if command in {"fixity-scan", "fixity-verify", "manifest-build"}:
        _set_archive_path_field(
            "archive_index",
            ("archive_index", "archiveIndex"),
            required=command in {"fixity-scan", "manifest-build"},
            must_exist=command in {"fixity-scan", "manifest-build"},
            must_be_file=command in {"fixity-scan", "manifest-build"},
        )
    if command == "fixity-scan" and any(key in args for key in ("archive_root", "archiveRoot")):
        _set_archive_path_field(
            "archive_root",
            ("archive_root", "archiveRoot"),
            must_exist=True,
            must_be_dir=True,
        )
    if command in {"fixity-verify", "manifest-build"}:
        _set_archive_path_field(
            "hash_manifest",
            ("hash_manifest", "hashManifest"),
            required=True,
            must_exist=True,
            must_be_file=True,
        )
    if command == "rights-apply":
        _set_archive_path_field(
            "policy_yaml",
            ("policy_yaml", "policyYaml"),
            required=True,
            must_exist=True,
            must_be_file=True,
        )
    if command in {"rights-apply", "bag-build", "dedup-plan", "mets-export", "prov-export", "stac-export"}:
        _set_archive_path_field(
            "manifest_jsonl",
            ("manifest_jsonl", "manifestJsonl"),
            required=True,
            must_exist=True,
            must_be_file=True,
        )
    if command == "bag-validate":
        _set_archive_path_field(
            "bag_dir",
            ("bag_dir", "bagDir"),
            required=True,
            must_exist=True,
            must_be_dir=True,
        )

    for field, keys, _scope in PATH_FIELD_SPECS:
        if field in handled_path_fields:
            continue
        if field in normalized_args:
            continue
        if not any(key in args for key in keys):
            continue
        _set_archive_path_field(field, keys)

    archive_integer_specs: List[Tuple[str, Tuple[str, ...], int, int]] = []
    if command in {"fixity-scan", "fixity-verify"}:
        archive_integer_specs.append(("workers", ("workers",), 1, 1))
    if command == "fixity-verify":
        archive_integer_specs.append(("verify_sample", ("verify_sample", "verifySample"), 0, 0))
    for canonical_field, keys, minimum, default_value in archive_integer_specs:
        value = _pick(args, *keys, default=None)
        if value is None or (isinstance(value, str) and not value.strip()):
            normalized_args[canonical_field] = default_value
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            errors.append(
                _portal_issue(
                    canonical_field,
                    "invalid_archive_integer_option",
                    _portal_safe_error_message("invalid_archive_integer_option"),
                )
            )
            continue
        if parsed < minimum:
            errors.append(
                _portal_issue(
                    canonical_field,
                    "invalid_archive_integer_option",
                    _portal_safe_error_message("invalid_archive_integer_option"),
                )
            )
            continue
        normalized_args[canonical_field] = parsed

    archive_path_error_fields = {str(item.get("field") or "") for item in errors if isinstance(item, Mapping)}
    if (
        command == "fixity-scan"
        and "archive_index" not in archive_path_error_fields
        and "input_dir" not in archive_path_error_fields
        and "archive_root" not in archive_path_error_fields
    ):
        archive_index_text = str(
            _pick(args, "archive_index", "archiveIndex", default=normalized_args.get("archive_index") or "") or ""
        ).strip()
        archive_root_was_explicit = _pick(args, "archive_root", "archiveRoot", default=None) is not None
        archive_root_text = str(
            _pick(
                args,
                "archive_root",
                "archiveRoot",
                default=_pick(args, "input_dir", "inputDir", default=normalized_args.get("input_dir") or ""),
            )
            or ""
        ).strip()
        if archive_index_text and archive_root_text:
            index_preflight = _validate_archive_index_against_root(
                Path(archive_index_text),
                Path(archive_root_text),
                scan_mode=archive_index_scan_mode,
            )
            root_reason = _archive_index_preflight_root_reason(index_preflight)
            if root_reason is not None:
                archive_root_field = "archive_root" if archive_root_was_explicit else "input_dir"
                errors.append(
                    _portal_issue(
                        archive_root_field,
                        "not_a_directory" if root_reason != "archive_root_symlink" else "invalid_path_value",
                        (
                            f"{archive_root_field} must be an existing directory."
                            if root_reason != "archive_root_symlink"
                            else f"{archive_root_field} must be a real directory, not a symlink."
                        ),
                        suggestion=f"Choose an existing directory for {archive_root_field} under the allowed input roots.",
                    )
                )
            elif not index_preflight["ok"]:
                errors.append(
                    _portal_issue(
                        "archive_index",
                        "archive_index_root_mismatch",
                        _archive_index_preflight_message(index_preflight),
                        suggestion="Rebuild the archive index from the selected archive root.",
                    )
                )

    if readiness_snapshot is None:
        readiness_snapshot = _evaluate_pipeline_readiness(
            pipeline,
            normalized_args,
            require_dispatch_inputs=True,
            archive_index_scan_mode=archive_index_scan_mode,
        )

    argv_preview = ""
    if not errors:
        try:
            argv_preview = _format_argv_preview(
                _argv_from_request(
                    {"pipeline": pipeline, "args": normalized_args},
                    execution_args=normalized_args,
                )
            )
        except ValueError as exc:
            reason = _portal_reason_from_exception(exc)
            errors.append(
                _portal_issue(
                    "payload",
                    reason,
                    _portal_safe_error_message(reason),
                )
            )

    return {
        "pipeline": pipeline,
        "normalized_args": normalized_args,
        "execution_args": dict(normalized_args),
        "argv_preview": argv_preview,
        "field_errors": errors,
        "field_warnings": warnings,
        "inactive_fields": [],
        "readiness": readiness_snapshot,
        "estimate_summary": {},
        "debug_bundle_summary": {},
        "next_best_action": _preview_next_best_action(
            pipeline=pipeline,
            errors=errors,
            warnings=warnings,
            readiness_snapshot=readiness_snapshot,
        ),
    }


def _build_config_preview(
    payload: Dict[str, Any],
    *,
    readiness_snapshot: Optional[Dict[str, Any]] = None,
    archive_index_scan_mode: str = "preview",
    portal_actor: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    pipeline = str(payload.get("pipeline") or "").strip()
    args = payload.get("args")
    if not isinstance(args, dict):
        args = {}
    if pipeline == "lux-depth-v3":
        return _build_lux_config_preview(
            args,
            readiness_snapshot=readiness_snapshot,
            portal_actor=portal_actor,
        )
    if pipeline in ARCHIVE_GATE_PIPELINES:
        return _build_archive_config_preview(
            pipeline,
            args,
            readiness_snapshot=readiness_snapshot,
            archive_index_scan_mode=archive_index_scan_mode,
        )
    raise ValueError("Unsupported pipeline")


async def _build_config_preview_threaded(
    payload: Dict[str, Any],
    *,
    readiness_snapshot: Optional[Dict[str, Any]] = None,
    archive_index_scan_mode: str = "preview",
    portal_actor: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return await asyncio.to_thread(
        _build_config_preview,
        payload,
        readiness_snapshot=readiness_snapshot,
        archive_index_scan_mode=archive_index_scan_mode,
        portal_actor=portal_actor,
    )


def _portal_sanitize_metadata(metadata: Any) -> Dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    sanitized: Dict[str, Any] = {}
    for key, value in metadata.items():
        key_text = str(key or "").strip().lower()
        if not _portal_is_token(key_text):
            continue
        if key_text == "pipeline":
            continue
        if isinstance(value, bool):
            sanitized[key_text] = value
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            sanitized[key_text] = value
            continue
        if isinstance(value, float):
            if not math.isfinite(value):
                continue
            sanitized[key_text] = round(value, 4)
            continue
        if isinstance(value, str):
            text = value.strip().lower()
            if _portal_is_token(text):
                sanitized[key_text] = text
    return sanitized


def _portal_actor_from_request(request: Request) -> Dict[str, str]:
    actor: Dict[str, str] = {}
    username = str(request.headers.get("x-tp-actor") or "").strip().lower()
    access_email = str(request.headers.get("x-tp-actor-email") or "").strip().lower()
    role = str(request.headers.get("x-tp-actor-role") or "").strip().lower()
    if username:
        actor["username"] = username
    if access_email:
        actor["accessEmail"] = access_email
    if role:
        actor["role"] = role
    return actor


def _portal_rum_auth_mode(request: Request) -> str:
    return "managed" if _portal_actor_from_request(request) else _auth_mode()


def _portal_request_trace_context(request: Request):
    existing = getattr(request.state, "trace_context", None)
    if existing is not None:
        return existing
    header_value = str(request.headers.get("traceparent") or "").strip()
    try:
        trace_context = get_or_create_trace_context(header_value or None)
    except ValueError:
        trace_context = get_or_create_trace_context(None)
    request.state.trace_context = trace_context
    return trace_context


def _record_portal_event(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    event_type = str(payload.get("event_type") or "").strip().lower()
    if event_type not in PORTAL_ALLOWED_EVENT_TYPES:
        return None, "invalid_event_type"

    pipeline = str(payload.get("pipeline") or "").strip()
    if pipeline and pipeline not in ALLOWED_PIPELINES:
        return None, "invalid_pipeline"

    surface = str(payload.get("surface") or "").strip().lower()
    if surface and surface not in PORTAL_ALLOWED_EVENT_SURFACES:
        return None, "invalid_surface"

    field = str(payload.get("field") or "").strip()
    if field and field not in PORTAL_ALLOWED_EVENT_FIELDS:
        return None, "invalid_field"

    reasons_raw = payload.get("reasons") or []
    reasons: List[str] = []
    if isinstance(reasons_raw, list):
        for item in reasons_raw[:8]:
            token = str(item or "").strip().lower()
            if _portal_is_token(token):
                reasons.append(token)

    record = {
        "schema": "tp.orchestrator.portal_event.v1",
        "timestamp": int(time.time()),
        "event_type": event_type,
        "pipeline": pipeline or "",
        "surface": surface or "",
        "field": field or "",
        "metadata": _portal_sanitize_metadata(payload.get("metadata")),
        "reasons": reasons,
    }
    LOGGER.info("portal_event %s", json.dumps(record, sort_keys=True))
    return record, None


def _persist_portal_event_record(record: Dict[str, Any], log_path: Optional[Path]) -> None:
    if log_path is None:
        return
    encoded_record = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    try:
        log_path = _portal_path_security.validate_portal_telemetry_sink_path(str(log_path), repo_root=REPO_ROOT)
        if os.open not in os.supports_dir_fd:
            raise OSError("dir_fd is required for portal telemetry sink writes")
        no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
        parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | no_follow_flag
        file_flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | no_follow_flag | getattr(os, "O_NONBLOCK", 0)
        expected_parent_stat = os.stat(log_path.parent, follow_symlinks=False)
        with _PORTAL_EVENT_LOG_WRITE_LOCK:
            parent_fd = os.open(log_path.parent, parent_flags)
            try:
                actual_parent_stat = os.fstat(parent_fd)
                if (
                    actual_parent_stat.st_dev != expected_parent_stat.st_dev
                    or actual_parent_stat.st_ino != expected_parent_stat.st_ino
                ):
                    raise OSError("portal telemetry sink parent changed before open")
                try:
                    existing_file_stat = os.stat(log_path.name, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    existing_file_stat = None
                if existing_file_stat is not None and not stat.S_ISREG(existing_file_stat.st_mode):
                    raise OSError("portal telemetry sink target is not a regular file")
                fd = os.open(log_path.name, file_flags, 0o600, dir_fd=parent_fd)
                try:
                    opened_file_stat = os.fstat(fd)
                    if not stat.S_ISREG(opened_file_stat.st_mode):
                        raise OSError("portal telemetry sink is not a regular file")
                    bytes_written = 0
                    while bytes_written < len(encoded_record):
                        chunk_size = os.write(fd, encoded_record[bytes_written:])
                        if chunk_size <= 0:
                            raise OSError("short write while appending portal telemetry")
                        bytes_written += chunk_size
                finally:
                    os.close(fd)
            finally:
                os.close(parent_fd)
    except (OSError, _portal_path_security.PathSecurityValidationError):
        LOGGER.warning(
            "failed to persist portal event telemetry to %s",
            log_path,
            exc_info=True,
        )


def _record_portal_rum(payload: Dict[str, Any], request: Request) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    event_type = str(payload.get("event_type") or "").strip().lower()
    if event_type not in PORTAL_ALLOWED_RUM_EVENT_TYPES:
        return None, "invalid_event_type"

    route = str(payload.get("route") or "").strip()
    if route not in PORTAL_ALLOWED_RUM_ROUTES:
        return None, "invalid_route"

    view = str(payload.get("view") or "").strip().lower()
    if view not in PORTAL_ALLOWED_RUM_VIEWS:
        return None, "invalid_view"

    unit = str(payload.get("unit") or "").strip().lower()
    if unit not in PORTAL_ALLOWED_RUM_UNITS:
        return None, "invalid_unit"

    value = payload.get("value")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
        return None, "invalid_value"

    metric = str(payload.get("metric") or "").strip().lower()
    allowed_metrics = PORTAL_ALLOWED_RUM_METRICS.get(event_type)
    if allowed_metrics is None:
        # Defense-in-depth: unknown event types already fail PORTAL_ALLOWED_RUM_EVENTS
        # above, but if the metric allowlist is missing the event is not acceptable.
        return None, "invalid_metric"
    if allowed_metrics:
        if metric not in allowed_metrics:
            return None, "invalid_metric"
    elif metric:
        # An empty allowlist means "this event type carries no metric"; reject
        # any caller-supplied token rather than silently accepting it.
        return None, "invalid_metric"

    actor = _portal_actor_from_request(request)
    cohort_key = _portal_rollout_cohort_key(actor)
    trace_context = _portal_request_trace_context(request)
    record = {
        "schema": "tp.orchestrator.portal_rum.v1",
        "timestamp": int(time.time()),
        "event_type": event_type,
        "route": route,
        "view": view,
        "metric": metric,
        "value": round(float(value), 4),
        "unit": unit,
        "metadata": _portal_sanitize_metadata(payload.get("metadata")),
        "trace_id": trace_context.trace_id,
        "cohort_bucket": _stable_rollout_bucket(cohort_key),
        "auth_mode": _portal_rum_auth_mode(request),
    }
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("portal_rum %s", json.dumps(record, sort_keys=True))
    return record, None


def _is_mutating_job_endpoint(method: str, path: str) -> bool:
    if method != "POST":
        return False
    if path in {"/v1/jobs", "/v2/jobs"}:
        return True
    return bool(re.fullmatch(r"/v[12]/jobs/[^/]+/cancel", path))


def _is_job_events_endpoint(path: str) -> bool:
    return bool(re.fullmatch(r"/v[12]/jobs/[^/]+/events", path))


def _is_protected_job_endpoint(path: str) -> bool:
    return path in {"/v1/jobs", "/v2/jobs"} or path.startswith(("/v1/jobs/", "/v2/jobs/"))


def _is_protected_api_key_endpoint(path: str) -> bool:
    normalized_path = path.rstrip("/") or "/"
    if _is_protected_job_endpoint(normalized_path):
        return True
    return normalized_path in {
        "/v1/config-metadata",
        "/v1/config-preview",
        "/v1/portal/events",
        "/v1/portal/rum",
        "/v1/uploads/staging",
    }


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


class _RateLimitDecision(NamedTuple):
    """Outcome of a rate-limit check, including retry hints for response headers."""

    limited: bool
    retry_after_seconds: int
    reset_epoch_seconds: int


def _is_rate_limited(client_ip: str, now: float) -> _RateLimitDecision:
    if RATE_LIMIT_PER_MINUTE <= 0:
        return _RateLimitDecision(
            limited=False,
            retry_after_seconds=0,
            reset_epoch_seconds=int(now),
        )

    timestamps = RATE_LIMIT_BUCKETS.get(client_ip)
    if timestamps is None:
        timestamps = deque()
        RATE_LIMIT_BUCKETS[client_ip] = timestamps

    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    while timestamps and timestamps[0] < cutoff:
        timestamps.popleft()

    if len(timestamps) >= RATE_LIMIT_PER_MINUTE:
        # The oldest entry exits the sliding window at oldest + WINDOW; that
        # is when a new request slot frees up for this client. Clamp to >= 1
        # so Retry-After is always a positive integer per HTTP semantics.
        oldest = timestamps[0]
        reset_epoch = oldest + RATE_LIMIT_WINDOW_SECONDS
        retry_after = max(1, math.ceil(reset_epoch - now))
        return _RateLimitDecision(
            limited=True,
            retry_after_seconds=int(retry_after),
            reset_epoch_seconds=int(reset_epoch),
        )

    timestamps.append(now)
    return _RateLimitDecision(
        limited=False,
        retry_after_seconds=0,
        reset_epoch_seconds=int(now + RATE_LIMIT_WINDOW_SECONDS),
    )


def _rate_limit_response_headers(
    *,
    limit: int,
    remaining: int,
    retry_after: int,
    reset_epoch: int,
) -> Dict[str, str]:
    """Build the standard rate-limit response header set for a 429 reply.

    Header set (additive, envelope-compatible):
      Retry-After          integer seconds (>= 1)
      X-RateLimit-Limit    integer request/concurrency limit (>= 0)
      X-RateLimit-Remaining integer requests left in window (>= 0)
      X-RateLimit-Reset    unix epoch seconds when the window/slot frees up
    """
    safe_retry_after = max(1, int(retry_after))
    safe_limit = max(0, int(limit))
    safe_remaining = max(0, int(remaining))
    safe_reset = max(0, int(reset_epoch))
    return {
        "Retry-After": str(safe_retry_after),
        "X-RateLimit-Limit": str(safe_limit),
        "X-RateLimit-Remaining": str(safe_remaining),
        "X-RateLimit-Reset": str(safe_reset),
    }


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
        if _is_versioned_api_path(request.url.path):
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

    limit_bytes = _request_body_limit_bytes(request.url.path)
    if size > limit_bytes:
        if _is_versioned_api_path(request.url.path):
            return _error_response(
                413,
                code="REQUEST_TOO_LARGE",
                message=_request_too_large_message(request.url.path),
                details={
                    "path": request.url.path,
                    "max_request_bytes": limit_bytes,
                },
            )
        return JSONResponse(
            status_code=413,
            content={
                "detail": _request_too_large_message(request.url.path),
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
    limit_bytes = _request_body_limit_bytes(request.url.path)

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
            if consumed > limit_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=_request_too_large_message(request.url.path),
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
    job = JOBS.get(job_id)
    if job is not None:
        job.last_event_at = _now()

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


def _signal_process_tree(
    proc: asyncio.subprocess.Process,
    sig: int,
) -> bool:
    """Deliver *sig* to the subprocess's full process group on POSIX.

    Returns True when the signal was delivered via :func:`os.killpg`; False
    when the caller must fall back to the direct ``proc`` methods (e.g. the
    spawn did not create a new session, or we are on Windows).
    """

    if os.name == "nt":
        return False
    pid = proc.pid
    if pid is None:
        return False
    try:
        pgid = os.getpgid(pid)
    except (OSError, ProcessLookupError):
        return False
    # Only escalate to the process group if this subprocess is in its own
    # session; otherwise we could accidentally signal the orchestrator itself.
    if pgid != pid:
        return False
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return True


async def _terminate_process(
    proc: asyncio.subprocess.Process,
    grace_seconds: float = CANCEL_GRACE_SECONDS,
) -> None:
    if proc.returncode is not None:
        return
    if not _signal_process_tree(proc, signal.SIGTERM):
        try:
            proc.terminate()
        except ProcessLookupError:
            return
        except Exception:
            return

    try:
        await asyncio.wait_for(proc.wait(), timeout=grace_seconds)
    except asyncio.TimeoutError:
        if not _signal_process_tree(proc, signal.SIGKILL):
            try:
                proc.kill()
            except ProcessLookupError:
                return
            except Exception:
                return
        await proc.wait()


async def _request_cancel(job: Job) -> None:
    already_requested = job.cancel_requested
    job.cancel_requested = True

    # Phase 2.C - broker-mediated cancel. When the broker owns the
    # lifecycle, route through ``broker.cancel`` so:
    #   - queued (pre-lease) jobs are dropped from the broker's FIFO
    #     atomically; the orchestrator publishes the terminal
    #     cancelled-done event here because no worker will ever see
    #     this job_id.
    #   - leased (in-flight) jobs surface ``LeaseStatus.cancelled``
    #     via the worker's next heartbeat, which sets the executor's
    #     cancellation_event → ``Job.cancel_requested`` bridge in
    #     ``_orchestrator_job_executor``. ``_run_job``'s existing
    #     readline loop then terminates the subprocess on the next
    #     iteration, identical to today's in-band cancel.
    if USE_QUEUE_BROKER:
        try:
            broker = get_queue_broker()
        except Exception:  # noqa: BLE001 - broker construction is best-effort
            broker = None
        if broker is not None:
            try:
                broker_knew_job = await broker.cancel(job.id)
            except Exception:  # noqa: BLE001 - cancel is best-effort
                LOGGER.exception("broker cancel failed for job %s", job.id)
                broker_knew_job = False
            if broker_knew_job:
                # Distinguish pre-lease vs in-flight by inspecting the
                # broker AFTER cancel(): the memory + Redis backends
                # both clear queued entries atomically on pre-lease
                # cancel and leave the lease entry intact on in-flight
                # cancel (only flipping cancellation_requested). The
                # post-cancel ``leased_job_ids`` is therefore the
                # atomic source of truth — no acquire-during-cancel
                # race can sneak a job past us here.
                try:
                    still_leased = job.id in await broker.leased_job_ids()
                except Exception:  # noqa: BLE001 - inspection is best-effort
                    LOGGER.exception("broker leased_job_ids failed after cancel for job %s", job.id)
                    still_leased = True  # be conservative: assume worker has it
                if not still_leased:
                    # Pre-lease cancel: no worker will publish the
                    # terminal done event for this job — the
                    # orchestrator does it. ``_publish_cancellation_terminal``
                    # is itself idempotent against an already-finished
                    # Job, so the narrow worker-released-between-cancel-
                    # and-check window collapses to a no-op.
                    await _publish_cancellation_terminal(job)
                    return

    if job.proc is None or job.proc.returncode is not None:
        return

    if not already_requested:
        try:
            await _publish_event(
                job.id,
                "state",
                {
                    "id": job.id,
                    "state": job.state,
                    "cancel_requested": True,
                },
            )
        except Exception:  # noqa: BLE001 - event publish is best-effort
            pass

    if job.terminate_task is None or job.terminate_task.done():
        job.terminate_task = asyncio.create_task(_terminate_process(job.proc))


def _job_output_dir(job: Job) -> Optional[Path]:
    request_payload = (
        job.effective_request if isinstance(job.effective_request, dict) and job.effective_request else job.request
    )
    args = request_payload.get("args") if isinstance(request_payload, dict) else None
    if not isinstance(args, dict):
        return None
    output_dir = str(
        _pick(args, "output_dir", "outputDir", default=""),
    ).strip()
    if not output_dir:
        return None
    try:
        return _resolve_allowed_request_path(output_dir, ALLOWED_OUTPUT_ROOTS)
    except ValueError:
        return None


def _resolve_portal_asset(asset_path: str) -> PortalAssetSpec:
    normalized = str(asset_path or "").strip()
    if not normalized:
        raise FileNotFoundError("missing portal asset path")

    try:
        candidate = PORTAL_ASSET_MANIFEST[normalized]
    except KeyError as exc:
        raise FileNotFoundError("portal asset not found") from exc

    if not candidate.path.is_file():
        raise FileNotFoundError("portal asset not found")
    return candidate


_infer_artifact_type = _portal_job_artifacts._infer_artifact_type
_artifact_content_type = _portal_job_artifacts._artifact_content_type
_artifact_media_kind = _portal_job_artifacts._artifact_media_kind
_artifact_is_previewable = _portal_job_artifacts._artifact_is_previewable
_BROWSER_PREVIEWABLE_MIME_TYPES = _portal_job_artifacts._BROWSER_PREVIEWABLE_MIME_TYPES
_artifact_is_browser_previewable = _portal_job_artifacts._artifact_is_browser_previewable
_artifact_preview_proxy_path = _portal_job_artifacts._artifact_preview_proxy_path
_add_artifact_preview_proxy_lookup = _portal_job_artifacts._add_artifact_preview_proxy_lookup
_safe_artifact_attachment_filename = _portal_job_artifacts._safe_artifact_attachment_filename
_artifact_response_headers = _portal_job_artifacts._artifact_response_headers
_artifact_display_label = _portal_job_artifacts._artifact_display_label
_STEM_NOISE_RE = _portal_job_artifacts._STEM_NOISE_RE
_artifact_compare_group = _portal_job_artifacts._artifact_compare_group
_artifact_display_hint = _portal_job_artifacts._artifact_display_hint
_artifact_url = _portal_job_artifacts._artifact_url
_coerce_nonnegative_int = _portal_job_artifacts._coerce_nonnegative_int
_captioning_artifact_counts_from_paths = _portal_job_artifacts._captioning_artifact_counts_from_paths
_captioning_artifact_counts_from_run_card = _portal_job_artifacts._captioning_artifact_counts_from_run_card
_captioning_artifact_counts_from_job_artifacts = _portal_job_artifacts._captioning_artifact_counts_from_job_artifacts


def _artifact_fingerprint(
    path: Path,
    size_bytes: Optional[int],
) -> Tuple[Optional[str], str]:
    return _portal_job_artifacts._artifact_fingerprint(
        path,
        size_bytes,
        max_bytes=ARTIFACT_FINGERPRINT_MAX_BYTES,
        chunk_bytes=_ARTIFACT_FINGERPRINT_CHUNK_BYTES,
    )


def _serialize_indexed_artifact(
    *,
    job_id: str,
    relative_path: str,
    path: Path,
) -> Dict[str, Any]:
    return _portal_job_artifacts._serialize_indexed_artifact(
        job_id=job_id,
        relative_path=relative_path,
        path=path,
        fingerprint_max_bytes=ARTIFACT_FINGERPRINT_MAX_BYTES,
        fingerprint_chunk_bytes=_ARTIFACT_FINGERPRINT_CHUNK_BYTES,
    )


def _fastvlm_model_role_from_value(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in ALLOWED_VLM_CAPTIONING_MODEL_ROLES else "custom"


def _normalize_fastvlm_run_status(
    raw_status: Any,
    *,
    artifact_counts: Optional[Mapping[str, int]] = None,
    requested: bool = False,
) -> Optional[Dict[str, Any]]:
    counts = {
        "sidecar_count": _coerce_nonnegative_int((artifact_counts or {}).get("sidecar_count")) or 0,
        "raw_count": _coerce_nonnegative_int((artifact_counts or {}).get("raw_count")) or 0,
        "proxy_count": _coerce_nonnegative_int((artifact_counts or {}).get("proxy_count")) or 0,
    }
    if not isinstance(raw_status, Mapping):
        if not requested and not any(counts.values()):
            return None
        raw_status = {"enabled": True, "status": "requested" if requested else "succeeded"}

    enabled = _as_bool(raw_status.get("enabled"), requested or any(counts.values()))
    backend = str(raw_status.get("backend") or "fastvlm").strip().lower() or "fastvlm"
    status_text = str(raw_status.get("status") or "").strip().lower()
    normalized_status = FASTVLM_RUNTIME_STATUS_ALIASES.get(status_text, "")
    policy_violation = raw_status.get("used_for_quality_gate") is True

    sidecar_count = max(counts["sidecar_count"], _coerce_nonnegative_int(raw_status.get("sidecar_count")) or 0)
    raw_count = max(counts["raw_count"], _coerce_nonnegative_int(raw_status.get("raw_count")) or 0)
    proxy_count = max(counts["proxy_count"], _coerce_nonnegative_int(raw_status.get("proxy_count")) or 0)
    failed_count = _coerce_nonnegative_int(raw_status.get("failed_count")) or 0

    if policy_violation:
        normalized_status = "failed"
        failed_count = max(failed_count, 1)
    elif backend != "fastvlm":
        normalized_status = "unsupported_backend"
    elif normalized_status not in FASTVLM_RUN_STATUS_VALUES:
        if not enabled:
            normalized_status = "off"
        elif failed_count > 0:
            normalized_status = "failed"
        elif sidecar_count > 0:
            normalized_status = "succeeded"
        elif requested:
            normalized_status = "requested"
        else:
            normalized_status = "skipped"

    if normalized_status == "off":
        enabled = False
    if normalized_status == "succeeded" and sidecar_count == 0 and not requested:
        normalized_status = "skipped"

    normalized: Dict[str, Any] = {
        "status": normalized_status,
        "enabled": bool(enabled),
        "backend": backend,
        "model_role": str(
            raw_status.get("model_role") or _fastvlm_model_role_from_value(raw_status.get("model") or "")
        ).strip()
        or "custom",
        "model_id": raw_status.get("model_id") if raw_status.get("model_id") is not None else None,
        "model_path": str(raw_status.get("model_path") or "").strip() or None,
        "role": "advisory",
        "sidecar_count": sidecar_count,
        "raw_count": raw_count,
        "proxy_count": proxy_count,
        "failed_count": failed_count,
        "used_for_quality_gate": False,
    }
    if policy_violation:
        normalized["policy_violation"] = True
        normalized["quality_gate_claimed"] = True
        normalized["error"] = "captioning_status.used_for_quality_gate must be false"
    elif raw_status.get("error"):
        normalized["error"] = str(raw_status.get("error"))
    return normalized


def _requested_fastvlm_captioning_status(job: Job) -> Optional[Dict[str, Any]]:
    args: Mapping[str, Any] = {}
    for request in (job.effective_request, job.request):
        candidate = request.get("args") if isinstance(request, Mapping) else None
        if isinstance(candidate, Mapping) and candidate:
            args = candidate
            break
    if not args:
        return None
    enabled = _as_bool(_pick(args, "vlm_captioning_enabled", "vlmCaptioningEnabled", default=False), False)
    if not enabled:
        return None
    model = str(_pick(args, "vlm_captioning_model", "vlmCaptioningModel", default="default") or "default").strip()
    backend = str(_pick(args, "vlm_captioning_backend", "vlmCaptioningBackend", default="fastvlm") or "fastvlm").strip()
    return _normalize_fastvlm_run_status(
        {
            "enabled": True,
            "status": "requested",
            "backend": backend,
            "model_role": _fastvlm_model_role_from_value(model),
            "model_id": None,
            "model_path": model if "/" in model or "\\" in model or model.startswith(".") else None,
            "used_for_quality_gate": False,
        },
        requested=True,
    )


def _load_bounded_json_object(path: Path, *, max_bytes: int = JOB_RUN_SUMMARY_MAX_BYTES) -> Optional[Dict[str, Any]]:
    return _portal_job_artifacts._load_bounded_json_object(path, max_bytes=max_bytes)


def _load_bounded_run_card_payload(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    return _portal_job_artifacts._load_bounded_run_card_payload(
        path,
        max_bytes=JOB_RUN_SUMMARY_MAX_BYTES,
    )


def _summarize_run_card_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"source": "run_card"}

    batch_id = str(payload.get("batch_id") or "").strip()
    if batch_id:
        summary["batch_id"] = batch_id

    total_images = _coerce_nonnegative_int(payload.get("total_images"))
    success_count = _coerce_nonnegative_int(payload.get("success_count"))
    error_count = _coerce_nonnegative_int(payload.get("error_count"))
    artifact_index = payload.get("artifact_index")
    artifact_index_count = len(artifact_index) if isinstance(artifact_index, list) else None

    if total_images is None and success_count is not None and error_count is not None:
        total_images = success_count + error_count

    if total_images is not None:
        summary["total_images"] = total_images
    if success_count is not None:
        summary["success_count"] = success_count
    if error_count is not None:
        summary["error_count"] = error_count
    if artifact_index_count is not None:
        summary["artifact_index_count"] = artifact_index_count

    reviewable_outputs = bool((success_count or 0) > 0)
    partial = reviewable_outputs and bool((error_count or 0) > 0)
    summary["reviewable_outputs"] = reviewable_outputs
    summary["partial"] = partial
    captioning_status = _normalize_fastvlm_run_status(
        payload.get("captioning_status"),
        artifact_counts=_captioning_artifact_counts_from_run_card(payload),
    )
    if captioning_status is not None:
        summary["captioning_status"] = captioning_status

    return summary


def _summarize_batch_manifest_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"source": "batch_manifest"}

    batch_id = str(payload.get("batch_id") or "").strip()
    if batch_id:
        summary["batch_id"] = batch_id

    results = payload.get("results")
    if isinstance(results, list):
        success_count = sum(1 for item in results if isinstance(item, dict) and item.get("status") == "ok")
        error_count = sum(1 for item in results if isinstance(item, dict) and item.get("status") == "error")
    else:
        success_count = 0
        error_count = 0

    stats = payload.get("stats")
    total_images = None
    if isinstance(stats, Mapping):
        total_images = _coerce_nonnegative_int(stats.get("total_images"))
    if total_images is None and isinstance(results, list):
        total_images = len(results)

    if total_images is not None:
        summary["total_images"] = total_images
    summary["success_count"] = success_count
    summary["error_count"] = error_count
    summary["reviewable_outputs"] = success_count > 0
    summary["partial"] = success_count > 0 and error_count > 0

    return summary


_artifact_batch_hint = _portal_job_artifacts._artifact_batch_hint
_artifact_recency_key = _portal_job_artifacts._artifact_recency_key
_find_newest_artifact_path = _portal_job_artifacts._find_newest_artifact_path
_resolve_artifact_path_within_output_dir = _portal_job_artifacts._resolve_artifact_path_within_output_dir


def _resolve_job_run_metadata(job: Job) -> Optional[JobRunMetadata]:
    return _portal_job_artifacts._resolve_job_run_metadata(
        _job_output_dir(job),
        max_bytes=JOB_RUN_SUMMARY_MAX_BYTES,
    )


def _build_scoped_job_artifacts(
    *,
    job: Job,
    output_dir: Path,
    candidate_paths: List[Path],
) -> Tuple[List[Dict[str, Any]], Dict[str, Path], bool]:
    return _portal_job_artifacts._build_scoped_job_artifacts(
        job_id=job.id,
        output_dir=output_dir,
        candidate_paths=candidate_paths,
        max_indexed_artifacts=MAX_INDEXED_ARTIFACTS,
        fingerprint_max_bytes=ARTIFACT_FINGERPRINT_MAX_BYTES,
        fingerprint_chunk_bytes=_ARTIFACT_FINGERPRINT_CHUNK_BYTES,
    )


def _build_scoped_job_artifacts_from_run_metadata(
    job: Job,
    metadata: JobRunMetadata,
) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Path], bool]]:
    return _portal_job_artifacts._build_scoped_job_artifacts_from_run_metadata(
        job_id=job.id,
        metadata=metadata,
        max_indexed_artifacts=MAX_INDEXED_ARTIFACTS,
        fingerprint_max_bytes=ARTIFACT_FINGERPRINT_MAX_BYTES,
        fingerprint_chunk_bytes=_ARTIFACT_FINGERPRINT_CHUNK_BYTES,
    )


def _refresh_job_run_summary(job: Job) -> Dict[str, Any]:
    if not job:
        return {}

    existing_summary = dict(job.run_summary) if isinstance(job.run_summary, dict) and job.run_summary else {}
    metadata = _resolve_job_run_metadata(job)
    summary: Dict[str, Any] = {}
    if metadata is not None and metadata.run_card_payload is not None:
        summary = _summarize_run_card_payload(metadata.run_card_payload)

    if not summary and metadata is not None and metadata.batch_manifest_payload is not None:
        summary = _summarize_batch_manifest_payload(metadata.batch_manifest_payload)

    if not summary and existing_summary:
        summary = existing_summary

    if summary:
        artifact_counts = _captioning_artifact_counts_from_job_artifacts(job.artifacts)
        existing_captioning_counts = summary.get("captioning_status")
        if isinstance(existing_captioning_counts, Mapping):
            artifact_counts = {
                "sidecar_count": max(
                    artifact_counts["sidecar_count"],
                    _coerce_nonnegative_int(existing_captioning_counts.get("sidecar_count")) or 0,
                ),
                "raw_count": max(
                    artifact_counts["raw_count"],
                    _coerce_nonnegative_int(existing_captioning_counts.get("raw_count")) or 0,
                ),
                "proxy_count": max(
                    artifact_counts["proxy_count"],
                    _coerce_nonnegative_int(existing_captioning_counts.get("proxy_count")) or 0,
                ),
            }
        raw_captioning_status = None
        if metadata is not None and metadata.run_card_payload is not None:
            raw_captioning_status = metadata.run_card_payload.get("captioning_status")
        if raw_captioning_status is None:
            raw_captioning_status = existing_summary.get("captioning_status")
        captioning_status = _normalize_fastvlm_run_status(
            raw_captioning_status,
            artifact_counts=artifact_counts,
        )
        if captioning_status is not None:
            summary["captioning_status"] = captioning_status

    job.run_summary = summary

    if job.state != "canceled" and summary.get("partial"):
        job.state = "partial"
        existing_code = ""
        if isinstance(job.error, dict):
            existing_code = str(job.error.get("code") or "").strip().upper()
        if existing_code in {"", "RUNNER_EXIT_NONZERO"}:
            total_images = summary.get("total_images")
            success_count = summary.get("success_count")
            error_count = summary.get("error_count")
            detail_text = "outputs remain reviewable"
            if (
                isinstance(total_images, int)
                and isinstance(success_count, int)
                and isinstance(error_count, int)
                and total_images > 0
            ):
                detail_text = f"{error_count}/{total_images} images failed; " f"{success_count} outputs remain reviewable"
            job.error = _error_obj(
                "RUNNER_PARTIAL_FAILURE",
                detail_text,
                {
                    "exit_code": job.exit_code,
                    "total_images": total_images,
                    "success_count": success_count,
                    "error_count": error_count,
                },
            )

    return summary


def _serialized_job_run_summary(job: Job) -> Optional[Dict[str, Any]]:
    if job.state not in ACTIVE_JOB_STATES and job.state != "canceled":
        _refresh_job_run_summary(job)
    summary = dict(job.run_summary) if isinstance(job.run_summary, dict) and job.run_summary else {}
    if job.state in ACTIVE_JOB_STATES:
        requested_status = _requested_fastvlm_captioning_status(job)
        if requested_status is not None:
            summary["captioning_status"] = requested_status
    return summary or None


def _validate_resolved_job_artifact_path(
    job: Job,
    resolved_artifact: Path,
) -> tuple[Path, Path, str]:
    return _portal_job_artifacts._validate_resolved_job_artifact_path(
        _job_output_dir(job),
        resolved_artifact,
    )


_normalize_artifact_relative_path = _portal_job_artifacts._normalize_artifact_relative_path


def _hydrate_artifact_lookup_from_items(job: Job) -> Dict[str, Path]:
    items = job.artifacts.get("items") if isinstance(job.artifacts, dict) else None
    if not isinstance(items, list) or not items:
        return {}

    output_dir = _job_output_dir(job)
    if output_dir is None:
        return {}

    lookup = _portal_job_artifacts._hydrate_artifact_lookup_from_items(
        items=items,
        output_dir=output_dir,
    )
    job.artifact_lookup = lookup
    return lookup


def _index_job_artifacts(job: Job) -> List[Dict[str, Any]]:
    result = _portal_job_artifacts._index_job_artifacts(
        job_id=job.id,
        output_dir=_job_output_dir(job),
        max_indexed_artifacts=MAX_INDEXED_ARTIFACTS,
        fingerprint_max_bytes=ARTIFACT_FINGERPRINT_MAX_BYTES,
        fingerprint_chunk_bytes=_ARTIFACT_FINGERPRINT_CHUNK_BYTES,
        run_summary_max_bytes=JOB_RUN_SUMMARY_MAX_BYTES,
    )
    job.artifacts = result.artifacts
    job.artifact_lookup = result.artifact_lookup
    return result.items


def _job_api_prefix(api_version: str = "v1") -> str:
    return "/v2/jobs" if str(api_version) == "v2" else "/v1/jobs"


def _job_events_url(job_id: str, *, api_version: str = "v1") -> str:
    return f"{_job_api_prefix(api_version)}/{job_id}/events"


def _serialize_job(job: Job, *, include_logs: bool = True, api_version: str = "v1") -> Dict[str, Any]:
    run_summary = _serialized_job_run_summary(job)
    data = {
        "id": job.id,
        "pipeline": str(job.request.get("pipeline") or ""),
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "state": job.state,
        "progress": job.progress,
        "exit_code": job.exit_code,
        "events_url": _job_events_url(job.id, api_version=api_version),
        "artifacts": job.artifacts,
        "error": job.error,
        "run_summary": run_summary,
        "last_event_at": job.last_event_at,
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
    except (TypeError, ValueError):
        raise ValueError("Invalid archive integer option") from None
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
            default=str(Path(output_dir) / "archive_index_normalized.csv.gz"),
            allowed_roots=ALLOWED_PATH_ROOTS,
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
        argv.extend(["--strict", "--strict-identity"])
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
            default=str(Path(output_dir) / "archive_index_normalized.csv.gz"),
            allowed_roots=ALLOWED_PATH_ROOTS,
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


def _argv_from_request(
    payload: Dict[str, Any],
    *,
    execution_args: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Build argv securely (no shell).
    Input validation: allowlist pipeline/backend/quality, require paths.
    """
    pipeline = str(payload.get("pipeline") or "").strip()
    if pipeline not in ALLOWED_PIPELINES:
        raise _PortalValidationReasonError("Unsupported pipeline", reason="unsupported_pipeline")

    args = execution_args if isinstance(execution_args, dict) else payload.get("args")
    if not isinstance(args, dict):
        args = {}
    if execution_args is None:
        args, _, path_errors = _normalize_operator_payload_paths(str(pipeline), args)
        if path_errors:
            raise _PortalValidationReasonError(
                "Path shorthand traversal disallowed",
                reason="path_shorthand_traversal_disallowed",
            )

    input_dir_raw = str(
        _pick(args, "input_dir", "inputDir", default=""),
    ).strip()
    output_dir_raw = str(
        _pick(args, "output_dir", "outputDir", default=""),
    ).strip()
    if not input_dir_raw or not output_dir_raw:
        raise _PortalValidationReasonError(
            "input_dir and output_dir are required",
            reason="missing_required_paths",
        )
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

    if pipeline == "lux-depth-v3":
        argv = [
            *_lux_depth_runner_command(),
            "--input-dir",
            input_dir,
            "--output-dir",
            output_dir,
        ]
    else:
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
        model_key = ""
        da3_model_spec: Optional[Any] = None
        if backend == "da3":
            model_key = _canonical_da3_model_key(
                _pick(
                    args,
                    "model_key",
                    "modelKey",
                    default=PORTAL_DEFAULT_DA3_MODEL_KEY,
                )
            )
            da3_model_spec = _da3_model_spec_for_portal_key(model_key)
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
        sam2_tiling_enabled = _pick(
            args,
            "sam2_tiling_enabled",
            "sam2TilingEnabled",
            default=False,
        )
        sam2_tile_size_px_raw = _pick(args, "sam2_tile_size_px", "sam2TileSizePx")
        sam2_overlap_px_raw = _pick(args, "sam2_overlap_px", "sam2OverlapPx")
        sam2_global_pass_longest_side_raw = _pick(
            args,
            "sam2_global_pass_longest_side",
            "sam2GlobalPassLongestSide",
        )
        sam2_max_concurrency_raw = _pick(args, "sam2_max_concurrency", "sam2MaxConcurrency")
        sam2_points_per_side_raw = _pick(args, "sam2_points_per_side", "sam2PointsPerSide")
        sam2_points_per_batch_raw = _pick(args, "sam2_points_per_batch", "sam2PointsPerBatch")
        sam2_pred_iou_thresh_raw = _pick(args, "sam2_pred_iou_thresh", "sam2PredIouThresh")
        sam2_stability_score_thresh_raw = _pick(
            args,
            "sam2_stability_score_thresh",
            "sam2StabilityScoreThresh",
        )
        sam2_crop_n_layers_raw = _pick(args, "sam2_crop_n_layers", "sam2CropNLayers")
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
        segmentation_cache = (
            str(
                _pick(
                    args,
                    "segmentation_cache",
                    "segmentationCache",
                    default="read_write",
                )
                or "read_write"
            )
            .strip()
            .lower()
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
            raise _PortalValidationReasonError("Invalid quality_tier", reason="invalid_quality_tier")
        if backend not in ALLOWED_BACKENDS:
            raise _PortalValidationReasonError("Invalid depth_backend", reason="invalid_depth_backend")
        if backend == "da3" and da3_model_spec is None:
            raise _PortalValidationReasonError("Invalid model_key", reason="invalid_model_key")
        if (
            backend == "da3"
            and da3_model_spec is not None
            and bool(getattr(da3_model_spec, "requires_non_commercial_ok", False))
            and not _as_bool(_pick(args, "non_commercial_ok", "nonCommercialOk", default=False), default=False)
        ):
            raise _PortalValidationReasonError(
                "DA3 model requires non-commercial acknowledgment",
                reason="da3_model_non_commercial_required",
            )
        if segmentation_backend not in ALLOWED_SEGMENTATION_BACKENDS:
            raise _PortalValidationReasonError(
                "Invalid segmentation_backend",
                reason="invalid_segmentation_backend",
            )
        if segmentation_backend == "sam2" and sam2_model_size not in ALLOWED_SAM2_MODEL_SIZES:
            raise _PortalValidationReasonError("Invalid sam2_model_size", reason="invalid_sam2_model_size")
        if segmentation_cache not in ALLOWED_SEGMENTATION_CACHE_POLICIES:
            raise _PortalValidationReasonError("Invalid segmentation_cache", reason="invalid_segmentation_cache")
        if grouping_mode not in ALLOWED_GROUPING_MODES:
            raise _PortalValidationReasonError("Invalid grouping_mode")

        reconstruction_tier_value = ""
        if reconstruction_tier is not None and str(reconstruction_tier).strip():
            reconstruction_tier_value = str(reconstruction_tier).strip().lower()
            if reconstruction_tier_value not in ALLOWED_RECONSTRUCTION_TIERS:
                raise _PortalValidationReasonError(
                    "Invalid reconstruction_tier",
                    reason="invalid_reconstruction_tier",
                )

        raw_ingest_mode_value = ""
        if raw_ingest_mode is not None and str(raw_ingest_mode).strip():
            raw_ingest_mode_value = str(raw_ingest_mode).strip().lower()
            if raw_ingest_mode_value not in ALLOWED_RAW_INGEST_MODES:
                raise _PortalValidationReasonError("Invalid raw_ingest_mode", reason="invalid_raw_ingest_mode")

        raw_wb_mode_value = ""
        if raw_wb_mode is not None and str(raw_wb_mode).strip():
            raw_wb_mode_value = str(raw_wb_mode).strip().lower()
            if raw_wb_mode_value not in ALLOWED_RAW_WB_MODES:
                raise _PortalValidationReasonError("Invalid raw_wb_mode", reason="invalid_raw_wb_mode")

        raw_demosaic_value = ""
        if raw_demosaic is not None and str(raw_demosaic).strip():
            raw_demosaic_value = str(raw_demosaic).strip().upper()
            if not _is_valid_demosaic_name(raw_demosaic_value):
                raise _PortalValidationReasonError("Invalid raw_demosaic", reason="invalid_raw_demosaic")

        log_level_value = ""
        if log_level_raw is not None and str(log_level_raw).strip():
            log_level_value = str(log_level_raw).strip().upper()
            if log_level_value not in ALLOWED_LOG_LEVELS:
                raise _PortalValidationReasonError("Invalid log_level", reason="invalid_log_level")

        def _parse_optional_positive_int(
            value: Any,
            field_name: str,
        ) -> Optional[int]:
            if value is None or (isinstance(value, str) and not value.strip()):
                return None
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                raise _PortalValidationReasonError(f"Invalid {field_name}") from None
            if parsed < 1:
                raise _PortalValidationReasonError(f"Invalid {field_name}")
            return parsed

        def _parse_optional_non_negative_int(
            value: Any,
            field_name: str,
        ) -> Optional[int]:
            if value is None or (isinstance(value, str) and not value.strip()):
                return None
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                raise _PortalValidationReasonError(f"Invalid {field_name}") from None
            if parsed < 0:
                raise _PortalValidationReasonError(f"Invalid {field_name}")
            return parsed

        def _parse_optional_probability(
            value: Any,
            field_name: str,
        ) -> Optional[float]:
            if value is None or (isinstance(value, str) and not value.strip()):
                return None
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                raise _PortalValidationReasonError(f"Invalid {field_name}") from None
            if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
                raise _PortalValidationReasonError(f"Invalid {field_name}")
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
        sam2_tile_size_px = _parse_optional_positive_int(sam2_tile_size_px_raw, "sam2_tile_size_px")
        sam2_overlap_px = _parse_optional_non_negative_int(sam2_overlap_px_raw, "sam2_overlap_px")
        sam2_global_pass_longest_side = _parse_optional_positive_int(
            sam2_global_pass_longest_side_raw,
            "sam2_global_pass_longest_side",
        )
        sam2_max_concurrency = _parse_optional_positive_int(sam2_max_concurrency_raw, "sam2_max_concurrency")
        sam2_points_per_side = _parse_optional_positive_int(sam2_points_per_side_raw, "sam2_points_per_side")
        sam2_points_per_batch = _parse_optional_positive_int(sam2_points_per_batch_raw, "sam2_points_per_batch")
        sam2_pred_iou_thresh = _parse_optional_probability(sam2_pred_iou_thresh_raw, "sam2_pred_iou_thresh")
        sam2_stability_score_thresh = _parse_optional_probability(
            sam2_stability_score_thresh_raw,
            "sam2_stability_score_thresh",
        )
        sam2_crop_n_layers = _parse_optional_non_negative_int(sam2_crop_n_layers_raw, "sam2_crop_n_layers")
        if sam2_tile_size_px is not None and sam2_overlap_px is not None and sam2_overlap_px >= sam2_tile_size_px:
            raise _PortalValidationReasonError("Invalid sam2_overlap_px")

        sam2_checkpoint_path = ""
        if sam2_checkpoint_path_raw is not None and str(sam2_checkpoint_path_raw).strip():
            if segmentation_backend == "sam2":
                validation = _resolve_managed_sam2_checkpoint_validation(str(sam2_checkpoint_path_raw))
                if validation.reason is not None:
                    raise _PortalValidationReasonError(
                        _managed_sam2_reason_message(validation.reason),
                        reason=validation.reason,
                    )
                sam2_checkpoint_path = str(validation.normalized_path or "")
            else:
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
                "--segmentation-cache",
                segmentation_cache,
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
        if segmentation_backend == "sam2" and _as_bool(sam2_tiling_enabled, default=False):
            argv.append("--sam2-tiling-enabled")
        if segmentation_backend == "sam2" and sam2_tile_size_px is not None:
            argv.extend(["--sam2-tile-size-px", str(sam2_tile_size_px)])
        if segmentation_backend == "sam2" and sam2_overlap_px is not None:
            argv.extend(["--sam2-overlap-px", str(sam2_overlap_px)])
        if segmentation_backend == "sam2" and sam2_global_pass_longest_side is not None:
            argv.extend(["--sam2-global-pass-longest-side", str(sam2_global_pass_longest_side)])
        if segmentation_backend == "sam2" and sam2_max_concurrency is not None:
            argv.extend(["--sam2-max-concurrency", str(sam2_max_concurrency)])
        if segmentation_backend == "sam2" and sam2_points_per_side is not None:
            argv.extend(["--sam2-points-per-side", str(sam2_points_per_side)])
        if segmentation_backend == "sam2" and sam2_points_per_batch is not None:
            argv.extend(["--sam2-points-per-batch", str(sam2_points_per_batch)])
        if segmentation_backend == "sam2" and sam2_pred_iou_thresh is not None:
            argv.extend(["--sam2-pred-iou-thresh", str(sam2_pred_iou_thresh)])
        if segmentation_backend == "sam2" and sam2_stability_score_thresh is not None:
            argv.extend(["--sam2-stability-score-thresh", str(sam2_stability_score_thresh)])
        if segmentation_backend == "sam2" and sam2_crop_n_layers is not None:
            argv.extend(["--sam2-crop-n-layers", str(sam2_crop_n_layers)])
        if _as_bool(strict_segmentation, default=False):
            argv.append("--strict-segmentation")

        if depth_device:
            argv.extend(["--depth-device", str(depth_device)])
        if backend == "da3":
            argv.extend(["--model-key", model_key])

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
                "--run-card-version",
                str(
                    _pick(
                        args,
                        "run_card_version",
                        "runCardVersion",
                        default="v1",
                    )
                    or "v1"
                )
                .strip()
                .lower(),
                "--run-card-include-proofs",
                onoff(
                    _pick(
                        args,
                        "run_card_include_proofs",
                        "runCardIncludeProofs",
                        default=False,
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

        vlm_captioning_enabled = _as_bool(
            _pick(args, "vlm_captioning_enabled", "vlmCaptioningEnabled", default=False),
            default=False,
        )
        if vlm_captioning_enabled:
            vlm_captioning_backend = (
                str(
                    _pick(
                        args,
                        "vlm_captioning_backend",
                        "vlmCaptioningBackend",
                        default="fastvlm",
                    )
                    or "fastvlm"
                )
                .strip()
                .lower()
            )
            if vlm_captioning_backend not in ALLOWED_VLM_CAPTIONING_BACKENDS:
                raise _PortalValidationReasonError(
                    "Invalid vlm_captioning_backend",
                    reason="invalid_vlm_captioning_backend",
                )
            vlm_captioning_model = (
                str(
                    _pick(
                        args,
                        "vlm_captioning_model",
                        "vlmCaptioningModel",
                        default="default",
                    )
                    or "default"
                ).strip()
                or "default"
            )
            vlm_captioning_proxy_format = (
                str(
                    _pick(
                        args,
                        "vlm_captioning_proxy_format",
                        "vlmCaptioningProxyFormat",
                        default="png",
                    )
                    or "png"
                )
                .strip()
                .lower()
            )
            if vlm_captioning_proxy_format not in ALLOWED_VLM_CAPTIONING_PROXY_FORMATS:
                raise _PortalValidationReasonError(
                    "Invalid vlm_captioning_proxy_format",
                    reason="invalid_vlm_captioning_proxy_format",
                )
            vlm_captioning_max_side_px = (
                _parse_optional_positive_int(
                    _pick(
                        args,
                        "vlm_captioning_max_side_px",
                        "vlmCaptioningMaxSidePx",
                        default=1600,
                    ),
                    "vlm_captioning_max_side_px",
                )
                or 1600
            )
            fastvlm_timeout_seconds = (
                _parse_optional_positive_int(
                    _pick(
                        args,
                        "fastvlm_timeout_seconds",
                        "fastvlmTimeoutSeconds",
                        default=180,
                    ),
                    "fastvlm_timeout_seconds",
                )
                or 180
            )
            argv.extend(
                [
                    "--vlm-captioning",
                    "on",
                    "--vlm-captioning-backend",
                    vlm_captioning_backend,
                    "--vlm-captioning-model",
                    vlm_captioning_model,
                    "--vlm-captioning-proxy-format",
                    vlm_captioning_proxy_format,
                    "--vlm-captioning-max-side-px",
                    str(vlm_captioning_max_side_px),
                    "--fastvlm-timeout-seconds",
                    str(fastvlm_timeout_seconds),
                ]
            )
            fastvlm_python = str(
                _pick(
                    args,
                    "fastvlm_python_executable",
                    "fastvlmPythonExecutable",
                    default="",
                )
                or ""
            ).strip()
            if fastvlm_python:
                argv.extend(["--fastvlm-python", fastvlm_python])
            fastvlm_mlx_vlm_dir = str(
                _pick(
                    args,
                    "fastvlm_mlx_vlm_dir",
                    "fastvlmMlxVlmDir",
                    default="",
                )
                or ""
            ).strip()
            if fastvlm_mlx_vlm_dir:
                argv.extend(["--fastvlm-mlx-vlm-dir", fastvlm_mlx_vlm_dir])

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
            raise _PortalValidationReasonError(
                "verbose and quiet are mutually exclusive",
                reason="conflicting_log_verbosity_flags",
            )
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


@asynccontextmanager
async def _orchestrator_lifespan(app: "FastAPI") -> "AsyncGenerator[None, None]":
    if _job_api_key_enforced() and not API_KEY_SECRET:
        LOGGER.warning(
            "TP_ENFORCE_JOB_API_KEY is enabled but"
            " TP_API_KEY is unset; protected /v1 endpoints"
            " will return AUTH_CONFIGURATION_ERROR."
        )

    # Phase 1.C restart recovery: at process startup the runtime registry
    # is empty by construction, so any job the repository still records as
    # queued/running is necessarily orphaned by a previous process. Mark
    # them failed with worker_lost_on_restart so SSE late-clients see a
    # terminal done event instead of hanging on a dead state.
    #
    # Memory backend (default): the repository is per-process so there are
    # no orphans and the sweep is a deterministic no-op.
    # Postgres backend (TP_ORCHESTRATOR_STATE_BACKEND=postgres): the sweep
    # does the real work.
    #
    # Cache the repository on app.state so startup and shutdown share the
    # same instance: avoids double construction and prevents a confusing
    # "config error during shutdown" if TP_DATABASE_URL was misconfigured
    # at boot - the misconfig surfaces once at startup and shutdown is a
    # quiet no-op rather than a second loud failure.
    try:
        repo = get_job_repository()
    except Exception:  # noqa: BLE001 - never block startup on recovery
        LOGGER.exception("orchestrator repository construction failed; skipping restart recovery")
        repo = None
    app.state.job_repository = repo

    if repo is None:
        app.state.restart_recovery_swept = []
    else:
        try:
            swept = await sweep_orphaned_jobs(repo)
            app.state.restart_recovery_swept = list(swept)
        except Exception:  # noqa: BLE001 - never block startup on recovery
            LOGGER.exception("orchestrator restart recovery sweep failed; continuing startup")
            app.state.restart_recovery_swept = []

    existing_task = getattr(app.state, "cleanup_task", None)
    if existing_task is None or existing_task.done():
        app.state.cleanup_task = asyncio.create_task(_cleanup_loop())

    # Phase 2.C - in-process WorkerRunner pool. Started only when
    # ``USE_QUEUE_BROKER`` is enabled so existing single-process
    # deployments (and the pre-Phase-2.C test suite that monkey-patches
    # ``_run_job``) keep their in-band ``asyncio.create_task`` dispatch.
    # Each worker task consumes one job at a time; the pool size
    # mirrors ``MAX_CONCURRENT_JOBS`` so the broker matches the
    # orchestrator's admission concurrency cap.
    app.state.worker_stop_event = None
    app.state.worker_tasks = []
    app.state.queue_broker = None
    app.state.reclaim_sweep_task = None
    if USE_QUEUE_BROKER:
        try:
            broker = get_queue_broker()
        except Exception:  # noqa: BLE001 - never block startup on broker
            LOGGER.exception("queue broker unavailable; in-process worker pool disabled")
            broker = None
        if broker is not None:
            app.state.queue_broker = broker
            stop_event = asyncio.Event()
            app.state.worker_stop_event = stop_event
            for slot in range(MAX_CONCURRENT_JOBS):
                worker_config = WorkerConfig(
                    worker_id=f"inproc-worker-{slot}",
                    lease_seconds=WORKER_LEASE_SECONDS,
                    heartbeat_interval_seconds=WORKER_HEARTBEAT_INTERVAL_SECONDS,
                    poll_interval_seconds=WORKER_POLL_INTERVAL_SECONDS,
                )
                app.state.worker_tasks.append(
                    asyncio.create_task(
                        run_worker_forever(
                            broker=broker,
                            config=worker_config,
                            executor=_orchestrator_job_executor,
                            stop_event=stop_event,
                        )
                    )
                )
            # Phase 2.D — periodic broker-reclaim reconciler. Shares
            # ``stop_event`` with the worker pool so a single signal
            # winds down the whole broker-dispatch substrate.
            app.state.reclaim_sweep_task = asyncio.create_task(_reclaim_sweep_loop(broker, repo, stop_event))
    try:
        yield
    finally:
        worker_stop_event = getattr(app.state, "worker_stop_event", None)
        worker_tasks = getattr(app.state, "worker_tasks", []) or []
        reclaim_sweep_task = getattr(app.state, "reclaim_sweep_task", None)
        if worker_stop_event is not None:
            worker_stop_event.set()
        # The reclaim sweep observes the same stop_event the worker pool
        # uses; awaiting it here is bounded by the timeout below.
        if reclaim_sweep_task is not None:
            try:
                await asyncio.wait_for(reclaim_sweep_task, timeout=WORKER_SHUTDOWN_GRACE_SECONDS)
            except asyncio.TimeoutError:
                reclaim_sweep_task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await reclaim_sweep_task
            except Exception:  # noqa: BLE001 - sweep loop failure must not block shutdown
                LOGGER.exception("reclaim sweep task raised during shutdown")
            app.state.reclaim_sweep_task = None
        if worker_tasks:
            # ``run_worker_forever`` only checks ``stop_event`` between
            # jobs, so a long-running executor would otherwise block
            # shutdown indefinitely. Give the pool a bounded grace
            # period to wind down cleanly, then escalate to
            # ``task.cancel()`` so the lifespan can complete and the
            # broker / repository close paths still run. ``gather`` with
            # ``return_exceptions=True`` so one worker failure cannot
            # mask the others or prevent the subsequent cleanup.
            try:
                await asyncio.wait_for(
                    asyncio.gather(*worker_tasks, return_exceptions=True),
                    timeout=WORKER_SHUTDOWN_GRACE_SECONDS,
                )
            except asyncio.TimeoutError:
                LOGGER.warning(
                    "worker pool did not stop within %.1fs; cancelling %d task(s)",
                    WORKER_SHUTDOWN_GRACE_SECONDS,
                    sum(1 for t in worker_tasks if not t.done()),
                )
                for task in worker_tasks:
                    if not task.done():
                        task.cancel()
                results = await asyncio.gather(*worker_tasks, return_exceptions=True)
                for task, result in zip(worker_tasks, results):
                    if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                        LOGGER.exception("worker task %r raised during shutdown", task, exc_info=result)
            else:
                # Even within the grace period, a worker may have raised
                # before stop_event observed; surface it for ops.
                for task in worker_tasks:
                    if task.done() and not task.cancelled():
                        exc = task.exception()
                        if exc is not None:
                            LOGGER.exception("worker task %r raised during shutdown", task, exc_info=exc)
        app.state.worker_stop_event = None
        app.state.worker_tasks = []
        queue_broker = getattr(app.state, "queue_broker", None)
        if queue_broker is not None:
            try:
                await queue_broker.close()
            except Exception:  # noqa: BLE001 - never block shutdown on close
                LOGGER.exception("queue broker close failed")
        app.state.queue_broker = None
        cleanup_task = getattr(app.state, "cleanup_task", None)
        if cleanup_task is not None:
            cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await cleanup_task
            app.state.cleanup_task = None
        # Best-effort dispose of the repository's connection pool. The
        # memory backend's close() is a no-op; the Postgres backend
        # disposes its shared AsyncEngine. Reuse the repository instance
        # cached at startup so a startup-time construction failure isn't
        # re-emitted at shutdown.
        repo_for_close = getattr(app.state, "job_repository", None)
        if repo_for_close is not None:
            try:
                await repo_for_close.close()
            except Exception:  # noqa: BLE001 - never block shutdown on close
                LOGGER.exception("orchestrator repository close failed")
        app.state.job_repository = None


app = FastAPI(
    title="Transformation Portal Orchestrator",
    version="0.3.0",
    docs_url="/docs" if ENABLE_API_DOCS else None,
    redoc_url="/redoc" if ENABLE_API_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_API_DOCS else None,
    lifespan=_orchestrator_lifespan,
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
    if not _is_versioned_api_path(request.url.path):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=exc.headers,
        )

    path = request.url.path
    status_code = exc.status_code
    headers = exc.headers
    raw_detail = exc.detail
    message = _public_http_error_message(status_code, path)
    if isinstance(raw_detail, str) and raw_detail.strip():
        LOGGER.warning(
            "Sanitized HTTPException detail for %s %s (%s)",
            request.method,
            path,
            status_code,
        )
    del exc
    return _error_response(
        status_code,
        code=_http_status_error_code(status_code),
        message=message,
        details={"path": path},
        headers=headers,
    )


@app.exception_handler(RequestValidationError)
async def request_validation_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    if not _is_versioned_api_path(request.url.path):
        return await fastapi_request_validation_exception_handler(
            request,
            exc,
        )
    path = request.url.path
    del exc
    return _error_response(
        400,
        code="INVALID_ARGUMENT",
        message="request validation failed",
        details={
            "path": path,
            "reason": "request_validation_failed",
        },
    )


@app.middleware("http")
async def security_layer(
    request: Request,
    call_next: Callable[[Request], Any],
) -> Response:
    should_echo_traceparent = request.url.path == "/portal/bootstrap" or _is_versioned_api_path(request.url.path)
    if should_echo_traceparent:
        _portal_request_trace_context(request)

    maybe_error = _enforce_content_length_limit(request)
    if maybe_error is not None:
        if should_echo_traceparent:
            maybe_error.headers.setdefault("traceparent", request.state.trace_context.traceparent)
        return maybe_error

    _install_stream_body_limit(request)

    if _is_protected_api_key_endpoint(request.url.path) and _job_api_key_enforced():
        if not API_KEY_SECRET:
            response = _error_response(
                503,
                code="AUTH_CONFIGURATION_ERROR",
                message=("protected endpoint authentication is" " enforced but TP_API_KEY is not" " configured"),
                details={"path": request.url.path, "env": "TP_API_KEY"},
            )
            if should_echo_traceparent:
                response.headers.setdefault("traceparent", request.state.trace_context.traceparent)
            return response
        if not _has_valid_api_key(request):
            response = _error_response(
                401,
                code="UNAUTHORIZED",
                message="invalid or missing API key",
                details={"path": request.url.path},
            )
            if should_echo_traceparent:
                response.headers.setdefault("traceparent", request.state.trace_context.traceparent)
            return response

    client_ip = _extract_client_ip(request)
    rate_decision = _is_rate_limited(client_ip, _now())
    if rate_decision.limited:
        rate_headers = _rate_limit_response_headers(
            limit=RATE_LIMIT_PER_MINUTE,
            remaining=0,
            retry_after=rate_decision.retry_after_seconds,
            reset_epoch=rate_decision.reset_epoch_seconds,
        )
        response = _error_response(
            429,
            code="RATE_LIMITED",
            message="rate limit exceeded",
            details={"client_ip": client_ip},
            headers=rate_headers,
        )
        if should_echo_traceparent:
            response.headers.setdefault("traceparent", request.state.trace_context.traceparent)
        return response

    response = await call_next(request)
    for name, value in SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    if _is_versioned_api_path(request.url.path):
        response.headers.setdefault("Cache-Control", "no-store")
    if should_echo_traceparent:
        response.headers.setdefault("traceparent", request.state.trace_context.traceparent)
    return response


def _portal_html_response() -> Response:
    if not PORTAL_HTML.exists():
        raise HTTPException(
            status_code=500,
            detail="portal.html is missing",
        )
    bundle = _get_portal_asset_bundle()
    return Response(
        content=bundle.html_bytes,
        headers={
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
            "Content-Type": "text/html; charset=utf-8",
        },
    )


@app.get("/")
async def serve_ui(request: Request) -> Response:
    # Preserve the original query string so legacy/deep links such as
    # ``/?view=review`` continue to land on the correct workspace tab after
    # the redirect to the canonical ``/portal`` route.
    query = request.url.query
    target = "/portal"
    if query:
        target = f"{target}?{query}"
    return RedirectResponse(
        url=target,
        status_code=307,
        headers={"Cache-Control": "no-store"},
    )


@app.get("/portal")
async def serve_portal() -> Response:
    """Serves the single-file portal UI at its canonical route."""
    return _portal_html_response()


@app.get("/portal/assets/{asset_path:path}")
async def serve_portal_asset(asset_path: str, request: Request) -> Response:
    try:
        resolved_asset = _resolve_portal_asset(asset_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="portal asset not found") from exc

    requested_fingerprint = _requested_portal_asset_fingerprint(request)
    if asset_path == "portal.css":
        css_asset = _get_portal_css_asset()
        cache_control = _portal_asset_cache_control(css_asset.fingerprint, requested_fingerprint)
        etag = _portal_asset_etag(css_asset.fingerprint)
        if _portal_asset_request_etag_matches(request, etag):
            return _portal_asset_not_modified_response(etag=etag, cache_control=cache_control)
        return Response(
            content=css_asset.content_bytes,
            headers={
                "Cache-Control": cache_control,
                "Content-Type": resolved_asset.media_type,
                "ETag": etag,
            },
        )

    direct_fingerprint = _get_portal_direct_asset_fingerprint(asset_path)
    cache_control = _portal_asset_cache_control(direct_fingerprint, requested_fingerprint)
    etag = _portal_asset_etag(direct_fingerprint)
    if _portal_asset_request_etag_matches(request, etag):
        return _portal_asset_not_modified_response(etag=etag, cache_control=cache_control)
    return FileResponse(
        str(resolved_asset.path),
        media_type=resolved_asset.media_type,
        headers={
            "Cache-Control": cache_control,
            "ETag": etag,
        },
    )


@app.get(f"/portal/video/{PORTAL_VIDEO_ASSET_NAME}")
async def serve_portal_video() -> Response:
    if not PORTAL_VIDEO_PATH.is_file():
        return _error_response(
            404,
            code="NOT_FOUND",
            message="portal video asset not found",
        )

    return FileResponse(
        str(PORTAL_VIDEO_PATH),
        media_type=_artifact_content_type(PORTAL_VIDEO_PATH),
        headers={"Cache-Control": PORTAL_VIDEO_CACHE_CONTROL},
    )


@app.get(f"/v1/portal/video/{PORTAL_VIDEO_ASSET_NAME}")
async def redirect_legacy_portal_video() -> Response:
    return RedirectResponse(
        url=f"/portal/video/{PORTAL_VIDEO_ASSET_NAME}",
        status_code=307,
        headers={"Cache-Control": "no-store"},
    )


@app.get("/portal/bootstrap")
async def portal_bootstrap(request: Request) -> JSONResponse:
    """Expose standalone portal auth mode for direct backend debugging."""
    actor = _portal_actor_from_request(request)
    return JSONResponse(
        {
            "authMode": _auth_mode(),
            "csrfToken": None,
            "actor": None,
            "features": {
                "apiKeyInput": True,
                "directDebug": True,
                "artifactViewerModal": _portal_artifact_viewer_modal_enabled(None),
                "reviewSurfaceDeferred": _portal_review_surface_deferred_enabled(None),
                "stagedUploads": _portal_staged_uploads_enabled(None),
                "rumTelemetry": _portal_rum_enabled(None),
                "fastVlmCaptioning": _portal_fastvlm_captioning_enabled(actor),
            },
        },
        headers={"Cache-Control": "no-store"},
    )


@app.get("/healthz", response_model=HealthzResponse)
async def healthz() -> JSONResponse:
    """Lightweight health check endpoint for managed front door and load balancers.

    Returns a minimal status response without verbose details, suitable for
    Kubernetes probes, external health monitors, and managed authentication flows.
    This endpoint is referenced by the portal UI when in managed auth mode or
    before bootstrap is ready.
    """
    return JSONResponse(
        {"ok": True, "time": _now()},
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


@app.get(
    "/ready",
    response_model=ReadyResponse,
    # Preserve current wire shape: when TP_READY_VERBOSE=false, the handler
    # returns a dict WITHOUT the cli/jobs/security keys. Without this flag,
    # FastAPI would fill in those Optional fields as null and emit them,
    # which is a wire-format change external probes shouldn't see.
    response_model_exclude_none=True,
)
async def ready() -> Dict[str, Any]:
    response: Dict[str, Any] = {
        "ok": True,
        "time": _now(),
        "version": APP_VERSION,
    }
    if READY_VERBOSE:
        response["cli"] = {
            "lux-depth-v3": _lux_depth_runner_available(),
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


@app.get("/v1/readiness", response_model=ReadinessEnvelope)
async def readiness() -> JSONResponse:
    pipeline_data: Dict[str, Any] = {}
    for pipeline_name in ("lux-depth-v3", "archive-gate-a", "archive-gate-b", "archive-gate-c"):
        pipeline_data[pipeline_name] = _evaluate_pipeline_readiness(pipeline_name)

    return JSONResponse(
        _api_envelope(
            "tp.orchestrator.readiness.v1",
            success=True,
            data={
                "server": {
                    "time": _now(),
                    "version": APP_VERSION,
                    "auth_mode": _auth_mode(),
                    "backend_live": True,
                },
                "pipelines": pipeline_data,
            },
            error=None,
        )
    )


@app.get("/v1/presets", response_model=PresetsEnvelope)
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


@app.get("/v1/config-metadata", response_model=ConfigMetadataEnvelope)
async def config_metadata(pipeline: str) -> JSONResponse:
    pipeline_name = str(pipeline or "").strip()
    if pipeline_name != "lux-depth-v3":
        return _error_response(
            400,
            code="INVALID_ARGUMENT",
            message="unsupported config metadata pipeline",
            details={"field": "pipeline", "allowed": ["lux-depth-v3"]},
        )

    return JSONResponse(
        _api_envelope(
            "tp.orchestrator.config_metadata.v1",
            success=True,
            data=_lux_config_metadata(),
            error=None,
        )
    )


@app.post("/v1/config-preview", response_model=ConfigPreviewEnvelope)
async def config_preview(request: Request, payload: Dict[str, Any]) -> JSONResponse:
    try:
        preview = await _build_config_preview_threaded(
            payload,
            portal_actor=_portal_actor_from_request(request),
        )
    except ValueError:
        return _error_response(
            400,
            code="INVALID_ARGUMENT",
            message="invalid config preview request",
            details={"field": "payload", "reason": "unsupported_pipeline"},
        )

    return JSONResponse(
        _api_envelope(
            "tp.orchestrator.config_preview.v1",
            success=True,
            data=preview,
            error=None,
        )
    )


@app.post("/v1/portal/events", response_model=PortalEventEnvelope)
async def portal_events(payload: Dict[str, Any]) -> JSONResponse:
    record, reason = _record_portal_event(payload)
    if reason is not None:
        return _error_response(
            400,
            code="INVALID_ARGUMENT",
            message="invalid portal telemetry payload",
            details={"field": "payload", "reason": reason},
        )
    assert record is not None
    if PORTAL_EVENT_LOG_PATH is not None:
        await asyncio.to_thread(_persist_portal_event_record, record, PORTAL_EVENT_LOG_PATH)

    return JSONResponse(
        _api_envelope(
            "tp.orchestrator.portal_event.v1",
            success=True,
            data={"accepted": True, "event": record},
            error=None,
        )
    )


@app.post("/v1/portal/rum", response_model=PortalRumIngestEnvelope)
async def portal_rum(request: Request, payload: Dict[str, Any]) -> JSONResponse:
    if not _portal_rum_enabled(_portal_actor_from_request(request)):
        return JSONResponse(
            _api_envelope(
                "tp.orchestrator.portal_rum_ingest.v1",
                success=True,
                data={"accepted": False, "disabled": True},
                error=None,
            )
        )

    record, reason = _record_portal_rum(payload, request)
    if reason is not None:
        return _error_response(
            400,
            code="INVALID_ARGUMENT",
            message="invalid portal rum payload",
            details={"field": "payload", "reason": reason},
        )
    assert record is not None
    if PORTAL_RUM_LOG_PATH is not None:
        await asyncio.to_thread(_persist_portal_event_record, record, PORTAL_RUM_LOG_PATH)

    return JSONResponse(
        _api_envelope(
            "tp.orchestrator.portal_rum_ingest.v1",
            success=True,
            data={"accepted": True, "event": record},
            error=None,
        )
    )


@app.post("/v1/uploads/staging", response_model=UploadStagingEnvelope)
async def stage_portal_uploads(request: Request) -> JSONResponse:
    if not _portal_staged_uploads_enabled(_portal_actor_from_request(request)):
        return _error_response(
            404,
            code="NOT_FOUND",
            message="not found",
            details={"path": request.url.path},
        )

    try:
        upload_root = _resolved_portal_upload_root()
    except _PortalValidationReasonError:
        return _error_response(
            503,
            code="SERVICE_UNAVAILABLE",
            message="service unavailable",
            details={"path": request.url.path, "reason": "upload_root_invalid"},
        )

    parsed_payload: Optional[_ParsedPortalUploadPayload] = None
    try:
        parsed_payload = await _parse_portal_upload_multipart(request)
        if not parsed_payload.uploads:
            return _error_response(
                400,
                code="INVALID_ARGUMENT",
                message="at least one upload file is required",
                details={"field": "files", "reason": "files_required"},
            )

        client_manifest_paths = parse_client_manifest_relative_paths(
            parsed_payload.client_manifest_raw,
            expected_count=len(parsed_payload.uploads),
        )
        result = await asyncio.to_thread(
            stage_upload_batch,
            upload_root=upload_root,
            uploads=parsed_payload.uploads,
            client_manifest_paths=client_manifest_paths,
            capture_metadata_enabled=_env_bool("TP_PORTAL_UPLOAD_CAPTURE_METADATA_ENABLED", False),
            capture_metadata_config_path=DEFAULT_CAPTURE_METADATA_CONFIG_PATH,
            capture_metadata_schema_path=DEFAULT_CAPTURE_METADATA_SCHEMA_PATH,
            now=_now(),
        )
    except UploadStagingError as exc:
        return _error_response(
            exc.status_code,
            code=_http_status_error_code(exc.status_code),
            message=exc.message,
            details={"field": exc.field, "reason": exc.reason},
        )
    finally:
        if parsed_payload is not None:
            parsed_payload.close()

    return JSONResponse(
        _api_envelope(
            "tp.orchestrator.upload_staging.v1",
            success=True,
            data=result.to_response_data(),
            error=None,
        )
    )


async def create_job(
    payload: Dict[str, Any],
    *,
    portal_actor: Optional[Mapping[str, Any]] = None,
) -> JSONResponse:
    return await _create_job(
        payload,
        api_version="v1",
        portal_actor=portal_actor,
    )


async def create_job_v2(
    payload: Dict[str, Any],
    *,
    portal_actor: Optional[Mapping[str, Any]] = None,
) -> JSONResponse:
    return await _create_job(
        payload,
        api_version="v2",
        portal_actor=portal_actor,
    )


@app.post("/v1/jobs", response_model=JobEnvelope)
async def create_job_http(request: Request, payload: Dict[str, Any]) -> JSONResponse:
    return await create_job(payload, portal_actor=_portal_actor_from_request(request))


@app.post("/v2/jobs", response_model=JobEnvelope)
async def create_job_v2_http(request: Request, payload: Dict[str, Any]) -> JSONResponse:
    return await create_job_v2(payload, portal_actor=_portal_actor_from_request(request))


async def _create_job(
    payload: Dict[str, Any],
    *,
    api_version: str = "v1",
    portal_actor: Optional[Mapping[str, Any]] = None,
) -> JSONResponse:
    try:
        preview_kwargs: Dict[str, Any] = {"archive_index_scan_mode": "full"}
        if portal_actor is not None:
            preview_kwargs["portal_actor"] = portal_actor
        preview = await _build_config_preview_threaded(payload, **preview_kwargs)
    except ValueError:
        return _error_response(
            400,
            code="INVALID_ARGUMENT",
            message=_portal_safe_error_message("unsupported_pipeline"),
            details={"field": "payload", "reason": "unsupported_pipeline"},
        )

    pipeline = str(preview.get("pipeline") or payload.get("pipeline") or "").strip()

    preview_errors = preview.get("field_errors") or []
    if preview_errors:
        first_error = preview_errors[0] if isinstance(preview_errors[0], dict) else {}
        field = str(first_error.get("field") or "payload")
        reason = _portal_reason_code(first_error.get("code"))
        return _error_response(
            400,
            code="INVALID_ARGUMENT",
            message=_portal_issue_public_message(first_error, field=field),
            details={"field": field, "reason": reason},
        )

    readiness_snapshot = preview.get("readiness")
    if not isinstance(readiness_snapshot, dict):
        readiness_snapshot = _evaluate_pipeline_readiness(
            pipeline,
            preview.get("execution_args") if isinstance(preview.get("execution_args"), dict) else {},
            require_dispatch_inputs=True,
        )
    try:
        _enforce_job_readiness_preflight(pipeline, readiness_snapshot)
    except JobPreflightError as exc:
        status_code = int(exc.status_code)
        field = str(exc.field or "payload")
        reason = _portal_reason_code(exc.reason)
        del exc
        return _error_response(
            status_code,
            code="INVALID_ARGUMENT",
            message=_portal_safe_error_message(reason, field=field),
            details={"field": field, "reason": reason},
        )

    execution_args = preview.get("execution_args")
    if not isinstance(execution_args, dict):
        execution_args = {}

    try:
        _enforce_dispatch_value_preflight(pipeline, execution_args)
    except JobPreflightError as exc:
        status_code = int(exc.status_code)
        field = str(exc.field or "payload")
        reason = _portal_reason_code(exc.reason)
        del exc
        return _error_response(
            status_code,
            code="INVALID_ARGUMENT",
            message=_portal_safe_error_message(reason, field=field),
            details={"field": field, "reason": reason},
        )

    try:
        # Read-only filesystem preflight: validates paths and returns the
        # trusted output_dir to materialise *after* admission succeeds.
        trusted_output_dir = _enforce_dispatch_filesystem_preflight(pipeline, execution_args)
    except JobPreflightError as exc:
        status_code = int(exc.status_code)
        field = str(exc.field or "payload")
        reason = _portal_reason_code(exc.reason)
        del exc
        return _error_response(
            status_code,
            code="INVALID_ARGUMENT",
            message=_portal_safe_error_message(reason, field=field),
            details={"field": field, "reason": reason},
        )

    try:
        argv = _argv_from_request(
            payload,
            execution_args=execution_args,
        )
    except ValueError as exc:
        reason = _portal_reason_from_exception(exc)
        return _error_response(
            400,
            code="INVALID_ARGUMENT",
            message=_portal_safe_error_message(reason),
            details={"field": "payload", "reason": reason},
        )

    async with JOB_ADMISSION_LOCK:
        _cleanup_expired_jobs(_now())
        active_jobs = _active_job_count()
        if active_jobs >= MAX_CONCURRENT_JOBS:
            job_admission_headers = _rate_limit_response_headers(
                limit=MAX_CONCURRENT_JOBS,
                remaining=0,
                retry_after=JOB_ADMISSION_RETRY_AFTER_SECONDS,
                reset_epoch=int(_now()) + JOB_ADMISSION_RETRY_AFTER_SECONDS,
            )
            return _error_response(
                429,
                code="RATE_LIMITED",
                message="too many active jobs; try again later",
                details={
                    "active_jobs": active_jobs,
                    "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
                },
                headers=job_admission_headers,
            )
        # Materialise output_dir only after admission succeeds so 429-rejected
        # requests never leave behind directories on disk.
        try:
            _materialize_dispatch_output_dir(pipeline, trusted_output_dir)
        except JobPreflightError as exc:
            status_code = int(exc.status_code)
            field = str(exc.field or "payload")
            reason = _portal_reason_code(exc.reason)
            del exc
            return _error_response(
                status_code,
                code="INVALID_ARGUMENT",
                message=_portal_safe_error_message(reason, field=field),
                details={"field": field, "reason": reason},
            )
        jid = "job_" + uuid.uuid4().hex[:8]
        effective_request = {"pipeline": pipeline, "args": dict(execution_args)}
        job = Job(id=jid, created_at=_now(), request=payload, effective_request=effective_request)
        JOBS[jid] = job
        EVENT_SUBSCRIBERS[jid] = {}

    try:
        await _dispatch_job(job, argv, pipeline=pipeline)
    except _DispatchError:
        # Broker enqueue raised after we already admitted the job. The
        # enqueue may have partially succeeded; roll back the JOBS entry
        # and let the client retry. Detailed cause is logged inside
        # ``_dispatch_job``.
        JOBS.pop(jid, None)
        EVENT_SUBSCRIBERS.pop(jid, None)
        return _error_response(
            503,
            code="QUEUE_UNAVAILABLE",
            message="job admission queue is temporarily unavailable; please retry",
            details={"job_id": jid},
        )

    # Phase 1.D: construct a Pydantic envelope so the wire shape is
    # validated at runtime (Pydantic raises on bad field types / missing
    # required fields). Serialize through ``model_dump(by_alias=True,
    # exclude_unset=True)`` to preserve byte-identity with the legacy
    # ``_api_envelope`` helper (verified by
    # ``tests/orchestrator/test_envelope_runtime_validation.py``).
    return JSONResponse(
        JobEnvelope(
            schema="tp.orchestrator.job.v1",
            success=True,
            data=JobBriefData(
                id=jid,
                state=job.state,
                events_url=_job_events_url(jid, api_version=api_version),
            ),
            error=None,
        ).model_dump(mode="json", by_alias=True, exclude_unset=True)
    )


@app.get("/v1/jobs", response_model=JobsListEnvelope)
async def list_jobs(limit: int = JOB_LIST_LIMIT) -> JSONResponse:
    return _list_jobs(limit=limit, api_version="v1")


@app.get("/v2/jobs", response_model=JobsListEnvelope)
async def list_jobs_v2(limit: int = JOB_LIST_LIMIT) -> JSONResponse:
    return _list_jobs(limit=limit, api_version="v2")


def _list_jobs(*, limit: int = JOB_LIST_LIMIT, api_version: str = "v1") -> JSONResponse:
    _cleanup_expired_jobs(_now())
    bounded_limit = max(1, min(limit, JOB_LIST_LIMIT))
    jobs_sorted = sorted(
        JOBS.values(),
        key=lambda item: item.created_at,
        reverse=True,
    )
    serialized = [_serialize_job(job, include_logs=False, api_version=api_version) for job in jobs_sorted[:bounded_limit]]

    # Phase 1.D: construct via Pydantic for runtime validation, then
    # serialize with ``exclude_unset=True`` so optional Job fields the
    # legacy serializer omits when ``include_logs=False`` (notably
    # ``logs_tail``) stay absent on the wire.
    return JSONResponse(
        JobsListEnvelope(
            schema="tp.orchestrator.jobs.v1",
            success=True,
            data=JobsListData(
                jobs=[JobStatusData(**entry) for entry in serialized],
                total=len(JOBS),
                returned=len(serialized),
            ),
            error=None,
        ).model_dump(mode="json", by_alias=True, exclude_unset=True)
    )


@app.get("/v1/jobs/{job_id}", response_model=JobStatusEnvelope)
async def get_job(job_id: str, include_logs: bool = True) -> JSONResponse:
    return _get_job(job_id, include_logs=include_logs, api_version="v1")


@app.get("/v2/jobs/{job_id}", response_model=JobStatusEnvelope)
async def get_job_v2(job_id: str, include_logs: bool = True) -> JSONResponse:
    return _get_job(job_id, include_logs=include_logs, api_version="v2")


def _get_job(job_id: str, *, include_logs: bool = True, api_version: str = "v1") -> JSONResponse:
    _cleanup_expired_jobs(_now())
    job = JOBS.get(job_id)
    if not job:
        return _error_response(
            404,
            code="NOT_FOUND",
            message="job not found",
            details={"job_id": job_id},
        )
    # Phase 1.D: construct via Pydantic for runtime validation, then
    # serialize with ``exclude_unset=True`` so the optional
    # ``logs_tail`` default (None) is omitted when the legacy
    # ``_serialize_job`` does not include it (``include_logs=False``).
    return JSONResponse(
        JobStatusEnvelope(
            schema="tp.orchestrator.job_status.v1",
            success=True,
            data=JobStatusData(**_serialize_job(job, include_logs=bool(include_logs), api_version=api_version)),
            error=None,
        ).model_dump(mode="json", by_alias=True, exclude_unset=True)
    )


@app.get("/v1/jobs/{job_id}/artifacts/{artifact_path:path}")
async def get_job_artifact(job_id: str, artifact_path: str) -> Response:
    return await _get_job_artifact(job_id, artifact_path)


@app.get("/v2/jobs/{job_id}/artifacts/{artifact_path:path}")
async def get_job_artifact_v2(job_id: str, artifact_path: str) -> Response:
    return await _get_job_artifact(job_id, artifact_path)


async def _get_job_artifact(job_id: str, artifact_path: str) -> Response:
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
        requested_relative_path = _normalize_artifact_relative_path(artifact_path)
    except InvalidArtifactPathError:
        reason_code = "invalid_artifact_path"
    except AbsoluteArtifactPathError:
        reason_code = "absolute_artifact_path"
    except ArtifactPathOutsideJobOutputDirError:
        reason_code = "artifact_path_outside_job_output_dir"
    except ArtifactPathValidationError:
        reason_code = "invalid_artifact_path"
    else:
        reason_code = None

    if reason_code is not None:
        LOGGER.warning(
            "Rejected artifact path for job %s with reason %s",
            job_id,
            reason_code,
        )
        return _error_response(
            400,
            code="INVALID_ARGUMENT",
            message="invalid artifact path",
            details={"job_id": job_id, "reason": reason_code},
        )

    if not job.artifact_lookup:
        if not _hydrate_artifact_lookup_from_items(job):
            # Fingerprint computation can do bounded synchronous IO; offload
            # the indexing pass to a worker thread to keep the event loop
            # responsive for SSE subscribers and concurrent API callers.
            await asyncio.to_thread(_index_job_artifacts, job)
    resolved_artifact = job.artifact_lookup.get(requested_relative_path)
    if resolved_artifact is None:
        return _error_response(
            404,
            code="NOT_FOUND",
            message="artifact not found",
            details={"job_id": job_id, "path": requested_relative_path},
        )

    try:
        _, resolved_artifact, relative_path = _validate_resolved_job_artifact_path(job, resolved_artifact)
    except (ValueError, FileNotFoundError):
        return _error_response(
            404,
            code="NOT_FOUND",
            message="artifact not found",
            details={"job_id": job_id, "path": requested_relative_path},
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
        headers=_artifact_response_headers(resolved_artifact),
    )


@app.post("/v1/jobs/{job_id}/cancel", response_model=JobEnvelope)
async def cancel_job(job_id: str) -> JSONResponse:
    return await _cancel_job(job_id)


@app.post("/v2/jobs/{job_id}/cancel", response_model=JobEnvelope)
async def cancel_job_v2(job_id: str) -> JSONResponse:
    return await _cancel_job(job_id)


async def _cancel_job(job_id: str) -> JSONResponse:
    job = JOBS.get(job_id)
    if not job:
        return _error_response(
            404,
            code="NOT_FOUND",
            message="job not found",
            details={"job_id": job_id},
        )
    await _request_cancel(job)
    # Phase 1.D: ``events_url`` is deliberately not passed to
    # ``JobBriefData`` so it stays unset; ``exclude_unset=True`` then
    # drops it from the wire, matching the legacy ``_api_envelope``
    # output that omitted it for cancel responses.
    return JSONResponse(
        JobEnvelope(
            schema="tp.orchestrator.job.v1",
            success=True,
            data=JobBriefData(id=job_id, state=job.state),
            error=None,
        ).model_dump(mode="json", by_alias=True, exclude_unset=True)
    )


@app.get("/v1/jobs/{job_id}/events")
async def job_events(
    request: Request,
    job_id: str,
) -> Response:
    return await _job_events(request, job_id)


@app.get("/v2/jobs/{job_id}/events")
async def job_events_v2(
    request: Request,
    job_id: str,
) -> Response:
    return await _job_events(request, job_id)


async def _job_events(
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
    if job.state not in ACTIVE_JOB_STATES and job.state != "canceled":
        _refresh_job_run_summary(job)

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
                        "run_summary": job.run_summary or None,
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


# ---------------------------------------------------------------------------
# Phase 2.C - broker-mediated dispatch.
#
# When ``USE_QUEUE_BROKER`` is enabled the orchestrator no longer spawns a
# dispatch subprocess in-band from the HTTP handler. ``_create_job`` calls
# ``_dispatch_job`` which either:
#   - enqueues a ``JobEnqueueRequest`` onto the broker (a ``WorkerRunner``
#     pool started in the FastAPI lifespan consumes leases and calls
#     ``_orchestrator_job_executor``, which delegates to the existing
#     ``_run_job``); or
#   - falls back to the legacy ``asyncio.create_task(_run_job(...))`` path so
#     existing single-process deployments and the pre-Phase-2.C test suite
#     keep working unchanged.
# Phase 2.D will flip the default and remove the in-band path.
# ---------------------------------------------------------------------------


class _DispatchError(Exception):
    """Raised when ``_dispatch_job`` cannot safely hand the job off.

    ``_create_job`` catches this, rolls back the partial ``JOBS``
    entry, and emits a 503 so the caller can retry. Distinct from
    ``QueueBrokerError`` (a contract violation) and from broker
    construction failures (handled in-line with a fallback to the
    in-band path).
    """


async def _dispatch_job(job: "Job", argv: List[str], *, pipeline: str) -> None:
    """Hand the job off for execution.

    When ``USE_QUEUE_BROKER`` is true, enqueue via the broker so the
    in-process ``WorkerRunner`` pool acquires the lease and runs the
    dispatch subprocess. Otherwise preserve the legacy in-band
    ``asyncio.create_task(_run_job(...))`` semantics.

    ``_create_job`` retains its admission control (concurrency cap +
    rate limit). The broker is the execution seam, not the admission
    seam: every job ``_dispatch_job`` sees has already passed
    admission and is recorded in ``JOBS``.

    Failure modes:

    - Broker construction failure (boot-time misconfig) → fall back
      to the in-band path so a transient broker outage does not
      block admission. Logged and continued.
    - Broker rejects with ``QueueBrokerError`` (e.g. job_id
      collision) → in-band fallback is safe because the broker did
      not admit the job.
    - Broker ``enqueue`` raises any other exception (e.g. Redis
      connection drop after the write hit the wire): the enqueue may
      have partially succeeded, so falling back to in-band risks
      duplicate execution. Raise ``_DispatchError`` so ``_create_job``
      fail-closes and rolls back the ``JOBS`` entry.
    """
    if not USE_QUEUE_BROKER:
        asyncio.create_task(_run_job(job, argv))
        return

    try:
        broker = get_queue_broker()
    except Exception:  # noqa: BLE001 - broker construction is best-effort
        LOGGER.exception("queue broker unavailable for job %s; falling back to in-band dispatch", job.id)
        asyncio.create_task(_run_job(job, argv))
        return

    request = JobEnqueueRequest(
        job_id=job.id,
        argv=list(argv),
        api_version="v1",
        metadata={"pipeline": pipeline},
    )
    try:
        await broker.enqueue(request)
    except QueueBrokerError:
        LOGGER.exception("broker rejected enqueue for job %s; falling back to in-band dispatch", job.id)
        asyncio.create_task(_run_job(job, argv))
    except Exception as exc:  # noqa: BLE001 - broker IO failures must fail closed
        LOGGER.exception("broker enqueue raised for job %s; failing closed", job.id)
        raise _DispatchError(str(exc)) from exc


async def _orchestrator_job_executor(
    request: JobEnqueueRequest,
    cancellation_event: "asyncio.Event",
) -> int:
    """``WorkerRunner.executor`` adapter that runs the orchestrator's job.

    The broker holds the dispatch payload (argv); the orchestrator
    still owns the per-job runtime state (``JOBS[jid]``, the SSE
    subscriber queues, the live ``asyncio.subprocess.Process``
    handle, log/event accumulation). The executor bridges the two:
    look up the in-process ``Job`` and run the existing
    ``_run_job`` body, mirroring broker-side cancellation into the
    ``job.cancel_requested`` flag that ``_run_job``'s loop already
    observes.

    Phase 2.D — terminal-state guard. The broker's
    ``reclaim_expired_leases`` re-queues abandoned jobs by design
    (Phase 2.A contract). If the orchestrator-level reclaim sweep
    already drove the in-process ``Job`` to a terminal state
    (``worker_lost``, ``canceled``, etc.), a stale worker pickup
    of the same ``job_id`` must NOT re-spawn the dispatch
    subprocess. Treat the terminal Job as a no-op and release the
    lease cleanly.

    When the job has been cleaned up entirely between enqueue and
    lease pickup (expiry sweep removed the JOBS entry), the
    executor also exits cleanly.
    """
    job = JOBS.get(request.job_id)
    if job is None:
        LOGGER.warning("worker leased unknown job_id=%s; releasing without dispatch", request.job_id)
        return 0
    if job.finished_at is not None or job.state in _TERMINAL_JOB_STATES:
        LOGGER.info(
            "worker leased already-terminal job_id=%s state=%s; releasing without dispatch",
            request.job_id,
            job.state,
        )
        return 0

    async def _bridge_cancel() -> None:
        await cancellation_event.wait()
        # Bridge broker-side cancellation into the existing in-process
        # ``Job.cancel_requested`` flag so ``_run_job``'s readline loop
        # observes it and terminates the dispatch subprocess.
        if not job.cancel_requested:
            job.cancel_requested = True

    bridge_task = asyncio.create_task(_bridge_cancel())
    try:
        await _run_job(job, list(request.argv))
    finally:
        bridge_task.cancel()
        with suppress(asyncio.CancelledError):
            await bridge_task
    return int(job.exit_code) if job.exit_code is not None else 0


async def _publish_cancellation_terminal(job: "Job") -> None:
    """Publish the terminal cancelled-done event for a pre-lease cancel.

    When ``USE_QUEUE_BROKER`` is enabled and ``_request_cancel`` runs
    before the worker has leased the job, the broker drops the queue
    entry silently. No worker will ever pick the job up to publish
    terminal events, so the orchestrator does it here. For jobs the
    worker has already leased, ``_run_job``'s finally-block remains
    the source of truth and this helper is a no-op.
    """
    if job.finished_at is not None:
        return
    job.state = "canceled"
    job.exit_code = job.exit_code if job.exit_code is not None else 0
    now = _now()
    await _publish_event(
        job.id,
        "done",
        {
            "id": job.id,
            "state": job.state,
            "exit_code": job.exit_code,
            "error": job.error,
            "artifacts": job.artifacts,
            "run_summary": job.run_summary or None,
        },
    )
    job.done_published_at = now
    job.finished_at = now


async def _publish_worker_lost_terminal(job: "Job", *, reason_code: str) -> None:
    """Publish the terminal ``worker_lost`` event for a reclaimed lease.

    Phase 2.D: the broker-reclaim reconciler calls this when
    ``broker.reclaim_expired_leases`` surfaces a job whose worker
    fell behind on its heartbeat. The error envelope carries
    ``retriable=True`` so callers can distinguish a transient worker
    death from an executor-level failure.

    Idempotent: a job that already finished (e.g. the worker
    actually completed and called release after the reclaim window
    closed) collapses this to a no-op so the terminal event is not
    duplicated.
    """
    if job.finished_at is not None:
        return
    job.state = "worker_lost"
    if job.exit_code is None:
        job.exit_code = None  # explicitly preserve "no exit" semantics
    job.error = {
        "code": reason_code,
        "message": "Worker lease expired without heartbeat; job marked worker_lost.",
        "retriable": True,
    }
    now = _now()
    await _publish_event(
        job.id,
        "done",
        {
            "id": job.id,
            "state": job.state,
            "exit_code": job.exit_code,
            "error": job.error,
            "artifacts": job.artifacts,
            "run_summary": job.run_summary or None,
        },
    )
    job.done_published_at = now
    job.finished_at = now


async def _reclaim_sweep_once(broker: "QueueBroker", repository: Any) -> List[str]:
    """One pass of the Phase 2.D broker-reclaim reconciler.

    Calls ``broker.reclaim_expired_leases(now=server_time())`` and,
    for each reclaimed ``job_id``, drives BOTH:

    - the in-process ``Job`` (so SSE subscribers see the terminal
      ``done`` event immediately), and
    - the durable ``JobRepository`` row (so Postgres-backed state
      reflects the same ``worker_lost`` + ``retriable=True``
      payload — without this dual-write a restart could resurrect
      the row as ``running``).

    Returns the reclaimed ids so callers (lifespan loop, tests)
    can log / metric the sweep. The broker's contract still
    re-queues reclaimed dispatch payloads at the head of the ready
    FIFO; the orchestrator's executor adapter has an explicit
    terminal-state guard so a stale lease pickup of a reclaimed-
    then-terminal job is a no-op.

    ``repository`` may be ``None`` when broker construction
    succeeded but the repository singleton did not; in that case
    only the in-process Job is updated and a warning is logged so
    the operator can see the missing persistence.
    """
    try:
        now = await broker.server_time()
        reclaimed = await broker.reclaim_expired_leases(now=now)
    except Exception:  # noqa: BLE001 - sweep failures must not crash the loop
        LOGGER.exception("broker reclaim sweep failed")
        return []
    for jid in reclaimed:
        job = JOBS.get(jid)
        if job is not None:
            try:
                await _publish_worker_lost_terminal(job, reason_code=WORKER_LOST_REASON_RECLAIMED)
            except Exception:  # noqa: BLE001 - one job's failure must not block others
                LOGGER.exception("publish worker_lost terminal failed for job %s", jid)
        if repository is None:
            LOGGER.warning("reclaim sweep skipped repository update for job %s (no repository configured)", jid)
            continue
        try:
            now_ts = _now()
            await repository.update(
                jid,
                state="worker_lost",
                finished_at=now_ts,
                done_published_at=now_ts,
                last_event_at=now_ts,
                error={
                    "code": WORKER_LOST_REASON_RECLAIMED,
                    "message": "Worker lease expired without heartbeat; job marked worker_lost.",
                    "retriable": True,
                },
            )
        except Exception:  # noqa: BLE001 - persistence failure must not block the loop
            LOGGER.exception("repository update for reclaimed job %s failed", jid)
    return list(reclaimed)


async def _reclaim_sweep_loop(broker: "QueueBroker", repository: Any, stop_event: "asyncio.Event") -> None:
    """Periodic reconciler driven by ``_orchestrator_lifespan``.

    Sleeps ``RECLAIM_SWEEP_INTERVAL_SECONDS`` between sweeps; the
    ``stop_event`` short-circuits the wait so shutdown completes
    quickly. Exceptions inside the sweep are already swallowed by
    ``_reclaim_sweep_once`` (logged at ERROR), but the outer loop
    keeps an additional guard so a transient bug here cannot brick
    the lifespan task.
    """
    while not stop_event.is_set():
        try:
            await _reclaim_sweep_once(broker, repository)
        except Exception:  # noqa: BLE001 - defense in depth; never crash the loop
            LOGGER.exception("reclaim sweep loop iteration raised")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=RECLAIM_SWEEP_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            pass


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
        spawn_kwargs: Dict[str, Any] = {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.STDOUT,
            "env": _sanitized_child_env(),
        }
        if os.name != "nt":
            # Put the runner in its own session so cancel can signal the
            # whole process tree, not just the direct child.
            spawn_kwargs["start_new_session"] = True
        proc = await asyncio.create_subprocess_exec(*argv, **spawn_kwargs)
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
            line = _redact_log_line(line)
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
                # Phase 2.D — ``retriable=False`` distinguishes executor-level
                # failures (the work itself is broken) from the broker-level
                # ``worker_lost`` state (the worker died holding the lease).
                job.error = _error_obj(
                    "RUNNER_EXIT_NONZERO",
                    f"runner exited with code {rc}",
                    {"exit_code": int(rc)},
                    retriable=False,
                )

    except FileNotFoundError:
        job.state = "failed"
        job.exit_code = 127
        runner_repr = " ".join(argv[:3]) if len(argv) >= 3 else argv[0]
        job.error = _error_obj(
            "RUNNER_NOT_FOUND",
            f"Runner executable not found: '{argv[0]}'.",
            {"command": argv[0], "runner": runner_repr},
            retriable=False,
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
            retriable=False,
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
        # safely synthesize a 'done' from job state. Indexing also computes
        # bounded SHA-256 fingerprints, so run it in a worker thread to keep
        # the event loop responsive while large jobs are wrapping up.
        indexed_artifacts = await asyncio.to_thread(_index_job_artifacts, job)
        _refresh_job_run_summary(job)
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
                "run_summary": job.run_summary or None,
            },
        )
        # Mark timestamps AFTER all events are published, so SSE endpoint knows
        # it's safe to synthesize 'done' if done_published_at is set.
        job.done_published_at = _now()
        job.finished_at = job.done_published_at
        _cleanup_expired_jobs(_now())
