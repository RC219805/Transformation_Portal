#!/usr/bin/env python3
"""
Live backend smoke validation for Lux Materials V3 segmentation.

This script validates the real FastAPI portal backend over HTTP by spawning an
isolated local backend by default, submitting `lux-depth-v3` jobs through the
same `/v1/config-preview` and `/v1/jobs` surfaces used by the portal, and then
checking material segmentation evidence in artifacts, manifests, and run cards.

Required coverage:
1. EfficientSAM-backed Materials V3 live job succeeds.
2. Preview normalization preserves the Materials V3/segmentation contract.
3. Job status reaches `succeeded` with exit code 0.
4. SSE replay exposes state and done events.
5. Segmentation mask NPZ, combined manifest metadata, and run-card summaries
   all agree on an enabled real segmentation backend.

Optional SAM2 coverage:
    TP_PORTAL_LUX_RUN_SAM2=1 scripts/validation/validate_portal_lux_materials_live.py
    TP_PORTAL_LUX_RUN_SAM2=1 TP_PORTAL_LUX_REQUIRE_SAM2=1 ...

Run via:
    python scripts/validation/validate_portal_lux_materials_live.py
    make validate-portal-lux-materials-live
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Optional


DEFAULT_API_KEY = "contract-secret"
DEFAULT_ORCHESTRATOR_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_SAM2_CHECKPOINT = "./checkpoints/sam2_hiera_base_plus.pt"
TERMINAL_JOB_STATES = {"succeeded", "failed", "canceled", "partial"}


class SmokeFailure(RuntimeError):
    """Raised when the live Lux materials smoke validation fails."""

    def __init__(self, message: str, *, kind: str = "generic") -> None:
        super().__init__(message)
        self.kind = kind


@dataclass
class LocalRuntimeHandle:
    process: subprocess.Popen[str]
    base_url: str
    log_path: Path
    temp_paths: tuple[Path, ...] = ()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_fixture_image() -> Path:
    return _repo_root() / "tests" / "fixtures" / "apex_images" / "apex_test_interior.jpg"


def _default_temp_dir(prefix: str) -> Path:
    kwargs: Dict[str, Any] = {"prefix": prefix}
    if os.name != "nt" and Path("/tmp").exists():
        kwargs["dir"] = "/tmp"
    return Path(tempfile.mkdtemp(**kwargs))


def _resolve_output_dir(raw_output_dir: str) -> tuple[Path, bool]:
    candidate = str(raw_output_dir).strip()
    if candidate:
        return Path(candidate).resolve(), False
    return _default_temp_dir("tp-portal-lux-materials-output-"), True


def _should_cleanup_output_dir(*, keep_output: bool, output_dir_is_temp: bool) -> bool:
    return output_dir_is_temp and not keep_output


def _prepare_input_dir(raw_input_dir: str, fixture_image: Path) -> tuple[Path, bool]:
    candidate = str(raw_input_dir).strip()
    if candidate:
        return Path(candidate).resolve(), False

    if not fixture_image.is_file():
        raise SmokeFailure(f"Fixture image does not exist: {fixture_image}", kind="environment")

    input_dir = _default_temp_dir("tp-portal-lux-materials-input-")
    shutil.copy2(fixture_image, input_dir / fixture_image.name)
    return input_dir, True


def _base_url(value: str) -> str:
    trimmed = str(value or "").strip()
    if not trimmed:
        raise SmokeFailure("Base URL cannot be empty", kind="environment")
    return trimmed.rstrip("/")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tail_text(path: Path, *, max_chars: int = 1600, max_bytes: int = 8192) -> str:
    if max_chars <= 0 or max_bytes <= 0:
        return ""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            byte_count = handle.tell()
            if byte_count <= 0:
                return ""
            window = min(byte_count, max_bytes)
            handle.seek(-window, os.SEEK_END)
            content = handle.read(window).decode("utf-8", errors="replace")
    except OSError:
        return ""
    content = content.strip()
    if len(content) <= max_chars:
        return content
    return content[-max_chars:]


def _terminate_runtime(handle: LocalRuntimeHandle) -> None:
    process = handle.process
    if process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=5)
        except Exception:
            try:
                process.kill()
                process.wait(timeout=5)
            except Exception:
                pass

    for temp_path in handle.temp_paths:
        if temp_path.is_dir():
            shutil.rmtree(temp_path, ignore_errors=True)
        else:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    api_key: str = "",
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0,
) -> tuple[int, Dict[str, Any]]:
    data = None
    headers = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        _base_url(base_url) + path,
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            raw_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw_body = exc.read().decode("utf-8")
    except (TimeoutError, urllib.error.URLError) as exc:
        reason = getattr(exc, "reason", exc)
        raise SmokeFailure(f"{method} {path} request failed: {reason}", kind="transport") from exc

    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise SmokeFailure(f"{method} {path} returned non-JSON response: {raw_body[:400]!r}", kind="contract") from exc
    return status, body


def _request_text(
    base_url: str,
    path: str,
    *,
    api_key: str = "",
    timeout: float = 30.0,
) -> str:
    headers = {"Accept": "text/event-stream"}
    if api_key:
        headers["x-api-key"] = api_key
    request = urllib.request.Request(
        _base_url(base_url) + path,
        headers=headers,
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise SmokeFailure(f"GET {path} SSE request failed: {reason}", kind="transport") from exc


def _wait_for_backend_ready(
    base_url: str,
    *,
    timeout_seconds: float,
    process: Optional[subprocess.Popen[str]] = None,
    log_path: Optional[Path] = None,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            break
        try:
            status, body = _request_json(base_url, "/ready", timeout=5.0)
            if status == 200 and body.get("ok") is True:
                return
            last_error = f"status={status} body={body}"
        except SmokeFailure as exc:
            last_error = str(exc)
        time.sleep(0.25)

    if process is not None and process.poll() is not None:
        log_tail = _tail_text(log_path) if log_path is not None else ""
        detail = f"local backend exited before readiness (code {process.returncode})"
        if log_tail:
            detail = f"{detail}. Recent log output:\n{log_tail}"
        raise SmokeFailure(detail, kind="environment")

    detail = last_error or "timed out waiting for /ready"
    raise SmokeFailure(
        f"Local backend did not become ready at {base_url}/ready within {timeout_seconds:.1f}s ({detail}).",
        kind="environment",
    )


def _spawn_local_backend(api_key: str, *, timeout_seconds: float) -> LocalRuntimeHandle:
    runtime_root = _default_temp_dir("tp-portal-lux-materials-backend-")
    log_path = runtime_root / "uvicorn.log"
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TP_RATE_LIMIT_PER_MINUTE"] = "0"
    env["TP_API_KEY"] = api_key

    log_handle = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=str(_repo_root()),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        log_handle.close()

    handle = LocalRuntimeHandle(process=process, base_url=base_url, log_path=log_path, temp_paths=(runtime_root,))
    try:
        _wait_for_backend_ready(
            base_url,
            timeout_seconds=timeout_seconds,
            process=process,
            log_path=log_path,
        )
    except Exception:
        _terminate_runtime(handle)
        raise
    return handle


def _expect_status(status: int, expected: int, context: str, body: Dict[str, Any]) -> None:
    if status != expected:
        raise SmokeFailure(f"{context} returned HTTP {status}, expected {expected}: {json.dumps(body, sort_keys=True)}")


def _build_lux_materials_payload(
    *,
    input_dir: Path,
    output_dir: Path,
    segmentation_backend: str = "efficientsam",
    sam2_checkpoint_path: Optional[Path] = None,
    sam2_model_size: str = "base",
) -> Dict[str, Any]:
    backend = str(segmentation_backend or "efficientsam").strip().lower()
    args: Dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "quality_tier": "apex",
        "depth_backend": "da3",
        "depth_device": "cpu",
        "materials_v3": True,
        "enable_segmentation": True,
        "segmentation_backend": backend,
        "strict_segmentation": True,
        "pbr": True,
        "emit_run_card": True,
        "run_card_version": "v2",
        "enable_v2": False,
        "non_commercial_ok": True,
        "emit_master16": True,
        "emit_upscaled16": False,
        "cache_depth": False,
        "save_float_depth": False,
    }
    if backend == "sam2":
        args.update(
            {
                "sam2_model_size": sam2_model_size,
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
            }
        )
        if sam2_checkpoint_path is not None:
            args["sam2_checkpoint_path"] = str(sam2_checkpoint_path)

    return {"pipeline": "lux-depth-v3", "args": args}


def _argv_preview_tokens(preview: Dict[str, Any]) -> list[str]:
    argv_preview = str(preview.get("argv_preview") or "").strip()
    if not argv_preview:
        return []
    try:
        return shlex.split(argv_preview)
    except ValueError as exc:
        raise SmokeFailure(f"Preview argv could not be parsed: {argv_preview!r}", kind="contract") from exc


def _tokens_contain_pair(tokens: list[str], flag: str, value: str) -> bool:
    return any(token == flag and index + 1 < len(tokens) and tokens[index + 1] == value for index, token in enumerate(tokens))


def _validate_lux_preview(preview: Dict[str, Any], *, expected_backend: str) -> None:
    errors = preview.get("field_errors") or []
    if errors:
        raise SmokeFailure(f"Preview returned field errors: {json.dumps(errors, sort_keys=True)}", kind="contract")

    for surface_name in ("normalized_args", "execution_args"):
        surface = preview.get(surface_name)
        if not isinstance(surface, dict):
            raise SmokeFailure(f"Preview missing {surface_name} object: {preview}", kind="contract")
        expected_values = {
            "materials_v3": True,
            "enable_segmentation": True,
            "segmentation_backend": expected_backend,
            "strict_segmentation": True,
            "pbr": True,
            "emit_run_card": True,
            "run_card_version": "v2",
            "enable_v2": False,
            "non_commercial_ok": True,
        }
        for key, expected in expected_values.items():
            if surface.get(key) != expected:
                raise SmokeFailure(
                    f"Preview {surface_name}.{key}={surface.get(key)!r}, expected {expected!r}",
                    kind="contract",
                )
        for deprecated_key in ("emit_marketing", "emitMarketing", "emit_report", "emitReport"):
            if deprecated_key in surface:
                raise SmokeFailure(
                    f"Preview {surface_name} retained deprecated key {deprecated_key}",
                    kind="contract",
                )

    tokens = _argv_preview_tokens(preview)
    required_pairs = {
        "--materials-v3": "on",
        "--enable-segmentation": "on",
        "--segmentation-backend": expected_backend,
        "--non-commercial-ok": "true",
    }
    for flag, value in required_pairs.items():
        if not _tokens_contain_pair(tokens, flag, value):
            raise SmokeFailure(f"Preview argv missing {flag} {value}: {tokens}", kind="contract")
    if "--strict-segmentation" not in tokens:
        raise SmokeFailure(f"Preview argv missing --strict-segmentation: {tokens}", kind="contract")
    for deprecated_flag in ("--emit-marketing", "--emit-report"):
        if deprecated_flag in tokens:
            raise SmokeFailure(f"Preview argv retained deprecated flag {deprecated_flag}: {tokens}", kind="contract")


def _preview_job(
    base_url: str,
    *,
    api_key: str,
    payload: Dict[str, Any],
    expected_backend: str,
) -> Dict[str, Any]:
    status, body = _request_json(
        base_url,
        "/v1/config-preview",
        method="POST",
        api_key=api_key,
        payload=payload,
    )
    _expect_status(status, 200, "POST /v1/config-preview", body)
    if body.get("success") is not True:
        raise SmokeFailure(f"Preview did not report success: {body}", kind="contract")
    data = body.get("data")
    if not isinstance(data, dict):
        raise SmokeFailure(f"Preview returned invalid data envelope: {body}", kind="contract")
    _validate_lux_preview(data, expected_backend=expected_backend)
    return data


def _submit_job(base_url: str, *, api_key: str, payload: Dict[str, Any]) -> str:
    status, body = _request_json(
        base_url,
        "/v1/jobs",
        method="POST",
        api_key=api_key,
        payload=payload,
    )
    _expect_status(status, 200, "POST /v1/jobs", body)
    if body.get("success") is not True:
        raise SmokeFailure(f"Job creation did not report success: {body}", kind="contract")
    job_id = str(((body.get("data") or {}).get("id") or "")).strip()
    if not job_id.startswith("job_"):
        raise SmokeFailure(f"Job creation returned invalid job id: {body}", kind="contract")
    return job_id


def _poll_terminal_job(
    base_url: str,
    *,
    api_key: str,
    job_id: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status, body = _request_json(base_url, f"/v1/jobs/{job_id}", api_key=api_key)
        _expect_status(status, 200, f"GET /v1/jobs/{job_id}", body)
        data = body.get("data") or {}
        state = data.get("state")
        if state in TERMINAL_JOB_STATES:
            return body
        time.sleep(poll_interval_seconds)
    raise SmokeFailure(f"Job {job_id} did not reach a terminal state within {timeout_seconds:.1f}s", kind="runtime")


def _classify_terminal_job_failure(terminal_body: Dict[str, Any]) -> str:
    data = terminal_body.get("data") if isinstance(terminal_body, dict) else {}
    if not isinstance(data, dict):
        return "contract"
    text_parts = [
        json.dumps(data.get("error") or {}, sort_keys=True),
        "\n".join(str(line) for line in (data.get("logs_tail") or [])),
    ]
    haystack = "\n".join(text_parts).lower()
    environment_markers = (
        "modulenotfounderror",
        "importerror",
        "no module named",
        "pytorch not available",
        "torchvision not available",
        "failed to load efficientsam backend",
        "efficientsam backend loading failed",
        "sam2 backend unavailable",
        "sam2 backend loading failed",
        "checkpoint",
        "model download",
        "runtime unavailable",
        "runner executable not found",
        "runner_not_found",
    )
    if any(marker in haystack for marker in environment_markers):
        return "environment"
    return "product"


def _ensure_job_succeeded(terminal_body: Dict[str, Any], *, backend: str) -> Dict[str, Any]:
    data = terminal_body.get("data")
    if not isinstance(data, dict):
        raise SmokeFailure(f"Job status returned invalid envelope: {terminal_body}", kind="contract")
    if data.get("state") != "succeeded" or data.get("exit_code") != 0:
        kind = _classify_terminal_job_failure(terminal_body)
        raise SmokeFailure(
            f"{backend} Lux materials job did not succeed: state={data.get('state')!r} exit_code={data.get('exit_code')!r} "
            f"error={data.get('error')!r} logs_tail={data.get('logs_tail')!r}",
            kind=kind,
        )
    return data


def _validate_sse_replay(base_url: str, *, api_key: str, job_id: str) -> None:
    events_text = _request_text(base_url, f"/v1/jobs/{job_id}/events", api_key=api_key)
    if "event: state" not in events_text:
        raise SmokeFailure(f"SSE replay missing state event for {job_id}: {events_text[:400]!r}", kind="contract")
    if "event: done" not in events_text:
        raise SmokeFailure(f"SSE replay missing done event for {job_id}: {events_text[:400]!r}", kind="contract")


def _artifact_items(job_data: Dict[str, Any]) -> list[Dict[str, Any]]:
    artifacts = job_data.get("artifacts")
    if not isinstance(artifacts, dict):
        raise SmokeFailure(f"Job data missing artifacts object: {job_data}", kind="contract")
    items = artifacts.get("items")
    if not isinstance(items, list):
        raise SmokeFailure(f"Job artifacts missing items list: {job_data}", kind="contract")
    return [item for item in items if isinstance(item, dict)]


def _output_dir_from_job(job_data: Dict[str, Any]) -> Path:
    artifacts = job_data.get("artifacts")
    output_dir = artifacts.get("output_dir") if isinstance(artifacts, dict) else None
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise SmokeFailure(f"Job artifacts missing output_dir: {job_data}", kind="contract")
    resolved = Path(output_dir)
    if not resolved.is_dir():
        raise SmokeFailure(f"Job output_dir is not a directory: {resolved}", kind="product")
    return resolved


def _find_relative_path(items: Iterable[Dict[str, Any]], *, prefix: str, suffix: str) -> Optional[str]:
    for item in items:
        relative_path = str(item.get("relative_path") or item.get("path") or "")
        if relative_path.startswith(prefix) and relative_path.endswith(suffix):
            return relative_path
    return None


def _safe_output_path(output_dir: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or any(part == ".." for part in pure.parts):
        raise SmokeFailure(f"Invalid artifact relative path: {relative_path}", kind="contract")
    return output_dir / Path(*pure.parts)


def _load_json_file(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SmokeFailure(f"Unable to load JSON artifact {path}: {exc}", kind="product") from exc
    if not isinstance(payload, dict):
        raise SmokeFailure(f"JSON artifact is not an object: {path}", kind="product")
    return payload


def _validate_mask_npz(mask_path: Path) -> Dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise SmokeFailure("numpy is required to validate segmentation mask NPZ artifacts", kind="environment") from exc

    if not mask_path.is_file():
        raise SmokeFailure(f"Segmentation mask artifact is missing: {mask_path}", kind="product")
    try:
        with np.load(mask_path) as data:
            names = list(data.files)
            non_empty = []
            for name in names:
                array = data[name]
                if array.size and int(np.count_nonzero(array)) > 0:
                    non_empty.append(name)
    except Exception as exc:
        raise SmokeFailure(f"Unable to read segmentation mask artifact {mask_path}: {exc}", kind="product") from exc

    if not names:
        raise SmokeFailure(f"Segmentation mask artifact contains no arrays: {mask_path}", kind="product")
    if not non_empty:
        raise SmokeFailure(f"Segmentation mask artifact contains no non-empty masks: {mask_path}", kind="product")
    return {"mask_count": len(names), "non_empty_mask_count": len(non_empty), "non_empty_masks": non_empty}


def _validate_combined_manifest(
    manifest: Dict[str, Any],
    *,
    expected_backend: str,
    mask_relative_path: str,
) -> Dict[str, Any]:
    materials = manifest.get("materials_v3")
    if not isinstance(materials, dict):
        raise SmokeFailure("Combined manifest missing materials_v3 section", kind="product")
    if materials.get("version") != "3.1":
        raise SmokeFailure(f"Combined manifest materials_v3.version={materials.get('version')!r}", kind="product")
    segmentation = materials.get("segmentation_metadata")
    if not isinstance(segmentation, dict):
        raise SmokeFailure("Combined manifest missing materials_v3.segmentation_metadata", kind="product")
    if segmentation.get("backend") != expected_backend:
        raise SmokeFailure(
            f"Combined manifest segmentation backend={segmentation.get('backend')!r}, expected {expected_backend!r}",
            kind="product",
        )
    if int(segmentation.get("mask_count") or 0) <= 0:
        raise SmokeFailure(f"Combined manifest mask_count is not positive: {segmentation}", kind="product")
    mask_artifact_path = str(segmentation.get("mask_artifact_path") or "")
    if not mask_artifact_path:
        raise SmokeFailure(f"Combined manifest missing mask_artifact_path: {segmentation}", kind="product")
    if not mask_artifact_path.replace("\\", "/").endswith(mask_relative_path):
        raise SmokeFailure(
            f"Combined manifest mask_artifact_path={mask_artifact_path!r} does not match {mask_relative_path!r}",
            kind="product",
        )
    return segmentation


def _validate_run_card(
    run_card: Dict[str, Any],
    *,
    expected_backend: str,
    mask_relative_path: str,
) -> None:
    rows = run_card.get("result_summary")
    if not isinstance(rows, list) or not rows:
        raise SmokeFailure("Run card missing non-empty result_summary", kind="product")

    statuses = [row.get("segmentation_status") for row in rows if isinstance(row, dict)]
    status = next((item for item in statuses if isinstance(item, dict) and item.get("enabled") is True), None)
    if not isinstance(status, dict):
        raise SmokeFailure(f"Run card missing enabled segmentation_status: {statuses}", kind="product")
    if status.get("backend") != expected_backend:
        raise SmokeFailure(
            f"Run card segmentation backend={status.get('backend')!r}, expected {expected_backend!r}",
            kind="product",
        )
    if status.get("errors"):
        raise SmokeFailure(f"Run card segmentation_status has errors: {status}", kind="product")
    if int(status.get("mask_count") or 0) <= 0:
        raise SmokeFailure(f"Run card segmentation_status mask_count is not positive: {status}", kind="product")
    if str(status.get("mask_artifact_path") or "") != mask_relative_path:
        raise SmokeFailure(
            f"Run card mask_artifact_path={status.get('mask_artifact_path')!r}, expected {mask_relative_path!r}",
            kind="product",
        )

    artifact_index = run_card.get("artifact_index")
    if not isinstance(artifact_index, list):
        raise SmokeFailure("Run card missing artifact_index list", kind="product")
    matching_artifact = next(
        (
            item
            for item in artifact_index
            if isinstance(item, dict)
            and item.get("relative_path") == mask_relative_path
            and item.get("artifact_type") == "segmentation_mask_npz"
        ),
        None,
    )
    if matching_artifact is None:
        raise SmokeFailure(
            f"Run card artifact_index missing segmentation_mask_npz entry for {mask_relative_path}",
            kind="product",
        )


def _latest_path(paths: Iterable[Path]) -> Optional[Path]:
    existing = [path for path in paths if path.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def _validate_lux_outputs(job_data: Dict[str, Any], *, expected_backend: str) -> Dict[str, Any]:
    items = _artifact_items(job_data)
    output_dir = _output_dir_from_job(job_data)
    mask_relative_path = _find_relative_path(
        items,
        prefix="segmentation/",
        suffix="_materials_v3_masks.npz",
    )
    if not mask_relative_path:
        raise SmokeFailure(f"Job artifacts missing segmentation mask NPZ: {items}", kind="product")

    mask_path = _safe_output_path(output_dir, mask_relative_path)
    mask_stats = _validate_mask_npz(mask_path)

    manifest_relative_path = _find_relative_path(items, prefix="manifests/", suffix="_combined.json")
    if not manifest_relative_path:
        raise SmokeFailure(f"Job artifacts missing combined manifest: {items}", kind="product")
    manifest = _load_json_file(_safe_output_path(output_dir, manifest_relative_path))
    segmentation_metadata = _validate_combined_manifest(
        manifest,
        expected_backend=expected_backend,
        mask_relative_path=mask_relative_path,
    )

    run_card_path = _latest_path(path for path in output_dir.glob("run_card_*.json") if not path.name.endswith(".self.json"))
    if run_card_path is None:
        raise SmokeFailure(f"Output directory missing run_card_*.json: {output_dir}", kind="product")
    run_card = _load_json_file(run_card_path)
    _validate_run_card(
        run_card,
        expected_backend=expected_backend,
        mask_relative_path=mask_relative_path,
    )

    return {
        "output_dir": str(output_dir),
        "mask_relative_path": mask_relative_path,
        "manifest_relative_path": manifest_relative_path,
        "run_card_path": str(run_card_path),
        "mask_stats": mask_stats,
        "segmentation_metadata": segmentation_metadata,
    }


def _run_backend_validation(
    base_url: str,
    *,
    api_key: str,
    input_dir: Path,
    output_dir: Path,
    backend: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
    sam2_checkpoint_path: Optional[Path] = None,
    sam2_model_size: str = "base",
) -> Dict[str, Any]:
    payload = _build_lux_materials_payload(
        input_dir=input_dir,
        output_dir=output_dir,
        segmentation_backend=backend,
        sam2_checkpoint_path=sam2_checkpoint_path,
        sam2_model_size=sam2_model_size,
    )
    _preview_job(base_url, api_key=api_key, payload=payload, expected_backend=backend)
    job_id = _submit_job(base_url, api_key=api_key, payload=payload)
    terminal = _poll_terminal_job(
        base_url,
        api_key=api_key,
        job_id=job_id,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    job_data = _ensure_job_succeeded(terminal, backend=backend)
    _validate_sse_replay(base_url, api_key=api_key, job_id=job_id)
    output_evidence = _validate_lux_outputs(job_data, expected_backend=backend)
    return {"job_id": job_id, **output_evidence}


def _sam2_prerequisite_failure(checkpoint_path: Path) -> Optional[str]:
    if not checkpoint_path.is_file():
        return f"checkpoint_missing:{checkpoint_path}"
    try:
        import transformation_portal.spatial_ai.segmentation.sam2_backend  # noqa: F401
    except Exception as exc:
        return f"sam2_runtime_unavailable:{type(exc).__name__}:{exc}"
    return None


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("TP_ORCHESTRATOR_BASE_URL", DEFAULT_ORCHESTRATOR_BASE_URL),
        help="Backend base URL when --no-spawn-local-backend is used (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("TP_API_KEY", DEFAULT_API_KEY),
        help=f"API key for protected job endpoints (default: TP_API_KEY or {DEFAULT_API_KEY})",
    )
    parser.add_argument(
        "--no-spawn-local-backend",
        dest="spawn_local_backend",
        action="store_false",
        default=True,
        help="Use --base-url instead of launching an isolated local backend",
    )
    parser.add_argument(
        "--backend-startup-timeout-seconds",
        type=float,
        default=45.0,
        help="Max time to wait for spawned backend readiness (default: %(default)s)",
    )
    parser.add_argument(
        "--input-dir",
        default="",
        help="Optional existing input directory. Defaults to a temp dir populated from --fixture-image.",
    )
    parser.add_argument(
        "--fixture-image",
        default=str(_default_fixture_image()),
        help="Fixture image copied into the temp input dir when --input-dir is omitted",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory for the smoke job (defaults to a temp dir)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180.0,
        help="Max time to wait for each terminal Lux job state (default: %(default)s)",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=2.0,
        help="Polling interval for job status checks (default: %(default)s)",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Preserve temp input/output directories instead of deleting them",
    )
    parser.add_argument(
        "--run-sam2",
        action="store_true",
        default=_env_bool("TP_PORTAL_LUX_RUN_SAM2", False),
        help="Also run the optional SAM2 live materials validation",
    )
    parser.add_argument(
        "--require-sam2",
        action="store_true",
        default=_env_bool("TP_PORTAL_LUX_REQUIRE_SAM2", False),
        help="Fail when optional SAM2 prerequisites or execution are unavailable",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        default=os.getenv("TP_PORTAL_LUX_SAM2_CHECKPOINT", DEFAULT_SAM2_CHECKPOINT),
        help="SAM2 checkpoint path for optional validation (default: %(default)s)",
    )
    parser.add_argument(
        "--sam2-model-size",
        default=os.getenv("TP_PORTAL_LUX_SAM2_MODEL_SIZE", "base"),
        choices=("base", "large"),
        help="SAM2 model size for optional validation (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    backend_runtime: Optional[LocalRuntimeHandle] = None
    input_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    input_dir_is_temp = False
    output_dir_is_temp = False

    try:
        fixture_image = Path(args.fixture_image).resolve()
        input_dir, input_dir_is_temp = _prepare_input_dir(str(args.input_dir), fixture_image)
        output_dir, output_dir_is_temp = _resolve_output_dir(str(args.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.spawn_local_backend:
            backend_runtime = _spawn_local_backend(
                str(args.api_key),
                timeout_seconds=float(args.backend_startup_timeout_seconds),
            )
            base_url = backend_runtime.base_url
        else:
            base_url = _base_url(args.base_url)

        ready_status, ready_body = _request_json(base_url, "/ready")
        _expect_status(ready_status, 200, "GET /ready", ready_body)
        if ready_body.get("ok") is not True:
            raise SmokeFailure(f"/ready did not report ok=true: {ready_body}", kind="environment")

        efficient_output_dir = output_dir / "efficientsam"
        efficient_output_dir.mkdir(parents=True, exist_ok=True)
        efficient_result = _run_backend_validation(
            base_url,
            api_key=str(args.api_key),
            input_dir=input_dir,
            output_dir=efficient_output_dir,
            backend="efficientsam",
            timeout_seconds=float(args.timeout_seconds),
            poll_interval_seconds=float(args.poll_interval_seconds),
        )

        sam2_result: Optional[Dict[str, Any]] = None
        sam2_skip_reason = ""
        if args.run_sam2 or args.require_sam2:
            sam2_checkpoint = Path(str(args.sam2_checkpoint))
            if not sam2_checkpoint.is_absolute():
                sam2_checkpoint = (_repo_root() / sam2_checkpoint).resolve()
            prereq_failure = _sam2_prerequisite_failure(sam2_checkpoint)
            if prereq_failure:
                if args.require_sam2:
                    raise SmokeFailure(f"SAM2 prerequisites missing: {prereq_failure}", kind="environment")
                sam2_skip_reason = prereq_failure
            else:
                sam2_output_dir = output_dir / "sam2"
                sam2_output_dir.mkdir(parents=True, exist_ok=True)
                try:
                    sam2_result = _run_backend_validation(
                        base_url,
                        api_key=str(args.api_key),
                        input_dir=input_dir,
                        output_dir=sam2_output_dir,
                        backend="sam2",
                        timeout_seconds=float(args.timeout_seconds),
                        poll_interval_seconds=float(args.poll_interval_seconds),
                        sam2_checkpoint_path=sam2_checkpoint,
                        sam2_model_size=str(args.sam2_model_size),
                    )
                except SmokeFailure as exc:
                    if args.require_sam2 or exc.kind not in {"environment"}:
                        raise
                    sam2_skip_reason = f"sam2_runtime_failure:{exc}"

        print("portal-lux-materials-live: ok")
        print(f"base_url: {base_url}")
        print(f"efficientsam_job_id: {efficient_result['job_id']}")
        print(f"efficientsam_mask: {efficient_result['mask_relative_path']}")
        print(f"efficientsam_run_card: {efficient_result['run_card_path']}")
        if sam2_result is not None:
            print(f"sam2_job_id: {sam2_result['job_id']}")
            print(f"sam2_mask: {sam2_result['mask_relative_path']}")
        elif args.run_sam2 or args.require_sam2:
            print(f"sam2_skipped: {sam2_skip_reason or 'not_run'}")
        if _should_cleanup_output_dir(keep_output=bool(args.keep_output), output_dir_is_temp=output_dir_is_temp):
            print(f"output_dir_cleaned: {output_dir}")
        else:
            print(f"output_dir: {output_dir}")
        if input_dir_is_temp and bool(args.keep_output):
            print(f"input_dir: {input_dir}")
        return 0
    finally:
        if backend_runtime is not None:
            _terminate_runtime(backend_runtime)
        if input_dir is not None and input_dir_is_temp and not bool(args.keep_output):
            shutil.rmtree(input_dir, ignore_errors=True)
        if (
            output_dir is not None
            and _should_cleanup_output_dir(keep_output=bool(args.keep_output), output_dir_is_temp=output_dir_is_temp)
        ):
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmokeFailure as exc:
        print(f"portal-lux-materials-live: failed ({exc.kind}): {exc}", file=sys.stderr)
        raise SystemExit(1)
