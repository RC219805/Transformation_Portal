#!/usr/bin/env python3
"""
Browser smoke validation for the portal UI against a live backend.

This script launches a disposable Chrome instance with the DevTools protocol
enabled, drives the real portal UI in a browser context, and verifies the build
surface across all four pipelines plus one safe archive dispatch.

Coverage:
1. Portal loads in a real browser and renders expected controls.
2. Health check reports the backend as online.
3. Build view cycles through `lux-depth-v3`, `archive-gate-a`, `archive-gate-b`, and `archive-gate-c`.
4. Archive gating fields and canonical command badges match the selected stage.
5. `archive-gate-b` and `archive-gate-c` stay blocked without a rights manifest.
6. A safe `archive-gate-a` dispatch succeeds from the real UI.
7. Queue, inspector, artifacts, and live log surfaces reflect completion.

Run via:
    python scripts/validation/validate_portal_browser_smoke.py
    make validate-portal-browser

Environment overrides:
    TP_ORCHESTRATOR_BASE_URL   Backend URL (default: http://127.0.0.1:8000)
    TP_API_KEY                 API key for protected job endpoints
    TP_PORTAL_BROWSER_BINARY   Chrome binary path override
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import secrets
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


class SmokeFailure(RuntimeError):
    """Raised when the browser smoke validation fails."""

    def __init__(self, message: str, *, kind: str = "generic") -> None:
        super().__init__(message)
        self.kind = kind


DEFAULT_ORCHESTRATOR_BASE_URL = "http://127.0.0.1:8000"


@dataclass
class LocalRuntimeHandle:
    process: subprocess.Popen[str]
    base_url: str
    log_path: Path
    temp_paths: tuple[Path, ...] = ()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _fixture_archive_root() -> Path:
    return _repo_root() / "tests" / "fixtures" / "archive_small" / "archive_root"


def _fixture_archive_index() -> Path:
    return _repo_root() / "tests" / "fixtures" / "archive_small" / "archive_index_normalized.csv.gz"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _default_output_dir() -> Path:
    if os.name != "nt" and Path("/tmp").exists():
        return Path("/tmp/gate-a-smoke-portal")
    kwargs: Dict[str, Any] = {"prefix": "tp-portal-browser-smoke-"}
    return Path(tempfile.mkdtemp(**kwargs))


def _default_profile_dir() -> Path:
    kwargs: Dict[str, Any] = {"prefix": "tp-portal-browser-profile-"}
    if os.name != "nt" and Path("/tmp").exists():
        kwargs["dir"] = "/tmp"
    return Path(tempfile.mkdtemp(**kwargs))


def _resolve_output_dir(raw_output_dir: str) -> tuple[Path, bool]:
    candidate = str(raw_output_dir).strip()
    if candidate:
        return Path(candidate).resolve(), False
    return _default_output_dir(), True


def _should_cleanup_output_dir(*, keep_output: bool, output_dir_is_temp: bool) -> bool:
    return output_dir_is_temp and not keep_output


def _default_chrome_binary() -> str:
    candidates = [
        os.getenv("TP_PORTAL_BROWSER_BINARY", "").strip(),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        shutil.which("google-chrome") or "",
        shutil.which("chrome") or "",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise SmokeFailure("Google Chrome binary not found. Set TP_PORTAL_BROWSER_BINARY to a valid Chrome executable.")


def _resolve_chrome_binary(raw_chrome_binary: str) -> str:
    candidate = str(raw_chrome_binary).strip()
    if candidate:
        return candidate
    return _default_chrome_binary()


def _base_url(value: str) -> str:
    trimmed = value.strip()
    if not trimmed:
        raise SmokeFailure("Base URL cannot be empty")
    return trimmed.rstrip("/")


def _tail_text(path: Path, *, max_chars: int = 1200, max_bytes: int = 4096) -> str:
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


def _wait_for_backend_ready(
    base_url: str,
    *,
    timeout_seconds: float,
    process: Optional[subprocess.Popen[str]] = None,
    log_path: Optional[Path] = None,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Optional[str] = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            break
        try:
            status, body = _request_json(base_url, "/ready")
            if status == 200 and body.get("ok") is True:
                return
            last_error = f"status={status} body={body}"
        except SmokeFailure as exc:
            last_error = str(exc)
        time.sleep(0.25)

    if process is not None and process.poll() is not None:
        exit_code = process.returncode
        log_tail = _tail_text(log_path) if log_path is not None else ""
        detail = f"local backend exited before readiness (code {exit_code})"
        if log_tail:
            detail = f"{detail}. Recent log output:\n{log_tail}"
        raise SmokeFailure(detail, kind="runtime")

    detail = last_error or "timed out waiting for /ready"
    raise SmokeFailure(
        f"Local backend did not become ready at {base_url}/ready within {timeout_seconds:.1f}s ({detail}).",
        kind="runtime",
    )


def _spawn_local_backend(api_key: str, *, timeout_seconds: float) -> LocalRuntimeHandle:
    runtime_root = Path(
        tempfile.mkdtemp(
            prefix="tp-portal-browser-backend-",
            dir="/tmp" if os.name != "nt" and Path("/tmp").exists() else None,
        )
    )
    log_path = runtime_root / "uvicorn.log"
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TP_RATE_LIMIT_PER_MINUTE"] = "0"
    if api_key:
        env["TP_API_KEY"] = api_key
    else:
        env.pop("TP_API_KEY", None)

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

    handle = LocalRuntimeHandle(
        process=process,
        base_url=base_url,
        log_path=log_path,
        temp_paths=(runtime_root,),
    )
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


def _http_get_json(url: str, timeout: float = 10.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _request_json(
    base_url: str,
    path: str,
    *,
    api_key: str = "",
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
) -> tuple[int, Dict[str, Any]]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        _base_url(base_url) + path,
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
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


def _list_job_ids(base_url: str, api_key: str) -> list[str]:
    status, body = _request_json(base_url, "/v1/jobs", api_key=api_key)
    if status != 200:
        raise SmokeFailure(f"GET /v1/jobs returned {status}: {body}")
    jobs = body.get("data", {}).get("jobs", [])
    if not isinstance(jobs, list):
        raise SmokeFailure(f"GET /v1/jobs returned unexpected payload: {body}")
    return [str(job.get("id") or "").strip() for job in jobs if str(job.get("id") or "").strip()]


def _lux_preview_payload(archive_root: Path, output_dir: Path) -> Dict[str, Any]:
    return {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": str(archive_root),
            "output_dir": str(output_dir),
        },
    }


def _preflight_lux_config_preview(
    base_url: str,
    api_key: str,
    *,
    archive_root: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    payload = _lux_preview_payload(archive_root, output_dir)

    try:
        status, body = _request_json(
            base_url,
            "/v1/config-preview",
            api_key=api_key,
            method="POST",
            payload=payload,
        )
    except SmokeFailure as exc:
        raise SmokeFailure(
            "Preview preflight failed: /v1/config-preview could not be reached. Check backend preview/readiness before running the browser smoke.",
            kind="environment",
        ) from exc

    error_payload = body.get("error") if isinstance(body, dict) else {}
    if not isinstance(error_payload, dict):
        error_payload = {}
    error_details = error_payload.get("details") if isinstance(error_payload.get("details"), dict) else {}
    error_reason = str(error_details.get("reason") or error_payload.get("code") or "").strip().lower()

    if status in {401, 403}:
        raise SmokeFailure(
            "Preview preflight failed: /v1/config-preview rejected the API key. Ensure TP_API_KEY matches the running backend before validate-portal-browser.",
            kind="environment",
        )
    if status == 400:
        detail = error_reason or "invalid_request"
        raise SmokeFailure(
            f"Preview preflight failed: /v1/config-preview rejected the Lux payload or contract ({detail}).",
            kind="contract",
        )
    if status >= 500:
        raise SmokeFailure(
            "Preview preflight failed: /v1/config-preview is unavailable. Check backend preview/readiness before dispatch validation.",
            kind="environment",
        )
    if status != 200:
        raise SmokeFailure(
            f"Preview preflight failed: /v1/config-preview returned unexpected status {status}.",
            kind="environment",
        )

    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, dict):
        raise SmokeFailure(
            "Preview preflight failed: /v1/config-preview returned an invalid JSON envelope.",
            kind="contract",
        )

    field_errors = data.get("field_errors") or []
    if field_errors:
        first_error = field_errors[0] if isinstance(field_errors[0], dict) else {}
        field = str(first_error.get("field") or "payload").strip()
        message = str(first_error.get("message") or "Preview validation blocked the Lux payload.").strip()
        raise SmokeFailure(f"Preview preflight failed: {field}: {message}", kind="contract")

    return data


def _poll_for_new_backend_job_id(
    base_url: str,
    api_key: str,
    *,
    known_job_ids: set[str],
    timeout_seconds: float,
    interval_seconds: float = 0.25,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        current_job_ids = _list_job_ids(base_url, api_key)
        for job_id in current_job_ids:
            if job_id not in known_job_ids:
                return job_id
        time.sleep(interval_seconds)
    raise SmokeFailure("Timed out waiting for submitted backend job to appear in GET /v1/jobs")


def _wait_for_devtools(port: int, timeout_seconds: float = 20.0) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            return _http_get_json(f"http://127.0.0.1:{port}/json/version", timeout=2.0)
        except Exception as exc:  # pragma: no cover - best effort polling
            last_error = exc
            time.sleep(0.25)
    raise SmokeFailure(f"DevTools endpoint on port {port} did not become ready: {last_error}")


def _list_devtools_targets(port: int) -> list[Dict[str, Any]]:
    payload = _http_get_json(f"http://127.0.0.1:{port}/json/list", timeout=5.0)
    if not isinstance(payload, list):
        raise SmokeFailure(f"Unexpected DevTools target payload: {payload!r}")
    return [item for item in payload if isinstance(item, dict)]


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


class DevToolsConnection:
    """Minimal Chrome DevTools WebSocket client using only the stdlib."""

    def __init__(self, websocket_url: str, timeout_seconds: float = 20.0) -> None:
        parsed = urllib.parse.urlparse(websocket_url)
        if parsed.scheme != "ws":
            raise SmokeFailure(f"Unsupported DevTools websocket URL: {websocket_url}")
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        self._sock = socket.create_connection((host, port), timeout_seconds)
        self._sock.settimeout(timeout_seconds)
        self._next_id = 1
        self._handshake(host, port, path)

    def _handshake(self, host: str, port: int, path: str) -> None:
        raw_key = secrets.token_bytes(16)
        key = base64.b64encode(raw_key).decode("ascii")
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self._sock.sendall(request.encode("ascii"))
        response = self._read_http_headers()
        header_text = response.decode("latin-1", errors="replace")
        if "101" not in header_text.splitlines()[0]:
            raise SmokeFailure(f"DevTools websocket handshake failed: {header_text}")

        expected_accept = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
        ).decode("ascii")
        if f"Sec-WebSocket-Accept: {expected_accept}" not in header_text:
            raise SmokeFailure("DevTools websocket handshake returned invalid Sec-WebSocket-Accept header")

    def _read_http_headers(self) -> bytes:
        data = bytearray()
        while b"\r\n\r\n" not in data:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise SmokeFailure("Unexpected EOF while reading DevTools websocket handshake")
            data.extend(chunk)
        return bytes(data)

    def close(self) -> None:
        try:
            self._send_frame(0x8, b"")
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass

    def _recv_exact(self, size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            chunk = self._sock.recv(size - len(chunks))
            if not chunk:
                raise SmokeFailure("Unexpected EOF while reading DevTools websocket frame")
            chunks.extend(chunk)
        return bytes(chunks)

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        frame = bytearray()
        frame.append(0x80 | (opcode & 0x0F))
        payload_length = len(payload)
        mask_bit = 0x80
        if payload_length < 126:
            frame.append(mask_bit | payload_length)
        elif payload_length < (1 << 16):
            frame.append(mask_bit | 126)
            frame.extend(struct.pack("!H", payload_length))
        else:
            frame.append(mask_bit | 127)
            frame.extend(struct.pack("!Q", payload_length))

        mask = secrets.token_bytes(4)
        frame.extend(mask)
        masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        frame.extend(masked)
        self._sock.sendall(frame)

    def _recv_message(self, timeout_seconds: float) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SmokeFailure("Timed out waiting for DevTools response")
            self._sock.settimeout(remaining)
            header = self._recv_exact(2)
            first, second = header[0], header[1]
            opcode = first & 0x0F
            masked = bool(second & 0x80)
            length = second & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._recv_exact(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._recv_exact(8))[0]
            mask = self._recv_exact(4) if masked else b""
            payload = self._recv_exact(length) if length else b""
            if masked:
                payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))

            if opcode == 0x9:  # ping
                self._send_frame(0xA, payload)
                continue
            if opcode == 0xA:  # pong
                continue
            if opcode == 0x8:  # close
                raise SmokeFailure("DevTools websocket closed unexpectedly")
            if opcode != 0x1:
                continue
            try:
                return json.loads(payload.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise SmokeFailure(f"Received invalid DevTools JSON payload: {payload!r}") from exc

    def call(self, method: str, params: Optional[Dict[str, Any]] = None, timeout_seconds: float = 20.0) -> Dict[str, Any]:
        command_id = self._next_id
        self._next_id += 1
        payload = {"id": command_id, "method": method, "params": params or {}}
        self._send_frame(0x1, json.dumps(payload).encode("utf-8"))

        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SmokeFailure(f"Timed out waiting for DevTools response to {method}")
            message = self._recv_message(remaining)
            if message.get("id") != command_id:
                continue
            if "error" in message:
                raise SmokeFailure(f"DevTools call {method} failed: {message['error']}")
            return message.get("result", {})

    def evaluate(self, expression: str, timeout_seconds: float = 20.0) -> Any:
        result = self.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": True,
            },
            timeout_seconds=timeout_seconds,
        )
        if "exceptionDetails" in result:
            raise SmokeFailure(f"Browser evaluation failed: {result['exceptionDetails']}")
        remote_object = result.get("result") or {}
        return remote_object.get("value")


def _wait_for_page_target(port: int, timeout_seconds: float = 20.0) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        targets = _list_devtools_targets(port)
        for target in targets:
            if target.get("type") == "page" and target.get("webSocketDebuggerUrl"):
                return target
        time.sleep(0.25)
    raise SmokeFailure("Chrome did not expose a page target for browser smoke validation")


def _poll(
    connection: DevToolsConnection,
    expression: str,
    *,
    predicate,
    timeout_seconds: float,
    interval_seconds: float = 0.25,
    description: str,
) -> Any:
    deadline = time.monotonic() + timeout_seconds
    last_value: Any = None
    while time.monotonic() < deadline:
        try:
            last_value = connection.evaluate(expression)
        except SmokeFailure as exc:
            message = str(exc)
            if "Inspected target navigated or closed" in message:
                time.sleep(interval_seconds)
                continue
            raise
        if predicate(last_value):
            return last_value
        time.sleep(interval_seconds)
    raise SmokeFailure(f"Timed out waiting for {description}: last value={last_value!r}")


def _portal_shell_ready(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if str(value.get("readyState", "")) != "complete":
        return False
    if "Transformation Portal" not in str(value.get("title", "")):
        return False
    if str(value.get("bootstrapStatus", "")).lower() not in {"ready", "degraded"}:
        return False
    return bool(value.get("overviewViewVisible"))


def _state_probe_expression() -> str:
    return r"""
(() => {
  const text = (id) => {
    const el = document.getElementById(id);
    return el ? (el.textContent || '').trim() : '';
  };
  const value = (id) => {
    const el = document.getElementById(id);
    return el ? String(el.value || '') : '';
  };
  const visible = (id) => {
    const el = document.getElementById(id);
    if (!el) return false;
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
  };
  const buttonMeta = (id) => {
    const el = document.getElementById(id);
    if (!el) {
      return { visible: false, label: '', key: '', tone: '' };
    }
    return {
      visible: !el.classList.contains('hidden') && !el.disabled,
      label: (el.textContent || '').trim(),
      key: String(el.dataset.actionKey || ''),
      tone: String(el.dataset.tone || '')
    };
  };
  const consoleActionPrimary = buttonMeta('consoleActionPrimaryBtn');
  const consoleActionSecondary1 = buttonMeta('consoleActionSecondaryBtn1');
  const consoleActionSecondary2 = buttonMeta('consoleActionSecondaryBtn2');
  const selectedRecoveryPrimary = buttonMeta('selectedJobRecoveryPrimaryBtn');
  const selectedRecoverySecondary = buttonMeta('selectedJobRecoverySecondaryBtn');
  const reviewStatusPrimary = buttonMeta('reviewStatusPrimaryBtn');
  const reviewStatusSecondary = buttonMeta('reviewStatusSecondaryBtn');
  return {
    title: document.title,
    readyState: document.readyState,
    locationSearch: window.location.search,
    bootstrapStatus: document.body ? String(document.body.dataset.bootstrapStatus || '') : '',
    currentView: document.body ? String(document.body.dataset.consoleView || '') : '',
    pipeline: value('pipelineSelect'),
    inputDir: value('inputDir'),
    outputDir: value('outputDir'),
    authModeBadge: text('authModeBadge'),
    healthText: text('healthText'),
    heroReadinessLabel: text('heroReadinessLabel'),
    connectionDetailsVisible: visible('connectionDetails'),
    connectionDetailsOpen: (() => {
      const el = document.getElementById('connectionDetails');
      return !!(el && el.open);
    })(),
    queueCount: text('queueCount'),
    selectedJobState: text('selectedJobStateBadge'),
    selectedJobId: text('selectedJobIdLabel'),
    selectedJobArtifactCount: text('selectedJobArtifactCount'),
    selectedJobStreamStatus: text('selectedJobStreamStatus'),
    selectedJobMetaLine: text('selectedJobMetaLine'),
    selectedJobFreshness: text('selectedJobFreshness'),
    selectedJobSummary: text('selectedJobSummary'),
    contextRibbonVisible: (() => {
      const el = document.getElementById('consoleContextRibbon');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    contextRibbonJob: text('contextRibbonJob'),
    contextRibbonState: text('contextRibbonState'),
    contextRibbonFreshness: text('contextRibbonFreshness'),
    contextRibbonArtifact: text('contextRibbonArtifact'),
    contextRibbonCompare: text('contextRibbonCompare'),
    actionRailVisible: (() => {
      const el = document.getElementById('consoleActionRail');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    actionRailTitle: text('consoleActionRailTitle'),
    actionRailDetail: text('consoleActionRailDetail'),
    actionPrimaryVisible: consoleActionPrimary.visible,
    actionPrimaryLabel: consoleActionPrimary.label,
    actionPrimaryKey: consoleActionPrimary.key,
    actionPrimaryTone: consoleActionPrimary.tone,
    actionSecondary1Visible: consoleActionSecondary1.visible,
    actionSecondary1Label: consoleActionSecondary1.label,
    actionSecondary1Key: consoleActionSecondary1.key,
    actionSecondary1Tone: consoleActionSecondary1.tone,
    actionSecondary2Visible: consoleActionSecondary2.visible,
    actionSecondary2Label: consoleActionSecondary2.label,
    actionSecondary2Key: consoleActionSecondary2.key,
    actionSecondary2Tone: consoleActionSecondary2.tone,
    reviewStatusTitle: text('reviewStatusTitle'),
    reviewStatusDetail: text('reviewStatusDetail'),
    reviewStatusTone: (() => {
      const el = document.getElementById('reviewStatusBanner');
      return el ? String(el.dataset.tone || '') : '';
    })(),
    reviewStatusState: (() => {
      const el = document.getElementById('reviewStatusBanner');
      return el ? String(el.dataset.reviewState || '') : '';
    })(),
    reviewStatusVisible: (() => {
      const el = document.getElementById('reviewStatusBanner');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    reviewStatusPrimaryVisible: reviewStatusPrimary.visible,
    reviewStatusPrimaryLabel: reviewStatusPrimary.label,
    reviewStatusPrimaryKey: reviewStatusPrimary.key,
    reviewStatusSecondaryVisible: reviewStatusSecondary.visible,
    reviewStatusSecondaryLabel: reviewStatusSecondary.label,
    reviewStatusSecondaryKey: reviewStatusSecondary.key,
    selectedRecoveryPrimaryVisible: selectedRecoveryPrimary.visible,
    selectedRecoveryPrimaryLabel: selectedRecoveryPrimary.label,
    selectedRecoveryPrimaryKey: selectedRecoveryPrimary.key,
    selectedRecoverySecondaryVisible: selectedRecoverySecondary.visible,
    selectedRecoverySecondaryLabel: selectedRecoverySecondary.label,
    selectedRecoverySecondaryKey: selectedRecoverySecondary.key,
    reviewProvenanceArtifactRole: text('reviewProvenanceArtifactRole'),
    reviewProvenanceRunState: text('reviewProvenanceRunState'),
    reviewProvenancePath: text('reviewProvenancePath'),
    reviewProvenanceFreshness: text('reviewProvenanceFreshness'),
    reviewProvenanceSource: text('reviewProvenanceSource'),
    reviewProvenanceBatch: text('reviewProvenanceBatch'),
    reviewCompareTitle: text('reviewCompareTitle'),
    reviewCompareDetail: text('reviewCompareDetail'),
    reviewCompareVisible: (() => {
      const el = document.getElementById('reviewCompareSummary');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    reviewCompareEnabled: (() => {
      const el = document.getElementById('artifactCompareBtn');
      return !!(el && String(el.getAttribute('aria-pressed') || '') === 'true');
    })(),
    artifactViewerVisible: (() => {
      const el = document.getElementById('artifactViewerModal');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    artifactViewerTitle: text('artifactViewerTitle'),
    artifactViewerPath: text('artifactViewerPath'),
    artifactViewerFingerprint: text('artifactViewerFingerprint'),
    artifactViewerZoomValue: text('artifactViewerZoomValue'),
    artifactViewerStatus: text('artifactViewerStatus'),
    artifactViewerFallbackVisible: (() => {
      const el = document.getElementById('artifactViewerFallback');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    artifactViewerFallbackTitle: text('artifactViewerFallbackTitle'),
    artifactViewerFallbackDetail: text('artifactViewerFallbackDetail'),
    summaryReconstructionState: text('summaryReconstructionState'),
    summaryRuntimeWorkers: text('summaryRuntimeWorkers'),
    summaryPreviewState: text('summaryPreviewState'),
    postureBandVisible: (() => {
      const el = document.querySelector('[data-ui="build-posture-band"]');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    summaryBandOutsideReconstruction: (() => {
      const summary = document.getElementById('reconstructionRuntimeSummary');
      const disclosure = document.getElementById('reconstructionDetails');
      return !!(summary && disclosure && !disclosure.contains(summary));
    })(),
    dispatchPrimaryLaneVisible: (() => {
      const el = document.querySelector('[data-ui="dispatch-primary-lane"]');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    dispatchReadinessReason: text('dispatchReadinessReason'),
    rawPreviewStatus: (() => {
      const preview = typeof state !== 'undefined' && state.preview && typeof state.preview === 'object'
        ? state.preview
        : null;
      return preview ? String(preview.status || '').trim() : '';
    })(),
    previewRequestKey: (() => {
      const preview = typeof state !== 'undefined' && state.preview && typeof state.preview === 'object'
        ? state.preview
        : null;
      return preview ? String(preview.requestKey || '').trim() : '';
    })(),
    currentPreviewRequestKey: (() => {
      try {
        return String(_configPreviewRequestKey(generatePayload()) || '').trim();
      } catch (_err) {
        return '';
      }
    })(),
    previewRequestKeyMatches: (() => {
      try {
        const preview = typeof state !== 'undefined' && state.preview && typeof state.preview === 'object'
          ? state.preview
          : null;
        if (!preview) return false;
        return String(preview.requestKey || '').trim() === String(_configPreviewRequestKey(generatePayload()) || '').trim();
      } catch (_err) {
        return false;
      }
    })(),
    cliFirstLine: (() => {
      const preview = text('cliPreview');
      return preview ? preview.split('\n')[0].trim() : '';
    })(),
    cliText: text('cliPreview'),
    archiveCanonicalCommand: text('archiveCanonicalCommand'),
    archiveIndexPath: value('archiveIndexPath'),
    rightsManifestPath: value('rightsManifestPath'),
    preRunWarnings: Array.from(document.querySelectorAll('#preRunWarnings li')).map((item) =>
      String(item.textContent || '').trim()
    ),
    dispatchChecklistRows: document.querySelectorAll('#preRunWarnings li[data-tone], #preRunWarnings li').length,
    dispatchChecklistHasPass: Array.from(document.querySelectorAll('#preRunWarnings li')).some((item) =>
      String(item.textContent || '').trim().startsWith('PASS:')
    ),
    dispatchChecklistHasBlock: Array.from(document.querySelectorAll('#preRunWarnings li')).some((item) =>
      String(item.textContent || '').trim().startsWith('BLOCK:')
    ),
    preRunWarningsEmptyVisible: (() => {
      const el = document.getElementById('preRunWarningsEmpty');
      if (!el) return false;
      return window.getComputedStyle(el).display !== 'none';
    })(),
    missingArchiveIndexWarningVisible: Array.from(document.querySelectorAll('#preRunWarnings li')).some((item) =>
      String(item.textContent || '').toLowerCase().includes('archive index')
    ),
    archiveIndexFieldVisible: (() => {
      const el = document.getElementById('archiveIndexField');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    rightsManifestFieldVisible: (() => {
      const el = document.getElementById('rightsManifestField');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    stagedUploadReceiptVisible: visible('stagedUploadSummary') && !!text('stagedUploadSummary'),
    logHasFixityWrite: (() => {
      const el = document.getElementById('logPane');
      return !!(el && (el.textContent || '').includes('Wrote 3 rows'));
    })(),
    queueRows: document.querySelectorAll('#jobList li[data-job-id]').length,
    queueJobIds: Array.from(document.querySelectorAll('#jobList li[data-job-id]')).map((row) =>
      String(row.getAttribute('data-job-id') || '')
    ),
    firstQueueJobId: (() => {
      const row = document.querySelector('#jobList li[data-job-id]');
      return row ? String(row.getAttribute('data-job-id') || '') : '';
    })(),
    buildViewVisible: (() => {
      return visible('build-shell');
    })(),
    operateViewVisible: (() => {
      return visible('jobs-shell');
    })(),
    overviewViewVisible: (() => {
      return visible('overview-shell');
    })(),
    buildStepperVisible: !!document.querySelector('[data-ui="build-stepper"]'),
    activeBuildStep: (() => {
      const el = document.querySelector('#buildStepTabs .build-step-tab.is-active');
      return el ? String(el.getAttribute('data-build-step-target') || '') : '';
    })(),
    runJobDisabled: (() => {
      const el = document.getElementById('runJobBtn');
      return !!(el && el.disabled);
    })(),
    heroRunDisabled: (() => {
      const el = document.getElementById('heroRunBtn');
      return !!(el && el.disabled);
    })(),
    queueShellHidden: (() => {
      const el = document.getElementById('queue-shell');
      return !!(el && el.classList.contains('hidden'));
    })(),
    queueEmptyStateVisible: visible('emptyQueueState'),
    artifactEmptyStateVisible: visible('emptyArtifactState'),
    archiveFieldsVisible: (() => {
      const el = document.getElementById('fieldsArchiveGate');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    luxFieldsVisible: (() => {
      const el = document.getElementById('fieldsLuxDepth');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    flagsShellVisible: (() => {
      const el = document.getElementById('flags-shell');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    enableSegmentationChecked: (() => {
      const el = document.getElementById('enableSegmentation');
      return !!(el && el.checked);
    })(),
    segmentationBackendVisible: (() => {
      const el = document.getElementById('segmentationBackendField');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    segmentationBackendValue: (() => {
      const el = document.getElementById('segmentationBackend');
      return el ? String(el.value || '') : '';
    })(),
    strictSegmentationVisible: (() => {
      const el = document.getElementById('strictSegmentationField');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    strictSegmentationChecked: (() => {
      const el = document.getElementById('strictSegmentation');
      return !!(el && el.checked);
    })(),
    sam2ModelSizeVisible: (() => {
      const el = document.getElementById('sam2ModelSizeField');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    sam2CheckpointVisible: (() => {
      const el = document.getElementById('sam2CheckpointField');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    governanceDetailsVisible: (() => {
      const el = document.getElementById('governanceDetails');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    advancedFlagsOpen: (() => {
      const el = document.getElementById('advancedFlagsDetails');
      return !!(el && el.open);
    })(),
    governanceDetailsOpen: (() => {
      const el = document.getElementById('governanceDetails');
      return !!(el && el.open);
    })(),
    licenseAppleVisible: (() => {
      const el = document.getElementById('licenseAppleField');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    licenseResearchToolsVisible: (() => {
      const el = document.getElementById('licenseResearchToolsField');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    reconstructionConfigVisible: (() => {
      const el = document.getElementById('reconstructionConfigFields');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    reconstructionDetailsOpen: (() => {
      const el = document.getElementById('reconstructionDetails');
      return !!(el && el.open);
    })(),
    debugBundleGuardrailVisible: (() => {
      const el = document.getElementById('debugBundleGuardrail');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    captioningDetailsVisible: (() => {
      const el = document.getElementById('captioningDetails');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    captioningDetailsOpen: (() => {
      const el = document.getElementById('captioningDetails');
      return !!(el && el.open);
    })(),
    captioningEnabledChecked: (() => {
      const el = document.getElementById('enableFastVlmCaptioning');
      return !!(el && el.checked);
    })(),
    captioningFieldsVisible: (() => {
      const el = document.getElementById('fastVlmCaptioningFields');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    captioningStatusText: text('captioningStatus'),
    captioningReadinessText: (() => {
      const scope = text('captioningReadinessScope');
      const checks = Array.from(document.querySelectorAll('#captioningReadinessList li'))
        .map((item) => String(item.textContent || '').trim())
        .filter(Boolean)
        .join(' ');
      return `${scope} ${checks}`.trim();
    })(),
    captioningReadinessStatus: (() => {
      const el = document.getElementById('captioningReadinessList');
      return el ? String(el.dataset.status || '') : '';
    })(),
    captioningCliHasFlag: text('cliPreview').includes('--vlm-captioning'),
    captioningExpectedOutput: Array.from(document.querySelectorAll('#expectedOutputsList li')).some((item) =>
      String(item.textContent || '').toLowerCase().includes('fastvlm')
    ),
    captioningAdvisoryWarningVisible: Array.from(document.querySelectorAll('#preRunWarnings li')).some((item) =>
      String(item.textContent || '').toLowerCase().includes('caption')
    ),
    dispatchToolsOpen: (() => {
      const el = document.getElementById('dispatchToolsDetails');
      return !!(el && el.open);
    })(),
    debugBundleAcknowledgeChecked: (() => {
      const el = document.getElementById('debugBundleAcknowledge');
      return !!(el && el.checked);
    })(),
    effectiveConfigDrawerVisible: (() => {
      const el = document.getElementById('effectiveConfigDrawer');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    reviewSurfaceVisible: (() => {
      const el = document.querySelector('[data-ui="review-surface"]');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    advisoryCaptionPanelVisible: (() => {
      const el = document.querySelector('[data-ui="advisory-caption-panel"]');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    advisoryCaptionPanelText: (() => {
      const el = document.querySelector('[data-ui="advisory-caption-panel"]');
      return el ? String(el.textContent || '').trim() : '';
    })(),
    v2PresetVisible: (() => {
      const el = document.getElementById('v2PresetField');
      return !!(el && !el.classList.contains('hidden'));
    })()
  };
})()
"""


def _accessibility_probe_expression() -> str:
    return r"""
(() => {
  const visible = (el) => {
    if (!el) return false;
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
  };
  const minTarget = (selector) => {
    const el = document.querySelector(selector);
    if (!visible(el)) return false;
    const rect = el.getBoundingClientRect();
    return rect.width >= 44 && rect.height >= 44;
  };
  const maxDisclosureDepth = (() => {
    const detailsNodes = Array.from(document.querySelectorAll('details'));
    if (!detailsNodes.length) return 0;
    const depthFor = (node) => {
      let depth = 1;
      let current = node.parentElement;
      while (current) {
        if (current.tagName === 'DETAILS') depth += 1;
        current = current.parentElement;
      }
      return depth;
    };
    return Math.max(...detailsNodes.map(depthFor));
  })();
  const focusTarget = () =>
    document.getElementById('pipelineSelect')
    || document.getElementById('presetSelect')
    || document.getElementById('buildStepTab1')
    || document.querySelector('[data-ui="view-link"]')
    || document.getElementById('themeBtn');
  const stickyBlockers = () =>
    Array.from(document.querySelectorAll('.portal-topbar, [data-ui="console-context-shell"]'))
      .filter((el) => {
        if (!visible(el)) return false;
        const position = window.getComputedStyle(el).position;
        return position === 'sticky' || position === 'fixed';
      });
  const measureStickyBlockerBottom = () =>
    stickyBlockers().reduce((max, el) => Math.max(max, el.getBoundingClientRect().bottom), 0);
  const focusVisibleWithStickyShells = (() => {
    const target = focusTarget();
    if (!visible(target)) return false;
    const absoluteTop = target.getBoundingClientRect().top + window.scrollY;
    const root = document.documentElement;
    const previousScrollBehavior = root.style.scrollBehavior;
    const previousScrollX = window.scrollX;
    const previousScrollY = window.scrollY;
    const clearance = Math.ceil(measureStickyBlockerBottom() + 16);
    root.style.scrollBehavior = 'auto';
    try {
      window.scrollTo(previousScrollX, Math.max(0, absoluteTop - clearance));
      try {
        target.focus({ preventScroll: true });
      } catch {
        target.focus();
      }
      const blockerBottom = measureStickyBlockerBottom();
      const targetRect = target.getBoundingClientRect();
      return targetRect.top >= blockerBottom - 2 && targetRect.bottom <= window.innerHeight + 2;
    } finally {
      window.scrollTo(previousScrollX, previousScrollY);
      root.style.scrollBehavior = previousScrollBehavior;
    }
  })();
  const focusTargetNode = focusTarget();
  const discoverableDisclosures = Array.from(document.querySelectorAll('details > summary'))
    .filter((summary) => visible(summary) && String(summary.textContent || '').trim().length > 0)
    .length;
  const shellNoise = document.querySelector('.shell-noise');
  const visiblePulseAnimation = Array.from(document.querySelectorAll('.animate-pulse'))
    .some((el) => visible(el) && window.getComputedStyle(el).animationName !== 'none' && window.getComputedStyle(el).animationDuration !== '0s');
  return {
    currentView: document.body ? String(document.body.dataset.consoleView || '') : '',
    themeTargetMin: minTarget('#themeBtn'),
    shortcutsTargetMin: minTarget('#shortcutsBtn'),
    workspaceLinkTargetMin: minTarget('[data-ui="view-link"]'),
    buildStepTargetMin: minTarget('#buildStepTab1'),
    connectionDetailsTargetMin: minTarget('#connectionDetails > summary'),
    focusTargetId: focusTargetNode ? String(focusTargetNode.id || focusTargetNode.getAttribute('data-ui') || focusTargetNode.tagName || '') : '',
    focusTargetTop: focusTargetNode ? Number(focusTargetNode.getBoundingClientRect().top || 0) : 0,
    focusTargetBottom: focusTargetNode ? Number(focusTargetNode.getBoundingClientRect().bottom || 0) : 0,
    stickyBlockerBottom: measureStickyBlockerBottom(),
    focusVisibleWithStickyShells,
    maxDisclosureDepth,
    discoverableDisclosures,
    reducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
    decorativeMotionStatic:
      (!shellNoise || window.getComputedStyle(shellNoise).display === 'none')
      && !visiblePulseAnimation
  };
})()
"""


def _navigate_to_console_view_expression(
    view: str,
    job_id: str = "",
    artifact_path: str = "",
    compare_enabled: bool = False,
) -> str:
    payload = json.dumps(
        {
            "view": view,
            "job_id": job_id,
            "artifact_path": artifact_path,
            "compare_enabled": compare_enabled,
        }
    )
    return f"""
(() => {{
  const cfg = {payload};
  const url = new URL(window.location.href);
  url.searchParams.set('view', cfg.view);
  if ((cfg.view === 'operate' || cfg.view === 'review') && cfg.job_id) {{
    url.searchParams.set('job', cfg.job_id);
    if (cfg.artifact_path) {{
      url.searchParams.set('artifact', cfg.artifact_path);
    }} else {{
      url.searchParams.delete('artifact');
    }}
    if (cfg.compare_enabled) {{
      url.searchParams.set('compare', '1');
    }} else {{
      url.searchParams.delete('compare');
    }}
  }} else {{
    url.searchParams.delete('job');
    url.searchParams.delete('artifact');
    url.searchParams.delete('compare');
  }}
  window.history.pushState({{}}, '', url.toString());
  window.dispatchEvent(new PopStateEvent('popstate'));
  return url.toString();
}})()
"""


def _set_pipeline_form_expression(
    *,
    api_key: str,
    pipeline: str,
    input_dir: str,
    output_dir: str,
    archive_index: str = "",
    manifest_jsonl: str = "",
    build_step: str = "",
) -> str:
    payload = json.dumps(
        {
            "api_key": api_key,
            "pipeline": pipeline,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "archive_index": archive_index,
            "manifest_jsonl": manifest_jsonl,
            "build_step": build_step,
        }
    )
    return f"""
(() => {{
  const cfg = {payload};
  const dispatch = (el, type) => el.dispatchEvent(new Event(type, {{ bubbles: true }}));
  const setValue = (id, value) => {{
    const el = document.getElementById(id);
    if (!el) throw new Error(`missing #${{id}}`);
    el.value = value;
    dispatch(el, 'input');
    dispatch(el, 'change');
  }};
  const rememberApiKey = document.getElementById('rememberApiKey');
  if (rememberApiKey) {{
    rememberApiKey.checked = false;
    dispatch(rememberApiKey, 'change');
  }}
  try {{
    sessionStorage.removeItem('tp_portal_transient_draft');
  }} catch {{}}
  setValue('apiKeyInput', cfg.api_key);
  try {{
    localStorage.removeItem('tp_api_key');
    if (cfg.api_key) {{
      sessionStorage.setItem('tp_api_key', cfg.api_key);
    }} else {{
      sessionStorage.removeItem('tp_api_key');
    }}
  }} catch {{}}
  if (typeof _persistApiKeyFromInputs === 'function') {{
    _persistApiKeyFromInputs();
  }}
  setValue('pipelineSelect', cfg.pipeline);
  setValue('inputDir', cfg.input_dir);
  setValue('outputDir', cfg.output_dir);
  setValue('archiveIndexPath', cfg.archive_index);
  setValue('rightsManifestPath', cfg.manifest_jsonl);
  if (cfg.build_step) {{
    const buildStepButton = document.querySelector(`[data-build-step-target="${{cfg.build_step}}"]`);
    if (buildStepButton) {{
      buildStepButton.click();
    }} else if (typeof setBuildStep === 'function') {{
      setBuildStep(cfg.build_step, {{ silent: true }});
    }}
  }}
  if (typeof reconcileBuildSurfaceFromDom === 'function') {{
    reconcileBuildSurfaceFromDom();
  }}
  if (typeof scheduleConfigPreview === 'function') {{
    scheduleConfigPreview(true);
  }}
  return {{
    currentView: document.body ? String(document.body.dataset.consoleView || '') : '',
    pipeline: document.getElementById('pipelineSelect').value,
    inputDir: document.getElementById('inputDir').value,
    outputDir: document.getElementById('outputDir').value,
    archiveFieldsVisible: !document.getElementById('fieldsArchiveGate').classList.contains('hidden'),
    luxFieldsVisible: !document.getElementById('fieldsLuxDepth').classList.contains('hidden'),
    flagsShellVisible: !document.getElementById('flags-shell').classList.contains('hidden'),
    archiveCanonicalCommand: (document.getElementById('archiveCanonicalCommand').textContent || '').trim(),
    archiveIndexFieldVisible: !document.getElementById('archiveIndexField').classList.contains('hidden'),
    rightsManifestFieldVisible: !document.getElementById('rightsManifestField').classList.contains('hidden'),
    archiveIndexPath: document.getElementById('archiveIndexPath').value,
    rightsManifestPath: document.getElementById('rightsManifestPath').value,
    runJobDisabled: !!document.getElementById('runJobBtn').disabled,
    buildStepperVisible: !!document.querySelector('[data-ui="build-stepper"]'),
    activeBuildStep: (() => {{
      const el = document.querySelector('#buildStepTabs .build-step-tab.is-active');
      return el ? String(el.getAttribute('data-build-step-target') || '') : '';
    }})(),
    heroReadinessLabel: (document.getElementById('heroReadinessLabel').textContent || '').trim(),
    cliFirstLine: ((document.getElementById('cliPreview').textContent || '').trim().split('\\n')[0] || '').trim(),
    cliText: (document.getElementById('cliPreview').textContent || '').trim(),
    summaryReconstructionState: (document.getElementById('summaryReconstructionState').textContent || '').trim(),
    summaryRuntimeWorkers: (document.getElementById('summaryRuntimeWorkers').textContent || '').trim(),
    summaryPreviewState: (document.getElementById('summaryPreviewState').textContent || '').trim(),
    enableSegmentationChecked: !!document.getElementById('enableSegmentation').checked,
    segmentationBackendVisible: !document.getElementById('segmentationBackendField').classList.contains('hidden'),
    segmentationBackendValue: document.getElementById('segmentationBackend').value,
    strictSegmentationVisible: !document.getElementById('strictSegmentationField').classList.contains('hidden'),
    strictSegmentationChecked: !!document.getElementById('strictSegmentation').checked,
    sam2ModelSizeVisible: !document.getElementById('sam2ModelSizeField').classList.contains('hidden'),
    sam2CheckpointVisible: !document.getElementById('sam2CheckpointField').classList.contains('hidden'),
    governanceDetailsVisible: !document.getElementById('governanceDetails').classList.contains('hidden'),
    advancedFlagsOpen: !!document.getElementById('advancedFlagsDetails').open,
    governanceDetailsOpen: !!document.getElementById('governanceDetails').open,
    licenseAppleVisible: !document.getElementById('licenseAppleField').classList.contains('hidden'),
    licenseResearchToolsVisible: !document.getElementById('licenseResearchToolsField').classList.contains('hidden'),
    reconstructionConfigVisible: !document.getElementById('reconstructionConfigFields').classList.contains('hidden'),
    reconstructionDetailsOpen: !!document.getElementById('reconstructionDetails').open,
    debugBundleGuardrailVisible: !document.getElementById('debugBundleGuardrail').classList.contains('hidden'),
    dispatchToolsOpen: !!document.getElementById('dispatchToolsDetails').open,
    effectiveConfigDrawerVisible: !document.getElementById('effectiveConfigDrawer').classList.contains('hidden'),
    v2PresetVisible: !document.getElementById('v2PresetField').classList.contains('hidden')
  }};
}})()
"""


def _restore_archive_gate_form_without_events_expression(
    *,
    input_dir: str,
    output_dir: str,
    archive_index: str,
) -> str:
    payload = json.dumps(
        {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "archive_index": archive_index,
        }
    )
    return f"""
(() => {{
  const cfg = {payload};
  const setValue = (id, value) => {{
    const el = document.getElementById(id);
    if (!el) throw new Error(`missing #${{id}}`);
    el.value = value;
  }};
  setValue('inputDir', cfg.input_dir);
  setValue('outputDir', cfg.output_dir);
  setValue('archiveIndexPath', cfg.archive_index);
  window.dispatchEvent(new Event('pageshow'));
  window.dispatchEvent(new Event('focus'));
  return {{
    inputDir: document.getElementById('inputDir').value,
    outputDir: document.getElementById('outputDir').value,
    archiveIndexPath: document.getElementById('archiveIndexPath').value
  }};
}})()
"""


def _set_lux_optional_controls_expression(
    *,
    depth_backend: str,
    enable_segmentation: bool,
    segmentation_backend: str,
    enable_reconstruction: bool,
    enable_v2: bool,
    emit_scene_debug_bundle: bool = False,
    enable_captioning: bool = False,
) -> str:
    payload = json.dumps(
        {
            "depth_backend": depth_backend,
            "enable_segmentation": enable_segmentation,
            "segmentation_backend": segmentation_backend,
            "enable_reconstruction": enable_reconstruction,
            "enable_v2": enable_v2,
            "emit_scene_debug_bundle": emit_scene_debug_bundle,
            "enable_captioning": enable_captioning,
        }
    )
    return f"""
(() => {{
  const cfg = {payload};
  const dispatch = (el, type) => el.dispatchEvent(new Event(type, {{ bubbles: true }}));
  const setValue = (id, value) => {{
    const el = document.getElementById(id);
    if (!el) throw new Error(`missing #${{id}}`);
    el.value = value;
    dispatch(el, 'input');
    dispatch(el, 'change');
  }};
  const setChecked = (id, checked) => {{
    const el = document.getElementById(id);
    if (!el) throw new Error(`missing #${{id}}`);
    el.checked = !!checked;
    dispatch(el, 'change');
  }};
  setValue('depthBackend', cfg.depth_backend);
  setChecked('enableSegmentation', cfg.enable_segmentation);
  setValue('segmentationBackend', cfg.segmentation_backend);
  setChecked('enableReconstruction', cfg.enable_reconstruction);
  setChecked('flagEnableV2', cfg.enable_v2);
  setChecked('emitSceneDebugBundle', cfg.emit_scene_debug_bundle);
  const captioningDetails = document.getElementById('captioningDetails');
  if (cfg.enable_captioning && captioningDetails && !captioningDetails.classList.contains('hidden')) {{
    captioningDetails.open = true;
    setChecked('enableFastVlmCaptioning', true);
    setValue('fastVlmCaptioningModel', 'smoke');
    setValue('fastVlmProxyFormat', 'png');
    setValue('fastVlmMaxSidePx', '960');
    setValue('fastVlmTimeoutSeconds', '45');
  }}
  return ({_state_probe_expression()});
}})()
"""


def _click_expression(selector: str) -> str:
    encoded = json.dumps(selector)
    return f"""
(() => {{
  const el = document.querySelector({encoded});
  if (!el) throw new Error('missing element for selector ' + {encoded});
  el.click();
  return true;
}})()
"""


def _key_expression(key: str) -> str:
    return f"""
(() => {{
  const event = new KeyboardEvent('keydown', {{
    key: {json.dumps(key)},
    bubbles: true,
    cancelable: true
  }});
  document.dispatchEvent(event);
  return ({_state_probe_expression()});
}})()
"""


def _simulate_bootstrap_degraded_expression(*, reason: str, http_status: int) -> str:
    payload = json.dumps({"reason": reason, "http_status": http_status})
    return f"""
(() => {{
  const cfg = {payload};
  _applyPortalBootstrap(portalInternals.defaultPortalBootstrapPayload(), {{
    status: 'degraded',
    reason: cfg.reason,
    httpStatus: cfg.http_status
  }});
  renderSelectedJobInspector();
  renderArtifactPanel();
  renderConsoleContextRibbon();
  return ({_state_probe_expression()});
}})()
"""


def _inject_compare_ready_review_expression(job_id: str) -> str:
    payload = json.dumps({"job_id": job_id})
    return f"""
(() => {{
  const cfg = {payload};
  const job = (typeof state !== 'undefined' && Array.isArray(state.jobs))
    ? state.jobs.find((item) => String(item && item.id || '') === String(cfg.job_id || ''))
    : null;
  if (!job) {{
    throw new Error(`missing job ${{cfg.job_id}}`);
  }}
  upsertArtifact(job, {{
    path: 'synthetic/review-primary.png',
    relative_path: 'synthetic/review-primary.png',
    artifact_type: 'image',
    media_kind: 'image',
    previewable: true,
    content_type: 'image/png',
    size_bytes: 2048,
    sha256: '1111111111111111111111111111111111111111111111111111111111111111',
    display_hint: {{
      label: 'Synthetic Primary',
      priority: 1000,
      compare_group: 'portal-smoke-compare'
    }}
  }});
  upsertArtifact(job, {{
    path: 'synthetic/review-compare.png',
    relative_path: 'synthetic/review-compare.png',
    artifact_type: 'image',
    media_kind: 'image',
    previewable: true,
    content_type: 'image/png',
    size_bytes: 1984,
    sha256: '2222222222222222222222222222222222222222222222222222222222222222',
    display_hint: {{
      label: 'Synthetic Compare',
      priority: 990,
      compare_group: 'portal-smoke-compare'
    }}
  }});
  upsertArtifact(job, {{
    path: 'synthetic/review-primary.png.vlm_captioning.sidecar.json',
    relative_path: 'synthetic/review-primary.png.vlm_captioning.sidecar.json',
    artifact_type: 'vlm_caption_sidecar',
    media_kind: 'json',
    previewable: false,
    content_type: 'application/json',
    size_bytes: 512,
    sha256: '4444444444444444444444444444444444444444444444444444444444444444',
    display_hint: {{
      label: 'Advisory Caption',
      role: 'vlm_caption'
    }}
  }});
  state.artifactUi.selectedByJob[String(cfg.job_id)] = 'synthetic/review-primary.png';
  state.artifactUi.compareByJob[String(cfg.job_id)] = false;
  renderReviewSurfaces();
  return ({_state_probe_expression()});
}})()
"""


def _inject_viewer_fallback_review_expression(job_id: str) -> str:
    payload = json.dumps({"job_id": job_id})
    return f"""
(() => {{
  const cfg = {payload};
  const job = (typeof state !== 'undefined' && Array.isArray(state.jobs))
    ? state.jobs.find((item) => String(item && item.id || '') === String(cfg.job_id || ''))
    : null;
  if (!job) {{
    throw new Error(`missing job ${{cfg.job_id}}`);
  }}
  upsertArtifact(job, {{
    path: 'synthetic/review-report.json',
    relative_path: 'synthetic/review-report.json',
    artifact_type: 'report',
    media_kind: 'json',
    previewable: false,
    content_type: 'application/json',
    size_bytes: 1024,
    sha256: '3333333333333333333333333333333333333333333333333333333333333333',
    display_hint: {{
      label: 'Synthetic Report',
      priority: 970
    }}
  }});
  state.artifactUi.selectedByJob[String(cfg.job_id)] = 'synthetic/review-report.json';
  state.artifactUi.compareByJob[String(cfg.job_id)] = false;
  renderReviewSurfaces();
  return ({_state_probe_expression()});
}})()
"""


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("TP_ORCHESTRATOR_BASE_URL", DEFAULT_ORCHESTRATOR_BASE_URL),
        help="Portal/backend base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("TP_API_KEY", "").strip(),
        help="API key for protected job endpoints (default: unset; uses TP_API_KEY when set)",
    )
    parser.add_argument(
        "--spawn-local-backend",
        action="store_true",
        help="Launch an isolated local backend on a free port and validate against it",
    )
    parser.add_argument(
        "--backend-startup-timeout-seconds",
        type=float,
        default=30.0,
        help="Wait budget for an auto-launched local backend to become ready (default: %(default)s)",
    )
    parser.add_argument(
        "--chrome-binary",
        default=os.getenv("TP_PORTAL_BROWSER_BINARY", "").strip(),
        help="Chrome executable path (default: TP_PORTAL_BROWSER_BINARY or auto-detect)",
    )
    parser.add_argument(
        "--archive-root",
        default=str(_fixture_archive_root()),
        help="Archive root for the safe archive-gate fixture job",
    )
    parser.add_argument(
        "--archive-index",
        default=str(_fixture_archive_index()),
        help="Archive index for the safe archive-gate fixture job",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory for the browser-submitted job (defaults to the canonical smoke path)",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Preserve the browser-submitted job output directory instead of deleting it",
    )
    parser.add_argument(
        "--keep-profile",
        action="store_true",
        help="Preserve the temporary Chrome profile for debugging",
    )
    parser.add_argument(
        "--debugging-port",
        type=int,
        default=0,
        help="Chrome remote debugging port (default: auto-select free port)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=45.0,
        help="Overall wait budget for portal/job transitions (default: %(default)s)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    runtime_handle: Optional[LocalRuntimeHandle] = None
    chrome_process: Optional[subprocess.Popen[str]] = None
    connection: Optional[DevToolsConnection] = None
    profile_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    cleanup_output_dir = False

    try:
        base_url = _base_url(str(args.base_url))
        if args.spawn_local_backend:
            print("portal-browser-smoke: launching isolated local backend", flush=True)
            runtime_handle = _spawn_local_backend(
                args.api_key,
                timeout_seconds=args.backend_startup_timeout_seconds,
            )
            base_url = runtime_handle.base_url
            print(f"portal-browser-smoke: isolated backend ready at {base_url}", flush=True)

        archive_root = Path(args.archive_root).resolve()
        _expect(archive_root.is_dir(), f"Archive root fixture does not exist: {archive_root}")
        archive_index = Path(args.archive_index).resolve()
        _expect(archive_index.is_file(), f"Archive index fixture does not exist: {archive_index}")

        output_dir, output_dir_is_temp = _resolve_output_dir(args.output_dir)
        if output_dir_is_temp and not args.keep_output and output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        profile_dir = _default_profile_dir()
        port = int(args.debugging_port or _find_free_port())
        chrome_binary = _resolve_chrome_binary(args.chrome_binary)
        _expect(Path(chrome_binary).exists(), f"Chrome binary does not exist: {chrome_binary}")
        cleanup_output_dir = _should_cleanup_output_dir(
            keep_output=bool(args.keep_output),
            output_dir_is_temp=output_dir_is_temp,
        )

        print("portal-browser-smoke: preflighting lux config preview", flush=True)
        _preflight_lux_config_preview(
            base_url,
            args.api_key,
            archive_root=archive_root,
            output_dir=output_dir,
        )

        print("portal-browser-smoke: launching chrome", flush=True)
        command = [
            chrome_binary,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile_dir}",
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-networking",
            "--disable-component-update",
            "--disable-sync",
            "--disable-extensions",
            "--disable-popup-blocking",
            "about:blank",
        ]
        chrome_process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print("portal-browser-smoke: connecting devtools", flush=True)
        _wait_for_devtools(port)
        target = _wait_for_page_target(port)
        websocket_url = str(target.get("webSocketDebuggerUrl") or "").strip()
        _expect(websocket_url.startswith("ws://"), f"Invalid DevTools websocket URL: {websocket_url!r}")

        connection = DevToolsConnection(websocket_url)
        connection.call("Page.enable")
        connection.call("Runtime.enable")
        connection.call("Page.navigate", {"url": base_url}, timeout_seconds=20.0)

        print("portal-browser-smoke: waiting for portal shell", flush=True)
        initial_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=_portal_shell_ready,
            timeout_seconds=args.timeout_seconds,
            description="portal document ready",
        )
        _expect(
            str(initial_state.get("currentView", "")) == "overview",
            f"Portal did not default to overview view: {initial_state}",
        )

        print("portal-browser-smoke: waiting for backend health", flush=True)
        online_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: isinstance(value, dict) and "Online" in str(value.get("healthText", "")),
            timeout_seconds=args.timeout_seconds,
            description="portal backend health to become online",
        )
        _expect(
            str(online_state.get("pipeline", "")) == "lux-depth-v3",
            f"Portal did not default to lux-depth-v3: {online_state}",
        )
        _expect(bool(online_state.get("overviewViewVisible")), f"Overview shell did not remain visible: {online_state}")
        overview_accessibility = connection.evaluate(_accessibility_probe_expression())
        _expect(
            bool(overview_accessibility.get("themeTargetMin"))
            and bool(overview_accessibility.get("shortcutsTargetMin"))
            and bool(overview_accessibility.get("workspaceLinkTargetMin")),
            f"Portal overview controls fell below the 44px contract: {overview_accessibility}",
        )
        _expect(
            int(overview_accessibility.get("maxDisclosureDepth", 0)) <= 1,
            f"Portal disclosure depth exceeded the single-level contract: {overview_accessibility}",
        )
        _expect(
            int(overview_accessibility.get("discoverableDisclosures", 0)) >= 1,
            f"Portal disclosures were no longer discoverable: {overview_accessibility}",
        )

        print("portal-browser-smoke: opening build view", flush=True)
        connection.evaluate(_navigate_to_console_view_expression("build"))
        build_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("currentView", "")) == "build"
                and bool(value.get("buildViewVisible"))
            ),
            timeout_seconds=args.timeout_seconds,
            description="build view to become active",
        )
        _expect(
            not bool(build_state.get("operateViewVisible")),
            f"Build view should suppress operate shell: {build_state}",
        )
        build_accessibility = connection.evaluate(_accessibility_probe_expression())
        _expect(
            bool(build_accessibility.get("buildStepTargetMin")),
            f"Build step tabs fell below the 44px contract: {build_accessibility}",
        )
        _expect(
            bool(build_accessibility.get("connectionDetailsTargetMin")),
            f"Connection details disclosure fell below the 44px target-size contract: {build_accessibility}",
        )
        _expect(
            bool(build_accessibility.get("focusVisibleWithStickyShells")),
            f"Sticky portal chrome obscured focused controls: {build_accessibility}",
        )
        _expect(
            bool(build_state.get("connectionDetailsVisible")),
            f"Build view should keep the connection-details disclosure visible: {build_state}",
        )

        print("portal-browser-smoke: checking empty operate/review states before dispatch", flush=True)
        connection.evaluate(_navigate_to_console_view_expression("operate"))
        empty_operate_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("currentView", "")) == "operate"
                and bool(value.get("queueEmptyStateVisible"))
                and bool(value.get("artifactEmptyStateVisible"))
            ),
            timeout_seconds=args.timeout_seconds,
            description="empty operate state before the first dispatch",
        )
        _expect(
            str(empty_operate_state.get("selectedJobId", "")).strip() == "No job selected",
            f"Operate empty state should keep the inspector explicit about the lack of a selected run: {empty_operate_state}",
        )
        connection.evaluate(_navigate_to_console_view_expression("build"))
        _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("currentView", "")) == "build"
                and bool(value.get("buildViewVisible"))
            ),
            timeout_seconds=args.timeout_seconds,
            description="build view to restore after the empty-state operate check",
        )

        draft_input_dir = str(archive_root)
        draft_output_dir = f"{output_dir}-session-draft"
        print("portal-browser-smoke: verifying transient build draft restore after reload", flush=True)
        draft_state = _poll(
            connection,
            _set_pipeline_form_expression(
                api_key="",
                pipeline="lux-depth-v3",
                input_dir=draft_input_dir,
                output_dir=draft_output_dir,
                build_step="3",
            ),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("currentView", "")) == "build"
                and str(value.get("pipeline", "")) == "lux-depth-v3"
                and str(value.get("inputDir", "")) == draft_input_dir
                and str(value.get("outputDir", "")) == draft_output_dir
                and str(value.get("activeBuildStep", "")) == "3"
            ),
            timeout_seconds=args.timeout_seconds,
            description="transient draft state to persist before reload",
        )
        _expect(
            str(draft_state.get("activeBuildStep", "")) == "3",
            f"Transient draft setup did not advance the builder to step 3: {draft_state}",
        )
        connection.call("Page.reload", {"ignoreCache": True})
        restored_draft_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("readyState", "")) == "complete"
                and str(value.get("currentView", "")) == "build"
                and str(value.get("pipeline", "")) == "lux-depth-v3"
                and str(value.get("inputDir", "")) == draft_input_dir
                and str(value.get("outputDir", "")) == draft_output_dir
                and str(value.get("activeBuildStep", "")) == "3"
            ),
            timeout_seconds=args.timeout_seconds,
            description="transient draft state to restore after reload",
        )
        _expect(
            "view=build" in str(restored_draft_state.get("locationSearch", "")),
            f"Transient draft reload should preserve the build route context: {restored_draft_state}",
        )

        print("portal-browser-smoke: checking reduced motion", flush=True)
        connection.call(
            "Emulation.setEmulatedMedia",
            {"features": [{"name": "prefers-reduced-motion", "value": "reduce"}]},
        )
        reduced_motion_state = _poll(
            connection,
            _accessibility_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and bool(value.get("reducedMotion"))
                and bool(value.get("decorativeMotionStatic"))
                and bool(value.get("buildStepTargetMin"))
            ),
            timeout_seconds=args.timeout_seconds,
            description="portal reduced-motion shell",
        )
        _expect(
            bool(reduced_motion_state.get("decorativeMotionStatic")),
            f"Reduced-motion mode left decorative portal motion active: {reduced_motion_state}",
        )
        connection.call(
            "Emulation.setEmulatedMedia",
            {"features": [{"name": "prefers-reduced-motion", "value": "no-preference"}]},
        )

        print("portal-browser-smoke: confirming lux-depth-v3 build defaults", flush=True)
        lux_state = connection.evaluate(
            _set_pipeline_form_expression(
                api_key=args.api_key,
                pipeline="lux-depth-v3",
                input_dir=str(archive_root),
                output_dir=str(output_dir),
                build_step="1",
            )
        )
        _expect(isinstance(lux_state, dict), f"Unexpected lux portal state: {lux_state!r}")
        _expect(lux_state.get("pipeline") == "lux-depth-v3", f"Lux pipeline did not remain selected: {lux_state}")
        _expect(
            bool(lux_state.get("buildStepperVisible")), f"Build stepper should stay visible in the Lux builder: {lux_state}"
        )
        _expect(str(lux_state.get("activeBuildStep", "")) == "1", f"Lux builder should begin on step 1: {lux_state}")
        _expect(not bool(lux_state.get("archiveFieldsVisible")), f"Lux build view should hide archive controls: {lux_state}")
        _expect(bool(lux_state.get("flagsShellVisible")), f"Lux build view should keep core flags visible: {lux_state}")
        _expect(
            bool(str(lux_state.get("heroReadinessLabel", "")).strip()),
            f"Lux build view did not expose any readiness label: {lux_state}",
        )
        lux_dispatch_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("currentView", "")) == "build"
                and str(value.get("pipeline", "")) == "lux-depth-v3"
                and int(value.get("dispatchChecklistRows", 0)) >= 4
            ),
            timeout_seconds=args.timeout_seconds,
            description="lux dispatch checklist rows to render",
        )
        _expect(
            bool(lux_dispatch_state.get("dispatchChecklistHasPass")),
            f"Dispatch checklist should expose explicit pass rows instead of qualitative-only copy: {lux_dispatch_state}",
        )
        _expect(
            bool(lux_state.get("enableSegmentationChecked")),
            f"Premium Lux should enable segmentation by default: {lux_state}",
        )
        _expect(
            bool(lux_state.get("segmentationBackendVisible")),
            f"Segmentation backend should be visible on the premium Lux default: {lux_state}",
        )
        _expect(
            str(lux_state.get("segmentationBackendValue", "")) == "efficientsam",
            f"Premium Lux should default to efficientsam segmentation: {lux_state}",
        )
        _expect(
            bool(lux_state.get("strictSegmentationVisible")),
            f"Strict segmentation should be visible on the premium Lux default: {lux_state}",
        )
        _expect(
            bool(lux_state.get("strictSegmentationChecked")),
            f"Premium Lux should enable strict segmentation by default: {lux_state}",
        )
        _expect(
            not bool(lux_state.get("sam2ModelSizeVisible")) and not bool(lux_state.get("sam2CheckpointVisible")),
            f"SAM2-only controls should stay hidden until SAM2 segmentation is selected: {lux_state}",
        )
        _expect(
            not bool(lux_state.get("governanceDetailsVisible")),
            f"Compliance acknowledgments should stay hidden on the default Lux configuration: {lux_state}",
        )
        _expect(
            not bool(lux_state.get("advancedFlagsOpen")),
            f"Advanced disclosure should stay collapsed for the default Lux run: {lux_state}",
        )
        _expect(
            not bool(lux_state.get("governanceDetailsOpen")),
            f"Governance disclosure should stay collapsed until the run requires it: {lux_state}",
        )
        _expect(
            not bool(lux_state.get("reconstructionConfigVisible")),
            f"Reconstruction-specific controls should stay hidden until reconstruction is enabled: {lux_state}",
        )
        _expect(
            not bool(lux_state.get("reconstructionDetailsOpen")),
            f"Reconstruction disclosure should stay collapsed for the default Lux run: {lux_state}",
        )
        _expect(
            str(lux_state.get("summaryReconstructionState", "")).strip() == "Off",
            f"Default reconstruction summary should report Off: {lux_state}",
        )
        _expect(
            bool(str(lux_state.get("summaryReconstructionState", "")).strip())
            and bool(str(lux_state.get("summaryRuntimeWorkers", "")).strip())
            and bool(str(lux_state.get("summaryPreviewState", "")).strip()),
            f"Step 3 posture summary should stay visible: {lux_state}",
        )
        _expect(
            bool(str(lux_state.get("heroReadinessLabel", "")).strip())
            and bool(str(lux_state.get("cliFirstLine", "")).strip()),
            f"Step 4 dispatch surface should stay visible: {lux_state}",
        )
        _expect(
            "Auto" in str(lux_state.get("summaryRuntimeWorkers", "")),
            f"Runtime worker summary should default to Auto: {lux_state}",
        )
        _expect(
            bool(str(lux_state.get("summaryPreviewState", "")).strip()),
            f"Preview status summary should be populated: {lux_state}",
        )
        _expect(
            bool(str(lux_state.get("heroReadinessLabel", "")).strip())
            and bool(str(lux_state.get("summaryPreviewState", "")).strip()),
            f"Dispatch lane state should stay visible while preview readiness settles: {lux_state}",
        )
        _expect(
            not bool(lux_state.get("v2PresetVisible")),
            f"V2 preset input should stay hidden until V2 compatibility is enabled: {lux_state}",
        )
        _expect(
            not bool(lux_state.get("dispatchToolsOpen")),
            f"Secondary dispatch tools should stay collapsed by default: {lux_state}",
        )

        lux_ready_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("pipeline", "")) == "lux-depth-v3"
                and not bool(value.get("runJobDisabled"))
                and str(value.get("summaryPreviewState", "")).strip() != "Refreshing"
            ),
            timeout_seconds=args.timeout_seconds,
            description="lux preview-backed dispatch readiness",
        )
        _expect(
            not bool(lux_ready_state.get("runJobDisabled")),
            f"Lux pipeline should become dispatchable once preview-backed validation settles: {lux_ready_state}",
        )
        _expect(
            "clear for dispatch" in str(lux_ready_state.get("dispatchReadinessReason", "")).strip().lower(),
            f"Dispatch lane should report a clear ready reason once Lux validation settles: {lux_ready_state}",
        )

        lux_context_state = _poll(
            connection,
            _set_lux_optional_controls_expression(
                depth_backend="depth_pro",
                enable_segmentation=True,
                segmentation_backend="sam2",
                enable_reconstruction=True,
                enable_v2=True,
                emit_scene_debug_bundle=True,
                enable_captioning=True,
            ),
            predicate=lambda value: (
                isinstance(value, dict)
                and bool(value.get("segmentationBackendVisible"))
                and bool(value.get("strictSegmentationVisible"))
                and bool(value.get("sam2ModelSizeVisible"))
                and bool(value.get("sam2CheckpointVisible"))
                and bool(value.get("governanceDetailsVisible"))
                and bool(value.get("governanceDetailsOpen"))
                and bool(value.get("licenseAppleVisible"))
                and bool(value.get("licenseResearchToolsVisible"))
                and bool(value.get("reconstructionConfigVisible"))
                and bool(value.get("reconstructionDetailsOpen"))
                and bool(value.get("debugBundleGuardrailVisible"))
                and bool(value.get("v2PresetVisible"))
            ),
            timeout_seconds=args.timeout_seconds,
            description="lux contextual control visibility",
        )
        _expect(
            bool(lux_context_state.get("governanceDetailsVisible")),
            f"Depth Pro and reconstruction should reveal governance acknowledgments: {lux_context_state}",
        )
        _expect(
            bool(lux_context_state.get("reconstructionConfigVisible")),
            f"Enabling reconstruction should reveal reconstruction controls: {lux_context_state}",
        )
        _expect(
            bool(lux_context_state.get("governanceDetailsOpen")),
            f"Governance disclosure should auto-open once acknowledgments become required: {lux_context_state}",
        )
        _expect(
            bool(lux_context_state.get("reconstructionDetailsOpen")),
            f"Reconstruction disclosure should auto-open once the feature is enabled: {lux_context_state}",
        )
        _expect(
            str(lux_context_state.get("summaryReconstructionState", "")).strip() == "On",
            f"Reconstruction summary should flip to On once the toggle is enabled: {lux_context_state}",
        )
        _expect(
            bool(lux_context_state.get("debugBundleGuardrailVisible")),
            f"Enabling debug bundle emission should reveal the guardrail warning: {lux_context_state}",
        )
        _expect(
            bool(lux_context_state.get("v2PresetVisible")),
            f"Enabling V2 should reveal the V2 preset input: {lux_context_state}",
        )
        _expect(
            bool(lux_context_state.get("advancedFlagsOpen")),
            f"Advanced disclosure should auto-open once advanced controls need operator attention: {lux_context_state}",
        )
        if bool(lux_context_state.get("captioningDetailsVisible")):
            _expect(
                bool(lux_context_state.get("captioningEnabledChecked")),
                f"FastVLM captioning should toggle on when the enabled feature gate exposes controls: {lux_context_state}",
            )
            _expect(
                bool(lux_context_state.get("captioningFieldsVisible")),
                f"FastVLM captioning fields should be visible after the advisory toggle is enabled: {lux_context_state}",
            )
            _expect(
                bool(lux_context_state.get("captioningCliHasFlag")),
                f"FastVLM captioning should be represented in the CLI preview when enabled: {lux_context_state}",
            )
            _expect(
                bool(lux_context_state.get("captioningExpectedOutput")),
                f"Expected outputs should include advisory FastVLM caption sidecars: {lux_context_state}",
            )
            _expect(
                bool(lux_context_state.get("captioningAdvisoryWarningVisible")),
                f"FastVLM captioning should surface advisory-only warning copy: {lux_context_state}",
            )
            _expect(
                "path-existence" in str(lux_context_state.get("captioningReadinessText") or ""),
                f"FastVLM captioning should expose path-existence readiness scope: {lux_context_state}",
            )
            _expect(
                bool(str(lux_context_state.get("captioningReadinessStatus") or "").strip()),
                f"FastVLM captioning should expose a stable readiness status: {lux_context_state}",
            )
        else:
            _expect(
                not bool(lux_context_state.get("captioningCliHasFlag")),
                f"FastVLM captioning args must stay out of the CLI preview when the feature gate hides controls: {lux_context_state}",
            )

        debug_bundle_reason_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and bool(value.get("runJobDisabled"))
                and "preview" in str(value.get("heroReadinessLabel", "")).strip().lower()
                and bool(value.get("debugBundleGuardrailVisible"))
                and bool(value.get("governanceDetailsOpen"))
            ),
            timeout_seconds=args.timeout_seconds,
            description="dispatch surface to report a blocked preview/governance state",
        )
        _expect(
            bool(debug_bundle_reason_state.get("runJobDisabled")),
            f"Dispatch should remain disabled while preview/governance blockers are active: {debug_bundle_reason_state}",
        )

        print("portal-browser-smoke: opening secondary dispatch tools", flush=True)
        connection.evaluate(_click_expression("#dispatchToolsDetails > summary"))
        dispatch_tools_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: isinstance(value, dict) and bool(value.get("dispatchToolsOpen")),
            timeout_seconds=args.timeout_seconds,
            description="secondary dispatch tools disclosure to open",
        )
        _expect(
            bool(dispatch_tools_state.get("dispatchToolsOpen")),
            f"Secondary dispatch tools should open on explicit operator request: {dispatch_tools_state}",
        )
        connection.evaluate(_click_expression("#dispatchToolsDetails > summary"))

        print("portal-browser-smoke: opening effective config drawer from the posture band", flush=True)
        connection.evaluate(_click_expression("#openEffectiveConfigBtn"))
        effective_config_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: isinstance(value, dict) and bool(value.get("effectiveConfigDrawerVisible")),
            timeout_seconds=args.timeout_seconds,
            description="effective config drawer to open",
        )
        _expect(
            bool(effective_config_state.get("effectiveConfigDrawerVisible")),
            f"Effective config drawer should open from the reconstruction summary strip: {effective_config_state}",
        )
        connection.evaluate(_click_expression("#closeEffectiveConfigBtn"))

        print("portal-browser-smoke: verifying archive-gate-b blocked state", flush=True)
        gate_b_state = _poll(
            connection,
            _set_pipeline_form_expression(
                api_key=args.api_key,
                pipeline="archive-gate-b",
                input_dir=str(archive_root),
                output_dir=str(output_dir),
            ),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("pipeline", "")) == "archive-gate-b"
                and bool(value.get("archiveFieldsVisible"))
                and str(value.get("archiveCanonicalCommand", "")) == "bag-build"
                and bool(value.get("runJobDisabled"))
                and str(value.get("heroReadinessLabel", "")).strip() == "Dispatch blocked"
            ),
            timeout_seconds=args.timeout_seconds,
            description="archive-gate-b build state",
        )
        _expect(
            bool(gate_b_state.get("rightsManifestFieldVisible")),
            f"archive-gate-b must expose the manifest field: {gate_b_state}",
        )
        _expect(
            str(gate_b_state.get("activeBuildStep", "")) == "2",
            f"archive-gate-b should move the builder to step 2: {gate_b_state}",
        )
        _expect(
            not bool(gate_b_state.get("archiveIndexFieldVisible")),
            f"archive-gate-b should hide archive index input: {gate_b_state}",
        )
        _expect(
            not bool(gate_b_state.get("flagsShellVisible")),
            f"archive-gate-b should hide Lux-only core flags: {gate_b_state}",
        )
        _expect(
            bool(gate_b_state.get("runJobDisabled")), f"archive-gate-b should stay blocked without manifest: {gate_b_state}"
        )
        _expect(
            str(gate_b_state.get("heroReadinessLabel", "")).strip() == "Dispatch blocked",
            f"archive-gate-b should advertise blocked readiness before manifest input: {gate_b_state}",
        )
        _expect(
            '--archive-command "bag-build"' in str(gate_b_state.get("cliText", "")),
            f"archive-gate-b CLI preview drifted from canonical command mapping: {gate_b_state}",
        )

        print("portal-browser-smoke: verifying archive-gate-c blocked state", flush=True)
        gate_c_state = _poll(
            connection,
            _set_pipeline_form_expression(
                api_key=args.api_key,
                pipeline="archive-gate-c",
                input_dir=str(archive_root),
                output_dir=str(output_dir),
            ),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("pipeline", "")) == "archive-gate-c"
                and str(value.get("archiveCanonicalCommand", "")) == "mets-export"
                and bool(value.get("runJobDisabled"))
            ),
            timeout_seconds=args.timeout_seconds,
            description="archive-gate-c build state",
        )
        _expect(
            bool(gate_c_state.get("rightsManifestFieldVisible")),
            f"archive-gate-c must expose the manifest field: {gate_c_state}",
        )
        _expect(
            str(gate_c_state.get("activeBuildStep", "")) == "2",
            f"archive-gate-c should move the builder to step 2: {gate_c_state}",
        )
        _expect(
            not bool(gate_c_state.get("archiveIndexFieldVisible")),
            f"archive-gate-c should hide archive index input: {gate_c_state}",
        )
        _expect(
            not bool(gate_c_state.get("flagsShellVisible")),
            f"archive-gate-c should hide Lux-only core flags: {gate_c_state}",
        )
        _expect(
            bool(gate_c_state.get("runJobDisabled")), f"archive-gate-c should stay blocked without manifest: {gate_c_state}"
        )
        _expect(
            '--archive-command "mets-export"' in str(gate_c_state.get("cliText", "")),
            f"archive-gate-c CLI preview drifted from canonical command mapping: {gate_c_state}",
        )

        print("portal-browser-smoke: configuring archive-gate-a without an archive index", flush=True)
        missing_index_state = _poll(
            connection,
            _set_pipeline_form_expression(
                api_key=args.api_key,
                pipeline="archive-gate-a",
                input_dir=str(archive_root),
                output_dir=str(output_dir),
                archive_index="",
            ),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("pipeline", "")) == "archive-gate-a"
                and str(value.get("archiveCanonicalCommand", "")) == "fixity-scan"
            ),
            timeout_seconds=args.timeout_seconds,
            description="archive-gate-a missing-index build state",
        )
        _expect(
            missing_index_state.get("pipeline") == "archive-gate-a",
            f"Pipeline switch to archive-gate-a failed: {missing_index_state}",
        )
        _expect(
            bool(missing_index_state.get("archiveFieldsVisible")) and not bool(missing_index_state.get("luxFieldsVisible")),
            f"Archive-specific UI did not toggle correctly: {missing_index_state}",
        )
        _expect(
            str(missing_index_state.get("activeBuildStep", "")) == "2",
            f"archive-gate-a should move the builder to step 2: {missing_index_state}",
        )
        _expect(
            not bool(missing_index_state.get("flagsShellVisible")),
            f"archive-gate-a should hide Lux-only core flags: {missing_index_state}",
        )
        _expect(
            bool(missing_index_state.get("archiveIndexFieldVisible"))
            and not bool(missing_index_state.get("rightsManifestFieldVisible")),
            f"archive-gate-a should expose only the archive index input: {missing_index_state}",
        )
        _expect(
            str(missing_index_state.get("archiveIndexPath", "")) == "",
            f"archive-gate-a should begin with a missing archive index for the restore-path regression: {missing_index_state}",
        )
        _expect(
            bool(missing_index_state.get("runJobDisabled")),
            f"archive-gate-a should stay blocked until the archive index is supplied: {missing_index_state}",
        )

        print("portal-browser-smoke: simulating browser-restored archive index", flush=True)
        restored_dom_state = connection.evaluate(
            _restore_archive_gate_form_without_events_expression(
                input_dir=str(archive_root),
                output_dir=str(output_dir),
                archive_index=str(archive_index),
            )
        )
        _expect(isinstance(restored_dom_state, dict), f"Unexpected restored DOM state: {restored_dom_state!r}")

        configured_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("pipeline", "")) == "archive-gate-a"
                and str(value.get("archiveIndexPath", "")) == str(archive_index)
                and '--archive-index "' in str(value.get("cliText", ""))
                and not bool(value.get("missingArchiveIndexWarningVisible"))
                and not bool(value.get("runJobDisabled"))
            ),
            timeout_seconds=args.timeout_seconds,
            description="archive-gate-a restored build state",
        )
        _expect(
            str(configured_state.get("inputDir", "")) == str(archive_root),
            f"Restored input directory did not survive reconciliation: {configured_state}",
        )
        _expect(
            str(configured_state.get("outputDir", "")) == str(output_dir),
            f"Restored output directory did not survive reconciliation: {configured_state}",
        )
        _expect(
            str(configured_state.get("cliFirstLine", "")).startswith("archive-gate-a"),
            f"CLI preview did not update for archive-gate-a: {configured_state}",
        )
        _expect(
            '--archive-command "fixity-scan"' in str(configured_state.get("cliText", "")),
            f"archive-gate-a CLI preview drifted from canonical command mapping: {configured_state}",
        )
        _expect(
            '--archive-index "' in str(configured_state.get("cliText", "")),
            f"archive-gate-a CLI preview should include archive index path: {configured_state}",
        )
        _expect(
            not bool(configured_state.get("preRunWarnings"))
            or not bool(configured_state.get("missingArchiveIndexWarningVisible")),
            f"archive-gate-a pre-run warnings should clear once the archive index is restored: {configured_state}",
        )

        known_job_ids = set(_list_job_ids(base_url, args.api_key))
        pre_submit_state = connection.evaluate(_state_probe_expression())
        _expect(isinstance(pre_submit_state, dict), f"Unexpected pre-submit portal state: {pre_submit_state!r}")
        pre_submit_queue_rows = int(pre_submit_state.get("queueRows") or 0)
        pre_submit_selected_job_id = str(pre_submit_state.get("selectedJobId") or "").strip()

        print("portal-browser-smoke: dispatching job", flush=True)
        connection.evaluate(_click_expression("#runJobBtn"))

        queued_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and (
                    int(value.get("queueRows") or 0) > pre_submit_queue_rows
                    or str(value.get("selectedJobId") or "").strip() != pre_submit_selected_job_id
                )
            ),
            timeout_seconds=args.timeout_seconds,
            description="queue and inspector to react to the submitted job",
        )
        submitted_job_id = _poll_for_new_backend_job_id(
            base_url,
            args.api_key,
            known_job_ids=known_job_ids,
            timeout_seconds=args.timeout_seconds,
        )
        _expect(submitted_job_id.startswith("job_"), f"Portal API did not expose a real backend job id: {queued_state}")

        print("portal-browser-smoke: opening operate view", flush=True)
        connection.evaluate(_navigate_to_console_view_expression("operate"))
        operate_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("currentView", "")) == "operate"
                and bool(value.get("operateViewVisible"))
                and submitted_job_id in [str(job_id).strip() for job_id in (value.get("queueJobIds") or [])]
            ),
            timeout_seconds=args.timeout_seconds,
            description="operate view to become active",
        )
        _expect(
            not bool(operate_state.get("buildViewVisible")),
            f"Operate view should suppress build shell: {operate_state}",
        )
        connection.evaluate(_click_expression(f'#jobList li[data-job-id="{submitted_job_id}"]'))

        print("portal-browser-smoke: waiting for terminal ui state", flush=True)
        terminal_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("selectedJobId", "")).startswith("job_")
                and str(value.get("selectedJobState", "")).strip().lower() in {"succeeded", "reviewable"}
                and "3 indexed" in str(value.get("selectedJobArtifactCount", ""))
                and bool(value.get("logHasFixityWrite"))
            ),
            timeout_seconds=args.timeout_seconds,
            description="browser-submitted job to reach a reviewable terminal state with artifacts and logs",
        )

        _expect(
            str(terminal_state.get("selectedJobId") or "").strip() == submitted_job_id,
            f"Portal inspector drifted from submitted job {submitted_job_id}: {terminal_state}",
        )
        _expect(
            f"view=operate&job={submitted_job_id}" in str(terminal_state.get("locationSearch", "")),
            f"Operate route should retain the selected job in the query string: {terminal_state}",
        )
        _expect(
            "Closed" in str(terminal_state.get("selectedJobStreamStatus", ""))
            or "Inactive" not in str(terminal_state.get("selectedJobStreamStatus", "")),
            f"Unexpected stream status after completion: {terminal_state}",
        )
        _expect(
            str(terminal_state.get("selectedJobFreshness", "")).strip().startswith("Updated "),
            f"Selected job freshness should stay immediately visible in operate: {terminal_state}",
        )
        _expect(
            bool(terminal_state.get("contextRibbonVisible"))
            and str(terminal_state.get("contextRibbonJob", "")).strip() == submitted_job_id,
            f"Operate view should expose the compact context ribbon for the selected job: {terminal_state}",
        )
        _expect(
            bool(terminal_state.get("actionRailVisible")),
            f"Operate view should surface the contextual action rail: {terminal_state}",
        )
        _expect(
            str(terminal_state.get("actionPrimaryKey", "")).strip() == "open_review",
            f"Completed runs should promote review entry as the primary action: {terminal_state}",
        )
        _expect(
            str(terminal_state.get("selectedRecoveryPrimaryKey", "")).strip() == "open_review",
            f"Selected-job recovery controls should reuse the review-entry action: {terminal_state}",
        )

        print("portal-browser-smoke: opening review from the contextual action rail", flush=True)
        connection.evaluate(_click_expression("#consoleActionPrimaryBtn"))
        run_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("currentView", "")) == "review"
                and str(value.get("selectedJobId", "")) == submitted_job_id
                and bool(value.get("queueShellHidden"))
                and bool(value.get("reviewSurfaceVisible"))
                and bool(value.get("reviewStatusVisible"))
                and str(value.get("reviewProvenancePath", "")).strip() != ""
                and "artifact=" in str(value.get("locationSearch", ""))
            ),
            timeout_seconds=args.timeout_seconds,
            description="review view to become active",
        )
        _expect(
            "3 indexed" in str(run_state.get("selectedJobArtifactCount", "")),
            f"Review view lost artifact context: {run_state}",
        )
        _expect(
            str(run_state.get("reviewStatusTone", "")).strip().lower() == "ready",
            f"Review workspace should expose a ready status banner after a successful run: {run_state}",
        )
        _expect(
            str(run_state.get("reviewStatusState", "")).strip() == "ready",
            f"Review workspace should expose the machine-readable review state token: {run_state}",
        )
        _expect(
            "Outputs ready for review" in str(run_state.get("reviewStatusTitle", "")),
            f"Review workspace should summarize output readiness directly: {run_state}",
        )
        _expect(
            str(run_state.get("reviewProvenanceRunState", "")).strip().lower().startswith("succeeded"),
            f"Review provenance should surface the terminal job state: {run_state}",
        )
        _expect(
            str(run_state.get("reviewProvenancePath", "")).strip()
            != "Preview, metadata, and actions will appear here when outputs are indexed.",
            f"Review provenance should identify the selected artifact path: {run_state}",
        )
        _expect(
            str(run_state.get("reviewProvenanceFreshness", "")).strip().startswith("Updated "),
            f"Review provenance should surface freshness for the selected run: {run_state}",
        )
        _expect(
            bool(run_state.get("contextRibbonVisible"))
            and str(run_state.get("contextRibbonArtifact", "")).strip()
            == str(run_state.get("reviewProvenancePath", "")).strip(),
            f"Review ribbon should stay aligned with the selected artifact context: {run_state}",
        )

        print("portal-browser-smoke: synthesizing compare-ready review state", flush=True)
        compare_ready_state = connection.evaluate(_inject_compare_ready_review_expression(submitted_job_id))
        _expect(isinstance(compare_ready_state, dict), f"Unexpected compare-ready portal state: {compare_ready_state!r}")
        compare_ready_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("currentView", "")) == "review"
                and str(value.get("selectedJobId", "")).strip() == submitted_job_id
                and str(value.get("reviewProvenancePath", "")).strip() == "synthetic/review-primary.png"
                and bool(value.get("advisoryCaptionPanelVisible"))
                and (
                    str(value.get("actionSecondary2Key", "")).strip() == "toggle_compare"
                    or str(value.get("actionSecondary1Key", "")).strip() == "toggle_compare"
                    or str(value.get("reviewStatusSecondaryKey", "")).strip() == "toggle_compare"
                )
            ),
            timeout_seconds=args.timeout_seconds,
            description="synthetic compare-capable review state",
        )
        selected_artifact_path = str(compare_ready_state.get("reviewProvenancePath", "")).strip()
        _expect(
            selected_artifact_path == "synthetic/review-primary.png",
            f"Compare-ready review state should promote the injected primary artifact: {compare_ready_state}",
        )
        _expect(
            "Advisory" in str(compare_ready_state.get("advisoryCaptionPanelText", "")),
            f"Review metadata should show the advisory FastVLM caption panel when a sidecar is indexed: {compare_ready_state}",
        )

        compare_toggle_selector = ""
        if str(compare_ready_state.get("actionSecondary2Key", "")).strip() == "toggle_compare":
            compare_toggle_selector = "#consoleActionSecondaryBtn2"
        elif str(compare_ready_state.get("actionSecondary1Key", "")).strip() == "toggle_compare":
            compare_toggle_selector = "#consoleActionSecondaryBtn1"
        elif str(compare_ready_state.get("reviewStatusSecondaryKey", "")).strip() == "toggle_compare":
            compare_toggle_selector = "#reviewStatusSecondaryBtn"
        _expect(
            bool(compare_toggle_selector),
            f"Ready review state should surface a compare toggle in the new action controls: {compare_ready_state}",
        )
        print("portal-browser-smoke: toggling compare from the new action controls", flush=True)
        connection.evaluate(_click_expression(compare_toggle_selector))
        compare_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("currentView", "")) == "review"
                and str(value.get("selectedJobId", "")).strip() == submitted_job_id
                and bool(value.get("reviewCompareEnabled"))
                and "compare=1" in str(value.get("locationSearch", ""))
            ),
            timeout_seconds=args.timeout_seconds,
            description="compare toggle action to enable compare mode",
        )
        _expect(
            str(compare_state.get("contextRibbonCompare", "")).strip().lower() == "compare on",
            f"Action-rail compare toggles should preserve the compare route contract: {compare_state}",
        )
        _expect(
            bool(compare_state.get("advisoryCaptionPanelVisible"))
            and "Advisory" in str(compare_state.get("advisoryCaptionPanelText", "")),
            f"Compare mode should keep a freshly rendered advisory FastVLM caption panel: {compare_state}",
        )

        print("portal-browser-smoke: opening the artifact viewer from review", flush=True)
        connection.evaluate(_click_expression("#openArtifactBtn"))
        viewer_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and bool(value.get("artifactViewerVisible"))
                and str(value.get("artifactViewerPath", "")).strip() == "synthetic/review-primary.png"
            ),
            timeout_seconds=args.timeout_seconds,
            description="artifact viewer to open for the selected review artifact",
        )
        _expect(
            bool(str(viewer_state.get("artifactViewerFingerprint", "")).strip()),
            f"Artifact viewer should expose the selected artifact fingerprint: {viewer_state}",
        )
        _expect(
            "zoom" in str(viewer_state.get("artifactViewerZoomValue", "")).lower(),
            f"Artifact viewer should expose the current zoom state: {viewer_state}",
        )

        print("portal-browser-smoke: navigating viewer artifacts with keyboard shortcuts", flush=True)
        connection.evaluate(_key_expression("ArrowRight"))
        viewer_next_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and bool(value.get("artifactViewerVisible"))
                and str(value.get("artifactViewerPath", "")).strip() == "synthetic/review-compare.png"
            ),
            timeout_seconds=args.timeout_seconds,
            description="artifact viewer keyboard next navigation",
        )
        _expect(
            "review-compare.png" in str(viewer_next_state.get("artifactViewerTitle", "")).lower(),
            f"Artifact viewer keyboard next should move to the paired artifact: {viewer_next_state}",
        )

        connection.evaluate(_key_expression("ArrowLeft"))
        viewer_prev_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and bool(value.get("artifactViewerVisible"))
                and str(value.get("artifactViewerPath", "")).strip() == "synthetic/review-primary.png"
            ),
            timeout_seconds=args.timeout_seconds,
            description="artifact viewer keyboard previous navigation",
        )
        _expect(
            "review-primary.png" in str(viewer_prev_state.get("artifactViewerTitle", "")).lower(),
            f"Artifact viewer keyboard previous should restore the primary artifact: {viewer_prev_state}",
        )

        print("portal-browser-smoke: adjusting viewer zoom with keyboard shortcuts", flush=True)
        connection.evaluate(_key_expression("+"))
        viewer_zoomed_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and bool(value.get("artifactViewerVisible"))
                and str(value.get("artifactViewerZoomValue", "")).strip() == "125% zoom"
            ),
            timeout_seconds=args.timeout_seconds,
            description="artifact viewer keyboard zoom in",
        )
        _expect(
            "125% zoom" == str(viewer_zoomed_state.get("artifactViewerZoomValue", "")).strip(),
            f"Artifact viewer keyboard zoom-in should update the viewer state: {viewer_zoomed_state}",
        )

        connection.evaluate(_key_expression("0"))
        viewer_reset_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and bool(value.get("artifactViewerVisible"))
                and str(value.get("artifactViewerZoomValue", "")).strip() == "100% zoom"
            ),
            timeout_seconds=args.timeout_seconds,
            description="artifact viewer keyboard zoom reset",
        )
        _expect(
            str(viewer_reset_state.get("artifactViewerStatus", "")).strip(),
            f"Artifact viewer should expose a live status message while open: {viewer_reset_state}",
        )

        print("portal-browser-smoke: closing the artifact viewer with Escape", flush=True)
        connection.evaluate(_key_expression("Escape"))
        _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: isinstance(value, dict) and not bool(value.get("artifactViewerVisible")),
            timeout_seconds=args.timeout_seconds,
            description="artifact viewer to close",
        )

        print("portal-browser-smoke: exercising artifact viewer degraded fallback", flush=True)
        fallback_review_state = connection.evaluate(_inject_viewer_fallback_review_expression(submitted_job_id))
        _expect(isinstance(fallback_review_state, dict), f"Unexpected viewer fallback state: {fallback_review_state!r}")
        fallback_review_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("reviewProvenancePath", "")).strip() == "synthetic/review-report.json"
            ),
            timeout_seconds=args.timeout_seconds,
            description="non-previewable review artifact to stage viewer fallback",
        )
        connection.evaluate(_click_expression("#openArtifactBtn"))
        fallback_viewer_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and bool(value.get("artifactViewerVisible"))
                and bool(value.get("artifactViewerFallbackVisible"))
                and str(value.get("artifactViewerFallbackTitle", "")).strip() == "Inline preview unavailable"
            ),
            timeout_seconds=args.timeout_seconds,
            description="artifact viewer fallback state for non-previewable artifacts",
        )
        _expect(
            "metadata fallback" in str(fallback_viewer_state.get("artifactViewerStatus", "")).lower(),
            f"Artifact viewer fallback should announce the degraded status through the live region: {fallback_viewer_state}",
        )
        _expect(
            str(fallback_viewer_state.get("artifactViewerPath", "")).strip() == "synthetic/review-report.json",
            f"Artifact viewer fallback should still expose the selected relative path: {fallback_viewer_state}",
        )
        _expect(
            bool(str(fallback_viewer_state.get("artifactViewerFingerprint", "")).strip()),
            f"Artifact viewer fallback should still expose the selected fingerprint: {fallback_viewer_state}",
        )
        connection.evaluate(_key_expression("Escape"))
        _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: isinstance(value, dict) and not bool(value.get("artifactViewerVisible")),
            timeout_seconds=args.timeout_seconds,
            description="artifact viewer fallback to close",
        )

        print("portal-browser-smoke: restoring review from an artifact deep link", flush=True)
        connection.evaluate(_navigate_to_console_view_expression("build"))
        _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("currentView", "")) == "build"
                and bool(value.get("buildViewVisible"))
            ),
            timeout_seconds=args.timeout_seconds,
            description="build view to restore before artifact deep-link replay",
        )
        connection.evaluate(_navigate_to_console_view_expression("review", submitted_job_id, selected_artifact_path))
        restored_review_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("currentView", "")) == "review"
                and str(value.get("selectedJobId", "")).strip() == submitted_job_id
                and str(value.get("reviewProvenancePath", "")).strip() == selected_artifact_path
                and "artifact=" in str(value.get("locationSearch", ""))
            ),
            timeout_seconds=args.timeout_seconds,
            description="artifact deep link to restore review context",
        )
        _expect(
            str(restored_review_state.get("contextRibbonArtifact", "")).strip() == selected_artifact_path,
            f"Artifact deep link should restore the ribbon artifact context: {restored_review_state}",
        )

        if bool(restored_review_state.get("reviewCompareVisible")):
            print("portal-browser-smoke: restoring compare mode from a compare-only deep link", flush=True)
            connection.evaluate(_navigate_to_console_view_expression("build"))
            _poll(
                connection,
                _state_probe_expression(),
                predicate=lambda value: (
                    isinstance(value, dict)
                    and str(value.get("currentView", "")) == "build"
                    and bool(value.get("buildViewVisible"))
                ),
                timeout_seconds=args.timeout_seconds,
                description="build view to restore before compare-only deep-link replay",
            )
            connection.evaluate(_navigate_to_console_view_expression("review", submitted_job_id, None, True))
            compare_only_review_state = _poll(
                connection,
                _state_probe_expression(),
                predicate=lambda value: (
                    isinstance(value, dict)
                    and str(value.get("currentView", "")) == "review"
                    and str(value.get("selectedJobId", "")).strip() == submitted_job_id
                    and bool(value.get("reviewCompareEnabled"))
                    and "compare=1" in str(value.get("locationSearch", ""))
                ),
                timeout_seconds=args.timeout_seconds,
                description="compare-only deep link to preserve compare mode",
            )
            _expect(
                str(compare_only_review_state.get("contextRibbonCompare", "")).strip().lower() == "compare on",
                f"Compare-only deep links should preserve compare mode for the default artifact: {compare_only_review_state}",
            )

        print("portal-browser-smoke: normalizing stale review deep-link params", flush=True)
        connection.evaluate(
            _navigate_to_console_view_expression("review", submitted_job_id, "missing/stale-artifact.png", True)
        )
        normalized_review_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("currentView", "")) == "review"
                and str(value.get("selectedJobId", "")).strip() == submitted_job_id
                and str(value.get("reviewProvenancePath", "")).strip() != "missing/stale-artifact.png"
                and "missing%2Fstale-artifact.png" not in str(value.get("locationSearch", ""))
                and "compare=1" not in str(value.get("locationSearch", ""))
            ),
            timeout_seconds=args.timeout_seconds,
            description="stale artifact and compare params to normalize",
        )
        _expect(
            str(normalized_review_state.get("contextRibbonCompare", "")).strip().lower() != "compare on",
            f"Invalid compare deep links should fall back to a valid single-view review state: {normalized_review_state}",
        )

        print("portal-browser-smoke: simulating degraded auth recovery controls", flush=True)
        degraded_state = connection.evaluate(_simulate_bootstrap_degraded_expression(reason="auth_failure", http_status=401))
        _expect(isinstance(degraded_state, dict), f"Unexpected degraded portal state: {degraded_state!r}")
        _expect(
            str(degraded_state.get("actionPrimaryKey", "")).strip() == "restore_access",
            f"Degraded auth state should promote Restore Access in the action rail: {degraded_state}",
        )
        _expect(
            str(degraded_state.get("actionSecondary1Key", "")).strip() == "retry_status_check",
            f"Degraded auth state should surface Retry Status Check in the action rail: {degraded_state}",
        )
        _expect(
            str(degraded_state.get("selectedRecoveryPrimaryKey", "")).strip() == "restore_access"
            and str(degraded_state.get("reviewStatusPrimaryKey", "")).strip() == "restore_access",
            f"Inspector and review recovery controls should stay aligned with degraded auth recovery: {degraded_state}",
        )
        connection.evaluate(_click_expression("#consoleActionSecondaryBtn1"))
        recovered_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("bootstrapStatus", "")).strip().lower() == "ready"
                and str(value.get("currentView", "")).strip() == "review"
                and str(value.get("selectedJobId", "")).strip() == submitted_job_id
            ),
            timeout_seconds=args.timeout_seconds,
            description="retry status check to recover degraded bootstrap state",
        )
        _expect(
            str(recovered_state.get("actionPrimaryKey", "")).strip() != "restore_access",
            f"Retry Status Check should restore the contextual action rail once bootstrap recovers: {recovered_state}",
        )

        print("portal-browser-smoke: round-tripping through build and back to operate", flush=True)
        connection.evaluate(_navigate_to_console_view_expression("build"))
        _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("currentView", "")) == "build"
                and bool(value.get("buildViewVisible"))
            ),
            timeout_seconds=args.timeout_seconds,
            description="build view to restore after review",
        )
        connection.evaluate(_navigate_to_console_view_expression("operate"))
        restored_operate_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("currentView", "")) == "operate"
                and str(value.get("selectedJobId", "")).strip() == submitted_job_id
                and f"view=operate&job={submitted_job_id}" in str(value.get("locationSearch", ""))
            ),
            timeout_seconds=args.timeout_seconds,
            description="operate view to restore the last selected job",
        )
        _expect(
            str(restored_operate_state.get("selectedJobSummary", "")).strip() != "",
            f"Operate view should keep selected-job summary context visible after returning from build: {restored_operate_state}",
        )

        print("portal-browser-smoke: ok")
        print(f"base_url: {base_url}")
        print(f"job_id: {submitted_job_id}")
        if cleanup_output_dir:
            print(f"output_dir_cleaned: {output_dir}")
        else:
            print(f"output_dir: {output_dir}")
        print(f"state: {terminal_state.get('selectedJobState')}")
        print(f"artifacts: {terminal_state.get('selectedJobArtifactCount')}")
        print(f"health: {terminal_state.get('healthText')}")
        return 0
    except SmokeFailure as exc:
        if runtime_handle is not None:
            log_path = getattr(runtime_handle, "log_path", None)
            log_tail = _tail_text(log_path, max_chars=2400, max_bytes=8192) if isinstance(log_path, Path) else ""
            if log_tail:
                raise SmokeFailure(f"{exc}\nbackend-log-tail:\n{log_tail}", kind=exc.kind) from exc
        raise
    finally:
        if connection is not None:
            connection.close()
        if chrome_process is not None:
            try:
                chrome_process.terminate()
                chrome_process.wait(timeout=5)
            except Exception:
                try:
                    chrome_process.kill()
                except Exception:
                    pass
        if runtime_handle is not None:
            _terminate_runtime(runtime_handle)
        if profile_dir is not None and not args.keep_profile:
            shutil.rmtree(profile_dir, ignore_errors=True)
        if cleanup_output_dir and output_dir is not None:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmokeFailure as exc:
        print(f"portal-browser-smoke: failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
