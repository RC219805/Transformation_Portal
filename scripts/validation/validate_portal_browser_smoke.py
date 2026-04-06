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
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


class SmokeFailure(RuntimeError):
    """Raised when the browser smoke validation fails."""


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
    kwargs: Dict[str, Any] = {"prefix": "tp-portal-browser-smoke-"}
    if os.name != "nt" and Path("/tmp").exists():
        kwargs["dir"] = "/tmp"
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


def _http_get_json(url: str, timeout: float = 10.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _request_json(base_url: str, path: str, *, api_key: str = "") -> tuple[int, Dict[str, Any]]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    request = urllib.request.Request(
        _base_url(base_url) + path,
        headers=headers,
        method="GET",
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
        raise SmokeFailure(f"GET {path} request failed: {reason}") from exc

    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise SmokeFailure(f"GET {path} returned non-JSON response: {raw_body[:400]!r}") from exc
    return status, body


def _list_job_ids(base_url: str, api_key: str) -> list[str]:
    status, body = _request_json(base_url, "/v1/jobs", api_key=api_key)
    if status != 200:
        raise SmokeFailure(f"GET /v1/jobs returned {status}: {body}")
    jobs = body.get("data", {}).get("jobs", [])
    if not isinstance(jobs, list):
        raise SmokeFailure(f"GET /v1/jobs returned unexpected payload: {body}")
    return [str(job.get("id") or "").strip() for job in jobs if str(job.get("id") or "").strip()]


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
        last_value = connection.evaluate(expression)
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
  return {
    title: document.title,
    readyState: document.readyState,
    bootstrapStatus: document.body ? String(document.body.dataset.bootstrapStatus || '') : '',
    currentView: document.body ? String(document.body.dataset.consoleView || '') : '',
    pipeline: value('pipelineSelect'),
    healthText: text('healthText'),
    heroReadinessLabel: text('heroReadinessLabel'),
    queueCount: text('queueCount'),
    selectedJobState: text('selectedJobStateBadge'),
    selectedJobId: text('selectedJobIdLabel'),
    selectedJobArtifactCount: text('selectedJobArtifactCount'),
    selectedJobStreamStatus: text('selectedJobStreamStatus'),
    selectedJobSummary: text('selectedJobSummary'),
    cliFirstLine: (() => {
      const preview = text('cliPreview');
      return preview ? preview.split('\n')[0].trim() : '';
    })(),
    cliText: text('cliPreview'),
    archiveCanonicalCommand: text('archiveCanonicalCommand'),
    archiveIndexPath: value('archiveIndexPath'),
    rightsManifestPath: value('rightsManifestPath'),
    archiveIndexFieldVisible: (() => {
      const el = document.getElementById('archiveIndexField');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    rightsManifestFieldVisible: (() => {
      const el = document.getElementById('rightsManifestField');
      return !!(el && !el.classList.contains('hidden'));
    })(),
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
      const el = document.getElementById('build-shell');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    operateViewVisible: (() => {
      const el = document.getElementById('jobs-shell');
      return !!(el && !el.classList.contains('hidden'));
    })(),
    overviewViewVisible: (() => {
      const el = document.getElementById('overview-shell');
      return !!(el && !el.classList.contains('hidden'));
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
    })()
  };
})()
"""


def _navigate_to_console_view_expression(view: str, job_id: str = "") -> str:
    payload = json.dumps({"view": view, "job_id": job_id})
    return f"""
(() => {{
  const cfg = {payload};
  const url = new URL(window.location.href);
  url.searchParams.set('view', cfg.view);
  if (cfg.view === 'run' && cfg.job_id) {{
    url.searchParams.set('job', cfg.job_id);
  }} else {{
    url.searchParams.delete('job');
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
) -> str:
    payload = json.dumps(
        {
            "api_key": api_key,
            "pipeline": pipeline,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "archive_index": archive_index,
            "manifest_jsonl": manifest_jsonl,
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
  setValue('apiKeyInput', cfg.api_key);
  setValue('pipelineSelect', cfg.pipeline);
  setValue('inputDir', cfg.input_dir);
  setValue('outputDir', cfg.output_dir);
  setValue('archiveIndexPath', cfg.archive_index);
  setValue('rightsManifestPath', cfg.manifest_jsonl);
  return {{
    pipeline: document.getElementById('pipelineSelect').value,
    archiveFieldsVisible: !document.getElementById('fieldsArchiveGate').classList.contains('hidden'),
    luxFieldsVisible: !document.getElementById('fieldsLuxDepth').classList.contains('hidden'),
    flagsShellVisible: !document.getElementById('flags-shell').classList.contains('hidden'),
    archiveCanonicalCommand: (document.getElementById('archiveCanonicalCommand').textContent || '').trim(),
    archiveIndexFieldVisible: !document.getElementById('archiveIndexField').classList.contains('hidden'),
    rightsManifestFieldVisible: !document.getElementById('rightsManifestField').classList.contains('hidden'),
    archiveIndexPath: document.getElementById('archiveIndexPath').value,
    rightsManifestPath: document.getElementById('rightsManifestPath').value,
    runJobDisabled: !!document.getElementById('runJobBtn').disabled,
    heroReadinessLabel: (document.getElementById('heroReadinessLabel').textContent || '').trim(),
    cliFirstLine: ((document.getElementById('cliPreview').textContent || '').trim().split('\\n')[0] || '').trim(),
    cliText: (document.getElementById('cliPreview').textContent || '').trim()
  }};
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


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("TP_ORCHESTRATOR_BASE_URL", "http://127.0.0.1:8000"),
        help="Portal/backend base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("TP_API_KEY", "local-dev-key"),
        help="API key for protected job endpoints (default: TP_API_KEY or %(default)s)",
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
        help="Optional output directory for the browser-submitted job (defaults to a temp dir)",
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
    base_url = str(args.base_url).strip().rstrip("/")
    if not base_url:
        raise SmokeFailure("Base URL cannot be empty")

    archive_root = Path(args.archive_root).resolve()
    _expect(archive_root.is_dir(), f"Archive root fixture does not exist: {archive_root}")
    archive_index = Path(args.archive_index).resolve()
    _expect(archive_index.is_file(), f"Archive index fixture does not exist: {archive_index}")

    output_dir, output_dir_is_temp = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = _default_profile_dir()
    port = int(args.debugging_port or _find_free_port())
    chrome_binary = _resolve_chrome_binary(args.chrome_binary)
    _expect(Path(chrome_binary).exists(), f"Chrome binary does not exist: {chrome_binary}")
    cleanup_output_dir = _should_cleanup_output_dir(
        keep_output=bool(args.keep_output),
        output_dir_is_temp=output_dir_is_temp,
    )

    chrome_process: Optional[subprocess.Popen[str]] = None
    connection: Optional[DevToolsConnection] = None

    try:
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

        print("portal-browser-smoke: confirming lux-depth-v3 build defaults", flush=True)
        lux_state = connection.evaluate(
            _set_pipeline_form_expression(
                api_key=args.api_key,
                pipeline="lux-depth-v3",
                input_dir=str(archive_root),
                output_dir=str(output_dir),
            )
        )
        _expect(isinstance(lux_state, dict), f"Unexpected lux portal state: {lux_state!r}")
        _expect(lux_state.get("pipeline") == "lux-depth-v3", f"Lux pipeline did not remain selected: {lux_state}")
        _expect(not bool(lux_state.get("archiveFieldsVisible")), f"Lux build view should hide archive controls: {lux_state}")
        _expect(bool(lux_state.get("flagsShellVisible")), f"Lux build view should keep core flags visible: {lux_state}")
        _expect(
            bool(str(lux_state.get("heroReadinessLabel", "")).strip()),
            f"Lux build view did not expose any readiness label: {lux_state}",
        )
        _expect(
            not bool(lux_state.get("runJobDisabled")),
            f"Lux pipeline should remain dispatchable in the build view: {lux_state}",
        )

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

        print("portal-browser-smoke: configuring archive-gate-a form", flush=True)
        configured_state = _poll(
            connection,
            _set_pipeline_form_expression(
                api_key=args.api_key,
                pipeline="archive-gate-a",
                input_dir=str(archive_root),
                output_dir=str(output_dir),
                archive_index=str(archive_index),
            ),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("pipeline", "")) == "archive-gate-a"
                and str(value.get("archiveCanonicalCommand", "")) == "fixity-scan"
                and not bool(value.get("runJobDisabled"))
            ),
            timeout_seconds=args.timeout_seconds,
            description="archive-gate-a build state",
        )
        _expect(
            configured_state.get("pipeline") == "archive-gate-a",
            f"Pipeline switch to archive-gate-a failed: {configured_state}",
        )
        _expect(
            bool(configured_state.get("archiveFieldsVisible")) and not bool(configured_state.get("luxFieldsVisible")),
            f"Archive-specific UI did not toggle correctly: {configured_state}",
        )
        _expect(
            not bool(configured_state.get("flagsShellVisible")),
            f"archive-gate-a should hide Lux-only core flags: {configured_state}",
        )
        _expect(
            bool(configured_state.get("archiveIndexFieldVisible"))
            and not bool(configured_state.get("rightsManifestFieldVisible")),
            f"archive-gate-a should expose only the archive index input: {configured_state}",
        )
        _expect(
            str(configured_state.get("archiveIndexPath", "")) == str(archive_index),
            f"Archive index field did not update for archive-gate-a: {configured_state}",
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
                and str(value.get("selectedJobState", "")).strip().lower() == "succeeded"
                and "3 indexed" in str(value.get("selectedJobArtifactCount", ""))
                and bool(value.get("logHasFixityWrite"))
            ),
            timeout_seconds=args.timeout_seconds,
            description="browser-submitted job to succeed with artifacts and logs",
        )

        _expect(
            str(terminal_state.get("selectedJobId") or "").strip() == submitted_job_id,
            f"Portal inspector drifted from submitted job {submitted_job_id}: {terminal_state}",
        )
        _expect(
            "Closed" in str(terminal_state.get("selectedJobStreamStatus", ""))
            or "Inactive" not in str(terminal_state.get("selectedJobStreamStatus", "")),
            f"Unexpected stream status after completion: {terminal_state}",
        )

        print("portal-browser-smoke: opening run details view", flush=True)
        connection.evaluate(_navigate_to_console_view_expression("run", submitted_job_id))
        run_state = _poll(
            connection,
            _state_probe_expression(),
            predicate=lambda value: (
                isinstance(value, dict)
                and str(value.get("currentView", "")) == "run"
                and str(value.get("selectedJobId", "")) == submitted_job_id
                and bool(value.get("queueShellHidden"))
            ),
            timeout_seconds=args.timeout_seconds,
            description="run details view to become active",
        )
        _expect(
            "3 indexed" in str(run_state.get("selectedJobArtifactCount", "")),
            f"Run details view lost artifact context: {run_state}",
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
        if not args.keep_profile:
            shutil.rmtree(profile_dir, ignore_errors=True)
        if cleanup_output_dir:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmokeFailure as exc:
        print(f"portal-browser-smoke: failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
