#!/usr/bin/env python3
"""
Live HTTP smoke validation for the portal orchestrator backend.

This script validates the running FastAPI service over HTTP using a safe
archive-gate job built from checked-in fixture data. It exercises:

1. `GET /ready`
2. `GET /v1/presets?pipeline=lux-depth-v3`
3. Optional auth posture validation for `GET /v1/jobs`
4. Authenticated `GET /v1/jobs`
5. Safe `POST /v1/jobs` submission for `archive-gate-a` `fixity-scan`
6. Polling `GET /v1/jobs/{job_id}` until completion
7. `GET /v1/jobs/{job_id}/events` SSE replay for the completed job

Run via:
    python scripts/validation/validate_orchestrator_http_smoke.py
    make validate-orchestrator-http

Environment overrides:
    TP_ORCHESTRATOR_BASE_URL  Backend URL (default: http://127.0.0.1:8000)
    TP_API_KEY                API key for protected job endpoints
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Tuple


class SmokeFailure(RuntimeError):
    """Raised when the live HTTP smoke validation fails."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_fixture_archive_root() -> Path:
    return _repo_root() / "tests" / "fixtures" / "archive_small" / "archive_root"


def _default_fixture_archive_index() -> Path:
    return _repo_root() / "tests" / "fixtures" / "archive_small" / "archive_index_normalized.csv.gz"


def _default_output_dir() -> Path:
    kwargs: Dict[str, Any] = {"prefix": "tp-orchestrator-http-smoke-"}
    if os.name != "nt" and Path("/tmp").exists():
        kwargs["dir"] = "/tmp"
    return Path(tempfile.mkdtemp(**kwargs))


def _base_url(value: str) -> str:
    trimmed = value.strip()
    if not trimmed:
        raise SmokeFailure("Base URL cannot be empty")
    return trimmed.rstrip("/")


def _request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    api_key: str = "",
    payload: Dict[str, Any] | None = None,
) -> Tuple[int, Dict[str, Any]]:
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
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
            raw_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw_body = exc.read().decode("utf-8")

    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise SmokeFailure(f"{method} {path} returned non-JSON response: {raw_body[:400]!r}") from exc
    return status, body


def _request_text(
    base_url: str,
    path: str,
    *,
    api_key: str = "",
) -> str:
    headers = {"Accept": "text/event-stream"}
    if api_key:
        headers["x-api-key"] = api_key

    request = urllib.request.Request(
        _base_url(base_url) + path,
        headers=headers,
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def _expect_status(status: int, expected: int, context: str, body: Dict[str, Any]) -> None:
    if status != expected:
        raise SmokeFailure(f"{context} returned HTTP {status}, expected {expected}: {json.dumps(body, sort_keys=True)}")


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
        if state in {"succeeded", "failed", "canceled"}:
            return body
        time.sleep(poll_interval_seconds)
    raise SmokeFailure(f"Job {job_id} did not reach a terminal state within {timeout_seconds:.1f}s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("TP_ORCHESTRATOR_BASE_URL", "http://127.0.0.1:8000"),
        help="Running backend base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("TP_API_KEY", "local-dev-key"),
        help="API key for protected job endpoints (default: TP_API_KEY or %(default)s)",
    )
    parser.add_argument(
        "--archive-root",
        default=str(_default_fixture_archive_root()),
        help="Archive root for the safe archive-gate fixture job",
    )
    parser.add_argument(
        "--archive-index",
        default=str(_default_fixture_archive_index()),
        help="Archive index for the safe archive-gate fixture job",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory for the smoke job (defaults to a temp dir)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=45.0,
        help="Max time to wait for terminal job state (default: %(default)s)",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=1.0,
        help="Polling interval for job status checks (default: %(default)s)",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Preserve the smoke job output directory instead of deleting it",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    base_url = _base_url(args.base_url)
    archive_root = Path(args.archive_root).resolve()
    archive_index = Path(args.archive_index).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else _default_output_dir()

    _expect(archive_root.is_dir(), f"Archive root fixture does not exist: {archive_root}")
    _expect(archive_index.is_file(), f"Archive index fixture does not exist: {archive_index}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        ready_status, ready_body = _request_json(base_url, "/ready")
        _expect_status(ready_status, 200, "GET /ready", ready_body)
        _expect(ready_body.get("ok") is True, f"/ready response missing ok=true: {ready_body}")

        presets_status, presets_body = _request_json(
            base_url,
            "/v1/presets?pipeline=lux-depth-v3",
        )
        _expect_status(
            presets_status,
            200,
            "GET /v1/presets?pipeline=lux-depth-v3",
            presets_body,
        )
        _expect(
            presets_body.get("success") is True,
            f"Presets response did not report success: {presets_body}",
        )

        ready_security = ready_body.get("security") if isinstance(ready_body.get("security"), dict) else {}
        auth_expected = bool(ready_security.get("api_key_enforced_for_jobs"))
        if auth_expected:
            unauth_status, unauth_body = _request_json(base_url, "/v1/jobs")
            _expect_status(unauth_status, 401, "GET /v1/jobs without API key", unauth_body)
            _expect(
                ((unauth_body.get("error") or {}).get("code") == "UNAUTHORIZED"),
                f"Unexpected unauthenticated /v1/jobs error envelope: {unauth_body}",
            )

        _expect(args.api_key or not auth_expected, "API key required for protected job endpoints")

        jobs_status, jobs_body = _request_json(base_url, "/v1/jobs", api_key=args.api_key)
        _expect_status(jobs_status, 200, "GET /v1/jobs", jobs_body)
        _expect(
            jobs_body.get("success") is True,
            f"Authenticated jobs list did not report success: {jobs_body}",
        )

        payload = {
            "pipeline": "archive-gate-a",
            "args": {
                "input_dir": str(archive_root),
                "output_dir": str(output_dir),
                "archive_command": "fixity-scan",
                "archive_index": str(archive_index),
                "archive_root": str(archive_root),
                "out_dir": str(output_dir),
                "workers": 1,
                "validate_schemas": False,
            },
        }
        create_status, create_body = _request_json(
            base_url,
            "/v1/jobs",
            method="POST",
            api_key=args.api_key,
            payload=payload,
        )
        _expect_status(create_status, 200, "POST /v1/jobs", create_body)
        _expect(
            create_body.get("success") is True,
            f"Job creation did not report success: {create_body}",
        )
        job_id = ((create_body.get("data") or {}).get("id") or "").strip()
        _expect(job_id.startswith("job_"), f"Job creation returned invalid job id: {create_body}")

        terminal_body = _poll_terminal_job(
            base_url,
            api_key=args.api_key,
            job_id=job_id,
            timeout_seconds=args.timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
        terminal_data = terminal_body.get("data") or {}
        _expect(
            terminal_data.get("state") == "succeeded",
            f"Smoke job did not succeed: {terminal_body}",
        )
        _expect(
            terminal_data.get("exit_code") == 0,
            f"Smoke job exit_code was not zero: {terminal_body}",
        )

        artifacts = terminal_data.get("artifacts") or {}
        artifact_names = {
            str(item.get("relative_path") or "") for item in (artifacts.get("items") or []) if isinstance(item, dict)
        }
        expected_artifacts = {"hash_manifest.csv.gz", "hash_summary.json", "merkle_roots.json"}
        missing_artifacts = sorted(expected_artifacts - artifact_names)
        _expect(
            not missing_artifacts,
            f"Smoke job artifacts missing expected files {missing_artifacts}: {terminal_body}",
        )

        events_text = _request_text(
            base_url,
            f"/v1/jobs/{job_id}/events",
            api_key=args.api_key,
        )
        _expect("event: state" in events_text, f"SSE replay missing state event: {events_text}")
        _expect("event: done" in events_text, f"SSE replay missing done event: {events_text}")

        print("orchestrator-http-smoke: ok")
        print(f"base_url: {base_url}")
        print(f"job_id: {job_id}")
        if args.keep_output:
            print(f"output_dir: {output_dir}")
        else:
            print(f"output_dir_cleaned: {output_dir}")
        print(f"artifacts: {', '.join(sorted(artifact_names))}")
        print(f"auth_expected: {'yes' if auth_expected else 'no'}")
        return 0
    finally:
        if not args.keep_output:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmokeFailure as exc:
        print(f"orchestrator-http-smoke: failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
