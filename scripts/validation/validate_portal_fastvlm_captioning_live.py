#!/usr/bin/env python3
"""Live portal backend validation for FastVLM advisory captioning."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastvlm_runtime_manifest import (
    default_manifest_path,
    load_manifest,
    runtime_root,
    selected_model_roles,
)
from validate_fastvlm_runtime import build_runtime_evidence
from validate_portal_lux_materials_live import (
    DEFAULT_API_KEY,
    DEFAULT_ORCHESTRATOR_BASE_URL,
    LocalRuntimeHandle,
    SmokeFailure,
    _artifact_items,
    _base_url,
    _default_fixture_image,
    _find_relative_path,
    _latest_path,
    _load_json_file,
    _output_dir_from_job,
    _poll_terminal_job,
    _prepare_input_dir,
    _request_json,
    _request_text,
    _resolve_output_dir,
    _safe_output_path,
    _should_cleanup_output_dir,
    _spawn_local_backend,
    _submit_job,
    _terminate_runtime,
)

TERMINAL_JOB_STATES = {"succeeded", "failed", "canceled", "partial"}
_FASTVLM_RUNTIME_PATH_ENV = (
    "TP_FASTVLM_PYTHON",
    "TP_FASTVLM_MLX_VLM_DIR",
    "TP_FASTVLM_MODEL",
    "TP_FASTVLM_REVIEW_MODEL",
)
_LOCAL_BACKEND_ENV = {
    "TP_ORCHESTRATOR_IN_PROCESS_WORKERS_ENABLED": "1",
    "TP_ORCHESTRATOR_QUEUE_BACKEND": "memory",
    "TP_ORCHESTRATOR_STATE_BACKEND": "memory",
}


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


def _expect_status(status: int, expected: int, context: str, body: Dict[str, Any]) -> None:
    if status != expected:
        raise SmokeFailure(f"{context} returned HTTP {status}, expected {expected}: {json.dumps(body, sort_keys=True)}")


def _validate_runtime_ready(model_role: str, *, expected_base_python: Path) -> None:
    manifest = load_manifest()
    root = runtime_root(manifest)
    roles = selected_model_roles(manifest, models=model_role)
    evidence = build_runtime_evidence(
        manifest_path=default_manifest_path(),
        root=root,
        roles=roles,
        manifest=manifest,
        include_sources=True,
        include_python=True,
        include_import_smoke=True,
        expected_base_python=expected_base_python,
    )
    if evidence["errors"]:
        raise SmokeFailure(
            "FastVLM runtime prerequisites are not ready; "
            f"audited validation reported {evidence['error_count']} error(s)",
            kind="environment",
        )


def _spawn_audited_local_backend(api_key: str, *, timeout_seconds: float) -> LocalRuntimeHandle:
    """Spawn canonical audited paths with no external worker delegation."""

    controlled_names = (*_FASTVLM_RUNTIME_PATH_ENV, *_LOCAL_BACKEND_ENV)
    preserved = {name: os.environ[name] for name in controlled_names if name in os.environ}
    for name in _FASTVLM_RUNTIME_PATH_ENV:
        os.environ.pop(name, None)
    os.environ.update(_LOCAL_BACKEND_ENV)
    try:
        return _spawn_local_backend(api_key, timeout_seconds=timeout_seconds)
    finally:
        for name in controlled_names:
            os.environ.pop(name, None)
        os.environ.update(preserved)


def _build_captioning_payload(
    *,
    input_dir: Path,
    output_dir: Path,
    model_role: str,
    timeout_seconds: int,
) -> Dict[str, Any]:
    return {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "quality_tier": "apex",
            "depth_backend": "da3",
            "depth_device": "cpu",
            "materials_v3": False,
            "enable_segmentation": False,
            "pbr": False,
            "emit_run_card": True,
            "run_card_version": "v2",
            "enable_v2": False,
            "non_commercial_ok": True,
            "emit_master16": True,
            "emit_upscaled16": False,
            "cache_depth": False,
            "save_float_depth": False,
            "vlm_captioning_enabled": True,
            "vlm_captioning_backend": "fastvlm",
            "vlm_captioning_model": model_role,
            "vlm_captioning_proxy_format": "png",
            "vlm_captioning_max_side_px": 960,
            "fastvlm_timeout_seconds": timeout_seconds,
        },
    }


def _preview_captioning_job(base_url: str, *, api_key: str, payload: Dict[str, Any], model_role: str) -> Dict[str, Any]:
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
    errors = data.get("field_errors") or []
    if errors:
        raise SmokeFailure(f"Preview returned field errors: {json.dumps(errors, sort_keys=True)}", kind="contract")

    summary = data.get("captioning_summary")
    if not isinstance(summary, dict):
        raise SmokeFailure(f"Preview missing captioning_summary: {data}", kind="contract")
    expected_summary = {
        "feature_enabled": True,
        "enabled": True,
        "backend": "fastvlm",
        "model": model_role,
        "role": "advisory",
        "used_for_quality_gate": False,
        "runtime_status": "ready",
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise SmokeFailure(
                f"captioning_summary.{key}={summary.get(key)!r}, expected {expected!r}",
                kind="contract",
            )

    for surface_name in ("normalized_args", "execution_args"):
        surface = data.get(surface_name)
        if not isinstance(surface, dict):
            raise SmokeFailure(f"Preview missing {surface_name} object: {data}", kind="contract")
        if surface.get("vlm_captioning_enabled") is not True:
            raise SmokeFailure(f"Preview {surface_name}.vlm_captioning_enabled is not true", kind="contract")
        if surface.get("vlm_captioning_model") != model_role:
            raise SmokeFailure(
                f"Preview {surface_name}.vlm_captioning_model={surface.get('vlm_captioning_model')!r}",
                kind="contract",
            )
        for deprecated_key in ("emit_marketing", "emitMarketing", "emit_report", "emitReport"):
            if deprecated_key in surface:
                raise SmokeFailure(
                    f"Preview {surface_name} retained deprecated key {deprecated_key}",
                    kind="contract",
                )

    tokens = _argv_preview_tokens(data)
    required_pairs = {
        "--vlm-captioning": "on",
        "--vlm-captioning-backend": "fastvlm",
        "--vlm-captioning-model": model_role,
        "--vlm-captioning-proxy-format": "png",
    }
    for flag, value in required_pairs.items():
        if not _tokens_contain_pair(tokens, flag, value):
            raise SmokeFailure(f"Preview argv missing {flag} {value}: {tokens}", kind="contract")
    for deprecated_flag in ("--emit-marketing", "--emit-report"):
        if deprecated_flag in tokens:
            raise SmokeFailure(f"Preview argv retained deprecated flag {deprecated_flag}: {tokens}", kind="contract")
    return data


def _classify_terminal_job_failure(terminal_body: Dict[str, Any]) -> str:
    data = terminal_body.get("data") if isinstance(terminal_body, dict) else {}
    if not isinstance(data, dict):
        return "contract"
    haystack = json.dumps(data, sort_keys=True).lower()
    environment_markers = (
        "modulenotfounderror",
        "importerror",
        "no module named",
        "pytorch not available",
        "depth anything",
        "da3",
        "runtime unavailable",
        "runner executable not found",
        "fastvlm runtime",
        "mlx",
        "mlx_vlm",
        "checkpoint",
        "model download",
    )
    if any(marker in haystack for marker in environment_markers):
        return "environment"
    return "product"


def _ensure_job_succeeded(terminal_body: Dict[str, Any]) -> Dict[str, Any]:
    data = terminal_body.get("data")
    if not isinstance(data, dict):
        raise SmokeFailure(f"Job status returned invalid envelope: {terminal_body}", kind="contract")
    if data.get("state") != "succeeded" or data.get("exit_code") != 0:
        kind = _classify_terminal_job_failure(terminal_body)
        raise SmokeFailure(
            f"FastVLM captioning job did not succeed: state={data.get('state')!r} "
            f"exit_code={data.get('exit_code')!r} error={data.get('error')!r} "
            f"logs_tail={data.get('logs_tail')!r}",
            kind=kind,
        )
    return data


def _validate_sse_replay(base_url: str, *, api_key: str, job_id: str) -> None:
    events_text = _request_text(base_url, f"/v1/jobs/{job_id}/events", api_key=api_key)
    if "event: state" not in events_text:
        raise SmokeFailure(f"SSE replay missing state event for {job_id}: {events_text[:400]!r}", kind="contract")
    if "event: done" not in events_text:
        raise SmokeFailure(f"SSE replay missing done event for {job_id}: {events_text[:400]!r}", kind="contract")


def _validate_captioning_outputs(job_data: Dict[str, Any]) -> Dict[str, Any]:
    items = _artifact_items(job_data)
    output_dir = _output_dir_from_job(job_data)
    sidecar_relative_path = _find_relative_path(items, prefix="captioning/", suffix=".vlm_captioning.sidecar.json")
    raw_relative_path = _find_relative_path(items, prefix="captioning/", suffix=".vlm_captioning.raw.txt")
    proxy_relative_path = _find_relative_path(items, prefix="captioning/", suffix="_proxy.png")
    missing = [
        name
        for name, value in (
            ("sidecar", sidecar_relative_path),
            ("raw", raw_relative_path),
            ("proxy", proxy_relative_path),
        )
        if not value
    ]
    if missing:
        raise SmokeFailure(f"Job artifacts missing FastVLM captioning outputs {missing}: {items}", kind="product")

    sidecar_path = _safe_output_path(output_dir, str(sidecar_relative_path))
    raw_path = _safe_output_path(output_dir, str(raw_relative_path))
    proxy_path = _safe_output_path(output_dir, str(proxy_relative_path))
    sidecar = _load_json_file(sidecar_path)
    captioning = sidecar.get("vlm_captioning")
    if not isinstance(captioning, dict):
        raise SmokeFailure(f"Sidecar missing vlm_captioning object: {sidecar_path}", kind="product")
    expected = {
        "provider": "fastvlm",
        "role": "advisory",
        "used_for_quality_gate": False,
    }
    for key, expected_value in expected.items():
        if captioning.get(key) != expected_value:
            raise SmokeFailure(
                f"Sidecar vlm_captioning.{key}={captioning.get(key)!r}, expected {expected_value!r}",
                kind="product",
            )
    diagnostics = captioning.get("runtime_diagnostics")
    if not isinstance(diagnostics, dict) or diagnostics.get("success") is not True:
        raise SmokeFailure(f"Sidecar runtime diagnostics are not successful: {diagnostics}", kind="product")
    if not raw_path.is_file() or raw_path.stat().st_size <= 0:
        raise SmokeFailure(f"Raw FastVLM caption output is empty: {raw_path}", kind="product")
    if not proxy_path.is_file() or proxy_path.stat().st_size <= 0:
        raise SmokeFailure(f"FastVLM proxy output is empty: {proxy_path}", kind="product")

    run_card_path = _latest_path(path for path in output_dir.glob("run_card_*.json") if not path.name.endswith(".self.json"))
    if run_card_path is None:
        raise SmokeFailure(f"Output directory missing run_card_*.json: {output_dir}", kind="product")
    run_card = _load_json_file(run_card_path)
    status = run_card.get("captioning_status")
    if not isinstance(status, dict):
        raise SmokeFailure("Run card missing captioning_status", kind="product")
    if status.get("role") != "advisory" or status.get("used_for_quality_gate") is not False:
        raise SmokeFailure(f"Run card captioning_status is not advisory-only: {status}", kind="product")
    if int(status.get("sidecar_count") or 0) < 1:
        raise SmokeFailure(f"Run card captioning_status missing sidecar count: {status}", kind="product")
    return {
        "output_dir": str(output_dir),
        "sidecar_relative_path": sidecar_relative_path,
        "raw_relative_path": raw_relative_path,
        "proxy_relative_path": proxy_relative_path,
        "run_card_path": str(run_card_path),
        "captioning_status": status,
    }


def _run_validation(
    base_url: str,
    *,
    api_key: str,
    input_dir: Path,
    output_dir: Path,
    model_role: str,
    fastvlm_timeout_seconds: int,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> Dict[str, Any]:
    payload = _build_captioning_payload(
        input_dir=input_dir,
        output_dir=output_dir,
        model_role=model_role,
        timeout_seconds=fastvlm_timeout_seconds,
    )
    _preview_captioning_job(base_url, api_key=api_key, payload=payload, model_role=model_role)
    job_id = _submit_job(base_url, api_key=api_key, payload=payload)
    terminal = _poll_terminal_job(
        base_url,
        api_key=api_key,
        job_id=job_id,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    job_data = _ensure_job_succeeded(terminal)
    _validate_sse_replay(base_url, api_key=api_key, job_id=job_id)
    return {"job_id": job_id, **_validate_captioning_outputs(job_data)}


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
        "--skip-local-runtime-check",
        action="store_true",
        help="Skip local manifest/runtime checks. Intended for validating an externally managed backend.",
    )
    parser.add_argument(
        "--require-local-runtime-check",
        action="store_true",
        help="Run local manifest/runtime checks even when --no-spawn-local-backend is used.",
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
        "--model-role",
        default=os.getenv("TP_PORTAL_FASTVLM_CAPTIONING_LIVE_MODEL", "smoke"),
        help="Manifest model role for live captioning validation (default: %(default)s)",
    )
    parser.add_argument(
        "--base-python",
        default=os.getenv("TP_FASTVLM_BASE_PYTHON", sys.executable),
        help="Caller-trusted interpreter that built the local FastVLM runtime venv (default: current Python)",
    )
    parser.add_argument(
        "--fastvlm-timeout-seconds",
        type=int,
        default=180,
        help="Timeout passed to FastVLM subprocesses (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=420.0,
        help="Max time to wait for terminal Lux job state (default: %(default)s)",
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
    args = parser.parse_args(argv)
    if args.skip_local_runtime_check and args.spawn_local_backend:
        parser.error("--skip-local-runtime-check requires --no-spawn-local-backend")
    if args.skip_local_runtime_check and args.require_local_runtime_check:
        parser.error("--skip-local-runtime-check cannot be combined with --require-local-runtime-check")
    return args


def _should_validate_local_runtime(
    *,
    spawn_local_backend: bool,
    skip_local_runtime_check: bool,
    require_local_runtime_check: bool,
) -> bool:
    if spawn_local_backend:
        return True
    if skip_local_runtime_check:
        return False
    return bool(require_local_runtime_check)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    backend_runtime: Optional[LocalRuntimeHandle] = None
    input_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    input_dir_is_temp = False
    output_dir_is_temp = False

    os.environ.setdefault("TP_PORTAL_FASTVLM_CAPTIONING_ENABLED", "1")
    os.environ.setdefault("TP_PORTAL_FASTVLM_CAPTIONING_ROLLOUT_PERCENT", "100")

    try:
        if _should_validate_local_runtime(
            spawn_local_backend=bool(args.spawn_local_backend),
            skip_local_runtime_check=bool(args.skip_local_runtime_check),
            require_local_runtime_check=bool(args.require_local_runtime_check),
        ):
            _validate_runtime_ready(
                str(args.model_role),
                expected_base_python=Path(args.base_python),
            )
        fixture_image = Path(args.fixture_image).resolve()
        input_dir, input_dir_is_temp = _prepare_input_dir(str(args.input_dir), fixture_image)
        output_dir, output_dir_is_temp = _resolve_output_dir(str(args.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.spawn_local_backend:
            backend_runtime = _spawn_audited_local_backend(
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

        result = _run_validation(
            base_url,
            api_key=str(args.api_key),
            input_dir=input_dir,
            output_dir=output_dir,
            model_role=str(args.model_role),
            fastvlm_timeout_seconds=int(args.fastvlm_timeout_seconds),
            timeout_seconds=float(args.timeout_seconds),
            poll_interval_seconds=float(args.poll_interval_seconds),
        )
        print(json.dumps({"status": "ok", **result}, indent=2, sort_keys=True))
        return 0
    except SmokeFailure as exc:
        print(f"FastVLM portal captioning live validation failed ({exc.kind}): {exc}", file=sys.stderr)
        return 2 if exc.kind == "environment" else 1
    finally:
        if backend_runtime is not None:
            _terminate_runtime(backend_runtime)
        if input_dir is not None and input_dir_is_temp and not args.keep_output:
            shutil.rmtree(input_dir, ignore_errors=True)
        if output_dir is not None and _should_cleanup_output_dir(
            keep_output=bool(args.keep_output),
            output_dir_is_temp=output_dir_is_temp,
        ):
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
