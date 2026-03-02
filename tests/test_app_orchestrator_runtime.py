#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for root FastAPI orchestrator app runtime behavior."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import pytest
from starlette.requests import Request as StarletteRequest

orchestrator_app = importlib.import_module("app")


class _FakeRequest:
    """Lightweight request stub for SSE generator tests."""

    def __init__(self) -> None:
        self.disconnected = False

    async def is_disconnected(self) -> bool:
        return self.disconnected


def _flag_value(argv: list[str], flag: str) -> str:
    idx = argv.index(flag)
    return argv[idx + 1]


def _extract_js_function_body(content: str, function_name: str) -> str:
    marker = f"function {function_name}("
    start = content.find(marker)
    if start < 0:
        raise AssertionError(f"{function_name} not found")
    brace_start = content.find("{", start)
    if brace_start < 0:
        raise AssertionError(f"{function_name} opening brace not found")

    depth = 0
    for idx in range(brace_start, len(content)):
        char = content[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[brace_start + 1 : idx]
    raise AssertionError(f"{function_name} closing brace not found")


def _extract_portal_canonical_lux_arg_keys(content: str) -> set[str]:
    body = _extract_js_function_body(content, "buildCanonicalLuxDepthArgs")
    args_match = re.search(r"const args = \{(.*?)\n\s*\};", body, flags=re.DOTALL)
    if args_match is None:
        raise AssertionError("buildCanonicalLuxDepthArgs args object not found")
    args_body = args_match.group(1)
    explicit_keys = set(re.findall(r"^\s*([a-z_][a-z0-9_]*)\s*:", args_body, flags=re.MULTILINE))
    shorthand_keys = set(re.findall(r"^\s*([a-z_][a-z0-9_]*)\s*,?\s*$", args_body, flags=re.MULTILINE))
    conditional_keys = set(re.findall(r"\bargs\.([a-z_][a-z0-9_]*)\s*=", body))
    return explicit_keys | shorthand_keys | conditional_keys


def _extract_portal_lux_cli_flags(content: str) -> set[str]:
    body = _extract_js_function_body(content, "renderCLI")
    lux_block = re.search(
        r"if \(payload\.pipeline === 'lux-depth-v3'\) \{(.*?)\n\s*\} else \{",
        body,
        flags=re.DOTALL,
    )
    if lux_block is None:
        raise AssertionError("renderCLI lux-depth-v3 block not found")
    return set(re.findall(r"--[a-z0-9-]+", lux_block.group(1)))


def _capture_lux_cli_config_from_args(cli_args: list[str]) -> object:
    from unittest.mock import MagicMock, patch

    from typer.testing import CliRunner

    from transformation_portal.lux_depth_v3.__main__ import app as lux_cli_app

    captured: Dict[str, object] = {}

    def _mock_orchestrator_init(config, output_root):
        del output_root
        captured["config"] = config
        mock_orchestrator = MagicMock()
        mock_orchestrator.enhance_batch.return_value = [{"status": "ok"}]
        return mock_orchestrator

    with patch(
        "transformation_portal.lux_depth_v3.__main__.EnhanceOrchestrator",
        side_effect=_mock_orchestrator_init,
    ):
        result = CliRunner().invoke(lux_cli_app, cli_args)
    assert result.exit_code == 0, result.stdout
    assert "config" in captured
    return captured["config"]


def _run_card_fingerprint_from_config(config: object) -> Dict[str, object]:
    from transformation_portal.lux_depth_v3.config import ModelVariant
    from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

    if getattr(config, "model_variant", None) is None:
        setattr(config, "model_variant", ModelVariant.METRIC_LARGE)

    orch = object.__new__(EnhanceOrchestrator)
    orch.config = config
    requested_backend = getattr(config, "depth_backend", None) or "da3"
    resolved_backend = (
        "da3"
        if str(requested_backend).strip().lower() in {"depth-anything-v3", "depth_anything_v3"}
        else str(requested_backend).strip().lower()
    )
    orch._backend_metadata = SimpleNamespace(
        requested_backend=str(requested_backend).strip().lower(),
        resolved_backend=resolved_backend,
        device=getattr(config, "depth_device", "cpu"),
    )
    orch._is_apex_tier = lambda: str(getattr(config, "quality_tier", "standard")).strip().lower() == "apex"
    return orch._build_run_card_config_fingerprint()


def _build_request(
    method: str,
    path: str,
    headers: Dict[str, str] | None = None,
    client_host: str = "127.0.0.1",
    query_string: str = "",
) -> StarletteRequest:
    raw_headers = []
    for key, value in (headers or {}).items():
        raw_headers.append((key.lower().encode("latin-1"), value.encode("latin-1")))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": query_string.encode("utf-8"),
        "headers": raw_headers,
        "client": (client_host, 12345),
        "server": ("testserver", 80),
        "scheme": "http",
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return StarletteRequest(scope, receive)


@pytest.fixture(autouse=True)
def _reset_global_state() -> None:
    orchestrator_app.JOBS.clear()
    orchestrator_app.EVENT_SUBSCRIBERS.clear()
    yield
    orchestrator_app.JOBS.clear()
    orchestrator_app.EVENT_SUBSCRIBERS.clear()


def test_argv_normalization_accepts_canonical_keys() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output",
            "preset": "premium",
            "quality_tier": "apex",
            "depth_backend": "da3",
            "depth_device": "cuda",
            "materials_v3": True,
            "pbr": True,
            "cache_depth": True,
            "emit_master16": True,
            "emit_upscaled16": True,
            "emit_marketing": False,
            "emit_report": True,
            "emit_run_card": True,
        },
    }

    argv = orchestrator_app._argv_from_request(payload)

    assert _flag_value(argv, "--materials-v3") == "on"
    assert _flag_value(argv, "--cache-depth") == "on"
    assert _flag_value(argv, "--depth-backend") == "da3"
    assert _flag_value(argv, "--depth-device") == "cuda"
    assert _flag_value(argv, "--emit-report") == "on"


def test_argv_normalization_accepts_legacy_keys() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output",
            "quality_tier": "standard",
            "depth_backend": "depth_anything_v3",
            "depthDevice": "cpu",
            "materials": True,
            "cacheDepth": True,
            "pbr": False,
            "enableV2": True,
            "v2Preset": "default",
        },
    }

    argv = orchestrator_app._argv_from_request(payload)

    assert _flag_value(argv, "--materials-v3") == "on"
    assert _flag_value(argv, "--cache-depth") == "on"
    assert _flag_value(argv, "--depth-backend") == "da3"
    assert _flag_value(argv, "--depth-device") == "cpu"
    assert _flag_value(argv, "--enable-v2") == "on"
    assert _flag_value(argv, "--v2-preset") == "default"


def test_argv_normalization_honors_explicit_v2_disable() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output",
            "enable_v2": False,
        },
    }

    argv = orchestrator_app._argv_from_request(payload)

    assert _flag_value(argv, "--enable-v2") == "off"
    assert "--v2-preset" not in argv


def test_argv_normalization_depth_backend_is_case_insensitive() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output",
            "depth_backend": "Depth-Anything-V3",
        },
    }

    argv = orchestrator_app._argv_from_request(payload)
    assert _flag_value(argv, "--depth-backend") == "da3"


def test_argv_normalization_trims_and_normalizes_string_values() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output",
            "preset": " premium ",
            "quality_tier": " APEX ",
            "depth_backend": " Depth_Pro ",
            "depth_device": "  cpu  ",
            "materials_v3": "on",
            "pbr": "false",
            "cache_depth": "off",
            "enable_v2": "off",
            "emit_master16": "true",
            "emit_upscaled16": "1",
            "emit_marketing": "0",
            "emit_report": "yes",
            "emit_run_card": "no",
        },
    }

    argv = orchestrator_app._argv_from_request(payload)

    assert _flag_value(argv, "--preset") == "premium"
    assert _flag_value(argv, "--quality-tier") == "apex"
    assert _flag_value(argv, "--depth-backend") == "depth_pro"
    assert _flag_value(argv, "--depth-device") == "cpu"
    assert _flag_value(argv, "--materials-v3") == "on"
    assert _flag_value(argv, "--pbr") == "off"
    assert _flag_value(argv, "--cache-depth") == "off"
    assert _flag_value(argv, "--enable-v2") == "off"
    assert "--v2-preset" not in argv
    assert _flag_value(argv, "--emit-marketing") == "off"
    assert _flag_value(argv, "--emit-report") == "on"
    assert _flag_value(argv, "--emit-run-card") == "off"


def test_portal_cli_template_excludes_unsupported_lux_flags() -> None:
    portal_html = Path(__file__).resolve().parents[1] / "portal.html"
    content = portal_html.read_text(encoding="utf-8")
    assert "--emit-manifest" not in content
    assert "--emit-provenance" not in content
    assert "--enable-segmentation" in content
    assert "--segmentation-backend" in content
    assert "--sam2-model-size" in content
    assert "--strict-segmentation" in content


def test_lux_cli_parity_links_portal_canonical_args_and_backend_argv() -> None:
    portal_html = Path(__file__).resolve().parents[1] / "portal.html"
    content = portal_html.read_text(encoding="utf-8")
    canonical_keys = _extract_portal_canonical_lux_arg_keys(content)
    portal_cli_flags = _extract_portal_lux_cli_flags(content)

    arg_to_flag = {
        "preset": "--preset",
        "quality_tier": "--quality-tier",
        "depth_backend": "--depth-backend",
        "depth_device": "--depth-device",
        "enable_segmentation": "--enable-segmentation",
        "segmentation_backend": "--segmentation-backend",
        "sam2_model_size": "--sam2-model-size",
        "strict_segmentation": "--strict-segmentation",
        "materials_v3": "--materials-v3",
        "pbr": "--pbr",
        "cache_depth": "--cache-depth",
        "enable_v2": "--enable-v2",
        "v2_preset": "--v2-preset",
        "emit_master16": "--emit-master16",
        "emit_upscaled16": "--emit-upscaled16",
        "emit_marketing": "--emit-marketing",
        "emit_report": "--emit-report",
        "emit_run_card": "--emit-run-card",
        "non_commercial_ok": "--non-commercial-ok",
        "accept_apple_depth_pro_research_license": "--accept-apple-depth-pro-research-license",
    }

    for key, flag in arg_to_flag.items():
        assert key in canonical_keys, f"portal canonical args missing key '{key}'"
        assert flag in portal_cli_flags, f"portal CLI preview missing flag '{flag}'"

    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output/lux_depth_v3_apex_verify",
            "preset": "depth-anything-v3.1-research-m4",
            "quality_tier": "apex",
            "depth_backend": "depth_pro",
            "depth_device": "cpu",
            "enable_segmentation": True,
            "segmentation_backend": "sam2",
            "sam2_model_size": "large",
            "strict_segmentation": True,
            "materials_v3": True,
            "pbr": True,
            "cache_depth": False,
            "emit_master16": True,
            "emit_upscaled16": False,
            "emit_marketing": False,
            "emit_report": True,
            "emit_run_card": True,
            "enable_v2": True,
            "v2_preset": "default",
            "non_commercial_ok": True,
            "accept_apple_depth_pro_research_license": True,
        },
    }
    argv = orchestrator_app._argv_from_request(payload)

    for flag in arg_to_flag.values():
        assert flag in argv, f"backend argv missing flag '{flag}'"

    assert _flag_value(argv, "--preset") == "depth-anything-v3.1-research-m4"
    assert _flag_value(argv, "--quality-tier") == "apex"
    assert _flag_value(argv, "--depth-backend") == "depth_pro"
    assert _flag_value(argv, "--depth-device") == "cpu"
    assert _flag_value(argv, "--enable-segmentation") == "on"
    assert _flag_value(argv, "--segmentation-backend") == "sam2"
    assert _flag_value(argv, "--sam2-model-size") == "large"
    assert "--strict-segmentation" in argv
    assert _flag_value(argv, "--materials-v3") == "on"
    assert _flag_value(argv, "--pbr") == "on"
    assert _flag_value(argv, "--cache-depth") == "off"
    assert _flag_value(argv, "--emit-master16") == "on"
    assert _flag_value(argv, "--emit-upscaled16") == "off"
    assert _flag_value(argv, "--emit-marketing") == "off"
    assert _flag_value(argv, "--emit-report") == "on"
    assert _flag_value(argv, "--emit-run-card") == "on"
    assert _flag_value(argv, "--enable-v2") == "on"
    assert _flag_value(argv, "--v2-preset") == "default"
    assert _flag_value(argv, "--non-commercial-ok") == "true"
    assert _flag_value(argv, "--accept-apple-depth-pro-research-license") == "true"


def test_lux_ui_backend_and_direct_cli_paths_share_config_fingerprint(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "frame.jpg").touch()

    portal_payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": str(input_dir),
            "output_dir": str(tmp_path / "output_ui"),
            "preset": "depth-anything-v3.1-research-m4",
            "quality_tier": "apex",
            "depth_backend": "depth_pro",
            "depth_device": "cpu",
            "enable_segmentation": True,
            "segmentation_backend": "sam2",
            "sam2_model_size": "large",
            "strict_segmentation": True,
            "materials_v3": True,
            "pbr": True,
            "cache_depth": False,
            "emit_master16": True,
            "emit_upscaled16": False,
            "emit_marketing": False,
            "emit_report": True,
            "emit_run_card": True,
            "enable_v2": True,
            "v2_preset": "default",
            "non_commercial_ok": True,
            "accept_apple_depth_pro_research_license": True,
        },
    }
    portal_argv = orchestrator_app._argv_from_request(portal_payload)
    assert portal_argv[0] == "lux-depth-v3"

    portal_path_config = _capture_lux_cli_config_from_args(portal_argv[1:])
    portal_path_fingerprint = _run_card_fingerprint_from_config(portal_path_config)

    direct_cli_args = [
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(tmp_path / "output_cli"),
        "--preset",
        "depth-anything-v3.1-research-m4",
        "--quality-tier",
        "apex",
        "--depth-backend",
        "depth_pro",
        "--depth-device",
        "cpu",
        "--materials-v3",
        "yes",
        "--enable-segmentation",
        "on",
        "--segmentation-backend",
        "sam2",
        "--sam2-model-size",
        "large",
        "--strict-segmentation",
        "--pbr",
        "true",
        "--cache-depth",
        "0",
        "--emit-master16",
        "1",
        "--emit-upscaled16",
        "off",
        "--emit-marketing",
        "false",
        "--emit-report",
        "on",
        "--emit-run-card",
        "on",
        "--enable-v2",
        "on",
        "--v2-preset",
        "default",
        "--non-commercial-ok",
        "true",
        "--accept-apple-depth-pro-research-license",
        "true",
    ]
    direct_cli_config = _capture_lux_cli_config_from_args(direct_cli_args)
    direct_cli_fingerprint = _run_card_fingerprint_from_config(direct_cli_config)

    assert portal_path_fingerprint["canonical_json"] == direct_cli_fingerprint["canonical_json"]
    assert portal_path_fingerprint["sha256"] == direct_cli_fingerprint["sha256"]


def test_argv_normalization_includes_segmentation_controls() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output",
            "enable_segmentation": True,
            "segmentation_backend": "sam2",
            "sam2_model_size": "large",
            "strict_segmentation": True,
        },
    }

    argv = orchestrator_app._argv_from_request(payload)
    assert _flag_value(argv, "--enable-segmentation") == "on"
    assert _flag_value(argv, "--segmentation-backend") == "sam2"
    assert _flag_value(argv, "--sam2-model-size") == "large"
    assert "--strict-segmentation" in argv


def test_argv_normalization_ignores_sam2_model_size_when_backend_is_not_sam2() -> None:
    payload: Dict[str, object] = {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": "./input_images",
            "output_dir": "./output",
            "segmentation_backend": "stub",
            "sam2_model_size": "tiny",
        },
    }

    argv = orchestrator_app._argv_from_request(payload)
    assert _flag_value(argv, "--segmentation-backend") == "stub"
    assert "--sam2-model-size" not in argv


def test_portal_segmentation_defaults_align_with_cli_defaults() -> None:
    portal_html = Path(__file__).resolve().parents[1] / "portal.html"
    content = portal_html.read_text(encoding="utf-8")
    assert "enable: false," in content
    assert "backend: 'stub'," in content
    assert "sam2ModelSize: 'base'," in content
    assert "strict: false" in content


def test_argv_archive_gate_a_defaults_to_fixity_scan_runner() -> None:
    payload: Dict[str, object] = {
        "pipeline": "archive-gate-a",
        "args": {
            "input_dir": "./archive_root",
            "output_dir": "./archive_reports",
        },
    }

    argv = orchestrator_app._argv_from_request(payload)

    assert argv[0] == sys.executable
    assert argv[1].endswith("tools/archive_governance.py")
    assert argv[2] == "--json"
    assert argv[3] == "fixity-scan"
    assert _flag_value(argv, "--archive-root") == "./archive_root"
    assert _flag_value(argv, "--out-dir") == "./archive_reports"


def test_argv_archive_gate_b_allows_command_override_and_sign_maps_to_bagit_validation() -> None:
    payload: Dict[str, object] = {
        "pipeline": "archive-gate-b",
        "args": {
            "input_dir": "./archive_root",
            "output_dir": "./archive_reports",
            "archive_command": "bag-validate",
            "sign": True,
        },
    }

    argv = orchestrator_app._argv_from_request(payload)

    assert argv[3] == "bag-validate"
    assert _flag_value(argv, "--bag-dir") == "archive_reports/bag"
    assert "--validate-with-bagit-python" in argv


def test_argv_archive_gate_fixity_verify_uses_canonical_default_report_filename() -> None:
    payload: Dict[str, object] = {
        "pipeline": "archive-gate-a",
        "args": {
            "input_dir": "./archive_root",
            "output_dir": "./archive_reports",
            "archive_command": "fixity-verify",
        },
    }

    argv = orchestrator_app._argv_from_request(payload)

    assert argv[3] == "fixity-verify"
    assert _flag_value(argv, "--report-path") == "archive_reports/verification_report.json"


def test_argv_archive_gate_rejects_workers_below_minimum() -> None:
    payload: Dict[str, object] = {
        "pipeline": "archive-gate-a",
        "args": {
            "input_dir": "./archive_root",
            "output_dir": "./archive_reports",
            "archive_command": "fixity-scan",
            "workers": 0,
        },
    }

    with pytest.raises(ValueError, match="Invalid archive integer option"):
        orchestrator_app._argv_from_request(payload)


def test_argv_archive_gate_rejects_negative_verify_sample() -> None:
    payload: Dict[str, object] = {
        "pipeline": "archive-gate-a",
        "args": {
            "input_dir": "./archive_root",
            "output_dir": "./archive_reports",
            "archive_command": "fixity-verify",
            "verify_sample": -1,
        },
    }

    with pytest.raises(ValueError, match="Invalid archive integer option"):
        orchestrator_app._argv_from_request(payload)


def test_argv_archive_gate_invalid_command_is_rejected() -> None:
    payload: Dict[str, object] = {
        "pipeline": "archive-gate-c",
        "args": {
            "input_dir": "./archive_root",
            "output_dir": "./archive_reports",
            "archive_command": "bag-build",
        },
    }

    with pytest.raises(ValueError, match="Invalid archive_command"):
        orchestrator_app._argv_from_request(payload)


def test_run_job_is_async_and_does_not_block_event_loop() -> None:
    async def scenario() -> None:
        job = orchestrator_app.Job(id="job_async", created_at=orchestrator_app._now())
        orchestrator_app.JOBS[job.id] = job
        orchestrator_app.EVENT_SUBSCRIBERS[job.id] = {}

        ticks = 0
        stop = asyncio.Event()

        async def ticker() -> None:
            nonlocal ticks
            while not stop.is_set():
                ticks += 1
                await asyncio.sleep(0.01)

        runner = asyncio.create_task(
            orchestrator_app._run_job(
                job,
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "import sys,time;print('progress=10%');sys.stdout.flush();time.sleep(0.15);"
                    "print('progress=100%');sys.stdout.flush()",
                ],
            )
        )
        ticker_task = asyncio.create_task(ticker())

        await asyncio.wait_for(runner, timeout=5)
        stop.set()
        await asyncio.wait_for(ticker_task, timeout=1)

        assert ticks > 5
        assert job.state == "succeeded"
        assert job.exit_code == 0
        assert job.progress == 100

    asyncio.run(scenario())


def test_cancel_request_terminates_running_job() -> None:
    async def scenario() -> None:
        job = orchestrator_app.Job(id="job_cancel", created_at=orchestrator_app._now())
        orchestrator_app.JOBS[job.id] = job
        orchestrator_app.EVENT_SUBSCRIBERS[job.id] = {}

        runner = asyncio.create_task(
            orchestrator_app._run_job(
                job,
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "import sys,time;print('progress=1%');sys.stdout.flush();time.sleep(30)",
                ],
            )
        )

        for _ in range(200):
            if job.proc is not None:
                break
            await asyncio.sleep(0.01)

        assert job.proc is not None
        await orchestrator_app._request_cancel(job)
        await asyncio.wait_for(runner, timeout=5)

        assert job.cancel_requested is True
        assert job.state == "canceled"

    asyncio.run(scenario())


def test_sse_broadcast_delivers_events_to_multiple_subscribers() -> None:
    async def scenario() -> None:
        job = orchestrator_app.Job(id="job_sse", created_at=orchestrator_app._now(), state="running")
        orchestrator_app.JOBS[job.id] = job
        orchestrator_app.EVENT_SUBSCRIBERS[job.id] = {}

        req_one = _FakeRequest()
        req_two = _FakeRequest()
        response_one = await orchestrator_app.job_events(req_one, job.id)
        response_two = await orchestrator_app.job_events(req_two, job.id)

        async def collect(response) -> str:
            chunks = []
            saw_done = False
            async for chunk in response.body_iterator:
                chunks.append(chunk)
                if "event: done" in chunk:
                    saw_done = True
            assert saw_done is True
            return "".join(chunks)

        task_one = asyncio.create_task(collect(response_one))
        task_two = asyncio.create_task(collect(response_two))

        await asyncio.sleep(0)
        await orchestrator_app._publish_event(job.id, "progress", {"id": job.id, "progress": 42})
        await orchestrator_app._publish_event(job.id, "done", {"id": job.id, "state": "succeeded", "exit_code": 0})

        out_one, out_two = await asyncio.wait_for(asyncio.gather(task_one, task_two), timeout=3)
        assert "event: progress" in out_one
        assert "event: progress" in out_two
        assert "event: done" in out_one
        assert "event: done" in out_two
        assert orchestrator_app.EVENT_SUBSCRIBERS[job.id] == {}

    asyncio.run(scenario())


def test_sse_disconnect_cleans_up_subscriber_queue() -> None:
    async def scenario() -> None:
        job = orchestrator_app.Job(id="job_disconnect", created_at=orchestrator_app._now(), state="running")
        orchestrator_app.JOBS[job.id] = job
        orchestrator_app.EVENT_SUBSCRIBERS[job.id] = {}

        request = _FakeRequest()
        response = await orchestrator_app.job_events(request, job.id)

        async def read_until_disconnect() -> None:
            async for _chunk in response.body_iterator:
                request.disconnected = True

        await asyncio.wait_for(read_until_disconnect(), timeout=3)
        assert orchestrator_app.EVENT_SUBSCRIBERS[job.id] == {}

    asyncio.run(scenario())


def test_cleanup_expired_jobs_prunes_old_finished_entries() -> None:
    now = orchestrator_app._now()
    old_job = orchestrator_app.Job(
        id="job_old",
        created_at=now - 5000,
        finished_at=now - orchestrator_app.JOB_RETENTION_SECONDS - 10,
        state="succeeded",
    )
    fresh_job = orchestrator_app.Job(
        id="job_fresh",
        created_at=now - 120,
        finished_at=now - 30,
        state="succeeded",
    )
    running_job = orchestrator_app.Job(id="job_running", created_at=now, state="running")

    orchestrator_app.JOBS[old_job.id] = old_job
    orchestrator_app.JOBS[fresh_job.id] = fresh_job
    orchestrator_app.JOBS[running_job.id] = running_job

    orchestrator_app.EVENT_SUBSCRIBERS[old_job.id] = {"s1": asyncio.Queue()}
    orchestrator_app.EVENT_SUBSCRIBERS[fresh_job.id] = {"s2": asyncio.Queue()}

    orchestrator_app._cleanup_expired_jobs(now)

    assert old_job.id not in orchestrator_app.JOBS
    assert old_job.id not in orchestrator_app.EVENT_SUBSCRIBERS
    assert fresh_job.id in orchestrator_app.JOBS
    assert running_job.id in orchestrator_app.JOBS


def test_mutating_job_route_detection() -> None:
    assert orchestrator_app._is_mutating_job_endpoint("POST", "/v1/jobs") is True
    assert orchestrator_app._is_mutating_job_endpoint("POST", "/v1/jobs/job_123/cancel") is True
    assert orchestrator_app._is_mutating_job_endpoint("GET", "/v1/jobs/job_123") is False
    assert orchestrator_app._is_mutating_job_endpoint("GET", "/v1/jobs/job_123/events") is False


def test_extract_client_ip_does_not_trust_forwarded_header_by_default() -> None:
    previous_trust = orchestrator_app.TRUST_X_FORWARDED_FOR
    previous_proxies = orchestrator_app.TRUSTED_PROXY_IPS
    try:
        orchestrator_app.TRUST_X_FORWARDED_FOR = False
        orchestrator_app.TRUSTED_PROXY_IPS = set()
        request = _build_request(
            "POST",
            "/v1/jobs",
            headers={"x-forwarded-for": "198.51.100.8, 203.0.113.3"},
            client_host="10.0.0.9",
        )
        assert orchestrator_app._extract_client_ip(request) == "10.0.0.9"
    finally:
        orchestrator_app.TRUST_X_FORWARDED_FOR = previous_trust
        orchestrator_app.TRUSTED_PROXY_IPS = previous_proxies


def test_extract_client_ip_trusts_forwarded_header_for_trusted_proxy() -> None:
    previous_trust = orchestrator_app.TRUST_X_FORWARDED_FOR
    previous_proxies = orchestrator_app.TRUSTED_PROXY_IPS
    try:
        orchestrator_app.TRUST_X_FORWARDED_FOR = False
        orchestrator_app.TRUSTED_PROXY_IPS = {"10.0.0.9"}
        request = _build_request(
            "POST",
            "/v1/jobs",
            headers={"x-forwarded-for": "198.51.100.8, 203.0.113.3"},
            client_host="10.0.0.9",
        )
        assert orchestrator_app._extract_client_ip(request) == "198.51.100.8"
    finally:
        orchestrator_app.TRUST_X_FORWARDED_FOR = previous_trust
        orchestrator_app.TRUSTED_PROXY_IPS = previous_proxies


def test_api_key_validation_accepts_header_and_bearer() -> None:
    previous_key = orchestrator_app.API_KEY_SECRET
    previous_header = orchestrator_app.API_KEY_HEADER
    try:
        orchestrator_app.API_KEY_SECRET = "test-api-key"
        orchestrator_app.API_KEY_HEADER = "x-api-key"
        missing = _build_request("POST", "/v1/jobs")
        header_ok = _build_request("POST", "/v1/jobs", headers={"x-api-key": "test-api-key"})
        bearer_ok = _build_request("POST", "/v1/jobs", headers={"authorization": "Bearer test-api-key"})
        wrong = _build_request("POST", "/v1/jobs", headers={"x-api-key": "wrong"})

        assert orchestrator_app._has_valid_api_key(missing) is False
        assert orchestrator_app._has_valid_api_key(header_ok) is True
        assert orchestrator_app._has_valid_api_key(bearer_ok) is True
        assert orchestrator_app._has_valid_api_key(wrong) is False
    finally:
        orchestrator_app.API_KEY_SECRET = previous_key
        orchestrator_app.API_KEY_HEADER = previous_header


def test_content_length_limit_blocks_oversized_payloads() -> None:
    previous_limit = orchestrator_app.MAX_REQUEST_BYTES
    try:
        orchestrator_app.MAX_REQUEST_BYTES = 64
        request = _build_request(
            "POST",
            "/v1/jobs",
            headers={"content-type": "application/json", "content-length": "256"},
        )
        response = orchestrator_app._enforce_content_length_limit(request)
    finally:
        orchestrator_app.MAX_REQUEST_BYTES = previous_limit

    assert response is not None
    assert response.status_code == 413


def test_stream_body_limit_blocks_oversized_chunked_payloads() -> None:
    previous_limit = orchestrator_app.MAX_REQUEST_BYTES
    try:
        orchestrator_app.MAX_REQUEST_BYTES = 8
        request = _build_request("POST", "/v1/jobs", headers={"content-type": "application/json"})
        chunks = [
            {"type": "http.request", "body": b"12345", "more_body": True},
            {"type": "http.request", "body": b"6789", "more_body": False},
        ]

        async def receive():
            if chunks:
                return chunks.pop(0)
            return {"type": "http.request", "body": b"", "more_body": False}

        setattr(request, "_receive", receive)
        orchestrator_app._install_stream_body_limit(request)

        first = asyncio.run(request._receive())  # type: ignore[attr-defined]
        assert first["body"] == b"12345"

        with pytest.raises(orchestrator_app.HTTPException) as exc:
            asyncio.run(request._receive())  # type: ignore[attr-defined]
        assert exc.value.status_code == 413
    finally:
        orchestrator_app.MAX_REQUEST_BYTES = previous_limit


def test_sanitized_child_env_redacts_secret_like_values() -> None:
    old_values = {
        "TP_API_KEY": os.environ.get("TP_API_KEY"),
        "HF_TOKEN": os.environ.get("HF_TOKEN"),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "PATH": os.environ.get("PATH"),
    }
    try:
        os.environ["TP_API_KEY"] = "tp-secret"
        os.environ["HF_TOKEN"] = "hf-secret"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "aws-secret"
        os.environ["PATH"] = "/usr/bin"
        child_env = orchestrator_app._sanitized_child_env()
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    assert "TP_API_KEY" not in child_env
    assert "HF_TOKEN" not in child_env
    assert "AWS_SECRET_ACCESS_KEY" not in child_env
    assert child_env["PATH"] == "/usr/bin"


def test_rate_limiting_returns_true_after_threshold() -> None:
    previous_limit = orchestrator_app.RATE_LIMIT_PER_MINUTE
    try:
        orchestrator_app.RATE_LIMIT_PER_MINUTE = 1
        orchestrator_app.RATE_LIMIT_BUCKETS.clear()
        now = orchestrator_app._now()
        first = orchestrator_app._is_rate_limited("127.0.0.1", now)
        second = orchestrator_app._is_rate_limited("127.0.0.1", now + 0.1)
    finally:
        orchestrator_app.RATE_LIMIT_PER_MINUTE = previous_limit
        orchestrator_app.RATE_LIMIT_BUCKETS.clear()

    assert first is False
    assert second is True


def test_client_ip_prefers_peer_by_default() -> None:
    request = _build_request("GET", "/ready", headers={"x-forwarded-for": "203.0.113.9, 127.0.0.1"}, client_host="10.0.0.1")
    assert orchestrator_app._extract_client_ip(request) == "10.0.0.1"


def test_api_key_validation_accepts_query_param() -> None:
    previous_key = orchestrator_app.API_KEY_SECRET
    try:
        orchestrator_app.API_KEY_SECRET = "query-secret"
        request = _build_request("GET", "/v1/jobs/job_1/events", query_string="api_key=query-secret")
        assert orchestrator_app._has_valid_api_key(request) is True
    finally:
        orchestrator_app.API_KEY_SECRET = previous_key


def test_api_key_query_param_is_rejected_for_non_event_endpoints() -> None:
    previous_key = orchestrator_app.API_KEY_SECRET
    try:
        orchestrator_app.API_KEY_SECRET = "query-secret"
        request = _build_request("GET", "/v1/jobs", query_string="api_key=query-secret")
        assert orchestrator_app._has_valid_api_key(request) is False
    finally:
        orchestrator_app.API_KEY_SECRET = previous_key


def test_protected_job_route_detection() -> None:
    assert orchestrator_app._is_protected_job_endpoint("/v1/jobs") is True
    assert orchestrator_app._is_protected_job_endpoint("/v1/jobs/job_123") is True
    assert orchestrator_app._is_protected_job_endpoint("/v1/jobs/job_123/events") is True
    assert orchestrator_app._is_protected_job_endpoint("/ready") is False


def test_index_job_artifacts_populates_job_payload(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (output_dir / "render.png").write_bytes(b"png")

    job = orchestrator_app.Job(
        id="job_artifacts",
        created_at=orchestrator_app._now(),
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
    )
    indexed = orchestrator_app._index_job_artifacts(job)

    assert len(indexed) == 2
    assert job.artifacts["output_dir"] == str(output_dir)
    assert {item["artifact_type"] for item in indexed} == {"metadata", "image"}
    assert {item["path"] for item in indexed} == {"manifest.json", "render.png"}


def test_index_job_artifacts_truncation_is_sorted_and_stable(tmp_path: Path) -> None:
    previous_limit = orchestrator_app.MAX_INDEXED_ARTIFACTS
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "zeta.txt").write_text("z", encoding="utf-8")
    (output_dir / "alpha.txt").write_text("a", encoding="utf-8")
    (output_dir / "mid.txt").write_text("m", encoding="utf-8")

    try:
        orchestrator_app.MAX_INDEXED_ARTIFACTS = 2
        job = orchestrator_app.Job(
            id="job_artifacts_sorted",
            created_at=orchestrator_app._now(),
            request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
        )
        indexed = orchestrator_app._index_job_artifacts(job)
    finally:
        orchestrator_app.MAX_INDEXED_ARTIFACTS = previous_limit

    assert [item["path"] for item in indexed] == ["alpha.txt", "mid.txt"]
    assert job.artifacts["truncated"] is True
    assert job.artifacts["indexed_count"] == 2


def test_create_job_validation_uses_typed_error_envelope() -> None:
    response = asyncio.run(orchestrator_app.create_job({"pipeline": "unsupported", "args": {}}))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert body["success"] is False
    assert body["error"]["code"] == "INVALID_ARGUMENT"


def test_create_job_archive_gate_invalid_command_uses_typed_error_envelope() -> None:
    response = asyncio.run(
        orchestrator_app.create_job(
            {
                "pipeline": "archive-gate-a",
                "args": {"input_dir": "./input_images", "output_dir": "./output", "archive_command": "stac-export"},
            }
        )
    )
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert body["success"] is False
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["details"]["reason"] == "invalid_archive_command"
    assert orchestrator_app.JOBS == {}


def test_create_job_archive_gate_invalid_integer_option_uses_typed_error_envelope() -> None:
    response = asyncio.run(
        orchestrator_app.create_job(
            {
                "pipeline": "archive-gate-a",
                "args": {
                    "input_dir": "./input_images",
                    "output_dir": "./output",
                    "archive_command": "fixity-scan",
                    "workers": 0,
                },
            }
        )
    )
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 400
    assert body["success"] is False
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["details"]["reason"] == "invalid_archive_integer_option"
    assert orchestrator_app.JOBS == {}


def test_importing_lux_depth_main_does_not_eagerly_import_depth_models() -> None:
    probe = (
        "import sys\n"
        "import transformation_portal.lux_depth_v3.__main__\n"
        "mods = sorted(m for m in sys.modules if m.startswith('transformation_portal.depth.models'))\n"
        "print('\\n'.join(mods))\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], check=True, capture_output=True, text=True)

    assert result.stdout.strip() == ""


def test_list_presets_filters_pipeline() -> None:
    response = asyncio.run(orchestrator_app.list_presets("lux-depth-v3"))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert body["success"] is True
    assert body["data"]["pipeline"] == "lux-depth-v3"
    assert any(item["name"] == "premium" for item in body["data"]["presets"])


def test_list_jobs_includes_error_and_artifacts() -> None:
    job = orchestrator_app.Job(
        id="job_summary",
        created_at=orchestrator_app._now(),
        state="failed",
        progress=72,
        request={"pipeline": "lux-depth-v3"},
        logs_tail=["line-1", "line-2"],
        artifacts={"output_dir": "/tmp/out", "items": [{"artifact_type": "metadata", "path": "/tmp/out/manifest.json"}]},
        error={"code": "RUNNER_ERROR", "message": "boom", "details": {}},
    )
    orchestrator_app.JOBS[job.id] = job

    response = asyncio.run(orchestrator_app.list_jobs())
    body = json.loads(response.body.decode("utf-8"))
    first = body["data"]["jobs"][0]

    assert response.status_code == 200
    assert first["id"] == "job_summary"
    assert first["error"]["code"] == "RUNNER_ERROR"
    assert first["artifacts"]["items"][0]["path"].endswith("manifest.json")


def test_get_job_includes_artifacts_and_error() -> None:
    job = orchestrator_app.Job(
        id="job_details",
        created_at=orchestrator_app._now(),
        state="failed",
        request={"pipeline": "lux-depth-v3"},
        artifacts={"output_dir": "/tmp/out", "items": []},
        error={"code": "RUNNER_ERROR", "message": "boom", "details": {}},
    )
    orchestrator_app.JOBS[job.id] = job

    response = asyncio.run(orchestrator_app.get_job(job.id))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 200
    assert body["data"]["artifacts"]["output_dir"] == "/tmp/out"
    assert body["data"]["error"]["code"] == "RUNNER_ERROR"
