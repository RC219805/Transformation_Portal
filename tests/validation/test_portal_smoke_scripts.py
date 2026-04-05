"""Unit tests for portal smoke validation scripts."""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.error
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PORTAL_BROWSER_SCRIPT_PATH = PROJECT_ROOT / "scripts/validation/validate_portal_browser_smoke.py"
FRONTDOOR_BROWSER_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "validation" / "validate_frontdoor_browser_smoke.py"
ORCHESTRATOR_HTTP_SCRIPT_PATH = PROJECT_ROOT / "scripts/validation/validate_orchestrator_http_smoke.py"
AUDIT_PIPELINE_READINESS_SCRIPT_PATH = PROJECT_ROOT / "scripts/validation/audit_pipeline_readiness.py"


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_portal_browser_parse_args_does_not_probe_chrome_for_explicit_override(monkeypatch: pytest.MonkeyPatch):
    module = _load_module(PORTAL_BROWSER_SCRIPT_PATH, "tests_validate_portal_browser_smoke")

    def _boom() -> str:
        raise AssertionError("_default_chrome_binary should not be called while parsing args")

    monkeypatch.setattr(module, "_default_chrome_binary", _boom)

    args = module._parse_args(["--chrome-binary", "/custom/chrome"])

    assert args.chrome_binary == "/custom/chrome"


def test_portal_browser_explicit_output_dirs_are_not_auto_cleaned(tmp_path: Path):
    module = _load_module(PORTAL_BROWSER_SCRIPT_PATH, "tests_validate_portal_browser_smoke_cleanup")
    explicit_output_dir = tmp_path / "browser-output"

    resolved_output_dir, output_dir_is_temp = module._resolve_output_dir(str(explicit_output_dir))

    assert resolved_output_dir == explicit_output_dir.resolve()
    assert output_dir_is_temp is False
    assert module._should_cleanup_output_dir(keep_output=False, output_dir_is_temp=output_dir_is_temp) is False
    assert module._should_cleanup_output_dir(keep_output=False, output_dir_is_temp=True) is True


def test_portal_browser_ready_probe_accepts_degraded_shell_after_stalled_bootstrap():
    module = _load_module(PORTAL_BROWSER_SCRIPT_PATH, "tests_validate_portal_browser_smoke_ready")

    assert module._portal_shell_ready(
        {
            "readyState": "complete",
            "title": "Transformation Portal",
            "bootstrapStatus": "degraded",
            "overviewViewVisible": True,
            "runJobDisabled": True,
        }
    )
    assert not module._portal_shell_ready(
        {
            "readyState": "complete",
            "title": "Transformation Portal",
            "bootstrapStatus": "pending",
            "overviewViewVisible": True,
            "runJobDisabled": True,
        }
    )


def test_orchestrator_http_request_json_wraps_transport_failures(monkeypatch: pytest.MonkeyPatch):
    module = _load_module(ORCHESTRATOR_HTTP_SCRIPT_PATH, "tests_validate_orchestrator_http_smoke")

    def _raise_url_error(*_args, **_kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(module.urllib.request, "urlopen", _raise_url_error)

    with pytest.raises(module.SmokeFailure, match="GET /ready request failed: connection refused"):
        module._request_json("http://127.0.0.1:8000", "/ready")


def test_orchestrator_http_explicit_output_dirs_are_not_auto_cleaned(tmp_path: Path):
    module = _load_module(ORCHESTRATOR_HTTP_SCRIPT_PATH, "tests_validate_orchestrator_http_smoke_cleanup")
    explicit_output_dir = tmp_path / "http-output"

    resolved_output_dir, output_dir_is_temp = module._resolve_output_dir(str(explicit_output_dir))

    assert resolved_output_dir == explicit_output_dir.resolve()
    assert output_dir_is_temp is False
    assert module._should_cleanup_output_dir(keep_output=False, output_dir_is_temp=output_dir_is_temp) is False
    assert module._should_cleanup_output_dir(keep_output=False, output_dir_is_temp=True) is True


def test_orchestrator_http_smoke_covers_readiness_and_fail_closed_archive_prereqs():
    content = ORCHESTRATOR_HTTP_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "GET /v1/readiness" in content
    assert '"archive-gate-b"' in content
    assert '"archive-gate-c"' in content
    assert "rights_manifest_required" in content
    assert "manifest-build" in content
    assert "rights-apply" in content
    assert "bag-build" in content
    assert "mets-export" in content


def test_frontdoor_browser_parse_args_does_not_probe_chrome_for_explicit_override(monkeypatch: pytest.MonkeyPatch):
    module = _load_module(FRONTDOOR_BROWSER_SCRIPT_PATH, "tests_validate_frontdoor_browser_smoke")

    def _boom() -> str:
        raise AssertionError("_resolve_chrome_binary should not be called while parsing args")

    monkeypatch.setattr(module, "_resolve_chrome_binary", _boom)

    args = module._parse_args(
        [
            "--chrome-binary",
            "/custom/chrome",
            "--frontdoor-base-url",
            "http://localhost:3000",
            "--username",
            "admin",
            "--password",
            "secret",
        ]
    )

    assert args.chrome_binary == "/custom/chrome"
    assert args.frontdoor_base_url == "http://localhost:3000"
    assert args.username == "admin"
    assert args.password == "secret"


def test_frontdoor_browser_waits_for_managed_portal_bootstrap_before_passing():
    content = FRONTDOOR_BROWSER_SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'and str(value.get("readyState", "")) == "complete"' in content
    assert 'and str(value.get("authModeBadge", "")).lower() == "managed"' in content
    assert "body: new FormData(form)," in content
    assert "window.location.assign(response.url);" in content


def test_portal_browser_smoke_tracks_archive_readiness_fields_and_canonical_commands():
    content = PORTAL_BROWSER_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "archiveCanonicalCommand" in content
    assert "archiveIndexFieldVisible" in content
    assert "rightsManifestFieldVisible" in content
    assert "heroReadinessLabel" in content
    assert "archive-gate-b" in content
    assert "archive-gate-c" in content
    assert '--archive-command "bag-build"' in content
    assert '--archive-command "mets-export"' in content


def test_audit_pipeline_readiness_generates_fixture_backed_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_module(AUDIT_PIPELINE_READINESS_SCRIPT_PATH, "tests_audit_pipeline_readiness")
    monkeypatch.setattr(
        module,
        "_lux_depth_audit_entry",
        lambda: {
            "canonical_command": "lux-depth-v3",
            "base_status": "ready",
            "canary_status": "unavailable",
            "missing_prerequisites": [],
            "runner_details": {"type": "python_module", "available": True},
            "notes": ["safe lane ready"],
        },
    )

    output_dir = tmp_path / "audit-output"
    json_output = tmp_path / "audit-matrix.json"

    exit_code = module.main(["--output-dir", str(output_dir), "--json-output", str(json_output)])

    assert exit_code == 0
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["schema"] == "tp.orchestrator.pipeline_readiness_audit.v1"
    assert payload["success"] is True
    assert payload["data"]["pipelines"]["lux-depth-v3"]["base_status"] == "ready"
    assert payload["data"]["pipelines"]["lux-depth-v3"]["canary_status"] == "unavailable"
    assert payload["data"]["pipelines"]["archive-gate-a"]["command_exit_code"] == 0
    assert payload["data"]["pipelines"]["archive-gate-b"]["blocked_without_manifest"]["status"] == "blocked"
    assert payload["data"]["pipelines"]["archive-gate-b"]["dispatch_readiness"]["status"] == "ready"
    assert payload["data"]["pipelines"]["archive-gate-c"]["blocked_without_manifest"]["status"] == "blocked"
    assert payload["data"]["pipelines"]["archive-gate-c"]["dispatch_readiness"]["status"] == "ready"
