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
    assert module._should_cleanup_output_dir(keep_output=True, output_dir_is_temp=True) is False
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


def test_portal_browser_parse_args_defaults_api_key_to_empty_when_env_is_unset(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("TP_API_KEY", raising=False)
    module = _load_module(PORTAL_BROWSER_SCRIPT_PATH, "tests_validate_portal_browser_smoke_api_key")

    args = module._parse_args([])

    assert args.api_key == ""


def test_portal_browser_help_text_describes_api_key_default(capsys: pytest.CaptureFixture[str]):
    module = _load_module(PORTAL_BROWSER_SCRIPT_PATH, "tests_validate_portal_browser_smoke_help")

    with pytest.raises(SystemExit, match="0"):
        module._parse_args(["--help"])

    help_output = " ".join(capsys.readouterr().out.split())
    assert "API key for protected job endpoints" in help_output
    assert "default: unset; uses TP_API_KEY when set" in help_output


def test_portal_browser_preview_preflight_classifies_auth_failures(monkeypatch: pytest.MonkeyPatch):
    module = _load_module(PORTAL_BROWSER_SCRIPT_PATH, "tests_validate_portal_browser_smoke_preflight_auth")

    monkeypatch.setattr(
        module,
        "_request_json",
        lambda *_args, **_kwargs: (401, {"error": {"code": "UNAUTHORIZED"}}),
    )

    with pytest.raises(module.SmokeFailure, match="rejected the API key"):
        module._preflight_lux_config_preview(
            "http://127.0.0.1:8000",
            "",
            archive_root=Path("/tmp/archive-root"),
            output_dir=Path("/tmp/output-root"),
        )


def test_portal_browser_preview_preflight_classifies_validation_failures(monkeypatch: pytest.MonkeyPatch):
    module = _load_module(PORTAL_BROWSER_SCRIPT_PATH, "tests_validate_portal_browser_smoke_preflight_validation")

    monkeypatch.setattr(
        module,
        "_request_json",
        lambda *_args, **_kwargs: (
            400,
            {
                "error": {
                    "code": "INVALID_ARGUMENT",
                    "details": {"reason": "unsafe_path"},
                }
            },
        ),
    )

    with pytest.raises(module.SmokeFailure, match="rejected the Lux payload or contract"):
        module._preflight_lux_config_preview(
            "http://127.0.0.1:8000",
            "contract-secret",
            archive_root=Path("/tmp/archive-root"),
            output_dir=Path("/tmp/output-root"),
        )


def test_portal_browser_preview_preflight_classifies_service_failures(monkeypatch: pytest.MonkeyPatch):
    module = _load_module(PORTAL_BROWSER_SCRIPT_PATH, "tests_validate_portal_browser_smoke_preflight_service")

    monkeypatch.setattr(
        module,
        "_request_json",
        lambda *_args, **_kwargs: (503, {"error": {"code": "SERVICE_UNAVAILABLE"}}),
    )

    with pytest.raises(module.SmokeFailure, match="is unavailable"):
        module._preflight_lux_config_preview(
            "http://127.0.0.1:8000",
            "contract-secret",
            archive_root=Path("/tmp/archive-root"),
            output_dir=Path("/tmp/output-root"),
        )


def test_portal_browser_preview_preflight_accepts_null_error_envelope(monkeypatch: pytest.MonkeyPatch):
    module = _load_module(PORTAL_BROWSER_SCRIPT_PATH, "tests_validate_portal_browser_smoke_preflight_null_error")

    expected_data = {
        "command": "lux-depth-v3 --input-dir /tmp/archive-root --output-dir /tmp/output-root",
        "field_errors": [],
    }

    monkeypatch.setattr(
        module,
        "_request_json",
        lambda *_args, **_kwargs: (200, {"data": expected_data, "error": None}),
    )

    result = module._preflight_lux_config_preview(
        "http://127.0.0.1:8000",
        "contract-secret",
        archive_root=Path("/tmp/archive-root"),
        output_dir=Path("/tmp/output-root"),
    )

    assert result == expected_data


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
    assert "Make premium media verifiable before it ships." in content
    assert ".hero-video, .homepage-video" in content
    assert "form.requestSubmit" in content
    assert "form.submit();" in content


def test_portal_browser_smoke_tracks_archive_readiness_fields_and_canonical_commands():
    content = PORTAL_BROWSER_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "archiveCanonicalCommand" in content
    assert "archiveIndexFieldVisible" in content
    assert "preRunWarnings" in content
    assert "missingArchiveIndexWarningVisible" in content
    assert "rightsManifestFieldVisible" in content
    assert "segmentationBackendVisible" in content
    assert "governanceDetailsVisible" in content
    assert "advancedFlagsOpen" in content
    assert "governanceDetailsOpen" in content
    assert "reconstructionConfigVisible" in content
    assert "reconstructionDetailsOpen" in content
    assert "dispatchToolsOpen" in content
    assert "v2PresetVisible" in content
    assert "_set_lux_optional_controls_expression" in content
    assert "_restore_archive_gate_form_without_events_expression" in content
    assert "heroReadinessLabel" in content
    assert "locationSearch" in content
    assert "contextRibbonVisible" in content
    assert "contextRibbonJob" in content
    assert "contextRibbonArtifact" in content
    assert "contextRibbonCompare" in content
    assert "postureBandVisible" in content
    assert "summaryBandOutsideReconstruction" in content
    assert "dispatchPrimaryLaneVisible" in content
    assert "dispatchReadinessReason" in content
    assert "/tmp/gate-a-smoke-portal" in content
    assert "archive-gate-b" in content
    assert "archive-gate-c" in content
    assert '--archive-command "bag-build"' in content
    assert '--archive-command "mets-export"' in content
    assert "view=operate&job=" in content
    assert "artifact=" in content
    assert "compare=1" in content


def test_portal_browser_smoke_probes_review_warning_and_provenance_contract():
    content = PORTAL_BROWSER_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "reviewStatusTitle" in content
    assert "reviewStatusDetail" in content
    assert "reviewStatusTone" in content
    assert "reviewStatusVisible" in content
    assert "reviewProvenanceArtifactRole" in content
    assert "reviewProvenanceRunState" in content
    assert "reviewProvenancePath" in content
    assert "reviewProvenanceFreshness" in content
    assert "reviewProvenanceSource" in content
    assert "reviewProvenanceBatch" in content
    assert "Outputs ready for review" in content
    assert "Review provenance should identify the selected artifact path" in content
    assert "Artifact deep link should restore the ribbon artifact context" in content
    assert "compare-only deep link to preserve compare mode" in content
    assert "Compare-only deep links should preserve compare mode for the default artifact" in content
    assert "stale artifact and compare params to normalize" in content


def test_portal_browser_smoke_probes_review_compare_contract():
    content = PORTAL_BROWSER_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "reviewCompareTitle" in content
    assert "reviewCompareDetail" in content
    assert "reviewCompareVisible" in content
    assert "reviewCompareEnabled" in content
    assert '_navigate_to_console_view_expression("review", submitted_job_id, "missing/stale-artifact.png", True)' in content


def test_portal_browser_smoke_tracks_reconstruction_runtime_summary_and_guardrails():
    content = PORTAL_BROWSER_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "summaryReconstructionState" in content
    assert "summaryRuntimeWorkers" in content
    assert "summaryPreviewState" in content
    assert "postureBandVisible" in content
    assert "summaryBandOutsideReconstruction" in content
    assert "rawPreviewStatus" in content
    assert "previewRequestKey" in content
    assert "currentPreviewRequestKey" in content
    assert "previewRequestKeyMatches" in content
    assert "debugBundleGuardrailVisible" in content
    assert "effectiveConfigDrawerVisible" in content
    assert "emit_scene_debug_bundle" in content
    assert "acknowledgement is required" in content
    assert "#openEffectiveConfigBtn" in content
    assert "#closeEffectiveConfigBtn" in content


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
