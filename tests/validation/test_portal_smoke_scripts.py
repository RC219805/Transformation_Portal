"""Unit tests for portal smoke validation scripts."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PORTAL_BROWSER_SCRIPT_PATH = PROJECT_ROOT / "scripts/validation/validate_portal_browser_smoke.py"
PORTAL_LUX_MATERIALS_SCRIPT_PATH = PROJECT_ROOT / "scripts/validation/validate_portal_lux_materials_live.py"
PORTAL_FASTVLM_CAPTIONING_SCRIPT_PATH = PROJECT_ROOT / "scripts/validation/validate_portal_fastvlm_captioning_live.py"
FRONTDOOR_BROWSER_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "validation" / "validate_frontdoor_browser_smoke.py"
ORCHESTRATOR_HTTP_SCRIPT_PATH = PROJECT_ROOT / "scripts/validation/validate_orchestrator_http_smoke.py"
AUDIT_PIPELINE_READINESS_SCRIPT_PATH = PROJECT_ROOT / "scripts/validation/audit_pipeline_readiness.py"
PORTAL_CSS_LAYER_PARITY_SCRIPT_PATH = PROJECT_ROOT / "scripts/validation/validate_portal_css_layer_parity.py"


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_portal_css_layer_parity_module(module_name: str):
    script_dir = str(PORTAL_CSS_LAYER_PARITY_SCRIPT_PATH.parent)
    sys.path.insert(0, script_dir)
    try:
        return _load_module(PORTAL_CSS_LAYER_PARITY_SCRIPT_PATH, module_name)
    finally:
        sys.path.remove(script_dir)


def _load_portal_fastvlm_captioning_module(module_name: str):
    script_dir = str(PORTAL_FASTVLM_CAPTIONING_SCRIPT_PATH.parent)
    sys.path.insert(0, script_dir)
    try:
        return _load_module(PORTAL_FASTVLM_CAPTIONING_SCRIPT_PATH, module_name)
    finally:
        sys.path.remove(script_dir)


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


def test_portal_browser_parse_args_supports_local_backend_spawn_flag():
    module = _load_module(PORTAL_BROWSER_SCRIPT_PATH, "tests_validate_portal_browser_smoke_spawn_backend")

    args = module._parse_args(["--spawn-local-backend", "--backend-startup-timeout-seconds", "12.5"])

    assert args.spawn_local_backend is True
    assert args.backend_startup_timeout_seconds == 12.5


def test_portal_browser_tail_text_reads_only_a_bounded_suffix(tmp_path: Path):
    module = _load_module(PORTAL_BROWSER_SCRIPT_PATH, "tests_validate_portal_browser_smoke_tail")
    log_path = tmp_path / "portal.log"
    log_path.write_text(("0123456789" * 1024) + "tail-marker", encoding="utf-8")

    tail = module._tail_text(log_path, max_chars=24, max_bytes=96)

    assert tail.endswith("tail-marker")
    assert len(tail) <= 24


def test_portal_browser_main_terminates_spawned_backend_on_setup_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = _load_module(PORTAL_BROWSER_SCRIPT_PATH, "tests_validate_portal_browser_smoke_runtime_cleanup")
    runtime_handle = SimpleNamespace(base_url="http://127.0.0.1:8123")
    terminated: list[object] = []
    archive_index = tmp_path / "archive_index.csv.gz"
    archive_index.write_text("fixture", encoding="utf-8")

    monkeypatch.setattr(module, "_spawn_local_backend", lambda *_args, **_kwargs: runtime_handle)
    monkeypatch.setattr(module, "_terminate_runtime", lambda handle: terminated.append(handle))

    with pytest.raises(module.SmokeFailure, match="Archive root fixture does not exist"):
        module.main(
            [
                "--spawn-local-backend",
                "--archive-root",
                str(tmp_path / "missing-archive-root"),
                "--archive-index",
                str(archive_index),
            ]
        )

    assert terminated == [runtime_handle]


def test_portal_browser_help_text_describes_api_key_default(capsys: pytest.CaptureFixture[str]):
    module = _load_module(PORTAL_BROWSER_SCRIPT_PATH, "tests_validate_portal_browser_smoke_help")

    with pytest.raises(SystemExit, match="0"):
        module._parse_args(["--help"])

    help_output = " ".join(capsys.readouterr().out.split())
    assert "API key for protected job endpoints" in help_output
    assert "default: unset; uses TP_API_KEY when set" in help_output


def test_portal_browser_state_probe_tracks_contextual_action_controls():
    module = _load_module(PORTAL_BROWSER_SCRIPT_PATH, "tests_validate_portal_browser_smoke_action_probe")

    expression = module._state_probe_expression()

    assert "consoleActionPrimaryBtn" in expression
    assert "consoleActionSecondaryBtn1" in expression
    assert "consoleActionSecondaryBtn2" in expression
    assert "selectedJobRecoveryPrimaryBtn" in expression
    assert "selectedJobRecoverySecondaryBtn" in expression
    assert "reviewStatusPrimaryBtn" in expression
    assert "reviewStatusSecondaryBtn" in expression
    assert "actionPrimaryKey" in expression
    assert "actionSecondary2Key" in expression
    assert "selectedRecoveryPrimaryKey" in expression
    assert "reviewStatusPrimaryKey" in expression
    assert "reviewStatusState" in expression
    assert "connectionDetailsVisible" in expression
    assert "dispatchChecklistRows" in expression
    assert "return visible('build-shell');" in expression
    assert "return visible('jobs-shell');" in expression
    assert "return visible('overview-shell');" in expression
    assert "enableSegmentationChecked" in expression
    assert "segmentationBackendValue" in expression
    assert "strictSegmentationChecked" in expression
    assert "queueEmptyStateVisible" in expression
    assert "artifactEmptyStateVisible" in expression


def test_portal_css_layer_parity_probe_guard_captures_mutated_state() -> None:
    module = _load_portal_css_layer_parity_module("tests_validate_portal_css_layer_parity_guard")

    guard_source = module._portal_parity_probe_guard_source()

    assert "createPortalParityProbeGuard" in guard_source
    assert "captureStorage" in guard_source
    assert "captureNodeAndAncestors" in guard_source
    assert "captureProperty" in guard_source
    assert "rootClassSnapshot" in guard_source
    assert "window.__portalParityProbeGuard" in module._portal_parity_probe_restore_expression()


def test_portal_css_layer_parity_mutating_probes_use_probe_guard() -> None:
    module = _load_portal_css_layer_parity_module("tests_validate_portal_css_layer_parity_probes")

    expressions = {
        "review status": module._review_status_tone_probe_expression(),
        "interaction outline": module._interaction_outline_setup_expression(),
        "skeleton": module._skeleton_visibility_probe_expression("dark"),
        "snapshot": module._force_snapshot_state_expression(),
        "class census": module._force_census_state_expression("dark"),
    }

    for label, expression in expressions.items():
        assert "createPortalParityProbeGuard" in expression, label
        assert "captureStorage('tp_theme', 'tp_theme_version')" in expression, label

    assert "guard.restore();" in expressions["review status"]
    assert "guard.restore();" in expressions["skeleton"]
    assert "window.__portalParityProbeGuard = guard" in expressions["interaction outline"]
    assert "window.__portalParityProbeGuard = guard" in expressions["snapshot"]
    assert "window.__portalParityProbeGuard = guard" in expressions["class census"]


def test_portal_browser_accessibility_probe_tracks_target_size_and_disclosure_contracts():
    module = _load_module(PORTAL_BROWSER_SCRIPT_PATH, "tests_validate_portal_browser_smoke_accessibility_probe")

    expression = module._accessibility_probe_expression()

    assert "#themeBtn" in expression
    assert "#shortcutsBtn" in expression
    assert '[data-ui="view-link"]' in expression
    assert "#buildStepTab1" in expression
    assert "#connectionDetails > summary" in expression
    assert "focusVisibleWithStickyShells" in expression
    assert "maxDisclosureDepth" in expression
    assert "discoverableDisclosures" in expression
    assert "prefers-reduced-motion" in expression
    assert "decorativeMotionStatic" in expression


def test_portal_browser_can_simulate_bootstrap_degraded_recovery_actions():
    module = _load_module(PORTAL_BROWSER_SCRIPT_PATH, "tests_validate_portal_browser_smoke_degraded_expr")

    expression = module._simulate_bootstrap_degraded_expression(reason="auth_failure", http_status=401)

    assert "_applyPortalBootstrap" in expression
    assert "status: 'degraded'" in expression
    assert '"reason": "auth_failure"' in expression
    assert '"http_status": 401' in expression
    assert "renderSelectedJobInspector();" in expression
    assert "renderArtifactPanel();" in expression
    assert "renderConsoleContextRibbon();" in expression


def test_portal_browser_can_inject_compare_ready_review_state():
    module = _load_module(PORTAL_BROWSER_SCRIPT_PATH, "tests_validate_portal_browser_smoke_compare_expr")

    expression = module._inject_compare_ready_review_expression("job_demo")

    assert "synthetic/review-primary.png" in expression
    assert "synthetic/review-compare.png" in expression
    assert "browser_previewable: true" in expression
    assert "portal-smoke-compare" in expression
    assert "renderReviewSurfaces();" in expression


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
    assert '"details") or {}).get("field") == "manifest_jsonl"' in content
    assert '"details") or {}).get("reason") == "required"' in content
    assert "manifest-build" in content
    assert "rights-apply" in content
    assert "bag-build" in content
    assert "mets-export" in content


def test_portal_lux_materials_payload_enforces_required_efficientsam_contract(tmp_path: Path):
    module = _load_module(PORTAL_LUX_MATERIALS_SCRIPT_PATH, "tests_validate_portal_lux_materials_payload")

    payload = module._build_lux_materials_payload(
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
    )

    args = payload["args"]
    assert payload["pipeline"] == "lux-depth-v3"
    assert args["quality_tier"] == "apex"
    assert args["depth_backend"] == "da3"
    assert args["depth_device"] == "cpu"
    assert args["materials_v3"] is True
    assert args["enable_segmentation"] is True
    assert args["segmentation_backend"] == "efficientsam"
    assert args["strict_segmentation"] is True
    assert args["pbr"] is True
    assert args["emit_report"] is True
    assert args["emit_run_card"] is True
    assert args["run_card_version"] == "v2"
    assert args["enable_v2"] is False
    assert args["non_commercial_ok"] is True


def test_portal_lux_materials_payload_adds_sam2_overrides(tmp_path: Path):
    module = _load_module(PORTAL_LUX_MATERIALS_SCRIPT_PATH, "tests_validate_portal_lux_materials_sam2_payload")
    checkpoint = tmp_path / "sam2.pt"

    payload = module._build_lux_materials_payload(
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        segmentation_backend="sam2",
        sam2_checkpoint_path=checkpoint,
        sam2_model_size="large",
    )

    args = payload["args"]
    assert args["segmentation_backend"] == "sam2"
    assert args["sam2_model_size"] == "large"
    assert args["sam2_checkpoint_path"] == str(checkpoint)
    assert args["sam2_max_concurrency"] == 1


def test_portal_lux_materials_preview_validation_requires_contract_flags(tmp_path: Path):
    module = _load_module(PORTAL_LUX_MATERIALS_SCRIPT_PATH, "tests_validate_portal_lux_materials_preview")
    payload = module._build_lux_materials_payload(input_dir=tmp_path / "input", output_dir=tmp_path / "output")

    preview = {
        "field_errors": [],
        "normalized_args": dict(payload["args"]),
        "execution_args": dict(payload["args"]),
        "argv_preview": (
            "python -m transformation_portal.lux_depth_v3 --materials-v3 on "
            "--enable-segmentation on --segmentation-backend efficientsam "
            "--strict-segmentation --non-commercial-ok true"
        ),
    }

    module._validate_lux_preview(preview, expected_backend="efficientsam")


def test_portal_lux_materials_preview_validation_fails_closed_on_field_errors(tmp_path: Path):
    module = _load_module(PORTAL_LUX_MATERIALS_SCRIPT_PATH, "tests_validate_portal_lux_materials_preview_errors")
    payload = module._build_lux_materials_payload(input_dir=tmp_path / "input", output_dir=tmp_path / "output")
    preview = {
        "field_errors": [{"field": "segmentation_backend", "code": "invalid_segmentation_backend"}],
        "normalized_args": dict(payload["args"]),
        "execution_args": dict(payload["args"]),
        "argv_preview": "",
    }

    with pytest.raises(module.SmokeFailure, match="Preview returned field errors") as exc_info:
        module._validate_lux_preview(preview, expected_backend="efficientsam")

    assert exc_info.value.kind == "contract"


def test_portal_lux_materials_output_validation_checks_masks_manifest_and_run_card(tmp_path: Path):
    import numpy as np

    module = _load_module(PORTAL_LUX_MATERIALS_SCRIPT_PATH, "tests_validate_portal_lux_materials_outputs")
    output_dir = tmp_path / "output"
    segmentation_dir = output_dir / "segmentation"
    manifests_dir = output_dir / "manifests"
    segmentation_dir.mkdir(parents=True)
    manifests_dir.mkdir(parents=True)

    mask_relative_path = "segmentation/fixture_materials_v3_masks.npz"
    mask_path = output_dir / mask_relative_path
    np.savez_compressed(mask_path, glass=np.ones((2, 2), dtype=np.float32))

    manifest_relative_path = "manifests/fixture_combined.json"
    (output_dir / manifest_relative_path).write_text(
        json.dumps(
            {
                "materials_v3": {
                    "enabled": True,
                    "version": "3.1",
                    "segmentation_metadata": {
                        "backend": "efficientsam",
                        "mask_count": 1,
                        "mask_artifact_path": str(mask_path),
                        "errors": [],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    run_card_path = output_dir / "run_card_fixture.json"
    run_card_path.write_text(
        json.dumps(
            {
                "result_summary": [
                    {
                        "segmentation_status": {
                            "enabled": True,
                            "backend": "efficientsam",
                            "mask_count": 1,
                            "errors": [],
                            "mask_artifact_path": mask_relative_path,
                        }
                    }
                ],
                "artifact_index": [
                    {
                        "relative_path": mask_relative_path,
                        "artifact_type": "segmentation_mask_npz",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = module._validate_lux_outputs(
        {
            "artifacts": {
                "output_dir": str(output_dir),
                "items": [
                    {"relative_path": mask_relative_path},
                    {"relative_path": manifest_relative_path},
                ],
            }
        },
        expected_backend="efficientsam",
    )

    assert result["mask_relative_path"] == mask_relative_path
    assert result["manifest_relative_path"] == manifest_relative_path
    assert result["run_card_path"] == str(run_card_path)
    assert result["mask_stats"]["non_empty_mask_count"] == 1


def test_portal_fastvlm_captioning_payload_enables_advisory_smoke_role(tmp_path: Path):
    module = _load_portal_fastvlm_captioning_module("tests_validate_portal_fastvlm_captioning_payload")

    payload = module._build_captioning_payload(
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        model_role="smoke",
        timeout_seconds=45,
    )

    args = payload["args"]
    assert payload["pipeline"] == "lux-depth-v3"
    assert args["vlm_captioning_enabled"] is True
    assert args["vlm_captioning_backend"] == "fastvlm"
    assert args["vlm_captioning_model"] == "smoke"
    assert args["vlm_captioning_proxy_format"] == "png"
    assert args["fastvlm_timeout_seconds"] == 45
    assert args["materials_v3"] is False
    assert args["enable_segmentation"] is False
    assert args["emit_run_card"] is True
    assert args["run_card_version"] == "v2"


def test_portal_fastvlm_captioning_preview_requires_ready_advisory_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_portal_fastvlm_captioning_module("tests_validate_portal_fastvlm_captioning_preview")
    payload = module._build_captioning_payload(
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        model_role="smoke",
        timeout_seconds=45,
    )
    preview = {
        "field_errors": [],
        "normalized_args": dict(payload["args"]),
        "execution_args": dict(payload["args"]),
        "captioning_summary": {
            "feature_enabled": True,
            "enabled": True,
            "backend": "fastvlm",
            "model": "smoke",
            "role": "advisory",
            "used_for_quality_gate": False,
            "runtime_status": "ready",
        },
        "argv_preview": (
            "python -m transformation_portal.lux_depth_v3 --vlm-captioning on "
            "--vlm-captioning-backend fastvlm --vlm-captioning-model smoke "
            "--vlm-captioning-proxy-format png"
        ),
    }

    def fake_request_json(*_args, **_kwargs):  # noqa: ANN001
        return 200, {"success": True, "data": preview}

    monkeypatch.setattr(module, "_request_json", fake_request_json)

    result = module._preview_captioning_job("http://127.0.0.1:8000", api_key="secret", payload=payload, model_role="smoke")

    assert result is preview


def test_portal_fastvlm_captioning_output_validation_requires_advisory_artifacts(tmp_path: Path):
    module = _load_portal_fastvlm_captioning_module("tests_validate_portal_fastvlm_captioning_outputs")
    output_dir = tmp_path / "output"
    caption_dir = output_dir / "captioning"
    caption_dir.mkdir(parents=True)
    sidecar_relative_path = "captioning/image.vlm_captioning.sidecar.json"
    raw_relative_path = "captioning/image.vlm_captioning.raw.txt"
    proxy_relative_path = "captioning/image_proxy.png"
    (output_dir / sidecar_relative_path).write_text(
        json.dumps(
            {
                "vlm_captioning": {
                    "provider": "fastvlm",
                    "role": "advisory",
                    "used_for_quality_gate": False,
                    "runtime_diagnostics": {"success": True},
                }
            }
        ),
        encoding="utf-8",
    )
    (output_dir / raw_relative_path).write_text("SCENE=Pool\n", encoding="utf-8")
    (output_dir / proxy_relative_path).write_bytes(b"png")
    run_card_path = output_dir / "run_card_fixture.json"
    run_card_path.write_text(
        json.dumps(
            {
                "captioning_status": {
                    "role": "advisory",
                    "used_for_quality_gate": False,
                    "sidecar_count": 1,
                }
            }
        ),
        encoding="utf-8",
    )

    result = module._validate_captioning_outputs(
        {
            "artifacts": {
                "output_dir": str(output_dir),
                "items": [
                    {"relative_path": sidecar_relative_path, "artifact_type": "vlm_caption_sidecar"},
                    {"relative_path": raw_relative_path, "artifact_type": "vlm_caption_raw_text"},
                    {"relative_path": proxy_relative_path, "artifact_type": "image"},
                ],
            }
        }
    )

    assert result["sidecar_relative_path"] == sidecar_relative_path
    assert result["raw_relative_path"] == raw_relative_path
    assert result["proxy_relative_path"] == proxy_relative_path
    assert result["captioning_status"]["used_for_quality_gate"] is False


def test_portal_fastvlm_captioning_runtime_check_is_local_backend_scoped():
    module = _load_portal_fastvlm_captioning_module("tests_validate_portal_fastvlm_captioning_runtime_scope")

    assert module._should_validate_local_runtime(
        spawn_local_backend=True,
        skip_local_runtime_check=False,
        require_local_runtime_check=False,
    )
    assert not module._should_validate_local_runtime(
        spawn_local_backend=False,
        skip_local_runtime_check=False,
        require_local_runtime_check=False,
    )
    assert module._should_validate_local_runtime(
        spawn_local_backend=False,
        skip_local_runtime_check=False,
        require_local_runtime_check=True,
    )
    assert not module._should_validate_local_runtime(
        spawn_local_backend=True,
        skip_local_runtime_check=True,
        require_local_runtime_check=True,
    )


def test_portal_lux_materials_sam2_prerequisite_reports_missing_checkpoint(tmp_path: Path):
    module = _load_module(PORTAL_LUX_MATERIALS_SCRIPT_PATH, "tests_validate_portal_lux_materials_sam2_prereq")

    reason = module._sam2_prerequisite_failure(tmp_path / "missing-sam2.pt")

    assert reason is not None
    assert reason.startswith("checkpoint_missing:")


def test_portal_lux_materials_terminal_failure_classifies_missing_runtime_as_environment():
    module = _load_module(PORTAL_LUX_MATERIALS_SCRIPT_PATH, "tests_validate_portal_lux_materials_failure_kind")

    kind = module._classify_terminal_job_failure(
        {
            "data": {
                "error": {"code": "RUNNER_EXIT_NONZERO"},
                "logs_tail": ["ModuleNotFoundError: No module named 'torchvision'"],
            }
        }
    )

    assert kind == "environment"


def test_portal_lux_materials_script_documents_sam2_optional_gate():
    content = PORTAL_LUX_MATERIALS_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "TP_PORTAL_LUX_RUN_SAM2" in content
    assert "TP_PORTAL_LUX_REQUIRE_SAM2" in content
    assert "sam2_skipped" in content
    assert "segmentation_mask_npz" in content


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


def test_frontdoor_browser_parse_args_supports_isolated_runtime_flags():
    module = _load_module(FRONTDOOR_BROWSER_SCRIPT_PATH, "tests_validate_frontdoor_browser_smoke_spawn_flags")

    args = module._parse_args(
        [
            "--spawn-local-frontdoor",
            "--spawn-local-backend",
            "--backend-base-url",
            "http://127.0.0.1:9000",
            "--backend-api-key",
            "backend-secret",
        ]
    )

    assert args.spawn_local_frontdoor is True
    assert args.spawn_local_backend is True
    assert args.backend_base_url == "http://127.0.0.1:9000"
    assert args.backend_api_key == "backend-secret"


def test_frontdoor_browser_spawned_local_frontdoor_defaults_credentials(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("TP_FRONTDOOR_USERNAME", raising=False)
    monkeypatch.delenv("TP_FRONTDOOR_PASSWORD", raising=False)
    module = _load_module(FRONTDOOR_BROWSER_SCRIPT_PATH, "tests_validate_frontdoor_browser_smoke_defaults")

    args = module._parse_args(["--spawn-local-frontdoor"])

    assert module._resolve_username(args) == module.DEFAULT_FRONTDOOR_USERNAME
    assert module._resolve_password(args) == module.DEFAULT_FRONTDOOR_PASSWORD
    assert module._resolve_access_email(module.DEFAULT_FRONTDOOR_USERNAME) == "smoke-admin@local.invalid"


def test_frontdoor_browser_tail_text_reads_only_a_bounded_suffix(tmp_path: Path):
    module = _load_module(FRONTDOOR_BROWSER_SCRIPT_PATH, "tests_validate_frontdoor_browser_smoke_tail")
    log_path = tmp_path / "frontdoor.log"
    log_path.write_text(("abcdefghij" * 1024) + "tail-marker", encoding="utf-8")

    tail = module._tail_text(log_path, max_chars=24, max_bytes=96)

    assert tail.endswith("tail-marker")
    assert len(tail) <= 24


def test_frontdoor_browser_prunes_only_stale_smoke_distdirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = _load_module(FRONTDOOR_BROWSER_SCRIPT_PATH, "tests_validate_frontdoor_browser_smoke_stale_distdirs")
    now = 1_000_000.0
    old_mtime = now - module.FRONTDOOR_STALE_DIST_DIR_MIN_AGE_SECONDS - 1
    fresh_mtime = now - module.FRONTDOOR_STALE_DIST_DIR_MIN_AGE_SECONDS + 1
    active = tmp_path / ".next-smoke-3000"
    stale = tmp_path / ".next-smoke-3001"
    fresh = tmp_path / ".next-smoke-3002"
    unrelated = tmp_path / ".next"
    active.mkdir()
    stale.mkdir()
    fresh.mkdir()
    unrelated.mkdir()
    for candidate in (active, stale, unrelated):
        os.utime(candidate, (old_mtime, old_mtime))
    os.utime(fresh, (fresh_mtime, fresh_mtime))

    monkeypatch.setattr(module, "FRONTDOOR_ROOT", tmp_path)

    module._prune_stale_frontdoor_dist_dirs(active, now=now)

    assert active.is_dir()
    assert not stale.exists()
    assert fresh.is_dir()
    assert unrelated.is_dir()


def test_frontdoor_browser_prune_reports_cleanup_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = _load_module(FRONTDOOR_BROWSER_SCRIPT_PATH, "tests_validate_frontdoor_browser_smoke_stale_failure")
    now = 1_000_000.0
    old_mtime = now - module.FRONTDOOR_STALE_DIST_DIR_MIN_AGE_SECONDS - 1
    active = tmp_path / ".next-smoke-3000"
    stale = tmp_path / ".next-smoke-3001"
    active.mkdir()
    stale.mkdir()
    for candidate in (active, stale):
        os.utime(candidate, (old_mtime, old_mtime))

    def _fail_rmtree(_candidate: Path) -> None:
        raise OSError("permission denied")

    monkeypatch.setattr(module, "FRONTDOOR_ROOT", tmp_path)
    monkeypatch.setattr(module.shutil, "rmtree", _fail_rmtree)

    with pytest.raises(module.SmokeFailure, match="Could not remove stale front-door smoke distDir.*permission denied"):
        module._prune_stale_frontdoor_dist_dirs(active, now=now)


def test_frontdoor_browser_main_terminates_spawned_runtimes_on_setup_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = _load_module(FRONTDOOR_BROWSER_SCRIPT_PATH, "tests_validate_frontdoor_browser_smoke_runtime_cleanup")
    backend_runtime = SimpleNamespace(base_url="http://127.0.0.1:8124")
    frontdoor_runtime = SimpleNamespace(base_url="http://localhost:3010")
    terminated: list[object] = []
    frontdoor_launches: list[dict[str, object]] = []

    monkeypatch.setattr(module, "_spawn_local_backend", lambda *_args, **_kwargs: backend_runtime)
    monkeypatch.setattr(
        module,
        "_spawn_local_frontdoor",
        lambda **kwargs: frontdoor_launches.append(kwargs) or frontdoor_runtime,
    )
    monkeypatch.setattr(module, "_terminate_runtime", lambda handle: terminated.append(handle))
    monkeypatch.setattr(module, "_resolve_chrome_binary", lambda _raw: str(tmp_path / "missing-chrome"))

    with pytest.raises(module.SmokeFailure, match="Chrome binary does not exist"):
        module.main(
            [
                "--spawn-local-backend",
                "--spawn-local-frontdoor",
                "--backend-api-key",
                "contract-secret",
                "--debugging-port",
                "9222",
            ]
        )

    assert frontdoor_launches == [
        {
            "username": module.DEFAULT_FRONTDOOR_USERNAME,
            "password": module.DEFAULT_FRONTDOOR_PASSWORD,
            "access_email": "smoke-admin@local.invalid",
            "backend_base_url": backend_runtime.base_url,
            "backend_api_key": "contract-secret",
            "timeout_seconds": 45.0,
        }
    ]
    assert terminated == [frontdoor_runtime, backend_runtime]


def test_frontdoor_browser_requires_explicit_credentials_for_non_spawned_frontdoor(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("TP_FRONTDOOR_USERNAME", raising=False)
    monkeypatch.delenv("TP_FRONTDOOR_PASSWORD", raising=False)
    module = _load_module(FRONTDOOR_BROWSER_SCRIPT_PATH, "tests_validate_frontdoor_browser_smoke_missing_creds")

    with pytest.raises(module.SmokeFailure, match="Front-door username and password are required"):
        module.main([])


def test_frontdoor_browser_waits_for_managed_portal_bootstrap_before_passing():
    content = FRONTDOOR_BROWSER_SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'and str(value.get("readyState", "")) == "complete"' in content
    assert content.count('and str(value.get("readyState", "")) == "complete"') >= 3
    assert 'and str(value.get("authModeBadge", "")).lower() == "managed"' in content
    assert "locationSearch: window.location.search" in content
    assert "homepageHeroReady" in content
    assert "homepageEntryRailReady" in content
    assert "homepageLearnLinkReady" in content
    assert "homepagePrimaryCtaHref" in content
    assert "loginEntryStateReady" in content
    assert "loginSequenceReady" in content
    assert "portalAccessStateReady" in content
    assert '[data-ui="homepage-hero-title"]' in content
    assert '[data-ui="homepage-entry-rail"]' in content
    assert '[data-ui="homepage-learn-link"]' in content
    assert '[data-ui="login-form"]' in content
    assert '[data-ui="login-entry-state"]' in content
    assert '[data-ui="login-sequence"]' in content
    assert '[data-ui="portal-access-state"]' in content
    assert ".hero-video, .homepage-video" in content
    assert "def _populate_login_expression" in content
    assert "def _click_expression" in content
    assert "connection.evaluate(_populate_login_expression(username, password))" in content
    assert "connection.evaluate(_click_expression('[data-ui=\"login-submit\"]'))" in content
    assert "/login?returnTo=%2Fportal%3Fview%3Dbuild" in content
    assert 'and "returnTo=%2Fportal%3Fview%3Dbuild" in str(value.get("locationSearch", ""))' in content
    assert 'and str(value.get("currentView", "")) == "build"' in content
    assert 'and "view=build" in str(value.get("locationSearch", ""))' in content
    assert 'and bool(value.get("buildViewVisible"))' in content
    assert 'and not bool(value.get("overviewViewVisible"))' in content
    assert 'and not bool(value.get("operateViewVisible"))' in content
    assert "/healthz" in content
    assert "--spawn-local-frontdoor" in content
    assert "--spawn-local-backend" in content


def test_frontdoor_browser_smoke_pins_managed_logout_click_flow():
    """Pin the #1713 governed logout click smoke against silent removal.

    The smoke at scripts/validation/validate_frontdoor_browser_smoke.py
    is the only repo-owned harness that exercises the real-bundle
    managed logout click flow end to end (the @portal-browser
    Playwright lane stubs portal.js so it can only assert
    server-rendered structure). Future edits could quietly drop the
    new probe fields, swap the click selector for the legacy id-only
    form, rename the poll descriptions, or skip the post-logout
    bounce check; none of those would fail a Chrome-less CI lane and
    none would fail flake8. This content-grep pins the exact tokens
    the smoke needs so any of those regressions trip an explicit
    test failure with a clear name.
    """
    content = FRONTDOOR_BROWSER_SCRIPT_PATH.read_text(encoding="utf-8")

    # Probe extension — three new fields the new poll predicates read.
    assert "logoutButtonPresent" in content
    assert "logoutButtonVisible" in content
    # Click target must be the data-ui hook, not a brittle text or id
    # selector (matches the data-ui="logout-button" attribute pinned
    # by tests/test_app_orchestrator_runtime.py for the rendered HTML).
    assert '[data-ui="logout-button"]' in content
    # Both _poll descriptions must remain stable so a poll-timeout
    # error message names the actual flow being asserted.
    assert "front-door logout to return to login" in content
    assert "post-logout portal access to require login" in content
    # The defense-in-depth post-logout bounce navigates to the same
    # managed deep link the entry block uses; keeping the literal
    # string here pins that we still exercise the deep link, not just
    # the bare /portal path.
    assert "/portal?view=build" in content


def test_frontdoor_browser_accessibility_probe_tracks_target_size_and_reduced_motion():
    module = _load_module(FRONTDOOR_BROWSER_SCRIPT_PATH, "tests_validate_frontdoor_browser_smoke_accessibility_probe")

    expression = module._frontdoor_accessibility_probe_expression()

    assert "readyState: document.readyState" in expression
    assert '[data-ui="homepage-primary-cta"]' in expression
    assert '[data-ui="homepage-secondary-cta"]' in expression
    assert '[data-ui="homepage-learn-link"]' in expression
    assert '[data-ui="login-submit"]' in expression
    assert '[data-ui="login-secondary-link"]' in expression
    assert "maxDisclosureDepth" in expression
    assert "focusVisibleWithStickyHeader" in expression
    assert "prefers-reduced-motion" in expression
    assert "decorativeMotionStatic" in expression


def test_frontdoor_browser_accessibility_snapshot_is_page_scoped():
    module = _load_module(FRONTDOOR_BROWSER_SCRIPT_PATH, "tests_validate_frontdoor_browser_smoke_accessibility_scope")

    snapshot = {
        "pathname": "/login",
        "readyState": "complete",
        "homepagePrimaryMinTarget": True,
        "homepageSecondaryMinTarget": True,
        "homepageLearnMinTarget": True,
        "focusVisibleWithStickyHeader": True,
        "loginSubmitMinTarget": False,
        "loginSecondaryMinTarget": True,
        "maxDisclosureDepth": 1,
        "reducedMotion": False,
        "decorativeMotionStatic": True,
    }

    homepage_snapshot = module._frontdoor_accessibility_snapshot(snapshot, page="homepage")
    login_snapshot = module._frontdoor_accessibility_snapshot(snapshot, page="login")

    assert homepage_snapshot["pathname"] == "/login"
    assert homepage_snapshot["readyState"] == "complete"
    assert homepage_snapshot["homepagePrimaryMinTarget"] is True
    assert "loginSubmitMinTarget" not in homepage_snapshot

    assert login_snapshot["pathname"] == "/login"
    assert login_snapshot["readyState"] == "complete"
    assert login_snapshot["loginSubmitMinTarget"] is False
    assert login_snapshot["decorativeMotionStatic"] is True
    assert "homepagePrimaryMinTarget" not in login_snapshot


def test_portal_browser_smoke_tracks_archive_readiness_fields_and_canonical_commands():
    content = PORTAL_BROWSER_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "archiveCanonicalCommand" in content
    assert "archiveIndexFieldVisible" in content
    assert "preRunWarnings" in content
    assert "missingArchiveIndexWarningVisible" in content
    assert "rightsManifestFieldVisible" in content
    assert "enableSegmentationChecked" in content
    assert "segmentationBackendVisible" in content
    assert "segmentationBackendValue" in content
    assert "strictSegmentationChecked" in content
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
    assert "connectionDetailsVisible" in content
    assert "dispatchChecklistRows" in content
    assert "dispatchChecklistHasPass" in content
    assert "queueEmptyStateVisible" in content
    assert "artifactEmptyStateVisible" in content
    assert "/tmp/gate-a-smoke-portal" in content
    assert "archive-gate-b" in content
    assert "archive-gate-c" in content
    assert '--archive-command "bag-build"' in content
    assert '--archive-command "mets-export"' in content
    assert "view=operate&job=" in content
    assert "artifact=" in content
    assert "compare=1" in content


def test_portal_browser_smoke_restores_transient_build_draft_after_reload():
    content = PORTAL_BROWSER_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "session-draft" in content
    assert 'build_step="3"' in content
    assert 'str(value.get("activeBuildStep", "")) == "3"' in content
    assert 'connection.call("Page.reload", {"ignoreCache": True})' in content
    assert "transient draft state to restore after reload" in content
    assert 'build_step="1"' in content


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
    assert "artifactViewerVisible" in content
    assert "artifactViewerPath" in content
    assert "artifactViewerFingerprint" in content
    assert "artifactViewerZoomValue" in content
    assert "artifactViewerStatus" in content
    assert "artifactViewerFallbackVisible" in content
    assert "artifactViewerFallbackTitle" in content
    assert "_key_expression" in content
    assert "_inject_viewer_fallback_review_expression" in content
    assert "artifact viewer keyboard next navigation" in content
    assert "artifact viewer keyboard zoom reset" in content
    assert "artifact viewer fallback state for non-previewable artifacts" in content
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
    assert "captioningDetailsVisible" in content
    assert "captioningCliHasFlag" in content
    assert "captioningExpectedOutput" in content
    assert "captioningReadinessText" in content
    assert "captioningReadinessStatus" in content
    assert "path-existence readiness scope" in content
    assert "enableFastVlmCaptioning" in content
    assert "advisoryCaptionPanelVisible" in content
    assert "captioningEvidenceStripVisible" in content
    assert "captioningEvidenceText" in content
    assert "captioningSidecarLinkVisible" in content
    assert "captioningRawLinkVisible" in content
    assert "captioningProxyLinkVisible" in content
    assert "review-primary.png.vlm_captioning.sidecar.json" in content
    assert "review-primary.png.vlm_captioning.raw.txt" in content
    assert "review-primary.png_proxy.png" in content
    assert "FastVLM evidence strip" in content
    assert "dispatch surface to report a blocked preview/governance state" in content
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
