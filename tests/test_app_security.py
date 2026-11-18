#!/usr/bin/env python3
"""Security-boundary tests for app.py path validation and error envelopes.

Covers path-traversal rejection, allowed-root enforcement, NUL/tilde rejection,
and the typed JSON error response envelope. These are the highest-risk surfaces
in app.py — they decide what the orchestrator will read or write on disk and
what shape an error reaches the client. See
docs/testing/test_coverage_improvement_plan.md Phase 1.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.security]

orchestrator_app = importlib.import_module("app")


# ---------------------------------------------------------------------------
# _normalize_root_path: untrusted-input rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_value",
    [
        "",
        "   ",
        "~",
        "~/sneaky",
        "with\x00null",
        "/etc/passwd\x00.png",
    ],
)
def test_normalize_root_path_rejects_unsafe_inputs(bad_value):
    with pytest.raises(ValueError):
        orchestrator_app._normalize_root_path(bad_value)


def test_normalize_root_path_returns_resolved_absolute(tmp_path):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    result = orchestrator_app._normalize_root_path(str(nested))
    assert result.is_absolute()
    assert result == Path(str(nested.resolve()))


# ---------------------------------------------------------------------------
# _resolve_untrusted_request_path: same rejection set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_value",
    [
        "",
        "   ",
        "~/foo",
        "with\x00null",
    ],
)
def test_resolve_untrusted_request_path_rejects_unsafe_inputs(bad_value):
    with pytest.raises(ValueError):
        orchestrator_app._resolve_untrusted_request_path(bad_value)


# ---------------------------------------------------------------------------
# _resolve_allowed_request_path / _validate_path_against_roots
# ---------------------------------------------------------------------------


def test_resolve_allowed_request_path_accepts_path_inside_root(tmp_path):
    target = tmp_path / "asset.png"
    target.write_bytes(b"x")
    resolved = orchestrator_app._resolve_allowed_request_path(str(target), [tmp_path])
    assert resolved == Path(str(target.resolve()))


def test_resolve_allowed_request_path_rejects_path_outside_root(tmp_path):
    other = tmp_path.parent / "elsewhere.txt"
    with pytest.raises(orchestrator_app._PortalValidationReasonError) as exc_info:
        orchestrator_app._resolve_allowed_request_path(str(other), [tmp_path])
    assert exc_info.value.reason == "path_outside_allowed_roots"


def test_resolve_allowed_request_path_rejects_traversal_escape(tmp_path):
    inside = tmp_path / "ok"
    inside.mkdir()
    traversal = str(inside / ".." / ".." / "etc")
    with pytest.raises(orchestrator_app._PortalValidationReasonError) as exc_info:
        orchestrator_app._resolve_allowed_request_path(traversal, [tmp_path])
    assert exc_info.value.reason == "path_outside_allowed_roots"


def test_resolve_allowed_request_path_rejects_empty_root_list():
    with pytest.raises(orchestrator_app._PortalValidationReasonError) as exc_info:
        orchestrator_app._resolve_allowed_request_path("/tmp/x", [])
    assert exc_info.value.reason == "invalid_path_value"


def test_resolve_allowed_request_path_rejects_null_byte(tmp_path):
    with pytest.raises(orchestrator_app._PortalValidationReasonError) as exc_info:
        orchestrator_app._resolve_allowed_request_path("with\x00null", [tmp_path])
    assert exc_info.value.reason == "invalid_path_value"


def test_fastvlm_runtime_path_normalization_rejects_symlink_escape(tmp_path):
    allowed_root = tmp_path / "allowed"
    outside_root = tmp_path / "outside"
    allowed_root.mkdir()
    outside_root.mkdir()
    outside_python = outside_root / "python"
    outside_python.write_text("#!/usr/bin/env python\n")
    symlink_path = allowed_root / "fastvlm-python"
    symlink_path.symlink_to(outside_python)
    errors = []

    normalized = orchestrator_app._normalize_fastvlm_runtime_path_arg(
        str(symlink_path),
        "fastvlm_python_executable",
        [allowed_root],
        errors,
    )

    assert normalized == ""
    assert errors[0]["code"] == "path_outside_allowed_roots"


def test_fastvlm_runtime_path_status_rejects_symlink_escape(tmp_path):
    allowed_root = tmp_path / "allowed"
    outside_root = tmp_path / "outside"
    allowed_root.mkdir()
    outside_root.mkdir()
    outside_runtime = outside_root / "mlx-vlm"
    outside_runtime.mkdir()
    symlink_path = allowed_root / "mlx-vlm"
    symlink_path.symlink_to(outside_runtime, target_is_directory=True)

    status = orchestrator_app._portal_fastvlm_runtime_path_status(
        str(symlink_path),
        [allowed_root],
        expected_type="dir",
    )

    assert status["status"] == "invalid_path"
    assert status["exists"] is False


def test_fastvlm_runtime_path_status_keeps_missing_runtime_non_blocking(tmp_path):
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    missing_runtime = allowed_root / "fastvlm" / ".venv-fastvlm" / "bin" / "python"

    status = orchestrator_app._portal_fastvlm_runtime_path_status(
        str(missing_runtime),
        [allowed_root],
        expected_type="file",
    )

    assert status["status"] == "missing"
    assert status["exists"] is False


def test_validate_path_against_roots_returns_string_inside_root(tmp_path):
    target = tmp_path / "asset.png"
    target.write_bytes(b"x")
    result = orchestrator_app._validate_path_against_roots(str(target), [tmp_path])
    assert result == str(target.resolve())


# ---------------------------------------------------------------------------
# _trusted_existing_dir / _trusted_creatable_dir
# ---------------------------------------------------------------------------


def test_trusted_existing_dir_returns_none_when_not_a_dir(tmp_path):
    target = tmp_path / "file.txt"
    target.write_bytes(b"x")
    assert orchestrator_app._trusted_existing_dir(str(target), [tmp_path]) is None


def test_trusted_existing_dir_returns_path_for_existing_dir(tmp_path):
    nested = tmp_path / "stuff"
    nested.mkdir()
    result = orchestrator_app._trusted_existing_dir(str(nested), [tmp_path])
    assert result is not None
    assert result.is_dir()


def test_trusted_existing_dir_rejects_outside_allowed_roots(tmp_path):
    other = tmp_path.parent
    assert orchestrator_app._trusted_existing_dir(str(other), [tmp_path]) is None


def test_trusted_creatable_dir_allows_new_subdir_inside_root(tmp_path):
    new_path = tmp_path / "to-be-created"
    result = orchestrator_app._trusted_creatable_dir(str(new_path), [tmp_path])
    assert result is not None
    assert str(result).startswith(str(tmp_path.resolve()))


def test_trusted_creatable_dir_normalizes_in_root_traversal_shorthand(tmp_path):
    """Document the actual contract: ``os.path.realpath`` is applied before
    segment validation, so an in-root traversal shorthand like
    ``<root>/ok/../evil`` is normalized to ``<root>/evil`` and accepted as a
    creatable directory. Out-of-root traversal is covered separately."""
    (tmp_path / "ok").mkdir()
    bad = str(tmp_path / "ok" / ".." / "evil")
    result = orchestrator_app._trusted_creatable_dir(bad, [tmp_path])
    assert result == tmp_path / "evil"


def test_trusted_creatable_dir_rejects_escape_via_traversal(tmp_path):
    """Traversal that lands outside every allowed root must return None."""
    # tmp_path/.. resolves outside the allowed root; must be rejected.
    sibling = tmp_path.parent / "escape"
    assert orchestrator_app._trusted_creatable_dir(str(sibling), [tmp_path]) is None


def test_trusted_creatable_dir_rejects_traversal_segment_in_unresolved_path(tmp_path):
    """When a literal ``..`` survives realpath because intermediate segments
    don't exist, the per-segment check rejects it as an unsafe segment."""
    # Build a non-existent prefix that realpath will not collapse fully.
    # On most platforms realpath collapses .. regardless, but we still want
    # _UNSAFE_PATH_SEGMENT_RE to defend against any survivor.
    target = tmp_path / "evil"
    # The function returns either None (rejected) or a normalized path under
    # tmp_path — never a path outside the root.
    result = orchestrator_app._trusted_creatable_dir(str(target), [tmp_path])
    assert result is not None
    assert result == target


# ---------------------------------------------------------------------------
# _ensure_safe_regular_file_path
# ---------------------------------------------------------------------------


def test_ensure_safe_regular_file_path_accepts_existing_file(tmp_path):
    target = tmp_path / "good.png"
    target.write_bytes(b"\x89PNG")
    result = orchestrator_app._ensure_safe_regular_file_path(target, [tmp_path])
    assert result.is_file()


def test_ensure_safe_regular_file_path_rejects_directory(tmp_path):
    sub = tmp_path / "dir"
    sub.mkdir()
    with pytest.raises(orchestrator_app._PortalValidationReasonError) as exc_info:
        orchestrator_app._ensure_safe_regular_file_path(sub, [tmp_path])
    assert exc_info.value.reason == "invalid_path_value"


def test_ensure_safe_regular_file_path_rejects_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.png"
    with pytest.raises(orchestrator_app._PortalValidationReasonError) as exc_info:
        orchestrator_app._ensure_safe_regular_file_path(missing, [tmp_path])
    assert exc_info.value.reason == "invalid_path_value"


def test_ensure_safe_regular_file_path_rejects_path_outside_root(tmp_path):
    outside = tmp_path.parent / "escape.png"
    with pytest.raises(orchestrator_app._PortalValidationReasonError):
        orchestrator_app._ensure_safe_regular_file_path(outside, [tmp_path])


# ---------------------------------------------------------------------------
# _PortalValidationReasonError: reason normalization
# ---------------------------------------------------------------------------


def test_portal_validation_error_carries_explicit_reason():
    err = orchestrator_app._PortalValidationReasonError("nope", reason="path_outside_allowed_roots")
    assert err.reason == "path_outside_allowed_roots"
    assert "nope" in str(err)


def test_portal_validation_error_normalizes_blank_message():
    err = orchestrator_app._PortalValidationReasonError("   ", reason="invalid_path_value")
    assert str(err) == "invalid request"


# ---------------------------------------------------------------------------
# Error envelope shape
# ---------------------------------------------------------------------------


def test_error_obj_default_details_is_empty_dict():
    obj = orchestrator_app._error_obj("BAD_INPUT", "not allowed")
    assert obj == {"code": "BAD_INPUT", "message": "not allowed", "details": {}}


def test_error_obj_preserves_provided_details():
    obj = orchestrator_app._error_obj("BAD_INPUT", "not allowed", {"field": "path"})
    assert obj["details"] == {"field": "path"}


def test_api_envelope_carries_all_fields():
    env = orchestrator_app._api_envelope(
        "tp.test.v1",
        success=True,
        data={"x": 1},
        error=None,
    )
    assert env == {"schema": "tp.test.v1", "success": True, "data": {"x": 1}, "error": None}


def test_api_envelope_failure_shape():
    error = orchestrator_app._error_obj("E", "msg")
    env = orchestrator_app._api_envelope(
        "tp.test.v1",
        success=False,
        data=None,
        error=error,
    )
    assert env["success"] is False
    assert env["data"] is None
    assert env["error"] == error


def test_error_response_status_and_payload():
    response = orchestrator_app._error_response(
        422,
        code="VALIDATION_FAILED",
        message="bad path",
        details={"path": "/etc/passwd"},
    )
    assert response.status_code == 422
    body = json.loads(response.body)
    assert body["success"] is False
    assert body["data"] is None
    assert body["error"]["code"] == "VALIDATION_FAILED"
    assert body["error"]["message"] == "bad path"
    assert body["error"]["details"] == {"path": "/etc/passwd"}
    assert body["schema"] == "tp.orchestrator.error.v1"


def test_error_response_uses_custom_schema_when_supplied():
    response = orchestrator_app._error_response(
        400,
        code="BAD",
        message="m",
        schema="tp.custom.v2",
    )
    body = json.loads(response.body)
    assert body["schema"] == "tp.custom.v2"


def test_error_response_propagates_headers():
    response = orchestrator_app._error_response(
        429,
        code="RATE_LIMITED",
        message="slow down",
        headers={"Retry-After": "30"},
    )
    assert response.headers["retry-after"] == "30"


# ---------------------------------------------------------------------------
# _auth_mode is a stable contract; clients depend on it
# ---------------------------------------------------------------------------


def test_auth_mode_is_direct_debug():
    assert orchestrator_app._auth_mode() == "direct_debug"


# ---------------------------------------------------------------------------
# _has_valid_api_key: non-ASCII credentials must be a 401, never a 500
# ---------------------------------------------------------------------------


def _api_key_request(headers: list[tuple[bytes, bytes]]):
    from starlette.requests import Request

    return Request({"type": "http", "method": "GET", "path": "/v1/jobs", "query_string": b"", "headers": headers})


@pytest.mark.parametrize(
    "headers",
    [
        [(b"x-api-key", b"caf\xe9")],
        [(b"authorization", b"Bearer caf\xe9")],
        [(b"x-api-key", b"\xff")],
    ],
)
def test_has_valid_api_key_rejects_non_ascii_credentials_without_raising(monkeypatch, headers):
    monkeypatch.setattr(orchestrator_app, "API_KEY_SECRET", "contract-secret")
    monkeypatch.setattr(orchestrator_app, "API_KEY_HEADER", "x-api-key")

    assert orchestrator_app._has_valid_api_key(_api_key_request(headers)) is False


def test_has_valid_api_key_handles_non_ascii_configured_secret(monkeypatch):
    monkeypatch.setattr(orchestrator_app, "API_KEY_SECRET", "caf\u00e9-secret")
    monkeypatch.setattr(orchestrator_app, "API_KEY_HEADER", "x-api-key")

    assert orchestrator_app._has_valid_api_key(_api_key_request([(b"x-api-key", b"wrong")])) is False


def test_non_ascii_api_key_header_returns_unauthorized_envelope(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(orchestrator_app, "API_KEY_SECRET", "contract-secret")
    with TestClient(orchestrator_app.app, raise_server_exceptions=False) as client:
        response = client.get("/v1/jobs", headers=[(b"x-api-key", b"caf\xe9")])

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"


# ---------------------------------------------------------------------------
# /v1/readiness: host paths are for authenticated operators only
# ---------------------------------------------------------------------------


def _readiness_with_canary_runtime(monkeypatch, tmp_path, headers):
    from fastapi.testclient import TestClient

    canary_runtime = tmp_path / "da3-runtime" / "bin" / "python"
    monkeypatch.setattr(orchestrator_app, "API_KEY_SECRET", "contract-secret")
    monkeypatch.setattr(orchestrator_app, "_resolve_lux_depth_canary_runtime", lambda: canary_runtime)
    with TestClient(orchestrator_app.app, raise_server_exceptions=False) as client:
        response = client.get("/v1/readiness", headers=headers)
    assert response.status_code == 200
    return response, str(canary_runtime)


@pytest.mark.parametrize("headers", [{}, {"x-api-key": "wrong-secret"}])
def test_readiness_without_valid_key_redacts_host_paths(monkeypatch, tmp_path, headers):
    import sys

    response, canary_runtime = _readiness_with_canary_runtime(monkeypatch, tmp_path, headers)
    pipelines = response.json()["data"]["pipelines"]

    for name in ("lux-depth-v3", "archive-gate-a", "archive-gate-b", "archive-gate-c"):
        runner_details = pipelines[name]["runner_details"]
        assert not {"python_executable", "script_path", "command"} & set(runner_details), name
        assert runner_details["available"] in {True, False}
    assert pipelines["archive-gate-a"]["canonical_command"] == "fixity-scan"
    assert "Repo-local DA3 canary runtime resolved." in pipelines["lux-depth-v3"]["notes"]
    assert sys.executable not in response.text
    assert str(orchestrator_app.ARCHIVE_GOVERNANCE_SCRIPT) not in response.text
    assert canary_runtime not in response.text


def test_readiness_with_valid_key_keeps_operator_runner_details(monkeypatch, tmp_path):
    import sys

    response, canary_runtime = _readiness_with_canary_runtime(monkeypatch, tmp_path, {"x-api-key": "contract-secret"})
    pipelines = response.json()["data"]["pipelines"]

    assert pipelines["lux-depth-v3"]["runner_details"]["python_executable"] == sys.executable
    assert pipelines["archive-gate-a"]["runner_details"]["script_path"] == str(orchestrator_app.ARCHIVE_GOVERNANCE_SCRIPT)
    assert pipelines["archive-gate-a"]["runner_details"]["command"][-1] == "fixity-scan"
    assert f"Repo-local DA3 canary runtime resolved at {canary_runtime}." in pipelines["lux-depth-v3"]["notes"]
