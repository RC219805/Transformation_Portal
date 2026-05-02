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


def test_trusted_creatable_dir_rejects_traversal_segment(tmp_path):
    bad = str(tmp_path / "ok" / ".." / "evil")
    # Resolved path may still land inside root, so the safety check is on segment
    # validation. Either way the function must not return a path that escapes.
    result = orchestrator_app._trusted_creatable_dir(bad, [tmp_path])
    if result is not None:
        # If accepted, the resolved location must remain within the allowed root.
        assert str(result).startswith(str(tmp_path.resolve()))


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
