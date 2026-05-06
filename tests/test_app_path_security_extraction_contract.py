#!/usr/bin/env python3
"""Phase 2B extraction-readiness contract for app path security helpers."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.security]

orchestrator_app = importlib.import_module("app")


_PHASE_2B_LEGACY_HELPERS = (
    "_normalize_root_path",
    "_default_allowed_path_roots",
    "_env_path_roots",
    "_resolve_untrusted_request_path",
    "_validate_path_against_roots",
    "_resolve_allowed_request_path",
    "_path_is_within_root",
    "_trusted_allowed_entry",
    "_trusted_existing_dir",
    "_trusted_creatable_dir",
    "_ensure_safe_regular_file_path",
    "_resolved_portal_upload_root",
)


def test_phase_2b_legacy_path_security_helpers_remain_available_from_app() -> None:
    for helper_name in _PHASE_2B_LEGACY_HELPERS:
        assert callable(getattr(orchestrator_app, helper_name))


def test_phase_2b_path_reason_codes_stay_distinct(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()

    with pytest.raises(orchestrator_app._PortalValidationReasonError) as invalid_exc:
        orchestrator_app._resolve_allowed_request_path("bad\x00path", [allowed_root])

    outside_path = tmp_path / "outside" / "asset.txt"
    with pytest.raises(orchestrator_app._PortalValidationReasonError) as outside_exc:
        orchestrator_app._resolve_allowed_request_path(str(outside_path), [allowed_root])

    assert invalid_exc.value.reason == "invalid_path_value"
    assert outside_exc.value.reason == "path_outside_allowed_roots"


def test_phase_2b_symlink_escape_is_rejected_for_existing_dir(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    outside_root = tmp_path / "outside"
    allowed_root.mkdir()
    outside_root.mkdir()
    outside_dir = outside_root / "data"
    outside_dir.mkdir()
    symlink_path = allowed_root / "linked-data"
    symlink_path.symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(orchestrator_app._PortalValidationReasonError) as exc_info:
        orchestrator_app._resolve_allowed_request_path(str(symlink_path), [allowed_root])

    assert exc_info.value.reason == "path_outside_allowed_roots"
    assert orchestrator_app._trusted_existing_dir(str(symlink_path), [allowed_root]) is None


def test_phase_2b_safe_regular_file_preserves_outside_root_reason(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    outside_root = tmp_path / "outside"
    allowed_root.mkdir()
    outside_root.mkdir()
    outside_file = outside_root / "asset.png"
    outside_file.write_bytes(b"x")

    with pytest.raises(orchestrator_app._PortalValidationReasonError) as exc_info:
        orchestrator_app._ensure_safe_regular_file_path(outside_file, [allowed_root])

    assert exc_info.value.reason == "path_outside_allowed_roots"


def test_phase_2b_trusted_creatable_dir_returns_path_without_creating_it(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    output_dir = allowed_root / "new-output"

    trusted = orchestrator_app._trusted_creatable_dir(str(output_dir), [allowed_root])

    assert trusted == output_dir
    assert not output_dir.exists()


def test_phase_2b_path_is_within_root_uses_realpath(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    outside_root = tmp_path / "outside"
    allowed_root.mkdir()
    outside_root.mkdir()

    inside = Path(os.path.realpath(allowed_root / "child"))
    outside = Path(os.path.realpath(outside_root / "child"))

    assert orchestrator_app._path_is_within_root(inside, allowed_root) is True
    assert orchestrator_app._path_is_within_root(outside, allowed_root) is False


def test_phase_2b_resolved_portal_upload_root_uses_allowed_input_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    allowed_root = tmp_path / "allowed-input"
    allowed_root.mkdir()
    upload_root = allowed_root / "uploads"

    monkeypatch.setattr(orchestrator_app, "ALLOWED_INPUT_ROOTS", [allowed_root])
    monkeypatch.setattr(orchestrator_app, "PORTAL_UPLOAD_ROOT", upload_root)

    resolved = orchestrator_app._resolved_portal_upload_root()

    assert resolved == Path(os.path.realpath(upload_root))
