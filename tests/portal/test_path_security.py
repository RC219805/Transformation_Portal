"""Unit tests for portal path security helpers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from transformation_portal.portal import path_security
from transformation_portal.portal.path_security import PathSecurityValidationError, _resolve_allowed_request_path

pytestmark = [pytest.mark.unit, pytest.mark.security]


def test_path_security_import_does_not_import_app() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from transformation_portal.portal import path_security; print('app' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_direct_helper_import_resolves_allowed_path(tmp_path: Path) -> None:
    target = tmp_path / "asset.png"
    target.write_bytes(b"x")

    assert _resolve_allowed_request_path(str(target), [tmp_path]) == Path(os.path.realpath(target))


def test_path_security_error_reasons_stay_distinct(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()

    with pytest.raises(PathSecurityValidationError) as invalid_exc:
        path_security._resolve_allowed_request_path("bad\x00path", [allowed_root])

    outside_path = tmp_path / "outside" / "asset.txt"
    with pytest.raises(PathSecurityValidationError) as outside_exc:
        path_security._resolve_allowed_request_path(str(outside_path), [allowed_root])

    assert invalid_exc.value.reason == "invalid_path_value"
    assert outside_exc.value.reason == "path_outside_allowed_roots"


def test_path_security_rejects_symlink_escape_for_existing_dir(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    outside_root = tmp_path / "outside"
    allowed_root.mkdir()
    outside_root.mkdir()
    outside_dir = outside_root / "data"
    outside_dir.mkdir()
    symlink_path = allowed_root / "linked-data"
    symlink_path.symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(PathSecurityValidationError) as exc_info:
        path_security._resolve_allowed_request_path(str(symlink_path), [allowed_root])

    assert exc_info.value.reason == "path_outside_allowed_roots"
    assert path_security._trusted_existing_dir(str(symlink_path), [allowed_root]) is None


def test_path_security_existing_dir_checks_match_app_contract(tmp_path: Path) -> None:
    existing_dir = tmp_path / "dir"
    existing_dir.mkdir()
    existing_file = tmp_path / "file.txt"
    existing_file.write_text("x", encoding="utf-8")

    assert path_security._trusted_existing_dir(str(existing_dir), [tmp_path]) == existing_dir
    assert path_security._trusted_existing_dir(str(existing_file), [tmp_path]) is None
    assert path_security._trusted_existing_dir(str(tmp_path.parent), [tmp_path]) is None


def test_path_security_creatable_dir_returns_path_without_creating_it(tmp_path: Path) -> None:
    output_dir = tmp_path / "new-output"

    trusted = path_security._trusted_creatable_dir(str(output_dir), [tmp_path])

    assert trusted == output_dir
    assert not output_dir.exists()


def test_path_security_safe_regular_file_validation(tmp_path: Path) -> None:
    target = tmp_path / "good.png"
    target.write_bytes(b"\x89PNG")
    subdir = tmp_path / "dir"
    subdir.mkdir()
    missing = tmp_path / "missing.png"
    outside = tmp_path.parent / "outside.png"

    assert path_security._ensure_safe_regular_file_path(target, [tmp_path]) == target

    with pytest.raises(PathSecurityValidationError) as dir_exc:
        path_security._ensure_safe_regular_file_path(subdir, [tmp_path])
    with pytest.raises(PathSecurityValidationError) as missing_exc:
        path_security._ensure_safe_regular_file_path(missing, [tmp_path])
    with pytest.raises(PathSecurityValidationError) as outside_exc:
        path_security._ensure_safe_regular_file_path(outside, [tmp_path])

    assert dir_exc.value.reason == "invalid_path_value"
    assert missing_exc.value.reason == "invalid_path_value"
    assert outside_exc.value.reason == "path_outside_allowed_roots"
