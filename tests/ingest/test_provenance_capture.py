"""Regression tests for ingest provenance capture helpers."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from transformation_portal.ingest import provenance as ingest_provenance
from transformation_portal.ingest.provenance import _capture_toolchain_versions

pytestmark = pytest.mark.unit


def test_capture_toolchain_versions_handles_rawpy_without_version_attr(monkeypatch) -> None:
    """rawpy builds without version.version should not crash extraction."""
    dummy_rawpy = types.ModuleType("rawpy")
    dummy_rawpy.libraw_version = (0, 21, 2)
    monkeypatch.setitem(sys.modules, "rawpy", dummy_rawpy)

    def fake_run(*_args, **_kwargs):
        return MagicMock(returncode=1, stdout="", stderr="")

    monkeypatch.setattr("transformation_portal.ingest.provenance.subprocess.run", fake_run)

    versions = _capture_toolchain_versions()
    by_name = {item.name: item.version for item in versions}

    assert by_name["rawpy"] == "unknown"
    assert by_name["libraw"] == "0.21.2"
    assert "python" in by_name


@pytest.mark.parametrize(
    "error",
    [
        PermissionError(13, "Permission denied", "exiftool"),
        NotADirectoryError(20, "Not a directory", "exiftool"),
        OSError(8, "Exec format error", "exiftool"),
    ],
    ids=["unsearchable-path-entry", "path-entry-is-a-file", "exec-format-error"],
)
def test_optional_tool_probes_treat_any_os_error_as_unavailable(monkeypatch, error: OSError) -> None:
    """A tool the process cannot execute is "unavailable", never a crash.

    ``subprocess.run`` raises ``PermissionError`` (not ``FileNotFoundError``)
    for a missing tool when PATH contains a directory the service account
    cannot search, so probes must degrade on every ``OSError``.
    """

    def fake_run(*_args, **_kwargs):
        raise error

    monkeypatch.setattr("transformation_portal.ingest.provenance.subprocess.run", fake_run)
    monkeypatch.delitem(sys.modules, "rawpy", raising=False)
    monkeypatch.setitem(sys.modules, "rawpy", None)

    assert ingest_provenance._check_exiftool_available() is False
    assert ingest_provenance._get_exiftool_version() is None
    assert ingest_provenance._get_git_commit() is None

    by_name = {item.name: item.version for item in _capture_toolchain_versions()}
    assert "exiftool" not in by_name
    assert "ImageMagick" not in by_name
    assert "python" in by_name
