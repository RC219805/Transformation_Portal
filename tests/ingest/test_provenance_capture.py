"""Regression tests for ingest provenance capture helpers."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

from transformation_portal.ingest.provenance import _capture_toolchain_versions


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
