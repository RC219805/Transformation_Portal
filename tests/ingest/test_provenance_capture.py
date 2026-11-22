"""Regression tests for ingest provenance capture helpers."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from transformation_portal.ingest import provenance as ingest_provenance
from transformation_portal.ingest.provenance import _capture_toolchain_versions
from transformation_portal.ingest.schemas import ExifMetadata

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


@pytest.mark.parametrize(
    ("file_path", "expected_argument"),
    [
        (Path("-x.jpg"), "./-x.jpg"),
        (Path("-sub") / "y.jpg", "./-sub/y.jpg"),
        (Path("frame.jpg"), "frame.jpg"),
        (Path("dir") / "-z.jpg", "dir/-z.jpg"),
        (Path("/abs/-w.jpg"), "/abs/-w.jpg"),
    ],
)
def test_extract_exif_never_passes_a_dash_prefixed_file_argument(monkeypatch, file_path, expected_argument) -> None:
    """D-13: a bare ``-x.jpg`` (input dir ``.``) must not reach exiftool as an option.

    Only names that would parse as an option are rewritten; other paths are
    passed unchanged so ``SourceFile`` and the sidecar bytes do not move.
    """
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        return MagicMock(returncode=0, stdout='[{"SourceFile": "x"}]', stderr="")

    monkeypatch.setattr(ingest_provenance, "_check_exiftool_available", lambda: True)
    monkeypatch.setattr("transformation_portal.ingest.provenance.subprocess.run", fake_run)

    assert ingest_provenance._extract_exif_with_exiftool(file_path) == {"SourceFile": "x"}
    assert calls[0][-1] == expected_argument
    assert not any(argument.startswith("-") and argument.endswith(".jpg") for argument in calls[0])


def test_parse_exif_fields_restores_text_for_numeric_exiftool_values() -> None:
    """D-14: exiftool JSON prints numeric-looking strings as numbers (a model named "6300").

    The text form is restored where the exiftool payload is parsed, so the
    extract no longer fails, while ``all_tags`` keeps the raw values and the
    parsed fields satisfy the strict sidecar schema.
    """
    tags = {
        "EXIF:Make": 1234,
        "EXIF:Model": 6300,
        "EXIF:LensModel": 50,
        "EXIF:ShutterSpeed": 0.004,
        "EXIF:WhiteBalance": 0,
        "EXIF:ColorSpace": 1,
        "EXIF:DateTimeOriginal": 2026,
    }

    exif = ingest_provenance._parse_exif_fields(dict(tags))

    assert (
        exif.camera_make,
        exif.camera_model,
        exif.lens_model,
        exif.shutter_speed,
        exif.white_balance,
        exif.color_space,
        exif.datetime_original,
    ) == ("1234", "6300", "50", "0.004", "0", "1", "2026")
    assert exif.all_tags == tags
    assert type(exif.all_tags["EXIF:Model"]) is int
    assert ExifMetadata(**exif.model_dump()) == exif


def test_parse_exif_fields_keeps_text_values_and_rejects_booleans() -> None:
    tags = {
        "EXIF:Make": "Canon",
        "EXIF:Model": "Canon EOS R5",
        "EXIF:ShutterSpeed": "1/250",
        "EXIF:DateTimeOriginal": "2026:01:02 03:04:05",
    }

    exif = ingest_provenance._parse_exif_fields(dict(tags))

    assert (
        exif.model_dump()
        == ExifMetadata(
            all_tags=tags,
            camera_make="Canon",
            camera_model="Canon EOS R5",
            shutter_speed="1/250",
            datetime_original="2026:01:02 03:04:05",
        ).model_dump()
    )
    with pytest.raises(ValueError, match="valid string"):
        ingest_provenance._parse_exif_fields({"EXIF:Model": True})
