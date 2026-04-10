"""Tests for raw_sidecar helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from transformation_portal.ingest import raw_sidecar as raw_sidecar_module

pytestmark = pytest.mark.unit


def test_build_raw_sidecar_payload_sanitizes_volatile_exif_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "frame.dng"
    input_path.write_bytes(b"raw-bytes")

    monkeypatch.setattr(raw_sidecar_module, "_get_exiftool_version", lambda _path: "13.55")
    monkeypatch.setattr(
        raw_sidecar_module,
        "_run_exiftool_json",
        lambda _input_path, _exiftool_path: {
            "File:FileName": "frame.dng",
            "File:FileAccessDate": "volatile",
            "EXIF:ISO": 100,
        },
    )
    monkeypatch.setattr(
        raw_sidecar_module,
        "_read_rawpy_metadata",
        lambda _input_path: (
            {"white_level": 16383},
            {"available": True, "ok": True, "error": None, "version": "0.26.1", "libraw_version": "0.21.4"},
        ),
    )

    payload = raw_sidecar_module.build_raw_sidecar_payload(input_path, exiftool_path="exiftool")

    assert payload["sidecar_schema"] == raw_sidecar_module.RAW_SIDECAR_SCHEMA
    assert payload["metadata_exiftool"]["EXIF:ISO"] == 100
    assert "File:FileAccessDate" not in payload["metadata_exiftool"]
    assert payload["metadata_rawpy"]["white_level"] == 16383


def test_generate_raw_sidecar_writes_json_and_preserves_rawpy_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "frame.cr2"
    input_path.write_bytes(b"raw-bytes")

    monkeypatch.setattr(raw_sidecar_module, "_get_exiftool_version", lambda _path: "13.55")
    monkeypatch.setattr(
        raw_sidecar_module,
        "_run_exiftool_json",
        lambda _input_path, _exiftool_path: {"File:FileName": "frame.cr2", "EXIF:ISO": 200},
    )
    monkeypatch.setattr(
        raw_sidecar_module,
        "_read_rawpy_metadata",
        lambda _input_path: (
            None,
            {"available": False, "ok": False, "error": "rawpy missing", "version": None, "libraw_version": None},
        ),
    )

    output_path = tmp_path / "frame.raw.sidecar.json"
    result = raw_sidecar_module.generate_raw_sidecar(input_path, output_path=output_path, exiftool_path="exiftool")

    assert result.output_path == output_path
    assert result.rawpy_available is False
    assert result.rawpy_ok is False
    assert result.rawpy_error == "rawpy missing"

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["capture_status"]["rawpy"]["error"] == "rawpy missing"
    assert payload["metadata_rawpy"] is None


def test_generate_raw_sidecar_uses_existing_ingest_json_style(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "frame.cr2"
    input_path.write_bytes(b"raw-bytes")

    monkeypatch.setattr(raw_sidecar_module, "_get_exiftool_version", lambda _path: "13.55")
    monkeypatch.setattr(
        raw_sidecar_module,
        "_run_exiftool_json",
        lambda _input_path, _exiftool_path: {"EXIF:LensModel": "Café Lens"},
    )
    monkeypatch.setattr(
        raw_sidecar_module,
        "_read_rawpy_metadata",
        lambda _input_path: (
            None,
            {"available": False, "ok": False, "error": "rawpy missing", "version": None, "libraw_version": None},
        ),
    )

    output_path = tmp_path / "frame.raw.sidecar.json"
    raw_sidecar_module.generate_raw_sidecar(input_path, output_path=output_path, exiftool_path="exiftool")

    text = output_path.read_text(encoding="utf-8")
    assert "Café Lens" in text
    assert "\\u00e9" not in text
    assert not text.endswith("\n")


def test_build_raw_sidecar_payload_uses_precomputed_file_integrity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "frame.dng"
    input_path.write_bytes(b"raw-bytes")

    monkeypatch.setattr(raw_sidecar_module, "_get_exiftool_version", lambda _path: "13.55")
    monkeypatch.setattr(
        raw_sidecar_module,
        "_run_exiftool_json",
        lambda _input_path, _exiftool_path: {"EXIF:ISO": 100},
    )
    monkeypatch.setattr(
        raw_sidecar_module,
        "_read_rawpy_metadata",
        lambda _input_path: (
            None,
            {"available": False, "ok": False, "error": "rawpy missing", "version": None, "libraw_version": None},
        ),
    )

    payload = raw_sidecar_module.build_raw_sidecar_payload(
        input_path,
        exiftool_path="exiftool",
        file_size_bytes=4321,
        file_sha256="b" * 64,
    )

    assert payload["file"]["size_bytes"] == 4321
    assert payload["file"]["sha256"] == "b" * 64


def test_generate_raw_sidecar_propagates_exiftool_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "frame.cr2"
    input_path.write_bytes(b"raw-bytes")

    monkeypatch.setattr(raw_sidecar_module, "_get_exiftool_version", lambda _path: "13.55")
    monkeypatch.setattr(
        raw_sidecar_module,
        "_run_exiftool_json",
        lambda _input_path, _exiftool_path: (_ for _ in ()).throw(subprocess.TimeoutExpired("exiftool", 30)),
    )

    with pytest.raises(subprocess.TimeoutExpired):
        raw_sidecar_module.generate_raw_sidecar(
            input_path,
            output_path=tmp_path / "frame.raw.sidecar.json",
            exiftool_path="exiftool",
        )
