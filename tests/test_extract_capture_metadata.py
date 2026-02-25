"""Tests for Phase 4C deterministic capture metadata extraction."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tp.phase4.canonicalize_capture_metadata import (
    PathNormalizationError,
    normalize_relative_path,
    write_capture_metadata_artifact,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = PROJECT_ROOT / "tools" / "extract_capture_metadata.py"
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "phase4" / "metadata.schema.json"
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "phase4"
GOLDEN_OUTPUT = PROJECT_ROOT / "tests" / "golden" / "phase4" / "expected_capture_metadata.tp.meta.capture.v1.json"
GOLDEN_CONFIG_FINGERPRINT = PROJECT_ROOT / "tests" / "golden" / "phase4" / "config_fingerprint.txt"

pytestmark = [pytest.mark.regression, pytest.mark.golden]


def _build_fake_exiftool(tmp_path: Path, *, mode: str) -> Path:
    fake_exiftool_path = tmp_path / "exiftool"
    script = f"""#!/usr/bin/env python3
import json
import sys

mode = {mode!r}
files = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
records = []

for source_file in files:
    record = {{
        "SourceFile": source_file,
    }}
    if mode == "valid":
        record.update({{
            "Make": " Canon ",
            "Model": "EOS R5",
            "LensModel": "RF24-70mm F2.8 L IS USM",
            "GPSDateStamp": "2024:06:30",
            "GPSTimeStamp": "12:34:56",
            "GPSLatitude": 34.123456789,
            "GPSLongitude": -118.987654321,
            "FocalLength": 24.98765,
            "FNumber": 5.6789,
            "ExposureTime": "1/120",
            "ExposureCompensation": "-0.3333",
            "Orientation": 6,
            "DateTimeOriginal": "2024:06:30 05:34:56",
            "OffsetTimeOriginal": "-07:00"
        }})
    elif mode == "gps-invalid-datetime":
        record.update({{
            "Make": "Canon",
            "Model": "EOS R5",
            "LensModel": "RF24-70mm F2.8 L IS USM",
            "GPSDateStamp": "2024:02:30",
            "GPSTimeStamp": "25:00:00",
            "DateTimeOriginal": "2024:06:30 05:34:56",
            "OffsetTimeOriginal": "-07:00"
        }})
    else:
        record.update({{
            "Make": "Canon",
            "Model": "EOS R5",
            "LensModel": "RF24-70mm F2.8 L IS USM",
            "DateTimeOriginal": "2024:06:30 05:34:56",
            "GPSLatitude": "invalid-gps",
            "GPSLongitude": -118.987654321,
            "FocalLength": 24.98765,
            "FNumber": 5.6789,
            "ExposureTime": "1/120",
            "ExposureCompensation": "-0.3333",
            "Orientation": 99
        }})
    records.append(record)

sys.stdout.write(json.dumps(records))
"""
    fake_exiftool_path.write_text(script, encoding="utf-8")
    fake_exiftool_path.chmod(0o755)
    return fake_exiftool_path


def _run_cli(
    *,
    input_root: Path,
    out_path: Path,
    fake_exiftool: Path | None,
    config_path: Path | None = None,
    strict: bool = False,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if fake_exiftool is not None:
        env["PATH"] = f"{fake_exiftool.parent}:{env.get('PATH', '')}"
    if env_overrides:
        env.update(env_overrides)
    command = [
        sys.executable,
        str(CLI_PATH),
        "--input-root",
        str(input_root),
        "--out",
        str(out_path),
    ]
    if config_path is not None:
        command.extend(["--config", str(config_path)])
    if strict:
        command.append("--strict")

    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_phase4c_cli_help_works_without_pythonpath() -> None:
    result = subprocess.run(
        [sys.executable, str(CLI_PATH), "--help"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": ""},
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_phase4c_writer_emits_single_trailing_newline(tmp_path: Path) -> None:
    out_path = tmp_path / "capture_metadata.tp.meta.capture.v1.json"
    write_capture_metadata_artifact([{"relative_path": "sample_01.dng"}], out_path=out_path)
    payload = out_path.read_bytes()
    assert payload.endswith(b"\n")
    assert not payload.endswith(b"\n\n")


def test_phase4c_golden_output_matches_expected(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    fake_exiftool = _build_fake_exiftool(tmp_path, mode="valid")
    out_path = tmp_path / "capture_metadata.tp.meta.capture.v1.json"
    result = _run_cli(input_root=FIXTURE_ROOT, out_path=out_path, fake_exiftool=fake_exiftool)
    assert result.returncode == 0, result.stderr
    assert out_path.read_bytes() == GOLDEN_OUTPUT.read_bytes()


def test_phase4c_output_is_byte_deterministic_across_runs(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    fake_exiftool = _build_fake_exiftool(tmp_path, mode="valid")
    out_a = tmp_path / "run_a.json"
    out_b = tmp_path / "run_b.json"
    first = _run_cli(input_root=FIXTURE_ROOT, out_path=out_a, fake_exiftool=fake_exiftool)
    second = _run_cli(input_root=FIXTURE_ROOT, out_path=out_b, fake_exiftool=fake_exiftool)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert out_a.read_bytes() == out_b.read_bytes()


def test_phase4c_output_validates_against_metadata_schema(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    fake_exiftool = _build_fake_exiftool(tmp_path, mode="valid")
    out_path = tmp_path / "capture_metadata.tp.meta.capture.v1.json"
    result = _run_cli(input_root=FIXTURE_ROOT, out_path=out_path, fake_exiftool=fake_exiftool)
    assert result.returncode == 0, result.stderr

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    for record in payload:
        validator.validate(record)


def test_phase4c_records_embed_config_fingerprint() -> None:
    expected_fingerprint = GOLDEN_CONFIG_FINGERPRINT.read_text(encoding="utf-8").strip()
    payload = json.loads(GOLDEN_OUTPUT.read_text(encoding="utf-8"))
    assert payload
    for record in payload:
        assert record["extractor"]["config_fingerprint_sha256"] == expected_fingerprint


def test_phase4c_path_normalization_rejects_unsafe_inputs() -> None:
    with pytest.raises(PathNormalizationError):
        normalize_relative_path("/absolute/path/sample_01.dng")
    with pytest.raises(PathNormalizationError):
        normalize_relative_path("../sample_01.dng")
    with pytest.raises(PathNormalizationError):
        normalize_relative_path("folder\\sample_01.dng")
    with pytest.raises(PathNormalizationError):
        normalize_relative_path("./sample_01.dng")
    with pytest.raises(PathNormalizationError):
        normalize_relative_path("C:\\capture\\sample_01.dng")


@pytest.mark.skipif(os.name == "nt", reason="backslash filename semantics differ on Windows")
def test_phase4c_cli_fails_deterministically_on_backslash_relative_path(tmp_path: Path) -> None:
    fake_exiftool = _build_fake_exiftool(tmp_path, mode="valid")
    input_root = tmp_path / "input"
    input_root.mkdir(parents=True, exist_ok=True)
    (input_root / "bad\\name.dng").write_bytes(b"phase4-path-safety")

    out_path = tmp_path / "capture_metadata.tp.meta.capture.v1.json"
    result = _run_cli(input_root=input_root, out_path=out_path, fake_exiftool=fake_exiftool)
    assert result.returncode == 3
    assert "Path normalization failure:" in result.stderr


def test_phase4c_warning_codes_are_stable_sorted_and_unique(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    fake_exiftool = _build_fake_exiftool(tmp_path, mode="warning")
    out_path = tmp_path / "capture_metadata.tp.meta.capture.v1.json"
    first = _run_cli(input_root=FIXTURE_ROOT, out_path=out_path, fake_exiftool=fake_exiftool)
    assert first.returncode == 0, first.stderr
    warnings_a = json.loads(out_path.read_text(encoding="utf-8"))[0]["extraction_warnings"]

    second_out = tmp_path / "capture_metadata_second.tp.meta.capture.v1.json"
    second = _run_cli(input_root=FIXTURE_ROOT, out_path=second_out, fake_exiftool=fake_exiftool)
    assert second.returncode == 0, second.stderr
    warnings_b = json.loads(second_out.read_text(encoding="utf-8"))[0]["extraction_warnings"]

    assert warnings_a == warnings_b
    assert warnings_a == sorted(set(warnings_a))
    assert "WARN_DATETIME_NO_TZ" in warnings_a
    assert "WARN_GPS_PARSE_FAIL" in warnings_a
    assert "WARN_INVALID_ORIENTATION" in warnings_a


def test_phase4c_invalid_gps_datetime_warns_and_uses_fallback(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    fake_exiftool = _build_fake_exiftool(tmp_path, mode="gps-invalid-datetime")
    out_path = tmp_path / "capture_metadata.tp.meta.capture.v1.json"
    result = _run_cli(input_root=FIXTURE_ROOT, out_path=out_path, fake_exiftool=fake_exiftool)
    assert result.returncode == 0, result.stderr

    record = json.loads(out_path.read_text(encoding="utf-8"))[0]
    assert record["capture_datetime_utc"] == "2024-06-30T12:34:56Z"
    assert "WARN_GPS_PARSE_FAIL" in record["extraction_warnings"]


def test_phase4c_strict_mode_fails_on_warnings(tmp_path: Path) -> None:
    fake_exiftool = _build_fake_exiftool(tmp_path, mode="warning")
    out_path = tmp_path / "capture_metadata.tp.meta.capture.v1.json"
    result = _run_cli(input_root=FIXTURE_ROOT, out_path=out_path, fake_exiftool=fake_exiftool, strict=True)
    assert result.returncode == 6
    assert "Strict-mode warning failure:" in result.stderr
    assert not out_path.exists()


def test_phase4c_cli_returns_exit_code_2_for_invalid_config(tmp_path: Path) -> None:
    fake_exiftool = _build_fake_exiftool(tmp_path, mode="valid")
    invalid_config = tmp_path / "invalid_config.json"
    invalid_config.write_text('{"metadata_contract_version":"tp.meta.capture.v1"}', encoding="utf-8")
    out_path = tmp_path / "capture_metadata.tp.meta.capture.v1.json"
    result = _run_cli(
        input_root=FIXTURE_ROOT,
        out_path=out_path,
        fake_exiftool=fake_exiftool,
        config_path=invalid_config,
    )
    assert result.returncode == 2
    assert "Config invalid:" in result.stderr


def test_phase4c_cli_returns_exit_code_2_for_invalid_nested_config(tmp_path: Path) -> None:
    fake_exiftool = _build_fake_exiftool(tmp_path, mode="valid")
    payload = json.loads((PROJECT_ROOT / "tools" / "capture_metadata_config.json").read_text(encoding="utf-8"))
    del payload["rounding_rules"]["rounding_mode"]
    invalid_config = tmp_path / "invalid_nested_config.json"
    invalid_config.write_text(json.dumps(payload), encoding="utf-8")

    out_path = tmp_path / "capture_metadata.tp.meta.capture.v1.json"
    result = _run_cli(
        input_root=FIXTURE_ROOT,
        out_path=out_path,
        fake_exiftool=fake_exiftool,
        config_path=invalid_config,
    )
    assert result.returncode == 2
    assert "Config invalid:" in result.stderr


def test_phase4c_cli_returns_exit_code_4_when_exiftool_missing(tmp_path: Path) -> None:
    out_path = tmp_path / "capture_metadata.tp.meta.capture.v1.json"
    result = _run_cli(
        input_root=FIXTURE_ROOT,
        out_path=out_path,
        fake_exiftool=None,
        env_overrides={"PATH": ""},
    )
    assert result.returncode == 4
    assert "Extraction failure:" in result.stderr
