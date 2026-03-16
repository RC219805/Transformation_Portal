"""CLI regression tests for tools/archive_bagit.py."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BAGIT_TOOL = PROJECT_ROOT / "tools" / "archive_bagit.py"


def _run_bagit(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(BAGIT_TOOL), *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_manifest_jsonl(path: Path, entries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_fixture_bag(tmp_path: Path) -> tuple[Path, Path]:
    archive_root = tmp_path / "archive_root"
    bag_dir = tmp_path / "bag"
    manifest_path = tmp_path / "archive_manifest_v2.jsonl"
    report_json = tmp_path / "build_report.json"

    payload_a = b"alpha payload\n"
    payload_b = b"bravo payload"

    (archive_root / "set1").mkdir(parents=True, exist_ok=True)
    (archive_root / "set1" / "alpha.bin").write_bytes(payload_a)
    (archive_root / "set1" / "bravo.bin").write_bytes(payload_b)

    entries = [
        {
            "relpath": "set1/alpha.bin",
            "sha256": _sha256_bytes(payload_a),
            "hash_status": "ok",
            "modified_utc": "2024-03-01T00:00:00Z",
        },
        {
            "relpath": "set1/bravo.bin",
            "sha256": _sha256_bytes(payload_b),
            "hash_status": "ok",
            "modified_utc": "2024-03-02T00:00:00Z",
        },
        {
            "relpath": "set1/skipped.bin",
            "sha256": "",
            "hash_status": "missing",
        },
    ]
    _write_manifest_jsonl(manifest_path, entries)

    result = _run_bagit(
        "build",
        "--manifest-jsonl",
        str(manifest_path),
        "--archive-root",
        str(archive_root),
        "--bag-dir",
        str(bag_dir),
        "--report-json",
        str(report_json),
        "--source-organization",
        "TEST_ORG",
    )
    assert result.returncode == 0, result.stderr
    return bag_dir, report_json


def test_archive_bagit_build_writes_expected_manifests_and_payload_oxum(tmp_path: Path) -> None:
    bag_dir, report_json = _build_fixture_bag(tmp_path)

    payload_manifest_path = bag_dir / "manifest-sha256.txt"
    tag_manifest_path = bag_dir / "tagmanifest-sha256.txt"
    bag_info_path = bag_dir / "bag-info.txt"
    assert payload_manifest_path.exists()
    assert tag_manifest_path.exists()
    assert bag_info_path.exists()

    payload_manifest_lines = payload_manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(payload_manifest_lines) == 2
    assert payload_manifest_lines[0].endswith("  data/set1/alpha.bin")
    assert payload_manifest_lines[1].endswith("  data/set1/bravo.bin")

    payload_size = (bag_dir / "data" / "set1" / "alpha.bin").stat().st_size + (
        bag_dir / "data" / "set1" / "bravo.bin"
    ).stat().st_size
    bag_info_text = bag_info_path.read_text(encoding="utf-8")
    assert f"Payload-Oxum: {payload_size}.2" in bag_info_text

    report = _read_json(report_json)
    assert report["copied_files"] == 2
    assert report["payload_bytes"] == payload_size
    assert report["payload_oxum"] == f"{payload_size}.2"

    tag_manifest_lines = tag_manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(tag_manifest_lines) == 3
    assert tag_manifest_lines[0].endswith("  bagit.txt")
    assert tag_manifest_lines[1].endswith("  bag-info.txt")
    assert tag_manifest_lines[2].endswith("  manifest-sha256.txt")


def test_archive_bagit_validate_reports_sha256_mismatch_after_payload_mutation(tmp_path: Path) -> None:
    bag_dir, _ = _build_fixture_bag(tmp_path)
    report_json = tmp_path / "validate_report.json"

    payload_file = bag_dir / "data" / "set1" / "alpha.bin"
    payload_file.write_bytes(payload_file.read_bytes() + b"x")

    result = _run_bagit(
        "validate",
        "--bag-dir",
        str(bag_dir),
        "--report-json",
        str(report_json),
    )
    assert result.returncode == 4

    report = _read_json(report_json)
    assert report["valid"] is False
    mismatch_issues = {(item["scope"], item["path"], item["issue"]) for item in report["mismatches"]}
    assert ("payload", "data/set1/alpha.bin", "sha256_mismatch") in mismatch_issues


def test_archive_bagit_build_fails_when_payload_missing(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive_root"
    bag_dir = tmp_path / "bag"
    manifest_path = tmp_path / "manifest.jsonl"
    archive_root.mkdir(parents=True, exist_ok=True)

    _write_manifest_jsonl(
        manifest_path,
        [
            {
                "relpath": "missing/file.bin",
                "sha256": "a" * 64,
                "hash_status": "ok",
            }
        ],
    )

    result = _run_bagit(
        "build",
        "--manifest-jsonl",
        str(manifest_path),
        "--archive-root",
        str(archive_root),
        "--bag-dir",
        str(bag_dir),
    )
    assert result.returncode == 3
    assert "source payload missing" in result.stderr


def test_archive_bagit_build_rejects_non_empty_bag_dir(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive_root"
    bag_dir = tmp_path / "bag"
    manifest_path = tmp_path / "manifest.jsonl"
    archive_root.mkdir(parents=True, exist_ok=True)
    (archive_root / "ok.bin").write_bytes(b"ok")

    bag_dir.mkdir(parents=True, exist_ok=True)
    (bag_dir / "stale.txt").write_text("stale", encoding="utf-8")

    _write_manifest_jsonl(
        manifest_path,
        [
            {
                "relpath": "ok.bin",
                "sha256": _sha256_bytes(b"ok"),
                "hash_status": "ok",
            }
        ],
    )

    result = _run_bagit(
        "build",
        "--manifest-jsonl",
        str(manifest_path),
        "--archive-root",
        str(archive_root),
        "--bag-dir",
        str(bag_dir),
    )
    assert result.returncode == 3
    assert "bag_dir must be empty before build" in result.stderr


def test_archive_bagit_validate_rejects_manifest_path_traversal(tmp_path: Path) -> None:
    bag_dir, _ = _build_fixture_bag(tmp_path)
    report_json = tmp_path / "validate_report.json"

    payload_manifest = bag_dir / "manifest-sha256.txt"
    payload_manifest.write_text(f"{'a' * 64}  ../outside.bin\n", encoding="utf-8")

    result = _run_bagit(
        "validate",
        "--bag-dir",
        str(bag_dir),
        "--report-json",
        str(report_json),
    )
    assert result.returncode == 4

    report = _read_json(report_json)
    mismatch_issues = {(item["scope"], item["path"], item["issue"]) for item in report["mismatches"]}
    assert ("payload", "../outside.bin", "invalid_relpath_parent_ref") in mismatch_issues


def test_archive_bagit_build_rejects_symlink_payload(tmp_path: Path) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlink is not supported on this platform")

    archive_root = tmp_path / "archive_root"
    outside_root = tmp_path / "outside"
    manifest_path = tmp_path / "manifest.jsonl"
    bag_dir = tmp_path / "bag"

    archive_root.mkdir(parents=True, exist_ok=True)
    outside_root.mkdir(parents=True, exist_ok=True)
    outside_payload = b"outside-by-symlink"
    (outside_root / "outside.bin").write_bytes(outside_payload)
    os.symlink(outside_root / "outside.bin", archive_root / "link.bin")

    _write_manifest_jsonl(
        manifest_path,
        [
            {
                "relpath": "link.bin",
                "sha256": _sha256_bytes(outside_payload),
                "hash_status": "ok",
            }
        ],
    )

    result = _run_bagit(
        "build",
        "--manifest-jsonl",
        str(manifest_path),
        "--archive-root",
        str(archive_root),
        "--bag-dir",
        str(bag_dir),
    )
    assert result.returncode == 3
    assert "symlink_skipped" in result.stderr
