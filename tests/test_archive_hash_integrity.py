"""Regression tests for Phase 3 archive hash integrity tooling."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HASH_TOOL = PROJECT_ROOT / "tools" / "archive_hash_manifest.py"
VERIFY_TOOL = PROJECT_ROOT / "tools" / "verify_hash_manifest.py"
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "archive_small"
FIXTURE_INDEX = FIXTURE_DIR / "archive_index_normalized.csv.gz"
FIXTURE_ARCHIVE_ROOT = FIXTURE_DIR / "archive_root"
FIXTURE_GOLDEN = FIXTURE_DIR / "golden"
HAS_JSONSCHEMA = importlib.util.find_spec("jsonschema") is not None

pytestmark = [pytest.mark.regression]


def _run_cli(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _run_hash_tool(index_path: Path, archive_root: Path, out_dir: Path, *, workers: int = 1, strict: bool = False):
    command = [
        sys.executable,
        str(HASH_TOOL),
        "--archive-index",
        str(index_path),
        "--archive-root",
        str(archive_root),
        "--out-dir",
        str(out_dir),
        "--workers",
        str(workers),
    ]
    if HAS_JSONSCHEMA:
        command.append("--validate-schemas")
    if strict:
        command.append("--strict")
    return _run_cli(command)


def _run_verify_tool(hash_manifest: Path, archive_root: Path, report_path: Path, *, sample: int = 0):
    command = [
        sys.executable,
        str(VERIFY_TOOL),
        "--hash-manifest",
        str(hash_manifest),
        "--archive-root",
        str(archive_root),
        "--report-path",
        str(report_path),
    ]
    if sample > 0:
        command.extend(["--verify-sample", str(sample)])
    else:
        command.append("--verify-all")
    return _run_cli(command)


def _copy_fixture_archive(tmpdir: Path) -> Path:
    target_root = tmpdir / "archive_root"
    shutil.copytree(FIXTURE_ARCHIVE_ROOT, target_root)
    return target_root


def test_phase3_fixture_outputs_match_golden_bytes() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        hash_result = _run_hash_tool(FIXTURE_INDEX, FIXTURE_ARCHIVE_ROOT, out_dir)
        assert hash_result.returncode == 0, hash_result.stderr

        report_path = out_dir / "verification_report.json"
        verify_result = _run_verify_tool(out_dir / "hash_manifest.csv.gz", FIXTURE_ARCHIVE_ROOT, report_path)
        assert verify_result.returncode == 0, verify_result.stderr

        for artifact_name in [
            "hash_manifest.csv.gz",
            "hash_summary.json",
            "merkle_roots.json",
            "verification_report.json",
        ]:
            expected = (FIXTURE_GOLDEN / artifact_name).read_bytes()
            actual = (out_dir / artifact_name).read_bytes()
            assert actual == expected, f"Golden mismatch for {artifact_name}"


def test_hash_outputs_are_deterministic_with_parallel_workers() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        temp = Path(tmpdir)
        out_a = temp / "out_a"
        out_b = temp / "out_b"

        first = _run_hash_tool(FIXTURE_INDEX, FIXTURE_ARCHIVE_ROOT, out_a, workers=2)
        second = _run_hash_tool(FIXTURE_INDEX, FIXTURE_ARCHIVE_ROOT, out_b, workers=2)
        assert first.returncode == 0, first.stderr
        assert second.returncode == 0, second.stderr

        for artifact_name in ["hash_manifest.csv.gz", "hash_summary.json", "merkle_roots.json"]:
            assert (out_a / artifact_name).read_bytes() == (out_b / artifact_name).read_bytes()


def test_verify_detects_single_byte_mutation() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        temp = Path(tmpdir)
        archive_root = _copy_fixture_archive(temp)
        out_dir = temp / "out"

        hash_result = _run_hash_tool(FIXTURE_INDEX, archive_root, out_dir)
        assert hash_result.returncode == 0, hash_result.stderr

        target_file = archive_root / "DriveA" / "Part1" / "alpha.txt"
        with target_file.open("ab") as handle:
            handle.write(b"X")

        report_path = out_dir / "verification_report.json"
        verify_result = _run_verify_tool(out_dir / "hash_manifest.csv.gz", archive_root, report_path)
        assert verify_result.returncode != 0

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["rows_mismatched"] >= 1
        issues = {entry["issue"] for entry in report["mismatches"]}
        assert "sha256_mismatch" in issues


def test_verify_detects_missing_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        temp = Path(tmpdir)
        archive_root = _copy_fixture_archive(temp)
        out_dir = temp / "out"

        hash_result = _run_hash_tool(FIXTURE_INDEX, archive_root, out_dir)
        assert hash_result.returncode == 0, hash_result.stderr

        missing_path = archive_root / "DriveB" / "Part2" / "sub" / "charlie.txt"
        missing_path.unlink()

        report_path = out_dir / "verification_report.json"
        verify_result = _run_verify_tool(out_dir / "hash_manifest.csv.gz", archive_root, report_path)
        assert verify_result.returncode != 0

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["rows_mismatched"] >= 1
        mismatch = next(item for item in report["mismatches"] if item["relpath"].endswith("charlie.txt"))
        assert mismatch["issue"] == "status_mismatch"
        assert mismatch["observed_status"] == "missing"


def test_hash_tool_strict_mode_fails_on_missing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        temp = Path(tmpdir)
        archive_root = _copy_fixture_archive(temp)
        (archive_root / "DriveA" / "Part1" / "bravo.bin").unlink()

        out_dir = temp / "out"
        strict_result = _run_hash_tool(FIXTURE_INDEX, archive_root, out_dir, strict=True)
        assert strict_result.returncode == 2


def test_verify_detects_unreadable_file() -> None:
    if os.name == "nt":
        pytest.skip("chmod-based unreadable test is not stable on Windows")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp = Path(tmpdir)
        archive_root = _copy_fixture_archive(temp)
        out_dir = temp / "out"

        hash_result = _run_hash_tool(FIXTURE_INDEX, archive_root, out_dir)
        assert hash_result.returncode == 0, hash_result.stderr

        unreadable_path = archive_root / "DriveA" / "Part1" / "bravo.bin"
        original_mode = stat.S_IMODE(unreadable_path.stat().st_mode)
        unreadable_path.chmod(0)

        try:
            report_path = out_dir / "verification_report.json"
            verify_result = _run_verify_tool(out_dir / "hash_manifest.csv.gz", archive_root, report_path)
            assert verify_result.returncode != 0

            report = json.loads(report_path.read_text(encoding="utf-8"))
            mismatch = next(item for item in report["mismatches"] if item["relpath"].endswith("bravo.bin"))
            assert mismatch["issue"] == "status_mismatch"
            assert mismatch["observed_status"] == "unreadable"
        finally:
            unreadable_path.chmod(original_mode)
