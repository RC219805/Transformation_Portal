"""Regression tests for Phase 3 archive hash integrity tooling."""

from __future__ import annotations

import csv
import gzip
import hashlib
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


def _load_tool_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ARCHIVE_HASH_MODULE = _load_tool_module(HASH_TOOL, "tests_archive_hash_manifest_tool")
VERIFY_HASH_MODULE = _load_tool_module(VERIFY_TOOL, "tests_verify_hash_manifest_tool")
RELPATH_REJECTION_CASES = [
    ("C:\\foo\\bar.txt", "invalid_relpath_drive_spec"),
    ("C:foo\\bar.txt", "invalid_relpath_drive_spec"),
    ("/tmp/escape.txt", "invalid_relpath_anchored"),
    ("\\\\server\\share\\file.txt", "invalid_relpath_anchored"),
    ("DriveA/Part1/alpha\x00.txt", "invalid_relpath_nul"),
]


def _run_cli(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _run_hash_tool(
    index_path: Path,
    archive_root: Path,
    out_dir: Path,
    *,
    workers: int = 1,
    strict: bool = False,
    strict_identity: bool = False,
):
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
    if strict_identity:
        command.append("--strict-identity")
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


def test_hash_tool_missing_archive_index_fails_without_traceback_or_output_dir(tmp_path: Path) -> None:
    archive_root = _copy_fixture_archive(tmp_path)
    out_dir = tmp_path / "out"

    result = _run_cli(
        [
            sys.executable,
            str(HASH_TOOL),
            "--archive-index",
            str(tmp_path / "missing_archive_index.csv.gz"),
            "--archive-root",
            str(archive_root),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert result.returncode == ARCHIVE_HASH_MODULE.EXIT_INPUT_ERROR
    assert "Error: archive index not found:" in result.stderr
    assert "Traceback (most recent call last):" not in result.stderr
    assert not out_dir.exists()


def test_hash_tool_blank_archive_index_reports_required_message(tmp_path: Path) -> None:
    archive_root = _copy_fixture_archive(tmp_path)
    out_dir = tmp_path / "out"

    result = _run_cli(
        [
            sys.executable,
            str(HASH_TOOL),
            "--archive-index",
            "   ",
            "--archive-root",
            str(archive_root),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert result.returncode == ARCHIVE_HASH_MODULE.EXIT_INPUT_ERROR
    assert "Error: archive index is required" in result.stderr
    assert "Traceback (most recent call last):" not in result.stderr
    assert not out_dir.exists()


def test_hash_tool_invalid_archive_root_fails_without_traceback_or_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    archive_root = tmp_path / "archive_root.txt"
    archive_root.write_text("not-a-directory", encoding="utf-8")

    result = _run_cli(
        [
            sys.executable,
            str(HASH_TOOL),
            "--archive-index",
            str(FIXTURE_INDEX),
            "--archive-root",
            str(archive_root),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert result.returncode == ARCHIVE_HASH_MODULE.EXIT_INPUT_ERROR
    assert "Error: archive root must be a directory:" in result.stderr
    assert "Traceback (most recent call last):" not in result.stderr
    assert not out_dir.exists()


def test_phase3_fixture_outputs_match_golden_bytes() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        hash_result = _run_hash_tool(FIXTURE_INDEX, FIXTURE_ARCHIVE_ROOT, out_dir)
        assert hash_result.returncode == 0, hash_result.stderr

        report_path = out_dir / "verification_report.json"
        verify_result = _run_verify_tool(out_dir / "hash_manifest.csv.gz", FIXTURE_ARCHIVE_ROOT, report_path)
        assert verify_result.returncode == 0, verify_result.stderr

        with gzip.open(out_dir / "hash_manifest.csv.gz", "rt", encoding="utf-8", newline="") as handle:
            assert handle.readline().rstrip("\n") == "# hash_algorithm=sha256"

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


@pytest.mark.parametrize(("raw_relpath", "expected_error"), RELPATH_REJECTION_CASES)
def test_hash_tool_materialize_relpath_rejects_anchored_and_drive_inputs(raw_relpath: str, expected_error: str) -> None:
    relpath_obj, error = ARCHIVE_HASH_MODULE._materialize_relpath(raw_relpath)
    assert relpath_obj is None
    assert error == expected_error


@pytest.mark.parametrize(("raw_relpath", "expected_error"), RELPATH_REJECTION_CASES)
def test_verify_tool_materialize_relpath_rejects_anchored_and_drive_inputs(raw_relpath: str, expected_error: str) -> None:
    relpath_obj, error = VERIFY_HASH_MODULE._materialize_relpath(raw_relpath)
    assert relpath_obj is None
    assert error == expected_error


def test_hash_tool_rejects_nul_in_identity_fields(tmp_path: Path) -> None:
    row = ARCHIVE_HASH_MODULE.ArchiveIndexRow(
        row_number=1,
        origin_drive="Drive\x00A",
        partition="Part1",
        relpath="DriveA/Part1/alpha.txt",
    )
    result = ARCHIVE_HASH_MODULE.hash_one_row(tmp_path, row)
    assert result.hash_status == "skipped"
    assert result.error == "invalid_identity_nul"
    assert result.sha256 == ""
    assert result.filesize_bytes == 0


def test_verify_tool_rejects_nul_in_identity_fields(tmp_path: Path) -> None:
    expected = VERIFY_HASH_MODULE.ExpectedRow(
        row_number=1,
        origin_drive="DriveA",
        partition="Part\x001",
        relpath="DriveA/Part1/alpha.txt",
        filesize_bytes=0,
        sha256="",
        hash_status="skipped",
        error="invalid_identity_nul",
    )
    observed = VERIFY_HASH_MODULE.observe_row(tmp_path, expected)
    assert observed.status == "skipped"
    assert observed.error == "invalid_identity_nul"
    assert observed.sha256 == ""
    assert observed.filesize_bytes == 0


def test_verify_reader_skips_comment_and_blank_preamble_lines(tmp_path: Path) -> None:
    manifest_path = tmp_path / "hash_manifest.csv.gz"
    with gzip.open(manifest_path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("# hash_algorithm=sha256\n")
        handle.write("\n")
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["origin_drive", "partition", "relpath", "filesize_bytes", "sha256", "hash_status", "error"])
        writer.writerow(["DriveA", "Part1", "DriveA/Part1/alpha.txt", "26", "abc", "ok", ""])

    rows = VERIFY_HASH_MODULE._open_manifest_reader(manifest_path)
    assert len(rows) == 1
    assert rows[0].origin_drive == "DriveA"
    assert rows[0].partition == "Part1"
    assert rows[0].relpath == "DriveA/Part1/alpha.txt"


def test_verify_reader_preserves_hash_prefixed_values_after_header(tmp_path: Path) -> None:
    manifest_path = tmp_path / "hash_manifest.csv.gz"
    with gzip.open(manifest_path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("# hash_algorithm=sha256\n")
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["origin_drive", "partition", "relpath", "filesize_bytes", "sha256", "hash_status", "error"])
        writer.writerow(["#DriveA", "Part1", "DriveA/Part1/alpha.txt", "26", "abc", "ok", ""])
        writer.writerow(["DriveB", "Part2", "DriveB/Part2/sub/charlie.txt", "8", "def", "ok", ""])

    rows = VERIFY_HASH_MODULE._open_manifest_reader(manifest_path)
    assert len(rows) == 2
    assert rows[0].origin_drive == "#DriveA"
    assert rows[1].origin_drive == "DriveB"


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
        assert report["rows_mismatched"] == 1
        assert report["rows_matched"] == report["rows_checked"] - 1
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


def test_hash_tool_strict_identity_fails_on_duplicate_identity_keys() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        temp = Path(tmpdir)
        archive_root = _copy_fixture_archive(temp)
        duplicate_index = temp / "archive_index_duplicates.csv"
        with duplicate_index.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["origin_drive", "partition", "relpath"])
            writer.writerow(["DriveA", "Part1", "DriveA/Part1/alpha.txt"])
            writer.writerow(["DriveA", "Part1", "DriveA/Part1/alpha.txt"])
            writer.writerow(["DriveB", "Part2", "DriveB/Part2/sub/charlie.txt"])

        relaxed_out = temp / "out_relaxed"
        relaxed_result = _run_hash_tool(duplicate_index, archive_root, relaxed_out)
        assert relaxed_result.returncode == 0, relaxed_result.stderr
        assert (relaxed_out / "hash_manifest.csv.gz").exists()

        strict_out = temp / "out_strict_identity"
        strict_identity_result = _run_hash_tool(
            duplicate_index,
            archive_root,
            strict_out,
            strict_identity=True,
        )
        assert strict_identity_result.returncode == 3
        assert "Duplicate identity keys detected" in strict_identity_result.stdout
        assert not (strict_out / "hash_manifest.csv.gz").exists()


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


def test_hash_tool_skips_symlinked_directory_traversal() -> None:
    if os.name == "nt":
        pytest.skip("symlink behavior is not stable on Windows CI hosts")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp = Path(tmpdir)
        archive_root = _copy_fixture_archive(temp)

        outside_dir = temp / "outside"
        outside_dir.mkdir(parents=True, exist_ok=True)
        outside_target = outside_dir / "escape.txt"
        outside_target.write_bytes(b"outside-data")

        link_dir = archive_root / "DriveA" / "Part1" / "link_out"
        link_dir.symlink_to(outside_dir, target_is_directory=True)

        index_path = temp / "index_with_symlink_dir.csv"
        with index_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["origin_drive", "partition", "relpath"])
            writer.writerow(["DriveA", "Part1", "DriveA/Part1/link_out/escape.txt"])

        out_dir = temp / "out"
        result = _run_hash_tool(index_path, archive_root, out_dir)
        assert result.returncode == 0, result.stderr

        with gzip.open(out_dir / "hash_manifest.csv.gz", "rt", encoding="utf-8", newline="") as handle:
            lines = [line for line in handle if line.strip() and not line.startswith("#")]
        rows = list(csv.DictReader(lines))
        assert len(rows) == 1
        assert rows[0]["hash_status"] == "skipped"
        assert rows[0]["error"] == "symlink_skipped"
        assert rows[0]["sha256"] == ""


def test_verify_tool_skips_symlinked_directory_traversal() -> None:
    if os.name == "nt":
        pytest.skip("symlink behavior is not stable on Windows CI hosts")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp = Path(tmpdir)
        archive_root = _copy_fixture_archive(temp)

        outside_dir = temp / "outside"
        outside_dir.mkdir(parents=True, exist_ok=True)
        outside_target = outside_dir / "escape.txt"
        outside_target.write_bytes(b"outside-data")
        outside_sha = hashlib.sha256(outside_target.read_bytes()).hexdigest()
        outside_size = outside_target.stat().st_size

        link_dir = archive_root / "DriveA" / "Part1" / "link_out"
        link_dir.symlink_to(outside_dir, target_is_directory=True)

        manifest_path = temp / "hash_manifest.csv.gz"
        with gzip.open(manifest_path, "wt", encoding="utf-8", newline="") as handle:
            handle.write("# hash_algorithm=sha256\n")
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["origin_drive", "partition", "relpath", "filesize_bytes", "sha256", "hash_status", "error"])
            writer.writerow(
                [
                    "DriveA",
                    "Part1",
                    "DriveA/Part1/link_out/escape.txt",
                    str(outside_size),
                    outside_sha,
                    "ok",
                    "",
                ]
            )

        report_path = temp / "verification_report.json"
        verify_result = _run_verify_tool(manifest_path, archive_root, report_path)
        assert verify_result.returncode != 0

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["rows_mismatched"] == 1
        mismatch = report["mismatches"][0]
        assert mismatch["issue"] == "status_mismatch"
        assert mismatch["observed_status"] == "skipped"
