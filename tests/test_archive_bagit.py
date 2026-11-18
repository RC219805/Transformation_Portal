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


def _validate_bag(bag_dir: Path, report_json: Path) -> tuple[subprocess.CompletedProcess[str], set[tuple[str, str, str]]]:
    result = _run_bagit("validate", "--bag-dir", str(bag_dir), "--report-json", str(report_json))
    report = _read_json(report_json)
    issues = {(item["scope"], item["path"], item["issue"]) for item in report["mismatches"]}
    assert report["valid"] is (result.returncode == 0)
    return result, issues


def _rewrite_tag_manifest(bag_dir: Path) -> None:
    """Re-seal the tag manifest so a test isolates one structural defect instead of a digest mismatch."""
    names = ("bagit.txt", "bag-info.txt", "manifest-sha256.txt")
    (bag_dir / "tagmanifest-sha256.txt").write_text(
        "".join(f"{_sha256_bytes((bag_dir / name).read_bytes())}  {name}\n" for name in names),
        encoding="utf-8",
    )


def test_archive_bagit_validate_accepts_freshly_built_bag(tmp_path: Path) -> None:
    bag_dir, _ = _build_fixture_bag(tmp_path)

    result, issues = _validate_bag(bag_dir, tmp_path / "validate_report.json")

    assert result.returncode == 0, result.stderr
    assert issues == set()
    report = _read_json(tmp_path / "validate_report.json")
    assert report["payload_rows"] == 2
    assert report["tag_rows"] == 3


def test_archive_bagit_validate_accepts_empty_payload_bag(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive_root"
    archive_root.mkdir()
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest_jsonl(manifest_path, [{"relpath": "gone.bin", "sha256": "", "hash_status": "missing"}])
    bag_dir = tmp_path / "bag"
    build = _run_bagit(
        "build", "--manifest-jsonl", str(manifest_path), "--archive-root", str(archive_root), "--bag-dir", str(bag_dir)
    )
    assert build.returncode == 0, build.stderr
    assert "Payload-Oxum: 0.0" in (bag_dir / "bag-info.txt").read_text(encoding="utf-8")

    result, issues = _validate_bag(bag_dir, tmp_path / "validate_report.json")

    assert result.returncode == 0, result.stderr
    assert issues == set()


def test_archive_bagit_validate_rejects_unlisted_payload_file(tmp_path: Path) -> None:
    bag_dir, _ = _build_fixture_bag(tmp_path)
    (bag_dir / "data" / "set1" / "injected.bin").write_bytes(b"not in any manifest")

    result, issues = _validate_bag(bag_dir, tmp_path / "validate_report.json")

    assert result.returncode == 4
    assert ("payload", "data/set1/injected.bin", "unlisted_payload") in issues
    assert ("payload", "data", "payload_oxum_mismatch") in issues


def test_archive_bagit_validate_rejects_truncated_manifests(tmp_path: Path) -> None:
    """Zero manifest rows must not turn a corrupted payload into a valid bag."""
    bag_dir, _ = _build_fixture_bag(tmp_path)
    (bag_dir / "manifest-sha256.txt").write_bytes(b"")
    (bag_dir / "tagmanifest-sha256.txt").write_bytes(b"")
    (bag_dir / "data" / "set1" / "alpha.bin").write_bytes(b"CORRUPTED")

    result, issues = _validate_bag(bag_dir, tmp_path / "validate_report.json")

    assert result.returncode == 4
    assert ("payload", "data/set1/alpha.bin", "unlisted_payload") in issues
    assert ("payload", "data/set1/bravo.bin", "unlisted_payload") in issues
    assert ("tag", "manifest-sha256.txt", "missing_from_tag_manifest") in issues
    report = _read_json(tmp_path / "validate_report.json")
    assert (report["payload_rows"], report["tag_rows"]) == (0, 0)


def test_archive_bagit_validate_rejects_payload_removed_together_with_its_manifest_row(tmp_path: Path) -> None:
    bag_dir, _ = _build_fixture_bag(tmp_path)
    declared = (bag_dir / "bag-info.txt").read_text(encoding="utf-8").strip().splitlines()[-1]
    removed_size = (bag_dir / "data" / "set1" / "bravo.bin").stat().st_size
    kept_size = (bag_dir / "data" / "set1" / "alpha.bin").stat().st_size
    assert declared == f"Payload-Oxum: {kept_size + removed_size}.2"

    (bag_dir / "data" / "set1" / "bravo.bin").unlink()
    payload_manifest = bag_dir / "manifest-sha256.txt"
    kept_rows = [line for line in payload_manifest.read_text(encoding="utf-8").splitlines() if "bravo.bin" not in line]
    payload_manifest.write_text("\n".join(kept_rows) + "\n", encoding="utf-8")
    _rewrite_tag_manifest(bag_dir)

    result, issues = _validate_bag(bag_dir, tmp_path / "validate_report.json")

    assert result.returncode == 4
    assert issues == {("payload", "data", "payload_oxum_mismatch")}
    mismatch = _read_json(tmp_path / "validate_report.json")["mismatches"][0]
    assert mismatch["expected"] == f"{kept_size + removed_size}.2"
    assert mismatch["observed"] == f"{kept_size}.1"


@pytest.mark.parametrize(
    "bag_info_tail",
    ["Payload-Oxum: many\n", "Payload-Oxum: 27.2\nPayload-Oxum: 27.2\n"],
    ids=["malformed", "repeated"],
)
def test_archive_bagit_validate_rejects_unusable_payload_oxum(tmp_path: Path, bag_info_tail: str) -> None:
    bag_dir, _ = _build_fixture_bag(tmp_path)
    (bag_dir / "bag-info.txt").write_text(f"Source-Organization: TEST_ORG\n{bag_info_tail}", encoding="utf-8")
    _rewrite_tag_manifest(bag_dir)

    result, issues = _validate_bag(bag_dir, tmp_path / "validate_report.json")

    assert result.returncode == 4
    assert issues == {("tag", "bag-info.txt", "invalid_payload_oxum")}


@pytest.mark.parametrize(
    "declaration",
    [
        "garbage\n",
        "BagIt-Version: 0.97\n",
        "Tag-File-Character-Encoding: UTF-8\nBagIt-Version: 0.97\n",
        "BagIt-Version: 0.97\nTag-File-Character-Encoding: UTF-16\n",
    ],
    ids=["garbage", "missing-encoding", "wrong-order", "unsupported-encoding"],
)
def test_archive_bagit_validate_rejects_invalid_bag_declaration(tmp_path: Path, declaration: str) -> None:
    bag_dir, _ = _build_fixture_bag(tmp_path)
    (bag_dir / "bagit.txt").write_text(declaration, encoding="utf-8")
    _rewrite_tag_manifest(bag_dir)

    result, issues = _validate_bag(bag_dir, tmp_path / "validate_report.json")

    assert result.returncode == 4
    assert issues == {("tag", "bagit.txt", "invalid_bag_declaration")}


def test_archive_bagit_validate_requires_tag_manifest_to_cover_core_tag_files(tmp_path: Path) -> None:
    bag_dir, _ = _build_fixture_bag(tmp_path)
    tag_manifest = bag_dir / "tagmanifest-sha256.txt"
    kept_rows = [line for line in tag_manifest.read_text(encoding="utf-8").splitlines() if line.endswith("  bagit.txt")]
    tag_manifest.write_text("\n".join(kept_rows) + "\n", encoding="utf-8")

    result, issues = _validate_bag(bag_dir, tmp_path / "validate_report.json")

    assert result.returncode == 4
    assert issues == {
        ("tag", "bag-info.txt", "missing_from_tag_manifest"),
        ("tag", "manifest-sha256.txt", "missing_from_tag_manifest"),
    }


def test_archive_bagit_validate_reports_symlinked_payload_directory(tmp_path: Path) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlink is not supported on this platform")

    bag_dir, _ = _build_fixture_bag(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "hidden.bin").write_bytes(b"hidden behind a directory symlink")
    os.symlink(outside, bag_dir / "data" / "linked", target_is_directory=True)

    result, issues = _validate_bag(bag_dir, tmp_path / "validate_report.json")

    assert result.returncode == 4
    assert issues == {("payload", "data/linked", "unlisted_payload")}


def test_archive_bagit_path_key_ignores_unicode_normalization_form(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(BAGIT_TOOL.parent))
    import archive_bagit

    decomposed = Path("data") / "café.bin"
    precomposed = Path("data") / "café.bin"
    assert decomposed.as_posix() != precomposed.as_posix()
    assert archive_bagit._bag_path_key(decomposed) == archive_bagit._bag_path_key(precomposed)


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
