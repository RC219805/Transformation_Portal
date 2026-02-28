"""CLI tests for tools/build_dedup_ledger.py."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEDUP_TOOL = PROJECT_ROOT / "tools" / "build_dedup_ledger.py"


def _run_dedup(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(DEDUP_TOOL), *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _write_manifest_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _read_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_ledger_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_build_dedup_ledger_groups_duplicates_and_prefers_raw_canonical(tmp_path: Path) -> None:
    manifest_path = tmp_path / "archive_manifest_v2.jsonl"
    out_ledger = tmp_path / "dedup_ledger.csv"
    out_summary = tmp_path / "dedup_summary.json"

    _write_manifest_jsonl(
        manifest_path,
        [
            {
                "relpath": "set/img_preview.jpg",
                "extension": ".jpg",
                "rights_flags": ["unspecified"],
                "sha256": "a" * 64,
                "hash_status": "ok",
                "created_utc": "2024-01-01T00:00:00Z",
            },
            {
                "relpath": "set/raw/master.nef",
                "extension": ".nef",
                "rights_flags": ["licensed"],
                "sha256": "a" * 64,
                "hash_status": "ok",
                "created_utc": "2024-01-02T00:00:00Z",
                "modified_utc": "2024-02-01T00:00:00Z",
                "accessed_utc": "2024-02-01T02:00:00Z",
            },
            {
                "relpath": "set/unique.png",
                "extension": ".png",
                "rights_flags": ["licensed"],
                "sha256": "b" * 64,
                "hash_status": "ok",
                "modified_utc": "2024-01-15T00:00:00Z",
            },
            {
                "relpath": "set/not_hashable.png",
                "extension": ".png",
                "rights_flags": ["licensed"],
                "sha256": "not-a-sha256",
                "hash_status": "ok",
            },
        ],
    )

    result = _run_dedup(
        "--manifest-jsonl",
        str(manifest_path),
        "--out-ledger",
        str(out_ledger),
        "--out-summary",
        str(out_summary),
        "--approver",
        "qa_user",
    )
    assert result.returncode == 0, result.stderr

    ledger_rows = _read_ledger_rows(out_ledger)
    assert len(ledger_rows) == 1
    row = ledger_rows[0]
    assert row["canonical_path"] == "set/raw/master.nef"
    assert row["duplicate_count"] == "2"
    assert row["approver"] == "qa_user"
    assert json.loads(row["duplicate_paths"]) == ["set/img_preview.jpg"]
    assert row["date_utc"] == "2024-02-01T02:00:00Z"

    summary = _read_summary(out_summary)
    assert summary["input_rows"] == 4
    assert summary["hashable_rows"] == 3
    assert summary["duplicate_groups"] == 1
    assert summary["duplicate_excess_files"] == 1
    assert summary["ledger_rows"] == 1
    assert summary["decision_date_utc"] == "2024-02-01T02:00:00Z"


def test_build_dedup_ledger_outputs_are_deterministic(tmp_path: Path) -> None:
    manifest_path = tmp_path / "archive_manifest_v2.jsonl"
    _write_manifest_jsonl(
        manifest_path,
        [
            {
                "relpath": "a/raw.nef",
                "extension": ".nef",
                "rights_flags": ["licensed"],
                "sha256": "c" * 64,
                "hash_status": "ok",
                "modified_utc": "2024-04-01T00:00:00Z",
            },
            {
                "relpath": "a/preview.jpg",
                "extension": ".jpg",
                "rights_flags": ["licensed"],
                "sha256": "c" * 64,
                "hash_status": "ok",
                "modified_utc": "2024-04-01T00:00:00Z",
            },
        ],
    )

    ledger_a = tmp_path / "a.csv"
    summary_a = tmp_path / "a.json"
    ledger_b = tmp_path / "b.csv"
    summary_b = tmp_path / "b.json"

    first = _run_dedup(
        "--manifest-jsonl",
        str(manifest_path),
        "--out-ledger",
        str(ledger_a),
        "--out-summary",
        str(summary_a),
    )
    second = _run_dedup(
        "--manifest-jsonl",
        str(manifest_path),
        "--out-ledger",
        str(ledger_b),
        "--out-summary",
        str(summary_b),
    )
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr

    assert ledger_a.read_bytes() == ledger_b.read_bytes()
    assert summary_a.read_bytes() == summary_b.read_bytes()


def test_build_dedup_ledger_rejects_invalid_decision_date_override(tmp_path: Path) -> None:
    manifest_path = tmp_path / "archive_manifest_v2.jsonl"
    out_ledger = tmp_path / "ledger.csv"
    out_summary = tmp_path / "summary.json"
    _write_manifest_jsonl(
        manifest_path,
        [
            {
                "relpath": "a/file.txt",
                "extension": ".txt",
                "rights_flags": ["licensed"],
                "sha256": "d" * 64,
                "hash_status": "ok",
            }
        ],
    )

    result = _run_dedup(
        "--manifest-jsonl",
        str(manifest_path),
        "--out-ledger",
        str(out_ledger),
        "--out-summary",
        str(out_summary),
        "--decision-date-utc",
        "not-a-date",
    )
    assert result.returncode == 2
    assert "Invalid --decision-date-utc value" in result.stderr
