"""Tests for the Phase 3.3 evidence bundle manifest CLIs."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATE_TOOL = PROJECT_ROOT / "tools" / "generate_evidence_bundle_manifest.py"
VERIFY_TOOL = PROJECT_ROOT / "tools" / "verify_evidence_bundle_manifest.py"
PHASE3_GOLDEN_DIR = PROJECT_ROOT / "tests" / "fixtures" / "archive_small" / "golden"

pytestmark = [pytest.mark.regression]


def _run_cli(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _generate_manifest(
    *,
    roots_path: Path,
    hash_manifest_path: Path,
    hash_summary_path: Path,
    signature_path: Path,
    timestamp_path: Path,
    timestamp_target: str,
    out_path: Path,
) -> subprocess.CompletedProcess[str]:
    return _run_cli(
        [
            sys.executable,
            str(GENERATE_TOOL),
            "--roots",
            str(roots_path),
            "--hash-manifest",
            str(hash_manifest_path),
            "--hash-summary",
            str(hash_summary_path),
            "--signature",
            str(signature_path),
            "--timestamp-target",
            timestamp_target,
            "--timestamp",
            str(timestamp_path),
            "--out",
            str(out_path),
        ]
    )


def _verify_manifest(manifest_path: Path, bundle_dir: Path | None = None) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(VERIFY_TOOL),
        "--bundle-manifest",
        str(manifest_path),
    ]
    if bundle_dir is not None:
        command.extend(["--bundle-dir", str(bundle_dir)])
    return _run_cli(command)


def _write_bundle_artifacts(tmp_path: Path, *, timestamp_target: str) -> dict[str, Path]:
    roots_path = tmp_path / "merkle_roots.json"
    roots_path.write_text(
        json.dumps(
            {
                "hash_algorithm": "sha256",
                "leaf_hash_algorithm": "sha256",
                "leaf_format_version": "1",
                "leaf_format": "v1",
                "tree_method_version": "1",
                "tree_method": "duplicate_last",
                "partitions": [],
                "global": {"leaf_count": 3, "root_sha256": "0" * 64},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    hash_manifest_path = tmp_path / "hash_manifest.csv.gz"
    hash_manifest_path.write_bytes(
        b"# hash_algorithm=sha256\n"
        b"origin_drive,partition,relpath,filesize_bytes,sha256,hash_status,error\n"
        b"driveA,partA,a.jpg,5,abc,ok,\n"
    )

    hash_summary_path = tmp_path / "hash_summary.json"
    hash_summary_path.write_text(
        json.dumps(
            {
                "hash_algorithm": "sha256",
                "hash_manifest_schema_version": "1",
                "rows_total": 1,
                "hashed_ok": 1,
                "missing": 0,
                "unreadable": 0,
                "skipped": 0,
                "total_bytes_hashed": 5,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    signature_path = tmp_path / "merkle_roots.sig.json"
    signature_path.write_text(
        json.dumps(
            {
                "envelope_version": "1",
                "signature_algorithm": "ed25519",
                "artifact_digest_algorithm": "sha256",
                "signed_artifact": "merkle_roots.json",
                "signed_artifact_sha256": hashlib.sha256(roots_path.read_bytes()).hexdigest(),
                "signature_base64": "c2ln",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    timestamp_filename = "merkle_roots.sig.tsr" if timestamp_target == "signature" else "merkle_roots.tsr"
    timestamp_path = tmp_path / timestamp_filename
    timestamp_path.write_bytes(b"\x30\x03\x30\x01\x00")

    out_path = tmp_path / "evidence_bundle_manifest.json"
    return {
        "roots": roots_path,
        "hash_manifest": hash_manifest_path,
        "hash_summary": hash_summary_path,
        "signature": signature_path,
        "timestamp": timestamp_path,
        "out": out_path,
    }


def _load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_generate_and_verify_signature_timestamp_bundle(tmp_path: Path) -> None:
    artifacts = _write_bundle_artifacts(tmp_path, timestamp_target="signature")

    generate_result = _generate_manifest(
        roots_path=artifacts["roots"],
        hash_manifest_path=artifacts["hash_manifest"],
        hash_summary_path=artifacts["hash_summary"],
        signature_path=artifacts["signature"],
        timestamp_path=artifacts["timestamp"],
        timestamp_target="signature",
        out_path=artifacts["out"],
    )
    assert generate_result.returncode == 0, generate_result.stderr

    manifest = _load_manifest(artifacts["out"])
    assert manifest["bundle_version"] == "1"
    assert manifest["hash_algorithm"] == "sha256"
    assert manifest["merkle_leaf_count"] == 3
    assert manifest["timestamp_target"] == "signature"
    assert manifest["timestamp_path"] == "merkle_roots.sig.tsr"

    verify_result = _verify_manifest(artifacts["out"], tmp_path)
    assert verify_result.returncode == 0, verify_result.stderr
    assert "Evidence bundle manifest valid" in verify_result.stdout


def test_generate_and_verify_roots_timestamp_bundle(tmp_path: Path) -> None:
    artifacts = _write_bundle_artifacts(tmp_path, timestamp_target="roots")

    generate_result = _generate_manifest(
        roots_path=artifacts["roots"],
        hash_manifest_path=artifacts["hash_manifest"],
        hash_summary_path=artifacts["hash_summary"],
        signature_path=artifacts["signature"],
        timestamp_path=artifacts["timestamp"],
        timestamp_target="roots",
        out_path=artifacts["out"],
    )
    assert generate_result.returncode == 0, generate_result.stderr

    manifest = _load_manifest(artifacts["out"])
    assert manifest["timestamp_target"] == "roots"
    assert manifest["timestamp_path"] == "merkle_roots.tsr"

    verify_result = _verify_manifest(artifacts["out"], tmp_path)
    assert verify_result.returncode == 0, verify_result.stderr


def test_generate_fails_on_timestamp_target_filename_mismatch(tmp_path: Path) -> None:
    artifacts = _write_bundle_artifacts(tmp_path, timestamp_target="roots")
    mismatched_timestamp_path = artifacts["timestamp"]

    generate_result = _generate_manifest(
        roots_path=artifacts["roots"],
        hash_manifest_path=artifacts["hash_manifest"],
        hash_summary_path=artifacts["hash_summary"],
        signature_path=artifacts["signature"],
        timestamp_path=mismatched_timestamp_path,
        timestamp_target="signature",
        out_path=artifacts["out"],
    )
    assert generate_result.returncode == 10
    assert "--timestamp must reference merkle_roots.sig.tsr" in generate_result.stdout


def test_generate_fails_on_invalid_output_filename(tmp_path: Path) -> None:
    artifacts = _write_bundle_artifacts(tmp_path, timestamp_target="signature")
    bad_out_path = tmp_path / "bundle.json"

    generate_result = _generate_manifest(
        roots_path=artifacts["roots"],
        hash_manifest_path=artifacts["hash_manifest"],
        hash_summary_path=artifacts["hash_summary"],
        signature_path=artifacts["signature"],
        timestamp_path=artifacts["timestamp"],
        timestamp_target="signature",
        out_path=bad_out_path,
    )
    assert generate_result.returncode == 10
    assert "--out must reference evidence_bundle_manifest.json" in generate_result.stdout


def test_generate_output_is_deterministic(tmp_path: Path) -> None:
    artifacts = _write_bundle_artifacts(tmp_path, timestamp_target="signature")
    out_a = tmp_path / "evidence_bundle_manifest.json"
    out_b = tmp_path / "subdir" / "evidence_bundle_manifest.json"

    first = _generate_manifest(
        roots_path=artifacts["roots"],
        hash_manifest_path=artifacts["hash_manifest"],
        hash_summary_path=artifacts["hash_summary"],
        signature_path=artifacts["signature"],
        timestamp_path=artifacts["timestamp"],
        timestamp_target="signature",
        out_path=out_a,
    )
    second = _generate_manifest(
        roots_path=artifacts["roots"],
        hash_manifest_path=artifacts["hash_manifest"],
        hash_summary_path=artifacts["hash_summary"],
        signature_path=artifacts["signature"],
        timestamp_path=artifacts["timestamp"],
        timestamp_target="signature",
        out_path=out_b,
    )
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert out_a.read_bytes() == out_b.read_bytes()


def test_verify_fails_on_digest_mismatch(tmp_path: Path) -> None:
    artifacts = _write_bundle_artifacts(tmp_path, timestamp_target="signature")
    generate_result = _generate_manifest(
        roots_path=artifacts["roots"],
        hash_manifest_path=artifacts["hash_manifest"],
        hash_summary_path=artifacts["hash_summary"],
        signature_path=artifacts["signature"],
        timestamp_path=artifacts["timestamp"],
        timestamp_target="signature",
        out_path=artifacts["out"],
    )
    assert generate_result.returncode == 0, generate_result.stderr

    artifacts["signature"].write_text('{"tampered": true}\n', encoding="utf-8")

    verify_result = _verify_manifest(artifacts["out"], tmp_path)
    assert verify_result.returncode == 11
    assert "digest mismatch for signature_path" in verify_result.stdout


def test_verify_fails_on_merkle_leaf_count_mismatch(tmp_path: Path) -> None:
    artifacts = _write_bundle_artifacts(tmp_path, timestamp_target="signature")
    generate_result = _generate_manifest(
        roots_path=artifacts["roots"],
        hash_manifest_path=artifacts["hash_manifest"],
        hash_summary_path=artifacts["hash_summary"],
        signature_path=artifacts["signature"],
        timestamp_path=artifacts["timestamp"],
        timestamp_target="signature",
        out_path=artifacts["out"],
    )
    assert generate_result.returncode == 0, generate_result.stderr

    manifest = _load_manifest(artifacts["out"])
    manifest["merkle_leaf_count"] = int(manifest["merkle_leaf_count"]) + 1
    artifacts["out"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verify_result = _verify_manifest(artifacts["out"], tmp_path)
    assert verify_result.returncode == 11
    assert "merkle_leaf_count mismatch" in verify_result.stdout


def test_verify_fails_on_timestamp_target_path_coupling(tmp_path: Path) -> None:
    artifacts = _write_bundle_artifacts(tmp_path, timestamp_target="signature")
    generate_result = _generate_manifest(
        roots_path=artifacts["roots"],
        hash_manifest_path=artifacts["hash_manifest"],
        hash_summary_path=artifacts["hash_summary"],
        signature_path=artifacts["signature"],
        timestamp_path=artifacts["timestamp"],
        timestamp_target="signature",
        out_path=artifacts["out"],
    )
    assert generate_result.returncode == 0, generate_result.stderr

    manifest = _load_manifest(artifacts["out"])
    manifest["timestamp_target"] = "roots"
    artifacts["out"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verify_result = _verify_manifest(artifacts["out"], tmp_path)
    assert verify_result.returncode == 12
    assert "timestamp_path must be 'merkle_roots.tsr'" in verify_result.stdout


def test_verify_fails_on_manifest_missing_required_field(tmp_path: Path) -> None:
    artifacts = _write_bundle_artifacts(tmp_path, timestamp_target="signature")
    generate_result = _generate_manifest(
        roots_path=artifacts["roots"],
        hash_manifest_path=artifacts["hash_manifest"],
        hash_summary_path=artifacts["hash_summary"],
        signature_path=artifacts["signature"],
        timestamp_path=artifacts["timestamp"],
        timestamp_target="signature",
        out_path=artifacts["out"],
    )
    assert generate_result.returncode == 0, generate_result.stderr

    manifest = _load_manifest(artifacts["out"])
    manifest.pop("bundle_tool_version")
    artifacts["out"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verify_result = _verify_manifest(artifacts["out"], tmp_path)
    assert verify_result.returncode == 12
    assert "missing required field(s): bundle_tool_version" in verify_result.stdout


def test_verify_fails_on_manifest_with_extra_field(tmp_path: Path) -> None:
    artifacts = _write_bundle_artifacts(tmp_path, timestamp_target="signature")
    generate_result = _generate_manifest(
        roots_path=artifacts["roots"],
        hash_manifest_path=artifacts["hash_manifest"],
        hash_summary_path=artifacts["hash_summary"],
        signature_path=artifacts["signature"],
        timestamp_path=artifacts["timestamp"],
        timestamp_target="signature",
        out_path=artifacts["out"],
    )
    assert generate_result.returncode == 0, generate_result.stderr

    manifest = _load_manifest(artifacts["out"])
    manifest["extra"] = "not-allowed"
    artifacts["out"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verify_result = _verify_manifest(artifacts["out"], tmp_path)
    assert verify_result.returncode == 12
    assert "unexpected field(s): extra" in verify_result.stdout


def test_generate_fails_on_boolean_roots_leaf_count(tmp_path: Path) -> None:
    artifacts = _write_bundle_artifacts(tmp_path, timestamp_target="signature")
    roots_payload = json.loads(artifacts["roots"].read_text(encoding="utf-8"))
    roots_payload["global"]["leaf_count"] = True
    artifacts["roots"].write_text(json.dumps(roots_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    generate_result = _generate_manifest(
        roots_path=artifacts["roots"],
        hash_manifest_path=artifacts["hash_manifest"],
        hash_summary_path=artifacts["hash_summary"],
        signature_path=artifacts["signature"],
        timestamp_path=artifacts["timestamp"],
        timestamp_target="signature",
        out_path=artifacts["out"],
    )
    assert generate_result.returncode == 10
    assert "leaf_count must be a non-negative integer" in generate_result.stdout


def test_verify_fails_on_boolean_roots_leaf_count(tmp_path: Path) -> None:
    artifacts = _write_bundle_artifacts(tmp_path, timestamp_target="signature")
    generate_result = _generate_manifest(
        roots_path=artifacts["roots"],
        hash_manifest_path=artifacts["hash_manifest"],
        hash_summary_path=artifacts["hash_summary"],
        signature_path=artifacts["signature"],
        timestamp_path=artifacts["timestamp"],
        timestamp_target="signature",
        out_path=artifacts["out"],
    )
    assert generate_result.returncode == 0, generate_result.stderr

    roots_payload = json.loads(artifacts["roots"].read_text(encoding="utf-8"))
    roots_payload["global"]["leaf_count"] = True
    artifacts["roots"].write_text(json.dumps(roots_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = _load_manifest(artifacts["out"])
    manifest["roots_sha256"] = hashlib.sha256(artifacts["roots"].read_bytes()).hexdigest()
    artifacts["out"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verify_result = _verify_manifest(artifacts["out"], tmp_path)
    assert verify_result.returncode == 11
    assert "leaf_count must be a non-negative integer" in verify_result.stdout


def test_verify_fails_on_boolean_manifest_merkle_leaf_count(tmp_path: Path) -> None:
    artifacts = _write_bundle_artifacts(tmp_path, timestamp_target="signature")
    generate_result = _generate_manifest(
        roots_path=artifacts["roots"],
        hash_manifest_path=artifacts["hash_manifest"],
        hash_summary_path=artifacts["hash_summary"],
        signature_path=artifacts["signature"],
        timestamp_path=artifacts["timestamp"],
        timestamp_target="signature",
        out_path=artifacts["out"],
    )
    assert generate_result.returncode == 0, generate_result.stderr

    manifest = _load_manifest(artifacts["out"])
    manifest["merkle_leaf_count"] = True
    artifacts["out"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verify_result = _verify_manifest(artifacts["out"], tmp_path)
    assert verify_result.returncode == 12
    assert "merkle_leaf_count must be a non-negative integer" in verify_result.stdout


def test_verify_fails_on_whitespace_only_phase_version(tmp_path: Path) -> None:
    artifacts = _write_bundle_artifacts(tmp_path, timestamp_target="signature")
    generate_result = _generate_manifest(
        roots_path=artifacts["roots"],
        hash_manifest_path=artifacts["hash_manifest"],
        hash_summary_path=artifacts["hash_summary"],
        signature_path=artifacts["signature"],
        timestamp_path=artifacts["timestamp"],
        timestamp_target="signature",
        out_path=artifacts["out"],
    )
    assert generate_result.returncode == 0, generate_result.stderr

    manifest = _load_manifest(artifacts["out"])
    manifest["phase3_version"] = "   "
    artifacts["out"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verify_result = _verify_manifest(artifacts["out"], tmp_path)
    assert verify_result.returncode == 12
    assert "phase3_version must be a non-empty string" in verify_result.stdout


def _write_consistent_bundle(tmp_path: Path) -> dict[str, Path]:
    """Bundle whose hash manifest, summary and Merkle roots are real Phase 3 outputs of one scan."""
    artifacts = _write_bundle_artifacts(tmp_path, timestamp_target="signature")
    for name in ("hash_manifest.csv.gz", "hash_summary.json", "merkle_roots.json"):
        shutil.copyfile(PHASE3_GOLDEN_DIR / name, tmp_path / name)

    generate_result = _generate_manifest(
        roots_path=artifacts["roots"],
        hash_manifest_path=artifacts["hash_manifest"],
        hash_summary_path=artifacts["hash_summary"],
        signature_path=artifacts["signature"],
        timestamp_path=artifacts["timestamp"],
        timestamp_target="signature",
        out_path=artifacts["out"],
    )
    assert generate_result.returncode == 0, generate_result.stdout + generate_result.stderr
    return artifacts


def _refresh_manifest_digest(artifacts: dict[str, Path], artifact_key: str, digest_field: str) -> None:
    manifest = _load_manifest(artifacts["out"])
    manifest[digest_field] = hashlib.sha256(artifacts[artifact_key].read_bytes()).hexdigest()
    artifacts["out"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _verify_merkle_binding(artifacts: dict[str, Path]) -> subprocess.CompletedProcess[str]:
    return _run_cli(
        [
            sys.executable,
            str(VERIFY_TOOL),
            "--bundle-manifest",
            str(artifacts["out"]),
            "--bundle-dir",
            str(artifacts["out"].parent),
            "--verify-merkle-binding",
        ]
    )


def test_verify_merkle_binding_accepts_consistent_bundle(tmp_path: Path) -> None:
    artifacts = _write_consistent_bundle(tmp_path)

    verify_result = _verify_merkle_binding(artifacts)

    assert verify_result.returncode == 0, verify_result.stdout + verify_result.stderr
    assert "Merkle binding verified: hash_manifest.csv.gz rows reproduce merkle_roots.json" in verify_result.stdout
    assert "Evidence bundle manifest valid" in verify_result.stdout


def test_verify_merkle_binding_rejects_hash_manifest_edited_after_signing(tmp_path: Path) -> None:
    """The signed roots are untouched; only the manifest rows and the unsigned bundle digest are rewritten."""
    artifacts = _write_consistent_bundle(tmp_path)
    signed_roots = artifacts["roots"].read_bytes()

    with gzip.open(artifacts["hash_manifest"], "rt", encoding="utf-8", newline="") as handle:
        lines = handle.read().splitlines(keepends=True)
    assert len(lines) == 5
    with gzip.open(artifacts["hash_manifest"], "wt", encoding="utf-8", newline="") as handle:
        handle.writelines(lines[:-1])
    _refresh_manifest_digest(artifacts, "hash_manifest", "hash_manifest_sha256")

    verify_result = _verify_merkle_binding(artifacts)

    assert artifacts["roots"].read_bytes() == signed_roots
    assert verify_result.returncode == 11, verify_result.stdout + verify_result.stderr
    assert "Verification failed: hash_manifest.csv.gz rows do not reproduce merkle_roots.json" in verify_result.stdout
    assert "Evidence bundle manifest valid" not in verify_result.stdout


def test_verify_merkle_binding_rejects_summary_that_disagrees_with_rows(tmp_path: Path) -> None:
    artifacts = _write_consistent_bundle(tmp_path)
    summary = json.loads(artifacts["hash_summary"].read_text(encoding="utf-8"))
    summary["rows_total"] += 1
    artifacts["hash_summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest_digest(artifacts, "hash_summary", "hash_summary_sha256")

    verify_result = _verify_merkle_binding(artifacts)

    assert verify_result.returncode == 11, verify_result.stdout + verify_result.stderr
    assert "Verification failed: hash_summary.json disagrees with hash_manifest.csv.gz rows" in verify_result.stdout
    assert "field='rows_total'" in verify_result.stdout


@pytest.mark.parametrize(
    "hash_manifest_bytes",
    [
        b"# hash_algorithm=sha256\norigin_drive,partition,relpath\n",
        gzip.compress(b"# hash_algorithm=sha256\norigin_drive,partition,relpath\nDriveA,Part1,a.txt\n", mtime=0),
        gzip.compress(b"# hash_algorithm=sha256\n" + b"x" * 4096, mtime=0)[:-12],
    ],
    ids=["not-gzip", "missing-columns", "truncated-gzip"],
)
def test_verify_merkle_binding_reports_unreadable_hash_manifest_as_failure(tmp_path: Path, hash_manifest_bytes: bytes) -> None:
    artifacts = _write_consistent_bundle(tmp_path)
    artifacts["hash_manifest"].write_bytes(hash_manifest_bytes)
    _refresh_manifest_digest(artifacts, "hash_manifest", "hash_manifest_sha256")

    verify_result = _verify_merkle_binding(artifacts)

    assert verify_result.returncode == 11, verify_result.stdout + verify_result.stderr
    assert verify_result.stdout.startswith("Verification failed: ")
    assert "Traceback (most recent call last):" not in verify_result.stderr
