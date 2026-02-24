"""
Shared fixture helpers for bundle-root determinism and parity checks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXPECTED_BUNDLE_ROOT_SHA256 = "47c09af843470b891e8c33d614fb5b4a4399218fdc7b8461f5c6b11d1fa000ce"


def write_bundle_fixture_artifacts(
    bundle_dir: Path,
    *,
    timestamp_target: str = "signature",
) -> dict[str, Path]:
    """
    Write deterministic fixture artifacts matching Phase 3.4 root tests.
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)

    roots_path = bundle_dir / "merkle_roots.json"
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

    hash_manifest_path = bundle_dir / "hash_manifest.csv.gz"
    hash_manifest_path.write_bytes(
        b"# hash_algorithm=sha256\n"
        b"origin_drive,partition,relpath,filesize_bytes,sha256,hash_status,error\n"
        b"driveA,partA,a.jpg,5,abc,ok,\n"
    )

    hash_summary_path = bundle_dir / "hash_summary.json"
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

    signature_path = bundle_dir / "merkle_roots.sig.json"
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

    if timestamp_target not in {"signature", "roots"}:
        raise ValueError("timestamp_target must be 'signature' or 'roots'")
    timestamp_filename = "merkle_roots.sig.tsr" if timestamp_target == "signature" else "merkle_roots.tsr"
    timestamp_path = bundle_dir / timestamp_filename
    timestamp_path.write_bytes(b"\x30\x03\x30\x01\x00")

    return {
        "roots": roots_path,
        "hash_manifest": hash_manifest_path,
        "hash_summary": hash_summary_path,
        "signature": signature_path,
        "timestamp": timestamp_path,
        "out": bundle_dir / "evidence_bundle_manifest.json",
    }
