#!/usr/bin/env python3
"""
Phase 3.3 Generate canonical evidence bundle manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid4

EXIT_BUNDLE_BUILD_FAILURE = 10
BUNDLE_VERSION = "1"
HASH_ALGORITHM = "sha256"
EXPECTED_MANIFEST_FILENAME = "evidence_bundle_manifest.json"
EXPECTED_ROOTS_FILENAME = "merkle_roots.json"
EXPECTED_HASH_MANIFEST_FILENAME = "hash_manifest.csv.gz"
EXPECTED_HASH_SUMMARY_FILENAME = "hash_summary.json"
EXPECTED_SIGNATURE_FILENAME = "merkle_roots.sig.json"
TIMESTAMP_FILENAME_BY_TARGET = {
    "roots": "merkle_roots.tsr",
    "signature": "merkle_roots.sig.tsr",
}


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _sha256_hexdigest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _require_expected_filename(path: Path, expected: str, arg_name: str) -> None:
    if path.name != expected:
        raise ValueError(f"--{arg_name} must reference {expected}")


def _load_merkle_leaf_count(roots_path: Path) -> int:
    try:
        roots_payload = json.loads(roots_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to parse {EXPECTED_ROOTS_FILENAME}: {exc}") from exc

    if not isinstance(roots_payload, dict):
        raise ValueError(f"{EXPECTED_ROOTS_FILENAME} must be a JSON object")

    global_block = roots_payload.get("global")
    if not isinstance(global_block, dict):
        raise ValueError(f"{EXPECTED_ROOTS_FILENAME} missing object field: global")

    leaf_count = global_block.get("leaf_count")
    if type(leaf_count) is not int or leaf_count < 0:
        raise ValueError(f"{EXPECTED_ROOTS_FILENAME}.global.leaf_count must be a non-negative integer")
    return leaf_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", required=True, help="Path to merkle_roots.json")
    parser.add_argument("--hash-manifest", required=True, help="Path to hash_manifest.csv.gz")
    parser.add_argument("--hash-summary", required=True, help="Path to hash_summary.json")
    parser.add_argument("--signature", required=True, help="Path to merkle_roots.sig.json")
    parser.add_argument("--timestamp-target", required=True, choices=sorted(TIMESTAMP_FILENAME_BY_TARGET))
    parser.add_argument("--timestamp", required=True, help="Path to detached .tsr")
    parser.add_argument("--phase3-version", default="1", help="Phase 3 contract version")
    parser.add_argument("--phase3-1-version", default="1", help="Phase 3.1 contract version")
    parser.add_argument("--phase3-2-version", default="1", help="Phase 3.2 contract version")
    parser.add_argument(
        "--bundle-tool-name",
        default="phase3_bundle_builder",
        help="Bundle-manifest emitting tool name",
    )
    parser.add_argument(
        "--bundle-tool-version",
        default="1.0.0",
        help="Bundle-manifest emitting tool version",
    )
    parser.add_argument("--out", required=True, help="Output path for evidence_bundle_manifest.json")
    args = parser.parse_args()

    roots_path = Path(args.roots)
    hash_manifest_path = Path(args.hash_manifest)
    hash_summary_path = Path(args.hash_summary)
    signature_path = Path(args.signature)
    timestamp_path = Path(args.timestamp)
    out_path = Path(args.out)

    try:
        _require_expected_filename(roots_path, EXPECTED_ROOTS_FILENAME, "roots")
        _require_expected_filename(hash_manifest_path, EXPECTED_HASH_MANIFEST_FILENAME, "hash-manifest")
        _require_expected_filename(hash_summary_path, EXPECTED_HASH_SUMMARY_FILENAME, "hash-summary")
        _require_expected_filename(signature_path, EXPECTED_SIGNATURE_FILENAME, "signature")

        expected_timestamp_name = TIMESTAMP_FILENAME_BY_TARGET[args.timestamp_target]
        _require_expected_filename(timestamp_path, expected_timestamp_name, "timestamp")
        _require_expected_filename(out_path, EXPECTED_MANIFEST_FILENAME, "out")

        if not args.bundle_tool_name.strip():
            raise ValueError("--bundle-tool-name must be non-empty")
        if not args.bundle_tool_version.strip():
            raise ValueError("--bundle-tool-version must be non-empty")
        if not args.phase3_version.strip():
            raise ValueError("--phase3-version must be non-empty")
        if not args.phase3_1_version.strip():
            raise ValueError("--phase3-1-version must be non-empty")
        if not args.phase3_2_version.strip():
            raise ValueError("--phase3-2-version must be non-empty")

        merkle_leaf_count = _load_merkle_leaf_count(roots_path)
        manifest = {
            "bundle_version": BUNDLE_VERSION,
            "hash_algorithm": HASH_ALGORITHM,
            "roots_path": roots_path.name,
            "roots_sha256": _sha256_hexdigest(roots_path),
            "hash_manifest_path": hash_manifest_path.name,
            "hash_manifest_sha256": _sha256_hexdigest(hash_manifest_path),
            "hash_summary_path": hash_summary_path.name,
            "hash_summary_sha256": _sha256_hexdigest(hash_summary_path),
            "signature_path": signature_path.name,
            "signature_sha256": _sha256_hexdigest(signature_path),
            "timestamp_target": args.timestamp_target,
            "timestamp_path": timestamp_path.name,
            "timestamp_sha256": _sha256_hexdigest(timestamp_path),
            "merkle_leaf_count": merkle_leaf_count,
            "phase3_version": args.phase3_version.strip(),
            "phase3_1_version": args.phase3_1_version.strip(),
            "phase3_2_version": args.phase3_2_version.strip(),
            "bundle_tool_name": args.bundle_tool_name.strip(),
            "bundle_tool_version": args.bundle_tool_version.strip(),
        }

        serialized = json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        ).encode("utf-8")
        atomic_write(out_path, serialized + b"\n")

        print(f"Evidence bundle manifest written to {out_path}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"Bundle generation failed: {exc}")
        return EXIT_BUNDLE_BUILD_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
