#!/usr/bin/env python3
"""
Phase 3.3 Verify canonical evidence bundle manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

EXIT_BUNDLE_VERIFY_FAILURE = 11
EXIT_BUNDLE_MALFORMED = 12
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
HEX64_RE = re.compile(r"^[a-f0-9]{64}$")

REQUIRED_FIELDS = {
    "bundle_version",
    "hash_algorithm",
    "roots_path",
    "roots_sha256",
    "hash_manifest_path",
    "hash_manifest_sha256",
    "hash_summary_path",
    "hash_summary_sha256",
    "signature_path",
    "signature_sha256",
    "timestamp_target",
    "timestamp_path",
    "timestamp_sha256",
    "merkle_leaf_count",
    "phase3_version",
    "phase3_1_version",
    "phase3_2_version",
    "bundle_tool_name",
    "bundle_tool_version",
}


def _sha256_hexdigest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_merkle_leaf_count(roots_path: Path) -> int:
    roots_payload = json.loads(roots_path.read_text(encoding="utf-8"))
    if not isinstance(roots_payload, dict):
        raise ValueError(f"{EXPECTED_ROOTS_FILENAME} must be a JSON object")

    global_block = roots_payload.get("global")
    if not isinstance(global_block, dict):
        raise ValueError(f"{EXPECTED_ROOTS_FILENAME} missing object field: global")

    leaf_count = global_block.get("leaf_count")
    if type(leaf_count) is not int or leaf_count < 0:
        raise ValueError(f"{EXPECTED_ROOTS_FILENAME}.global.leaf_count must be a non-negative integer")
    return leaf_count


def _require_string_field(manifest: dict[str, object], field: str) -> str:
    value = manifest.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_hex_digest(manifest: dict[str, object], field: str) -> str:
    value = _require_string_field(manifest, field)
    if HEX64_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be 64 lowercase hex characters")
    return value


def _validate_manifest_structure(manifest: dict[str, object]) -> None:
    keys = set(manifest)
    missing = sorted(REQUIRED_FIELDS - keys)
    if missing:
        raise ValueError(f"missing required field(s): {', '.join(missing)}")
    unexpected = sorted(keys - REQUIRED_FIELDS)
    if unexpected:
        raise ValueError(f"unexpected field(s): {', '.join(unexpected)}")

    if manifest["bundle_version"] != BUNDLE_VERSION:
        raise ValueError(f"bundle_version must be {BUNDLE_VERSION!r}")
    if manifest["hash_algorithm"] != HASH_ALGORITHM:
        raise ValueError(f"hash_algorithm must be {HASH_ALGORITHM!r}")

    if manifest["roots_path"] != EXPECTED_ROOTS_FILENAME:
        raise ValueError(f"roots_path must be {EXPECTED_ROOTS_FILENAME!r}")
    if manifest["hash_manifest_path"] != EXPECTED_HASH_MANIFEST_FILENAME:
        raise ValueError(f"hash_manifest_path must be {EXPECTED_HASH_MANIFEST_FILENAME!r}")
    if manifest["hash_summary_path"] != EXPECTED_HASH_SUMMARY_FILENAME:
        raise ValueError(f"hash_summary_path must be {EXPECTED_HASH_SUMMARY_FILENAME!r}")
    if manifest["signature_path"] != EXPECTED_SIGNATURE_FILENAME:
        raise ValueError(f"signature_path must be {EXPECTED_SIGNATURE_FILENAME!r}")

    timestamp_target = manifest["timestamp_target"]
    if timestamp_target not in TIMESTAMP_FILENAME_BY_TARGET:
        raise ValueError("timestamp_target must be 'roots' or 'signature'")
    expected_timestamp_path = TIMESTAMP_FILENAME_BY_TARGET[str(timestamp_target)]
    if manifest["timestamp_path"] != expected_timestamp_path:
        raise ValueError(f"timestamp_path must be {expected_timestamp_path!r} for timestamp_target={timestamp_target!r}")

    _require_hex_digest(manifest, "roots_sha256")
    _require_hex_digest(manifest, "hash_manifest_sha256")
    _require_hex_digest(manifest, "hash_summary_sha256")
    _require_hex_digest(manifest, "signature_sha256")
    _require_hex_digest(manifest, "timestamp_sha256")

    merkle_leaf_count = manifest["merkle_leaf_count"]
    if type(merkle_leaf_count) is not int or merkle_leaf_count < 0:
        raise ValueError("merkle_leaf_count must be a non-negative integer")

    _require_string_field(manifest, "phase3_version")
    _require_string_field(manifest, "phase3_1_version")
    _require_string_field(manifest, "phase3_2_version")
    _require_string_field(manifest, "bundle_tool_name")
    _require_string_field(manifest, "bundle_tool_version")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-manifest", required=True, help="Path to evidence_bundle_manifest.json")
    parser.add_argument(
        "--bundle-dir",
        default=None,
        help="Directory containing bundle artifacts (defaults to bundle-manifest parent directory)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.bundle_manifest)
    bundle_dir = Path(args.bundle_dir) if args.bundle_dir is not None else manifest_path.parent

    if manifest_path.name != EXPECTED_MANIFEST_FILENAME:
        print(f"Malformed manifest: --bundle-manifest must reference {EXPECTED_MANIFEST_FILENAME}")
        return EXIT_BUNDLE_MALFORMED

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be a JSON object")
        _validate_manifest_structure(manifest)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"Malformed manifest: {exc}")
        return EXIT_BUNDLE_MALFORMED

    digest_fields = [
        ("roots_path", "roots_sha256"),
        ("hash_manifest_path", "hash_manifest_sha256"),
        ("hash_summary_path", "hash_summary_sha256"),
        ("signature_path", "signature_sha256"),
        ("timestamp_path", "timestamp_sha256"),
    ]

    try:
        for path_field, digest_field in digest_fields:
            artifact_path = bundle_dir / str(manifest[path_field])
            computed = _sha256_hexdigest(artifact_path)
            expected = str(manifest[digest_field])
            if computed != expected:
                print(f"Verification failed: digest mismatch for {path_field}")
                return EXIT_BUNDLE_VERIFY_FAILURE

        roots_path = bundle_dir / str(manifest["roots_path"])
        actual_leaf_count = _load_merkle_leaf_count(roots_path)
        if actual_leaf_count != int(manifest["merkle_leaf_count"]):
            print("Verification failed: merkle_leaf_count mismatch")
            return EXIT_BUNDLE_VERIFY_FAILURE

        print("Evidence bundle manifest valid")
        return 0
    except OSError as exc:
        print(f"Verification failed: {exc}")
        return EXIT_BUNDLE_VERIFY_FAILURE
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"Verification failed: {exc}")
        return EXIT_BUNDLE_VERIFY_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
