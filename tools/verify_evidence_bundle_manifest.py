#!/usr/bin/env python3
"""
Phase 3.3 Verify canonical evidence bundle manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bundle_root_common import (
    EXPECTED_MANIFEST_FILENAME,
    compute_bundle_root_sha256,
    load_merkle_leaf_count,
    sha256_hexdigest,
    validate_manifest_structure,
)

EXIT_BUNDLE_VERIFY_FAILURE = 11
EXIT_BUNDLE_MALFORMED = 12


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
        validate_manifest_structure(manifest, strict=True)
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
            computed = sha256_hexdigest(artifact_path)
            expected = str(manifest[digest_field])
            if computed != expected:
                print(f"Verification failed: digest mismatch for {path_field}")
                return EXIT_BUNDLE_VERIFY_FAILURE

        roots_path = bundle_dir / str(manifest["roots_path"])
        actual_leaf_count = load_merkle_leaf_count(roots_path)
        if actual_leaf_count != int(manifest["merkle_leaf_count"]):
            print("Verification failed: merkle_leaf_count mismatch")
            return EXIT_BUNDLE_VERIFY_FAILURE

        if "bundle_root_sha256" in manifest:
            computed_root = compute_bundle_root_sha256(manifest)
            expected_root = str(manifest["bundle_root_sha256"])
            if computed_root != expected_root:
                print("Verification failed: bundle_root_sha256 mismatch")
                return EXIT_BUNDLE_VERIFY_FAILURE

        notarization_block = manifest.get("notarization")
        if isinstance(notarization_block, dict):
            rfc3161 = notarization_block.get("rfc3161")
            if isinstance(rfc3161, dict):
                timestamp_path = bundle_dir / str(rfc3161["timestamp_path"])
                if sha256_hexdigest(timestamp_path) != str(rfc3161["timestamp_sha256"]):
                    print("Verification failed: digest mismatch for notarization.rfc3161.timestamp_path")
                    return EXIT_BUNDLE_VERIFY_FAILURE

            sigstore = notarization_block.get("sigstore")
            if isinstance(sigstore, dict):
                bundle_path = bundle_dir / str(sigstore["bundle_path"])
                if sha256_hexdigest(bundle_path) != str(sigstore["bundle_sha256"]):
                    print("Verification failed: digest mismatch for notarization.sigstore.bundle_path")
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
