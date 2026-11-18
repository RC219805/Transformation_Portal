#!/usr/bin/env python3
"""
Phase 3.3 Verify canonical evidence bundle manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import zlib
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


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return payload


def _describe_binding_issue(issue: dict[str, object]) -> str:
    return ", ".join(f"{key}={issue[key]!r}" for key in sorted(issue))


def find_merkle_binding_failure(bundle_dir: Path, manifest: dict[str, object]) -> str | None:
    """Recompute the bundled Merkle roots and summary counts from the bundled hash manifest rows.

    Digest checks only prove that the artifacts match the (unsigned) bundle manifest. The signature
    and timestamp cover ``merkle_roots.json``, so the hash manifest is anchored only once its rows
    are shown to reproduce those roots.
    """
    from verify_hash_manifest import (  # pylint: disable=import-outside-toplevel
        _open_manifest_reader,
        check_hash_summary_binding,
        check_merkle_binding,
    )

    hash_manifest_name = str(manifest["hash_manifest_path"])
    try:
        rows = _open_manifest_reader(bundle_dir / hash_manifest_name)
    except SystemExit as exc:  # the row reader reports malformed manifests through SystemExit(message)
        raise ValueError(str(exc)) from exc
    except (EOFError, zlib.error, csv.Error) as exc:
        raise ValueError(f"unreadable {hash_manifest_name}: {exc}") from exc

    roots_name = str(manifest["roots_path"])
    merkle_issues = check_merkle_binding(rows, _load_json_object(bundle_dir / roots_name))
    if merkle_issues:
        return (
            f"{hash_manifest_name} rows do not reproduce {roots_name} "
            f"({len(merkle_issues)} issue(s); first: {_describe_binding_issue(merkle_issues[0])})"
        )

    summary_name = str(manifest["hash_summary_path"])
    summary_issues = check_hash_summary_binding(rows, _load_json_object(bundle_dir / summary_name))
    if summary_issues:
        return (
            f"{summary_name} disagrees with {hash_manifest_name} rows "
            f"({len(summary_issues)} issue(s); first: {_describe_binding_issue(summary_issues[0])})"
        )
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-manifest", required=True, help="Path to evidence_bundle_manifest.json")
    parser.add_argument(
        "--bundle-dir",
        default=None,
        help="Directory containing bundle artifacts (defaults to bundle-manifest parent directory)",
    )
    parser.add_argument(
        "--verify-merkle-binding",
        action="store_true",
        help=(
            "Also recompute the Merkle roots and summary counts from the bundled hash manifest rows and fail "
            "when they do not reproduce the bundled merkle_roots.json / hash_summary.json"
        ),
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

        if args.verify_merkle_binding:
            binding_failure = find_merkle_binding_failure(bundle_dir, manifest)
            if binding_failure is not None:
                print(f"Verification failed: {binding_failure}")
                return EXIT_BUNDLE_VERIFY_FAILURE
            print(f"Merkle binding verified: {manifest['hash_manifest_path']} rows reproduce {manifest['roots_path']}")

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
