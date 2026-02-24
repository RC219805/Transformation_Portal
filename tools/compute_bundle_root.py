#!/usr/bin/env python3
"""
Phase 3.4 compute canonical evidence bundle root digest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from bundle_root_common import (
    BUNDLE_ROOT_ALGORITHM,
    BUNDLE_ROOT_PREIMAGE_VERSION,
    EXPECTED_MANIFEST_FILENAME,
    compute_bundle_root_sha256,
    validate_manifest_structure,
)

EXIT_BUNDLE_MALFORMED = 21
EXIT_BUNDLE_WRITE_FAILURE = 22
EXIT_BUNDLE_ROOT_MISMATCH = 23


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-manifest", required=True, help="Path to evidence_bundle_manifest.json")
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for evidence_bundle_manifest.json (used only with --write)",
    )
    parser.add_argument("--write", action="store_true", help="Write bundle_root fields into output manifest")
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject unknown top-level fields (default: true)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.bundle_manifest)
    if manifest_path.name != EXPECTED_MANIFEST_FILENAME:
        print(f"Malformed manifest: --bundle-manifest must reference {EXPECTED_MANIFEST_FILENAME}")
        return EXIT_BUNDLE_MALFORMED

    if args.out is not None and not args.write:
        print("Malformed manifest: --out requires --write")
        return EXIT_BUNDLE_MALFORMED

    out_path = manifest_path
    if args.write and args.out is not None:
        out_path = Path(args.out)
    if args.write and out_path.name != EXPECTED_MANIFEST_FILENAME:
        print(f"Malformed manifest: --out must reference {EXPECTED_MANIFEST_FILENAME}")
        return EXIT_BUNDLE_MALFORMED

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be a JSON object")
        validate_manifest_structure(manifest, strict=args.strict)
        computed_root = compute_bundle_root_sha256(manifest)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"Malformed manifest: {exc}")
        return EXIT_BUNDLE_MALFORMED

    existing_root = manifest.get("bundle_root_sha256")
    if existing_root is not None and existing_root != computed_root:
        print("Bundle root mismatch: manifest bundle_root_sha256 does not match computed value")
        return EXIT_BUNDLE_ROOT_MISMATCH

    if not args.write:
        print(computed_root)
        return 0

    manifest["bundle_root_algorithm"] = BUNDLE_ROOT_ALGORITHM
    manifest["bundle_root_preimage_version"] = BUNDLE_ROOT_PREIMAGE_VERSION
    manifest["bundle_root_sha256"] = computed_root

    serialized = json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
        separators=(",", ": "),
    ).encode("utf-8")

    try:
        atomic_write(out_path, serialized + b"\n")
    except OSError as exc:
        print(f"Bundle root write failed: {exc}")
        return EXIT_BUNDLE_WRITE_FAILURE

    print(f"Bundle root written to {out_path}: {computed_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
