#!/usr/bin/env python3
"""Build and verify Merkle-backed ingest evidence bundles from batch outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from transformation_portal.ingest.batch import BATCH_MANIFEST_SCHEMA
from transformation_portal.ingest.normalize_machine_json import canonical_json_bytes

EVIDENCE_BUNDLE_SCHEMA = "tp.ingest.evidence_bundle.v1"

EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 2
EXIT_BUILD_FAILURE = 5
EXIT_VERIFICATION_FAILURE = 6


class VerificationError(RuntimeError):
    """Raised when a bundle verification check fails."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load {label} {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _validate_batch_manifest(manifest: dict[str, Any]) -> list[dict[str, str]]:
    if manifest.get("schema") != BATCH_MANIFEST_SCHEMA:
        raise ValueError(f"batch manifest schema must be {BATCH_MANIFEST_SCHEMA}")

    items = manifest.get("items")
    if not isinstance(items, list):
        raise ValueError("batch manifest items must be a list")

    normalized_items: list[dict[str, str]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"batch manifest item at index {index} must be an object")
        normalized_items.append(
            {
                "relative_path": _require_string(item.get("relative_path"), field=f"items[{index}].relative_path"),
                "normalized_json_relpath": _require_string(
                    item.get("normalized_json_relpath"), field=f"items[{index}].normalized_json_relpath"
                ),
                "normalized_json_sha256": _require_string(
                    item.get("normalized_json_sha256"), field=f"items[{index}].normalized_json_sha256"
                ),
            }
        )

    if manifest.get("item_count") != len(normalized_items):
        raise ValueError("batch manifest item_count does not match items length")

    return normalized_items


def _leaf_hash(relative_path: str, normalized_sha256: str) -> str:
    return _sha256_bytes(f"{relative_path}\n{normalized_sha256}\n".encode("utf-8"))


def _build_merkle_levels(leaves: list[str]) -> list[list[str]]:
    if not leaves:
        return [[_sha256_bytes(b"")]]

    levels = [leaves]
    current = leaves
    while len(current) > 1:
        next_level: list[str] = []
        for index in range(0, len(current), 2):
            left = current[index]
            right = current[index + 1] if index + 1 < len(current) else left
            next_level.append(_sha256_bytes(f"{left}{right}".encode("utf-8")))
        levels.append(next_level)
        current = next_level
    return levels


def _build_inclusion_proof(levels: list[list[str]], index: int) -> list[dict[str, str]]:
    proof: list[dict[str, str]] = []
    cursor = index
    for level in levels[:-1]:
        sibling_index = cursor + 1 if cursor % 2 == 0 else cursor - 1
        if sibling_index >= len(level):
            sibling_index = cursor
        proof.append(
            {
                "position": "right" if sibling_index > cursor else "left",
                "hash": level[sibling_index],
            }
        )
        cursor //= 2
    return proof


def _verify_inclusion_proof(leaf_hash: str, proof: list[dict[str, str]], expected_root: str) -> bool:
    digest = leaf_hash
    for step in proof:
        sibling_hash = _require_string(step.get("hash"), field="inclusion_proof.hash")
        position = _require_string(step.get("position"), field="inclusion_proof.position")
        if position == "left":
            digest = _sha256_bytes(f"{sibling_hash}{digest}".encode("utf-8"))
        elif position == "right":
            digest = _sha256_bytes(f"{digest}{sibling_hash}".encode("utf-8"))
        else:
            raise ValueError(f"inclusion proof position must be left/right, got {position}")
    return digest == expected_root


def _compute_manifest_state(manifest_path: Path, manifest: dict[str, Any]) -> tuple[list[dict[str, str]], str, str]:
    items = _validate_batch_manifest(manifest)
    verified_items: list[dict[str, str]] = []

    for item in items:
        relpath = item["normalized_json_relpath"]
        expected_sha = item["normalized_json_sha256"]
        artifact_path = manifest_path.parent / relpath
        if not artifact_path.exists():
            raise VerificationError(f"missing normalized artifact: {relpath}")
        actual_sha = _sha256_file(artifact_path)
        if actual_sha != expected_sha:
            raise VerificationError(f"digest mismatch for normalized artifact: {relpath}")

        verified_items.append(
            {
                "relative_path": item["relative_path"],
                "normalized_json_relpath": relpath,
                "normalized_json_sha256": expected_sha,
                "leaf_sha256": _leaf_hash(item["relative_path"], expected_sha),
            }
        )

    leaves = [item["leaf_sha256"] for item in verified_items]
    merkle_root = _build_merkle_levels(leaves)[-1][0]
    return verified_items, merkle_root, _sha256_file(manifest_path)


def build_bundle(*, batch_manifest_path: Path, output_path: Path, proof_target: str | None) -> dict[str, Any]:
    manifest = _load_json_object(batch_manifest_path, label="batch manifest")
    items, merkle_root, manifest_sha256 = _compute_manifest_state(batch_manifest_path, manifest)

    bundle: dict[str, Any] = {
        "schema": EVIDENCE_BUNDLE_SCHEMA,
        "batch_manifest_schema": manifest["schema"],
        "normalization_profile": manifest["normalization_profile"],
        "item_count": manifest["item_count"],
        "batch_manifest_sha256": manifest_sha256,
        "batch_root_sha256": manifest["batch_root_sha256"],
        "merkle_root_sha256": merkle_root,
        "items": [
            {
                "relative_path": item["relative_path"],
                "normalized_json_sha256": item["normalized_json_sha256"],
                "leaf_sha256": item["leaf_sha256"],
            }
            for item in items
        ],
    }

    if proof_target is not None:
        matching_index = next((index for index, item in enumerate(items) if item["relative_path"] == proof_target), None)
        if matching_index is None:
            raise ValueError(f"proof target not found in manifest items: {proof_target}")
        levels = _build_merkle_levels([item["leaf_sha256"] for item in items])
        proof = _build_inclusion_proof(levels, matching_index)
        bundle["inclusion_proof"] = {
            "relative_path": proof_target,
            "leaf_sha256": items[matching_index]["leaf_sha256"],
            "proof": proof,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(bundle))
    return bundle


def verify_bundle(*, batch_manifest_path: Path, bundle_path: Path) -> None:
    manifest = _load_json_object(batch_manifest_path, label="batch manifest")
    bundle = _load_json_object(bundle_path, label="evidence bundle")

    if bundle.get("schema") != EVIDENCE_BUNDLE_SCHEMA:
        raise VerificationError(f"bundle schema must be {EVIDENCE_BUNDLE_SCHEMA}")

    items, merkle_root, manifest_sha256 = _compute_manifest_state(batch_manifest_path, manifest)
    if bundle.get("batch_manifest_sha256") != manifest_sha256:
        raise VerificationError("batch manifest sha256 mismatch")
    if bundle.get("batch_root_sha256") != manifest.get("batch_root_sha256"):
        raise VerificationError("batch_root_sha256 mismatch")
    if bundle.get("merkle_root_sha256") != merkle_root:
        raise VerificationError("merkle_root_sha256 mismatch")
    if bundle.get("item_count") != len(items):
        raise VerificationError("bundle item_count mismatch")

    inclusion = bundle.get("inclusion_proof")
    if inclusion is not None:
        if not isinstance(inclusion, dict):
            raise VerificationError("inclusion_proof must be an object")
        relative_path = _require_string(inclusion.get("relative_path"), field="inclusion_proof.relative_path")
        proof = inclusion.get("proof")
        if not isinstance(proof, list):
            raise VerificationError("inclusion_proof.proof must be a list")
        indexed_items = {item["relative_path"]: item["leaf_sha256"] for item in items}
        if relative_path not in indexed_items:
            raise VerificationError(f"inclusion proof target missing from manifest: {relative_path}")
        typed_proof = []
        for index, step in enumerate(proof):
            if not isinstance(step, dict):
                raise VerificationError(f"inclusion_proof.proof[{index}] must be an object")
            typed_proof.append(
                {
                    "position": _require_string(step.get("position"), field=f"inclusion_proof.proof[{index}].position"),
                    "hash": _require_string(step.get("hash"), field=f"inclusion_proof.proof[{index}].hash"),
                }
            )
        if not _verify_inclusion_proof(indexed_items[relative_path], typed_proof, merkle_root):
            raise VerificationError("inclusion proof does not validate against merkle root")


def _build_subparser(parent: Any) -> None:
    build_parser = parent.add_parser("build", help="Build an evidence bundle from a batch manifest.")
    build_parser.add_argument("--batch-manifest", required=True, help="Path to batch_manifest.normalized.json.")
    build_parser.add_argument("--out", required=True, help="Output bundle path.")
    build_parser.add_argument(
        "--proof-target",
        default=None,
        help="Optional relative_path entry for inclusion proof generation.",
    )

    verify_parser = parent.add_parser("verify", help="Verify an evidence bundle against batch artifacts.")
    verify_parser.add_argument("--batch-manifest", required=True, help="Path to batch_manifest.normalized.json.")
    verify_parser.add_argument("--bundle", required=True, help="Path to ingest evidence bundle JSON.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _build_subparser(subparsers)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.command == "build":
        try:
            bundle = build_bundle(
                batch_manifest_path=Path(args.batch_manifest),
                output_path=Path(args.out),
                proof_target=args.proof_target,
            )
        except VerificationError as exc:
            print(f"Build failed: {exc}", file=sys.stderr)
            return EXIT_BUILD_FAILURE
        except ValueError as exc:
            print(f"Input error: {exc}", file=sys.stderr)
            return EXIT_INPUT_ERROR
        print(f"Evidence bundle written: items={bundle['item_count']} schema={bundle['schema']}")
        return EXIT_SUCCESS

    if args.command == "verify":
        try:
            verify_bundle(
                batch_manifest_path=Path(args.batch_manifest),
                bundle_path=Path(args.bundle),
            )
        except VerificationError as exc:
            print(f"Verification failed: {exc}", file=sys.stderr)
            return EXIT_VERIFICATION_FAILURE
        except ValueError as exc:
            print(f"Input error: {exc}", file=sys.stderr)
            return EXIT_INPUT_ERROR
        print("Evidence bundle verification passed")
        return EXIT_SUCCESS

    print(f"Unsupported command: {args.command}", file=sys.stderr)
    return EXIT_INPUT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
