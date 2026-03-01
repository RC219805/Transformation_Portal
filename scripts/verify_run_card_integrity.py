#!/usr/bin/env python3
"""Verify run-card integrity invariants.

Checks:
1. Run card validates against run_card.v1 schema.
2. Each artifact entry includes a valid lowercase SHA256 digest.
3. artifact_merkle_root matches recomputed value from artifact_index.
4. artifact_index ordering is lexicographically sorted by relative_path.
5. Optional canonical JSON serialization check (sorted keys + indent=2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from json import JSONDecodeError
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover - runtime guard for environments missing optional deps
    Draft202012Validator = None  # type: ignore[assignment]


SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "docs" / "schemas" / "run_card" / "run_card.v1.schema.json"


def _load_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle), None
    except FileNotFoundError:
        return None, f"JSON file not found: {path}"
    except PermissionError:
        return None, f"Permission denied reading JSON file: {path}"
    except JSONDecodeError as exc:
        return None, f"Invalid JSON in {path}: {exc.msg} (line {exc.lineno}, column {exc.colno})"
    except OSError as exc:
        return None, f"Failed to read JSON file {path}: {exc}"


def _canonical_json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _format_error_path(error_path: Any) -> str:
    parts = [str(item) for item in error_path]
    return ".".join(parts) if parts else "<root>"


def compute_artifact_merkle_root(artifact_index: list[dict[str, Any]]) -> str:
    """Compute deterministic Merkle root from artifact SHA256 digests."""
    sorted_artifacts = sorted(artifact_index, key=lambda item: item["relative_path"])
    leaf_hashes = [bytes.fromhex(item["sha256"]) for item in sorted_artifacts]
    return hashlib.sha256(b"".join(leaf_hashes)).hexdigest()


def verify_run_card_integrity(
    run_card_path: Path,
    *,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    check_canonical_json: bool = False,
) -> list[str]:
    """Return list of integrity errors for a run card."""
    errors: list[str] = []

    if not run_card_path.exists():
        return [f"Run card not found: {run_card_path}"]
    if not schema_path.exists():
        return [f"Run card schema not found: {schema_path}"]
    if Draft202012Validator is None:
        return ["jsonschema dependency is required (install jsonschema>=4.21.0,<5)"]

    run_card_payload, run_card_load_error = _load_json(run_card_path)
    if run_card_load_error:
        return [run_card_load_error]
    schema_payload, schema_load_error = _load_json(schema_path)
    if schema_load_error:
        return [schema_load_error]

    validator = Draft202012Validator(schema_payload)
    schema_errors = sorted(validator.iter_errors(run_card_payload), key=lambda item: list(item.path))
    for item in schema_errors:
        errors.append(f"Schema validation failed at {_format_error_path(item.path)}: {item.message}")

    artifact_index = run_card_payload.get("artifact_index")
    if not isinstance(artifact_index, list):
        errors.append("artifact_index must be an array")
        return errors

    relative_paths: list[str] = []
    for index, artifact in enumerate(artifact_index):
        if not isinstance(artifact, dict):
            errors.append(f"artifact_index[{index}] must be an object")
            continue

        relative_path = artifact.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            errors.append(f"artifact_index[{index}].relative_path must be a non-empty string")
        else:
            relative_paths.append(relative_path)

        sha256_hex = artifact.get("sha256")
        if not isinstance(sha256_hex, str) or not SHA256_HEX_RE.fullmatch(sha256_hex):
            errors.append(f"artifact_index[{index}].sha256 must be a lowercase 64-char hex digest")

    if relative_paths:
        sorted_paths = sorted(relative_paths)
        if relative_paths != sorted_paths:
            errors.append("artifact_index ordering is non-deterministic (relative_path must be lexicographically sorted)")
        if len(set(relative_paths)) != len(relative_paths):
            errors.append("artifact_index contains duplicate relative_path entries")

    expected_merkle_root = run_card_payload.get("artifact_merkle_root")
    if not isinstance(expected_merkle_root, str) or not SHA256_HEX_RE.fullmatch(expected_merkle_root):
        errors.append("artifact_merkle_root must be a lowercase 64-char hex digest")
    elif not any("artifact_index[" in error for error in errors):
        recomputed_merkle_root = compute_artifact_merkle_root(artifact_index)
        if recomputed_merkle_root != expected_merkle_root:
            errors.append(
                "artifact_merkle_root mismatch: " f"expected={expected_merkle_root}, recomputed={recomputed_merkle_root}"
            )

    if check_canonical_json:
        raw_text = run_card_path.read_text(encoding="utf-8")
        canonical = _canonical_json_text(run_card_payload)
        if raw_text not in (canonical, canonical + "\n"):
            errors.append("JSON canonical serialization drift detected (expected sort_keys=True, indent=2)")

    return errors


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify run-card integrity invariants.")
    parser.add_argument("run_cards", nargs="+", help="Run card JSON file path(s)")
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help=f"Path to run-card schema (default: {DEFAULT_SCHEMA_PATH})",
    )
    parser.add_argument(
        "--check-canonical-json",
        action="store_true",
        help="Fail if file text is not canonical JSON serialization (sort_keys=True, indent=2).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    exit_code = 0

    for run_card_arg in args.run_cards:
        run_card_path = Path(run_card_arg)
        errors = verify_run_card_integrity(
            run_card_path,
            schema_path=args.schema_path,
            check_canonical_json=args.check_canonical_json,
        )
        if errors:
            exit_code = 1
            print(f"❌ Run card integrity verification failed: {run_card_path}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"✅ Run card integrity verified: {run_card_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
