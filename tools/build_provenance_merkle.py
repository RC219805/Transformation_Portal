#!/usr/bin/env python3
"""Phase 4E provenance merkle root builder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tp.phase4.provenance_capture import (  # noqa: E402
    ProvenanceInputError,
    ProvenanceMerkleSchemaValidationError,
    ProvenanceSchemaValidationError,
    build_provenance_merkle_payload,
    serialize_provenance_merkle,
)

EXIT_SUCCESS = 0
EXIT_INPUT_PARSE_ERROR = 2
EXIT_INPUT_INVARIANT_FAILURE = 3
EXIT_SCHEMA_VALIDATION_FAILURE = 4
EXIT_MERKLE_WRITE_FAILURE = 5

DEFAULT_PROVENANCE_MANIFEST_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "phase4" / "provenance_manifest.schema.json"
DEFAULT_PROVENANCE_MERKLE_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "phase4" / "provenance_merkle.schema.json"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        tmp_path.write_bytes(data)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Input path (provenance_manifest.tp.meta.provenance.v1.json).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path (provenance_merkle.tp.meta.provenance_merkle.v1.json).",
    )
    parser.add_argument(
        "--strict-input-order",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require provenance manifest entries to already be sorted by relative_path (default: true).",
    )
    return parser.parse_args()


def _load_json_file(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load {label} {path}: {exc}") from exc


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    out_path = Path(args.out)

    try:
        provenance_manifest_payload = _load_json_file(input_path, label="provenance manifest artifact")
    except ValueError as exc:
        print(f"Input read/parse error: {exc}", file=sys.stderr)
        return EXIT_INPUT_PARSE_ERROR

    if not isinstance(provenance_manifest_payload, dict):
        print("Input invariant failure: provenance manifest payload must be a JSON object", file=sys.stderr)
        return EXIT_INPUT_INVARIANT_FAILURE

    try:
        provenance_manifest_schema = _load_json_file(
            DEFAULT_PROVENANCE_MANIFEST_SCHEMA_PATH, label="provenance manifest schema"
        )
        provenance_merkle_schema = _load_json_file(DEFAULT_PROVENANCE_MERKLE_SCHEMA_PATH, label="provenance merkle schema")
    except ValueError as exc:
        print(f"Schema validation failure: {exc}", file=sys.stderr)
        return EXIT_SCHEMA_VALIDATION_FAILURE

    try:
        provenance_merkle_payload = build_provenance_merkle_payload(
            provenance_manifest_payload,
            provenance_manifest_schema=provenance_manifest_schema,
            provenance_merkle_schema=provenance_merkle_schema,
            strict_input_order=args.strict_input_order,
        )
    except ProvenanceInputError as exc:
        print(f"Input invariant failure: {exc}", file=sys.stderr)
        return EXIT_INPUT_INVARIANT_FAILURE
    except (ProvenanceSchemaValidationError, ProvenanceMerkleSchemaValidationError) as exc:
        print(f"Schema validation failure: {exc}", file=sys.stderr)
        return EXIT_SCHEMA_VALIDATION_FAILURE

    try:
        _atomic_write_bytes(out_path, serialize_provenance_merkle(provenance_merkle_payload))
    except OSError as exc:
        print(f"Merkle write failure: {exc}", file=sys.stderr)
        return EXIT_MERKLE_WRITE_FAILURE

    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
