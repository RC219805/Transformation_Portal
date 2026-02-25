#!/usr/bin/env python3
"""Phase 4E provenance manifest builder."""

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

from tp.phase4.canonicalize_capture_metadata import (  # noqa: E402
    ConfigValidationError,
    compute_config_fingerprint_sha256,
    load_capture_metadata_config,
)
from tp.phase4.provenance_capture import (  # noqa: E402
    ProvenanceInputError,
    ProvenanceSchemaValidationError,
    build_provenance_manifest_payload,
    serialize_provenance_manifest,
)

EXIT_SUCCESS = 0
EXIT_INPUT_PARSE_ERROR = 2
EXIT_INPUT_INVARIANT_FAILURE = 3
EXIT_SCHEMA_VALIDATION_FAILURE = 4
EXIT_MANIFEST_WRITE_FAILURE = 5

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "tools" / "capture_metadata_config.json"
DEFAULT_METADATA_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "phase4" / "metadata.schema.json"
DEFAULT_METADATA_MANIFEST_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "phase4" / "metadata_manifest.schema.json"
DEFAULT_PROVENANCE_MANIFEST_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "phase4" / "provenance_manifest.schema.json"


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
        "--capture-metadata",
        required=True,
        help="Input Phase 4C artifact path (capture_metadata.tp.meta.capture.v1.json).",
    )
    parser.add_argument(
        "--metadata-manifest",
        required=True,
        help="Input Phase 4D artifact path (metadata_manifest.tp.meta.capture_manifest.v1.json).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path (provenance_manifest.tp.meta.provenance.v1.json).",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Canonicalization config path (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--strict-input-order",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require capture metadata and metadata manifest to already be sorted by relative_path (default: true).",
    )
    parser.add_argument(
        "--require-fingerprint-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require extractor config fingerprint match with current config (default: true).",
    )
    return parser.parse_args()


def _load_json_file(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load {label} {path}: {exc}") from exc


def main() -> int:
    args = parse_args()
    capture_metadata_path = Path(args.capture_metadata)
    metadata_manifest_path = Path(args.metadata_manifest)
    out_path = Path(args.out)
    config_path = Path(args.config)

    try:
        capture_payload = _load_json_file(capture_metadata_path, label="capture metadata artifact")
        metadata_manifest_payload = _load_json_file(metadata_manifest_path, label="metadata manifest artifact")
    except ValueError as exc:
        print(f"Input read/parse error: {exc}", file=sys.stderr)
        return EXIT_INPUT_PARSE_ERROR

    if not isinstance(capture_payload, list):
        print("Input invariant failure: capture metadata payload must be a JSON array", file=sys.stderr)
        return EXIT_INPUT_INVARIANT_FAILURE
    if not isinstance(metadata_manifest_payload, dict):
        print("Input invariant failure: metadata manifest payload must be a JSON object", file=sys.stderr)
        return EXIT_INPUT_INVARIANT_FAILURE

    try:
        metadata_schema = _load_json_file(DEFAULT_METADATA_SCHEMA_PATH, label="metadata schema")
        metadata_manifest_schema = _load_json_file(DEFAULT_METADATA_MANIFEST_SCHEMA_PATH, label="metadata manifest schema")
        provenance_manifest_schema = _load_json_file(
            DEFAULT_PROVENANCE_MANIFEST_SCHEMA_PATH, label="provenance manifest schema"
        )
    except ValueError as exc:
        print(f"Schema validation failure: {exc}", file=sys.stderr)
        return EXIT_SCHEMA_VALIDATION_FAILURE

    expected_fingerprint: str | None = None
    if args.require_fingerprint_match:
        try:
            config_payload = load_capture_metadata_config(config_path)
        except ConfigValidationError as exc:
            print(f"Input invariant failure: invalid capture metadata config: {exc}", file=sys.stderr)
            return EXIT_INPUT_INVARIANT_FAILURE
        expected_fingerprint = compute_config_fingerprint_sha256(config_payload)

    try:
        provenance_manifest_payload = build_provenance_manifest_payload(
            capture_payload,
            metadata_manifest_payload,
            metadata_schema=metadata_schema,
            metadata_manifest_schema=metadata_manifest_schema,
            provenance_manifest_schema=provenance_manifest_schema,
            strict_input_order=args.strict_input_order,
            required_config_fingerprint_sha256=expected_fingerprint,
        )
    except ProvenanceInputError as exc:
        print(f"Input invariant failure: {exc}", file=sys.stderr)
        return EXIT_INPUT_INVARIANT_FAILURE
    except ProvenanceSchemaValidationError as exc:
        print(f"Schema validation failure: {exc}", file=sys.stderr)
        return EXIT_SCHEMA_VALIDATION_FAILURE

    try:
        _atomic_write_bytes(out_path, serialize_provenance_manifest(provenance_manifest_payload))
    except OSError as exc:
        print(f"Manifest write failure: {exc}", file=sys.stderr)
        return EXIT_MANIFEST_WRITE_FAILURE

    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
