"""Pure Phase 4D helpers for metadata object hashing and manifest assembly."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .exceptions import (
    MetadataManifestInputError,
    MetadataManifestSchemaValidationError,
    MetadataSchemaValidationError,
)
from .validation_helpers import (
    require_sorted_relative_paths,
    require_unique_relative_paths,
    validate_payload_with_schema,
    validate_records_with_schema,
)

METADATA_CONTRACT_VERSION = "tp.meta.capture.v1"
METADATA_MANIFEST_CONTRACT_VERSION = "tp.meta.capture_manifest.v1"


def canonical_json_bytes(payload: Any, *, trailing_newline: bool = False) -> bytes:
    """Serialize payload using canonical JSON settings required by Phase 4."""
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if trailing_newline:
        return raw + b"\n"
    return raw


def compute_metadata_sha256(metadata_object: dict[str, Any]) -> str:
    """Compute canonical object-level metadata digest."""
    return hashlib.sha256(canonical_json_bytes(metadata_object)).hexdigest()


def _validate_metadata_records(records: list[dict[str, Any]], metadata_schema: dict[str, Any]) -> None:
    validate_records_with_schema(
        records,
        metadata_schema,
        error_cls=MetadataSchemaValidationError,
        label="metadata",
    )


def _validate_metadata_manifest(payload: dict[str, Any], manifest_schema: dict[str, Any]) -> None:
    validate_payload_with_schema(
        payload,
        manifest_schema,
        error_cls=MetadataManifestSchemaValidationError,
        label="metadata manifest",
    )


def _require_metadata_contract_version(records: list[dict[str, Any]]) -> None:
    for index, record in enumerate(records):
        contract_version = record.get("metadata_contract_version")
        if contract_version != METADATA_CONTRACT_VERSION:
            relative_path = record.get("relative_path", "<unknown>")
            raise MetadataManifestInputError(
                f"record[{index}] contract mismatch for {relative_path}: "
                f"expected {METADATA_CONTRACT_VERSION}, got {contract_version!r}"
            )


def _require_fingerprint_match(records: list[dict[str, Any]], expected_fingerprint: str) -> None:
    for index, record in enumerate(records):
        extractor = record.get("extractor")
        if not isinstance(extractor, dict):
            raise MetadataManifestInputError(f"record[{index}] missing extractor object")
        fingerprint = extractor.get("config_fingerprint_sha256")
        if fingerprint != expected_fingerprint:
            relative_path = record.get("relative_path", "<unknown>")
            raise MetadataManifestInputError(
                f"record[{index}] fingerprint mismatch for {relative_path}: "
                f"expected {expected_fingerprint}, got {fingerprint!r}"
            )


def build_metadata_manifest_payload(
    records: list[dict[str, Any]],
    *,
    metadata_schema: dict[str, Any],
    manifest_schema: dict[str, Any],
    strict_input_order: bool = True,
    required_config_fingerprint_sha256: str | None = None,
) -> dict[str, Any]:
    """Build validated metadata manifest payload from validated capture metadata records."""
    if not isinstance(records, list):
        raise MetadataManifestInputError("input metadata payload must be a JSON array")

    _validate_metadata_records(records, metadata_schema=metadata_schema)
    _require_metadata_contract_version(records)
    require_unique_relative_paths(records, label="input metadata array", error_cls=MetadataManifestInputError)
    if strict_input_order:
        require_sorted_relative_paths(records, label="input metadata array", error_cls=MetadataManifestInputError)
    if required_config_fingerprint_sha256 is not None:
        _require_fingerprint_match(records, expected_fingerprint=required_config_fingerprint_sha256)

    sorted_records = sorted(records, key=lambda record: record["relative_path"])
    entries: list[dict[str, str]] = []
    for record in sorted_records:
        relative_path = record["relative_path"]
        file_sha256 = record["file_sha256"]
        try:
            metadata_sha256 = compute_metadata_sha256(record)
        except (TypeError, ValueError) as exc:
            raise MetadataSchemaValidationError(
                f"canonical serialization failed for record with relative_path={relative_path!r}: {exc}"
            ) from exc
        entries.append(
            {
                "relative_path": relative_path,
                "file_sha256": file_sha256,
                "metadata_sha256": metadata_sha256,
            }
        )

    payload = {
        "metadata_manifest_contract_version": METADATA_MANIFEST_CONTRACT_VERSION,
        "metadata_contract_version": METADATA_CONTRACT_VERSION,
        "entries": entries,
    }
    _validate_metadata_manifest(payload, manifest_schema=manifest_schema)
    return payload


def serialize_metadata_manifest(payload: dict[str, Any]) -> bytes:
    """Serialize metadata manifest payload with canonical bytes and trailing newline."""
    return canonical_json_bytes(payload, trailing_newline=True)
