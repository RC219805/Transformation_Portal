"""Pure Phase 4D helpers for metadata object hashing and manifest assembly."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .schema_validation import build_draft202012_validator

METADATA_CONTRACT_VERSION = "tp.meta.capture.v1"
METADATA_MANIFEST_CONTRACT_VERSION = "tp.meta.capture_manifest.v1"


class MetadataManifestInputError(ValueError):
    """Raised when metadata input violates Phase 4D invariants."""


class MetadataSchemaValidationError(ValueError):
    """Raised when capture metadata fails schema validation."""


class MetadataManifestSchemaValidationError(ValueError):
    """Raised when generated metadata manifest fails schema validation."""


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


def _build_validator(schema: dict[str, Any], *, error_cls: type[Exception], label: str) -> Any:
    return build_draft202012_validator(schema, error_cls=error_cls, label=label)


def _validate_metadata_records(records: list[dict[str, Any]], metadata_schema: dict[str, Any]) -> None:
    validator = _build_validator(
        metadata_schema,
        error_cls=MetadataSchemaValidationError,
        label="metadata",
    )
    for index, record in enumerate(records):
        try:
            errors = sorted(validator.iter_errors(record), key=lambda error: list(error.path))
        except (TypeError, ValueError) as exc:
            raise MetadataSchemaValidationError(
                f"record[{index}] schema validation failed due to validator runtime error ({type(exc).__name__})"
            ) from exc
        if errors:
            first = errors[0]
            path = ".".join(str(part) for part in first.path) or "<root>"
            raise MetadataSchemaValidationError(f"record[{index}] schema validation failed at {path}: {first.message}")


def _validate_metadata_manifest(payload: dict[str, Any], manifest_schema: dict[str, Any]) -> None:
    validator = _build_validator(
        manifest_schema,
        error_cls=MetadataManifestSchemaValidationError,
        label="metadata_manifest",
    )
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
    if errors:
        first = errors[0]
        path = ".".join(str(part) for part in first.path) or "<root>"
        raise MetadataManifestSchemaValidationError(f"metadata manifest schema validation failed at {path}: {first.message}")


def _require_unique_relative_paths(records: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for index, record in enumerate(records):
        relative_path = record.get("relative_path")
        if not isinstance(relative_path, str):
            raise MetadataManifestInputError(f"record[{index}] missing relative_path")
        if relative_path in seen:
            raise MetadataManifestInputError(f"duplicate relative_path: {relative_path}")
        seen.add(relative_path)


def _require_sorted_relative_paths(records: list[dict[str, Any]]) -> None:
    relative_paths = [record["relative_path"] for record in records]
    if relative_paths != sorted(relative_paths):
        raise MetadataManifestInputError("input metadata array must be sorted by relative_path")


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
    _require_unique_relative_paths(records)
    if strict_input_order:
        _require_sorted_relative_paths(records)
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
