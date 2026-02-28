"""Pure Phase 4E helpers for provenance binding and Merkle payload assembly."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from tp.crypto.merkle import merkle_root_sha256

from .hash_capture_metadata import (
    METADATA_CONTRACT_VERSION,
    METADATA_MANIFEST_CONTRACT_VERSION,
    canonical_json_bytes,
    compute_metadata_sha256,
)
from .schema_validation import build_draft202012_validator

PROVENANCE_CONTRACT_VERSION = "tp.meta.provenance.v1"
PROVENANCE_MERKLE_CONTRACT_VERSION = "tp.meta.provenance_merkle.v1"

_SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")


class ProvenanceInputError(ValueError):
    """Raised when Phase 4E inputs violate deterministic invariants."""


class ProvenanceSchemaValidationError(ValueError):
    """Raised when schema validation fails for Phase 4E inputs/outputs."""


class ProvenanceMerkleSchemaValidationError(ValueError):
    """Raised when generated Phase 4E merkle payload fails schema validation."""


def _build_validator(schema: dict[str, Any], *, error_cls: type[Exception], label: str) -> Any:
    return build_draft202012_validator(schema, error_cls=error_cls, label=label)


def _validate_capture_records(records: list[dict[str, Any]], metadata_schema: dict[str, Any]) -> None:
    validator = _build_validator(
        metadata_schema,
        error_cls=ProvenanceSchemaValidationError,
        label="metadata",
    )
    for index, record in enumerate(records):
        try:
            errors = sorted(validator.iter_errors(record), key=lambda error: list(error.path))
        except (TypeError, ValueError) as exc:
            raise ProvenanceSchemaValidationError(
                f"capture record[{index}] schema validation failed due to validator runtime error ({type(exc).__name__})"
            ) from exc
        if errors:
            first = errors[0]
            path = ".".join(str(part) for part in first.path) or "<root>"
            raise ProvenanceSchemaValidationError(
                f"capture record[{index}] schema validation failed at {path}: {first.message}"
            )


def _validate_metadata_manifest(payload: dict[str, Any], manifest_schema: dict[str, Any]) -> None:
    validator = _build_validator(
        manifest_schema,
        error_cls=ProvenanceSchemaValidationError,
        label="metadata_manifest",
    )
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
    if errors:
        first = errors[0]
        path = ".".join(str(part) for part in first.path) or "<root>"
        raise ProvenanceSchemaValidationError(f"metadata manifest schema validation failed at {path}: {first.message}")


def _validate_provenance_manifest(payload: dict[str, Any], provenance_manifest_schema: dict[str, Any]) -> None:
    validator = _build_validator(
        provenance_manifest_schema,
        error_cls=ProvenanceSchemaValidationError,
        label="provenance_manifest",
    )
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
    if errors:
        first = errors[0]
        path = ".".join(str(part) for part in first.path) or "<root>"
        raise ProvenanceSchemaValidationError(f"provenance manifest schema validation failed at {path}: {first.message}")


def _validate_provenance_merkle(payload: dict[str, Any], provenance_merkle_schema: dict[str, Any]) -> None:
    validator = _build_validator(
        provenance_merkle_schema,
        error_cls=ProvenanceMerkleSchemaValidationError,
        label="provenance_merkle",
    )
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
    if errors:
        first = errors[0]
        path = ".".join(str(part) for part in first.path) or "<root>"
        raise ProvenanceMerkleSchemaValidationError(f"provenance merkle schema validation failed at {path}: {first.message}")


def _require_unique_relative_paths(records: list[dict[str, Any]], *, label: str) -> None:
    seen: set[str] = set()
    for index, record in enumerate(records):
        relative_path = record.get("relative_path")
        if not isinstance(relative_path, str):
            raise ProvenanceInputError(f"{label} record[{index}] missing relative_path")
        if relative_path in seen:
            raise ProvenanceInputError(f"{label} duplicate relative_path: {relative_path}")
        seen.add(relative_path)


def _require_sorted_relative_paths(records: list[dict[str, Any]], *, label: str) -> None:
    relative_paths = [record["relative_path"] for record in records]
    if relative_paths != sorted(relative_paths):
        raise ProvenanceInputError(f"{label} must be sorted by relative_path")


def _require_capture_contract_version(records: list[dict[str, Any]]) -> None:
    for index, record in enumerate(records):
        contract_version = record.get("metadata_contract_version")
        if contract_version != METADATA_CONTRACT_VERSION:
            relative_path = record.get("relative_path", "<unknown>")
            raise ProvenanceInputError(
                f"capture record[{index}] contract mismatch for {relative_path}: expected {METADATA_CONTRACT_VERSION}, got {contract_version!r}"
            )


def _require_metadata_manifest_contract_version(payload: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_contract_version = payload.get("metadata_manifest_contract_version")
    if manifest_contract_version != METADATA_MANIFEST_CONTRACT_VERSION:
        raise ProvenanceInputError(
            f"metadata manifest contract mismatch: expected {METADATA_MANIFEST_CONTRACT_VERSION}, got {manifest_contract_version!r}"
        )

    metadata_contract_version = payload.get("metadata_contract_version")
    if metadata_contract_version != METADATA_CONTRACT_VERSION:
        raise ProvenanceInputError(
            f"metadata manifest metadata_contract_version mismatch: expected {METADATA_CONTRACT_VERSION}, got {metadata_contract_version!r}"
        )

    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ProvenanceInputError("metadata manifest entries must be an array")
    return entries


def _require_provenance_manifest_contract_version(payload: dict[str, Any]) -> list[dict[str, Any]]:
    provenance_contract_version = payload.get("provenance_contract_version")
    if provenance_contract_version != PROVENANCE_CONTRACT_VERSION:
        raise ProvenanceInputError(
            f"provenance manifest contract mismatch: expected {PROVENANCE_CONTRACT_VERSION}, got {provenance_contract_version!r}"
        )

    metadata_contract_version = payload.get("metadata_contract_version")
    if metadata_contract_version != METADATA_CONTRACT_VERSION:
        raise ProvenanceInputError(
            f"provenance manifest metadata_contract_version mismatch: expected {METADATA_CONTRACT_VERSION}, got {metadata_contract_version!r}"
        )

    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ProvenanceInputError("provenance manifest entries must be an array")
    return entries


def _require_fingerprint_match(records: list[dict[str, Any]], expected_fingerprint: str) -> None:
    for index, record in enumerate(records):
        extractor = record.get("extractor")
        if not isinstance(extractor, dict):
            raise ProvenanceInputError(f"capture record[{index}] missing extractor object")
        fingerprint = extractor.get("config_fingerprint_sha256")
        if fingerprint != expected_fingerprint:
            relative_path = record.get("relative_path", "<unknown>")
            raise ProvenanceInputError(
                f"capture record[{index}] fingerprint mismatch for {relative_path}: expected {expected_fingerprint}, got {fingerprint!r}"
            )


def _path_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record["relative_path"]: record for record in records}


def _ensure_sha256_hex(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_HEX_RE.match(value):
        raise ProvenanceInputError(f"{label} must be a lowercase 64-character sha256 hex digest")
    return value


def _ensure_contract_version(value: str, *, expected: str, label: str) -> str:
    if value != expected:
        raise ProvenanceInputError(f"{label} mismatch: expected {expected}, got {value!r}")
    return value


def compute_provenance_entry_sha256(
    *,
    file_sha256: str,
    metadata_sha256: str,
    capture_contract_version: str = METADATA_CONTRACT_VERSION,
    metadata_contract_version: str = METADATA_CONTRACT_VERSION,
    provenance_contract_version: str = PROVENANCE_CONTRACT_VERSION,
) -> str:
    """Compute SHA256(F || M || C || Mv || Pv) over binary digests + UTF-8 version strings."""
    capture_version = _ensure_contract_version(
        capture_contract_version,
        expected=METADATA_CONTRACT_VERSION,
        label="capture_contract_version",
    )
    metadata_version = _ensure_contract_version(
        metadata_contract_version,
        expected=METADATA_CONTRACT_VERSION,
        label="metadata_contract_version",
    )
    provenance_version = _ensure_contract_version(
        provenance_contract_version,
        expected=PROVENANCE_CONTRACT_VERSION,
        label="provenance_contract_version",
    )

    file_digest = bytes.fromhex(_ensure_sha256_hex(file_sha256, label="file_sha256"))
    metadata_digest = bytes.fromhex(_ensure_sha256_hex(metadata_sha256, label="metadata_sha256"))
    capture_version_bytes = capture_version.encode("utf-8")
    metadata_version_bytes = metadata_version.encode("utf-8")
    provenance_version_bytes = provenance_version.encode("utf-8")
    return hashlib.sha256(
        file_digest + metadata_digest + capture_version_bytes + metadata_version_bytes + provenance_version_bytes
    ).hexdigest()


def build_provenance_manifest_payload(
    capture_records: list[dict[str, Any]],
    metadata_manifest_payload: dict[str, Any],
    *,
    metadata_schema: dict[str, Any],
    metadata_manifest_schema: dict[str, Any],
    provenance_manifest_schema: dict[str, Any],
    strict_input_order: bool = True,
    required_config_fingerprint_sha256: str | None = None,
) -> dict[str, Any]:
    """Build validated Phase 4E provenance manifest from Phase 4C/4D artifacts."""
    if not isinstance(capture_records, list):
        raise ProvenanceInputError("capture metadata payload must be a JSON array")
    if not isinstance(metadata_manifest_payload, dict):
        raise ProvenanceInputError("metadata manifest payload must be a JSON object")

    _validate_capture_records(capture_records, metadata_schema=metadata_schema)
    _validate_metadata_manifest(metadata_manifest_payload, manifest_schema=metadata_manifest_schema)

    _require_capture_contract_version(capture_records)
    metadata_manifest_entries = _require_metadata_manifest_contract_version(metadata_manifest_payload)

    _require_unique_relative_paths(capture_records, label="capture metadata")
    _require_unique_relative_paths(metadata_manifest_entries, label="metadata manifest")
    if strict_input_order:
        _require_sorted_relative_paths(capture_records, label="capture metadata array")
        _require_sorted_relative_paths(metadata_manifest_entries, label="metadata manifest entries")
    if required_config_fingerprint_sha256 is not None:
        _require_fingerprint_match(capture_records, expected_fingerprint=required_config_fingerprint_sha256)

    capture_by_path = _path_index(capture_records)
    manifest_by_path = _path_index(metadata_manifest_entries)

    capture_paths = set(capture_by_path)
    manifest_paths = set(manifest_by_path)
    missing_in_manifest = sorted(capture_paths - manifest_paths)
    missing_in_capture = sorted(manifest_paths - capture_paths)
    if missing_in_manifest or missing_in_capture:
        raise ProvenanceInputError(
            f"relative_path alignment mismatch between capture metadata and metadata manifest: missing_in_manifest={missing_in_manifest}, "
            f"missing_in_capture={missing_in_capture}"
        )

    entries: list[dict[str, str]] = []
    for relative_path in sorted(capture_by_path):
        capture_record = capture_by_path[relative_path]
        manifest_entry = manifest_by_path[relative_path]

        capture_file_sha256 = _ensure_sha256_hex(capture_record.get("file_sha256"), label=f"{relative_path} file_sha256")
        manifest_file_sha256 = _ensure_sha256_hex(
            manifest_entry.get("file_sha256"), label=f"{relative_path} metadata manifest file_sha256"
        )
        if capture_file_sha256 != manifest_file_sha256:
            raise ProvenanceInputError(
                f"file_sha256 mismatch for {relative_path}: capture={capture_file_sha256}, manifest={manifest_file_sha256}"
            )

        try:
            recomputed_metadata_sha256 = compute_metadata_sha256(capture_record)
        except (TypeError, ValueError) as exc:
            raise ProvenanceSchemaValidationError(
                f"canonical metadata serialization failed for record with relative_path={relative_path!r}: {exc}"
            ) from exc

        manifest_metadata_sha256 = _ensure_sha256_hex(
            manifest_entry.get("metadata_sha256"), label=f"{relative_path} metadata manifest metadata_sha256"
        )
        if recomputed_metadata_sha256 != manifest_metadata_sha256:
            raise ProvenanceInputError(
                f"metadata_sha256 mismatch for {relative_path}: recomputed={recomputed_metadata_sha256}, manifest={manifest_metadata_sha256}"
            )

        provenance_entry_sha256 = compute_provenance_entry_sha256(
            file_sha256=manifest_file_sha256,
            metadata_sha256=manifest_metadata_sha256,
            capture_contract_version=capture_record["metadata_contract_version"],
            metadata_contract_version=metadata_manifest_payload["metadata_contract_version"],
            provenance_contract_version=PROVENANCE_CONTRACT_VERSION,
        )
        entries.append(
            {
                "relative_path": relative_path,
                "file_sha256": manifest_file_sha256,
                "metadata_sha256": manifest_metadata_sha256,
                "provenance_entry_sha256": provenance_entry_sha256,
            }
        )

    payload = {
        "provenance_contract_version": PROVENANCE_CONTRACT_VERSION,
        "metadata_contract_version": METADATA_CONTRACT_VERSION,
        "entries": entries,
    }
    _validate_provenance_manifest(payload, provenance_manifest_schema=provenance_manifest_schema)
    return payload


def serialize_provenance_manifest(payload: dict[str, Any]) -> bytes:
    """Serialize provenance manifest payload with canonical bytes and trailing newline."""
    return canonical_json_bytes(payload, trailing_newline=True)


def build_provenance_merkle_payload(
    provenance_manifest_payload: dict[str, Any],
    *,
    provenance_manifest_schema: dict[str, Any],
    provenance_merkle_schema: dict[str, Any],
    strict_input_order: bool = True,
) -> dict[str, Any]:
    """Build validated Phase 4E provenance merkle payload from provenance manifest entries."""
    if not isinstance(provenance_manifest_payload, dict):
        raise ProvenanceInputError("provenance manifest payload must be a JSON object")

    _validate_provenance_manifest(provenance_manifest_payload, provenance_manifest_schema=provenance_manifest_schema)
    manifest_entries = _require_provenance_manifest_contract_version(provenance_manifest_payload)

    _require_unique_relative_paths(manifest_entries, label="provenance manifest")
    if strict_input_order:
        _require_sorted_relative_paths(manifest_entries, label="provenance manifest entries")

    sorted_entries = sorted(manifest_entries, key=lambda entry: entry["relative_path"])
    leaf_hashes: list[bytes] = []
    for index, entry in enumerate(sorted_entries):
        digest_hex = _ensure_sha256_hex(entry.get("provenance_entry_sha256"), label=f"entry[{index}] provenance_entry_sha256")
        leaf_hashes.append(bytes.fromhex(digest_hex))

    if not leaf_hashes:
        raise ProvenanceInputError("provenance manifest entries must be non-empty")

    payload = {
        "provenance_merkle_contract_version": PROVENANCE_MERKLE_CONTRACT_VERSION,
        "provenance_contract_version": PROVENANCE_CONTRACT_VERSION,
        "leaf_count": len(leaf_hashes),
        "provenance_merkle_root": merkle_root_sha256(leaf_hashes),
    }
    _validate_provenance_merkle(payload, provenance_merkle_schema=provenance_merkle_schema)
    return payload


def serialize_provenance_merkle(payload: dict[str, Any]) -> bytes:
    """Serialize provenance merkle payload with canonical bytes and trailing newline."""
    return canonical_json_bytes(payload, trailing_newline=True)
