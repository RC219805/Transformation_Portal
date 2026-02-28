"""Phase 4F verifier helpers for Phase 4 capture provenance artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from tp.crypto.merkle import merkle_root_sha256

from .hash_capture_metadata import METADATA_CONTRACT_VERSION, METADATA_MANIFEST_CONTRACT_VERSION, compute_metadata_sha256
from .provenance_capture import (
    PROVENANCE_CONTRACT_VERSION,
    PROVENANCE_MERKLE_CONTRACT_VERSION,
    compute_provenance_entry_sha256,
)
from .schema_validation import build_draft202012_validator

FAILURE_LABEL_MALFORMED_INPUT = "MALFORMED_INPUT"
FAILURE_LABEL_SCHEMA_VALIDATION_FAILURE = "SCHEMA_VALIDATION_FAILURE"
FAILURE_LABEL_ALIGNMENT_FAILURE = "ALIGNMENT_FAILURE"
FAILURE_LABEL_METADATA_HASH_MISMATCH = "METADATA_HASH_MISMATCH"
FAILURE_LABEL_PROVENANCE_ENTRY_HASH_MISMATCH = "PROVENANCE_ENTRY_HASH_MISMATCH"
FAILURE_LABEL_MERKLE_MISMATCH = "MERKLE_MISMATCH"

_SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")


class Phase4VerificationInputError(ValueError):
    """Raised when verifier inputs cannot be loaded or parsed."""


class Phase4SchemaValidationError(ValueError):
    """Raised when schema validation fails for artifacts."""


class Phase4AlignmentError(ValueError):
    """Raised when deterministic alignment/order/version invariants fail."""


class Phase4MetadataHashMismatchError(ValueError):
    """Raised when recomputed metadata digest mismatches Phase 4D manifest."""


class Phase4ProvenanceEntryHashMismatchError(ValueError):
    """Raised when recomputed provenance-entry digest mismatches Phase 4E manifest."""


class Phase4MerkleMismatchError(ValueError):
    """Raised when recomputed provenance Merkle metadata mismatches Phase 4E artifact."""


def _build_validator(schema: dict[str, Any], *, error_cls: type[Exception], label: str) -> Any:
    return build_draft202012_validator(schema, error_cls=error_cls, label=label)


def _ensure_sha256_hex(value: Any, *, label: str, error_cls: type[Exception]) -> str:
    if not isinstance(value, str) or _SHA256_HEX_RE.fullmatch(value) is None:
        raise error_cls(f"{label} must be a lowercase 64-character sha256 hex digest")
    return value


def _read_json_file(path: Path, *, label: str) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise Phase4VerificationInputError(f"unable to load {label} {path}: {exc}") from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase4VerificationInputError(f"unable to parse {label} {path}: {exc}") from exc
    return payload


def _read_json_schema(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase4SchemaValidationError(f"unable to load {label} schema {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Phase4SchemaValidationError(f"{label} schema must be a JSON object")
    return payload


def _validate_capture_records(records: list[dict[str, Any]], metadata_schema: dict[str, Any]) -> None:
    validator = _build_validator(
        metadata_schema,
        error_cls=Phase4SchemaValidationError,
        label="metadata",
    )
    for index, record in enumerate(records):
        try:
            errors = sorted(validator.iter_errors(record), key=lambda error: list(error.path))
        except (TypeError, ValueError) as exc:
            raise Phase4SchemaValidationError(
                f"capture record[{index}] schema validation failed due to validator runtime error ({type(exc).__name__})"
            ) from exc
        if errors:
            first = errors[0]
            path = ".".join(str(part) for part in first.path) or "<root>"
            raise Phase4SchemaValidationError(f"capture record[{index}] schema validation failed at {path}: {first.message}")


def _validate_payload(payload: dict[str, Any], schema: dict[str, Any], *, label: str) -> None:
    validator = _build_validator(
        schema,
        error_cls=Phase4SchemaValidationError,
        label=label,
    )
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
    if errors:
        first = errors[0]
        path = ".".join(str(part) for part in first.path) or "<root>"
        raise Phase4SchemaValidationError(f"{label} schema validation failed at {path}: {first.message}")


def _require_unique_relative_paths(records: list[dict[str, Any]], *, label: str) -> None:
    seen: set[str] = set()
    for index, record in enumerate(records):
        relative_path = record.get("relative_path")
        if not isinstance(relative_path, str):
            raise Phase4AlignmentError(f"{label} record[{index}] missing relative_path")
        if relative_path in seen:
            raise Phase4AlignmentError(f"{label} duplicate relative_path: {relative_path}")
        seen.add(relative_path)


def _require_sorted_relative_paths(records: list[dict[str, Any]], *, label: str) -> None:
    relative_paths = [record["relative_path"] for record in records]
    if relative_paths != sorted(relative_paths):
        raise Phase4AlignmentError(f"{label} must be sorted by relative_path")


def _path_index(records: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for position, record in enumerate(records):
        relative_path = record.get("relative_path")
        if not isinstance(relative_path, str):
            raise Phase4AlignmentError(f"{label} record[{position}] missing relative_path")
        index[relative_path] = record
    return index


def _require_capture_contract_version(records: list[dict[str, Any]]) -> None:
    for index, record in enumerate(records):
        contract_version = record.get("metadata_contract_version")
        if contract_version != METADATA_CONTRACT_VERSION:
            relative_path = record.get("relative_path", "<unknown>")
            raise Phase4AlignmentError(
                f"capture record[{index}] contract mismatch for {relative_path}: "
                f"expected {METADATA_CONTRACT_VERSION}, got {contract_version!r}"
            )


def _require_metadata_manifest_contract_version(payload: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_contract_version = payload.get("metadata_manifest_contract_version")
    if manifest_contract_version != METADATA_MANIFEST_CONTRACT_VERSION:
        raise Phase4AlignmentError(
            "metadata manifest contract mismatch: "
            f"expected {METADATA_MANIFEST_CONTRACT_VERSION}, got {manifest_contract_version!r}"
        )

    metadata_contract_version = payload.get("metadata_contract_version")
    if metadata_contract_version != METADATA_CONTRACT_VERSION:
        raise Phase4AlignmentError(
            "metadata manifest metadata_contract_version mismatch: "
            f"expected {METADATA_CONTRACT_VERSION}, got {metadata_contract_version!r}"
        )

    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise Phase4AlignmentError("metadata manifest entries must be an array")
    return entries


def _require_provenance_manifest_contract_version(payload: dict[str, Any]) -> list[dict[str, Any]]:
    provenance_contract_version = payload.get("provenance_contract_version")
    if provenance_contract_version != PROVENANCE_CONTRACT_VERSION:
        raise Phase4AlignmentError(
            "provenance manifest contract mismatch: "
            f"expected {PROVENANCE_CONTRACT_VERSION}, got {provenance_contract_version!r}"
        )

    metadata_contract_version = payload.get("metadata_contract_version")
    if metadata_contract_version != METADATA_CONTRACT_VERSION:
        raise Phase4AlignmentError(
            "provenance manifest metadata_contract_version mismatch: "
            f"expected {METADATA_CONTRACT_VERSION}, got {metadata_contract_version!r}"
        )

    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise Phase4AlignmentError("provenance manifest entries must be an array")
    return entries


def _require_provenance_merkle_contract_version(payload: dict[str, Any]) -> None:
    merkle_contract_version = payload.get("provenance_merkle_contract_version")
    if merkle_contract_version != PROVENANCE_MERKLE_CONTRACT_VERSION:
        raise Phase4AlignmentError(
            "provenance merkle contract mismatch: "
            f"expected {PROVENANCE_MERKLE_CONTRACT_VERSION}, got {merkle_contract_version!r}"
        )

    provenance_contract_version = payload.get("provenance_contract_version")
    if provenance_contract_version != PROVENANCE_CONTRACT_VERSION:
        raise Phase4AlignmentError(
            "provenance merkle provenance_contract_version mismatch: "
            f"expected {PROVENANCE_CONTRACT_VERSION}, got {provenance_contract_version!r}"
        )


def verify_phase4_chain_payloads(
    capture_payload: list[dict[str, Any]],
    metadata_manifest_payload: dict[str, Any],
    provenance_manifest_payload: dict[str, Any],
    provenance_merkle_payload: dict[str, Any],
    *,
    metadata_schema: dict[str, Any],
    metadata_manifest_schema: dict[str, Any],
    provenance_manifest_schema: dict[str, Any],
    provenance_merkle_schema: dict[str, Any],
    strict_input_order: bool = True,
) -> dict[str, Any]:
    """Recompute and validate the full Phase 4C/4D/4E provenance chain."""
    if not isinstance(capture_payload, list):
        raise Phase4SchemaValidationError("capture metadata payload must be a JSON array")
    if not isinstance(metadata_manifest_payload, dict):
        raise Phase4SchemaValidationError("metadata manifest payload must be a JSON object")
    if not isinstance(provenance_manifest_payload, dict):
        raise Phase4SchemaValidationError("provenance manifest payload must be a JSON object")
    if not isinstance(provenance_merkle_payload, dict):
        raise Phase4SchemaValidationError("provenance merkle payload must be a JSON object")

    _validate_capture_records(capture_payload, metadata_schema=metadata_schema)
    _validate_payload(metadata_manifest_payload, metadata_manifest_schema, label="metadata_manifest")
    _validate_payload(provenance_manifest_payload, provenance_manifest_schema, label="provenance_manifest")
    _validate_payload(provenance_merkle_payload, provenance_merkle_schema, label="provenance_merkle")

    _require_capture_contract_version(capture_payload)
    metadata_manifest_entries = _require_metadata_manifest_contract_version(metadata_manifest_payload)
    provenance_manifest_entries = _require_provenance_manifest_contract_version(provenance_manifest_payload)
    _require_provenance_merkle_contract_version(provenance_merkle_payload)

    _require_unique_relative_paths(capture_payload, label="capture metadata")
    _require_unique_relative_paths(metadata_manifest_entries, label="metadata manifest")
    _require_unique_relative_paths(provenance_manifest_entries, label="provenance manifest")
    if strict_input_order:
        _require_sorted_relative_paths(capture_payload, label="capture metadata array")
        _require_sorted_relative_paths(metadata_manifest_entries, label="metadata manifest entries")
        _require_sorted_relative_paths(provenance_manifest_entries, label="provenance manifest entries")

    capture_by_path = _path_index(capture_payload, label="capture metadata")
    metadata_manifest_by_path = _path_index(metadata_manifest_entries, label="metadata manifest")
    provenance_manifest_by_path = _path_index(provenance_manifest_entries, label="provenance manifest")

    capture_paths = set(capture_by_path)
    metadata_manifest_paths = set(metadata_manifest_by_path)
    provenance_manifest_paths = set(provenance_manifest_by_path)

    missing_in_metadata_manifest = sorted(capture_paths - metadata_manifest_paths)
    missing_in_capture_from_metadata_manifest = sorted(metadata_manifest_paths - capture_paths)
    if missing_in_metadata_manifest or missing_in_capture_from_metadata_manifest:
        raise Phase4AlignmentError(
            "relative_path alignment mismatch between capture metadata and metadata manifest: "
            f"missing_in_metadata_manifest={missing_in_metadata_manifest}, "
            f"missing_in_capture={missing_in_capture_from_metadata_manifest}"
        )

    missing_in_provenance_manifest = sorted(capture_paths - provenance_manifest_paths)
    missing_in_capture_from_provenance_manifest = sorted(provenance_manifest_paths - capture_paths)
    if missing_in_provenance_manifest or missing_in_capture_from_provenance_manifest:
        raise Phase4AlignmentError(
            "relative_path alignment mismatch between capture metadata and provenance manifest: "
            f"missing_in_provenance_manifest={missing_in_provenance_manifest}, "
            f"missing_in_capture={missing_in_capture_from_provenance_manifest}"
        )

    recomputed_metadata_hashes: list[str] = []
    recomputed_provenance_entry_hashes: list[str] = []

    for relative_path in sorted(capture_by_path):
        capture_record = capture_by_path[relative_path]
        metadata_manifest_entry = metadata_manifest_by_path[relative_path]
        provenance_manifest_entry = provenance_manifest_by_path[relative_path]

        capture_file_sha256 = _ensure_sha256_hex(
            capture_record.get("file_sha256"),
            label=f"{relative_path} capture file_sha256",
            error_cls=Phase4AlignmentError,
        )
        metadata_manifest_file_sha256 = _ensure_sha256_hex(
            metadata_manifest_entry.get("file_sha256"),
            label=f"{relative_path} metadata manifest file_sha256",
            error_cls=Phase4AlignmentError,
        )
        provenance_manifest_file_sha256 = _ensure_sha256_hex(
            provenance_manifest_entry.get("file_sha256"),
            label=f"{relative_path} provenance manifest file_sha256",
            error_cls=Phase4AlignmentError,
        )

        if capture_file_sha256 != metadata_manifest_file_sha256:
            raise Phase4AlignmentError(
                f"file_sha256 mismatch between capture metadata and metadata manifest for {relative_path}: "
                f"capture={capture_file_sha256}, metadata_manifest={metadata_manifest_file_sha256}"
            )
        if capture_file_sha256 != provenance_manifest_file_sha256:
            raise Phase4AlignmentError(
                f"file_sha256 mismatch between capture metadata and provenance manifest for {relative_path}: "
                f"capture={capture_file_sha256}, provenance_manifest={provenance_manifest_file_sha256}"
            )

        metadata_manifest_metadata_sha256 = _ensure_sha256_hex(
            metadata_manifest_entry.get("metadata_sha256"),
            label=f"{relative_path} metadata manifest metadata_sha256",
            error_cls=Phase4AlignmentError,
        )
        provenance_manifest_metadata_sha256 = _ensure_sha256_hex(
            provenance_manifest_entry.get("metadata_sha256"),
            label=f"{relative_path} provenance manifest metadata_sha256",
            error_cls=Phase4AlignmentError,
        )

        if metadata_manifest_metadata_sha256 != provenance_manifest_metadata_sha256:
            raise Phase4AlignmentError(
                f"metadata_sha256 mismatch between metadata manifest and provenance manifest for {relative_path}: "
                f"metadata_manifest={metadata_manifest_metadata_sha256}, provenance_manifest={provenance_manifest_metadata_sha256}"
            )

        try:
            recomputed_metadata_sha256 = compute_metadata_sha256(capture_record)
        except (TypeError, ValueError) as exc:
            raise Phase4SchemaValidationError(
                f"canonical metadata serialization failed for record with relative_path={relative_path!r}: {exc}"
            ) from exc
        if recomputed_metadata_sha256 != metadata_manifest_metadata_sha256:
            raise Phase4MetadataHashMismatchError(
                f"metadata_sha256 mismatch for {relative_path}: "
                f"recomputed={recomputed_metadata_sha256}, expected={metadata_manifest_metadata_sha256}"
            )

        provenance_manifest_entry_sha256 = _ensure_sha256_hex(
            provenance_manifest_entry.get("provenance_entry_sha256"),
            label=f"{relative_path} provenance manifest provenance_entry_sha256",
            error_cls=Phase4ProvenanceEntryHashMismatchError,
        )
        recomputed_provenance_entry_sha256 = compute_provenance_entry_sha256(
            file_sha256=metadata_manifest_file_sha256,
            metadata_sha256=metadata_manifest_metadata_sha256,
            capture_contract_version=capture_record["metadata_contract_version"],
            metadata_contract_version=metadata_manifest_payload["metadata_contract_version"],
            provenance_contract_version=provenance_manifest_payload["provenance_contract_version"],
        )
        if recomputed_provenance_entry_sha256 != provenance_manifest_entry_sha256:
            raise Phase4ProvenanceEntryHashMismatchError(
                f"provenance_entry_sha256 mismatch for {relative_path}: "
                f"recomputed={recomputed_provenance_entry_sha256}, expected={provenance_manifest_entry_sha256}"
            )

        recomputed_metadata_hashes.append(recomputed_metadata_sha256)
        recomputed_provenance_entry_hashes.append(recomputed_provenance_entry_sha256)

    leaf_hashes = [bytes.fromhex(digest_hex) for digest_hex in recomputed_provenance_entry_hashes]
    recomputed_provenance_merkle_root = merkle_root_sha256(leaf_hashes)
    expected_leaf_count = provenance_merkle_payload.get("leaf_count")
    if expected_leaf_count != len(leaf_hashes):
        raise Phase4MerkleMismatchError(
            f"leaf_count mismatch: recomputed={len(leaf_hashes)}, expected={expected_leaf_count!r}"
        )

    expected_provenance_merkle_root = _ensure_sha256_hex(
        provenance_merkle_payload.get("provenance_merkle_root"),
        label="provenance merkle provenance_merkle_root",
        error_cls=Phase4MerkleMismatchError,
    )
    if recomputed_provenance_merkle_root != expected_provenance_merkle_root:
        raise Phase4MerkleMismatchError(
            "provenance_merkle_root mismatch: "
            f"recomputed={recomputed_provenance_merkle_root}, expected={expected_provenance_merkle_root}"
        )

    metadata_sha256_summary_sha256 = hashlib.sha256("".join(recomputed_metadata_hashes).encode("ascii")).hexdigest()
    provenance_entry_sha256_summary_sha256 = hashlib.sha256(
        "".join(recomputed_provenance_entry_hashes).encode("ascii")
    ).hexdigest()
    return {
        "computed": {
            "metadata_sha256_summary_sha256": metadata_sha256_summary_sha256,
            "metadata_entry_count": len(recomputed_metadata_hashes),
            "provenance_entry_sha256_summary_sha256": provenance_entry_sha256_summary_sha256,
            "provenance_entry_count": len(recomputed_provenance_entry_hashes),
            "provenance_merkle_root": recomputed_provenance_merkle_root,
            "provenance_leaf_count": len(leaf_hashes),
        },
    }


def verify_phase4_chain_from_paths(
    *,
    capture_metadata_path: Path,
    metadata_manifest_path: Path,
    provenance_manifest_path: Path,
    provenance_merkle_path: Path,
    metadata_schema_path: Path,
    metadata_manifest_schema_path: Path,
    provenance_manifest_schema_path: Path,
    provenance_merkle_schema_path: Path,
    strict_input_order: bool = True,
) -> dict[str, Any]:
    """Load JSON artifacts from disk and verify the full Phase 4 chain."""
    capture_payload = _read_json_file(capture_metadata_path, label="capture metadata artifact")
    metadata_manifest_payload = _read_json_file(metadata_manifest_path, label="metadata manifest artifact")
    provenance_manifest_payload = _read_json_file(provenance_manifest_path, label="provenance manifest artifact")
    provenance_merkle_payload = _read_json_file(provenance_merkle_path, label="provenance merkle artifact")

    metadata_schema = _read_json_schema(metadata_schema_path, label="metadata")
    metadata_manifest_schema = _read_json_schema(metadata_manifest_schema_path, label="metadata manifest")
    provenance_manifest_schema = _read_json_schema(provenance_manifest_schema_path, label="provenance manifest")
    provenance_merkle_schema = _read_json_schema(provenance_merkle_schema_path, label="provenance merkle")

    return verify_phase4_chain_payloads(
        capture_payload,
        metadata_manifest_payload,
        provenance_manifest_payload,
        provenance_merkle_payload,
        metadata_schema=metadata_schema,
        metadata_manifest_schema=metadata_manifest_schema,
        provenance_manifest_schema=provenance_manifest_schema,
        provenance_merkle_schema=provenance_merkle_schema,
        strict_input_order=strict_input_order,
    )
