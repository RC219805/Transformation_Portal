"""Phase 4 deterministic capture metadata and provenance helpers.

This package provides deterministic provenance capture and verification
for the Transformation Portal. It implements a multi-stage chain:

- Phase 4C: Capture metadata extraction from image files (EXIF, GPS, camera data)
- Phase 4D: Metadata hashing and manifest assembly
- Phase 4E: Provenance binding and Merkle tree construction
- Phase 4F: Chain verification and report generation

Public API modules:
- types: TypedDict definitions for payload structures
- exceptions: Unified exception hierarchy
- validation_helpers: Shared validation utilities
"""

# Legacy imports for backward compatibility
# Phase 4C: Capture metadata extraction
from .canonicalize_capture_metadata import (
    ConfigValidationError,
    ExtractionFailure,
    PathNormalizationError,
    SchemaValidationError,
    StrictWarningsError,
    compute_config_fingerprint_sha256,
    extract_capture_metadata_records,
    load_capture_metadata_config,
    normalize_relative_path,
    write_capture_metadata_artifact,
)

# New unified exception hierarchy (for direct use)
from .exceptions import (
    Phase4ConfigError,
    Phase4Error,
    Phase4ExtractionError,
    Phase4InputError,
    Phase4IntegrityError,
    Phase4MerkleError,
    Phase4MetadataHashError,
    Phase4ProvenanceHashError,
    Phase4SchemaError,
)

# Phase 4D: Metadata hashing and manifest
from .hash_capture_metadata import (
    METADATA_CONTRACT_VERSION,
    METADATA_MANIFEST_CONTRACT_VERSION,
    MetadataManifestInputError,
    MetadataManifestSchemaValidationError,
    MetadataSchemaValidationError,
    build_metadata_manifest_payload,
    canonical_json_bytes,
    compute_metadata_sha256,
    serialize_metadata_manifest,
)

# Phase 4E: Provenance and Merkle
from .provenance_capture import (
    PROVENANCE_CONTRACT_VERSION,
    PROVENANCE_MERKLE_CONTRACT_VERSION,
    ProvenanceInputError,
    ProvenanceMerkleSchemaValidationError,
    ProvenanceSchemaValidationError,
    build_provenance_manifest_payload,
    build_provenance_merkle_payload,
    compute_provenance_entry_sha256,
    serialize_provenance_manifest,
    serialize_provenance_merkle,
)

# Shared validation helpers
from .validation_helpers import (
    SHA256_HEX_RE,
    build_path_index,
    ensure_sha256_hex,
    is_valid_sha256_hex,
    require_contract_version,
    require_sorted_relative_paths,
    require_unique_relative_paths,
    string_or_none,
    validate_payload_with_schema,
    validate_records_with_schema,
)

# Phase 4F: Verification and reporting
from .verify_phase4_chain import (
    FAILURE_LABEL_ALIGNMENT_FAILURE,
    FAILURE_LABEL_MALFORMED_INPUT,
    FAILURE_LABEL_MERKLE_MISMATCH,
    FAILURE_LABEL_METADATA_HASH_MISMATCH,
    FAILURE_LABEL_PROVENANCE_ENTRY_HASH_MISMATCH,
    FAILURE_LABEL_REPORT_WRITE_FAILURE,
    FAILURE_LABEL_SCHEMA_VALIDATION_FAILURE,
    VERIFICATION_REPORT_CONTRACT_VERSION,
    VERIFIER_BUILD_ID,
    VERIFIER_NAME,
    VERIFIER_VERSION,
    Phase4AlignmentError,
    Phase4MerkleMismatchError,
    Phase4MetadataHashMismatchError,
    Phase4ProvenanceEntryHashMismatchError,
    Phase4SchemaValidationError,
    Phase4VerificationInputError,
    build_verification_report_payload,
    collect_report_inputs_from_paths,
    default_failure_computed_block,
    serialize_verification_report_payload,
    validate_verification_report_payload,
    verify_phase4_chain_from_paths,
    verify_phase4_chain_payloads,
)

__all__ = [
    # New unified exception hierarchy
    "Phase4Error",
    "Phase4InputError",
    "Phase4SchemaError",
    "Phase4IntegrityError",
    "Phase4MetadataHashError",
    "Phase4ProvenanceHashError",
    "Phase4MerkleError",
    "Phase4ConfigError",
    "Phase4ExtractionError",
    # Validation helpers
    "SHA256_HEX_RE",
    "is_valid_sha256_hex",
    "ensure_sha256_hex",
    "require_unique_relative_paths",
    "require_sorted_relative_paths",
    "build_path_index",
    "require_contract_version",
    "validate_records_with_schema",
    "validate_payload_with_schema",
    "string_or_none",
    # Legacy exceptions (backward compatibility)
    "ConfigValidationError",
    "ExtractionFailure",
    "PathNormalizationError",
    "SchemaValidationError",
    "StrictWarningsError",
    # Phase 4C: Capture metadata
    "compute_config_fingerprint_sha256",
    "compute_metadata_sha256",
    "extract_capture_metadata_records",
    "load_capture_metadata_config",
    "normalize_relative_path",
    "write_capture_metadata_artifact",
    # Phase 4D: Metadata manifest
    "METADATA_CONTRACT_VERSION",
    "METADATA_MANIFEST_CONTRACT_VERSION",
    "MetadataManifestInputError",
    "MetadataSchemaValidationError",
    "MetadataManifestSchemaValidationError",
    "build_metadata_manifest_payload",
    "canonical_json_bytes",
    "serialize_metadata_manifest",
    # Phase 4E: Provenance and Merkle
    "PROVENANCE_CONTRACT_VERSION",
    "PROVENANCE_MERKLE_CONTRACT_VERSION",
    "ProvenanceInputError",
    "ProvenanceSchemaValidationError",
    "ProvenanceMerkleSchemaValidationError",
    "build_provenance_manifest_payload",
    "build_provenance_merkle_payload",
    "compute_provenance_entry_sha256",
    "serialize_provenance_manifest",
    "serialize_provenance_merkle",
    # Phase 4F: Verification
    "FAILURE_LABEL_ALIGNMENT_FAILURE",
    "FAILURE_LABEL_MALFORMED_INPUT",
    "FAILURE_LABEL_MERKLE_MISMATCH",
    "FAILURE_LABEL_METADATA_HASH_MISMATCH",
    "FAILURE_LABEL_PROVENANCE_ENTRY_HASH_MISMATCH",
    "FAILURE_LABEL_REPORT_WRITE_FAILURE",
    "FAILURE_LABEL_SCHEMA_VALIDATION_FAILURE",
    "VERIFICATION_REPORT_CONTRACT_VERSION",
    "VERIFIER_BUILD_ID",
    "VERIFIER_NAME",
    "VERIFIER_VERSION",
    "Phase4AlignmentError",
    "Phase4MerkleMismatchError",
    "Phase4MetadataHashMismatchError",
    "Phase4ProvenanceEntryHashMismatchError",
    "Phase4SchemaValidationError",
    "Phase4VerificationInputError",
    "build_verification_report_payload",
    "collect_report_inputs_from_paths",
    "default_failure_computed_block",
    "serialize_verification_report_payload",
    "validate_verification_report_payload",
    "verify_phase4_chain_from_paths",
    "verify_phase4_chain_payloads",
]
