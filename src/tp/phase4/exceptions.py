"""Unified exception hierarchy for Phase 4 deterministic provenance tooling.

This module provides a structured exception hierarchy for Phase 4 operations:

- Phase4Error: Base exception for all Phase 4 operations
  - Phase4InputError: Input validation failures
  - Phase4SchemaError: JSON Schema validation failures
  - Phase4IntegrityError: Hash/Merkle verification failures
    - Phase4MetadataHashError: Metadata hash mismatches
    - Phase4ProvenanceHashError: Provenance entry hash mismatches
    - Phase4MerkleError: Merkle root/leaf count mismatches

Legacy exceptions are preserved for backward compatibility but inherit
from the new hierarchy.
"""

from __future__ import annotations


class Phase4Error(Exception):
    """Base exception for all Phase 4 operations.

    This is the root exception class for the Phase 4 deterministic provenance
    system. All Phase 4 exceptions inherit from this class, enabling broad
    exception handling when desired.

    Example:
        try:
            verify_phase4_chain_from_paths(...)
        except Phase4Error as e:
            # Handle any Phase 4 failure
            log_error(f"Phase 4 verification failed: {e}")
    """


class Phase4InputError(Phase4Error, ValueError):
    """Input validation failures for Phase 4 operations.

    Raised when Phase 4 inputs violate deterministic invariants such as:
    - Missing required fields
    - Invalid contract versions
    - Path alignment mismatches
    - Duplicate relative paths
    - Unsorted input arrays
    - Invalid fingerprint matches

    Also inherits from ValueError for compatibility with existing code
    that catches ValueError.
    """


class Phase4SchemaError(Phase4Error, ValueError):
    """JSON Schema validation failures.

    Raised when:
    - Artifacts fail schema validation
    - Schema files cannot be loaded
    - Schema structure is invalid
    - Report payloads fail validation

    Also inherits from ValueError for compatibility.
    """


class Phase4IntegrityError(Phase4Error, ValueError):
    """Hash or Merkle verification failures.

    Base class for cryptographic integrity failures:
    - Metadata hash mismatches
    - Provenance entry hash mismatches
    - Merkle root mismatches
    - Leaf count mismatches

    Also inherits from ValueError for compatibility.
    """


class Phase4MetadataHashError(Phase4IntegrityError):
    """Metadata SHA256 hash mismatch.

    Raised when the recomputed metadata_sha256 from a capture record
    does not match the value recorded in the metadata manifest.
    """


class Phase4ProvenanceHashError(Phase4IntegrityError):
    """Provenance entry SHA256 hash mismatch.

    Raised when the recomputed provenance_entry_sha256 does not match
    the value recorded in the provenance manifest.
    """


class Phase4MerkleError(Phase4IntegrityError):
    """Merkle tree verification failure.

    Raised when:
    - Recomputed Merkle root does not match expected root
    - Leaf count does not match expected count
    """


class Phase4ConfigError(Phase4Error, ValueError):
    """Configuration validation failures.

    Raised when capture metadata configuration is invalid:
    - Missing required keys
    - Invalid contract versions
    - Malformed rounding rules
    - Invalid path normalization policy

    Also inherits from ValueError for compatibility.
    """


class Phase4ExtractionError(Phase4Error, RuntimeError):
    """Metadata extraction failures.

    Raised when metadata extraction from image files fails:
    - ExifTool execution failures
    - Timeouts during extraction
    - Parse failures

    Also inherits from RuntimeError for compatibility.
    """


# ============================================================================
# Legacy Exception Aliases
#
# These maintain backward compatibility with existing code that imports
# exceptions from individual modules.
# ============================================================================

# From canonicalize_capture_metadata.py
ConfigValidationError = Phase4ConfigError
PathNormalizationError = Phase4InputError
ExtractionFailure = Phase4ExtractionError
SchemaValidationError = Phase4SchemaError
StrictWarningsError = Phase4ExtractionError

# From hash_capture_metadata.py
MetadataManifestInputError = Phase4InputError
MetadataSchemaValidationError = Phase4SchemaError
MetadataManifestSchemaValidationError = Phase4SchemaError

# From provenance_capture.py
ProvenanceInputError = Phase4InputError
ProvenanceSchemaValidationError = Phase4SchemaError
ProvenanceMerkleSchemaValidationError = Phase4SchemaError

# From verify_phase4_chain.py
Phase4VerificationInputError = Phase4InputError
Phase4AlignmentError = Phase4InputError
Phase4MetadataHashMismatchError = Phase4MetadataHashError
Phase4ProvenanceEntryHashMismatchError = Phase4ProvenanceHashError
Phase4MerkleMismatchError = Phase4MerkleError

# Note: Phase4SchemaValidationError is kept as-is in verify_phase4_chain.py
# but maps to Phase4SchemaError in the new hierarchy
