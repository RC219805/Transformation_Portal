"""Type definitions for Phase 4 deterministic provenance tooling.

This module provides TypedDict definitions for the various payload structures
used across Phase 4C/4D/4E/4F stages. These types enable better static analysis
and IDE support when working with Phase 4 artifacts.

Contract versions:
- tp.meta.capture.v1 - Capture metadata records (Phase 4C)
- tp.meta.capture_manifest.v1 - Metadata manifest (Phase 4D)
- tp.meta.provenance.v1 - Provenance manifest (Phase 4E)
- tp.meta.provenance_merkle.v1 - Provenance Merkle root (Phase 4E)
- tp.meta.verification_report.v1 - Verification report (Phase 4F)
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Protocol, TypedDict


# ============================================================================
# Phase 4C: Capture Metadata Types
# ============================================================================


class ExtractorInfo(TypedDict):
    """Extractor metadata embedded in capture records."""

    name: str
    version: str
    config_fingerprint_sha256: str


class CaptureMetadataRecord(TypedDict, total=False):
    """A single capture metadata record from Phase 4C extraction.

    Note: This TypedDict uses total=False for type checking flexibility.
    At runtime, the actual JSON schema enforces which fields are required.
    Refer to schemas/phase4/metadata.schema.json for the authoritative contract.

    The fields marked as 'Required' below are enforced by the schema validator,
    while fields marked 'Optional' may be absent or null.
    """

    # Required fields (enforced by schema validator)
    metadata_contract_version: str
    relative_path: str
    file_sha256: str
    extractor: ExtractorInfo
    extraction_warnings: List[str]

    # Optional capture fields (may be absent or None)
    capture_datetime_utc: Optional[str]
    camera_make: Optional[str]
    camera_model: Optional[str]
    lens_model: Optional[str]
    gps_latitude: Optional[float]
    gps_longitude: Optional[float]
    focal_length_mm: Optional[float]
    aperture_fnumber: Optional[float]
    shutter_speed_seconds: Optional[float]
    exposure_compensation_ev: Optional[float]
    orientation: Optional[str]


# ============================================================================
# Phase 4D: Metadata Manifest Types
# ============================================================================


class MetadataManifestEntry(TypedDict):
    """A single entry in the metadata manifest."""

    relative_path: str
    file_sha256: str
    metadata_sha256: str


class MetadataManifestPayload(TypedDict):
    """Complete metadata manifest payload (Phase 4D artifact)."""

    metadata_manifest_contract_version: str
    metadata_contract_version: str
    entries: List[MetadataManifestEntry]


# ============================================================================
# Phase 4E: Provenance Manifest and Merkle Types
# ============================================================================


class ProvenanceManifestEntry(TypedDict):
    """A single entry in the provenance manifest."""

    relative_path: str
    file_sha256: str
    metadata_sha256: str
    provenance_entry_sha256: str


class ProvenanceManifestPayload(TypedDict):
    """Complete provenance manifest payload (Phase 4E artifact)."""

    provenance_contract_version: str
    metadata_contract_version: str
    entries: List[ProvenanceManifestEntry]


class ProvenanceMerklePayload(TypedDict):
    """Complete provenance Merkle payload (Phase 4E artifact)."""

    provenance_merkle_contract_version: str
    provenance_contract_version: str
    leaf_count: int
    provenance_merkle_root: str


# ============================================================================
# Phase 4F: Verification Report Types
# ============================================================================


class VerificationInputArtifact(TypedDict, total=False):
    """Input artifact summary for verification reports."""

    file_sha256: Optional[str]
    metadata_contract_version: Optional[str]
    metadata_manifest_contract_version: Optional[str]
    provenance_contract_version: Optional[str]
    provenance_merkle_contract_version: Optional[str]


class VerificationInputsBlock(TypedDict):
    """All input artifact summaries for a verification report."""

    capture_metadata: VerificationInputArtifact
    metadata_manifest: VerificationInputArtifact
    provenance_manifest: VerificationInputArtifact
    provenance_merkle: VerificationInputArtifact


class VerificationComputedBlock(TypedDict, total=False):
    """Computed verification results (None on failure)."""

    metadata_sha256_summary_sha256: Optional[str]
    metadata_entry_count: Optional[int]
    provenance_entry_sha256_summary_sha256: Optional[str]
    provenance_entry_count: Optional[int]
    provenance_merkle_root: Optional[str]
    provenance_leaf_count: Optional[int]


class VerifierInfo(TypedDict):
    """Verifier identification for reports."""

    name: str
    version: str
    build_id: str


class VerificationStatus(TypedDict):
    """Pass/fail status with optional failure details."""

    passed: bool
    failure_code_label: Optional[str]
    failure_message: Optional[str]


class VerificationReportPayload(TypedDict):
    """Complete verification report payload (Phase 4F artifact)."""

    verification_contract_version: str
    inputs: VerificationInputsBlock
    computed: VerificationComputedBlock
    verifier: VerifierInfo
    verification_status: VerificationStatus


# ============================================================================
# Progress Callback Protocol
# ============================================================================


class ProgressCallback(Protocol):
    """Protocol for progress reporting callbacks.

    Implementations receive progress updates during long-running operations.
    """

    def __call__(self, current: int, total: int, message: str) -> None:
        """Report progress on an operation.

        Args:
            current: Current progress count (0 to total).
            total: Total expected count.
            message: Human-readable progress message.
        """
        ...


# Type alias for the callback (for simpler type hints)
ProgressCallbackType = Callable[[int, int, str], None]


# ============================================================================
# ExifTool Runner Protocol
# ============================================================================


class ExifToolRunner(Protocol):
    """Protocol for custom ExifTool execution implementations."""

    def __call__(
        self,
        file_paths: list[Any],  # list[Path]
        tag_whitelist: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Execute ExifTool and return tag data by resolved file path.

        Args:
            file_paths: List of Path objects to extract metadata from.
            tag_whitelist: List of EXIF tags to extract.

        Returns:
            Dictionary mapping resolved file path strings to tag dictionaries.
        """
        ...
