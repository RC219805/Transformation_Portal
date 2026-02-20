"""Linear ingest pipeline for Spatial AI Foundation.

This module provides RAW/TIFF → float32 linear light decoding for research workflows.

Architecture (ADR-023, Issue #890 Phase I):
- Complete isolation from lux_depth_v3.raw_loader (no shared decode logic)
- Linear gamma (1.0) enforcement
- 16-bit → float32 pipeline (no 8-bit collapse)
- HDR preservation (values >1.0 allowed)
- Full provenance tracking (EXIF + ingest metadata)
- Versioned manifest schema
- Hard failure guardrails

Modules:
- linear_decoder: Core RAW/TIFF/PNG/EXR decoder
- provenance: EXIF extraction and metadata tracking
- manifest_schema: Versioned dataset manifests
- validators: Hard-constraint validation
- exceptions: Clear, actionable error messages

Usage:
    >>> from transformation_portal.spatial_ai.ingest import LinearDecoder, ProvenanceCapture
    >>> decoder = LinearDecoder(gamma=1.0, strict_ingest=True)
    >>> result = decoder.decode("scene.CR2", emit_exr=True, emit_provenance=True)
    >>> assert result.linear_rgb.max() > 1.0  # HDR preserved
    >>> assert result.gamma == 1.0  # Linear light

    >>> # Or use manifest builder for datasets
    >>> from transformation_portal.spatial_ai.ingest import DatasetManifestBuilder
    >>> builder = DatasetManifestBuilder(name="training_set_v1")
    >>> # ... add images ...
    >>> manifest = builder.build()
    >>> manifest.write(Path("manifest.json"))
"""

from __future__ import annotations

from .contracts import IngestOptions, decode_contract
from .exceptions import (
    BitDepthViolationError,
    ColorSpaceError,
    LinearIngestError,
    LinearityViolationError,
    ManifestError,
    ProvenanceError,
    RangeViolationError,
    SchemaVersionError,
    UnsupportedFormatError,
)
from .linear_decoder import LinearDecoder, LinearIngestResult, decode
from .manifest_schema import DatasetManifestBuilder, ImageManifestEntry, ManifestSchema
from .provenance import CameraMetadata, ProvenanceCapture, ProvenanceData
from .telemetry import IngestTelemetry, NullTelemetry
from .validators import (
    CURRENT_SCHEMA_VERSION,
    validate_bit_depth,
    validate_dtype,
    validate_gamma,
    validate_linear_output,
    validate_range,
    validate_schema_version,
)

__all__ = [
    # Phase II contract dispatcher
    "IngestOptions",
    "decode_contract",
    # Core decoder
    "LinearDecoder",
    "LinearIngestResult",
    "decode",
    # Provenance
    "ProvenanceCapture",
    "ProvenanceData",
    "CameraMetadata",
    # Telemetry
    "IngestTelemetry",
    "NullTelemetry",
    # Manifest schema
    "ManifestSchema",
    "DatasetManifestBuilder",
    "ImageManifestEntry",
    # Validators
    "validate_bit_depth",
    "validate_dtype",
    "validate_gamma",
    "validate_range",
    "validate_linear_output",
    "validate_schema_version",
    "CURRENT_SCHEMA_VERSION",
    # Exceptions
    "LinearIngestError",
    "BitDepthViolationError",
    "ColorSpaceError",
    "LinearityViolationError",
    "RangeViolationError",
    "SchemaVersionError",
    "ProvenanceError",
    "ManifestError",
    "UnsupportedFormatError",
]
