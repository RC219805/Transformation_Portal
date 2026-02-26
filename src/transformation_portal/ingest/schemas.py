"""Versioned JSON schemas for ingest contract (v1.0.0).

Defines immutable, audit-grade schemas for:
- IngestManifest: Ingest output contract
- ProvenanceSidecar: Full metadata and provenance record

Schema guarantees:
- Versioned and backward-compatible
- Required fields enforced at validation time
- Type safety via Pydantic
- Deterministic serialization (sorted keys, normalized types)
- No silent fallbacks or inference

Contract version: 1.0.0
Schema version: 1.0.0
"""

from __future__ import annotations

import datetime
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, validator


class ToolchainVersion(BaseModel):
    """Toolchain version metadata for reproducibility.

    Attributes:
        name: Tool name (e.g., "exiftool", "ImageMagick", "libraw")
        version: Semantic version or commit SHA
        path: Optional path to binary
    """

    name: str
    version: str
    path: Optional[str] = None

    model_config = ConfigDict(
        frozen=True,  # Immutable for audit integrity
        extra="forbid",  # Reject unknown fields (schema drift detection)
    )


class HostEnvironment(BaseModel):
    """Host environment metadata.

    Attributes:
        hostname: System hostname
        os: Operating system (e.g., "Linux", "Darwin", "Windows")
        os_version: OS version string
        python_version: Python interpreter version
        arch: CPU architecture (e.g., "x86_64", "arm64")
    """

    hostname: str
    os: str
    os_version: str
    python_version: str
    arch: str

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )


class IngestTimestamps(BaseModel):
    """Ingest timing metadata (all UTC).

    Attributes:
        ingest_start: ISO 8601 timestamp when ingest began
        ingest_end: ISO 8601 timestamp when ingest completed
        exiftool_extract_duration_sec: Time spent in exiftool extraction
    """

    ingest_start: str  # ISO 8601 with timezone
    ingest_end: str
    exiftool_extract_duration_sec: Optional[float] = None

    @validator("ingest_start", "ingest_end")
    def validate_iso_format(cls, v):
        """Ensure timestamps are valid ISO 8601 format with timezone.

        Rejects naive timestamps (no timezone).
        Requires either 'Z' suffix or explicit offset (e.g., '+00:00', '-08:00').
        """
        try:
            # Parse timestamp
            dt = datetime.datetime.fromisoformat(v.replace("Z", "+00:00"))

            # Reject naive timestamps (no timezone info)
            if dt.tzinfo is None:
                raise ValueError(
                    f"Naive timestamp not allowed (missing timezone): {v}. "
                    f"Use 'Z' suffix or explicit offset (e.g., '+00:00')"
                )

        except ValueError as e:
            raise ValueError(f"Invalid ISO 8601 timestamp: {v}") from e
        return v

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )


class FileIntegrity(BaseModel):
    """File integrity metadata.

    Attributes:
        sha256: SHA256 hash of input file
        size_bytes: File size in bytes
        path: Relative or absolute path to file
        mime_type: MIME type detected (e.g., "image/tiff", "image/x-canon-cr2")
    """

    sha256: str
    size_bytes: int
    path: str
    mime_type: Optional[str] = None

    @validator("sha256")
    def validate_sha256(cls, v):
        """Ensure SHA256 is 64 hex characters."""
        if not (len(v) == 64 and all(c in "0123456789abcdef" for c in v.lower())):
            raise ValueError(f"Invalid SHA256 hash: {v}")
        return v.lower()

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )


class ExifMetadata(BaseModel):
    """Complete EXIF metadata extracted via exiftool.

    All fields are optional as EXIF availability varies by format and camera.

    Attributes:
        all_tags: Complete exiftool JSON output (all groups + tags)
        camera_make: Camera manufacturer (e.g., "Canon", "Nikon")
        camera_model: Camera model (e.g., "Canon EOS 5D Mark IV")
        lens_model: Lens model if available
        iso: ISO speed rating
        aperture: Aperture value (f-number)
        shutter_speed: Shutter speed (e.g., "1/250")
        focal_length: Focal length in mm
        white_balance: White balance mode (e.g., "Auto", "Daylight")
        color_space: Color space (e.g., "sRGB", "Adobe RGB")
        width: Image width in pixels
        height: Image height in pixels
        bit_depth: Bits per channel
        datetime_original: Original capture timestamp (EXIF DateTimeOriginal)
        gps_latitude: GPS latitude if available
        gps_longitude: GPS longitude if available
    """

    all_tags: Dict[str, Any]  # Full exiftool JSON

    # Commonly used fields (extracted for convenience)
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    lens_model: Optional[str] = None
    iso: Optional[int] = None
    aperture: Optional[float] = None
    shutter_speed: Optional[str] = None
    focal_length: Optional[float] = None
    white_balance: Optional[str] = None
    color_space: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    bit_depth: Optional[int] = None
    datetime_original: Optional[str] = None
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None

    @field_validator("focal_length", mode="before")
    @classmethod
    def normalize_focal_length(cls, value: Any) -> Any:
        """Normalize exiftool focal length strings (e.g. '24.0 mm')."""
        if value is None or isinstance(value, (int, float)):
            return value
        if not isinstance(value, str):
            return value

        text = value.strip()
        if not text:
            return None

        if text.lower().endswith("mm"):
            text = text[:-2].strip()

        # Some EXIF payloads can be emitted as simple fractions.
        fraction = re.fullmatch(r"([-+]?\d+)\s*/\s*(\d+)", text)
        if fraction:
            numerator, denominator = fraction.groups()
            if int(denominator) == 0:
                raise ValueError(f"Invalid focal_length: {value!r}")
            return float(numerator) / float(denominator)

        number = re.search(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", text)
        if number is None:
            raise ValueError(f"Invalid focal_length: {value!r}")
        return float(number.group(0))

    @field_validator("bit_depth", mode="before")
    @classmethod
    def normalize_bit_depth(cls, value: Any) -> Any:
        """Normalize exiftool bit depth strings (e.g. '8 8 8')."""
        if value is None or isinstance(value, int):
            return value

        if isinstance(value, (list, tuple)):
            if not value:
                return None
            return value[0]

        if not isinstance(value, str):
            return value

        text = value.strip()
        if not text:
            return None

        token = re.split(r"[\s,]+", text)[0]
        try:
            return int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid bit_depth: {value!r}") from exc

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )


class PipelineConfig(BaseModel):
    """Pipeline configuration fingerprint.

    Attributes:
        config_sha256: SHA256 hash of config dict (for determinism)
        cli_args: Command-line arguments used
        preset: Preset name if used
        custom_params: Custom parameter overrides
    """

    config_sha256: str
    cli_args: Optional[List[str]] = None
    preset: Optional[str] = None
    custom_params: Optional[Dict[str, Any]] = None

    @validator("config_sha256")
    def validate_sha256(cls, v):
        """Ensure config SHA256 is valid."""
        if not (len(v) == 64 and all(c in "0123456789abcdef" for c in v.lower())):
            raise ValueError(f"Invalid config SHA256: {v}")
        return v.lower()

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )


class ProvenanceSidecar(BaseModel):
    """Provenance sidecar schema (v1.0.0).

    Complete, lossless provenance record for audit-grade traceability.
    Emitted deterministically for every ingested RAW/TIFF file.

    Attributes:
        schema_version: Schema version (semantic versioning)
        file_integrity: File hash, size, and path
        exif: Complete EXIF metadata via exiftool
        toolchain: Versions of all tools used
        host: Host environment metadata
        timestamps: Ingest timing metadata
        pipeline_config: Pipeline configuration fingerprint
        git_commit: Git commit SHA at ingest time
        run_id: Unique run identifier (UUID v4) - non-deterministic by design
    """

    schema_version: Literal["1.0.0"] = "1.0.0"

    file_integrity: FileIntegrity
    exif: ExifMetadata
    toolchain: List[ToolchainVersion]
    host: HostEnvironment
    timestamps: IngestTimestamps
    pipeline_config: PipelineConfig
    git_commit: Optional[str] = None
    run_id: str  # UUID v4 (only non-deterministic field by design)

    @validator("schema_version")
    def validate_schema_version(cls, v):
        """Ensure schema version is supported."""
        if v != "1.0.0":
            raise ValueError(f"Unsupported ProvenanceSidecar schema version: {v}. " f"This code supports version 1.0.0 only.")
        return v

    @validator("git_commit")
    def validate_git_commit(cls, v):
        """Validate git commit SHA format if present."""
        if v is not None:
            if not (len(v) == 40 and all(c in "0123456789abcdef" for c in v.lower())):
                raise ValueError(f"Invalid git commit SHA: {v}")
            return v.lower()
        return v

    model_config = ConfigDict(
        frozen=True,  # Immutable for audit integrity
        extra="forbid",  # Reject unknown fields (schema drift detection)
    )

    def to_json_deterministic(self) -> str:
        """Serialize to deterministic JSON.

        Guarantees:
        - Sorted keys
        - Normalized whitespace
        - No non-deterministic fields except run_id (explicitly allowed)

        Returns:
            JSON string with sorted keys and 2-space indentation
        """
        import json

        return json.dumps(
            self.model_dump(),
            sort_keys=True,
            indent=2,
            separators=(",", ": "),
            ensure_ascii=False,
        )


class IngestManifest(BaseModel):
    """Ingest manifest schema (v1.0.0).

    Output contract for the ingest stage.
    Lighter-weight than ProvenanceSidecar (summary only).

    Attributes:
        schema_version: Schema version (semantic versioning)
        input_file: Input file path and integrity
        output_file: Output file path and integrity (if different from input)
        status: Ingest status ("success", "error", "skipped")
        error_message: Error message if status is "error"
        provenance_sidecar_path: Path to full provenance sidecar JSON
        ingest_duration_sec: Total ingest duration in seconds
    """

    schema_version: Literal["1.0.0"] = "1.0.0"

    input_file: FileIntegrity
    output_file: Optional[FileIntegrity] = None
    status: str  # "success", "error", "skipped"
    error_message: Optional[str] = None
    provenance_sidecar_path: str
    ingest_duration_sec: float

    @validator("schema_version")
    def validate_schema_version(cls, v):
        """Ensure schema version is supported."""
        if v != "1.0.0":
            raise ValueError(f"Unsupported IngestManifest schema version: {v}. " f"This code supports version 1.0.0 only.")
        return v

    @validator("status")
    def validate_status(cls, v):
        """Ensure status is one of allowed values."""
        allowed = {"success", "error", "skipped"}
        if v not in allowed:
            raise ValueError(f"Invalid status: {v}. Must be one of {allowed}")
        return v

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    def to_json_deterministic(self) -> str:
        """Serialize to deterministic JSON.

        Returns:
            JSON string with sorted keys and 2-space indentation
        """
        import json

        return json.dumps(
            self.model_dump(),
            sort_keys=True,
            indent=2,
            separators=(",", ": "),
            ensure_ascii=False,
        )
