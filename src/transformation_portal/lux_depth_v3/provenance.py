"""Provenance and metadata capture for audit-grade reproducibility.

This module provides complete, lossless provenance capture for RAW/TIFF inputs,
producing deterministic, versioned sidecar records suitable for audit, replay,
and dataset governance.

Key features:
- Complete EXIF/metadata extraction via exiftool
- Deterministic JSON serialization (stable key ordering, normalized types)
- Versioned schema with validation at write time
- Policy-driven failure modes (strict for RAW/TIFF, best-effort for others)
- No silent drops or inference
- Capture of toolchain versions, CLI args, git SHA, environment

Provenance Contract:
- Schema version: 1.0.0
- Required fields enforced at write time
- Deterministic output (same input → same sidecar, except ingest_timestamp_utc)
- Colocated sidecar JSON files for audit trail

Policy Boundaries:
- RAW/TIFF inputs: require_exiftool=True
  (audit-grade, hard-fail on missing exiftool)
- JPG/PNG inputs: require_exiftool=False
  (best-effort, skip EXIF if unavailable)
- Enforcement via capture_provenance(require_exiftool=...) parameter
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..ingest.canonical_json import dumps_json

logger = logging.getLogger(__name__)

# Schema version for provenance records
PROVENANCE_SCHEMA_VERSION = "1.0.0"


class ProvenanceError(Exception):
    """Base exception for provenance capture failures."""


class ExiftoolNotFoundError(ProvenanceError):
    """Raised when exiftool is not available."""


class MissingRequiredFieldError(ProvenanceError):
    """Raised when required provenance field is missing."""


class SchemaValidationError(ProvenanceError):
    """Raised when schema validation fails."""


def _check_exiftool_available() -> bool:
    """Check if exiftool is available in PATH.

    Returns:
        True if exiftool is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["exiftool", "-ver"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_exiftool_version() -> Optional[str]:
    """Get exiftool version.

    Returns:
        Version string or None if not available
    """
    try:
        result = subprocess.run(
            ["exiftool", "-ver"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug(f"Failed to get exiftool version: {e}")
    return None


def extract_exif_metadata(image_path: Path) -> Dict[str, Any]:
    """Extract complete EXIF and file-level metadata using exiftool.

    This function captures ALL metadata tags and groups from the input file
    using exiftool's JSON output for maximum fidelity.

    Args:
        image_path: Path to image file (RAW or TIFF)

    Returns:
        Dictionary with complete metadata (all tags + groups)

    Raises:
        FileNotFoundError: If image file doesn't exist
        ExiftoolNotFoundError: If exiftool is not available
        ProvenanceError: If metadata extraction fails
    """
    # Pure precondition: file must exist (check first)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Policy constraint: tool must be available (check second)
    if not _check_exiftool_available():
        raise ExiftoolNotFoundError(
            "exiftool not found in PATH. "
            "Install with: apt-get install"
            " libimage-exiftool-perl"
            " (Ubuntu/Debian) "
            "or brew install exiftool (macOS)"
        )

    try:
        # Use -G (show group names) and -j (JSON output) for complete metadata
        # -a: allow duplicate tags
        # -s: use tag names instead of descriptions
        result = subprocess.run(
            ["exiftool", "-G", "-a", "-s", "-j", str(image_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        if result.returncode != 0:
            raise ProvenanceError("exiftool failed with code" f" {result.returncode}:" f" {result.stderr}")

        # Parse JSON output (exiftool returns a list with one dict per file)
        metadata_list = json.loads(result.stdout)
        if not metadata_list:
            raise ProvenanceError("exiftool returned no" f" metadata for {image_path}")

        return metadata_list[0]

    except json.JSONDecodeError as e:
        raise ProvenanceError("Failed to parse exiftool" f" JSON output: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise ProvenanceError("exiftool timed out after 30s") from e


def get_toolchain_versions() -> Dict[str, Optional[str]]:
    """Capture versions of all tools in the processing toolchain.

    Returns:
        Dictionary with tool versions (None if tool not available)
    """
    versions = {
        "python_version": sys.version,
        "exiftool_version": get_exiftool_version(),
    }

    # Try to get rawpy/LibRaw version (optional dependency)
    try:
        import rawpy

        versions["rawpy_version"] = getattr(rawpy, "__version__", "unknown")
        # LibRaw version is embedded in rawpy
        try:
            versions["libraw_version"] = rawpy.libraw_version
        except AttributeError:
            versions["libraw_version"] = None
    except ImportError:
        versions["rawpy_version"] = None
        versions["libraw_version"] = None

    # Try to get ImageMagick version (optional - used in some pipelines)
    try:
        result = subprocess.run(
            ["convert", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            # Extract version from first line: "Version: ImageMagick 7.1.0-..."
            first_line = result.stdout.split("\n")[0]
            if "Version:" in first_line:
                versions["imagemagick_version"] = first_line.split("Version:")[-1].strip()
            else:
                versions["imagemagick_version"] = None
        else:
            versions["imagemagick_version"] = None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        versions["imagemagick_version"] = None

    return versions


def get_git_commit_sha(repo_root: Optional[Path] = None) -> Optional[str]:
    """Get current git commit SHA.

    Args:
        repo_root: Repository root directory (defaults to current directory)

    Returns:
        Git commit SHA or None if not in a git repo
    """
    if repo_root is None:
        repo_root = Path.cwd()

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug(f"Failed to get git commit SHA: {e}")

    return None


@dataclass
class InputFileMetadata:
    """Metadata about the input file itself.

    All fields are required for audit trail.
    """

    file_path: str
    file_sha256: str
    file_size_bytes: int
    file_mtime_utc: str  # ISO 8601 timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)


@dataclass
class IngestContext:
    """Context information about the ingestion process.

    This captures information about the
    processing environment and configuration.

    Determinism Contract:
    - ingest_timestamp_utc is NONDETERMINISTIC
      by design (captures actual ingest time)
    - All other fields are deterministic (same input + config → same value)
    """

    git_commit_sha: Optional[str]
    config_fingerprint: str
    ingest_timestamp_utc: str  # NONDETERMINISTIC: actual ingest time
    host_os: str
    host_machine: str
    cli_args: Optional[list] = None
    working_directory: Optional[str] = None
    ingest_profile: Optional[str] = None
    ingest_settings_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)


@dataclass
class ProvenanceMetadata:
    """Complete provenance metadata for an ingested file.

    Schema version: 1.0.0

    This is the top-level provenance record
    that gets written to the sidecar JSON.
    All required fields are enforced at write time.

    Determinism Contract:
    - File hash, EXIF, toolchain versions, config fingerprint: DETERMINISTIC
    - ingest_timestamp_utc: NONDETERMINISTIC (captures actual ingest time)
    - Same input file + config → same provenance (except timestamp)

    Attributes:
        schema_version: Provenance schema version for forward compatibility
        input: Input file metadata (path, hash, size, mtime)
        exif: Complete EXIF/metadata from exiftool
        toolchain: Versions of all tools in the processing chain
        ingest_context: Context about the ingestion process
    """

    schema_version: str
    input: InputFileMetadata
    exif: Dict[str, Any]
    toolchain: Dict[str, Optional[str]]
    ingest_context: IngestContext

    def validate_required_fields(self) -> None:
        """Validate that all required fields are present and well-formed.

        Raises:
            MissingRequiredFieldError: If required field is missing
            SchemaValidationError: If field is malformed
        """
        # Validate schema version
        if self.schema_version != PROVENANCE_SCHEMA_VERSION:
            raise SchemaValidationError(
                "Schema version mismatch:" f" expected {PROVENANCE_SCHEMA_VERSION}," f" got {self.schema_version}"
            )

        # Validate input metadata fields
        if not self.input.file_path:
            raise MissingRequiredFieldError("input.file_path is required")
        if not self.input.file_sha256:
            raise MissingRequiredFieldError("input.file_sha256 is required")
        if self.input.file_size_bytes <= 0:
            raise MissingRequiredFieldError("input.file_size_bytes" " must be positive")
        if not self.input.file_mtime_utc:
            raise MissingRequiredFieldError("input.file_mtime_utc is required")

        # Validate EXIF metadata is present
        # (can be empty dict for files without EXIF)
        if self.exif is None:
            raise MissingRequiredFieldError("exif metadata is required")

        # Validate toolchain has required tools
        if "python_version" not in self.toolchain or not self.toolchain["python_version"]:
            raise MissingRequiredFieldError("toolchain.python_version" " is required")
        # Note: exiftool_version is only required
        # for audit-grade RAW/TIFF provenance.
        # For non-audit inputs, exiftool_version
        # may be None. Enforcement is via
        # require_exiftool parameter in
        # capture_provenance()

        # Validate ingest context
        if not self.ingest_context.config_fingerprint:
            raise MissingRequiredFieldError("ingest_context" ".config_fingerprint" " is required")
        if not self.ingest_context.ingest_timestamp_utc:
            raise MissingRequiredFieldError("ingest_context" ".ingest_timestamp_utc" " is required")
        if not self.ingest_context.host_os:
            raise MissingRequiredFieldError("ingest_context.host_os" " is required")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary with deterministic key ordering.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "schema_version": self.schema_version,
            "input": self.input.to_dict(),
            "exif": self.exif,
            "toolchain": self.toolchain,
            "ingest_context": self.ingest_context.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to deterministic JSON string.

        Uses stable key ordering and normalized formatting for reproducibility.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string
        """
        return dumps_json(
            self.to_dict(),
            indent=indent,
            sort_keys=True,  # Deterministic key ordering
            separators=(",", ": "),  # Normalized separators
            ensure_ascii=False,
            allow_nan=False,
        )

    def write_sidecar(self, sidecar_path: Path) -> None:
        """Write provenance metadata to sidecar JSON file.

        Validates all required fields before writing.
        Uses atomic write pattern for safety.

        Args:
            sidecar_path: Path to sidecar JSON file

        Raises:
            MissingRequiredFieldError: If required field is missing
            SchemaValidationError: If validation fails
        """
        # Validate before writing
        self.validate_required_fields()

        # Create parent directory if needed
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp file, then rename
        temp_path = sidecar_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w") as f:
                f.write(self.to_json())
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename
            temp_path.replace(sidecar_path)
            logger.info(f"Wrote provenance sidecar: {sidecar_path}")

        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise ProvenanceError(f"Failed to write sidecar: {e}") from e

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProvenanceMetadata:
        """Deserialize from dictionary with schema validation.

        Args:
            data: Dictionary representation

        Returns:
            ProvenanceMetadata instance

        Raises:
            SchemaValidationError: If schema version is unsupported
        """
        schema_version = data.get("schema_version")
        if schema_version != PROVENANCE_SCHEMA_VERSION:
            raise SchemaValidationError(
                f"Unsupported provenance schema version: {schema_version}. "
                f"This code supports {PROVENANCE_SCHEMA_VERSION} only."
            )

        return cls(
            schema_version=schema_version,
            input=InputFileMetadata(**data["input"]),
            exif=data["exif"],
            toolchain=data["toolchain"],
            ingest_context=IngestContext(**data["ingest_context"]),
        )

    @classmethod
    def load_sidecar(cls, sidecar_path: Path) -> ProvenanceMetadata:
        """Load provenance metadata from sidecar JSON file.

        Args:
            sidecar_path: Path to sidecar JSON file

        Returns:
            ProvenanceMetadata instance

        Raises:
            FileNotFoundError: If sidecar doesn't exist
            SchemaValidationError: If schema validation fails
        """
        with open(sidecar_path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)


def capture_provenance(
    image_path: Path,
    config_fingerprint: str,
    cli_args: Optional[list] = None,
    repo_root: Optional[Path] = None,
    require_exiftool: bool = True,
    ingest_profile: Optional[str] = None,
    ingest_settings_hash: Optional[str] = None,
) -> ProvenanceMetadata:
    """Capture complete provenance metadata for an input file.

    This is the main entry point for provenance capture.

    Args:
        image_path: Path to input file (RAW or TIFF)
        config_fingerprint: SHA256 hash of pipeline configuration
        cli_args: Command-line arguments (optional)
        repo_root: Repository root for git SHA capture (optional)
        require_exiftool: If True, hard-fail when exiftool unavailable.
                         If False, skip EXIF
                         extraction and continue
                         (default: True)
        ingest_profile: Optional canonical ingest profile identifier.
        ingest_settings_hash: Optional SHA-256
                         digest of canonical
                         ingest settings.

    Returns:
        ProvenanceMetadata instance ready for writing

    Raises:
        ExiftoolNotFoundError: If exiftool is
            not available and
            require_exiftool=True
        FileNotFoundError: If image file doesn't exist
        ProvenanceError: If provenance capture fails
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Compute file hash
    file_sha256 = _compute_file_sha256(image_path)

    # Get file stats
    stat = image_path.stat()
    file_size_bytes = stat.st_size
    file_mtime_utc = datetime.datetime.fromtimestamp(
        stat.st_mtime,
        tz=datetime.timezone.utc,
    ).isoformat()

    # Extract EXIF metadata
    # If exiftool not required and unavailable, use empty dict
    if require_exiftool:
        exif_metadata = extract_exif_metadata(image_path)
    else:
        # Try to extract EXIF, but don't fail if exiftool unavailable
        if _check_exiftool_available():
            try:
                exif_metadata = extract_exif_metadata(image_path)
            except Exception as e:
                logger.debug(f"EXIF extraction failed (non-fatal): {e}")
                exif_metadata = {}
        else:
            logger.debug("exiftool not available, skipping EXIF extraction")
            exif_metadata = {}

    # Capture toolchain versions
    toolchain = get_toolchain_versions()

    # Capture ingest context
    git_sha = get_git_commit_sha(repo_root)
    ingest_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    host_os = platform.platform()
    host_machine = platform.machine()

    # Build provenance record
    input_metadata = InputFileMetadata(
        file_path=str(image_path),
        file_sha256=file_sha256,
        file_size_bytes=file_size_bytes,
        file_mtime_utc=file_mtime_utc,
    )

    ingest_context = IngestContext(
        git_commit_sha=git_sha,
        config_fingerprint=config_fingerprint,
        ingest_timestamp_utc=ingest_timestamp,
        host_os=host_os,
        host_machine=host_machine,
        cli_args=cli_args,
        working_directory=str(Path.cwd()),
        ingest_profile=ingest_profile,
        ingest_settings_hash=ingest_settings_hash,
    )

    provenance = ProvenanceMetadata(
        schema_version=PROVENANCE_SCHEMA_VERSION,
        input=input_metadata,
        exif=exif_metadata,
        toolchain=toolchain,
        ingest_context=ingest_context,
    )

    # Validate before returning
    provenance.validate_required_fields()

    return provenance


def _compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of file.

    Uses chunked reading for large files to avoid memory issues.

    Args:
        file_path: Path to file

    Returns:
        Hex digest of SHA256 hash
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in 64KB chunks
        for chunk in iter(lambda: f.read(65536), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()
