"""Provenance capture for RAW/TIFF ingest.

Provides full, lossless metadata extraction using exiftool and captures
ingest-time metadata for audit-grade traceability.

Key features:
- Complete EXIF extraction via exiftool (all tags + groups)
- Toolchain version capture (exiftool, ImageMagick/libraw if used)
- Git commit SHA at ingest time
- Host environment metadata
- Deterministic output (except run_id UUID)

Requirements:
- exiftool must be installed and in PATH
- Git repository (optional, gracefully handles missing)

Usage:
    from transformation_portal.ingest import capture_provenance
    
    sidecar = capture_provenance(
        input_path=Path("input.cr2"),
        cli_args=["--preset", "luxury"],
        config_dict={"model": "da3", "device": "mps"},
    )
    
    # Write deterministic sidecar
    with open("input_provenance.json", "w") as f:
        f.write(sidecar.to_json_deterministic())
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import (
    ExifMetadata,
    FileIntegrity,
    HostEnvironment,
    IngestTimestamps,
    PipelineConfig,
    ProvenanceSidecar,
    ToolchainVersion,
)

logger = logging.getLogger(__name__)


class ExiftoolNotFoundError(Exception):
    """Raised when exiftool is not found in PATH."""
    pass


class ProvenanceCaptureError(Exception):
    """Raised when provenance capture fails."""
    pass


def _check_exiftool_available() -> bool:
    """Check if exiftool is installed and in PATH.
    
    Returns:
        True if exiftool is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["exiftool", "-ver"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _get_exiftool_version() -> Optional[str]:
    """Get exiftool version.
    
    Returns:
        Version string or None if unavailable
    """
    try:
        result = subprocess.run(
            ["exiftool", "-ver"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _extract_exif_with_exiftool(file_path: Path) -> Dict[str, Any]:
    """Extract complete EXIF metadata using exiftool.
    
    Captures all tags from all groups (EXIF, XMP, IPTC, Composite, etc.).
    
    Args:
        file_path: Path to image file
        
    Returns:
        Dictionary with complete exiftool JSON output
        
    Raises:
        ExiftoolNotFoundError: If exiftool not installed
        ProvenanceCaptureError: If exiftool extraction fails
    """
    if not _check_exiftool_available():
        raise ExiftoolNotFoundError(
            "exiftool not found in PATH. "
            "Install with: brew install exiftool (macOS) or apt-get install libimage-exiftool-perl (Linux)"
        )
    
    try:
        result = subprocess.run(
            [
                "exiftool",
                "-j",  # JSON output
                "-a",  # Allow duplicate tags
                "-G",  # Show group names
                "-struct",  # Enable struct feature
                "-coordFormat", "%.8f",  # GPS coordinate format
                str(file_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode != 0:
            raise ProvenanceCaptureError(
                f"exiftool extraction failed for {file_path.name}: {result.stderr}"
            )
        
        # Parse JSON output (exiftool returns array with single object)
        data = json.loads(result.stdout)
        if not data:
            raise ProvenanceCaptureError(f"exiftool returned empty data for {file_path.name}")
        
        return data[0]  # First (and only) object in array
        
    except subprocess.TimeoutExpired as e:
        raise ProvenanceCaptureError(
            f"exiftool extraction timed out for {file_path.name}"
        ) from e
    except json.JSONDecodeError as e:
        raise ProvenanceCaptureError(
            f"Failed to parse exiftool JSON output for {file_path.name}"
        ) from e


def _compute_file_sha256(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA256 hash with minimal memory overhead.
    
    Args:
        file_path: Path to file
        chunk_size: Size of chunks to read (default 8KB)
        
    Returns:
        Hexadecimal SHA256 hash (lowercase)
    """
    sha256 = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    
    return sha256.hexdigest()


def _get_git_commit() -> Optional[str]:
    """Get current git commit SHA.
    
    Returns:
        Commit SHA or None if not in git repo or git unavailable
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return None


def _capture_toolchain_versions() -> List[ToolchainVersion]:
    """Capture versions of all tools in the toolchain.
    
    Returns:
        List of ToolchainVersion objects
    """
    versions = []
    
    # exiftool (required)
    exiftool_ver = _get_exiftool_version()
    if exiftool_ver:
        versions.append(
            ToolchainVersion(
                name="exiftool",
                version=exiftool_ver,
                path=None,  # Don't capture path for security
            )
        )
    
    # ImageMagick (optional)
    try:
        result = subprocess.run(
            ["convert", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Parse version from first line: "Version: ImageMagick 7.1.0-62"
            lines = result.stdout.split("\n")
            if lines:
                version_line = lines[0]
                if "ImageMagick" in version_line:
                    parts = version_line.split()
                    if len(parts) >= 3:
                        versions.append(
                            ToolchainVersion(
                                name="ImageMagick",
                                version=parts[2],
                                path=None,
                            )
                        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # libraw (via rawpy - optional)
    try:
        import rawpy
        versions.append(
            ToolchainVersion(
                name="rawpy",
                version=rawpy.version.version,
                path=None,
            )
        )
        # Also capture libraw version if available
        if hasattr(rawpy, "libraw_version"):
            versions.append(
                ToolchainVersion(
                    name="libraw",
                    version=rawpy.libraw_version,
                    path=None,
                )
            )
    except ImportError:
        pass
    
    # Python version (always present)
    versions.append(
        ToolchainVersion(
            name="python",
            version=sys.version.split()[0],
            path=sys.executable,
        )
    )
    
    return versions


def _capture_host_environment() -> HostEnvironment:
    """Capture host environment metadata.
    
    Returns:
        HostEnvironment object
    """
    return HostEnvironment(
        hostname=socket.gethostname(),
        os=platform.system(),
        os_version=platform.release(),
        python_version=sys.version.split()[0],
        arch=platform.machine(),
    )


def _compute_config_sha256(config_dict: Dict[str, Any]) -> str:
    """Compute deterministic SHA256 hash of config dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Hexadecimal SHA256 hash (lowercase)
    """
    # Serialize with sorted keys for determinism
    payload = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_exif_fields(exif_all_tags: Dict[str, Any]) -> ExifMetadata:
    """Parse commonly used EXIF fields from exiftool output.
    
    Args:
        exif_all_tags: Complete exiftool JSON output
        
    Returns:
        ExifMetadata object with parsed fields
    """
    # Helper to safely get values with various group prefixes
    def get_field(*keys: str) -> Any:
        for key in keys:
            if key in exif_all_tags:
                return exif_all_tags[key]
        return None
    
    return ExifMetadata(
        all_tags=exif_all_tags,
        camera_make=get_field("EXIF:Make", "Make"),
        camera_model=get_field("EXIF:Model", "Model"),
        lens_model=get_field("EXIF:LensModel", "LensModel"),
        iso=get_field("EXIF:ISO", "ISO"),
        aperture=get_field("EXIF:FNumber", "FNumber", "Aperture"),
        shutter_speed=get_field("EXIF:ShutterSpeed", "ShutterSpeed", "ExposureTime"),
        focal_length=get_field("EXIF:FocalLength", "FocalLength"),
        white_balance=get_field("EXIF:WhiteBalance", "WhiteBalance"),
        color_space=get_field("EXIF:ColorSpace", "ColorSpace"),
        width=get_field("EXIF:ImageWidth", "ImageWidth"),
        height=get_field("EXIF:ImageHeight", "ImageHeight"),
        bit_depth=get_field("EXIF:BitsPerSample", "BitsPerSample"),
        datetime_original=get_field("EXIF:DateTimeOriginal", "DateTimeOriginal"),
        gps_latitude=get_field("EXIF:GPSLatitude", "GPSLatitude"),
        gps_longitude=get_field("EXIF:GPSLongitude", "GPSLongitude"),
    )


def capture_provenance(
    input_path: Path,
    cli_args: Optional[List[str]] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    preset: Optional[str] = None,
    run_id: Optional[str] = None,
) -> ProvenanceSidecar:
    """Capture complete provenance metadata for input file.
    
    Extracts:
    - Complete EXIF metadata via exiftool (all tags + groups)
    - File integrity (SHA256, size, MIME type)
    - Toolchain versions (exiftool, ImageMagick, libraw, Python)
    - Host environment (hostname, OS, architecture)
    - Pipeline configuration fingerprint
    - Git commit SHA (if in git repo)
    - Timestamps (UTC)
    
    Args:
        input_path: Path to input file (RAW or TIFF)
        cli_args: Command-line arguments used (if any)
        config_dict: Configuration dictionary
        preset: Preset name (if used)
        run_id: Optional run UUID (generated if not provided)
        
    Returns:
        ProvenanceSidecar object with complete metadata
        
    Raises:
        ExiftoolNotFoundError: If exiftool not installed
        ProvenanceCaptureError: If metadata extraction fails
        FileNotFoundError: If input file doesn't exist
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    start_time = datetime.now(timezone.utc)
    
    # Extract EXIF metadata via exiftool
    logger.info(f"Extracting EXIF metadata from {input_path.name} using exiftool...")
    exiftool_start = datetime.now(timezone.utc)
    exif_all_tags = _extract_exif_with_exiftool(input_path)
    exiftool_duration = (datetime.now(timezone.utc) - exiftool_start).total_seconds()
    
    exif_metadata = _parse_exif_fields(exif_all_tags)
    
    # Compute file integrity
    logger.debug(f"Computing SHA256 hash for {input_path.name}...")
    file_sha256 = _compute_file_sha256(input_path)
    file_size = input_path.stat().st_size
    
    # Detect MIME type from exiftool if available
    mime_type = exif_all_tags.get("File:MIMEType") or exif_all_tags.get("MIMEType")
    
    file_integrity = FileIntegrity(
        sha256=file_sha256,
        size_bytes=file_size,
        path=str(input_path),
        mime_type=mime_type,
    )
    
    # Capture toolchain versions
    toolchain = _capture_toolchain_versions()
    
    # Capture host environment
    host = _capture_host_environment()
    
    # Pipeline configuration
    if config_dict is None:
        config_dict = {}
    
    pipeline_config = PipelineConfig(
        config_sha256=_compute_config_sha256(config_dict),
        cli_args=cli_args,
        preset=preset,
        custom_params=config_dict if config_dict else None,
    )
    
    # Git commit
    git_commit = _get_git_commit()
    if git_commit:
        logger.debug(f"Captured git commit: {git_commit[:8]}")
    
    # Timestamps
    end_time = datetime.now(timezone.utc)
    timestamps = IngestTimestamps(
        ingest_start=start_time.isoformat(),
        ingest_end=end_time.isoformat(),
        exiftool_extract_duration_sec=exiftool_duration,
    )
    
    # Generate run_id if not provided
    if run_id is None:
        run_id = str(uuid.uuid4())
    
    # Construct sidecar
    sidecar = ProvenanceSidecar(
        file_integrity=file_integrity,
        exif=exif_metadata,
        toolchain=toolchain,
        host=host,
        timestamps=timestamps,
        pipeline_config=pipeline_config,
        git_commit=git_commit,
        run_id=run_id,
    )
    
    logger.info(
        f"Captured provenance for {input_path.name} "
        f"(SHA256: {file_sha256[:8]}..., size: {file_size} bytes, "
        f"camera: {exif_metadata.camera_make} {exif_metadata.camera_model})"
    )
    
    return sidecar
