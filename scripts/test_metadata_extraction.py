#!/usr/bin/env python3
"""Test Phase 3.7 metadata extraction capabilities on image directories.

This script provides commands for testing the Phase 3.7 metadata extraction
capabilities, including:
- Provenance capture (complete EXIF + toolchain + environment metadata)
- Sidecar generation (deterministic JSON output)
- Schema validation (schema compliance verification)
- Batch processing (multiple images)

Requirements:
- Package must be installed through the repo-managed setup path (`make install-core`)
- exiftool must be installed (brew install exiftool / apt-get install libimage-exiftool-perl)

Usage:
    # Single image extraction
    .venv/bin/python scripts/test_metadata_extraction.py extract /path/to/image.tif

    # Batch extraction (entire directory)
    .venv/bin/python scripts/test_metadata_extraction.py extract-batch /path/to/images/ --output ./output_sidecars/

    # Validate existing sidecar
    .venv/bin/python scripts/test_metadata_extraction.py validate /path/to/provenance.json

    # Check system readiness
    .venv/bin/python scripts/test_metadata_extraction.py check-system

    # Summary of extraction results
    .venv/bin/python scripts/test_metadata_extraction.py summarize /path/to/sidecars/

Exit Codes:
    0: Success
    1: Schema validation failed
    2: 8-bit conversion detected
    3: Gamma correction detected
    4: Schema drift detected
    5: Other failure (e.g., exiftool not found)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformation_portal.ingest import EXIT_OTHER_FAILURE, EXIT_SUCCESS
from transformation_portal.ingest import BatchExtractResult as ServiceBatchExtractResult
from transformation_portal.ingest import ExtractResult as ServiceExtractResult
from transformation_portal.ingest import IngestError, IngestExitCode, OtherIngestFailure
from transformation_portal.ingest import ValidateResult as ServiceValidateResult
from transformation_portal.ingest.machine_output import (
    MACHINE_SCHEMA_VERSION,
    batch_result_to_dict,
    dump_json,
    error_to_dict,
    extract_result_to_dict,
    validate_result_to_dict,
)
from transformation_portal.ingest.service import MetadataExtractionService as OrchestrationMetadataExtractionService
from transformation_portal.ingest.service import ServiceRunRequest

# =============================================================================
# Supported Image Extensions
# =============================================================================

SUPPORTED_EXTENSIONS = {
    # RAW formats
    ".cr2",
    ".cr3",  # Canon
    ".nef",
    ".nrw",  # Nikon
    ".arw",
    ".srf",  # Sony
    ".dng",  # Adobe DNG
    ".raf",  # Fujifilm
    ".orf",  # Olympus
    ".rw2",  # Panasonic
    ".pef",  # Pentax
    ".srw",  # Samsung
    # TIFF formats
    ".tif",
    ".tiff",
    # Common formats
    ".jpg",
    ".jpeg",
    ".png",
    ".heic",
    ".heif",
}


# =============================================================================
# Result Dataclasses (Execution/Rendering Separation)
# =============================================================================


@dataclass
class SystemCheckResult:
    """Result of system check command."""

    exiftool_available: bool = False
    exiftool_version: Optional[str] = None
    pydantic_available: bool = False
    pydantic_version: Optional[str] = None
    git_available: bool = False
    git_version: Optional[str] = None
    rawpy_available: bool = False
    rawpy_version: Optional[str] = None
    libraw_version: Optional[str] = None
    ingest_module_available: bool = False
    errors: List[str] = field(default_factory=list)

    @property
    def all_required_ok(self) -> bool:
        """Check if all required dependencies are available."""
        return self.exiftool_available and self.pydantic_available and self.ingest_module_available


@dataclass
class ExtractResult:
    """Result of single image extraction."""

    success: bool
    image_path: Path
    output_path: Optional[Path] = None
    sidecar: Optional[Any] = None  # ProvenanceSidecar
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    ingest_error: Optional[IngestError] = None


@dataclass
class ValidateResult:
    """Result of sidecar validation."""

    success: bool
    sidecar_path: Path
    errors: List[str] = field(default_factory=list)
    exit_code: int = EXIT_SUCCESS
    sidecar_data: Optional[Dict[str, Any]] = None
    typed_errors: List[IngestError] = field(default_factory=list)
    dominant_error: Optional[IngestError] = None
    strict: bool = True


@dataclass
class SummarizeResult:
    """Result of sidecar summarization."""

    sidecar_dir: Path
    sidecar_count: int
    total_size_bytes: int = 0
    total_tags: int = 0
    gps_count: int = 0
    cameras: Dict[str, int] = field(default_factory=dict)
    dimensions: List[Tuple[int, int]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# =============================================================================
# Helper Functions
# =============================================================================


def find_images(directory: Path, recursive: bool = True) -> List[Path]:
    """Find all supported image files in directory.

    Args:
        directory: Directory to search
        recursive: If True, search subdirectories

    Returns:
        List of image file paths
    """
    images = []
    pattern = "**/*" if recursive else "*"

    for file_path in directory.glob(pattern):
        # Normalize suffix to lowercase so extension matching is case-insensitive.
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(file_path)

    return sorted(images)


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    size_float = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size_float < 1024:
            return f"{size_float:.1f} {unit}"
        size_float /= 1024
    return f"{size_float:.1f} TB"


# =============================================================================
# Execution Functions (Business Logic - Testable)
# =============================================================================


def run_check_system() -> SystemCheckResult:
    """Execute system check and return structured result."""
    result = SystemCheckResult()

    # Check exiftool
    try:
        from transformation_portal.ingest.provenance import _check_exiftool_available, _get_exiftool_version

        result.exiftool_available = _check_exiftool_available()
        if result.exiftool_available:
            result.exiftool_version = _get_exiftool_version()
    except ImportError as e:
        result.errors.append(f"exiftool check failed: {e}")

    # Check pydantic
    try:
        import pydantic

        result.pydantic_available = True
        result.pydantic_version = pydantic.__version__
    except ImportError:
        result.errors.append("pydantic not found")

    # Check git (optional)
    try:
        import subprocess

        git_result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if git_result.returncode == 0:
            result.git_available = True
            result.git_version = git_result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # Optional dependency

    # Check rawpy (optional)
    try:
        import rawpy

        result.rawpy_available = True
        rawpy_version = None
        rawpy_version_obj = getattr(rawpy, "version", None)
        if rawpy_version_obj is not None:
            rawpy_version = getattr(rawpy_version_obj, "version", None)
        if rawpy_version is None:
            rawpy_version = getattr(rawpy, "__version__", None)
        if rawpy_version is not None:
            result.rawpy_version = str(rawpy_version)

        libraw_version = getattr(rawpy, "libraw_version", None)
        if libraw_version is not None:
            if isinstance(libraw_version, (tuple, list)):
                result.libraw_version = ".".join(str(part) for part in libraw_version)
            else:
                result.libraw_version = str(libraw_version)
    except ImportError:
        pass  # Optional dependency

    # Check ingest module
    try:
        from transformation_portal.ingest import capture_provenance, validate_schema, write_sidecar  # noqa: F401

        result.ingest_module_available = True
    except ImportError as e:
        result.errors.append(f"ingest module not available: {e}")

    return result


def run_extract(
    image_path: Path,
    output_path: Optional[Path] = None,
    preset: Optional[str] = None,
    fsync: bool = False,
    cli_args: Optional[List[str]] = None,
) -> ExtractResult:
    """Execute single image extraction via orchestration service."""
    service = OrchestrationMetadataExtractionService()
    requested_output = output_path
    service_result = service.run(
        ServiceRunRequest(
            command="extract",
            input_path=image_path,
            args={
                "output_path": requested_output,
                "preset": preset,
                "fsync": fsync,
                "cli_args": cli_args or [],
                "config_dict": {"mode": "test_extraction", "phase": "3.7"},
            },
        )
    )

    payload = service_result.payload or {}
    extracted = payload.get("extract_result")
    # Extract command enrichment payload uses "sidecar" in the service contract.
    sidecar = payload.get("sidecar")
    if not isinstance(extracted, ServiceExtractResult):
        error_message = str(payload.get("error") or "Unknown extraction error")
        fallback_error = OtherIngestFailure(error_message)
        return ExtractResult(
            success=False,
            image_path=image_path,
            output_path=requested_output,
            elapsed_seconds=0.0,
            error=error_message,
            ingest_error=fallback_error,
        )

    if not extracted.success:
        return ExtractResult(
            success=False,
            image_path=image_path,
            output_path=extracted.output_path,
            elapsed_seconds=extracted.elapsed_seconds,
            error=str(extracted.error) if extracted.error else "Unknown extraction error",
            ingest_error=extracted.error,
        )

    return ExtractResult(
        success=True,
        image_path=image_path,
        output_path=extracted.output_path,
        sidecar=sidecar,
        elapsed_seconds=extracted.elapsed_seconds,
    )


def _build_batch_summary(
    *,
    total: int,
    success: int,
    failure: int,
    by_exit: Optional[Counter] = None,
) -> Dict[str, Any]:
    """Build deterministic batch summary payload keyed by ingest exit-code names."""
    by_exit = by_exit or Counter()
    return {
        "total": total,
        "success": success,
        "failure": failure,
        "by_exit_code": {
            code.name: by_exit.get(code, 0)
            for code in sorted(IngestExitCode, key=lambda code: code.value)
            if code != IngestExitCode.SUCCESS
        },
    }


def run_extract_batch(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    recursive: bool = True,
    fsync: bool = False,
    fail_fast: bool = False,
    cli_args: Optional[List[str]] = None,
) -> ServiceBatchExtractResult:
    """Execute batch extraction via orchestration service."""
    service = OrchestrationMetadataExtractionService()
    service_result = service.run(
        ServiceRunRequest(
            command="extract-batch",
            input_path=input_dir,
            output_dir=output_dir,
            args={
                "recursive": recursive,
                "fsync": fsync,
                "fail_fast": fail_fast,
                "cli_args": cli_args or [],
                "config_dict": {"mode": "batch_extraction", "phase": "3.7"},
            },
        )
    )
    payload = service_result.payload or {}
    batch_result = payload.get("batch_result")
    if isinstance(batch_result, ServiceBatchExtractResult):
        return batch_result

    fallback_error = OtherIngestFailure(str(payload.get("error") or "Unknown batch extraction error"))
    return ServiceBatchExtractResult(
        items=[],
        total_elapsed=0.0,
        summary_counts=_build_batch_summary(total=0, success=0, failure=0),
        dominant_error=fallback_error,
    )


def run_validate(sidecar_path: Path, strict: bool = True) -> ValidateResult:
    """Execute sidecar validation via orchestration service."""
    service = OrchestrationMetadataExtractionService()
    service_result = service.run(
        ServiceRunRequest(
            command="validate",
            input_path=sidecar_path,
            strict=strict,
            args={"schema_type": "provenance"},
        )
    )

    payload = service_result.payload or {}
    validated = payload.get("validate_result")
    if not isinstance(validated, ServiceValidateResult):
        error_message = str(payload.get("error") or "Unknown validation error")
        fallback_error = OtherIngestFailure(error_message)
        return ValidateResult(
            success=False,
            sidecar_path=sidecar_path,
            errors=[error_message],
            exit_code=EXIT_OTHER_FAILURE,
            typed_errors=[fallback_error],
            dominant_error=fallback_error,
            strict=strict,
        )

    if not validated.success:
        dominant_error = validated.dominant_error
        exit_code = int(dominant_error.exit_code) if dominant_error is not None else EXIT_OTHER_FAILURE
        return ValidateResult(
            success=False,
            sidecar_path=sidecar_path,
            errors=[error.message for error in validated.errors],
            exit_code=exit_code,
            typed_errors=validated.errors,
            dominant_error=dominant_error,
            strict=strict,
        )

    # Validate command enrichment payload uses "sidecar_data" in the service contract.
    data = payload.get("sidecar_data")

    return ValidateResult(
        success=True,
        sidecar_path=sidecar_path,
        exit_code=EXIT_SUCCESS,
        sidecar_data=data,
        strict=strict,
    )


def run_summarize(sidecar_dir: Path) -> SummarizeResult:
    """Execute sidecar summarization and return structured result."""
    if not sidecar_dir.exists():
        return SummarizeResult(
            sidecar_dir=sidecar_dir,
            sidecar_count=0,
            errors=[f"Directory not found: {sidecar_dir}"],
        )

    # Find sidecar files produced by both legacy and service-backed naming.
    sidecars = list(sidecar_dir.rglob("*_provenance.json"))
    sidecars.extend(sidecar_dir.rglob("*.provenance.json"))
    sidecars = sorted(set(sidecars))

    result = SummarizeResult(
        sidecar_dir=sidecar_dir,
        sidecar_count=len(sidecars),
    )

    if not sidecars:
        return result

    # Aggregate statistics
    for sidecar_path in sidecars:
        try:
            with open(sidecar_path) as f:
                data = json.load(f)

            # File integrity
            fi = data.get("file_integrity", {})
            result.total_size_bytes += fi.get("size_bytes", 0)

            # EXIF
            exif = data.get("exif", {})
            all_tags = exif.get("all_tags", {})
            result.total_tags += len(all_tags)

            # Camera
            make = exif.get("camera_make", "Unknown")
            model = exif.get("camera_model", "")
            camera = f"{make} {model}".strip()
            result.cameras[camera] = result.cameras.get(camera, 0) + 1

            # Dimensions
            width = exif.get("width")
            height = exif.get("height")
            if width and height:
                result.dimensions.append((width, height))

            # GPS
            if exif.get("gps_latitude") and exif.get("gps_longitude"):
                result.gps_count += 1

        except Exception as e:
            result.errors.append(f"Error reading {sidecar_path.name}: {e}")

    return result


# =============================================================================
# Rendering Functions (Presentation - Decoupled from Logic)
# =============================================================================


def render_check_system(result: SystemCheckResult) -> None:
    """Render system check result to stdout."""
    print("=" * 70)
    print("Phase 3.7 Metadata Extraction - System Check")
    print("=" * 70)
    print()

    # exiftool
    print("Checking exiftool...")
    if result.exiftool_available:
        print(f"  ✅ exiftool found: version {result.exiftool_version}")
    else:
        print("  ❌ exiftool not found")
        print("     Install with: brew install exiftool (macOS)")
        print("                   apt-get install libimage-exiftool-perl (Linux)")

    # pydantic
    print("\nChecking pydantic...")
    if result.pydantic_available:
        print(f"  ✅ pydantic found: version {result.pydantic_version}")
    else:
        print("  ❌ pydantic not found")
        print("     Install with: pip install pydantic>=2.0")

    # git
    print("\nChecking git (optional)...")
    if result.git_available:
        print(f"  ✅ git found: {result.git_version}")
    else:
        print("  ⚠️  git not available (git commit SHA will not be captured)")

    # rawpy
    print("\nChecking rawpy (optional, for RAW file support)...")
    if result.rawpy_available:
        print(f"  ✅ rawpy found: version {result.rawpy_version}")
        if result.libraw_version:
            print(f"      libraw version: {result.libraw_version}")
    else:
        print("  ⚠️  rawpy not found (RAW file reading may be limited)")
        print("     Bootstrap RAW runtime with: ./scripts/setup/install_raw_runtime.sh")

    # ingest module
    print("\nChecking transformation_portal.ingest module...")
    if result.ingest_module_available:
        print("  ✅ ingest module available")
    else:
        print("  ❌ ingest module not available")

    # Summary
    print()
    print("=" * 70)
    if result.all_required_ok:
        print("✅ System is ready for Phase 3.7 metadata extraction")
    else:
        print("❌ System has missing dependencies")


def render_extract(result: ExtractResult) -> None:
    """Render extraction result to stdout."""
    if not result.success:
        print(f"❌ Extraction failed: {result.error}")
        return

    sidecar = result.sidecar

    print("✅ Metadata extraction complete")
    if sidecar is None:
        print()
        print(f"⏱️  Extraction time:   {result.elapsed_seconds:.2f}s")
        print(f"📄 Sidecar written:   {result.output_path}")
        return

    print()
    print("📊 Extraction Summary:")
    print(f"   Schema version:    {sidecar.schema_version}")
    print(f"   File SHA256:       {sidecar.file_integrity.sha256[:16]}...")
    print(f"   File size:         {format_size(sidecar.file_integrity.size_bytes)}")
    print(f"   MIME type:         {sidecar.file_integrity.mime_type or 'unknown'}")
    print()
    print("📸 EXIF Summary:")
    exif = sidecar.exif
    if exif.camera_make:
        print(f"   Camera:            {exif.camera_make} {exif.camera_model or ''}")
    if exif.lens_model:
        print(f"   Lens:              {exif.lens_model}")
    if exif.iso:
        print(f"   ISO:               {exif.iso}")
    if exif.aperture:
        print(f"   Aperture:          f/{exif.aperture}")
    if exif.shutter_speed:
        print(f"   Shutter:           {exif.shutter_speed}")
    if exif.focal_length:
        print(f"   Focal length:      {exif.focal_length}mm")
    if exif.width and exif.height:
        print(f"   Dimensions:        {exif.width} x {exif.height}")
    if exif.bit_depth:
        print(f"   Bit depth:         {exif.bit_depth} bits")
    if exif.datetime_original:
        print(f"   Date taken:        {exif.datetime_original}")
    if exif.gps_latitude and exif.gps_longitude:
        print("   GPS present:       yes")
    print(f"   Total EXIF tags:   {len(exif.all_tags)}")
    print()
    print("🔧 Toolchain:")
    for tool in sidecar.toolchain:
        print(f"   {tool.name}: {tool.version}")
    print()
    print("💻 Host Environment:")
    print(f"   Hostname:          {sidecar.host.hostname}")
    print(f"   OS:                {sidecar.host.os} {sidecar.host.os_version}")
    print(f"   Architecture:      {sidecar.host.arch}")
    print(f"   Python:            {sidecar.host.python_version}")
    if sidecar.git_commit:
        print(f"   Git commit:        {sidecar.git_commit[:12]}")
    print()
    print(f"⏱️  Extraction time:   {result.elapsed_seconds:.2f}s")
    print(f"📄 Sidecar written:   {result.output_path}")


def render_extract_batch(
    result: ServiceBatchExtractResult,
    *,
    input_dir: Path,
    output_dir: Path,
    verbose: bool = False,
) -> None:
    """Render batch extraction result to stdout."""

    def display_path(path: Path) -> str:
        if path == input_dir:
            return "<input_dir>"
        try:
            return str(path.relative_to(input_dir))
        except ValueError:
            return str(path)

    print("=" * 70)
    print("Phase 3.7 Metadata Extraction - Batch Mode")
    print("=" * 70)
    print()
    summary = result.summary_counts
    print(f"📁 Input directory:  {input_dir}")
    print(f"📄 Output directory: {output_dir}")
    print(f"🖼️  Images found:     {summary['total']}")
    print()

    if verbose:
        for item in result.items:
            item_path = display_path(item.path)
            status = "✅" if item.success else "❌"
            if item.success:
                print(f"  {status} {item_path} ({item.elapsed_seconds:.2f}s)")
            else:
                print(f"  {status} {item_path}: {item.error or 'Unknown error'}")
        print()

    # Summary
    print("=" * 70)
    print("Batch Extraction Summary")
    print("=" * 70)

    print(f"Total images:     {summary['total']}")
    print(f"Processed:        {len(result.items)}")
    print(f"Successful:       {summary['success']}")
    print(f"Failed:           {summary['failure']}")
    print(f"Total time:       {result.total_elapsed:.2f}s")

    successful_timings = [item.elapsed_seconds for item in result.items if item.success and item.elapsed_seconds > 0]
    if successful_timings:
        avg_time = sum(successful_timings) / len(successful_timings)
        print(f"Average per image: {avg_time:.2f}s")

    print(f"Output directory: {output_dir}")

    if summary["failure"] > 0:
        print()
        print("Failed images:")
        for item in result.items:
            if item.success:
                continue
            print(f"  ❌ {display_path(item.path)}: {item.error or 'Unknown error'}")
    elif result.dominant_error is not None:
        print()
        print(f"❌ Batch setup failed: {result.dominant_error}")


def render_validate(result: ValidateResult, verbose: bool = False) -> None:
    """Render validation result to stdout."""
    print(f"🔍 Validating: {result.sidecar_path}")
    print()

    if not result.success:
        print("❌ Validation failed:")
        for error in result.errors:
            print(f"   - {error}")
        return

    print("✅ Sidecar is valid")

    if verbose and result.sidecar_data:
        data = result.sidecar_data
        print()
        print("📊 Sidecar Summary:")
        print(f"   Schema version: {data.get('schema_version')}")
        print(f"   Run ID:         {data.get('run_id')}")
        fi = data.get("file_integrity", {})
        sha256 = fi.get("sha256", "N/A")
        sha256_display = f"{sha256[:16]}..." if len(sha256) > 16 else sha256
        print(f"   File SHA256:    {sha256_display}")
        print(f"   File size:      {format_size(fi.get('size_bytes', 0))}")


def render_summarize(result: SummarizeResult) -> None:
    """Render summarization result to stdout."""
    print("=" * 70)
    print("Phase 3.7 Metadata Extraction - Summary")
    print("=" * 70)
    print()
    print(f"📁 Directory: {result.sidecar_dir}")
    print(f"📄 Sidecars found: {result.sidecar_count}")
    print()

    if result.sidecar_count == 0:
        return

    # Print summary
    print("📊 Aggregate Statistics:")
    print(f"   Total file size:    {format_size(result.total_size_bytes)}")
    print(f"   Total EXIF tags:    {result.total_tags}")
    average_tags = result.total_tags / result.sidecar_count if result.sidecar_count else 0
    gps_percent = (result.gps_count / result.sidecar_count * 100) if result.sidecar_count else 0
    print(f"   Average tags/image: {average_tags:.0f}")
    print(f"   Images with GPS:    {result.gps_count} ({gps_percent:.1f}%)")
    print()

    print("📷 Cameras:")
    for camera, count in sorted(result.cameras.items(), key=lambda x: -x[1]):
        print(f"   {camera}: {count}")
    print()

    if result.dimensions:
        print("📐 Dimensions:")
        dim_counts = Counter(result.dimensions)
        for (w, h), count in sorted(dim_counts.items(), key=lambda item: -(item[0][0] * item[0][1])):
            mp = (w * h) / 1_000_000
            print(f"   {w} x {h} ({mp:.1f} MP): {count}")

    if result.errors:
        print()
        print("⚠️  Warnings:")
        for error in result.errors:
            print(f"   {error}")


# =============================================================================
# Machine Output Helpers
# =============================================================================


def _build_machine_envelope(
    *,
    command: str,
    exit_code: int,
    data: Dict[str, Any],
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # Intentionally omit timestamps to keep machine-mode payloads deterministic.
    return {
        "schema": MACHINE_SCHEMA_VERSION,
        "command": command,
        "success": exit_code == EXIT_SUCCESS,
        "exit_code": exit_code,
        "data": data,
        "error": error,
    }


def _command_error_payload(exc: Exception) -> Dict[str, Any]:
    if isinstance(exc, IngestError):
        return error_to_dict(exc)
    return {"type": exc.__class__.__name__, "message": str(exc)}


def _summarize_result_to_machine_data(result: SummarizeResult) -> Dict[str, Any]:
    parsed_errors: List[Dict[str, Optional[str]]] = []
    for message in result.errors:
        if message.startswith("Error reading ") and ": " in message:
            prefix, detail = message.split(": ", 1)
            parsed_errors.append(
                {
                    "path": prefix.removeprefix("Error reading "),
                    "message": detail,
                }
            )
            continue
        parsed_errors.append({"path": None, "message": message})

    invalid = len(parsed_errors)
    valid = max(result.sidecar_count - invalid, 0)
    return {
        "sidecar_dir": str(result.sidecar_dir),
        "total_sidecars": result.sidecar_count,
        "valid": valid,
        "invalid": invalid,
        "errors": parsed_errors,
    }


def _check_system_result_to_machine_data(result: SystemCheckResult) -> Dict[str, Any]:
    return {
        "exiftool_available": result.exiftool_available,
        "exiftool_version": result.exiftool_version,
        "pydantic_available": result.pydantic_available,
        "pydantic_version": result.pydantic_version,
        "git_available": result.git_available,
        "git_version": result.git_version,
        "rawpy_available": result.rawpy_available,
        "rawpy_version": result.rawpy_version,
        "libraw_version": result.libraw_version,
        "ingest_module_available": result.ingest_module_available,
        "all_required_ok": result.all_required_ok,
        "errors": list(result.errors),
    }


def _emit_machine(envelope: Dict[str, Any], args: argparse.Namespace) -> None:
    try:
        payload = dump_json(envelope, pretty=args.json_pretty)
    except ValueError as exc:
        raise OtherIngestFailure(f"Machine JSON serialization rejected non-finite payload: {exc}") from exc
    if args.json_output:
        destination = Path(args.json_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                tmp_path = Path(handle.name)
            tmp_path.replace(destination)
        finally:
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
        return
    print(payload)


# =============================================================================
# Command Functions (Thin Wrappers: Execute -> Render -> Exit)
# =============================================================================


def cmd_check_system(args: argparse.Namespace) -> int:
    """Check system readiness for metadata extraction."""
    try:
        result = run_check_system()
        exit_code = EXIT_SUCCESS if result.all_required_ok else EXIT_OTHER_FAILURE
        if args.json:
            _emit_machine(
                _build_machine_envelope(
                    command="check-system",
                    exit_code=exit_code,
                    data=_check_system_result_to_machine_data(result),
                ),
                args,
            )
            return exit_code

        render_check_system(result)
        return exit_code
    except Exception as e:
        if args.debug:
            traceback.print_exc()
        if args.json:
            _emit_machine(
                _build_machine_envelope(
                    command="check-system",
                    exit_code=EXIT_OTHER_FAILURE,
                    data=_check_system_result_to_machine_data(SystemCheckResult()),
                    error=_command_error_payload(e),
                ),
                args,
            )
            return EXIT_OTHER_FAILURE
        print(f"❌ System check failed: {e}")
        return EXIT_OTHER_FAILURE


def cmd_extract(args: argparse.Namespace) -> int:
    """Extract metadata from a single image file."""
    image_path = Path(args.image_path)
    output_path = Path(args.output) if args.output else None

    if not args.json:
        print(f"📷 Extracting metadata from: {image_path.name}")
        print(f"   Path: {image_path}")
        if image_path.exists():
            print(f"   Size: {format_size(image_path.stat().st_size)}")
        print()

    try:
        result = run_extract(
            image_path=image_path,
            output_path=output_path,
            preset=args.preset,
            fsync=args.fsync,
            cli_args=sys.argv[1:],
        )
        exit_code = EXIT_SUCCESS
        if not result.success:
            exit_code = int(result.ingest_error.exit_code) if result.ingest_error is not None else EXIT_OTHER_FAILURE

        if args.json:
            service_result = ServiceExtractResult(
                path=result.image_path,
                success=result.success,
                output_path=result.output_path,
                elapsed_seconds=result.elapsed_seconds,
                error=result.ingest_error,
            )
            _emit_machine(
                _build_machine_envelope(
                    command="extract",
                    exit_code=exit_code,
                    data=extract_result_to_dict(service_result, preset=args.preset),
                ),
                args,
            )
            return exit_code

        render_extract(result)
        return exit_code
    except Exception as e:
        if args.debug:
            traceback.print_exc()
        if args.json:
            service_result = ServiceExtractResult(
                path=image_path,
                success=False,
                output_path=output_path,
                elapsed_seconds=0.0,
                error=None,
            )
            _emit_machine(
                _build_machine_envelope(
                    command="extract",
                    exit_code=EXIT_OTHER_FAILURE,
                    data=extract_result_to_dict(service_result, preset=args.preset),
                    error=_command_error_payload(e),
                ),
                args,
            )
            return EXIT_OTHER_FAILURE
        print(f"❌ Extraction failed: {e}")
        return EXIT_OTHER_FAILURE


def cmd_extract_batch(args: argparse.Namespace) -> int:
    """Extract metadata from all images in a directory."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output) if args.output else None

    try:
        result = run_extract_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            recursive=args.recursive,
            fsync=args.fsync,
            fail_fast=args.fail_fast,
            cli_args=sys.argv[1:],
        )
        resolved_output_dir = output_dir or input_dir / "provenance_sidecars"
        exit_code = EXIT_SUCCESS
        if result.dominant_error is None:
            if result.summary_counts.get("failure", 0) > 0:
                exit_code = EXIT_OTHER_FAILURE
        else:
            exit_code = int(result.dominant_error.exit_code)

        if args.json:
            _emit_machine(
                _build_machine_envelope(
                    command="extract-batch",
                    exit_code=exit_code,
                    data=batch_result_to_dict(
                        result,
                        input_root=input_dir,
                        output_dir=resolved_output_dir,
                        fail_fast=args.fail_fast,
                        preserve_structure=True,
                    ),
                ),
                args,
            )
            return exit_code

        render_extract_batch(
            result,
            input_dir=input_dir,
            output_dir=resolved_output_dir,
            verbose=args.verbose,
        )
        return exit_code

    except Exception as e:
        if args.debug:
            traceback.print_exc()
        if args.json:
            empty_batch_result = ServiceBatchExtractResult(
                items=[],
                total_elapsed=0.0,
                summary_counts=_build_batch_summary(total=0, success=0, failure=0),
                dominant_error=None,
            )
            data = batch_result_to_dict(
                empty_batch_result,
                input_root=input_dir,
                output_dir=output_dir or input_dir / "provenance_sidecars",
                fail_fast=args.fail_fast,
                preserve_structure=True,
            )
            data["success"] = False
            _emit_machine(
                _build_machine_envelope(
                    command="extract-batch",
                    exit_code=EXIT_OTHER_FAILURE,
                    data=data,
                    error=_command_error_payload(e),
                ),
                args,
            )
            return EXIT_OTHER_FAILURE
        print(f"❌ Batch extraction failed: {e}")
        return EXIT_OTHER_FAILURE


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a provenance sidecar JSON file."""
    sidecar_path = Path(args.sidecar_path)

    try:
        result = run_validate(sidecar_path, strict=args.strict)
        if args.json:
            service_result = ServiceValidateResult(
                success=result.success,
                errors=result.typed_errors,
                dominant_error=result.dominant_error,
            )
            _emit_machine(
                _build_machine_envelope(
                    command="validate",
                    exit_code=result.exit_code,
                    data=validate_result_to_dict(
                        service_result,
                        sidecar_path=result.sidecar_path,
                        strict=result.strict,
                    ),
                ),
                args,
            )
            return result.exit_code

        render_validate(result, verbose=args.verbose)
        return result.exit_code
    except Exception as e:
        if args.debug:
            traceback.print_exc()
        if args.json:
            service_result = ServiceValidateResult(
                success=False,
                errors=[],
                dominant_error=None,
            )
            _emit_machine(
                _build_machine_envelope(
                    command="validate",
                    exit_code=EXIT_OTHER_FAILURE,
                    data=validate_result_to_dict(service_result, sidecar_path=sidecar_path, strict=args.strict),
                    error=_command_error_payload(e),
                ),
                args,
            )
            return EXIT_OTHER_FAILURE
        print(f"❌ Validation error: {e}")
        return EXIT_OTHER_FAILURE


def cmd_summarize(args: argparse.Namespace) -> int:
    """Summarize metadata from multiple sidecar files."""
    sidecar_dir = Path(args.sidecar_dir)

    try:
        result = run_summarize(sidecar_dir)
        exit_code = EXIT_SUCCESS if result.sidecar_count > 0 or not result.errors else EXIT_OTHER_FAILURE
        if args.json:
            _emit_machine(
                _build_machine_envelope(
                    command="summarize",
                    exit_code=exit_code,
                    data=_summarize_result_to_machine_data(result),
                ),
                args,
            )
            return exit_code

        render_summarize(result)
        return exit_code
    except Exception as e:
        if args.debug:
            traceback.print_exc()
        if args.json:
            _emit_machine(
                _build_machine_envelope(
                    command="summarize",
                    exit_code=EXIT_OTHER_FAILURE,
                    data={
                        "sidecar_dir": str(sidecar_dir),
                        "total_sidecars": 0,
                        "valid": 0,
                        "invalid": 0,
                        "errors": [],
                    },
                    error=_command_error_payload(e),
                ),
                args,
            )
            return EXIT_OTHER_FAILURE
        print(f"❌ Summarization failed: {e}")
        return EXIT_OTHER_FAILURE


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test Phase 3.7 metadata extraction capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Global debug flag
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (show full tracebacks on errors)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output",
    )
    parser.add_argument(
        "--json-pretty",
        action="store_true",
        help="Pretty-print machine JSON output (requires --json)",
    )
    parser.add_argument(
        "--json-output",
        help="Write machine JSON to a file path (requires --json)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # check-system command
    parser_check = subparsers.add_parser(
        "check-system",
        help="Check system readiness for metadata extraction",
    )
    parser_check.set_defaults(func=cmd_check_system)

    # extract command
    parser_extract = subparsers.add_parser(
        "extract",
        help="Extract metadata from a single image",
    )
    parser_extract.add_argument(
        "image_path",
        help="Path to image file",
    )
    parser_extract.add_argument(
        "-o",
        "--output",
        help="Output path for sidecar JSON (default: <image>.provenance.json)",
    )
    parser_extract.add_argument(
        "--preset",
        help="Preset name to record in provenance",
    )
    parser_extract.add_argument(
        "--fsync",
        action="store_true",
        help="Use fsync for durable writes",
    )
    parser_extract.set_defaults(func=cmd_extract)

    # extract-batch command
    parser_batch = subparsers.add_parser(
        "extract-batch",
        help="Extract metadata from all images in a directory",
    )
    parser_batch.add_argument(
        "input_dir",
        help="Directory containing images",
    )
    parser_batch.add_argument(
        "-o",
        "--output",
        help="Output directory for sidecars (default: <input_dir>/provenance_sidecars/)",
    )
    recursive_group = parser_batch.add_mutually_exclusive_group()
    recursive_group.add_argument(
        "--recursive",
        action="store_true",
        dest="recursive",
        help="Recursively search subdirectories (default behavior)",
    )
    recursive_group.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Only process images in the top-level directory",
    )
    parser_batch.add_argument(
        "--fsync",
        action="store_true",
        help="Use fsync for durable writes",
    )
    parser_batch.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first error",
    )
    parser_batch.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show per-file processing status",
    )
    parser_batch.set_defaults(func=cmd_extract_batch, recursive=True)

    # validate command
    parser_validate = subparsers.add_parser(
        "validate",
        help="Validate a provenance sidecar file",
    )
    parser_validate.add_argument(
        "sidecar_path",
        help="Path to sidecar JSON file",
    )
    strict_group = parser_validate.add_mutually_exclusive_group()
    strict_group.add_argument(
        "--strict",
        action="store_true",
        dest="strict",
        help="Enable strict mode (default, fail on unknown fields)",
    )
    strict_group.add_argument(
        "--no-strict",
        action="store_false",
        dest="strict",
        help="Disable strict mode (allow unknown fields)",
    )
    parser_validate.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show sidecar summary after validation",
    )
    parser_validate.set_defaults(func=cmd_validate, strict=True)

    # summarize command
    parser_summary = subparsers.add_parser(
        "summarize",
        help="Summarize metadata from multiple sidecar files",
    )
    parser_summary.add_argument(
        "sidecar_dir",
        help="Directory containing sidecar JSON files",
    )
    parser_summary.set_defaults(func=cmd_summarize)

    args = parser.parse_args()

    if (args.json_pretty or args.json_output) and not args.json:
        parser.error("--json-pretty and --json-output require --json")

    if not args.command:
        parser.print_help()
        return EXIT_SUCCESS

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
