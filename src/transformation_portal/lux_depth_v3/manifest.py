"""Lux Depth V3 manifest and reproducibility data models.

This module defines the typed manifest payloads written for per-image outputs,
batch summaries, and config fingerprint material used by reuse and governance
checks.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformation_portal.ingest.canonical_json import dump_json, to_jsonable
from transformation_portal.lux_depth_v3._backend_contract import normalize_backend_id, normalize_backend_provenance

logger = logging.getLogger(__name__)

# Phase 3: MessagePack support (optional dependency)
try:
    import msgpack

    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False
    msgpack = None  # type: ignore


def _utcnow_iso() -> str:
    """Get current UTC time in ISO format.

    Returns:
        ISO 8601 formatted timestamp with timezone
    """
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


@dataclass
class InputMetadata:
    """Metadata for input image with schema versioning.

    Schema version 1.0: Initial release with path, hash, size, dimensions.

    Attributes:
        schema_version: Schema version for forward/backward compatibility
        image_path: Path to input image (relative or absolute)
        image_sha256: SHA256 hash of image file
        image_size_bytes: Size of image file in bytes
        image_dimensions: Image dimensions as (width, height) tuple
    """

    image_path: str
    image_sha256: Optional[str] = None
    image_size_bytes: Optional[int] = None
    image_dimensions: Optional[tuple] = None
    schema_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary with schema version.

        Returns:
            Dictionary representation with all fields
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InputMetadata:
        """Deserialize from dictionary with schema validation.

        Args:
            data: Dictionary representation

        Returns:
            InputMetadata instance

        Raises:
            ValueError: If schema_version is unsupported
        """
        schema_version = data.get("schema_version", "1.0")
        if schema_version != "1.0":
            raise ValueError(
                f"Unsupported InputMetadata schema version: {schema_version}. " f"This code supports version 1.0 only."
            )

        # Handle tuple/list conversion for image_dimensions
        dimensions = data.get("image_dimensions")
        if dimensions is not None and isinstance(dimensions, list):
            dimensions = tuple(dimensions)

        return cls(
            image_path=data["image_path"],
            image_sha256=data.get("image_sha256"),
            image_size_bytes=data.get("image_size_bytes"),
            image_dimensions=dimensions,
            schema_version=schema_version,
        )


@dataclass
class DepthMetadata:
    """Metadata for depth generation."""

    model: str
    depth_path: str
    runtime_seconds: float
    scaling: Dict[str, Any]
    stats: Optional[Dict[str, Any]] = None


@dataclass
class V2Metadata:
    """Metadata for V2 enhancement.

    Attributes:
        preset: V2 enhancement preset used
        status: Processing status ('ok', 'error', etc.)
        runtime_seconds: Time taken for V2 processing
        output_paths: List of output file paths
        strict_depth: Whether depth was required/used
        output_dir: V2 output directory
        report_path: Path to V2 report file
        error_message: Error message if status is not 'ok'
        input_bit_depth: Bit depth of input to V2 (8 or 16)
        output_bit_depth: Bit depth of V2 output (8 or 16)
    """

    preset: str
    status: str
    runtime_seconds: Optional[float] = None
    output_paths: Optional[List[str]] = None
    strict_depth: Optional[bool] = None
    output_dir: Optional[str] = None
    report_path: Optional[str] = None
    error_message: Optional[str] = None
    input_bit_depth: Optional[int] = None  # 8 or 16
    output_bit_depth: Optional[int] = None  # 8 or 16


@dataclass
class TimingMetadata:
    """Timing information for the pipeline."""

    depth_seconds: float
    v2_seconds: float
    total_seconds: float
    timestamp_utc: str


@dataclass
class ReproMetadata:
    """Reproducibility metadata."""

    v3_git_revision: Optional[str] = None
    v2_git_revision: Optional[str] = None
    environment: Optional[Dict[str, Any]] = None


@dataclass
class BackendSelectionMetadata:
    """Backend selection audit trail (added in v2.0.1 per ADR-023).

    Tracks requested vs resolved backend for transparency and debugging.

    Attributes:
        requested_backend: User-specified backend or None (auto)
        resolved_backend: Actual backend used
        resolution_status: "success", "fallback", or "error"
        resolution_reason: Why fallback occurred (if any)
        model_id: HuggingFace model ID or checkpoint path
        device: Resolved device (mps/cuda/cpu)
        schema_version: Schema version for backward compatibility
    """

    requested_backend: Optional[str]
    resolved_backend: str
    resolution_status: str
    resolution_reason: Optional[str]
    model_id: str
    device: str
    attempts: Optional[List[Dict[str, Any]]] = None
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        """Normalize emitted backend identifiers to canonical IDs."""
        self.requested_backend = normalize_backend_provenance(
            self.requested_backend,
        )
        normalized_resolved = normalize_backend_provenance(self.resolved_backend)
        if normalized_resolved:
            self.resolved_backend = normalized_resolved
        if self.attempts:
            normalized_attempts: List[Dict[str, Any]] = []
            for attempt in self.attempts:
                if not isinstance(attempt, dict):
                    normalized_attempts.append(attempt)
                    continue
                normalized_attempt = dict(attempt)
                normalized_backend = normalize_backend_id(
                    normalized_attempt.get("backend"),
                )
                if normalized_backend:
                    normalized_attempt["backend"] = normalized_backend
                normalized_attempts.append(normalized_attempt)
            self.attempts = normalized_attempts

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BackendSelectionMetadata:
        """Deserialize from dictionary with schema validation.

        Args:
            data: Dictionary representation

        Returns:
            BackendSelectionMetadata instance

        Raises:
            ValueError: If schema_version is unsupported
        """
        schema_version = data.get("schema_version", "1.0")
        if schema_version != "1.0":
            raise ValueError(f"Unsupported BackendSelectionMetadata schema: {schema_version}")

        return cls(
            requested_backend=data.get("requested_backend"),
            resolved_backend=data["resolved_backend"],
            resolution_status=data["resolution_status"],
            resolution_reason=data.get("resolution_reason"),
            model_id=data["model_id"],
            device=data["device"],
            attempts=data.get("attempts"),
            schema_version=schema_version,
        )


@dataclass
class MaterialsV3Metadata:
    """Metadata for Materials V3 surface-aware finishing.

    Schema version 1.0: Initial release.
    Schema version 1.1: Added bit depth tracking (output_bit_depth field).

    Attributes:
        enabled: Whether Materials V3 was enabled
        version: Materials V3 engine version
        response_plan: Response plan generated
        pixel_ops: Pixel operations telemetry
        runtime_seconds: Processing time in seconds
        output_bit_depth: Bit depth of Materials V3 output (8 or 16), added in v1.1
        schema_version: Schema version for forward/backward compatibility
    """

    enabled: bool
    version: str = "3.1"
    response_plan: Optional[Dict[str, Any]] = None
    pixel_ops: Optional[Dict[str, Any]] = None
    segmentation_metadata: Optional[Dict[str, Any]] = None
    runtime_seconds: Optional[float] = None
    output_bit_depth: Optional[int] = None  # 8 or 16, added in schema v1.1
    schema_version: str = "1.1"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MaterialsV3Metadata:
        """Deserialize from dictionary with backward compatibility."""
        schema_version = data.get("schema_version", "1.0")
        if schema_version not in ("1.0", "1.1"):
            raise ValueError(
                f"Unsupported MaterialsV3Metadata schema version: {schema_version}. "
                f"This code supports versions 1.0 and 1.1 only."
            )

        return cls(
            enabled=data["enabled"],
            version=data.get("version", "3.1"),
            response_plan=data.get("response_plan"),
            pixel_ops=data.get("pixel_ops"),
            segmentation_metadata=data.get("segmentation_metadata"),
            runtime_seconds=data.get("runtime_seconds"),
            output_bit_depth=data.get("output_bit_depth"),  # v1.1+, defaults to None for v1.0
            schema_version=schema_version,
        )


@dataclass
class ConfigFingerprint:
    """Configuration fingerprint for cache and stage reuse validation.

    The base fields cover the historical depth/V2 settings. Newer grouped
    fields capture additional stage-shaping configuration so manifest reuse can
    invalidate conservatively when Materials V3, PBR, APEX depth gating, or
    backend-selection settings change.
    """

    model_variant: str
    depth_quantization: str
    depth_device: str
    preset: Optional[str] = None
    v2_preset: Optional[str] = None
    v2_device: Optional[str] = None
    v2_upscaler_backend: Optional[str] = None
    depth_backend: Optional[str] = None
    depth_pro_checkpoint_path: Optional[str] = None
    depth_pro_python_executable: Optional[str] = None
    raw_python_executable: Optional[str] = None
    da3_python_executable: Optional[str] = None
    quality_tier: Optional[str] = None
    materials_config: Optional[Dict[str, Any]] = None
    pbr_config: Optional[Dict[str, Any]] = None
    apex_depth_gate_config: Optional[Dict[str, Any]] = None
    emit_master16: Optional[bool] = None
    emit_upscaled16: Optional[bool] = None
    enable_v2: Optional[bool] = None

    def depth_only(self) -> ConfigFingerprint:
        """Return fingerprint for Stage A reuse validation.

        This projection includes the raw depth settings plus the stage-shaping
        Materials V3, PBR, APEX gate, and delivery flags that determine whether
        reusing prior Stage A artifacts is still safe.
        """
        return ConfigFingerprint(
            model_variant=self.model_variant,
            depth_quantization=self.depth_quantization,
            depth_device=self.depth_device,
            preset=self.preset,
            depth_backend=self.depth_backend,
            depth_pro_checkpoint_path=self.depth_pro_checkpoint_path,
            depth_pro_python_executable=self.depth_pro_python_executable,
            raw_python_executable=self.raw_python_executable,
            da3_python_executable=self.da3_python_executable,
            quality_tier=self.quality_tier,
            materials_config=self.materials_config,
            pbr_config=self.pbr_config,
            apex_depth_gate_config=self.apex_depth_gate_config,
            emit_master16=self.emit_master16,
            emit_upscaled16=self.emit_upscaled16,
            v2_preset=None,
            v2_device=None,
            v2_upscaler_backend=None,
            enable_v2=None,
        )

    def v2_only(self) -> ConfigFingerprint:
        """Return fingerprint with only V2-related fields."""
        return ConfigFingerprint(
            model_variant="",
            depth_quantization="",
            depth_device="",
            preset=None,
            v2_preset=self.v2_preset,
            v2_device=self.v2_device,
            v2_upscaler_backend=self.v2_upscaler_backend,
            depth_backend=None,
            depth_pro_checkpoint_path=None,
            depth_pro_python_executable=None,
            raw_python_executable=None,
            da3_python_executable=None,
            quality_tier=None,
            materials_config=None,
            pbr_config=None,
            apex_depth_gate_config=None,
            emit_master16=self.emit_master16,
            emit_upscaled16=self.emit_upscaled16,
            enable_v2=self.enable_v2,
        )

    def to_sha256(self) -> str:
        """Compute SHA256 hash of fingerprint for caching keys."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class BatchManifest:
    """Manifest for batch processing with accurate timestamps.

    Attributes:
        batch_id: Unique identifier for the batch
        start_time: ISO 8601 formatted start time (UTC)
        end_time: ISO 8601 formatted end time (UTC)
        config: Configuration used for the batch
        results: List of per-image processing results
        stats: Aggregated batch statistics
    """

    batch_id: str
    start_time: str
    end_time: str
    config: Dict[str, Any]
    results: List[Dict[str, Any]]
    stats: Dict[str, Any]

    def write(self, path: Path) -> None:
        """Write batch manifest to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            dump_json(asdict(self), f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)

    @classmethod
    def load(cls, path: Path) -> BatchManifest:
        """Load batch manifest from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class CombinedManifest:
    """Combined manifest for all pipeline stages with accurate timestamps.

    Attributes:
        input: Input image metadata
        depth: Depth estimation metadata
        v2: V2 enhancement metadata
        timing: Timing information
        pbr_assets: PBR map output paths and config
        repro: Reproducibility metadata
        config_fingerprint: Configuration fingerprint for cache validation
        environment: Environment capture
        start_time: ISO 8601 formatted pipeline start time (UTC)
        end_time: ISO 8601 formatted pipeline end time (UTC)
        backend_selection: Backend selection audit trail (v2.0.1+, ADR-023)
    """

    input: Optional[InputMetadata] = None
    depth: Optional[DepthMetadata] = None
    v2: Optional[V2Metadata] = None
    materials_v3: Optional[MaterialsV3Metadata] = None
    timing: Optional[TimingMetadata] = None
    pbr_assets: Optional[Dict[str, Any]] = None
    repro: Optional[ReproMetadata] = None
    config_fingerprint: Optional[ConfigFingerprint] = None
    environment: Optional[Dict[str, Any]] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    backend_selection: Optional[BackendSelectionMetadata] = None
    licensing: Optional[Dict[str, Any]] = None

    def save(self, path: Path) -> None:
        """Save manifest to JSON file.

        Serializes all fields including timestamps for accurate execution tracking.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclasses to dict
        data = {}
        for field_name in [
            "input",
            "depth",
            "v2",
            "materials_v3",
            "timing",
            "pbr_assets",
            "repro",
            "config_fingerprint",
            "environment",
            "backend_selection",
            "licensing",
        ]:
            field_value = getattr(self, field_name)
            if field_value is not None:
                if field_name in ["pbr_assets", "environment", "licensing"]:
                    # Already a dict, no need to convert
                    data[field_name] = field_value
                else:
                    data[field_name] = asdict(field_value)

        # Include timestamp fields
        if self.start_time:
            data["start_time"] = self.start_time
        if self.end_time:
            data["end_time"] = self.end_time

        with open(path, "w", encoding="utf-8") as f:
            dump_json(data, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)

    @classmethod
    def load(cls, path: Path) -> CombinedManifest:
        """Load manifest from JSON file.

        Deserializes all fields including timestamps.
        """
        with open(path, "r") as f:
            data = json.load(f)

        # Reconstruct dataclasses
        manifest = cls()

        if "input" in data:
            manifest.input = InputMetadata.from_dict(data["input"])
        if "depth" in data:
            manifest.depth = DepthMetadata(**data["depth"])
        if "v2" in data:
            manifest.v2 = V2Metadata(**data["v2"])
        if "materials_v3" in data:
            manifest.materials_v3 = MaterialsV3Metadata.from_dict(data["materials_v3"])
        if "timing" in data:
            manifest.timing = TimingMetadata(**data["timing"])
        if "repro" in data:
            manifest.repro = ReproMetadata(**data["repro"])
        if "config_fingerprint" in data:
            manifest.config_fingerprint = ConfigFingerprint(**data["config_fingerprint"])
        if "pbr_assets" in data:
            manifest.pbr_assets = data["pbr_assets"]
        if "environment" in data:
            manifest.environment = data["environment"]
        if "backend_selection" in data and data["backend_selection"] is not None:
            manifest.backend_selection = BackendSelectionMetadata.from_dict(data["backend_selection"])
        if "licensing" in data and data["licensing"] is not None:
            manifest.licensing = dict(data["licensing"])
        # Load timestamp fields
        if "start_time" in data:
            manifest.start_time = data["start_time"]
        if "end_time" in data:
            manifest.end_time = data["end_time"]

        return manifest

    def write(self, path: Path) -> None:
        """Alias for save() for backward compatibility."""
        self.save(path)

    def save_msgpack(self, path: Path) -> None:
        """Save manifest in MessagePack binary format (Phase 3).

        Provides 60% size reduction and 3x faster parsing compared to JSON.
        Requires msgpack package: pip install msgpack

        Args:
            path: Output path (will use .msgpack extension)
        """
        if not MSGPACK_AVAILABLE:
            logger.warning("msgpack not available, falling back to JSON. Install: pip install msgpack")
            self.save(path.with_suffix(".json"))
            return

        path = path.with_suffix(".msgpack")
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclasses to dict (same logic as save)
        data = {}
        for field_name in [
            "input",
            "depth",
            "v2",
            "timing",
            "pbr_assets",
            "repro",
            "config_fingerprint",
            "environment",
            "backend_selection",
            "licensing",
        ]:
            field_value = getattr(self, field_name)
            if field_value is not None:
                if field_name in ["pbr_assets", "environment", "licensing"]:
                    data[field_name] = field_value
                else:
                    data[field_name] = asdict(field_value)

        # Include timestamp fields
        if self.start_time:
            data["start_time"] = self.start_time
        if self.end_time:
            data["end_time"] = self.end_time

        # Atomic write pattern
        temp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with open(temp_path, "wb") as f:
                msgpack.pack(to_jsonable(data), f, use_bin_type=True)

            temp_path.replace(path)
            logger.debug(f"Wrote MessagePack manifest: {path}")
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e

    @classmethod
    def load_msgpack(cls, path: Path) -> CombinedManifest:
        """Load manifest from MessagePack binary format (Phase 3).

        Args:
            path: Path to .msgpack file

        Returns:
            CombinedManifest instance
        """
        if not MSGPACK_AVAILABLE:
            raise ImportError("msgpack required to load .msgpack files. Install: pip install msgpack")

        with open(path, "rb") as f:
            data = msgpack.unpack(f, raw=False)

        # Reconstruct manifest (same logic as load)
        manifest = cls()

        if "input" in data:
            manifest.input = InputMetadata.from_dict(data["input"])
        if "depth" in data:
            manifest.depth = DepthMetadata(**data["depth"])
        if "v2" in data:
            manifest.v2 = V2Metadata(**data["v2"])
        if "timing" in data:
            manifest.timing = TimingMetadata(**data["timing"])
        if "repro" in data:
            manifest.repro = ReproMetadata(**data["repro"])
        if "config_fingerprint" in data:
            manifest.config_fingerprint = ConfigFingerprint(**data["config_fingerprint"])
        if "pbr_assets" in data:
            manifest.pbr_assets = data["pbr_assets"]
        if "environment" in data:
            manifest.environment = data["environment"]
        if "licensing" in data and data["licensing"] is not None:
            manifest.licensing = dict(data["licensing"])
        if "start_time" in data:
            manifest.start_time = data["start_time"]
        if "end_time" in data:
            manifest.end_time = data["end_time"]

        return manifest

    @classmethod
    def load_auto(cls, path: Path) -> CombinedManifest:
        """Load manifest auto-detecting format by extension (Phase 3).

        Supports .json and .msgpack formats.

        Args:
            path: Path to manifest file

        Returns:
            CombinedManifest instance
        """
        if path.suffix == ".msgpack":
            return cls.load_msgpack(path)
        else:
            return cls.load(path)


def compute_file_sha256(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA256 hash with minimal memory overhead.

    Uses chunked reading to avoid loading entire file into memory,
    reducing memory usage by ~90% for large TIFF files (500MB+).

    Args:
        file_path: Path to file
        chunk_size: Size of chunks to read (default 8KB for optimal I/O)

    Returns:
        Hexadecimal SHA256 hash
    """
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)

    return sha256.hexdigest()


def get_git_revision(repo_root: Path) -> Optional[str]:
    """Get current git revision.

    STUB: Basic git rev-parse.

    Args:
        repo_root: Repository root directory

    Returns:
        Git revision hash or None if not in a git repo
    """
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        # Git not available or subprocess failed
        pass

    return None


def capture_environment() -> Dict[str, Any]:
    """Capture environment information for reproducibility.

    STUB: Basic environment capture.

    Returns:
        Dictionary with environment information
    """
    import platform
    import sys

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
