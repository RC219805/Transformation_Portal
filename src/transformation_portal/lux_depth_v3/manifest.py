"""Manifest management for pipeline reproducibility.

STUB IMPLEMENTATION - Critical types to enable package imports.
Full implementation pending.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import hashlib
import subprocess
import datetime


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
        schema_version = data.get('schema_version', '1.0')
        if schema_version != '1.0':
            raise ValueError(
                f"Unsupported InputMetadata schema version: {schema_version}. "
                f"This code supports version 1.0 only."
            )

        # Handle tuple/list conversion for image_dimensions
        dimensions = data.get('image_dimensions')
        if dimensions is not None and isinstance(dimensions, list):
            dimensions = tuple(dimensions)

        return cls(
            image_path=data['image_path'],
            image_sha256=data.get('image_sha256'),
            image_size_bytes=data.get('image_size_bytes'),
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
    """
    preset: str
    status: str
    runtime_seconds: Optional[float] = None
    output_paths: Optional[List[str]] = None
    strict_depth: Optional[bool] = None
    output_dir: Optional[str] = None
    report_path: Optional[str] = None
    error_message: Optional[str] = None


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
class ConfigFingerprint:
    """Configuration fingerprint for caching validation."""
    model_variant: str
    depth_quantization: str
    depth_device: str
    preset: Optional[str] = None
    v2_preset: Optional[str] = None
    v2_device: Optional[str] = None
    v2_upscaler_backend: Optional[str] = None

    def depth_only(self) -> ConfigFingerprint:
        """Return fingerprint with only depth-related fields."""
        return ConfigFingerprint(
            model_variant=self.model_variant,
            depth_quantization=self.depth_quantization,
            depth_device=self.depth_device,
            preset=self.preset,
            v2_preset=None,
            v2_device=None,
            v2_upscaler_backend=None,
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

    def write(self, path: Path):
        """Write batch manifest to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> BatchManifest:
        """Load batch manifest from JSON file."""
        with open(path, 'r') as f:
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
    """
    input: Optional[InputMetadata] = None
    depth: Optional[DepthMetadata] = None
    v2: Optional[V2Metadata] = None
    timing: Optional[TimingMetadata] = None
    pbr_assets: Optional[Dict[str, Any]] = None
    repro: Optional[ReproMetadata] = None
    config_fingerprint: Optional[ConfigFingerprint] = None
    environment: Optional[Dict[str, Any]] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    def save(self, path: Path):
        """Save manifest to JSON file.

        Serializes all fields including timestamps for accurate execution tracking.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclasses to dict
        data = {}
        for field_name in ['input', 'depth', 'v2', 'timing', 'pbr_assets', 'repro', 'config_fingerprint', 'environment']:
            field_value = getattr(self, field_name)
            if field_value is not None:
                if field_name in ['pbr_assets', 'environment']:
                    # Already a dict, no need to convert
                    data[field_name] = field_value
                else:
                    data[field_name] = asdict(field_value)

        # Include timestamp fields
        if self.start_time:
            data['start_time'] = self.start_time
        if self.end_time:
            data['end_time'] = self.end_time

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> CombinedManifest:
        """Load manifest from JSON file.

        Deserializes all fields including timestamps.
        """
        with open(path, 'r') as f:
            data = json.load(f)

        # Reconstruct dataclasses
        manifest = cls()

        if 'input' in data:
            manifest.input = InputMetadata.from_dict(data['input'])
        if 'depth' in data:
            manifest.depth = DepthMetadata(**data['depth'])
        if 'v2' in data:
            manifest.v2 = V2Metadata(**data['v2'])
        if 'timing' in data:
            manifest.timing = TimingMetadata(**data['timing'])
        if 'repro' in data:
            manifest.repro = ReproMetadata(**data['repro'])
        if 'config_fingerprint' in data:
            manifest.config_fingerprint = ConfigFingerprint(**data['config_fingerprint'])
        if 'pbr_assets' in data:
            manifest.pbr_assets = data['pbr_assets']
        if 'environment' in data:
            manifest.environment = data['environment']
        # Load timestamp fields
        if 'start_time' in data:
            manifest.start_time = data['start_time']
        if 'end_time' in data:
            manifest.end_time = data['end_time']

        return manifest

    def write(self, path: Path):
        """Alias for save() for backward compatibility."""
        self.save(path)


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file.

    Args:
        file_path: Path to file

    Returns:
        Hexadecimal SHA256 hash
    """
    sha256 = hashlib.sha256()

    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
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
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
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
        'python_version': sys.version,
        'platform': platform.platform(),
        'machine': platform.machine(),
    }
