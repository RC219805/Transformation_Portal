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


@dataclass
class InputMetadata:
    """Metadata for input image."""
    image_path: str
    image_sha256: Optional[str] = None
    image_size_bytes: Optional[int] = None
    image_dimensions: Optional[tuple] = None


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
    """Metadata for V2 enhancement."""
    preset: str
    status: str
    runtime_seconds: Optional[float] = None
    output_paths: Optional[List[str]] = None


@dataclass
class TimingMetadata:
    """Timing information for the pipeline."""
    total_seconds: float
    depth_seconds: float
    v2_seconds: float
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


@dataclass
class BatchManifest:
    """Manifest for batch processing statistics."""
    total_images: int
    successful: int
    failed: int
    skipped_depth: int
    skipped_v2: int
    total_runtime_seconds: float


@dataclass
class CombinedManifest:
    """Combined manifest for all pipeline stages."""
    input: Optional[InputMetadata] = None
    depth: Optional[DepthMetadata] = None
    v2: Optional[V2Metadata] = None
    timing: Optional[TimingMetadata] = None
    repro: Optional[ReproMetadata] = None
    config_fingerprint: Optional[ConfigFingerprint] = None

    def save(self, path: Path):
        """Save manifest to JSON file.

        STUB: Basic JSON serialization.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclasses to dict
        data = {}
        for field_name in ['input', 'depth', 'v2', 'timing', 'repro', 'config_fingerprint']:
            field_value = getattr(self, field_name)
            if field_value is not None:
                data[field_name] = asdict(field_value)

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> CombinedManifest:
        """Load manifest from JSON file.

        STUB: Basic JSON deserialization.
        """
        with open(path, 'r') as f:
            data = json.load(f)

        # Reconstruct dataclasses
        manifest = cls()

        if 'input' in data:
            manifest.input = InputMetadata(**data['input'])
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

        return manifest


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
