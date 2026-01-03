"""Combined manifest schema for V3 + V2 integration.

This module defines the manifest structure that links DA3 depth generation
with V2 enhancement outputs, providing full provenance and reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import hashlib
import subprocess
import sys
import logging
import os

from .security import validate_git_repository

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = "lux-depth-v3.enhance.v1"


@dataclass
class ConfigFingerprint:
    """Fingerprint of all configuration parameters that affect outputs.

    Used for cache validation: if config changes, outputs must be regenerated.
    """

    # Depth config
    model_variant: str
    depth_quantization: str
    depth_device: str
    preset: Optional[str]

    # V2 config
    v2_preset: str
    v2_device: str
    v2_upscaler_backend: str

    # Execution config (affects quality/timing but not visual output)
    # Omitted: execution_mode, force_depth, force_v2, timeout

    def to_sha256(self) -> str:
        """Compute deterministic SHA256 hash of config.

        Returns:
            64-character hex string (SHA256)
        """
        # Convert to dict, sort keys for determinism
        config_dict = {
            "model_variant": self.model_variant,
            "depth_quantization": self.depth_quantization,
            "depth_device": self.depth_device,
            "preset": self.preset or "",
            "v2_preset": self.v2_preset,
            "v2_device": self.v2_device,
            "v2_upscaler_backend": self.v2_upscaler_backend,
        }

        # JSON dump with sorted keys for reproducibility
        json_str = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))

        # SHA256 hash
        return hashlib.sha256(json_str.encode()).hexdigest()

    def depth_only(self) -> str:
        """Compute fingerprint of depth-only parameters.

        Returns:
            SHA256 hash of depth config subset
        """
        depth_config = {
            "model_variant": self.model_variant,
            "depth_quantization": self.depth_quantization,
            "depth_device": self.depth_device,
            "preset": self.preset or "",
        }
        json_str = json.dumps(depth_config, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode()).hexdigest()

    def v2_only(self) -> str:
        """Compute fingerprint of V2-only parameters.

        Returns:
            SHA256 hash of V2 config subset
        """
        v2_config = {
            "v2_preset": self.v2_preset,
            "v2_device": self.v2_device,
            "v2_upscaler_backend": self.v2_upscaler_backend,
        }
        json_str = json.dumps(v2_config, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode()).hexdigest()


@dataclass
class InputMetadata:
    """Input image metadata."""

    image_path: str
    image_sha256: str
    exif_normalized: bool = False  # True if EXIF orientation was normalized
    normalized_path: Optional[str] = None  # Path to normalized file if applicable


@dataclass
class DepthScalingMetadata:
    """Detailed depth quantization metadata for provenance and debugging."""

    method: str  # "p1p99", "p0.5p99.5", "minmax"
    p_low_percentile: float  # e.g., 1.0 for p1p99
    p_high_percentile: float  # e.g., 99.0 for p1p99
    v_low_value: float  # Actual depth value at p_low
    v_high_value: float  # Actual depth value at p_high
    clipped_low_frac: float  # Fraction of pixels clipped at low end
    clipped_high_frac: float  # Fraction of pixels clipped at high end
    invalid_frac: float  # Fraction of NaN/Inf pixels (pre-cleaning)


@dataclass
class DepthMetadata:
    """Depth generation metadata."""

    backend: str  # "da3"
    model: str  # e.g., "DepthAnything3-Large-Metric"
    license: str  # e.g., "CC-BY-NC"
    non_commercial_ok: bool
    depth_path: str  # Relative path: "depth/{stem}_depth.png"
    dtype: str  # "uint16"
    shape: List[int]  # [H, W]
    scaling: Dict[str, Any]  # Legacy dict or DepthScalingMetadata
    runtime_ms: float
    # NEW FIELDS for enhanced provenance
    representation: str = "depth"  # "depth" vs "inverse_depth" vs "disparity"
    convention: str = "higher_is_farther"  # vs "higher_is_nearer"
    unit: str = "relative"  # "relative" vs "metric_meters"


@dataclass
class V2Metadata:
    """V2 enhancement metadata."""

    preset: str
    strict_depth: bool
    output_dir: str  # "v2/"
    report_path: str  # "v2/{stem}_report.json"
    status: str  # "ok", "error", "skipped"
    error_message: Optional[str] = None


@dataclass
class TimingMetadata:
    """Timing breakdown."""

    depth_s: float
    v2_s: float
    total_s: float


@dataclass
class EnvironmentMetadata:
    """Toolchain and hardware environment for reproducibility."""

    python: str
    torch: Optional[str] = None
    cuda_runtime: Optional[str] = None
    gpu_name: Optional[str] = None
    driver: Optional[str] = None
    os_platform: Optional[str] = None


@dataclass
class ReproMetadata:
    """Reproducibility metadata."""

    v3_git: Optional[str] = None
    v2_git: Optional[str] = None
    python: str = field(default_factory=lambda: sys.version.split()[0])
    device: str = "cpu"


@dataclass
class CombinedManifest:
    """Combined manifest linking V3 depth and V2 enhancement."""

    schema: str = MANIFEST_SCHEMA_VERSION
    input: Optional[InputMetadata] = None
    depth: Optional[DepthMetadata] = None
    v2: Optional[V2Metadata] = None
    timing: Optional[TimingMetadata] = None
    repro: Optional[ReproMetadata] = None
    config_fingerprint: Optional[str] = None  # SHA256 of config
    environment: Optional[EnvironmentMetadata] = None  # NEW: Toolchain environment

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def write(self, path: Path) -> None:
        """Write manifest to JSON file atomically."""
        atomic_write_json(path, self.to_dict())
        logger.info(f"Wrote manifest to {path}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CombinedManifest:
        """Load from dictionary."""
        # Reconstruct nested dataclasses
        input_data = data.get("input")
        if input_data:
            input_data = InputMetadata(**input_data)

        depth_data = data.get("depth")
        if depth_data:
            depth_data = DepthMetadata(**depth_data)

        v2_data = data.get("v2")
        if v2_data:
            v2_data = V2Metadata(**v2_data)

        timing_data = data.get("timing")
        if timing_data:
            timing_data = TimingMetadata(**timing_data)

        repro_data = data.get("repro")
        if repro_data:
            repro_data = ReproMetadata(**repro_data)

        environment_data = data.get("environment")
        if environment_data:
            environment_data = EnvironmentMetadata(**environment_data)

        return cls(
            schema=data.get("schema", MANIFEST_SCHEMA_VERSION),
            input=input_data,
            depth=depth_data,
            v2=v2_data,
            timing=timing_data,
            repro=repro_data,
            config_fingerprint=data.get("config_fingerprint"),
            environment=environment_data,
        )

    @classmethod
    def from_json(cls, json_str: str) -> CombinedManifest:
        """Load from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: Path) -> CombinedManifest:
        """Load manifest from JSON file."""
        path = Path(path)
        with open(path, "r") as f:
            return cls.from_json(f.read())


def compute_file_sha256(path: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_git_revision(repo_path: Path) -> Optional[str]:
    """Get current git revision for reproducibility.

    Args:
        repo_path: Path to git repository

    Returns:
        Git commit SHA if available, None otherwise

    Security considerations:
        - Validates repository path before executing git
        - Uses explicit GIT_DIR to prevent malicious hook execution
        - Disables system-wide git config
        - Disables template directory to prevent hook injection
        - Sets timeout to prevent hanging
    """
    # Validate repository path
    validated_repo = validate_git_repository(repo_path)
    if not validated_repo:
        logger.debug(f"Not a git repository: {repo_path}")
        return None

    try:
        git_dir = validated_repo / ".git"
        # Secure git environment: start with parent env, override with security settings
        import os

        secure_env = os.environ.copy()
        secure_env.update(
            {
                "GIT_DIR": str(git_dir),
                "GIT_TEMPLATE_DIR": "",  # Disable templates
                "GIT_CONFIG_NOSYSTEM": "1",  # Disable system-wide config
            }
        )
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
            env=secure_env,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as exc:
        # Best-effort: failure to resolve git revision is non-fatal; return None.
        logger.debug("Unable to determine git revision for %s: %s", repo_path, exc)
    return None


def atomic_write_json(path: Path, data: Dict[str, Any], indent: int = 2) -> None:
    """Write JSON with atomic rename to prevent partial files on crash.

    Args:
        path: Final output path
        data: Dictionary to serialize
        indent: JSON indentation level

    Raises:
        IOError: If write fails

    Notes:
        - Temp file is written in same directory (same filesystem)
        - os.replace() provides atomic rename on POSIX systems
        - Cleanup is guaranteed via finally block
    """
    path = Path(path)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(".tmp.json")

    try:
        # Write to temp file
        tmp_path.write_text(json.dumps(data, indent=indent))

        # Atomic rename
        os.replace(str(tmp_path), str(path))

        logger.debug(f"Atomically wrote JSON to {path}")
    except Exception as e:
        # Clean up partial write
        if tmp_path.exists():
            try:
                tmp_path.unlink()
                logger.debug(f"Cleaned up partial write: {tmp_path}")
            except Exception as cleanup_error:
                logger.warning(f"Could not clean up {tmp_path}: {cleanup_error}")
        raise IOError(f"Failed to write JSON to {path}: {e}") from e


def capture_environment() -> EnvironmentMetadata:
    """Capture current toolchain and hardware environment for reproducibility.

    Returns:
        EnvironmentMetadata with Python version, torch, CUDA, GPU info

    Notes:
        - Best-effort capture: missing dependencies return None
        - GPU info only captured if torch with CUDA is available
    """
    import platform

    env = EnvironmentMetadata(
        python=sys.version.split()[0],
        os_platform=platform.system(),
    )

    try:
        import torch

        env.torch = torch.__version__

        if torch.cuda.is_available():
            env.cuda_runtime = torch.version.cuda
            try:
                env.gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                pass  # GPU name not critical

            try:
                # Get driver version if available
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.returncode == 0:
                    env.driver = result.stdout.strip()
            except Exception:
                pass  # Driver version not critical
    except ImportError:
        pass  # torch not available

    return env
