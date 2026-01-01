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

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = "lux-depth-v3.enhance.v1"


@dataclass
class InputMetadata:
    """Input image metadata."""
    image_path: str
    image_sha256: str


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
    scaling: Dict[str, float]  # {"method": "p1p99", "p1": ..., "p99": ...}
    runtime_ms: float


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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def write(self, path: Path) -> None:
        """Write manifest to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.to_json())
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

        return cls(
            schema=data.get("schema", MANIFEST_SCHEMA_VERSION),
            input=input_data,
            depth=depth_data,
            v2=v2_data,
            timing=timing_data,
            repro=repro_data,
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
        with open(path, 'r') as f:
            return cls.from_json(f.read())


def compute_file_sha256(path: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_git_revision(repo_path: Path) -> Optional[str]:
    """Get current git revision for reproducibility."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as exc:
        # Best-effort: failure to resolve git revision is non-fatal; return None.
        logger.debug("Unable to determine git revision for %s: %s", repo_path, exc)
    return None
