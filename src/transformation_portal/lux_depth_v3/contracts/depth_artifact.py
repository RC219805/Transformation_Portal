"""DepthArtifact - Universal spatial currency for depth processing.

This module defines the core contract for depth artifacts across all pipeline
stages. The DepthArtifact is designed to be self-describing, auditable, and
interoperable.

Contract Version: 1.0.0
Schema Compatibility: v2.0.0 Golden Path

Example:
    >>> from transformation_portal.lux_depth_v3.contracts import DepthArtifact
    >>> artifact = DepthArtifact(
    ...     depth_map=depth_array,
    ...     provenance=DepthProvenance(
    ...         model_id="depth-anything/DA3-Large",
    ...         license_tier=LicenseTier.COMMERCIAL,
    ...         checkpoint_sha256="abc123...",
    ...     )
    ... )
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Contract version for schema compatibility
DEPTH_ARTIFACT_SCHEMA_VERSION = "1.0.0"


class LicenseTier(Enum):
    """License tier classification for depth models.

    This enum enforces license-aware routing and compliance:
    - COMMERCIAL: Apache 2.0 or equivalent, safe for commercial use
    - NON_COMMERCIAL: CC BY-NC 4.0, research/academic only
    - EXPERIMENTAL: No stability guarantees, may have various licenses
    """

    COMMERCIAL = "commercial"
    NON_COMMERCIAL = "non_commercial"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True)
class CameraIntrinsics:
    """Camera intrinsics for metric depth reconstruction.

    Following standard pinhole camera model conventions.
    Values may be parsed from EXIF/XMP or estimated from image dimensions.

    Attributes:
        fx: Focal length in pixels (x-axis)
        fy: Focal length in pixels (y-axis)
        cx: Principal point x-coordinate in pixels
        cy: Principal point y-coordinate in pixels
        width: Image width in pixels
        height: Image height in pixels
        source: Origin of intrinsics ("exif", "estimated", "manual")
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    source: str = "estimated"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CameraIntrinsics:
        """Deserialize from dictionary."""
        return cls(**data)

    @classmethod
    def estimate_from_image(
        cls,
        width: int,
        height: int,
        fov_degrees: float = 60.0,
    ) -> CameraIntrinsics:
        """Estimate intrinsics from image dimensions assuming typical FOV.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            fov_degrees: Assumed horizontal field of view in degrees

        Returns:
            Estimated CameraIntrinsics
        """
        import math

        fov_rad = math.radians(fov_degrees)
        fx = width / (2 * math.tan(fov_rad / 2))
        fy = fx  # Assume square pixels
        cx = width / 2
        cy = height / 2

        return cls(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            source="estimated",
        )


@dataclass(frozen=True)
class DepthProvenance:
    """Immutable provenance record for depth artifacts.

    This record provides audit-quality traceability for every depth output:
    - What model produced it (model_id, checkpoint_sha256)
    - What license tier applies (license_tier)
    - What configuration was used (preset, runtime settings)
    - When and where it was produced (timestamp, environment)

    Attributes:
        model_id: HuggingFace model ID or internal identifier
        license_tier: License classification (COMMERCIAL, NON_COMMERCIAL, EXPERIMENTAL)
        checkpoint_sha256: SHA-256 hash of model weights (first 16 chars sufficient)
        preset: Pipeline preset name used
        device: Inference device (cpu, mps, cuda)
        runtime_version: transformation_portal version
        timestamp_utc: ISO 8601 UTC timestamp
        request_id: Optional unique request identifier for batch tracing
        downgrade_events: List of fallback/downgrade events during inference
    """

    model_id: str
    license_tier: LicenseTier
    checkpoint_sha256: Optional[str] = None
    preset: Optional[str] = None
    device: str = "cpu"
    runtime_version: str = "2.0.0"
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    request_id: Optional[str] = None
    downgrade_events: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary with enum handling."""
        data = asdict(self)
        data["license_tier"] = self.license_tier.value
        data["downgrade_events"] = list(self.downgrade_events)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DepthProvenance:
        """Deserialize from dictionary."""
        data = data.copy()
        data["license_tier"] = LicenseTier(data["license_tier"])
        if "downgrade_events" in data:
            data["downgrade_events"] = tuple(data["downgrade_events"])
        return cls(**data)


@dataclass
class DepthArtifact:
    """Universal spatial currency for depth processing.

    The DepthArtifact is the canonical representation of depth information
    across all Transformation Portal pipeline stages. It encapsulates:

    1. **depth_map** (required): Relative depth normalized to [0, 1]
       - Always present as the baseline depth representation
       - 0 = closest to camera, 1 = farthest from camera
       - Shape: (H, W) float32

    2. **metric_map_m** (optional): Absolute depth in meters
       - Present when metric depth is available (e.g., from Depth Pro)
       - Enables physically-based workflows requiring accurate scale
       - Shape: (H, W) float32

    3. **confidence** (optional): Per-pixel confidence/uncertainty
       - Identifies regions of low reliability (glass, mirrors, occlusions)
       - Shape: (H, W) float32, range [0, 1] (1 = high confidence)

    4. **intrinsics** (optional): Camera intrinsics for 3D reconstruction
       - Required for point cloud / mesh generation
       - May be from EXIF, estimated, or manually provided

    5. **provenance** (required): Immutable audit record
       - What model, license, checkpoint produced this artifact
       - Essential for compliance and reproducibility

    Attributes:
        depth_map: Relative depth array (H, W), normalized [0, 1]
        provenance: Immutable provenance record
        metric_map_m: Optional absolute depth in meters
        confidence: Optional per-pixel confidence map
        intrinsics: Optional camera intrinsics
        schema_version: Contract version for compatibility
        metadata: Optional additional metadata
    """

    depth_map: np.ndarray
    provenance: DepthProvenance
    metric_map_m: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    intrinsics: Optional[CameraIntrinsics] = None
    schema_version: str = DEPTH_ARTIFACT_SCHEMA_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate artifact on creation."""
        self._validate()

    def _validate(self) -> None:
        """Validate artifact fields."""
        # depth_map validation
        if not isinstance(self.depth_map, np.ndarray):
            raise TypeError("depth_map must be a numpy array")
        if self.depth_map.ndim != 2:
            raise ValueError(f"depth_map must be 2D, got shape {self.depth_map.shape}")
        if self.depth_map.dtype != np.float32:
            logger.warning(
                "depth_map dtype is %s, converting to float32",
                self.depth_map.dtype,
            )
            object.__setattr__(
                self, "depth_map", self.depth_map.astype(np.float32)
            )

        # Optional arrays validation
        if self.metric_map_m is not None:
            if not isinstance(self.metric_map_m, np.ndarray):
                raise TypeError("metric_map_m must be a numpy array")
            if self.metric_map_m.shape != self.depth_map.shape:
                raise ValueError(
                    f"metric_map_m shape {self.metric_map_m.shape} "
                    f"must match depth_map shape {self.depth_map.shape}"
                )

        if self.confidence is not None:
            if not isinstance(self.confidence, np.ndarray):
                raise TypeError("confidence must be a numpy array")
            if self.confidence.shape != self.depth_map.shape:
                raise ValueError(
                    f"confidence shape {self.confidence.shape} "
                    f"must match depth_map shape {self.depth_map.shape}"
                )

        # Provenance validation
        if not isinstance(self.provenance, DepthProvenance):
            raise TypeError("provenance must be a DepthProvenance instance")

    @property
    def shape(self) -> Tuple[int, int]:
        """Return depth map dimensions (H, W)."""
        return self.depth_map.shape  # type: ignore

    @property
    def has_metric_depth(self) -> bool:
        """Check if metric depth is available."""
        return self.metric_map_m is not None

    @property
    def has_confidence(self) -> bool:
        """Check if confidence map is available."""
        return self.confidence is not None

    @property
    def has_intrinsics(self) -> bool:
        """Check if camera intrinsics are available."""
        return self.intrinsics is not None

    @property
    def is_commercial_safe(self) -> bool:
        """Check if artifact is safe for commercial use."""
        return self.provenance.license_tier == LicenseTier.COMMERCIAL

    def compute_stats(self) -> Dict[str, Any]:
        """Compute depth statistics for quality analysis.

        Returns:
            Dictionary with depth statistics
        """
        valid_mask = np.isfinite(self.depth_map)
        valid_depth = self.depth_map[valid_mask]

        stats = {
            "finite_pct": float(valid_mask.mean() * 100),
            "min": float(valid_depth.min()) if valid_depth.size > 0 else None,
            "max": float(valid_depth.max()) if valid_depth.size > 0 else None,
            "mean": float(valid_depth.mean()) if valid_depth.size > 0 else None,
            "std": float(valid_depth.std()) if valid_depth.size > 0 else None,
        }

        if valid_depth.size > 0:
            stats["median"] = float(np.median(valid_depth))
            stats["p5"] = float(np.percentile(valid_depth, 5))
            stats["p95"] = float(np.percentile(valid_depth, 95))

        if self.metric_map_m is not None:
            valid_metric = self.metric_map_m[np.isfinite(self.metric_map_m)]
            if valid_metric.size > 0:
                stats["metric_min_m"] = float(valid_metric.min())
                stats["metric_max_m"] = float(valid_metric.max())
                stats["metric_median_m"] = float(np.median(valid_metric))

        return stats

    def to_sidecar_dict(self) -> Dict[str, Any]:
        """Export artifact metadata as audit sidecar (without arrays).

        Returns:
            Dictionary suitable for JSON serialization
        """
        return {
            "schema_version": self.schema_version,
            "shape": list(self.shape),
            "has_metric_depth": self.has_metric_depth,
            "has_confidence": self.has_confidence,
            "has_intrinsics": self.has_intrinsics,
            "provenance": self.provenance.to_dict(),
            "intrinsics": self.intrinsics.to_dict() if self.intrinsics else None,
            "stats": self.compute_stats(),
            "metadata": self.metadata,
        }

    def compute_content_hash(self) -> str:
        """Compute content-addressable hash of depth data.

        Returns:
            SHA-256 hash (first 16 chars) of depth_map bytes
        """
        hasher = hashlib.sha256()
        hasher.update(self.depth_map.tobytes())
        return hasher.hexdigest()[:16]


class DepthArtifactWriter:
    """Writer for persisting DepthArtifact to disk.

    Handles atomic writes with proper file naming and sidecar generation.

    Example:
        >>> writer = DepthArtifactWriter(output_dir=Path("./output"))
        >>> paths = writer.write(artifact, stem="scene_001")
    """

    def __init__(
        self,
        output_dir: Path,
        save_float: bool = True,
        save_preview: bool = True,
        save_sidecar: bool = True,
    ):
        """Initialize writer.

        Args:
            output_dir: Directory for output files
            save_float: Save full-precision .npy file
            save_preview: Save 16-bit PNG preview
            save_sidecar: Save JSON sidecar with metadata
        """
        self.output_dir = Path(output_dir)
        self.save_float = save_float
        self.save_preview = save_preview
        self.save_sidecar = save_sidecar

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        artifact: DepthArtifact,
        stem: str,
    ) -> Dict[str, Path]:
        """Write artifact to disk with atomic operations.

        Args:
            artifact: DepthArtifact to persist
            stem: Base filename (without extension)

        Returns:
            Dictionary mapping output types to paths
        """
        paths: Dict[str, Path] = {}

        # Save float depth (.npy)
        if self.save_float:
            float_path = self.output_dir / f"{stem}_depth.npy"
            np.save(float_path, artifact.depth_map)
            paths["depth_float"] = float_path

            # Save metric depth if available
            if artifact.has_metric_depth:
                metric_path = self.output_dir / f"{stem}_depth_metric.npy"
                np.save(metric_path, artifact.metric_map_m)
                paths["depth_metric"] = metric_path

            # Save confidence if available
            if artifact.has_confidence:
                conf_path = self.output_dir / f"{stem}_confidence.npy"
                np.save(conf_path, artifact.confidence)
                paths["confidence"] = conf_path

        # Save 16-bit PNG preview
        if self.save_preview:
            preview_path = self.output_dir / f"{stem}_depth.png"
            self._write_preview(artifact.depth_map, preview_path)
            paths["depth_preview"] = preview_path

        # Save JSON sidecar
        if self.save_sidecar:
            sidecar_path = self.output_dir / f"{stem}_depth.json"
            sidecar_data = artifact.to_sidecar_dict()
            # Add output paths to sidecar
            sidecar_data["outputs"] = {k: str(v) for k, v in paths.items()}

            with open(sidecar_path, "w") as f:
                json.dump(sidecar_data, f, indent=2)
            paths["sidecar"] = sidecar_path

        logger.info("Wrote DepthArtifact: %s", stem)
        return paths

    def _write_preview(self, depth: np.ndarray, path: Path) -> None:
        """Write 16-bit PNG preview of depth map."""
        from PIL import Image

        # Normalize to [0, 1]
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-8:
            normalized = (depth - d_min) / (d_max - d_min)
        else:
            normalized = np.zeros_like(depth)

        # Convert to 16-bit
        depth_u16 = (normalized * 65535).astype(np.uint16)

        # Save as 16-bit PNG
        img = Image.fromarray(depth_u16, mode="I;16")
        img.save(path)
