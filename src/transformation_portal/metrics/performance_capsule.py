"""Performance capsule schema for investor-grade tracking.

This module defines the comprehensive performance capture schema that enables
scene-dependent analysis, regression detection, and Quality Firewall enforcement.

Design principles:
- Capture everything needed to explain variance (scene content, dimensions, config)
- Make schema stable and versionable (contract tests required)
- Support efficient querying and bucketing
- Enable deterministic regression detection

Usage:
    from transformation_portal.metrics.performance_capsule import PerformanceCapsule

    capsule = PerformanceCapsule(
        image_id="750_Picacho_Pool",
        timings={"total": 11.49, "inference": 8.2, ...},
        scene_type="pool",
        ...
    )
    log_performance_capsule(capsule)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional, Tuple

__version__ = "2.0.0"


@dataclass
class PerformanceCapsule:
    """Per-image performance capsule for ledger tracking.

    This schema is contract-stable. Changes require version bump and migration plan.

    v2.0.0 changes:
    - Added workflow_version field (v1/v2 tracking)
    - Added zone field (multi-zone performance tracking)

    Attributes:
        image_id: Unique identifier for the image (e.g., filename stem)
        image_path: Absolute or relative path to input image
        input_hash: SHA256 of input image bytes (for cache validation)

        original_shape: (height, width) before dimension enforcement
        enforced_shape: (height, width) after dimension multiple enforcement
        pixel_count: Total pixels in enforced shape
        dimension_adjustment: Human-readable description (e.g., "cropped_0.2%")

        tile_config: Tiling configuration dict (if tiled inference used)
        tile_count: Number of tiles processed (None if not tiled)

        backend_id: Backend identifier (e.g., "da3", "depth_pro")
        model_variant: Model version/variant string
        device: Execution device ("mps", "cuda", "cpu")
        dtype: Data type used for inference (e.g., "float16", "float32")

        cache_hit: Whether depth was loaded from cache
        cache_key: Cache key used for lookup

        timings: Phase-level timing breakdown in seconds
            Required keys: "total"
            Optional keys: "load_decode", "preprocess", "inference", "postprocess",
                          "write_depth", "pbr_normals", "pbr_roughness", "pbr_ao"

        scene_type: Optional scene classification ("pool", "aerial", "interior", etc.)
        texture_complexity: Optional texture descriptor ("high_frequency", "smooth", "mixed")

        config_hash: SHA256 of exact configuration dict used
        pipeline_version: Pipeline version string (e.g., "2.0.0")
        workflow_version: Workflow version ("v1" or "v2") for V1/V2 comparison (v2.0.0+)
        zone: Deployment zone identifier (e.g., "us-west-2a", "local") (v2.0.0+)

        quality_score: Optional quality metric (e.g., edge coherence score)
        firewall_status: Quality Firewall verdict ("pass", "warn", "block")

        captured_at: ISO8601 timestamp of capture
        schema_version: Schema version for forward/backward compatibility
    """

    # Image Identity
    image_id: str
    image_path: str
    input_hash: str

    # Effective Dimensions
    original_shape: Tuple[int, int]
    enforced_shape: Tuple[int, int]
    pixel_count: int
    dimension_adjustment: str

    # Tiling Configuration
    tile_config: Optional[Dict[str, Any]] = None
    tile_count: Optional[int] = None

    # Backend Configuration
    backend_id: str = "unknown"
    model_variant: str = "unknown"
    device: str = "cpu"
    dtype: str = "float32"

    # Cache Behavior
    cache_hit: bool = False
    cache_key: str = ""

    # Phase Timings (CRITICAL)
    timings: Dict[str, float] = field(default_factory=dict)

    # Scene Characteristics (for bucketing)
    scene_type: Optional[str] = None
    texture_complexity: Optional[str] = None

    # Config Fingerprint
    config_hash: str = ""
    pipeline_version: str = "unknown"
    workflow_version: Literal["v1", "v2"] = "v1"  # v2.0.0: V1/V2 tracking
    zone: Optional[str] = None  # v2.0.0: multi-zone tracking

    # Quality Metrics
    quality_score: Optional[float] = None
    firewall_status: str = "unknown"

    # Metadata
    captured_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = __version__

    def __post_init__(self) -> None:
        """Validate required fields and compute derived fields."""
        if not self.image_id:
            raise ValueError("image_id is required")
        if not self.timings:
            raise ValueError("timings dict is required")
        if "total" not in self.timings:
            raise ValueError("timings must include 'total' key")
        if self.pixel_count <= 0:
            raise ValueError(f"pixel_count must be positive, got {self.pixel_count}")
        if self.timings["total"] < 0:
            raise ValueError(f"total timing must be non-negative, got {self.timings['total']}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PerformanceCapsule:
        """Reconstruct from dict (with schema migration support)."""
        schema_version = data.get("schema_version", "1.0.0")

        # Migration: 1.0.0 -> 2.0.0
        if schema_version == "1.0.0":
            # Add v2.0.0 fields with safe defaults
            data.setdefault("workflow_version", "v1")
            data.setdefault("zone", None)
            data["schema_version"] = "2.0.0"

        # Convert shape tuples (may be serialized as lists)
        if "original_shape" in data and isinstance(data["original_shape"], list):
            data["original_shape"] = tuple(data["original_shape"])
        if "enforced_shape" in data and isinstance(data["enforced_shape"], list):
            data["enforced_shape"] = tuple(data["enforced_shape"])

        return cls(**data)


@dataclass
class PerformanceBucket:
    """Performance bucket definition for scene-dependent thresholds.

    Attributes:
        name: Bucket identifier
        filters: Dict of filter conditions for bucket membership
        p50_threshold_sec: Median latency threshold (seconds)
        p95_threshold_sec: p95 latency threshold (seconds)
        p90_threshold_sec: Optional p90 latency threshold (seconds)
        p99_threshold_sec: Optional p99 latency threshold (seconds)
        description: Human-readable description
    """

    name: str
    filters: Dict[str, Any]
    p50_threshold_sec: float
    p95_threshold_sec: float
    p90_threshold_sec: Optional[float] = None
    p99_threshold_sec: Optional[float] = None
    description: str = ""

    def matches(self, capsule: PerformanceCapsule) -> bool:
        """Check if capsule matches this bucket's filters."""
        for key, value in self.filters.items():
            if key == "scene_type":
                if capsule.scene_type != value:
                    return False
            elif key == "workflow_version":
                if capsule.workflow_version != value:
                    return False
            elif key == "zone":
                if capsule.zone != value:
                    return False
            elif key == "pixel_count_min":
                if capsule.pixel_count < value:
                    return False
            elif key == "pixel_count_max":
                if capsule.pixel_count > value:
                    return False
            elif key == "backend_id":
                if capsule.backend_id != value:
                    return False
            elif key == "device":
                if capsule.device != value:
                    return False
            else:
                # Unknown filter key - conservative: don't match
                return False
        return True

    @property
    def specificity(self) -> int:
        """Compute specificity score for bucket ordering.

        Higher score = more specific.
        Uses concept-based scoring to prevent "fake specificity" from multi-key ranges.
        """
        return compute_specificity(self.filters)

    def check_threshold(self, percentile: int, value: float) -> bool:
        """Check if value passes threshold for given percentile.

        Args:
            percentile: Percentile to check (50, 90, 95, or 99)
            value: Value to check against threshold

        Returns:
            True if value passes threshold (or threshold not set)
        """
        threshold_attr = f"p{percentile}_threshold_sec"
        threshold = getattr(self, threshold_attr, None)
        if threshold is None:
            return True  # No threshold = pass
        return value <= threshold


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute deterministic SHA256 hash of config dict.

    Args:
        config: Configuration dictionary

    Returns:
        Hex-encoded SHA256 hash
    """
    # Sort keys for determinism
    json_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def compute_specificity(filters: Dict[str, Any]) -> int:
    """Compute specificity score based on concepts, not key count.

    Higher score = more specific.

    Scoring:
    - workflow_version: +10 (critical for V1/V2 comparison)
    - scene_type: +10 (primary discriminator)
    - zone: +5 (deployment topology boundary)
    - device: +5 (hardware-specific)
    - backend_id: +5 (model-specific)
    - pixel_count range: +3 (counts once even if min+max)
    - dimension_adjustment: +1 (secondary detail)

    This prevents "fake specificity" from multi-key ranges.

    Args:
        filters: Filter dictionary

    Returns:
        Specificity score (higher = more specific)
    """
    score = 0

    if "workflow_version" in filters:
        score += 10

    if "scene_type" in filters:
        score += 10

    if "zone" in filters:
        score += 5

    if "device" in filters:
        score += 5

    if "backend_id" in filters:
        score += 5

    # pixel_count range counts as ONE concept (not two)
    if "pixel_count_min" in filters or "pixel_count_max" in filters:
        score += 3

    if "dimension_adjustment" in filters:
        score += 1

    return score


def compute_dimension_adjustment(
    original: Tuple[int, int],
    enforced: Tuple[int, int]
) -> str:
    """Compute human-readable dimension adjustment description.

    Args:
        original: (height, width) before enforcement
        enforced: (height, width) after enforcement

    Returns:
        Human-readable string (e.g., "cropped_0.2%", "padded_1.5%", "exact")
    """
    if original == enforced:
        return "exact"

    orig_pixels = original[0] * original[1]
    enf_pixels = enforced[0] * enforced[1]

    if enf_pixels < orig_pixels:
        pct = 100.0 * (orig_pixels - enf_pixels) / orig_pixels
        return f"cropped_{pct:.1f}%"
    else:
        pct = 100.0 * (enf_pixels - orig_pixels) / orig_pixels
        return f"padded_{pct:.1f}%"


# Default bucket definitions (per APEX research workflow analysis)
DEFAULT_BUCKETS = [
    PerformanceBucket(
        name="aerial_large_mps",
        filters={
            "scene_type": "aerial",
            "pixel_count_min": 20_000_000,  # 6000×3600+
            "device": "mps",
        },
        p50_threshold_sec=8.5,
        p95_threshold_sec=12.0,
        description="Large aerial scenes with high-frequency texture on Apple Silicon",
    ),
    PerformanceBucket(
        name="pool_medium_mps",
        filters={
            "scene_type": "pool",
            "pixel_count_min": 10_000_000,
            "device": "mps",
        },
        p50_threshold_sec=11.0,
        p95_threshold_sec=15.0,
        description="Pool scenes with specular highlights and reflections on Apple Silicon",
    ),
    PerformanceBucket(
        name="interior_standard_mps",
        filters={
            "scene_type": "interior",
            "pixel_count_max": 15_000_000,
            "device": "mps",
        },
        p50_threshold_sec=7.0,
        p95_threshold_sec=10.0,
        description="Standard interior scenes on Apple Silicon",
    ),
    PerformanceBucket(
        name="generic_large",
        filters={
            "pixel_count_min": 20_000_000,
        },
        p50_threshold_sec=10.0,
        p95_threshold_sec=15.0,
        description="Fallback bucket for large images (device-agnostic)",
    ),
    PerformanceBucket(
        name="generic_medium",
        filters={
            "pixel_count_min": 5_000_000,
            "pixel_count_max": 20_000_000,
        },
        p50_threshold_sec=6.0,
        p95_threshold_sec=10.0,
        description="Fallback bucket for medium images (device-agnostic)",
    ),
    PerformanceBucket(
        name="unknown",
        filters={},
        p50_threshold_sec=60.0,
        p95_threshold_sec=120.0,
        description="Catch-all bucket for unclassified scenarios (very lenient thresholds)",
    ),
]


def get_bucket_for_capsule(
    capsule: PerformanceCapsule,
    buckets: Optional[list[PerformanceBucket]] = None
) -> PerformanceBucket:
    """Find the most specific bucket matching the capsule.

    Contract: Always returns a bucket. Never returns None.
    If no bucket matches, raises ValueError (should never happen with catch-all).

    Args:
        capsule: Performance capsule to match
        buckets: List of buckets to search (defaults to DEFAULT_BUCKETS)

    Returns:
        Most specific matching bucket (based on concept scoring, not key count)

    Raises:
        ValueError: If no bucket matches (indicates missing catch-all bucket)
    """
    if buckets is None:
        buckets = DEFAULT_BUCKETS

    # Find all matching buckets
    matches = [b for b in buckets if b.matches(capsule)]

    if not matches:
        raise ValueError(
            f"No bucket matched capsule (missing catch-all bucket?). "
            f"Capsule: scene_type={capsule.scene_type}, device={capsule.device}, "
            f"pixel_count={capsule.pixel_count}"
        )

    # Sort by specificity (descending), tie-break by name
    matches.sort(key=lambda b: (-b.specificity, b.name))
    return matches[0]
