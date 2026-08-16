"""Unified depth backend protocol and result dataclass.

Defines the contract for all depth estimation backends (DA2, DA3, Depth Pro)
with consistent input/output types and license governance.

See ADR-019 for architectural rationale.
See ADR-026 §2.3 for StatefulBackend lifecycle protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Literal, Optional, Protocol, Union

import numpy as np
from PIL import Image


class LicenseType(Enum):
    """License classification for depth backends.

    Used for runtime license enforcement.
    """

    COMMERCIAL = "commercial"
    RESEARCH_ONLY = "research_only"
    MODEL_DEPENDENT = "model_dependent"


class LicenseRestrictionError(Exception):
    """Raised when license requirements are not met.

    Provides actionable error messages with license URLs.
    """


@dataclass
class DepthResult:
    """Unified depth estimation result contract.

    Supports both relative depth (0-1 normalized) and metric depth (meters).
    Backward compatible with existing DepthResult usage in lux_depth_v3.

    Attributes:
        depth_map: Depth values, shape (H, W). For relative depth, values
            are in [0, 1]. For metric depth, values are in meters.
        original_image: Input image as numpy array,
            shape (H, W, 3), RGB [0-255].
        metadata: Backend-specific metadata (provenance, timing, etc.).
        depth_units: "relative" (0-1 normalized) or "meters" (absolute scale).
        focal_length_px: Focal length in pixels
            (metric depth backends only).
        field_of_view_deg: Horizontal field of view
            in degrees (metric depth only).
        backend_id: Identifier of the backend that produced this result.
        device: Device used for inference (cpu, cuda, mps).
        dtype: Data type used for inference (float32, float16, bfloat16).
        input_size: Original image dimensions as (height, width).
        warnings: Any warnings generated during inference.

    Example:
        >>> result = DepthResult(
        ...     depth_map=np.zeros((100, 100), dtype=np.float32),
        ...     original_image=np.zeros((100, 100, 3), dtype=np.uint8),
        ...     metadata={"engine": "depth_pro"},
        ...     depth_units="meters",
        ...     focal_length_px=525.0,
        ...     field_of_view_deg=65.0,
        ...     backend_id="depth_pro",
        ...     device="mps",
        ... )
        >>> result.is_metric
        True
    """

    # Core fields (v2.0.0 contract - unchanged)
    depth_map: np.ndarray
    original_image: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    # New fields for metric depth (backward compatible with defaults)
    depth_units: Literal["relative", "meters"] = "relative"
    focal_length_px: Optional[float] = None
    field_of_view_deg: Optional[float] = None

    # Backend identification
    backend_id: Optional[str] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    input_size: Optional[tuple] = None

    # Warnings and diagnostics
    warnings: list = field(default_factory=list)

    @property
    def depth(self) -> np.ndarray:
        """Alias for depth_map (backward compatibility with lux_depth_v3)."""
        return self.depth_map

    @property
    def is_metric(self) -> bool:
        """Check if depth is metric (absolute scale in meters)."""
        return self.depth_units == "meters"

    def to_relative(self) -> "DepthResult":
        """Convert metric depth to relative (0-1 normalized).

        Returns a new DepthResult with normalized depth values.
        """
        if not self.is_metric:
            return self

        # Robust normalization using percentiles to handle outliers
        depth = self.depth_map.astype(np.float32)
        valid = np.isfinite(depth)

        if not valid.any():
            normalized = np.zeros_like(depth)
        else:
            vmin = float(np.percentile(depth[valid], 1))
            vmax = float(np.percentile(depth[valid], 99))
            if vmax <= vmin:
                vmax = vmin + 1e-6
            normalized = np.clip((depth - vmin) / (vmax - vmin), 0.0, 1.0)

        return DepthResult(
            depth_map=normalized.astype(np.float32),
            original_image=self.original_image,
            metadata=self.metadata,
            depth_units="relative",
            focal_length_px=self.focal_length_px,
            field_of_view_deg=self.field_of_view_deg,
            backend_id=self.backend_id,
            device=self.device,
            dtype=self.dtype,
            input_size=self.input_size,
            warnings=self.warnings + ["converted from metric to relative"],
        )


class DepthBackend(Protocol):
    """Protocol for unified depth estimation backends.

    All depth backends (DA2, DA3, Depth Pro) must implement this interface
    to be compatible with DepthBackendRegistry.

    Attributes:
        name: Unique backend identifier
            (e.g., "depth_pro", "depth_anything_v3").
        license_type: License classification for governance.
        requires_checkpoint: Whether backend requires external checkpoint file.

    Example:
        >>> class MyDepthBackend:
        ...     name = "my_backend"
        ...     license_type = LicenseType.COMMERCIAL
        ...     requires_checkpoint = False
        ...
        ...     def compute(self, image, device=None):
        ...         return DepthResult(...)
        ...
        ...     def get_cache_key(self, image):
        ...         return "cache_key"
    """

    name: str
    license_type: LicenseType
    requires_checkpoint: bool

    def __init__(
        self,
        config: Any = None,
    ) -> None: ...

    def compute(
        self,
        image: Union[Image.Image, np.ndarray],
        device: Optional[str] = None,
    ) -> DepthResult:
        """Estimate depth from image.

        Args:
            image: Input image as PIL Image or numpy array (H, W, 3).
            device: Optional device override (cpu, cuda, mps).

        Returns:
            DepthResult with depth map and metadata.

        Raises:
            RuntimeError: If inference fails.
            LicenseRestrictionError: If license requirements not met.
        """

    def get_cache_key(self, image: Union[Image.Image, np.ndarray]) -> str:
        """Generate deterministic cache key for this image.

        Cache key should incorporate:
        - Image content hash
        - Model version/checkpoint hash
        - Backend configuration

        Args:
            image: Input image.

        Returns:
            Cache key string.
        """

    def ensure_available(self) -> None:
        """Ensure backend dependencies and resources are available.

        Raises:
            ImportError: If required packages are not installed.
            FileNotFoundError: If checkpoint is missing.
        """

    @classmethod
    def required_packages(cls) -> list[str]:
        """Return list of required import module names for this backend.

        Returns module names (not pip package names) that must be importable
        for this backend to function. For example: ["transformers"].

        The APEX runner treats Torch plus this list as the default host-process
        requirements. A backend may expose ``runtime_required_packages()`` to
        provide the complete host requirement set for its configured runtime;
        an isolated subprocess backend can therefore return an empty list.

        Returns:
            List of import module names (e.g., ["transformers"]).

        Example:
            >>> DA3Backend.required_packages()
            ['transformers']
        """


class StatefulBackend(Protocol):
    """Protocol for backends that maintain temporal or sequence state.

    Backends that accumulate state across frames (e.g., temporal filters,
    video trackers) must implement this protocol so the orchestrator can
    reset state at sequence boundaries and prevent
    cross-sequence contamination.

    See ADR-026 §2.3 for sequence lifecycle requirements.

    Example:
        >>> class MyTemporalBackend:
        ...     def reset_state(self, sequence_id=None):
        ...         self._buffer.clear()
    """

    def reset_state(self, sequence_id: Optional[str] = None) -> None:
        """Reset internal state for a new sequence.

        Called by the orchestrator at sequence boundaries to prevent
        temporal blending between unrelated sequences.

        Args:
            sequence_id: Optional identifier for the new sequence.
                If provided, backends may use it for logging or
                provenance tracking. None means "anonymous reset".
        """
