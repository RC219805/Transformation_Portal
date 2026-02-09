"""Depth Pro backend adapter for unified backend registry.

Wraps existing DepthProStage to provide DepthBackend interface with
consistent contract and license governance.

See ADR-019 for architectural rationale.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from PIL import Image

from .protocol import DepthResult, LicenseRestrictionError, LicenseType

if TYPE_CHECKING:
    from ...lux_depth_v3.config import EnhanceConfig

logger = logging.getLogger(__name__)


class DepthProBackend:
    """Depth Pro backend adapter implementing DepthBackend protocol.

    Wraps DepthProStage for use with DepthBackendRegistry.
    Provides metric depth (meters) with focal length estimation.

    Attributes:
        name: Backend identifier ("depth_pro").
        license_type: RESEARCH_ONLY (Apple AMLR license).
        requires_checkpoint: True (1.9 GB checkpoint required).

    License Requirements:
        Depth Pro requires BOTH flags to be True:
        - non_commercial_ok: Acknowledge non-commercial use only
        - accept_apple_depth_pro_research_license: Accept Apple AMLR license

    Example:
        >>> from transformation_portal.depth.backends import DepthBackendRegistry
        >>> from transformation_portal.lux_depth_v3 import EnhanceConfig
        >>>
        >>> config = EnhanceConfig(
        ...     non_commercial_ok=True,
        ...     accept_apple_depth_pro_research_license=True,
        ...     depth_device="mps",
        ... )
        >>> registry = DepthBackendRegistry()
        >>> backend = registry.get_backend("depth_pro", config)
        >>> result = backend.compute(image)
        >>> print(f"Depth in meters, shape: {result.depth_map.shape}")
    """

    # Backend protocol attributes
    name = "depth_pro"
    license_type = LicenseType.RESEARCH_ONLY
    requires_checkpoint = True

    # Checkpoint configuration
    CHECKPOINT_URL = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"
    DEFAULT_CHECKPOINT = Path("checkpoints/depth_pro.pt")
    # Actual SHA256 of the checkpoint (verified against downloaded file)
    EXPECTED_SHA256 = "3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce"

    def __init__(self, config: Optional["EnhanceConfig"] = None):
        """Initialize Depth Pro backend.

        Args:
            config: EnhanceConfig with depth backend settings.
                If None, uses defaults (not recommended).
        """
        self._config = config
        self._stage = None
        self._device = self._resolve_device(config)
        self._checkpoint_path = self._resolve_checkpoint_path(config)

    def _resolve_device(self, config: Optional["EnhanceConfig"]) -> str:
        """Resolve device from config or auto-detect."""
        if config is not None:
            device = getattr(config, "depth_device", None)
            if device:
                return device

        # Auto-detect device
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
        except ImportError:
            logger.debug("PyTorch is not installed; falling back to CPU for DepthProBackend.")

        return "cpu"

    def _resolve_checkpoint_path(self, config: Optional["EnhanceConfig"]) -> Path:
        """Resolve checkpoint path from config, env var, or default.

        Resolution order:
        1. config.depth_pro_checkpoint_path (if set)
        2. TRANSFORMATION_PORTAL_DEPTH_PRO_CHECKPOINT env var
        3. Default: checkpoints/depth_pro.pt
        """
        # Option 1: Config value
        if config is not None:
            path = getattr(config, "depth_pro_checkpoint_path", None)
            if path:
                return Path(path)

        # Option 2: Environment variable
        env_path = os.environ.get("TRANSFORMATION_PORTAL_DEPTH_PRO_CHECKPOINT")
        if env_path:
            return Path(env_path)

        # Option 3: Default
        return self.DEFAULT_CHECKPOINT

    def ensure_available(self) -> None:
        """Ensure Depth Pro dependencies and checkpoint are available.

        Raises:
            ImportError: If depth_pro package not installed.
            FileNotFoundError: If checkpoint file missing.
        """
        # Check depth_pro package
        try:
            import depth_pro  # noqa: F401
        except ImportError:
            raise ImportError(
                "depth_pro package not installed.\n\n"
                "Install with:\n"
                "  pip install depth-pro\n\n"
                "See: https://github.com/apple/ml-depth-pro"
            )

        # Check checkpoint file
        if not self._checkpoint_path.exists():
            raise FileNotFoundError(
                f"Depth Pro checkpoint not found: {self._checkpoint_path}\n\n"
                f"Download checkpoint (1.9 GB) with:\n"
                f"  mkdir -p {self._checkpoint_path.parent}\n"
                f"  curl -L {self.CHECKPOINT_URL} -o {self._checkpoint_path}\n\n"
                f"Or set path via:\n"
                f"  - Config: depth_pro_checkpoint_path='path/to/checkpoint.pt'\n"
                f"  - Env: TRANSFORMATION_PORTAL_DEPTH_PRO_CHECKPOINT='path/to/checkpoint.pt'"
            )

    @classmethod
    def required_packages(cls) -> list[str]:
        """Return required import module names for Depth Pro backend.

        Depth Pro has its own package that wraps torch dependencies.
        torch is handled by the APEX runner and not listed here.

        Returns:
            ["depth_pro"]
        """
        return ["depth_pro"]

    def compute(
        self,
        image: Union[Image.Image, np.ndarray],
        device: Optional[str] = None,
    ) -> DepthResult:
        """Estimate metric depth from image.

        Args:
            image: Input image as PIL Image or numpy array (H, W, 3).
            device: Optional device override (cpu, cuda, mps).
                    Note: Device override requires stage reload; currently uses
                    device from initialization. See issue for enhancement.

        Returns:
            DepthResult with metric depth in meters and focal length.

        Raises:
            RuntimeError: If inference fails.
            LicenseRestrictionError: If license requirements not met.
        """
        # Layer 3: Runtime license enforcement (defense-in-depth)
        self._validate_license_runtime()

        # Ensure dependencies available
        self.ensure_available()

        # Lazy-load stage
        if self._stage is None:
            self._load_stage()

        # Convert image to format expected by DepthProStage
        if isinstance(image, np.ndarray):
            image_array = image
            image_pil = Image.fromarray((image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8))
        else:
            image_pil = image.convert("RGB")
            image_array = np.array(image_pil)

        # Import stage context
        from ...stage_graph.stage import StageContext, StageStatus

        # Create context and run inference
        use_device = device or self._device
        context = StageContext(
            artifacts={"image": image_pil},
            device=use_device,
        )

        result = self._stage.compute(context)

        if result.status != StageStatus.COMPLETED:
            raise RuntimeError(f"Depth Pro inference failed: {result.error}\n" f"Traceback:\n{result.error_traceback}")

        # Extract depth and metadata
        depth_map = result.artifacts.get("depth_map")
        if depth_map is None:
            raise RuntimeError("Depth Pro did not return depth_map artifact")

        provenance = result.artifacts.get("depth_provenance", {})

        # Build DepthResult with metric depth
        return DepthResult(
            depth_map=depth_map.astype(np.float32),
            original_image=image_array,
            metadata=provenance,
            depth_units="meters",
            focal_length_px=result.metadata.get("focal_length_px"),
            field_of_view_deg=result.metadata.get("fov_deg"),
            backend_id=self.name,
            device=use_device,
            dtype="float32",
            input_size=(image_array.shape[0], image_array.shape[1]),
        )

    def get_cache_key(self, image: Union[Image.Image, np.ndarray]) -> str:
        """Generate deterministic cache key for this image.

        Cache key includes:
        - Image content hash
        - Checkpoint hash (truncated)
        - Device
        - Backend version

        Args:
            image: Input image.

        Returns:
            Cache key string.
        """
        # Hash image content
        if isinstance(image, np.ndarray):
            image_hash = hashlib.sha256(image.tobytes()).hexdigest()[:16]
        else:
            image_hash = hashlib.sha256(image.tobytes()).hexdigest()[:16]

        # Get checkpoint hash (expensive, but cached in stage)
        ckpt_hash = self._get_checkpoint_hash()[:16] if self._checkpoint_path.exists() else "no_ckpt"

        return f"depthpro_{ckpt_hash}_{image_hash}_{self._device}_v1"

    def _load_stage(self) -> None:
        """Lazy-load DepthProStage."""
        from ...stage_graph.stages.depth_pro import DepthProStage

        self._stage = DepthProStage(
            checkpoint_path=self._checkpoint_path,
            device=self._device,
            strict_validation=True,
        )

    def _validate_license_runtime(self) -> None:
        """Runtime license validation (Layer 3: defense-in-depth).

        Raises:
            LicenseRestrictionError: If config missing required flags.
        """
        if self._config is None:
            raise LicenseRestrictionError(
                "Depth Pro requires EnhanceConfig with license flags.\n"
                "Create config with:\n"
                "  config = EnhanceConfig(\n"
                "      non_commercial_ok=True,\n"
                "      accept_apple_depth_pro_research_license=True,\n"
                "  )"
            )

        if not getattr(self._config, "non_commercial_ok", False):
            raise LicenseRestrictionError("Depth Pro requires non_commercial_ok=True in config.")

        if not getattr(self._config, "accept_apple_depth_pro_research_license", False):
            raise LicenseRestrictionError("Depth Pro requires accept_apple_depth_pro_research_license=True in config.")

        logger.debug("Runtime license validation passed for depth_pro")

    def _get_checkpoint_hash(self) -> str:
        """Get SHA256 of checkpoint file (cached)."""
        if not hasattr(self, "_checkpoint_hash_cached"):
            h = hashlib.sha256()
            with open(self._checkpoint_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            self._checkpoint_hash_cached = h.hexdigest()
        return self._checkpoint_hash_cached
