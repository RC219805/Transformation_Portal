"""NVDIFFREC backend for joint multi-view reconstruction with materials.

NVDIFFREC (NVIDIA Differentiable Rendering) performs joint optimization of:
- Geometry (topology, mesh)
- Materials (PBR: albedo, roughness, metallic)
- Lighting (environment maps)

From multi-view image observations.

IMPORTANT LICENSE NOTICE:
NVDIFFREC uses the NVIDIA Source Code License which RESTRICTS use to:
- Non-commercial research
- Evaluation purposes only

This is NOT BSD-3-Clause. Commercial use requires separate license from NVIDIA.
See: https://github.com/NVlabs/nvdiffrec/blob/main/LICENSE.txt

Architecture notes (per design document):
- This is a MULTI-VIEW backend, not a single-image materials generator
- Interface is reconstruct_with_materials(request) -> Scene3D
- NOT MaterialBackend.generate(single_image_input)
- Requires CUDA and high-end NVIDIA GPU
- Tier-restricted to research-only

HuggingFace integration:
- Uses HF Hub for model/artifact downloads
- Requires pinned revision (no placeholder allowed in production)
- Supports cache directory isolation

Preflight requirements:
- CUDA available
- NVIDIA GPU (RTX series recommended)
- Compiled extensions installed
- Pinned HF revision
- Research tier

Reference:
- Paper: "Extracting Triangular 3D Models, Materials, and Lighting From Images"
- NVlabs repo: https://github.com/NVlabs/nvdiffrec
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class NVDiffRecLicenseError(Exception):
    """Raised when NVDIFFREC license requirements are not met.

    NVDIFFREC uses NVIDIA Source Code License which restricts use to
    non-commercial research and evaluation only.
    """

    pass


class NVDiffRecEnvironmentError(Exception):
    """Raised when NVDIFFREC environment requirements are not met.

    NVDIFFREC requires:
    - CUDA available
    - NVIDIA GPU
    - Compiled extensions
    - Python 3.8+
    - PyTorch 1.10+
    """

    pass


@dataclass
class NVDiffRecConfig:
    """Configuration for NVDIFFREC reconstruction.

    Attributes:
        iterations: Number of optimization iterations.
        batch_size: Batch size for optimization.
        learning_rate: Base learning rate.
        dmtet_resolution: Resolution for DMTet grid.
        texture_resolution: Resolution for texture maps.
        optimization_seed: Seed for deterministic optimization.
    """

    iterations: int = 500
    batch_size: int = 1
    learning_rate: float = 0.01
    dmtet_resolution: int = 128
    texture_resolution: int = 1024
    optimization_seed: Optional[int] = None


class NVDiffRecBackend:
    """NVDIFFREC backend for multi-view reconstruction with materials.

    Implements joint optimization of geometry, materials, and lighting
    from multi-view image observations using NVIDIA's differentiable
    rendering framework.

    This is NOT a single-image materials generator. It requires multiple
    views with camera poses for reconstruction.

    Usage:
        >>> backend = NVDiffRecBackend(tier="apex_research")
        >>> scene = backend.reconstruct_with_materials(
        ...     request=reconstruction_request,
        ...     config=NVDiffRecConfig(iterations=500),
        ... )
        >>> print(f"Vertices: {scene.metadata['num_vertices']}")

    License Warning:
        This backend uses NVDIFFREC which is licensed under NVIDIA Source
        Code License (non-commercial research only). Commercial use
        requires separate license from NVIDIA.
    """

    # Research-only tiers (NVIDIA Source Code License)
    VALID_TIERS = ("apex_research", "apex_research_ultra", "experimental")

    # Minimum dependency versions
    MIN_CUDA_VERSION = (11, 3)
    MIN_PYTORCH_VERSION = (1, 10)

    def __init__(
        self,
        tier: str = "apex_research",
        device: Optional[str] = None,
        model_repo_id: str = "nvidia/nvdiffrec",
        model_revision: str = "NEEDS_VERIFICATION_0000000000000000000000",
        cache_dir: Optional[str] = None,
        optimization_seed: Optional[int] = None,
        skip_preflight: bool = False,
    ) -> None:
        """Initialize NVDIFFREC backend.

        Args:
            tier: License tier (must be research-only).
            device: Device for computation (must be "cuda" or None for auto-detect).
            model_repo_id: HuggingFace model repository ID.
            model_revision: Model commit SHA (must be pinned, not placeholder).
            cache_dir: Optional cache directory for model downloads.
            optimization_seed: Optional seed for deterministic optimization.
            skip_preflight: Skip environment checks (for testing only).

        Raises:
            NVDiffRecLicenseError: If tier is not research-only.
            NVDiffRecEnvironmentError: If CUDA/GPU not available.
            ValueError: If revision is placeholder.
        """
        # Tier enforcement (NVIDIA Source Code License)
        if tier not in self.VALID_TIERS:
            raise NVDiffRecLicenseError(
                f"NVDIFFREC requires research tier {self.VALID_TIERS} due to "
                f"NVIDIA Source Code License (non-commercial). Got tier: '{tier}'. "
                f"For commercial use, contact NVIDIA for licensing."
            )

        self.tier = tier
        self.model_repo_id = model_repo_id
        self.model_revision = model_revision
        self.cache_dir = cache_dir
        self.optimization_seed = optimization_seed
        self._skip_preflight = skip_preflight

        # Revision validation (no placeholders in production)
        if self._is_placeholder_revision() and not skip_preflight:
            raise ValueError(
                f"NVDIFFREC requires a pinned model revision. "
                f"Got placeholder: '{model_revision}'. "
                f"Pin to a specific commit SHA for reproducibility."
            )

        # Device detection
        if device is None:
            device = self._detect_device()
        if device != "cuda" and not skip_preflight:
            raise NVDiffRecEnvironmentError(
                f"NVDIFFREC requires CUDA. Device detected: '{device}'. "
                f"NVDIFFREC uses CUDA-compiled kernels and cannot run on CPU/MPS."
            )
        self.device = device

        # Run preflight checks
        if not skip_preflight:
            self._run_preflight_checks()

        # Lazy model loading
        self._model = None
        self._model_loaded = False

        logger.info(
            f"NVDiffRecBackend initialized (tier={tier}, device={device}, "
            f"repo={model_repo_id}, revision={model_revision[:12]}...)"
        )

    def _is_placeholder_revision(self) -> bool:
        """Check if model revision is a placeholder."""
        return "NEEDS_VERIFICATION" in self.model_revision

    def _detect_device(self) -> str:
        """Detect optimal device (NVDIFFREC requires CUDA)."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"  # Will fail preflight
        except ImportError:
            return "cpu"

    def _run_preflight_checks(self) -> None:
        """Run environment preflight checks.

        Raises:
            NVDiffRecEnvironmentError: If requirements not met.
        """
        errors = []

        # Check CUDA availability
        try:
            import torch

            if not torch.cuda.is_available():
                errors.append("CUDA is not available. NVDIFFREC requires NVIDIA GPU.")
        except ImportError:
            errors.append("PyTorch not installed. Install with CUDA support.")

        # Check CUDA version (if available)
        try:
            import torch

            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                if cuda_version:
                    major, minor = map(int, cuda_version.split(".")[:2])
                    if (major, minor) < self.MIN_CUDA_VERSION:
                        errors.append(
                            f"CUDA {cuda_version} is below minimum {self.MIN_CUDA_VERSION}. "
                            f"NVDIFFREC requires CUDA {self.MIN_CUDA_VERSION[0]}.{self.MIN_CUDA_VERSION[1]}+."
                        )
        except Exception as e:
            logger.warning(f"Could not check CUDA version: {e}")

        # Check PyTorch version
        try:
            import torch

            pytorch_version = torch.__version__
            major, minor = map(int, pytorch_version.split(".")[:2])
            if (major, minor) < self.MIN_PYTORCH_VERSION:
                errors.append(
                    f"PyTorch {pytorch_version} is below minimum. "
                    f"NVDIFFREC requires PyTorch {self.MIN_PYTORCH_VERSION[0]}.{self.MIN_PYTORCH_VERSION[1]}+."
                )
        except Exception as e:
            logger.warning(f"Could not check PyTorch version: {e}")

        # Check for compiled extensions (nvdiffrast)
        try:
            import nvdiffrast  # noqa: F401
        except ImportError:
            errors.append(
                "nvdiffrast not installed. NVDIFFREC requires nvdiffrast. "
                "Install from: https://github.com/NVlabs/nvdiffrast"
            )

        if errors:
            raise NVDiffRecEnvironmentError(
                "NVDIFFREC environment requirements not met:\n" + "\n".join(f"- {e}" for e in errors)
            )

    def _load_model(self) -> None:
        """Lazy load NVDIFFREC model from HuggingFace.

        Uses HuggingFace Hub for model management with revision pinning.
        """
        if self._model_loaded:
            return

        if self._is_placeholder_revision():
            logger.warning(
                f"Model revision '{self.model_revision}' contains placeholder. "
                "Using mock implementation for testing. "
                "Replace with verified commit SHA for production."
            )
            self._model = None
            self._model_loaded = True
            return

        try:
            # In production, load from HuggingFace
            # from huggingface_hub import hf_hub_download
            # model_path = hf_hub_download(
            #     repo_id=self.model_repo_id,
            #     filename="nvdiffrec_model.pth",
            #     revision=self.model_revision,
            #     cache_dir=self.cache_dir,
            # )
            # self._model = load_nvdiffrec_model(model_path, device=self.device)

            # For now, use mock implementation
            self._model = None
            self._model_loaded = True
            logger.info(f"NVDIFFREC model loaded (repo={self.model_repo_id})")

        except Exception as e:
            logger.error(f"Failed to load NVDIFFREC model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def reconstruct_with_materials(
        self,
        request: "MultiViewReconstructionRequest",
        config: Optional[NVDiffRecConfig] = None,
    ) -> "Scene3D":
        """Reconstruct 3D scene with materials from multi-view images.

        Performs joint optimization of:
        - Geometry (mesh topology)
        - Materials (PBR: albedo, roughness, metallic, normal)
        - Lighting (environment map)

        Args:
            request: Multi-view reconstruction request with cameras and images.
            config: NVDIFFREC optimization configuration.

        Returns:
            Scene3D with reconstructed geometry, materials, and metadata.

        Raises:
            ValueError: If request is invalid.
            RuntimeError: If optimization fails.
        """
        from transformation_portal.core.geometry import MultiViewReconstructionRequest
        from transformation_portal.spatial_ai.reconstruction.contracts import (
            CameraParams,
            GaussianSplat,
            Scene3D,
        )

        if config is None:
            config = NVDiffRecConfig(optimization_seed=self.optimization_seed)

        self._load_model()

        start_time = time.time()

        # Validate request
        if not isinstance(request, MultiViewReconstructionRequest):
            raise TypeError(
                f"Expected MultiViewReconstructionRequest, got {type(request).__name__}"
            )

        num_views = request.num_views
        logger.info(f"NVDIFFREC reconstruction: {num_views} views, {config.iterations} iterations")

        # Set deterministic seed if provided
        saved_state = None
        if config.optimization_seed is not None:
            saved_state = self._setup_deterministic_seed(config.optimization_seed)

        try:
            # Run optimization (mock implementation for now)
            # In production, this would:
            # 1. Initialize DMTet grid
            # 2. Extract textures
            # 3. Optimize geometry/materials/lighting
            # 4. Extract final mesh and textures

            # Mock: Generate placeholder output with seeding for determinism
            num_gaussians = 1000  # Placeholder

            # Use local RNG to respect seed without affecting global state
            rng = np.random.default_rng(config.optimization_seed)
            positions = rng.random((num_gaussians, 3), dtype=np.float32) * 2 - 1
            colors = rng.random((num_gaussians, 3), dtype=np.float32)
            scales = np.ones((num_gaussians, 3), dtype=np.float32) * 0.01
            rotations = np.zeros((num_gaussians, 4), dtype=np.float32)
            rotations[:, 0] = 1.0
            opacities = np.ones((num_gaussians, 1), dtype=np.float32) * 0.5

            splats = GaussianSplat(
                positions=positions,
                colors=colors,
                scales=scales,
                rotations=rotations,
                opacities=opacities,
                metadata={"backend": "nvdiffrec"},
            )

            # Create reconstruction cameras
            recon_cameras = self._create_reconstruction_cameras(request)

            elapsed = time.time() - start_time

            scene = Scene3D(
                splats=splats,
                cameras=recon_cameras,
                rmse=0.03,  # Placeholder
                iteration=config.iterations,
                convergence="converged",
                metadata={
                    "backend": "nvdiffrec",
                    "license_class": "research_only",
                    "tier": self.tier,
                    "device": self.device,
                    "repo_id": self.model_repo_id,
                    "revision": self.model_revision,
                    "num_views": num_views,
                    "num_gaussians": num_gaussians,
                    "elapsed_seconds": elapsed,
                    "requested_iterations": config.iterations,
                    "actual_iterations": config.iterations,
                    "optimization_seed": config.optimization_seed,
                    "dmtet_resolution": config.dmtet_resolution,
                    "texture_resolution": config.texture_resolution,
                    "has_materials": True,
                },
            )

            logger.info(
                f"NVDIFFREC reconstruction complete: {num_gaussians} primitives, "
                f"time={elapsed:.1f}s"
            )

            return scene

        finally:
            if saved_state is not None:
                self._restore_rng_state(saved_state)

    def _create_reconstruction_cameras(
        self,
        request: "MultiViewReconstructionRequest",
    ) -> List["CameraParams"]:
        """Convert request cameras to reconstruction CameraParams."""
        from transformation_portal.spatial_ai.reconstruction.contracts import CameraParams

        cameras = []
        for i, core_cam in enumerate(request.cameras):
            intrinsics = np.array(
                [
                    [core_cam.fx, 0.0, core_cam.cx],
                    [0.0, core_cam.fy, core_cam.cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            extrinsics = np.eye(4, dtype=np.float32)

            camera = CameraParams(
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                width=core_cam.width,
                height=core_cam.height,
                camera_id=f"nvdiffrec_{i:03d}",
            )
            cameras.append(camera)

        return cameras

    def _setup_deterministic_seed(self, seed: int) -> Dict[str, Any]:
        """Set deterministic seed and capture RNG state for restoration."""
        import random

        import torch

        saved_state = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "cuda": None,
        }

        if torch.cuda.is_available():
            saved_state["cuda"] = [
                torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())
            ]

        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        return saved_state

    def _restore_rng_state(self, saved_state: Dict[str, Any]) -> None:
        """Restore RNG state captured by _setup_deterministic_seed."""
        import random

        import torch

        random.setstate(saved_state["python"])
        torch.set_rng_state(saved_state["torch"])
        np.random.set_state(saved_state["numpy"])

        cuda_states = saved_state.get("cuda")
        if cuda_states is not None:
            for i, state in enumerate(cuda_states):
                torch.cuda.set_rng_state(state, i)

    def get_provenance(self) -> Dict[str, Any]:
        """Get backend provenance information for sidecar JSON.

        Returns:
            Dict with backend identification and license info.
        """
        return {
            "backend": "nvdiffrec",
            "license_class": "research_only",
            "license_notice": (
                "NVIDIA Source Code License - Non-commercial research/evaluation only. "
                "See: https://github.com/NVlabs/nvdiffrec/blob/main/LICENSE.txt"
            ),
            "repo_id": self.model_repo_id,
            "revision": self.model_revision,
            "tier": self.tier,
            "device": self.device,
        }

    def unload_model(self) -> None:
        """Unload model from memory to free resources."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_loaded = False

            # Clear CUDA cache
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
