"""Multi-view scene construction with depth and material integration.

Orchestrates 3D reconstruction pipeline:
1. Multi-view image loading and preprocessing
2. Integration with depth maps (Phase 1)
3. Integration with segmentation masks (Phase 2.1)
4. Integration with PBR textures (Phase 2.2)
5. Scene optimization and validation

Architecture:
- Lazy loading of dependencies
- Progressive reconstruction (coarse-to-fine)
- Memory-efficient batch processing
- Integration hooks for all spatial_ai phases
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .contracts import CameraParams, ReconstructionInput, Scene3D
from .gaussian_backend import GaussianBackend

logger = logging.getLogger(__name__)


class SceneBuilder:
    """Multi-view scene construction with phase integration.

    Builds 3D scenes from multi-view images with optional:
    - Depth priors from Phase 1 (LinearDecoder)
    - Segmentation masks from Phase 2.1 (SAM2)
    - PBR textures from Phase 2.2 (MaterialBackend)

    Usage:
        >>> builder = SceneBuilder(tier="apex_research")
        >>> scene = builder.build_from_images(
        ...     image_paths=["view1.png", "view2.png", "view3.png"],
        ...     cameras=cameras,
        ...     depth_maps=depth_maps,
        ... )
        >>> print(f"Scene quality: {scene.quality_score:.1f}/100")
    """

    def __init__(
        self,
        tier: str = "apex_research",
        device: Optional[str] = None,
        backend_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize scene builder.

        Args:
            tier: Tier restriction for license enforcement.
            device: Device for computation ("cuda", "mps", "cpu").
            backend_config: Optional backend configuration overrides.
        """
        self.tier = tier
        self.device = device
        self.backend_config = backend_config or {}

        # Lazy backend initialization
        self._backend: Optional[GaussianBackend] = None

        logger.info(f"SceneBuilder initialized (tier={tier})")

    @property
    def backend(self) -> GaussianBackend:
        """Lazy-load Gaussian backend."""
        if self._backend is None:
            self._backend = GaussianBackend(tier=self.tier, device=self.device, **self.backend_config)
        return self._backend

    def build_from_images(
        self,
        image_paths: List[Path],
        cameras: List[CameraParams],
        depth_maps: Optional[List[np.ndarray]] = None,
        masks: Optional[List[np.ndarray]] = None,
        material_maps: Optional[List[Dict[str, np.ndarray]]] = None,
        iterations: int = 30000,
        gamma: float = 1.0,
    ) -> Scene3D:
        """Build 3D scene from image files.

        Args:
            image_paths: Paths to multi-view images.
            cameras: Camera parameters for each view.
            depth_maps: Optional depth priors (H, W) float32.
            masks: Optional segmentation masks (H, W) bool.
            material_maps: Optional PBR texture maps.
            iterations: Optimization iterations.
            gamma: Gamma value (must be 1.0 for linear RGB).

        Returns:
            Reconstructed 3D scene.

        Raises:
            ValueError: If input validation fails.
            FileNotFoundError: If image files not found.
        """
        # Load images
        images = self._load_images(image_paths, gamma)

        # Build reconstruction input
        reconstruction_input = ReconstructionInput(
            images=images,
            gamma=gamma,
            cameras=cameras,
            depth_maps=depth_maps,
            masks=masks,
            material_maps=material_maps,
            tier=self.tier,
        )

        # Reconstruct scene
        scene = self.backend.reconstruct(
            reconstruction_input,
            iterations=iterations,
            use_depth_prior=(depth_maps is not None),
            use_segmentation=(masks is not None),
            use_pbr_textures=(material_maps is not None),
        )

        return scene

    def build_from_arrays(
        self,
        images: List[np.ndarray],
        cameras: List[CameraParams],
        depth_maps: Optional[List[np.ndarray]] = None,
        masks: Optional[List[np.ndarray]] = None,
        material_maps: Optional[List[Dict[str, np.ndarray]]] = None,
        iterations: int = 30000,
        gamma: float = 1.0,
    ) -> Scene3D:
        """Build 3D scene from numpy arrays (in-memory).

        Args:
            images: Multi-view images (H, W, 3) float32 in linear RGB.
            cameras: Camera parameters for each view.
            depth_maps: Optional depth priors (H, W) float32.
            masks: Optional segmentation masks (H, W) bool.
            material_maps: Optional PBR texture maps.
            iterations: Optimization iterations.
            gamma: Gamma value (must be 1.0 for linear RGB).

        Returns:
            Reconstructed 3D scene.
        """
        # Build reconstruction input
        reconstruction_input = ReconstructionInput(
            images=images,
            gamma=gamma,
            cameras=cameras,
            depth_maps=depth_maps,
            masks=masks,
            material_maps=material_maps,
            tier=self.tier,
        )

        # Reconstruct scene
        scene = self.backend.reconstruct(
            reconstruction_input,
            iterations=iterations,
            use_depth_prior=(depth_maps is not None),
            use_segmentation=(masks is not None),
            use_pbr_textures=(material_maps is not None),
        )

        return scene

    def _load_images(self, image_paths: List[Path], gamma: float) -> List[np.ndarray]:
        """Load images from disk as linear RGB float32.

        Args:
            image_paths: Paths to image files.
            gamma: Gamma value for linearization.

        Returns:
            List of images (H, W, 3) float32.

        Raises:
            FileNotFoundError: If image file not found.
            ValueError: If gamma != 1.0 (must be pre-linearized).
        """
        if abs(gamma - 1.0) > 1e-6:
            raise ValueError(
                f"SceneBuilder requires gamma=1.0 (pre-linearized images), got {gamma}. "
                "Linearize images before passing to SceneBuilder."
            )

        images = []
        for path in image_paths:
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")

            # Load image
            from PIL import Image

            img = Image.open(path)
            img_array = np.array(img).astype(np.float32) / 255.0

            # Ensure RGB
            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]  # Drop alpha

            images.append(img_array)

        return images

    def render_novel_view(self, scene: Scene3D, camera: CameraParams) -> np.ndarray:
        """Render novel view from reconstructed scene.

        Args:
            scene: Reconstructed 3D scene.
            camera: Target camera viewpoint.

        Returns:
            Rendered image (H, W, 3) float32 in linear RGB.
        """
        return self.backend.render_view(scene, camera)

    def extract_camera_path(
        self,
        scene: Scene3D,
        num_frames: int = 100,
        interpolation: str = "linear",
    ) -> List[CameraParams]:
        """Extract smooth camera path for video rendering.

        Interpolates camera poses using mathematically correct interpolation
        that preserves rigid transformations (no shearing/skewing).

        Modes:
        - ``"linear"`` (default): Two-keyframe path between the first and last
          camera. Translation uses linear interpolation (LERP); rotation uses
          spherical linear interpolation (SLERP) via quaternions. Intermediate
          cameras in the scene are ignored.
        - ``"spline"``: Multi-keyframe path that passes through every camera
          in ``scene.cameras`` (uniformly spaced in path parameter ``t``).
          Translation uses a natural cubic spline; rotation uses piecewise
          SLERP across all keyframes. With only two cameras this degenerates
          to the same result as ``"linear"``.

        Args:
            scene: Reconstructed 3D scene.
            num_frames: Number of frames in camera path.
            interpolation: Interpolation method, ``"linear"`` or ``"spline"``.

        Returns:
            List of camera parameters along path.

        Raises:
            ValueError: If scene has fewer than 2 cameras.
            NotImplementedError: If ``interpolation`` is neither ``"linear"``
                nor ``"spline"``.
        """
        if len(scene.cameras) < 2:
            raise ValueError("Need at least 2 cameras for path extraction")

        if interpolation not in ("linear", "spline"):
            raise NotImplementedError(
                f"Interpolation method '{interpolation}' not implemented. " "Supported methods: 'linear', 'spline'."
            )

        # Lazy import scipy to maintain module's lazy-loading contract
        from scipy.spatial.transform import Rotation, Slerp

        if interpolation == "linear":
            keyframes = [scene.cameras[0], scene.cameras[-1]]
        else:  # "spline"
            keyframes = list(scene.cameras)

        n_keys = len(keyframes)
        key_times = np.linspace(0.0, 1.0, n_keys)

        # Stack rotations and translations from keyframe extrinsics.
        # Extrinsics format: [[R|t], [0 0 0 1]] where R is 3x3 rotation, t is 3x1 translation
        key_rotations = np.stack([cam.extrinsics[:3, :3] for cam in keyframes])
        key_translations = np.stack([cam.extrinsics[:3, 3] for cam in keyframes])

        # SLERP handles both two-key (linear) and multi-key (spline) cases.
        slerp = Slerp(key_times, Rotation.from_matrix(key_rotations))

        if interpolation == "spline" and n_keys >= 3:
            # Natural cubic spline through all keyframe positions. With <3
            # keyframes scipy still produces a valid spline but it degenerates
            # to a straight line, which is identical to LERP — fall through to
            # the LERP path below to avoid the extra dependency call.
            from scipy.interpolate import CubicSpline

            translation_fn = CubicSpline(key_times, key_translations, bc_type="natural")
        else:
            # Piecewise linear interpolation across keyframes (covers both
            # 2-keyframe "linear" mode and 2-keyframe "spline" degeneracy).
            def translation_fn(t: float) -> np.ndarray:
                return np.array(
                    [np.interp(t, key_times, key_translations[:, axis]) for axis in range(3)],
                    dtype=np.float64,
                )

        cam0 = keyframes[0]
        path: List[CameraParams] = []
        for i in range(num_frames):
            t = i / (num_frames - 1) if num_frames > 1 else 0.0

            t_interp = np.asarray(translation_fn(t))
            R_interp = slerp(t).as_matrix()

            # Reconstruct 4x4 extrinsics matrix
            extrinsics = np.eye(4, dtype=np.float32)
            extrinsics[:3, :3] = R_interp.astype(np.float32)
            extrinsics[:3, 3] = t_interp.astype(np.float32)

            # Copy intrinsics from first keyframe
            cam = CameraParams(
                intrinsics=cam0.intrinsics.copy(),
                extrinsics=extrinsics,
                width=cam0.width,
                height=cam0.height,
                camera_id=f"path_{i:04d}",
            )
            path.append(cam)

        return path
