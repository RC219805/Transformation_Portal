"""Tiny camera-to-world initialization regressions; no rendering or fitting."""

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch is required to import the Gaussian backend")

from transformation_portal.spatial_ai.reconstruction.contracts import CameraParams, ReconstructionInput
from transformation_portal.spatial_ai.reconstruction.gaussian_backend import GaussianBackend

pytestmark = [pytest.mark.ml, pytest.mark.unit]


@pytest.mark.parametrize("use_depth_prior", [True, False], ids=["depth", "fixed-5m"])
@pytest.mark.parametrize(
    "extrinsics",
    [
        np.eye(4, dtype=np.float32),
        np.array([[1, 0, 0, 1], [0, 1, 0, -2], [0, 0, 1, 0.5], [0, 0, 0, 1]], dtype=np.float32),
        np.array([[0, 0, 1, -1], [0, 1, 0, -2], [-1, 0, 0, 6], [0, 0, 0, 1]], dtype=np.float32),
    ],
    ids=["identity", "translation", "rotation-translation"],
)
def test_initialize_gaussians_camera_roundtrip(extrinsics: np.ndarray, use_depth_prior: bool) -> None:
    intrinsics = np.array([[2, 0, 1], [0, 4, 1], [0, 0, 1]], dtype=np.float32)
    cameras = [CameraParams(intrinsics.copy(), extrinsics.copy(), width=2, height=2) for _ in range(2)]
    pixel_colors = np.array([[0.25, 0.5, 0.75], [0.75, 0.25, 0.5]], dtype=np.float32)
    images = [np.zeros((2, 2, 3), dtype=np.float32) for _ in range(2)]
    masks = [np.zeros((2, 2), dtype=bool) for _ in range(2)]
    depths = [np.full((2, 2), 30.0, dtype=np.float32) for _ in range(2)]
    for i, depth in enumerate([2.0, 4.0]):
        images[i][0, 0] = pixel_colors[i]
        masks[i][0, 0] = True
        depths[i][0, 0] = depth
    reconstruction_input = ReconstructionInput(
        images=images,
        gamma=1.0,
        cameras=cameras,
        depth_maps=depths,
        masks=masks,
        tier="experimental",
    )
    backend = GaussianBackend(tier="experimental", device="cpu")

    splats = backend._initialize_gaussians(reconstruction_input, use_depth_prior=use_depth_prior, use_segmentation=True)

    # The sole retained pixel is (u,v)=(0,0). Its ray is (-1/2,-1/4,1),
    # so supplied Z=2/4 gives the first oracle; fallback must use fixed Z=5
    # despite depth maps being available. Neither oracle uses E or its inverse.
    expected_camera = (
        np.array([[-1.0, -0.5, 2.0], [-2.0, -1.0, 4.0]], dtype=np.float32)
        if use_depth_prior
        else np.array([[-2.5, -1.25, 5.0], [-2.5, -1.25, 5.0]], dtype=np.float32)
    )
    assert splats.positions.shape == (2, 3)
    assert splats.positions.dtype == np.float32
    for i, camera in enumerate(cameras):
        world_homogeneous = np.append(splats.positions[i], np.float32(1.0))
        camera_homogeneous = camera.extrinsics @ world_homogeneous
        np.testing.assert_allclose(camera_homogeneous[:3], expected_camera[i], rtol=1e-6, atol=1e-6)
        assert camera_homogeneous[3] == pytest.approx(1.0)
    np.testing.assert_array_equal(splats.colors, pixel_colors)
    assert splats.metadata["initialization"] == ("depth" if use_depth_prior else "sfm")
    assert backend._model_loaded is False
