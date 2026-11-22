"""Tiny camera-to-world initialization regressions; no rendering or fitting."""

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch is required to import the Gaussian backend")

from transformation_portal.spatial_ai.reconstruction.contracts import CameraParams, ReconstructionInput, Scene3D
from transformation_portal.spatial_ai.reconstruction.gaussian_backend import GaussianBackend
from transformation_portal.spatial_ai.reconstruction.geometric_validator import GeometricValidator

pytestmark = [pytest.mark.ml, pytest.mark.unit]


@pytest.mark.parametrize("use_depth_prior", [True, False], ids=["depth", "fixed-5m"])
def test_initialize_gaussians_camera_roundtrip(use_depth_prior: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    for pose_name, extrinsics in [
        ("identity", np.eye(4, dtype=np.float32)),
        (
            "translation",
            np.array([[1, 0, 0, 1], [0, 1, 0, -2], [0, 0, 1, 0.5], [0, 0, 0, 1]], dtype=np.float32),
        ),
        (
            "rotation-translation",
            np.array([[0, 0, 1, -1], [0, 1, 0, -2], [-1, 0, 0, 6], [0, 0, 0, 1]], dtype=np.float32),
        ),
    ]:
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
        assert splats.positions.shape == (2, 3), pose_name
        assert splats.positions.dtype == np.float32, pose_name
        for i, camera in enumerate(cameras):
            world_homogeneous = np.append(splats.positions[i], np.float32(1.0))
            camera_homogeneous = camera.extrinsics @ world_homogeneous
            np.testing.assert_allclose(
                camera_homogeneous[:3], expected_camera[i], rtol=1e-6, atol=1e-6, err_msg=f"{pose_name}: view {i}"
            )
            assert camera_homogeneous[3] == pytest.approx(1.0), pose_name
        # Compose the actual initializer with the public validator. The original
        # pixel is independently known, and the camera-coordinate oracle above
        # prevents two inverse-direction errors from canceling each other.
        scene = Scene3D(splats=splats, cameras=cameras, rmse=1.0, iteration=0, convergence="max_iterations")
        validator = GeometricValidator()
        source_pixels = np.zeros((2, 2), dtype=np.float32)
        for view_idx in range(len(cameras)):
            assert validator.compute_reprojection_error(scene, view_idx, points_2d=source_pixels) == pytest.approx(
                0.0, abs=1e-6
            ), f"{pose_name}: view {view_idx}"
        np.testing.assert_array_equal(splats.colors, pixel_colors, err_msg=pose_name)
        assert splats.metadata["initialization"] == ("depth" if use_depth_prior else "sfm"), pose_name
        assert backend._model_loaded is False, pose_name

        if use_depth_prior:
            # Pixels without a finite positive depth carry no measurement: with every pixel
            # unmasked they are skipped, never unprojected to (or behind) the camera centre.
            for depth in reconstruction_input.depth_maps:
                depth[0, 1], depth[1, 0], depth[1, 1] = 0.0, np.nan, -3.0
            unmasked = ReconstructionInput(
                images=images, gamma=1.0, cameras=cameras, depth_maps=reconstruction_input.depth_maps, tier="experimental"
            )
            depth_only = backend._initialize_gaussians(unmasked, use_depth_prior=True, use_segmentation=True)
            np.testing.assert_array_equal(depth_only.positions, splats.positions, err_msg=pose_name)
            np.testing.assert_array_equal(depth_only.colors, splats.colors, err_msg=pose_name)
            for depth in unmasked.depth_maps:
                depth[0, 0] = np.inf
            with pytest.raises(ValueError, match="no pixel with a finite positive depth"):
                backend._initialize_gaussians(unmasked, use_depth_prior=True, use_segmentation=True)

            # R-4: without depth maps the fixed-depth fallback runs although the depth
            # prior was requested, and the metadata must name the initializer that ran.
            no_depth = ReconstructionInput(images=images, gamma=1.0, cameras=cameras, masks=masks, tier="experimental")
            fallback = backend._initialize_gaussians(no_depth, use_depth_prior=True, use_segmentation=True)
            assert fallback.metadata["initialization"] == "sfm", pose_name

            # R-4: the subsample above MAX_GAUSSIANS runs before the optimizer seeds the
            # global RNGs, so a seeded backend must draw it from its own seeded generator.
            dense = ReconstructionInput(
                images=[np.full((4, 4, 3), 0.5, dtype=np.float32) for _ in range(2)],
                gamma=1.0,
                cameras=[CameraParams(intrinsics.copy(), extrinsics.copy(), width=4, height=4) for _ in range(2)],
                depth_maps=[np.arange(16, dtype=np.float32).reshape(4, 4) + 1.0 for _ in range(2)],
                tier="experimental",
            )
            monkeypatch.setattr(GaussianBackend, "MAX_GAUSSIANS", 5)
            subsamples = []
            for global_seed in (1, 2):
                np.random.seed(global_seed)
                seeded = GaussianBackend(tier="experimental", device="cpu", optimization_seed=7)
                subsamples.append(seeded._initialize_gaussians(dense, use_depth_prior=True, use_segmentation=False))
            assert subsamples[0].num_gaussians == 5, pose_name
            np.testing.assert_array_equal(subsamples[0].positions, subsamples[1].positions, err_msg=pose_name)
            monkeypatch.undo()
