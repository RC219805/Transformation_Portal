"""Tests for GaussianBackend (Phase 2.3)."""

import os

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch is required for Gaussian backend tests")
pytestmark = pytest.mark.ml

from transformation_portal.spatial_ai.reconstruction import (  # pylint: disable=wrong-import-position
    CameraParams,
    GaussianBackend,
    LicenseRestrictionError,
    ReconstructionInput,
)

# ---------------------------------------------------------------------
# Test Budget Helpers
# ---------------------------------------------------------------------


def _tier_mode() -> str:
    if os.getenv("TP_LONG_TESTS", "").lower() in {"1", "true", "yes"}:
        return "long"
    if os.getenv("CI"):
        return "ci"
    return "standard"


def _is_ci() -> bool:
    return bool(os.getenv("CI", "").strip())


def _require_heavy() -> bool:
    return os.getenv("TP_LONG_TESTS", "").strip().lower() in {"1", "true", "yes"}


def _iterations(default: int) -> int:
    mode = _tier_mode()
    if mode == "long":
        return default
    if mode == "ci":
        return max(5, default // 20)
    return max(20, default // 5)


def _image_size() -> tuple[int, int]:
    mode = _tier_mode()
    if mode == "long":
        return 480, 640
    if mode == "ci":
        return 120, 160
    return 240, 320


# ---------------------------------------------------------------------
# Core Tests
# ---------------------------------------------------------------------


class TestGaussianBackend:
    def test_initialization_valid_tier(self):
        backend = GaussianBackend(tier="apex_research")
        assert backend.tier == "apex_research"
        assert backend.device in {"cuda", "mps", "cpu"}

    def test_initialization_invalid_tier(self):
        with pytest.raises(LicenseRestrictionError):
            GaussianBackend(tier="commercial")

    def test_device_detection(self):
        backend = GaussianBackend(tier="apex_research", device=None)
        assert backend.device in {"cuda", "mps", "cpu"}

    def test_optimization_seed_deterministic(self, seed_all_rngs):
        seed_all_rngs(7)
        backend = GaussianBackend(
            tier="apex_research",
            device="cpu",
            optimization_seed=42,
        )

        h, w = 60, 80
        images = [np.ones((h, w, 3), dtype=np.float32) * 0.5 for _ in range(2)]
        intrinsics = np.eye(3, dtype=np.float32)
        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), w, h) for _ in range(2)]

        ri_a = ReconstructionInput(images=images, gamma=1.0, cameras=cameras, tier="apex_research")
        ri_b = ReconstructionInput(images=images, gamma=1.0, cameras=cameras, tier="apex_research")

        scene_a = backend.reconstruct(ri_a, iterations=20)
        scene_b = backend.reconstruct(ri_b, iterations=20)

        assert np.allclose(scene_a.splats.positions, scene_b.splats.positions, atol=1e-6)

    def test_scene_iteration_tracking(self):
        backend = GaussianBackend(tier="apex_research", device="cpu")

        h, w = 60, 80
        images = [np.ones((h, w, 3), dtype=np.float32) for _ in range(2)]
        intrinsics = np.eye(3, dtype=np.float32)
        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), w, h) for _ in range(2)]

        requested = 20
        ri = ReconstructionInput(images=images, gamma=1.0, cameras=cameras, tier="apex_research")
        scene = backend.reconstruct(ri, iterations=requested)

        assert scene.iteration == scene.metadata["actual_iterations"]
        assert scene.metadata["requested_iterations"] == requested


# ---------------------------------------------------------------------
# Reconstruction Smoke Tests (Bounded)
# ---------------------------------------------------------------------


class TestGaussianBackendReconstruction:
    def _build_basic_input(self, views: int = 3):
        h, w = _image_size()
        images = [np.random.rand(h, w, 3).astype(np.float32) for _ in range(views)]
        intrinsics = np.array(
            [[0.82 * w, 0, w / 2], [0, 0.82 * w, h / 2], [0, 0, 1]],
            dtype=np.float32,
        )
        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), w, h) for _ in range(views)]
        return images, cameras

    def test_multiview_reconstruction_smoke(self, seed_all_rngs):
        seed_all_rngs(42)
        backend = GaussianBackend(tier="apex_research")
        images, cameras = self._build_basic_input(3)

        ri = ReconstructionInput(images=images, gamma=1.0, cameras=cameras, tier="apex_research")

        if _is_ci() and not _require_heavy():
            h, w = _image_size()
            assert len(images) == 3
            assert all(img.shape == (h, w, 3) for img in images)
            assert len(cameras) == 3
            assert all(cam.width == w and cam.height == h for cam in cameras)
            return

        scene = backend.reconstruct(ri, iterations=_iterations(1000))

        assert scene.splats.num_gaussians > 0
        assert len(scene.cameras) == 3

    def test_render_view(self, seed_all_rngs):
        seed_all_rngs(42)
        backend = GaussianBackend(tier="apex_research")
        images, cameras = self._build_basic_input(2)

        ri = ReconstructionInput(images=images, gamma=1.0, cameras=cameras, tier="apex_research")
        scene = backend.reconstruct(ri, iterations=_iterations(100))

        h, w = _image_size()
        novel_camera = CameraParams(cameras[0].intrinsics, np.eye(4, dtype=np.float32), w, h)
        rendered = backend.render_view(scene, novel_camera)

        assert rendered.shape == (h, w, 3)
        assert rendered.dtype == np.float32
        assert np.all((rendered >= 0) & (rendered <= 1))


# ---------------------------------------------------------------------
# License Enforcement
# ---------------------------------------------------------------------


class TestGaussianBackendLicense:
    @pytest.mark.parametrize(
        "invalid",
        ["commercial", "elite", "production"],
    )
    def test_invalid_tiers(self, invalid):
        with pytest.raises(LicenseRestrictionError):
            GaussianBackend(tier=invalid)

    @pytest.mark.parametrize(
        "valid",
        ["apex_research", "experimental"],
    )
    def test_valid_tiers(self, valid):
        backend = GaussianBackend(tier=valid)
        assert backend.tier == valid
