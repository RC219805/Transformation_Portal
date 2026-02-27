"""Integration tests for Phase 2.3 with prior spatial_ai phases."""

import os

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch required for reconstruction integration")
pytestmark = pytest.mark.ml

from transformation_portal.spatial_ai.reconstruction import (  # pylint: disable=wrong-import-position
    CameraParams,
    GeometricValidator,
    SceneBuilder,
)

# ---------------------------------------------------------------------
# Budget Helpers
# ---------------------------------------------------------------------


def _tier_mode() -> str:
    if os.getenv("TP_LONG_TESTS", "").lower() in {"1", "true", "yes"}:
        return "long"
    if _is_ci():
        return "ci"
    return "standard"


def _is_ci() -> bool:
    return os.getenv("CI", "").strip().lower() in {"1", "true", "yes"}


def _require_heavy() -> bool:
    return os.getenv("TP_LONG_TESTS", "").strip().lower() in {"1", "true", "yes"}


def _iterations(default: int = 100) -> int:
    mode = _tier_mode()
    if mode == "long":
        return default
    if mode == "ci":
        return 5
    return 25


def _image_size() -> tuple[int, int]:
    mode = _tier_mode()
    if mode == "long":
        return 240, 320
    if mode == "ci":
        return 120, 160
    return 240, 320


# ---------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------


def test_ci_env_detection_requires_truthy_values(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CI", raising=False)
    assert _is_ci() is False

    monkeypatch.setenv("CI", "false")
    assert _is_ci() is False

    monkeypatch.setenv("CI", "0")
    assert _is_ci() is False

    monkeypatch.setenv("CI", "true")
    assert _is_ci() is True

    monkeypatch.setenv("CI", "1")
    assert _is_ci() is True


class TestPhase23Integration:
    def _build_scene_inputs(self, views: int = 3):
        h, w = _image_size()
        images = [np.random.rand(h, w, 3).astype(np.float32) for _ in range(views)]
        depth = [np.random.rand(h, w).astype(np.float32) * 10 for _ in range(views)]
        masks = [np.random.rand(h, w) > 0.5 for _ in range(views)]

        intrinsics = np.array(
            [[0.82 * w, 0, w / 2], [0, 0.82 * w, h / 2], [0, 0, 1]],
            dtype=np.float32,
        )
        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), w, h) for _ in range(views)]

        return images, depth, masks, cameras

    def test_integration_with_depth(self):
        images, depth, _, cameras = self._build_scene_inputs()

        h, w = _image_size()
        if _is_ci() and not _require_heavy():
            assert len(images) == len(depth) == len(cameras)
            assert all(img.shape == (h, w, 3) for img in images)
            assert all(dm.shape == (h, w) for dm in depth)
            assert all(cam.width == w and cam.height == h for cam in cameras)
            intrinsics = cameras[0].intrinsics
            assert intrinsics.shape == (3, 3)
            assert intrinsics[2, 2] == pytest.approx(1.0)
            return

        builder = SceneBuilder(tier="apex_research")
        scene = builder.build_from_arrays(
            images=images,
            cameras=cameras,
            depth_maps=depth,
            gamma=1.0,
            iterations=_iterations(),
        )

        assert scene.metadata["use_depth_prior"] is True
        assert scene.splats.metadata["initialization"] == "depth"

    def test_integration_with_segmentation(self):
        images, _, masks, cameras = self._build_scene_inputs()

        builder = SceneBuilder(tier="apex_research")
        scene = builder.build_from_arrays(
            images=images,
            cameras=cameras,
            masks=masks,
            gamma=1.0,
            iterations=_iterations(),
        )

        assert scene.metadata["use_segmentation"] is True

    def test_full_pipeline_smoke(self):
        images, depth, masks, cameras = self._build_scene_inputs()

        h, w = _image_size()
        if _is_ci() and not _require_heavy():
            assert len(images) == len(depth) == len(masks) == len(cameras)
            assert all(img.shape == (h, w, 3) for img in images)
            assert all(dm.shape == (h, w) for dm in depth)
            assert all(msk.shape == (h, w) for msk in masks)
            assert all(cam.width == w and cam.height == h for cam in cameras)
            return

        builder = SceneBuilder(tier="apex_research")
        scene = builder.build_from_arrays(
            images=images,
            cameras=cameras,
            depth_maps=depth,
            masks=masks,
            gamma=1.0,
            iterations=_iterations(),
        )

        assert scene.metadata["num_views"] == len(images)

        validator = GeometricValidator()
        results = validator.validate_scene(scene)

        assert isinstance(results.get("rmse_pass"), bool)
        assert results.get("quality_grade") in {"A", "B", "C", "D"}
        if results["rmse_pass"]:
            assert results["quality_grade"] in {"A", "B"}
        else:
            assert results["quality_grade"] in {"C", "D"}

        coverage = results.get("coverage")
        assert isinstance(coverage, dict)
        assert {"mean_points_per_view", "min_points_per_view", "max_points_per_view", "coverage_std"} <= set(coverage)

    def test_gamma_contract(self):
        images, _, _, cameras = self._build_scene_inputs(2)

        builder = SceneBuilder(tier="apex_research")

        # Valid gamma
        scene = builder.build_from_arrays(
            images=images,
            cameras=cameras,
            gamma=1.0,
            iterations=_iterations(),
        )
        assert scene is not None

        # Invalid gamma
        with pytest.raises(ValueError, match="gamma=1.0"):
            builder.build_from_arrays(
                images=images,
                cameras=cameras,
                gamma=2.2,
                iterations=_iterations(),
            )
