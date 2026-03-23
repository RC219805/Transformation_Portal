"""Tests for multi-view reconstruction pipeline (Phase 2.3 MVP).

Tests the process_multiview entrypoint with:
- Camera validation (fail-closed policy)
- Tier enforcement
- PLY export
- Provenance sidecar
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch is required for multiview pipeline tests")
pytestmark = pytest.mark.ml

from transformation_portal.core.geometry import CoreCameraParams, MultiViewReconstructionRequest
from transformation_portal.core.geometry.multiview_request import CameraValidationError
from transformation_portal.spatial_ai.orchestration.pipeline import (
    MultiViewReconstructionResult,
    PipelineConfig,
    SpatialAIPipeline,
)
from transformation_portal.spatial_ai.orchestration.error_handler import PipelineError


class TestMultiviewPipelineValidation:
    """Tests for request validation in process_multiview."""

    def _make_cameras(self, count: int, source: str = "explicit") -> list:
        """Create test cameras."""
        return [
            CoreCameraParams(
                fx=800.0, fy=800.0, cx=512.0, cy=384.0,
                width=1024, height=768, source=source
            )
            for _ in range(count)
        ]

    def _make_images(self, count: int) -> list:
        """Create test image arrays."""
        return [
            np.ones((768, 1024, 3), dtype=np.float32) * 0.5
            for _ in range(count)
        ]

    def _make_pipeline(self, tier: str = "apex_research") -> SpatialAIPipeline:
        """Create test pipeline."""
        config = PipelineConfig(
            tier=tier,
            stages=["reconstruction"],
            reconstruction={
                "iterations": 20,  # Very low for fast tests
            },
        )
        return SpatialAIPipeline(config)

    def test_rejects_single_view_via_request(self, tmp_path):
        """Single-view requests are rejected in request validation."""
        cameras = self._make_cameras(1)
        images = self._make_images(1)

        # Request validation should catch this
        with pytest.raises(ValueError, match="at least 2 views"):
            MultiViewReconstructionRequest(
                cameras=cameras,
                images=images,
                tier="apex_research",
            )

    def test_rejects_synthetic_cameras_via_request(self, tmp_path):
        """Synthetic cameras are rejected in request validation."""
        cameras = self._make_cameras(2, source="synthetic")
        images = self._make_images(2)

        with pytest.raises(CameraValidationError, match="verified cameras"):
            MultiViewReconstructionRequest(
                cameras=cameras,
                images=images,
                tier="apex_research",
            )

    def test_rejects_non_research_tier_via_request(self, tmp_path):
        """Non-research tiers are rejected in request validation."""
        cameras = self._make_cameras(2)
        images = self._make_images(2)

        with pytest.raises(ValueError, match="requires research tier"):
            MultiViewReconstructionRequest(
                cameras=cameras,
                images=images,
                tier="standard",
            )


class TestMultiviewPipelineExecution:
    """Integration tests for multi-view reconstruction execution."""

    def _make_cameras(self, count: int, source: str = "explicit") -> list:
        return [
            CoreCameraParams(
                fx=800.0, fy=800.0, cx=512.0, cy=384.0,
                width=64, height=48, source=source  # Small images for tests
            )
            for _ in range(count)
        ]

    def _make_images(self, count: int) -> list:
        return [
            np.random.rand(48, 64, 3).astype(np.float32)  # Small random images
            for _ in range(count)
        ]

    def _make_pipeline(self, iterations: int = 20) -> SpatialAIPipeline:
        config = PipelineConfig(
            tier="apex_research",
            stages=["reconstruction"],
            reconstruction={
                "iterations": iterations,
            },
        )
        return SpatialAIPipeline(config)

    def test_two_view_reconstruction_smoke(self, tmp_path):
        """Two-view reconstruction produces scene and PLY export."""
        pipeline = self._make_pipeline(iterations=20)

        cameras = self._make_cameras(2)
        images = self._make_images(2)

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            tier="apex_research",
            optimization_seed=42,  # Deterministic
        )

        result = pipeline.process_multiview(request, tmp_path)

        # Check result type
        assert isinstance(result, MultiViewReconstructionResult)

        # Check scene properties
        assert result.scene is not None
        assert result.scene.splats.num_gaussians > 0

        # Check PLY export
        assert result.ply_path.exists()
        assert result.ply_path.suffix == ".ply"

        # Check sidecar
        assert result.sidecar_path.exists()
        with open(result.sidecar_path) as f:
            provenance = json.load(f)
        assert provenance["backend"] == "gaussian_splatting"
        assert provenance["tier"] == "apex_research"

    def test_reconstruction_with_depth_priors(self, tmp_path):
        """Reconstruction accepts depth priors from Phase 1."""
        pipeline = self._make_pipeline(iterations=20)

        cameras = self._make_cameras(2)
        images = self._make_images(2)
        depth_maps = [np.random.rand(48, 64).astype(np.float32) * 10 for _ in range(2)]

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            depth_maps=depth_maps,
            tier="apex_research",
            optimization_seed=42,
        )

        result = pipeline.process_multiview(request, tmp_path)

        assert result.scene is not None
        assert result.request_metadata["has_depth_priors"] is True

    def test_reconstruction_saves_summary(self, tmp_path):
        """Reconstruction saves summary JSON when save_intermediates=True."""
        pipeline = self._make_pipeline(iterations=20)

        cameras = self._make_cameras(2)
        images = self._make_images(2)

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            tier="apex_research",
            optimization_seed=42,
        )

        result = pipeline.process_multiview(request, tmp_path, save_intermediates=True)

        summary_path = tmp_path / "reconstruction_summary.json"
        assert summary_path.exists()

        with open(summary_path) as f:
            summary = json.load(f)

        assert "scene" in summary
        assert summary["scene"]["num_gaussians"] > 0
        assert "stages_completed" in summary
        assert "reconstruction" in summary["stages_completed"]

    def test_accepts_synthetic_with_override(self, tmp_path):
        """Synthetic cameras work with explicit override."""
        pipeline = self._make_pipeline(iterations=20)

        cameras = self._make_cameras(2, source="synthetic")
        images = self._make_images(2)

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            tier="apex_research",
            allow_synthetic_cameras=True,
            optimization_seed=42,
        )

        result = pipeline.process_multiview(request, tmp_path)

        assert result.scene is not None
        assert result.request_metadata["allow_synthetic_cameras"] is True


class TestMultiviewDeterminism:
    """Tests for deterministic reconstruction with fixed seed."""

    def _make_cameras(self, count: int) -> list:
        return [
            CoreCameraParams(
                fx=800.0, fy=800.0, cx=32.0, cy=24.0,
                width=64, height=48, source="explicit"
            )
            for _ in range(count)
        ]

    def _make_images(self, count: int, seed: int = 42) -> list:
        np.random.seed(seed)
        return [
            np.random.rand(48, 64, 3).astype(np.float32)
            for _ in range(count)
        ]

    def _make_pipeline(self) -> SpatialAIPipeline:
        config = PipelineConfig(
            tier="apex_research",
            stages=["reconstruction"],
            reconstruction={"iterations": 20},
        )
        return SpatialAIPipeline(config)

    def test_same_seed_produces_same_artifact_presence(self, tmp_path):
        """Same request + seed produces same artifacts."""
        pipeline = self._make_pipeline()

        cameras = self._make_cameras(2)
        images_a = self._make_images(2, seed=100)
        images_b = self._make_images(2, seed=100)  # Same seed, same images

        output_a = tmp_path / "run_a"
        output_b = tmp_path / "run_b"

        request_a = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images_a,
            tier="apex_research",
            optimization_seed=12345,
        )
        request_b = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images_b,
            tier="apex_research",
            optimization_seed=12345,
        )

        result_a = pipeline.process_multiview(request_a, output_a)
        result_b = pipeline.process_multiview(request_b, output_b)

        # Same artifacts present
        assert result_a.ply_path.exists()
        assert result_b.ply_path.exists()
        assert result_a.sidecar_path.exists()
        assert result_b.sidecar_path.exists()

        # Same metadata fields
        with open(result_a.sidecar_path) as f:
            provenance_a = json.load(f)
        with open(result_b.sidecar_path) as f:
            provenance_b = json.load(f)

        assert provenance_a["num_gaussians"] == provenance_b["num_gaussians"]
        assert provenance_a["rmse"] == provenance_b["rmse"]
