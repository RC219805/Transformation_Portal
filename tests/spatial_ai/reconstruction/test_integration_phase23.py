"""Integration tests for Phase 2.3 with Phases 2.1 and 2.2."""

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch is required for reconstruction integration tests")
pytestmark = pytest.mark.ml

from transformation_portal.spatial_ai.reconstruction import CameraParams, GeometricValidator, SceneBuilder


class TestPhase23Integration:
    """Test integration with previous spatial_ai phases."""

    def test_integration_with_depth_maps(self):
        """Test Phase 2.3 integration with Phase 1 (depth estimation)."""
        # Simulate Phase 1 depth output
        depth_maps = [np.random.rand(240, 320).astype(np.float32) * 10 for _ in range(3)]

        # Phase 2.3: Reconstruct with depth priors
        builder = SceneBuilder(tier="apex_research")

        images = [np.random.rand(240, 320, 3).astype(np.float32) for _ in range(3)]
        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = intrinsics[1, 1] = 262.5
        intrinsics[0, 2] = 160.0
        intrinsics[1, 2] = 120.0

        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), 320, 240) for _ in range(3)]

        scene = builder.build_from_arrays(images=images, cameras=cameras, depth_maps=depth_maps, gamma=1.0, iterations=100)

        assert scene.metadata["use_depth_prior"] is True
        assert scene.splats.metadata["initialization"] == "depth"

    def test_integration_with_segmentation(self):
        """Test Phase 2.3 integration with Phase 2.1 (segmentation)."""
        # Simulate Phase 2.1 segmentation masks
        masks = [np.random.rand(240, 320) > 0.5 for _ in range(3)]

        # Phase 2.3: Reconstruct with segmentation
        builder = SceneBuilder(tier="apex_research")

        images = [np.random.rand(240, 320, 3).astype(np.float32) for _ in range(3)]
        intrinsics = np.eye(3, dtype=np.float32)
        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), 320, 240) for _ in range(3)]

        scene = builder.build_from_arrays(images=images, cameras=cameras, masks=masks, gamma=1.0, iterations=100)

        assert scene.metadata["use_segmentation"] is True

    def test_integration_full_pipeline(self):
        """Test full spatial_ai pipeline (Phases 1 + 2.1 + 2.2 + 2.3)."""
        # Phase 1: Depth maps
        depth_maps = [np.random.rand(240, 320).astype(np.float32) * 10 for _ in range(3)]

        # Phase 2.1: Segmentation masks
        masks = [np.random.rand(240, 320) > 0.5 for _ in range(3)]

        # Phase 2.2: PBR material maps
        material_maps = []
        for _ in range(3):
            material_maps.append(
                {
                    "albedo": np.random.rand(240, 320, 3).astype(np.float32),
                    "roughness": np.random.rand(240, 320).astype(np.float32),
                    "metallic": np.random.rand(240, 320).astype(np.float32),
                    "normal": np.random.rand(240, 320, 3).astype(np.float32),
                }
            )

        # Phase 2.3: 3D Reconstruction
        builder = SceneBuilder(tier="apex_research")

        images = [np.random.rand(240, 320, 3).astype(np.float32) for _ in range(3)]
        intrinsics = np.eye(3, dtype=np.float32)
        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), 320, 240) for _ in range(3)]

        scene = builder.build_from_arrays(
            images=images,
            cameras=cameras,
            depth_maps=depth_maps,
            masks=masks,
            material_maps=material_maps,
            gamma=1.0,
            iterations=100,
        )

        # Validate integration
        assert scene.metadata["use_depth_prior"] is True
        assert scene.metadata["use_segmentation"] is True
        assert scene.metadata["use_pbr_textures"] is True
        assert scene.metadata["num_views"] == 3

        # Validate quality
        validator = GeometricValidator()
        results = validator.validate_scene(scene)

        assert results["rmse_pass"] is True
        assert results["quality_grade"] in ["A", "B"]

    def test_gamma_consistency_across_phases(self):
        """Test gamma=1.0 enforcement across all phases."""
        # All phases require gamma=1.0 (SpatialCaptureV1 contract)

        builder = SceneBuilder(tier="apex_research")

        images = [np.random.rand(240, 320, 3).astype(np.float32) for _ in range(2)]
        intrinsics = np.eye(3, dtype=np.float32)
        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), 320, 240) for _ in range(2)]

        # Should succeed with gamma=1.0
        scene = builder.build_from_arrays(images=images, cameras=cameras, gamma=1.0, iterations=100)

        assert scene is not None

        # Should fail with gamma != 1.0
        with pytest.raises(ValueError, match="gamma=1.0"):
            builder.build_from_arrays(images=images, cameras=cameras, gamma=2.2, iterations=100)
