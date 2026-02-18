"""Tests for GaussianBackend (Phase 2.3)."""

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch is required for Gaussian backend tests")

from transformation_portal.spatial_ai.reconstruction import (
    CameraParams,
    GaussianBackend,
    LicenseRestrictionError,
    ReconstructionInput,
)


@pytest.mark.slow
class TestGaussianBackend:
    """Test Gaussian Splatting backend."""

    def test_initialization_valid_tier(self):
        """Test backend initialization with valid tier."""
        backend = GaussianBackend(tier="apex_research")
        assert backend.tier == "apex_research"
        assert backend.device in ["cuda", "mps", "cpu"]

    def test_initialization_invalid_tier(self):
        """Test backend initialization with invalid tier (license restriction)."""
        with pytest.raises(LicenseRestrictionError, match="3D Gaussian Splatting requires research tier"):
            GaussianBackend(tier="commercial")

    def test_initialization_experimental_tier(self):
        """Test backend initialization with experimental tier."""
        backend = GaussianBackend(tier="experimental")
        assert backend.tier == "experimental"

    def test_device_detection(self):
        """Test automatic device detection."""
        backend = GaussianBackend(tier="apex_research", device=None)
        assert backend.device in ["cuda", "mps", "cpu"]

    def test_explicit_device(self):
        """Test explicit device specification."""
        backend = GaussianBackend(tier="apex_research", device="cpu")
        assert backend.device == "cpu"

    def test_optimization_seed_defaults_to_none(self):
        """Test deterministic optimization is opt-in by default."""
        backend = GaussianBackend(tier="apex_research")
        assert backend.optimization_seed is None

    def test_optimization_seed_enables_deterministic_runs(self):
        """Test deterministic optimization when seed is explicitly configured."""
        backend = GaussianBackend(tier="apex_research", device="cpu", optimization_seed=42)

        images = [np.ones((60, 80, 3), dtype=np.float32) * 0.5 for _ in range(2)]
        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = intrinsics[1, 1] = 100.0
        intrinsics[0, 2] = 40.0
        intrinsics[1, 2] = 30.0
        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), 80, 60) for _ in range(2)]

        reconstruction_input_a = ReconstructionInput(
            images=[img.copy() for img in images],
            gamma=1.0,
            cameras=cameras,
            tier="apex_research",
        )
        reconstruction_input_b = ReconstructionInput(
            images=[img.copy() for img in images],
            gamma=1.0,
            cameras=cameras,
            tier="apex_research",
        )

        scene_a = backend.reconstruct(reconstruction_input_a, iterations=20)
        scene_b = backend.reconstruct(reconstruction_input_b, iterations=20)

        assert np.allclose(scene_a.splats.positions, scene_b.splats.positions, atol=1e-6)
        assert np.allclose(scene_a.splats.colors, scene_b.splats.colors, atol=1e-6)
        assert np.allclose(scene_a.splats.scales, scene_b.splats.scales, atol=1e-6)
        assert np.allclose(scene_a.splats.opacities, scene_b.splats.opacities, atol=1e-6)

    def test_model_lazy_loading(self):
        """Test that model is not loaded on initialization."""
        backend = GaussianBackend(tier="apex_research")
        assert not backend._model_loaded

    def test_reconstruct_multiview(self):
        """Test multi-view reconstruction."""
        backend = GaussianBackend(tier="apex_research")

        # Create multi-view input
        images = [np.random.rand(480, 640, 3).astype(np.float32) for _ in range(3)]

        intrinsics = np.array([[525.0, 0, 320.0], [0, 525.0, 240.0], [0, 0, 1.0]], dtype=np.float32)

        cameras = []
        for i in range(3):
            extrinsics = np.eye(4, dtype=np.float32)
            extrinsics[0, 3] = i * 0.5  # Shift camera along x-axis
            cameras.append(CameraParams(intrinsics, extrinsics, 640, 480))

        reconstruction_input = ReconstructionInput(images=images, gamma=1.0, cameras=cameras, tier="apex_research")

        # Reconstruct
        scene = backend.reconstruct(reconstruction_input, iterations=1000)

        # Validate scene
        assert scene is not None
        assert scene.splats.num_gaussians > 0
        assert scene.rmse < 0.02  # Within target
        assert len(scene.cameras) == 3

    def test_reconstruct_with_depth_prior(self):
        """Test reconstruction with depth priors."""
        backend = GaussianBackend(tier="apex_research")

        images = [np.random.rand(480, 640, 3).astype(np.float32) for _ in range(2)]
        depth_maps = [np.random.rand(480, 640).astype(np.float32) * 10 for _ in range(2)]  # 0-10m depth

        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = intrinsics[1, 1] = 525.0
        intrinsics[0, 2] = 320.0
        intrinsics[1, 2] = 240.0

        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), 640, 480) for _ in range(2)]

        reconstruction_input = ReconstructionInput(
            images=images, gamma=1.0, cameras=cameras, depth_maps=depth_maps, tier="apex_research"
        )

        scene = backend.reconstruct(reconstruction_input, use_depth_prior=True)

        assert "use_depth_prior" in scene.metadata
        assert scene.metadata["use_depth_prior"] is True

    def test_reconstruct_with_segmentation(self):
        """Test reconstruction with segmentation masks."""
        backend = GaussianBackend(tier="apex_research")

        images = [np.random.rand(480, 640, 3).astype(np.float32) for _ in range(2)]
        masks = [np.random.rand(480, 640) > 0.5 for _ in range(2)]

        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = intrinsics[1, 1] = 525.0
        intrinsics[0, 2] = 320.0
        intrinsics[1, 2] = 240.0

        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), 640, 480) for _ in range(2)]

        reconstruction_input = ReconstructionInput(
            images=images, gamma=1.0, cameras=cameras, masks=masks, tier="apex_research"
        )

        scene = backend.reconstruct(reconstruction_input, use_segmentation=True)

        assert "use_segmentation" in scene.metadata
        assert scene.metadata["use_segmentation"] is True

    def test_reconstruct_performance(self):
        """Test reconstruction performance targets."""
        backend = GaussianBackend(tier="apex_research")

        images = [np.random.rand(240, 320, 3).astype(np.float32) for _ in range(3)]  # Smaller images

        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = intrinsics[1, 1] = 262.5
        intrinsics[0, 2] = 160.0
        intrinsics[1, 2] = 120.0

        cameras = []
        for i in range(3):
            extrinsics = np.eye(4, dtype=np.float32)
            extrinsics[0, 3] = i * 0.3
            cameras.append(CameraParams(intrinsics, extrinsics, 320, 240))

        reconstruction_input = ReconstructionInput(images=images, gamma=1.0, cameras=cameras, tier="apex_research")

        # Short iteration count for testing
        scene = backend.reconstruct(reconstruction_input, iterations=100)

        # Check performance metadata
        assert "elapsed_seconds" in scene.metadata
        assert scene.metadata["elapsed_seconds"] < 60  # Should be fast for mock

    def test_render_view(self):
        """Test novel view rendering."""
        backend = GaussianBackend(tier="apex_research")

        # Create simple scene
        images = [np.random.rand(240, 320, 3).astype(np.float32) for _ in range(2)]

        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = intrinsics[1, 1] = 262.5
        intrinsics[0, 2] = 160.0
        intrinsics[1, 2] = 120.0

        cameras = []
        for i in range(2):
            extrinsics = np.eye(4, dtype=np.float32)
            cameras.append(CameraParams(intrinsics, extrinsics, 320, 240))

        reconstruction_input = ReconstructionInput(images=images, gamma=1.0, cameras=cameras, tier="apex_research")

        scene = backend.reconstruct(reconstruction_input, iterations=100)

        # Render novel view
        novel_camera = CameraParams(intrinsics, np.eye(4, dtype=np.float32), 320, 240)
        rendered = backend.render_view(scene, novel_camera)

        assert rendered.shape == (240, 320, 3)
        assert rendered.dtype == np.float32
        assert np.all(rendered >= 0) and np.all(rendered <= 1)

    def test_convergence_status(self):
        """Test convergence status tracking."""
        backend = GaussianBackend(tier="apex_research")

        images = [np.random.rand(240, 320, 3).astype(np.float32) for _ in range(2)]

        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = intrinsics[1, 1] = 262.5
        intrinsics[0, 2] = 160.0
        intrinsics[1, 2] = 120.0

        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), 320, 240) for _ in range(2)]

        reconstruction_input = ReconstructionInput(images=images, gamma=1.0, cameras=cameras, tier="apex_research")

        scene = backend.reconstruct(reconstruction_input, iterations=50)  # Reduced for speed

        assert scene.convergence in ["converged", "improving", "stalled"]
        assert "optimized" in scene.splats.metadata
        assert scene.splats.metadata["optimized"] is True

    def test_optimization_reduces_loss(self):
        """Test that optimization actually reduces loss over iterations."""
        backend = GaussianBackend(tier="apex_research")

        # Create simple synthetic scene
        images = [np.ones((120, 160, 3), dtype=np.float32) * 0.5 for _ in range(2)]  # Gray images

        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = intrinsics[1, 1] = 200.0
        intrinsics[0, 2] = 80.0
        intrinsics[1, 2] = 60.0

        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), 160, 120) for _ in range(2)]

        reconstruction_input = ReconstructionInput(images=images, gamma=1.0, cameras=cameras, tier="apex_research")

        scene = backend.reconstruct(reconstruction_input, iterations=100)

        # Check that loss history exists
        assert "loss_history" in scene.splats.metadata
        loss_history = scene.splats.metadata["loss_history"]

        # Loss should decrease over time (first loss > last loss)
        if len(loss_history) > 10:
            initial_loss = np.mean(loss_history[:5])
            final_loss = np.mean(loss_history[-5:])
            assert final_loss < initial_loss, "Loss should decrease during optimization"

    def test_gaussian_initialization_depth_guided(self):
        """Test depth-guided Gaussian initialization."""
        backend = GaussianBackend(tier="apex_research")

        images = [np.random.rand(240, 320, 3).astype(np.float32) for _ in range(2)]
        depth_maps = [np.ones((240, 320), dtype=np.float32) * 5.0 for _ in range(2)]  # 5m constant depth

        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = intrinsics[1, 1] = 262.5
        intrinsics[0, 2] = 160.0
        intrinsics[1, 2] = 120.0

        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), 320, 240) for _ in range(2)]

        reconstruction_input = ReconstructionInput(
            images=images, gamma=1.0, cameras=cameras, depth_maps=depth_maps, tier="apex_research"
        )

        scene = backend.reconstruct(reconstruction_input, use_depth_prior=True)

        # Check that Gaussians were initialized
        assert scene.splats.num_gaussians > 0
        assert "initialization" in scene.splats.metadata
        assert scene.splats.metadata["initialization"] == "depth"

    def test_gaussian_initialization_sfm_fallback(self):
        """Test structure-from-motion initialization (fallback)."""
        backend = GaussianBackend(tier="apex_research")

        images = [np.random.rand(240, 320, 3).astype(np.float32) for _ in range(2)]

        intrinsics = np.eye(3, dtype=np.float32)
        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), 320, 240) for _ in range(2)]

        reconstruction_input = ReconstructionInput(images=images, gamma=1.0, cameras=cameras, tier="apex_research")

        scene = backend.reconstruct(reconstruction_input, use_depth_prior=False)

        # Check SfM initialization
        assert scene.splats.num_gaussians > 0
        assert "initialization" in scene.splats.metadata
        assert scene.splats.metadata["initialization"] == "sfm"


class TestGaussianBackendLicenseEnforcement:
    """Test license restriction enforcement."""

    @pytest.mark.parametrize(
        "invalid_tier",
        [
            "commercial",
            "apex",
            "elite",
            "max_quality",
            "production",
        ],
    )
    def test_invalid_tiers_rejected(self, invalid_tier):
        """Test that invalid tiers are rejected."""
        with pytest.raises(LicenseRestrictionError, match="3D Gaussian Splatting requires research tier"):
            GaussianBackend(tier=invalid_tier)

    @pytest.mark.parametrize(
        "valid_tier",
        [
            "apex_research",
            "apex_research_ultra",
            "experimental",
        ],
    )
    def test_valid_tiers_accepted(self, valid_tier):
        """Test that valid research tiers are accepted."""
        backend = GaussianBackend(tier=valid_tier)
        assert backend.tier == valid_tier

    def test_license_error_message_contains_url(self):
        """Test that license error includes Inria URL."""
        with pytest.raises(LicenseRestrictionError) as exc_info:
            GaussianBackend(tier="commercial")

        assert "github.com/graphdeco-inria/gaussian-splatting" in str(exc_info.value)

    def test_reconstruction_input_tier_enforcement(self):
        """Test tier enforcement in ReconstructionInput."""
        images = [np.random.rand(240, 320, 3).astype(np.float32) for _ in range(2)]
        intrinsics = np.eye(3, dtype=np.float32)
        cameras = [CameraParams(intrinsics, np.eye(4, dtype=np.float32), 320, 240) for _ in range(2)]

        with pytest.raises(LicenseRestrictionError):
            ReconstructionInput(images=images, gamma=1.0, cameras=cameras, tier="commercial")
