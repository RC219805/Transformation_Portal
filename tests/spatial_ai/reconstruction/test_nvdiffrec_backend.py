"""Tests for NVDIFFREC backend (Phase 2.3 PR B).

Tests tier enforcement, environment validation, and license compliance
for the NVDIFFREC research-only backend.

Note: Most tests skip preflight checks since CUDA/nvdiffrast may not be
available in CI. Use skip_preflight=True for unit tests.
"""

import numpy as np
import pytest

from transformation_portal.core.geometry import CoreCameraParams, MultiViewReconstructionRequest
from transformation_portal.spatial_ai.reconstruction.nvdiffrec_backend import (
    NVDiffRecBackend,
    NVDiffRecConfig,
    NVDiffRecEnvironmentError,
    NVDiffRecLicenseError,
)

pytestmark = pytest.mark.unit


class TestNVDiffRecBackendTierEnforcement:
    """Tests for license tier enforcement."""

    def test_reject_standard_tier(self):
        """Standard tier rejected (research license required)."""
        with pytest.raises(NVDiffRecLicenseError, match="research tier"):
            NVDiffRecBackend(tier="standard", skip_preflight=True)

    def test_reject_commercial_tier(self):
        """Commercial tier rejected (research license required)."""
        with pytest.raises(NVDiffRecLicenseError, match="research tier"):
            NVDiffRecBackend(tier="commercial", skip_preflight=True)

    def test_reject_invalid_tier(self):
        """Invalid tier rejected."""
        with pytest.raises(NVDiffRecLicenseError, match="research tier"):
            NVDiffRecBackend(tier="production", skip_preflight=True)

    def test_accept_apex_research_tier(self):
        """apex_research tier accepted."""
        backend = NVDiffRecBackend(
            tier="apex_research",
            skip_preflight=True,
            model_revision="NEEDS_VERIFICATION_placeholder",
        )
        assert backend.tier == "apex_research"

    def test_accept_apex_research_ultra_tier(self):
        """apex_research_ultra tier accepted."""
        backend = NVDiffRecBackend(
            tier="apex_research_ultra",
            skip_preflight=True,
            model_revision="NEEDS_VERIFICATION_placeholder",
        )
        assert backend.tier == "apex_research_ultra"

    def test_accept_experimental_tier(self):
        """experimental tier accepted."""
        backend = NVDiffRecBackend(
            tier="experimental",
            skip_preflight=True,
            model_revision="NEEDS_VERIFICATION_placeholder",
        )
        assert backend.tier == "experimental"


class TestNVDiffRecBackendRevisionEnforcement:
    """Tests for model revision pinning enforcement."""

    def test_reject_placeholder_revision(self):
        """Placeholder revision rejected in production mode."""
        with pytest.raises(ValueError, match="pinned model revision"):
            NVDiffRecBackend(
                tier="apex_research",
                model_revision="NEEDS_VERIFICATION_0000000000000000000000",
                skip_preflight=False,  # Production mode
            )

    def test_accept_placeholder_with_skip_preflight(self):
        """Placeholder revision accepted with skip_preflight=True."""
        backend = NVDiffRecBackend(
            tier="apex_research",
            model_revision="NEEDS_VERIFICATION_0000000000000000000000",
            skip_preflight=True,
        )
        assert "NEEDS_VERIFICATION" in backend.model_revision

    def test_accept_pinned_revision(self):
        """Pinned revision accepted."""
        backend = NVDiffRecBackend(
            tier="apex_research",
            model_revision="abc123def456789",
            skip_preflight=True,
        )
        assert backend.model_revision == "abc123def456789"


class TestNVDiffRecBackendDeviceEnforcement:
    """Tests for CUDA device requirement."""

    def test_reject_cpu_device(self):
        """CPU device rejected (CUDA required)."""
        with pytest.raises(NVDiffRecEnvironmentError, match="CUDA"):
            NVDiffRecBackend(
                tier="apex_research",
                device="cpu",
                model_revision="abc123",
                skip_preflight=False,
            )

    def test_reject_mps_device(self):
        """MPS device rejected (CUDA required)."""
        with pytest.raises(NVDiffRecEnvironmentError, match="CUDA"):
            NVDiffRecBackend(
                tier="apex_research",
                device="mps",
                model_revision="abc123",
                skip_preflight=False,
            )

    def test_accept_cpu_with_skip_preflight(self):
        """CPU device accepted with skip_preflight=True."""
        backend = NVDiffRecBackend(
            tier="apex_research",
            device="cpu",
            model_revision="abc123",
            skip_preflight=True,
        )
        assert backend.device == "cpu"


class TestNVDiffRecConfig:
    """Tests for NVDIFFREC configuration."""

    def test_default_config(self):
        """Default config has expected values."""
        config = NVDiffRecConfig()

        assert config.iterations == 500
        assert config.batch_size == 1
        assert config.learning_rate == 0.01
        assert config.dmtet_resolution == 128
        assert config.texture_resolution == 1024
        assert config.optimization_seed is None

    def test_custom_config(self):
        """Custom config overrides defaults."""
        config = NVDiffRecConfig(
            iterations=1000,
            batch_size=2,
            optimization_seed=42,
        )

        assert config.iterations == 1000
        assert config.batch_size == 2
        assert config.optimization_seed == 42


class TestNVDiffRecBackendProvenance:
    """Tests for provenance information."""

    def test_provenance_contains_license_info(self):
        """Provenance includes license class and notice."""
        backend = NVDiffRecBackend(
            tier="apex_research",
            model_revision="abc123def456",
            skip_preflight=True,
        )

        provenance = backend.get_provenance()

        assert provenance["backend"] == "nvdiffrec"
        assert provenance["license_class"] == "research_only"
        assert "NVIDIA Source Code License" in provenance["license_notice"]
        assert provenance["tier"] == "apex_research"

    def test_provenance_contains_revision(self):
        """Provenance includes model revision."""
        backend = NVDiffRecBackend(
            tier="apex_research",
            model_revision="abc123def456789",
            skip_preflight=True,
        )

        provenance = backend.get_provenance()

        assert provenance["revision"] == "abc123def456789"
        assert provenance["repo_id"] == "nvidia/nvdiffrec"


class TestNVDiffRecBackendReconstruction:
    """Tests for reconstruction functionality (mock mode).

    NOTE: These tests require torch and are marked as ML tests.
    """

    @pytest.fixture(autouse=True)
    def skip_without_torch(self):
        """Skip if torch not available."""
        pytest.importorskip("torch", reason="torch required for reconstruction tests")

    def _make_cameras(self, count: int) -> list:
        """Create test cameras."""
        return [
            CoreCameraParams(fx=800.0, fy=800.0, cx=32.0, cy=24.0, width=64, height=48, source="explicit")
            for _ in range(count)
        ]

    def _make_images(self, count: int, seed: int = 42) -> list:
        """Create test image arrays with fixed seed for determinism."""
        np.random.seed(seed)
        return [np.random.rand(48, 64, 3).astype(np.float32) for _ in range(count)]

    @pytest.mark.ml
    def test_reconstruct_with_materials_returns_scene(self):
        """reconstruct_with_materials returns Scene3D."""
        backend = NVDiffRecBackend(
            tier="apex_research",
            model_revision="NEEDS_VERIFICATION_placeholder",
            skip_preflight=True,
        )

        cameras = self._make_cameras(2)
        images = self._make_images(2)

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            tier="apex_research",
        )

        config = NVDiffRecConfig(iterations=10, optimization_seed=42)
        scene = backend.reconstruct_with_materials(request, config)

        assert scene is not None
        assert scene.splats.num_gaussians > 0
        assert scene.metadata["backend"] == "nvdiffrec"
        assert scene.metadata["license_class"] == "research_only"

    @pytest.mark.ml
    def test_reconstruction_metadata_completeness(self):
        """Reconstruction metadata contains required fields."""
        backend = NVDiffRecBackend(
            tier="apex_research",
            model_revision="NEEDS_VERIFICATION_abc123def456",
            skip_preflight=True,
        )

        cameras = self._make_cameras(2)
        images = self._make_images(2)

        request = MultiViewReconstructionRequest(
            cameras=cameras,
            images=images,
            tier="apex_research",
        )

        config = NVDiffRecConfig(iterations=10, optimization_seed=42)
        scene = backend.reconstruct_with_materials(request, config)

        metadata = scene.metadata

        # Required provenance fields
        assert metadata["backend"] == "nvdiffrec"
        assert metadata["license_class"] == "research_only"
        assert metadata["tier"] == "apex_research"
        assert metadata["repo_id"] == "nvidia/nvdiffrec"
        assert metadata["revision"] == "NEEDS_VERIFICATION_abc123def456"

        # Optimization fields
        assert metadata["num_views"] == 2
        assert metadata["requested_iterations"] == 10
        assert metadata["actual_iterations"] == 10
        assert metadata["optimization_seed"] == 42

        # Material flag
        assert metadata["has_materials"] is True


class TestNVDiffRecPreset:
    """Tests for NVDIFFREC preset file."""

    def test_preset_file_exists(self):
        """NVDIFFREC preset file exists."""
        from pathlib import Path

        preset_path = (
            Path(__file__).parent.parent.parent.parent
            / "config"
            / "presets"
            / "experimental"
            / "nvdiffrec_reconstruction.yaml"
        )

        if not preset_path.exists():
            preset_path = Path("config/presets/experimental/nvdiffrec_reconstruction.yaml")

        assert preset_path.exists(), f"NVDIFFREC preset not found at {preset_path}"

    def test_preset_structure(self):
        """NVDIFFREC preset has required structure."""
        from pathlib import Path

        import yaml

        preset_path = (
            Path(__file__).parent.parent.parent.parent
            / "config"
            / "presets"
            / "experimental"
            / "nvdiffrec_reconstruction.yaml"
        )

        if not preset_path.exists():
            preset_path = Path("config/presets/experimental/nvdiffrec_reconstruction.yaml")

        with open(preset_path) as f:
            preset = yaml.safe_load(f)

        # Required fields
        assert preset["tier"] == "apex_research"
        assert preset["license_restriction"] == "research_only"
        assert preset["backend"]["type"] == "nvdiffrec"
        assert preset["backend"]["device"] == "cuda"

        # License notice is critical
        assert "license_notice" in preset
        assert "NVIDIA Source Code License" in preset["license_notice"]
        assert "non-commercial" in preset["license_notice"].lower()


class TestNVDiffRecProductionLoadPath:
    """Tests for the production _load_model() HuggingFace download path."""

    def test_load_model_raises_on_network_failure_with_real_revision(self):
        """Production _load_model() raises RuntimeError when HF download fails.

        In CI there is no real HuggingFace access and no actual NVDiffRec repo
        at 'nvidia/nvdiffrec'. Providing a non-placeholder revision triggers the
        real download path, which must raise RuntimeError (not NotImplementedError).
        """
        backend = NVDiffRecBackend(
            tier="apex_research",
            model_revision="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",  # fake but non-placeholder
            skip_preflight=True,
        )
        with pytest.raises(RuntimeError, match="NVDiffRec model loading failed"):
            backend._load_model()

    def test_load_model_mock_path_sets_model_to_none(self):
        """Placeholder revision keeps _model=None (mock mode)."""
        backend = NVDiffRecBackend(
            tier="apex_research",
            model_revision="NEEDS_VERIFICATION_placeholder",
            skip_preflight=True,
        )
        backend._load_model()
        assert backend._model is None
        assert backend._model_loaded is True

    def test_load_model_idempotent(self):
        """Calling _load_model() twice does not re-run the load logic."""
        backend = NVDiffRecBackend(
            tier="apex_research",
            model_revision="NEEDS_VERIFICATION_placeholder",
            skip_preflight=True,
        )
        backend._load_model()
        backend._load_model()  # second call must be a no-op
        assert backend._model_loaded is True


class TestNVDiffRecMockPathBehaviour:
    """Verify mock-path reconstruction after the production-path split."""

    @pytest.fixture(autouse=True)
    def skip_without_torch(self):
        pytest.importorskip("torch", reason="torch required for reconstruction tests")

    def _make_request(self, num_views: int = 2) -> "MultiViewReconstructionRequest":
        cameras = [
            CoreCameraParams(fx=400.0, fy=400.0, cx=32.0, cy=24.0, width=64, height=48, source="explicit")
            for _ in range(num_views)
        ]
        rng = np.random.default_rng(99)
        images = [rng.random((48, 64, 3)).astype(np.float32) for _ in range(num_views)]
        return MultiViewReconstructionRequest(cameras=cameras, images=images, tier="apex_research")

    @pytest.mark.ml
    def test_mock_convergence_is_max_iterations(self):
        """Mock path sets convergence='max_iterations' (valid Literal value)."""
        backend = NVDiffRecBackend(
            tier="apex_research",
            model_revision="NEEDS_VERIFICATION_placeholder",
            skip_preflight=True,
        )
        config = NVDiffRecConfig(iterations=5, optimization_seed=7)
        scene = backend.reconstruct_with_materials(self._make_request(), config)
        assert scene.convergence == "max_iterations"

    @pytest.mark.ml
    def test_mock_output_is_deterministic_with_seed(self):
        """Same seed produces identical mock output."""
        backend = NVDiffRecBackend(
            tier="apex_research",
            model_revision="NEEDS_VERIFICATION_placeholder",
            skip_preflight=True,
        )
        config = NVDiffRecConfig(iterations=5, optimization_seed=42)
        request = self._make_request()
        scene_a = backend.reconstruct_with_materials(request, config)
        scene_b = backend.reconstruct_with_materials(request, config)
        np.testing.assert_array_equal(scene_a.splats.positions, scene_b.splats.positions)


class TestNVDiffRecBuildMvpMatrix:
    """Tests for the _build_mvp_matrix helper."""

    def test_returns_4x4_float32(self):
        """_build_mvp_matrix returns a (4, 4) float32 array."""
        backend = NVDiffRecBackend(
            tier="apex_research",
            model_revision="NEEDS_VERIFICATION_placeholder",
            skip_preflight=True,
        )
        cam = CoreCameraParams(fx=800.0, fy=800.0, cx=320.0, cy=240.0, width=640, height=480, source="explicit")
        mvp = backend._build_mvp_matrix(cam)
        assert mvp.shape == (4, 4)
        assert mvp.dtype == np.float32

    def test_focal_length_reflected_in_matrix(self):
        """Focal length and principal point appear in the correct matrix entries."""
        backend = NVDiffRecBackend(
            tier="apex_research",
            model_revision="NEEDS_VERIFICATION_placeholder",
            skip_preflight=True,
        )
        cam = CoreCameraParams(fx=400.0, fy=500.0, cx=200.0, cy=150.0, width=400, height=300, source="explicit")
        mvp = backend._build_mvp_matrix(cam)
        # proj[0,0] = 2*fx/w = 2*400/400 = 2.0
        assert abs(mvp[0, 0] - 2.0) < 1e-5
        # proj[1,1] = 2*fy/h = 2*500/300 ≈ 3.333
        assert abs(mvp[1, 1] - (2.0 * 500.0 / 300.0)) < 1e-5
        # bottom-left entry is -1 (perspective divide)
        assert mvp[3, 2] == -1.0
