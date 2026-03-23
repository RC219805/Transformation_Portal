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
            CoreCameraParams(
                fx=800.0, fy=800.0, cx=32.0, cy=24.0,
                width=64, height=48, source="explicit"
            )
            for _ in range(count)
        ]

    def _make_images(self, count: int) -> list:
        """Create test image arrays."""
        return [
            np.random.rand(48, 64, 3).astype(np.float32)
            for _ in range(count)
        ]

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
            model_revision="abc123def456",
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
        assert metadata["revision"] == "abc123def456"

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

        preset_path = Path(__file__).parent.parent.parent.parent / \
            "config" / "presets" / "experimental" / "nvdiffrec_reconstruction.yaml"

        if not preset_path.exists():
            preset_path = Path("config/presets/experimental/nvdiffrec_reconstruction.yaml")

        assert preset_path.exists(), f"NVDIFFREC preset not found at {preset_path}"

    def test_preset_structure(self):
        """NVDIFFREC preset has required structure."""
        import yaml
        from pathlib import Path

        preset_path = Path(__file__).parent.parent.parent.parent / \
            "config" / "presets" / "experimental" / "nvdiffrec_reconstruction.yaml"

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
