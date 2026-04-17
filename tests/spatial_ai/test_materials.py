"""Tests for spatial_ai materials module (Phase 5 coverage).

Tests for:
- MaterialInput contract validation
- PBRTextures contract validation
- MaterialProperties validation
- MaterialGenerationConfig validation
- BackendDecision data structures

All tests use mocks - no ML model downloads or GPU requirements.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transformation_portal.spatial_ai.materials.contracts import (
    AvailabilityState,
    BackendDecision,
    MaterialGenerationConfig,
    MaterialInput,
    MaterialProperties,
    PBRGenerationMetadata,
    PBRTextures,
)

pytestmark = [pytest.mark.unit, pytest.mark.ml]


@pytest.fixture
def linear_image():
    """Create a linear RGB image (gamma=1.0) for testing."""
    return np.random.rand(512, 512, 3).astype(np.float32)


@pytest.fixture
def sample_mask():
    """Create a sample boolean mask."""
    mask = np.zeros((512, 512), dtype=bool)
    mask[100:400, 100:400] = True
    return mask


@pytest.fixture
def sample_depth():
    """Create a sample depth map."""
    depth = np.random.rand(512, 512).astype(np.float32) * 10.0  # 0-10 meters
    return depth


@pytest.fixture
def valid_pbr_textures():
    """Create valid PBR textures for testing."""
    H, W = 512, 512
    return PBRTextures(
        albedo=np.random.rand(H, W, 3).astype(np.float32),
        normal=(np.random.rand(H, W, 3).astype(np.float32) * 2 - 1),  # [-1, 1]
        roughness=np.random.rand(H, W).astype(np.float32),
        metallic=np.random.rand(H, W).astype(np.float32),
        ambient_occlusion=np.random.rand(H, W).astype(np.float32),
    )


class TestMaterialInput:
    """Test MaterialInput contract validation."""

    def test_valid_basic_input(self, linear_image):
        """Test valid basic input."""
        mat_input = MaterialInput(
            image=linear_image,
            gamma=1.0,
        )

        assert mat_input.gamma == 1.0
        assert mat_input.image.dtype == np.float32
        assert mat_input.mask is None
        assert mat_input.depth is None

    def test_valid_input_with_mask(self, linear_image, sample_mask):
        """Test valid input with mask."""
        mat_input = MaterialInput(
            image=linear_image,
            gamma=1.0,
            mask=sample_mask,
        )

        assert mat_input.mask is not None
        assert mat_input.mask.dtype == bool

    def test_valid_input_with_depth(self, linear_image, sample_depth):
        """Test valid input with depth."""
        mat_input = MaterialInput(
            image=linear_image,
            gamma=1.0,
            depth=sample_depth,
        )

        assert mat_input.depth is not None
        assert mat_input.depth.dtype == np.float32

    def test_valid_input_with_material_hint(self, linear_image):
        """Test valid input with material hint."""
        for hint in ["wood", "stone", "metal", "glass", "fabric", "concrete", "leather", "ceramic"]:
            mat_input = MaterialInput(
                image=linear_image,
                gamma=1.0,
                material_hint=hint,
            )
            assert mat_input.material_hint == hint

    def test_invalid_gamma_raises(self, linear_image):
        """Test that non-linear gamma is rejected."""
        with pytest.raises(ValueError, match="gamma=1.0"):
            MaterialInput(
                image=linear_image,
                gamma=2.2,
            )

    def test_invalid_image_dtype_raises(self):
        """Test that non-float32 image is rejected."""
        uint8_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="float32"):
            MaterialInput(
                image=uint8_image,
                gamma=1.0,
            )

    def test_invalid_image_shape_raises(self):
        """Test that invalid image shape is rejected."""
        grayscale = np.random.rand(512, 512).astype(np.float32)

        with pytest.raises(ValueError, match="\\(H, W, 3\\)"):
            MaterialInput(
                image=grayscale,
                gamma=1.0,
            )

    def test_invalid_mask_dtype_raises(self, linear_image):
        """Test that non-bool mask is rejected."""
        float_mask = np.random.rand(512, 512).astype(np.float32)

        with pytest.raises(ValueError, match="bool"):
            MaterialInput(
                image=linear_image,
                gamma=1.0,
                mask=float_mask,
            )

    def test_invalid_mask_shape_raises(self, linear_image):
        """Test that mismatched mask shape is rejected."""
        wrong_size_mask = np.zeros((256, 256), dtype=bool)

        with pytest.raises(ValueError, match="must match image"):
            MaterialInput(
                image=linear_image,
                gamma=1.0,
                mask=wrong_size_mask,
            )

    def test_invalid_depth_dtype_raises(self, linear_image):
        """Test that non-float32 depth is rejected."""
        uint16_depth = np.random.randint(0, 65535, (512, 512), dtype=np.uint16)

        with pytest.raises(ValueError, match="float32"):
            MaterialInput(
                image=linear_image,
                gamma=1.0,
                depth=uint16_depth,
            )

    def test_invalid_depth_shape_raises(self, linear_image):
        """Test that mismatched depth shape is rejected."""
        wrong_size_depth = np.random.rand(256, 256).astype(np.float32)

        with pytest.raises(ValueError, match="must match image"):
            MaterialInput(
                image=linear_image,
                gamma=1.0,
                depth=wrong_size_depth,
            )

    def test_negative_depth_raises(self, linear_image):
        """Test that negative depth values are rejected."""
        negative_depth = np.random.rand(512, 512).astype(np.float32) - 0.5

        with pytest.raises(ValueError, match="non-negative"):
            MaterialInput(
                image=linear_image,
                gamma=1.0,
                depth=negative_depth,
            )

    def test_invalid_material_hint_raises(self, linear_image):
        """Test that invalid material hint is rejected."""
        with pytest.raises(ValueError, match="must be one of"):
            MaterialInput(
                image=linear_image,
                gamma=1.0,
                material_hint="plastic",  # Not in valid list
            )


class TestMaterialProperties:
    """Test MaterialProperties validation."""

    def test_valid_properties(self):
        """Test valid material properties."""
        props = MaterialProperties(
            roughness_mean=0.5,
            metallic_mean=0.0,
            ao_strength=0.8,
        )

        assert props.roughness_mean == 0.5
        assert props.metallic_mean == 0.0
        assert props.ao_strength == 0.8

    def test_valid_properties_with_optionals(self):
        """Test properties with optional fields."""
        props = MaterialProperties(
            roughness_mean=0.3,
            metallic_mean=0.9,
            ao_strength=0.7,
            normal_strength=1.5,
            specular_intensity=0.6,
            subsurface_scattering=0.2,
        )

        assert props.normal_strength == 1.5
        assert props.specular_intensity == 0.6
        assert props.subsurface_scattering == 0.2

    def test_invalid_roughness_raises(self):
        """Test that out-of-range roughness is rejected."""
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            MaterialProperties(roughness_mean=1.5, metallic_mean=0.0, ao_strength=0.5)

        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            MaterialProperties(roughness_mean=-0.1, metallic_mean=0.0, ao_strength=0.5)

    def test_invalid_metallic_raises(self):
        """Test that out-of-range metallic is rejected."""
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            MaterialProperties(roughness_mean=0.5, metallic_mean=1.2, ao_strength=0.5)

    def test_invalid_ao_strength_raises(self):
        """Test that out-of-range AO strength is rejected."""
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            MaterialProperties(roughness_mean=0.5, metallic_mean=0.0, ao_strength=-0.1)

    def test_invalid_normal_strength_raises(self):
        """Test that out-of-range normal strength is rejected."""
        with pytest.raises(ValueError, match="\\[0, 2\\]"):
            MaterialProperties(
                roughness_mean=0.5,
                metallic_mean=0.0,
                ao_strength=0.5,
                normal_strength=2.5,
            )

    def test_invalid_specular_intensity_raises(self):
        """Test that out-of-range specular intensity is rejected."""
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            MaterialProperties(
                roughness_mean=0.5,
                metallic_mean=0.0,
                ao_strength=0.5,
                specular_intensity=1.5,
            )

    def test_invalid_subsurface_raises(self):
        """Test that out-of-range subsurface scattering is rejected."""
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            MaterialProperties(
                roughness_mean=0.5,
                metallic_mean=0.0,
                ao_strength=0.5,
                subsurface_scattering=1.1,
            )


class TestPBRTextures:
    """Test PBRTextures contract validation."""

    def test_valid_textures(self, valid_pbr_textures):
        """Test valid PBR textures."""
        assert valid_pbr_textures.albedo.shape == (512, 512, 3)
        assert valid_pbr_textures.normal.shape == (512, 512, 3)
        assert valid_pbr_textures.roughness.shape == (512, 512)
        assert valid_pbr_textures.metallic.shape == (512, 512)
        assert valid_pbr_textures.ambient_occlusion.shape == (512, 512)

    def test_valid_textures_with_height(self):
        """Test PBR textures with optional height map."""
        H, W = 512, 512
        pbr = PBRTextures(
            albedo=np.random.rand(H, W, 3).astype(np.float32),
            normal=(np.random.rand(H, W, 3).astype(np.float32) * 2 - 1),
            roughness=np.random.rand(H, W).astype(np.float32),
            metallic=np.random.rand(H, W).astype(np.float32),
            ambient_occlusion=np.random.rand(H, W).astype(np.float32),
            height=np.random.rand(H, W).astype(np.float32),
        )

        assert pbr.height is not None
        assert pbr.height.shape == (H, W)

    def test_valid_textures_with_properties(self, valid_pbr_textures):
        """Test PBR textures with material properties."""
        props = MaterialProperties(roughness_mean=0.5, metallic_mean=0.1, ao_strength=0.8)
        pbr = PBRTextures(
            albedo=valid_pbr_textures.albedo,
            normal=valid_pbr_textures.normal,
            roughness=valid_pbr_textures.roughness,
            metallic=valid_pbr_textures.metallic,
            ambient_occlusion=valid_pbr_textures.ambient_occlusion,
            properties=props,
        )

        assert pbr.properties is not None
        assert pbr.properties.roughness_mean == 0.5

    def test_invalid_albedo_dtype_raises(self):
        """Test that non-float32 albedo is rejected."""
        H, W = 512, 512
        with pytest.raises(ValueError, match="float32"):
            PBRTextures(
                albedo=np.random.randint(0, 255, (H, W, 3), dtype=np.uint8),
                normal=(np.random.rand(H, W, 3).astype(np.float32) * 2 - 1),
                roughness=np.random.rand(H, W).astype(np.float32),
                metallic=np.random.rand(H, W).astype(np.float32),
                ambient_occlusion=np.random.rand(H, W).astype(np.float32),
            )

    def test_invalid_albedo_shape_raises(self):
        """Test that invalid albedo shape is rejected."""
        H, W = 512, 512
        with pytest.raises(ValueError, match="\\(H, W, 3\\)"):
            PBRTextures(
                albedo=np.random.rand(H, W).astype(np.float32),  # Missing channel dim
                normal=(np.random.rand(H, W, 3).astype(np.float32) * 2 - 1),
                roughness=np.random.rand(H, W).astype(np.float32),
                metallic=np.random.rand(H, W).astype(np.float32),
                ambient_occlusion=np.random.rand(H, W).astype(np.float32),
            )

    def test_invalid_albedo_range_raises(self):
        """Test that out-of-range albedo is rejected."""
        H, W = 512, 512
        bad_albedo = np.random.rand(H, W, 3).astype(np.float32) + 0.5  # > 1.0

        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            PBRTextures(
                albedo=bad_albedo,
                normal=(np.random.rand(H, W, 3).astype(np.float32) * 2 - 1),
                roughness=np.random.rand(H, W).astype(np.float32),
                metallic=np.random.rand(H, W).astype(np.float32),
                ambient_occlusion=np.random.rand(H, W).astype(np.float32),
            )

    def test_invalid_normal_shape_raises(self):
        """Test that mismatched normal shape is rejected."""
        H, W = 512, 512
        with pytest.raises(ValueError, match="must match albedo"):
            PBRTextures(
                albedo=np.random.rand(H, W, 3).astype(np.float32),
                normal=(np.random.rand(H // 2, W // 2, 3).astype(np.float32) * 2 - 1),
                roughness=np.random.rand(H, W).astype(np.float32),
                metallic=np.random.rand(H, W).astype(np.float32),
                ambient_occlusion=np.random.rand(H, W).astype(np.float32),
            )

    def test_invalid_roughness_shape_raises(self):
        """Test that mismatched roughness shape is rejected."""
        H, W = 512, 512
        with pytest.raises(ValueError, match="must match albedo"):
            PBRTextures(
                albedo=np.random.rand(H, W, 3).astype(np.float32),
                normal=(np.random.rand(H, W, 3).astype(np.float32) * 2 - 1),
                roughness=np.random.rand(H // 2, W // 2).astype(np.float32),
                metallic=np.random.rand(H, W).astype(np.float32),
                ambient_occlusion=np.random.rand(H, W).astype(np.float32),
            )


class TestMaterialGenerationConfig:
    """Test MaterialGenerationConfig validation."""

    def test_valid_config_heuristic(self):
        """Test valid heuristic backend config."""
        config = MaterialGenerationConfig(backend="heuristic")
        assert config.backend == "heuristic"
        assert config.resolution == 1024
        assert config.device == "cuda"

    def test_valid_config_custom_settings(self):
        """Test valid config with custom settings."""
        config = MaterialGenerationConfig(
            backend="heuristic",
            resolution=2048,
            optimize_iterations=200,
            use_depth=False,
            normal_strength=1.5,
            ao_intensity=0.5,
            device="cpu",
        )

        assert config.resolution == 2048
        assert config.optimize_iterations == 200
        assert config.use_depth is False
        assert config.normal_strength == 1.5
        assert config.ao_intensity == 0.5
        assert config.device == "cpu"

    def test_invalid_backend_raises(self):
        """Test that invalid backend is rejected."""
        with pytest.raises(ValueError, match="must be one of"):
            MaterialGenerationConfig(backend="invalid_backend")

    def test_invalid_resolution_raises(self):
        """Test that invalid resolution is rejected."""
        with pytest.raises(ValueError, match="512/1024/2048/4096"):
            MaterialGenerationConfig(backend="heuristic", resolution=768)

    def test_invalid_iterations_raises(self):
        """Test that invalid iterations is rejected."""
        with pytest.raises(ValueError, match="positive"):
            MaterialGenerationConfig(backend="heuristic", optimize_iterations=0)

    def test_invalid_normal_strength_raises(self):
        """Test that invalid normal_strength is rejected."""
        with pytest.raises(ValueError, match="\\[0, 2\\]"):
            MaterialGenerationConfig(backend="heuristic", normal_strength=2.5)

    def test_invalid_ao_intensity_raises(self):
        """Test that invalid ao_intensity is rejected."""
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            MaterialGenerationConfig(backend="heuristic", ao_intensity=1.5)

    def test_strict_backend_flag(self):
        """Test strict_backend flag."""
        config = MaterialGenerationConfig(backend="heuristic", strict_backend=True)
        assert config.strict_backend is True

        config = MaterialGenerationConfig(backend="heuristic", strict_backend=False)
        assert config.strict_backend is False

    def test_invalid_strict_backend_type_raises(self):
        """Test that non-bool strict_backend is rejected."""
        with pytest.raises(ValueError, match="strict_backend must be bool"):
            MaterialGenerationConfig(backend="heuristic", strict_backend="yes")


class TestAvailabilityState:
    """Test AvailabilityState enum."""

    def test_availability_states_exist(self):
        """Test all availability states are defined."""
        assert AvailabilityState.AVAILABLE.value == "available"
        assert AvailabilityState.INPUT_CONTRACT_MISMATCH.value == "input_contract_mismatch"
        assert AvailabilityState.RUNTIME_MISSING.value == "runtime_missing"
        assert AvailabilityState.INTEGRATION_MISSING.value == "integration_missing"
        assert AvailabilityState.LICENSE_GATED.value == "license_gated"
        assert AvailabilityState.ATTESTATION_INCOMPLETE.value == "attestation_incomplete"


class TestBackendDecision:
    """Test BackendDecision dataclass."""

    def test_backend_decision_creation(self):
        """Test creating backend decision."""
        decision = BackendDecision(
            requested_backend="nvdiffrec",
            executed_backend="heuristic",
            availability_state=AvailabilityState.RUNTIME_MISSING,
            fallback_reason="NVDIFFREC not installed",
            required_inputs=["depth"],
            required_runtime=["nvdiffrec"],
        )

        assert decision.requested_backend == "nvdiffrec"
        assert decision.executed_backend == "heuristic"
        assert decision.availability_state == AvailabilityState.RUNTIME_MISSING
        assert "NVDIFFREC" in decision.fallback_reason

    def test_backend_decision_to_dict(self):
        """Test backend decision serialization."""
        decision = BackendDecision(
            requested_backend="pbr_fusion",
            executed_backend="pbr_fusion",
            availability_state=AvailabilityState.AVAILABLE,
            fallback_reason=None,
            required_inputs=["image", "depth"],
            required_runtime=["torch"],
        )

        d = decision.to_dict()

        assert d["requested_backend"] == "pbr_fusion"
        assert d["executed_backend"] == "pbr_fusion"
        assert d["availability_state"] == "available"
        assert d["fallback_reason"] is None
        assert "image" in d["required_inputs"]


class TestPBRGenerationMetadata:
    """Test PBRGenerationMetadata dataclass."""

    def test_metadata_creation(self):
        """Test creating PBR generation metadata."""
        metadata = PBRGenerationMetadata(
            backend="heuristic_v5.0.0",
            normal_scale=1.0,
            ao_blend_ratio="0.7_concavity_0.3_variance",
            bilateral_enabled=True,
        )

        assert metadata.backend == "heuristic_v5.0.0"
        assert metadata.normal_scale == 1.0
        assert metadata.bilateral_enabled is True

    def test_metadata_with_backend_decision(self):
        """Test metadata with backend decision."""
        decision = BackendDecision(
            requested_backend="heuristic",
            executed_backend="heuristic",
            availability_state=AvailabilityState.AVAILABLE,
            fallback_reason=None,
            required_inputs=[],
            required_runtime=[],
        )

        metadata = PBRGenerationMetadata(
            backend="heuristic_v5.0.0",
            normal_scale=1.0,
            ao_blend_ratio="0.7_concavity_0.3_variance",
            bilateral_enabled=True,
            backend_decision=decision,
        )

        assert metadata.backend_decision is not None

    def test_metadata_to_dict(self):
        """Test metadata serialization."""
        decision = BackendDecision(
            requested_backend="heuristic",
            executed_backend="heuristic",
            availability_state=AvailabilityState.AVAILABLE,
            fallback_reason=None,
            required_inputs=[],
            required_runtime=[],
        )

        metadata = PBRGenerationMetadata(
            backend="heuristic_v5.0.0",
            normal_scale=1.2,
            ao_blend_ratio="0.7_concavity_0.3_variance",
            bilateral_enabled=False,
            material_hint="wood",
            depth_used=True,
            backend_decision=decision,
        )

        d = metadata.to_dict()

        assert d["backend"] == "heuristic_v5.0.0"
        assert d["normal_scale"] == 1.2
        assert d["bilateral_enabled"] is False
        assert d["material_hint"] == "wood"
        assert d["depth_used"] is True
        assert d["backend_decision"]["availability_state"] == "available"
