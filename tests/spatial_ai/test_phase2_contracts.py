"""Contract tests for Phase 2 integration points.

This module validates data flow contracts between Phase 2 components:
- Phase 2.1 (Segmentation) → Phase 2.2 (Materials)
- Phase 2.2 (Materials) → Phase 2.3 (Reconstruction)
- End-to-end pipeline contracts

Contract tests ensure:
1. Gamma linearity is preserved (gamma=1.0) across all phase boundaries
2. Data format compatibility between phases
3. Metadata preservation through the pipeline
4. Error handling at integration points
"""

import numpy as np
import pytest

from transformation_portal.spatial_ai.materials.contracts import MaterialInput, PBRTextures
from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput  # noqa: F401


@pytest.fixture
def linear_test_image():
    """Create a test image in linear gamma space (gamma=1.0)."""
    # Create a simple gradient in linear space as float32
    arr = np.linspace(0, 1, 256, dtype=np.float32).reshape(1, -1)
    arr = np.repeat(arr, 256, axis=0)
    arr = np.stack([arr, arr * 0.5, arr * 0.2], axis=-1)
    return arr


@pytest.fixture
def test_mask():
    """Create a test segmentation mask."""
    mask = np.zeros((256, 256), dtype=bool)
    mask[50:150, 50:150] = True
    return mask


class TestPhase21ToPhase22Contract:
    """Contract tests for Segmentation → Materials integration."""

    def test_segmentation_input_enforces_gamma(self, linear_test_image):
        """Verify segmentation input validates gamma=1.0."""
        # Valid gamma
        seg_input = SegmentationInput(
            image=linear_test_image,
            gamma=1.0,
            mode="auto",
        )
        assert seg_input.gamma == 1.0

        # Invalid gamma should raise
        with pytest.raises(ValueError, match="gamma=1.0"):
            SegmentationInput(
                image=linear_test_image,
                gamma=2.2,
                mode="auto",
            )

    def test_segmentation_output_to_material_input(self, linear_test_image, test_mask):
        """Verify segmentation output is compatible with material input."""
        # Segmentation produces masks (bool dtype)
        assert test_mask.dtype == bool
        assert test_mask.ndim == 2

        # Material input should accept mask from segmentation
        mat_input = MaterialInput(
            image=linear_test_image,
            gamma=1.0,
            mask=test_mask,
            material_hint="wood",
        )

        assert mat_input.gamma == 1.0
        assert mat_input.mask is not None
        assert mat_input.mask.dtype == bool

    def test_gamma_propagation_seg_to_mat(self, linear_test_image, test_mask):
        """Verify gamma=1.0 propagates from segmentation to materials."""
        # Create segmentation input
        seg_input = SegmentationInput(
            image=linear_test_image,
            gamma=1.0,
            mode="auto",
        )

        # Create material input with same gamma
        mat_input = MaterialInput(
            image=linear_test_image,
            gamma=seg_input.gamma,
            mask=test_mask,
        )

        assert mat_input.gamma == seg_input.gamma == 1.0

    def test_mask_format_compatibility(self, test_mask):
        """Verify mask formats are compatible across phases."""
        # Segmentation produces bool masks
        assert test_mask.dtype == bool
        assert test_mask.ndim == 2

        # Mask should have reasonable size
        assert test_mask.shape[0] > 0
        assert test_mask.shape[1] > 0


class TestPhase22ToPhase23Contract:
    """Contract tests for Materials → Reconstruction integration."""

    def test_pbr_textures_have_correct_format(self):
        """Verify PBR textures are in expected format."""
        pbr = PBRTextures(
            albedo=np.random.rand(512, 512, 3).astype(np.float32),
            normal=np.random.rand(512, 512, 3).astype(np.float32),
            roughness=np.random.rand(512, 512).astype(np.float32),
            metallic=np.random.rand(512, 512).astype(np.float32),
            ambient_occlusion=np.random.rand(512, 512).astype(np.float32),
        )

        # All textures should be float32 numpy arrays
        assert pbr.albedo.dtype == np.float32
        assert pbr.normal.dtype == np.float32
        assert pbr.roughness.dtype == np.float32
        assert pbr.metallic.dtype == np.float32

        # All should have same spatial dimensions
        H, W = pbr.albedo.shape[:2]
        assert pbr.normal.shape[:2] == (H, W)
        assert pbr.roughness.shape == (H, W)
        assert pbr.metallic.shape == (H, W)

    def test_pbr_texture_value_ranges(self):
        """Verify PBR texture values are in expected ranges."""
        pbr = PBRTextures(
            albedo=np.random.rand(512, 512, 3).astype(np.float32),
            normal=np.random.rand(512, 512, 3).astype(np.float32),
            roughness=np.random.rand(512, 512).astype(np.float32),
            metallic=np.random.rand(512, 512).astype(np.float32),
            ambient_occlusion=np.random.rand(512, 512).astype(np.float32),
        )

        # Values should be in reasonable ranges (0 to ~1 for normalized)
        assert pbr.albedo.min() >= 0
        assert pbr.roughness.min() >= 0
        assert pbr.metallic.min() >= 0


class TestEndToEndPipelineContract:
    """Contract tests for complete E2E pipeline."""

    def test_gamma_consistency_through_pipeline(self, linear_test_image, test_mask):
        """Verify gamma=1.0 is enforced throughout pipeline."""
        # Start with segmentation input
        seg_input = SegmentationInput(
            image=linear_test_image,
            gamma=1.0,
            mode="auto",
        )
        assert seg_input.gamma == 1.0

        # Materials input preserves gamma
        mat_input = MaterialInput(
            image=linear_test_image,
            gamma=seg_input.gamma,
            mask=test_mask,
        )
        assert mat_input.gamma == 1.0

    def test_data_format_compatibility(self, linear_test_image, test_mask):
        """Verify data formats are compatible across phases."""
        # Segmentation: float32 image in, bool masks out
        seg_input = SegmentationInput(
            image=linear_test_image,
            gamma=1.0,
            mode="auto",
        )
        assert seg_input.image.dtype == np.float32
        assert test_mask.dtype == bool

        # Materials: float32 image + bool mask in, float32 textures out
        mat_input = MaterialInput(
            image=linear_test_image,
            gamma=1.0,
            mask=test_mask,
        )
        assert mat_input.image.dtype == np.float32
        assert mat_input.mask.dtype == bool


class TestErrorHandlingContracts:
    """Contract tests for error handling at integration points."""

    def test_invalid_gamma_raises_error(self, linear_test_image):
        """Verify invalid gamma values are rejected."""
        with pytest.raises(ValueError, match="gamma=1.0"):
            SegmentationInput(
                image=linear_test_image,
                gamma=2.2,
                mode="auto",
            )

        with pytest.raises(ValueError, match="gamma=1.0"):
            MaterialInput(
                image=linear_test_image,
                gamma=2.2,
            )

    def test_mismatched_dimensions_raise_error(self, linear_test_image):
        """Verify dimension mismatches are caught."""
        # Create mismatched mask (different size than image)
        wrong_size_mask = np.zeros((128, 128), dtype=bool)

        with pytest.raises(ValueError, match="must match image"):
            MaterialInput(
                image=linear_test_image,
                gamma=1.0,
                mask=wrong_size_mask,
            )

    def test_invalid_dtypes_raise_error(self, linear_test_image):
        """Verify invalid dtypes are rejected."""
        # Wrong dtype for image (uint8 instead of float32)
        wrong_dtype_image = (linear_test_image * 255).astype(np.uint8)

        with pytest.raises(ValueError, match="must be float32"):
            SegmentationInput(
                image=wrong_dtype_image,
                gamma=1.0,
                mode="auto",
            )

    def test_invalid_mask_dtype_raises_error(self, linear_test_image):
        """Verify invalid mask dtypes are rejected."""
        # Wrong dtype for mask (uint8 instead of bool)
        wrong_mask = np.zeros((256, 256), dtype=np.uint8)

        with pytest.raises(ValueError, match="bool"):
            MaterialInput(
                image=linear_test_image,
                gamma=1.0,
                mask=wrong_mask,
            )


class TestDataFormatContracts:
    """Contract tests for data format compatibility."""

    def test_image_format_consistency(self, linear_test_image):
        """Verify image formats are consistent across the pipeline."""
        # Images should be float32, (H, W, 3)
        assert linear_test_image.dtype == np.float32
        assert linear_test_image.ndim == 3
        assert linear_test_image.shape[2] == 3

    def test_mask_format_consistency(self, test_mask):
        """Verify mask formats are consistent across the pipeline."""
        # Masks should be bool, (H, W)
        assert test_mask.dtype == bool
        assert test_mask.ndim == 2

    def test_pbr_texture_format_consistency(self):
        """Verify PBR texture formats are consistent."""
        pbr = PBRTextures(
            albedo=np.random.rand(512, 512, 3).astype(np.float32),
            normal=np.random.rand(512, 512, 3).astype(np.float32),
            roughness=np.random.rand(512, 512).astype(np.float32),
            metallic=np.random.rand(512, 512).astype(np.float32),
            ambient_occlusion=np.random.rand(512, 512).astype(np.float32),
        )

        # All should be float32
        assert pbr.albedo.dtype == np.float32
        assert pbr.normal.dtype == np.float32
        assert pbr.roughness.dtype == np.float32
        assert pbr.metallic.dtype == np.float32

        # Albedo and normal should be (H, W, 3)
        assert pbr.albedo.ndim == 3 and pbr.albedo.shape[2] == 3
        assert pbr.normal.ndim == 3 and pbr.normal.shape[2] == 3

        # Roughness and metallic should be (H, W)
        assert pbr.roughness.ndim == 2
        assert pbr.metallic.ndim == 2
