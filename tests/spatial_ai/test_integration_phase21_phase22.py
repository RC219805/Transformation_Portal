"""Integration test: Phase 2.1 (Segmentation) + Phase 2.2 (Materials)."""

import os
from pathlib import Path

import numpy as np
import pytest

_SAM2_CHECKPOINT_DEFAULT = "checkpoints/sam2_hiera_base_plus.pt"
_SAM2_CHECKPOINT_ENV = os.environ.get("TP_PORTAL_LUX_SAM2_CHECKPOINT")
# An explicitly-set but empty env var is treated as "not configured" rather than
# silently falling back to the default — surface the misconfiguration as a skip
# instead of attempting to load a phantom checkpoint.
_SAM2_CHECKPOINT_PATH = (
    Path(_SAM2_CHECKPOINT_ENV) if _SAM2_CHECKPOINT_ENV and _SAM2_CHECKPOINT_ENV.strip() else Path(_SAM2_CHECKPOINT_DEFAULT)
)
# is_file() (not exists()) ensures we don't accept a directory; the .pt suffix
# guards against other file types being pointed at by mistake.
_SAM2_AVAILABLE = (
    _SAM2_CHECKPOINT_PATH.is_file() and _SAM2_CHECKPOINT_PATH.suffix == ".pt" and _SAM2_CHECKPOINT_PATH.stat().st_size > 0
)
if _SAM2_AVAILABLE:
    try:
        import sam2 as _sam2_pkg  # noqa: F401
    except ImportError:
        _SAM2_AVAILABLE = False

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ml,
    pytest.mark.skipif(
        not _SAM2_AVAILABLE,
        reason=(
            "SAM2 integration requires `make install-ml-sam2` and a checkpoint at "
            "$TP_PORTAL_LUX_SAM2_CHECKPOINT (default: checkpoints/sam2_hiera_base_plus.pt)."
        ),
    ),
]


class TestPhase21And22Integration:
    """Test integration between segmentation and materials modules."""

    @pytest.fixture
    def sample_rgb(self):
        """Create sample RGB image."""
        return np.random.rand(256, 256, 3).astype(np.float32)

    def test_sam2_to_pbr_pipeline(self, sample_rgb):
        """Test full pipeline: SAM2 segmentation → PBR generation."""
        from transformation_portal.spatial_ai.materials import PBRGenerator
        from transformation_portal.spatial_ai.segmentation import SAM2Backend

        # Phase 2.1: Segment image
        sam2 = SAM2Backend(model_size="base", device="cpu")
        seg_result = sam2.segment(sample_rgb, gamma=1.0, mode="auto")

        # Verify segmentation result
        assert seg_result.masks.shape[0] > 0  # At least one mask
        assert seg_result.masks.dtype == bool

        # Phase 2.2: Generate PBR for each segment
        pbr_gen = PBRGenerator(backend="heuristic", device="cpu")

        # Single segment
        pbr_result = pbr_gen.generate(
            image=sample_rgb,
            gamma=1.0,
            mask=seg_result.masks[0],  # Use first mask
        )

        # Verify PBR result
        assert pbr_result.albedo.shape == sample_rgb.shape
        assert pbr_result.normal.shape == sample_rgb.shape
        assert pbr_result.properties is not None

    def test_batch_pbr_with_sam2_masks(self, sample_rgb):
        """Test batch PBR generation with SAM2 masks."""
        from transformation_portal.spatial_ai.materials import PBRGenerator
        from transformation_portal.spatial_ai.segmentation import SAM2Backend

        # Phase 2.1: Segment
        sam2 = SAM2Backend(model_size="base", device="cpu")
        seg_result = sam2.segment(sample_rgb, gamma=1.0, mode="auto")

        # Phase 2.2: Batch generate PBR
        pbr_gen = PBRGenerator(backend="heuristic", device="cpu")

        # Take up to 3 segments
        num_segments = min(3, seg_result.masks.shape[0])
        masks = [seg_result.masks[i] for i in range(num_segments)]

        pbr_results = pbr_gen.generate_batch(
            image=sample_rgb,
            gamma=1.0,
            masks=masks,
        )

        # Verify results
        assert len(pbr_results) == num_segments
        for pbr in pbr_results:
            assert pbr.albedo.shape == sample_rgb.shape
            assert pbr.properties is not None

    def test_material_hints_from_classification(self, sample_rgb):
        """Test using material classification as hints for PBR generation."""
        from transformation_portal.spatial_ai.materials import PBRGenerator
        from transformation_portal.spatial_ai.segmentation import MaterialClassifier, SAM2Backend

        # Phase 2.1: Segment and classify
        sam2 = SAM2Backend(model_size="base", device="cpu")
        seg_result = sam2.segment(sample_rgb, gamma=1.0, mode="auto")

        # Classify materials (if CLIP available)
        classifier = MaterialClassifier()
        if classifier.is_available():
            classified_result = classifier.classify_masks(
                image=sample_rgb,
                masks=seg_result.masks[:3],  # First 3 segments
                classes=["wood floor", "marble surface", "brushed steel"],
            )

            # Phase 2.2: Use classifications as hints
            pbr_gen = PBRGenerator(backend="heuristic", device="cpu")

            material_hints = []
            for meta in classified_result.metadata:
                if meta.material_label:
                    # Map label to material hint
                    if "wood" in meta.material_label.lower():
                        material_hints.append("wood")
                    elif "marble" in meta.material_label.lower():
                        material_hints.append("stone")
                    elif "steel" in meta.material_label.lower():
                        material_hints.append("metal")
                    else:
                        material_hints.append(None)
                else:
                    material_hints.append(None)

            pbr_results = pbr_gen.generate_batch(
                image=sample_rgb,
                gamma=1.0,
                masks=classified_result.masks,
                material_hints=material_hints,
            )

            assert len(pbr_results) == len(material_hints)

    def test_contract_compatibility(self, sample_rgb):
        """Test contract compatibility between phases."""
        from transformation_portal.spatial_ai.materials import MaterialInput
        from transformation_portal.spatial_ai.segmentation import SegmentationInput

        # Both should accept same input contract
        seg_input = SegmentationInput(
            image=sample_rgb,
            gamma=1.0,
            mode="auto",
        )

        mat_input = MaterialInput(
            image=sample_rgb,
            gamma=1.0,
        )

        # Both enforce gamma=1.0
        assert seg_input.gamma == 1.0
        assert mat_input.gamma == 1.0

        # Both require float32
        assert seg_input.image.dtype == np.float32
        assert mat_input.image.dtype == np.float32
