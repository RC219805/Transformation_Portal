"""Tests for EnhancementStage defensive shape handling."""

import numpy as np

from transformation_portal.stage_graph.stage import StageContext, StageStatus
from transformation_portal.stage_graph.stages.enhancement import EnhancementStage


def test_enhancement_stage_resizes_mismatched_depth_map():
    """Depth map should be resized when it does not match image dimensions."""
    image = np.random.randint(0, 256, (64, 96, 3), dtype=np.uint8)
    depth_map = np.linspace(0.0, 1.0, 72 * 104, dtype=np.float32).reshape(72, 104)

    stage = EnhancementStage(
        enhancement_strength=0.7,
        clarity_strength=0.0,
        material_strength=0.0,
    )
    context = StageContext(
        artifacts={
            "image": image,
            "depth_map": depth_map,
        }
    )

    result = stage.compute(context)

    assert result.status == StageStatus.COMPLETED
    assert result.artifacts["enhanced_image"].shape == image.shape
    assert result.metadata["has_depth"] is True
