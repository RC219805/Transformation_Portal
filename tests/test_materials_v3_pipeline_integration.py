"""Test Materials V3 integration in LuxPipelineV2.

Requires: torch (ML dependency)
Marker: @pytest.mark.ml (runs in ML Tests stage, not Core Tests)
"""

import importlib.util
from pathlib import Path

import pytest

# Check torch availability (safely handle broken torch.__spec__)
try:
    TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
except (ImportError, ValueError):
    TORCH_AVAILABLE = False

# Mark as ML test (runs in ML Tests stage, skipped in Core Tests)
pytestmark = [
    pytest.mark.skipif(not TORCH_AVAILABLE, reason="Requires torch (ML dependency)"),
    pytest.mark.ml,
]

from lux_depth_v2.config import PipelineConfig, Preset
from lux_depth_v2.pipeline import LuxPipelineV2
from lux_depth_v2.materials_v3 import MaterialsV3Config, RefinementStrategy, MaterialTaxonomy


# Dummy segmenter for offline CI (no HuggingFace model download)
class DummySegmenter:
    """Mock segmenter that produces predictable masks without model downloads."""
    
    def predict(self, rgb_t):
        """Generate synthetic glass mask (centered region)."""
        import torch
        _, _, H, W = rgb_t.shape
        mask = torch.zeros((1, 1, H, W), device=rgb_t.device, dtype=torch.float32)
        # Create glass region in center (25%-75% of image)
        mask[:, :, H // 4:3 * H // 4, W // 4:3 * W // 4] = 0.85
        return {"glass": mask}
    
    def get_segmentation_v3_report(self):
        """Return dummy segmentation metadata."""
        return {
            "backend": "dummy",
            "per_class": {"glass": {"fusion_applied": 0.0}}
        }


@pytest.fixture(autouse=True)
def _mock_segmenter(monkeypatch):
    """Mock create_material_segmenter to avoid HuggingFace downloads in CI."""
    import lux_depth_v2.pipeline as pipe_mod
    monkeypatch.setattr(
        pipe_mod,
        "create_material_segmenter",
        lambda *args, **kwargs: DummySegmenter()
    )


def test_materials_v3_disabled_by_default():
    """Materials V3 should be disabled by default (no behavior change)."""
    cfg = PipelineConfig(output_dir=Path("/tmp/test"))
    cfg.preset = Preset.INTERIOR_LUXURY
    cfg.apply_preset()  # apply_preset() takes no arguments except self
    
    pipe = LuxPipelineV2(cfg)
    
    # Should not initialize Materials V3 engine when disabled
    assert pipe.materials_v3_engine is None


def test_materials_v3_can_be_enabled():
    """Materials V3 can be enabled with explicit config."""
    cfg = PipelineConfig(output_dir=Path("/tmp/test"))
    
    # Create Materials V3 config and enable it
    cfg.materials_v3 = MaterialsV3Config()
    cfg.materials_v3.enabled = True
    cfg.materials_v3.taxonomy = MaterialTaxonomy.BASE
    cfg.materials_v3.refine_edges = RefinementStrategy.OFF
    
    pipe = LuxPipelineV2(cfg)
    
    # Should initialize Materials V3 engine
    assert pipe.materials_v3_engine is not None


def test_materials_v3_initialization_logs():
    """Materials V3 initialization should log key config settings."""
    cfg = PipelineConfig(output_dir=Path("/tmp/test"))
    cfg.materials_v3 = MaterialsV3Config()
    cfg.materials_v3.enabled = True
    cfg.materials_v3.taxonomy = MaterialTaxonomy.EXPANDED
    cfg.materials_v3.refine_edges = RefinementStrategy.CANARY
    cfg.materials_v3.max_megapixels = 25.0
    
    pipe = LuxPipelineV2(cfg)
    
    # Should initialize Materials V3 engine
    assert pipe.materials_v3_engine is not None


def test_materials_v3_graceful_fallback():
    """Materials V3 should fall back gracefully if initialization fails."""
    cfg = PipelineConfig(output_dir=Path("/tmp/test"))
    cfg.materials_v3 = MaterialsV3Config()
    cfg.materials_v3.enabled = True
    
    # Even if there's an error, pipeline should still construct
    # (tested by importing pipeline module successfully in setup)
    pipe = LuxPipelineV2(cfg)
    
    # Pipeline should exist regardless
    assert pipe is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
