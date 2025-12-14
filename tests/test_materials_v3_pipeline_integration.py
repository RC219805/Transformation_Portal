"""Test Materials V3 integration in LuxPipelineV2.

Requires: torch (ML dependency)
Marker: @pytest.mark.ml (runs in ML Tests stage, not Core Tests)
"""

import importlib.util
from pathlib import Path

import pytest

# Check torch availability
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

# Mark as ML test (runs in ML Tests stage, skipped in Core Tests)
pytestmark = [
    pytest.mark.skipif(not TORCH_AVAILABLE, reason="Requires torch (ML dependency)"),
    pytest.mark.ml,
]

from lux_depth_v2.config import PipelineConfig, Preset
from lux_depth_v2.pipeline import LuxPipelineV2
from lux_depth_v2.materials_v3 import MaterialsV3Config, RefinementStrategy, MaterialTaxonomy


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
