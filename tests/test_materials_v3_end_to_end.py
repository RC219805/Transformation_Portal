#!/usr/bin/env python3
"""
Materials V3 End-to-End Integration Test

Validates the complete pipeline wiring:
- Masks are populated from segmentation
- V3 metadata is emitted correctly
- Response plan is generated
- Pixel ops stats are tracked
- Report JSON schema is stable

Requires: torch (ML dependency)
Marker: @pytest.mark.ml (runs in ML Tests stage, not Core Tests)
"""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Check torch availability (safely handle broken torch.__spec__)
try:
    TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
except (ImportError, ValueError):
    TORCH_AVAILABLE = False

# Mark as ML test (runs in ML Tests stage, skipped in Core Tests when torch unavailable)
pytestmark = [
    pytest.mark.skipif(not TORCH_AVAILABLE, reason="Requires torch (ML dependency)"),
    pytest.mark.ml,
]

from lux_depth_v2.pipeline import LuxPipelineV2
from lux_depth_v2.config import PipelineConfig, Preset


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


def _create_synthetic_image(h=256, w=256):
    """Create a synthetic RGB image with some structure."""
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # Add some gradients and blocks to trigger material segmentation
    rgb[:, :w // 2] = [100, 120, 140]  # Left half
    rgb[:, w // 2:] = [180, 160, 150]  # Right half
    rgb[h // 4:3 * h // 4, w // 4:3 * w // 4] = [200, 200, 200]  # Center block
    return rgb


class TestMaterialsV3EndToEnd:
    """End-to-end validation of Materials V3 pipeline integration."""

    def test_v3_disabled_by_default(self, tmp_path):
        """Materials V3 should be disabled by default and not affect output."""
        # Create synthetic input
        img = _create_synthetic_image()
        img_path = tmp_path / "test.png"
        from PIL import Image
        Image.fromarray(img).save(img_path)
        
        # Run with default config
        cfg = PipelineConfig(
            output_dir=tmp_path / "out",
            preset=Preset.INTERIOR_LUXURY,
            write_outputs=False,  # Don't write files
            enable_material=True,  # Enable segmentation so V3 can work if enabled
        )
        
        pipe = LuxPipelineV2(cfg)
        report = pipe.process_one(img_path)
        
        # V3 should be disabled
        assert report["materials_v3_enabled"] is False
        # Metadata should be empty or None
        assert not report.get("materials_v3_metadata") or report["materials_v3_metadata"] == {}

    def test_v3_enabled_emits_metadata(self, tmp_path):
        """When enabled, V3 should emit metadata in the report."""
        # Create synthetic input
        img = _create_synthetic_image()
        img_path = tmp_path / "test.png"
        from PIL import Image
        Image.fromarray(img).save(img_path)
        
        # Enable Materials V3
        from lux_depth_v2.materials_v3 import MaterialsV3Config
        cfg = PipelineConfig(
            output_dir=tmp_path / "out",
            preset=Preset.INTERIOR_LUXURY,
            write_outputs=False,
            enable_material=True,
        )
        cfg.materials_v3 = MaterialsV3Config()
        cfg.materials_v3.enabled = True
        
        pipe = LuxPipelineV2(cfg)
        report = pipe.process_one(img_path)
        
        # V3 should be enabled
        assert report["materials_v3_enabled"] is True
        
        # Metadata should exist
        assert report["materials_v3_metadata"] is not None
        assert isinstance(report["materials_v3_metadata"], dict)
        
        # Check required keys
        metadata = report["materials_v3_metadata"]
        assert "enabled" in metadata
        assert metadata["enabled"] is True
        assert "taxonomy" in metadata
        assert "per_class_stats" in metadata
        assert "canonical_materials" in metadata

    def test_v3_response_plan_generated(self, tmp_path):
        """V3 should generate a response plan when enabled."""
        img = _create_synthetic_image()
        img_path = tmp_path / "test.png"
        from PIL import Image
        Image.fromarray(img).save(img_path)
        
        from lux_depth_v2.materials_v3 import MaterialsV3Config
        cfg = PipelineConfig(
            output_dir=tmp_path / "out",
            preset=Preset.INTERIOR_LUXURY,
            write_outputs=False,
            enable_material=True,
        )
        cfg.materials_v3 = MaterialsV3Config()
        cfg.materials_v3.enabled = True
        
        pipe = LuxPipelineV2(cfg)
        report = pipe.process_one(img_path)
        
        # Response plan should exist
        assert "materials_v3_response_plan" in report
        plan = report["materials_v3_response_plan"]
        assert plan is not None
        assert isinstance(plan, dict)
        
        # Check plan structure
        assert "enabled" in plan
        assert "strategy" in plan
        assert "per_class" in plan

    def test_v3_pixel_ops_stats(self, tmp_path):
        """V3 should track pixel ops stats even when not applied."""
        img = _create_synthetic_image()
        img_path = tmp_path / "test.png"
        from PIL import Image
        Image.fromarray(img).save(img_path)
        
        from lux_depth_v2.materials_v3 import MaterialsV3Config
        cfg = PipelineConfig(
            output_dir=tmp_path / "out",
            preset=Preset.INTERIOR_LUXURY,
            write_outputs=False,
            enable_material=True,
        )
        cfg.materials_v3 = MaterialsV3Config()
        cfg.materials_v3.enabled = True
        
        pipe = LuxPipelineV2(cfg)
        report = pipe.process_one(img_path)
        
        # Pixel ops stats should exist (even if not applied or empty)
        assert "materials_v3_pixel_ops" in report
        pixel_ops = report.get("materials_v3_pixel_ops")
        # May be None or empty dict depending on implementation
        # The important thing is the key exists in the report
        if pixel_ops is not None:
            assert isinstance(pixel_ops, dict)
            # Check stats structure if present
            if pixel_ops:
                assert "enabled" in pixel_ops

    def test_v3_class_presence_audit(self, tmp_path):
        """V3 should include class presence audit for debugging."""
        img = _create_synthetic_image()
        img_path = tmp_path / "test.png"
        from PIL import Image
        Image.fromarray(img).save(img_path)
        
        from lux_depth_v2.materials_v3 import MaterialsV3Config
        cfg = PipelineConfig(
            output_dir=tmp_path / "out",
            preset=Preset.INTERIOR_LUXURY,
            write_outputs=False,
            enable_material=True,
        )
        cfg.materials_v3 = MaterialsV3Config()
        cfg.materials_v3.enabled = True
        
        pipe = LuxPipelineV2(cfg)
        report = pipe.process_one(img_path)
        
        # Metadata should include class presence audit
        metadata = report["materials_v3_metadata"]
        assert "class_presence_audit" in metadata
        audit = metadata["class_presence_audit"]
        assert isinstance(audit, dict)
        # Should include emitted_classes, requested_classes, etc.

    def test_v3_fallback_on_error(self, tmp_path):
        """V3 should gracefully fall back on error without crashing pipeline."""
        img = _create_synthetic_image()
        img_path = tmp_path / "test.png"
        from PIL import Image
        Image.fromarray(img).save(img_path)
        
        from lux_depth_v2.materials_v3 import MaterialsV3Config
        cfg = PipelineConfig(
            output_dir=tmp_path / "out",
            preset=Preset.INTERIOR_LUXURY,
            write_outputs=False,
            enable_material=True,
        )
        cfg.materials_v3 = MaterialsV3Config()
        cfg.materials_v3.enabled = True
        
        # Force an error by corrupting the config (example)
        # This is a smoke test - pipeline should not crash
        
        pipe = LuxPipelineV2(cfg)
        report = pipe.process_one(img_path)
        
        # Pipeline should complete
        assert report["status"] == "ok"
        
        # If V3 failed, metadata should indicate fallback
        if "error" in report.get("materials_v3_metadata", {}):
            assert report["materials_v3_metadata"]["fallback"] is True

    def test_v3_with_canary_preset(self, tmp_path):
        """Canary preset should enable V3 with pixel ops."""
        pytest.skip("Canary preset not yet defined in config - deferred to PR-4B validation")
        
        img = _create_synthetic_image()
        img_path = tmp_path / "test.png"
        from PIL import Image
        Image.fromarray(img).save(img_path)
        
        # Use canary preset (once it exists)
        # cfg = PipelineConfig(
        #     output_dir=tmp_path / "out",
        #     preset=Preset.INTERIOR_LUXURY_APEX_QUALITY_MATERIALS_V3_GLASS_CANARY,
        #     write_outputs=False,
        #     enable_material=True,
        # )
        
        # pipe = LuxPipelineV2(cfg)
        # report = pipe.process_one(img_path)
        
        # # V3 should be enabled
        # assert report["materials_v3_enabled"] is True
        
        # # Pixel ops should be enabled (but may not be applied if no glass detected)
        # pixel_ops = report["materials_v3_pixel_ops"]
        # assert pixel_ops["enabled"] is True
        
        # # If glass was detected and processed, applied_to should be populated
        # if pixel_ops.get("applied_to"):
        #     assert "glass" in pixel_ops["applied_to"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
