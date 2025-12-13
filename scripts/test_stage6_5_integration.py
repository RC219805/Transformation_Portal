#!/usr/bin/env python3
"""
Quick smoke test: verify segmentation_v3 appears in canary preset reports.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lux_depth_v2.config import PipelineConfig, Preset
from lux_depth_v2.pipeline import LuxPipelineV2
import numpy as np
from PIL import Image


def create_test_image(path: Path, size=(256, 256)):
    """Create a simple test TIFF image."""
    img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    Image.fromarray(img).save(path, format="TIFF")


def test_canary_preset_emits_v3_stats():
    """Test that canary preset produces segmentation_v3 in report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test input
        input_img = tmpdir / "test_input.tiff"
        create_test_image(input_img)
        
        # Create output dir
        output_dir = tmpdir / "output"
        output_dir.mkdir()
        
        # Configure canary preset
        cfg = PipelineConfig()
        cfg.preset = Preset.INTERIOR_LUXURY_APEX_QUALITY_EFFICIENTSAM
        cfg.apply_preset()
        cfg.output_dir = str(output_dir)
        cfg.write_outputs = True
        
        print(f"Running canary preset: {cfg.preset.value}")
        print(f"  Backend V3: {cfg.segmentation.backend_v3}")
        print(f"  Fusion mode: {cfg.segmentation.fusion_mode}")
        print(f"  EfficientSAM model: {cfg.segmentation.efficientSAM_model}")
        
        # Run pipeline
        pipeline = LuxPipelineV2(cfg)
        report = pipeline.process_one(input_img)
        
        # Verify segmentation_v3 exists
        assert "segmentation_v3" in report, "❌ Missing segmentation_v3 in report"
        
        v3 = report["segmentation_v3"]
        print("\n✓ segmentation_v3 present in report:")
        print(f"  backend_v3: {v3.get('backend_v3')}")
        print(f"  fusion_mode: {v3.get('fusion_mode')}")
        print(f"  model: {v3.get('model')}")
        print(f"  refined_classes: {v3.get('refined_classes')}")
        print(f"  per_class stats: {list(v3.get('per_class', {}).keys())}")
        
        # Verify structure
        assert "backend_v3" in v3
        assert "fusion_mode" in v3
        assert "model" in v3
        assert "refined_classes" in v3
        assert "per_class" in v3
        
        # Check report JSON file was written
        report_path = output_dir / "test_input_report.json"
        assert report_path.exists(), f"❌ Report JSON not written: {report_path}"
        
        # Verify JSON contains segmentation_v3
        with open(report_path) as f:
            saved_report = json.load(f)
        
        assert "segmentation_v3" in saved_report, "❌ segmentation_v3 missing from saved JSON"
        
        print(f"\n✓ Report JSON written to: {report_path}")
        print("✓ segmentation_v3 properly serialized to JSON")
        
        return True


if __name__ == "__main__":
    try:
        print("="*60)
        print("Stage 6.5 Pipeline Integration Smoke Test")
        print("="*60)
        test_canary_preset_emits_v3_stats()
        print("\n" + "="*60)
        print("✓ ALL CHECKS PASSED - Stage 6.5 integration verified")
        print("="*60)
        raise SystemExit(0)
    except Exception as e:
        print(f"\n✗ SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
