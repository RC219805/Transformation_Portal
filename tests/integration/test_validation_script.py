#!/usr/bin/env python3
"""Integration test: validate script calls V2 classifier."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def test_image_dir(tmp_path):
    """Create test images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    # Create 2 test images (small for speed)
    for i in range(2):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(img).save(img_dir / f"test_{i}.jpg")
    
    return img_dir


def test_validation_script_calls_v2_classifier(test_image_dir, tmp_path):
    """
    Integration test: validation script must call V2 classifier.
    
    This test would have caught the P0 silent failure.
    """
    import os
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Set PYTHONPATH to include project root
    env = os.environ.copy()
    project_root = Path(__file__).parent.parent.parent
    env['PYTHONPATH'] = str(project_root)
    
    # Run validation script
    result = subprocess.run(
        [
            "python",
            "scripts/automation/production_depth_validation_fixed.py",
            "--input-dir", str(test_image_dir),
            "--output-dir", str(output_dir),
            "--tile-size", "512",
            "--overlap", "64"
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(project_root)
    )
    
    # Must exit 0 (success)
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    
    # Check metrics files exist
    metrics_files = list(output_dir.glob("*_metrics.json"))
    assert len(metrics_files) == 2, f"Expected 2 metrics files, got {len(metrics_files)}"
    
    # Validate metrics content
    for mf in metrics_files:
        with open(mf) as f:
            metrics = json.load(f)
        
        # CRITICAL: These must NOT be null
        assert metrics['scene_type'] is not None, \
            f"scene_type is null in {mf.name} - V2 classifier not called!"
        
        assert metrics['edge_f1'] is not None, \
            f"edge_f1 is null in {mf.name}"
        
        assert isinstance(metrics['edge_f1'], (int, float)), \
            f"edge_f1 must be numeric, got {type(metrics['edge_f1'])}"
        
        assert metrics['lenient_pass'] is not None, \
            f"lenient_pass is null in {mf.name}"
        
        assert isinstance(metrics['lenient_pass'], bool), \
            f"lenient_pass must be bool, got {type(metrics['lenient_pass'])}"
        
        assert metrics['classification_factors'] is not None, \
            f"classification_factors is null in {mf.name}"
        
        assert 'ratio' in metrics['classification_factors'], \
            f"classification_factors missing 'ratio' in {mf.name}"
        
        print(f"✓ {mf.name}: scene_type={metrics['scene_type']}, F1={metrics['edge_f1']:.3f}")


def test_validation_script_fails_on_incomplete_metrics():
    """Script must exit non-zero if metrics are incomplete."""
    # This would test the fail-fast behavior
    # (requires mock/patch to inject None metrics)
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
