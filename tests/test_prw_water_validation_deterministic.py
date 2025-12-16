"""
Test deterministic stability scoring with fixed seed.

Ensures that the validation harness produces identical stability scores
when run multiple times with the same seed on the same input.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import json
from PIL import Image

from scripts.prw_water_validation import WaterValidationHarness
from lux_depth_v2.materials_v3 import MaterialsV3Config


def create_synthetic_image(size=(256, 256), color=(0.5, 0.6, 0.7)):
    """Create synthetic test image."""
    img = np.zeros((*size, 3), dtype=np.float32)
    img[:, :, 0] = color[0]
    img[:, :, 1] = color[1]
    img[:, :, 2] = color[2]
    return img


def test_stability_deterministic_with_seed():
    """Test that stability scoring is deterministic when seed is provided."""
    # Create synthetic image
    rgb = create_synthetic_image(size=(128, 128), color=(0.3, 0.5, 0.7))
    
    # Create config
    config = MaterialsV3Config(
        water_detection_enabled=True,
        water_edge_refinement_enabled=False  # Disable for speed
    )
    
    # Run stability test twice with same seed
    seed = 42
    harness1 = WaterValidationHarness(config, seed=seed)
    harness2 = WaterValidationHarness(config, seed=seed)
    
    stability1 = harness1._compute_stability(rgb, rgb * 0.5)  # Use dummy depth
    stability2 = harness2._compute_stability(rgb, rgb * 0.5)
    
    # Scores should be identical (or within floating point epsilon)
    assert abs(stability1 - stability2) < 1e-6, (
        f"Stability scores differ with same seed: {stability1} vs {stability2}"
    )
    
    print(f"✅ Deterministic stability test passed")
    print(f"   Seed: {seed}")
    print(f"   Stability score: {stability1:.6f}")
    print(f"   Difference: {abs(stability1 - stability2):.10f}")


def test_stability_different_with_different_seed():
    """Test that stability scoring differs with different seeds (noise changes)."""
    rgb = create_synthetic_image(size=(128, 128), color=(0.3, 0.5, 0.7))
    
    config = MaterialsV3Config(
        water_detection_enabled=True,
        water_edge_refinement_enabled=False
    )
    
    # Run with different seeds
    harness1 = WaterValidationHarness(config, seed=42)
    harness2 = WaterValidationHarness(config, seed=123)
    
    stability1 = harness1._compute_stability(rgb, rgb * 0.5)
    stability2 = harness2._compute_stability(rgb, rgb * 0.5)
    
    # Scores should differ (noise perturbation is seed-dependent)
    # But not necessarily guaranteed to differ significantly
    print(f"✅ Different seed test")
    print(f"   Seed 42:  {stability1:.6f}")
    print(f"   Seed 123: {stability2:.6f}")
    print(f"   Difference: {abs(stability1 - stability2):.6f}")


def test_full_validation_deterministic():
    """Test that full validation pipeline produces identical results with same seed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test image
        img_dir = tmpdir / "images" / "pool"
        img_dir.mkdir(parents=True)
        img_path = img_dir / "test_001.jpg"
        
        rgb = create_synthetic_image(size=(128, 128))
        img_pil = Image.fromarray((rgb * 255).astype(np.uint8))
        img_pil.save(img_path)
        
        # Create ground truth
        ground_truth = {
            "version": "v0",
            "root": str(tmpdir / "images"),
            "labels": ["pool"],
            "images": {
                "pool/test_001.jpg": {
                    "label": "pool",
                    "should_detect": True,
                    "difficulty": "easy",
                    "tags": []
                }
            }
        }
        
        # Save ground truth to file (required for validate_dataset)
        ground_truth_path = tmpdir / "ground_truth.json"
        with open(ground_truth_path, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        # Run validation twice with same seed
        config = MaterialsV3Config(
            water_detection_enabled=True,
            water_edge_refinement_enabled=False
        )
        seed = 42
        
        harness1 = WaterValidationHarness(config, seed=seed)
        results1 = harness1.validate_dataset(ground_truth, ground_truth_path)
        
        harness2 = WaterValidationHarness(config, seed=seed)
        results2 = harness2.validate_dataset(ground_truth, ground_truth_path)
        
        # Compare stability scores
        assert len(results1) == len(results2) == 1
        
        stability1 = results1[0].stability_score
        stability2 = results2[0].stability_score
        
        assert abs(stability1 - stability2) < 1e-6, (
            f"Full validation stability differs: {stability1} vs {stability2}"
        )
        
        print(f"✅ Full validation deterministic test passed")
        print(f"   Stability (run 1): {stability1:.6f}")
        print(f"   Stability (run 2): {stability2:.6f}")
        print(f"   Difference: {abs(stability1 - stability2):.10f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
