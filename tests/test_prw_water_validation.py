#!/usr/bin/env python3
"""
Tests for PR-W4: Water Validation Harness

Validates:
- ValidationResult dataclass schema
- Edge alignment computation (primary metric)
- Stability computation across perturbations
- Boundary extraction
- False positive detection
- Report generation
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Import validation components (must modify sys.path first)
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from prw_water_validation import (  # noqa: E402
    ValidationResult,
    WaterValidationHarness,
)

from lux_depth_v2.materials_v3 import MaterialsV3Config  # noqa: E402


def create_test_image(size=(256, 256), pool_region=True):
    """Create a synthetic test image with pool-like colors."""
    h, w = size
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    if pool_region:
        # Blue pool region in center
        rgb[h//4:3*h//4, w//4:3*w//4, 2] = 0.6  # Blue channel
        rgb[h//4:3*h//4, w//4:3*w//4, 1] = 0.4  # Some green
    else:
        # Green foliage (non-water)
        rgb[:, :, 1] = 0.5  # Green channel
        rgb[:, :, 0] = 0.2  # Some red

    return rgb


def test_validation_result_dataclass():
    """Verify ValidationResult schema has all required fields (v0 schema)."""
    result = ValidationResult(
        image_path="test.jpg",
        scene_type="pool",
        should_detect=True,
        difficulty="medium",
        tags=["test"],
        detected=True,  # Canonical detection flag from water_candidate.present
        coverage=0.25,
        coverage_px=1000,
        confidence=0.75,
        source="heuristic",
        implementation="stub_v0",
        edge_alignment_score=0.65,
        boundary_px=500,
        stability_score=0.85,
        is_false_positive=False,
        is_false_trigger=False,
        processing_time_ms=45.2
    )

    # Verify all fields are present
    assert result.image_path == "test.jpg"
    assert result.scene_type == "pool"
    assert result.should_detect is True
    assert result.difficulty == "medium"
    assert result.tags == ["test"]
    assert result.detected is True
    assert result.coverage == 0.25
    assert result.coverage_px == 1000
    assert result.confidence == 0.75
    assert result.source == "heuristic"
    assert result.implementation == "stub_v0"
    assert result.edge_alignment_score == 0.65
    assert result.boundary_px == 500
    assert result.stability_score == 0.85
    assert result.is_false_positive is False
    assert result.is_false_trigger is False
    assert result.processing_time_ms == 45.2


def test_edge_alignment_computation():
    """Verify edge alignment metric computation."""
    pytest.importorskip("scipy")  # Skip if scipy not available
    
    config = MaterialsV3Config(
        water_detection_enabled=True,
        enabled=False  # Disable Materials V3 to avoid ML dependencies
    )
    harness = WaterValidationHarness(config)

    # Create test image with edges
    rgb = create_test_image(size=(128, 128), pool_region=True)

    # Create mask with sharp boundary
    mask = np.zeros((128, 128), dtype=np.float32)
    mask[32:96, 32:96] = 1.0  # Center square

    # Compute edge alignment
    edge_score = harness._compute_edge_alignment(rgb, mask)

    # Should be between 0 and 1
    assert 0.0 <= edge_score <= 1.0

    # Test with None mask
    edge_score_none = harness._compute_edge_alignment(rgb, None)
    assert edge_score_none == 0.0


def test_boundary_extraction():
    """Verify boundary extraction from mask."""
    pytest.importorskip("scipy")  # Skip if scipy not available
    
    config = MaterialsV3Config(
        water_detection_enabled=True,
        enabled=False
    )
    harness = WaterValidationHarness(config)

    # Create simple mask
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[16:48, 16:48] = 1.0  # Center square

    # Extract boundary (width=3)
    boundary = harness._extract_boundary(mask, width=3)

    # Boundary should be non-zero
    assert np.sum(boundary) > 0

    # Boundary should be at edges, not in center or far outside
    assert boundary[32, 32] == 0  # Center should be 0
    assert boundary[0, 0] == 0  # Far outside should be 0

    # Test with None mask
    boundary_none = harness._extract_boundary(None, width=3)
    assert boundary_none.shape == (1, 1)
    assert np.sum(boundary_none) == 0


def test_count_boundary_pixels():
    """Verify boundary pixel counting."""
    pytest.importorskip("scipy")  # Skip if scipy not available
    
    config = MaterialsV3Config(
        water_detection_enabled=True,
        enabled=False
    )
    harness = WaterValidationHarness(config)

    # Create mask
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[16:48, 16:48] = 1.0

    # Count boundary pixels
    count = harness._count_boundary_pixels(mask)

    # Should be positive
    assert count > 0

    # Test with None mask
    count_none = harness._count_boundary_pixels(None)
    assert count_none == 0


def test_stability_computation():
    """Verify stability metric computation (deterministic with seed)."""
    config = MaterialsV3Config(
        water_detection_enabled=True,
        enabled=False
    )
    harness = WaterValidationHarness(config, seed=42)

    # Create test image
    rgb = create_test_image(size=(128, 128), pool_region=True)
    depth = np.ones((128, 128), dtype=np.float32)

    # Compute stability twice (should be identical with seed)
    stability1 = harness._compute_stability(rgb, depth)
    harness.seed = 42  # Reset seed
    np.random.seed(42)
    stability2 = harness._compute_stability(rgb, depth)

    # Should be between 0 and 1
    assert 0.0 <= stability1 <= 1.0
    assert 0.0 <= stability2 <= 1.0
    
    # Should be deterministic (same results)
    assert abs(stability1 - stability2) < 0.01


def test_false_trigger_detection():
    """Verify false trigger detection logic (should_detect=false)."""
    config = MaterialsV3Config(
        water_detection_enabled=True,
        enabled=False
    )
    harness = WaterValidationHarness(config, seed=42)

    # Create hard negative image (blue but not water)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.fromarray((create_test_image(pool_region=False) * 255).astype(np.uint8))
        img.save(f.name)
        temp_path = Path(f.name)

    try:
        # Validate as hard negative (should_detect=false)
        result = harness.validate_single(
            temp_path, 
            label="pool",
            should_detect=False,
            difficulty="hard",
            tags=["hard_negative"]
        )

        # Verify schema
        assert result.should_detect is False
        assert result.scene_type == "pool"  # Still in pool folder
        assert "hard_negative" in result.tags

        # If water is detected despite should_detect=false, mark as false trigger
        if result.coverage > 0 and result.confidence > 0:
            assert result.is_false_trigger is True
            assert result.is_false_positive is True  # Legacy alias should match
        else:
            assert result.is_false_trigger is False
            assert result.is_false_positive is False  # Legacy alias should match

    finally:
        temp_path.unlink()


def test_validate_single_image():
    """Test single image validation (v0 schema)."""
    config = MaterialsV3Config(
        water_detection_enabled=True,
        enabled=False
    )
    harness = WaterValidationHarness(config, seed=42)

    # Create test image
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.fromarray((create_test_image(pool_region=True) * 255).astype(np.uint8))
        img.save(f.name)
        temp_path = Path(f.name)

    try:
        result = harness.validate_single(
            temp_path, 
            label="pool",
            should_detect=True,
            difficulty="easy",
            tags=[]
        )

        # Verify result structure
        assert isinstance(result, ValidationResult)
        assert result.image_path == str(temp_path)
        assert result.scene_type == "pool"
        assert result.should_detect is True
        assert result.difficulty == "easy"
        assert result.tags == []
        assert result.processing_time_ms >= 0
        assert 0.0 <= result.edge_alignment_score <= 1.0
        assert 0.0 <= result.stability_score <= 1.0
        assert result.boundary_px >= 0
        assert result.is_false_positive is False
        assert isinstance(result.is_false_trigger, bool)

    finally:
        temp_path.unlink()


def test_validate_dataset():
    """Test dataset validation (v0 schema)."""
    config = MaterialsV3Config(
        water_detection_enabled=True,
        enabled=False
    )
    harness = WaterValidationHarness(config, seed=42)

    # Create test dataset
    temp_dir = Path(tempfile.mkdtemp())
    pool_dir = temp_dir / "pool"
    pool_dir.mkdir()

    try:
        # Create 2 test images
        pool_img = pool_dir / "pool_001.jpg"
        neg_img = pool_dir / "neg_blue_wall_001.jpg"
        
        img1 = Image.fromarray((create_test_image(pool_region=True) * 255).astype(np.uint8))
        img1.save(pool_img)
        
        img2 = Image.fromarray((create_test_image(pool_region=False) * 255).astype(np.uint8))
        img2.save(neg_img)

        # Ground truth (v0 schema)
        ground_truth = {
            "version": "v0",
            "root": str(temp_dir),
            "labels": ["pool"],
            "images": {
                "pool/pool_001.jpg": {
                    "label": "pool",
                    "should_detect": True,
                    "difficulty": "easy",
                    "tags": []
                },
                "pool/neg_blue_wall_001.jpg": {
                    "label": "pool",
                    "should_detect": False,
                    "difficulty": "hard",
                    "tags": ["hard_negative"]
                }
            }
        }

        # Validate dataset
        results = harness.validate_dataset(ground_truth)

        # Verify results
        assert len(results) == 2
        assert all(isinstance(r, ValidationResult) for r in results)
        
        # Check schema fields
        assert results[0].should_detect is True
        assert results[1].should_detect is False
        assert results[1].tags == ["hard_negative"]

    finally:
        # Cleanup
        pool_img.unlink(missing_ok=True)
        neg_img.unlink(missing_ok=True)
        pool_dir.rmdir()
        temp_dir.rmdir()


def test_report_generation():
    """Verify JSON report structure (v0 schema)."""
    config = MaterialsV3Config(
        water_detection_enabled=True,
        enabled=False
    )
    harness = WaterValidationHarness(config, seed=42)

    # Create mock results (v0 schema)
    results = [
        ValidationResult(
            image_path="pool_001.jpg",
            scene_type="pool",
            should_detect=True,
            difficulty="easy",
            tags=[],
            detected=True,
            coverage=0.30,
            coverage_px=1200,
            confidence=0.80,
            source="heuristic",
            implementation="stub_v0",
            edge_alignment_score=0.65,
            boundary_px=450,
            stability_score=0.85,
            is_false_positive=False,
            is_false_trigger=False,
            processing_time_ms=42.0
        ),
        ValidationResult(
            image_path="ocean_001.jpg",
            scene_type="ocean",
            should_detect=True,
            difficulty="medium",
            tags=[],
            detected=True,
            coverage=0.55,
            coverage_px=2200,
            confidence=0.75,
            source="heuristic",
            implementation="stub_v0",
            edge_alignment_score=0.70,
            boundary_px=600,
            stability_score=0.80,
            is_false_positive=False,
            is_false_trigger=False,
            processing_time_ms=45.0
        ),
        ValidationResult(
            image_path="neg_blue_wall_001.jpg",
            scene_type="pool",
            should_detect=False,
            difficulty="hard",
            tags=["hard_negative"],
            detected=False,
            coverage=0.0,
            coverage_px=0,
            confidence=0.0,
            source="none",
            implementation="stub_v0",
            edge_alignment_score=0.0,
            boundary_px=0,
            stability_score=1.0,
            is_false_positive=False,
            is_false_trigger=False,
            processing_time_ms=38.0
        )
    ]

    # Mock ground truth
    ground_truth = {
        "version": "v0",
        "labels": ["pool", "ocean"]
    }

    # Generate report
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        output_path = Path(f.name)

    try:
        harness.generate_report(results, output_path, ground_truth)

        # Load and verify report
        with open(output_path) as f:
            report = json.load(f)

        # Verify structure
        assert "summary" in report
        assert "results" in report

        # Verify summary fields (v0 schema)
        summary = report["summary"]
        assert summary["dataset_version"] == "v0"
        assert summary["total_images"] == 3
        assert summary["pool_images"] == 2  # 1 true + 1 neg
        assert summary["ocean_images"] == 1
        assert summary["should_detect_true"] == 2
        assert summary["should_detect_false"] == 1
        
        # New fields
        assert "pool_recall" in summary
        assert "ocean_recall" in summary
        assert "pool_median_coverage" in summary
        assert "ocean_median_coverage" in summary
        assert "false_trigger_count" in summary
        assert "false_trigger_rate" in summary
        
        # Backward compatibility
        assert "false_positive_count" in summary
        assert summary["false_positive_count"] == 0
        assert summary["false_positive_rate"] == 0.0

        # Verify results array
        assert len(report["results"]) == 3

        # Verify individual result structure
        result = report["results"][0]
        assert "image_path" in result
        assert "scene_type" in result
        assert "should_detect" in result
        assert "difficulty" in result
        assert "tags" in result
        assert "coverage" in result
        assert "confidence" in result
        assert "edge_alignment_score" in result
        assert "boundary_px" in result
        assert "stability_score" in result
        assert "is_false_positive" in result
        assert "is_false_trigger" in result
        assert "processing_time_ms" in result

    finally:
        output_path.unlink()


def test_report_summary_statistics():
    """Verify summary statistics calculations (v0 schema)."""
    config = MaterialsV3Config(
        water_detection_enabled=True,
        enabled=False
    )
    harness = WaterValidationHarness(config, seed=42)

    # Create results with known values (v0 schema)
    results = [
        ValidationResult(
            image_path="pool_001.jpg",
            scene_type="pool",
            should_detect=True,
            difficulty="easy",
            tags=[],
            detected=True,
            coverage=0.30,
            coverage_px=1200,
            confidence=0.80,
            source="heuristic",
            implementation="stub_v0",
            edge_alignment_score=0.60,
            boundary_px=450,
            stability_score=0.80,
            is_false_positive=False,
            is_false_trigger=False,
            processing_time_ms=40.0
        ),
        ValidationResult(
            image_path="pool_002.jpg",
            scene_type="pool",
            should_detect=True,
            difficulty="medium",
            tags=[],
            detected=True,
            coverage=0.40,
            coverage_px=1600,
            confidence=0.85,
            source="heuristic",
            implementation="stub_v0",
            edge_alignment_score=0.70,
            boundary_px=500,
            stability_score=0.90,
            is_false_positive=False,
            is_false_trigger=False,
            processing_time_ms=42.0
        ),
        ValidationResult(
            image_path="neg_blue_wall_001.jpg",
            scene_type="pool",
            should_detect=False,
            difficulty="hard",
            tags=["hard_negative"],
            detected=True,
            coverage=0.05,
            coverage_px=200,
            confidence=0.20,
            source="heuristic",
            implementation="stub_v0",
            edge_alignment_score=0.10,
            boundary_px=50,
            stability_score=0.95,
            is_false_positive=True,  # Legacy alias for is_false_trigger
            is_false_trigger=True,  # Detected despite should_detect=false
            processing_time_ms=38.0
        )
    ]

    # Mock ground truth
    ground_truth = {
        "version": "v0",
        "labels": ["pool"]
    }

    # Generate report
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        output_path = Path(f.name)

    try:
        harness.generate_report(results, output_path, ground_truth)

        with open(output_path) as f:
            report = json.load(f)

        summary = report["summary"]

        # Verify recall (2 detected out of 2 should_detect=true)
        assert summary["pool_recall"] == pytest.approx(1.0, abs=0.01)

        # Verify coverage (mean + median for detected images)
        assert summary["pool_avg_coverage"] == pytest.approx(0.35, abs=0.01)  # (0.30 + 0.40) / 2
        assert summary["pool_median_coverage"] == pytest.approx(0.35, abs=0.01)  # median([0.30, 0.40])

        # Verify edge alignment (only for detected water)
        assert summary["pool_avg_edge_alignment"] == pytest.approx(0.65, abs=0.01)  # (0.60 + 0.70) / 2

        # Verify stability (for should_detect=true images)
        assert summary["pool_avg_stability"] == pytest.approx(0.85, abs=0.01)  # (0.80 + 0.90) / 2

        # Verify false trigger rate (1 out of 1 should_detect=false)
        assert summary["false_trigger_count"] == 1
        assert summary["false_trigger_rate"] == pytest.approx(1.0, abs=0.01)

        # Verify backward compatibility (should match false_trigger fields)
        assert summary["false_positive_count"] == 1
        assert summary["false_positive_rate"] == pytest.approx(1.0, abs=0.01)

        # Verify performance
        assert summary["overall_avg_processing_time_ms"] == pytest.approx(40.0, abs=0.1)  # (40 + 42 + 38) / 3

    finally:
        output_path.unlink()


def test_edge_alignment_with_strong_edges():
    """Test edge alignment with image containing strong edges."""
    pytest.importorskip("scipy")  # Skip if scipy not available
    
    config = MaterialsV3Config(
        water_detection_enabled=True,
        enabled=False
    )
    harness = WaterValidationHarness(config, seed=42)

    # Create image with strong edges
    rgb = np.zeros((128, 128, 3), dtype=np.float32)
    rgb[:64, :, :] = 0.0  # Top half black
    rgb[64:, :, :] = 1.0  # Bottom half white

    # Create mask aligned with edge
    mask = np.zeros((128, 128), dtype=np.float32)
    mask[64:, :] = 1.0  # Bottom half

    edge_score = harness._compute_edge_alignment(rgb, mask)

    # Should have high alignment since mask boundary matches strong edge
    assert edge_score > 0.3  # Should detect some alignment


def test_edge_alignment_with_misaligned_mask():
    """Test edge alignment with mask that doesn't match edges."""
    pytest.importorskip("scipy")  # Skip if scipy not available
    
    config = MaterialsV3Config(
        water_detection_enabled=True,
        enabled=False
    )
    harness = WaterValidationHarness(config, seed=42)

    # Create image with horizontal edge
    rgb = np.zeros((128, 128, 3), dtype=np.float32)
    rgb[:64, :, :] = 0.0
    rgb[64:, :, :] = 1.0

    # Create mask with vertical split (misaligned)
    mask = np.zeros((128, 128), dtype=np.float32)
    mask[:, 64:] = 1.0  # Right half

    edge_score = harness._compute_edge_alignment(rgb, mask)

    # Alignment should be lower than perfectly aligned case
    assert 0.0 <= edge_score <= 1.0


def test_edge_alignment_with_detector_enabled():
    """Edge alignment computed when water detection enabled."""
    pytest.importorskip("scipy")  # Skip if scipy not available
    
    config = MaterialsV3Config(
        water_detection_enabled=True,
        enabled=False
    )
    harness = WaterValidationHarness(config, seed=42)
    
    # Create synthetic pool-like image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        rgb[50:200, 50:200, 2] = 180  # Blue region
        rgb[50:200, 50:200, 1] = 76   # Some green
        img = Image.fromarray(rgb)
        img.save(f.name)
        img_path = Path(f.name)
    
    try:
        result = harness.validate_single(
            img_path, 
            label="pool",
            should_detect=True,
            difficulty="easy",
            tags=[]
        )
        
        # Verify mask-based metrics are computed (not fallback 0.0)
        # Stub detector should produce a mask for this blue image
        assert result.edge_alignment_score > 0.0, "Should compute real edge score when mask available"
        assert result.boundary_px > 0, "Should count actual boundary pixels when mask available"
        
        # Also verify other fields populated
        assert result.coverage >= 0.0
        assert result.confidence >= 0.0
        assert result.should_detect is True
        
    finally:
        img_path.unlink()  # Cleanup
