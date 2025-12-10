"""
Unit tests for autotune_helpers module.

Phase 2 Slice 3: ImageStats computation and scene complexity estimation.
"""
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from transformation_portal.core.storage.autotune_helpers import (
    ImageStats,
    compute_image_stats,
    _estimate_scene_complexity,
)


def test_image_stats_immutable():
    """Verify ImageStats is frozen (immutable)."""
    stats = ImageStats(width=1000, height=750, megapixels=0.75)
    
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        stats.width = 2000


def test_compute_image_stats_from_array():
    """Compute stats from pre-loaded array (no file I/O)."""
    # Create test array (750 x 1000 x 3)
    rgb = np.random.rand(750, 1000, 3).astype(np.float32)
    
    stats = compute_image_stats(Path("dummy.jpg"), rgb_array=rgb)
    
    assert stats.width == 1000
    assert stats.height == 750
    assert abs(stats.megapixels - 0.75) < 0.01
    assert stats.scene_complexity is not None
    assert 0.0 <= stats.scene_complexity <= 1.0


def test_compute_image_stats_from_file(tmp_path):
    """Compute stats from file path (loads image)."""
    from PIL import Image
    
    # Create test image
    img_path = tmp_path / "test.jpg"
    arr = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_path)
    
    stats = compute_image_stats(img_path)
    
    assert stats.width == 800
    assert stats.height == 600
    assert abs(stats.megapixels - 0.48) < 0.01
    # Complexity is None when loaded from file only (no array)
    assert stats.scene_complexity is None


def test_estimate_scene_complexity_simple():
    """Simple gradient (sky-like) should have low complexity."""
    # Create smooth gradient (sky-like)
    H, W = 500, 1000
    x = np.linspace(0, 1, W, dtype=np.float32)
    y = np.linspace(0, 1, H, dtype=np.float32)[:, np.newaxis]
    rgb = np.stack([x + y, x + y, x + y], axis=-1) * 0.5
    
    complexity = _estimate_scene_complexity(rgb)
    
    # Should be low (homogeneous regions)
    assert 0.0 <= complexity < 0.5


def test_estimate_scene_complexity_random():
    """Random noise (texture-like) should have high complexity."""
    # Random high-frequency content
    rgb = np.random.rand(500, 1000, 3).astype(np.float32)
    
    complexity = _estimate_scene_complexity(rgb)
    
    # Should be higher (lots of edges)
    assert 0.3 <= complexity <= 1.0


def test_estimate_scene_complexity_grayscale():
    """Should handle grayscale input (2D array)."""
    gray = np.random.rand(500, 1000).astype(np.float32)
    
    complexity = _estimate_scene_complexity(gray)
    
    assert 0.0 <= complexity <= 1.0


def test_compute_image_stats_megapixels():
    """Verify megapixels calculation for various sizes."""
    test_cases = [
        (1000, 750, 0.75),
        (4000, 3000, 12.0),
        (6000, 4000, 24.0),
        (8192, 5464, 44.7),  # ~45 MP
    ]
    
    for width, height, expected_mp in test_cases:
        rgb = np.zeros((height, width, 3), dtype=np.float32)
        stats = compute_image_stats(Path("dummy.jpg"), rgb_array=rgb)
        
        assert stats.width == width
        assert stats.height == height
        assert abs(stats.megapixels - expected_mp) < 0.1


def test_compute_image_stats_complexity_none_without_array():
    """Complexity should be None when no array provided."""
    with patch('PIL.Image') as mock_image:
        mock_img = MagicMock()
        mock_img.size = (800, 600)
        mock_image.open.return_value.__enter__ = MagicMock(return_value=mock_img)
        mock_image.open.return_value.__exit__ = MagicMock()
        
        stats = compute_image_stats(Path("dummy.jpg"))
        
        assert stats.width == 800
        assert stats.height == 600
        assert stats.scene_complexity is None


def test_estimate_scene_complexity_bounds():
    """Complexity score should always be in [0, 1]."""
    # Extreme cases
    test_arrays = [
        np.zeros((100, 100, 3), dtype=np.float32),  # Uniform black
        np.ones((100, 100, 3), dtype=np.float32),   # Uniform white
        np.random.rand(100, 100, 3).astype(np.float32),  # Random
        np.tile(np.arange(100), (100, 1)).astype(np.float32) / 100,  # Gradient
    ]
    
    for arr in test_arrays:
        complexity = _estimate_scene_complexity(arr)
        assert 0.0 <= complexity <= 1.0, f"Complexity {complexity} out of bounds"


def test_image_stats_json_serializable():
    """Verify ImageStats can be converted to dict for JSON serialization."""
    import dataclasses
    
    stats = ImageStats(
        width=1920,
        height=1080,
        megapixels=2.07,
        scene_complexity=0.45
    )
    
    # Convert to dict
    stats_dict = dataclasses.asdict(stats)
    
    assert stats_dict["width"] == 1920
    assert stats_dict["height"] == 1080
    assert abs(stats_dict["megapixels"] - 2.07) < 0.01
    assert abs(stats_dict["scene_complexity"] - 0.45) < 0.01
    
    # Should be JSON-serializable
    import json
    json_str = json.dumps(stats_dict)
    assert len(json_str) > 0
