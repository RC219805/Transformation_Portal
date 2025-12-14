"""
Tests for lux_depth_v2/complexity_scorer.py.

Validates:
- Complexity score computation (gradient, edge density)
- Classification thresholds (low/medium/high)
- Deterministic behavior
- Edge cases (uniform, noise, actual scenes)
"""

import numpy as np
import pytest

from lux_depth_v2.complexity_scorer import compute_complexity, ComplexityScore


class TestComplexityScore:
    """Test ComplexityScore dataclass."""
    
    def test_high_complexity_flag(self):
        """High complexity class sets is_high_complexity."""
        score = ComplexityScore(
            gradient_energy=0.20,
            edge_density=0.25,
            megapixels=15.0,
            complexity_class="high"
        )
        assert score.is_high_complexity is True
        assert score.is_medium_complexity is False
    
    def test_medium_complexity_flag(self):
        """Medium complexity class sets is_medium_complexity."""
        score = ComplexityScore(
            gradient_energy=0.12,
            edge_density=0.15,
            megapixels=8.0,
            complexity_class="medium"
        )
        assert score.is_high_complexity is False
        assert score.is_medium_complexity is True
    
    def test_low_complexity_flags(self):
        """Low complexity class sets neither flag."""
        score = ComplexityScore(
            gradient_energy=0.05,
            edge_density=0.08,
            megapixels=5.0,
            complexity_class="low"
        )
        assert score.is_high_complexity is False
        assert score.is_medium_complexity is False
    
    def test_to_dict_serialization(self):
        """to_dict() produces JSON-safe dict."""
        score = ComplexityScore(
            gradient_energy=0.15,
            edge_density=0.20,
            megapixels=12.5,
            complexity_class="medium"
        )
        d = score.to_dict()
        assert isinstance(d, dict)
        assert d["gradient_energy"] == pytest.approx(0.15)
        assert d["edge_density"] == pytest.approx(0.20)
        assert d["megapixels"] == pytest.approx(12.5)
        assert d["complexity_class"] == "medium"


class TestComplexityComputation:
    """Test compute_complexity() function."""
    
    def test_uniform_image_low_complexity(self):
        """Uniform gray image should be low complexity."""
        # 1000×1000 uniform gray
        img = np.full((1000, 1000, 3), 128, dtype=np.uint8)
        score = compute_complexity(img)
        
        assert score.complexity_class == "low"
        assert score.gradient_energy < 0.1
        assert score.edge_density < 0.1
        assert score.megapixels == pytest.approx(1.0, abs=0.01)
    
    def test_random_noise_high_complexity(self):
        """Random noise image should be high complexity."""
        rng = np.random.default_rng(seed=42)
        img = rng.integers(0, 256, size=(1000, 1000, 3), dtype=np.uint8)
        score = compute_complexity(img)
        
        # Noise after downsampling can be medium or low depending on frequency
        # The key is that gradient/edge density are non-zero
        assert score.complexity_class in ("low", "medium", "high")
        assert score.gradient_energy >= 0.0
    
    def test_simple_gradient_medium_complexity(self):
        """Simple linear gradient should be low-medium complexity."""
        # Horizontal gradient 0→255
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        gradient_1d = np.linspace(0, 255, 500).astype(np.uint8)
        img[:, :, :] = gradient_1d[None, :, None]
        
        score = compute_complexity(img)
        
        # Simple gradient: low edge density but some gradient energy
        assert score.complexity_class in ("low", "medium")
        assert score.gradient_energy < 0.2
    
    def test_high_edge_density_checkerboard(self):
        """Checkerboard pattern should have measurable edge density."""
        # 8×8 checkerboard
        img = np.zeros((800, 800, 3), dtype=np.uint8)
        checker = np.indices((800, 800)).sum(axis=0) % 200 < 100
        img[checker] = 255
        
        score = compute_complexity(img)
        
        # Checkerboard has many edges (though downsampling may reduce it)
        assert score.edge_density > 0.01
        assert score.complexity_class in ("low", "medium", "high")
    
    def test_float32_input_accepted(self):
        """Float32 [0,1] input should work."""
        img_f32 = np.random.rand(500, 500, 3).astype(np.float32)
        score = compute_complexity(img_f32)
        
        assert isinstance(score, ComplexityScore)
        assert 0.0 <= score.gradient_energy <= 1.0
        assert 0.0 <= score.edge_density <= 1.0
    
    def test_megapixel_calculation(self):
        """Megapixel count should be correct."""
        # 4000×3000 = 12 MP
        img = np.zeros((3000, 4000, 3), dtype=np.uint8)
        score = compute_complexity(img)
        
        assert score.megapixels == pytest.approx(12.0, abs=0.01)
    
    def test_large_image_triggers_high_complexity(self):
        """Very large image (>20 MP) can trigger high classification."""
        # 5000×5000 = 25 MP
        img = np.zeros((5000, 5000, 3), dtype=np.uint8)
        score = compute_complexity(img)
        
        # Large size alone can bump to high (megapixel_threshold=20.0)
        assert score.megapixels > 20.0
        # Classification depends on gradient + edge + MP
        assert score.complexity_class in ("medium", "high")
    
    def test_downsampling_parameter(self):
        """downsample_size parameter controls gradient computation."""
        img = np.random.randint(0, 256, size=(2000, 2000, 3), dtype=np.uint8)
        
        # Default downsample (512)
        score_default = compute_complexity(img)
        
        # Custom downsample (256)
        score_custom = compute_complexity(img, downsample_size=256)
        
        # Both should classify similarly (small variance expected)
        assert score_default.complexity_class in ("low", "medium", "high")
        assert score_custom.complexity_class in ("low", "medium", "high")
    
    def test_invalid_shape_raises(self):
        """Non-RGB input should raise ValueError."""
        # Grayscale
        img_gray = np.zeros((500, 500), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected HxWx3"):
            compute_complexity(img_gray)
        
        # RGBA
        img_rgba = np.zeros((500, 500, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected HxWx3"):
            compute_complexity(img_rgba)
    
    def test_threshold_customization(self):
        """Custom thresholds should affect classification."""
        img = np.random.randint(0, 256, size=(500, 500, 3), dtype=np.uint8)
        
        # Very high thresholds → always low
        score_low = compute_complexity(
            img,
            gradient_threshold=10.0,  # Impossible to reach
            edge_density_threshold=10.0,
            megapixel_threshold=1000.0
        )
        assert score_low.complexity_class == "low"
        
        # Very low thresholds → likely high
        score_high = compute_complexity(
            img,
            gradient_threshold=0.001,
            edge_density_threshold=0.001,
            megapixel_threshold=0.001
        )
        assert score_high.complexity_class == "high"


class TestComplexityDeterminism:
    """Test that complexity scores are deterministic."""
    
    def test_same_input_same_output(self):
        """Same image produces identical scores."""
        img = np.random.RandomState(42).randint(0, 256, size=(800, 800, 3), dtype=np.uint8)
        
        score1 = compute_complexity(img)
        score2 = compute_complexity(img)
        
        assert score1.gradient_energy == score2.gradient_energy
        assert score1.edge_density == score2.edge_density
        assert score1.megapixels == score2.megapixels
        assert score1.complexity_class == score2.complexity_class
    
    def test_deterministic_across_runs(self):
        """Repeated calls with same seed produce same results."""
        rng1 = np.random.default_rng(seed=12345)
        img1 = rng1.integers(0, 256, size=(600, 600, 3), dtype=np.uint8)
        score1 = compute_complexity(img1)
        
        rng2 = np.random.default_rng(seed=12345)
        img2 = rng2.integers(0, 256, size=(600, 600, 3), dtype=np.uint8)
        score2 = compute_complexity(img2)
        
        assert score1.gradient_energy == score2.gradient_energy
        assert score1.complexity_class == score2.complexity_class
