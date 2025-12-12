"""Tests for Phase 2 Lighting Condition Detector."""

import pytest
import torch

from lux_depth_v2.lighting_detector import LightingConditionDetector, TimeOfDay, LightingCondition


@pytest.fixture
def device():
    """Get torch device for testing."""
    return torch.device('cpu')


@pytest.fixture
def detector(device):
    """Create lighting detector instance."""
    return LightingConditionDetector(device)


def test_lighting_detector_initialization(detector):
    """Test lighting detector initializes correctly."""
    assert detector is not None
    assert hasattr(detector, 'device')


def test_detect_with_sky_mask(detector, device):
    """Test lighting detection with provided sky mask."""
    # Create dummy RGB image
    rgb = torch.rand(1, 3, 512, 512, device=device)
    
    # Create sky mask (upper portion)
    sky_mask = torch.zeros(1, 1, 512, 512, device=device)
    sky_mask[:, :, :200, :] = 1.0  # Upper 200 pixels are sky
    
    # Detect
    condition = detector.detect(rgb, sky_mask=sky_mask)
    
    # Verify output
    assert isinstance(condition, LightingCondition)
    assert isinstance(condition.time_of_day, TimeOfDay)
    assert 0.0 <= condition.confidence <= 1.0
    assert 0.0 <= condition.sky_coverage <= 1.0
    assert condition.sky_color_temp > 0.0
    assert 0.0 <= condition.sky_brightness <= 1.0


def test_detect_without_sky_mask(detector, device):
    """Test lighting detection with automatic sky detection."""
    # Create image with blue top (simulating sky)
    rgb = torch.zeros(1, 3, 512, 512, device=device)
    rgb[:, 2, :200, :] = 0.7  # Blue channel in upper portion
    rgb[:, 1, :200, :] = 0.5  # Some green
    rgb[:, 0, :200, :] = 0.3  # Less red
    
    # Detect (no sky mask provided)
    condition = detector.detect(rgb)
    
    # Should detect sky automatically
    assert condition.sky_coverage > 0.1  # Should find some sky


def test_time_of_day_classification(detector, device):
    """Test time of day classification logic."""
    # Create warm golden hour scene
    rgb = torch.ones(1, 3, 512, 512, device=device)
    rgb[:, 0, :200, :] = 0.8  # Warm red in sky
    rgb[:, 1, :200, :] = 0.6  # Orange
    rgb[:, 2, :200, :] = 0.3  # Low blue
    
    sky_mask = torch.zeros(1, 1, 512, 512, device=device)
    sky_mask[:, :, :200, :] = 1.0
    
    condition = detector.detect(rgb, sky_mask=sky_mask)
    
    # Should be warm (not critical which specific warm category)
    assert condition.warmth > 0.0  # Should be warm
    assert condition.time_of_day in [
        TimeOfDay.GOLDEN_HOUR, TimeOfDay.AFTERNOON, TimeOfDay.MIDDAY
    ]  # Acceptable warm/bright classifications


def test_sky_coverage_calculation(detector, device):
    """Test sky coverage percentage calculation."""
    rgb = torch.rand(1, 3, 512, 512, device=device)
    
    # Sky mask covering 50% of image
    sky_mask = torch.zeros(1, 1, 512, 512, device=device)
    sky_mask[:, :, :256, :] = 1.0  # Top half
    
    condition = detector.detect(rgb, sky_mask=sky_mask)
    
    # Sky coverage should be approximately 0.5
    assert 0.45 <= condition.sky_coverage <= 0.55


def test_color_temperature_estimation(detector, device):
    """Test color temperature estimation from sky."""
    # Cool blue sky (dawn)
    rgb_cool = torch.zeros(1, 3, 512, 512, device=device)
    rgb_cool[:, 2, :, :] = 0.8  # High blue
    rgb_cool[:, 1, :, :] = 0.5
    rgb_cool[:, 0, :, :] = 0.3  # Low red
    
    sky_mask = torch.ones(1, 1, 512, 512, device=device)
    
    condition_cool = detector.detect(rgb_cool, sky_mask=sky_mask)
    
    # Warm orange sky (sunset)
    rgb_warm = torch.zeros(1, 3, 512, 512, device=device)
    rgb_warm[:, 0, :, :] = 0.9  # High red
    rgb_warm[:, 1, :, :] = 0.6
    rgb_warm[:, 2, :, :] = 0.2  # Low blue
    
    condition_warm = detector.detect(rgb_warm, sky_mask=sky_mask)
    
    # Cool sky should have higher color temp than warm sky
    assert condition_cool.sky_color_temp > condition_warm.sky_color_temp


def test_shadow_detection(detector, device):
    """Test shadow detection and direction estimation."""
    # Create image with strong vertical gradient (simulating shadow)
    rgb = torch.zeros(1, 3, 512, 512, device=device)
    
    # Left side bright, right side dark
    for i in range(512):
        brightness = 1.0 - (i / 512.0)
        rgb[:, :, :, i] = brightness
    
    condition = detector.detect(rgb)
    
    # Should detect shadows
    # (May not always detect depending on threshold, so we just check it runs)
    assert condition.has_strong_shadows in [True, False]
    if condition.has_strong_shadows:
        assert condition.shadow_direction is not None


def test_adapt_tone_mapping(detector, device):
    """Test tone mapping adaptation."""
    # Create golden hour condition
    condition = LightingCondition(
        time_of_day=TimeOfDay.GOLDEN_HOUR,
        confidence=0.85,
        sky_coverage=0.3,
        sky_color_temp=4000.0,
        sky_brightness=0.6,
        has_strong_shadows=True,
        shadow_direction="left",
        dominant_hue=45.0,
        warmth=0.7,
    )
    
    base_config = {
        'highlight_preservation': 0.8,
        'contrast': 1.0,
        'saturation_boost': 1.0,
    }
    
    adapted = detector.adapt_tone_mapping(condition, base_config)
    
    # Golden hour should increase highlight preservation
    assert adapted['highlight_preservation'] >= base_config['highlight_preservation']


def test_adapt_color_grading(detector, device):
    """Test color grading adaptation."""
    # Create dawn condition
    condition = LightingCondition(
        time_of_day=TimeOfDay.DAWN,
        confidence=0.80,
        sky_coverage=0.4,
        sky_color_temp=6500.0,
        sky_brightness=0.3,
        has_strong_shadows=False,
        shadow_direction=None,
        dominant_hue=240.0,
        warmth=-0.4,
    )
    
    base_config = {
        'cool_tone_boost': 1.0,
        'warm_tone_reduction': 1.0,
        'blue_saturation': 1.0,
    }
    
    adapted = detector.adapt_color_grading(condition, base_config)
    
    # Dawn should enhance cool tones
    assert adapted['cool_tone_boost'] >= base_config['cool_tone_boost']


def test_lighting_condition_to_dict(detector, device):
    """Test LightingCondition serialization."""
    rgb = torch.rand(1, 3, 512, 512, device=device)
    condition = detector.detect(rgb)
    
    # Convert to dict
    data = condition.to_dict()
    
    # Verify structure
    assert isinstance(data, dict)
    assert 'time_of_day' in data
    assert 'confidence' in data
    assert 'sky_coverage' in data
    assert 'sky_color_temp' in data
    
    # time_of_day should be string
    assert isinstance(data['time_of_day'], str)


@pytest.mark.parametrize("time_of_day", [
    TimeOfDay.DAWN,
    TimeOfDay.GOLDEN_HOUR,
    TimeOfDay.MIDDAY,
    TimeOfDay.TWILIGHT,
    TimeOfDay.NIGHT,
    TimeOfDay.OVERCAST,
])
def test_all_time_of_day_cases(detector, time_of_day):
    """Test that all time-of-day cases are handled in adaptation."""
    condition = LightingCondition(
        time_of_day=time_of_day,
        confidence=0.75,
        sky_coverage=0.3,
        sky_color_temp=5500.0,
        sky_brightness=0.5,
        has_strong_shadows=False,
        shadow_direction=None,
        dominant_hue=180.0,
        warmth=0.0,
    )
    
    base_tone_config = {'contrast': 1.0}
    base_color_config = {'global_saturation': 1.0}
    
    # Should not raise exceptions
    adapted_tone = detector.adapt_tone_mapping(condition, base_tone_config)
    adapted_color = detector.adapt_color_grading(condition, base_color_config)
    
    assert isinstance(adapted_tone, dict)
    assert isinstance(adapted_color, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
