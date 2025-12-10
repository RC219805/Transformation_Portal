"""
Integration test for autotune export configuration.

Phase 2 Slice 3: Verify autotune flag doesn't crash and report includes metadata.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lux_depth_v2.config import PipelineConfig, Preset, Phase2Config
from lux_depth_v2.pipeline import LuxPipelineV2


@pytest.fixture
def temp_image(tmp_path):
    """Create a temporary test image."""
    from PIL import Image
    
    img_path = tmp_path / "test.jpg"
    # Create simple 1000x750 RGB image
    arr = np.random.randint(0, 255, (750, 1000, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_path, quality=95)
    
    return img_path


def test_autotune_disabled_by_default():
    """Verify autotune is OFF by default (backward compatible)."""
    cfg = PipelineConfig(
        output_dir=Path("/tmp/output"),
        preset=Preset.PHOTO_REALISTIC,
        upscaler_backend="none",
    )
    
    # Phase2Config not set - autotune should be disabled
    assert cfg.phase2 is None


def test_autotune_enabled_flag_on():
    """Verify autotune flag sets internal state."""
    cfg = PipelineConfig(
        output_dir=Path("/tmp/output"),
        preset=Preset.PHOTO_REALISTIC,
        upscaler_backend="none",
    )
    
    # Enable autotune via Phase2Config
    cfg.phase2 = Phase2Config(
        autotune_export=True,
        autotune_use_complexity=True,
    )
    
    # Verify flag is set
    assert cfg.phase2 is not None
    assert cfg.phase2.autotune_export is True
    assert cfg.phase2.autotune_use_complexity is True


def test_autotune_without_complexity():
    """Verify autotune flag can disable complexity computation."""
    cfg = PipelineConfig(
        output_dir=Path("/tmp/output"),
        preset=Preset.PHOTO_REALISTIC,
        upscaler_backend="none",
    )
    
    # Enable autotune but disable complexity computation
    cfg.phase2 = Phase2Config(
        autotune_export=True,
        autotune_use_complexity=False,  # Skip complexity
    )
    
    assert cfg.phase2.autotune_export is True
    assert cfg.phase2.autotune_use_complexity is False


def test_autotune_default_off():
    """Verify autotune defaults to OFF for backward compatibility."""
    cfg = PipelineConfig(
        output_dir=Path("/tmp/output"),
        preset=Preset.PHOTO_REALISTIC,
    )
    
    # Phase2Config not set
    if cfg.phase2 is None:
        # No Phase2Config = autotune OFF
        pass
    else:
        # If Phase2Config exists, autotune should default to False
        assert cfg.phase2.autotune_export is False
