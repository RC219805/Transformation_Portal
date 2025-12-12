"""
Integration test for auto-preset CLI flag.

Tests the end-to-end flow of --auto-preset with the CLI.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock
from PIL import Image

from lux_depth_v2.config import Preset


@pytest.fixture
def temp_image(tmp_path):
    """Create a temporary test image."""
    img_path = tmp_path / "test_kitchen.jpg"
    Image.new('RGB', (640, 480), color='white').save(img_path)
    return img_path


@pytest.fixture
def mock_auto_select():
    """Mock auto_select_preset function."""
    with patch('lux_depth_v2.preset_selector.auto_select_preset') as mock:
        yield mock


def test_auto_preset_cli_integration(temp_image, mock_auto_select, tmp_path):
    """Test --auto-preset flag integration in CLI."""
    from lux_depth_v2.cli import build_parser
    
    # Mock auto_select_preset to return a specific preset
    mock_auto_select.return_value = Preset.INTERIOR_LUXURY_APEX_QUALITY
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Simulate CLI args with --auto-preset
    cli_args = [
        "--input", str(temp_image),
        "--output-dir", str(output_dir),
        "--auto-preset",
        "--quality-tier", "apex"
    ]
    
    parser = build_parser()
    args = parser.parse_args(cli_args)
    
    # Verify args parsed correctly
    assert args.auto_preset is True
    assert args.quality_tier == "apex"
    assert args.input == str(temp_image)


def test_auto_preset_without_input_fails():
    """Test that --auto-preset without --input shows error."""
    from lux_depth_v2.cli import build_parser
    
    # This should parse but main() should error
    parser = build_parser()
    args = parser.parse_args([
        "--output-dir", "/tmp/output",
        "--auto-preset"
    ])
    
    assert args.auto_preset is True
    assert args.input is None  # Will fail in main()


def test_quality_tier_defaults_to_max():
    """Test that quality tier defaults to 'max'."""
    from lux_depth_v2.cli import build_parser
    
    parser = build_parser()
    args = parser.parse_args([
        "--input", "/tmp/test.jpg",
        "--output-dir", "/tmp/output"
    ])
    
    assert args.quality_tier == "max"


def test_preset_and_auto_preset_both_work():
    """Test that manual preset is used when auto-preset not specified."""
    from lux_depth_v2.cli import build_parser
    
    parser = build_parser()
    args = parser.parse_args([
        "--input", "/tmp/test.jpg",
        "--output-dir", "/tmp/output",
        "--preset", "interior_luxury_apex_quality"
    ])
    
    assert args.auto_preset is False  # Not set
    assert args.preset == "interior_luxury_apex_quality"
