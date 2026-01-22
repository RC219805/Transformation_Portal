"""
Test suite for package installation integrity.

Verifies that after `pip install -e .`, all expected packages
are importable, including top-level modules like lux_depth_v2.
"""

import subprocess
import sys


def test_lux_depth_v2_import():
    """Test that lux_depth_v2 module can be imported after editable install."""
    try:
        import lux_depth_v2

        assert lux_depth_v2.__file__ is not None
        assert "lux_depth_v2" in lux_depth_v2.__file__
    except ImportError as e:
        raise AssertionError(
            f"Failed to import lux_depth_v2 after editable install: {e}\n"
            "This indicates the package is not properly configured in pyproject.toml"
        )


def test_lux_depth_v2_submodules():
    """Test that lux_depth_v2 submodules are accessible."""
    try:
        from lux_depth_v2 import config
        from lux_depth_v2 import pipeline

        assert config is not None
        assert pipeline is not None
    except ImportError as e:
        raise AssertionError(
            f"Failed to import lux_depth_v2 submodules: {e}\nSubmodules should be accessible after package installation"
        )


def test_lux_depth_v2_cli_entry_point():
    """Test that lux-depth-v2 CLI entry point is registered."""
    result = subprocess.run(
        [sys.executable, "-m", "lux_depth_v2.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, (
        f"CLI entry point failed with exit code {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "lux-depth-v2" in result.stdout.lower() or "usage" in result.stdout.lower()


def test_transformation_portal_import():
    """Test that transformation_portal from src/ layout still works."""
    try:
        import transformation_portal

        assert transformation_portal.__file__ is not None
        assert "transformation_portal" in transformation_portal.__file__
    except ImportError as e:
        raise AssertionError(f"Failed to import transformation_portal from src/ layout: {e}")


def test_both_packages_coexist():
    """Test that both src/ and top-level packages can be imported together."""
    try:
        import lux_depth_v2
        import transformation_portal

        # Verify they are different modules
        assert lux_depth_v2.__file__ != transformation_portal.__file__
        assert "lux_depth_v2" in lux_depth_v2.__file__
        assert "transformation_portal" in transformation_portal.__file__
    except ImportError as e:
        raise AssertionError(f"Failed to import both packages simultaneously: {e}")
