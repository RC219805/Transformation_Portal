"""Tests for lux_depth_v3 CLI module.

Verifies argument parsing, validation, and non-commercial license checks.
"""
import pytest
from typer.testing import CliRunner
from transformation_portal.lux_depth_v3.__main__ import app, _parse_bool_flag


runner = CliRunner()


class TestBoolFlagParsing:
    """Test boolean flag parsing."""

    @pytest.mark.parametrize("value,expected", [
        ("on", True),
        ("ON", True),
        ("true", True),
        ("TRUE", True),
        ("True", True),
        ("yes", True),
        ("YES", True),
        ("1", True),
        ("off", False),
        ("OFF", False),
        ("false", False),
        ("FALSE", False),
        ("False", False),
        ("no", False),
        ("NO", False),
        ("0", False),
        ("  on  ", True),
        ("  off  ", False),
    ])
    def test_parse_bool_flag(self, value, expected):
        """Test that _parse_bool_flag correctly parses various boolean string formats."""
        assert _parse_bool_flag(value) == expected


class TestCLIValidation:
    """Test CLI validation logic."""

    def test_missing_input_dir(self):
        """Test that missing input directory is rejected."""
        result = runner.invoke(app, [
            "--input-dir", "/nonexistent/path",
            "--output-dir", "/tmp/output",
        ])
        assert result.exit_code == 1
        assert "does not exist" in result.stdout.lower() or "does not exist" in str(result.exception).lower()

    def test_depth_pro_requires_non_commercial(self, tmp_path):
        """Test that depth_pro backend requires --non-commercial-ok true."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        result = runner.invoke(app, [
            "--input-dir", str(input_dir),
            "--output-dir", str(tmp_path / "output"),
            "--depth-backend", "depth_pro",
            "--non-commercial-ok", "false",
        ])
        assert result.exit_code == 1
        assert "non-commercial" in result.stdout.lower()

    def test_depth_pro_requires_apple_license(self, tmp_path):
        """Test that depth_pro backend requires --accept-apple-depth-pro-research-license true."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        result = runner.invoke(app, [
            "--input-dir", str(input_dir),
            "--output-dir", str(tmp_path / "output"),
            "--depth-backend", "depth_pro",
            "--non-commercial-ok", "true",
            "--accept-apple-depth-pro-research-license", "false",
        ])
        assert result.exit_code == 1
        assert "apple" in result.stdout.lower() or "depth pro" in result.stdout.lower()

    def test_v31_preset_requires_non_commercial(self, tmp_path):
        """Test that v3.1 preset requires --non-commercial-ok true."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        result = runner.invoke(app, [
            "--input-dir", str(input_dir),
            "--output-dir", str(tmp_path / "output"),
            "--preset", "depth-anything-v3.1-research-m4",
            "--non-commercial-ok", "false",
        ])
        assert result.exit_code == 1
        assert "non-commercial" in result.stdout.lower()


class TestCLIConfiguration:
    """Test CLI configuration building."""

    def test_apex_commercial_config(self, tmp_path):
        """Test APEX commercial-safe configuration."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        # Create a dummy image so the directory isn't empty
        (input_dir / "test.jpg").touch()

        # This should not raise validation errors (but will fail due to missing image data)
        result = runner.invoke(app, [
            "--input-dir", str(input_dir),
            "--output-dir", str(tmp_path / "output"),
            "--preset", "premium",
            "--quality-tier", "apex",
            "--depth-backend", "depth_anything_v3",
            "--materials-v3", "on",
            "--pbr", "on",
            "--cache-depth", "on",
            "--emit-master16", "on",
            "--emit-upscaled16", "on",
            "--emit-marketing", "on",
            "--emit-report", "on",
            "--emit-run-card", "on",
        ])
        # Should pass validation (will fail later due to missing dependencies, but that's expected)
        # Exit code 1 is acceptable here as long as it's not a validation error
        # The important thing is no "non-commercial" or "license" errors
        if result.exit_code == 1:
            assert "non-commercial" not in result.stdout.lower()
            assert "license" not in result.stdout.lower()

    def test_apex_research_depth_pro_config(self, tmp_path):
        """Test APEX+ research configuration with Depth Pro."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.jpg").touch()

        # This should not raise validation errors (with proper license flags)
        result = runner.invoke(app, [
            "--input-dir", str(input_dir),
            "--output-dir", str(tmp_path / "output"),
            "--preset", "premium",
            "--quality-tier", "apex",
            "--depth-backend", "depth_pro",
            "--non-commercial-ok", "true",
            "--accept-apple-depth-pro-research-license", "true",
            "--depth-device", "mps",
            "--materials-v3", "on",
            "--pbr", "on",
        ])
        # Should pass validation
        if result.exit_code == 1:
            # Should not be a license validation error
            assert "requires --non-commercial-ok" not in result.stdout
            assert "requires --accept-apple-depth-pro-research-license" not in result.stdout


class TestCLIHelp:
    """Test CLI help output."""

    def test_help_output(self):
        """Test that --help produces expected output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "lux-depth-v3" in result.stdout.lower() or "lux depth v3" in result.stdout.lower()
        assert "apex" in result.stdout.lower()
        assert "quality" in result.stdout.lower() and "tier" in result.stdout.lower()
        assert "materials" in result.stdout.lower()
        assert "non-commercial" in result.stdout.lower() or "non commercial" in result.stdout.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
