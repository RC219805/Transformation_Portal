"""Tests for lux_depth_v3 CLI module.

Verifies argument parsing, validation, and non-commercial license checks.
"""

import re

import pytest
from typer.testing import CliRunner

from transformation_portal.lux_depth_v3.__main__ import _parse_bool_flag, app

runner = CliRunner()


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    ansi_escape = re.compile(r"\x1b\[[0-9;]*[mGKHf]")
    return ansi_escape.sub("", text)


class TestBoolFlagParsing:
    """Test boolean flag parsing."""

    @pytest.mark.parametrize(
        "value,expected",
        [
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
        ],
    )
    def test_parse_bool_flag(self, value, expected):
        """Test that _parse_bool_flag correctly parses various boolean string formats."""
        assert _parse_bool_flag(value) == expected


class TestCLIValidation:
    """Test CLI validation logic."""

    def test_missing_input_dir(self):
        """Test that missing input directory is rejected."""
        result = runner.invoke(
            app,
            [
                "--input-dir",
                "/nonexistent/path",
                "--output-dir",
                "/tmp/output",
            ],
        )
        assert result.exit_code == 1
        assert "does not exist" in result.stdout.lower() or "does not exist" in str(result.exception).lower()

    def test_depth_pro_requires_non_commercial(self, tmp_path):
        """Test that depth_pro backend requires --non-commercial-ok true."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "output"),
                "--depth-backend",
                "depth_pro",
                "--non-commercial-ok",
                "false",
            ],
        )
        assert result.exit_code == 1
        assert "non-commercial" in result.stdout.lower()

    def test_depth_pro_requires_apple_license(self, tmp_path):
        """Test that depth_pro backend requires --accept-apple-depth-pro-research-license true."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "output"),
                "--depth-backend",
                "depth_pro",
                "--non-commercial-ok",
                "true",
                "--accept-apple-depth-pro-research-license",
                "false",
            ],
        )
        assert result.exit_code == 1
        assert "apple" in result.stdout.lower() or "depth pro" in result.stdout.lower()

    def test_v31_preset_requires_non_commercial(self, tmp_path):
        """Test that v3.1 preset requires --non-commercial-ok true."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "output"),
                "--preset",
                "depth-anything-v3.1-research-m4",
                "--non-commercial-ok",
                "false",
            ],
        )
        assert result.exit_code == 1
        assert "non-commercial" in result.stdout.lower()

    def test_invalid_quality_tier(self, tmp_path):
        """Test that invalid quality tier is rejected."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "output"),
                "--quality-tier",
                "invalid_tier",
            ],
        )
        assert result.exit_code == 1
        assert "invalid" in result.stdout.lower() or "quality tier" in result.stdout.lower()

    def test_apex_materials_v3_requires_segmentation_enabled(self, tmp_path):
        """APEX strict gate should require explicit segmentation when Materials V3 is on."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "output"),
                "--quality-tier",
                "apex",
                "--materials-v3",
                "on",
                "--enable-segmentation",
                "off",
            ],
        )
        assert result.exit_code == 1
        assert "apex strict gate" in result.stdout.lower()
        assert "enable-segmentation" in result.stdout.lower()

    def test_apex_materials_v3_rejects_stub_segmentation_backend(self, tmp_path):
        """APEX strict gate should reject stub backend for Materials V3."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "output"),
                "--quality-tier",
                "apex",
                "--materials-v3",
                "on",
                "--enable-segmentation",
                "on",
                "--segmentation-backend",
                "stub",
            ],
        )
        assert result.exit_code == 1
        assert "apex strict gate" in result.stdout.lower()
        assert "stub segmentation backend" in result.stdout.lower()

    def test_apex_materials_v3_requires_strict_segmentation(self, tmp_path):
        """APEX strict gate should require strict segmentation mode."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "output"),
                "--quality-tier",
                "apex",
                "--materials-v3",
                "on",
                "--enable-segmentation",
                "on",
                "--segmentation-backend",
                "efficientsam",
            ],
        )
        assert result.exit_code == 1
        assert "apex strict gate" in result.stdout.lower()
        assert "strict-segmentation" in result.stdout.lower()


class TestCLIConfiguration:
    """Test CLI configuration building."""

    def test_apex_commercial_config(self, tmp_path):
        """Test APEX commercial-safe configuration."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        # Create a dummy image so the directory isn't empty
        (input_dir / "test.jpg").touch()

        # This should not raise validation errors (but will fail due to missing image data)
        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "output"),
                "--preset",
                "premium",
                "--quality-tier",
                "apex",
                "--depth-backend",
                "depth_anything_v3",
                "--materials-v3",
                "on",
                "--pbr",
                "on",
                "--cache-depth",
                "on",
                "--emit-master16",
                "on",
                "--emit-upscaled16",
                "on",
                "--emit-marketing",
                "on",
                "--emit-report",
                "on",
                "--emit-run-card",
                "on",
            ],
        )
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
        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "output"),
                "--preset",
                "premium",
                "--quality-tier",
                "apex",
                "--depth-backend",
                "depth_pro",
                "--non-commercial-ok",
                "true",
                "--accept-apple-depth-pro-research-license",
                "true",
                "--depth-device",
                "mps",
                "--materials-v3",
                "on",
                "--pbr",
                "on",
            ],
        )
        # Should pass validation
        if result.exit_code == 1:
            # Should not be a license validation error
            assert "requires --non-commercial-ok" not in result.stdout
            assert "requires --accept-apple-depth-pro-research-license" not in result.stdout

    def test_enable_v2_off_disables_v2_stage(self, tmp_path, monkeypatch):
        """Test that --enable-v2 off disables V2 enhancement stage."""
        from unittest.mock import MagicMock, patch

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.jpg").touch()

        # Mock EnhanceOrchestrator to capture config
        captured_config = None

        def mock_orch_init(config, output_root):
            nonlocal captured_config
            captured_config = config
            mock_orch = MagicMock()
            mock_orch.enhance_batch.return_value = {"total": 0, "succeeded": 0}
            return mock_orch

        with patch(
            "transformation_portal.lux_depth_v3.__main__.EnhanceOrchestrator",
            side_effect=mock_orch_init,
        ):
            result = runner.invoke(
                app,
                [
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(tmp_path / "output"),
                    "--enable-v2",
                    "off",
                ],
            )

        # Should not fail during config construction
        assert captured_config is not None
        assert captured_config.enable_v2 is False

    def test_v2_preset_none_skips_v2_stage(self, tmp_path, monkeypatch):
        """Test that --v2-preset none results in V2 skip behavior."""
        from unittest.mock import MagicMock, patch

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.jpg").touch()

        # Mock EnhanceOrchestrator to capture config
        captured_config = None

        def mock_orch_init(config, output_root):
            nonlocal captured_config
            captured_config = config
            mock_orch = MagicMock()
            mock_orch.enhance_batch.return_value = {"total": 0, "succeeded": 0}
            return mock_orch

        with patch(
            "transformation_portal.lux_depth_v3.__main__.EnhanceOrchestrator",
            side_effect=mock_orch_init,
        ):
            result = runner.invoke(
                app,
                [
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(tmp_path / "output"),
                    "--v2-preset",
                    "none",
                ],
            )

        # Should capture config with v2_preset set to None (from "none" string)
        assert captured_config is not None
        assert captured_config.v2_preset is None

    def test_save_float_depth_defaults_false(self, tmp_path):
        """save_float_depth should default to False when flag is omitted."""
        from unittest.mock import MagicMock, patch

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.jpg").touch()

        captured_config = None

        def mock_orch_init(config, output_root):
            nonlocal captured_config
            captured_config = config
            mock_orch = MagicMock()
            mock_orch.enhance_batch.return_value = {"total": 0, "succeeded": 0}
            return mock_orch

        with patch(
            "transformation_portal.lux_depth_v3.__main__.EnhanceOrchestrator",
            side_effect=mock_orch_init,
        ):
            _result = runner.invoke(
                app,
                [
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(tmp_path / "output"),
                ],
            )

        assert captured_config is not None
        assert captured_config.save_float_depth is False

    def test_save_float_depth_flag_on(self, tmp_path):
        """--save-float-depth on should set save_float_depth=True."""
        from unittest.mock import MagicMock, patch

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.jpg").touch()

        captured_config = None

        def mock_orch_init(config, output_root):
            nonlocal captured_config
            captured_config = config
            mock_orch = MagicMock()
            mock_orch.enhance_batch.return_value = {"total": 0, "succeeded": 0}
            return mock_orch

        with patch(
            "transformation_portal.lux_depth_v3.__main__.EnhanceOrchestrator",
            side_effect=mock_orch_init,
        ):
            _result = runner.invoke(
                app,
                [
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(tmp_path / "output"),
                    "--save-float-depth",
                    "on",
                ],
            )

        assert captured_config is not None
        assert captured_config.save_float_depth is True


class TestSegmentationCLI:
    """Test material segmentation CLI flags."""

    def test_invalid_segmentation_backend(self, tmp_path):
        """Test that invalid segmentation backend is rejected."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "output"),
                "--materials-v3",
                "on",
                "--enable-segmentation",
                "on",
                "--segmentation-backend",
                "invalid_backend",
            ],
        )
        assert result.exit_code == 1
        assert "invalid" in result.stdout.lower() or "segmentation backend" in result.stdout.lower()

    def test_segmentation_config_defaults(self, tmp_path):
        """Test segmentation config with default values."""
        from unittest.mock import MagicMock, patch

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.jpg").touch()

        captured_config = None

        def mock_orch_init(config, output_root):
            nonlocal captured_config
            captured_config = config
            mock_orch = MagicMock()
            mock_orch.enhance_batch.return_value = {"total": 0, "succeeded": 0}
            return mock_orch

        with patch(
            "transformation_portal.lux_depth_v3.__main__.EnhanceOrchestrator",
            side_effect=mock_orch_init,
        ):
            _result = runner.invoke(
                app,
                [
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(tmp_path / "output"),
                ],
            )

        assert captured_config is not None
        assert captured_config.enable_material_segmentation is False
        assert captured_config.material_segmentation_backend == "stub"
        assert captured_config.strict_backend is False

    def test_segmentation_config_enabled(self, tmp_path):
        """Test segmentation config when enabled via CLI."""
        from unittest.mock import MagicMock, patch

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.jpg").touch()

        captured_config = None

        def mock_orch_init(config, output_root):
            nonlocal captured_config
            captured_config = config
            mock_orch = MagicMock()
            mock_orch.enhance_batch.return_value = {"total": 0, "succeeded": 0}
            return mock_orch

        with patch(
            "transformation_portal.lux_depth_v3.__main__.EnhanceOrchestrator",
            side_effect=mock_orch_init,
        ):
            _result = runner.invoke(
                app,
                [
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(tmp_path / "output"),
                    "--materials-v3",
                    "on",
                    "--enable-segmentation",
                    "on",
                    "--segmentation-backend",
                    "efficientsam",
                    "--strict-segmentation",
                ],
            )

        assert captured_config is not None
        assert captured_config.enable_material_segmentation is True
        assert captured_config.material_segmentation_backend == "efficientsam"
        assert captured_config.strict_backend is True


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

    def test_segmentation_flags_in_help(self):
        """Test that segmentation flags appear in help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        # Strip ANSI codes to handle different terminal capabilities in CI
        output = strip_ansi(result.stdout.lower())
        assert "segmentation" in output
        assert "enable-segmentation" in output
        assert "segmentation-backend" in output
        assert "strict-segmentation" in output

    def test_save_float_depth_flag_in_help(self):
        """Test that save float depth flag appears in help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.stdout.lower())
        assert "save-float-depth" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
