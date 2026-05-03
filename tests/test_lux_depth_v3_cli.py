"""Tests for lux_depth_v3 CLI module.

Verifies argument parsing, validation, and non-commercial license checks.
"""

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from transformation_portal.lux_depth_v3.__main__ import _parse_bool_flag, app

pytestmark = pytest.mark.unit

runner = CliRunner()
DEFAULT_APACHE_MODEL_ARGS = ["--model-key", "da3-metric"]


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

    def test_invalid_grouping_mode(self, tmp_path):
        """Scene grouping mode should fail fast with supported values."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "output"),
                "--grouping-mode",
                "invalid_mode",
            ],
        )
        assert result.exit_code == 1
        assert "grouping mode" in result.stdout.lower()

    def test_invalid_raw_wb_mode_rejected_for_legacy_contract(self, tmp_path):
        """Legacy ingest contract should reject unsupported RAW white-balance modes."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "output"),
                "--raw-wb-mode",
                "auto",
            ],
        )
        assert result.exit_code == 1
        assert "raw-wb-mode" in result.stdout.lower()
        assert "legacy_linear_srgb" in result.stdout.lower()

    def test_invalid_raw_demosaic_rejected_for_legacy_contract(self, tmp_path):
        """Legacy ingest contract should reject unsupported RAW demosaic algorithms."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "output"),
                "--raw-demosaic",
                "VNG",
            ],
        )
        assert result.exit_code == 1
        assert "raw-demosaic" in result.stdout.lower()
        assert "legacy_linear_srgb" in result.stdout.lower()

    def test_invalid_raw_ingest_mode_rejected(self, tmp_path):
        """RAW ingest mode should fail fast with clear supported values."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "output"),
                "--raw-ingest-mode",
                "bad_mode",
            ],
        )
        assert result.exit_code == 1
        assert "raw-ingest-mode" in result.stdout.lower()
        assert "auto|force_rawpy|force_preview" in result.stdout

    def test_raw_inputs_fail_fast_when_rawpy_unavailable(self, monkeypatch, tmp_path):
        """RAW batches should fail before dispatch when canonical RAW support is unavailable."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "scene_01.DNG").write_bytes(b"raw-payload")

        class FakeOrchestrator:
            def __init__(self, *_args, **_kwargs):
                raise AssertionError("orchestrator should not be constructed when RAW preflight fails")

        monkeypatch.setattr("transformation_portal.lux_depth_v3.__main__.EnhanceOrchestrator", FakeOrchestrator)
        monkeypatch.setattr(
            "transformation_portal.lux_depth_v3.__main__.apply_effective_raw_runtime_config",
            lambda config: config,
        )
        monkeypatch.setattr(
            "transformation_portal.lux_depth_v3.__main__._canonical_raw_ingest_status",
            lambda *_args: (False, "rawpy is not installed"),
        )

        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "output"),
                *DEFAULT_APACHE_MODEL_ARGS,
            ],
        )

        assert result.exit_code == 1
        assert "raw inputs detected but canonical raw ingest is unavailable" in result.stdout.lower()
        assert 'install with: pip install -e ".[raw]" or pip install rawpy' in result.stdout.lower()
        assert "--raw-python" not in result.stdout

    def test_non_raw_inputs_do_not_trigger_rawpy_preflight(self, monkeypatch, tmp_path):
        """Non-RAW batches should not be blocked by the optional RAW dependency."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "scene_01.png").write_bytes(b"png-payload")

        class FakeOrchestrator:
            def __init__(self, *_args, **_kwargs):
                pass

            def enhance_batch(self, *_args, **_kwargs):
                return [{"status": "ok"}]

        monkeypatch.setattr("transformation_portal.lux_depth_v3.__main__.EnhanceOrchestrator", FakeOrchestrator)
        monkeypatch.setattr(
            "transformation_portal.lux_depth_v3.__main__._canonical_raw_ingest_status",
            lambda *_args: (False, "rawpy is not installed"),
        )

        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "output"),
                *DEFAULT_APACHE_MODEL_ARGS,
            ],
        )

        assert result.exit_code == 0
        assert "raw inputs detected" not in result.stdout.lower()

    def test_force_preview_requires_preview_escape_env_for_raw_inputs(self, monkeypatch, tmp_path):
        """force_preview should still require the explicit preview escape hatch."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "scene_01.DNG").write_bytes(b"raw-payload")

        class FakeOrchestrator:
            def __init__(self, *_args, **_kwargs):
                raise AssertionError("orchestrator should not be constructed when preview escape is missing")

        monkeypatch.setattr("transformation_portal.lux_depth_v3.__main__.EnhanceOrchestrator", FakeOrchestrator)
        monkeypatch.delenv("TP_ALLOW_RAW_PREVIEW", raising=False)

        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "output"),
                "--raw-ingest-mode",
                "force_preview",
                *DEFAULT_APACHE_MODEL_ARGS,
            ],
        )

        assert result.exit_code == 1
        assert "tp_allow_raw_preview=1" in result.stdout.lower()

    def test_raw_inputs_report_runtime_unavailability_without_claiming_rawpy_missing(self, monkeypatch, tmp_path):
        """RAW preflight should distinguish missing rawpy from other import/runtime failures."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "scene_01.DNG").write_bytes(b"raw-payload")

        class FakeOrchestrator:
            def __init__(self, *_args, **_kwargs):
                raise AssertionError("orchestrator should not be constructed when RAW preflight fails")

        monkeypatch.setattr("transformation_portal.lux_depth_v3.__main__.EnhanceOrchestrator", FakeOrchestrator)
        monkeypatch.setattr(
            "transformation_portal.lux_depth_v3.__main__.apply_effective_raw_runtime_config",
            lambda config: config,
        )
        monkeypatch.setattr(
            "transformation_portal.lux_depth_v3.__main__._canonical_raw_ingest_status",
            lambda *_args: (False, "rawpy is unavailable in this environment"),
        )

        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "output"),
                *DEFAULT_APACHE_MODEL_ARGS,
            ],
        )

        assert result.exit_code == 1
        assert "rawpy is unavailable in this environment" in result.stdout.lower()
        assert "not installed" not in result.stdout.lower()
        assert 'install with: pip install -e ".[raw]" or pip install rawpy' in result.stdout.lower()
        assert "inspect the import/runtime error in the logs" in result.stdout.lower()
        assert "working interpreter" not in result.stdout.lower()

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

    def test_reconstruction_requires_non_commercial(self, tmp_path):
        """Scene reconstruction should require --non-commercial-ok true."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "output"),
                "--enable-reconstruction",
                "on",
                "--accept-research-tools-license",
                "true",
                "--non-commercial-ok",
                "false",
            ],
        )
        assert result.exit_code == 1
        assert "non-commercial" in result.stdout.lower()

    def test_reconstruction_requires_research_tools_license(self, tmp_path):
        """Scene reconstruction should require --accept-research-tools-license true."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "output"),
                "--enable-reconstruction",
                "on",
                "--non-commercial-ok",
                "true",
                "--accept-research-tools-license",
                "false",
            ],
        )
        assert result.exit_code == 1
        assert "accept-research-tools-license" in result.stdout.lower()

    def test_cameras_sidecar_path_must_exist(self, tmp_path):
        """Missing camera sidecar path should fail validation early."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        missing_sidecar = tmp_path / "missing_scene_cameras.json"
        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "output"),
                "--cameras-sidecar-path",
                str(missing_sidecar),
            ],
        )
        assert result.exit_code == 1
        assert "camera sidecar file does not exist" in result.stdout.lower()


class TestCLIConfiguration:
    """Test CLI configuration building."""

    @pytest.mark.parametrize(
        ("name", "args", "expected_backend", "expected_device", "expected_non_commercial"),
        [
            (
                "apache_default",
                ["--model-key", "da3-metric"],
                "da3",
                "cpu",
                False,
            ),
            (
                "apple_silicon_da3",
                ["--depth-backend", "da3", "--depth-device", "mps", "--model-key", "da3-metric"],
                "da3",
                "mps",
                False,
            ),
            (
                "research_depth_pro",
                [
                    "--depth-backend",
                    "depth_pro",
                    "--depth-device",
                    "mps",
                    "--non-commercial-ok",
                    "true",
                    "--accept-apple-depth-pro-research-license",
                    "true",
                ],
                "depth_pro",
                "mps",
                True,
            ),
        ],
    )
    def test_tier_matrix_builds_expected_config(
        self,
        tmp_path,
        name,
        args,
        expected_backend,
        expected_device,
        expected_non_commercial,
    ):
        """CLI should preserve the supported Apache, Apple Silicon, and research tiers."""
        from unittest.mock import MagicMock, patch

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / f"{name}.jpg").touch()

        captured_config = None

        def mock_orch_init(config, output_root):
            nonlocal captured_config
            captured_config = config
            mock_orch = MagicMock()
            mock_orch.enhance_batch.return_value = [{"status": "ok"}]
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
                    str(tmp_path / f"output_{name}"),
                    *args,
                ],
            )

        assert result.exit_code == 0
        assert captured_config is not None
        assert (captured_config.depth_backend or "da3") == expected_backend
        assert captured_config.depth_device == expected_device
        assert captured_config.non_commercial_ok == expected_non_commercial

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
                "da3",
                "--model-key",
                "da3-metric",
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

    def test_legacy_backend_alias_normalizes_to_da3(self, tmp_path):
        """Legacy backend aliases should remain accepted but normalize in config."""
        from unittest.mock import MagicMock, patch

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.jpg").touch()

        captured_config = None

        def mock_orch_init(config, output_root):
            nonlocal captured_config
            captured_config = config
            mock_orch = MagicMock()
            mock_orch.enhance_batch.return_value = [
                {"status": "ok"},
            ]
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
                    "--depth-backend",
                    "depth_anything_v3",
                    "--model-key",
                    "da3-metric",
                ],
            )

        assert result.exit_code == 0
        assert captured_config is not None
        assert captured_config.depth_backend == "da3"

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
                    *DEFAULT_APACHE_MODEL_ARGS,
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
                    *DEFAULT_APACHE_MODEL_ARGS,
                ],
            )

        # Should capture config with v2_preset set to None (from "none" string)
        assert captured_config is not None
        assert captured_config.v2_preset is None

    def test_run_card_include_proofs_flag_sets_config(self, tmp_path):
        """--run-card-include-proofs should wire through to EnhanceConfig."""
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
            runner.invoke(
                app,
                [
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(tmp_path / "output"),
                    "--run-card-version",
                    "v2",
                    "--run-card-include-proofs",
                    "on",
                    *DEFAULT_APACHE_MODEL_ARGS,
                ],
            )

        assert captured_config is not None
        assert captured_config.run_card_include_proofs is True

    def test_vlm_captioning_flags_set_config(self, tmp_path):
        """VLM captioning flags should remain explicit and default-off."""
        from unittest.mock import MagicMock, patch

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.jpg").touch()

        captured_config = None

        def mock_orch_init(config, output_root):
            nonlocal captured_config
            captured_config = config
            mock_orch = MagicMock()
            mock_orch.enhance_batch.return_value = [{"status": "ok"}]
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
                    "--vlm-captioning",
                    "on",
                    "--vlm-captioning-backend",
                    "fastvlm",
                    "--vlm-captioning-model",
                    "review",
                    "--vlm-captioning-proxy-format",
                    "jpeg",
                    "--vlm-captioning-max-side-px",
                    "1200",
                    "--fastvlm-python",
                    "/tmp/fastvlm-python",
                    "--fastvlm-mlx-vlm-dir",
                    "/tmp/mlx-vlm",
                    "--fastvlm-timeout-seconds",
                    "60",
                    *DEFAULT_APACHE_MODEL_ARGS,
                ],
            )

        assert result.exit_code == 0
        assert captured_config is not None
        assert captured_config.vlm_captioning_enabled is True
        assert captured_config.vlm_captioning_backend == "fastvlm"
        assert captured_config.vlm_captioning_model == "review"
        assert captured_config.vlm_captioning_proxy_format == "jpeg"
        assert captured_config.vlm_captioning_max_side_px == 1200
        assert captured_config.fastvlm_python_executable == "/tmp/fastvlm-python"
        assert captured_config.fastvlm_mlx_vlm_dir == "/tmp/mlx-vlm"
        assert captured_config.fastvlm_timeout_seconds == 60

    def test_invalid_vlm_captioning_backend_rejected(self, tmp_path):
        """Only the governed subprocess FastVLM backend is supported."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "output"),
                "--vlm-captioning-backend",
                "inline",
            ],
        )

        assert result.exit_code == 1
        assert "vlm-captioning-backend" in result.stdout.lower()

    def test_depth_pro_python_flag_sets_config(self, tmp_path):
        """--depth-pro-python should be forwarded into EnhanceConfig."""
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
                    "--depth-pro-python",
                    "./.venv-depth-pro/bin/python",
                    *DEFAULT_APACHE_MODEL_ARGS,
                ],
            )

        assert captured_config is not None
        assert captured_config.depth_pro_python_executable == "./.venv-depth-pro/bin/python"

    def test_da3_python_flag_sets_config(self, tmp_path):
        """--da3-python should be forwarded into EnhanceConfig."""
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
                    "--da3-python",
                    "./.runtime/Depth-Anything-3/.venv-da3/bin/python",
                    *DEFAULT_APACHE_MODEL_ARGS,
                ],
            )

        assert captured_config is not None
        assert captured_config.da3_python_executable == "./.runtime/Depth-Anything-3/.venv-da3/bin/python"

    def test_raw_python_flag_sets_config(self, tmp_path):
        """--raw-python should be forwarded into EnhanceConfig."""
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
                    "--raw-python",
                    "./.venv-raw/bin/python",
                    *DEFAULT_APACHE_MODEL_ARGS,
                ],
            )

        assert captured_config is not None
        assert captured_config.raw_python_executable == "./.venv-raw/bin/python"

    def test_raw_preflight_uses_dedicated_raw_runtime(self, monkeypatch, tmp_path):
        """RAW preflight should validate the dedicated RAW runtime when configured."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "scene_01.DNG").write_bytes(b"raw-payload")

        captured: dict[str, str | None] = {"raw_python": None}

        class FakeOrchestrator:
            def __init__(self, *_args, **_kwargs):
                pass

            def enhance_batch(self, *_args, **_kwargs):
                return [{"status": "ok"}]

        def fake_status(raw_python_executable=None):
            captured["raw_python"] = raw_python_executable
            return True, None

        monkeypatch.setattr("transformation_portal.lux_depth_v3.__main__.EnhanceOrchestrator", FakeOrchestrator)
        monkeypatch.setattr("transformation_portal.lux_depth_v3.__main__._canonical_raw_ingest_status", fake_status)

        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "output"),
                "--raw-python",
                "./.venv-raw/bin/python",
                *DEFAULT_APACHE_MODEL_ARGS,
            ],
        )

        assert result.exit_code == 0
        assert captured["raw_python"] == "./.venv-raw/bin/python"

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
                    *DEFAULT_APACHE_MODEL_ARGS,
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
                    *DEFAULT_APACHE_MODEL_ARGS,
                ],
            )

        assert captured_config is not None
        assert captured_config.save_float_depth is True

    def test_reconstruction_flags_wire_into_config(self, tmp_path):
        """Reconstruction CLI flags should be propagated to EnhanceConfig."""
        from unittest.mock import MagicMock, patch

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.jpg").touch()

        sidecar_path = tmp_path / "scene_cameras.json"
        sidecar_path.write_text('{"schema":"tp.scene_cameras.v1","scenes":{}}', encoding="utf-8")

        captured_config = None

        def mock_orch_init(config, output_root):
            del output_root
            nonlocal captured_config
            captured_config = config
            mock_orch = MagicMock()
            mock_orch.enhance_batch.return_value = [{"status": "ok"}]
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
                    "--enable-reconstruction",
                    "on",
                    "--grouping-mode",
                    "parent_dir",
                    "--cameras-sidecar-path",
                    str(sidecar_path),
                    "--reconstruction-iterations",
                    "777",
                    "--reconstruction-tier",
                    "apex_research",
                    "--emit-scene-debug-bundle",
                    "on",
                    "--non-commercial-ok",
                    "true",
                    "--accept-research-tools-license",
                    "true",
                ],
            )

        assert result.exit_code == 0
        assert captured_config is not None
        assert captured_config.enable_reconstruction is True
        assert captured_config.grouping_mode == "parent_dir"
        assert Path(captured_config.cameras_sidecar_path) == sidecar_path
        assert captured_config.reconstruction_iterations == 777
        assert captured_config.reconstruction_tier == "apex_research"
        assert captured_config.emit_scene_debug_bundle is True
        assert captured_config.accept_research_tools_license is True


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

    def test_invalid_sam2_model_size(self, tmp_path):
        """Test that invalid --sam2-model-size is rejected."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "output"),
                "--segmentation-backend",
                "sam2",
                "--sam2-model-size",
                "tiny",
            ],
        )
        assert result.exit_code == 1
        assert "sam2-model-size" in result.stdout.lower()

    def test_invalid_sam2_model_size_is_ignored_for_non_sam2_backend(self, tmp_path):
        """Invalid SAM2 size should not block runs when backend is not SAM2."""
        from unittest.mock import MagicMock, patch

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.jpg").touch()

        captured_config = None

        def mock_orch_init(config, output_root):
            del output_root
            nonlocal captured_config
            captured_config = config
            mock_orch = MagicMock()
            mock_orch.enhance_batch.return_value = [{"status": "ok"}]
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
                    "--segmentation-backend",
                    "efficientsam",
                    "--sam2-model-size",
                    "tiny",
                    *DEFAULT_APACHE_MODEL_ARGS,
                ],
            )

        assert result.exit_code == 0
        assert captured_config is not None
        assert captured_config.material_segmentation_backend == "efficientsam"

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
                    *DEFAULT_APACHE_MODEL_ARGS,
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
                    *DEFAULT_APACHE_MODEL_ARGS,
                ],
            )

        assert captured_config is not None
        assert captured_config.enable_material_segmentation is True
        assert captured_config.material_segmentation_backend == "efficientsam"
        assert captured_config.strict_backend is True

    def test_segmentation_config_enabled_sam2(self, tmp_path):
        """Test SAM2 segmentation backend configuration via CLI."""
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
                    "sam2",
                    "--sam2-model-size",
                    "large",
                    "--sam2-checkpoint-path",
                    str(tmp_path / "sam2_hiera_large.pt"),
                    "--strict-segmentation",
                    *DEFAULT_APACHE_MODEL_ARGS,
                ],
            )

        assert captured_config is not None
        assert captured_config.enable_material_segmentation is True
        assert captured_config.material_segmentation_backend == "sam2"
        assert captured_config.sam2_model_size == "large"
        assert str(captured_config.sam2_checkpoint_path).endswith("sam2_hiera_large.pt")
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
        assert "enable-segmentation" in output or "enable-segmentati" in output
        assert "segmentation-backend" in output or "segmentation-back" in output
        assert "strict-segmentation" in output or "strict-segmentati" in output

    def test_save_float_depth_flag_in_help(self):
        """Test that save float depth flag appears in help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.stdout.lower())
        assert "save-float-depth" in output

    def test_reconstruction_flags_in_help(self):
        """Reconstruction-specific flags should appear in help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.stdout.lower())
        assert "enable-reconstruction" in output or "enable-reconstruc" in output
        assert "grouping-mode" in output
        assert "cameras-sidecar-path" in output or "cameras-sidecar-p" in output
        assert "reconstruction-iterations" in output or "reconstruction-it" in output
        assert "reconstruction-tier" in output or "reconstruction-ti" in output
        assert "emit-scene-debug-bundle" in output or "emit-scene-debug-" in output
        assert "accept-research-tools-license" in output or "accept-research-t" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
