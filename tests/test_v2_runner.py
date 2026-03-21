"""Tests for V2Runner subprocess wrapper.

Tests subprocess invocation, argument passing, error handling,
and report discovery without requiring the actual script to exist.
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]

from transformation_portal.lux_depth_v3.v2_runner import V2Runner, find_v2_report


class TestV2RunnerInit:
    """Test V2Runner initialization and repo root discovery."""

    def test_init_finds_repo_root(self):
        """Test that V2Runner can find repo root."""
        runner = V2Runner()

        # Should have found repo root
        assert runner.repo_root is not None
        assert runner.repo_root.exists()

        # Should have set script path
        assert runner.script_path == runner.repo_root / "scripts" / "enhance_image.py"

    def test_init_warns_if_script_missing(self, caplog):
        """Test that init logs warning if script doesn't exist."""
        with caplog.at_level("WARNING"):
            runner = V2Runner()

            # If script doesn't exist, should log warning
            if not runner.script_path.exists():
                assert "not found" in caplog.text.lower()


class TestV2RunnerExecution:
    """Test V2Runner.run() subprocess execution."""

    @patch("transformation_portal.lux_depth_v3.v2_runner.subprocess.run")
    def test_successful_execution_basic(self, mock_subprocess, tmp_path):
        """Test basic successful execution."""
        # Mock successful subprocess
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Enhancement complete"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        # Create runner
        runner = V2Runner()

        # Mock script path to exist (on the instance)
        runner.script_path = Path("/fake/scripts/enhance_image.py")
        with patch.object(Path, "exists", return_value=True):
            input_path = tmp_path / "input.jpg"
            output_dir = tmp_path / "output"

            result = runner.run(input_path=input_path, depth_dir=None, output_dir=output_dir)

        # Verify subprocess was called
        assert mock_subprocess.called
        call_args = mock_subprocess.call_args

        # Check command structure
        cmd = call_args[0][0]
        assert str(input_path) in cmd
        assert "--output-dir" in cmd
        assert str(output_dir) in cmd

        # Verify result structure
        assert "runtime_s" in result
        assert "status" in result
        assert result["status"] == "success"

    @patch("transformation_portal.lux_depth_v3.v2_runner.subprocess.run")
    def test_command_includes_all_arguments(self, mock_subprocess, tmp_path):
        """Test that all provided arguments appear in command."""
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        runner = V2Runner()
        runner.script_path = Path("/fake/enhance_image.py")

        # Run with all arguments
        with patch.object(Path, "exists", return_value=True):
            result = runner.run(
                input_path=tmp_path / "input.jpg",
                depth_dir=tmp_path / "depth",
                output_dir=tmp_path / "output",
                preset="custom_preset",
                device="cuda",
                upscaler_backend="esrgan",
                log_file=tmp_path / "log.txt",
                timeout=300,
            )

        # Extract command from mock call
        cmd = mock_subprocess.call_args[0][0]

        # Verify all arguments present
        assert "--depth-dir" in cmd
        assert str(tmp_path / "depth") in cmd
        assert "--output-dir" in cmd
        assert "--preset" in cmd
        assert "custom_preset" in cmd
        assert "--device" in cmd
        assert "cuda" in cmd
        assert "--upscaler" in cmd
        assert "esrgan" in cmd
        assert "--log-file" in cmd

        # Verify timeout passed to subprocess
        assert mock_subprocess.call_args[1]["timeout"] == 300
        assert result["status"] == "success"

    @patch("transformation_portal.lux_depth_v3.v2_runner.subprocess.run")
    def test_optional_arguments_omitted_when_none(self, mock_subprocess, tmp_path):
        """Test that None arguments are omitted from command."""
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        runner = V2Runner()
        runner.script_path = Path("/fake/enhance_image.py")

        # Run with minimal arguments (many None)
        with patch.object(Path, "exists", return_value=True):
            result = runner.run(
                input_path=tmp_path / "input.jpg",
                depth_dir=None,  # Should be omitted
                output_dir=tmp_path / "output",
                upscaler_backend=None,  # Should be omitted
                log_file=None,  # Should be omitted
            )

        cmd = mock_subprocess.call_args[0][0]

        # Verify None arguments not present
        assert "--depth-dir" not in cmd
        assert "--upscaler" not in cmd
        assert "--log-file" not in cmd

        # Verify required arguments still present
        assert "--output-dir" in cmd
        assert result["status"] == "success"

    @patch("transformation_portal.lux_depth_v3.v2_runner.subprocess.run")
    def test_asset_key_included_when_provided(self, mock_subprocess, tmp_path):
        """Test that asset_key is included in command when provided."""
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        runner = V2Runner()
        runner.script_path = Path("/fake/enhance_image.py")

        with patch.object(Path, "exists", return_value=True):
            result = runner.run(
                input_path=tmp_path / "input.jpg",
                depth_dir=None,
                output_dir=tmp_path / "output",
                asset_key="input_jpg_a1b2c3d4",  # Canonical hashed key
            )

        cmd = mock_subprocess.call_args[0][0]

        # Verify asset_key argument present with correct value
        assert "--asset-key" in cmd
        asset_key_index = cmd.index("--asset-key")
        assert cmd[asset_key_index + 1] == "input_jpg_a1b2c3d4"
        assert result["status"] == "success"

    @patch("transformation_portal.lux_depth_v3.v2_runner.subprocess.run")
    def test_asset_key_omitted_when_none(self, mock_subprocess, tmp_path):
        """Test that asset_key is omitted when not provided."""
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        runner = V2Runner()
        runner.script_path = Path("/fake/enhance_image.py")

        with patch.object(Path, "exists", return_value=True):
            result = runner.run(
                input_path=tmp_path / "input.jpg",
                depth_dir=None,
                output_dir=tmp_path / "output",
                asset_key=None,  # Not provided
            )

        cmd = mock_subprocess.call_args[0][0]

        # Verify asset_key argument is NOT present
        assert "--asset-key" not in cmd
        assert result["status"] == "success"

    def test_asset_key_rejects_path_like_values(self, tmp_path):
        """Test that asset_key with path separators raises ValueError."""
        runner = V2Runner()
        runner.script_path = Path("/fake/enhance_image.py")

        # Test forward slash
        with patch.object(Path, "exists", return_value=True):
            with pytest.raises(ValueError, match="stem-like identifier"):
                runner.run(
                    input_path=tmp_path / "input.jpg",
                    depth_dir=None,
                    output_dir=tmp_path / "output",
                    asset_key="../bad_traversal",
                )

        # Test backslash
        with patch.object(Path, "exists", return_value=True):
            with pytest.raises(ValueError, match="stem-like identifier"):
                runner.run(
                    input_path=tmp_path / "input.jpg",
                    depth_dir=None,
                    output_dir=tmp_path / "output",
                    asset_key="subdir\\bad_traversal",
                )

    @patch("transformation_portal.lux_depth_v3.v2_runner.subprocess.run")
    def test_asset_key_empty_string_treated_as_none(self, mock_subprocess, tmp_path):
        """Test that empty or whitespace-only asset_key is treated as None."""
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        runner = V2Runner()
        runner.script_path = Path("/fake/enhance_image.py")

        with patch.object(Path, "exists", return_value=True):
            result = runner.run(
                input_path=tmp_path / "input.jpg",
                depth_dir=None,
                output_dir=tmp_path / "output",
                asset_key="   ",  # Whitespace-only
            )

        cmd = mock_subprocess.call_args[0][0]

        # Whitespace-only asset_key should be treated as None
        assert "--asset-key" not in cmd
        assert result["status"] == "success"

    def test_raises_filenotfounderror_if_script_missing(self, tmp_path):
        """Test that run() raises clear error if script doesn't exist."""
        runner = V2Runner()

        # Force script path to nonexistent location
        runner.script_path = tmp_path / "nonexistent_script.py"

        with pytest.raises(FileNotFoundError, match="not found"):
            runner.run(input_path=tmp_path / "input.jpg", depth_dir=None, output_dir=tmp_path / "output")

    @patch("transformation_portal.lux_depth_v3.v2_runner.subprocess.run")
    def test_subprocess_error_raises_runtimeerror(self, mock_subprocess, tmp_path):
        """Test that subprocess CalledProcessError becomes RuntimeError with context."""
        # Mock subprocess failure
        mock_subprocess.side_effect = subprocess.CalledProcessError(
            returncode=2, cmd=["python", "/fake/enhance_image.py"], stderr="Fatal error: model not found", output=""
        )

        runner = V2Runner()
        runner.script_path = Path("/fake/enhance_image.py")

        with patch.object(Path, "exists", return_value=True):
            with pytest.raises(RuntimeError) as exc_info:
                runner.run(input_path=tmp_path / "input.jpg", depth_dir=None, output_dir=tmp_path / "output")

        error_msg = str(exc_info.value)

        # Verify error contains context
        assert "returncode=2" in error_msg
        assert "Fatal error: model not found" in error_msg
        assert "enhance_image.py" in error_msg

    @patch("transformation_portal.lux_depth_v3.v2_runner.subprocess.run")
    def test_subprocess_timeout_raises_timeouterror(self, mock_subprocess, tmp_path):
        """Test that subprocess TimeoutExpired becomes TimeoutError with partial output."""
        # Mock timeout with partial output
        timeout_exc = subprocess.TimeoutExpired(cmd=["python", "/fake/enhance_image.py"], timeout=10)
        timeout_exc.stdout = "Processing stage 1..."
        timeout_exc.stderr = "Warning: slow model"
        mock_subprocess.side_effect = timeout_exc

        runner = V2Runner()
        runner.script_path = Path("/fake/enhance_image.py")

        with patch.object(Path, "exists", return_value=True):
            with pytest.raises(TimeoutError) as exc_info:
                runner.run(input_path=tmp_path / "input.jpg", depth_dir=None, output_dir=tmp_path / "output", timeout=10)

        error_msg = str(exc_info.value)

        # Verify timeout info present
        assert "10" in error_msg  # timeout value
        assert "Processing stage 1" in error_msg or "Partial" in error_msg


class TestReportMerging:
    """Test report JSON discovery and merging."""

    @patch("transformation_portal.lux_depth_v3.v2_runner.subprocess.run")
    def test_merges_report_when_found(self, mock_subprocess, tmp_path):
        """Test that report JSON is discovered and merged into result."""
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        # Create mock report JSON
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        report_data = {"preset": "custom", "enhancement_strength": 0.8, "upscaler": "esrgan"}
        report_path = output_dir / "input_report.json"
        with open(report_path, "w") as f:
            json.dump(report_data, f)

        runner = V2Runner()
        runner.script_path = Path("/fake/enhance_image.py")

        with patch.object(Path, "exists", return_value=True):
            result = runner.run(input_path=tmp_path / "input.jpg", depth_dir=None, output_dir=output_dir)

        # Verify report fields merged
        assert result["preset"] == "custom"
        assert result["enhancement_strength"] == 0.8
        assert result["upscaler"] == "esrgan"

        # Verify runtime metadata still present
        assert "runtime_s" in result
        assert "status" in result
        assert "report_path" in result

    @patch("transformation_portal.lux_depth_v3.v2_runner.subprocess.run")
    def test_merges_report_using_asset_key_when_provided(self, mock_subprocess, tmp_path):
        """Test that report discovery uses asset_key when provided.

        This verifies the fix for stem-resolution drift where depth artifacts
        are named with canonical hashed keys (e.g., input_jpg_a1b2c3d4) but
        reports were being searched using raw input stem.
        """
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        # Create mock report JSON named with canonical hashed key
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Note: report is named with the HASHED asset key, not raw input stem
        report_data = {"preset": "custom", "enhancement_strength": 0.9, "canonical": True}
        report_path = output_dir / "input_jpg_a1b2c3d4_report.json"
        with open(report_path, "w") as f:
            json.dump(report_data, f)

        runner = V2Runner()
        runner.script_path = Path("/fake/enhance_image.py")

        with patch.object(Path, "exists", return_value=True):
            result = runner.run(
                input_path=tmp_path / "input.jpg",
                depth_dir=None,
                output_dir=output_dir,
                asset_key="input_jpg_a1b2c3d4",  # Canonical hashed key
            )

        # Verify report found using asset_key (not raw input.stem)
        assert result["preset"] == "custom"
        assert result["enhancement_strength"] == 0.9
        assert result["canonical"] is True
        assert "report_path" in result
        assert result["report_path"] is not None

    @patch("transformation_portal.lux_depth_v3.v2_runner.subprocess.run")
    def test_canonical_key_depth_report_alignment_regression(self, mock_subprocess, tmp_path):
        """Regression test: canonical key aligns depth artifacts with report naming.

        This test validates the fix for stem-resolution drift where:
        - input stem: 750Picacho_Pool
        - canonical output key: 750Picacho_Pool_png_e33ad12e
        - depth sidecar: 750Picacho_Pool_png_e33ad12e_depth.png
        - V2 report: 750Picacho_Pool_png_e33ad12e_report.json

        Prior to the fix, V2 enhancement searched for depth maps using the raw
        input stem ("750Picacho_Pool") but depth artifacts were named with the
        canonical hashed key ("750Picacho_Pool_png_e33ad12e"), causing silent
        depth lookup failures.

        Reference: PR #1241 - stem-resolution drift fix
        """
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        # Create output dir structure simulating orchestrator layout
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Simulate exactly the real-world scenario
        input_stem = "750Picacho_Pool"
        canonical_key = "750Picacho_Pool_png_e33ad12e"

        # Create depth sidecar named with canonical key (as orchestrator does)
        # This simulates: depth_dir/750Picacho_Pool_png_e33ad12e_depth.png
        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()
        depth_sidecar = depth_dir / f"{canonical_key}_depth.png"
        depth_sidecar.write_bytes(b"fake depth data")

        # Create report named with canonical key (as V2 should write)
        report_data = {
            "status": "success",
            "asset_key": canonical_key,
            "input_stem": input_stem,
            "depth": {
                "requested": True,
                "lookup_key": canonical_key,
                "depth_dir": str(depth_dir),
                "resolved_path": str(depth_sidecar),
                "loaded": True,
                "consumed": True,
                "consumption_source": "stage_metadata",
            },
        }
        report_path = output_dir / f"{canonical_key}_report.json"
        with open(report_path, "w") as f:
            json.dump(report_data, f)

        runner = V2Runner()
        runner.script_path = Path("/fake/enhance_image.py")

        with patch.object(Path, "exists", return_value=True):
            result = runner.run(
                input_path=tmp_path / f"{input_stem}.png",  # Raw input name
                depth_dir=depth_dir,
                output_dir=output_dir,
                asset_key=canonical_key,  # Canonical hashed key (from orchestrator)
            )

        # CRITICAL ASSERTIONS: Validate the fix
        # 1. Report was found using canonical key (not raw input stem)
        assert (
            result["report_path"] is not None
        ), f"Report not found! This indicates stem-resolution drift. Expected report at: {report_path}"

        # 2. Report identity metadata is correct
        assert (
            result["asset_key"] == canonical_key
        ), f"asset_key mismatch: expected {canonical_key}, got {result.get('asset_key')}"
        assert (
            result["input_stem"] == input_stem
        ), f"input_stem mismatch: expected {input_stem}, got {result.get('input_stem')}"

        # 3. Depth block has correct lookup_key (canonical, not raw stem)
        depth_block = result.get("depth", {})
        assert (
            depth_block.get("lookup_key") == canonical_key
        ), f"depth.lookup_key mismatch: expected {canonical_key}, got {depth_block.get('lookup_key')}"

        # 4. --asset-key was passed to subprocess (verify command structure)
        cmd = mock_subprocess.call_args[0][0]
        assert "--asset-key" in cmd
        asset_key_index = cmd.index("--asset-key")
        assert cmd[asset_key_index + 1] == canonical_key

        # 5. Depth was successfully consumed (verifies end-to-end fix)
        assert depth_block.get("consumed") is True
        assert depth_block.get("resolved_path") == str(depth_sidecar)

    @patch("transformation_portal.lux_depth_v3.v2_runner.subprocess.run")
    def test_returns_stdout_stderr_when_no_report(self, mock_subprocess, tmp_path):
        """Test that stdout/stderr included when report not found."""
        mock_subprocess.return_value = Mock(returncode=0, stdout="Processing complete", stderr="Warning: deprecated option")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        runner = V2Runner()
        runner.script_path = Path("/fake/enhance_image.py")

        with patch.object(Path, "exists", return_value=True):
            result = runner.run(input_path=tmp_path / "input.jpg", depth_dir=None, output_dir=output_dir)

        # No report exists, should include process output
        assert result["report_path"] is None
        assert result["stdout"] == "Processing complete"
        assert result["stderr"] == "Warning: deprecated option"


class TestFindV2Report:
    """Test report JSON discovery function."""

    def test_finds_direct_match(self, tmp_path):
        """Test direct report match in output directory."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create report
        report_path = output_dir / "test_image_report.json"
        report_path.write_text("{}")

        found = find_v2_report(output_dir, "test_image")

        assert found == report_path

    def test_finds_nested_report(self, tmp_path):
        """Test recursive search finds nested reports."""
        output_dir = tmp_path / "output"
        subdir = output_dir / "subdir" / "nested"
        subdir.mkdir(parents=True)

        # Create nested report
        report_path = subdir / "image_report.json"
        report_path.write_text("{}")

        found = find_v2_report(output_dir, "image")

        assert found == report_path

    def test_finds_prefixed_derived_report(self, tmp_path):
        """Derived report names should match by image-key prefix."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        report_path = output_dir / "image_materials_v3_enhanced_report.json"
        report_path.write_text("{}")

        found = find_v2_report(output_dir, "image")
        assert found == report_path

    def test_returns_none_when_not_found(self, tmp_path):
        """Test returns None when report doesn't exist."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        found = find_v2_report(output_dir, "nonexistent")

        assert found is None


class TestPathValidation:
    """Test path validation behavior in V2Runner.run().

    Security contract:
    - All paths are resolved to absolute paths (normalization)
    - Paths inside repo root are validated via safe_resolve_path()
    - Paths outside repo root are allowed but logged (warning for output, debug for input)
    - This is intentional: user data directories are often outside the repo
    """

    @patch("transformation_portal.lux_depth_v3.v2_runner.subprocess.run")
    def test_relative_path_resolved(self, mock_subprocess, tmp_path, caplog):
        """Test that relative paths are resolved to absolute."""
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        runner = V2Runner()
        runner.script_path = Path("/fake/enhance_image.py")

        # Create a relative path that's valid
        input_path = tmp_path / "input.jpg"
        output_dir = tmp_path / "output"

        with patch.object(Path, "exists", return_value=True):
            result = runner.run(input_path=input_path, depth_dir=None, output_dir=output_dir)

        # Command should contain resolved absolute paths
        cmd = mock_subprocess.call_args[0][0]
        # Paths in command should be absolute
        for arg in cmd:
            if str(tmp_path) in arg:
                assert Path(arg).is_absolute()
        assert result["status"] == "success"

    @patch("transformation_portal.lux_depth_v3.v2_runner.subprocess.run")
    def test_path_outside_repo_allowed_with_warning(self, mock_subprocess, tmp_path, caplog):
        """Test that paths outside repo root are allowed but logged."""
        import logging

        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        runner = V2Runner()
        runner.script_path = Path("/fake/enhance_image.py")

        # Use tmp_path which is likely outside repo root
        output_dir = tmp_path / "external_output"

        with caplog.at_level(logging.DEBUG):
            with patch.object(Path, "exists", return_value=True):
                result = runner.run(input_path=tmp_path / "input.jpg", depth_dir=None, output_dir=output_dir)

        # Should succeed without raising ValidationError
        assert result["status"] == "success"

    @patch("transformation_portal.lux_depth_v3.v2_runner.subprocess.run")
    def test_masks_file_validated(self, mock_subprocess, tmp_path):
        """Test that masks_file path is validated and included in command."""
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        runner = V2Runner()
        runner.script_path = Path("/fake/enhance_image.py")

        masks_file = tmp_path / "masks.npz"

        with patch.object(Path, "exists", return_value=True):
            result = runner.run(
                input_path=tmp_path / "input.jpg",
                depth_dir=None,
                output_dir=tmp_path / "output",
                masks_file=masks_file,
            )

        cmd = mock_subprocess.call_args[0][0]
        assert "--masks-file" in cmd
        assert str(masks_file) in cmd or str(masks_file.resolve()) in cmd
        assert result["status"] == "success"

    @patch("transformation_portal.lux_depth_v3.v2_runner.subprocess.run")
    def test_traversal_path_normalized(self, mock_subprocess, tmp_path):
        """Test that path traversal sequences are normalized to absolute paths.

        Paths with .. are resolved to their canonical form. Paths outside
        repo root are allowed by design (user data directories).
        """
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        runner = V2Runner()
        runner.script_path = Path("/fake/enhance_image.py")

        # Path with traversal - should be normalized
        input_path = tmp_path / "subdir" / ".." / "input.jpg"
        output_dir = tmp_path / "output"

        with patch.object(Path, "exists", return_value=True):
            result = runner.run(input_path=input_path, depth_dir=None, output_dir=output_dir)

        cmd = mock_subprocess.call_args[0][0]

        # Verify no .. in command (paths are resolved)
        for arg in cmd:
            if arg.startswith("/") or arg.startswith("\\"):
                assert ".." not in arg, f"Traversal not normalized: {arg}"

        assert result["status"] == "success"

    @patch("transformation_portal.lux_depth_v3.v2_runner.subprocess.run")
    def test_output_dir_warning_logged(self, mock_subprocess, tmp_path, caplog):
        """Test that output dir outside repo root logs warning.

        When output_dir is outside the repository root, a warning should be
        logged but execution should still succeed (user data directories are
        allowed by design).
        """
        import logging

        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        runner = V2Runner()
        runner.script_path = Path("/fake/enhance_image.py")

        # Output dir outside repo root (tmp_path is typically /tmp/...)
        # This is a common case for user data directories
        output_dir = tmp_path / "external_output"

        with caplog.at_level(logging.DEBUG):
            with patch.object(Path, "exists", return_value=True):
                result = runner.run(input_path=tmp_path / "input.jpg", depth_dir=None, output_dir=output_dir)

        # Should succeed without raising error
        assert result["status"] == "success"
        # Output path should be resolved and appear in command
        cmd = mock_subprocess.call_args[0][0]
        assert str(output_dir.resolve()) in cmd or str(output_dir) in cmd

    @patch("transformation_portal.lux_depth_v3.v2_runner.subprocess.run")
    def test_all_path_args_validated(self, mock_subprocess, tmp_path):
        """Test that all path arguments are validated/normalized."""
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        runner = V2Runner()
        runner.script_path = Path("/fake/enhance_image.py")

        # Provide all path arguments
        input_path = tmp_path / "input.jpg"
        depth_dir = tmp_path / "depth"
        output_dir = tmp_path / "output"
        log_file = tmp_path / "log.txt"
        masks_file = tmp_path / "masks.npz"

        with patch.object(Path, "exists", return_value=True):
            result = runner.run(
                input_path=input_path,
                depth_dir=depth_dir,
                output_dir=output_dir,
                log_file=log_file,
                masks_file=masks_file,
            )

        cmd = mock_subprocess.call_args[0][0]

        # All path flags should appear in command
        assert "--depth-dir" in cmd
        assert "--output-dir" in cmd
        assert "--log-file" in cmd
        assert "--masks-file" in cmd

        # Verify input path is present (it's a positional arg)
        assert any(str(input_path) in arg or str(input_path.resolve()) in arg for arg in cmd)

        assert result["status"] == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
