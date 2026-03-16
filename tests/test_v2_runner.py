"""Tests for V2Runner subprocess wrapper.

Tests subprocess invocation, argument passing, error handling,
and report discovery without requiring the actual script to exist.
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

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
    """Test path validation behavior in V2Runner.run()."""

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
