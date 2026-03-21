"""Tests for fix_quality_issues.py script.

Verifies that subprocess calls are executed safely without shell=True,
preventing command injection vulnerabilities (SEC-001).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch
import pytest


pytestmark = pytest.mark.unit

# Load scripts/utilities/fix_quality_issues.py without sys.path hacks
_REPO_ROOT = Path(__file__).resolve().parents[1]
_MOD_PATH = _REPO_ROOT / "scripts" / "utilities" / "fix_quality_issues.py"

_spec = importlib.util.spec_from_file_location("fix_quality_issues", _MOD_PATH)
assert _spec and _spec.loader, f"Failed to load module spec from {_MOD_PATH}"
_fix_quality_issues = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fix_quality_issues)  # type: ignore[attr-defined]

run_command = _fix_quality_issues.run_command


class TestRunCommandSecurity:
    """Test that run_command uses safe subprocess execution."""

    def test_run_command_uses_argument_list_not_shell(self):
        """Verify that run_command does NOT use shell=True (SEC-001)."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Success"
        mock_result.stderr = ""

        with patch.object(_fix_quality_issues.subprocess, "run", return_value=mock_result) as mock_subprocess:
            result = run_command("echo hello world", "Test command")

            assert mock_subprocess.called
            call_args = mock_subprocess.call_args

            # Critical: Verify first argument is a list (not a string)
            first_arg = call_args[0][0]
            assert isinstance(first_arg, list), "Command should be passed as list, not string"
            assert first_arg == ["echo", "hello", "world"], "Command should be split into arguments"

            # Critical: Verify shell=True is NOT used
            if "shell" in call_args[1]:
                assert call_args[1]["shell"] is False, "shell parameter should be False or absent"

            assert result == 0

    def test_run_command_splits_complex_command(self):
        """Test that complex commands with flags are properly split."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch.object(_fix_quality_issues.subprocess, "run", return_value=mock_result) as mock_subprocess:
            cmd = "flake8 . --count --select=E9,F63 --show-source"
            run_command(cmd, "Test flake8")

            call_args = mock_subprocess.call_args
            first_arg = call_args[0][0]
            assert isinstance(first_arg, list)
            assert first_arg[0] == "flake8"
            assert "." in first_arg
            assert "--count" in first_arg
            assert "--select=E9,F63" in first_arg
            assert "--show-source" in first_arg

    def test_run_command_handles_return_code(self):
        """Test that return codes are properly propagated."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error occurred"

        with patch.object(_fix_quality_issues.subprocess, "run", return_value=mock_result):
            result = run_command("flake8 .", "Test error handling")
            assert result == 1

    def test_run_command_uses_capture_output(self):
        """Verify capture_output and text parameters are set correctly."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "output"
        mock_result.stderr = ""

        with patch.object(_fix_quality_issues.subprocess, "run", return_value=mock_result) as mock_subprocess:
            run_command("echo test", "Test capture")

            call_args = mock_subprocess.call_args
            assert call_args[1]["capture_output"] is True
            assert call_args[1]["text"] is True
            assert call_args[1]["check"] is False

    def test_run_command_no_shell_injection_risk(self):
        """Verify that shell metacharacters are treated as literals (SEC-001)."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch.object(_fix_quality_issues.subprocess, "run", return_value=mock_result) as mock_subprocess:
            # A command that would be dangerous with shell=True
            # With shlex.split(), these are just literal arguments
            cmd = "echo test && ls"
            run_command(cmd, "Test injection safety")

            call_args = mock_subprocess.call_args
            first_arg = call_args[0][0]

            # With shlex.split(), "&&" is treated as a literal argument, not a shell operator
            assert isinstance(first_arg, list)
            assert "&&" in first_arg  # It's a literal string in the argument list
            # Verify shell is not True (or not present)
            assert call_args[1].get("shell", False) is False


class TestRunCommandOutput:
    """Test output handling of run_command."""

    def test_run_command_prints_stdout(self):
        """Test that stdout is printed."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Command output"
        mock_result.stderr = ""

        with patch.object(_fix_quality_issues.subprocess, "run", return_value=mock_result):
            with patch("builtins.print") as mock_print:
                run_command("echo test", "Test output")

                # Check that stdout was printed
                print_calls = [str(call) for call in mock_print.call_args_list]
                assert any("Command output" in str(call) for call in print_calls)

    def test_run_command_prints_stderr_on_error(self):
        """Test that stderr is printed when return code is non-zero."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error message"

        with patch.object(_fix_quality_issues.subprocess, "run", return_value=mock_result):
            with patch("builtins.print") as mock_print:
                run_command("false", "Test error output")

                # Check that stderr was printed
                print_calls = [str(call) for call in mock_print.call_args_list]
                assert any("Error message" in str(call) for call in print_calls)
