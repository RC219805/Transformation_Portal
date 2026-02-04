"""Tests for fix_quality_issues.py script.

Verifies that subprocess calls are executed safely without shell=True,
preventing command injection vulnerabilities (SEC-001).
"""
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, call
import pytest
import sys

# Add scripts/utilities to path for import
scripts_path = Path(__file__).parent.parent / "scripts" / "utilities"
sys.path.insert(0, str(scripts_path))

from fix_quality_issues import run_command


class TestRunCommandSecurity:
    """Test that run_command uses safe subprocess execution."""

    @patch('fix_quality_issues.subprocess.run')
    def test_run_command_uses_argument_list_not_shell(self, mock_subprocess):
        """Verify that run_command does NOT use shell=True (SEC-001)."""
        # Setup mock
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Success"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        # Call run_command with a simple command
        result = run_command("echo hello world", "Test command")

        # Verify subprocess.run was called
        assert mock_subprocess.called
        call_args = mock_subprocess.call_args

        # Critical: Verify first argument is a list (not a string)
        first_arg = call_args[0][0]
        assert isinstance(first_arg, list), "Command should be passed as list, not string"
        assert first_arg == ["echo", "hello", "world"], "Command should be split into arguments"

        # Critical: Verify shell=True is NOT used
        # The call should not have shell=True in kwargs
        if 'shell' in call_args[1]:
            assert call_args[1]['shell'] is False, "shell parameter should be False or absent"

        assert result == 0

    @patch('fix_quality_issues.subprocess.run')
    def test_run_command_splits_complex_command(self, mock_subprocess):
        """Test that complex commands with flags are properly split."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        # Test with a command that has multiple flags (similar to actual usage)
        cmd = "flake8 . --count --select=E9,F63 --show-source"
        run_command(cmd, "Test flake8")

        # Verify command was split correctly
        call_args = mock_subprocess.call_args
        first_arg = call_args[0][0]
        assert isinstance(first_arg, list)
        assert first_arg[0] == "flake8"
        assert "." in first_arg
        assert "--count" in first_arg
        assert "--select=E9,F63" in first_arg
        assert "--show-source" in first_arg

    @patch('fix_quality_issues.subprocess.run')
    def test_run_command_handles_return_code(self, mock_subprocess):
        """Test that return codes are properly propagated."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error occurred"
        mock_subprocess.return_value = mock_result

        result = run_command("flake8 .", "Test error handling")

        assert result == 1

    @patch('fix_quality_issues.subprocess.run')
    def test_run_command_uses_capture_output(self, mock_subprocess):
        """Verify capture_output and text parameters are set correctly."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "output"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        run_command("echo test", "Test capture")

        call_args = mock_subprocess.call_args
        assert call_args[1]['capture_output'] is True
        assert call_args[1]['text'] is True
        assert call_args[1]['check'] is False

    @patch('fix_quality_issues.subprocess.run')
    def test_run_command_no_shell_injection_risk(self, mock_subprocess):
        """Verify that shell metacharacters are treated as literals (SEC-001)."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

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
        assert call_args[1].get('shell', False) is False


class TestRunCommandOutput:
    """Test output handling of run_command."""

    @patch('fix_quality_issues.subprocess.run')
    @patch('builtins.print')
    def test_run_command_prints_stdout(self, mock_print, mock_subprocess):
        """Test that stdout is printed."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Command output"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        run_command("echo test", "Test output")

        # Check that stdout was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("Command output" in str(call) for call in print_calls)

    @patch('fix_quality_issues.subprocess.run')
    @patch('builtins.print')
    def test_run_command_prints_stderr_on_error(self, mock_print, mock_subprocess):
        """Test that stderr is printed when return code is non-zero."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error message"
        mock_subprocess.return_value = mock_result

        run_command("false", "Test error output")

        # Check that stderr was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("Error message" in str(call) for call in print_calls)
