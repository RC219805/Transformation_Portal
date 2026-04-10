"""Tests for environment pre-flight validation scripts."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECK_LOCAL_ENVIRONMENT_PATH = PROJECT_ROOT / "scripts" / "validation" / "check_local_environment.py"


def _load_module(module_path: Path, module_name: str):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class TestCheckLocalEnvironment:
    """Tests for check_local_environment.py."""

    @pytest.fixture
    def env_module(self):
        """Load the check_local_environment module."""
        return _load_module(CHECK_LOCAL_ENVIRONMENT_PATH, "test_check_local_environment")

    def test_check_python_version_passes_for_current(self, env_module):
        """Current Python version should pass the check."""
        result = env_module.check_python_version()
        # We require Python 3.11+, and tests run on 3.11+
        assert result.passed is True
        assert "Python" in result.name
        assert result.is_hard_requirement is True

    def test_check_node_version_returns_result(self, env_module):
        """Node version check should return a CheckResult."""
        result = env_module.check_node_version()
        assert hasattr(result, "passed")
        assert hasattr(result, "message")
        assert hasattr(result, "name")
        # Result depends on whether Node is installed

    def test_check_chrome_returns_result(self, env_module):
        """Chrome check should return a CheckResult."""
        result = env_module.check_chrome_available()
        assert hasattr(result, "passed")
        assert hasattr(result, "message")
        assert result.is_hard_requirement is False  # Chrome is optional

    def test_check_port_available(self, env_module):
        """Port availability check should work for unused ports."""
        # Use a high port unlikely to be in use
        result = env_module.check_port_available(59999, "test port")
        # Should pass unless something is using that port
        assert hasattr(result, "passed")
        assert "59999" in result.name

    def test_check_venv_active_returns_result(self, env_module):
        """Venv check should return a CheckResult."""
        result = env_module.check_venv_active()
        assert hasattr(result, "passed")
        assert result.is_hard_requirement is False  # Venv is optional

    def test_check_venv_active_detects_venv_without_virtual_env_envvar(self, env_module, monkeypatch, tmp_path):
        """Interpreter-based venv detection should work without VIRTUAL_ENV."""
        monkeypatch.setattr(env_module, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(env_module, "REPO_VENV_PYTHON", tmp_path / ".venv" / "bin" / "python")
        monkeypatch.setattr(env_module.sys, "prefix", str(tmp_path / ".venv"))
        monkeypatch.setattr(env_module.sys, "base_prefix", str(tmp_path / "python-base"))
        monkeypatch.setattr(env_module.sys, "executable", str(tmp_path / ".venv" / "bin" / "python"))
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)

        result = env_module.check_venv_active()

        assert result.passed is True
        assert result.is_hard_requirement is False
        assert "Active:" in result.message

    def test_check_venv_active_hard_fails_when_repo_venv_exists_but_interpreter_differs(
        self,
        env_module,
        monkeypatch,
        tmp_path,
    ):
        """Preflight should fail closed when the repo venv exists but is not in use."""
        repo_python = tmp_path / ".venv" / "bin" / "python"
        repo_python.parent.mkdir(parents=True, exist_ok=True)
        repo_python.write_text("", encoding="utf-8")
        current_python = tmp_path / "bin" / "python"
        current_python.parent.mkdir(parents=True, exist_ok=True)
        current_python.write_text("", encoding="utf-8")

        monkeypatch.setattr(env_module, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(env_module, "REPO_VENV_PYTHON", repo_python)
        monkeypatch.setattr(env_module.sys, "executable", str(current_python))
        monkeypatch.setattr(env_module.sys, "prefix", str(tmp_path / "system"))
        monkeypatch.setattr(env_module.sys, "base_prefix", str(tmp_path / "system"))
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)

        result = env_module.check_venv_active()

        assert result.passed is False
        assert result.is_hard_requirement is True
        assert "make repair-core-venv" in (result.guidance or "")

    def test_dependency_health_check_surfaces_da3_contamination_guidance(self, env_module, monkeypatch):
        """DA3 contamination should point operators to the repair and isolated-runtime paths."""
        completed = subprocess.CompletedProcess(
            args=[sys.executable, "-m", "pip", "check"],
            returncode=1,
            stdout=(
                "depth-anything-3 0.0.0 requires xformers, which is not installed.\n"
                "depth-anything-3 0.0.0 has requirement numpy<2, but you have numpy 2.4.4.\n"
            ),
            stderr="",
        )

        monkeypatch.setattr(env_module.sys, "executable", "/tmp/repo/.venv/bin/python")
        monkeypatch.setattr(env_module.subprocess, "run", lambda *args, **kwargs: completed)

        result = env_module.check_dependency_health()

        assert result.passed is False
        assert result.is_hard_requirement is True
        assert "depth-anything-3" in result.message
        assert "make repair-core-venv" in (result.guidance or "")
        assert "./scripts/setup/install_da3_runtime.sh" in (result.guidance or "")

    def test_run_all_checks_returns_results_and_exit_code(self, env_module):
        """run_all_checks should return results and exit code."""
        results, exit_code = env_module.run_all_checks()
        assert isinstance(results, list)
        assert len(results) > 0
        assert exit_code in (0, 1, 2)  # ExitCode enum values

    def test_run_specific_check_python(self, env_module):
        """Running only Python check should work."""
        results, exit_code = env_module.run_all_checks(["python"])
        assert len(results) == 1
        assert "Python" in results[0].name

    def test_run_specific_check_ports(self, env_module):
        """Running only ports check should return multiple results."""
        results, exit_code = env_module.run_all_checks(["ports"])
        # Should check both 3000 and 8000
        assert len(results) >= 2
        assert all("Port" in r.name for r in results)

    def test_check_result_has_guidance_on_failure(self, env_module):
        """Failed checks should include guidance when available."""
        # Mock a failing check
        result = env_module.CheckResult(
            name="Test Check",
            passed=False,
            message="Test failure",
            guidance="Fix by doing X",
        )
        assert result.guidance is not None

    def test_exit_code_hard_fail_on_python_check_failure(self, env_module):
        """Python check failure should be a hard failure."""
        # Check that Python is marked as hard requirement
        result = env_module.check_python_version()
        assert result.is_hard_requirement is True

    def test_exit_code_soft_fail_for_optional_checks(self, env_module):
        """Optional check failures should be soft failures."""
        # Chrome is optional
        result = env_module.check_chrome_available()
        assert result.is_hard_requirement is False


class TestCheckLocalEnvironmentCLI:
    """Test the CLI interface of check_local_environment.py."""

    def test_script_runs_without_args(self):
        """Script should run without arguments."""
        result = subprocess.run(
            [sys.executable, str(CHECK_LOCAL_ENVIRONMENT_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should complete (exit code depends on environment)
        assert result.returncode in (0, 1, 2)

    def test_script_json_output(self):
        """Script should support JSON output."""
        result = subprocess.run(
            [sys.executable, str(CHECK_LOCAL_ENVIRONMENT_PATH), "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode in (0, 1, 2)
        # Should be valid JSON
        output = json.loads(result.stdout)
        assert "results" in output
        assert "exit_code" in output
        assert "status" in output

    def test_script_check_specific(self):
        """Script should support checking specific items."""
        result = subprocess.run(
            [sys.executable, str(CHECK_LOCAL_ENVIRONMENT_PATH), "--check", "python", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode in (0, 1, 2)
        output = json.loads(result.stdout)
        # Should only have Python-related checks
        assert len(output["results"]) == 1

    def test_script_check_specific_dependency_health(self):
        """Script should accept dependency-health as a specific check."""
        result = subprocess.run(
            [sys.executable, str(CHECK_LOCAL_ENVIRONMENT_PATH), "--check", "dependency-health", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode in (0, 2)
        output = json.loads(result.stdout)
        assert len(output["results"]) == 1
        assert output["results"][0]["name"] == "Python dependency health"

    def test_script_quiet_mode(self):
        """Quiet mode should reduce output."""
        result = subprocess.run(
            [sys.executable, str(CHECK_LOCAL_ENVIRONMENT_PATH), "--quiet"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode in (0, 1, 2)
        # Output should be minimal (only failures if any)

    def test_script_strict_mode(self):
        """Strict mode should treat soft failures as hard failures."""
        result = subprocess.run(
            [sys.executable, str(CHECK_LOCAL_ENVIRONMENT_PATH), "--strict", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = json.loads(result.stdout)
        # If there are any failures, exit code should be 2 (hard fail)
        if output["status"] != "pass":
            assert output["exit_code"] == 2


class TestEnsureNodeVersionScript:
    """Tests for ensure_node_version.sh."""

    ENSURE_NODE_VERSION_PATH = PROJECT_ROOT / "scripts" / "setup" / "ensure_node_version.sh"

    def test_script_exists_and_executable(self):
        """Script should exist."""
        assert self.ENSURE_NODE_VERSION_PATH.exists()

    def test_script_syntax_valid(self):
        """Script should have valid bash syntax."""
        result = subprocess.run(
            ["bash", "-n", str(self.ENSURE_NODE_VERSION_PATH)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"Syntax error: {result.stderr}"


class TestRunFullValidationSuite:
    """Tests for run_full_validation_suite.sh."""

    SCRIPT_PATH = PROJECT_ROOT / "scripts" / "validation" / "run_full_validation_suite.sh"

    def test_script_exists(self):
        """Script should exist."""
        assert self.SCRIPT_PATH.exists()

    def test_script_syntax_valid(self):
        """Script should have valid bash syntax."""
        result = subprocess.run(
            ["bash", "-n", str(self.SCRIPT_PATH)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"Syntax error: {result.stderr}"

    def test_script_help_works(self):
        """Script should display help."""
        result = subprocess.run(
            ["bash", str(self.SCRIPT_PATH), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "Usage" in result.stdout


class TestCheckWorktreeClean:
    """Tests for check_worktree_clean.sh."""

    SCRIPT_PATH = PROJECT_ROOT / "scripts" / "validation" / "check_worktree_clean.sh"

    def test_script_exists(self):
        """Script should exist."""
        assert self.SCRIPT_PATH.exists()

    def test_script_syntax_valid(self):
        """Script should have valid bash syntax."""
        result = subprocess.run(
            ["bash", "-n", str(self.SCRIPT_PATH)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"Syntax error: {result.stderr}"

    def test_script_help_works(self):
        """Script should display help."""
        result = subprocess.run(
            ["bash", str(self.SCRIPT_PATH), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "Usage" in result.stdout
