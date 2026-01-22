#!/usr/bin/env python3
"""
Tests for Manifest Environment Capture
=======================================

Tests for lux_depth_v3/enhance/manifest.py focusing on:
- Environment capture without torch
- Environment capture with torch (CPU-only)
- Environment capture with CUDA (missing nvidia-smi)
- Environment capture with CUDA (driver version captured)
- subprocess.run with shell=False verification
"""

import pytest
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from lux_depth_v3.enhance.manifest import (
    capture_environment,
    EnvironmentMetadata,
    get_git_revision,
)


class FakeCompletedProcess:
    """Fake subprocess.CompletedProcess for testing."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestCaptureEnvironmentWithoutTorch:
    """Test environment capture when torch is not available."""

    def test_capture_without_torch(self, monkeypatch):
        """Test environment capture when torch is not installed."""
        # Mock torch import to fail
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        env = capture_environment()

        # Should capture Python version
        assert env.python == sys.version.split()[0]

        # Torch-related fields should be None
        assert env.torch is None
        assert env.cuda_runtime is None
        assert env.gpu_name is None
        assert env.driver is None

        # OS platform should be captured
        assert env.os_platform is not None

    def test_basic_environment_fields(self):
        """Test that basic environment fields are always captured."""
        env = capture_environment()

        # Python version should always be present
        assert env.python is not None
        assert len(env.python) > 0

        # OS platform should be present
        assert env.os_platform in ["Linux", "Darwin", "Windows", "Java"]


class TestCaptureEnvironmentWithTorchCPU:
    """Test environment capture with torch (CPU-only)."""

    def test_capture_with_torch_cpu_only(self, monkeypatch):
        """Test environment capture when torch is available but CUDA is not."""
        # Mock torch module
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.1.0"
        mock_torch.cuda.is_available.return_value = False

        # Replace torch in sys.modules
        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        env = capture_environment()

        # Should capture torch version
        assert env.torch == "2.1.0"

        # CUDA not available → CUDA fields should be None
        assert env.cuda_runtime is None
        assert env.gpu_name is None
        assert env.driver is None


class TestCaptureEnvironmentWithCUDA:
    """Test environment capture with CUDA available."""

    def test_capture_with_cuda_missing_nvidia_smi(self, monkeypatch):
        """Test environment capture when CUDA is available but nvidia-smi fails."""
        # Mock torch with CUDA
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.1.0+cu121"
        mock_torch.cuda.is_available.return_value = True
        mock_torch.version.cuda = "12.1"
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 4090"

        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        # Mock subprocess.run to fail (nvidia-smi not found)
        def mock_run(*args, **kwargs):
            raise FileNotFoundError("nvidia-smi not found")

        monkeypatch.setattr(subprocess, "run", mock_run)

        env = capture_environment()

        # Should capture torch and CUDA runtime
        assert env.torch == "2.1.0+cu121"
        assert env.cuda_runtime == "12.1"
        assert env.gpu_name == "NVIDIA GeForce RTX 4090"

        # Driver version should be None (nvidia-smi failed)
        assert env.driver is None

    def test_capture_with_cuda_and_driver(self, monkeypatch):
        """Test environment capture with successful nvidia-smi call."""
        # Mock torch with CUDA
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.1.0+cu121"
        mock_torch.cuda.is_available.return_value = True
        mock_torch.version.cuda = "12.1"
        mock_torch.cuda.get_device_name.return_value = "NVIDIA A100"

        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        # Mock subprocess.run to succeed
        def mock_run(*args, **kwargs):
            return FakeCompletedProcess(returncode=0, stdout="535.104.12\n")

        monkeypatch.setattr(subprocess, "run", mock_run)

        env = capture_environment()

        # Should capture all CUDA info including driver
        assert env.torch == "2.1.0+cu121"
        assert env.cuda_runtime == "12.1"
        assert env.gpu_name == "NVIDIA A100"
        assert env.driver == "535.104.12"

    def test_capture_with_cuda_nvidia_smi_timeout(self, monkeypatch):
        """Test that nvidia-smi has a timeout."""
        # Mock torch with CUDA
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.1.0"
        mock_torch.cuda.is_available.return_value = True
        mock_torch.version.cuda = "12.1"
        mock_torch.cuda.get_device_name.return_value = "NVIDIA H100"

        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        # Mock subprocess.run to raise timeout
        def mock_run(*args, **kwargs):
            raise subprocess.TimeoutExpired("nvidia-smi", timeout=2)

        monkeypatch.setattr(subprocess, "run", mock_run)

        env = capture_environment()

        # Should handle timeout gracefully
        assert env.torch == "2.1.0"
        assert env.cuda_runtime == "12.1"
        assert env.gpu_name == "NVIDIA H100"
        assert env.driver is None  # Timeout → no driver version

    def test_capture_with_cuda_gpu_name_exception(self, monkeypatch):
        """Test that GPU name exception is handled gracefully."""
        # Mock torch with CUDA but get_device_name raises exception
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.1.0"
        mock_torch.cuda.is_available.return_value = True
        mock_torch.version.cuda = "12.1"
        mock_torch.cuda.get_device_name.side_effect = RuntimeError("No GPU available")

        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        env = capture_environment()

        # Should handle GPU name exception gracefully
        assert env.torch == "2.1.0"
        assert env.cuda_runtime == "12.1"
        assert env.gpu_name is None  # Exception → no GPU name


class TestNvidiaSmiSecurityShellFalse:
    """Test that nvidia-smi is called with shell=False for security."""

    def test_capture_environment_enforces_shell_false(self, monkeypatch):
        """Verify nvidia-smi is called with shell=False for security."""
        # Mock torch with CUDA
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.1.0"
        mock_torch.cuda.is_available.return_value = True
        mock_torch.version.cuda = "12.1"
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GPU"

        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        # Track subprocess.run calls
        run_calls = []

        def mock_run(cmd, **kwargs):
            run_calls.append({"cmd": cmd, "kwargs": kwargs})
            return FakeCompletedProcess(returncode=0, stdout="535.104.12\n")

        monkeypatch.setattr(subprocess, "run", mock_run)

        env = capture_environment()

        # Should have called subprocess.run
        assert len(run_calls) == 1

        # Verify shell=False (either explicitly set or not present, which defaults to False)
        call_kwargs = run_calls[0]["kwargs"]
        shell_param = call_kwargs.get("shell", False)
        assert shell_param is False, "nvidia-smi must be called with shell=False for security"

        # Verify command is a list (required when shell=False)
        cmd = run_calls[0]["cmd"]
        assert isinstance(cmd, list), "Command must be a list when shell=False"
        assert cmd[0] == "nvidia-smi"

    def test_nvidia_smi_command_structure(self, monkeypatch):
        """Test that nvidia-smi command is properly structured."""
        # Mock torch with CUDA
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.1.0"
        mock_torch.cuda.is_available.return_value = True
        mock_torch.version.cuda = "12.1"
        mock_torch.cuda.get_device_name.return_value = "GPU"

        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        # Track subprocess.run calls
        run_calls = []

        def mock_run(cmd, **kwargs):
            run_calls.append({"cmd": cmd, "kwargs": kwargs})
            return FakeCompletedProcess(returncode=0, stdout="535.104.12\n")

        monkeypatch.setattr(subprocess, "run", mock_run)

        env = capture_environment()

        # Verify command structure
        cmd = run_calls[0]["cmd"]
        assert cmd == [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ]

        # Verify timeout is set
        assert "timeout" in run_calls[0]["kwargs"]
        assert run_calls[0]["kwargs"]["timeout"] == 2


class TestGitRevisionSecurity:
    """Test git revision capture with security validation."""

    def test_get_git_revision_validates_repository(self, tmp_path):
        """Test that get_git_revision validates repository path."""
        # Non-git directory
        non_git_dir = tmp_path / "not_a_repo"
        non_git_dir.mkdir()

        # Should return None for non-git directory
        result = get_git_revision(non_git_dir)
        assert result is None

    def test_get_git_revision_with_valid_repo(self, tmp_path, monkeypatch):
        """Test get_git_revision with a valid repository."""
        # Create mock .git directory
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        git_dir = repo_dir / ".git"
        git_dir.mkdir()

        # Mock validate_git_repository to return the repo
        def mock_validate(path):
            if path == repo_dir and git_dir.exists():
                return repo_dir
            return None

        # Mock subprocess.run to return a commit SHA
        def mock_run(*args, **kwargs):
            # Verify security settings
            env = kwargs.get("env", {})
            assert env.get("GIT_TEMPLATE_DIR") == "", "GIT_TEMPLATE_DIR should be disabled"
            assert env.get("GIT_CONFIG_NOSYSTEM") == "1", "System git config should be disabled"
            assert "GIT_DIR" in env, "GIT_DIR should be set"

            return FakeCompletedProcess(returncode=0, stdout="a1b2c3d4e5f6789012345678901234567890abcd\n")

        from lux_depth_v3.enhance import manifest

        monkeypatch.setattr(manifest, "validate_git_repository", mock_validate)
        monkeypatch.setattr(subprocess, "run", mock_run)

        result = get_git_revision(repo_dir)
        assert result == "a1b2c3d4e5f6789012345678901234567890abcd"

    def test_get_git_revision_timeout(self, tmp_path, monkeypatch):
        """Test that get_git_revision has a timeout."""
        # Create mock .git directory
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        git_dir = repo_dir / ".git"
        git_dir.mkdir()

        # Mock validate_git_repository
        def mock_validate(path):
            return repo_dir if path == repo_dir else None

        # Mock subprocess.run to timeout
        def mock_run(*args, **kwargs):
            assert "timeout" in kwargs, "Git command should have timeout"
            assert kwargs["timeout"] == 5, "Timeout should be 5 seconds"
            raise subprocess.TimeoutExpired("git", timeout=5)

        from lux_depth_v3.enhance import manifest

        monkeypatch.setattr(manifest, "validate_git_repository", mock_validate)
        monkeypatch.setattr(subprocess, "run", mock_run)

        result = get_git_revision(repo_dir)
        assert result is None  # Timeout should be handled gracefully


class TestEnvironmentMetadataDataclass:
    """Test EnvironmentMetadata dataclass structure."""

    def test_environment_metadata_creation(self):
        """Test creating EnvironmentMetadata with all fields."""
        env = EnvironmentMetadata(
            python="3.10.12",
            torch="2.1.0",
            cuda_runtime="12.1",
            gpu_name="NVIDIA A100",
            driver="535.104.12",
            os_platform="Linux",
        )

        assert env.python == "3.10.12"
        assert env.torch == "2.1.0"
        assert env.cuda_runtime == "12.1"
        assert env.gpu_name == "NVIDIA A100"
        assert env.driver == "535.104.12"
        assert env.os_platform == "Linux"

    def test_environment_metadata_partial(self):
        """Test creating EnvironmentMetadata with partial fields."""
        env = EnvironmentMetadata(
            python="3.11.5",
            os_platform="Darwin",
        )

        assert env.python == "3.11.5"
        assert env.torch is None
        assert env.cuda_runtime is None
        assert env.gpu_name is None
        assert env.driver is None
        assert env.os_platform == "Darwin"


class TestCaptureEnvironmentIntegration:
    """Integration tests for capture_environment."""

    def test_capture_environment_returns_valid_metadata(self):
        """Test that capture_environment returns valid EnvironmentMetadata."""
        env = capture_environment()

        # Should always return EnvironmentMetadata instance
        assert isinstance(env, EnvironmentMetadata)

        # Python version should always be present
        assert env.python is not None
        assert isinstance(env.python, str)
        assert len(env.python) > 0

        # OS platform should be present
        assert env.os_platform is not None
        assert isinstance(env.os_platform, str)

    def test_capture_environment_idempotent(self):
        """Test that calling capture_environment multiple times is safe."""
        env1 = capture_environment()
        env2 = capture_environment()

        # Should return consistent results
        assert env1.python == env2.python
        assert env1.os_platform == env2.os_platform

        # Torch-related fields should be consistent
        assert env1.torch == env2.torch
        assert env1.cuda_runtime == env2.cuda_runtime
