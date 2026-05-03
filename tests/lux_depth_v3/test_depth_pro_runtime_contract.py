"""Contract tests for the Depth Pro subprocess runtime.

Depth Pro is research-license, runs in its own ``.venv-depth-pro`` because
of the NumPy 1.x / torch 2.7.1 / torchvision 0.22.1 pin conflict, and is
governed by an explicit license-acknowledgement gate. The end-to-end
inference path is exercised in the ML lane; this file pins the contract
that lives outside the model:

* The constants the orchestrator and install script depend on
  (default checkpoint path, expected SHA256, worker module name,
  HF download URL).
* The subprocess argv shape (``--check`` mode vs inference mode).
* The class-level resolver behaviour for the checkpoint path
  (config / env / default precedence) — none of which require torch.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit]

from transformation_portal.depth.backends.depth_pro import DepthProBackend  # noqa: E402


class TestDepthProBackendConstants:
    """The backend constants are the wire-format of the runtime contract."""

    def test_default_checkpoint_path_is_governed(self):
        # Any change to this path requires updating
        # scripts/setup/install_depth_pro_runtime.sh and the runbook;
        # lock the contract.
        assert DepthProBackend.DEFAULT_CHECKPOINT == Path("checkpoints/depth_pro.pt")

    def test_expected_sha256_is_lowercase_64_char_hex(self):
        # Integrity verification expects a lowercase 64-char hex digest;
        # smoke-test the constant before runtime hashing relies on it.
        sha = DepthProBackend.EXPECTED_SHA256
        assert isinstance(sha, str)
        assert len(sha) == 64
        assert sha == sha.lower()
        assert all(c in "0123456789abcdef" for c in sha)

    def test_checkpoint_url_is_apple_cdn(self):
        # Source of truth for ``install_depth_pro_runtime.sh``: if Apple
        # moves the artifact, this constant must change too.
        assert DepthProBackend.CHECKPOINT_URL.startswith("https://")
        assert "depth-pro" in DepthProBackend.CHECKPOINT_URL

    def test_worker_module_name_matches_runtime_dispatch(self):
        # The orchestrator launches the worker with ``-m WORKER_MODULE``;
        # this string is part of the cross-process protocol.
        assert DepthProBackend.WORKER_MODULE == "transformation_portal.depth.backends.depth_pro_worker"

    def test_license_metadata(self):
        # Depth Pro is research-only; the registry uses these fields to
        # gate selection on ``non_commercial_ok`` plus the explicit
        # ``accept_apple_depth_pro_research_license`` flag.
        assert DepthProBackend.name == "depth_pro"
        assert DepthProBackend.requires_checkpoint is True
        # license_type should signal "research-only" via the protocol enum.
        assert DepthProBackend.license_type.name == "RESEARCH_ONLY"


class TestCheckpointResolution:
    """``_resolve_checkpoint_path`` precedence: config > env > default."""

    def _backend(self, **config_attrs):
        # Build a backend without invoking the heavy initializer paths
        # that touch git / repo discovery: we only need the resolver.
        backend = DepthProBackend.__new__(DepthProBackend)
        return backend, SimpleNamespace(**config_attrs)

    def test_config_wins_over_env(self, monkeypatch, tmp_path: Path):
        backend, config = self._backend(depth_pro_checkpoint_path=str(tmp_path / "from-config.pt"))
        monkeypatch.setenv("TRANSFORMATION_PORTAL_DEPTH_PRO_CHECKPOINT", str(tmp_path / "from-env.pt"))

        result = backend._resolve_checkpoint_path(config)
        assert result == tmp_path / "from-config.pt"

    def test_env_used_when_config_missing(self, monkeypatch, tmp_path: Path):
        backend, _ = self._backend()
        monkeypatch.setenv("TRANSFORMATION_PORTAL_DEPTH_PRO_CHECKPOINT", str(tmp_path / "from-env.pt"))

        result = backend._resolve_checkpoint_path(SimpleNamespace())
        assert result == tmp_path / "from-env.pt"

    def test_default_used_when_neither_config_nor_env(self, monkeypatch):
        backend, _ = self._backend()
        monkeypatch.delenv("TRANSFORMATION_PORTAL_DEPTH_PRO_CHECKPOINT", raising=False)

        result = backend._resolve_checkpoint_path(SimpleNamespace())
        assert result == DepthProBackend.DEFAULT_CHECKPOINT

    def test_tilde_is_expanded(self, monkeypatch, tmp_path: Path):
        backend, _ = self._backend()
        monkeypatch.setenv("TRANSFORMATION_PORTAL_DEPTH_PRO_CHECKPOINT", "~/depth_pro.pt")
        monkeypatch.setenv("HOME", str(tmp_path))

        result = backend._resolve_checkpoint_path(SimpleNamespace())
        assert "~" not in str(result)
        assert result == tmp_path / "depth_pro.pt"


class TestPythonExecutableResolution:
    """``_resolve_python_executable`` resolves the dedicated venv interpreter."""

    def _backend(self, *, repo_root: Path | None = None):
        backend = DepthProBackend.__new__(DepthProBackend)
        backend._repo_root = repo_root
        return backend

    def test_returns_none_when_unconfigured(self, monkeypatch):
        backend = self._backend()
        monkeypatch.delenv("TRANSFORMATION_PORTAL_DEPTH_PRO_PYTHON", raising=False)
        assert backend._resolve_python_executable(None) is None
        assert backend._resolve_python_executable(SimpleNamespace()) is None

    def test_relative_path_must_exist(self, monkeypatch, tmp_path: Path):
        # Failing closed on a missing interpreter is the security contract:
        # we don't fall through to a system Python that wouldn't have the
        # depth_pro package.
        backend = self._backend(repo_root=tmp_path)
        monkeypatch.chdir(tmp_path)

        with pytest.raises(FileNotFoundError, match="Depth Pro Python executable not found"):
            backend._resolve_python_executable(SimpleNamespace(depth_pro_python_executable="./.venv-depth-pro/bin/python"))

    def test_relative_path_is_absolutized(self, monkeypatch, tmp_path: Path):
        backend = self._backend(repo_root=tmp_path)
        venv = tmp_path / ".venv-depth-pro" / "bin"
        venv.mkdir(parents=True)
        interpreter = venv / "python"
        interpreter.write_text("#!/bin/sh\nexec /bin/false\n")
        interpreter.chmod(0o755)
        monkeypatch.chdir(tmp_path)

        result = backend._resolve_python_executable(
            SimpleNamespace(depth_pro_python_executable="./.venv-depth-pro/bin/python")
        )
        assert result is not None
        assert os.path.isabs(result)
        # Symlink/relative path is preserved into an absolute form (without
        # collapsing through symlinks); the runtime needs the venv path.
        assert result.endswith(".venv-depth-pro/bin/python")

    def test_env_var_used_when_config_silent(self, monkeypatch, tmp_path: Path):
        backend = self._backend(repo_root=tmp_path)
        venv = tmp_path / ".venv-depth-pro" / "bin"
        venv.mkdir(parents=True)
        interpreter = venv / "python"
        interpreter.write_text("#!/bin/sh\nexec /bin/false\n")
        interpreter.chmod(0o755)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(
            "TRANSFORMATION_PORTAL_DEPTH_PRO_PYTHON",
            "./.venv-depth-pro/bin/python",
        )

        result = backend._resolve_python_executable(SimpleNamespace())
        assert result is not None
        assert result.endswith(".venv-depth-pro/bin/python")

    def test_bare_name_must_be_on_path(self, monkeypatch):
        # When neither "." nor a separator is in the candidate, we delegate
        # to PATH lookup. A non-existent name must fail closed.
        backend = self._backend()
        monkeypatch.setenv("PATH", "/this/path/does/not/exist")
        with pytest.raises(FileNotFoundError, match="not found on PATH"):
            backend._resolve_python_executable(
                SimpleNamespace(depth_pro_python_executable="definitely-not-a-real-binary-name")
            )


class TestDepthProWorkerArgvContract:
    """The subprocess worker's argv flags are the cross-process protocol."""

    def _parser(self) -> argparse.ArgumentParser:
        from transformation_portal.depth.backends import depth_pro_worker  # type: ignore[attr-defined]

        return depth_pro_worker._build_parser()

    def test_check_only_invocation_parses(self, tmp_path: Path):
        parser = self._parser()
        args = parser.parse_args(["--check", "--checkpoint", str(tmp_path / "checkpoint.pt")])
        assert args.check is True
        assert args.checkpoint == tmp_path / "checkpoint.pt"
        assert args.device == "cpu"
        # Inference args are optional in --check mode.
        assert args.input_image is None
        assert args.output_depth is None
        assert args.output_json is None

    def test_inference_invocation_parses(self, tmp_path: Path):
        parser = self._parser()
        args = parser.parse_args(
            [
                "--checkpoint",
                str(tmp_path / "checkpoint.pt"),
                "--device",
                "mps",
                "--input-image",
                str(tmp_path / "in.png"),
                "--output-depth",
                str(tmp_path / "out.npy"),
                "--output-json",
                str(tmp_path / "out.json"),
            ]
        )
        assert args.check is False
        assert args.device == "mps"
        assert args.input_image == tmp_path / "in.png"
        assert args.output_depth == tmp_path / "out.npy"
        assert args.output_json == tmp_path / "out.json"

    def test_checkpoint_is_required(self):
        parser = self._parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--check"])

    def test_device_default_is_cpu(self, tmp_path: Path):
        # CPU is the safe default — the orchestrator always passes an
        # explicit device, but ad-hoc smoke invocations rely on this.
        parser = self._parser()
        args = parser.parse_args(["--check", "--checkpoint", str(tmp_path / "ck.pt")])
        assert args.device == "cpu"
