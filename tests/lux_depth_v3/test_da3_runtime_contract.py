"""Contract tests for the DA3 subprocess runtime discovery surface.

The DA3 runtime is wired together by three loosely-coupled pieces:

* ``transformation_portal.core.da3_runtime`` — the venv-discovery contract
  (``REPO_LOCAL_DA3_PYTHON``, ``find_repo_root``, ``repo_local_da3_python_path``).
* ``transformation_portal.depth.backends.da3_worker`` — the subprocess
  argv contract (``--check``, ``--model-variant``, ``--input-image`` …).
* ``transformation_portal.lux_depth_v3.config_resolver`` — the resolver
  that pipes the discovered interpreter into ``EnhanceConfig``.

Each piece has its own tests, but the venv-path and worker-argv parts of
the contract are not exercised together. This file pins them so a quiet
edit (renaming the path constant, dropping a worker flag, changing
worker module name) cannot ship without a visible test break.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

from transformation_portal.core import da3_runtime  # noqa: E402


class TestRepoLocalDA3PythonConstant:
    """The path constant is the wire-format of the runtime contract."""

    def test_canonical_constant_value(self):
        # The orchestrator, install script, and runtime auto-discovery
        # all rely on this exact string. Lock it.
        assert da3_runtime.REPO_LOCAL_DA3_PYTHON == "./.runtime/Depth-Anything-3/.venv-da3/bin/python"

    def test_constant_components_are_stable(self):
        # ``_REPO_LOCAL_DA3_PYTHON_PARTS`` is the structured form used by
        # ``repo_local_da3_python_path``; both must agree on the layout.
        assert da3_runtime._REPO_LOCAL_DA3_PYTHON_PARTS == (
            ".runtime",
            "Depth-Anything-3",
            ".venv-da3",
            "bin",
            "python",
        )


class TestFindRepoRoot:
    """``find_repo_root`` walks up to the directory containing pyproject + src."""

    def _make_repo(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "pyproject.toml").write_text("[project]\nname = 'fake'\n", encoding="utf-8")
        (root / "src").mkdir()
        return root

    def test_finds_root_when_started_inside_subdir(self, tmp_path: Path):
        repo = self._make_repo(tmp_path / "repo")
        nested = repo / "src" / "pkg" / "deep" / "module"
        nested.mkdir(parents=True)

        assert da3_runtime.find_repo_root(nested) == repo.resolve()

    def test_finds_root_when_started_at_a_file(self, tmp_path: Path):
        repo = self._make_repo(tmp_path / "repo")
        leaf = repo / "src" / "pkg" / "module.py"
        leaf.parent.mkdir(parents=True)
        leaf.write_text("# placeholder\n", encoding="utf-8")

        assert da3_runtime.find_repo_root(leaf) == repo.resolve()

    def test_returns_none_when_no_marker_found(self, tmp_path: Path):
        # No pyproject.toml + src marker anywhere up the tree.
        nested = tmp_path / "no_repo" / "deep"
        nested.mkdir(parents=True)
        assert da3_runtime.find_repo_root(nested) is None

    def test_requires_both_pyproject_and_src(self, tmp_path: Path):
        # A ``pyproject.toml`` without ``src/`` (or vice-versa) must NOT
        # be mistaken for the repo root — we'd ship a wrong PYTHONPATH.
        bare = tmp_path / "bare"
        bare.mkdir()
        (bare / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
        # No src/ — should walk past it.
        assert da3_runtime.find_repo_root(bare) is None


class TestRepoLocalDA3PythonPath:
    def test_returns_canonical_layout_under_repo_root(self, tmp_path: Path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
        (repo / "src").mkdir()

        result = da3_runtime.repo_local_da3_python_path(repo)
        assert result == repo / ".runtime" / "Depth-Anything-3" / ".venv-da3" / "bin" / "python"

    def test_returns_none_when_outside_a_checkout(self, tmp_path: Path):
        # Caller code uses the None return as the signal to fall back to
        # the configured / env-provided interpreter.
        elsewhere = tmp_path / "loose"
        elsewhere.mkdir()
        assert da3_runtime.repo_local_da3_python_path(elsewhere) is None


class TestDA3WorkerArgvContract:
    """The subprocess worker's argv flags are part of the runtime contract.

    We import the parser builder directly rather than spawning the worker
    so the test stays offline and independent of torch / transformers.
    """

    def _parser(self) -> argparse.ArgumentParser:
        # The worker imports torch lazily inside _check_availability, so the
        # parser builder itself is import-safe even when ML deps are absent.
        from transformation_portal.depth.backends import da3_worker  # type: ignore[attr-defined]

        return da3_worker._build_parser()

    def test_check_only_invocation_parses(self):
        parser = self._parser()
        args = parser.parse_args(["--check", "--model-variant", "METRIC_LARGE"])
        assert args.check is True
        assert args.model_variant == "METRIC_LARGE"
        # Inference arguments are optional in --check mode.
        assert args.input_image is None
        assert args.output_depth is None
        assert args.output_json is None

    def test_inference_invocation_parses_all_required_flags(self, tmp_path: Path):
        parser = self._parser()
        args = parser.parse_args(
            [
                "--model-variant",
                "METRIC_LARGE",
                "--model-key",
                "da3_metric",
                "--device",
                "cpu",
                "--input-image",
                str(tmp_path / "in.png"),
                "--output-depth",
                str(tmp_path / "out.npy"),
                "--output-json",
                str(tmp_path / "out.json"),
            ]
        )
        assert args.check is False
        assert args.model_variant == "METRIC_LARGE"
        assert args.model_key == "da3_metric"
        assert args.device == "cpu"
        assert args.input_image == tmp_path / "in.png"
        assert args.output_depth == tmp_path / "out.npy"
        assert args.output_json == tmp_path / "out.json"

    def test_use_coreml_and_non_commercial_flags_are_opt_in(self):
        parser = self._parser()
        baseline = parser.parse_args(["--check", "--model-variant", "METRIC_LARGE"])
        assert baseline.use_coreml is False
        assert baseline.non_commercial_ok is False

        opted_in = parser.parse_args(
            [
                "--check",
                "--model-variant",
                "METRIC_LARGE",
                "--use-coreml",
                "--non-commercial-ok",
            ]
        )
        assert opted_in.use_coreml is True
        assert opted_in.non_commercial_ok is True

    def test_model_variant_is_required(self):
        parser = self._parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--check"])
