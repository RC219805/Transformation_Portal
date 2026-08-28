"""CLI tests for the resolver-only ``--plan`` mode (P0-1, #2065).

``--plan`` must run the same validation and resolution a real run performs,
emit the canonical plan JSON, and stop before orchestrator construction —
writing nothing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from typer.testing import CliRunner

from transformation_portal.lux_depth_v3.__main__ import app

pytestmark = [pytest.mark.unit]

runner = CliRunner()

# Root level at import time, before any test in this module runs; the
# leak-guard test compares against it after other CLI tests have executed.
# (Handler lists are not comparable across tests — pytest's logging plugin
# installs its own per-test capture handlers — but the root LEVEL is what
# gates record creation, and the leaked INFO level is what broke unrelated
# tests that monkeypatch time.time with finite iterators.)
_ROOT_LEVEL_BASELINE = logging.getLogger().level


@pytest.fixture(autouse=True)
def _restore_root_logging():
    """Undo the CLI's global logging side effects after each test.

    The CLI calls logging.basicConfig(force=True), which replaces root
    handlers and raises the root level to INFO for the whole process.
    Left in place, that leaks into unrelated tests later in the session
    (e.g. tests that monkeypatch time.time with finite iterators and then
    trip on logging's own time.time() call once INFO records start
    emitting).
    """
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def _make_input_dir(tmp_path: Path) -> Path:
    from PIL import Image

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    Image.new("RGB", (8, 8), color=(128, 128, 128)).save(input_dir / "sample.png")
    return input_dir


def _extract_plan_json(stdout: str) -> dict:
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("{"):
            return json.loads(stripped)
    raise AssertionError(f"no JSON object found in stdout: {stdout!r}")


class TestPlanMode:
    def test_plan_emits_canonical_json_and_writes_nothing(self, tmp_path: Path) -> None:
        input_dir = _make_input_dir(tmp_path)
        output_dir = tmp_path / "out"
        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--model-key",
                "da3-metric",
                "--plan",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = _extract_plan_json(result.output)
        assert payload["schema"] == "tp.lux.resolved_invocation.v1"
        assert payload["resolved_model"]["canonical_key"] == "da3_metric"
        assert payload["input_files"] == ["sample.png"]
        assert "executed_backend" not in payload
        # Resolver-only: the orchestrator was never constructed, so the
        # output root must not exist and the input dir must be untouched.
        assert not output_dir.exists()
        assert sorted(p.name for p in input_dir.iterdir()) == ["sample.png"]

    def test_plan_default_model_fails_license_gate_like_run(self, tmp_path: Path) -> None:
        input_dir = _make_input_dir(tmp_path)
        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "out"),
                "--plan",
            ],
        )
        assert result.exit_code == 1
        assert "non-commercial" in result.output

    def test_plan_apex_strict_gate_parity(self, tmp_path: Path) -> None:
        input_dir = _make_input_dir(tmp_path)
        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "out"),
                "--model-key",
                "da3-metric",
                "--quality-tier",
                "apex",
                "--materials-v3",
                "on",
                "--plan",
            ],
        )
        assert result.exit_code == 1
        assert "APEX strict gate" in result.output

    def test_run_mode_attaches_invocation_to_config(self, tmp_path: Path, monkeypatch) -> None:
        """Without --plan, the run path attaches the exact invocation object
        to the config before orchestrator construction (the single-resolution
        invariant's consumption side)."""
        import transformation_portal.lux_depth_v3.__main__ as cli_module

        captured: dict = {}

        class _StubOrchestrator:
            def __init__(self, config, output_root):
                captured["invocation"] = config.resolved_invocation
                captured["output_root"] = output_root

            def enhance_batch(self, input_dir, image_extensions, input_files=None):
                captured["input_files"] = input_files
                return []

        monkeypatch.setattr(cli_module, "EnhanceOrchestrator", _StubOrchestrator)
        input_dir = _make_input_dir(tmp_path)
        output_dir = tmp_path / "out"
        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--model-key",
                "da3-metric",
            ],
        )
        assert result.exit_code == 0, result.output
        invocation = captured["invocation"]
        assert invocation is not None
        assert invocation.schema == "tp.lux.resolved_invocation.v1"
        assert invocation.resolved_model.canonical_key == "da3_metric"
        # Run mode (unlike --plan) does create the output root.
        assert output_dir.exists()
        # The run consumes the plan's frozen input selection — the same
        # files the invocation recorded, not a rediscovered list.
        assert captured["input_files"] is not None
        assert [p.name for p in captured["input_files"]] == list(invocation.input_files)

    def test_cli_invocation_mutates_then_fixture_restores_root_logging(self, tmp_path: Path) -> None:
        """Regression guard for the session-wide logging leak.

        The CLI's basicConfig(force=True) raises the root level to INFO
        during invoke — prove that mutation happens here, and prove the
        autouse fixture restored the module-import baseline after every
        preceding CLI test in this class — any leak from those earlier
        invokes would show up in the pre-invoke state asserted below.
        """
        root = logging.getLogger()
        # Earlier tests in this class already invoked the CLI; without the
        # fixture their basicConfig(force=True) INFO level would still be
        # in effect here.
        assert root.level == _ROOT_LEVEL_BASELINE
        input_dir = _make_input_dir(tmp_path)
        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "out"),
                "--model-key",
                "da3-metric",
                "--plan",
            ],
        )
        assert result.exit_code == 0, result.output
        # The mutation the fixture exists to undo really does happen.
        assert root.level == logging.INFO

    def test_plan_json_deterministic_across_invocations(self, tmp_path: Path) -> None:
        input_dir = _make_input_dir(tmp_path)
        args = [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--model-key",
            "da3-metric",
            "--plan",
        ]
        first = runner.invoke(app, args)
        second = runner.invoke(app, args)
        assert first.exit_code == 0 and second.exit_code == 0
        assert _extract_plan_json(first.output) == _extract_plan_json(second.output)
