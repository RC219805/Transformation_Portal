"""CLI tests for the resolver-only ``--plan`` mode (P0-1, #2065).

``--plan`` must run the same validation and resolution a real run performs,
emit the canonical plan JSON, and stop before orchestrator construction —
writing nothing.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import pytest
from typer.testing import CliRunner

from transformation_portal.lux_depth_v3.__main__ import app
from transformation_portal.lux_depth_v3.config import (
    DeprecatedOutputFlagWarning,
    EnhanceConfig,
)

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
    @pytest.mark.parametrize(
        ("args", "expected_depth", "expected_warning_count"),
        [
            ([], 8, 0),
            (["--output-bit-depth", "8"], 8, 0),
            (["--output-bit-depth", "16"], 16, 0),
            (["--emit-master16", "on"], 16, 1),
            (["--emit-upscaled16", "on"], 16, 1),
            (["--emit-master16", "on", "--emit-upscaled16", "on"], 16, 1),
            (["--output-bit-depth", "16", "--emit-master16", "on"], 16, 1),
        ],
    )
    def test_output_bit_depth_cli_compatibility_matrix(
        self,
        tmp_path: Path,
        args: list[str],
        expected_depth: int,
        expected_warning_count: int,
    ) -> None:
        input_dir = _make_input_dir(tmp_path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = runner.invoke(
                app,
                [
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(tmp_path / "out"),
                    "--model-key",
                    "da3-metric",
                    *args,
                    "--plan",
                ],
            )

        assert result.exit_code == 0, result.output
        payload = _extract_plan_json(result.output)
        assert payload["output_bit_depth"] == expected_depth
        assert "bit_depth_16_intermediates" not in payload["requested_artifacts"]
        deprecations = [item for item in caught if isinstance(item.message, DeprecatedOutputFlagWarning)]
        assert len(deprecations) == expected_warning_count

    @pytest.mark.parametrize("legacy_flag", ["--emit-master16", "--emit-upscaled16"])
    def test_explicit_8_bit_rejects_truthy_legacy_cli_alias(self, tmp_path: Path, legacy_flag: str) -> None:
        input_dir = _make_input_dir(tmp_path)
        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "out"),
                "--output-bit-depth",
                "8",
                legacy_flag,
                "on",
                "--plan",
            ],
        )

        assert result.exit_code == 1
        assert "conflicts with a truthy deprecated" in result.output

    @pytest.mark.parametrize(
        ("flag", "value", "config_field", "config_value"),
        [
            ("--emit-marketing", "on", "emit_marketing", True),
            ("--emit-marketing", "off", "emit_marketing", False),
            ("--emit-report", "on", "emit_report", True),
            ("--emit-report", "off", "emit_report", False),
        ],
    )
    def test_deprecated_output_flag_warning_matches_direct_python_and_plan(
        self,
        tmp_path: Path,
        flag: str,
        value: str,
        config_field: str,
        config_value: bool,
    ) -> None:
        with pytest.warns(DeprecatedOutputFlagWarning) as direct_warnings:
            EnhanceConfig(model_key="da3-metric", **{config_field: config_value})
        assert len(direct_warnings) == 1
        expected_notice = str(direct_warnings[0].message)

        input_dir = _make_input_dir(tmp_path)
        with pytest.warns(DeprecatedOutputFlagWarning) as cli_warnings:
            result = runner.invoke(
                app,
                [
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(tmp_path / "out"),
                    "--model-key",
                    "da3-metric",
                    flag,
                    value,
                    "--plan",
                ],
            )

        assert result.exit_code == 0, result.output
        assert [str(item.message) for item in cli_warnings] == [expected_notice]
        payload = _extract_plan_json(result.output)
        assert payload["warnings"].count(expected_notice) == 1
        assert payload["requested_artifacts"].count("combined_manifest_json") == 1
        assert all("marketing" not in artifact for artifact in payload["requested_artifacts"])
        assert not (tmp_path / "out").exists()

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

    def test_plan_default_model_resolves_commercial_safe(self, tmp_path: Path) -> None:
        # Repair 1.2 (#2066, option A): the bare default plans the Apache-2.0
        # metric model with the distinct "default" selector recorded.
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
        assert result.exit_code == 0
        payload = _extract_plan_json(result.output)
        assert payload["resolved_model"]["canonical_key"] == "da3_metric"
        assert payload["resolved_model"]["requested_selector"] == "default"

    def test_plan_research_model_fails_license_gate_like_run(self, tmp_path: Path) -> None:
        input_dir = _make_input_dir(tmp_path)
        result = runner.invoke(
            app,
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(tmp_path / "out"),
                "--model-key",
                "da3-research",
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
