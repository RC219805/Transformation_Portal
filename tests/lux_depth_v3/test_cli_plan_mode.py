"""CLI tests for the resolver-only ``--plan`` mode (P0-1, #2065).

``--plan`` must run the same validation and resolution a real run performs,
emit the canonical plan JSON, and stop before orchestrator construction —
writing nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from transformation_portal.lux_depth_v3.__main__ import app

pytestmark = [pytest.mark.unit]

runner = CliRunner()


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
