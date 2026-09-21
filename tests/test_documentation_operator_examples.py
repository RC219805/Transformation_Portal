"""Execute maintained operator examples without models or external services."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from transformation_portal.lux_depth_v3 import __main__ as lux_cli
from transformation_portal.lux_depth_v3.v2_presets import V2EnhancementConfig

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
APEX_GUIDES = (
    "docs/cli/CLI_REFERENCE.md",
    "docs/guides/SUPPORTED_FILE_FORMATS.md",
    "docs/pipeline_docs/PIPELINE_OPERATIONS_GUIDE.md",
)


def _documented_command(path: str, program: str) -> list[str]:
    """Extract a complete command, retaining the document's actual options."""
    text = (REPO_ROOT / path).read_text(encoding="utf-8")
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", text, re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            arguments = shlex.split(line, comments=True)
            if program in arguments and "--help" not in arguments:
                commands.append(arguments)
    assert len(commands) == 1, f"Expected one {program} example in {path}, found {len(commands)}"
    return commands[0]


def _replace_path(arguments: list[str], option: str, path: Path) -> None:
    """Keep flags intact while supplying temporary local fixture paths."""
    arguments[arguments.index(option) + 1] = str(path)


@pytest.mark.parametrize("guide", APEX_GUIDES)
def test_documented_apex_example_passes_real_plan_admission(
    guide: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch missing strict-segmentation flags in copied runnable examples."""
    arguments = _documented_command(guide, ".venv/bin/lux-depth-v3")[1:]
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    Image.new("RGB", (16, 16), color=(96, 128, 160)).save(inputs / "scene.png")
    outputs = tmp_path / "outputs"
    _replace_path(arguments, "--input-dir", inputs)
    _replace_path(arguments, "--output-dir", outputs)
    monkeypatch.setattr(lux_cli, "_configure_logging", lambda *_args, **_kwargs: None)

    result = CliRunner().invoke(lux_cli.app, [*arguments, "--plan"])

    assert result.exit_code == 0, result.output
    plans = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(plans) == 1
    assert plans[0]["schema"] == "tp.execution.plan.v1"
    assert not outputs.exists(), "Planning must not create the documented execution output"


def test_documented_material_finishing_command_runs_with_named_preset(tmp_path: Path) -> None:
    """Exercise argument routing and preset resolution through the real script."""
    arguments = _documented_command("docs/guides/MATERIAL_PBR_GUIDE.md", "scripts/enhance_image.py")
    preset = arguments[arguments.index("--preset") + 1]
    assert V2EnhancementConfig.from_preset(preset).preset == preset
    source = tmp_path / "scene.png"
    Image.new("RGB", (32, 32), color=(96, 128, 160)).save(source)
    arguments[0] = sys.executable
    arguments[2] = str(source)
    outputs = tmp_path / "finishing"
    _replace_path(arguments, "--output-dir", outputs)
    environment = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))

    result = subprocess.run(
        arguments,
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((outputs / "scene_report.json").read_text(encoding="utf-8"))
    assert report["status"] != "error"
    assert any(path.suffix == ".png" for path in outputs.iterdir())
