"""Machine-readable contract for the 11 maintained Lux CLI workflows."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import stat
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from typer.testing import CliRunner

from transformation_portal.core.execution_plan import parse_execution_plan_json
from transformation_portal.lux_depth_v3 import __main__ as cli_module

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "tests/fixtures/lux_depth_v3/documented_workflows.v1.json"
MAINTAINED_SECTIONS = {
    "Apache APEX Mode",
    "Research-Only APEX+ Variants",
    "Example Workflows",
}


@pytest.fixture(autouse=True)
def _contain_cli_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep plan-only CLI checks from replacing process-wide log handlers."""

    monkeypatch.setattr(cli_module, "_configure_logging", lambda *_args, **_kwargs: None)


def _anchor(heading: str) -> str:
    normalized = re.sub(r"[^\w\- ]", "", heading.casefold())
    return "#" + re.sub(r"\s+", "-", normalized.strip())


def _maintained_documented_commands(document: str) -> dict[str, list[str]]:
    """Extract the sole Lux command from every maintained workflow heading."""

    lines = document.splitlines()
    current_section: str | None = None
    commands: dict[str, list[str]] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("## "):
            current_section = line[3:].strip()
            index += 1
            continue
        if not line.startswith("### ") or current_section not in MAINTAINED_SECTIONS:
            index += 1
            continue

        heading = line[4:].strip()
        cursor = index + 1
        workflow_commands: list[list[str]] = []
        while cursor < len(lines) and not lines[cursor].startswith("##"):
            if lines[cursor] != "```bash":
                cursor += 1
                continue
            cursor += 1
            command_lines: list[str] = []
            while cursor < len(lines) and lines[cursor] != "```":
                command_lines.append(lines[cursor])
                cursor += 1
            command = " ".join(item.strip().removesuffix("\\").rstrip() for item in command_lines)
            argv = shlex.split(command)
            if argv and argv[0] == "lux-depth-v3":
                workflow_commands.append(argv[1:])
            cursor += 1

        anchor = _anchor(heading)
        if len(workflow_commands) != 1:
            raise AssertionError(
                f"Maintained workflow {anchor} must contain exactly one fenced lux-depth-v3 command; "
                f"found {len(workflow_commands)}"
            )
        if anchor in commands:
            raise AssertionError(f"Duplicate maintained workflow anchor: {anchor}")
        commands[anchor] = workflow_commands[0]
        index = cursor
    return commands


def _file_tree_snapshot(root: Path) -> dict[str, tuple[str, int, int, str]]:
    """Capture every entry so plan-mode filesystem mutations cannot hide."""

    snapshot: dict[str, tuple[str, int, int, str]] = {}
    for path in sorted(root.rglob("*")):
        relative_path = path.relative_to(root).as_posix()
        entry_stat = path.lstat()
        if stat.S_ISLNK(entry_stat.st_mode):
            snapshot[relative_path] = ("symlink", entry_stat.st_mode, 0, os.readlink(path))
            continue
        if stat.S_ISDIR(entry_stat.st_mode):
            snapshot[relative_path] = ("directory", entry_stat.st_mode, 0, "")
            continue
        if stat.S_ISREG(entry_stat.st_mode):
            data = path.read_bytes()
            snapshot[relative_path] = (
                "file",
                entry_stat.st_mode,
                len(data),
                hashlib.sha256(data).hexdigest(),
            )
            continue
        snapshot[relative_path] = ("other", entry_stat.st_mode, entry_stat.st_size, "")
    return snapshot


def _load_fixture() -> dict[str, Any]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert payload["schema"] == "tp.lux.documented_workflows.v1"
    assert payload["source_document"] == "docs/cli/LUX_DEPTH_V3_CLI_GUIDE.md"
    return payload


def _assert_complete_output_determinism(first: bytes, second: bytes) -> None:
    if first != second:
        raise AssertionError("Complete CLI output bytes are not deterministic")


def _canonical_plan_bytes(output: bytes) -> bytes:
    lines = [line for line in output.splitlines() if line.startswith(b"{")]
    assert len(lines) == 1, output.decode("utf-8", errors="replace")
    canonical_bytes = lines[0]
    parsed = parse_execution_plan_json(canonical_bytes)
    assert parsed.to_canonical_json().encode("utf-8") == canonical_bytes
    return canonical_bytes


def test_documented_workflow_fixture_exactly_matches_maintained_commands() -> None:
    payload = _load_fixture()
    workflows = payload["workflows"]
    assert len(workflows) == 11
    assert len({item["id"] for item in workflows}) == 11
    assert len({item["source_anchor"] for item in workflows}) == 11

    document_path = REPO_ROOT / payload["source_document"]
    documented = _maintained_documented_commands(document_path.read_text(encoding="utf-8"))
    fixture_by_anchor = {item["source_anchor"]: item["argv"] for item in workflows}
    assert documented == fixture_by_anchor


@pytest.mark.parametrize("workflow", _load_fixture()["workflows"], ids=lambda item: item["id"])
def test_documented_workflow_plan_contract(
    workflow: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "input_images"
    input_dir.mkdir()
    Image.new("RGB", (8, 8), color=(127, 127, 127)).save(input_dir / "sample.png")

    cache_root = tmp_path / "runtime-cache"
    cache_root.mkdir()
    for variable, relative in (
        ("HOME", "home"),
        ("XDG_CACHE_HOME", "xdg"),
        ("HF_HOME", "huggingface"),
        ("TRANSFORMERS_CACHE", "transformers"),
        ("TORCH_HOME", "torch"),
    ):
        location = cache_root / relative
        location.mkdir()
        monkeypatch.setenv(variable, str(location))

    def forbid_runtime_activity(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("--plan reached a model, inference, or execution seam")

    from transformation_portal.lux_depth_v3 import inference as inference_module
    from transformation_portal.lux_depth_v3 import materials_v3 as materials_module
    from transformation_portal.lux_depth_v3 import reconstruction_runner, v2_runner

    monkeypatch.setattr(cli_module.EnhanceOrchestrator, "from_prepared", forbid_runtime_activity)
    monkeypatch.setattr(cli_module.EnhanceOrchestrator, "enhance_batch", forbid_runtime_activity)
    monkeypatch.setattr(inference_module.DA3InferenceEngine, "_load_model", forbid_runtime_activity)
    monkeypatch.setattr(inference_module.DA3InferenceEngine, "infer", forbid_runtime_activity)
    monkeypatch.setattr(v2_runner.V2Runner, "run", forbid_runtime_activity)
    monkeypatch.setattr(materials_module.MaterialsV3Engine, "process", forbid_runtime_activity)
    monkeypatch.setattr(reconstruction_runner, "run_scene_reconstruction", forbid_runtime_activity)

    before = _file_tree_snapshot(tmp_path)
    runner = CliRunner()
    first = runner.invoke(cli_module.app, [*workflow["argv"], "--plan"])
    second = runner.invoke(cli_module.app, [*workflow["argv"], "--plan"])

    assert workflow["expected_plan_status"] == "success"
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    _assert_complete_output_determinism(first.output_bytes, second.output_bytes)
    first_bytes = _canonical_plan_bytes(first.output_bytes)
    plan = json.loads(first_bytes)

    assert plan["schema"] == "tp.execution.plan.v1"
    assert plan["planned_backend"] == workflow["expected_planned_backend"]
    resolved_model = plan["resolved_model"]
    resolved_model_key = None if resolved_model is None else resolved_model["canonical_key"]
    assert resolved_model_key == workflow["expected_resolved_model"]
    assert [node["stage_registry_id"] for node in plan["nodes"]] == workflow["expected_stage_registry_ids"]
    assert plan["requested_outputs"] == workflow["expected_requested_artifacts"]
    assert plan["warnings"] == workflow["expected_warnings"]
    assert "executed_backend" not in plan

    assert _file_tree_snapshot(tmp_path) == before


def test_maintained_heading_rejects_a_second_lux_command() -> None:
    document = """\
## Example Workflows
### Workflow
```bash
lux-depth-v3 --input-dir in --output-dir out
```
```bash
lux-depth-v3 --input-dir in --output-dir escaped
```
"""

    with pytest.raises(AssertionError, match="exactly one"):
        _maintained_documented_commands(document)


def test_file_tree_snapshot_detects_directory_only_writes(tmp_path: Path) -> None:
    before = _file_tree_snapshot(tmp_path)

    (tmp_path / "forbidden-output-root").mkdir()

    assert _file_tree_snapshot(tmp_path) != before


@pytest.mark.parametrize(
    "drifted_output",
    [
        b'INFO: planning\n{"plan":true}\nINFO: extra diagnostic\n',
        b'INFO: planning\n {"plan":true}\n',
        b'INFO: planning\n{"plan":true} \n',
        b'INFO: planning\n{"plan":true}\n\n',
    ],
)
def test_complete_output_determinism_detects_non_plan_and_whitespace_drift(
    drifted_output: bytes,
) -> None:
    original_output = b'INFO: planning\n{"plan":true}\n'

    with pytest.raises(AssertionError, match="Complete CLI output bytes"):
        _assert_complete_output_determinism(original_output, drifted_output)
