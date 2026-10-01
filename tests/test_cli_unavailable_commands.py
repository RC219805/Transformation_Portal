"""Unavailable compatibility handlers must never report successful execution."""

from __future__ import annotations

import builtins
import importlib
from pathlib import Path

import click
import pytest
from typer.testing import CliRunner

from transformation_portal import cli

pytestmark = [pytest.mark.unit, pytest.mark.regression]

COMMANDS = [
    ("render_app", "lux", "file", ["--prompt", "example", "--strength", "0.5", "--upscale"]),
    ("render_app", "depth", "file", ["--preset", "interior"]),
    ("process_app", "material", "file", ["--strength", "0.5", "--surfaces", "wood,metal"]),
    ("process_app", "video", "file", ["--preset", "signature_estate", "--lut-strength", "0.5"]),
    ("process_app", "tif", "directory", ["--preset", "signature", "--recursive"]),
    ("analyze_app", "philosophy", "analysis", []),
    ("analyze_app", "decay", "analysis", ["--threshold", "30"]),
    ("analyze_app", "workflow", "analysis", []),
]


@pytest.fixture(autouse=True)
def forbid_optional_implementation_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    forbidden = (
        "transformation_portal.pipelines",
        "transformation_portal.processors",
        "transformation_portal.analyzers",
        "luxury_tiff_batch_processor",
        "torch",
        "transformers",
        "cv2",
        "tifffile",
    )
    original_import = builtins.__import__
    original_import_module = importlib.import_module

    def check_name(name):
        assert not any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden), name

    def guarded_import(name, *args, **kwargs):
        check_name(name)
        return original_import(name, *args, **kwargs)

    def guarded_import_module(name, *args, **kwargs):
        check_name(name)
        return original_import_module(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(importlib, "import_module", guarded_import_module)


@pytest.mark.parametrize("app_name,command,input_kind,extra_args", COMMANDS)
def test_unavailable_command_exits_without_work(app_name, command, input_kind, extra_args, tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    source.write_bytes(b"unchanged source fixture")
    output = tmp_path / "output"
    if input_kind == "analysis":
        args = ["--path", str(tmp_path)]
        if command == "philosophy":
            args += ["--output", str(output)]
    else:
        args = ["--input", str(source if input_kind == "file" else tmp_path), "--output", str(output)]

    result = CliRunner().invoke(getattr(cli, app_name), [command, *args, *extra_args])

    assert result.exit_code == 1, result.output
    assert "not implemented" in result.output
    assert "no work was performed" in result.output
    assert "loaded successfully" not in result.output
    assert "Install with:" not in result.output
    assert not output.exists()
    assert list(tmp_path.iterdir()) == [source]
    assert source.read_bytes() == b"unchanged source fixture"


@pytest.mark.parametrize("app_name,command,input_kind,extra_args", COMMANDS)
def test_unavailable_command_help_remains_available(app_name, command, input_kind, extra_args) -> None:
    result = CliRunner().invoke(getattr(cli, app_name), [command, "--help"], color=False)
    # Rich can force terminal styling from GITHUB_ACTIONS despite color=False.
    help_text = click.unstyle(result.output)

    assert result.exit_code == 0, result.output
    assert "Unavailable compatibility command" in help_text
    assert ("--path" if input_kind == "analysis" else "--input") in help_text
    for option in extra_args:
        if option.startswith("--"):
            assert option in help_text
