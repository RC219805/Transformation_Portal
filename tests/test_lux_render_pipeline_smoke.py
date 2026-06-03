"""Smoke coverage for Lux Render import safety and public failure behavior."""

from __future__ import annotations

import importlib
import sys

import pytest
from typer.testing import CliRunner

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]

PORTAL_MODULE = "transformation_portal"
PIPELINES_PACKAGE = "transformation_portal.pipelines"
PIPELINE_MODULE = "transformation_portal.pipelines.lux_render_pipeline"
TORCH_MODULE = "torch"
DIFFUSERS_MODULE = "diffusers"
CONTROLNET_AUX_MODULE = "controlnet_aux"
REALESRGAN_MODULE = "realesrgan"


def _load_lux_render_pipeline_with_missing_ml(monkeypatch):
    """Import Lux Render after forcing its ML extras unavailable."""

    monkeypatch.setitem(sys.modules, TORCH_MODULE, None)
    monkeypatch.setitem(sys.modules, DIFFUSERS_MODULE, None)
    monkeypatch.setitem(sys.modules, CONTROLNET_AUX_MODULE, None)
    monkeypatch.setitem(sys.modules, REALESRGAN_MODULE, None)

    original_pipeline_module = sys.modules.get(PIPELINE_MODULE)
    portal_module = importlib.import_module(PORTAL_MODULE)
    pipelines_package = importlib.import_module(PIPELINES_PACKAGE)
    original_cached_pipeline = getattr(portal_module, "_lux_render", None)
    had_cached_submodule = hasattr(pipelines_package, "lux_render_pipeline")
    original_cached_submodule = getattr(pipelines_package, "lux_render_pipeline", None)

    monkeypatch.delitem(sys.modules, PIPELINE_MODULE, raising=False)
    portal_module._lux_render = None
    if had_cached_submodule:
        delattr(pipelines_package, "lux_render_pipeline")

    return (
        portal_module,
        pipelines_package,
        had_cached_submodule,
        original_cached_submodule,
        original_pipeline_module,
        original_cached_pipeline,
        portal_module.get_lux_render_pipeline(),
    )


def test_lux_render_pipeline_import_is_graceful_without_ml_extras(monkeypatch, capsys) -> None:
    (
        portal_module,
        pipelines_package,
        had_cached_submodule,
        original_cached_submodule,
        original_pipeline_module,
        original_cached_pipeline,
        pipeline_module,
    ) = _load_lux_render_pipeline_with_missing_ml(monkeypatch)
    runner = CliRunner()

    try:
        assert pipeline_module.__name__ == PIPELINE_MODULE
        assert pipeline_module is portal_module.get_lux_render_pipeline()
        assert hasattr(pipeline_module, "LuxuryRenderPipeline")

        help_result = runner.invoke(pipeline_module.app, ["--help"])
        assert help_result.exit_code == 0
        assert "Batch CLI entry point for the luxury render pipeline." in help_result.output
        assert "Positive prompt" in help_result.output

        monkeypatch.setattr(sys, "argv", ["lux_render", "--help"])
        with pytest.raises(SystemExit) as console_exit:
            pipeline_module.main()
        assert console_exit.value.code == 0
        console_output = capsys.readouterr().out
        assert "Batch CLI entry point for the luxury render pipeline." in console_output
        assert "--input-glob" in console_output

        run_result = runner.invoke(
            pipeline_module.app,
            [
                "--input-glob",
                "./missing/*.png",
                "--prompt",
                "luxury interior",
            ],
        )
        assert run_result.exit_code == 1
        combined_output = (run_result.stdout or "") + (run_result.stderr or "")
        assert "lux_render requires optional ML dependencies" in combined_output
        assert "controlnet-aux" in combined_output
        assert "diffusers" in combined_output
        assert "make install-ml-core" in combined_output
        assert "requirements/README.md" in combined_output
        assert "make install-ml`" not in combined_output

        with pytest.raises(RuntimeError) as excinfo:
            pipeline_module.RealESRGANer()
        error_message = str(excinfo.value)
        assert "intentionally unsupported" in error_message
        assert "Pillow Lanczos fallback" in error_message
        assert "pip install realesrgan" not in error_message
    finally:
        if original_pipeline_module is not None:
            sys.modules[PIPELINE_MODULE] = original_pipeline_module
        else:
            sys.modules.pop(PIPELINE_MODULE, None)
        if had_cached_submodule:
            setattr(pipelines_package, "lux_render_pipeline", original_cached_submodule)
        else:
            pipelines_package.__dict__.pop("lux_render_pipeline", None)
        portal_module._lux_render = original_cached_pipeline
