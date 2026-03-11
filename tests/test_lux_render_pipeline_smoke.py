"""Smoke coverage for Lux Render import safety and public failure behavior."""

from __future__ import annotations

import importlib
import sys

from typer.testing import CliRunner

PORTAL_MODULE = "transformation_portal"
PIPELINE_MODULE = "transformation_portal.pipelines.lux_render_pipeline"
DIFFUSERS_MODULE = "diffusers"
CONTROLNET_AUX_MODULE = "controlnet_aux"


def _load_lux_render_pipeline_with_missing_ml(monkeypatch):
    """Import Lux Render after forcing its ML extras unavailable."""

    monkeypatch.setitem(sys.modules, DIFFUSERS_MODULE, None)
    monkeypatch.setitem(sys.modules, CONTROLNET_AUX_MODULE, None)

    original_pipeline_module = sys.modules.get(PIPELINE_MODULE)
    portal_module = importlib.import_module(PORTAL_MODULE)
    original_cached_pipeline = getattr(portal_module, "_lux_render", None)

    monkeypatch.delitem(sys.modules, PIPELINE_MODULE, raising=False)
    portal_module._lux_render = None

    return portal_module, original_pipeline_module, original_cached_pipeline, portal_module.get_lux_render_pipeline()


def test_lux_render_pipeline_import_is_graceful_without_ml_extras(monkeypatch) -> None:
    portal_module, original_pipeline_module, original_cached_pipeline, pipeline_module = (
        _load_lux_render_pipeline_with_missing_ml(monkeypatch)
    )
    runner = CliRunner()

    try:
        assert pipeline_module.__name__ == PIPELINE_MODULE
        assert pipeline_module is portal_module.get_lux_render_pipeline()
        assert hasattr(pipeline_module, "LuxuryRenderPipeline")

        help_result = runner.invoke(pipeline_module.app, ["--help"])
        assert help_result.exit_code == 0
        assert "--input-glob" in help_result.output
        assert "--base-model" in help_result.output

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
        assert "lux_render requires optional ML dependencies" in run_result.output
        assert "controlnet-aux, diffusers" in run_result.output
    finally:
        if original_pipeline_module is not None:
            sys.modules[PIPELINE_MODULE] = original_pipeline_module
        else:
            sys.modules.pop(PIPELINE_MODULE, None)
        portal_module._lux_render = original_cached_pipeline
