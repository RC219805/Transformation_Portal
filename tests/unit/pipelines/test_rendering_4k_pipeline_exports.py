"""Compatibility tests for the Phase 4D rendering_4k pipeline extraction."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

PACKAGE_NAME = "transformation_portal.pipelines.rendering_4k"
PIPELINE_NAME = f"{PACKAGE_NAME}.pipeline"
LEGACY_NAME = "transformation_portal.pipelines.rendering_4k_pipeline"


def test_legacy_rendering_module_reexports_phase_4d_pipeline() -> None:
    legacy = importlib.import_module(LEGACY_NAME)
    extracted = importlib.import_module(PIPELINE_NAME)

    assert legacy.Rendering4KPipeline is extracted.Rendering4KPipeline
    assert legacy.main is extracted.main
    assert legacy._json_default is extracted._json_default
    assert legacy.logger is extracted.logger


def test_rendering_4k_package_lazily_reexports_phase_4d_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, PIPELINE_NAME, raising=False)
    monkeypatch.delitem(sys.modules, PACKAGE_NAME, raising=False)

    package = importlib.import_module(PACKAGE_NAME)

    assert PIPELINE_NAME not in sys.modules
    assert "Rendering4KPipeline" in package.__all__

    extracted = importlib.import_module(PIPELINE_NAME)
    assert getattr(package, "Rendering4KPipeline") is extracted.Rendering4KPipeline


def test_legacy_rendering_module_cli_dry_run_still_reaches_main(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    input_path = tmp_path / "input.png"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            LEGACY_NAME,
            "-i",
            str(input_path),
            "--preset",
            "preview",
            "--dry-run",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"name": "preview"' in f"{result.stdout}\n{result.stderr}"


def test_rendering_4k_presets_are_copied_before_pipeline_mutation() -> None:
    extracted = importlib.import_module(PIPELINE_NAME)
    preset = extracted.Rendering4KPipeline.PRESETS["preview"]
    original_depth_enabled = preset.depth.enabled

    first = extracted.Rendering4KPipeline.from_preset("preview")
    second = extracted.Rendering4KPipeline.from_preset("preview")

    assert first.config is not preset
    assert second.config is not preset
    assert first.config is not second.config
    assert first.config.depth is not preset.depth
    assert first.config.material_response.surface_types is not preset.material_response.surface_types

    first.config.depth.enabled = not original_depth_enabled
    first.config.material_response.surface_types.append("review-only-surface")

    assert preset.depth.enabled is original_depth_enabled
    assert "review-only-surface" not in preset.material_response.surface_types
    assert "review-only-surface" not in second.config.material_response.surface_types


def test_rendering_4k_cache_key_uses_buffer_sample_without_full_tobytes() -> None:
    import numpy as np

    extracted = importlib.import_module(PIPELINE_NAME)

    class NoToBytesArray(np.ndarray):
        def tobytes(self, *args: object, **kwargs: object) -> bytes:
            raise AssertionError("cache key must not call ndarray.tobytes()")

    image = np.arange(8192, dtype=np.uint8).reshape(64, 128).view(NoToBytesArray)

    key = extracted.Rendering4KPipeline._compute_cache_key(object(), image)

    assert len(key) == 64
    assert key == extracted.Rendering4KPipeline._compute_cache_key(object(), image)


def test_rendering_4k_cache_key_includes_shape_and_dtype() -> None:
    import numpy as np

    extracted = importlib.import_module(PIPELINE_NAME)
    raw = np.arange(4096, dtype=np.uint8)

    flat_key = extracted.Rendering4KPipeline._compute_cache_key(object(), raw)
    shaped_key = extracted.Rendering4KPipeline._compute_cache_key(object(), raw.reshape(64, 64))
    float_key = extracted.Rendering4KPipeline._compute_cache_key(object(), raw.view(np.float32))

    assert shaped_key != flat_key
    assert float_key != flat_key
