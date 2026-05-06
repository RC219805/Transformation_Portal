"""Compatibility tests for the Phase 4C rendering_4k quality extraction."""

from __future__ import annotations

import importlib
import sys

import pytest

pytestmark = [pytest.mark.unit]


PHASE_4C_SYMBOLS = (
    "GPUMemoryManager",
    "QualityAssessor",
)


def test_legacy_rendering_module_reexports_phase_4c_quality_helpers() -> None:
    legacy = importlib.import_module("transformation_portal.pipelines.rendering_4k_pipeline")
    extracted = importlib.import_module("transformation_portal.pipelines.rendering_4k.quality")

    for symbol in PHASE_4C_SYMBOLS:
        assert getattr(legacy, symbol) is getattr(extracted, symbol)


def test_rendering_4k_package_reexports_phase_4c_quality_helpers() -> None:
    package = importlib.import_module("transformation_portal.pipelines.rendering_4k")
    extracted = importlib.import_module("transformation_portal.pipelines.rendering_4k.quality")

    for symbol in PHASE_4C_SYMBOLS:
        assert getattr(package, symbol) is getattr(extracted, symbol)
        assert symbol in package.__all__


def test_rendering_4k_package_defers_quality_module_import(monkeypatch: pytest.MonkeyPatch) -> None:
    package_name = "transformation_portal.pipelines.rendering_4k"
    quality_name = f"{package_name}.quality"

    monkeypatch.delitem(sys.modules, quality_name, raising=False)
    monkeypatch.delitem(sys.modules, package_name, raising=False)

    package = importlib.import_module(package_name)

    assert quality_name not in sys.modules

    quality = importlib.import_module(quality_name)
    assert getattr(package, "QualityAssessor") is quality.QualityAssessor
