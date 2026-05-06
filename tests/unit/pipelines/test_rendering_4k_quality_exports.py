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

PACKAGE_NAME = "transformation_portal.pipelines.rendering_4k"
QUALITY_NAME = f"{PACKAGE_NAME}.quality"
LEGACY_NAME = "transformation_portal.pipelines.rendering_4k_pipeline"


def _clear_cached_package_quality_exports() -> None:
    package = sys.modules.get(PACKAGE_NAME)
    if package is None:
        return

    for symbol in (*PHASE_4C_SYMBOLS, "quality"):
        package.__dict__.pop(symbol, None)


def _reload_quality_and_legacy_modules() -> tuple[object, object]:
    extracted = importlib.reload(importlib.import_module(QUALITY_NAME))
    legacy = importlib.reload(importlib.import_module(LEGACY_NAME))
    _clear_cached_package_quality_exports()
    return legacy, extracted


def test_legacy_rendering_module_reexports_phase_4c_quality_helpers() -> None:
    legacy, extracted = _reload_quality_and_legacy_modules()

    for symbol in PHASE_4C_SYMBOLS:
        assert getattr(legacy, symbol) is getattr(extracted, symbol)


def test_rendering_4k_package_reexports_phase_4c_quality_helpers() -> None:
    _, extracted = _reload_quality_and_legacy_modules()
    package = importlib.import_module(PACKAGE_NAME)

    for symbol in PHASE_4C_SYMBOLS:
        assert getattr(package, symbol) is getattr(extracted, symbol)
        assert symbol in package.__all__


def test_rendering_4k_package_defers_quality_module_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, QUALITY_NAME, raising=False)
    monkeypatch.delitem(sys.modules, PACKAGE_NAME, raising=False)

    package = importlib.import_module(PACKAGE_NAME)

    assert QUALITY_NAME not in sys.modules

    quality = importlib.import_module(QUALITY_NAME)
    assert getattr(package, "QualityAssessor") is quality.QualityAssessor
