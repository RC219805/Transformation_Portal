"""CPU/core importability tests for VLM modules without optional ML runtime."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.unit]


def _clear_vlm_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    for module_name in list(sys.modules):
        if module_name.startswith("transformation_portal.vlm"):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
    parent = sys.modules.get("transformation_portal")
    if isinstance(parent, ModuleType):
        monkeypatch.delattr(parent, "vlm", raising=False)


def test_vlm_package_imports_without_torch(monkeypatch: pytest.MonkeyPatch):
    _clear_vlm_modules(monkeypatch)
    monkeypatch.setitem(sys.modules, "torch", None)

    vlm = importlib.import_module("transformation_portal.vlm")

    assert "transformation_portal.vlm.llava" not in sys.modules
    assert vlm.__all__ == ["LLaVAProcessor", "SceneAnalyzer", "QualityValidator"]


def test_vlm_exports_remain_available_without_torch(monkeypatch: pytest.MonkeyPatch):
    _clear_vlm_modules(monkeypatch)
    monkeypatch.setitem(sys.modules, "torch", None)

    from transformation_portal.vlm import LLaVAProcessor, QualityValidator, SceneAnalyzer

    fake_processor = MagicMock()
    quality_validator = QualityValidator(llava_processor=fake_processor)
    scene_analyzer = SceneAnalyzer(llava_processor=fake_processor)

    assert quality_validator.processor is fake_processor
    assert scene_analyzer.processor is fake_processor
    llava_module = sys.modules["transformation_portal.vlm.llava"]
    with pytest.raises(ImportError) as exc_info:
        LLaVAProcessor()
    assert llava_module.LLAVA_INSTALL_GUIDANCE in str(exc_info.value)


def test_quality_and_scene_constructors_lazy_import_llava(monkeypatch: pytest.MonkeyPatch):
    _clear_vlm_modules(monkeypatch)
    monkeypatch.setitem(sys.modules, "torch", None)

    quality_module = importlib.import_module("transformation_portal.vlm.quality_validator")
    scene_module = importlib.import_module("transformation_portal.vlm.scene_analyzer")

    assert "transformation_portal.vlm.llava" not in sys.modules
    with pytest.raises(ImportError, match="pip install transformers"):
        quality_module.QualityValidator()
    with pytest.raises(ImportError, match="pip install transformers"):
        scene_module.SceneAnalyzer()
