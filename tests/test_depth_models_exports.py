"""Regression tests for the depth.models lazy export surface."""

from __future__ import annotations

import types

import pytest

pytestmark = pytest.mark.unit


def test_coreml_depth_estimator_resolves_to_backend_estimator(monkeypatch: pytest.MonkeyPatch) -> None:
    """The estimator export should still resolve to the actual CoreML backend implementation."""
    from transformation_portal.depth import models

    fake_backend_estimator = type("FakeCoreMLDepthEstimator", (), {})
    fake_module = types.SimpleNamespace(CoreMLDepthEstimator=fake_backend_estimator)

    monkeypatch.setattr(models, "import_module", lambda module_name, package=None: fake_module)
    models.__dict__.pop("CoreMLDepthEstimator", None)

    assert models.CoreMLDepthEstimator is fake_backend_estimator


def test_coreml_exporter_is_explicit_compatibility_shim() -> None:
    """CoreMLExporter should not silently alias to the estimator class."""
    from transformation_portal.depth import models

    fake_backend_estimator = type("FakeCoreMLDepthEstimator", (), {})
    assert models.CoreMLExporter is not fake_backend_estimator

    with pytest.raises(ModuleNotFoundError, match="CoreMLExporter is not bundled"):
        models.CoreMLExporter()
