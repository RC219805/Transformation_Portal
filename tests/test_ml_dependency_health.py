"""Unit tests for ML dependency compatibility guards."""

from __future__ import annotations

import importlib
import sys
import types

import pytest

from transformation_portal.core.ml_dependency_health import (
    detect_transformers_torch_runtime_issue,
    detect_transformers_torch_version_issue,
)

pytestmark = pytest.mark.unit


def _import_with_fake_ml_stack(monkeypatch: pytest.MonkeyPatch, module_name: str):
    """Import a module under a deterministic fake torch/transformers stack."""
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "2.2.2"
    fake_torch.float16 = "float16"  # type: ignore[attr-defined]
    fake_torch.Tensor = type("FakeTensor", (), {})  # type: ignore[attr-defined]
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__version__ = "5.3.0"
    fake_transformers.__path__ = []  # type: ignore[attr-defined]
    fake_transformers.pipeline = lambda **_kwargs: None  # type: ignore[attr-defined]
    fake_transformers.AutoImageProcessor = types.SimpleNamespace(from_pretrained=lambda **_kwargs: None)
    fake_transformers.AutoModelForDepthEstimation = types.SimpleNamespace(from_pretrained=lambda **_kwargs: None)

    fake_utils = types.ModuleType("transformers.utils")
    fake_utils.is_torch_available = lambda: False  # type: ignore[attr-defined]
    fake_pipelines = types.ModuleType("transformers.pipelines")
    fake_pipelines.__path__ = []  # type: ignore[attr-defined]
    fake_depth_estimation = types.ModuleType("transformers.pipelines.depth_estimation")
    fake_depth_estimation.DepthEstimationPipeline = object

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "transformers.pipelines", fake_pipelines)
    monkeypatch.setitem(
        sys.modules,
        "transformers.pipelines.depth_estimation",
        fake_depth_estimation,
    )

    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_detect_transformers_torch_runtime_issue_reports_disabled_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """The helper should explain when transformers disables torch at runtime."""
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__version__ = "5.3.0"
    fake_transformers.__path__ = []  # type: ignore[attr-defined]
    fake_utils = types.ModuleType("transformers.utils")
    fake_utils.is_torch_available = lambda: False  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils", fake_utils)

    fake_torch = types.SimpleNamespace(__version__="2.2.2")

    message = detect_transformers_torch_runtime_issue(fake_torch, fake_transformers)

    assert message is not None
    assert "disabled its PyTorch backend" in message
    assert "torch 2.2.2" in message
    assert "transformers 5.3.0" in message


def test_detect_transformers_torch_version_issue_allows_repo_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The supported torch 2.12.x + transformers 5.x baseline must not be rejected."""
    monkeypatch.setattr(
        "transformation_portal.core.ml_dependency_health._installed_version",
        lambda _distribution: "2.4.3",
    )
    monkeypatch.setattr(
        "transformation_portal.core.ml_dependency_health._is_darwin_x86_64_runtime",
        lambda: False,
    )

    message = detect_transformers_torch_version_issue("2.13.0", "5.5.0")

    assert message is None


def test_detect_transformers_torch_version_issue_rejects_retired_old_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retired old torch/transformers baselines are outside the supported runtime envelope."""
    monkeypatch.setattr(
        "transformation_portal.core.ml_dependency_health._installed_version",
        lambda _distribution: "1.26.4",
    )
    monkeypatch.setattr(
        "transformation_portal.core.ml_dependency_health._is_darwin_x86_64_runtime",
        lambda: False,
    )

    message = detect_transformers_torch_version_issue("2.2.2", "4.57.6")

    assert message is not None
    assert "supported security baseline 2.13.0" in message
    assert "supported security baseline 5.5.0" in message


def test_detect_transformers_torch_version_issue_rejects_transformers_53_with_old_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transformers 5.3+ with torch<2.4 remains outside the supported runtime envelope."""
    monkeypatch.setattr(
        "transformation_portal.core.ml_dependency_health._installed_version",
        lambda _distribution: "1.26.4",
    )
    monkeypatch.setattr(
        "transformation_portal.core.ml_dependency_health._is_darwin_x86_64_runtime",
        lambda: False,
    )

    message = detect_transformers_torch_version_issue("2.2.2", "5.3.0")

    assert message is not None
    assert "supported runtime envelope" in message
    assert "torch 2.2.2" in message
    assert "transformers 5.3.0" in message


def test_detect_transformers_torch_version_issue_rejects_darwin_intel_numpy2_torch22(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Darwin x86_64 keeps the NumPy<2 guard even when transformers stays on 4.57.x."""
    monkeypatch.setattr(
        "transformation_portal.core.ml_dependency_health._installed_version",
        lambda _distribution: "2.4.3",
    )
    monkeypatch.setattr(
        "transformation_portal.core.ml_dependency_health._is_darwin_x86_64_runtime",
        lambda: True,
    )

    message = detect_transformers_torch_version_issue("2.2.2", "4.57.6")

    assert message is not None
    assert "numpy 2.4.3" in message
    assert "torch 2.2.2" in message


def test_da2_backend_ensure_available_rejects_transformers_torch_incompatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DA2 should reject a transformers/torch combination that cannot load models."""
    from transformation_portal.depth.backends.da2 import DA2Backend

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__version__ = "5.3.0"
    fake_transformers.__path__ = []  # type: ignore[attr-defined]
    fake_utils = types.ModuleType("transformers.utils")
    fake_utils.is_torch_available = lambda: False  # type: ignore[attr-defined]
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "2.2.2"
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(ImportError, match="supported runtime envelope"):
        DA2Backend().ensure_available()


def test_da3_inference_da3_custom_path_ignores_transformers_pipeline_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DA3 custom-model loading should remain available even when pipeline fallback is blocked."""
    da3_inference = _import_with_fake_ml_stack(
        monkeypatch,
        "transformation_portal.lux_depth_v3.inference",
    )
    from transformation_portal.lux_depth_v3.config import DA3Config, DeviceConfig
    from transformation_portal.lux_depth_v3.config import ModelVariant as DA3ModelVariant

    captured: dict[str, str] = {}

    monkeypatch.setattr(da3_inference, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(da3_inference, "TRANSFORMERS_AVAILABLE", False)
    monkeypatch.setattr(
        da3_inference,
        "TRANSFORMERS_TORCH_BACKEND_ISSUE",
        "transformers disabled torch for this environment",
    )
    monkeypatch.setattr(
        da3_inference.DA3InferenceEngine,
        "_auto_detect_backend",
        lambda self: da3_inference.ModelBackend.PYTORCH_CPU,
    )
    monkeypatch.setattr(
        da3_inference.DA3InferenceEngine,
        "_load_da3_model",
        lambda self, model_id: captured.setdefault("model_id", model_id),
    )

    engine = da3_inference.DA3InferenceEngine(
        DA3Config(
            model_variant=DA3ModelVariant.METRIC_LARGE,
            non_commercial_ok=True,
            device=DeviceConfig(device="cpu", use_fp16=False),
        )
    )

    engine._load_pytorch_model()

    assert captured["model_id"] == DA3ModelVariant.METRIC_LARGE.value.huggingface_id


def test_depth_anything_v2_manual_fallback_uses_slow_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Manual V2 fallback should force the slow image processor for torch-free safety."""
    da2_model_module = _import_with_fake_ml_stack(
        monkeypatch,
        "transformation_portal.depth.models.depth_anything_v2",
    )

    captured: dict[str, object] = {}

    class _FakeManualModel:
        def to(self, _device: str) -> "_FakeManualModel":
            return self

    def _fake_pipeline(**_kwargs):
        raise RuntimeError("pipeline unavailable")

    def _fake_processor_from_pretrained(model_id: str, revision: str | None = None, use_fast: bool = True):
        captured["model_id"] = model_id
        captured["revision"] = revision
        captured["use_fast"] = use_fast
        return object()

    def _fake_model_from_pretrained(model_id: str, revision: str | None = None):
        captured["model_model_id"] = model_id
        captured["model_revision"] = revision
        return _FakeManualModel()

    monkeypatch.setattr(da2_model_module, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(da2_model_module, "TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(da2_model_module, "TRANSFORMERS_TORCH_BACKEND_ISSUE", None)
    monkeypatch.setattr(da2_model_module, "pipeline", _fake_pipeline)
    monkeypatch.setattr(
        da2_model_module,
        "AutoImageProcessor",
        types.SimpleNamespace(from_pretrained=_fake_processor_from_pretrained),
    )
    monkeypatch.setattr(
        da2_model_module,
        "AutoModelForDepthEstimation",
        types.SimpleNamespace(from_pretrained=_fake_model_from_pretrained),
    )

    model = da2_model_module.DepthAnythingV2Model.__new__(da2_model_module.DepthAnythingV2Model)
    model.variant = da2_model_module.ModelVariant.SMALL
    model.backend = da2_model_module.ModelBackend.PYTORCH_CPU
    model.device = "cpu"
    model.model_revision = None
    model.model = None
    model.processor = None

    model._load_pytorch_model()

    assert captured["use_fast"] is False
    assert captured["model_id"] == da2_model_module.ModelVariant.SMALL.value
