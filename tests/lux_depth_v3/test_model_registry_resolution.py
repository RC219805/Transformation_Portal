"""Registry and resolver contract tests for Lux Depth V3."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from transformation_portal.lux_depth_v3.config import DA3Config, DeviceConfig, ModelVariant
from transformation_portal.lux_depth_v3.da3_model_backend import DA3ModelBackend, DA3ModelBackendConfig
from transformation_portal.lux_depth_v3.inference import DA3InferenceEngine, ModelBackend
from transformation_portal.lux_depth_v3.model_registry import (
    ConsumableOutput,
    get_model_spec,
    resolve_legacy_model_variant_key,
    resolve_model_spec,
    visible_cli_model_specs,
)
from transformation_portal.lux_depth_v3.model_resolution import (
    BackendCapabilityError,
    DeprecatedModelSelectorWarning,
    ModelLicenseError,
    ModelRequest,
    UnknownModelError,
    resolve_model_contract,
)

pytestmark = [pytest.mark.unit]


def test_every_public_alias_resolves_to_known_spec() -> None:
    assert resolve_model_spec("da3").key == "da3_research"
    assert resolve_model_spec("da3-research").key == "da3_research"
    assert resolve_model_spec("da3-metric").key == "da3_metric"


@pytest.mark.parametrize(
    ("variant", "expected_key"),
    [
        (ModelVariant.METRIC_LARGE, "da3_research"),
        (ModelVariant.METRIC_BASE, "da3_base"),
        (ModelVariant.METRIC_SMALL, "da3_small"),
    ],
)
def test_legacy_model_variants_warn_and_map_to_registry_keys(variant: ModelVariant, expected_key: str) -> None:
    with pytest.warns(DeprecatedModelSelectorWarning):
        resolved = resolve_model_contract(ModelRequest(model_variant=variant, non_commercial_ok=True))
    assert resolved.canonical_key == expected_key


def test_legacy_model_variants_no_longer_point_to_stale_depth_anything_v3_metric_ids() -> None:
    assert ModelVariant.METRIC_BASE.value.huggingface_id == "depth-anything/DA3-BASE"
    assert ModelVariant.METRIC_SMALL.value.huggingface_id == "depth-anything/DA3-SMALL"


def test_unknown_selector_fails_closed() -> None:
    with pytest.raises(UnknownModelError):
        resolve_model_contract(ModelRequest(model_key="depth-anything/Depth-Anything-V3-Metric-Base-hf"))


def test_da3_requires_non_commercial_ok() -> None:
    with pytest.raises(ModelLicenseError):
        resolve_model_contract(ModelRequest(model_key="da3"))


def test_metric_large_legacy_selector_requires_non_commercial_ok() -> None:
    with pytest.warns(DeprecatedModelSelectorWarning):
        with pytest.raises(ModelLicenseError):
            resolve_model_contract(ModelRequest(model_variant=ModelVariant.METRIC_LARGE))


def test_da3_metric_passes_without_non_commercial_ok() -> None:
    resolved = resolve_model_contract(ModelRequest(model_key="da3-metric"))
    assert resolved.spec.repo_id == "depth-anything/DA3METRIC-LARGE"
    assert resolved.spec.consumable_outputs == frozenset(
        {
            ConsumableOutput.NORMALIZED_RELATIVE_DEPTH,
            ConsumableOutput.DEPTH_METADATA,
        }
    )


def test_metric_base_and_small_use_apache_policy_after_mapping() -> None:
    with pytest.warns(DeprecatedModelSelectorWarning):
        base = resolve_model_contract(ModelRequest(model_variant=ModelVariant.METRIC_BASE))
    with pytest.warns(DeprecatedModelSelectorWarning):
        small = resolve_model_contract(ModelRequest(model_variant=ModelVariant.METRIC_SMALL))
    assert base.spec.license_id == "apache-2.0"
    assert small.spec.license_id == "apache-2.0"


def test_da3_coreml_fails_closed() -> None:
    with pytest.raises(BackendCapabilityError):
        resolve_model_contract(
            ModelRequest(
                model_key="da3",
                non_commercial_ok=True,
                use_coreml_backend=True,
            )
        )


def test_only_published_coreml_artifact_is_coreml_eligible() -> None:
    resolved = resolve_model_contract(
        ModelRequest(
            model_key="coreml_depth_anything_v2_small",
            use_coreml_backend=True,
        )
    )
    assert resolved.spec.repo_id == "apple/coreml-depth-anything-v2-small"
    assert resolved.accelerator_kind.value == "coreml"


def test_coreml_estimator_cache_key_includes_revision(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import transformation_portal.lux_depth_v3.coreml_backend as coreml_backend

    monkeypatch.setattr(coreml_backend, "COREML_AVAILABLE", True)
    monkeypatch.setattr(
        coreml_backend.CoreMLDepthEstimator,
        "_load_or_convert",
        lambda self, force_reconvert=False: object(),
    )

    estimator = coreml_backend.CoreMLDepthEstimator(
        "apple/coreml-depth-anything-v2-small",
        cache_dir=tmp_path,
        revision="a" * 40,
    )

    assert estimator._get_cache_path().name.endswith(f"_{'a' * 40}.mlpackage")


def test_coreml_estimator_download_pins_hub_revision(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import transformation_portal.lux_depth_v3.coreml_backend as coreml_backend

    calls = {}
    downloaded_path = tmp_path / "DepthAnythingV2SmallF16.mlpackage"
    downloaded_path.write_text("stub", encoding="utf-8")

    def fake_download(**kwargs):
        calls.update(kwargs)
        return str(downloaded_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=fake_download))
    monkeypatch.setattr(coreml_backend, "CoreMLDepthModel", lambda model_path: model_path)

    estimator = object.__new__(coreml_backend.CoreMLDepthEstimator)
    estimator.model_id = "apple/coreml-depth-anything-v2-small"
    estimator.cache_dir = tmp_path
    estimator.revision = "b" * 40

    estimator._load_published_coreml_model(estimator._get_cache_path())

    assert calls["revision"] == "b" * 40


def test_da3_inference_coreml_loader_receives_resolved_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    import transformation_portal.lux_depth_v3.coreml_backend as coreml_backend
    import transformation_portal.lux_depth_v3.inference as inference_module

    config = DA3Config(device=DeviceConfig(device="cpu"))
    engine = DA3InferenceEngine(config)
    resolved = resolve_model_contract(
        ModelRequest(
            model_key="coreml_depth_anything_v2_small",
            use_coreml_backend=True,
        )
    )
    captured = {}

    class FakeEstimator:
        def __init__(self, model_id, cache_dir=None, force_reconvert=False, revision=None):
            del cache_dir, force_reconvert
            captured["model_id"] = model_id
            captured["revision"] = revision

    monkeypatch.setattr(inference_module, "COREML_AVAILABLE", True)
    monkeypatch.setattr(inference_module, "_ensure_optional_runtime_imports", lambda: None)
    monkeypatch.setattr(engine, "_resolve_model_contract", lambda use_coreml_backend=None: resolved)
    monkeypatch.setattr(coreml_backend, "CoreMLDepthEstimator", FakeEstimator)

    engine._load_coreml_model()

    assert captured == {
        "model_id": "apple/coreml-depth-anything-v2-small",
        "revision": resolved.revision,
    }


def test_da3_inference_engine_accepts_positional_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        DA3InferenceEngine,
        "_auto_detect_backend",
        lambda self: ModelBackend.PYTORCH_CPU,
    )
    config = DA3Config(model_variant=ModelVariant.METRIC_BASE, device=DeviceConfig(device="cpu"))
    engine = DA3InferenceEngine(config)
    assert engine.config is config


def test_da3_stub_backend_raises_immediately() -> None:
    with pytest.raises(RuntimeError, match="retired"):
        DA3ModelBackend(DA3ModelBackendConfig())


def test_public_cli_model_visibility_is_limited_to_research_and_metric() -> None:
    assert [spec.key for spec in visible_cli_model_specs()] == ["da3_research", "da3_metric"]
    assert resolve_legacy_model_variant_key(ModelVariant.METRIC_BASE) == "da3_base"
    assert get_model_spec("da3_base").exposed_in_cli is False
