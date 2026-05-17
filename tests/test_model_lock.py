"""Tests for model lock manifest resolution and strict enforcement."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import yaml

# Pytest markers
pytestmark = [
    pytest.mark.unit,
    pytest.mark.security,
]

from transformation_portal.core.security.model_lock import (
    ModelLockError,
    is_model_lock_strict_enabled,
    is_pinned_revision,
    load_model_lock_manifest,
    model_lock_manifest_path,
    resolve_model_lock_revision,
)
from transformation_portal.depth.models.depth_anything_v2 import DepthAnythingV2Model, ModelBackend, ModelVariant


def _write_manifest(path: Path, repositories: dict[str, dict[str, str]]) -> None:
    payload = {
        "version": 1,
        "updated_at": "2026-02-28",
        "repositories": repositories,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def test_resolve_model_lock_revision_uses_manifest_in_non_strict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = tmp_path / "model_lock.yaml"
    pinned = "a" * 40
    _write_manifest(
        manifest_path,
        {"depth-anything/Depth-Anything-V2-Small-hf": {"revision": pinned}},
    )
    monkeypatch.setenv("TP_MODEL_LOCK_MANIFEST", str(manifest_path))
    monkeypatch.delenv("TP_STRICT_MODEL_LOCK", raising=False)

    assert load_model_lock_manifest()["version"] == 1
    assert resolve_model_lock_revision("depth-anything/Depth-Anything-V2-Small-hf", None) == pinned


def test_resolve_model_lock_revision_strict_rejects_unpinned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = tmp_path / "model_lock.yaml"
    _write_manifest(
        manifest_path,
        {"depth-anything/Depth-Anything-V2-Small-hf": {"revision": "NEEDS_VERIFICATION_DEPTH_ANYTHING_V2_SMALL"}},
    )
    monkeypatch.setenv("TP_MODEL_LOCK_MANIFEST", str(manifest_path))
    monkeypatch.setenv("TP_STRICT_MODEL_LOCK", "1")

    with pytest.raises(ModelLockError):
        resolve_model_lock_revision("depth-anything/Depth-Anything-V2-Small-hf", None)


def test_resolve_model_lock_revision_strict_allows_explicit_pinned_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "model_lock.yaml"
    _write_manifest(
        manifest_path,
        {"depth-anything/Depth-Anything-V2-Small-hf": {"revision": "NEEDS_VERIFICATION_DEPTH_ANYTHING_V2_SMALL"}},
    )
    monkeypatch.setenv("TP_MODEL_LOCK_MANIFEST", str(manifest_path))
    monkeypatch.setenv("TP_STRICT_MODEL_LOCK", "1")

    explicit = "b" * 40
    assert resolve_model_lock_revision("depth-anything/Depth-Anything-V2-Small-hf", explicit) == explicit


def test_resolve_model_lock_revision_strict_detects_manifest_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = tmp_path / "model_lock.yaml"
    _write_manifest(
        manifest_path,
        {"depth-anything/Depth-Anything-V2-Small-hf": {"revision": "c" * 40}},
    )
    monkeypatch.setenv("TP_MODEL_LOCK_MANIFEST", str(manifest_path))
    monkeypatch.setenv("TP_STRICT_MODEL_LOCK", "1")

    with pytest.raises(ModelLockError):
        resolve_model_lock_revision("depth-anything/Depth-Anything-V2-Small-hf", "d" * 40)


def test_resolve_model_lock_revision_strict_allows_manifest_sha_with_whitespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "model_lock.yaml"
    sha = "1" * 40
    _write_manifest(
        manifest_path,
        {"depth-anything/Depth-Anything-V2-Small-hf": {"revision": f" {sha} "}},
    )
    monkeypatch.setenv("TP_MODEL_LOCK_MANIFEST", str(manifest_path))
    monkeypatch.setenv("TP_STRICT_MODEL_LOCK", "1")

    assert resolve_model_lock_revision("depth-anything/Depth-Anything-V2-Small-hf", sha) == sha


def test_resolve_model_lock_revision_strict_canonicalizes_sha_casing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = tmp_path / "model_lock.yaml"
    sha_lower = "2" * 40
    _write_manifest(
        manifest_path,
        {"depth-anything/Depth-Anything-V2-Small-hf": {"revision": sha_lower}},
    )
    monkeypatch.setenv("TP_MODEL_LOCK_MANIFEST", str(manifest_path))
    monkeypatch.setenv("TP_STRICT_MODEL_LOCK", "1")

    assert resolve_model_lock_revision("depth-anything/Depth-Anything-V2-Small-hf", sha_lower.upper()) == sha_lower


def test_depth_anything_model_enforces_strict_model_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = tmp_path / "model_lock.yaml"
    _write_manifest(
        manifest_path,
        {"depth-anything/Depth-Anything-V2-Small-hf": {"revision": "NEEDS_VERIFICATION_DEPTH_ANYTHING_V2_SMALL"}},
    )
    monkeypatch.setenv("TP_MODEL_LOCK_MANIFEST", str(manifest_path))
    monkeypatch.setenv("TP_STRICT_MODEL_LOCK", "1")
    monkeypatch.setattr(DepthAnythingV2Model, "_load_model", lambda self: None)

    with pytest.raises(ModelLockError):
        DepthAnythingV2Model(
            variant=ModelVariant.SMALL,
            backend=ModelBackend.PYTORCH_CPU,
            strict_model_lock=True,
        )


def test_depth_anything_coreml_large_variant_fails_with_unsupported_message() -> None:
    with pytest.raises(ValueError, match="CoreML model not available for variant"):
        DepthAnythingV2Model(
            variant=ModelVariant.LARGE,
            backend=ModelBackend.COREML,
        )


def test_depth_anything_coreml_base_variant_fails_with_unpublished_repo_message() -> None:
    with pytest.raises(ValueError, match="Base CoreML repo is not published on Hugging Face"):
        DepthAnythingV2Model(
            variant=ModelVariant.BASE,
            backend=ModelBackend.COREML,
        )


def test_depth_anything_onnx_small_uses_variant_specific_model_lock_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "model_lock.yaml"
    pinned = "d" * 40
    _write_manifest(
        manifest_path,
        {"onnx-community/depth-anything-v2-small": {"revision": pinned}},
    )
    monkeypatch.setenv("TP_MODEL_LOCK_MANIFEST", str(manifest_path))
    monkeypatch.setenv("TP_STRICT_MODEL_LOCK", "1")
    monkeypatch.setattr(DepthAnythingV2Model, "_load_model", lambda self: None)

    model = DepthAnythingV2Model(
        variant=ModelVariant.SMALL,
        backend=ModelBackend.ONNX,
        strict_model_lock=True,
    )
    assert model.onnx_revision == pinned


def test_is_pinned_revision_and_strict_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert is_pinned_revision("e" * 40)
    assert not is_pinned_revision("main")
    assert not is_pinned_revision("NEEDS_VERIFICATION_X")

    monkeypatch.setenv("TP_STRICT_MODEL_LOCK", "1")
    assert is_model_lock_strict_enabled(None) is True
    assert is_model_lock_strict_enabled(False) is False


def test_resolve_model_lock_revision_strict_without_manifest_allows_explicit_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_manifest = tmp_path / "missing_manifest.yaml"
    monkeypatch.setenv("TP_MODEL_LOCK_MANIFEST", str(missing_manifest))
    monkeypatch.setenv("TP_STRICT_MODEL_LOCK", "1")

    explicit = "f" * 40
    assert resolve_model_lock_revision("depth-anything/Depth-Anything-V2-Small-hf", explicit) == explicit


def test_resolve_model_lock_revision_strict_without_manifest_rejects_unpinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_manifest = tmp_path / "missing_manifest.yaml"
    monkeypatch.setenv("TP_MODEL_LOCK_MANIFEST", str(missing_manifest))
    monkeypatch.setenv("TP_STRICT_MODEL_LOCK", "1")

    with pytest.raises(ModelLockError):
        resolve_model_lock_revision("depth-anything/Depth-Anything-V2-Small-hf", None)


def test_model_lock_manifest_path_falls_back_to_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "config" / "model_lock_manifest.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("version: 1\nrepositories: {}\n", encoding="utf-8")

    monkeypatch.delenv("TP_MODEL_LOCK_MANIFEST", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "transformation_portal.core.security.model_lock._repo_root",
        lambda: tmp_path / "does-not-contain-manifest",
    )

    assert model_lock_manifest_path() == manifest


def test_load_model_lock_manifest_normalizes_v2_models_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "model_lock_v2.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "models": {
                    "depth-anything/DA3METRIC-LARGE": {
                        "revision": "1" * 40,
                        "canonical_key": "da3_metric",
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TP_MODEL_LOCK_MANIFEST", str(manifest))

    payload = load_model_lock_manifest()

    assert payload["manifest_schema_version"] == 2
    assert payload["repositories"]["depth-anything/DA3METRIC-LARGE"]["canonical_key"] == "da3_metric"


def test_da3_inference_pipeline_load_uses_pinned_revision_in_strict_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from transformation_portal.lux_depth_v3 import inference as da3_inference
    from transformation_portal.lux_depth_v3.config import DA3Config, DeviceConfig
    from transformation_portal.lux_depth_v3.config import ModelVariant as DA3ModelVariant

    manifest_path = tmp_path / "model_lock.yaml"
    model_id = "depth-anything/DA3-BASE"
    pinned = "7" * 40
    _write_manifest(manifest_path, {model_id: {"revision": pinned}})

    monkeypatch.setenv("TP_MODEL_LOCK_MANIFEST", str(manifest_path))
    monkeypatch.setenv("TP_STRICT_MODEL_LOCK", "1")
    monkeypatch.setattr(
        da3_inference.DA3InferenceEngine,
        "_auto_detect_backend",
        lambda self: da3_inference.ModelBackend.PYTORCH_CPU,
    )

    captured: dict[str, str | None] = {}

    class _FakeDA3Model:
        def to(self, device: str) -> "_FakeDA3Model":
            captured["device"] = device
            return self

        def eval(self) -> "_FakeDA3Model":
            captured["eval"] = "true"
            return self

    class _FakeDepthAnything3:
        @staticmethod
        def from_pretrained(model: str, **kwargs):
            captured["model"] = model
            captured["revision"] = kwargs.get("revision")
            return _FakeDA3Model()

    fake_pkg = types.ModuleType("depth_anything_3")
    fake_api = types.ModuleType("depth_anything_3.api")
    fake_api.DepthAnything3 = _FakeDepthAnything3
    monkeypatch.setitem(sys.modules, "depth_anything_3", fake_pkg)
    monkeypatch.setitem(sys.modules, "depth_anything_3.api", fake_api)

    config = DA3Config(
        model_variant=DA3ModelVariant.METRIC_BASE,
        device=DeviceConfig(device="cpu", use_fp16=False),
    )
    engine = da3_inference.DA3InferenceEngine(config)
    engine._load_da3_model(model_id)

    assert captured["model"] == model_id
    assert captured["revision"] == pinned
    assert captured["device"] == "cpu"


def test_da3_inference_pipeline_load_uses_mps_device_string(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from transformation_portal.lux_depth_v3 import inference as da3_inference
    from transformation_portal.lux_depth_v3.config import DA3Config, DeviceConfig
    from transformation_portal.lux_depth_v3.config import ModelVariant as DA3ModelVariant

    manifest_path = tmp_path / "model_lock.yaml"
    model_id = "depth-anything/DA3-BASE"
    pinned = "6" * 40
    _write_manifest(manifest_path, {model_id: {"revision": pinned}})

    monkeypatch.setenv("TP_MODEL_LOCK_MANIFEST", str(manifest_path))
    monkeypatch.setenv("TP_STRICT_MODEL_LOCK", "1")
    monkeypatch.setattr(
        da3_inference.DA3InferenceEngine,
        "_auto_detect_backend",
        lambda self: da3_inference.ModelBackend.PYTORCH_MPS,
    )

    captured: dict[str, str | None] = {}

    class _FakeDA3Model:
        def to(self, device: str) -> "_FakeDA3Model":
            captured["device"] = device
            return self

        def eval(self) -> "_FakeDA3Model":
            return self

    class _FakeDepthAnything3:
        @staticmethod
        def from_pretrained(model: str, **kwargs):
            captured["model"] = model
            captured["revision"] = kwargs.get("revision")
            return _FakeDA3Model()

    fake_pkg = types.ModuleType("depth_anything_3")
    fake_api = types.ModuleType("depth_anything_3.api")
    fake_api.DepthAnything3 = _FakeDepthAnything3
    monkeypatch.setitem(sys.modules, "depth_anything_3", fake_pkg)
    monkeypatch.setitem(sys.modules, "depth_anything_3.api", fake_api)

    config = DA3Config(
        model_variant=DA3ModelVariant.METRIC_BASE,
        device=DeviceConfig(device="mps", use_fp16=False),
    )
    engine = da3_inference.DA3InferenceEngine(config)
    engine._load_da3_model(model_id)

    assert captured["model"] == model_id
    assert captured["revision"] == pinned
    assert captured["device"] == "mps"


def test_da3_inference_strict_mode_rejects_revisionless_da3_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from transformation_portal.lux_depth_v3 import inference as da3_inference
    from transformation_portal.lux_depth_v3.config import DA3Config, DeviceConfig
    from transformation_portal.lux_depth_v3.config import ModelVariant as DA3ModelVariant

    manifest_path = tmp_path / "model_lock.yaml"
    model_id = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    pinned = "8" * 40
    _write_manifest(manifest_path, {model_id: {"revision": pinned}})

    monkeypatch.setenv("TP_MODEL_LOCK_MANIFEST", str(manifest_path))
    monkeypatch.setenv("TP_STRICT_MODEL_LOCK", "1")
    monkeypatch.setattr(
        da3_inference.DA3InferenceEngine,
        "_auto_detect_backend",
        lambda self: da3_inference.ModelBackend.PYTORCH_CPU,
    )

    fallback_used = {"called_without_revision": False}

    class _FakeDA3Model:
        def to(self, _device: str) -> "_FakeDA3Model":
            return self

        def eval(self) -> "_FakeDA3Model":
            return self

    class _FakeDepthAnything3:
        @staticmethod
        def from_pretrained(_model_id: str, **kwargs):
            if "revision" in kwargs:
                raise TypeError("revision unsupported by test stub")
            fallback_used["called_without_revision"] = True
            return _FakeDA3Model()

    fake_pkg = types.ModuleType("depth_anything_3")
    fake_api = types.ModuleType("depth_anything_3.api")
    fake_api.DepthAnything3 = _FakeDepthAnything3
    monkeypatch.setitem(sys.modules, "depth_anything_3", fake_pkg)
    monkeypatch.setitem(sys.modules, "depth_anything_3.api", fake_api)

    config = DA3Config(
        model_variant=DA3ModelVariant.METRIC_LARGE,
        device=DeviceConfig(device="cpu", use_fp16=False),
    )
    engine = da3_inference.DA3InferenceEngine(config)

    with pytest.raises(RuntimeError, match="revision unsupported by test stub"):
        engine._load_da3_model(model_id)

    assert fallback_used["called_without_revision"] is False


def test_artifact_attestation_gaussian_splatting_uses_source_only_shape() -> None:
    """Regression: the 3DGS attestation block in the shipped manifest is source-only.

    The block previously declared `source_type: direct_checkpoint` and a non-existent
    `gaussian_splatting_base.pt` artifact. Inria distributes source, not weights, so
    the block was rewritten to `source_type: git_release` with `artifacts: []`.
    """
    import re

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "config" / "model_lock_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    gs = manifest["artifact_attestation"]["gaussian_splatting"]
    assert gs["source_type"] == "git_release"
    assert gs["source_url"] == "https://github.com/graphdeco-inria/gaussian-splatting"
    assert gs["artifacts"] == []
    assert gs["verification"]["method"] == "source_commit"

    commit = gs["source_commit_or_tag"]
    assert commit == "PENDING_VERIFICATION" or re.fullmatch(
        r"[0-9a-f]{40}", commit
    ), f"source_commit_or_tag must be PENDING_VERIFICATION or a 40-hex commit SHA, got {commit!r}"
