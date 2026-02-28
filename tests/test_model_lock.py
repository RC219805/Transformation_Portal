"""Tests for model lock manifest resolution and strict enforcement."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from transformation_portal.core.security.model_lock import (
    ModelLockError,
    is_model_lock_strict_enabled,
    is_pinned_revision,
    load_model_lock_manifest,
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
