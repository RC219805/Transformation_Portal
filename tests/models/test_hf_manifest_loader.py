"""Tests for manifest-aware HF model loader utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from transformation_portal.models import hf_manifest_loader as loader_module
from transformation_portal.models.hf_manifest_loader import (
    HFManifestLoaderError,
    HFResolvedLocalModel,
    _common_local_root,
    validate_local_model_shard_indexes,
)

pytestmark = pytest.mark.unit


def test_manifest_resolution_validates_existing_local_index_before_returning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "model.bin.index.json").write_text(json.dumps({"weight_map": {"a": "../outside.bin"}}))
    monkeypatch.setattr(loader_module, "_infer_local_root_from_hub", lambda **kwargs: tmp_path)
    with pytest.raises(HFManifestLoaderError, match="unsafe weight-map path"):
        loader_module.resolve_manifest_model("test", {"repo_id": "org/repo", "revision": "a" * 40})


def test_verified_model_files_cannot_cross_snapshot_roots() -> None:
    with pytest.raises(HFManifestLoaderError, match="one snapshot root"):
        _common_local_root([Path("/cache/snapshots/one/model.bin"), Path("/cache/snapshots/two/config.json")])


@pytest.mark.parametrize("shard", ["../outside.bin", "/tmp/outside.bin", "C:\\outside.bin"])
def test_local_model_index_rejects_traversal(tmp_path: Path, shard: str) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": shard}}))
    with pytest.raises(HFManifestLoaderError, match="unsafe weight-map path"):
        validate_local_model_shard_indexes(tmp_path)


def test_local_model_index_rejects_external_symlink(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    (tmp_path / "outside.bin").write_bytes(b"model")
    (root / "shard.bin").symlink_to(tmp_path / "outside.bin")
    (root / "model.bin.index.json").write_text(json.dumps({"weight_map": {"a": "shard.bin"}}))
    with pytest.raises(HFManifestLoaderError, match="outside"):
        validate_local_model_shard_indexes(root)


def test_hf_blob_symlinks_remain_supported(tmp_path: Path) -> None:
    cache = tmp_path / "models--org--repo"
    root = cache / "snapshots" / ("a" * 40)
    root.mkdir(parents=True)
    blobs = cache / "blobs"
    blobs.mkdir()
    (blobs / "model").write_bytes(b"weights")
    (root / "shard.bin").symlink_to(blobs / "model")
    (blobs / "index").write_text(json.dumps({"weight_map": {"a": "shard.bin"}}))
    (root / "model.bin.index.json").symlink_to(blobs / "index")
    validate_local_model_shard_indexes(root)


def test_hf_blob_directory_cannot_redirect_outside_cache(tmp_path: Path) -> None:
    cache = tmp_path / "models--org--repo"
    root = cache / "snapshots" / ("a" * 40)
    root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "model.bin").write_bytes(b"weights")
    (cache / "blobs").symlink_to(outside, target_is_directory=True)
    (root / "shard.bin").symlink_to(cache / "blobs" / "model.bin")
    (root / "model.bin.index.json").write_text(json.dumps({"weight_map": {"a": "shard.bin"}}))
    with pytest.raises(HFManifestLoaderError, match="blob directory must not be a symlink"):
        validate_local_model_shard_indexes(root)


def test_local_model_index_limits_shards_before_resolving_them(tmp_path: Path) -> None:
    (tmp_path / "model.bin.index.json").write_text(
        json.dumps({"weight_map": {f"layer{i}": f"shard{i}.bin" for i in range(1025)}})
    )
    with pytest.raises(HFManifestLoaderError, match="too many checkpoint shards"):
        validate_local_model_shard_indexes(tmp_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFO")
@pytest.mark.parametrize("fifo_name", ["shard.bin", "model.bin.index.json"])
def test_local_model_index_rejects_fifo_without_opening_it_blocking(tmp_path: Path, fifo_name: str) -> None:
    (tmp_path / "model.bin.index.json").write_text(json.dumps({"weight_map": {"a": "shard.bin"}}))
    (tmp_path / "shard.bin").write_bytes(b"weights")
    (tmp_path / fifo_name).unlink()
    os.mkfifo(tmp_path / fifo_name)
    with pytest.raises(HFManifestLoaderError, match="not a regular file"):
        validate_local_model_shard_indexes(tmp_path)


class TestCommonLocalRoot:
    """Tests for _common_local_root utility."""

    def test_empty_paths_raises(self) -> None:
        """_common_local_root should raise for empty path list."""
        with pytest.raises(HFManifestLoaderError, match="empty file list"):
            _common_local_root([])

    def test_no_snapshots_raises(self) -> None:
        """_common_local_root should raise when snapshots not in path."""
        paths = [Path("/some/random/path/file.txt")]
        with pytest.raises(HFManifestLoaderError, match="snapshot root"):
            _common_local_root(paths)

    def test_typical_hf_cache_path(self) -> None:
        """_common_local_root should extract root from typical HF cache path."""
        # Typical HF cache structure
        paths = [
            Path("/home/user/.cache/huggingface/hub/models--org--repo/snapshots/abc123/config.json"),
        ]
        root = _common_local_root(paths)
        expected = Path("/home/user/.cache/huggingface/hub/models--org--repo/snapshots/abc123")
        assert root == expected

    def test_multiple_files_same_root(self) -> None:
        """_common_local_root should handle multiple files with same root."""
        base = Path("/home/user/.cache/huggingface/hub/models--org--repo/snapshots/abc123")
        paths = [
            base / "config.json",
            base / "model.safetensors",
            base / "tokenizer.json",
        ]
        root = _common_local_root(paths)
        assert root == base


class TestHFResolvedLocalModel:
    """Tests for HFResolvedLocalModel dataclass."""

    def test_basic_construction(self) -> None:
        """HFResolvedLocalModel should store all fields correctly."""
        model = HFResolvedLocalModel(
            model_key="test_model",
            repo_id="org/repo",
            revision="abc123",
            local_root=Path("/path/to/snapshot"),
            resolved_files={"config.json": Path("/path/to/snapshot/config.json")},
        )
        assert model.model_key == "test_model"
        assert model.repo_id == "org/repo"
        assert model.revision == "abc123"
        assert model.local_root == Path("/path/to/snapshot")
        assert "config.json" in model.resolved_files

    def test_is_frozen(self) -> None:
        """HFResolvedLocalModel should be frozen (immutable)."""
        model = HFResolvedLocalModel(
            model_key="test",
            repo_id="org/repo",
            revision="abc",
            local_root=Path("/path"),
            resolved_files={},
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            model.model_key = "other"  # type: ignore
