"""Tests for manifest-aware HF model loader utilities."""

from __future__ import annotations

import json
import os
import sys
import types
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

    def test_cache_dir_containing_snapshots_component(self) -> None:
        """A 'snapshots' directory above the HF repo cache must not be taken as the snapshot root."""
        sha = "a" * 40
        base = Path(f"/mnt/snapshots/hf-cache/models--org--repo/snapshots/{sha}")
        root = _common_local_root([base / "config.json", base / "model.safetensors"], sha)
        assert root == base

    def test_repository_subdirectory_named_snapshots(self) -> None:
        """A repository subdirectory called 'snapshots' must not move the snapshot root."""
        sha = "b" * 40
        base = Path(f"/cache/hub/models--org--repo/snapshots/{sha}")
        root = _common_local_root([base / "snapshots" / "weights" / "model.bin", base / "config.json"], sha)
        assert root == base

    def test_snapshot_root_must_match_pinned_revision(self) -> None:
        """The inferred snapshot directory must be the pinned commit."""
        path = Path(f"/cache/hub/models--org--repo/snapshots/{'c' * 40}/config.json")
        with pytest.raises(HFManifestLoaderError, match="does not match the pinned revision"):
            _common_local_root([path], "d" * 40)

    def test_manifest_resolution_uses_repo_snapshot_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """resolve_manifest_model must validate and return the repo snapshot, not a cache-dir ancestor."""
        sha = "e" * 40
        base = tmp_path / "snapshots" / "hf-cache" / "models--org--repo" / "snapshots" / sha
        base.mkdir(parents=True)
        (base / "config.json").write_text("{}")
        monkeypatch.setattr(
            loader_module,
            "resolve_all_required_files",
            lambda record, **kwargs: {"config.json": base / "config.json"},
        )
        resolved = loader_module.resolve_manifest_model(
            "test",
            {"repo_id": "org/repo", "revision": sha, "required_files": [{"path": "config.json"}]},
        )
        assert resolved.local_root == base


class TestInferLocalRootFromHub:
    """The hub-inferred snapshot root follows the same repo-cache and revision rules."""

    @staticmethod
    def _fake_hub(monkeypatch: pytest.MonkeyPatch, downloaded: Path) -> list[str]:
        requested: list[str] = []

        def hf_hub_download(**kwargs: object) -> str:
            requested.append(str(kwargs["filename"]))
            return str(downloaded)

        monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=hf_hub_download))
        return requested

    def test_cache_dir_containing_snapshots_component(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sha = "a" * 40
        base = Path(f"/mnt/snapshots/hf-cache/models--org--repo/snapshots/{sha}")
        self._fake_hub(monkeypatch, base / "config.json")

        root = loader_module._infer_local_root_from_hub(repo_id="org/repo", revision=sha, cache_dir="/mnt/snapshots/hf-cache")

        assert root == base

    def test_revision_mismatch_is_reported_instead_of_trying_other_files(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The root error must surface; it used to be swallowed and retried with the next config file.
        requested = self._fake_hub(monkeypatch, Path(f"/cache/hub/models--org--repo/snapshots/{'c' * 40}/config.json"))

        with pytest.raises(HFManifestLoaderError, match="does not match the pinned revision"):
            loader_module._infer_local_root_from_hub(repo_id="org/repo", revision="d" * 40)

        assert requested == ["config.json"]


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
