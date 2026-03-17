"""Tests for manifest-aware HF model loader utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from transformation_portal.models.hf_manifest_loader import (
    HFManifestLoaderError,
    HFResolvedLocalModel,
    _common_local_root,
)


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
