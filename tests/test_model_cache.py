"""Tests for DA3 model cache management."""

import pytest
from pathlib import Path
import json
from unittest.mock import patch, MagicMock
from datetime import datetime

# Try to import dependencies - skip entire module if unavailable
try:
    import cv2  # noqa: F401
    import torch  # noqa: F401
    from lux_depth_v3.model_cache import ModelCacheInfo, CacheStrategy, ModelCacheManager, precache_models

    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    SKIP_REASON = f"Dependencies not available: {e}"

pytestmark = pytest.mark.skipif(not DEPS_AVAILABLE, reason=getattr(globals(), "SKIP_REASON", "Dependencies not available"))


class TestModelCacheInfo:
    """Test ModelCacheInfo dataclass."""

    def test_cache_info_creation(self):
        """Test creating cache info."""
        info = ModelCacheInfo(
            model_id="depth-anything/DA3-LARGE-1.1",
            local_path=Path("/cache/models/da3-large"),
            size_bytes=1500000000,
            cached_at="2025-12-19T10:00:00",
            verified=True,
            version="1.1",
        )

        assert info.model_id == "depth-anything/DA3-LARGE-1.1"
        assert info.local_path == Path("/cache/models/da3-large")
        assert info.size_bytes == 1500000000
        assert info.verified is True
        assert info.version == "1.1"

    def test_cache_info_to_dict(self):
        """Test converting cache info to dictionary."""
        info = ModelCacheInfo(
            model_id="depth-anything/DA3-LARGE-1.1",
            local_path=Path("/cache/models/da3-large"),
            size_bytes=1500000000,
            cached_at="2025-12-19T10:00:00",
            verified=True,
            version="1.1",
        )

        data = info.to_dict()

        assert data["model_id"] == "depth-anything/DA3-LARGE-1.1"
        assert data["local_path"] == "/cache/models/da3-large"
        assert data["size_bytes"] == 1500000000
        assert data["size_mb"] == pytest.approx(1430.51, rel=0.01)
        assert data["size_gb"] == pytest.approx(1.40, rel=0.01)
        assert data["verified"] is True
        assert data["version"] == "1.1"


class TestCacheStrategy:
    """Test CacheStrategy enum."""

    def test_cache_strategies(self):
        """Test cache strategy enum values."""
        assert CacheStrategy.HF_CACHE.value == "hf_cache"
        assert CacheStrategy.SNAPSHOT.value == "snapshot"
        assert CacheStrategy.SYMLINK.value == "symlink"


class TestModelCacheManager:
    """Test ModelCacheManager."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create temporary cache directory."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.fixture
    def mock_metadata(self):
        """Mock metadata."""
        return {
            "models": {
                "depth-anything/DA3-LARGE-1.1": {
                    "model_id": "depth-anything/DA3-LARGE-1.1",
                    "local_path": "/cache/da3-large",
                    "size_bytes": 1500000000,
                    "cached_at": "2025-12-19T10:00:00",
                    "verified": True,
                    "version": "1.1",
                }
            },
            "last_updated": "2025-12-19T10:00:00",
        }

    def test_cache_manager_initialization(self, temp_cache_dir):
        """Test cache manager initialization."""
        manager = ModelCacheManager(cache_dir=temp_cache_dir)

        assert manager.cache_dir == temp_cache_dir
        assert manager.strategy == CacheStrategy.HF_CACHE
        assert manager.metadata == {"models": {}, "last_updated": None}

    def test_cache_manager_default_location(self):
        """Test default cache location."""
        manager = ModelCacheManager()

        # Should be HF default or environment variable
        assert manager.cache_dir.exists() or str(manager.cache_dir).endswith("huggingface/hub")

    def test_get_default_cache_dir_with_env(self, monkeypatch, tmp_path):
        """Test default cache directory with environment variables."""
        hf_home = tmp_path / "hf_home"
        hf_home.mkdir()

        monkeypatch.setenv("HF_HOME", str(hf_home))

        manager = ModelCacheManager()
        assert manager.cache_dir == hf_home / "hub"

    def test_official_models_list(self):
        """Test official models are defined."""
        manager = ModelCacheManager()

        # Check all expected models
        expected_models = [
            "nested-giant-large-v1.1",
            "giant-v1.1",
            "large-v1.1",
            "nested-giant-large",
            "giant",
            "large",
            "base",
            "small",
            "metric-large",
            "mono-large",
        ]

        for model_key in expected_models:
            assert model_key in manager.OFFICIAL_MODELS
            assert manager.OFFICIAL_MODELS[model_key].startswith("depth-anything/")

    def test_recommended_sets(self):
        """Test recommended model sets."""
        manager = ModelCacheManager()

        # Essential set
        assert "nested-giant-large-v1.1" in manager.RECOMMENDED_SETS["essential"]
        assert "metric-large" in manager.RECOMMENDED_SETS["essential"]

        # Production set
        assert len(manager.RECOMMENDED_SETS["production"]) == 4
        assert "nested-giant-large-v1.1" in manager.RECOMMENDED_SETS["production"]

        # Benchmark set
        assert len(manager.RECOMMENDED_SETS["benchmark"]) >= 7

        # All set
        assert len(manager.RECOMMENDED_SETS["all"]) == 10

    def test_model_id_to_path(self, temp_cache_dir):
        """Test model ID to path conversion."""
        manager = ModelCacheManager(cache_dir=temp_cache_dir)

        path = manager._model_id_to_path("depth-anything/DA3-LARGE-1.1")
        assert path == "depth-anything--DA3-LARGE-1.1"
        assert "/" not in path

    def test_load_metadata(self, temp_cache_dir, mock_metadata):
        """Test loading metadata from file."""
        metadata_file = temp_cache_dir / "lux_depth_v3_cache.json"
        with open(metadata_file, "w") as f:
            json.dump(mock_metadata, f)

        manager = ModelCacheManager(cache_dir=temp_cache_dir)

        assert manager.metadata == mock_metadata
        assert "depth-anything/DA3-LARGE-1.1" in manager.metadata["models"]

    def test_save_metadata(self, temp_cache_dir):
        """Test saving metadata to file."""
        manager = ModelCacheManager(cache_dir=temp_cache_dir)

        manager.metadata["models"]["test-model"] = {"model_id": "test-model", "size_bytes": 1000000}

        manager._save_metadata()

        metadata_file = temp_cache_dir / "lux_depth_v3_cache.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            data = json.load(f)

        assert "test-model" in data["models"]

    def test_is_cached(self, temp_cache_dir, mock_metadata):
        """Test checking if model is cached."""
        # Create actual model directory structure that _is_cached checks for
        model_dir = temp_cache_dir / "da3-large"
        model_dir.mkdir(parents=True)
        # Add a file so rglob("*") returns something
        (model_dir / "model.safetensors").touch()

        # Update metadata to use the real path
        mock_metadata["models"]["depth-anything/DA3-LARGE-1.1"]["local_path"] = str(model_dir)

        metadata_file = temp_cache_dir / "lux_depth_v3_cache.json"
        with open(metadata_file, "w") as f:
            json.dump(mock_metadata, f)

        manager = ModelCacheManager(cache_dir=temp_cache_dir)

        assert manager._is_cached("depth-anything/DA3-LARGE-1.1") is True
        assert manager._is_cached("depth-anything/DA3-BASE") is False

    def test_get_cache_info(self, temp_cache_dir, mock_metadata):
        """Test getting cache info for a model."""
        metadata_file = temp_cache_dir / "lux_depth_v3_cache.json"
        with open(metadata_file, "w") as f:
            json.dump(mock_metadata, f)

        manager = ModelCacheManager(cache_dir=temp_cache_dir)

        info = manager._get_cache_info("depth-anything/DA3-LARGE-1.1")

        assert info.model_id == "depth-anything/DA3-LARGE-1.1"
        assert info.size_bytes == 1500000000
        assert info.verified is True

    def test_verify_model(self, temp_cache_dir):
        """Test model verification."""
        manager = ModelCacheManager(cache_dir=temp_cache_dir)

        # Create mock model directory with files
        model_dir = temp_cache_dir / "model"
        model_dir.mkdir()
        (model_dir / "config.json").touch()
        (model_dir / "model.safetensors").touch()

        cache_info = ModelCacheInfo(
            model_id="test-model",
            local_path=model_dir,
            size_bytes=1000000,
            cached_at=datetime.now().isoformat(),
            verified=False,
        )

        # Should pass verification (directory exists with files)
        assert manager._verify_model(cache_info) is True

        # Test non-existent directory
        cache_info.local_path = temp_cache_dir / "nonexistent"
        assert manager._verify_model(cache_info) is False

    def test_get_directory_size(self, temp_cache_dir):
        """Test calculating directory size."""
        manager = ModelCacheManager(cache_dir=temp_cache_dir)

        # Create test files
        model_dir = temp_cache_dir / "model"
        model_dir.mkdir()

        (model_dir / "file1.bin").write_bytes(b"x" * 1000)
        (model_dir / "file2.bin").write_bytes(b"y" * 2000)

        subdir = model_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.bin").write_bytes(b"z" * 500)

        size = manager._get_directory_size(model_dir)
        assert size == 3500

    def test_list_cached_models(self, temp_cache_dir, mock_metadata):
        """Test listing cached models."""
        metadata_file = temp_cache_dir / "lux_depth_v3_cache.json"
        with open(metadata_file, "w") as f:
            json.dump(mock_metadata, f)

        manager = ModelCacheManager(cache_dir=temp_cache_dir)

        cached = manager.list_cached_models()

        assert len(cached) == 1
        assert cached[0].model_id == "depth-anything/DA3-LARGE-1.1"
        assert cached[0].size_bytes == 1500000000

    def test_get_cache_stats(self, temp_cache_dir, mock_metadata):
        """Test getting cache statistics."""
        metadata_file = temp_cache_dir / "lux_depth_v3_cache.json"
        with open(metadata_file, "w") as f:
            json.dump(mock_metadata, f)

        manager = ModelCacheManager(cache_dir=temp_cache_dir)

        stats = manager.get_cache_stats()

        assert stats["cache_dir"] == str(temp_cache_dir)
        assert stats["num_models"] == 1
        assert stats["total_size_bytes"] == 1500000000
        assert stats["total_size_gb"] == pytest.approx(1.40, rel=0.01)
        assert "models" in stats
        assert stats["last_updated"] == "2025-12-19T10:00:00"

    def test_download_model_invalid_key(self, temp_cache_dir):
        """Test downloading with invalid model key."""
        manager = ModelCacheManager(cache_dir=temp_cache_dir)

        with pytest.raises(ValueError, match="Unknown model"):
            manager.download_model("invalid-model")

    @patch("lux_depth_v3.model_cache.ModelCacheManager._download_hf_cache")
    def test_download_model_already_cached(self, mock_download, temp_cache_dir, mock_metadata):
        """Test downloading already cached model."""
        # Create actual model directory structure so _is_cached returns True
        model_dir = temp_cache_dir / "da3-large"
        model_dir.mkdir(parents=True)
        # Add a file so rglob("*") returns something
        (model_dir / "model.safetensors").touch()

        # Update metadata to use the real path
        mock_metadata["models"]["depth-anything/DA3-LARGE-1.1"]["local_path"] = str(model_dir)

        metadata_file = temp_cache_dir / "lux_depth_v3_cache.json"
        with open(metadata_file, "w") as f:
            json.dump(mock_metadata, f)

        manager = ModelCacheManager(cache_dir=temp_cache_dir)

        # Should not download, return cached info
        info = manager.download_model("large-v1.1", force=False)

        assert mock_download.call_count == 0
        assert info.model_id == "depth-anything/DA3-LARGE-1.1"

    def test_download_models_with_set(self, temp_cache_dir):
        """Test downloading model set."""
        manager = ModelCacheManager(cache_dir=temp_cache_dir)

        with patch.object(manager, "download_model") as mock_download:
            mock_download.return_value = ModelCacheInfo(
                model_id="test",
                local_path=temp_cache_dir,
                size_bytes=1000000,
                cached_at=datetime.now().isoformat(),
                verified=True,
            )

            results = manager.download_models(model_set="essential")

            # Essential set has 2 models
            assert mock_download.call_count == 2
            assert len(results) == 2

    def test_download_models_with_keys(self, temp_cache_dir):
        """Test downloading specific model keys."""
        manager = ModelCacheManager(cache_dir=temp_cache_dir)

        with patch.object(manager, "download_model") as mock_download:
            mock_download.return_value = ModelCacheInfo(
                model_id="test",
                local_path=temp_cache_dir,
                size_bytes=1000000,
                cached_at=datetime.now().isoformat(),
                verified=True,
            )

            results = manager.download_models(model_keys=["large-v1.1", "base"])

            assert mock_download.call_count == 2
            assert len(results) == 2

    def test_download_models_invalid_set(self, temp_cache_dir):
        """Test downloading with invalid model set."""
        manager = ModelCacheManager(cache_dir=temp_cache_dir)

        with pytest.raises(ValueError, match="Unknown model set"):
            manager.download_models(model_set="invalid-set")


class TestPrecacheModels:
    """Test precache_models convenience function."""

    @patch("lux_depth_v3.model_cache.ModelCacheManager")
    def test_precache_models_default(self, mock_manager_class):
        """Test precache_models with defaults."""
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        mock_manager.download_models.return_value = []

        precache_models()

        mock_manager_class.assert_called_once_with(cache_dir=None)
        mock_manager.download_models.assert_called_once_with(model_set="essential", force=False)

    @patch("lux_depth_v3.model_cache.ModelCacheManager")
    def test_precache_models_custom(self, mock_manager_class):
        """Test precache_models with custom parameters."""
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager
        mock_manager.download_models.return_value = []

        cache_dir = Path("/custom/cache")
        precache_models(model_set="production", cache_dir=cache_dir, force=True)

        mock_manager_class.assert_called_once_with(cache_dir=cache_dir)
        mock_manager.download_models.assert_called_once_with(model_set="production", force=True)


class TestOfflineOperation:
    """Test offline operation capabilities."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create temporary cache directory."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.fixture
    def mock_metadata(self):
        """Mock metadata."""
        return {
            "models": {
                "depth-anything/DA3-LARGE-1.1": {
                    "model_id": "depth-anything/DA3-LARGE-1.1",
                    "local_path": "/cache/da3-large",
                    "size_bytes": 1500000000,
                    "cached_at": "2025-12-19T10:00:00",
                    "verified": True,
                    "version": "1.1",
                }
            },
            "last_updated": "2025-12-19T10:00:00",
        }

    def test_offline_mode_with_cached_models(self, temp_cache_dir, mock_metadata, monkeypatch):
        """Test that cached models can be used offline."""
        # Setup cached models
        metadata_file = temp_cache_dir / "lux_depth_v3_cache.json"
        with open(metadata_file, "w") as f:
            json.dump(mock_metadata, f)

        # Set offline mode
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")

        manager = ModelCacheManager(cache_dir=temp_cache_dir)

        # Should be able to list cached models offline
        cached = manager.list_cached_models()
        assert len(cached) == 1

        # Should be able to get cache stats offline
        stats = manager.get_cache_stats()
        assert stats["num_models"] == 1
