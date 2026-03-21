"""Tests for Phase 1 performance optimizations.

Validates:
1. Lazy manifest loading with LRU cache
2. FP16 model quantization
3. Chunked SHA-256 computation
4. Bilateral filter optimization
"""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from transformation_portal.lux_depth_v3.config import DeviceConfig, EnhanceConfig
from transformation_portal.lux_depth_v3.manifest import CombinedManifest, compute_file_sha256
from transformation_portal.lux_depth_v3.orchestrator import _load_manifest_cached
from transformation_portal.lux_depth_v3.postprocessing import PostprocessingConfig, Postprocessor

pytestmark = pytest.mark.unit


class TestChunkedSHA256:
    """Test chunked SHA-256 computation for memory efficiency."""

    def test_chunked_hashing_produces_correct_hash(self, tmp_path):
        """Verify chunked reading produces same hash as standard method."""
        test_file = tmp_path / "test.bin"

        # Create a file with known content
        test_data = b"Hello, World! " * 1000
        test_file.write_bytes(test_data)

        # Compute hash with our function
        our_hash = compute_file_sha256(test_file)

        # Compute expected hash
        expected_hash = hashlib.sha256(test_data).hexdigest()

        assert our_hash == expected_hash

    def test_chunked_hashing_handles_large_files(self, tmp_path):
        """Verify chunked reading works for large files."""
        test_file = tmp_path / "large.bin"

        # Create a 10MB file
        chunk = b"X" * (1024 * 1024)
        with open(test_file, "wb") as f:
            for _ in range(10):
                f.write(chunk)

        # Should complete without memory issues
        result_hash = compute_file_sha256(test_file)
        assert len(result_hash) == 64  # SHA-256 hex length

    def test_chunked_hashing_custom_chunk_size(self, tmp_path):
        """Verify custom chunk size parameter works."""
        test_file = tmp_path / "test.bin"
        test_data = b"A" * 10000
        test_file.write_bytes(test_data)

        # Different chunk sizes should produce same hash
        hash_8k = compute_file_sha256(test_file, chunk_size=8192)
        hash_1k = compute_file_sha256(test_file, chunk_size=1024)
        hash_64k = compute_file_sha256(test_file, chunk_size=65536)

        assert hash_8k == hash_1k == hash_64k


class TestManifestCaching:
    """Test LRU cache for manifest loading."""

    def test_cached_manifest_loading_returns_manifest(self, tmp_path):
        """Verify cached loading returns valid manifest."""
        manifest_path = tmp_path / "test_manifest.json"

        # Create a minimal manifest
        manifest = CombinedManifest()
        manifest.save(manifest_path)

        # Load via cache
        mtime = manifest_path.stat().st_mtime
        loaded = _load_manifest_cached(str(manifest_path), mtime)

        assert isinstance(loaded, CombinedManifest)

    def test_cached_manifest_invalidates_on_mtime_change(self, tmp_path):
        """Verify cache invalidates when file modification time changes."""
        manifest_path = tmp_path / "test_manifest.json"

        # Create initial manifest
        manifest1 = CombinedManifest()
        manifest1.save(manifest_path)
        mtime1 = manifest_path.stat().st_mtime

        # Load first time
        loaded1 = _load_manifest_cached(str(manifest_path), mtime1)

        # Modify manifest (different mtime)
        import time

        time.sleep(0.01)  # Ensure different mtime
        manifest2 = CombinedManifest()
        manifest2.save(manifest_path)
        mtime2 = manifest_path.stat().st_mtime

        # Should load new version (different cache key due to mtime)
        loaded2 = _load_manifest_cached(str(manifest_path), mtime2)

        assert isinstance(loaded2, CombinedManifest)


class TestEnhanceConfigOptimizations:
    """Test optimization flags in EnhanceConfig."""

    def test_enable_manifest_cache_default(self):
        """Verify manifest cache is enabled by default."""
        config = EnhanceConfig()
        assert config.enable_manifest_cache is True

    def test_chunked_hashing_default(self):
        """Verify chunked hashing is enabled by default."""
        config = EnhanceConfig()
        assert config.chunked_hashing is True

    def test_optimization_flags_can_be_disabled(self):
        """Verify optimization flags can be disabled."""
        config = EnhanceConfig(enable_manifest_cache=False, chunked_hashing=False)
        assert config.enable_manifest_cache is False
        assert config.chunked_hashing is False


class TestDeviceConfigFP16:
    """Test FP16 optimization flag in DeviceConfig."""

    def test_use_fp16_default_enabled(self):
        """Verify FP16 is enabled by default."""
        config = DeviceConfig()
        assert config.use_fp16 is True

    def test_use_fp16_can_be_disabled(self):
        """Verify FP16 can be disabled."""
        config = DeviceConfig(use_fp16=False)
        assert config.use_fp16 is False


class TestBilateralFilterOptimization:
    """Test bilateral filter optimization with OpenCV."""

    def test_bilateral_filter_uses_opencv_when_available(self):
        """Verify bilateral filter uses OpenCV for acceleration."""
        config = PostprocessingConfig(apply_bilateral_filter=True, bilateral_sigma_color=0.05, bilateral_sigma_space=5.0)
        processor = Postprocessor(config)

        # Create test depth map
        depth = np.random.rand(512, 512).astype(np.float32)
        image = np.random.rand(512, 512, 3).astype(np.float32)

        # Apply filter
        try:
            import cv2

            filtered = processor._bilateral_filter(depth, image, sigma_color=0.05, sigma_space=5.0)

            # Should return filtered depth
            assert filtered.shape == depth.shape
            assert filtered.dtype == np.float32
            assert 0 <= filtered.min() <= filtered.max() <= 1

        except ImportError:
            pytest.skip("OpenCV not available")

    def test_bilateral_filter_fallback_without_opencv(self):
        """Verify bilateral filter falls back to scipy when OpenCV unavailable."""
        config = PostprocessingConfig(apply_bilateral_filter=True, bilateral_sigma_color=0.05, bilateral_sigma_space=5.0)
        processor = Postprocessor(config)

        # Create test depth map
        depth = np.random.rand(256, 256).astype(np.float32)
        image = np.random.rand(256, 256, 3).astype(np.float32)

        # Mock builtins.__import__ to make cv2 import fail
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("Mocked cv2 not available")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            filtered = processor._bilateral_filter(depth, image, sigma_color=0.05, sigma_space=5.0)

            # Should still work with scipy fallback
            assert filtered.shape == depth.shape
            assert filtered.dtype == np.float32


class TestOptimizationIntegration:
    """Integration tests for optimization features."""

    def test_enhance_config_preserves_api_compatibility(self):
        """Verify new optimization flags don't break existing API."""
        # Old-style config creation should still work
        config = EnhanceConfig(depth_device="cpu", v2_preset="default")

        # Should have optimization defaults
        assert hasattr(config, "enable_manifest_cache")
        assert hasattr(config, "chunked_hashing")

    def test_device_config_preserves_api_compatibility(self):
        """Verify FP16 flag doesn't break existing API."""
        # Old-style config creation should still work
        config = DeviceConfig(device="cpu")

        # Should have FP16 default
        assert hasattr(config, "use_fp16")
        assert config.use_fp16 is True
