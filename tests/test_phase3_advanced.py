"""
Tests for Phase 3 Advanced Optimizations (lux_depth_v3).

Tests CoreML backend, PBR GPU batching, MessagePack manifests, and xxHash.
"""

import platform
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Import Phase 3 features
from src.transformation_portal.lux_depth_v3.config import DeviceConfig, EnhanceConfig
from src.transformation_portal.lux_depth_v3.manifest import CombinedManifest, InputMetadata
from src.transformation_portal.lux_depth_v3.orchestrator import make_output_key
from src.transformation_portal.lux_depth_v3.pbr import PBRConfig, generate_pbr_maps, generate_pbr_maps_batched

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_depth():
    """Create sample depth map for testing."""
    return np.random.rand(512, 512).astype(np.float32)


@pytest.fixture
def sample_depths():
    """Create multiple sample depth maps for batch testing."""
    return [np.random.rand(256, 256).astype(np.float32) for _ in range(5)]


# ============================================================================
# Test 1: CoreML Backend Configuration
# ============================================================================


def test_coreml_config_flag():
    """Test CoreML configuration flag in DeviceConfig."""
    config = DeviceConfig(use_coreml=True)
    assert config.use_coreml is True

    config_default = DeviceConfig()
    assert config_default.use_coreml is False  # Opt-in by default


def test_coreml_should_use_logic():
    """Test CoreML backend selection logic."""
    from src.transformation_portal.lux_depth_v3.coreml_backend import should_use_coreml

    config_enabled = DeviceConfig(use_coreml=True)
    config_disabled = DeviceConfig(use_coreml=False)

    # Should respect config flag
    result_disabled = should_use_coreml(config_disabled)
    assert result_disabled is False

    # Force flag should override
    result_forced = should_use_coreml(config_disabled, force=True)
    # Result depends on platform and dependencies
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        # On Apple Silicon, depends on coremltools availability
        try:
            import coremltools

            assert isinstance(result_forced, bool)
        except ImportError:
            assert result_forced is False
    else:
        assert result_forced is False


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64", reason="CoreML only available on Apple Silicon"
)
def test_coreml_cache_stats():
    """Test CoreML cache statistics."""
    from src.transformation_portal.lux_depth_v3.coreml_backend import get_coreml_cache_stats

    stats = get_coreml_cache_stats()

    assert "cache_dir" in stats
    assert "model_count" in stats
    assert "total_size_mb" in stats
    assert isinstance(stats["model_count"], int)
    assert stats["model_count"] >= 0


# ============================================================================
# Test 2: PBR GPU Batching
# ============================================================================


def test_pbr_batched_empty_list():
    """Test batched PBR generation with empty input."""
    result = generate_pbr_maps_batched([], PBRConfig())
    assert result == []


def test_pbr_batched_correctness(sample_depths):
    """Test batched PBR generation produces same results as sequential."""
    config = PBRConfig(normal_strength=1.0, roughness_strength=1.0, ao_strength=1.0)

    # Sequential generation
    sequential_results = [generate_pbr_maps(depth, config) for depth in sample_depths]

    # Batched generation (CPU fallback)
    batched_results = generate_pbr_maps_batched(sample_depths, config, device="cpu")

    assert len(batched_results) == len(sequential_results)

    # Compare results (allow small numerical differences)
    for seq, batch in zip(sequential_results, batched_results):
        seq_normal, seq_roughness, seq_ao = seq
        batch_normal, batch_roughness, batch_ao = batch

        # Shape check
        assert seq_normal.shape == batch_normal.shape
        assert seq_roughness.shape == batch_roughness.shape
        assert seq_ao.shape == batch_ao.shape

        # Numerical check (within tolerance for float32 precision)
        np.testing.assert_allclose(seq_normal, batch_normal, rtol=1e-5, atol=2)
        np.testing.assert_allclose(seq_roughness, batch_roughness, rtol=1e-5, atol=2)
        np.testing.assert_allclose(seq_ao, batch_ao, rtol=1e-5, atol=2)


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64", reason="MPS only available on Apple Silicon"
)
def test_pbr_batched_gpu_acceleration(sample_depths):
    """Test GPU-accelerated PBR batching on MPS."""
    try:
        import torch

        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")
    except ImportError:
        pytest.skip("torch not available")

    config = PBRConfig()

    # Should not raise on MPS device
    results = generate_pbr_maps_batched(sample_depths, config, device="mps")

    assert len(results) == len(sample_depths)
    for normal, roughness, ao in results:
        assert normal.dtype == np.uint8
        assert roughness.dtype == np.uint8
        assert ao.dtype == np.uint8


def test_pbr_batched_fallback_no_torch(sample_depths):
    """Test batched PBR falls back to sequential when torch unavailable."""
    with patch("src.transformation_portal.lux_depth_v3.pbr.TORCH_AVAILABLE", False):
        config = PBRConfig()
        results = generate_pbr_maps_batched(sample_depths, config, device="mps")

        # Should still work via fallback
        assert len(results) == len(sample_depths)


# ============================================================================
# Test 3: MessagePack Manifests
# ============================================================================


def test_msgpack_config_flag():
    """Test MessagePack configuration flag."""
    config = EnhanceConfig(use_msgpack_manifests=True)
    assert config.use_msgpack_manifests is True

    config_default = EnhanceConfig()
    assert config_default.use_msgpack_manifests is False  # Opt-in


def test_manifest_msgpack_save_load(temp_dir):
    """Test MessagePack manifest save and load round-trip."""
    try:
        import msgpack
    except ImportError:
        pytest.skip("msgpack not available")

    # Create manifest
    manifest = CombinedManifest()
    manifest.input = InputMetadata(
        image_path="/test/input.jpg", image_sha256="abc123", image_size_bytes=1024, image_dimensions=(512, 512)
    )
    manifest.start_time = "2024-01-01T00:00:00Z"
    manifest.end_time = "2024-01-01T00:01:00Z"

    # Save as MessagePack
    msgpack_path = temp_dir / "manifest.msgpack"
    manifest.save_msgpack(msgpack_path)

    assert msgpack_path.exists()

    # Load back
    loaded = CombinedManifest.load_msgpack(msgpack_path)

    assert loaded.input is not None
    assert loaded.input.image_path == "/test/input.jpg"
    assert loaded.input.image_sha256 == "abc123"
    assert loaded.start_time == "2024-01-01T00:00:00Z"
    assert loaded.end_time == "2024-01-01T00:01:00Z"


def test_manifest_msgpack_fallback_no_msgpack(temp_dir):
    """Test MessagePack save falls back to JSON when msgpack unavailable."""
    with patch("src.transformation_portal.lux_depth_v3.manifest.MSGPACK_AVAILABLE", False):
        manifest = CombinedManifest()
        manifest.input = InputMetadata(image_path="/test/input.jpg")

        msgpack_path = temp_dir / "manifest.msgpack"
        manifest.save_msgpack(msgpack_path)

        # Should create JSON instead
        json_path = temp_dir / "manifest.json"
        assert json_path.exists()


def test_manifest_auto_load_json(temp_dir):
    """Test auto-loading JSON manifests."""
    manifest = CombinedManifest()
    manifest.input = InputMetadata(image_path="/test/input.jpg")

    json_path = temp_dir / "manifest.json"
    manifest.save(json_path)

    # Auto-load should detect JSON
    loaded = CombinedManifest.load_auto(json_path)
    assert loaded.input.image_path == "/test/input.jpg"


def test_manifest_auto_load_msgpack(temp_dir):
    """Test auto-loading MessagePack manifests."""
    try:
        import msgpack
    except ImportError:
        pytest.skip("msgpack not available")

    manifest = CombinedManifest()
    manifest.input = InputMetadata(image_path="/test/input.jpg")

    msgpack_path = temp_dir / "manifest.msgpack"
    manifest.save_msgpack(msgpack_path)

    # Auto-load should detect MessagePack
    loaded = CombinedManifest.load_auto(msgpack_path)
    assert loaded.input.image_path == "/test/input.jpg"


# ============================================================================
# Test 4: xxHash Output Keys
# ============================================================================


def test_xxhash_config_flag():
    """Test xxHash configuration flag."""
    config = EnhanceConfig(use_xxhash=True)
    assert config.use_xxhash is True

    config_default = EnhanceConfig()
    assert config_default.use_xxhash is False  # Opt-in


def test_output_key_sha1_default(temp_dir):
    """Test output key generation with SHA-1 (default)."""
    input_path = temp_dir / "photos" / "scene1" / "image.jpg"
    input_root = temp_dir / "photos"

    key = make_output_key(input_path, input_root, use_xxhash=False)

    # Should contain stem, extension, and hash
    assert "image" in key.name
    assert "jpg" in key.name
    assert len(key.name.split("_")[-1]) == 8  # 8-char hash


def test_output_key_xxhash(temp_dir):
    """Test output key generation with xxHash."""
    try:
        import xxhash
    except ImportError:
        pytest.skip("xxhash not available")

    input_path = temp_dir / "photos" / "scene1" / "image.jpg"
    input_root = temp_dir / "photos"

    key_sha1 = make_output_key(input_path, input_root, use_xxhash=False)
    key_xxhash = make_output_key(input_path, input_root, use_xxhash=True)

    # Both should have same structure but different hashes
    assert key_sha1.parent == key_xxhash.parent
    assert "image" in key_xxhash.name
    assert "jpg" in key_xxhash.name

    # Hashes should differ
    hash_sha1 = key_sha1.name.split("_")[-1]
    hash_xxhash = key_xxhash.name.split("_")[-1]
    assert hash_sha1 != hash_xxhash


def test_output_key_xxhash_fallback(temp_dir):
    """Test xxHash falls back to SHA-1 when unavailable."""
    with patch("src.transformation_portal.lux_depth_v3.orchestrator.XXHASH_AVAILABLE", False):
        input_path = temp_dir / "photos" / "image.jpg"
        input_root = temp_dir / "photos"

        # Should not raise, falls back to SHA-1
        key = make_output_key(input_path, input_root, use_xxhash=True)
        assert "image" in key.name


# ============================================================================
# Test 5: Phase 3 Config Integration
# ============================================================================


def test_enhance_config_phase3_defaults():
    """Test Phase 3 configuration defaults."""
    config = EnhanceConfig()

    # Phase 3 features should default to opt-in (False)
    assert config.use_coreml_backend is False
    assert config.enable_pbr_gpu_batching is False
    assert config.use_msgpack_manifests is False
    assert config.use_xxhash is False


def test_enhance_config_phase3_enabled():
    """Test Phase 3 configuration can be enabled."""
    config = EnhanceConfig(use_coreml_backend=True, enable_pbr_gpu_batching=True, use_msgpack_manifests=True, use_xxhash=True)

    assert config.use_coreml_backend is True
    assert config.enable_pbr_gpu_batching is True
    assert config.use_msgpack_manifests is True
    assert config.use_xxhash is True


# ============================================================================
# Test 6: Backward Compatibility
# ============================================================================


def test_phase3_backward_compatible_pbr():
    """Test Phase 3 PBR batching is backward compatible."""
    depth = np.random.rand(256, 256).astype(np.float32)
    config = PBRConfig()

    # Old API should still work
    normal, roughness, ao = generate_pbr_maps(depth, config)

    assert normal.shape == (256, 256, 3)
    assert roughness.shape == (256, 256)
    assert ao.shape == (256, 256)


def test_phase3_backward_compatible_manifest(temp_dir):
    """Test Phase 3 manifests remain JSON compatible."""
    manifest = CombinedManifest()
    manifest.input = InputMetadata(image_path="/test/input.jpg")

    json_path = temp_dir / "manifest.json"
    manifest.save(json_path)

    # Old load() method should still work
    loaded = CombinedManifest.load(json_path)
    assert loaded.input.image_path == "/test/input.jpg"


# ============================================================================
# Test 7: Error Handling
# ============================================================================


def test_pbr_batched_invalid_device():
    """Test batched PBR with invalid device falls back gracefully."""
    depth = np.random.rand(128, 128).astype(np.float32)

    # Invalid device should fall back to CPU
    results = generate_pbr_maps_batched([depth], PBRConfig(), device="invalid_device")

    assert len(results) == 1


def test_manifest_msgpack_load_missing_file(temp_dir):
    """Test loading missing MessagePack file raises error."""
    try:
        import msgpack
    except ImportError:
        pytest.skip("msgpack not available")

    missing_path = temp_dir / "missing.msgpack"

    with pytest.raises(FileNotFoundError):
        CombinedManifest.load_msgpack(missing_path)


# ============================================================================
# Test 8: Performance Validation (Smoke Tests)
# ============================================================================


@pytest.mark.benchmark
def test_pbr_batched_faster_than_sequential(sample_depths):
    """Smoke test: batched PBR should not be slower than sequential."""
    import time

    config = PBRConfig()

    # Sequential
    start = time.time()
    _ = [generate_pbr_maps(depth, config) for depth in sample_depths]
    seq_time = time.time() - start

    # Batched (CPU, should have minimal overhead)
    start = time.time()
    _ = generate_pbr_maps_batched(sample_depths, config, device="cpu")
    batch_time = time.time() - start

    # Batched should not be significantly slower (allow 2x overhead for setup)
    assert batch_time < seq_time * 2.0


@pytest.mark.benchmark
def test_xxhash_faster_than_sha1():
    """Smoke test: xxHash should not be slower than SHA-1."""
    try:
        import xxhash
    except ImportError:
        pytest.skip("xxhash not available")

    import hashlib
    import time

    test_data = b"test_path/to/image.jpg" * 1000

    # SHA-1
    start = time.time()
    for _ in range(1000):
        hashlib.sha1(test_data).hexdigest()[:8]
    sha1_time = time.time() - start

    # xxHash
    start = time.time()
    for _ in range(1000):
        xxhash.xxh64(test_data).hexdigest()[:8]
    xxhash_time = time.time() - start

    # xxHash should be faster (allow equal time, not slower)
    assert xxhash_time <= sha1_time


# ============================================================================
# Test 9: Integration Tests
# ============================================================================


def test_phase3_full_pipeline_dry_run():
    """Integration test: verify Phase 3 config can be constructed."""
    config = EnhanceConfig(
        # Phase 1
        enable_manifest_cache=True,
        chunked_hashing=True,
        # Phase 2
        enable_parallel_processing=True,
        enable_depth_cache=False,
        # Phase 3
        use_coreml_backend=False,  # Safe on all platforms
        enable_pbr_gpu_batching=False,
        use_msgpack_manifests=False,
        use_xxhash=False,
    )

    # Should not raise
    assert config is not None
    assert config.enable_manifest_cache is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
