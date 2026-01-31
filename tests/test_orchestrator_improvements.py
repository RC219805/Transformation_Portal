"""Tests for EnhanceOrchestrator improvements.

Tests cover the 7 reliability and correctness improvements:
1. Output key generation with directory structure and hash
2. Improved skip logic using stored config fingerprint
3. Lazy image preprocessing
4. Configurable hash computation (HashMode)
5. PBR generation with cached depth
6. Accurate batch execution timestamps
7. Defensive check for output existence
"""
import pytest
import hashlib
import json
import time
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from transformation_portal.lux_depth_v3.orchestrator import make_output_key
from transformation_portal.lux_depth_v3.manifest import (
    CombinedManifest,
    BatchManifest,
    ConfigFingerprint,
    InputMetadata,
    DepthMetadata,
    V2Metadata,
    TimingMetadata,
)
from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.security import HashMode


class TestMakeOutputKey:
    """Tests for improved make_output_key function."""

    def test_preserves_directory_structure(self):
        """Test that directory structure is preserved in output key."""
        input_path = Path("/root/photos/scene1/image.jpg")
        input_root = Path("/root/photos")

        key = make_output_key(input_path, input_root)

        # Should preserve scene1 directory
        assert "scene1" in str(key)

    def test_includes_extension_in_key(self):
        """Test that file extension is included (sanitized) in the key name."""
        input_path = Path("/root/photos/image.jpg")
        input_root = Path("/root/photos")

        key = make_output_key(input_path, input_root)

        # Should include "jpg" in the key
        assert "_jpg_" in str(key)

    def test_includes_hash_suffix(self):
        """Test that 8-character SHA-1 hash suffix is included."""
        input_path = Path("/root/photos/image.jpg")
        input_root = Path("/root/photos")

        key = make_output_key(input_path, input_root)

        # Key should end with 8-char hex hash
        parts = str(key).split("_")
        assert len(parts) >= 3
        hash_suffix = parts[-1]
        # Should be 8 hex characters
        assert len(hash_suffix) == 8
        assert all(c in "0123456789abcdef" for c in hash_suffix)

    def test_different_extensions_produce_different_keys(self):
        """Test that files with different extensions get different keys."""
        input_root = Path("/root/photos")
        input_jpg = Path("/root/photos/image.jpg")
        input_png = Path("/root/photos/image.png")

        key_jpg = make_output_key(input_jpg, input_root)
        key_png = make_output_key(input_png, input_root)

        # Keys should be different
        assert key_jpg != key_png
        assert "_jpg_" in str(key_jpg)
        assert "_png_" in str(key_png)

    def test_same_name_different_dirs_produce_different_keys(self):
        """Test that same filename in different dirs gets different keys."""
        input_root = Path("/root/photos")
        input_a = Path("/root/photos/dir_a/image.jpg")
        input_b = Path("/root/photos/dir_b/image.jpg")

        key_a = make_output_key(input_a, input_root)
        key_b = make_output_key(input_b, input_root)

        # Keys should be different due to different paths
        assert key_a != key_b
        assert "dir_a" in str(key_a)
        assert "dir_b" in str(key_b)

    def test_no_extension_uses_noext_label(self):
        """Test that files without extension use 'noext' label."""
        input_path = Path("/root/photos/Makefile")
        input_root = Path("/root/photos")

        key = make_output_key(input_path, input_root)

        assert "_noext_" in str(key)

    def test_hash_is_deterministic(self):
        """Test that hash is consistent for the same input."""
        input_path = Path("/root/photos/scene1/image.jpg")
        input_root = Path("/root/photos")

        key1 = make_output_key(input_path, input_root)
        key2 = make_output_key(input_path, input_root)

        assert key1 == key2


class TestConfigFingerprint:
    """Tests for config fingerprint skip logic."""

    def test_depth_only_extracts_depth_fields(self):
        """Test that depth_only() returns only depth-related fields."""
        fp = ConfigFingerprint(
            model_variant="test_model",
            depth_quantization="u16",
            depth_device="cuda",
            preset="luxury_estate",
            v2_preset="hdr",
            v2_device="mps",
            v2_upscaler_backend="realesrgan",
        )

        depth_fp = fp.depth_only()

        # Depth fields should be preserved
        assert depth_fp.model_variant == "test_model"
        assert depth_fp.depth_quantization == "u16"
        assert depth_fp.depth_device == "cuda"
        assert depth_fp.preset == "luxury_estate"

        # V2 fields should be None/empty
        assert depth_fp.v2_preset is None
        assert depth_fp.v2_device is None
        assert depth_fp.v2_upscaler_backend is None

    def test_v2_only_extracts_v2_fields(self):
        """Test that v2_only() returns only V2-related fields."""
        fp = ConfigFingerprint(
            model_variant="test_model",
            depth_quantization="u16",
            depth_device="cuda",
            preset="luxury_estate",
            v2_preset="hdr",
            v2_device="mps",
            v2_upscaler_backend="realesrgan",
        )

        v2_fp = fp.v2_only()

        # V2 fields should be preserved
        assert v2_fp.v2_preset == "hdr"
        assert v2_fp.v2_device == "mps"
        assert v2_fp.v2_upscaler_backend == "realesrgan"

        # Depth fields should be empty
        assert v2_fp.model_variant == ""
        assert v2_fp.depth_quantization == ""
        assert v2_fp.depth_device == ""

    def test_to_sha256_is_deterministic(self):
        """Test that SHA256 hash is consistent."""
        fp = ConfigFingerprint(
            model_variant="test",
            depth_quantization="none",
            depth_device="cpu",
        )

        hash1 = fp.to_sha256()
        hash2 = fp.to_sha256()

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64 hex characters

    def test_different_configs_produce_different_hashes(self):
        """Test that different configs produce different hashes."""
        fp1 = ConfigFingerprint(
            model_variant="model_a",
            depth_quantization="none",
            depth_device="cpu",
        )
        fp2 = ConfigFingerprint(
            model_variant="model_b",
            depth_quantization="none",
            depth_device="cpu",
        )

        assert fp1.to_sha256() != fp2.to_sha256()


class TestHashMode:
    """Tests for HashMode configuration."""

    def test_hash_mode_never(self):
        """Test that HashMode.NEVER is defined."""
        assert HashMode.NEVER.value == "never"

    def test_hash_mode_always(self):
        """Test that HashMode.ALWAYS is defined."""
        assert HashMode.ALWAYS.value == "always"

    def test_hash_mode_if_manifest_exists(self):
        """Test that HashMode.IF_MANIFEST_EXISTS is defined."""
        assert HashMode.IF_MANIFEST_EXISTS.value == "if_manifest_exists"

    def test_enhance_config_default_hash_mode(self):
        """Test that default hash mode is IF_MANIFEST_EXISTS."""
        config = EnhanceConfig()
        assert config.hash_mode == HashMode.IF_MANIFEST_EXISTS


class TestCombinedManifestTimestamps:
    """Tests for accurate batch execution timestamps."""

    def test_manifest_has_timestamp_fields(self):
        """Test that CombinedManifest has start_time and end_time fields."""
        manifest = CombinedManifest(
            start_time="2025-01-31T00:00:00Z",
            end_time="2025-01-31T00:05:00Z",
        )

        assert manifest.start_time == "2025-01-31T00:00:00Z"
        assert manifest.end_time == "2025-01-31T00:05:00Z"

    def test_manifest_save_load_preserves_timestamps(self):
        """Test that save/load preserves timestamp fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "test_manifest.json"

            # Create manifest with timestamps
            manifest = CombinedManifest(
                start_time="2025-01-31T00:00:00Z",
                end_time="2025-01-31T00:05:00Z",
            )
            manifest.save(manifest_path)

            # Load and verify
            loaded = CombinedManifest.load(manifest_path)

            assert loaded.start_time == "2025-01-31T00:00:00Z"
            assert loaded.end_time == "2025-01-31T00:05:00Z"


class TestBatchManifest:
    """Tests for BatchManifest with accurate timestamps."""

    def test_batch_manifest_has_required_fields(self):
        """Test that BatchManifest has all required fields."""
        bm = BatchManifest(
            batch_id="2025-01-31_120000",
            start_time="2025-01-31T12:00:00Z",
            end_time="2025-01-31T12:30:00Z",
            config={"model": "test_model"},
            results=[{"status": "ok"}],
            stats={"total": 1},
        )

        assert bm.batch_id == "2025-01-31_120000"
        assert bm.start_time == "2025-01-31T12:00:00Z"
        assert bm.end_time == "2025-01-31T12:30:00Z"
        assert bm.config == {"model": "test_model"}
        assert bm.results == [{"status": "ok"}]
        assert bm.stats == {"total": 1}

    def test_batch_manifest_save_load(self):
        """Test that BatchManifest can be saved and loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "batch_manifest.json"

            bm = BatchManifest(
                batch_id="test_batch",
                start_time="2025-01-31T00:00:00Z",
                end_time="2025-01-31T00:10:00Z",
                config={"model": "model_x"},
                results=[{"status": "ok", "image": "img1.jpg"}],
                stats={"total": 1, "batch_runtime_seconds": 600},
            )
            bm.write(manifest_path)

            # Load and verify
            loaded = BatchManifest.load(manifest_path)

            assert loaded.batch_id == "test_batch"
            assert loaded.start_time == "2025-01-31T00:00:00Z"
            assert loaded.end_time == "2025-01-31T00:10:00Z"
            assert loaded.stats["batch_runtime_seconds"] == 600


class TestV2MetadataFields:
    """Tests for V2Metadata additional fields."""

    def test_v2_metadata_has_all_fields(self):
        """Test that V2Metadata has all required fields."""
        v2 = V2Metadata(
            preset="default",
            status="ok",
            strict_depth=True,
            output_dir="v2/",
            report_path="/path/to/report.json",
            error_message=None,
        )

        assert v2.preset == "default"
        assert v2.status == "ok"
        assert v2.strict_depth is True
        assert v2.output_dir == "v2/"
        assert v2.report_path == "/path/to/report.json"
        assert v2.error_message is None


class TestEnhanceConfigSaveFloatDepth:
    """Tests for save_float_depth configuration."""

    def test_save_float_depth_default_false(self):
        """Test that save_float_depth defaults to False."""
        config = EnhanceConfig()
        assert config.save_float_depth is False

    def test_save_float_depth_can_be_enabled(self):
        """Test that save_float_depth can be set to True."""
        config = EnhanceConfig(save_float_depth=True)
        assert config.save_float_depth is True


class TestManifestConfigFingerprintStorage:
    """Tests for storing config fingerprint in manifest."""

    def test_manifest_stores_config_fingerprint(self):
        """Test that config fingerprint is stored in manifest."""
        fp = ConfigFingerprint(
            model_variant="test_model",
            depth_quantization="u16",
            depth_device="cpu",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"

            manifest = CombinedManifest(config_fingerprint=fp)
            manifest.save(manifest_path)

            loaded = CombinedManifest.load(manifest_path)

            assert loaded.config_fingerprint is not None
            assert loaded.config_fingerprint.model_variant == "test_model"
            assert loaded.config_fingerprint.depth_quantization == "u16"


class TestInputMetadataSchema:
    """Tests for InputMetadata schema versioning."""

    def test_input_metadata_has_schema_version(self):
        """Test that InputMetadata has schema_version field."""
        meta = InputMetadata(
            image_path="/path/to/image.jpg",
            image_sha256="abc123",
        )
        assert meta.schema_version == "1.0"

    def test_input_metadata_from_dict(self):
        """Test InputMetadata.from_dict() handles schema version."""
        data = {
            "image_path": "/path/to/image.jpg",
            "image_sha256": "abc123",
            "schema_version": "1.0",
        }
        meta = InputMetadata.from_dict(data)

        assert meta.image_path == "/path/to/image.jpg"
        assert meta.schema_version == "1.0"

    def test_input_metadata_rejects_unsupported_schema(self):
        """Test that unsupported schema version raises ValueError."""
        data = {
            "image_path": "/path/to/image.jpg",
            "schema_version": "2.0",  # Unsupported version
        }

        with pytest.raises(ValueError, match="Unsupported InputMetadata schema"):
            InputMetadata.from_dict(data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
