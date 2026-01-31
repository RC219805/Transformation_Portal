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


class TestHashModeIfManifestExistsBaselineHash:
    """Tests for HashMode.IF_MANIFEST_EXISTS baseline hash behavior (Fix 1)."""

    def test_if_manifest_exists_stores_baseline_hash(self):
        """Test that first run with IF_MANIFEST_EXISTS stores hash for future comparisons.

        This is a critical fix: on first run, we must compute and store a baseline hash
        so that future runs can detect file changes.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a test image file
            test_image = tmpdir / "test.jpg"
            test_image.write_bytes(b"fake image data")

            # Mock compute_file_sha256 to return a predictable hash
            expected_hash = "abc123def456"

            with patch('transformation_portal.lux_depth_v3.orchestrator.compute_file_sha256') as mock_hash:
                mock_hash.return_value = expected_hash

                from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

                # Create minimal config with IF_MANIFEST_EXISTS mode
                config = EnhanceConfig(hash_mode=HashMode.IF_MANIFEST_EXISTS)

                # Create orchestrator (mocking dependencies)
                with patch('transformation_portal.lux_depth_v3.orchestrator.DA3InferenceEngine'):
                    with patch('transformation_portal.lux_depth_v3.orchestrator.Postprocessor'):
                        with patch('transformation_portal.lux_depth_v3.orchestrator.V2Runner'):
                            orchestrator = EnhanceOrchestrator(config, tmpdir / "output", verify_outputs=False)

                            # First run: no manifest exists
                            # for_manifest_write=True should compute hash (establishing baseline)
                            hash_for_write = orchestrator._compute_or_skip_hash(
                                test_image,
                                manifest_exists=False,
                                saved_hash=None,
                                for_manifest_write=True
                            )

                            assert hash_for_write == expected_hash, "First run with for_manifest_write=True must compute hash"

                            # Comparison call (for skip check) on first run should NOT compute hash
                            hash_for_compare = orchestrator._compute_or_skip_hash(
                                test_image,
                                manifest_exists=False,
                                saved_hash=None,
                                for_manifest_write=False
                            )

                            assert hash_for_compare is None, "First run with for_manifest_write=False should skip hash"

                            # Second run: manifest exists with saved hash
                            # for_manifest_write=False should now compute hash for comparison
                            hash_for_compare_2nd = orchestrator._compute_or_skip_hash(
                                test_image,
                                manifest_exists=True,
                                saved_hash=expected_hash,
                                for_manifest_write=False
                            )

                            assert hash_for_compare_2nd == expected_hash, "Second run should compute hash for comparison"


class TestCachedDepthNoDoubleNormalization:
    """Tests for cached depth loading without double normalization (Fix 2)."""

    def test_cached_depth_no_double_normalization(self):
        """Test that loading cached depth doesn't double-normalize float32 values.

        This is a critical fix: if read_depth_u16_png() returns float32 in [0,1],
        dividing by 65535 again crushes the range to ~[0, 0.00002], breaking PBR maps.
        """
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test depth files
            depth_png = tmpdir / "depth.png"
            float_depth_npy = tmpdir / "depth.npy"

            # Test Case 1: Reader returns uint16 (should normalize)
            with patch('transformation_portal.lux_depth_v3.depth_writer.read_depth_u16_png') as mock_read:
                # Simulate reader returning uint16 values
                mock_read.return_value = np.array([[0, 32767, 65535]], dtype=np.uint16)

                from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator
                config = EnhanceConfig()

                with patch('transformation_portal.lux_depth_v3.orchestrator.DA3InferenceEngine'):
                    with patch('transformation_portal.lux_depth_v3.orchestrator.Postprocessor'):
                        with patch('transformation_portal.lux_depth_v3.orchestrator.V2Runner'):
                            orchestrator = EnhanceOrchestrator(config, tmpdir / "output", verify_outputs=False)

                            # Create the PNG file so exists() returns True
                            depth_png.write_bytes(b"fake png")

                            result = orchestrator._load_cached_depth(depth_png, float_depth_npy)

                            assert result is not None, "Should load depth data"
                            assert result.dtype == np.float32, "Should convert to float32"
                            # Check normalization is correct
                            assert np.allclose(result, [0.0, 0.5, 1.0], atol=0.01), f"Expected [0, 0.5, 1], got {result}"

            # Test Case 2: Reader returns pre-normalized float32 (should NOT double-normalize)
            with patch('transformation_portal.lux_depth_v3.depth_writer.read_depth_u16_png') as mock_read:
                # Simulate reader returning already normalized float32 values
                mock_read.return_value = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)

                config = EnhanceConfig()

                with patch('transformation_portal.lux_depth_v3.orchestrator.DA3InferenceEngine'):
                    with patch('transformation_portal.lux_depth_v3.orchestrator.Postprocessor'):
                        with patch('transformation_portal.lux_depth_v3.orchestrator.V2Runner'):
                            orchestrator = EnhanceOrchestrator(config, tmpdir / "output", verify_outputs=False)

                            result = orchestrator._load_cached_depth(depth_png, float_depth_npy)

                            assert result is not None, "Should load depth data"
                            # Values should remain in [0, 1] range (not crushed to near-zero)
                            assert result[0, 1] > 0.4 and result[0, 1] < 0.6, f"Expected ~0.5, got {result[0, 1]} (double normalization bug)"
                            assert result[0, 2] > 0.9, f"Expected ~1.0, got {result[0, 2]} (double normalization bug)"


class TestV2SkipIndependentOfGeneratePBR:
    """Tests for V2 skip logic independent of generate_pbr flag (Fix 3)."""

    def test_v2_skip_independent_of_generate_pbr(self):
        """Test that should_skip_v2() evaluates V2 enhancement independent of PBR generation.

        This is a critical fix: V2 enhancement and PBR generation are separate stages.
        The should_skip_v2() method should evaluate V2 config changes and V2 output existence,
        not gate on generate_pbr flag.
        """
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test files
            test_image = tmpdir / "test.jpg"
            test_image.write_bytes(b"fake image")
            v2_report = tmpdir / "v2_report.json"
            v2_report.write_text('{"status": "ok"}')
            manifest_path = tmpdir / "manifest.json"

            # Create manifest with V2 metadata and config fingerprint
            # Use matching config values to ensure fingerprint matches
            manifest = CombinedManifest(
                config_fingerprint=ConfigFingerprint(
                    model_variant="test",
                    depth_quantization="u16",
                    depth_device="cpu",
                    v2_preset="default",
                    v2_device="cpu",
                    v2_upscaler_backend="default",  # Match EnhanceConfig default
                ),
                v2=V2Metadata(
                    preset="default",
                    status="ok",
                    strict_depth=True,
                    output_dir="v2/",
                    report_path=str(v2_report),
                ),
            )
            manifest.save(manifest_path)

            # Test with generate_pbr=False
            config_no_pbr = EnhanceConfig(generate_pbr=False, v2_preset="default")

            with patch('transformation_portal.lux_depth_v3.orchestrator.DA3InferenceEngine'):
                with patch('transformation_portal.lux_depth_v3.orchestrator.Postprocessor'):
                    with patch('transformation_portal.lux_depth_v3.orchestrator.V2Runner'):
                        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

                        orchestrator = EnhanceOrchestrator(config_no_pbr, tmpdir / "output", verify_outputs=False)

                        # V2 should be skippable even without PBR enabled
                        # (V2 outputs are valid, config matches)
                        image_input = ImageInput(test_image)
                        skip = orchestrator.should_skip_v2(
                            v2_report,
                            manifest_path,
                            image_input,
                            depth_was_skipped=True
                        )

                        # Fix verification: should evaluate V2 independently
                        # V2 outputs exist and config matches, so should skip
                        assert skip is True, "should_skip_v2() should evaluate V2 independently of generate_pbr"

                        # Test with changed V2 config - should NOT skip
                        config_changed = EnhanceConfig(generate_pbr=False, v2_preset="different_preset")
                        orchestrator_changed = EnhanceOrchestrator(config_changed, tmpdir / "output", verify_outputs=False)

                        skip_changed = orchestrator_changed.should_skip_v2(
                            v2_report,
                            manifest_path,
                            image_input,
                            depth_was_skipped=True
                        )

                        assert skip_changed is False, "Changed V2 config should invalidate skip"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
