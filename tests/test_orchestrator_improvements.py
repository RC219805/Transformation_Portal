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

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]

from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.manifest import (
    BackendSelectionMetadata,
    BatchManifest,
    CombinedManifest,
    ConfigFingerprint,
    DepthMetadata,
    InputMetadata,
    MaterialsV3Metadata,
    V2Metadata,
)
from transformation_portal.lux_depth_v3.orchestrator import make_output_key
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
        """Test that depth_only() returns Stage A fields only."""
        fp = ConfigFingerprint(
            model_variant="test_model",
            depth_quantization="u16",
            depth_device="cuda",
            preset="luxury_estate",
            v2_preset="hdr",
            v2_device="mps",
            v2_upscaler_backend="realesrgan",
            depth_backend="depth_pro",
            quality_tier="apex",
            materials_config={"enable_materials_v3": True},
            pbr_config={"generate_pbr": True},
            apex_depth_gate_config={"min_upper_iqr": 1e-4},
            output_bit_depth=16,
            enable_v2=True,
        )

        depth_fp = fp.depth_only()

        # Stage A fields should be preserved
        assert depth_fp.model_variant == "test_model"
        assert depth_fp.depth_quantization == "u16"
        assert depth_fp.depth_device == "cuda"
        assert depth_fp.preset == "luxury_estate"
        assert depth_fp.depth_backend == "depth_pro"
        assert depth_fp.quality_tier == "apex"
        assert depth_fp.materials_config == {"enable_materials_v3": True}
        assert depth_fp.pbr_config == {"generate_pbr": True}
        assert depth_fp.apex_depth_gate_config == {"min_upper_iqr": 1e-4}
        assert depth_fp.output_bit_depth == 16

        # V2 fields should be None/empty
        assert depth_fp.v2_preset is None
        assert depth_fp.v2_device is None
        assert depth_fp.v2_upscaler_backend is None
        assert depth_fp.enable_v2 is None

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
            depth_backend="depth_pro",
            quality_tier="apex",
            materials_config={"enable_materials_v3": True},
            pbr_config={"generate_pbr": True},
            apex_depth_gate_config={"min_upper_iqr": 1e-4},
            output_bit_depth=16,
            enable_v2=True,
        )

        v2_fp = fp.v2_only()

        # V2 fields should be preserved
        assert v2_fp.v2_preset == "hdr"
        assert v2_fp.v2_device == "mps"
        assert v2_fp.v2_upscaler_backend == "realesrgan"
        assert v2_fp.output_bit_depth == 16
        assert v2_fp.enable_v2 is True

        # Stage A fields should be empty
        assert v2_fp.model_variant == ""
        assert v2_fp.depth_quantization == ""
        assert v2_fp.depth_device == ""
        assert v2_fp.depth_backend is None
        assert v2_fp.materials_config is None
        assert v2_fp.pbr_config is None
        assert v2_fp.apex_depth_gate_config is None

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

    def test_manifest_save_load_preserves_timestamps(self, temp_workspace):
        """Test that save/load preserves timestamp fields."""
        tmpdir = temp_workspace["root"]
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

    def test_batch_manifest_save_load(self, temp_workspace):
        """Test that BatchManifest can be saved and loaded."""
        tmpdir = temp_workspace["root"]
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

    def test_batch_manifest_routes_legacy_json_bytes_through_atomic_writer(self, tmp_path):
        """The durable migration must not change the prior UTF-8 JSON bytes."""
        manifest_path = tmp_path / "batch_manifest.json"
        manifest = BatchManifest(
            batch_id="batch-α",
            start_time="2026-08-31T00:00:00Z",
            end_time="2026-08-31T00:01:00Z",
            config={"preset": "café"},
            results=[{"status": "ok"}],
            stats={"total": 1},
        )
        expected = """{
  "batch_id": "batch-α",
  "config": {
    "preset": "café"
  },
  "end_time": "2026-08-31T00:01:00Z",
  "results": [
    {
      "status": "ok"
    }
  ],
  "start_time": "2026-08-31T00:00:00Z",
  "stats": {
    "total": 1
  }
}""".encode("utf-8")

        with patch("transformation_portal.lux_depth_v3.manifest.atomic_write_bytes") as durable_write:
            manifest.write(manifest_path)

        durable_write.assert_called_once_with(manifest_path, expected)

    def test_batch_manifest_serialization_failure_preserves_prior_file(self, tmp_path):
        """Serialization must complete before the destination can be touched."""
        manifest_path = tmp_path / "batch_manifest.json"
        manifest_path.write_bytes(b"prior-valid-manifest")
        manifest = BatchManifest(
            batch_id="bad",
            start_time="start",
            end_time="end",
            config={"invalid": float("nan")},
            results=[],
            stats={},
        )

        with (
            patch("transformation_portal.lux_depth_v3.manifest.atomic_write_bytes") as durable_write,
            pytest.raises(ValueError, match="Out of range float values"),
        ):
            manifest.write(manifest_path)

        durable_write.assert_not_called()
        assert manifest_path.read_bytes() == b"prior-valid-manifest"


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

    def test_manifest_stores_config_fingerprint(self, temp_workspace):
        """Test that config fingerprint is stored in manifest."""
        fp = ConfigFingerprint(
            model_variant="test_model",
            depth_quantization="u16",
            depth_device="cpu",
        )

        tmpdir = temp_workspace["root"]
        manifest_path = Path(tmpdir) / "manifest.json"

        manifest = CombinedManifest(config_fingerprint=fp)
        manifest.save(manifest_path)

        loaded = CombinedManifest.load(manifest_path)

        assert loaded.config_fingerprint is not None
        assert loaded.config_fingerprint.model_variant == "test_model"
        assert loaded.config_fingerprint.depth_quantization == "u16"


class TestFingerprintDrivenSkipInvalidation:
    """Tests for expanded config-fingerprint invalidation paths."""

    def test_should_skip_depth_invalidates_on_backend_change(self, temp_workspace):
        """Changing depth backend should invalidate Stage A reuse."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        tmpdir = Path(temp_workspace["root"])
        test_image = tmpdir / "backend_change.jpg"
        test_image.write_bytes(b"fake image")
        depth_path = tmpdir / "depth.png"
        depth_path.write_bytes(b"depth")
        manifest_path = tmpdir / "manifest.json"

        base_config = EnhanceConfig(depth_backend="da3", enable_v2=False)
        changed_config = EnhanceConfig(depth_backend="depth_pro", enable_v2=False)

        with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry"):
            base_orchestrator = EnhanceOrchestrator(base_config, tmpdir / "output_base", verify_outputs=False)
            manifest = CombinedManifest(
                config_fingerprint=base_orchestrator.compute_config_fingerprint(),
                depth=DepthMetadata(model="da3", depth_path=str(depth_path), runtime_seconds=0.1, scaling={}),
            )
            manifest.save(manifest_path)

            changed_orchestrator = EnhanceOrchestrator(
                changed_config,
                tmpdir / "output_changed",
                verify_outputs=False,
            )

        should_skip = changed_orchestrator.should_skip_depth(
            depth_path,
            manifest_path,
            ImageInput(test_image),
        )

        assert should_skip is False

    def test_should_skip_depth_invalidates_on_materials_change(self, temp_workspace):
        """Enabling Materials V3 should invalidate cached Stage A reuse."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        tmpdir = Path(temp_workspace["root"])
        test_image = tmpdir / "materials_change.jpg"
        test_image.write_bytes(b"fake image")
        depth_path = tmpdir / "materials_depth.png"
        depth_path.write_bytes(b"depth")
        manifest_path = tmpdir / "materials_manifest.json"

        base_config = EnhanceConfig(enable_materials_v3=False, enable_v2=False)
        changed_config = EnhanceConfig(enable_materials_v3=True, enable_v2=False)

        with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry"):
            base_orchestrator = EnhanceOrchestrator(base_config, tmpdir / "output_base", verify_outputs=False)
            manifest = CombinedManifest(
                config_fingerprint=base_orchestrator.compute_config_fingerprint(),
                depth=DepthMetadata(model="da3", depth_path=str(depth_path), runtime_seconds=0.1, scaling={}),
            )
            manifest.save(manifest_path)

            changed_orchestrator = EnhanceOrchestrator(
                changed_config,
                tmpdir / "output_changed",
                verify_outputs=False,
            )

        should_skip = changed_orchestrator.should_skip_depth(
            depth_path,
            manifest_path,
            ImageInput(test_image),
        )

        assert should_skip is False

    def test_should_skip_depth_invalidates_on_pbr_change(self, temp_workspace):
        """Changing PBR parameters should invalidate Stage A reuse."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        tmpdir = Path(temp_workspace["root"])
        test_image = tmpdir / "pbr_change.jpg"
        test_image.write_bytes(b"fake image")
        depth_path = tmpdir / "pbr_depth.png"
        depth_path.write_bytes(b"depth")
        manifest_path = tmpdir / "pbr_manifest.json"

        base_config = EnhanceConfig(generate_pbr=True, pbr_normal_strength=1.0, enable_v2=False)
        changed_config = EnhanceConfig(generate_pbr=True, pbr_normal_strength=2.0, enable_v2=False)

        with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry"):
            base_orchestrator = EnhanceOrchestrator(base_config, tmpdir / "output_base", verify_outputs=False)
            manifest = CombinedManifest(
                config_fingerprint=base_orchestrator.compute_config_fingerprint(),
                depth=DepthMetadata(model="da3", depth_path=str(depth_path), runtime_seconds=0.1, scaling={}),
            )
            manifest.save(manifest_path)

            changed_orchestrator = EnhanceOrchestrator(
                changed_config,
                tmpdir / "output_changed",
                verify_outputs=False,
            )

        should_skip = changed_orchestrator.should_skip_depth(
            depth_path,
            manifest_path,
            ImageInput(test_image),
        )

        assert should_skip is False

    def test_should_skip_depth_invalidates_legacy_manifest_without_stage_a_groups(self, temp_workspace):
        """Legacy manifests without expanded fingerprint fields should not skip."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        tmpdir = Path(temp_workspace["root"])
        test_image = tmpdir / "legacy_manifest.jpg"
        test_image.write_bytes(b"fake image")
        depth_path = tmpdir / "legacy_depth.png"
        depth_path.write_bytes(b"depth")
        manifest_path = tmpdir / "legacy_manifest.json"

        legacy_fingerprint = ConfigFingerprint(
            model_variant="depth-anything-v3-metric-large",
            depth_quantization="none",
            depth_device="cpu",
            preset=None,
            v2_preset="default",
            v2_device="cpu",
            v2_upscaler_backend="default",
        )
        manifest = CombinedManifest(
            config_fingerprint=legacy_fingerprint,
            depth=DepthMetadata(model="da3", depth_path=str(depth_path), runtime_seconds=0.1, scaling={}),
        )
        manifest.save(manifest_path)

        with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry"):
            orchestrator = EnhanceOrchestrator(EnhanceConfig(enable_v2=False), tmpdir / "output", verify_outputs=False)

        should_skip = orchestrator.should_skip_depth(
            depth_path,
            manifest_path,
            ImageInput(test_image),
        )

        assert should_skip is False

    def test_legacy_manifest_invalidates_once_then_restabilizes_after_rewrite(self, temp_workspace):
        """Legacy manifests should invalidate once, then stabilize after rewrite."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        tmpdir = Path(temp_workspace["root"])
        test_image = tmpdir / "legacy_restabilize.jpg"
        test_image.write_bytes(b"fake image")
        depth_path = tmpdir / "legacy_restabilize_depth.png"
        depth_path.write_bytes(b"depth")
        manifest_path = tmpdir / "legacy_restabilize_manifest.json"
        image_input = ImageInput(test_image)

        legacy_fingerprint = ConfigFingerprint(
            model_variant="depth-anything-v3-metric-large",
            depth_quantization="none",
            depth_device="cpu",
            preset=None,
            v2_preset="default",
            v2_device="cpu",
            v2_upscaler_backend="default",
        )
        CombinedManifest(
            config_fingerprint=legacy_fingerprint,
            depth=DepthMetadata(model="da3", depth_path=str(depth_path), runtime_seconds=0.1, scaling={}),
        ).save(manifest_path)

        with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry"):
            orchestrator = EnhanceOrchestrator(EnhanceConfig(enable_v2=False), tmpdir / "output", verify_outputs=False)

        assert orchestrator.should_skip_depth(depth_path, manifest_path, image_input) is False

        CombinedManifest(
            config_fingerprint=orchestrator.compute_config_fingerprint(),
            depth=DepthMetadata(model="da3", depth_path=str(depth_path), runtime_seconds=0.1, scaling={}),
            backend_selection=orchestrator._capture_backend_metadata(),
        ).save(manifest_path)

        assert orchestrator.should_skip_depth(depth_path, manifest_path, image_input) is True

    def test_should_skip_depth_invalidates_on_resolved_backend_drift(self, temp_workspace):
        """Resolved backend provenance drift should deny manifest reuse."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        tmpdir = Path(temp_workspace["root"])
        test_image = tmpdir / "resolved_backend_drift.jpg"
        test_image.write_bytes(b"fake image")
        depth_path = tmpdir / "resolved_backend_drift_depth.png"
        depth_path.write_bytes(b"depth")
        manifest_path = tmpdir / "resolved_backend_drift_manifest.json"

        with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry"):
            orchestrator = EnhanceOrchestrator(EnhanceConfig(enable_v2=False), tmpdir / "output", verify_outputs=False)

        manifest = CombinedManifest(
            config_fingerprint=orchestrator.compute_config_fingerprint(),
            depth=DepthMetadata(model="da3", depth_path=str(depth_path), runtime_seconds=0.1, scaling={}),
            backend_selection=BackendSelectionMetadata(
                requested_backend="da3",
                resolved_backend="da2",
                resolution_status="fallback",
                resolution_reason="Simulated backend drift",
                model_id="depth-anything/Depth-Anything-V2-Small-hf",
                device="cpu",
                attempts=[],
            ),
        )
        manifest.save(manifest_path)

        should_skip = orchestrator.should_skip_depth(
            depth_path,
            manifest_path,
            ImageInput(test_image),
        )

        assert should_skip is False

    def test_unprepared_cache_configuration_is_rejected(self, temp_workspace):
        """Legacy direct construction cannot silently disable an enabled cache."""
        from transformation_portal.lux_depth_v3.execution_plan_adapter import LuxExecutionPlanAuthorityError
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        output_root = Path(temp_workspace["root"]) / "output"
        with pytest.raises(LuxExecutionPlanAuthorityError, match="from_prepared"):
            EnhanceOrchestrator(
                EnhanceConfig(enable_v2=False, enable_depth_cache=True),
                output_root,
                verify_outputs=False,
            )

        assert not (output_root / ".depth_cache").exists()
        assert not (output_root / "manifests").exists()

    def test_should_skip_v2_invalidates_on_emit_bit_depth_change(self, temp_workspace):
        """Changing V2 output bit-depth flags should invalidate V2 reuse."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        tmpdir = Path(temp_workspace["root"])
        test_image = tmpdir / "emit_change.jpg"
        test_image.write_bytes(b"fake image")
        v2_report = tmpdir / "emit_change_report.json"
        v2_report.write_text('{"status":"ok"}', encoding="utf-8")
        manifest_path = tmpdir / "emit_change_manifest.json"

        base_config = EnhanceConfig(output_bit_depth=8, enable_v2=False, v2_preset="default")
        changed_config = EnhanceConfig(output_bit_depth=16, enable_v2=False, v2_preset="default")

        with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry"):
            base_orchestrator = EnhanceOrchestrator(base_config, tmpdir / "output_base", verify_outputs=False)
            manifest = CombinedManifest(
                config_fingerprint=base_orchestrator.compute_config_fingerprint(),
                v2=V2Metadata(
                    preset="default",
                    status="ok",
                    strict_depth=True,
                    output_dir="v2/",
                    report_path=str(v2_report),
                ),
            )
            manifest.save(manifest_path)

            changed_orchestrator = EnhanceOrchestrator(
                changed_config,
                tmpdir / "output_changed",
                verify_outputs=False,
            )

        should_skip = changed_orchestrator.should_skip_v2(
            v2_report,
            manifest_path,
            ImageInput(test_image),
            depth_was_skipped=True,
        )

        assert should_skip is False

    def test_should_skip_v2_accepts_legacy_success_status(self, temp_workspace):
        """Legacy manifests written with V2 status='success' should still be reusable."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        tmpdir = Path(temp_workspace["root"])
        test_image = tmpdir / "legacy_success.jpg"
        test_image.write_bytes(b"legacy success image")
        v2_report = tmpdir / "legacy_success_report.json"
        v2_report.write_text('{"status":"ok"}', encoding="utf-8")
        manifest_path = tmpdir / "legacy_success_manifest.json"

        with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry"):
            orchestrator = EnhanceOrchestrator(
                EnhanceConfig(enable_v2=True, v2_preset="default"),
                tmpdir / "output",
                verify_outputs=False,
            )
            CombinedManifest(
                config_fingerprint=orchestrator.compute_config_fingerprint(),
                v2=V2Metadata(
                    preset="default",
                    status="success",
                    strict_depth=True,
                    output_dir="v2/",
                    report_path=str(v2_report),
                ),
            ).save(manifest_path)

        should_skip = orchestrator.should_skip_v2(
            v2_report,
            manifest_path,
            ImageInput(test_image),
            depth_was_skipped=True,
        )

        assert should_skip is True


class TestCachedRunFidelity:
    """Tests for cached-run metadata preservation across reruns."""

    def test_cached_depth_restores_materials_v3_segmentation_provenance_and_enhanced_path(self, temp_workspace):
        """Cached depth reuse should restore full Materials V3 metadata needed by downstream stages."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        tmpdir = Path(temp_workspace["root"])
        test_image = tmpdir / "cached_materials.jpg"
        test_image.write_bytes(b"cached-materials-image")
        depth_path = tmpdir / "cached_materials_depth.png"
        depth_path.write_bytes(b"depth")
        float_depth_path = tmpdir / "cached_materials_depth.npy"
        manifest_path = tmpdir / "cached_materials_manifest.json"

        with (
            patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry"),
            patch("transformation_portal.lux_depth_v3.materials_v3.MaterialsV3Engine"),
        ):
            orchestrator = EnhanceOrchestrator(
                EnhanceConfig(enable_materials_v3=True, enable_v2=False),
                tmpdir / "output",
                verify_outputs=False,
            )

        output_key = Path("interiors/cached_materials")
        expected_enhanced_path = orchestrator._expected_materials_v3_enhanced_path(output_key)
        expected_enhanced_path.parent.mkdir(parents=True, exist_ok=True)
        expected_enhanced_path.write_bytes(b"enhanced")

        mask_artifact_path = orchestrator._segmentation_mask_artifact_path(output_key)
        mask_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        mask_artifact_path.write_bytes(b"npz")

        CombinedManifest(
            depth=DepthMetadata(model="da3", depth_path=str(depth_path), runtime_seconds=0.1, scaling={}),
            backend_selection=orchestrator._capture_backend_metadata(),
            materials_v3=MaterialsV3Metadata(
                enabled=True,
                version="3.1",
                response_plan={"plan": "preserve"},
                pixel_ops={"applied": 2},
                segmentation_metadata={
                    "mask_artifact_path": str(mask_artifact_path),
                    "mask_artifact_format": "npz",
                },
                runtime_seconds=0.42,
            ),
        ).save(manifest_path)

        (
            depth_metadata,
            _depth_runtime_s,
            _pbr_assets,
            materials_v3_result,
            materials_v3_runtime_s,
            enhanced_image_path,
            backend_selection,
            _depth_attempts,
        ) = orchestrator._compute_depth_stage(
            image_input=ImageInput(test_image),
            output_key=output_key,
            depth_path=depth_path,
            float_depth_path=float_depth_path,
            manifest_path=manifest_path,
            skip_depth=True,
        )

        assert depth_metadata is not None
        assert backend_selection.resolved_backend == "da3"
        assert materials_v3_result is not None
        assert materials_v3_result["materials_v3_response_plan"] == {"plan": "preserve"}
        assert materials_v3_result["materials_v3_pixel_ops"] == {"applied": 2}
        assert materials_v3_result["materials_v3_metadata"]["segmentation_metadata"]["mask_artifact_path"] == str(
            mask_artifact_path
        )
        assert materials_v3_runtime_s == pytest.approx(0.42, rel=1e-6, abs=1e-6)
        assert enhanced_image_path == expected_enhanced_path

    def test_run_v2_stage_skip_rehydrates_prior_report_and_output_paths(self, temp_workspace):
        """V2 skip path should preserve prior report/output references for manifest rewrite and API output."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        tmpdir = Path(temp_workspace["root"])
        test_image = tmpdir / "v2_skip.jpg"
        test_image.write_bytes(b"v2-skip-image")
        manifest_path = tmpdir / "v2_skip_manifest.json"
        output_key = Path("v2_skip")
        report_path = tmpdir / "output" / "v2" / f"{output_key.name}_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text('{"status":"ok"}', encoding="utf-8")
        output_path = tmpdir / "output" / "v2" / "v2_skip_output.png"

        with (
            patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry"),
            patch("transformation_portal.lux_depth_v3.orchestrator.V2Runner"),
        ):
            orchestrator = EnhanceOrchestrator(
                EnhanceConfig(enable_v2=True, v2_preset="default"),
                tmpdir / "output",
                verify_outputs=False,
            )

        CombinedManifest(
            v2=V2Metadata(
                preset="default",
                status="ok",
                strict_depth=True,
                output_dir="v2/",
                report_path=str(report_path),
                output_paths=[str(output_path)],
                runtime_seconds=1.23,
            )
        ).save(manifest_path)

        with patch.object(orchestrator, "should_skip_v2", return_value=True):
            v2_result, v2_runtime_s, v2_report_path = orchestrator._run_v2_stage(
                image_input=ImageInput(test_image),
                depth_path=None,
                output_key=output_key,
                v2_log_path=tmpdir / "output" / "logs" / "v2_skip.log",
                manifest_path=manifest_path,
                skip_depth=True,
                materials_v3_result=None,
            )

        assert v2_runtime_s == 0.0
        assert v2_report_path == report_path
        assert v2_result["report_path"] == str(report_path)
        assert v2_result["output_paths"] == [str(output_path)]
        assert v2_result["output"] == str(output_path)

    def test_write_manifest_preserves_prior_v2_and_materials_metadata_on_skip(self, temp_workspace):
        """Manifest rewrites should not erase prior V2 outputs or Materials V3 segmentation provenance."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        tmpdir = Path(temp_workspace["root"])
        test_image = tmpdir / "manifest_preservation.jpg"
        test_image.write_bytes(b"manifest-preservation-image")
        manifest_path = tmpdir / "output" / "manifests" / "manifest_preservation.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        report_path = tmpdir / "output" / "v2" / "manifest_preservation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        output_path = tmpdir / "output" / "v2" / "manifest_preservation_output.png"
        mask_artifact_path = tmpdir / "output" / "segmentation" / "manifest_preservation_masks.npz"
        mask_artifact_path.parent.mkdir(parents=True, exist_ok=True)

        with (
            patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry"),
            patch("transformation_portal.lux_depth_v3.orchestrator.V2Runner"),
            patch("transformation_portal.lux_depth_v3.materials_v3.MaterialsV3Engine"),
        ):
            orchestrator = EnhanceOrchestrator(
                EnhanceConfig(enable_v2=True, v2_preset="default", enable_materials_v3=True),
                tmpdir / "output",
                verify_outputs=False,
            )

        previous_depth = DepthMetadata(
            model="da3",
            depth_path=str(tmpdir / "output" / "depth" / "manifest_preservation_depth.png"),
            runtime_seconds=0.5,
            scaling={},
        )
        CombinedManifest(
            config_fingerprint=orchestrator.compute_config_fingerprint(),
            depth=previous_depth,
            v2=V2Metadata(
                preset="default",
                status="ok",
                strict_depth=True,
                output_dir="v2/",
                report_path=str(report_path),
                output_paths=[str(output_path)],
                runtime_seconds=1.5,
            ),
            materials_v3=MaterialsV3Metadata(
                enabled=True,
                version="3.1",
                response_plan={"plan": "persist"},
                pixel_ops={"applied": 3},
                segmentation_metadata={
                    "mask_artifact_path": str(mask_artifact_path),
                    "mask_artifact_format": "npz",
                },
                runtime_seconds=0.25,
            ),
            backend_selection=orchestrator._capture_backend_metadata(),
        ).save(manifest_path)

        fake_provenance = Mock()

        with (
            patch("transformation_portal.lux_depth_v3.orchestrator.capture_provenance", return_value=fake_provenance),
            patch(
                "transformation_portal.lux_depth_v3.raw_loader.is_raw_file",
                return_value=False,
            ),
        ):
            orchestrator._write_manifest(
                manifest_path=manifest_path,
                image_input=ImageInput(test_image),
                depth_metadata=previous_depth,
                v2_result={"status": "ok"},
                v2_report_path=None,
                pbr_assets=None,
                depth_runtime_s=0.5,
                v2_runtime_s=0.0,
                pipeline_start_time=10.0,
                pipeline_end_time=11.0,
                materials_v3_result={"materials_v3_metadata": {"version": "3.1"}},
                materials_v3_runtime_s=0.0,
                backend_selection_metadata=orchestrator._capture_backend_metadata(),
            )

        loaded = CombinedManifest.load(manifest_path)

        assert loaded.v2 is not None
        assert loaded.v2.report_path == str(report_path)
        assert loaded.v2.output_paths == [str(output_path)]
        assert loaded.v2.runtime_seconds == pytest.approx(1.5, rel=1e-6, abs=1e-6)
        assert loaded.materials_v3 is not None
        assert loaded.materials_v3.response_plan == {"plan": "persist"}
        assert loaded.materials_v3.pixel_ops == {"applied": 3}
        assert loaded.materials_v3.segmentation_metadata["mask_artifact_path"] == str(mask_artifact_path)
        assert loaded.materials_v3.runtime_seconds == pytest.approx(0.25, rel=1e-6, abs=1e-6)

    def test_write_manifest_normalizes_v2_success_status_and_reuses_previous_hash(self, temp_workspace):
        """Manifest rewrite should canonicalize V2 status and reuse the already-loaded manifest hash."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        tmpdir = Path(temp_workspace["root"])
        test_image = tmpdir / "manifest_status_normalization.jpg"
        test_image.write_bytes(b"manifest-status-image")
        manifest_path = tmpdir / "output" / "manifests" / "manifest_status_normalization.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("{}", encoding="utf-8")
        previous_depth = DepthMetadata(
            model="da3",
            depth_path=str(tmpdir / "output" / "depth" / "manifest_status_normalization_depth.png"),
            runtime_seconds=0.25,
            scaling={},
        )

        with (
            patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry"),
            patch("transformation_portal.lux_depth_v3.orchestrator.V2Runner"),
        ):
            orchestrator = EnhanceOrchestrator(
                EnhanceConfig(enable_v2=True, v2_preset="default"),
                tmpdir / "output",
                verify_outputs=False,
            )

        previous_manifest = CombinedManifest(
            input=InputMetadata(
                image_path=str(test_image),
                image_sha256="saved-input-hash",
            ),
            config_fingerprint=orchestrator.compute_config_fingerprint(),
            depth=previous_depth,
            v2=V2Metadata(
                preset="default",
                status="ok",
                strict_depth=True,
                output_dir="v2/",
                report_path="",
            ),
            backend_selection=orchestrator._capture_backend_metadata(),
        )
        fake_provenance = Mock()
        hash_call: dict[str, object] = {}

        def _capture_hash_args(
            _image_path: Path,
            *,
            manifest_exists: bool,
            saved_hash: str | None,
            for_manifest_write: bool,
        ) -> str:
            hash_call["manifest_exists"] = manifest_exists
            hash_call["saved_hash"] = saved_hash
            hash_call["for_manifest_write"] = for_manifest_write
            return saved_hash or "computed-input-hash"

        with (
            patch.object(
                orchestrator,
                "_load_existing_manifest",
                return_value=previous_manifest,
            ),
            patch(
                "transformation_portal.lux_depth_v3.orchestrator.CombinedManifest.load",
                side_effect=AssertionError("unexpected manifest reload"),
            ),
            patch.object(
                orchestrator,
                "_compute_or_skip_hash",
                side_effect=_capture_hash_args,
            ),
            patch(
                "transformation_portal.lux_depth_v3.orchestrator.capture_provenance",
                return_value=fake_provenance,
            ),
            patch(
                "transformation_portal.lux_depth_v3.raw_loader.is_raw_file",
                return_value=False,
            ),
        ):
            orchestrator._write_manifest(
                manifest_path=manifest_path,
                image_input=ImageInput(test_image),
                depth_metadata=previous_depth,
                v2_result={"status": "success"},
                v2_report_path=None,
                pbr_assets=None,
                depth_runtime_s=0.25,
                v2_runtime_s=0.0,
                pipeline_start_time=10.0,
                pipeline_end_time=11.0,
                materials_v3_result=None,
                materials_v3_runtime_s=0.0,
                backend_selection_metadata=orchestrator._capture_backend_metadata(),
            )

        loaded = CombinedManifest.load(manifest_path)

        assert hash_call == {
            "manifest_exists": True,
            "saved_hash": "saved-input-hash",
            "for_manifest_write": True,
        }
        assert loaded.v2 is not None
        assert loaded.v2.status == "ok"


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

    def test_if_manifest_exists_stores_baseline_hash(self, temp_workspace):
        """Test that first run with IF_MANIFEST_EXISTS stores hash for future comparisons.

        This is a critical fix: on first run, we must compute and store a baseline hash
        so that future runs can detect file changes.
        """
        tmpdir = temp_workspace["root"]
        tmpdir = Path(tmpdir)

        # Create a test image file
        test_image = tmpdir / "test.jpg"
        test_image.write_bytes(b"fake image data")

        # Mock compute_file_sha256 to return a predictable hash
        expected_hash = "abc123def456"

        with patch("transformation_portal.lux_depth_v3.orchestrator.compute_file_sha256") as mock_hash:
            mock_hash.return_value = expected_hash

            from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

            # Create minimal config with IF_MANIFEST_EXISTS mode
            config = EnhanceConfig(hash_mode=HashMode.IF_MANIFEST_EXISTS)

            # Create orchestrator (mocking dependencies)
            with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry"):
                with patch("transformation_portal.lux_depth_v3.orchestrator.Postprocessor"):
                    with patch("transformation_portal.lux_depth_v3.orchestrator.V2Runner"):
                        orchestrator = EnhanceOrchestrator(config, tmpdir / "output", verify_outputs=False)

                        # First run: no manifest exists
                        # for_manifest_write=True should compute hash (establishing baseline)
                        hash_for_write = orchestrator._compute_or_skip_hash(
                            test_image, manifest_exists=False, saved_hash=None, for_manifest_write=True
                        )

                        assert hash_for_write == expected_hash, "First run with for_manifest_write=True must compute hash"

                        # Comparison call (for skip check) on first run should NOT compute hash
                        hash_for_compare = orchestrator._compute_or_skip_hash(
                            test_image, manifest_exists=False, saved_hash=None, for_manifest_write=False
                        )

                        assert hash_for_compare is None, "First run with for_manifest_write=False should skip hash"

                        # Second run: manifest exists with saved hash
                        # for_manifest_write=False should now compute hash for comparison
                        hash_for_compare_2nd = orchestrator._compute_or_skip_hash(
                            test_image, manifest_exists=True, saved_hash=expected_hash, for_manifest_write=False
                        )

                        assert hash_for_compare_2nd == expected_hash, "Second run should compute hash for comparison"


class TestCachedDepthNoDoubleNormalization:
    """Tests for cached depth loading without double normalization (Fix 2)."""

    def test_cached_depth_no_double_normalization(self, temp_workspace):
        """Test that loading cached depth doesn't double-normalize float32 values.

        This is a critical fix: if read_depth_u16_png() returns float32 in [0,1],
        dividing by 65535 again crushes the range to ~[0, 0.00002], breaking PBR maps.
        """
        import numpy as np

        tmpdir = temp_workspace["root"]
        tmpdir = Path(tmpdir)

        # Create test depth files
        depth_png = tmpdir / "depth.png"
        float_depth_npy = tmpdir / "depth.npy"

        # Test Case 1: Reader returns uint16 (should normalize)
        with patch("transformation_portal.lux_depth_v3.depth_writer.read_depth_u16_png") as mock_read:
            # Simulate reader returning uint16 values
            mock_read.return_value = np.array([[0, 32767, 65535]], dtype=np.uint16)

            from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

            config = EnhanceConfig()

            with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry"):
                with patch("transformation_portal.lux_depth_v3.orchestrator.Postprocessor"):
                    with patch("transformation_portal.lux_depth_v3.orchestrator.V2Runner"):
                        orchestrator = EnhanceOrchestrator(config, tmpdir / "output", verify_outputs=False)

                        # Create the PNG file so exists() returns True
                        depth_png.write_bytes(b"fake png")

                        result = orchestrator._load_cached_depth(depth_png, float_depth_npy)

                        assert result is not None, "Should load depth data"
                        assert result.dtype == np.float32, "Should convert to float32"
                        # Check normalization is correct
                        assert np.allclose(result, [0.0, 0.5, 1.0], atol=0.01), f"Expected [0, 0.5, 1], got {result}"

        # Test Case 2: Reader returns pre-normalized float32 (should NOT double-normalize)
        with patch("transformation_portal.lux_depth_v3.depth_writer.read_depth_u16_png") as mock_read:
            # Simulate reader returning already normalized float32 values
            mock_read.return_value = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)

            config = EnhanceConfig()

            with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry"):
                with patch("transformation_portal.lux_depth_v3.orchestrator.Postprocessor"):
                    with patch("transformation_portal.lux_depth_v3.orchestrator.V2Runner"):
                        orchestrator = EnhanceOrchestrator(config, tmpdir / "output", verify_outputs=False)

                        result = orchestrator._load_cached_depth(depth_png, float_depth_npy)

                        assert result is not None, "Should load depth data"
                        # Values should remain in [0, 1] range (not crushed to near-zero)
                        assert (
                            result[0, 1] > 0.4 and result[0, 1] < 0.6
                        ), f"Expected ~0.5, got {result[0, 1]} (double normalization bug)"
                        assert result[0, 2] > 0.9, f"Expected ~1.0, got {result[0, 2]} (double normalization bug)"


class TestV2SkipIndependentOfGeneratePBR:
    """Tests for V2 skip logic independent of generate_pbr flag (Fix 3)."""

    def test_v2_skip_independent_of_generate_pbr(self, temp_workspace):
        """Test that should_skip_v2() evaluates V2 enhancement independent of PBR generation.

        This is a critical fix: V2 enhancement and PBR generation are separate stages.
        The should_skip_v2() method should evaluate V2 config changes and V2 output existence,
        not gate on generate_pbr flag.
        """
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        tmpdir = temp_workspace["root"]
        tmpdir = Path(tmpdir)

        # Create test files
        test_image = tmpdir / "test.jpg"
        test_image.write_bytes(b"fake image")
        v2_report = tmpdir / "v2_report.json"
        v2_report.write_text('{"status": "ok"}')
        manifest_path = tmpdir / "manifest.json"

        # Test with generate_pbr=False
        config_no_pbr = EnhanceConfig(generate_pbr=False, v2_preset="default")

        with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry"):
            with patch("transformation_portal.lux_depth_v3.orchestrator.Postprocessor"):
                with patch("transformation_portal.lux_depth_v3.orchestrator.V2Runner"):
                    from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

                    orchestrator = EnhanceOrchestrator(config_no_pbr, tmpdir / "output", verify_outputs=False)
                    manifest = CombinedManifest(
                        config_fingerprint=orchestrator.compute_config_fingerprint(),
                        v2=V2Metadata(
                            preset="default",
                            status="ok",
                            strict_depth=True,
                            output_dir="v2/",
                            report_path=str(v2_report),
                        ),
                    )
                    manifest.save(manifest_path)

                    # V2 should be skippable even without PBR enabled
                    # (V2 outputs are valid, config matches)
                    image_input = ImageInput(test_image)
                    skip = orchestrator.should_skip_v2(v2_report, manifest_path, image_input, depth_was_skipped=True)

                    # Fix verification: should evaluate V2 independently
                    # V2 outputs exist and config matches, so should skip
                    assert skip is True, "should_skip_v2() should evaluate V2 independently of generate_pbr"

                    # Test with changed V2 config - should NOT skip
                    config_changed = EnhanceConfig(generate_pbr=False, v2_preset="different_preset")
                    orchestrator_changed = EnhanceOrchestrator(config_changed, tmpdir / "output", verify_outputs=False)

                    skip_changed = orchestrator_changed.should_skip_v2(
                        v2_report, manifest_path, image_input, depth_was_skipped=True
                    )

                    assert skip_changed is False, "Changed V2 config should invalidate skip"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
