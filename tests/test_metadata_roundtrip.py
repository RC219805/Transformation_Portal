"""Roundtrip stability tests for manifest metadata classes.

These tests prevent silent corruption bugs by verifying that:
1. Serialization -> Deserialization is lossless
2. Field order is irrelevant (keyword args only)
3. Schema versioning works correctly
4. Forward/backward compatibility is enforced
5. Missing optional fields are handled correctly
6. Extra fields are handled according to policy
"""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]

from src.transformation_portal.lux_depth_v3.manifest import (
    CombinedManifest,
    ConfigFingerprint,
    DepthMetadata,
    InputMetadata,
    MaterialsV3Metadata,
    ReproMetadata,
    TimingMetadata,
    V2Metadata,
)


class TestInputMetadataRoundtrip:
    """Test InputMetadata serialization/deserialization stability."""

    def test_roundtrip_preserves_all_fields(self):
        """Verify serialize -> deserialize is lossless."""
        original = InputMetadata(
            image_path="/path/to/image.jpg",
            image_sha256="abc123def456",
            image_size_bytes=1024,
            image_dimensions=(1920, 1080),
        )

        # Serialize
        data = original.to_dict()

        # Deserialize
        restored = InputMetadata.from_dict(data)

        # Verify equality
        assert restored == original
        assert restored.schema_version == "1.0"
        assert restored.image_path == "/path/to/image.jpg"
        assert restored.image_sha256 == "abc123def456"
        assert restored.image_size_bytes == 1024
        assert restored.image_dimensions == (1920, 1080)

    def test_roundtrip_with_minimal_fields(self):
        """Verify roundtrip works with only required fields."""
        original = InputMetadata(
            image_path="/path/to/minimal.jpg",
        )

        data = original.to_dict()
        restored = InputMetadata.from_dict(data)

        assert restored == original
        assert restored.image_path == "/path/to/minimal.jpg"
        assert restored.image_sha256 is None
        assert restored.image_size_bytes is None
        assert restored.image_dimensions is None
        assert restored.schema_version == "1.0"

    def test_roundtrip_with_none_values(self):
        """Verify explicit None values survive roundtrip."""
        original = InputMetadata(
            image_path="/path/to/image.jpg",
            image_sha256=None,
            image_size_bytes=None,
            image_dimensions=None,
        )

        data = original.to_dict()
        restored = InputMetadata.from_dict(data)

        assert restored == original

    def test_schema_version_defaults_to_1_0(self):
        """Verify schema_version defaults to 1.0."""
        metadata = InputMetadata(image_path="/test.jpg")
        assert metadata.schema_version == "1.0"

        data = metadata.to_dict()
        assert data["schema_version"] == "1.0"

    def test_from_dict_handles_missing_schema_version(self):
        """Verify backward compatibility: missing schema_version defaults to 1.0."""
        data = {
            "image_path": "/test.jpg",
            "image_sha256": "hash123",
        }

        metadata = InputMetadata.from_dict(data)
        assert metadata.schema_version == "1.0"

    def test_from_dict_rejects_unsupported_schema_version(self):
        """Verify forward compatibility: reject unsupported schema versions."""
        data = {
            "schema_version": "2.0",
            "image_path": "/test.jpg",
        }

        with pytest.raises(ValueError, match="Unsupported InputMetadata schema version: 2.0"):
            InputMetadata.from_dict(data)

    def test_from_dict_handles_list_dimensions(self):
        """Verify list-to-tuple conversion for image_dimensions."""
        data = {
            "image_path": "/test.jpg",
            "image_dimensions": [1920, 1080],  # List instead of tuple
        }

        metadata = InputMetadata.from_dict(data)
        assert metadata.image_dimensions == (1920, 1080)
        assert isinstance(metadata.image_dimensions, tuple)

    def test_from_dict_handles_none_dimensions(self):
        """Verify None dimensions are preserved."""
        data = {
            "image_path": "/test.jpg",
            "image_dimensions": None,
        }

        metadata = InputMetadata.from_dict(data)
        assert metadata.image_dimensions is None

    def test_keyword_only_construction(self):
        """Verify construction enforces keyword arguments (prevents positional arg bugs)."""
        # This should work (keyword args)
        metadata = InputMetadata(
            image_path="/test.jpg",
            image_sha256="hash123",
            image_size_bytes=1024,
            image_dimensions=(800, 600),
        )
        assert metadata.image_path == "/test.jpg"

        # Note: Python dataclasses don't enforce keyword-only by default,
        # but the PR #764 fixed positional arg bugs. This test documents
        # the expected usage pattern.

    def test_field_order_independence(self):
        """Verify field order doesn't matter (keyword args)."""
        metadata1 = InputMetadata(
            image_path="/test.jpg",
            image_sha256="hash123",
            image_size_bytes=1024,
            image_dimensions=(800, 600),
        )

        metadata2 = InputMetadata(
            image_dimensions=(800, 600),
            image_size_bytes=1024,
            image_sha256="hash123",
            image_path="/test.jpg",
        )

        assert metadata1 == metadata2

    def test_roundtrip_through_combined_manifest(self):
        """Verify InputMetadata roundtrip through CombinedManifest save/load."""
        import tempfile
        from pathlib import Path

        original = InputMetadata(
            image_path="/test.jpg",
            image_sha256="hash123",
            image_size_bytes=2048,
            image_dimensions=(1024, 768),
        )

        manifest = CombinedManifest(input=original)

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "test_manifest.json"
            manifest.save(manifest_path)

            loaded_manifest = CombinedManifest.load(manifest_path)

            assert loaded_manifest.input == original
            assert loaded_manifest.input.schema_version == "1.0"


class TestDepthMetadataRoundtrip:
    """Test DepthMetadata serialization stability."""

    def test_roundtrip_preserves_fields(self):
        """Verify DepthMetadata roundtrip."""
        from dataclasses import asdict

        original = DepthMetadata(
            model="depth-anything-v2-large",
            depth_path="/output/depth.png",
            runtime_seconds=1.234,
            scaling={"min": 0.0, "max": 1.0},
            stats={"mean": 0.5, "std": 0.2},
        )

        data = asdict(original)
        restored = DepthMetadata(**data)

        assert restored == original


class TestConfigFingerprintRoundtrip:
    """Test ConfigFingerprint serialization and hash stability."""

    def test_roundtrip_preserves_fields(self):
        """Verify ConfigFingerprint roundtrip."""
        from dataclasses import asdict

        original = ConfigFingerprint(
            model_variant="large",
            depth_quantization="float16",
            depth_device="mps",
            preset="architectural",
            v2_preset="enhance",
            v2_device="cuda",
            v2_upscaler_backend="realesrgan",
        )

        data = asdict(original)
        restored = ConfigFingerprint(**data)

        assert restored == original

    def test_sha256_hash_stability(self):
        """Verify SHA256 hash is deterministic and stable."""
        config1 = ConfigFingerprint(
            model_variant="large",
            depth_quantization="float16",
            depth_device="mps",
        )

        config2 = ConfigFingerprint(
            model_variant="large",
            depth_quantization="float16",
            depth_device="mps",
        )

        # Same config should produce same hash
        assert config1.to_sha256() == config2.to_sha256()

        # Different config should produce different hash
        config3 = ConfigFingerprint(
            model_variant="small",  # Changed
            depth_quantization="float16",
            depth_device="mps",
        )

        assert config1.to_sha256() != config3.to_sha256()

    def test_sha256_hash_field_order_independence(self):
        """Verify hash is independent of field construction order."""
        # ConfigFingerprint uses asdict + sort_keys=True, so order shouldn't matter
        hash1 = ConfigFingerprint(
            model_variant="large",
            depth_quantization="float16",
            depth_device="mps",
        ).to_sha256()

        hash2 = ConfigFingerprint(
            depth_device="mps",
            model_variant="large",
            depth_quantization="float16",
        ).to_sha256()

        assert hash1 == hash2


class TestCombinedManifestRoundtrip:
    """Test CombinedManifest save/load stability."""

    def test_roundtrip_full_manifest(self):
        """Verify full CombinedManifest roundtrip through JSON."""
        import tempfile
        from pathlib import Path

        original = CombinedManifest(
            input=InputMetadata(
                image_path="/test.jpg",
                image_sha256="hash123",
                image_size_bytes=1024,
                image_dimensions=(800, 600),
            ),
            depth=DepthMetadata(
                model="depth-anything-v2-large",
                depth_path="/output/depth.png",
                runtime_seconds=1.5,
                scaling={"min": 0.0, "max": 1.0},
            ),
            timing=TimingMetadata(
                depth_seconds=1.5,
                v2_seconds=2.0,
                total_seconds=3.5,
                timestamp_utc="2025-01-30T00:00:00Z",
            ),
            config_fingerprint=ConfigFingerprint(
                model_variant="large",
                depth_quantization="float16",
                depth_device="mps",
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            original.save(manifest_path)

            restored = CombinedManifest.load(manifest_path)

            # Verify all sections match
            assert restored.input == original.input
            assert restored.depth == original.depth
            assert restored.timing == original.timing
            assert restored.config_fingerprint == original.config_fingerprint

    def test_roundtrip_partial_manifest(self):
        """Verify partial manifest (only some fields) survives roundtrip."""
        import tempfile
        from pathlib import Path

        original = CombinedManifest(
            input=InputMetadata(image_path="/test.jpg"),
            # Only input, no depth/v2/timing
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            original.save(manifest_path)

            restored = CombinedManifest.load(manifest_path)

            assert restored.input == original.input
            assert restored.depth is None
            assert restored.v2 is None
            assert restored.timing is None

    def test_write_alias_compatibility(self):
        """Verify write() alias works for backward compatibility."""
        import tempfile
        from pathlib import Path

        manifest = CombinedManifest(
            input=InputMetadata(image_path="/test.jpg"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"

            # Use write() alias
            manifest.write(manifest_path)

            # Verify file was created
            assert manifest_path.exists()

            # Verify load works
            restored = CombinedManifest.load(manifest_path)
            assert restored.input == manifest.input

    def test_save_normalizes_numpy_scalars_in_nested_metadata(self):
        """Verify nested NumPy scalar/array values are normalized before JSON write."""
        import tempfile
        from pathlib import Path

        manifest = CombinedManifest(
            materials_v3=MaterialsV3Metadata(
                enabled=True,
                segmentation_metadata={
                    "mask_count": np.int64(4),
                    "coverage": np.float32(0.75),
                    "strict": np.bool_(True),
                    "indices": np.array([1, 3, 7], dtype=np.int64),
                },
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest.save(manifest_path)

            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            segmentation = saved["materials_v3"]["segmentation_metadata"]
            assert segmentation["mask_count"] == 4
            assert segmentation["coverage"] == pytest.approx(0.75, rel=1e-6, abs=1e-6)
            assert segmentation["strict"] is True
            assert segmentation["indices"] == [1, 3, 7]

    def test_save_routes_legacy_json_bytes_through_atomic_writer(self, tmp_path):
        """The durable migration must retain exact existing JSON formatting."""
        manifest_path = tmp_path / "combined_manifest.json"
        manifest = CombinedManifest(
            input=InputMetadata(image_path="/图像.jpg"),
            start_time="2026-08-31T00:00:00Z",
        )
        expected = """{
  "input": {
    "image_dimensions": null,
    "image_path": "/图像.jpg",
    "image_sha256": null,
    "image_size_bytes": null,
    "schema_version": "1.0"
  },
  "start_time": "2026-08-31T00:00:00Z"
}""".encode("utf-8")

        with patch("src.transformation_portal.lux_depth_v3.manifest.atomic_write_bytes") as durable_write:
            manifest.save(manifest_path)

        durable_write.assert_called_once_with(manifest_path, expected)

    def test_serialization_failure_preserves_prior_manifest(self, tmp_path):
        """Invalid JSON must fail before invoking the durable writer."""
        manifest_path = tmp_path / "combined_manifest.json"
        manifest_path.write_bytes(b"prior-valid-manifest")
        manifest = CombinedManifest(environment={"invalid": float("nan")})

        with (
            patch("src.transformation_portal.lux_depth_v3.manifest.atomic_write_bytes") as durable_write,
            pytest.raises(ValueError, match="Out of range float values"),
        ):
            manifest.save(manifest_path)

        durable_write.assert_not_called()
        assert manifest_path.read_bytes() == b"prior-valid-manifest"


# Edge case tests for robustness
class TestMetadataEdgeCases:
    """Test edge cases and error conditions."""

    def test_input_metadata_with_empty_path(self):
        """Verify empty path is allowed (edge case)."""
        metadata = InputMetadata(image_path="")
        assert metadata.image_path == ""

    def test_input_metadata_with_very_long_path(self):
        """Verify very long paths are handled."""
        long_path = "/very/long/path/" + "a" * 1000 + "/image.jpg"
        metadata = InputMetadata(image_path=long_path)

        data = metadata.to_dict()
        restored = InputMetadata.from_dict(data)

        assert restored.image_path == long_path

    def test_input_metadata_with_unicode_path(self):
        """Verify unicode paths are preserved."""
        unicode_path = "/path/to/图像/image.jpg"
        metadata = InputMetadata(image_path=unicode_path)

        data = metadata.to_dict()
        restored = InputMetadata.from_dict(data)

        assert restored.image_path == unicode_path

    def test_input_metadata_from_dict_missing_required_field(self):
        """Verify from_dict raises error when required field is missing."""
        data = {
            "image_sha256": "hash123",
            # Missing 'image_path'
        }

        with pytest.raises(KeyError):
            InputMetadata.from_dict(data)

    def test_config_fingerprint_depth_only_projection(self):
        """Verify depth_only() projection preserves Stage A fields only."""
        config = ConfigFingerprint(
            model_variant="large",
            depth_quantization="float16",
            depth_device="mps",
            preset="architectural",
            v2_preset="enhance",
            v2_device="cuda",
            v2_upscaler_backend="realesrgan",
            depth_backend="depth_pro",
            quality_tier="apex",
            materials_config={"enable_materials_v3": True},
            pbr_config={"generate_pbr": True},
            apex_depth_gate_config={"min_upper_iqr": 1e-4},
            emit_master16=True,
            emit_upscaled16=False,
            enable_v2=True,
        )

        depth_only = config.depth_only()

        assert depth_only.model_variant == "large"
        assert depth_only.depth_quantization == "float16"
        assert depth_only.depth_device == "mps"
        assert depth_only.preset == "architectural"
        assert depth_only.depth_backend == "depth_pro"
        assert depth_only.quality_tier == "apex"
        assert depth_only.materials_config == {"enable_materials_v3": True}
        assert depth_only.pbr_config == {"generate_pbr": True}
        assert depth_only.apex_depth_gate_config == {"min_upper_iqr": 1e-4}
        assert depth_only.output_bit_depth == 16
        assert depth_only.v2_preset is None
        assert depth_only.v2_device is None
        assert depth_only.v2_upscaler_backend is None
        assert depth_only.enable_v2 is None

    def test_config_fingerprint_v2_only_projection(self):
        """Verify v2_only() projection preserves V2 fields only."""
        config = ConfigFingerprint(
            model_variant="large",
            depth_quantization="float16",
            depth_device="mps",
            preset="architectural",
            v2_preset="enhance",
            v2_device="cuda",
            v2_upscaler_backend="realesrgan",
            depth_backend="depth_pro",
            quality_tier="apex",
            materials_config={"enable_materials_v3": True},
            pbr_config={"generate_pbr": True},
            apex_depth_gate_config={"min_upper_iqr": 1e-4},
            emit_master16=True,
            emit_upscaled16=False,
            enable_v2=True,
        )

        v2_only = config.v2_only()

        assert v2_only.model_variant == ""
        assert v2_only.depth_quantization == ""
        assert v2_only.depth_device == ""
        assert v2_only.preset is None
        assert v2_only.depth_backend is None
        assert v2_only.materials_config is None
        assert v2_only.pbr_config is None
        assert v2_only.apex_depth_gate_config is None
        assert v2_only.v2_preset == "enhance"
        assert v2_only.v2_device == "cuda"
        assert v2_only.v2_upscaler_backend == "realesrgan"
        assert v2_only.output_bit_depth == 16
        assert v2_only.enable_v2 is True
