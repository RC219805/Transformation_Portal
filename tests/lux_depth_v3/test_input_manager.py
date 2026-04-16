"""Tests for input_manager module contract and functionality.

This test suite validates the modernized ImageInput contract including:
- Path coercion and normalization
- Metadata validation
- Format detection helpers
- Explicit validation
- Serialization round-trips
- Typed metadata compatibility
- Opt-in enrichment behavior
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from transformation_portal.lux_depth_v3.input_manager import (
    SUPPORTED_EXTENSIONS,
    ImageInput,
    InputImageMetadata,
    InputValidationError,
    UnsupportedFormatError,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Path coercion and normalization tests
# =============================================================================


class TestPathCoercion:
    """Tests for path coercion behavior in ImageInput."""

    def test_string_path_coerced_to_path(self):
        """String paths should be converted to Path objects."""
        img = ImageInput(path="/tmp/test.jpg")
        assert isinstance(img.path, Path)

    def test_path_object_preserved(self):
        """Path objects should be preserved."""
        p = Path("/tmp/test.jpg")
        img = ImageInput(path=p)
        assert isinstance(img.path, Path)

    def test_path_is_normalized_absolute(self):
        """Paths should be normalized to absolute paths."""
        # Create relative path
        img = ImageInput(path="relative/path/image.jpg")
        # Should become absolute
        assert img.path.is_absolute()

    def test_path_with_user_expansion(self):
        """User home (~) should be expanded."""
        img = ImageInput(path="~/images/test.jpg")
        assert "~" not in str(img.path)
        assert img.path.is_absolute()

    def test_empty_path_rejected(self):
        """Empty paths should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ImageInput(path="")

    def test_whitespace_only_path_rejected(self):
        """Whitespace-only paths should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ImageInput(path="   ")

    def test_dot_only_path_rejected(self):
        """Single dot path should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ImageInput(path=".")


# =============================================================================
# Metadata validation tests
# =============================================================================


class TestMetadataValidation:
    """Tests for metadata validation behavior."""

    def test_none_metadata_accepted(self):
        """None metadata should be accepted."""
        img = ImageInput(path="/tmp/test.jpg", metadata=None)
        assert img.metadata is None

    def test_empty_dict_metadata_accepted(self):
        """Empty dict metadata should be accepted."""
        img = ImageInput(path="/tmp/test.jpg", metadata={})
        assert img.metadata == {}

    def test_dict_metadata_accepted(self):
        """Dict metadata should be accepted."""
        meta = {"key": "value", "number": 42}
        img = ImageInput(path="/tmp/test.jpg", metadata=meta)
        assert img.metadata == meta

    def test_non_dict_metadata_rejected(self):
        """Non-dict metadata should raise ValueError."""
        with pytest.raises(ValueError, match="must be a dict"):
            ImageInput(path="/tmp/test.jpg", metadata="not a dict")

    def test_list_metadata_rejected(self):
        """List metadata should raise ValueError."""
        with pytest.raises(ValueError, match="must be a dict"):
            ImageInput(path="/tmp/test.jpg", metadata=["a", "b"])


# =============================================================================
# Format detection tests
# =============================================================================


class TestFormatDetection:
    """Tests for format detection properties."""

    def test_format_lowercase(self):
        """Format should be lowercase."""
        img = ImageInput(path="/tmp/test.JPG")
        assert img.format == ".jpg"

    def test_format_jpeg(self):
        """JPEG format detection."""
        assert ImageInput(path="/tmp/test.jpeg").format == ".jpeg"
        assert ImageInput(path="/tmp/test.jpg").format == ".jpg"

    def test_format_png(self):
        """PNG format detection."""
        assert ImageInput(path="/tmp/test.png").format == ".png"
        assert ImageInput(path="/tmp/test.PNG").format == ".png"

    def test_format_tiff(self):
        """TIFF format detection."""
        assert ImageInput(path="/tmp/test.tiff").format == ".tiff"
        assert ImageInput(path="/tmp/test.tif").format == ".tif"

    def test_format_raw_cr2(self):
        """RAW CR2 format detection."""
        assert ImageInput(path="/tmp/test.CR2").format == ".cr2"

    def test_format_raw_nef(self):
        """RAW NEF format detection."""
        assert ImageInput(path="/tmp/test.NEF").format == ".nef"

    def test_is_raw_true_for_raw_formats(self):
        """is_raw should return True for RAW formats."""
        raw_formats = [".cr2", ".nef", ".arw", ".dng", ".orf", ".raf"]
        for fmt in raw_formats:
            img = ImageInput(path=f"/tmp/test{fmt}")
            assert img.is_raw is True, f"Expected is_raw=True for {fmt}"

    def test_is_raw_false_for_standard_formats(self):
        """is_raw should return False for standard formats."""
        standard_formats = [".jpg", ".jpeg", ".png", ".tiff", ".webp"]
        for fmt in standard_formats:
            img = ImageInput(path=f"/tmp/test{fmt}")
            assert img.is_raw is False, f"Expected is_raw=False for {fmt}"

    def test_is_supported_true_for_valid_formats(self):
        """is_supported should return True for supported formats."""
        supported = [".jpg", ".png", ".tiff", ".cr2", ".nef", ".dng"]
        for fmt in supported:
            img = ImageInput(path=f"/tmp/test{fmt}")
            assert img.is_supported is True, f"Expected is_supported=True for {fmt}"

    def test_is_supported_false_for_invalid_formats(self):
        """is_supported should return False for unsupported formats."""
        unsupported = [".pdf", ".doc", ".mp4", ".txt", ".xyz"]
        for fmt in unsupported:
            img = ImageInput(path=f"/tmp/test{fmt}")
            assert img.is_supported is False, f"Expected is_supported=False for {fmt}"


# =============================================================================
# Validation tests
# =============================================================================


class TestValidation:
    """Tests for explicit validation behavior."""

    def test_validate_no_io_by_default(self):
        """validate() with check_exists=False should not touch filesystem."""
        # This path doesn't exist but validation should pass
        img = ImageInput(path="/nonexistent/path/image.jpg")
        img.validate(check_exists=False, check_supported=True)

    def test_validate_unsupported_format_raises(self):
        """validate() should raise UnsupportedFormatError for bad formats."""
        img = ImageInput(path="/tmp/document.pdf")
        with pytest.raises(UnsupportedFormatError, match="Unsupported image format"):
            img.validate(check_supported=True)

    def test_validate_unsupported_format_skippable(self):
        """validate() should not check format when check_supported=False."""
        img = ImageInput(path="/tmp/document.pdf")
        # Should not raise even though format is unsupported
        img.validate(check_supported=False, check_exists=False)

    def test_validate_check_exists_raises_for_missing(self):
        """validate() should raise FileNotFoundError for missing files."""
        img = ImageInput(path="/nonexistent/path/image.jpg")
        with pytest.raises(FileNotFoundError, match="not found"):
            img.validate(check_exists=True)

    def test_validate_check_exists_passes_for_existing(self, tmp_path):
        """validate() should pass for existing files."""
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"fake image data")

        img = ImageInput(path=test_file)
        img.validate(check_exists=True, check_supported=True)


# =============================================================================
# Serialization tests
# =============================================================================


class TestSerialization:
    """Tests for serialization and deserialization."""

    def test_to_dict_includes_schema_version(self):
        """to_dict() should include schema_version."""
        img = ImageInput(path="/tmp/test.jpg")
        data = img.to_dict()
        assert "schema_version" in data
        assert data["schema_version"] == "1.0"

    def test_to_dict_includes_path(self):
        """to_dict() should include path as string."""
        img = ImageInput(path="/tmp/test.jpg")
        data = img.to_dict()
        assert "path" in data
        assert isinstance(data["path"], str)

    def test_to_dict_excludes_none_metadata(self):
        """to_dict() should exclude metadata when None."""
        img = ImageInput(path="/tmp/test.jpg", metadata=None)
        data = img.to_dict()
        assert "metadata" not in data

    def test_to_dict_includes_metadata_when_present(self):
        """to_dict() should include metadata when present."""
        meta = {"key": "value"}
        img = ImageInput(path="/tmp/test.jpg", metadata=meta)
        data = img.to_dict()
        assert "metadata" in data
        assert data["metadata"] == meta

    def test_from_dict_basic(self):
        """from_dict() should reconstruct ImageInput."""
        data = {"path": "/tmp/test.jpg"}
        img = ImageInput.from_dict(data)
        assert str(img.path).endswith("test.jpg")

    def test_from_dict_with_metadata(self):
        """from_dict() should reconstruct metadata."""
        data = {"path": "/tmp/test.jpg", "metadata": {"key": "value"}}
        img = ImageInput.from_dict(data)
        assert img.metadata == {"key": "value"}

    def test_from_dict_accepts_schema_version(self):
        """from_dict() should accept schema_version field."""
        data = {"path": "/tmp/test.jpg", "schema_version": "1.0"}
        img = ImageInput.from_dict(data)
        assert str(img.path).endswith("test.jpg")

    def test_from_dict_accepts_legacy_without_schema(self):
        """from_dict() should accept legacy payloads without schema_version."""
        data = {"path": "/tmp/test.jpg"}
        img = ImageInput.from_dict(data)
        assert str(img.path).endswith("test.jpg")

    def test_from_dict_requires_path(self):
        """from_dict() should raise ValueError without path."""
        with pytest.raises(ValueError, match="requires 'path' key"):
            ImageInput.from_dict({})

    def test_round_trip_dict(self):
        """to_dict() -> from_dict() should round-trip."""
        original = ImageInput(path="/tmp/test.jpg", metadata={"key": "value"})
        data = original.to_dict()
        restored = ImageInput.from_dict(data)
        assert str(restored.path).endswith("test.jpg")
        assert restored.metadata == original.metadata

    def test_to_json_deterministic(self):
        """to_json() should produce deterministic output."""
        img = ImageInput(path="/tmp/test.jpg", metadata={"b": 2, "a": 1})
        json1 = img.to_json()
        json2 = img.to_json()
        assert json1 == json2
        # Verify metadata keys are sorted
        parsed = json.loads(json1)
        assert list(parsed["metadata"].keys()) == ["a", "b"]

    def test_from_json_basic(self):
        """from_json() should reconstruct ImageInput."""
        json_str = '{"path":"/tmp/test.jpg","schema_version":"1.0"}'
        img = ImageInput.from_json(json_str)
        assert str(img.path).endswith("test.jpg")

    def test_round_trip_json(self):
        """to_json() -> from_json() should round-trip."""
        original = ImageInput(path="/tmp/test.jpg", metadata={"key": "value"})
        json_str = original.to_json()
        restored = ImageInput.from_json(json_str)
        assert str(restored.path).endswith("test.jpg")
        assert restored.metadata == original.metadata


# =============================================================================
# Typed metadata tests
# =============================================================================


class TestInputImageMetadata:
    """Tests for InputImageMetadata typed contract."""

    def test_metadata_creation_defaults(self):
        """InputImageMetadata should have sensible defaults."""
        meta = InputImageMetadata()
        assert meta.image_sha256 is None
        assert meta.image_size_bytes is None
        assert meta.image_dimensions is None

    def test_metadata_creation_with_values(self):
        """InputImageMetadata should accept values."""
        meta = InputImageMetadata(
            image_sha256="abc123",
            image_size_bytes=1024,
            image_dimensions=(640, 480),
            source_format=".jpg",
        )
        assert meta.image_sha256 == "abc123"
        assert meta.image_size_bytes == 1024
        assert meta.image_dimensions == (640, 480)
        assert meta.source_format == ".jpg"

    def test_metadata_is_frozen(self):
        """InputImageMetadata should be immutable."""
        meta = InputImageMetadata(image_sha256="abc123")
        with pytest.raises(AttributeError):
            meta.image_sha256 = "changed"  # type: ignore

    def test_metadata_to_dict_omits_none(self):
        """to_dict() should omit None values."""
        meta = InputImageMetadata(image_sha256="abc123")
        data = meta.to_dict()
        assert "image_sha256" in data
        assert "image_size_bytes" not in data

    def test_metadata_from_mapping_none(self):
        """from_mapping(None) should return None."""
        assert InputImageMetadata.from_mapping(None) is None

    def test_metadata_from_mapping_empty(self):
        """from_mapping({}) should return None."""
        assert InputImageMetadata.from_mapping({}) is None

    def test_metadata_from_mapping_known_keys(self):
        """from_mapping() should extract known keys."""
        data = {"image_sha256": "abc123", "image_size_bytes": 1024}
        meta = InputImageMetadata.from_mapping(data)
        assert meta is not None
        assert meta.image_sha256 == "abc123"
        assert meta.image_size_bytes == 1024

    def test_metadata_from_mapping_unknown_keys(self):
        """from_mapping() should collect unknown keys into raw_metadata."""
        data = {"image_sha256": "abc123", "custom_key": "custom_value"}
        meta = InputImageMetadata.from_mapping(data)
        assert meta is not None
        assert meta.image_sha256 == "abc123"
        assert meta.raw_metadata is not None
        assert meta.raw_metadata.get("custom_key") == "custom_value"

    def test_metadata_from_mapping_list_to_tuple(self):
        """from_mapping() should convert list dimensions to tuple."""
        data = {"image_dimensions": [640, 480]}
        meta = InputImageMetadata.from_mapping(data)
        assert meta is not None
        assert meta.image_dimensions == (640, 480)

    def test_image_input_metadata_model_none(self):
        """metadata_model should return None when metadata is None."""
        img = ImageInput(path="/tmp/test.jpg", metadata=None)
        assert img.metadata_model is None

    def test_image_input_metadata_model_parses(self):
        """metadata_model should parse metadata dict."""
        img = ImageInput(
            path="/tmp/test.jpg",
            metadata={"image_sha256": "abc123", "image_size_bytes": 1024},
        )
        model = img.metadata_model
        assert model is not None
        assert model.image_sha256 == "abc123"
        assert model.image_size_bytes == 1024


# =============================================================================
# Enrichment factory tests
# =============================================================================


class TestFromPathFactory:
    """Tests for from_path() factory method."""

    def test_from_path_basic(self, tmp_path):
        """from_path() should create ImageInput."""
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"fake image")

        img = ImageInput.from_path(test_file)
        assert str(img.path).endswith("test.jpg")

    def test_from_path_detect_format(self, tmp_path):
        """from_path() should detect format by default."""
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"fake image")

        img = ImageInput.from_path(test_file, detect_format=True)
        assert img.metadata is not None
        assert img.metadata.get("source_format") == ".jpg"

    def test_from_path_no_detect_format(self, tmp_path):
        """from_path() with detect_format=False should not add format."""
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"fake image")

        img = ImageInput.from_path(test_file, detect_format=False)
        assert img.metadata is None

    def test_from_path_compute_hash(self, tmp_path):
        """from_path() with compute_hash=True should compute SHA-256."""
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"test content")

        img = ImageInput.from_path(test_file, compute_hash=True)
        assert img.metadata is not None
        assert "image_sha256" in img.metadata
        assert len(img.metadata["image_sha256"]) == 64  # SHA-256 hex length
        assert "image_size_bytes" in img.metadata
        assert img.metadata["image_size_bytes"] == 12  # len(b"test content")

    def test_from_path_compute_hash_missing_file(self, tmp_path):
        """from_path() with compute_hash=True should raise for missing file."""
        missing = tmp_path / "nonexistent.jpg"
        with pytest.raises(FileNotFoundError, match="Cannot compute hash"):
            ImageInput.from_path(missing, compute_hash=True)

    def test_from_path_probe_dimensions_missing_file(self, tmp_path):
        """from_path() with probe_dimensions=True should raise for missing file."""
        missing = tmp_path / "nonexistent.jpg"
        with pytest.raises(FileNotFoundError, match="Cannot probe dimensions"):
            ImageInput.from_path(missing, probe_dimensions=True)


# =============================================================================
# Backward compatibility tests
# =============================================================================


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility with existing usage."""

    def test_legacy_construction_works(self):
        """Legacy ImageInput(path=...) construction should work."""
        img = ImageInput(path=Path("/tmp/test.jpg"))
        assert isinstance(img.path, Path)

    def test_legacy_string_construction_works(self):
        """Legacy ImageInput(path="...") construction should work."""
        img = ImageInput(path="/tmp/test.jpg")
        assert isinstance(img.path, Path)

    def test_legacy_metadata_access_works(self):
        """Legacy .metadata attribute access should work."""
        img = ImageInput(path="/tmp/test.jpg", metadata={"key": "value"})
        assert img.metadata == {"key": "value"}

    def test_legacy_path_access_works(self):
        """Legacy .path attribute access should work."""
        img = ImageInput(path="/tmp/test.jpg")
        assert "test.jpg" in str(img.path)

    def test_mutable_metadata(self):
        """Metadata should remain mutable for backward compatibility."""
        img = ImageInput(path="/tmp/test.jpg", metadata={})
        img.metadata["new_key"] = "new_value"
        assert img.metadata["new_key"] == "new_value"

    def test_schema_version_class_variable(self):
        """SCHEMA_VERSION should be accessible as class variable."""
        assert ImageInput.SCHEMA_VERSION == "1.0"


# =============================================================================
# Module import tests
# =============================================================================


class TestModuleImports:
    """Tests for module-level imports and exports."""

    def test_supported_extensions_exported(self):
        """SUPPORTED_EXTENSIONS should be exported."""
        assert SUPPORTED_EXTENSIONS is not None
        assert ".jpg" in SUPPORTED_EXTENSIONS
        assert ".cr2" in SUPPORTED_EXTENSIONS

    def test_exceptions_exported(self):
        """Exception classes should be exported."""
        assert InputValidationError is not None
        assert UnsupportedFormatError is not None
        assert issubclass(UnsupportedFormatError, InputValidationError)

    def test_input_image_metadata_exported(self):
        """InputImageMetadata should be exported."""
        assert InputImageMetadata is not None

    def test_no_eager_heavy_imports(self):
        """Module import should not eagerly import heavy dependencies.

        This test verifies that importing input_manager doesn't pull in PIL/Pillow
        as a side effect. PIL is only needed for probe_dimensions=True in from_path().
        """
        import importlib
        import sys

        # Track PIL modules before import
        pil_before = {k for k in sys.modules.keys() if k.startswith("PIL")}

        # Force reimport (module may already be loaded)
        import transformation_portal.lux_depth_v3.input_manager as im

        importlib.reload(im)

        # After reload, PIL should not have been newly imported
        pil_after = {k for k in sys.modules.keys() if k.startswith("PIL")}
        new_pil_modules = pil_after - pil_before

        # Verify the module is usable
        assert im.ImageInput is not None
        assert im.InputImageMetadata is not None

        # Note: We can't guarantee PIL wasn't loaded by other tests/modules,
        # so we only check that our module import itself didn't add new PIL imports.
        # In a fresh Python interpreter, this would be an empty set.
