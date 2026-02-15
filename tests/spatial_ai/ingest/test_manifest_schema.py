"""Tests for manifest schema.

Tests cover:
- Manifest creation and validation
- Dataset builder API
- Schema versioning
- Manifest writing and loading
- Image inventory management
- Tag filtering
- Schema validation errors

Architecture: ADR-023, Issue #890 Phase I
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from transformation_portal.spatial_ai.ingest import (
    DatasetManifestBuilder,
    ImageManifestEntry,
    ManifestError,
    ManifestSchema,
    SchemaVersionError,
)


class TestManifestSchema:
    """Test manifest schema functionality."""

    def test_create_empty_manifest(self):
        """Test creating an empty manifest via builder."""
        builder = DatasetManifestBuilder(
            name="test_dataset",
            description="Test dataset for unit tests",
        )

        manifest = builder.build()

        assert manifest.dataset_name == "test_dataset"
        assert manifest.total_images == 0
        assert manifest.gamma == 1.0
        assert manifest.color_space == "linear_sRGB"

    def test_add_images_to_manifest(self):
        """Test adding images to manifest."""
        builder = DatasetManifestBuilder(name="test_dataset")

        # Add images
        builder.add_image(
            ImageManifestEntry(
                file_path="images/img001.tiff",
                content_hash="a" * 64,  # Valid SHA-256 hex
                input_format="TIFF",
                color_space="linear_sRGB",
                dimensions=(1000, 1500, 3),
                value_range=(0.0, 1.0),
            )
        )

        builder.add_image(
            ImageManifestEntry(
                file_path="images/img002.exr",
                content_hash="b" * 64,
                input_format="EXR",
                color_space="linear_sRGB",
                dimensions=(2000, 3000, 3),
                value_range=(0.0, 5.0),
                has_hdr=True,
            )
        )

        manifest = builder.build()

        assert manifest.total_images == 2
        assert len(manifest.get_images()) == 2

    def test_write_and_load_manifest(self, tmp_path: Path):
        """Test writing and loading manifest from file."""
        builder = DatasetManifestBuilder(
            name="test_dataset",
            description="Test dataset",
        )

        builder.add_image(
            ImageManifestEntry(
                file_path="test.tiff",
                content_hash="a" * 64,
                input_format="TIFF",
                color_space="linear_sRGB",
                dimensions=(100, 100, 3),
                value_range=(0.0, 1.0),
            )
        )

        manifest = builder.build()

        # Write
        manifest_path = tmp_path / "manifest.json"
        manifest.write(manifest_path)

        assert manifest_path.exists()

        # Load
        loaded = ManifestSchema.from_file(manifest_path)

        assert loaded.dataset_name == "test_dataset"
        assert loaded.total_images == 1
        assert loaded.gamma == 1.0

    def test_manifest_validation(self):
        """Test that manifest validates on build."""
        builder = DatasetManifestBuilder(name="test_dataset")

        # Add invalid image (invalid hash length)
        with pytest.raises(ManifestError):
            builder.add_image(
                ImageManifestEntry(
                    file_path="test.tiff",
                    content_hash="invalid_hash",  # Too short
                    input_format="TIFF",
                    color_space="linear_sRGB",
                    dimensions=(100, 100, 3),
                    value_range=(0.0, 1.0),
                )
            )
            builder.build()

    def test_gamma_validation(self):
        """Test that non-linear gamma is rejected."""
        with pytest.raises(ManifestError, match="gamma"):
            builder = DatasetManifestBuilder(
                name="test_dataset",
                gamma=2.2,  # Non-linear gamma
            )
            builder.build()

    def test_dimension_validation(self):
        """Test that invalid dimensions are rejected."""
        builder = DatasetManifestBuilder(name="test_dataset")

        with pytest.raises(ManifestError, match="3 channels"):
            builder.add_image(
                ImageManifestEntry(
                    file_path="test.tiff",
                    content_hash="a" * 64,
                    input_format="TIFF",
                    color_space="linear_sRGB",
                    dimensions=(100, 100, 4),  # RGBA, not RGB
                    value_range=(0.0, 1.0),
                )
            )
            builder.build()


class TestManifestBuilder:
    """Test manifest builder API."""

    def test_builder_chaining(self):
        """Test that builder methods support chaining."""
        manifest = (
            DatasetManifestBuilder(name="test_dataset")
            .add_tag("training")
            .add_tag("linear_sRGB")
            .add_image(
                ImageManifestEntry(
                    file_path="test.tiff",
                    content_hash="a" * 64,
                    input_format="TIFF",
                    color_space="linear_sRGB",
                    dimensions=(100, 100, 3),
                    value_range=(0.0, 1.0),
                )
            )
            .build()
        )

        assert "training" in manifest.schema.dataset.tags
        assert "linear_sRGB" in manifest.schema.dataset.tags
        assert manifest.total_images == 1

    def test_builder_tag_deduplication(self):
        """Test that duplicate tags are not added."""
        builder = DatasetManifestBuilder(name="test_dataset")
        builder.add_tag("training")
        builder.add_tag("training")  # Duplicate

        manifest = builder.build()

        # Should only have one occurrence
        assert manifest.schema.dataset.tags.count("training") == 1


class TestImageInventory:
    """Test image inventory management."""

    def test_get_all_images(self):
        """Test getting all images from manifest."""
        builder = DatasetManifestBuilder(name="test_dataset")

        for i in range(5):
            builder.add_image(
                ImageManifestEntry(
                    file_path=f"img{i:03d}.tiff",
                    content_hash=f"{i}" * 64,
                    input_format="TIFF",
                    color_space="linear_sRGB",
                    dimensions=(100, 100, 3),
                    value_range=(0.0, 1.0),
                )
            )

        manifest = builder.build()

        all_images = manifest.get_images()
        assert len(all_images) == 5

    def test_filter_images_by_tag(self):
        """Test filtering images by tag."""
        builder = DatasetManifestBuilder(name="test_dataset")

        # Add images with different tags
        builder.add_image(
            ImageManifestEntry(
                file_path="img001.tiff",
                content_hash="a" * 64,
                input_format="TIFF",
                color_space="linear_sRGB",
                dimensions=(100, 100, 3),
                value_range=(0.0, 1.0),
                tags=["interior", "training"],
            )
        )

        builder.add_image(
            ImageManifestEntry(
                file_path="img002.tiff",
                content_hash="b" * 64,
                input_format="TIFF",
                color_space="linear_sRGB",
                dimensions=(100, 100, 3),
                value_range=(0.0, 1.0),
                tags=["exterior", "training"],
            )
        )

        builder.add_image(
            ImageManifestEntry(
                file_path="img003.tiff",
                content_hash="c" * 64,
                input_format="TIFF",
                color_space="linear_sRGB",
                dimensions=(100, 100, 3),
                value_range=(0.0, 1.0),
                tags=["interior", "validation"],
            )
        )

        manifest = builder.build()

        # Filter by tag
        interior_images = manifest.get_images(tag="interior")
        assert len(interior_images) == 2

        training_images = manifest.get_images(tag="training")
        assert len(training_images) == 2

    def test_get_image_by_path(self):
        """Test getting image by file path."""
        builder = DatasetManifestBuilder(name="test_dataset")

        builder.add_image(
            ImageManifestEntry(
                file_path="images/special.tiff",
                content_hash="a" * 64,
                input_format="TIFF",
                color_space="linear_sRGB",
                dimensions=(100, 100, 3),
                value_range=(0.0, 1.0),
            )
        )

        manifest = builder.build()

        # Find image
        img = manifest.get_image_by_path("images/special.tiff")
        assert img is not None
        assert img.file_path == "images/special.tiff"

        # Non-existent image
        not_found = manifest.get_image_by_path("nonexistent.tiff")
        assert not_found is None


class TestSchemaVersioning:
    """Test schema versioning functionality."""

    def test_current_schema_version_in_manifest(self):
        """Test that current schema version is embedded in manifest."""
        from transformation_portal.spatial_ai.ingest.validators import CURRENT_SCHEMA_VERSION

        builder = DatasetManifestBuilder(name="test_dataset")
        manifest = builder.build()

        assert manifest.schema_version == CURRENT_SCHEMA_VERSION

    def test_unsupported_version_rejected_on_load(self, tmp_path: Path):
        """Test that unsupported schema version is rejected on load."""
        # Create manifest with unsupported version
        manifest_data = {
            "schema_version": "99.0.0",
            "dataset": {
                "name": "test",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "total_images": 0,
                "gamma": 1.0,
            },
            "images": [],
        }

        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f)

        # Should raise SchemaVersionError
        with pytest.raises(SchemaVersionError, match="99.0.0"):
            ManifestSchema.from_file(manifest_path)

    def test_missing_version_field_rejected(self, tmp_path: Path):
        """Test that missing schema_version field is rejected."""
        # Create manifest without version field
        manifest_data = {
            "dataset": {
                "name": "test",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "total_images": 0,
                "gamma": 1.0,
            },
            "images": [],
        }

        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f)

        # Should raise validation error
        with pytest.raises(ManifestError):
            ManifestSchema.from_file(manifest_path)


class TestManifestFromDirectory:
    """Test loading manifest from dataset directory."""

    def test_load_from_directory(self, tmp_path: Path):
        """Test loading manifest from directory."""
        # Create manifest
        builder = DatasetManifestBuilder(name="test_dataset")
        builder.add_image(
            ImageManifestEntry(
                file_path="test.tiff",
                content_hash="a" * 64,
                input_format="TIFF",
                color_space="linear_sRGB",
                dimensions=(100, 100, 3),
                value_range=(0.0, 1.0),
            )
        )

        manifest = builder.build()

        # Write to directory
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        manifest.write(dataset_dir / "manifest.json")

        # Load from directory
        loaded = ManifestSchema.from_directory(dataset_dir)

        assert loaded.dataset_name == "test_dataset"
        assert loaded.total_images == 1

    def test_load_from_directory_missing_manifest(self, tmp_path: Path):
        """Test that missing manifest raises error."""
        dataset_dir = tmp_path / "empty_dataset"
        dataset_dir.mkdir()

        with pytest.raises(ManifestError, match="Manifest not found"):
            ManifestSchema.from_directory(dataset_dir)


class TestManifestToDictRoundtrip:
    """Test manifest to_dict and JSON roundtrip."""

    def test_to_dict_roundtrip(self):
        """Test that manifest can be converted to dict and back."""
        builder = DatasetManifestBuilder(name="test_dataset")
        builder.add_image(
            ImageManifestEntry(
                file_path="test.tiff",
                content_hash="a" * 64,
                input_format="TIFF",
                color_space="linear_sRGB",
                dimensions=(100, 100, 3),
                value_range=(0.0, 1.0),
            )
        )

        manifest = builder.build()

        # Convert to dict
        manifest_dict = manifest.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(manifest_dict, indent=2)
        assert len(json_str) > 0

        # Load back
        reloaded_dict = json.loads(json_str)
        assert reloaded_dict["dataset"]["name"] == "test_dataset"


class TestManifestDAGForwardCompatibility:
    """Tests for P2-3: Manifest DAG forward compatibility fields."""

    def test_manifest_dag_fields_optional(self):
        """Test that DAG fields (parent_artifact_hash, pipeline_stage) are optional."""
        builder = DatasetManifestBuilder(name="test_dataset")

        # Create image WITHOUT DAG fields (should use defaults)
        builder.add_image(
            ImageManifestEntry(
                file_path="test.tiff",
                content_hash="a" * 64,
                input_format="TIFF",
                color_space="linear_sRGB",
                dimensions=(100, 100, 3),
                value_range=(0.0, 1.0),
                # parent_artifact_hash and pipeline_stage not specified
            )
        )

        # Should build successfully with defaults
        manifest = builder.build()
        images = manifest.get_images()
        assert len(images) == 1
        assert images[0].parent_artifact_hash is None
        assert images[0].pipeline_stage == "linear_ingest"

    def test_manifest_parent_hash_validation(self):
        """Test that parent_artifact_hash is validated if provided."""
        # Valid parent hash (64 char hex)
        valid_entry = ImageManifestEntry(
            file_path="test.tiff",
            content_hash="a" * 64,
            input_format="TIFF",
            color_space="linear_sRGB",
            dimensions=(100, 100, 3),
            value_range=(0.0, 1.0),
            parent_artifact_hash="b" * 64,  # Valid SHA-256
            pipeline_stage="depth_estimation",
        )

        builder = DatasetManifestBuilder(name="test_dataset")
        builder.add_image(valid_entry)
        manifest = builder.build()

        images = manifest.get_images()
        assert images[0].parent_artifact_hash == "b" * 64
        assert images[0].pipeline_stage == "depth_estimation"

        # Invalid parent hash (wrong length)
        with pytest.raises(ManifestError):
            invalid_entry = ImageManifestEntry(
                file_path="test2.tiff",
                content_hash="c" * 64,
                input_format="TIFF",
                color_space="linear_sRGB",
                dimensions=(100, 100, 3),
                value_range=(0.0, 1.0),
                parent_artifact_hash="short",  # Invalid - too short
            )
            builder2 = DatasetManifestBuilder(name="test_dataset2")
            builder2.add_image(invalid_entry)
            builder2.build()  # Should raise validation error

    def test_manifest_forward_compatibility(self):
        """Test that DAG fields enable Phase II artifact lineage tracking."""
        builder = DatasetManifestBuilder(name="lineage_test")

        # Simulate a processing chain:
        # 1. Linear ingest (no parent)
        linear_hash = "a" * 64
        builder.add_image(
            ImageManifestEntry(
                file_path="linear/img001.exr",
                content_hash=linear_hash,
                input_format="EXR",
                color_space="linear_sRGB",
                dimensions=(1000, 1500, 3),
                value_range=(0.0, 2.5),
                has_hdr=True,
                parent_artifact_hash=None,  # Root artifact
                pipeline_stage="linear_ingest",
            )
        )

        # 2. Depth estimation (parent = linear)
        depth_hash = "b" * 64
        builder.add_image(
            ImageManifestEntry(
                file_path="depth/img001_depth.exr",
                content_hash=depth_hash,
                input_format="EXR",
                color_space="linear_sRGB",
                dimensions=(1000, 1500, 3),
                value_range=(0.0, 100.0),
                has_hdr=True,
                parent_artifact_hash=linear_hash,  # Links to linear ingest
                pipeline_stage="depth_estimation",
            )
        )

        # 3. Enhancement (parent = depth)
        builder.add_image(
            ImageManifestEntry(
                file_path="enhanced/img001_enhanced.exr",
                content_hash="c" * 64,
                input_format="EXR",
                color_space="linear_sRGB",
                dimensions=(1000, 1500, 3),
                value_range=(0.0, 1.2),
                has_hdr=True,
                parent_artifact_hash=depth_hash,  # Links to depth
                pipeline_stage="enhancement",
            )
        )

        manifest = builder.build()
        images = manifest.get_images()

        # Verify DAG structure
        assert len(images) == 3

        linear_img = images[0]
        assert linear_img.pipeline_stage == "linear_ingest"
        assert linear_img.parent_artifact_hash is None  # Root

        depth_img = images[1]
        assert depth_img.pipeline_stage == "depth_estimation"
        assert depth_img.parent_artifact_hash == linear_hash  # Points to linear

        enhanced_img = images[2]
        assert enhanced_img.pipeline_stage == "enhancement"
        assert enhanced_img.parent_artifact_hash == depth_hash  # Points to depth


# Pytest markers
pytestmark = [
    pytest.mark.unit,  # Fast unit tests
]
