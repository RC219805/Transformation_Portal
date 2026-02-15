"""Manifest schema for Spatial AI training datasets.

This module defines versioned, validated schemas for dataset manifests:
- Dataset-level metadata (name, description, version)
- Image inventory with provenance references
- Schema versioning with forward/backward compatibility
- JSON schema validation

All manifests are validated on load to prevent schema drift and silent corruption.

Architecture: ADR-023 (Isolation), Issue #890 (Phase I)
Schema Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from .exceptions import ManifestError, SchemaVersionError
from .validators import CURRENT_SCHEMA_VERSION, validate_schema_version

logger = logging.getLogger(__name__)


# Pydantic models for schema validation


class ImageMetadataV1(BaseModel):
    """Metadata for a single image in the dataset (schema v1.0.0).

    This is the validated schema for image entries in manifests.
    """

    file_path: str = Field(..., description="Relative path to image file")
    provenance_path: Optional[str] = Field(None, description="Relative path to provenance JSON sidecar")
    content_hash: str = Field(..., description="SHA-256 hash of linear RGB tensor")
    input_format: str = Field(..., description="Input format (TIFF, PNG, EXR, RAW)")
    color_space: str = Field(..., description="Color space of linear RGB (e.g., 'linear_sRGB', 'camera_native_linear')")
    dimensions: tuple[int, int, int] = Field(..., description="Image dimensions (H, W, C)")
    value_range: tuple[float, float] = Field(..., description="Value range (min, max)")
    has_hdr: bool = Field(False, description="True if max value > 1.0")
    camera_make: Optional[str] = Field(None, description="Camera manufacturer")
    camera_model: Optional[str] = Field(None, description="Camera model")
    tags: List[str] = Field(default_factory=list, description="Custom tags for filtering/organization")

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, v):
        """Validate dimensions are (H, W, 3)."""
        if len(v) != 3:
            raise ValueError(f"Dimensions must be (H, W, C), got {v}")
        if v[2] != 3:
            raise ValueError(f"Expected 3 channels (RGB), got {v[2]}")
        return v

    @field_validator("content_hash")
    @classmethod
    def validate_hash(cls, v):
        """Validate hash is 64-character hex string (SHA-256)."""
        if len(v) != 64:
            raise ValueError(f"SHA-256 hash must be 64 characters, got {len(v)}")
        if not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError("Hash must be hex string")
        return v


class DatasetMetadataV1(BaseModel):
    """Dataset-level metadata (schema v1.0.0)."""

    name: str = Field(..., description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    version: str = Field("1.0.0", description="Dataset version (semantic versioning)")
    created_at: str = Field(..., description="Creation timestamp (ISO 8601)")
    updated_at: str = Field(..., description="Last update timestamp (ISO 8601)")
    total_images: int = Field(..., description="Total number of images")
    color_space: str = Field("linear_sRGB", description="Color space of images")
    gamma: float = Field(1.0, description="Gamma value (must be 1.0 for linear)")
    dtype: str = Field("float32", description="Data type of tensors")
    tags: List[str] = Field(default_factory=list, description="Dataset-level tags")

    @field_validator("gamma")
    @classmethod
    def validate_gamma(cls, v):
        """Validate gamma is 1.0 (linear)."""
        if abs(v - 1.0) > 1e-6:
            raise ValueError(f"Linear datasets must have gamma=1.0, got {v}")
        return v


class ManifestSchemaV1(BaseModel):
    """Complete manifest schema (v1.0.0).

    This is the top-level validated schema for dataset manifests.
    """

    schema_version: str = Field(CURRENT_SCHEMA_VERSION, description="Manifest schema version")
    dataset: DatasetMetadataV1 = Field(..., description="Dataset metadata")
    images: List[ImageMetadataV1] = Field(default_factory=list, description="Image inventory")
    adr_references: List[str] = Field(
        default_factory=lambda: ["ADR-023", "ADR-026"], description="Architecture decision records"
    )
    notes: Optional[str] = Field(None, description="Additional notes")

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v):
        """Validate schema version is supported."""
        from .validators import is_schema_version_supported

        if not is_schema_version_supported(v):
            raise ValueError(f"Unsupported schema version: {v}")
        return v


# Dataclass-based builders (for convenience)


@dataclass
class ImageManifestEntry:
    """Builder for image manifest entries.

    This is a convenience class for constructing ImageMetadataV1 instances.
    """

    file_path: str
    content_hash: str
    input_format: str
    color_space: str
    dimensions: tuple[int, int, int]
    value_range: tuple[float, float]
    provenance_path: Optional[str] = None
    has_hdr: bool = False
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_pydantic(self) -> ImageMetadataV1:
        """Convert to validated Pydantic model."""
        return ImageMetadataV1(**asdict(self))


@dataclass
class DatasetManifestBuilder:
    """Builder for dataset manifests.

    This class provides a fluent API for constructing validated manifests.

    Example:
        >>> builder = DatasetManifestBuilder(name="luxury_estate_training")
        >>> builder.add_image(ImageManifestEntry(...))
        >>> builder.add_image(ImageManifestEntry(...))
        >>> manifest = builder.build()
        >>> manifest.write(Path("manifest.json"))
    """

    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    color_space: str = "linear_sRGB"
    gamma: float = 1.0
    dtype: str = "float32"
    tags: List[str] = field(default_factory=list)
    images: List[ImageManifestEntry] = field(default_factory=list)

    def add_image(self, image: ImageManifestEntry) -> DatasetManifestBuilder:
        """Add image to manifest.

        Args:
            image: Image manifest entry.

        Returns:
            Self for chaining.
        """
        self.images.append(image)
        return self

    def add_tag(self, tag: str) -> DatasetManifestBuilder:
        """Add dataset-level tag.

        Args:
            tag: Tag to add.

        Returns:
            Self for chaining.
        """
        if tag not in self.tags:
            self.tags.append(tag)
        return self

    def build(self) -> ManifestSchema:
        """Build validated manifest.

        Returns:
            ManifestSchema instance.

        Raises:
            ManifestError: If validation fails.
        """
        try:
            now = datetime.now(timezone.utc).isoformat()

            dataset_meta = DatasetMetadataV1(
                name=self.name,
                description=self.description,
                version=self.version,
                created_at=now,
                updated_at=now,
                total_images=len(self.images),
                color_space=self.color_space,
                gamma=self.gamma,
                dtype=self.dtype,
                tags=self.tags,
            )

            image_metas = [img.to_pydantic() for img in self.images]

            schema = ManifestSchemaV1(
                dataset=dataset_meta,
                images=image_metas,
            )

            return ManifestSchema(schema=schema)

        except Exception as e:
            raise ManifestError(detail=f"Manifest build failed: {e}") from e


class ManifestSchema:
    """Versioned, validated manifest for training datasets.

    This class wraps the Pydantic schema and provides high-level operations:
    - Load from JSON file
    - Write to JSON file
    - Validate schema version
    - Query image inventory

    Example:
        >>> manifest = ManifestSchema.from_file(Path("manifest.json"))
        >>> print(f"Dataset: {manifest.dataset_name}")
        >>> print(f"Images: {manifest.total_images}")
        >>> manifest.write(Path("updated_manifest.json"))
    """

    def __init__(self, schema: ManifestSchemaV1):
        """Initialize manifest with validated schema.

        Args:
            schema: Validated Pydantic schema.
        """
        self.schema = schema

    @classmethod
    def from_file(cls, manifest_path: Path) -> ManifestSchema:
        """Load manifest from JSON file.

        Args:
            manifest_path: Path to manifest JSON.

        Returns:
            ManifestSchema instance.

        Raises:
            ManifestError: If load or validation fails.
            SchemaVersionError: If schema version is unsupported.
        """
        try:
            with open(manifest_path) as f:
                data = json.load(f)

            # Validate schema version first
            validate_schema_version(data, manifest_path)

            # Parse and validate with Pydantic
            schema = ManifestSchemaV1(**data)

            return cls(schema=schema)

        except SchemaVersionError:
            raise  # Re-raise schema version errors

        except Exception as e:
            raise ManifestError(
                detail=f"Failed to load manifest: {e}",
                manifest_path=manifest_path,
            ) from e

    @classmethod
    def from_directory(
        cls,
        dataset_dir: Path,
        manifest_filename: str = "manifest.json",
    ) -> ManifestSchema:
        """Load manifest from dataset directory.

        Args:
            dataset_dir: Dataset directory containing manifest.
            manifest_filename: Manifest filename (default: "manifest.json").

        Returns:
            ManifestSchema instance.

        Raises:
            ManifestError: If manifest not found or invalid.
        """
        manifest_path = dataset_dir / manifest_filename

        if not manifest_path.exists():
            raise ManifestError(
                detail=f"Manifest not found: {manifest_filename}",
                manifest_path=manifest_path,
            )

        return cls.from_file(manifest_path)

    def write(self, output_path: Path, indent: int = 2) -> None:
        """Write manifest to JSON file.

        Args:
            output_path: Output path for manifest JSON.
            indent: JSON indentation (default: 2).

        Raises:
            ManifestError: If write fails.
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dict using Pydantic's model_dump
            data = self.schema.model_dump()

            # Update timestamp
            data["dataset"]["updated_at"] = datetime.now(timezone.utc).isoformat()

            with open(output_path, "w") as f:
                json.dump(data, f, indent=indent)

            logger.info(f"Wrote manifest: {output_path}")

        except Exception as e:
            raise ManifestError(
                detail=f"Failed to write manifest: {e}",
                manifest_path=output_path,
            ) from e

    def validate(self) -> None:
        """Validate manifest schema.

        This is a no-op if manifest was constructed properly (Pydantic validates on init),
        but provided for explicit validation after modifications.

        Raises:
            ManifestError: If validation fails.
        """
        try:
            # Re-validate with Pydantic
            ManifestSchemaV1(**self.schema.model_dump())
            logger.debug("Manifest validation passed")

        except Exception as e:
            raise ManifestError(detail=f"Validation failed: {e}") from e

    # Convenience properties

    @property
    def dataset_name(self) -> str:
        """Get dataset name."""
        return self.schema.dataset.name

    @property
    def total_images(self) -> int:
        """Get total image count."""
        return self.schema.dataset.total_images

    @property
    def schema_version(self) -> str:
        """Get schema version."""
        return self.schema.schema_version

    @property
    def color_space(self) -> str:
        """Get color space."""
        return self.schema.dataset.color_space

    @property
    def gamma(self) -> float:
        """Get gamma value."""
        return self.schema.dataset.gamma

    def get_images(self, tag: Optional[str] = None) -> List[ImageMetadataV1]:
        """Get images, optionally filtered by tag.

        Args:
            tag: Optional tag to filter by.

        Returns:
            List of image metadata entries.
        """
        if tag is None:
            return self.schema.images

        return [img for img in self.schema.images if tag in img.tags]

    def get_image_by_path(self, file_path: str) -> Optional[ImageMetadataV1]:
        """Get image metadata by file path.

        Args:
            file_path: File path to search for.

        Returns:
            Image metadata if found, None otherwise.
        """
        for img in self.schema.images:
            if img.file_path == file_path:
                return img
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary.

        Returns:
            Dictionary representation.
        """
        return self.schema.model_dump()
