"""Input modeling helpers for Lux Depth V3 batch execution.

This module provides the canonical ``ImageInput`` contract used by discovery,
grouping, and orchestrator scheduling paths. The contract is designed to be:

- **Backward compatible**: Existing ``ImageInput(path=...)`` construction works
- **Lightweight**: No filesystem access or expensive operations at construction
- **Serializable**: Deterministic ``to_dict()``/``to_json()`` for machine-mode
- **Validatable**: Explicit ``validate()`` method for opt-in integrity checks

Usage:
    >>> from transformation_portal.lux_depth_v3.input_manager import ImageInput
    >>> img = ImageInput(path="my_image.jpg")
    >>> img.format
    '.jpg'
    >>> img.is_supported
    True
    >>> img.validate(check_exists=True)  # raises if file missing

Schema Version: 1.0
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple, Union

from transformation_portal.core.raw_formats import RAW_EXTENSIONS as _RAW_EXTENSIONS
from transformation_portal.ingest.canonical_json import dumps_json
from transformation_portal.lux_depth_v3.path_aliasing import normalize_lexical_path

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InputValidationError(Exception):
    """Base exception for input validation failures."""


class UnsupportedFormatError(InputValidationError):
    """Raised when image format is not supported."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Standard image formats supported by PIL
_STANDARD_EXTENSIONS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".tiff",
        ".tif",
        ".webp",
        ".bmp",
    }
)

# _RAW_EXTENSIONS is imported from raw_loader (single source of truth);
# see the import block at the top of this module.
SUPPORTED_EXTENSIONS = _STANDARD_EXTENSIONS | _RAW_EXTENSIONS


# ---------------------------------------------------------------------------
# Typed metadata contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputImageMetadata:
    """Typed metadata for input images.

    This is a frozen (immutable) contract for structured metadata. The fields
    capture file-level and discovery-level facts about an input image.

    Attributes:
        image_sha256: SHA-256 hash of file contents (hex string)
        image_size_bytes: File size in bytes
        image_dimensions: (width, height) tuple
        source_format: Detected format (lowercase extension, e.g. ".jpg")
        color_space: Color space if known (e.g. "sRGB", "ProPhoto")
        bit_depth: Bits per channel if known (8, 16, etc.)
        raw_metadata: Escape hatch for additional unstructured metadata
    """

    image_sha256: Optional[str] = None
    image_size_bytes: Optional[int] = None
    image_dimensions: Optional[Tuple[int, int]] = None
    source_format: Optional[str] = None
    color_space: Optional[str] = None
    bit_depth: Optional[int] = None
    raw_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary, omitting None values."""
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None}

    @classmethod
    def from_mapping(cls, data: Optional[Dict[str, Any]]) -> Optional["InputImageMetadata"]:
        """Create from a dictionary, extracting known fields.

        Unknown keys are collected into raw_metadata for round-trip preservation.

        Args:
            data: Dictionary of metadata, or None

        Returns:
            InputImageMetadata instance, or None if data is None/empty
        """
        if not data:
            return None

        known_keys = {
            "image_sha256",
            "image_size_bytes",
            "image_dimensions",
            "source_format",
            "color_space",
            "bit_depth",
            "raw_metadata",
        }

        known = {}
        extra = {}

        for key, value in data.items():
            if key in known_keys:
                # JSON serialization converts tuples to lists, so we restore
                # tuple type for image_dimensions on deserialization
                if key == "image_dimensions" and isinstance(value, list):
                    value = tuple(value)
                known[key] = value
            else:
                extra[key] = value

        # Merge extra into raw_metadata
        if extra:
            existing_raw = known.get("raw_metadata") or {}
            if isinstance(existing_raw, dict):
                merged = {**existing_raw, **extra}
            else:
                merged = extra
            known["raw_metadata"] = merged

        return cls(**known)


# ---------------------------------------------------------------------------
# ImageInput contract
# ---------------------------------------------------------------------------


@dataclass
class ImageInput:
    """Represents an input image for processing.

    This is the canonical input contract for Lux Depth V3 pipelines. It wraps
    a file path with optional metadata and provides validation/serialization
    helpers.

    The class is intentionally kept mutable and lightweight:
    - No filesystem access during construction
    - Path is normalized lexically (no symlink resolution)
    - Metadata remains a flexible dict for backward compatibility

    Attributes:
        path: Normalized absolute path to the image file
        metadata: Optional dictionary of metadata (legacy format)

    Class Variables:
        SCHEMA_VERSION: Version string for serialization compatibility

    Example:
        >>> img = ImageInput(path="/path/to/image.jpg")
        >>> img.format
        '.jpg'
        >>> img.validate(check_exists=True)
    """

    path: Path
    metadata: Optional[Dict[str, Any]] = field(default=None)

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    def __post_init__(self) -> None:
        """Normalize path and validate construction arguments.

        Raises:
            ValueError: If path is empty or metadata is not dict-like
        """
        # Coerce path to Path if needed
        if not isinstance(self.path, Path):
            self.path = Path(self.path)

        # Validate non-empty path (reject empty, whitespace-only, or bare ".")
        path_str = str(self.path).strip()
        if not path_str or path_str == ".":
            raise ValueError("ImageInput path cannot be empty")

        # Normalize path lexically (no filesystem access)
        self.path = normalize_lexical_path(self.path)

        # Validate metadata is dict-like if provided
        if self.metadata is not None:
            if not isinstance(self.metadata, dict):
                raise ValueError(f"ImageInput metadata must be a dict, got {type(self.metadata).__name__}")

    # -----------------------------------------------------------------------
    # Format detection properties
    # -----------------------------------------------------------------------

    @property
    def format(self) -> str:
        """Return the file format (lowercase extension including dot).

        Example:
            >>> ImageInput(path="photo.JPG").format
            '.jpg'
        """
        return self.path.suffix.lower()

    @property
    def is_raw(self) -> bool:
        """Check if the input is a RAW camera file.

        Example:
            >>> ImageInput(path="photo.CR2").is_raw
            True
        """
        return self.format in _RAW_EXTENSIONS

    @property
    def is_supported(self) -> bool:
        """Check if the format is in the supported extensions set.

        Example:
            >>> ImageInput(path="photo.jpg").is_supported
            True
            >>> ImageInput(path="doc.pdf").is_supported
            False
        """
        return self.format in SUPPORTED_EXTENSIONS

    # -----------------------------------------------------------------------
    # Typed metadata access
    # -----------------------------------------------------------------------

    @property
    def metadata_model(self) -> Optional[InputImageMetadata]:
        """Parse legacy metadata dict into typed InputImageMetadata.

        Returns:
            InputImageMetadata if metadata is present and parseable, else None
        """
        return InputImageMetadata.from_mapping(self.metadata)

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def validate(
        self,
        *,
        check_exists: bool = False,
        check_supported: bool = True,
    ) -> None:
        """Validate the input image.

        Args:
            check_exists: If True, verify the file exists on disk
            check_supported: If True, verify the format is supported

        Raises:
            UnsupportedFormatError: If format not in SUPPORTED_EXTENSIONS
            FileNotFoundError: If check_exists=True and file doesn't exist
            InputValidationError: For other validation failures
        """
        if check_supported and not self.is_supported:
            supported_list = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise UnsupportedFormatError(f"Unsupported image format: {self.format}. " f"Supported formats: {supported_list}")

        if check_exists and not self.path.exists():
            raise FileNotFoundError(f"Image file not found: {self.path}")

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary with schema version.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        result: Dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "path": str(self.path),
        }
        if self.metadata is not None:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageInput":
        """Deserialize from dictionary.

        Accepts payloads with or without schema_version for backward
        compatibility with legacy serialized data.

        Args:
            data: Dictionary with 'path' and optional 'metadata'

        Returns:
            ImageInput instance

        Raises:
            ValueError: If required 'path' key is missing
        """
        if "path" not in data:
            raise ValueError("ImageInput.from_dict requires 'path' key")

        # schema_version is accepted but not enforced (forward compatibility)
        return cls(
            path=Path(data["path"]),
            metadata=data.get("metadata"),
        )

    def to_json(self) -> str:
        """Serialize to deterministic JSON string.

        Uses sorted keys and compact separators for reproducibility.

        Returns:
            JSON string representation
        """
        return dumps_json(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ImageInput":
        """Deserialize from JSON string.

        Args:
            json_str: JSON string representation

        Returns:
            ImageInput instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    # -----------------------------------------------------------------------
    # Factory methods with opt-in enrichment
    # -----------------------------------------------------------------------

    @classmethod
    def from_path(
        cls,
        path: Union[str, Path],
        *,
        compute_hash: bool = False,
        probe_dimensions: bool = False,
        detect_format: bool = True,
    ) -> "ImageInput":
        """Create ImageInput with optional metadata enrichment.

        This factory method allows opt-in collection of expensive metadata
        like file hashes and image dimensions without making the default
        constructor slow.

        Args:
            path: Path to the image file
            compute_hash: If True, compute SHA-256 hash (requires file read)
            probe_dimensions: If True, read image dimensions (requires PIL)
            detect_format: If True, include format in metadata (no I/O)

        Returns:
            ImageInput with populated metadata

        Raises:
            FileNotFoundError: If file doesn't exist and enrichment requested
        """
        # Normalize path first (expand ~, resolve symlinks, etc.) to ensure
        # consistent behavior for existence checks, hashing, probing, and the
        # returned ImageInput.path value.
        path = normalize_lexical_path(Path(path))
        metadata: Dict[str, Any] = {}

        if detect_format:
            metadata["source_format"] = path.suffix.lower()

        if compute_hash:
            if not path.exists():
                raise FileNotFoundError(f"Cannot compute hash: file not found: {path}")
            sha256_hash = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)
            metadata["image_sha256"] = sha256_hash.hexdigest()
            metadata["image_size_bytes"] = path.stat().st_size

        if probe_dimensions:
            if not path.exists():
                raise FileNotFoundError(f"Cannot probe dimensions: file not found: {path}")
            # Lazy import PIL only when needed
            try:
                from PIL import Image
            except ImportError as e:
                raise ImportError("PIL/Pillow is required for probe_dimensions=True") from e

            with Image.open(path) as img:
                metadata["image_dimensions"] = img.size  # (width, height)

        return cls(path=path, metadata=metadata if metadata else None)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "ImageInput",
    "InputImageMetadata",
    "InputValidationError",
    "UnsupportedFormatError",
    "SUPPORTED_EXTENSIONS",
]
