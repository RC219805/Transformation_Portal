"""Provenance capture for Spatial AI linear ingest pipeline.

This module provides comprehensive metadata extraction and tracking:
- EXIF metadata from images (camera, lens, exposure settings)
- Ingest metadata (timestamps, file hashes, loader version)
- Transform chain documentation
- Sidecar JSON generation

All provenance data is versioned and validated for reproducibility.

Architecture: ADR-023 (Isolation), Issue #890 (Phase I)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

from .exceptions import ProvenanceError
from .validators import CURRENT_SCHEMA_VERSION

logger = logging.getLogger(__name__)


@dataclass
class CameraMetadata:
    """Camera and lens metadata extracted from EXIF.

    All fields are optional since not all images have full EXIF data.
    """

    make: Optional[str] = None
    model: Optional[str] = None
    lens_make: Optional[str] = None
    lens_model: Optional[str] = None
    focal_length: Optional[float] = None  # mm
    focal_length_35mm: Optional[int] = None  # 35mm equivalent
    aperture: Optional[float] = None  # f-number (e.g., 2.8)
    shutter_speed: Optional[str] = None  # e.g., "1/250"
    iso: Optional[int] = None
    flash: Optional[str] = None
    orientation: Optional[int] = None
    exposure_bias: Optional[str] = None
    exposure_mode: Optional[str] = None
    exposure_program: Optional[str] = None
    metering_mode: Optional[str] = None
    white_balance: Optional[str] = None
    scene_type: Optional[str] = None
    datetime_original: Optional[str] = None
    datetime_digitized: Optional[str] = None
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    gps_altitude: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class IngestMetadata:
    """Metadata about the ingest process itself."""

    timestamp: str
    source_file: str
    source_file_size_bytes: int
    source_file_hash_sha256: str
    loader_module: str
    loader_version: str = "1.0.0"  # Corresponds to Issue #890 Phase I
    schema_version: str = field(default_factory=lambda: CURRENT_SCHEMA_VERSION)


@dataclass
class TransformMetadata:
    """Metadata about transforms applied during ingest."""

    gamma: float
    bit_depth: int
    dtype: str
    color_space: str = "linear_sRGB"  # Phase I: linear sRGB, Phase II: ACEScg
    demosaic_method: Optional[str] = None  # For RAW files
    white_balance_method: Optional[str] = None  # For RAW files
    color_matrix: Optional[str] = None  # For RAW files


@dataclass
class OutputMetadata:
    """Metadata about the output from ingest."""

    content_hash_sha256: str
    hash_algorithm: str = "sha256"
    output_shape: tuple[int, int, int] = (0, 0, 0)  # (H, W, C)
    value_range_min: float = 0.0
    value_range_max: float = 0.0
    has_hdr_values: bool = False  # True if max > 1.0


@dataclass
class ProvenanceData:
    """Complete provenance record for an ingested image.

    This is the top-level container for all provenance metadata.
    """

    camera: CameraMetadata
    ingest: IngestMetadata
    transform: TransformMetadata
    output: OutputMetadata
    adr_references: list[str] = field(default_factory=lambda: ["ADR-023", "ADR-026"])
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "camera": self.camera.to_dict(),
            "ingest": asdict(self.ingest),
            "transform": asdict(self.transform),
            "output": asdict(self.output),
            "adr_references": self.adr_references,
            "notes": self.notes,
        }


class ProvenanceCapture:
    """Capture and track provenance for linear ingest pipeline.

    This class handles:
    - EXIF extraction from images
    - File hash computation
    - Ingest metadata collection
    - Provenance JSON generation
    - Sidecar file writing

    Example:
        >>> capture = ProvenanceCapture()
        >>> tensor = load_image("test.tiff")
        >>> prov = capture.capture(
        ...     source_path=Path("test.tiff"),
        ...     tensor=tensor,
        ...     gamma=1.0,
        ...     bit_depth=32,
        ... )
        >>> capture.write_sidecar(prov, Path("test_provenance.json"))
    """

    def __init__(self, loader_module: str = "spatial_ai.ingest.linear_decoder"):
        """Initialize provenance capture.

        Args:
            loader_module: Module name for provenance tracking.
        """
        self.loader_module = loader_module

    def capture(
        self,
        source_path: Path,
        tensor: np.ndarray,
        gamma: float,
        bit_depth: int,
        dtype: str = "float32",
        color_space: str = "linear_sRGB",
        demosaic_method: Optional[str] = None,
        white_balance_method: Optional[str] = None,
        color_matrix: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> ProvenanceData:
        """Capture complete provenance for an ingested image.

        Args:
            source_path: Path to source file.
            tensor: Output tensor (H, W, C) float32.
            gamma: Gamma used for decode.
            bit_depth: Output bit depth.
            dtype: Output dtype.
            color_space: Output color space.
            demosaic_method: Demosaic method (for RAW).
            white_balance_method: White balance method (for RAW).
            color_matrix: Color matrix applied (for RAW).
            notes: Optional notes about this ingest.

        Returns:
            ProvenanceData with all metadata.

        Raises:
            ProvenanceError: If extraction fails.
        """
        try:
            # Extract EXIF
            camera = self._extract_exif(source_path)

            # Compute source file hash
            source_hash = self._compute_file_hash(source_path)

            # Create ingest metadata
            ingest = IngestMetadata(
                timestamp=datetime.now(timezone.utc).isoformat(),
                source_file=str(source_path),
                source_file_size_bytes=source_path.stat().st_size,
                source_file_hash_sha256=source_hash,
                loader_module=self.loader_module,
            )

            # Create transform metadata
            transform = TransformMetadata(
                gamma=gamma,
                bit_depth=bit_depth,
                dtype=dtype,
                color_space=color_space,
                demosaic_method=demosaic_method,
                white_balance_method=white_balance_method,
                color_matrix=color_matrix,
            )

            # Compute output metadata
            content_hash = self._compute_array_hash(tensor)
            output = OutputMetadata(
                content_hash_sha256=content_hash,
                output_shape=tensor.shape,
                value_range_min=float(np.min(tensor)),
                value_range_max=float(np.max(tensor)),
                has_hdr_values=float(np.max(tensor)) > 1.0,
            )

            # Assemble provenance
            return ProvenanceData(
                camera=camera,
                ingest=ingest,
                transform=transform,
                output=output,
                notes=notes,
            )

        except Exception as e:
            raise ProvenanceError(source="provenance capture", detail=str(e)) from e

    def write_sidecar(
        self,
        provenance: ProvenanceData,
        output_path: Path,
        indent: int = 2,
    ) -> None:
        """Write provenance data to JSON sidecar file.

        Args:
            provenance: Provenance data to write.
            output_path: Path for sidecar JSON file.
            indent: JSON indentation (default: 2).

        Raises:
            ProvenanceError: If write fails.
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(provenance.to_dict(), f, indent=indent)

            logger.debug(f"Wrote provenance sidecar: {output_path}")

        except Exception as e:
            raise ProvenanceError(source="sidecar write", detail=str(e)) from e

    def load_sidecar(self, sidecar_path: Path) -> Dict[str, Any]:
        """Load provenance from JSON sidecar file.

        Args:
            sidecar_path: Path to sidecar JSON.

        Returns:
            Provenance dictionary.

        Raises:
            ProvenanceError: If load fails.
        """
        try:
            with open(sidecar_path) as f:
                return json.load(f)

        except Exception as e:
            raise ProvenanceError(source="sidecar load", detail=str(e)) from e

    def _extract_exif(self, image_path: Path) -> CameraMetadata:
        """Extract EXIF metadata from image file.

        Args:
            image_path: Path to image file.

        Returns:
            CameraMetadata with extracted fields.

        Note:
            Missing EXIF data is not an error; fields will be None.
        """
        camera = CameraMetadata()

        try:
            with Image.open(image_path) as img:
                exif_data = img.getexif()

                if not exif_data:
                    logger.debug(f"No EXIF data found in {image_path.name}")
                    return camera

                # Map EXIF tags to CameraMetadata fields
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)

                    # Camera
                    if tag_name == "Make":
                        camera.make = str(value).strip()
                    elif tag_name == "Model":
                        camera.model = str(value).strip()
                    elif tag_name == "LensMake":
                        camera.lens_make = str(value).strip()
                    elif tag_name == "LensModel":
                        camera.lens_model = str(value).strip()

                    # Exposure
                    elif tag_name == "FocalLength":
                        camera.focal_length = self._parse_focal_length(value)
                    elif tag_name == "FocalLengthIn35mmFilm":
                        camera.focal_length_35mm = int(value) if value else None
                    elif tag_name == "FNumber":
                        camera.aperture = self._parse_aperture(value)
                    elif tag_name == "ExposureTime":
                        camera.shutter_speed = self._parse_shutter_speed(value)
                    elif tag_name == "ISOSpeedRatings" or tag_name == "PhotographicSensitivity":
                        camera.iso = int(value) if value else None
                    elif tag_name == "Flash":
                        camera.flash = str(value)
                    elif tag_name == "ExposureBiasValue":
                        camera.exposure_bias = str(value)
                    elif tag_name == "ExposureMode":
                        camera.exposure_mode = str(value)
                    elif tag_name == "ExposureProgram":
                        camera.exposure_program = self._parse_exposure_program(value)
                    elif tag_name == "MeteringMode":
                        camera.metering_mode = self._parse_metering_mode(value)
                    elif tag_name == "WhiteBalance":
                        camera.white_balance = self._parse_white_balance(value)
                    elif tag_name == "SceneCaptureType":
                        camera.scene_type = str(value)

                    # Orientation
                    elif tag_name == "Orientation":
                        camera.orientation = int(value) if value else None

                    # Datetime
                    elif tag_name == "DateTimeOriginal":
                        camera.datetime_original = str(value)
                    elif tag_name == "DateTimeDigitized":
                        camera.datetime_digitized = str(value)

                    # GPS (if present)
                    elif tag_name == "GPSInfo":
                        gps_data = self._parse_gps(value)
                        if gps_data:
                            camera.gps_latitude = gps_data.get("latitude")
                            camera.gps_longitude = gps_data.get("longitude")
                            camera.gps_altitude = gps_data.get("altitude")

        except Exception as e:
            logger.warning(f"EXIF extraction failed for {image_path.name}: {e}")
            # Return empty CameraMetadata (not an error)

        return camera

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file.

        Args:
            file_path: Path to file.

        Returns:
            Hex string of SHA-256 hash.
        """
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _compute_array_hash(self, array: np.ndarray) -> str:
        """Compute SHA-256 hash of array content.

        Args:
            array: NumPy array.

        Returns:
            Hex string of SHA-256 hash.
        """
        hasher = hashlib.sha256()
        hasher.update(array.tobytes())
        return hasher.hexdigest()

    # EXIF parsing helpers

    @staticmethod
    def _parse_focal_length(value: Any) -> Optional[float]:
        """Parse focal length from EXIF value."""
        try:
            if isinstance(value, tuple) and len(value) == 2:
                return float(value[0]) / float(value[1])
            return float(value)
        except (ValueError, TypeError, ZeroDivisionError):
            return None

    @staticmethod
    def _parse_aperture(value: Any) -> Optional[float]:
        """Parse aperture (f-number) from EXIF value."""
        try:
            if isinstance(value, tuple) and len(value) == 2:
                return float(value[0]) / float(value[1])
            return float(value)
        except (ValueError, TypeError, ZeroDivisionError):
            return None

    @staticmethod
    def _parse_shutter_speed(value: Any) -> Optional[str]:
        """Parse shutter speed from EXIF value."""
        try:
            if isinstance(value, tuple) and len(value) == 2:
                num, denom = value
                if num == 1:
                    return f"1/{denom}"
                else:
                    return f"{num}/{denom}"
            return str(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_exposure_program(value: int) -> str:
        """Parse exposure program from EXIF value."""
        programs = {
            0: "Not defined",
            1: "Manual",
            2: "Program AE",
            3: "Aperture-priority AE",
            4: "Shutter-priority AE",
            5: "Creative (slow speed)",
            6: "Action (high speed)",
            7: "Portrait",
            8: "Landscape",
        }
        return programs.get(value, f"Unknown ({value})")

    @staticmethod
    def _parse_metering_mode(value: int) -> str:
        """Parse metering mode from EXIF value."""
        modes = {
            0: "Unknown",
            1: "Average",
            2: "Center-weighted average",
            3: "Spot",
            4: "Multi-spot",
            5: "Multi-segment",
            6: "Partial",
            255: "Other",
        }
        return modes.get(value, f"Unknown ({value})")

    @staticmethod
    def _parse_white_balance(value: int) -> str:
        """Parse white balance from EXIF value."""
        wb_modes = {
            0: "Auto",
            1: "Manual",
        }
        return wb_modes.get(value, f"Unknown ({value})")

    @staticmethod
    def _parse_gps(gps_info: Dict[int, Any]) -> Optional[Dict[str, float]]:
        """Parse GPS coordinates from EXIF GPSInfo.

        Args:
            gps_info: GPS info dictionary from EXIF.

        Returns:
            Dictionary with latitude, longitude, altitude, or None.
        """
        try:
            # GPS tag IDs (not all PILs expose GPSInfo tags properly)
            # This is a simplified parser
            lat = gps_info.get(2)  # GPSLatitude
            lat_ref = gps_info.get(1)  # GPSLatitudeRef
            lon = gps_info.get(4)  # GPSLongitude
            lon_ref = gps_info.get(3)  # GPSLongitudeRef
            alt = gps_info.get(6)  # GPSAltitude

            if not lat or not lon:
                return None

            # Convert DMS to decimal
            def dms_to_decimal(dms, ref):
                if isinstance(dms, tuple) and len(dms) == 3:
                    deg = float(dms[0][0]) / float(dms[0][1]) if isinstance(dms[0], tuple) else float(dms[0])
                    min = float(dms[1][0]) / float(dms[1][1]) if isinstance(dms[1], tuple) else float(dms[1])
                    sec = float(dms[2][0]) / float(dms[2][1]) if isinstance(dms[2], tuple) else float(dms[2])
                    decimal = deg + (min / 60.0) + (sec / 3600.0)
                    if ref in ["S", "W"]:
                        decimal = -decimal
                    return decimal
                return None

            latitude = dms_to_decimal(lat, lat_ref)
            longitude = dms_to_decimal(lon, lon_ref)

            altitude = None
            if alt:
                if isinstance(alt, tuple) and len(alt) == 2:
                    altitude = float(alt[0]) / float(alt[1])
                else:
                    altitude = float(alt)

            return {
                "latitude": latitude,
                "longitude": longitude,
                "altitude": altitude,
            }

        except Exception as e:
            logger.debug(f"GPS parsing failed: {e}")
            return None
