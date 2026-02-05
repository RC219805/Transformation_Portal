"""
Export Manager & Atomic Writer.

Handles safe image writing with automatic format selection based on data properties.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Literal, Optional, Union

import cv2
import numpy as np
from pydantic import BaseModel, Field

from .autotune_helpers import ImageStats, compute_image_stats

logger = logging.getLogger(__name__)

# Try importing specialized IO
try:
    import imageio

    IMAGEIO_AVAIL = True
except ImportError:
    IMAGEIO_AVAIL = False


class ExportConfig(BaseModel):
    """Configuration for image export."""

    format: Literal["auto", "jpg", "png", "tiff", "exr"] = "auto"
    quality_jpg: int = Field(default=95, ge=1, le=100)
    compression_png: int = Field(default=4, ge=0, le=9)
    use_16bit_tiff: bool = True
    preserve_metadata: bool = True
    overwrite: bool = True


def autotune_export_config(image: np.ndarray, base_config: ExportConfig) -> ExportConfig:
    """
    Update config format based on image statistics.

    Logic:
    - HDR data (>1.0 float) -> EXR
    - Transparency -> PNG
    - High bit depth (>8b) -> TIFF or PNG 16-bit
    - Standard photo -> JPG
    """
    if base_config.format != "auto":
        return base_config

    stats = compute_image_stats(image)
    new_config = base_config.copy()

    if stats.is_hdr:
        new_config.format = "exr"
        logger.info("Autotune: Detected HDR content. Selecting EXR.")
    elif stats.has_alpha:
        new_config.format = "png"
        logger.info("Autotune: Detected transparency. Selecting PNG.")
    elif stats.bit_depth_hint > 8:
        new_config.format = "tiff"  # or png 16
        logger.info("Autotune: Detected high bit-depth. Selecting TIFF.")
    else:
        new_config.format = "jpg"
        logger.info("Autotune: Standard dynamic range. Selecting JPG.")

    return new_config


class ExportManager:
    """
    Robust image writer.

    Features:
    - Atomic writes (write temp -> rename) to prevents corruption
    - Directory creation
    - Format Autotuning
    """

    @staticmethod
    def save_image(image: np.ndarray, path: Union[str, Path], config: Optional[ExportConfig] = None) -> Path:
        """
        Save image to disk safely.

        Args:
            image: Numpy array image data.
            path: Destination path.
            config: Export settings.

        Returns:
            Final Path object of saved file.
        """
        path = Path(path)
        config = config or ExportConfig()

        # 1. Resolve Format
        if config.format == "auto":
            config = autotune_export_config(image, config)
            # Update extension if auto changed it
            if path.suffix.lower().strip(".") != config.format:
                path = path.with_suffix(f".{config.format}")

        # 2. Prepare Directory
        path.parent.mkdir(parents=True, exist_ok=True)

        # 3. Atomic Write Pattern
        # Write to .tmp file first, then rename
        temp_path = path.with_suffix(f".{path.suffix}.tmp")

        try:
            ExportManager._write_file(image, temp_path, config)

            # Atomic Move
            if path.exists() and not config.overwrite:
                raise FileExistsError(f"File already exists: {path}")

            shutil.move(str(temp_path), str(path))
            logger.info(f"Saved image to {path} ({config.format})")
            return path

        except Exception as e:
            logger.error(f"Failed to save image to {path}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    @staticmethod
    def _write_file(img: np.ndarray, path: Path, config: ExportConfig):
        """Internal dispatcher for writing formats."""
        ext = config.format.lower()

        # Convert RGB to BGR for OpenCV if needed
        # Assuming input is usually RGB from PIL/PyTorch
        if img.ndim == 3 and img.shape[2] == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        else:
            img_bgr = img

        if ext in ["jpg", "jpeg"]:
            cv2.imwrite(str(path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), config.quality_jpg])

        elif ext == "png":
            # 16-bit PNG check
            if img.dtype == np.uint16:
                cv2.imwrite(str(path), img_bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), config.compression_png])
            else:
                # Standard PNG
                cv2.imwrite(str(path), img_bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), config.compression_png])

        elif ext == "exr":
            if not IMAGEIO_AVAIL:
                logger.warning("imageio not installed. Falling back to OpenCV EXR.")
                # OpenCV handles float32 EXR
                cv2.imwrite(str(path), img_bgr.astype(np.float32))
            else:
                # imageio is often better for multi-layer EXR
                imageio.imwrite(str(path), img.astype(np.float32), format="EXR")

        elif ext in ["tif", "tiff"]:
            if config.use_16bit_tiff and img.dtype == np.uint8:
                # Upscale to 16-bit if requested? Usually we just save what we have.
                # Here we assume if user wants 16-bit, they provided 16-bit or float.
                pass
            cv2.imwrite(str(path), img_bgr)

        else:
            # Fallback
            cv2.imwrite(str(path), img_bgr)
