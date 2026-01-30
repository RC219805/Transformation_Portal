"""PBR Map File Writer for Lux Depth V3.

Handles atomic writing of PBR maps (normal, roughness, AO) using shared atomic write primitives.
"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

from .io_atomic import atomic_write_pil_png

logger = logging.getLogger(__name__)


def write_pbr_maps(
    normal_map: np.ndarray,
    roughness_map: np.ndarray,
    ao_map: np.ndarray,
    output_dir: Path,
    base_name: str
) -> Dict[str, Path]:
    """Write PBR maps to disk with atomic operations.

    Args:
        normal_map: RGB uint8 normal map (H, W, 3)
        roughness_map: Grayscale uint8 roughness map (H, W)
        ao_map: Grayscale uint8 AO map (H, W)
        output_dir: Output directory
        base_name: Base filename (without extension)

    Returns:
        Dict mapping map type to written file path:
            {"normal": Path, "roughness": Path, "ao": Path}

    Raises:
        IOError: If all maps fail to write

    Example:
        >>> paths = write_pbr_maps(
        ...     normal, roughness, ao,
        ...     Path("output"), "render_001"
        ... )
        >>> assert paths["normal"] == Path("output/render_001_normal.png")
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    written_paths = {}
    errors = []

    maps_to_write = [
        ("normal", normal_map, f"{base_name}_normal.png"),
        ("roughness", roughness_map, f"{base_name}_roughness.png"),
        ("ao", ao_map, f"{base_name}_ao.png"),
    ]

    for map_type, map_data, filename in maps_to_write:
        output_path = output_dir / filename

        try:
            # Convert numpy array to PIL Image
            if map_data.ndim == 2:
                pil_image = Image.fromarray(map_data, 'L')
            elif map_data.ndim == 3 and map_data.shape[2] == 3:
                pil_image = Image.fromarray(map_data, 'RGB')
            else:
                raise ValueError(f"Invalid map shape for {map_type}: {map_data.shape}")

            # Use shared atomic write helper
            atomic_write_pil_png(output_path, pil_image, optimize=True)

            written_paths[map_type] = output_path
            logger.info(f"Wrote {map_type} map: {output_path}")

        except Exception as e:
            error_msg = f"Failed to write {map_type} map: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    # Only raise if ALL maps failed
    if not written_paths:
        raise IOError(f"All PBR map writes failed: {'; '.join(errors)}")

    return written_paths
