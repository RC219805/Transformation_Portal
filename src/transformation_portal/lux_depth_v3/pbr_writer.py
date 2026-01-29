"""PBR Map File Writer for Lux Depth V3.

Handles atomic writing of PBR maps (normal, roughness, AO) using PIL only.
"""

import logging
import os
from pathlib import Path
import tempfile
from typing import Dict

import numpy as np
from PIL import Image

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
        IOError: If writing fails

    Example:
        >>> paths = write_pbr_maps(
        ...     normal, roughness, ao,
        ...     Path("output"), "render_001"
        ... )
        >>> assert paths["normal"] == Path("output/render_001_normal.png")
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    written_paths = {}
    temp_files = []

    try:
        # Define output paths
        maps_to_write = [
            ("normal", normal_map, f"{base_name}_normal.png"),
            ("roughness", roughness_map, f"{base_name}_roughness.png"),
            ("ao", ao_map, f"{base_name}_ao.png"),
        ]

        for map_type, map_data, filename in maps_to_write:
            output_path = output_dir / filename

            # Create temporary file in same directory (atomic rename requirement)
            temp_fd, temp_path = tempfile.mkstemp(
                suffix=".png",
                dir=output_dir,
                prefix=f".tmp_{base_name}_"
            )
            temp_files.append(temp_path)

            try:
                # Convert numpy array to PIL Image
                if map_data.ndim == 2:
                    # Grayscale
                    pil_image = Image.fromarray(map_data, mode='L')
                elif map_data.ndim == 3 and map_data.shape[2] == 3:
                    # RGB
                    pil_image = Image.fromarray(map_data, mode='RGB')
                else:
                    raise ValueError(f"Invalid map shape for {map_type}: {map_data.shape}")

                # Write to temp file (convert file descriptor to file object)
                with os.fdopen(temp_fd, 'wb') as f:
                    pil_image.save(f, format='PNG', optimize=True)

                # Atomic rename
                Path(temp_path).rename(output_path)
                temp_files.remove(temp_path)  # Successfully written

                written_paths[map_type] = output_path
                logger.info(f"Wrote {map_type} map: {output_path}")

            except Exception as e:
                logger.error(f"Failed to write {map_type} map: {e}")
                raise IOError(f"Failed to write {map_type} map to {output_path}") from e

        return written_paths

    except Exception as e:
        # Cleanup any remaining temp files
        for temp_path in temp_files:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {cleanup_error}")

        raise IOError(f"PBR map writing failed: {e}") from e
