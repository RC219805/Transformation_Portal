"""Standalone PBR processor - decoupled from orchestrator.

This module provides a clean, standalone API for PBR map generation
that doesn't require the full EnhanceOrchestrator pipeline.

Usage
-----
From cached depth::

    from transformation_portal.lux_depth_v3 import PBRProcessor, get_preset

    config = get_preset("premium").to_pbr_config()
    paths = PBRProcessor.from_cached_depth(
        depth_path=Path("output/scene1_depth.npy"),
        config=config,
        output_dir=Path("output/pbr/"),
        base_name="scene1"
    )

From depth array::

    processor = PBRProcessor(config=config, output_dir=Path("output/pbr/"))
    maps = processor.from_depth(depth_array, save=True, base_name="scene1")
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .pbr import PBRConfig, generate_pbr_maps
from .pbr_writer import write_pbr_maps

logger = logging.getLogger(__name__)


@dataclass
class PBRProcessor:
    """Standalone PBR map processor.

    Decouples PBR generation from orchestrator for:
    - PBR-only workflows (from cached depth)
    - Alternative depth sources
    - Custom output handling
    - Easier testing

    Attributes:
        config: PBR generation configuration
        output_dir: Optional output directory for automatic saving
    """

    config: PBRConfig
    output_dir: Optional[Path] = None

    def from_depth(self, depth: np.ndarray, save: bool = True, base_name: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Generate PBR maps from depth array.

        Args:
            depth: 2D depth array (H, W), normalized 0-1
            save: If True, write maps to output_dir
            base_name: Base filename for outputs (required if save=True)

        Returns:
            Dictionary with keys: "normal", "roughness", "ao"
            Values are numpy arrays (uint8)

        Raises:
            ValueError: If save=True but base_name or output_dir not provided
        """
        # Generate maps
        normal, roughness, ao = generate_pbr_maps(depth, self.config)

        maps = {
            "normal": normal,
            "roughness": roughness,
            "ao": ao,
        }

        # Optionally save to disk
        if save:
            if not self.output_dir:
                raise ValueError("output_dir required when save=True")
            if not base_name:
                raise ValueError("base_name required when save=True")

            self.output_dir.mkdir(parents=True, exist_ok=True)
            write_pbr_maps(normal, roughness, ao, self.output_dir, base_name)
            logger.info(f"Saved PBR maps to {self.output_dir / base_name}_*.png")

        return maps

    @classmethod
    def from_cached_depth(cls, depth_path: Path, config: PBRConfig, output_dir: Path, base_name: str) -> Dict[str, Path]:
        """Generate PBR from cached depth file (PNG or NPY).

        Standalone entry point for PBR-only workflows without
        running full depth estimation pipeline.

        Args:
            depth_path: Path to depth file (.npy preferred, .png supported)
            config: PBR generation configuration
            output_dir: Directory for output PBR maps
            base_name: Base filename for outputs

        Returns:
            Dictionary mapping map type to output path:
                {"normal": Path, "roughness": Path, "ao": Path}

        Raises:
            FileNotFoundError: If depth_path doesn't exist
            ValueError: If depth file is invalid

        Example:
            >>> from transformation_portal.lux_depth_v3 import get_preset
            >>> config = get_preset("premium").to_pbr_config()
            >>> paths = PBRProcessor.from_cached_depth(
            ...     depth_path=Path("output/scene1_depth.npy"),
            ...     config=config,
            ...     output_dir=Path("output/pbr/"),
            ...     base_name="scene1"
            ... )
            >>> print(paths["normal"])
            output/pbr/scene1_normal.png
        """
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth file not found: {depth_path}")

        # Load depth - prefer .npy (float precision) over .png (16-bit quantized)
        npy_path = depth_path.with_suffix(".npy")
        if npy_path.exists():
            logger.info(f"Loading float depth from: {npy_path}")
            depth = np.load(str(npy_path))
        else:
            logger.info(f"Loading quantized depth from: {depth_path}")
            from .depth_writer import read_depth_u16_png

            depth_raw = read_depth_u16_png(depth_path)

            # Robust normalization - check dtype
            if depth_raw.dtype == np.uint16:
                depth = depth_raw.astype(np.float32) / 65535.0
            else:
                # Already normalized or needs different handling
                depth = depth_raw.astype(np.float32, copy=False)
                maxv = float(np.nanmax(depth)) if depth.size else 0.0
                if maxv > 1.5:  # Likely unnormalized uint16 in float array
                    depth /= 65535.0

        # Validate depth
        if depth.ndim != 2:
            raise ValueError(f"Expected 2D depth array, got shape {depth.shape}")
        if np.any(np.isnan(depth)) or np.any(np.isinf(depth)):
            raise ValueError("Depth contains NaN or Inf values")

        # Generate PBR maps
        processor = cls(config=config, output_dir=output_dir)
        maps = processor.from_depth(depth, save=True, base_name=base_name)

        # Return paths
        return {
            "normal": output_dir / f"{base_name}_normal.png",
            "roughness": output_dir / f"{base_name}_roughness.png",
            "ao": output_dir / f"{base_name}_ao.png",
        }

    def __enter__(self) -> "PBRProcessor":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - cleanup resources."""
        # Future: GPU memory cleanup if GPU acceleration added
        pass
