"""Export module for depth estimation results.

Supports multiple output formats: PNG, NPZ, PLY, GLB, TIFF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict

import numpy as np
from PIL import Image

from lux_depth_v3.config import ExportConfig, ExportFormat
from lux_depth_v3.inference import DepthResult
from lux_depth_v3.postprocessing import Postprocessor


class Exporter:
    """Exporter for depth estimation results."""

    def __init__(self, config: ExportConfig):
        """Initialize exporter.

        Args:
            config: Export configuration
        """
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        result: DepthResult,
        filename_base: str,
    ) -> Dict[str, Path]:
        """Export depth result in configured formats.

        Args:
            result: Depth estimation result
            filename_base: Base filename (without extension)

        Returns:
            Dictionary of format -> output_path
        """
        exported = {}

        for fmt in self.config.formats:
            if fmt == ExportFormat.PNG:
                path = self._export_png(result, filename_base)
            elif fmt == ExportFormat.NPZ:
                path = self._export_npz(result, filename_base)
            elif fmt == ExportFormat.PLY:
                path = self._export_ply(result, filename_base)
            elif fmt == ExportFormat.GLB:
                path = self._export_glb(result, filename_base)
            elif fmt == ExportFormat.TIFF:
                path = self._export_tiff(result, filename_base)
            else:
                print(f"Unknown export format: {fmt}")
                continue

            exported[fmt.value] = path

        return exported

    def _export_png(
        self,
        result: DepthResult,
        filename_base: str,
    ) -> Path:
        """Export depth as 16-bit PNG.

        Args:
            result: Depth result
            filename_base: Base filename

        Returns:
            Output path
        """
        output_path = self.config.output_dir / f"{filename_base}_depth.png"

        # Convert to uint16
        depth_uint16 = result.to_uint16(scale=self.config.depth_scale)

        # Save as 16-bit grayscale PNG
        img = Image.fromarray(depth_uint16, mode="I;16")
        img.save(output_path)

        return output_path

    def _export_npz(
        self,
        result: DepthResult,
        filename_base: str,
    ) -> Path:
        """Export depth as NPZ (NumPy compressed).

        Args:
            result: Depth result
            filename_base: Base filename

        Returns:
            Output path
        """
        output_path = self.config.output_dir / f"{filename_base}_depth.npz"

        # Save depth and metadata
        np.savez_compressed(
            output_path,
            depth=result.depth_map,
            metadata=result.metadata,
        )

        return output_path

    def _export_ply(
        self,
        result: DepthResult,
        filename_base: str,
    ) -> Path:
        """Export as PLY point cloud.

        Args:
            result: Depth result
            filename_base: Base filename

        Returns:
            Output path
        """
        output_path = self.config.output_dir / f"{filename_base}_pointcloud.ply"

        # Generate point cloud if not already present
        if result.point_cloud is None:
            postprocessor = Postprocessor(None)
            point_cloud = postprocessor.to_point_cloud(
                result.depth_map,
                result.original_image,
            )
        else:
            point_cloud = result.point_cloud

        # Downsample if needed
        if len(point_cloud) > self.config.point_cloud_max_points:
            step = len(point_cloud) // self.config.point_cloud_max_points
            point_cloud = point_cloud[::step]

        # Write PLY file
        self._write_ply(output_path, point_cloud)

        return output_path

    def _write_ply(
        self,
        output_path: Path,
        point_cloud: np.ndarray,
    ):
        """Write point cloud to PLY file.

        Args:
            output_path: Output file path
            point_cloud: Point cloud (N, 6) with XYZ and RGB
        """
        with open(output_path, "w") as f:
            # PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(point_cloud)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            # Write vertices
            for point in point_cloud:
                x, y, z, r, g, b = point
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

    def _export_glb(
        self,
        result: DepthResult,
        filename_base: str,
    ) -> Path:
        """Export as GLB (GLTF binary).

        Args:
            result: Depth result
            filename_base: Base filename

        Returns:
            Output path
        """
        output_path = self.config.output_dir / f"{filename_base}_mesh.glb"

        # Note: This requires trimesh or similar library
        # For now, we'll create a placeholder
        print("GLB export not yet implemented, saving PLY instead")
        return self._export_ply(result, filename_base)

    def _export_tiff(
        self,
        result: DepthResult,
        filename_base: str,
    ) -> Path:
        """Export depth as 32-bit float TIFF.

        Args:
            result: Depth result
            filename_base: Base filename

        Returns:
            Output path
        """
        output_path = self.config.output_dir / f"{filename_base}_depth.tiff"

        try:
            import tifffile

            # Save as float32 TIFF
            tifffile.imwrite(
                output_path,
                result.depth_map.astype(np.float32),
                photometric="minisblack",
            )

        except ImportError:
            # Fallback to PIL
            # Convert to uint16 for PIL
            depth_uint16 = result.to_uint16(scale=self.config.depth_scale)
            img = Image.fromarray(depth_uint16)
            img.save(output_path)

        return output_path
