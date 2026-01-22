"""Export module for depth estimation results.

Supports multiple output formats: PNG, NPZ, PLY, GLB, TIFF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from lux_depth_v3.config import ExportConfig, ExportFormat
from lux_depth_v3.inference import DepthResult

# NOTE: We intentionally do NOT import lux_depth_v3.postprocessing here.
# That module may include optional dependencies (e.g. edge refinement), and export
# should remain usable even when those extras aren't installed.


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
            try:
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
            except Exception as e:
                # Do not fail the whole export operation if one format is unsupported.
                print(f"Failed to export {fmt.value}: {e}")
                continue

        return exported

    def _extract_intrinsics(self, metadata: dict) -> Optional[Tuple[float, float, float, float]]:
        """Best-effort extraction of (fx, fy, cx, cy) from a metadata dict."""
        K = metadata.get("intrinsics")
        if K is None:
            return None

        try:
            arr = np.asarray(K)
        except Exception:
            return None

        if arr.shape == (4,):
            fx, fy, cx, cy = arr.tolist()
            return float(fx), float(fy), float(cx), float(cy)

        if arr.ndim == 2 and arr.shape == (3, 3):
            fx = float(arr[0, 0])
            fy = float(arr[1, 1])
            cx = float(arr[0, 2])
            cy = float(arr[1, 2])
            return fx, fy, cx, cy

        if arr.ndim == 3 and arr.shape[1:] == (3, 3):
            fx = float(arr[0, 0, 0])
            fy = float(arr[0, 1, 1])
            cx = float(arr[0, 0, 2])
            cy = float(arr[0, 1, 2])
            return fx, fy, cx, cy

        return None

    def _depth_to_point_cloud(
        self,
        depth: np.ndarray,
        image: np.ndarray,
        intrinsics: Optional[Tuple[float, float, float, float]] = None,
    ) -> np.ndarray:
        """Convert depth map to a simple XYZRGB point cloud.

        Returns an (N, 6) array with float XYZ and uint8 RGB.
        """

        if depth.ndim != 2:
            raise ValueError(f"Depth must be 2D for point cloud export, got shape {depth.shape}")

        h, w = depth.shape

        # Default intrinsics (assume ~60° horizontal FOV)
        if intrinsics is None:
            fx = fy = w / (2 * np.tan(np.radians(30)))
            cx, cy = w / 2.0, h / 2.0
        else:
            fx, fy, cx, cy = intrinsics

        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

        z = depth.astype(np.float32, copy=False)
        x = (u - float(cx)) * z / float(fx)
        y = (v - float(cy)) * z / float(fy)

        points_xyz = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

        # Colors
        if image.ndim == 3:
            rgb = image.reshape(-1, 3)
        else:
            rgb = np.stack([image.reshape(-1)] * 3, axis=1)

        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        return np.concatenate([points_xyz, rgb.astype(np.float32)], axis=1)

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
            intrinsics = self._extract_intrinsics(result.metadata)
            point_cloud = self._depth_to_point_cloud(result.depth_map, result.original_image, intrinsics=intrinsics)
        else:
            point_cloud = result.point_cloud

        # Downsample if needed (ceil so we actually respect the max)
        if len(point_cloud) > self.config.point_cloud_max_points:
            import math

            step = int(math.ceil(len(point_cloud) / float(self.config.point_cloud_max_points)))
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

        # GLB export is implemented via trimesh (optional dependency).
        try:
            import trimesh
        except ImportError as e:
            raise RuntimeError("GLB export requires `trimesh`. Install with: pip install trimesh") from e

        depth = result.depth_map
        if depth.ndim != 2:
            raise ValueError(f"GLB export expects a 2D depth map, got {depth.shape}")

        image = result.original_image
        if image is None:
            raise ValueError("GLB export requires original_image for vertex colors")

        # Downsample grid to keep the mesh size reasonable.
        h, w = depth.shape
        max_vertices = int(getattr(self.config, "point_cloud_max_points", 250_000))
        max_vertices = max(10_000, max_vertices)
        import math

        step = int(math.ceil(math.sqrt((h * w) / float(max_vertices)))) if (h * w) > max_vertices else 1
        depth_ds = depth[::step, ::step].astype(np.float32, copy=False)
        img_ds = image[::step, ::step]

        hs, ws = depth_ds.shape

        intrinsics = self._extract_intrinsics(result.metadata)

        u, v = np.meshgrid(np.arange(ws, dtype=np.float32), np.arange(hs, dtype=np.float32))
        z = depth_ds

        if intrinsics is None:
            # Pixel grid is already in downsampled resolution.
            fx = fy = ws / (2 * np.tan(np.radians(30)))
            cx, cy = ws / 2.0, hs / 2.0
            x = (u - float(cx)) * z / float(fx)
            y = (v - float(cy)) * z / float(fy)
        else:
            # Intrinsics are in the original pixel coordinate system.
            fx, fy, cx, cy = intrinsics
            u0 = u * float(step)
            v0 = v * float(step)
            x = (u0 - float(cx)) * z / float(fx)
            y = (v0 - float(cy)) * z / float(fy)
        vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        # Colors (RGBA)
        if img_ds.ndim == 2:
            rgb = np.stack([img_ds] * 3, axis=-1)
        else:
            rgb = img_ds[..., :3]

        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        colors = rgb.reshape(-1, 3)
        alpha = np.full((colors.shape[0], 1), 255, dtype=np.uint8)
        colors_rgba = np.concatenate([colors, alpha], axis=1)

        # Faces for a regular grid
        faces = []
        for r in range(hs - 1):
            base = r * ws
            next_base = (r + 1) * ws
            for c in range(ws - 1):
                i0 = base + c
                i1 = base + c + 1
                i2 = next_base + c
                i3 = next_base + c + 1
                faces.append([i0, i1, i2])
                faces.append([i1, i3, i2])

        mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False)
        mesh.visual.vertex_colors = colors_rgba

        mesh.export(output_path)
        return output_path

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


def export(result: DepthResult, filename_base: str, config: ExportConfig) -> Dict[str, Path]:
    """Convenience wrapper around `Exporter`."""
    return Exporter(config).export(result, filename_base)
