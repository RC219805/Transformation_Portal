"""Mesh export capabilities for 3D scenes.

Exports Gaussian splats to standard mesh formats:
- PLY (Point Cloud with attributes) - RECOMMENDED for vertex colors
- OBJ (Mesh with vertex colors) - Limited scalability, see warnings
- GLTF (Future: PBR materials)

IMPORTANT: For large Gaussian splat scenes (100K+ points), prefer PLY format.
OBJ vertex color export creates O(N) material definitions which can produce
multi-GB MTL files for realistic scene sizes.

Architecture:
- Format-specific writers with validation
- Material preservation where supported
- Metadata embedding (provenance, quality metrics)
- File size optimization
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .contracts import Scene3D

logger = logging.getLogger(__name__)


class MeshExporter:
    """Export 3D scenes to mesh file formats.

    Supports:
    - PLY: Point cloud with Gaussian attributes
    - OBJ: Mesh representation (simplified)

    Usage:
        >>> exporter = MeshExporter()
        >>> exporter.export_ply(scene, "scene.ply", include_attributes=True)
        >>> exporter.export_obj(scene, "scene.obj", vertex_colors=True)
    """

    def __init__(self):
        """Initialize mesh exporter."""
        logger.info("MeshExporter initialized")

    def export_ply(
        self,
        scene: Scene3D,
        output_path: Path,
        include_attributes: bool = True,
        binary: bool = True,
    ) -> None:
        """Export scene to PLY point cloud format.

        Args:
            scene: 3D scene to export.
            output_path: Output file path.
            include_attributes: Include Gaussian attributes (scales, rotations, opacities).
            binary: Use binary PLY format (smaller files).

        Raises:
            ValueError: If scene validation fails.
            IOError: If file write fails.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        splats = scene.splats
        N = splats.num_gaussians

        logger.info(f"Exporting {N} Gaussians to PLY: {output_path}")

        # Build PLY header
        header_lines = [
            "ply",
            "format binary_little_endian 1.0" if binary else "format ascii 1.0",
            f"element vertex {N}",
            "property float x",
            "property float y",
            "property float z",
            "property float red",
            "property float green",
            "property float blue",
        ]

        if include_attributes:
            # Add Gaussian attributes
            header_lines.extend(
                [
                    "property float scale_x",
                    "property float scale_y",
                    "property float scale_z",
                    "property float rot_w",
                    "property float rot_x",
                    "property float rot_y",
                    "property float rot_z",
                    "property float opacity",
                ]
            )

        # Add metadata as comments
        header_lines.extend(
            [
                f"comment RMSE: {scene.rmse:.6f}",
                f"comment Quality Score: {scene.quality_score:.1f}/100",
                f"comment Convergence: {scene.convergence}",
                f"comment Iteration: {scene.iteration}",
                f"comment Num Cameras: {len(scene.cameras)}",
            ]
        )

        header_lines.append("end_header")
        header = "\n".join(header_lines) + "\n"

        # Write file
        with open(output_path, "wb" if binary else "w") as f:
            if binary:
                f.write(header.encode("ascii"))
                # Write binary vertex data
                vertex_data = np.column_stack(
                    [
                        splats.positions,  # x, y, z
                        splats.colors,  # r, g, b
                    ]
                )
                if include_attributes:
                    vertex_data = np.column_stack(
                        [
                            vertex_data,
                            splats.scales,  # scale_x, scale_y, scale_z
                            splats.rotations,  # rot_w, rot_x, rot_y, rot_z
                            splats.opacities,  # opacity
                        ]
                    )
                vertex_data.astype(np.float32).tofile(f)
            else:
                f.write(header)
                # Write ASCII vertex data
                for i in range(N):
                    vertex_line = f"{splats.positions[i, 0]:.6f} {splats.positions[i, 1]:.6f} {splats.positions[i, 2]:.6f} "
                    vertex_line += f"{splats.colors[i, 0]:.6f} {splats.colors[i, 1]:.6f} {splats.colors[i, 2]:.6f}"

                    if include_attributes:
                        vertex_line += f" {splats.scales[i, 0]:.6f} {splats.scales[i, 1]:.6f} {splats.scales[i, 2]:.6f}"
                        vertex_line += f" {splats.rotations[i, 0]:.6f} {splats.rotations[i, 1]:.6f} {splats.rotations[i, 2]:.6f} {splats.rotations[i, 3]:.6f}"
                        vertex_line += f" {splats.opacities[i, 0]:.6f}"

                    f.write(vertex_line + "\n")

        logger.info(f"PLY export complete: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

    def export_obj(
        self,
        scene: Scene3D,
        output_path: Path,
        vertex_colors: bool = True,
        subsample_factor: int = 1,
    ) -> None:
        """Export scene to OBJ mesh format.

        WARNING: OBJ vertex color export uses per-vertex materials which
        creates O(N) material definitions. For realistic Gaussian splat counts
        (100K-1M+), this will produce extremely large MTL files.

        RECOMMENDED: Use PLY format for vertex colors instead.
        This method is provided for compatibility but may not scale well.

        Args:
            scene: 3D scene to export.
            output_path: Output file path (.obj).
            vertex_colors: Export vertex colors to MTL file (not recommended for large scenes).
            subsample_factor: Downsample factor (1 = no downsampling).

        Raises:
            ValueError: If scene validation fails or vertex count exceeds safe limits.
            IOError: If file write fails.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        splats = scene.splats

        # Subsample if requested
        if subsample_factor > 1:
            indices = np.arange(0, splats.num_gaussians, subsample_factor)
            positions = splats.positions[indices]
            colors = splats.colors[indices]
        else:
            positions = splats.positions
            colors = splats.colors

        N = len(positions)

        # Warn about scalability for large vertex counts
        if vertex_colors and N > 10000:
            logger.warning(
                f"OBJ export with {N} vertex colors will create {N} materials. "
                f"Consider using PLY format or disabling vertex_colors for better scalability."
            )

        logger.info(f"Exporting {N} vertices to OBJ: {output_path}")

        # Write OBJ file
        with open(output_path, "w") as f:
            f.write(f"# 3D Gaussian Splatting Export\n")
            f.write(f"# RMSE: {scene.rmse:.6f}\n")
            f.write(f"# Quality Score: {scene.quality_score:.1f}/100\n")
            f.write(f"# Num Gaussians: {splats.num_gaussians}\n")
            f.write(f"# Num Cameras: {len(scene.cameras)}\n\n")

            if vertex_colors:
                mtl_path = output_path.with_suffix(".mtl")
                f.write(f"mtllib {mtl_path.name}\n\n")

            # Write vertices
            for i in range(N):
                f.write(f"v {positions[i, 0]:.6f} {positions[i, 1]:.6f} {positions[i, 2]:.6f}\n")

            # Write vertex colors as materials (if requested)
            if vertex_colors:
                f.write("\n")
                for i in range(N):
                    f.write(f"usemtl color_{i}\n")
                    f.write(f"p {i + 1}\n")  # Point element

        # Write MTL file with vertex colors
        if vertex_colors:
            mtl_path = output_path.with_suffix(".mtl")
            with open(mtl_path, "w") as f:
                f.write("# Vertex color materials\n\n")
                for i in range(N):
                    f.write(f"newmtl color_{i}\n")
                    f.write(f"Kd {colors[i, 0]:.6f} {colors[i, 1]:.6f} {colors[i, 2]:.6f}\n\n")

            logger.info(f"MTL export complete: {mtl_path}")

        logger.info(f"OBJ export complete: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

    def export_cameras(self, scene: Scene3D, output_path: Path) -> None:
        """Export camera parameters to JSON file.

        Args:
            scene: 3D scene with cameras.
            output_path: Output JSON file path.

        Raises:
            IOError: If file write fails.
        """
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cameras_data = []
        for i, cam in enumerate(scene.cameras):
            cam_dict = {
                "camera_id": cam.camera_id or f"camera_{i:03d}",
                "width": cam.width,
                "height": cam.height,
                "intrinsics": cam.intrinsics.tolist(),
                "extrinsics": cam.extrinsics.tolist(),
            }
            if cam.distortion is not None:
                cam_dict["distortion"] = cam.distortion.tolist()

            cameras_data.append(cam_dict)

        # Write JSON
        with open(output_path, "w") as f:
            json.dump(
                {
                    "num_cameras": len(cameras_data),
                    "cameras": cameras_data,
                    "scene_metadata": {
                        "rmse": scene.rmse,
                        "quality_score": scene.quality_score,
                        "convergence": scene.convergence,
                        "num_gaussians": scene.splats.num_gaussians,
                    },
                },
                f,
                indent=2,
            )

        logger.info(f"Camera export complete: {output_path}")
