"""PLY export for 3D Gaussian Splatting reconstructions.

Exports Scene3D as PLY (Polygon File Format / Stanford Triangle Format).
PLY is the natural export format for Gaussian Splatting since it's point-cloud
based, unlike OBJ which is mesh-first.

Export includes:
- Gaussian positions (x, y, z)
- RGB colors (normalized to 0-255)
- Optional scales, rotations, opacities as vertex properties
- Binary encoding for efficiency (default) or ASCII for debugging

Reference:
- PLY format: https://paulbourke.net/dataformats/ply/
- 3DGS PLY extension: includes scale/rotation/opacity properties
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from transformation_portal.ingest.canonical_json import dump_json

logger = logging.getLogger(__name__)


@dataclass
class PLYExportConfig:
    """Configuration for PLY export.

    Attributes:
        binary: Use binary encoding (more compact) vs ASCII.
        include_attributes: Include scales, rotations, opacities.
        include_sh: Include spherical harmonics (future extension).
        color_scale: Scale for RGB output. Default is 255 (uint8 0-255).
            Colors are assumed to be in [0, 1] float range and are
            multiplied by this value then clamped to [0, 255] for output.
    """

    binary: bool = True
    include_attributes: bool = True
    include_sh: bool = False
    color_scale: float = 255.0


class PLYExporter:
    """Export 3D Gaussian Splatting scenes to PLY format.

    Usage:
        >>> exporter = PLYExporter()
        >>> output_path = exporter.export(scene, output_dir / "scene.ply")
        >>> print(f"Exported {scene.splats.num_gaussians} Gaussians to {output_path}")
    """

    def __init__(self, config: Optional[PLYExportConfig] = None):
        """Initialize PLY exporter.

        Args:
            config: Export configuration. Defaults to binary with attributes.
        """
        self.config = config or PLYExportConfig()

    def export(
        self,
        scene: Any,  # Scene3D - avoid import cycle
        output_path: Path,
        write_sidecar: bool = True,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Export Scene3D to PLY file.

        Args:
            scene: Scene3D object with Gaussian splats.
            output_path: Output PLY file path.
            write_sidecar: Write provenance JSON sidecar next to PLY.
            additional_metadata: Extra metadata to include in sidecar.

        Returns:
            Path to the written PLY file.

        Raises:
            ValueError: If scene data is invalid.
            IOError: If file write fails.
        """
        splats = scene.splats
        num_points = splats.num_gaussians

        logger.info(f"Exporting {num_points} Gaussians to {output_path}")

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.config.binary:
            self._write_binary(output_path, splats)
        else:
            self._write_ascii(output_path, splats)

        # Write provenance sidecar
        if write_sidecar:
            sidecar_path = output_path.with_suffix(".provenance.json")
            self._write_sidecar(
                sidecar_path,
                output_path,
                scene,
                additional_metadata,
            )

        logger.info(f"PLY export complete: {output_path}")
        return output_path

    def _build_header(self, num_points: int, binary: bool, include_attrs: bool) -> str:
        """Build PLY header."""
        lines = [
            "ply",
            f"format {'binary_little_endian' if binary else 'ascii'} 1.0",
            "comment Transformation Portal Gaussian Splatting Export",
            f"comment Generated: {datetime.now(timezone.utc).isoformat()}",
            f"element vertex {num_points}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]

        if include_attrs:
            # Scales (3D)
            lines.extend(
                [
                    "property float scale_x",
                    "property float scale_y",
                    "property float scale_z",
                ]
            )
            # Rotations (quaternion)
            lines.extend(
                [
                    "property float rot_w",
                    "property float rot_x",
                    "property float rot_y",
                    "property float rot_z",
                ]
            )
            # Opacity
            lines.append("property float opacity")

        lines.append("end_header")
        return "\n".join(lines) + "\n"

    def _get_effective_color_scale(self) -> float:
        """Get effective color scale value, defaulting to 255.0 if invalid."""
        scale = self.config.color_scale
        if scale <= 0:
            return 255.0
        return scale

    def _write_binary(self, output_path: Path, splats: Any) -> None:
        """Write binary PLY file.

        Uses vectorized NumPy operations for efficient export of large
        point clouds (tens/hundreds of thousands of Gaussians).
        """
        num_points = splats.num_gaussians
        include_attrs = self.config.include_attributes

        header = self._build_header(num_points, binary=True, include_attrs=include_attrs)

        # Use configurable color scale; clamp to 255 for uint8 output
        scale = self._get_effective_color_scale()

        with open(output_path, "wb") as f:
            # Write header as ASCII
            f.write(header.encode("ascii"))

            # Vectorized write of vertex data in binary.
            # Build a structured NumPy array matching the exact field order
            # and types, then write it in one shot.

            # Positions: ensure float32 little-endian, shape (N, 3)
            positions = np.asarray(splats.positions, dtype="<f4")
            if positions.shape[0] != num_points or positions.shape[1] != 3:
                raise ValueError(f"Expected positions shape ({num_points}, 3), got {positions.shape}")

            # Colors: input is assumed float in [0, 1], scale and cast to uint8
            colors = np.asarray(splats.colors, dtype=np.float32)
            if colors.shape[0] != num_points or colors.shape[1] != 3:
                raise ValueError(f"Expected colors shape ({num_points}, 3), got {colors.shape}")
            colors_uint8 = np.clip(colors * scale, 0.0, 255.0).astype(np.uint8)

            if include_attrs:
                # Scales: float32, shape (N, 3)
                scales = np.asarray(splats.scales, dtype="<f4")
                if scales.shape[0] != num_points or scales.shape[1] != 3:
                    raise ValueError(f"Expected scales shape ({num_points}, 3), got {scales.shape}")

                # Rotations: float32 quaternion, shape (N, 4)
                rotations = np.asarray(splats.rotations, dtype="<f4")
                if rotations.shape[0] != num_points or rotations.shape[1] != 4:
                    raise ValueError(f"Expected rotations shape ({num_points}, 4), got {rotations.shape}")

                # Opacities: take first component to mirror opacities[i, 0]
                opacities_arr = np.asarray(splats.opacities, dtype="<f4")
                if opacities_arr.shape[0] != num_points:
                    raise ValueError(f"Expected opacities first dimension {num_points}, got {opacities_arr.shape}")
                # Support shapes (N, 1) or (N,)
                if opacities_arr.ndim == 2:
                    if opacities_arr.shape[1] < 1:
                        raise ValueError(f"Expected opacities second dimension >= 1, got {opacities_arr.shape}")
                    opacities = opacities_arr[:, 0]
                else:
                    opacities = opacities_arr

                vertex_dtype = np.dtype(
                    [
                        ("x", "<f4"),
                        ("y", "<f4"),
                        ("z", "<f4"),
                        ("red", "u1"),
                        ("green", "u1"),
                        ("blue", "u1"),
                        ("scale_0", "<f4"),
                        ("scale_1", "<f4"),
                        ("scale_2", "<f4"),
                        ("rot_0", "<f4"),
                        ("rot_1", "<f4"),
                        ("rot_2", "<f4"),
                        ("rot_3", "<f4"),
                        ("opacity", "<f4"),
                    ]
                )

                vertices = np.empty(num_points, dtype=vertex_dtype)
                vertices["x"] = positions[:, 0]
                vertices["y"] = positions[:, 1]
                vertices["z"] = positions[:, 2]
                vertices["red"] = colors_uint8[:, 0]
                vertices["green"] = colors_uint8[:, 1]
                vertices["blue"] = colors_uint8[:, 2]
                vertices["scale_0"] = scales[:, 0]
                vertices["scale_1"] = scales[:, 1]
                vertices["scale_2"] = scales[:, 2]
                vertices["rot_0"] = rotations[:, 0]
                vertices["rot_1"] = rotations[:, 1]
                vertices["rot_2"] = rotations[:, 2]
                vertices["rot_3"] = rotations[:, 3]
                vertices["opacity"] = opacities
            else:
                vertex_dtype = np.dtype(
                    [
                        ("x", "<f4"),
                        ("y", "<f4"),
                        ("z", "<f4"),
                        ("red", "u1"),
                        ("green", "u1"),
                        ("blue", "u1"),
                    ]
                )

                vertices = np.empty(num_points, dtype=vertex_dtype)
                vertices["x"] = positions[:, 0]
                vertices["y"] = positions[:, 1]
                vertices["z"] = positions[:, 2]
                vertices["red"] = colors_uint8[:, 0]
                vertices["green"] = colors_uint8[:, 1]
                vertices["blue"] = colors_uint8[:, 2]

            # Write all vertex records in one contiguous block.
            f.write(vertices.tobytes(order="C"))

    def _write_ascii(self, output_path: Path, splats: Any) -> None:
        """Write ASCII PLY file (for debugging)."""
        num_points = splats.num_gaussians
        include_attrs = self.config.include_attributes

        header = self._build_header(num_points, binary=False, include_attrs=include_attrs)

        # Use configurable color scale; clamp to 255 for uint8 output
        scale = self._get_effective_color_scale()

        with open(output_path, "w", encoding="ascii") as f:
            f.write(header)

            for i in range(num_points):
                pos = splats.positions[i]
                color = splats.colors[i]
                r = int(np.clip(color[0] * scale, 0, 255))
                g = int(np.clip(color[1] * scale, 0, 255))
                b = int(np.clip(color[2] * scale, 0, 255))

                line_parts = [f"{pos[0]:.6f}", f"{pos[1]:.6f}", f"{pos[2]:.6f}"]
                line_parts.extend([str(r), str(g), str(b)])

                if include_attrs:
                    splat_scales = splats.scales[i]
                    rot = splats.rotations[i]
                    opacity = splats.opacities[i, 0]
                    line_parts.extend(
                        [
                            f"{splat_scales[0]:.6f}",
                            f"{splat_scales[1]:.6f}",
                            f"{splat_scales[2]:.6f}",
                            f"{rot[0]:.6f}",
                            f"{rot[1]:.6f}",
                            f"{rot[2]:.6f}",
                            f"{rot[3]:.6f}",
                            f"{opacity:.6f}",
                        ]
                    )

                f.write(" ".join(line_parts) + "\n")

    def _write_sidecar(
        self,
        sidecar_path: Path,
        ply_path: Path,
        scene: Any,
        additional_metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Write provenance sidecar JSON.

        Contains:
        - Backend info
        - Tier and device
        - Optimization parameters
        - View count and camera summary
        - Output file hash
        """
        # Compute file hash
        file_hash = self._compute_file_hash(ply_path)

        provenance: Dict[str, Any] = {
            "schema_version": "1.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "output_file": str(ply_path.name),
            "output_hash_sha256": file_hash,
            "backend": scene.metadata.get("backend", "unknown"),
            "tier": scene.metadata.get("tier", "unknown"),
            "device": scene.metadata.get("device", "unknown"),
            "num_views": scene.metadata.get("num_views", len(scene.cameras)),
            "num_gaussians": scene.splats.num_gaussians,
            "requested_iterations": scene.metadata.get("requested_iterations", -1),
            "actual_iterations": scene.iteration,
            "convergence": scene.convergence,
            "rmse": scene.rmse,
            "quality_score": scene.quality_score,
            "optimization_seed": scene.metadata.get("optimization_seed"),
            "use_depth_prior": scene.metadata.get("use_depth_prior", False),
            "use_segmentation": scene.metadata.get("use_segmentation", False),
            "use_pbr_textures": scene.metadata.get("use_pbr_textures", False),
            "elapsed_seconds": scene.metadata.get("elapsed_seconds", -1),
            "export_config": {
                "binary": self.config.binary,
                "include_attributes": self.config.include_attributes,
                "include_sh": self.config.include_sh,
            },
        }

        if additional_metadata:
            provenance["request_metadata"] = additional_metadata

        with open(sidecar_path, "w", encoding="utf-8") as f:
            dump_json(provenance, f, indent=2, ensure_ascii=False)

        logger.debug(f"Provenance sidecar written: {sidecar_path}")

    def _compute_file_hash(self, path: Path) -> str:
        """Compute SHA-256 hash of file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()


def export_scene_to_ply(
    scene: Any,
    output_path: Path,
    binary: bool = True,
    include_attributes: bool = True,
    write_sidecar: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Convenience function to export Scene3D to PLY.

    Args:
        scene: Scene3D object.
        output_path: Output PLY file path.
        binary: Use binary encoding.
        include_attributes: Include scale/rotation/opacity.
        write_sidecar: Write provenance JSON.
        additional_metadata: Extra metadata for sidecar.

    Returns:
        Path to written PLY file.
    """
    config = PLYExportConfig(
        binary=binary,
        include_attributes=include_attributes,
    )
    exporter = PLYExporter(config)
    return exporter.export(
        scene,
        output_path,
        write_sidecar=write_sidecar,
        additional_metadata=additional_metadata,
    )
