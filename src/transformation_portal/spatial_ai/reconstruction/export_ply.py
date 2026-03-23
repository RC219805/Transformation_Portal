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
import json
import logging
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PLYExportConfig:
    """Configuration for PLY export.

    Attributes:
        binary: Use binary encoding (more compact) vs ASCII.
        include_attributes: Include scales, rotations, opacities.
        include_sh: Include spherical harmonics (future extension).
        color_scale: Scale for RGB output (255 for uint8, 1.0 for float).
    """

    binary: bool = True
    include_attributes: bool = True
    include_sh: bool = False
    color_scale: int = 255


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
            f"comment Transformation Portal Gaussian Splatting Export",
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
            lines.extend([
                "property float scale_x",
                "property float scale_y",
                "property float scale_z",
            ])
            # Rotations (quaternion)
            lines.extend([
                "property float rot_w",
                "property float rot_x",
                "property float rot_y",
                "property float rot_z",
            ])
            # Opacity
            lines.append("property float opacity")

        lines.append("end_header")
        return "\n".join(lines) + "\n"

    def _write_binary(self, output_path: Path, splats: Any) -> None:
        """Write binary PLY file."""
        num_points = splats.num_gaussians
        include_attrs = self.config.include_attributes

        header = self._build_header(num_points, binary=True, include_attrs=include_attrs)

        with open(output_path, "wb") as f:
            # Write header as ASCII
            f.write(header.encode("ascii"))

            # Write vertex data in binary
            for i in range(num_points):
                # Position (float32)
                pos = splats.positions[i]
                f.write(struct.pack("<fff", pos[0], pos[1], pos[2]))

                # Color (uint8)
                color = splats.colors[i]
                r = int(np.clip(color[0] * 255, 0, 255))
                g = int(np.clip(color[1] * 255, 0, 255))
                b = int(np.clip(color[2] * 255, 0, 255))
                f.write(struct.pack("<BBB", r, g, b))

                if include_attrs:
                    # Scales (float32)
                    scale = splats.scales[i]
                    f.write(struct.pack("<fff", scale[0], scale[1], scale[2]))

                    # Rotations (float32 quaternion)
                    rot = splats.rotations[i]
                    f.write(struct.pack("<ffff", rot[0], rot[1], rot[2], rot[3]))

                    # Opacity (float32)
                    opacity = splats.opacities[i, 0]
                    f.write(struct.pack("<f", opacity))

    def _write_ascii(self, output_path: Path, splats: Any) -> None:
        """Write ASCII PLY file (for debugging)."""
        num_points = splats.num_gaussians
        include_attrs = self.config.include_attributes

        header = self._build_header(num_points, binary=False, include_attrs=include_attrs)

        with open(output_path, "w", encoding="ascii") as f:
            f.write(header)

            for i in range(num_points):
                pos = splats.positions[i]
                color = splats.colors[i]
                r = int(np.clip(color[0] * 255, 0, 255))
                g = int(np.clip(color[1] * 255, 0, 255))
                b = int(np.clip(color[2] * 255, 0, 255))

                line_parts = [f"{pos[0]:.6f}", f"{pos[1]:.6f}", f"{pos[2]:.6f}"]
                line_parts.extend([str(r), str(g), str(b)])

                if include_attrs:
                    scale = splats.scales[i]
                    rot = splats.rotations[i]
                    opacity = splats.opacities[i, 0]
                    line_parts.extend([
                        f"{scale[0]:.6f}", f"{scale[1]:.6f}", f"{scale[2]:.6f}",
                        f"{rot[0]:.6f}", f"{rot[1]:.6f}", f"{rot[2]:.6f}", f"{rot[3]:.6f}",
                        f"{opacity:.6f}",
                    ])

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
            json.dump(provenance, f, indent=2, ensure_ascii=False)

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
