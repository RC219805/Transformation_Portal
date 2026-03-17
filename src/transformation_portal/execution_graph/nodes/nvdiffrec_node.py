"""NVDIFFREC material reconstruction DAG node.

This node wraps the NVDIFFREC backend for material/PBR reconstruction
in DAG-based pipeline orchestration.

Note: The actual NVDIFFREC backend integration is a phase-gated TODO
(documented in TODO_INVENTORY.md §2.0.4-5). This node provides the
DAG interface for when the backend becomes available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from transformation_portal.execution_graph.nodes.base import DAGNode, NodeResult

logger = logging.getLogger(__name__)


@dataclass
class NVDiffRecOutputs:
    """Outputs from NVDIFFREC reconstruction.

    Attributes:
        albedo_map: Path to reconstructed albedo/diffuse texture
        normal_map: Path to reconstructed normal map
        roughness_map: Path to reconstructed roughness map (optional)
        metallic_map: Path to reconstructed metallic map (optional)
        mesh_path: Path to output mesh (optional)
        diagnostics_json: Path to diagnostics JSON (optional)
    """

    albedo_map: Optional[Path] = None
    normal_map: Optional[Path] = None
    roughness_map: Optional[Path] = None
    metallic_map: Optional[Path] = None
    mesh_path: Optional[Path] = None
    diagnostics_json: Optional[Path] = None


class NVDiffRecNode(DAGNode):
    """DAG node for NVDIFFREC material reconstruction.

    Wraps the NVDIFFREC backend for use in execution graphs.
    Currently returns stub outputs as the full backend integration
    is a phase-gated TODO.

    Example:
        >>> node = NVDiffRecNode()
        >>> result = node.run(
        ...     image_paths=[Path("view1.png"), Path("view2.png")],
        ...     output_dir=Path("out/materials"),
        ... )
        >>> if result.success:
        ...     print(f"Albedo: {result.outputs['albedo']}")
    """

    def __init__(
        self,
        backend: Optional[Any] = None,
    ) -> None:
        """Initialize node with optional backend.

        Args:
            backend: Optional NVDiffRecBackend instance.
                     If None, uses stub implementation.
        """
        self.backend = backend

    def validate_inputs(
        self,
        **inputs: Any,
    ) -> Optional[str]:
        """Validate node inputs.

        Required inputs:
            image_paths: List of input image paths
            output_dir: Directory for output files

        Optional inputs:
            geometry_input: Optional geometry file for guided reconstruction
        """
        image_paths = inputs.get("image_paths")
        if not image_paths:
            return "image_paths is required and must be non-empty"

        if not isinstance(image_paths, list):
            return "image_paths must be a list"

        output_dir = inputs.get("output_dir")
        if not output_dir:
            return "output_dir is required"

        return None

    def run(
        self,
        *,
        image_paths: list[Path],
        output_dir: Path,
        geometry_input: Optional[Path] = None,
    ) -> NodeResult:
        """Execute material reconstruction.

        Args:
            image_paths: List of input images for multi-view reconstruction
            output_dir: Directory for output files
            geometry_input: Optional geometry file for guided reconstruction

        Returns:
            NodeResult with material reconstruction outputs
        """
        # Validate inputs
        error = self.validate_inputs(
            image_paths=image_paths,
            output_dir=output_dir,
            geometry_input=geometry_input,
        )
        if error:
            return NodeResult(error=error)

        # Use real backend if available
        if self.backend is not None:
            return self._run_with_backend(
                image_paths=image_paths,
                output_dir=output_dir,
                geometry_input=geometry_input,
            )

        # Stub implementation (backend not yet integrated)
        logger.warning(
            "NVDiffRecNode: backend not available, returning stub outputs. "
            "Full integration is phase-gated (TODO_INVENTORY.md §2.0.4-5)"
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        # Return stub outputs indicating backend is not available
        return NodeResult(
            outputs={
                "albedo": None,
                "normal": None,
                "roughness": None,
                "metallic": None,
                "mesh": None,
                "diagnostics": None,
                "backend_available": False,
            },
            metadata={
                "num_input_images": len(image_paths),
                "output_dir": str(output_dir),
                "stub_mode": True,
            },
        )

    def _run_with_backend(
        self,
        *,
        image_paths: list[Path],
        output_dir: Path,
        geometry_input: Optional[Path],
    ) -> NodeResult:
        """Run reconstruction with actual backend.

        This method is called when a real backend is provided.
        """
        try:
            outputs = self.backend.reconstruct(
                image_paths=image_paths,
                geometry_input=geometry_input,
                output_dir=output_dir,
            )

            return NodeResult(
                outputs={
                    "albedo": outputs.albedo_map,
                    "normal": outputs.normal_map,
                    "roughness": outputs.roughness_map,
                    "metallic": outputs.metallic_map,
                    "mesh": outputs.mesh_path,
                    "diagnostics": outputs.diagnostics_json,
                    "backend_available": True,
                },
                metadata={
                    "num_input_images": len(image_paths),
                    "output_dir": str(output_dir),
                    "stub_mode": False,
                },
            )

        except Exception as exc:
            logger.exception("NVDIFFREC reconstruction failed")
            return NodeResult(error=str(exc))
