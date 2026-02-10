"""V2 Depth-Aware Enhancement Implementation.

Consumes depth maps from V3 stage and applies perceptual finishing
for luxury real estate marketing output.

Architecture:
- Reuses existing EnhancementStage for core enhancement logic
- Depth-aware tone mapping for spatial hierarchy
- Material-aware processing using Materials V3 taxonomy
- NO ML DEPENDENCIES (image processing only)

Performance Target: <2s/image typical, <5s max (400-600 images/hour)

Dependencies:
    - numpy, scipy (core)
    - PIL (image I/O)
    - EnhancementStage (reuse existing)

Reference: V2_ENHANCEMENT_ARCHITECTURAL_GUIDANCE.md
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from ..stage_graph.stage import StageContext, StageStatus
from ..stage_graph.stages.enhancement import EnhancementStage
from .v2_presets import V2EnhancementConfig

logger = logging.getLogger(__name__)


class V2EnhancementError(Exception):
    """Raised when V2 enhancement operation fails."""

    pass


def find_depth_map(depth_dir: Path, image_stem: str) -> Optional[Path]:
    """Find depth map for input image in depth directory.

    Searches for depth maps with common naming patterns:
    - {image_stem}_depth.png
    - {image_stem}_depth_u16.png
    - {image_stem}.png (in depth_dir)

    Args:
        depth_dir: Directory containing depth maps
        image_stem: Input image filename stem

    Returns:
        Path to depth map or None if not found
    """
    if not depth_dir or not depth_dir.exists():
        return None

    # Try common depth map naming patterns
    patterns = [
        f"{image_stem}_depth.png",
        f"{image_stem}_depth_u16.png",
        f"{image_stem}_depth_f32.png",
        f"{image_stem}.png",
    ]

    for pattern in patterns:
        depth_path = depth_dir / pattern
        if depth_path.exists():
            logger.debug(f"Found depth map: {depth_path}")
            return depth_path

    logger.warning(f"No depth map found for '{image_stem}' in {depth_dir}")
    return None


def load_depth_map(depth_path: Path) -> np.ndarray:
    """Load and normalize depth map.

    Args:
        depth_path: Path to depth map image

    Returns:
        Normalized depth map [0, 1] as float32 (H, W)

    Raises:
        V2EnhancementError: If depth map cannot be loaded
    """
    try:
        depth_image = Image.open(depth_path)

        # Convert to grayscale if needed
        if depth_image.mode != "L" and depth_image.mode != "I" and depth_image.mode != "F":
            depth_image = depth_image.convert("L")

        depth_map = np.array(depth_image, dtype=np.float32)

        # Normalize to [0, 1]
        if depth_map.max() > 1.0:
            depth_map = depth_map / depth_map.max()

        return depth_map

    except Exception as e:
        raise V2EnhancementError(f"Failed to load depth map from {depth_path}: {e}") from e


def enhance_image(
    input_path: Path,
    output_path: Path,
    depth_map_path: Optional[Path] = None,
    material_masks: Optional[Dict[str, np.ndarray]] = None,
    config: Optional[V2EnhancementConfig] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Apply V2 depth-aware enhancement to input image.

    Main entry point for V2 enhancement. Applies perceptual finishing
    using depth-aware tone mapping, clarity enhancement, and material-specific
    processing.

    Args:
        input_path: Path to input image
        output_path: Path to output enhanced image
        depth_map_path: Optional path to depth map from V3 stage
        material_masks: Optional material segmentation masks
        config: Enhancement configuration (uses default if None)
        device: Processing device (cpu/cuda/mps) - currently only cpu supported

    Returns:
        Dict containing processing metadata:
            - status: "success" or "error"
            - input: Input image path
            - output: Output image path
            - depth_map: Depth map path (if provided)
            - preset: Preset name
            - runtime_s: Processing time in seconds
            - metadata: Enhancement metadata from stage
            - timestamp: Processing timestamp

    Raises:
        V2EnhancementError: If enhancement fails
        FileNotFoundError: If input image not found
    """
    start_time = time.perf_counter()

    # Validate input
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    # Use default config if not provided
    if config is None:
        config = V2EnhancementConfig()

    logger.info(f"V2 Enhancement: {input_path.name} with preset '{config.preset}'")

    # Check for "none" preset (skip enhancement)
    if config.preset == "none" or (
        config.enhancement_strength == 0.0 and config.clarity_strength == 0.0 and config.material_strength == 0.0
    ):
        logger.info("Preset 'none' - skipping enhancement (passthrough)")
        # Just copy input to output
        Image.open(input_path).save(output_path)
        return {
            "status": "passthrough",
            "implementation": "v2_enhance",
            "input": str(input_path),
            "output": str(output_path),
            "depth_map": str(depth_map_path) if depth_map_path else None,
            "preset": config.preset,
            "runtime_s": time.perf_counter() - start_time,
            "timestamp": time.time(),
            "message": "Preset 'none' - enhancement skipped",
        }

    try:
        # Load input image
        image = np.array(Image.open(input_path))
        logger.debug(f"Loaded image: {image.shape}, dtype={image.dtype}")

        # Load depth map if provided
        depth_map = None
        if depth_map_path and depth_map_path.exists():
            depth_map = load_depth_map(depth_map_path)
            logger.debug(f"Loaded depth map: {depth_map.shape}")
        elif depth_map_path:
            logger.warning(f"Depth map path provided but not found: {depth_map_path}")

        # Apply enhancement using existing EnhancementStage
        enhancer = EnhancementStage(
            enhancement_strength=config.enhancement_strength,
            clarity_strength=config.clarity_strength,
            material_strength=config.material_strength,
            version=config.version,
        )

        # Create minimal context for stage execution
        context = StageContext(device=device)
        context.set_artifact("image", image)

        if depth_map is not None:
            context.set_artifact("depth_map", depth_map)

        if material_masks:
            context.set_artifact("material_masks", material_masks)

        # Execute enhancement
        logger.debug("Executing EnhancementStage...")
        result = enhancer.compute(context)

        if result.status != StageStatus.COMPLETED:
            error_msg = result.error or "Unknown error"
            raise V2EnhancementError(f"Enhancement failed: {error_msg}")

        # Extract enhanced image
        enhanced_image = result.artifacts.get("enhanced_image")
        if enhanced_image is None:
            raise V2EnhancementError("EnhancementStage did not produce 'enhanced_image' artifact")

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save enhanced image
        Image.fromarray(enhanced_image).save(output_path)
        logger.info(f"Saved enhanced image: {output_path}")

        runtime_s = time.perf_counter() - start_time

        # Build metadata report
        return {
            "status": "success",
            "implementation": "v2_enhance",
            "input": str(input_path),
            "output": str(output_path),
            "depth_map": str(depth_map_path) if depth_map_path else None,
            "preset": config.preset,
            "config": config.to_dict(),
            "runtime_s": runtime_s,
            "timestamp": time.time(),
            "stage_metadata": result.metadata,
            "enhancement_metadata": result.artifacts.get("enhancement_metadata", {}),
        }

    except V2EnhancementError:
        raise
    except Exception as e:
        runtime_s = time.perf_counter() - start_time
        logger.exception("V2 enhancement failed")
        raise V2EnhancementError(f"Enhancement failed after {runtime_s:.2f}s: {e}") from e
