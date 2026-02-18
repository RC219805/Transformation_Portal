"""SAM2 backend for segmentation (Phase 2.1 - Production Implementation).

This module wraps Meta's Segment Anything Model 2 (SAM2) for:
- Automatic mask generation (full image)
- Prompted segmentation (points/bboxes)
- Video temporal tracking (stub for future)

Architecture:
- Direct checkpoint loading (not HuggingFace Hub)
- GPU/CPU/MPS device selection
- Batched inference for efficiency
- Contract-driven input/output

Model Variants:
- sam2_hiera_base_plus: Faster, good quality
- sam2_hiera_large: Slower, best quality

Example:
    >>> backend = SAM2Backend(model_size="large", device="cuda")
    >>> from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput
    >>> seg_input = SegmentationInput(
    ...     image=linear_rgb,  # (H, W, 3) float32
    ...     gamma=1.0,
    ...     mode="auto"
    ... )
    >>> result = backend.segment(seg_input)
    >>> print(f"Found {len(result.masks)} segments")

License: Apache 2.0 (commercial OK, no tier restrictions)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput, SegmentationResult

logger = logging.getLogger(__name__)


class SAM2Backend:
    """SAM2 segmentation backend with direct checkpoint loading.

    Attributes:
        model_size: Model variant ("base" or "large").
        device: Compute device ("cuda", "cpu", "mps").
        checkpoint_path: Path to model checkpoint file.
    """

    # Model configurations (relative to sam2 package)
    MODEL_CONFIGS = {
        "base": "sam2_hiera_b+.yaml",
        "large": "sam2_hiera_l.yaml",
    }

    # Default checkpoint names
    DEFAULT_CHECKPOINTS = {
        "base": "sam2_hiera_base_plus.pt",
        "large": "sam2_hiera_large.pt",
    }

    def __init__(
        self,
        model_size: Literal["base", "large"] = "base",
        device: Literal["cuda", "cpu", "mps"] = "cuda",
        checkpoint_path: Optional[str] = None,
    ):
        """Initialize SAM2 backend.

        Args:
            model_size: Model variant ("base" or "large").
            device: Compute device.
            checkpoint_path: Path to checkpoint file. If None, uses default
                location in checkpoints/ directory.

        Raises:
            ValueError: If model_size invalid or checkpoint not found.
            ImportError: If SAM2 package missing.
        """
        if model_size not in self.MODEL_CONFIGS:
            raise ValueError(f"Invalid model_size '{model_size}', " f"must be one of {list(self.MODEL_CONFIGS.keys())}")

        self.model_size = model_size
        self.device = device

        # Determine checkpoint path
        if checkpoint_path is None:
            checkpoint_path = os.path.join("checkpoints", self.DEFAULT_CHECKPOINTS[model_size])
        self.checkpoint_path = Path(checkpoint_path)

        # Check checkpoint exists (with helpful error message)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {self.checkpoint_path}\n"
                f"Download from: https://github.com/facebookresearch/sam2\n"
                f"Or use: python scripts/download_sam2_checkpoint.py"
            )

        self._model = None
        self._mask_generator = None

        logger.info(
            f"SAM2Backend initialized: model={model_size}, " f"device={device}, checkpoint={self.checkpoint_path.name}"
        )

    def _load_model(self):
        """Lazy load SAM2 model and mask generator.

        Raises:
            ImportError: If sam2 package missing.
            RuntimeError: If model loading fails.
        """
        if self._model is not None:
            return  # Already loaded

        try:
            import torch
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            raise ImportError("SAM2 requires sam2 and torch. " "Install with: pip install sam2 torch torchvision") from e

        # Find config file in sam2 package
        import sam2

        sam2_pkg_path = Path(sam2.__file__).parent
        config_file = sam2_pkg_path / "configs" / self.MODEL_CONFIGS[self.model_size]

        if not config_file.exists():
            raise FileNotFoundError(
                f"SAM2 config not found: {config_file}\n" f"This may indicate an incomplete sam2 installation."
            )

        logger.info(f"Loading SAM2 model: {config_file.name} @ {self.checkpoint_path.name}")

        try:
            # Build SAM2 model
            self._model = build_sam2(
                config_file=str(config_file),
                ckpt_path=str(self.checkpoint_path),
                device=self.device,
                mode="eval",
            )

            # Create automatic mask generator (for auto mode)
            self._mask_generator = SAM2AutomaticMaskGenerator(
                model=self._model,
                points_per_side=32,  # Quality vs speed tradeoff
                points_per_batch=64,
                pred_iou_thresh=0.88,  # High confidence threshold
                stability_score_thresh=0.85,
                box_nms_thresh=0.7,
                crop_n_layers=1,
                crop_nms_thresh=0.7,
            )

            # Create image predictor (for prompted mode)
            self._image_predictor = SAM2ImagePredictor(self._model)

            logger.info("SAM2 model loaded successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to load SAM2 model: {e}") from e

    def segment(
        self,
        seg_input: SegmentationInput,
    ) -> SegmentationResult:
        """Segment image with SAM2.

        Args:
            seg_input: Validated segmentation input contract.

        Returns:
            SegmentationResult with masks, scores, and metadata.

        Raises:
            ValueError: If input contract violated or mode unsupported.
            RuntimeError: If inference fails.
        """
        # Contract validation already done in SegmentationInput.__post_init__

        # Lazy load model
        self._load_model()

        # Execute segmentation based on mode
        if seg_input.mode == "auto":
            return self._segment_auto(seg_input)
        elif seg_input.mode in ["points", "bbox"]:
            return self._segment_prompted(seg_input)
        elif seg_input.mode == "video":
            return self._segment_video(seg_input)
        else:
            raise ValueError(f"Unsupported mode: {seg_input.mode}")

    def _segment_auto(self, seg_input: SegmentationInput) -> SegmentationResult:
        """Automatic mask generation (entire image).

        Uses SAM2's automatic mask generator to detect all objects
        in the image without any prompts.

        Args:
            seg_input: Validated segmentation input.

        Returns:
            SegmentationResult with all detected masks.

        Raises:
            RuntimeError: If mask generation fails.
        """
        image = seg_input.image

        # Convert to uint8 RGB for SAM2 (expects 0-255 range)
        if image.dtype == np.float32 or image.dtype == np.float64:
            # Assume [0, 1] range, scale to [0, 255]
            image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)

        try:
            # Generate masks
            masks_data = self._mask_generator.generate(image_uint8)

            # Extract masks and scores
            if not masks_data:
                # No masks found - return empty result
                logger.warning("SAM2 found no masks in auto mode")
                return SegmentationResult(
                    masks=np.zeros((0, *image.shape[:2]), dtype=bool),
                    scores=np.zeros(0, dtype=np.float32),
                    metadata={
                        "backend": "sam2",
                        "model_size": self.model_size,
                        "mode": "auto",
                        "num_masks": 0,
                    },
                )

            # Convert SAM2 output to our format
            masks = np.stack([m["segmentation"] for m in masks_data])
            iou_scores = np.array([m["predicted_iou"] for m in masks_data], dtype=np.float32)
            stability_scores = np.array([m["stability_score"] for m in masks_data], dtype=np.float32)

            # Use average of IoU and stability as final score
            scores = (iou_scores + stability_scores) / 2.0

            logger.info(f"SAM2 auto mode: generated {len(masks)} masks")

            return SegmentationResult(
                masks=masks,
                scores=scores,
                metadata={
                    "backend": "sam2",
                    "model_size": self.model_size,
                    "mode": "auto",
                    "num_masks": len(masks),
                    "iou_scores": iou_scores.tolist(),
                    "stability_scores": stability_scores.tolist(),
                },
            )

        except Exception as e:
            raise RuntimeError(f"SAM2 auto mode segmentation failed: {e}") from e

    def _segment_prompted(self, seg_input: SegmentationInput) -> SegmentationResult:
        """Prompted segmentation (points or bounding boxes).

        Args:
            seg_input: Validated segmentation input with prompts.

        Returns:
            SegmentationResult with prompted masks.

        Raises:
            ValueError: If prompts are invalid.
            RuntimeError: If segmentation fails.
        """
        image = seg_input.image
        mode = seg_input.mode

        # Convert to uint8 RGB for SAM2
        if image.dtype == np.float32 or image.dtype == np.float64:
            image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)

        try:
            # Set image in predictor
            self._image_predictor.set_image(image_uint8)

            if mode == "points":
                if seg_input.points is None:
                    raise ValueError("Points mode requires points to be provided")

                points = np.array(seg_input.points)
                labels = (
                    np.array(seg_input.point_labels)
                    if seg_input.point_labels is not None
                    else np.ones(len(points), dtype=np.int32)
                )

                # Predict masks
                masks, scores, logits = self._image_predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=True,  # Get multiple mask proposals
                )

            elif mode == "bbox":
                if seg_input.bbox is None:
                    raise ValueError("Bbox mode requires bbox to be provided")

                bbox = np.array(seg_input.bbox)  # [x1, y1, x2, y2]

                # Predict masks
                masks, scores, logits = self._image_predictor.predict(
                    box=bbox,
                    multimask_output=True,
                )

            else:
                raise ValueError(f"Unsupported prompted mode: {mode}")

            logger.info(f"SAM2 {mode} mode: generated {len(masks)} masks")

            return SegmentationResult(
                masks=masks,
                scores=scores.astype(np.float32),
                metadata={
                    "backend": "sam2",
                    "model_size": self.model_size,
                    "mode": mode,
                    "num_masks": len(masks),
                },
            )

        except Exception as e:
            raise RuntimeError(f"SAM2 {mode} mode segmentation failed: {e}") from e

    def _segment_video(self, seg_input: SegmentationInput) -> SegmentationResult:
        """Video segmentation with temporal tracking.

        Note: Video tracking requires sequential frames and state management.
        This is a stub for future implementation.

        Args:
            seg_input: Validated segmentation input.

        Returns:
            SegmentationResult (not yet implemented).

        Raises:
            NotImplementedError: Video tracking not yet implemented.
        """
        raise NotImplementedError(
            "SAM2 video tracking not yet implemented. "
            "Use 'auto' or 'points'/'bbox' modes for single images. "
            "For video, process frames individually or use sam2.video_predictor "
            "API directly (requires state management across frames)."
        )


# Utility function for download script
def download_sam2_checkpoint(model_size: Literal["base", "large"] = "large", output_dir: str = "checkpoints") -> Path:
    """Download SAM2 checkpoint from official repository.

    Args:
        model_size: Model variant to download.
        output_dir: Directory to save checkpoint.

    Returns:
        Path to downloaded checkpoint.

    Raises:
        RuntimeError: If download fails.
    """
    import urllib.request
    from pathlib import Path

    CHECKPOINT_URLS = {
        "base": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
        "large": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
    }

    url = CHECKPOINT_URLS[model_size]
    filename = SAM2Backend.DEFAULT_CHECKPOINTS[model_size]
    output_path = Path(output_dir) / filename

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info(f"Checkpoint already exists: {output_path}")
        return output_path

    logger.info(f"Downloading SAM2 {model_size} checkpoint from {url}...")
    logger.info(f"This may take several minutes (checkpoint is ~200-400 MB)...")

    try:
        urllib.request.urlretrieve(url, output_path)
        logger.info(f"✅ Downloaded: {output_path}")
        return output_path
    except Exception as e:
        raise RuntimeError(f"Failed to download SAM2 checkpoint: {e}") from e
