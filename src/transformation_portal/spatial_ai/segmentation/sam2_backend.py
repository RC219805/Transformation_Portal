"""SAM2 backend for segmentation (Phase 2.1).

This module wraps Meta's Segment Anything Model 2 (SAM2) for:
- Automatic mask generation (full image)
- Prompted segmentation (points/bboxes)
- Video temporal tracking

Architecture (ADR-027):
- HuggingFace model loading with revision pinning
- GPU/CPU device selection
- Batched inference for efficiency
- Contract-driven input/output

Model Variants:
- sam2-hiera-base-plus: Faster, good quality
- sam2-hiera-large: Slower, best quality (research tier)

Example:
    >>> backend = SAM2Backend(model_size="large", device="cuda")
    >>> result = backend.segment(
    ...     image=linear_rgb,  # (H, W, 3) float32
    ...     gamma=1.0,
    ...     mode="auto"
    ... )
    >>> print(f"Found {len(result.masks)} segments")

License: Apache 2.0 (commercial OK, no tier restrictions)
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np

from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput, SegmentationResult

logger = logging.getLogger(__name__)


class SAM2Backend:
    """SAM2 segmentation backend with HuggingFace integration.

    Attributes:
        model_size: Model variant ("base" or "large").
        device: Compute device ("cuda", "cpu", "mps").
        revision: HuggingFace model revision (commit SHA or placeholder).
    """

    SUPPORTED_MODELS = {
        "base": "facebook/sam2-hiera-base-plus",
        "large": "facebook/sam2-hiera-large",
    }

    def __init__(
        self,
        model_size: Literal["base", "large"] = "base",
        device: Literal["cuda", "cpu", "mps"] = "cuda",
        revision: Optional[str] = None,
    ):
        """Initialize SAM2 backend.

        Args:
            model_size: Model variant ("base" or "large").
            device: Compute device.
            revision: HuggingFace commit SHA (None = use latest).
                For experimental presets: "NEEDS_VERIFICATION_SAM2_..."
                For stable presets: Must be 40-char commit SHA.

        Raises:
            ValueError: If model_size invalid.
            ImportError: If SAM2 dependencies missing.
        """
        if model_size not in self.SUPPORTED_MODELS:
            raise ValueError(f"Invalid model_size '{model_size}', " f"must be one of {list(self.SUPPORTED_MODELS.keys())}")

        self.model_size = model_size
        self.device = device
        self.revision = revision or "NEEDS_VERIFICATION_SAM2_BASE_20260211"

        self._model = None
        self._processor = None
        self._mask_pipeline = None  # Lazy-loaded mask-generation pipeline

        logger.info(f"SAM2Backend initialized: model={model_size}, " f"device={device}, revision={self.revision[:12]}...")

    def _load_model(self):
        """Lazy load SAM2 model and processor.

        Raises:
            ImportError: If transformers/torch missing.
            RuntimeError: If model download fails.
        """
        if self._model is not None:
            return  # Already loaded

        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            raise ImportError("SAM2 requires transformers and torch. " "Install with: pip install transformers torch") from e

        model_id = self.SUPPORTED_MODELS[self.model_size]

        logger.info(f"Loading SAM2 model: {model_id} @ {self.revision[:12]}...")

        try:
            # Load processor (handles image preprocessing)
            self._processor = AutoProcessor.from_pretrained(
                model_id,
                revision=self.revision if not self.revision.startswith("NEEDS_VERIFICATION") else None,
            )

            # Load model
            self._model = AutoModel.from_pretrained(
                model_id,
                revision=self.revision if not self.revision.startswith("NEEDS_VERIFICATION") else None,
            )

            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.to("cuda")
            elif self.device == "mps" and torch.backends.mps.is_available():
                self._model = self._model.to("mps")
            else:
                self._model = self._model.to("cpu")
                if self.device != "cpu":
                    logger.warning(f"Device '{self.device}' unavailable, using CPU")

            self._model.eval()  # Inference mode
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
            ValueError: If input contract violated.
            RuntimeError: If inference fails.
        """
        # Contract is already validated in SegmentationInput.__post_init__

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

        Uses HuggingFace's mask-generation pipeline for SAM2.

        Args:
            seg_input: Validated segmentation input.

        Returns:
            SegmentationResult with all detected masks.

        Raises:
            RuntimeError: If mask generation fails.
        """
        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                "SAM2 auto mode requires transformers pipeline. " "Install with: pip install transformers"
            ) from e

        # Convert linear RGB to sRGB uint8 for SAM2
        srgb_uint8 = self._linear_to_srgb(seg_input.image)

        # Lazy load mask generation pipeline
        if not hasattr(self, "_mask_pipeline") or self._mask_pipeline is None:
            model_id = self.SUPPORTED_MODELS[self.model_size]
            logger.info(f"Loading SAM2 mask-generation pipeline: {model_id}")

            # Determine device for pipeline
            device_id = -1  # CPU default
            if self.device == "cuda":
                import torch

                if torch.cuda.is_available():
                    device_id = 0
            elif self.device == "mps":
                # MPS not directly supported by pipeline, falls back to CPU
                logger.warning("MPS device not supported by mask-generation pipeline, using CPU")
                device_id = -1

            try:
                self._mask_pipeline = pipeline(
                    "mask-generation",
                    model=model_id,
                    revision=self.revision if not self.revision.startswith("NEEDS_VERIFICATION") else None,
                    device=device_id,
                )
                logger.info("SAM2 mask-generation pipeline loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to load SAM2 mask-generation pipeline: {e}") from e

        # Run automatic mask generation
        # Pipeline expects PIL Image, path, or URL
        try:
            from PIL import Image

            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(srgb_uint8)

            logger.debug(f"Running SAM2 automatic mask generation on {srgb_uint8.shape[:2]} image...")

            # Generate masks with configurable parameters
            outputs = self._mask_pipeline(
                pil_image,  # PIL Image instead of numpy array
                points_per_batch=64,  # Balance speed/memory
                pred_iou_thresh=0.7,  # Filter low-quality masks
            )

            # Extract masks and scores
            masks_list = outputs.get("masks", [])
            scores_list = outputs.get("scores", [])

            if not masks_list:
                logger.warning("SAM2 generated no masks, returning empty result")
                return SegmentationResult(
                    masks=np.zeros((0, *seg_input.image.shape[:2]), dtype=bool),
                    scores=np.array([], dtype=np.float32),
                    metadata=[],
                )

            # Convert to numpy arrays
            # Pipeline returns list of (H, W) masks
            masks = np.stack([np.array(m, dtype=bool) for m in masks_list], axis=0)

            # Scores may be missing - use stability heuristic if needed
            if scores_list is not None and len(scores_list) > 0:
                scores = np.array(scores_list, dtype=np.float32)
            else:
                # Fallback: estimate stability from mask properties
                scores = np.array([self._estimate_mask_stability(m) for m in masks], dtype=np.float32)

            # Generate metadata for each mask
            metadata = []
            for i, mask in enumerate(masks):
                area = int(mask.sum())
                if area == 0:
                    continue  # Skip empty masks

                # Compute bounding box
                rows, cols = np.where(mask)
                if len(rows) == 0:
                    continue

                x1, x2 = int(cols.min()), int(cols.max())
                y1, y2 = int(rows.min()), int(rows.max())
                bbox = (x1, y1, x2 - x1 + 1, y2 - y1 + 1)

                # Use score as stability score
                stability = float(scores[i]) if i < len(scores) else 0.5

                from transformation_portal.spatial_ai.segmentation.contracts import MaskMetadata

                metadata.append(
                    MaskMetadata(
                        area=area,
                        bbox=bbox,
                        stability_score=stability,
                    )
                )

            logger.info(f"SAM2 generated {len(masks)} masks with avg score {scores.mean():.3f}")

            return SegmentationResult(masks=masks, scores=scores, metadata=metadata)

        except Exception as e:
            logger.error(f"SAM2 mask generation failed: {e}", exc_info=True)
            raise RuntimeError(f"SAM2 automatic mask generation failed: {e}") from e

    def _estimate_mask_stability(self, mask: np.ndarray) -> float:
        """Estimate mask stability from geometric properties.

        Args:
            mask: Binary mask (H, W).

        Returns:
            Stability score [0, 1] (higher = more stable).
        """
        area = mask.sum()
        if area == 0:
            return 0.0

        # Compute perimeter
        import scipy.ndimage as ndimage

        eroded = ndimage.binary_erosion(mask)
        perimeter = (mask & ~eroded).sum()

        if perimeter == 0:
            return 0.5

        # Compactness metric (circle = 1.0, irregular = lower)
        # Normalized isoperimetric ratio
        compactness = 4 * np.pi * area / (perimeter**2)
        compactness = min(compactness, 1.0)

        # Use compactness as stability proxy
        return float(compactness)

    def _segment_prompted(self, seg_input: SegmentationInput) -> SegmentationResult:
        """Prompted segmentation (points/bboxes).

        Args:
            seg_input: Validated segmentation input with prompts.

        Returns:
            SegmentationResult with prompted masks.
        """
        # TODO: Implement prompted segmentation
        # This requires parsing prompts and passing to SAM2
        raise NotImplementedError("Prompted segmentation not yet implemented")

    def _segment_video(self, seg_input: SegmentationInput) -> SegmentationResult:
        """Video temporal tracking.

        Args:
            seg_input: Validated segmentation input with prev_masks.

        Returns:
            SegmentationResult with temporally-tracked masks.
        """
        # TODO: Implement video tracking
        # This requires temporal propagation of masks
        raise NotImplementedError("Video tracking not yet implemented")

    def _linear_to_srgb(self, linear_rgb: np.ndarray) -> np.ndarray:
        """Convert linear RGB to sRGB uint8 for SAM2 preprocessing.

        Args:
            linear_rgb: (H, W, 3) float32 linear RGB [0, ∞).

        Returns:
            (H, W, 3) uint8 sRGB [0, 255].
        """
        # Clip HDR values (SAM2 expects [0, 1])
        linear_clipped = np.clip(linear_rgb, 0, 1)

        # Apply sRGB gamma (approximation: gamma 2.2)
        srgb = np.power(linear_clipped, 1.0 / 2.2)

        # Convert to uint8
        srgb_uint8 = (srgb * 255).astype(np.uint8)

        return srgb_uint8
