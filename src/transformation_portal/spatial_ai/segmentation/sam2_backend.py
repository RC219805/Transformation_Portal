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

from transformation_portal.spatial_ai.segmentation.contracts import MaskMetadata, SegmentationInput, SegmentationResult

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
        image: np.ndarray,
        gamma: float,
        mode: Literal["auto", "points", "bbox", "video"] = "auto",
        prompts: Optional[list] = None,
        prev_masks: Optional[np.ndarray] = None,
        frame_idx: Optional[int] = None,
    ) -> SegmentationResult:
        """Segment image with SAM2.

        Args:
            image: Linear RGB image (H, W, 3) float32.
            gamma: Gamma value (must be 1.0).
            mode: Segmentation mode.
            prompts: Optional prompts for interactive modes.
            prev_masks: Previous masks for video tracking.
            frame_idx: Frame index for video mode.

        Returns:
            SegmentationResult with masks, scores, and metadata.

        Raises:
            ValueError: If input contract violated.
            RuntimeError: If inference fails.
        """
        # Validate input via contract
        seg_input = SegmentationInput(
            image=image,
            gamma=gamma,
            mode=mode,
            prompts=prompts,
            prev_masks=prev_masks,
            frame_idx=frame_idx,
        )

        # Lazy load model
        self._load_model()

        # Execute segmentation based on mode
        if mode == "auto":
            return self._segment_auto(seg_input)
        elif mode in ["points", "bbox"]:
            return self._segment_prompted(seg_input)
        elif mode == "video":
            return self._segment_video(seg_input)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def _segment_auto(self, seg_input: SegmentationInput) -> SegmentationResult:
        """Automatic mask generation (entire image).

        Args:
            seg_input: Validated segmentation input.

        Returns:
            SegmentationResult with all detected masks.
        """
        import torch

        # Convert linear RGB to uint8 sRGB for SAM2 preprocessing
        # SAM2 expects display-ready images, so apply gamma correction
        image_srgb = self._linear_to_srgb(seg_input.image)

        # Preprocess image
        inputs = self._processor(images=image_srgb, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Extract masks and scores
        # Note: SAM2 API may vary, this is a placeholder structure
        # Real implementation should use SAM2's automatic mask generator
        masks = outputs.pred_masks.cpu().numpy()  # (N, H, W)
        scores = outputs.iou_scores.cpu().numpy()  # (N,)

        # Convert to boolean masks
        masks_bool = masks > 0.5

        # Generate metadata for each mask
        metadata = []
        for i, mask in enumerate(masks_bool):
            area = int(mask.sum())
            if area == 0:
                continue  # Skip empty masks

            # Compute bounding box
            ys, xs = np.where(mask)
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            bbox = (int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1))

            metadata.append(
                MaskMetadata(
                    area=area,
                    bbox=bbox,
                    stability_score=float(scores[i]),
                )
            )

        # Filter out empty masks
        valid_indices = [i for i, m in enumerate(masks_bool) if m.sum() > 0]
        masks_bool = masks_bool[valid_indices]
        scores = scores[valid_indices]

        return SegmentationResult(
            masks=masks_bool,
            scores=scores.astype(np.float32),
            metadata=metadata,
        )

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
